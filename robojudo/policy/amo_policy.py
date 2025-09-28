from collections import deque

import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import AMOPolicyCfg
from robojudo.utils.util_func import command_remap


@policy_registry.register
class AMOPolicy(Policy):
    """
    Only Unitree G1 supported now.
    """

    cfg_policy: AMOPolicyCfg

    def __init__(self, cfg_policy, device):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(cfg_policy=cfg_policy, device=device)

        # if cpu device:
        if self.device == "cpu":
            for node in self.model.graph.nodes():  # type: ignore
                if node.kind() == "prim::Constant":
                    node: torch.Node
                    if "value" in node.attributeNames():
                        value = node.output().toIValue()
                        if isinstance(value, torch.device):
                            node.removeAttribute("value")
                            node.s_("value", "cpu")

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = np.zeros(8)  # self.default_pos[15:]
        self.prev_arm_action = self.default_pos[15:]
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = self.cfg_policy.obs_scales.ang_vel
        self.scales_dof_vel = self.cfg_policy.obs_scales.dof_vel
        self.control_dt = 1.0 / self.cfg_policy.freq
        self.commands_map = self.cfg_policy.commands_map

        self._init_history(np.zeros(self.history_obs_size))

        # TODO: move robot specific to cfg
        self.nj = 23
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[: self._n_demo_dof] = self.default_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75

        self.target_yaw = 0.0
        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))

        self.cmd = np.zeros(8, dtype=np.float32)
        self.cmd[0:4] = [map[1] for map in self.commands_map]

        self.adapter = torch.jit.load(self.cfg_policy.policy_adapter_file, map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False

        norm_stats = torch.load(self.cfg_policy.policy_adapter_norm_file, weights_only=False)
        self.input_mean = torch.tensor(norm_stats["input_mean"], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats["input_std"], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats["output_mean"], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats["output_std"], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        self.reset()

    def reset(self):
        self.timestep: int = 0

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_commands(self, ctrl_data):
        commands = self.cmd.copy()

        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(rx, self.commands_map[1])
                commands[2] = command_remap(lx, self.commands_map[2])
                commands[3] = command_remap(ry, self.commands_map[3])

                button_event = ctrl_data[key]["button_event"]
                for event in button_event:
                    if event["type"] == "button" and event["name"] == "Y" and event["pressed"]:
                        commands[7] = not commands[7]

                break
        return commands

    def get_observation(self, env_data, ctrl_data):
        self.cmd = self._get_commands(ctrl_data)
        rpy = self.quatToEuler(env_data.base_quat[[3, 0, 1, 2]])

        self.target_yaw = self.cmd[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = env_data.dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), env_data.dof_pos[15:]])

        self.adapter_input[0] = self.cmd[3]  # + 0.75
        self.adapter_input[1] = self.cmd[4]
        self.adapter_input[2] = self.cmd[5]
        self.adapter_input[3] = self.cmd[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)

        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean
        obs_prop = np.concatenate(
            [
                env_data.base_ang_vel * self.scales_ang_vel,
                rpy[:2],
                (np.sin(dyaw), np.cos(dyaw)),
                (env_data.dof_pos - self.default_pos),
                env_data.dof_vel * self.scales_dof_vel,
                self.last_action,
                gait_obs,
                self.adapter_output.cpu().numpy().squeeze(),
            ]
        )
        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[: self._n_demo_dof] = env_data.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.cmd[0]
        obs_demo[self._n_demo_dof + 1] = self.cmd[2]
        self._in_place_stand_flag = np.abs(self.cmd[0]) < 0.1
        obs_demo[self._n_demo_dof + 3] = self.cmd[4]
        obs_demo[self._n_demo_dof + 4] = self.cmd[5]
        obs_demo[self._n_demo_dof + 5] = self.cmd[6]
        obs_demo[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75 + self.cmd[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)

        self.dof_pos = env_data.dof_pos.copy()
        extras = {
            "extra_hist": torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float)
            .view(1, -1)
            .to(self.device)
            .numpy(),
            "commands": self.cmd.copy(),
        }
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist)), extras

    def get_action(self, obs):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            extra_hist = (
                torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float)
                .view(1, -1)
                .to(self.device)
            )
            raw_action = self.model(obs_tensor, extra_hist).cpu().numpy().squeeze()

        raw_action = np.clip(raw_action, -40.0, 40.0)
        self.last_action = np.concatenate(
            [raw_action.copy(), (self.dof_pos - self.default_pos)[15:] / self.action_scale]
        )
        scaled_actions = raw_action * self.action_scale

        if self.timestep % 300 == 0 and self.cmd[7]:
            # self.arm_blend = 0
            self.prev_arm_action = self.dof_pos[15:].copy()
            # self.arm_action = self.cmd_ctrl.get_motion()
            self.arm_action = np.random.uniform(0, 1, 8) * (np.full((8), 0.8)) + np.full((8), -0.4)
            self.toggle_arm = True
        elif not self.cmd[7]:
            if self.toggle_arm:
                self.toggle_arm = False
                # self.arm_blend = 0
                self.prev_arm_action = self.dof_pos[15:].copy()
                self.arm_action = np.zeros(8)
        pd_target = np.concatenate([scaled_actions, np.zeros(8)])  # + self.default_pos
        pd_target[15:] = (
            self.arm_action
        )  # (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
        # print(pd_target[15:])
        self.arm_blend = min(1.0, self.arm_blend + 0.01)

        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if self._in_place_stand_flag and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.75])
        return pd_target

    @staticmethod
    def quatToEuler(quat):  # TODO
        eulerVec = np.zeros(3)
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            eulerVec[1] = np.arcsin(sinp)

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

        return eulerVec
