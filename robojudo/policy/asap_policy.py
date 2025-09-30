import logging
import os
import sys
import time

import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import AsapLocoPolicyCfg, AsapPolicyCfg
from robojudo.utils.util_func import command_remap, quat_rotate_inverse_np

logger = logging.getLogger(__name__)


def vis_process(name, alpha):
    bar_length = 30  # Adjust bar length as needed
    filled_length = int(bar_length * alpha)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\r{name} Progress: |{bar}| {alpha:.2%}")
    sys.stdout.flush()


@policy_registry.register
class AsapPolicy(Policy):
    """From [ASAP](https://github.com/LeCAR-Lab/ASAP), sim2real/rl_policy/deepmimic_dec_loco.py"""

    cfg_policy: AsapPolicyCfg

    def __init__(self, cfg_policy: AsapPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading mimic policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(cfg_policy.policy_file)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        # TODO: move to onnx policy

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = cfg_policy.obs_scales
        # self.start_upper_dof_pos = cfg_policy.start_upper_body_dof_pos  # for interpolation loco to mimic
        # self.end_upper_dof_pos = np.zeros((1, 17))
        self.motion_length_s = cfg_policy.motion_length_s
        self.use_history = cfg_policy.USE_HISTORY

        if self.use_history:
            self.history_obs_dims = cfg_policy.history_obs_dims
            default_history = [np.zeros(dim, dtype=np.float32) for dim in self.history_obs_dims.values()]
            self._init_history(default_history)

        self.reset()

    def reset(self):
        self.timestep: float = 0
        self.play_speed: float = 1.0

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1 * self.play_speed

        # TODO: If current mimic policy is done, switch to locomotion policy
        if self._get_frame_encoding() >= 1.0:
            self.reset()

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0
        pass

    def _get_frame_encoding(self):
        # 11 bins for 11 seconds, if (current_time-self.frame_start_time) > 1, increment frame_idx
        # the frame encoding is maped to 0-1
        phase = (self.timestep * self.dt) / self.motion_length_s
        # print("phase", phase)
        vis_process("Mimic", phase)
        return phase

    def _get_obs_history(self):
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)

    def get_observation(self, env_data, ctrl_data):
        base_quat = env_data.base_quat  # [x, y, z, w]
        base_ang_vel = env_data.base_ang_vel
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel

        dof_pos_minus_default = dof_pos - self.default_dof_pos

        v = np.array([0, 0, -1])
        projected_gravity = quat_rotate_inverse_np(base_quat, v)

        # prepare frame encoding for deepmimic
        phase_np = np.array([self._get_frame_encoding()])

        if self.use_history:
            history = self._get_obs_history()
            history *= self.obs_scales.history
        else:
            history = []
        obs = np.concatenate(
            [
                self.last_action * self.obs_scales.actions,
                base_ang_vel * self.obs_scales.base_ang_vel,
                dof_pos_minus_default * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                history,
                projected_gravity * self.obs_scales.projected_gravity,
                phase_np * self.obs_scales.ref_motion_phase,
            ],
            axis=0,
        )

        # Note: this should be aligned with history_obs_dims
        # IMPORTANT: the order should be SORTED by obs name!!!
        obs_a = [
            self.last_action * self.obs_scales.actions,
            base_ang_vel * self.obs_scales.base_ang_vel,
            dof_pos_minus_default * self.obs_scales.dof_pos,
            dof_vel * self.obs_scales.dof_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            phase_np * self.obs_scales.ref_motion_phase,
        ]
        self.history_buf.appendleft(obs_a)

        extras = {}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            "actor_obs": np.expand_dims(obs, axis=0).astype(np.float32),
        }

        ort_outputs = self.session.run(
            ["action"],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)

        self.last_action = actions.copy()  # TODO: check all policies after process
        processed_actions = processed_actions * self.action_scale
        return processed_actions

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        pass


@policy_registry.register
class AsapLocoPolicy(Policy):
    cfg_policy: AsapLocoPolicyCfg

    def __init__(self, cfg_policy: AsapLocoPolicyCfg, device):
        if not os.path.isfile(cfg_policy.policy_file):
            raise FileNotFoundError(f"Model file not found at {cfg_policy.policy_file}")

        logger.debug(f"Loading mimic policy '{cfg_policy.policy_name}' from {cfg_policy.policy_file}")
        self.session = ort.InferenceSession(cfg_policy.policy_file)

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        # TODO: move to onnx policy

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.obs_scales = cfg_policy.obs_scales

        self.use_history = cfg_policy.USE_HISTORY

        if self.use_history:
            self.history_obs_dims = cfg_policy.history_obs_dims
            default_history = [np.zeros(dim) for dim in self.history_obs_dims.values()]
            self._init_history(default_history)

        self.num_upper_dofs = cfg_policy.NUM_UPPER_BODY_JOINTS
        self.num_lower_dofs = self.num_dofs - self.num_upper_dofs
        self.ref_upper_dof_pos = np.zeros(self.num_upper_dofs)
        self.ref_upper_dof_pos[4] = 0.3
        self.ref_upper_dof_pos[11] = -0.3
        self.ref_upper_dof_pos[6] = 1.0
        self.ref_upper_dof_pos[13] = 1.0

        self.gait_period = cfg_policy.GAIT_PERIOD
        self.lin_vel_command = np.array([0.0, 0.0])
        self.ang_vel_command = np.array([0.0])
        self.stand_command = np.array([0])
        self.base_height_command = np.array([0.78])

        self.reset()

    def reset(self):
        self.timestep: float = 0

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1

    def _get_obs_phase_time(self):
        cur_time = time.time() * self.stand_command[0]
        phase_time = cur_time % self.gait_period / self.gait_period
        vis_process("Loco", phase_time)
        return np.array([phase_time])

    def _get_obs_history(self):
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        return np.concatenate(history_list, axis=0)

    def get_observation(self, env_data, ctrl_data):
        self._update_commands(ctrl_data)

        base_quat = env_data.base_quat  # [x, y, z, w]
        base_ang_vel = env_data.base_ang_vel
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel

        dof_pos_minus_default = dof_pos - self.default_dof_pos

        v = np.array([0, 0, -1])
        projected_gravity = quat_rotate_inverse_np(base_quat, v)

        phase_time = self._get_obs_phase_time()
        sin_phase = np.sin(2 * np.pi * phase_time)
        cos_phase = np.cos(2 * np.pi * phase_time)

        if self.use_history:
            history = self._get_obs_history()
            history *= self.obs_scales.history
        else:
            history = []
        obs = np.concatenate(
            [
                self.last_action[: self.num_lower_dofs] * self.obs_scales.actions,
                base_ang_vel * self.obs_scales.base_ang_vel,
                self.ang_vel_command * self.obs_scales.command_ang_vel,
                self.base_height_command * self.obs_scales.command_base_height,
                self.lin_vel_command * self.obs_scales.command_lin_vel,
                self.stand_command * self.obs_scales.command_stand,
                cos_phase * self.obs_scales.cos_phase,
                dof_pos_minus_default * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                history,
                # phase_time,
                projected_gravity * self.obs_scales.projected_gravity,
                self.ref_upper_dof_pos * self.obs_scales.ref_upper_dof_pos,
                sin_phase * self.obs_scales.sin_phase,
            ],
            axis=0,
        )

        # Note: this should be aligned with history_obs_dims
        # IMPORTANT: the order should be SORTED by obs name!!!
        obs_a = [
            self.last_action[: self.num_lower_dofs] * self.obs_scales.actions,
            base_ang_vel * self.obs_scales.base_ang_vel,
            self.ang_vel_command * self.obs_scales.command_ang_vel,
            self.base_height_command * self.obs_scales.command_base_height,
            self.lin_vel_command * self.obs_scales.command_lin_vel,
            self.stand_command * self.obs_scales.command_stand,
            cos_phase * self.obs_scales.cos_phase,
            dof_pos_minus_default * self.obs_scales.dof_pos,
            dof_vel * self.obs_scales.dof_vel,
            # phase_time * self.obs_scales.phase_time,
            projected_gravity * self.obs_scales.projected_gravity,
            self.ref_upper_dof_pos * self.obs_scales.ref_upper_dof_pos,
            sin_phase * self.obs_scales.sin_phase,
        ]
        self.history_buf.appendleft(obs_a)

        obs = obs.astype(np.float32)
        extras = {
            "commands": [self.lin_vel_command[0], self.lin_vel_command[1], self.ang_vel_command[0]],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            "actor_obs": np.expand_dims(obs, axis=0).astype(np.float32),
        }

        ort_outputs = self.session.run(
            ["action"],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()
        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)

        self.last_action[: self.num_lower_dofs] = processed_actions.copy()
        self.last_action[self.num_lower_dofs :] = self.ref_upper_dof_pos.copy()

        processed_actions = processed_actions * self.action_scale

        full_actions = np.concatenate([processed_actions, self.ref_upper_dof_pos], axis=0)
        return full_actions

    def _update_commands(self, ctrl_data):
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrk"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx, _ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

                self.lin_vel_command[1] = command_remap(lx, [0.5, 0, -0.5]) * self.stand_command[0]
                self.lin_vel_command[0] = command_remap(ly, [-0.5, 0, 0.5]) * self.stand_command[0]
                self.ang_vel_command[0] = command_remap(rx, [1, 0, -1]) * self.stand_command[0]

                button_event = ctrl_data[key]["button_event"]
                for event in button_event:
                    if event["type"] == "button" and event["pressed"]:
                        match event["name"]:
                            case "Left":
                                self.stand_command = 1 - self.stand_command
                                if self.stand_command == 0:
                                    self.ang_vel_command[0] = 0.0
                                    self.lin_vel_command[0] = 0.0
                                    self.lin_vel_command[1] = 0.0
                            case "Up":
                                self.base_height_command[0] += 0.05
                            case "Down":
                                self.base_height_command[0] -= 0.05
                break
            elif key == "KeyboardCtrl":
                for event in ctrl_data[key]["keyboard_event"]:
                    if event["type"] == "keyboard" and event["pressed"]:
                        match event["name"]:
                            case "w":
                                self.lin_vel_command[0] += 0.1 if self.stand_command else 0.0
                            case "s":
                                self.lin_vel_command[0] -= 0.1 if self.stand_command else 0.0
                            case "a":
                                self.lin_vel_command[1] += 0.1 if self.stand_command else 0.0
                            case "d":
                                self.lin_vel_command[1] -= 0.1 if self.stand_command else 0.0
                            case "q":
                                self.ang_vel_command[0] -= 0.1
                            case "e":
                                self.ang_vel_command[0] += 0.1
                            case "z":
                                self.ang_vel_command[0] = 0.0
                                self.lin_vel_command[0] = 0.0
                                self.lin_vel_command[1] = 0.0
                            case "1":
                                self.base_height_command += 0.05
                            case "2":
                                self.base_height_command -= 0.05
                            case "=":
                                self.stand_command = 1 - self.stand_command
                                if self.stand_command == 0:
                                    self.ang_vel_command[0] = 0.0
                                    self.lin_vel_command[0] = 0.0
                                    self.lin_vel_command[1] = 0.0

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        command_x = extras["commands"][0]
        command_y = extras["commands"][1]
        command_yaw = extras["commands"][2]

        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )
