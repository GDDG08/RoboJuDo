import sys

import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import GoalkeeperPolicyCfg
from robojudo.utils.util_func import quat_rotate_inverse_np


@policy_registry.register
class GoalkeeperPolicy(Policy):
    cfg_policy: GoalkeeperPolicyCfg

    def __init__(self, cfg_policy: GoalkeeperPolicyCfg, device="cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.history_len = self.cfg_policy.history_length
        self._init_history(np.zeros(self.cfg_policy.history_obs_size, dtype=np.float32))
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        self.camera_R_tc = np.asarray(self.cfg_policy.camera_R_tc, dtype=np.float32)
        self.camera_t_tc = np.asarray(self.cfg_policy.camera_t_tc, dtype=np.float32)

        if self.cfg_policy.load_checkpoint:
            self._load_checkpoint()
        else:
            self.actor_critic = None

    def _load_checkpoint(self):
        repo_path = self.cfg_policy.goalkeeper_repo_path
        if repo_path and repo_path not in sys.path:
            sys.path.append(repo_path)
        from rsl_rl.modules.actor_critic import ActorCritic  # type: ignore

        self.actor_critic = ActorCritic(
            num_actor_obs=self.cfg_policy.num_actor_obs,
            num_critic_obs=self.cfg_policy.num_critic_obs,
            num_one_step_obs=self.cfg_policy.num_one_step_obs,
            actor_history_length=self.cfg_policy.actor_history_length,
            num_actions=self.num_actions,
            actor_hidden_dims=self.cfg_policy.actor_hidden_dims,
            critic_hidden_dims=self.cfg_policy.critic_hidden_dims,
            activation=self.cfg_policy.activation,
        ).to(self.device)
        ckpt = torch.load(self.cfg_policy.checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.actor_critic.load_state_dict(state)
        self.actor_critic.eval()

    def reset(self):
        self.last_action[:] = 0.0
        self._init_history(np.zeros(self.cfg_policy.history_obs_size, dtype=np.float32))

    def post_step_callback(self, commands=None):
        return

    def _projected_gravity(self, quat):
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quat_rotate_inverse_np(quat, gravity)

    def _ball_pos_pelvis(self, env_data, ctrl_data):
        data = ctrl_data.get("GoalkeeperBallDdsCtrl", {})
        if not data.get("ball_valid", False):
            return np.zeros(3, dtype=np.float32)
        ball_cam = np.asarray(data["ball_xyz_cam"], dtype=np.float32)
        ball_torso = self.camera_R_tc @ ball_cam + self.camera_t_tc

        torso_pos = env_data.torso_pos
        torso_quat = env_data.torso_quat
        ball_world = torso_pos + quat_rotate_inverse_np(torso_quat, -ball_torso)
        return quat_rotate_inverse_np(env_data.base_quat, ball_world - torso_pos)

    def get_observation(self, env_data, ctrl_data):
        ball_pos = self._ball_pos_pelvis(env_data, ctrl_data) * self.cfg_policy.ball_pos_scale
        ang_vel = env_data.base_ang_vel * self.cfg_policy.ang_vel_scale
        proj_grav = self._projected_gravity(env_data.torso_quat)
        dof_pos = (env_data.dof_pos - self.default_dof_pos) * self.cfg_policy.dof_pos_scale
        dof_vel = env_data.dof_vel * self.cfg_policy.dof_vel_scale
        obs = np.concatenate([ball_pos, ang_vel, proj_grav, dof_pos, dof_vel, self.last_action])
        return obs.astype(np.float32), {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if self.actor_critic is None:
            raise RuntimeError("ActorCritic not loaded. Set load_checkpoint=True to enable inference.")

        self.history_buf.append(obs)
        # ActorCritic expects obs_history length = num_one_step_obs * actor_history_length
        obs_input = np.array(self.history_buf).flatten()
        with torch.no_grad():
            action = self.actor_critic.act_inference(torch.from_numpy(obs_input).float().unsqueeze(0))
        action = action.cpu().numpy().squeeze().astype(np.float32)
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        self.last_action = action.copy()
        return action

    def get_pd_target(self, obs: np.ndarray) -> np.ndarray:
        action = self.get_action(obs)
        return action * self.action_scale + self.default_pos
