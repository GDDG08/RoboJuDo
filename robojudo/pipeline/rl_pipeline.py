import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.reset()
        self.self_check()
        self.policy.reset()  # reset frame counter after dry-run steps

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def _inner_policy(self):
        """Return the unwrapped inner Policy (e.g. ProtoMotionsTrackerPolicy)."""
        return getattr(self.policy, "policy", self.policy)

    @property
    def _has_default_pose_mode(self) -> bool:
        return hasattr(self._inner_policy(), "set_default_pose_mode")

    def _set_default_pose_mode(self, enabled: bool):
        """Enable/disable default-pose mode on the inner policy (if supported)."""
        inner = self._inner_policy()
        if hasattr(inner, "set_default_pose_mode"):
            inner.set_default_pose_mode(enabled)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        self.policy.reset()
        self.ctrl_manager.reset()

        # Blend-out state: transitions policy → init pose at end of motion.
        self._blend_out_active = False
        self._blend_out_step = 0
        self._blend_out_duration = int(5.0 * self.freq)  # 5 seconds

        # For tracker policies with default-pose mode, ramp/blend target is
        # the env's default standing pose.  Otherwise, use motion frame 0.
        if self._has_default_pose_mode:
            self._init_dof_pos = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32)
        else:
            self._init_dof_pos = np.asarray(self.policy.get_init_dof_pos(), dtype=np.float32)

        self._pending_blend_in = False
        self._blend_in_completed = False
        self._user_fade_out = False  # True when fade-out was user-triggered (not auto)
        self._prepare_seconds = None  # set by prepare() for re-use on reset

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                self.policy.reset_alignment()
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                        self.policy.reset_alignment()
                case "[MOTION_RESET]" | "[MOTION_FADE_IN]":
                    self._blend_out_active = False
                    self._blend_out_step = 0
                    self._user_fade_out = False
                    if self._has_default_pose_mode:
                        # Policy is already active — just switch target
                        # from default pose to motion (instant, no blend).
                        logger.info(
                            f"{command} — starting motion from frame 0"
                        )
                        self._set_default_pose_mode(False)
                    else:
                        # Legacy path: full blend-in needed.
                        logger.info(
                            f"{command} — re-entering blend-in phase"
                        )
                        self._pending_blend_in = True
                case "[MOTION_FADE_OUT]":
                    if self._has_default_pose_mode:
                        logger.info("Fade out — switching to default pose mode")
                        self._set_default_pose_mode(True)
                        self._user_fade_out = True
                    elif not self._blend_out_active:
                        logger.info("Fade out — blending to default pose")
                        self._blend_out_active = True
                        self._blend_out_step = 0
                        self._user_fade_out = True
                        inner = self._inner_policy()
                        if hasattr(inner, "_paused"):
                            inner._paused = True

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(obs)

        # -- Detect motion done --
        callbacks = extras.get("CALLBACK", [])
        if "[MOTION_DONE]" in callbacks and not self._blend_out_active:
            if self._has_default_pose_mode:
                logger.info("Motion done — switching to default pose mode")
                self._set_default_pose_mode(True)
            else:
                logger.info("Motion done — blending out to default pose")
                self._blend_out_active = True
                self._blend_out_step = 0

        # -- Blend policy output → init pose (frame 0) --
        if self._blend_out_active:
            alpha = min(self._blend_out_step / max(self._blend_out_duration, 1), 1.0)
            pd_target = (1 - alpha) * pd_target + alpha * self._init_dof_pos
            self._blend_out_step += 1

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

        # Handle pending blend-in (after MOTION_RESET / FADE_IN).
        if self._pending_blend_in:
            self._pending_blend_in = False
            self._run_blend_in()
            self._blend_in_completed = True

    def _run_blend_in(self):
        """Phase 2 of prepare: blend from default pose to policy output.

        Policy runs in default-pose mode (if supported); frame stays at 0
        (no post_step_callback).  After blend, switches to motion tracking.
        """
        secs = self._prepare_seconds or 3.0
        blend_steps = int(secs * self.freq)

        # Enter default-pose mode for the blend-in period.
        self._set_default_pose_mode(True)

        logger.warning(f"Blend-in: default DOF → policy ({blend_steps} steps, {secs:.1f}s)")
        pbar = ProgressBar("Blend in", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            action = (1 - alpha) * self._init_dof_pos + alpha * policy_pd

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Switch to motion tracking — policy sees the jump.
        self._set_default_pose_mode(False)
        logger.warning("Blend-in done — motion starting")

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        elif self._has_default_pose_mode:
            # Ramp to the env's default standing pose (not motion frame 0).
            desired_motor_angle = np.array(
                self.env.dof_cfg.default_pos, dtype=np.float32
            )
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # Convert seconds to steps (at policy frequency).
        # Default: 3s ramp + 5s blend.  CLI --prepare-seconds overrides both.
        if prepare_seconds is not None:
            ramp_steps = int(prepare_seconds * self.freq)
            blend_steps = int(prepare_seconds * self.freq)
        else:
            ramp_steps = int(3.0 * self.freq)
            blend_steps = int(5.0 * self.freq)

        # ── Phase 1: Ramp joints to default pose ──
        logger.warning(
            f"prepare: phase 1 — ramp joints ({ramp_steps} steps, "
            f"{ramp_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: ramp joints", ramp_steps)

        last_step_time = time.time()
        for t in range(ramp_steps):
            current_motor_angle = np.array(self.env.dof_pos)
            alpha = min(t / max(ramp_steps - 1, 1), 1.0)
            action = (1 - alpha) * current_motor_angle + alpha * desired_motor_angle

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Reset policy for a clean start — frame goes back to 0.
        self.reset()
        self._prepare_seconds = prepare_seconds  # restore after reset

        # ── Phase 2: Blend in policy (holding default pose) ──
        # Policy runs in default-pose mode (if supported): it sees synthetic
        # references for the standing pose, not the real motion.
        # Actions blend from raw default DOF to policy output.
        self._set_default_pose_mode(True)

        logger.warning(
            f"prepare: phase 2 — blend policy ({blend_steps} steps, "
            f"{blend_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: blend policy", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            # Run policy observation + action (frame stays at 0).
            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            # Blend: default DOF → policy output
            action = (1 - alpha) * desired_motor_angle + alpha * policy_pd

            self.env.step(action)

            # Do NOT call post_step_callback — frame stays at 0.

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # ── Phase 3: Hold default pose — wait for R to start motion ──
        # Stay in default-pose mode. Motion starts when [MOTION_RESET] is
        # received (user presses R), which calls _set_default_pose_mode(False).
        logger.warning("prepare done — holding default pose, press R to start motion")


if __name__ == "__main__":
    pass
