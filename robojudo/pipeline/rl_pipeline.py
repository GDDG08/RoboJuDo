import logging
import time

import numpy as np

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy
from robojudo.tools.dof import DoFAdapter
from robojudo.utils.progress import ProgressBar

logger = logging.getLogger(__name__)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        policy_class: type[Policy] = getattr(robojudo.policy, self.cfg.policy.policy_type)
        self.policy: Policy = policy_class(
            cfg_policy=self.cfg.policy,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.obs_adapter = DoFAdapter(
            self.env.joint_names,
            self.policy.cfg_obs_dof.joint_names,
        )
        self.actions_adapter = DoFAdapter(
            self.policy.cfg_action_dof.joint_names,
            self.env.joint_names,
        )

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.self_check()
        self.reset()

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        # self.env.reset(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Killed by remote!")
                    self.env.shutdown()
                case "[ENV_RESET]":
                    logger.warning("Env reset!")
                    self.env.reset()

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

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
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data_adapted)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        obs, extras = self.policy.get_observation(env_data_adapted, ctrl_data)
        actions = self.policy.get_action(obs)

        actions_adapted = self.actions_adapter.fit(actions)
        pd_target = actions_adapted + self.env.default_pos

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data_adapted, ctrl_data, extras, pd_target)

    def prepare(self, wait=True):
        # TODO: init pos align
        desired_motor_angle = self.env.default_pos.copy()

        logger.info(f"{desired_motor_angle=}")
        current_motor_angle = np.array(self.env.dof_pos)
        logger.info(f"{current_motor_angle=}")

        traj_len = 1000
        last_step_time = time.time()
        logger.warning("prepare_init")
        pbar = ProgressBar("Prepare", traj_len)

        for t in range(traj_len):
            current_motor_angle = np.array(self.env.dof_pos)

            blend_ratio = np.minimum(t / 300, 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle

            # warm up network
            self.step(dry_run=True)

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            # print(t)
            pbar.update()

            if t == 0.9 * traj_len:
                logger.info(f"{'=' * 10} RESET ZERO POSITION {'=' * 10}")
                self.reset()

        time.sleep(0.01)
        pbar.close()
        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
