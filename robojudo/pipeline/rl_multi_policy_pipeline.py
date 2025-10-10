import logging

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlMultiPolicyPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter

logger = logging.getLogger(__name__)


class PolicySwitch(Policy):
    def __init__(self, cfg_policies: list[PolicyCfg], env_joint_names: list[str], device: str = "cpu"):
        # Duck Policy, no init
        self.device = device
        self.policies: dict[str, dict] = {}

        for cfg_policy in cfg_policies:
            policy_type = cfg_policy.policy_type
            policy_name = policy_type
            if hasattr(cfg_policy, "policy_name"):
                policy_name += "@" + cfg_policy.policy_name  # type: ignore

            while policy_name in self.policies.keys():
                policy_name += "_new"

            policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
            policy: Policy = policy_class(
                cfg_policy=cfg_policy,
                device=self.device,
            )
            self.policies[policy_name] = {
                "policy": policy,
                "cfg": cfg_policy,
                "obs_adapter": DoFAdapter(
                    env_joint_names,
                    policy.cfg_obs_dof.joint_names,
                ),
                "actions_adapter": DoFAdapter(
                    policy.cfg_action_dof.joint_names,
                    env_joint_names,
                ),
            }
        self.current_policy_name: str
        self.set_policy(list(self.policies.keys())[0])

    def set_policy(self, policy_name: str):
        if policy_name not in self.policies.keys():
            raise ValueError(f"Policy {policy_name} not found in policies.")
        self.current_policy_name = policy_name
        logger.warning(f"Switched to policy: {policy_name}")

    def get_policy_set(self) -> dict:
        if self.current_policy_name in self.policies:
            return self.policies[self.current_policy_name]
        else:
            raise ValueError(f"Current policy {self.current_policy_name} not found in policies.")

    def get_policy_inst(self) -> Policy:
        return self.get_policy_set()["policy"]

    def get_policy_obs_adapter(self) -> DoFAdapter:
        return self.get_policy_set()["obs_adapter"]

    def get_policy_actions_adapter(self) -> DoFAdapter:
        return self.get_policy_set()["actions_adapter"]

    def get_observation(self, env_data: Box, ctrl_data: Box):
        observation_all = {}
        for policy_name, policy_set in self.policies.items():
            obs_adapter = policy_set["obs_adapter"]

            env_data_adapted = env_data.copy()
            env_data_adapted.dof_pos = obs_adapter.fit(env_data_adapted.dof_pos)
            env_data_adapted.dof_vel = obs_adapter.fit(env_data_adapted.dof_vel)

            observation = policy_set["policy"].get_observation(env_data_adapted, ctrl_data)
            observation_all[policy_name] = observation

        return observation_all[self.current_policy_name]

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # run single policy
        return self.get_policy_inst().get_action(obs)
        # # run all policies
        # action_all = {}
        # for policy_name, policy_set in self.policies.items():
        #     actions = policy_set["policy"].get_action(obs)
        #     action_all[policy_name] = actions

        # return action_all[self.current_policy_name]

    def get_action_adapted(self, obs: np.ndarray) -> np.ndarray:
        action = self.get_action(obs)
        actions_adapter = self.get_policy_actions_adapter()
        return actions_adapter.fit(action)

    def reset(self):
        return self.get_policy_inst().reset()

    def post_step_callback(self, commands: list[str] | None = None):
        # for _, policy_set in self.policies.items():
        #     policy = policy_set["policy"]
        #     policy.post_step_callback(commands)
        return self.get_policy_inst().post_step_callback(commands)

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        return self.get_policy_inst().debug_viz(visualizer, env_data, ctrl_data, extras)


@pipeline_registry.register
class RlMultiPolicyPipeline(RlPipeline):
    cfg: RlMultiPolicyPipelineCfg

    def __init__(self, cfg: RlMultiPolicyPipelineCfg):
        # Skip RlPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicySwitch(
            cfg_policies=[self.cfg.policy] + self.cfg.policy_extra,
            env_joint_names=self.env.joint_names,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.get_policy_inst().cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.self_check()
        self.reset()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        policy_switch_target = None
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Killed by remote!")
                    self.env.shutdown()
                case "[ENV_RESET]":
                    logger.warning("Env reset!")
                    self.env.reset()
                case "[POLICY_TOGGLE]":
                    logger.warning("Policy toggled!")
                    next_policy_name = self.policy.current_policy_name
                    policy_names = list(self.policy.policies.keys())
                    if len(policy_names) > 1:
                        next_idx = (policy_names.index(self.policy.current_policy_name) + 1) % len(policy_names)
                        next_policy_name = policy_names[next_idx]
                    policy_switch_target = next_policy_name
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    policy_id = int(cmd.split(",")[1])
                    policy_names = list(self.policy.policies.keys())
                    if policy_id < len(policy_names):
                        policy_switch_target = policy_names[policy_id]

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # Handle policy switch after step to avoid mid-step change
        if policy_switch_target is not None:
            self.policy.reset()  # TODO: reset then warm up
            self.policy.set_policy(policy_switch_target)
            self.env.reset()
            self.env.update_dof_cfg(override_cfg=self.policy.get_policy_inst().cfg_action_dof)
            logger.warning(f"Policy switched to {policy_switch_target}!")

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

        actions_adapted = self.policy.get_action_adapted(obs)
        pd_target = actions_adapted + self.env.default_pos

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            # logger.debug(pd_target)

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)


if __name__ == "__main__":
    pass
