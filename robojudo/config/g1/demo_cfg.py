from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import G1MotionCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg  # noqa: F401
from .g1_cfg import g1_beyondmimic
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg  # noqa: F401


@cfg_registry.register
class G1DemoCfg(RlMultiPolicyPipelineCfg):
    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "!": "[POLICY_SWITCH],0",
                "@": "[POLICY_SWITCH],1",
                "#": "[POLICY_SWITCH],2",
                "$": "[POLICY_SWITCH],3",
                "%": "[POLICY_SWITCH],4",
                "^": "[POLICY_SWITCH],5",
                "&": "[POLICY_SWITCH],6",
                "*": "[POLICY_SWITCH],7",
                "(": "[POLICY_SWITCH],8",
                ")": "[POLICY_SWITCH],9",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
                "RB+A": "[POLICY_SWITCH],4",
                "RB+X": "[POLICY_SWITCH],5",
                "RB+Y": "[POLICY_SWITCH],6",
                "RB+B": "[POLICY_SWITCH],7",
            }
        ),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    policy_extra: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name="2025-09-10_11-05-25_dance1_subject2_wose",
            use_motion_from_model=True,
            without_state_estimator=True,
            max_timestep=1080,
        ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-08-17_15-35-10_dance1_subject2",
        #     use_motion_from_model=True,
        #     without_state_estimator=False,
        #     max_timestep=700,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-03_21-00-31_Box",
        #     use_motion_from_model=True,
        #     max_timestep=900,
        # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-03_21-54-01_violin",
        # #     use_motion_from_model=True,
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-04_15-42-30_Box_wose",
        # #     without_state_estimator=True,
        # #     use_motion_from_model=True,
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-15_22-31-04_charleston_g1", use_motion_from_model=True, max_timestep=320
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-15_22-32-11_walk_backward_g1", use_motion_from_model=True, max_timestep=300
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-15_23-56-49_walk_straight_g1", use_motion_from_model=True, max_timestep=900
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-15_21-14-31_violin_g1", use_motion_from_model=True, max_timestep=550
        # # ),
        # # G1BeyondMimicPolicyCfg(policy_name="2025-09-15_21-14-24_box_g1", use_motion_from_model=True, max_timestep=150),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-16_00-03-03_jump_g1",
        # #     use_motion_from_model=True,
        # #     max_timestep=220
        # # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-16_01-19-47_waltz_g1", use_motion_from_model=True, max_timestep=900
        # # ),
        # # G1BeyondMimicPolicyCfg(policy_name="2025-09-16_01-38-22_run_g1", use_motion_from_model=True, max_timestep=160),
        # # batch
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_02-08-44_conduct-orcestra_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=850,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_02-09-20_akimbo_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=650,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_04-43-32_one_step_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=400,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_06-16-13_elephant_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=900,
        #     start_timestep=50,
        # ),
        # # G1BeyondMimicPolicyCfg(
        # #     policy_name="2025-09-24_07-22-46_run_g1_wose",
        # #     use_motion_from_model=True,
        # #     without_state_estimator=True,
        # #     max_timestep=180,
        # # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_10-03-55_violin_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=-1,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_10-29-27_hip-hop_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=-1,
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="2025-09-24_12-34-17_walk_backward_g1_wose",
        #     use_motion_from_model=True,
        #     without_state_estimator=True,
        #     max_timestep=420,
        # ),
        G1BeyondMimicPolicyCfg(
            policy_name="2025-09-24_15-51-19_side-stepping_g1_wose",
            use_motion_from_model=True,
            without_state_estimator=True,
            max_timestep=-1,
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="2025-09-24_16-05-36_jumping_jacks_g1_wose",
            use_motion_from_model=True,
            without_state_estimator=True,
            max_timestep=150,
        ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="", use_motion_from_model=True, without_state_estimator=True, max_timestep=-1
        # ),
    ]


@cfg_registry.register
class DR(G1DemoCfg):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers_extra={
                "R1+Down": "[POLICY_SWITCH],0",
                "R1+Left": "[POLICY_SWITCH],1",
                "R1+Up": "[POLICY_SWITCH],2",
                "R1+Right": "[POLICY_SWITCH],3",
                "R1+A": "[POLICY_SWITCH],4",
                "R1+X": "[POLICY_SWITCH],5",
                "R1+Y": "[POLICY_SWITCH],6",
                "R1+B": "[POLICY_SWITCH],7",
            }
        )
    ]


@cfg_registry.register
class DR2(g1_beyondmimic):
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            triggers_extra={
                "R1+Down": "[POLICY_SWITCH],0",
                "R1+Left": "[POLICY_SWITCH],1",
                "R1+Up": "[POLICY_SWITCH],2",
                "R1+Right": "[POLICY_SWITCH],3",
                "R1+A": "[POLICY_SWITCH],4",
                "R1+X": "[POLICY_SWITCH],5",
                "R1+Y": "[POLICY_SWITCH],6",
                "R1+B": "[POLICY_SWITCH],7",
            }
        )
    ]
