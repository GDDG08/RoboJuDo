"""H1_2 pipeline configs for RoboJuDo.

Minimal H1_2 support authored for the RoboJuDo verification prep task
(see /oscar/data/stellex/glvov/imprint/wbc_push/robojudo_prep.md). Mirrors
robojudo/config/g1/g1_cfg.py's ``g1_protomotions_tracker`` entry.
"""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.h1_2_mujoco_env_cfg import H1_2MujocoEnvCfg
from .policy.h1_2_protomotions_tracker_cfg import H1_2ProtoMotionsTrackerPolicyCfg


@cfg_registry.register
class h1_2_protomotions_tracker(RlPipelineCfg):
    """ProtoMotions H1_2 tracker with cached motion library.

    Use ``scripts/run_tracker_pipeline.py`` -- it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic
    ``run_pipeline.py`` does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c h1_2_protomotions_tracker \\
            --motion-path assets/motions/h1_2/h1_2_random_subset_tiny.pt \\
            --motion-index 0
    """

    robot: str = "h1_2"
    env: H1_2MujocoEnvCfg = H1_2MujocoEnvCfg(
        born_place_align=False,
        random_heading=False,
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "r": "[MOTION_RESET]",
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "<": "[MOTION_FADE_IN]",
                ">": "[MOTION_FADE_OUT]",
            },
        ),
    ]

    policy: H1_2ProtoMotionsTrackerPolicyCfg = H1_2ProtoMotionsTrackerPolicyCfg()
