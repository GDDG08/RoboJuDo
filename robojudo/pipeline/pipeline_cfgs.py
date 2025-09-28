from typing import Any

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg
from robojudo.tools.debug_log import DebugCfg


class PipelineCfg(Config):
    pipeline_type: str  # name of the pipeline class
    # ===== Pipeline Config =====
    device: str = "cpu"

    debug: DebugCfg = DebugCfg()

    run_fullspeed: bool = False
    """If True, run the pipeline at full speed, ignoring the desired frequency"""


class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    policy: PolicyCfg | Any
    ctrl: list[CtrlCfg | Any] = []


class RlMultiPolicyPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlMultiPolicyPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    policy: PolicyCfg | Any
    """Main policy, as init"""
    policy_extra: list[PolicyCfg | Any] = []
    """Extra policies, can be switched to"""

    # TODO: single policy, multi chpt
