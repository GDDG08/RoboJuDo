"""Configuration for the H1_2 ProtoMotions tracker policy.

Near-verbatim copy of ``g1_protomotions_tracker_cfg.py`` with
``robot: str = "h1_2"`` so ``ProtoMotionsTrackerPolicyCfg.policy_file``
resolves to ``assets/models/h1_2/protomotions_tracker/<onnx_name>.onnx``.
"""

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class H1_2ProtoMotionsTrackerPolicyCfg(PolicyCfg):
    """Config for :class:`ProtoMotionsTrackerPolicy` on H1_2."""

    policy_type: str = "ProtoMotionsTrackerPolicy"
    robot: str = "h1_2"
    disable_autoload: bool = True

    onnx_name: str = "unified_pipeline"
    onnx_path: str | None = None
    motion_path: str = ""
    motion_index: int = 0

    @property
    def policy_file(self) -> str:
        if self.onnx_path is not None:
            return self.onnx_path
        return (ASSETS_DIR / f"models/{self.robot}/protomotions_tracker/{self.onnx_name}.onnx").as_posix()

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    obs_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
    action_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
