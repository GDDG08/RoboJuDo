"""G1 AGILE policy configs.

Each cfg points at one of the two policies populated by
``scripts/pull_agile_example_assets.py`` under
``RoboJuDo/assets/models/g1/agile/<label>/policy.{yaml,pt}``. The asset folder
is git-ignored; assets are pulled on demand.

Command semantics are picked per policy via the
``AgileCommandSourceCfg``-derived ``command_source`` field — the policy class
itself is command-shape-agnostic.
"""

from pydantic import Field

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import (
    AgilePolicyCfg,
    AnyAgileCommandSourceCfg,
    VelocityHeightCommandSourceCfg,
)

_AGILE_ASSETS = ASSETS_DIR / "models" / "g1" / "agile"
_VEL_REMAP = [
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
]


class G1AgileVelocityHeightPolicyCfg(AgilePolicyCfg):
    """RNN-LSTM recurrent student (TorchScript). Default for ``g1_agile``."""

    robot: str = "g1"
    yaml_path: str = (_AGILE_ASSETS / "velocity_height_g1" / "policy.yaml").as_posix()
    checkpoint_path: str = (_AGILE_ASSETS / "velocity_height_g1" / "policy.pt").as_posix()
    # Layout = (num_layers, batch=1, hidden).
    rnn_hidden_shape: list[int] | None = [2, 1, 128]
    command_source: AnyAgileCommandSourceCfg | None = Field(
        default_factory=lambda: VelocityHeightCommandSourceCfg(command_remap=_VEL_REMAP)
    )


class G1AgileVelocityHeightOnnxPolicyCfg(G1AgileVelocityHeightPolicyCfg):
    """Same RNN-LSTM recurrent student, loaded from the shipped ``policy.onnx``.

    Verifies the ONNX runtime path. Requires ``onnxruntime`` to be installed.
    """

    checkpoint_path: str = (_AGILE_ASSETS / "velocity_height_g1" / "policy.onnx").as_posix()
    rnn_hidden_shape: list[int] | None = None  # ignored for ONNX; metadata only


class G1AgileVelocityHistoryPolicyCfg(AgilePolicyCfg):
    """MLP with per-term ``history_length=5`` in the YAML.

    Velocity-only command (no height); exercises the HistoryBuffer code path.
    """

    robot: str = "g1"
    yaml_path: str = (_AGILE_ASSETS / "velocity_g1" / "policy.yaml").as_posix()
    checkpoint_path: str = (_AGILE_ASSETS / "velocity_g1" / "policy.pt").as_posix()
    rnn_hidden_shape: list[int] | None = None
    command_source: AnyAgileCommandSourceCfg | None = Field(
        default_factory=lambda: VelocityHeightCommandSourceCfg(command_remap=_VEL_REMAP)
    )
