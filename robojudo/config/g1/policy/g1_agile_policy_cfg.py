"""G1 AGILE policy configs.

Each cfg points at one of the two policies populated by
``scripts/pull_agile_example_assets.py`` under
``RoboJuDo/assets/models/g1/agile/<label>/policy.{yaml,pt}``. The asset folder
is git-ignored; assets are pulled on demand.
"""

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import AgilePolicyCfg

_AGILE_ASSETS = ASSETS_DIR / "models" / "g1" / "agile"


class G1AgileVelocityHeightPolicyCfg(AgilePolicyCfg):
    """RNN-LSTM recurrent student (TorchScript). Default for ``g1_agile``."""

    robot: str = "g1"
    yaml_path: str = (_AGILE_ASSETS / "velocity_height_g1" / "policy.yaml").as_posix()
    checkpoint_path: str = (_AGILE_ASSETS / "velocity_height_g1" / "policy.pt").as_posix()
    # Layout = (num_layers, batch=1, hidden).
    rnn_hidden_shape: list[int] | None = [2, 1, 128]
    command_remap: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


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
    command_remap: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
