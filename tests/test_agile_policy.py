"""Layer-1 unit tests for ``AgilePolicy``.

Covers both shipped policies:
  - velocity_height_g1 — RNN-LSTM recurrent student (TorchScript)
  - velocity_g1        — MLP with per-term ``history_length=5`` (TorchScript)

Assets are auto-pulled at ``setUpClass`` time if a sibling ``WBC_AGILE`` checkout
(or ``$WBC_AGILE_DIR``) is available, so a fresh clone of RoboJuDo can run these
tests without a manual pre-step.

For each policy:
  - cfg loads through the pydantic chain
  - obs/action dims match the YAML
  - two consecutive ``get_action()`` steps run without NaNs (exercises RNN
    hidden carry and per-term history-buffer push)
  - ``reset()`` returns the policy to its t=0 action (verifies RNN hidden state
    and history buffers are cleared)
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from box import Box

from robojudo.config import ROOT_DIR
from robojudo.config.g1.policy.g1_agile_policy_cfg import (
    G1AgileVelocityHeightOnnxPolicyCfg,
    G1AgileVelocityHeightPolicyCfg,
    G1AgileVelocityHistoryPolicyCfg,
)
from robojudo.policy.agile_policy import AgilePolicy


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


# (label, cfg_class, expected_num_action_joints)
CFG_MATRIX = [
    ("rnn-recurrent-student-torchscript", G1AgileVelocityHeightPolicyCfg, 12),
    # velocity_history's action term includes waist_roll + waist_pitch.
    ("mlp-history-5", G1AgileVelocityHistoryPolicyCfg, 14),
]
if _has_onnxruntime():
    CFG_MATRIX.append(
        ("rnn-recurrent-student-onnx", G1AgileVelocityHeightOnnxPolicyCfg, 12),
    )


def _make_env_data(num_dofs: int = 29) -> Box:
    return Box(
        {
            "dof_pos": np.zeros(num_dofs, dtype=np.float32),
            "dof_vel": np.zeros(num_dofs, dtype=np.float32),
            "base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # xyzw
            "base_ang_vel": np.zeros(3, dtype=np.float32),
            "base_lin_vel": np.zeros(3, dtype=np.float32),
            "base_pos": np.array([0.0, 0.0, 0.76], dtype=np.float32),
            "base_lin_acc": None,
            "torso_pos": None,
            "torso_quat": None,
            "torso_ang_vel": None,
            "fk_info": None,
        }
    )


def _load_puller():
    script_path = Path(ROOT_DIR) / "scripts" / "pull_agile_example_assets.py"
    spec = importlib.util.spec_from_file_location("_pull_agile_assets", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAgilePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Auto-pull AGILE assets if a sibling WBC_AGILE checkout exists.
        puller = _load_puller()
        try:
            puller.ensure_agile_assets(verbose=False)
        except FileNotFoundError as e:
            raise unittest.SkipTest(f"AGILE assets unavailable and no WBC_AGILE source found: {e}") from e

    def test_all_shapes_load_and_step(self):
        for name, CfgCls, num_action in CFG_MATRIX:
            with self.subTest(policy=name):
                pol = AgilePolicy(CfgCls(), device="cpu")
                self.assertEqual(pol.cfg_obs_dof.num_dofs, 29)
                self.assertEqual(pol.cfg_action_dof.num_dofs, num_action)

                for _ in range(2):
                    obs, _ = pol.get_observation(_make_env_data(), Box({}))
                    self.assertEqual(obs.dtype, np.float32)
                    self.assertTrue(np.isfinite(obs).all(), f"non-finite obs for {name}")

                    action = pol.get_action(obs)
                    self.assertEqual(action.shape, (num_action,))
                    self.assertTrue(np.isfinite(action).all(), f"non-finite action for {name}")

    def test_reset_clears_rnn_hidden(self):
        pol = AgilePolicy(G1AgileVelocityHeightPolicyCfg(), device="cpu")

        obs0, _ = pol.get_observation(_make_env_data(), Box({}))
        a0 = pol.get_action(obs0).copy()

        # Drive the RNN hidden state away from zero.
        for _ in range(10):
            obs, _ = pol.get_observation(_make_env_data(), Box({}))
            pol.get_action(obs)

        pol.reset()
        obs_again, _ = pol.get_observation(_make_env_data(), Box({}))
        a_after_reset = pol.get_action(obs_again)

        np.testing.assert_allclose(
            a0,
            a_after_reset,
            atol=1e-6,
            err_msg="reset() did not restore initial RNN hidden state",
        )

    def test_reset_clears_history_buffer(self):
        pol = AgilePolicy(G1AgileVelocityHistoryPolicyCfg(), device="cpu")

        obs0, _ = pol.get_observation(_make_env_data(), Box({}))
        a0 = pol.get_action(obs0).copy()

        # Warm up the history buffer with non-zero env data.
        env_data_perturbed = _make_env_data()
        env_data_perturbed.dof_pos = np.full(29, 0.05, dtype=np.float32)
        env_data_perturbed.dof_vel = np.full(29, 0.05, dtype=np.float32)
        for _ in range(5):
            obs, _ = pol.get_observation(env_data_perturbed, Box({}))
            pol.get_action(obs)

        pol.reset()
        obs_again, _ = pol.get_observation(_make_env_data(), Box({}))
        a_after_reset = pol.get_action(obs_again)

        np.testing.assert_allclose(
            a0,
            a_after_reset,
            atol=1e-6,
            err_msg="reset() did not clear per-term history buffers",
        )


if __name__ == "__main__":
    unittest.main()
