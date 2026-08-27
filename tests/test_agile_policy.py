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
from robojudo.policy.policy_cfgs import (
    MotionFileCommandSourceCfg,
    VelocityHeightCommandSourceCfg,
    ZeroCommandSourceCfg,
)
from robojudo.policy.utils.agile_commands import (
    CommandPayload,
    MotionFileCommandSource,
    VelocityHeightCommandSource,
    ZeroCommandSource,
    build_command_source,
)


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


class TestAgileCommandSources(unittest.TestCase):
    """Coverage for the polymorphic command-source plug-ins.

    These tests do not load any AGILE checkpoint — they exercise the sources
    standalone with a minimal YAML stub so the policy can be swapped to any
    new command shape (ROS, hand-tracker, …) without touching ``AgilePolicy``.
    """

    YAML_STUB = {
        "articulations": {
            "robot": {
                "joint_names": ["j0", "j1", "j2"],
            }
        }
    }

    def _ctrl_data_joystick(self, lx=0.0, ly=0.0, rx=0.0, ry=0.0):
        return Box({"JoystickCtrl": {"axes": {"LeftX": lx, "LeftY": ly, "RightX": rx, "RightY": ry}}})

    def test_velocity_height_source(self):
        remap = [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
        src = build_command_source(
            VelocityHeightCommandSourceCfg(command_remap=remap),
            yaml=self.YAML_STUB,
            command_dim=4,
            policy_dt=0.02,
            device="cpu",
        )
        self.assertIsInstance(src, VelocityHeightCommandSource)
        payload = src.get(env_data=None, ctrl_data=self._ctrl_data_joystick(ly=0.5), sim_state=None)
        self.assertIsInstance(payload, CommandPayload)
        self.assertEqual(payload.vector.shape, (4,))
        # LeftY > 0 → positive vx via the remap [-1, 0, 1]
        self.assertGreater(float(payload.vector[0]), 0.0)
        keyboard_payload = src.get(
            env_data=None,
            ctrl_data=Box(
                {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "d", "pressed": True}]}}
            ),
            sim_state=None,
        )
        # Keyboard uses the same remap as joystick: positive raw lateral input maps negative for G1.
        self.assertLess(float(keyboard_payload.vector[1]), 0.0)
        # Truncation: command_dim=3 yields [vx,vy,wz]
        src3 = build_command_source(
            VelocityHeightCommandSourceCfg(),
            yaml=self.YAML_STUB,
            command_dim=3,
            policy_dt=0.02,
            device="cpu",
        )
        self.assertEqual(src3.get(None, Box({}), None).vector.shape, (3,))

    def test_zero_source(self):
        src = build_command_source(
            ZeroCommandSourceCfg(command_dim=7),
            yaml=self.YAML_STUB,
            command_dim=4,  # overridden by cfg.command_dim
            policy_dt=0.02,
            device="cpu",
        )
        self.assertIsInstance(src, ZeroCommandSource)
        payload = src.get(env_data=None, ctrl_data=Box({}), sim_state=None)
        self.assertEqual(payload.vector.shape, (7,))
        self.assertTrue(np.all(payload.vector == 0.0))

    def test_command_source_config_roundtrip_preserves_concrete_type(self):
        cfg = G1AgileVelocityHeightPolicyCfg()
        roundtrip = G1AgileVelocityHeightPolicyCfg.model_validate(cfg.model_dump())
        self.assertIsInstance(roundtrip.command_source, VelocityHeightCommandSourceCfg)
        self.assertEqual(roundtrip.command_source.command_remap, cfg.command_source.command_remap)
        self.assertEqual(roundtrip.command_source.defaults["height"], cfg.command_source.defaults["height"])

    def test_motion_file_source_playback_and_reset(self):
        """Synthesize a tiny .npz motion clip, then exercise playback + reset.

        Verifies:
          - command vector = ``concat(joint_pos[t], joint_vel[t])`` reordered into
            the YAML's joint order
          - ``extras["motion_anchor_pos_w" / "motion_anchor_quat_w"]`` come from
            the configured anchor body at frame t
          - ``post_step`` advances the timestep (looping enabled)
          - ``reset`` rewinds playback to frame 0
        """
        import tempfile
        from pathlib import Path as _Path

        T = 4  # frames
        # Motion order: ["mA", "mB"] — must be remapped to YAML order ["j0", "j1"].
        joint_pos = np.array(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [12.0, 22.0],
                [13.0, 23.0],
            ],
            dtype=np.float32,
        )
        joint_vel = joint_pos * 0.1
        # 2 bodies — anchor body is index 1.
        body_pos_w = np.zeros((T, 2, 3), dtype=np.float32)
        body_pos_w[:, 1, 0] = np.arange(T, dtype=np.float32)  # anchor x advances with t
        body_quat_w = np.zeros((T, 2, 4), dtype=np.float32)
        body_quat_w[:, :, 0] = 1.0  # identity quat [w=1, x=y=z=0]

        with tempfile.TemporaryDirectory() as tmp:
            motion_path = _Path(tmp) / "motion.npz"
            np.savez(
                motion_path,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                body_pos_w=body_pos_w,
                body_quat_w=body_quat_w,
            )

            yaml_stub = {
                "articulations": {"robot": {"joint_names": ["jA", "jB"]}},
            }
            cfg = MotionFileCommandSourceCfg(
                motion_path=str(motion_path),
                anchor_body_name="anchor",
                motion_body_names=["other", "anchor"],
                # Motion stores joints in the order ["jB", "jA"]; the source must
                # reorder them to the YAML order ["jA", "jB"] (i.e. swap indices).
                motion_joint_names=["jB", "jA"],
                loop=True,
            )
            src = build_command_source(
                cfg,
                yaml=yaml_stub,
                command_dim=4,  # 2 joint_pos + 2 joint_vel
                policy_dt=0.02,
                device="cpu",
            )
            self.assertIsInstance(src, MotionFileCommandSource)

            # Frame 0 — joint order should be [mB→j0, mA→j1] applied to (jp || jv)
            p0 = src.get(None, Box({}), None)
            np.testing.assert_array_equal(p0.vector, np.array([20.0, 10.0, 2.0, 1.0], dtype=np.float32))
            np.testing.assert_array_equal(p0.extras["motion_anchor_pos_w"], np.array([0.0, 0.0, 0.0]))
            np.testing.assert_array_equal(p0.extras["motion_anchor_quat_w"], np.array([1.0, 0.0, 0.0, 0.0]))

            src.post_step()
            p1 = src.get(None, Box({}), None)
            self.assertEqual(p1.ui["timestep"], 1)
            np.testing.assert_array_equal(p1.extras["motion_anchor_pos_w"][:1], np.array([1.0]))

            # reset() returns to frame 0.
            for _ in range(3):
                src.post_step()
            src.reset()
            p_reset = src.get(None, Box({}), None)
            np.testing.assert_array_equal(p_reset.vector, p0.vector)
            np.testing.assert_array_equal(p_reset.extras["motion_anchor_pos_w"], p0.extras["motion_anchor_pos_w"])

    def test_swap_command_source_on_existing_policy_cfg(self):
        """Demonstrate that ``AgilePolicy`` is command-shape-agnostic by swapping
        its ``command_source`` to :class:`ZeroCommandSource` without touching the
        policy class or YAML."""
        cfg = G1AgileVelocityHeightPolicyCfg()
        cfg.command_source = ZeroCommandSourceCfg()  # let YAML drive command_dim
        pol = AgilePolicy(cfg, device="cpu")
        self.assertIsInstance(pol._command_source, ZeroCommandSource)
        obs, _ = pol.get_observation(_make_env_data(), Box({}))
        action = pol.get_action(obs)
        self.assertEqual(action.shape, (12,))
        self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()
