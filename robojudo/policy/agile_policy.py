"""AGILE policy plug-in for RoboJuDo.

Consumes the AGILE deploy contract — an IO-descriptor YAML + checkpoint
(TorchScript MLP/RNN, ONNX, or raw RSL-RL) — without importing the ``agile``
Python package at runtime.

The policy class is command-shape-agnostic: it composes a polymorphic
:class:`AgileCommandSource` (velocity-height, motion-file, zero, …) instead of
baking in any particular command semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from robojudo.config import ROOT_DIR
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import AgilePolicyCfg
from robojudo.policy.utils.agile_commands import build_command_source
from robojudo.policy.utils.agile_io import (
    AgilePolicyRunner,
    AgileYamlObservationBuilder,
    SimState,
    quat_xyzw_to_wxyz,
)
from robojudo.tools.tool_cfgs import DoFConfig

logger = logging.getLogger(__name__)


@policy_registry.register
class AgilePolicy(Policy):
    """RoboJuDo plug-in for AGILE-exported policies."""

    cfg_policy: AgilePolicyCfg

    @staticmethod
    def _auto_pull_assets(yaml_path: Path, ckpt_path: Path) -> None:
        """Invoke the pull script's ``ensure_agile_assets`` helper by file path.

        ``scripts/`` is not a Python package, so we load it dynamically. If a
        sibling ``WBC_AGILE`` checkout (or ``$WBC_AGILE_DIR``) is available, the
        assets are auto-pulled. Otherwise the user is asked to run the script
        with ``--repo/--commit``.
        """
        import importlib.util

        script_path = Path(ROOT_DIR) / "scripts" / "pull_agile_example_assets.py"
        try:
            spec = importlib.util.spec_from_file_location("_robojudo_pull_agile_assets", script_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.ensure_agile_assets(verbose=True)
        except Exception as e:
            raise FileNotFoundError(
                f"AGILE deploy asset missing ({yaml_path} / {ckpt_path}). "
                f"Run scripts/pull_agile_example_assets.py to fetch it. "
                f"Auto-pull error: {e}"
            ) from e

    def __init__(self, cfg_policy: AgilePolicyCfg, device: str = "cpu"):
        # --- Load YAML ----------------------------------------------------
        yaml_path = Path(cfg_policy.yaml_path)
        ckpt_path = Path(cfg_policy.checkpoint_path)
        if not yaml_path.exists() or not ckpt_path.exists():
            self._auto_pull_assets(yaml_path, ckpt_path)
        with open(yaml_path) as f:
            self._yaml: dict = yaml.safe_load(f)

        # --- Build DoFConfigs from YAML -----------------------------------
        art = self._yaml["articulations"]["robot"]
        all_names: list[str] = list(art["joint_names"])
        obs_dof = DoFConfig(
            joint_names=all_names,
            default_pos=list(art["default_joint_pos"]),
            stiffness=list(art["default_joint_stiffness"]),
            damping=list(art["default_joint_damping"]),
        )

        # Some YAMLs include auxiliary action terms with no controlled joints
        # (e.g. ``harness_action``). Keep only terms that actually drive joints.
        all_act_terms = self._yaml["actions"]
        act_terms = [t for t in all_act_terms if t.get("joint_names")]
        if len(act_terms) != 1:
            raise ValueError(
                f"AgilePolicy currently supports exactly one non-empty action term, got "
                f"{len(act_terms)} in {yaml_path}. Multi-action YAMLs are reserved for a "
                f"future multi-body pipeline."
            )
        act_term = act_terms[0]
        act_names: list[str] = list(act_term["joint_names"])
        all_idx = [all_names.index(j) for j in act_names]

        action_dof = DoFConfig(
            joint_names=act_names,
            default_pos=list(act_term["offset"]),
            stiffness=[art["default_joint_stiffness"][i] for i in all_idx],
            damping=[art["default_joint_damping"][i] for i in all_idx],
        )

        cfg = cfg_policy.model_copy()
        cfg.obs_dof = obs_dof
        cfg.action_dof = action_dof

        # Pull control freq from YAML scene block if present.
        scene = self._yaml.get("scene", {})
        if "dt" in scene and scene["dt"] > 0:
            cfg.freq = int(round(1.0 / float(scene["dt"])))

        super().__init__(cfg_policy=cfg, device=device)

        # --- Per-joint action scale + clip -------------------------------
        scale_raw = act_term.get("scale")
        if isinstance(scale_raw, list) and len(scale_raw) > 0 and isinstance(scale_raw[0], list):
            scale_raw = scale_raw[0]
        if scale_raw is None:
            scale_raw = [1.0] * self.num_actions
        self._action_scales = np.asarray(scale_raw, dtype=np.float32)

        clip_raw = act_term.get("clip")
        if clip_raw is not None:
            self._clip_min = np.array([c[0] for c in clip_raw], dtype=np.float32)
            self._clip_max = np.array([c[1] for c in clip_raw], dtype=np.float32)
        else:
            self._clip_min = None
            self._clip_max = None

        # --- Build runtime obs/runner ------------------------------------
        td = torch.device(self.device)
        self._obs_builder = AgileYamlObservationBuilder(self._yaml, joint_names=all_names, device=td)
        self._policy_runner = AgilePolicyRunner.from_checkpoint(
            Path(cfg_policy.checkpoint_path),
            device=td,
            rnn_hidden_shape=cfg_policy.rnn_hidden_shape,
        )

        # --- Build the command source (single point of policy-shape decoupling) ---
        self._command_dim = self._obs_builder.command_dim
        self._command_source = build_command_source(
            cfg_policy.command_source,
            yaml=self._yaml,
            command_dim=self._command_dim,
            policy_dt=self.dt,
            device=self.device,
        )

        # Latest CommandPayload returned by the source — surfaced for debug_viz.
        self._last_payload = None

        self.reset()

    # ------------------------------------------------------------------
    # Policy API
    # ------------------------------------------------------------------

    def reset(self):
        self._obs_builder.reset()
        self._policy_runner.reset()
        if getattr(self, "_command_source", None) is not None:
            self._command_source.reset()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._obs_builder.set_last_action(torch.from_numpy(self.last_action.astype(np.float32)))

    def post_step_callback(self, commands=None):
        # Advance any time-driven command source (motion playback, schedulers, …).
        if getattr(self, "_command_source", None) is not None:
            self._command_source.post_step()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sim_state(self, env_data) -> SimState:
        td = torch.device(self.device)
        root_quat_wxyz = quat_xyzw_to_wxyz(np.asarray(env_data.base_quat, dtype=np.float32))
        lin = env_data.get("base_lin_vel") if hasattr(env_data, "get") else getattr(env_data, "base_lin_vel", None)
        # Anchor body pose — populated when available so motion-tracking terms work.
        # RoboJuDo's env exposes torso_pos / torso_quat in xyzw; we convert to wxyz.
        anchor_pos = None
        anchor_quat = None
        torso_pos = getattr(env_data, "torso_pos", None) if not hasattr(env_data, "get") else env_data.get("torso_pos")
        torso_quat = (
            getattr(env_data, "torso_quat", None) if not hasattr(env_data, "get") else env_data.get("torso_quat")
        )
        if torso_pos is not None:
            anchor_pos = torch.from_numpy(np.asarray(torso_pos, dtype=np.float32)).to(td)
        if torso_quat is not None:
            anchor_quat = torch.from_numpy(quat_xyzw_to_wxyz(np.asarray(torso_quat, dtype=np.float32))).to(td)
        return SimState(
            joint_pos=torch.from_numpy(np.asarray(env_data.dof_pos, dtype=np.float32)).to(td),
            joint_vel=torch.from_numpy(np.asarray(env_data.dof_vel, dtype=np.float32)).to(td),
            root_quat=torch.from_numpy(root_quat_wxyz).to(td),
            root_ang_vel=torch.from_numpy(np.asarray(env_data.base_ang_vel, dtype=np.float32)).to(td),
            root_lin_vel=torch.from_numpy(np.asarray(lin, dtype=np.float32)).to(td) if lin is not None else None,
            anchor_body_pos=anchor_pos,
            anchor_body_quat=anchor_quat,
        )

    # ------------------------------------------------------------------
    # Pipeline-facing API
    # ------------------------------------------------------------------

    def get_observation(self, env_data, ctrl_data):
        sim_state = self._build_sim_state(env_data)
        td = torch.device(self.device)

        payload = self._command_source.get(env_data, ctrl_data, sim_state)
        self._last_payload = payload

        cmd_tensor = (
            torch.from_numpy(np.asarray(payload.vector, dtype=np.float32)).to(td)
            if payload.vector is not None
            else None
        )
        extras_tensors: dict[str, torch.Tensor] = {
            k: torch.from_numpy(np.asarray(v, dtype=np.float32)).to(td) for k, v in payload.extras.items()
        }

        obs = self._obs_builder.compute(
            sim_state,
            command=cmd_tensor,
            command_extras=extras_tensors,
            last_action=torch.from_numpy(self.last_action.astype(np.float32)).to(td),
        )
        extras_out = {"command": payload}
        return obs.detach().cpu().numpy().astype(np.float32), extras_out

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        td = torch.device(self.device)
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(td)
        with torch.no_grad():
            action_t = self._policy_runner(obs_t)
        action = action_t.detach().cpu().numpy().squeeze().astype(np.float32)
        if action.ndim == 0:
            action = action.reshape(1)

        action = (1 - self.action_beta) * self.last_action + self.action_beta * action
        self.last_action = action.copy()

        if self._clip_min is not None and self._clip_max is not None:
            action = np.clip(action, self._clip_min, self._clip_max)

        self._obs_builder.set_last_action(torch.from_numpy(action.astype(np.float32)).to(td))

        return action * self._action_scales

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        payload = extras.get("command")
        if visualizer is None or payload is None or payload.vector is None:
            return
        cmd = payload.vector
        if cmd.shape[0] < 3:
            return  # nothing meaningful to draw for non-velocity command shapes
        base_pos = env_data["base_pos"] if env_data["base_pos"] is not None else np.zeros(3)
        base_quat = env_data["base_quat"]
        cmd_x, cmd_y, cmd_yaw = float(cmd[0]), float(cmd[1]), float(cmd[2])
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [cmd_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,
        )
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, cmd_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,
        )
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, cmd_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,
        )
