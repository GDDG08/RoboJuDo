"""AGILE policy plug-in for RoboJuDo.

Consumes the AGILE deploy contract — an IO-descriptor YAML + checkpoint
(TorchScript MLP/RNN, ONNX, or raw RSL-RL) — without importing the ``agile``
Python package at runtime.
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
from robojudo.policy.utils.agile_io import (
    AgilePolicyRunner,
    AgileYamlObservationBuilder,
    SimState,
    quat_xyzw_to_wxyz,
)
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.util_func import command_remap

logger = logging.getLogger(__name__)


@policy_registry.register
class AgilePolicy(Policy):
    """RoboJuDo plug-in for AGILE-exported policies."""

    cfg_policy: AgilePolicyCfg

    @staticmethod
    def _auto_pull_assets(yaml_path: Path, ckpt_path: Path) -> None:
        """Load and invoke the pull script's ``ensure_agile_assets`` helper.

        ``scripts/`` is not a Python package, so we load it by file path. If a
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

        # Inject derived DoFConfigs into the cfg (pydantic stores them on the cfg copy)
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

        # Command-vector length: take it from the first command obs term.
        self._command_dim = 0
        for term in self._obs_builder.terms:
            if term.name in (
                "generated_commands",
                "velocity_and_height_command",
                "locomotion_command",
                "navigation_command",
            ):
                self._command_dim = term.obs_dim
                break

        self.reset()

    # ------------------------------------------------------------------
    # Policy API
    # ------------------------------------------------------------------

    def reset(self):
        self._obs_builder.reset()
        self._policy_runner.reset()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        # Push the zero last_action so terms have it available before the first step.
        self._obs_builder.set_last_action(torch.from_numpy(self.last_action.astype(np.float32)))

    def post_step_callback(self, commands=None):
        # AGILE policies are stateless beyond RNN hidden / history; nothing else to advance.
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_command(self, ctrl_data) -> np.ndarray:
        """Build the command vector for this policy from controller input.

        Length matches the policy's command obs term. For velocity-only policies
        we emit ``[vx, vy, wz]``; for velocity+height we additionally append
        the configured target height.
        """
        defaults = self.cfg_policy.command_defaults
        vx = float(defaults.get("linear_x", 0.0))
        vy = float(defaults.get("linear_y", 0.0))
        wz = float(defaults.get("angular_z", 0.0))
        height = float(defaults.get("height", 0.72))

        remap = self.cfg_policy.command_remap

        for key, data in ctrl_data.items():
            if key in ("JoystickCtrl", "UnitreeCtrl"):
                axes = data["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                if len(remap) >= 3:
                    vx = float(command_remap(ly, remap[0]))
                    vy = float(command_remap(lx, remap[1]))
                    wz = float(command_remap(rx, remap[2]))
                else:
                    vx, vy, wz = float(ly), float(lx), float(rx)
                break
            if key == "KeyboardCtrl":
                events = data.get("keyboard_event", [])
                for event in events:
                    if event.get("type") != "keyboard":
                        continue
                    pressed = float(event["pressed"]) * 1.0
                    name = event["name"]
                    if name == "w":
                        vx = pressed
                    elif name == "s":
                        vx = -pressed
                    elif name == "a":
                        vy = -pressed
                    elif name == "d":
                        vy = pressed
                    elif name == "e":
                        wz = pressed
                    elif name == "q":
                        wz = -pressed
                break

        full = np.array([vx, vy, wz, height], dtype=np.float32)
        if self._command_dim <= 0:
            return full
        if self._command_dim >= 4:
            out = np.zeros(self._command_dim, dtype=np.float32)
            out[:4] = full
            return out
        return full[: self._command_dim]

    def _build_sim_state(self, env_data) -> SimState:
        td = torch.device(self.device)
        root_quat_wxyz = quat_xyzw_to_wxyz(np.asarray(env_data.base_quat, dtype=np.float32))
        lin = env_data.get("base_lin_vel") if hasattr(env_data, "get") else getattr(env_data, "base_lin_vel", None)
        return SimState(
            joint_pos=torch.from_numpy(np.asarray(env_data.dof_pos, dtype=np.float32)).to(td),
            joint_vel=torch.from_numpy(np.asarray(env_data.dof_vel, dtype=np.float32)).to(td),
            root_quat=torch.from_numpy(root_quat_wxyz).to(td),
            root_ang_vel=torch.from_numpy(np.asarray(env_data.base_ang_vel, dtype=np.float32)).to(td),
            root_lin_vel=torch.from_numpy(np.asarray(lin, dtype=np.float32)).to(td) if lin is not None else None,
        )

    # ------------------------------------------------------------------
    # Pipeline-facing API
    # ------------------------------------------------------------------

    def get_observation(self, env_data, ctrl_data):
        command = self._get_command(ctrl_data)
        sim_state = self._build_sim_state(env_data)
        td = torch.device(self.device)
        obs = self._obs_builder.compute(
            sim_state,
            command=torch.from_numpy(command).to(td),
            last_action=torch.from_numpy(self.last_action.astype(np.float32)).to(td),
        )
        return obs.detach().cpu().numpy().astype(np.float32), {"commands": command}

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

        # Update obs builder with the raw (unscaled) last action for the next step.
        self._obs_builder.set_last_action(torch.from_numpy(action.astype(np.float32)).to(td))

        return action * self._action_scales

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        commands = extras.get("commands")
        if commands is None or visualizer is None:
            return
        base_pos = env_data["base_pos"] if env_data["base_pos"] is not None else np.zeros(3)
        base_quat = env_data["base_quat"]
        cmd_x = float(commands[0]) if len(commands) > 0 else 0.0
        cmd_y = float(commands[1]) if len(commands) > 1 else 0.0
        cmd_yaw = float(commands[2]) if len(commands) > 2 else 0.0
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
