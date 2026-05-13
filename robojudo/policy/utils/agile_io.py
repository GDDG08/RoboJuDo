"""Runtime utilities for AGILE-exported policies.

This module provides RoboJuDo-local implementations of the small runtime surface
needed to deploy an AGILE policy (IO-descriptor YAML + checkpoint):

- ``HistoryBuffer`` — per-term circular history buffer mirroring AGILE semantics.
- ``YamlObservationTerm`` / ``AgileYamlObservationBuilder`` — parse the YAML
  ``observations.policy`` list and compute the full obs tensor from a small
  ``SimState`` dataclass.
- ``AgilePolicyRunner`` — dispatch to one of TorchScript MLP, TorchScript RNN,
  ONNX, or raw RSL-RL checkpoint runners based on the file contents.
- Quaternion helpers for ``xyzw`` (RoboJuDo env) ↔ ``wxyz`` (AGILE) conventions.

No runtime dependency on the ``agile`` Python package — the YAML/checkpoint
"deploy contract" is consumed directly here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] → [w, x, y, z]."""
    q = np.asarray(q, dtype=np.float32)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → [x, y, z, w]."""
    q = np.asarray(q, dtype=np.float32)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by the inverse of quaternion ``q`` (``[w, x, y, z]``).

    Mirrors AGILE's implementation (which mirrors isaac-deploy/IsaacLab) so the
    projected gravity term is bit-for-bit consistent with training.
    """
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = torch.cross(q_vec, v, dim=0) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# SimState — minimal struct passed into the obs builder
# ---------------------------------------------------------------------------


@dataclass
class SimState:
    """Subset of AGILE's ``SimState`` that the supported obs terms need."""

    joint_pos: torch.Tensor  # (num_dofs,)
    joint_vel: torch.Tensor  # (num_dofs,)
    root_quat: torch.Tensor  # (4,) [w, x, y, z]
    root_ang_vel: torch.Tensor  # (3,)
    root_lin_vel: torch.Tensor | None = None  # (3,) — root frame


# ---------------------------------------------------------------------------
# History buffer + observation term + builder
# ---------------------------------------------------------------------------


class HistoryBuffer:
    """Circular per-term obs history. Equivalent to AGILE's ``HistoryBuffer``."""

    def __init__(self, obs_dim: int, history_length: int, device: torch.device):
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.device = device

        if history_length > 0:
            self.buffer = torch.zeros(history_length, obs_dim, device=device)
            self.index = 0
            self.filled = False
        else:
            self.buffer = None  # type: ignore[assignment]
            self.index = 0
            self.filled = False

    def push(self, obs: torch.Tensor):
        if self.history_length == 0:
            return
        self.buffer[self.index] = obs
        self.index = (self.index + 1) % self.history_length
        if self.index == 0:
            self.filled = True

    def get(self, flatten: bool = True) -> torch.Tensor:
        if self.history_length == 0:
            raise ValueError("Cannot get history when history_length is 0")
        if self.filled:
            ordered = torch.cat([self.buffer[self.index :], self.buffer[: self.index]], dim=0)
        else:
            ordered = self.buffer
        return ordered.flatten() if flatten else ordered

    def reset(self):
        if self.history_length > 0:
            self.buffer.zero_()
            self.index = 0
            self.filled = False


_SUPPORTED_TERMS = (
    "generated_commands",
    "velocity_and_height_command",
    "locomotion_command",
    "navigation_command",
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_pos",
    "joint_vel",
    "joint_vel_rel",
    "last_action",
    "zero_padding",
)


class YamlObservationTerm:
    """Single observation term parsed from the AGILE IO YAML.

    Replicates AGILE's ``ObservationTerm`` raw→clip→scale→history pipeline (no noise
    at deploy time).
    """

    def __init__(self, config: dict, joint_names: list[str], device: torch.device):
        self.name: str = config["name"]
        if self.name not in _SUPPORTED_TERMS:
            raise ValueError(
                f"Unsupported observation term '{self.name}'.\n"
                f"Supported terms: {_SUPPORTED_TERMS}.\n"
                f"Extend robojudo/policy/utils/agile_io.py to add new terms."
            )

        overloads = config.get("overloads", {}) or {}
        self.history_length: int = int(overloads.get("history_length", 0) or 0)
        self.flatten_history: bool = bool(overloads.get("flatten_history_dim", True))
        self.clip = overloads.get("clip", None)
        self.scale = overloads.get("scale", None)

        self.device = device
        self.joint_names_full = joint_names

        shape = config.get("shape", [1])
        if isinstance(shape, list):
            self.obs_dim = int(shape[0]) if len(shape) == 1 else int(np.prod(shape))
        else:
            self.obs_dim = int(shape)

        if "joint_names" in config:
            term_joint_names = list(config["joint_names"])
            self.term_joint_names = term_joint_names
            self.joint_indices = torch.tensor(
                [joint_names.index(jn) for jn in term_joint_names],
                device=device,
                dtype=torch.long,
            )
        else:
            self.term_joint_names = None
            self.joint_indices = None

        if "joint_pos_offsets" in config:
            self.joint_pos_offsets = torch.tensor(config["joint_pos_offsets"], device=device, dtype=torch.float32)
        else:
            self.joint_pos_offsets = None

        if "joint_vel_offsets" in config:
            self.joint_vel_offsets = torch.tensor(config["joint_vel_offsets"], device=device, dtype=torch.float32)
        else:
            self.joint_vel_offsets = None

        self.history = HistoryBuffer(self.obs_dim, self.history_length, device)

        # Filled by builder.
        self.compute_raw_fn: Callable[..., torch.Tensor] | None = None

    def output_dim(self) -> int:
        return self.obs_dim if self.history_length == 0 else self.obs_dim * self.history_length

    def compute(self, builder: AgileYamlObservationBuilder, sim_state: SimState) -> torch.Tensor:
        assert self.compute_raw_fn is not None, f"compute_raw_fn not set for term '{self.name}'"
        obs = self.compute_raw_fn(builder, self, sim_state)

        if self.clip is not None:
            obs = torch.clamp(obs, float(self.clip[0]), float(self.clip[1]))

        if self.scale is not None:
            if isinstance(self.scale, list):
                scale_tensor = torch.tensor(self.scale, device=self.device, dtype=torch.float32)
                obs = obs * scale_tensor
            else:
                obs = obs * float(self.scale)

        if self.history_length == 0:
            return obs

        self.history.push(obs)
        return self.history.get(flatten=self.flatten_history)

    def reset(self):
        self.history.reset()


class AgileYamlObservationBuilder:
    """Compute the AGILE policy observation tensor from a ``SimState`` + command vector.

    Owns one :class:`YamlObservationTerm` per ``observations.policy`` entry in the
    IO-descriptor YAML, plus the cached ``last_action`` and the latest command vector.
    """

    def __init__(self, yaml_dict: dict, joint_names: list[str], device: torch.device):
        self.device = device
        self.joint_names = joint_names

        obs_policy = yaml_dict["observations"]["policy"]

        # Cached state used by the per-term compute_fns.
        self._last_action: torch.Tensor | None = None
        self._command: torch.Tensor | None = None

        self.terms: list[YamlObservationTerm] = []
        for term_cfg in obs_policy:
            term = YamlObservationTerm(term_cfg, joint_names, device)
            term.compute_raw_fn = _COMPUTE_FNS[term.name]
            self.terms.append(term)

        # Sanity-check dimensions against YAML's declared shapes (incl. history).
        for term, term_cfg in zip(self.terms, obs_policy, strict=True):
            decl_shape = term_cfg.get("shape", [term.obs_dim])
            base_dim = decl_shape[0] if isinstance(decl_shape, list) else int(decl_shape)
            expected = base_dim * term.history_length if term.history_length > 0 else base_dim
            if expected != term.output_dim():
                raise ValueError(
                    f"YAML/AGILE obs term dim mismatch for '{term.name}': expected {expected}, got {term.output_dim()}"
                )

        self.total_obs_dim = sum(t.output_dim() for t in self.terms)

    # --- public API -------------------------------------------------------

    def set_last_action(self, action: torch.Tensor):
        self._last_action = action.detach().to(self.device).float().flatten()

    def set_command(self, command: torch.Tensor):
        self._command = command.detach().to(self.device).float().flatten()

    def compute(
        self,
        sim_state: SimState,
        command: torch.Tensor | None = None,
        last_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if command is not None:
            self.set_command(command)
        if last_action is not None:
            self.set_last_action(last_action)

        chunks = [term.compute(self, sim_state).flatten() for term in self.terms]
        return torch.cat(chunks, dim=0)

    def reset(self):
        for term in self.terms:
            term.reset()
        self._last_action = None
        self._command = None


# --- per-term raw compute functions --------------------------------------


def _raw_generated_commands(
    builder: AgileYamlObservationBuilder, term: YamlObservationTerm, sim_state: SimState
) -> torch.Tensor:
    cmd = builder._command
    if cmd is None:
        cmd = torch.zeros(term.obs_dim, device=builder.device, dtype=torch.float32)
    if cmd.shape[0] < term.obs_dim:
        pad = torch.zeros(term.obs_dim - cmd.shape[0], device=builder.device, dtype=torch.float32)
        cmd = torch.cat([cmd, pad], dim=0)
    elif cmd.shape[0] > term.obs_dim:
        cmd = cmd[: term.obs_dim]
    return cmd.float()


def _raw_base_ang_vel(builder, term, sim_state: SimState) -> torch.Tensor:
    return sim_state.root_ang_vel.float()


def _raw_base_lin_vel(builder, term, sim_state: SimState) -> torch.Tensor:
    if sim_state.root_lin_vel is None:
        return torch.zeros(3, device=builder.device, dtype=torch.float32)
    return sim_state.root_lin_vel.float()


def _raw_projected_gravity(builder, term, sim_state: SimState) -> torch.Tensor:
    gravity_world = torch.tensor([0.0, 0.0, -1.0], device=builder.device, dtype=torch.float32)
    return quat_rotate_inverse_wxyz(sim_state.root_quat.float(), gravity_world)


def _raw_joint_pos_rel(builder, term, sim_state: SimState) -> torch.Tensor:
    jp = sim_state.joint_pos[term.joint_indices].float()
    if term.joint_pos_offsets is not None:
        return jp - term.joint_pos_offsets
    return jp


def _raw_joint_pos(builder, term, sim_state: SimState) -> torch.Tensor:
    return sim_state.joint_pos[term.joint_indices].float()


def _raw_joint_vel(builder, term, sim_state: SimState) -> torch.Tensor:
    return sim_state.joint_vel[term.joint_indices].float()


def _raw_joint_vel_rel(builder, term, sim_state: SimState) -> torch.Tensor:
    jv = sim_state.joint_vel[term.joint_indices].float()
    if term.joint_vel_offsets is not None:
        return jv - term.joint_vel_offsets
    return jv


def _raw_last_action(builder, term, sim_state: SimState) -> torch.Tensor:
    if builder._last_action is None:
        return torch.zeros(term.obs_dim, device=builder.device, dtype=torch.float32)
    return builder._last_action.float()


def _raw_zero_padding(builder, term, sim_state: SimState) -> torch.Tensor:
    return torch.zeros(term.obs_dim, device=builder.device, dtype=torch.float32)


_COMPUTE_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "generated_commands": _raw_generated_commands,
    "velocity_and_height_command": _raw_generated_commands,
    "locomotion_command": _raw_generated_commands,
    "navigation_command": _raw_generated_commands,
    "base_ang_vel": _raw_base_ang_vel,
    "base_lin_vel": _raw_base_lin_vel,
    "projected_gravity": _raw_projected_gravity,
    "joint_pos_rel": _raw_joint_pos_rel,
    "joint_pos": _raw_joint_pos,
    "joint_vel": _raw_joint_vel,
    "joint_vel_rel": _raw_joint_vel_rel,
    "last_action": _raw_last_action,
    "zero_padding": _raw_zero_padding,
}


# ---------------------------------------------------------------------------
# Policy runners — TorchScript MLP / RNN, ONNX, raw RSL-RL checkpoint
# ---------------------------------------------------------------------------


class _TorchScriptRunner:
    """TorchScript wrapper.

    AGILE's ``_TorchPolicyExporter`` has a single-arg ``forward(x)`` regardless
    of MLP / RNN; for RNN policies the hidden state lives as registered buffers
    (``hidden_state`` and optionally ``cell_state``) inside the scripted module.

    Because the AGILE exporter assigns ``self.reset = policy.reset`` as a plain
    instance attribute (not ``@torch.jit.export``-decorated), the JIT module does
    not expose a callable ``reset`` for RNN policies. We therefore clear the
    buffers directly when present.
    """

    def __init__(self, model: Any, device: torch.device):
        self.model = model
        self.device = device
        # Detect AGILE-style internal RNN buffers.
        self._hidden_state = getattr(self.model, "hidden_state", None)
        self._cell_state = getattr(self.model, "cell_state", None)
        try:
            jit_reset = getattr(self.model, "reset", None)
        except (AttributeError, RuntimeError):
            jit_reset = None
        self._jit_reset = jit_reset if callable(jit_reset) else None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(obs)

    def reset(self):
        if self._jit_reset is not None:
            try:
                self._jit_reset()
                return
            except Exception:
                pass
        if isinstance(self._hidden_state, torch.Tensor):
            self._hidden_state.zero_()
        if isinstance(self._cell_state, torch.Tensor):
            self._cell_state.zero_()


class _ExternalHiddenRnnRunner:
    """TorchScript RNN wrapper for the ``(obs, hidden) → (action, hidden)`` style.

    Used when the user explicitly sets ``rnn_hidden_shape`` AND the loaded module
    accepts hidden state as a second argument (i.e. not the AGILE exporter style).
    """

    def __init__(self, model: Any, hidden_shape: list[int], device: torch.device):
        self.model = model
        self.device = device
        self.hidden_shape = list(hidden_shape)
        self.hidden = torch.zeros(*self.hidden_shape, device=device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs_batched = obs.unsqueeze(0)
            output, self.hidden = self.model(obs_batched, self.hidden)
            return output.squeeze(0)

    def reset(self):
        self.hidden = torch.zeros(*self.hidden_shape, device=self.device)


def _infer_torchscript_input_dim(model: Any) -> int | None:
    """Best-effort probe of a scripted AGILE policy's input dim for runner detection."""
    # RNN: the AGILE exporter keeps ``self.rnn`` as a module attribute.
    try:
        rnn = getattr(model, "rnn", None)
        if rnn is not None and hasattr(rnn, "input_size"):
            return int(rnn.input_size)
    except Exception:
        pass
    # MLP: ``actor`` is a Sequential whose first Linear has ``in_features``.
    try:
        actor = getattr(model, "actor", None)
        if actor is not None:
            for layer in actor.children():
                if hasattr(layer, "in_features"):
                    return int(layer.in_features)
    except Exception:
        pass
    return None


class _OnnxRunner:
    """ONNX runner.

    Handles two AGILE export shapes:

      - MLP (one input ``obs``, one output ``actions``).
      - LSTM (three inputs ``obs / h_in / c_in``, three outputs
        ``actions / h_out / c_out``). Hidden + cell state are tracked on the
        runner and reset to zeros on :meth:`reset`.

    GRU exports (``obs + h_in → actions + h_out``) are also supported.
    """

    def __init__(self, checkpoint_path: Path, device: torch.device):
        try:
            import numpy as _np  # local import (numpy already at module top)
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to load ONNX AGILE policies. Install with `pip install onnxruntime`."
            ) from e
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        self._np = _np
        self.session = ort.InferenceSession(str(checkpoint_path), providers=providers)
        self.device = device

        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())
        self.input_names = [i.name for i in inputs]
        self.output_names = [o.name for o in outputs]

        # Decide architecture from input count.
        self._is_lstm = len(inputs) == 3
        self._is_gru = len(inputs) == 2
        self._is_mlp = len(inputs) == 1

        if self._is_mlp:
            self._h_shape = None
            self._c_shape = None
            self._hidden = None
            self._cell = None
        elif self._is_gru:
            self._h_shape = tuple(inputs[1].shape)
            self._c_shape = None
            self._hidden = _np.zeros(self._h_shape, dtype=_np.float32)
            self._cell = None
        elif self._is_lstm:
            self._h_shape = tuple(inputs[1].shape)
            self._c_shape = tuple(inputs[2].shape)
            self._hidden = _np.zeros(self._h_shape, dtype=_np.float32)
            self._cell = _np.zeros(self._c_shape, dtype=_np.float32)
        else:
            raise ValueError(
                f"Unsupported ONNX input arity ({len(inputs)}) for AGILE policy at "
                f"{checkpoint_path}. Expected 1 (MLP), 2 (GRU), or 3 (LSTM)."
            )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        np = self._np
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]

        if self._is_mlp:
            outputs = self.session.run([self.output_names[0]], {self.input_names[0]: obs_np})
            return torch.from_numpy(outputs[0]).to(self.device).squeeze(0)

        if self._is_gru:
            outs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: obs_np,
                    self.input_names[1]: self._hidden,
                },
            )
            actions, h = outs[0], outs[1]
            self._hidden = h
            return torch.from_numpy(actions).to(self.device).squeeze(0)

        # LSTM
        outs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: obs_np,
                self.input_names[1]: self._hidden,
                self.input_names[2]: self._cell,
            },
        )
        actions, h, c = outs[0], outs[1], outs[2]
        self._hidden = h
        self._cell = c
        return torch.from_numpy(actions).to(self.device).squeeze(0)

    def reset(self):
        np = self._np
        if self._hidden is not None:
            self._hidden = np.zeros(self._h_shape, dtype=np.float32)
        if self._cell is not None:
            self._cell = np.zeros(self._c_shape, dtype=np.float32)


# ---- raw RSL-RL checkpoint inference (architecture inferred from state dict) ----

_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "crelu": nn.CELU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


def _resolve_activation(name: str) -> nn.Module:
    cls = _ACTIVATION_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown activation '{name}'. Supported: {list(_ACTIVATION_MAP)}")
    return cls()


def _infer_architecture(model_state: dict[str, torch.Tensor]) -> dict:
    """Mirror of AGILE's ``_infer_architecture`` — kept local to avoid runtime AGILE dep."""
    if any(k.startswith("actor.layers.") for k in model_state):
        actor_prefix = "actor.layers."
    elif any(k.startswith("student.layers.") for k in model_state):
        actor_prefix = "student.layers."
    else:
        raise ValueError(
            "Cannot find actor weights in checkpoint. "
            f"Expected keys starting with 'actor.layers.' or 'student.layers.', "
            f"got: {list(model_state)[:10]}"
        )

    weight_keys = sorted(
        [k for k in model_state if k.startswith(actor_prefix) and k.endswith(".weight")],
        key=lambda k: int(k.replace(actor_prefix, "").split(".")[0]),
    )
    if not weight_keys:
        raise ValueError(f"No actor weight keys found with prefix '{actor_prefix}'")

    actor_input_dim = model_state[weight_keys[0]].shape[1]
    actor_hidden_dims = [model_state[k].shape[0] for k in weight_keys[:-1]]
    actor_output_dim = model_state[weight_keys[-1]].shape[0]

    if "std" in model_state:
        noise_std_type = "scalar"
        num_actions = model_state["std"].shape[0]
    elif "log_std" in model_state:
        noise_std_type = "log"
        num_actions = model_state["log_std"].shape[0]
    else:
        noise_std_type = "pred"
        num_actions = actor_output_dim // 2

    is_recurrent = any(k.startswith("memory_a.rnn.") for k in model_state)
    rnn_type = None
    rnn_input_dim = 0
    rnn_hidden_dim = 0
    rnn_num_layers = 0
    if is_recurrent:
        wih = model_state.get("memory_a.rnn.weight_ih_l0")
        whh = model_state.get("memory_a.rnn.weight_hh_l0")
        if wih is None or whh is None:
            raise ValueError("Recurrent policy detected but missing memory_a.rnn weights")
        rnn_input_dim = wih.shape[1]
        rnn_hidden_dim = whh.shape[1]
        gate_size = wih.shape[0]
        if gate_size == 3 * rnn_hidden_dim:
            rnn_type = "gru"
        elif gate_size == 4 * rnn_hidden_dim:
            rnn_type = "lstm"
        else:
            raise ValueError(f"Cannot determine RNN type from gate size {gate_size}")
        while f"memory_a.rnn.weight_ih_l{rnn_num_layers}" in model_state:
            rnn_num_layers += 1

    return {
        "actor_prefix": actor_prefix,
        "actor_input_dim": actor_input_dim,
        "actor_hidden_dims": actor_hidden_dims,
        "actor_output_dim": actor_output_dim,
        "num_actions": num_actions,
        "noise_std_type": noise_std_type,
        "is_recurrent": is_recurrent,
        "rnn_type": rnn_type,
        "rnn_input_dim": rnn_input_dim,
        "rnn_hidden_dim": rnn_hidden_dim,
        "rnn_num_layers": rnn_num_layers,
    }


class _SimpleNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", mean.squeeze(0))
        self.register_buffer("_std", std.squeeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class _CheckpointInferenceModel(nn.Module):
    """Reconstructs the actor (+ optional normalizer / RNN) from a raw RSL-RL ckpt."""

    def __init__(self, arch: dict, activation: nn.Module):
        super().__init__()
        self.noise_std_type = arch["noise_std_type"]
        self.num_actions = arch["num_actions"]
        self.is_recurrent = arch["is_recurrent"]

        layers: list[nn.Module] = []
        in_dim = arch["actor_input_dim"]
        for h_dim in arch["actor_hidden_dims"]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation)
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, arch["actor_output_dim"]))
        self.actor = nn.Sequential(*layers)

        self.normalizer: nn.Module = nn.Identity()

        self._rnn_type = arch["rnn_type"]
        if self.is_recurrent:
            rnn_cls = nn.LSTM if arch["rnn_type"] == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=arch["rnn_input_dim"],
                hidden_size=arch["rnn_hidden_dim"],
                num_layers=arch["rnn_num_layers"],
                batch_first=False,
            )
            self.register_buffer("hidden_state", torch.zeros(arch["rnn_num_layers"], arch["rnn_hidden_dim"]))
            if arch["rnn_type"] == "lstm":
                self.register_buffer("cell_state", torch.zeros(arch["rnn_num_layers"], arch["rnn_hidden_dim"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        if self.is_recurrent:
            x = x.unsqueeze(0).unsqueeze(0)
            if self._rnn_type == "lstm":
                hidden = (self.hidden_state.unsqueeze(1), self.cell_state.unsqueeze(1))
                x, (h, c) = self.rnn(x, hidden)
                self.hidden_state[:] = h.squeeze(1)
                self.cell_state[:] = c.squeeze(1)
            else:
                hidden = self.hidden_state.unsqueeze(1)
                x, h = self.rnn(x, hidden)
                self.hidden_state[:] = h.squeeze(1)
            x = x.squeeze(0).squeeze(0)
        out = self.actor(x)
        if self.noise_std_type == "pred":
            out = out[: self.num_actions]
        return out

    def reset_hidden(self):
        if self.is_recurrent:
            self.hidden_state.zero_()
            if self._rnn_type == "lstm":
                self.cell_state.zero_()


class _CheckpointRunner:
    def __init__(self, model: _CheckpointInferenceModel, device: torch.device):
        self.model = model
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(obs)

    def reset(self):
        self.model.reset_hidden()


# ---------------------------------------------------------------------------
# Public runner factory
# ---------------------------------------------------------------------------


class AgilePolicyRunner:
    """Dispatches to the correct underlying runner for a given checkpoint file."""

    def __init__(self, runner: Any):
        self._runner = runner

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return self._runner(obs)

    def reset(self):
        self._runner.reset()

    # --- factory -----------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        rnn_hidden_shape: list[int] | None = None,
        activation: str = "elu",
    ) -> AgilePolicyRunner:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"AGILE checkpoint not found: {checkpoint_path}")

        if checkpoint_path.suffix == ".onnx":
            return cls(_OnnxRunner(checkpoint_path, device))

        # Try TorchScript first; fall back to raw RSL-RL checkpoint.
        try:
            model = torch.jit.load(str(checkpoint_path), map_location=device)
            model.eval()
        except RuntimeError:
            return cls._from_raw_checkpoint(checkpoint_path, device, activation)

        # AGILE's standard exporter (``_TorchPolicyExporter``) keeps the RNN
        # hidden state as a buffer inside the scripted module and exposes a
        # single-arg ``forward(x)`` — regardless of whether the policy is MLP
        # or recurrent. ``_TorchScriptRunner`` covers both cases and calls
        # ``model.reset()`` if the module exposes one. The ``rnn_hidden_shape``
        # cfg field is retained as documentation metadata; we only use the
        # legacy ``(obs, hidden)`` runner if the module's forward signature
        # actually rejects the single-arg call below.
        runner = _TorchScriptRunner(model, device)
        if rnn_hidden_shape is not None:
            try:
                # Probe with a zero obs the size of the AGILE exporter's
                # declared first-layer input (best effort — actor[0] for MLP,
                # or rnn.input_size for RNN). If the probe raises a signature
                # error, fall back to the external-hidden style.
                input_dim = _infer_torchscript_input_dim(model)
                if input_dim is not None:
                    runner(torch.zeros(input_dim, device=device))
            except (RuntimeError, TypeError):
                return cls(_ExternalHiddenRnnRunner(model, rnn_hidden_shape, device))
        return cls(runner)

    @classmethod
    def _from_raw_checkpoint(cls, checkpoint_path: Path, device: torch.device, activation: str) -> AgilePolicyRunner:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"File {checkpoint_path} is not a TorchScript module and lacks "
                f"'model_state_dict' — cannot infer architecture. "
                f"Keys: {list(checkpoint.keys())[:10]}"
            )

        model_state = checkpoint["model_state_dict"]
        arch = _infer_architecture(model_state)
        model = _CheckpointInferenceModel(arch, _resolve_activation(activation))

        actor_prefix = arch["actor_prefix"]
        actor_state = {k.replace(actor_prefix, ""): v for k, v in model_state.items() if k.startswith(actor_prefix)}
        model.actor.load_state_dict(actor_state)

        if arch["is_recurrent"]:
            rnn_state = {
                k.replace("memory_a.rnn.", ""): v for k, v in model_state.items() if k.startswith("memory_a.rnn.")
            }
            model.rnn.load_state_dict(rnn_state)

        if "obs_norm_state_dict" in checkpoint:
            norm_state = checkpoint["obs_norm_state_dict"]
            model.normalizer = _SimpleNormalizer(
                mean=norm_state["_mean"],
                std=norm_state["_std"],
                eps=1e-2,
            )

        model.eval()
        model.to(device)

        arch_type = arch["rnn_type"].upper() if arch["is_recurrent"] else "MLP"
        logger.info(
            f"[AgilePolicyRunner] Loaded raw RSL-RL checkpoint ({arch_type}, "
            f"hidden={arch['actor_hidden_dims']}, num_actions={arch['num_actions']})"
        )

        return cls(_CheckpointRunner(model, device))
