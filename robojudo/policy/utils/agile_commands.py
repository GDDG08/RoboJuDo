"""Polymorphic command-source plug-ins for :class:`AgilePolicy`.

The policy class itself is command-shape-agnostic. All velocity / height / motion
semantics live behind the :class:`AgileCommandSource` ABC; AGILE policies of any
command flavor are deployed by composing the right :class:`AgileCommandSourceCfg`
subclass with the YAML+checkpoint cfg.

Three reference sources ship in this file:

  - :class:`VelocityHeightCommandSource` — joystick / keyboard / unitree-pad
  - :class:`MotionFileCommandSource` — ``.npz`` motion playback for tracking
  - :class:`ZeroCommandSource` — all-zeros vector for CI / smoke runs
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robojudo.policy.policy_cfgs import (
    AgileCommandSourceCfg,
    MotionFileCommandSourceCfg,
    VelocityHeightCommandSourceCfg,
    ZeroCommandSourceCfg,
)
from robojudo.utils.module_registry import Registry
from robojudo.utils.util_func import command_remap

logger = logging.getLogger(__name__)


@dataclass
class CommandPayload:
    """What an :class:`AgileCommandSource` returns each step.

    - ``vector``: what gets fed into the policy as the YAML ``generated_commands``
      term (or related command-shaped term). Length must match the YAML command
      term's ``shape[0]``, or ``None`` for command-less policies.
    - ``extras``: world-frame poses / scalars used by tracking-style obs terms
      (e.g. ``motion_anchor_pos_w``, ``motion_anchor_quat_w``).
    - ``ui``: arbitrary debugging payload (drawn arrows, raw joystick axes, …).
    """

    vector: np.ndarray | None
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)


class AgileCommandSource(ABC):
    """Single point of policy-shape decoupling for :class:`AgilePolicy`.

    Implementations are free to read joystick/keyboard ctrl_data, play motion
    files, accept ROS messages, etc. They are the ONLY place that knows the
    semantics of the command vector.
    """

    cfg: AgileCommandSourceCfg

    def __init__(
        self,
        cfg: AgileCommandSourceCfg,
        *,
        yaml: dict,
        command_dim: int | None,
        policy_dt: float,
        device: str,
    ):
        self.cfg = cfg
        self.yaml = yaml
        self.command_dim = command_dim
        self.policy_dt = policy_dt
        self.device = device

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get(self, env_data, ctrl_data, sim_state) -> CommandPayload: ...

    def post_step(self) -> None:
        """Advance any time-driven internal state. Called once per pipeline step
        from :meth:`AgilePolicy.post_step_callback`."""
        return


command_source_registry = Registry(
    package="robojudo.policy.utils.agile_commands",
    base_class=AgileCommandSource,
)


def build_command_source(
    cfg: AgileCommandSourceCfg,
    *,
    yaml: dict,
    command_dim: int | None,
    policy_dt: float,
    device: str,
) -> AgileCommandSource:
    """Resolve ``cfg.source_type`` against the registry and construct a source."""
    cls = command_source_registry.get(cfg.source_type)
    return cls(cfg, yaml=yaml, command_dim=command_dim, policy_dt=policy_dt, device=device)


# ---------------------------------------------------------------------------
# VelocityHeightCommandSource — joystick / keyboard / unitree-pad
# ---------------------------------------------------------------------------


@command_source_registry.register
class VelocityHeightCommandSource(AgileCommandSource):
    """Maps the active controller's stick / WASD input to ``[vx, vy, wz, height]``,
    truncated/padded to the YAML command term's declared length.

    Mirrors the behavior previously hard-coded inside ``AgilePolicy._get_command``.
    """

    cfg: VelocityHeightCommandSourceCfg

    def reset(self) -> None:
        # Stateless.
        return

    def get(self, env_data, ctrl_data, sim_state) -> CommandPayload:
        defaults = self.cfg.defaults
        vx = float(defaults.get("linear_x", 0.0))
        vy = float(defaults.get("linear_y", 0.0))
        wz = float(defaults.get("angular_z", 0.0))
        height = float(defaults.get("height", 0.72))

        remap = self.cfg.command_remap

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
                raw_vx = 0.0
                raw_vy = 0.0
                raw_wz = 0.0
                has_keyboard_input = False
                for event in events:
                    if event.get("type") != "keyboard":
                        continue
                    has_keyboard_input = True
                    pressed = float(event["pressed"])
                    name = event["name"]
                    if name == "w":
                        raw_vx = pressed
                    elif name == "s":
                        raw_vx = -pressed
                    elif name == "a":
                        raw_vy = -pressed
                    elif name == "d":
                        raw_vy = pressed
                    elif name == "e":
                        raw_wz = pressed
                    elif name == "q":
                        raw_wz = -pressed
                if has_keyboard_input:
                    if len(remap) >= 3:
                        vx = float(command_remap(raw_vx, remap[0]))
                        vy = float(command_remap(raw_vy, remap[1]))
                        wz = float(command_remap(raw_wz, remap[2]))
                    else:
                        vx, vy, wz = raw_vx, raw_vy, raw_wz
                break

        full = np.array([vx, vy, wz, height], dtype=np.float32)
        cd = self.command_dim
        if cd is None or cd <= 0:
            vector = full
        elif cd >= 4:
            vector = np.zeros(cd, dtype=np.float32)
            vector[:4] = full
        else:
            vector = full[:cd]
        return CommandPayload(vector=vector, ui={"vx": vx, "vy": vy, "wz": wz, "height": height})


# ---------------------------------------------------------------------------
# ZeroCommandSource — CI / smoke
# ---------------------------------------------------------------------------


@command_source_registry.register
class ZeroCommandSource(AgileCommandSource):
    """Emits an all-zeros command vector. Length comes from ``cfg.command_dim`` if
    set; otherwise from the YAML's command term."""

    cfg: ZeroCommandSourceCfg

    def reset(self) -> None:
        return

    def get(self, env_data, ctrl_data, sim_state) -> CommandPayload:
        cd = self.cfg.command_dim if self.cfg.command_dim is not None else self.command_dim
        if cd is None or cd <= 0:
            return CommandPayload(vector=None)
        return CommandPayload(vector=np.zeros(int(cd), dtype=np.float32))


# ---------------------------------------------------------------------------
# MotionFileCommandSource — .npz playback for tracking policies
# ---------------------------------------------------------------------------


@command_source_registry.register
class MotionFileCommandSource(AgileCommandSource):
    """Plays back an ``.npz`` motion clip.

    On each :meth:`get`:
      - ``vector`` = ``concat([joint_pos[t], joint_vel[t]])`` reordered into the
        YAML's articulation joint order if ``motion_joint_names`` is given.
      - ``extras["motion_anchor_pos_w"]`` and ``extras["motion_anchor_quat_w"]``
        (``[w, x, y, z]``) are taken from the motion's anchor body at frame ``t``.

    :meth:`post_step` advances the playback timestep (looping if configured).
    :meth:`reset` rewinds to frame 0.

    The ``.npz`` file is expected to expose at least ``joint_pos`` and
    ``joint_vel`` arrays (shape ``(T, J)``); ``body_pos_w`` and ``body_quat_w``
    (shape ``(T, B, 3)`` / ``(T, B, 4)``) are required for tracking obs terms.
    Quaternions are assumed to be ``[w, x, y, z]`` (matching AGILE convention).
    """

    cfg: MotionFileCommandSourceCfg

    def __init__(
        self,
        cfg: MotionFileCommandSourceCfg,
        *,
        yaml: dict,
        command_dim: int | None,
        policy_dt: float,
        device: str,
    ):
        super().__init__(cfg, yaml=yaml, command_dim=command_dim, policy_dt=policy_dt, device=device)

        path = Path(cfg.motion_path)
        if not path.exists():
            raise FileNotFoundError(f"MotionFileCommandSource: motion file not found: {path}")
        data = np.load(str(path), allow_pickle=False)
        self._joint_pos: np.ndarray = np.asarray(data["joint_pos"], dtype=np.float32)
        self._joint_vel: np.ndarray = np.asarray(data["joint_vel"], dtype=np.float32)
        self._body_pos_w: np.ndarray | None = (
            np.asarray(data["body_pos_w"], dtype=np.float32) if "body_pos_w" in data.files else None
        )
        self._body_quat_w: np.ndarray | None = (
            np.asarray(data["body_quat_w"], dtype=np.float32) if "body_quat_w" in data.files else None
        )
        self._num_frames = int(self._joint_pos.shape[0])

        # Resolve motion_body_names → anchor body index.
        motion_body_names = list(cfg.motion_body_names)
        if not motion_body_names:
            # Fall back to YAML's motion_tracking block if present.
            mt = (yaml or {}).get("motion_tracking", {}) or {}
            motion_body_names = list(mt.get("motion_body_names", []) or [])
        if not motion_body_names:
            self._anchor_idx: int | None = None
            logger.warning(
                "MotionFileCommandSource: no motion_body_names provided; motion_anchor_* extras will be omitted."
            )
        else:
            if cfg.anchor_body_name not in motion_body_names:
                raise ValueError(
                    f"MotionFileCommandSource: anchor_body_name={cfg.anchor_body_name!r} "
                    f"not in motion_body_names={motion_body_names}"
                )
            self._anchor_idx = motion_body_names.index(cfg.anchor_body_name)

        # Joint reorder from motion order → YAML articulation order.
        target_joint_names = list((yaml or {}).get("articulations", {}).get("robot", {}).get("joint_names", []))
        motion_joint_names = cfg.motion_joint_names
        if motion_joint_names is None:
            mt = (yaml or {}).get("motion_tracking", {}) or {}
            motion_joint_names = mt.get("motion_joint_names")
        if motion_joint_names is not None and target_joint_names:
            try:
                self._joint_remap_idx = np.asarray(
                    [list(motion_joint_names).index(j) for j in target_joint_names],
                    dtype=np.int64,
                )
            except ValueError as e:
                raise ValueError(
                    f"MotionFileCommandSource: joint remap failed; YAML joint not in "
                    f"motion_joint_names. Underlying error: {e}"
                ) from e
        else:
            self._joint_remap_idx = None

        self._timestep: int = 0

    # --- API ----------------------------------------------------------

    def reset(self) -> None:
        self._timestep = 0

    def _current_frame(self) -> int:
        if self.cfg.loop:
            return int(self._timestep % self._num_frames)
        return int(min(self._timestep, self._num_frames - 1))

    def get(self, env_data, ctrl_data, sim_state) -> CommandPayload:
        t = self._current_frame()
        jp = self._joint_pos[t]
        jv = self._joint_vel[t]
        if self._joint_remap_idx is not None:
            jp = jp[self._joint_remap_idx]
            jv = jv[self._joint_remap_idx]
        vector = np.concatenate([jp, jv], axis=0).astype(np.float32)

        extras: dict[str, np.ndarray] = {}
        if self._anchor_idx is not None and self._body_pos_w is not None and self._body_quat_w is not None:
            extras["motion_anchor_pos_w"] = self._body_pos_w[t, self._anchor_idx].copy()
            extras["motion_anchor_quat_w"] = self._body_quat_w[t, self._anchor_idx].copy()

        ui = {"timestep": t, "num_frames": self._num_frames}
        return CommandPayload(vector=vector, extras=extras, ui=ui)

    def post_step(self) -> None:
        self._timestep += int(round(self.cfg.play_speed))
