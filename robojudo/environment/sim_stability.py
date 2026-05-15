"""Permissive sim-only stability monitor for headless policy testing.

This module is exercised exclusively by the heavyweight sim-test harness in
``tests/sim/``. It is intentionally separate from ``RlPipeline.safety_check``:

- The production ``safety_check`` is a strict guard intended for real-hardware
  runs. We don't want to relax it.
- The sim monitor is meant to answer a different question: did the policy
  catastrophically fail (fell, tipped over, slammed an arm into the floor)? We
  bias toward false negatives — a wobble is fine, a sustained fall is not.

To keep the sim-test path from leaking into production, the monitor is only
constructed when ``MujocoEnvCfg.sim_stability`` is set (the test harness sets
it; configs do not).
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SimStabilityCfg(BaseModel):
    """Per-robot permissive failure criteria for sim tests.

    All durations are wall-clock seconds (multiplied by control_dt internally),
    so behavior is timestep-independent.
    """

    base_body_name: str = "pelvis"
    """MuJoCo body name whose world-z height is used as the CoM proxy."""

    min_base_height: float = 0.35
    """Failure when the base body sits below this height in world z."""
    min_height_duration: float = 0.5
    """Sustained-below-threshold seconds before declaring failure."""

    max_tilt_rad: float = 1.4
    """Failure when the base tilt from world-z up exceeds this angle (~80deg by default)."""
    max_tilt_duration: float = 0.5

    illegal_contact_bodies: list[str] = Field(default_factory=list)
    """Body names whose contact with the world body indicates a fall (e.g. upper arms)."""
    illegal_contact_duration: float = 0.15

    grace_seconds: float = 0.5
    """No failures reported during this initial settle window — covers the dry-run / first PD step jolt."""


class SimStabilityMonitor:
    """Watches a MuJoCo MjData stream and flags failure once accumulated.

    Cheap to call every control step. Stateful: once ``failed`` is True, stays
    True until ``reset()``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cfg: SimStabilityCfg,
        control_dt: float,
    ):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.control_dt = control_dt

        self._base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_body_name)
        if self._base_body_id < 0:
            logger.warning("[SimStability] base_body '%s' not found", cfg.base_body_name)

        self._illegal_body_ids: set[int] = set()
        for name in cfg.illegal_contact_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._illegal_body_ids.add(bid)
            else:
                logger.warning("[SimStability] illegal_contact body '%s' not found", name)

        self.reset()

    def reset(self) -> None:
        self._elapsed = 0.0
        self._low_height_acc = 0.0
        self._tilt_acc = 0.0
        self._contact_acc = 0.0
        self.failed = False
        self.failure_reason: str | None = None
        self._last_base_height: float | None = None
        self._last_tilt_rad: float | None = None

    @property
    def in_grace(self) -> bool:
        return self._elapsed < self.cfg.grace_seconds

    def update(self) -> None:
        """Advance the monitor by one control step.

        Failures latch — once set, subsequent calls are no-ops.
        """
        if self.failed:
            return
        self._elapsed += self.control_dt

        # --- base height ---
        if self._base_body_id >= 0:
            base_z = float(self.data.xpos[self._base_body_id, 2])
            self._last_base_height = base_z
            if base_z < self.cfg.min_base_height:
                self._low_height_acc += self.control_dt
                if not self.in_grace and self._low_height_acc >= self.cfg.min_height_duration:
                    self._fail(
                        f"base '{self.cfg.base_body_name}' below "
                        f"{self.cfg.min_base_height:.2f} m for "
                        f"{self._low_height_acc:.2f}s (now {base_z:.2f})"
                    )
                    return
            else:
                self._low_height_acc = 0.0

        # --- tilt from upright ---
        # qpos[3:7] is base quaternion as [w, x, y, z] (MuJoCo).
        # body-frame z-component of the world up vector is 1 - 2*(x^2 + y^2).
        w, x, y, z = (float(v) for v in self.data.qpos[3:7])
        body_up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = float(np.arccos(np.clip(body_up_z, -1.0, 1.0)))
        self._last_tilt_rad = tilt
        if tilt > self.cfg.max_tilt_rad:
            self._tilt_acc += self.control_dt
            if not self.in_grace and self._tilt_acc >= self.cfg.max_tilt_duration:
                self._fail(
                    f"tilt {tilt:.2f} rad > {self.cfg.max_tilt_rad:.2f} for "
                    f"{self._tilt_acc:.2f}s"
                )
                return
        else:
            self._tilt_acc = 0.0

        # --- illegal upper-body contact with world ---
        if self._illegal_body_ids:
            had_illegal_contact = self._scan_contacts()
            if had_illegal_contact:
                self._contact_acc += self.control_dt
                if not self.in_grace and self._contact_acc >= self.cfg.illegal_contact_duration:
                    self._fail(
                        f"illegal upper-body contact for {self._contact_acc:.2f}s"
                    )
                    return
            else:
                self._contact_acc = 0.0

    def _scan_contacts(self) -> bool:
        ncon = int(self.data.ncon)
        if ncon <= 0:
            return False
        geom_bodyid = self.model.geom_bodyid
        for i in range(ncon):
            con = self.data.contact[i]
            b1 = int(geom_bodyid[con.geom1])
            b2 = int(geom_bodyid[con.geom2])
            # MuJoCo body id 0 is the world body (floor lives there).
            if (b1 in self._illegal_body_ids and b2 == 0) or (
                b2 in self._illegal_body_ids and b1 == 0
            ):
                return True
        return False

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.failure_reason = reason
        logger.error("[SimStability] FAILURE: %s", reason)

    def snapshot(self) -> dict:
        return {
            "elapsed": self._elapsed,
            "base_height": self._last_base_height,
            "tilt_rad": self._last_tilt_rad,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }
