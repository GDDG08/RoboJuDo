# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Motion playback and heading-alignment utilities.

Vendored from the ProtoMotions ``deployment`` module so that RoboJuDo can
run inference without requiring the ProtoMotions source tree.

Quaternion convention: **xyzw** throughout (ProtoMotions common format).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "MotionPlayer",
    "compute_yaw_offset_np",
    "apply_heading_offset_np",
    "_extract_yaw_quat_np",
]

# ---------------------------------------------------------------------------
# Quaternion helpers (pure NumPy)
# ---------------------------------------------------------------------------


def _extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract the yaw-only quaternion from a full orientation (xyzw)."""
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _quat_mul_np(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (pure NumPy)."""
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def _quat_conjugate_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats) of an xyzw quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def compute_yaw_offset_np(
    robot_quat_xyzw: np.ndarray,
    motion_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a yaw-only heading offset between robot and motion frames.

    Returns a quaternion ``R_offset`` such that
    ``R_offset * motion_body_rot`` is in the robot's heading frame.
    """
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(
    offset_quat_xyzw: np.ndarray,
    body_rots_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a heading offset to an array of body rotations.

    Computes ``offset * body_rot`` for every quaternion in the array.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = _quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------

_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")


def _is_cache_file(data: dict) -> bool:
    """Return True if *data* looks like a pre-resampled cache."""
    return "control_dt" in data and "body_rot" in data


class MotionPlayer:
    """Lightweight player for a single motion clip at a fixed control rate.

    Accepts three input formats (auto-detected):

    1. Single ``.motion`` file -- RobotState dict with ``fps``, ``dof_pos``, etc.
    2. Packaged ``.pt`` library -- multi-motion with ``length_starts``, ``gts``, etc.
       Requires ``motion_index``.
    3. Pre-resampled cache -- written by :meth:`cache_to_file`.

    Formats 1 and 2 require ``protomotions`` for interpolation on first load.
    Format 3 (cached) is pure NumPy -- no external dependencies.
    """

    def __init__(
        self,
        motion_file: str,
        motion_index: int = 0,
        control_dt: float = 0.02,
    ):
        import torch

        self._torch = torch
        motion_file = str(motion_file)
        data = torch.load(motion_file, map_location="cpu", weights_only=False)

        if _is_cache_file(data):
            self._load_cache(data)
        else:
            self._load_raw(data, motion_index, control_dt)

    @property
    def total_frames(self) -> int:
        return self._num_frames

    @property
    def num_bodies(self) -> int:
        return self._body_rot.shape[1]

    @property
    def num_dofs(self) -> int:
        return self._dof_pos.shape[1]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return the motion state at *frame_idx* (clamped)."""
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos":      self._dof_pos[idx],
            "dof_vel":      self._dof_vel[idx],
            "body_rot":     self._body_rot[idx],
            "body_pos":     self._body_pos[idx],
            "body_vel":     self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Return stacked future motion states at ``frame_idx + offset``."""
        future_states = [
            self.get_state_at_frame(frame_idx + s) for s in step_indices
        ]
        return {
            key: np.stack([s[key] for s in future_states], axis=0)
            for key in _STATE_KEYS
        }

    def cache_to_file(self, output_path: str) -> None:
        """Write a pre-resampled cache file at the current control rate."""
        import torch

        cache = {
            "dof_pos":      self._dof_pos,
            "dof_vel":      self._dof_vel,
            "body_rot":     self._body_rot,
            "body_pos":     self._body_pos,
            "body_vel":     self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt":   self._control_dt,
            "num_frames":   self._num_frames,
        }
        torch.save(cache, output_path)
        print(
            f"[MotionPlayer] Cached {self._num_frames} frames @ "
            f"{1.0 / self._control_dt:.0f} Hz -> {output_path}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self, data: dict) -> None:
        self._dof_pos      = np.asarray(data["dof_pos"],      dtype=np.float32)
        self._dof_vel      = np.asarray(data["dof_vel"],      dtype=np.float32)
        self._body_rot     = np.asarray(data["body_rot"],     dtype=np.float32)
        self._body_pos     = np.asarray(data["body_pos"],     dtype=np.float32)
        self._body_vel     = np.asarray(data["body_vel"],     dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt   = float(data["control_dt"])
        self._num_frames   = int(data["num_frames"])
        self._cached = True
        print(
            f"[MotionPlayer] Loaded cache: {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz"
        )

    def _load_raw(self, data: dict, motion_index: int, control_dt: float) -> None:
        """Load from a raw ProtoMotions motion file and resample."""
        self._control_dt = control_dt
        self._cached = False

        if "length_starts" in data:
            length_starts     = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all     = data["motion_dt"]

            start  = int(length_starts[motion_index].item())
            nf     = int(motion_num_frames[motion_index].item())
            src_dt = float(motion_dt_all[motion_index].item())

            gts  = np.asarray(data["gts"][start:start + nf],  dtype=np.float32)
            grs  = np.asarray(data["grs"][start:start + nf],  dtype=np.float32)
            gvs  = np.asarray(data["gvs"][start:start + nf],  dtype=np.float32)
            gavs = np.asarray(data["gavs"][start:start + nf], dtype=np.float32)
            dps  = np.asarray(data["dps"][start:start + nf],  dtype=np.float32)
            dvs  = np.asarray(data["dvs"][start:start + nf],  dtype=np.float32)
            motion_length = src_dt * (nf - 1)

        elif "rigid_body_pos" in data:
            fps    = float(data["fps"])
            src_dt = 1.0 / fps

            gts  = np.asarray(data["rigid_body_pos"],     dtype=np.float32)
            grs  = np.asarray(data["rigid_body_rot"],      dtype=np.float32)
            gvs  = np.asarray(data["rigid_body_vel"],      dtype=np.float32)
            gavs = np.asarray(data["rigid_body_ang_vel"],  dtype=np.float32)
            dps  = np.asarray(data["dof_pos"],             dtype=np.float32)
            dvs  = np.asarray(data["dof_vel"],             dtype=np.float32)
            nf   = gts.shape[0]
            motion_length = src_dt * (nf - 1)
        else:
            raise ValueError(
                "Unrecognised raw motion format.  Expected either:\n"
                "  - packaged library: keys 'length_starts', 'gts', 'grs', ...\n"
                "  - single-motion:   keys 'rigid_body_pos', 'fps', 'dof_pos', ..."
            )

        # Resample to control rate (lerp for positions, slerp for quaternions)
        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)
        ctrl_times = np.linspace(0.0, motion_length, num_ctrl_frames)

        phase = np.clip(ctrl_times / motion_length, 0.0, 1.0)
        f0 = (phase * (nf - 1)).astype(np.int64)
        f1 = np.minimum(f0 + 1, nf - 1)
        blend = ((ctrl_times - f0 * src_dt) / src_dt).astype(np.float32)

        def _lerp(src):
            b = blend.reshape(-1, *([1] * (src.ndim - 1)))
            return ((1.0 - b) * src[f0] + b * src[f1]).astype(np.float32)

        def _slerp(src):
            q0, q1 = src[f0], src[f1]
            cos_half = np.sum(q0 * q1, axis=-1, keepdims=True)
            # Flip to shortest path
            neg = cos_half < 0
            q1 = np.where(neg, -q1, q1)
            cos_half = np.abs(cos_half)
            half_theta = np.arccos(np.clip(cos_half, -1.0, 1.0))
            sin_half = np.sqrt(np.maximum(1.0 - cos_half * cos_half, 0.0))
            b = blend.reshape(-1, *([1] * (q0.ndim - 1)))
            # Safe divide — degenerate cases handled by np.where below
            safe_sin = np.where(sin_half > 0, sin_half, 1.0)
            ratio_a = np.sin((1.0 - b) * half_theta) / safe_sin
            ratio_b = np.sin(b * half_theta) / safe_sin
            result = ratio_a * q0 + ratio_b * q1
            # Fallback: near-zero sin_half → linear blend; cos_half ≈ 1 → q0
            near_zero = np.abs(sin_half) < 0.001
            linear = 0.5 * q0 + 0.5 * q1
            result = np.where(near_zero, linear, result)
            identical = np.abs(cos_half) >= 1.0
            result = np.where(identical, q0, result)
            return result.astype(np.float32)

        self._body_pos     = _lerp(gts)
        self._body_rot     = _slerp(grs)
        self._body_vel     = _lerp(gvs)
        self._body_ang_vel = _lerp(gavs)
        self._dof_pos      = _lerp(dps)
        self._dof_vel      = _lerp(dvs)
        self._num_frames   = num_ctrl_frames

        print(
            f"[MotionPlayer] Loaded raw motion #{motion_index}: "
            f"{nf} source frames @ {1.0 / src_dt:.1f} Hz -> "
            f"{num_ctrl_frames} resampled frames @ {1.0 / control_dt:.0f} Hz"
        )
