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
        """Load from a raw ProtoMotions motion file and resample.

        This path requires ``protomotions`` to be importable (for interpolation
        utilities).  Use :meth:`cache_to_file` afterwards to create a cached
        version that needs no external dependencies.
        """
        import torch

        try:
            from protomotions.utils.motion_interpolation_utils import (
                calc_frame_blend,
                interpolate_pos,
                interpolate_quat,
            )
        except ImportError:
            raise ImportError(
                "Loading raw (non-cached) motion files requires the 'protomotions' "
                "package for interpolation.  Either:\n"
                "  1. Use a pre-cached .pt file (recommended), or\n"
                "  2. Install protomotions: pip install protomotions"
            ) from None

        self._control_dt = control_dt
        self._cached = False

        if "length_starts" in data:
            length_starts     = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all     = data["motion_dt"]

            start  = int(length_starts[motion_index].item())
            nf     = int(motion_num_frames[motion_index].item())
            end    = start + nf
            src_dt = float(motion_dt_all[motion_index].item())

            gts  = data["gts"][start:end]
            grs  = data["grs"][start:end]
            gvs  = data["gvs"][start:end]
            gavs = data["gavs"][start:end]
            dps  = data["dps"][start:end]
            dvs  = data["dvs"][start:end]
            motion_length = src_dt * (nf - 1)

        elif "rigid_body_pos" in data:
            fps    = float(data["fps"])
            src_dt = 1.0 / fps

            gts  = data["rigid_body_pos"]
            grs  = data["rigid_body_rot"]
            gvs  = data["rigid_body_vel"]
            gavs = data["rigid_body_ang_vel"]
            dps  = data["dof_pos"]
            dvs  = data["dof_vel"]
            nf   = gts.shape[0]
            motion_length = src_dt * (nf - 1)
        else:
            raise ValueError(
                "Unrecognised raw motion format.  Expected either:\n"
                "  - packaged library: keys 'length_starts', 'gts', 'grs', ...\n"
                "  - single-motion:   keys 'rigid_body_pos', 'fps', 'dof_pos', ..."
            )

        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)
        ctrl_times = torch.linspace(0.0, motion_length, num_ctrl_frames)

        motion_len_t  = torch.tensor([motion_length])
        num_frames_t  = torch.tensor([nf])
        motion_dt_t   = torch.tensor([src_dt])

        f0_list, f1_list, blend_list = [], [], []
        for t in ctrl_times:
            t_t = t.unsqueeze(0)
            f0, f1, bl = calc_frame_blend(t_t, motion_len_t, num_frames_t, motion_dt_t)
            f0_list.append(f0)
            f1_list.append(f1)
            blend_list.append(bl)

        f0    = torch.cat(f0_list)
        f1    = torch.cat(f1_list)
        blend = torch.cat(blend_list)

        def _interp_pos(src):
            s0 = src[f0]
            s1 = src[f1]
            return interpolate_pos(s0, s1, blend)

        def _interp_quat(src):
            s0 = src[f0]
            s1 = src[f1]
            return interpolate_quat(s0, s1, blend)

        body_pos     = _interp_pos(gts)
        body_rot     = _interp_quat(grs)
        body_vel     = _interp_pos(gvs)
        body_ang_vel = _interp_pos(gavs)
        dof_pos      = _interp_pos(dps)
        dof_vel      = _interp_pos(dvs)

        self._dof_pos      = dof_pos.numpy().astype(np.float32)
        self._dof_vel      = dof_vel.numpy().astype(np.float32)
        self._body_rot     = body_rot.numpy().astype(np.float32)
        self._body_pos     = body_pos.numpy().astype(np.float32)
        self._body_vel     = body_vel.numpy().astype(np.float32)
        self._body_ang_vel = body_ang_vel.numpy().astype(np.float32)
        self._num_frames   = num_ctrl_frames

        print(
            f"[MotionPlayer] Loaded raw motion #{motion_index}: "
            f"{nf} source frames @ {1.0 / src_dt:.1f} Hz -> "
            f"{num_ctrl_frames} resampled frames @ {1.0 / control_dt:.0f} Hz"
        )
