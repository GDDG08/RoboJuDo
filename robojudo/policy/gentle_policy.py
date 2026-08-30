import logging
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import GentlePolicyCfg
from robojudo.utils.util_func import get_gravity_orientation


logger = logging.getLogger(__name__)


def _quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize quaternion(s) in wxyz order with numerical stability."""
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return (q / n).astype(np.float32)


def _quat_apply_inv_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply inverse quaternion rotation (wxyz order) to vector(s)."""
    rot = Rotation.from_quat(q_wxyz, scalar_first=True)
    return rot.inv().apply(v).astype(np.float32)


def _yaw_component_wxyz(quat: np.ndarray) -> np.ndarray:
    """Extract yaw-only component (wxyz order) from quaternion(s)."""
    q = np.asarray(quat)
    shp = q.shape
    qv = q.reshape(-1, 4)

    w = qv[:, 0]
    x = qv[:, 1]
    y = qv[:, 2]
    z = qv[:, 3]

    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw

    q_yaw = np.zeros_like(qv, dtype=np.float32)
    q_yaw[:, 0] = np.cos(half)
    q_yaw[:, 3] = np.sin(half)
    q_yaw = _quat_normalize_wxyz(q_yaw)

    return q_yaw.reshape(shp).astype(np.float32)


def _clamp_future_indices(idx: np.ndarray, n: int) -> np.ndarray:
    """Clamp indices to valid range [0, n-1]."""
    return np.clip(idx, 0, max(0, n - 1))


def _linspace_rows(a: np.ndarray, b: np.ndarray, steps: int) -> List[np.ndarray]:
    if steps <= 0:
        return []
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = steps + 1
    rows: List[np.ndarray] = []
    for i in range(1, steps + 1):
        t = i / denom
        rows.append(((1.0 - t) * a + t * b).astype(np.float32))
    return rows


def _slerp_many_wxyz(q0: np.ndarray, q1: np.ndarray, steps: int) -> List[np.ndarray]:
    if steps <= 0:
        return []
    q0 = _quat_normalize_wxyz(q0)
    q1 = _quat_normalize_wxyz(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = (-q1).astype(np.float32)
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    from scipy.spatial.transform import Slerp

    key_rots = Rotation.concatenate([r0, r1])
    slerp = Slerp([0.0, 1.0], key_rots)
    denom = steps + 1
    ts = [i / denom for i in range(1, steps + 1)]
    rots = slerp(ts)
    out: List[np.ndarray] = []
    for r in rots:
        x, y, z, w = r.as_quat()
        out.append(np.array([w, x, y, z], dtype=np.float32))
    return out


@dataclass
class _TrackingState:
    ref_joint_pos: List[np.ndarray]
    ref_root_pos: List[np.ndarray]
    ref_root_quat: List[np.ndarray]
    ref_idx: int
    transition_len: int

    @property
    def ref_len(self) -> int:
        return len(self.ref_joint_pos)


@policy_registry.register
class GentlePolicy(Policy):
    """Gentle Policy implementation based on gentleHum sim2sim project"""
    
    cfg_policy: GentlePolicyCfg
    
    def __init__(self, cfg_policy: GentlePolicyCfg, device):
        # Initialize policy configuration first
        super().__init__(cfg_policy=cfg_policy, device=device)
        
        # Initialize ONNX session
        sess_options = ort.SessionOptions()
        
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError(f"Unknown device: {device}")
        
        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Policy state
        self.timestep = 0
        self.last_action = np.zeros(self.cfg_action_dof.num_dofs, dtype=np.float32)
        
        # History buffers
        self.joint_pos_history = np.zeros(
            (max(cfg_policy.joint_pos_steps) + 1, self.cfg_action_dof.num_dofs), 
            dtype=np.float32
        )
        self.prev_actions_history = np.zeros(
            (cfg_policy.prev_actions_steps, self.cfg_action_dof.num_dofs), 
            dtype=np.float32
        )
        
        # Tracking state
        self.tracking_state: _TrackingState | None = None
        self.tracking_future_steps: List[int] = list(getattr(cfg_policy, "future_steps", [0, 2, 4, 8, 16]))
        self.tracking_transition_steps: int = int(getattr(cfg_policy, "transition_steps", 100))
        self.tracking_dataset_joint_names: List[str] = []
        self.tracking_policy_joint_names: List[str] = list(self.cfg_action_dof.joint_names)
        self.tracking_map_dataset_to_policy: List[int] = []
        self.tracking_loaded: bool = False
        
        # Store latest env_data for post_step_callback use
        self._latest_env_data = None
        
        # Motion switching state
        self.current_motion_idx: int = 0
        self.motion_list: List[str] = ["default"]
        self.pending_motion_idx: int = 0  # Selected but inactive motion
        self._load_motion_list()
        
        # Compliance state
        self.compliance_enabled = cfg_policy.compliance_enabled
        self.compliance_threshold = cfg_policy.compliance_threshold
        
        self.reset()
    
    def reset(self):
        """Reset policy state"""
        self.timestep = 0
        self.last_action.fill(0.0)
        self.joint_pos_history.fill(0.0)
        self.prev_actions_history.fill(0.0)
        self.tracking_state = None
        
        # Initialize joint position history with default positions
        if hasattr(self, 'default_dof_pos'):
            for i in range(self.joint_pos_history.shape[0]):
                self.joint_pos_history[i] = self.default_dof_pos

    def _xyzw_to_wxyz(self, quat_xyzw: np.ndarray) -> np.ndarray:
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    def _current_tracking_state(self, env_data) -> dict:
        base_pos = env_data.base_pos
        base_quat_xyzw = env_data.base_quat
        base_quat_wxyz = self._xyzw_to_wxyz(base_quat_xyzw)
        return {
            "jointPos": np.asarray(env_data.dof_pos, dtype=np.float32),
            "rootPos": np.asarray(base_pos, dtype=np.float32),
            "rootQuat": _quat_normalize_wxyz(base_quat_wxyz),
        }

    def _load_tracking_config_from_assets(self) -> None:
        if self.tracking_loaded:
            return

        policy_dir = Path(self.cfg_policy.policy_file).resolve().parent
        cfg_path = policy_dir / "tracking_policy_latest.json"
        if not cfg_path.exists():
            self.tracking_loaded = True
            return
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        tracking_cfg = cfg.get("tracking", {}) if isinstance(cfg, dict) else {}
        ds_names = tracking_cfg.get("dataset_joint_names", [])
        if isinstance(ds_names, list):
            self.tracking_dataset_joint_names = [str(x) for x in ds_names]
        fut = tracking_cfg.get("future_steps", None)
        if isinstance(fut, list) and len(fut) > 0:
            self.tracking_future_steps = [int(x) for x in fut]
        trans = tracking_cfg.get("transition_steps", None)
        if trans is not None:
            self.tracking_transition_steps = int(trans)


    def _build_dataset_to_policy_map(self) -> None:
        if self.tracking_map_dataset_to_policy:
            return
        if not self.tracking_dataset_joint_names or not self.tracking_policy_joint_names:
            return
        idx = {name: i for i, name in enumerate(self.tracking_dataset_joint_names)}
        mapping: List[int] = []
        for name in self.tracking_policy_joint_names:
            if name not in idx:
                raise ValueError(f"Tracking joint '{name}' missing in dataset_joint_names")
            mapping.append(int(idx[name]))
        self.tracking_map_dataset_to_policy = mapping

    def _map_dataset_joint_pos_to_policy(self, joint_pos_dataset: np.ndarray) -> np.ndarray:
        joint_pos_dataset = np.asarray(joint_pos_dataset, dtype=np.float32)
        if not self.tracking_map_dataset_to_policy:
            return joint_pos_dataset.astype(np.float32)
        out = np.zeros(len(self.tracking_map_dataset_to_policy), dtype=np.float32)
        for i, ds_i in enumerate(self.tracking_map_dataset_to_policy):
            out[i] = float(joint_pos_dataset[ds_i]) if ds_i < joint_pos_dataset.shape[0] else 0.0
        return out

    def _load_motion_list(self) -> None:
        """Scan motions directory to build a list of available motions"""
        motions_config_path = Path(self.cfg_policy.motions_path) / "motions.json"
        if motions_config_path.exists():
            with motions_config_path.open("r", encoding="utf-8") as f:
                motions_config = json.load(f)
            
            # Build motion list from config, ensuring default is first
            self.motion_list = []
            self.motion_names = []  # Store display names
            self.motion_compliance = {}  # Store compliance suitability
            
            # First, add default motion if it exists
            default_motion = None
            for motion in motions_config.get("motions", []):
                file_stem = Path(motion["file"]).stem
                if file_stem == "default":
                    default_motion = motion
                    break
            
            if default_motion:
                self.motion_list.append("default")
                self.motion_names.append(default_motion["name"])
                self.motion_compliance["default"] = default_motion.get("compliance_suitable", True)
            
            # Then add all other motions
            for motion in motions_config.get("motions", []):
                file_stem = Path(motion["file"]).stem
                if file_stem != "default":
                    self.motion_list.append(file_stem)
                    self.motion_names.append(motion["name"])
                    self.motion_compliance[file_stem] = motion.get("compliance_suitable", True)
            
            logger.info(f"Loaded {len(self.motion_list)} motions from config")
        else:
            # Fallback to directory scanning
            motions_dir = Path(self.cfg_policy.motions_path) / "motions"
            if motions_dir.exists():
                motion_files = list(motions_dir.glob("*.json"))
                other_motions = [f.stem for f in motion_files if f.stem != "default"]
                self.motion_list = ["default"] + sorted(other_motions)
                self.motion_names = self.motion_list.copy()
                self.motion_compliance = {name: True for name in self.motion_list}
                logger.info(f"Loaded motion list from directory: {self.motion_list}")

    def _get_motion_display_name(self, motion_file_stem: str) -> str:
        """Get display name for motion from file stem"""
        if hasattr(self, 'motion_names') and motion_file_stem in self.motion_list:
            idx = self.motion_list.index(motion_file_stem)
            if idx < len(self.motion_names):
                return self.motion_names[idx]
        return motion_file_stem
    
    def _is_compliance_suitable(self, motion_file_stem: str) -> bool:
        """Check if motion is suitable for compliance"""
        if hasattr(self, 'motion_compliance'):
            return self.motion_compliance.get(motion_file_stem, True)
        return True

    def _load_motion_by_name(self, name: str) -> dict | None:
        """Load a specific motion by its filename (without .json extension)"""
        motion_path = Path(self.cfg_policy.motions_path) / "motions" / f"{name}.json"
        if not motion_path.exists():
            logger.warning(f"Motion file not found: {motion_path}")
            return None
        with motion_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _start_tracking_by_name(self, name: str, env_data) -> None:
        """Start tracking a specific motion by name"""
        self._load_tracking_config_from_assets()
        if not self.tracking_dataset_joint_names:
            self.tracking_dataset_joint_names = list(self.tracking_policy_joint_names)
        self._build_dataset_to_policy_map()

        motion = self._load_motion_by_name(name)
        
        logger.info(f"Successfully loaded motion '{name}', parsing data...")

        # Parse motion clip
        joint_pos_ds = [np.asarray(row, dtype=np.float32) for row in motion.get("joint_pos", [])]
        root_pos = [np.asarray(row, dtype=np.float32) for row in motion.get("root_pos", [])]
        root_quat = [
            _quat_normalize_wxyz(np.asarray(row, dtype=np.float32))
            for row in motion.get("root_quat", [])
        ]
        if len(joint_pos_ds) == 0 or len(root_pos) == 0 or len(root_quat) == 0:
            return

        joint_pos = [self._map_dataset_joint_pos_to_policy(row) for row in joint_pos_ds]
        curr = self._current_tracking_state(env_data)
        
        # Align motion to current state
        p0 = root_pos[0]
        q0_yaw = _yaw_component_wxyz(root_quat[0])
        pa = curr["rootPos"]
        qa_yaw = _yaw_component_wxyz(curr["rootQuat"])

        r0 = Rotation.from_quat(q0_yaw, scalar_first=True)
        ra = Rotation.from_quat(qa_yaw, scalar_first=True)
        r_delta = ra * r0.inv()

        aligned_root_pos: List[np.ndarray] = []
        for pos in root_pos:
            rel = np.asarray(pos - p0, dtype=np.float32)
            rot_rel = r_delta.apply(rel)
            aligned_pos = (rot_rel.astype(np.float32) + pa).astype(np.float32)
            aligned_pos[2] = pos[2] # Keep original Z height
            aligned_root_pos.append(aligned_pos)

        aligned_root_quat: List[np.ndarray] = []
        for q in root_quat:
            q_all = Rotation.from_quat(q, scalar_first=True)
            q_aligned = (r_delta * q_all).as_quat(scalar_first=True)
            aligned_root_quat.append(_quat_normalize_wxyz(q_aligned.astype(np.float32)))

        # Build transition
        steps = max(0, int(self.tracking_transition_steps))
        anchor = {"joint_pos": curr["jointPos"], "root_pos": curr["rootPos"], "root_quat": curr["rootQuat"]}
        tgt_first = {"joint_pos": joint_pos[0], "root_pos": aligned_root_pos[0], "root_quat": aligned_root_quat[0]}

        trans_joint = _linspace_rows(anchor["joint_pos"], tgt_first["joint_pos"], steps)
        trans_pos = _linspace_rows(anchor["root_pos"], tgt_first["root_pos"], steps)
        trans_quat = _slerp_many_wxyz(anchor["root_quat"], tgt_first["root_quat"], steps)

        ref_joint = [*trans_joint, *joint_pos]
        ref_pos = [*trans_pos, *aligned_root_pos]
        ref_quat = [*trans_quat, *aligned_root_quat]

        self.tracking_state = _TrackingState(
            ref_joint_pos=ref_joint,
            ref_root_pos=ref_pos,
            ref_root_quat=ref_quat,
            ref_idx=0,
            transition_len=len(trans_joint),
        )
        logger.info(f"Started tracking motion: {name} (total frames: {len(ref_joint)}, transition frames: {len(trans_joint)})")

    def _start_default_tracking_from_current(self, env_data) -> None:
        self._start_tracking_by_name("default", env_data)

    def _advance_tracking(self) -> None:
        if not self.tracking_state:
            return
        if self.tracking_state.ref_len <= 1:
            return
        if self.tracking_state.ref_idx < self.tracking_state.ref_len - 1:
            self.tracking_state.ref_idx += 1
            # Log advancement every 100 steps for debugging
            if self.tracking_state.ref_idx % 100 == 0:
                logger.info(f"Tracking advanced to ref_idx: {self.tracking_state.ref_idx}/{self.tracking_state.ref_len}")
        else:

            # Auto-switch to default if current motion is not default and not compliance suitable
            current_motion_file = self.motion_list[self.current_motion_idx] if self.current_motion_idx < len(self.motion_list) else "default"
            if current_motion_file != "default" and self._latest_env_data is not None:
                # Check if current motion is compliance suitable
                if self._is_compliance_suitable(current_motion_file):
                    # For compliance suitable motions, hold at last frame
                    display_name = self._get_motion_display_name(current_motion_file)
                    # logger.info(f"Motion '{display_name}' finished, holding at last frame (compliance suitable)")
                    # Keep tracking at the last frame by not incrementing ref_idx
                    # This effectively holds the last pose
                else:
                    # For non-compliance motions, auto-switch to default
                    display_name = self._get_motion_display_name(current_motion_file)
                    logger.info(f"Motion '{display_name}' finished, auto-switching to default")
                    self.current_motion_idx = 0
                    self.pending_motion_idx = 0
                    self._start_tracking_by_name("default", self._latest_env_data)
    
    def _compute_boot_indicator(self) -> np.ndarray:
        """Compute boot indicator observation"""
        return np.array([0.0], dtype=np.float32)
    
    def _compute_compliance_flag_obs(self) -> np.ndarray:
        """Compute compliance flag observation"""
        enabled = 1.0 if self.compliance_enabled else 0.0
        threshold = self.compliance_threshold
        kp = threshold / 0.05 if enabled else 0.0
        return np.array([enabled, enabled * threshold, enabled * kp], dtype=np.float32)
    
    def _compute_root_ang_vel_b(self, env_data) -> np.ndarray:
        """Compute root angular velocity in body frame"""
        return env_data.base_ang_vel.astype(np.float32)
    
    def _compute_projected_gravity_b(self, env_data) -> np.ndarray:
        """Compute projected gravity in body frame"""
        # Gravity vector in world frame (consistent with original observation.py)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        # Transform to body frame using inverse rotation
        quat_xyzw = env_data.base_quat # [x, y, z, w]
        rotation = Rotation.from_quat(quat_xyzw) # scipy default is [x, y, z, w]
        gravity_body = rotation.inv().apply(gravity_world)
        
        return gravity_body.astype(np.float32)
    
    def _compute_joint_pos_obs(self, env_data) -> np.ndarray:
        """Compute joint position history observation"""
        # Update history (FIFO)
        self.joint_pos_history = np.roll(self.joint_pos_history, 1, axis=0)
        self.joint_pos_history[0] = env_data.dof_pos
        
        # Extract specified steps
        obs = self.joint_pos_history[self.cfg_policy.joint_pos_steps].reshape(-1)
        return obs.astype(np.float32)
    
    def _compute_prev_actions_obs(self) -> np.ndarray:
        """Compute previous actions observation"""
        # Update history (FIFO)
        self.prev_actions_history = np.roll(self.prev_actions_history, 1, axis=0)
        self.prev_actions_history[0] = self.last_action
        
        return self.prev_actions_history.flatten().astype(np.float32)
    
    def _compute_tracking_command_obs_raw(self) -> np.ndarray:
        """Compute tracking command observation (raw position and rotation)"""
        future_steps = np.asarray(self.tracking_future_steps, dtype=np.int32)

        tr = self.tracking_state
        base = tr.ref_idx
        T = tr.ref_len
        fut_idx = _clamp_future_indices(base + future_steps, T)

        root_pos_w = np.asarray(tr.ref_root_pos, dtype=np.float32)[fut_idx]
        root_quat_w = np.asarray(tr.ref_root_quat, dtype=np.float32)[fut_idx]

        # pos_diff_b
        pos_diff_w = root_pos_w[1:] - root_pos_w[0:1]
        pos_diff_b = _quat_apply_inv_wxyz(root_quat_w[0], pos_diff_w)

        # rot6d
        q_cur = getattr(self, "_latest_root_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        q_cur = _quat_normalize_wxyz(q_cur)
        
        # Use scipy for rotation arithmetic to match original exactly
        r_cur = Rotation.from_quat(q_cur, scalar_first=True)
        r_ref = Rotation.from_quat(root_quat_w, scalar_first=True)
        rel_rot = (r_cur.inv() * r_ref).as_matrix()
        
        if rel_rot.ndim == 2:
            rel_rot = rel_rot[None, ...]
        
        # Original logic: rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1)
        rot6d = rel_rot[:, :, :2].transpose(0, 2, 1).reshape(-1).astype(np.float32)

        return np.concatenate([pos_diff_b.reshape(-1), rot6d], axis=-1).astype(np.float32)
    
    def _compute_target_joint_pos_obs(self) -> np.ndarray:
        """Compute target joint position observation"""
        future_steps = np.asarray(self.tracking_future_steps, dtype=np.int32)
        if self.tracking_state is None:
            n_j = self.cfg_action_dof.num_dofs
            return np.zeros(len(future_steps) * n_j * 2, dtype=np.float32)

        tr = self.tracking_state
        base = tr.ref_idx
        T = tr.ref_len
        fut_idx = _clamp_future_indices(base + future_steps, T)
        
        tgt_joints = np.asarray(tr.ref_joint_pos, dtype=np.float32)[fut_idx]
        cur_joints = getattr(self, "_latest_joint_pos", np.zeros(self.cfg_action_dof.num_dofs, dtype=np.float32))
        cur_joints = cur_joints.reshape(1, -1)
        
        tgt_minus_cur = tgt_joints - cur_joints
        return np.concatenate([tgt_joints.reshape(-1), tgt_minus_cur.reshape(-1)], axis=-1).astype(np.float32)

    def _compute_target_root_z_obs(self) -> np.ndarray:
        """Compute target root height observation"""
        future_steps = np.asarray(self.tracking_future_steps, dtype=np.int32)
        if self.tracking_state is None:
            return np.zeros(len(future_steps), dtype=np.float32)
        
        tr = self.tracking_state
        base = tr.ref_idx
        T = tr.ref_len
        fut_idx = _clamp_future_indices(base + future_steps, T)
        
        root_pos_w = np.asarray(tr.ref_root_pos, dtype=np.float32)[fut_idx]
        return (root_pos_w[:, 2] + 0.035).astype(np.float32)

    def _compute_target_projected_gravity_b_obs(self) -> np.ndarray:
        """Compute target projected gravity observation"""
        future_steps = np.asarray(self.tracking_future_steps, dtype=np.int32)
        if self.tracking_state is None:
            return np.zeros(len(future_steps) * 3, dtype=np.float32)
        
        tr = self.tracking_state
        base = tr.ref_idx
        T = tr.ref_len
        fut_idx = _clamp_future_indices(base + future_steps, T)
        
        root_quat_w = np.asarray(tr.ref_root_quat, dtype=np.float32)[fut_idx]
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32).reshape(1, 3)
        g_local = _quat_apply_inv_wxyz(root_quat_w, g_world)
        return g_local.reshape(-1).astype(np.float32)
    def get_observation(self, env_data, ctrl_data):
        """Compute complete observation vector"""
        # Store latest env_data for post_step_callback use
        self._latest_env_data = env_data
        
        # cache current state for tracking obs
        self._latest_joint_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        self._latest_root_quat_wxyz = self._xyzw_to_wxyz(env_data.base_quat)

        # if self.cfg_policy.tracking_enabled:
        if self.tracking_state is None:
            self._start_default_tracking_from_current(env_data)
        # Update observation state (advance tracking index)
        self._advance_tracking()

        projected_gravity =get_gravity_orientation(env_data.base_quat)
        # # print(projected_gravity)
        # # print(self._compute_projected_gravity_b(env_data))

        obs = np.concatenate(
            [

                # 1. Boot indicator
                self._compute_boot_indicator(),
                # 2. Tracking command observation
                self._compute_tracking_command_obs_raw(),
                # 3. Compliance flag
                self._compute_compliance_flag_obs(),
                # 4. Target joint positions
                self._compute_target_joint_pos_obs(),
                # 5. Target root Z
                self._compute_target_root_z_obs(),
                # 6. Target projected gravity
                self._compute_target_projected_gravity_b_obs(),
                # 7. Root angular velocity base_ang_vel
                self._compute_root_ang_vel_b(env_data),
                # 8. Projected gravity
                # self._compute_projected_gravity_b(env_data),
                projected_gravity,
                # 9. Joint positions
                self._compute_joint_pos_obs(env_data),
                # 10. Previous actions
                self._compute_prev_actions_obs(),

            ]
        )

        # Apply observation scaling
        # scales = self.cfg_policy.obs_scales
        # Note: Would need to apply appropriate scaling to each observation segment
        
        extras = {
            "timestep": self.timestep,
            "compliance_enabled": self.compliance_enabled,
            "compliance_threshold": self.compliance_threshold,
        }
        
        return obs, extras
    
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from policy network"""
        # Prepare input for ONNX inference
        ort_inputs = {
            self.input_names[0]: np.expand_dims(obs, axis=0).astype(np.float32)
        }
        # Run inference
        ort_outputs = self.session.run(self.output_names, ort_inputs)

        action = ort_outputs[6].squeeze(0).astype(np.float32, copy=False)# 4=6 linear_6 

        # Apply action clipping (like original JavaScript)
        if self.cfg_policy.action_clip is not None:
            action = np.clip(action, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        
        # Update last_actions (like original JavaScript)
        self.last_action = action.copy()
        
        action_scale_array = np.asarray(self.cfg_policy.action_scale)
        
        self.timestep += 1
        
        return action_scale_array * self.last_action
        
    
    def get_init_dof_pos(self) -> np.ndarray:
        """Get initial joint positions"""
        return self.default_dof_pos.copy()
    
    def post_step_callback(self, commands: List[str] | None = None):
        """Handle post-step callbacks"""
        # Use stored env_data from get_observation
        env_data = self._latest_env_data
        if env_data is None:
            logger.warning("No env_data available in post_step_callback, motion commands will be ignored")
            return
            
        for command in commands or []:
            match command:
                case "[RESET]":
                    self.reset()
                case "[COMPLIANCE_ON]":
                    current_motion_file = self.motion_list[self.current_motion_idx] if self.current_motion_idx < len(self.motion_list) else "default"
                    if self._is_compliance_suitable(current_motion_file):
                        self.compliance_enabled = True
                        self.compliance_threshold = 10.0
                        logger.info("Compliance threshold reset to 10.0")
                    else:
                        display_name = self._get_motion_display_name(current_motion_file)
                        logger.warning(f"Compliance not suitable for motion '{display_name}', COMPLIANCE_ON ignored")
                case "[COMPLIANCE_OFF]":
                    self.compliance_enabled = False
                case "[MOTION_LOAD_NEXT]":
                    self.pending_motion_idx = (self.pending_motion_idx + 1) % len(self.motion_list)
                    motion_file = self.motion_list[self.pending_motion_idx]
                    display_name = self._get_motion_display_name(motion_file)
                    compliance_status = "✓" if self._is_compliance_suitable(motion_file) else "✗"
                    msg = (f"Selected motion {self.pending_motion_idx}: {display_name} "
                            f"(compliance: {compliance_status}, use MOTION_FADE_IN to start)")
                    logger.info(msg)
                case "[MOTION_LOAD_PREV]":
                    self.pending_motion_idx = (self.pending_motion_idx - 1) % len(self.motion_list)
                    motion_file = self.motion_list[self.pending_motion_idx]
                    display_name = self._get_motion_display_name(motion_file)
                    compliance_status = "✓" if self._is_compliance_suitable(motion_file) else "✗"
                    msg = (f"Selected motion {self.pending_motion_idx}: {display_name} "
                            f"(compliance: {compliance_status}, use MOTION_FADE_IN to start)")
                    logger.info(msg)
                case "[MOTION_FADE_OUT]":
                    self.current_motion_idx = 0
                    self.pending_motion_idx = 0
                    logger.info("Fading out to default motion")
                    self._start_tracking_by_name("default", env_data)
                case "[MOTION_FADE_IN]":
                    motion_file = self.motion_list[self.pending_motion_idx]
                    self.current_motion_idx = self.pending_motion_idx
                    display_name = self._get_motion_display_name(motion_file)
                    msg = (f"Starting motion {self.current_motion_idx}: "
                            f"{display_name}")
                    logger.info(msg)
                    self._start_tracking_by_name(motion_file, env_data)

                case "[TRESH_UP]":
                    # 最大不大于180,会出现抖动。
                    if self.compliance_enabled and self.compliance_threshold < 100:
                        self.compliance_threshold += 5
                        logger.info(f"Compliance threshold: {self.compliance_threshold}")
                    else:
                        logger.warning("Compliance threshold already at maximum or compliance disabled, cannot increase threshold")
                case "[TRESH_DOWN]":
                    # 最小不能小于-10,会出现右手后伸现象，最后跌倒。
                    if self.compliance_enabled and self.compliance_threshold > -10:
                        self.compliance_threshold -= 5
                        logger.info(f"Compliance threshold: {self.compliance_threshold}")
                    else:
                        logger.warning("Compliance threshold already at minimum or compliance disabled, cannot decrease threshold")
    
    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        """Debug visualization"""
        # # debug print all obs in correct order
        # print(30*"-")
        # print("OBSERVATION DEBUG PRINT (CORRECT ORDER):")
        # print("1. Boot indicator:")
        # print(self._compute_boot_indicator())
        # print("2. Tracking command observation:")
        # print(self._compute_tracking_command_obs_raw())
        # print("3. Compliance flag:")
        # print(self._compute_compliance_flag_obs())
        # print("4. Target joint positions:")
        # print(self._compute_target_joint_pos_obs())
        # print("5. Target root Z:")
        # print(self._compute_target_root_z_obs())
        # print("6. Target projected gravity:")
        # print(self._compute_target_projected_gravity_b_obs())
        # print("7. Root angular velocity:")
        # print(self._compute_root_ang_vel_b(env_data))
        # print("8. Projected gravity:")
        # print(projected_gravity)
        # print("9. Joint positions:")
        # print(self._compute_joint_pos_obs(env_data))
        # print("10. Previous actions:")
        # print(self._compute_prev_actions_obs())
        # print(f"Total obs shape: {obs.shape}")
        # print(30*"-")
        pass