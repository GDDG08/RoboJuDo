# Goalkeeper 部署实现计划（RoboJuDo）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 RoboJuDo 中新增 Goalkeeper 策略与 DDS 订阅控制器，完成 G1 真机闭环部署（29DoF）。

**Architecture:** 新增 DDS 订阅控制器提供球检测输入；新增 GoalkeeperPolicy 构建 96 维观测 + 10 帧历史，并加载 ActorCritic checkpoint 推理；新增 g1 配置类串联 UnitreeEnv + 新控制器 + 新策略。

**Tech Stack:** Python, torch, cyclonedds, unitree_sdk2py, pytest

---

### Task 1: 添加 DDS 消息类型与球检测控制器

**Files:**
- Create: `robojudo/dds/__init__.py`
- Create: `robojudo/dds/detection_module.py`
- Create: `robojudo/controller/goalkeeper_ball_dds_ctrl.py`
- Modify: `robojudo/controller/__init__.py`
- Modify: `robojudo/controller/ctrl_cfgs.py`
- Create: `robojudo/config/g1/ctrl/g1_goalkeeper_ball_dds_ctrl_cfg.py`
- Test: `tests/test_goalkeeper_ball_dds_ctrl.py`

**Step 1: Write the failing test**

```python
# tests/test_goalkeeper_ball_dds_ctrl.py
import numpy as np

from robojudo.controller.goalkeeper_ball_dds_ctrl import pick_best_ball_detection


class _DummyResult:
    def __init__(self, class_id, class_name, score, xyz):
        self.class_id = class_id
        self.class_name = class_name
        self.score = score
        self.xyz = xyz


def test_pick_best_ball_detection():
    results = [
        _DummyResult("1", "Goalpost", 0.9, [1, 2, 3]),
        _DummyResult("0", "Ball", 0.4, [0.1, 0.2, 0.3]),
        _DummyResult("0", "Ball", 0.8, [0.4, 0.5, 0.6]),
    ]
    best = pick_best_ball_detection(results)
    assert best is not None
    assert np.allclose(best["xyz"], np.array([0.4, 0.5, 0.6], dtype=np.float32))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_goalkeeper_ball_dds_ctrl.py::test_pick_best_ball_detection -v`  
Expected: FAIL with "No module named" or "pick_best_ball_detection not defined"

**Step 3: Write minimal implementation**

```python
# robojudo/dds/detection_module.py
from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DetectionResult(idl.IdlStruct, typename="DetectionModule::DetectionResult"):
    class_id: str
    class_name: str
    box: types.array[types.float32, 4]
    score: types.float32
    xyz: types.array[types.float32, 3]
    offset: types.array[types.float32, 2]
    offset_fov: types.array[types.float32, 2]


@dataclass
@annotate.appendable
@annotate.autoid("sequential")
class DetectionResults(idl.IdlStruct, typename="DetectionModule::DetectionResults"):
    results: types.sequence["robojudo.dds.detection_module.DetectionResult"]
```

```python
# robojudo/controller/goalkeeper_ball_dds_ctrl.py
import threading
import time
from typing import Any

import numpy as np
from dds import DomainParticipant, Subscriber, Topic, DataReader  # cyclonedds

from robojudo.controller.base_ctrl import Controller
from robojudo.dds.detection_module import DetectionResults


def pick_best_ball_detection(results: list[Any]):
    best = None
    best_score = -1.0
    for r in results:
        if str(getattr(r, "class_id", "")) != "0" and str(getattr(r, "class_name", "")).lower() != "ball":
            continue
        score = float(getattr(r, "score", -1.0))
        if score > best_score:
            best = r
            best_score = score
    if best is None:
        return None
    return {"xyz": np.asarray(best.xyz, dtype=np.float32), "score": best_score}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_goalkeeper_ball_dds_ctrl.py::test_pick_best_ball_detection -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_goalkeeper_ball_dds_ctrl.py \
  robojudo/dds/__init__.py \
  robojudo/dds/detection_module.py \
  robojudo/controller/goalkeeper_ball_dds_ctrl.py \
  robojudo/controller/__init__.py \
  robojudo/controller/ctrl_cfgs.py \
  robojudo/config/g1/ctrl/g1_goalkeeper_ball_dds_ctrl_cfg.py
git commit -m "feat: add DDS ball detection controller"
```

---

### Task 2: 添加 GoalkeeperPolicy 与配置

**Files:**
- Create: `robojudo/policy/goalkeeper_policy.py`
- Modify: `robojudo/policy/__init__.py`
- Modify: `robojudo/policy/policy_cfgs.py`
- Create: `robojudo/config/g1/policy/g1_goalkeeper_policy_cfg.py`
- Test: `tests/test_goalkeeper_policy.py`

**Step 1: Write the failing test**

```python
# tests/test_goalkeeper_policy.py
import numpy as np
from box import Box

from robojudo.policy.goalkeeper_policy import GoalkeeperPolicy
from robojudo.config.g1.policy.g1_goalkeeper_policy_cfg import G1GoalkeeperPolicyCfg


def test_goalkeeper_obs_shape():
    cfg = G1GoalkeeperPolicyCfg(load_checkpoint=False)
    policy = GoalkeeperPolicy(cfg_policy=cfg, device="cpu")

    env_data = Box(
        dof_pos=np.zeros(29),
        dof_vel=np.zeros(29),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        base_ang_vel=np.zeros(3),
        torso_pos=np.zeros(3),
        torso_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    ctrl_data = Box({"GoalkeeperBallDdsCtrl": {"ball_xyz_cam": np.array([1.0, 0.0, 0.0]), "ball_valid": True}})
    obs, extras = policy.get_observation(env_data, ctrl_data)
    assert obs.shape == (96,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_goalkeeper_policy.py::test_goalkeeper_obs_shape -v`  
Expected: FAIL (GoalkeeperPolicy not found)

**Step 3: Write minimal implementation**

```python
# robojudo/policy/goalkeeper_policy.py
import sys
import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import GoalkeeperPolicyCfg
from robojudo.utils.util_func import quat_rotate_inverse_np


@policy_registry.register
class GoalkeeperPolicy(Policy):
    cfg_policy: GoalkeeperPolicyCfg

    def __init__(self, cfg_policy: GoalkeeperPolicyCfg, device="cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.history_len = self.cfg_policy.history_length
        self._init_history(np.zeros(self.cfg_policy.history_obs_size, dtype=np.float32))
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        if self.cfg_policy.load_checkpoint:
            self._load_checkpoint()

    def _load_checkpoint(self):
        repo_path = self.cfg_policy.goalkeeper_repo_path
        if repo_path and repo_path not in sys.path:
            sys.path.append(repo_path)
        from rsl_rl.modules.actor_critic import ActorCritic  # type: ignore

        self.actor_critic = ActorCritic(
            num_actor_obs=self.cfg_policy.num_actor_obs,
            num_critic_obs=self.cfg_policy.num_critic_obs,
            num_one_step_obs=self.cfg_policy.num_one_step_obs,
            actor_history_length=self.cfg_policy.actor_history_length,
            num_actions=self.num_actions,
            actor_hidden_dims=self.cfg_policy.actor_hidden_dims,
            critic_hidden_dims=self.cfg_policy.critic_hidden_dims,
            activation=self.cfg_policy.activation,
        ).to(self.device)
        ckpt = torch.load(self.cfg_policy.checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.actor_critic.load_state_dict(state)
        self.actor_critic.eval()

    def reset(self):
        self.last_action[:] = 0

    def post_step_callback(self, commands=None):
        return

    def _projected_gravity(self, quat):
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quat_rotate_inverse_np(quat, gravity)

    def _ball_pos_pelvis(self, env_data, ctrl_data):
        data = ctrl_data.get("GoalkeeperBallDdsCtrl", {})
        if not data.get("ball_valid", False):
            return np.zeros(3, dtype=np.float32)
        ball_cam = np.asarray(data["ball_xyz_cam"], dtype=np.float32)
        # camera -> torso
        ball_torso = self.cfg_policy.camera_R_tc @ ball_cam + self.cfg_policy.camera_t_tc
        # torso -> world
        torso_pos = env_data.torso_pos
        torso_quat = env_data.torso_quat
        ball_world = torso_pos + quat_rotate_inverse_np(torso_quat, -ball_torso)  # R * p
        # world -> base/pelvis
        return quat_rotate_inverse_np(env_data.base_quat, ball_world - torso_pos)

    def get_observation(self, env_data, ctrl_data):
        ball_pos = self._ball_pos_pelvis(env_data, ctrl_data) * self.cfg_policy.ball_pos_scale
        ang_vel = env_data.base_ang_vel * self.cfg_policy.ang_vel_scale
        proj_grav = self._projected_gravity(env_data.torso_quat)
        dof_pos = (env_data.dof_pos - self.default_dof_pos) * self.cfg_policy.dof_pos_scale
        dof_vel = env_data.dof_vel * self.cfg_policy.dof_vel_scale
        obs = np.concatenate([ball_pos, ang_vel, proj_grav, dof_pos, dof_vel, self.last_action])
        return obs.astype(np.float32), {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_hist = np.array(self.history_buf).flatten()
        self.history_buf.append(obs)
        obs_input = np.concatenate([obs, obs_hist])
        with torch.no_grad():
            action = self.actor_critic.act_inference(torch.from_numpy(obs_input).float().unsqueeze(0))
        action = action.cpu().numpy().squeeze().astype(np.float32)
        self.last_action = action.copy()
        return action
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_goalkeeper_policy.py::test_goalkeeper_obs_shape -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add robojudo/policy/goalkeeper_policy.py \
  robojudo/policy/__init__.py \
  robojudo/policy/policy_cfgs.py \
  robojudo/config/g1/policy/g1_goalkeeper_policy_cfg.py \
  tests/test_goalkeeper_policy.py
git commit -m "feat: add Goalkeeper policy and config"
```

---

### Task 3: 新增 g1 真机部署配置

**Files:**
- Modify: `robojudo/config/g1/g1_cfg.py`

**Step 1: Write the failing test**

```python
# tests/test_goalkeeper_config.py
from robojudo.config.config_manager import ConfigManager

def test_goalkeeper_config_loads():
    cfg = ConfigManager(config_name="g1_goalkeeper_real").get_cfg()
    assert cfg.policy.policy_type == "GoalkeeperPolicy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_goalkeeper_config.py::test_goalkeeper_config_loads -v`  
Expected: FAIL (config not registered)

**Step 3: Write minimal implementation**

```python
# robojudo/config/g1/g1_cfg.py
@cfg_registry.register
class g1_goalkeeper_real(RlPipelineCfg):
    robot: str = "g1"
    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeEnv",
        unitree=G1UnitreeCfg(net_if="eth0"),
    )
    ctrl = [UnitreeCtrlCfg(), G1GoalkeeperBallDdsCtrlCfg()]
    policy: G1GoalkeeperPolicyCfg = G1GoalkeeperPolicyCfg()
    do_safety_check: bool = True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_goalkeeper_config.py::test_goalkeeper_config_loads -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add robojudo/config/g1/g1_cfg.py tests/test_goalkeeper_config.py
git commit -m "feat: add g1 goalkeeper real config"
```

---

### Task 4: 依赖补充（cyclonedds）

**Files:**
- Modify: `requirements.txt`

**Step 1: Write the failing test**

```python
# tests/test_cyclonedds_import.py
import importlib

def test_cyclonedds_import():
    assert importlib.import_module("cyclonedds") is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cyclonedds_import.py::test_cyclonedds_import -v`  
Expected: FAIL if cyclonedds not installed

**Step 3: Write minimal implementation**

Add to `requirements.txt`:
```
cyclonedds
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cyclonedds_import.py::test_cyclonedds_import -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add requirements.txt tests/test_cyclonedds_import.py
git commit -m "chore: add cyclonedds dependency"
```

---

### Task 5: 运行总测试与校验

**Files:**
- Test: `tests/`

**Step 1: Run full test suite**

Run: `pytest -q`  
Expected: PASS

**Step 2: Commit (if needed)**

```bash
git status --short
```

---

Plan complete and saved to `docs/plans/2026-01-26-goalkeeper-robojudo-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session)  
2. Parallel Session (separate)

Which approach?  
