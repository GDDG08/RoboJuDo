# Goalkeeper Policy Toggle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 启动后默认站立，按手柄 Start 切换策略启停；停止时回到站立姿态。

**Architecture:** 在 `RlPipeline` 中维护 `policy_enabled` 开关，根据 `[POLICY_TOGGLE]` 命令切换状态；关闭时用 `policy.get_init_dof_pos()` 输出站立姿态。`g1_goalkeeper_real` 将 Start 绑定到 `[POLICY_TOGGLE]`。

**Tech Stack:** Python, pytest

### Task 1: Pipeline 策略开关（TDD）

**Files:**
- Modify: `robojudo/pipeline/rl_pipeline.py`
- Test: `tests/test_policy_toggle.py`

**Step 1: Write the failing test**

```python
from robojudo.pipeline.rl_pipeline import _toggle_policy_enabled


def test_toggle_policy_enabled():
    assert _toggle_policy_enabled(False, []) is False
    assert _toggle_policy_enabled(False, ["[POLICY_TOGGLE]"]) is True
    assert _toggle_policy_enabled(True, ["[POLICY_TOGGLE]"]) is False
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_policy_toggle.py::test_toggle_policy_enabled -v`  
Expected: FAIL with `ImportError`/`AttributeError` because `_toggle_policy_enabled` not defined.

**Step 3: Write minimal implementation**

```python
def _toggle_policy_enabled(policy_enabled: bool, commands: list[str]) -> bool:
    if commands.count("[POLICY_TOGGLE]") % 2 == 1:
        return not policy_enabled
    return policy_enabled
```

在 `RlPipeline.__init__` 初始化 `self.policy_enabled = False`。  
在 `step()` 中读取 commands 后调用 `_toggle_policy_enabled` 更新状态；若状态变化则 `self.policy.reset()`。  
当 `policy_enabled` 为 False 时，`pd_target = self.policy.get_init_dof_pos()`；为 True 时按原逻辑 `get_pd_target`。

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_policy_toggle.py::test_toggle_policy_enabled -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_policy_toggle.py robojudo/pipeline/rl_pipeline.py
git commit -m "feat: add policy toggle support in rl pipeline"
```

### Task 2: Start 键触发策略开关（TDD）

**Files:**
- Modify: `robojudo/config/g1/g1_cfg.py`
- Test: `tests/test_goalkeeper_config.py`

**Step 1: Write the failing test**

在 `tests/test_goalkeeper_config.py` 增加：

```python
from robojudo.config.g1.g1_cfg import g1_goalkeeper_real


def test_goalkeeper_real_start_toggle():
    cfg = g1_goalkeeper_real()
    unitree_ctrl = next(ctrl for ctrl in cfg.ctrl if ctrl.ctrl_type == "UnitreeCtrl")
    assert unitree_ctrl.triggers_extra.get("Start") == "[POLICY_TOGGLE]"
```

**Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_goalkeeper_config.py::test_goalkeeper_real_start_toggle -v`  
Expected: FAIL because `Start` 未绑定。

**Step 3: Write minimal implementation**

在 `g1_goalkeeper_real` 的 `UnitreeCtrlCfg()` 增加：

```python
UnitreeCtrlCfg(
    triggers_extra={"Start": "[POLICY_TOGGLE]"},
),
```

**Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_goalkeeper_config.py::test_goalkeeper_real_start_toggle -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_goalkeeper_config.py robojudo/config/g1/g1_cfg.py
git commit -m "feat: bind Start to policy toggle in g1_goalkeeper_real"
```

### Task 3: 更新运行说明

**Files:**
- Modify: `docs/goalkeeper_run.md`

**Step 1: Update doc**

补充说明：
- 程序启动后机器人进入站立
- 按 Start 开始策略（Goalkeeper）
- 再按 Start 停止策略并回到站立

**Step 2: Commit**

```bash
git add docs/goalkeeper_run.md
git commit -m "docs: describe Start toggle behavior for goalkeeper"
```
