"""SimTestRunner: drive a configured pipeline headlessly and watch for falls.

Loads a config by name, mutates the env cfg to enable the
:class:`SimStabilityMonitor` and skip the MuJoCo viewer, strips interactive
controllers (joystick/keyboard) so the test can run without a display, then
steps the pipeline for ``spec.run_seconds`` and asserts that the monitor did
not trip.

Scripted ``[MOTION_*]`` commands from ``spec.startup_commands`` are injected
into ``ctrl_manager.get_ctrl_data`` so we can simulate the human pressing
``R`` for tracker-style policies.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import socket
from dataclasses import dataclass
from pathlib import Path

from box import Box

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.environment.sim_stability import SimStabilityCfg
from robojudo.pipeline import Pipeline

from .specs import PolicySimSpec

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SimTestResult:
    cfg_name: str
    passed: bool
    display_name: str | None = None
    skipped: bool = False
    reason: str | None = None
    elapsed: float = 0.0
    failure_reason: str | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.cfg_name


def _resolve_stability_cfg(robot: str) -> SimStabilityCfg:
    """Per-robot default fall thresholds. Add new robots here."""
    if robot == "g1":
        from robojudo.config.g1.env.g1_mujuco_env_cfg import g1_default_sim_stability_cfg

        return g1_default_sim_stability_cfg()
    if robot == "h1":
        from robojudo.config.h1.env.h1_mujuco_env_cfg import h1_default_sim_stability_cfg

        return h1_default_sim_stability_cfg()
    raise ValueError(
        f"No sim-stability defaults registered for robot '{robot}'. "
        f"Add a factory in robojudo/config/<robot>/env/*.py and wire it here."
    )


def _spec_should_skip(spec: PolicySimSpec) -> str | None:
    if spec.skip_reason:
        return spec.skip_reason
    for mod in spec.required_modules:
        if importlib.util.find_spec(mod) is None:
            return f"required module '{mod}' not installed"
    for asset in spec.required_assets:
        if not (REPO_ROOT / asset).exists():
            return f"required asset '{asset}' missing"
    if spec.requires_redis_server is not None:
        host, port = spec.requires_redis_server
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            return f"redis server not reachable at {host}:{port}"
    return None


def _patch_command_injection(
    pipeline: Pipeline,
    schedule: list[tuple[float, str]],
    dt: float,
) -> None:
    """Wrap ``ctrl_manager.get_ctrl_data`` so scripted commands appear in the
    COMMANDS list at the right wall-clock offsets.

    Steps are counted post-init (after the dry-run self_check), so ``time=0.0``
    is the first real ``pipeline.step()`` of the run loop.
    """
    if not schedule:
        return

    queue = sorted(
        ((max(0, int(round(t / dt))), cmd) for t, cmd in schedule),
        key=lambda x: x[0],
    )
    state = {"step": 0, "queue": list(queue)}
    original = pipeline.ctrl_manager.get_ctrl_data

    def patched(env_data):
        ctrl_data = original(env_data)
        commands = list(ctrl_data.get("COMMANDS", []))
        while state["queue"] and state["queue"][0][0] <= state["step"]:
            _, cmd = state["queue"].pop(0)
            commands.append(cmd)
            logger.info("[SimRunner] inject command at step %d: %s", state["step"], cmd)
        ctrl_data["COMMANDS"] = commands
        # Box subclasses dict, so mutation is fine but we re-wrap to be safe.
        if not isinstance(ctrl_data, Box):
            ctrl_data = Box(ctrl_data)
        state["step"] += 1
        return ctrl_data

    pipeline.ctrl_manager.get_ctrl_data = patched  # type: ignore[method-assign]


def run_spec(spec: PolicySimSpec) -> SimTestResult:
    skip_reason = _spec_should_skip(spec)
    if skip_reason:
        return SimTestResult(
            cfg_name=spec.cfg_name,
            display_name=spec.display_name,
            passed=True,
            skipped=True,
            reason=skip_reason,
        )

    label = spec.display_name or spec.cfg_name
    logger.info("[SimRunner] === %s ===", label)
    logger.info("[SimRunner] %s", spec.description)

    cfg = ConfigManager(config_name=spec.cfg_name).get_cfg()

    # --- Mutate env for sim tests ---
    if not getattr(cfg.env, "is_sim", False):
        return SimTestResult(
            cfg_name=spec.cfg_name,
            display_name=spec.display_name,
            passed=False,
            reason=f"config '{spec.cfg_name}' is not a sim config (is_sim=False).",
        )
    cfg.env.headless = True
    cfg.env.sim_stability = _resolve_stability_cfg(cfg.robot)
    # The test only checks stability; production safety_check is unaffected.
    cfg.do_safety_check = False
    cfg.run_fullspeed = True
    cfg.debug.log_obs = False

    # The spec may pin a specific controller (e.g. TwistPolicy docs prescribe
    # MotionTwistCtrl for offline tests instead of the redis-backed default).
    if spec.ctrl_factory is not None:
        cfg.ctrl = spec.ctrl_factory(cfg)
    else:
        # Otherwise, drop interactive controllers (keyboard / joystick / unitree
        # pad) that need a display or physical hardware. Keep data-feeding
        # controllers (motion playback, BeyondMimic ref ctrl, etc.) since the
        # policy relies on them for observation streams.
        _INTERACTIVE_CTRLS = {"KeyboardCtrl", "JoystickCtrl", "UnitreeCtrl"}
        cfg.ctrl = [c for c in (cfg.ctrl or []) if c.ctrl_type not in _INTERACTIVE_CTRLS]
    # Force the motion-controller GUI off (Tkinter window from a background
    # thread is unsupported on macOS and unwanted in CI).
    for ctrl_cfg in cfg.ctrl:
        if hasattr(ctrl_cfg, "motion_ctrl_gui"):
            ctrl_cfg.motion_ctrl_gui = False

    # --- Apply policy overrides (e.g. tracker motion path) ---
    policy_cfgs = []
    if hasattr(cfg, "policy") and cfg.policy is not None:
        policy_cfgs.append(cfg.policy)
    if hasattr(cfg, "policies"):
        policy_cfgs.extend(cfg.policies)
    if hasattr(cfg, "loco_policy"):
        policy_cfgs.append(cfg.loco_policy)
    if hasattr(cfg, "mimic_policies"):
        policy_cfgs.extend(cfg.mimic_policies)
    for key, value in spec.policy_overrides.items():
        for pc in policy_cfgs:
            if hasattr(pc, key):
                setattr(pc, key, value)

    # --- Build the pipeline ---
    pipeline_type = cfg.pipeline_type
    pipeline_class = getattr(robojudo.pipeline, pipeline_type)
    pipeline: Pipeline = pipeline_class(cfg=cfg)

    # The pipeline's __init__ runs 10 dry-run steps for self-check that may
    # touch the monitor's accumulators; reset them so step 0 is "real".
    monitor = pipeline.env.stability_monitor  # type: ignore[attr-defined]
    if monitor is None:
        return SimTestResult(
            cfg_name=spec.cfg_name,
            display_name=spec.display_name,
            passed=False,
            reason="stability monitor was not installed on env (cfg path bug?)",
        )
    monitor.reset()

    # --- Wire scripted commands ---
    _patch_command_injection(pipeline, spec.startup_commands, pipeline.dt)

    # --- Optional action sabotage (negative specs) ---
    if spec.action_sabotage is not None:
        _patch_action_sabotage(pipeline, spec.action_sabotage)

    # --- Step loop ---
    total_seconds = spec.settle_seconds + spec.run_seconds
    total_steps = int(round(total_seconds / pipeline.dt))
    monitor_failed = False
    try:
        for _ in range(total_steps):
            pipeline.step()
            if monitor.failed:
                monitor_failed = True
                break
    finally:
        try:
            pipeline.env.shutdown()
        except Exception:
            logger.exception("[SimRunner] env.shutdown raised; ignoring")

    elapsed = monitor.snapshot()["elapsed"]

    if spec.expect_failure:
        # Negative spec: passing means the monitor correctly detected the
        # sabotage. Failing means the robot stayed upright despite our best
        # attempt to topple it — that's a monitor coverage gap.
        if monitor_failed:
            return SimTestResult(
                cfg_name=spec.cfg_name,
                display_name=spec.display_name,
                passed=True,
                elapsed=elapsed,
                # Surfaced for the "ok" line so the operator can see *what*
                # the monitor caught, confirming it was the intended failure
                # mode and not a false trip.
                failure_reason=f"monitor correctly caught: {monitor.failure_reason}",
            )
        return SimTestResult(
            cfg_name=spec.cfg_name,
            display_name=spec.display_name,
            passed=False,
            elapsed=elapsed,
            failure_reason=(
                "expected the stability monitor to fire under sabotage, but "
                "it never did — thresholds may be too loose or the sabotage "
                "function is too gentle."
            ),
        )

    # Positive spec: monitor must NOT have tripped.
    if monitor_failed:
        return SimTestResult(
            cfg_name=spec.cfg_name,
            display_name=spec.display_name,
            passed=False,
            elapsed=elapsed,
            failure_reason=monitor.failure_reason,
        )
    return SimTestResult(
        cfg_name=spec.cfg_name,
        display_name=spec.display_name,
        passed=True,
        elapsed=elapsed,
    )


def _patch_action_sabotage(pipeline: Pipeline, sabotage) -> None:
    """Wrap ``env.step`` so each pd_target passes through ``sabotage`` first."""
    env = pipeline.env
    original_step = env.step

    def patched_step(pd_target, hand_pose=None):
        return original_step(sabotage(pd_target), hand_pose=hand_pose)

    env.step = patched_step  # type: ignore[method-assign]
