"""Per-policy sim-test specifications.

Each :class:`PolicySimSpec` records:

- ``cfg_name``: the registered config name (e.g. ``g1``, ``g1_protomotions_tracker``).
- ``run_seconds``: how long to step the pipeline after ``settle_seconds``.
- ``settle_seconds``: warm-up window during which the stability monitor is in
  grace mode (no failures reported); long enough to cover the initial PD
  ramp.
- ``startup_commands``: scripted ``[MOTION_*]`` events emitted into
  ``ctrl_manager`` at fixed wall-clock offsets — replaces the human pressing
  ``R`` to start the motion.
- ``policy_overrides``: optional attribute overrides applied to ``cfg.policy``
  before the pipeline is constructed (e.g. ``motion_path`` for the
  ProtoMotions tracker).
- ``required_modules``: optional list of importable modules. If any are missing
  the spec is skipped (e.g. ``phc`` for motion controllers, ``redis`` for
  twist).
- ``required_assets``: optional list of repo-relative file paths that must
  exist.
- ``skip_reason``: if set, the spec is unconditionally skipped (with the given
  reason in the report).

The goal here is **failure detection**, not policy benchmarking. Specs only
exist for policies that are known to be stable today on existing assets — the
test is intended as a regression net for "did the last refactor break
something obvious."
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets"


@dataclass
class PolicySimSpec:
    cfg_name: str
    description: str
    display_name: str | None = None
    """Human-facing label. Defaults to ``cfg_name`` but can disambiguate two
    specs that share a config (e.g. start vs. switched-to AMO on g1_switch)."""
    run_seconds: float = 4.0
    settle_seconds: float = 1.0
    startup_commands: list[tuple[float, str]] = field(default_factory=list)
    policy_overrides: dict = field(default_factory=dict)
    required_modules: list[str] = field(default_factory=list)
    required_assets: list[str] = field(default_factory=list)
    requires_redis_server: tuple[str, int] | None = None
    """If set, ``(host, port)`` is probed at runtime; skip if not reachable."""
    ctrl_factory: Callable[[Any], list] | None = None
    """If set, called with ``cfg`` to produce the controller list that overrides
    ``cfg.ctrl`` for the test. Use this when the config-shipped controller is not
    the one the policy docs recommend for offline testing (e.g. TwistPolicy ships
    with ``TwistRedisCtrl`` but its docs prescribe ``MotionTwistCtrl`` for local
    .pkl playback).
    """
    action_sabotage: Callable[[np.ndarray], np.ndarray] | None = None
    """If set, called on every step's pd_target before it reaches ``env.step``.
    Used by the *negative* specs that prove the stability monitor actually
    detects catastrophic failure (e.g. multiplying actions by a large factor
    so the robot dives into joint limits and topples).
    """
    expect_failure: bool = False
    """If True, the spec *expects* the stability monitor to trip — passing means
    monitor.failed went True before run_seconds elapsed, failing means the
    robot stayed upright despite the sabotage. Pairs naturally with
    ``action_sabotage``.
    """
    skip_reason: str | None = None


# Common protomotions tracker motion + onnx assets shipped in the repo.
_TRACKER_MOTION = "assets/motions/g1/g1_bones_seed_mini.pt"

# Local-motion PHC pickle used by MotionTwistCtrl (and MotionKungfuBotCtrl). The
# TwistPolicy docs prescribe MotionTwistCtrl for offline tests so we avoid
# needing a redis high-level-motion server.
_TWIST_LOCAL_MOTION_PKL = "assets/motions/g1/phc/kungfubot/Horse-stance_pose.pkl"


def _twist_local_ctrl_factory(_cfg):
    """Per docs/policy.md TwistPolicy: use MotionTwistCtrl (local .pkl via PHC
    MotionLib) instead of TwistRedisCtrl for offline / pre-commit testing."""
    from robojudo.config.g1.ctrl.g1_motion_ctrl_cfg import G1MotionTwistCtrlCfg

    return [G1MotionTwistCtrlCfg()]


SPECS: list[PolicySimSpec] = [
    # ---- Plain locomotion: should stand stably with zero commands ----
    PolicySimSpec(
        cfg_name="g1",
        description="G1 + UnitreePolicy, stand-in-place (no commands).",
        run_seconds=4.0,
    ),
    PolicySimSpec(
        cfg_name="h1",
        description="H1 + UnitreePolicy, stand-in-place (no commands).",
        run_seconds=4.0,
    ),
    # ---- ASAP mimic: motion auto-starts; we just need to survive the clip ----
    PolicySimSpec(
        cfg_name="g1_asap",
        description="G1 + ASAP mimic (default motion), motion auto-runs.",
        run_seconds=5.0,
        settle_seconds=0.5,
    ),
    # ---- BeyondMimic, motion-from-onnx mode (docs/policy.md > BeyondMimicPolicy
    # variant 1: use_motion_from_model=True) ----
    PolicySimSpec(
        cfg_name="g1_beyondmimic",
        description="G1 + BeyondMimic Jump_wose, motion baked into onnx.",
        run_seconds=4.0,
        settle_seconds=0.5,
    ),
    # ---- BeyondMimic, external-ctrl mode (docs variant 2: BeyondmimicCtrl with
    # npz file in assets/motions/g1/beyondmimic/) ----
    PolicySimSpec(
        cfg_name="g1_beyondmimic_with_ctrl",
        description="G1 + BeyondMimic Dance_wose driven by external BeyondmimicCtrl (npz).",
        run_seconds=4.0,
        settle_seconds=0.5,
        required_assets=["assets/motions/g1/beyondmimic/dance1_subject2.npz"],
    ),
    # ---- ASAP decoupled-locomotion variant (docs > AsapPolicy mentions both
    # deepmimic [tested above] and decoupled_locomotion) ----
    PolicySimSpec(
        cfg_name="g1_asap_loco",
        description="G1 + ASAP decoupled-locomotion (zero command, stand-in-place).",
        run_seconds=4.0,
        settle_seconds=0.5,
    ),
    # ---- ProtoMotions tracker: needs [MOTION_RESET] to leave default-pose mode ----
    PolicySimSpec(
        cfg_name="g1_protomotions_tracker",
        description="G1 + ProtoMotions tracker, press R after 1s settle.",
        run_seconds=5.0,
        settle_seconds=1.5,
        startup_commands=[(1.0, "[MOTION_RESET]")],
        required_assets=[_TRACKER_MOTION],
        policy_overrides={
            "motion_path": _TRACKER_MOTION,
            "motion_index": 0,
        },
    ),
    # ---- AMOPolicy as a basic single-policy config ----
    # docs/policy.md > AMOPolicy: with no joystick the policy holds mid-range
    # commands (0 velocity, 0.75 m height) and should stand stably.
    PolicySimSpec(
        cfg_name="g1_amo",
        description="G1 + AMOPolicy stand-in-place (no commands).",
        run_seconds=4.0,
    ),
    # ---- Multi-policy switching: hold the first policy (UnitreePolicy) steady ----
    PolicySimSpec(
        cfg_name="g1_switch",
        description="G1 multi-policy pipeline holding the first policy (UnitreePolicy).",
        run_seconds=4.0,
    ),
    # ---- LocoMimic: stay on loco policy ----
    PolicySimSpec(
        cfg_name="g1_locomimic",
        description="G1 loco<->mimic pipeline, stay on loco.",
        run_seconds=4.0,
    ),
    # ---- Optional-dependency policies. Auto-skipped on a default install ----
    PolicySimSpec(
        cfg_name="g1_h2h",
        description="G1 + H2H student policy (needs phc motion ctrl).",
        run_seconds=4.0,
        required_modules=["phc"],
    ),
    PolicySimSpec(
        cfg_name="g1_kungfubot2",
        description="G1 + KungfuBot general policy (needs phc motion ctrl).",
        run_seconds=4.0,
        required_modules=["phc"],
    ),
    PolicySimSpec(
        cfg_name="g1_twist",
        description=(
            "G1 + Twist policy via MotionTwistCtrl (local PHC .pkl). "
            "docs/policy.md prescribes MotionTwistCtrl for offline testing."
        ),
        run_seconds=4.0,
        required_modules=["phc"],
        required_assets=[_TWIST_LOCAL_MOTION_PKL],
        ctrl_factory=_twist_local_ctrl_factory,
    ),
    # ====================================================================
    # Negative tests — these specs *expect* the SimStabilityMonitor to fire.
    # They prove the detector actually catches catastrophic failures by
    # deliberately sabotaging a known-good policy. If any of them pass
    # without tripping the monitor, the monitor's thresholds are too loose.
    # ====================================================================
    PolicySimSpec(
        cfg_name="g1",
        display_name="fail/g1_action_x5",
        description=(
            "Negative: pd_target multiplied by 5 → joint-limit slam → topple. "
            "Monitor must flag failure."
        ),
        run_seconds=6.0,
        settle_seconds=0.5,
        action_sabotage=lambda a: a * 5.0,
        expect_failure=True,
    ),
    PolicySimSpec(
        cfg_name="g1",
        display_name="fail/g1_action_negated",
        description=(
            "Negative: pd_target sign-flipped → policy actively destabilizes. "
            "Monitor must flag failure."
        ),
        run_seconds=6.0,
        settle_seconds=0.5,
        action_sabotage=lambda a: -a,
        expect_failure=True,
    ),
    PolicySimSpec(
        cfg_name="g1",
        display_name="fail/g1_action_zero",
        description=(
            "Negative: pd_target forced to all zeros → no joint command → "
            "robot collapses under gravity. Monitor must flag failure."
        ),
        run_seconds=6.0,
        settle_seconds=0.5,
        action_sabotage=lambda a: np.zeros_like(a),
        expect_failure=True,
    ),
]
