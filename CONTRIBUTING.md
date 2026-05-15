# Contributing to RoboJuDo

Thanks for taking the time to contribute. RoboJuDo aims to stay
plug-and-play, so most useful contributions slot into one of the existing
modular slots — a new policy, a new environment, a new controller — without
rewiring the framework. The sections below describe how to set up a dev
environment and what we expect of each kind of change.

## Table of Contents

1. [Development setup](#1-development-setup)
2. [Project layout](#2-project-layout)
3. [Adding a new module](#3-adding-a-new-module)
4. [Testing](#4-testing)
5. [Style and conventions](#5-style-and-conventions)
6. [Submitting changes](#6-submitting-changes)

---

## 1. Development setup

See [README § Easy Setup](README.md#%EF%B8%8Feasy-setup) for the canonical
install path (Python 3.11, optional submodules via `submodule_install.py`).

For development we additionally recommend:

```bash
pip install -e ".[dev]"   # ruff + pre-commit
pre-commit install
```

> _TODO: expand with editor / linter configuration tips._

## 2. Project layout

A 30-second tour of the repo:

```
robojudo/
  pipeline/      orchestrates the step/prepare loop
  environment/   MuJoCo / real / dummy backends
  controller/    joystick, keyboard, motion / redis ctrls
  policy/        the actual networks + obs assembly
  config/        per-robot configs (g1, h1, ...) and the cfg registry
tests/
  test_full_imports.py   fast import smoke tests
  sim/                   heavyweight sim policy tests (see §4)
docs/
  policy.md      authoritative per-policy usage notes
  controller.md, environment.md, ...
```

For deeper architectural context (registries, command bus, pipeline state
machine), read [`docs/agent/AGENT-ANALYSIS-0514.md`](docs/agent/AGENT-ANALYSIS-0514.md).

> _TODO: link to a dedicated architecture doc once it lands._

## 3. Adding a new module

The framework is intentionally easy to extend. Each module type has its own
per-policy / per-env / per-ctrl how-to in `docs/`:

- **New policy** — read [`docs/policy.md`](docs/policy.md) for the abstract
  interface and per-policy notes, then register a `Policy` subclass under
  `robojudo/policy/` and a `PolicyCfg` under `robojudo/policy/policy_cfgs.py`.
  Every new policy must come with a sim test spec — see §4.2.
- **New controller** — see [`docs/controller.md`](docs/controller.md).
- **New environment** — see [`docs/environment.md`](docs/environment.md).
- **New robot** — add a config tree under `robojudo/config/<robot>/`
  mirroring `g1/` or `h1/`, plus a `<robot>_default_sim_stability_cfg()`
  factory (see §4.2).

> _TODO: short worked example for each module type._

## 4. Testing

Two layers, with very different costs:

| Layer | Cost | When to run |
| --- | --- | --- |
| Import smoke (`tests/test_full_imports.py`) | seconds | every change |
| Sim policy tests (`tests/sim/`) | minutes | before any commit that touches `policy/`, `pipeline/`, `environment/`, or robot configs |

### 4.1 Fast smoke tests

```bash
python -m unittest tests.test_full_imports
```

This checks every registered config / controller / env / policy / pipeline
imports cleanly. It tolerates optional-dependency failures (e.g. missing
`phc` / `redis` / Unitree SDK) by design — those are gated through
`submodule_cfg.yaml`.

### 4.2 Sim policy tests (pre-commit)

```bash
python -m tests.sim.run_all          # all specs
python -m tests.sim.run_all -c g1    # one spec
python -m tests.sim.run_all --list   # what's registered
```

Each registered policy ships with a `PolicySimSpec` in
[`tests/sim/specs.py`](tests/sim/specs.py). The harness boots MuJoCo
headlessly, runs the policy for a few seconds, and asserts that an opt-in
fall detector (`SimStabilityMonitor`, attached only to the test env) did not
trip. Specs whose optional dependencies are missing are *skipped*, not
failed, so the suite stays runnable on a clean install.

When adding a new policy, add a spec that mirrors the documented offline
workflow from `docs/policy.md` — same controller, same motion source, same
startup command sequence (e.g. tracker policies need `[MOTION_RESET]` after
settle). When the docs prescribe an offline controller that differs from the
config-shipped one (e.g. TwistPolicy: docs use `MotionTwistCtrl`, config
ships `TwistRedisCtrl`), use `PolicySimSpec.ctrl_factory` to swap it in.

The suite also includes a few `fail/...` specs that deliberately sabotage a
policy to verify the detector still catches catastrophic failure. If you
change `SimStabilityMonitor` or its per-robot thresholds, the `fail/...`
specs are the contract — don't relax them, fix the monitor.

The sim test is also exposed as a `unittest.TestCase`:

```bash
RUN_SIM_TESTS=1 python -m unittest tests.sim.test_sim_policies
```

The env-var gate prevents accidental `unittest discover` runs from booting
MuJoCo for every policy.

> _Important:_ do not loosen `RlPipeline.safety_check` to make a sim test
> pass — `safety_check` protects real hardware. The sim monitor is the loose
> one. Likewise, never set `headless=True` or `sim_stability=...` on a
> registered config; those fields exist for the test harness to flip at
> runtime.

## 5. Style and conventions

- Python 3.11+, formatted by `ruff` (config in `pyproject.toml`).
- Pre-commit hooks run `ruff check` and `ruff format`. CI rejects
  unformatted code.

> _TODO: docstring conventions, type-hint expectations, naming._

## 6. Submitting changes

1. Branch off `release` using `dev/<topic>` (e.g. `dev/sim-test-harness`).
2. Run the fast smoke tests on every iteration; run the sim tests before
   you push.
3. Open a PR; fill in a brief description and link any related issue.

> _TODO: commit message conventions, PR review process, CLA._
