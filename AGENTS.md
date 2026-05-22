# AGENTS.md — Working contract for AI coding agents

This file is the first thing an AI coding agent should read when working in
this repository. It captures (a) what RoboJuDo *is* in one paragraph, (b) the
hard rules that survive across tasks, and (c) the code conventions an agent
must reproduce. Developers can append project-specific rules in the marked
section at the bottom — that section overrides anything above.

For human-facing contribution / test workflow see
[`CONTRIBUTING.md`](CONTRIBUTING.md). For deeper architectural context see the per-module docs in [`docs/`](docs/).

---

## 1. What this repo is

RoboJuDo is a **deployment** framework for humanoid robots. The code in this
repo runs on real Unitree G1 / H1 hardware as well as in MuJoCo sim. The
current code on `release` has been validated on physical robots; sim is the
fast path, real hardware is the truth.

Two consequences flow from that:

- **Real-robot validation is slow, manual, and not automatable by an agent.**
  You cannot prove your change works on hardware. Treat that as a hard
  constraint: keep the workflow / dataflow boring and recognizable.
- **The framework's selling point is plug-and-play modularity** — every
  module type (policy, environment, controller, pipeline, config) is loaded
  via a registry and freely composable. Preserve that modularity in every
  change.

## 2. Setup & commands

Install and run via the README — `pip install -e .` plus optional submodules
through `python submodule_install.py`. Active dev install: `pip install -e
".[dev]"` to pull `ruff` and `pre-commit`.

Always-run commands:

```bash
ruff check robojudo/ tests/        # required, no exceptions
ruff format --check robojudo/ tests/

python -m unittest tests.test_full_imports   # seconds, runs on every change
python -m tests.sim.run_all                  # minutes, runs before commit
```

`tests.sim.run_all` is the heavyweight pre-commit gate; see
`CONTRIBUTING.md § Testing` for details. It is *the* substitute for real-robot
validation an agent can run on its own — never skip it.

## 3. Known gotchas

> _Drop new env-prep / setup gotchas here as you hit them. When this list
> grows past a handful of entries, promote it to its own document._

- **`python submodule_install.py` calls `pip install -e ...` directly.**
  That resolves to whatever `pip` is on `$PATH`, which under a `uv venv`
  is usually *not* this project's venv — the submodule lands in a sibling
  install and `import phc` fails. TODO: teach `submodule_install.py` to
  honor `VIRTUAL_ENV` / call `uv pip install` when available. Workaround
  for now: run the install (so the git submodule + patches + addons land
  on disk), then `uv pip install -e third_party/<name>` manually with
  `--python /path/to/.venv/bin/python` to land the package in the right
  venv.

## 4. Architecture in 30 seconds

Three-tier composition, orchestrated by a `Pipeline`:

```
Pipeline (step / prepare loop)
 ├── Environment        — MuJoCo / Unitree real / Dummy
 ├── Controller(s)      — joystick, keyboard, motion, redis, ...
 └── Policy             — ONNX/JIT network + obs assembly
```

Each module type lives behind a `Registry` (see
`robojudo/utils/module_registry.py`). New classes register themselves with
`@<type>_registry.register`; configs refer to them by `str` (e.g.
`policy_type: "UnitreePolicy"`). The `__getattr__` on each package
lazy-imports modules on first access, so optional dependencies do not break
import time.

Commands flow through `CtrlManager.get_ctrl_data()` as a de-duplicated
`COMMANDS` list inside the `ctrl_data` `Box`. Tokens like `[MOTION_RESET]`,
`[SHUTDOWN]`, `[POLICY_SWITCH],<id>` are the inter-module language.

## 5. Hard rules

These rules are non-negotiable. Violating them is a regression even if tests
pass.

### 5.1 Don't lightly touch workflow / dataflow

The pipeline step loop, controller command bus, observation/action shapes,
and policy lifecycle (`reset` / `reset_alignment` / `post_step_callback`)
are validated on real hardware. Any change to these surfaces must:

- be opt-in (gated by a new cfg flag, defaulting to current behavior), or
- be invisible to existing policies (the existing call sites still get the
  same return shapes / side effects).

If you cannot frame a change that way, write up the proposal and stop —
ask for confirmation. *"It's just a small refactor"* is the failure mode.

### 5.2 Code stability and quality over cleverness

Real-robot verification is complex and not agent-driven. To compensate,
agents must:

- prefer the smallest diff that solves the problem,
- not refactor adjacent code that the task didn't ask for,
- not introduce abstractions on speculation ("we might need this later"),
- treat all warnings (ruff, pyright) as errors and fix them in the same
  change, not in a follow-up.

### 5.3 Type-check & lint, every commit

The entire codebase passes `ruff check` (config in `pyproject.toml`: rules
`E, F, I, B, UP`, line-length 120) and type-checks under pyright. New code
must do the same. Where pyright cannot see through a C-extension (e.g.
`mujoco.mj_step`), use a *targeted* `# pyright: ignore[reportAttributeAccessIssue]`
matching the existing style — never a blanket ignore.

### 5.4 All configs are Pydantic

Every config class extends `robojudo.config.Config` (which extends
`pydantic.BaseModel`). This is not optional. Concretely:

- Fields must be typed (`name: str`, `freq: int = 50`, `policy: PolicyCfg`).
- Cross-field invariants use `@model_validator(mode="after")` or
  `@field_validator`.
- Derived paths / sizes use `@property`, not `__init__` assignment.
- Optional dependencies on other configs use `T | None = None`.
- Literal-set fields use `Literal["a", "b", ...]`.

Do not introduce `dataclass`, `attrs`, plain dict-configs, or
`SimpleNamespace`. Configs are the public surface — they must validate.

### 5.5 OOP discipline

This codebase leans hard on inheritance + composition; don't fight it.

- Module-type base classes (`Policy`, `Environment`, `Controller`,
  `Pipeline`) are `ABC`s with `@abstractmethod` hooks. Subclasses override
  exact hooks, not unrelated methods.
- Robot-specific cfgs subclass the generic cfg (`G1MujocoEnvCfg(G1EnvCfg,
  MujocoEnvCfg)`); robot specifics belong in the robot folder, not in the
  base.
- Prefer composition (Pipeline composes env + ctrls + policy) over
  multi-inheritance across orthogonal concerns.

## 6. Code conventions (observed from the codebase)

Reproduce these in any new code. If you find yourself wanting to deviate,
that's usually a signal you've misunderstood the existing structure. These
are observations, not hard rules — when a robot SDK glue file legitimately
needs to be longer than the median, that's fine.

**File & function length**

- Median module is ~100–250 lines; the largest implementation file
  (`unitree_env.py` ≈ 430 lines) earns it through SDK glue, not policy
  logic. As a soft target, aim for **< ~300 lines per module**; when a
  file grows past that, split along an existing seam (cfg vs. impl, or
  per-robot subclass) before adding more.
- Functions tend to stay under ~50 lines. Past that, look for an
  extractable helper or a method on the class.

**Naming**

- Files: `snake_case.py`. Cfg files end in `_cfg.py` or `_cfgs.py` (the
  plural form is for the aggregator module).
- Classes: `PascalCase`. Cfg classes end in `Cfg`
  (`UnitreePolicyCfg`, `G1MujocoEnvCfg`). Per-robot configs prefix the
  robot (`G1...`, `H1...`).
- Vocabulary is terse and consistent: `cfg`, `ctrl`, `env`, `dof`, `obs`,
  `dt`, `freq`. Use the same spellings — don't introduce `controller_cfg`
  next to `cfg_ctrl`.

**Registration & imports**

- A new module class registers via the decorator
  (`@policy_registry.register`) at the top of its class definition. For
  lazy loading, add `<registry>.add("Name", ".module_path")` in the
  package's `__init__.py` next to the existing entries.
- Per-robot configs are surfaced by importing them in `robojudo/config/<robot>/__init__.py`
  and listing them in the robot's top-level cfg file. The `# noqa: F401`
  pattern there is intentional.

**Logging**

- Every module that logs uses `logger = logging.getLogger(__name__)` at
  module top. Use `logger.info / debug / warning / error`. No `print`
  inside `robojudo/` (the registry's `print` is the documented exception).

**Numpy & torch**

- `np.asarray(...)` over `np.array(...)` when the input may already be an
  array.
- `dtype=np.float32` for observation tensors going into ONNX/JIT.
- Hand-aligned arrays use `# fmt: off` / `# fmt: on` to opt out of `ruff
  format` so the visual layout survives.

**Comments**

- Docstrings on abstract base classes and on non-obvious public methods.
  Inline comments call out *why* something is the way it is (e.g.
  `# MuJoCo body id 0 is the world body`). Do not narrate the code.

## 7. Verification flow

Before declaring work done:

1. `ruff check robojudo/ tests/` — must be clean.
2. `python -m unittest tests.test_full_imports` — must be green.
3. `python -m tests.sim.run_all` — every non-skipped spec must pass; if you
   added or changed a policy, add a corresponding `PolicySimSpec` (see
   `CONTRIBUTING.md § 4.2`).
4. If you changed `RlPipeline`, the controller bus, or any robot env, also
   re-read your diff against §5.1 *Don't lightly touch workflow / dataflow*.
   Hardware can't tell you you broke it.

## 8. Definition of done

A change is done when:

- the rules in §5 hold,
- the conventions in §6 are reproduced in the new code,
- §7 verification passes locally,
- the commit message explains *why* (not just what), in the style of
  recent commits on `release`.

---

## 9. Developer-supplied rules

> _This section is reserved for human maintainers to append project-specific
> guidance. Anything written below this line takes precedence over §1–8._

<!-- Add rules below this comment. Suggested format: a level-3 heading per
     rule, with a short rationale so future agents understand the why. -->
