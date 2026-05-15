"""CLI runner for the heavyweight sim policy tests.

Usage::

    # Run all registered specs (skips ones whose optional deps are missing):
    python -m tests.sim.run_all

    # Run a single config:
    python -m tests.sim.run_all -c g1

    # List the specs without running them:
    python -m tests.sim.run_all --list

Exit code is non-zero if any spec failed. Skipped specs (e.g. missing optional
dependencies) are NOT counted as failures, since the test must be runnable on
a clean default install.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .runner import run_spec
from .specs import SPECS, PolicySimSpec


def _format(result, spec: PolicySimSpec) -> str:
    tag = "PASS" if result.passed and not result.skipped else "SKIP" if result.skipped else "FAIL"
    label = spec.display_name or spec.cfg_name
    head = f"[{tag}] {label:<32s}"
    if result.skipped:
        return f"{head}  skipped: {result.reason}"
    if not result.passed:
        return f"{head}  {result.failure_reason or result.reason} (at t={result.elapsed:.2f}s)"
    # Positive-pass = silent ok. Negative-pass = surface what the monitor caught.
    if spec.expect_failure and result.failure_reason:
        return f"{head}  ok ({result.elapsed:.2f}s) — {result.failure_reason}"
    return f"{head}  ok ({result.elapsed:.2f}s)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", dest="configs", action="append", default=None,
        help="Run only the named config(s). Repeatable.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List specs without running."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    specs = SPECS
    if args.configs:
        wanted = set(args.configs)
        # Accept either cfg_name or display_name for `-c`.
        specs = [s for s in SPECS if s.cfg_name in wanted or (s.display_name in wanted)]
        all_labels = {s.cfg_name for s in SPECS} | {s.display_name for s in SPECS if s.display_name}
        missing = wanted - all_labels
        if missing:
            print(f"Unknown config(s): {sorted(missing)}", file=sys.stderr)
            return 2

    if args.list:
        for s in specs:
            label = s.display_name or s.cfg_name
            print(f"{label:<32s}  {s.description}")
        return 0

    n_pass = n_skip = n_fail = 0
    lines: list[str] = []
    for spec in specs:
        result = run_spec(spec)
        line = _format(result, spec)
        print(line, flush=True)
        lines.append(line)
        if result.skipped:
            n_skip += 1
        elif result.passed:
            n_pass += 1
        else:
            n_fail += 1

    print("\n=== Summary ===")
    for line in lines:
        print(line)
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
