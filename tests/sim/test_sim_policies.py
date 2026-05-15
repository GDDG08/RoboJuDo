"""Heavyweight sim-policy tests — pre-commit final check.

Excluded from the default test discovery to keep regular ``unittest`` /
``pytest`` runs fast. Invoke explicitly::

    python -m tests.sim.run_all
    # or one config:
    python -m tests.sim.run_all -c g1_protomotions_tracker

See ``AGENTS.md`` for the intended workflow.
"""

from __future__ import annotations

import os
import unittest

from .runner import run_spec
from .specs import SPECS

# Gate: only runs when explicitly opted-in, so a stray ``python -m unittest
# discover`` from the repo root doesn't boot MuJoCo + ONNX models for every
# registered policy. The CLI entry point in ``tests.sim.run_all`` bypasses
# this gate and is the intended pre-commit invocation.
_OPT_IN = os.environ.get("RUN_SIM_TESTS") == "1"


@unittest.skipUnless(_OPT_IN, "Heavy sim tests — set RUN_SIM_TESTS=1 or use tests.sim.run_all.")
class SimPolicyTests(unittest.TestCase):
    """One sub-test per registered :class:`PolicySimSpec`."""

    def test_all_specs(self):
        failures: list[str] = []
        for spec in SPECS:
            label = spec.display_name or spec.cfg_name
            with self.subTest(spec=label):
                result = run_spec(spec)
                if result.skipped:
                    self.skipTest(f"{label}: {result.reason}")
                if not result.passed:
                    msg = (
                        f"[{label}] FAILED after {result.elapsed:.2f}s: "
                        f"{result.failure_reason or result.reason}"
                    )
                    failures.append(msg)
                    self.fail(msg)
        if failures:
            raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
