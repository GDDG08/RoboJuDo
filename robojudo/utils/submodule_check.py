"""Lightweight, import-time checks that compiled submodules are up to date.

Submodules such as ``unitree_cpp`` are installed editable, but a ``git pull`` on
the main repo that bumps a submodule pointer does NOT rebuild the compiled
extension (``-e`` only re-links Python sources, not C/C++ binaries). That can
silently pair new RoboJuDo code with a stale binary, e.g. 1.0.4 sources running
against a 1.0.3 build that lacks ``torso_imu_state``.

We compare the *installed* distribution version against the version pinned in the
submodule's checked-out ``pyproject.toml`` (which a ``git pull`` does update) and
warn on mismatch. Submodules that are not installed are skipped silently, so
users who don't need them are never nagged.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

logger = logging.getLogger(__name__)

# Distribution name -> submodule pyproject.toml, relative to the repo root.
# Only compiled submodules whose binary won't auto-rebuild on `git pull` belong here.
_VERSIONED_SUBMODULES = {
    "unitree_cpp": "packages/unitree_cpp/pyproject.toml",
}

# robojudo/utils/submodule_check.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _source_version(pyproject: Path) -> str | None:
    try:
        with open(pyproject, "rb") as f:
            return tomllib.load(f).get("project", {}).get("version")
    except (OSError, tomllib.TOMLDecodeError):
        return None


def check_submodule_versions() -> None:
    """Warn (never raise) if any installed submodule is older than its source."""
    for dist_name, rel_path in _VERSIONED_SUBMODULES.items():
        try:
            installed = version(dist_name)
        except PackageNotFoundError:
            # Not installed -> the user doesn't use this submodule. Stay quiet.
            continue

        pyproject = _REPO_ROOT / rel_path
        if not pyproject.exists():
            continue  # not a source checkout (e.g. installed from a wheel)

        expected = _source_version(pyproject)
        if expected is None or installed == expected:
            continue

        logger.warning(
            "Submodule '%s' looks out of date: installed %s but source pins %s. "
            "`git pull` does not rebuild compiled submodules, so the binary may be "
            "stale. Re-run `python submodule_install.py %s` to rebuild.",
            dist_name,
            installed,
            expected,
            dist_name,
        )
