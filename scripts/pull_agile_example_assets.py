#!/usr/bin/env python3
"""Pull pinned AGILE example assets (IO YAML + checkpoint) into RoboJuDo/assets/.

Two policies are mirrored — one per AGILE source folder:

  - ``velocity_height_g1`` — RNN-LSTM recurrent student (TorchScript)
  - ``velocity_g1``        — MLP with per-term ``history_length=5`` (TorchScript)

These checkpoints are not committed into the RoboJuDo repo (see ``.gitignore``).
Users either run this script once before launching sim, or let test/sim entry
points auto-invoke :func:`ensure_agile_assets` when a sibling ``WBC_AGILE``
checkout is detected.

Source modes:

  1. ``--local <path>`` — copy from a local ``WBC_AGILE/agile/data/policy``
     directory.
  2. ``--repo <url> --commit <sha>`` — ``git clone --depth 1`` into a temp
     directory and check out the pinned commit.

Layout populated::

    assets/models/g1/agile/
        velocity_height_g1/{policy.yaml, policy.pt, metadata.yaml}
        velocity_g1/{policy.yaml, policy.pt, metadata.yaml}

This script never imports the ``agile`` Python package.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "models" / "g1" / "agile"

# Mapping (label) → (src yaml relative to <policy dir>, src checkpoint, arch metadata)
# ``src_onnx`` is optional — if present in the source folder, it is also pulled
# alongside ``policy.pt`` so the user can swap between TorchScript and ONNX at
# runtime by editing the cfg's ``checkpoint_path``.
POLICIES: dict[str, dict] = {
    "velocity_height_g1": {
        "src_yaml": "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml",
        "src_ckpt": "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt",
        "src_onnx": "velocity_height_g1/unitree_g1_velocity_height_recurrent_student.onnx",
        "architecture": "RNN-LSTM (TorchScript / ONNX)",
        "rnn_hidden_shape": [2, 1, 128],
    },
    "velocity_g1": {
        "src_yaml": "velocity_g1/unitree_g1_velocity_history.yaml",
        "src_ckpt": "velocity_g1/unitree_g1_velocity_history.pt",
        "src_onnx": None,  # this policy is not exported to ONNX upstream
        "architecture": "MLP (TorchScript) + per-term history_length=5",
        "rnn_hidden_shape": None,
    },
}


def _copy_one(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("[copy] %s -> %s", src, dst)


def _copy_one_optional(src: Path, dst: Path):
    if not src.exists():
        logger.info("[skip] %s (not present)", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("[copy] %s -> %s", src, dst)
    return True


def _copy_from_local(policy_dir: Path, dest_root: Path):
    for label, info in POLICIES.items():
        _copy_one(policy_dir / info["src_yaml"], dest_root / label / "policy.yaml")
        _copy_one(policy_dir / info["src_ckpt"], dest_root / label / "policy.pt")
        if info.get("src_onnx"):
            _copy_one_optional(policy_dir / info["src_onnx"], dest_root / label / "policy.onnx")
        _write_policy_metadata(dest_root / label, label, info, source=str(policy_dir), commit=None)


def _copy_from_remote(repo: str, commit: str, dest_root: Path):
    with tempfile.TemporaryDirectory() as tmp:
        clone_path = Path(tmp) / "WBC_AGILE"
        logger.info("[clone] %s -> %s", repo, clone_path)
        subprocess.check_call(["git", "clone", "--quiet", repo, str(clone_path)])
        subprocess.check_call(["git", "-C", str(clone_path), "checkout", "--quiet", commit])
        policy_dir = clone_path / "agile" / "data" / "policy"
        for label, info in POLICIES.items():
            _copy_one(policy_dir / info["src_yaml"], dest_root / label / "policy.yaml")
            _copy_one(policy_dir / info["src_ckpt"], dest_root / label / "policy.pt")
            if info.get("src_onnx"):
                _copy_one_optional(policy_dir / info["src_onnx"], dest_root / label / "policy.onnx")
            _write_policy_metadata(dest_root / label, label, info, source=repo, commit=commit)


def _write_policy_metadata(dest_dir: Path, label: str, info: dict, source: str, commit: str | None):
    metadata = {
        "source": source,
        "commit": commit,
        "label": label,
        "policy": {
            "path": "policy.pt",
            "yaml": "policy.yaml",
            "architecture": info["architecture"],
            "rnn_hidden_shape": info["rnn_hidden_shape"],
        },
    }
    out = dest_dir / "metadata.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(metadata, sort_keys=False))
    logger.info("[meta] wrote %s", out)


def _is_complete(dest_root: Path) -> bool:
    """Return True if all expected policy.{yaml,pt} files exist under ``dest_root``."""
    for label in POLICIES:
        if not (dest_root / label / "policy.yaml").exists():
            return False
        if not (dest_root / label / "policy.pt").exists():
            return False
    return True


def _autodetect_local_policy_dir() -> Path | None:
    """Best-effort search for a sibling ``WBC_AGILE/agile/data/policy`` directory.

    Looked up in order:
      1. ``$WBC_AGILE_DIR/agile/data/policy``
      2. ``<REPO_ROOT>/../WBC_AGILE/agile/data/policy`` (typical workspace layout)
      3. ``<REPO_ROOT>/../../WBC_AGILE/agile/data/policy``
    """
    env_root = os.environ.get("WBC_AGILE_DIR")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root) / "agile" / "data" / "policy")
    candidates.append(REPO_ROOT.parent / "WBC_AGILE" / "agile" / "data" / "policy")
    candidates.append(REPO_ROOT.parent.parent / "WBC_AGILE" / "agile" / "data" / "policy")
    for c in candidates:
        if c.exists():
            return c
    return None


def ensure_agile_assets(verbose: bool = True) -> Path:
    """Make sure AGILE deploy assets exist under :data:`ASSETS_DIR`.

    If anything is missing, look for a sibling ``WBC_AGILE`` checkout (or
    ``$WBC_AGILE_DIR``) and copy the two policies in. Raises ``FileNotFoundError``
    with a clear message if no source is available — the caller (test, sim
    entrypoint) can then ask the user to run this script manually with
    ``--repo/--commit``.
    """
    if _is_complete(ASSETS_DIR):
        return ASSETS_DIR

    local = _autodetect_local_policy_dir()
    if local is None:
        raise FileNotFoundError(
            "AGILE deploy assets are missing under "
            f"{ASSETS_DIR}. No sibling WBC_AGILE checkout was found.\n"
            "Run scripts/pull_agile_example_assets.py with --repo <url> --commit <sha>, "
            "or set the WBC_AGILE_DIR env var to point at your WBC_AGILE checkout."
        )

    if verbose:
        logger.warning("[ensure_agile_assets] auto-pulling from %s", local)
    _copy_from_local(local, ASSETS_DIR)
    return ASSETS_DIR


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--local", type=Path, help="Path to local WBC_AGILE/agile/data/policy directory.")
    src.add_argument("--repo", help="Remote WBC_AGILE repo URL.")
    parser.add_argument("--commit", help="Pinned commit SHA to checkout (required with --repo).")
    args = parser.parse_args(argv)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if args.local is not None:
        _copy_from_local(args.local, ASSETS_DIR)
    elif args.repo:
        if not args.commit:
            parser.error("--commit is required with --repo")
        _copy_from_remote(args.repo, args.commit, ASSETS_DIR)
    else:
        # No source flags — fall back to auto-detect (developer convenience).
        ensure_agile_assets(verbose=True)

    print(f"\nAGILE example assets ready under {ASSETS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
