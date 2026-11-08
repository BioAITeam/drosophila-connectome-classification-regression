"""Locate a checkout and separate published inputs from fresh derived results."""
from __future__ import annotations

import os
from pathlib import Path


def repository_root() -> Path:
    configured = os.environ.get("FLYBRAIN_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "3 Connectome").is_dir():
            return candidate
    raise RuntimeError("Run from a restored checkout or set FLYBRAIN_REPO to its root")


def relative_path(value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Artifact paths must be relative to the repository root")
    return value


REPO = repository_root()
ARCHIVE = Path("6 Ki67/5 Outputs")
PUBLISHED = ARCHIVE / "ml-baselines"
PATIENT_DISJOINT = ARCHIVE / "shid-v2-20261031/binary"
WORKSPACE = relative_path(os.environ.get("KI67_ML_WORKSPACE", "6 Ki67/6 Workspace/ml-baselines"))
if WORKSPACE.parts[:2] != ("6 Ki67", "6 Workspace"):
    raise ValueError("KI67_ML_WORKSPACE must be inside 6 Ki67/6 Workspace")


def writable(path: str | Path, *, repo: Path = REPO) -> Path:
    """Reject writes to archived outputs, raw inputs or paths outside Workspace."""
    candidate = Path(path).resolve()
    if not candidate.is_relative_to((repo / "6 Ki67/6 Workspace").resolve()):
        raise ValueError(f"ML derived results must be written under 6 Ki67/6 Workspace: {candidate}")
    return candidate
