"""Atomic artifacts and paths confined to this experiment's output tree."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from flybrain.paths import REPO, TEMP, INTERMEDIATE, PAPER_OUTPUTS, KI67_WORKSPACE, REPOSITORY_TREES, allowed_derived_write, repo_path
if __package__:
    from .tasks import CLASS_TASKS, TASKS
else:
    from tasks import CLASS_TASKS, TASKS
ROOT = REPO / "6 Ki67"
CONFIG = ROOT / "1 Configuration"
SOURCE = ROOT / "2 Source"
OUTPUT = ROOT / "5 Outputs"
WORKSPACE = KI67_WORKSPACE

def resolve_output_reference(path):
    """Resolve stored output paths without rewriting archived provenance."""
    path = Path(path).expanduser()
    if path.is_absolute():
        for index in range(len(path.parts) - 1):
            if path.parts[index:index + 2] in (("ki67", "outputs"), ("6 Ki67", "5 Outputs")):
                return OUTPUT.joinpath(*path.parts[index + 2:]).resolve()
        return repo_path(path).resolve()
    if not path.is_absolute():
        if path.parts[:2] == ("ki67", "outputs"):
            return OUTPUT.joinpath(*path.parts[2:]).resolve()
        if path.parts[:1] == ("outputs",):
            return OUTPUT.joinpath(*path.parts[1:]).resolve()
        if path.parts and path.parts[0].casefold() == "zenodo":
            return (REPO / "zenodo").joinpath(*path.parts[1:]).resolve()
        if path.parts and path.parts[0] in REPOSITORY_TREES:
            return repo_path(path).resolve()
    return (path if path.is_absolute() else ROOT / path).resolve()

def portable_path(path):
    """Serialize a repository artifact independently of its checkout location."""
    path = repo_path(path).absolute()
    try:
        return path.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)

def portable_metadata(value):
    """Store repository paths without changing numerical values or array objects."""
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, str):
        if value.startswith("/"):
            return portable_path(value)
        return value
    if isinstance(value, dict):
        return {portable_metadata(key): portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(portable_metadata(item) for item in value)
    return value

def now():
    return datetime.now(timezone.utc).isoformat()

def writable_location(path):
    path = allowed_derived_write(path)
    if not any(path.is_relative_to(root.resolve()) for root in (WORKSPACE, PAPER_OUTPUTS, INTERMEDIATE, TEMP)):
        raise ValueError(f"Ki67 outputs must remain inside the Workspace, paper outputs, or temporary directories: {path}")
    return path

def destination(path):
    path = writable_location(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def write_json(path, data):
    path = destination(path)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_plain(portable_metadata(data)), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True).encode()).hexdigest()

def atomic_npz(path, **arrays):
    import numpy as np
    path = destination(path)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def atomic_torch(path, value):
    import torch
    path = destination(path)
    fd, temp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
