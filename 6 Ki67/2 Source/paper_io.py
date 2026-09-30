"""Read archived evidence and write new publication editions outside the archive."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from flybrain.paths import TEMP, allowed_derived_write

if __package__:
    from .common import WORKSPACE, destination as write_destination, portable_metadata, resolve_output_reference, writable_location
else:
    from common import WORKSPACE, destination as write_destination, portable_metadata, resolve_output_reference, writable_location


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path = write_destination(allowed_derived_write(path))
    content = json.dumps(portable_metadata(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def default_destination(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return WORKSPACE / "paper" / f"{name}-{stamp}"


def source_outputs(path: Path) -> Path:
    """Resolve read-only run storage without creating or updating it."""
    path = resolve_output_reference(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Source outputs are missing: {path}")
    return path


def prepare_destination(path: Path) -> Path:
    """Create a fresh edition after rejecting raw inputs and archived outputs."""
    path = writable_location(allowed_derived_write(path))
    if path.exists():
        raise ValueError("Historical editions cannot be overwritten; choose a new destination")
    path.mkdir(parents=True, exist_ok=False)
    return path


def configure_plot_cache() -> None:
    for name in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        cache = write_destination(TEMP / "ki67" / "paper-cache" / name.lower() / ".location").parent
        os.environ[name] = str(cache)
