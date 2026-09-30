"""Shared file handling for manuscript generators."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from flybrain.paths import TEMP, allowed_derived_write, portable_metadata, portable_path, writable_output


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _relative_keys(value):
    if isinstance(value, dict):
        return {(portable_path(key) if isinstance(key, str) and Path(key).is_absolute() else key):
                _relative_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relative_keys(item) for item in value]
    return value


def write_json(path, value, *, relative_keys=False, ensure_ascii=True):
    value = portable_metadata(value)
    if relative_keys:
        value = _relative_keys(value)
    writable_output(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=ensure_ascii) + "\n")


def directory(path):
    path = allowed_derived_write(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def temporary_directory(path):
    path = allowed_derived_write(path)
    if not path.is_relative_to(TEMP.resolve()):
        raise ValueError(f"Generator caches must stay under {TEMP}")
    return directory(path)
