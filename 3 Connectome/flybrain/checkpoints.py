"""Load original or losslessly deduplicated Iris checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .paths import repo_path

STORAGE_FORMAT = "flybrain-shared-tensors-v1"
_SHARED_FIELDS = frozenset({"initial_gain", "plastic_edge_indices"})


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _storage_record(path, current):
    sidecar = path.with_suffix(".storage.json")
    if not sidecar.exists():
        return None
    record = json.loads(sidecar.read_text())
    if (record.get("schema_version") != 1
            or not _is_sha256(record.get("original_checkpoint_sha256"))
            or record.get("checkpoint_sha256") != current):
        raise ValueError(f"Checkpoint storage SHA-256 mismatch: {sidecar}")
    references = record.get("shared_tensors")
    if not isinstance(references, dict) or not references or not set(references) <= _SHARED_FIELDS:
        raise ValueError(f"Invalid shared checkpoint fields: {sidecar}")
    for reference in references.values():
        if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                or not reference["path"] or Path(reference["path"]).is_absolute()
                or not _is_sha256(reference.get("sha256"))
                or not _is_sha256(reference.get("tensor_sha256"))):
            raise ValueError(f"Invalid shared tensor reference: {sidecar}")
    return record


def _shared_path(checkpoint, reference):
    shared = (checkpoint.parent / reference["path"]).resolve()
    bundle = next((parent for parent in checkpoint.parents if parent.name == "payload"), None)
    if bundle is not None and not shared.is_relative_to(bundle):
        raise ValueError(f"Shared tensor path escapes the artifact payload: {shared}")
    return shared


def _verified_checkpoint(path, expected_sha256=None):
    path = repo_path(path).resolve()
    current = _sha256(path)
    record = _storage_record(path, current)
    accepted = {current}
    if record is not None:
        accepted.add(record["original_checkpoint_sha256"])
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or expected_sha256.lower() not in accepted:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    if record is not None:
        for reference in record["shared_tensors"].values():
            shared = _shared_path(path, reference)
            if _sha256(shared) != reference["sha256"]:
                raise ValueError(f"Shared tensor SHA-256 mismatch: {shared}")
    return path, current, record


def verify_checkpoint(path, expected_sha256=None):
    """Verify stored bytes and shared dependencies without deserializing tensors."""
    return _verified_checkpoint(path, expected_sha256)[0]


def checkpoint_source_sha256(path):
    """Return the verified original-file hash used by historical provenance."""
    _, current, record = _verified_checkpoint(path)
    return current if record is None else record["original_checkpoint_sha256"]


def load_checkpoint(path, expected_sha256=None):
    """Hydrate the original checkpoint dictionary on CPU without changing tensors."""
    import torch

    path, _, record = _verified_checkpoint(path, expected_sha256)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    storage_format = checkpoint.get("_storage_format")
    if storage_format is None:
        if record is not None or "_shared_tensors" in checkpoint:
            raise ValueError(f"Checkpoint storage metadata does not match its contents: {path}")
        return checkpoint
    if storage_format != STORAGE_FORMAT or record is None:
        raise ValueError(f"Missing or unsupported checkpoint storage metadata: {path}")
    references = checkpoint.get("_shared_tensors")
    if references != record["shared_tensors"] or any(name in checkpoint for name in references):
        raise ValueError(f"Checkpoint shared tensor metadata does not match: {path}")
    hydrated = {key: value for key, value in checkpoint.items()
                if key not in {"_storage_format", "_shared_tensors"}}
    for name, reference in references.items():
        shared = _shared_path(path, reference)
        tensor = torch.load(shared, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Shared checkpoint value must be a tensor: {shared}")
        if (("dtype" in reference and reference["dtype"] != str(tensor.dtype))
                or ("shape" in reference and reference["shape"] != list(tensor.shape))):
            raise ValueError(f"Shared tensor dtype or shape mismatch: {shared}")
        actual = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
        if actual != reference["tensor_sha256"]:
            raise ValueError(f"Shared tensor content SHA-256 mismatch: {shared}")
        hydrated[name] = tensor
    return hydrated
