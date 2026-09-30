"""Portable repository paths and immutable raw-data boundaries.

Archived absolute paths are resolved against the current checkout without
rewriting the original experiment metadata.
"""
from __future__ import annotations

import os
from pathlib import Path

RENAMED_ROOTS = {
    "9 Documentation": "8 Documentation",
    "10 zenodo": "zenodo",
    "11 temp_intermediate": "9 temp_intermediate",
    "12 temp": "10 temp",
}

LEGACY_PREFIXES = tuple(sorted((
    (Path(old).parts, Path(new)) for old, new in {
        "scripts/launch.py": "1 Environment/launch.py",
        "scripts/run_iris_cpu.sh": "5 Iris/2 Execution/run_iris_cpu.sh",
        "scripts/status_round4.py": "5 Iris/2 Execution/status_round4.py",
        "scripts/paper_results": "7 Paper Results/1 Generators",
        "scripts/reproduce_paper.py": "7 Paper Results/reproduce_paper.py",
        "requirements": "1 Environment",
        "data/raw": "2 Data/1 Raw",
        "data/manifests": "2 Data/2 Manifests",
        "configs/datasets": "2 Data/2 Manifests/datasets",
        "data/processed": "2 Data/3 Processed",
        "src": "3 Connectome",
        "artifacts/model-index.csv": "4 Trained Models/1 Catalogs/model-index.csv",
        "artifacts/ki67-model-index.csv": "4 Trained Models/1 Catalogs/ki67-model-index.csv",
        "artifacts/ki67-model-index.json": "4 Trained Models/1 Catalogs/ki67-model-index.json",
        "artifacts/graph-index.csv": "4 Trained Models/1 Catalogs/graph-index.csv",
        "artifacts/figure-provenance.json": "4 Trained Models/1 Catalogs/figure-provenance.json",
        "artifacts": "4 Trained Models",
        "configs/iris": "5 Iris/1 Configuration",
        "configs/benchmark": "5 Iris/1 Configuration/benchmark",
        "examples": "5 Iris/4 Inference",
        "outputs": "5 Iris/5 Outputs",
        "ki67/configs": "6 Ki67/1 Configuration",
        "ki67/src/ki67": "6 Ki67/2 Source",
        "6 Ki67/2 Source/ki67": "6 Ki67/2 Source",
        "ki67/src": "6 Ki67/2 Source",
        "ki67/docs": "6 Ki67/4 Documentation",
        "ki67/outputs": "6 Ki67/5 Outputs",
        "ki67": "6 Ki67",
        "docs": "8 Documentation",
        "ZENODO": "zenodo",
        "temp_intermediate": "9 temp_intermediate",
        "temp": "10 temp",
        **RENAMED_ROOTS,
    }.items()
), key=lambda item: len(item[0]), reverse=True))
REPOSITORY_TREES = ({prefix[0] for prefix, _ in LEGACY_PREFIXES}
                    | {target.parts[0] for _, target in LEGACY_PREFIXES})

_original_root = os.environ.get("FLYBRAIN_ORIGINAL_REPO")
ORIGINAL_REPO = Path(_original_root).expanduser().absolute() if _original_root else None


def _repository_root():
    configured = os.environ.get("FLYBRAIN_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    current = Path.cwd().resolve()
    for candidate in (source_root, current, *current.parents):
        if ((candidate / "pyproject.toml").is_file()
                and (candidate / "3 Connectome/flybrain").is_dir()
                and (candidate / "5 Iris/1 Configuration").is_dir()):
            return candidate
    raise RuntimeError(
        "The experiment checkout was not found. Run from its directory or set FLYBRAIN_REPO."
    )


REPO = _repository_root()
PROJECT = REPO.parent


def _migrated_relative(relative):
    parts = relative.parts
    for prefix, target in LEGACY_PREFIXES:
        if parts[:len(prefix)] == prefix:
            return target.joinpath(*parts[len(prefix):])
    return relative


def _configured_path(variable, default):
    value = Path(os.environ.get(variable, default)).expanduser()
    return _renamed_checkout_path(value).resolve()


def _renamed_checkout_path(value):
    if value.is_absolute() and not value.is_relative_to(REPO):
        return value
    relative = value.relative_to(REPO) if value.is_absolute() else value
    if relative.parts and relative.parts[0] in RENAMED_ROOTS:
        relative = Path(RENAMED_ROOTS[relative.parts[0]]).joinpath(*relative.parts[1:])
    return REPO / relative


SOURCE = REPO / "3 Connectome"
IRIS_CONFIG = REPO / "5 Iris/1 Configuration"
OUTPUTS = REPO / "5 Iris/5 Outputs"
WORKSPACE = REPO / "5 Iris/6 Workspace"
KI67_WORKSPACE = REPO / "6 Ki67/6 Workspace"
PAPER_OUTPUTS = REPO / "7 Paper Results/2 Outputs"
INTERMEDIATE = REPO / "9 temp_intermediate"
PROCESSED = REPO / "2 Data/3 Processed"
MANIFESTS = REPO / "2 Data/2 Manifests"
RAW = _configured_path("FLYBRAIN_RAW_ROOT", REPO / "2 Data/1 Raw")
TEMP = _configured_path("FLYBRAIN_TEMP_DIR", REPO / "10 temp")


def allowed_write(path):
    """Check a write destination before creating it.

    Permit the repository and explicitly configured raw/temp roots. The
    downloader uses this boundary when installing immutable source files.
    """
    value = Path(path).expanduser()
    destination = _renamed_checkout_path(value).resolve()
    roots = (REPO, RAW, TEMP)
    if not any(destination.is_relative_to(root) for root in roots):
        raise ValueError(f"Write destination is outside the configured project roots: {destination}")
    archived = (REPO / name for name in (
        "zenodo", "2 Data/3 Processed", "5 Iris/5 Outputs", "6 Ki67/5 Outputs"))
    if any(destination.is_relative_to(root.resolve()) for root in archived):
        raise ValueError(f"Archived research artifacts are read-only: {destination}. Use a Workspace destination.")
    return destination


def allowed_derived_write(path):
    """Permit a derived file or cache while keeping raw inputs immutable."""
    destination = allowed_write(path)
    if any(destination.is_relative_to(root.resolve()) for root in (RAW, REPO / "2 Data/1 Raw")):
        raise ValueError(f"Raw input files are immutable: {destination}")
    return destination


def writable_output(path):
    """Create a parent directory for a derived artifact inside this repository."""
    destination = allowed_derived_write(path)
    if not destination.is_relative_to(REPO):
        raise ValueError(f"Derived outputs must stay inside {REPO}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def repo_path(path):
    """Resolve checkout-relative paths and archived paths without machine names."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        return REPO / _migrated_relative(value)
    if value.is_relative_to(REPO):
        return REPO / _migrated_relative(value.relative_to(REPO))
    if ORIGINAL_REPO is not None and value.is_relative_to(ORIGINAL_REPO):
        return REPO / _migrated_relative(value.relative_to(ORIGINAL_REPO))
    if ORIGINAL_REPO is not None:
        return value
    parts = value.parts
    for index, part in enumerate(parts[:-1]):
        if (part in {"b4-coding", "drosophila-connectome-classification-regression"}
                and parts[index + 1] in REPOSITORY_TREES - {
                    "temp", "temp_intermediate", "9 temp_intermediate", "10 temp",
                    "11 temp_intermediate", "12 temp"}):
            return REPO / _migrated_relative(Path(*parts[index + 1:]))
        if part == "b2-data-raw" and parts[index + 1] == "malecns-v1.0":
            return RAW / Path(*parts[index + 1:])
    return value


def portable_path(path):
    """Record paths inside this checkout relative to its root."""
    value = repo_path(path)
    return value.relative_to(REPO).as_posix() if value.is_relative_to(REPO) else str(value)


def resolve_record_path(path, base=None):
    """Read repository-relative provenance or an older artifact-relative record."""
    value = Path(path).expanduser()
    if value.is_absolute() or base is None or (value.parts and value.parts[0] in REPOSITORY_TREES):
        return repo_path(value)
    local = repo_path(base) / value
    relative = repo_path(value)
    return relative if not local.exists() and relative.exists() else local


def portable_metadata(value):
    """Normalize path values and absolute path keys in newly written metadata."""
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute() or (candidate.parts and candidate.parts[0] in REPOSITORY_TREES):
            return portable_path(candidate)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            portable_key = (portable_path(key) if isinstance(key, Path)
                            or (isinstance(key, str) and Path(key).is_absolute()) else key)
            if portable_key in result:
                raise ValueError(f"Metadata keys resolve to the same portable path: {portable_key}")
            result[portable_key] = portable_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [portable_metadata(item) for item in value]
    return value


def verified_repo_file(path, expected_sha256):
    """Resolve an archived file and require its recorded SHA-256 before use."""
    import hashlib
    import re

    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A valid recorded SHA-256 is required.")
    destination = repo_path(path).resolve()
    if not destination.is_file():
        raise FileNotFoundError(f"Required archived file was not found: {destination}")
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for archived file: {destination}")
    return destination
