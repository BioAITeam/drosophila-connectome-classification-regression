"""Immutable inputs for the Ki-67 conventional-classifier experiments.

All recorded paths are relative to the supplied repository root. Frozen features
retain their original training-only standardization. RGB is flattened in row,
column, channel order at native resolution and divided by 255.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import PIL
from PIL import Image


from .paths import REPO, WORKSPACE as OUTPUT, ARCHIVE, PATIENT_DISJOINT, writable

DATASETS = ("bcdata", "shid", "shid_v2")
REPRESENTATIONS = ("dinov2", "phikonv2")
SPLITS = ("train", "val", "test")
TASKS = ("multiclass", "binary5", "binary10", "binary30")
CLASS_NAMES = {
    "multiclass": ["Low", "Intermediate", "High"],
    "binary5": ["PI <= 5", "PI > 5"],
    "binary10": ["PI <= 10", "PI > 10"],
    "binary30": ["PI < 30", "PI >= 30"],
}
EXPECTED_COUNTS = {"bcdata": (803, 133, 402), "shid": (1325, 331, 700),
                   "shid_v2": (1325, 331, 700)}
_VERIFIED_HASHES: dict[tuple[str, int, int, int], str] = {}


def sha256(path: Path) -> str:
    """Hash a file, reusing only a same-process unchanged-stat verification."""
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _VERIFIED_HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError("An input changed during checksum verification")
        _VERIFIED_HASHES[key] = digest.hexdigest()
    return _VERIFIED_HASHES[key]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve(repo: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Input paths must be relative to the repository")
    candidate = (repo / relative).resolve()
    candidate.relative_to(repo.resolve())
    return candidate


def _source(dataset: str) -> tuple[Path, str]:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return (PATIENT_DISJOINT, "shid") if dataset == "shid_v2" else (ARCHIVE, dataset)


def labels_from_counts(positive: int, negative: int) -> dict[str, int]:
    """Use exact threshold cross-products and retain the empty-annotation rule."""
    if (isinstance(positive, bool) or isinstance(negative, bool)
            or not isinstance(positive, (int, np.integer))
            or not isinstance(negative, (int, np.integer)) or min(positive, negative) < 0):
        raise ValueError("Cell counts must be nonnegative integers")
    numerator = 100 * int(positive)
    denominator = max(1, int(positive) + int(negative))
    return {
        "multiclass": 0 if numerator < 16 * denominator else (1 if numerator < 30 * denominator else 2),
        "binary5": int(numerator > 5 * denominator),
        "binary10": int(numerator > 10 * denominator),
        "binary30": int(numerator >= 30 * denominator),
    }


def row_identity(rows: list[dict]) -> str:
    payload = [(row["sample_id"], row["image_sha256"]) for row in rows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def manifest(repo: Path, dataset: str) -> tuple[dict[str, list[dict]], dict]:
    root, token = _source(dataset)
    relative = root / "manifests" / f"{token}.json"
    path = _resolve(repo, relative)
    document = _read(path)
    rows = document["rows"]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sample identifiers in manifest")
    if len({row["image_sha256"] for row in rows}) != len(rows):
        raise ValueError("Repeated image content within a dataset")
    groups = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    if tuple(len(groups[split]) for split in SPLITS) != EXPECTED_COUNTS[dataset]:
        raise ValueError("Unexpected partition sizes")
    for row in rows:
        if row["channels"] != 3:
            raise ValueError("Native input must have three RGB channels")
        expected_size = 640 if dataset == "bcdata" else 1228
        if (row["height"], row["width"]) != (expected_size, expected_size):
            raise ValueError("Unexpected native image size")
        targets = labels_from_counts(row["positive_count"], row["negative_count"])
        if any(row[task] != targets[task] for task in TASKS):
            raise ValueError("Stored targets disagree with exact cell-count thresholds")
        _resolve(repo, row["image_path"])
        _resolve(repo, row["json_path"])
    patients = {split: sorted({row["patient_id"] for row in groups[split]
                               if row.get("patient_id") is not None}) for split in SPLITS}
    if dataset == "shid_v2":
        for i, first in enumerate(SPLITS):
            for second in SPLITS[i + 1:]:
                if set(patients[first]) & set(patients[second]):
                    raise ValueError("SHIDC-B-Ki-67-V2 contains overlapping patient groups")
    provenance = {
        "dataset": dataset, "manifest": relative.as_posix(), "manifest_sha256": sha256(path),
        "input_identity_sha256": {split: row_identity(groups[split]) for split in SPLITS},
        "counts": {split: len(groups[split]) for split in SPLITS}, "patient_ids": patients,
        "labels": "Integer cross-products of positive/negative cell counts, empty annotations assigned PI zero",
        "class_counts": {task: {split: {str(label): count for label, count in
                          sorted(Counter(labels_from_counts(row['positive_count'], row['negative_count'])[task]
                                         for row in groups[split]).items())}
                          for split in SPLITS} for task in TASKS},
    }
    return groups, provenance


def audit_sources(repo: Path, verify_annotations: bool = True) -> dict:
    """Check split identities, exact labels and the shared SHID image pool."""
    manifests = {dataset: manifest(repo, dataset) for dataset in DATASETS}
    pools = {}
    for dataset, (groups, _) in manifests.items():
        all_rows = [row for split in SPLITS for row in groups[split]]
        pools[dataset] = {row["image_sha256"]: (row["pixel_sha256"], row["json_sha256"],
                          row["positive_count"], row["negative_count"], row["patient_id"])
                          for row in all_rows}
        if verify_annotations:
            for row in all_rows:
                path = _resolve(repo, row["json_path"])
                if sha256(path) != row["json_sha256"]:
                    raise ValueError("Annotation content differs from source manifest")
                records = json.loads(path.read_text(encoding="utf-8"))
                counts = Counter(record["label_id"] for record in records)
                if set(counts) - {1, 2}:
                    raise ValueError("Unsupported cell annotation label")
                if (counts[1], counts[2]) != (row["positive_count"], row["negative_count"]):
                    raise ValueError("Annotation counts do not reproduce manifest targets")
    if pools["shid"] != pools["shid_v2"]:
        raise ValueError("Original and patient-disjoint SHID partitions do not contain the same image pool")
    if set(pools["bcdata"]) & set(pools["shid"]):
        raise ValueError("Unexpected shared image content between datasets")
    result = {
        "schema_version": 1, "path_base": "repository root", "datasets": {
            dataset: provenance for dataset, (_, provenance) in manifests.items()},
        "same_shid_image_pixel_annotation_counts_and_patient_pool": True,
        "cross_dataset_exact_image_overlap": 0,
        "annotation_files_recounted": verify_annotations,
    }
    _write(_resolve(repo, OUTPUT / "data" / "source_audit.json"), result)
    return result


@contextmanager
def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
        fcntl.flock(stream, fcntl.LOCK_UN)


def _stat_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev, "inode": stat.st_ino}


def _verified_rgb_hash(repo: Path, path: Path, expected_hash: str) -> str:
    """Avoid repeated multi-gigabyte scans while immutable cache files are unchanged."""
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    receipt_path = _resolve(repo, OUTPUT / "data" / "checksum-receipts.json")
    with _lock(receipt_path.with_suffix(".lock")):
        receipts = (_read(receipt_path) if receipt_path.exists() else
                    {"schema_version": 1, "path_base": "repository root", "files": {}})
        signature = _stat_signature(path)
        receipt = receipts["files"].get(relative)
        if receipt and receipt["stat"] == signature and receipt["sha256"] == expected_hash:
            return expected_hash
        digest = sha256(path)
        if signature != _stat_signature(path):
            raise RuntimeError("RGB cache changed during checksum verification")
        if digest != expected_hash:
            raise ValueError("Native RGB cache checksum mismatch")
        receipts["files"][relative] = {"stat": signature, "sha256": digest}
        _write(receipt_path, receipts)
        return digest


def _verify_matrix(path: Path, expected_hash: str, shape: tuple[int, int],
                   rgb_repo: Path | None = None) -> np.memmap:
    actual_hash = sha256(path) if rgb_repo is None else _verified_rgb_hash(rgb_repo, path, expected_hash)
    if actual_hash != expected_hash:
        raise ValueError("Feature cache checksum mismatch")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError("Feature cache shape or dtype mismatch")
    return array


def prepare_rgb(repo: Path, datasets=DATASETS) -> dict:
    """Build immutable contiguous native-resolution matrices, one split at a time."""
    repo = Path(repo).resolve()
    result = {}
    for dataset in datasets:
        groups, provenance = manifest(repo, dataset)
        directory = _resolve(repo, OUTPUT / "data" / dataset / "rgb")
        with _lock(directory / ".prepare.lock"):
            metadata_path = directory / "metadata.json"
            if metadata_path.exists():
                metadata = _read(metadata_path)
                if metadata["input_identity_sha256"] != provenance["input_identity_sha256"]:
                    raise ValueError("Existing RGB cache belongs to different image partitions")
                for split in SPLITS:
                    _verify_matrix(directory / f"{split}.npy", metadata["artifact_sha256"][f"{split}.npy"],
                                   (len(groups[split]), metadata["dimension"]), rgb_repo=repo)
                result[dataset] = metadata
                continue
            started = time.perf_counter()
            dimension = groups["train"][0]["width"] * groups["train"][0]["height"] * 3
            artifacts, timings = {}, {}
            for split in SPLITS:
                split_started = time.perf_counter()
                target = directory / f"{split}.npy"
                partial = directory / f"{split}.partial.npy"
                array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                                 shape=(len(groups[split]), dimension))
                for index, row in enumerate(groups[split]):
                    image_path = _resolve(repo, row["image_path"])
                    if sha256(image_path) != row["image_sha256"]:
                        raise ValueError("Image content differs from the original manifest")
                    with Image.open(image_path) as image:
                        if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                            raise ValueError("Unexpected image format or dimensions")
                        image.load()
                        if hashlib.sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]:
                            raise ValueError("Decoded pixels differ from the original manifest")
                        pixels = np.asarray(image, dtype=np.uint8).reshape(-1)
                        np.divide(pixels, np.float32(255.0), out=array[index], casting="unsafe")
                array.flush()
                del array
                os.replace(partial, target)
                target.chmod(0o444)
                artifacts[target.name] = sha256(target)
                _verified_rgb_hash(repo, target, artifacts[target.name])
                timings[split] = time.perf_counter() - split_started
                print(json.dumps({"dataset": dataset, "representation": "rgb", "split": split,
                                  "images": len(groups[split]), "seconds": timings[split]}), flush=True)
            metadata = {
                **provenance, "schema_version": 1, "path_base": "repository root",
                "representation": "rgb", "dimension": dimension, "dtype": "float32",
                "preprocessing": {"resize": False, "crop": False, "PCA": False,
                                  "flatten_order": "row,column,channel", "divisor": 255,
                                  "fitted_scaling": False},
                "sample_ids": {split: [row["sample_id"] for row in groups[split]] for split in SPLITS},
                "artifact_sha256": artifacts, "split_seconds": timings,
                "elapsed_seconds": time.perf_counter() - started,
                "preparation_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                         "Pillow": PIL.__version__},
            }
            _write(metadata_path, metadata)
            result[dataset] = metadata
    return result


def load_data(repo: Path, dataset: str, representation: str, task: str) -> dict:
    """Return read-only matrices and exact targets without fitting any transform.

    ``prepare_rgb`` must be called explicitly before loading RGB. Every returned
    array is a read-only file mapping. Test rows are never part of fitting here.
    """
    repo = Path(repo).resolve()
    if representation not in REPRESENTATIONS or task not in TASKS:
        raise ValueError("Unknown representation or classification task")
    groups, provenance = manifest(repo, dataset)
    if representation == "rgb":
        relative = OUTPUT / "data" / dataset / "rgb"
    else:
        source, token = _source(dataset)
        relative = source / "features" / token / representation
    directory = _resolve(repo, relative)
    metadata = _read(directory / "metadata.json")
    if metadata["input_identity_sha256"] != provenance["input_identity_sha256"]:
        raise ValueError("Feature metadata does not match the source partition identity")
    if metadata["representation"] != representation:
        raise ValueError("Wrong feature representation")
    dimension = metadata["dimension"]
    if representation != "rgb":
        if metadata.get("frozen") is not True or dimension != {"dinov2": 768, "phikonv2": 1024}[representation]:
            raise ValueError("Frozen ViT provenance is incomplete")
        for filename in ("mean.npy", "std.npy"):
            if sha256(directory / filename) != metadata["artifact_sha256"][filename]:
                raise ValueError("Source-training normalizer checksum mismatch")
            normalizer = np.load(directory / filename, allow_pickle=False)
            if normalizer.shape != (dimension,) or not np.isfinite(normalizer).all():
                raise ValueError("Invalid source-training normalization")
            if filename == "std.npy" and np.any(normalizer <= 0):
                raise ValueError("Standardization divisors must be positive")
    output = {}
    for split in SPLITS:
        ids = [row["sample_id"] for row in groups[split]]
        if metadata["sample_ids"][split] != ids:
            raise ValueError("Cached matrix order differs from the target order")
        filename = f"{split}.npy"
        output[f"X_{split}"] = _verify_matrix(directory / filename, metadata["artifact_sha256"][filename],
                                               (len(ids), dimension), rgb_repo=repo if representation == "rgb" else None)
        if representation != "rgb" and not np.isfinite(output[f"X_{split}"]).all():
            raise ValueError("Frozen features contain non-finite values")
        output[f"y_{split}"] = np.asarray([labels_from_counts(row["positive_count"], row["negative_count"])[task]
                                            for row in groups[split]], dtype=np.int64)
        output[f"ids_{split}"] = np.asarray(ids, dtype=str)
        output[f"content_ids_{split}"] = np.asarray([row["image_sha256"] for row in groups[split]], dtype=str)
    output["provenance"] = {
        **provenance, "task": task, "representation": representation, "dimension": dimension,
        "class_names": CLASS_NAMES[task], "feature_directory": relative.as_posix(),
        "feature_metadata_sha256": sha256(directory / "metadata.json"),
        "feature_artifact_sha256": metadata["artifact_sha256"],
        "preprocessing": metadata["preprocessing"],
    }
    if representation != "rgb":
        output["provenance"]["frozen_model"] = metadata["model"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--verify-annotations", action="store_true", help="Also recount restored raw annotation files")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="Frozen-feature datasets to verify; source manifests are audited for all datasets")
    args = parser.parse_args()
    repo = args.repo.resolve()
    audit_sources(repo, verify_annotations=args.verify_annotations)
    for dataset in args.datasets:
        for representation in ("dinov2", "phikonv2"):
            for task in TASKS:
                load_data(repo, dataset, representation, task)
    print("Manifest labels, split identities and frozen-feature checksums verified.", flush=True)


if __name__ == "__main__":
    main()
