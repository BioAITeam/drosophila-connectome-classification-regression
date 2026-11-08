"""Read image-annotation pairs and derive Ki-67 targets without changing source files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .common import REPO, portable_path, read_json, sha256, write_json
from .tasks import CLASS_NAMES, TASK_NAMES as TASKS

SPLITS = ("train", "val", "test")
DEFAULT_PARTITION_POLICY = "Original image-level splits are retained; patient independence is not asserted"


def partition_policy(specification: dict) -> str:
    policy = specification.get("partition_policy", DEFAULT_PARTITION_POLICY)
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("partition_policy must be a nonempty string")
    return policy


def resolve_image_path(row: dict) -> Path:
    """Locate an immutable image after moving an archived dataset to a new root."""
    recorded = Path(row["image_path"]).expanduser()
    dataset = row.get("dataset")
    override = os.environ.get(f"KI67_{dataset.upper()}_ROOT") if dataset else None
    candidate = recorded if recorded.is_absolute() else REPO / recorded
    if not override and candidate.is_file():
        if row.get("image_sha256") and sha256(candidate) != row["image_sha256"]:
            raise ValueError(f"Audited image checksum mismatch: {candidate}")
        return candidate
    if dataset not in ("bcdata", "shid") or row.get("split") not in SPLITS:
        raise FileNotFoundError(f"Image is missing: {recorded}")
    root = Path(override).expanduser() if override else REPO / "2 Data/1 Raw/ki67" / dataset
    root = root if root.is_absolute() else REPO / root
    folders = ({"train": "train", "val": "validation", "test": "test"} if dataset == "bcdata"
               else {"train": "Train", "val": "Validation", "test": "Test"})
    canonical_root = Path("2 Data/1 Raw/ki67") / dataset
    relative = (recorded.relative_to(canonical_root) if recorded.is_relative_to(canonical_root)
                else Path(folders[row["split"]]) / recorded.name)
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Restore {dataset} images or set KI67_{dataset.upper()}_ROOT: {path}")
    if row.get("image_sha256") and sha256(path) != row["image_sha256"]:
        raise ValueError(f"Relocated image checksum mismatch: {path}")
    return path


def labels_from_counts(positive: int, negative: int) -> dict:
    """Use integer cross-products at thresholds and an explicit empty PI=0 policy."""
    if isinstance(positive, bool) or isinstance(negative, bool):
        raise ValueError("Cell counts must be nonnegative integers")
    if not isinstance(positive, (int, np.integer)) or not isinstance(negative, (int, np.integer)):
        raise ValueError("Cell counts must be nonnegative integers")
    if min(positive, negative) < 0:
        raise ValueError("Cell counts cannot be negative")
    positive, negative = int(positive), int(negative)
    total = positive + negative
    numerator = 100 * positive
    denominator = total or 1
    return {
        "positive_count": positive, "negative_count": negative,
        "pi": numerator / denominator,
        "multiclass": 0 if numerator < 16 * denominator else (1 if numerator < 30 * denominator else 2),
        "binary5": int(numerator > 5 * denominator),
        "binary10": int(numerator > 10 * denominator),
        "binary30": int(numerator >= 30 * denominator),
        "empty_annotation": total == 0,
    }


def _annotation_counts(records: list, width: int, height: int) -> tuple[int, int, dict]:
    if not isinstance(records, list):
        raise ValueError("Annotation JSON must be a list")
    counts = Counter()
    coordinates = Counter()
    outside = 0
    for record in records:
        if not isinstance(record, dict) or not {"x", "y", "label_id"}.issubset(record):
            raise ValueError("Each annotation must contain x, y and label_id")
        x, y, label = (record[key] for key in ("x", "y", "label_id"))
        if any(not isinstance(value, int) or isinstance(value, bool) for value in (x, y, label)):
            raise ValueError("Annotation coordinates and label_id must be integers")
        if label not in (1, 2):
            raise ValueError(f"Unexpected label_id={label}; only positive=1 and negative=2 are accepted")
        counts[label] += 1
        coordinates[x, y, label] += 1
        outside += int(not (0 <= x < width and 0 <= y < height))
    duplicates = sum(number - 1 for number in coordinates.values())
    return counts[1], counts[2], {
        "out_of_bounds_annotations": outside,
        "duplicate_annotation_records": duplicates,
    }


def _source_reference(path: Path, dataset: str, split: str | None = None,
                      source_root: Path | None = None) -> str:
    reference = portable_path(path)
    if not Path(reference).is_absolute():
        return reference
    root = Path("2 Data/1 Raw/ki67") / dataset
    if source_root is not None:
        return (root / path.relative_to(source_root)).as_posix()
    if split is None:
        return root.as_posix()
    folders = {"train": "train", "val": "validation", "test": "test"}
    if dataset == "shid":
        folders = {"train": "Train", "val": "Validation", "test": "Test"}
    return (root / folders[split] / path.name).as_posix()


def inspect_pair(image_path: Path, json_path: Path, dataset: str, split: str,
                 source_root: Path | None = None) -> dict:
    """Validate one pair without changing it; retain every supplied cell record."""
    image_path, json_path = Path(image_path).resolve(), Path(json_path).resolve()
    before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (image_path, json_path)]
    with Image.open(image_path) as image:
        image.load()
        if image.mode != "RGB":
            raise ValueError(f"Expected an existing RGB image, got {image.mode}: {image_path}")
        width, height = image.size
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    records = json.loads(json_path.read_text(encoding="utf-8"))
    positive, negative, issues = _annotation_counts(records, width, height)
    stem = image_path.stem
    patient = re.match(r"^(p\d+)_", stem) if dataset == "shid" else None
    prefix = re.match(r"^(p\d+_[^_]+)_", stem) if dataset == "shid" else None
    result = {
        "sample_id": f"{dataset}/{split}/{stem}", "dataset": dataset, "split": split,
        "image_path": _source_reference(image_path, dataset, split, source_root),
        "json_path": _source_reference(json_path, dataset, split, source_root),
        "width": width, "height": height, "channels": 3,
        "image_sha256": sha256(image_path), "json_sha256": sha256(json_path),
        "pixel_sha256": pixel_hash, "patient_id": patient.group(1) if patient else None,
        "source_image_prefix": prefix.group(1) if prefix else None,
        **labels_from_counts(positive, negative), **issues,
    }
    after = [(path.stat().st_size, path.stat().st_mtime_ns) for path in (image_path, json_path)]
    if before != after:
        raise RuntimeError(f"Source changed during audit: {image_path}")
    return result


def _summary(rows: list[dict], policy: str = DEFAULT_PARTITION_POLICY) -> dict:
    split_summary = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "images": len(selected),
            **{key: sum(row[key] for row in selected) for key in (
                "positive_count", "negative_count", "empty_annotation",
                "out_of_bounds_annotations", "duplicate_annotation_records")},
            "class_counts": {
                task: {name: sum(row[task] == number for row in selected)
                       for number, name in enumerate(names)}
                for task, names in CLASS_NAMES.items()
            },
            "patient_ids": sorted({row["patient_id"] for row in selected if row["patient_id"]}),
            "dimensions": sorted({f'{row["width"]}x{row["height"]}x3' for row in selected}),
        }
    duplicates = {}
    for key in ("image_sha256", "pixel_sha256"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row["sample_id"])
        duplicates[key] = [group for group in groups.values() if len(group) > 1]
    overlaps = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            overlaps[f"{first}:{second}"] = {}
            for key in ("patient_id", "source_image_prefix"):
                a = {row[key] for row in rows if row["split"] == first and row[key]}
                b = {row[key] for row in rows if row["split"] == second and row[key]}
                overlaps[f"{first}:{second}"][key] = sorted(a & b)
    return {
        "images": len(rows), "splits": split_summary, "duplicate_images": duplicates,
        "split_overlap": overlaps,
        "empty_pi_policy": "PI = 0 when positive_count + negative_count = 0; explicitly marked convention",
        "annotation_policy": "All supplied label 1/2 records are counted, including repeated and out-of-bounds records",
        "partition_policy": policy,
    }


def audit(config: dict, output: Path) -> dict:
    """Audit configured split folders and publish derived dataset manifests."""
    output = Path(output)
    summaries = {}
    for dataset, specification in config["datasets"].items():
        policy = partition_policy(specification)
        root = Path(os.environ.get(f"KI67_{dataset.upper()}_ROOT", specification["root"])).expanduser()
        root = (root if root.is_absolute() else REPO / root).resolve()
        rows = []
        for split in SPLITS:
            directory = (root / specification["splits"][split]).resolve()
            if not directory.is_relative_to(root) or not directory.is_dir():
                raise ValueError(f"Invalid source split directory: {directory}")
            images = {}
            annotations = {}
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".json"):
                    mapping = annotations if path.suffix.lower() == ".json" else images
                    if path.stem in mapping:
                        raise ValueError(f"Duplicate stem within a split: {path.stem}")
                    mapping[path.stem] = path
            if not images or images.keys() != annotations.keys():
                raise ValueError(f"Missing/orphan image-JSON pairs in {directory}: "
                                 f"images_only={sorted(images.keys() - annotations.keys())[:10]}, "
                                 f"json_only={sorted(annotations.keys() - images.keys())[:10]}")
            rows.extend(inspect_pair(images[stem], annotations[stem], dataset, split, root)
                        for stem in sorted(images))
        summary = _summary(rows, policy)
        summaries[dataset] = summary
        write_json(output / "manifests" / f"{dataset}.json", {
            "schema_version": 1, "dataset": dataset, "source_root": _source_reference(root, dataset),
            "class_names": CLASS_NAMES, "rows": rows, "summary": summary,
        })
    return summaries


def load_rows(output: Path, dataset: str, split: str | None = None) -> list[dict]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    document = read_json(Path(output) / "manifests" / f"{dataset}.json")
    return [row for row in document["rows"] if split is None or row["split"] == split]


def targets(rows: list[dict], task: str) -> np.ndarray:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    dtype = np.int64 if task in CLASS_NAMES else np.float32
    return np.asarray([row[task] for row in rows], dtype=dtype)
