"""Create the SHIDC-B-Ki-67-V2 patient-disjoint partition from prepared pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import floor, lcm
from pathlib import Path
import random
import re
import shutil
import tempfile

from PIL import Image


REPO = Path(__file__).resolve().parents[2]
SPLITS = ("Train", "Validation", "Test")
CLASSES = ("Low", "Intermediate", "High")
PATTERN = re.compile(r"^(p[1-9]\d*)_(\d+)_(\d+)$")
EXPECTED_IMAGES = (1325, 331, 700)
IMAGE_LEVEL_SEED = 64
VALIDATION_FRACTION = 0.2
IMAGE_LEVEL_REFERENCE = (
    "https://github.com/BioAITeam/ki67-open-and-closed-vlm-benchmarking/blob/main/"
    "2.preprocess/4.split_shid_train_validation.py"
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Use a path relative to the repository root without '..'")
    resolved = (REPO / path).resolve()
    if not resolved.is_relative_to(REPO):
        raise ValueError("The path must resolve inside the repository")
    return resolved


def patient_number(patient: str) -> int:
    return int(patient[1:])


def inspect_source(root: Path, layout: str = "prepared") -> list[dict]:
    rows = []
    seen_stems = set()
    image_hashes = set()
    pixel_hashes = set()
    if layout == "train-full":
        training = "Train_full" if (root / "Train_full").is_dir() else "Train"
        specification = ((training, sum(EXPECTED_IMAGES[:2])), ("Test", EXPECTED_IMAGES[2]))
    else:
        specification = tuple(zip(SPLITS, EXPECTED_IMAGES))
    for split, expected in specification:
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f"Missing source directory: {split}")
        images, annotations = {}, {}
        for path in sorted(folder.iterdir()):
            if path.name == ".gitkeep":
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected source entry: {split}/{path.name}")
            suffix = path.suffix.lower()
            if suffix not in (".jpg", ".jpeg", ".json"):
                raise ValueError(f"Unexpected source file: {split}/{path.name}")
            target = annotations if suffix == ".json" else images
            if path.stem in target:
                raise ValueError(f"Duplicate stem: {split}/{path.stem}")
            target[path.stem] = path
        if images.keys() != annotations.keys() or len(images) != expected:
            raise ValueError(f"Expected {expected} complete image/JSON pairs in {split}")
        for stem in sorted(images):
            match = PATTERN.fullmatch(stem)
            if match is None or stem in seen_stems:
                raise ValueError(f"Invalid or repeated image identifier: {stem}")
            seen_stems.add(stem)
            image_path, annotation_path = images[stem], annotations[stem]
            image_hash = checksum(image_path)
            with Image.open(image_path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (1228, 1228):
                    raise ValueError(f"Expected a 1228 by 1228 RGB image: {stem}")
                pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if image_hash in image_hashes or pixel_hash in pixel_hashes:
                raise ValueError(f"Duplicate image content: {stem}")
            image_hashes.add(image_hash)
            pixel_hashes.add(pixel_hash)
            records = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"Expected an annotation list: {stem}")
            counts, coordinates = Counter(), Counter()
            outside = 0
            for record in records:
                if not isinstance(record, dict) or not {"x", "y", "label_id"} <= record.keys():
                    raise ValueError(f"Invalid annotation record: {stem}")
                x, y, label = (record[key] for key in ("x", "y", "label_id"))
                if any(type(value) is not int for value in (x, y, label)) or label not in (1, 2):
                    raise ValueError(f"Expected integer coordinates and label 1 or 2: {stem}")
                counts[label] += 1
                coordinates[x, y, label] += 1
                outside += not (0 <= x < 1228 and 0 <= y < 1228)
            positive, negative = counts[1], counts[2]
            total = positive + negative
            label = 0 if total == 0 or 100 * positive < 16 * total else (
                1 if 100 * positive < 30 * total else 2)
            rows.append({
                "patient_id": match[1], "source_image_id": f"{match[1]}_{match[2]}",
                "image_id": stem, "source_split": split,
                "source_image": image_path.relative_to(root).as_posix(),
                "source_annotation": annotation_path.relative_to(root).as_posix(),
                "image_sha256": image_hash, "annotation_sha256": checksum(annotation_path),
                "pixel_sha256": pixel_hash, "positive_cells": positive,
                "negative_cells": negative, "total_cells": total,
                "class_id": label, "class_name": CLASSES[label],
                "empty_annotation": total == 0,
                "out_of_bounds_annotations": outside,
                "duplicate_annotation_records": sum(count - 1 for count in coordinates.values()),
            })
    patients = {row["patient_id"] for row in rows}
    if patients != {f"p{number}" for number in range(1, 24)}:
        raise ValueError("Expected the 23 patient identifiers p1 through p23")
    return rows


def reconstruct_image_level_split(rows: list[dict], layout: str) -> dict:
    training_stems = sorted(row["image_id"] for row in rows if row["source_split"] != "Test")
    shuffled = list(training_stems)
    random.Random(IMAGE_LEVEL_SEED).shuffle(shuffled)
    validation_count = floor(len(shuffled) * VALIDATION_FRACTION)
    validation = set(shuffled[:validation_count])
    if (len(shuffled) - validation_count, validation_count) != EXPECTED_IMAGES[:2]:
        raise ValueError("The seed-64 split must produce 1325 training and 331 validation images")
    for row in rows:
        split = "Test" if row["source_split"] == "Test" else (
            "Validation" if row["image_id"] in validation else "Train")
        row["benchmark_split"] = split
        if layout == "prepared" and row["source_split"] != split:
            raise ValueError(f"Prepared membership differs from the seed-64 benchmark: {row['image_id']}")
    return {
        "reference": IMAGE_LEVEL_REFERENCE,
        "reference_script_sha256": "1eb66e280e913ea9c0525e991961f851e050de01f7c96d3ed0d04932f677e379",
        "performed_by": "BioAITeam preprocessing for the preceding Ki-67 benchmark study",
        "source_training_images": len(training_stems),
        "seed": IMAGE_LEVEL_SEED, "validation_fraction": VALIDATION_FRACTION,
        "algorithm": "Sort stems lexicographically, shuffle with random.Random(64), assign the first floor(N * 0.2) to Validation and the remainder to Train",
        "test_images_used_in_train_validation_split": 0,
        "test_membership_preserved_in_this_stage": True,
        "matches_prepared_membership": True if layout == "prepared" else None,
        "partition_images": {split: sum(row["benchmark_split"] == split for row in rows) for split in SPLITS},
    }


def summarize(rows: list[dict]) -> dict:
    result = {
        "patients": len({row["patient_id"] for row in rows}), "images": len(rows),
        "patient_ids": sorted({row["patient_id"] for row in rows}, key=patient_number),
        "source_images": len({row["source_image_id"] for row in rows}),
        "class_counts": {name: sum(row["class_id"] == index for row in rows)
                         for index, name in enumerate(CLASSES)},
    }
    for key in ("positive_cells", "negative_cells", "total_cells", "empty_annotation",
                "out_of_bounds_annotations", "duplicate_annotation_records"):
        result[key] = sum(row[key] for row in rows)
    return result


def select_partition(rows: list[dict]) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row)
    patients = sorted(grouped, key=patient_number)
    counts = [len(grouped[patient]) for patient in patients]
    classes = [tuple(sum(row["class_id"] == c for row in grouped[patient])
                     for c in range(3)) for patient in patients]
    total_classes = tuple(sum(values[c] for values in classes) for c in range(3))
    total_images = len(rows)
    quotas = [Fraction(len(patients) * target, total_images) for target in EXPECTED_IMAGES]
    patient_counts = [quota.numerator // quota.denominator for quota in quotas]
    for index in sorted(range(3), key=lambda i: (-(quotas[i] - patient_counts[i]), i))[
            :len(patients) - sum(patient_counts)]:
        patient_counts[index] += 1
    denominator = lcm(*(target ** 2 for target in EXPECTED_IMAGES))
    factors = [denominator // target ** 2 for target in EXPECTED_IMAGES]
    best = None
    feasible = 0
    validation_candidates = 0
    all_indices = tuple(range(len(patients)))
    for validation in combinations(all_indices, patient_counts[1]):
        if sum(counts[index] for index in validation) != EXPECTED_IMAGES[1]:
            continue
        validation_candidates += 1
        validation_classes = tuple(sum(classes[index][c] for index in validation) for c in range(3))
        remaining = tuple(index for index in all_indices if index not in validation)
        for test in combinations(remaining, patient_counts[2]):
            if sum(counts[index] for index in test) != EXPECTED_IMAGES[2]:
                continue
            feasible += 1
            test_classes = tuple(sum(classes[index][c] for index in test) for c in range(3))
            training_classes = tuple(total_classes[c] - validation_classes[c] - test_classes[c]
                                     for c in range(3))
            score = sum(factors[s] * (total_images * values[c] - EXPECTED_IMAGES[s] * total_classes[c]) ** 2
                        for s, values in enumerate((training_classes, validation_classes, test_classes))
                        for c in range(3))
            candidate = (score, validation, test)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No patient-disjoint partition preserves the requested image counts")
    score, validation, test = best
    assignment = {patient: ("Validation" if index in validation else "Test" if index in test else "Train")
                  for index, patient in enumerate(patients)}
    exact_score = Fraction(score, denominator * total_images ** 2)
    method = {
        "unit": "patient identifier from the filename prefix before the first underscore",
        "target_images": dict(zip(SPLITS, EXPECTED_IMAGES)),
        "patient_allocation": dict(zip(SPLITS, patient_counts)),
        "patient_allocation_rule": "largest remainder of the original image proportions",
        "selection": "exhaustive search with exact image counts and patient counts",
        "objective": "sum over splits and classes of (split class prevalence minus global class prevalence) squared",
        "tie_break": "ascending numeric validation patient IDs, then ascending numeric test patient IDs",
        "randomization": "none",
        "validation_candidates": validation_candidates, "feasible_partitions": feasible,
        "objective_exact": f"{exact_score.numerator}/{exact_score.denominator}",
        "objective_value": float(exact_score),
        "uses_predictions_or_model_metrics": False,
    }
    return assignment, method


def make_report(rows: list[dict], assignment: dict, method: dict, source: Path, destination: Path,
                image_level_split: dict, layout: str) -> dict:
    for row in rows:
        row["split"] = assignment[row["patient_id"]]
        row["image"] = f"{row['split']}/{Path(row['source_image']).name}"
        row["annotation"] = f"{row['split']}/{Path(row['source_annotation']).name}"
    patients = {split: {patient for patient, target in assignment.items() if target == split} for split in SPLITS}
    overlaps = {f"{a}:{b}": sorted(patients[a] & patients[b]) for a, b in combinations(SPLITS, 2)}
    if any(overlaps.values()):
        raise RuntimeError("A patient appears in more than one output partition")
    return {
        "schema_version": 2, "dataset": "SHIDC-B-Ki-67-V2",
        "description": "Proposed patient-disjoint repartition of the prepared SHIDC-B-Ki-67 data",
        "path_base": "repository root",
        "source_root": source.relative_to(REPO).as_posix(),
        "source_layout": layout,
        "output_root": destination.relative_to(REPO).as_posix(),
        "manifest_path_bases": {"source_image": "source_root", "source_annotation": "source_root",
                                "image": "output_root", "annotation": "output_root"},
        "script": Path(__file__).resolve().relative_to(REPO).as_posix(),
        "script_sha256": checksum(Path(__file__)),
        "image_level_split": image_level_split,
        "v2_input_pool": "Union of benchmark Train, Validation and Test, followed by patient-disjoint repartitioning",
        "method": method, "total": summarize(rows),
        "source_splits": {split: summarize([row for row in rows if row["source_split"] == split])
                          for split in sorted({row["source_split"] for row in rows})},
        "benchmark_splits": {split: summarize([row for row in rows if row["benchmark_split"] == split])
                             for split in SPLITS},
        "splits": {split: summarize([row for row in rows if row["split"] == split]) for split in SPLITS},
        "patient_overlap": overlaps,
        "annotation_policy": "Count all existing label 1/2 records, retaining empty lists and out-of-bounds coordinates",
        "empty_annotation_policy": "Retain each image and use PI zero for class stratification",
        "class_thresholds_percent": {"Low": "PI <16", "Intermediate": "16 <= PI <30", "High": "PI >=30"},
        "source_release": "https://doi.org/10.1038/s41598-021-86912-w",
        "patient_identifier_reference": "https://github.com/SHIDCenter/PathoNet/blob/master/evaluation.py",
        "original_results_retained": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def placeholder_only(destination: Path) -> bool:
    if not destination.exists():
        return True
    if not destination.is_dir() or destination.is_symlink():
        return False
    allowed = {Path(split) for split in SPLITS}
    markers = {Path(".gitkeep"), *(Path(split) / ".gitkeep" for split in SPLITS)}
    for path in destination.rglob("*"):
        relative = path.relative_to(destination)
        if path.is_symlink():
            return False
        if path.is_dir() and relative in allowed:
            continue
        if path.is_file() and relative in markers and path.stat().st_size == 0:
            continue
        return False
    return True


def write_metadata(destination: Path, rows: list[dict], report: dict) -> None:
    patient_rows = []
    for patient in sorted({row["patient_id"] for row in rows}, key=patient_number):
        selected = [row for row in rows if row["patient_id"] == patient]
        summary = summarize(selected)
        patient_rows.append({"patient_id": patient, "split": selected[0]["split"],
                             **{key: summary[key] for key in ("images", "positive_cells", "negative_cells", "total_cells")},
                             **summary["class_counts"]})
    report["file_integrity"] = {"image_annotation_pairs": len(rows), "copied_files": 2 * len(rows),
                                "all_copy_hashes_match_source": True}
    with tempfile.TemporaryDirectory(prefix=".metadata-", dir=destination) as temporary:
        staging = Path(temporary)
        write_csv(staging / "split_manifest.csv", rows)
        write_csv(staging / "patient_summary.csv", patient_rows)
        (staging / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for path in staging.iterdir():
            path.replace(destination / path.name)
    for split in SPLITS:
        (destination / split / ".gitkeep").touch(exist_ok=True)


def refresh_metadata(rows: list[dict], report: dict, destination: Path) -> None:
    expected = {row[key]: row[hash_key] for row in rows
                for key, hash_key in (("image", "image_sha256"), ("annotation", "annotation_sha256"))}
    allowed_metadata = {"split_manifest.csv", "patient_summary.csv", "split_summary.json"}
    markers = {f"{split}/.gitkeep" for split in SPLITS}
    observed = set()
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link: {relative}")
        if path.is_dir():
            if relative not in SPLITS:
                raise ValueError(f"Unexpected directory: {relative}")
            continue
        if relative in expected:
            if checksum(path) != expected[relative]:
                raise ValueError(f"Existing file differs from source: {relative}")
            observed.add(relative)
        elif relative in markers and path.stat().st_size != 0:
            raise ValueError(f"Expected an empty .gitkeep placeholder: {relative}")
        elif relative not in allowed_metadata and relative not in markers:
            raise ValueError(f"Unexpected output file: {relative}")
    if observed != expected.keys():
        raise ValueError("The existing output must contain every assigned image/JSON pair")
    write_metadata(destination, rows, report)


def publish_copy(rows: list[dict], report: dict, source: Path, destination: Path) -> None:
    if not placeholder_only(destination):
        raise FileExistsError("The destination must be absent, empty or contain only .gitkeep placeholders")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shid-v2-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "data"
        for split in SPLITS:
            (staging / split).mkdir(parents=True)
        for row in rows:
            for source_key, destination_key, hash_key in (
                ("source_image", "image", "image_sha256"),
                ("source_annotation", "annotation", "annotation_sha256"),
            ):
                original, copied = source / row[source_key], staging / row[destination_key]
                if checksum(original) != row[hash_key]:
                    raise RuntimeError(f"Source changed during preparation: {row[source_key]}")
                shutil.copy2(original, copied)
                if checksum(copied) != row[hash_key]:
                    raise RuntimeError(f"Copy checksum mismatch: {row[destination_key]}")
        write_metadata(staging, rows, report)
        if destination.exists():
            if not placeholder_only(destination):
                raise FileExistsError("The destination changed during preparation")
            for path in destination.rglob(".gitkeep"):
                path.unlink()
            for split in SPLITS:
                folder = destination / split
                if folder.exists():
                    folder.rmdir()
            destination.rmdir()
        staging.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Paths are relative to the repository root. Input files and annotations are copied unchanged. "
        "First reproduce the prior benchmark Train/Validation split with seed 64 and validation fraction 0.2, "
        "keeping Test separate. Then pool all 2356 images for the deterministic patient-disjoint V2 partition. "
        "V2 preserves 1325/331/700 images with 13/3/7 patients and uses no random seed. "
        "Annotations must already contain only labels 1 and 2. Existing image/JSON files are never overwritten."))
    parser.add_argument("--source", default="2 Data/1 Raw/ki67/shid")
    parser.add_argument("--destination", default="2 Data/1 Raw/ki67/shid_v2")
    parser.add_argument("--source-layout", choices=("prepared", "train-full"), default="prepared",
                        help="prepared: Train/Validation/Test; train-full: Train_full (or Train with 1656 pairs) and Test")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan-only", action="store_true", help="Inspect and select both partitions without writing files")
    actions.add_argument("--refresh-metadata", action="store_true",
                         help="Verify every existing V2 file, then update only manifests and .gitkeep placeholders")
    args = parser.parse_args()
    source, destination = repository_path(args.source), repository_path(args.destination)
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        parser.error("Source and destination must be separate directories")
    if args.refresh_metadata and not destination.is_dir():
        parser.error("Refreshing metadata requires an existing V2 dataset")
    if not args.plan_only and not args.refresh_metadata and not placeholder_only(destination):
        parser.error("The destination must be absent, empty or contain only .gitkeep placeholders")
    rows = inspect_source(source, args.source_layout)
    image_level_split = reconstruct_image_level_split(rows, args.source_layout)
    assignment, method = select_partition(rows)
    report = make_report(rows, assignment, method, source, destination, image_level_split, args.source_layout)
    if args.refresh_metadata:
        refresh_metadata(rows, report, destination)
    elif not args.plan_only:
        publish_copy(rows, report, source, destination)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
