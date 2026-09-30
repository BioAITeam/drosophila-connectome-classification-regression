"""Prepare pooled BCData and SHIDC-B-Ki-67 features from archived CLS matrices.

Source matrices were standardized separately. Recovering
``z * source_std + source_mean`` retains float32 rounding in ``z``.
A new scaler is fitted on the pooled training split. Source artifacts remain
unchanged, and preparation does not execute the image encoders.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil

import numpy as np

from . import data, encoders, features, graphs
from .common import destination, digest, portable_metadata, portable_path, read_json, sha256, write_json, resolve_output_reference


EXPECTED_SPLIT_COUNTS = {"train": 2128, "val": 464, "test": 1102}
ARTIFACT_NAMES = ("mean.npy", "std.npy", "train.npy", "val.npy", "test.npy")


def _same_json(path: Path, value: dict) -> None:
    value = portable_metadata(value)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing joint metadata mismatch: {path}")
    else:
        write_json(path, value)


def _same_npy(path: Path, value: np.ndarray) -> None:
    if path.exists():
        stored = np.load(path, allow_pickle=False)
        if stored.dtype != value.dtype or not np.array_equal(stored, value):
            raise ValueError(f"Existing joint feature/scaler mismatch: {path}")
    else:
        features._save_npy(path, value)


def _copy_verified(source: Path, target: Path, expected: str) -> None:
    if sha256(source) != expected:
        raise ValueError(f"Source checksum mismatch: {source}")
    target = destination(target)
    if target.exists():
        if sha256(target) != expected:
            raise ValueError(f"Existing joint artifact checksum mismatch: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != expected:
            raise ValueError(f"Source changed while copying: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot_path(source: Path, model: dict) -> Path:
    return (source / "cache" / "huggingface" / "hub" /
            ("models--" + model["repo_id"].replace("/", "--")) /
            "snapshots" / model["revision"])


def _verify_snapshot(source: Path, representation: str, model: dict,
                     observed: dict[str, str]) -> dict:
    revision_path = source / "cache" / "huggingface" / f"{representation}-revision.json"
    if read_json(revision_path) != model:
        raise ValueError(f"Pinned model revision metadata mismatch: {representation}")
    observed[portable_path(revision_path)] = sha256(revision_path)
    snapshot = _snapshot_path(source, model)
    hashes = model.get("file_sha256", {})
    required = {"config.json", "preprocessor_config.json", *model.get("weight_files", [])}
    if not model.get("weight_files") or not required.issubset(hashes):
        raise ValueError("Pinned model provenance must include config, preprocessing and weight hashes")
    for name, expected in hashes.items():
        # Pinned filenames may contain subdirectories, but may not escape the snapshot.
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid pinned model filename: {name}")
        path = snapshot / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned local model checksum mismatch: {path}")
        observed[portable_path(path)] = expected
    return {"source_output": portable_path(source), "snapshot_path": portable_path(snapshot),
            "revision_metadata_path": portable_path(revision_path),
            "revision_metadata_sha256": observed[portable_path(revision_path)],
            "policy": "Reuse verified local pretrained files read-only; no download or weight copy"}


def _load_sources(config: dict, output: Path) -> tuple[dict, dict, dict, dict]:
    joint = config["joint"]
    source = resolve_output_reference(joint["source_output"])
    datasets = joint.get("datasets", ["bcdata", "shid"])
    representations = config["representations"]
    graph_names = config.get("graphs", ["anatomical", "degree241"])
    if set(config["datasets"]) != {"joint"}:
        raise ValueError("Joint preparation requires datasets={'joint': ...}")
    if not isinstance(datasets, list) or len(datasets) != 2 or set(datasets) != {"bcdata", "shid"}:
        raise ValueError("Joint datasets must contain bcdata and shid once each")
    if not representations or len(set(representations)) != len(representations) or any(
            rep not in features.MODELS for rep in representations):
        raise ValueError("Joint preparation supports unique frozen ViT representations only")
    if not graph_names or len(set(graph_names)) != len(graph_names) or any(
            name not in ("anatomical", "degree241") for name in graph_names):
        raise ValueError("Invalid joint graph selection")
    if output == source or source.is_relative_to(output):
        raise ValueError("Joint output must not overwrite or contain the source output")
    expected_counts = joint.get("expected_split_counts", EXPECTED_SPLIT_COUNTS)
    if set(expected_counts) != set(data.SPLITS) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in expected_counts.values()):
        raise ValueError("Expected split counts must be positive integers for train, val and test")

    observed, rows_by_dataset, source_manifests = {}, {}, {}
    seen_ids = set()
    for dataset in datasets:
        path = source / "manifests" / f"{dataset}.json"
        document = read_json(path)
        if document.get("dataset") != dataset or document.get("class_names") != data.CLASS_NAMES:
            raise ValueError(f"Source manifest dataset/classes mismatch: {dataset}")
        observed[portable_path(path)] = sha256(path)
        source_manifests[dataset] = {"path": portable_path(path), "sha256": observed[portable_path(path)]}
        groups = {split: [] for split in data.SPLITS}
        for row in document["rows"]:
            if row.get("dataset") != dataset or row.get("split") not in groups:
                raise ValueError(f"Source row dataset/split mismatch: {dataset}")
            sample = row["sample_id"]
            if sample in seen_ids or not sample.startswith(f"{dataset}/{row['split']}/"):
                raise ValueError(f"Duplicate or invalid source sample_id: {sample}")
            expected_labels = data.labels_from_counts(row["positive_count"], row["negative_count"])
            if any(row.get(key) != value for key, value in expected_labels.items()):
                raise ValueError(f"Source count-derived target mismatch: {sample}")
            seen_ids.add(sample)
            groups[row["split"]].append(portable_metadata(copy.deepcopy(row)))
        if any(not group for group in groups.values()):
            raise ValueError(f"Empty source split: {dataset}")
        rows_by_dataset[dataset] = groups
    pooled = {split: [row for dataset in datasets for row in rows_by_dataset[dataset][split]]
              for split in data.SPLITS}
    counts = {split: len(rows) for split, rows in pooled.items()}
    if counts != expected_counts:
        raise ValueError(f"Joint split counts mismatch: expected {expected_counts}, found {counts}")

    loaded_features = {}
    for representation in representations:
        loaded = {}
        reference = None
        for dataset in datasets:
            directory = source / "features" / dataset / representation
            path = directory / "metadata.json"
            metadata = read_json(path)
            observed[portable_path(path)] = sha256(path)
            dimension = features.MODELS[representation]["dimension"]
            if (metadata.get("dataset") != dataset or metadata.get("representation") != representation or
                    metadata.get("dimension") != dimension or metadata.get("frozen") is not True):
                raise ValueError(f"Source frozen feature metadata mismatch: {dataset}/{representation}")
            model = metadata["model"]
            configured = config.get("vit_models", {}).get(representation, {})
            if model["repo_id"] != configured.get("repo_id", features.MODELS[representation]["repo_id"]):
                raise ValueError(f"Configured feature model mismatch: {representation}")
            if configured.get("revision") and configured["revision"] != model["revision"]:
                raise ValueError(f"Configured feature revision mismatch: {representation}")
            identity = {"model": model, "preprocessing": metadata["preprocessing"], "dimension": dimension}
            if reference is not None and identity != reference:
                raise ValueError(f"Source model revision/preprocessing mismatch: {representation}")
            reference = identity
            hashes = metadata["artifact_sha256"]
            if not set(ARTIFACT_NAMES).issubset(hashes):
                raise ValueError("Incomplete source feature checksum provenance")
            for name in ARTIFACT_NAMES:
                artifact = directory / name
                if sha256(artifact) != hashes[name]:
                    raise ValueError(f"Source feature checksum mismatch: {artifact}")
                observed[portable_path(artifact)] = hashes[name]
            mean = np.load(directory / "mean.npy", allow_pickle=False)
            std = np.load(directory / "std.npy", allow_pickle=False)
            if (mean.shape != (dimension,) or std.shape != (dimension,) or
                    not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
                raise ValueError("Invalid source CLS standardizer")
            recovered = {}
            for split, rows in rows_by_dataset[dataset].items():
                if (metadata["input_identity_sha256"][split] != features._row_identity(rows) or
                        metadata["sample_ids"][split] != [row["sample_id"] for row in rows]):
                    raise ValueError(f"Source feature sample order/content mismatch: {dataset}/{split}")
                z = np.load(directory / f"{split}.npy", mmap_mode="r", allow_pickle=False)
                if z.dtype != np.float32 or z.shape != (len(rows), dimension) or not np.isfinite(z).all():
                    raise ValueError("Invalid source standardized feature matrix")
                recovered[split] = np.asarray(z, dtype=np.float64) * std + mean
                if not np.isfinite(recovered[split]).all():
                    raise ValueError("Non-finite reconstructed CLS values")
            loaded[dataset] = {"metadata": metadata, "recovered": recovered,
                               "source_metadata_sha256": observed[portable_path(path)]}
        snapshot = _verify_snapshot(source, representation, reference["model"], observed)
        loaded_features[representation] = {"datasets": loaded, "reference": reference, "snapshot": snapshot}

    graph_files = {}
    for name in graph_names:
        graph_path = source / "graphs" / f"{name}.npz"
        metadata_path = graph_path.with_suffix(".json")
        metadata = read_json(metadata_path)
        graph = graphs.load_graph(source, name)
        if metadata.get("name") != name or len(graph["node_ids"]) != metadata["nodes"] or len(graph["src"]) != metadata["edges"]:
            raise ValueError(f"Source graph metadata dimensions mismatch: {name}")
        graph_files[name] = (graph_path, metadata_path)
        observed[portable_path(graph_path)] = sha256(graph_path)
        observed[portable_path(metadata_path)] = sha256(metadata_path)
    context = {"source_output": portable_path(source), "datasets": datasets,
               "representations": representations, "graphs": graph_names,
               "expected_split_counts": expected_counts, "source_sha256": observed,
               "source_manifests": source_manifests}
    return context, pooled, loaded_features, graph_files


def prepare(config: dict, output: Path) -> dict:
    """Prepare pooled feature artifacts from fixed source data.

    Row order is train/val/test, then ``joint.datasets`` order, then each source
    manifest's split order. Original ``dataset`` and ``sample_id`` fields remain
    unchanged. Repeated calls verify the complete source/output fingerprints and
    return the same preparation record, refusing changed inputs or outputs.
    """
    output = Path(output).expanduser().resolve()
    source_output = resolve_output_reference(config["joint"]["source_output"])
    if output == source_output or source_output.is_relative_to(output):
        raise ValueError("Joint output must not overwrite or contain the source output")
    lock = destination(output / "manifests" / ".joint-prepare.lock")
    with features._lock(lock):
        context, pooled, feature_sources, graph_files = _load_sources(config, output)
        signature = digest(context)
        record_path = output / "manifests" / "joint_preparation.json"
        if record_path.exists():
            record = read_json(record_path)
            if record["source_signature"] != signature:
                raise ValueError("Joint preparation source/configuration mismatch")
            for relative, expected in record["artifact_sha256"].items():
                if sha256(output / relative) != expected:
                    raise ValueError(f"Existing joint artifact checksum mismatch: {relative}")
            return record

        rows = [row for split in data.SPLITS for row in pooled[split]]
        manifest = {"schema_version": 1, "dataset": "joint", "source_root": context["source_output"],
                    "source_datasets": context["datasets"], "class_names": data.CLASS_NAMES,
                    "rows": rows, "summary": data._summary(rows),
                    "source_manifests": context["source_manifests"],
                    "row_order": "split order train/val/test, configured source dataset order, original source split order",
                    "source_signature": signature}
        paths = [output / "manifests" / "joint.json"]
        _same_json(paths[0], manifest)
        for representation, source in feature_sources.items():
            arrays = {split: np.concatenate([source["datasets"][name]["recovered"][split]
                                             for name in context["datasets"]], axis=0)
                      for split in data.SPLITS}
            center, scale = features.fit_standardization(arrays["train"])
            directory = output / "features" / "joint" / representation
            _same_npy(directory / "mean.npy", center)
            _same_npy(directory / "std.npy", scale)
            for split, array in arrays.items():
                _same_npy(directory / f"{split}.npy", features.standardize(array, center, scale))
            artifact_hashes = {name: sha256(directory / name) for name in ARTIFACT_NAMES}
            metadata = {"schema_version": 1, "dataset": "joint", "representation": representation,
                        "dimension": source["reference"]["dimension"], "model": source["reference"]["model"],
                        "frozen": True, "input_identity_sha256": {
                            split: features._row_identity(part) for split, part in pooled.items()},
                        "sample_ids": {split: [row["sample_id"] for row in part] for split, part in pooled.items()},
                        "preprocessing": {**source["reference"]["preprocessing"],
                            "standardization": "One per-CLS-dimension mean/std fitted on pooled training images only"},
                        "artifact_sha256": artifact_hashes, "source_signature": signature,
                        "source_metadata_sha256": {name: source["datasets"][name]["source_metadata_sha256"]
                                                   for name in context["datasets"]},
                        "snapshot_reference": source["snapshot"],
                        "reconstruction": {"formula": "raw_cls_approx = float64(stored_float32_z) * source_std + source_mean",
                            "fit_split": "train", "source_order": context["datasets"],
                            "arithmetic_dtype": "float64", "output_dtype": "float32",
                            "roundoff": "Inverting stored float32 standardized values retains their rounding error; it does not recover bit-identical original CLS vectors",
                            "test_used_for_fit": False, "validation_used_for_fit": False,
                            "models_reexecuted": False}, "device": "cpu"}
            _same_json(directory / "metadata.json", metadata)
            paths.extend(directory / name for name in (*ARTIFACT_NAMES, "metadata.json"))
            revision_source = resolve_output_reference(source["snapshot"]["revision_metadata_path"])
            revision_target = output / "cache" / "huggingface" / revision_source.name
            _copy_verified(revision_source, revision_target, source["snapshot"]["revision_metadata_sha256"])
            paths.append(revision_target)
        for graph_path, metadata_path in graph_files.values():
            for source_path in (graph_path, metadata_path):
                target = output / "graphs" / source_path.name
                _copy_verified(source_path, target, context["source_sha256"][portable_path(source_path)])
                paths.append(target)
        # Detect an external write racing preparation before declaring completion.
        for source_path, expected in context["source_sha256"].items():
            if sha256(resolve_output_reference(source_path)) != expected:
                raise ValueError(f"Source changed during joint preparation: {source_path}")
        record = {"schema_version": 1, "dataset": "joint", "source_signature": signature,
                  "source": context, "split_counts": {split: len(part) for split, part in pooled.items()},
                  "per_dataset_split_counts": {name: {split: sum(row["dataset"] == name for row in part)
                                                       for split, part in pooled.items()} for name in context["datasets"]},
                  "artifact_sha256": {str(path.relative_to(output)): sha256(path) for path in paths}}
        _same_json(record_path, record)
        return record


def inference_snapshot(output: Path, representation: str, metadata: dict) -> Path:
    """Locate a relocated frozen snapshot while enforcing its original hashes."""
    reference = metadata["snapshot_reference"]
    original = resolve_output_reference(reference["source_output"])
    model = metadata["model"]
    if (resolve_output_reference(reference["snapshot_path"]) != _snapshot_path(original, model) or
            resolve_output_reference(reference["revision_metadata_path"]) != original / "cache" / "huggingface" / f"{representation}-revision.json"):
        raise ValueError("Joint snapshot reference paths are inconsistent")
    candidates = [resolve_output_reference(original), Path(output).resolve().parent]
    for source in dict.fromkeys(candidates):
        if not _snapshot_path(source, model).is_dir():
            continue
        verified = _verify_snapshot(source, representation, model, {})
        if verified["revision_metadata_sha256"] != reference["revision_metadata_sha256"]:
            raise ValueError("Joint snapshot reference checksum mismatch")
        return resolve_output_reference(verified["snapshot_path"])
    raise FileNotFoundError(
        "Pinned frozen ViT files are missing. Restore the shared Ki67 model cache "
        "from the companion archive next to the joint output directory."
    )


def encode_images(output: Path, representation: str, images: list[Path],
                  device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Encode annotation-free images using verified local weights and the joint scaler.

    This inference-only function imports PyTorch on demand. It never downloads
    weights, fits preprocessing, or opens annotations. Source model files remain
    read-only; mutable runtime caches stay in the repository's temporary tree.
    """
    if not images or batch_size < 1 or representation not in features.MODELS:
        raise ValueError("Provide images, a positive batch size and a frozen representation")
    output = Path(output).resolve()
    directory = output / "features" / "joint" / representation
    metadata = read_json(directory / "metadata.json")
    if metadata.get("dataset") != "joint" or metadata.get("representation") != representation or not metadata.get("frozen"):
        raise ValueError("Invalid joint feature metadata")
    for name in ("mean.npy", "std.npy"):
        if sha256(directory / name) != metadata["artifact_sha256"][name]:
            raise ValueError(f"Joint training scaler checksum mismatch: {name}")
    snapshot = inference_snapshot(output, representation, metadata)
    features._configure_cache(output)
    import torch

    model = encoders.frozen_model(snapshot, device)
    preprocessing = metadata["preprocessing"]
    vectors = encoders.encode_cls(
        model, images, image_mean=preprocessing["image_mean"], image_std=preprocessing["image_std"],
        dimension=metadata["dimension"], batch_size=batch_size, device=device,
    )
    mean = np.load(directory / "mean.npy", allow_pickle=False)
    std = np.load(directory / "std.npy", allow_pickle=False)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return features.standardize(vectors, mean, std)
