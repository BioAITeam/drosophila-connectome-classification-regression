"""Reproducible benchmarks with group-disjoint splits and fixed train scaling.

Only preprocessing statistics are fitted here; no feature encoder or predictor is
trained.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .dataset_specs import IRIS_DATASET_NAMES, load_specified_dataset
from .paths import portable_metadata, repo_path, writable_output


DATASET_NAMES = IRIS_DATASET_NAMES


def duplicate_groups(x: np.ndarray) -> np.ndarray:
    """Group equal feature rows, including any rows with conflicting targets."""
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite two-dimensional array.")
    return np.unique(x, axis=0, return_inverse=True)[1].astype(np.int64)


def grouped_split(
    x: np.ndarray, y: np.ndarray, *, task: str, seed: int = 0, outer_fold: int | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """5 outer folds + 4 inner folds: approximately 60/20/20, duplicates together.

    The inner validation fold is always fold zero; it is not selected by test
    performance. All five outer test folds partition the original observations.
    """
    fold = 0 if outer_fold is None else outer_fold
    if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)) or not 0 <= fold < 5:
        raise ValueError("outer_fold must be None or an integer from 0 through 4.")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression.")
    groups = duplicate_groups(x) if groups is None else np.asarray(groups)
    if len(y) != len(x) or len(np.unique(groups)) < 5:
        raise ValueError("At least five feature groups and matching x/y lengths are required.")
    splitter = StratifiedGroupKFold if task == "classification" else GroupKFold
    outer = splitter(n_splits=5, shuffle=True, random_state=seed)
    train_val, test = list(outer.split(x, y, groups))[fold]
    inner = splitter(n_splits=4, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(x[train_val], y[train_val], groups[train_val]))
    return {"train": np.sort(train_val[tr]), "val": np.sort(train_val[va]), "test": np.sort(test)}


def merge_source_groups(x: np.ndarray, supplied_groups: np.ndarray | None = None) -> np.ndarray:
    """Join exact-feature duplicates and user-declared entities before splitting."""
    duplicates = duplicate_groups(x)
    if supplied_groups is None:
        return duplicates
    supplied_groups = np.asarray(supplied_groups)
    if supplied_groups.shape != (len(x),):
        raise ValueError("Source group identifiers must have one entry per row.")
    parents = np.arange(len(x))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = int(parents[i])
        return i

    for grouping in (duplicates, supplied_groups):
        first = {}
        for i, group in enumerate(grouping.tolist()):
            if group in first:
                parents[root(i)] = root(first[group])
            else:
                first[group] = i
    return np.unique([root(i) for i in range(len(x))], return_inverse=True)[1].astype(np.int64)


def permute_grouped_targets(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Globally permute target vectors between equally sized feature/entity groups.

    Unlike a per-epoch shuffled-label training control, this permutation defines
    an entire null dataset before any outer/inner stratification. It preserves
    the target multiset exactly and each group's within-group target pattern.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    if groups.shape != y.shape:
        raise ValueError("Groups and one-dimensional targets must have matching shapes.")
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    bins: dict[int, list[int]] = {}
    for number, indices in enumerate(members):
        bins.setdefault(len(indices), []).append(number)
    rng = np.random.default_rng(seed)
    out = y.copy()
    donors = np.arange(len(members))
    for size in sorted(bins):
        destinations = np.asarray(bins[size])
        source = rng.permutation(destinations)
        donors[destinations] = source
        for destination, donor in zip(destinations, source):
            out[members[destination]] = y[members[donor]]
    metadata = {"seed": int(seed), "policy": "whole-group permutation within equal-size strata before all splits",
                "scope": "entire dataset; one fixed mapping shared across outer folds",
                "source_group_by_destination": donors.tolist(), "group_sizes": [len(ix) for ix in members],
                "n_groups": len(members), "n_changed_targets": int(np.count_nonzero(y != out)),
                "n_exchangeable_groups": sum(len(indices) for indices in bins.values() if len(indices) > 1),
                "target_multiset_preserved": bool(np.array_equal(np.sort(y), np.sort(out)))}
    return out, metadata


def feature_quantile_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, task: str,
                           seed: int, spec: dict[str, Any], metadata: dict[str, Any],
                           outer_fold: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reserve an input-defined Iris domain, without inspecting target values."""
    if outer_fold not in (None, 0):
        raise ValueError("Feature-quantile holdout has a single test domain; outer_fold must be None or 0.")
    feature = spec.get("feature", 0)
    if isinstance(feature, str):
        names = metadata["feature_keys"] if feature in metadata["feature_keys"] else metadata["feature_names"]
        if feature not in names:
            raise ValueError(f"Unknown input feature for domain holdout: {feature}.")
        feature = names.index(feature)
    if isinstance(feature, bool) or not isinstance(feature, (int, np.integer)) or not 0 <= feature < metadata["n_active_features"]:
        raise ValueError("Domain holdout must select an active input feature, never the target or zero padding.")
    quantile = float(spec.get("quantile", 0.8))
    tail = spec.get("tail", "upper")
    if not 0 < quantile < 1 or tail not in {"upper", "lower"}:
        raise ValueError("Domain holdout requires 0 < quantile < 1 and tail upper or lower.")
    threshold = float(np.quantile(x[:, feature], quantile))
    heldout = x[:, feature] >= threshold if tail == "upper" else x[:, feature] <= threshold
    # Reserve the whole entity when any row meets the domain criterion.
    heldout = np.isin(groups, np.unique(groups[heldout]))
    test, train_val = np.flatnonzero(heldout), np.flatnonzero(~heldout)
    if not len(test) or len(np.unique(groups[train_val])) < 4:
        raise ValueError("Domain holdout leaves insufficient independent groups for train/validation.")
    splitter = StratifiedGroupKFold if task == "classification" else GroupKFold
    inner = splitter(n_splits=4, shuffle=True, random_state=seed + 1)
    train, val = next(inner.split(x[train_val], y[train_val], groups[train_val]))
    indices = {"train": np.sort(train_val[train]), "val": np.sort(train_val[val]), "test": np.sort(test)}
    domain = {"kind": "feature_quantile_holdout", "feature_index": int(feature),
              "feature_name": metadata["feature_names"][feature], "quantile": quantile,
              "tail": tail, "threshold": threshold, "units": "original input units",
              "threshold_source": "all input rows only; no target values",
              "group_extension": "any qualifying row moves its whole entity group to holdout",
              "interpretation": "within-dataset input-domain shift; not an external dataset",
              "outer_test_policy": "same fixed held-out domain across split/optimization seeds"}
    return indices, domain


def feature_quantile_transfer_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, task: str,
                                    seed: int, spec: dict[str, Any], metadata: dict[str, Any],
                                    outer_fold: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    """Two disjoint training domains with one shared, untouched target-domain test.

    Both source and adaptation runs use the exact same source-training unit
    conversions. The target domain itself has a grouped 60/20/20 split, whereas
    source observations use the source train/validation split only.
    """
    phase = spec.get("phase", "source")
    if phase not in {"source", "adapt"}:
        raise ValueError("Transfer phase must be source or adapt.")
    source, domain = feature_quantile_split(x, y, groups, task=task, seed=seed, spec=spec,
                                           metadata=metadata, outer_fold=outer_fold)
    domain_indices = source["test"]
    local = grouped_split(x[domain_indices], y[domain_indices], task=task, seed=seed + 100,
                          outer_fold=0, groups=groups[domain_indices])
    target = {part: np.sort(domain_indices[ix]) for part, ix in local.items()}
    chosen = {"train": source["train"], "val": source["val"], "test": target["test"]} if phase == "source" else target
    domain.update(kind="feature_quantile_transfer", phase=phase,
                  source_train_indices=source["train"].tolist(), source_val_indices=source["val"].tolist(),
                  adaptation_train_indices=target["train"].tolist(), adaptation_val_indices=target["val"].tolist(),
                  common_test_indices=target["test"].tolist(),
                  scaling_policy="fit source train only; identical feature and target scalers in both phases",
                  test_partition_note="single common target-domain holdout, not complete OOF coverage",
                  domain_split_policy="grouped 5 outer folds (test 0), 4 inner folds (validation 0); split seed + 100")
    return chosen, domain, source["train"]


def fit_feature_scaler(x_train: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x_train, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Training features must be a nonempty finite matrix.")
    low, high = x.min(axis=0), x.max(axis=0)
    return {"min": low.tolist(), "max": high.tolist(), "constant": (low == high).tolist(),
            "fit_partition": "train", "clip": True, "output_range": [0.0, 1.0]}


def transform_features(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    low, high = np.asarray(scaler["min"]), np.asarray(scaler["max"])
    span = high - low
    scaled = (x - low) / np.where(span > 0, span, 1.0)
    scaled[:, span == 0] = 0.0
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def inverse_targets(y: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Invert the fixed target unit conversion, including unconstrained outputs."""
    spec = metadata.get("target_scaler")
    return np.asarray(y) if spec is None else np.asarray(y) * spec["scale"] + spec["offset"]


def prepare_dataset(
    name: str | Path | dict[str, Any], seed: int = 0, output_dir: str | Path | None = None,
    outer_fold: int | None = None, *, label_permutation_seed: int | None = None,
    split_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare a declared four-channel dataset without train/test leakage.

    Classification y is an integer class ID. Regression y uses a train-only
    affine conversion to [-1, 1] and is not clipped outside the training range.
    All original-unit targets remain available as ``y_raw_<partition>``.
    If output_dir is supplied it must resolve inside this repository.
    """
    x, y, metadata, supplied_groups = load_specified_dataset(name)
    extra = {}
    groups = merge_source_groups(x, supplied_groups)
    original_y = y.copy()
    original_fingerprint = hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    if label_permutation_seed is not None:
        if isinstance(label_permutation_seed, bool) or not isinstance(label_permutation_seed, (int, np.integer)):
            raise ValueError("label_permutation_seed must be an integer or None.")
        y, permutation = permute_grouped_targets(y, groups, int(label_permutation_seed))
        metadata["label_permutation"] = permutation
    if split_spec is None:
        split_spec = metadata.get("dataset_spec", {}).get("split")
    source_scaler_indices = None
    if split_spec is None or split_spec.get("kind", "grouped_kfold") == "grouped_kfold":
        indices = grouped_split(x, y, task=metadata["task"], seed=seed, outer_fold=outer_fold, groups=groups)
        policy = "group-disjoint outer 5 folds, inner 4 folds, fixed inner fold 0"
    elif split_spec.get("kind") == "feature_quantile_holdout":
        indices, domain = feature_quantile_split(x, y, groups, task=metadata["task"], seed=seed,
                                                 spec=split_spec, metadata=metadata, outer_fold=outer_fold)
        metadata["domain_shift"] = domain
        policy = "fixed input-defined test domain; group-disjoint inner 4 folds, fixed inner fold 0"
    elif split_spec.get("kind") == "feature_quantile_transfer":
        indices, domain, source_scaler_indices = feature_quantile_transfer_split(
            x, y, groups, task=metadata["task"], seed=seed, spec=split_spec,
            metadata=metadata, outer_fold=outer_fold)
        metadata["domain_shift"] = domain
        policy = "source/adaptation disjoint domains, shared held-out target-domain test; source-training scalers fixed"
    else:
        raise ValueError("Unknown split kind; choose grouped_kfold, feature_quantile_holdout or feature_quantile_transfer.")
    scaler_indices = indices["train"] if source_scaler_indices is None else source_scaler_indices
    scaler = fit_feature_scaler(x[scaler_indices])
    if source_scaler_indices is not None:
        scaler["fit_partition"] = "source_train"
    metadata["scaler_fit_indices"] = scaler_indices.tolist()
    fingerprint = hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()
    metadata.update(split_seed=int(seed), outer_fold=0 if outer_fold is None else int(outer_fold),
                    n_features=int(x.shape[1]), n_samples=len(x), n_unique_feature_groups=len(np.unique(groups)),
                    scaler=scaler, raw_array_sha256=fingerprint, original_raw_array_sha256=original_fingerprint,
                    split_policy=policy,
                    stratified=metadata["task"] == "classification",
                    sizes={key: len(value) for key, value in indices.items()})
    metadata["groups_sha256"] = hashlib.sha256(np.ascontiguousarray(groups, dtype=np.int64).tobytes()).hexdigest()
    metadata["group_policy"] = "exact-feature duplicate groups" if supplied_groups is None else "transitive union of exact-feature duplicates and declared source entities"
    if metadata["task"] == "regression":
        low, high = float(y[scaler_indices].min()), float(y[scaler_indices].max())
        metadata["target_scaler"] = {"offset": (low + high) / 2, "scale": (high - low) / 2 or 1.0,
                                     "fit_partition": "train" if source_scaler_indices is None else "source_train", "clip": False,
                                     "formula": "(y - offset) / scale"}
    else:
        metadata["target_scaler"] = None
    result: dict[str, Any] = {"indices": indices, "metadata": metadata, "groups": groups,
                              "raw_x": x.copy(), "raw_y": y.copy()}
    if supplied_groups is not None:
        result["source_groups"] = np.asarray(supplied_groups, dtype=str)
        canonical_groups = json.dumps(result["source_groups"].tolist(), ensure_ascii=False, separators=(",", ":"))
        metadata["source_groups_sha256"] = hashlib.sha256(canonical_groups.encode("utf-8")).hexdigest()
    if label_permutation_seed is not None:
        result["original_raw_y"] = original_y
    for partition, ix in indices.items():
        result[f"x_raw_{partition}"] = x[ix].copy()
        result[f"y_raw_{partition}"] = y[ix].copy()
        result[f"groups_{partition}"] = groups[ix]
    result.update(extra)
    partitions = list(indices) + (["test_extrapolation"] if extra else [])
    for partition in partitions:
        raw_x, raw_y = result[f"x_raw_{partition}"], result[f"y_raw_{partition}"]
        result[f"x_{partition}"] = transform_features(raw_x, scaler)
        if metadata["task"] == "regression":
            target_scaler = metadata["target_scaler"]
            result[f"y_{partition}"] = ((raw_y - target_scaler["offset"]) / target_scaler["scale"]).astype(np.float32)
        else:
            result[f"y_{partition}"] = raw_y.astype(np.int64)
        outside = ((raw_x < np.asarray(scaler["min"])) | (raw_x > np.asarray(scaler["max"])))
        metadata.setdefault("fraction_outside_train_feature_range", {})[partition] = float(outside.mean())
    if extra:
        metadata["extrapolation_note"] = "Fixed held-out domain; input clipping saturates outside train ranges. Never tune on it."
    if output_dir is not None:
        destination = writable_output(repo_path(output_dir) / "dataset.npz").parent
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        arrays.update({f"indices_{key}": value for key, value in indices.items()})
        np.savez_compressed(destination / "dataset.npz", **arrays)
        (destination / "metadata.json").write_text(json.dumps(portable_metadata(metadata), indent=2) + "\n", encoding="utf-8")
    return result
