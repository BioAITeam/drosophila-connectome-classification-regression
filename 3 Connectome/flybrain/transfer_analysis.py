"""Audit source training, target-domain adaptation and matched reset controls.

These are within-dataset domain holdouts, never complete out-of-fold predictions
or independent external validation. Only saved artifacts are read.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import classification_metrics, paired_bootstrap_delta, regression_metrics
from .checkpoints import checkpoint_source_sha256
from .paths import portable_metadata, repo_path, writable_output

_ROLES = ("source", "adapt", "reset")
_DOMAIN_PARTITIONS = ("source_train_indices", "source_val_indices", "adaptation_train_indices",
                      "adaptation_val_indices", "common_test_indices")
_BUDGET_KEYS = ("epochs", "batch_size", "learning_rate", "patience", "select_checkpoint", "algorithm", "local_rule")


def _read(path):
    return json.loads(path.read_text())


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def _metrics(true, predicted, metadata):
    if metadata["task"] == "classification":
        return classification_metrics(true, predicted, n_classes=metadata["n_classes"])
    return regression_metrics(true, predicted)


def _paired_domain_delta(true, before, after, *, metadata, metric, groups, n_bootstrap, seed):
    """Keep the declared class set fixed in every domain-bootstrap draw.

    The frozen general bootstrap infers one label range before resampling.
    Any additional, globally declared classes have zero F1 in every draw.
    Extending that range therefore multiplies each macro F1 (and its paired
    delta/quantiles) by inferred_classes / declared_classes exactly. Other
    metrics are unchanged. This correction touches derived analysis only.
    """
    result = paired_bootstrap_delta(true, before, after, task=metadata["task"], metric=metric,
                                    groups=groups, n_bootstrap=n_bootstrap, seed=seed)
    if metadata["task"] == "classification":
        first = classification_metrics(true, before, n_classes=metadata["n_classes"])
        last = classification_metrics(true, after, n_classes=metadata["n_classes"])
        result["n_classes"] = metadata["n_classes"]
        if metric == "macro_f1":
            labels = [np.asarray(pred).argmax(axis=1) if np.asarray(pred).ndim == 2 else np.asarray(pred)
                      for pred in (before, after)]
            inferred = int(max(np.max(true), *(np.max(pred) for pred in labels))) + 1
            scale = inferred / metadata["n_classes"]
            for key in ("ci_low", "ci_high"):
                if result[key] is not None:
                    result[key] *= scale
            result["macro_f1_convention"] = "mean F1 over all declared classes, including absent classes"
            result["inferred_to_declared_class_scale"] = scale
        result.update(before=first[metric], after=last[metric], delta=last[metric] - first[metric])
    return result


def _load(record):
    directory = repo_path(record["run"]).resolve()
    config = _read(directory / "config.json")
    role = config.get("transfer_role", record.get("transfer_role"))
    if role is None:
        return None
    if role not in _ROLES:
        raise ValueError(f"Unknown transfer role {role}: {directory}")
    metadata = _read(directory / "dataset/metadata.json")
    domain = metadata.get("domain_shift", {})
    if domain.get("kind") != "feature_quantile_transfer":
        raise ValueError(f"Transfer analysis requires feature_quantile_transfer: {directory}")
    if domain.get("phase") != ("source" if role == "source" else "adapt"):
        raise ValueError(f"Transfer role and dataset phase disagree: {directory}")
    arrays = _arrays(directory / "dataset/dataset.npz")
    fingerprint = hashlib.sha256(np.ascontiguousarray(arrays["raw_x"]).tobytes()
                                 + np.ascontiguousarray(arrays["raw_y"]).tobytes()).hexdigest()
    if fingerprint != metadata.get("raw_array_sha256"):
        raise ValueError(f"Transfer dataset fingerprint mismatch: {directory}")
    predictions = _arrays(directory / "predictions.npz")
    rows = predictions["row_index"]
    if (rows.ndim != 1 or len(np.unique(rows)) != len(rows)
            or not np.array_equal(np.sort(rows), np.sort(arrays["indices_test"]))):
        raise ValueError(f"Transfer test prediction coverage is invalid: {directory}")
    if not np.array_equal(predictions["true"], arrays["raw_y"][rows]):
        raise ValueError(f"Transfer predictions use different targets: {directory}")
    order = np.argsort(rows)
    predictions = {key: value[order] if value.ndim and len(value) == len(rows) else value
                   for key, value in predictions.items()}
    result = _read(directory / "results.json")
    invariants = result.get("invariants", {})
    if (invariants.get("trainable_parameters") != ["gain"] or invariants.get("extra_learnable_neurons") != 0
            or invariants.get("learnable_interface_parameters") != 0 or invariants.get("buffers_unchanged") is not True):
        raise ValueError(f"Transfer run violates the internal-learning-only contract: {directory}")
    recomputed = {stage: _metrics(predictions["true"], predictions[key], metadata)
                  for stage, key in (("initial", "before"), ("final", "after"))}
    keys = ("accuracy", "balanced_accuracy", "macro_f1") if metadata["task"] == "classification" else ("rmse", "mae", "r2")
    for stage in recomputed:
        for key in keys:
            expected, saved = recomputed[stage][key], result[stage].get(key)
            if expected is None and saved is None:
                continue
            if expected is None or saved is None or not np.isclose(expected, saved, atol=1e-10, rtol=1e-9):
                raise ValueError(f"Stored transfer metric disagrees with predictions: {stage}.{key}: {directory}")
    gains = _arrays(directory / "plastic_edges.npz")
    groups = arrays["groups"]
    all_parts = [np.asarray(domain[key], dtype=np.int64) for key in _DOMAIN_PARTITIONS]
    if any(not len(part) for part in all_parts):
        raise ValueError(f"Empty source/adaptation/test partition: {directory}")
    for index, first in enumerate(all_parts):
        for second in all_parts[index + 1:]:
            if np.intersect1d(groups[first], groups[second]).size:
                raise ValueError(f"Group leakage across transfer domains or partitions: {directory}")
    concatenated = np.concatenate(all_parts)
    if not np.array_equal(np.sort(concatenated), np.arange(len(groups))):
        raise ValueError(f"Transfer partition description does not cover each raw row once: {directory}")
    expected_parts = {"train": all_parts[0] if role == "source" else all_parts[2],
                      "val": all_parts[1] if role == "source" else all_parts[3], "test": all_parts[4]}
    for part, expected in expected_parts.items():
        if not np.array_equal(np.sort(arrays[f"indices_{part}"]), np.sort(expected)):
            raise ValueError(f"Transfer {part} differs from its declared phase: {directory}")
    if not np.array_equal(metadata.get("scaler_fit_indices"), all_parts[0]):
        raise ValueError(f"Transfer scalers were not fitted on source training rows: {directory}")
    if metadata["scaler"].get("fit_partition") != "source_train":
        raise ValueError(f"Transfer feature-scaler policy is not source_train: {directory}")
    if metadata["task"] == "regression" and metadata["target_scaler"].get("fit_partition") != "source_train":
        raise ValueError(f"Transfer target-scaler policy is not source_train: {directory}")
    split = int(config.get("split_seed", config.get("seed", 0)))
    optimization = int(config.get("optimization_seed", config.get("seed", 0)))
    if metadata.get("split_seed") != split:
        raise ValueError(f"Transfer split seed mismatch: {directory}")
    key = (config.get("transfer_task", metadata["name"]), config["transfer_domain"], split, optimization,
           config.get("rewire_seed"))
    return {"key": key, "role": role, "run": directory, "config": config, "metadata": metadata,
            "domain": domain, "data": arrays, "predictions": predictions, "result": result,
            "metrics": recomputed, "gains": gains, "graph": _read(directory / "graph.json")}


def _compare(triple, *, n_bootstrap, bootstrap_seed):
    source, adapt, reset = (triple[role] for role in _ROLES)
    for candidate in (adapt, reset):
        for key in ("raw_array_sha256", "task", "target_names", "target_units", "feature_names", "scaler", "target_scaler"):
            if candidate["metadata"].get(key) != source["metadata"].get(key):
                raise ValueError(f"Transfer runs disagree on metadata {key}")
        for key in _DOMAIN_PARTITIONS:
            if candidate["domain"][key] != source["domain"][key]:
                raise ValueError(f"Transfer runs disagree on {key}")
        for key in ("feature_index", "tail", "threshold", "quantile"):
            if candidate["domain"].get(key) != source["domain"].get(key):
                raise ValueError(f"Transfer runs disagree on input domain {key}")
        for key in ("row_index", "true"):
            if not np.array_equal(candidate["predictions"][key], source["predictions"][key]):
                raise ValueError(f"Transfer runs do not share the same test {key}")
        if candidate["result"]["invariants"]["buffers_sha256"] != source["result"]["invariants"]["buffers_sha256"]:
            raise ValueError("Transfer changed fixed neural buffers")
        if candidate["graph"].get("processed_sha256") != source["graph"].get("processed_sha256"):
            raise ValueError("Transfer changed the source graph")
        for key in ("model", "encoding", "plastic_scope", "freeze_candidate"):
            if candidate["config"].get(key) != source["config"].get(key):
                raise ValueError(f"Transfer changed the fixed model configuration {key}")
        for key in ("pre_body_id", "post_body_id", "original_contact_count"):
            if not np.array_equal(candidate["gains"][key], source["gains"][key]):
                raise ValueError(f"Transfer changed the existing plastic-edge identity {key}")
    for key in _BUDGET_KEYS:
        if adapt["config"].get(key) != reset["config"].get(key):
            raise ValueError(f"Adaptation and reset have unequal declared budgets: {key}")
    if not np.array_equal(source["predictions"]["after"], adapt["predictions"]["before"]):
        raise ValueError("Source final predictions do not exactly match adaptation initial predictions")
    if not np.array_equal(source["gains"]["trained_gain"], adapt["gains"]["initial_gain"]):
        raise ValueError("Source trained internal weights do not match adaptation initial weights")
    if not np.array_equal(source["gains"]["initial_gain"], reset["gains"]["initial_gain"]) or np.any(reset["gains"]["initial_gain"] != 0):
        raise ValueError("Reset did not start from unchanged anatomical internal gains")
    training_source = _read(adapt["run"] / "training_source.json")
    actual_source_hash = checkpoint_source_sha256(source["run"] / "model.pt")
    if training_source.get("checkpoint_sha256") != actual_source_hash or source["result"].get("checkpoint_sha256") != actual_source_hash:
        raise ValueError("Adaptation source-checkpoint provenance mismatch")
    if repo_path(training_source["checkpoint"]).resolve() != (source["run"] / "model.pt").resolve():
        raise ValueError("Adaptation source-checkpoint path mismatch")
    if training_source.get("optimizer_reinitialized") is not True:
        raise ValueError("Adaptation must restart the optimizer for the declared comparison")
    metadata = source["metadata"]
    rows = source["predictions"]["row_index"]
    groups = source["data"]["groups"][rows]
    comparisons = {}
    keys = ("accuracy", "balanced_accuracy", "macro_f1") if metadata["task"] == "classification" else ("rmse", "mae", "r2")
    for name, before, after in (("source_to_adapt", source, adapt), ("reset_to_adapt", reset, adapt)):
        comparisons[name] = {metric: _paired_domain_delta(source["predictions"]["true"], before["predictions"]["after"],
                                                           after["predictions"]["after"], metadata=metadata, metric=metric,
                                                           groups=groups, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
                             for metric in keys}
    key = source["key"]
    return {"transfer_task": key[0], "transfer_domain": key[1], "split_seed": key[2], "optimization_seed": key[3],
            "rewire_seed": key[4], "task": metadata["task"], "target_names": metadata.get("target_names"),
            "target_units": metadata.get("target_units", "class_id"), "status": "complete",
            "domain_feature": source["domain"]["feature_name"], "domain_tail": source["domain"]["tail"],
            "domain_threshold": source["domain"]["threshold"], "domain_quantile": source["domain"]["quantile"],
            "software_validation_only": any(triple[role]["result"].get("software_validation_only", False) for role in _ROLES),
            "n_test": len(rows), "n_test_groups": len(np.unique(groups)),
            "partition_sizes": {key: len(source["domain"][key]) for key in _DOMAIN_PARTITIONS},
            "metrics": {role: triple[role]["metrics"] for role in _ROLES}, "comparisons": comparisons,
            "checks": {"partitions_group_disjoint": True, "same_test": True, "same_source_scalers": True,
                       "source_final_equals_adapt_initial_predictions": True, "source_final_equals_adapt_initial_weights": True,
                       "same_fixed_neural_buffers": True, "same_adaptation_reset_budget": True, "reset_gains_zero": True,
                       "source_checkpoint_verified": True, "test_metrics_recomputed": True, "internal_learning_only": True},
            "budget": {key: adapt["config"].get(key) for key in _BUDGET_KEYS},
            "runs": {role: {"path": str(triple[role]["run"]), "selected_epoch": triple[role]["result"]["selected_epoch"],
                            "epochs_completed": triple[role]["result"]["epochs_completed"],
                            "training_seconds": triple[role]["result"].get("training_seconds"),
                            "config_sha256": _digest(triple[role]["run"] / "config.json"),
                            "predictions_sha256": _digest(triple[role]["run"] / "predictions.npz")}
                     for role in _ROLES}}


def analyze_transfer_campaign(folder, records, *, n_bootstrap=1000, bootstrap_seed=1729):
    """Compare matched transfer runs and write JSON, CSV and Markdown summaries."""
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) != n_bootstrap or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    folder = writable_output(repo_path(folder) / "transfer_analysis.json").parent
    grouped = defaultdict(dict)
    seen = set()
    skipped = []
    for record in records:
        run = _load(record)
        if run is None:
            skipped.append(str(record["run"]))
            continue
        if run["run"] in seen or run["role"] in grouped[run["key"]]:
            raise ValueError(f"Duplicate transfer role/run in a matched triple: {run['run']}")
        seen.add(run["run"])
        grouped[run["key"]][run["role"]] = run
    complete, incomplete = [], []
    for key, triple in sorted(grouped.items(), key=lambda item: str(item[0])):
        missing = [role for role in _ROLES if role not in triple]
        if missing:
            incomplete.append({"transfer_task": key[0], "transfer_domain": key[1], "split_seed": key[2],
                               "optimization_seed": key[3], "rewire_seed": key[4], "status": "incomplete", "missing_roles": missing})
        else:
            complete.append(_compare(triple, n_bootstrap=int(n_bootstrap), bootstrap_seed=int(bootstrap_seed)))
    report = {"format_version": 1, "n_runs": len(seen), "n_complete_triples": len(complete),
              "n_incomplete_triples": len(incomplete), "triples": complete, "incomplete": incomplete, "skipped": skipped,
              "scope": "within-dataset input-domain transfer; shared held-out target-domain test; not complete OOF or external validation",
              "no_test_based_selection": True, "external_learned_readout": False,
              "software_validation_only": any(row.get("software_validation_only", False) for row in complete),
              "uncertainty": "Paired group percentile intervals conditional on each fixed test split and trained models; overlapping seeds are not independent samples.",
              "budget_note": "Adapted and reset runs share the adaptation budget; adapted models additionally retain source-training experience."}
    writable_output(folder / "transfer_analysis.json").write_text(json.dumps(portable_metadata(report), indent=2, allow_nan=False)+"\n")
    csv_path = writable_output(folder / "transfer_analysis.csv")
    columns = ["transfer_task", "transfer_domain", "split_seed", "optimization_seed", "n_test", "target_units", "role", "stage", "metric", "value"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in complete:
            metrics = ("accuracy", "balanced_accuracy", "macro_f1") if row["task"] == "classification" else ("rmse", "mae", "r2")
            for role in _ROLES:
                for stage in ("initial", "final"):
                    for metric in metrics:
                        writer.writerow({**{key: row[key] for key in columns[:6]}, "role": role, "stage": stage,
                                         "metric": metric, "value": row["metrics"][role][stage][metric]})
    lines = ["# Within-dataset domain transfer", "",
             f"Verified complete triplets: {len(complete)}; incomplete: {len(incomplete)}.", "",
             "The comparison covers source training, adaptation from learned connections, and training from initial gains. "
             "Adaptation and the reset control use the same observations and budget; adaptation additionally inherits source training. "
             "All use target-domain test observations and scalers fitted on source training data. "
             "These are neither external data nor complete OOF evaluations. Seeds may reuse individuals.", "",
             "| Task | Domain | Seed | Metric | Source | Adapted | Reset | n test |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in complete:
        metrics = ("balanced_accuracy", "macro_f1") if row["task"] == "classification" else ("rmse", "mae", "r2")
        for metric in metrics:
            values = [row["metrics"][role]["final"][metric] for role in _ROLES]
            values = ["Undefined" if value is None else f"{value:.6g}" for value in values]
            lines.append(f"| {row['transfer_task']} | {row['transfer_domain']} | {row['split_seed']} | {metric} | " + " | ".join(values) + f" | {row['n_test']} |")
    lines += ["", "Metrics are recalculated from saved predictions. The selected source state must match the initial adaptation state "
              "in both predictions and internal connections. "
              "Group-paired bootstrap intervals, hashes, sizes, budgets, and checks are available in "
              "[transfer_analysis.json](transfer_analysis.json); tabular values in [CSV](transfer_analysis.csv).", ""]
    writable_output(folder / "transfer_analysis.md").write_text("\n".join(lines))
    return report
