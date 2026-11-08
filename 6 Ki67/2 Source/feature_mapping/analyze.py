#!/usr/bin/env python3
"""Analyze the paired frozen-Phikon-v2 input-assignment experiment.

Run from any working directory. All persisted paths are repository-relative.
Only completed, provenance-checked runs enter the final analysis. Bootstrap
intervals condition on the fitted models and use images, not patients, as the
sampling units. Training seeds are paired and are not resampled as independent
observations. The family comparison additionally resamples permutation IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np


METRICS = ("balanced_accuracy", "accuracy", "macro_f1")
SCOPES = ("pooled", "bcdata", "shid")
from runtime import REPO, ARCHIVE, WORKSPACE, activate, writable
DEFAULT_EXPERIMENT = ARCHIVE


def portable(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def resolve_relative(value: str, base: Path = REPO) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Stored paths must be relative: " + path.name)
    result = (base / path).resolve()
    result.relative_to(base.resolve())
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=".analysis-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(document) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_protocol(protocol: dict):
    seeds = protocol["training_seeds"]
    if seeds != [1729, 2718, 3141]:
        raise ValueError("Expected the three paired training seeds 1729, 2718, 3141.")
    mappings = protocol["mappings"]
    if len(mappings) != 11 or [m["id"] for m in mappings] != [f"mapping-{i:02d}" for i in range(11)]:
        raise ValueError("Expected mapping-00 identity followed by ten permutations.")
    expected = np.arange(1024, dtype=np.int64)
    hashes = []
    for i, mapping in enumerate(mappings):
        permutation = np.asarray(mapping["permutation"], dtype="<i8")
        if permutation.shape != (1024,) or not np.array_equal(np.sort(permutation), expected):
            raise ValueError("Mapping is not a permutation of all 1,024 feature coordinates.")
        computed = hashlib.sha256(permutation.tobytes()).hexdigest()
        recorded = mapping.get("permutation_sha256", mapping.get("sha256"))
        if computed != recorded:
            raise ValueError("Permutation digest mismatch for " + mapping["id"])
        hashes.append(computed)
        if i == 0 and (not np.array_equal(permutation, expected) or mapping["mapping_seed"] is not None):
            raise ValueError("mapping-00 must be the unchanged coordinate assignment.")
    if len(set(hashes)) != 11:
        raise ValueError("Mappings must be distinct.")
    epochs = int(protocol["training"]["classification_epochs"])
    if epochs != 120:
        raise ValueError("The final campaign analysis requires the 120-epoch protocol.")
    return mappings, seeds, epochs


def verify_artifacts(protocol: dict, experiment: Path) -> dict:
    """Check sealed inputs and training implementation before final analysis."""
    checked = {}
    for group in ("source_hashes", "implementation_sha256", "campaign_hashes"):
        hashes = protocol.get(group)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Missing sealed provenance group: " + group)
        base = experiment if group == "campaign_hashes" else REPO
        for name, expected in hashes.items():
            path = resolve_relative(name, base)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Sealed artifact mismatch: {group}/{name}")
        checked[group] = len(hashes)
    return checked


def validate_selection(history: list[dict], best: dict, epochs: int, min_delta: float):
    """Replay validation-only selection, including initialization and ties."""
    if [row["epoch"] for row in history] != list(range(epochs + 1)):
        raise ValueError("Expected exactly 121 unique, ordered epochs, including initialization.")
    if any(row.get("attempt", 0) != 0 for row in history):
        raise ValueError("This campaign expects a single uninterrupted attempt for each run.")
    selected_epoch, selected_score = None, -float("inf")
    selected_row = None
    for row in history:
        score = float(row["val"]["balanced_accuracy"])
        if not np.isfinite(score):
            raise ValueError("Non-finite validation BA in training history.")
        improved = score > selected_score + min_delta
        if improved:
            selected_epoch, selected_score, selected_row = row["epoch"], score, row
        if row.get("selected") != improved or row.get("best_epoch") != selected_epoch:
            raise ValueError("Recorded epoch selection differs from validation-only selection.")
    if (best["epoch"] != selected_epoch or best["score"] != selected_score
            or best["prediction_path"] != selected_row["prediction_path"]
            or best["validation_metrics"] != selected_row["val"]
            or best["test_metrics"] != selected_row["test"]):
        raise ValueError("Selected checkpoint is not the earliest validation optimum under min_delta.")


def prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = ("y_true", "y_pred", "probabilities", "sample_id", "source_dataset", "epoch", "split")
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"Prediction archive missing fields: {sorted(missing)}")
        result = {key: np.array(data[key]) for key in required}
    y = result["y_true"].astype(np.int64)
    pred = result["y_pred"].astype(np.int64)
    probabilities = result["probabilities"]
    n = len(y)
    if (y.shape != (n,) or pred.shape != (n,) or probabilities.shape != (n, 3)
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.array_equal(pred, probabilities.argmax(1))
            or not set(y.tolist()).issubset({0, 1, 2})
            or str(result["split"].item()) != "test"):
        raise ValueError("Invalid three-class test predictions in " + portable(path))
    datasets = result["source_dataset"].astype(str)
    ids = result["sample_id"].astype(str)
    if datasets.shape != (n,) or ids.shape != (n,):
        raise ValueError("Prediction identifiers are misaligned.")
    if (n != 1102 or set(datasets.tolist()) != {"bcdata", "shid"}
            or int(np.sum(datasets == "bcdata")) != 402
            or int(np.sum(datasets == "shid")) != 700):
        raise ValueError("Expected the combined BCData (402) and SHIDC-B-Ki-67-V2 (700) test sets.")
    if len(set(zip(datasets.tolist(), ids.tolist()))) != n:
        raise ValueError("Repeated dataset/sample_id pair in prediction archive.")
    result.update(y_true=y, y_pred=pred, sample_id=ids, source_dataset=datasets)
    return result


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = np.bincount(3 * y + pred, minlength=9).reshape(3, 3).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    if np.any(support == 0):
        raise ValueError("Every reported scope must contain all three classes.")
    diagonal = np.diag(cm)
    f1 = np.divide(2 * diagonal, support + predicted,
                   out=np.zeros(3), where=support + predicted != 0)
    return {"balanced_accuracy": float(np.mean(diagonal / support)),
            "accuracy": float(diagonal.sum() / cm.sum()), "macro_f1": float(f1.mean())}


def check_metrics(document: dict, payload: dict):
    for scope in SCOPES:
        mask = np.ones(len(payload["y_true"]), dtype=bool) if scope == "pooled" else payload["source_dataset"] == scope
        observed = point_metrics(payload["y_true"][mask], payload["y_pred"][mask])
        recorded = document if scope == "pooled" else document["by_dataset"][scope]
        for metric in METRICS:
            if not np.isclose(observed[metric], recorded[metric], atol=1e-8, rtol=1e-7):
                raise ValueError(f"Saved metric differs from predictions: {scope}/{metric}")


def load_run(directory: Path, mapping: dict, seed: int, epochs: int, training: dict):
    import torch

    complete = read_json(directory / "complete.json")
    best = read_json(directory / "best.json")
    config = read_json(directory / "config.json")
    history = read_json(directory / "history.json")
    identity = {"dataset": "joint", "representation": "phikonv2", "task": "multiclass",
                "graph": "anatomical", "seed": seed}
    for document in (complete, best, config):
        if any(document.get(key) != value for key, value in identity.items()):
            raise ValueError("Run identity mismatch at " + portable(directory))
    if complete["last_epoch"] != epochs or complete["best_epoch"] != best["epoch"]:
        raise ValueError("Run has not completed the required epoch budget.")
    if config["training"] != training:
        raise ValueError("Run hyperparameters differ from the sealed training protocol.")
    validate_selection(history, best, epochs, float(training["min_delta"]))
    assignment = config["input_mapping"]
    expected_hash = mapping.get("permutation_sha256", mapping.get("sha256"))
    if (assignment["mapping_id"] != mapping["id"]
            or assignment["mapping_seed"] != mapping["mapping_seed"]
            or assignment["permutation_sha256"] != expected_hash):
        raise ValueError("Checkpoint config does not match the assigned mapping.")
    mapping_path = resolve_relative(assignment["permutation_path"])
    if read_json(mapping_path) != mapping:
        raise ValueError("Stored mapping differs from its sealed permutation definition.")
    if best["criterion"] != "validation_balanced_accuracy" or best.get("selection_scope") != "pooled_validation":
        raise ValueError("Checkpoint must be selected using combined validation BA.")
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    if (saved.get("spec") != config or saved.get("spec_digest") != digest(config)
            or saved.get("best_metadata") != best or saved.get("epoch") != best["epoch"]
            or saved.get("score") != best["score"]):
        raise ValueError("Selected checkpoint provenance does not match its metadata.")
    gains = saved.get("gain")
    if not isinstance(gains, torch.Tensor) or gains.numel() != 40755 or not torch.isfinite(gains).all():
        raise ValueError("Checkpoint must contain 40,755 finite anatomical gains.")
    initial_rows = [row for row in history if row["epoch"] == 0]
    if not initial_rows or set(range(epochs + 1)) - {row["epoch"] for row in history}:
        raise ValueError("History does not contain initialization and all training epochs.")
    initial_row = initial_rows[0]
    initial_path = resolve_relative(initial_row["prediction_path"], directory)
    selected_path = resolve_relative(best["prediction_path"], directory)
    initial, selected = prediction(initial_path), prediction(selected_path)
    if initial["epoch"].item() != 0 or selected["epoch"].item() != best["epoch"]:
        raise ValueError("Prediction epoch differs from the recorded checkpoint epoch.")
    check_metrics(initial_row["test"], initial)
    check_metrics(best["test_metrics"], selected)
    for key in ("sample_id", "source_dataset", "y_true"):
        if not np.array_equal(initial[key], selected[key]):
            raise ValueError("Initialization and selected predictions have different sample orders or labels.")
    metadata = {"mapping_id": mapping["id"], "mapping_seed": mapping["mapping_seed"],
                "training_seed": seed, "run_directory": portable(directory),
                "selected_epoch": int(best["epoch"]), "validation_ba": float(best["score"]),
                "elapsed_seconds": float(complete["elapsed_seconds"]),
                "checkpoint_sha256": sha256(directory / "best.pt"),
                "initial_predictions_sha256": sha256(initial_path),
                "selected_predictions_sha256": sha256(selected_path),
                "permutation_sha256": expected_hash}
    return metadata, initial, selected


def bootstrap_ba(correct: np.ndarray, y: np.ndarray, datasets: np.ndarray,
                 repetitions: int, seed: int, output: Path | None = None):
    """Shared image resampling for the 2 x 11 x 3 prediction cube.

    Dataset/class strata retain their observed counts. The same image weights
    apply to every mapping, training seed and model stage. Family confidence
    intervals also resample the ten permutation IDs, keeping each selected ID's
    three training seeds together. This is not resampling 33 independent cohorts.
    """
    rng = np.random.default_rng(seed)
    stages, n_mappings, n_seeds, n_images = correct.shape
    flat = correct.reshape(-1, n_images).astype(np.float64)
    matrices = []
    for scope in SCOPES:
        mask = np.ones(n_images, bool) if scope == "pooled" else datasets == scope
        weight = np.zeros(n_images, dtype=float)
        for class_id in range(3):
            members = mask & (y == class_id)
            weight[members] = 1.0 / (3 * members.sum())
        matrices.append((flat * weight[None, :]).T)
    strata = [np.flatnonzero((datasets == dataset) & (y == class_id))
              for dataset in ("bcdata", "shid") for class_id in range(3)]
    boot = np.empty((repetitions, len(SCOPES), stages, n_mappings, n_seeds), dtype=np.float64)
    draws = np.empty((repetitions, 10), dtype=np.int64)
    for start in range(0, repetitions, 128):
        stop = min(start + 128, repetitions)
        weights = np.zeros((stop - start, n_images), dtype=np.float64)
        for indices in strata:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                                 size=stop - start)
        for scope_index, matrix in enumerate(matrices):
            boot[start:stop, scope_index] = (weights @ matrix).reshape(-1, stages, n_mappings, n_seeds)
        draws[start:stop] = rng.integers(0, 10, size=(stop - start, 10))
    if output is not None:
        np.savez_compressed(output / "bootstrap-replicates.npz",
                            balanced_accuracy=boot, permutation_draws=draws,
                            scopes=np.asarray(SCOPES), stages=np.asarray(["initial", "selected"]),
                            training_seeds=np.asarray([1729, 2718, 3141]),
                            bootstrap_seed=np.asarray(seed))
    mean_seeds = boot.mean(axis=-1)
    intervals = []
    for scope_index, scope in enumerate(SCOPES):
        selected = mean_seeds[:, scope_index, 1]
        initial = mean_seeds[:, scope_index, 0]
        selected_family = np.take_along_axis(selected[:, 1:], draws, axis=1).mean(1)
        initial_family = np.take_along_axis(initial[:, 1:], draws, axis=1).mean(1)
        samples = {"identity_selected_ba": selected[:, 0],
                   "permutation_family_selected_ba": selected_family,
                   "permutation_family_minus_identity_pp": 100 * (selected_family - selected[:, 0]),
                   "identity_training_gain_pp": 100 * (selected[:, 0] - initial[:, 0]),
                   "permutation_family_training_gain_pp": 100 * (selected_family - initial_family)}
        for i in range(1, 11):
            samples[f"mapping-{i:02d}_minus_identity_pp"] = 100 * (selected[:, i] - selected[:, 0])
        for comparison, values in samples.items():
            low, high = np.quantile(values, [0.025, 0.975])
            intervals.append({"scope": scope, "quantity": comparison,
                              "lower_95": float(low), "upper_95": float(high),
                              "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
                              "sampling": "images_and_permutation_ids" if "family" in comparison else "images",
                              "units": "percentage_points" if comparison.endswith("_pp") else "proportion"})
    return intervals


def historical_comparison(protocol: dict, run_metadata: list[dict], selected_payloads: list[dict]):
    source = protocol.get("source_campaign", protocol.get("source"))
    if isinstance(source, dict):
        source = source.get("campaign", source.get("path"))
    if not isinstance(source, str):
        return {"status": "source_campaign_not_specified"}
    root = resolve_relative(source)
    comparisons = []
    for metadata, payload in zip(run_metadata[:3], selected_payloads[:3]):
        directory = root / "runs/joint/phikonv2/multiclass/anatomical" / f"seed-{metadata['training_seed']}"
        best_path = directory / "best.json"
        if not best_path.is_file():
            return {"status": "source_best_metadata_unavailable", "source_campaign": source}
        old = read_json(best_path)
        old_prediction = prediction(resolve_relative(old["prediction_path"], directory))
        for key in ("sample_id", "source_dataset", "y_true"):
            if not np.array_equal(old_prediction[key], payload[key]):
                raise ValueError("The identity control and source campaign use different test samples or labels.")
        check_metrics(old["test_metrics"], old_prediction)
        observed = point_metrics(payload["y_true"], payload["y_pred"])["balanced_accuracy"]
        comparisons.append({"training_seed": metadata["training_seed"],
                            "historical_selected_epoch": old["epoch"],
                            "repeated_selected_epoch": metadata["selected_epoch"],
                            "historical_pooled_ba": old["test_metrics"]["balanced_accuracy"],
                            "repeated_pooled_ba": observed,
                            "predicted_label_agreement": float(np.mean(old_prediction["y_pred"] == payload["y_pred"])),
                            "maximum_probability_difference": float(np.max(np.abs(old_prediction["probabilities"] - payload["probabilities"]))),
                            "difference_pp": 100 * (observed - old["test_metrics"]["balanced_accuracy"])})
    return {"status": "compared", "source_campaign": source, "runs": comparisons,
            "maximum_absolute_ba_difference_pp": max(abs(row["difference_pp"]) for row in comparisons)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="Experiment folder relative to the repository root.")
    parser.add_argument("--output", default=WORKSPACE + "/analysis",
                        help="Fresh analysis folder under 6 Ki67/6 Workspace.")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--partial", action="store_true",
                        help="Write progress and return successfully if the campaign is incomplete.")
    args = parser.parse_args(argv)
    if args.bootstrap < 1000:
        parser.error("Use at least 1,000 bootstrap repetitions for final confidence intervals.")
    experiment = resolve_relative(args.experiment)
    protocol = read_json(experiment / "protocol.json")
    mappings, seeds, epochs = validate_protocol(protocol)
    output = writable(resolve_relative(args.output))
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(mapping, seed, experiment / "mappings" / mapping["id"] /
             "runs/joint/phikonv2/multiclass/anatomical" / f"seed-{seed}")
            for mapping in mappings for seed in seeds]
    required = ("best.json", "complete.json", "history.json", "config.json", "best.pt")
    missing = [{"mapping_id": mapping["id"], "training_seed": seed,
                "missing": [name for name in required if not (directory / name).is_file()]}
               for mapping, seed, directory in jobs if any(not (directory / name).is_file() for name in required)]
    status = {"created_at": datetime.now(timezone.utc).isoformat(), "expected_runs": 33,
              "runs_with_required_files": 33 - len(missing), "missing_runs": missing,
              "analysis_complete": False, "protocol_sha256": sha256(experiment / "protocol.json")}
    write_json(output / "completion-status.json", status)
    if missing:
        print(json.dumps(status, indent=2))
        if args.partial:
            return 0
        raise RuntimeError(f"Final analysis requires all 33 runs; {len(missing)} remain incomplete.")
    checked_artifacts = verify_artifacts(protocol, experiment)
    metadata, initial_payloads, selected_payloads = [], [], []
    for mapping, seed, directory in jobs:
        record, initial, selected = load_run(directory, mapping, seed, epochs, protocol["training"])
        if initial_payloads:
            for key in ("sample_id", "source_dataset", "y_true"):
                if not np.array_equal(initial[key], initial_payloads[0][key]):
                    raise ValueError("Sample IDs, labels or ordering differ across training runs.")
        metadata.append(record)
        initial_payloads.append(initial)
        selected_payloads.append(selected)
    rows = []
    for record, initial, selected in zip(metadata, initial_payloads, selected_payloads):
        for stage, payload in (("initial", initial), ("selected", selected)):
            for scope in SCOPES:
                mask = np.ones(len(payload["y_true"]), bool) if scope == "pooled" else payload["source_dataset"] == scope
                rows.append({"mapping_id": record["mapping_id"], "mapping_seed": record["mapping_seed"],
                             "training_seed": record["training_seed"], "stage": stage, "scope": scope,
                             "n_images": int(mask.sum()), "epoch": int(payload["epoch"].item()),
                             **point_metrics(payload["y_true"][mask], payload["y_pred"][mask])})
    mapping_summary, differences, family_summary = [], [], []
    for scope in SCOPES:
        for stage in ("initial", "selected"):
            for metric in METRICS:
                values = np.asarray([[next(r[metric] for r in rows if r["mapping_id"] == mapping["id"]
                                          and r["training_seed"] == seed and r["stage"] == stage and r["scope"] == scope)
                                      for seed in seeds] for mapping in mappings])
                for i, mapping in enumerate(mappings):
                    mapping_summary.append({"mapping_id": mapping["id"], "scope": scope, "stage": stage,
                                            "metric": metric, "mean": float(values[i].mean()),
                                            "sample_sd_across_training_seeds": float(values[i].std(ddof=1)),
                                            "n_training_seeds": 3})
                    if i > 0:
                        paired = 100 * (values[i] - values[0])
                        differences.append({"mapping_id": mapping["id"], "scope": scope, "stage": stage,
                                            "metric": metric, "difference_direction": "permuted_minus_identity",
                                            "mean_difference_pp": float(paired.mean()),
                                            "sample_sd_paired_seed_differences_pp": float(paired.std(ddof=1))})
                means = values[1:].mean(1)
                family_summary.append({"scope": scope, "stage": stage, "metric": metric,
                                       "identity_mean": float(values[0].mean()),
                                       "identity_sample_sd_across_training_seeds": float(values[0].std(ddof=1)),
                                       "permutations_mean": float(means.mean()),
                                       "sample_sd_across_ten_mapping_means": float(means.std(ddof=1)),
                                       "minimum_mapping_mean": float(means.min()),
                                       "maximum_mapping_mean": float(means.max()),
                                       "permutations_minus_identity_pp": float(100 * (means.mean() - values[0].mean()))})
    reference = initial_payloads[0]
    correct = np.stack([[payload["y_pred"] == payload["y_true"] for payload in collection]
                        for collection in (initial_payloads, selected_payloads)]).reshape(2, 11, 3, -1)
    intervals = bootstrap_ba(correct, reference["y_true"], reference["source_dataset"], args.bootstrap, args.seed, output)
    history_check = historical_comparison(protocol, metadata, selected_payloads)
    write_csv(output / "run-metrics.csv", rows)
    write_csv(output / "mapping-summary.csv", mapping_summary)
    write_csv(output / "paired-differences.csv", differences)
    write_csv(output / "family-summary.csv", family_summary)
    write_csv(output / "confidence-intervals.csv", intervals)
    write_json(output / "run-provenance.json", metadata)
    write_json(output / "historical-identity-comparison.json", history_check)
    document = {"experiment": portable(experiment), "protocol_sha256": status["protocol_sha256"],
                "completed_runs": 33, "training_seeds": seeds, "permutations": 10,
                "verified_sealed_artifacts": checked_artifacts,
                "bootstrap_repetitions": args.bootstrap, "bootstrap_seed": args.seed,
                "interval_method": "95% percentile, shared image resampling within dataset/class strata; permutation IDs also resampled for family estimates",
                "scope": "Conditional on fitted models and observed images; assumes image independence within strata; not patient-level inference or uncertainty from retraining.",
                "training_seed_handling": "Metrics averaged across the same three paired seeds; seeds not independently resampled.",
                "family_summary": family_summary, "confidence_intervals": intervals,
                "historical_identity_check": history_check}
    write_json(output / "summary.json", document)
    names = {"pooled": "Combined", **protocol.get("dataset_names", {"bcdata": "BCData", "shid": "SHIDC-B-Ki-67-V2"})}
    lines = ["# Phikon-v2 feature-assignment sensitivity", "", "All 33 runs completed and passed checkpoint, prediction and sample-alignment validation.", "",
             "Identity retains the original coordinate order. Ten fixed, distinct permutations each reorder all 1,024 standardized descriptors before the unchanged two-block interface. All mappings use the same three training seeds and partitions.", "",
             "| Test scope | Identity BA, mean ± seed SD (%) | Permuted BA, mean ± mapping SD (%) | Permuted minus identity (pp), 95% CI |", "|---|---:|---:|---:|"]
    for scope in SCOPES:
        result = next(r for r in family_summary if r["scope"] == scope and r["stage"] == "selected" and r["metric"] == "balanced_accuracy")
        ci = next(r for r in intervals if r["scope"] == scope and r["quantity"] == "permutation_family_minus_identity_pp")
        lines.append(f"| {names[scope]} | {100*result['identity_mean']:.2f} ± {100*result['identity_sample_sd_across_training_seeds']:.2f} | {100*result['permutations_mean']:.2f} ± {100*result['sample_sd_across_ten_mapping_means']:.2f} | {result['permutations_minus_identity_pp']:+.2f} [{ci['lower_95']:+.2f}, {ci['upper_95']:+.2f}] |")
    lines += ["", "Identity SD describes the three training runs. Permutation SD describes the ten mapping means, each averaged over three training runs. These are different sources of variation and are labeled separately.", "",
              f"Confidence intervals use {args.bootstrap:,} shared image bootstrap samples stratified by dataset and true class, with additional resampling of the ten mapping IDs for family estimates. Training seeds remain paired. The intervals condition on these fitted models and assume independence among sampled images. They do not estimate uncertainty across patients or retrained cohorts.", "",
              "`run-metrics.csv` contains initialization and validation-selected BA, accuracy and macro-F1 for each run and test scope. `mapping-summary.csv` averages the three training seeds. `paired-differences.csv` preserves their pairing. `run-provenance.json` records checkpoint and prediction hashes. `historical-identity-comparison.json` checks the identity control against the source campaign.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    status.update(analysis_complete=True, validated_runs=33, bootstrap_repetitions=args.bootstrap,
                  analysis_directory=portable(output))
    write_json(output / "completion-status.json", status)
    print(json.dumps({"analysis_complete": True, "validated_runs": 33, "output": portable(output)}, indent=2))
    return 0


if __name__ == "__main__":
    activate()
    raise SystemExit(main())
