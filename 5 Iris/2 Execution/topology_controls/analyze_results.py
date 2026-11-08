"""Summarize Iris topology and classifier experiments from complete OOF predictions.

No model is fitted here. Partial campaigns produce explicitly incomplete
summaries, and uncertainty intervals are calculated only after every job has
completed successfully.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("balanced_accuracy", "accuracy", "macro_f1")
REPETITIONS = ((1729, 0), (1729, 1), (2718, 0), (2718, 1))
FAMILIES = ("anatomical", "rewired", "random")


def classification_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or not len(truth):
        raise ValueError("Expected equally sized, nonempty label vectors")
    if np.any((truth < 0) | (truth >= 3) | (predicted < 0) | (predicted >= 3)):
        raise ValueError("Iris class labels must be integers from zero through two")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    support, chosen = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = np.diag(matrix)
    recall = np.divide(tp, support, out=np.zeros(3, float), where=support != 0)
    f1 = np.divide(2 * tp, support + chosen, out=np.zeros(3, float), where=support + chosen != 0)
    return {"balanced_accuracy": float(recall[support > 0].mean()),
            "accuracy": float(tp.sum() / len(truth)), "macro_f1": float(f1.mean()),
            "n_images": int(len(truth)), "confusion_matrix": matrix.tolist()}


def mean_sd(values):
    values = [float(x) for x in values]
    return {"mean": statistics.mean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) >= 2 else None,
            "n": len(values)}


def hierarchical_bootstrap(truth, groups, anatomical_predictions, rewired_predictions,
                           random_predictions, replicates=10000, seed=13007):
    """Resample class-stratified input groups and graph identities, keeping four OOF repetitions.

    Prediction shapes are [4, 150] for anatomy and [30, 4, 150] for each
    alternative topology family. Group resamples are shared by every topology
    and repetition. Graph indices are sampled independently within each family.
    """
    truth = np.asarray(truth, dtype=np.int64)
    groups = np.asarray(groups)
    anatomical = np.asarray(anatomical_predictions)
    rewired, random = np.asarray(rewired_predictions), np.asarray(random_predictions)
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    if anatomical.ndim != 2 or anatomical.shape[1] != len(truth):
        raise ValueError("Anatomical predictions must have shape [repetitions, observations]")
    for name, values in [("rewired", rewired), ("random", random)]:
        if values.ndim != 3 or values.shape[1:] != anatomical.shape or not len(values):
            raise ValueError(f"Invalid prediction dimensions for {name}")
    if groups.shape != truth.shape:
        raise ValueError("One duplicate-input group is required per observation")
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    group_sizes = np.bincount(inverse, minlength=len(unique)).astype(float)
    group_classes = np.empty(len(unique), dtype=np.int64)
    for group in range(len(unique)):
        labels = np.unique(truth[inverse == group])
        if len(labels) != 1:
            raise ValueError("Class-stratified bootstrap requires one target class per input group")
        group_classes[group] = labels[0]
    classes = np.unique(truth)
    rng = np.random.default_rng(seed)
    group_counts = np.zeros((replicates, len(unique)), dtype=np.float64)
    class_group_counts = {}
    for label in classes:
        indices = np.flatnonzero(group_classes == label)
        class_group_counts[str(int(label))] = len(indices)
        group_counts[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)),
                                                  size=replicates)
    # Averaging correctness across the fixed repetitions commutes with weighted recall.
    predictions = np.concatenate((anatomical[None], rewired, random), axis=0)
    correct = (predictions == truth[None, None, :]).mean(axis=1)
    grouped_correct = np.zeros((len(unique), len(predictions)), dtype=np.float64)
    np.add.at(grouped_correct, inverse, correct.T)
    ba = np.zeros((replicates, len(predictions)), dtype=np.float64)
    for label in classes:
        mask = group_classes == label
        weights = group_counts[:, mask]
        denominator = weights @ group_sizes[mask]
        if np.any(denominator <= 0):
            raise ValueError("A bootstrap class has no observations")
        ba += (weights @ grouped_correct[mask]) / denominator[:, None] / len(classes)
    anatomical_ba = ba[:, 0]
    draws = {"anatomical_ba_percent": 100 * anatomical_ba}
    offset = 1
    for name, values in [("rewired", rewired), ("random", random)]:
        graph_counts = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=replicates)
        family_ba = (ba[:, offset:offset + len(values)] * graph_counts).sum(axis=1) / len(values)
        draws[name + "_ba_percent"] = 100 * family_ba
        draws["anatomical_minus_" + name + "_pp"] = 100 * (anatomical_ba - family_ba)
        offset += len(values)
    return draws, {"seed": int(seed), "replicates": int(replicates),
                   "n_observations": len(truth), "n_duplicate_input_groups": len(unique),
                   "groups_per_class": class_group_counts,
                   "graph_draws_per_replicate": {"rewired": len(rewired), "random": len(random)},
                   "fixed_oof_repetitions": anatomical.shape[0],
                   "interval": "95% percentile, quantiles 0.025 and 0.975 with linear interpolation",
                   "sampling": "Class-stratified duplicate-input groups sampled with replacement. All members of a sampled group share its multiplicity. The same group sample is used across all graph families and the four fixed OOF repetitions. Graphs are resampled independently within each alternative family.",
                   "interpretation": "Conditional on the fitted models and observed Iris flowers. Repeated partitions and optimization seeds are not additional independent observations."}


def _identifier(value):
    value = str(value)
    if not value or value in (".", "..") or Path(value).name != value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Invalid job or model identifier")
    return value


def _integer_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name} vector")
    if not np.array_equal(value, value.astype(np.int64)):
        raise ValueError(f"Noninteger values in {name}")
    return value.astype(np.int64)


def _timings(result, complete):
    # Read known timer names without manufacturing unavailable measurements.
    aliases = {
        "elapsed_seconds": ("elapsed_seconds", "run_seconds", "total_seconds", "wall_seconds"),
        "training_seconds": ("training_seconds", "fit_seconds", "train_seconds", "selected_fit_seconds"),
        "test_evaluation_seconds": ("test_seconds", "test_evaluation_seconds", "test_prediction_seconds", "test_inference_seconds", "inference_seconds", "predict_seconds"),
        "search_seconds": ("search_seconds", "grid_search_seconds"),
    }
    sources = []
    for obj in [result, complete]:
        for key in ["timings", "timing"]:
            if isinstance(obj.get(key), dict):
                sources.append(obj[key])
        sources.append(obj)
    extracted = {}
    for canonical, options in aliases.items():
        for source in sources:
            found = next((source[k] for k in options if k in source and isinstance(source[k], (int, float))
                          and not isinstance(source[k], bool) and math.isfinite(source[k]) and source[k] >= 0), None)
            if found is not None:
                extracted[canonical] = float(found)
                break
    return extracted


def _csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(mean, sd=None, scale=100):
    if mean is None:
        return "pending"
    if sd is None:
        return f"{scale * mean:.2f} (one complete evaluation)"
    return f"{scale * mean:.2f} +/- {scale * sd:.2f}"


def analyze(*, partial=False, bootstrap=10000, seed=13007, output=None):
    common = importlib.import_module("common")
    root = Path(common.HERE)
    output = Path(output) if output is not None else root / "analysis"
    output = output.resolve()
    if output == root.resolve() or any(output.is_relative_to((root / name).resolve()) for name in ("runs", "graphs", "data")):
        raise ValueError("Analysis output must be separate from campaign inputs and run evidence")
    output.mkdir(parents=True, exist_ok=True)
    spec = common.load_json(root / "jobs.json")
    jobs = spec["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job IDs")
    by_setting = defaultdict(list)
    inventories = {}
    for job in jobs:
        _identifier(job["id"])
        kind = job["kind"]
        if kind not in ("graph", "ml"):
            raise ValueError("Unknown job kind")
        identifier = _identifier(job["graph_id"] if kind == "graph" else job["model"])
        family = job["family"] if kind == "graph" else "ml"
        if kind == "graph" and family not in FAMILIES:
            raise ValueError("Unknown graph family")
        repetition = (int(job["split_seed"]), int(job["optimization_seed"]))
        if repetition not in REPETITIONS or int(job["fold"]) not in range(5):
            raise ValueError("Unexpected partition, optimization seed or fold")
        identity = (kind, identifier)
        item = {"kind": kind, "identifier": identifier, "family": family,
                "graph_seed": job.get("graph_seed")}
        if identity in inventories and inventories[identity] != item:
            raise ValueError("Inconsistent graph or model metadata across jobs")
        inventories[identity] = item
        by_setting[(kind, identifier, *repetition)].append(job)
    if any(sorted(int(j["fold"]) for j in setting) != list(range(5)) for setting in by_setting.values()):
        raise ValueError("Each declared OOF evaluation must contain five distinct folds")
    for kind, identifier in inventories:
        if {key[2:] for key in by_setting if key[:2] == (kind, identifier)} != set(REPETITIONS):
            raise ValueError("Each graph and classifier requires the four declared OOF repetitions")
    counts = {family: sum(v["family"] == family for v in inventories.values()) for family in FAMILIES}
    n_ml = sum(v["kind"] == "ml" for v in inventories.values())
    if counts != {"anatomical": 1, "rewired": 30, "random": 30} or n_ml != 14 or len(jobs) != 1500:
        raise ValueError("Expected 61 graphs, 14 classifiers and 1,500 five-fold jobs")
    partitions = {}
    for split, _ in REPETITIONS:
        for fold in range(5):
            if (split, fold) not in partitions:
                partitions[(split, fold)] = common.load_partition(split, fold)
    truth = _integer_vector(partitions[(1729, 0)]["raw_y"], "raw_y")
    groups = np.asarray(partitions[(1729, 0)]["groups"]).astype(str)
    if len(truth) != 150 or len(np.unique(groups)) != 149 or not np.array_equal(np.bincount(truth, minlength=3), [50, 50, 50]):
        raise ValueError("Expected 150 Iris flowers, 149 input groups and three equally sized classes")
    for key, part in partitions.items():
        if not np.array_equal(truth, part["raw_y"]) or not np.array_equal(groups, np.asarray(part["groups"]).astype(str)):
            raise ValueError("Observation ordering or input groups differ between partitions")
    missing, invalid, completed, runs = [], [], {}, []
    for job in jobs:
        directory = root / "runs" / job["id"]
        if not (directory / "complete.json").is_file():
            missing.append(job["id"])
            continue
        try:
            completion = common.load_json(directory / "complete.json")
            result = common.load_json(directory / "result.json")
            if completion.get("id", job["id"]) != job["id"] or any(result.get(k) != v for k, v in job.items()):
                raise ValueError("Completed record does not match the scheduled job")
            if result.get("software_validation_only") or completion.get("software_validation_only"):
                raise ValueError("Software smoke tests cannot contribute to experimental results")
            if job["kind"] == "graph" and result.get("epochs_completed") != 120:
                raise ValueError("The graph run did not complete the 120-epoch protocol")
            if completion.get("result_sha256") != common.sha256(directory / "result.json"):
                raise ValueError("Completed result checksum mismatch")
            if result.get("predictions_sha256") != common.sha256(directory / "predictions.npz"):
                raise ValueError("Saved prediction checksum mismatch")
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                index = _integer_vector(saved["row_index"], "row_index")
                target = _integer_vector(saved["true"], "true")
                predicted = _integer_vector(saved["predicted"], "predicted")
            expected = np.asarray(partitions[(int(job["split_seed"]), int(job["fold"]))]["index_test"])
            if not np.array_equal(np.sort(index), np.sort(expected)) or len(np.unique(index)) != len(index):
                raise ValueError("Prediction rows do not match the designated test fold")
            if not np.array_equal(target, truth[index]) or predicted.shape != index.shape:
                raise ValueError("Prediction targets or dimensions do not match the input partition")
            classification_metrics(target, predicted)
            completed[job["id"]] = (index, predicted)
            runs.append({"job_id": job["id"], "kind": job["kind"],
                         "identifier": job.get("graph_id", job.get("model")),
                         "family": job.get("family", "ml"), **_timings(result, completion)})
        except (OSError, KeyError, ValueError, TypeError) as exc:
            invalid.append({"job_id": job["id"], "error": str(exc).replace(str(common.REPO) + "/", "")})
    oof = {}
    oof_rows = []
    for key, setting in sorted(by_setting.items()):
        if not all(job["id"] in completed for job in setting):
            continue
        predicted = np.full(len(truth), -1, dtype=np.int64)
        appearances = np.zeros(len(truth), dtype=np.int64)
        for job in setting:
            index, values = completed[job["id"]]
            predicted[index] = values
            np.add.at(appearances, index, 1)
        if not np.all(appearances == 1):
            raise ValueError("A completed five-fold OOF evaluation does not cover every flower exactly once")
        kind, identifier, split, optimization = key
        metrics = classification_metrics(truth, predicted)
        oof[key] = {"predicted": predicted, "metrics": metrics}
        destination = output / "oof" / (kind + "-" + identifier)
        destination.mkdir(parents=True, exist_ok=True)
        filename = destination / f"split-{split}-optimization-{optimization}.npz"
        np.savez_compressed(filename, row_index=np.arange(len(truth)), true=truth, predicted=predicted,
                            groups=groups, split_seed=split, optimization_seed=optimization)
        oof_rows.append({"kind": kind, "identifier": identifier, "split_seed": split,
                         "optimization_seed": optimization, **{m: metrics[m] for m in METRICS},
                         "predictions": filename.relative_to(output).as_posix()})
    entities = []
    for key, inventory in sorted(inventories.items()):
        available = [oof[(*key, *repetition)] for repetition in REPETITIONS if (*key, *repetition) in oof]
        selected_jobs = [j for j in jobs if (j["kind"], j.get("graph_id", j.get("model"))) == key]
        row = {**inventory, "complete_oof_evaluations": len(available), "expected_oof_evaluations": 4,
               "complete_fold_runs": sum(j["id"] in completed for j in selected_jobs),
               "expected_fold_runs": 20, "complete": len(available) == 4}
        for metric in METRICS:
            summary = mean_sd(item["metrics"][metric] for item in available)
            row[metric + "_mean"] = summary["mean"]
            row[metric + "_sd"] = summary["sd"]
        entities.append(row)
    family_rows = []
    for family in FAMILIES:
        full = [r for r in entities if r["family"] == family and r["complete"]]
        row = {"family": family, "complete_graphs": len(full), "expected_graphs": counts[family],
               "complete": len(full) == counts[family],
               "dispersion_unit": "four OOF evaluations of one anatomical graph" if family == "anatomical"
                                    else "graph means after averaging four OOF evaluations per graph"}
        for metric in METRICS:
            if family == "anatomical" and full:
                row[metric + "_mean"], row[metric + "_sd"] = full[0][metric + "_mean"], full[0][metric + "_sd"]
            else:
                values = mean_sd(r[metric + "_mean"] for r in full)
                row[metric + "_mean"], row[metric + "_sd"] = values["mean"], values["sd"]
        family_rows.append(row)
    is_complete = len(completed) == len(jobs) and not invalid
    differences, bootstrap_metadata = [], None
    if is_complete:
        graph_ids = {family: sorted(r["identifier"] for r in entities if r["family"] == family) for family in FAMILIES}
        def predictions(identifier):
            return np.stack([oof[("graph", identifier, *r)]["predicted"] for r in REPETITIONS])
        draws, bootstrap_metadata = hierarchical_bootstrap(truth, groups,
            predictions(graph_ids["anatomical"][0]), np.stack([predictions(i) for i in graph_ids["rewired"]]),
            np.stack([predictions(i) for i in graph_ids["random"]]), bootstrap, seed)
        np.savez_compressed(output / "bootstrap_replicates.npz", **draws,
                            seed=np.int64(seed), anatomical_graph=np.asarray(graph_ids["anatomical"]),
                            rewired_graphs=np.asarray(graph_ids["rewired"]), random_graphs=np.asarray(graph_ids["random"]))
        anatomy_mean = next(r["balanced_accuracy_mean"] for r in family_rows if r["family"] == "anatomical")
        for family in ("rewired", "random"):
            alternative_mean = next(r["balanced_accuracy_mean"] for r in family_rows if r["family"] == family)
            delta = draws["anatomical_minus_" + family + "_pp"]
            lo, hi = np.quantile(delta, [0.025, 0.975], method="linear")
            differences.append({"comparison": "anatomical minus " + family,
                                "effect_size_pp": 100 * (anatomy_mean - alternative_mean),
                                "ci95_low_pp": float(lo), "ci95_high_pp": float(hi),
                                "bootstrap_replicates": bootstrap, "bootstrap_seed": seed})
    elif (output / "bootstrap_replicates.npz").exists():
        # Prevent a stale final interval from appearing beside a later partial report.
        (output / "bootstrap_replicates.npz").unlink()
    timing_rows = []
    for scope in ["all", *FAMILIES, "ml"]:
        selected = runs if scope == "all" else [r for r in runs if r["family"] == scope]
        for timer in ("elapsed_seconds", "training_seconds", "test_evaluation_seconds", "search_seconds"):
            values = [r[timer] for r in selected if timer in r]
            timing_rows.append({"scope": scope, "timer": timer, "measured_runs": len(values),
                                "valid_completed_runs": len(selected),
                                "accumulated_seconds": math.fsum(values) if values else None,
                                "median_seconds": statistics.median(values) if values else None,
                                "p25_seconds": float(np.quantile(values, 0.25)) if values else None,
                                "p75_seconds": float(np.quantile(values, 0.75)) if values else None})
    summary = {"status": "complete" if is_complete else "partial", "analysis_version": 1,
               "source_sha256": common.sha256(Path(__file__)),
               "expected_jobs": len(jobs), "valid_completed_jobs": len(completed),
               "missing_jobs": missing, "invalid_jobs": invalid,
               "complete_oof_evaluations": len(oof), "expected_oof_evaluations": len(by_setting),
               "observations": len(truth), "input_groups": len(np.unique(groups)),
               "repetitions": [{"split_seed": s, "optimization_seed": o} for s, o in REPETITIONS],
               "metrics": "Metrics use all 150 OOF predictions, never a mean of fold metrics. Standard deviations use ddof=1.",
               "family_metrics": family_rows, "graph_metrics": [r for r in entities if r["kind"] == "graph"],
               "ml_metrics": [r for r in entities if r["kind"] == "ml"],
               "differences": differences, "bootstrap": bootstrap_metadata,
               "timings": timing_rows,
               "timing_interpretation": "Accumulated measured run durations are not campaign elapsed time. Missing timer fields are not replaced by zero. Timers retain the recorded scope of each worker.",
               "partial_interpretation": None if is_complete else "Only complete five-fold OOF evaluations are summarized. Incomplete entities identify their available evaluations. Family statistics use only graphs with all four complete evaluations. No uncertainty intervals are reported until all 1,500 runs are complete."}
    common.save_json(output / "summary.json", summary)
    _csv(output / "graph_metrics.csv", summary["graph_metrics"])
    _csv(output / "ml_metrics.csv", summary["ml_metrics"])
    _csv(output / "oof_metrics.csv", oof_rows)
    _csv(output / "family_metrics.csv", family_rows)
    _csv(output / "differences.csv", differences)
    _csv(output / "timings.csv", timing_rows)
    _csv(output / "run_timings.csv", runs)
    lines = ["# Iris topology and classifier results", "", "Status: **" + summary["status"] + "**.", "",
             f"Valid completed runs: {len(completed)}/{len(jobs)}. Complete five-fold OOF evaluations: {len(oof)}/{len(by_setting)}.", ""]
    if not is_complete:
        lines += [summary["partial_interpretation"], ""]
    lines += ["## Graph families", "", "| Family | Complete graphs | BA (%) | Accuracy (%) | Macro-F1 (%) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in family_rows:
        lines.append("| " + " | ".join([row["family"], f"{row['complete_graphs']}/{row['expected_graphs']}",
                     *[_format(row[m + "_mean"], row[m + "_sd"]) for m in METRICS]]) + " |")
    lines += ["", "For anatomy, the standard deviation describes four OOF evaluations of the same graph. For each alternative family, it describes the graph means after averaging four OOF evaluations per graph.",
              "", "## Traditional classifiers", "", "| Model | Complete OOF evaluations | BA (%) | Accuracy (%) | Macro-F1 (%) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["ml_metrics"]:
        lines.append("| " + " | ".join([row["identifier"], f"{row['complete_oof_evaluations']}/4",
                     *[_format(row[m + "_mean"], row[m + "_sd"]) for m in METRICS]]) + " |")
    if differences:
        lines += ["", "## Paired topology differences", "", "| Comparison | BA difference (pp) | Conditional 95% interval (pp) |",
                  "| --- | ---: | ---: |"]
        for row in differences:
            lines.append(f"| {row['comparison']} | {row['effect_size_pp']:.2f} | [{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] |")
        lines += ["", bootstrap_metadata["sampling"], "", bootstrap_metadata["interpretation"]]
    lines += ["", "Timing summaries are in `timings.csv`. They sum or summarize recorded run durations and do not represent campaign elapsed time."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": summary["status"], "completed_jobs": len(completed), "expected_jobs": len(jobs),
                      "complete_oof": len(oof), "output": output.name}, indent=2))
    if not partial and not is_complete:
        raise RuntimeError("Campaign is incomplete. See the saved partial summary, or use --partial for progress reporting.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13007)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, help="Destination for regenerated analysis (default: campaign/analysis)")
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    analyze(partial=args.partial, bootstrap=args.bootstrap, seed=args.seed, output=args.output)


if __name__ == "__main__":
    main()
