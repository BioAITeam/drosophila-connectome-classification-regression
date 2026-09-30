#!/usr/bin/env python3
"""Export manuscript Tables 1 and 2 from the saved Iris OOF evaluations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "3 Connectome"))

from flybrain.paths import REPO, portable_path, repo_path, writable_output


SEEDS = {(1729, 0), (1729, 1), (2718, 0), (2718, 1)}
CLASSIFICATION = (
    ("core", "iris_anatomical_40", "Anatomical subgraph, 730 inputs", 40),
    ("core", "iris_anatomical_120", "Anatomical subgraph, 730 inputs", 120),
    ("full", "iris_full_native_40", "Full anatomical graph, 1,783 inputs", 40),
    ("full", "iris_full_native_120", "Full anatomical graph, 1,783 inputs", 120),
    ("full", "iris_full_matched_40", "Full anatomical graph, 730 matched inputs", 40),
    ("full", "iris_full_matched_120", "Full anatomical graph, 730 matched inputs", 120),
)
TARGETS = ("sepal_length", "sepal_width", "petal_length", "petal_width")


def _read(path, sources):
    path = repo_path(path)
    payload = path.read_bytes()
    sources[str(path.relative_to(REPO))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def _replicas(aggregate, family, task, budget, sources):
    rows = [row for row in aggregate["replicas"] if row["family"] == family]
    if len(rows) != 4 or {(r["split_seed"], r["optimization_seed"]) for r in rows} != SEEDS:
        raise ValueError(f"{family} requires the four declared partition/optimization pairs")
    for row in rows:
        if (row["status"] != "complete" or row["task"] != task or row.get("rewire_seed") is not None
                or row["n_observations"] != 150 or row["n_covered_observations"] != 150
                or row["software_validation_only"] or sorted(row["folds_present"]) != list(range(5))
                or len(row["runs"]) != 5 or sorted(r["fold"] for r in row["runs"]) != list(range(5))):
            raise ValueError(f"{family} has an incomplete or incompatible OOF evaluation")
        for fold in row["runs"]:
            run = repo_path(fold["run"])
            config = _read(run / "config.json", sources)
            result = _read(run / "results.json", sources)
            metadata = _read(run / "dataset/metadata.json", sources)
            if (config["name"] != family or config["epochs"] != budget
                    or config["select_checkpoint"] != "best_validation"
                    or config["split_seed"] != row["split_seed"]
                    or config["optimization_seed"] != row["optimization_seed"]
                    or config["outer_fold"] != fold["fold"] or metadata["task"] != task
                    or result["selected_epoch"] != fold["selected_epoch"]
                    or result["epochs_completed"] != fold["epochs_completed"]
                    or not 0 <= result["selected_epoch"] <= result["epochs_completed"] <= budget):
                raise ValueError(f"Run metadata disagree with the declared table row: {run}")
            if task == "regression" and metadata.get("target_key") != family.removeprefix("iris_").removesuffix("_anatomical_600"):
                raise ValueError(f"Regression target does not match {family}: {run}")
    return rows


def _summary(rows, stage, metric, scale=1):
    values = [float(row["oof_metrics"][stage][metric]) * scale for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Undefined {stage} {metric} in a manuscript table")
    return statistics.mean(values), statistics.stdev(values)


def collect(core, full):
    sources = {}
    aggregates = {"core": _read(core, sources), "full": _read(full, sources)}
    classification = []
    for campaign, family, label, budget in CLASSIFICATION:
        rows = _replicas(aggregates[campaign], family, "classification", budget, sources)
        initial, _ = _summary(rows, "before", "balanced_accuracy", 100)
        selected, sd = _summary(rows, "after", "balanced_accuracy", 100)
        f1, f1_sd = _summary(rows, "after", "macro_f1", 100)
        classification.append({"family": family, "configuration": label, "epoch_budget": budget,
                               "initial_ba_percent": initial, "selected_ba_percent": selected,
                               "selected_ba_sd": sd, "selected_macro_f1_percent": f1,
                               "selected_macro_f1_sd": f1_sd, "n_oof_evaluations": 4,
                               "n_observations_per_evaluation": 150})
    regression = []
    for target in TARGETS:
        family = f"iris_{target}_anatomical_600"
        rows = _replicas(aggregates["core"], family, "regression", 600, sources)
        initial, _ = _summary(rows, "before", "r2")
        result = {"family": family, "target": target.replace("_", " ").capitalize(),
                  "initial_r2": initial, "n_oof_evaluations": 4, "n_observations_per_evaluation": 150}
        for metric in ("r2", "rmse", "mae"):
            result[f"selected_{metric}"], result[f"selected_{metric}_sd"] = _summary(rows, "after", metric)
        regression.append(result)
    return classification, regression, sources


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pm(mean, sd, places, align_sd=False):
    padding = r"\phantom{0}" if align_sd and sd < 10 else ""
    return f"${mean:.{places}f}\\pm{padding}{sd:.{places}f}$"


def _table(caption, label, header, lines):
    return "\n".join([
        r"\begin{table*}[!tbp]", r"\centering", r"\caption{" + caption + "}",
        r"\label{" + label + "}", r"\small", r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule", header + r" \\", r"\midrule", *[line + r" \\" for line in lines],
        r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
    ])


def export(output, core, full):
    output = repo_path(output)
    writable_output(output / "Table-1.csv")
    if any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    classification, regression, sources = collect(core, full)
    _csv(output / "Table-1.csv", classification)
    _csv(output / "Table-2.csv", regression)
    lines = [" & ".join([row["configuration"], str(row["epoch_budget"]),
                         f"{row['initial_ba_percent']:.2f}",
                         _pm(row["selected_ba_percent"], row["selected_ba_sd"], 2),
                         _pm(row["selected_macro_f1_percent"], row["selected_macro_f1_sd"], 2, True)])
             for row in classification]
    text = _table(
        "Iris classification on held-out test folds. Each row summarizes four complete OOF evaluations "
        "of the same 150 flowers. Selected values are mean $\\pm$ sample standard deviation; initial BA is the mean. "
        "BA and macro-F1 are percentages. Checkpoints maximize validation BA.",
        "tab:results-classification",
        r"Configuration & Epoch budget & Initial BA (\%) & Selected BA (\%) & Selected macro-F1 (\%)", lines)
    (output / "Table-1.tex").write_text(text, encoding="utf-8")
    lines = [" & ".join([row["target"], f"{row['initial_r2']:.4f}", *[
        _pm(row[f"selected_{metric}"], row[f"selected_{metric}_sd"], 4) for metric in ("r2", "rmse", "mae")]])
        for row in regression]
    text = _table(
        "Anatomical subgraph Iris regression with a maximum of 600 epochs. Each target is predicted from the other "
        "three measurements. Checkpoints minimize validation RMSE. Selected values are mean $\\pm$ sample standard "
        "deviation across four complete OOF evaluations of the same 150 flowers; initial $R^2$ is the mean. "
        "RMSE and MAE are in centimeters; $R^2$ is dimensionless.",
        "tab:results-regression", r"Target & Initial $R^2$ & Selected $R^2$ & RMSE (cm) & MAE (cm)", lines)
    (output / "Table-2.tex").write_text(text, encoding="utf-8")
    provenance = {"source_sha256": sources, "aggregation": "equal mean and sample SD over four OOF evaluations",
                  "model_training_performed": False, "model_inference_performed": False,
                  "classification_units": "percent", "regression_error_units": "cm"}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New or empty directory inside the checkout")
    parser.add_argument("--core", default="5 Iris/5 Outputs/campaigns/round4-core/aggregate.json")
    parser.add_argument("--full", default="5 Iris/5 Outputs/campaigns/round4-full/aggregate.json")
    args = parser.parse_args()
    print(portable_path(export(args.output, args.core, args.full)))


if __name__ == "__main__":
    main()
