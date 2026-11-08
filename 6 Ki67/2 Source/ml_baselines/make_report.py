"""Summarize completed Ki-67 baseline experiments and matched connectome runs.

Reports are written to the experiment's summary directory. Incomplete results
remain pending. Model selection uses validation performance.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from .data import ARCHIVE, DATASETS, PATIENT_DISJOINT, REPRESENTATIONS, TASKS
from .paths import REPO, PUBLISHED as OUTPUT, WORKSPACE, writable, relative_path
from .data import CLASS_NAMES, labels_from_counts, manifest, sha256
from .models import MODEL_NAMES
from .repeat_results import apply_selected_repeats


DATASET_NAMES = {"bcdata": "BCData", "shid": "SHIDC-B-Ki-67", "shid_v2": "SHIDC-B-Ki-67-V2"}
REPRESENTATION_NAMES = {"rgb": "RGB", "dinov2": "DINOv2", "phikonv2": "Phikon-v2"}
MODEL_LABELS = {
    "knn": "k-nearest neighbors", "decision_tree": "Decision tree", "bagging": "Bagging",
    "random_forest": "Random forest", "adaboost": "AdaBoost", "svc": "Support vector classifier",
    "extra_trees": "Extra Trees", "xgboost": "XGBoost", "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting", "qda": "Quadratic discriminant analysis",
    "lda": "Linear discriminant analysis", "gaussian_nb": "Gaussian naive Bayes", "mlp": "MLP",
    "anatomical_connectome": "Anatomical connectome",
}
SEEDS = (1729, 2718, 3141)
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc")
HEADS = ("logistic_regression", "mlp", "anatomical_connectome")
ALL_REPRESENTATIONS = tuple(REPRESENTATIONS)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: dict) -> None:
    write_text(path, json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def relative(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def campaign_representations(repo: Path) -> tuple[str, ...]:
    path = repo / OUTPUT / "provenance/campaign-scope.json"
    if not path.exists():
        return ALL_REPRESENTATIONS
    scope = read(path)
    included = tuple(scope["included_representations"])
    excluded = set(scope.get("excluded_representations", []))
    if (not included or len(set(included)) != len(included)
            or not set(included).issubset(ALL_REPRESENTATIONS)
            or set(included) & excluded):
        raise ValueError("Invalid campaign representation scope")
    return included


def expected_contexts() -> list[tuple[str, str, str, str]]:
    return [(dataset, representation, task, model) for dataset in DATASETS
            for representation in REPRESENTATIONS for task in TASKS for model in MODEL_NAMES
            if not (representation == "rgb" and model == "mlp")]


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("A source metric is non-finite")
    return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0}


def format_ba(result: dict | None) -> str:
    if result is None:
        return "Pending"
    value = result["test_metrics"]["balanced_accuracy"]
    mean = 100 * value["mean"]
    if len(result["seeds"]) != len(SEEDS):
        return "Pending repetitions"
    return f"{mean:.2f} +/- {100 * value['sample_sd']:.2f}"


@lru_cache(maxsize=None)
def canonical_inputs(repo: Path, dataset: str, representation: str) -> tuple[dict, dict, dict]:
    groups, provenance = manifest(repo, dataset)
    source = PATIENT_DISJOINT if dataset == "shid_v2" else ARCHIVE
    token = "shid" if dataset == "shid_v2" else dataset
    feature_metadata = read(repo / source / "features" / token / representation / "metadata.json")
    if feature_metadata["input_identity_sha256"] != provenance["input_identity_sha256"]:
        raise ValueError("Canonical feature metadata does not match source partitions")
    return groups, provenance, feature_metadata


def read_baselines(repo: Path) -> dict[tuple[str, str, str, str], dict]:
    results = {}
    expected = set(expected_contexts())
    for path in sorted((repo / OUTPUT / "runs").glob("*/*/*/*/complete.json")):
        result = read(path)
        key = tuple(result[name] for name in ("dataset", "representation", "task", "model"))
        if result["representation"] in ALL_REPRESENTATIONS and result["representation"] not in REPRESENTATIONS:
            continue
        if key not in expected or key in results:
            raise ValueError("Unexpected or repeated completed baseline context")
        if result["selection"]["test_used_for_selection"] is not False:
            raise ValueError("Test predictions entered model selection")
        if result["selection"]["seeds"] != result["seeds"]:
            raise ValueError("Selected seeds differ from the completed experiment")
        if any(seed["trained_on"] != "train" or seed["selected_on"] != "val"
               for seed in result["selected_models"]):
            raise ValueError("The final model does not use the intended split protocol")
        for metric in METRICS:
            recomputed = aggregate([seed["metrics"]["test"][metric] for seed in result["selected_models"]])
            for statistic in ("mean", "sample_sd"):
                if not np.isclose(recomputed[statistic], result["test_metrics"][metric][statistic], rtol=0, atol=1e-12):
                    raise ValueError("Completed baseline metric aggregation is inconsistent")
        if result["representation"] != "rgb":
            _, canonical, features = canonical_inputs(repo, result["dataset"], result["representation"])
            recorded = result["input_provenance"]
            if (recorded["input_identity_sha256"] != canonical["input_identity_sha256"]
                    or recorded["feature_artifact_sha256"] != features["artifact_sha256"]):
                raise ValueError("Baseline feature inputs differ from the archived connectome protocol")
        result = apply_selected_repeats(result, path, repo)
        result["completed_source"] = relative(repo, path)
        results[key] = result
    return results


def source_root(dataset: str, task: str) -> Path:
    if dataset != "shid_v2":
        return ARCHIVE
    if task == "multiclass":
        return ARCHIVE / "shid-separate-multiclass-20261101"
    return PATIENT_DISJOINT


def read_connectome(repo: Path, dataset: str, representation: str, task: str) -> dict:
    """Recompute selected-test metrics and verify every matched input identity."""
    root = source_root(dataset, task)
    token = "shid" if dataset == "shid_v2" else dataset
    groups, canonical, canonical_features = canonical_inputs(repo, dataset, representation)
    metadata_path = repo / root / "features" / token / representation / "metadata.json"
    metadata = read(metadata_path)
    if (metadata["input_identity_sha256"] != canonical["input_identity_sha256"]
            or metadata["artifact_sha256"] != canonical_features["artifact_sha256"]):
        raise ValueError("Connectome and conventional models do not use identical cached descriptors")
    if metadata["sample_ids"]["test"] != [row["sample_id"] for row in groups["test"]]:
        raise ValueError("Connectome test feature order differs from the canonical manifest")
    manifest_path = repo / root / "manifests" / f"{token}.json"
    graph_path = repo / root / "graphs/anatomical.npz"
    with np.load(graph_path, allow_pickle=False) as graph:
        gain_count = int(len(graph["src"]))
    expected_ids = np.asarray([row["sample_id"] for row in groups["test"]])
    expected_y = np.asarray([labels_from_counts(row["positive_count"], row["negative_count"])[task]
                             for row in groups["test"]], dtype=np.int64)
    selected = []
    for seed in SEEDS:
        directory = repo / root / "runs" / token / representation / task / "anatomical" / f"seed-{seed}"
        config, best = read(directory / "config.json"), read(directory / "best.json")
        if (config["manifest_sha256"] != sha256(manifest_path)
                or config["feature_metadata_sha256"] != sha256(metadata_path)
                or config["graph_sha256"] != sha256(graph_path)):
            raise ValueError("Archived model source hashes no longer match its inputs")
        if best["criterion"] != "validation_balanced_accuracy":
            raise ValueError("Connectome checkpoint was selected using a different criterion")
        prediction_relative = Path(best["prediction_path"])
        if prediction_relative.is_absolute() or ".." in prediction_relative.parts:
            raise ValueError("Selected prediction path must be relative to its run")
        prediction_path = directory / prediction_relative
        with np.load(prediction_path, allow_pickle=False) as values:
            y = values["y_true"]
            predicted = values["y_pred"]
            probabilities = values["probabilities"]
            if (not np.array_equal(values["sample_id"], expected_ids)
                    or not np.array_equal(y, expected_y)
                    or not np.array_equal(predicted, probabilities.argmax(axis=1))
                    or values["class_names"].tolist() != CLASS_NAMES[task]
                    or int(values["epoch"]) != best["epoch"]):
                raise ValueError("Selected prediction identities or labels do not reproduce the source protocol")
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
                raise ValueError("Invalid connectome class probabilities")
            classes = np.arange(probabilities.shape[1])
            metrics = {
                "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                "accuracy": float(accuracy_score(y, predicted)),
                "macro_f1": float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)),
                "roc_auc": float(np.mean([roc_auc_score(y == label, probabilities[:, label]) for label in classes])),
            }
        for metric, value in metrics.items():
            if not np.isclose(value, best["test_metrics"][metric], rtol=0, atol=1e-10):
                raise ValueError(f"Archived {metric} disagrees with its selected predictions")
        selected.append({"seed": seed, "metrics": metrics, "epoch": best["epoch"],
                         "validation_ba": best["validation_metrics"]["balanced_accuracy"],
                         "best_record": relative(repo, directory / "best.json"),
                         "prediction": relative(repo, prediction_path),
                         "prediction_sha256": sha256(prediction_path)})
    return {
        "dataset": dataset, "representation": representation, "task": task,
        "model": "anatomical_connectome", "seeds": list(SEEDS), "deterministic": False,
        "test_metrics": {metric: aggregate([item["metrics"][metric] for item in selected]) for metric in METRICS},
        "validation_ba": float(np.mean([item["validation_ba"] for item in selected])),
        "head_trainable_parameters": gain_count, "selected_models": selected,
        "source_root": root.as_posix(), "input_identity_sha256": canonical["input_identity_sha256"],
        "feature_artifact_sha256": metadata["artifact_sha256"],
    }


def all_context_rows(repo: Path, baselines: dict) -> list[dict]:
    rows = []
    for dataset, representation, task, model in expected_contexts():
        key = dataset, representation, task, model
        result = baselines.get(key)
        directory = repo / OUTPUT / "runs" / dataset / representation / task / model
        spec = read(directory / "spec.json") if (directory / "spec.json").exists() else None
        row = {"dataset": dataset, "representation": representation, "task": task, "model": model,
               "status": "complete" if result else "pending", "tuning_grid": json.dumps(spec["grid"], sort_keys=True) if spec else "",
               "best_hyperparameters": "", "validation_ba": "", "n_candidates": len(spec["grid"]) if spec else "",
               "seeds": json.dumps(result["seeds"]) if result else "",
               "search_seeds": json.dumps(result["search_seeds"]) if result else "",
               "search_fits": "", "selected_fits": "", "search_seconds": "", "total_seconds": "",
               "selected_repeat_fits": "", "selected_repeat_training_seconds": "",
               "selected_repeat_total_seconds": "", "selected_repeat_record": "", "complete_record": ""}
        if result:
            row.update(best_hyperparameters=json.dumps(result["selection"]["best_params"], sort_keys=True),
                       validation_ba=result["selection"]["mean_validation_balanced_accuracy"],
                       complete_record=result["completed_source"])
            for name in ("search_fits", "selected_fits", "search_seconds", "total_seconds",
                         "selected_repeat_fits", "selected_repeat_training_seconds",
                         "selected_repeat_total_seconds", "selected_repeat_record"):
                row[name] = result[name]
        for metric in METRICS:
            for statistic in ("mean", "sample_sd"):
                row[f"test_{metric}_{statistic}"] = result["test_metrics"][metric][statistic] if result else ""
        rows.append(row)
    return rows


def main_table(baselines: dict, connectomes: dict) -> tuple[list[dict], list[dict], list[str]]:
    wide, long, lines = [], [], []
    lines.extend(["| Dataset | Frozen extractor | Classification head | Three-class BA | PI 5% BA | PI 10% BA | PI 30% BA |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for dataset in DATASETS:
        for representation in ("dinov2", "phikonv2"):
            for model in HEADS:
                wide_row = {"dataset": dataset, "representation": representation, "head": model, "completed_tasks": 0}
                formatted = []
                for task in TASKS:
                    key = dataset, representation, task, model
                    result = connectomes[key] if model == "anatomical_connectome" else baselines.get(key)
                    formatted.append(format_ba(result))
                    row = {"dataset": dataset, "representation": representation, "task": task, "head": model,
                           "status": "complete" if result else "pending"}
                    if result and len(result["seeds"]) == len(SEEDS):
                        wide_row["completed_tasks"] += 1
                    for metric in METRICS:
                        for statistic in ("mean", "sample_sd"):
                            row[f"test_{metric}_{statistic}"] = result["test_metrics"][metric][statistic] if result else ""
                    for statistic in ("mean", "sample_sd"):
                        wide_row[f"{task}_ba_{statistic}"] = result["test_metrics"]["balanced_accuracy"][statistic] if result else ""
                    long.append(row)
                lines.append("| " + " | ".join([DATASET_NAMES[dataset], REPRESENTATION_NAMES[representation], MODEL_LABELS[model], *formatted]) + " |")
                wide.append(wide_row)
    return wide, long, lines


def complete_metric_tables(baselines: dict) -> list[str]:
    lines = []
    for dataset in DATASETS:
        for representation in REPRESENTATIONS:
            lines.extend([f"### {DATASET_NAMES[dataset]} with {REPRESENTATION_NAMES[representation]}", "",
                          "| Model | Three-class BA | PI 5% BA | PI 10% BA | PI 30% BA |",
                          "|---|---:|---:|---:|---:|"])
            for model in MODEL_NAMES:
                if representation == "rgb" and model == "mlp":
                    continue
                values = [format_ba(baselines.get((dataset, representation, task, model))) for task in TASKS]
                lines.append("| " + " | ".join([MODEL_LABELS[model], *values]) + " |")
            lines.append("")
    return lines


def validation_selected_models(baselines: dict) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for representation in REPRESENTATIONS:
            names = [name for name in MODEL_NAMES if not (representation == "rgb" and name == "mlp")]
            for task in TASKS:
                candidates = [baselines.get((dataset, representation, task, name)) for name in names]
                row = {"dataset": dataset, "representation": representation, "task": task,
                       "status": "complete" if all(item is not None for item in candidates) else "pending",
                       "selected_model": "", "validation_ba": "", "test_ba": "", "test_ba_sd": ""}
                if row["status"] == "complete":
                    best = max(candidates, key=lambda item: item["selection"]["mean_validation_balanced_accuracy"])
                    row.update(selected_model=best["model"], validation_ba=best["selection"]["mean_validation_balanced_accuracy"],
                               test_ba=best["test_metrics"]["balanced_accuracy"]["mean"],
                               test_ba_sd=best["test_metrics"]["balanced_accuracy"]["sample_sd"])
                rows.append(row)
    return rows


def capacity_and_compute(repo: Path, baselines: dict, connectomes: dict) -> list[dict]:
    backbone_path = repo / OUTPUT / "provenance/frozen_backbone_complexity.json"
    backbone = read(backbone_path) if backbone_path.exists() else {"backbones": []}
    frozen_counts = {item["representation"]: item["frozen_backbone_parameters"] for item in backbone["backbones"]}
    for item in backbone["backbones"]:
        if item["task_trainable_backbone_parameters"] != 0:
            raise ValueError("The registered backbone is not frozen")
        for dataset in DATASETS:
            _, _, features = canonical_inputs(repo, dataset, item["representation"])
            if features["model"]["file_sha256"]["model.safetensors"] != item["weights_sha256"]:
                raise ValueError("Backbone parameter counts belong to different pretrained weights")
    rows = []
    for key in sorted(baselines):
        result = baselines[key]
        frozen = frozen_counts.get(result["representation"], 0 if result["representation"] == "rgb" else "")
        for seed_result in result["selected_models"]:
            complexity = seed_result["complexity"]
            head = complexity.get("optimized_weight_parameters")
            if result["model"] in ("mlp", "logistic_regression"):
                dimension = result["input_provenance"]["dimension"]
                outputs = 3 if result["task"] == "multiclass" else 1
                expected = ((dimension + 1) * 32 + 33 * outputs if result["model"] == "mlp"
                            else (dimension + 1) * outputs)
                if head != expected:
                    raise ValueError("The fitted head does not match its declared parameter budget")
            # Instance storage and tree structure have separate complexity measures.
            total = frozen + head if isinstance(frozen, int) and head is not None else ""
            rows.append({"dataset": result["dataset"], "representation": result["representation"], "task": result["task"],
                         "model": result["model"], "seed": seed_result["seed"], "frozen_backbone_parameters": frozen,
                         "optimized_head_weight_parameters": head if head is not None else "",
                         "total_weight_parameters": total, "complexity": json.dumps(complexity, sort_keys=True),
                         "model_bytes": seed_result["model_bytes"], "selected_training_seconds": seed_result["fit_seconds"],
                         "test_prediction_seconds": seed_result["prediction_seconds"]["test"],
                         "selected_repetition": bool(result["selected_repeat_fits"] and seed_result["seed"] != SEEDS[0]),
                         "context_search_fits": result["search_fits"], "context_search_seconds": result["search_seconds"],
                         "timing_scope": "Selected classifier fitting and prediction, frozen feature extraction excluded"})
    for key in sorted(connectomes):
        result = connectomes[key]
        frozen = frozen_counts.get(result["representation"], "")
        for seed_result in result["selected_models"]:
            rows.append({"dataset": result["dataset"], "representation": result["representation"], "task": result["task"],
                         "model": result["model"], "seed": seed_result["seed"], "frozen_backbone_parameters": frozen,
                         "optimized_head_weight_parameters": result["head_trainable_parameters"],
                         "total_weight_parameters": frozen + result["head_trainable_parameters"] if isinstance(frozen, int) else "",
                         "complexity": json.dumps({"kind": "anatomical_connectome", "gain_parameters": result["head_trainable_parameters"],
                                                   "fixed_readout_trainable_parameters": 0}, sort_keys=True),
                         "model_bytes": "", "selected_training_seconds": "", "test_prediction_seconds": "",
                         "selected_repetition": False,
                         "context_search_fits": "", "context_search_seconds": "",
                         "timing_scope": "Archived connectome durations have different timing boundaries and are not substituted here"})
    return rows



def main() -> None:
    global REPRESENTATIONS, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--input", type=Path, default=OUTPUT, help="Repository-relative completed campaign directory")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "summary", help="Repository-relative report directory under 6 Ki67/6 Workspace")
    args = parser.parse_args()
    repo = args.repo.resolve()
    OUTPUT = relative_path(args.input)
    summary = repo / relative_path(args.output)
    writable(summary, repo=repo)
    REPRESENTATIONS = campaign_representations(repo)
    baselines = read_baselines(repo)
    from .verification import verify_selected_predictions
    verification = verify_selected_predictions(repo, baselines)
    write_json(summary / "prediction_verification.json", verification)
    connectomes = {(dataset, representation, task, "anatomical_connectome"):
                  read_connectome(repo, dataset, representation, task)
                  for dataset in DATASETS for representation in ("dinov2", "phikonv2") for task in TASKS}
    context_rows = all_context_rows(repo, baselines)
    write_csv(summary / "all_contexts_best_hyperparameters.csv", list(context_rows[0]), context_rows)
    wide, long, table = main_table(baselines, connectomes)
    write_csv(summary / "head_comparison.csv", list(wide[0]), wide)
    write_csv(summary / "head_comparison_long.csv", list(long[0]), long)
    selected = validation_selected_models(baselines)
    write_csv(summary / "validation_selected_classifiers.csv", list(selected[0]), selected)
    capacities = capacity_and_compute(repo, baselines, connectomes)
    write_csv(summary / "model_complexity_and_compute.csv", list(capacities[0]), capacities)
    write_json(summary / "connectome_comparison_audit.json", {
        "schema_version": 1, "path_base": "repository root", "contexts_verified": len(connectomes),
        "selected_test_prediction_files_verified": sum(len(item["selected_models"]) for item in connectomes.values()),
        "same_descriptor_artifacts_and_input_identities": True,
        "results": list(connectomes.values()),
    })
    head_complete = all(row["completed_tasks"] == 4 for row in wide)
    from .release_tables import write_tables
    write_tables(summary, baselines, connectomes, context_rows, capacities)
    full_lines = ["# Ki-67 classifier result tables", "", f"Completed contexts: {len(baselines)}/{len(expected_contexts())}.", "",
                  "Balanced accuracy is reported as a percentage. Every classifier shows mean +/- sample standard deviation across three executions with seeds 1729, 2718, and 3141. Deterministic classifiers reuse the validation-selected configuration for their additional executions without repeating GridSearchCV. Pending cells contain no inferred result.", "", *complete_metric_tables(baselines)]
    write_text(summary / "full_model_tables.md", "\n".join(full_lines))
    three_execution_contexts = sum(item["seeds"] == list(SEEDS) for item in baselines.values())
    status = {"completed_contexts": len(baselines), "planned_contexts": len(expected_contexts()),
              "provisional": three_execution_contexts != len(expected_contexts()),
              "three_execution_contexts": three_execution_contexts, "main_head_comparison_complete": head_complete,
              "head_comparison_rows": len(wide), "connectome_contexts_verified": len(connectomes),
              "test_used_for_model_selection": False,
              "classifier_selection": "Highest mean validation balanced accuracy after all planned classifiers finish",
              "classifier_tie_rule": "First classifier in the declared MODEL_NAMES order"}
    status["included_representations"] = list(REPRESENTATIONS)
    status["excluded_representations"] = [item for item in ALL_REPRESENTATIONS if item not in REPRESENTATIONS]
    write_json(summary / "report_status.json", status)
    result_rows = []
    for key in sorted(baselines):
        result = baselines[key]
        row = {name: result[name] for name in ("dataset", "representation", "task", "model")}
        row.update(seeds=json.dumps(result["seeds"]), search_seeds=json.dumps(result["search_seeds"]),
                   best_params=json.dumps(result["selection"]["best_params"], sort_keys=True),
                   validation_ba=result["selection"]["mean_validation_balanced_accuracy"])
        for metric, column in (("balanced_accuracy", "ba"), ("accuracy", "accuracy"), ("macro_f1", "macro_f1"), ("roc_auc", "roc_auc")):
            row[f"test_{column}"] = result["test_metrics"][metric]["mean"]
            row[f"test_{column}_sd"] = result["test_metrics"][metric]["sample_sd"]
        row.update({name: result[name] for name in ("search_fits", "selected_fits", "search_seconds", "total_seconds",
                   "selected_repeat_fits", "selected_repeat_training_seconds", "selected_repeat_total_seconds",
                   "selected_repeat_record")})
        result_rows.append(row)
    if result_rows:
        write_csv(summary / "results.csv", list(result_rows[0]), result_rows)
    write_json(summary / "counts.json", {
        "completed_comparisons": len(baselines), "planned_comparisons": len(expected_contexts()),
        "search_fits": sum(item["search_fits"] for item in baselines.values()),
        "selected_models": sum(item["selected_fits"] for item in baselines.values()),
        "three_execution_comparisons": three_execution_contexts,
        "additional_selected_repetitions": sum(item["selected_repeat_fits"] for item in baselines.values()),
        "additional_selected_training_seconds": sum(item["selected_repeat_training_seconds"] for item in baselines.values()),
        "additional_selected_total_seconds": sum(item["selected_repeat_total_seconds"] for item in baselines.values()),
        "accumulated_job_seconds": sum(item["total_seconds"] for item in baselines.values()),
        "included_representations": list(REPRESENTATIONS),
    })
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
