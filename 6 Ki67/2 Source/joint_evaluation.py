"""Joint-model reports with one pooled-validation selection for every test scope."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import read_json, write_json, writable_location
from .evaluation import (CLASSIFICATION_TASKS, _write_csv, classification_metrics,
                         metrics_by_dataset, regression_report, select_score)


def _scalars(metrics):
    return {k: v for k, v in metrics.items()
            if v is None or isinstance(v, (int, float)) and not isinstance(v, bool)}


def _prediction_metrics(path, task):
    with np.load(path, allow_pickle=False) as saved:
        truth = saved["y_true"]
        values = saved["probabilities" if task in CLASSIFICATION_TASKS else "y_pred"]
        pooled = (classification_metrics(truth, values) if task in CLASSIFICATION_TASKS
                  else regression_report(truth, values, task))
        return {"pooled": pooled, **metrics_by_dataset(truth, values, task, saved["source_dataset"])}


def summarize_joint(output, destination=None):
    """Rebuild per-domain results and choose configurations using validation only.

    Individual configurations retain all three seed results. A single-checkpoint
    recommendation per task/graph is additionally identified by highest pooled
    validation score among completed runs; identical weights serve all domains.
    This recommendation is explicit and never based on subgroup or test scores.
    """
    output = Path(output)
    report = writable_location(output if destination is None else destination)
    rows, candidates = [], []
    for best_path in sorted((output / "runs" / "joint").glob("**/best.json")):
        run = best_path.parent
        spec, best = read_json(run / "config.json"), read_json(best_path)
        if not spec.get("joint_training"):
            continue
        task = spec["task"]
        selected = _prediction_metrics(run / best["prediction_path"], task)
        initial_files = sorted((run / "predictions").glob("attempt-*/epoch-0000.npz"))
        if not initial_files:
            raise ValueError(f"Missing original epoch-zero predictions: {run}")
        initial = _prediction_metrics(initial_files[0], task)
        val = best["validation_metrics"]
        state = "complete" if (run / "complete.json").is_file() else "in_progress"
        identity = {k: spec[k] for k in ("representation", "task", "graph", "seed")}
        base = {**identity, "architecture_family": "BRAP-Net", "status": state,
                "run_path": str(run.relative_to(output)), "checkpoint": str((run / "best.pt").relative_to(output)),
                "selected_epoch": best["epoch"], "selected_attempt": best.get("attempt", 0),
                "selection_scope": "pooled_validation", "validation_score": select_score(val, task),
                "selection_metric": "balanced_accuracy" if task in CLASSIFICATION_TASKS else "mae"}
        for scope, metrics in selected.items():
            row = {**base, "evaluation_dataset": scope}
            for prefix, values in (("initial_test", initial[scope]), ("test", metrics),
                                    ("validation", val if scope == "pooled" else val["by_dataset"][scope])):
                row.update({f"{prefix}_{k}": v for k, v in _scalars(values).items()})
                if "clipped" in values:
                    row.update({f"{prefix}_clipped_{k}": v for k, v in _scalars(values["clipped"]).items()})
            metric = base["selection_metric"]
            row["test_improvement"] = ((metrics[metric] - initial[scope][metric])
                                       * (100 if task in CLASSIFICATION_TASKS else -1))
            row["improvement_units"] = "percentage points" if task in CLASSIFICATION_TASKS else metrics["prediction_units"]
            rows.append(row)
        if state == "complete":
            candidates.append({**base, "test": selected, "pooled_validation": val,
                               "prediction_path": str((run / best["prediction_path"]).relative_to(output))})

    identities = ("representation", "task", "graph", "status", "evaluation_dataset")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in identities), []).append(row)
    grouped = []
    for key, entries in sorted(groups.items()):
        seeds = [r["seed"] for r in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Repeated seeds cannot be treated as additional observations.")
        item = {**dict(zip(identities, key)), "n_seeds": len(seeds), "seeds": ",".join(map(str, sorted(seeds))),
                "uncertainty": "sample SD across optimization seeds on the same original split"}
        for name in sorted({n for row in entries for n in row if n.startswith(("test_", "initial_test_", "validation_"))}):
            values = [row[name] for row in entries if isinstance(row.get(name), (int, float))]
            if values:
                item.update({f"{name}_mean": float(np.mean(values)),
                             f"{name}_std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
        grouped.append(item)

    # Sorting resolves exact validation ties deterministically, without test input.
    winners = {}
    for row in sorted(candidates, key=lambda r: (-r["validation_score"], r["representation"], r["seed"])):
        winners.setdefault((row["task"], row["graph"]), row)
    paths = [report / "tables" / "joint_per_dataset.csv", report / "tables" / "joint_grouped.csv",
             report / "tables" / "joint_selected_models.json"]
    _write_csv(paths[0], rows)
    _write_csv(paths[1], grouped)
    jobs_path = output / "manifests" / "jobs.json"
    expected = sum(j["kind"] == "train" for j in read_json(jobs_path)) if jobs_path.exists() else None
    write_json(paths[2], {
        "selection_rule": "highest pooled validation score among completed runs per task and graph; test never selects",
        "provisional_until_campaign_complete": expected is None or len(candidates) < expected,
        "expected_completed_candidates": expected,
        "scope": "same selected checkpoint for pooled, bcdata and shid test evaluations",
        "candidate_count": len(candidates), "models": list(winners.values())})
    return paths
