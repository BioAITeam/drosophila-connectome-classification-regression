"""Prediction-based evaluation and validation-based checkpoint selection.

Returned values are JSON-compatible. Undefined statistics are ``None``;
valid negative R-squared values are retained.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


from .tasks import CLASSIFICATION_TASKS, REGRESSION_TASKS, DISPLAY_CLASS_NAMES as CLASS_NAMES
from .common import destination as safe_destination, writable_location


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _classification_arrays(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    prob = np.asarray(probabilities, dtype=np.float64)
    if truth.ndim != 1 or prob.ndim != 2 or not len(truth) or prob.shape[0] != len(truth):
        raise ValueError("Expected nonempty truth [N] and probabilities [N, classes].")
    if prob.shape[1] < 2 or not np.isfinite(prob).all() or not np.isfinite(truth).all():
        raise ValueError("Labels and probabilities must be finite, with at least two classes.")
    if (truth != np.floor(truth)).any() or (truth < 0).any() or (truth >= prob.shape[1]).any():
        raise ValueError("Class IDs must be integers indexing the probability columns.")
    if (prob < 0).any() or (prob > 1).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-6, rtol=1e-6):
        raise ValueError("Probabilities must be in [0, 1] and each row must sum to one.")
    return truth.astype(np.int64), prob


def classification_metrics(y_true: Any, probabilities: Any) -> dict[str, Any]:
    """Compute classification metrics from probabilities, never class IDs alone.

    Balanced accuracy averages recall over truth classes present. Macro F1,
    precision and recall use the full output vocabulary and zero contributions
    for undefined classes. Per-class undefined ratios remain explicit ``None``.
    Macro ROC-AUC/AP average only classes with both positive and negative truth
    observations; the number of included classes is reported.
    """
    truth, prob = _classification_arrays(y_true, probabilities)
    pred = prob.argmax(axis=1)
    count = prob.shape[1]
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    rows: list[dict[str, Any]] = []
    for c in range(count):
        tp = int(matrix[c, c])
        fp, fn = int(predicted[c] - tp), int(support[c] - tp)
        tn = len(truth) - tp - fp - fn
        target = truth == c
        curve_defined = bool(target.any() and (~target).any())
        rows.append({
            "class_id": c, "support": int(support[c]), "predicted": int(predicted[c]),
            "precision": tp / int(predicted[c]) if predicted[c] else None,
            "recall": tp / int(support[c]) if support[c] else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "roc_auc": float(roc_auc_score(target, prob[:, c])) if curve_defined else None,
            "average_precision": float(average_precision_score(target, prob[:, c])) if curve_defined else None,
        })
    def macro(name: str, defined_only: bool = False) -> float | None:
        values = [r[name] for r in rows if r[name] is not None] if defined_only else [r[name] or 0.0 for r in rows]
        return float(np.mean(values)) if values else None
    result = {
        "n_samples": len(truth), "n_classes": count,
        "loss": float(-np.log(np.clip(prob[np.arange(len(truth)), truth], np.finfo(np.float64).tiny, 1)).mean()),
        "accuracy": float((truth == pred).mean()),
        "balanced_accuracy": macro("recall", defined_only=True),
        "macro_f1": macro("f1"), "macro_precision": macro("precision"), "macro_recall": macro("recall"),
        "roc_auc": macro("roc_auc", defined_only=True),
        "average_precision": macro("average_precision", defined_only=True),
        "n_curve_classes": sum(r["roc_auc"] is not None for r in rows),
        "per_class": rows, "confusion_matrix": matrix.tolist(),
        "confusion_convention": "rows=true, columns=predicted",
        "balanced_accuracy_convention": "mean recall over classes present in y_true",
        "curve_average_convention": "macro over classes with positive and negative truth observations",
    }
    if count == 2:
        result["positive_class_roc_auc"] = rows[1]["roc_auc"]
        result["positive_class_average_precision"] = rows[1]["average_precision"]
    return result


def metrics_by_dataset(y_true: Any, predictions: Any, task: str,
                       dataset_ids: Any) -> dict[str, dict[str, Any]]:
    """Disaggregate one model's predictions without selecting another checkpoint.

    Dataset origin is reporting metadata only. The pooled validation criterion
    remains outside this helper and is never replaced by a subgroup optimum.
    """
    truth = np.asarray(y_true)
    values = np.asarray(predictions)
    origins = np.asarray(dataset_ids)
    if origins.ndim != 1 or len(origins) != len(truth) or len(values) != len(truth):
        raise ValueError("Dataset IDs must align with every prediction.")
    if origins.dtype.kind not in {"U", "S"} or any(not str(x).strip() for x in origins):
        raise ValueError("Dataset IDs must be nonempty strings.")
    result = {}
    for origin in sorted(np.unique(origins)):
        mask = origins == origin
        result[str(origin)] = (classification_metrics(truth[mask], values[mask])
                               if task in CLASSIFICATION_TASKS
                               else regression_report(truth[mask], values[mask], task))
    return result


def _regression_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    truth, pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != pred.shape or not len(truth):
        raise ValueError("Expected matching, nonempty one-dimensional regression arrays.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Regression values must be finite.")
    return truth, pred


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """MAE, MSE, RMSE and R² in the supplied units; no target clipping."""
    truth, pred = _regression_arrays(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = pred - truth
        mse = np.square(errors).mean()
        denominator = np.square(truth - truth.mean()).sum()
        r2 = None if len(truth) < 2 or denominator == 0 else _finite_or_none(1 - np.square(errors).sum() / denominator)
        return {
            "n_samples": len(truth), "mae": _finite_or_none(np.abs(errors).mean()),
            "mse": _finite_or_none(mse), "rmse": _finite_or_none(np.sqrt(mse)),
            "r2": r2, "r2_defined": r2 is not None,
        }


def regression_report(y_true: Any, y_pred: Any, task: str) -> dict[str, Any]:
    """Raw metrics at top level, domain-limited metrics under ``clipped``."""
    if task not in REGRESSION_TASKS:
        raise ValueError(f"Unknown regression task: {task}")
    truth, pred = _regression_arrays(y_true, y_pred)
    clipped = np.clip(pred, 0, 100) if task == "pi" else np.maximum(pred, 0)
    result = regression_metrics(truth, pred)
    result.update({
        "loss": result["mse"], "clipped": regression_metrics(truth, clipped),
        "correction_fraction": float(np.mean(pred != clipped)),
        "correction_count": int(np.count_nonzero(pred != clipped)),
        "prediction_units": "percentage points" if task == "pi" else "cells",
        "selection_prediction": "raw",
    })
    return result


def select_score(metrics: dict[str, Any], task: str) -> float:
    """Convert validation metrics to a score for which larger values indicate improvement."""
    if task in CLASSIFICATION_TASKS:
        key, sign = "balanced_accuracy", 1
    elif task in REGRESSION_TASKS:
        key, sign = "mae", -1
    else:
        raise ValueError(f"Unknown task: {task}")
    value = metrics.get(key)
    if value is None or not np.isfinite(value):
        raise ValueError(f"Checkpoint selection requires a finite {key}.")
    return sign * float(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = safe_destination(path)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    # Unique temporary names prevent collisions with concurrent monitor writes.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize(output: Path | str, destination: Path | str | None = None) -> list[Path]:
    """Rebuild selected-checkpoint test tables from saved predictions.

    In-progress runs are included with an explicit status, never presented as
    completed results. Standard deviation is sample SD across distinct seeds;
    it describes optimization variability on fixed partitions, not a CI.
    """
    output = Path(output)
    report = writable_location(output if destination is None else destination)
    records: list[dict[str, Any]] = []
    for best_path in sorted((output / "runs").glob("**/best.json")):
        run = best_path.parent
        best = _read_json(best_path)
        config = _read_json(run / "config.json") if (run / "config.json").exists() else {}
        task = best.get("task", config.get("task"))
        history = _read_json(run / "history.json") if (run / "history.json").exists() else []
        selected = next((r for r in reversed(history) if r["epoch"] == best["epoch"] and r.get("attempt", 0) == best.get("attempt", 0)), {})
        status = _read_json(run / "status.json") if (run / "status.json").exists() else {}
        state = status.get("state", status.get("status", "in_progress"))
        if (run / "complete.json").exists():
            state = "complete"
        with np.load(run / best["prediction_path"], allow_pickle=False) as predictions:
            metrics = classification_metrics(predictions["y_true"], predictions["probabilities"]) if task in CLASSIFICATION_TASKS else regression_report(predictions["y_true"], predictions["y_pred"], task)
        row = {key: best.get(key, config.get(key)) for key in ("dataset", "representation", "task", "graph", "seed")}
        row.update({"run_path": str(run.relative_to(output)), "status": state,
                    "selected_epoch": best["epoch"], "selected_attempt": best.get("attempt", 0),
                    "selection_metric": "balanced_accuracy" if task in CLASSIFICATION_TASKS else "mae",
                    "selection_split": "validation"})
        for split, values in (("validation", best.get("validation_metrics", selected.get("val", {}))), ("test", metrics)):
            for name, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) or value is None:
                    row[f"{split}_{name}"] = value
                elif name == "clipped":
                    row.update({f"{split}_clipped_{k}": v for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool) or v is None})
        records.append(row)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    identity = ("dataset", "representation", "task", "graph", "status")
    for row in records:
        groups.setdefault(tuple(row[k] for k in identity), []).append(row)
    grouped: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        seeds = [r["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in summary group {key}; refusing to pool repeated attempts.")
        row = dict(zip(identity, key))
        row.update({"n_seeds": len(seeds), "seeds": ",".join(str(s) for s in sorted(seeds)),
                    "uncertainty": "sample SD across optimization seeds on fixed data splits"})
        names = sorted({name for entry in rows for name in entry if name.startswith(("test_", "validation_"))})
        for name in names:
            values = np.asarray([entry[name] for entry in rows if entry.get(name) is not None], dtype=float)
            if len(values):
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
                row[f"{name}_n"] = len(values)
        grouped.append(row)
    paths = [report / "tables" / "per_run.csv", report / "tables" / "grouped.csv"]
    _write_csv(paths[0], records)
    _write_csv(paths[1], grouped)
    if any(row.get("dataset") == "joint" for row in records):
        from .joint_evaluation import summarize_joint
        paths.extend(summarize_joint(output, destination=report))
    return paths
