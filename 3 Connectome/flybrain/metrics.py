"""Evaluation without learned readouts, including paired sample uncertainty."""
from __future__ import annotations

from typing import Any

import numpy as np


def _classification_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if pred.ndim == 2:
        if not pred.shape[1] or not np.isfinite(pred).all():
            raise ValueError("Score matrices must have finite values and at least one class.")
        pred = pred.argmax(axis=1)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred) or not len(truth):
        raise ValueError("Expected matching, nonempty class-label vectors or a prediction score matrix.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Class labels must be finite.")
    if (truth < 0).any() or (pred < 0).any() or (truth != np.floor(truth)).any() or (pred != np.floor(pred)).any():
        raise ValueError("Class IDs must be nonnegative integers.")
    return truth.astype(np.int64), pred.astype(np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> dict[str, Any]:
    truth, pred = _classification_arrays(y_true, y_pred)
    required = int(max(truth.max(), pred.max())) + 1
    n_classes = required if n_classes is None else n_classes
    if n_classes < required:
        raise ValueError("n_classes cannot exclude an observed class ID.")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (truth, pred), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = np.diag(matrix).astype(np.float64)
    recall = np.divide(diagonal, support, out=np.zeros(n_classes), where=support > 0)
    f1 = np.divide(2 * diagonal, support + predicted, out=np.zeros(n_classes), where=(support + predicted) > 0)
    return {"accuracy": float((truth == pred).mean()), "balanced_accuracy": float(recall[support > 0].mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "per_class_recall": recall.tolist(), "support": support.tolist(), "n_samples": len(truth),
            "confusion_convention": "rows=true, columns=predicted",
            "balanced_accuracy_convention": "mean recall over classes present in y_true"}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or not len(truth):
        raise ValueError("Expected matching, nonempty one-dimensional regression vectors.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Regression values must be finite.")
    residual = truth - pred
    denominator = float(np.square(truth - truth.mean()).sum())
    r2 = None if len(truth) < 2 or denominator == 0 else 1.0 - float(np.square(residual).sum()) / denominator
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.square(residual).mean())),
            "r2": r2, "n_samples": len(truth), "r2_defined": r2 is not None}


def paired_bootstrap_delta(
    y_true: np.ndarray, pred_before: np.ndarray, pred_after: np.ndarray, *,
    task: str = "classification", metric: str | None = None, groups: np.ndarray | None = None,
    n_bootstrap: int = 2000, seed: int = 0, confidence: float = 0.95
) -> dict[str, Any]:
    """Percentile CI for after-before, resampling identical sample/group indices.

    This is sampling uncertainty conditional on one trained model and split,
    not an estimate over new training sets, subjects or experiment seeds.
    Pass duplicate feature groups to bootstrap whole groups together.
    """
    if task not in {"classification", "regression"}:
        raise ValueError("Unknown task.")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence must lie strictly between zero and one.")
    if task == "classification":
        truth, before = _classification_arrays(y_true, pred_before)
        _, after = _classification_arrays(y_true, pred_after)
        n_classes = int(max(truth.max(), before.max(), after.max())) + 1
        metric = metric or "balanced_accuracy"
        if metric not in {"accuracy", "balanced_accuracy", "macro_f1"}:
            raise ValueError("Unsupported classification metric.")
        def evaluate(ix: np.ndarray, pred: np.ndarray) -> float:
            return classification_metrics(truth[ix], pred[ix], n_classes)[metric]
    else:
        truth, before, after = map(lambda a: np.asarray(a, dtype=float), (y_true, pred_before, pred_after))
        regression_metrics(truth, before)
        regression_metrics(truth, after)
        metric = metric or "rmse"
        if metric not in {"mae", "rmse", "r2"}:
            raise ValueError("Unsupported regression metric.")
        def evaluate(ix: np.ndarray, pred: np.ndarray) -> float | None:
            return regression_metrics(truth[ix], pred[ix])[metric]
    if groups is None:
        groups = np.arange(len(truth))
    groups = np.asarray(groups)
    if groups.shape != truth.shape:
        raise ValueError("groups must match y_true.")
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        ix = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        initial, final = evaluate(ix, before), evaluate(ix, after)
        if initial is not None and final is not None:
            deltas.append(float(final - initial))
    ix = np.arange(len(truth))
    initial, final = evaluate(ix, before), evaluate(ix, after)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(deltas, [alpha, 1 - alpha]).tolist() if deltas else (None, None)
    return {"metric": metric, "before": initial, "after": final,
            "delta": None if initial is None or final is None else final - initial,
            "ci_low": low, "ci_high": high, "confidence": confidence,
            "higher_is_better": metric not in {"mae", "rmse"}, "delta_convention": "after - before",
            "n_bootstrap": n_bootstrap, "n_valid_bootstrap": len(deltas), "n_resampling_groups": len(members),
            "seed": seed, "method": "paired group percentile bootstrap",
            "limitation": "Conditional on this split/model; repeated seeds are not independent biological samples."}


def summarize_seed_metrics(values: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite seed metrics.")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()), "n_runs": len(values),
            "limitation": "Variation across runs; seeds sharing observations are not independent sample evidence."}
