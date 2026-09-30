"""Read predictions and aggregate confusion matrices for manuscript figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .data import CLASS_NAMES
from .evaluation import CLASSIFICATION_TASKS, classification_metrics


def read_prediction(path: Path, task: str, epoch: int, attempt: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if int(arrays["epoch"]) != epoch or int(arrays["attempt"]) != attempt:
        raise ValueError(f"Prediction epoch/attempt mismatch: {path}")
    if len(np.unique(arrays["sample_id"])) != len(arrays["sample_id"]):
        raise ValueError(f"Repeated image IDs inside one prediction file: {path}")
    if task in CLASSIFICATION_TASKS:
        if tuple(arrays["class_names"]) != tuple(CLASS_NAMES[task]):
            raise ValueError(f"Unexpected class order for {task}: {path}")
        if not np.array_equal(arrays["probabilities"].argmax(axis=1), arrays["y_pred"]):
            raise ValueError(f"Saved class prediction disagrees with probabilities: {path}")
    return arrays


def mean_confusion(predictions: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    """Average seed-specific row percentages without inflating cohort size."""
    first = predictions[0]
    matrices = []
    for prediction in predictions:
        for key in ("sample_id", "y_true", "class_names"):
            if not np.array_equal(first[key], prediction[key]):
                raise ValueError(f"Seed evaluations must use identical {key}")
        report = classification_metrics(prediction["y_true"], prediction["probabilities"])
        matrix = np.asarray(report["confusion_matrix"], dtype=float)
        support = matrix.sum(axis=1)
        if (support == 0).any():
            raise ValueError("Every displayed true class must have support")
        matrices.append(100 * matrix / support[:, None])
    return {
        "n_unique_images": len(first["sample_id"]),
        "n_seed_evaluations": len(predictions),
        "class_names": first["class_names"].tolist(),
        "class_support": np.bincount(first["y_true"], minlength=len(first["class_names"])).tolist(),
        "percentages": np.mean(matrices, axis=0).tolist(),
        "per_seed_percentages": np.asarray(matrices).tolist(),
        "convention": "rows=true, columns=predicted; mean row percentages across fixed-partition seeds",
    }
