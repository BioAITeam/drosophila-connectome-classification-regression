"""Read separately recorded selected-model repetitions without changing searches."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .data import OUTPUT, sha256


SEEDS = (1729, 2718, 3141)
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_path(repo: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Selected repetition paths must be relative to the repository root")
    resolved = (repo / path).resolve()
    if not resolved.is_relative_to(repo.resolve()):
        raise ValueError("Selected repetition path resolves outside the repository")
    return resolved


def apply_selected_repeats(result: dict, original_complete_path: Path, repo: Path) -> dict:
    """Merge a verified completed extension, retaining the original search record.

    Missing or unfinished extensions leave the initial selected fit available.
    A completed extension with inconsistent provenance is rejected rather than
    silently omitted. The caller's original dictionary is never modified.
    """
    merged = copy.deepcopy(result)
    merged["search_seeds"] = list(result["selection"]["seeds"])
    merged["selected_repeat_fits"] = 0
    merged["selected_repeat_training_seconds"] = 0.0
    merged["selected_repeat_total_seconds"] = 0.0
    merged["selected_repeat_record"] = ""
    context = tuple(result[name] for name in ("dataset", "representation", "task", "model"))
    extension_path = original_complete_path.parents[5] / "selected-repeats" / Path(*context) / "complete.json"
    if not extension_path.exists():
        return merged
    extension = _read(extension_path)
    if extension.get("status") != "completed":
        return merged
    if not result["deterministic"] or result["seeds"] != [SEEDS[0]]:
        raise ValueError("Selected repetition extension belongs to an ineligible original context")
    if (extension["original_complete_sha256"] != sha256(original_complete_path)
            or extension["original_selection_sha256"] != sha256(original_complete_path.parent / "selection.json")):
        raise ValueError("Selected repetition provenance differs from its original search or completion record")
    if extension["seeds"] != list(SEEDS) or extension["added_selected_fits"] != 2:
        raise ValueError("Selected repetitions must add seeds 2718 and 3141 to the existing seed 1729")
    expected_paths = [original_complete_path.parent / f"seed-{SEEDS[0]}" / "result.json"]
    expected_paths += [extension_path.parent / f"seed-{seed}" / "result.json" for seed in SEEDS[1:]]
    recorded_paths = [_relative_path(repo, value) for value in extension["seed_results"]]
    if recorded_paths != [path.resolve() for path in expected_paths]:
        raise ValueError("Selected repetition results differ from the declared context and seed paths")
    selected = [_read(path) for path in recorded_paths]
    if selected[0] != result["selected_models"][0]:
        raise ValueError("Original selected-model result changed after its completion record")
    for seed, item in zip(SEEDS, selected, strict=True):
        if (item["seed"] != seed or item["best_params"] != result["selection"]["best_params"]
                or item["trained_on"] != "train" or item["selected_on"] != "val"):
            raise ValueError("Selected repetition changed the configuration, seed, or split protocol")
        model_path = _relative_path(repo, item["model"])
        if model_path != (recorded_paths[SEEDS.index(seed)].parent / "best.joblib"):
            raise ValueError("Selected repetition points to a different model directory")
        if sha256(model_path) != item["model_sha256"]:
            raise ValueError("Selected repetition checkpoint hash does not match its result")
    test_metrics = {}
    for metric in METRICS:
        values = np.asarray([item["metrics"]["test"][metric] for item in selected], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Selected repetition contains a non-finite metric")
        computed = {"mean": float(values.mean()),
                    "sample_sd": 0.0 if np.ptp(values) == 0 else float(values.std(ddof=1))}
        for statistic, value in computed.items():
            if not np.isclose(value, extension["aggregate"][metric][statistic], rtol=0, atol=1e-12):
                raise ValueError("Selected repetition aggregate does not reproduce its recorded seed results")
        test_metrics[metric] = computed
    for name in ("added_training_seconds", "added_total_seconds"):
        if not math.isfinite(extension[name]) or extension[name] < 0:
            raise ValueError("Selected repetition timing must be finite and nonnegative")
    training_seconds = sum(item["fit_seconds"] for item in selected[1:])
    if not np.isclose(training_seconds, extension["added_training_seconds"], rtol=0, atol=1e-9):
        raise ValueError("Selected repetition training duration does not match its seed records")
    merged.update(
        seeds=list(SEEDS), selected_models=selected, test_metrics=test_metrics,
        selected_fits=result["selected_fits"] + extension["added_selected_fits"],
        total_seconds=result["total_seconds"] + extension["added_total_seconds"],
        selected_repeat_fits=extension["added_selected_fits"],
        selected_repeat_training_seconds=extension["added_training_seconds"],
        selected_repeat_total_seconds=extension["added_total_seconds"],
        selected_repeat_record=extension_path.resolve().relative_to(repo.resolve()).as_posix(),
    )
    return merged
