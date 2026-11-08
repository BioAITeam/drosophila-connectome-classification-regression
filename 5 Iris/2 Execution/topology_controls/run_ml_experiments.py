"""Run one Iris ML job, selecting exclusively on its frozen validation fold.

Usage: python run_ml_experiments.py --job-id JOB_ID [--smoke]
Smoke mode uses the complete partition, one grid candidate, and two MLP epochs;
its artifacts live under qa/, independently of all production outputs in runs/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import io
import os
from pathlib import Path
import re
import sys
import time
import traceback
import warnings

# common sets the campaign's thread limits before numerical libraries import.
from common import HERE, REPO, SOURCE, load_json, load_partition, relative, save_json, sha256

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from threadpoolctl import threadpool_limits

from ml_models import available_models, build_spec, model_complexity, model_scores


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value):
    """Convert sklearn CV results, including masked entries, to strict JSON."""
    if np.ma.isMaskedArray(value):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    save_json(path, clean(value))


def portable_text(value):
    return str(value).replace(str(HERE), "<campaign>").replace(str(REPO), "<repo>").replace(str(Path.home()), "<home>")


def warning_summary(captured, phase):
    counts = Counter((item.category.__name__, portable_text(item.message)) for item in captured)
    return [{"phase": phase, "category": category, "message": message, "count": count}
            for (category, message), count in counts.items()]


class SearchLog(io.TextIOBase):
    """Persist sklearn's own progress while reporting completed CV candidates."""

    def __init__(self, stream, log, on_candidate):
        self.stream, self.log, self.on_candidate = stream, log, on_candidate
        self.pending = ""

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            # verbose=3 emits exactly one END line per one-fold candidate.
            if re.match(r"^\[CV \d+/\d+\] END ", line):
                self.on_candidate()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def runtime():
    return {
        "python": sys.version.split()[0],
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl")},
        "thread_environment": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "grid_search_n_jobs": 1,
        "estimator_n_jobs": 1,
    }


def load_job(job_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", job_id):
        raise ValueError("Job ID must be a simple filename-safe identifier")
    matches = [job for job in load_json(HERE / "jobs.json")["jobs"] if job["id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest entry for {job_id!r}")
    job = dict(matches[0])
    if job["kind"] != "ml" or job["model"] not in available_models():
        raise ValueError("Requested job is not a supported ML job")
    if job["split_seed"] not in (1729, 2718) or job["optimization_seed"] not in (0, 1) or job["fold"] not in range(5):
        raise ValueError("Job does not belong to the frozen campaign seed/fold plan")
    return job


def validate_partition(data):
    """Check row identity and train-fitted preprocessing without fitting a scaler."""
    indices = []
    for split in ("train", "val", "test"):
        x, y, index = (np.asarray(data[f"{key}_{split}"]) for key in ("x", "y", "index"))
        if x.shape != (len(y), 4) or index.shape != y.shape or y.ndim != 1:
            raise ValueError(f"Malformed {split} partition")
        if not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
            raise ValueError(f"Expected finite, clipped [0,1] features for {split}")
        if len(np.unique(index)) != len(index) or not np.array_equal(np.unique(y), [0, 1, 2]):
            raise ValueError(f"Invalid row identities or missing classes in {split}")
        if not np.array_equal(y, np.asarray(data["raw_y"])[index]):
            raise ValueError(f"Label identity mismatch in {split}")
        indices.extend(index.tolist())
    if sorted(indices) != list(range(150)):
        raise ValueError("The three disjoint partitions must cover all 150 Iris rows")
    if np.asarray(data["raw_x"]).shape != (150, 4) or np.asarray(data["groups"]).shape != (150,):
        raise ValueError("Malformed complete Iris dataset or fold assignments")
    groups = [set(np.asarray(data["groups"])[data[f"index_{split}"]].tolist())
              for split in ("train", "val", "test")]
    if any(left & right for index, left in enumerate(groups) for right in groups[index + 1:]):
        raise ValueError("Duplicate-flower groups cross partition boundaries")


def search_inputs(data):
    """Only training and validation features are reachable by GridSearchCV."""
    x = np.concatenate((data["x_train"], data["x_val"]), axis=0)
    y = np.concatenate((data["y_train"], data["y_val"]), axis=0)
    n_train, n_val = len(data["y_train"]), len(data["y_val"])
    cv = PredefinedSplit(np.r_[np.full(n_train, -1), np.zeros(n_val)].astype(int))
    train, val = next(cv.split())
    if not np.array_equal(train, np.arange(n_train)) or not np.array_equal(val, np.arange(n_train, n_train + n_val)):
        raise AssertionError("PredefinedSplit differs from the frozen train/validation partition")
    return x, y, cv


def first_validation_max(cv_results):
    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Every candidate must have a finite validation score")
    maximum = float(np.max(scores))
    tied = np.flatnonzero(scores == maximum).tolist()
    best = tied[0]
    return {
        "selection_metric": "validation_balanced_accuracy",
        "tie_rule": "first exact maximum in sklearn ParameterGrid order",
        "best_index": best,
        "best_validation_score": maximum,
        "best_params": cv_results["params"][best],
        "tied_best_indices": tied,
        "tied_best_count": len(tied),
        "refit_training_partition_only": True,
        "test_used_for_selection": False,
    }


def atomic_npz(path, values):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def run_job(job, smoke=False):
    if isinstance(job, str):
        job = load_job(job)
    folder = HERE / ("qa" if smoke else "runs") / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".job.lock").open("a") as lock:
        # A second scheduler must not duplicate or race a running job.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(job, folder, smoke)


def _run_locked(job, folder, smoke):
    started = time.perf_counter()
    progress = {"job": job, "id": job["id"], "smoke": bool(smoke), "phase": "loading",
                "started_utc": utc_now(), "grid_candidates_completed": 0,
                "grid_candidates_total": None, "selected_training_fits_completed": 0,
                "test_evaluations_completed": 0}
    captured_warnings = []

    def update(phase=None, **fields):
        if phase is not None:
            progress["phase"] = phase
        progress.update(fields)
        progress["updated_utc"] = utc_now()
        progress["elapsed_seconds"] = time.perf_counter() - started
        write(folder / "progress.json", progress)

    def candidate_finished():
        update(grid_candidates_completed=progress["grid_candidates_completed"] + 1)

    try:
        partition_path = HERE / "data" / f"split-{job['split_seed']}-fold-{job['fold']}.npz"
        partition_sha = sha256(partition_path)
        if (folder / "complete.json").exists():
            previous = load_json(folder / "complete.json")
            if previous.get("job") != job or previous.get("smoke") != bool(smoke):
                raise ValueError("Existing completion metadata belongs to a different job")
            if previous["partition_sha256"] != partition_sha or previous["checkpoint_sha256"] != sha256(folder / "best.joblib"):
                raise ValueError("Completed job artifacts do not match recorded hashes")
            if previous["result_sha256"] != sha256(folder / "result.json") or previous["predictions_sha256"] != sha256(folder / "predictions.npz"):
                raise ValueError("Completed results or predictions do not match recorded hashes")
            return load_json(folder / "result.json")

        update()
        data = load_partition(job["split_seed"], job["fold"])
        validate_partition(data)
        if importlib.metadata.version("scikit-learn") != "1.7.2":
            raise RuntimeError("Use scikit-learn 1.7.2, matching the frozen model specifications")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                spec = build_spec(job["model"], "iris", 4, 3, job["optimization_seed"], n_jobs=1)
            finally:
                captured_warnings.extend(warning_summary(caught, "model_construction"))
                write(folder / "warnings.json", captured_warnings)
        full_candidate_count = spec.candidate_count
        estimator, grid = clone(spec.estimator), spec.param_grid
        if smoke:
            grid = {key: [value] for key, value in next(iter(ParameterGrid(grid))).items()}
            if job["model"] == "mlp":
                estimator.set_params(max_iter=2)
        candidate_count = len(ParameterGrid(grid))
        settings = {"job": job, "smoke": bool(smoke), "model_spec": spec.settings,
                    "full_param_grid": spec.param_grid, "effective_param_grid": grid,
                    "effective_estimator_params": {key: value for key, value in estimator.get_params(deep=False).items()
                                                   if isinstance(value, (str, int, float, bool, list, tuple, type(None)))},
                    "full_grid_candidate_count": full_candidate_count,
                    "grid_candidate_count": candidate_count, "runtime": runtime(),
                    "partition_sha256": partition_sha,
                    "source_sha256": {relative(SOURCE / name): sha256(SOURCE / name) for name in
                                      ("common.py", "ml_models.py", "run_ml_experiments.py")},
                    "preprocessing": "Provided train-fitted min-max features clipped to [0,1]; no additional scaling",
                    "selection": "Fixed validation fold balanced accuracy; first exact maximum; refit on training only"}
        write(folder / "settings.json", settings)
        update("grid_search", grid_candidates_total=candidate_count,
               partition_sha256=partition_sha,
               training_rows=len(data["y_train"]), validation_rows=len(data["y_val"]), test_rows=len(data["y_test"]))

        x_search, y_search, cv = search_inputs(data)
        search = GridSearchCV(estimator, param_grid=grid, scoring="balanced_accuracy", cv=cv,
                              n_jobs=1, refit=False, error_score="raise", return_train_score=True,
                              verbose=3)
        search_start = time.perf_counter()
        with (folder / "search.log").open("w") as log:
            tee = SearchLog(sys.stdout, log, candidate_finished)
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(tee), redirect_stderr(tee), threadpool_limits(limits=1):
                warnings.simplefilter("always")
                try:
                    search.fit(x_search, y_search)
                finally:
                    captured_warnings.extend(warning_summary(caught, "grid_search"))
                    write(folder / "warnings.json", captured_warnings)
        search_seconds = time.perf_counter() - search_start
        cv_results = clean(search.cv_results_)
        write(folder / "cv_results.json", {
            "job": job, "smoke": bool(smoke), "scoring": "balanced_accuracy",
            "cv": "one PredefinedSplit: -1=training, 0=validation",
            "test_score_field_means": "validation score, as named by sklearn's CV API",
            "candidate_count": candidate_count, "results": cv_results,
        })
        selection = first_validation_max(search.cv_results_)
        if selection["best_index"] != int(search.best_index_):
            raise AssertionError("Explicit tie rule differs from GridSearchCV selection")
        write(folder / "selection.json", selection)
        update("training_selected", grid_candidates_completed=candidate_count,
               best_validation_score=selection["best_validation_score"], best_params=selection["best_params"],
               search_seconds=search_seconds)

        # Do not call search.refit or concatenate validation into final training.
        selected = clone(estimator).set_params(**selection["best_params"])
        training_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            try:
                selected.fit(data["x_train"], data["y_train"])
            finally:
                captured_warnings.extend(warning_summary(caught, "training_selected"))
                write(folder / "warnings.json", captured_warnings)
        training_seconds = time.perf_counter() - training_start
        checkpoint_path = folder / "best.joblib"
        temporary = checkpoint_path.with_name(checkpoint_path.name + f".tmp-{os.getpid()}")
        joblib.dump(selected, temporary, compress=3)
        os.replace(temporary, checkpoint_path)
        checkpoint_sha = sha256(checkpoint_path)
        update("testing", selected_training_fits_completed=1, training_seconds=training_seconds)

        test_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            try:
                scores = model_scores(selected, data["x_test"])
                test_metrics = {"balanced_accuracy": float(balanced_accuracy_score(data["y_test"], scores["predicted"])),
                                "accuracy": float(accuracy_score(data["y_test"], scores["predicted"])),
                                "macro_f1": float(f1_score(data["y_test"], scores["predicted"], labels=[0, 1, 2], average="macro", zero_division=0))}
            finally:
                captured_warnings.extend(warning_summary(caught, "testing"))
                write(folder / "warnings.json", captured_warnings)
        test_seconds = time.perf_counter() - test_start
        predictions = {"row_index": data["index_test"], "true": data["y_test"],
                       "predicted": scores["predicted"], "classes": scores["classes"]}
        score_key = "probabilities" if scores["score_type"] == "probability" else "decision_scores"
        predictions[score_key] = scores["scores"]
        atomic_npz(folder / "predictions.npz", predictions)
        complexity = model_complexity(selected)
        complexity["checkpoint_bytes"] = checkpoint_path.stat().st_size
        complexity["training_rows"] = len(data["y_train"])
        if job["model"] == "mlp" and (complexity["layers"] != [4, 32, 3] or selected.out_activation_ != "softmax"):
            raise AssertionError("Iris MLP architecture must be 4 -> 32 -> 3 with softmax output")
        if sha256(partition_path) != partition_sha:
            raise RuntimeError("Partition source changed during the job")
        result = {
            **job, "job": job, "status": "complete", "success": True, "smoke": bool(smoke),
            "software_validation_only": bool(smoke),
            "test_metrics": test_metrics, "best_params": selection["best_params"],
            "best_validation_score": selection["best_validation_score"],
            "best_candidate_index": selection["best_index"],
            "tied_best_indices": selection["tied_best_indices"], "tied_best_count": selection["tied_best_count"],
            "grid_candidate_count": candidate_count, "full_grid_candidate_count": full_candidate_count,
            "grid_candidates_completed": candidate_count, "selected_training_fits_completed": 1,
            "test_evaluations_completed": 1,
            "search_seconds": search_seconds, "training_seconds": training_seconds,
            "test_seconds": test_seconds, "elapsed_seconds": time.perf_counter() - started,
            "partition_sha256": partition_sha, "checkpoint_sha256": checkpoint_sha,
            "predictions_sha256": sha256(folder / "predictions.npz"),
            "model_complexity": complexity, "score_type": scores["score_type"],
            "n_train": len(data["y_train"]), "n_val": len(data["y_val"]), "n_test": len(data["y_test"]),
            "warning_count": sum(item["count"] for item in captured_warnings),
            "warnings": captured_warnings, "runtime": settings["runtime"],
            "source_sha256": settings["source_sha256"], "completed_utc": utc_now(),
            "artifacts": {name: relative(folder / name) for name in
                          ("best.joblib", "predictions.npz", "settings.json", "cv_results.json", "selection.json", "warnings.json", "search.log")},
            "test_used_for_selection": False, "selected_model_fitted_on": "training partition only",
        }
        write(folder / "result.json", result)
        update("complete", test_evaluations_completed=1, test_metrics=test_metrics,
               test_seconds=test_seconds, success=True)
        write(folder / "complete.json", {**result, "result_sha256": sha256(folder / "result.json")})
        return clean(result)
    except BaseException as exc:
        write(folder / "warnings.json", captured_warnings)
        write(folder / "error.json", {"job": job, "smoke": bool(smoke), "status": "failed", "success": False,
                                    "exception_type": type(exc).__name__, "message": portable_text(exc),
                                    "traceback": portable_text(traceback.format_exc()), "failed_utc": utc_now()})
        update("failed", success=False, error=portable_text(exc))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="ID from this campaign's jobs.json")
    parser.add_argument("--smoke", action="store_true", help="One candidate on complete real partitions; output only under qa/")
    args = parser.parse_args(argv)
    result = run_job(load_job(args.job_id), smoke=args.smoke)
    print(f"{result['id']}: completed; balanced_accuracy={result['test_metrics']['balanced_accuracy']:.6f}; smoke={result['smoke']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
