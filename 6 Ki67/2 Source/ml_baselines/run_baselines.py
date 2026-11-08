"""Validation-selected conventional classifiers for the fixed Ki-67 partitions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, ParameterGrid, PredefinedSplit
from threadpoolctl import threadpool_limits

from .data import DATASETS, OUTPUT, REPRESENTATIONS, TASKS, load_data, sha256
from .models import available_models, build_spec, model_complexity, model_scores
from .repeat_results import apply_selected_repeats

from .paths import REPO
SEEDS = (1729, 2718, 3141)
_formatwarning = warnings.formatwarning


def portable_warning(message, category, filename, lineno, line=None):
    filename = filename.replace(str(REPO), ".").replace(str(Path.home()), "<home>")
    return _formatwarning(message, category, filename, lineno, line)


warnings.formatwarning = portable_warning


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, document):
    from .paths import writable
    writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(clean(document), indent=2, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def relative(path):
    return Path(path).resolve().relative_to(REPO).as_posix()


def runtime():
    return {"python": sys.version.split()[0], "versions": {name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl", "pillow")},
        "threads": {key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def input_for_search(data, dataset, representation):
    """Share a read-only concatenation while preserving the explicit split."""
    train, valid = data["X_train"], data["X_val"]
    if representation != "rgb":
        matrix = np.concatenate((train, valid), axis=0)
    else:
        folder = REPO / OUTPUT / "data" / dataset / representation
        target = folder / "search.npy"
        with (folder / ".search.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            expected = {"shape": [len(train) + len(valid), train.shape[1]], "dtype": "float32",
                        "source_identity": data["provenance"]["input_identity_sha256"],
                        "source_artifact_sha256": data["provenance"]["feature_artifact_sha256"]}
            metadata_path = folder / "search-metadata.json"
            if target.exists() and (not metadata_path.exists() or
                    json.loads(metadata_path.read_text()) != expected):
                raise RuntimeError("Search matrix provenance differs from source partitions")
            if not target.exists():
                temporary = folder / "search.partial.npy"
                matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                    shape=(len(train) + len(valid), train.shape[1]))
                for source, offset in ((train, 0), (valid, len(train))):
                    for i in range(len(source)):
                        matrix[offset + i] = source[i]
                matrix.flush()
                del matrix
                os.replace(temporary, target)
                write(metadata_path, expected)
        matrix = np.load(target, mmap_mode="r", allow_pickle=False)
        if list(matrix.shape) != expected["shape"] or matrix.dtype != np.float32:
            raise ValueError("Unexpected search matrix dimensions or dtype")
    labels = np.concatenate((data["y_train"], data["y_val"]))
    cv = PredefinedSplit(np.r_[np.full(len(train), -1), np.zeros(len(valid))].astype(int))
    train_indices, valid_indices = next(cv.split())
    assert np.array_equal(train_indices, np.arange(len(train)))
    assert np.array_equal(valid_indices, np.arange(len(train), len(train) + len(valid)))
    return matrix, labels, cv


def metrics(y, predicted, scores, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=classes, zero_division=0)
    aucs = []
    if len(classes) == 2:
        positive = scores if scores.ndim == 1 else scores[:, 1]
        aucs = [roc_auc_score(y == classes[1], positive)]
    else:
        if scores.shape != (len(y), len(classes)):
            raise ValueError("Invalid multiclass score shape")
        aucs = [roc_auc_score(y == label, scores[:, i]) for i, label in enumerate(classes)]
    return {"balanced_accuracy": balanced_accuracy_score(y, predicted),
            "accuracy": accuracy_score(y, predicted), "macro_f1": float(f1.mean()),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "roc_auc": float(np.mean(aucs)), "class_auc": aucs,
            "confusion_matrix": confusion_matrix(y, predicted, labels=classes),
            "precision": precision, "recall": recall, "f1": f1, "support": support}


def warning_summary(captured):
    counts = Counter((item.category.__name__, str(item.message).replace(str(REPO), "."))
                     for item in captured)
    return [{"category": category, "message": message, "count": count}
            for (category, message), count in counts.items()]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def merge_warning_summaries(*summaries):
    counts = Counter()
    for summary in summaries:
        for item in summary:
            counts[(item["category"], item["message"])] += item["count"]
    return [{"category": category, "message": message, "count": count}
            for (category, message), count in counts.items()]


def validation_balanced_accuracy(estimator, features, targets):
    return float(balanced_accuracy_score(targets, estimator.predict(features)))


class ResumableGridSearchCV(GridSearchCV):
    """Checkpoint complete candidate evaluations without changing the CV split."""

    def _run_search(self, evaluate_candidates):
        context = self.checkpoint_context
        state, grid, pending = context["state"], context["grid"], context["pending"]
        chunk_size = context["chunk_size"]
        evaluated_this_call = 0
        for offset in range(0, len(pending), chunk_size):
            indices = pending[offset:offset + chunk_size]
            context["progress"]("grid_search", seed=state["seed"],
                evaluated=len(state["rows"]), candidates=len(grid), active_candidates=indices,
                candidate_parameters=[grid[index] for index in indices])
            started, cpu_started = time.perf_counter(), time.process_time()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    results = evaluate_candidates([grid[index] for index in indices])
                except BaseException:
                    state["elapsed_seconds"] += time.perf_counter() - started
                    state["cpu_seconds"] += time.process_time() - cpu_started
                    state["warnings"] = merge_warning_summaries(
                        state["warnings"], warning_summary(captured))
                    state["updated_utc"] = utc_now()
                    write(context["path"], state)
                    raise
            state["elapsed_seconds"] += time.perf_counter() - started
            state["cpu_seconds"] += time.process_time() - cpu_started
            state["warnings"] = merge_warning_summaries(state["warnings"], warning_summary(captured))
            for position, index in enumerate(indices, start=evaluated_this_call):
                score = float(results["mean_test_score"][position])
                if not np.isfinite(score):
                    raise ValueError("A grid candidate produced a non-finite validation metric")
                if clean(results["params"][position]) != clean(grid[index]):
                    raise RuntimeError("GridSearchCV returned an unexpected candidate order")
                state["rows"].append({"candidate": index, "params": clean(grid[index]),
                    "validation_balanced_accuracy": score,
                    "fit_seconds": float(results["mean_fit_time"][position]),
                    "validation_seconds": float(results["mean_score_time"][position])})
            evaluated_this_call += len(indices)
            state["rows"].sort(key=lambda row: row["candidate"])
            state["updated_utc"] = utc_now()
            write(context["path"], state)
            context["progress"]("grid_search", seed=state["seed"],
                evaluated=len(state["rows"]), candidates=len(grid), active_candidates=[],
                completed_candidates=indices,
                last_validation_ba=state["rows"][-1]["validation_balanced_accuracy"])


def resumable_grid_search(estimator, param_grid, X, y, cv, *, directory, seed,
                          contract_hash, grid_jobs, progress, scoring=validation_balanced_accuracy):
    """Return the original search-result schema in canonical ParameterGrid order."""
    grid = clean(list(ParameterGrid(param_grid)))
    path = directory / f"candidate-results-seed-{seed}.json"
    if path.exists():
        state = json.loads(path.read_text())
        if (state.get("schema") != "resumable-gridsearch-candidates-v1" or
                state.get("contract_sha256") != contract_hash or state.get("seed") != seed or
                state.get("params") != grid):
            raise RuntimeError("Candidate checkpoints do not match the current training contract")
    else:
        state = {"schema": "resumable-gridsearch-candidates-v1", "contract_sha256": contract_hash,
            "seed": seed, "params": grid, "rows": [], "warnings": [],
            "elapsed_seconds": 0.0, "cpu_seconds": 0.0, "created_utc": utc_now()}
    seen = set()
    for row in state["rows"]:
        index = row["candidate"]
        if (not isinstance(index, int) or index < 0 or index >= len(grid) or index in seen or
                row["params"] != grid[index]):
            raise RuntimeError("Invalid or duplicate checkpoint candidate")
        if (not 0 <= row["validation_balanced_accuracy"] <= 1 or
                not all(np.isfinite(row[key]) and row[key] >= 0
                    for key in ("fit_seconds", "validation_seconds"))):
            raise RuntimeError("Invalid checkpoint metric or timing")
        seen.add(index)
    pending = [index for index in range(len(grid)) if index not in seen]
    progress("grid_search", seed=seed, evaluated=len(seen), candidates=len(grid),
             resumed_candidates=len(seen), active_candidates=[])
    if pending:
        search = ResumableGridSearchCV(estimator, param_grid, scoring=scoring,
            cv=cv, refit=False, n_jobs=grid_jobs, pre_dispatch=grid_jobs,
            error_score="raise", return_train_score=False)
        search.checkpoint_context = {"state": state, "grid": grid, "pending": pending,
            "chunk_size": grid_jobs, "progress": progress, "path": path}
        search.fit(X, y)
    rows = sorted(state["rows"], key=lambda row: row["candidate"])
    if [row["candidate"] for row in rows] != list(range(len(grid))):
        raise RuntimeError("The candidate search is incomplete")
    return {"seed": seed, "params": grid,
        "validation_balanced_accuracy": [row["validation_balanced_accuracy"] for row in rows],
        "fit_seconds": [row["fit_seconds"] for row in rows],
        "validation_seconds": [row["validation_seconds"] for row in rows],
        "elapsed_seconds": state["elapsed_seconds"], "cpu_seconds": state["cpu_seconds"],
        "grid_jobs": grid_jobs, "cpu_timing_scope": "Parent process only, excludes parallel search workers",
        "warnings": state["warnings"], "fitted_models_saved": 0}


def fit_job(dataset, representation, name, task, threads=1):
    directory = REPO / OUTPUT / "runs" / dataset / representation / task / name
    complete = directory / "complete.json"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        cpu_started = time.process_time()

        def progress(phase, **details):
            document = {"phase": phase, "updated_utc": utc_now(), "pid": os.getpid(),
                "elapsed_seconds": time.perf_counter() - started,
                "context": f"{dataset}/{representation}/{task}/{name}", **details}
            write(directory / "progress.json", document)
            print(json.dumps(clean(document)), flush=True)

        if not complete.exists():
            progress("preparing_data")
        data = load_data(REPO, dataset, representation, task)
        n_classes = len(np.unique(data["y_train"]))
        spec = build_spec(name, representation, data["X_train"].shape[1], n_classes, SEEDS[0])
        seeds = SEEDS if spec.stochastic else SEEDS[:1]
        grid = list(ParameterGrid(spec.param_grid))
        contract = {"settings": spec.settings, "grid": grid, "seeds": seeds,
                    "input": data["provenance"], "versions": runtime()["versions"],
                    "models_source_sha256": sha256(Path(__file__).parent / "models.py"),
                    "protocol": "predefined-train-valid_mean-seed-BA_refit-train-only_v1"}
        contract_hash = hashlib.sha256(json.dumps(clean(contract), sort_keys=True).encode()).hexdigest()
        if (directory / "contract.json").exists():
            if json.loads((directory / "contract.json").read_text())["sha256"] != contract_hash:
                raise RuntimeError("The existing experiment has a different training contract")
        elif (directory / "spec.json").exists():
            raise RuntimeError("Existing experiment lacks a verifiable training contract")
        else:
            write(directory / "contract.json", {"sha256": contract_hash, "contract": contract})
        if complete.exists():
            return json.loads(complete.read_text())
        write(directory / "spec.json", {"dataset": dataset, "representation": representation,
            "task": task, "model": name, "seeds": seeds, "settings": spec.settings,
            "grid": grid, "selection": "Highest mean validation balanced accuracy over optimization seeds",
            "tie_rule": "First candidate in ParameterGrid order", "refit": "Training partition only",
            "input": data["provenance"], "runtime": runtime(), "class_counts": {
                split: np.bincount(data[f"y_{split}"], minlength=n_classes) for split in ("train", "val", "test")}})
        X, y, cv = input_for_search(data, dataset, representation)
        preparation_seconds = time.perf_counter() - started
        searches = []
        with threadpool_limits(limits=threads):
            for seed in seeds:
                path = directory / f"grid-seed-{seed}.json"
                if path.exists():
                    search_result = json.loads(path.read_text())
                    if search_result["params"] != clean(grid):
                        raise RuntimeError("Stored search differs from the current grid")
                    searches.append(search_result)
                    continue
                spec = build_spec(name, representation, X.shape[1], n_classes, seed)
                grid_jobs = 4 if name == "gradient_boosting" and representation != "rgb" else 1
                search_result = resumable_grid_search(spec.estimator, spec.param_grid, X, y, cv,
                    directory=directory, seed=seed, contract_hash=contract_hash,
                    grid_jobs=grid_jobs, progress=progress)
                write(path, search_result)
                searches.append(clean(search_result))
            values = np.array([result["validation_balanced_accuracy"] for result in searches])
            mean_validation = values.mean(axis=0)
            selected_index = int(np.argmax(mean_validation))
            best_params = grid[selected_index]
            selection = {"candidate_index": selected_index, "best_params": best_params,
                "mean_validation_balanced_accuracy": mean_validation[selected_index],
                "seed_validation_balanced_accuracy": values[:, selected_index],
                "all_candidate_mean_validation_balanced_accuracy": mean_validation,
                "seeds": seeds, "n_candidates": len(grid), "test_used_for_selection": False}
            write(directory / "selection.json", selection)
            selected = []
            for position, seed in enumerate(seeds):
                seed_directory = directory / f"seed-{seed}"
                seed_directory.mkdir(exist_ok=True)
                spec = build_spec(name, representation, X.shape[1], n_classes, seed)
                estimator = clone(spec.estimator).set_params(**best_params)
                progress("selected_model_fit", seed=seed, best_params=best_params,
                         completed_seeds=position, selected_seeds=len(seeds))
                final_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    estimator.fit(data["X_train"], data["y_train"])
                fit_seconds = time.perf_counter() - final_started
                train_warnings = warning_summary(captured)
                inference = {}
                prediction_seconds = {}
                for split in ("val", "test"):
                    progress("selected_model_prediction", seed=seed, split=split,
                             completed_seeds=position, selected_seeds=len(seeds))
                    prediction_started = time.perf_counter()
                    scores = model_scores(estimator, data[f"X_{split}"])
                    prediction_seconds[split] = time.perf_counter() - prediction_started
                    if not np.array_equal(scores["classes"], np.arange(n_classes)):
                        raise ValueError("Unexpected class order")
                    inference[split] = metrics(data[f"y_{split}"], scores["predicted"],
                                               scores["scores"], scores["classes"])
                    np.savez_compressed(seed_directory / f"{split}-predictions.npz",
                        sample_ids=data[f"ids_{split}"], y_true=data[f"y_{split}"],
                        y_pred=scores["predicted"], scores=scores["scores"],
                        classes=scores["classes"], score_type=np.asarray(scores["score_type"]))
                if abs(inference["val"]["balanced_accuracy"] - values[position, selected_index]) > 1e-12:
                    raise ValueError("Final training-only refit did not reproduce its selected validation BA")
                model_path = seed_directory / "best.joblib"
                progress("selected_model_serialization", seed=seed,
                         completed_seeds=position, selected_seeds=len(seeds))
                serialization_started = time.perf_counter()
                joblib.dump(estimator, model_path, compress=3)
                serialization_seconds = time.perf_counter() - serialization_started
                summary = {"seed": seed, "best_params": best_params, "metrics": inference,
                    "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                    "serialization_seconds": serialization_seconds,
                    "model": relative(model_path), "model_bytes": model_path.stat().st_size,
                    "model_sha256": sha256(model_path), "complexity": model_complexity(estimator),
                    "warnings": train_warnings, "trained_on": "train", "selected_on": "val"}
                write(seed_directory / "result.json", summary)
                selected.append(clean(summary))
                del estimator
        metrics_summary = {}
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc"):
            scores = np.array([item["metrics"]["test"][metric] for item in selected])
            metrics_summary[metric] = {"mean": scores.mean(), "sample_sd": scores.std(ddof=1) if len(scores) > 1 else 0.0}
        attempt_seconds = time.perf_counter() - started
        result = {"dataset": dataset, "representation": representation, "task": task, "model": name,
            "seeds": seeds, "deterministic": not spec.stochastic,
            "search_fits": len(grid) * len(seeds), "selected_fits": len(seeds),
            "selection": selection, "test_metrics": metrics_summary,
            "preparation_seconds": preparation_seconds,
            "search_seconds": sum(item["elapsed_seconds"] for item in searches),
            "total_seconds": attempt_seconds,
            "attempt_elapsed_seconds": attempt_seconds,
            "cpu_seconds": time.process_time() - cpu_started,
            "timing_scope": {
                "total_seconds": "Elapsed time of this worker invocation for this context, including preparation",
                "attempt_elapsed_seconds": "Same as total_seconds, excludes earlier worker invocations",
                "search_seconds": "Sum of recorded grid durations across seeds, including preserved candidate checkpoints from earlier invocations",
                "cpu_seconds": "CPU time of this worker invocation, excludes parallel child processes",
                "campaign_elapsed": "Not represented by sums of worker or search durations"},
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "selected_models": selected, "input_provenance": data["provenance"]}
        write(complete, result)
        progress("complete", total_seconds=result["total_seconds"])
        print(json.dumps({"completed": f"{dataset}/{representation}/{task}/{name}",
            "ba": metrics_summary["balanced_accuracy"], "seconds": result["total_seconds"]}), flush=True)
        return clean(result)


def summarize():
    root = REPO / OUTPUT
    results = [apply_selected_repeats(json.loads(path.read_text()), path, REPO)
               for path in sorted((root / "runs").glob("*/*/*/*/complete.json"))]
    scope_path = root / "provenance/campaign-scope.json"
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    included = scope.get("included_representations", list(REPRESENTATIONS))
    results = [item for item in results if item["representation"] in included]
    fields = ["dataset", "representation", "task", "model", "seeds", "best_params", "validation_ba",
              "test_ba", "test_ba_sd", "test_accuracy", "test_accuracy_sd", "test_macro_f1", "test_macro_f1_sd",
              "test_roc_auc", "test_roc_auc_sd", "search_fits", "selected_fits", "search_seconds", "total_seconds",
              "search_seeds", "selected_repeat_fits", "selected_repeat_training_seconds", "selected_repeat_total_seconds",
              "selected_repeat_record"]
    (root / "summary").mkdir(exist_ok=True)
    temporary = root / "summary" / f"results.csv.tmp-{os.getpid()}"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            row = {key: item[key] for key in ("dataset", "representation", "task", "model", "search_fits", "selected_fits", "search_seconds", "total_seconds")}
            row.update(seeds=json.dumps(item["seeds"]), best_params=json.dumps(item["selection"]["best_params"], sort_keys=True),
                       validation_ba=item["selection"]["mean_validation_balanced_accuracy"],
                       search_seeds=json.dumps(item["search_seeds"]))
            for field in ("selected_repeat_fits", "selected_repeat_training_seconds", "selected_repeat_total_seconds", "selected_repeat_record"):
                row[field] = item[field]
            for source, destination in (("balanced_accuracy", "test_ba"), ("accuracy", "test_accuracy"),
                                        ("macro_f1", "test_macro_f1"), ("roc_auc", "test_roc_auc")):
                row[destination] = item["test_metrics"][source]["mean"]
                row[destination + "_sd"] = item["test_metrics"][source]["sample_sd"]
            writer.writerow(row)
    os.replace(temporary, root / "summary/results.csv")
    write(root / "summary/counts.json", {"completed_comparisons": len(results), "planned_comparisons": scope.get("planned_comparisons", 336),
        "search_fits": sum(item["search_fits"] for item in results),
        "selected_models": sum(item["selected_fits"] for item in results),
        "three_execution_comparisons": sum(item["seeds"] == list(SEEDS) for item in results),
        "additional_selected_repetitions": sum(item["selected_repeat_fits"] for item in results),
        "additional_selected_training_seconds": sum(item["selected_repeat_training_seconds"] for item in results),
        "additional_selected_total_seconds": sum(item["selected_repeat_total_seconds"] for item in results),
        "included_representations": included,
        "accumulated_job_seconds": sum(item["total_seconds"] for item in results)})
    return len(results)


def orchestrate(args):
    root = REPO / OUTPUT
    groups = [(dataset, representation, name) for representation in args.representations
              for name in args.models for dataset in args.datasets if not (name == "mlp" and representation == "rgb")]
    pending = [group for group in groups if any(not (root / "runs" / group[0] / group[1] / task / group[2] / "complete.json").exists() for task in args.tasks)]
    runtime_path = root / "provenance/runtime.json"
    if not runtime_path.exists():
        write(runtime_path, runtime())
    script_hashes = {relative(Path(__file__).parent / name): sha256(Path(__file__).parent / name)
                     for name in ("run_baselines.py", "data.py", "models.py")}
    label = args.label
    write(root / "provenance" / f"source-{label}.json", script_hashes)
    write(root / f"plan-{label}.json", {"groups": groups, "tasks": args.tasks, "workers": args.workers,
          "threads_per_worker": args.threads, "seeds": SEEDS, "deterministic_models": "One search seed and initial fit; repeat command adds two selected fits",
          "candidate_models_saved": False, "test_used_for_selection": False,
          "representation_preprocessing": "Unchanged source-training-standardized frozen DINOv2 or Phikon-v2 features"})
    (root / "logs").mkdir(parents=True, exist_ok=True)
    active, failures = [], []
    started = time.perf_counter()
    while pending or active:
        while pending and len(active) < args.workers:
            group = pending.pop(0)
            log_path = root / "logs" / ("-".join(group) + ".log")
            log = log_path.open("a")
            environment = os.environ.copy()
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(args.threads)
            process = subprocess.Popen([sys.executable, "-m", __package__ + ".run_baselines", "worker", "--group", *group,
                       "--tasks", *args.tasks, "--threads", str(args.threads)], cwd=REPO, env=environment,
                       stdout=log, stderr=subprocess.STDOUT)
            active.append((group, process, log))
        remaining = []
        for group, process, log in active:
            code = process.poll()
            if code is None:
                remaining.append((group, process, log))
            else:
                log.close()
                if code:
                    failures.append({"group": group, "returncode": code})
        active = remaining
        completed = summarize()
        write(root / f"status-{label}.json", {"active": [group for group, _, _ in active],
            "queued_groups": len(pending), "completed_comparisons": completed, "failures": failures,
            "elapsed_seconds": time.perf_counter() - started})
        if pending or active:
            time.sleep(5)
    print(json.dumps({"completed_comparisons": summarize(), "failures": failures,
                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker", "summary"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--representations", nargs="+", default=["dinov2", "phikonv2"], choices=REPRESENTATIONS)
    parser.add_argument("--models", nargs="+", default=list(available_models()), choices=available_models())
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--label", default="features")
    parser.add_argument("--group", nargs=3)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be positive integers")
    if args.command == "worker":
        if args.group is None:
            parser.error("worker requires --group DATASET REPRESENTATION MODEL")
        dataset, representation, model = args.group
        if dataset not in DATASETS or representation not in REPRESENTATIONS or model not in available_models():
            parser.error("--group must contain a supported dataset, representation and model")
    if args.command == "summary":
        print(summarize())
    elif args.command == "worker":
        dataset, representation, model = args.group
        for task in args.tasks:
            fit_job(dataset, representation, model, task, args.threads)
    else:
        raise SystemExit(orchestrate(args))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc().replace(str(REPO), ".").replace(str(Path.home()), "<home>")
        print(message, file=sys.stderr, flush=True)
        raise SystemExit(1)
