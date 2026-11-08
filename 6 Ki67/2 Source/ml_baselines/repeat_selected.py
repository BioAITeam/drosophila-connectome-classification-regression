"""Repeat validation-selected deterministic classifiers with two additional seeds.

Original searches and fitted artifacts remain immutable. Repetitions fit only
the training partition and retain the original validation-selected parameters.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
import traceback
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from .data import OUTPUT, load_data, sha256
from .models import build_spec, model_complexity, model_scores
from .run_baselines import clean, metrics, relative, runtime, utc_now, warning_summary, write

from .paths import REPO
ROOT = REPO / OUTPUT
SEEDS = (1729, 2718, 3141)
MODELS = {"knn", "svc", "xgboost", "logistic_regression", "qda", "lda", "gaussian_nb"}
METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc")
SNAPSHOT = ROOT / "provenance/selected-repeats-originals.json"


def read(path):
    return json.loads(path.read_text())


def contexts():
    output = []
    for path in sorted((ROOT / "runs").glob("*/*/*/*/complete.json")):
        item = read(path)
        if item["representation"] not in {"dinov2", "phikonv2"} or item["model"] not in MODELS:
            continue
        if not item["deterministic"] or item["seeds"] != [1729] or item["selected_fits"] != 1:
            raise ValueError("Unexpected source repetition protocol")
        output.append((tuple(item[k] for k in ("dataset", "representation", "task", "model")), path))
    if not output or len({k for k, _ in output}) != len(output):
        raise ValueError("Expected unique completed deterministic feature contexts")
    return output


def ensure_snapshot():
    paths = []
    for _, source in contexts():
        paths.extend(p for p in source.parent.rglob("*") if p.is_file() and not p.name.startswith("."))
    paths.extend(Path(__file__).parent / name for name in ("models.py", "data.py"))
    if SNAPSHOT.exists():
        snapshot = read(SNAPSHOT)
        if set(snapshot["files"]) != {relative(p) for p in paths}:
            raise ValueError("Original artifact inventory changed")
        for path in paths:
            if sha256(path) != snapshot["files"][relative(path)]:
                raise ValueError(f"Original artifact changed: {relative(path)}")
    else:
        snapshot = {"schema": "selected-repeats-originals-v1", "created_utc": utc_now(),
            "files": {relative(p): sha256(p) for p in paths},
            "original_contexts": len(contexts()), "additional_seeds": list(SEEDS[1:])}
        write(SNAPSHOT, snapshot)
    return snapshot


@lru_cache(maxsize=24)
def inputs(dataset, representation, task):
    return load_data(REPO, dataset, representation, task)


def progress(path, phase, **kwargs):
    value = {"phase": phase, "updated_utc": utc_now(), "pid": os.getpid(), **kwargs}
    write(path / "progress.json", value)
    print(json.dumps(clean(value)), flush=True)


def compare_predictions(source, repeated):
    result = {}
    with np.load(source, allow_pickle=False) as a, np.load(repeated, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Prediction schema mismatch")
        for key in ("sample_ids", "y_true", "classes", "score_type"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Prediction identity mismatch: {key}")
        result["class_predictions_identical"] = bool(np.array_equal(a["y_pred"], b["y_pred"]))
        result["scores_identical"] = bool(np.array_equal(a["scores"], b["scores"]))
        result["max_absolute_score_difference"] = float(np.max(np.abs(a["scores"] - b["scores"])))
        result["changed_predictions"] = int(np.count_nonzero(a["y_pred"] != b["y_pred"]))
    return result


def run_context(key):
    dataset, representation, task, name = key
    original_dir = ROOT / "runs" / dataset / representation / task / name
    target = ROOT / "selected-repeats" / dataset / representation / task / name
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source = read(original_dir / "complete.json")
        selection = read(original_dir / "selection.json")
        source_hashes = {"original_complete_sha256": sha256(original_dir / "complete.json"),
            "original_selection_sha256": sha256(original_dir / "selection.json"),
            "original_contract_sha256": sha256(original_dir / "contract.json")}
        if (target / "complete.json").exists():
            prior = read(target / "complete.json")
            if any(prior[k] != v for k, v in source_hashes.items()):
                raise ValueError("Repeat source changed")
            for record in prior["seed_results"]:
                item = read(REPO / record)
                if sha256(REPO / item["model"]) != item["model_sha256"]:
                    raise ValueError("Previously repeated model checksum changed")
            return prior
        started = time.perf_counter()
        progress(target, "loading_verified_inputs", context="/".join(key))
        data = inputs(dataset, representation, task)
        contract = read(original_dir / "contract.json")["contract"]
        if data["provenance"] != contract["input"] or runtime()["versions"] != contract["versions"]:
            raise ValueError("Input or package version changed since the original experiment")
        if sha256(Path(__file__).parent / "models.py") != contract["models_source_sha256"]:
            raise ValueError("Model implementation changed since hyperparameter selection")
        if selection != source["selection"] or selection["test_used_for_selection"] is not False:
            raise ValueError("Original validation selection is inconsistent")
        n_classes = len(np.unique(data["y_train"]))
        preparation_seconds = time.perf_counter() - started
        original_result_path = original_dir / "seed-1729/result.json"
        original_result = read(original_result_path)
        records = [relative(original_result_path)]
        results = [original_result]
        with threadpool_limits(limits=1):
            for seed in SEEDS[1:]:
                seed_dir = target / f"seed-{seed}"
                seed_dir.mkdir(exist_ok=True)
                record_path = seed_dir / "result.json"
                if record_path.exists():
                    result = read(record_path)
                    if any(result[k] != v for k, v in source_hashes.items()):
                        raise ValueError("Partial repeat provenance differs")
                    if result["best_params"] != selection["best_params"]:
                        raise ValueError("Partial repeat parameters differ")
                    if sha256(REPO / result["model"]) != result["model_sha256"]:
                        raise ValueError("Partial repeated model checksum differs")
                    for split, checksum in result["prediction_sha256"].items():
                        if sha256(seed_dir / f"{split}-predictions.npz") != checksum:
                            raise ValueError("Partial predictions checksum differs")
                    results.append(result)
                    records.append(relative(record_path))
                    continue
                random.seed(seed)
                np.random.seed(seed)
                spec = build_spec(name, representation, data["X_train"].shape[1], n_classes, seed)
                if spec.stochastic:
                    raise ValueError("This extension only repeats originally deterministic models")
                estimator = clone(spec.estimator).set_params(**selection["best_params"])
                progress(target, "selected_model_fit", context="/".join(key), seed=seed)
                tick = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    estimator.fit(data["X_train"], data["y_train"])
                fit_seconds = time.perf_counter() - tick
                inference, prediction_seconds, comparisons, prediction_hashes = {}, {}, {}, {}
                for split in ("val", "test"):
                    tick = time.perf_counter()
                    scores = model_scores(estimator, data[f"X_{split}"])
                    prediction_seconds[split] = time.perf_counter() - tick
                    if not np.array_equal(scores["classes"], np.arange(n_classes)):
                        raise ValueError("Unexpected class order")
                    if not np.isfinite(scores["scores"]).all():
                        raise ValueError("Nonfinite classifier scores")
                    inference[split] = metrics(data[f"y_{split}"], scores["predicted"], scores["scores"], scores["classes"])
                    prediction_path = seed_dir / f"{split}-predictions.npz"
                    np.savez_compressed(prediction_path, sample_ids=data[f"ids_{split}"],
                        y_true=data[f"y_{split}"], y_pred=scores["predicted"],
                        scores=scores["scores"], classes=scores["classes"],
                        score_type=np.asarray(scores["score_type"]))
                    prediction_hashes[split] = sha256(prediction_path)
                    comparisons[split] = compare_predictions(original_dir / "seed-1729" / prediction_path.name, prediction_path)
                model_path = seed_dir / "best.joblib"
                tick = time.perf_counter()
                joblib.dump(estimator, model_path, compress=3)
                serialization_seconds = time.perf_counter() - tick
                result = {"seed": seed, "best_params": selection["best_params"], "metrics": inference,
                    "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
                    "serialization_seconds": serialization_seconds,
                    "model": relative(model_path), "model_bytes": model_path.stat().st_size,
                    "model_sha256": sha256(model_path), "complexity": model_complexity(estimator),
                    "warnings": warning_summary(captured), "trained_on": "train", "selected_on": "val",
                    **source_hashes, "prediction_sha256": prediction_hashes,
                    "comparison_with_original": comparisons, "input_provenance": data["provenance"],
                    "runtime": runtime(), "completed_utc": utc_now(),
                    "selection_protocol": "Reuse original validation-selected configuration, no new search"}
                write(record_path, result)
                results.append(clean(result))
                records.append(relative(record_path))
                progress(target, "selected_model_complete", context="/".join(key), seed=seed,
                    fit_seconds=fit_seconds, comparison=comparisons)
                del estimator
        aggregate = {}
        for metric in METRICS:
            values = np.asarray([x["metrics"]["test"][metric] for x in results], dtype=float)
            aggregate[metric] = {"mean": float(values.mean()),
                "sample_sd": 0.0 if np.ptp(values) == 0 else float(values.std(ddof=1))}
        value = {"schema": "selected-repeats-v1", "status": "completed",
            **dict(zip(("dataset", "representation", "task", "model"), key)), **source_hashes,
            "seeds": list(SEEDS), "search_seeds": selection["seeds"], "seed_results": records,
            "aggregate": aggregate, "added_selected_fits": 2, "added_search_fits": 0,
            "added_training_seconds": sum(r["fit_seconds"] for r in results[1:]),
            "added_total_seconds": time.perf_counter() - started,
            "preparation_seconds": preparation_seconds, "completed_utc": utc_now(),
            "all_predictions_identical": all(v["class_predictions_identical"] and v["scores_identical"]
                for r in results[1:] for v in r["comparison_with_original"].values()),
            "timing_scope": "Repeat context worker invocation including input verification, two fits, predictions and serialization"}
        write(target / "complete.json", value)
        progress(target, "completed", context="/".join(key))
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--context", nargs=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("Use between 1 and 16 CPU workers")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    ensure_snapshot()
    available = contexts()
    if args.context:
        available = [(k, p) for k, p in available if k == tuple(args.context)]
        if not available:
            parser.error("Unknown deterministic context")
    # Dispatch expensive saved configurations first using the observed fit time.
    available.sort(key=lambda pair: read(pair[1])["selected_models"][0]["fit_seconds"], reverse=True)
    plan = {"schema": "selected-repeats-plan-v1", "created_utc": utc_now(),
        "contexts": [list(k) for k, _ in available], "additional_fits": len(available) * 2,
        "additional_seeds": list(SEEDS[1:]), "workers": args.workers,
        "threads_per_worker": 1, "additional_search_fits": 0,
        "trained_on": "train", "selected_on": "val", "original_snapshot": relative(SNAPSHOT),
        "source": {relative(Path(__file__)): sha256(Path(__file__))}}
    label = "pilot" if args.context else "campaign"
    write(ROOT / "selected-repeats" / f"plan-{label}.json", plan)
    if args.plan_only:
        print(json.dumps(plan), flush=True)
        return
    start = time.perf_counter()
    status_path = ROOT / "selected-repeats" / f"status-{label}.json"
    failures, completed = [], []
    def status(state):
        write(status_path, {"status": state, "updated_utc": utc_now(), "pid": os.getpid(),
            "completed_contexts": len(completed), "planned_contexts": len(available),
            "completed_additional_fits": 2 * len(completed), "planned_additional_fits": len(available) * 2,
            "failures": failures, "elapsed_seconds": time.perf_counter() - start})
    status("running")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_context, k): k for k, _ in available}
        for future in as_completed(futures):
            key = futures[future]
            try:
                completed.append(future.result())
            except Exception:
                message = traceback.format_exc().replace(str(REPO), ".").replace(str(Path.home()), "<home>")
                failures.append({"context": list(key), "error": message})
                print(json.dumps(failures[-1]), flush=True)
            status("running")
    ensure_snapshot()
    status("failed" if failures else "completed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
