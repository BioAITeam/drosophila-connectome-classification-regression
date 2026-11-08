"""Evaluate selected anatomical BRAP-Net models across image datasets."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .run_shid_multiclass import REPO, relative, resolve_relative, load, sha256, record, timestamp, worker_environment


DEFAULT_OUTPUT = "6 Ki67/6 Workspace/shid-cross-dataset-20261101"
ARCHIVE = "6 Ki67/5 Outputs"
SAVED_EVALUATIONS = ARCHIVE + "/shid-cross-dataset-20261101"
GPU_IDS = []
PATIENT_BINARY = "6 Ki67/5 Outputs/shid-v2-20261031/binary"
PATIENT_MULTICLASS = "6 Ki67/5 Outputs/shid-separate-multiclass-20261101"
TASKS = ("multiclass", "binary5", "binary10", "binary30")
SEEDS = (1729, 2718, 3141)
NAMES = {"bcdata": "BCData", "shid": "SHIDC-B-Ki-67", "shid_patient_disjoint": "SHIDC-B-Ki-67-V2"}
DIRECTIONS = (("bcdata", "shid"), ("bcdata", "shid_patient_disjoint"),
              ("shid", "bcdata"), ("shid_patient_disjoint", "bcdata"))
PROBABILITY_ATOL = 2e-5


def archive_path(value):
    """Resolve archived provenance without modifying checkpoint bytes or hashes."""
    value = str(value)
    if value.startswith("zenodo/payload/ki67/outputs/"):
        value = ARCHIVE + "/" + value.split("zenodo/payload/ki67/outputs/", 1)[1]
    for name in ("shid-v2-20261031", "shid-separate-multiclass-20261101", "shid-cross-dataset-20261101"):
        prefix = "6 Ki67/6 Workspace/" + name
        if value == prefix or value.startswith(prefix + "/"):
            value = ARCHIVE + "/" + value[len("6 Ki67/6 Workspace/"):]
    return resolve_relative(value)


def source_root(cohort, task):
    if cohort == "shid_patient_disjoint":
        return REPO / (PATIENT_MULTICLASS if task == "multiclass" else PATIENT_BINARY)
    return REPO / ARCHIVE


def dataset_key(cohort):
    return "bcdata" if cohort == "bcdata" else "shid"


def prepare(output):
    import torch
    from ki67 import data, features
    from ki67.common import digest
    hashes = {}

    def track(path):
        path = Path(path)
        name = relative(path)
        hashes.setdefault(name, sha256(path))
        return hashes[name]

    cohorts, images, identities = {}, [], {}
    for name in NAMES:
        root = source_root(name, "multiclass")
        path = root / "manifests" / f"{dataset_key(name)}.json"
        track(path)
        document = load(path)
        rows = [row for row in document["rows"] if row["split"] == "test"]
        expected = 402 if name == "bcdata" else 700
        if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
            raise ValueError(f"Invalid target cohort: {name}")
        indices = []
        for row in rows:
            if row["dataset"] != dataset_key(name):
                raise ValueError("Target dataset identity mismatch")
            expected_labels = data.labels_from_counts(row["positive_count"], row["negative_count"])
            if any(row[key] != value for key, value in expected_labels.items()):
                raise ValueError("Target labels differ from annotated cell counts")
            path = data.resolve_image_path(row)
            image_hash = row["image_sha256"]
            if image_hash not in identities:
                identities[image_hash] = len(images)
                images.append({"image": relative(path), "image_sha256": image_hash})
            indices.append(identities[image_hash])
        cohorts[name] = {"name": NAMES[name], "manifest": relative(root / "manifests" / f"{dataset_key(name)}.json"),
                         "rows": rows, "raw_cls_indices": indices, "n_images": expected}
    if len(images) != 1595:
        raise ValueError("Expected 1595 unique target images")
    checkpoints, checkpoint_ids = [], {}
    shared_identity = None
    for cohort in NAMES:
        for task in TASKS:
            root, dataset = source_root(cohort, task), dataset_key(cohort)
            directory = root / "features" / dataset / "phikonv2"
            metadata = load(directory / "metadata.json")
            identity = {"model": metadata["model"], "dimension": metadata["dimension"],
                        "preprocessing": metadata["preprocessing"]}
            if shared_identity is not None and identity != shared_identity:
                raise ValueError("Frozen extractor identity or preprocessing differs across sources")
            shared_identity = identity
            if metadata.get("frozen") is not True or metadata["dimension"] != 1024:
                raise ValueError("Expected frozen Phikon-v2 descriptors")
            metadata_hash = track(directory / "metadata.json")
            manifest_hash = track(root / "manifests" / f"{dataset}.json")
            graph_hash = track(root / "graphs/anatomical.npz")
            track(root / "graphs/anatomical.json")
            for name in ("mean.npy", "std.npy"):
                if track(directory / name) != metadata["artifact_sha256"][name]:
                    raise ValueError("Source scaler checksum mismatch")
            mean = np.load(directory / "mean.npy", allow_pickle=False)
            std = np.load(directory / "std.npy", allow_pickle=False)
            if mean.shape != (1024,) or std.shape != (1024,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("Invalid source training scaler")
            for seed in SEEDS:
                run = root / "runs" / dataset / "phikonv2" / task / "anatomical" / f"seed-{seed}"
                path = run / "best.pt"
                selected = load(run / "best.json")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                spec = saved["spec"]
                expected = {"dataset": dataset, "representation": "phikonv2", "task": task,
                            "graph": "anatomical", "seed": seed, "graph_sha256": graph_hash,
                            "manifest_sha256": manifest_hash, "feature_metadata_sha256": metadata_hash}
                if any(spec.get(key) != value for key, value in expected.items()) or spec.get("joint_training"):
                    raise ValueError("Expected a source-only anatomical Phikon-v2 checkpoint")
                if digest(spec) != saved["spec_digest"] or selected["epoch"] != saved["epoch"]:
                    raise ValueError("Checkpoint configuration or selected epoch mismatch")
                if selected["criterion"] != "validation_balanced_accuracy" or abs(selected["score"] - saved["score"]) > 1e-12:
                    raise ValueError("Checkpoint was not selected by source validation balanced accuracy")
                selected_predictions = run / selected["prediction_path"]
                track(run / "best.json")
                track(selected_predictions)
                entry = {"id": f"{cohort}-{task}-seed-{seed}", "source_cohort": cohort,
                         "source_name": NAMES[cohort], "source_output": relative(root), "dataset": dataset,
                         "task": task, "seed": seed, "checkpoint": relative(path), "checkpoint_sha256": track(path),
                         "selected_epoch": int(saved["epoch"]), "source_validation_score": float(saved["score"]),
                         "source_selected_predictions": relative(selected_predictions),
                         "source_feature_metadata": relative(directory / "metadata.json"),
                         "source_scaler_mean": relative(directory / "mean.npy"),
                         "source_scaler_std": relative(directory / "std.npy")}
                checkpoint_ids[cohort, task, seed] = len(checkpoints)
                checkpoints.append(entry)
    if len(checkpoints) != 36 or len({item["checkpoint"] for item in checkpoints}) != 36:
        raise ValueError("Expected 36 distinct source-only checkpoints")
    snapshot = features.verified_snapshot(REPO / ARCHIVE, "phikonv2", shared_identity["model"])
    for name in shared_identity["model"]["file_sha256"]:
        track(snapshot / name)
    track(REPO / ARCHIVE / "cache/huggingface/phikonv2-revision.json")
    jobs = [{"id": f"{source}_to_{target}-{task}-seed-{seed}", "direction": f"{source}_to_{target}",
             "source": source, "target": target, "source_name": NAMES[source], "target_name": NAMES[target],
             "task": task, "seed": seed, "checkpoint_index": checkpoint_ids[source, task, seed]}
            for source, target in DIRECTIONS for task in TASKS for seed in SEEDS]
    plan = {"schema_version": 1, "representation": "phikonv2", "graph": "anatomical", "seeds": list(SEEDS),
            "tasks": list(TASKS), "evaluations": len(jobs), "unique_checkpoints": len(checkpoints),
            "unique_target_images": len(images), "jobs": jobs, "gpus": GPU_IDS,
            "snapshot": relative(snapshot), "extractor_identity": shared_identity, "batch_size": 16,
            "raw_cls_method": "Direct frozen Phikon-v2 extraction from each distinct target image, without fitted feature standardization",
            "prediction_transform": "Apply the selected source model's unchanged training mean and standard deviation",
            "normalization_refitted": False, "task_training_performed": False, "calibration_fitted": False,
            "source_checkpoint_selection": "Source validation balanced accuracy, unchanged before target evaluation",
            "own_domain_probability_absolute_tolerance": PROBABILITY_ATOL}
    record(output / "inputs/cohorts.json", cohorts)
    record(output / "inputs/images.json", images)
    record(output / "inputs/checkpoints.json", checkpoints)
    record(output / "provenance/input_sha256.json", hashes)
    record(output / "plan.json", plan)
    return hashes


def probabilities(model, vectors, task, device, batch_size=16):
    import torch
    from ki67.tasks import CLASS_TASKS
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(vectors), batch_size):
            x = torch.as_tensor(vectors[start:start + batch_size], dtype=torch.float32, device=device)
            pieces.append(model(x)[:, :len(CLASS_TASKS[task])].softmax(1).cpu().numpy())
    return np.concatenate(pieces)


def metrics(truth, probability):
    from ki67.evaluation import classification_metrics
    result = classification_metrics(truth, probability)
    for item in result["per_class"]:
        for key in ("precision", "recall", "f1"):
            if item[key] is None:
                item[key] = 0.0
    return result


def encode_worker(output, gpu, device):
    from ki67 import encoders
    import torch
    plan, images = load(output / "plan.json"), load(output / "inputs/images.json")
    indices = np.array_split(np.arange(len(images)), len(plan["gpus"]))[plan["gpus"].index(gpu)]
    selected = [images[int(index)] for index in indices]
    identity = plan["extractor_identity"]
    started = time.perf_counter()
    model = encoders.frozen_model(REPO / plan["snapshot"], device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("The image extractor must be frozen")
    raw = encoders.encode_cls(model, [REPO / row["image"] for row in selected],
        image_mean=identity["preprocessing"]["image_mean"], image_std=identity["preprocessing"]["image_std"],
        dimension=1024, batch_size=16, device=device, image_sha256=[row["image_sha256"] for row in selected])
    target = output / "features/shards" / f"gpu-{gpu}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, indices=indices, raw_cls=raw)
    record(target.with_suffix(".json"), {"gpu": gpu, "images": len(selected), "elapsed_seconds": time.perf_counter() - started,
                                       "device": device, "raw_cls_sha256": sha256(target), "standardization_applied": False})
    del model
    torch.cuda.empty_cache()


def evaluate_worker(output, gpu, device):
    import torch
    from ki67 import features, data
    from ki67.training import make_model
    from ki67.tasks import CLASS_TASKS
    plan, cohorts = load(output / "plan.json"), load(output / "inputs/cohorts.json")
    checkpoints = load(output / "inputs/checkpoints.json")
    raw = np.load(output / "features/raw_cls.npy", mmap_mode="r", allow_pickle=False)
    loaded, vectors = {}, {}
    worker_index = plan["gpus"].index(gpu)
    worker_count = len(plan["gpus"])

    def get_model(index):
        if index not in loaded:
            entry = checkpoints[index]
            if sha256(archive_path(entry["checkpoint"])) != entry["checkpoint_sha256"]:
                raise ValueError("A selected checkpoint changed after preparation")
            saved = torch.load(archive_path(entry["checkpoint"]), map_location="cpu", weights_only=False)
            model = make_model(archive_path(entry["source_output"]), "anatomical", saved["spec"]["training"], device)
            if saved["gain"].shape != model.gain.shape or not torch.isfinite(saved["gain"]).all():
                raise ValueError("Invalid saved synaptic gains")
            with torch.no_grad():
                model.gain.copy_(saved["gain"].to(device))
            model.eval()
            loaded[index] = model
        return loaded[index]

    def get_vectors(index, target):
        entry = checkpoints[index]
        key = (entry["source_scaler_mean"], entry["source_scaler_std"], target)
        if key not in vectors:
            center = np.load(archive_path(key[0]), allow_pickle=False)
            scale = np.load(archive_path(key[1]), allow_pickle=False)
            values = np.asarray(raw[cohorts[target]["raw_cls_indices"]])
            vectors[key] = features.standardize(values, center, scale)
        return vectors[key]

    for index in range(worker_index, len(checkpoints), worker_count):
        entry = checkpoints[index]
        cohort = cohorts[entry["source_cohort"]]
        predicted = probabilities(get_model(index), get_vectors(index, entry["source_cohort"]), entry["task"], device)
        with np.load(archive_path(entry["source_selected_predictions"]), allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_id"], np.asarray([row["sample_id"] for row in cohort["rows"]])):
                raise ValueError("Own-domain reference sample order mismatch")
            reference = saved["probabilities"]
            maximum_error = float(np.max(np.abs(predicted - reference)))
            disagreement = int(np.count_nonzero(predicted.argmax(1) != reference.argmax(1)))
        if maximum_error > PROBABILITY_ATOL or disagreement:
            raise ValueError(f"Own-domain direct-image reproduction failed for {entry['id']}: {maximum_error}, {disagreement}")
        record(output / "validation/own_domain" / f"{entry['id']}.json", {
            "checkpoint": entry["checkpoint"], "selected_epoch": entry["selected_epoch"],
            "reference_predictions": entry["source_selected_predictions"], "images": len(predicted),
            "maximum_probability_absolute_error": maximum_error, "class_disagreements": disagreement,
            "direct_image_cls": True, "source_training_scaler_unchanged": True, "gpu": gpu})
    for job in plan["jobs"][worker_index::worker_count]:
        started = time.perf_counter()
        index, target = job["checkpoint_index"], cohorts[job["target"]]
        entry = checkpoints[index]
        predicted = probabilities(get_model(index), get_vectors(index, job["target"]), job["task"], device)
        truth = data.targets(target["rows"], job["task"])
        path = output / "predictions" / job["direction"] / job["task"] / f"seed-{job['seed']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, y_true=truth, probabilities=predicted, y_pred=predicted.argmax(1),
            sample_id=np.asarray([row["sample_id"] for row in target["rows"]]),
            image_sha256=np.asarray([row["image_sha256"] for row in target["rows"]]),
            patient_id=np.asarray([row["patient_id"] or "" for row in target["rows"]]),
            class_names=np.asarray(CLASS_TASKS[job["task"]]), selected_epoch=np.asarray(entry["selected_epoch"]),
            source_checkpoint_sha256=np.asarray(entry["checkpoint_sha256"]))
        record(path.with_suffix(".json"), {**job, "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"], "selected_epoch": entry["selected_epoch"],
            "source_scaler_mean": entry["source_scaler_mean"], "source_scaler_std": entry["source_scaler_std"],
            "predictions": relative(path), "predictions_sha256": sha256(path), "n_images": len(truth),
            "metrics": metrics(truth, predicted), "gpu": gpu, "elapsed_seconds": time.perf_counter() - started,
            "normalization_refitted": False, "task_training_performed": False, "calibration_fitted": False})
    if plan.get("validate_raw_image_api", True):
        for index in range(worker_index, 3, worker_count):
            validate_prediction_api(output, index, device, checkpoints, cohorts)


def validate_prediction_api(output, gpu, device, checkpoints, cohorts):
    from ki67.prediction import predict_images
    source, target_name = [("bcdata", "shid"), ("shid", "bcdata"), ("shid_patient_disjoint", "bcdata")][gpu]
    entry = next(row for row in checkpoints if row["source_cohort"] == source and row["task"] == "multiclass" and row["seed"] == 1729)
    rows = cohorts[target_name]["rows"]
    indices = [next(index for index, row in enumerate(rows) if row["multiclass"] == label) for label in range(3)]
    from ki67.data import resolve_image_path
    images = [resolve_image_path(rows[index]) for index in indices]
    source_output = archive_path(entry["source_output"])
    if source == "shid_patient_disjoint":
        view = output / "validation/prediction-view"
        for name in ("graphs/anatomical.npz", "graphs/anatomical.json", "features/shid/phikonv2/metadata.json",
                     "features/shid/phikonv2/mean.npy", "features/shid/phikonv2/std.npy"):
            destination = view / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_output / name, destination)
        cache = view / "cache/huggingface"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.symlink_to(os.path.relpath(REPO / ARCHIVE / "cache/huggingface", cache.parent), target_is_directory=True)
        source_output = view
    destination = output / "validation/predict_images" / f"{source}.npz"
    predict_images(archive_path(entry["checkpoint"]), images, destination, source_output, device=device)
    reference_path = output / "predictions" / f"{source}_to_{target_name}" / "multiclass/seed-1729.npz"
    deadline = time.monotonic() + 120
    while not reference_path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Reference cross-dataset prediction was not produced")
        time.sleep(1)
    with np.load(destination, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as expected:
        maximum = float(np.max(np.abs(actual["probabilities"] - expected["probabilities"][indices])))
        disagreements = int(np.count_nonzero(actual["y_pred"] != expected["y_pred"][indices]))
    if maximum > PROBABILITY_ATOL or disagreements:
        raise ValueError("Direct predict_images and cached raw-CLS predictions differ")
    record(destination.with_name(f"{source}-comparison.json"), {
        "checkpoint": entry["checkpoint"], "target_indices": indices, "one_image_per_true_class": True,
        "maximum_probability_absolute_error": maximum, "class_disagreements": disagreements,
        "existing_predict_images_api_verified": True})


def worker(output, phase, gpu):
    from flybrain.launcher import configure_runtime, reserve_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    configure_runtime()
    lock = reserve_gpu(gpu)
    try:
        import torch
        torch.set_num_threads(4)
        device = os.environ["FLYBRAIN_DEVICE"]
        started = time.perf_counter()
        if phase == "encode":
            encode_worker(output, gpu, device)
        else:
            evaluate_worker(output, gpu, device)
        record(output / "timings/workers" / f"{phase}-gpu-{gpu}.json", {
            "phase": phase, "gpu": gpu, "elapsed_seconds": time.perf_counter() - started,
            "finished_at": timestamp()})
    finally:
        lock.close()


def phase(output, name):
    started, first = time.perf_counter(), timestamp()
    env = worker_environment(output)
    children = []
    for gpu in load(output / "plan.json")["gpus"]:
        log = output / "logs" / f"{name}-gpu-{gpu}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            child = subprocess.Popen([sys.executable, "-B", "-m", "ki67.run_cross_dataset", "--output", relative(output),
                "--worker", name, "--gpu", str(gpu)], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        children.append((child, log))
    while any(child.poll() is None for child, _ in children):
        failures = [(child.returncode, relative(log)) for child, log in children if child.poll() not in (None, 0)]
        if failures:
            raise RuntimeError(f"Worker failed: {failures}")
        completed = len(list((output / "predictions").glob("*/*/seed-*.json")))
        status = {"phase": name, "timestamp": timestamp(), "workers_alive": sum(child.poll() is None for child, _ in children),
                  "completed_cross_evaluations": completed, "expected_cross_evaluations": 48,
                  "elapsed_wall_seconds": time.perf_counter() - started}
        record(output / "progress.json", status)
        print(json.dumps(status), flush=True)
        time.sleep(10)
    failures = [(child.returncode, relative(log)) for child, log in children if child.returncode]
    if failures:
        raise RuntimeError(f"Worker failed: {failures}")
    record(output / "timings" / f"{name}.json", {"started_at": first, "finished_at": timestamp(),
           "elapsed_wall_seconds": time.perf_counter() - started})


def combine_raw(output):
    images = load(output / "inputs/images.json")
    raw, seen = np.empty((len(images), 1024), dtype=np.float32), set()
    for gpu in load(output / "plan.json")["gpus"]:
        with np.load(output / "features/shards" / f"gpu-{gpu}.npz", allow_pickle=False) as shard:
            indices = shard["indices"]
            if any(int(index) in seen for index in indices):
                raise ValueError("Duplicate raw feature assignment")
            raw[indices] = shard["raw_cls"]
            seen.update(map(int, indices))
    if seen != set(range(len(images))) or not np.isfinite(raw).all():
        raise ValueError("Raw feature coverage is incomplete or nonfinite")
    np.save(output / "features/raw_cls.npy", raw)
    record(output / "features/metadata.json", {"images": len(images), "dimension": 1024, "dtype": "float32",
        "image_sha256": [row["image_sha256"] for row in images], "standardization_applied": False,
        "source": "Direct frozen Phikon-v2 encoding of target images", "sha256": sha256(output / "features/raw_cls.npy")})


def reports(output, destination=None):
    plan = load(output / "plan.json")
    collected, summary, per_class = [], [], []
    for job in plan["jobs"]:
        path = output / "predictions" / job["direction"] / job["task"] / f"seed-{job['seed']}.json"
        row = load(path)
        predictions = path.with_suffix(".npz")
        if sha256(predictions) != row["predictions_sha256"]:
            raise ValueError("Prediction archive checksum mismatch")
        with np.load(predictions, allow_pickle=False) as saved:
            recovered = metrics(saved["y_true"], saved["probabilities"])
            if not np.array_equal(saved["y_pred"], saved["probabilities"].argmax(1)):
                raise ValueError("Saved categories do not match probability columns")
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", "roc_auc", "average_precision"):
            if not np.isclose(recovered[metric], row["metrics"][metric], atol=1e-12, rtol=1e-12):
                raise ValueError(f"Saved predictions disagree with {metric}")
        row["metrics"] = recovered
        collected.append(row)
        for item in row["metrics"]["per_class"]:
            per_class.append({"direction": job["direction"], "source": job["source_name"], "target": job["target_name"],
                              "task": job["task"], "seed": job["seed"], **item})
    for source, target in DIRECTIONS:
        for task in TASKS:
            selected = [row for row in collected if row["source"] == source and row["target"] == target and row["task"] == task]
            if len(selected) != 3:
                raise ValueError("A direction/task does not contain all three seeds")
            row = {"direction": f"{source}_to_{target}", "source": NAMES[source], "target": NAMES[target],
                   "task": task, "n_test_images": selected[0]["n_images"], "seeds": 3}
            for metric in ("balanced_accuracy", "accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc", "average_precision", "loss"):
                values = np.asarray([item["metrics"][metric] for item in selected], dtype=np.float64)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1))
            summary.append(row)
    destination = destination or output / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("cross_dataset_summary.csv", summary), ("cross_dataset_per_class.csv", per_class)):
        with (destination / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    record(destination / "cross_dataset_results.json", {"summary": summary, "evaluations": collected,
        "metric_units": "Fractions except loss and image counts", "aggregation": "Mean and sample standard deviation across three seeds, without ensembling"})
    text = ["# Cross-dataset anatomical BRAP-Net classification", "",
            "Balanced accuracy (%) is reported as mean ± sample standard deviation across three source-selected models.", "",
            "| Training dataset | Test dataset | Three classes | PI 5% | PI 10% | PI 30% |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for source, target in DIRECTIONS:
        cells = [NAMES[source], NAMES[target]]
        for task in TASKS:
            row = next(item for item in summary if item["source"] == NAMES[source] and item["target"] == NAMES[target] and item["task"] == task)
            cells.append(f"{100*row['balanced_accuracy_mean']:.2f} ± {100*row['balanced_accuracy_sd']:.2f}")
        text.append("| " + " | ".join(cells) + " |")
    (destination / "cross_dataset_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    global GPU_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=SAVED_EVALUATIONS, help="Saved cross-dataset evaluation directory")
    parser.add_argument("--mode", choices=("report", "replay", "images"), default="report",
                        help="report: recover metrics from NPZ; replay: infer again from saved raw CLS; images: encode local images again")
    parser.add_argument("--gpus", help="Explicit comma-separated physical GPU indices for replay/images")
    parser.add_argument("--worker", choices=("encode", "evaluate"))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    output = resolve_relative(args.output)
    if not output.is_relative_to(REPO / "6 Ki67/6 Workspace"):
        raise ValueError("Use a Ki67 Workspace output")
    if args.worker:
        if args.gpu is None:
            parser.error("Workers require an explicit GPU")
        worker(output, args.worker, args.gpu)
        return 0
    if output.exists():
        raise FileExistsError("The evaluation output directory must be new")
    source = resolve_relative(args.source)
    if args.mode == "report":
        reports(source, output)
        print(json.dumps({"status": "complete", "evaluations": 48, "output": relative(output)}))
        return 0
    if not args.gpus:
        parser.error("Replay and image extraction require explicit --gpus")
    GPU_IDS = [int(value) for value in args.gpus.split(",")]
    if not GPU_IDS or len(set(GPU_IDS)) != len(GPU_IDS) or min(GPU_IDS) < 0:
        parser.error("GPU indices must be nonnegative and unique")
    if args.mode == "replay":
        output.mkdir(parents=True)
        for name in ("inputs", "features"):
            shutil.copytree(source / name, output / name)
        plan = load(source / "plan.json")
        plan.update(gpus=GPU_IDS, validate_raw_image_api=False)
        record(output / "plan.json", plan)
        phase(output, "evaluate")
        reports(output)
        print(json.dumps({"status": "complete", "evaluations": 48, "raw_images_required": False}))
        return 0
    started, first = time.perf_counter(), timestamp()
    output.mkdir(parents=True)
    os.chdir(REPO)
    for name in ("KI67_SHID_ROOT", "KI67_BCDATA_ROOT"):
        os.environ.pop(name, None)
    hashes = prepare(output)
    sources = list((REPO / "6 Ki67/2 Source").glob("*.py")) + list((REPO / "3 Connectome/flybrain").glob("*.py"))
    sources += [Path(__file__), Path(__file__).with_name("run_shid_multiclass.py")]
    source_hashes = {}
    for source in sources:
        name = relative(source)
        destination = output / "provenance/source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[name] = sha256(source)
    record(output / "provenance/source_sha256.json", source_hashes)
    record(output / "provenance/runtime.json", {"python": sys.version.split()[0], "gpus": GPU_IDS,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "Pillow", "scikit-learn")}})
    print("Extracting raw Phikon-v2 descriptors for 1595 distinct target images", flush=True)
    phase(output, "encode")
    combine_raw(output)
    print("Evaluating the 48 source-selected anatomical models on their target cohorts", flush=True)
    phase(output, "evaluate")
    reports(output)
    for path, expected in {**hashes, **source_hashes}.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Source artifact changed during evaluation: {path}")
    if len(list((output / "validation/own_domain").glob("*.json"))) != 36:
        raise ValueError("Incomplete own-domain validation")
    record(output / "timings/complete.json", {"started_at": first, "finished_at": timestamp(),
        "elapsed_wall_seconds": time.perf_counter() - started, "cross_evaluations": 48,
        "source_checkpoints": 36, "raw_target_images": 1595, "inputs_and_source_code_unchanged": True,
        "normalization_refitted": False, "task_training_performed": False})
    print("All 48 cross-dataset evaluations completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
