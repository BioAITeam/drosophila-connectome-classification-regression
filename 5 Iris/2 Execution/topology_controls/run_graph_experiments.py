"""One CPU-only graph job with validation selection and one retained checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import tempfile
import time

from common import HERE, REPO, SOURCE, load_json, load_partition, relative, save_json, sha256

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(REPO / "3 Connectome"))
from flybrain.metrics import classification_metrics
from flybrain.model import SparseConnectome


MODEL = {"plastic_scope": "all", "dynamics": "rate", "steps": 24, "dt": 0.5,
         "leak": 1.0, "input_amplitude": 2.0, "temperature": 0.02}
BATCH_SIZE = 16
LEARNING_RATE = 0.03
GRADIENT_CLIP = 1.0
MIN_DELTA = 1e-8
EPOCHS = 120


def buffers_hash(model: SparseConnectome) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_buffers()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model: SparseConnectome, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Same batched logits, cross-entropy and metrics as flybrain.experiment."""
    model.eval()
    total, predictions, probabilities = 0.0, [], []
    for start in range(0, len(x), BATCH_SIZE):
        inputs = torch.as_tensor(x[start:start + BATCH_SIZE], dtype=torch.float32, device="cpu")
        labels = torch.as_tensor(y[start:start + len(inputs)], dtype=torch.long, device="cpu")
        logits = model(inputs)[:, :3]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite graph logits")
        total += float(F.cross_entropy(logits, labels, reduction="sum"))
        predictions.append(logits.argmax(1).cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    pred, probability = np.concatenate(predictions), np.concatenate(probabilities)
    metrics = classification_metrics(y, pred, n_classes=3)
    metrics["cross_entropy"] = total / len(y)
    return pred, probability, metrics


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".best.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".predictions.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checked_repo_path(value: str) -> Path:
    if Path(value).is_absolute():
        raise ValueError("Campaign manifests must contain campaign-relative paths")
    path = (HERE / value).resolve()
    path.relative_to(HERE.resolve())
    return path


def _checkpoint_payload(model: SparseConnectome, config: dict, hashes: dict,
                        selected_epoch: int, validation_score: float) -> dict:
    return {"schema_version": 1, "gain": model.gain.detach().cpu().clone(),
        "config": config, **hashes, "selected_epoch": selected_epoch,
        "validation_balanced_accuracy": validation_score, "trainable_parameter_count": 40755,
        "selection": "earliest maximum validation balanced accuracy; epoch zero eligible; min_delta=1e-8"}


def train(model: SparseConnectome, data: dict, config: dict, directory: Path,
          hashes: dict, run_start: float) -> dict:
    """Dedicated loop reproducing the supervised numerical training algorithm.

    This intentionally does not call experiment.train: that API also writes a
    last-parameters checkpoint. Only best.pt is retained by this campaign.
    """
    epochs = config["epochs"]
    rng = np.random.default_rng(config["optimization_seed"] + 103)
    _, _, initial_validation = evaluate(model, data["x_val"], data["y_val"])
    _, _, initial_training = evaluate(model, data["x_train"], data["y_train"])
    best_score = initial_validation["balanced_accuracy"]
    best_epoch = 0
    optimizer = torch.optim.Adam([model.gain], lr=LEARNING_RATE)
    start = time.monotonic()
    history = [{"epoch": 0, "training": initial_training, "validation": initial_validation,
                "elapsed_seconds": 0.0}]
    checkpoint = directory / "best.pt"
    _atomic_checkpoint(checkpoint, _checkpoint_payload(model, config, hashes, 0, best_score))
    save_json(directory / "history.json", history)
    save_json(directory / "progress.json", {
        "id": config["id"], "phase": "training", "epoch": 0, "epochs": epochs,
        "selected_epoch": 0, "best_validation_balanced_accuracy": best_score,
        "elapsed_seconds": time.monotonic() - run_start,
    })
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(data["x_train"]))
        losses, gradient_norms, gradient_fractions = [], [], []
        for start_index in range(0, len(order), BATCH_SIZE):
            indices = order[start_index:start_index + BATCH_SIZE]
            inputs = torch.as_tensor(data["x_train"][indices], dtype=torch.float32, device="cpu")
            labels = torch.as_tensor(data["y_train"][indices], dtype=torch.long, device="cpu")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)[:, :3]
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            gradient_fractions.append(float((model.gain.grad != 0).float().mean()))
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(
                [model.gain], GRADIENT_CLIP, error_if_nonfinite=True)))
            optimizer.step()
            losses.append((float(loss.detach()), len(indices)))
        _, _, validation = evaluate(model, data["x_val"], data["y_val"])
        _, _, training = evaluate(model, data["x_train"], data["y_train"])
        score = validation["balanced_accuracy"]
        history.append({
            "epoch": epoch, "training": training, "validation": validation,
            "training_loss": sum(loss * size for loss, size in losses) / sum(size for _, size in losses),
            "gradient_l2_before_clip_mean": float(np.mean(gradient_norms)),
            "nonzero_gradient_fraction_mean": float(np.mean(gradient_fractions)),
            "elapsed_seconds": time.monotonic() - start,
        })
        if score > best_score + MIN_DELTA:
            best_score, best_epoch = score, epoch
            _atomic_checkpoint(checkpoint, _checkpoint_payload(model, config, hashes, best_epoch, best_score))
        save_json(directory / "history.json", history)
        save_json(directory / "progress.json", {
            "id": config["id"], "phase": "training", "epoch": epoch, "epochs": epochs,
            "selected_epoch": best_epoch, "best_validation_balanced_accuracy": best_score,
            "training_balanced_accuracy": training["balanced_accuracy"],
            "validation_balanced_accuracy": score,
            "training_seconds": time.monotonic() - start,
            "elapsed_seconds": time.monotonic() - run_start,
        })
        print(f"{config['id']} epoch {epoch}/{epochs}: validation={score:.6f}, best={best_score:.6f}@{best_epoch}", flush=True)
    return {"history": history, "selected_epoch": best_epoch, "epochs_completed": epochs,
            "training_seconds": time.monotonic() - start}


def _load_job(job_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
        raise ValueError("Job ID must be a safe single path component")
    matches = [job for job in load_json(HERE / "jobs.json")["jobs"] if job["id"] == job_id]
    if len(matches) != 1 or matches[0].get("kind") != "graph":
        raise ValueError(f"Expected one graph job named {job_id!r}")
    return matches[0]


def run_job(job_id: str, smoke: bool = False) -> dict:
    start = time.monotonic()
    job = _load_job(job_id)
    directory = HERE / ("qa" if smoke else "runs") / job_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(job, directory, smoke, start)


def _run_locked(job: dict, directory: Path, smoke: bool, start: float) -> dict:
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = load_json(complete_path)
        result = load_json(directory / "result.json")
        if complete.get("result_sha256") != sha256(directory / "result.json"):
            raise RuntimeError("Existing completed result checksum mismatch")
        if result["checkpoint_sha256"] != sha256(directory / "best.pt"):
            raise RuntimeError("Existing completed checkpoint checksum mismatch")
        if result["predictions_sha256"] != sha256(directory / "predictions.npz"):
            raise RuntimeError("Existing completed predictions checksum mismatch")
        print(f"{job['id']}: already complete", flush=True)
        return result
    # A retried incomplete job starts from the original zero gains, never the
    # previous best checkpoint. Success markers are written only after audits.
    for filename in ("result.json", "failure.json"):
        (directory / filename).unlink(missing_ok=True)
    torch.manual_seed(int(job["optimization_seed"]))
    np.random.seed(int(job["optimization_seed"]))
    threads = int(os.environ.get("FLYBRAIN_THREADS", "1"))
    if threads < 1:
        raise ValueError("FLYBRAIN_THREADS must be positive")
    torch.set_num_threads(threads)
    graph_index_path = HERE / "graphs/index.json"
    candidates = [row for row in load_json(graph_index_path)["graphs"] if row["id"] == job["graph_id"]]
    if len(candidates) != 1:
        raise ValueError("Graph ID is missing or duplicated in graph manifest")
    graph_record = candidates[0]
    if graph_record["family"] != job["family"] or graph_record["seed"] != job["graph_seed"]:
        raise ValueError("Job graph family/seed does not match graph manifest")
    graph_path = _checked_repo_path(graph_record["path"])
    graph_digest = sha256(graph_path)
    if graph_digest != graph_record["sha256"]:
        raise RuntimeError("Graph checksum mismatch")
    with np.load(graph_path, allow_pickle=False) as archive:
        graph = {key: archive[key].copy() for key in archive.files}
    partition_path = HERE / "data" / f"split-{job['split_seed']}-fold-{job['fold']}.npz"
    partition_metadata_path = partition_path.with_suffix(".json")
    partition_digest = sha256(partition_path)
    partition_metadata = load_json(partition_metadata_path)
    data = load_partition(int(job["split_seed"]), int(job["fold"]))
    config = {
        **job, "schema_version": 1, "dataset": "iris", "algorithm": "supervised",
        "model": MODEL, "plastic_scope": "all", "device": "cpu", "torch_threads": threads,
        "epochs": 2 if smoke else EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "gradient_clip": GRADIENT_CLIP, "min_delta": MIN_DELTA,
        "select_checkpoint": "best_validation", "selection_metric": "balanced_accuracy",
        "selection_ties": "earliest epoch retained", "epoch_zero_eligible": True,
        "early_stopping": False, "software_validation_only": smoke,
        "graph_path": relative(graph_path), "partition_path": relative(partition_path),
        "partition_metadata_path": relative(partition_metadata_path),
        "run_directory": relative(directory), "trainable_parameter_count": 40755,
        "test_access": "initial and selected tests evaluated only after validation checkpoint selection",
    }
    model = SparseConnectome(graph, **MODEL).to("cpu")
    trainables = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainables != [("gain", 40755)] or model.num_edges != 40755 or model.gain.numel() != model.plastic_edge_indices.numel():
        raise RuntimeError(f"Unexpected trainable graph parameters: {trainables}")
    if torch.count_nonzero(model.gain).item() != 0:
        raise RuntimeError("Initial graph gains are not exactly zero")
    fixed_buffers = buffers_hash(model)
    hashes = {"graph_sha256": graph_digest, "partition_sha256": partition_digest,
              "fixed_buffers_sha256": fixed_buffers}
    save_json(directory / "config.json", config)
    save_json(directory / "graph.json", graph_record)
    save_json(directory / "partitions.json", {
        "metadata": partition_metadata, "metadata_path": relative(partition_metadata_path),
        "metadata_sha256": sha256(partition_metadata_path),
        "partition_path": relative(partition_path), "partition_sha256": partition_digest,
        "row_indices": {part: data[f"index_{part}"].tolist() for part in ("train", "val", "test")},
        "group_ids": {part: data["groups"][data[f"index_{part}"]].tolist() for part in ("train", "val", "test")},
    })
    save_json(directory / "environment.json", {
        "device": "cpu", "torch_threads": threads,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy", "scikit-learn")},
        "source_sha256": {relative(path): sha256(path) for path in (
            Path(__file__), SOURCE / "common.py", SOURCE / "prepare_partitions.py",
            REPO / "3 Connectome/flybrain/model.py", REPO / "3 Connectome/flybrain/metrics.py",
            REPO / "3 Connectome/flybrain/experiment.py")},
        "numerical_reference": "flybrain.experiment.train supervised path; no last-parameters output",
    })
    training = train(model, data, config, directory, hashes, start)
    if buffers_hash(model) != fixed_buffers:
        raise RuntimeError("Fixed model buffers changed during training")
    checkpoint_path = directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["selected_epoch"] != training["selected_epoch"]:
        raise RuntimeError("Selected checkpoint epoch mismatch")
    for key, value in hashes.items():
        if checkpoint[key] != value:
            raise RuntimeError(f"Selected checkpoint provenance mismatch: {key}")
    selected_gain = checkpoint["gain"].detach().clone()
    if selected_gain.shape != model.gain.shape or not torch.isfinite(selected_gain).all():
        raise RuntimeError("Invalid selected gain tensor")
    # Baseline test scoring is deferred until training/selection has finished.
    with torch.no_grad():
        model.gain.zero_()
    _, _, reset_validation = evaluate(model, data["x_val"], data["y_val"])
    if reset_validation != training["history"][0]["validation"]:
        raise RuntimeError("Zero-gain reset did not reproduce epoch-zero validation")
    initial_predicted, initial_probabilities, initial_test_metrics = evaluate(model, data["x_test"], data["y_test"])
    with torch.no_grad():
        model.gain.copy_(selected_gain)
    if not torch.equal(model.gain.detach(), selected_gain):
        raise RuntimeError("Selected gain restoration failed")
    _, _, selected_validation = evaluate(model, data["x_val"], data["y_val"])
    if selected_validation != training["history"][training["selected_epoch"]]["validation"]:
        raise RuntimeError("Restored checkpoint does not reproduce selected validation metrics")
    predicted, probabilities, test_metrics = evaluate(model, data["x_test"], data["y_test"])
    if buffers_hash(model) != fixed_buffers:
        raise RuntimeError("Fixed model buffers changed during checkpoint/reset evaluation")
    if sha256(graph_path) != graph_digest or sha256(partition_path) != partition_digest:
        raise RuntimeError("Graph or partition source changed during the job")
    if (directory / "last_parameters.pt").exists():
        raise RuntimeError("Only the selected best checkpoint may be retained")
    prediction_path = directory / "predictions.npz"
    _atomic_predictions(prediction_path, {
        "row_index": data["index_test"], "true": data["y_test"], "predicted": predicted,
        "probabilities": probabilities, "initial_predicted": initial_predicted,
        "initial_probabilities": initial_probabilities,
    })
    validation = {
        "fixed_buffers_unchanged": True, "zero_gain_reset_reproduces_initial_validation": True,
        "selected_gain_restored_exactly": True, "selected_validation_reproduced": True,
        "graph_and_partition_sources_unchanged": True,
        "only_selected_checkpoint_retained": True, "trainable_parameters": trainables,
        "trainable_parameter_count": 40755, "selected_gain_sha256": tensor_hash(selected_gain),
        "fixed_buffers_sha256": fixed_buffers,
    }
    save_json(directory / "validation.json", validation)
    result = {
        **job, "schema_version": 1, "software_validation_only": smoke,
        "run_directory": relative(directory), "test_metrics": test_metrics,
        "initial_test_metrics": initial_test_metrics,
        "selected_validation_metrics": selected_validation,
        "selected_epoch": training["selected_epoch"], "epochs_completed": training["epochs_completed"],
        "training_seconds": training["training_seconds"], "elapsed_seconds": time.monotonic() - start,
        **hashes, "checkpoint_sha256": sha256(checkpoint_path),
        "partition_metadata_sha256": sha256(partition_metadata_path),
        "predictions_sha256": sha256(prediction_path),
        "checkpoint_path": relative(checkpoint_path), "predictions_path": relative(prediction_path),
        "trainable_parameter_count": 40755,
    }
    save_json(directory / "result.json", result)
    save_json(directory / "progress.json", {
        "id": job["id"], "phase": "complete", "epoch": training["epochs_completed"],
        "epochs": config["epochs"], "selected_epoch": training["selected_epoch"],
        "training_seconds": training["training_seconds"], "elapsed_seconds": result["elapsed_seconds"],
    })
    save_json(complete_path, {
        "id": job["id"], "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "result_path": relative(directory / "result.json"), "result_sha256": sha256(directory / "result.json"),
        "checkpoint_sha256": result["checkpoint_sha256"], "software_validation_only": smoke,
    })
    print(f"{job['id']} complete: test balanced accuracy={test_metrics['balanced_accuracy']:.6f}; "
          f"selected epoch={training['selected_epoch']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--smoke", action="store_true", help="Two QA epochs under qa/, excluded from campaign results")
    args = parser.parse_args()
    run_job(args.job_id, smoke=args.smoke)


if __name__ == "__main__":
    main()
