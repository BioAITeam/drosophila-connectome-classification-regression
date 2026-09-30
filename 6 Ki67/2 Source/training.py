"""Image-level training, all-split epoch evaluation, and one validation checkpoint."""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from .common import CLASS_TASKS, atomic_npz, atomic_torch, digest, now, portable_metadata, portable_path, read_json, sha256, write_json
from . import data, features, graphs
from .sequence import SequenceConnectome
from .evaluation import classification_metrics, regression_report, select_score, metrics_by_dataset

def run_path(output, job):
    return Path(output) / "runs" / job["dataset"] / job["representation"] / job["task"] / job["graph"] / f"seed-{job['seed']}"

def make_model(output, graph_name, training, device):
    keys = ["dt", "leak", "input_amplitude", "temperature", "dwell_steps", "checkpoint_blocks"]
    model = SequenceConnectome(graphs.load_graph(output, graph_name),
                               **{k: training[k] for k in keys}).to(device)
    if [n for n, p in model.named_parameters() if p.requires_grad] != ["gain"]:
        raise AssertionError("Only the connectome gain tensor may be learned.")
    return model

class Inputs:
    def __init__(self, output, dataset, representation, task):
        self.dataset = dataset
        self.rows = {s: data.load_rows(output, dataset, s) for s in ["train", "val", "test"]}
        self.y = {s: np.asarray(data.targets(rows, task)) for s, rows in self.rows.items()}
        self.arrays = {} if representation == "rgb" else {
            s: features.get_features(output, dataset, representation, s) for s in self.rows}
        self.task, self.representation = task, representation
        self.scale = 1.0 if task in CLASS_TASKS else (100.0 if task == "pi" else max(1.0, float(self.y["train"].max())))
    def x(self, split, indices, device):
        if self.representation == "rgb":
            value = features.load_rgb([self.rows[split][int(i)] for i in indices])
        else:
            value = np.asarray(self.arrays[split][indices], dtype=np.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate(model, inputs, split, microbatch, device, progress=None):
    model.eval()
    started = time.monotonic()
    values = []
    n = len(inputs.rows[split])
    for start in range(0, n, microbatch):
        if progress:
            progress(phase="evaluation", split=split, batch=start // microbatch, batches=(n+microbatch-1)//microbatch)
        x = inputs.x(split, np.arange(start, min(n, start + microbatch)), device)
        if inputs.task in CLASS_TASKS:
            prediction = model(x)[:, :len(CLASS_TASKS[inputs.task])].softmax(1)
        else:
            prediction = model.regression(x) * inputs.scale
        values.append(prediction.cpu().numpy())
    predictions = np.concatenate(values)
    y = inputs.y[split]
    if inputs.task in CLASS_TASKS:
        metrics = classification_metrics(y, predictions)
        payload = dict(y_true=y, probabilities=predictions, y_pred=predictions.argmax(1),
                       class_names=np.asarray(CLASS_TASKS[inputs.task]))
        names = np.asarray(CLASS_TASKS[inputs.task])
        payload["y_true_label"] = names[y.astype(np.int64)]
        payload["y_pred_label"] = names[predictions.argmax(1)]
    else:
        metrics = regression_report(y, predictions, inputs.task)
        clipped = np.clip(predictions, 0, 100) if inputs.task == "pi" else np.maximum(predictions, 0)
        payload = dict(y_true=y, y_pred=predictions, y_pred_clipped=clipped)
        metrics["scaled_mse_loss"] = float(np.mean(((predictions-y)/inputs.scale)**2))
        metrics["target_above_representable_max_fraction"] = float(np.mean(y > 2 * inputs.scale))
    payload["sample_id"] = np.asarray([r["sample_id"] for r in inputs.rows[split]])
    origins = np.asarray([r.get("dataset", getattr(inputs, "dataset", "unspecified"))
                          for r in inputs.rows[split]])
    payload["source_dataset"] = origins
    if len(np.unique(origins)) > 1:
        metrics["by_dataset"] = metrics_by_dataset(y, predictions, inputs.task, origins)
        if inputs.task not in CLASS_TASKS:
            for origin, subset_metrics in metrics["by_dataset"].items():
                subset_metrics["scaled_mse_loss"] = subset_metrics["mse"] / inputs.scale**2
                subset_metrics["target_above_representable_max_fraction"] = float(np.mean(y[origins == origin] > 2 * inputs.scale))
        metrics["dataset_scope"] = "pooled observations; one shared model for all source datasets"
    metrics["evaluation_seconds"] = time.monotonic() - started
    metrics["n_samples"] = n
    return metrics, payload

def _rng_state(generator):
    return {"numpy": generator.bit_generator.state, "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def _restore_rng(state, generator):
    generator.bit_generator.state = state["numpy"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])

def _completed_run(directory, job):
    required = ("config.json", "best.pt", "best.json", "history.json")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("Completed run is missing its selected checkpoint or experiment records.")
    complete = read_json(directory / "complete.json")
    selected = read_json(directory / "best.json")
    identity = ("dataset", "representation", "task", "graph", "seed")
    if any(document.get(key) != job[key] for document in (complete, selected) for key in identity):
        raise ValueError("Completed run identity differs from the requested experiment.")
    prediction = directory / selected["prediction_path"]
    if (not prediction.resolve().is_relative_to(directory.resolve()) or not prediction.is_file()
            or complete.get("best_epoch") != selected.get("epoch")):
        raise ValueError("Completed run is missing its selected predictions or has inconsistent metadata.")
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    spec = read_json(directory / "config.json")
    if (saved.get("spec") != spec or saved.get("spec_digest") != digest(spec)
            or saved.get("epoch") != selected.get("epoch")
            or saved.get("best_metadata") != selected
            or saved.get("score") != selected.get("score")
            or saved.get("target_scale") != selected.get("target_scale")):
        raise ValueError("Completed run checkpoint provenance differs from its experiment records.")
    gain = saved.get("gain")
    if (not isinstance(gain, torch.Tensor) or gain.dtype != torch.float32
            or gain.ndim != 1 or gain.numel() == 0 or not torch.isfinite(gain).all()):
        raise ValueError("Completed run checkpoint gains are invalid.")
    return complete

def train_job(config, output, job, device="cpu", progress=None):
    from .common import writable_location
    directory = writable_location(run_path(output, job))
    directory.mkdir(parents=True, exist_ok=True)
    training = config["training"].copy()
    profile_path = Path(output) / "status/profiles" / f"{job['dataset']}-{job['representation']}-{job['graph']}.json"
    if profile_path.exists():
        training["microbatch_size"] = read_json(profile_path)["microbatch_size"]
    spec = {**job, "training": training, "manifest_sha256": sha256(Path(output)/"manifests"/f"{job['dataset']}.json"),
            "graph_sha256": sha256(Path(output)/"graphs"/f"{job['graph']}.npz")}
    if job["representation"] != "rgb":
        spec["feature_metadata_sha256"] = sha256(Path(output)/"features"/job["dataset"]/job["representation"]/"metadata.json")
    if config.get("joint"):
        spec["joint_training"] = config["joint"]
        spec["architecture_family"] = "BRAP-Net"
        spec["validation_scope"] = "pooled_validation"
        spec["test_scopes"] = ["pooled", *config["joint"]["datasets"]]
    spec = portable_metadata(spec)
    if (directory/"config.json").exists():
        stored_spec = read_json(directory/"config.json")
        if (stored_spec.get("architecture_family") == "ARPM-Net"
                and spec.get("architecture_family") == "BRAP-Net"
                and {**stored_spec, "architecture_family": "BRAP-Net"} == spec):
            spec = stored_spec
        elif stored_spec != spec:
            raise ValueError("Refusing to mix different configurations in an existing run.")
    if (directory / "complete.json").exists():
        return _completed_run(directory, job)
    write_json(directory/"config.json", spec)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    rng = np.random.default_rng(job["seed"] + 103)
    inputs = Inputs(output, job["dataset"], job["representation"], job["task"])
    model = make_model(output, job["graph"], training, device)
    optimizer = torch.optim.Adam([model.gain], lr=training["learning_rate"])
    history = read_json(directory/"history.json") if (directory/"history.json").exists() else []
    prediction_root = directory/"predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    existing_attempts = [int(p.name.split("-", 1)[1]) for p in prediction_root.glob("attempt-*")
                         if p.is_dir() and p.name.split("-", 1)[1].isdigit()]
    attempt = max([int(r.get("attempt", 0)) for r in history] + existing_attempts, default=-1) + 1
    while True:
        prediction_dir = prediction_root/f"attempt-{attempt:03d}"
        try:
            prediction_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            attempt += 1
    checkpoint = directory/"best.pt"
    epoch_start, best_epoch, stale, best_score = 0, 0, 0, -float("inf")
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["spec_digest"] != digest(spec):
            raise ValueError("Checkpoint provenance does not match this run.")
        with torch.no_grad():
            model.gain.copy_(saved["gain"])
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], rng)
        epoch_start = best_epoch = saved["epoch"]
        best_score = saved["score"]
        # Recover metadata even if a process stopped after replacing best.pt.
        write_json(directory/"best.json", saved["best_metadata"])
    is_class = job["task"] in CLASS_TASKS
    epochs = training["classification_epochs"] if is_class else training["regression_epochs"]
    micro = int(training["microbatch_size"])
    effective = int(training["batch_size"])
    if micro < 1 or effective < micro:
        raise ValueError("Require 1 <= microbatch_size <= effective batch size.")
    started = time.monotonic()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    def update(**fields):
        if progress:
            progress(**fields)
    model.progress_callback = lambda info: update(**info)
    for epoch in range(epoch_start, epochs+1):
        update(epoch=epoch, phase="epoch_start", attempt=attempt)
        epoch_started = time.monotonic()
        loss_sum, seen, gradient_sum, steps = 0.0, 0, 0.0, 0
        # Resume re-evaluates the restored checkpoint; later epochs retrain from it.
        if epoch > epoch_start:
            model.train()
            order = rng.permutation(len(inputs.y["train"]))
            for batch_start in range(0, len(order), effective):
                batch_indices = order[batch_start:batch_start+effective]
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(batch_indices), micro):
                    indices = batch_indices[micro_start:micro_start+micro]
                    update(phase="training", split="train", batch=batch_start//effective,
                           batches=(len(order)+effective-1)//effective, microbatch=micro_start//micro)
                    x = inputs.x("train", indices, device)
                    y = torch.as_tensor(inputs.y["train"][indices], device=device,
                                        dtype=torch.long if is_class else torch.float32)
                    if is_class:
                        loss = F.cross_entropy(model(x)[:, :len(CLASS_TASKS[job['task']])], y)
                    else:
                        loss = F.mse_loss(model.regression(x), y / inputs.scale)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss.")
                    (loss * (len(indices)/len(batch_indices))).backward()
                    loss_sum += float(loss.detach()) * len(indices)
                    seen += len(indices)
                grad_norm = torch.nn.utils.clip_grad_norm_([model.gain], training["gradient_clip"])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gain gradients.")
                gradient_sum += float(grad_norm)
                steps += 1
                optimizer.step()
        train_seconds = time.monotonic() - epoch_started
        row = {"epoch": epoch, "attempt": attempt, "timestamp": now(), "training_seconds": train_seconds}
        for split in ["train", "val", "test"]:
            metrics, payload = evaluate(model, inputs, split, micro, device, update)
            row[split] = metrics
            if split == "test":
                prediction_path = prediction_dir/f"epoch-{epoch:04d}.npz"
                atomic_npz(prediction_path, **payload, epoch=np.asarray(epoch), attempt=np.asarray(attempt),
                           run_id=np.asarray(job.get("id", "/".join(str(job[k]) for k in
                                      ["dataset", "representation", "task", "graph", "seed"]))),
                           split=np.asarray("test"))
        row["prediction_path"] = str(prediction_path.relative_to(directory))
        row["elapsed_seconds"] = time.monotonic() - started
        row["epoch_seconds"] = time.monotonic() - epoch_started
        row["optimization_loss"] = loss_sum/seen if seen else None
        row["gradient_norm_mean"] = gradient_sum/steps if steps else None
        row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        score = select_score(row["val"], job["task"])
        if score is None or not np.isfinite(score):
            raise FloatingPointError("Validation selection metric is undefined.")
        improved = score > best_score + training["min_delta"]
        if improved:
            best_epoch, best_score, stale = epoch, score, 0
            best_metadata = {**job, "epoch": epoch, "attempt": attempt, "score": score,
                             "criterion": "validation_balanced_accuracy" if is_class else "validation_mae_raw",
                             "prediction_path": row["prediction_path"], "validation_metrics": row["val"],
                             "test_metrics": row["test"], "target_scale": inputs.scale,
                             "timestamp": now()}
            if config.get("joint"):
                best_metadata.update(architecture_family=spec["architecture_family"], selection_scope="pooled_validation",
                                     evaluation_scopes=["pooled", *config["joint"]["datasets"]])
            atomic_torch(checkpoint, {"gain": model.gain.detach().cpu(), "optimizer": optimizer.state_dict(),
                         "rng": _rng_state(rng), "epoch": epoch, "score": score,
                         "spec": spec, "spec_digest": digest(spec), "target_scale": inputs.scale,
                         "best_metadata": best_metadata})
            write_json(directory/"best.json", best_metadata)
        elif epoch > epoch_start:
            stale += 1
        row.update(best_epoch=best_epoch, selected=improved, stale_epochs=stale)
        history.append(row)
        write_json(directory/"history.json", history)
        update(phase="epoch_complete", epoch=epoch, best_epoch=best_epoch, score=score,
               epoch_seconds=row["epoch_seconds"])
        print(json.dumps({"job": job, "epoch": epoch, "selection_score": score, "best_epoch": best_epoch,
                          "test": {k: row["test"].get(k) for k in (["balanced_accuracy"] if is_class else ["mae", "rmse", "r2"])}}), flush=True)
        if not is_class and epoch >= training["regression_min_epochs"] and stale >= training["regression_patience"]:
            break
    result = {**job, "finished_at": now(), "last_epoch": epoch, "best_epoch": best_epoch,
              "attempt": attempt, "elapsed_seconds": time.monotonic()-started,
              "checkpoint": portable_path(checkpoint), "figures": []}
    write_json(directory/"complete.json", result)
    return result

def profile(config, output, dataset, representation, graph, device, progress=None):
    """Measure complete native-image forward/backward and all-split inference."""
    training = config["training"].copy()
    inputs = Inputs(output, dataset, representation, "multiclass")
    for micro in sorted({training["microbatch_size"], 8, 4, 2, 1}, reverse=True):
        if micro > min(training["batch_size"], training["microbatch_size"]):
            continue
        try:
            model = make_model(output, graph, training, device)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            x = inputs.x("train", np.arange(micro), device)
            y = torch.as_tensor(inputs.y["train"][:micro], dtype=torch.long, device=device)
            t = time.monotonic()
            if progress:
                model.progress_callback = lambda info: progress(**info)
            F.cross_entropy(model(x), y).backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_backward = time.monotonic()-t
            if not torch.isfinite(model.gain.grad).all() or not bool((model.gain.grad != 0).any()):
                raise FloatingPointError("Profile requires finite, nonzero gain gradients.")
            model.eval()
            infer_times = {}
            with torch.no_grad():
                for split in ["train", "val", "test"]:
                    t = time.monotonic()
                    model(inputs.x(split, np.arange(min(micro, len(inputs.y[split]))), device))
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    infer_times[split] = (time.monotonic()-t)/min(micro, len(inputs.y[split]))
            result = {"dataset": dataset, "representation": representation, "graph": graph,
                      "device": device, "gpu_record": json.loads(os.environ.get("FLYBRAIN_GPU_RECORD", "null")),
                      "timestamp": now(), "microbatch_size": micro, "forward_backward_seconds": forward_backward,
                      "inference_seconds_per_image": infer_times,
                      "estimated_epoch_seconds": forward_backward/micro*len(inputs.y["train"])+sum(infer_times[s]*len(inputs.y[s]) for s in infer_times),
                      "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
                      "n_features": int(x.shape[1]), "sequence": model.sequence_stats(x.shape[1]),
                      "nonzero_gradient_fraction": float((model.gain.grad != 0).float().mean())}
            write_json(Path(output)/"status/profiles"/f"{dataset}-{representation}-{graph}.json", result)
            return result
        except torch.cuda.OutOfMemoryError:
            if micro == 1:
                raise
            del model
            if "x" in locals():
                del x
            torch.cuda.empty_cache()
    raise RuntimeError("No supported profile microbatch size.")
