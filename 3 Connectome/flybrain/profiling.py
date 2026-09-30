"""Measure sparse training cost without evaluating the held-out test set.

Profiling updates are discarded. Launch through run.sh to check GPU availability
before importing torch.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .datasets import prepare_dataset
from .experiment import file_hash, load_graph, make_model, write_json
from .paths import repo_path, writable_output


def estimate_training_cost(batch_seconds, validation_seconds, *, train_samples,
                           batch_size, epochs, plastic_edges, nodes, steps):
    """Estimate training time and tensor storage from measured operation times."""
    batches_per_epoch = math.ceil(train_samples / batch_size)
    epoch_seconds = batches_per_epoch * batch_seconds + validation_seconds
    return {
        "training_batches_per_epoch": batches_per_epoch,
        "epoch_seconds": epoch_seconds,
        "epochs": epochs,
        "training_and_validation_hours": epochs * epoch_seconds / 3600,
        "gain_float32_bytes": plastic_edges * 4,
        "gain_gradient_two_adam_moments_bytes": plastic_edges * 4 * 4,
        "model_pt_tensor_payload_bytes": plastic_edges * (4 + 4 + 8),
        "plastic_edges_npz_uncompressed_payload_bytes": plastic_edges * (8 + 8 + 4 + 4 + 4 + 4),
        "last_parameters_pt_tensor_payload_bytes": plastic_edges * 4,
        "single_rollout_state_sequence_bytes": nodes * batch_size * steps * 4,
        "note": "Timing excludes graph construction, controls, hashing, bootstrap, checkpoints and serialization. The final short batch is conservatively charged as a full batch. Artifact sizes are raw tensor payloads, excluding container overhead; NPZ compression can reduce size. State estimate counts only one float32 activity per step, not all autograd tensors.",
    }


def profile_training(graph_path, config, output_path, batches=2):
    """Profile supervised float32 training using only train/validation partitions.

    ``config`` has the regular experiment format. The GPU must have been
    reserved by the launcher, which sets FLYBRAIN_DEVICE/FLYBRAIN_GPU_RECORD.
    No checkpoint, test predictions, test scores or persistent dataset is saved.
    """
    config = copy.deepcopy(config)
    if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
        raise ValueError("batches must be a positive integer")
    if config.get("algorithm", "supervised") != "supervised":
        raise ValueError("This profile measures supervised Adam only")
    if config.get("frozen") or config.get("freeze_candidate"):
        raise ValueError("Use an explicit plastic_scope for training profiles")
    batch_size = int(config.get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    destination = writable_output(repo_path(output_path))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite profile: {destination}")
    device = os.environ.get("FLYBRAIN_DEVICE", "cpu")
    gpu_record = json.loads(os.environ.get("FLYBRAIN_GPU_RECORD", "null"))
    is_cuda = device.startswith("cuda")
    if is_cuda and not gpu_record:
        raise RuntimeError("CUDA profiling requires a GPU reserved by run.sh --device cuda")
    legacy_seed = int(config.get("seed", 0))
    seed = int(config.get("optimization_seed", legacy_seed))
    torch.set_num_threads(int(os.environ.get("FLYBRAIN_THREADS", "4")))
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.init()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    def timed(operation):
        sync()
        start = time.perf_counter()
        result = operation()
        sync()
        return result, time.perf_counter() - start

    started = time.perf_counter()
    (graph, provenance), load_seconds = timed(lambda: load_graph(
        graph_path, config.get("encoding", "visual"), config.get("rewire_seed")))
    model, construction_seconds = timed(lambda: make_model(graph, config, device))
    print(f"Profile: {model.num_nodes} neurons, {model.num_edges} edges; "
          f"steps={model.steps}, batch={batch_size}, device={device}", flush=True)
    prepared = prepare_dataset(config.get("dataset", "iris"),
                               seed=int(config.get("split_seed", legacy_seed)),
                               outer_fold=config.get("outer_fold"),
                               label_permutation_seed=config.get("label_permutation_seed"),
                               split_spec=config.get("split_spec"))
    task = prepared["metadata"]["task"]
    n_classes = prepared["metadata"].get("n_classes") or 3
    train_x, train_y = prepared["x_train"], prepared["y_train"]
    val_x, val_y = prepared["x_val"], prepared["y_val"]
    split_metadata = {key: prepared["metadata"][key] for key in
                      ("name", "split_seed", "outer_fold", "split_policy", "raw_array_sha256")}
    del prepared
    optimizer = torch.optim.Adam([model.gain], lr=float(config.get("learning_rate", .03)))
    rng = np.random.default_rng(seed + 103)

    def training_batch():
        ix = rng.choice(len(train_x), size=min(batch_size, len(train_x)), replace=False)
        x = torch.as_tensor(train_x[ix], dtype=torch.float32, device=device)
        y = torch.as_tensor(train_y[ix], dtype=torch.long if task == "classification" else torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)

        def forward_loss():
            output = model(x)[:, :n_classes] if task == "classification" else model.regression(x)
            return F.cross_entropy(output, y) if task == "classification" else F.mse_loss(output, y)

        loss, forward_seconds = timed(forward_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite profile loss")
        _, backward_seconds = timed(loss.backward)
        gradient = model.gain.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError("Missing or non-finite profile gradient")
        gradient_stats = {"nonzero_edges": int(torch.count_nonzero(gradient)),
                          "l2": float(gradient.norm()), "max_abs": float(gradient.abs().max())}
        _, clip_seconds = timed(lambda: torch.nn.utils.clip_grad_norm_(
            [model.gain], float(config.get("gradient_clip", 1.))))
        _, adam_seconds = timed(optimizer.step)
        return {"examples": len(ix), "loss": float(loss.detach()),
                "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
                "gradient_clip_seconds": clip_seconds, "adam_seconds": adam_seconds,
                "total_compute_seconds": forward_seconds + backward_seconds + clip_seconds + adam_seconds,
                "gradient_before_clipping": gradient_stats}

    # Fixed unit-channel probes are instrumentation, independent of all labels.
    # All-zero stimulation starts from a reset fast state and yields zero here.
    with torch.no_grad():
        probes = torch.cat([torch.zeros(1, 4), torch.eye(4)]).to(device)
        logits, activity = model(probes, return_activity=True)
        means = model.population_means(activity)
        output_delta = means[1:] - means[:1]
        propagation = {
            "probe_definition": "zero current and four separate unit input channels; reset state per example; before Adam updates",
            "input_population_sizes": [int((model.input_group == group).sum()) for group in range(4)],
            "output_population_sizes": [int((model.output_group == group).sum()) for group in range(3)],
            "active_neurons_abs_above_1e_minus_12_per_probe": (activity.abs() > 1e-12).sum(dim=1).cpu().tolist(),
            "population_means_per_probe": means.cpu().tolist(),
            "output_change_per_input_channel": output_delta.cpu().tolist(),
            "all_channels_reach_all_output_populations_at_1e_minus_12": bool((output_delta.abs() > 1e-12).all()),
            "finite": bool(torch.isfinite(logits).all() and torch.isfinite(activity).all()),
        }
        del probes, logits, activity, means, output_delta
    if not propagation["finite"]:
        raise FloatingPointError("Non-finite activity during fixed channel probes")

    warmup = training_batch()
    print(f"  warmup minibatch: {warmup['total_compute_seconds']:.3f}s", flush=True)
    measured = []
    for index in range(batches):
        row = training_batch()
        measured.append(row)
        print(f"  measured minibatch {index+1}/{batches}: {row['total_compute_seconds']:.3f}s; "
              f"nonzero gradients={row['gradient_before_clipping']['nonzero_edges']}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()

    @torch.no_grad()
    def validation_forward():
        loss_sum = 0.
        for start in range(0, len(val_x), batch_size):
            x = torch.as_tensor(val_x[start:start+batch_size], dtype=torch.float32, device=device)
            y = torch.as_tensor(val_y[start:start+batch_size], dtype=torch.long if task == "classification" else torch.float32, device=device)
            output = model(x)[:, :n_classes] if task == "classification" else model.regression(x)
            loss = F.cross_entropy(output, y) if task == "classification" else F.mse_loss(output, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation profile loss")
            loss_sum += float(loss) * len(x)
        return loss_sum / len(val_x)

    validation_loss, validation_seconds = timed(validation_forward)
    batch_seconds = float(np.median([row["total_compute_seconds"] for row in measured]))
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(p.numel() * p.element_size() for p in model.buffers())
    memory = {"parameter_tensor_bytes": parameter_bytes, "buffer_tensor_bytes": buffer_bytes,
              "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if is_cuda else None,
              "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if is_cuda else None}
    result = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compute and propagation profile; disposable Adam updates, not a learning/generalization experiment",
        "test_set_evaluated": False, "checkpoint_saved": False,
        "device": device, "gpu": gpu_record, "torch_version": torch.__version__,
        "dtype": "float32", "config": config, "graph": provenance,
        "model_source_sha256": file_hash(Path(__file__).with_name("model.py")),
        "profiler_source_sha256": file_hash(Path(__file__)),
        "neurons": model.num_nodes, "edges": model.num_edges,
        "plastic_edges": model.num_plastic_edges,
        "zero_base_plastic_edges": int(torch.count_nonzero(model.plastic_base == 0)),
        "steps": model.steps, "batch_size": batch_size,
        "train_samples": len(train_x), "validation_samples": len(val_x),
        "split": split_metadata, "graph_load_and_hash_seconds": load_seconds,
        "model_construction_and_transfer_seconds": construction_seconds,
        "propagation": propagation, "warmup": warmup, "measured_batches": measured,
        "median_training_batch_seconds": batch_seconds,
        "validation_forward_seconds": validation_seconds,
        "validation_loss_after_disposable_updates": validation_loss,
        "memory": memory,
        "estimate": estimate_training_cost(batch_seconds, validation_seconds,
            train_samples=len(train_x), batch_size=batch_size,
            epochs=int(config.get("epochs", 40)), plastic_edges=model.num_plastic_edges,
            nodes=model.num_nodes, steps=model.steps),
        "wall_seconds": time.perf_counter() - started,
        "timing_note": "CUDA synchronized at each measured phase; gradient diagnostics and tensor conversion excluded. Median excludes the warmup batch. Brief measurements may not represent sustained workloads or thermal contention.",
    }
    write_json(destination, result)
    print(f"  estimated epoch: {result['estimate']['epoch_seconds']:.3f}s; "
          f"{result['estimate']['epochs']} epochs: {result['estimate']['training_and_validation_hours']:.3f}h", flush=True)
    return result
