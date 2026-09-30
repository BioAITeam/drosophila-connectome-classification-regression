"""Training, reserved evaluation and causal memory controls for a fixed graph."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from .controls import rewire_graph
from .datasets import prepare_dataset, inverse_targets
from .metrics import classification_metrics, regression_metrics, paired_bootstrap_delta
from .model import SparseConnectome
from .paths import REPO, portable_metadata, portable_path, repo_path, writable_output
from .plasticity import LocalRewardLearner


def write_json(path, value):
    path = writable_output(path)
    def default(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(portable_metadata(value), indent=2, ensure_ascii=False, default=default, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def buffers_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_buffers()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_graph(path, encoding="visual", rewire_seed=None):
    path = repo_path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        graph = {name: data[name] for name in data.files}
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update(processed_file=portable_path(path), processed_sha256=file_hash(path), encoding=encoding)
    if encoding == "direct":
        if "direct_input_group" not in graph:
            raise ValueError("Graph lacks anatomically declared direct_input_group; prepare it again")
        graph["input_group"] = graph["direct_input_group"].copy()
    elif encoding != "visual":
        raise ValueError("encoding must be visual or direct")
    if rewire_seed is not None:
        graph = rewire_graph(graph, seed=rewire_seed, swaps_per_edge=1)
        metadata["rewire"] = graph.pop("rewire_metadata")
        metadata["anatomical_topology_preserved"] = False
    return graph, metadata


def make_model(graph, config, device):
    if config.get("rewire_seed") is not None and config.get("plastic_scope", "all") != "all":
        raise ValueError("Rewired controls require plastic_scope=all; original KC→MBON masks do not identify rewired pairs")
    kwargs = dict(config.get("model", {}))
    if "plastic_scope" in kwargs and "plastic_scope" in config and kwargs["plastic_scope"] != config["plastic_scope"]:
        raise ValueError("Conflicting plastic_scope in model and experiment configuration")
    kwargs.setdefault("plastic_scope", config.get("plastic_scope", "all"))
    model = SparseConnectome(graph, **kwargs).to(device)
    model.graph_arrays = graph
    trainables = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if [name for name, p in trainables] != ["gain"]:
        raise RuntimeError(f"Unexpected trainable components: {[name for name,p in trainables]}")
    if model.gain.numel() != model.plastic_edge_indices.numel():
        raise RuntimeError("Each trainable gain must correspond to an existing graph edge")
    return model


@torch.no_grad()
def predict(model, x, *, task, n_classes, batch_size=16, device="cpu"):
    model.eval()
    result = []
    for start in range(0, len(x), batch_size):
        inputs = torch.as_tensor(x[start:start+batch_size], dtype=torch.float32, device=device)
        output = model(inputs)[:, :n_classes] if task == "classification" else model.regression(inputs)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite connectome output")
        result.append(output.detach().cpu().numpy())
    values = np.concatenate(result)
    return values.argmax(1) if task == "classification" else values


def evaluate(model, dataset, partition, config, device, x_override=None):
    meta = dataset["metadata"]
    task = meta["task"]
    if task == "classification" and partition in {"train", "val"}:
        with torch.no_grad():
            model.eval()
            total, predictions = 0., []
            values = dataset[f"x_{partition}"] if x_override is None else x_override
            for start in range(0, len(values), config.get("batch_size", 16)):
                x = torch.as_tensor(values[start:start+config.get("batch_size", 16)], dtype=torch.float32, device=device)
                y = torch.as_tensor(dataset[f"y_{partition}"][start:start+len(x)], dtype=torch.long, device=device)
                logits = model(x)[:, :meta["n_classes"]]
                total += float(F.cross_entropy(logits, y, reduction="sum"))
                predictions.append(logits.argmax(1).cpu().numpy())
            pred = np.concatenate(predictions)
            metrics = classification_metrics(dataset[f"y_{partition}"], pred, n_classes=meta["n_classes"])
            metrics["cross_entropy"] = total / len(values)
            return pred, metrics
    pred = predict(model, dataset[f"x_{partition}"] if x_override is None else x_override,
                   task=task, n_classes=meta.get("n_classes") or 3,
                   batch_size=config.get("batch_size", 16), device=device)
    if task == "classification":
        metrics = classification_metrics(dataset[f"y_{partition}"], pred, n_classes=meta["n_classes"])
        return pred, metrics
    original = inverse_targets(pred, meta)
    return original, regression_metrics(dataset[f"y_raw_{partition}"], original)


def _validation_score(metrics, task):
    return metrics["balanced_accuracy"] if task == "classification" else -metrics["rmse"]


@torch.no_grad()
def development_diagnostics(model, data, config, device):
    """A fixed prefix of train stimuli; never reads test or a teacher signal."""
    size = min(int(config.get("diagnostic_samples", 8)), len(data["x_train"]))
    x = torch.as_tensor(data["x_train"][:size], dtype=torch.float32, device=device)
    activity = model.simulate(x, return_trajectory=True)
    absolute = activity.abs()
    by_cell = absolute.amax(dim=(0, 1))
    probability = model.gain.sigmoid()
    result = {"partition": "train", "n_samples": size,
              "mean_absolute_activity": float(absolute.mean()),
              "silent_neuron_fraction": float((by_cell <= 1e-8).float().mean()),
              "activity_saturation_fraction": float((absolute >= .99).float().mean()),
              "efficacy_boundary_fraction": float(((probability < .01) | (probability > .99)).float().mean()),
              "output_population_mean": model.population_means(activity).mean(0).cpu().tolist(),
              "output_population_std": model.population_means(activity).std(0, unbiased=False).cpu().tolist(),
              "mean_absolute_activity_by_step": absolute.mean(dim=(1, 2)).cpu().tolist()}
    for name, mask in (("input", model.input_group >= 0), ("output", model.output_group >= 0),
                       ("reward_dan", model.reward_mask), ("punishment_dan", model.punishment_mask)):
        if mask.any():
            result[name + "_mean_absolute_activity"] = float(absolute[:, :, mask].mean())
            result[name + "_silent_fraction"] = float((by_cell[mask] <= 1e-8).float().mean())
    if model.dynamics == "lif":
        result["spike_fraction"] = float(activity.mean())
    if config.get("algorithm") == "local_reward" or model.dynamics == "lif":
        active = data["metadata"].get("n_active_features", 4)
        baseline = x.mean(0)
        stimuli = baseline.repeat(active + 1, 1)
        for channel in range(active):
            stimuli[channel + 1, channel] = (baseline[channel] + .1).clamp_max(1.)
        responses = model.simulate(stimuli, return_trajectory=True)
        means = model.population_means(responses)
        result["channel_sensitivity"] = {
            "stimuli": stimuli.cpu().tolist(), "labels_used": False,
            "neural_response_l2": (responses[:, 1:] - responses[:, :1]).square().sum(2).sqrt().amax(0).cpu().tolist(),
            "output_response_l2": (means[1:] - means[:1]).norm(dim=1).cpu().tolist()}
    return result


def stable_gain_norm(value):
    import math
    flat = value.detach().cpu().numpy()
    return math.sqrt(math.fsum(float(np.square(flat[i:i+1_000_000].astype(np.float64)).sum())
                               for i in range(0, len(flat), 1_000_000)))


def validate_training_config(config):
    if config.get("algorithm") not in {"supervised", "local_reward"}:
        raise ValueError("algorithm must be supervised or local_reward")
    if config.get("select_checkpoint", "best_validation") not in {"best_validation", "last"}:
        raise ValueError("select_checkpoint must be best_validation or last")
    integer_defaults = {"epochs": (40, 0), "batch_size": (16, 1), "min_epochs": (0, 0),
                        "diagnostic_samples": (8, 1), "bootstrap_samples": (2000, 1)}
    for name, (default, minimum) in integer_defaults.items():
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    patience = config.get("patience", config.get("epochs", 40) + 1)
    if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 1:
        raise ValueError("patience must be a positive integer")
    for name, default in (("learning_rate", .03), ("gradient_clip", 1.), ("min_delta", 1e-8)):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def train(model, data, config, directory, device, *, reward_schedule=None):
    validate_training_config(config)
    task = data["metadata"]["task"]
    n_classes = data["metadata"].get("n_classes") or 3
    algorithm = config["algorithm"]
    epochs = int(config.get("epochs", 40))
    batch_size = int(config.get("batch_size", 16))
    seed = int(config.get("optimization_seed", config.get("seed", 0)))
    rng = np.random.default_rng(seed + 103)
    label_rng = np.random.default_rng(seed + 701)
    generator = torch.Generator(device=device).manual_seed(seed + 1009)
    history = []
    rewards = []
    _, initial_validation = evaluate(model, data, "val", config, device)
    _, initial_training = evaluate(model, data, "train", config, device)
    best_score = _validation_score(initial_validation, task)
    best_gain = model.gain.detach().clone()
    best_epoch = 0
    patience = int(config.get("patience", epochs + 1))
    stale = 0
    optimizer = torch.optim.Adam([model.gain], lr=float(config.get("learning_rate", .03))) if algorithm == "supervised" else None
    learner = LocalRewardLearner(model, learning_rate=config.get("learning_rate", .02),
                                 **config.get("local", {})) if algorithm == "local_reward" else None
    if algorithm not in {"supervised", "local_reward"}:
        raise ValueError("Unknown training algorithm")
    start_time = time.monotonic()
    history.append({"epoch": 0, "training": initial_training, "validation": initial_validation,
                    "diagnostics": development_diagnostics(model, data, config, device), "elapsed_seconds": 0.})
    write_json(directory / "history.json", history)
    fixed_targets = data["y_train"].copy()
    if config.get("fixed_train_label_permutation_seed") is not None:
        from .datasets import permute_grouped_targets
        # Train-only diagnostic, distinct from the global inferential null.
        fixed_targets, permutation_record = permute_grouped_targets(
            fixed_targets, data["groups_train"], int(config["fixed_train_label_permutation_seed"]))
        permutation_record["scope"] = "training partition only; fixed across all epochs; diagnostic, not inferential null"
        write_json(directory / "train_label_permutation.json", permutation_record)
    if reward_schedule is not None:
        epochs = len(reward_schedule)
        patience = epochs + 1  # A matched-pulse control must replay the complete source schedule.
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(data["x_train"]))
        if config.get("shuffle_labels", False):
            targets = label_rng.permutation(data["y_train"])
        else:
            targets = fixed_targets
        losses, epoch_rewards, local_changes = [], [], 0
        gradient_norms, gradient_fractions, local_stats = [], [], []
        replay = None
        if reward_schedule is not None:
            if epoch >= len(reward_schedule):
                raise ValueError("Reward replay shorter than training schedule")
            replay = np.asarray(reward_schedule[epoch], dtype=np.float32)
            if len(replay) != len(order):
                raise ValueError("Reward replay must preserve number of examples and pulses")
            replay = np.random.default_rng(seed + 7001 + epoch).permutation(replay)
        for start in range(0, len(order), batch_size):
            ix = order[start:start+batch_size]
            x = torch.as_tensor(data["x_train"][ix], dtype=torch.float32, device=device)
            y = torch.as_tensor(targets[ix], dtype=torch.long if task == "classification" else torch.float32, device=device)
            if algorithm == "supervised":
                optimizer.zero_grad(set_to_none=True)
                out = model(x)[:, :n_classes] if task == "classification" else model.regression(x)
                loss = F.cross_entropy(out, y) if task == "classification" else F.mse_loss(out, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                gradient_fractions.append(float((model.gain.grad != 0).float().mean()))
                gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(
                    [model.gain], float(config.get("gradient_clip", 1.)), error_if_nonfinite=True)))
                optimizer.step()
                losses.append((float(loss.detach()), len(ix)))
            else:
                override = replay[start:start+len(ix)] if replay is not None else None
                if config.get("no_reward", False):
                    override = np.zeros(len(ix), dtype=np.float32)
                stats = learner.step(x, y, task=task, n_classes=n_classes, reward_override=override, generator=generator)
                epoch_rewards.extend(stats["reward"])
                local_changes += stats["changed_edges"]
                local_stats.append(stats)
        _, metrics = evaluate(model, data, "val", config, device)
        _, training_metrics = evaluate(model, data, "train", config, device)
        score = _validation_score(metrics, task)
        row = {"epoch": epoch + 1, "training": training_metrics, "validation": metrics,
               "diagnostics": development_diagnostics(model, data, config, device),
               "elapsed_seconds": time.monotonic() - start_time}
        if gradient_norms:
            row["diagnostics"].update(gradient_l2_before_clip_mean=float(np.mean(gradient_norms)),
                                       nonzero_gradient_fraction_mean=float(np.mean(gradient_fractions)))
        if losses:
            row["training_loss"] = sum(v*n for v,n in losses) / sum(n for _,n in losses)
        if learner:
            row.update(reward_mean=float(np.mean(epoch_rewards)), local_edge_updates=local_changes,
                       dan_to_plastic_coverage=learner.coverage)
            rewards.append(epoch_rewards)
            row["diagnostics"].update({key: float(np.mean([s[key] for s in local_stats]))
                for key in local_stats[0] if key not in {"reward", "predictions"}})
        history.append(row)
        write_json(directory / "history.json", history)
        if score > best_score + float(config.get("min_delta", 1e-8)):
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_gain = model.gain.detach().clone()
        else:
            stale += 1
        print(f"  epoch {epoch+1}/{epochs}: validation score={score:.6g}", flush=True)
        if stale >= patience and epoch + 1 >= int(config.get("min_epochs", 0)):
            break
    # Checkpoint selection uses validation only. Last state is saved separately.
    torch.save({"gain": model.gain.detach().cpu()}, directory / "last_parameters.pt")
    if config.get("select_checkpoint", "best_validation") == "best_validation":
        with torch.no_grad():
            model.gain.copy_(best_gain)
    if rewards:
        write_json(directory / "reward_schedule.json", rewards)
    return {"epochs_completed": len(history)-1, "selected_epoch": best_epoch if config.get("select_checkpoint", "best_validation") == "best_validation" else len(history)-1,
            "selection": config.get("select_checkpoint", "best_validation"), "history": history,
            "reward_schedule": rewards, "duration_seconds": time.monotonic() - start_time}


def _save_predictions(path, data, before, after, extra):
    arrays = {"row_index": data["indices"]["test"], "true": data["y_raw_test"],
              "before": before, "after": after, **extra}
    np.savez_compressed(writable_output(path), **arrays)


def _code_fingerprint():
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted((REPO / "3 Connectome").rglob("*.py"))}


def run_experiment(config, *, graph_override=None, quick=False, output_root=None, reward_schedule=None, initial_checkpoint=None):
    config = copy.deepcopy(config)
    if quick:
        config.update(epochs=2, patience=3, bootstrap_samples=100, software_validation_only=True)
    validate_training_config(config)
    device = os.environ.get("FLYBRAIN_DEVICE", "cpu")
    seed = int(config.get("optimization_seed", config.get("seed", 0)))
    split_seed = int(config.get("split_seed", config.get("seed", 0)))
    config.update(split_seed=split_seed, optimization_seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get("FLYBRAIN_THREADS", "4")))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    graph_path = graph_override or config.get("graph", "2 Data/3 Processed/visual_mb.npz")
    graph, provenance = load_graph(graph_path, config.get("encoding", "visual"), config.get("rewire_seed"))
    if provenance.get("required_plastic_scope") and config.get("plastic_scope", "all") != provenance["required_plastic_scope"]:
        raise ValueError("Cached rewired graphs require plastic_scope=all")
    if config.get("freeze_candidate", False):
        graph["plastic_mask"] = ~graph["plastic_mask"]
        config["plastic_scope"] = "kc_mbon"
        provenance["plasticity_override"] = "all existing edges EXCEPT original KC→MBON; mask complemented for the model API"
    model = make_model(graph, config, device)
    initial_checkpoint = initial_checkpoint or config.get("initial_checkpoint")
    if initial_checkpoint:
        from .checkpoints import load_checkpoint
        source = load_checkpoint(initial_checkpoint)
        if source["graph_sha256"] != provenance["processed_sha256"] or source["fixed_buffers_sha256"] != buffers_hash(model):
            raise ValueError("Transfer requires the identical graph and immutable interfaces")
        if source["config"].get("model", {}) != config.get("model", {}):
            raise ValueError("Transfer must preserve neuronal dynamics and fixed readout")
        with torch.no_grad():
            model.gain.copy_(source["gain"].to(device))
    initial_gain = model.gain.detach().clone()
    initial_buffers = buffers_hash(model)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = config.get("name", config["dataset"])
    directory = writable_output(Path(output_root or REPO / "5 Iris/6 Workspace/runs") / f"{name}-p{split_seed}-s{seed}-f{config.get('outer_fold', 0)}-{stamp}")
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    write_json(directory / "graph.json", provenance)
    if initial_checkpoint:
        from .checkpoints import checkpoint_source_sha256
        write_json(directory / "training_source.json", {"checkpoint": str(initial_checkpoint),
                    "checkpoint_sha256": checkpoint_source_sha256(initial_checkpoint), "optimizer_reinitialized": True})
    write_json(directory / "environment.json", {"device": device, "gpu": json.loads(os.environ.get("FLYBRAIN_GPU_RECORD", "null")),
               "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "pyarrow")},
               "code_sha256": _code_fingerprint(), "determinism_note": "CUDA sparse reductions may not be bitwise deterministic; compare repeated seeds."})
    data = prepare_dataset(config["dataset"], seed=split_seed, outer_fold=config.get("outer_fold"), output_dir=directory / "dataset",
                           label_permutation_seed=config.get("label_permutation_seed"), split_spec=config.get("split_spec"))
    if initial_checkpoint:
        source_meta = json.loads((repo_path(initial_checkpoint).parent / "dataset/metadata.json").read_text())
        for key in ("scaler", "target_scaler", "task", "feature_names", "target_names"):
            if source_meta.get(key) != data["metadata"].get(key):
                raise ValueError(f"Transfer must retain the source {key}")
    task = data["metadata"]["task"]
    print(f"{name}: {len(graph['node_ids'])} neurons, {len(graph['src'])} edges, {model.gain.numel()} plastic, {device}", flush=True)
    before, initial_metrics = evaluate(model, data, "test", config, device)
    # No test evaluation is used while choosing epochs or modifying the configuration.
    if config.get("frozen", False):
        _, tm = evaluate(model, data, "train", config, device)
        _, vm = evaluate(model, data, "val", config, device)
        count = int(config.get("epochs", 0)) if config.get("record_frozen_epochs", False) else 0
        diagnostic = development_diagnostics(model, data, config, device)
        history = [{"epoch": i, "training": tm, "validation": vm, "diagnostics": diagnostic,
                    "frozen": True, "elapsed_seconds": 0.} for i in range(count + 1)]
        write_json(directory / "history.json", history)
        training = {"epochs_completed": count, "selected_epoch": 0, "selection": "frozen_control", "history": history, "duration_seconds": 0.}
    else:
        training = train(model, data, config, directory, device, reward_schedule=reward_schedule)
    after, final_metrics = evaluate(model, data, "test", config, device)
    trained_gain = model.gain.detach().clone()
    repeated, repeated_metrics = evaluate(model, data, "test", config, device)
    with torch.no_grad():
        model.gain.copy_(initial_gain)
    restored, restored_metrics = evaluate(model, data, "test", config, device)
    with torch.no_grad():
        model.gain.copy_(trained_gain)
    zeros, zeros_metrics = evaluate(model, data, "test", config, device, np.zeros_like(data["x_test"]))
    permutation = np.random.default_rng(seed + 171).permutation(len(after))
    permuted, permuted_metrics = evaluate(model, data, "test", config, device, data["x_test"][permutation])
    final_buffers = buffers_hash(model)
    if final_buffers != initial_buffers:
        raise AssertionError("A fixed interface or anatomical buffer changed during training")
    if not np.allclose(restored, before, atol=1e-6, rtol=1e-5):
        raise AssertionError("Restoring initial internal weights failed to restore predictions")
    if not np.allclose(repeated, after, atol=1e-6, rtol=1e-5):
        raise AssertionError("Fast-state reset changed repeated inference")
    delta = paired_bootstrap_delta(data["y_raw_test"], before, after, task=task,
                                  groups=data["groups_test"], n_bootstrap=int(config.get("bootstrap_samples", 2000)), seed=seed)
    reference = np.full(len(after), int(np.bincount(data["y_train"]).argmax())) if task == "classification" else np.full(len(after), data["y_raw_train"].mean())
    reference_metrics = classification_metrics(data["y_test"], reference, n_classes=data["metadata"]["n_classes"]) if task == "classification" else regression_metrics(data["y_raw_test"], reference)
    results = {"name": name, "task": task, "software_validation_only": bool(config.get("software_validation_only", False)),
               "initial": initial_metrics, "final": final_metrics, "paired_change": delta,
               "frozen_initial_reference": initial_metrics, "constant_reference": reference_metrics,
               "controls": {"restored_initial_weights": restored_metrics, "reset_fast_state": repeated_metrics,
                            "zero_input": zeros_metrics, "permuted_test_input": permuted_metrics},
               "epochs_completed": training["epochs_completed"], "selected_epoch": training["selected_epoch"],
               "training_seconds": training["duration_seconds"],
               "invariants": {"buffers_unchanged": True, "buffers_sha256": initial_buffers,
                              "memory_reset_reproduces_initial": True, "fast_state_reset_reproduces_final": True,
                              "trainable_parameters": [name for name,_ in model.named_parameters()],
                              "n_neurons": len(graph["node_ids"]), "n_edges": len(graph["src"]),
                              "n_plastic_edges": model.gain.numel(), "changed_gains": int((trained_gain != initial_gain).sum()),
                              "gain_change_l2": stable_gain_norm(trained_gain-initial_gain),
                              "extra_learnable_neurons": 0, "learnable_interface_parameters": 0}}
    if "x_test_extrapolation" in data:
        extra_pred, extra_metrics = evaluate(model, data, "test_extrapolation", config, device)
        results["extrapolation"] = extra_metrics
        np.savez_compressed(directory / "extrapolation.npz", prediction=extra_pred, true=data["y_raw_test_extrapolation"])
    _save_predictions(directory / "predictions.npz", data, before, after,
                      {"restored": restored, "reset_state": repeated, "zero_input": zeros,
                       "permuted_input": permuted, "input_permutation": permutation})
    checkpoint = {"format_version": 1, "graph_sha256": provenance["processed_sha256"], "config": portable_metadata(config),
                  "gain": trained_gain.cpu(), "initial_gain": initial_gain.cpu(),
                  "plastic_edge_indices": model.plastic_edge_indices.detach().cpu(),
                  "fixed_buffers_sha256": initial_buffers,
                  "note": "Load only with the exact graph, fixed encoding and constructor config; no learned readout."}
    torch.save(checkpoint, directory / "model.pt")
    plastic = model.plastic_edge_indices.detach().cpu().numpy()
    np.savez_compressed(directory / "plastic_edges.npz", pre_body_id=graph["node_ids"][graph["src"][plastic]],
                        post_body_id=graph["node_ids"][graph["dst"][plastic]],
                        original_contact_count=graph["weight"][plastic],
                        efficacy=model.edge_efficacies().detach().cpu().numpy(),
                        initial_gain=initial_gain.cpu().numpy(), trained_gain=trained_gain.cpu().numpy())
    results["checkpoint_sha256"] = file_hash(directory / "model.pt")
    write_json(directory / "results.json", results)
    return directory, results
