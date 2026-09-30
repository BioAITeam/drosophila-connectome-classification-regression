"""Reproducible command-line operations; device admission belongs to the launcher."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from .paths import REPO, RAW, TEMP, portable_metadata, portable_path, repo_path, writable_output


def _config(path):
    path = repo_path(path)
    if not path.exists() and not str(path).endswith(".json"):
        candidates = sorted((REPO / "5 Iris/1 Configuration").rglob(Path(path).name + "*.json"))
        if len(candidates) == 1:
            path = candidates[0]
    return json.loads(path.read_text())


def _run(args):
    from .experiment import run_experiment
    config = _config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.split_seed is not None:
        config["split_seed"] = args.split_seed
    if args.optimization_seed is not None:
        config["optimization_seed"] = args.optimization_seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.fold is not None:
        config["outer_fold"] = args.fold
    path, result = run_experiment(config, graph_override=args.graph, quick=args.quick)
    print(json.dumps({"run": portable_path(path), "initial": result["initial"], "final": result["final"]}, indent=2))


def _suite(args):
    from .experiment import run_experiment, write_json
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = writable_output(REPO / "5 Iris/6 Workspace/suites" / stamp)
    folder.mkdir(parents=True, exist_ok=False)
    config_paths = [repo_path(x) for x in args.configs] if args.configs else sorted((REPO / "5 Iris/1 Configuration/tasks").glob("*.json"))
    if not config_paths:
        raise ValueError("No active task configs; run benchmark --prepare first")
    results = []
    def execute(config, replay=None):
        path, result = run_experiment(config, graph_override=args.graph, quick=args.quick,
                                      output_root=folder / "runs", reward_schedule=replay)
        record = {"config": config["name"], "seed": config.get("seed", 0), "fold": config.get("outer_fold", 0),
                  "split_seed": config.get("split_seed", config.get("seed", 0)),
                  "optimization_seed": config.get("optimization_seed", config.get("seed", 0)),
                  "rewire_seed": config.get("rewire_seed"),
                  "run": str(path), "result": result}
        results.append(record)
        write_json(folder / "suite.json", {"quick": args.quick, "runs": results})
        return path
    for path in config_paths:
        base = _config(path)
        for seed in args.seeds:
            for fold in args.folds:
                config = {**base, "seed": seed, "outer_fold": fold}
                if args.epochs is not None:
                    config["epochs"] = args.epochs
                main = execute(config)
                if args.controls:
                    frozen = {**config, "name": config["name"] + "_frozen", "frozen": True}
                    execute(frozen)
                    if config["algorithm"] == "local_reward":
                        replay = json.loads((main / "reward_schedule.json").read_text())
                        execute({**config, "name": config["name"] + "_shuffled_reward"}, replay)
                        execute({**config, "name": config["name"] + "_no_reward", "no_reward": True})
                    else:
                        execute({**config, "name": config["name"] + "_shuffled_labels", "shuffle_labels": True})
                        if config.get("plastic_scope", "all") == "all":
                            execute({**config, "name": config["name"] + "_freeze_kc_mbon", "freeze_candidate": True})
                    if args.rewire:
                        if config.get("plastic_scope", "all") == "all" and config["algorithm"] == "supervised":
                            execute({**config, "name": config["name"] + "_rewired", "rewire_seed": seed + 137})
    print(f"Suite saved: {portable_path(folder)}")


def _predict(args):
    out = repo_path(args.output or "5 Iris/6 Workspace/predictions/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    out = writable_output(out)
    import numpy as np
    import torch
    from .datasets import transform_features, inverse_targets
    from .experiment import load_graph, make_model, buffers_hash, write_json
    from .checkpoints import load_checkpoint
    checkpoint_path = repo_path(args.checkpoint)
    result_path = checkpoint_path.parent / "results.json"
    saved_result = json.loads(result_path.read_text())
    expected_sha256 = saved_result.get("checkpoint_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("A valid recorded SHA-256 is required.")
    checkpoint = load_checkpoint(checkpoint_path, expected_sha256)
    config = checkpoint["config"]
    graph_info = json.loads((checkpoint_path.parent / "graph.json").read_text())
    graph, meta = load_graph(args.graph or graph_info["processed_file"], config.get("encoding", "visual"), config.get("rewire_seed"))
    if meta["processed_sha256"] != checkpoint["graph_sha256"]:
        raise ValueError("The graph SHA-256 does not match the checkpoint")
    if config.get("freeze_candidate"):
        graph["plastic_mask"] = ~graph["plastic_mask"]
    device = os.environ.get("FLYBRAIN_DEVICE", "cpu")
    model = make_model(graph, config, device)
    if buffers_hash(model) != checkpoint["fixed_buffers_sha256"]:
        raise ValueError("Fixed model buffers do not match training")
    if not torch.equal(model.plastic_edge_indices.cpu(), checkpoint["plastic_edge_indices"]):
        raise ValueError("Plastic edge indices do not match training")
    with torch.no_grad():
        model.gain.copy_(checkpoint["gain"].to(device))
    dataset_meta = json.loads((checkpoint_path.parent / "dataset/metadata.json").read_text())
    values = np.loadtxt(repo_path(args.inputs), delimiter=",", ndmin=2)
    active = dataset_meta.get("n_active_features", 3 if dataset_meta["name"] == "iris_regression" else 4)
    if values.shape[1] == active and active < 4:
        values = np.column_stack((values, np.zeros((len(values), 4-active))))
    if values.shape[1] != 4:
        raise ValueError(f"Headerless CSV requires {active} predictors in metadata.feature_names order; zero padding to four channels is allowed")
    if active < 4 and np.any(values[:, active:] != 0):
        raise ValueError("Padding channels must be zero; the excluded regression target must never be an input")
    if not np.isfinite(values).all():
        raise ValueError("Input predictors must contain only finite numeric values.")
    batch_size = getattr(args, "batch_size", None)
    if batch_size is None:
        batch_size = config.get("batch_size", 16)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Inference batch size must be a positive integer.")
    if not len(values):
        raise ValueError("The input CSV must contain at least one observation.")
    scaled = transform_features(values, dataset_meta["scaler"])
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            x = torch.as_tensor(scaled[start:start + batch_size], dtype=torch.float32, device=device)
            if dataset_meta["task"] == "classification":
                output = model(x)[:, :dataset_meta["n_classes"]].softmax(dim=1)
            else:
                output = model.regression(x)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Non-finite connectome output")
            chunks.append(output.cpu().numpy())
    predicted = np.concatenate(chunks)
    if dataset_meta["task"] == "classification":
        result = {"class_id": predicted.argmax(1), "probabilities": predicted,
                  "class_names": dataset_meta["target_names"]}
    else:
        result = {"prediction": inverse_targets(predicted, dataset_meta), "units": dataset_meta.get("target_units")}
    result.update(checkpoint=str(checkpoint_path), plasticity_enabled=False, teacher_enabled=False)
    write_json(out, result)
    print(portable_path(out))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MaleCNS internal synaptic learning with fixed interfaces and reproducible evaluation",
        epilog="Launcher options: --device cpu (default), cuda, or auto; --gpu INDEX selects an idle physical GPU."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List active Iris experiments")
    listing.add_argument("--historical", action="store_true")
    benchmark = commands.add_parser("benchmark", help="Prepare the active five-task Iris experiment plans")
    benchmark.add_argument("--prepare", action="store_true", required=True)
    benchmark.add_argument("--task-specs", help="JSON task specifications for applying the same recipes to another dataset")
    benchmark.add_argument("--output", help="Directory for generated plans; must be inside the repository")
    benchmark.add_argument("--graph-root", help="Repository directory containing the prepared anatomical, full and control graphs")
    download = commands.add_parser("download", help="Download verified originals without overwriting raw data")
    download.add_argument("--raw-root", type=Path, default=RAW)
    prepare = commands.add_parser("prepare", help="Prepare the anatomical graph, annotations, and fixed interfaces")
    prepare.add_argument("--mode", choices=("visual_mb", "full"), default="visual_mb")
    prepare.add_argument("--output")
    prepare.add_argument("--raw-root", type=Path, default=RAW)
    profile = commands.add_parser("profile", help="Measure train/validation runtime without evaluating test data")
    profile.add_argument("--config", required=True)
    profile.add_argument("--graph", default="2 Data/3 Processed/full.npz")
    profile.add_argument("--output", required=True)
    profile.add_argument("--batches", type=int, default=2)
    run = commands.add_parser("run", help="Train and evaluate one experiment")
    run.add_argument("config")
    run.add_argument("--graph")
    run.add_argument("--seed", type=int)
    run.add_argument("--split-seed", type=int, help="Partition seed; overrides the general seed for data splitting")
    run.add_argument("--optimization-seed", type=int, help="Optimization seed; overrides the general seed for training")
    run.add_argument("--fold", type=int, choices=range(5))
    run.add_argument("--epochs", type=int)
    run.add_argument("--quick", action="store_true")
    suite = commands.add_parser("suite", help="Run experiments, repeated evaluations, and controls")
    suite.add_argument("--configs", nargs="+")
    suite.add_argument("--graph")
    suite.add_argument("--seeds", type=int, nargs="+", default=[0])
    suite.add_argument("--folds", type=int, nargs="+", default=[0], choices=range(5))
    suite.add_argument("--epochs", type=int)
    suite.add_argument("--quick", action="store_true")
    suite.add_argument("--controls", action="store_true")
    suite.add_argument("--rewire", action="store_true")
    campaign = commands.add_parser("campaign", help="Execute or resume a fixed repeated-evaluation plan")
    campaign.add_argument("--plan", required=True)
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--workers", type=int, default=1)
    campaign.add_argument("--no-aggregate", action="store_true")
    worker = commands.add_parser("campaign-worker", help="Run independent jobs from a shared queue; one idle GPU per process")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--output", required=True)
    aggregate = commands.add_parser("aggregate", help="Aggregate out-of-fold predictions from a campaign")
    aggregate.add_argument("folder")
    aggregate.add_argument("--sources", nargs="+", help="Compare campaigns and write combined outputs into folder")
    aggregate.add_argument("--bootstrap-samples", type=int, default=1000)
    prediction = commands.add_parser("predict", help="Predict with a saved checkpoint and fixed interfaces, without learning or feedback")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--inputs", required=True)
    prediction.add_argument("--output")
    prediction.add_argument("--graph")
    prediction.add_argument("--batch-size", type=int,
                            help="Inference batch size; defaults to the saved training configuration")
    args = parser.parse_args(argv)
    if args.command == "suite" and args.rewire and not args.controls:
        parser.error("--rewire requires --controls")
    if args.command == "list":
        root = REPO / "5 Iris/1 Configuration" if args.historical else REPO / "5 Iris/1 Configuration/tasks"
        for path in sorted(root.rglob("*.json" if not args.historical else "e*.json")):
            cfg = _config(path)
            print(f"{path.relative_to(REPO)}: {cfg.get('description', cfg.get('name'))}")
    elif args.command == "benchmark":
        from .benchmark import build_iris_plans, build_benchmark_plans
        if args.task_specs:
            specifications = json.loads(repo_path(args.task_specs).read_text())
            if isinstance(specifications, dict):
                specifications = specifications["tasks"]
            print(portable_metadata(build_benchmark_plans(
                specifications, args.output or "5 Iris/6 Workspace/configuration", graph_root=args.graph_root)))
        else:
            print(portable_metadata(build_iris_plans(
                args.output or "5 Iris/6 Workspace/configuration", graph_root=args.graph_root)))
    elif args.command == "download":
        from .download import download_sources
        print(portable_metadata(download_sources(args.raw_root, REPO / "2 Data/2 Manifests", TEMP / "downloads")))
    elif args.command == "prepare":
        from .data import prepare_graph
        destination = writable_output(repo_path(args.output or f"5 Iris/6 Workspace/graphs/{args.mode}.npz"))
        print(portable_metadata(prepare_graph(args.raw_root, destination, mode=args.mode)))
    elif args.command == "run":
        _run(args)
    elif args.command == "profile":
        from .profiling import profile_training
        result = profile_training(args.graph, _config(args.config), args.output, batches=args.batches)
        print(json.dumps(portable_metadata(result), indent=2))
    elif args.command == "suite":
        _suite(args)
    elif args.command == "predict":
        _predict(args)
    elif args.command == "campaign":
        from .campaign import run_campaign
        print(portable_metadata(run_campaign(args.plan, args.output, workers=args.workers, aggregate=not args.no_aggregate)))
    elif args.command == "campaign-worker":
        from .distributed import run_worker
        print(portable_metadata(run_worker(args.plan, args.output)))
    elif args.command == "aggregate":
        from .aggregation import aggregate_campaign
        folder = repo_path(args.folder)
        sources = [repo_path(path) for path in args.sources] if args.sources else [folder]
        records = [record for source in sources for record in json.loads((source / "campaign.json").read_text())["runs"]]
        aggregate_campaign(folder, records, n_bootstrap=args.bootstrap_samples)
        if args.sources:
            from .experiment import file_hash, write_json
            write_json(folder / "aggregation_sources.json", [
                {"campaign": str(source), "campaign_sha256": file_hash(source / "campaign.json")}
                for source in sources])
        print(portable_path(folder / "aggregate.md"))


if __name__ == "__main__":
    from .launcher import main as launch
    launch()
