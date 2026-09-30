"""Command-line interface for Ki-67 experiments."""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from .common import CONFIG, OUTPUT, WORKSPACE, CLASS_TASKS, writable_location, read_json, write_json, resolve_output_reference

MUTATING_COMMANDS = frozenset({"audit", "download", "extract", "profile", "campaign", "launch", "worker", "monitor"})

def parse_arguments(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "download", "extract", "profile", "campaign", "evaluate", "predict", "launch", "status", "worker", "monitor"]:
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path, required=name in MUTATING_COMMANDS, default=None if name in MUTATING_COMMANDS else OUTPUT)
        sub.add_argument("--plan", type=Path, required=name in MUTATING_COMMANDS)
        if name in ["extract", "profile"]:
            sub.add_argument("--dataset", choices=["bcdata", "shid", "joint"], required=True)
            sub.add_argument("--representation", choices=["rgb", "dinov2", "phikonv2"], required=True)
        if name == "profile":
            sub.add_argument("--graph", choices=["anatomical", "degree241"], default="anatomical")
        if name == "launch":
            sub.add_argument("--gpus", default=None)
            sub.add_argument("--background", action="store_true")
            sub.add_argument("--restart", action="store_true", help="Restart this campaign's identity-verified workers from selected checkpoints")
        if name == "evaluate":
            sub.add_argument("--run", type=Path)
            sub.add_argument("--destination", type=Path, help="New report directory inside the Ki67 Workspace")
        if name == "predict":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--images", type=Path, nargs="+", required=True)
            sub.add_argument("--save", type=Path, required=True)
        if name == "status":
            sub.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def command_context(args):
    output = resolve_output_reference(args.output)
    if args.command in MUTATING_COMMANDS:
        config = read_json(args.plan)
        from .campaign import validate_context
        output = validate_context(config, output)
    else:
        frozen = output / "manifests/campaign.json"
        plan = args.plan if args.plan is not None else frozen if frozen.is_file() else CONFIG / "campaign_separate_vit.json"
        config = read_json(plan)
    if args.command == "evaluate":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.destination = writable_location(args.destination or WORKSPACE / args.command / stamp)
    elif args.command == "predict":
        args.save = writable_location(args.save)
    return config, output


def main(argv=None):
    args = parse_arguments(argv)
    config, output = command_context(args)
    if args.command in ["extract", "profile"] and args.dataset not in config["datasets"]:
        raise ValueError("Requested dataset is not part of the selected campaign.")
    device = os.environ.get("FLYBRAIN_DEVICE", "cpu")
    result = None
    if args.command == "audit":
        if "joint" in config:
            from .joint_data import prepare as prepare_joint
            result = {"data": prepare_joint(config, output)}
        else:
            from .data import audit
            from .graphs import prepare
            result = {"data": audit(config, output), "graphs": prepare(config, output)}
    elif args.command == "download":
        from .features import download_models
        result = download_models(config, output)
    elif args.command == "extract":
        from .features import extract
        result = extract(config, output, args.dataset, args.representation, device=device)
    elif args.command == "profile":
        from .training import profile
        result = profile(config, output, args.dataset, args.representation, args.graph, device,
                         progress=lambda **r: print(json.dumps(r), flush=True) if r.get("segment", 0) % 25 == 0 else None)
    elif args.command in ["launch", "campaign", "worker", "monitor", "status"]:
        from . import campaign
        if args.command == "campaign":
            jobs = campaign.expand(config)
            write_json(output/"manifests/jobs.json", jobs)
            result = {"jobs": len(jobs), "training_runs": sum(j["kind"] == "train" for j in jobs),
                      "manifest": str(output/"manifests/jobs.json")}
        elif args.command == "launch":
            if not all((output/"manifests"/f"{d}.json").exists() for d in config["datasets"]):
                raise RuntimeError("Run audit before launching the campaign.")
            gpus = ([int(v) for v in args.gpus.split(",")] if args.gpus is not None
                    else config["runtime"]["gpu_indices"])
            if args.restart:
                campaign.restart_workers(output)
            result = campaign.launch(config, output, gpus, args.background)
        elif args.command == "worker":
            return campaign.worker(config, output)
        elif args.command == "monitor":
            return campaign.monitor(config, output)
        else:
            result = campaign.status(config, output)
    elif args.command == "evaluate":
        from .evaluation import summarize
        if args.run:
            from .evaluation import classification_metrics, regression_report
            import numpy as np
            best = read_json(args.run/"best.json")
            with np.load(args.run/best["prediction_path"], allow_pickle=False) as p:
                result = classification_metrics(p["y_true"], p["probabilities"]) if best["task"] in CLASS_TASKS else regression_report(p["y_true"], p["y_pred"], best["task"])
            write_json(args.destination/"evaluation_reconstructed.json", result)
        else:
            result = summarize(output, destination=args.destination)
    elif args.command == "predict":
        from .prediction import predict_images
        result = predict_images(args.checkpoint, args.images, args.save, output, device)
    if result is not None:
        print(json.dumps(result, indent=2, default=str), flush=True)
    return 0
