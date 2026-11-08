"""Train or report the published separate-dataset SHIDC-B-Ki-67-V2 anatomical models."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO = next(path for path in Path(__file__).resolve().parents
            if (path / "pyproject.toml").is_file() and (path / "6 Ki67").is_dir())
SOURCE = "6 Ki67/5 Outputs/shid-v2-20261031/binary"
PARTITION = "6 Ki67/5 Outputs/shid-v2-partition"
ARCHIVED_RUNS = "6 Ki67/5 Outputs/shid-separate-multiclass-20261101"
DEFAULT_OUTPUT = "6 Ki67/6 Workspace/shid-separate-multiclass-20261101"
GPU_IDS = list(range(8))
REPRESENTATIONS = ("dinov2", "phikonv2")
GRAPHS = ("anatomical",)
SEEDS = (1729, 2718, 3141)


def relative(path):
    return Path(path).absolute().relative_to(REPO).as_posix()


def resolve_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("Use repository-relative paths without parent traversal")
    resolved = (REPO / path).absolute()
    resolved.relative_to(REPO)
    return resolved


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, value):
    from ki67.common import write_json
    write_json(path, value)


def timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def make_plan(output):
    from ki67.campaign import expand
    plan = load(REPO / SOURCE / "manifests/campaign.json")
    plan["tasks"] = ["multiclass"]
    plan["output_root"] = relative(output)
    plan["seeds"] = list(SEEDS)
    plan["representations"] = list(REPRESENTATIONS)
    plan["graphs"] = list(GRAPHS)
    plan["runtime"].update(gpu_indices=GPU_IDS, threads_per_worker=4)
    if set(plan["datasets"]) != {"shid"} or "joint" in plan:
        raise ValueError("Only the separate SHID dataset may be trained")
    if plan["datasets"]["shid"]["root"] != "2 Data/1 Raw/ki67/shid_v2":
        raise ValueError("The source campaign must use SHIDC-B-Ki-67-V2")
    if plan["training"]["classification_epochs"] != 120:
        raise ValueError("This experiment requires 120 epochs")
    if plan["selection"]["classification"] != "validation_balanced_accuracy":
        raise ValueError("Checkpoint selection must use validation balanced accuracy")
    jobs = expand(plan)
    counts = Counter(job["kind"] for job in jobs)
    if dict(counts) != {"extract": 2, "profile": 2 * len(GRAPHS), "train": 6 * len(GRAPHS)}:
        raise ValueError("Unexpected experiment inventory")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Repeated job identifiers")
    return plan, jobs


def copy_inputs(output):
    source = REPO / SOURCE
    names = ["manifests/shid.json"]
    names += [f"features/shid/{rep}/{name}" for rep in REPRESENTATIONS
              for name in ("metadata.json", "mean.npy", "std.npy", "train.npy", "val.npy", "test.npy")]
    names += [f"graphs/{graph}.{suffix}" for graph in GRAPHS for suffix in ("npz", "json")]
    hashes = {}
    for name in names:
        original, target = source / name, output / name
        if not original.is_file() or original.is_symlink():
            raise ValueError(f"Expected an immutable regular input file: {name}")
        expected = sha256(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if sha256(target) != expected or sha256(original) != expected:
            raise ValueError(f"Input copy differs from its source: {name}")
        hashes[relative(original)] = expected
    record(output / "provenance/input_sha256.json", hashes)
    return hashes


def validate_inputs(output, plan):
    from ki67 import data, features, graphs
    from ki67.training import make_model
    import torch
    manifest = load(output / "manifests/shid.json")
    if manifest["source_root"] != plan["datasets"]["shid"]["root"]:
        raise ValueError("Source manifest root differs from V2")
    v2 = REPO / PARTITION
    with (v2 / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        expected = {row["image_id"]: row for row in csv.DictReader(stream)}
    patients = {split: set() for split in data.SPLITS}
    split_names = {"Train": "train", "Validation": "val", "Test": "test"}
    if len(expected) != 2356 or len(manifest["rows"]) != 2356:
        raise ValueError("Expected exactly 2356 SHID-V2 images")
    seen = set()
    for row in manifest["rows"]:
        stem = Path(row["image_path"]).stem
        original = expected[stem]
        if stem in seen or row["dataset"] != "shid":
            raise ValueError("Duplicate image or unexpected dataset")
        seen.add(stem)
        pairs = {"split": split_names[original["split"]], "patient_id": original["patient_id"],
                 "image_sha256": original["image_sha256"], "json_sha256": original["annotation_sha256"],
                 "positive_count": int(original["positive_cells"]),
                 "negative_count": int(original["negative_cells"]), "multiclass": int(original["class_id"])}
        if any(row[key] != value for key, value in pairs.items()):
            raise ValueError(f"Image differs from the V2 split manifest: {stem}")
        patients[row["split"]].add(row["patient_id"])
    if [len(patients[s]) for s in data.SPLITS] != [13, 3, 7]:
        raise ValueError("Expected 13, 3 and 7 patients")
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("A patient appears in different partitions")
    counts = {s: sum(row["split"] == s for row in manifest["rows"]) for s in data.SPLITS}
    if list(counts.values()) != [1325, 331, 700]:
        raise ValueError("The V2 image counts changed")
    for rep in REPRESENTATIONS:
        for split in data.SPLITS:
            features.get_features(output, "shid", rep, split)
    for graph in GRAPHS:
        graphs.load_graph(output, graph)
        model = make_model(output, graph, plan["training"], "cpu")
        if bool(torch.count_nonzero(model.gain)):
            raise ValueError("New models must start with unit synaptic gains")
    if (output / "runs").exists():
        raise ValueError("Training outputs must not exist before launch")
    record(output / "provenance/partition.json", {
        "dataset": "SHIDC-B-Ki-67-V2", "source_root": plan["datasets"]["shid"]["root"],
        "images": counts, "patient_ids": {s: sorted(patients[s]) for s in data.SPLITS},
        "patient_overlap": False, "input_pairs_match_v2_manifest": True,
        "feature_cache_checksums_and_sample_order_verified": True,
        "features_and_training_scalers_reused_without_change": True,
        "initial_gain_parameters_zero": True,
        "metadata_sha256": {name: sha256(v2 / name) for name in
                            ("split_manifest.csv", "split_summary.json", "patient_summary.csv")}})


def freeze_source(output):
    paths = list((REPO / "6 Ki67/2 Source").glob("*.py"))
    paths += list((REPO / "3 Connectome/flybrain").glob("*.py"))
    paths.append(Path(__file__))
    hashes = {}
    for original in paths:
        name = relative(original)
        target = output / "provenance/source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[name] = sha256(original)
        if sha256(target) != hashes[name]:
            raise ValueError(f"Source snapshot mismatch: {name}")
    record(output / "provenance/source_sha256.json", hashes)
    return hashes


def worker_environment(output):
    env = dict(os.environ)
    for name in ("KI67_SHID_ROOT", "KI67_BCDATA_ROOT", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(FLYBRAIN_REPO=str(REPO), FLYBRAIN_PYTHON=sys.executable,
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FLYBRAIN_GPU_LOCK_DIR=str(REPO / "10 temp/gpu-locks"),
               FLYBRAIN_TEMP_DIR=str(output / "runtime-cache"))
    return env


def gpu_sample(output):
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    with (output / "gpu_samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": timestamp(), "devices": list(csv.reader(text.splitlines()))}) + "\n")


def run_campaign(output, jobs):
    from ki67.campaign import process_identity
    from ki67.training import run_path
    expected = {relative(run_path(output, job) / "complete.json")
                for job in jobs if job["kind"] == "train"}
    started, first = time.perf_counter(), timestamp()
    (output / "logs").mkdir(exist_ok=True)
    command = [sys.executable, "-B", "-m", "ki67.bootstrap", "launch", "--plan", relative(output / "plan.json"),
               "--output", relative(output), "--gpus", ",".join(map(str, GPU_IDS)), "--background"]
    record(output / "provenance/launch_command.json", {"cwd": ".", "command": ["python", *command[1:]]})
    with (output / "logs/launch.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, cwd=REPO, env=worker_environment(output), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    while True:
        complete = {relative(path) for path in (output / "runs").glob("*/*/*/*/seed-*/complete.json")}
        if not complete <= expected:
            raise RuntimeError("An unexpected training run was found")
        launch = load(output / "status/launch.json")
        alive = [worker for worker in launch["workers"]
                 if process_identity(worker["pid"]) == worker.get("process_start_identity")]
        errors = sorted((output / "status/errors").glob("*.json"))
        status = {"timestamp": timestamp(), "complete": len(complete), "expected": len(expected),
                  "workers_alive": len(alive), "errors": [relative(path) for path in errors],
                  "elapsed_wall_seconds": time.perf_counter() - started}
        record(output / "progress.json", status)
        print(json.dumps(status), flush=True)
        gpu_sample(output)
        if errors:
            raise RuntimeError(f"Campaign error: {relative(errors[0])}")
        if complete == expected:
            break
        if not alive:
            raise RuntimeError("All workers exited before all requested runs completed")
        time.sleep(20)
    while any(process_identity(worker["pid"]) == worker.get("process_start_identity")
              for worker in launch["workers"]):
        time.sleep(2)
    completed = [load(REPO / path) for path in sorted(complete)]
    if any(row["last_epoch"] != 120 or row["attempt"] != 0 for row in completed):
        raise RuntimeError("Expected uninterrupted 120-epoch fits")
    result = {"started_at": first, "finished_at": timestamp(), "training_runs": len(completed),
              "elapsed_wall_seconds": time.perf_counter() - started,
              "sum_recorded_run_seconds": sum(row["elapsed_seconds"] for row in completed)}
    record(output / "timings/training.json", result)
    return result


def summarize_saved(source, output):
    """Rebuild the published anatomical groups from saved epoch predictions."""
    import numpy as np
    from .summarize_shid_v2 import load_manifest, collect_run, aggregate, csv_file, verify_patient_partition
    if not output.is_relative_to(REPO / "6 Ki67/6 Workspace") or output.exists():
        raise ValueError("Reports require a new Ki67 Workspace directory")
    hashes = {}
    manifest = load_manifest(source, "shid", hashes)
    verify_patient_partition(manifest)
    runs = [collect_run(source, manifest, dict(dataset="shid", representation=rep,
             task="multiclass", graph="anatomical", seed=seed), hashes, {})
            for rep in REPRESENTATIONS for seed in SEEDS]
    groups = [aggregate(runs, "shid", rep, "anatomical", "multiclass") for rep in REPRESENTATIONS]
    output.mkdir(parents=True)
    record(output / "summary.json", {"dataset": "SHIDC-B-Ki-67-V2", "status": "complete",
        "published_fits": 6, "groups": groups, "runs": runs, "source_sha256": hashes,
        "scope": "Two frozen extractors, anatomical graph, three seeds; original campaign also ran unpublished rewired controls"})
    csv_file(output / "multiclass_summary.csv", groups)
    print(json.dumps({"status": "complete", "fits": 6, "output": relative(output)}))


def main():
    global SOURCE, GPU_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report", action="store_true", help="Recompute the six published anatomical results without training")
    parser.add_argument("--source", default=SOURCE, help="Archived V2 binary feature inputs")
    parser.add_argument("--runs", default=ARCHIVED_RUNS, help="Archived separate multiclass fits for --report")
    parser.add_argument("--gpus", help="Explicit comma-separated physical GPU indices for training")
    args = parser.parse_args()
    SOURCE = args.source
    if args.gpus:
        GPU_IDS = [int(value) for value in args.gpus.split(",")]
        if not GPU_IDS or len(set(GPU_IDS)) != len(GPU_IDS) or min(GPU_IDS) < 0:
            parser.error("GPU indices must be nonnegative and unique")
    elif not (args.report or args.plan_only):
        parser.error("Training requires explicit --gpus")
    if args.report:
        summarize_saved(resolve_relative(args.runs), resolve_relative(args.output))
        return 0
    output = resolve_relative(args.output)
    if not output.is_relative_to(REPO / "6 Ki67/6 Workspace"):
        raise ValueError("Outputs must be inside the Ki67 Workspace")
    plan, jobs = make_plan(output)
    if args.plan_only:
        print(json.dumps({"plan": plan, "jobs": jobs}, indent=2))
        return 0
    if output.exists():
        raise FileExistsError("Use a new output directory without existing checkpoints")
    os.chdir(REPO)
    started, first = time.perf_counter(), timestamp()
    output.mkdir(parents=True)
    record(output / "plan.json", plan)
    record(output / "provenance/runtime.json", {
        "python": sys.version.split()[0], "versions": {name: importlib.metadata.version(name) for name in
        ("torch", "numpy", "scipy", "scikit-learn", "transformers", "Pillow")},
        "gpus": GPU_IDS, "seeds": list(SEEDS), "epochs": 120,
        "initialization": "Unit gains. No task-trained checkpoints are inherited.",
        "run_timer_scope": "Initial evaluation, optimization, all-split evaluations and artifact recording. Input and model preparation are excluded."})
    inputs = copy_inputs(output)
    validate_inputs(output, plan)
    sources = freeze_source(output)
    record(output / "timings/preparation.json", {"elapsed_wall_seconds": time.perf_counter() - started})
    run_campaign(output, jobs)
    for name, expected in {**inputs, **sources}.items():
        if sha256(REPO / name) != expected:
            raise RuntimeError(f"An immutable input or source file changed: {name}")
    record(output / "timings/campaign.json", {
        "started_at": first, "finished_at": timestamp(), "training_runs": 6 * len(GRAPHS),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "source_inputs_unchanged": True, "source_code_unchanged": True})
    print(f"All {6 * len(GRAPHS)} separate SHID-V2 multiclass fits completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
