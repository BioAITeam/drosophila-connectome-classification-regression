"""Run the SHIDC-B-Ki-67-V2 frozen-feature classification experiments."""
from __future__ import annotations

import argparse
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


REPO = next(p for p in Path(__file__).resolve().parents
            if (p / "pyproject.toml").is_file() and (p / "6 Ki67").is_dir())
DEFAULT_OUTPUT = "6 Ki67/6 Workspace/shid-v2-20261031"
V2_ROOT = "2 Data/1 Raw/ki67/shid_v2"
ARCHIVE = "6 Ki67/5 Outputs"
V2_ARCHIVE = ARCHIVE + "/shid-v2-20261031/binary"
PARTITION = ARCHIVE + "/shid-v2-partition"
GPU_IDS = list(range(8))
SEEDS = [1729, 2718, 3141]
REPRESENTATIONS = ["dinov2", "phikonv2"]
V2_POLICY = ("SHIDC-B-Ki-67-V2: mutually exclusive filename-derived patient groups "
             "in training, validation and test, preserving all prepared images and annotations")
JOINT_POLICY = ("BCData retains its published partitions. SHIDC-B-Ki-67-V2 has mutually "
                "exclusive patient groups across training, validation and test")


def relative(path):
    return Path(path).absolute().relative_to(REPO).as_posix()


def resolve_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError("Use a repository-relative path without parent traversal")
    result = (REPO / path).resolve()
    result.relative_to(REPO)
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path, value):
    from ki67.common import write_json
    write_json(path, value)


def load(path):
    return json.loads(Path(path).read_text())


def timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def check_partition():
    source = REPO / PARTITION
    summary = load(source / "split_summary.json")
    with (source / "split_manifest.csv").open(newline="") as handle:
        records = list(csv.DictReader(handle))
    groups = {s: set(summary["splits"][s]["patient_ids"])
              for s in ("Train", "Validation", "Test")}
    assert [summary["splits"][s]["images"] for s in groups] == [1325, 331, 700]
    assert list(map(len, groups.values())) == [13, 3, 7]
    assert len(set.union(*groups.values())) == 23
    assert not any(groups[a] & groups[b] for a, b in
                   [("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")])
    assert len(records) == 2356
    return {"root": V2_ROOT, "patient_ids": {s: sorted(v) for s, v in groups.items()},
            "metadata_sha256": {name: file_hash(source / name) for name in
                                ("split_summary.json", "split_manifest.csv", "patient_summary.csv")}}


def prepare_plans(root):
    binary = root / "binary"
    joint = root / "joint"
    plan_dir = root / "plans"
    binary_plan = load(REPO / "6 Ki67/1 Configuration/campaign_separate_vit.json")
    binary_plan["datasets"] = {"shid": {"root": V2_ROOT,
        "splits": {"train": "Train", "val": "Validation", "test": "Test"},
        "partition_policy": V2_POLICY}}
    binary_plan["tasks"] = ["binary5", "binary10", "binary30"]
    binary_plan["output_root"] = relative(binary)
    joint_plan = load(REPO / "6 Ki67/1 Configuration/campaign_joint.json")
    joint_plan["joint"]["source_output"] = relative(binary)
    joint_plan["joint"]["partition_policy"] = JOINT_POLICY
    joint_plan["output_root"] = relative(joint)
    for name, plan in [("binary", binary_plan), ("joint", joint_plan)]:
        plan["runtime"].update(gpu_indices=GPU_IDS, threads_per_worker=4)
        plan["seeds"] = SEEDS
        plan["representations"] = REPRESENTATIONS
        plan["graphs"] = ["anatomical", "degree241"]
        assert plan["training"]["classification_epochs"] == 120
        record(plan_dir / f"{name}.json", plan)
    return binary_plan, joint_plan


def archive_inputs(root, binary):
    source = REPO / ARCHIVE
    observed = {}
    for name in ["manifests/bcdata.json", "features/bcdata"]:
        src, dst = source / name, binary / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
            files = [p for p in src.rglob("*") if p.is_file()]
        else:
            shutil.copy2(src, dst)
            files = [src]
        for path in files:
            digest = file_hash(path)
            target = binary / path.relative_to(source)
            assert file_hash(target) == digest
            observed[relative(path)] = digest
    cache = binary / "cache/huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    # Mutable extractor locks stay in Workspace; the shared weight tree is read-only.
    for representation in REPRESENTATIONS:
        shutil.copy2(source / "cache/huggingface" / f"{representation}-revision.json",
                     cache / f"{representation}-revision.json")
    (cache / "hub").symlink_to(os.path.relpath(source / "cache/huggingface/hub", cache), target_is_directory=True)
    record(root / "provenance/archive_sources.json", observed)


def environment(root):
    env = dict(os.environ)
    for name in ("KI67_SHID_ROOT", "KI67_BCDATA_ROOT", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    env.update(FLYBRAIN_REPO=str(REPO), FLYBRAIN_PYTHON=sys.executable,
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               FLYBRAIN_GPU_LOCK_DIR=str(REPO / "10 temp/gpu-locks"),
               FLYBRAIN_TEMP_DIR=str(root / "runtime-cache"))
    return env


def cli(root, name, arguments, env):
    log = root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-m", "ki67.bootstrap", *map(str, arguments)]
    with log.open("w") as stream:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    return proc, log, command


def finish(processes):
    for proc, log, command in processes:
        if proc.wait() != 0:
            raise RuntimeError(f"Command failed. Read {relative(log)}")


def gpu_sample(root, phase):
    query = ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"]
    text = subprocess.check_output(query, text=True)
    with (root / "gpu_samples.jsonl").open("a") as stream:
        stream.write(json.dumps({"timestamp": timestamp(), "phase": phase,
                                 "devices": list(csv.reader(text.splitlines()))}) + "\n")


def run_phase(root, name, count, env):
    from ki67.campaign import process_identity
    output = root / name
    started = time.perf_counter()
    first = timestamp()
    finish([cli(root, f"launch-{name}", ["launch", "--plan", relative(root / "plans" / f"{name}.json"),
              "--output", relative(output), "--gpus", ",".join(map(str, GPU_IDS)), "--background"], env)])
    while True:
        completed = list((output / "runs").glob("*/*/*/*/seed-*/complete.json"))
        errors = list((output / "status/errors").glob("*.json"))
        launch = load(output / "status/launch.json")
        alive = [w for w in launch["workers"] if
                 process_identity(w["pid"]) == w.get("process_start_identity")]
        status = {"phase": name, "timestamp": timestamp(), "completed": len(completed),
                  "expected": count, "workers_alive": len(alive),
                  "errors": [relative(p) for p in errors],
                  "elapsed_wall_seconds": time.perf_counter() - started}
        record(root / "progress.json", status)
        print(json.dumps(status), flush=True)
        gpu_sample(root, name)
        if len(completed) == count:
            break
        if not alive:
            raise RuntimeError(f"Campaign ended with {len(completed)}/{count} completed runs")
        if errors:
            raise RuntimeError(f"Campaign error recorded in {relative(errors[0])}")
        time.sleep(30)
    while any(process_identity(w["pid"]) == w.get("process_start_identity")
              for w in launch["workers"]):
        time.sleep(2)
    result = {"started_at": first, "finished_at": timestamp(),
              "elapsed_wall_seconds": time.perf_counter() - started, "training_runs": count,
              "sum_recorded_run_seconds": sum(load(p)["elapsed_seconds"] for p in completed)}
    record(root / "timings" / f"{name}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=V2_ARCHIVE, help="Archived V2 binary inputs, relative to the repository")
    parser.add_argument("--gpus", required=True, help="Explicit comma-separated physical GPU indices; workers wait for availability")
    parser.add_argument("--from-images", action="store_true", help="Audit local V2 image/JSON pairs and extract frozen features again")
    args = parser.parse_args()
    global GPU_IDS
    GPU_IDS = [int(value) for value in args.gpus.split(",")]
    if not GPU_IDS or len(set(GPU_IDS)) != len(GPU_IDS) or min(GPU_IDS) < 0:
        parser.error("GPU indices must be nonnegative and unique")
    archived_source = resolve_relative(args.source)
    root = resolve_relative(args.output)
    if not root.is_relative_to(REPO / "6 Ki67/6 Workspace"):
        raise ValueError("The experiment must have a new Ki67 Workspace directory")
    if root.exists():
        raise FileExistsError("Use a new destination. Existing runs are never replaced")
    from ki67 import data, graphs, features, joint_data
    from ki67.common import write_json
    os.chdir(REPO)
    started, first = time.perf_counter(), timestamp()
    root.mkdir(parents=True)
    record(root / "provenance/partition.json", check_partition())
    binary_plan, joint_plan = prepare_plans(root)
    env = environment(root)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "numpy", "scipy", "scikit-learn", "transformers", "Pillow"]}
    record(root / "provenance/runtime.json", {"python": sys.version.split()[0],
           "versions": versions, "gpus": GPU_IDS, "seeds": SEEDS, "epochs": 120,
           "initialization": "Unit gains, no inherited task-trained checkpoints",
           "run_timer_scope": "Initialization evaluation, optimization, all-split evaluations, and artifact recording; excludes input/model preparation"})
    code = list((REPO / "6 Ki67/2 Source").glob("*.py")) + list((REPO / "3 Connectome/flybrain").glob("*.py"))
    code.append(Path(__file__))
    code_hashes = {}
    for src in code:
        rel = relative(src)
        dst = root / "provenance/source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        code_hashes[rel] = file_hash(src)
    record(root / "provenance/source_sha256.json", code_hashes)
    binary, joint = root / "binary", root / "joint"
    archive_inputs(root, binary)
    print("Auditing new patient partitions", flush=True)
    audit_started = time.perf_counter()
    os.environ.pop("KI67_SHID_ROOT", None)
    if args.from_images:
        data.audit(binary_plan, binary)
        graphs.prepare(binary_plan, binary)
    else:
        for name in ("manifests/shid.json", "features/shid", "graphs"):
            original, target = archived_source / name, binary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if original.is_dir():
                shutil.copytree(original, target)
            else:
                shutil.copy2(original, target)
            for path in ([original] if original.is_file() else original.rglob("*")):
                if path.is_file() and file_hash(path) != file_hash(target if original.is_file() else target / path.relative_to(original)):
                    raise ValueError("Archived V2 input copy changed")
    manifest = load(binary / "manifests/shid.json")
    assert len(manifest["rows"]) == 2356
    assert all(not ids for values in manifest["summary"]["split_overlap"].values() for ids in values.values())
    assert [manifest["summary"]["splits"][s]["images"] for s in data.SPLITS] == [1325, 331, 700]
    for rep in REPRESENTATIONS:
        for split in data.SPLITS:
            features.get_features(binary, "bcdata", rep, split)
    record(root / "timings/data_audit.json", {"elapsed_wall_seconds": time.perf_counter() - audit_started})
    print("Preparing frozen V2 features", flush=True)
    feature_started = time.perf_counter()
    processes = [cli(root, f"extract-{rep}", ["--device", "cuda", "--gpu", str(gpu), "extract",
        "--plan", relative(root / "plans/binary.json"), "--output", relative(binary),
        "--dataset", "shid", "--representation", rep], env) for gpu, rep in zip((GPU_IDS * 2)[:2], REPRESENTATIONS)] if args.from_images else []
    finish(processes)
    record(root / "timings/feature_extraction.json", {"elapsed_wall_seconds": time.perf_counter()-feature_started,
        "reused_archived_features": not args.from_images,
        "source_extraction_seconds": {rep: load(binary / "features/shid" / rep / "metadata.json")["extraction_seconds"]
                            for rep in REPRESENTATIONS}})
    joint_started = time.perf_counter()
    joint_data.prepare(joint_plan, joint)
    record(root / "timings/joint_preparation.json", {"elapsed_wall_seconds": time.perf_counter()-joint_started})
    for name, count in [("binary", 36), ("joint", 12)]:
        run_phase(root, name, count, env)
    for src, digest in load(root / "provenance/archive_sources.json").items():
        assert file_hash(REPO / src) == digest, src
    for src, digest in code_hashes.items():
        assert file_hash(REPO / src) == digest, src
    record(root / "timings/campaign.json", {"started_at": first, "finished_at": timestamp(),
        "elapsed_wall_seconds": time.perf_counter()-started, "training_runs": 48,
        "original_archive_files_unchanged": True})
    print("All 48 experiments completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
