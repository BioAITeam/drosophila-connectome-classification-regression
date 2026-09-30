"""Select CPU/GPU and configure local caches before importing numerical libraries."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .paths import REPO, TEMP, RAW, allowed_derived_write

GPU_LOCK_DIR = Path(os.environ.get("FLYBRAIN_GPU_LOCK_DIR", TEMP / "gpu-locks")).expanduser()
if not GPU_LOCK_DIR.is_absolute():
    GPU_LOCK_DIR = REPO / GPU_LOCK_DIR


def configure_runtime():
    """Keep temporary/cache writes in the configured project temporary directory."""
    temporary = allowed_derived_write(TEMP)
    if temporary.is_relative_to(RAW):
        raise ValueError("The temporary directory must not be inside immutable raw data.")
    directories = {
        "TMPDIR": temporary / "tmp",
        "XDG_CACHE_HOME": temporary / "cache",
        "MPLCONFIGDIR": temporary / "cache/matplotlib",
        "TORCH_HOME": temporary / "cache/torch",
        "HF_HOME": temporary / "cache/huggingface",
        "HF_HUB_CACHE": temporary / "cache/huggingface/hub",
        "HF_XET_CACHE": temporary / "cache/huggingface/xet",
        "HF_ASSETS_CACHE": temporary / "cache/huggingface/assets",
        "NUMBA_CACHE_DIR": temporary / "cache/numba",
        "CUDA_CACHE_PATH": temporary / "cache/cuda",
        "TRITON_CACHE_DIR": temporary / "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": temporary / "cache/torchinductor",
    }
    for variable, path in directories.items():
        allowed_derived_write(path).mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    threads = int(os.environ.get("FLYBRAIN_THREADS", "4"))
    if threads < 1:
        raise ValueError("FLYBRAIN_THREADS must be a positive integer.")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True, timeout=15)
    rows = csv.reader(query(["--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu"]).splitlines())
    devices = []
    for row in rows:
        index, uuid, free, total, util = [s.strip() for s in row]
        devices.append(dict(index=int(index), uuid=uuid, free=float(free), total=float(total), utilization=float(util)))
    processes = query(["--query-compute-apps=gpu_uuid,pid,used_memory"])
    occupied = {row[0].strip() for row in csv.reader(processes.splitlines()) if row}
    for device in devices:
        device["free_for_work"] = (
            device["uuid"] not in occupied and device["total"] > 0
            and device["free"] / device["total"] >= .90 and device["utilization"] <= 10
        )
    return devices


def reserve_gpu(requested=None):
    """Hold an advisory GPU lock and recheck idleness before exposing the device."""
    lockdir = allowed_derived_write(GPU_LOCK_DIR)
    if lockdir.is_relative_to(RAW):
        raise ValueError("GPU locks must not be stored inside immutable raw data.")
    lockdir.mkdir(parents=True, exist_ok=True)
    devices = gpu_snapshot()
    for device in sorted(devices, key=lambda d: (-d["free"], d["index"])):
        if not device["free_for_work"] or (requested is not None and device["index"] != requested):
            continue
        handle = (lockdir / f"{device['uuid']}.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        try:
            # Other cluster users may not use these advisory locks.
            fresh = next((d for d in gpu_snapshot() if d["uuid"] == device["uuid"]), None)
        except BaseException:
            handle.close()
            raise
        if fresh is None or not fresh["free_for_work"]:
            handle.close()
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        os.environ["FLYBRAIN_DEVICE"] = "cuda:0"
        os.environ["FLYBRAIN_GPU_RECORD"] = json.dumps(fresh)
        return handle
    raise RuntimeError(
        "No idle GPU is available (no active CUDA processes, at least 90% memory free, utilization at most 10%)."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--gpu", type=int)
    options, remaining = parser.parse_known_args(argv)
    if options.gpu is not None and (options.gpu < 0 or options.device == "cpu"):
        parser.error("--gpu requires --device cuda or auto and a nonnegative physical GPU index.")
    try:
        configure_runtime()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLYBRAIN_DEVICE"] = "cpu"
    os.environ.pop("FLYBRAIN_GPU_RECORD", None)
    lock = None
    if options.device != "cpu":
        try:
            lock = reserve_gpu(options.gpu)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            if options.device == "cuda":
                parser.error(str(exc))
            print(f"GPU unavailable: {exc}; --device auto is continuing on CPU.", file=sys.stderr)
    try:
        from flybrain.cli import main as cli_main
        return cli_main(remaining)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
