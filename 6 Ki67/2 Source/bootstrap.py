"""Configure caches and acquire a GPU before importing torch or transformers."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import REPO, now, write_json

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", type=int)
    options, remaining = parser.parse_known_args()
    from .cli import parse_arguments, command_context
    args = parse_arguments(remaining)
    try:
        config, output = command_context(args)
    except ValueError as exc:
        parser.error(str(exc))
    os.environ.setdefault("FLYBRAIN_TEMP_DIR", str(REPO / "10 temp/ki67"))
    from flybrain.launcher import configure_runtime, reserve_gpu
    configure_runtime()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLYBRAIN_DEVICE"] = "cpu"
    lock = None
    if options.device == "cuda":
        from .campaign import allowed_gpus
        authorized = allowed_gpus(config, output)
        if options.gpu not in authorized:
            parser.error(f"This Ki67 plan permits physical GPUs {','.join(map(str, sorted(authorized)))} only.")
        while lock is None:
            if options.gpu not in allowed_gpus(config, output):
                parser.error("The execution policy no longer authorizes this GPU.")
            try:
                lock = reserve_gpu(options.gpu)
            except (RuntimeError, OSError, subprocess.SubprocessError):
                if not remaining or remaining[0] != "worker":
                    raise
                from .campaign import process_identity
                write_json(output / "status" / f"worker-gpu{options.gpu}.json",
                           {"pid": os.getpid(), "process_start_identity": process_identity(os.getpid()),
                            "gpu": options.gpu, "state": "waiting_for_idle_gpu", "heartbeat": now()})
                time.sleep(30)
    elif options.gpu is not None:
        parser.error("--gpu requires --device cuda.")
    os.environ["KI67_PHYSICAL_GPU"] = str(options.gpu) if options.gpu is not None else "cpu"
    try:
        from .cli import main as command_main
        return command_main(remaining)
    finally:
        if lock is not None:
            lock.close()

if __name__ == "__main__":
    sys.exit(main())
