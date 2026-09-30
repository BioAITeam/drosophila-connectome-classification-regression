#!/usr/bin/env python3
"""Predict an Iris measurement with an archived validation-selected connectome."""
from pathlib import Path
import os
import subprocess
import sys

if __name__ == "__main__":
    if not os.environ.get("FLYBRAIN_PYTHON") and not os.environ.get("FLYBRAIN_ENV"):
        os.environ["FLYBRAIN_PYTHON"] = sys.executable
    raise SystemExit(subprocess.call(["bash", str(Path(__file__).with_name("_run.sh")), "regression", *sys.argv[1:]]))
