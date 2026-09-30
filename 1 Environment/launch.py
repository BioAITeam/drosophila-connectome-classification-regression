#!/usr/bin/env python3
"""Source-checkout wrapper for the same launcher as the installed flybrain CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3 Connectome"))

from flybrain.launcher import configure_runtime, gpu_snapshot, main, reserve_gpu

if __name__ == "__main__":
    main()
