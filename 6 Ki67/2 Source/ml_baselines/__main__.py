"""Command-line entry point for CPU-only Ki-67 ML reproduction."""
from __future__ import annotations

import argparse
import importlib
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "audit", "run", "repeat"),
                        help="report/audit read saved results; run/repeat explicitly train models")
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    modules = {"report": "make_report", "audit": "data", "run": "run_baselines", "repeat": "repeat_selected"}
    sys.argv = [sys.argv[0], *(["run"] if args.command == "run" else []), *remaining]
    importlib.import_module(f"{__package__}.{modules[args.command]}").main()


if __name__ == "__main__":
    main()
