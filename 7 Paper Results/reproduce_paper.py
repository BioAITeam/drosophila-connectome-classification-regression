#!/usr/bin/env python3
"""Reconstruct paper figures and tables from archived measurements on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "3 Connectome"))
from flybrain.paths import INTERMEDIATE, PAPER_OUTPUTS, allowed_derived_write


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--family", choices=("all", "iris", "ki67"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.tag):
        parser.error("Use letters, numbers, hyphens and underscores in --tag")
    output = allowed_derived_write(PAPER_OUTPUTS / args.tag)
    staging = allowed_derived_write(INTERMEDIATE / "paper-reproduction" / args.tag)
    ki = REPO / "6 Ki67/5 Outputs"
    original = REPO / "5 Iris/5 Outputs/paper-results/results-20260914T025839Z-ilnj5d7n"
    iris = staging / "iris"
    joint = staging / "ki67/joint"
    binary = staging / "ki67/binary-confusions"
    table = staging / "ki67/binary-table"
    rgb = staging / "ki67/rgb"
    commands = []
    sources = []
    tables = []
    if args.family in {"all", "iris"}:
        commands.extend([
            ["7 Paper Results/1 Generators/condensed_figures.py", "--classification", relative(original / "classification"),
             "--regression", relative(original / "regression"), "--output", relative(iris)],
            ["7 Paper Results/1 Generators/iris_tables.py", "--output", relative(staging / "iris-tables")],
        ])
        sources.extend((number, iris) for number in range(1, 7))
        tables.extend((number, staging / "iris-tables" / f"Table-{number}.tex") for number in (1, 2))
    if args.family in {"all", "ki67"}:
        commands.extend([
            ["-m", "ki67.paper_joint", "--outputs", relative(ki / "joint-bcdata-shid-v1"), "--destination", relative(joint)],
            ["-m", "ki67.paper_binary_confusions", "--outputs", relative(ki), "--destination", relative(binary),
             "--combined", "--first-figure-number", "9"],
            ["-m", "ki67.paper_binary_table", "--summary", relative(ki / "paper/summary.json"),
             "--destination", relative(table)],
            ["-m", "ki67.paper_rgb", "--outputs", relative(ki), "--destination", relative(rgb),
             "--datasets", "bcdata", "shid", "--figure-number", "10"],
        ])
        sources.extend([(7, joint), (8, joint), (9, binary), (10, rgb)])
        tables.extend([(3, joint / "table-3.tex"), (4, table / "table-4.tex"), (5, rgb / "table.tex")])
    if args.dry_run:
        print(json.dumps({"output": relative(output), "intermediate": relative(staging), "commands": commands}, indent=2))
        return
    for folder in (output, staging):
        if folder.exists():
            raise FileExistsError(f"Choose a fresh tag; output already exists: {folder}")
    if not (REPO / "2 Data/3 Processed").is_dir():
        raise FileNotFoundError("Restore zenodo/payload and run 4 Trained Models/artifacts.py link first")
    output.mkdir(parents=True)
    cache = allowed_derived_write(staging / "cache")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               PYTHONPATH=str(REPO / "3 Connectome"),
               FLYBRAIN_REPO=str(REPO), FLYBRAIN_TEMP_DIR=str(cache))
    for key in ("TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        path = allowed_derived_write(cache / key.lower())
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for command in commands:
        print("Running:", shlex.join(command), flush=True)
        subprocess.run([sys.executable, "-B", *command], cwd=REPO, env=env, check=True)
    manifest = {"family": args.family, "commands": commands, "figures": {}, "tables": {}}
    for number, folder in sources:
        for extension in ("pdf", "png"):
            source = folder / f"Figure-{number}.{extension}"
            destination = allowed_derived_write(output / source.name)
            shutil.move(source, destination)
            manifest["figures"][source.name] = {"intermediate_directory": relative(folder), "sha256": digest(destination)}
    for number, source in tables:
        destination = allowed_derived_write(output / f"Table-{number}.tex")
        shutil.move(source, destination)
        manifest["tables"][destination.name] = {"intermediate_directory": relative(source.parent), "sha256": digest(destination)}
    allowed_derived_write(output / "reproduction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Reproduced artifacts: {relative(output)}")


if __name__ == "__main__":
    main()
