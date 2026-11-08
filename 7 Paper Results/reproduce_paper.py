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
    classification = staging / "ki67/classification-analysis"
    binary = staging / "ki67/binary-confusions"
    table = staging / "ki67/binary-table"
    rgb = staging / "ki67/rgb"
    workspace = REPO / "6 Ki67/6 Workspace/paper-reproduction" / args.tag
    v2 = workspace / "shid-v2"
    cross = workspace / "cross-dataset"
    fine = workspace / "phikon-finetuning"
    ml = workspace / "ml-baselines"
    revised = staging / "ki67/revised-tables"
    commands = []
    sources = []
    schematic = staging / "schematic"
    iris_controls = staging / "iris-controls"
    shid_fine = workspace / "phikon-shid-finetuning"
    fine_cm = staging / "ki67/fine-confusions"
    xai = staging / "ki67/local-explanations"
    literature = staging / "ki67/literature"
    tables = []
    if args.family in {"all", "iris"}:
        commands.extend([
            ["7 Paper Results/1 Generators/condensed_figures.py", "--classification", relative(original / "classification"),
             "--regression", relative(original / "regression"), "--output", relative(iris)],
            ["7 Paper Results/1 Generators/iris_tables.py", "--output", relative(staging / "iris-tables")],
            ["5 Iris/2 Execution/topology_controls/paper_tables.py", "--archive",
             "5 Iris/5 Outputs/campaigns/iris-topology-ml", "--output", relative(iris_controls)],
            ["7 Paper Results/1 Generators/schematic/build_schematic.py", "--output", relative(schematic)],
        ])
        sources.extend((number + 1, iris / f"Figure-{number}") for number in range(1, 7))
        sources.append((1, schematic / "connectome-learning-overview"))
        tables.extend([(2, staging / "iris-tables/Table-1.tex"), (4, staging / "iris-tables/Table-2.tex"),
                       (3, iris_controls / "Table-3.tex"), (15, iris_controls / "Table-15.tex")])
    if args.family in {"all", "ki67"}:
        commands.extend([
            ["-m", "ki67.paper_joint", "--outputs", relative(ki / "joint-bcdata-shid-v1"), "--destination", relative(joint)],
            ["-m", "ki67.paper_classification_analysis", "--outputs", relative(ki),
             "--destination", relative(classification), "--resamples", "10000", "--seed", "1729"],
            ["-m", "ki67.paper_binary_confusions", "--outputs", relative(ki), "--destination", relative(binary),
             "--combined", "--first-figure-number", "9"],
            ["-m", "ki67.paper_binary_table", "--summary", relative(ki / "paper/summary.json"),
             "--destination", relative(table)],
            ["-m", "ki67.paper_rgb", "--outputs", relative(ki), "--destination", relative(rgb),
             "--datasets", "bcdata", "shid", "--figure-number", "10"],
            ["-m", "ki67.summarize_shid_v2", "--output", relative(v2)],
            ["-m", "ki67.run_cross_dataset", "--output", relative(cross)],
            ["-m", "ki67.ml_baselines", "report", "--output", relative(ml)],
            ["-m", "ki67.phikon_finetuning", "report", "--output", relative(fine)],
            ["-m", "ki67.paper_revision_tables", "--joint-summary", relative(joint / "metrics_summary.json"),
             "--v2-summary", relative(v2 / "results.json"),
             "--cross-summary", relative(cross / "cross_dataset_results.json"),
             "--finetune-summary", relative(fine / "summary.json"), "--destination", relative(revised)],
            ["-m", "ki67.phikon_finetuning.shid", "report", "--output", relative(shid_fine)],
            ["-m", "ki67.phikon_finetuning.render_confusions", "--output", relative(fine_cm)],
            ["-m", "ki67.phikon_xai", "render", "--output", relative(xai)],
            ["7 Paper Results/1 Generators/literature_comparison.py", "--finetune-summary",
             relative(shid_fine / "shid-finetuning-summary.json"), "--output", relative(literature)],
        ])
        sources.extend([(8, joint / "Figure-7"), (9, joint / "Figure-8"),
                        (10, binary / "Figure-9"), (11, rgb / "Figure-10"),
                        (12, fine_cm / "Figure-12"), (13, xai / "figures/Figure-13")])
        tables.extend([(6, classification / "Table-4.tex"), (7, classification / "Table-5.tex"),
                       (10, rgb / "table.tex"), (16, literature / "Table-16.tex")])
        tables.extend((target, revised / f"Table-{source}.tex")
                      for target, source in [(1, 1), (5, 4), (8, 7), (9, 8), (14, 13)])
        tables.extend((n + 1, ml / f"Table-{n}.tex") for n in (10, 11, 12))
    if args.dry_run:
        print(json.dumps({"output": relative(output), "intermediate": relative(staging), "commands": commands}, indent=2))
        return
    for folder in (output, staging, workspace):
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
    for number, stem in sources:
        for extension in ("pdf", "png"):
            source = stem.with_suffix("." + extension)
            destination = allowed_derived_write(output / f"Figure-{number}.{extension}")
            shutil.move(source, destination)
            manifest["figures"][destination.name] = {"intermediate_directory": relative(source.parent), "sha256": digest(destination)}
    for number, source in tables:
        destination = allowed_derived_write(output / f"Table-{number}.tex")
        shutil.move(source, destination)
        manifest["tables"][destination.name] = {"intermediate_directory": relative(source.parent), "sha256": digest(destination)}
    allowed_derived_write(output / "reproduction.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Reproduced artifacts: {relative(output)}")


if __name__ == "__main__":
    main()
