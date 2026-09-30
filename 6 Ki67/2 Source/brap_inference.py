"""CPU image and folder inference with the archived joint BRAP-Net classifiers."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from flybrain.paths import KI67_WORKSPACE, REPO, repo_path
from .common import portable_path, read_json, sha256, writable_location, write_json

EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
CATALOG = REPO / "4 Trained Models/1 Catalogs/ki67-model-index.json"


def collect_images(source: Path, mode: str, recursive: bool = False) -> list[Path]:
    source = source.expanduser().resolve()
    if mode == "image":
        if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Provide an existing supported image: {source}")
        return [source]
    if mode != "folder" or not source.is_dir():
        raise ValueError(f"Provide an existing image folder: {source}")
    candidates = source.rglob("*") if recursive else source.iterdir()
    images = list(dict.fromkeys(path.resolve() for path in sorted(candidates)
                               if path.is_file() and path.suffix.lower() in EXTENSIONS))
    if not images:
        raise ValueError(f"No supported images found in {source}")
    return images


def select_model(catalog: dict, graph: str, seed: int | None) -> dict:
    graph = "degree241" if graph == "rewired" else graph
    choices = [row for row in catalog["models"]
               if row["dataset"] == "joint" and row["representation"] == "phikonv2"
               and row["task"] == "multiclass" and row["graph"] == graph
               and (seed is None or row["seed"] == seed)]
    if not choices:
        raise ValueError("The requested BRAP-Net classifier is absent from the model catalog")
    for row in choices:
        score = row["validation_value"]
        if (row["criterion"] != "validation_balanced_accuracy"
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 1):
            raise ValueError("BRAP-Net model selection requires a finite validation balanced accuracy")
    return max(choices, key=lambda row: (row["validation_value"], -row["seed"]))


def archive_path(value: str) -> Path:
    path = repo_path(value).resolve()
    if not path.is_relative_to((REPO / "zenodo/payload").resolve()):
        raise ValueError("BRAP-Net inference assets must be inside zenodo/payload")
    return path


def validate_model(row: dict, catalog: dict) -> tuple[Path, Path]:
    checkpoint = archive_path(row["checkpoint_path"])
    metadata_path = archive_path(row["best_metadata_path"])
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Restore the selected BRAP-Net checkpoint and metadata in zenodo/payload")
    if sha256(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError("BRAP-Net checkpoint checksum differs from the publication catalog")
    expected = catalog["resources"][row["best_metadata_path"]]["sha256"]
    if sha256(metadata_path) != expected:
        raise ValueError("BRAP-Net selection metadata checksum differs from the publication catalog")
    selected = read_json(metadata_path)
    if (any(selected[key] != row[key] for key in ("dataset", "representation", "task", "graph", "seed"))
            or selected["epoch"] != row["selected_epoch"]
            or selected["validation_metrics"]["balanced_accuracy"] != row["validation_value"]):
        raise ValueError("BRAP-Net checkpoint selection metadata is inconsistent")
    return checkpoint, archive_path(row["output_root"])


def prediction_destination(value: Path) -> Path:
    path = writable_location(value)
    if path.suffix != ".npz" or not path.is_relative_to(KI67_WORKSPACE.resolve()):
        raise ValueError("--save must be an .npz path inside 6 Ki67/6 Workspace")
    if any(path.with_suffix(suffix).exists() for suffix in (".npz", ".json", ".csv")):
        raise FileExistsError("Prediction output already exists; choose a new --save filename")
    return path


def export_csv(predictions: Path) -> Path:
    import numpy as np

    target = predictions.with_suffix(".csv")
    with np.load(predictions, allow_pickle=False) as values, target.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class_id", "predicted_class",
                         *[f"probability_{name}" for name in values["class_names"]]])
        for path, category, label, probabilities in zip(values["image_path"], values["y_pred"],
                                                       values["predicted_class"], values["probabilities"]):
            writer.writerow([path, int(category), label, *map(float, probabilities)])
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("image", "folder"):
        sub = commands.add_parser(mode)
        sub.add_argument(f"--{mode}", type=Path, required=True)
        if mode == "folder":
            sub.add_argument("--recursive", action="store_true", help="Include nested image folders")
        sub.add_argument("--graph", choices=("anatomical", "degree241", "rewired"), default="anatomical")
        sub.add_argument("--seed", type=int, choices=(1729, 2718, 3141),
                         help="Default: seed with highest saved validation balanced accuracy")
        sub.add_argument("--save", type=Path, required=True,
                         help="New .npz output inside 6 Ki67/6 Workspace; CSV and JSON are also saved")
    args = parser.parse_args(argv)
    try:
        images = collect_images(getattr(args, args.mode), args.mode, getattr(args, "recursive", False))
        save = prediction_destination(args.save)
        catalog = read_json(CATALOG)
        selected = select_model(catalog, args.graph, args.seed)
        checkpoint, output = validate_model(selected, catalog)
        os.environ.update(CUDA_VISIBLE_DEVICES="", FLYBRAIN_DEVICE="cpu", HF_HUB_OFFLINE="1",
                          TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                          HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        from flybrain.launcher import configure_runtime
        configure_runtime()
        import torch
        from .prediction import predict_images
        torch.set_num_threads(int(os.environ.get("FLYBRAIN_THREADS", "4")))
        summary = predict_images(checkpoint, images, save, output, device="cpu")
        summary.update(model_id=selected["id"], seed=selected["seed"], input_mode=args.mode,
                       recursive=getattr(args, "recursive", False),
                       seed_selection="explicit" if args.seed is not None else "highest_validation_balanced_accuracy",
                       validation_balanced_accuracy=selected["validation_value"],
                       csv=portable_path(export_csv(save)))
        write_json(save.with_suffix(".json"), summary)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
