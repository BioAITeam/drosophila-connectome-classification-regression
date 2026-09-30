"""Annotation-free image inference from the single selected Ki67 checkpoint."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from . import features
from .common import CLASS_TASKS, TASKS, atomic_npz, digest, portable_path, read_json, resolve_output_reference, sha256, write_json
from flybrain.paths import repo_path
from .training import make_model


def predict_images(checkpoint: Path, images: list[Path], save: Path, output: Path,
                   device: str = "cpu") -> dict:
    """Predict images without reading annotations or fitting any preprocessing."""
    from .common import writable_location
    checkpoint, output, save = repo_path(checkpoint), resolve_output_reference(output), writable_location(save)
    paths = [repo_path(path).resolve() for path in images]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("Provide at least one existing image")
    started = time.perf_counter()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = saved["spec"]
    if digest(spec) != saved["spec_digest"]:
        raise ValueError("Checkpoint configuration digest mismatch")
    if spec.get("task") not in TASKS:
        raise ValueError("Checkpoint declares an unsupported prediction task")
    if sha256(output / "graphs" / f"{spec['graph']}.npz") != spec["graph_sha256"]:
        raise ValueError("Checkpoint graph checksum mismatch")
    representation = spec["representation"]
    if representation != "rgb":
        metadata_path = output / "features" / spec["dataset"] / representation / "metadata.json"
        if sha256(metadata_path) != spec["feature_metadata_sha256"]:
            raise ValueError("Checkpoint feature metadata checksum mismatch")
        if spec.get("joint_training"):
            from .joint_data import encode_images
            vectors = encode_images(output, representation, paths, device=device,
                                    batch_size=int(spec["training"]["microbatch_size"]))
        else:
            vectors = features.encode_images(output, spec["dataset"], representation,
                                             paths, device=device,
                                             batch_size=int(spec["training"]["microbatch_size"]))
    else:
        vectors = None
    model = make_model(output, spec["graph"], spec["training"], device)
    gain = saved["gain"]
    if (not isinstance(gain, torch.Tensor) or gain.dtype != model.gain.dtype
            or gain.shape != model.gain.shape or not torch.isfinite(gain).all()):
        raise ValueError("Checkpoint gains have an invalid dtype, shape or non-finite values")
    with torch.no_grad():
        model.gain.copy_(saved["gain"].to(device))
    model.eval()
    task = spec["task"]
    scale = float(saved["target_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Checkpoint target scale must be positive and finite")
    result = []
    # Per-image evaluation accommodates different native RGB dimensions.
    with torch.inference_mode():
        for index, path in enumerate(paths):
            if vectors is None:
                with Image.open(path) as image:
                    row = {"image_path": str(path), "width": image.width, "height": image.height}
                array = features.load_rgb([row])
            else:
                array = vectors[index:index + 1]
            x = torch.as_tensor(array, dtype=torch.float32, device=device)
            value = model(x)[:, :len(CLASS_TASKS[task])].softmax(1) if task in CLASS_TASKS else model.regression(x) * scale
            result.append(value.cpu().numpy())
    predictions = np.concatenate(result)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite image predictions")
    payload = {
        "sample_id": np.asarray([f"prediction/{index:06d}/{path.name}" for index, path in enumerate(paths)]),
        "image_path": np.asarray([portable_path(path) for path in paths]),
        "image_sha256": np.asarray([sha256(path) for path in paths]),
        "epoch": np.asarray(saved["epoch"]),
    }
    if task in CLASS_TASKS:
        categories = predictions.argmax(1)
        names = np.asarray(CLASS_TASKS[task])
        payload.update(probabilities=predictions, y_pred=categories, class_names=names,
                       predicted_class=names[categories])
    else:
        clipped = np.clip(predictions, 0, 100) if task == "pi" else np.maximum(predictions, 0)
        payload.update(y_pred=predictions, y_pred_clipped=clipped)
        if task == "pi":
            categories = np.where(clipped < 16, 0, np.where(clipped < 30, 1, 2)).astype(np.int64)
            payload.update(pi_multiclass=categories,
                           pi_class_name=np.asarray(CLASS_TASKS["multiclass"])[categories],
                           pi_binary5=(clipped > 5).astype(np.int64),
                           pi_binary10=(clipped > 10).astype(np.int64),
                           pi_binary30=(clipped >= 30).astype(np.int64))
    atomic_npz(save, **payload)
    summary = {
        "checkpoint": portable_path(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "selected_epoch": int(saved["epoch"]), "dataset": spec["dataset"],
        "representation": representation, "graph": spec["graph"], "task": task,
        "images": len(paths), "predictions": portable_path(save), "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "annotations_required": False, "normalization_refitted": False,
    }
    if spec.get("joint_training"):
        summary.update(architecture_family="BRAP-Net", training_datasets=spec["joint_training"]["datasets"],
                       dataset_origin_required=False)
    write_json(save.with_suffix(".json"), summary)
    return summary
