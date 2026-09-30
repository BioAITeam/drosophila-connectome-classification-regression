"""Complete RGB inputs and cached, frozen ViT CLS representations.

Model libraries are imported on demand. Reading manifests or raw images
does not initialize CUDA or download weights.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import time

import numpy as np
from PIL import Image

from flybrain.paths import TEMP
from . import encoders
from .encoders import preprocess_image
from .common import destination, portable_path, read_json, sha256, write_json
from .data import SPLITS, load_rows, resolve_image_path

MODELS = {
    "dinov2": {"repo_id": "facebook/dinov2-base", "dimension": 768},
    "phikonv2": {"repo_id": "owkin/phikon-v2", "dimension": 1024},
}


def _configure_cache(output: Path):
    """Keep mutable library caches separate from the frozen model artifacts."""
    cache = destination(TEMP / "ki67" / "huggingface" / ".location").parent
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_XET_CACHE"] = str(cache / "xet")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TORCH_HOME"] = str(destination(TEMP / "ki67" / "torch" / ".location").parent)


def download_models(config: dict, output: Path) -> dict:
    """Pin and download both pretrained backbones without importing PyTorch."""
    output = Path(output).resolve()
    _configure_cache(output)
    result = {}
    for representation in MODELS:
        print(f"Downloading pinned pretrained files for {representation}", flush=True)
        path, metadata = _snapshot(config, output, representation)
        result[representation] = {"snapshot": portable_path(path), **metadata}
        print(f"Ready: {representation} revision {metadata['revision']}", flush=True)
    return result


@contextmanager
def _lock(path: Path):
    path = destination(path)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_rgb(rows: list[dict]) -> np.ndarray:
    """Decode every native RGB value; no resize, crop, projection or pixel cache."""
    if not rows:
        raise ValueError("Cannot construct an empty RGB batch")
    batches = []
    for row in rows:
        with Image.open(resolve_image_path(row)) as image:
            image.load()
            if image.mode != "RGB" or image.size != (row["width"], row["height"]):
                raise ValueError(f'Image shape/mode changed after audit: {row["image_path"]}')
            batches.append(np.asarray(image, dtype=np.float32).reshape(-1) / np.float32(255.0))
    if len({array.shape for array in batches}) != 1:
        raise ValueError("Native RGB batches must contain images with equal dimensions")
    return np.stack(batches)


def fit_standardization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit column scaling on training CLS vectors only, retaining all dimensions."""
    train = np.asarray(train)
    if train.ndim != 2 or not len(train) or not np.isfinite(train).all():
        raise ValueError("Training features must be a nonempty finite matrix")
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, dtype=np.float64)
    std[std <= np.finfo(np.float32).eps] = 1.0
    return mean, std


def standardize(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = np.asarray((array - mean) / std, dtype=np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite standardized CLS features")
    return result


def _save_npy(path: Path, array: np.ndarray):
    path = destination(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _portable_model_config(content: bytes, repo_id: str) -> bytes:
    """Replace only an upstream machine path in a pretrained model config."""
    config = json.loads(content)
    name = config.get("_name_or_path")
    if isinstance(name, str) and (Path(name).is_absolute() or PureWindowsPath(name).is_absolute()):
        config["_name_or_path"] = repo_id
        return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return content


def _snapshot(config: dict, output: Path, representation: str) -> tuple[Path, dict]:
    from huggingface_hub import HfApi, snapshot_download

    descriptor = MODELS[representation]
    configured = config.get("vit_models", {}).get(representation, {})
    repo_id = configured.get("repo_id", descriptor["repo_id"])
    cache = destination(Path(output) / "cache" / "huggingface" / ".location").parent
    revision_path = cache / f"{representation}-revision.json"
    with _lock(cache / f"{representation}.lock"):
        if revision_path.exists():
            pinned = read_json(revision_path)
            if pinned["repo_id"] != repo_id:
                raise ValueError("Configured model differs from the pinned feature model")
            requested = configured.get("revision")
            if requested and requested != pinned["revision"]:
                raise ValueError("Requested model revision differs from the pinned revision")
            if pinned.get("file_sha256"):
                return verified_snapshot(output, representation, pinned), pinned
        else:
            info = HfApi().model_info(repo_id, revision=configured.get("revision", "main"))
            filenames = [item.rfilename for item in info.siblings]
            weights = [name for name in filenames if name.endswith(".safetensors")]
            if not weights:
                weights = [name for name in filenames if name.startswith("pytorch_model") and name.endswith(".bin")]
            if not weights:
                raise ValueError(f"No supported model weights in {repo_id}")
            pinned = {"repo_id": repo_id, "revision": info.sha,
                      "weight_files": sorted(weights), "dimension": descriptor["dimension"]}
            write_json(revision_path, pinned)
        download_cache = destination(TEMP / "ki67" / "huggingface" / "hub" / ".location").parent
        downloaded = Path(snapshot_download(
            repo_id, revision=pinned["revision"], cache_dir=str(download_cache),
            allow_patterns=["config.json", "preprocessor_config.json", "*.index.json", "LICENSE*", "README.md", *pinned["weight_files"]],
        ))
        snapshot = cache / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / pinned["revision"]
        for source in downloaded.rglob("*"):
            if not source.is_file():
                continue
            target = destination(snapshot / source.relative_to(downloaded))
            content = _portable_model_config(source.read_bytes(), repo_id) if source.name == "config.json" else None
            if target.exists():
                expected = hashlib.sha256(content).hexdigest() if content is not None else sha256(source)
                if sha256(target) != expected:
                    raise ValueError(f"Existing pinned model file differs from its download: {target.name}")
                continue
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            if content is None:
                shutil.copyfile(source, temporary)
            else:
                temporary.write_bytes(content)
            os.replace(temporary, target)
        file_hashes = {name: sha256(snapshot / name) for name in
                       ["config.json", "preprocessor_config.json", *pinned["weight_files"]]}
        if pinned.get("file_sha256") and pinned["file_sha256"] != file_hashes:
            raise ValueError("Pinned pretrained files failed checksum verification")
        pinned["file_sha256"] = file_hashes
        write_json(revision_path, pinned)
    return snapshot, pinned


def _row_identity(rows: list[dict]) -> str:
    payload = [(row["sample_id"], row["image_sha256"]) for row in rows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def verified_snapshot(output: Path, representation: str, pinned: dict) -> Path:
    """Read pinned local files without network access or cache metadata writes."""
    cache = Path(output) / "cache/huggingface"
    if read_json(cache / f"{representation}-revision.json") != pinned:
        raise ValueError("Pretrained snapshot differs from the training feature model")
    snapshot = cache / "hub" / ("models--" + pinned["repo_id"].replace("/", "--")) / "snapshots" / pinned["revision"]
    hashes = pinned.get("file_sha256", {})
    required = {"config.json", "preprocessor_config.json", *pinned.get("weight_files", [])}
    if not pinned.get("weight_files") or not required.issubset(hashes):
        raise ValueError("Incomplete pretrained model checksum provenance")
    for name, expected in hashes.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"Invalid pinned model filename: {name}")
        if not (snapshot / name).is_file() or sha256(snapshot / name) != expected:
            raise ValueError(f"Pinned pretrained file checksum mismatch: {name}")
    return snapshot


def extract(config: dict, output: Path, dataset: str, representation: str,
            device: str = "cpu", progress=None) -> dict:
    """Extract frozen CLS features once, with immutable provenance and train scaling."""
    if representation not in MODELS:
        raise ValueError(f"Only ViT representations require extraction: {representation}")
    output = Path(output).resolve()
    directory = output / "features" / dataset / representation
    destination(directory / ".extract.lock")
    rows_by_split = {split: load_rows(output, dataset, split) for split in SPLITS}
    identities = {split: _row_identity(rows) for split, rows in rows_by_split.items()}
    with _lock(directory / ".extract.lock"):
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["input_identity_sha256"] != identities:
                raise ValueError("Feature manifest no longer matches the audited image order/content")
            configured = config.get("vit_models", {}).get(representation, {})
            if configured.get("repo_id", MODELS[representation]["repo_id"]) != metadata["model"]["repo_id"]:
                raise ValueError("Feature cache model mismatch")
            if configured.get("revision") and configured["revision"] != metadata["model"]["revision"]:
                raise ValueError("Feature cache revision mismatch")
            for name, digest in metadata["artifact_sha256"].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f"Cached feature checksum mismatch: {name}")
            return metadata

        _configure_cache(output)
        import torch

        start = time.perf_counter()
        snapshot, pinned = _snapshot(config, output, representation)
        processor = read_json(snapshot / "preprocessor_config.json")
        mean, std = processor["image_mean"], processor["image_std"]
        model = encoders.frozen_model(snapshot, device)
        dimension = MODELS[representation]["dimension"]
        batch_size = int(config.get("extraction_batch_size", 16))
        if batch_size < 1:
            raise ValueError("Extraction batch size must be positive")
        arrays = {}
        split_times = {}
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for split, rows in rows_by_split.items():
            split_start = time.perf_counter()

            def report(completed):
                if progress:
                    progress({"stage": "extract", "dataset": dataset, "representation": representation,
                              "split": split, "completed": completed, "total": len(rows)})

            arrays[split] = encoders.encode_cls(
                model, [resolve_image_path(row) for row in rows],
                image_mean=mean, image_std=std, dimension=dimension,
                batch_size=batch_size, device=device,
                image_sha256=[row["image_sha256"] for row in rows], progress=report,
            )
            split_times[split] = time.perf_counter() - split_start
        center, scale = fit_standardization(arrays["train"])
        _save_npy(directory / "mean.npy", center)
        _save_npy(directory / "std.npy", scale)
        for split, array in arrays.items():
            _save_npy(directory / f"{split}.npy", standardize(array, center, scale))
        artifact_names = ["mean.npy", "std.npy", *(f"{split}.npy" for split in SPLITS)]
        metadata = {
            "schema_version": 1, "dataset": dataset, "representation": representation,
            "dimension": dimension, "model": pinned, "frozen": True,
            "input_identity_sha256": identities,
            "sample_ids": {split: [row["sample_id"] for row in rows] for split, rows in rows_by_split.items()},
            "preprocessing": {"size": [224, 224], "interpolation": "Pillow BICUBIC",
                              "crop": False, "rescale_factor": 1 / 255,
                              "image_mean": mean, "image_std": std,
                              "standardization": "per-CLS-dimension mean/std fitted on training split only"},
            "artifact_sha256": {name: sha256(directory / name) for name in artifact_names},
            "extraction_seconds": time.perf_counter() - start, "split_seconds": split_times,
            "device": str(device), "batch_size": batch_size,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "huggingface-hub", "Pillow", "numpy")},
        }
        write_json(metadata_path, metadata)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return metadata


def get_features(output: Path, dataset: str, representation: str, split: str) -> np.ndarray:
    if representation not in MODELS or split not in SPLITS:
        raise ValueError("Unknown ViT representation or split")
    directory = Path(output) / "features" / dataset / representation
    metadata = read_json(directory / "metadata.json")
    if (metadata.get("dataset") != dataset or metadata.get("representation") != representation
            or metadata.get("dimension") != MODELS[representation]["dimension"]
            or metadata.get("frozen") is not True):
        raise ValueError("Invalid cached feature provenance")
    rows = load_rows(output, dataset, split)
    if (metadata["input_identity_sha256"][split] != _row_identity(rows)
            or metadata["sample_ids"][split] != [row["sample_id"] for row in rows]):
        raise ValueError("Cached feature order/content differs from the current manifest")
    name = f"{split}.npy"
    expected_hash = metadata.get("artifact_sha256", {}).get(name)
    if expected_hash is None or sha256(directory / name) != expected_hash:
        raise ValueError(f"Cached feature checksum mismatch: {name}")
    result = np.load(directory / name, mmap_mode="r", allow_pickle=False)
    if result.dtype != np.float32 or result.shape != (len(rows), metadata["dimension"]):
        raise ValueError("Invalid cached feature matrix")
    if not np.isfinite(result).all():
        raise ValueError("Cached feature matrix contains non-finite values")
    return result


def encode_images(output: Path, dataset: str, representation: str, images: list[Path],
                  device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Encode new unannotated images using the exact fitted training transform."""
    if not images or batch_size < 1:
        raise ValueError("Provide images and a positive batch size")
    if representation not in MODELS:
        raise ValueError("encode_images requires a frozen ViT representation")
    output = Path(output).resolve()
    directory = output / "features" / dataset / representation
    metadata = read_json(directory / "metadata.json")
    if metadata["representation"] != representation or not metadata["frozen"]:
        raise ValueError("Invalid feature provenance")
    for name in ("mean.npy", "std.npy"):
        if sha256(directory / name) != metadata["artifact_sha256"][name]:
            raise ValueError(f"Training scaler checksum mismatch: {name}")
    _configure_cache(output)
    pinned = metadata["model"]
    snapshot = verified_snapshot(output, representation, pinned)
    import torch
    model = encoders.frozen_model(snapshot, device)
    mean = metadata["preprocessing"]["image_mean"]
    std = metadata["preprocessing"]["image_std"]
    result = encoders.encode_cls(model, images, image_mean=mean, image_std=std,
                                dimension=metadata["dimension"], batch_size=batch_size, device=device)
    center = np.load(directory / "mean.npy", allow_pickle=False)
    scale = np.load(directory / "std.npy", allow_pickle=False)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return standardize(result, center, scale)
