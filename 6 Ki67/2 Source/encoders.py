"""Frozen local ViT loading and full-field CLS extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .common import sha256


def preprocess_image(image: Image.Image, mean, std) -> np.ndarray:
    """Resize the entire field to 224 square, then apply pretrained normalization."""
    mean, std = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("Expected three channel means and strictly positive standard deviations")
    resized = image.convert("RGB").resize((224, 224), resample=Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(((array - mean) / std).transpose(2, 0, 1))


def frozen_model(snapshot: Path, device: str = "cpu"):
    """Load verified local model files without network access or trainable weights."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model.requires_grad_(False)
    model.eval().to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Pretrained feature model must be entirely frozen")
    return model


def encode_cls(model, images: list[Path], *, image_mean, image_std,
               dimension: int, batch_size: int, device: str = "cpu",
               image_sha256: list[str] | None = None, progress=None) -> np.ndarray:
    """Encode images in input order, preserving float32 preprocessing and CLS values."""
    import torch

    if batch_size < 1 or dimension < 1:
        raise ValueError("Batch size and CLS dimension must be positive")
    if image_sha256 is not None and len(image_sha256) != len(images):
        raise ValueError("Image checksums must match the image list")
    result = np.empty((len(images), dimension), dtype=np.float32)
    with torch.inference_mode():
        for index in range(0, len(images), batch_size):
            batch = []
            for offset, path in enumerate(images[index:index + batch_size], start=index):
                if image_sha256 is not None and sha256(path) != image_sha256[offset]:
                    raise ValueError(f"Image changed after audit: {path}")
                with Image.open(path) as image:
                    batch.append(preprocess_image(image, image_mean, image_std))
            pixels = torch.from_numpy(np.stack(batch)).to(device)
            values = model(pixel_values=pixels).last_hidden_state[:, 0, :].float().cpu().numpy()
            if values.shape != (len(batch), dimension) or not np.isfinite(values).all():
                raise ValueError("Unexpected/nonfinite CLS representation")
            result[index:index + len(batch)] = values
            if progress is not None:
                progress(index + len(batch))
    return result
