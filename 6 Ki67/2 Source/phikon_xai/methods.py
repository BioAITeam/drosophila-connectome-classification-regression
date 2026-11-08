"""Image explanations for the complete fine-tuned Phikon-v2 connectome model.

These functions do not load checkpoints, choose cases, or create figures. They
return NumPy arrays and JSON-compatible diagnostics for the caller to save.
SHAP and LIME explain a class probability using the same superpixels and the
same replacement image: each superpixel replaced by its own mean RGB value.
Grad-CAM explains the target logit through the complete prediction function.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import time
from typing import Callable

import numpy as np


def _prepare_image(image, segments, target):
    image = np.asarray(image, dtype=np.float32)
    segments = np.asarray(segments)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("The image must be an H x W x 3 RGB array.")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Image values must be finite and between zero and one.")
    if segments.shape != image.shape[:2] or not np.issubdtype(segments.dtype, np.integer):
        raise ValueError("Segments must be an integer label image with matching dimensions.")
    labels = np.unique(segments)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError("Segment identifiers must be contiguous and start at zero.")
    if len(labels) < 2:
        raise ValueError("At least two superpixels are needed for this comparison.")
    if int(target) != target or target not in (0, 1):
        raise ValueError("The binary target class must be zero or one.")
    segments = segments.astype(np.int32, copy=False)
    # Match LIME's per-channel averaging order, including float32 rounding.
    means = np.asarray([
        [image[segments == i][:, channel].mean() for channel in range(3)]
        for i in labels
    ], dtype=np.float32)
    baseline = means[segments].astype(np.float32)
    return image, segments, means, baseline


def _check_probabilities(probabilities, n):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (n, 2) or not np.isfinite(values).all():
        raise ValueError("predict_fn must return a finite N x 2 probability array.")
    if values.min() < -1e-7 or values.max() > 1 + 1e-7:
        raise ValueError("predict_fn returned values outside the probability range.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Binary class probabilities must sum to one.")
    return values


@contextmanager
def _numpy_seed(seed):
    """KernelExplainer uses NumPy's global generator in this version."""
    previous_state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(previous_state)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def kernel_shap(
    predict_fn: Callable,
    image_rgb_float32_0_1,
    segments,
    target: int,
    nsamples: int = 2048,
    seed: int = 1729,
):
    """Kernel SHAP for binary superpixel-presence features and class probability.

    The zero vector means all segments contain their mean RGB color. The one
    vector is the observed image. Positive weights support the target class
    relative to this replacement image. Pixel maps repeat each segment's
    weight over its pixels, so pixel sums are not SHAP additive sums.
    """
    import shap

    started = time.perf_counter()
    image, segments, means, baseline = _prepare_image(
        image_rgb_float32_0_1, segments, target
    )
    k = len(means)
    if int(nsamples) != nsamples or nsamples < 2:
        raise ValueError("nsamples must be an integer of at least two.")
    image_evaluations = 0
    predict_calls = 0
    endpoint_predictions = {}

    def mask_predict(switches):
        nonlocal image_evaluations, predict_calls
        switches = np.asarray(switches, dtype=np.float64)
        if switches.ndim != 2 or switches.shape[1] != k:
            raise ValueError("Expected one binary switch per superpixel.")
        if not np.isin(switches, (0.0, 1.0)).all():
            raise ValueError("Only binary segment presence is defined.")
        outputs = []
        for start in range(0, len(switches), 16):
            batch_switches = switches[start:start + 16]
            present = batch_switches[:, segments].astype(bool)
            batch = np.where(present[..., None], image[None], baseline[None])
            probabilities = _check_probabilities(
                predict_fn(np.ascontiguousarray(batch, dtype=np.float32)), len(batch)
            )
            outputs.append(probabilities[:, target])
            image_evaluations += len(batch)
            predict_calls += 1
            for mask, probabilities_row in zip(batch_switches, probabilities):
                if np.all(mask == 0):
                    endpoint_predictions["baseline"] = probabilities_row.copy()
                elif np.all(mask == 1):
                    endpoint_predictions["image"] = probabilities_row.copy()
        return np.concatenate(outputs)

    with _numpy_seed(seed):
        explainer = shap.KernelExplainer(
            mask_predict, np.zeros((1, k), dtype=np.float64), link="identity"
        )
        values = explainer.shap_values(
            np.ones((1, k), dtype=np.float64),
            nsamples=int(nsamples),
            l1_reg=0.0,
            silent=True,
        )
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights.shape != (k,) or not np.isfinite(weights).all():
        raise RuntimeError("Kernel SHAP did not produce one finite weight per segment.")
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    original_probability = float(endpoint_predictions["image"][target])
    reconstructed_probability = base_value + float(weights.sum())
    reconstruction_error = reconstructed_probability - original_probability
    arrays = {
        "segment_weights": weights,
        "map": weights[segments],
        "segments": segments,
        "baseline_rgb": baseline,
        "segment_mean_rgb": means,
        "image_probabilities": endpoint_predictions["image"],
        "baseline_probabilities": endpoint_predictions["baseline"],
    }
    # Retain the actual sampled masks and predictions for an inspectable fit.
    generated = int(getattr(explainer, "nsamplesAdded", 0))
    if generated:
        arrays["sampled_masks"] = np.asarray(explainer.maskMatrix[:generated], dtype=np.uint8)
        arrays["sampled_probabilities"] = np.asarray(explainer.y[:generated], dtype=np.float64).reshape(-1)
    metadata = {
        "method": "Kernel SHAP",
        "library": "shap",
        "library_version": shap.__version__,
        "target_class": int(target),
        "explained_quantity": "target-class probability",
        "link": "identity",
        "background": "one zero segment-presence vector",
        "replacement": "mean RGB color within each superpixel",
        "segment_count": k,
        "seed": int(seed),
        "nsamples_requested": int(nsamples),
        "nsamples_generated": generated,
        "l1_reg": 0.0,
        "base_value": base_value,
        "target_probability": original_probability,
        "sum_weights": float(weights.sum()),
        "reconstructed_probability": reconstructed_probability,
        "reconstruction_error": reconstruction_error,
        "absolute_reconstruction_error": abs(reconstruction_error),
        "image_evaluations": image_evaluations,
        "predict_fn_calls": predict_calls,
        "pixel_map_definition": "Each pixel displays its segment's SHAP value.",
        "runtime_seconds": time.perf_counter() - started,
    }
    return arrays, metadata


def lime_explain(
    predict_fn: Callable,
    image_rgb_float32_0_1,
    segments,
    target: int,
    nsamples: int = 2048,
    seed: int = 1729,
):
    """LIME's weighted local surrogate on the supplied superpixel segmentation."""
    from lime.lime_image import LimeImageExplainer

    started = time.perf_counter()
    image, segments, means, baseline = _prepare_image(
        image_rgb_float32_0_1, segments, target
    )
    k = len(means)
    if int(nsamples) != nsamples or nsamples < 2:
        raise ValueError("nsamples must be an integer of at least two.")
    image_evaluations = 0
    predict_calls = 0
    original_probabilities = None

    def tracked_predict(batch):
        nonlocal image_evaluations, predict_calls, original_probabilities
        batch = np.asarray(batch, dtype=np.float32)
        probabilities = _check_probabilities(predict_fn(batch), len(batch))
        # Official LIME always places the observed image first in its samples.
        if original_probabilities is None:
            if not np.array_equal(batch[0], image):
                raise RuntimeError("LIME's first sample is not the observed image.")
            original_probabilities = probabilities[0].copy()
        image_evaluations += len(batch)
        predict_calls += 1
        return probabilities

    explainer = LimeImageExplainer(
        feature_selection="none", random_state=int(seed)
    )
    explanation = explainer.explain_instance(
        image,
        tracked_predict,
        labels=(int(target),),
        hide_color=None,
        top_labels=None,
        num_features=k,
        num_samples=int(nsamples),
        batch_size=16,
        segmentation_fn=lambda _: segments,
        distance_metric="cosine",
        random_seed=int(seed),
    )
    weights = np.zeros(k, dtype=np.float64)
    for segment, coefficient in explanation.local_exp[target]:
        weights[int(segment)] = float(coefficient)
    if len(explanation.local_exp[target]) != k or not np.isfinite(weights).all():
        raise RuntimeError("LIME did not produce one finite coefficient per segment.")
    intercept = float(explanation.intercept[target])
    local_prediction = float(np.asarray(explanation.local_pred).reshape(-1)[0])
    model_prediction = float(original_probabilities[target])
    if not np.isclose(intercept + weights.sum(), local_prediction, atol=1e-10, rtol=1e-10):
        raise RuntimeError("LIME's fitted surrogate does not reconstruct its local prediction.")
    arrays = {
        "segment_weights": weights,
        "map": weights[segments],
        "segments": segments,
        "baseline_rgb": baseline,
        "segment_mean_rgb": means,
        "image_probabilities": original_probabilities,
    }
    metadata = {
        "method": "LIME",
        "library": "lime",
        "library_version": _package_version("lime"),
        "target_class": int(target),
        "explained_quantity": "target-class probability",
        "replacement": "mean RGB color within each superpixel",
        "segment_count": k,
        "seed": int(seed),
        "num_samples": int(nsamples),
        "feature_selection": "none",
        "distance_metric": "cosine",
        "kernel_width": 0.25,
        "surrogate": "Ridge regression with alpha=1 and intercept",
        "weighted_r2": float(explanation.score),
        "intercept": intercept,
        "sum_weights": float(weights.sum()),
        "local_prediction": local_prediction,
        "target_probability": model_prediction,
        "prediction_gap": local_prediction - model_prediction,
        "absolute_prediction_gap": abs(local_prediction - model_prediction),
        "image_evaluations": image_evaluations,
        "predict_fn_calls": predict_calls,
        "pixel_map_definition": "Each pixel displays its segment's local-surrogate coefficient.",
        "runtime_seconds": time.perf_counter() - started,
    }
    return arrays, metadata


def gradcam(model, input_tensor, target: int, *, block: int | None = None):
    """Grad-CAM at a ViT block's first LayerNorm, excluding the CLS token.

    Block numbers start at one. The default uses the final block.

    Gradients pass through the entire fine-tuned encoder and connectome. The
    target logit is scaled before differentiation, then gradients are unscaled,
    to reduce FP16 underflow in models whose forward pass uses autocast.
    """
    import torch
    from torch.nn import functional as F

    started = time.perf_counter()
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1 or input_tensor.shape[1] != 3:
        raise ValueError("Grad-CAM expects exactly one preprocessed NCHW RGB image.")
    if int(target) != target or target not in (0, 1):
        raise ValueError("The binary target class must be zero or one.")
    # The normal wrapper exposes a Hugging Face Dinov2Model as model.encoder.
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "encoder"):
        raise ValueError("The complete model must expose its ViT as model.encoder.")
    blocks = encoder.encoder.layer
    block_number = len(blocks) if block is None else block
    if isinstance(block_number, bool) or not isinstance(block_number, int) or not 1 <= block_number <= len(blocks):
        raise ValueError(f'Block must be an integer from 1 to {len(blocks)}.')
    layer = blocks[block_number - 1].norm1
    target_layer_name = next(
        (name for name, module in model.named_modules() if module is layer),
        f"encoder.encoder.layer[{block_number - 1}].norm1",
    )
    patch_size = encoder.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = map(int, patch_size)
    else:
        patch_h = patch_w = int(patch_size)
    height, width = map(int, input_tensor.shape[-2:])
    if height % patch_h or width % patch_w:
        raise ValueError("The input dimensions must be divisible by the ViT patch size.")
    rows, columns = height // patch_h, width // patch_w
    captured = []

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError("The target LayerNorm must return a tensor.")
        captured.append(output)

    was_training = model.training
    handle = layer.register_forward_hook(hook)
    gradient_scale = 1024.0
    try:
        model.eval()
        with torch.enable_grad():
            # This also enables gradients if a caller has frozen parameters.
            image = input_tensor.detach().requires_grad_(True)
            output = model(image)
            if torch.is_tensor(output):
                logits = output
            elif hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            else:
                raise TypeError("The model must return a tensor or an object containing logits.")
            if logits.shape != (1, 2):
                raise ValueError("This binary connectome model must return two logits.")
            if len(captured) != 1:
                raise RuntimeError("Expected one execution of the selected ViT LayerNorm.")
            activations = captured[0]
            if activations.ndim != 3 or activations.shape[1] != rows * columns + 1:
                raise ValueError("Expected a CLS token followed by one token per image patch.")
            gradients, = torch.autograd.grad(
                logits[0, target].float() * gradient_scale,
                activations,
                retain_graph=False,
                create_graph=False,
            )
            gradients = gradients.detach().float() / gradient_scale
            activations = activations.detach().float()
            if not torch.isfinite(gradients).all() or not torch.isfinite(activations).all():
                raise RuntimeError("Non-finite Grad-CAM gradients or activations.")
            patch_gradients = gradients[:, 1:, :]
            patch_activations = activations[:, 1:, :]
            channel_weights = patch_gradients.mean(dim=1)
            raw = (patch_activations * channel_weights[:, None, :]).sum(dim=-1)
            raw = raw.reshape(1, 1, rows, columns)
            positive = torch.relu(raw)
            maximum = float(positive.max().item())
            patch_cam = positive / maximum if maximum > 0 else torch.zeros_like(positive)
            cam = F.interpolate(patch_cam, size=(height, width), mode="bilinear", align_corners=False)
            # Interpolation can reduce the maximum. Normalize the displayed map
            # separately while retaining the original patch-scale result.
            resized_maximum = float(cam.max().item())
            if resized_maximum > 0:
                cam = cam / resized_maximum
            logits_cpu = logits.detach().float().cpu().numpy()[0]
            probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()[0]
            arrays = {
                "map": cam[0, 0].cpu().numpy(),
                "patch_map": patch_cam[0, 0].cpu().numpy(),
                "raw_patch_map": raw[0, 0].cpu().numpy(),
                "channel_weights": channel_weights[0].cpu().numpy(),
                "patch_gradient_norm": patch_gradients.norm(dim=-1).reshape(rows, columns).cpu().numpy(),
                "patch_activation_norm": patch_activations.norm(dim=-1).reshape(rows, columns).cpu().numpy(),
                "logits": logits_cpu,
                "image_probabilities": probabilities,
            }
            metadata = {
                "method": "Grad-CAM",
                "library": "torch",
                "library_version": torch.__version__,
                "target_class": int(target),
                "explained_quantity": "target-class logit",
                "target_layer": target_layer_name,
                "block_number": block_number,
                "excluded_tokens": "CLS",
                "patch_grid": [rows, columns],
                "patch_size": [patch_h, patch_w],
                "target_logit": float(logits_cpu[target]),
                "target_probability": float(probabilities[target]),
                "gradient_scale": gradient_scale,
                "gradient_min": float(patch_gradients.min().item()),
                "gradient_max": float(patch_gradients.max().item()),
                "gradient_nonzero_count": int(torch.count_nonzero(patch_gradients).item()),
                "gradient_element_count": int(patch_gradients.numel()),
                "raw_positive_maximum": maximum,
                "zero_positive_map": maximum <= 0,
                "normalization": "ReLU, then divide by the positive spatial maximum",
                "runtime_seconds": time.perf_counter() - started,
            }
    finally:
        handle.remove()
        model.train(was_training)
    return arrays, metadata
