# Phikon-v2 explanations and Figure 13

Figure 13 explains the complete fine-tuned Phikon-v2 encoder and anatomical
connectome using Grad-CAM, Kernel SHAP and LIME. The selected joint binary30
checkpoint is seed 2718, epoch 3, chosen by the highest pooled validation
balanced accuracy among the three anatomical runs. All methods target each
example's predicted class. The BCData image `131` belongs to the test partition;
SHIDC-B-Ki-67 image `p2_0191_1` belongs to the validation partition of the patient
repartition. The latter is an illustrative explanation, not a held-out test
result. These are the panel C and panel E examples in Figure 2 of the referenced
Ki-67 vision-language benchmarking study, selected independently of explanation
maps.

## Render and validate without inference

After installing the repository and restoring `zenodo/`, run:

```bash
python -B -m ki67.phikon_xai validate
python -B -m ki67.phikon_xai render
```

Both commands run on CPU without original images or model inference. They read
`zenodo/payload/ki67/outputs/phikon-xai/` and write to
`6 Ki67/6 Workspace/phikon-xai/`. Rendering writes `figures/Figure-13.pdf`,
`figures/Figure-13.png` and `provenance.json`. Validation checks artifact hashes,
finite arrays, the two examples, checkpoint and target identity, the Grad-CAM
layer, agreement with archived predictions, superpixel map construction and
Kernel SHAP additivity. LIME's weighted R² and local probability gap are
reported explicitly: it is an approximate local surrogate.

The release includes only the two published examples and six attribution maps.
It retains their two 224 × 224 RGB display inputs and SLIC masks, sampled SHAP
masks/probabilities, method diagnostics and final figure exports. Original
database images, unused example panels and alternative Grad-CAM layers are
excluded. The selected checkpoint, graph and fixed scaler remain in the shared
`phikon-joint-finetune/` archive, avoiding a duplicate model copy.

The final publication raster was produced with NumPy 1.26.4, Matplotlib 3.3.4 and
Pillow 10.4.0; those versions reproduce its pixels exactly. Current supported
plotting versions reproduce the same maps and layout but may differ in nearest
neighbor image rasterization. The archived final PDF/PNG is retained unchanged
in `phikon-xai/figures/`. `rendering_versions.json` distinguishes the publication
renderer from the explanation-generation environment in `execution_versions.json`.

## Recreate inputs or explanations

Acquire BCData and SHIDC-B-Ki-67 using the [data guide](<../../2 Data/README.md>)
and construct the [patient partition](shid_v2.md). The archived selection
manifest gives the exact image paths and SHA-256 hashes. To reconstruct the two
resized inputs and check them against the released arrays:

```bash
python -B -m ki67.phikon_xai prepare --output "6 Ki67/6 Workspace/phikon-xai-inputs"
python -B -m ki67.phikon_xai render --verify-originals
```

Install the `xai` optional dependency group, then recompute the six maps from
the retained checkpoint without training:

```bash
python -B -m ki67.phikon_xai explain --gpu 0 --output "6 Ki67/6 Workspace/phikon-xai-recomputed"
python -B -m ki67.phikon_xai render --archive "6 Ki67/6 Workspace/phikon-xai-recomputed" --output "6 Ki67/6 Workspace/phikon-xai-recomputed-figure"
```

Omitting `--gpu` uses CPU, which is considerably slower and uses full-precision
encoder operations. The recorded explanations used PyTorch 2.2.2+cu121,
Transformers 4.49.0, NumPy 1.26.4, Pillow 12.3.0, scikit-image 0.26.0,
SHAP 0.49.1 and LIME 0.2.0.1. Small numerical differences can occur across
hardware and precision modes. No training runs start on import; `explain`
requires its own explicit command and writes outside the archived inputs.

## Method and display definitions

Grad-CAM differentiates the predicted-class logit through the entire model at
`encoder.encoder.layer.11.norm1`, the first normalization of ViT block 12. It
excludes the CLS token, spatially averages patch gradients into channel weights,
applies ReLU and resizes the 14 × 14 patch map to 224 × 224. The gradient scale
of 1024 is undone after differentiation to limit float16 underflow.

SHAP and LIME explain the predicted-class probability on SLIC superpixels
(`n_segments=64`, compactness 10, sigma 1). An absent superpixel is replaced by
its own mean RGB color. Both use 2,048 perturbations, with seed 1731 for BCData
and 1730 for SHIDC-B-Ki-67. Kernel SHAP uses an identity link, an all-absent
background, and no L1 feature selection. LIME uses cosine distance, kernel
width 0.25, all superpixels and ridge regression with alpha 1 and an intercept.

Grad-CAM maps are nonnegative. SHAP and LIME maps retain signed superpixel
coefficients: blue opposes and orange supports the target class. Each map is
independently normalized for display, preserving the original arrays and signs.
The colors therefore describe relative contributions within one map; they do
not establish equal absolute effects across methods or images. Attribution
overlays use a grayscale background, with the RGB input shown separately.
