# BRAP-Net image inference

## 1. Restore the inference assets

Complete the [environment installation](<../../1 Environment/README.md>) and [model restoration and verification](<../../4 Trained Models/README.md>). Restore the BRAP-Net checkpoints, graphs, training normalization and frozen Phikon-v2 weights.

The classifiers were trained jointly on BCData and SHIDC-B-Ki-67. They predict three categories of the Ki-67 proliferation index (PI): **Low** (PI <16%), **Intermediate** (16% ≤ PI <30%) and **High** (PI ≥30%). Each image is converted to RGB, resized in full to 224 × 224 pixels, encoded by frozen Phikon-v2 and standardized with the saved pooled-training statistics before connectome inference.

Execution uses CPU and local weights. Annotation JSON, dataset-origin labels and an Internet connection are not required.

## 2. Predict one image

Run commands from the repository root using the environment configured in the installation guide. Replace the example input paths with existing files or directories; the illustrated inputs are not distributed. Relative input paths are resolved from the current working directory, while relative `--save` paths are resolved from the repository root. Absolute paths are also accepted, subject to the output-location requirement below.

```bash
python -B "6 Ki67/3 Inference/predict_image.py" \
  --image "2 Data/1 Raw/ki67/example.jpg" \
  --save "6 Ki67/6 Workspace/inference/brap-net-image.npz"
```

## 3. Predict a folder

Use `--recursive` to include subfolders.

```bash
python -B "6 Ki67/3 Inference/predict_folder.py" \
  --folder "2 Data/1 Raw/ki67/images" \
  --recursive \
  --save "6 Ki67/6 Workspace/inference/brap-net-folder.npz"
```

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp` and `.webp`, including uppercase variants. Folder inputs are processed in a deterministic order.

## 4. Select a connectome variant

Both scripts accept `--graph anatomical` (the default) or `--graph degree241` (the rewired control); `--graph rewired` is an equivalent alias. An optional `--seed` selects `1729`, `2718` or `3141`. Each seed's checkpoint was selected by pooled validation balanced accuracy. If `--seed` is omitted, the script selects the highest of these validation scores within the requested graph:

| Graph | Default seed | Selected epoch |
| --- | ---: | ---: |
| Anatomical | 2718 | 1 |
| Rewired (`degree241`) | 1729 | 61 |

Each command uses one saved checkpoint, without combining predictions across seeds. To use the rewired variant, append `--graph degree241` to either command and choose a different `--save` filename.

## 5. Read the predictions

Use a new `.npz` filename under `6 Ki67/6 Workspace/` for `--save`. Each command creates the following three files with that filename stem:

| File | Content |
| --- | --- |
| `.csv` | One row per image, predicted class and the probability of each class |
| `.npz` | Image identifiers and hashes, class names, predicted labels and probability arrays |
| `.json` | Model provenance, checkpoint hash, selected epoch, graph, seed and execution details |

In the NPZ, `class_names` gives the probability-column order: Low, Intermediate, High. `probabilities` contains values between zero and one; `y_pred` contains zero-based class indices, and `predicted_class` contains their names. `image_path` and `image_sha256` identify each input.

```python
import numpy as np

with np.load("6 Ki67/6 Workspace/inference/brap-net-image.npz", allow_pickle=False) as result:
    print(result["class_names"])
    for path, label, probabilities in zip(
        result["image_path"], result["predicted_class"], result["probabilities"]
    ):
        print(path, label, probabilities)
```

## 6. Other saved models

For other archived tasks and extractors, see `"6 Ki67/run.sh" --device cpu predict --help`. Supply the chosen checkpoint with `--checkpoint` and its campaign directory with `--output`; the saved campaign plan is loaded automatically, or can be specified with `--plan`. The BRAP-Net scripts above use joint Phikon-v2 three-class weights; binary and regression models have separate checkpoints.

Small numerical differences can arise when reencoding images on different devices or library versions.
