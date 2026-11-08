# Cross-dataset evaluation of frozen-feature anatomical models

This experiment evaluates source-trained Phikon-v2 + anatomical-connectome models without target training, target calibration, or refitting feature normalization. Each checkpoint was selected only by its source validation balanced accuracy. The unchanged source-training mean and standard deviation transform raw target CLS vectors.

The four directions are BCData to original SHIDC-B-Ki-67, BCData to patient-disjoint SHIDC-B-Ki-67-V2, original SHIDC-B-Ki-67 to BCData, and V2 to BCData. Four classification tasks (three classes and binary thresholds 5%, 10%, 30%) and three seeds give 48 evaluations using 36 distinct source checkpoints. The target sets contain 402 BCData images or 700 SHIDC images. The original and V2 SHIDC test sets overlap, so the 48 evaluations are not independent external cohorts.

## Rebuild reports from saved predictions

```bash
python -m ki67.run_cross_dataset --output "6 Ki67/6 Workspace/cross-dataset-reports"
```

The default `--mode report` reads `6 Ki67/5 Outputs/shid-cross-dataset-20261101`. It verifies prediction-file hashes, reconstructs predicted categories and metrics from all 48 NPZ files, checks them against the saved records, and writes `cross_dataset_results.json`, `cross_dataset_summary.csv`, `cross_dataset_per_class.csv`, and `cross_dataset_summary.md` to a new Workspace directory. No images, GPU, feature extraction, or training are needed.

## Recompute predictions or image features

```bash
python -m ki67.run_cross_dataset --mode replay --gpus 0,1 --output "6 Ki67/6 Workspace/cross-dataset-replay"
python -m ki67.run_cross_dataset --mode images --gpus 0,1 --output "6 Ki67/6 Workspace/cross-dataset-images"
```

Replay uses the archived raw CLS matrix for 1,595 distinct target images and the original source checkpoints/scalers. It also checks predictions against the source models' own-domain saved results. The images mode requires the locally prepared image datasets and shared pinned Phikon-v2 weights, repeats frozen image encoding, and checks the direct image-prediction API. Neither mode optimizes a model. Workers receive explicit GPU indices and use the existing UUID reservation mechanism; an unavailable GPU is never taken over.

The archive preserves source identities, selected epochs, source-only validation scores, per-image probabilities, labels, sample identifiers, and timings. The source checkpoint references in historical metadata remain repository-relative; the reader maps their historical Workspace locations to the immutable `6 Ki67/5 Outputs` archive at runtime. No numerical artifact or checkpoint is rewritten for relocation. Shared extractor weights remain in the global cache, outside this experiment directory.
