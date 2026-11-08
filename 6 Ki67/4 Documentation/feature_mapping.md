# Fixed Phikon-v2 visual-feature assignments

This control measures sensitivity to the assignment of the 1,024 frozen Phikon-v2 descriptors to the anatomical BRAP-Net input sequence. It uses the combined BCData and SHIDC-B-Ki-67-V2 task, with 2,128 training, 464 validation and 1,102 test images. SHIDC-B-Ki-67-V2 patients are disjoint across the three partitions (13/3/7 patients).

The identity assignment and ten fixed permutations each use training seeds 1729, 2718 and 3141, giving 33 runs. Permutation seeds are 10001 to 10010. Each permutation acts on the standardized feature coordinates before the unchanged 730 + 294 input blocking and is shared across all images, splits, epochs and training seeds. The anatomical graph, 40,755 trainable edge gains, optimizer and training schedule remain fixed. Only edge gains are trained; Phikon-v2 remains frozen.

Each run uses 120 training epochs. Checkpoint selection maximizes pooled validation balanced accuracy, includes epoch zero, requires an improvement above `1e-8`, and retains the earliest tie. Test predictions are recorded for analysis and do not select checkpoints.

| Test scope | Identity BA, mean ± seed SD (%) | Permuted BA, mean ± mapping SD (%) | Permuted minus identity (pp), 95% CI |
|---|---:|---:|---:|
| Combined | 55.31 ± 0.66 | 50.63 ± 5.09 | -4.68 [-8.19, -1.20] |
| BCData | 64.91 ± 0.56 | 52.03 ± 7.08 | -12.88 [-17.85, -8.00] |
| SHIDC-B-Ki-67-V2 | 49.37 ± 0.95 | 49.85 ± 4.13 | +0.47 [-3.05, +3.83] |

Identity SD describes three training seeds. Permutation SD describes ten mapping means, each averaged over those three seeds. Intervals use 10,000 shared image bootstrap replicates within dataset/class strata, plus resampling of permutation IDs for family estimates; bootstrap seed is 20261105. Seeds remain paired and are not independently resampled. Intervals condition on the fitted models and observed images, assume image independence within strata, and do not quantify patient-level or retraining uncertainty.

## Restore and dependencies

Restore Zenodo `payload/ki67/outputs/phikon-feature-mapping/` to `6 Ki67/5 Outputs/phikon-feature-mapping/` in the GitHub checkout. All commands below run from the checkout root. Install the project dependencies and Pillow, for example:

```bash
python -m pip install -e '.[vision]'
```

The recorded environment is Python 3.12, NumPy 1.26.4 and PyTorch 2.2.2+cu121; use compatible hardware and libraries when comparing numerical predictions. Analysis needs NumPy and PyTorch. Training and prediction verification also use the released `ki67`/`flybrain` modules, scikit-learn and Pillow. The archived standardized descriptors, scalers, labels and graph are sufficient for this control; raw images and pretrained backbone weights are not loaded by these commands.

## Verify and reanalyze the saved experiment

```bash
python '6 Ki67/2 Source/feature_mapping/run.py' verify
python '6 Ki67/2 Source/feature_mapping/analyze.py'
python '6 Ki67/2 Source/feature_mapping/run.py' smoke --device cpu
```

Analysis checks all 33 configurations, gain checkpoints, validation-selection histories, prediction metrics, image alignment, permutation definitions and released-file checksums. It writes tables, provenance, the 10,000-replicate bootstrap archive and a summary to `6 Ki67/6 Workspace/phikon-feature-mapping/analysis/`. Saved outputs under `5 Outputs` remain read-only. The smoke check tests feature permutations and their inverse on every split, identity equivalence, nonzero finite gradients, gain-only training, and full-test predictions for one identity and one permuted checkpoint. Add `--all-checkpoints` to verify the predictions of all 33 selected checkpoints. CUDA can be selected with `--device cuda:0`.

## Reproduce training

```bash
python '6 Ki67/2 Source/feature_mapping/run.py' prepare
python '6 Ki67/2 Source/feature_mapping/run.py' smoke \
  --experiment '6 Ki67/6 Workspace/phikon-feature-mapping/training'
python '6 Ki67/2 Source/feature_mapping/run.py' train --device cuda:0
python '6 Ki67/2 Source/feature_mapping/analyze.py' \
  --experiment '6 Ki67/6 Workspace/phikon-feature-mapping/training' \
  --output '6 Ki67/6 Workspace/phikon-feature-mapping/retrained-analysis'
```

`prepare` creates the same eleven assignments and 33 job definitions in a fresh Workspace directory. `train` executes all jobs sequentially and skips valid completed jobs. To distribute independent jobs across devices, supply `--mapping mapping-00 --seed 1729` and the appropriate device; launch each mapping/seed pair only once. `--experiment` can select another repository-relative Workspace directory. A fresh directory is required for a new campaign. CUDA arithmetic may differ across devices and software versions; the archived results are the reported completed campaign.

## Payload contents

- `inputs/`: fixed joint manifest, standardized split features, training-derived mean/std, frozen encoder provenance, anatomical graph, and original joint-campaign selection metadata and predictions used for the historical identity comparison. The three historical gain checkpoints are retained once in `shid-v2-20261031/joint/`; the identity comparison uses the metadata and predictions and does not load those checkpoints.
- `mappings/`: eleven permutation definitions and 33 completed runs, including selected gain checkpoints, all 121 epoch histories and all recorded per-epoch predictions.
- `protocol.json`, `config.json`, `jobs.json`: settings, assignments, split counts and checksums.
- `analysis/`: run metrics, mapping and family summaries, paired differences, confidence intervals, bootstrap replicates and provenance.
- `provenance/`: numerical-preservation record. Release metadata uses repository-relative paths and updated metadata checksums; gains, inputs and prediction arrays preserve their recorded numerical values.

The implementation uses the released `ki67.training`, `ki67.sequence` and `flybrain.model` components. It has no dependency on temporary experiment directories or fine-tuning workflows.
