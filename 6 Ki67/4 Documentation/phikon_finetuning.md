# Phikon-v2 full fine-tuning

The joint comparison trains all Phikon-v2 layers jointly with either
40,755 anatomical synaptic gains or a linear output layer with 1,024 weights
and one bias. It covers the binary PI threshold of 30% only, with seeds 1729,
2718 and 3141 for each architecture.

BCData and SHIDC-B-Ki-67-V2 supply 2,128 pooled training images, 464 validation
images and 1,102 test images. Each checkpoint is selected by pooled validation
balanced accuracy and evaluated unchanged on the combined test set and its
402 BCData and 700 SHIDC-B-Ki-67-V2 images.

## Reconstruct the reported results

After installing the repository and restoring `zenodo/`, run from the repository
root:

```bash
python -B -m ki67.phikon_finetuning report --verify-checkpoints
```

The command verifies the six selected checkpoints, recomputes balanced accuracy,
accuracy and macro-F1 from saved test probabilities, and writes mean and sample
standard deviation across three seeds. It also reproduces parameter counts and
recorded timing medians. Results go to
`6 Ki67/6 Workspace/phikon-joint-finetune-report/`. No images or GPU are needed.

The source artifacts are in
`zenodo/payload/ki67/outputs/phikon-joint-finetune/`. They include only the six
reported runs, with selected weights, predictions, training histories, image
manifests, the anatomical graph and fixed feature standardization statistics.
The original checkpoint metadata is retained as provenance. Its broader task
inventory and scheduler dependency describe the historical execution and are
not requirements of this six-run reproduction workflow.

## Repeat training or inference

Download BCData and SHIDC-B-Ki-67 following the [data guide](<../../2 Data/README.md>),
then create SHIDC-B-Ki-67-V2 using the [patient partition guide](shid_v2.md).
The archived image manifests specify each expected repository-relative image
path and checksum. Preparation verifies those files and recreates the exact
pixel cache. No raw images or pixel caches are distributed here.

```bash
python -B -m ki67.phikon_finetuning prepare
python -B -m ki67.phikon_finetuning train --job joint-anatomical-binary30-seed-1729 --gpu 0
python -B -m ki67.phikon_finetuning train --job joint-linear-binary30-seed-1729 --gpu 1
```

Repeat the training commands with seeds 2718 and 3141. Each command starts one
run on the explicitly selected GPU. Runs can execute concurrently on different
GPUs. There is no dependency on another training campaign and no automatic
launch on import. Use a new `--output` directory to repeat an existing run.
`status --archive "6 Ki67/6 Workspace/phikon-joint-finetune"` lists progress.

To evaluate a released checkpoint with the prepared images:

```bash
python -B -m ki67.phikon_finetuning evaluate --job joint-anatomical-binary30-seed-1729 --gpu 0
```

Omit `--gpu` for CPU inference. Both inference modes write outside the archive.
To summarize retrained models, use `report --archive` with their Workspace path.

## Protocol and timing

Complete images are resized to 224 by 224 pixels with Pillow bicubic
interpolation and normalized using the pinned Phikon-v2 preprocessing values.
Both architectures use the same fixed feature mean and standard deviation,
estimated from the pooled training descriptors before fine-tuning.

Training uses 120 epochs, Adam, effective minibatches of 16, and learning rates
of `1e-5` for Phikon-v2, `0.03` for anatomical gains and `1e-3` for the linear
output layer. Global gradient norm is clipped at one. The encoder uses float16
autocast, while the downstream model and loss use float32. The anatomical model
uses two fixed population scores, cross-entropy and softmax. The linear model
uses one logit, binary cross-entropy and sigmoid. Epoch zero is eligible for
selection and validation ties retain the earlier checkpoint. `protocol.json`
records all settings and dependency versions.

Training duration sums `training_seconds` for epochs 1 through 120, excluding
validation and checkpoint storage. Test evaluation includes prediction and
metric calculation for all 1,102 images. Both reported durations are medians
across three runs per architecture, with one GPU per run.

## SHIDC-B-Ki-67 three-class experiment (Table 16)

The separate three-class experiment uses the original SHIDC-B-Ki-67 partition:
1,325 training, 331 validation and 700 test images. It fine-tunes the complete
Phikon-v2 encoder and 40,755 anatomical synaptic gains. These are distinct from
the pooled binary30 runs and the patient-disjoint SHIDC-B-Ki-67-V2 partition.
The three seeds are 1729, 2718 and 3141; their selected epochs are 63, 100 and 78.
Mean test accuracy is 78.24 ± 0.22%, balanced accuracy 77.04 ± 0.91%, macro
precision 77.11 ± 0.79%, and macro F1 76.52 ± 0.56% (sample SD across seeds).

The release contains exactly these three selected checkpoints and their train,
validation and test predictions, training histories, row manifests, scalers and
graph under `zenodo/payload/ki67/outputs/phikon-shid-finetune/`. Checkpoints retain
historical source hashes and protocol metadata; the release protocol limits
retraining to the reported three-class experiment. Model tensors are unchanged.

```bash
python -B -m ki67.phikon_finetuning.shid report --verify-checkpoints
python -B -m ki67.phikon_finetuning.shid prepare
python -B -m ki67.phikon_finetuning.shid train --job shid-multiclass-seed-1729 --gpu 0
python -B -m ki67.phikon_finetuning.shid evaluate --job shid-multiclass-seed-1729 --gpu 0
```

Repeat training or evaluation for seeds 2718 and 3141. Download the original
images using the [data guide](<../../2 Data/README.md>) before `prepare`; no
patient repartition is needed for this experiment. Preparation checks every
image and reproduces the recorded pixel-cache hashes. Reports require no raw
images or GPU, and write `shid-finetuning-summary.json` and its Markdown table
under `6 Ki67/6 Workspace/phikon-shid-finetune/`. The JSON records each seed,
selected epoch, metric, prediction checksum, mean and sample SD. Training and
checkpoint selection use the protocol above, with three anatomical population
scores, cross-entropy and softmax.

## Figure 12: joint fine-tuning confusion matrices

```bash
python -B -m ki67.phikon_finetuning.render_confusions
```

This CPU-only command reads saved predictions from the six joint binary30 runs
and writes `Figure-12.pdf`, `Figure-12.png` and the numerical provenance JSON to
`6 Ki67/6 Workspace/phikon-finetuning-confusions/`. Each cell gives the arithmetic
mean count and true-class percentage across the three independent seeds.
Predictions are not ensembled. The generator checks every per-seed confusion
matrix against its completion record, verifies consistent sample ordering and
supports, and checks that the BCData and SHIDC-B-Ki-67-V2 counts sum to the
combined count. It also validates panel geometry and text bounds. The archived
publication exports are in `phikon-joint-finetune/figures/`.

Figure 13 and the complete explanation workflow are described in
[phikon_xai.md](phikon_xai.md).
