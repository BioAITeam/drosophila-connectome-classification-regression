# Joint Ki-67 training

Joint training fits one connectome model per frozen extractor, graph and optimization seed. The same fitted model processes both datasets. The [model naming guide](nomenclature.md) defines BRAP-Net and its comparison configurations.

## Data and representations

Corresponding partitions are pooled without changing image membership or annotation-derived targets. Dataset identity is retained for reporting and is not an input feature.

| Dataset | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| BCData | 803 | 133 | 402 |
| SHIDC-B-Ki-67 | 1,325 | 331 | 700 |
| Combined | 2,128 | 464 | 1,102 |

The frozen extractors resize the full image to 224 × 224 without cropping. Each produces an image-level classification-token (CLS) feature vector: 768 values for DINOv2-base and 1,024 for Phikon-v2. Existing dataset-specific standardized caches are transformed back to approximate CLS values using their saved means and standard deviations in float64. This retains the rounding error already present in the float32 caches. One pooled mean and standard deviation are fitted on training coordinates and applied to all splits. Joint feature preparation reuses these caches without rerunning or fine-tuning the ViTs.

## Training and selection

`6 Ki67/1 Configuration/campaign_joint.json` reproduces the paper's 12 joint three-class fits: two extractors, two graph variants and three seeds.

Newly initialized runs start with unit gains. Input blocks contain up to 730 coordinates and persist for four recurrent updates. Three-class training runs for 120 epochs.

At initialization and after every epoch, the model is evaluated on training, validation and test data, both pooled and by dataset. Three-class checkpoints maximize pooled validation balanced accuracy. Improvements must exceed 1e-8 and ties retain the earlier checkpoint. Per-dataset validation and test scores do not select or stop training. Resumption restores the selected checkpoint, optimizer and random states.

Pooled metrics are calculated from all predictions together rather than by averaging dataset metrics. Results report individual-seed metrics and their mean and sample standard deviation. The manuscript selects each graph's extractor by mean validation performance across seeds; the inference scripts select their default checkpoint by its individual validation score.

## Start a new joint campaign

Run from the repository root after restoring the separate-dataset feature caches and fixed graphs. Copy the template before preparing a campaign:

```bash
mkdir -p "6 Ki67/6 Workspace/plans"
cp "6 Ki67/1 Configuration/campaign_joint.json" \
  "6 Ki67/6 Workspace/plans/joint.json"
```

Set these fields in the copy before running the commands below:

| Field | Value for this example |
| --- | --- |
| `tasks` | `["multiclass"]` |
| `runtime.gpu_indices` | `[0]`, or the free physical GPUs on your system |
| `output_root` | `"6 Ki67/6 Workspace/joint-reproduction"` |

`--gpus` must name devices from `runtime.gpu_indices`. Use the same `--output` directory throughout preparation and launch.

```bash
"6 Ki67/run.sh" audit \
  --plan "6 Ki67/6 Workspace/plans/joint.json" \
  --output "6 Ki67/6 Workspace/joint-reproduction"
"6 Ki67/run.sh" campaign \
  --plan "6 Ki67/6 Workspace/plans/joint.json" \
  --output "6 Ki67/6 Workspace/joint-reproduction"
"6 Ki67/run.sh" launch \
  --plan "6 Ki67/6 Workspace/plans/joint.json" \
  --output "6 Ki67/6 Workspace/joint-reproduction" \
  --gpus 0 --background
```

To use newly extracted features, set `joint.source_output` in the copied plan to the prepared separate-dataset Workspace campaign; both extractors must have complete caches for both datasets. Set `output_root` to a fresh joint Workspace directory and use that same path for `--output` throughout preparation and launch. For resumption, use the campaign's saved `manifests/campaign.json` and the same output directory.

## Interpretation

Both datasets contribute training images, so pooled evaluation does not test transfer to an unseen dataset. The 23 SHIDC-B-Ki-67 patients appear across the original partitions; BCData patient independence cannot be established from available identifiers. Optimization seeds reuse fixed observations. The ViTs contribute pretraining knowledge, while supervised Ki-67 learning updates connectome gains. Rewiring is an artificial control.
