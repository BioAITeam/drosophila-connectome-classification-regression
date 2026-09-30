# Fixed protocol and interpretation

The [data-format guide](data_format.md) defines the required image–JSON pairs, label meanings, split folders and preparation requirements.

Choose a template in `6 Ki67/1 Configuration/` for the experiment to reproduce:

| Template | Scope |
| --- | --- |
| `campaign_separate_vit.json` | 168 separately trained ViT-based fits across both datasets |
| `campaign_rgb_multiclass.json` | 12 native-RGB three-class fits |
| `campaign_joint.json` | 12 joint three-class fits |

Training preserves the dataset partitions; test metrics do not select preprocessing, checkpoints or stopping. Final figures are generated separately through the [paper reproduction command](<../../7 Paper Results/README.md>).

## Prepare and launch a new campaign

Run all commands from the repository root. Restore the archived fixed graphs following the [model guide](<../../4 Trained Models/README.md>) and supply the prepared image datasets. New campaigns require a directory under `6 Ki67/6 Workspace/`. For a separate-dataset ViT campaign, first make a local plan:

```bash
mkdir -p "6 Ki67/6 Workspace/plans"
cp "6 Ki67/1 Configuration/campaign_separate_vit.json" \
  "6 Ki67/6 Workspace/plans/separate-vit.json"
```

In that copy, set `runtime.gpu_indices` to available physical GPUs, for example `[0]`, and `output_root` to `"6 Ki67/6 Workspace/separate-vit-reproduction"`. `--gpus` must name devices from the plan's list. Keep `--output` equal to `output_root` throughout preparation and launch.

```bash
"6 Ki67/run.sh" audit \
  --plan "6 Ki67/6 Workspace/plans/separate-vit.json" \
  --output "6 Ki67/6 Workspace/separate-vit-reproduction"
"6 Ki67/run.sh" download \
  --plan "6 Ki67/6 Workspace/plans/separate-vit.json" \
  --output "6 Ki67/6 Workspace/separate-vit-reproduction"
"6 Ki67/run.sh" campaign \
  --plan "6 Ki67/6 Workspace/plans/separate-vit.json" \
  --output "6 Ki67/6 Workspace/separate-vit-reproduction"
"6 Ki67/run.sh" launch \
  --plan "6 Ki67/6 Workspace/plans/separate-vit.json" \
  --output "6 Ki67/6 Workspace/separate-vit-reproduction" \
  --gpus 0 --background
```

The queue extracts each frozen representation, profiles it and trains dependent jobs. To inspect progress:

```bash
"6 Ki67/run.sh" status \
  --plan "6 Ki67/6 Workspace/plans/separate-vit.json" \
  --output "6 Ki67/6 Workspace/separate-vit-reproduction"
```

For the 12 native-RGB multiclass fits, copy `campaign_rgb_multiclass.json` instead, set its local GPU list, and set both `output_root` and `--output` to `6 Ki67/6 Workspace/rgb-reproduction`; omit the ViT download step. To resume a Workspace campaign, use its frozen `manifests/campaign.json` and the same output directory.

For joint training, follow the [joint campaign instructions](joint_training.md). It can use the restored separate-dataset features without access to the original images.

If graphs are regenerated instead of restored, set the plan's `graph_sources.anatomical` and `graph_sources.degree241` to the new graph files and keep their matching JSON sidecars. See the [connectome guide](<../../3 Connectome/README.md>) for graph preparation.

## Scientific protocol

Supervised updates act on the internal gain tensor. Pretrained image encoders remain frozen, input coordinate mapping is deterministic, and population readouts have no trainable weights. The anatomical graph retains measured connections; the rewired control uses artificial destinations. Neuronal dynamics and gain adaptation are computational modeling assumptions.

Native pixels and ViT features differ in scaling, dimensionality and recurrent sequence length, so their training costs are not matched. Before training, profiling measures full-image processing on small batches, records peak GPU memory and verifies finite, nonzero gain gradients.

Classification checkpoints maximize validation balanced accuracy. Regression checkpoints and early stopping minimize raw validation mean absolute error (MAE), measured in percentage points for PI or cells for counts. Regression metrics are retained before and after restricting PI to [0, 100] and counts to nonnegative values. Standard deviations across three optimization seeds describe variability on the same observations.

Recovery restores the validation-selected checkpoint, including optimizer and random-number-generator state. Later epochs may be repeated under a new attempt identifier. Archived predictions from previous attempts are retained; summaries select the canonical attempt specified by `best.json` and never pool attempts as additional seeds.
