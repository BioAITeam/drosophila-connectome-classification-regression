# 6. Ki-67 experiments

BRAP-Net classifies breast cancer histopathology images using frozen Phikon-v2 features and a recurrent *Drosophila* connectome trained jointly on BCData and SHIDC-B-Ki-67. Comparisons cover anatomical and rewired graphs, DINOv2 features, separate-dataset training and native RGB inputs.

| Folder | Contents |
| --- | --- |
| `1 Configuration/` | Experiment plans for joint, separate-dataset and RGB training |
| `2 Source/` | The `ki67` Python package |
| `3 Inference/` | Scripts and instructions for one image or a folder |
| `4 Documentation/` | Data format, training protocols and figure definitions |
| `5 Outputs/` | Restored models and experimental records, read-only |
| `6 Workspace/` | New predictions, training runs and reports |

The [model naming guide](<4 Documentation/nomenclature.md>) distinguishes BRAP-Net variants, separate-dataset models and the graph and optimization seeds used in saved records.

## 1. Prepare

Follow the [environment guide](<../1 Environment/README.md>) and [model restoration guide](<../4 Trained Models/README.md>). Run commands from the repository root.

## 2. Predict images

The [inference guide](<3 Inference/README.md>) provides BRAP-Net commands for one image or a folder, including checkpoint selection and output definitions. Prediction uses the restored models without annotation files or retraining.

## 3. Reconstruct results

Use the [paper reproduction command](<../7 Paper Results/README.md>) for the final figures and tables. The [artifact map](<4 Documentation/manuscript_confusion_matrices.md>) explains individual panels and their saved inputs.

The [classification analysis](<4 Documentation/classification_analysis.md>) reconstructs Tables 6-7, class prevalence, class-specific metrics, image-bootstrap intervals and validation-fitted temperature scaling from archived predictions and logits.

To recompute a campaign summary in a new directory:

```bash
"6 Ki67/run.sh" evaluate \
  --plan "6 Ki67/5 Outputs/joint-bcdata-shid-v1/manifests/campaign.json" \
  --output "6 Ki67/5 Outputs/joint-bcdata-shid-v1" \
  --destination "6 Ki67/6 Workspace/evaluations/joint"
```

## 4. Train

The [data-format guide](<4 Documentation/data_format.md>) defines image-annotation pairs, original partitions, targets and configurable dataset locations.

Follow the [training protocol](<4 Documentation/protocol.md>) for campaign scope, checkpoint selection, preparation and launch. Use [joint training](<4 Documentation/joint_training.md>) for pooled BCData and SHIDC-B-Ki-67 experiments.

## 5. Additional experiments

| Guide | Reproduction scope |
| --- | --- |
| [SHIDC-B-Ki-67-V2](<4 Documentation/shid_v2.md>) | Patient-disjoint partition, joint training and separate classification |
| [Cross-dataset evaluation](<4 Documentation/cross_dataset.md>) | Frozen Phikon-v2 anatomical models tested on another dataset |
| [Traditional machine learning](<4 Documentation/ml_baselines.md>) | Fourteen classifiers, validation-based grid search and three-seed evaluation |
| [Phikon-v2 fine-tuning](<4 Documentation/phikon_finetuning.md>) | Joint binary models, three-class SHIDC-B-Ki-67 models, parameter counts and timings |
| [Visual-feature assignments](<4 Documentation/feature_mapping.md>) | Ten random permutations and the original Phikon-v2 assignment |
| [Local explanations](<4 Documentation/phikon_xai.md>) | Grad-CAM, SHAP and LIME for the fine-tuned anatomical model |

Each guide separates reconstruction from saved results, prediction with selected weights and new training. Raw images are required for partition creation and fine-tuning.
