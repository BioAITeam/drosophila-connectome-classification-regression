# Ki-67 classification analysis

Tables 6-7 are reconstructed from the original multiclass model predictions and archived logits. Table 6 reports class prevalence and selected-checkpoint precision, recall and F1 for joint Phikon-v2 BRAP-Net with anatomical or rewired connectivity and for separately trained anatomical RGB models. Table 7 compares the original and temperature-scaled probabilities of the joint Phikon-v2 models on pooled test data.

## Reproduction

Complete the [environment setup](<../../1 Environment/README.md>) and [artifact restoration](<../../4 Trained Models/README.md>). From the repository root, choose a new destination:

```bash
python -B -m ki67.paper_classification_analysis \
  --outputs "6 Ki67/5 Outputs" \
  --destination "6 Ki67/6 Workspace/classification-analysis" \
  --resamples 10000 --seed 1729
```

The optional `--analysis` argument selects the archived calibration evidence; its default is `6 Ki67/5 Outputs/paper-classification-analysis/`, restored from `zenodo/payload/ki67/outputs/paper-classification-analysis/`. Source predictions, logits and checkpoints remain read-only. The command evaluates the saved predictions and refits temperatures without training models or loading images.

| Output | Contents |
| --- | --- |
| `Table-4.tex`, `Table-5.tex` | Generator filenames, exported as manuscript Tables 6-7 by the paper reproduction command |
| `results.json`, `metrics_with_ci.csv` | Per-seed metrics, seed means and confidence intervals |
| `prevalence.csv` | Class counts and prevalence by dataset and partition |
| `calibration.json` | Temperature factors, per-seed probability metrics and reliability bins |
| `bootstrap_image_weights.npz` | Image multiplicities for bootstrap replicates |
| `bootstrap_metric_replicates.npz` | Recomputed metric values for each replicate |

The [paper reproduction command](<../../7 Paper Results/README.md>) includes this analysis and exports Tables 1-16. Table 16 includes the archived literature comparison.

## Cohorts and aggregation

BCData and SHIDC-B-Ki-67 contribute 402 and 700 test images, respectively, giving 1,102 pooled observations. The corresponding validation partitions contain 133 and 331 images, giving 464 observations. Training, validation and test class prevalence are retained separately. Class order and annotation-derived targets follow the [data specification](data_format.md).

Each metric is calculated separately for seeds 1729, 2718 and 3141 before averaging. Predictions are not ensembled, and the three evaluations do not increase the number of observations. Undefined precision, recall or F1 is set to zero (`zero_division=0`). The numerical records also include the joint DINOv2 comparisons and separate-dataset frozen-feature and RGB models; these remain distinct training configurations.

## Calibration and uncertainty

One positive scalar temperature is fitted by minimizing negative log-likelihood on the same 464 validation images used for checkpoint selection for each of the six selected joint Phikon-v2 checkpoints: two graph configurations and three seeds. Each fitted factor is applied unchanged to that checkpoint's test logits. Calibration does not alter the checkpoint, fitted gains or predicted class.

Confidence intervals use 10,000 image-bootstrap replicates with random seed 1729. Images are sampled with replacement separately within BCData and SHIDC-B-Ki-67; pooled replicates retain 402 and 700 draws. The same image multiplicities are shared across seeds, models and analysis stages. Each replicate recomputes the per-seed metrics and then their mean. Reported intervals are the 2.5th and 97.5th percentiles. Temperature factors remain fixed during test resampling.

These intervals are conditional on the fitted models, fixed partitions and validation-fitted calibration factors, under an independent-image assumption. They do not estimate patient-level generalization or uncertainty from retraining, partition selection or temperature fitting. The [joint-training guide](joint_training.md) describes the patient-overlap limitations of the original partitions.
