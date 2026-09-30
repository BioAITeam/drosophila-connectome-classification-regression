# Repeated partitions, optimization seeds, and graph controls

A complete out-of-fold (OOF) evaluation combines all five outer folds for one fixed family, partition seed, optimization seed, and graph seed. Each observation appears once as held-out test. Each fold retains its inner validation split for checkpoint selection. Aggregation selects no checkpoint, hyperparameter, or winning family from test results.

| Dimension | What changes | Interpretation |
| --- | --- | --- |
| `split_seed` | Grouped outer and inner partitions | Sensitivity to dividing the same observations |
| `optimization_seed` | Minibatch order and declared training randomness | Optimization repeatability under a fixed partition |
| `rewire_seed` | Edge-partner changes in a graph control | Sensitivity to artificial topologies; `null` denotes original anatomy |
| `fold` | Outer held-out fold, from 0 to 4 | One part of a complete OOF evaluation |

The paper crosses two partition seeds with two optimization seeds, giving four OOF evaluations from 20 fits per repeated condition. These evaluations reuse the same 150 flowers. Duplicate-input groups remain intact in splitting and resampling; rewired graphs are artificial controls, not additional anatomical specimens.

## Aggregation API

The following example aggregates the five saved folds of the anatomical 120-epoch classifier for partition seed 1729 and optimization seed 0. Run it from the repository root after artifact restoration.

```python
import json
from pathlib import Path
from flybrain.aggregation import aggregate_campaign

campaign = json.loads(Path(
    "5 Iris/5 Outputs/campaigns/round4-core/campaign.json"
).read_text())
records = [
    run for run in campaign["runs"]
    if run["family"] == "iris_anatomical_120"
    and run["split_seed"] == 1729 and run["optimization_seed"] == 0
]
aggregate = aggregate_campaign(
    "5 Iris/6 Workspace/reports/anatomical-120-p1729-s0", records,
    n_bootstrap=1000, bootstrap_seed=1729
)
```

The aggregator reads each run's `config.json`, `dataset/metadata.json`, `dataset/dataset.npz`, `predictions.npz`, and results. It verifies original array fingerprints, seeds, task identity, split indices, group separation, and exact target alignment. Predictions are restored to original `row_index` order; concatenation in fold order alone does not guarantee alignment.

Missing folds produce an `incomplete` evaluation with explicit missing-fold and coverage records. It receives no complete-OOF metric and is excluded from complete-evaluation bootstraps. Duplicate folds, test overlap, wrong target units, and inconsistent source data cause errors. A family containing complete and incomplete evaluations is `partially_complete`; summaries use only complete evaluations.

## Outputs

| Artifact | Content |
| --- | --- |
| `aggregate.json` | Evaluation dimensions, present/missing folds, recorded selected epochs, OOF metrics, controls, seed summaries, paired comparisons, and interval scope |
| `aggregate.md` | Initial and selected metrics, descriptive variation, and paired changes, separated by classification/regression |
| `oof/*.npz` | Original row indices, groups, outer-fold assignments, targets, predictions, shared controls, and the per-fold training-derived constant reference |

Metrics are recomputed from pooled predictions rather than averaging fold metrics. Classification reports accuracy, balanced accuracy, and macro F1. Regression reports RMSE and MAE in original units and dimensionless R². Undefined R² remains `null`. The regression constant reference uses each fold's training-target mean for its held-out rows, not the mean of the full dataset.

## Descriptive variation

Means, sample standard deviations, minima, and maxima summarize metrics across complete evaluations. They describe sensitivity on shared observations. The standard deviation is not divided by the square root of the seed count, and seeds are not treated as independent sampled datasets.

`hierarchical_mean` weights partition seeds equally, then optimization seeds within each partition equally, then available graph instances within each combination equally. Testing extra graphs only under one optimization seed therefore does not give that seed disproportionate weight. An incomplete factorial design cannot separate every effect or interaction; the mean remains conditional on tested combinations.

Predictions from different seeds are never ensembled into a new predictor. Aggregation averages metrics or paired metric differences, retaining the interpretation of each individual network with its fixed interfaces.

## Paired bootstrap and between-family comparisons

Within-model change is selected minus initial performance on the same observations. The group bootstrap samples original groups with replacement and applies the same sampled indices to every paired evaluation. It computes a metric difference per evaluation, then the hierarchical mean. Percentiles 2.5 and 97.5 form the conditional 95% interval. The primary displayed metric is balanced accuracy for classification and RMSE for regression; lower RMSE is better.

Family comparisons require the same dataset, partition seed, optimization seed, and actual test assignment for every observation. Shared graph seeds are paired where applicable. An anatomical model compared with several artificial graphs uses each available pair with balanced hierarchical weights. The exact matched and unmatched combinations are retained. Between-family differences are always **B minus A** according to the stored names.

This bootstrap resamples already computed predictions and **does not retrain models**. It does not capture all dependence from overlapping training sets or uncertainty from collecting new training data. Nor does it estimate variation between anatomical specimens. Conditional intervals accompany descriptive seed variation; they are not used to select a test-winning configuration after many comparisons. Confirmatory inference in the paper is limited to the separately declared [five grouped permutation tests](permutation_protocol.md).
