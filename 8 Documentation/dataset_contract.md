# Dataset specifications and fixed inputs

The five Iris tasks use the corrected 150-flower resource supplied by scikit-learn. Iris requires no separate download. The specification files are in `2 Data/2 Manifests/datasets/`; image datasets follow the [Ki-67 protocol](<../6 Ki67/4 Documentation/protocol.md>).

| Identifier | Target | Input order | Target unit |
| --- | --- | --- | --- |
| `iris` | Species | Sepal length, sepal width, petal length, petal width | Class label |
| `iris_sepal_length` | Sepal length | Sepal width, petal length, petal width, zero | cm |
| `iris_sepal_width` | Sepal width | Sepal length, petal length, petal width, zero | cm |
| `iris_petal_length` | Petal length | Sepal length, sepal width, petal width, zero | cm |
| `iris_petal_width` | Petal width | Sepal length, sepal width, petal length, zero | cm |

`iris_regression` is an archived alias for `iris_petal_width`. Regression excludes the target measurement and species label from its inputs.

## Loading a specification

An experiment's `dataset` field accepts an identifier, a JSON specification path or a specification object. For example:

```python
from flybrain.datasets import prepare_dataset

data = prepare_dataset("iris_petal_length", seed=1729, outer_fold=2)
```

The returned object contains raw and transformed arrays, groups, partition indices and metadata. Supplying `output_dir=` also saves `dataset.npz` and `metadata.json`. The metadata identifies the original resource and effective arrays by SHA-256, records the scikit-learn version, and names the target, units and input columns.

Identical retained inputs form indivisible groups even when targets differ. Group counts are 149 for classification and 143, 145, 145 and 144 for the four regressions in the table's order. The [evaluation guide](evaluation.md) defines grouped splits, training-derived feature/target scaling and conversion back to original units.

## CSV adapter for a new dataset

A new JSON specification can contain:

```json
{
  "name": "measurements",
  "source": "csv",
  "task": "regression",
  "path": "observations.csv",
  "feature_columns": ["measurement_a", "measurement_b", "measurement_c"],
  "target_column": "measurement_d",
  "target_units": "cm",
  "group_column": "individual_id"
}
```

The CSV path is relative to the specification JSON, or to the repository root when the specification is passed as a Python dictionary. `group_column` is optional. Entity groups and exact-input groups are merged transitively, preventing either from crossing partitions. Neither the target nor the group identifier can be an input feature.

The adapter accepts one to four finite numerical inputs and at most three classes. Remaining channels receive zeros. Classification uses `task="classification"`; an optional `classes` list fixes the order of two or three string labels, otherwise lexicographic order is recorded. Missing values, infinities and invalid columns are rejected. At least five independent groups and sufficient support for inner splitting are required. There is no imputation or automatic feature selection.

Metadata records the source CSV hash and fingerprints of entity identifiers and resolved groups. Use the [plan-generation API](benchmark_matrix.md#python-api) to define experiments for the new dataset.

## Permutation and domain options

`prepare_dataset(..., label_permutation_seed=10000)` exchanges complete target vectors between equal-sized groups before splitting. The same seed produces the same target assignment across all outer folds. Metadata retains the original targets and donor mapping. The paper uses seeds 10000–10198; the [permutation protocol](permutation_protocol.md) defines the test and its assumptions.

The optional `split_spec` selects an input-defined domain:

| `kind` | Additional fields | Behavior |
| --- | --- | --- |
| `feature_quantile_holdout` | `feature`, `quantile`, `tail` | Hold out the declared input tail; split the remaining groups into training and validation |
| `feature_quantile_transfer` | Same fields plus `phase` | Use `source` or `adapt` partitions with a shared target-domain test |

`feature` must be a retained input, never the target or padding; `tail` is `lower` or `upper`. Domain splits require `outer_fold=0` or `None`. Entity groups stay intact. Target-only training, saved as `reset`, uses the `adapt` partition without source-trained gains. All three training conditions retain source-training feature and target scales. These are domain holdouts, not complete OOF evaluations. The paper's median-tail design and initialization of each phase are in the [transfer protocol](round4_protocol.md#transfer-within-iris).
