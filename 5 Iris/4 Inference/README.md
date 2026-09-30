# Iris classification and regression inference

## 1. Prerequisites

Complete the [environment setup](<../../1 Environment/README.md>) and [artifact restoration](<../../4 Trained Models/README.md>). The scripts load selected checkpoints, training scalers and graphs from the restored Zenodo payload.

Inference runs on CPU without network access or parameter updates. Run the commands below from the repository root with the installed environment's `python`.

## 2. Prepare one flower or a CSV

Pass one flower with `--values`, or use `--inputs` for a CSV containing one or more flowers. CSV files have no header and contain one flower per row. All measurements are in centimeters; do not scale them manually.

| Task | Input columns, in order | Example CSV |
| --- | --- | --- |
| Species classification | Sepal length, sepal width, petal length, petal width | `iris_inputs.csv` |
| Sepal length regression | Sepal width, petal length, petal width | `iris_sepal_length_inputs.csv` |
| Sepal width regression | Sepal length, petal length, petal width | `iris_sepal_width_inputs.csv` |
| Petal length regression | Sepal length, sepal width, petal width | `iris_petal_length_inputs.csv` |
| Petal width regression | Sepal length, sepal width, petal length | `iris_regression_inputs.csv` |

Regression takes exactly three measurements, excluding the quantity being predicted. The script adds the model's zero padding channel automatically.

## 3. Predict species

One flower:

```bash
python -B "5 Iris/4 Inference/predict_classification.py" \
  --values 5.1 3.5 1.4 0.2 \
  --save "5 Iris/6 Workspace/inference/species-one.npz"
```

Several flowers from a CSV:

```bash
python -B "5 Iris/4 Inference/predict_classification.py" \
  --inputs "5 Iris/4 Inference/iris_inputs.csv" \
  --save "5 Iris/6 Workspace/inference/species-csv.npz"
```

The predictions contain the species and its probabilities in the order setosa, versicolor, virginica.

## 4. Predict a measurement

Choose `--target sepal_length`, `sepal_width`, `petal_length` or `petal_width`. Each target uses its own trained checkpoint and the corresponding three input columns from the table above.

Predict petal width for one flower:

```bash
python -B "5 Iris/4 Inference/predict_regression.py" \
  --target petal_width --values 5.1 3.5 1.4 \
  --save "5 Iris/6 Workspace/inference/petal-width-one.npz"
```

Predict sepal length from a CSV:

```bash
python -B "5 Iris/4 Inference/predict_regression.py" \
  --target sepal_length \
  --inputs "5 Iris/4 Inference/iris_sepal_length_inputs.csv" \
  --save "5 Iris/6 Workspace/inference/sepal-length-csv.npz"
```

Regression predictions are returned in centimeters.

## 5. Select a saved model

Each invocation uses one validation-selected checkpoint, with the following options:

| Option | Available values | Default |
| --- | --- | --- |
| `--graph` | `anatomical`, `degree241`, `rewired` (alias of `degree241`) | `anatomical` |
| `--split-seed` | `1729`, `2718` | `1729` |
| `--optimization-seed` | `0`, `1` | `0` |
| `--fold` | `0`, `1`, `2`, `3`, `4` | `0` |

The anatomical checkpoints use a 120-epoch classification budget and a regression budget of up to 600 epochs with early stopping. The rewired 241 checkpoints use 120-epoch budgets for both tasks and are available for partition seed `1729` and optimization seed `0`. An unavailable combination produces an error instead of substituting another model.

Classification checkpoints maximize validation balanced accuracy; regression checkpoints minimize validation root mean squared error (RMSE). The [model catalog](<../../4 Trained Models/1 Catalogs/model-index.csv>) records their selected epochs and paths. Other archived configurations remain accessible through `./run.sh --device cpu predict --help`.

## 6. Read the outputs

Each command creates three files with the `--save` filename stem. The destination must end in `.npz` and be under `5 Iris/6 Workspace/`; missing parent directories are created. Existing output files are preserved, so use a new filename for each invocation.

| File | Content |
| --- | --- |
| `.csv` | Input measurements and predictions, with one row per flower |
| `.npz` | Input and prediction arrays; classification includes class probabilities |
| `.json` | Predictions and model provenance, including checkpoint hash, selected epoch, graph and seed/fold settings |

Output rows follow input row order. Relative CSV paths are resolved from the current working directory; `--save` is relative to the repository root. Absolute paths are accepted, with the same restriction on the output directory. For more detail on all five tasks, see the [dataset contract](<../../8 Documentation/dataset_contract.md>).

The paper's out-of-fold results combine held-out predictions from all five outer folds; a single fold model does not reproduce that aggregate. See [evaluation](<../3 Evaluation/README.md>) to recompute these metrics. Small numerical differences can arise across devices or library versions.
