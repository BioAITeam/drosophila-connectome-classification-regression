# Evaluation, scaling, and learning controls

This guide covers the five Iris tasks. The [dataset contract](dataset_contract.md) gives input columns and identifiers; [Ki-67 evaluation](<../6 Ki67/4 Documentation/protocol.md>) uses separate image partitions and selection rules.

## Data provenance and splits

Iris comes from the corrected resource distributed with scikit-learn. Each run records the library version, resource checksum, and the SHA-256 of the effective arrays. The resource includes the corrections to records 35 and 38 documented by its distributors: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) and the [scikit-learn loader documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.datasets.load_iris.html).

Rows with exactly identical retained inputs are grouped together even when their targets differ. A group cannot cross partitions. Classification uses stratified grouped splitting; regression uses grouped splitting without discretizing continuous targets. Five outer folds and four inner folds give approximately 60/20/20 training/validation/test proportions. Inner fold zero is always validation. Group constraints mean that Iris does not necessarily split into exactly 90/30/30 rows.

`outer_fold=0..4` evaluates each observation once as outer test. Combining these five fits forms one complete **out-of-fold (OOF) evaluation**. Fold predictions are joined by original row index. Metrics are then calculated from all 150 predictions, not averaged across folds. The [repeated evaluation protocol](repeated_evaluation.md) explains seed combinations and conditional uncertainty.

## Fixed scaling

Each feature is scaled by its training minimum and maximum and clipped to `[0, 1]`. Constant columns, including padding, map to zero. Metadata records the fraction of validation/test values outside the training range. Clipping can hide the magnitude of a domain shift.

Regression uses a fixed conversion derived from training targets:

```text
offset = (min(y_train_raw) + max(y_train_raw)) / 2
scale = (max(y_train_raw) - min(y_train_raw)) / 2
y = (y_raw - offset) / scale
y_raw = y * scale + offset
```

If training targets are constant, `scale=1`. Normalized validation/test targets and predictions are not clipped to `[-1, 1]`. Use `inverse_targets(prediction, dataset['metadata'])` when comparing against raw targets. Transfer experiments retain source-training scales for all conditions, including target-only training.

## Checkpoint selection and metrics

Training and validation metrics are saved at epoch zero and after every executed epoch. Validation selects the checkpoint using balanced accuracy for classification or root mean squared error (RMSE) for regression, with epoch zero eligible and the earliest checkpoint retained on a tie. Test is evaluated for the initial and selected checkpoints and the declared endpoint controls. There is no test curve used for epoch selection.

Classification reports accuracy, balanced accuracy, macro F1, and count confusion matrices with true classes in rows and predicted classes in columns. Balanced accuracy averages recall over classes present in the evaluated targets. Macro F1 uses all configured classes and assigns zero when a class has neither support nor predictions.

Regression reports RMSE and mean absolute error (MAE) in the original units, and dimensionless R². Perfect prediction has R² equal to one; negative values indicate more squared error than predicting the mean of the evaluated targets. That metric reference differs from the explicit training-mean baseline saved per fold. Undefined R², including constant-target cases, remains `None` rather than being replaced by zero.

The manuscript's ROC and precision-recall diagnostics use one-versus-rest softmax scores recovered from saved initial and selected checkpoints on their original outer-test inputs. Recovered class labels must match the original saved labels. Area under the ROC curve and average precision (AP) are distinct metrics. AP weights precision by successive recall increments, without trapezoidal interpolation. Macro averages give each species equal weight. Score recovery fits no model or probability calibration.

## Paired changes

`paired_bootstrap_delta(y_true, before, after, task=..., metric=..., groups=...)` samples the same observations for both predictions. When groups are supplied, duplicate-input groups are resampled intact. Its default percentile interval is 95%; the recorded difference is **after minus before**. Increasing accuracy, F1, or R² is favorable; decreasing RMSE or MAE is favorable.

These intervals are conditional on the saved models and evaluated cases. They do not refit models or make repeated uses of the same 150 flowers independent. See [repeated evaluation](repeated_evaluation.md) for aggregation across fits.

## Controls and their interpretation

| Control | Intervention | Question addressed |
| --- | --- | --- |
| Frozen gains | Disable parameter updates with the same input interface and dynamics | Does updating internal parameters improve over the initial model? |
| Initial gains restored | Restore initial parameters and clear transient state | Does the improvement depend on persistent parameter changes? |
| Activity reset | Keep learned gains, disable feedback, and clear transient activity | Is the prediction preserved without transient carryover? |
| No feedback | Suppress the local teaching signal | Does adaptation require the implemented feedback? |
| Shuffled feedback | Reuse the paired run's saved feedback values in a permuted sequence | Does the association between feedback and examples matter? |
| Zero input | Suppress dataset current | Are predictions dominated by baseline activity? |
| Input-row permutation | Keep complete feature vectors but break alignment with targets | Does prediction depend on the corresponding input sample? |
| Independent column permutations | Preserve marginal values but disrupt their joint structure | Does prediction depend on feature combinations? |
| Rewired graph | Change edge partners under declared constraints | Does a particular anatomical topology confer an advantage? |

Fixed training-target permutation leaves the original validation and test targets available, including for checkpoint selection. It is a diagnostic control; the [global permutation null](permutation_protocol.md) instead reruns the full procedure on permuted targets.

Shuffled local feedback reuses a precomputed, paired sequence to preserve the pulse values and count. Recalculating rewards along another trajectory would change that sequence and invalidate the pairing.

Graph controls are artificial graphs with separately recorded preservation checks. Rewiring can invalidate the anatomical meaning of an edge-index KC-to-MBON mask; the paper's topology comparisons adapt all retained edges. See [graph controls](graph_controls.md).

These checks locate adaptation within the simulated model. A positive result does not establish that a living fly learns Iris, that its anatomical wiring is uniquely advantageous, or that the simulated parameter changes reproduce biological memory.
