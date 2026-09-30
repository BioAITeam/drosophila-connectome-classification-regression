# Conditional grouped permutation inference

The Iris protocol declares five primary tests: species classification and four
regressions, predicting each measurement in cm from the other three. Species
classification uses out-of-fold (OOF) balanced accuracy; regression uses negative
OOF root mean squared error (RMSE) in centimeters. R² and mean absolute error (MAE) are additional descriptive
metrics. They do not increase the primary family beyond five tests.

Each task has one observed fit procedure and 199 null procedures, seeded
10000 through 10198. Every procedure refits five outer folds with split seed
1729, optimization seed 0, the declared common graph/configuration, and a
budget of 40 epochs. The initial state and every subsequent epoch retain
training and validation metrics. Validation selects the checkpoint, including
epoch zero. Outer test is evaluated only after that procedure is fixed.

The inferential null permutes complete target vectors between groups with
equal numbers of observations, using exact-feature duplicate groups (and any
declared entity groups). It conserves target multiplicities and the full
within-group target vector. Null labels are assigned once, before making
outer and inner splits, then retained across all five folds. Classification
stratification is therefore rerun using null labels. Scaling is fitted from
each null training fold. This is different from the diagnostic that shuffles
only training labels while leaving validation labels unchanged.

[`analyze_permutation_campaign(folder, records, expected_permutations=199)`](<../3 Connectome/flybrain/null_analysis.py>)
checks each declared role/task/seed, original feature and target fingerprints,
source hashes, grouping, target donor mapping, complete group-disjoint OOF
evaluations and the selected validation epoch. It reconstructs the grouped
splits and verifies target multiplicities. Missing folds/tasks, repeated
permutation identities and inconsistent evidence cause failure rather than a
partial p-value. OOF metrics are recomputed from saved predictions, with folds
combined into a single evaluation rather than treated as independent observations.

For a statistic where larger is better, the Monte Carlo p-value is

```text
p = (1 + number of null statistics >= observed statistic) / (B + 1)
```

Here, `B` is the number of null permutations.

Ties count as exceedances. With 199 permutations, the smallest attainable
unadjusted p-value is 0.005. Holm's step-down method controls the declared
five-test family at alpha 0.05 under its usual assumptions. All five raw and
adjusted values are reported. Comparisons between graph/hyperparameter
families are outside this confirmatory test family. No all-family pairwise
bootstrap is computed for these 5,000 fits.

Artifacts are `permutation-analysis.json`, `.csv`, `.md` and one compact
`permutation-oof-TASK.npz` per task. Each NPZ retains original row ordering,
groups, original targets, observed/null targets and predictions, permutation
IDs and fold assignments. JSON includes the run paths and selected epochs.

Iris already informed earlier exploration. These are conditional tests on
this dataset and the declared exchangeability groups, not evidence from an
untouched external dataset. A small corrected p-value does not demonstrate
generalization across datasets or animals, superiority over conventional ML,
or physiological fidelity of the computational learning rule.

Changing the budget, tasks or seed design requires a separately specified
protocol and matching analysis constraints.
