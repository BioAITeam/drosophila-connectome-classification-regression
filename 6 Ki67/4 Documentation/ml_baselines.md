# Frozen-feature Ki-67 machine learning comparisons

The published comparison contains 336 classifier configurations: three dataset partitions (BCData, original SHIDC-B-Ki-67 and patient-disjoint SHIDC-B-Ki-67-V2), two frozen extractors (DINOv2-base and Phikon-v2), four classification tasks, and 14 classifier families. Native RGB experiments are outside this published ML package.

The classifier families are KNN, decision tree, bagging, random forest, AdaBoost, support vector classifier, Extra Trees, XGBoost, logistic regression, gradient boosting, QDA, LDA, Gaussian naive Bayes and a 32-unit MLP. Frozen feature matrices and training-only standardizers are identical to those used by the corresponding anatomical connectome models. The anatomical comparison uses separately trained dataset models, not the jointly trained BCData + SHIDC architecture.

## Selection, repetition and interpretation

GridSearchCV with PredefinedSplit fits each candidate on the original training partition and evaluates it on the existing validation partition. Validation balanced accuracy selects hyperparameters. Each grid varies at most three parameters and at most three values per parameter. Stochastic searches use seeds 1729, 2718 and 3141; deterministic searches run once. Every selected configuration is then evaluated after three fits on training data alone. Additional deterministic fits reuse the chosen configuration without repeating the search. Test results never select hyperparameters.

The archive contains 13,896 candidate fits represented by their search records and 1,008 selected fitted models. Mean and sample standard deviation summarize the three selected training seeds. Repeated seeds reuse the same observations and do not create additional test images. The complete grid specification and selected parameters for every context are retained in the report outputs.

## Rebuild Tables 11-13 without fitting models

Install the repository and restore the Zenodo artifact links as described in the main README, then run from the repository root:

```bash
python -m ki67.ml_baselines report
```

The command reads `6 Ki67/5 Outputs/ml-baselines` and writes to `6 Ki67/6 Workspace/ml-baselines/summary`. Use `--output` with another repository-relative Workspace path to keep reports separate. The report verifies the 1,008 selected model-file hashes, recomputes metrics from their 2,016 validation/test prediction files, and checks sample identifiers, targets and feature fingerprints. It additionally recomputes the matched anatomical connectome comparisons from saved predictions. It neither deserializes model objects nor trains models.

The main outputs are:

| Files | Content |
| --- | --- |
| `Table-10.tex`, `table10_ml_grid.csv`, `table10_ml_grid.json` | Hyperparameter grids and selected three-class Phikon-v2 configurations |
| `Table-11.tex`, `table11_ml_baselines.csv`, `table11_ml_baselines.json` | Test balanced accuracy for both extractors, all tasks, all classifiers and matched anatomical models |
| `Table-12.tex`, `table12_ml_compute.csv`, `table12_ml_compute.json` | Model-specific fitted state and search, selected training and prediction times |
| `all_contexts_best_hyperparameters.csv` | Full 336-context grid, selection and seed information |
| `model_complexity_and_compute.csv` | Per-selected-model fitted state, storage and timings |
| `prediction_verification.json`, `connectome_comparison_audit.json` | Reproduction checks and matched feature provenance |
| `counts.json`, `report_status.json`, `table_schema.json` | Campaign counts, completeness and table definitions |

CSV and JSON values remain unrounded. LaTeX floats contain the same values formatted for the manuscript and retain stable table labels. They require `booktabs` and `makecell`, as used in the paper template.

Model size is specific to each estimator: trees retain thresholds and leaves, SVC retains support vectors and fitted coefficients, and KNN retains observations. QDA/LDA counts include redundant stored statistics and are not counts of independent free parameters. Frozen extractor parameters are reported separately. Search times sum the 24 dataset/extractor/task contexts per classifier; selected-model training and test times are medians over 72 fits. They exclude frozen feature extraction, and test times cover the complete test partition.

## Input audit and optional reproduction of training

```bash
python -m ki67.ml_baselines audit
python -m ki67.ml_baselines run --workers 4 --threads 1
python -m ki67.ml_baselines repeat --workers 4
```

`audit` checks the archived feature inputs without fitting. `audit --verify-annotations` additionally needs locally restored annotation files. `run` and `repeat` explicitly train CPU models into a fresh Workspace campaign. `KI67_ML_WORKSPACE` can select another repository-relative directory under `6 Ki67/6 Workspace`. These commands never write to the archived outputs or original image datasets. Use the versions recorded in the package's `requirements.txt` and the archived runtime provenance when reproducing training or loading the trusted `best.joblib` files.

The archive keeps selected models, validation/test scores and predictions, grids, selection records, repeated-fit records and runtime provenance. Shared feature matrices and extractor weights remain in the common Ki-67 archive. Publication remaps metadata paths and dependent JSON fingerprints to repository-relative locations; selected model and prediction bytes remain identical to the completed experiments, with hashes recorded in `provenance/publication.json`.

The report generator retains its campaign filenames `Table-10.tex` through `Table-12.tex`. The paper reproduction command exports these as manuscript Tables 11-13.
