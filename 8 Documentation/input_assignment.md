# Iris input-assignment control

This experiment compares the original Iris feature assignment with ten fixed random assignments to the same 730 visual input neurons. It retains the anatomical graph, output populations, training protocol and number of neurons receiving each feature: 201, 166, 193 and 170. Only the feature assigned to each input neuron changes.

Each assignment uses two partition seeds (1729 and 2718), two optimization seeds (0 and 1), five outer folds and 120 epochs. The 11 assignments therefore require 220 fits. Random assignments use seeds 10001 to 10010. Checkpoints maximize validation balanced accuracy. The [evaluation protocol](evaluation.md) describes grouping, preprocessing and out-of-fold aggregation.

## 1. Recompute the reported results

Install the [environment](<../1 Environment/README.md>) and follow the [artifact restoration guide](<../4 Trained Models/README.md>) to obtain the repository's `zenodo/` directory. Run these commands from the repository root with the environment activated:

```bash
source "1 Environment/environment.sh"
flybrain_environment "$PWD"
"$PYTHON_BIN" "5 Iris/2 Execution/run_input_mapping.py" analyze
```

The command reads `zenodo/payload/outputs/campaigns/iris-input-mapping/` and writes `5 Iris/6 Workspace/iris-input-mapping-analysis/`. Its outputs include fit records, repetition metrics, assignment summaries, OOF predictions, a JSON summary and a Markdown results table. The archived campaign remains unchanged.

Mean balanced accuracy is 92.00% for the original assignment and 94.17% across the ten random assignments. Their means range from 92.00% to 96.17%. The mean paired difference is 2.17 percentage points, with a conditional 95% interval of −1.03 to 5.52 points. This interval uses 1,000 paired bootstrap resamples of identical-input groups and is conditional on the fitted models, observations and sampled assignments.

## 2. Run the experiment again

```bash
"$PYTHON_BIN" "5 Iris/2 Execution/run_input_mapping.py" run --workers 16
```

Adjust `--workers` to the available CPU cores. Each worker uses one computational thread. New fits and summaries are saved under `5 Iris/6 Workspace/iris-input-mapping/`. Repeating the command resumes completed fits after checking their saved records. Use `--output "5 Iris/6 Workspace/NAME"` for a separate campaign.

The original assignment runs first and must reproduce the 20 archived reference fits before the random assignments begin. The supplied environment and unchanged model source are required for this comparison. To prepare assignments without training, use `prepare` instead of `run`.

## 3. Inspect the saved models

The archived campaign contains:

| Path within the campaign | Contents |
| --- | --- |
| `manifest.json` | Protocol, seeds, source checksums and assignment inventory |
| `mappings/` | Eleven graphs and neuron-to-feature assignments |
| `runs/` | 220 selected checkpoints, configurations, partitions, histories and predictions |
| `status/` | Fit locations, selected epochs and file checksums |
| `analysis/` | Reported metrics and OOF predictions |
| `provenance/` | Training script, recipe and source-graph metadata |

Paths stored in campaign metadata start at the repository root. To reload all selected checkpoints and compare their predictions with the saved results:

```bash
"$PYTHON_BIN" "5 Iris/2 Execution/run_input_mapping.py" verify
```

This command also checks validation selection and the original reference predictions. It does not retrain models or modify the archived campaign.
