# 5. Iris

Iris experiments evaluate species classification and prediction of each floral measurement from the other three. They use the shared `flybrain` implementation in [3 Connectome](<../3 Connectome/README.md>).

| Step | Folder | Purpose |
| --- | --- | --- |
| 1 | `1 Configuration/` | Frozen campaign plans and task configurations |
| 2 | `2 Execution/` | CPU campaign launcher and progress summaries |
| 3 | `3 Evaluation/` | Recomputing metrics from saved predictions |
| 4 | `4 Inference/` | Classification and four-target regression scripts, with CSV examples |
| 5 | `5 Outputs/` | Restored paper results and selected models, read-only |
| 6 | `6 Workspace/` | New campaigns, predictions and numerical summaries |

The [Iris protocol](<../8 Documentation/round4_protocol.md>) describes the 6,390 original runs, partitions, selection criteria and controls. The additional [input-assignment experiment](<../8 Documentation/input_assignment.md>) comprises 220 classification fits on the anatomical graph.

## 1. Prerequisites and inference

Follow the [environment guide](<../1 Environment/README.md>) and [artifact restoration guide](<../4 Trained Models/README.md>) before running commands from the repository root. The [inference guide](<4 Inference/README.md>) documents predictions from one flower or a CSV, including input order, model selection and output formats.

## 2. Evaluate saved predictions

Use the [evaluation guide](<3 Evaluation/README.md>) to recompute OOF metrics and paired comparisons, or [7 Paper Results](<../7 Paper Results/README.md>) to reproduce the figures and tables.

## 3. Run a new campaign

The six CPU campaigns contain 6,270 runs. The launcher defaults to 18 workers with one thread each; set `FLYBRAIN_WORKERS` for the resources available. For example, use two workers:

```bash
FLYBRAIN_WORKERS=2 bash "5 Iris/2 Execution/run_iris_cpu.sh"
```

Campaigns run sequentially in a timestamped directory under `5 Iris/6 Workspace/retrained/`, with a `round4-<campaign>.log` for each campaign. To resume, set `FLYBRAIN_CAMPAIGN_ROOT` to that directory; relative paths start at the repository root. Completed runs are reused, and interrupted runs restart from their declared initial state.

Run the full-connectome campaign on a free GPU:

```bash
./run.sh --device cuda --gpu 1 campaign-worker \
  --plan "5 Iris/1 Configuration/plan_full.json" \
  --output "5 Iris/6 Workspace/retrained/SESSION/round4-full"
```

Read progress for a new session with:

```bash
python -B "5 Iris/2 Execution/status_round4.py" \
  --campaigns "5 Iris/6 Workspace/retrained/SESSION" --brief
```

Replace `SESSION` in both commands with the directory name printed by the CPU launcher. Adjust `--gpu` to the available physical device.

After the full-graph campaign finishes, [aggregate its predictions](<3 Evaluation/README.md>) using its directory as `--sources`.

The supplied plans use the restored graphs. For newly prepared graphs, set `--graph-root` when [generating training plans](<../8 Documentation/benchmark_matrix.md>).

## Topology controls and traditional classifiers

The [topology and classifier guide](<../8 Documentation/topology_controls.md>) covers 30 degree-preserving rewired graphs, 30 random recurrent graphs and 14 traditional classifiers under the same grouped Iris partitions. Saved predictions reproduce the control comparisons and Tables 3 and 15 without retraining.
