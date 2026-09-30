# Iris experiment plans

The seven plans in `5 Iris/1 Configuration/` define the paper's 6,390 computational runs. [`inventory.json`](<../5 Iris/1 Configuration/inventory.json>) records their counts and SHA-256 values. The [Iris protocol](round4_protocol.md) explains budgets, seeds and control conditions.

| Plan | Runs | Design |
| --- | ---: | --- |
| `plan_core.json` | 510 | Anatomical budgets and three seeds of each rewiring control |
| `plan_full.json` | 120 | Full graph with native and matched inputs |
| `plan_null.json` | 5,000 | Five tasks × observed/199 permuted targets × five folds |
| `plan_controls.json` | 50 | Frozen gains and fixed training-target permutations |
| `plan_lif.json` | 250 | Ten spiking configurations × five tasks × five folds |
| `plan_local.json` | 400 | Four local regimes × four conditions × five tasks × five folds |
| `plan_transfer.json` | 60 | Five tasks × two domains × two seeds × three training phases |
| **Total** | **6,390** | |

## Generate a new plan set

After [restoring the artifacts](<../4 Trained Models/README.md>), run:

```bash
./run.sh --device cpu benchmark --prepare
```

The generated plans are in `5 Iris/6 Workspace/configuration/` and reference the restored graphs in `2 Data/3 Processed/`. This command prepares configurations without training models.

To use newly prepared graphs, first follow [graph preparation](data.md#preparation-commands) and [graph controls](graph_controls.md#api-and-artifacts), then run:

```bash
./run.sh --device cpu benchmark --prepare \
  --graph-root "5 Iris/6 Workspace/graphs" \
  --output "5 Iris/6 Workspace/configuration/new-graphs"
```

`--graph-root` must identify a repository directory containing all nine NPZ files and their matching JSON sidecars. The command sets graph paths without constructing graphs. Follow the [Iris training guide](<../5 Iris/README.md>) to launch a plan.

Base task configurations are in `5 Iris/1 Configuration/tasks/`; expanded conditions are in `5 Iris/1 Configuration/recipes/<plan>/`. Cached rewired graphs use `topology_seed` to identify their preparation seed and receive no further rewiring during training.

## Python API

```python
from flybrain.benchmark import build_iris_plans

inventory = build_iris_plans("5 Iris/6 Workspace/configuration")
```

For a different dataset, `build_benchmark_plans(task_specs, output, graph_root=...)` accepts identifiers, JSON specification paths, specification objects, or records declaring `name`, `dataset`, `task` and `holdout_feature`. Follow the [dataset contract](dataset_contract.md) for columns, units, groups and the four-channel limit. The transfer-domain feature must be a retained input; by default it is the first retained feature. Use a separate plan directory for each new design.
