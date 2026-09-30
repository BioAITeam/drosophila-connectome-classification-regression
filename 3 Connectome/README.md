# 3. Connectome

The `flybrain` package provides anatomical graph construction, fixed input/output populations, recurrent dynamics and internal synaptic-gain adaptation. Iris and Ki-67 use these shared components.

For inference with saved checkpoints, continue to [4 Trained Models](<../4 Trained Models/README.md>). Graph construction below is for preparing new experiments; the archived checkpoints already have matching graphs in Zenodo.

| Module | Role |
| --- | --- |
| `flybrain/download.py` | Source download and checksum verification |
| `flybrain/data.py` | Anatomical graph preparation |
| `flybrain/model.py` | Recurrent connectome and trainable gains |
| `flybrain/experiment.py` | Iris training, evaluation and model-state controls |
| `flybrain/iris_inference.py` | Archived Iris model selection and inference |
| `flybrain/paths.py` | Repository-relative paths and archived-path resolution |

The [model specification](<../8 Documentation/model.md>) defines the dynamics, gains and fixed interfaces. The [data provenance guide](<../8 Documentation/data.md>) documents anatomical selection, contact counts and sign assumptions.

## Prepare a new anatomical graph

After [installing the package](<../1 Environment/README.md>) and [downloading the original exports](<../2 Data/1 Raw/README.md>), run from the repository root:

```bash
./run.sh --device cpu prepare --mode visual_mb \
  --output "5 Iris/6 Workspace/graphs/visual_mb.npz"
./run.sh --device cpu prepare --mode full \
  --output "5 Iris/6 Workspace/graphs/full.npz"
```

`visual_mb` constructs the visual and mushroom-body subgraph; `full` retains all neurons with a recorded `superclass` annotation and the connections between them. Each command writes an NPZ file and a matching JSON provenance sidecar, and refuses to replace existing outputs. Graph construction reads the connectome exports without task labels.

Follow the [graph-control protocol](<../8 Documentation/graph_controls.md>) to derive matched-input and rewired graphs. Set `--graph-root` when [generating training plans](<../8 Documentation/benchmark_matrix.md>) to use the newly prepared graphs. Saved-checkpoint inference uses its matching archived graphs through `2 Data/3 Processed/`.
