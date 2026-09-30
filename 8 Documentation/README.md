# Scientific documentation

For installation, data downloads, inference and paper reproduction, start with the [repository guide](../README.md). The documents below describe the scientific protocols and interpretation of results.

## Model and data

| Document | Scope |
| --- | --- |
| [Anatomical data](data.md) | MaleCNS sources, neuron selection and graph preparation |
| [Connectome model](model.md) | Fixed interfaces, neuronal dynamics and adaptable gains |
| [Graph controls](graph_controls.md) | Matched inputs and the two rewiring procedures |
| [Local plasticity](plasticity.md) | Complementary perturbation-and-feedback learning rule |
| [Dataset specifications](dataset_contract.md) | Iris task identifiers, input columns and CSV adapter |

## Experiments and evaluation

| Document | Scope |
| --- | --- |
| [Iris protocol](round4_protocol.md) | Tasks, budgets, seeds and complementary experiments |
| [Experiment plans](benchmark_matrix.md) | Seven plan files and generation of new plans |
| [Evaluation](evaluation.md) | Grouped partitions, scaling, selection, metrics and controls |
| [Repeated evaluation](repeated_evaluation.md) | Out-of-fold aggregation and conditional bootstrap intervals |
| [Permutation inference](permutation_protocol.md) | Five grouped target-permutation tests |
| [Computational cost](full_graph_compute.md) | Recorded hardware, timing boundaries and optional profiling |

The Ki-67 [protocol](<../6 Ki67/4 Documentation/protocol.md>) and [joint-training guide](<../6 Ki67/4 Documentation/joint_training.md>) describe image preprocessing and BRAP-Net.

## Results and figures

| Document | Scope |
| --- | --- |
| [Paper evidence](manuscript_evidence.md) | Artifact locations for reported and complementary results |
| [Saved-prediction evaluation](<../5 Iris/3 Evaluation/README.md>) | Recomputing Iris OOF metrics and comparisons |
| [Paper reproduction](<../7 Paper Results/README.md>) | Commands for Figures 1–10 and Tables 1–5 |
