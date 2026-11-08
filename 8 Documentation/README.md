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
| [Topology controls and Iris classifiers](topology_controls.md) | Expanded graph controls, paired differences and the 14-classifier comparison |
| [Input assignments](input_assignment.md) | Original and randomized feature assignments in anatomical Iris classification |
| [Experiment plans](benchmark_matrix.md) | Seven plan files and generation of new plans |
| [Evaluation](evaluation.md) | Grouped partitions, scaling, selection, metrics and controls |
| [Repeated evaluation](repeated_evaluation.md) | Out-of-fold aggregation and conditional bootstrap intervals |
| [Ki-67 classification analysis](<../6 Ki67/4 Documentation/classification_analysis.md>) | Class prevalence and metrics, image-bootstrap intervals and temperature scaling |
| [SHIDC-B-Ki-67-V2](<../6 Ki67/4 Documentation/shid_v2.md>) | Patient-disjoint partition and frozen-feature experiments |
| [Cross-dataset evaluation](<../6 Ki67/4 Documentation/cross_dataset.md>) | Source-only training and evaluation on another dataset |
| [Traditional machine learning](<../6 Ki67/4 Documentation/ml_baselines.md>) | Grids, selected models and computation |
| [Phikon-v2 fine-tuning](<../6 Ki67/4 Documentation/phikon_finetuning.md>) | Joint binary and separate three-class models |
| [Visual-feature assignments](<../6 Ki67/4 Documentation/feature_mapping.md>) | Phikon-v2 feature permutations with fixed anatomical connectivity |
| [Local explanations](<../6 Ki67/4 Documentation/phikon_xai.md>) | Grad-CAM, SHAP and LIME |
| [Permutation inference](permutation_protocol.md) | Five grouped target-permutation tests |
| [Computational cost](full_graph_compute.md) | Recorded hardware, timing boundaries and optional profiling |

The Ki-67 [protocol](<../6 Ki67/4 Documentation/protocol.md>) and [joint-training guide](<../6 Ki67/4 Documentation/joint_training.md>) describe image preprocessing and BRAP-Net.

## Results and figures

| Document | Scope |
| --- | --- |
| [Paper evidence](manuscript_evidence.md) | Artifact locations for reported and complementary results |
| [Saved-prediction evaluation](<../5 Iris/3 Evaluation/README.md>) | Recomputing Iris OOF metrics and comparisons |
| [Paper reproduction](<../7 Paper Results/README.md>) | Commands for Figures 1-13 and Tables 1-16 |
