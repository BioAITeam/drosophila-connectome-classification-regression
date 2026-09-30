# Supervised Learning Through Synaptic Adaptation in a Drosophila Connectome Model for Classification and Regression

Research code for supervised adaptation of internal synaptic gains in a model of the MaleCNS *Drosophila* connectome. Experiments cover Iris species classification, regression of floral measurements, and Ki-67 prediction in breast cancer histopathology. BRAP-Net combines a frozen vision transformer (ViT) with the connectome model. Task-specific learning adjusts internal gains; the visual extractor and input/output interfaces remain fixed.

Link a Zenodo: URL

## Getting started

Clone the repository, then run commands from its root in Bash (Linux or WSL2):

```bash
git clone https://github.com/BioAITeam/drosophila-connectome-classification-regression.git
cd drosophila-connectome-classification-regression
```

1. [Install the environment](<1 Environment/README.md>).
2. [Restore the trained models](<4 Trained Models/README.md>): download `zenodo.zip` using the link above and extract it here. The resulting path must be `zenodo/payload/`. Follow that guide to create and verify the artifact links.
3. Predict new samples with the [Iris scripts](<5 Iris/4 Inference/README.md>) or [BRAP-Net image scripts](<6 Ki67/3 Inference/README.md>). Both workflows use CPU and saved weights.
4. [Reconstruct Figures 1–10 and Tables 1–5](<7 Paper Results/README.md>) from the archived evidence.

Inference and figure/table reconstruction use the restored artifacts; original image datasets are unnecessary. BRAP-Net inference takes the images you supply. To retrain from images or rebuild graphs from source, follow the [data guide](<2 Data/README.md>) before the corresponding experiment guide.

## Folder guide

| Folder | Read for |
| --- | --- |
| [1 Environment](<1 Environment/README.md>) | Installation, dependency versions and interpreter selection |
| [2 Data](<2 Data/README.md>) | Data sources, expected layouts and dataset specifications |
| [3 Connectome](<3 Connectome/README.md>) | Shared model implementation and anatomical graph construction |
| [4 Trained Models](<4 Trained Models/README.md>) | Restoring Zenodo, finding checkpoints and their dependencies |
| [5 Iris](<5 Iris/README.md>) | Iris classification, regression, inference and training |
| [6 Ki67](<6 Ki67/README.md>) | BRAP-Net inference and Ki-67 experiments |
| [7 Paper Results](<7 Paper Results/README.md>) | Regenerating the paper's figures and tables |
| [8 Documentation](<8 Documentation/README.md>) | Scientific protocols, evaluation and supporting evidence |

Cite the study using [CITATION.cff](CITATION.cff).

## License

Original code, documentation and project-authored research artifacts are available under the [Noncommercial Research and Education License](LICENSE), which requires citation. Datasets, frozen ViT weights and other third-party materials retain their source terms.
