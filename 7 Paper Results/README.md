# Paper figures and tables

Complete the [environment setup](<../1 Environment/README.md>) and [artifact restoration](<../4 Trained Models/README.md>), then run from the repository root:

```bash
python -B "7 Paper Results/reproduce_paper.py"
```

The command reconstructs Figures 1-13 and Tables 1-16 from archived results on CPU, without training models or downloading raw images. It writes PDF/PNG figures, TeX tables and `reproduction.json` to `7 Paper Results/2 Outputs/<timestamp>/`. Figure 1 export requires `rsvg-convert`.

Use `--family iris` or `--family ki67` for a subset, `--tag <new-name>` to name the output directory, and `--dry-run` to inspect the commands. Choose a new tag for each execution. Intermediate calculations remain under `9 temp_intermediate/paper-reproduction/<tag>/` and `6 Ki67/6 Workspace/paper-reproduction/<tag>/`.

| Paper item | Generator or guide | Evidence |
| --- | --- | --- |
| Figure 1 | [Editable schematic](<1 Generators/schematic/README.md>) | Draw.io cells, archived activity snapshots and anatomical image assets |
| Table 1 | [Patient partition](<../6 Ki67/4 Documentation/shid_v2.md>) | SHIDC-B-Ki-67-V2 patient and annotation counts |
| Figures 2-7, Tables 2 and 4 | [Iris figures](<1 Generators/condensed_figures.py>), [Iris tables](<1 Generators/iris_tables.py>) | Learning curves, predictions and OOF aggregates |
| Tables 3 and 15 | [Iris topology and classifiers](<../8 Documentation/topology_controls.md>) | Four OOF evaluations per classifier, selected hyperparameters, model sizes and timings |
| Figures 8-9, Table 5 | [Joint Ki-67 results](<../6 Ki67/2 Source/paper_joint.py>), [table generator](<../6 Ki67/2 Source/paper_revision_tables.py>) | Joint frozen-feature models using either SHIDC-B-Ki-67 partition |
| Tables 6-7 | [Classification analysis](<../6 Ki67/4 Documentation/classification_analysis.md>) | Class metrics, image-bootstrap intervals and validation-fitted calibration |
| Figure 10, Table 8 | [Binary matrices](<../6 Ki67/2 Source/paper_binary_confusions.py>), [table generator](<../6 Ki67/2 Source/paper_revision_tables.py>) | Separately fitted binary models |
| Table 9 | [Cross-dataset evaluation](<../6 Ki67/4 Documentation/cross_dataset.md>) | Four source-to-test directions and four tasks |
| Figure 11, Table 10 | [Native-RGB comparison](<../6 Ki67/2 Source/paper_rgb.py>) | Native-RGB and frozen-feature models |
| Tables 11-13 | [Traditional machine learning](<../6 Ki67/4 Documentation/ml_baselines.md>) | Hyperparameter grids, test metrics, model sizes and timings |
| Figure 12, Table 14 | [Phikon-v2 fine-tuning](<../6 Ki67/4 Documentation/phikon_finetuning.md>) | Joint binary anatomical and linear models |
| Figure 13 | [Local explanations](<../6 Ki67/4 Documentation/phikon_xai.md>) | Grad-CAM at block 12, SHAP and LIME |
| Table 16 | [Literature comparison](<1 Generators/literature_comparison.py>) | Frozen and fully fine-tuned anatomical predictions and published VLM results |

The final exported filenames follow the manuscript numbering. Some individual campaign generators retain older filenames internally. `reproduce_paper.py` maps those files to the current numbers. The [figure catalog](<../4 Trained Models/1 Catalogs/figure-provenance.json>) identifies the exact archived PDFs used in the manuscript. Rasterization and fonts can vary by plotting-library version. Archived PDFs preserve the published layout, while the generators reproduce its numerical content.

The additional [Iris graph analysis](<../8 Documentation/topology_controls.md>) and [Phikon-v2 assignment analysis](<../6 Ki67/4 Documentation/feature_mapping.md>) reconstruct the comparisons reported in the text. Their guides include commands for new training. The [schematic source](<1 Generators/schematic/connectome-learning-overview.drawio>) is editable in diagrams.net.
