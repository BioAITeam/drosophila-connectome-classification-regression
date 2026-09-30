# Paper evidence

The paper's numerical evidence is distributed in `zenodo/payload/`. Complete [artifact restoration](<../4 Trained Models/README.md>) before using the locations below.

BRAP-Net denotes joint BCData and SHIDC-B-Ki-67 training with a frozen Phikon-v2 extractor and an anatomical or rewired connectome. Separate-dataset models are identified by their extractor and graph.

## Models and manuscript results

| Evidence | Location |
| --- | --- |
| Iris checkpoints, configurations and evaluated folds | [Iris model catalog](<../4 Trained Models/1 Catalogs/model-index.csv>) |
| Ki-67 checkpoints and experiment roles | [Ki-67 model catalog](<../4 Trained Models/1 Catalogs/ki67-model-index.csv>) |
| Manuscript PDFs and their source artifacts | [Figure provenance](<../4 Trained Models/1 Catalogs/figure-provenance.json>) |
| Iris probabilities, curves and regression predictions | `5 Iris/5 Outputs/paper-results/results-20260914T025839Z-ilnj5d7n/` |
| Joint Ki-67 runs and per-epoch test predictions | `6 Ki67/5 Outputs/joint-bcdata-shid-v1/` |
| Separate Ki-67 runs and per-epoch test predictions | `6 Ki67/5 Outputs/runs/` |

Use the [paper reproduction guide](<../7 Paper Results/README.md>) to reconstruct Figures 1–10 and Tables 1–5. Reconstructed plots use saved results; prediction replay uses the corresponding checkpoint, graph and preprocessing state. Floating-point probabilities can vary across devices or library versions, so archived predictions remain the numerical reference.

## Iris campaign records

The seven plans contain 6,390 runs of five tasks on the same 150 flowers. Their design is in the [Iris protocol](round4_protocol.md); the [plan index](benchmark_matrix.md) lists run counts. Under `5 Iris/5 Outputs/campaigns/`, the following files provide the full numerical results:

| Directory | Result files | Scope |
| --- | --- | --- |
| `round4-core/` | `aggregate.json`, `aggregate.md` | Anatomical budgets and rewired controls |
| `round4-full/` | `aggregate.json`, `aggregate.md` | Full graph with native and matched inputs |
| `round4-comparison/common-120/` | `aggregate.json`, `selection_provenance.json` | Matched 120-epoch comparisons |
| `round4-controls/` | `aggregate.json`, `aggregate.md` | Frozen gains and training-target shuffles |
| `round4-null/` | `permutation-analysis.json`, `permutation-analysis.csv` | Five observed procedures and 199 permutations per task |
| `round4-lif/` | `aggregate.json`, `aggregate.md` | Complementary spiking models |
| `round4-local/` | `aggregate.json`, `aggregate.md` | Complementary local-plasticity models |
| `round4-transfer/` | `transfer_analysis.json`, `transfer_analysis.csv` | Source, adapted and target-trained models |

Each campaign also retains `campaign.json` and its run records. The two-campaign `round4-comparison/` aggregate reuses the 510 core and 120 full-graph fits without additional training. The `common-120/` subset retains 120 fits and 24 complete out-of-fold (OOF) evaluations with shared partition/optimization settings.

## Full-graph regression detail

The manuscript summarizes these results as ranges. Each row below uses 120 epochs, partition seed 1729, optimization seed 0 and one complete five-fold OOF evaluation. RMSE is in centimeters; R² is dimensionless.

| Target | Subgraph R² | Subgraph RMSE | Full native R² | Full native RMSE | Full matched R² | Full matched RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sepal length | 0.6396 | 0.4955 | 0.0039 | 0.8237 | -0.0403 | 0.8418 |
| Sepal width | 0.4643 | 0.3180 | 0.0990 | 0.4123 | 0.0797 | 0.4167 |
| Petal length | 0.8851 | 0.5965 | 0.1132 | 1.6568 | 0.0696 | 1.6971 |
| Petal width | 0.8403 | 0.3036 | 0.0982 | 0.7214 | 0.0596 | 0.7367 |

A single OOF evaluation does not estimate variation between seeds. Matched inputs preserve the subgraph's input identifiers while retaining the full graph's surrounding connections and incoming-contact normalization; this comparison does not isolate neuron count.

## Ki-67 scope

The reported analyses comprise 168 separate frozen-feature fits (96 classification and 72 auxiliary regression fits), 12 joint three-class fits and 12 native-RGB three-class fits. The catalog distinguishes these experiment roles. The [Ki-67 protocol](<../6 Ki67/4 Documentation/protocol.md>) specifies annotation-derived targets, preprocessing and checkpoint selection; the [Ki-67 figure guide](<../6 Ki67/4 Documentation/manuscript_confusion_matrices.md>) identifies the complementary pixel-input results.

## Interpretation of saved endpoints

- Iris regression selects by validation RMSE; auxiliary Ki-67 regression selects by validation MAE in original units. Both train with mean squared error on scaled targets.
- Iris test endpoints cover initialization, selected checkpoints and controls. Ki-67 stores test predictions every epoch. Test metrics do not select checkpoints.
- Repeated evaluations reuse the same observations. Seed summaries average metrics without ensembling predictions or adding independent samples.
- Iris graph seed 241 was chosen retrospectively for the descriptive rewired display. Joint Ki-67 extractor selection instead used combined validation balanced accuracy.
- Activity reset clears transient neuronal state while retaining gains. The saved transfer phase `reset` means target-domain training from initial gains, a different intervention.
