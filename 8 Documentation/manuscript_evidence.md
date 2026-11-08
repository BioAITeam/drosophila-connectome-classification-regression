# Paper evidence

The paper's numerical evidence is distributed in `zenodo/payload/`. Complete [artifact restoration](<../4 Trained Models/README.md>) before using the locations below.

BRAP-Net combines Phikon-v2 with a connectome model. The archived records distinguish frozen-feature training, joint fine-tuning, dataset composition and anatomical or rewired connectivity.

## Models and manuscript results

| Evidence | Location |
| --- | --- |
| Iris checkpoints, configurations and evaluated folds | [Iris model catalog](<../4 Trained Models/1 Catalogs/model-index.csv>) |
| Ki-67 checkpoints and experiment roles | [Ki-67 model catalog](<../4 Trained Models/1 Catalogs/ki67-model-index.csv>) |
| Manuscript PDFs and their source artifacts | [Figure provenance](<../4 Trained Models/1 Catalogs/figure-provenance.json>) |
| Iris probabilities, curves and regression predictions | `5 Iris/5 Outputs/paper-results/results-20260914T025839Z-ilnj5d7n/` |
| Joint Ki-67 runs and per-epoch test predictions | `6 Ki67/5 Outputs/joint-bcdata-shid-v1/` |
| Separate Ki-67 runs and per-epoch test predictions | `6 Ki67/5 Outputs/runs/` |
| Ki-67 class metrics, bootstrap records and calibration logits | `6 Ki67/5 Outputs/paper-classification-analysis/` |

Use the [paper reproduction guide](<../7 Paper Results/README.md>) to reconstruct Figures 1-13 and Tables 1-16. Table 16 uses archived literature values together with prediction-derived model results. Reconstructed plots use saved results; prediction replay uses the corresponding checkpoint, graph and preprocessing state. Floating-point probabilities can vary across devices or library versions, so archived predictions remain the numerical reference.

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

A single OOF evaluation does not estimate variation between seeds. Matched inputs preserve the subgraph's input identifiers while retaining the full graph's surrounding connections and incoming-contact normalization; this comparison does not isolate neuron count.

## Ki-67 scope

The initial Ki-67 campaigns comprise 168 separate frozen-feature fits (96 classification and 72 auxiliary regression fits), 12 joint three-class fits and 12 native-RGB three-class fits. The catalog distinguishes these experiment roles. The [Ki-67 protocol](<../6 Ki67/4 Documentation/protocol.md>) specifies annotation-derived targets, preprocessing and checkpoint selection; the [Ki-67 figure guide](<../6 Ki67/4 Documentation/manuscript_confusion_matrices.md>) identifies the complementary pixel-input results.

The [classification analysis](<../6 Ki67/4 Documentation/classification_analysis.md>) derives Tables 6-7 from the same fitted models. It records class prevalence, per-seed precision, recall and F1, 10,000 image-bootstrap replicates and validation-fitted temperature scaling. These intervals describe fixed test cohorts under an independent-image assumption; they do not establish patient-level generalization.

## Additional Ki-67 evidence

All locations below are under `6 Ki67/5 Outputs/` after restoration.

| Directory | Contents |
| --- | --- |
| `shid-v2-partition/` | Patient allocation, image manifest and annotation counts |
| `shid-v2-20261031/` | Twelve joint three-class and 36 separate binary fits |
| `shid-separate-multiclass-20261101/` | Six anatomical three-class fits, two extractors and three seeds |
| `shid-cross-dataset-20261101/` | Forty-eight evaluations of source-trained checkpoints, four directions and four tasks |
| `ml-baselines/` | 336 classifier contexts, 1,008 selected models and their grid-search records |
| `phikon-joint-finetune/` | Six joint binary models, selected checkpoints, predictions, confusion matrices and timings |
| `phikon-shid-finetune/` | Three selected three-class anatomical models on SHIDC-B-Ki-67 |
| `phikon-feature-mapping/` | Original and ten permuted Phikon-v2 assignments, 33 selected models and paired bootstrap results |
| `phikon-xai/` | Two publication examples, six attribution maps and the final explanation figure |
| `paper-literature-comparison/` | Published comparison values and source identifiers |

The guides linked from [Ki-67 experiments](<../6 Ki67/README.md>) describe these records and the corresponding commands. The initial model catalogs remain the index for the earlier campaigns. Each additional campaign carries its own run and selection records.

## Interpretation of saved endpoints

- Iris regression selects by validation RMSE; auxiliary Ki-67 regression selects by validation MAE in original units. Both train with mean squared error on scaled targets.
- Iris test endpoints cover initialization, selected checkpoints and controls. Ki-67 stores test predictions every epoch. Test metrics do not select checkpoints.
- Repeated evaluations reuse the same observations. Seed summaries average metrics without ensembling predictions or adding independent samples.
- Iris graph seed 241 was chosen retrospectively for the descriptive rewired display. Joint Ki-67 extractor selection instead used combined validation balanced accuracy.
- Activity reset clears transient neuronal state while retaining gains. The saved transfer phase `reset` means target-domain training from initial gains, a different intervention.

## Expanded Iris controls and schematic

`5 Iris/5 Outputs/campaigns/iris-topology-ml/` contains 1,220 graph training runs and 280 selected traditional-classifier runs, their partitions, histories, predictions, selected hyperparameters and analysis results. The [guide](topology_controls.md) documents the graph-family comparison, paired bootstrap intervals and Tables 3 and 15.

`5 Iris/5 Outputs/paper-schematic/` contains Figure 1, its assets and numerical activity traces. The editable Draw.io file and export script are under `7 Paper Results/1 Generators/schematic/`.
