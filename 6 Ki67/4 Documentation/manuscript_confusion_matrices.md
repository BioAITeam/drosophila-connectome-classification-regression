# Ki-67 figures and tables

Use the [paper reproduction command](<../../7 Paper Results/README.md>) to regenerate Figures 8-13 and the Ki-67 tables from saved numerical evidence. Its `--family ki67` option limits the output to these items. The [figure catalog](<../../4 Trained Models/1 Catalogs/figure-provenance.json>) identifies the archived editions and their generators.

## Panel layout

| Figure | Content | Arrangement |
| --- | --- | --- |
| 8 | Joint three-class learning with frozen Phikon-v2 and rewired connectivity | Training and validation curves; selected checkpoints marked |
| 9 | Joint BRAP-Net three-class confusion matrices | One row of eight panels, A-H |
| 10 | Separately trained binary Phikon-v2 models | Three rows of eight panels: 5% (A-H), 10% (I-P), 30% (Q-X) |
| 11 | Separately trained anatomical native-RGB models | One row: BCData before/after (A/B), SHIDC-B-Ki-67 before/after (C/D) |

In Figures 9 and 10, the first four columns show BCData and the last four show SHIDC-B-Ki-67. Each dataset block contains anatomical before/after followed by rewired before/after. Arrows connect initialization to validation-selected checkpoints. Figure 9 uses the same fitted gains for both datasets within each graph/seed combination; Figure 10 uses separate gains for each dataset, threshold and graph.

## Counts and percentages

Rows are true classes and columns are predictions. Each cell gives the mean image count across three seeds, followed by the mean percentage within the true class. Counts can be fractional; seeds reuse the same 402 BCData and 700 SHIDC-B-Ki-67 test images. Predictions are not ensembled. All matrices share a 0-100% color scale.

Before training denotes epoch zero; after training denotes the validation-selected checkpoint, which may precede the final epoch.

| Binary boundary | BCData lower/upper class counts | SHIDC-B-Ki-67 lower/upper class counts |
| --- | ---: | ---: |
| PI ≤5 / >5 | 14 / 388 | 78 / 622 |
| PI ≤10 / >10 | 38 / 364 | 123 / 577 |
| PI <30 / ≥30 | 191 / 211 | 355 / 345 |

## Tables

The [paper reproduction guide](<../../7 Paper Results/README.md>) lists all table numbers, generators and input records. The [classification-analysis guide](classification_analysis.md) defines the image-bootstrap intervals and probability calibration.

## RGB timing records

Per-fit median epoch durations use the last trained record for each scheduled epoch, counting each epoch once. Final-attempt duration can cover part of a resumed run rather than the complete 120-epoch sequence. Neither measure is elapsed campaign time. The [computational-cost guide](<../../8 Documentation/full_graph_compute.md>) defines the corresponding Iris timing boundaries.

Figure 12 summarizes the joint fine-tuned anatomical and linear models at the 30% threshold. Figure 13 displays local explanations of the anatomical model. See [fine-tuning](phikon_finetuning.md) and [local explanations](phikon_xai.md) for their generators and archived inputs.
