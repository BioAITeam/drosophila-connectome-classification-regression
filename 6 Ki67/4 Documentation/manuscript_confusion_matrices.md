# Ki-67 figures and tables

Use the [paper reproduction command](<../../7 Paper Results/README.md>) to regenerate Figures 7–10 and Tables 3–5 from saved numerical evidence. Its `--family ki67` option limits the output to these items. The [figure catalog](<../../4 Trained Models/1 Catalogs/figure-provenance.json>) identifies the archived editions and their generators.

## Panel layout

| Figure | Content | Arrangement |
| --- | --- | --- |
| 7 | Joint three-class learning with frozen Phikon-v2 and rewired connectivity | Training and validation curves; selected checkpoints marked |
| 8 | Joint BRAP-Net three-class confusion matrices | One row of eight panels, A–H |
| 9 | Separately trained binary Phikon-v2 models | Three rows of eight panels: 5% (A–H), 10% (I–P), 30% (Q–X) |
| 10 | Separately trained anatomical native-RGB models | One row: BCData before/after (A/B), SHIDC-B-Ki-67 before/after (C/D) |

In Figures 8 and 9, the first four columns show BCData and the last four show SHIDC-B-Ki-67. Each dataset block contains anatomical before/after followed by rewired before/after. Arrows connect initialization to validation-selected checkpoints. Figure 8 uses the same fitted gains for both datasets within each graph/seed combination; Figure 9 uses separate gains for each dataset, threshold and graph.

## Counts and percentages

Rows are true classes and columns are predictions. Each cell gives the mean image count across three seeds, followed by the mean percentage within the true class. Counts can be fractional; seeds reuse the same 402 BCData and 700 SHIDC-B-Ki-67 test images. Predictions are not ensembled. All matrices share a 0–100% color scale.

Before training denotes epoch zero; after training denotes the validation-selected checkpoint, which may precede the final epoch.

| Binary boundary | BCData lower/upper class counts | SHIDC-B-Ki-67 lower/upper class counts |
| --- | ---: | ---: |
| PI ≤5 / >5 | 14 / 388 | 78 / 622 |
| PI ≤10 / >10 | 38 / 364 | 123 / 577 |
| PI <30 / ≥30 | 191 / 211 | 355 / 345 |

## Tables

- **Table 3:** four joint configurations evaluated on combined test data and each dataset subset. Phikon-v2 rows correspond to BRAP-Net; DINOv2 rows are comparisons.
- **Table 4:** separately fitted binary models. Each threshold reports initial balanced accuracy, selected mean ± sample standard deviation, and paired change in percentage points.
- **Table 5:** separate-dataset three-class comparison of RGB, DINOv2 and Phikon-v2. It uses 12 RGB fits and 24 frozen-feature fits, all with 120-epoch budgets. Figure 10 shows the anatomical RGB subset. Those selected models make no Intermediate predictions in either dataset; two BCData rewired RGB runs select epoch zero.

## RGB timing records

Per-fit median epoch durations use the last trained record for each scheduled epoch, counting each epoch once. Final-attempt duration can cover part of a resumed run rather than the complete 120-epoch sequence. Neither measure is elapsed campaign time. The [computational-cost guide](<../../8 Documentation/full_graph_compute.md>) defines the corresponding Iris timing boundaries.
