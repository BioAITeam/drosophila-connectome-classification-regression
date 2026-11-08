# Iris topology and classifier comparison

The released campaign contains 1,220 graph fits and 280 selected traditional classifiers. The graph fits cover one anatomical graph, 30 degree-preserving rewired graphs and 30 random-destination recurrent graphs. Each graph and each of 14 classifier families has 20 fits: two partition seeds, two optimization seeds and five outer folds. The classifier searches evaluated 4,740 additional configurations.

Source code is in `5 Iris/2 Execution/topology_controls`. Restore the Zenodo payload and run `python '4 Trained Models/artifacts.py' link` so that the campaign is available at `5 Iris/5 Outputs/campaigns/iris-topology-ml`. Its physical archive location is `zenodo/payload/outputs/campaigns/iris-topology-ml`.

## Reconstruct Tables 3 and 15

From the repository root:

```bash
python '5 Iris/2 Execution/topology_controls/paper_tables.py' \
  --archive '5 Iris/5 Outputs/campaigns/iris-topology-ml' \
  --output 'Workspace/iris-tables'
```

This writes `Table-3.tex`, `Table-3.csv`, `Table-15.tex`, `Table-15.csv` and `table-provenance.json`. The generator checks the result and prediction hashes, assembles all 56 complete out-of-fold classifier evaluations, and derives performance, selected hyperparameter frequencies, model sizes and recorded computation times. It requires NumPy and fits no models. In Table 3, KNN's 12/20 means that the listed hyperparameter combination was selected in 12 of its 20 validation searches.

To reconstruct topology summaries, all 300 out-of-fold evaluations and the 10,000-replicate conditional bootstrap:

```bash
python '5 Iris/2 Execution/topology_controls/analyze_results.py' \
  --output 'Workspace/iris-topology-analysis'
```

The graph-family CSVs include the 30 rewired and 30 random controls. Means use all 150 held-out flower predictions per evaluation. Anatomical dispersion describes four evaluations of one graph; control-family dispersion describes 30 graph means after averaging four evaluations per graph. The bootstrap resamples graph identities and class-stratified groups of identical flower measurements. Its intervals are conditional on the available flowers and fitted models.

## Inspect selected models

The exact environment used for fitting is recorded in the campaign's `training_provenance.json`. Python 3.12 with the direct dependencies in `5 Iris/2 Execution/topology_controls/requirements.txt` supports archived-model inference. Install these dependencies in a dedicated environment; the older Ki-67 classifier campaign uses a different XGBoost version.

```bash
python -m pip install -r '5 Iris/2 Execution/topology_controls/requirements.txt'
python '5 Iris/2 Execution/topology_controls/verify_artifacts.py' --inference
```

Verification checks all 1,500 completion records, checkpoint and prediction hashes, all 61 graphs and the ten shared data partitions. The inference option repeats predictions for one anatomical, one rewired and one random graph plus one selected model from each of the 14 classifier families. The `IRIS_CAMPAIGN_ROOT` environment variable selects another restored campaign directory.

## Repeat training

Use a new output directory to preserve the released evidence:

```bash
export IRIS_CAMPAIGN_ROOT="$PWD/Workspace/iris-topology-new"
python '5 Iris/2 Execution/topology_controls/prepare_partitions.py'
python '5 Iris/2 Execution/topology_controls/build_graph_controls.py'
python '5 Iris/2 Execution/topology_controls/run_campaign.py' --prepare
python '5 Iris/2 Execution/topology_controls/run_campaign.py' --workers 24
```

The graph builder uses `2 Data/3 Processed/visual_mb.npz` and its sidecar, which are supplied in the shared processed-data payload. Graph training uses 120 epochs, 24 recurrent steps, Adam at learning rate 0.03, batches of 16 and validation balanced accuracy for checkpoint selection, including epoch zero. Partitions keep identical feature vectors together and use training-fitted scaling. Every classifier search uses the same training and validation partitions, resolves ties by grid order and fits its selected estimator on the training partition only.

## Artifact layout and provenance

- `graphs/`: all 61 graph arrays, generation settings and structural audits.
- `data/`: ten grouped partitions with raw Iris observations, scaled inputs and row indices.
- `runs/<job-id>/`: selected model, held-out predictions, settings, results and completion hashes. Graph runs include all epoch histories; classifier runs include grid scores and selection records.
- `analysis/`: archived out-of-fold predictions, graph and classifier summaries, timing tables and bootstrap replicates.
- `paper/`: reconstructed Tables 3 and 15, CSV values and evidence hashes.

Artifact paths in campaign metadata are relative to the campaign root. Source-code and processed-source references are relative to the repository root. Training source hashes identify the code used for the original fit; `training_provenance.json` retains those original digests. Release normalization changes path strings and dependent serialized-file checksums while preserving numerical arrays, model tensors and statistical results. The publication manifest identifies the released files. Runtime logs, process state, locks and implementation-test fixtures are excluded.
