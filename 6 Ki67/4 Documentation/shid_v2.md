# Patient-disjoint SHIDC-B-Ki-67-V2 experiments

The V2 partition uses the same 2,356 prepared image/JSON pairs as the original SHIDC-B-Ki-67 experiment. Filename-derived patient groups are disjoint across training (1,325 images, 13 patients), validation (331 images, three patients), and test (700 images, seven patients). The split retains all annotations and the original counting conventions. It does not create a new external cohort.

The reproducible partition procedure first verifies the preceding benchmark's seed-64 image-level split, then selects the deterministic patient-disjoint partition. `split_manifest.csv`, `split_summary.json`, and `patient_summary.csv` are archived under `6 Ki67/5 Outputs/shid-v2-partition`. Images and annotations are not redistributed in these revision artifacts.

## Archived experiments

| Archive directory under `6 Ki67/5 Outputs` | Included evidence |
| --- | --- |
| `shid-v2-20261031/binary` | 36 separate V2 binary fits: two frozen extractors, two graphs, three thresholds and three seeds |
| `shid-v2-20261031/joint` | 12 BCData + V2 three-class fits: two frozen extractors, two graphs and three seeds |
| `shid-separate-multiclass-20261101` | Six published separate V2 three-class anatomical fits: two frozen extractors and three seeds |

Both extractors are DINOv2-base and Phikon-v2; the graph choices are anatomical and degree-preserving rewiring 241. Only internal gains are trained. Each fit uses 120 epochs, including an archived initial evaluation at epoch zero, and selects its single `best.pt` using validation balanced accuracy. Test predictions are preserved for every completed epoch. Seeds are 1729, 2718 and 3141. Joint training selects using pooled validation and reports pooled and dataset-specific test results.

The original separate multiclass campaign also ran six rewired fits. This release includes the six anatomical fits used in the published comparison; its original campaign plan and timing records describe the full 12-fit campaign. `publication_scope.json` records the included subset. Do not interpret that campaign's elapsed time as the time of only the six published fits.

## Rebuild numerical reports without training

After installing the repository and attaching the Zenodo payload as described in the main README, run from the repository root:

```bash
python -m ki67.summarize_shid_v2
python -m ki67.run_shid_multiclass --report --output "6 Ki67/6 Workspace/shid-separate-reports"
```

The first command reads the archived 48-fit campaign and writes `6 Ki67/6 Workspace/shid-v2-reports`. It recomputes initial and selected metrics from saved NPZ predictions, verifies identities, source hashes, all 121 history entries, validation-based selection, and patient separation. The second verifies and summarizes the six published anatomical multiclass fits. Both require a new output directory and leave archived inputs unchanged. Mean and sample standard deviation summarize separate seed predictions; seeds reuse the same test observations.

## Reproduce training or partition preparation

```bash
python -m ki67.run_shid_v2 --gpus 0,1 --output "6 Ki67/6 Workspace/shid-v2-reproduction"
python -m ki67.run_shid_multiclass --gpus 0,1 --output "6 Ki67/6 Workspace/shid-separate-reproduction"
python -m ki67.prepare_shid_v2 --plan-only
```

The training commands reuse the archived frozen feature matrices and training scalers by default. The separate command trains only the six published anatomical fits. GPU identifiers are explicit; the existing launcher reserves available GPUs with UUID locks. Output directories must be new. `run_shid_v2 --from-images` instead audits locally prepared V2 pairs and repeats frozen extraction; create the local V2 copy first using `prepare_shid_v2` without `--plan-only`. Preparation copies source image/JSON bytes unchanged into a separate destination and never edits the original pairs.

The global `6 Ki67/5 Outputs/cache/huggingface` contains the shared pinned extractor cache. Revision archives do not duplicate its weights. Original checkpoints, feature arrays, manifests and prediction files retain their bytes and recorded hashes. Historical repository-relative provenance is retained; new command defaults point to the archived outputs and new results go to Workspace.
