# 4. Trained models

The Zenodo archive supplies trained checkpoints, their graphs and scalers, frozen ViT extractors, and saved predictions. Use the companion `zenodo.zip` archive supplied with the repository release.

## 1. Restore and verify

After [installing the environment](<../1 Environment/README.md>), place `zenodo.zip` in the repository root. Extract it with a tool that preserves symbolic links, such as Linux `unzip`, then create the artifact links:

```bash
unzip zenodo.zip
python -B "4 Trained Models/artifacts.py" link
python -B "4 Trained Models/artifacts.py" verify
python -B "4 Trained Models/artifacts.py" verify-release
```

The archive contains `zenodo/`, so the resulting payload must be `zenodo/payload/`, not `zenodo/zenodo/payload/`. Allow approximately 33 GB for extracted artifacts in addition to the downloaded archive.

`link` creates the relative directory links below, reuses matching links and refuses to replace existing files or directories:

| Repository location | Archived location |
| --- | --- |
| `2 Data/3 Processed/` | `zenodo/payload/data/processed/` |
| `5 Iris/5 Outputs/` | `zenodo/payload/outputs/` |
| `6 Ki67/5 Outputs/` | `zenodo/payload/ki67/outputs/` |

`verify` checks cataloged Ki-67 resources and figure hashes, Iris checkpoint sizes and graph/dependency hashes, and one Iris checkpoint hash per campaign. Add `--all-iris` to hash all 6,390 Iris checkpoints. Successful verification ends with `"status": "passed"`. `verify-release` checks the size and SHA-256 of every file in the publication manifest, including all additional campaigns.

## 2. Choose a workflow

Use the [Iris inference guide](<../5 Iris/4 Inference/README.md>) for one flower or a CSV, the [BRAP-Net guide](<../6 Ki67/3 Inference/README.md>) for images, or [7 Paper Results](<../7 Paper Results/README.md>) to reconstruct the article's figures and tables. The scripts locate each checkpoint and its dependencies automatically.

## 3. Model catalogs

The catalogs in `1 Catalogs/` identify the saved models and their dependencies. Paths are relative to the repository root.

| Catalog | Contents |
| --- | --- |
| `model-index.csv` | 6,390 Iris fits, including graph, seeds, fold, selected epoch, weight hash and companion paths |
| `graph-index.csv` | Processed graphs and provenance, with sizes and SHA-256 hashes |
| `ki67-model-index.csv` | 264 completed Ki-67 fits with model settings and checkpoint paths |
| `ki67-model-index.json` | Ki-67 model records and shared hashes for graphs, scalers, frozen ViTs and initial/selected predictions |
| `figure-provenance.json` | The thirteen manuscript figure PDFs, their generators and archived provenance |

The Ki-67 catalog identifies 120 main-paper classifiers: 12 joint three-class models, 72 separate binary models and 36 separate three-class models. The remaining entries describe auxiliary fits.

The [paper evidence guide](<../8 Documentation/manuscript_evidence.md>) lists the additional campaigns. Their selected checkpoints and preprocessing are indexed within each campaign rather than in the initial catalogs above.

## 4. Checkpoint dependencies

Keep the restored payload together. Iris checkpoints use companion metadata and shared tensors under `zenodo/payload/shared/iris/`. Ki-67 models require their graph and configuration; ViT-based models also require training scalers and pinned extractor snapshots under `6 Ki67/5 Outputs/cache/huggingface/`. Preserve each snapshot's links and blob files. The inference scripts load these dependencies automatically.

## 5. License

Project-authored checkpoint contents use the repository [LICENSE](../LICENSE). Frozen ViT weights and other third-party content retain their original terms. Preserve the licenses and notices included in the payload.

## Additional campaign records

The original model catalogs cover the initial campaigns. Additional experiments retain their selected checkpoints and validation-selection records in their own restored directories. See the [evidence map](<../8 Documentation/manuscript_evidence.md>) for the full inventory. The publication manifest verifies all additional files as well as the original payload.

The [figure catalog](<1 Catalogs/figure-provenance.json>) uses the final Figure 1-13 numbering, including the editable schematic, fine-tuning confusion matrices and local explanations.
