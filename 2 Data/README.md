# 2. Data

Choose the data required by your workflow. All paths start at the repository root.

| Workflow | Required data |
| --- | --- |
| Reconstruct paper figures and tables | [Zenodo artifacts](<../4 Trained Models/README.md>) |
| Predict Iris measurements or species | Zenodo artifacts and the supplied example CSVs or your own measurements |
| Predict Ki-67 classes | Zenodo artifacts and your image files; cell annotations are not required |
| Retrain Iris | Iris from scikit-learn and restored graphs from Zenodo |
| Retrain joint Ki-67 from archived ViT features | Zenodo artifacts; see [joint training](<../6 Ki67/4 Documentation/joint_training.md>) |
| Rebuild connectome graphs | Original MaleCNS exports; see the [raw-data guide](<1 Raw/README.md>) |
| Retrain Ki-67 from images or evaluate original images | Prepared BCData and SHIDC-B-Ki-67 image–annotation pairs |

## Locations

| Folder | Contents |
| --- | --- |
| `2 Data/1 Raw/` | MaleCNS source exports and prepared Ki-67 image–annotation datasets |
| `2 Data/2 Manifests/` | Source URLs, checksums and dataset specifications |
| `2 Data/3 Processed/` | Restored prepared graphs and provenance sidecars, read-only |

## Obtain the datasets

Follow the [raw-data guide](<1 Raw/README.md>) for download sources, Ki-67 preparation and exact folder placement. The MaleCNS [source manifest](<2 Manifests/malecns-v1.0-sources.json>) records URLs, sizes and SHA-256 checksums.

Iris is loaded through scikit-learn. Five [dataset specifications](<2 Manifests/datasets/>) define species classification and prediction of each floral measurement from the other three.

The [Ki-67 data contract](<../6 Ki67/4 Documentation/data_format.md>) defines annotation fields and label thresholds. Retain the study's recorded split membership for reproduction.
