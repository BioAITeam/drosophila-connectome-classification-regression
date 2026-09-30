# Raw datasets

Place original MaleCNS exports and prepared Ki-67 datasets here. Paths below start at the repository root.

## 1. Directory layout

```text
2 Data/1 Raw/
├── README.md
├── malecns-v1.0/
│   ├── body-annotations-male-cns-v1.0-minconf-0.5.feather
│   ├── body-neurotransmitters-male-cns-v1.0.feather
│   └── connectome-weights-male-cns-v1.0-minconf-0.5.feather
└── ki67/
    ├── bcdata/
    │   ├── train/
    │   ├── validation/
    │   └── test/
    └── shid/
        ├── Train/
        ├── Validation/
        └── Test/
```

Keep the folder names and capitalization shown above. Each Ki-67 split contains JPEG images and their matching JSON files in the same directory, for example `42.jpg` and `42.json`. The `shid` folder contains SHIDC-B-Ki-67. Iris is supplied by scikit-learn and requires no raw-data folder here.

## 2. MaleCNS v1.0

The [official MaleCNS catalog](https://male-cns.janelia.org/download/) provides the three exports used by the study. The repository downloader places them in `2 Data/1 Raw/malecns-v1.0/`. After [installing the environment](<../../1 Environment/README.md>), run from the repository root:

```bash
./run.sh --device cpu download
```

Alternatively, download the same files manually, preserving these names:

| Export | Direct download | Size |
| --- | --- | ---: |
| Neuron annotations | [body-annotations-male-cns-v1.0-minconf-0.5.feather](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather) | 14.48 MB |
| Neurotransmitter predictions | [body-neurotransmitters-male-cns-v1.0.feather](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather) | 43.28 MB |
| Segment connectivity | [connectome-weights-male-cns-v1.0-minconf-0.5.feather](https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather) | 1,051.24 MB |

The [source manifest](<../2 Manifests/malecns-v1.0-sources.json>) records their SHA-256 hashes. Running the downloader again verifies existing files without replacing them. The exports total approximately **1.11 GB**. The annotation and connectivity files use a minimum synapse-detection confidence score of 0.5. Retain the CC BY 4.0 source attribution described in [data provenance](<../../8 Documentation/data.md>).

## 3. BCData and SHIDC-B-Ki-67

1. Obtain the full datasets from [BCData](https://sites.google.com/view/bcdataset) and the [SHIDC-B-Ki-67 access instructions](https://github.com/SHIDCenter/PathoNet#datasets). Follow each source's access terms; if its download service is unavailable, request access from the dataset authors.
2. Use the [BioAITeam preprocessing workflow](https://github.com/BioAITeam/ki67-open-and-closed-vlm-benchmarking#2-data-preprocessing) to prepare the full datasets. Its example sample folders are not the full study data. BCData requires PNG-to-JPEG and HDF5-to-JSON conversion. SHIDC-B-Ki-67 requires removal of annotation label `3` and the seed-64 training/validation split.
3. Copy the prepared split folders to the destinations below. Paths in the first column are relative to the benchmark repository; destinations are relative to this repository.

| Prepared directory in the benchmark repository | Destination in this repository |
| --- | --- |
| `1.data_access/data_full_BCData/3.data_processed/` | `2 Data/1 Raw/ki67/bcdata/` |
| `1.data_access/data_full_SHIDC-B-Ki-67/2.data_processed/` | `2 Data/1 Raw/ki67/shid/` |

The split folders must sit directly under their destination, with no additional `data_processed/` level. Expected image–JSON pair counts are:

| Dataset | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| BCData | 803 | 133 | 402 |
| SHIDC-B-Ki-67 | 1,325 | 331 | 700 |

The loader reads these prepared pairs and does not perform the upstream conversions. The [data-format guide](<../../6 Ki67/4 Documentation/data_format.md>) defines the JSON schema and targets.

## 4. Reproduction and verification

Original images, annotations and Feather exports are distributed by their source projects. Zenodo supplies models, derived graphs, frozen extractors and saved numerical results. [Restore those artifacts](<../../4 Trained Models/README.md>) to reconstruct paper figures without downloading raw datasets.

For image-based reproduction, retain the filenames and splits in the restored `6 Ki67/5 Outputs/manifests/bcdata.json` and `6 Ki67/5 Outputs/manifests/shid.json`. These manifests contain image, pixel and annotation hashes; matching counts alone does not establish identical inputs. Run the [dataset audit](<../../6 Ki67/4 Documentation/protocol.md>) before training.
