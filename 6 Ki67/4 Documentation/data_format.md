# Ki-67 image and annotation format

The experiment loader reads prepared RGB JPEG images and matching JSON annotations. Follow the [raw-data guide](<../../2 Data/1 Raw/README.md>) for download sources, conversion instructions and the required split folders. The default roots are `2 Data/1 Raw/ki67/bcdata/` and `2 Data/1 Raw/ki67/shid/`; `KI67_BCDATA_ROOT` and `KI67_SHID_ROOT` can override them. This loader does not download or convert the original releases.

Each split is a flat directory containing both files of every pair. Names must have the same stem, for example `42.jpg` and `42.json`. Image extensions `.jpg` and `.jpeg` are accepted case-insensitively; images must already be RGB. PNG and HDF5 files are not read by this loader. Duplicate stems or missing partners cause the audit to fail. `Train_full` is not scanned.

BCData stems in the prepared study data are sample identifiers. SHIDC-B-Ki-67 names use the form `p<patient>_<image>_<patch>`, such as the illustrative pair `p1_0001_1.jpg` and `p1_0001_1.json`. Existing names and partition membership must be retained when reproducing archived results. Sample IDs combine dataset, split and stem, so the same stem in different splits remains distinguishable.

## JSON records and targets

The JSON document is a list of cell records. Each record requires integer `x`, `y` and `label_id` fields; booleans and noninteger values are rejected. Coordinates identify the column and row in image pixels. Label `1` denotes a Ki-67-positive cell and label `2` a Ki-67-negative cell. The following example is synthetic:

```json
[
  {"x": 120, "y": 85, "label_id": 1},
  {"x": 340, "y": 210, "label_id": 2}
]
```

This example contains one positive and one negative cell, giving `PI = 100 × 1 / (1 + 1) = 50%`. The loader calculates targets from record counts rather than reading a precomputed image-level label. Repeated records and coordinates outside image bounds retain their source counts and are flagged in the derived inventory. An empty list `[]` produces zero positive cells, zero negative cells and PI=0, with an explicit empty-annotation flag.

| Task | Class 0 | Class 1 | Class 2 |
| --- | --- | --- | --- |
| Multiclass | PI <16 | 16 ≤ PI <30 | PI ≥30 |
| Binary 5 | PI ≤5 | PI >5 | — |
| Binary 10 | PI ≤10 | PI >10 | — |
| Binary 30 | PI <30 | PI ≥30 | — |

All PI thresholds are percentages. Comparisons use integer counts without first rounding PI. Regression targets are PI in percentage points and the two cell counts. JSON annotations supply training and evaluation targets; prediction on a new image does not require them.

## Relationship to the upstream datasets

The study used data prepared by the [BioAITeam Ki-67 benchmark project](https://github.com/BioAITeam/ki67-open-and-closed-vlm-benchmarking), with these transformations:

- **BCData:** original PNG images were converted to JPEG, and positive/negative coordinate records from HDF5 annotations were combined into the JSON schema above. The prepared images are 640 × 640 RGB.
- **SHIDC-B-Ki-67:** the full-size JPEG images are 1,228 × 1,228 RGB. Upstream JSON records labeled `3`, corresponding to tumor-infiltrating lymphocytes, were excluded before these experiments. The present loader accepts labels `1` and `2` and rejects other labels. The preceding project also supplies the prepared Train/Validation membership used here.

For exact input reproduction, compare prepared files and split membership with `6 Ki67/5 Outputs/manifests/bcdata.json` and `6 Ki67/5 Outputs/manifests/shid.json` after artifact restoration. Their file hashes identify the evaluated inputs; a new conversion can change image bytes or split membership.

Run the [data audit](protocol.md) before training to verify image–annotation pairs.
