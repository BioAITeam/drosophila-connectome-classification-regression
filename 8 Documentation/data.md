# Anatomical data and reproducible graph preparation

The models use three original **MaleCNS v1.0** exports from the [official Janelia catalog](https://male-cns.janelia.org/download/), recorded as accessed on 13 September 2026. The specimen is an adult male; these data are distinct from the female FlyWire and larval connectomes. Attribution is to FlyEM, HHMI Janelia, and collaborators: Berg et al., *Sexual dimorphism in the complete Drosophila male central nervous system connectome*, Cell (2026), [DOI](https://doi.org/10.1016/j.cell.2026.08.015). The source catalog declares CC-BY licensing for its data; this does not assign a license to this repository's code.

## Immutable original exports

The original Feather files are retained without transformation in `2 Data/1 Raw/malecns-v1.0/`, relative to the repository root. The [download guide](<../2 Data/1 Raw/README.md>) lists the required files and official links. `FLYBRAIN_RAW_ROOT` optionally selects another read-only raw-data location.

| Role | Original filename | Rows |
| --- | --- | ---: |
| Annotations | `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 211,577 |
| Neurotransmitters | `body-neurotransmitters-male-cns-v1.0.feather` | 1,835,518 |
| Segment connectivity | `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 151,856,684 |

The downloader verifies source size and MD5 against the official release and SHA-256 against an existing source manifest. Previously recorded SHA-256 values also permit local verification without network access.

`2 Data/2 Manifests/malecns-v1.0-sources.json` records the release, source URLs and file checksums. Synaptic point, partner, T-bar, and body-statistics files are not required to construct these neuron-level models.

## Full neuronal graph

Preparation selects annotation rows with non-null `superclass`, yielding **166,700 neurons**. Restricting to `status == 'Traced'` would retain only 165,122 rows and omit, among others, many annotated R1-R6 axons. The exact selection rule is recorded in the graph sidecar. Original IDs remain `uint64` in ascending order.

The connectivity table is streamed in Arrow batches. Edges are retained when both endpoints are in the selected universe and the contact count reaches `min_weight`, which defaults to one. The resulting full graph has **25,582,938 directed edges** and **124,177,617 contacts**. Original self-connections are retained; no nodes, edges, or learning routes are added. Isolated annotated neurons are retained in full mode.

The `weight` array stores original contact counts. Model normalization and trainable efficacy multipliers do not replace these counts. The `minconf-0.5` filename refers to source synaptic confidence, not the edge-weight threshold.

## Anatomical visual and mushroom-body subgraph

The `visual_mb` mode induces a subgraph using annotations and connectivity only, without Iris observations or labels:

1. Select R7p/R7y/R7d and R8p/R8y/R8d inputs, grouped as left R7, right R7, left R8, and right R8. `rootSide` supplies laterality, with `somaSide` as fallback; ambiguous visual types are excluded.
2. Retain nodes on any path of at most two edges from these inputs to a KCg-d Kenyon cell. The path limit is configurable; a different value defines a different graph.
3. Include all 206 KCg-d neurons, all dopaminergic neurons (DANs), and the three mushroom body output neuron (MBON) types receiving the most KCg-d contacts, resolving ties by type name. The selected types are MBON09, MBON27, and MBON05. The choice does not use task activity or labels.
4. Retain every original edge between the selected nodes, including recurrent connections. No layered network or artificial route is constructed.

The prepared subgraph has **1,323 neurons, 40,755 edges and 132,277 contacts**. Its neurons comprise 730 photoreceptors, 39 visual relays, 206 KCg-d neurons, eight MBONs and 340 DANs. The R7-left, R7-right, R8-left and R8-right input groups contain 201, 166, 193 and 170 neurons, respectively. Output groups contain four MBON09, two MBON27 and two MBON05 neurons. All retained edges are eligible for supervised gain adaptation; the separate local-plasticity protocol selects 837 KC-to-MBON edges. The sidecar's `connectivity_validation` records path lengths and modulatory contacts.

This is a visual and mushroom-body subgraph, not the full central nervous system. Removing surrounding neurons changes recurrent context and incoming-contact normalization. Full and input-matched graph comparisons are described in [graph controls](graph_controls.md).

## Fixed channels and functional sign assumptions

The four numeric channels stimulate populations defined by photoreceptor type and hemisphere. Source annotations do not supply a retinotopic assignment for these cells. Sidecars record `retinotopic=false` and `encoding_fixed=bilateral_r7_r8_intensity_channels`: the channels represent population stimulation, not a reconstructed retina.

For the `direct` diagnostic, existing KCs are ordered by side, type, and ID, then divided into four fixed groups in `direct_input_group`. This bypasses the visual interface without adding neurons or connections, so its results belong to a separate condition.

Signs are assigned from `consensus_nt`: acetylcholine +1; GABA, glutamate, and histamine -1; dopamine, serotonin, and octopamine zero for modeled fast transmission. Unknown or `unclear` entries use +1 as an explicit assumption. Neurons and edges remain present even when their fast sign is zero. DAN contacts participate separately in the [local modulation hypothesis](plasticity.md).

Every anatomically annotated DAN has fast sign zero, including two PPL203 neurons with original `unclear` consensus, IDs **13312 and 15693**. The raw consensus is unchanged; `dan_consensus_exceptions` records the override. The other 338 DAN already have monoaminergic consensus and zero fast sign.

These assignments are not measured postsynaptic effects. Receptor information and richer dynamics are absent. Glutamate inhibition, excitatory unknowns, and the task's PAM/PPL1 feedback valences are assumptions requiring separate biological validation.

## Prepared artifact contract

Each NPZ has a same-stem JSON sidecar recording raw-source and graph hashes, node order, filters, populations, sign rules, and connectivity checks. Open the NPZ with `np.load(path, allow_pickle=False)`.

| Key | Shape and dtype | Meaning |
| --- | --- | --- |
| `node_ids` | N, uint64 | Original MaleCNS IDs |
| `src`, `dst` | E, int64 | Source and destination indices in `node_ids` |
| `weight` | E, float32 | Original contact counts |
| `sign` | N, float32 | Fixed sign assumption: -1, 0, or +1 |
| `input_group` | N, int64 | -1 or visual channel 0 to 3 |
| `output_group` | N, int64 | -1 or MBON population 0 to 2 |
| `direct_input_group` | N, int64 | -1 or diagnostic KC channel 0 to 3 |
| `reward_mask`, `punishment_mask` | N, bool | Disjoint PAM and PPL1 DAN populations |
| `plastic_mask` | E, bool | Original KC-to-MBON edges |
| `kc_mask`, `mbon_mask` | N, bool | Anatomical classes for auditing |

## Preparation commands

Follow the [source download guide](<../2 Data/1 Raw/README.md>) and the [graph preparation commands](<../3 Connectome/README.md#prepare-a-new-anatomical-graph>). Preparation of matched-input and rewired variants is documented in [graph controls](graph_controls.md).

Inference and training can use the restored graphs in `2 Data/3 Processed/` directly. For a newly prepared graph, set the training configuration's graph path to its NPZ and retain the matching JSON sidecar.
