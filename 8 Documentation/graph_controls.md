# Anatomical interfaces and restricted topology controls

`flybrain.graph_controls` prepares graph controls without labels, features,
model predictions or learned parameters. Each graph's JSON sidecar records
the source and derived graph hashes, preservation checks and realized
fraction of changed edge destinations.

## Matching full-network stimulation

`match_input_population(full_graph, reference_graph)` retains every full graph
node, directed edge, contact count, sign and output group. It maps the reference
input population by exact body ID and channel, assigning `-1` to other full
graph input neurons. It rejects missing/duplicate IDs, inconsistent channel
annotations and any difference in output body IDs or output groups.

For MaleCNS, `full_matched_inputs.npz` therefore retains 166,700 neurons and
25,582,938 edges while stimulating the exact 730 reference R7/R8 cells across
four fixed channels, rather than 1,783. The eight output MBON neurons retain their
IDs and groups. Direct KC encoding is not changed: this control is intended
for `encoding="visual"` comparisons. Full context and incoming normalization
still differ from the subgraph, so matching stimulation does not isolate
neuron count from every other network property.

## Restricted swaps

`restricted_rewire_graph(graph, type_codes, seed=137)` exchanges destinations
of two directed edges only when these edge strata match exactly:

1. Integer anatomical contact count.
2. Presynaptic sign hypothesis.
3. Original annotated source cell type.
4. Original annotated target cell type.

Swapping `(a,b),(c,d)` to `(a,d),(c,b)` then preserves each neuron's exact
in/out degree and incoming/outgoing contact strength. It preserves per-target
incoming source-sign counts and strength, incoming source-type profiles, and
per-source outgoing target-type profiles. Source IDs and contact counts remain
at their original edge indices. Preparation recomputes these quantities
from the final graph. It uses node/edge arrays and sparse profiles, without
constructing a dense neuron-by-neuron matrix.

Candidates introducing duplicate edges or new self loops are rejected. Work
is bounded to one accepted swap per original edge, at most 100,000 swaps and
1,000,000 attempts. Some strata cannot be changed. Reports expose eligible
strata, attempts, accepted swaps and final changed-edge fraction; this is a
partial constrained perturbation, not a uniformly random graph or proof that
anatomical organization has been removed. Repeated swaps can restore an edge,
so accepted swaps and changed destinations are different quantities.

Type codes come from immutable `bodyId,type` annotation columns. Missing
types form a separate unknown stratum. No class label participates in graph preparation.

## Degree-only comparison

The same preparation caches existing `controls.rewire_graph` results for
seeds 137, 241 and 997. These preserve degree, source-indexed counts and
source signs, but can change incoming strength and typed/signed profiles.
Their sidecars record the same preservation checks, including constraints
that are not preserved.

Both control types contain artificial partners and must use
`plastic_scope="all"`. The saved original `plastic_mask` is retained only
as an edge-index artifact; it does not classify newly swapped edges as
biological KC-to-MBON connections. Cached control graphs must not receive
an additional `rewire_seed` unless a second perturbation is explicitly intended.

## API and artifacts

First prepare `visual_mb.npz` and `full.npz`, with their JSON sidecars, in `5 Iris/6 Workspace/graphs/` using the [preparation commands](data.md). Then run the following API from an installed environment and the repository root:

```python
from flybrain.graph_controls import prepare_round4_graphs
from flybrain.paths import RAW

manifest = prepare_round4_graphs(
    RAW, "5 Iris/6 Workspace/graphs", seeds=(137, 241, 997)
)
```

This produces `full_matched_inputs.npz`,
`visual_mb_restricted_s{137,241,997}.npz`,
`visual_mb_degree_s{137,241,997}.npz`, their JSON sidecars and
`round4-graph-controls.json`. Training code can load each NPZ with the normal
graph loader. The source graph and raw annotation hashes remain in provenance.
The original anatomical connectivity summary is labelled as a parent summary
for rewired graphs. It describes the source graph rather than measured rewired reachability.

`RAW` is the source-data location, optionally set with `FLYBRAIN_RAW_ROOT`.

For training from these graphs, pass `--graph-root "5 Iris/6 Workspace/graphs"` when [generating campaign plans](benchmark_matrix.md).

## Prepared MaleCNS controls

The three restricted controls each accepted 40,755 swaps and passed every
independent preservation check. Their changed destination fractions are:

| Rewiring seed | Restricted control | Degree-only control |
| --- | ---: | ---: |
| 137 | 71.910% | 84.579% |
| 241 | 71.643% | 84.480% |
| 997 | 72.001% | 84.733% |

These are realized graph perturbations, not predictive results. All original
full-network arrays except `input_group` were exactly equal in the matched
input derivative. Detailed preservation checks, constraints and source hashes
are stored in each graph's JSON sidecar.
