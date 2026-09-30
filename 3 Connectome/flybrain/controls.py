"""Untrained perturbations for separating learning, inputs and graph topology."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


def ablate_inputs(x: np.ndarray, mode: str = "zero", seed: int = 0) -> np.ndarray:
    """Zero input, permute rows jointly, or shuffle each feature independently.

    Both permutations preserve feature marginals. Labels are never inspected.
    A row permutation deliberately breaks alignment with the original targets.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Expected a feature matrix.")
    rng = np.random.default_rng(seed)
    if mode == "zero":
        return np.zeros_like(x)
    if mode == "row_permutation":
        return x[rng.permutation(len(x))].copy()
    if mode == "feature_permutation":
        return np.column_stack([rng.permutation(x[:, i]) for i in range(x.shape[1])])
    raise ValueError("mode must be zero, row_permutation or feature_permutation.")


def shuffle_feedback(feedback: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute a precomputed feedback sequence; retain reward/punishment counts."""
    feedback = np.asarray(feedback)
    return feedback[np.random.default_rng(seed).permutation(len(feedback))].copy()


def rewire_graph(
    graph: dict[str, Any], seed: int = 0, swaps_per_edge: float = 1.0,
    max_attempts: int = 1_000_000, max_swaps: int = 100_000,
    forbid_self_loops: bool = True
) -> dict[str, Any]:
    """Directed double-edge swaps preserve exact in/out degree, not anatomy.

    Replace (a,b),(c,d) with (a,d),(c,b), leaving src and per-edge weights
    unchanged. Source neuron signs therefore remain unchanged. Existing loops
    can remain; no new loops or duplicate directed edges are introduced.

    Bounded work means this is a partial degree-preserving perturbation, not a
    claim of uniformly random mixing. For a 25M-edge graph membership storage
    can be substantial on CPU. Apply to a selected subgraph when appropriate.
    The original graph is never modified. A plastic_mask travels with its
    original edge index and is not reclassified as biological after rewiring;
    a topology comparison must use all-edge plasticity or rebuild that mask.
    """
    src = np.asarray(graph["src"])
    dst = np.asarray(graph["dst"])
    if src.ndim != 1 or dst.shape != src.shape:
        raise ValueError("src/dst must be matching one-dimensional arrays.")
    if not np.issubdtype(src.dtype, np.integer) or not np.issubdtype(dst.dtype, np.integer):
        raise ValueError("src/dst must contain integer neuron indices.")
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("Negative neuron indices are invalid.")
    if not np.isfinite(swaps_per_edge) or swaps_per_edge < 0 or max_swaps < 0 or max_attempts < 0:
        raise ValueError("Swap budgets must be finite and nonnegative.")
    n_nodes = len(graph["node_ids"]) if "node_ids" in graph else int(graph.get("n_nodes", max(src.max(initial=0), dst.max(initial=0)) + 1))
    if len(src) and max(src.max(), dst.max()) >= n_nodes:
        raise ValueError("Edge endpoints exceed the declared neuron count.")
    result = {key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value) for key, value in graph.items()}
    result["src"], result["dst"] = src.copy(), dst.copy()
    rewired = result["dst"]
    edge_keys = {int(a) * n_nodes + int(b) for a, b in zip(src, dst)}
    if len(edge_keys) != len(src):
        raise ValueError("Aggregate duplicate directed edges before rewiring.")
    target = min(int(np.ceil(len(src) * swaps_per_edge)), int(max_swaps))
    rng = np.random.default_rng(seed)
    accepted = attempts = 0
    while accepted < target and attempts < max_attempts and len(src) >= 2:
        attempts += 1
        i, j = rng.integers(0, len(src), size=2)
        if i == j:
            continue
        a, c, b, d = int(src[i]), int(src[j]), int(rewired[i]), int(rewired[j])
        if a == c or b == d or (forbid_self_loops and (a == d or c == b)):
            continue
        first, second = a * n_nodes + d, c * n_nodes + b
        if first in edge_keys or second in edge_keys:
            continue
        edge_keys.remove(a * n_nodes + b)
        edge_keys.remove(c * n_nodes + d)
        edge_keys.add(first)
        edge_keys.add(second)
        rewired[i], rewired[j] = d, b
        accepted += 1
    result["rewire_metadata"] = {
        "topology": "degree_preserving_rewired_control_not_original_malecns", "seed": int(seed),
        "accepted_swaps": accepted, "attempts": attempts, "requested_swaps": target,
        "swaps_per_edge": float(swaps_per_edge), "max_swaps": int(max_swaps),
        "max_attempts": int(max_attempts), "budget_reached": accepted == target,
        "n_changed_edge_destinations": int(np.count_nonzero(rewired != dst)),
        "preserves": ["node_ids", "in_degree", "out_degree", "source_indexed_weight", "source_neuron_sign"],
        "does_not_preserve": ["anatomical_partners", "in_strength", "paths", "cell_type_edge_classes"],
        "plastic_mask_policy": "Original edge-index mask retained; use all-edge training or explicitly rebuild mask.",
        "mixing": "Bounded partial perturbation; uniform random graph distribution is not asserted."}
    return result
