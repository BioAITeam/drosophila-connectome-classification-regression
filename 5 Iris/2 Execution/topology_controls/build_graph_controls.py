"""Construct independent, label-free Iris topology controls in this campaign only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

from common import HERE, REPO, load_json, relative, save_json, sha256

sys.path.insert(0, str(REPO / "3 Connectome"))
from flybrain.controls import rewire_graph


SEEDS = [137, 241, 997] + list(range(1001, 1028))
SOURCE = REPO / "2 Data/3 Processed/visual_mb.npz"
HORIZON = 24


def read_graph(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def random_destinations(graph: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Draw each source's nonself targets uniformly without replacement.

    An independent SeedSequence stream separates this family from rewiring.
    The original self-loop edge indices, sources, contacts and signs stay fixed.
    """
    result = {key: value.copy() for key, value in graph.items()}
    src, dst = graph["src"], graph["dst"]
    n = len(graph["node_ids"])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 90217]))
    nonself = np.flatnonzero(src != dst)
    order = nonself[np.argsort(src[nonself], kind="stable")]
    counts = np.bincount(src[nonself], minlength=n)
    boundaries = np.r_[0, np.cumsum(counts)]
    for source in np.flatnonzero(counts):
        indices = order[boundaries[source]:boundaries[source + 1]]
        # Uniform ordered sample from all n-1 nonself nodes. Mapping around
        # the source avoids allocating or sorting a candidate list per edge.
        chosen = rng.choice(n - 1, size=len(indices), replace=False)
        result["dst"][indices] = chosen + (chosen >= source)
    return result


def _shortest_hops(graph: dict[str, np.ndarray], active: np.ndarray) -> np.ndarray:
    n = len(graph["node_ids"])
    src, dst = graph["src"][active], graph["dst"][active]
    distances = np.full((4, n), -1, dtype=np.int64)
    for channel in range(4):
        distances[channel, graph["input_group"] == channel] = 0
        for hop in range(1, HORIZON + 1):
            reached = dst[distances[channel, src] == hop - 1]
            unseen = reached[distances[channel, reached] == -1]
            if not len(unseen):
                break
            distances[channel, unseen] = hop
    return distances


def _population_hops(graph: dict[str, np.ndarray], distances: np.ndarray) -> list[list[int | None]]:
    return [[int(reachable.min()) if len(reachable := distances[channel, graph["output_group"] == group][
        distances[channel, graph["output_group"] == group] >= 0]) else None
        for group in range(3)] for channel in range(4)]


def reachability_audit(graph: dict[str, np.ndarray]) -> dict:
    """Inspect positive-contact paths and paths with nonzero model base weights.

    Maximum absolute path products avoid sign cancellation in this structural
    audit; they are not an estimated response of the nonlinear trained model.
    """
    src, dst, contacts = (graph[key] for key in ("src", "dst", "weight"))
    n = len(graph["node_ids"])
    incoming = np.bincount(dst, weights=contacts, minlength=n)
    base = contacts.astype(np.float64) / incoming[dst] * graph["sign"][src]
    structural = _shortest_hops(graph, contacts > 0)
    functional = _shortest_hops(graph, (contacts > 0) & (base != 0))
    best = np.zeros((4, n), dtype=np.float64)
    for channel in range(4):
        best[channel, graph["input_group"] == channel] = 1.0
    current = best.copy()
    for _ in range(HORIZON):
        following = np.zeros_like(current)
        for channel in range(4):
            np.maximum.at(following[channel], dst, current[channel, src] * np.abs(base))
        np.maximum(best, following, out=best)
        current = following
    functional_hops = _population_hops(graph, functional)
    return {
        "horizon_recurrent_edges": HORIZON,
        "positive_contact_shortest_hops": _population_hops(graph, structural),
        "nonzero_signed_weight_shortest_hops": functional_hops,
        "all_input_output_group_pairs_reachable_within_24_edges": all(
            hop is not None for row in functional_hops for hop in row),
        "all_input_output_group_pairs_reachable_in_24_simulation_steps": all(
            hop is not None and hop <= HORIZON - 1 for row in functional_hops for hop in row),
        "output_neuron_fraction_reachable_within_24_edges": [[float(np.mean(
            functional[channel, graph["output_group"] == group] >= 0)) for group in range(3)]
            for channel in range(4)],
        "maximum_absolute_base_weight_path_product_within_24_edges": [[float(
            best[channel, graph["output_group"] == group].max()) for group in range(3)]
            for channel in range(4)],
        "nonzero_signed_edges": int(np.count_nonzero(base)),
        "zero_signed_edges": int(np.count_nonzero(base == 0)),
        "base_weight_definition": "contact / destination_total_incoming_contacts * source_sign; initial efficacy=1",
        "simulation_horizon_note": "Input enters at simulation step 1; an L-edge recurrent path first affects an output at step L+1.",
        "interpretation": "Population reachability requires at least one output neuron. Absolute path products test nonzero transmission support, not signed cancellation or nonlinear response magnitude.",
        "used_to_reject_or_resample_graph": False,
    }


def audit_graph(original: dict[str, np.ndarray], candidate: dict[str, np.ndarray], family: str) -> dict:
    src, dst, weight = (original[key] for key in ("src", "dst", "weight"))
    new_src, new_dst, new_weight = (candidate[key] for key in ("src", "dst", "weight"))
    n, e = len(original["node_ids"]), len(src)
    if set(candidate) != set(original) or new_dst.shape != dst.shape:
        raise ValueError("Graph array keys or edge count changed")
    if not np.issubdtype(new_dst.dtype, np.integer) or np.any(new_dst < 0) or np.any(new_dst >= n):
        raise ValueError("Invalid destination node")
    invariant = {
        "node_ids": np.array_equal(original["node_ids"], candidate["node_ids"]),
        "all_arrays_except_destination": all(np.array_equal(value, candidate[key])
            for key, value in original.items() if key != "dst"),
        "source_per_edge": np.array_equal(src, new_src),
        "contact_count_per_edge": np.array_equal(weight, new_weight),
        "source_sign_per_edge": np.array_equal(original["sign"][src], candidate["sign"][new_src]),
        "out_degree": np.array_equal(np.bincount(src, minlength=n), np.bincount(new_src, minlength=n)),
        "in_degree": np.array_equal(np.bincount(dst, minlength=n), np.bincount(new_dst, minlength=n)),
        "outgoing_contacts": np.array_equal(np.bincount(src, weights=weight, minlength=n),
            np.bincount(new_src, weights=new_weight, minlength=n)),
        "incoming_contacts": np.array_equal(np.bincount(dst, weights=weight, minlength=n),
            np.bincount(new_dst, weights=new_weight, minlength=n)),
        "no_duplicate_directed_edges": len(np.unique(new_src.astype(np.int64) * n + new_dst)) == e,
        "no_new_self_loops": not np.any((new_src == new_dst) & (src != dst)),
        "existing_self_loops_preserved_exactly": np.array_equal(src == dst, new_src == new_dst),
    }
    required = set(invariant) - {"in_degree", "incoming_contacts", "existing_self_loops_preserved_exactly"}
    if family in {"anatomical", "rewired"}:
        required.add("in_degree")
    if family in {"anatomical", "random"}:
        required.add("existing_self_loops_preserved_exactly")
    if not all(invariant[key] for key in required):
        raise RuntimeError(f"Required {family} graph invariants failed: {invariant}")
    changed = int(np.count_nonzero(dst != new_dst))
    return {
        "nodes": n, "edges": e, "contacts": int(np.sum(new_weight, dtype=np.float64)),
        "preservation": {key: bool(value) for key, value in invariant.items()},
        "required_invariants": sorted(required),
        "changed_edge_destinations": changed, "changed_edge_fraction": changed / e,
        "self_loops": int(np.count_nonzero(new_src == new_dst)),
        "reachability": reachability_audit(candidate),
    }


def _write_graph(path: Path, graph: dict[str, np.ndarray], *, exact_source: Path | None = None) -> None:
    if path.exists():
        existing = read_graph(path)
        if set(existing) != set(graph) or any(not np.array_equal(existing[key], value) for key, value in graph.items()):
            raise RuntimeError(f"Existing campaign graph differs from deterministic reconstruction: {relative(path)}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if exact_source is not None:
                with exact_source.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            else:
                np.savez_compressed(stream, **graph)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build() -> dict:
    start = time.monotonic()
    output = HERE / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(SOURCE)
    source_metadata = load_json(SOURCE.with_suffix(".json"))
    if source_metadata.get("processed_sha256") != source_digest:
        raise RuntimeError("Original processed graph does not match its source sidecar")
    original = read_graph(SOURCE)
    if len(original["src"]) != 40755:
        raise RuntimeError("Expected 40,755 original edges and trainable gains")
    rows = []
    for family, seeds in (("anatomical", [None]), ("rewired", SEEDS), ("random", SEEDS)):
        for seed in seeds:
            graph_id = "anatomical" if seed is None else f"{family}-s{seed}"
            generation_start = time.monotonic()
            generation = {}
            if family == "anatomical":
                graph = original
            elif family == "rewired":
                graph = rewire_graph(original, seed=seed, swaps_per_edge=1.0,
                    max_attempts=1_000_000, max_swaps=100_000, forbid_self_loops=True)
                generation = graph.pop("rewire_metadata")
                if generation["accepted_swaps"] != len(original["src"]):
                    raise RuntimeError(f"Incomplete rewiring protocol for {graph_id}")
            else:
                graph = random_destinations(original, seed)
                generation = {
                    "rng": "numpy.default_rng(SeedSequence([seed, 90217]))",
                    "target_sampling": "uniform ordered sample without replacement per source, excluding that source",
                    "existing_self_loops": "retained at exactly their original edge indices",
                    "preserves": ["nodes", "edge_count", "out_degree", "source_per_edge", "contact_count_per_edge", "source_sign_per_edge", "interfaces"],
                    "does_not_preserve": ["in_degree", "incoming_contacts", "anatomical_partners", "paths"],
                }
            audit = audit_graph(original, graph, family)
            path = output / f"{graph_id}.npz"
            _write_graph(path, graph, exact_source=SOURCE if family == "anatomical" else None)
            digest = sha256(path)
            if family == "anatomical" and digest != source_digest:
                raise RuntimeError("Anatomical graph must be an exact source-file copy")
            row = {"id": graph_id, "family": family, "seed": seed, "path": relative(path),
                "sha256": digest, "metadata_path": relative(path.with_suffix(".json")),
                "generation": generation, "audit": audit}
            save_json(path.with_suffix(".json"), {
                **row, "schema_version": 1,
                "source_path": relative(SOURCE), "source_sha256": source_digest,
                "source_metadata_sha256": sha256(SOURCE.with_suffix(".json")),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "generation_seconds": time.monotonic() - generation_start,
                "task_labels_used": False, "topology_filtered_by_reachability": False,
                "required_plastic_scope": "all", "trainable_gain_count": 40755,
                "source_code_sha256": {relative(Path(__file__)): sha256(Path(__file__)),
                    relative(REPO / "3 Connectome/flybrain/controls.py"): sha256(REPO / "3 Connectome/flybrain/controls.py")},
            })
            rows.append(row)
            print(f"{graph_id}: changed={audit['changed_edge_fraction']:.6f}, "
                f"functional_pairs_24={audit['reachability']['all_input_output_group_pairs_reachable_within_24_edges']}", flush=True)
    if sha256(SOURCE) != source_digest:
        raise RuntimeError("Original graph changed during graph construction")
    result = {"schema_version": 1, "graphs": rows, "seeds_per_control_family": SEEDS,
        "source_path": relative(SOURCE), "source_sha256": source_digest,
        "n_graphs": len(rows), "task_labels_used": False,
        "all_graphs_retained_regardless_of_reachability": True,
        "elapsed_seconds": time.monotonic() - start}
    save_json(output / "index.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build()
    print(f"Saved {result['n_graphs']} graphs in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
