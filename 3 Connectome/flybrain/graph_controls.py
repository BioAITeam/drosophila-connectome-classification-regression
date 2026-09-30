"""Label-free graph controls with explicit conservation checks.

Original NPZ and Feather inputs are read-only. Derived graphs use new paths
and an integrity-checked provenance manifest.
No function in this module accepts task features, classes or target values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .controls import rewire_graph
from .data import _source_paths
from .download import ensure_ha, sha256_file
from .paths import portable_metadata, portable_path, repo_path


def _load_verified(path: str | Path) -> tuple[dict, dict]:
    path = repo_path(path).resolve()
    metadata = json.loads(path.with_suffix('.json').read_text())
    digest = sha256_file(path)
    if metadata.get('processed_sha256') != digest:
        raise ValueError(f'Processed graph checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        graph = {key: archive[key] for key in archive.files}
    return graph, metadata


def _validate_edges(graph: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    src, dst, weight = (np.asarray(graph[name]) for name in ('src', 'dst', 'weight'))
    n = len(graph['node_ids'])
    if src.ndim != 1 or dst.shape != src.shape or weight.shape != src.shape:
        raise ValueError('Edge arrays must have matching one-dimensional shapes')
    if not all(np.issubdtype(a.dtype, np.integer) for a in (src, dst)):
        raise ValueError('Edge endpoints must be integer indices')
    if len(src) and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n):
        raise ValueError('Invalid edge endpoint')
    if not np.all(np.isfinite(weight) & (weight > 0) & (weight == np.floor(weight))):
        raise ValueError('Restricted controls require exact positive integer contact counts')
    if np.asarray(graph['sign']).shape != (n,) or not np.isin(graph['sign'], [-1, 0, 1]).all():
        raise ValueError('Signs must be one fixed -1, 0 or +1 value per neuron')
    if len(np.unique(graph['node_ids'])) != n:
        raise ValueError('Duplicate body IDs')
    keys = src.astype(np.int64) * n + dst
    if len(np.unique(keys)) != len(src):
        raise ValueError('Aggregate duplicate directed edges before constructing controls')
    return src, dst, weight, n


def anatomical_type_codes(node_ids: np.ndarray, annotation_path: str | Path) -> tuple[np.ndarray, dict]:
    """Align original annotation types by bodyId; missing types remain explicit."""
    annotation_path = repo_path(annotation_path).resolve()
    annotations = pd.read_feather(annotation_path, columns=['bodyId', 'type'])
    if annotations.bodyId.duplicated().any() or len(np.unique(node_ids)) != len(node_ids):
        raise ValueError('Duplicate body IDs prevent unambiguous annotation alignment')
    rows = pd.Index(annotations.bodyId).get_indexer(node_ids)
    if np.any(rows < 0):
        raise ValueError('Graph body IDs missing from original annotation table')
    selected = annotations.iloc[rows]['type']
    # Distinguish missing annotations from literal sentinel-like strings.
    labels = [(bool(pd.isna(value)), '' if pd.isna(value) else str(value)) for value in selected]
    vocabulary = sorted(set(labels))
    lookup = {label: number for number, label in enumerate(vocabulary)}
    codes = np.asarray([lookup[label] for label in labels], dtype=np.int32)
    return codes, {
        'source_path': portable_path(annotation_path), 'source_sha256': sha256_file(annotation_path),
        'columns': ['bodyId', 'type'], 'alignment': 'exact bodyId, no task labels',
        'unknown_type_neurons': int(selected.isna().sum()),
        'type_vocabulary': [{'code': index, 'missing': missing, 'type': label or None}
                            for index, (missing, label) in enumerate(vocabulary)],
        'warning': 'Missing annotation types form an explicit unknown stratum; unknown is not a recovered biological cell type.',
    }


def match_input_population(full_graph: dict, reference_graph: dict) -> tuple[dict, dict]:
    """Keep every full-graph node/edge but stimulate only reference input IDs.

    Other arrays are shared with the caller and never modified. The new input
    array is independent. Full and reference output bodyId/group pairs must
    already be identical; this function refuses to change the decoder.
    """
    full_ids, reference_ids = np.asarray(full_graph['node_ids']), np.asarray(reference_graph['node_ids'])
    if len(np.unique(full_ids)) != len(full_ids) or len(np.unique(reference_ids)) != len(reference_ids):
        raise ValueError('Duplicate body IDs prevent exact input mapping')
    positions = pd.Index(full_ids).get_indexer(reference_ids)
    if np.any(positions < 0):
        raise ValueError('Reference contains body IDs absent from the full graph')
    for graph in (full_graph, reference_graph):
        for key in ('input_group', 'output_group'):
            if np.asarray(graph[key]).shape != np.asarray(graph['node_ids']).shape:
                raise ValueError(f'Invalid {key} shape')
    def pairs(graph, key):
        selected = graph[key] >= 0
        return set(zip(graph['node_ids'][selected].tolist(), graph[key][selected].tolist()))
    if pairs(full_graph, 'output_group') != pairs(reference_graph, 'output_group'):
        raise ValueError('Full/reference output body IDs or groups differ')
    if any(not np.any(reference_graph['input_group'] == group) for group in range(4)):
        raise ValueError('Reference must supply all four fixed input channels')
    selected = reference_graph['input_group'] >= 0
    if not np.isin(reference_graph['input_group'], [-1, 0, 1, 2, 3]).all():
        raise ValueError('Invalid fixed input channel code')
    mapped = np.full(len(full_ids), -1, dtype=np.int64)
    mapped[positions[selected]] = reference_graph['input_group'][selected]
    if np.any(full_graph['input_group'][positions[selected]] != reference_graph['input_group'][selected]):
        raise ValueError('Reference input IDs have inconsistent original anatomical channel labels')
    result = dict(full_graph)
    result['input_group'] = mapped
    unchanged = {key: bool(np.array_equal(value, result[key])) for key, value in full_graph.items() if key != 'input_group'}
    audit = {
        'all_other_arrays_unchanged': all(unchanged.values()), 'unchanged_arrays': unchanged,
        'exact_reference_input_ids_and_groups': pairs(result, 'input_group') == pairs(reference_graph, 'input_group'),
        'exact_reference_output_ids_and_groups': True,
        'full_nodes': int(len(full_ids)), 'full_edges': int(len(full_graph['src'])),
        'original_input_neurons': int(np.count_nonzero(full_graph['input_group'] >= 0)),
        'matched_input_neurons': int(selected.sum()),
        'matched_input_group_sizes': [int(np.sum(mapped == group)) for group in range(4)],
        'output_group_sizes': [int(np.sum(result['output_group'] == group)) for group in range(3)],
        'input_body_ids_by_group': {str(group): sorted(full_ids[mapped == group].tolist()) for group in range(4)},
        'output_body_ids_by_group': {str(group): sorted(full_ids[result['output_group'] == group].tolist()) for group in range(3)},
        'direct_encoding_changed': False,
    }
    return result, audit


def audit_rewiring(original: dict, candidate: dict, type_codes: np.ndarray | None = None) -> dict:
    """Check realized conservation using O(N+E) arrays, never an N by N matrix."""
    src, dst, weight, n = _validate_edges(original)
    new_src, new_dst, new_weight, new_n = _validate_edges(candidate)
    if n != new_n or len(src) != len(new_src):
        raise ValueError('Control changed graph size')
    same = lambda first, second: bool(np.array_equal(first, second))
    incoming = lambda destination, values: np.bincount(destination, weights=values, minlength=n)
    signs = np.asarray(original['sign'])
    new_signs = np.asarray(candidate['sign'])
    sign_key = dst * 3 + signs[src].astype(np.int64) + 1
    new_sign_key = new_dst * 3 + new_signs[new_src].astype(np.int64) + 1
    profile = lambda keys, values=None: np.bincount(keys, weights=values, minlength=n * 3)
    audit = {
        'node_ids': same(original['node_ids'], candidate['node_ids']),
        'source_indexed_weight': same(weight, new_weight) and same(src, new_src),
        'source_neuron_sign': same(signs, new_signs),
        'in_degree': same(incoming(dst, None), incoming(new_dst, None)),
        'out_degree': same(incoming(src, None), incoming(new_src, None)),
        'incoming_strength': same(incoming(dst, weight), incoming(new_dst, new_weight)),
        'outgoing_strength': same(incoming(src, weight), incoming(new_src, new_weight)),
        'incoming_source_sign_counts': same(profile(sign_key), profile(new_sign_key)),
        'incoming_source_sign_strength': same(profile(sign_key, weight), profile(new_sign_key, new_weight)),
        'no_new_self_loops': bool(not np.any((new_src == new_dst) & (src != dst))),
        'no_duplicate_directed_edges': True,  # _validate_edges checks both graphs independently.
        'fixed_node_arrays': all(same(original[key], candidate[key]) for key in original
                                 if isinstance(original[key], np.ndarray)
                                 and original[key].shape == (n,) and key not in ('src', 'dst', 'weight', 'plastic_mask')),
    }
    if type_codes is not None:
        type_codes = np.asarray(type_codes)
        if type_codes.shape != (n,) or not np.issubdtype(type_codes.dtype, np.integer) or (type_codes < 0).any():
            raise ValueError('Type codes must be nonnegative integers aligned to graph nodes')
        # Per-edge equality is stronger than conservation of a global mixing table.
        audit['source_type_per_edge'] = same(type_codes[src], type_codes[new_src])
        audit['target_type_per_edge'] = same(type_codes[dst], type_codes[new_dst])
        def sparse_profile(a, b, values):
            keys = np.column_stack((a, b))
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            return unique, np.bincount(inverse), np.bincount(inverse, weights=values)
        def profile_equal(a, b):
            return all(same(x, y) for x, y in zip(a, b))
        audit['incoming_source_type_profile'] = profile_equal(
            sparse_profile(dst, type_codes[src], weight), sparse_profile(new_dst, type_codes[new_src], new_weight))
        audit['outgoing_target_type_profile'] = profile_equal(
            sparse_profile(src, type_codes[dst], weight), sparse_profile(new_src, type_codes[new_dst], new_weight))
    changed = int(np.count_nonzero(new_dst != dst))
    return {'preservation': audit, 'changed_edge_destinations': changed,
            'changed_edge_fraction': changed / len(src) if len(src) else 0.,
            'nodes': n, 'edges': len(src)}


def restricted_rewire_graph(graph: dict, type_codes: np.ndarray, *, seed: int = 0,
                            swaps_per_edge: float = 1., max_attempts: int = 1_000_000,
                            max_swaps: int = 100_000) -> tuple[dict, dict]:
    """Swap destinations within exact contact/sign/source-type/target-type strata.

    A swap conserves each neuron's in/out degree and strength, its incoming
    source-sign and source-type profiles, and its outgoing target-type profile.
    Contact counts and source IDs stay at their original edge indices. Mixing
    is explicitly partial: constrained strata may have few admissible swaps.
    """
    src, dst, weight, n = _validate_edges(graph)
    type_codes = np.asarray(type_codes)
    if type_codes.shape != (n,) or not np.issubdtype(type_codes.dtype, np.integer) or (type_codes < 0).any():
        raise ValueError('Type codes must be nonnegative integers aligned to graph nodes')
    if not np.isfinite(swaps_per_edge) or swaps_per_edge < 0 or max_attempts < 0 or max_swaps < 0:
        raise ValueError('Swap budgets must be finite and nonnegative')
    result = dict(graph)
    result['dst'] = dst.copy()
    destination = result['dst']
    keys = np.rec.fromarrays([weight, graph['sign'][src], type_codes[src], type_codes[dst]],
                             names='contacts,sign,source_type,target_type')
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    groups = np.split(order, np.cumsum(counts)[:-1])
    eligible = [group for group in groups if len(group) >= 2
                and len(np.unique(src[group])) >= 2 and len(np.unique(dst[group])) >= 2]
    probabilities = np.asarray([len(group) for group in eligible], dtype=np.float64)
    if probabilities.size:
        probabilities /= probabilities.sum()
    edge_keys = set((src.astype(np.int64) * n + dst).tolist())
    target = min(int(np.ceil(len(src) * swaps_per_edge)), int(max_swaps))
    rng = np.random.default_rng(seed)
    attempts = accepted = 0
    while eligible and accepted < target and attempts < max_attempts:
        attempts += 1
        group = eligible[int(rng.choice(len(eligible), p=probabilities))]
        first, second = rng.choice(group, size=2, replace=False)
        a, c, b, d = int(src[first]), int(src[second]), int(destination[first]), int(destination[second])
        if a == c or b == d or a == d or c == b:
            continue
        new_first, new_second = a * n + d, c * n + b
        if new_first in edge_keys or new_second in edge_keys:
            continue
        edge_keys.remove(a * n + b)
        edge_keys.remove(c * n + d)
        edge_keys.update((new_first, new_second))
        destination[first], destination[second] = d, b
        accepted += 1
    audit = audit_rewiring(graph, result, type_codes)
    if not all(audit['preservation'].values()):
        raise RuntimeError(f'Restricted rewiring failed its independent preservation audit: {audit}')
    metadata = {
        'topology': 'restricted_rewired_control_not_original_malecns', 'seed': int(seed),
        'strata': ['integer_contact_count', 'source_neuron_sign', 'source_cell_type', 'target_cell_type'],
        'total_strata': len(groups), 'eligible_strata': len(eligible),
        'eligible_edge_indices': int(sum(len(group) for group in eligible)),
        'requested_swaps': target, 'accepted_swaps': accepted, 'attempts': attempts,
        'swaps_per_edge': float(swaps_per_edge), 'max_swaps': int(max_swaps), 'max_attempts': int(max_attempts),
        'budget_reached': accepted == target, 'audit': audit,
        'plastic_mask_policy': 'Original edge-index mask retained; require plastic_scope=all for this artificial graph.',
        'mixing': 'Bounded partial perturbation within constrained strata; neither uniform random mixing nor erased anatomy is asserted.',
        'task_labels_used': False,
    }
    return result, metadata


def _publish(graph: dict, output_path: Path, parent_metadata: dict, *, mode: str, additions: dict) -> dict:
    output_path = ensure_ha(output_path)
    if output_path.exists() or output_path.with_suffix('.json').exists():
        raise FileExistsError(f'Refusing to replace a processed graph: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('xb') as stream:
        np.savez_compressed(stream, **{key: value for key, value in graph.items() if isinstance(value, np.ndarray)})
    metadata = dict(parent_metadata)
    metadata.update(additions)
    metadata.update(schema_version=2, mode=mode, created_utc=datetime.now(timezone.utc).isoformat(),
                    processed_sha256=sha256_file(output_path), parent_processed_sha256=parent_metadata['processed_sha256'],
                    task_labels_used_in_selection=False)
    with output_path.with_suffix('.json').open('x') as stream:
        json.dump(portable_metadata(metadata), stream, indent=2)
        stream.write('\n')
    return {'path': portable_path(output_path), 'sha256': metadata['processed_sha256'], 'metadata': portable_path(output_path.with_suffix('.json'))}


def prepare_round4_graphs(raw_root: str | Path, processed_dir: str | Path,
                          seeds=(137, 241, 997)) -> dict:
    """Prepare or reuse verified anatomical and control graphs."""
    processed_dir = ensure_ha(processed_dir)
    raw_paths = _source_paths(raw_root)
    if processed_dir.is_relative_to(raw_paths['annotations'].parent):
        raise ValueError('Processed controls cannot be written into raw data')
    reference_path = processed_dir / 'visual_mb.npz'
    reference, reference_metadata = _load_verified(reference_path)
    full_path = processed_dir / 'full.npz'
    full_digest = sha256_file(full_path)
    original_full_metadata = json.loads(full_path.with_suffix('.json').read_text())
    if full_digest != original_full_metadata.get('processed_sha256'):
        raise ValueError('Full graph checksum mismatch')
    if reference_metadata['source_sha256']['annotations'] != sha256_file(raw_paths['annotations']):
        raise ValueError('Annotation source checksum mismatch')
    manifest = {'schema_version': 1, 'task_labels_used': False,
                'original_full': {'path': portable_path(full_path), 'sha256': full_digest},
                'original_subgraph': {'path': portable_path(reference_path), 'sha256': reference_metadata['processed_sha256']},
                'graphs': {}}

    def existing(path, parent_sha, mode, seed=None):
        if not path.exists() and not path.with_suffix('.json').exists():
            return None
        metadata = json.loads(path.with_suffix('.json').read_text())
        digest = sha256_file(path)
        if (metadata.get('processed_sha256') != digest or metadata.get('parent_processed_sha256') != parent_sha
                or metadata.get('mode') != mode or (seed is not None and metadata.get('rewire', {}).get('seed') != seed)):
            raise ValueError(f'Existing derived graph has incompatible provenance: {path}')
        return {'path': portable_path(path), 'sha256': digest, 'metadata': portable_path(path.with_suffix('.json'))}

    matched_path = processed_dir / 'full_matched_inputs.npz'
    record = existing(matched_path, full_digest, 'full_matched_inputs')
    if record is None:
        full, full_metadata = _load_verified(full_path)
        matched, audit = match_input_population(full, reference)
        connectivity = dict(full_metadata['connectivity_validation'])
        connectivity['input_population_sizes'] = audit['matched_input_group_sizes']
        # Every reference edge remains in the full graph; exact shortest paths
        # may differ from both parent graphs and are not reused as measurements.
        connectivity.pop('input_to_output_shortest_hops', None)
        connectivity['input_to_output_path_note'] = 'Reference paths remain present; full-graph shortest distances were not recomputed.'
        record = _publish(matched, matched_path, full_metadata, mode='full_matched_inputs', additions={
            'anatomical_topology_preserved': True, 'interface_audit': audit,
            'interface_reference': portable_path(reference_path), 'interface_reference_sha256': reference_metadata['processed_sha256'],
            'encoding_fixed': 'exact reference-subgraph R7/R8 input body IDs; all full-graph neurons and edges retained',
            'connectivity_validation': connectivity,
        })
        del matched, full
    manifest['graphs']['full_matched_inputs'] = record
    codes, type_metadata = anatomical_type_codes(reference['node_ids'], raw_paths['annotations'])
    for seed in seeds:
        for kind in ('restricted', 'degree'):
            key = f'visual_mb_{kind}_s{seed}'
            path = processed_dir / f'{key}.npz'
            record = existing(path, reference_metadata['processed_sha256'], f'{kind}_rewired', int(seed))
            if record is None:
                if kind == 'restricted':
                    graph, rewire = restricted_rewire_graph(reference, codes, seed=int(seed))
                else:
                    graph = rewire_graph(reference, seed=int(seed))
                    rewire = graph.pop('rewire_metadata')
                    rewire['audit'] = audit_rewiring(reference, graph, codes)
                record = _publish(graph, path, reference_metadata, mode=f'{kind}_rewired', additions={
                    'anatomical_topology_preserved': False, 'rewire': rewire,
                    'type_annotations': type_metadata, 'required_plastic_scope': 'all',
                    'anatomical_edges_added': None,
                    'connectivity_validation': None,
                    'parent_connectivity_validation': reference_metadata['connectivity_validation'],
                    'warning': 'Artificial topology control. Contact counts and node IDs originate in MaleCNS; changed partners are not anatomical observations.',
                })
                print(f'{key}: {rewire["audit"]["changed_edge_fraction"]:.3%} of edge destinations changed', flush=True)
            manifest['graphs'][key] = record
    manifest_path = processed_dir / 'round4-graph-controls.json'
    text = json.dumps(portable_metadata(manifest), indent=2) + '\n'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Existing round4 graph manifest differs; select a new artifact directory')
    else:
        with manifest_path.open('x') as stream:
            stream.write(text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--processed-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_round4_graphs(args.raw_root, args.processed_dir), indent=2))


if __name__ == '__main__':
    main()
