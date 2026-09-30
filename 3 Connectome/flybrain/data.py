"""Prepare connectome graphs from immutable MaleCNS Feather exports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .download import SOURCE_FILES, ensure_ha, sha256_file
from .paths import RAW, portable_metadata, portable_path, repo_path, writable_output


def _source_paths(raw_root: str | Path) -> dict[str, Path]:
    root = repo_path(raw_root).resolve()
    if (root / 'malecns-v1.0').is_dir():
        root = root / 'malecns-v1.0'
    paths = {role: root / filename for role, filename in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing official raw files: ' + ', '.join(missing))
    return paths


def _load_anatomical_edges(path: Path, node_ids: np.ndarray,
                           min_weight: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read Arrow record batches: never materialize the 151 M segment table."""
    index = pd.Index(node_ids)
    sources, targets, weights = [], [], []
    total_rows = 0
    with pa.memory_map(str(path), 'r') as mapped:
        reader = pa.ipc.open_file(mapped)
        if not {'body_pre', 'body_post', 'weight'}.issubset(reader.schema.names):
            raise ValueError(f'Unexpected connectivity schema: {reader.schema}')
        for batch_no in range(reader.num_record_batches):
            batch = reader.get_batch(batch_no)
            total_rows += batch.num_rows
            pre = batch.column('body_pre').to_numpy(zero_copy_only=False)
            post = batch.column('body_post').to_numpy(zero_copy_only=False)
            counts = batch.column('weight').to_numpy(zero_copy_only=False)
            si, di = index.get_indexer(pre), index.get_indexer(post)
            keep = (si >= 0) & (di >= 0) & (counts >= min_weight)
            if keep.any():
                sources.append(si[keep].astype(np.int64))
                targets.append(di[keep].astype(np.int64))
                weights.append(counts[keep].astype(np.float32))
    if not sources:
        raise ValueError('No edges survived the anatomical filter')
    return np.concatenate(sources), np.concatenate(targets), np.concatenate(weights), total_rows


def _cell_masks(annotations: pd.DataFrame) -> dict[str, np.ndarray]:
    types = annotations['type'].fillna('')
    classes = annotations['class'].fillna('')
    side = annotations['rootSide'].fillna(annotations['somaSide']).fillna('')
    groups = np.full(len(annotations), -1, dtype=np.int64)
    for group, (family, hemisphere) in enumerate((('R7', 'L'), ('R7', 'R'), ('R8', 'L'), ('R8', 'R'))):
        groups[(types.isin([family + suffix for suffix in ('p', 'y', 'd')]) & side.eq(hemisphere)).to_numpy()] = group
    return {
        'input_group': groups,
        'kc': classes.eq('Kenyon_Cell').to_numpy(),
        'visual_kc': types.eq('KCg-d').to_numpy(),
        'mbon': classes.eq('MBON').to_numpy(),
        'dan': classes.eq('DAN').to_numpy(),
        'reward_mask': (classes.eq('DAN') & types.str.startswith('PAM')).to_numpy(),
        'punishment_mask': (classes.eq('DAN') & types.str.startswith('PPL1')).to_numpy(),
    }


def _signs(annotations: pd.DataFrame, nt_path: Path) -> tuple[np.ndarray, dict]:
    nt = pd.read_feather(nt_path, columns=['body', 'consensus_nt'])
    if nt['body'].duplicated().any():
        raise ValueError('Duplicate body IDs in neurotransmitter table')
    consensus = nt.set_index('body')['consensus_nt'].reindex(annotations.bodyId).fillna('unknown')
    # A fixed modeling hypothesis, not a receptor-specific physiological measurement.
    mapping = {'acetylcholine': 1., 'gaba': -1., 'glutamate': -1., 'histamine': -1.,
               'dopamine': 0., 'serotonin': 0., 'octopamine': 0.}
    signs = consensus.map(mapping).fillna(1.).to_numpy(dtype=np.float32)
    dan_mask = annotations['class'].eq('DAN').to_numpy()
    dan_exceptions = dan_mask & (signs != 0)
    signs[dan_mask] = 0.
    return signs, {
        'source_column': 'consensus_nt', 'sign_map': mapping,
        'unknown_default': 1., 'consensus_counts': consensus.value_counts().to_dict(),
        'dan_class_override': 0.,
        'dan_class_override_count': int(dan_mask.sum()),
        'dan_consensus_exceptions': [
            {'bodyId': int(body), 'consensus_nt': str(nt)}
            for body, nt in zip(annotations.bodyId.to_numpy()[dan_exceptions], consensus.to_numpy()[dan_exceptions])
        ],
        'warning': 'Glutamate=-1, histamine=-1 and unknown=+1 are explicit hypotheses; receptor effects are unavailable. Monoamines have zero fast transmission; DAN anatomical edges support separate plasticity modulation.',
    }


def _adjacency(n: int, src: np.ndarray, dst: np.ndarray) -> sparse.csr_matrix:
    return sparse.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))


def _distances(adjacency: sparse.csr_matrix, starts: np.ndarray, limit: int = 50) -> np.ndarray:
    if len(starts) == 0:
        raise ValueError('An anatomical population required for path analysis is empty')
    return dijkstra(adjacency, directed=True, indices=starts, unweighted=True,
                    min_only=True, limit=limit)


def prepare_graph(raw_root: str | Path, output_path: str | Path, mode: str = 'visual_mb',
                  min_weight: int = 1, max_visual_hops: int = 2,
                  full_graph_path: str | Path | None = None) -> dict:
    """Save NPZ arrays and JSON provenance for the full graph or a visual/MB circuit.

    Selection depends only on anatomy. No task examples, targets, train/test
    splits, activity or learned parameters are inputs to this function.
    ``full_graph_path`` reuses a previously prepared full graph to avoid reading
    the segment-level export twice; its raw hashes and filter are verified.
    """
    if mode not in {'full', 'visual_mb'}:
        raise ValueError('mode must be full or visual_mb')
    if min_weight < 1 or max_visual_hops < 1:
        raise ValueError('min_weight and max_visual_hops must be positive integers')
    output_path = writable_output(output_path)
    if output_path.suffix != '.npz':
        raise ValueError('Processed graph output must have .npz suffix')
    if output_path.exists() or output_path.with_suffix('.json').exists():
        raise FileExistsError(f'Choose a new processed filename; refusing replacement: {output_path}')
    paths = _source_paths(raw_root)
    if output_path.is_relative_to(paths['annotations'].parent):
        raise ValueError('Processed output must not be written into raw data')
    source_hashes = {role: sha256_file(path) for role, path in paths.items()}
    annotations = pd.read_feather(paths['annotations'])
    if annotations.bodyId.duplicated().any():
        raise ValueError('Duplicate IDs in raw annotation table')
    # Includes annotated sensory axons lacking status=Traced; excludes glia and fragments.
    annotations = annotations.loc[annotations.superclass.notna()].sort_values('bodyId').reset_index(drop=True)
    node_ids = annotations.bodyId.to_numpy(dtype=np.uint64)
    masks = _cell_masks(annotations)
    signs, sign_metadata = _signs(annotations, paths['neurotransmitters'])
    if full_graph_path is not None:
        full_path = repo_path(full_graph_path)
        old_meta = json.loads(full_path.with_suffix('.json').read_text())
        if (old_meta['mode'] != 'full' or old_meta['source_sha256'] != source_hashes or
                old_meta['min_weight'] != min_weight):
            raise ValueError('Cached full graph has incompatible provenance or filters')
        if sha256_file(full_path) != old_meta['processed_sha256']:
            raise ValueError('Cached full graph checksum mismatch')
        with np.load(full_path, allow_pickle=False) as cached:
            if not np.array_equal(cached['node_ids'], node_ids):
                raise ValueError('Cached full graph has a different node ordering')
            src, dst, weight = cached['src'], cached['dst'], cached['weight']
        raw_edge_rows = old_meta['raw_segment_edge_rows']
    else:
        src, dst, weight, raw_edge_rows = _load_anatomical_edges(paths['weights'], node_ids, min_weight)
    n_full, e_full = len(node_ids), len(src)
    print(f'Annotated universe: {n_full:,} neurons, {e_full:,} directed edges', flush=True)
    adjacency = _adjacency(n_full, src, dst)
    input_dist = _distances(adjacency, np.flatnonzero(masks['input_group'] >= 0))
    if not masks['visual_kc'].any():
        raise ValueError('No KCg-d cells in annotations; refusing a fabricated fallback')
    # Three MBON cell types with strongest observed input from visual KCg-d;
    # tie-breaking uses type names only, never task labels.
    kc_to_mbon = masks['visual_kc'][src] & masks['mbon'][dst]
    mbon_counts = np.bincount(dst[kc_to_mbon], weights=weight[kc_to_mbon], minlength=n_full)
    candidates = annotations.loc[masks['mbon'] & np.isfinite(input_dist), ['type']].copy()
    candidates['kc_contacts'] = mbon_counts[candidates.index]
    totals = candidates.groupby('type')['kc_contacts'].sum().sort_index().sort_values(ascending=False, kind='stable')
    output_types = totals[totals > 0].index[:3].tolist()
    if len(output_types) != 3:
        raise ValueError('Need three reachable MBON types with real visual-KC input')
    output_group = np.full(n_full, -1, dtype=np.int64)
    for group, cell_type in enumerate(output_types):
        output_group[(annotations.type.eq(cell_type) & np.isfinite(input_dist)).to_numpy()] = group
    if mode == 'visual_mb':
        reverse = adjacency.transpose().tocsr()
        distance_to_kc = _distances(reverse, np.flatnonzero(masks['visual_kc']), limit=max_visual_hops)
        # Keep the complete induced graph of the bounded visual corridor and MB cells.
        keep_node = ((input_dist + distance_to_kc) <= max_visual_hops)
        keep_node |= masks['visual_kc'] | (output_group >= 0) | masks['dan']
    else:
        keep_node = np.ones(n_full, dtype=bool)
    del adjacency
    keep_edge = keep_node[src] & keep_node[dst]
    remap = np.full(n_full, -1, dtype=np.int64)
    remap[keep_node] = np.arange(keep_node.sum())
    src, dst, weight = remap[src[keep_edge]], remap[dst[keep_edge]], weight[keep_edge]
    selected = annotations.loc[keep_node].reset_index(drop=True)
    selected_masks = {name: values[keep_node] for name, values in masks.items()}
    output_group = output_group[keep_node]
    plastic = selected_masks['kc'][src] & selected_masks['mbon'][dst]
    # Diagnostic direct current encoding uses only existing KC cells, distributed
    # deterministically by side then cell type then body ID into four equal bins.
    direct_group = np.full(len(selected), -1, dtype=np.int64)
    direct_rows = selected.loc[selected_masks['kc']].sort_values(['rootSide', 'somaSide', 'type', 'bodyId'], na_position='last').index.to_numpy()
    for group, rows in enumerate(np.array_split(direct_rows, 4)):
        direct_group[rows] = group
    arrays = {
        'node_ids': node_ids[keep_node], 'src': src, 'dst': dst,
        'weight': weight.astype(np.float32), 'sign': signs[keep_node],
        'input_group': selected_masks['input_group'], 'output_group': output_group,
        'direct_input_group': direct_group,
        'reward_mask': selected_masks['reward_mask'],
        'punishment_mask': selected_masks['punishment_mask'], 'plastic_mask': plastic,
        'kc_mask': selected_masks['kc'], 'mbon_mask': selected_masks['mbon'],
    }
    connectivity = validate_graph(arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse replacement to preserve experiment provenance; choose a new filename.
    with output_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    metadata = {
        'schema_version': 1, 'dataset': 'MaleCNS', 'version': 'v1.0', 'mode': mode,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_paths': {key: portable_path(value) for key, value in paths.items()},
        'source_sha256': source_hashes, 'processed_sha256': sha256_file(output_path),
        'node_selection': 'body annotations with non-null superclass; ascending bodyId',
        'raw_segment_edge_rows': raw_edge_rows, 'full_annotated_nodes': n_full,
        'full_annotated_edges': e_full, 'nodes': len(arrays['node_ids']), 'edges': len(src),
        'synaptic_contacts': int(weight.sum(dtype=np.float64)), 'min_weight': min_weight,
        'node_order_sha256': hashlib.sha256(arrays['node_ids'].tobytes()).hexdigest(),
        'max_visual_hops': max_visual_hops if mode == 'visual_mb' else None,
        'subgraph_selection': 'Induced graph of nodes on input-to-KCg-d paths <= max_visual_hops, all KCg-d, selected MBON types and all DAN' if mode == 'visual_mb' else None,
        'input_group_names': ['R7_L', 'R7_R', 'R8_L', 'R8_R'],
        'encoding_fixed': 'bilateral_r7_r8_intensity_channels', 'retinotopic': False,
        'input_types': ['R7p', 'R7y', 'R7d', 'R8p', 'R8y', 'R8d'],
        'output_types': output_types,
        'output_selection': 'Three reachable MBON types ranked by total KCg-d contacts; lexical tie-break',
        'output_type_visual_kc_contacts': {key: int(totals[key]) for key in output_types},
        'direct_input_selection': 'Existing KCs sorted rootSide,somaSide,type,bodyId; four equal contiguous bins; diagnostic encoding only',
        'plastic_selection': 'Every retained anatomical KC-to-MBON edge',
        'reward_selection': 'class DAN and type prefix PAM',
        'punishment_selection': 'class DAN and type prefix PPL1',
        'dopamine_warning': 'PAM reward and PPL1 punishment are task-specific modeling hypotheses, not universal physiological labels.',
        'sign_hypothesis': sign_metadata,
        'connectivity_validation': connectivity,
        'anatomical_edges_added': 0, 'artificial_neurons_added': 0,
        'task_labels_used_in_selection': False,
    }
    with output_path.with_suffix('.json').open('x') as stream:
        json.dump(portable_metadata(metadata), stream, indent=2)
        stream.write('\n')
    print(f'Prepared {mode}: {len(arrays["node_ids"]):,} neurons, {len(src):,} edges, {plastic.sum():,} KC→MBON plastic edges → {output_path}', flush=True)
    return metadata


def validate_graph(arrays: dict[str, np.ndarray]) -> dict:
    """Validate node indexing, fixed populations and anatomical task routes."""
    n = len(arrays['node_ids'])
    src, dst = arrays['src'], arrays['dst']
    for name in ('sign', 'input_group', 'output_group', 'reward_mask', 'punishment_mask'):
        if arrays[name].shape != (n,):
            raise ValueError(f'Wrong node-array shape: {name}')
    if any(arrays[name].shape != src.shape for name in ('dst', 'weight', 'plastic_mask')):
        raise ValueError('Edge arrays have inconsistent shapes')
    if not np.isin(arrays['sign'], [-1., 0., 1.]).all():
        raise ValueError('Invalid neurotransmitter sign hypothesis')
    if len(np.unique(arrays['node_ids'])) != n:
        raise ValueError('Duplicate graph node IDs')
    if len(src) == 0 or src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n:
        raise ValueError('Invalid graph edge indexing')
    if not np.all(np.isfinite(arrays['weight']) & (arrays['weight'] > 0)):
        raise ValueError('Weights must be finite positive anatomical contact counts')
    for name, count in [('input_group', 4), ('output_group', 3), ('direct_input_group', 4)]:
        if name in arrays and any(not np.any(arrays[name] == group) for group in range(count)):
            raise ValueError(f'Missing fixed population in {name}')
    if np.any(arrays['reward_mask'] & arrays['punishment_mask']):
        raise ValueError('Reward/punishment populations overlap')
    adjacency = _adjacency(n, src, dst)
    reachability = []
    for group in range(4):
        distance = _distances(adjacency, np.flatnonzero(arrays['input_group'] == group))
        row = [float(distance[arrays['output_group'] == target].min()) for target in range(3)]
        if not np.isfinite(row).all():
            raise ValueError(f'Input group {group} cannot reach every output group: {row}')
        reachability.append(row)
    plastic_targets = np.zeros(n, dtype=bool)
    plastic_targets[dst[arrays['plastic_mask']]] = True
    modulation = {}
    modulation_by_output = {}
    for name in ('reward_mask', 'punishment_mask'):
        edges = arrays[name][src] & plastic_targets[dst]
        modulation[name] = int(edges.sum())
        if not edges.any():
            raise ValueError(f'No anatomical {name}→plastic-target connections')
        modulation_by_output[name] = [
            int(np.sum(edges & (arrays['output_group'][dst] == group))) for group in range(3)
        ]
    return {
        'input_to_output_shortest_hops': reachability,
        'input_population_sizes': [int((arrays['input_group'] == i).sum()) for i in range(4)],
        'output_population_sizes': [int((arrays['output_group'] == i).sum()) for i in range(3)],
        'plastic_edges': int(arrays['plastic_mask'].sum()),
        'modulatory_edges_to_plastic_targets': modulation,
        'modulatory_edges_by_output_group': modulation_by_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=RAW)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--mode', choices=['full', 'visual_mb'], default='visual_mb')
    parser.add_argument('--min-weight', type=int, default=1)
    parser.add_argument('--max-visual-hops', type=int, default=2)
    parser.add_argument('--full-graph-path', type=Path)
    args = parser.parse_args()
    prepare_graph(**vars(args))


if __name__ == '__main__':
    main()
