"""Verify archived Iris run checksums and optionally repeat selected-model inference."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from common import HERE, load_json, load_partition, sha256
from run_campaign import complete


def verify(inference=False):
    jobs = load_json(HERE / 'jobs.json')['jobs']
    if len(jobs) != 1500 or len({job['id'] for job in jobs}) != 1500:
        raise ValueError('Expected 1,500 unique jobs')
    for job in jobs:
        if not complete(job):
            raise ValueError(f'Missing completed run: {job["id"]}')
    graphs = load_json(HERE / 'graphs/index.json')['graphs']
    if Counter(g['family'] for g in graphs) != {'anatomical': 1, 'rewired': 30, 'random': 30}:
        raise ValueError('Graph family counts differ from the registered protocol')
    for graph in graphs:
        if sha256(HERE / graph['path']) != graph['sha256']:
            raise ValueError(f'Graph checksum mismatch: {graph["id"]}')
    for seed in (1729, 2718):
        for fold in range(5):
            load_partition(seed, fold)
    report = {'status': 'passed', 'completed_runs_checked': len(jobs), 'graph_checkpoints': 1220,
              'classifier_checkpoints': 280, 'graphs_checked': len(graphs), 'partitions_checked': 10,
              'inference_jobs': []}
    if inference:
        import joblib
        import torch
        from flybrain.model import SparseConnectome
        from run_graph_experiments import evaluate
        torch.set_num_threads(1)
        wanted = [j for j in jobs if j['split_seed'] == 1729 and j['optimization_seed'] == 0 and j['fold'] == 0
                  and (j['kind'] == 'ml' or j['graph_id'] in ('anatomical', 'rewired-s137', 'random-s137'))]
        for job in wanted:
            folder = HERE / 'runs' / job['id']
            part = load_partition(job['split_seed'], job['fold'])
            with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
                expected = saved['predicted']
                if job['kind'] == 'ml':
                    model = joblib.load(folder / 'best.joblib')
                    prediction = model.predict(part['x_test'])
                else:
                    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
                    with np.load(HERE / checkpoint['config']['graph_path'], allow_pickle=False) as graph:
                        model = SparseConnectome({k: graph[k] for k in graph.files}, **checkpoint['config']['model'])
                    with torch.no_grad():
                        model.gain.copy_(checkpoint['gain'])
                    prediction, probability, _ = evaluate(model, part['x_test'], part['y_test'])
                    if 'probabilities' in saved and not np.allclose(probability, saved['probabilities'], rtol=1e-6, atol=1e-7):
                        raise ValueError(f'Graph probabilities differ: {job["id"]}')
                if not np.array_equal(prediction, expected):
                    raise ValueError(f'Inference differs: {job["id"]}')
            report['inference_jobs'].append(job['id'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', action='store_true', help='Repeat inference for 3 graph families and all 14 classifier families')
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    report = verify(args.inference)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
