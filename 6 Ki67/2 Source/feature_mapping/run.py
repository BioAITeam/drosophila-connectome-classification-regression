#!/usr/bin/env python3
"""Prepare, train, verify, or evaluate the 33 fixed feature-assignment runs.

Training uses the archived standardized descriptors and unchanged gain-only
BRAP-Net trainer. Each permutation is fixed across images, splits and epochs.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from runtime import (ARCHIVE, WORKSPACE, REPO, activate, implementation_hashes,
                     local, read, relative, sha, writable, write)

SEEDS = (1729, 2718, 3141)
MAPPING_SEEDS = tuple(range(10001, 10011))


def verify(experiment):
    from analyze import validate_protocol, verify_artifacts
    protocol = read(experiment / 'protocol.json')
    validate_protocol(protocol)
    verify_artifacts(protocol, experiment)
    for mapping in protocol['mappings']:
        if mapping != read(experiment / 'mappings' / mapping['id'] / 'mapping.json'):
            raise ValueError('Stored mapping differs from its sealed definition')
    return protocol


def prepare(experiment, archive):
    import numpy as np
    import torch
    from ki67 import training
    writable(experiment)
    if (experiment / 'protocol.json').exists():
        return verify(experiment)
    if experiment.exists() and any(experiment.iterdir()):
        raise ValueError('Use a new empty experiment directory')
    archived = verify(archive)
    source = local(archived['source'])
    inputs = training.Inputs(source, 'joint', 'phikonv2', 'multiclass')
    assert {s: len(y) for s, y in inputs.y.items()} == {'train': 2128, 'val': 464, 'test': 1102}
    config = read(archive / 'config.json')
    config['output_root'] = relative(experiment)
    mappings = []
    for index, seed in enumerate((None, *MAPPING_SEEDS)):
        permutation = np.arange(1024, dtype=np.int64) if seed is None else np.random.default_rng(seed).permutation(1024)
        mapping = {'id': f'mapping-{index:02d}', 'mapping_seed': seed,
                   'permutation': permutation.tolist(),
                   'permutation_sha256': hashlib.sha256(np.asarray(permutation, dtype='<i8').tobytes()).hexdigest()}
        assert mapping == archived['mappings'][index]
        directory = experiment / 'mappings' / mapping['id']
        write(directory / 'mapping.json', mapping)
        for name in ('graphs', 'features'):
            link = directory / name
            link.symlink_to(os.path.relpath(source / name, link.parent), target_is_directory=True)
        (directory / 'manifests').mkdir()
        link = directory / 'manifests/joint.json'
        link.symlink_to(os.path.relpath(source / 'manifests/joint.json', link.parent))
        mappings.append(mapping)
    protocol = {**archived, 'created_at': datetime.now(timezone.utc).isoformat(),
                'mappings': mappings, 'implementation_sha256': implementation_hashes(),
                'versions': {'python': __import__('sys').version.split()[0],
                             'torch': torch.__version__, 'numpy': np.__version__}}
    protocol.pop('release', None)
    write(experiment / 'config.json', config)
    write(experiment / 'jobs.json', read(archive / 'jobs.json'))
    protocol['campaign_hashes'] = {name: sha(experiment / name) for name in ('config.json', 'jobs.json')}
    write(experiment / 'protocol.json', protocol)
    return protocol


@contextmanager
def permuted_inputs(permutation):
    import torch
    from ki67 import training
    original = training.Inputs
    class MappedInputs(original):
        def x(self, split, indices, device):
            values = super().x(split, indices, device)
            order = torch.as_tensor(permutation, dtype=torch.long, device=values.device)
            return values.index_select(1, order)
    training.Inputs = MappedInputs
    try:
        yield
    finally:
        training.Inputs = original


def run_directory(experiment, mapping, seed):
    return experiment / 'mappings' / mapping / 'runs/joint/phikonv2/multiclass/anatomical' / f'seed-{seed}'


def train(experiment, device, mapping_id=None, seed=None):
    from ki67 import training
    writable(experiment)
    protocol = verify(experiment)
    config = read(experiment / 'config.json')
    for mapping in protocol['mappings']:
        if mapping_id and mapping['id'] != mapping_id:
            continue
        for selected_seed in SEEDS:
            if seed is not None and selected_seed != seed:
                continue
            job = {'kind': 'train', 'dataset': 'joint', 'representation': 'phikonv2',
                   'graph': 'anatomical', 'task': 'multiclass', 'seed': selected_seed,
                   'id': f"{mapping['id']}-seed-{selected_seed}",
                   'input_mapping': {key: mapping[key] for key in ('mapping_seed', 'permutation_sha256')}}
            job['input_mapping'].update(mapping_id=mapping['id'],
                permutation_path=relative(experiment / 'mappings' / mapping['id'] / 'mapping.json'))
            print(json.dumps({'event': 'training', 'job': job['id']}), flush=True)
            with permuted_inputs(mapping['permutation']):
                result = training.train_job(config, experiment / 'mappings' / mapping['id'], job, device)
            print(json.dumps({'event': 'complete', 'job': job['id'], 'best_epoch': result['best_epoch']}), flush=True)


def smoke(experiment, output, device, all_checkpoints=False):
    import numpy as np
    import torch
    from ki67 import training
    protocol = verify(experiment)
    source = local(protocol['source'])
    inputs = training.Inputs(source, 'joint', 'phikonv2', 'multiclass')
    model = training.make_model(source, 'anatomical', protocol['training'], device)
    permutation = np.asarray(protocol['mappings'][1]['permutation'])
    with permuted_inputs(permutation):
        mapped = training.Inputs(source, 'joint', 'phikonv2', 'multiclass')
        for split in ('train', 'val', 'test'):
            raw = inputs.x(split, np.arange(2), device)
            actual = mapped.x(split, np.arange(2), device)
            assert torch.equal(actual, raw[:, permutation])
            assert torch.equal(actual[:, np.argsort(permutation)], raw)
            assert np.array_equal(mapped.y[split], inputs.y[split])
    x = inputs.x('train', np.arange(2), device)
    with permuted_inputs(np.arange(1024)):
        identity = training.Inputs(source, 'joint', 'phikonv2', 'multiclass')
        assert torch.equal(model(x), model(identity.x('train', np.arange(2), device)))
    model.zero_grad()
    torch.nn.functional.cross_entropy(model(x[:, permutation]),
        torch.as_tensor(inputs.y['train'][:2], device=device)).backward()
    assert torch.isfinite(model.gain.grad).all() and torch.count_nonzero(model.gain.grad) > 0
    assert [name for name, value in model.named_parameters() if value.requires_grad] == ['gain']
    results = []
    mappings = protocol['mappings'] if all_checkpoints else protocol['mappings'][:2]
    seeds = SEEDS if all_checkpoints else SEEDS[:1]
    for mapping in mappings:
        for seed in seeds:
            directory = run_directory(experiment, mapping['id'], seed)
            if not (directory / 'best.pt').exists():
                continue
            saved = torch.load(directory / 'best.pt', map_location=device, weights_only=False)
            with torch.no_grad():
                model.gain.copy_(saved['gain'])
            with permuted_inputs(mapping['permutation']):
                mapped = training.Inputs(source, 'joint', 'phikonv2', 'multiclass')
                metrics, payload = training.evaluate(model, mapped, 'test', 16, device)
            best = read(directory / 'best.json')
            with np.load(directory / best['prediction_path'], allow_pickle=False) as expected:
                assert np.array_equal(payload['y_pred'], expected['y_pred'])
                assert np.allclose(payload['probabilities'], expected['probabilities'], atol=2e-6, rtol=2e-5)
                maximum_error = float(np.max(np.abs(payload['probabilities'] - expected['probabilities'])))
            assert abs(metrics['balanced_accuracy'] - best['test_metrics']['balanced_accuracy']) < 1e-12
            results.append({'mapping_id': mapping['id'], 'seed': seed,
                            'balanced_accuracy': metrics['balanced_accuracy'],
                            'maximum_probability_difference': maximum_error})
    report = {'passed': True, 'fixed_permutation_all_splits': True, 'inverse_roundtrip': True,
              'identity_exact': True, 'finite_nonzero_gain_gradient': True,
              'only_gain_trainable': True, 'sequence': model.sequence_stats(1024),
              'checkpoint_predictions_reproduced': results}
    write(output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'smoke', 'verify', 'status'))
    parser.add_argument('--archive', default=ARCHIVE)
    parser.add_argument('--experiment', help='Repository-relative input/output experiment directory')
    parser.add_argument('--output', default=WORKSPACE + '/verification/smoke.json')
    parser.add_argument('--device', default='cpu', help='cpu or a PyTorch CUDA device such as cuda:0')
    parser.add_argument('--mapping', choices=[f'mapping-{i:02d}' for i in range(11)])
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--all-checkpoints', action='store_true')
    args = parser.parse_args()
    activate()
    import torch
    torch.set_num_threads(4)
    experiment = local(args.experiment or (WORKSPACE + '/training' if args.command in ('prepare', 'train') else ARCHIVE))
    if args.command == 'prepare':
        protocol = prepare(experiment, local(args.archive))
        print(json.dumps({'prepared': relative(experiment), 'expected_runs': protocol['expected_runs']}))
    elif args.command == 'train':
        train(experiment, args.device, args.mapping, args.seed)
    elif args.command == 'smoke':
        smoke(experiment, local(args.output), args.device, args.all_checkpoints)
    else:
        protocol = verify(experiment)
        complete = sum((run_directory(experiment, m['id'], seed) / 'complete.json').exists()
                       for m in protocol['mappings'] for seed in SEEDS)
        print(json.dumps({'verified': True, 'expected_runs': 33, 'complete_runs': complete}))


if __name__ == '__main__':
    main()
