"""Reproduce the three reported SHIDC-B-Ki-67 three-class fine-tuning runs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil

from .runtime import activate, code_hashes, load, local, record, relative, sha256, writable

ARCHIVE = 'zenodo/payload/ki67/outputs/phikon-shid-finetune'
OUTPUT = '6 Ki67/6 Workspace/phikon-shid-finetune'
JOBS = [{'id': f'shid-multiclass-seed-{seed}', 'dataset': 'shid',
         'task': 'multiclass', 'seed': seed} for seed in (1729, 2718, 3141)]


def prepare(archive, output, workers):
    if output.exists():
        raise FileExistsError('Use a new output directory for preparation')
    protocol, inputs = load(archive / 'protocol.json'), load(archive / 'release-inputs.json')
    snapshot = local(protocol['pretrained_snapshot'])
    for name, digest in protocol['pretrained']['file_sha256'].items():
        if sha256(snapshot / name) != digest:
            raise ValueError(f'Pretrained snapshot checksum mismatch: {name}')
    output.mkdir(parents=True)
    activate(output)
    import numpy as np
    from PIL import Image
    from ki67.data import labels_from_counts
    from ki67.encoders import preprocess_image
    target = output / 'data/shid'
    target.mkdir(parents=True)
    (output / 'graphs').mkdir()
    if sha256(archive / 'graphs/anatomical.npz') != protocol['graph_sha256']:
        raise ValueError('Graph checksum mismatch')
    shutil.copy2(archive / 'graphs/anatomical.npz', output / 'graphs/anatomical.npz')
    for name, digest in inputs['statistics'].items():
        source = archive / 'data/shid' / name
        if sha256(source) != digest:
            raise ValueError(f'Scaler checksum mismatch: {name}')
        shutil.copy2(source, target / name)
    for split, count in [('train', 1325), ('val', 331), ('test', 700)]:
        source = archive / 'data/shid' / f'{split}-rows.json'
        if sha256(source) != inputs['splits'][split]['rows_sha256']:
            raise ValueError(f'Row manifest checksum mismatch: {split}')
        rows = load(source)
        if len(rows) != count or len({r['sample_id'] for r in rows}) != count:
            raise ValueError(f'Invalid sample inventory: {split}')

        def convert(row):
            image = local(row['image_path'])
            if not image.is_file():
                raise FileNotFoundError('Restore image using the data guide: ' + row['image_path'])
            if sha256(image) != row['image_sha256']:
                raise ValueError('Image checksum mismatch: ' + row['image_path'])
            if labels_from_counts(row['positive_count'], row['negative_count'])['multiclass'] != row['multiclass']:
                raise ValueError('Label does not match cell counts')
            pp = protocol['image_preprocessing']
            with Image.open(image) as handle:
                return preprocess_image(handle, pp['image_mean'], pp['image_std'])

        cache = np.lib.format.open_memmap(target / f'{split}-pixels.npy', mode='w+',
                                        dtype=np.float32, shape=(count, 3, 224, 224))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, pixels in enumerate(pool.map(convert, rows)):
                cache[i] = pixels
        cache.flush()
        del cache
        if sha256(target / f'{split}-pixels.npy') != inputs['splits'][split]['pixels_sha256']:
            raise ValueError('Preprocessing differs from the archived pixel checksum')
        shutil.copy2(source, target / source.name)
    record(output / 'protocol.json', protocol)
    record(output / 'jobs.json', JOBS)
    record(output / 'input_audit.json', inputs)
    record(output / 'source_sha256.json', code_hashes())
    print('Prepared: ' + relative(output))


def report(archive, output, verify_checkpoints=False):
    import numpy as np
    from ki67.evaluation import classification_metrics
    rows, reference_ids = [], None
    for job in JOBS:
        run = archive / 'runs' / job['id']
        complete = load(run / 'complete.json')
        if complete['job'] != job:
            raise ValueError('Run identity mismatch')
        if verify_checkpoints and sha256(run / 'best.pt') != complete['checkpoint_sha256']:
            raise ValueError('Selected checkpoint checksum mismatch')
        with np.load(run / 'selected-test.npz', allow_pickle=False) as saved:
            truth, probs, ids = saved['y_true'], saved['probabilities'], saved['sample_id']
            assert len(truth) == len(set(ids)) == 700
            assert int(saved['epoch']) == complete['best_epoch']
            assert np.array_equal(saved['y_pred'], probs.argmax(1))
            if reference_ids is not None:
                assert np.array_equal(ids, reference_ids)
            reference_ids = ids.copy()
            metrics = classification_metrics(truth, probs)
        for name in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision', 'macro_recall'):
            if not np.isclose(metrics[name], complete['metrics']['test'][name], atol=1e-12, rtol=0):
                raise ValueError(f'Recomputed metric differs: {job["id"]}: {name}')
        rows.append({'seed': job['seed'], 'selected_epoch': complete['best_epoch'], 'metrics': metrics,
                     'prediction_file': relative(run / 'selected-test.npz'),
                     'prediction_sha256': sha256(run / 'selected-test.npz')})
    summary = {name: {'mean_percent': float(np.mean([r['metrics'][name] for r in rows]) * 100),
                      'sample_sd_percent': float(np.std([r['metrics'][name] for r in rows], ddof=1) * 100)}
               for name in ('accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision', 'macro_recall')}
    result = {'dataset': 'SHIDC-B-Ki-67', 'task': 'multiclass', 'n_test_images': 700,
              'selection': 'Highest validation balanced accuracy within each seed; earliest epoch breaks ties',
              'runs': rows, 'summary': summary, 'checkpoints_verified': verify_checkpoints}
    record(output / 'shid-finetuning-summary.json', result)
    (output / 'shid-finetuning-summary.md').write_text(
        '# SHIDC-B-Ki-67 full Phikon-v2 fine-tuning\n\n'
        'Three seeds; mean ± sample standard deviation, percentage points.\n\n'
        '| Metric | Mean | SD |\n|---|---:|---:|\n' + ''.join(
            f'| {name} | {v["mean_percent"]:.2f} | {v["sample_sd_percent"]:.2f} |\n'
            for name, v in summary.items()))
    print(json.dumps(summary, indent=2))


def evaluate(archive, output, job, gpu):
    from .train import Inputs, build_model, evaluate as infer, libraries, save_predictions
    protocol = load(output / 'protocol.json')
    if protocol != load(archive / 'protocol.json'):
        raise ValueError('Prepared and archived protocols differ')
    if gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        activate(output)
        import torch
        torch.set_num_threads(2)
        device = 'cpu'
    else:
        torch, device = libraries(output, gpu), 'cuda'
    run = archive / 'runs' / job['id']
    complete = load(run / 'complete.json')
    if sha256(run / 'best.pt') != complete['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    model = build_model(output, 'shid', protocol, 'multiclass', 'anatomical', device=device)
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    if saved['job'] != job or saved['epoch'] != complete['best_epoch']:
        raise ValueError('Checkpoint identity mismatch')
    model.load_state_dict(saved['model_state'])
    del saved
    inputs = Inputs(output, 'shid', 'multiclass', device=device)
    metrics, probabilities = infer(model, inputs, 'test', protocol['training']['microbatch_size'])
    destination = output / 'evaluation' / job['id']
    destination.mkdir(parents=True, exist_ok=True)
    save_predictions(destination / 'selected-test.npz', inputs, 'test', probabilities, complete['best_epoch'])
    record(destination / 'metrics.json', metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['report', 'prepare', 'train', 'evaluate', 'status'])
    parser.add_argument('--archive', default=ARCHIVE)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--job', choices=[j['id'] for j in JOBS])
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--workers', type=int, choices=range(1, 33), default=8)
    parser.add_argument('--verify-checkpoints', action='store_true')
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    archive, output = local(args.archive), writable(args.output)
    if args.command == 'report':
        report(archive, output, args.verify_checkpoints)
    elif args.command == 'prepare':
        prepare(archive, output, args.workers)
    elif args.command == 'status':
        for job in JOBS:
            print(job['id'] + ': ' + ('complete' if (archive / 'runs' / job['id'] / 'complete.json').is_file() else 'missing'))
    else:
        if args.job is None or (args.command == 'train' and args.gpu is None):
            parser.error('Supply --job and, for training, --gpu')
        job = next(j for j in JOBS if j['id'] == args.job)
        if args.command == 'evaluate':
            evaluate(archive, output, job, args.gpu)
        else:
            if load(output / 'jobs.json') != JOBS:
                raise ValueError('Unexpected prepared run inventory')
            from .train import libraries, train
            libraries(output, args.gpu)
            train(output, job, load(output / 'protocol.json'), args.gpu)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
