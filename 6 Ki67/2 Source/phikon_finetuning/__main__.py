"""Prepare, train, evaluate or summarize the six reported Phikon-v2 runs."""
from __future__ import annotations
import argparse
import json
import os
from .runtime import DEFAULT_ARCHIVE, DEFAULT_OUTPUT, activate, expected_jobs, load, local, record, relative, sha256, verify_jobs, writable


def evaluate_selected(archive_value, output_value, job_id, gpu):
    import numpy as np
    archive, output = local(archive_value), writable(output_value)
    job = next(j for j in expected_jobs() if j['id'] == job_id)
    protocol = load(output / 'protocol.json')
    if protocol != load(archive / 'protocol.json'):
        raise ValueError('Prepared data and archived model protocols differ')
    from .train import Inputs, build_model, evaluate, libraries, save_predictions
    if gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        activate(output)
        import torch
        torch.set_num_threads(2)
        device = 'cpu'
    else:
        torch = libraries(output, gpu)
        device = 'cuda'
    checkpoint = archive / 'runs' / job_id / 'best.pt'
    complete = load(archive / 'runs' / job_id / 'complete.json')
    if sha256(checkpoint) != complete['checkpoint_sha256']:
        raise ValueError('Archived checkpoint checksum mismatch')
    model = build_model(output, 'joint', protocol, 'binary30', job['architecture'], device=device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['job'] != job or saved['epoch'] != complete['best_epoch']:
        raise ValueError('Checkpoint identity mismatch')
    model.load_state_dict(saved['model_state'])
    del saved
    inputs = Inputs(output, 'joint', 'binary30', device=device)
    metrics, probabilities = evaluate(model, inputs, 'test', protocol['training']['microbatch_size'])
    destination = output / 'evaluation' / job_id
    destination.mkdir(parents=True, exist_ok=True)
    save_predictions(destination / 'selected-test.npz', inputs, 'test', probabilities, complete['best_epoch'])
    record(destination / 'metrics.json', {'job': job, 'best_epoch': complete['best_epoch'],
                                        'metrics': metrics, 'source_checkpoint': relative(checkpoint)})
    print(json.dumps({'job': job_id, 'balanced_accuracy': metrics['balanced_accuracy'],
                      'output': relative(destination)}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Build normalized pixel caches from downloaded raw images')
    prepare.add_argument('--archive', default=DEFAULT_ARCHIVE)
    prepare.add_argument('--output', default=DEFAULT_OUTPUT)
    prepare.add_argument('--workers', type=int, choices=range(1, 33), default=8)
    train = sub.add_parser('train', help='Explicitly train one of the six runs on a specified GPU')
    train.add_argument('--output', default=DEFAULT_OUTPUT)
    train.add_argument('--job', required=True, choices=[j['id'] for j in expected_jobs()])
    train.add_argument('--gpu', required=True, type=int)
    evaluate = sub.add_parser('evaluate', help='Evaluate one archived checkpoint on the prepared test images')
    evaluate.add_argument('--archive', default=DEFAULT_ARCHIVE)
    evaluate.add_argument('--output', default=DEFAULT_OUTPUT)
    evaluate.add_argument('--job', required=True, choices=[j['id'] for j in expected_jobs()])
    evaluate.add_argument('--gpu', type=int, help='Physical CUDA index. Omit for CPU inference.')
    report = sub.add_parser('report', help='Reconstruct results, parameter counts and timings without images or training')
    report.add_argument('--archive', default=DEFAULT_ARCHIVE)
    report.add_argument('--output', default=DEFAULT_OUTPUT + '-report')
    report.add_argument('--verify-checkpoints', action='store_true')
    status = sub.add_parser('status', help='List the six runs and their completion records')
    status.add_argument('--archive', default=DEFAULT_ARCHIVE)
    args = parser.parse_args(argv)
    if getattr(args, 'gpu', None) is not None and args.gpu < 0:
        parser.error('--gpu must be a nonnegative CUDA device index')
    if args.command == 'prepare':
        from .prepare import prepare as build
        build(args.archive, args.output, args.workers)
    elif args.command == 'train':
        output = writable(args.output)
        jobs = load(output / 'jobs.json')
        verify_jobs(jobs)
        job = next(j for j in jobs if j['id'] == args.job)
        from .train import libraries, train as execute
        libraries(output, args.gpu)
        execute(output, job, load(output / 'protocol.json'), args.gpu)
    elif args.command == 'evaluate':
        evaluate_selected(args.archive, args.output, args.job, args.gpu)
    elif args.command == 'report':
        from .report import report as summarize
        summarize(args.archive, args.output, args.verify_checkpoints)
    else:
        archive = local(args.archive)
        for job in expected_jobs():
            directory = archive / 'runs' / job['id']
            completed = directory / 'complete.json'
            state = 'complete' if completed.is_file() else ('partial' if directory.exists() else 'not started')
            print(f"{job['id']}: {state}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
