"""Recompute the published fine-tuning results from saved predictions."""
from __future__ import annotations
import csv
import math
import statistics
from .runtime import expected_jobs, load, local, record, sha256, verify_jobs, writable


def report(archive_value, output_value, verify_checkpoints=False):
    import numpy as np
    from ki67.evaluation import classification_metrics
    archive, output = local(archive_value), writable(output_value)
    jobs = load(archive / 'jobs.json')
    verify_jobs(jobs)
    output.mkdir(parents=True, exist_ok=True)
    runs, grouped = [], {}
    for job in jobs:
        folder = archive / 'runs' / job['id']
        completed = load(folder / 'complete.json')
        if completed['job'] != job:
            raise ValueError('Run metadata does not match its declared job')
        if verify_checkpoints and sha256(folder / 'best.pt') != completed['checkpoint_sha256']:
            raise ValueError('Selected checkpoint checksum mismatch: ' + job['id'])
        history = [__import__('json').loads(line) for line in (folder / 'history.jsonl').read_text().splitlines()]
        if [row['epoch'] for row in history] != list(range(121)):
            raise ValueError('Expected initialization followed by exactly 120 recorded epochs')
        best_score, best_epoch = -math.inf, None
        for row in history:
            if row['validation']['balanced_accuracy'] > best_score + 1e-8:
                best_score, best_epoch = row['validation']['balanced_accuracy'], row['epoch']
        if best_epoch != completed['best_epoch']:
            raise ValueError('Selected checkpoint does not match the validation selection rule')
        encoder = completed['trainable_encoder_parameters']
        downstream = completed['trainable_gain_parameters'] + completed['trainable_head_parameters']
        if encoder != 303351808 or downstream != (40755 if job['architecture'] == 'anatomical' else 1025):
            raise ValueError('Unexpected parameter count')
        row = {**job, 'best_epoch': best_epoch, 'encoder_parameters': encoder,
               'downstream_parameters': downstream, 'trainable_parameters': encoder + downstream,
               'training_seconds': sum(h['training_seconds'] for h in history if h['epoch'] > 0),
               'test_evaluation_seconds': completed['metrics']['test']['evaluation_seconds'],
               'run_elapsed_seconds': completed['elapsed_seconds']}
        with np.load(folder / 'selected-test.npz', allow_pickle=False) as saved:
            truth, prob, origins = saved['y_true'], saved['probabilities'], saved['source_dataset']
            sample_ids = saved['sample_id']
            if int(saved['epoch']) != best_epoch or len(truth) != 1102 or len(set(sample_ids)) != 1102:
                raise ValueError('Selected test predictions do not match the run')
        rows = load(archive / 'data/joint/test-rows.json')
        if list(sample_ids) != [r['sample_id'] for r in rows] or not np.array_equal(truth, [r['binary30'] for r in rows]):
            raise ValueError('Prediction ordering or targets differ from the test manifest')
        row['metrics'] = {}
        for subset, count in [('combined', 1102), ('bcdata', 402), ('shid_v2', 700)]:
            mask = np.ones(len(truth), bool) if subset == 'combined' else origins == subset
            if int(mask.sum()) != count:
                raise ValueError('Unexpected test subset size')
            metrics = classification_metrics(truth[mask], prob[mask])
            stored = completed['metrics']['test'] if subset == 'combined' else completed['metrics']['test']['by_dataset'][subset]
            for key in ['balanced_accuracy', 'accuracy', 'macro_f1']:
                if not math.isclose(metrics[key], stored[key], abs_tol=1e-12):
                    raise ValueError('Saved metric disagrees with retained predictions')
            row['metrics'][subset] = {k: metrics[k] for k in ['balanced_accuracy', 'accuracy', 'macro_f1']}
            grouped.setdefault((subset, job['architecture']), []).append(row['metrics'][subset])
        runs.append(row)
    result = []
    for subset in ['combined', 'bcdata', 'shid_v2']:
        for architecture in ['anatomical', 'linear']:
            values = grouped[(subset, architecture)]
            result.append({'test_set': subset, 'architecture': architecture,
                           **{key + suffix: factor * function(v[key] for v in values)
                              for key in ['balanced_accuracy', 'accuracy', 'macro_f1']
                              for suffix, function, factor in [('_mean_percent', statistics.mean, 100),
                                                               ('_sd_percent', statistics.stdev, 100)]}})
    costs = []
    for architecture in ['anatomical', 'linear']:
        selected = [r for r in runs if r['architecture'] == architecture]
        costs.append({'architecture': architecture, 'runs': 3,
                      'trainable_parameters': selected[0]['trainable_parameters'],
                      'encoder_parameters': selected[0]['encoder_parameters'],
                      'downstream_parameters': selected[0]['downstream_parameters'],
                      'median_training_minutes': statistics.median(r['training_seconds'] for r in selected) / 60,
                      'median_test_evaluation_seconds': statistics.median(r['test_evaluation_seconds'] for r in selected),
                      'median_run_elapsed_minutes': statistics.median(r['run_elapsed_seconds'] for r in selected) / 60})
    record(output / 'summary.json', {'source': archive_value, 'runs': runs, 'table': result,
                                    'parameter_counts_and_timing': costs,
                                    'timing': {'training': 'Sum of training_seconds over epochs 1-120, excluding validation and checkpoint storage',
                                               'test': 'Prediction and metric calculation for all 1102 test images using the selected model',
                                               'aggregation': 'Median across three independent runs per architecture'}})
    for filename, values in [('table.csv', result), ('parameters-and-timing.csv', costs)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    lines = ['# Exploratory joint fine-tuning', '',
             '| Test set | Architecture | BA (%) | Accuracy (%) | Macro-F1 (%) |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in result:
        fields = [f"{row[k + '_mean_percent']:.2f} ± {row[k + '_sd_percent']:.2f}" for k in ['balanced_accuracy', 'accuracy', 'macro_f1']]
        lines.append('| ' + ' | '.join([row['test_set'], row['architecture'], *fields]) + ' |')
    lines += ['', '| Architecture | Trainable parameters | Median training (min) | Median test evaluation (s) |',
              '| --- | ---: | ---: | ---: |']
    for row in costs:
        lines.append(f"| {row['architecture']} | {row['trainable_parameters']:,} | {row['median_training_minutes']:.2f} | {row['median_test_evaluation_seconds']:.2f} |")
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return {'table': result, 'parameter_counts_and_timing': costs}
