"""Strict OOF permutation inference without pairwise-family bootstrap expansion."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .aggregation import _load_run
from .dataset_specs import IRIS_DATASET_NAMES
from .datasets import grouped_split, permute_grouped_targets
from .metrics import classification_metrics, regression_metrics
from .paths import portable_metadata, repo_path, writable_output


LIMITATION = (
    'Iris was used in earlier exploratory rounds. These conditional permutation tests do not constitute '
    'independent external validation or establish generalization to other datasets or animals. '
    'The exchangeability null concerns complete target vectors within equal-size feature/entity groups. '
    'Five outer folds are combined into one statistic, not treated as independent replicates.'
)


def _digest(x, y):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest()


def holm_adjust(p_values):
    """Holm's step-down family-wise correction, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError('p-values must be a nonempty finite vector in [0,1]')
    order = np.argsort(p, kind='stable')
    ranked = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)).clip(max=1.)
    adjusted = np.empty_like(ranked)
    adjusted[order] = ranked
    return adjusted.tolist()


def empirical_p(observed, null_statistics):
    """One-sided Monte Carlo p with conservative inclusion of ties and +1."""
    values = np.asarray(null_statistics, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError('Observed and null statistics must be finite, with at least one null')
    exceedances = int(np.count_nonzero(values >= observed))
    return {'p_value': (1 + exceedances) / (len(values) + 1), 'exceedances_including_ties': exceedances,
            'n_permutations': len(values), 'minimum_p_value': 1. / (len(values) + 1)}


def _validate_selection(run):
    config = run.config
    if config.get('select_checkpoint', 'best_validation') != 'best_validation':
        raise ValueError('Inferential checkpoints must be selected by validation')
    if run.result.get('software_validation_only', False) or config.get('software_validation_only', False):
        raise ValueError('Software smoke tests cannot enter permutation inference')
    if config.get('shuffle_labels') or config.get('fixed_train_label_permutation_seed') is not None or config.get('frozen'):
        raise ValueError('Inferential nulls require complete fixed target permutations, not train-only/frozen controls')
    history = json.loads((run.directory / 'history.json').read_text())
    completed = run.result['epochs_completed']
    if not isinstance(completed, int) or completed < 0 or completed > int(config['epochs']):
        raise ValueError('Invalid completed epoch count')
    if [row.get('epoch') for row in history] != list(range(completed + 1)):
        raise ValueError('Epoch metrics are missing or not consecutive from initial state')
    metric = 'balanced_accuracy' if run.metadata['task'] == 'classification' else 'rmse'
    best = None
    selected = 0
    for row in history:
        if any('test' in str(key).lower() for key in row):
            raise ValueError('Per-epoch histories must not consult outer test metrics')
        if not isinstance(row.get('training'), dict) or not isinstance(row.get('validation'), dict):
            raise ValueError('Every epoch must retain train and validation metrics')
        value = float(row['validation'][metric])
        if not np.isfinite(value):
            raise ValueError('Non-finite checkpoint-selection metric')
        score = value if metric == 'balanced_accuracy' else -value
        if best is None or score > best + float(config.get('min_delta', 1e-8)):
            best, selected = score, row['epoch']
    if selected != run.result['selected_epoch']:
        raise ValueError(f'Saved checkpoint epoch disagrees with validation history: {run.directory}')


def _protocol(config):
    ignored = {'name', 'outer_fold', 'inference_role', 'inference_task', 'label_permutation_seed'}
    return {key: value for key, value in config.items() if key not in ignored}


def _coverage(runs):
    if len(runs) != 5 or {run.fold for run in runs} != set(range(5)):
        raise ValueError('Each observed/permuted dataset requires exactly five distinct outer folds')
    runs = sorted(runs, key=lambda run: run.fold)
    first = runs[0]
    n = len(first.y)
    assignment = np.full(n, -1, dtype=np.int64)
    after = np.empty(n, dtype=np.float64)
    before = np.empty(n, dtype=np.float64)
    with np.load(first.directory / 'dataset/dataset.npz', allow_pickle=False) as saved:
        x = saved['raw_x'].copy()
        original_y = saved['original_raw_y'].copy() if 'original_raw_y' in saved else first.y.copy()
    original_fingerprint = _digest(x, original_y)
    if first.metadata.get('original_raw_array_sha256') != original_fingerprint:
        raise ValueError('Original target fingerprint mismatch')
    permutation = first.metadata.get('label_permutation')
    role = first.config['inference_role']
    seed = first.config.get('label_permutation_seed')
    if role == 'null':
        expected, expected_permutation = permute_grouped_targets(original_y, first.groups, seed)
        if not np.array_equal(first.y, expected) or permutation != expected_permutation:
            raise ValueError('Target permutation or group-preserving donor metadata is inconsistent')
        if not np.array_equal(np.sort(original_y), np.sort(first.y)):
            raise ValueError('Target multiplicities changed under the null')
    elif permutation is not None or seed is not None or not np.array_equal(first.y, original_y):
        raise ValueError('Observed dataset unexpectedly contains a permutation')
    for run in runs:
        if (run.fingerprint != first.fingerprint or not np.array_equal(run.groups, first.groups)
                or _protocol(run.config) != _protocol(first.config)
                or run.metadata.get('label_permutation') != permutation
                or run.metadata.get('original_raw_array_sha256') != original_fingerprint
                or run.metadata.get('source') != first.metadata.get('source')):
            raise ValueError('Outer folds do not share one dataset, permutation and training protocol')
        expected_indices = grouped_split(x, first.y, task=first.metadata['task'], seed=1729,
                                         outer_fold=run.fold, groups=first.groups)
        with np.load(run.directory / 'dataset/dataset.npz', allow_pickle=False) as saved:
            for partition, expected in expected_indices.items():
                if not np.array_equal(saved[f'indices_{partition}'], expected):
                    raise ValueError('Saved train/validation/test splits differ from complete null-dataset resplitting')
            if role == 'null' and ('original_raw_y' not in saved or not np.array_equal(saved['original_raw_y'], original_y)):
                raise ValueError('Original targets differ across null folds')
        if np.any(assignment[run.row_index] >= 0):
            raise ValueError('An observation occurs in multiple outer test folds')
        assignment[run.row_index] = run.fold
        after[run.row_index] = run.predictions['after']
        before[run.row_index] = run.predictions['before']
    if np.any(assignment < 0):
        raise ValueError('OOF coverage does not include every original observation')
    if any(len(np.unique(assignment[first.groups == group])) != 1 for group in np.unique(first.groups)):
        raise ValueError('A dependent feature/entity group crosses outer test folds')
    task = first.metadata['task']
    measure = (lambda p: classification_metrics(first.y, p, n_classes=first.metadata['n_classes'])) if task == 'classification' else (lambda p: regression_metrics(first.y, p))
    metrics, initial = measure(after), measure(before)
    statistic = metrics['balanced_accuracy'] if task == 'classification' else -metrics['rmse']
    return {
        'task_name': first.config['inference_task'], 'task': task, 'role': role, 'seed': seed,
        'statistic': statistic, 'metrics': metrics, 'initial_metrics': initial,
        'original_fingerprint': original_fingerprint, 'permuted_fingerprint': first.fingerprint,
        'source': first.metadata['source'], 'target_units': first.metadata.get('target_units'),
        'protocol': _protocol(first.config), 'permutation': permutation,
        'x': x, 'original_y': original_y, 'y': first.y, 'groups': first.groups,
        'after': after, 'before': before, 'assignment': assignment,
        'runs': [{'fold': run.fold, 'run': str(run.directory), 'selected_epoch': run.result['selected_epoch'],
                  'epochs_completed': run.result['epochs_completed']} for run in runs],
    }


def analyze_permutation_campaign(folder, records, expected_permutations=199, *,
                                 expected_tasks=IRIS_DATASET_NAMES, expected_seeds=None):
    """Audit complete 5-fold observed/null fits and correct the task-level tests.

    Round 4 defaults require five Iris tasks and seeds 10000..10198. Optional
    explicit task names/seeds support later declared datasets without changing
    the calculation. No result is emitted from incomplete or inconsistent fits.
    Complexity is linear in saved runs plus sorting the small task p-value set.
    """
    if isinstance(expected_permutations, bool) or not isinstance(expected_permutations, int) or expected_permutations < 1:
        raise ValueError('expected_permutations must be a positive integer')
    expected_tasks = tuple(expected_tasks)
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise ValueError('Expected task names must be nonempty and unique')
    seeds = tuple(range(10000, 10000 + expected_permutations)) if expected_seeds is None else tuple(expected_seeds)
    if len(seeds) != expected_permutations or len(set(seeds)) != len(seeds):
        raise ValueError('Expected permutation seeds must be distinct and match the declared count')
    buckets = defaultdict(list)
    for record in records:
        run = _load_run(record)
        role, task_name = run.config.get('inference_role'), run.config.get('inference_task')
        seed = run.config.get('label_permutation_seed')
        if role not in ('observed', 'null') or task_name not in expected_tasks or run.metadata['name'] != task_name:
            raise ValueError('Record is outside the declared inferential roles/tasks')
        if (role == 'observed' and seed is not None) or (role == 'null' and seed not in seeds):
            raise ValueError('Unexpected permutation seed or permuted observed record')
        if run.key[1] != 1729 or run.key[2] != 0 or run.config.get('epochs') != 40:
            raise ValueError('Round4 inference requires split 1729, optimization 0 and maximum 40 epochs')
        _validate_selection(run)
        buckets[(task_name, role, seed)].append(run)
    expected_keys = {(task, 'observed', None) for task in expected_tasks}
    expected_keys |= {(task, 'null', seed) for task in expected_tasks for seed in seeds}
    if set(buckets) != expected_keys:
        raise ValueError(f'Incomplete inferential campaign: expected {len(expected_keys)} datasets, found {len(buckets)}')
    coverages = {key: _coverage(runs) for key, runs in buckets.items()}
    reports = []
    output = writable_output(repo_path(folder))
    output.mkdir(parents=True, exist_ok=True)
    for task_name in expected_tasks:
        observed = coverages[(task_name, 'observed', None)]
        nulls = [coverages[(task_name, 'null', seed)] for seed in seeds]
        mappings = set()
        for null in nulls:
            if (null['original_fingerprint'] != observed['original_fingerprint']
                    or not np.array_equal(null['x'], observed['x'])
                    or not np.array_equal(null['original_y'], observed['original_y'])
                    or not np.array_equal(null['groups'], observed['groups'])
                    or null['source'] != observed['source'] or null['protocol'] != observed['protocol']):
                raise ValueError('Observed and null fits must share original features/targets, groups, source and protocol')
            mapping = tuple(null['permutation']['source_group_by_destination'])
            if mapping in mappings:
                raise ValueError('Repeated group permutation mapping in declared null fits')
            mappings.add(mapping)
        inference = empirical_p(observed['statistic'], [null['statistic'] for null in nulls])
        report = {
            'task_name': task_name, 'task': observed['task'], 'target_units': observed['target_units'],
            'statistic_name': 'balanced_accuracy' if observed['task'] == 'classification' else 'negative_rmse',
            'direction': 'higher is better; null ties count as exceedances',
            'observed_statistic': observed['statistic'], 'observed_metrics': observed['metrics'],
            'initial_metrics': observed['initial_metrics'], **inference,
            'original_array_sha256': observed['original_fingerprint'], 'source': observed['source'],
            'n_observations': len(observed['y']), 'n_groups': len(np.unique(observed['groups'])),
            'observed_runs': observed['runs'],
            'nulls': [{'seed': null['seed'], 'statistic': null['statistic'], 'metrics': null['metrics'],
                      'array_sha256': null['permuted_fingerprint'], 'runs': null['runs']} for null in nulls],
        }
        slug = ''.join(character if character.isalnum() or character in '_-' else '_' for character in task_name)
        target = writable_output(output / f'permutation-oof-{slug}.npz')
        all_coverage = [observed] + nulls
        np.savez_compressed(target, row_index=np.arange(len(observed['y'])), groups=observed['groups'],
                            original_true=observed['y'], permutation_seed=np.asarray([-1, *seeds]),
                            true=np.stack([c['y'] for c in all_coverage]),
                            prediction=np.stack([c['after'] for c in all_coverage]),
                            initial_prediction=np.stack([c['before'] for c in all_coverage]),
                            outer_fold=np.stack([c['assignment'] for c in all_coverage]))
        report['oof_file'] = target.name
        reports.append(report)
    for report, adjusted in zip(reports, holm_adjust([item['p_value'] for item in reports])):
        report['holm_adjusted_p_value'] = adjusted
        report['holm_reject_at_0_05'] = adjusted <= .05
    result = {
        'schema_version': 1, 'method': 'complete grouped OOF refitting under fixed global target permutations',
        'observed_and_null_checkpoint_selection': 'best validation epoch including epoch zero',
        'split_seed': 1729, 'optimization_seed': 0, 'max_epochs': 40,
        'expected_permutations_per_task': expected_permutations, 'expected_tasks': list(expected_tasks),
        'permutation_seeds': list(seeds), 'n_runs_audited': sum(len(runs) for runs in buckets.values()),
        'multiple_testing': {'method': 'Holm step-down', 'family_size': len(reports), 'alpha': .05},
        'test_folds_treated_as_independent': False, 'models_refitted_under_null': True,
        'extra_pairwise_bootstraps': False, 'limitation': LIMITATION, 'tasks': reports,
    }
    writable_output(output / 'permutation-analysis.json').write_text(json.dumps(portable_metadata(result), indent=2, allow_nan=False) + '\n')
    columns = ['task_name', 'statistic_name', 'observed_statistic', 'n_permutations',
               'exceedances_including_ties', 'p_value', 'holm_adjusted_p_value', 'holm_reject_at_0_05',
               'balanced_accuracy', 'rmse', 'mae', 'r2']
    with writable_output(output / 'permutation-analysis.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report.get(key, report['observed_metrics'].get(key)) for key in columns})
    lines = ['# Grouped permutation inference', '',
             f'{len(reports)} task-level tests; {expected_permutations} complete null permutations per task; '
             f'{result["n_runs_audited"]} audited fits. Each statistic combines five outer test folds.', '',
             '| Task | BA | RMSE | MAE | R² | Raw p | Holm p |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    number = lambda value: '—' if value is None else f'{value:.6g}'
    for report in reports:
        metric = report['observed_metrics']
        values = [metric.get(key) for key in ('balanced_accuracy', 'rmse', 'mae', 'r2')]
        lines.append('| ' + report['task_name'] + ' | ' + ' | '.join(number(value) for value in [*values, report['p_value'], report['holm_adjusted_p_value']]) + ' |')
    lines += ['', 'Classification uses balanced accuracy; regression uses negative RMSE in original units. '
              'R² is reported additionally, not tested as a sixth or subsequent contrast. '
              'The empirical p-value is (1 + number of null statistics ≥ observed)/(B + 1). '
              'Holm correction covers the declared task family.', '', LIMITATION, '']
    writable_output(output / 'permutation-analysis.md').write_text('\n'.join(lines))
    return portable_metadata(result)
