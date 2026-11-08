"""Reconstruct manuscript Tables 3 and 15 from the selected Iris classifier runs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from analyze_results import classification_metrics, mean_sd
from common import HERE

MODELS = [
    ('knn', 'KNN'), ('decision_tree', 'Decision tree'), ('bagging', 'Bagging'),
    ('random_forest', 'Random forest'), ('adaboost', 'AdaBoost'), ('svc', 'SVC'),
    ('extra_trees', 'Extra Trees'), ('xgboost', 'XGBoost'),
    ('logistic_regression', 'Logistic regression'), ('gradient_boosting', 'Gradient boosting'),
    ('qda', 'QDA'), ('lda', 'LDA'), ('gaussian_nb', 'Gaussian NB'), ('mlp', 'MLP'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def value_range(values):
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f'{lo:,}-{hi:,}'


def model_size(model, runs):
    values = [r['model_complexity'] for r in runs]
    def count(*fields):
        return value_range([sum(v[k] for k in fields) for v in values])
    if model == 'knn':
        return f'{count("stored_training_rows")} stored observations, {count("stored_feature_values")} feature values'
    if model in {'decision_tree', 'bagging', 'random_forest', 'extra_trees', 'xgboost', 'gradient_boosting', 'adaboost'}:
        result = f'{count("split_thresholds")} thresholds, {count("leaf_output_values")} leaf outputs'
        if model == 'adaboost':
            result += f', {count("fitted_ensemble_weights")} ensemble weights'
        return result
    if model == 'svc':
        return f'{count("fitted_dual_coefficients", "fitted_intercepts")} dual coefficients and intercepts, {count("support_vectors")} support vectors'
    if model == 'logistic_regression':
        return f'{count("weights", "biases")} coefficients and intercepts'
    if model == 'mlp':
        return f'{count("weights", "biases")} trainable weights and biases'
    if model == 'qda':
        return f'{count("fitted_means", "stored_rotation_values", "stored_scaling_values", "fitted_priors")} stored mean, rotation, scaling, and prior values'
    if model == 'lda':
        return f'{count("fitted_mean_values", "decision_coefficients", "decision_intercepts", "fitted_priors", "stored_scaling_values", "stored_covariance_values")} stored mean, decision, covariance, and prior values'
    if model == 'gaussian_nb':
        return f'{count("fitted_means", "fitted_variances", "fitted_priors")} fitted mean, variance, and prior values'
    raise ValueError(model)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output == campaign or any(output.is_relative_to(campaign / part) for part in ('runs', 'graphs', 'data')):
        raise ValueError('Table output must be separate from campaign inputs and run evidence')
    output.mkdir(parents=True, exist_ok=True)
    jobs = read_json(campaign / 'jobs.json')['jobs']
    performance, compute, rows3, rows15, evidence = [], [], [], [], []
    for model, label in MODELS:
        selected = [j for j in jobs if j.get('model') == model]
        if len(selected) != 20:
            raise ValueError(f'Expected 20 selected fits for {model}')
        runs, oof = [], {}
        for job in selected:
            folder = campaign / 'runs' / job['id']
            result = read_json(folder / 'result.json')
            complete = read_json(folder / 'complete.json')
            if complete['result_sha256'] != digest(folder / 'result.json') or result['predictions_sha256'] != digest(folder / 'predictions.npz'):
                raise ValueError(f'Corrupt evidence for {job["id"]}')
            if result.get('smoke') or result.get('software_validation_only') or any(result[k] != v for k, v in job.items()):
                raise ValueError(f'Unexpected result for {job["id"]}')
            runs.append(result)
            key = (job['split_seed'], job['optimization_seed'])
            with np.load(folder / 'predictions.npz', allow_pickle=False) as data:
                oof.setdefault(key, []).append({k: data[k] for k in ('row_index', 'true', 'predicted')})
            evidence.append({'job': job['id'], 'result_sha256': digest(folder / 'result.json'), 'predictions_sha256': digest(folder / 'predictions.npz')})
        metrics = []
        if set(oof) != {(1729, 0), (1729, 1), (2718, 0), (2718, 1)}:
            raise ValueError(f'Incomplete OOF repetitions for {model}')
        for parts in oof.values():
            index = np.concatenate([p['row_index'] for p in parts])
            if len(parts) != 5 or sorted(index.tolist()) != list(range(150)):
                raise ValueError(f'Incomplete five-fold evaluation for {model}')
            metrics.append(classification_metrics(np.concatenate([p['true'] for p in parts]), np.concatenate([p['predicted'] for p in parts])))
        counts = Counter(json.dumps(r['best_params'], sort_keys=True) for r in runs)
        params_json, frequency = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        params = json.loads(params_json)
        params_text = ', '.join(f'{key}={value}' for key, value in sorted(params.items()))
        sizes = {r['grid_candidate_count'] for r in runs}
        if len(sizes) != 1:
            raise ValueError(f'Grid size differs for {model}')
        row = {'model': model, 'label': label}
        for metric in ('balanced_accuracy', 'accuracy', 'macro_f1'):
            summary = mean_sd(m[metric] for m in metrics)
            row[metric + '_mean_percent'] = 100 * summary['mean']
            row[metric + '_sd_percent'] = 100 * summary['sd']
        row.update(most_frequent_hyperparameters=params_text, times_selected=frequency, searches=20, grid_size=sizes.pop())
        performance.append(row)
        def formatted(metric):
            return f"${row[metric + '_mean_percent']:.2f}\\pm{row[metric + '_sd_percent']:.2f}$"
        rows3.append(f'{label} & {formatted("balanced_accuracy")} & {formatted("macro_f1")} & {params_text.replace("_", chr(92) + "_")} & {frequency}/20 & {row["grid_size"]} ' + r'\\')
        timing = {'model': model, 'label': label, 'model_size': model_size(model, runs), 'selected_models': 20,
                  'search_seconds_sum': sum(r['search_seconds'] for r in runs),
                  'fit_seconds_median': statistics.median(r['training_seconds'] for r in runs),
                  'test_seconds_median': statistics.median(r['test_seconds'] for r in runs)}
        compute.append(timing)
        rows15.append(f'{label} & {timing["model_size"]} & {timing["search_seconds_sum"]:.2f} & {timing["fit_seconds_median"]:.4f} & {timing["test_seconds_median"]:.4f} ' + r'\\')
    head3 = r'''\begin{table*}[!tbp]
\centering
\caption{Iris classifier performance and selected hyperparameters. BA and macro-F1 are percentages, expressed as mean $\pm$ sample standard deviation across four complete OOF evaluations. Grid size is the number of configurations per search.}
\label{tab:iris-classifier-performance}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{0.14\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\centering\arraybackslash}p{0.09\textwidth}>{\raggedright\arraybackslash}p{0.48\textwidth}>{\centering\arraybackslash}p{0.07\textwidth}>{\centering\arraybackslash}p{0.045\textwidth}@{}}
\toprule
Model & BA (\%) & Macro-F1 (\%) & Most frequently selected hyperparameters & \makecell{Times\\selected\\(out of 20)} & \makecell{Grid\\size} \\
\midrule
'''
    head15 = r'''\begin{table*}[!tbp]
\centering
\caption{Iris model sizes and computation. Size ranges cover 20 selected models per classifier. Search is the accumulated search duration. Fit and Test are median durations for fitting one selected model and evaluating its complete held-out fold. All times are in seconds.}
\label{tab:iris-classifier-compute}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lp{0.45\textwidth}rrr@{}}
\toprule
Model & Model size & Search (s) & Fit (s) & \shortstack{Test\\evaluation (s)} \\
\midrule
'''
    for number, header, rows, values in [(3, head3, rows3, performance), (15, head15, rows15, compute)]:
        text = header + '\n'.join(row + '\n' + r'\addlinespace[3pt]' for row in rows) + '\n' + r'\bottomrule' + '\n' + r'\end{tabular*}' + '\n' + r'\end{table*}' + '\n'
        (output / f'Table-{number}.tex').write_text(text)
        write_csv(output / f'Table-{number}.csv', values)
    report = {'selected_models': len(evidence), 'oof_evaluations': 56, 'table_numbers': [3, 15],
              'path_base': 'campaign root', 'generator_sha256': digest(Path(__file__)), 'evidence': evidence}
    (output / 'table-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'tables': [3, 15], 'selected_models': 280, 'output': output.name}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', '--campaign-root', dest='campaign_root', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.campaign_root, args.output)


if __name__ == '__main__':
    main()
