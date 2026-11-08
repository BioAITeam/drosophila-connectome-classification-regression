"""Export stable inputs for manuscript Tables 11-13; retain campaign filenames 10-12."""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np

from .models import MODEL_NAMES

LABELS = dict(zip(MODEL_NAMES, ("KNN", "Decision tree", "Bagging", "Random forest", "AdaBoost", "SVC",
    "Extra Trees", "XGBoost", "Logistic regression", "Gradient boosting", "QDA", "LDA", "Gaussian NB", "MLP")))
DATASETS = ("bcdata", "shid", "shid_v2")
REPS = ("dinov2", "phikonv2")
TASKS = ("multiclass", "binary5", "binary10", "binary30")


def size_description(rows, model):
    values = [json.loads(row["complexity"]) for row in rows]
    def span(*names):
        counts=[sum(item.get(name,0) for name in names) for item in values]
        return f"{min(counts):,}-{max(counts):,}"
    if model=="knn":
        return f"{span('stored_training_rows')} stored observations, {span('stored_feature_values')} feature values"
    if model=="svc":
        return f"{span('fitted_dual_coefficients','fitted_intercepts')} dual coefficients and intercepts, {span('support_vectors')} support vectors"
    if model=="qda":
        return f"{span('fitted_means','stored_rotation_values','stored_scaling_values','fitted_priors')} stored mean, rotation, scaling, and prior values"
    if model=="lda":
        return f"{span('fitted_mean_values','decision_coefficients','decision_intercepts','fitted_priors','stored_scaling_values','stored_covariance_values')} stored mean, decision, covariance, and prior values"
    if model=="gaussian_nb":
        return f"{span('fitted_means','fitted_variances','fitted_priors')} fitted mean, variance, and prior values"
    if model=="logistic_regression":
        return f"{span('optimized_weight_parameters')} coefficients and intercepts"
    if model=="mlp":
        return f"{span('optimized_weight_parameters')} trainable weights and biases"
    description=f"{span('split_thresholds')} thresholds, {span('leaf_output_values')} leaf outputs"
    if model=="adaboost": description+=f", {span('fitted_ensemble_weights')} ensemble weights"
    return description


def write_tables(summary, baselines, connectomes, contexts, capacities):
    from .make_report import write_csv, write_json, write_text
    context_lookup={(r['dataset'],r['representation'],r['task'],r['model']):r for r in contexts}
    grid=[]
    for model in MODEL_NAMES:
        row=context_lookup['bcdata','phikonv2','multiclass',model]
        entry={'classifier':model,'label':LABELS[model],'candidates':row['n_candidates'],
               'candidate_grid_json':row['tuning_grid']}
        for dataset in DATASETS:
            source=baselines[dataset,'phikonv2','multiclass',model]
            entry[f'{dataset}_selected_json']=json.dumps(source['selection']['best_params'],sort_keys=True)
        grid.append(entry)
    ba=[]
    for dataset in DATASETS:
        for model in (*MODEL_NAMES,'anatomical_connectome'):
            entry={'dataset':dataset,'model':model,'label':LABELS.get(model,'Anatomical connectome')}
            for rep in REPS:
                for task in TASKS:
                    result=(connectomes if model=='anatomical_connectome' else baselines)[dataset,rep,task,model]
                    metric=result['test_metrics']['balanced_accuracy']
                    entry[f'{rep}_{task}_mean_percent']=100*metric['mean']
                    entry[f'{rep}_{task}_sample_sd_percent']=100*metric['sample_sd']
            ba.append(entry)
    compute=[]
    grouped=defaultdict(list)
    for row in capacities: grouped[row['model']].append(row)
    for model in MODEL_NAMES:
        rows=grouped[model]
        if len(rows)!=72: raise ValueError(f'Expected 72 selected models for {model}')
        records=[v for k,v in baselines.items() if k[-1]==model]
        compute.append({'classifier':model,'label':LABELS[model],
            'model_size':size_description(rows,model),'selected_training_runs':len(rows),
            'search_seconds_sum':sum(r['search_seconds'] for r in records),
            'training_seconds_median':float(np.median([r['selected_training_seconds'] for r in rows])),
            'test_seconds_median':float(np.median([r['test_prediction_seconds'] for r in rows]))})
    for name,rows in [('table10_ml_grid',grid),('table11_ml_baselines',ba),('table12_ml_compute',compute)]:
        write_csv(summary/f'{name}.csv',list(rows[0]),rows)
        write_json(summary/f'{name}.json',rows)
    write_json(summary/'table_schema.json',{
        'schema':'ki67-ml-manuscript-tables-v1',
        'tables':{'tab:ki67-ml-grid':'table10_ml_grid','tab:ki67-ml-baselines':'table11_ml_baselines','tab:ki67-ml-compute':'table12_ml_compute'},
        'balanced_accuracy_units':'percent','uncertainty':'sample SD across three training seeds',
        'timing_units':'seconds','search_aggregation':'sum over 24 contexts per classifier',
        'training_and_test_aggregation':'median over 72 selected models per classifier',
        'test_scope':'complete test partition','selection':'existing validation partition',
        'representations':list(REPS),'datasets':list(DATASETS),'tasks':list(TASKS)})
    lines=['# Ki-67 ML manuscript tables','',
           'CSV and JSON files preserve unrounded values. Balanced accuracy is in percent and time is in seconds.',
           'The table_schema.json file maps the files to stable manuscript labels.','',
           '| Classifier | Search (s) | Training median (s) | Test median (s) |',
           '|---|---:|---:|---:|']
    lines += [f"| {r['label']} | {r['search_seconds_sum']:.2f} | {r['training_seconds_median']:.3f} | {r['test_seconds_median']:.4f} |" for r in compute]
    write_text(summary/'tables10-12.md','\n'.join(lines)+'\n')
    latex_tables(summary, grid, ba, compute)


PARAMETERS = {
    'knn': (('n_neighbors', 'Neighbors'), ('weights', 'Weights'), ('p', 'Minkowski exponent')),
    'decision_tree': (('max_depth', 'Maximum depth'), ('min_samples_leaf', 'Minimum leaf size'), ('class_weight', 'Class weights')),
    'bagging': (('n_estimators', 'Trees'), ('max_samples', 'Sample fraction'), ('estimator__max_depth', 'Maximum depth')),
    'random_forest': (('n_estimators', 'Trees'), ('max_depth', 'Maximum depth'), ('min_samples_leaf', 'Minimum leaf size')),
    'adaboost': (('n_estimators', 'Trees'), ('learning_rate', 'Learning rate'), ('estimator__max_depth', 'Maximum depth')),
    'svc': (('C', '$C$'), ('kernel', 'Kernel'), ('gamma', r'RBF $\gamma$')),
    'extra_trees': (('n_estimators', 'Trees'), ('max_depth', 'Maximum depth'), ('min_samples_leaf', 'Minimum leaf size')),
    'xgboost': (('n_estimators', 'Boosting rounds'), ('max_depth', 'Maximum depth'), ('learning_rate', 'Learning rate')),
    'logistic_regression': (('C', '$C$'), ('class_weight', 'Class weights')),
    'gradient_boosting': (('n_estimators', 'Boosting stages'), ('max_depth', 'Maximum depth'), ('learning_rate', 'Learning rate')),
    'qda': (('reg_param', 'Covariance regularization'),),
    'lda': (('shrinkage', 'Covariance shrinkage'),),
    'gaussian_nb': (('var_smoothing', 'Variance smoothing'),),
    'mlp': (('alpha', '$L_2$ penalty'), ('learning_rate_init', 'Initial learning rate'), ('activation', 'Activation')),
}


def latex_value(value):
    if value is None:
        return r'\mathrm{N}'
    if isinstance(value, str):
        aliases = {'uniform': 'U', 'distance': 'D', 'balanced': 'B', 'rbf': 'RBF', 'relu': 'ReLU'}
        return r'\tanh' if value == 'tanh' else r'\mathrm{' + aliases.get(value, value) + '}'
    if value and abs(value) < 0.001:
        exponent = int(round(np.log10(abs(value))))
        if np.isclose(abs(value), 10.0 ** exponent, rtol=1e-12, atol=0):
            return '10^{' + str(exponent) + '}'
    return f'{value:g}'


def latex_tables(summary, grid, ba, compute):
    """Export complete LaTeX floats from the same unrounded table records."""
    from .make_report import write_text
    def begin(caption, label, columns, header):
        return [r'\begin{table*}[!tbp]', r'\centering', r'\caption{' + caption + '}',
                r'\label{' + label + '}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
                r'\renewcommand{\arraystretch}{1.12}',
                r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}' + columns + r'@{}}',
                r'\toprule', header + r' \\', r'\midrule']
    def save(number, lines):
        write_text(summary / f'Table-{number}.tex', '\n'.join(lines + [r'\bottomrule', r'\end{tabular*}', r'\end{table*}', '']))
    lines = begin(
        'Hyperparameter grids for traditional machine learning classifiers. Selected configurations correspond to three-class classification with Phikon-v2 and follow the listed parameter order. B, S and S2 denote BCData, SHIDC-B-Ki-67 and SHIDC-B-Ki-67-V2. N denotes no depth limit or no class weighting, B balanced class weights, U uniform weights and D distance weights. RBF denotes a radial basis function kernel. Gamma applies only to the RBF SVC.',
        'tab:ki67-ml-grid', 'llrlll', 'Classifier & Hyperparameters tested & Candidates & Selected (B) & Selected (S) & Selected (S2)')
    for index, row in enumerate(grid):
        candidates = json.loads(row['candidate_grid_json'])
        parameters = PARAMETERS[row['classifier']]
        cells = []
        for key, label in parameters:
            values = []
            for candidate in candidates:
                if key in candidate and candidate[key] not in values:
                    values.append(candidate[key])
            cells.append(label + r' $\{' + ','.join(latex_value(value) for value in values) + r'\}$')
        tested = r'\makecell[l]{' + r'\\'.join(cells) + '}'
        selected = []
        for dataset in DATASETS:
            values = json.loads(row[f'{dataset}_selected_json'])
            selected.append('$(' + ','.join(latex_value(values[key]) if key in values else r'\text{-}' for key, _ in parameters) + ')$')
        if index:
            lines.append(r'\midrule')
        lines.append(' & '.join((row['label'], tested, str(row['candidates']), *selected)) + r' \\')
    save(10, lines)
    lines = begin(
        'Test balanced accuracy (percent) for classifiers trained on identical frozen features and dataset partitions. Values are mean and sample standard deviation across three training seeds. Each dataset is fitted separately. Binary columns denote PI thresholds 5, 10 and 30 percent. Selection uses validation data.',
        'tab:ki67-ml-baselines', 'lrrrrrrrr',
        r' & \multicolumn{4}{c}{DINOv2} & \multicolumn{4}{c}{Phikon-v2}')
    lines[-1:] = [r'\cmidrule(lr){2-5}\cmidrule(l){6-9}', r'Model & Three-class & 5 & 10 & 30 & Three-class & 5 & 10 & 30 \\', r'\midrule']
    names = {'bcdata': 'BCData', 'shid': 'SHIDC-B-Ki-67', 'shid_v2': 'SHIDC-B-Ki-67-V2'}
    for index, dataset in enumerate(DATASETS):
        if index:
            lines.append(r'\midrule')
        count = 402 if dataset == 'bcdata' else 700
        lines.append(r'\multicolumn{9}{l}{\textbf{' + names[dataset] + f'}} ({count} test images)' + r'} \\')
        for row in (item for item in ba if item['dataset'] == dataset):
            if row['model'] == 'anatomical_connectome':
                lines.append(r'\cmidrule(l){2-9}')
            cells = [row['label']]
            for rep in REPS:
                for task in TASKS:
                    cells.append(f"${row[f'{rep}_{task}_mean_percent']:.2f}\\pm{row[f'{rep}_{task}_sample_sd_percent']:.2f}$")
            lines.append(' & '.join(cells) + r' \\')
    save(11, lines)
    lines = begin(
        'Model size and computation times for traditional machine learning classifiers. Size entries span 72 training runs per classifier (24 combinations, three seeds). Search sums recorded search durations over the 24 combinations. Training and Test report medians over 72 runs. Test covers the complete test partition. Times are in seconds and exclude frozen feature extraction.',
        'tab:ki67-ml-compute', 'llrrr', 'Model & Model size (model-specific counts) & Search (s) & Training (s) & Test (s)')
    for row in compute:
        lines.append(' & '.join((row['label'], row['model_size'], f"{row['search_seconds_sum']:,.2f}",
                                f"{row['training_seconds_median']:.3f}", f"{row['test_seconds_median']:.4f}")) + r' \\')
    save(12, lines)
