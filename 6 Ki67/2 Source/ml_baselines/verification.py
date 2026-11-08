"""Recompute saved ML metrics without loading models or fitting parameters."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .data import load_data, sha256


def verify_selected_predictions(repo, baselines):
    from .run_baselines import metrics
    @lru_cache(maxsize=24)
    def inputs(dataset, representation, task):
        return load_data(repo, dataset, representation, task)
    checked = models = 0
    for context, result in baselines.items():
        data=inputs(*context[:3])
        for selected in result['selected_models']:
            relative=Path(selected['model'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Model reference must be repository-relative')
            model=repo/relative
            if sha256(model)!=selected['model_sha256']:
                raise ValueError(f'Model checksum mismatch: {relative}')
            models+=1
            for split in ('val','test'):
                path=model.parent/f'{split}-predictions.npz'
                with np.load(path,allow_pickle=False) as values:
                    if not np.array_equal(values['sample_ids'],data[f'ids_{split}']):
                        raise ValueError(f'Prediction sample identity mismatch: {relative}')
                    if not np.array_equal(values['y_true'],data[f'y_{split}']):
                        raise ValueError(f'Prediction target mismatch: {relative}')
                    actual=metrics(values['y_true'],values['y_pred'],values['scores'],values['classes'])
                for key in ('balanced_accuracy','accuracy','macro_f1','roc_auc'):
                    if not np.isclose(actual[key],selected['metrics'][split][key],rtol=0,atol=1e-12):
                        raise ValueError(f'Saved {key} differs from predictions: {relative}')
                checked+=1
    return {'selected_models_verified':models,'prediction_files_verified':checked,
            'metric_comparisons':checked*4,'feature_contexts_verified':inputs.cache_info().currsize,
            'models_loaded':0,'models_trained':0,'absolute_tolerance':1e-12}
