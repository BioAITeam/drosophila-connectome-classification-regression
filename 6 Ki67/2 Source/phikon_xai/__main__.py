"""Validate, render or recompute the two published Figure 13 explanations."""
from __future__ import annotations

import argparse
import json
import shutil

from .common import activate, load, local, model_from_checkpoint, Predictor, relative, save, sha256
from ..phikon_finetuning.runtime import writable

ARCHIVE = 'zenodo/payload/ki67/outputs/phikon-xai'
OUTPUT = '6 Ki67/6 Workspace/phikon-xai'


def inventory(directory):
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob('*')) if p.is_file()
            and p.suffix in ('.npz', '.json') and p.name != 'artifact-sha256.json'
            and 'runtime-cache' not in p.parts and 'figures' not in p.parts}


def validate(archive, output):
    import numpy as np
    manifest = load(archive / 'selection.json')
    hashes = load(archive / 'artifact-sha256.json')
    for name, expected in hashes.items():
        if sha256(archive / name) != expected:
            raise ValueError('Artifact checksum mismatch: ' + name)
    diagnostics = []
    if len(manifest['examples']) != 2:
        raise ValueError('This release contains the two published examples')
    for example in manifest['examples']:
        key, target = example['key'], example['predicted_class']
        with np.load(archive / 'inputs' / f'{key}.npz', allow_pickle=False) as saved:
            rgb, segments = saved['rgb'], saved['segments']
            assert rgb.shape == (224, 224, 3) and np.isfinite(rgb).all()
            assert 0 <= rgb.min() <= rgb.max() <= 1
            assert segments.shape == (224, 224)
            assert np.array_equal(np.unique(segments), np.arange(example['n_segments']))
        for method in ('gradcam', 'shap', 'lime'):
            metadata = load(archive / method / f'{key}.json')
            assert metadata['target_class'] == target
            assert metadata['checkpoint_sha256'] == manifest['checkpoint_sha256']
            with np.load(archive / method / f'{key}.npz', allow_pickle=False) as saved:
                assert all(np.isfinite(saved[k]).all() for k in saved.files)
                values = saved['map']
                assert values.shape == (224, 224) and np.count_nonzero(values)
                probability_error = float(np.max(np.abs(saved['image_probabilities'] - example['stored_probabilities'])))
                assert probability_error < 1e-4
                if method == 'gradcam':
                    assert metadata['block_number'] == 12
                    assert metadata['target_layer'] == 'encoder.encoder.layer.11.norm1'
                    assert 0 <= values.min() <= values.max() <= 1
                else:
                    assert metadata['seed'] == example['explanation_seed']
                    assert np.array_equal(saved['segments'], segments)
                    assert np.array_equal(values, saved['segment_weights'][segments])
                row = {'example': key, 'method': method, 'probability_max_error': probability_error,
                       'map_min': float(values.min()), 'map_max': float(values.max()),
                       'nonzero_pixels': int(np.count_nonzero(values))}
                if method == 'shap':
                    reconstructed = saved['baseline_probabilities'][target] + saved['segment_weights'].sum()
                    assert abs(reconstructed - saved['image_probabilities'][target]) < 1e-8
                    row['additivity_error'] = float(abs(reconstructed - saved['image_probabilities'][target]))
                if method == 'lime':
                    row.update(weighted_r2=metadata['weighted_r2'], probability_gap=metadata['prediction_gap'])
                diagnostics.append(row)
    result = {'examples': 2, 'maps': 6, 'artifact_checksums_verified': True,
              'diagnostics': diagnostics, 'source_archive': relative(archive)}
    save(output / 'validation.json', result)
    print(json.dumps(result, indent=2))


def prepare(archive, output):
    """Recreate only the two publication input arrays from acquired raw images."""
    import numpy as np
    from PIL import Image
    from skimage.segmentation import slic
    manifest = load(archive / 'selection.json')
    destination = output / 'inputs'
    destination.mkdir(parents=True, exist_ok=True)
    for example in manifest['examples']:
        source = local(example['image_path'])
        if not source.is_file():
            raise FileNotFoundError('Acquire the source image using the data guide: ' + example['image_path'])
        if sha256(source) != example['image_sha256']:
            raise ValueError('Original image checksum mismatch')
        with Image.open(source) as image:
            rgb = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC),
                             dtype=np.float32) / np.float32(255)
        settings = manifest['segmentation']
        segments = slic(rgb, n_segments=settings['n_segments_requested'], compactness=settings['compactness'],
                        sigma=settings['sigma'], start_label=0).astype(np.int32)
        with np.load(archive / 'inputs' / f'{example["key"]}.npz', allow_pickle=False) as saved:
            if not np.array_equal(rgb, saved['rgb']) or not np.array_equal(segments, saved['segments']):
                raise ValueError('Recreated input differs; use the recorded Pillow and scikit-image versions')
        np.savez_compressed(destination / f'{example["key"]}.npz', rgb=rgb, segments=segments)
    save(output / 'selection.json', manifest)
    print('Verified original images and recreated inputs: ' + relative(destination))


def explain(archive, output, gpu):
    import numpy as np
    from . import methods
    if output.exists() and any((output / method).exists() for method in ('gradcam', 'shap', 'lime')):
        raise FileExistsError('Use a new output directory for explanation results')
    manifest = load(archive / 'selection.json')
    activate(output, gpu)
    model, protocol = model_from_checkpoint(local(manifest['campaign']), local(manifest['run']),
                                           device='cpu' if gpu is None else 'cuda')
    predictor = Predictor(model, protocol)
    save(output / 'selection.json', manifest)
    (output / 'inputs').mkdir(parents=True, exist_ok=True)
    for example in manifest['examples']:
        key, target = example['key'], example['predicted_class']
        source = archive / 'inputs' / f'{key}.npz'
        with np.load(source, allow_pickle=False) as saved:
            rgb, segments = saved['rgb'], saved['segments']
        shutil.copy2(source, output / 'inputs' / source.name)
        probabilities = predictor(rgb[None])[0]
        error = float(np.max(np.abs(probabilities - example['stored_probabilities'])))
        if error >= 1e-4 or int(probabilities.argmax()) != target:
            raise ValueError('Loaded model does not reproduce the selected prediction')
        for method in ('gradcam', 'shap', 'lime'):
            if method == 'gradcam':
                arrays, metadata = methods.gradcam(model, predictor.tensor(rgb[None]), target, block=12)
            else:
                fn = methods.kernel_shap if method == 'shap' else methods.lime_explain
                arrays, metadata = fn(predictor, rgb, segments, target,
                                      nsamples=manifest['nsamples'], seed=example['explanation_seed'])
            destination = output / method
            destination.mkdir(exist_ok=True)
            np.savez_compressed(destination / f'{key}.npz', **arrays)
            metadata.update(example=key, checkpoint_sha256=manifest['checkpoint_sha256'],
                            saved_prediction_max_error=error)
            save(destination / f'{key}.json', metadata)
            print(json.dumps({'example': key, 'method': method}), flush=True)
    save(output / 'artifact-sha256.json', inventory(output))
    validate(output, output / 'checks')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'render', 'prepare', 'explain'])
    parser.add_argument('--archive', default=ARCHIVE)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--gpu', type=int, help='CUDA device for explanation recomputation; omit for CPU')
    parser.add_argument('--verify-originals', action='store_true')
    args = parser.parse_args(argv)
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    archive, output = local(args.archive), writable(args.output)
    if args.command == 'render':
        from .render import main as render
        render(['--archive', args.archive, '--output', args.output] +
               (['--verify-originals'] if args.verify_originals else []))
    elif args.command == 'validate':
        validate(archive, output)
    elif args.command == 'prepare':
        prepare(archive, output)
    else:
        explain(archive, output, args.gpu)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
