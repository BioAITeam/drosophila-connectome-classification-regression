"""Recreate pixel caches from downloaded images and retained preprocessing."""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
from .runtime import activate, code_hashes, expected_jobs, load, local, now, record, relative, sha256, writable


def prepare(archive_value, output_value, workers=8):
    archive, output = local(archive_value), writable(output_value)
    if output.exists():
        raise FileExistsError('Use a new output directory for preparation')
    if not (archive / 'protocol.json').is_file():
        raise FileNotFoundError('Restore the joint fine-tuning artifacts from Zenodo first')
    output.mkdir(parents=True)
    activate(output)
    import numpy as np
    from PIL import Image
    from ki67.data import labels_from_counts
    from ki67.encoders import preprocess_image
    start = time.perf_counter()
    protocol = copy.deepcopy(load(archive / 'protocol.json'))
    if protocol['tasks'] != ['binary30']:
        raise ValueError('Only the six reported binary30 runs are supported')
    snapshot = local(protocol['pretrained_snapshot'])
    for name, expected in protocol['pretrained']['file_sha256'].items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f'Pretrained snapshot checksum mismatch: {name}')
    records = load(archive / 'release-inputs.json')
    data = output / 'data/joint'
    data.mkdir(parents=True)
    (output / 'graphs').mkdir()
    for name in ['mean.npy', 'std.npy']:
        source = archive / 'data/joint' / name
        if sha256(source) != records['statistics'][name]:
            raise ValueError(f'Feature scaler checksum mismatch: {name}')
        shutil.copy2(source, data / name)
    graph = archive / 'graphs/anatomical.npz'
    if sha256(graph) != protocol['graph_sha256']:
        raise ValueError('Anatomical graph checksum mismatch')
    shutil.copy2(graph, output / 'graphs/anatomical.npz')
    audit = {'archive': archive_value, 'preprocessing': protocol['image_preprocessing'], 'splits': {}}
    for split, expected_count in [('train', 2128), ('val', 464), ('test', 1102)]:
        source = archive / 'data/joint' / f'{split}-rows.json'
        if sha256(source) != records['splits'][split]['rows_sha256']:
            raise ValueError(f'Row manifest checksum mismatch: {split}')
        rows = load(source)
        if len(rows) != expected_count or len({r['sample_id'] for r in rows}) != expected_count:
            raise ValueError(f'Invalid sample count or duplicate IDs: {split}')
        def convert(row):
            image = local(row['image_path'])
            if not image.is_file():
                raise FileNotFoundError('Restore image: ' + row['image_path'])
            if sha256(image) != row['image_sha256']:
                raise ValueError('Image checksum mismatch: ' + row['image_path'])
            if labels_from_counts(row['positive_count'], row['negative_count'])['binary30'] != row['binary30']:
                raise ValueError('Label does not match annotated cell counts')
            with Image.open(image) as handle:
                pp = protocol['image_preprocessing']
                return preprocess_image(handle, pp['image_mean'], pp['image_std'])
        cache = np.lib.format.open_memmap(data / f'{split}-pixels.npy', mode='w+',
                                        dtype=np.float32, shape=(len(rows), 3, 224, 224))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, pixels in enumerate(pool.map(convert, rows)):
                cache[i] = pixels
        cache.flush()
        del cache
        shutil.copy2(source, data / f'{split}-rows.json')
        pixels_hash = sha256(data / f'{split}-pixels.npy')
        if pixels_hash != records['splits'][split]['pixels_sha256']:
            raise ValueError('Pixel cache differs from the recorded preprocessing. Check Pillow and NumPy versions.')
        audit['splits'][split] = {'n_images': len(rows), 'pixels_sha256': pixels_hash,
                                  'rows_sha256': sha256(data / f'{split}-rows.json')}
    audit['preparation_seconds'] = time.perf_counter() - start
    record(output / 'input_audit.json', audit)
    record(output / 'protocol.json', protocol)
    record(output / 'jobs.json', expected_jobs())
    record(output / 'source_sha256.json', code_hashes())
    record(output / 'prepared.json', {'created_at': now(), 'output': output_value,
                                      'archive': archive_value, 'jobs': 6})
    print(f'Prepared six runs in {relative(output)}')
