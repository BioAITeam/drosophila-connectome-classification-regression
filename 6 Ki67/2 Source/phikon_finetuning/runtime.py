"""Shared paths and atomic records for the joint Phikon-v2 fine-tuning campaign."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
SOURCE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = '6 Ki67/6 Workspace/phikon-joint-finetune'
DEFAULT_ARCHIVE = 'zenodo/payload/ki67/outputs/phikon-joint-finetune'
TASKS = ['binary30']
SEEDS = [1729, 2718, 3141]

def relative(path):
    return Path(path).absolute().relative_to(REPO).as_posix()

def local(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Paths must be repository-relative')
    result = (REPO / path).resolve()
    result.relative_to(REPO)
    return result

def now():
    return datetime.now(timezone.utc).isoformat()

def load(path):
    return json.loads(Path(path).read_text())

def record(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def activate(output):
    os.environ['FLYBRAIN_REPO'] = str(REPO)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    for name in ['KI67_SHID_ROOT', 'KI67_BCDATA_ROOT']:
        os.environ.pop(name, None)
    for name, sub in {'FLYBRAIN_TEMP_DIR': 'flybrain', 'XDG_CACHE_HOME': 'xdg',
                      'CUDA_CACHE_PATH': 'cuda', 'TMPDIR': 'tmp'}.items():
        path = Path(output) / 'runtime-cache' / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[name] = '2'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / '3 Connectome'))

def code_hashes():
    files = list(SOURCE.glob('*.py'))
    files += list((REPO / '6 Ki67/2 Source').glob('*.py'))
    files += list((REPO / '3 Connectome/flybrain').glob('*.py'))
    return {relative(p): sha256(p) for p in sorted(files)}


def writable(value):
    path = local(value)
    workspace = (REPO / '6 Ki67/6 Workspace').resolve()
    intermediate = (REPO / '9 temp_intermediate').resolve()
    if not any(path.is_relative_to(root) and path != root for root in [workspace, intermediate]):
        raise ValueError('Output must be a subdirectory of 6 Ki67/6 Workspace or 9 temp_intermediate')
    if path.is_relative_to((REPO / 'zenodo').resolve()):
        raise ValueError('Archived inputs are read-only')
    return path


def expected_jobs():
    return [{'id': f'joint-{architecture}-binary30-seed-{seed}',
             'dataset': 'joint', 'architecture': architecture,
             'task': 'binary30', 'seed': seed}
            for seed in SEEDS for architecture in ['anatomical', 'linear']]


def verify_jobs(jobs):
    if jobs != expected_jobs():
        raise ValueError('This workflow contains exactly six joint binary30 runs')
