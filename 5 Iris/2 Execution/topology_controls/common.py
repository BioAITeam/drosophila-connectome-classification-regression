"""Shared paths and immutable inputs for the dedicated Iris campaign."""
from __future__ import annotations
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
DEFAULT_CAMPAIGN = REPO / '5 Iris/5 Outputs/campaigns/iris-topology-ml'
HERE = Path(os.environ.get('IRIS_CAMPAIGN_ROOT', DEFAULT_CAMPAIGN)).expanduser().resolve()
os.environ['IRIS_CAMPAIGN_ROOT'] = str(HERE)
os.environ['FLYBRAIN_REPO'] = str(REPO)
os.environ['FLYBRAIN_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(REPO / '3 Connectome'))

import hashlib
import json
import tempfile
from datetime import datetime, timezone
import numpy as np

SPLIT_SEEDS = (1729, 2718)
OPTIMIZATION_SEEDS = (0, 1)
GRAPH_SEEDS = (137, 241, 997, *range(1001, 1028))

def now():
    return datetime.now(timezone.utc).isoformat()

def relative(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve()
    if path.is_relative_to(HERE):
        return path.relative_to(HERE).as_posix()
    return path.relative_to(REPO).as_posix()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text())

def portable(value):
    if isinstance(value,Path):return relative(value)
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,str):return value.replace(str(HERE)+os.sep,'').replace(str(REPO)+os.sep,'')
    return value

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(portable(value),indent=2,allow_nan=False,ensure_ascii=True)+'\n'
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name,delete=False) as f:
        f.write(data); temp=Path(f.name)
    temp.replace(path)

def partition_path(split_seed,fold):
    return HERE/'data'/f'split-{split_seed}-fold-{fold}.npz'

def load_partition(split_seed,fold):
    path=partition_path(split_seed,fold)
    metadata=load_json(path.with_suffix('.json'))
    if sha256(path)!=metadata['npz_sha256']:
        raise ValueError('Partition checksum mismatch')
    with np.load(path,allow_pickle=False) as stored:
        return {key:stored[key] for key in stored.files}

def get_job(job_id):
    matches=[j for j in load_json(HERE/'jobs.json')['jobs'] if j['id']==job_id]
    if len(matches)!=1:raise ValueError(f'Unknown or repeated job ID: {job_id}')
    return matches[0]


def resolve_artifact(value):
    """Resolve a campaign-relative artifact reference without traversing parents."""
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a campaign-relative artifact path')
    return HERE / path
