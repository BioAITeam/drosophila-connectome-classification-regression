"""Load the joint anatomical binary30 model without updating any parameters."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

REPO = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
CAMPAIGN = 'zenodo/payload/ki67/outputs/phikon-joint-finetune'
OUTPUT = '6 Ki67/6 Workspace/phikon-xai'


def local(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Use a repository-relative path')
    result = (REPO / path).resolve()
    result.relative_to(REPO)
    return result


def relative(value):
    return Path(value).resolve().relative_to(REPO).as_posix()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def activate(output, gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['FLYBRAIN_REPO'] = str(REPO)
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY']:
        os.environ[key] = '1'
    for key in ['TOKENIZERS_PARALLELISM', 'USE_TF', 'USE_FLAX']:
        os.environ[key] = 'false' if key == 'TOKENIZERS_PARALLELISM' else '0'
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '2'
    for key, sub in [('FLYBRAIN_TEMP_DIR', 'flybrain'), ('XDG_CACHE_HOME', 'xdg'),
                     ('CUDA_CACHE_PATH', 'cuda'), ('TMPDIR', 'tmp'), ('MPLCONFIGDIR', 'mpl')]:
        target = output / 'runtime-cache' / sub
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / '3 Connectome'))
    package = REPO / '6 Ki67/2 Source'
    spec = importlib.util.spec_from_file_location('ki67', package / '__init__.py',
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules['ki67'] = module
    spec.loader.exec_module(module)


def model_from_checkpoint(campaign, run, device='cuda'):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel
    from ki67.sequence import SequenceConnectome
    protocol = load(run / 'config.json')['protocol']
    expected = load(run / 'complete.json')['checkpoint_sha256']
    assert sha256(run / 'best.pt') == expected, 'Checkpoint checksum mismatch'
    assert sha256(campaign / 'graphs/anatomical.npz') == protocol['graph_sha256']
    with np.load(campaign / 'graphs/anatomical.npz', allow_pickle=False) as z:
        graph = {k: z[k] for k in z.files}

    class FullAnatomicalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = AutoConfig.from_pretrained(local(protocol['pretrained_snapshot']), local_files_only=True)
            self.encoder = AutoModel.from_config(config, attn_implementation='sdpa')
            self.connectome = SequenceConnectome(graph, **protocol['connectome'])
            for key in ['mean', 'std']:
                self.register_buffer(key, torch.from_numpy(np.load(campaign / 'data/joint' / f'{key}.npy')))

        def forward(self, pixels):
            with torch.autocast('cuda', dtype=torch.float16, enabled=pixels.is_cuda):
                cls = self.encoder(pixel_values=pixels).last_hidden_state[:, 0, :].float()
            standardized = ((cls.double() - self.mean) / self.std).float()
            with torch.autocast('cuda', enabled=False):
                return self.connectome(standardized)[:, :2]

    model = FullAnatomicalModel()
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    assert saved['job']['architecture'] == 'anatomical' and saved['job']['task'] == 'binary30'
    model.load_state_dict(saved['model_state'], strict=True)
    del saved
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    return model.to(device).eval(), protocol


class Predictor:
    def __init__(self, model, protocol, batch_size=16):
        import numpy as np
        self.model, self.batch_size = model, batch_size
        settings = protocol['image_preprocessing']
        self.mean = np.asarray(settings['image_mean'], dtype=np.float32)
        self.std = np.asarray(settings['image_std'], dtype=np.float32)
        self.calls = 0
        self.images = 0

    def tensor(self, images):
        import numpy as np
        import torch
        values = np.asarray(images, dtype=np.float32)
        if values.ndim != 4 or values.shape[1:] != (224, 224, 3):
            raise ValueError('Expected NHWC RGB at 224 x 224')
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError('RGB values must be finite and in [0,1]')
        array = ((values - self.mean) / self.std).transpose(0, 3, 1, 2).copy()
        return torch.from_numpy(array).to(next(self.model.parameters()).device)

    def __call__(self, images):
        import numpy as np
        import torch
        self.calls += 1
        self.images += len(images)
        values = []
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                values.append(self.model(self.tensor(images[start:start+self.batch_size])).softmax(1).cpu().numpy())
        return np.concatenate(values)

