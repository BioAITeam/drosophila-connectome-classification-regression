"""Fine-tune Phikon-v2 with anatomical gains or a linear output layer."""
from __future__ import annotations
import argparse
import fcntl
import random
import time
import traceback
import warnings
from .runtime import *

def libraries(output, gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    activate(output)
    import torch
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    warnings.filterwarnings('ignore', message='Sparse CSR tensor support is in beta state.*')
    return torch

def build_model(output, dataset, protocol, task, architecture, device="cuda"):
    import numpy as np
    import torch
    from transformers import AutoModel
    from ki67.sequence import SequenceConnectome
    if architecture not in ['anatomical', 'linear']:
        raise ValueError('Unknown architecture')
    graph = None
    if architecture == 'anatomical':
        if sha256(output / 'graphs/anatomical.npz') != protocol['graph_sha256']:
            raise ValueError('Anatomical graph checksum mismatch')
        with np.load(output / 'graphs/anatomical.npz', allow_pickle=False) as archive:
            graph = {key: archive[key] for key in archive.files}
    class EndToEnd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(local(protocol['pretrained_snapshot']),
                local_files_only=True, trust_remote_code=False,
                attn_implementation=protocol['training']['attention_implementation'])
            self.encoder.requires_grad_(True)
            if architecture == 'anatomical':
                self.connectome = SequenceConnectome(graph, **protocol['connectome'])
            else:
                self.head = torch.nn.Linear(1024, 3 if task == 'multiclass' else 1)
            for name in ['mean', 'std']:
                self.register_buffer(name, torch.from_numpy(np.load(output / 'data' / dataset / f'{name}.npy')))
        def forward(self, pixels):
            with torch.autocast('cuda', dtype=torch.float16, enabled=pixels.is_cuda):
                cls = self.encoder(pixel_values=pixels).last_hidden_state[:, 0, :].float()
            standardized = ((cls.double() - self.mean) / self.std).float()
            if architecture == 'anatomical':
                with torch.autocast('cuda', enabled=False):
                    return self.connectome(standardized)[:, :3 if task == 'multiclass' else 2]
            return self.head(standardized)
    model = EndToEnd().to(device)
    assert all(p.requires_grad for p in model.parameters())
    if architecture == 'anatomical':
        assert [n for n,p in model.connectome.named_parameters()] == ['gain']
        assert model.connectome.gain.numel() == 40755
    else:
        assert sum(p.numel() for p in model.head.parameters()) == (3075 if task == 'multiclass' else 1025)
    return model


def task_loss(logits, targets):
    import torch.nn.functional as F
    if logits.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(logits[:, 0], targets.float())
    return F.cross_entropy(logits, targets)


def task_probabilities(logits):
    import torch
    if logits.shape[1] == 1:
        positive = logits[:, 0].sigmoid()
        return torch.stack((1 - positive, positive), dim=1)
    return logits.softmax(1)


class Inputs:
    def __init__(self, output, dataset, task, device='cuda'):
        import numpy as np
        self.task = task
        self.device = device
        self.rows = {s: load(output / 'data' / dataset / f'{s}-rows.json') for s in ['train', 'val', 'test']}
        self.pixels = {s: np.load(output / 'data' / dataset / f'{s}-pixels.npy', mmap_mode='r') for s in self.rows}
        self.targets = {s: np.array([r[task] for r in rows], dtype=np.int64) for s, rows in self.rows.items()}
        self.classes = 3 if task == 'multiclass' else 2
    def batch(self, split, indices):
        import numpy as np
        import torch
        pixels = torch.from_numpy(np.array(self.pixels[split][indices], copy=True)).to(self.device)
        target = torch.as_tensor(self.targets[split][indices], device=self.device, dtype=torch.long)
        return pixels, target

def evaluate(model, inputs, split, batch_size, progress=None):
    import numpy as np
    import torch
    from ki67.evaluation import classification_metrics
    model.eval()
    values = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(inputs.rows[split]), batch_size):
            if progress:
                progress(phase='evaluation', split=split, batch=start // batch_size,
                         batches=(len(inputs.rows[split]) + batch_size - 1) // batch_size)
            x, _ = inputs.batch(split, np.arange(start, min(start + batch_size, len(inputs.rows[split]))))
            logits = model(x)
            values.append(task_probabilities(logits).cpu().numpy())
    probs = np.concatenate(values)
    metrics = classification_metrics(inputs.targets[split], probs)
    if all('source_dataset' in row for row in inputs.rows[split]):
        origins = np.asarray([row['source_dataset'] for row in inputs.rows[split]])
        metrics['by_dataset'] = {name: classification_metrics(inputs.targets[split][origins == name], probs[origins == name])
                                 for name in sorted(np.unique(origins))}
    metrics['evaluation_seconds'] = time.perf_counter() - started
    return metrics, probs

def save_predictions(path, inputs, split, probabilities, epoch, source_dataset=None):
    import numpy as np
    from ki67.tasks import CLASS_TASKS
    origins = np.asarray([r.get('source_dataset', 'shid') for r in inputs.rows[split]])
    mask = np.ones(len(origins), dtype=bool) if source_dataset is None else origins == source_dataset
    if not mask.any():
        raise ValueError('Requested prediction scope has no samples')
    np.savez_compressed(path, y_true=inputs.targets[split][mask], probabilities=probabilities[mask],
        y_pred=probabilities[mask].argmax(1), sample_id=np.asarray([r['sample_id'] for r in inputs.rows[split]])[mask],
        class_names=np.asarray(CLASS_TASKS[inputs.task]), epoch=np.int64(epoch),
        source_dataset=origins[mask])

def checkpoint(path, model, job, protocol, epoch, score, validation, source_sha256):
    import torch
    temporary = path.with_name('.best.pt.tmp')
    torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'job': job, 'protocol': protocol, 'epoch': epoch, 'validation_balanced_accuracy': score,
        'validation_metrics': validation, 'source_sha256': source_sha256}, temporary)
    os.replace(temporary, path)

def train(output, job, protocol, gpu):
    import numpy as np
    import torch
    directory = output / 'runs' / job['id']
    directory.mkdir(parents=True, exist_ok=True)
    run_lock = (directory / '.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / 'complete.json').exists():
        complete = load(directory / 'complete.json')
        if (complete['job'] != job or load(directory / 'config.json')['protocol'] != protocol
                or sha256(directory / 'best.pt') != complete['checkpoint_sha256']):
            raise ValueError('Completed run failed identity, protocol, or checkpoint verification')
        return
    if (directory / 'config.json').exists():
        raise RuntimeError('Partial run exists. Preserve it and use an explicit fresh attempt.')
    t = protocol['training']
    start = time.perf_counter()
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    torch.cuda.manual_seed_all(job['seed'])
    rng = np.random.default_rng(job['seed'] + 103)
    hashes = code_hashes()
    sealed_sources = output / 'source_sha256.json'
    if sealed_sources.exists() and hashes != load(sealed_sources):
        raise ValueError('Source files changed after campaign preparation. Do not mix implementations.')
    record(directory / 'config.json', {'job': job, 'protocol': protocol,
        'source_sha256': hashes, 'input_audit_sha256': sha256(output / 'input_audit.json')})
    state = {'job': job, 'pid': os.getpid(), 'gpu': gpu, 'epoch': 0, 'epochs': t['epochs'],
             'started_at': now(), 'phase': 'loading'}
    def update(**fields):
        state.update(fields, heartbeat=now(), elapsed_seconds=time.perf_counter() - start)
        record(directory / 'status.json', state)
    update()
    model = build_model(output, job['dataset'], protocol, job['task'], job.get('architecture', 'anatomical'))
    inputs = Inputs(output, job['dataset'], job['task'])
    downstream = ([model.connectome.gain] if job.get('architecture', 'anatomical') == 'anatomical' else list(model.head.parameters()))
    downstream_lr = t['gain_learning_rate'] if job.get('architecture', 'anatomical') == 'anatomical' else t['head_learning_rate']
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(), 'lr': t['encoder_learning_rate']},
        {'params': downstream, 'lr': downstream_lr}],
        betas=tuple(t['betas']), eps=t['eps'], weight_decay=t['weight_decay'], foreach=False)
    scaler = torch.cuda.amp.GradScaler(init_scale=t['initial_loss_scale'])
    best_score, best_epoch = -float('inf'), -1
    total_skips = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(t['epochs'] + 1):
        epoch_started = time.perf_counter()
        update(epoch=epoch, phase='training' if epoch else 'initial_validation', split='train', batch=0)
        loss_sum, seen, steps, skipped, norm_sum = 0.0, 0, 0, 0, 0.0
        if epoch:
            model.train()
            order = rng.permutation(len(inputs.targets['train']))
            effective, micro = t['effective_batch_size'], t['microbatch_size']
            for start_idx in range(0, len(order), effective):
                indices = order[start_idx:start_idx + effective]
                optimizer.zero_grad(set_to_none=True)
                update(phase='training', split='train', batch=start_idx // effective,
                       batches=(len(order) + effective - 1) // effective)
                for part in range(0, len(indices), micro):
                    selected = indices[part:part + micro]
                    x, y = inputs.batch('train', selected)
                    logits = model(x)
                    loss = task_loss(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(loss * (len(selected) / len(indices))).backward()
                    loss_sum += float(loss.detach()) * len(selected)
                    seen += len(selected)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'])
                if torch.isfinite(norm):
                    norm_sum += float(norm)
                elif all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Gradient norm overflow with finite individual gradients')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                was_skipped = scaler.get_scale() < old_scale
                skipped += int(was_skipped)
                steps += int(not was_skipped)
                if skipped > 10:
                    raise FloatingPointError('More than ten AMP skipped updates in an epoch')
        training_seconds = time.perf_counter() - epoch_started
        validation, probabilities = evaluate(model, inputs, 'val', t['microbatch_size'], update)
        score = validation['balanced_accuracy']
        selected = score > best_score + t['min_delta']
        save_seconds = 0.0
        if selected:
            best_score, best_epoch = score, epoch
            update(phase='saving_best', best_epoch=best_epoch, best_validation_ba=best_score)
            save_start = time.perf_counter()
            checkpoint(directory / 'best.pt', model, job, protocol, epoch, score, validation, hashes)
            record(directory / 'best.json', {'epoch': epoch, 'validation': validation, 'timestamp': now()})
            save_predictions(directory / 'selected-val.npz', inputs, 'val', probabilities, epoch)
            save_seconds = time.perf_counter() - save_start
        total_skips += skipped
        row = {'epoch': epoch, 'timestamp': now(), 'training_seconds': training_seconds,
            'training_loss': loss_sum / seen if seen else None, 'optimizer_updates': steps,
            'mean_unclipped_gradient_norm': norm_sum / steps if steps else None,
            'amp_skipped_updates': skipped, 'loss_scale': scaler.get_scale(),
            'validation': validation, 'selected': selected, 'best_epoch': best_epoch,
            'best_validation_balanced_accuracy': best_score, 'checkpoint_seconds': save_seconds,
            'epoch_seconds': time.perf_counter() - epoch_started,
            'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
        with (directory / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        update(phase='epoch_complete', best_epoch=best_epoch, best_validation_ba=best_score,
               last_epoch_seconds=row['epoch_seconds'], last_training_loss=row['training_loss'])
        print(json.dumps({'job': job['id'], 'epoch': epoch, 'val_ba': score,
            'best_epoch': best_epoch, 'seconds': row['epoch_seconds'], 'amp_skips': skipped}), flush=True)
    # Release Adam buffers before loading the only retained model.
    del optimizer
    saved = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state'])
    assert saved['epoch'] == best_epoch
    del saved
    final = {}
    for split in ['val', 'train', 'test']:
        metrics, probs = evaluate(model, inputs, split, t['microbatch_size'], update)
        if split == 'val':
            stored = np.load(directory / 'selected-val.npz')['probabilities']
            if not np.allclose(probs, stored, atol=1e-6, rtol=1e-5):
                raise AssertionError('Reloaded checkpoint does not reproduce selected validation predictions')
        save_predictions(directory / f'selected-{split}.npz', inputs, split, probs, best_epoch)
        if split == 'test' and job['dataset'] == 'joint':
            for name in ['bcdata', 'shid_v2']:
                save_predictions(directory / f'selected-test-{name}.npz', inputs, split, probs, best_epoch, name)
        final[split] = metrics
    result = {'job': job, 'best_epoch': best_epoch, 'metrics': final,
        'elapsed_seconds': time.perf_counter() - start, 'gpu': gpu, 'completed_at': now(),
        'amp_skipped_updates': total_skips, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'trainable_encoder_parameters': sum(p.numel() for p in model.encoder.parameters()),
        'trainable_gain_parameters': model.connectome.gain.numel() if job.get('architecture', 'anatomical') == 'anatomical' else 0,
        'trainable_head_parameters': sum(p.numel() for p in model.head.parameters()) if job.get('architecture', 'anatomical') == 'linear' else 0,
        'checkpoint_sha256': sha256(directory / 'best.pt')}
    record(directory / 'complete.json', result)
    update(phase='complete')
    print(json.dumps({'completed': job['id'], 'test_ba': final['test']['balanced_accuracy']}), flush=True)
