"""Read-only DARG, cached E82.2 spectra, and checkpoint I/O.

Minimal historical loader/EMA glue adapted from E82.3.2 at source review.
The old experiment's Python modules are never imported.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ps_dso import FOLD2_ROWS, PROMPTS

HERE = Path(__file__).resolve().parent
RT, RESULTS = HERE / 'runtime', HERE / 'results'
REPO = Path('/data/adodas/SPromptGL-main')
HISTORY_ROOT = Path('/model/spromptgl/experiment')
E822 = HISTORY_ROOT / 'experiment82_2_psce'
E823 = HISTORY_ROOT / 'experiment82_3_cspe'
PAPER = E822 / 'runtime/spromptgl.pdf'
PROMPT_CACHE = E822 / 'runtime/prompts.pt'
TEXT_SOURCE = REPO / 'experiment/experiment60_dss_pen/.runtime/text_embeddings.pt'
ADAPTER = REPO / 'experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py'
CFG = json.loads((HERE / 'config.json').read_text())
BASE_GROUP = CFG['base_group']
HIST_OPTS = tuple(CFG['historical_optimizers'])
METRICS = ('acc', 'hard_auc', 'prob_auc', 'sen', 'spe', 'asd_f1')
START = CFG['start_epoch']
BASE = None


EPS = 1e-8


def legal_spectral_features(z_train, labels, spectrum):
    z_train = z_train.detach().cpu().double()
    labels = labels.detach().cpu().long()
    assert z_train.shape == (len(labels), 144) and set(labels.tolist()) == {0, 1}
    hh = ((z_train - spectrum['mu']) @ spectrum['U']) / (spectrum['lambda'] + EPS).sqrt()
    return hh, [hh[labels == label] for label in (0, 1)]


def stabilize_variance(variance, counts, config):
    pooled = (counts[:, None] * variance).sum(0) / counts.sum()
    floor = (config['relative_floor'] * pooled).clamp_min(config['variance_floor'])
    stable = config['own_variance'] * variance + config['pooled_variance'] * pooled + floor
    assert torch.isfinite(stable).all() and (stable > 0).all()
    return pooled, stable


def class_statistics(z_train, labels, spectrum, config):
    """Estimate once from legal training rows in fixed ASD/TD order."""
    hh, by_class = legal_spectral_features(z_train, labels, spectrum)
    counts = torch.tensor([len(rows) for rows in by_class], dtype=torch.long)
    assert counts.min() > config['variance_ddof']
    mean = torch.stack([rows.mean(0) for rows in by_class])
    variance = torch.stack([rows.var(0, correction=config['variance_ddof']) for rows in by_class])
    pooled, stable = stabilize_variance(variance, counts, config)
    return {'mean': mean, 'variance': variance, 'pooled': pooled, 'stable': stable, 'counts': counts}, hh


@torch.no_grad()
def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sha(value):
    """E82.2 content digest, independent of a tensor's current device."""
    digest = hashlib.sha256()

    def visit(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            digest.update(str((t.dtype, tuple(t.shape))).encode())
            digest.update(t.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            digest.update(str((x.dtype, x.shape)).encode())
            digest.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                digest.update(str(key).encode() + b'\0')
                visit(x[key])
        elif isinstance(x, (tuple, list)):
            digest.update(str(type(x)).encode())
            for item in x:
                visit(item)
        else:
            digest.update(json.dumps(x, sort_keys=True, default=str).encode() + b'\0')

    visit(value)
    return digest.hexdigest()


@contextmanager
def atomic_writer(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    kwargs = {} if 'b' in mode else {'encoding': 'utf-8', 'newline': ''}
    with temporary.open(mode, **kwargs) as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path, value):
    with atomic_writer(path, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def atomic_torch(path, value):
    with atomic_writer(path, 'wb') as stream:
        torch.save(value, stream)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def write_csv(path, rows):
    with atomic_writer(path, 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def update_manifest(**fields):
    path = HERE / 'manifest.json'
    manifest = read_json(path)
    manifest.update(fields)
    atomic_json(path, manifest)
    return manifest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime():
    global BASE
    if BASE is None:
        anchor = read_json(HERE / 'manifest.json')['protected_sha256']
        assert sha_file(ADAPTER) == anchor[str(ADAPTER)], 'Historical path adapter changed'
        adapter = load_module('ps_dso_readonly_path_adapter', ADAPTER)
        BASE = adapter.load_base_runner()
        base_runtime().runtime()
        torch.set_num_threads(CFG['gpu']['threads_per_worker'])
    return BASE


def base_runtime():
    return runtime().load_history().load_e69().load_e61().load_base()


def rng_sha():
    base = base_runtime()
    return base.rng_sha(base.pack_rng())


@contextmanager
def isolated_rng(seed=None):
    base = base_runtime()
    saved = base.pack_rng()
    before = base.rng_sha(saved)
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        base.restore_rng(saved)
        assert rng_sha() == before


def paths(group, fold):
    name = f'{group}_f{fold:02d}'
    return (RT / 'tasks' / f'{name}.json', RT / 'checkpoints' / f'{name}.best.pt',
            RT / 'checkpoints' / f'{name}.latest.pt', RT / 'logs' / f'{name}.jsonl')


def historical_task(fold):
    return read_json(runtime().EXP / '.runtime/tasks' / f'{BASE_GROUP}_f{fold:02d}.json')


def legal_train(state):
    train = np.asarray(state['train_np'], dtype=np.int64)
    return train[~np.isin(train, FOLD2_ROWS)] if state['fold'] == 2 else train


def init_optimizers(state, warmup):
    checks = {}
    names = {id(p): f'{key}.{n}' for key in ('model', 'cam') for n, p in state[key].named_parameters()}
    for key, (name, lr, wd) in CFG['historical_optimizers'].items():
        opt = state[key]
        saved = warmup['optimizer_' + key.removeprefix('opt_')]
        digest = sha(opt.state_dict())
        assert digest == sha(saved), f'{key}: epoch-10 state differs'
        assert type(opt).__name__ == name
        assert all(pg['lr'] == lr and pg['weight_decay'] == wd for pg in opt.param_groups)
        param_names = [[names[id(p)] for p in pg['params']] for pg in opt.param_groups]
        checks[key] = {'state_sha': digest, 'ordered_parameters_sha': sha(param_names),
                       'n_tensors': sum(map(len, param_names)), 'n_moment_slots': len(saved['state']),
                       'lr': lr, 'weight_decay': wd}
    return checks


def prepare(fold):
    state, warmup, identity = runtime().prepare(fold)
    state['fold'] = fold
    task = historical_task(fold)
    hist = runtime().load_history()
    assert task['status'] == 'COMPLETE' and task['identity'] == identity
    assert warmup['completed_epoch'] == START and warmup['seed'] == CFG['seed']
    assert hist.state_model_sha(state) == identity['model_cam_sha']
    inherited_opts = {key: state[key].state_dict() for key in HIST_OPTS}
    assert runtime().sha_object(inherited_opts) == identity['optimizer_sha']
    assert rng_sha() == warmup['rng_sha'] == identity['warmup_rng_sha']
    assert tuple(state['x'].shape) == (CFG['subjects'], 320)
    legal = legal_train(state)
    test = np.asarray(state['test_np'])
    assert np.array_equal(warmup['test_indices'], test) and np.array_equal(warmup['train_indices'], legal)
    assert np.intersect1d(state['train_np'], test).tolist() == (list(FOLD2_ROWS) if fold == 2 else [])
    pairs = state['train_pairs_np']
    assert pairs.shape == (4096, 2) and np.isin(pairs, state['train_np']).all()
    assert np.array_equal(pairs, warmup['train_pairs']) or fold == 2
    assert np.array_equal(state['train_same_np'], state['labels_np'][pairs[:, 0]] == state['labels_np'][pairs[:, 1]])
    state['legal_rows'] = torch.as_tensor(legal, device=state['x'].device, dtype=torch.long)
    return state, warmup, identity


def cache_path(fold):
    return E823 / 'runtime' / f'class_stats_f{fold:02d}.pt'


def spectrum_cache(fold):
    return torch_load(E822 / 'runtime' / f'spectrum_f{fold:02d}.pt')


def verify_text(text):
    assert text['texts'] == PROMPTS and text['rank'] == 8 and text['seed'] == 821991
    assert sha({k: v for k, v in text.items() if k != 'identity_sha'}) == text['identity_sha']
    source = torch_load(TEXT_SOURCE)
    assert source['text_map']['original'] == PROMPTS and sha_file(TEXT_SOURCE) == text['source_sha256']
    assert torch.equal(source['embeddings']['original'].float(), text['medical'])
    med, anon, p = (text[k].double().numpy() for k in ('medical', 'anonymous', 'P_text'))
    gram_delta = float(np.abs(med @ med.T - anon @ anon.T).max())
    ortho = float(np.abs(p.T @ p - np.eye(32)).max())
    assert gram_delta < 1e-3 and ortho < 1e-6
    query_delta = {}
    for key, vectors in [('q_medical', med), ('q_anonymous', anon)]:
        projected = vectors @ p
        assert np.allclose(projected @ p.T, vectors, atol=1e-5, rtol=0)
        q = projected / np.linalg.norm(projected, axis=1, keepdims=True)
        query_delta[key] = float(np.abs(q - text[key].numpy()).max())
        assert query_delta[key] < 1e-6
    q_gram_error = float((text['q_medical'] @ text['q_medical'].T - text['q_anonymous'] @ text['q_anonymous'].T).abs().max())
    assert q_gram_error < 1e-6 and not torch.equal(text['q_anonymous'], text['q_medical'])
    return {'raw_gram_max_abs': gram_delta, 'projection_orthogonality_max_abs': ortho,
            'query_projection_max_abs': query_delta, 'normalized_query_gram_max_abs': q_gram_error,
            'embedding_sha': text['identity_sha'], 'texts': PROMPTS, 'source_sha': text['source_sha256']}


def fixed_stats(state, identity, spectrum):
    fold = state['fold']
    indices = legal_train(state)
    labels = torch.as_tensor(state['labels_np'][indices])
    old_cfg = read_json(E823 / 'config.json')['class_statistics']
    scope = {'fold': fold, 'identity': identity, 'legal_train_sha': sha(indices), 'legal_label_sha': sha(labels),
             'warmup_cam_sha': spectrum['feature_sha'], 'spectrum_sha': spectrum['spectra_sha'],
             'estimator': old_cfg}
    payload = torch_load(cache_path(fold))
    assert payload['scope'] == scope and np.array_equal(payload['legal_train'], indices)
    stats, _ = class_statistics(spectrum['warmup_cam'][indices], labels, spectrum['spectrum'], old_cfg)
    assert sha(stats) == sha(payload['stats']) == payload['stats_sha']
    return payload


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def metrics_for(probability, labels):
    pred = probability.argmax(-1)
    asd, td = labels == 0, labels == 1
    tp, tn = int((pred[asd] == 0).sum()), int((pred[td] == 1).sum())
    sen, spe = tp / asd.sum(), tn / td.sum()
    return {'acc': float((pred == labels).mean()), 'hard_auc': float((sen + spe) / 2),
            'prob_auc': float(roc_auc_score(asd, probability[:, 0])), 'sen': float(sen), 'spe': float(spe),
            'asd_f1': float(2 * tp / (2 * tp + (asd.sum() - tp) + (td.sum() - tn)))}


@torch.no_grad()
def warmup_cam(state):
    captured = []
    handle = state['cam'].Outputlayer.register_forward_pre_hook(lambda _head, args: captured.append(args[0].cpu().clone()))
    try:
        out = runtime().forward(state, BASE_GROUP, training=False)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (871, 144)
    return captured[0], out


@contextmanager
def count_calls(state):
    counts = dict.fromkeys(('cam', 'output', 'gcn0', 'gcn1', 'gcn2', 'gcn3'), 0)
    handles = []
    for name, module in [('cam', state['cam']), ('output', state['cam'].Outputlayer),
                         *[(f'gcn{i}', m) for i, m in enumerate(state['model'].gcn)]]:
        handles.append(module.register_forward_hook(lambda _m, _a, _o, k=name: counts.__setitem__(k, counts[k] + 1)))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
    assert set(counts.values()) == {1}, counts


def best_pending(best_path):
    return best_path.with_name(best_path.name + '.pending')


def reconcile_best(best_path, expected_sha):
    pending = best_pending(best_path)
    if expected_sha is None:
        assert not best_path.exists()
    elif not best_path.exists() or sha_file(best_path) != expected_sha:
        assert pending.is_file() and sha_file(pending) == expected_sha, 'committed Best lacks matching checkpoint'
        os.replace(pending, best_path)
    pending.unlink(missing_ok=True)


def committed_log(log, latest):
    n = latest['log_bytes']
    log.seek(0)
    committed = log.read(n)
    digest = hashlib.sha256(committed)
    assert len(committed) == n and digest.hexdigest() == latest['log_sha'], 'committed epoch journal changed'
    assert len(committed.splitlines()) == latest['next_epoch'] - START
    log.seek(n)
    log.truncate()
    return digest


def commit_latest(log, digest, latest_path, best_path, pending, best_sha, payload):
    # Pending Best becomes committed only when Latest records its SHA. Recovery handles
    # interruption on either side of the Latest rename, using the committed journal prefix.
    log.flush()
    os.fsync(log.fileno())
    if pending is not None:
        atomic_torch(best_pending(best_path), pending)
        best_sha = sha_file(best_pending(best_path))
    payload.update(best_sha=best_sha, log_bytes=log.tell(), log_sha=digest.hexdigest())
    atomic_torch(latest_path, payload)
    if pending is not None:
        os.replace(best_pending(best_path), best_path)
    directory = os.open(latest_path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return best_sha
