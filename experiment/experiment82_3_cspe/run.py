#!/usr/bin/env python3
"""Historical inheritance, Preflight and formal CSPE training."""
from __future__ import annotations

import os
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True

import argparse
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import random
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cspe import (CSPE, EPS, FOLD2_ROWS, PROMPTS, QUERIES, RESIDUAL_LIMIT, SPECTRUM_BUFFERS,
                  TEMPERATURE, balanced_ce, class_statistics, log_likelihood_ratio,
                  select_prompt_evidence, smooth_limit)

HERE = Path(__file__).resolve().parent
RT, RESULTS = HERE / 'runtime', HERE / 'results'
REPO = Path('/data/adodas/SPromptGL-main')
E822 = Path('/model/spromptgl/experiment/experiment82_2_psce')
E82 = REPO / 'experiment/experiment82_pssec'
E61 = REPO / 'experiment/experiment61_cadr_postalign_vs_e2e'
PAPER = E822 / 'runtime/spromptgl.pdf'
PROMPT_CACHE = E822 / 'runtime/prompts.pt'
TEXT_SOURCE = REPO / 'experiment/experiment60_dss_pen/.runtime/text_embeddings.pt'
ADAPTER = REPO / 'experiment/experiment81_3_e2e_pairwise_relational_learning/path_adapter.py'
CFG = json.loads((HERE / 'config.json').read_text())
GROUPS = tuple(CFG['groups'])
FOLDS = range(1, CFG['folds'] + 1)
BASE_GROUP = CFG['base_group']
HIST_OPTS = ('opt_base', 'opt_cam', 'opt_added')
METRICS = ('acc', 'hard_auc', 'prob_auc', 'sen', 'spe', 'asd_f1')
LOSS_NAMES = ('cls', 'graph', 'pair', 'adv', 'preserve', 'darg_total', 'res', 'prompt', 'total')
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')
START, LIMIT = CFG['start_epoch'], CFG['max_epochs']
BASE = None
LOCAL_FILE_SPEC = importlib.util.spec_from_file_location


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
    spec = LOCAL_FILE_SPEC(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime():
    global BASE
    if BASE is None:
        adapter = load_module('e823_readonly_path_adapter', ADAPTER)
        BASE = adapter.load_base_runner()
        base_runtime().runtime()
        torch.set_num_threads(CFG['threads_per_worker'])
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
    state['cspe_legal'] = torch.as_tensor(legal, device=state['x'].device, dtype=torch.long)
    return state, warmup, identity


def implementation():
    return {name: sha_file(HERE / name) for name in ('cspe.py', 'run.py', 'audit.py', 'config.json')}


def sources():
    # Includes the history chain, source data, root modules, original B0 and B1 results,
    # and the read-only records consulted to verify prior experiments.
    prev = Path('/model/spromptgl/experiment/experiment82_2_2_rce_psce')
    frozen = read_json(prev / 'manifest.json')['source_maps']['protected']
    external = [Path(p) for p in frozen]
    external.extend([ADAPTER, TEXT_SOURCE, PAPER, prev / 'manifest.json', prev / 'runtime/environment.json'])
    for directory in (prev, E822):
        external.extend(directory.glob('*.py'))
        external.extend(directory / name for name in ('config.json', 'manifest.json', 'results/report.md',
                                                     'results/fold_metrics.csv', 'results/audit_status.json'))
    external.extend([REPO / 'graph/ABIDE_weighted-cosine_graph_.npz',
                     REPO / 'dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'])
    external.extend((REPO / 'CLIP').glob('*.py'))
    external.extend((REPO / 'CLIP/clip').glob('*.py'))
    external.extend([PROMPT_CACHE])
    for fold in FOLDS:
        external.extend([E822 / 'runtime' / f'spectrum_f{fold:02d}.pt',
                         E82 / 'runtime' / f'spectrum_f{fold:02d}.pt',
                         E61 / '.runtime/warmup' / f'fold{fold:02d}.pt',
                         runtime().EXP / '.runtime/checkpoints' / f'{BASE_GROUP}_f{fold:02d}.best.pt',
                         runtime().EXP / '.runtime/checkpoints' / f'{BASE_GROUP}_f{fold:02d}.latest.pt'])
        for group in GROUPS:
            for which in ('best', 'latest'):
                external.append(E822 / 'runtime/checkpoints' / f'{group}_f{fold:02d}.{which}.pt')
            external.append(E822 / 'runtime/tasks' / f'{group}_f{fold:02d}.json')
    spec = runtime().source_manifest()
    for path in sorted({p.resolve() for p in external}):
        if str(path) not in spec:
            spec[str(path)] = sha_file(path)
    for path, expected in frozen.items():
        assert spec[str(Path(path).resolve())] == expected, f'historical protection baseline mismatch: {path}'
    return spec


def cache_path(fold):
    return RT / f'class_stats_f{fold:02d}.pt'


def spectrum_cache(fold):
    return torch_load(E822 / 'runtime' / f'spectrum_f{fold:02d}.pt')


def verify_text(text):
    assert text['texts'] == PROMPTS and text['rank'] == 8 and text['seed'] == CFG['historical_anonymous_seed']
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
    scope = {'fold': fold, 'identity': identity, 'legal_train_sha': sha(indices), 'legal_label_sha': sha(labels),
             'warmup_cam_sha': spectrum['feature_sha'], 'spectrum_sha': spectrum['spectra_sha'],
             'estimator': CFG['class_statistics']}
    path = cache_path(fold)
    if path.exists():
        payload = torch_load(path)
        assert payload['scope'] == scope and sha(payload['stats']) == payload['stats_sha']
        return payload
    stats, _ = class_statistics(spectrum['warmup_cam'][indices], labels, spectrum['spectrum'], CFG['class_statistics'])
    payload = {'scope': scope, 'legal_train': indices, 'stats': stats, 'stats_sha': sha(stats)}
    atomic_torch(path, payload)
    return payload


def make_module(state, group, spectrum, cache, text):
    fold = state['fold']
    with isolated_rng(CFG['initialization_seed_base'] + fold):
        module = CSPE(spectrum['spectrum'], text, cache['stats'], group, fold,
                      CFG['class_statistics']['llr_bound']).to(state['x'].device)
    hp = CFG['psce_optimizer']
    opt = torch.optim.AdamW(module.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    return module, opt, sha(dict(module.named_parameters()))


def forward(state, module, training=False, intervention='full'):
    """Exactly one historical forward. The auxiliary pass touches only legal CAM rows."""
    assert not training or intervention in ('full', 'residual_off')
    module.train(training)
    details, auxiliary = {}, {}
    legal = state['cspe_legal']

    def replace_cam(_head, args):
        assert not details and len(args) == 2 and args[0].shape == (871, 144)
        before = rng_sha()
        z = args[0]
        changed, scores = module(z, intervention)
        details.update(scores)
        if training and intervention == 'full':
            auxiliary['scores'] = module.prompt_aux(z.detach().index_select(0, legal))
            auxiliary['loss'] = balanced_ce(auxiliary['scores'], state['y'][legal])
        assert rng_sha() == before, 'CSPE operation changed historical RNG'
        return changed, args[1]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(replace_cam)
    try:
        out = runtime().forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    assert details
    delta, z = details['delta'][legal], details['z'][legal]
    res = (delta.square().sum(-1) / (z.square().sum(-1) + EPS)).mean()
    out.update(darg_total=out['total'], res=res, details=details)
    if auxiliary:
        out.update(prompt=auxiliary['loss'], aux_scores=auxiliary['scores'])
        out['total'] = out['darg_total'] + CFG['residual_loss_coefficient'] * res + CFG['prompt_loss_coefficient'] * out['prompt']
    return out


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def bundle(state, module, opt):
    return {'base': runtime().load_history().state_bundle(state), 'cspe': cpu_state(module),
            'opt_psce': copy.deepcopy(opt.state_dict())}


def load_bundle(state, module, opt, saved):
    runtime().load_history().load_bundle(state, saved['base'])
    module.load_state_dict(saved['cspe'], strict=True)
    opt.load_state_dict(saved['opt_psce'])


def component_digests(value):
    base = value['base']
    return {**{key: sha(base[key]) for key in ('model', 'cam', *HIST_OPTS)},
            **{key: sha(value[key]) for key in ('cspe', 'opt_psce')}}


def optimizer_sequence(state, opt):
    return [state[key] for key in HIST_OPTS] + [opt]


def evaluate(state, module):
    with torch.no_grad():
        out = forward(state, module)
        validate_output(out)
        return out['logp'].exp()[state['test_np']].cpu().numpy()


def metrics_for(probability, labels):
    pred = probability.argmax(-1)
    asd, td = labels == 0, labels == 1
    tp, tn = int((pred[asd] == 0).sum()), int((pred[td] == 1).sum())
    sen, spe = tp / asd.sum(), tn / td.sum()
    return {'acc': float((pred == labels).mean()), 'hard_auc': float((sen + spe) / 2),
            'prob_auc': float(roc_auc_score(asd, probability[:, 0])), 'sen': float(sen), 'spe': float(spe),
            'asd_f1': float(2 * tp / (2 * tp + (asd.sum() - tp) + (td.sum() - tn)))}


def validate_output(out):
    details = out['details']
    tolerance = CFG['preflight']['row_sum_absolute_tolerance']
    checks = [torch.isfinite(value).all() for value in (out['total'], out['logp'], *details.values())]
    for key in ('w_n', 'w_a', 'w_sem', 'w_sp', 'w', 'pi'):
        w = details[key]
        checks.extend([(w >= 0).all(), ((w.sum(-1) - 1).abs() <= tolerance).all()])
    ratio = details['delta'].norm(dim=-1) / details['z'].norm(dim=-1).clamp_min(EPS)
    checks.append((ratio <= RESIDUAL_LIMIT + 1e-6).all())
    if not torch.stack(checks).all():
        raise FloatingPointError('CSPE nonfinite, invalid probability mass or residual bound failure')


def named_parameters(state, module):
    return [(f'{prefix}.{name}', value) for prefix, part in
            [('darg', state['model']), ('cam', state['cam']), ('cspe', module)]
            for name, value in part.named_parameters() if value.requires_grad]


def sizes(named, gradients):
    squares = {name: float(grad.detach().double().square().sum()) if grad is not None else 0.
               for (name, _), grad in zip(named, gradients)}
    groups = {'darg': 'darg.', 'cam': 'cam.', 'output': 'cam.Outputlayer.',
              'direction': 'cspe.direction.', 'evidence': 'cspe.evidence.', 'norm': 'cspe.norm.',
              'key': 'cspe.key.', 'null': 'cspe.null', 'spectral': 'cspe.spectral.'}
    by_module = {key: math.sqrt(sum(v for name, v in squares.items() if name.startswith(prefix)))
                 for key, prefix in groups.items()}
    by_module['prompt_parameters'] = math.hypot(*(by_module[k] for k in ('direction', 'evidence', 'norm', 'key', 'null')))
    return by_module, squares


def gradient_audit(state, module, out, *, enforce_gate=True):
    named = named_parameters(state, module)
    tensors = [param for _, param in named]
    assert all(not buffer.requires_grad for buffer in module.buffers())
    assert not out['details']['beta'].requires_grad
    measurements = {}
    for name, loss in {'classification': out['cls'], 'weighted_auxiliary': CFG['prompt_loss_coefficient'] * out['prompt'],
                       'weighted_residual': CFG['residual_loss_coefficient'] * out['res']}.items():
        grad = torch.autograd.grad(loss, tensors + [out['details']['z'], out['fused']],
                                   allow_unused=True, retain_graph=True)
        assert all(g is None or torch.isfinite(g).all() for g in grad), name
        norms, per_tensor_sq = sizes(named, grad[:-2])
        norms['cam144'], norms['fused320'] = [float(g.norm()) if g is not None else 0. for g in grad[-2:]]
        if name == 'weighted_auxiliary':
            assert all(v == 0 for k, v in per_tensor_sq.items() if not k.startswith('cspe.') or k.startswith('cspe.spectral.'))
            assert norms['cam144'] == norms['fused320'] == 0
            assert all(norms[k] > 0 for k in ('direction', 'evidence', 'norm', 'key', 'null')), norms
        if name == 'classification':
            assert all(v > 0 for v in norms.values()), norms
            assert all(v > 0 for k, v in per_tensor_sq.items() if k.startswith(('cspe.', 'cam.Outputlayer.')))
        measurements[name] = norms
    denom = measurements['classification']['prompt_parameters']
    ratios = {key: measurements[key]['prompt_parameters'] / denom for key in ('weighted_auxiliary', 'weighted_residual')}
    ceiling = CFG['preflight']['maximum_weighted_aux_or_residual_to_prompt_classification_grad_ratio']
    if enforce_gate and (not all(math.isfinite(x) for x in ratios.values()) or max(ratios.values()) > ceiling):
        raise FloatingPointError(f'Preflight gradient ratio above {ceiling}: {ratios}')
    return {'norms': measurements, 'relative_prompt_parameter_norms': ratios}


def train_step(state, module, opt, record_gradients=False):
    optimizers = optimizer_sequence(state, opt)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    out = forward(state, module, training=True)
    validate_output(out)
    out['total'].backward()
    named = named_parameters(state, module)
    grads = [param.grad for _, param in named]
    if not torch.stack([torch.isfinite(g).all() for g in grads if g is not None]).all():
        raise FloatingPointError('nonfinite formal training gradient')
    norms = sizes(named, grads)[0] if record_gradients else None
    for optimizer in optimizers:
        optimizer.step()
    losses = torch.stack([out[key].detach() for key in LOSS_NAMES]).cpu().tolist()
    return dict(zip(LOSS_NAMES, losses)), norms


def llr_diagnostics(raw, bounded, labels):
    raw = raw.detach().cpu().numpy()
    bounded = bounded.detach().cpu().numpy()
    target = np.asarray(labels) == 0
    assert np.isfinite(raw).all() and np.isfinite(bounded).all()
    assert np.abs(bounded).max() <= CFG['class_statistics']['llr_bound']
    return {
        'raw_auc_direction_mean_score': float(roc_auc_score(target, raw.mean(1))),
        'bounded_auc_direction_mean_score': float(roc_auc_score(target, bounded.mean(1))),
        'raw_auc_by_direction': roc_auc_score(target[:, None].repeat(16, axis=1), raw, average=None).tolist(),
        'bounded_auc_by_direction': roc_auc_score(target[:, None].repeat(16, axis=1), bounded, average=None).tolist(),
    }


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


def noop_regression(state, module, opt):
    """Compare both full updates with CSPE residual and auxiliary CE disabled."""
    hist = runtime().load_history()
    base = base_runtime()
    saved = hist.state_bundle(state)
    rng = base.pack_rng()
    results = []
    for bypass in (False, True):
        hist.load_bundle(state, saved)
        base.restore_rng(rng)
        hist.zero_optimizers(state)
        if bypass:
            out = forward(state, module, training=True, intervention='residual_off')
            assert 'prompt' not in out and torch.count_nonzero(out['details']['delta']) == 0
        else:
            out = runtime().forward(state, BASE_GROUP, training=True)
        out['total'].backward()
        grads = {key: {n: p.grad for n, p in state[key].named_parameters()} for key in ('model', 'cam')}
        grad_sha = sha(grads)
        for key in HIST_OPTS:
            state[key].step()
        after_train = rng_sha()
        with torch.no_grad():
            ev = (forward(state, module, intervention='residual_off') if bypass
                  else runtime().forward(state, BASE_GROUP, training=False))
        results.append({'training_logp_sha': sha(out['logp']), 'eval_logp_sha': sha(ev['logp']),
                        'loss_sha': sha({k: out[k] for k in ('cls', 'graph', 'pair', 'adv', 'preserve', 'total')}),
                        'gradients_sha': grad_sha, 'next_update_sha': sha(hist.state_bundle(state)),
                        'rng_after_train': after_train, 'rng_after_eval': rng_sha()})
    hist.load_bundle(state, saved)
    base.restore_rng(rng)
    hist.zero_optimizers(state)
    opt.zero_grad(set_to_none=True)
    assert results[0] == results[1], 'DARG no-op forward, losses, gradients, update or RNG mismatch'
    return results[0]


def environment():
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
            'gpu_driver_and_memory': subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip(),
            'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'networkx')},
            'threads_per_worker': torch.get_num_threads()}


def preflight():
    runtime()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert CFG['epsilon'] == EPS and CFG['temperature'] == TEMPERATURE and CFG['residual_limit'] == RESIDUAL_LIMIT
    assert CFG['prompt_zero_based'] == {'normal_like': list(range(4)), 'abnormal_like': list(range(4, 8))}
    assert CFG['labels'] == {'ASD': 0, 'TD': 1} and tuple(i - 1 for i in CFG['fold2_excluded_from_new_paths_rows_1based']) == FOLD2_ROWS
    old_config = read_json(E822 / 'config.json')
    for key in ('seed', 'folds', 'base_group', 'subjects', 'max_epochs', 'start_epoch', 'early_stop_wait_gt',
                'initialization_seed_base', 'spectral_dimensions', 'token_dimension', 'temperature', 'epsilon',
                'residual_limit', 'residual_loss_coefficient', 'historical_optimizers', 'psce_optimizer', 'save_latest_every_epochs'):
        assert old_config[key] == CFG[key], key
    code = implementation()
    protection = sources()
    update_manifest(status='PREFLIGHT_RUNNING', config=CFG, environment=environment(),
                    source_maps={'protected': protection, 'implementation': code},
                    source_resolution={
                        'fold2': 'The E74.4-G2 loader applies E69.3 Fold2 additions after loading E61, and regenerates the effective 4096 pairs from the locked historical rule and local RNG. Use those historical effective pairs and indices; CSPE uses the original disjoint E61 train rows.',
                        'gate': 'The DARG loader applies E61-E2 identity calibration, reliability=1 and frozen zero reliability logits, and then the historical E74.4-G2 adaptive gate start. All remaining model tensors and all three optimizer states must match E61.',
                        'prob_auc': 'Use ASD probability column 0 with ASD as positive. Historical reference files retain their original TD-column ROC-AUC, independently checked and also recomputed in ASD orientation.',
                        'variance': 'Unbiased n-1 as used to build E82.2 spectra, with the registered count-weighted pooling, shrinkage and floor.'})
    e822_psce = load_module('e823_pinned_psce_preflight', E822 / 'psce.py')
    previous = read_json(E822 / 'runtime/preflight.json')
    text = torch_load(PROMPT_CACHE)
    text_checks = verify_text(text)
    pre = {'status': 'RUNNING', 'implementation': code, 'text': text_checks, 'folds': []}
    atomic_json(RT / 'preflight.json', pre)
    for fold in FOLDS:
        print(f'PREFLIGHT fold={fold:02d} started', flush=True)
        started = time.monotonic()
        state, warm, identity = prepare(fold)
        initial_rng = rng_sha()
        optimizers = init_optimizers(state, warm)
        assert sha(cpu_state(state['cam'])) == sha(warm['cam'])
        raw_model, current_model = warm['model'], state['model'].state_dict()
        assert raw_model.keys() == current_model.keys()
        deliberate = [name for name in raw_model if not torch.equal(raw_model[name].cpu(), current_model[name].detach().cpu())]
        assert set(deliberate).issubset({'calibration', 'reliability', 'reliability_gate_logits', 'adaptive_gate_logits'}), deliberate
        assert torch.equal(current_model['calibration'].cpu(), torch.tensor([[1., 0.]] * 4))
        assert torch.equal(current_model['reliability'].cpu(), torch.ones(4))
        assert torch.count_nonzero(current_model['reliability_gate_logits']) == 0
        assert current_model['adaptive_gate_logits'].cpu().tolist() == identity['adaptive_gate_theta_initial']
        assert not state['model'].reliability_gate_logits.requires_grad
        legal = legal_train(state)
        labels = state['labels_np'][legal]
        historical = spectrum_cache(fold)
        z, z_out = warmup_cam(state)
        warm_pred = z_out['logp'].detach().cpu().argmax(-1).numpy()[legal]
        assert rng_sha() == initial_rng
        assert historical['identity'] == identity and np.array_equal(historical['legal_train'], legal)
        assert torch.equal(z, historical['warmup_cam']) and sha(z) == historical['feature_sha']
        assert sha(legal) == historical['legal_train_sha'] and sha(torch.as_tensor(labels)) == historical['legal_label_sha']
        assert [sha(v) for v in z_out['original_adjs']] == historical['warmup_graph_sha']
        assert [sha(v) for v in z_out['unified_adjs']] == historical['warmup_unified_graph_sha']
        del z_out
        rebuilt = e822_psce.build_spectra(z[legal], torch.as_tensor(labels))
        restored = historical['spectrum']
        old = torch_load(E82 / 'runtime' / f'spectrum_f{fold:02d}.pt')['spectra']['O']
        assert historical['spectra_sha'] == sha(restored) and historical['e82_original_U_sha'] == sha(old['U'])
        for key in SPECTRUM_BUFFERS.values():
            assert torch.equal(rebuilt[key], restored[key]), (fold, key)
        for key in ('U', 'mu', 'lambda'):
            assert torch.equal(restored[key], old[key]), (fold, key)
        cache = fixed_stats(state, identity, historical)
        assert np.array_equal(cache['legal_train'], legal)
        cached = cache['stats']
        hh = ((z[legal].double() - restored['mu']) @ restored['U']) / (restored['lambda'] + EPS).sqrt()
        raw, bounded = log_likelihood_ratio(hh, cached['mean'], cached['stable'], CFG['class_statistics']['llr_bound'])
        llr = llr_diagnostics(raw, bounded, labels)
        all_auc = [x for a in llr.values() for x in (a if isinstance(a, list) else [a])]
        if all(abs(value - .5) <= 1e-12 for value in all_auc):
            raise FloatingPointError(f'Fold {fold}: legal training LLR has no class discrimination')
        llr.update(raw_min=float(raw.min()), raw_max=float(raw.max()), bounded_min=float(bounded.min()),
                   bounded_max=float(bounded.max()), share_raw_abs_gt_8=float((raw.abs() > 8).double().mean()),
                   mean_abs_smoothing=float((raw - bounded).abs().mean()))
        initializer_sha = previous['inheritance'][fold - 1]['new_parameter_sha']
        reference_modules = []
        group_checks = {}
        for group in GROUPS:
            module, opt, init_sha = make_module(state, group, historical, cache, text)
            assert init_sha == initializer_sha and sum(p.numel() for p in module.parameters() if p.requires_grad) == 6081
            with isolated_rng(CFG['initialization_seed_base'] + fold):
                original = e822_psce.PSCE(restored, text, group, fold).to(state['x'].device)
            assert all(torch.equal(p, dict(original.named_parameters())[name]) for name, p in module.named_parameters())
            assert sha(opt.state_dict()['state']) == sha({})
            assert torch.equal(module.q0.cpu(), text[QUERIES[group]])
            reference_modules.append(cpu_state(module))
            with count_calls(state) as calls:
                out = forward(state, module, training=True)
            validate_output(out)
            d = out['details']
            assert d['a'].shape == (871, 8, 17) and d['tokens'].shape == (871, 16, 32)
            assert out['aux_scores'].shape == (len(legal), 2)
            with torch.no_grad():
                _, original_detail = original(d['z'].detach())
                assert all(torch.equal(d[key], original_detail[key]) for key in ('h', 'hhat', 'a', 'tokens', 'w_sp'))
                torch.testing.assert_close(out['aux_scores'], d['scores'][legal], atol=2e-6, rtol=2e-6)
                a = d['a'].detach()
                r = d['bounded_llr'].detach()
                evidence = select_prompt_evidence(a, r)
                permuted = select_prompt_evidence(a[:, [3, 1, 0, 2, 6, 4, 7, 5]], r)
                for actual, expected in zip(permuted, evidence):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
                swapped = select_prompt_evidence(a[:, [4, 5, 6, 7, 0, 1, 2, 3]], -r)
                for actual, expected in zip(swapped, [evidence[i] for i in (1, 0, 3, 2)] +
                                            [evidence[4].flip(-1), evidence[5].flip(-1), evidence[6]]):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
                if fold == 2:
                    assert torch.count_nonzero(d['delta'][list(FOLD2_ROWS)]) == 0
                    assert not np.isin(FOLD2_ROWS, legal).any()
            ga = gradient_audit(state, module, out)
            relative = d['delta'].norm(dim=-1) / d['z'].norm(dim=-1).clamp_min(EPS)
            group_checks[group] = {'new_parameter_sha': init_sha, 'trainable_parameters': 6081, 'calls': calls,
                                   'main_attention_and_w_sp_exact_vs_e822': True, 'gradients': ga,
                                   'max_relative_delta': float(relative.detach().max()),
                                   'initial_prompt_ce': float(out['prompt'].detach()),
                                   'rng_isolation_and_group_symmetry': True}
            del out, d, original_detail, original
            base_runtime().restore_rng(warm['rng'])
        assert reference_modules[0].keys() == reference_modules[1].keys()
        assert [k for k, v in reference_modules[0].items() if not torch.equal(v, reference_modules[1][k])] == ['q0']
        no_op = noop_regression(state, module, opt)
        first_b0 = historical_task(fold)['rng_trace'][0]
        assert no_op['rng_after_train'] == first_b0['rng_after_train'] and no_op['rng_after_eval'] == first_b0['rng_after_eval']
        with torch.no_grad():
            logits = forward(state, module)['logp']
            idx = state['test_np']
            truth = state['y'][idx].clone()
            state['y'][idx] = 1 - truth
            try:
                assert torch.equal(logits, forward(state, module)['logp'])
            finally:
                state['y'][idx] = truth
        assert rng_sha() == initial_rng
        record = {'fold': fold, 'identity': identity, 'groups': group_checks, 'warmup_cam_and_graphs_exact': True,
                  'optimizer_identity': optimizers, 'historic_pairs': len(state['train_pairs_np']),
                  'history_declared_e61_to_darg_tensor_differences': deliberate,
                  'warmup_pairs_sha': sha(warm['train_pairs']), 'effective_pairs_sha': sha(state['train_pairs_np']),
                  'effective_train_index_sha': sha(state['train_np']), 'test_index_sha': sha(state['test_np']),
                  'legal_train_sha': sha(legal), 'legal_counts_asd_td': cached['counts'].tolist(),
                  'legal_train_wrong_at_darg_warmup_eval': int((warm_pred != labels).sum()),
                  'e822_parameter_init_sha': initializer_sha, 'stats_sha': cache['stats_sha'],
                  'spectrum_tensors_sha': {key: sha(restored[key]) for key in SPECTRUM_BUFFERS.values()},
                  'llr': llr, 'no_op_full_darg': no_op, 'test_label_swap_bitwise': True,
                  'elapsed_seconds': time.monotonic() - started}
        pre['folds'].append(record)
        atomic_json(RT / 'preflight.json', pre)
        print(f'PREFLIGHT fold={fold:02d} PASS; LLR train AUC {llr["bounded_auc_direction_mean_score"]:.4f}', flush=True)
        del state, module, opt, warm
    # Verify the original limiter also at exact zero and very small and large raw updates.
    raw = torch.tensor([0., 1e-10, .01, 1e3], dtype=torch.float64).unsqueeze(1).repeat(1, 144).requires_grad_()
    z = torch.arange(144, dtype=torch.float64)[None].expand_as(raw) / 144
    bounded = smooth_limit(raw, z)
    derivative = torch.autograd.grad(bounded.sum(), raw)[0]
    assert (bounded.norm(dim=-1) <= RESIDUAL_LIMIT * z.norm(dim=-1) + 1e-12).all()
    assert torch.isfinite(derivative).all() and torch.equal(derivative[0], torch.ones_like(derivative[0]))
    assert code == implementation()
    pre.update(status='PASS', peak_gpu_bytes=torch.cuda.max_memory_allocated(), limiter_zero_gradient_identity=True,
               class_cache_files={cache_path(f).name: sha_file(cache_path(f)) for f in FOLDS})
    atomic_json(RT / 'preflight.json', pre)
    update_manifest(status='PREFLIGHT_PASS', preflight_sha=sha_file(RT / 'preflight.json'),
                    protection_count=len(protection), protection_map_sha=sha(protection),
                    preflight_passed_folds=len(pre['folds']))
    print('PREFLIGHT PASS 10/10', flush=True)


def passing_preflight():
    manifest = read_json(HERE / 'manifest.json')
    path = RT / 'preflight.json'
    pre = read_json(path)
    assert pre['status'] == 'PASS' and len(pre['folds']) == CFG['folds']
    assert manifest['preflight_sha'] == sha_file(path) and pre['implementation'] == implementation()
    return pre


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


def checkpoint_state(state, module, opt):
    base = base_runtime()
    current = bundle(state, module, opt)
    rng = base.pack_rng()
    return {'current': current, 'digests': component_digests(current), 'rng': rng, 'rng_sha': base.rng_sha(rng)}


def task_complete(group, fold):
    task, best, latest, log = paths(group, fold)
    if not task.exists():
        return False
    record = read_json(task)
    assert record['status'] == 'COMPLETE' and record['implementation'] == implementation()
    assert all(record[key] == sha_file(path) for key, path in
               [('best_sha', best), ('latest_sha', latest), ('log_sha', log)]), f'completed task corrupt: {group}/{fold}'
    return True


def train_worker(group, fold, pre):
    if task_complete(group, fold):
        return
    start_time = time.monotonic()
    state, warmup, identity = prepare(fold)
    task, best_path, latest_path, log_path = paths(group, fold)
    spectrum, text = spectrum_cache(fold), torch_load(PROMPT_CACHE)
    cachefile = cache_path(fold)
    cache = torch_load(cachefile)
    checks = pre['folds'][fold - 1]
    assert sha_file(cachefile) == pre['class_cache_files'][cachefile.name]
    assert cache['scope']['identity'] == spectrum['identity'] == identity
    assert np.array_equal(cache['legal_train'], legal_train(state))
    module, opt, init_sha = make_module(state, group, spectrum, cache, text)
    assert init_sha == checks['e822_parameter_init_sha'] and rng_sha() == warmup['rng_sha']
    metadata = {'group': group, 'fold': fold, 'seed': CFG['seed'], 'identity': identity,
                'implementation': pre['implementation'], 'initial_parameter_sha': init_sha,
                'stats_sha': cache['stats_sha'], 'spectrum_sha': spectrum['spectra_sha'],
                'query_sha': sha(module.q0), 'legal_train_sha': cache['scope']['legal_train_sha']}
    historical = {row['epoch']: row for row in historical_task(fold)['rng_trace']}
    base = base_runtime()
    labels = state['labels_np'][state['test_np']]
    torch.cuda.reset_peak_memory_stats()
    with log_path.open('a+b') as log:
        if latest_path.exists():
            latest = torch_load(latest_path)
            assert all(latest[key] == value for key, value in metadata.items()), 'Latest provenance does not match this task'
            reconcile_best(best_path, latest['best_sha'])
            digest = committed_log(log, latest)
            load_bundle(state, module, opt, latest['current'])
            base.restore_rng(latest['rng'])
            assert component_digests(bundle(state, module, opt)) == latest['digests'] and rng_sha() == latest['rng_sha']
            epoch0, wait, best_correct, best_epoch, best_sha, probability = (latest[key] for key in
                ('next_epoch', 'wait', 'best_correct', 'best_epoch', 'best_sha', 'probability'))
            assert START <= epoch0 <= LIMIT
            resumes = latest['resumes'] + [epoch0]
            print(f'{group} fold={fold:02d} RESUME exact Latest epoch={epoch0}', flush=True)
            del latest
        else:
            assert not best_path.exists() and log.tell() == 0
            epoch0, wait, best_correct, best_epoch, best_sha, probability, resumes = START, 0, -1, None, None, None, []
            digest = hashlib.sha256()
            commit_latest(log, digest, latest_path, best_path, None, None,
                          {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': START, 'wait': 0,
                           'best_correct': best_correct, 'best_epoch': None, 'probability': None, 'resumes': []})
        executed = epoch0
        pending = None
        for epoch in range(epoch0 + 1, LIMIT + 1):
            if wait > CFG['early_stop_wait_gt']:
                break
            before = rng_sha()
            sample = epoch == START + 1 or epoch % CFG['save_latest_every_epochs'] == 0
            losses, gradients = train_step(state, module, opt, record_gradients=sample)
            after_train = rng_sha()
            probability = evaluate(state, module)
            after_eval = rng_sha()
            trace = dict(zip(RNG_KEYS, (epoch, before, after_train, after_eval)))
            if epoch in historical:
                assert trace == historical[epoch], f'historical RNG mismatch {group}/{fold}/{epoch}'
            assert after_train == after_eval, f'eval RNG changed {group}/{fold}/{epoch}'
            prediction = probability.argmax(-1)
            correct = int((prediction == labels).sum())
            improved = correct > best_correct
            if improved:
                best_correct, best_epoch, wait = correct, epoch, 0
                pending = {**metadata, **checkpoint_state(state, module, opt), 'best_epoch': epoch,
                           'metrics': metrics_for(probability, labels), 'prediction': prediction,
                           'probability': probability, 'test_indices': state['test_np'].copy()}
            else:
                wait += 1
            row = {**trace, 'correct': correct, 'n_test': len(labels), 'improved': improved, 'wait': wait,
                   'best_epoch': best_epoch, 'losses': losses, 'prediction_bits': np.packbits(prediction).tobytes().hex()}
            if gradients is not None:
                row['total_gradient_norms'] = gradients
            encoded = (json.dumps(row, allow_nan=False) + '\n').encode()
            log.write(encoded)
            digest.update(encoded)
            executed = epoch
            if epoch % CFG['save_latest_every_epochs'] == 0 or epoch == LIMIT or wait > CFG['early_stop_wait_gt']:
                best_sha = commit_latest(log, digest, latest_path, best_path, pending, best_sha,
                                         {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': epoch,
                                          'wait': wait, 'best_correct': best_correct, 'best_epoch': best_epoch,
                                          'probability': probability, 'resumes': resumes})
                pending = None
                print(f'{group} fold={fold:02d} epoch={epoch} wait={wait}', flush=True)
    assert best_epoch is not None and (executed == LIMIT or wait > CFG['early_stop_wait_gt'])
    assert sha_file(best_path) == best_sha and sha_file(log_path) == digest.hexdigest()
    # Make a live next-step witness. Only the in-memory terminal object is updated;
    # the formal Best, Latest and epoch journal are already committed.
    terminal = checkpoint_state(state, module, opt)
    losses, _ = train_step(state, module, opt)
    after_train = rng_sha()
    next_probability = evaluate(state, module)
    next_state = checkpoint_state(state, module, opt)
    witness = {'after_train_rng': after_train, 'after_eval_rng': next_state['rng_sha'],
               'components_after_update': next_state['digests'], 'probability_sha': sha(next_probability),
               'losses': losses, 'live_terminal_object': executed > epoch0}
    atomic_json(task, {**metadata, 'status': 'COMPLETE', 'epochs_executed': executed,
                       'best_epoch': best_epoch, 'best_sha': best_sha, 'latest_sha': sha_file(latest_path),
                       'log_sha': digest.hexdigest(), 'terminal_digests': terminal['digests'],
                       'terminal_rng_sha': terminal['rng_sha'], 'terminal_probability_sha': sha(probability),
                       'next_step_witness': witness, 'resumes': resumes,
                       'wall_seconds': time.monotonic() - start_time,
                       'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
    print(f'COMPLETE {group} fold={fold:02d}', flush=True)


def run_all(requested_workers):
    assert 1 <= requested_workers <= CFG['workers_max']
    passing_preflight()
    for directory in (RT / 'tasks', RT / 'logs', RT / 'checkpoints'):
        directory.mkdir(parents=True, exist_ok=True)
    with (RT / 'controller.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manage_workers(requested_workers)


def manage_workers(limit):
    finished = {(g, f) for f in FOLDS for g in GROUPS if task_complete(g, f)}
    pending = [(f, 0) for f in FOLDS if any((g, f) not in finished for g in GROUPS)]
    active, failures = {}, []
    start = time.monotonic()
    max_workers = peak = 0
    update_manifest(status='FORMAL_TRAINING', formal_launch_time=time.time(), completed_tasks=len(finished))
    try:
        while pending or active:
            while pending and len(active) < limit:
                fold, attempt = pending.pop(0)
                logfile = RT / 'logs' / f'fold{fold:02d}.stdout.log'
                stream = logfile.open('a')
                argv = [sys.executable, '-B', str(HERE / 'run.py'), 'worker', '--fold', str(fold)]
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE)
                active[fold] = (process, stream, attempt, logfile)
                max_workers = max(max_workers, len(active))
            for fold, (process, stream, attempt, logfile) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[fold]
                finished.update((g, fold) for g in GROUPS if (g, fold) in finished or task_complete(g, fold))
                if code == 0:
                    assert all((g, fold) in finished for g in GROUPS)
                    print(f'FINISHED fold={fold:02d} ({len(finished)}/{CFG["expected_tasks"]} tasks)', flush=True)
                else:
                    tail = logfile.read_text()[-3500:]
                    failures.append({'fold': fold, 'attempt': attempt + 1, 'exit_code': code, 'log_tail': tail})
                    if (code == 75 or code < 0) and attempt < 2:
                        if 'out of memory' in tail.lower():
                            limit = max(1, limit - 1)
                        pending.insert(0, (fold, attempt + 1))
                        print(f'Retry fold={fold:02d} from committed Latest where present', flush=True)
                    else:
                        raise RuntimeError(f'formal fold {fold} stopped with exit={code}: {tail[-1200:]}')
            # Count tasks from durable worker receipts without rehashing checkpoints every poll.
            available = sum(paths(g, f)[0].exists() for f in FOLDS for g in GROUPS)
            peak = max(peak, sampled_device_memory())
            atomic_json(RT / 'execution.json', {'status': 'FORMAL_TRAINING', 'completed_tasks': available,
                'expected_tasks': CFG['expected_tasks'], 'active_workers': len(active), 'pending_folds': len(pending),
                'max_workers': max_workers, 'sampled_peak_device_mib': peak, 'failures': failures,
                'elapsed_seconds': time.monotonic() - start})
            if active:
                time.sleep(2)
    finally:
        for process, _, _, _ in active.values():
            process.terminate()
        for process, stream, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        status = 'TRAINED_PENDING_INDEPENDENT_AUDIT' if len(finished) == CFG['expected_tasks'] else 'TRAINING_STOPPED'
        completed = sum(paths(g, f)[0].exists() for f in FOLDS for g in GROUPS)
        execution = {'status': status, 'completed_tasks': completed, 'expected_tasks': CFG['expected_tasks'],
                     'max_workers': max_workers, 'sampled_peak_device_mib': peak,
                     'wall_seconds': time.monotonic() - start, 'failures': failures}
        atomic_json(RT / 'execution.json', execution)
        update_manifest(status=status, completed_tasks=completed, execution=execution)
    assert len(finished) == CFG['expected_tasks']


def sampled_device_memory():
    try:
        value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        text=True, timeout=4).strip()
        return int(value)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'all', 'worker'])
    parser.add_argument('--workers', type=int, default=CFG['workers_max'])
    parser.add_argument('--fold', type=int, choices=FOLDS)
    args = parser.parse_args()
    try:
        if args.action == 'preflight':
            preflight()
        elif args.action == 'all':
            run_all(args.workers)
        else:
            assert args.fold is not None
            pre = passing_preflight()
            for group in GROUPS:
                train_worker(group, args.fold, pre)
    except Exception as exc:
        report = {'type': type(exc).__name__, 'reason': str(exc), 'traceback': traceback.format_exc()}
        if args.action == 'preflight':
            path = RT / 'preflight.json'
            previous = read_json(path) if path.exists() else {}
            atomic_json(path, {**previous, 'status': 'FAIL', 'failure': report})
            update_manifest(status='PREFLIGHT_BLOCKED', completed_tasks=0, blocking_failure=report)
        traceback.print_exc()
        if args.action == 'worker' and (isinstance(exc, (OSError, torch.OutOfMemoryError)) or 'out of memory' in str(exc).lower()):
            return 75
        return 65
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
