#!/usr/bin/env python3
"""GPU0-only PS-DSO training and transactional checkpoint recovery."""
from __future__ import annotations

import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
if __name__ == '__main__':
    sys.modules['run'] = sys.modules[__name__]

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import pickle
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / 'config.json').read_text())
sys.path.insert(0, cfg['environment']['readonly_numeric_packages'])

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import history as h
from history import BASE_GROUP, CFG, HIST_OPTS, METRICS, RESULTS, RT, START, sha, sha_file
from ps_dso import (EMA_KEYS, FOLD2_ROWS, PSDSO, balanced_mean, difficulty_matrix, gaussian_separation,
                    gradient_coverage, head_jacobian, observed_moments, true_margin)

GROUPS = tuple(CFG['groups'])
FOLDS = tuple(range(1, CFG['folds'] + 1))
LIMIT = CFG['max_epochs']
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')
LOSS_KEYS = ('cls', 'graph', 'pair', 'adv', 'preserve', 'total')
AUX_KEYS = ('spectral', 'semantic', 'protect', 'smooth')


def implementation():
    names = ('config.json', 'design_review.md', 'ps_dso.py', 'history.py', 'run.py')
    return {name: sha_file(HERE / name) for name in names}


def gpu_sample():
    query = 'index,uuid,name,memory.total,memory.used,utilization.gpu'
    raw = subprocess.check_output(['nvidia-smi', '-i', '0', f'--query-gpu={query}',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, name, total, used, utilization = (item.strip() for item in raw.split(','))
    assert int(index) == CFG['gpu']['physical_index'] and uuid == CFG['gpu']['uuid']
    return dict(index=int(index), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used),
                utilization_pct=int(utilization))


def environment():
    gpu = gpu_sample()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0' and torch.cuda.get_device_name(0) == gpu['name']
    assert torch.__version__.split('+')[0] == CFG['environment']['torch']
    assert torch.version.cuda == CFG['environment']['cuda']
    packages = {key: importlib.metadata.version(key) for key in CFG['environment']['packages']}
    assert packages == CFG['environment']['packages'], packages
    h.runtime()
    assert torch.cuda.current_device() == 0
    assert torch.get_num_threads() == CFG['gpu']['threads_per_worker']
    return dict(gpu=gpu, packages=packages, cuda=torch.version.cuda, torch=torch.__version__,
                visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], python=sys.version)


def protection():
    files = h.read_json(HERE / 'manifest.json')['protected_sha256']

    def check(item):
        path, expected = item
        try:
            actual = sha_file(path)
        except OSError as exc:
            return {'path': path, 'exception': str(exc)}
        return None if actual == expected else {'path': path, 'expected': expected, 'actual': actual}

    with ThreadPoolExecutor(max_workers=8) as pool:
        changed = [value for value in pool.map(check, files.items()) if value is not None]
    return dict(count=len(files), pass_=not changed, differences=changed)


def streams(saved):
    return dict(python=hashlib.sha256(pickle.dumps(saved['python'], protocol=4)).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(saved['numpy'], protocol=4)).hexdigest(),
                cpu=sha(saved['cpu']), cuda=[sha(raw) for raw in saved['cuda']])


def rng_mark():
    base = h.base_runtime()
    packed = base.pack_rng()
    return base.rng_sha(packed), streams(packed)


def init_module(state, spectrum, text):
    fold = state['fold']
    legal = h.legal_train(state)
    cam = spectrum['warmup_cam'][legal]
    obs = observed_moments(cam, torch.as_tensor(state['labels_np'][legal]))
    with h.isolated_rng(CFG['seeds']['prompt'] + fold):
        module = PSDSO(spectrum['spectrum']['U'], text['q_medical'], obs, fold, CFG)
    with h.isolated_rng(CFG['seeds']['fusion'] + fold):
        module.initialize_fusion()
    with h.isolated_rng(CFG['seeds']['gate'] + fold):
        module.initialize_gate()
    module = module.to(state['x'].device)
    module.calibrate(cam.to(state['x'].device))
    optimizer = torch.optim.AdamW(module.parameters(), **CFG['new_optimizer'])
    return module, optimizer


def parameter_layout(state, module, optimizer):
    head_ids = {id(p) for p in state['cam'].Outputlayer.parameters()}
    old, new = {}, module.parameter_groups()
    owned = []
    for key in HIST_OPTS:
        for i, pg in enumerate(state[key].param_groups):
            owned.extend(pg['params'])
            trainable = tuple(p for p in pg['params'] if p.requires_grad)
            if key == 'opt_cam':
                old[f'{key}.{i}.representation'] = tuple(p for p in trainable if id(p) not in head_ids)
                old[f'{key}.{i}.classifier'] = tuple(p for p in trainable if id(p) in head_ids)
            else:
                old[f'{key}.{i}'] = trainable
    added = [p for group in optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in (*owned, *added)}) == len(owned) + len(added)
    assert {id(p) for ps in old.values() for p in ps} == {
        id(p) for key in ('model', 'cam') for p in state[key].parameters() if p.requires_grad}
    assert {id(p) for p in added} == {id(p) for ps in new.values() for p in ps}
    assert all(old.values()) and all(new.values())
    return old, {key: tuple(value) for key, value in new.items()}


def flatten(parameters, grads=None):
    values = parameters if grads is None else [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]
    return torch.cat([v.detach().flatten() for v in values])


def setup(fold, group):
    state, warmup, identity = h.prepare(fold)
    spectrum, text = h.spectrum_cache(fold), h.torch_load(h.PROMPT_CACHE)
    assert spectrum['identity'] == identity
    assert sha(spectrum['spectrum']) == spectrum['spectra_sha']
    module, optimizer = init_module(state, spectrum, text)
    old, new = parameter_layout(state, module, optimizer)
    state.update(old_layout=old, new_layout=new, label_legal=state['y'].index_select(0, state['legal_rows']),
                 origin={key: flatten(ps).double() for key, ps in old.items()})
    assert rng_mark()[0] == warmup['rng_sha']
    meta = dict(group=group, fold=fold, identity=identity, initial_module_sha=sha(h.cpu_state(module)),
                initial_ema_sha=sha(module.ema_state()), spectrum_sha=spectrum['spectra_sha'],
                query_sha=sha(module.q0), legal_train_sha=sha(h.legal_train(state)), implementation=implementation())
    return state, module, optimizer, spectrum, warmup, meta


def historical_forward(state, training, module=None, intervention='full'):
    """One original forward. A hook changes only the input to the original head."""
    head = state['cam'].Outputlayer
    capture, dropouts = {}, []
    dropout = head.mlp_head[-1]
    assert isinstance(dropout, nn.Dropout)

    def pre(current_head, args):
        assert 'z' not in capture and args[0].shape == (CFG['subjects'], 144)
        z = args[0]
        capture['z'] = z
        if module is not None:
            capture['details'] = module(z, head_jacobian(current_head, z), intervention)
            return (capture['details']['enhanced'], *args[1:])

    def head_dropout(_drop, args, value):
        capture['dropout'] = args[0], value

    def raw_logits(_affine, _args, value):
        capture['logits'] = value

    def log_dropout(current, args, _value, name):
        dropouts.append((name, list(args[0].shape), current.training, current.p))

    handles = [head.register_forward_pre_hook(pre), dropout.register_forward_hook(head_dropout),
               head.classifier.register_forward_hook(raw_logits)]
    for which in ('model', 'cam'):
        for name, candidate in state[which].named_modules():
            if isinstance(candidate, nn.Dropout):
                handles.append(candidate.register_forward_hook(
                    lambda current, args, value, key=f'{which}.{name}': log_dropout(current, args, value, key)))
    try:
        with torch.set_grad_enabled(training):
            out = h.runtime().forward(state, BASE_GROUP, training=training)
    finally:
        for handle in handles:
            handle.remove()
    assert len(dropouts) > 0 and 'dropout' in capture and 'logits' in capture
    assert sum(name == 'cam.Outputlayer.mlp_head.3' for name, *_ in dropouts) == 1
    return out, capture, dropouts


def mask_from_native(capture):
    x, y = capture['dropout']
    mask, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=False)
    return mask.detach()


def off_scores(head, features, mask=None):
    norm, affine, activation, _ = head.mlp_head
    hidden = activation(affine(norm(features)))
    if mask is not None:
        hidden = hidden * mask
    return head.classifier(hidden)


def total_for_off(state, out, logp):
    # The additional evaluation estimates the reference gradient. It is never
    # added as a second cross-entropy to the actual optimization objective.
    history_loss = h.base_runtime().runtime()['utils'].ClsLoss(logp, state['y'], state['train'], state['weights'])
    return history_loss + out['graph'] + 0.05*out['pair'] + 0.01*out['adv'] + 0.01*out['preserve']


def auxiliary_terms(state, module, group, out, d, off):
    rows, y = state['legal_rows'], state['label_legal']
    z = d['z']
    basis = d['basis']
    off_legal = off.index_select(0, rows)
    m_off = true_margin(off_legal, y).detach()
    if group == 'G1':
        obs = observed_moments(z.index_select(0, rows), y)
        score, mean_part, cov_part = gaussian_separation(basis, obs, CFG)
        spectral = -score / (1 + score)
        details = {'bhattacharyya': score, 'mean': mean_part, 'covariance': cov_part}
    else:
        legal_only = F.nll_loss(off_legal, y, reduction='sum')
        gradients, = torch.autograd.grad(legal_only, z, retain_graph=True, create_graph=False)
        gradients = gradients.index_select(0, rows).detach()
        covariance = difficulty_matrix(gradients, m_off, y, CFG)
        score = gradient_coverage(basis, covariance)
        spectral = -score
        details = {'coverage': score, 'total_gradient_energy': covariance.trace()}
    hp = CFG['loss']
    anchor = (1 - (module.U0.T @ basis).square().sum() / basis.shape[1]).clamp_min(0)
    spectral = spectral + CFG['stiefel']['anchor_weight'] * anchor
    signed_score = d['first_order_score'].index_select(0, rows) * (1 - 2*y)
    semantic = balanced_mean(F.softplus(-signed_score), y)
    on = true_margin(out['logp'].index_select(0, rows), y)
    easy = (m_off - hp['protection_margin_center']).sigmoid()
    protect = (easy * (m_off - on).relu().square()).mean()
    rel = d['delta'].index_select(0, rows).double().square().sum(1) / (
        z.index_select(0, rows).detach().double().square().sum(1) + hp['norm_floor'])
    smooth = rel.mean()
    return (spectral, semantic, protect, smooth), details


def flat_norm(values):
    return float(values.double().norm())


def gradient_cos(left, right):
    denom = left.double().norm()*right.double().norm()
    return float(left.double().dot(right.double())/denom) if denom.item() else None


@torch.no_grad()
def coordinate_shared(base, proposed):
    """Project the enhanced-only difference, then certify the represented update."""
    if not torch.isfinite(base).all() or not torch.isfinite(proposed).all():
        raise FloatingPointError('Nonfinite shared gradient')
    b = base.double()
    raw = proposed.double() - b
    bb, dot, nr = map(float, (b.dot(b), b.dot(raw), raw.norm()))
    floor = CFG['loss']['norm_floor']**2
    nb = math.sqrt(bb)
    budget = CFG['loss']['shared_ratio_max'] * nb
    if bb <= floor:
        return base, dict(base=nb, enhanced=flat_norm(proposed), raw=nr, cosine=None, conflict=False,
                          limited=False, applied=0., ratio=0., dot=0., tiny_base=True, cast_fallback=False)
    conflict = dot < 0
    increment = raw - (dot / bb) * b if conflict else raw
    norm = flat_norm(increment)
    limited = norm > budget
    if limited:
        increment = increment * ((1 - 1e-4) * budget / norm)
    actual = (b + increment).to(base.dtype)
    difference = actual.double() - b
    after_dot, after_norm = float(b.dot(difference)), flat_norm(difference)
    if after_dot < 0:
        repair = max(-after_dot/bb, 0.) + 4*torch.finfo(base.dtype).eps
        actual = (b + increment*(1-1e-4) + repair*b).to(base.dtype)
        difference = actual.double() - b
        after_dot, after_norm = float(b.dot(difference)), flat_norm(difference)
    fallback = after_dot < 0 or after_norm > budget
    if fallback:
        actual, after_dot, after_norm = base, 0., 0.
    return actual, dict(base=nb, enhanced=flat_norm(proposed), raw=nr, cosine=dot/(nb*nr) if nr else None,
                        conflict=conflict, limited=limited, applied=after_norm, ratio=after_norm/nb, dot=after_dot,
                        tiny_base=False, cast_fallback=fallback)


@torch.no_grad()
def coordinate_added(classification, grad_aux):
    reference = classification.double()
    norm = flat_norm(reference)
    aux = sum((CFG['loss'][name]*value.double() for name, value in zip(AUX_KEYS, grad_aux)), start=torch.zeros_like(reference))
    raw = flat_norm(aux)
    budget = CFG['loss']['aux_ratio_max']*norm
    scale = min(1., (1-1e-4)*budget/raw) if raw else 0.
    actual = (reference + scale*aux).to(classification.dtype)
    applied = flat_norm(actual.double()-reference)
    if applied > budget:
        actual, applied, scale = classification, 0., 0.
    assert torch.isfinite(actual).all()
    details = {name: {'norm': flat_norm(value), 'cosine': gradient_cos(reference, value)}
               for name, value in zip(AUX_KEYS, grad_aux)}
    return actual, dict(classification=norm, weighted_aux=raw, applied=applied, ratio=applied/norm if norm else 0.,
                        cosine=gradient_cos(reference, aux), scale=scale, terms=details)


def load_gradients(parameters, update, active):
    for param, values, keep in zip(parameters, update.split([p.numel() for p in parameters]), active):
        param.grad = values.reshape_as(param).clone() if keep else None


def native_digests(state):
    return {name: sha(state[name].state_dict()) for name in ('model', 'cam', *HIST_OPTS)}


def native_gradient_sha(state):
    return sha({key: {name: p.grad for name, p in state[key].named_parameters()} for key in ('model', 'cam')})


def train_step(state, module, optimizer, group, inspect=False):
    ordered = [state[key] for key in HIST_OPTS] + [optimizer]
    for opt in ordered:
        opt.zero_grad(set_to_none=True)
    module.train()
    out, capture, dropout_calls = historical_forward(state, True, module)
    after_native = rng_mark()
    d = capture['details']
    mask = mask_from_native(capture)
    off = off_scores(state['cam'].Outputlayer, capture['z'], mask).log_softmax(1)
    if inspect:
        assert torch.equal(capture['dropout'][0]*mask, capture['dropout'][1])
        assert torch.equal(d['enhanced'].detach(), d['z'].detach()+d['delta'].detach())
        assert torch.allclose(out['logp'].detach(), capture['logits'].detach().log_softmax(1), rtol=0., atol=0.)
    total_b = total_for_off(state, out, off)
    old, new = state['old_layout'], state['new_layout']
    old_params, new_params = [p for ps in old.values() for p in ps], [p for ps in new.values() for p in ps]
    terms, spectral_notes = auxiliary_terms(state, module, group, out, d, off)
    b = torch.autograd.grad(total_b, old_params, retain_graph=True, allow_unused=True)
    if inspect:
        b_cls = torch.autograd.grad(out['cls'], old_params, retain_graph=True, allow_unused=True)
        direct = torch.autograd.grad(out['cls'], (d['enhanced'], d['delta'], *new_params), retain_graph=True, allow_unused=True)
        auxiliary_owners = torch.autograd.grad(sum(terms), old_params, retain_graph=True, allow_unused=True)
        # The auxiliary graphs can reach z via genuine enhancement. Gradient
        # extraction below targets only new leaves and never populates old .grad.
        assert all(p.grad is None for p in (*old_params, *new_params))
    aux_grads = [torch.autograd.grad(term, new_params, retain_graph=True, allow_unused=True) for term in terms]
    assert all(p.grad is None for p in (*old_params, *new_params))
    out['total'].backward()
    shared, fresh = {}, {}
    cursor = 0
    for name, ps in old.items():
        ref = b[cursor:cursor+len(ps)]
        current = [p.grad for p in ps]
        assert [g is not None for g in ref] == [g is not None for g in current]
        update, info = coordinate_shared(flatten(ps, ref), flatten(ps, current))
        load_gradients(ps, update, [g is not None for g in ref])
        if inspect:
            info['classification_norm'] = flat_norm(flatten(ps, b_cls[cursor:cursor+len(ps)]))
            info['auxiliary_graph_connected'] = sum(g is not None for g in auxiliary_owners[cursor:cursor+len(ps)])
        shared[name] = info
        cursor += len(ps)
    cursor = 0
    for name, ps in new.items():
        main = [p.grad for p in ps]
        parts = [flatten(ps, grads[cursor:cursor+len(ps)]) for grads in aux_grads]
        update, info = coordinate_added(flatten(ps, main), parts)
        load_gradients(ps, update, [g is not None for g in main])
        if inspect:
            direct_gradient = direct[2+cursor:2+cursor+len(ps)]
            assert torch.equal(flatten(ps, main), flatten(ps, direct_gradient))
            info['class_only_verified'] = True
        fresh[name] = info
        cursor += len(ps)

    proof = {}
    if inspect:
        assert direct[0] is not None and direct[1] is not None
        proof = dict(logits_sha=sha(capture['logits']), logp_sha=sha(out['logp']), off_logp_sha=sha(off),
                     shared_gradient_sha=native_gradient_sha(state), mask_sha=sha(mask),
                     delta_nonzero=int(torch.count_nonzero(d['delta'])),
                     delta_grad_norm=flat_norm(direct[1]), enhanced_grad_norm=flat_norm(direct[0]))
    for opt in ordered:
        opt.step()
    if inspect:
        proof['native_state'] = native_digests(state)
    for name, ps in old.items():
        original = state['origin'][name]
        shared[name]['relative_drift'] = float((flatten(ps).double()-original).norm()/original.norm().clamp_min(1e-12))
    assert rng_mark() == after_native, 'An added training computation changed the shared RNG'
    losses = {key: float(out[key].detach()) for key in LOSS_KEYS}
    losses.update({key: float(value.detach()) for key, value in zip(AUX_KEYS, terms)})
    notes = {key: float(value.detach()) for key, value in spectral_notes.items()}
    assert all(math.isfinite(v) for v in (*losses.values(), *notes.values()))
    return dict(losses=losses, shared=shared, added=fresh, spectral=notes, dropout_calls=dropout_calls, proof=proof)


def initialize_probe(state):
    """Native epoch 11 from the exact fold restore, no augmentation."""
    for key in HIST_OPTS:
        state[key].zero_grad(set_to_none=True)
    out, capture, calls = historical_forward(state, True)
    mask = mask_from_native(capture)
    assert torch.equal(capture['dropout'][0]*mask, capture['dropout'][1])
    out['total'].backward()
    proof = dict(logits_sha=sha(capture['logits']), logp_sha=sha(out['logp']), mask_sha=sha(mask),
                 shared_gradient_sha=native_gradient_sha(state))
    for key in HIST_OPTS:
        state[key].step()
    proof['native_state'] = native_digests(state)
    proof['losses'] = {key: float(out[key].detach()) for key in LOSS_KEYS}
    return proof, calls


def bundle(state, module, optimizer):
    return dict(base=h.runtime().load_history().state_bundle(state), dso=h.cpu_state(module),
                opt_new=copy.deepcopy(optimizer.state_dict()))


def restore(state, module, optimizer, saved, rng):
    h.runtime().load_history().load_bundle(state, saved['base'])
    module.load_state_dict(saved['dso'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved['opt_new']))
    h.base_runtime().restore_rng(rng)


def digests(saved):
    return {**{key: sha(value) for key, value in saved['base'].items()},
            'dso': sha(saved['dso']), 'opt_new': sha(saved['opt_new'])}


def checkpoint(state, module, optimizer):
    saved = bundle(state, module, optimizer)
    rng = h.base_runtime().pack_rng()
    return dict(current=saved, digests=digests(saved), rng=rng, rng_sha=h.base_runtime().rng_sha(rng))


@torch.no_grad()
def evaluate(state, module):
    module.eval()
    out, capture, calls = historical_forward(state, False, module)
    d = capture['details']
    head = state['cam'].Outputlayer
    off = off_scores(head, d['z']).log_softmax(1)
    rows = state['test_np']
    return dict(probability=out['logp'].exp()[rows].cpu().numpy(), off_probability=off.exp()[rows].cpu().numpy(),
                legal_cam=d['z'].index_select(0, state['legal_rows']).cpu(), dropout_calls=calls)


def advance_ema(state, module, evaluation):
    obs = observed_moments(evaluation['legal_cam'], state['label_legal'])
    module.update_ema(obs)
    return obs


def witness(state, module, optimizer, group):
    step = train_step(state, module, optimizer, group, inspect=True)
    train_rng = rng_mark()
    evaluation = evaluate(state, module)
    assert rng_mark() == train_rng
    before_ema = sha(module.ema_state())
    obs = advance_ema(state, module, evaluation)
    assert rng_mark() == train_rng
    return dict(step=step, evaluation_sha=sha(evaluation), observed_sha=sha(obs), before_ema=before_ema,
                post_ema=sha(module.ema_state()), digests=digests(bundle(state, module, optimizer)),
                rng_sha=train_rng[0], rng_streams=train_rng[1])


def check_zero_start(base, first):
    assert all(first['proof'][key] == base[key] for key in
               ('logits_sha', 'logp_sha', 'mask_sha', 'shared_gradient_sha', 'native_state'))
    assert first['proof']['off_logp_sha'] == base['logp_sha'] and first['proof']['delta_nonzero'] == 0
    assert first['proof']['delta_grad_norm'] > 0
    assert all(first['losses'][key] == base['losses'][key] for key in LOSS_KEYS)
    assert all(row['applied'] == 0 for row in first['shared'].values())
    assert first['added']['gate_net']['classification'] > 0


def certify_preflight():
    import audit

    start = time.monotonic()
    env = environment()
    before = protection()
    assert before['pass_'], before
    text = h.torch_load(h.PROMPT_CACHE)
    text_proof = h.verify_text(text)
    audit.synthetic_checks()
    records = []
    for fold in FOLDS:
        cuda = torch.cuda
        cuda.reset_peak_memory_stats()
        state, module, opt, spectrum, warm, meta = setup(fold, 'G1')
        options = h.init_optimizers(state, warm)
        cache = h.fixed_stats(state, meta['identity'], spectrum)
        marker = rng_mark()
        warm_cam, native_eval = h.warmup_cam(state)
        assert torch.equal(warm_cam, spectrum['warmup_cam'])
        assert sha(warm_cam) == spectrum['feature_sha']
        assert rng_mark() == marker
        basis_info = audit.initial_basis(state, module, spectrum, cache)
        snapshot = checkpoint(state, module, opt)
        archived = h.historical_task(fold)['rng_trace'][0]
        assert archived['epoch'] == START+1 and archived['rng_before'] == marker[0]
        initial_native, calls = initialize_probe(state)
        assert rng_mark()[0] == archived['rng_after_train']
        del native_eval
        groups = {}
        for group in GROUPS:
            restore(state, module, opt, snapshot['current'], snapshot['rng'])
            assert rng_mark() == marker
            assert digests(bundle(state, module, opt)) == snapshot['digests']
            with torch.no_grad():
                checked = audit.check_formulas(state, module, warm_cam.to(state['x'].device))
            tick = time.monotonic()
            step = train_step(state, module, opt, group, inspect=True)
            one_step_seconds = time.monotonic()-tick
            check_zero_start(initial_native, step)
            assert step['dropout_calls'] == calls and rng_mark()[0] == archived['rng_after_train']
            evaluation = evaluate(state, module)
            assert rng_mark()[0] == archived['rng_after_eval']
            obs = advance_ema(state, module, evaluation)
            audit.check_ema(module, snapshot['current']['dso'], obs)
            second = train_step(state, module, opt, group, inspect=True)
            assert all(row['classification'] > 0 for row in second['added'].values()), second['added']
            assert all(row['classification_norm'] > 0 for row in second['shared'].values())
            assert any(row['applied'] > 0 for row in second['shared'].values())
            post_update = audit.check_formulas(state, module, warm_cam.to(state['x'].device))
            moved_basis = audit.basis_diagnostics(module)
            assert moved_basis['projector_change_frobenius'] > 0
            groups[group] = dict(first=step['proof'], second_gradients=second['added'],
                                 dynamic_formula=post_update, dynamic_basis=moved_basis,
                                 second_shared=second['shared'], first_spectral=step['spectral'],
                                 formula=checked, step_wall_seconds=one_step_seconds)
        records.append(dict(fold=fold, identity=meta['identity'], initial_module_sha=meta['initial_module_sha'],
                            initial_ema_sha=meta['initial_ema_sha'], initializer=marker[1],
                            initial_rng=marker[0], legal_train_sha=meta['legal_train_sha'], spectrum_sha=meta['spectrum_sha'],
                            historical_optimizers=options, basis=basis_info, historical_epoch11=initial_native,
                            dropout_calls=calls, groups=groups, peak_worker_reserved_bytes=cuda.max_memory_reserved(),
                            peak_worker_allocated_bytes=cuda.max_memory_allocated()))
        print(f'PREFLIGHT f{fold:02d} G1/G2 passed', flush=True)
        del state, module, opt, snapshot
        cuda.empty_cache()
    value = dict(pass_=20, folds=records, environment=env, implementation=implementation(), protection_before=before,
                 text=text_proof, preflight_audit_sha256=sha_file(HERE/'audit.py'), wall_seconds=time.monotonic()-start)
    path = RT/'preflight.json'
    h.atomic_json(path, value)
    h.update_manifest(status='PREFLIGHT_PASSED', implementation=value['implementation'], environment=env,
                      preflight_sha256=sha_file(path), preflight_pass=20)
    return value


def accepted_preflight():
    manifest = h.read_json(HERE/'manifest.json')
    path = RT/'preflight.json'
    pre = h.read_json(path)
    assert pre['pass_'] == 20 and pre['implementation'] == implementation()
    assert sha_file(path) == manifest['preflight_sha256']
    return pre


def task_complete(group, fold):
    task, best, latest, log = h.paths(group, fold)
    if not task.is_file():
        return False
    value = h.read_json(task)
    assert value['status'] == 'COMPLETE'
    assert value['best_sha'] == sha_file(best) and value['latest_sha'] == sha_file(latest)
    assert value['log_sha'] == sha_file(log)
    assert value['implementation'] == implementation()
    return True


def train_fold(group, fold):
    if task_complete(group, fold):
        return
    started = time.monotonic()
    pre = accepted_preflight()['folds'][fold-1]
    state, module, opt, _, _, meta = setup(fold, group)
    initial = rng_mark()
    assert initial == (pre['initial_rng'], pre['initializer'])
    for key in ('identity', 'initial_module_sha', 'initial_ema_sha', 'spectrum_sha', 'legal_train_sha'):
        assert meta[key] == pre[key], key
    task_path, best_path, latest_path, log_path = h.paths(group, fold)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    reference = {row['epoch']: row for row in h.historical_task(fold)['rng_trace']}
    paired = {}
    if group == 'G2':
        assert task_complete('G1', fold)
        with h.paths('G1', fold)[3].open() as stream:
            paired = {row['epoch']: row for row in map(json.loads, stream)}
    truth = state['labels_np'][state['test_np']]
    torch.cuda.reset_peak_memory_stats()
    with log_path.open('a+b') as log:
        resumed_seconds = 0.
        pending = None
        if latest_path.is_file():
            saved = h.torch_load(latest_path)
            assert all(saved[key] == value for key, value in meta.items()), 'Latest provenance mismatch'
            restored = saved['current']
            h.reconcile_best(best_path, saved['best_sha'])
            prefix = h.committed_log(log, saved)
            restore(state, module, opt, restored, saved['rng'])
            assert digests(bundle(state, module, opt)) == saved['digests'] and rng_mark()[0] == saved['rng_sha']
            epoch0, wait, best_correct, best_epoch, best_sha = (saved[key] for key in
                ('next_epoch', 'wait', 'best_correct', 'best_epoch', 'best_sha'))
            resumes = saved['resumes'] + [epoch0]
            resumed_seconds = saved['worker_progress_seconds']
            assert int(module.ema_updates) == epoch0-START
            print(f'{group} f{fold:02d} restored Latest epoch {epoch0}', flush=True)
        else:
            assert not best_path.exists() and log.tell() == 0
            epoch0, wait, best_correct, best_epoch, best_sha = START, 0, -1, None, None
            resumes = []
            prefix = hashlib.sha256()
            h.commit_latest(log, prefix, latest_path, best_path, None, None, {
                **meta, **checkpoint(state, module, opt), 'next_epoch': START,
                'wait': 0, 'best_correct': -1, 'best_epoch': None, 'resumes': [], 'worker_progress_seconds': 0.})
        previous = rng_mark()
        end = epoch0
        for epoch in range(epoch0+1, LIMIT+1):
            if wait > CFG['early_stop_wait_gt']:
                break
            before = rng_mark()
            assert before == previous
            step = train_step(state, module, opt, group, inspect=epoch <= START+2)
            assert step['dropout_calls'] == [tuple(item) for item in pre['dropout_calls']]
            if epoch == START+1:
                check_zero_start(pre['historical_epoch11'], step)
            after_train = rng_mark()
            evaluation = evaluate(state, module)
            after_eval = rng_mark()
            trace = dict(zip(RNG_KEYS, (epoch, before[0], after_train[0], after_eval[0])))
            if epoch in reference:
                assert trace == reference[epoch], (group, fold, epoch, 'archived RNG')
            if epoch in paired:
                assert all(trace[key] == paired[epoch][key] for key in RNG_KEYS)
                assert after_eval[1] == paired[epoch]['rng_after_streams']
            assert before[1]['python'] == after_eval[1]['python']
            assert before[1]['numpy'] == after_eval[1]['numpy'] and before[1]['cpu'] == after_eval[1]['cpu']
            assert after_train == after_eval
            probability, off_probability = evaluation['probability'], evaluation['off_probability']
            pred = probability.argmax(1)
            correct = int((pred == truth).sum())
            best_at_epoch = correct > best_correct
            prior_ema = module.ema_state()
            assert int(prior_ema['ema_updates']) == epoch-START-1
            if best_at_epoch:
                best_correct, best_epoch, wait = correct, epoch, 0
                pending = dict(**meta, **checkpoint(state, module, opt), best_epoch=epoch,
                               test_rows=state['test_np'].copy(), probability=probability, off_probability=off_probability,
                               metrics=h.metrics_for(probability, truth), evaluation_ema_sha=sha(prior_ema))
            else:
                wait += 1
            observed = advance_ema(state, module, evaluation)
            assert rng_mark() == after_eval and int(module.ema_updates) == epoch-START
            previous = after_eval
            row = dict(**trace, **step, correct=correct, wait=wait, improved=best_at_epoch, best_epoch=best_epoch,
                       pred_bits=np.packbits(pred).tobytes().hex(), prob_sha=sha(probability),
                       off_prob_sha=sha(off_probability), legal_cam_sha=sha(evaluation['legal_cam']),
                       observation_sha=sha(observed), ema_before=sha(prior_ema), ema_after=sha(module.ema_state()),
                       rng_before_streams=before[1], rng_after_streams=after_eval[1])
            encoded = (json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode()
            log.write(encoded)
            prefix.update(encoded)
            end = epoch
            if epoch % CFG['save_latest_every_epochs'] == 0 or epoch == LIMIT or wait > CFG['early_stop_wait_gt']:
                payload = dict(**meta, **checkpoint(state, module, opt), next_epoch=epoch, wait=wait,
                               best_correct=best_correct, best_epoch=best_epoch, resumes=resumes,
                               probability=probability, off_probability=off_probability, last_eval_ema=prior_ema,
                               worker_progress_seconds=resumed_seconds+time.monotonic()-started)
                best_sha = h.commit_latest(log, prefix, latest_path, best_path, pending, best_sha, payload)
                pending = None
                print(f'{group} f{fold:02d} ep={epoch} ACC={100*correct/len(truth):.2f} '
                      f'Best={best_epoch}/{100*best_correct/len(truth):.2f}', flush=True)
    assert best_epoch is not None and (end == LIMIT or wait > CFG['early_stop_wait_gt'])
    assert sha_file(best_path) == best_sha and sha_file(log_path) == prefix.hexdigest()
    next_step = witness(state, module, opt, group)
    result = dict(**meta, status='COMPLETE', best_epoch=best_epoch, end_epoch=end, best_sha=best_sha,
                  latest_sha=sha_file(latest_path), log_sha=prefix.hexdigest(), resumes=resumes,
                  next_step=next_step, wall_seconds=resumed_seconds+time.monotonic()-started,
                  peak_worker_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_worker_reserved_bytes=torch.cuda.max_memory_reserved())
    h.atomic_json(task_path, result)
    print(f'COMPLETE {group} f{fold:02d}', flush=True)


def manage_training(group, requested):
    pre = accepted_preflight()
    lim = CFG['gpu']
    assert 1 <= requested <= lim['workers_max']
    if group == 'G2':
        group_status = h.read_json(RESULTS/'audit_status.json')
        assert sum(item['group'] == 'G1' and item['pass_'] for item in group_status['audits']) == 10
    started = time.monotonic()
    memory = gpu_sample()
    worker_estimate = max(x['peak_worker_reserved_bytes'] for x in pre['folds']) * 1.5 / 2**20
    free = memory['total_mib']-memory['used_mib']-lim['gpu_free_reserve_mib']
    assert free >= worker_estimate, (free, worker_estimate)
    workers = min(requested, int(free/worker_estimate), max(1, len(os.sched_getaffinity(0))//2))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    todo = [fold for fold in FOLDS if not task_complete(group, fold)]
    execution_path = RT/f'execution_{group}.json'
    if not todo and execution_path.is_file():
        return
    previous = h.read_json(execution_path) if execution_path.is_file() else {}
    running, failed = {}, []
    peak = memory['used_mib']
    high = 0
    last_notice = 0.
    print(f'{group} GPU0 workers={workers} pending={len(todo)} estimate={worker_estimate:.0f} MiB/worker', flush=True)
    while todo or running:
        while todo and len(running) < workers and not failed:
            fold = todo.pop(0)
            log = RT/'logs'/f'{group}_f{fold:02d}.worker.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a', encoding='utf-8') as handle:
                running[fold] = subprocess.Popen([sys.executable, '-B', str(HERE/'run.py'), 'fold',
                                                  '--group', group, '--fold', str(fold)], cwd=HERE,
                                                 env=env, stdout=handle, stderr=subprocess.STDOUT)
        high = max(high, len(running))
        for fold, proc in list(running.items()):
            code = proc.poll()
            if code is not None:
                del running[fold]
                if code:
                    failed.append(dict(fold=fold, return_code=code))
        sample = gpu_sample()
        peak = max(peak, sample['used_mib'])
        if time.monotonic()-last_notice >= 15:
            done = len(FOLDS)-len(todo)-len(running)-len(failed)
            print(f'{group} {done}/10 running={sorted(running)} GPU0={sample["used_mib"]} MiB '
                  f'util={sample["utilization_pct"]}%', flush=True)
            last_notice = time.monotonic()
        if failed and not running:
            break
        if running:
            time.sleep(2)
    result = dict(group=group, worker_limit=workers, actual_max_workers=max(high, previous.get('actual_max_workers', 0)),
                  sampled_gpu0_peak_mib=max(peak, previous.get('sampled_gpu0_peak_mib', 0)),
                  wall_seconds=time.monotonic()-started+previous.get('wall_seconds', 0.),
                  completed=sum(task_complete(group, f) for f in FOLDS), failures=failed)
    h.atomic_json(execution_path, result)
    print(json.dumps(result), flush=True)
    assert not failed, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight', 'train', 'fold'))
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--fold', type=int, choices=FOLDS)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.action == 'fold':
        assert args.group and args.fold
        environment()
        train_fold(args.group, args.fold)
        return
    with (RT/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            certify_preflight()
        else:
            assert args.group
            manage_training(args.group, args.workers)


if __name__ == '__main__':
    main()
