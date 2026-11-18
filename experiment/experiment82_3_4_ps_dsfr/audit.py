#!/usr/bin/env python3
"""Independent PS-DSFR equations, metrics, checkpoint replay, and research decision."""
from __future__ import annotations

import run as r  # Sets physical GPU0 before torch import.
import history as h

import argparse
import csv
import json
import math
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F

from ps_dsfr import EMA_BUFFERS, FOLD2_ROWS, SPECTRUM_BUFFERS


def independent_metrics(probability, labels):
    pred = probability.argmax(-1)
    return dict(zip(r.METRICS, map(float, (
        accuracy_score(labels, pred), balanced_accuracy_score(labels, pred), roc_auc_score(labels == 0, probability[:, 0]),
        recall_score(labels, pred, pos_label=0), recall_score(labels, pred, pos_label=1),
        f1_score(labels, pred, pos_label=0, zero_division=0)))))


def same_metrics(left, right):
    assert all(abs(left[k] - right[k]) < 1e-12 for k in r.METRICS), (left, right)


def legal_rows(state):
    rows = np.setdiff1d(state['train_np'], state['test_np'], assume_unique=True)
    assert np.array_equal(rows, h.legal_train(state))
    assert (np.intersect1d(state['train_np'], state['test_np']).tolist() ==
            (list(FOLD2_ROWS) if state['fold'] == 2 else []))
    return rows


def initial_ema(cache):
    base = cache['stats']
    count = base['counts'][:, None]
    second = base['variance'] * ((count - 1) / count.double()) + base['mean'].square()
    return {'ema_mean': base['mean'], 'ema_second': second, 'ema_counts': base['counts'],
            'ema_updates': torch.zeros((), dtype=torch.long), 'class_mean': base['mean'].float(),
            'class_var': base['stable'].float()}


def ema_update(previous, observation):
    count = previous['ema_counts']
    assert torch.equal(observation['counts'], count)
    alpha = r.CFG['ema']['alpha']
    mean = previous['ema_mean'] * (1 - alpha) + alpha * observation['mean']
    second = previous['ema_second'] * (1 - alpha) + alpha * observation['second']
    correction = r.CFG['class_statistics']['variance_ddof']
    variance = (second - mean.square()).clamp_min(0) * (count.double() / (count - correction))[:, None]
    pooled = (count[:, None] * variance).sum(dim=0) / count.sum()
    cfg = r.CFG['class_statistics']
    floor = torch.maximum(cfg['relative_floor'] * pooled, torch.full_like(pooled, cfg['variance_floor']))
    scaled = cfg['own_variance'] * variance + cfg['pooled_variance'] * pooled + floor
    assert torch.isfinite(scaled).all() and (scaled > 0).all()
    return {'ema_counts': count, 'ema_mean': mean, 'ema_second': second, 'class_mean': mean.float(),
            'class_var': scaled.float(), 'ema_updates': previous['ema_updates'] + 1}


def assert_ema_equal(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


@torch.no_grad()
def set_ema(module, snapshot):
    for name, value in snapshot.items():
        getattr(module, name).copy_(value)


def observe(state, out, spectrum):
    rows = legal_rows(state)
    z = out['details']['z'].detach()[rows].cpu()
    sp = spectrum['spectrum']
    h = torch.matmul(z.double() - sp['mu'], sp['U']).div((sp['lambda'] + r.CFG['epsilon']).sqrt())
    targets = state['labels_np'][rows]
    a, b = h[targets == 0], h[targets == 1]
    return z, {'counts': torch.tensor([len(a), len(b)]), 'mean': torch.stack((a.mean(0), b.mean(0))),
               'second': torch.stack((a.square().mean(0), b.square().mean(0)))}


def graph(state, module, intervention='full', *, training=False, grad=False):
    module.train(training)
    seen = []

    def inject(_head, args):
        before = r.rng_sha()
        enhanced, details = module(args[0], intervention=intervention)
        assert r.rng_sha() == before
        seen.append(details)
        return enhanced, args[1]

    def confirm_head_input(_head, args):
        assert len(seen) == 1 and args[0] is seen[0]['z_enhanced']

    head = state['cam'].Outputlayer
    handles = (head.register_forward_pre_hook(inject), head.register_forward_pre_hook(confirm_head_input))
    try:
        with h.count_calls(state), torch.set_grad_enabled(training or grad):
            out = h.runtime().forward(state, r.BASE_GROUP, training=training)
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 1
    out.update(details=seen[0], darg_total=out['total'])
    return out


def probability(out, state):
    return out['logp'].detach().exp()[state['test_np']].cpu().numpy()


def true_margin(logp, rows, labels):
    chosen = logp.index_select(0, rows)
    cls = labels.index_select(0, rows)[:, None]
    return (chosen.gather(1, cls) - chosen.gather(1, 1 - cls)).flatten()


def assert_witness(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key] == expected[key], (key, actual[key], expected[key])


@contextmanager
def replace_query(module, query):
    old = module.q0.detach().clone()
    with torch.no_grad():
        module.q0.copy_(query)
    try:
        yield
    finally:
        with torch.no_grad():
            module.q0.copy_(old)


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    assert x.size and np.isfinite(x).all()
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=0)), 'min': float(x.min()),
            'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'max': float(x.max())}


def correlation(a, b, *, ranks=False):
    a, b = np.asarray(a), np.asarray(b)
    if ranks:
        a, b = rankdata(a), rankdata(b)
    if not a.size or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def patient_changes(source, candidate, y, ids):
    src, new = source == y, candidate == y
    counts = {}
    for cls, name in ((0, 'asd'), (1, 'td')):
        fixed = ids[(~src & new) & (y == cls)].tolist()
        harmed = ids[(src & ~new) & (y == cls)].tolist()
        counts.update({f'{name}_fixed': len(fixed), f'{name}_harmed': len(harmed),
                       f'{name}_net': len(fixed) - len(harmed),
                       f'{name}_fixed_rows_1based': fixed, f'{name}_harmed_rows_1based': harmed})
    counts['fixed'] = counts['asd_fixed'] + counts['td_fixed']
    counts['harmed'] = counts['asd_harmed'] + counts['td_harmed']
    counts['net'] = counts['fixed'] - counts['harmed']
    counts['flips'] = int((source != candidate).sum())
    assert counts['fixed'] + counts['harmed'] == counts['flips']
    return counts


def history_paths(fold):
    return (h.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.best.pt',
            h.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.latest.pt')


def inputs_from_checkpoint(group, fold, checkpoint, pre):
    state, module, opt, spectrum, cache, text, metadata = r.task_setup(group, fold, pre)
    for key, expected in metadata.items():
        assert checkpoint[key] == expected, key
    assert r.sha(initial_ema(cache)) == checkpoint['initial_ema_sha']
    assert checkpoint['digests'] == r.component_digests(checkpoint['current'])
    # Validate legal initialization directly from cached raw warmup CAM.
    zz = spectrum['warmup_cam'][legal_rows(state)].double().numpy()
    sp = spectrum['spectrum']
    whitened = ((zz - sp['mu'].numpy()) @ sp['U'].numpy()) / np.sqrt(sp['lambda'].numpy() + r.CFG['epsilon'])
    labels = state['labels_np'][legal_rows(state)]
    splits = [whitened[labels == c] for c in (0, 1)]
    np.testing.assert_allclose(np.stack([x.mean(axis=0) for x in splits]), cache['stats']['mean'], atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(np.stack([x.var(axis=0, ddof=1) for x in splits]), cache['stats']['variance'],
                               atol=2e-12, rtol=1e-12)
    assert [len(x) for x in splits] == cache['stats']['counts'].tolist()
    r.load_bundle(state, module, opt, checkpoint['current'])
    r.base_runtime().restore_rng(checkpoint['rng'])
    assert r.rng_mark() == (checkpoint['rng_sha'], r.streams(checkpoint['rng']))
    assert checkpoint['digests'] == r.component_digests(r.bundle(state, module, opt))
    assert torch.equal(module.q0.cpu(), text[r.CFG['queries'][group]])
    assert all(torch.equal(getattr(module, k).cpu(), sp[v].float()) for k, v in SPECTRUM_BUFFERS.items())
    assert all(not buf.requires_grad for buf in module.buffers())
    assert not any(isinstance(x, torch.nn.Dropout) for x in module.modules())
    r.optimizer_ownership(state, module, opt)
    return state, module, opt, spectrum, cache, text


def optimizer_state_audit(state, module, opt, epoch):
    slots = {}
    for name, optimizer in zip((*r.HIST_OPTS, 'opt_prompt'), r.optimizers(state, opt)):
        assert optimizer.state
        setting = r.CFG['historical_optimizers'].get(name)
        if setting is None:
            hp = r.CFG['psce_optimizer']
            setting = (hp['class'], hp['lr'], hp['weight_decay'])
            assert len(optimizer.state) == len(list(module.parameters()))
        assert type(optimizer).__name__ == setting[0]
        for pg in optimizer.param_groups:
            assert pg['lr'] == setting[1] and pg['weight_decay'] == setting[2]
        for param, moments in optimizer.state.items():
            assert {'step', 'exp_avg', 'exp_avg_sq'} <= moments.keys()
            assert float(moments['step']) > 0
            if name == 'opt_prompt':
                assert float(moments['step']) == epoch - r.START
            for key in ('exp_avg', 'exp_avg_sq'):
                value = moments[key]
                assert value.shape == param.shape and value.dtype == param.dtype and torch.isfinite(value).all()
        slots[name] = len(optimizer.state)
    return slots


def csv_lookup(path):
    with path.open(newline='') as handle:
        return {(row['group'], int(row['fold'])): row for row in csv.DictReader(handle)}


@torch.no_grad()
def check_formulas(module, recorded, intervention='full'):
    """Rebuild the full feature residual from raw CAM and saved parameters."""
    eps = r.CFG['epsilon']
    z = recorded['z']
    basis = module.U
    h_raw = torch.mm(z, basis)
    white = torch.div(torch.mm(z - module.mu, basis), torch.sqrt(module.lam + eps))
    basis_embedding = F.linear(basis.T, module.direction.weight, module.direction.bias)
    inputs = torch.stack((white, torch.abs(white), module.loglam.expand_as(white), module.nu.expand_as(white)), -1)
    token = F.layer_norm(basis_embedding + F.linear(inputs, module.evidence.weight, module.evidence.bias),
                         (32,), module.norm.weight, module.norm.bias, module.norm.eps)
    directions = F.linear(token, module.key.weight)
    scaled = torch.matmul(directions, module.q0.T).transpose(-1, -2) / math.sqrt(32)
    null_score = torch.mv(module.q0, module.null)[None, :, None].expand(z.shape[0], -1, -1) / math.sqrt(32)
    values = torch.cat((scaled, null_score), -1) / r.CFG['temperature']
    att = (values - torch.logsumexp(values, -1, keepdim=True)).exp()

    c0, c1 = module.class_mean
    v0, v1 = module.class_var
    contrast_raw = ((white - c1).square() / v1 + v1.log() -
                    ((white - c0).square() / v0 + v0.log())) / 2
    limit = r.CFG['class_statistics']['llr_bound']
    contrast = limit * (contrast_raw / limit).tanh()
    spec_in = inputs[..., 1:]
    mlp = module.spectral
    spec = F.linear(F.gelu(F.linear(spec_in, mlp[0].weight, mlp[0].bias)), mlp[2].weight, mlp[2].bias)
    w = spec[..., 0].softmax(-1)
    reliable = (1 + torch.tanh(16*w)) / 2

    coverage, null, semantic = [], [], []
    for class_name in ('ASD', 'TD'):
        prompts = r.CFG['prompt_rows_0based'][class_name]
        c_a = 1 - torch.prod(1-att[:, prompts, :-1], dim=1)
        null_a = att[:, prompts, -1].mean(1)
        coverage.append(c_a)
        null.append(null_a)
        semantic.append(c_a / (c_a.sum(1, keepdim=True) + eps) * (1 - null_a[:, None]))
    disparity = semantic[0] - semantic[1]
    informed = 1 / (1 + torch.exp(-contrast * torch.tanh(16*disparity)))
    class_evidence = torch.stack([c * informed * reliable for c in semantic], dim=1)
    difference = class_evidence[:, 0] - class_evidence[:, 1]
    null_mass = torch.stack(null, dim=1)
    if intervention == 'semantic_off':
        difference = torch.zeros_like(difference)
        null_mass = torch.zeros_like(null_mass)
    normal = F.layer_norm(z, (144,), eps=r.CFG['signed_gate']['layer_norm_eps'])
    gate_in = torch.cat([normal, difference, reliable, null_mass], dim=1)
    g0, _, g1 = module.signed_gate
    gate = torch.tanh(F.linear(F.gelu(F.linear(gate_in, g0.weight, g0.bias)), g1.weight, g1.bias))
    if intervention == 'gate_off':
        gate = torch.zeros_like(gate)
    spectral = gate * difference * h_raw
    raw = torch.mm(spectral, basis.T)
    zlen = z.double().square().sum(1, keepdim=True).sqrt()
    rlen = raw.double().square().sum(1, keepdim=True).sqrt()
    bound = r.CFG['residual_limit']
    radial = bound * zlen * torch.tanh(rlen / (bound * torch.maximum(zlen, torch.full_like(zlen, eps))))
    radial *= zlen / (zlen + torch.finfo(z.dtype).tiny)
    unit = raw.double() / torch.where(rlen == 0, 1., rlen)
    delta = (radial * unit).to(z.dtype)
    if module.fold == 2:
        delta[list(FOLD2_ROWS)] = 0.
    if intervention == 'residual_off':
        delta.zero_()

    comparisons = {'h': h_raw, 'hh': white, 'tokens': token, 'attention': att,
                   'coverage': torch.stack(coverage, 1), 'null': null_mass, 'semantic': torch.stack(semantic, 1),
                   'raw_llr': contrast_raw, 'llr': contrast, 'w_sp': w, 'reliability': reliable,
                   'evidence': class_evidence, 'difference': difference, 'gate_input': gate_in,
                   'gate': gate, 'spectral_residual': spectral, 'raw': raw, 'delta': delta, 'z_enhanced': z + delta}
    cfg = r.CFG['preflight']
    for name, value in comparisons.items():
        torch.testing.assert_close(recorded[name], value, atol=cfg['independent_formula_atol'],
                                   rtol=cfg['independent_formula_rtol'], msg=lambda m: f'{name}: {m}')
    assert recorded['attention'].shape == (871, 8, 17) and recorded['semantic'].shape == (871, 2, 16)
    assert recorded['gate'].shape == (871, 16) and recorded['delta'].shape == z.shape == (871, 144)
    assert recorded['delta'].double().norm(dim=1).le(
        (bound + cfg['residual_bound_roundoff']) * zlen[:, 0]).all()
    assert recorded['difference'].abs().le(
        (recorded['semantic'][:, 0] - recorded['semantic'][:, 1]).abs() + 2e-8).all()
    if module.fold == 2:
        assert not recorded['delta'][list(FOLD2_ROWS)].count_nonzero()
    # U is frozen. The independently rebuilt back-projection must be in its fixed span.
    delta64, b64 = recorded['delta'].double(), basis.double()
    outside = delta64 - (delta64 @ b64) @ b64.T
    assert outside.norm() <= 1e-6 * delta64.norm()
    return {'delta_max_abs_error': float((recorded['delta'] - delta).abs().max()),
            'evidence_max_abs_error': float((recorded['difference'] - difference).abs().max())}


def check_interventions(state, module, text, group):
    before = r.rng_mark()
    outputs = {'T0': graph(state, module)}
    for test, intervention in (('T1', 'residual_off'), ('T2', 'gate_off'), ('T3', 'semantic_off')):
        outputs[test] = graph(state, module, intervention)
    opposite = 'G2' if group == 'G1' else 'G1'
    with replace_query(module, text[r.CFG['queries'][opposite]]):
        outputs['T4'] = graph(state, module)
        disabled = graph(state, module, 'semantic_off')
    for key in ('T1', 'T2', 'T3'):
        d = outputs[key]['details']
        assert not d['delta'].count_nonzero() and torch.equal(d['z_enhanced'], d['z'])
        assert torch.equal(outputs['T1']['logp'], outputs[key]['logp'])
    assert torch.equal(disabled['logp'], outputs['T3']['logp'])
    assert torch.equal(outputs['T0']['details']['z'], outputs['T4']['details']['z'])
    assert r.rng_mark() == before
    return outputs


def auxiliary_formulas(state, out):
    ids = legal_rows(state)
    d = out['details']
    y = state['y'][ids]
    row = torch.arange(len(ids), device=y.device)
    class_mass = d['evidence'][ids].sum(dim=2) + r.CFG['epsilon']
    nll = class_mass.sum(dim=1).log() - class_mass[row, y].log()
    class_losses = [nll[y == c].mean() for c in range(2)]
    sem = .5 * (class_losses[0] + class_losses[1])
    m0 = out['logp'][ids]
    margin = (m0[row, y] - m0[row, 1-y]).detach()
    cfg = r.CFG['loss']
    protection = torch.sigmoid((margin-cfg['margin_center']) / cfg['margin_temperature'])
    relative = d['delta'][ids].square().sum(1) / (d['z'][ids].square().sum(1) + r.CFG['epsilon'])
    return sem, (protection * relative).mean()


def replay_step(state, module, opt, spectrum):
    """Use a fresh optimizer load, additive leaf VJPs and independent post-eval EMA."""
    ordered = r.optimizers(state, opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    out = graph(state, module, training=True)
    sem, safe = auxiliary_formulas(state, out)
    trainable = list(module.parameters())
    vjp = [torch.autograd.grad(value, trainable, retain_graph=True, allow_unused=True) for value in (out['cls'], sem, safe)]
    hp = r.CFG['loss']
    lam, mu = hp['lambda_sem'], hp['lambda_safe']
    by_parameter = {id(p): n for n, p in enumerate(trainable)}
    ratios = []
    for tensors in module.parameter_groups().values():
        norms = []
        for derivatives in vjp:
            pieces = [derivatives[by_parameter[id(p)]] for p in tensors]
            vector = torch.cat([torch.zeros_like(p).reshape(-1) if grad is None else grad.reshape(-1)
                                for p, grad in zip(tensors, pieces)]).double()
            norms.append(float(vector.norm()))
        scale_denom = lam * norms[1] + mu * norms[2]
        if scale_denom > 0:
            ratios.append(norms[0] / (scale_denom + hp['gradient_eps']))
    scale = min(1., hp['auxiliary_ratio_max'] * min(ratios)) if ratios else 0.
    out['total'].backward()
    with torch.no_grad():
        for p, g_sem, g_safe in zip(trainable, *vjp[1:]):
            if g_sem is not None or g_safe is not None:
                add = scale * ((0. if g_sem is None else lam*g_sem) + (0. if g_safe is None else mu*g_safe))
                if p.grad is None:
                    p.grad = add
                else:
                    p.grad.add_(add)
    for optimizer in ordered:
        optimizer.step()
    terms = [out[key].detach() for key in (*r.DARG_LOSSES, 'darg_total')] + [sem.detach(), safe.detach()]
    terms += [out['darg_total'].detach() + scale * (lam*sem.detach() + mu*safe.detach())]
    losses = dict(zip((*r.DARG_LOSSES, 'darg_total', 'sem', 'safe', 'total'), torch.stack(terms).cpu().tolist()))
    step_rng = r.rng_mark()
    after = graph(state, module)
    p, (cam, obs) = probability(after, state), observe(state, after, spectrum)
    prev = h.calibration_state(module)
    set_ema(module, ema_update(prev, obs))
    assert r.rng_mark() == step_rng
    return dict(losses=losses, auxiliary_scale=scale, rng=step_rng[0], rng_streams=step_rng[1],
                digests=r.component_digests(r.bundle(state, module, opt)), probability_sha=r.sha(p),
                legal_cam_sha=r.sha(cam), ema_before_sha=r.sha(prev), ema_after_sha=r.sha(h.calibration_state(module)),
                observation_sha=r.sha(obs))


def reference_fold(state, identity, source_rows):
    fold = state['fold']
    saved = r.torch_load(history_paths(fold)[0])
    ids, labels = state['test_np'], state['labels_np'][state['test_np']]
    table = source_rows[(r.BASE_GROUP, fold)]
    p = saved['test_probability']
    assert saved['identity'] == identity and saved['identity']['split_sha'] == state['split_sha']
    assert saved['best_epoch_1based'] == int(table['best_epoch_1based'])
    assert r.sha_file(history_paths(fold)[0]) == table['checkpoint_sha256']
    prediction = p.argmax(-1)
    assert np.array_equal(prediction, saved['test_prediction'])
    metrics = independent_metrics(p, labels)
    td_auc = float(roc_auc_score(labels, p[:, 1]))
    assert abs(td_auc - metrics['prob_auc']) < 1e-12
    same_metrics(saved['metrics'], {k: float(table[k]) for k in r.METRICS})
    same_metrics(metrics, saved['metrics'])
    history = h.runtime().load_history()
    before = history.state_bundle(state)
    rng = r.rng_mark()
    try:
        history.load_bundle(state, saved['best_state'])
        with torch.no_grad(), h.count_calls(state):
            fresh = r.historical_forward(state)
        assert np.array_equal(probability(fresh, state), p)
    finally:
        history.load_bundle(state, before)
    assert r.rng_mark() == rng
    return {'group': 'DARG', 'fold': fold, 'best_epoch': saved['best_epoch_1based'],
            'end_epoch': int(table['epochs_executed']), 'metrics': metrics, 'rows_1based': (ids+1).tolist(),
            'labels': labels.tolist(), 'prediction': prediction.tolist(), 'probability': p.tolist(),
            'historical_best_reproduced': True}


def check_gradient_row(row):
    loss, hp = row['losses'], r.CFG['loss']
    assert all(math.isfinite(value) for value in loss.values())
    expected = loss['cls'] + loss['graph'] + .05*loss['pair'] + .01*(loss['adv'] + loss['preserve'])
    assert np.isclose(expected, loss['darg_total'], atol=1e-7, rtol=1e-6)
    gradients = row['gradients']
    assert tuple(gradients) == r.PARAMETER_GROUPS
    lam, mu = hp['lambda_sem'], hp['lambda_safe']
    denominators = {name: lam*g['sem_norm'] + mu*g['safe_norm'] for name, g in gradients.items()}
    valid = [gradients[name]['cls_norm'] / (size + hp['gradient_eps'])
             for name, size in denominators.items() if size > 0.]
    s = min(1., hp['auxiliary_ratio_max'] * min(valid)) if valid else 0.
    assert s == row['auxiliary_scale']
    total = loss['darg_total'] + s * (lam*loss['sem'] + mu*loss['safe'])
    assert np.isclose(total, loss['total'], atol=1e-7, rtol=1e-6)
    for name, g in gradients.items():
        cls, sem, safe = (g[k] for k in ('cls_norm', 'sem_norm', 'safe_norm'))
        assert min(cls, sem, safe, g['mixed_norm']) >= 0.
        measured = s * g['mixed_norm'] / cls if cls else 0.
        conservative = s * denominators[name] / cls if cls else 0.
        assert measured == g['weighted_ratio'] and conservative == g['budget_ratio']
        assert 0 <= measured <= conservative + 1e-12 <= hp['auxiliary_ratio_max'] + 2e-12
        for key, exists in [('cosine_cls_sem', bool(cls*sem)), ('cosine_cls_safe', bool(cls*safe)),
                            ('cosine_sem_safe', bool(sem*safe))]:
            assert (g[key] is not None and abs(g[key]) <= 1+1e-12) if exists else g[key] is None
        assert 0 <= g['nonzero_update_fraction'] <= 1
    if row['epoch'] <= r.START + 2:
        r.verify_new_gradients(row['classification'], first=row['epoch'] == r.START+1)


def audit_journal(state, task, best, latest, records, cache, best_cam, best_obs, pre):
    y = state['labels_np'][state['test_np']]
    end = task['end_epoch']
    assert [v['epoch'] for v in records] == list(range(r.START+1, end+1))
    historical = {v['epoch']: v for v in h.historical_task(state['fold'])['rng_trace']}
    saved = r.torch_load(history_paths(state['fold'])[1])
    ref_components = r.streams(saved['rng'])
    ref_end = saved['next_epoch']
    assert r.base_runtime().rng_sha(saved['rng']) == historical[ref_end]['rng_after_eval']
    prev, streams = task['identity']['warmup_rng_sha'], pre['initial_rng_streams']
    ema = initial_ema(cache)
    assert r.sha(ema) == task['initial_ema_sha']
    max_correct, wait, chosen, n_shared, paired_shared = -1, 0, None, 0, 0
    ref_end_match = False
    paired = {}
    if task['group'] == 'G2':
        paired = {row['epoch']: row for row in map(json.loads, r.paths('G1', state['fold'])[3].read_text().splitlines())}
    for row in records:
        epoch = row['epoch']
        assert wait <= r.CFG['early_stop_wait_gt']
        pred = np.unpackbits(np.frombuffer(bytes.fromhex(row['prediction_bits']), dtype=np.uint8), count=len(y))
        correct = int((pred == y).sum())
        better = correct > max_correct
        if better:
            max_correct, wait, chosen = correct, 0, epoch
        else:
            wait += 1
        assert (correct, better, wait, chosen) == (row['correct'], row['improved'], row['wait'], row['best_epoch'])
        assert (row['rng_before'], row['rng_before_streams']) == (prev, streams)
        assert row['rng_after_train'] == row['rng_after_eval'] == row['rng_after_ema']
        prev, streams = row['rng_after_ema'], row['rng_after_streams']
        if epoch in historical:
            assert {k: row[k] for k in r.RNG_KEYS} == historical[epoch]
            n_shared += 1
        if epoch in paired:
            keys = (*r.RNG_KEYS, 'rng_before_streams', 'rng_after_streams')
            assert all(row[k] == paired[epoch][k] for k in keys)
            paired_shared += 1
        if epoch == ref_end:
            assert streams == ref_components
            ref_end_match = True
        assert int(ema['ema_updates']) == epoch-r.START-1 and r.sha(ema) == row['ema_before']
        moment = {'mean': torch.tensor(row['observed_mean'], dtype=torch.float64),
                  'second': torch.tensor(row['observed_second'], dtype=torch.float64),
                  'counts': cache['stats']['counts']}
        if epoch == best['best_epoch']:
            assert r.sha(ema) == best['calibration_sha']
            assert_ema_equal(ema, {k: best['current']['ps_dsfr'][k] for k in EMA_BUFFERS})
            assert r.sha(moment) == r.sha(best_obs) and r.sha(best_cam) == row['legal_cam_sha']
            assert row['probability_sha'] == r.sha(best['probability']) and np.array_equal(pred, best['prediction'])
            assert (prev, streams) == (best['rng_sha'], r.streams(best['rng']))
        if epoch == end:
            assert_ema_equal(ema, latest['last_eval_ema'])
        ema = ema_update(ema, moment)
        assert r.sha(ema) == row['ema_after']
        check_gradient_row(row)
    assert best['best_epoch'] == task['best_epoch'] == latest['best_epoch'] == chosen
    assert (max_correct, wait, end) == (latest['best_correct'], latest['wait'], latest['next_epoch'])
    assert wait > r.CFG['early_stop_wait_gt'] or end == r.LIMIT
    assert (prev, streams) == (latest['rng_sha'], r.streams(latest['rng']))
    assert records[-1]['probability_sha'] == r.sha(latest['probability'])
    assert_ema_equal(ema, {k: latest['current']['ps_dsfr'][k] for k in EMA_BUFFERS})
    return dict(epochs=len(records), ema_updates=len(records), common_darg_rng_epochs=n_shared,
                extended_rng_epochs=len(records)-n_shared, paired_g1_rng_epochs=paired_shared,
                darg_terminal_streams_matched=ref_end_match)


def mechanism(state, module, outputs, reference):
    ids = state['test_np']
    base = outputs['T0']['details']
    y = state['labels_np'][ids]
    assert np.array_equal(ids+1, reference['rows_1based']) and np.array_equal(y, reference['labels'])
    mask = ~np.isin(ids, FOLD2_ROWS) if state['fold'] == 2 else np.ones(len(ids), dtype=bool)
    keep, labels, rows = ids[mask], y[mask], ids[mask]+1
    probs = {tag: probability(out, state) for tag, out in outputs.items()}
    pred0 = probs['T0'][mask].argmax(1)
    treatments = {}
    for tag, arr in probs.items():
        evaluated = arr[mask]
        change = np.abs(probs['T0'][:, 0] - arr[:, 0])
        record = dict(metrics=independent_metrics(evaluated, labels),
                      probability_abs_delta=distribution(change),
                      versus_full=patient_changes(pred0, evaluated.argmax(1), labels, rows))
        if tag in ('T2', 'T3'):
            record['structural_equivalent_to'] = 'T1'
        treatments[tag] = record
    direct = patient_changes(probs['T1'][mask].argmax(1), pred0, labels, rows)
    reference_correct = np.asarray(reference['prediction'])[mask] == labels
    yidx = torch.as_tensor(keep, device=state['y'].device)
    margin_off = true_margin(outputs['T1']['logp'], yidx, state['y']).cpu().numpy()
    margin_on = true_margin(outputs['T0']['logp'], yidx, state['y']).cpu().numpy()
    increment = margin_on - margin_off
    darg_pred = np.asarray(reference['prediction'])[mask]
    backbone = patient_changes(darg_pred, probs['T1'][mask].argmax(1), labels, rows)
    cam_norm = base['z'][ids].double().norm(dim=1)
    magnitude = 100*(base['delta'][ids].double().norm(dim=1) / cam_norm.clamp_min(1e-300))
    gate = base['gate'][ids].cpu().numpy()
    diff = base['difference'][ids]
    att_diff = base['coverage'][ids, 0] - base['coverage'][ids, 1]
    gates = [{'direction_1based': j+1, 'positive': int((gate[:, j] > 0).sum()),
              'negative': int((gate[:, j] < 0).sum()), 'zero': int((gate[:, j] == 0).sum()),
              'value': distribution(gate[:, j])} for j in range(16)]
    applied = (base['delta'][ids].double() @ module.U.double()).cpu().numpy()
    raw = base['spectral_residual'][ids].cpu().numpy()
    for j, rec in enumerate(gates):
        rec.update(raw_abs_mean=float(np.abs(raw[:, j]).mean()),
                   applied_abs_mean=float(np.abs(applied[:, j]).mean()),
                   applied_signed_mean=float(applied[:, j].mean()))
    summary = {
        'oof_rows_1based': (ids+1).tolist(), 'diagnostic_rows_1based': rows.tolist(),
        'diagnostic_labels': labels.tolist(), 'gate_by_direction': gates,
        'signed_gate_by_patient': gate.tolist(),
        'direction_magnitude_by_patient': np.abs(applied).tolist(),
        'attention_difference_l2': att_diff.double().norm(dim=1).cpu().tolist(),
        'evidence_difference_l2': diff.double().norm(dim=1).cpu().tolist(),
        'class_null_mean': base['null'][ids].double().mean(0).cpu().tolist(),
        'relative_residual_pct': magnitude.cpu().tolist(),
        'asdp_absolute_changes': {tag: np.abs(probs['T0'][:, 0] - p[:, 0]).tolist() for tag, p in probs.items()},
        'treatments': treatments, 'direct_hard_contribution': direct, 'residual_off_vs_darg': backbone,
        'true_margin_off': margin_off.tolist(), 'true_margin_full_minus_off': increment.tolist(),
        'historical_darg_correct': reference_correct.tolist(),
        'disease_evidence_true_mass_advantage': ((base['evidence'][keep, 0].sum(1) -
                                                base['evidence'][keep, 1].sum(1)).cpu().numpy() * (1-2*labels)).tolist(),
    }
    if state['fold'] == 2:
        assert not base['delta'][list(FOLD2_ROWS)].count_nonzero() and len(rows) == len(ids)-2
        assert all(np.array_equal(p[~mask], probs['T0'][~mask]) for p in probs.values())
    return summary


def parameter_drift(values, restored):
    groups = {}
    for prefix, state_key in [('darg', 'model'), ('cam', 'cam'), ('output', 'cam'), ('spectral_module', 'ps_dsfr')]:
        first = restored['base'][state_key] if state_key in ('model', 'cam') else restored['ps_dsfr']
        last = values['base'][state_key] if state_key in ('model', 'cam') else values['ps_dsfr']
        keys = [key for key in first if first[key].is_floating_point()]
        if prefix in ('cam', 'output'):
            keys = [k for k in keys if k.startswith('Outputlayer.') == (prefix == 'output')]
        if prefix == 'spectral_module':
            exclude = {*SPECTRUM_BUFFERS, *EMA_BUFFERS, 'q0'}
            keys = [key for key in keys if key not in exclude]
        denominator = sum(float(first[k].double().square().sum()) for k in keys)
        change = sum(float((last[k].double()-first[k].double()).square().sum()) for k in keys)
        groups[prefix] = {'absolute_l2': math.sqrt(change), 'relative_l2': math.sqrt(change/max(denominator, 1e-24)),
                          'changed_tensors': sum(not torch.equal(first[k], last[k]) for k in keys), 'tensors': len(keys)}
    return groups


def audit_task(group, fold, pre, source_rows):
    task_path, best_path, latest_path, log_path = r.paths(group, fold)
    assert r.task_complete(group, fold)
    task, best, latest = h.read_json(task_path), h.torch_load(best_path), h.torch_load(latest_path)
    assert best['identity'] == task['identity'] == latest['identity'] and latest['best_sha'] == task['best_sha']
    assert latest['log_bytes'] == log_path.stat().st_size and latest['log_sha'] == task['log_sha']
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    state, module, opt, spectrum, cache, text = inputs_from_checkpoint(group, fold, best, pre)
    reference = reference_fold(state, best['identity'], source_rows)
    ids = state['test_np']
    assert np.array_equal(best['test_indices'], ids)
    bslots = optimizer_state_audit(state, module, opt, best['best_epoch'])
    outputs = check_interventions(state, module, text, group)
    full = outputs['T0']
    equations = check_formulas(module, full['details'])
    for tag, mode in (('T1','residual_off'), ('T2','gate_off'), ('T3','semantic_off')):
        check_formulas(module, outputs[tag]['details'], mode)
    swap = 'G2' if group == 'G1' else 'G1'
    with replace_query(module, text[r.CFG['queries'][swap]]):
        check_formulas(module, outputs['T4']['details'])
    p = probability(full, state)
    assert np.array_equal(p, best['probability']) and np.array_equal(p.argmax(1), best['prediction'])
    y = state['labels_np'][ids]
    metrics = independent_metrics(p, y)
    same_metrics(metrics, best['metrics'])
    m = mechanism(state, module, outputs, reference)
    cam, obs = observe(state, full, spectrum)
    del outputs, full
    hist = audit_journal(state, task, best, latest, records, cache, cam, obs, pre['folds'][fold-1])
    state, _, _ = h.prepare(fold)
    pristine, pristine_opt = r.make_module(state, group, spectrum, cache, text)
    initial = r.bundle(state, pristine, pristine_opt)
    r.load_bundle(state, module, opt, latest['current'])
    # The latest optimizer must own the module and the freshly restored historical parameters.
    actual_old_optimizers = [state[k] for k in r.HIST_OPTS]
    assert len(actual_old_optimizers) == 3 and r.optimizer_ownership(state, module, opt)
    assert r.component_digests(r.bundle(state, module, opt)) == latest['digests']
    r.base_runtime().restore_rng(latest['rng'])
    ls = optimizer_state_audit(state, module, opt, task['end_epoch'])
    latest_ema = h.calibration_state(module)
    set_ema(module, latest['last_eval_ema'])
    terminal = graph(state, module)
    assert np.array_equal(probability(terminal, state), latest['probability'])
    terminal_cam, terminal_obs = observe(state, terminal, spectrum)
    assert r.sha(terminal_cam) == records[-1]['legal_cam_sha']
    assert_ema_equal(latest_ema, ema_update(latest['last_eval_ema'], terminal_obs))
    set_ema(module, latest_ema)
    assert r.rng_mark() == (latest['rng_sha'], r.streams(latest['rng']))
    assert_witness(replay_step(state, module, opt, spectrum), task['next_step_witness'])
    training = {'max_auxiliary_ratio': max(g['weighted_ratio'] for e in records for g in e['gradients'].values()),
                'mean_auxiliary_scale': float(np.mean([e['auxiliary_scale'] for e in records])),
                'semantic_loss_first': records[0]['losses']['sem'],
                'semantic_loss_at_best': records[best['best_epoch']-r.START-1]['losses']['sem'],
                'semantic_loss_latest': records[-1]['losses']['sem'],
                'best_parameter_drift': parameter_drift(best['current'], initial),
                'latest_parameter_drift': parameter_drift(latest['current'], initial)}
    return {'group': group, 'fold': fold, 'best_epoch': best['best_epoch'], 'end_epoch': task['end_epoch'],
            'metrics': metrics, 'rows_1based': (ids+1).tolist(), 'labels': y.tolist(), 'prediction': p.argmax(1).tolist(),
            'probability': p.tolist(), 'interventions': m, 'equations': equations,
            'best': 'PASS', 'latest': 'PASS', 'best_optimizer_slots': bslots, 'latest_optimizer_slots': ls,
            'history': hist, 'training': training, 'resumes': task['resumes'],
            'fold2_excluded_rows_1based': [377,407] if fold == 2 else []}, reference


def merge_patients(rows):
    keys = ('asd_fixed','asd_harmed','asd_net','td_fixed','td_harmed','td_net','fixed','harmed','net','flips')
    return {k: sum(v[k] for v in rows) for k in keys}


def summaries(records):
    return {metric: (float(np.mean([v['metrics'][metric] for v in records])*100),
                     float(np.std([v['metrics'][metric] for v in records], ddof=0)*100))
            for metric in r.METRICS}


def paired(left, right):
    assert left['fold'] == right['fold']
    assert left['rows_1based'] == right['rows_1based'] and left['labels'] == right['labels']
    return {'fold': left['fold'],
            'delta_pp': {k: 100*(left['metrics'][k] - right['metrics'][k]) for k in r.METRICS},
            'patients': patient_changes(np.asarray(right['prediction']), np.asarray(left['prediction']),
                                        np.asarray(left['labels']), np.asarray(left['rows_1based']))}


def summarize_pairs(rows):
    delta = {key: [v['delta_pp'][key] for v in rows] for key in r.METRICS}
    decisions = {key: [sum(fn(x) for x in delta[key]) for fn in
                       (lambda x: x > 1e-10, lambda x: abs(x) <= 1e-10, lambda x: x < -1e-10)]
                 for key in ('acc','hard_auc')}
    return {'delta_pp': {k: float(np.mean(v)) for k, v in delta.items()},
            'win_tie_loss': decisions, 'patients': merge_patients([v['patients'] for v in rows])}


def group_mechanism(tasks):
    parts = [v['interventions'] for v in tasks]
    gate = np.concatenate([v['signed_gate_by_patient'] for v in parts], axis=0)
    contributions = np.concatenate([v['direction_magnitude_by_patient'] for v in parts], 0)
    evidence = np.concatenate([v['evidence_difference_l2'] for v in parts])
    att = np.concatenate([v['attention_difference_l2'] for v in parts])
    residual = np.concatenate([v['relative_residual_pct'] for v in parts])
    true_margin = np.concatenate([v['true_margin_full_minus_off'] for v in parts])
    off = np.concatenate([v['true_margin_off'] for v in parts])
    label = np.concatenate([v['diagnostic_labels'] for v in parts])
    correct = np.concatenate([v['historical_darg_correct'] for v in parts])
    low = off < 2.
    def margin_profile(mask):
        x = true_margin[mask]
        if not x.size:
            return {'n': 0, 'change': None, 'fraction_positive': None}
        return {'n': len(x), 'change': distribution(x), 'fraction_positive': float((x > 0).mean())}
    variants = {}
    for tag in ('T0','T1','T2','T3','T4'):
        v = [p['treatments'][tag] for p in parts]
        variants[tag] = {'metrics': summaries(v),
                         'probability_change': distribution(np.concatenate([p['asdp_absolute_changes'][tag] for p in parts])),
                         'versus_full': merge_patients([v0['versus_full'] for v0 in v])}
    output = {'gates': {'positive': int((gate > 0).sum()), 'negative': int((gate < 0).sum()),
                        'zero': int((gate == 0).sum()), 'values': distribution(gate)},
              'by_direction': [{'direction_1based': j+1, 'positive': int((gate[:, j] > 0).sum()),
                                'negative': int((gate[:, j] < 0).sum()), 'zero': int((gate[:, j] == 0).sum()),
                                'applied_abs_mean': float(contributions[:, j].mean())} for j in range(16)],
              'attention_difference_l2': distribution(att), 'evidence_difference_l2': distribution(evidence),
              'residual_relative_pct': distribution(residual), 'nonzero_residual_rows': int((residual > 0).sum()),
              'margin': {'all': margin_profile(np.ones(len(label),dtype=bool)),
                         'asd': margin_profile(label == 0), 'td': margin_profile(label == 1),
                         'darg_correct': margin_profile(correct), 'darg_wrong': margin_profile(~correct),
                         'low_margin': margin_profile(low), 'high_margin': margin_profile(~low),
                         'spearman_vs_off': correlation(off,true_margin,ranks=True)},
              'treatments': variants,
              'direct_hard_contribution': merge_patients([v['direct_hard_contribution'] for v in parts]),
              'residual_off_vs_darg': merge_patients([v['residual_off_vs_darg'] for v in parts]),
              'evidence_true_mass_advantage': distribution(np.concatenate([p['disease_evidence_true_mass_advantage'] for p in parts])),
              'max_actual_aux_ratio': max(v['training']['max_auxiliary_ratio'] for v in tasks),
              'mean_sem_loss_first': float(np.mean([v['training']['semantic_loss_first'] for v in tasks])),
              'mean_sem_loss_at_best': float(np.mean([v['training']['semantic_loss_at_best'] for v in tasks])),
              'mean_sem_loss_latest': float(np.mean([v['training']['semantic_loss_latest'] for v in tasks]))}
    return output


def aggregate(tasks, references):
    groups = {'DARG': sorted(references, key=lambda v: v['fold'])}
    for group in r.GROUPS:
        rows = sorted([v for v in tasks if v['group'] == group], key=lambda v: v['fold'])
        if rows:
            groups[group] = rows
    all_ten = set(range(1, 872))
    for records in groups.values():
        if len(records) == len(r.FOLDS):
            all_rows = [row for v in records for row in v['rows_1based']]
            assert set(all_rows) == all_ten and len(all_rows) == len(all_ten)
    official = {g: summaries(v) for g, v in groups.items() if v}
    reduced = {g: summaries([v for v in rows if v['fold'] != 2]) for g, rows in groups.items()
               if any(v['fold'] != 2 for v in rows)}
    per_fold, panel = {}, {}
    for a,b in (('G1','DARG'),('G2','DARG'),('G1','G2')):
        if a not in groups or b not in groups:
            continue
        comparator = {v['fold']: v for v in groups[b]}
        rows = [paired(v, comparator[v['fold']]) for v in groups[a] if v['fold'] in comparator]
        if rows:
            key = f'{a}-{b}'
            per_fold[key] = rows
            panel[key] = {'all': summarize_pairs(rows)}
            if any(v['fold'] != 2 for v in rows):
                panel[key]['excluding_fold2'] = summarize_pairs([v for v in rows if v['fold'] != 2])
    mechanism_groups = {g: group_mechanism(records) for g, records in groups.items() if g != 'DARG'}
    return dict(metrics_10=official, metrics_without_fold2=reduced, pairs=panel, per_fold_pairs=per_fold,
                mechanism=mechanism_groups)


def science(status):
    out = status['aggregate']
    records = [v for v in status['tasks'] if v['group'] == 'G1']
    enough = len(records) == len(r.FOLDS) and not status['failures'] and status['protection']['pass']
    if not enough:
        return dict(verdict='NO-GO', performance_go=False, mechanism_go=False, analyze_g2=False,
                    medical_advantage=None, evidence_reason='G1 十折独立审计未完成。')
    hp = r.CFG['decision']
    m = out['metrics_10']['G1']
    delta = out['pairs']['G1-DARG']['all']['delta_pp']
    p = {'best_and_latest_10': enough, 'acc': m['acc'][0] > hp['darg_reference_pct']['acc'],
         'ba': m['hard_auc'][0] >= hp['darg_reference_pct']['hard_auc'],
         'prob_auc': delta['prob_auc'] >= hp['prob_auc_min_delta_pp'],
         'sen': delta['sen'] >= hp['sen_min_delta_pp'],
         'patient_net': out['pairs']['G1-DARG']['all']['patients']['net'] > 0}
    mechanisms = out['mechanism']['G1']
    q = {
        'endpoint_gradients': status['preflight_pass'] == 10 and all(v['training']['best_parameter_drift']['spectral_module']['absolute_l2'] > 0 for v in records),
        'nonzero_evidence': all(max(v['interventions']['attention_difference_l2']) > 0 and
                                max(v['interventions']['evidence_difference_l2']) > 0 for v in records),
        'nonzero_residual': all(max(v['interventions']['relative_residual_pct']) > 0 for v in records),
        'direct_hard_net': mechanisms['direct_hard_contribution']['net'] > 0,
    }
    performance_go, mechanism_go = all(p.values()), all(q.values())
    g2_rows = [v for v in status['tasks'] if v['group'] == 'G2']
    eligible = performance_go and mechanism_go
    medical = None
    if len(g2_rows) == 10:
        comparisons = out['pairs']['G1-G2']['all']
        diff = comparisons['delta_pp']
        medical = diff['acc'] > 0 and diff['hard_auc'] >= 0 and comparisons['patients']['net'] > 0
    return dict(verdict='GO' if eligible else 'NO-GO', performance_go=performance_go, mechanism_go=mechanism_go,
                analyze_g2=eligible, medical_advantage=medical,
                performance_checks=p, mechanism_checks=q)


def protected_status():
    protection = h.read_json(r.HERE / 'manifest.json')['protected_sha256']
    def verify(entry):
        path, old = entry
        try:
            now = r.sha_file(path)
        except OSError as exc:
            return {'path': path, 'error': repr(exc)}
        return {'path': path, 'before': old, 'after': now} if old != now else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures = [value for value in pool.map(verify, protection.items()) if value is not None]
    return {'count': len(protection), 'pass': not failures, 'mismatches': failures}


def audited_execution(group, failure=None):
    import report

    started = time.monotonic()
    manifest = h.read_json(r.HERE / 'manifest.json')
    previous = {'tasks': [], 'references': []}
    if group == 'G2':
        gate_path = r.RT / 'g1_gate.json'
        if gate_path.exists():
            r.require_anonymous_authorization()
            previous = h.read_json(r.RESULTS / 'audit_status.json')
            assert h.sha_file(r.RESULTS / 'audit_status.json') == h.read_json(gate_path)['stage_audit_sha256']
    status = {'tasks': previous['tasks'], 'references': previous['references'], 'failures': [],
              'preflight_pass': 0, 'implementation': r.implementation()}
    if failure is not None:
        status['failures'].append({'stage': group, 'detail': failure})
    try:
        r.assert_gpu()
        pre = r.passing_preflight()
        status['preflight_pass'] = len(pre['folds'])
    except Exception:
        status['failures'].append({'stage': 'Preflight', 'detail': traceback.format_exc()})
        pre = None
    if pre:
        source_rows = csv_lookup(h.runtime().EXP / 'results/fold_metrics.csv')
        for fold in r.FOLDS:
            try:
                if not r.task_complete(group, fold):
                    raise RuntimeError('Formal task lacks a completed checkpoint pair')
                row, reference = audit_task(group, fold, pre, source_rows)
                status['tasks'].append(row)
                if group == 'G1':
                    status['references'].append(reference)
                else:
                    saved = next(v for v in status['references'] if v['fold'] == fold)
                    assert reference == saved
                print(f'AUDIT {group} f{fold:02d} Best Latest RNG EMA PASS', flush=True)
            except Exception:
                detail = traceback.format_exc()
                status['failures'].append({'group': group, 'fold': fold, 'detail': detail})
                print(detail, flush=True)
    status['group_execution'] = {g: h.read_json(path) for g in r.GROUPS
                                 if (path := r.RT / f'execution_{g}.json').exists()}
    status['best_pass'] = sum(v['best'] == 'PASS' for v in status['tasks'])
    status['latest_pass'] = sum(v['latest'] == 'PASS' for v in status['tasks'])
    status['protection'] = protected_status()
    status['rng'] = {k: sum(v['history'][k] for v in status['tasks']) for k in
                     ('ema_updates','common_darg_rng_epochs','extended_rng_epochs','paired_g1_rng_epochs')}
    status['rng']['historical_terminal_stream_matches'] = sum(v['history']['darg_terminal_streams_matched'] for v in status['tasks'])
    status['aggregate'] = aggregate(status['tasks'], status['references'])
    if len(status['references']) == 10 and any(round(pair[0],4)!=r.CFG['decision']['darg_reference_pct'][key]
                                               for key,pair in status['aggregate']['metrics_10']['DARG'].items()):
        status['failures'].append({'stage':'DARG', 'detail':'Historical means disagree with preregistered reference.'})
    status['science'] = science(status)
    authorized = group == 'G1' and status['science']['analyze_g2']
    status['g2_status'] = ('AUTHORIZED' if authorized else 'AUDITED' if group == 'G2' and not status['failures'] else
                           'PARTIAL' if group == 'G2' else 'SKIPPED_PREREGISTERED' if not status['failures'] and
                           status['protection']['pass'] else 'BLOCKED_PENDING_G1_AUDIT')
    complete = not status['failures'] and status['protection']['pass']
    status['status'] = ('AUDITED_G1_G2_AUTHORIZED' if authorized else 'COMPLETE_AUDITED') if complete else (
        'PARTIAL' if status['best_pass'] or sum(v['completed_folds'] for v in status['group_execution'].values()) else 'FAILED')
    status['audit_wall_seconds'] = time.monotonic() - started + previous.get('audit_wall_seconds', 0.)
    status['elapsed_seconds'] = time.time() - datetime.fromisoformat(manifest['started_utc']).timestamp()
    report.write_tables(status)
    if not authorized:
        status['handoff'] = report.append_handoff(status, manifest)
        if status['handoff']['status'] == 'BLOCKED' and status['status'] == 'COMPLETE_AUDITED':
            status['status'] = 'PARTIAL'
    report.write_report(status, manifest)
    status_path = r.RESULTS / 'audit_status.json'
    h.atomic_json(status_path, status)
    if authorized:
        path = r.RT / 'g1_gate.json'
        h.atomic_json(path, {'implementation': status['implementation'], 'performance_go': True, 'mechanism_go': True,
                             'best_pass': 10, 'latest_pass': 10, 'stage_audit_sha256': h.sha_file(status_path),
                             'task_sha256': {str(f): h.sha_file(r.paths('G1', f)[0]) for f in r.FOLDS}})
        h.update_manifest(g1_gate_sha256=h.sha_file(path))
    result_files = [path for path in r.RESULTS.iterdir() if path.is_file()]
    h.update_manifest(status=status['status'], scientific_verdict=status['science']['verdict'], g2_status=status['g2_status'],
                      best_audit_pass=status['best_pass'], latest_audit_pass=status['latest_pass'],
                      result_sha256={p.name:h.sha_file(p) for p in result_files}, handoff=status.get('handoff'),
                      elapsed_seconds=status['elapsed_seconds'], verified_protection=status['protection'])
    print(f"{status['status']} {status['science']['verdict']} G2 {status['g2_status']}", flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=r.GROUPS, default='G1')
    parser.add_argument('--failure', help='Record an unrecoverable execution failure before handoff')
    args = parser.parse_args()
    result = audited_execution(args.group, args.failure)
    if result['status'] in ('PARTIAL', 'FAILED'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
