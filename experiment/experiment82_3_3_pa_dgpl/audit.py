#!/usr/bin/env python3
"""Independent PA-DGPL mathematics, checkpoint replay and patient-level audit."""
from __future__ import annotations

import run as r  # Pins GPU0 and numeric threads before importing torch.

import argparse
import csv
import fcntl
import hashlib
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F

from pa_dgpl import EMA_BUFFERS, FOLD2_ROWS, SPECTRUM_BUFFERS

DISPLAY = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')
BASELINES = ('DARG', 'DA-SRPE', 'E82.3.1-G1', 'E82.3.1-G2')
PAIRS = (('G1', 'DARG'), ('G2', 'DARG'), ('G2', 'G1'), ('G1', 'DA-SRPE'), ('G2', 'DA-SRPE'))
HANDOFF = r.HERE.parent / 'handoff.md'


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
    assert np.array_equal(rows, r.legal_train(state))
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
        with r.legacy.count_calls(state), torch.set_grad_enabled(training or grad):
            out = r.legacy.runtime().forward(state, r.BASE_GROUP, training=training)
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 1
    out['details'] = seen[0]
    return out


def probability(out, state):
    return out['logp'].detach().exp()[state['test_np']].cpu().numpy()


def true_margin(logp, rows, labels):
    chosen = logp.index_select(0, rows)
    cls = labels.index_select(0, rows)[:, None]
    return (chosen.gather(1, cls) - chosen.gather(1, 1 - cls)).flatten()


def reconstruct_gain(state, out):
    cfg = r.CFG['gain']
    d = out['details']
    rows = torch.as_tensor(legal_rows(state), device=d['z'].device)
    margin = true_margin(out['logp'], rows, state['y'])
    derivative, = torch.autograd.grad(margin.sum(), d['z_enhanced'], retain_graph=True, create_graph=False)
    first_order = torch.sum(derivative[rows].detach() * d['delta'][rows], dim=1)
    weight = torch.sigmoid((cfg['difficulty_center'] - margin.detach()) / cfg['difficulty_temperature'])
    weight = torch.clamp(weight, cfg['difficulty_min'], cfg['difficulty_max'])
    labels = state['y'][rows]
    terms = weight * F.softplus(-first_order / cfg['softplus_temperature'])
    class_means = [terms[labels == c].mean() for c in (0, 1)]
    return (class_means[0] + class_means[1]) * .5, first_order, margin, derivative, weight


def check_gain(state, out):
    before = r.rng_mark()
    loss, gain, margin, v, weight = reconstruct_gain(state, out)
    found = out['gain']
    legal = legal_rows(state)
    assert torch.equal(found['labels'].cpu(), torch.as_tensor(state['labels_np'][legal]))
    assert torch.equal(loss, found['loss']) and torch.equal(gain, found['gain'])
    assert torch.equal(margin, found['margin']) and torch.equal(weight, found['weight'])
    assert not v.requires_grad and not found['v'].requires_grad and not found['weight'].requires_grad
    assert torch.equal(v[legal], found['v'])
    assert not v[np.setdiff1d(np.arange(len(v)), legal)].count_nonzero()
    d = out['details']
    for cls in (0, 1):
        first = int(legal[np.flatnonzero(state['labels_np'][legal] == cls)[0]])
        sample = out['logp'][first, cls] - out['logp'][first, 1 - cls]
        single, = torch.autograd.grad(sample, d['z_enhanced'], retain_graph=True, create_graph=False)
        assert torch.equal(single[first], v[first])
        assert not single[np.arange(len(single)) != first].count_nonzero()
    changed = state['y'].clone()
    i = state['test_np']
    changed[i] = 1 - changed[i]
    recomputed = r.decision_gain(out['logp'], d['z_enhanced'], d['delta'], state['cspe_legal'], changed, r.CFG['gain'])
    assert all(torch.equal(recomputed[k], found[k]) for k in found)
    assert before == r.rng_mark()


def replay_step(state, module, opt, spectrum, group):
    ordered = r.optimizers(state, opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    out = graph(state, module, training=True)
    d = out['details']
    rows = legal_rows(state)
    residual = ((d['delta'][rows] ** 2).sum(1) / ((d['z'][rows] ** 2).sum(1) + r.CFG['epsilon'])).mean()
    historical = out['total']
    primary = historical + r.CFG['residual_loss_coefficient'] * residual
    scalar_total, coefficient = primary.detach(), 0.
    report = {k: float(out[k].detach()) for k in r.DARG_LOSSES}
    if group == 'G2':
        gain_loss, gain, _, _, _ = reconstruct_gain(state, out)
        parameters = list(module.parameter_groups().values())
        flat = [p for part in parameters for p in part]
        cls = torch.autograd.grad(out['cls'], flat, retain_graph=True)
        aux = torch.autograd.grad(gain_loss, flat, retain_graph=True)
        norms = [[float(piece.norm()) for piece in torch.split(torch.cat([v.flatten() for v in values]).double(),
                                                               [sum(p.numel() for p in part) for part in parameters])]
                 for values in (cls, aux)]
        cfg = r.CFG['gain']
        ratios = [main / (extra + cfg['grad_epsilon']) for main, extra in zip(*norms) if extra > 0]
        coefficient = min(cfg['lambda_max'], cfg['rho'] * min(ratios)) if ratios else 0.
        # Feed the separately computed derivative directly to parameter leaves.
        torch.autograd.backward((primary, *flat), (torch.ones_like(primary), *(coefficient * v for v in aux)))
        scalar_total = primary.detach() + coefficient * gain_loss.detach()
        report.update(gain=float(gain_loss.detach()), mean_gain=float(gain.detach().mean()),
                      positive_gain_fraction=float((gain.detach() > 0).float().mean()))
    else:
        primary.backward()
    for optimizer in ordered:
        optimizer.step()
    report.update(darg_total=float(historical.detach()), res=float(residual.detach()), total=float(scalar_total))
    rng = r.rng_mark()
    evaluation = graph(state, module)
    p = probability(evaluation, state)
    cam, obs = observe(state, evaluation, spectrum)
    prior = r.legacy.calibration_state(module)
    set_ema(module, ema_update(prior, obs))
    assert r.rng_mark() == rng
    return {'losses': report, 'lambda_gain': coefficient, 'rng': rng[0], 'rng_streams': rng[1],
            'digests': r.component_digests(r.bundle(state, module, opt)), 'probability_sha': r.sha(p),
            'legal_cam_sha': r.sha(cam), 'ema_before_sha': r.sha(prior),
            'ema_after_sha': r.sha(r.legacy.calibration_state(module)), 'observation_sha': r.sha(obs)}


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


def check_interventions(state, module, text, *, verify_only=False):
    rng = r.rng_mark()
    residual = graph(state, module, 'residual_off')
    gate = graph(state, module, 'gate_off')
    semantic = graph(state, module, 'semantic_off')
    with replace_query(module, text['q_anonymous']):
        anon_off = graph(state, module, 'semantic_off')
        anon = None if verify_only else graph(state, module)
    for name, a, b in (('T1=T2', residual, gate), ('semantic_off Query swap', semantic, anon_off)):
        assert torch.equal(a['logp'], b['logp']), name
        assert all(torch.equal(a['details'][k], b['details'][k]) for k in ('z_enhanced', 'delta')), name
    for intervention in (residual, gate):
        assert torch.equal(intervention['details']['z_enhanced'], intervention['details']['z'])
    assert not semantic['details']['gate_input'][:, 144:].count_nonzero()
    assert torch.equal(semantic['details']['w'], semantic['details']['w_sp'])
    assert r.rng_mark() == rng
    return {'T1': residual, 'T2': gate, 'T3': semantic, 'T4': anon}


@torch.no_grad()
def check_formulas(module, d):
    """Recompute every operation independently at the model's device and precision."""
    values = module.state_dict()
    z = d['z'].detach()
    eps = r.CFG['epsilon']
    tol = r.CFG['preflight']

    def linear(x, layer):
        return F.linear(x, values[layer + '.weight'], values.get(layer + '.bias'))

    def mlp(x, name):
        return linear(F.gelu(linear(x, name + '.0')), name + '.2')

    h = torch.mm(z - values['mu'], values['U'])
    hh = h / (values['lam'] + eps).sqrt()
    loglam, nu = (values[k].expand_as(hh) for k in ('loglam', 'nu'))
    es = torch.stack((hh, hh.abs(), loglam, nu), dim=-1)
    token_sum = linear(values['U'].T, 'direction')[None] + linear(es, 'evidence')
    tokens = F.layer_norm(token_sum, (32,), values['norm.weight'], values['norm.bias'])
    q = values['q0']
    keys = linear(tokens, 'key')
    content = torch.einsum('kd,njd->nkj', q, keys) / math.sqrt(32)
    null_dot = ((q @ values['null']) / math.sqrt(32)).view(1, 8, 1).expand(len(z), -1, -1)
    attention = (torch.cat((content, null_dot), dim=2) / r.CFG['temperature']).softmax(2)
    mean, var = values['class_mean'], values['class_var']
    quadratic = (hh[:, None, :] - mean[None]).square() / var[None] + var.log()[None]
    raw_llr = .5 * (quadratic[:, 1] - quadratic[:, 0])
    bound = r.CFG['class_statistics']['llr_bound']
    llr = bound * torch.tanh(raw_llr / bound)
    normal, abnormal = (1 - (1 - a).prod(dim=1) for a in attention[:, :, :16].split(4, dim=1))
    pos, neg = F.softplus(llr), F.softplus(-llr)

    def unit_sum(w):
        return w / (w.sum(1, keepdim=True) + eps)

    scores = torch.stack(((unit_sum(abnormal) * pos).sum(1), (unit_sum(normal) * neg).sum(1)), dim=1)
    weights = scores.softmax(-1)
    ws = weights[:, :1] * unit_sum(abnormal * pos) + weights[:, 1:] * unit_sum(normal * neg)
    wp = mlp(torch.stack((hh.abs(), loglam, nu), -1), 'spectral').squeeze(-1).softmax(-1)
    null = attention[:, :, -1].mean(1)
    entropy = -torch.sum(ws * torch.log(ws + eps), 1) / math.log(16)
    beta_ref = (1 - null) * (1 - entropy).clamp(.2, .8)
    difference = ws - wp
    beta_input = torch.stack((1 - entropy, null, scores[:, 0] - scores[:, 1], difference.abs().sum(1)), dim=1)
    beta_adjust = torch.tanh(mlp(beta_input, 'beta_net').squeeze(-1))
    small = r.CFG['beta']['logit_eps']
    b = beta_ref.clamp(small, 1 - small)
    beta = torch.sigmoid(b.log() - torch.log1p(-b) + beta_adjust)
    mixture = beta.unsqueeze(1) * ws + (1 - beta.unsqueeze(1)) * wp
    raw = ((mixture - 1 / 16) * h) @ values['U'].T
    z_norm = z.norm(dim=1, keepdim=True)
    t = raw.norm(dim=1, keepdim=True) / (r.CFG['residual_limit'] * z_norm.clamp_min(eps))
    factor = torch.where(t < 1e-4, 1 - t.square() / 3 + 2 * t.pow(4) / 15, torch.tanh(t) / t)
    base = raw * factor * (z_norm / eps).clamp(max=1)
    normalized = F.layer_norm(z, (144,), eps=r.CFG['signed_gate']['layer_norm_eps'])
    gi = torch.cat((normalized, difference, beta.unsqueeze(1), null.unsqueeze(1)), dim=1)
    gate = torch.tanh(mlp(gi, 'signed_gate').squeeze(1))
    delta = gate.unsqueeze(1) * base
    if module.fold == 2:
        delta[list(FOLD2_ROWS)] = 0
        assert not d['delta'][list(FOLD2_ROWS)].count_nonzero()
    reconstruct = {'h': h, 'hhat': hh, 'a': attention, 'bounded_llr': llr, 'scores': scores, 'w_sem': ws, 'w_sp': wp,
                   'entropy': entropy, 'null': null, 'beta_old': beta_ref, 'beta_input': beta_input,
                   'beta_adjustment': beta_adjust, 'beta': beta, 'w': mixture,
                   'delta_raw': raw, 'delta_base': base, 'gate_input': gi, 'gate': gate, 'delta': delta,
                   'z_enhanced': z + delta}
    errors = {}
    for key, estimate in reconstruct.items():
        found = d[key].detach()
        torch.testing.assert_close(found, estimate, atol=tol['independent_math_atol'], rtol=tol['independent_math_rtol'],
                                   msg=lambda msg: key + '\n' + msg)
        errors[key] = float((found - estimate).abs().max())
    assert torch.equal(d['z_enhanced'], d['z'] + d['delta'])
    assert d['beta'].min() > 0 and d['beta'].max() < 1 and d['gate'].abs().max() <= 1
    for key in ('delta_base', 'delta'):
        assert (d[key].norm(dim=-1) <= (r.CFG['residual_limit'] + tol['residual_bound_roundoff']) * d['z'].norm(dim=-1)).all()
    return errors


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


def margin_profile(gate, first_order, real):
    return {'n': len(real), 'delta': distribution(real), 'positive_fraction': float((real > 0).mean()),
            'negative_fraction': float((real < 0).mean()),
            'gain': distribution(first_order), 'gain_to_margin_pearson': correlation(first_order, real),
            'gain_to_margin_spearman': correlation(first_order, real, ranks=True),
            'gate_to_margin_pearson': correlation(gate, real),
            'gate_to_margin_spearman': correlation(gate, real, ranks=True)}


def mechanism(state, module, full, variants):
    d = full['details']
    ids = state['test_np']
    dev = d['z'].device
    test = torch.as_tensor(ids, device=dev)
    y = state['labels_np'][ids]
    p0 = probability(full, state)
    included = ~np.isin(ids, FOLD2_ROWS) if state['fold'] == 2 else np.ones(len(ids), dtype=bool)
    diagnostic_rows = torch.as_tensor(ids[included], device=dev)
    diagnostic_labels = y[included]
    margin = true_margin(full['logp'], diagnostic_rows, state['y'])
    slope, = torch.autograd.grad(margin.sum(), d['z_enhanced'], create_graph=False)
    mg0 = true_margin(variants['T1']['logp'], diagnostic_rows, state['y'])
    with torch.no_grad():
        real = (margin - mg0).cpu().numpy()
        gain = (slope[diagnostic_rows] * d['delta'][diagnostic_rows]).sum(1).cpu().numpy()
        tensor = {k: d[k][test].detach().cpu().numpy() for k in ('gate', 'beta', 'beta_old', 'beta_adjustment')}
        gate = tensor['gate']
        norm = d['z'][test].norm(dim=-1).clamp_min(r.CFG['epsilon'])
        rel0 = (d['delta_base'][test].norm(dim=-1) / norm).cpu().numpy()
        rel = (d['delta'][test].norm(dim=-1) / norm).cpu().numpy()
        basis = module.U.detach().cpu().double().numpy()
        delta = d['delta'][test].cpu().double().numpy()
        projected = delta @ basis
        peak = np.argmax(np.abs(projected), axis=1)
        principal = np.sign(projected[np.arange(len(peak)), peak]) * (peak + 1)
        energy = np.square(projected).sum(0)
        error = float(np.abs(delta - projected @ basis.T).max())
        signed = {'positive': int((gate > 0).sum()), 'negative': int((gate < 0).sum()), 'zero': int((gate == 0).sum())}
        assert all(p >= 0 and math.isfinite(p) for p in rel)
        if state['fold'] == 2:
            assert ids[~included].tolist() == list(FOLD2_ROWS)
            assert not slope[list(FOLD2_ROWS)].count_nonzero()
            assert torch.equal(full['logp'][list(FOLD2_ROWS)], variants['T1']['logp'][list(FOLD2_ROWS)])
        beta_delta = tensor['beta'] - tensor['beta_old']

    interventions = {}
    for kind, out in {'T0': full, **variants}.items():
        p = probability(out, state)
        change_p = np.abs(p[:, 0] - p0[:, 0])
        interventions[kind] = {'metrics': independent_metrics(p, y), 'probability_change': distribution(change_p),
                               **patient_changes(p0.argmax(-1), p.argmax(-1), y, ids + 1)}
    return {
        'gate': {**distribution(gate), **signed, 'abs': distribution(abs(gate)),
                 'one_sign': signed['positive'] == 0 or signed['negative'] == 0},
        'beta': {**distribution(tensor['beta']), 'old': distribution(tensor['beta_old']),
                 'change': distribution(beta_delta), 'abs_change': distribution(abs(beta_delta)),
                 'adjustment': distribution(tensor['beta_adjustment']),
                 'beta_saturated_n': int(((tensor['beta'] <= .01) | (tensor['beta'] >= .99)).sum()),
                 'adjustment_saturated_n': int((abs(tensor['beta_adjustment']) >= .99).sum())},
        'residual': {'pre_gate': distribution(rel0), 'actual': distribution(rel),
                     'nonzero_n': int((rel > 0).sum()), 'projection_max_error': error,
                     'energy_16d': (energy / energy.sum()).tolist() if energy.sum() else [0.] * 16,
                     'actually_entered_output_layer': True},
        'margin': {label: margin_profile(gate[included][mask], gain[mask], real[mask])
                   for label, mask in [('all', np.ones(len(real), bool)), ('ASD', diagnostic_labels == 0),
                                       ('TD', diagnostic_labels == 1)]},
        'interventions': interventions,
        'patients': {'rows_1based': (ids + 1).tolist(), 'gate': gate.tolist(),
                     'beta_old': tensor['beta_old'].tolist(), 'beta': tensor['beta'].tolist(),
                     'relative_residual_before_gate': rel0.tolist(), 'relative_residual_actual': rel.tolist(),
                     'largest_residual_direction_signed_1based': principal.astype(int).tolist()},
        'margin_patients': {'rows_1based': (ids[included] + 1).tolist(), 'labels': diagnostic_labels.tolist(),
                            'gate': gate[included].tolist(), 'true_margin_full': margin.detach().cpu().tolist(),
                            'true_margin_residual_off': mg0.detach().cpu().tolist(),
                            'first_order_gain': gain.tolist(), 'true_margin_change': real.tolist()},
        'excluded_from_labelled_margin_audit_rows_1based': (ids[~included] + 1).tolist(),
    }


def history_paths(fold):
    return (r.legacy.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.best.pt',
            r.legacy.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.latest.pt')


def inputs_from_checkpoint(group, fold, checkpoint, pre):
    state, module, opt, spectrum, cache, text, metadata = r.task_setup(group, fold, pre)
    for key, expected in metadata.items():
        assert checkpoint[key] == expected, key
    assert r.sha(initial_ema(cache)) == checkpoint['initial_ema_sha']
    assert checkpoint['digests'] == r.component_digests(checkpoint['current'])
    # Validate legal initialization directly from cached raw warmup CAM.
    zz = spectrum['warmup_cam'][legal_rows(state)].double().numpy()
    sp = spectrum['spectrum']
    h = ((zz - sp['mu'].numpy()) @ sp['U'].numpy()) / np.sqrt(sp['lambda'].numpy() + r.CFG['epsilon'])
    labels = state['labels_np'][legal_rows(state)]
    splits = [h[labels == c] for c in (0, 1)]
    np.testing.assert_allclose(np.stack([x.mean(axis=0) for x in splits]), cache['stats']['mean'], atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(np.stack([x.var(axis=0, ddof=1) for x in splits]), cache['stats']['variance'],
                               atol=2e-12, rtol=1e-12)
    assert [len(x) for x in splits] == cache['stats']['counts'].tolist()
    r.load_bundle(state, module, opt, checkpoint['current'])
    r.base_runtime().restore_rng(checkpoint['rng'])
    assert r.rng_mark() == (checkpoint['rng_sha'], r.streams(checkpoint['rng']))
    assert checkpoint['digests'] == r.component_digests(r.bundle(state, module, opt))
    assert torch.equal(module.q0.cpu(), text['q_medical'])
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


def reference_fold(state, identity, label, rows):
    fold = state['fold']
    if label == 'DARG':
        ckpt = r.torch_load(history_paths(fold)[0])
        p, prediction, epoch = (ckpt[k] for k in ('test_probability', 'test_prediction', 'best_epoch_1based'))
        source = rows[(r.BASE_GROUP, fold)]
        scale = 1.
        assert ckpt['identity']['split_sha'] == state['split_sha']
    else:
        history = r.PREVIOUS if label == 'DA-SRPE' else r.legacy.E831
        g = 'G1' if label.endswith('-G1') else 'G2'
        ckpt = r.torch_load(history / 'runtime/checkpoints' / f'{g}_f{fold:02d}.best.pt')
        p, prediction, epoch = (ckpt[k] for k in ('probability', 'prediction', 'best_epoch'))
        assert np.array_equal(ckpt['test_indices'], state['test_np'])
        source = rows[(g, fold)]
        scale = 100.
    assert ckpt['identity'] == identity and np.array_equal(prediction, p.argmax(-1))
    ids = state['test_np']
    y = state['labels_np'][ids]
    metrics = independent_metrics(p, y)
    published = {**metrics, 'prob_auc': float(roc_auc_score(y, p[:, 1]))} if label == 'DARG' else metrics
    same_metrics(published, ckpt['metrics'])
    same_metrics(published, {k: float(source[k]) / scale for k in r.METRICS})
    return {'group': label, 'fold': fold, 'best_epoch': epoch,
            'end_epoch': int(source.get('end_epoch') or source['epochs_executed']),
            'metrics': metrics, 'rows_1based': (ids + 1).tolist(), 'labels': y.tolist(),
            'prediction': prediction.tolist(), 'probability': p.tolist(), 'historical_csv_and_checkpoint_agree': True}


def audit_journal(state, task, best, latest, records, cache, best_cam, best_obs):
    y = state['labels_np'][state['test_np']]
    end = task['end_epoch']
    assert [e['epoch'] for e in records] == list(range(r.START + 1, end + 1))
    historical = {e['epoch']: e for e in r.historical_task(state['fold'])['rng_trace']}
    reference_latest = r.torch_load(history_paths(state['fold'])[1])
    raw_ref_streams = r.streams(reference_latest['rng'])
    raw_ref_epoch = reference_latest['next_epoch']
    assert r.base_runtime().rng_sha(reference_latest['rng']) == historical[raw_ref_epoch]['rng_after_eval']
    preflight = r.read_json(r.RT / 'preflight.json')['folds'][state['fold'] - 1]
    prev, previous_components = task['identity']['warmup_rng_sha'], preflight['initial_rng_streams']
    ema = initial_ema(cache)
    assert r.sha(ema) == task['initial_ema_sha']
    maximum, wait, selected, shared = -1, 0, None, 0
    ref_components_checked = False
    gradient_config = r.CFG['gain']
    for row in records:
        epoch = row['epoch']
        assert wait <= r.CFG['early_stop_wait_gt']
        p = np.unpackbits(np.frombuffer(bytes.fromhex(row['prediction_bits']), dtype=np.uint8), count=len(y))
        correct = int((p == y).sum())
        assert correct == row['correct']
        better = correct > maximum
        if better:
            maximum, wait, selected = correct, 0, epoch
        else:
            wait += 1
        assert (better, wait, selected) == (row['improved'], row['wait'], row['best_epoch'])
        assert row['rng_before'] == prev and row['rng_before_streams'] == previous_components
        assert row['rng_after_train'] == row['rng_after_eval'] == row['rng_after_ema']
        prev, previous_components = row['rng_after_ema'], row['rng_after_streams']
        if epoch in historical:
            assert {k: row[k] for k in r.RNG_KEYS} == historical[epoch]
            shared += 1
        if epoch == raw_ref_epoch:
            assert previous_components == raw_ref_streams
            ref_components_checked = True
        assert int(ema['ema_updates']) == epoch - r.START - 1 and r.sha(ema) == row['ema_before']
        obs = {'mean': torch.tensor(row['observed_mean'], dtype=torch.float64),
               'second': torch.tensor(row['observed_second'], dtype=torch.float64),
               'counts': cache['stats']['counts']}
        if epoch == best['best_epoch']:
            assert r.sha(ema) == best['calibration_sha']
            assert_ema_equal(ema, {k: best['current']['pa_dgpl'][k] for k in EMA_BUFFERS})
            assert r.sha(obs) == r.sha(best_obs) and r.sha(best_cam) == row['legal_cam_sha']
            assert row['probability_sha'] == r.sha(best['probability'])
            assert np.array_equal(p, best['prediction'])
            assert (row['rng_after_eval'], previous_components) == (best['rng_sha'], r.streams(best['rng']))
        if epoch == end:
            assert_ema_equal(ema, latest['last_eval_ema'])
        ema = ema_update(ema, obs)
        assert r.sha(ema) == row['ema_after']
        loss, coefficient = row['losses'], row['lambda_gain']
        darg = loss['cls'] + loss['graph'] + .05 * loss['pair'] + .01 * (loss['adv'] + loss['preserve'])
        assert all(math.isfinite(x) for x in loss.values())
        assert np.isclose(darg, loss['darg_total'], atol=1e-7, rtol=1e-6)
        total = loss['darg_total'] + r.CFG['residual_loss_coefficient'] * loss['res'] + coefficient * loss.get('gain', 0.)
        assert np.isclose(total, loss['total'], atol=1e-7, rtol=1e-6)
        if epoch <= r.START + 2:
            r.verify_new_gradients(row['classification'], first=epoch == r.START + 1)
        if task['group'] == 'G2':
            gradients = row['gradients']
            assert tuple(gradients) == r.PARAMETER_GROUPS
            ratios = [g['cls_norm'] / (g['gain_norm'] + gradient_config['grad_epsilon'])
                      for g in gradients.values() if g['gain_norm'] > 0]
            want = min(gradient_config['lambda_max'], gradient_config['rho'] * min(ratios)) if ratios else 0.
            assert want == coefficient and 0 <= loss['positive_gain_fraction'] <= 1
            for g in gradients.values():
                c, a = g['cls_norm'], g['gain_norm']
                measured = coefficient * a / c if c else (0. if a else None)
                assert 0 <= c and 0 <= a and g['weighted_ratio'] == measured
                assert measured is None or 0 <= measured <= gradient_config['rho'] + 1e-12
                assert (g['cosine'] is not None and abs(g['cosine']) <= 1) if (c > 0 and a > 0) else g['cosine'] is None
                assert 0 <= g['nonzero_update_fraction'] <= 1
        else:
            assert coefficient == 0 and row['gradients'] is None and 'gain' not in loss
    assert best['best_epoch'] == task['best_epoch'] == latest['best_epoch'] == selected
    assert maximum == latest['best_correct'] and wait == latest['wait']
    assert end == latest['next_epoch'] and (wait > r.CFG['early_stop_wait_gt'] or end == r.LIMIT)
    assert (prev, previous_components) == (latest['rng_sha'], r.streams(latest['rng']))
    assert records[-1]['probability_sha'] == r.sha(latest['probability'])
    assert_ema_equal(ema, {k: latest['current']['pa_dgpl'][k] for k in EMA_BUFFERS})
    return {'epochs': len(records), 'ema_updates': len(records), 'common_darg_rng_epochs': shared,
            'extended_rng_epochs': len(records) - shared, 'darg_latest_rng_components_checked': ref_components_checked}


def gradient_history(records, best_epoch):
    if records[0]['gradients'] is None:
        return {'first_classification': records[0]['classification'], 'second_classification': records[1]['classification']}
    phases = [records[0], records[best_epoch - r.START - 1], records[-1]]
    profile = {
        'first_classification': records[0]['classification'], 'second_classification': records[1]['classification'],
        'gain_loss_first_best_last': [row['losses']['gain'] for row in phases],
        'mean_gain_first_best_last': [row['losses']['mean_gain'] for row in phases],
        'positive_gain_fraction_first_best_last': [row['losses']['positive_gain_fraction'] for row in phases],
        'lambda': distribution([row['lambda_gain'] for row in records]),
        'lambda_at_cap_n': sum(row['lambda_gain'] == r.CFG['gain']['lambda_max'] for row in records),
        'groups': {},
    }
    for name in r.PARAMETER_GROUPS:
        aligned = [row['gradients'][name] for row in records]
        metrics = {}
        for field in ('cls_norm', 'gain_norm', 'weighted_ratio', 'cosine', 'nonzero_update_fraction'):
            finite = [row[field] for row in aligned if row[field] is not None]
            metrics[field] = distribution(finite) if finite else None
        profile['groups'][name] = metrics
    return profile


def audit_task(group, fold, pre):
    assert r.task_complete(group, fold)
    receipt, best_path, latest_path, log = r.paths(group, fold)
    task, best, latest = r.read_json(receipt), r.torch_load(best_path), r.torch_load(latest_path)
    for key in ('identity', 'initial_module_sha', 'initial_ema_sha', 'spectrum_sha', 'stats_sha', 'query_sha',
                'group', 'fold', 'seed', 'legal_train_sha', 'implementation'):
        assert task[key] == best[key] == latest[key], key
    assert latest['best_sha'] == task['best_sha'] and latest['log_bytes'] == log.stat().st_size
    assert not r.legacy.best_pending(best_path).exists()
    records = [r.json.loads(line) for line in log.read_text().splitlines()]
    state, module, opt, spectrum, cache, text = inputs_from_checkpoint(group, fold, best, pre)
    assert np.array_equal(best['test_indices'], state['test_np'])
    assert int(module.ema_updates) == best['best_epoch'] - r.START - 1
    best_opt = optimizer_state_audit(state, module, opt, best['best_epoch'])
    before = r.rng_mark()
    out = graph(state, module, grad=True)
    assert r.rng_mark() == before
    p = probability(out, state)
    y = state['labels_np'][state['test_np']]
    assert np.array_equal(p, best['probability']) and np.array_equal(p.argmax(-1), best['prediction'])
    metrics = independent_metrics(p, y)
    same_metrics(metrics, best['metrics'])
    formulas = check_formulas(module, out['details'])
    variants = check_interventions(state, module, text)
    result = mechanism(state, module, out, variants)
    z, obs = observe(state, out, spectrum)
    journal = audit_journal(state, task, best, latest, records, cache, z, obs)
    journal['gradient_history'] = gradient_history(records, best['best_epoch'])
    data = {'group': group, 'fold': fold, 'best_epoch': best['best_epoch'], 'end_epoch': task['end_epoch'],
            'metrics': metrics, 'rows_1based': (state['test_np'] + 1).tolist(), 'labels': y.tolist(),
            'prediction': p.argmax(-1).tolist(), 'probability': p.tolist(),
            'best_audit': {'status': 'PASS', 'probability_bitwise': True, 'formula_max_abs_errors': formulas,
                           'optimizer_slots': best_opt, 'test_indices_exact': True, 't1_t2_bitwise': True,
                           'semantic_off_swap_bitwise': True, 'fold2_protected_zero_residual': True},
            'mechanism': result, 'journal': journal, 'resumes': task['resumes']}
    del state, module, opt, out, variants, best
    state, module, opt, spectrum, _, _ = inputs_from_checkpoint(group, fold, latest, pre)
    last_opt = optimizer_state_audit(state, module, opt, latest['next_epoch'])
    complete_ema = r.legacy.calibration_state(module)
    assert int(complete_ema['ema_updates']) == task['end_epoch'] - r.START
    set_ema(module, latest['last_eval_ema'])
    terminal_eval = graph(state, module)
    assert np.array_equal(probability(terminal_eval, state), latest['probability'])
    cam, obs = observe(state, terminal_eval, spectrum)
    assert r.sha(cam) == records[-1]['legal_cam_sha']
    assert_ema_equal(complete_ema, ema_update(latest['last_eval_ema'], obs))
    set_ema(module, complete_ema)
    assert_witness(replay_step(state, module, opt, spectrum, group), task['next_step_witness'])
    data['latest_audit'] = {'status': 'PASS', 'optimizer_slots': last_opt, 'terminal_probability_bitwise': True,
                            'next_step_optimizers_eval_ema_rng_exact': True, 'final_epoch': latest['next_epoch']}
    return data, records


def protected_differences(manifest):
    expected = manifest['protected_sha256']

    def compare(pair):
        file, recorded = pair
        try:
            found = r.sha_file(file)
        except OSError as e:
            return file, {'expected': recorded, 'error': str(e)}
        return file, None if found == recorded else {'expected': recorded, 'found': found}

    with ThreadPoolExecutor(max_workers=8) as executor:
        return {p: difference for p, difference in executor.map(compare, expected.items()) if difference is not None}


def fold_summary(records):
    return {key: {'mean': float(np.mean([100 * x['metrics'][key] for x in records])),
                  'sd': float(np.std([100 * x['metrics'][key] for x in records], ddof=0))} for key in r.METRICS}


def hard_counts(y, pred):
    return (int((pred == y).sum()), len(y), int(((pred == 0) & (y == 0)).sum()), int((y == 0).sum()),
            int(((pred == 1) & (y == 1)).sum()), int((y == 1).sum()))


def fold_pair(left, right):
    from fractions import Fraction

    for key in ('rows_1based', 'labels'):
        assert left[key] == right[key], (left['group'], right['group'], left['fold'], key)
    ids, y = np.asarray(left['rows_1based']), np.asarray(left['labels'])
    current, control = (np.asarray(x['prediction']) for x in (left, right))
    a, b = hard_counts(y, current), hard_counts(y, control)
    acc = 100 * Fraction(a[0] - b[0], a[1])
    ba = 50 * (Fraction(a[2] - b[2], a[3]) + Fraction(a[4] - b[4], a[5]))
    d = {k: 100 * (left['metrics'][k] - right['metrics'][k]) for k in r.METRICS}
    assert abs(d['acc'] - float(acc)) < 1e-12 and abs(d['hard_auc'] - float(ba)) < 1e-12
    d.update(acc=float(acc), hard_auc=float(ba))
    changes = patient_changes(control, current, y, ids)
    assert changes['net'] == a[0] - b[0]
    return {'fold': left['fold'], 'delta_pp': d, 'patients': changes,
            'exact_acc': (acc.numerator, acc.denominator), 'exact_ba': (ba.numerator, ba.denominator)}


def tally(values):
    return {'wins': sum(v > 0 for v in values), 'ties': sum(v == 0 for v in values), 'losses': sum(v < 0 for v in values)}


def sum_patients(values):
    return {key: sum(row[key] for row in values) for key in
            ('asd_fixed', 'asd_harmed', 'asd_net', 'td_fixed', 'td_harmed', 'td_net', 'fixed', 'harmed', 'net', 'flips')}


def summarize_pairs(pairs):
    from fractions import Fraction

    means = {k: float(np.mean([x['delta_pp'][k] for x in pairs])) for k in r.METRICS}
    for key, field in (('acc', 'exact_acc'), ('hard_auc', 'exact_ba')):
        means[key] = float(sum(Fraction(*x[field]) for x in pairs) / len(pairs))
    return {'delta_pp': means, 'acc_wtl': tally([x['delta_pp']['acc'] for x in pairs]),
            'ba_wtl': tally([x['delta_pp']['hard_auc'] for x in pairs]), 'n_folds': len(pairs),
            'patients': sum_patients([p['patients'] for p in pairs]), 'folds': pairs}


def pooled_mechanism(tasks):
    from fractions import Fraction

    data = [row['mechanism'] for row in tasks]
    patients = [m['patients'] for m in data]
    diagnostic = [m['margin_patients'] for m in data]
    stack = {key: np.concatenate([np.asarray(p[key]) for p in patients]) for key in patients[0]}
    margin_data = {key: np.concatenate([np.asarray(p[key]) for p in diagnostic]) for key in diagnostic[0]}
    assert len(np.unique(stack['rows_1based'])) == len(stack['rows_1based']) == r.CFG['subjects']
    assert np.setdiff1d(stack['rows_1based'], margin_data['rows_1based']).tolist() == [i + 1 for i in FOLD2_ROWS]
    gate, beta, old = (stack[key] for key in ('gate', 'beta', 'beta_old'))
    y, gain, real, mgate = (margin_data[key] for key in ('labels', 'first_order_gain', 'true_margin_change', 'gate'))
    tests = {}
    for kind in r.CFG['interventions']:
        rows = [m['interventions'][kind] for m in data]
        sizes = [len(p['rows_1based']) for p in patients]
        exact_acc_difference = 100 * sum(Fraction(x['net'], n) for x, n in zip(rows, sizes)) / len(rows)
        tests[kind] = {'metrics_percent': fold_summary(rows), 'patients': sum_patients(rows),
                       'acc_delta_pp_vs_full': float(exact_acc_difference),
                       'probability_mean_abs_pp': float(sum(n * x['probability_change']['mean']
                                                           for n, x in zip(sizes, rows)) / sum(sizes) * 100),
                       'probability_max_abs_pp': 100 * max(x['probability_change']['max'] for x in rows)}
    net = tests['T1']['patients']['net']
    assert net == sum(x['interventions']['T1']['fixed'] - x['interventions']['T1']['harmed'] for x in data)
    return {
        'gate': {**distribution(gate), 'positive': int((gate > 0).sum()), 'negative': int((gate < 0).sum()),
                 'zero': int((gate == 0).sum()), 'abs': distribution(abs(gate)),
                 'one_sign_folds': [row['fold'] for row in tasks if row['mechanism']['gate']['one_sign']]},
        'beta': {**distribution(beta), 'old': distribution(old), 'change': distribution(beta - old),
                 'abs_change': distribution(abs(beta - old)),
                 'beta_saturated_n': sum(m['beta']['beta_saturated_n'] for m in data),
                 'adjustment_saturated_n': sum(m['beta']['adjustment_saturated_n'] for m in data)},
        'residual': {'pre_gate': distribution(100 * stack['relative_residual_before_gate']),
                     'applied': distribution(100 * stack['relative_residual_actual']),
                     'nonzero_n': int((stack['relative_residual_actual'] > 0).sum())},
        'margin': {name: margin_profile(mgate[indices], gain[indices], real[indices]) for name, indices in
                   [('all', np.ones(len(y), dtype=bool)), ('ASD', y == 0), ('TD', y == 1)]},
        'interventions': tests,
    }


def verdict(aggregated, audited):
    c = r.CFG['decision']
    by_group = {}
    for group in r.GROUPS:
        pair = aggregated['pairs_10'][f'{group}-DARG']
        delta = pair['delta_pp']
        m = aggregated['mechanism'][group]
        off = m['interventions']['T1']
        performance = {
            'all_folds_audited': audited,
            'acc_above_darg': delta['acc'] > 0,
            'ba_at_least_darg': delta['hard_auc'] >= 0,
            'sensitivity_within_floor': delta['sen'] >= c['sensitivity_delta_min_pp'],
            'prob_auc_within_floor': delta['prob_auc'] >= c['prob_auc_delta_min_pp'],
            'patient_net_positive': pair['patients']['net'] > 0,
        }
        mechanism = {
            'audits_and_gradients_pass': audited,
            'gate_and_residual_active': m['gate']['abs']['max'] > 0 and m['residual']['nonzero_n'] > 0,
            'residual_off_loses_correct_subject': off['patients']['harmed'] > 0,
            'residual_off_lowers_fold_mean_acc': off['acc_delta_pp_vs_full'] < 0,
            'pooled_true_margin_improves': m['margin']['all']['delta']['mean'] > 0,
        }
        by_group[group] = {'performance_checks': performance, 'mechanism_checks': mechanism,
                           'performance_go': all(performance.values()), 'mechanism_go': all(mechanism.values())}
    successful = [group for group, evidence in by_group.items() if evidence['performance_go'] and evidence['mechanism_go']]
    g2g1 = aggregated['pairs_10']['G2-G1']
    g2_effect = g2g1['delta_pp']['acc'] > 0 and g2g1['patients']['net'] > 0 and by_group['G2']['mechanism_go']
    above_darg = any(x['performance_checks']['acc_above_darg'] for x in by_group.values())
    outcome = 'GO' if successful else ('MECHANISM_PARTIAL' if g2_effect and above_darg else 'NO-GO')
    return {'verdict': outcome, 'groups': by_group, 'qualifying_groups': successful,
            'G2_paired_decision_gain': g2_effect, 'either_group_exceeds_DARG_ACC': above_darg}


def aggregates(tasks, references):
    all_rows = {(x['group'], x['fold']): x for x in [*tasks, *references]}
    groups = (*r.GROUPS, *BASELINES)
    assert set(all_rows) == {(g, f) for g in groups for f in r.FOLDS}
    combined = {}
    for folds, name in ((list(r.FOLDS), '10'), ([f for f in r.FOLDS if f != 2], '9')):
        combined[f'metrics_{name}'] = {g: fold_summary([all_rows[(g, f)] for f in folds]) for g in groups}
        combined[f'pairs_{name}'] = {f'{g}-{b}': summarize_pairs([fold_pair(all_rows[(g, f)], all_rows[(b, f)]) for f in folds])
                                     for g, b in PAIRS}
    combined['mechanism'] = {g: pooled_mechanism([all_rows[(g, f)] for f in r.FOLDS]) for g in r.GROUPS}
    g2 = [all_rows[('G2', f)]['journal']['gradient_history'] for f in r.FOLDS]
    paired_margin = []
    paired_labels = []
    for fold in r.FOLDS:
        first, second = (all_rows[(group, fold)]['mechanism']['margin_patients'] for group in r.GROUPS)
        assert first['rows_1based'] == second['rows_1based']
        assert first['labels'] == second['labels']
        paired_margin.extend(np.asarray(second['true_margin_change']) - np.asarray(first['true_margin_change']))
        paired_labels.extend(first['labels'])
    differences, y = np.asarray(paired_margin), np.asarray(paired_labels)
    combined['paired_margin_effect_G2_minus_G1'] = {
        label: {'n': int(mask.sum()), 'margin_delta': distribution(differences[mask]),
                'positive_fraction': float((differences[mask] > 0).mean())}
        for label, mask in [('all', np.ones(len(y), dtype=bool)), ('ASD', y == 0), ('TD', y == 1)]
    }
    combined['g2_training'] = {
        'loss_first_best_last': np.mean([x['gain_loss_first_best_last'] for x in g2], axis=0).tolist(),
        'gain_first_best_last': np.mean([x['mean_gain_first_best_last'] for x in g2], axis=0).tolist(),
        'positive_gain_fraction_first_best_last': np.mean([x['positive_gain_fraction_first_best_last'] for x in g2], axis=0).tolist(),
        'max_weighted_ratio': max(x['groups'][g]['weighted_ratio']['max'] for g in r.PARAMETER_GROUPS for x in g2),
        'lambda_min': min(x['lambda']['min'] for x in g2), 'lambda_max': max(x['lambda']['max'] for x in g2),
        'lambda_at_cap_n': sum(x['lambda_at_cap_n'] for x in g2),
        'epochs': sum(all_rows[('G2', f)]['journal']['epochs'] for f in r.FOLDS),
    }
    return combined


def write_tables(status):
    rows = [*status['tasks'], *status['historical_references']]
    expected = {}
    for t in rows:
        key = (t['group'], t['fold'])
        metric = independent_metrics(np.asarray(t['probability']), np.asarray(t['labels']))
        same_metrics(metric, t['metrics'])
        expected[key] = {'group': key[0], 'fold': key[1], 'best_epoch': t['best_epoch'], 'end_epoch': t['end_epoch'],
                         **{k: 100 * metric[k] for k in r.METRICS}}
    path = r.RESULTS / 'fold_metrics.csv'
    r.write_csv(path, list(expected.values()))
    loaded = csv_lookup(path)
    assert set(loaded) == set(expected)
    assert all(np.isclose(float(loaded[index][k]), want[k], atol=1e-12, rtol=0)
               for index, want in expected.items() for k in (*r.METRICS, 'best_epoch', 'end_epoch'))
    a = status['aggregates']
    pairs = []
    for comparison, rec in a['pairs_10'].items():
        for fold in rec['folds']:
            values = {k: v for k, v in fold['patients'].items() if not k.endswith('_rows_1based')}
            pairs.append({'comparison': comparison, 'scope': 'fold', 'fold': fold['fold'], **fold['delta_pp'], **values})
        for scope, data in (('10-fold', rec), ('9-fold', a['pairs_9'][comparison])):
            wins = {f'{metric}_{k}': v for metric in ('acc', 'ba') for k, v in data[f'{metric}_wtl'].items()}
            pairs.append({'comparison': comparison, 'scope': scope, 'fold': '', 'n_folds': data['n_folds'],
                          **data['delta_pp'], **data['patients'], **wins})
    path = r.RESULTS / 'paired_comparisons.csv'
    r.write_csv(path, pairs)
    with path.open(newline='') as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == len(pairs)
    for actual, expected in zip(persisted, pairs):
        assert all(actual[k] == str(v) for k, v in expected.items())
    for scope, name in (('10-fold', '10'), ('9-fold', '9')):
        actual = {row['comparison']: row for row in persisted if row['scope'] == scope}
        for g, b in PAIRS:
            delta = a[f'pairs_{name}'][f'{g}-{b}']['delta_pp']
            for metric in r.METRICS:
                folds = [f for f in r.FOLDS if name == '10' or f != 2]
                calc = np.mean([float(loaded[(g, f)][metric]) - float(loaded[(b, f)][metric]) for f in folds])
                assert np.isclose(float(actual[f'{g}-{b}'][metric]), calc, atol=1e-12, rtol=0)
                assert float(actual[f'{g}-{b}'][metric]) == delta[metric]
    return True


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|',
                      *['|' + '|'.join(map(str, row)) + '|' for row in rows]])


def performances(summary):
    return table(('组别', *DISPLAY), [
        (group, *[f"{metrics[k]['mean']:.4f} ± {metrics[k]['sd']:.4f}" for k in r.METRICS])
        for group, metrics in summary.items()])


def paired_table(pairs):
    rows = []
    for label, data in pairs.items():
        d, p = data['delta_pp'], data['patients']
        tallies = ['/'.join(map(str, data[f'{metric}_wtl'].values())) for metric in ('acc', 'ba')]
        cls = [f"{p[f'{c}_fixed']}/{p[f'{c}_harmed']}/{p[f'{c}_net']:+d}" for c in ('asd', 'td')]
        rows.append((label, *[f'{d[k]:+.4f}' for k in r.METRICS], *tallies, *cls, f"{p['fixed']}/{p['harmed']}/{p['net']:+d}"))
    return table(('配对', *DISPLAY, 'ACC 胜/平/负', 'BA 胜/平/负', 'ASD 纠错/误伤/净', 'TD 纠错/误伤/净',
                  '全部纠错/误伤/净'), rows)


def treatment_table(mechanism):
    rows = []
    for name, data in mechanism['interventions'].items():
        p = data['patients']
        m = data['metrics_percent']
        rows.append((name, r.CFG['interventions'][name], f"{data['probability_mean_abs_pp']:.6f}",
                     f"{data['probability_max_abs_pp']:.6f}", p['flips'],
                     f"{p['asd_fixed']}/{p['asd_harmed']}", f"{p['td_fixed']}/{p['td_harmed']}", f"{p['net']:+d}",
                     *[f"{m[k]['mean']:.4f}" for k in r.METRICS]))
    return table(('干预', '处理', '平均 |ΔP| pp', '最大 |ΔP| pp', '硬翻转', 'ASD 纠错/误伤', 'TD 纠错/误伤', '净纠错',
                  *DISPLAY), rows)


def format_correlation(value):
    return f'{value:.5f}' if value is not None else '未定义'


def render_report(status):
    a = status['aggregates']
    run = status['execution']
    science = status['science']
    protection = status['protection']
    lines = [
        '# Experiment 82.3.3 · PA-DGPL',
        '**Patient-Adaptive Decision-Gain Prompt Learning（患者自适应决策收益 Prompt 学习）**',
        f"状态 **{status['status']}**，科研判定 **{science['verdict']}**。",
        '研究问题是医学 Prompt 引导的固定谱证据能否经患者级正负残差提高 DARG 的硬分类正确性。'
        'E82.3.2 Medical 的门控由全折共享符号决定，实际残差约 0.064%，关闭残差零翻转，方向 KL 的辅助梯度长期偏大。',
        '## 论文与源码审查',
        '原论文的 GFR 为各模态建立患者关系图，并对共享患者特征做 GCN 消息传播。CAM 在同一患者内进行跨模态交互。'
        'PEN 通过冻结 CLIP 编码疾病文本，SPGS 用语义相似度重加权局部 token，RES 将局部线索融合到全局特征。'
        '论文端到端目标为图损失与分类交叉熵之和。原作者分类头由 LayerNorm、Linear、GELU、Dropout 和二分类 Linear 构成。 '
        '[论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · '
        '[原作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a)',
        '历史 E61/E69.3/E74.4 的实际路径为 PLNNoPEN 和原 CAM144、OutputLayer。E74.4-G2 继承 E61 epoch 10 的 CAM、优化器和 RNG，'
        '按已审计实现恢复关系校准和门控，使用 4096 个有效训练关系对与 Positive-DCR。'
        'E82.2 固定 8 个全局协方差方向和 8 个补空间 ASD−TD 对比方向。E82.3 估计合法训练类别的 Gaussian LLR。'
        'E82.3.1 引入 EMA 和分方向 LLR 加权。E82.3.2 上传提交的模型、runner、auditor、配置和报告与服务器文件逐字节匹配。 '
        '[E82.3.2 固定提交](https://github.com/tangaii/SPromptGL/tree/fdafee986b9dc5609cfb5e8e5e1f59bb156c2b6c/experiment/experiment82_3_2_da_srpe)',
        '## 冻结方法与运行',
        '两组逐折从独立恢复的同一 E74.4-G2 状态开始，逐元素继承 6081 个新分支历史初始化参数、固定医学 Query、八条提示、谱基和 EMA 状态。'
        '新增 162→16→1 signed gate 与 4→8→1 beta 网络使用隔离种子 823300+fold、823400+fold，两个末层严格为零。'
        'beta 使用旧 beta 的 logit 加上幅值至多 1 的学习修正。实际残差为 signed gate 乘历史平滑限幅谱重建。'
        '数值实现对低于 EPS 的 CAM 范数应用比例缩放，以维持字面 8% 上限。',
        'G1 保留完整历史损失 cls + graph + 0.05 pair + 0.01 adv + 0.01 preserve，增加 0.001 倍合法训练残差惩罚。'
        'G2 额外使用同一次分类前向图的真类 log 概率差的一阶 CAM 梯度，与实际残差作内积，使用困难度权重和两类等权 softplus。'
        '辅助梯度仅传入四组新增参数，系数同时受 0.001 上限和四组最小梯度比的 0.5 倍约束。'
        '分类路径端到端训练 DARG、CAM、OutputLayer 与全部新增参数，第四优化器为历史新增分支 AdamW 配置。',
        '所有折使用 seed 0、历史 CV 前 RFE 和 scaling、转导患者图和 val=test。ACC 严格提高才替换 Best，平局保留最早 epoch。'
        '训练从 epoch 11 开始，最多至 450，wait>200 早停。Best 核验评估前 EMA，Latest 每 10 轮和终止时保存评估后更新一次的 EMA。'
        'Fold2 第 377、407 行的 DARG 历史训练参与照旧，新增类统计、收益监督与标签诊断排除两行，新增残差逐元素为零。',
        f"运行目录 `{r.HERE}`。首选 OSSFS 路径返回 errno 5。GPU0 单卡，最多 {run['max_workers']} 个折进程，"
        f"每进程一个数值线程。正式任务 {run['completed_tasks']}/20，训练耗时 {run['wall_seconds']:.1f} 秒，"
        f"GPU0 采样峰值 {run['peak_sampled_gpu0_mib']} MiB，训练失败事件 {len(run['failures'])}。"
        f"会话中断 {run['session_interruptions']} 次，"
        f"提交的 Latest 精确恢复 {sum(len(x['resumes']) for x in status['tasks'])} 个任务。",
        '系统自带旧数值依赖时，历史 runner 在 Fold1 的 RNG 校验可重现失败，部分 checkpoint 需要 NumPy 的新版序列化模块。'
        '按历史记录恢复 NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1 和 pandas 3.0.6 后重新完成十折 Preflight。'
        '固定阈值与历史 RNG 校验在正式启动前均通过。',
        '## 十折与去 Fold2 性能',
        '单位为百分数，均值 ± 总体标准差。BA 对应历史 Hard-AUC。Prob-AUC 使用连续 ASD 概率，ASD=0、TD=1。',
        '### 全部十折',
        performances(a['metrics_10']),
        '### 去 Fold2 九折',
        performances(a['metrics_9']),
        '### 新组逐折',
        table(('组', '折', 'Best epoch', '终止 epoch', *DISPLAY), [
            (x['group'], x['fold'], x['best_epoch'], x['end_epoch'], *[f"{100*x['metrics'][k]:.4f}" for k in r.METRICS])
            for x in sorted(status['tasks'], key=lambda t: (t['group'], t['fold']))]),
        '## 精确患者配对',
        '差值以百分点计。硬预测按历史一基行号和真实标签逐行对齐，对照 → 新组定义纠错和误伤。',
        '### 全部十折', paired_table(a['pairs_10']),
        '### 去 Fold2 九折', paired_table(a['pairs_9']),
        '## 固定 Best 机制',
    ]
    for g, m in a['mechanism'].items():
        b, ga, re, margins = (m[k] for k in ('beta', 'gate', 'residual', 'margin'))
        margin_rows = [(name, row['n'], f"{row['delta']['mean']:+.7f}", f"{100*row['positive_fraction']:.2f}%",
                        f"{row['gain']['mean']:+.7f}", format_correlation(row['gain_to_margin_pearson']),
                        format_correlation(row['gate_to_margin_pearson']))
                       for name, row in margins.items()]
        lines.extend([
            f'### {g}',
            f"门控正/负/零 {ga['positive']}/{ga['negative']}/{ga['zero']} 人，均值 {ga['mean']:+.6f}，"
            f"绝对值 P95 {ga['abs']['p95']:.6f}，绝对值最大 {ga['abs']['max']:.6f}。"
            f"测试行仅出现单一非零符号的折 {ga['one_sign_folds']}。",
            f"beta 为 {b['mean']:.6f} ± {b['sd']:.6f}，旧公式折内同患者平均 {b['old']['mean']:.6f}，"
            f"平均绝对变化 {b['abs_change']['mean']:.6f}。beta 饱和 {b['beta_saturated_n']} 例，"
            f"logit 修正饱和 {b['adjustment_saturated_n']} 例。饱和诊断采用 beta≤0.01 或≥0.99，或 |修正|≥0.99。",
            f"门控前平均相对范数 {re['pre_gate']['mean']:.6f}%，实际施加 {re['applied']['mean']:.6f}%，"
            f"最大 {re['applied']['max']:.6f}%，非零 {re['nonzero_n']} 人。原分类头的输入钩子确认收到增强 CAM。"
            '每折门控分布、谱方向投影、每患者主导残差坐标和有符号幅度见审计 JSON。'
            'Margin 与一阶 gain 的有标签诊断排除了 Fold2 的两名重叠患者，分类指标保留原测试成员。',
            table(('诊断样本', '人数', '真类 Margin 平均变化', '改善比例', '一阶 gain 平均', 'gain 与变化 Pearson',
                   'gate 与变化 Pearson'), margin_rows),
            'T0 → 干预用于计算概率变化、硬翻转与纠错。T1 和 T2 逐元素等价。T3 在交换 Query 后逐元素不变。',
            treatment_table(m),
        ])
    g2 = a['g2_training']
    paired_margins = a['paired_margin_effect_G2_minus_G1']
    lines.extend([
        '### G2 的训练监督',
        f"首轮、Best 轮、末轮的 L_gain 折均分别为 {g2['loss_first_best_last'][0]:.8f}、"
        f"{g2['loss_first_best_last'][1]:.8f}、{g2['loss_first_best_last'][2]:.8f}。"
        f"对应的实际一阶 gain 均值为 {g2['gain_first_best_last'][0]:+.8f}、"
        f"{g2['gain_first_best_last'][1]:+.8f}、{g2['gain_first_best_last'][2]:+.8f}。"
        '困难度权重随训练 Margin 变化，因此同时检查实际 gain 与固定 Best 的真类 Margin。',
        f"lambda_gain 取值 [{g2['lambda_min']:.8g}, {g2['lambda_max']:.8g}]，"
        f"{g2['lambda_at_cap_n']}/{g2['epochs']} 轮触及预注册上限。"
        f"四组参数实际加权辅助梯度/分类梯度比的最大值为 {g2['max_weighted_ratio']:.6f}。"
        '各轮分类和收益梯度范数、余弦和参数非零更新比例见折日志。'
        f"G2−G1 的实际残差真类 Margin 效应差为 {paired_margins['all']['margin_delta']['mean']:+.7f}"
        f"（ASD {paired_margins['ASD']['margin_delta']['mean']:+.7f}，"
        f"TD {paired_margins['TD']['margin_delta']['mean']:+.7f}）。",
        '## 独立审计',
        f"Preflight {status['preflight_pass']}/10，Best {status['best_pass']}/20，Latest {status['latest_pass']}/20。"
        '独立加载逐患者概率位级重现，六指标从概率与标签复算，独立重算新模块公式并检查四优化器状态。'
        'Latest 的下一训练步经独立辅助梯度路由重放，模型、四优化器、评估概率、EMA 和完整 RNG 摘要逐项精确一致。',
        f"核对 {status['rng']['darg_common']} 个历史共同轮次、{status['rng']['darg_extension']} 个超档连续轮次、"
        f"{status['rng']['paired_overlap']} 个组间共同轮次。"
        f"在 {status['rng']['darg_terminal_component_matches']} 个可比较的 DARG 历史 Latest 终点独立核对四类 RNG。"
        f"{status['rng']['ema_updates']} 次 EMA 更新逐轮审计。"
        f"保护文件 {protection['count']} 项，SHA-256 {'全部一致' if protection['match'] else '存在差异'}。"
        '逐折 CSV、配对 CSV 和报告的统计均由审计概率重算并回读核对。',
        '## 预注册判定',
    ])
    checks = []
    for area in ('performance', 'mechanism'):
        for name in science['groups']['G1'][f'{area}_checks']:
            checks.append((area, name, *['通过' if science['groups'][g][f'{area}_checks'][name] else '未通过' for g in r.GROUPS]))
    lines.extend([
        table(('判定', '冻结项', 'G1', 'G2'), checks),
        f"**{science['verdict']}**。G1 的 Performance / Mechanism 为 "
        f"{'通过' if science['groups']['G1']['performance_go'] else '未通过'} / "
        f"{'通过' if science['groups']['G1']['mechanism_go'] else '未通过'}，G2 为 "
        f"{'通过' if science['groups']['G2']['performance_go'] else '未通过'} / "
        f"{'通过' if science['groups']['G2']['mechanism_go'] else '未通过'}。",
        '医学与匿名 Query 的比较来自固定 Best 后的即时替换。医学文本的独立重训练收益仍未知。'
        '现有估计限于历史转导图、val=test 选模和单 seed 协议。去 Fold2 九折是同一实验的敏感性分析。',
        f"独立审计耗时 {status['audit_wall_seconds']:.1f} 秒。"
        f"交接状态 **{status['handoff']['status']}**，路径 `{HANDOFF}`。",
    ])
    return '\n\n'.join(lines) + '\n'


def handoff_text(status):
    a = status['aggregates']
    science = status['science']
    results = {g: a['metrics_10'][g] for g in ('DARG', 'G1', 'G2')}
    changes = {k: a['pairs_10'][k] for k in ('G1-DARG', 'G2-DARG', 'G2-G1')}
    by_group = []
    for g, m in a['mechanism'].items():
        gate, beta, residual = m['gate'], m['beta'], m['residual']
        t = m['interventions']['T1']
        by_group.append(f"- {g} 正/负/零门控 {gate['positive']}/{gate['negative']}/{gate['zero']} 人，"
                        f"beta {beta['mean']:.5f} ± {beta['sd']:.5f}，实际相对残差 {residual['applied']['mean']:.5f}%。"
                        f"关闭残差翻转 {t['patients']['flips']} 人，净变化 {t['patients']['net']:+d} 人，"
                        f"完整模型真类 Margin 平均变化 {m['margin']['all']['delta']['mean']:+.6f}。")
    recommendation = ('若将本模块推进为论文候选，建议以独立验证检验患者净纠错和 ASD 灵敏度，并用配对匿名 Query '
                      '重训练估计医学文本的独立贡献。' if science['verdict'] == 'GO' else
                      '保留 DARG 为该协议下的比较基座。若后续继续此方向，建议以独立验证上的患者净纠错、ASD 灵敏度'
                      '及固定 Best 的实际 Margin 为决策依据。医学文本的独立贡献需要配对的匿名 Query 完整重训练。')
    return '\n\n'.join([
        '## Experiment 82.3.3 · PA-DGPL',
        '**Patient-Adaptive Decision-Gain Prompt Learning（患者自适应决策收益 Prompt 学习）**',
        f"正式训练 20/20，状态 **{status['status']}**，判定 **{science['verdict']}**。"
        f"GPU0 单卡，最多 {status['execution']['max_workers']} 个进程，seed 0 十折。",
        'E82.3.2 的方向 KL 降低，但共享符号和较小残差使决定级收益不足。'
        '本次在同一 E74.4-G2 上保持固定 16 维谱基、原医学 Query、Gaussian LLR、EMA、CAM144 与分类头。'
        'G1 采用患者级有符号门控和有界 beta，由历史分类路径端到端训练。G2 另加实际残差的一阶真类 Margin 收益监督，'
        '辅助梯度仅作用于新增参数，逐组受 0.5 比例保护。',
        performances(results),
        paired_table(changes),
        '\n'.join(by_group),
        f"G2 辅助梯度/分类梯度最大比 {a['g2_training']['max_weighted_ratio']:.6f}。"
        'T1/T2 逐元素等价，关闭语义后替换 Query 逐元素不变。医学语义比较限于固定 Best 的即时效应。',
        f"Preflight {status['preflight_pass']}/10，Best {status['best_pass']}/20，Latest {status['latest_pass']}/20。"
        f"重放 EMA {status['rng']['ema_updates']} 次，历史共同 RNG {status['rng']['darg_common']} 轮，组间共同 "
        f"{status['rng']['paired_overlap']} 轮，保护 SHA-256 {status['protection']['count']} 项全部一致。"
        'Fold2 行 377、407 继续参与历史 DARG 路径，新增监督与统计排除，两行残差为零。',
        f"实验目录 `{r.HERE}`。 [正式报告]({r.RESULTS / 'report.md'}) · "
        f"[独立审计]({r.RESULTS / 'audit_status.json'})",
        recommendation,
    ]) + '\n'


def append_handoff(status, manifest):
    section = handoff_text(status).encode('utf-8')
    marker = b'## Experiment 82.3.3 \xc2\xb7 PA-DGPL'
    before = manifest['handoff_before']
    try:
        with HANDOFF.open('r+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            original = handle.read()
            prefix = original[:before['size_bytes']]
            assert hashlib.sha256(prefix).hexdigest() == before['sha256'], 'historical handoff prefix changed'
            if marker in original:
                assert section in original and original.count(marker) == 1
            else:
                handle.seek(0, os.SEEK_END)
                handle.write(b'\n\n' + section)
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            updated = handle.read()
            assert updated[:len(original)] == original and section in updated and updated.count(marker) == 1
        # Read via a fresh handle to verify the persisted append.
        assert HANDOFF.read_bytes() == updated
        return {'status': 'APPENDED_VERIFIED', 'path': str(HANDOFF), 'sha256': hashlib.sha256(updated).hexdigest(),
                'section_sha256': hashlib.sha256(section).hexdigest(), 'prefix_unchanged': True}
    except (OSError, AssertionError) as exc:
        pending = r.RESULTS / 'handoff_pending.md'
        with r.legacy.atomic_writer(pending, 'wb') as handle:
            handle.write(section)
        return {'status': 'BLOCKED', 'path': str(HANDOFF), 'pending': str(pending), 'error': repr(exc)}


def audit_all():
    r.assert_gpu()
    wall = time.monotonic()
    pre = r.passing_preflight()
    manifest = r.read_json(r.HERE / 'manifest.json')
    execution = r.read_json(r.RT / 'execution.json')
    assert execution['completed_tasks'] == r.CFG['expected_tasks']
    assert execution['status'] == 'TRAINED_PENDING_INDEPENDENT_AUDIT'
    interruptions = manifest.get('training_segments_before_resume', [])
    segments = [*interruptions, execution]
    execution = {**execution, 'wall_seconds': sum(s['wall_seconds'] for s in segments),
                 'max_workers': max(s['max_workers'] for s in segments),
                 'peak_sampled_gpu0_mib': max(s['peak_sampled_gpu0_mib'] for s in segments),
                 'failures': [f for s in segments for f in s['failures']],
                 'session_interruptions': len(interruptions), 'training_segments': segments}
    assert r.sha_file(r.HERE / 'config.json') == manifest['config_sha256']
    darg_csv = csv_lookup(r.legacy.runtime().EXP / 'results/fold_metrics.csv')
    dasrpe_csv = csv_lookup(r.PREVIOUS / 'results/fold_metrics.csv')
    ema_csv = csv_lookup(r.legacy.E831 / 'results/fold_metrics.csv')
    tasks, references, failures = [], [], []
    paired = 0
    status = {}
    r.update_manifest(status='AUDITING')
    with ThreadPoolExecutor(max_workers=1) as executor:
        guard = executor.submit(protected_differences, manifest)
        for fold in r.FOLDS:
            rows = {}
            for group in r.GROUPS:
                try:
                    result, records = audit_task(group, fold, pre)
                    tasks.append(result)
                    rows[group] = records
                    print(f'AUDIT PASS {group} f{fold:02d} Best+Latest', flush=True)
                except Exception as e:
                    fail = {'group': group, 'fold': fold, 'error': repr(e), 'traceback': traceback.format_exc()}
                    failures.append(fail)
                    print(f"AUDIT FAIL {group} f{fold:02d} {fail['traceback']}", flush=True)
            try:
                if set(rows) == set(r.GROUPS):
                    for first, second in zip(rows['G1'], rows['G2']):
                        keys = (*r.RNG_KEYS, 'rng_before_streams', 'rng_after_streams')
                        assert all(first[k] == second[k] for k in keys)
                        paired += 1
                state, _, identity = r.legacy.prepare(fold)
                for g, csv_rows in zip(BASELINES, (darg_csv, dasrpe_csv, ema_csv, ema_csv)):
                    references.append(reference_fold(state, identity, g, csv_rows))
            except Exception as e:
                failures.append({'group': 'reference_or_pair', 'fold': fold, 'error': repr(e), 'traceback': traceback.format_exc()})
                print(f'AUDIT REFERENCE/PAIR FAIL f{fold:02d} {e!r}', flush=True)
        differences = guard.result()
    if differences:
        failures.append({'group': 'protected_files', 'differences': differences})
    audit_complete = len(tasks) == r.CFG['expected_tasks'] and len(references) == len(BASELINES) * len(r.FOLDS) and not failures
    status.update(status='COMPLETE_AUDITED' if audit_complete else 'PARTIAL', execution=execution,
                  preflight_pass=len(pre['folds']), best_pass=len(tasks), latest_pass=len(tasks), tasks=tasks,
                  historical_references=references, failures=failures,
                  protection={'count': manifest['protection_count'], 'match': not differences, 'differences': differences},
                  rng={'paired_overlap': paired,
                       'darg_common': sum(x['journal']['common_darg_rng_epochs'] for x in tasks),
                       'darg_extension': sum(x['journal']['extended_rng_epochs'] for x in tasks),
                       'darg_terminal_component_matches': sum(x['journal']['darg_latest_rng_components_checked'] for x in tasks),
                       'ema_updates': sum(x['journal']['ema_updates'] for x in tasks)},
                  implementation={**r.implementation(), 'audit.py': r.sha_file(r.HERE / 'audit.py')},
                  audit_wall_seconds=time.monotonic() - wall)
    if not audit_complete:
        r.atomic_json(r.RESULTS / 'audit_status.json', status)
        r.update_manifest(status='PARTIAL', best_audit_pass=len(tasks), latest_audit_pass=len(tasks), audit_failures=failures)
        raise RuntimeError(f'independent audit incomplete with {len(failures)} failed checks')
    status['aggregates'] = aggregates(tasks, references)
    status['science'] = verdict(status['aggregates'], audit_complete)
    status['tables_recomputed_and_verified'] = write_tables(status)
    # Save sufficient evidence before touching the shared handoff.
    status['handoff'] = {'status': 'PENDING', 'path': str(HANDOFF)}
    r.atomic_json(r.RESULTS / 'audit_status.json', status)
    report = r.RESULTS / 'report.md'
    with r.legacy.atomic_writer(report, 'w') as handle:
        handle.write(render_report(status))
    status['handoff'] = append_handoff(status, manifest)
    if status['handoff']['status'] != 'APPENDED_VERIFIED':
        status['status'] = 'PARTIAL'
    status['audit_wall_seconds'] = time.monotonic() - wall
    status['elapsed_since_protection_seconds'] = time.time() - manifest['protection_started_unix']
    report_text = render_report(status)
    with r.legacy.atomic_writer(report, 'w') as handle:
        handle.write(report_text)
    assert report.read_text() == report_text
    r.atomic_json(r.RESULTS / 'audit_status.json', status)
    assert r.read_json(r.RESULTS / 'audit_status.json')['science']['verdict'] == status['science']['verdict']
    artifacts = {name: r.sha_file(r.RESULTS / name)
                 for name in ('report.md', 'fold_metrics.csv', 'paired_comparisons.csv', 'audit_status.json')}
    r.update_manifest(status=status['status'], best_audit_pass=len(tasks), latest_audit_pass=len(tasks),
                      scientific_verdict=status['science']['verdict'], handoff=status['handoff'],
                      verified_protection=status['protection'], result_sha256=artifacts,
                      audit_wall_seconds=status['audit_wall_seconds'],
                      elapsed_since_protection_seconds=status['elapsed_since_protection_seconds'])
    print(status['status'], status['science']['verdict'], status['handoff']['status'], flush=True)


if __name__ == '__main__':
    audit_all()
