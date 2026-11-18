#!/usr/bin/env python3
"""Independent DA-SRPE checkpoint replay, paired analysis and handoff."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import fcntl
import math
import os
import time
import traceback
from pathlib import Path

import run as r
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F

from da_srpe import EMA_BUFFERS, EPS, FOLD2_ROWS, QUERIES, SPECTRUM_BUFFERS

METRICS = r.METRICS
DISPLAY = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')
BASELINES = ('B0', 'B1', 'B2')
COMPARISONS = (('G1', 'B0'), ('G2', 'B0'), ('G2', 'G1'), ('G1', 'B1'), ('G2', 'B2'))
TREATMENTS = {'T0': 'full', 'T1': 'full', 'T2': 'semantic_off', 'T3': 'residual_off', 'T4': 'gate_off'}
HANDOFF = Path('/model/spromptgl/experiment/handoff.md')


def independent_metrics(probability, labels):
    asd = labels == 0
    pred = probability.argmax(-1)
    return {'acc': float(accuracy_score(labels, pred)), 'hard_auc': float(balanced_accuracy_score(labels, pred)),
            'prob_auc': float(roc_auc_score(asd, probability[:, 0])),
            'sen': float(recall_score(labels, pred, pos_label=0)), 'spe': float(recall_score(labels, pred, pos_label=1)),
            'asd_f1': float(f1_score(labels, pred, pos_label=0, zero_division=0))}


def probability(out, state):
    return out['logp'].detach().exp()[state['test_np']].cpu().numpy()


def same_metrics(actual, expected):
    assert all(abs(actual[key] - expected[key]) < 1e-12 for key in METRICS), (actual, expected)


def independent_forward(state, module, intervention='full', *, training=False, grad=False):
    module.train(training)
    captured = []
    legal_np = np.setdiff1d(state['train_np'], state['test_np'], assume_unique=True)
    assert np.array_equal(legal_np, r.legal_train(state))

    def replacement(_head, args):
        assert len(args) == 2 and args[0].shape == (871, 144)
        z = args[0]
        before = r.rng_sha()
        changed, details = module(z, intervention=intervention)
        assert before == r.rng_sha()
        captured.append(details)
        return changed, args[1]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(replacement)
    try:
        with r.count_calls(state), torch.set_grad_enabled(training or grad):
            out = r.runtime().forward(state, r.BASE_GROUP, training=training)
    finally:
        handle.remove()
    assert len(captured) == 1
    d = captured[0]
    out['details'] = d
    if training or grad:
        rows = torch.as_tensor(legal_np, device=d['z'].device)
        labels = state['y'][rows]
        before = r.rng_sha()
        target, degenerate, v, contribution = independent_direction(state, module, out, rows)
        weights = module.prompt_aux(d['z'].detach()[rows])
        row_kl = (target * (target.clamp_min(EPS).log() - weights.clamp_min(EPS).log())).sum(-1)
        align = (row_kl[labels == 0].mean() + row_kl[labels == 1].mean()) * .5
        delta, z = d['delta'][rows], d['z'][rows]
        res = (delta.square().sum(-1) / (z.square().sum(-1) + EPS)).mean()
        out.update(align=align, res=res, darg_total=out['total'], aux_weights=weights, direction_target=target,
                   degenerate=degenerate, margin_derivative=v, spectral_contribution=contribution)
        out['total'] = out['total'] + .001 * res + .001 * align
        assert r.rng_sha() == before
    return out


def independent_direction(state, module, out, rows):
    details = out['details']
    labels = state['y'][rows]
    # Gather the real and opposing class without relying on the runner's label-sign implementation.
    selected = out['logp'][rows]
    correct = selected.gather(1, labels[:, None]).flatten()
    other = selected.gather(1, (1-labels)[:, None]).flatten()
    v, = torch.autograd.grad((correct - other).sum(), details['z_enhanced'], retain_graph=True, create_graph=False)
    with torch.no_grad():
        v = v[rows]
        contribution = (v @ module.U) * details['h'][rows]
        constant = contribution.amax(-1) == contribution.amin(-1)
        centered = contribution - contribution.mean(-1, keepdim=True)
        rms = centered.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        p = torch.where(constant[:, None], 0., (centered / rms).clamp(-3, 3)).softmax(-1)
        assert all(torch.isfinite(x).all() for x in (v, contribution, p))
    return p, constant, v, contribution


def independent_stats(state, warm, identity, spectrum, cache):
    idx = np.setdiff1d(state['train_np'], state['test_np'], assume_unique=True)
    labels = state['labels_np'][idx]
    assert np.array_equal(idx, cache['legal_train']) and np.array_equal(idx, warm['train_indices'])
    assert cache['scope']['identity'] == spectrum['identity'] == identity
    assert cache['scope']['legal_train_sha'] == r.sha(idx) and cache['scope']['legal_label_sha'] == r.sha(torch.as_tensor(labels))
    rng = r.rng_sha()
    z, out = r.warmup_cam(state)
    assert r.rng_sha() == rng and torch.equal(z, spectrum['warmup_cam'])
    assert spectrum['warmup_graph_sha'] == [r.sha(a) for a in out['original_adjs']]
    assert spectrum['warmup_unified_graph_sha'] == [r.sha(a) for a in out['unified_adjs']]
    z = z.numpy()[idx].astype(np.float64)
    s = spectrum['spectrum']
    u, mu, lam = (s[k].numpy() for k in ('U', 'mu', 'lambda'))
    np.testing.assert_allclose(u.T @ u, np.eye(16), rtol=0, atol=1e-10)
    x = (z - mu) @ u
    x0, x1 = x[labels == 0], x[labels == 1]
    np.testing.assert_allclose(mu, z.mean(0), atol=2e-12, rtol=0)
    np.testing.assert_allclose(lam, x.var(0, ddof=1), atol=2e-12, rtol=1e-10)
    global_dirs, disease_dirs = u[:, :8], u[:, 8:]
    cov = np.cov(z, rowvar=False)
    np.testing.assert_allclose(lam[:8], np.linalg.eigvalsh(cov)[-8:][::-1], rtol=1e-8, atol=1e-9)
    projector = np.eye(144) - global_dirs @ global_dirs.T
    contrast = (np.cov(z[labels == 0], rowvar=False) - np.cov(z[labels == 1], rowvar=False)) if not s['fallback'] else cov
    target = projector @ contrast @ projector
    values = np.diag(disease_dirs.T @ target @ disease_dirs)
    assert np.linalg.norm(target @ disease_dirs - disease_dirs * values) < 1e-8 * max(1, np.linalg.norm(target))
    np.testing.assert_allclose(np.sort(abs(values)), np.sort(abs(np.linalg.eigvalsh(target)))[-8:], atol=1e-8, rtol=1e-7)
    cfg = r.CFG['class_statistics']
    norm = np.sqrt(lam + EPS)
    split = (x0 / norm, x1 / norm)
    counts = np.array(list(map(len, split)))
    means = np.stack([values.mean(0) for values in split])
    variance = np.stack([values.var(0, ddof=cfg['variance_ddof']) for values in split])
    pooled = np.average(variance, axis=0, weights=counts)
    stable = cfg['own_variance'] * variance + cfg['pooled_variance'] * pooled + np.maximum(cfg['variance_floor'], cfg['relative_floor'] * pooled)
    among = {'counts': counts, 'mean': means, 'variance': variance, 'pooled': pooled, 'stable': stable}
    errors = {}
    for key, value in among.items():
        got = cache['stats'][key].numpy()
        np.testing.assert_allclose(got, value, rtol=1e-10, atol=2e-12)
        errors[key] = float(np.max(abs(got - value)))
    assert r.sha(cache['stats']) == cache['stats_sha']
    return {'independent_fit_max_abs_errors': errors, 'class_counts_asd_td': counts.tolist(),
            'legal_rows_equal_original_disjoint_training': True, 'spectra_and_warmup_graph_verified': True}


def independent_initial_ema(cache):
    base = cache['stats']
    count = base['counts'][:, None]
    second = base['variance'] * ((count - 1) / count.double()) + base['mean'].square()
    return {'ema_mean': base['mean'], 'ema_second': second, 'ema_counts': base['counts'],
            'ema_updates': torch.zeros((), dtype=torch.long), 'class_mean': base['mean'].float(),
            'class_var': base['stable'].float()}


def independent_observation(state, out, spectrum):
    legal = np.setdiff1d(state['train_np'], state['test_np'], assume_unique=True)
    labels = state['labels_np'][legal]
    z = out['details']['z'].detach()[legal].cpu()
    sp = spectrum['spectrum']
    hh = torch.matmul(z.double() - sp['mu'], sp['U']).div((sp['lambda'] + EPS).sqrt())
    asd, td = hh[labels == 0], hh[labels == 1]
    return z, {'counts': torch.tensor([len(asd), len(td)], dtype=torch.long),
               'mean': torch.stack([asd.mean(dim=0), td.mean(dim=0)]),
               'second': torch.stack([asd.square().mean(dim=0), td.square().mean(dim=0)])}


def independent_ema_step(previous, observation):
    count = previous['ema_counts']
    assert torch.equal(observation['counts'], count)
    alpha = r.CFG['ema']['alpha']
    assert alpha == .1
    mean = previous['ema_mean'] * .9 + observation['mean'] * .1
    second = previous['ema_second'] * .9 + observation['second'] * .1
    numerator = (second - mean.square()).clamp_min(0)
    variance = numerator * (count.double() / (count - 1))[:, None]
    pooled = (count[:, None] * variance).sum(dim=0) / count.sum()
    floor = torch.maximum(.01 * pooled, torch.full_like(pooled, 1e-4))
    scaled = .9 * variance + .1 * pooled + floor
    assert torch.isfinite(scaled).all() and (scaled > 0).all()
    return {'ema_counts': count, 'ema_mean': mean, 'ema_second': second, 'class_mean': mean.float(),
            'class_var': scaled.float(), 'ema_updates': previous['ema_updates'] + 1}


def observation_from_journal(item):
    return {'counts': torch.tensor(item['counts_asd_td'], dtype=torch.long),
            'mean': torch.tensor(item['observed_mean'], dtype=torch.float64),
            'second': torch.tensor(item['observed_second'], dtype=torch.float64)}


@torch.no_grad()
def restore_ema(module, ema):
    for name, value in ema.items():
        getattr(module, name).copy_(value)


@torch.no_grad()
def independent_formula_check(module, d):
    z = d['z']
    h = (z - module.mu) @ module.U
    hh = h / (module.lam + EPS).sqrt()
    loglam, nu = module.loglam.expand_as(hh), module.nu.expand_as(hh)
    token_e = torch.stack((hh, hh.abs(), loglam, nu), -1)
    p = module
    tokens = F.layer_norm(F.linear(token_e, p.evidence.weight, p.evidence.bias) +
                          F.linear(p.U.T, p.direction.weight, p.direction.bias)[None],
                          (32,), p.norm.weight, p.norm.bias, p.norm.eps)
    keys = F.linear(tokens, p.key.weight)
    scores = (keys @ p.q0.T).transpose(1, 2) / math.sqrt(32)
    nulls = (p.q0 @ p.null / math.sqrt(32))[None, :, None].expand(len(h), -1, -1)
    a = torch.cat((scores, nulls), dim=-1).div(r.CFG['temperature']).softmax(-1)
    # Explicit class log densities, including the constant here to independently verify cancellation.
    dev = hh[:, None, :] - p.class_mean[None, :, :]
    log_density = -.5 * (math.log(2 * math.pi) + p.class_var.log()[None] + dev.square() / p.class_var[None])
    raw = log_density[:, 0] - log_density[:, 1]
    bounded = p.llr_bound * torch.tanh(raw / p.llr_bound)
    coverage = 1 - (1 - a[:, :, :16].reshape(len(a), 2, 4, 16)).prod(2)
    prior = coverage / (coverage.sum(-1, keepdim=True) + EPS)
    evidence = torch.stack((torch.logaddexp(torch.zeros_like(bounded), -bounded),
                            torch.logaddexp(torch.zeros_like(bounded), bounded)), dim=1)
    s1 = (prior[:, 0] * evidence[:, 0]).sum(-1)
    s0 = (prior[:, 1] * evidence[:, 1]).sum(-1)
    gated = coverage * evidence
    weights = gated / (gated.sum(-1, keepdim=True) + EPS)
    class_scores = torch.stack((s0, s1), 1)
    pi = class_scores.softmax(-1)
    ws = (pi.flip(-1)[:, :, None] * weights).sum(1)
    spec = p.spectral
    wp = F.linear(F.gelu(F.linear(torch.stack([hh.abs(), loglam, nu], -1), spec[0].weight, spec[0].bias)),
                  spec[2].weight, spec[2].bias).squeeze(-1).softmax(-1)
    entropy = -(ws * (ws + EPS).log()).sum(-1) / math.log(16)
    null = a[:, :, 16].mean(-1)
    beta = ((1 - null) * (1 - entropy).clamp(.2, .8))
    w = wp + beta[:, None] * (ws - wp)
    unbounded = ((w - 1/16) * h) @ p.U.T
    limit = .08 * z.norm(dim=-1, keepdim=True).clamp_min(EPS)
    relative = unbounded.norm(dim=-1, keepdim=True) / limit
    scale = torch.where(relative == 0, torch.ones_like(relative), relative.tanh() / relative.clamp_min(1e-30))
    delta_base = unbounded * scale
    if module.fold == 2:
        delta_base[list(FOLD2_ROWS)] = 0
        assert torch.count_nonzero(d['delta'][list(FOLD2_ROWS)]) == 0
    gate_net = p.patient_gate
    normalized = F.layer_norm(z, (144,), eps=r.CFG['patient_gate']['affine_free_layer_norm_eps'])
    gate_input = torch.cat((normalized, ws - wp, beta[:, None], null[:, None]), dim=1)
    gate_strength = F.linear(F.gelu(F.linear(gate_input, gate_net[0].weight, gate_net[0].bias)),
                             gate_net[2].weight, gate_net[2].bias).flatten().sigmoid()
    gate = p.gamma.tanh() * gate_strength
    delta = gate[:, None] * delta_base
    tracked = {'h': h, 'hhat': hh, 'a': a, 'tokens': tokens, 'raw_llr': raw, 'bounded_llr': bounded,
               'c_n': coverage[:, 0], 'c_a': coverage[:, 1], 'p_n': prior[:, 0], 'p_a': prior[:, 1],
               'g_td': evidence[:, 0], 'g_asd': evidence[:, 1], 'w_n': weights[:, 0], 'w_a': weights[:, 1],
               'scores': class_scores, 'pi': pi, 'w_sem': ws, 'w_sp': wp, 'entropy': entropy, 'null': null,
               'beta': beta, 'w': w, 'delta_raw': unbounded, 'delta': delta, 'delta_base': delta_base,
               'z_enhanced': z + delta, 'gate_input': gate_input, 'gate_strength': gate_strength, 'gate': gate}
    errors = {}
    for key, value in tracked.items():
        torch.testing.assert_close(value, d[key], rtol=2e-5, atol=5e-6)
        errors[key] = float((value - d[key]).abs().max())
    assert not d['beta'].requires_grad
    return errors


def distribution(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    assert x.size and np.isfinite(x).all()
    q = np.quantile(x, [0, .05, .25, .5, .75, .95, 1])
    return {'mean': float(x.mean()), 'population_sd': float(x.std()),
            **dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'), map(float, q)))}


def changes(reference, candidate, labels):
    result = {}
    for name, scope in [('all', np.ones_like(labels, dtype=bool)), ('asd', labels == 0), ('td', labels == 1)]:
        before, after = reference[scope] == labels[scope], candidate[scope] == labels[scope]
        corrected, harmed = int((~before & after).sum()), int((before & ~after).sum())
        result[name] = {'flips': corrected + harmed, 'corrected': corrected, 'harmed': harmed, 'net': corrected - harmed}
    return result


def llr_profile(raw, bounded, labels):
    return {'raw': distribution(raw), 'bounded': distribution(bounded),
            'raw_by_class': {name: distribution(raw[labels == code]) for code, name in enumerate(('ASD', 'TD'))},
            'bounded_by_class': {name: distribution(bounded[labels == code]) for code, name in enumerate(('ASD', 'TD'))},
            'raw_abs_gt_8_share': float((np.abs(raw) > 8).mean()),
            'bounded_abs_ge_7_6_share': float((np.abs(bounded) >= 7.6).mean()),
            'abs_smoothing': distribution(np.abs(raw - bounded)),
            'raw_mean_score_roc_auc': float(roc_auc_score(labels == 0, raw.mean(1))),
            'bounded_mean_score_roc_auc': float(roc_auc_score(labels == 0, bounded.mean(1))),
            'raw_auc_by_direction': [float(roc_auc_score(labels == 0, raw[:, j])) for j in range(16)],
            'bounded_auc_by_direction': [float(roc_auc_score(labels == 0, bounded[:, j])) for j in range(16)]}


def routing_profile(value):
    top1 = np.bincount(value.argmax(1), minlength=16) / len(value)
    entropy = -(value * np.log(value.clip(1e-8))).sum(1)
    return {'mean_by_direction': value.mean(0).tolist(), 'sd_by_direction': value.std(0).tolist(),
            'patient_variation_l1': float(np.abs(value - value.mean(0)).sum(1).mean()),
            'top1_by_direction': top1.tolist(), 'top1_mode_share': float(top1.max()),
            'top1_fixed': bool(np.count_nonzero(top1) == 1), 'effective_direction_count': distribution(np.exp(entropy))}


def finish_ce(scores, y):
    p = torch.from_numpy(scores)
    target = torch.as_tensor(y)
    losses = F.cross_entropy(p, target, reduction='none').numpy()
    asd = y == 0
    margins = np.where(asd, scores[:, 0] - scores[:, 1], scores[:, 1] - scores[:, 0])
    return {'balanced_ce': float(0.5 * (losses[asd].mean() + losses[~asd].mean())),
            'ce_by_class': [float(losses[asd].mean()), float(losses[~asd].mean())],
            'balanced_accuracy': float(balanced_accuracy_score(y, scores.argmax(-1))),
            'scores_roc_auc': float(roc_auc_score(asd, scores[:, 0] - scores[:, 1])),
            's_asd': distribution(scores[:, 0]), 's_td': distribution(scores[:, 1]),
            'true_class_margin': distribution(margins), 'true_class_win_share': float((margins > 0).mean()),
            'true_class_margin_by_class': [distribution(margins[asd]), distribution(margins[~asd])]}


def cosine(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)).clip(1e-12)


def decision_direction_profile(state, module, previous, out, ids, *, legal_train):
    y = state['labels_np'][ids]
    d = out['details']
    rows = torch.as_tensor(ids, device=d['z'].device)
    if legal_train:
        target, constant, derivative, contributions = (out[k] for k in (
            'direction_target', 'degenerate', 'margin_derivative', 'spectral_contribution'))
    else:
        # Outcome labels enter a retrospective diagnostic after fixed prediction.
        target, constant, derivative, contributions = independent_direction(state, module, out, rows)
    with torch.no_grad():
        previous_weights = previous(d['z'].detach())[1]['w_sem'][rows]
        explicit = (derivative[:, None] * (d['h'][rows, :, None] * module.U.T[None])).sum(-1)
        torch.testing.assert_close(contributions, explicit, atol=1e-6, rtol=2e-5)
        profiles = {}
        weights = {'w_sem': d['w_sem'][rows], 'w_sp': d['w_sp'][rows], 'e8231_on_same_cam': previous_weights}
        for key, w in weights.items():
            kl = (target * (target.clamp_min(EPS).log() - w.clamp_min(EPS).log())).sum(-1).cpu().numpy()
            defined = ~constant
            top1 = (w.argmax(-1) == target.argmax(-1))
            profiles[key] = {'balanced_kl': float((kl[y == 0].mean() + kl[y == 1].mean()) / 2),
                             'spearman': r.direction_rank(target, w, y),
                             'top1_share_nondegenerate': float(top1[defined].float().mean()) if defined.any() else None}
        profiles['target'] = routing_profile(target.cpu().numpy())
        profiles['degenerate_share'] = float(constant.float().mean())
        profiles['margin_derivative_l2'] = distribution(derivative.norm(dim=-1).cpu().numpy())
        profiles['abs_direction_contribution'] = distribution(contributions.abs().cpu().numpy())
        profiles['w_sem_vs_e8231_l1'] = distribution((weights['w_sem'] - previous_weights).abs().sum(-1).cpu().numpy())
        profiles['n'] = len(ids)
    return profiles


def pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mechanism(state, module, cache, spectrum, details, out, previous):
    legal = cache['legal_train']
    y_full = state['labels_np']
    plain = {key: value.detach().cpu().numpy() for key, value in details.items()}
    head = state['cam'].Outputlayer
    with torch.no_grad():
        norm, linear, _, dropout = head.mlp_head
        assert not dropout.training
        hidden = F.gelu(F.linear(F.layer_norm(details['z'], norm.normalized_shape, norm.weight, norm.bias, norm.eps),
                                 linear.weight, linear.bias))
        bare = F.linear(hidden, head.classifier.weight, head.classifier.bias)
        margin_gain_asd = ((out['logp'][:, 0] - out['logp'][:, 1]) - (bare[:, 0] - bare[:, 1])).cpu().numpy()
    denom = np.linalg.norm(plain['z'], axis=1).clip(EPS)
    rel = np.linalg.norm(plain['delta'], axis=1) / denom
    rel_base = np.linalg.norm(plain['delta_base'], axis=1) / denom
    warm_stats = cache['stats']
    hh_best = details['hhat'].detach().cpu()
    warm_mean, warm_stable = (warm_stats[key].float() for key in ('mean', 'stable'))
    warm_log_density = -.5 * (warm_stable.log() + (hh_best[:, None] - warm_mean).square() / warm_stable)
    fixed_raw = warm_log_density[:, 0] - warm_log_density[:, 1]
    fixed_bounded = module.llr_bound * torch.tanh(fixed_raw / module.llr_bound)
    result = {'gamma': float(module.gamma.detach()),
              'test_patient_gate': [{'row_1based': int(i+1), 'gate': float(plain['gate'][i]),
                                     'actual_relative_residual': float(rel[i])} for i in state['test_np']]}
    for split, ids in [('legal_train', legal), ('test', state['test_np'])]:
        d = {key: value[ids] for key, value in plain.items()}
        labels = y_full[ids]
        attn = d['a'][:, :, :16]
        attn_norm = attn / np.linalg.norm(attn, axis=-1, keepdims=True).clip(1e-12)
        pair = (attn_norm @ attn_norm.transpose(0, 2, 1)).mean(0)
        gain = np.where(labels == 0, 1, -1) * margin_gain_asd[ids]
        original_margin = np.where(labels == 0, 1, -1) * (bare[:, 0] - bare[:, 1]).cpu().numpy()[ids]
        gate = d['gate']
        direction_ids = ids[~np.isin(ids, FOLD2_ROWS)] if state['fold'] == 2 else ids
        result[split] = {
            'n': len(ids), 'coverage_n_mean_by_direction': d['c_n'].mean(0).tolist(),
            'coverage_a_mean_by_direction': d['c_a'].mean(0).tolist(),
            'coverage_n': distribution(d['c_n']), 'coverage_a': distribution(d['c_a']),
            'routing': {key: routing_profile(d[key]) for key in ('p_n', 'p_a', 'w_n', 'w_a', 'w_sem', 'w_sp', 'w')},
            'g_n': distribution(d['g_td']), 'g_a': distribution(d['g_asd']),
            'gate_n_shift_l1': distribution(np.abs(d['w_n'] - d['p_n']).sum(1)),
            'gate_a_shift_l1': distribution(np.abs(d['w_a'] - d['p_a']).sum(1)),
            'attention_mean_8x17': d['a'].mean(0).tolist(), 'prompt_cosine_8x8': pair.tolist(),
            'prompt_pairwise_cosine': float(pair[~np.eye(8, dtype=bool)].mean()),
            'null_mean_each_prompt': d['a'][:, :, 16].mean(0).tolist(), 'null': distribution(d['null']),
            'within_group_entropy': {key: distribution(-(d[key] * np.log(d[key].clip(EPS))).sum(1) / math.log(16)) for key in ('w_n', 'w_a')},
            'entropy': distribution(d['entropy']), 'pi_asd': distribution(d['pi'][:, 0]), 'pi_td': distribution(d['pi'][:, 1]),
            'pi_asd_by_class': [distribution(d['pi'][labels == code, 0]) for code in (0, 1)],
            'prompt_ce': finish_ce(d['scores'], labels), 'llr': llr_profile(d['raw_llr'], d['bounded_llr'], labels),
            'frozen_warmup_llr_same_best_cam': llr_profile(fixed_raw[ids].numpy(), fixed_bounded[ids].numpy(), labels),
            'w_n_w_a_l1': distribution(np.abs(d['w_n'] - d['w_a']).sum(1)),
            'w_n_w_a_cosine': distribution(cosine(d['w_n'], d['w_a'])),
            'w_sem_w_sp_l1': distribution(np.abs(d['w_sem'] - d['w_sp']).sum(1)),
            'w_sem_w_sp_cosine': distribution(cosine(d['w_sem'], d['w_sp'])),
            'beta': distribution(d['beta']), 'relative_residual': distribution(rel[ids]),
            'relative_residual_before_gate': distribution(rel_base[ids]),
            'gate': distribution(gate), 'abs_gate': distribution(np.abs(gate)), 'gate_strength': distribution(d['gate_strength']),
            'gate_positive_n': int((gate > 0).sum()), 'gate_negative_n': int((gate < 0).sum()),
            'gate_zero_n': int((gate == 0).sum()),
            'abs_gate_vs_bare_true_margin_pearson': pearson(np.abs(gate), original_margin),
            'signed_gate_vs_true_margin_gain_pearson': pearson(gate, gain),
            'true_logit_margin_change': distribution(gain),
            'true_margin_increased_share': float((gain > 0).mean()),
            'true_logit_margin_change_by_class': [distribution(gain[labels == code]) for code in (0, 1)],
            'decision_direction': decision_direction_profile(state, module, previous, out, direction_ids,
                                                             legal_train=split == 'legal_train'),
        }
    sp = spectrum['spectrum']
    u, mu, lam = [sp[key].numpy() for key in ('U', 'mu', 'lambda')]
    current_hh = ((plain['z'][legal].astype(np.float64) - mu) @ u) / np.sqrt(lam + EPS)
    current = [current_hh[y_full[legal] == code] for code in (0, 1)]
    mean = np.stack([h.mean(0) for h in current])
    variance = np.stack([h.var(0, ddof=1) for h in current])
    warm_mean = warm_stats['mean'].numpy()
    warm_var = warm_stats['variance'].numpy()
    pooled = warm_stats['pooled'].numpy()
    stable = warm_stats['stable'].numpy()
    dmu = warm_mean[0] - warm_mean[1]
    hh = ((spectrum['warmup_cam'][legal].double() - sp['mu']) @ sp['U']) / (sp['lambda'] + EPS).sqrt()
    warm_dist = -.5 * (warm_stats['stable'].log() + (hh[:, None] - warm_stats['mean']).square() / warm_stats['stable'])
    raw = (warm_dist[:, 0] - warm_dist[:, 1]).numpy()
    limit = r.CFG['class_statistics']['llr_bound']
    result['warmup_train_llr'] = llr_profile(raw, limit * np.tanh(raw / limit), y_full[legal])
    ema = r.calibration_state(module)
    ema_mu, ema_var = ema['ema_mean'].numpy(), ema['class_var'].numpy()
    current_pooled = np.average(variance, axis=0, weights=warm_stats['counts'].numpy())
    current_stable = .9 * variance + .1 * current_pooled + np.maximum(1e-4, .01 * current_pooled)
    ema_gap = ema_mu[0] - ema_mu[1]
    result['class_separation_and_drift'] = {
        'warm_mean': warm_mean.tolist(), 'warm_variance': warm_var.tolist(), 'stable_variance': stable.tolist(),
        'warm_standardized_mean_gap': (dmu / np.sqrt(pooled + EPS)).tolist(),
        'warm_log_variance_ratio_asd_td': np.log(stable[0] / stable[1]).tolist(),
        'warm_symmetric_gaussian_kl': (.25 * (stable[0]/stable[1] + stable[1]/stable[0] - 2 + dmu**2 * (1/stable[0] + 1/stable[1]))).tolist(),
        'best_current_mean': mean.tolist(), 'best_current_variance': variance.tolist(),
        'best_current_stable_variance': current_stable.tolist(),
        'best_ema_updates': int(ema['ema_updates']),
        'ema_mean': ema_mu.tolist(), 'ema_stable_variance': ema_var.tolist(),
        'ema_mean_mismatch_rmse': float(np.sqrt(np.mean((mean - ema_mu)**2))),
        'ema_log_stable_variance_mismatch': float(np.mean(np.abs(np.log(current_stable / ema_var)))),
        'warm_log_stable_variance_mismatch_to_best': float(np.mean(np.abs(np.log(current_stable / stable)))),
        'ema_symmetric_gaussian_kl': (.25 * (ema_var[0]/ema_var[1] + ema_var[1]/ema_var[0] - 2 +
                                             ema_gap**2 * (1/ema_var[0] + 1/ema_var[1]))).tolist(),
        'mean_drift_by_class_direction': (mean - warm_mean).tolist(),
        'log_variance_drift_by_class_direction': np.log((variance + EPS) / (warm_var + EPS)).tolist(),
        'mean_drift_rmse': float(np.sqrt(np.mean((mean - warm_mean)**2))),
        'log_var_drift_abs_mean': float(np.mean(np.abs(np.log((variance + EPS) / (warm_var + EPS))))),
    }
    return result


def routing_difference(reference, candidate, idx):
    result = {'attention_mean_abs': float((reference['a'][idx] - candidate['a'][idx]).abs().mean())}
    for key in ('w_n', 'w_a', 'w_sem', 'w_sp'):
        a, b = reference[key][idx], candidate[key][idx]
        result[key + '_l1'] = float((a - b).abs().sum(-1).mean())
        result[key + '_top1_agreement'] = float((a.argmax(-1) == b.argmax(-1)).float().mean())
    return result


@torch.no_grad()
def interventions(state, module, text, best_out, comparison_module, y):
    full = probability(best_out, state)
    pred = full.argmax(-1)
    d = best_out['details']
    idx = state['test_np']
    saved = module.q0.clone()
    initial = r.sha(r.cpu_state(module))
    rng = r.rng_sha()
    response = []
    treatments = TREATMENTS if comparison_module is not None else {'T3': 'residual_off'}
    zero_prediction = None
    for code, method in treatments.items():
        try:
            if code == 'T1':
                module.q0.copy_(text['q_anonymous'].to(module.q0.device))
            out = best_out if code == 'T0' else independent_forward(state, module, method)
            p = probability(out, state)
            change = np.abs(p[:, 0] - full[:, 0])
            if code == 'T2':
                assert torch.equal(out['details']['w'], d['w_sp'])
                assert not torch.count_nonzero(out['details']['gate_input'][:, 144:])
                module.q0.copy_(text['q_anonymous'].to(module.q0.device))
                alternate = independent_forward(state, module, 'semantic_off')
                assert torch.equal(out['logp'], alternate['logp'])
                assert torch.equal(out['details']['gate'], alternate['details']['gate'])
            if code == 'T3':
                assert torch.count_nonzero(out['details']['delta']) == 0
                zero_prediction = out['logp']
            if code == 'T4':
                assert torch.count_nonzero(out['details']['gate']) == torch.count_nonzero(out['details']['delta']) == 0
                assert torch.equal(out['logp'], zero_prediction)
            result = {'intervention': code, 'metrics': independent_metrics(p, y),
                      'mean_abs_asd_probability_change': float(change.astype(np.float64).mean()),
                      'max_abs_asd_probability_change': float(change.max()), 'n': int(len(y)),
                      'changes': changes(pred, p.argmax(-1), y),
                      'flipped_rows_1based': (idx[p.argmax(-1) != pred] + 1).tolist(),
                      'probability_asd': p[:, 0].tolist(), 'prediction': p.argmax(-1).tolist()}
            if code == 'T1':
                assert torch.equal(out['details']['w_sp'], d['w_sp'])
                result['routing_change_same_best_cam'] = routing_difference(d, out['details'], idx)
                result['mean_abs_gate_change'] = float((d['gate'][idx] - out['details']['gate'][idx]).abs().mean())
            response.append(result)
        finally:
            if code in ('T1', 'T2'):
                module.q0.copy_(saved)
    difference = None
    if comparison_module is not None:
        _, other = comparison_module(d['z'].detach())
        difference = routing_difference(d, other, idx)
    assert r.sha(r.cpu_state(module)) == initial and r.rng_sha() == rng
    return response, difference


def gradient_history(records):
    def summarize(rows):
        samples = [item['prompt_gradient_alignment'] for item in rows]
        cosines = [s['cosine'] for s in samples if s['cosine'] is not None]
        ratios = [s['weighted_auxiliary_to_cls_ratio'] for s in samples if s['weighted_auxiliary_to_cls_ratio'] is not None]
        assert cosines and ratios and all(-1 <= v <= 1 for v in cosines)
        longest = streak = 0
        for s in samples:
            value = s['cosine']
            if value is None or value >= 0:
                streak = 0
            else:
                streak += 1
                longest = max(longest, streak)
        return {'epochs': len(samples), 'defined_cosines': len(cosines), 'negative': sum(v < 0 for v in cosines),
                'negative_share': float(np.mean(np.asarray(cosines) < 0)), 'longest_negative_streak': longest,
                'cosine': distribution(cosines), 'weighted_auxiliary_to_cls_ratio': distribution(ratios),
                'classification_norm': distribution([s['classification_norm'] for s in samples]),
                'weighted_auxiliary_norm': distribution([s['weighted_auxiliary_norm'] for s in samples])}

    return {'all': summarize(records), 'final_quartile': summarize(records[(3 * len(records)) // 4:])}


def verify_selection(state, task, best, latest, log, cache, best_cam, best_observation):
    records = [r.json.loads(line) for line in log.read_bytes().splitlines()]
    assert r.sha_file(log) == latest['log_sha'] == task['log_sha'] and log.stat().st_size == latest['log_bytes']
    stop = task['epochs_executed']
    assert [row['epoch'] for row in records] == list(range(r.START + 1, stop + 1))
    historical = {row['epoch']: row for row in r.historical_task(state['fold'])['rng_trace']}
    y = state['labels_np'][state['test_np']]
    prior = task['identity']['warmup_rng_sha']
    ema = independent_initial_ema(cache)
    assert r.sha(ema) == task['initial_ema_sha']
    mean_steps, variance_steps = [], []
    wait, maximum, selected = 0, -1, None
    common = 0
    for row in records:
        epoch = row['epoch']
        assert wait <= r.CFG['early_stop_wait_gt'] and row['n_test'] == len(y)
        prediction = np.unpackbits(np.frombuffer(bytes.fromhex(row['prediction_bits']), dtype=np.uint8), count=len(y))
        correct = int((prediction == y).sum())
        assert correct == row['correct']
        improved = correct > maximum
        if improved:
            maximum, selected, wait = correct, epoch, 0
        else:
            wait += 1
        assert (improved, wait, selected) == (row['improved'], row['wait'], row['best_epoch'])
        assert tuple(row['losses']) == r.LOSS_NAMES and all(math.isfinite(v) for v in row['losses'].values())
        cls, graph, pair, adv, preserve, darg, res, align, total = row['losses'].values()
        assert np.isclose(darg, cls + graph + .05*pair + .01*adv + .01*preserve, rtol=1e-6, atol=1e-7)
        assert np.isclose(total, darg + .001*res + .001*align, rtol=1e-6, atol=1e-7)
        gate = row['direction_and_gate']
        assert math.isfinite(gate['gamma_after_step']) and 0 <= gate['degenerate_share'] <= 1
        assert 0 <= gate['mean_abs_applied_gate'] <= gate['max_abs_applied_gate'] <= 1
        if epoch == r.START + 1:
            assert gate['max_abs_applied_gate'] == 0 and gate['gamma_after_step'] != 0
            assert row['prompt_gradient_alignment']['cosine'] is None
        if epoch == best['best_epoch']:
            assert gate['gamma_after_step'] == float(best['current']['cspe']['gamma'])
        if epoch == stop:
            assert gate['gamma_after_step'] == float(latest['current']['cspe']['gamma'])
        assert row['rng_before'] == prior and row['rng_after_train'] == row['rng_after_eval'] == row['rng_after_ema']
        prior = row['rng_after_ema']
        hard = {'acc': accuracy_score(y, prediction), 'hard_auc': balanced_accuracy_score(y, prediction),
                'sen': recall_score(y, prediction, pos_label=0), 'spe': recall_score(y, prediction, pos_label=1),
                'asd_f1': f1_score(y, prediction, pos_label=0, zero_division=0)}
        assert all(abs(row['metrics'][key] - val) < 1e-12 for key, val in hard.items())
        assert 0 <= row['metrics']['prob_auc'] <= 1
        em = row['ema']
        assert r.sha(ema) == em['before_sha'] and int(ema['ema_updates']) == epoch - r.START - 1
        observed = observation_from_journal(em)
        if epoch == best['best_epoch']:
            assert r.sha(ema) == best['calibration_sha'] == r.sha({k: best['current']['cspe'][k] for k in EMA_BUFFERS})
            assert r.sha(observed) == r.sha(best_observation) and em['legal_cam_sha'] == r.sha(best_cam)
        if epoch == stop:
            assert r.sha(latest['last_eval_ema']) == r.sha(ema) == task['terminal_eval_calibration_sha']
        updated = independent_ema_step(ema, observed)
        assert int(updated['ema_updates']) == em['updates'] and r.sha(updated) == em['after_sha']
        assert em['mean'] == updated['ema_mean'].tolist() and em['second'] == updated['ema_second'].tolist()
        assert em['stable_variance'] == updated['class_var'].tolist()
        mean_steps.append(float((updated['ema_mean'] - ema['ema_mean']).square().mean().sqrt()))
        variance_steps.append(float((updated['class_var'].double().log() - ema['class_var'].double().log()).abs().mean()))
        ema = updated
        if epoch in historical:
            assert {key: row[key] for key in r.RNG_KEYS} == historical[epoch]
            common += 1
    assert selected == task['best_epoch'] == best['best_epoch'] == latest['best_epoch']
    assert stop == latest['next_epoch'] and wait == latest['wait'] and maximum == latest['best_correct']
    assert wait > r.CFG['early_stop_wait_gt'] or stop == r.LIMIT
    chosen = records[selected - r.START - 1]
    assert chosen['correct'] == (best['prediction'] == y).sum() and maximum / len(y) == best['metrics']['acc']
    assert chosen['prediction_bits'] == np.packbits(best['prediction']).tobytes().hex()
    assert chosen['probability_sha'] == r.sha(best['probability'])
    assert records[-1]['probability_sha'] == r.sha(latest['probability']) == task['terminal_probability_sha']
    same_metrics(chosen['metrics'], best['metrics'])
    same_metrics(records[-1]['metrics'], independent_metrics(latest['probability'], y))
    assert best['rng_sha'] == chosen['rng_after_eval'] and prior == latest['rng_sha'] == task['terminal_rng_sha']
    assert r.sha(ema) == r.sha({k: latest['current']['cspe'][k] for k in EMA_BUFFERS})
    measured = [row['total_gradient_norms'] for row in records if 'total_gradient_norms' in row]
    assert measured and 'total_gradient_norms' in records[0]
    probe = {key: distribution([row[key] for row in measured]) for key in measured[0]}
    sampled_ranks = [row['direction_and_gate']['direction_rank']['mean'] for row in records
                     if 'direction_rank' in row['direction_and_gate']]
    final_quartile = records[(3 * len(records)) // 4:]
    return records, {'common_b0_rng_epochs': common, 'epochs': len(records),
                     'gradient_samples': len(measured), 'training_total_gradient_norms': probe,
                     'prompt_gradient_alignment': gradient_history(records),
                     'ema_trajectory': {'verified_updates': len(records), 'legal_counts_asd_td': ema['ema_counts'].tolist(),
                                        'mean_update_rmse': distribution(mean_steps),
                                        'log_variance_update_abs_mean': distribution(variance_steps),
                                        'last_quartile_mean_update_rmse': distribution(mean_steps[(3*len(records))//4:]),
                                        'last_quartile_log_variance_update_abs_mean': distribution(variance_steps[(3*len(records))//4:]),
                                        'minimum_stable_variance': min(min(map(min, x['ema']['stable_variance'])) for x in records),
                                        'maximum_stable_variance': max(max(map(max, x['ema']['stable_variance'])) for x in records),
                                        'journal_sha256': task['log_sha']},
                     'alignment_kl_first_best_last': [records[0]['losses']['align'], chosen['losses']['align'], records[-1]['losses']['align']],
                     'direction_rank_first': sampled_ranks[0],
                     'direction_rank_late': distribution([x for x in sampled_ranks[-max(1, len(sampled_ranks)//4):] if x is not None]),
                     'degenerate_share': distribution([x['direction_and_gate']['degenerate_share'] for x in records]),
                     'gamma_first_best_last': [x['direction_and_gate']['gamma_after_step'] for x in (records[0], chosen, records[-1])],
                     'max_abs_applied_gate': max(x['direction_and_gate']['max_abs_applied_gate'] for x in records),
                     'late_mean_abs_applied_gate': distribution([x['direction_and_gate']['mean_abs_applied_gate'] for x in final_quartile])}


def reference_fold(state, identity, label, csv_lookup):
    fold = state['fold']
    y = state['labels_np'][state['test_np']]
    if label == 'B0':
        ckpt = r.torch_load(r.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('test_probability', 'test_prediction', 'best_epoch_1based'))
        source_row = csv_lookup[(r.BASE_GROUP, fold)]
    else:
        group = 'G1' if label == 'B1' else 'G2'
        ckpt = r.torch_load(r.E831 / 'runtime/checkpoints' / f'{group}_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('probability', 'prediction', 'best_epoch'))
        assert ckpt['group'] == group and np.array_equal(ckpt['test_indices'], state['test_np'])
        source_row = csv_lookup[(group, fold)]
    assert ckpt['identity'] == identity and len(p) == len(y) and np.array_equal(pred, p.argmax(-1))
    metric = independent_metrics(p, y)
    published = {**metric, 'prob_auc': float(roc_auc_score(y, p[:, 1]))} if label == 'B0' else metric
    same_metrics(published, ckpt['metrics'])
    same_metrics(published, {k: float(source_row[k]) / (1 if label == 'B0' else 100) for k in METRICS})
    return {'group': label, 'fold': fold, 'best_epoch': epoch, 'metrics': metric, 'prediction': pred.tolist(),
            'probability': p.tolist(), 'source_original_prob_auc': published['prob_auc'],
            'source_original_positive_class': 'TD' if label == 'B0' else 'ASD',
            'source_best_checkpoint_original_metrics_checked': True}


def read_task(group, fold, pre):
    assert r.task_complete(group, fold)
    task_file, bp, lp, log = r.paths(group, fold)
    task, best, latest = r.read_json(task_file), r.torch_load(bp), r.torch_load(lp)
    shared = ('group', 'fold', 'seed', 'identity', 'implementation', 'initial_parameter_sha',
              'stats_sha', 'spectrum_sha', 'query_sha', 'legal_train_sha', 'initial_ema_sha')
    for ckpt in (best, latest):
        assert all(task[k] == ckpt[k] for k in shared)
        assert r.component_digests(ckpt['current']) == ckpt['digests']
        assert r.sha(ckpt['current']['cspe']['q0']) == task['query_sha']
    assert latest['best_sha'] == task['best_sha'] and latest['digests'] == task['terminal_digests']
    assert task['implementation'] == pre['implementation']
    assert not r.best_pending(bp).exists()
    return task, best, latest, log


def restored_object(group, fold, ckpt, spectrum, cache, text):
    state, warm, identity = r.prepare(fold)
    module, opt, init_sha = r.make_module(state, group, spectrum, cache, text)
    assert ckpt['identity'] == identity and ckpt['initial_parameter_sha'] == init_sha
    assert ckpt['initial_ema_sha'] == r.sha(r.calibration_state(module)) == r.sha(independent_initial_ema(cache))
    assert ckpt['stats_sha'] == r.sha(cache['stats']) == cache['stats_sha']
    assert ckpt['spectrum_sha'] == r.sha(spectrum['spectrum']) == spectrum['spectra_sha']
    r.load_bundle(state, module, opt, ckpt['current'])
    base = r.base_runtime()
    base.restore_rng(ckpt['rng'])
    assert r.rng_sha() == ckpt['rng_sha'] and r.component_digests(r.bundle(state, module, opt)) == ckpt['digests']
    assert torch.equal(module.q0.cpu(), text[QUERIES[group]])
    for name, key in SPECTRUM_BUFFERS.items():
        assert torch.equal(getattr(module, name).cpu(), spectrum['spectrum'][key].float())
    assert torch.equal(module.ema_counts.cpu(), cache['stats']['counts'])
    assert torch.equal(module.class_mean.cpu(), module.ema_mean.cpu().float())
    saved_calibration = {name: ckpt['current']['cspe'][name] for name in EMA_BUFFERS}
    assert r.sha(r.calibration_state(module)) == r.sha(saved_calibration)
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in module.modules())
    assert all(not b.requires_grad for b in module.buffers())
    return state, module, opt, identity


def audit_best(group, fold, task, best, latest, log, spectrum, cache, text, pre, previous, other_module=None):
    state, module, opt, _ = restored_object(group, fold, best, spectrum, cache, text)
    assert np.array_equal(best['test_indices'], state['test_np'])
    assert int(module.ema_updates) == best['best_epoch'] - r.START - 1
    assert r.sha(r.calibration_state(module)) == best['calibration_sha']
    optimizers = {}
    for name, optimizer in zip((*r.HIST_OPTS, 'opt_psce'), r.optimizer_sequence(state, opt)):
        od = optimizer.state_dict()
        assert od['state'] and all(torch.isfinite(v).all() for s in od['state'].values() for v in s.values() if torch.is_tensor(v))
        optimizers[name] = {'state_slots': len(od['state']), 'digest': r.sha(od)}
    out = independent_forward(state, module, grad=True)
    assert r.rng_sha() == best['rng_sha']
    p = probability(out, state)
    pred = p.argmax(-1)
    assert np.array_equal(p, best['probability']) and np.array_equal(pred, best['prediction'])
    y = state['labels_np'][state['test_np']]
    metrics = independent_metrics(p, y)
    same_metrics(metrics, best['metrics'])
    d = out['details']
    formulas = independent_formula_check(module, d)
    r.validate_output(out)
    grad = r.gradient_audit(state, module, out)
    assert d['scores'].shape == (871, 2) and out['aux_weights'].shape == (len(cache['legal_train']), 16)
    torch.testing.assert_close(out['aux_weights'], d['w_sem'][cache['legal_train']], rtol=2e-6, atol=2e-6)
    best_cam, actual_observation = independent_observation(state, out, spectrum)
    logs, selection_trace = verify_selection(state, task, best, latest, log, cache, best_cam, actual_observation)
    diagnose = mechanism(state, module, cache, spectrum, d, out, previous)
    original_y = state['y'][state['test_np']].clone()
    original_y_numpy = state['labels_np'][state['test_np']].copy()
    state['y'][state['test_np']] = 1 - original_y
    state['labels_np'][state['test_np']] = 1 - original_y_numpy
    try:
        changed = independent_forward(state, module)
        assert torch.equal(changed['logp'], out['logp'])
        assert all(torch.equal(changed['details'][key], d[key]) for key in ('raw_llr', 'scores', 'pi', 'w_sem', 'gate', 'delta'))
        _, new_observation = independent_observation(state, changed, spectrum)
        assert r.sha(new_observation) == r.sha(actual_observation)
    finally:
        state['y'][state['test_np']] = original_y
        state['labels_np'][state['test_np']] = original_y_numpy
    assert r.rng_sha() == best['rng_sha']
    readonly, comparison = interventions(state, module, text, out, other_module, y)
    assert r.component_digests(r.bundle(state, module, opt)) == best['digests']
    return {'status': 'PASS', 'group': group, 'fold': fold, 'metrics': metrics, 'best_epoch': best['best_epoch'],
            'epochs_executed': task['epochs_executed'], 'initial_parameter_sha': task['initial_parameter_sha'],
            'initial_ema_sha': task['initial_ema_sha'], 'best_calibration_sha': best['calibration_sha'],
            'best_pre_update_ema_count': int(module.ema_updates),
            'best_file_sha': task['best_sha'], 'probability_max_abs_replay_error': 0., 'probability_sha': r.sha(p),
            'best_component_digests': best['digests'], 'best_rng_sha': best['rng_sha'], 'optimizers': optimizers,
            'test_rows_1based': (state['test_np'] + 1).tolist(), 'labels': y.tolist(), 'probability': p.tolist(),
            'prediction': pred.tolist(), 'formula_max_abs_errors': formulas, 'best_gradients': grad,
            'historical_preflight_gradients': pre['folds'][fold - 1]['groups'][group]['gradients'],
            'selection_trace': selection_trace, 'mechanism': diagnose,
            'rng_trace': [{key: row[key] for key in r.RNG_KEYS} for row in logs],
            'test_predictions_and_legal_observation_label_swap_bitwise': True, 'treatments': readonly,
            'trained_g1_vs_g2_same_g2_cam': comparison, 'trainable_parameters': sum(p.numel() for p in module.parameters()),
            'peak_gpu_bytes': task['peak_gpu_bytes'], 'resumes': task['resumes']}


def audit_latest(group, fold, task, latest, spectrum, cache, text, log):
    # Build fresh model and optimizer objects even when the independent Best replay has passed.
    state, module, opt, _ = restored_object(group, fold, latest, spectrum, cache, text)
    before = r.rng_sha()
    terminal_ema = r.calibration_state(module)
    assert int(terminal_ema['ema_updates']) == latest['next_epoch'] - r.START
    assert int(latest['last_eval_ema']['ema_updates']) == int(terminal_ema['ema_updates']) - 1
    restore_ema(module, latest['last_eval_ema'])
    final_eval = independent_forward(state, module)
    p = probability(final_eval, state)
    assert np.array_equal(p, latest['probability']) and r.sha(p) == task['terminal_probability_sha']
    assert r.sha(r.calibration_state(module)) == task['terminal_eval_calibration_sha']
    legal_cam, observed = independent_observation(state, final_eval, spectrum)
    last = r.json.loads(log.read_bytes().splitlines()[-1])
    assert r.sha(legal_cam) == last['ema']['legal_cam_sha']
    assert r.sha(observed) == r.sha(observation_from_journal(last['ema']))
    assert r.sha(independent_ema_step(latest['last_eval_ema'], observed)) == r.sha(terminal_ema)
    restore_ema(module, terminal_ema)
    assert r.component_digests(r.bundle(state, module, opt)) == latest['digests']
    assert before == r.rng_sha() == task['terminal_rng_sha']
    optimizer_order = [state[k] for k in r.HIST_OPTS] + [opt]
    for item in optimizer_order:
        item.zero_grad(set_to_none=True)
    out = independent_forward(state, module, training=True)
    r.validate_output(out)
    out['total'].backward()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for _, param in r.named_parameters(state, module))
    for item in optimizer_order:
        item.step()
    after_train = r.rng_sha()
    ev = independent_forward(state, module)
    p = probability(ev, state)
    after_eval = r.rng_sha()
    ema_before = r.calibration_state(module)
    legal_cam, observed = independent_observation(state, ev, spectrum)
    independent_next = independent_ema_step(ema_before, observed)
    module.update_calibration(observed)
    ema_after = r.calibration_state(module)
    assert r.sha(ema_after) == r.sha(independent_next)
    checkpoint = r.checkpoint_state(state, module, opt)
    assert after_train == after_eval == checkpoint['rng_sha']
    expected = task['next_step_witness']
    actual = {'after_train_rng': after_train, 'after_eval_rng': after_eval, 'after_ema_rng': checkpoint['rng_sha'],
              'components_after_update': checkpoint['digests'], 'probability_sha': r.sha(p),
              'ema_before_sha': r.sha(ema_before), 'ema_after_sha': r.sha(ema_after),
              'observation_sha': r.sha(observed), 'legal_cam_sha': r.sha(legal_cam),
              'losses': dict(zip(r.LOSS_NAMES, torch.stack([out[k].detach() for k in r.LOSS_NAMES]).cpu().tolist()))}
    for key, value in actual.items():
        assert value == expected[key], (group, fold, key)
    return {'status': 'PASS', 'latest_file_sha': task['latest_sha'], 'epoch': latest['next_epoch'],
            'restored_components': latest['digests'], 'rng': before, 'terminal_probability_exact': True,
            'terminal_probability_max_abs_error': 0., 'terminal_probability_sha': task['terminal_probability_sha'],
            'terminal_post_update_ema_count': int(terminal_ema['ema_updates']), 'terminal_post_update_ema_sha': r.sha(terminal_ema),
            'next_step_matches_training_witness': True, 'live_terminal_witness': expected['live_terminal_object'],
            'next_step': actual}


def summed_changes(items):
    return {scope: {metric: sum(x[scope][metric] for x in items)
                    for metric in ('flips', 'corrected', 'harmed', 'net')}
            for scope in ('all', 'asd', 'td')}


def scientific_decision(paired, corrections, treatments, residual_off, tasks, audits_pass):
    def delta(group, baseline, key):
        return paired[f'{group}−{baseline}']['mean_pp'][key]

    def net(group):
        return corrections[f'{group}−B0']['all']['net']

    cfg = r.CFG['decision']
    performance, mechanism = {}, {}
    for group in r.GROUPS:
        performance[group] = {
            'all_audits_pass': audits_pass,
            'acc_higher_than_darg': delta(group, 'B0', 'acc') > 0,
            'ba_at_least_darg': delta(group, 'B0', 'hard_auc') >= -1e-10,
            'asd_sensitivity_within_tolerance': delta(group, 'B0', 'sen') >= cfg['asd_sensitivity_min_delta_pp_vs_darg'],
            'prob_auc_within_tolerance': delta(group, 'B0', 'prob_auc') >= cfg['prob_auc_min_delta_pp_vs_darg'],
            'positive_net_patient_corrections': net(group) > 0,
        }
        split = [a['mechanism']['test'] for a in tasks if a['group'] == group]
        effect = residual_off[group]
        mechanism[group] = {
            'all_audits_pass': audits_pass,
            'residual_off_removes_net_correct_predictions': effect['changes']['all']['net'] < 0,
            'residual_off_lowers_fold_mean_acc': effect['mean_delta_pp']['acc'] < 0,
            'prompt_direction_weights_differ_from_spectral': float(np.mean(
                [a['w_sem_w_sp_l1']['mean'] for a in split])) > cfg['identity_tolerance'],
            'gate_is_active': max(a['abs_gate']['max'] for a in split) > cfg['identity_tolerance'],
        }
    query_change = treatments['T1']
    semantic = {'positive_g2_vs_g1_acc_delta': delta('G2', 'G1', 'acc') > 0,
                'positive_g2_vs_g1_net_correct': corrections['G2−G1']['all']['net'] > 0,
                'query_replacement_lowers_acc': query_change['mean_delta_pp']['acc'] < 0,
                'query_replacement_removes_net_correct_predictions': query_change['changes']['all']['net'] < 0}
    p_go, m_go = ({g: all(checks.values()) for g, checks in criterium.items()} for criterium in (performance, mechanism))
    qualifying = [g for g in r.GROUPS if p_go[g] and m_go[g]]
    return {'verdict': 'GO' if qualifying else 'NO-GO',
            'qualifying_groups': qualifying, 'performance_go': p_go, 'mechanism_go': m_go,
            'performance_checks': performance, 'mechanism_checks': mechanism, 'medical_query_checks': semantic,
            'independent_medical_query_evidence_under_this_protocol': audits_pass and all(semantic.values())}


def summarize_intervention(by, group, treatment, folds):
    selected = [next(row for row in by(group, f)['treatments'] if row['intervention'] == treatment) for f in folds]
    original = [by(group, f)['metrics'] for f in folds]
    return {'mean_delta_pp': {k: float(100 * np.mean([a['metrics'][k] - ref[k] for a, ref in zip(selected, original)]))
                              for k in METRICS},
            'mean_abs_asd_probability_change': float(sum(a['n'] * a['mean_abs_asd_probability_change'] for a in selected)
                                                    / sum(a['n'] for a in selected)),
            'max_abs_asd_probability_change': max(a['max_abs_asd_probability_change'] for a in selected),
            'changes': summed_changes([a['changes'] for a in selected]),
            'folds_with_negative_net': sum(a['changes']['all']['net'] < 0 for a in selected)}


def aggregate(tasks, references, audits_pass):
    data = {(a['group'], a['fold']): a for a in [*tasks, *references]}
    by = lambda g, f: data[(g, f)]
    groups = (*r.GROUPS, *BASELINES)
    fold_rows, pair_rows = [], []
    for fold in r.FOLDS:
        for group in groups:
            row = by(group, fold)
            fold_rows.append({'group': group, 'fold': fold, 'best_epoch': row['best_epoch'],
                              'end_epoch': row.get('epochs_executed', ''), **{k: 100 * row['metrics'][k] for k in METRICS}})
    summaries, comparisons, corrections, treatments, residual_off = {}, {}, {}, {}, {}
    for scope, folds in [('all10', list(r.FOLDS)), ('without_fold2', [f for f in r.FOLDS if f != 2])]:
        summaries[scope] = {}
        comparisons[scope] = {}
        corrections[scope] = {}
        treatments[scope] = {}
        for group in groups:
            values = {k: np.array([100 * by(group, f)['metrics'][k] for f in folds]) for k in METRICS}
            summaries[scope][group] = {k: {'mean': float(v.mean()), 'population_sd': float(v.std(ddof=0))} for k, v in values.items()}
        for group, reference in COMPARISONS:
            label = f'{group}−{reference}'
            deltas = [{key: 100 * (by(group, fold)['metrics'][key] - by(reference, fold)['metrics'][key]) for key in METRICS} for fold in folds]
            mean = {key: float(np.mean([a[key] for a in deltas])) for key in METRICS}
            win_tie_loss = {key: [sum(a[key] > 1e-10 for a in deltas), sum(abs(a[key]) <= 1e-10 for a in deltas),
                                  sum(a[key] < -1e-10 for a in deltas)] for key in METRICS}
            changes_per_fold = []
            for fold, delta in zip(folds, deltas):
                y = np.asarray(by(group, fold)['labels'])
                to = np.asarray(by(group, fold)['prediction'])
                before = np.asarray(by(reference, fold)['prediction'])
                changed = changes(before, to, y)
                changes_per_fold.append(changed)
                pair_rows.append({'scope': scope, 'comparison': label, 'fold': fold, **{f'{key}_delta_pp': v for key, v in delta.items()},
                                  **{f'{class_name}_{key}': value for class_name, report in changed.items() for key, value in report.items()}})
            comparisons[scope][label] = {'mean_pp': mean, 'w_t_l': win_tie_loss}
            corrections[scope][label] = summed_changes(changes_per_fold)
        treatments[scope] = {t: summarize_intervention(by, 'G2', t, folds) for t in TREATMENTS}
        residual_off[scope] = {g: summarize_intervention(by, g, 'T3', folds) for g in r.GROUPS}
    r.write_csv(r.RESULTS / 'fold_metrics.csv', fold_rows)
    r.write_csv(r.RESULTS / 'paired_comparisons.csv', pair_rows)
    decision = scientific_decision(comparisons['all10'], corrections['all10'], treatments['all10'],
                                   residual_off['all10'], tasks, audits_pass)
    return {'summary': summaries, 'paired': comparisons, 'corrections': corrections, 'treatments': treatments,
            'residual_off': residual_off, 'decision': decision, 'fold_rows': fold_rows}


SOURCE_NOTES = [
    'SPromptGL 论文第 2.2 节的 PLN 通过 CAM 进行跨模态交换。PEN 的 SPGS 使用文本相似度重加权局部 token，RES 再融合局部和全局表征。原作者代码调用冻结 CLIP RN50 编码和投影文本，以局部相似度和全局聚合实现 PEN。历史 E74.4-G2 通过 E61/E53.3 继承私有 vendor 的 `PLNNoPEN`。四种患者关系及 Positive-DCR 产生 320 维融合表征，经 InputLayer、Encoder 和 FeedForward 后展平为 CAM144，传给原 OutputLayer。该头逐患者执行 LayerNorm、Linear、GELU、历史 Dropout、Linear 与 log_softmax。 [论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者 network.py](https://github.com/wanxixi11/SPromptGL/blob/ba3b9446fa451fde6864d6555bdf324de0c4b64a/network.py)',
    '项目固定提交 f27117495166dc2320c5cb2a76b7e85c3972950c 的 ac_cspe.py、run.py、audit.py、config.json 和 report.md 与服务器历史文件逐字节匹配。E82.2 采用统一覆盖率和反事实教师。E82.3 用固定 warmup 类别统计及类平衡 Prompt 分数 CE。E82.3.1 沿用其谱基，并引入动态类别校准与逐方向 LLR 加权。E82.3.1-G2 测试 LLR AUC 从同一 Best CAM 的冻结统计 46.49% 升至 EMA 89.57%，ACC 为 91.4995%，DARG 为 92.1891%。Query 替换造成零翻转，关闭残差造成一例翻转。旧 CE 的目标是 Prompt 自身类别分数，它无需改善原 OutputLayer 使用残差后的真类 margin。 [项目仓库](https://github.com/tangaii/SPromptGL) · [E82.3.1 固定提交](https://github.com/tangaii/SPromptGL/tree/f27117495166dc2320c5cb2a76b7e85c3972950c/experiment/experiment82_3_1_ac_cspe) · [历史报告](/model/spromptgl/experiment/experiment82_3_1_ac_cspe/results/report.md)',
    '原 E61 epoch 10 合法 CAM 用于固定谱空间。前 8 维为总协方差主方向，后 8 维为正交补上 ASD−TD 协方差差异绝对特征值最大的方向。U、mu、lambda、标准化 loglambda 和 nu 均由保护缓存继承。四加四 Prompt 的 8×17 注意力沿用温度 0.5 和 null。各组连续 coverage 分别按带符号平滑 LLR 的 softplus 加权并归一化。门控前 coverage 加权得两类分数，经 softmax 混合得到 w_sem。原谱 MLP 3–32–1 输出 w_sp。beta 仍为 detach((1−null)×clamp(1−H(w_sem), 0.2, 0.8))。混合权重相对 1/16 的偏差作用于 h，经 U 转置重建，再由原 8% 平滑限幅形成 delta_base。',
    '正式组 G1 使用历史 Gram 保持匿名 Query，G2 使用历史医学 Query。保留八条原始文本，包括 Nromal brain volume。正常样与异常样分组是研究假设。每折继承 E82.3.1 的全部 6081 个参数。新增 162–16–1 GELU 门控和 gamma 共 2626 个参数，门控单独按 823200+fold 初始化。162 维输入包括无可训练仿射的 CAM LayerNorm、w_sem−w_sp、beta 和均值 null 注意力。gamma 精确设为 0，初始 CAM 恒等。实际残差为 tanh(gamma)×sigmoid(MLP)×delta_base。gamma 的首步分类导数可非零，随后分类梯度到达 Prompt、谱网络与门控 MLP。',
    '训练复用原分类头的既有前向图。对合法 ASD=0 与 TD=1 患者求真类 logp 差对增强 CAM 的一阶导数。各方向贡献为该导数与 h_j·U_j 的内积。逐患者中心化、以不小于 1e−4 的 RMS 归一化、裁剪到 [−3,3]、温度 1 softmax。常量向量采用均匀目标。目标、CAM 输入和类别统计均与辅助梯度隔离。类平衡 KL 直接监督经过 LLR 逐方向加权的 w_sem，更新 direction、evidence、norm、key、null。原分类损失仍端到端更新 DARG、CAM、OutputLayer 和新增模块。总损失为历史 cls+graph+0.05 pair+0.01 adv+0.01 preserve，再加 0.001 实际残差惩罚和 0.001 对齐 KL。依次更新 opt_base、opt_cam、opt_added、opt_psce，第四个为原 AdamW，lr=0.0005，weight decay=0.01。',
    '逐折恢复 E61 epoch 10 的模型、CAM、三个优化器和 Python、NumPy、torch CPU、设备 RNG，应用历史 E61-E2/DARG 校准及门控转换。Fold2 保持 377 和 407 参与历史训练，保留该折实际的 4096 关系对。两人排除于新增训练统计、方向目标与审计的测试方向标签诊断，新增残差为零。原分类指标仍覆盖完整历史测试集。其余各折继承原关系对与索引。',
    '每轮完成四优化器更新后，以旧 EMA 评估一次。原 test ACC 严格升高时冻结 Best 的参数、四优化器、RNG、EMA 和预测。原评估捕获合法训练 CAM，CPU float64 计算类别一阶与原始二阶矩，以 alpha=0.1 更新，按 n/(n−1) 校正方差，执行旧方差收缩和下限后转为 float32 推理 buffer。更新后生成 Latest，含早停状态及当轮用于复核预测的上一 EMA。训练至 450 或 wait>200，Latest 每 10 轮及终止提交。独立审计和只读干预不自动更新 EMA。',
]


def table(headers, rows):
    return ['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|',
            *['|' + '|'.join(map(str, row)) + '|' for row in rows]]


def summary_table(summary, groups):
    return table(['组别', *DISPLAY], [[g, *[f'{summary[g][key]["mean"]:.4f} ± {summary[g][key]["population_sd"]:.4f}'
                                          for key in METRICS]] for g in groups])


def tally_label(tally):
    return '/'.join(str(tally[key]) for key in ('corrected', 'harmed', 'net'))


def report_mechanisms(status):
    lines = ['## 方向目标、梯度与患者门控', '',
             '方向 KL 与 Spearman 对两类等权平均，Spearman 使用患者内 16 个方向的平均秩。测试方向目标在保存 Best 后用真实类别计算，仅用于回顾诊断。E82.3.1 行使用历史同组 Best 的 Prompt 和 EMA，输入同一 DA-SRPE CAM 和同一方向目标。该比较控制诊断输入，仍包含模型共同适配的差异。', '']
    for g in r.GROUPS:
        tasks = [a for a in status['tasks'] if a['group'] == g]
        avg = lambda values: float(np.mean(list(values)))
        diagnostics = [a['mechanism'] for a in tasks]
        lines.extend([f'### {g}', '', *table(['患者', '选择权重', 'KL', 'Spearman', 'Top1 一致率'],
                         [[{'legal_train': '合法训练', 'test': '测试'}[split], key,
                           f"{avg(a[split]['decision_direction'][key]['balanced_kl'] for a in diagnostics):.5f}",
                           f"{avg(a[split]['decision_direction'][key]['spearman']['mean'] for a in diagnostics):+.5f}",
                           f"{100*avg(a[split]['decision_direction'][key]['top1_share_nondegenerate'] for a in diagnostics):.2f}%"]
                          for split in ('legal_train', 'test') for key in ('w_sem', 'w_sp', 'e8231_on_same_cam')]), ''])
        first, at_best, final = np.mean([a['selection_trace']['alignment_kl_first_best_last'] for a in tasks], axis=0)
        early = avg(a['selection_trace']['direction_rank_first'] for a in tasks)
        late = avg(a['selection_trace']['direction_rank_late']['mean'] for a in tasks)
        degenerate = avg(a['selection_trace']['degenerate_share']['mean'] for a in tasks)
        changed = avg(a['test']['decision_direction']['w_sem_vs_e8231_l1']['mean'] for a in diagnostics)
        wsem_spec = avg(a['test']['w_sem_w_sp_l1']['mean'] for a in diagnostics)
        lines.extend([f'训练首轮、Best 轮、末轮的类平衡 KL 折均为 {first:.5f}、{at_best:.5f}、{final:.5f}。首轮及靠后采样轮次 Spearman 为 {early:+.5f}、{late:+.5f}。训练退化方向目标比例 {100*degenerate:.3f}%。测试 w_sem 与历史选择器在同一 CAM 上的 L1 差为 {changed:.5f}，与同期 w_sp 的 L1 差为 {wsem_spec:.5f}。', ''])
        for label, split in [('全部轮次', 'all'), ('最后四分之一轮次', 'final_quartile')]:
            grads = [a['selection_trace']['prompt_gradient_alignment'][split] for a in tasks]
            negative = sum(a['negative'] for a in grads)
            defined = sum(a['defined_cosines'] for a in grads)
            lines.extend([f"{label}的加权辅助/分类梯度范数比折均 {avg(a['weighted_auxiliary_to_cls_ratio']['mean'] for a in grads):.4f}，平均余弦 {avg(a['cosine']['mean'] for a in grads):+.4f}，负余弦 {negative}/{defined}（{100*negative/defined:.2f}%），最长连续负余弦 {max(a['longest_negative_streak'] for a in grads)} 轮。", ''])
        best_norms = [a['best_gradients']['norms'] for a in tasks]
        names = ('darg', 'cam', 'output', 'prompt_parameters', 'spectral', 'patient_gate', 'gamma')
        lines.extend([*table(['Best 梯度范数折均', '主分类', '0.001×对齐 KL', '0.001×实际残差'],
                            [[name, *[f"{avg(a[key][name] for a in best_norms):.6g}" for key in
                               ('classification', 'weighted_auxiliary', 'weighted_residual')]] for name in names]), ''])
        ga = [a['gamma'] for a in diagnostics]
        test = [a['test'] for a in diagnostics]
        pos, neg, zero = [sum(a[f'gate_{k}_n'] for a in test) for k in ('positive', 'negative', 'zero')]
        margin = avg(a['true_logit_margin_change']['mean'] for a in test)
        margin_share = avg(a['true_margin_increased_share'] for a in test)
        base_margin_corr = [a['abs_gate_vs_bare_true_margin_pearson'] for a in test if a['abs_gate_vs_bare_true_margin_pearson'] is not None]
        gain_corr = [a['signed_gate_vs_true_margin_gain_pearson'] for a in test if a['signed_gate_vs_true_margin_gain_pearson'] is not None]
        lines.extend([f"Best gamma 范围 [{min(ga):+.6f}, {max(ga):+.6f}]，非零 {sum(x != 0 for x in ga)}/10。测试门控正、负、零为 {pos}/{neg}/{zero} 人。符号由每折共享的 gamma 决定，患者条件化 MLP 改变幅度。逐患者门控数值及实际相对残差见审计 JSON。", '',
                      *table(['Best 测试统计', '折均', '折中位数均值', '折 P95 均值', '全折最大'],
                             [[name, f"{avg(scale*x[key]['mean'] for x in test):.6f}",
                               f"{avg(scale*x[key]['p50'] for x in test):.6f}", f"{avg(scale*x[key]['p95'] for x in test):.6f}",
                               f"{max(scale*x[key]['max'] for x in test):.6f}"]
                              for name, key, scale in [('绝对门控', 'abs_gate', 1), ('门控前相对残差 %', 'relative_residual_before_gate', 100),
                                   ('实际相对残差 %', 'relative_residual', 100), ('beta', 'beta', 1),
                                   ('归一化 Prompt 熵', 'entropy', 1), ('null 概率', 'null', 1)]]), '',
                      f"测试真类 logit margin 平均变化 {margin:+.6g}，margin 提高的折均比例 {100*margin_share:.2f}%。|g| 与原真类 margin、g 与真类 margin 变化的折均 Pearson 为 {avg(base_margin_corr):+.4f}、{avg(gain_corr):+.4f}。", '',
                      f"平滑方向均值 LLR AUC 在 Warmup 合法训练、Best 合法训练和 Best 测试分别为 {100*avg(a['warmup_train_llr']['bounded_mean_score_roc_auc'] for a in diagnostics):.2f}%、{100*avg(a['legal_train']['llr']['bounded_mean_score_roc_auc'] for a in diagnostics):.2f}% 和 {100*avg(a['test']['llr']['bounded_mean_score_roc_auc'] for a in diagnostics):.2f}%。该诊断描述类别校准保留的分离度。", ''])
    return lines


def brief_evidence(status):
    ag = status['aggregates']
    verdict = ag['decision']
    paragraphs = []
    for g in r.GROUPS:
        change = ag['corrections']['all10'][f'{g}−B0']
        residual = ag['residual_off']['all10'][g]['changes']['all']
        p, m = verdict['performance_go'][g], verdict['mechanism_go'][g]
        residual_effect = ('移除造成净损失' if residual['net'] < 0 else
                           '移除提高净正确数' if residual['net'] > 0 else '移除后的净正确数持平')
        paragraphs.append(f"{g} 的 Performance GO 和 Mechanism GO 分别{'通过' if p else '未通过'}、{'通过' if m else '未通过'}。相对 DARG，ASD 纠错/误伤 {change['asd']['corrected']}/{change['asd']['harmed']}，TD {change['td']['corrected']}/{change['td']['harmed']}，总净纠错 {change['all']['net']:+d}。该组关闭 Best 残差时翻转 {residual['flips']} 人、净纠错 {residual['net']:+d}，{residual_effect}。")
    t = ag['treatments']['all10']['T1']
    t_net = t['changes']['all']['net']
    t_flips = t['changes']['all']['flips']
    direct = [next(x for x in task['treatments'] if x['intervention'] == 'T1') for task in status['tasks'] if task['group'] == 'G2']
    query_l1 = float(np.mean([x['routing_change_same_best_cam']['w_sem_l1'] for x in direct]))
    av_gate = float(np.mean([x['mean_abs_gate_change'] for x in direct]))
    paired = ag['paired']['all10']['G2−G1']['mean_pp']
    semantics = verdict['independent_medical_query_evidence_under_this_protocol']
    paragraphs.append(f"G2−G1 的 ACC/BA 差为 {paired['acc']:+.4f}/{paired['hard_auc']:+.4f} pp。固定 G2 换成匿名 Query 后，平均 w_sem L1 差 {query_l1:.5f}，门控平均绝对差 {av_gate:.6g}，ASD 概率平均绝对变化 {100*t['mean_abs_asd_probability_change']:.6f} pp，硬翻转 {t_flips} 人，净纠错 {t_net:+d}。本协议下医学 Query 的独立硬分类收益{'获得对照与干预的共同支持' if semantics else '尚未建立'}。")
    return paragraphs


def make_report(status, manifest):
    code = status['status']
    best, latest = status['best_pass'], status['latest_pass']
    lines = ['# Experiment 82.3.2 · DA-SRPE', '',
             '**Decision-Aligned Safe Residual Prompt Evidence Learning（决策对齐的安全残差 Prompt 证据学习）**', '',
             f'状态 **{code}**。正式训练 {status["completed_tasks"]}/20，Best 独立审计 {best}/20，Latest 独立恢复 {latest}/20。运行目录 `{r.HERE}`。', '',
             '本实验检验从原分类头的真类 Margin 梯度学习谱方向，并以零初始化患者门控施加受限残差，能否改善 ABIDE 上的 ASD/TD 硬分类和患者净纠错。医学 Query 与同几何匿名 Query 用于评估医学文本的额外贡献。', '',
             '## 论文和源码依据', '']
    for paragraph in SOURCE_NOTES:
        lines.extend([paragraph, ''])
    lines.extend(['## Preflight 与受保护来源', ''])
    for paragraph in preflight_summary(status):
        lines.extend([paragraph, ''])
    if not status.get('aggregates'):
        lines.extend(['## 阻断记录', '', str(status.get('blocking_reason', '独立审计未通过，详见 audit_status.json。')), '',
                      '当前科研状态为停止，待产生完整有效的独立审计结果后才能对方法性能作判断。', ''])
        return lines
    ag = status['aggregates']
    lines.extend(['## 逐折与汇总性能', '',
                  '各指标单位为百分数。Hard-AUC 按历史定义为硬预测的 Balanced Accuracy。Prob-AUC 用 ASD=阳性和连续 ASD 概率计算。历史逐折 CSV 和 Best 原始预测均已逐项读取、复算，并采用同一 ASD 正类比较。', ''])
    for scope, title in [('all10', '全部 10 折'), ('without_fold2', '去 Fold2 的 9 折')]:
        lines.extend([f'### {title}', '', '均值 ± population SD。B0 为 E74.4-G2 DARG，B1 为 E82.3.1-G1，B2 为 E82.3.1-G2。', '',
                      *summary_table(ag['summary'][scope], (*r.GROUPS, *BASELINES)), ''])
    lines.extend(['### 正式 20 个任务', '', *table(['组', '折', 'Best epoch', '终止 epoch', *DISPLAY],
        [[row['group'], row['fold'], row['best_epoch'], row['end_epoch'], *[f'{row[k]:.4f}' for k in METRICS]]
         for row in ag['fold_rows'] if row['group'] in r.GROUPS]), ''])
    auc_orientation = max(abs(a['source_original_prob_auc'] - a['metrics']['prob_auc']) for a in status['historical_references'])
    lines.extend([f'B0 历史 Prob-AUC 使用 TD 正类，B1 和 B2 使用 ASD 正类。历史方向与统一重算 ASD 方向的最大逐折绝对差为 {100*auc_orientation:.8f} pp。三组历史逐折指标同样列入 `results/fold_metrics.csv`。', '',
                  '## 配对差值和患者级硬预测', '', '差值为新模型减对照，单位 pp。每折六指标差值和逐类纠错详见 `results/paired_comparisons.csv`。', ''])
    for scope, title in [('all10', '全部 10 折'), ('without_fold2', '去 Fold2 的 9 折')]:
        lines.extend([f'### {title}', '', *table(['比较', *DISPLAY, 'ACC 胜/平/负', 'BA 胜/平/负'],
            [[name, *[f'{v["mean_pp"][key]:+.4f}' for key in METRICS], '/'.join(map(str, v['w_t_l']['acc'])),
              '/'.join(map(str, v['w_t_l']['hard_auc']))] for name, v in ag['paired'][scope].items()]), '',
            *table(['比较', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总纠错/误伤/净值'],
                   [[name, v['all']['flips'], tally_label(v['asd']), tally_label(v['td']), tally_label(v['all'])] for name, v in ag['corrections'][scope].items()]), ''])
    lines.extend(report_mechanisms(status))
    lines.extend(['## G2 Best 只读干预', '',
                  'T0 为完整医学模型。T1 替换为历史匿名 Query。T2 用原 w_sp 计算残差，并将所有 18 个依赖 Query 的门控输入设为零。T3 将最终新增残差置零。T4 将门控输出置零。硬翻转和纠错方向为 T0 → 干预，概率变化以 ASD 概率百分点计算。逐折与去 Fold2 结果见审计文件。', '',
                  *table(['干预', '平均绝对 ΔP', '最大绝对 ΔP', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总净纠错', *DISPLAY],
                         [[name, f'{100*v["mean_abs_asd_probability_change"]:.6f}', f'{100*v["max_abs_asd_probability_change"]:.6f}',
                           v['changes']['all']['flips'], tally_label(v['changes']['asd']), tally_label(v['changes']['td']), v['changes']['all']['net'],
                           *[f'{v["mean_delta_pp"][k]:+.4f}' for k in METRICS]] for name, v in ag['treatments']['all10'].items()]), '',
                  'T2 的门控与分类概率在再次替换 Query 后均保持逐元素一致。T3/T4 预测逐元素一致。干预只衡量固定 Best 的即时效应。主干和分类头已共同适配，删除模块并重新训练的因果效应仍未识别。G1 额外执行同一 T3 以核对 Mechanism GO。', '',
                  '## 独立审计', ''])
    execution = manifest['execution']
    rng = status['rng_audit']
    lines.extend([
        f"{best}/20 个 Best 的逐患者预测与概率位级重现。独立程序复算六指标、最早 ACC 平局和 wait>200，并重算 {rng['ema_updates']} 次 EMA 转移。Best 与当轮更新前状态对应，Latest 与更新后状态对应。"
        f"训练与 DARG 的共同 RNG 三元组 {rng['b0_common']} 条，两组同折三元组 {rng['g1_g2_common']} 条，逐条一致。超出历史 DARG 记录的 {rng['b0_unavailable']} 个训练轮次核对了连续 RNG 链及可用的同折组间重叠。", '',
        f"{latest}/20 个 Latest 在全新对象恢复四个优化器、EMA 和完整 RNG，复算终止预测后独立执行下一次训练、评估和 EMA 更新。"
        f"通过项的终止预测最大绝对恢复误差为 0。模型、四个优化器、逐患者概率、合法训练观测、EMA 与 RNG 的哈希见审计 JSON。保护集合含模型根代码、数据和历史产物，共 {status['protected_files']} 项，SHA-256 核对{'通过' if status['protected_files_sha_match'] else '失败'}。", '',
        f"只用单设备 PPU-ZW810E（98,304 MiB），同时最多 {execution['max_workers']} 个折任务进程，每进程一个 CPU 数值线程。正式训练耗时 {execution['wall_seconds']:.1f} 秒，独立审计耗时 {status['audit_wall_seconds']:.1f} 秒，记录运行失败事件 {len(execution['failures'])}，整卡采样峰值 {execution['sampled_peak_device_mib']} MiB。", '',
        *table(['本次冻结源码', 'SHA-256'], list(status['implementation'].items())), '',
        '## 研究判断与限制', '', f"**{ag['decision']['verdict']}**。", ''])
    for paragraph in brief_evidence(status):
        lines.extend([paragraph, ''])
    decision = ag['decision']
    for group in r.GROUPS:
        nine = ag['paired']['without_fold2'][f'{group}−B0']['mean_pp']
        lines.extend([f"去 Fold2 后，{group} 对 DARG 的 ACC、BA、SEN-ASD 差值分别为 {nine['acc']:+.4f}、{nine['hard_auc']:+.4f}、{nine['sen']:+.4f} pp，患者净纠错 {ag['corrections']['without_fold2'][f'{group}−B0']['all']['net']:+d}。", ''])
    for title, key in [('Performance GO', 'performance_checks'), ('Mechanism GO', 'mechanism_checks')]:
        a, b = (decision[key][g] for g in r.GROUPS)
        lines.extend([f'### {title}', '', *table(['冻结条件', 'G1', 'G2'], [[k, '是' if a[k] else '否', '是' if b[k] else '否'] for k in a]), ''])
    lines.extend(['历史 val=test 选模、CV 前 RFE 与 scaling、转导患者图、Fold2 训练重叠及单 seed 限制评估范围。十折均值与方差描述该历史协议，独立外部泛化仍待验证。Margin 方向贡献是分类器局部一阶近似，不能验证方向的医学因果作用。两项方法改动联合训练，各自重新训练的贡献仍未识别。', ''])
    return lines


def preflight_summary(status):
    pre = r.read_json(r.RT / 'preflight.json') if (r.RT / 'preflight.json').exists() else {}
    count = len(pre.get('folds', []))
    lines = [f"Preflight 完成 {count}/10 折，状态 {pre.get('status', 'PENDING')}。首选 OSSFS 创建目录返回 errno 5，此后统一在 `/model` 运行。"]
    if pre.get('status') == 'PASS':
        folds = pre['folds']
        auc = [x['llr']['bounded_auc_direction_mean_score'] for x in folds]
        gamma_grad = [g['gradients']['norms']['classification']['gamma'] for x in folds for g in x['groups'].values()]
        ratios = [x['ema_replay']['after_first_update_gradients']['relative_prompt_parameter_norms']['weighted_auxiliary'] for x in folds]
        lines.append('逐折核验 E61 模型、三个优化器、4096 关系对、原索引和四类 RNG。谱基、Query 与 E82.3.1 同折 6081 个参数逐元素一致。G1/G2 新老参数初始化和 EMA 一致，仅 q0 不同。保留主动零 gamma 门控且关闭辅助项时，原前向、损失、梯度、三优化器输出和 RNG 逐位回归 DARG。')
        lines.append(f"Warmup 合法训练 LLR AUC 范围 {100*min(auc):.2f}%–{100*max(auc):.2f}%。首轮 gamma 的分类梯度范数范围 {min(gamma_grad):.4g}–{max(gamma_grad):.4g}。首个真实联合更新后，主分类可达全部可训练路径，最大 0.001×辅助/主分类 Prompt 梯度范数比为 {max(ratios):.4g}。两组方向辅助梯度只进入五类 Prompt 参数。逐患者 Jacobian、144D 投影、均匀退化目标、权重归一化和 8% 上限均通过。")
        lines.append('十折独立数学核对一阶、二阶矩及样本方差校正。评估中 CAM、四路 GCN 与分类头均只执行一次。Best 保存更新前的 EMA，加载和评估均无自动更新。Latest 恢复后下一次训练、评估、EMA、概率和完整 RNG 与运行对象精确一致。')
    return lines


def save_report(lines):
    with r.atomic_writer(r.RESULTS / 'report.md', 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def append_handoff(status):
    marker = f'## Experiment 82.3.2 · DA-SRPE（{status["status"]}）'
    lines = ['', marker, '', f'实际目录 `{r.HERE}`。正式训练 {status["completed_tasks"]}/20，Best 审计 {status["best_pass"]}/20，Latest 四优化器恢复 {status["latest_pass"]}/20。', '',
             '核对原论文、原作者代码及 E82.3.1 固定提交。以 E74.4-G2 的 PLNNoPEN 和 CAM144 为基座，沿用 E82.3.1 谱统计与 EMA。以原分类头真类 Margin 梯度派生的方向 KL 替换 Prompt 类别 CE，新增从精确零启动的患者残差门控。Best 使用更新前 EMA，Latest 保存更新后 EMA。', '']
    for paragraph in preflight_summary(status):
        lines.extend([paragraph, ''])
    if status.get('aggregates'):
        ag = status['aggregates']
        lines.extend(['十折均值 ± population SD，单位%。B0、B1、B2 分别为 DARG、E82.3.1-G1、E82.3.1-G2。', '', *summary_table(ag['summary']['all10'], (*r.GROUPS, *BASELINES)), ''])
        for label in ag['paired']['all10']:
            parts = ag['paired']['all10'][label]['mean_pp']
            lines.extend([f"{label} 的六项差值依次为 " + '/'.join(f'{parts[k]:+.4f}' for k in METRICS) + ' pp。', ''])
        for paragraph in brief_evidence(status):
            lines.extend([paragraph, ''])
        counts = status['rng_audit']
        lines.extend([f"共同 DARG RNG 三元组 {counts['b0_common']} 条，同折 G1/G2 三元组 {counts['g1_g2_common']} 条。核查 {counts['ema_updates']} 次 EMA 状态转移。模型根代码、数据及历史保护文件共 {status['protected_files']} 项，SHA-256 核对{'通过' if status['protected_files_sha_match'] else '失败'}。科研判定为 **{ag['decision']['verdict']}**。", ''])
    else:
        lines.extend([f"当前阻断原因见 `{r.HERE / 'results/audit_status.json'}`。{status.get('blocking_reason', '')}", ''])
    lines.extend([f"完整报告见 [Experiment 82.3.2 报告]({r.HERE / 'results/report.md'})。", ''])
    addition = '\n'.join(lines).encode('utf-8')
    with HANDOFF.open('r+b') as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        original = fp.read()
        if marker.encode() in original:
            assert addition in original, 'handoff marker exists with a different record'
            return {'path': str(HANDOFF), 'verified': True, 'sha256': r.sha_file(HANDOFF), 'already_present': True}
        fp.seek(0, 2)
        fp.write(addition)
        fp.flush()
        os.fsync(fp.fileno())
        fp.seek(0)
        assert fp.read() == original + addition
    readback = HANDOFF.read_bytes()
    assert readback[:len(original)] == original and readback[len(original):].startswith(addition)
    return {'path': str(HANDOFF), 'verified': True, 'prefix_unchanged': True, 'sha256': r.sha_file(HANDOFF)}


def protected_differences(manifest):
    protected = manifest.get('source_maps', {}).get('protected', {})
    return [file for file, digest in protected.items() if not Path(file).is_file() or r.sha_file(file) != digest]


def csv_map(path):
    with path.open(newline='') as fp:
        return {(row['group'], int(row['fold'])): row for row in csv.DictReader(fp)}


def audit_all():
    pre = r.passing_preflight()
    manifest = r.read_json(r.HERE / 'manifest.json')
    assert all(r.task_complete(g, f) for g in r.GROUPS for f in r.FOLDS)
    before_changes = protected_differences(manifest)
    if before_changes:
        raise RuntimeError(f'protected historical sources changed: {before_changes}')
    code = r.implementation()
    assert code == manifest['source_maps']['implementation']
    text = r.torch_load(r.PROMPT_CACHE)
    prompt = r.verify_text(text)
    sites = (r.runtime().EXP, r.E831, r.E831)
    baseline_tables = {b: csv_map(site / 'results/fold_metrics.csv') for b, site in zip(BASELINES, sites)}
    tasks, references, latest_checks, failures, spectrum_checks = [], [], [], [], []
    ac = r.load_module('e8232_pinned_ac_cspe_audit', r.E831 / 'ac_cspe.py')
    start = time.monotonic()
    for fold in r.FOLDS:
        spectrum, cache = r.spectrum_cache(fold), r.torch_load(r.cache_path(fold))
        assert r.sha_file(r.cache_path(fold)) == pre['class_cache_files'][r.cache_path(fold).name]
        state, warm, identity = r.prepare(fold)
        try:
            spectrum_checks.append({'fold': fold, **independent_stats(state, warm, identity, spectrum, cache)})
            references.extend([reference_fold(state, identity, b, baseline_tables[b]) for b in BASELINES])
        except Exception:
            failures.append({'fold': fold, 'stage': 'inheritance', 'traceback': traceback.format_exc()})
        for group in r.GROUPS:
            try:
                task, best, latest, log = read_task(group, fold, pre)
            except Exception:
                failures.append({'fold': fold, 'group': group, 'stage': 'file_provenance', 'traceback': traceback.format_exc()})
                continue
            try:
                old = r.torch_load(r.E831 / 'runtime/checkpoints' / f'{group}_f{fold:02d}.best.pt')
                with r.isolated_rng(r.CFG['initialization_seed_base'] + fold):
                    previous = ac.ACCSPE(spectrum['spectrum'], text, cache['stats'], group, fold,
                                         r.CFG['class_statistics'], r.CFG['ema']['alpha']).to(state['x'].device)
                previous.load_state_dict(old['current']['cspe'], strict=True)
                previous.eval()
                if group == 'G2':
                    g1, _, _ = r.make_module(state, 'G1', spectrum, cache, text)
                    snapshot = r.torch_load(r.paths('G1', fold)[1])
                    g1.load_state_dict(snapshot['current']['cspe'], strict=True)
                    g1.eval()
                else:
                    g1 = None
                tasks.append(audit_best(group, fold, task, best, latest, log, spectrum, cache, text, pre, previous, g1))
                print(f'BEST PASS {group}/{fold:02d}', flush=True)
            except Exception:
                failures.append({'fold': fold, 'group': group, 'stage': 'Best', 'traceback': traceback.format_exc()})
                print(f'BEST FAIL {group}/{fold:02d}', flush=True)
            try:
                latest_checks.append({'fold': fold, 'group': group, **audit_latest(group, fold, task, latest, spectrum, cache, text, log)})
                print(f'LATEST PASS {group}/{fold:02d}', flush=True)
            except Exception:
                failures.append({'fold': fold, 'group': group, 'stage': 'Latest', 'traceback': traceback.format_exc()})
                print(f'LATEST FAIL {group}/{fold:02d}', flush=True)
        del state, warm
    paired_epochs = 0
    if len(tasks) == r.CFG['expected_tasks']:
        lookup = {(a['group'], a['fold']): a for a in tasks}
        rows = [row for f in r.FOLDS for row in lookup[('G1', f)]['test_rows_1based']]
        assert sorted(rows) == list(range(1, r.CFG['subjects']+1))
        for fold in r.FOLDS:
            a, b = lookup[('G1', fold)], lookup[('G2', fold)]
            assert a['initial_parameter_sha'] == b['initial_parameter_sha'] and a['test_rows_1based'] == b['test_rows_1based']
            assert a['initial_ema_sha'] == b['initial_ema_sha']
            for first, second in zip(a['rng_trace'], b['rng_trace']):
                assert first == second, (fold, first['epoch'])
                paired_epochs += 1
    after_changes = protected_differences(manifest)
    if after_changes:
        failures.append({'stage': 'protected_sources', 'changed': after_changes})
    assert code == r.implementation()
    passed = not failures and len(tasks) == len(latest_checks) == r.CFG['expected_tasks']
    common = sum(a['selection_trace']['common_b0_rng_epochs'] for a in tasks)
    status = {'status': 'COMPLETE_AUDITED' if passed else 'AUDIT_FAILED', 'completed_tasks': r.CFG['expected_tasks'],
              'best_pass': len(tasks), 'latest_pass': len(latest_checks), 'tasks': tasks,
              'historical_references': references, 'latest': latest_checks, 'failures': failures,
              'protected_files': len(manifest['source_maps']['protected']), 'protected_files_sha_match': not after_changes,
              'fold_spectra_and_stats': spectrum_checks, 'prompt_geometry': prompt,
              'rng_audit': {'b0_common': common, 'g1_g2_common': paired_epochs,
                            'ema_updates': sum(a['selection_trace']['ema_trajectory']['verified_updates'] for a in tasks),
                            'b0_unavailable': sum(len(a['rng_trace']) for a in tasks)-common},
              'implementation': code, 'audit_wall_seconds': time.monotonic() - start}
    if len(tasks) == r.CFG['expected_tasks'] and len(references) == len(BASELINES) * r.CFG['folds']:
        status['aggregates'] = aggregate(tasks, references, passed)
    if failures:
        status['blocking_reason'] = '独立审计存在失败项。详情在 audit_status.json 的 failures 字段。'
    write_delivery(status, manifest)
    print(f"{status['status']} best={len(tasks)}/20 latest={len(latest_checks)}/20", flush=True)
    return 0 if passed else 65


def write_delivery(status, manifest):
    r.atomic_json(r.RESULTS / 'audit_status.json', status)
    lines = make_report(status, manifest)
    save_report(lines)
    status['handoff'] = append_handoff(status)
    r.atomic_json(r.RESULTS / 'audit_status.json', status)
    lines.extend([f"Handoff 已追加至 `{HANDOFF}` 并重新读取确认。", ''])
    save_report(lines)
    verdict = status.get('aggregates', {}).get('decision', {}).get('verdict')
    r.update_manifest(status=status['status'], completed_tasks=status['completed_tasks'], best_audit_pass=status['best_pass'],
                      latest_audit_pass=status['latest_pass'], scientific_verdict=verdict,
                      handoff=status['handoff'], result_sha256={str(p.relative_to(r.HERE)): r.sha_file(p)
                         for p in sorted(r.RESULTS.iterdir()) if p.is_file()})


def record_block(failure=None):
    manifest = r.read_json(r.HERE / 'manifest.json')
    reason = failure or manifest.get('blocking_failure') or manifest.get('execution', {}).get('failures')
    status_name = manifest['status']
    if status_name not in ('PREFLIGHT_BLOCKED', 'TRAINING_STOPPED', 'AUDIT_FAILED'):
        status_name = 'AUDIT_FAILED'
    status = {'status': status_name, 'completed_tasks': manifest.get('completed_tasks', 0),
              'best_pass': manifest.get('best_audit_pass', 0), 'latest_pass': manifest.get('latest_audit_pass', 0),
              'blocking_reason': reason, 'failures': [reason] if reason else []}
    write_delivery(status, manifest)
    return 65


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-block', action='store_true')
    args = parser.parse_args()
    if args.record_block:
        return record_block()
    try:
        return audit_all()
    except Exception as exc:
        traceback.print_exc()
        return record_block({'type': type(exc).__name__, 'reason': str(exc), 'traceback': traceback.format_exc()})


if __name__ == '__main__':
    raise SystemExit(main())
