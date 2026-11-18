#!/usr/bin/env python3
"""Independent AC-CSPE checkpoint replay, paired analysis and handoff."""
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

from ac_cspe import EMA_BUFFERS, EPS, FOLD2_ROWS, QUERIES, SPECTRUM_BUFFERS

METRICS = r.METRICS
DISPLAY = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')
BASELINES = ('B0', 'B1', 'B2')
COMPARISONS = tuple((group, baseline) for group in r.GROUPS for baseline in BASELINES) + (('G2', 'G1'),)
TREATMENTS = {'T0': 'full', 'T1': 'full', 'T2': 'semantic_off', 'T3': 'residual_off', 'T4': 'full', 'T5': 'gate_off'}
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
    aux = []
    legal_np = np.setdiff1d(state['train_np'], state['test_np'], assume_unique=True)
    assert np.array_equal(legal_np, r.legal_train(state))

    def replacement(_head, args):
        assert len(args) == 2 and args[0].shape == (871, 144)
        z = args[0]
        before = r.rng_sha()
        changed, details = module(z, intervention=intervention)
        if training or grad:
            rows = torch.as_tensor(legal_np, device=z.device)
            aux.append(module.prompt_aux(z.detach()[rows]))
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
    if aux:
        rows = torch.as_tensor(legal_np, device=d['z'].device)
        labels = state['y'][rows]
        ce = F.cross_entropy(aux[0], labels, reduction='none')
        prompt = (ce[labels == 0].mean() + ce[labels == 1].mean()) * 0.5
        delta, z = d['delta'][rows], d['z'][rows]
        res = (delta.square().sum(-1) / (z.square().sum(-1) + EPS)).mean()
        out.update(prompt=prompt, res=res, darg_total=out['total'], aux_scores=aux[0])
        out['total'] = out['total'] + r.CFG['residual_loss_coefficient'] * res + r.CFG['prompt_loss_coefficient'] * prompt
    return out


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
    delta = unbounded * scale
    if module.fold == 2:
        delta[list(FOLD2_ROWS)] = 0
        assert torch.count_nonzero(d['delta'][list(FOLD2_ROWS)]) == 0
    tracked = {'h': h, 'hhat': hh, 'a': a, 'tokens': tokens, 'raw_llr': raw, 'bounded_llr': bounded,
               'c_n': coverage[:, 0], 'c_a': coverage[:, 1], 'p_n': prior[:, 0], 'p_a': prior[:, 1],
               'g_td': evidence[:, 0], 'g_asd': evidence[:, 1], 'w_n': weights[:, 0], 'w_a': weights[:, 1],
               'scores': class_scores, 'pi': pi, 'w_sem': ws, 'w_sp': wp, 'entropy': entropy, 'null': null,
               'beta': beta, 'w': w, 'delta_raw': unbounded, 'delta': delta}
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


def mechanism(state, module, cache, spectrum, details, out):
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
    rel = np.linalg.norm(plain['delta'], axis=1) / np.linalg.norm(plain['z'], axis=1).clip(EPS)
    warm_stats = cache['stats']
    hh_best = details['hhat'].detach().cpu()
    warm_mean, warm_stable = (warm_stats[key].float() for key in ('mean', 'stable'))
    warm_log_density = -.5 * (warm_stable.log() + (hh_best[:, None] - warm_mean).square() / warm_stable)
    fixed_raw = warm_log_density[:, 0] - warm_log_density[:, 1]
    fixed_bounded = module.llr_bound * torch.tanh(fixed_raw / module.llr_bound)
    result = {}
    for split, ids in [('legal_train', legal), ('test', state['test_np'])]:
        d = {key: value[ids] for key, value in plain.items()}
        labels = y_full[ids]
        attn = d['a'][:, :, :16]
        attn_norm = attn / np.linalg.norm(attn, axis=-1, keepdims=True).clip(1e-12)
        pair = (attn_norm @ attn_norm.transpose(0, 2, 1)).mean(0)
        gain = np.where(labels == 0, 1, -1) * margin_gain_asd[ids]
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
            'true_logit_margin_change': distribution(gain),
            'true_margin_increased_share': float((gain > 0).mean()),
            'true_logit_margin_change_by_class': [distribution(gain[labels == code]) for code in (0, 1)],
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
def interventions(state, module, text, best_out, comparison_module, y, cache):
    full = probability(best_out, state)
    pred = full.argmax(-1)
    d = best_out['details']
    idx = state['test_np']
    saved = module.q0.clone()
    ema = r.calibration_state(module)
    initial = r.sha(r.cpu_state(module))
    rng = r.rng_sha()
    response = []
    for code, method in TREATMENTS.items():
        try:
            if code == 'T1':
                module.q0.copy_(text['q_anonymous'].to(module.q0.device))
            if code == 'T4':
                restore_ema(module, {'class_mean': cache['stats']['mean'].float(),
                                     'class_var': cache['stats']['stable'].float()})
            out = best_out if code == 'T0' else independent_forward(state, module, method)
            p = probability(out, state)
            change = np.abs(p[:, 0] - full[:, 0])
            if code == 'T2':
                assert torch.equal(out['details']['w'], d['w_sp'])
            if code == 'T3':
                assert torch.count_nonzero(out['details']['delta']) == 0
            if code == 'T5':
                d5 = out['details']
                assert torch.equal(d5['w_n'], d['p_n']) and torch.equal(d5['w_a'], d['p_a'])
                assert all(torch.equal(d5[k], d[k]) for k in ('scores', 'pi', 'raw_llr', 'w_sp'))
            result = {'intervention': code, 'metrics': independent_metrics(p, y),
                      'mean_abs_asd_probability_change': float(change.astype(np.float64).mean()),
                      'max_abs_asd_probability_change': float(change.max()), 'n': int(len(y)),
                      'changes': changes(pred, p.argmax(-1), y),
                      'flipped_rows_1based': (idx[p.argmax(-1) != pred] + 1).tolist(),
                      'probability_asd': p[:, 0].tolist(), 'prediction': p.argmax(-1).tolist()}
            if code in ('T1', 'T4', 'T5'):
                assert torch.equal(out['details']['w_sp'], d['w_sp'])
                result['routing_change_same_best_cam'] = routing_difference(d, out['details'], idx)
            response.append(result)
        finally:
            if code == 'T1':
                module.q0.copy_(saved)
            if code == 'T4':
                restore_ema(module, ema)
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
        cls, graph, pair, adv, preserve, darg, res, prompt, total = row['losses'].values()
        assert np.isclose(darg, cls + graph + .05*pair + .01*adv + .01*preserve, rtol=1e-6, atol=1e-7)
        assert np.isclose(total, darg + .001*res + .01*prompt, rtol=1e-6, atol=1e-7)
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
                     'prompt_ce_first_best_last': [records[0]['losses']['prompt'], chosen['losses']['prompt'], records[-1]['losses']['prompt']]}


def reference_fold(state, identity, label, csv_lookup):
    fold = state['fold']
    y = state['labels_np'][state['test_np']]
    if label == 'B0':
        ckpt = r.torch_load(r.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('test_probability', 'test_prediction', 'best_epoch_1based'))
        source_row = csv_lookup[(r.BASE_GROUP, fold)]
    else:
        source = r.E822 if label == 'B1' else r.E823
        ckpt = r.torch_load(source / 'runtime/checkpoints' / f'G2_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('probability', 'prediction', 'best_epoch'))
        source_row = csv_lookup[('G2', fold)]
    assert ckpt['identity'] == identity and len(p) == len(y) and np.array_equal(pred, p.argmax(-1))
    metric = independent_metrics(p, y)
    published = metric if label == 'B2' else {**metric, 'prob_auc': float(roc_auc_score(y, p[:, 1]))}
    same_metrics(published, ckpt['metrics'])
    same_metrics(published, {k: float(source_row[k]) / (1 if label == 'B0' else 100) for k in METRICS})
    return {'group': label, 'fold': fold, 'best_epoch': epoch, 'metrics': metric, 'prediction': pred.tolist(),
            'probability': p.tolist(), 'source_original_prob_auc': published['prob_auc'],
            'source_original_positive_class': 'ASD' if label == 'B2' else 'TD',
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


def audit_best(group, fold, task, best, latest, log, spectrum, cache, text, pre, other_module=None):
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
    grad = r.gradient_audit(state, module, out, enforce_gate=False)
    assert d['scores'].shape == (871, 2) and out['aux_scores'].shape == (len(cache['legal_train']), 2)
    torch.testing.assert_close(out['aux_scores'], d['scores'][cache['legal_train']], rtol=2e-6, atol=2e-6)
    best_cam, actual_observation = independent_observation(state, out, spectrum)
    logs, selection_trace = verify_selection(state, task, best, latest, log, cache, best_cam, actual_observation)
    diagnose = mechanism(state, module, cache, spectrum, d, out)
    original_y = state['y'][state['test_np']].clone()
    original_y_numpy = state['labels_np'][state['test_np']].copy()
    state['y'][state['test_np']] = 1 - original_y
    state['labels_np'][state['test_np']] = 1 - original_y_numpy
    try:
        changed = independent_forward(state, module)
        assert torch.equal(changed['logp'], out['logp'])
        assert all(torch.equal(changed['details'][key], d[key]) for key in ('raw_llr', 'scores', 'pi', 'w_sem', 'delta'))
        _, new_observation = independent_observation(state, changed, spectrum)
        assert r.sha(new_observation) == r.sha(actual_observation)
    finally:
        state['y'][state['test_np']] = original_y
        state['labels_np'][state['test_np']] = original_y_numpy
    assert r.rng_sha() == best['rng_sha']
    readonly, comparison = interventions(state, module, text, out, other_module, y, cache) if group == 'G2' else ([], None)
    assert r.component_digests(r.bundle(state, module, opt)) == best['digests']
    return {'status': 'PASS', 'group': group, 'fold': fold, 'metrics': metrics, 'best_epoch': best['best_epoch'],
            'epochs_executed': task['epochs_executed'], 'initial_parameter_sha': task['initial_parameter_sha'],
            'initial_ema_sha': task['initial_ema_sha'], 'best_calibration_sha': best['calibration_sha'],
            'best_pre_update_ema_count': int(module.ema_updates),
            'best_file_sha': task['best_sha'], 'probability_max_abs_replay_error': 0., 'optimizers': optimizers,
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
            'terminal_post_update_ema_count': int(terminal_ema['ema_updates']), 'terminal_post_update_ema_sha': r.sha(terminal_ema),
            'next_step_matches_training_witness': True, 'live_terminal_witness': expected['live_terminal_object'],
            'next_step': actual}


def summed_changes(items):
    return {scope: {metric: sum(x[scope][metric] for x in items)
                    for metric in ('flips', 'corrected', 'harmed', 'net')}
            for scope in ('all', 'asd', 'td')}


def scientific_decision(paired, corrections, treatments, audits_pass):
    def delta(group, baseline, key):
        return paired[f'{group}−{baseline}']['mean_pp'][key]

    def net(group):
        return corrections[f'{group}−B0']['all']['net']

    cfg = r.CFG['decision']
    go = {
        'all_audits_pass': audits_pass,
        'g2_acc_higher_than_darg': delta('G2', 'B0', 'acc') > 0,
        'g2_ba_at_least_darg': delta('G2', 'B0', 'hard_auc') >= -1e-10,
        'g2_asd_sensitivity_within_tolerance': delta('G2', 'B0', 'sen') >= cfg['asd_sensitivity_min_delta_pp_vs_darg'],
        'g2_prob_auc_within_tolerance': delta('G2', 'B0', 'prob_auc') >= cfg['prob_auc_min_delta_pp_vs_darg'],
        'g2_acc_and_ba_higher_than_e822': all(delta('G2', 'B1', key) > 0 for key in METRICS[:2]),
        'g2_acc_and_ba_higher_than_g1': all(delta('G2', 'G1', key) > 0 for key in METRICS[:2]),
        'g2_positive_net_patient_corrections': net('G2') > 0,
    }
    query_change = treatments['T1']
    go['T1_removal_diagnostic_loss'] = (query_change['changes']['all']['net'] < 0 and query_change['mean_delta_pp']['acc'] < 0 and
        query_change['mean_delta_pp']['hard_auc'] <= 1e-10 and
        query_change['folds_with_negative_net'] >= cfg['min_query_removal_net_loss_folds'])
    partial = {g: ((delta(g, 'B0', 'acc') > 0 or delta(g, 'B0', 'hard_auc') > 0) and
                   delta(g, 'B0', 'acc') >= -1e-10 and delta(g, 'B0', 'hard_auc') >= -1e-10 and net(g) > 0) for g in r.GROUPS}
    verdict = 'GO' if all(go.values()) else 'PARTIAL' if audits_pass and any(partial.values()) else 'NO-GO'
    return {'verdict': verdict, 'go_checks': go, 'partial_by_group': partial,
            'clinical_semantic_contribution_established_under_this_protocol': all(go.values())}


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
    summaries, comparisons, corrections, treatments = {}, {}, {}, {}
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
        for treatment in TREATMENTS:
            selected = [next(row for row in by('G2', f)['treatments'] if row['intervention'] == treatment) for f in folds]
            original = [by('G2', f)['metrics'] for f in folds]
            delta = {k: float(100 * np.mean([a['metrics'][k] - ref[k] for a, ref in zip(selected, original)])) for k in METRICS}
            treatments[scope][treatment] = {'mean_delta_pp': delta,
                'mean_abs_asd_probability_change': float(sum(a['n'] * a['mean_abs_asd_probability_change'] for a in selected) / sum(a['n'] for a in selected)),
                'max_abs_asd_probability_change': max(a['max_abs_asd_probability_change'] for a in selected),
                'changes': summed_changes([a['changes'] for a in selected]),
                'folds_with_negative_net': sum(a['changes']['all']['net'] < 0 for a in selected)}
    r.write_csv(r.RESULTS / 'fold_metrics.csv', fold_rows)
    r.write_csv(r.RESULTS / 'paired_comparisons.csv', pair_rows)
    decision = scientific_decision(comparisons['all10'], corrections['all10'], treatments['all10'], audits_pass)
    return {'summary': summaries, 'paired': comparisons, 'corrections': corrections, 'treatments': treatments,
            'decision': decision, 'fold_rows': fold_rows}


SOURCE_NOTES = [
    '原论文的 PLN 通过 CAM 做患者内跨模态交互，PEN 中的 SPGS 用文本相似度重加权局部 token，RES 融合局部和全局表示。原作者 `network.py` 用冻结 CLIP RN50 编码文本并参与分类前嵌入。E74.4-G2 历史基座运行 vendor `PLNNoPEN`，DARG 四种患者关系及 Positive-DCR 产生的 320 维融合特征走 InputLayer、Encoder 和 FeedForward，CAM144 展平后进入原 OutputLayer。AC-CSPE 只在该 CAM144 接口加入残差。 [原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者 network.py](https://github.com/wanxixi11/SPromptGL/blob/ba3b9446fa451fde6864d6555bdf324de0c4b64a/network.py)',
    'GitHub 的 E82.3 分支中 `cspe.py`、`run.py`、`audit.py`、`config.json` 与 `results/report.md` 全部和本地原件匹配。E82.2 PSCE 使用统一 coverage 和反事实教师。E82.3 使用固定合法训练类别高斯拟合，LLR 仅经类别分数影响混合系数 pi，谱方向权重由各组 coverage 直接归一化。原辅助 CE 只更新 Prompt 参数，主分类保持端到端。E82.3-G2 的合法训练平滑 LLR AUC 从 warmup 71.39% 降至 Best 45.22%，三项 Best 只读干预均零硬翻转。 [E82.3 固定提交](https://github.com/tangaii/SPromptGL/tree/45ff89565b7c154d3299f4efc8d1dc176d6302c1/experiment/experiment82_3_cspe) · [E82.3 原始报告](/model/spromptgl/experiment/experiment82_3_cspe/results/report.md) · [E82.2 原始报告](/model/spromptgl/experiment/experiment82_2_psce/results/report.md)',
    'AC-CSPE 的第一项改动从 E82.3 的合法 E61 epoch 10 类别拟合启动。先由 E82.3 样本方差换算总体二阶原始矩，再以 alpha=0.1 对当前合法训练 CAM 的一阶和二阶原始矩做 EMA。计算使用 CPU float64 和原谱缓存，按 n/(n−1) 校正样本方差并应用原稳定公式，推理沿用 float32 buffer。每轮依次更新四个优化器、按旧 EMA 做一次历史评估并冻结选中的 Best、从该评估捕获的合法训练 CAM 更新 EMA。Latest 存更新后的 buffer 和用于复核终止预测的上一状态。',
    '第二项改动保留正常样和异常样的四加四连续 coverage。类别分数仍以门控前归一化 coverage 加权正负 LLR 的 softplus。方向权重对原 coverage 逐方向乘同一 softplus，再分组归一化并按两个类别分数的 softmax 混合。LLR 保持 ASD 对 TD 的一维高斯密度差，以 `8*tanh(r/8)` 平滑。ABIDE `DX_GROUP` 减 1 后为 ASD=0、TD=1。',
    'G1 用历史 Gram 保持匿名 Query，G2 用历史医学 Query。八条文本原样缓存，包括 `Nromal brain volume`。正常样和异常样是描述性分组假设，其临床标志物身份待验证。两组同折的 6081 个参数与 E82.3 同折启动参数逐元素一致，只有固定 Query 不同。原 U、mu、lambda、loglambda、nu、方向 token、8×17 注意力、温度 0.5、null、3–32–1 谱网络及 `w_sp`、detach beta、谱残差重建与 8% 平滑上限均沿用。',
    '主分类更新 DARG、CAM、原 OutputLayer 和完整谱证据分支。类别平衡 CE 只读取合法训练 `z.detach()` 以更新 Prompt 选择参数。总损失保留分类、图、关系、对抗、保留项和 `0.001*L_res + 0.01*L_Prompt`，依次更新 opt_base、opt_cam、opt_added 和 AdamW opt_psce（lr=0.0005，weight decay=0.01）。',
    '每折承接 E61 epoch 10 模型、三个优化器和完整 RNG，按 E61-E2 设置恒等校准、reliability=1、冻结零 reliability 门控并继承 DARG 原自适应门控起点。Fold2 继续按 E69.3 历史规则追加 377、407 和生成该折有效 4096 关系对。两人排除于新增统计和辅助损失，新增残差严格为零。程序只读复用旧谱基、Query 和类别拟合文件。详细 SHA 和源码身份列于 manifest.json。'
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
    lines = ['## 类别分数、方向选择与梯度', '']
    for group in r.GROUPS:
        subset = [a for a in status['tasks'] if a['group'] == group]
        def avg(extract):
            return float(np.mean([extract(a) for a in subset]))
        lines.extend([
            f'### {group}', '',
            f"Warmup 合法训练行的平均原始/平滑方向均值 LLR AUC 为 {100*avg(lambda a: a['mechanism']['warmup_train_llr']['raw_mean_score_roc_auc']):.2f}%/{100*avg(lambda a: a['mechanism']['warmup_train_llr']['bounded_mean_score_roc_auc']):.2f}%。Best 合法训练和测试行的平滑 LLR AUC 为 {100*avg(lambda a: a['mechanism']['legal_train']['llr']['bounded_mean_score_roc_auc']):.2f}%/{100*avg(lambda a: a['mechanism']['test']['llr']['bounded_mean_score_roc_auc']):.2f}%。测试 |原始 LLR|>8 占 {100*avg(lambda a: a['mechanism']['test']['llr']['raw_abs_gt_8_share']):.3f}%，|平滑 LLR|≥7.6 占 {100*avg(lambda a: a['mechanism']['test']['llr']['bounded_abs_ge_7_6_share']):.3f}%，平滑前后绝对差均值 {avg(lambda a: a['mechanism']['test']['llr']['abs_smoothing']['mean']):.4f}。", '',
            f"Warmup 各方向对称高斯 KL 均值 {avg(lambda a: np.mean(a['mechanism']['class_separation_and_drift']['warm_symmetric_gaussian_kl'])):.4f}。从 Warmup 到 Best，合法训练两类均值漂移 RMSE {avg(lambda a: a['mechanism']['class_separation_and_drift']['mean_drift_rmse']):.4f}，对数方差漂移绝对均值 {avg(lambda a: a['mechanism']['class_separation_and_drift']['log_var_drift_abs_mean']):.4f}。两类均值、方差、标准化均值差和逐方向分离度均保存在审计结果。", '',
            f"对同一 Best 的 CAM 应用冻结 warmup 统计，合法训练和测试 LLR AUC 分别为 "
            f"{100*avg(lambda a: a['mechanism']['legal_train']['frozen_warmup_llr_same_best_cam']['bounded_mean_score_roc_auc']):.2f}% 和 "
            f"{100*avg(lambda a: a['mechanism']['test']['frozen_warmup_llr_same_best_cam']['bounded_mean_score_roc_auc']):.2f}%。"
            f"EMA 对当前合法训练类分布的均值 RMSE 为 {avg(lambda a: a['mechanism']['class_separation_and_drift']['ema_mean_mismatch_rmse']):.4f}。"
            f"稳定方差的绝对对数失配从固定统计的 "
            f"{avg(lambda a: a['mechanism']['class_separation_and_drift']['warm_log_stable_variance_mismatch_to_best']):.4f} "
            f"变为 EMA 的 {avg(lambda a: a['mechanism']['class_separation_and_drift']['ema_log_stable_variance_mismatch']):.4f}。", '',
            f"每轮 EMA 均值更新 RMSE 的折均为 {avg(lambda a: a['selection_trace']['ema_trajectory']['mean_update_rmse']['mean']):.5f}，"
            f"最后四分之一轮为 {avg(lambda a: a['selection_trace']['ema_trajectory']['last_quartile_mean_update_rmse']['mean']):.5f}。"
            f"对应的稳定方差对数变化为 {avg(lambda a: a['selection_trace']['ema_trajectory']['log_variance_update_abs_mean']['mean']):.5f} 和 "
            f"{avg(lambda a: a['selection_trace']['ema_trajectory']['last_quartile_log_variance_update_abs_mean']['mean']):.5f}。"
            f"全部轮次稳定方差范围 "
            f"{min(a['selection_trace']['ema_trajectory']['minimum_stable_variance'] for a in subset):.6g} 至 "
            f"{max(a['selection_trace']['ema_trajectory']['maximum_stable_variance'] for a in subset):.6g}。", '',
            f"Best 的 Prompt 类平衡 CE 在合法训练为 {avg(lambda a: a['mechanism']['legal_train']['prompt_ce']['balanced_ce']):.4f}，在测试为 {avg(lambda a: a['mechanism']['test']['prompt_ce']['balanced_ce']):.4f}。测试 Prompt 分数 AUC {100*avg(lambda a: a['mechanism']['test']['prompt_ce']['scores_roc_auc']):.2f}%，真类分数优势均值 {avg(lambda a: a['mechanism']['test']['prompt_ce']['true_class_margin']['mean']):+.4f}，真类分数严格占优比例 {100*avg(lambda a: a['mechanism']['test']['prompt_ce']['true_class_win_share']):.2f}%。`pi_ASD`/`pi_TD` 测试均值 {avg(lambda a: a['mechanism']['test']['pi_asd']['mean']):.4f}/{avg(lambda a: a['mechanism']['test']['pi_td']['mean']):.4f}。", '',
            f"测试 Normal-like/Abnormal-like coverage 均值 {avg(lambda a: a['mechanism']['test']['coverage_n']['mean']):.4f}/{avg(lambda a: a['mechanism']['test']['coverage_a']['mean']):.4f}。两组 `w_N`/`w_A` 的 L1 差 {avg(lambda a: a['mechanism']['test']['w_n_w_a_l1']['mean']):.4f}，八条 Prompt 两两注意力余弦均值 {avg(lambda a: a['mechanism']['test']['prompt_pairwise_cosine']):.4f}，null 均值 {avg(lambda a: a['mechanism']['test']['null']['mean']):.4f}。`w_sem` 的归一化熵 {avg(lambda a: a['mechanism']['test']['entropy']['mean']):.4f}，与 `w_sp` 的余弦相似度 {avg(lambda a: a['mechanism']['test']['w_sem_w_sp_cosine']['mean']):.4f}。", '',
            f"测试 `w_sem` 的折内 Top1 众数占比均值 {100*avg(lambda a: a['mechanism']['test']['routing']['w_sem']['top1_mode_share']):.2f}%，全测试行共享同一 Top1 的折数 {sum(a['mechanism']['test']['routing']['w_sem']['top1_fixed'] for a in subset)}/10。患者方向权重相对折内均值的平均 L1 偏差 {avg(lambda a: a['mechanism']['test']['routing']['w_sem']['patient_variation_l1']):.4f}，平均有效方向数 {avg(lambda a: a['mechanism']['test']['routing']['w_sem']['effective_direction_count']['mean']):.2f}/16。", '',
            f"门控前后 Normal-like 和 Abnormal-like 权重的平均 L1 差为 "
            f"{avg(lambda a: a['mechanism']['test']['gate_n_shift_l1']['mean']):.4f} 和 {avg(lambda a: a['mechanism']['test']['gate_a_shift_l1']['mean']):.4f}。"
            f"`w_sem` 与 `w_sp` 的平均 L1 差为 {avg(lambda a: a['mechanism']['test']['w_sem_w_sp_l1']['mean']):.4f}。", '',
            f"测试 beta 均值 {avg(lambda a: a['mechanism']['test']['beta']['mean']):.4f}。残差相对范数均值 {100*avg(lambda a: a['mechanism']['test']['relative_residual']['mean']):.4f}%，最大值 {100*max(a['mechanism']['test']['relative_residual']['max'] for a in subset):.4f}%。真类 margin 变化均值 {avg(lambda a: a['mechanism']['test']['true_logit_margin_change']['mean']):+.5f}，改善比例 {100*avg(lambda a: a['mechanism']['test']['true_margin_increased_share']):.2f}%。", '',
            f"Preflight 主分类、0.01 倍辅助和 0.001 倍残差的 Prompt 参数梯度范数折均分别为 "
            f"{avg(lambda a: a['historical_preflight_gradients']['norms']['classification']['prompt_parameters']):.6g}、"
            f"{avg(lambda a: a['historical_preflight_gradients']['norms']['weighted_auxiliary']['prompt_parameters']):.6g}、"
            f"{avg(lambda a: a['historical_preflight_gradients']['norms']['weighted_residual']['prompt_parameters']):.6g}。"
            f"Best 上的对应数值为 {avg(lambda a: a['best_gradients']['norms']['classification']['prompt_parameters']):.6g}、"
            f"{avg(lambda a: a['best_gradients']['norms']['weighted_auxiliary']['prompt_parameters']):.6g}、"
            f"{avg(lambda a: a['best_gradients']['norms']['weighted_residual']['prompt_parameters']):.6g}。"
            f"训练首轮/Best轮/末轮辅助 CE 折均分别为 "
            f"{'/'.join(f'{avg(lambda a: a["selection_trace"]["prompt_ce_first_best_last"][i]):.4f}' for i in range(3))}。", '',
        ])
        gradients = [a['selection_trace']['prompt_gradient_alignment'] for a in subset]
        def grad_mean(part, extract):
            return float(np.mean([extract(g[part]) for g in gradients]))
        lines.extend([
            f"训练全部轮次中，加权辅助梯度对分类梯度的范数比折均 "
            f"{grad_mean('all', lambda g: g['weighted_auxiliary_to_cls_ratio']['mean']):.4f}。"
            f"余弦均值和负余弦比例为 {grad_mean('all', lambda g: g['cosine']['mean']):+.4f} 和 "
            f"{100*grad_mean('all', lambda g: g['negative_share']):.2f}%。"
            f"训练最后四分之一轮的余弦均值和负余弦比例为 "
            f"{grad_mean('final_quartile', lambda g: g['cosine']['mean']):+.4f} 和 "
            f"{100*grad_mean('final_quartile', lambda g: g['negative_share']):.2f}%。"
            f"最后四分之一轮中负余弦占多数的折有 {sum(g['final_quartile']['negative_share'] > .5 for g in gradients)}/10。"
            f"最长连续负余弦为 {max(g['all']['longest_negative_streak'] for g in gradients)} 轮。"
            "负余弦描述 Prompt 参数上的梯度分量冲突，正负比例和梯度大小一起判断其持续性。", '',
        ])
    g2 = [a for a in status['tasks'] if a['group'] == 'G2']
    for explanation, rows in [
        ('同一 G2 Best、EMA 和 CAM 下替换固定 Query', [a['treatments'][1]['routing_change_same_best_cam'] for a in g2]),
        ('在同一 G2 CAM 上分别运行 G1 和 G2 的已训练选择器与各自 EMA', [a['trained_g1_vs_g2_same_g2_cam'] for a in g2]),
        ('同一 G2 Best 上将 EMA 替换为 warmup 统计', [a['treatments'][4]['routing_change_same_best_cam'] for a in g2]),
        ('同一 G2 Best 上关闭逐方向门控', [a['treatments'][5]['routing_change_same_best_cam'] for a in g2]),
    ]:
        lines.extend([f"{explanation}时，注意力平均绝对差 {np.mean([v['attention_mean_abs'] for v in rows]):.6f}，`w_N`/`w_A`/`w_sem` 的平均 L1 差 "
                      f"{'/'.join(f'{np.mean([v[k + "_l1"] for v in rows]):.4f}' for k in ('w_n','w_a','w_sem'))}，"
                      f"`w_sem` Top1 一致率 {100*np.mean([v['w_sem_top1_agreement'] for v in rows]):.2f}%。", ''])
    lines.extend(['各折的 LLR 分位数、注意力、逐方向分布、类漂移、逐患者预测和梯度汇总见审计 JSON。每轮的一阶及二阶原始矩、稳定方差、合法类别数及梯度余弦和范数均在 `runtime/logs/G*_f*.jsonl`。方向编号由各折谱基决定。', ''])
    return lines


def calibration_evidence(status):
    g2 = [a for a in status['tasks'] if a['group'] == 'G2']
    def mean(extractor):
        return float(np.mean([extractor(row['mechanism']) for row in g2]))
    prior = 100 * mean(lambda m: m['warmup_train_llr']['bounded_mean_score_roc_auc'])
    train = 100 * mean(lambda m: m['legal_train']['llr']['bounded_mean_score_roc_auc'])
    test = 100 * mean(lambda m: m['test']['llr']['bounded_mean_score_roc_auc'])
    prompt = 100 * mean(lambda m: m['test']['prompt_ce']['scores_roc_auc'])
    fixed_train = 100 * mean(lambda m: m['legal_train']['frozen_warmup_llr_same_best_cam']['bounded_mean_score_roc_auc'])
    fixed_test = 100 * mean(lambda m: m['test']['frozen_warmup_llr_same_best_cam']['bounded_mean_score_roc_auc'])
    treatment = status['aggregates']['treatments']['all10']
    gate_off = treatment['T5']['changes']['all']
    l1 = np.mean([a['treatments'][5]['routing_change_same_best_cam']['w_sem_l1'] for a in g2])
    patient_variation = mean(lambda m: m['test']['routing']['w_sem']['patient_variation_l1'])
    return [
        f"G2 warmup 的合法训练平滑 LLR AUC 为 {prior:.2f}%。Best 的合法训练和测试 AUC 为 {train:.2f}% 和 {test:.2f}%。"
        f"同一 Best CAM 使用冻结统计得到 {fixed_train:.2f}% 和 {fixed_test:.2f}%，EMA 对应变化 {train-fixed_train:+.2f} 和 {test-fixed_test:+.2f} pp。"
        f"Best 测试 Prompt 分数 AUC 为 {prompt:.2f}%。历史 E82.3-G2 Best 合法训练 LLR AUC 为 45.22%。",
        f"G2 测试患者 `w_sem` 相对折内均值的平均 L1 为 {patient_variation:.4f}。固定 Best 关闭逐方向门控后，"
        f"`w_sem` 平均 L1 变化 {l1:.4f}，硬翻转 {gate_off['flips']} 例，净纠错 {gate_off['net']:+d}。"
        f"换回 warmup 统计后，硬翻转 {treatment['T4']['changes']['all']['flips']} 例，净纠错 {treatment['T4']['changes']['all']['net']:+d}。",
    ]


def brief_evidence(status):
    ag = status['aggregates']
    paired = ag['paired']['all10']
    corr = ag['corrections']['all10']
    t = ag['treatments']['all10']
    verdict = ag['decision']['verdict']
    medical = t['T1']['changes']['all']
    if medical['flips'] == 0:
        semantics = f"固定 G2 Best 的医学 Query 换成匿名 Query 后，871 个测试预测中有 0 个硬翻转，ASD 概率平均绝对变化为 {100*t['T1']['mean_abs_asd_probability_change']:.6f} pp。医学语义的独立诊断贡献尚未建立。"
    elif not ag['decision']['go_checks']['T1_removal_diagnostic_loss']:
        semantics = f"固定 G2 Best 换用匿名 Query 后硬翻转 {medical['flips']} 人，干预相对 T0 净纠错 {medical['net']:+d}。定向诊断损失未达到预先冻结标准，医学语义的独立诊断贡献尚未建立。"
    else:
        semantics = f"固定 G2 Best 换用匿名 Query 后硬翻转 {medical['flips']} 人，干预相对 T0 净纠错 {medical['net']:+d}。该方向支持当前模型在此协议下利用医学 Query，证据仍限于这一次固定模型干预与配对训练。"
    g1, g2 = (paired[f'{g}−B0']['mean_pp'] for g in r.GROUPS)
    g21 = paired['G2−G1']['mean_pp']
    g2b1, g2b2 = (paired[f'G2−{b}']['mean_pp'] for b in ('B1', 'B2'))
    core = (f"G1/G2 相对 DARG 的 ACC 分别变化 {g1['acc']:+.4f}/{g2['acc']:+.4f} pp，BA 分别变化 {g1['hard_auc']:+.4f}/{g2['hard_auc']:+.4f} pp。"
            f"G2 相对 G1 的 ACC 与 BA 差为 {g21['acc']:+.4f}/{g21['hard_auc']:+.4f} pp，"
            f"相对 E82.2-G2 为 {g2b1['acc']:+.4f}/{g2b1['hard_auc']:+.4f} pp，相对 E82.3-G2 为 {g2b2['acc']:+.4f}/{g2b2['hard_auc']:+.4f} pp。"
            f"G2 相对 DARG 的患者净纠错为 {corr['G2−B0']['all']['net']:+d}。")
    if verdict == 'GO':
        boundary = '按冻结条件，AC-CSPE 可留作第二创新模块候选。独立选模和外部验证仍是评估泛化能力的必要证据。'
    elif verdict == 'PARTIAL':
        boundary = '至少一组满足相对 DARG 的预定硬分类改善条件。临床语义和硬分类门槛按冻结判定表逐项核对。'
    else:
        boundary = '按冻结条件停止本候选的推进。LLR 分离、概率排序或方向权重变化本身均不足以推进候选。'
    return [core, semantics, boundary]


def make_report(status, manifest):
    code = status['status']
    best, latest = status['best_pass'], status['latest_pass']
    lines = ['# Experiment 82.3.1 · AC-CSPE', '',
             '**Adaptive Class-Calibrated Prompt Spectral Evidence Learning（自适应类别校准的疾病 Prompt 谱证据学习）**', '',
             f'状态 **{code}**。正式训练 {status["completed_tasks"]}/20，Best 独立审计 {best}/20，Latest 独立恢复 {latest}/20。运行目录 `{r.HERE}`。', '',
             '本实验检验在 DARG 和谱统计分支固定的条件下，动态类别校准与逐方向证据门控能否减轻 E82.3 的 LLR 漂移并改善 ABIDE 的 ASD 分类。医学 Query 与同几何匿名 Query 用于评估医学文本的额外贡献。', '',
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
        lines.extend([f'### {title}', '', '均值 ± population SD。B0 为 DARG，B1 为 E82.2-G2 PSCE-Medical，B2 为 E82.3-G2 CSPE-Medical。', '',
                      *summary_table(ag['summary'][scope], (*r.GROUPS, *BASELINES)), ''])
    lines.extend(['### 正式 20 个任务', '', *table(['组', '折', 'Best epoch', '终止 epoch', *DISPLAY],
        [[row['group'], row['fold'], row['best_epoch'], row['end_epoch'], *[f'{row[k]:.4f}' for k in METRICS]]
         for row in ag['fold_rows'] if row['group'] in r.GROUPS]), ''])
    auc_orientation = max(abs(a['source_original_prob_auc'] - a['metrics']['prob_auc']) for a in status['historical_references'])
    lines.extend([f'B0 和 B1 历史参考的 Prob-AUC 使用 TD 正类，B2 使用 ASD 正类。历史方向与统一重算 ASD 方向的最大逐折绝对差为 {100*auc_orientation:.8f} pp。三组历史逐折指标同样列入 `results/fold_metrics.csv`。', '',
                  '## 配对差值和患者级硬预测', '', '差值为新模型减对照，单位 pp。每折六指标差值和逐类纠错详见 `results/paired_comparisons.csv`。', ''])
    for scope, title in [('all10', '全部 10 折'), ('without_fold2', '去 Fold2 的 9 折')]:
        lines.extend([f'### {title}', '', *table(['比较', *DISPLAY, 'ACC 胜/平/负', 'BA 胜/平/负'],
            [[name, *[f'{v["mean_pp"][key]:+.4f}' for key in METRICS], '/'.join(map(str, v['w_t_l']['acc'])),
              '/'.join(map(str, v['w_t_l']['hard_auc']))] for name, v in ag['paired'][scope].items()]), '',
            *table(['比较', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总纠错/误伤/净值'],
                   [[name, v['all']['flips'], tally_label(v['asd']), tally_label(v['td']), tally_label(v['all'])] for name, v in ag['corrections'][scope].items()]), ''])
    lines.extend(report_mechanisms(status))
    lines.extend(['## G2 Best 只读干预', '',
                  'T0 为完整医学模型。T1 替换为历史匿名 Query。T2 仅使用原 `w_sp`。T3 将新增残差置零。T4 换回该折 E61 warmup 固定统计，保留门控。T5 使用门控前归一化 coverage，保留 Best 的 EMA 与类别分数。硬翻转和纠错方向为 T0 → 干预，概率变化以 ASD 概率百分点计算。逐折与去 Fold2 结果见审计文件。', '',
                  *table(['干预', '平均绝对 ΔP', '最大绝对 ΔP', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总净纠错', 'Prob-AUC Δ', 'ACC Δ', 'BA Δ', 'SEN-ASD Δ', 'SPE-TD Δ'],
                         [[name, f'{100*v["mean_abs_asd_probability_change"]:.6f}', f'{100*v["max_abs_asd_probability_change"]:.6f}',
                           v['changes']['all']['flips'], tally_label(v['changes']['asd']), tally_label(v['changes']['td']), v['changes']['all']['net'],
                           *[f'{v["mean_delta_pp"][k]:+.4f}' for k in ('prob_auc','acc','hard_auc','sen','spe')]] for name, v in ag['treatments']['all10'].items()]), '',
                  'T1 测量固定 G2 的即时 Query 效应。G2−G1 比较两组完整训练的结果。T4 和 T5 衡量固定 Best 下替换类别校准或方向加权的即时作用。由于 DARG、CAM 和分类头已共同适配，重新训练后的效应仍未识别。', '',
                  '## 独立审计', ''])
    execution = manifest['execution']
    rng = status['rng_audit']
    lines.extend([
        f"{best}/20 个 Best 的逐患者预测与概率位级重现。独立程序复算六指标、最早 ACC 平局和 wait>200，并重算 {rng['ema_updates']} 次 EMA 转移。Best 与当轮更新前状态对应，Latest 与更新后状态对应。"
        f"训练与 DARG 的共同 RNG 三元组 {rng['b0_common']} 条，两组同折三元组 {rng['g1_g2_common']} 条，逐条一致。超出历史 DARG 记录的 {rng['b0_unavailable']} 个训练轮次核对了连续 RNG 链及可用的同折组间重叠。", '',
        f"{latest}/20 个 Latest 在全新对象恢复四个优化器、EMA 和完整 RNG，以记录的上一 EMA 复算终止预测后，按保存的更新后状态独立执行下一次训练、评估和 EMA 更新。"
        f"比较模型、四个优化器、损失、逐患者概率、合法训练观测、EMA 和 RNG 的哈希。保护集合含模型根代码、数据和历史产物，共 {status['protected_files']} 项，SHA-256 核对{'通过' if status['protected_files_sha_match'] else '失败'}。", '',
        f"只用单设备 PPU-ZW810E（98,304 MiB），同时最多 {execution['max_workers']} 个折任务进程，每进程一个 CPU 数值线程。正式耗时 {execution['wall_seconds']:.1f} 秒，记录运行失败事件 {len(execution['failures'])}，整卡采样峰值 {execution['sampled_peak_device_mib']} MiB。", '',
        '## 研究判断与限制', '', f"**{ag['decision']['verdict']}**。", ''])
    for paragraph in brief_evidence(status):
        lines.extend([paragraph, ''])
    for paragraph in calibration_evidence(status):
        lines.extend([paragraph, ''])
    nine = ag['paired']['without_fold2']['G2−B0']['mean_pp']
    lines.extend([f"去 Fold2 后，G2 对 DARG 的 ACC、BA、SEN-ASD 差值分别为 {nine['acc']:+.4f}、{nine['hard_auc']:+.4f}、{nine['sen']:+.4f} pp，"
                  f"患者净纠错 {ag['corrections']['without_fold2']['G2−B0']['all']['net']:+d}。", ''])
    gates = ag['decision']['go_checks']
    lines.extend(['正式训练前冻结的 T1 诊断损失要求总净纠错为负、至少两折净纠错为负、折均 ACC 降低且 BA 不升高。PARTIAL 要求审计全过，至少一组较 DARG 提高 ACC 或 BA 且另一项不下降，并有正的患者净纠错。', '',
                  *table(['条件', '满足'], [[k, '是' if v else '否'] for k, v in gates.items()]), '',
                  '历史 val=test 选模、CV 前 RFE 与 scaling、转导患者图、Fold2 训练重叠及单 seed 限制了证据范围。十折均值与方差仅描述本协议。类别证据采用单方向高斯近似。相对 E82.3 的训练差异同时包含动态校准和逐方向门控，两项各自的重训练因果贡献未识别。CLIP 与 CAM 的语义对齐、描述性 Prompt 分组及外部诊断效度仍待独立验证。', ''])
    return lines


def preflight_summary(status):
    pre = r.read_json(r.RT / 'preflight.json') if (r.RT / 'preflight.json').exists() else {}
    count = len(pre.get('folds', []))
    lines = [f"Preflight 完成 {count}/10 折，状态 {pre.get('status', 'PENDING')}。首选 OSSFS 创建目录返回 errno 5，此后统一在 `/model` 运行。"]
    if pre.get('status') == 'PASS':
        folds = pre['folds']
        auc = [x['llr']['bounded_auc_direction_mean_score'] for x in folds]
        ratios = [g['gradients']['relative_prompt_parameter_norms'] for x in folds for g in x['groups'].values()]
        lines.append(f"逐折核验 E61 模型、三个优化器、4096 关系对、原索引和四类 RNG。E82.3 谱基及 Prompt 缓存精确匹配，G1/G2 与 E82.3 历史同折的 6081 个参数逐元素一致。十折同时关闭残差与辅助监督时，前向、原损失、梯度、下一次历史优化器更新及 RNG 均严格回归 DARG。")
        lines.append(f"合法训练的平滑 LLR 均值分数 AUC 逐折范围 {100*min(auc):.2f}%–{100*max(auc):.2f}%。预检的最大加权辅助/主分类和加权残差/主分类梯度比为 {max(x['weighted_auxiliary'] for x in ratios):.4f} 与 {max(x['weighted_residual'] for x in ratios):.4f}。辅助梯度在每折两组均限于 Prompt 参数，头、CAM、DARG 和 `spectral` 收到零辅助梯度。Prompt 组内置换与两组数学对称性、归一化及 8% 平滑上限均通过。")
        lines.append('十折独立数学核对一阶、二阶矩及样本方差校正。评估中 CAM、四路 GCN 与分类头均只执行一次。Best 保存更新前的 EMA，加载和评估均无自动更新。Latest 恢复后下一次训练、评估、EMA、概率和完整 RNG 与运行对象精确一致。')
    return lines


def save_report(lines):
    with r.atomic_writer(r.RESULTS / 'report.md', 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def append_handoff(status):
    marker = f'## Experiment 82.3.1 · AC-CSPE（{status["status"]}）'
    lines = ['', marker, '', f'实际目录 `{r.HERE}`。正式训练 {status["completed_tasks"]}/20，Best 审计 {status["best_pass"]}/20，Latest 四优化器恢复 {status["latest_pass"]}/20。', '',
             '核对原论文、原作者代码及 E82.3 GitHub 固定分支。E74.4-G2 的 PLNNoPEN、CAM144、E82.3 谱统计和类平衡辅助 CE 沿用原实现。新增 alpha=0.1 的合法训练 EMA 和类别证据逐方向门控。Best 使用更新前 EMA，Latest 保存更新后 EMA。', '']
    for paragraph in preflight_summary(status):
        lines.extend([paragraph, ''])
    if status.get('aggregates'):
        ag = status['aggregates']
        lines.extend(['十折均值 ± population SD，单位%。B0、B1、B2 分别为 DARG、E82.2-G2、E82.3-G2。', '', *summary_table(ag['summary']['all10'], (*r.GROUPS, *BASELINES)), ''])
        for label in ag['paired']['all10']:
            parts = ag['paired']['all10'][label]['mean_pp']
            lines.extend([f"{label} 的六项差值依次为 " + '/'.join(f'{parts[k]:+.4f}' for k in METRICS) + ' pp。', ''])
        for paragraph in [*brief_evidence(status), *calibration_evidence(status)]:
            lines.extend([paragraph, ''])
        counts = status['rng_audit']
        lines.extend([f"共同 DARG RNG 三元组 {counts['b0_common']} 条，同折 G1/G2 三元组 {counts['g1_g2_common']} 条。核查 {counts['ema_updates']} 次 EMA 状态转移。模型根代码、数据及历史保护文件共 {status['protected_files']} 项，SHA-256 核对{'通过' if status['protected_files_sha_match'] else '失败'}。科研判定为 **{ag['decision']['verdict']}**。", ''])
    else:
        lines.extend([f"当前阻断原因见 `{r.HERE / 'results/audit_status.json'}`。{status.get('blocking_reason', '')}", ''])
    lines.extend([f"完整报告见 [Experiment 82.3.1 报告]({r.HERE / 'results/report.md'})。", ''])
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
    sites = (r.runtime().EXP, r.E822, r.E823)
    baseline_tables = {b: csv_map(site / 'results/fold_metrics.csv') for b, site in zip(BASELINES, sites)}
    tasks, references, latest_checks, failures, spectrum_checks = [], [], [], [], []
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
                if group == 'G2':
                    g1, _, _ = r.make_module(state, 'G1', spectrum, cache, text)
                    snapshot = r.torch_load(r.paths('G1', fold)[1])
                    g1.load_state_dict(snapshot['current']['cspe'], strict=True)
                    g1.eval()
                else:
                    g1 = None
                tasks.append(audit_best(group, fold, task, best, latest, log, spectrum, cache, text, pre, g1))
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
