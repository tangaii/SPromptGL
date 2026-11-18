#!/usr/bin/env python3
"""Independent CSPE checkpoint replay, paired analysis and handoff."""
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

from cspe import EPS, FOLD2_ROWS, QUERIES, SPECTRUM_BUFFERS

METRICS = r.METRICS
DISPLAY = ('ACC', 'BA', 'Prob-AUC', 'SEN-ASD', 'SPE-TD', 'ASD-F1')
COMPARISONS = (('G1', 'B0'), ('G1', 'B1'), ('G2', 'B0'), ('G2', 'B1'), ('G2', 'G1'))
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
    weights = coverage / (coverage.sum(-1, keepdim=True) + EPS)
    s0 = (weights[:, 1] * torch.logaddexp(torch.zeros_like(bounded), bounded)).sum(-1)
    s1 = (weights[:, 0] * torch.logaddexp(torch.zeros_like(bounded), -bounded)).sum(-1)
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
               'c_n': coverage[:, 0], 'c_a': coverage[:, 1], 'w_n': weights[:, 0], 'w_a': weights[:, 1],
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


def mechanism(state, best, cache, details, out):
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
            'routing': {key: routing_profile(d[key]) for key in ('w_n', 'w_a', 'w_sem', 'w_sp', 'w')},
            'attention_mean_8x17': d['a'].mean(0).tolist(), 'prompt_cosine_8x8': pair.tolist(),
            'prompt_pairwise_cosine': float(pair[~np.eye(8, dtype=bool)].mean()),
            'null_mean_each_prompt': d['a'][:, :, 16].mean(0).tolist(), 'null': distribution(d['null']),
            'within_group_entropy': {key: distribution(-(d[key] * np.log(d[key].clip(EPS))).sum(1) / math.log(16)) for key in ('w_n', 'w_a')},
            'entropy': distribution(d['entropy']), 'pi_asd': distribution(d['pi'][:, 0]), 'pi_td': distribution(d['pi'][:, 1]),
            'pi_asd_by_class': [distribution(d['pi'][labels == code, 0]) for code in (0, 1)],
            'prompt_ce': finish_ce(d['scores'], labels), 'llr': llr_profile(d['raw_llr'], d['bounded_llr'], labels),
            'w_n_w_a_l1': distribution(np.abs(d['w_n'] - d['w_a']).sum(1)),
            'w_n_w_a_cosine': distribution(cosine(d['w_n'], d['w_a'])),
            'w_sem_w_sp_l1': distribution(np.abs(d['w_sem'] - d['w_sp']).sum(1)),
            'w_sem_w_sp_cosine': distribution(cosine(d['w_sem'], d['w_sp'])),
            'beta': distribution(d['beta']), 'relative_residual': distribution(rel[ids]),
            'true_logit_margin_change': distribution(gain),
            'true_margin_increased_share': float((gain > 0).mean()),
            'true_logit_margin_change_by_class': [distribution(gain[labels == code]) for code in (0, 1)],
        }
    warm_stats = cache['stats']
    current = [plain['hhat'][legal[y_full[legal] == code]].astype(np.float64) for code in (0, 1)]
    mean = np.stack([h.mean(0) for h in current])
    variance = np.stack([h.var(0, ddof=1) for h in current])
    warm_mean = warm_stats['mean'].numpy()
    warm_var = warm_stats['variance'].numpy()
    pooled = warm_stats['pooled'].numpy()
    stable = warm_stats['stable'].numpy()
    dmu = warm_mean[0] - warm_mean[1]
    warm_spectrum = r.spectrum_cache(state['fold'])
    spectrum = warm_spectrum['spectrum']
    hh = ((warm_spectrum['warmup_cam'][legal].double() - spectrum['mu']) @ spectrum['U']) / (spectrum['lambda'] + EPS).sqrt()
    warm_dist = -.5 * (warm_stats['stable'].log() + (hh[:, None] - warm_stats['mean']).square() / warm_stats['stable'])
    raw = (warm_dist[:, 0] - warm_dist[:, 1]).numpy()
    limit = r.CFG['class_statistics']['llr_bound']
    result['warmup_train_llr'] = llr_profile(raw, limit * np.tanh(raw / limit), y_full[legal])
    result['class_separation_and_drift'] = {
        'warm_mean': warm_mean.tolist(), 'warm_variance': warm_var.tolist(), 'stable_variance': stable.tolist(),
        'warm_standardized_mean_gap': (dmu / np.sqrt(pooled + EPS)).tolist(),
        'warm_log_variance_ratio_asd_td': np.log(stable[0] / stable[1]).tolist(),
        'warm_symmetric_gaussian_kl': (.25 * (stable[0]/stable[1] + stable[1]/stable[0] - 2 + dmu**2 * (1/stable[0] + 1/stable[1]))).tolist(),
        'best_current_mean': mean.tolist(), 'best_current_variance': variance.tolist(),
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


def interventions(state, module, text, best_out, comparison_module, y):
    full = probability(best_out, state)
    pred = full.argmax(-1)
    d = best_out['details']
    idx = state['test_np']
    saved = module.q0.clone()
    initial = r.sha(r.cpu_state(module))
    rng = r.rng_sha()
    response = []
    for code, method in (('T0', 'full'), ('T1', 'full'), ('T2', 'semantic_off'), ('T3', 'residual_off')):
        try:
            if code == 'T1':
                module.q0.copy_(text['q_anonymous'].to(module.q0.device))
            out = best_out if code == 'T0' else independent_forward(state, module, method)
            p = probability(out, state)
            change = np.abs(p[:, 0] - full[:, 0])
            if code == 'T2':
                assert torch.equal(out['details']['w'], d['w_sp'])
            if code == 'T3':
                assert torch.count_nonzero(out['details']['delta']) == 0
            result = {'intervention': code, 'metrics': independent_metrics(p, y),
                      'mean_abs_asd_probability_change': float(change.astype(np.float64).mean()),
                      'max_abs_asd_probability_change': float(change.max()), 'n': int(len(y)),
                      'changes': changes(pred, p.argmax(-1), y),
                      'flipped_rows_1based': (idx[p.argmax(-1) != pred] + 1).tolist(),
                      'probability_asd': p[:, 0].tolist(), 'prediction': p.argmax(-1).tolist()}
            if code == 'T1':
                assert torch.equal(out['details']['w_sp'], d['w_sp'])
                result['routing_change_same_best_cam'] = routing_difference(d, out['details'], idx)
            response.append(result)
        finally:
            if code == 'T1':
                module.q0.copy_(saved)
    with torch.no_grad():
        _, other = comparison_module(d['z'].detach())
        difference = routing_difference(d, other, idx)
    assert r.sha(r.cpu_state(module)) == initial and r.rng_sha() == rng
    return response, difference


def verify_selection(state, task, best, latest, log):
    records = [r.json.loads(line) for line in log.read_bytes().splitlines()]
    assert r.sha_file(log) == latest['log_sha'] == task['log_sha'] and log.stat().st_size == latest['log_bytes']
    stop = task['epochs_executed']
    assert [row['epoch'] for row in records] == list(range(r.START + 1, stop + 1))
    historical = {row['epoch']: row for row in r.historical_task(state['fold'])['rng_trace']}
    y = state['labels_np'][state['test_np']]
    prior = task['identity']['warmup_rng_sha']
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
        assert row['rng_before'] == prior and row['rng_after_train'] == row['rng_after_eval']
        prior = row['rng_after_eval']
        if epoch in historical:
            assert {key: row[key] for key in r.RNG_KEYS} == historical[epoch]
            common += 1
    assert selected == task['best_epoch'] == best['best_epoch'] == latest['best_epoch']
    assert stop == latest['next_epoch'] and wait == latest['wait'] and maximum == latest['best_correct']
    assert wait > r.CFG['early_stop_wait_gt'] or stop == r.LIMIT
    chosen = records[selected - r.START - 1]
    assert chosen['correct'] == (best['prediction'] == y).sum() and maximum / len(y) == best['metrics']['acc']
    assert chosen['prediction_bits'] == np.packbits(best['prediction']).tobytes().hex()
    assert best['rng_sha'] == chosen['rng_after_eval'] and prior == latest['rng_sha'] == task['terminal_rng_sha']
    measured = [row['total_gradient_norms'] for row in records if 'total_gradient_norms' in row]
    assert measured and 'total_gradient_norms' in records[0]
    probe = {key: distribution([row[key] for row in measured]) for key in measured[0]}
    return records, {'common_b0_rng_epochs': common, 'epochs': len(records),
                     'gradient_samples': len(measured), 'training_total_gradient_norms': probe,
                     'prompt_ce_first_best_last': [records[0]['losses']['prompt'], chosen['losses']['prompt'], records[-1]['losses']['prompt']]}


def reference_fold(state, identity, label, csv_lookup):
    fold = state['fold']
    y = state['labels_np'][state['test_np']]
    if label == 'B0':
        ckpt = r.torch_load(r.runtime().EXP / '.runtime/checkpoints' / f'{r.BASE_GROUP}_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('test_probability', 'test_prediction', 'best_epoch_1based'))
        source_row = csv_lookup[(r.BASE_GROUP, fold)]
    else:
        ckpt = r.torch_load(r.E822 / 'runtime/checkpoints' / f'G2_f{fold:02d}.best.pt')
        p, pred, epoch = (ckpt[k] for k in ('probability', 'prediction', 'best_epoch'))
        source_row = csv_lookup[('G2', fold)]
    assert ckpt['identity'] == identity and len(p) == len(y) and np.array_equal(pred, p.argmax(-1))
    metric = independent_metrics(p, y)
    published = {**metric, 'prob_auc': float(roc_auc_score(y, p[:, 1]))}
    same_metrics(published, ckpt['metrics'])
    same_metrics(published, {k: float(source_row[k]) / (1 if label == 'B0' else 100) for k in METRICS})
    return {'group': label, 'fold': fold, 'best_epoch': epoch, 'metrics': metric, 'prediction': pred.tolist(),
            'probability': p.tolist(), 'source_original_prob_auc_td': published['prob_auc'],
            'source_best_checkpoint_original_metrics_checked': True}


def read_task(group, fold, pre):
    assert r.task_complete(group, fold)
    task_file, bp, lp, log = r.paths(group, fold)
    task, best, latest = r.read_json(task_file), r.torch_load(bp), r.torch_load(lp)
    shared = ('group', 'fold', 'seed', 'identity', 'implementation', 'initial_parameter_sha',
              'stats_sha', 'spectrum_sha', 'query_sha', 'legal_train_sha')
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
    assert ckpt['stats_sha'] == r.sha(cache['stats']) == cache['stats_sha']
    assert ckpt['spectrum_sha'] == r.sha(spectrum['spectrum']) == spectrum['spectra_sha']
    r.load_bundle(state, module, opt, ckpt['current'])
    base = r.base_runtime()
    base.restore_rng(ckpt['rng'])
    assert r.rng_sha() == ckpt['rng_sha'] and r.component_digests(r.bundle(state, module, opt)) == ckpt['digests']
    assert torch.equal(module.q0.cpu(), text[QUERIES[group]])
    for name, key in SPECTRUM_BUFFERS.items():
        assert torch.equal(getattr(module, name).cpu(), spectrum['spectrum'][key].float())
    assert torch.equal(module.class_mean.cpu(), cache['stats']['mean'].float())
    assert torch.equal(module.class_var.cpu(), cache['stats']['stable'].float())
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in module.modules())
    assert all(not b.requires_grad for b in module.buffers())
    return state, module, opt, identity


def audit_best(group, fold, task, best, latest, log, spectrum, cache, text, pre, other_module=None):
    state, module, opt, _ = restored_object(group, fold, best, spectrum, cache, text)
    assert np.array_equal(best['test_indices'], state['test_np'])
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
    logs, selection_trace = verify_selection(state, task, best, latest, log)
    diagnose = mechanism(state, best, cache, d, out)
    original_y = state['y'][state['test_np']].clone()
    state['y'][state['test_np']] = 1 - original_y
    try:
        changed = independent_forward(state, module)
        assert torch.equal(changed['logp'], out['logp'])
        assert all(torch.equal(changed['details'][key], d[key]) for key in ('raw_llr', 'scores', 'pi', 'w_sem', 'delta'))
    finally:
        state['y'][state['test_np']] = original_y
    assert r.rng_sha() == best['rng_sha']
    readonly, comparison = interventions(state, module, text, out, other_module, y) if group == 'G2' else ([], None)
    assert r.component_digests(r.bundle(state, module, opt)) == best['digests']
    return {'status': 'PASS', 'group': group, 'fold': fold, 'metrics': metrics, 'best_epoch': best['best_epoch'],
            'epochs_executed': task['epochs_executed'], 'initial_parameter_sha': task['initial_parameter_sha'],
            'best_file_sha': task['best_sha'], 'probability_max_abs_replay_error': 0., 'optimizers': optimizers,
            'test_rows_1based': (state['test_np'] + 1).tolist(), 'labels': y.tolist(), 'probability': p.tolist(),
            'prediction': pred.tolist(), 'formula_max_abs_errors': formulas, 'best_gradients': grad,
            'historical_preflight_gradients': pre['folds'][fold - 1]['groups'][group]['gradients'],
            'selection_trace': selection_trace, 'mechanism': diagnose,
            'rng_trace': [{key: row[key] for key in r.RNG_KEYS} for row in logs],
            'test_label_swap_bitwise': True, 'four_treatments': readonly,
            'trained_g1_vs_g2_same_g2_cam': comparison, 'trainable_parameters': sum(p.numel() for p in module.parameters()),
            'peak_gpu_bytes': task['peak_gpu_bytes'], 'resumes': task['resumes']}


def audit_latest(group, fold, task, latest, spectrum, cache, text):
    # Build fresh model and optimizer objects even when the independent Best replay has passed.
    state, module, opt, _ = restored_object(group, fold, latest, spectrum, cache, text)
    before = r.rng_sha()
    p = probability(independent_forward(state, module), state)
    assert np.array_equal(p, latest['probability']) and r.sha(p) == task['terminal_probability_sha']
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
    p = probability(independent_forward(state, module), state)
    checkpoint = r.checkpoint_state(state, module, opt)
    expected = task['next_step_witness']
    actual = {'after_train_rng': after_train, 'after_eval_rng': checkpoint['rng_sha'],
              'components_after_update': checkpoint['digests'], 'probability_sha': r.sha(p),
              'losses': dict(zip(r.LOSS_NAMES, torch.stack([out[k].detach() for k in r.LOSS_NAMES]).cpu().tolist()))}
    for key, value in actual.items():
        assert value == expected[key], (group, fold, key)
    return {'status': 'PASS', 'latest_file_sha': task['latest_sha'], 'epoch': latest['next_epoch'],
            'restored_components': latest['digests'], 'rng': before, 'terminal_probability_exact': True,
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
    for treatment in ('T1', 'T2'):
        v = treatments[treatment]
        go[treatment + '_removal_diagnostic_loss'] = (v['changes']['all']['net'] < 0 and v['mean_delta_pp']['acc'] < 0 and
            v['mean_delta_pp']['hard_auc'] <= 1e-10 and
            v['folds_with_negative_net'] >= cfg['min_semantic_removal_net_loss_folds_per_intervention'])
    partial = {g: ((delta(g, 'B0', 'acc') > 0 or delta(g, 'B0', 'hard_auc') > 0) and
                   delta(g, 'B0', 'acc') >= -1e-10 and delta(g, 'B0', 'hard_auc') >= -1e-10 and net(g) > 0) for g in r.GROUPS}
    verdict = 'GO' if all(go.values()) else 'PARTIAL' if audits_pass and any(partial.values()) else 'NO-GO'
    return {'verdict': verdict, 'go_checks': go, 'partial_by_group': partial,
            'clinical_semantic_contribution_established_under_this_protocol': all(go.values())}


def aggregate(tasks, references, audits_pass):
    data = {(a['group'], a['fold']): a for a in [*tasks, *references]}
    by = lambda g, f: data[(g, f)]
    groups = (*r.GROUPS, 'B0', 'B1')
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
        for treatment in ('T0', 'T1', 'T2', 'T3'):
            selected = [next(row for row in by('G2', f)['four_treatments'] if row['intervention'] == treatment) for f in folds]
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
    '原论文的 PLN 先用 CAM 做患者内跨模态交互，再通过 PEN 的 SPGS 以文本相似度加权局部 token，并通过 RES 融合局部与全局表示。原作者 `network.py` 使用冻结 CLIP RN50，将八条文本投影到局部特征空间，并在分类前作局部嵌入。历史 E74.4-G2 实际加载 vendor `PLNNoPEN`，由 InputLayer、Encoder 和 FeedForward 产生 CAM144 后直接 flatten 送入原 OutputLayer。E82.3 继续走这条历史路径。 [MICCAI 2025 论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者 network.py](https://github.com/wanxixi11/SPromptGL/blob/ba3b9446fa451fde6864d6555bdf324de0c4b64a/network.py)',
    'E74.4-G2 继承 GraphLearn 正边支持、统一图归一化、四个 GCN 和仅正邻接增量的 Positive-DCR。DARG 的 320 维融合表示只作为历史 CAM 输入。CSPE 接在 `Outputlayer` 的第一个参数 CAM144 上，沿用 LayerNorm(144) → Linear(144,36) → GELU → Dropout → Linear(36,2) → log-softmax。历史数据预处理读取 ABIDE `DX_GROUP`，训练入口减 1 后得到 ASD=0、TD=1。',
    '已逐文件核对用户仓库 `upload/experiment82_2_psce` 分支与本地的 `psce.py`、`run.py`、`audit.py`、`config.json`、`results/report.md`。原 E82.2 的八个 Query 对 16 个谱 token 加 null 做注意力，统一 coverage 生成 `w_sem`，谱统计 MLP 生成 `w_sp`，在残差重建前按 `w=beta*w_sem+(1-beta)*w_sp` 融合。 [E82.2 固定提交](https://github.com/tangaii/SPromptGL/tree/9f104bda39cc9c4deaf90aa25cb09fd2c4a8d302/experiment/experiment82_2_psce)',
    'E82.2 每轮复制当前 OutputLayer 为冻结 eval 教师，对合法训练 CAM144 计算原始输入与 16 个受限方向遮蔽的真实类 logit margin 差。只对教师正确、margin 为正且正贡献大于阈值的行监督。正贡献归一化后 detach，KL 梯度经 `w_sem` 进入 Prompt 参数及其输入 CAM 特征，教师与目标端无梯度。E82.2-G2 相对 DARG 的 ACC/BA/Prob-AUC 分别下降 0.1149/0.1485/0.3350 pp，且净纠错 −1。',
    'E82.2.1 加入增强、抑制和不动的双向动作及 harm 约束，覆盖原分类错误训练患者。G2 教师动作分布接近均匀，动作准确率约 43%，移除双向纠正为零硬翻转。G2 的 Prob-AUC 上升 0.1212 pp，但相对 DARG 的 ASD 净纠错 +1 和 TD 净纠错 −5 合计 −4，ACC/BA 下降。E82.2.2 的双尺度一致性提升了训练目标对齐，且 G2 Prob-AUC 上升 0.3441 pp。其平均残差降到 0.3602%，低于 E82.2 的 3.5888%，移除语义或残差均零翻转，ACC/BA/SEN 和净纠错仍下降。两份报告支持训练局部 margin 或排序改善与最终硬决策之间存在落差，无法分离每条反事实机制的独立效应。 [E82.2.1 完整报告](../experiment82_2_1_bscp/results/report.md) · [E82.2.2 完整报告](../experiment82_2_2_rce_psce/results/report.md)',
    'E82.3 用两组四条 Prompt 的连续 coverage 生成 Normal-like 的 `w_N` 和 Abnormal-like 的 `w_A`。固定训练类条件高斯统计产生 ASD 对 TD 的单方向 LLR，`8*tanh(r/8)` 平滑后由两组权重分别汇聚正负证据。两个分数同时参与 softmax，得到 `pi_ASD*w_A+pi_TD*w_N`。八条原文本保留 `Nromal brain volume` 拼写。Normal-like 与 Abnormal-like 代表操作性假设。这些描述的 ASD/TD 临床标志物身份、潜在谱方向的解剖含义以及类别统计的因果含义均待验证。',
    '每折类统计在原始 E61 epoch 10 的合法训练 CAM 上以 float64 拟合一次，类方差采用 n−1，按类人数加权汇总后以 0.9 自身方差、0.1 池化方差和 max(1e−4, 0.01 池化方差) 稳定化。前向以冻结的 float32 buffer 运算。辅助 CE 按 ASD、TD 各自求平均再等权平均，使用 `z.detach()` 的合法训练行，只更新 direction、evidence、norm、key 和 null。主分类梯度仍贯穿 DARG、CAM、原头和全部 CSPE 选择参数。总损失为 `L_DARG + 0.001*L_res + 0.01*L_Prompt`。',
    '保留 E82.2 的 Original O 谱基、中心、特征值、8 个全局 PCA 方向和 8 个正交类别协方差差异方向。原 6081 个可训练参数、3–32–1 谱统计 MLP、文本投影和归一化、8×17 注意力、温度 0.5、null、detach beta、谱残差重建和 8% 平滑上限均沿用 E82.2。原 DARG 分类与图、关系、对抗、保留损失及三个优化器按历史顺序更新，最后更新第四个 AdamW，lr=0.0005、weight decay=0.01。G1/G2 每折共享 822000+fold 的隔离初始化，唯一实验差异是固定 Query。',
    '历史入口在恢复 E61 后按 E61-E2 设置恒等校准、reliability=1 和冻结零 reliability 门控，再应用 E74.4-G2 的确定性自适应门控启动值。Fold2 按 E69.3 历史方法添加 377、407，并按既定局部 RNG 重新生成该折有效的 4096 个关系对。E82.3 由真实 E74.4-G2 入口恢复，所以同时保有这一历史例外和原 warmup 索引。CSPE 类统计和辅助监督取原无重叠训练行，377、407 的新增残差严格为零。'
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
            f"Best 的 Prompt 类平衡 CE 在合法训练为 {avg(lambda a: a['mechanism']['legal_train']['prompt_ce']['balanced_ce']):.4f}，在测试为 {avg(lambda a: a['mechanism']['test']['prompt_ce']['balanced_ce']):.4f}。测试 Prompt 分数 AUC {100*avg(lambda a: a['mechanism']['test']['prompt_ce']['scores_roc_auc']):.2f}%，真类分数优势均值 {avg(lambda a: a['mechanism']['test']['prompt_ce']['true_class_margin']['mean']):+.4f}，真类分数严格占优比例 {100*avg(lambda a: a['mechanism']['test']['prompt_ce']['true_class_win_share']):.2f}%。`pi_ASD`/`pi_TD` 测试均值 {avg(lambda a: a['mechanism']['test']['pi_asd']['mean']):.4f}/{avg(lambda a: a['mechanism']['test']['pi_td']['mean']):.4f}。", '',
            f"测试 Normal-like/Abnormal-like coverage 均值 {avg(lambda a: a['mechanism']['test']['coverage_n']['mean']):.4f}/{avg(lambda a: a['mechanism']['test']['coverage_a']['mean']):.4f}。两组 `w_N`/`w_A` 的 L1 差 {avg(lambda a: a['mechanism']['test']['w_n_w_a_l1']['mean']):.4f}，八条 Prompt 两两注意力余弦均值 {avg(lambda a: a['mechanism']['test']['prompt_pairwise_cosine']):.4f}，null 均值 {avg(lambda a: a['mechanism']['test']['null']['mean']):.4f}。`w_sem` 的归一化熵 {avg(lambda a: a['mechanism']['test']['entropy']['mean']):.4f}，与 `w_sp` 的余弦相似度 {avg(lambda a: a['mechanism']['test']['w_sem_w_sp_cosine']['mean']):.4f}。", '',
            f"测试 `w_sem` 的折内 Top1 众数占比均值 {100*avg(lambda a: a['mechanism']['test']['routing']['w_sem']['top1_mode_share']):.2f}%，全测试行共享同一 Top1 的折数 {sum(a['mechanism']['test']['routing']['w_sem']['top1_fixed'] for a in subset)}/10。患者方向权重相对折内均值的平均 L1 偏差 {avg(lambda a: a['mechanism']['test']['routing']['w_sem']['patient_variation_l1']):.4f}，平均有效方向数 {avg(lambda a: a['mechanism']['test']['routing']['w_sem']['effective_direction_count']['mean']):.2f}/16。", '',
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
    g2 = [a for a in status['tasks'] if a['group'] == 'G2']
    for explanation, rows in [
        ('同一个 G2 Best 和 CAM 下，只替换固定 Query', [a['four_treatments'][1]['routing_change_same_best_cam'] for a in g2]),
        ('在同一 G2 CAM 上分别运行 G1 和 G2 的已训练选择器', [a['trained_g1_vs_g2_same_g2_cam'] for a in g2]),
    ]:
        lines.extend([f"{explanation}时，注意力平均绝对差 {np.mean([v['attention_mean_abs'] for v in rows]):.6f}，`w_N`/`w_A`/`w_sem` 的平均 L1 差 "
                      f"{'/'.join(f'{np.mean([v[k + "_l1"] for v in rows]):.4f}' for k in ('w_n','w_a','w_sem'))}，"
                      f"`w_sem` Top1 一致率 {100*np.mean([v['w_sem_top1_agreement'] for v in rows]):.2f}%。", ''])
    lines.extend(['各折的 LLR 分位数及各类分布、8×17 注意力、逐方向分布、类条件漂移、逐患者硬预测和梯度详见 `results/audit_status.json`。方向编号与谱基逐折拟合，跨折同编号的均值只作描述。', ''])
    return lines


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
    g2b1 = paired['G2−B1']['mean_pp']
    core = (f"G1/G2 相对 DARG 的 ACC 分别变化 {g1['acc']:+.4f}/{g2['acc']:+.4f} pp，BA 分别变化 {g1['hard_auc']:+.4f}/{g2['hard_auc']:+.4f} pp。"
            f"G2 相对 G1 的 ACC 与 BA 差为 {g21['acc']:+.4f}/{g21['hard_auc']:+.4f} pp，相对 E82.2-G2 为 {g2b1['acc']:+.4f}/{g2b1['hard_auc']:+.4f} pp。"
            f"G2 相对 DARG 的患者净纠错为 {corr['G2−B0']['all']['net']:+d}。")
    if verdict == 'GO':
        boundary = '按冻结条件，CSPE 可留作第二创新模块候选。复核需要独立选模和外部验证，才能判断能否泛化。'
    elif verdict == 'PARTIAL':
        boundary = '至少一组满足预定的相对 DARG 硬分类改善条件。类别监督与权重结构同时变化，收益归于这组 CSPE 变更，医学文本的独立价值需结合干预证据判断。'
    else:
        boundary = '按冻结条件停止本候选的推进。本次证据未满足类别监督改善硬分类并支持医学贡献的要求。'
    return [core, semantics, boundary]


def make_report(status, manifest):
    code = status['status']
    best, latest = status['best_pass'], status['latest_pass']
    lines = ['# Experiment 82.3 · CSPE', '', '**Class-Supervised Prompt Evidence Learning（类别监督的疾病 Prompt 证据学习）**', '',
             f'状态 **{code}**。正式训练 {status["completed_tasks"]}/20，Best 独立审计 {best}/20，Latest 独立恢复 {latest}/20。运行目录 `{r.HERE}`。', '',
             '实验检验在同一 DARG、谱基和残差机制下，真实类别监督与分组证据选择能否比 E82.2 的反事实监督改善 ABIDE 的 ASD 分类，同时用同几何匿名 Query 检验医学文本贡献。', '',
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
        lines.extend([f'### {title}', '', '均值 ± population SD。B0 为 DARG，B1 为 E82.2-G2 PSCE-Medical。', '',
                      *summary_table(ag['summary'][scope], (*r.GROUPS, 'B0', 'B1')), ''])
    lines.extend(['### 正式 20 个任务', '', *table(['组', '折', 'Best epoch', '终止 epoch', *DISPLAY],
        [[row['group'], row['fold'], row['best_epoch'], row['end_epoch'], *[f'{row[k]:.4f}' for k in METRICS]]
         for row in ag['fold_rows'] if row['group'] in r.GROUPS]), ''])
    auc_orientation = max(abs(a['source_original_prob_auc_td'] - a['metrics']['prob_auc']) for a in status['historical_references'])
    lines.extend([f'原 CSV 的 TD 正类 Prob-AUC 与重算 ASD 正类 Prob-AUC 的最大逐折绝对差为 {100*auc_orientation:.8f} pp。两组历史逐折指标同样列入 `results/fold_metrics.csv`。', '',
                  '## 配对差值和患者级硬预测', '', '差值为新模型减对照，单位 pp。每折六指标差值和逐类纠错详见 `results/paired_comparisons.csv`。', ''])
    for scope, title in [('all10', '全部 10 折'), ('without_fold2', '去 Fold2 的 9 折')]:
        lines.extend([f'### {title}', '', *table(['比较', *DISPLAY, 'ACC 胜/平/负', 'BA 胜/平/负'],
            [[name, *[f'{v["mean_pp"][key]:+.4f}' for key in METRICS], '/'.join(map(str, v['w_t_l']['acc'])),
              '/'.join(map(str, v['w_t_l']['hard_auc']))] for name, v in ag['paired'][scope].items()]), '',
            *table(['比较', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总纠错/误伤/净值'],
                   [[name, v['all']['flips'], tally_label(v['asd']), tally_label(v['td']), tally_label(v['all'])] for name, v in ag['corrections'][scope].items()]), ''])
    lines.extend(report_mechanisms(status))
    lines.extend(['## G2 Best 只读干预', '',
                  'T0 为完整医学模型。T1 只换历史匿名 Query。T2 用原 `w_sp` 作为唯一方向权重。T3 将新增残差置零。硬翻转和纠错方向为 T0 → 干预，概率变化用 ASD 概率，单位 pp。下表汇总全部 10 折，逐折与去 Fold2 结果在审计文件。', '',
                  *table(['干预', '平均绝对 ΔP', '最大绝对 ΔP', '硬翻转', 'ASD 纠错/误伤/净值', 'TD 纠错/误伤/净值', '总净纠错', 'Prob-AUC Δ', 'ACC Δ', 'BA Δ'],
                         [[name, f'{100*v["mean_abs_asd_probability_change"]:.6f}', f'{100*v["max_abs_asd_probability_change"]:.6f}',
                           v['changes']['all']['flips'], tally_label(v['changes']['asd']), tally_label(v['changes']['td']), v['changes']['all']['net'],
                           *[f'{v["mean_delta_pp"][k]:+.4f}' for k in ('prob_auc','acc','hard_auc')]] for name, v in ag['treatments']['all10'].items()]), '',
                  'T1 测量固定 G2 上的即时 Query 效应，G2−G1 比较两组的完整训练结果。T3 使用已共同适配的 DARG/CAM 与原分类头。跨患者、跨折的诊断贡献依赖方向选择差异、净纠错及干预方向的联合证据。', '',
                  '## 独立审计', ''])
    execution = manifest['execution']
    rng = status['rng_audit']
    lines.extend([
        f"逐患者 Best 概率和硬预测位级一致，六指标由 sklearn 重算。独立审计读取每轮硬预测复核 ACC、最早平局和 wait>200。训练与 DARG 的共同 RNG 三元组 {rng['b0_common']} 条，两组同折共同 RNG 三元组 {rng['g1_g2_common']} 条，逐条一致。其余超出历史 DARG 日志的 {rng['b0_unavailable']} 个训练轮次核对了不间断 RNG 链及可用的同折组间重叠。", '',
        f"Latest 均使用全新对象恢复原三个优化器及第四优化器，复算终止概率和完整 RNG。随后分别独立执行下一次训练加评估，并与训练终止对象的审计见证比较。比较涵盖模型、四个优化器、损失、逐患者概率和完整 RNG 的哈希。正式权重和日志留在注册终止轮。保护集合共 {status['protected_files']} 项，SHA-256 前后核对通过。", '',
        f"只用单设备 PPU-ZW810E（98,304 MiB），同时最多 {execution['max_workers']} 个折任务进程，每进程一个 CPU 数值线程。正式耗时 {execution['wall_seconds']:.1f} 秒，记录运行失败事件 {len(execution['failures'])}，整卡采样峰值 {execution['sampled_peak_device_mib']} MiB。", '',
        '## 研究判断与限制', '', f"**{ag['decision']['verdict']}**。", ''])
    for paragraph in brief_evidence(status):
        lines.extend([paragraph, ''])
    gates = ag['decision']['go_checks']
    lines.extend(['训练前冻结的 GO 条件逐项见下表。至少一组达到相对 DARG 的预定硬分类和净纠错门槛时，未满足 GO 的结果可以归为 PARTIAL。', '',
                  *table(['条件', '满足'], [[k, '是' if v else '否'] for k, v in gates.items()]), '',
                  '历史 val=test 选模、CV 前 RFE 与 scaling、转导患者图和 Fold2 训练重叠限制了结论。去 Fold2 仍保留其余历史偏倚。单 seed 及反复使用同一 ABIDE 划分的科研探索使折间波动和均值差只能描述本协议。新增统计是一维高斯近似，依赖 E61 特征，在端到端训练后存在分布漂移。两组同时改变 Prompt 监督和类别路由，相对 B1 的效果无法单独归给其中一个。CLIP 与 CAM 的语义对齐和医学 Prompt 的临床特异性仍待独立验证。', '',
                  '后续若另行立项，可以用训练折内完成预处理和调参、独立选择 Best、外部锁定测试以及多 seed 的设计复核。也可在外部验证中检验稳定谱统计和更明确的文本—影像对齐。本次预注册不启动 E82.3.1 或参数搜索。', ''])
    return lines


def preflight_summary(status):
    pre = r.read_json(r.RT / 'preflight.json') if (r.RT / 'preflight.json').exists() else {}
    count = len(pre.get('folds', []))
    lines = [f"Preflight 完成 {count}/10 折，状态 {pre.get('status', 'PENDING')}。首选 OSSFS 创建目录返回 errno 5，此后统一在 `/model` 运行。"]
    if pre.get('status') == 'PASS':
        folds = pre['folds']
        auc = [x['llr']['bounded_auc_direction_mean_score'] for x in folds]
        ratios = [g['gradients']['relative_prompt_parameter_norms'] for x in folds for g in x['groups'].values()]
        lines.append(f"逐折核验 E61 模型、三个优化器、4096 关系对、原索引和四类 RNG。E82.2 谱基及 Prompt 缓存身份精确匹配，G1/G2 每折 6081 个参数逐元素同初始化。十折在同时关闭残差与辅助监督时，前向、原损失、梯度、下一次历史优化器更新及 RNG 均严格回归 DARG。")
        lines.append(f"合法训练的平滑 LLR 均值分数 AUC 逐折范围 {100*min(auc):.2f}%–{100*max(auc):.2f}%。预检的最大加权辅助/主分类和加权残差/主分类梯度比为 {max(x['weighted_auxiliary'] for x in ratios):.4f} 与 {max(x['weighted_residual'] for x in ratios):.4f}。辅助梯度在每折两组均限于 Prompt 参数，头、CAM、DARG 和 `spectral` 收到零辅助梯度。Prompt 组内置换与两组数学对称性、归一化及 8% 平滑上限均通过。")
    return lines


def save_report(lines):
    with r.atomic_writer(r.RESULTS / 'report.md', 'w') as fp:
        fp.write('\n'.join(lines) + '\n')


def append_handoff(status):
    marker = f'## Experiment 82.3 · CSPE（{status["status"]}）'
    lines = ['', marker, '', f'实际目录 `{r.HERE}`。正式训练 {status["completed_tasks"]}/20，Best 审计 {status["best_pass"]}/20，Latest 四优化器恢复 {status["latest_pass"]}/20。', '',
             '核对原论文、原作者代码及用户 E82.2 固定分支，历史基座继续使用 E74.4-G2 与 PLNNoPEN。用冻结的合法训练统计和 ASD/TD 类平衡辅助 CE 替换反事实监督。', '']
    for paragraph in preflight_summary(status):
        lines.extend([paragraph, ''])
    if status.get('aggregates'):
        ag = status['aggregates']
        lines.extend(['十折均值 ± population SD，单位%。', '', *summary_table(ag['summary']['all10'], (*r.GROUPS, 'B0', 'B1')), ''])
        for label in ('G1−B0', 'G1−B1', 'G2−B0', 'G2−B1', 'G2−G1'):
            parts = ag['paired']['all10'][label]['mean_pp']
            lines.extend([f"{label} 的六项差值依次为 " + '/'.join(f'{parts[k]:+.4f}' for k in METRICS) + ' pp。', ''])
        for paragraph in brief_evidence(status):
            lines.extend([paragraph, ''])
        counts = status['rng_audit']
        lines.extend([f"共同 DARG RNG 三元组 {counts['b0_common']} 条，同折 G1/G2 三元组 {counts['g1_g2_common']} 条。完整审计和科研判定为 **{ag['decision']['verdict']}**。去 Fold2、干预和漂移见报告。", ''])
    else:
        lines.extend([f"当前阻断原因见 `{r.HERE / 'results/audit_status.json'}`。{status.get('blocking_reason', '')}", ''])
    lines.extend([f"完整报告 `{r.HERE / 'results/report.md'}`。本次不启动后续变体或参数搜索。", ''])
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
    b0, b1 = csv_map(r.runtime().EXP / 'results/fold_metrics.csv'), csv_map(r.E822 / 'results/fold_metrics.csv')
    tasks, references, latest_checks, failures, spectrum_checks = [], [], [], [], []
    start = time.monotonic()
    for fold in r.FOLDS:
        spectrum, cache = r.spectrum_cache(fold), r.torch_load(r.cache_path(fold))
        assert r.sha_file(r.cache_path(fold)) == pre['class_cache_files'][r.cache_path(fold).name]
        state, warm, identity = r.prepare(fold)
        try:
            spectrum_checks.append({'fold': fold, **independent_stats(state, warm, identity, spectrum, cache)})
            references.extend([reference_fold(state, identity, 'B0', b0), reference_fold(state, identity, 'B1', b1)])
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
                latest_checks.append({'fold': fold, 'group': group, **audit_latest(group, fold, task, latest, spectrum, cache, text)})
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
                            'b0_unavailable': sum(len(a['rng_trace']) for a in tasks)-common},
              'implementation': code, 'audit_wall_seconds': time.monotonic() - start}
    if len(tasks) == len(references) == r.CFG['expected_tasks']:
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
