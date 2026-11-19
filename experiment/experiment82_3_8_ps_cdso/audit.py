#!/usr/bin/env python3
"""Independent PS-CDSO equations, patient pairing, and checkpoint audit."""
from __future__ import annotations

import run as r  # Physical GPU0 is selected before torch is imported.

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F

import history as h
from ps_cdso import EMA_KEYS, FOLD2_ROWS, INTERVENTIONS, RISK_KEYS

CFG = r.CFG
TOL = CFG['preflight']
AUC_KEYS = r.METRICS
BASE_DIR = r.RT/'audit'
EPS = 1e-8


def metrics(prob, targets):
    pred = np.asarray(prob).argmax(1)
    p = np.asarray(prob)[:, 0]
    values = (accuracy_score(targets, pred), balanced_accuracy_score(targets, pred),
              roc_auc_score(targets == 0, p), recall_score(targets, pred, pos_label=0),
              recall_score(targets, pred, pos_label=1), f1_score(targets, pred, pos_label=0, zero_division=0))
    return dict(zip(AUC_KEYS, map(float, values)))


def same_metrics(a, b):
    assert all(abs(a[key]-b[key]) <= 1e-12 for key in AUC_KEYS), (a, b)


def rows_legal(state):
    train = np.asarray(state['train_np'])
    test = np.asarray(state['test_np'])
    excluded = list(FOLD2_ROWS) if state['fold'] == 2 else []
    assert np.intersect1d(train, test).tolist() == excluded
    ids = np.setdiff1d(train, test, assume_unique=True)
    assert np.array_equal(ids, h.legal_train(state)) and np.array_equal(state['legal_rows'].cpu(), ids)
    return ids


def array(x):
    return x.detach().cpu().numpy()


def compare(a, b):
    torch.testing.assert_close(a, b, atol=TOL['audit_feature_atol'], rtol=TOL['audit_feature_rtol'])


def clean_basis(module):
    origin, w = array(module.U0), array(module.offset)
    tang = (np.eye(len(origin)) - origin@origin.T) @ w
    tangent = tang / np.hypot(1, np.linalg.norm(tang)/CFG['stiefel']['offset_frobenius_radius'])
    q, rr = np.linalg.qr(origin+tangent, mode='reduced')
    return q*np.sign(np.diag(rr)), float(np.linalg.cond(np.eye(16)+tangent.T@tangent))


def basis_diagnostics(module):
    measured = array(module.basis())
    expected, condition = clean_basis(module)
    assert measured.shape == (144, 16) and measured.dtype == np.float64
    np.testing.assert_allclose(measured, expected, atol=TOL['audit_fp64_atol'], rtol=0)
    initial = array(module.U0)
    angles = np.degrees(np.arccos(np.clip(np.linalg.svd(initial.T@measured, compute_uv=False), 0, 1)))
    error = float(np.abs(measured.T@measured-np.eye(16)).max())
    rank = int(np.linalg.matrix_rank(measured))
    movement = float(np.linalg.norm(measured@measured.T-initial@initial.T))
    assert error <= TOL['gram_max_abs'] and rank == 16
    assert condition <= TOL['max_cholesky_gram_condition']
    return dict(rank=rank, gram_max_abs=error, gram_condition=condition,
                projector_change_frobenius=movement, offset_norm=float(np.linalg.norm(array(module.offset))),
                basis_change_frobenius=float(np.linalg.norm(measured-initial)),
                mean_angle_deg=float(angles.mean()), max_angle_deg=float(angles.max()), angles_deg=angles.tolist())


def historical_observation(legal_cam, y):
    # Exact canonical FP64 uncentered moments, plus a centered estimator check.
    x = legal_cam.detach().cpu().double()
    samples = [x[torch.from_numpy(y == c)] for c in (0, 1)]
    counts = torch.tensor([s.size(0) for s in samples], dtype=torch.long)
    means = torch.stack([s.mean(0) for s in samples])
    seconds = torch.stack([torch.mm(s.T.contiguous(), s.contiguous())/s.size(0) for s in samples])
    for i, subset in enumerate(samples):
        centered = array(subset) - array(means[i])
        want = centered.T @ centered / (len(subset)-1)
        have = (array(seconds[i])-np.outer(array(means[i]), array(means[i])))*len(subset)/(len(subset)-1)
        np.testing.assert_allclose(have, want, atol=TOL['audit_fp64_atol'], rtol=1e-10)
    return dict(ema_mean=means, ema_second=seconds, ema_counts=counts)


def check_ema(module, before, observation):
    current = module.ema_state()
    alpha = CFG['ema']['alpha']
    assert torch.equal(current['ema_counts'], observation['ema_counts'])
    for key in ('ema_mean', 'ema_second'):
        np.testing.assert_allclose(array(current[key]), (1-alpha)*array(before[key])+alpha*array(observation[key]),
                                   atol=4e-15, rtol=1e-14)
    assert int(current['ema_updates']) == int(before['ema_updates']) + 1


def initial_basis(state, module, spectrum, old_cache):
    basis = array(module.U0)
    assert torch.equal(module.U0.cpu(), spectrum['spectrum']['U'])
    assert torch.equal(module.basis(), module.U0)
    assert np.all(basis[np.abs(basis).argmax(0), np.arange(16)] > 0)
    assert not module.offset.count_nonzero()
    ids = rows_legal(state)
    y = state['labels_np'][ids]
    cam = spectrum['warmup_cam'][ids]
    obs = historical_observation(cam, y)
    for key in obs:
        torch.testing.assert_close(module.ema_state()[key], obs[key], atol=2e-15, rtol=1e-14)
    assert torch.equal(old_cache['stats']['counts'], module.ema_counts.cpu())
    assert int(module.risk_updates) == 0 and not module.margin_memory.count_nonzero()

    return basis_diagnostics(module)


def analytic_jacobian(head, z):
    norm, affine, gelu, drop = head.mlp_head
    assert gelu.approximate == 'none' and isinstance(drop, torch.nn.Dropout)
    x = z.double()
    centered = x-x.mean(1, keepdim=True)
    inverse = (centered.square().mean(1, keepdim=True)+norm.eps).rsqrt()
    normalized = centered*inverse
    input_affine = (normalized*norm.weight.double()+norm.bias.double())@affine.weight.double().T+affine.bias.double()
    d_gelu = (1+torch.erf(input_affine/math.sqrt(2)))/2 + input_affine*torch.exp(-input_affine.square()/2)/math.sqrt(2*math.pi)
    weight_gap = head.classifier.weight[0].double()-head.classifier.weight[1].double()
    upstream = (d_gelu*weight_gap) @ affine.weight.double()*norm.weight.double()
    jac = inverse*(upstream-upstream.mean(1, keepdim=True)-normalized*(upstream*normalized).mean(1, keepdim=True))
    output = F.gelu(input_affine)@head.classifier.weight.double().T+head.classifier.bias.double()
    return jac.float(), output


def native_head_logits(head, z):
    # Verify the deterministic gap against the real FP32 classifier. The
    # FP64 analytic reference is compared per logit before subtracting them.
    values = []
    hook = head.classifier.register_forward_hook(lambda _h, _args, value: values.append(value))
    dropout = head.mlp_head[-1]
    was_training = dropout.training
    try:
        dropout.eval()
        head(z, None)
    finally:
        dropout.train(was_training)
        hook.remove()
    assert len(values) == 1
    return values[0]


def manual_projection(module, basis, z):
    count, center, second = (array(getattr(module, name)) for name in ('ema_counts', 'ema_mean', 'ema_second'))
    weights = count/count.sum()
    mean = weights@center
    covariance = np.stack([(m2-np.outer(m, m))*n/(n-1) for m, m2, n in zip(center, second, count)])
    pooled_total = (np.einsum('c,cij->ij', weights, second)-np.outer(mean, mean))*count.sum()/(count.sum()-1)
    u = array(basis)
    raw_var = np.einsum('id,ij,jd->d', u, pooled_total, u).clip(min=EPS)
    within = np.einsum('id,cij,jd->cd', u, covariance, u).clip(min=0)
    hp = CFG['class_statistics']
    ridge = max(hp['relative_floor']*np.trace(pooled_total)/len(mean), hp['variance_floor'])
    covs = np.stack([u.T@(hp['own_variance']*cov+hp['pooled_variance']*pooled_total)@u+ridge*np.eye(16)
                     for cov in covariance])
    posdef, logdet = np.linalg.slogdet(covs)
    assert (posdef == 1).all() and (np.linalg.eigvalsh(covs) > 0).all()
    h64 = (array(z).astype(np.float64)-mean)@u
    means = (center-mean)@u
    centered = h64[None]-means[:, None]
    precision = np.linalg.solve(covs, np.broadcast_to(np.eye(16), covs.shape))
    terms = centered*np.einsum('cij,cnj->cni', precision, centered)
    attribution = ((terms[1]-terms[0])/2+(logdet[1]-logdet[0])/32)
    log_densities = -.5*(terms.sum(-1)+logdet[:, None]+16*np.log(2*np.pi))
    total_llr = log_densities[0]-log_densities[1]
    np.testing.assert_allclose(attribution.sum(-1), total_llr, atol=1e-10, rtol=1e-11)
    stable = (1/(np.diagonal(precision, axis1=-1, axis2=-2)*np.diagonal(covs, axis1=-1, axis2=-2))).clip(0,1)

    def normalize(x):
        return (x-x.mean())/math.sqrt(x.var()+EPS)

    contrast = np.abs(within[0]-within[1])/(weights@within+EPS)
    return {key: torch.as_tensor(value, device=z.device, dtype=torch.float64) for key,value in dict(
        hh=h64/np.sqrt(raw_var+EPS), projected=h64, class_mean=means, class_cov=covs, class_logdet=logdet,
        stability=stable.mean(0), raw_llr=attribution, llr_total=total_llr,
        loglam=normalize(np.log(raw_var)), nu=normalize(np.log1p(contrast))).items()}


def independent_distance(d):
    mean, cov = array(d['class_mean']), array(d['class_cov'])
    mid = np.mean(cov, axis=0)
    dm = mean[0]-mean[1]
    location = float(dm@np.linalg.solve(mid, dm)/8)
    volume = float((np.linalg.slogdet(mid)[1]-np.linalg.slogdet(cov)[1].mean())/2)
    return location+max(0.,volume), location, max(0.,volume)


@torch.no_grad()
def manual_forward(module, z, jac, gap, intervention):
    basis = module.U0 if intervention == 'spectral_guidance_off' else torch.as_tensor(
        clean_basis(module)[0], dtype=module.U0.dtype, device=z.device)
    full = manual_projection(module, basis, z)
    hh, loglam, nu, stability = (full[key].to(z.dtype) for key in ('hh', 'loglam', 'nu', 'stability'))
    limit = CFG['class_statistics']['llr_bound']
    llr_token = torch.tanh(full['raw_llr']/limit).to(z.dtype)
    coords = (hh.abs(), loglam.broadcast_to(hh.shape), nu.broadcast_to(hh.shape))
    token = module.norm(module.direction(basis.float().T)[None, :, :] +
                        module.evidence(torch.stack((hh, *coords, llr_token), dim=2)))
    keys = module.key(token)
    q = module.q0
    # Match the contraction layout used on this device. The division by a
    # small calibration scale magnifies upstream FP32 accumulation roundoff.
    score = torch.einsum('kd,njd->nkj', q, keys) / math.sqrt(q.shape[1])
    nscore = (q @ module.null_key / math.sqrt(q.shape[1]))[None, :, None].broadcast_to((len(z), 8, 1))
    attention = (torch.cat((score, nscore), dim=2)/CFG['fusion']['temperature']).softmax(2)
    responses, nulls = [], []
    for prompt_rows in (CFG['prompt_rows_0based']['ASD'], CFG['prompt_rows_0based']['TD']):
        a = attention[:, prompt_rows]
        nulls.append(a[..., -1].mean(1))
        hits = 1-(1-a[..., :-1]).prod(1)
        responses.append((1-nulls[-1])[:, None]*hits/(hits.sum(1, keepdim=True)+EPS))
    ns = torch.stack(nulls, dim=1)
    s = ((responses[0]-responses[1])/module.tau_s).tanh()
    l = torch.tanh(limit*llr_token/module.tau_l)
    stable = stability.broadcast_to(s.shape)
    res = module.spectral(torch.stack((*coords, stable), dim=2)).squeeze(2)
    rel = (1+(16*res.softmax(1)).tanh())*(1+stable.sqrt())/4
    if intervention == 'semantic_evidence_off':
        ns = torch.zeros_like(ns)
        s = torch.zeros_like(s)
    n0 = ns[:, 0, None].expand_as(s)
    n1 = ns[:, 1, None].expand_as(s)
    mix_x = torch.stack((s, l, s.abs(), l.abs(), s*l, n0, n1, rel), dim=2)
    hp = CFG['fusion']
    blend = (hp['alpha_max']-hp['alpha_min'])*torch.sigmoid(module.fusion_net(mix_x)[..., 0])+hp['alpha_min']
    difference = (1-(n0+n1)/2)*rel*(blend*s+(1-blend)*l)
    confidence = torch.tanh(0.5*gap[:, None])
    stat_gap = torch.tanh(full['llr_total'][:, None]/(4*module.tau_l)).float()
    c = confidence.broadcast_to(s.shape)
    p = stat_gap.broadcast_to(s.shape)
    info = torch.stack((c, c.abs(), s, l, s*l, n0, n1, stable, p), dim=2)
    credibility = torch.sigmoid(module.gate_net['credibility'](info)[..., 0])
    gate_input = torch.cat((F.layer_norm(z, (144,)), difference, rel, ns, confidence, stat_gap), dim=1)
    gate = credibility * torch.tanh(module.gate_net['direction'](gate_input))
    theta = gate*difference
    j = jac.double() @ basis
    norm_sq = torch.sum(j*j, dim=1, keepdim=True)+CFG['jacobian_epsilon']**2
    first_order = (theta.double()*j.square()).sum(1)/norm_sq[:, 0]
    raw = (theta*(j/norm_sq.sqrt()).float())@basis.float().T
    rad = raw.double().norm(dim=1, keepdim=True)
    z_norm = z.double().norm(dim=1, keepdim=True)
    cap = CFG['residual_limit']*z_norm.clamp_min(EPS)
    radius = cap*torch.tanh(rad/cap)
    scale = torch.where(rad == 0, torch.ones_like(rad), radius/rad.clamp_min(1e-300))
    delta = (raw.double()*scale*(z_norm/z_norm.clamp_min(EPS))*
             (z_norm/(z_norm+torch.finfo(z.dtype).tiny))).float()
    if module.fold == 2:
        delta[list(FOLD2_ROWS)]=0
    if intervention == 'residual_off':
        delta.zero_()
    return dict(**{key:full[key] for key in ('projected','class_mean','class_cov','class_logdet','raw_llr','llr_total')},
                basis=basis, hh=hh, attention=attention, llr=limit*llr_token,
                signed_semantic=s, signed_statistical=l, active_null=ns, reliability=rel, alpha=blend,
                difference=difference, gate=gate, theta=theta, first_order_score=first_order,
                raw=raw, delta=delta, enhanced=z+delta, jacobian_projected=j, credibility=credibility)


@torch.no_grad()
def check_formulas(state, module, z, intervention='full'):
    from ps_cdso import head_sensitivity, gaussian_separation

    before = r.rng_mark()
    head = state['cam'].Outputlayer
    j, gap = head_sensitivity(head, z)
    analytic, analytic_logits = analytic_jacobian(head, z)
    compare(j, analytic)
    native_logits = native_head_logits(head, z)
    compare(native_logits, analytic_logits.float())
    native_gap = native_logits[:, 0] - native_logits[:, 1]
    assert torch.equal(gap, native_gap)
    actual = module(z, j, gap, intervention)
    expected = manual_forward(module, z, j, native_gap, intervention)
    np.testing.assert_allclose(array(actual['raw_llr'].sum(-1)), array(actual['llr_total']), atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose([float(v) for v in gaussian_separation(actual)], independent_distance(expected),
                               atol=TOL['audit_fp64_atol'], rtol=1e-10)
    for key in expected:
        try:
            if key in ('projected','class_mean','class_cov','class_logdet','raw_llr','llr_total'):
                torch.testing.assert_close(actual[key], expected[key], atol=TOL['audit_fp64_atol'], rtol=1e-10)
            else:
                compare(actual[key], expected[key])
        except AssertionError as exc:
            raise AssertionError((key, intervention, str(exc))) from exc
        assert torch.isfinite(actual[key]).all(), key
    assert float((actual['attention'].sum(-1)-1).abs().max()) < 2e-6
    relative = actual['delta'].double().norm(dim=-1) / z.double().norm(dim=-1).clamp_min(1e-300)
    assert float(relative.max()) <= CFG['residual_limit']+TOL['residual_bound_roundoff']
    if state['fold'] == 2:
        assert not actual['delta'][list(FOLD2_ROWS)].count_nonzero()
    if intervention == 'residual_off':
        assert torch.equal(actual['enhanced'], z)
    if intervention == 'semantic_evidence_off':
        assert not actual['signed_semantic'].count_nonzero() and not actual['active_null'].count_nonzero()
    if intervention == 'spectral_guidance_off':
        assert torch.equal(actual['basis'], module.U0)
    assert r.rng_mark() == before
    return dict(max_relative=float(relative.max()),
                gaussian_llr_max_error=float((actual['llr_total']-expected['llr_total']).abs().max()),
                llr_attribution_max_error=float((actual['raw_llr'].sum(-1)-actual['llr_total']).abs().max()),
                class_cov_condition=[float(x) for x in np.linalg.cond(array(expected['class_cov']))],
                delta_max_error=float((actual['delta']-expected['delta']).abs().max()),
                jacobian_max_error=float((j-analytic).abs().max()),
                deterministic_logits_fp64_max_error=float((native_logits.double()-analytic_logits).abs().max()),
                deterministic_gap_exact=True)


def check_risk(module, before, observation):
    current = module.risk_state()
    count = int(before['risk_updates'])
    observed = array(observation)
    past = array(before['margin_memory'])
    hp = CFG['long_protection']
    rate = np.where(observed >= past, hp['rise_alpha'], hp['fall_alpha']).astype(observed.dtype)
    expected = past+rate*(observed-past) if count else observed
    np.testing.assert_array_equal(array(current['margin_memory']), expected)
    assert int(current['risk_updates']) == count+1


def check_true_gain(state, probe, zero=False):
    ids = rows_legal(state)
    y = state['labels_np'][ids]
    assert probe['gain_grad_main_output'].isfinite().all()
    w0, b0, w1, b1, w2, b2 = probe['head_parameters']
    ln = state['cam'].Outputlayer.mlp_head[0]
    z = probe['zero_cam']
    hidden = F.gelu(F.linear(F.layer_norm(z, ln.normalized_shape, w0, b0, ln.eps), w1, b1))
    logits_off = F.linear(hidden*probe['native_mask'], w2,b2)
    assert torch.equal(logits_off.log_softmax(1), probe['off'])
    assert torch.equal(probe['native_logits'].log_softmax(1), probe['on'])
    on, off = (array(probe[key][ids]).astype(np.float64) for key in ('on','off'))
    sign = (1-2*y)
    signed = np.stack((sign, -sign), axis=1)
    m_on, m_off = (np.sum(v*signed, axis=1) for v in (on,off))
    hp = CFG['loss']
    q = hp['g2_weight_floor']+(1-hp['g2_weight_floor'])/(1+np.exp(np.clip(m_off-hp['g2_margin_center'],-100,100)))
    t, target = hp['g2_gain_temperature'], hp['g2_gain_target']
    diff = t*(np.logaddexp(0,(target-m_on)/t)-np.logaddexp(0,(target-m_off)/t))
    mean = .5*sum(np.mean((q*diff)[y == c]) for c in (0,1))
    slope = -q/(1+np.exp(np.clip((m_on-target)/t,-100,100)))
    slope /= (2*np.bincount(y)[y])
    grad = array(probe['gain_grad_main_output'])
    np.testing.assert_allclose(grad[ids], slope[:, None]*signed, atol=1e-8, rtol=2e-6)
    assert not np.count_nonzero(grad[np.setdiff1d(np.arange(CFG['subjects']), ids)])
    assert abs(probe['gain'] - mean) <= 1e-7
    assert abs(probe['margin_delta'] - np.mean(m_on-m_off)) <= 1e-7
    if zero:
        assert probe['gain'] == 0 and not np.count_nonzero(m_on-m_off)
    else:
        assert probe['gain_grad_u_norm'] > 0 and np.count_nonzero(m_on-m_off)
    return dict(loss=float(mean), margin_delta=float(np.mean(m_on-m_off)),
                gain_grad_u_norm=probe['gain_grad_u_norm'], legal_n=len(ids))


def check_shared_probes(probe):
    result = {}
    for name, raw in probe['shared'].items():
        base, enhanced, risk, step = (array(value).astype(np.float64) for value in raw)
        inc = step-base
        nb, nr = np.linalg.norm(base), np.linalg.norm(risk)
        ratio = np.linalg.norm(inc)/nb if nb else 0
        # Recompute actual representable update from the parameter vectors.
        assert ratio <= CFG['loss']['shared_ratio_max']+1e-13
        assert float(inc@base) >= -1e-12
        assert nr <= CFG['loss']['norm_floor'] or float(inc@risk) >= -1e-12
        if nb <= CFG['loss']['norm_floor']:
            np.testing.assert_array_equal(base, step)
        result[name] = dict(ratio=float(ratio), risk=float(nr),
                            baseline_inner_product=float(inc@base), risk_inner_product=float(inc@risk))
    return result


def synthetic_checks():
    from ps_cdso import PSCDSO, gaussian_separation, smooth_limit

    z = torch.full((4, 144), 1e-18)
    z[0].zero_()
    z[1].fill_(1e-36)
    z[2].fill_(1e-12)
    z[3].fill_(1.0)
    raw = torch.ones_like(z).requires_grad_()
    delta = smooth_limit(raw, z, CFG['residual_limit'])
    assert (delta.double().norm(dim=1) <= (CFG['residual_limit']+2e-8)*z.double().norm(dim=1)).all()
    delta.sum().backward()
    assert torch.isfinite(raw.grad).all()
    zero = torch.zeros_like(z).requires_grad_()
    grad, = torch.autograd.grad(smooth_limit(zero, torch.ones_like(zero), CFG['residual_limit']).sum(), zero)
    assert torch.equal(grad, torch.ones_like(grad))
    test = torch.tensor([1., -1.])
    labels = torch.tensor([0, 1])
    assert torch.equal(r.true_margin(torch.stack((test, torch.zeros_like(test)), 1), labels), torch.ones(2))
    v = torch.eye(144, dtype=torch.float64)[:, :16]
    means = torch.zeros(2,144,dtype=torch.float64)
    means[:,0] = torch.tensor([2.,-2.])
    cov = torch.eye(144, dtype=torch.float64).expand(2,144,144).clone()
    cov[:,0,1] = cov[:,1,0] = torch.tensor([.6,-.4])
    seconds = cov+means[:, :, None]*means[:,None,:]
    with h.isolated_rng(823800):
        model = PSCDSO(v, torch.zeros(8,32), dict(ema_mean=means,ema_second=seconds,ema_counts=torch.tensor([8,11])), 1, CFG)
    inputs = means.clone()
    score = model.projection(inputs,model.basis())
    assert score['llr_total'][0] > 0 > score['llr_total'][1]
    assert score['class_cov'][:,0,1].abs().min() > 0
    expect = manual_projection(model,model.basis(),inputs)
    np.testing.assert_allclose(array(score['raw_llr']),array(expect['raw_llr']),atol=1e-12,rtol=1e-11)
    sep = gaussian_separation(score)
    np.testing.assert_allclose([float(x.detach()) for x in sep], independent_distance(expect),atol=1e-12)
    for b, e, q in (([1.,1.,1.],[-2.,3.,1.5],[1.,0.,0.]),
                    ([0.,0.,0.],[1.,2.,3.],[0.,1.,2.]),
                    ([3.,-1.,0.],[4.,-1.,1.],[-3.,1.,0.]),
                    ([1.,1.,0.],[.5,1.,.5],[0.,1.,-1.])):
        base, proposed, risk = map(torch.tensor, (b,e,q))
        actual, details = r.coordinate_shared(base, proposed, risk)
        check_shared_probes({'shared':{'synthetic':(base,proposed,risk,actual)}})
        assert details['ratio'] <= CFG['loss']['shared_ratio_max']
    # Nonzero reference pressure changes the native proposal under a small,
    # feasible update, and is disabled for exact initialization.
    a = torch.tensor([1.,1.,0.])
    protected, info = r.coordinate_shared(a,a,torch.tensor([.1,.1,-.1]))
    assert info['risk_dot'] > 0 and not torch.equal(a,protected)



def original_eval(state):
    head = state['cam'].Outputlayer
    found = []
    capture = head.register_forward_pre_hook(lambda _head, args: found.append(args[0]))
    try:
        out = h.runtime().forward(state, r.BASE_GROUP, training=False)
    finally:
        capture.remove()
    assert len(found) == 1 and found[0].shape == (CFG['subjects'], 144)
    return out['logp'], found[0]


def baseline(state, identity, initial):
    source = h.runtime()
    fold = state['fold']
    filename = source.best_path(r.BASE_GROUP, fold)
    archived = h.torch_load(filename)
    with (source.EXP/'results/fold_metrics.csv').open() as stream:
        rows = [x for x in csv.DictReader(stream) if x['group'] == r.BASE_GROUP and int(x['fold']) == fold]
    assert len(rows) == 1
    row = rows[0]
    assert archived['identity'] == identity and identity['split_sha'] == state['split_sha']
    assert r.sha_file(filename) == row['checkpoint_sha256']
    assert archived['best_epoch_1based'] == int(row['best_epoch_1based'])
    ids = state['test_np']
    truth = state['labels_np'][ids]
    before = r.rng_mark()
    source.load_history().load_bundle(state, archived['best_state'])
    try:
        logp, _ = original_eval(state)
        p = array(logp.exp()[ids])
        assert np.array_equal(p, archived['test_probability'])
        assert np.array_equal(p.argmax(1), archived['test_prediction'])
        found = metrics(p, truth)
        same_metrics(found, archived['metrics'])
        same_metrics(found, {key: float(row[key]) for key in AUC_KEYS})
        historical_moments = optimizer_moment_snapshot(state)
        historical_native_digests = r.native_digests(state)
    finally:
        source.load_history().load_bundle(state, initial['current']['base'])
    assert before == r.rng_mark()
    return dict(group='B0', fold=fold, best_epoch=archived['best_epoch_1based'],
                end_epoch=int(row['epochs_executed']), metrics=found, probability=p,
                gap=array((logp[:, 0]-logp[:, 1])[ids]), checkpoint_sha=row['checkpoint_sha256'],
                native_digests=historical_native_digests, optimizer_moments=historical_moments)


def optimizer_moment_snapshot(state):
    snapshot = {}
    for name, parameters in state['old_layout'].items():
        slots = state[name.split('.')[0]].state
        snapshot[name] = {field: math.sqrt(sum(float(np.square(array(slots[p][field]).astype(np.float64)).sum())
                                               for p in parameters if p in slots))
                          for field in ('exp_avg','exp_avg_sq')}
    return snapshot


def verify_moments(state, step, epoch):
    values = optimizer_moment_snapshot(state)
    for name, norms in values.items():
        reported = step['shared'][name]['optimizer_moments']
        np.testing.assert_allclose(list(reported.values()), list(norms.values()), rtol=1e-12, atol=1e-14,
                                   err_msg=f'{name}, epoch {epoch}')
    return values


def check_optimizers(state, module, opt, epoch):
    result = {}
    for key, optimizer in zip((*r.HIST_OPTS, 'opt_new'), [state[k] for k in r.HIST_OPTS]+[opt]):
        settings = CFG['historical_optimizers'].get(key, ('AdamW', CFG['new_optimizer']['lr'], CFG['new_optimizer']['weight_decay']))
        assert type(optimizer).__name__ == settings[0]
        assert all(pg['lr'] == settings[1] and pg['weight_decay'] == settings[2] for pg in optimizer.param_groups)
        steps = []
        for parameter, moments in optimizer.state.items():
            assert {'step', 'exp_avg', 'exp_avg_sq'} <= moments.keys()
            for name in ('exp_avg', 'exp_avg_sq'):
                val = moments[name]
                assert val.shape == parameter.shape and val.dtype == parameter.dtype and torch.isfinite(val).all()
            assert (moments['exp_avg_sq'] >= 0).all()
            steps.append(int(moments['step']))
        assert steps and min(steps) > 0 and max(steps) <= epoch
        if key == 'opt_new':
            assert len(steps) == len(list(module.parameters()))
            assert min(steps) == max(steps) == epoch-r.START
        result[key] = dict(slots=len(steps), min_step=min(steps), max_step=max(steps))
    r.parameter_layout(state, module, opt)
    return result


def check_journal(state, group, task, best, latest, rows, pre):
    assert [row['epoch'] for row in rows] == list(range(r.START+1, task['end_epoch']+1))
    historical = {row['epoch']:row for row in h.historical_task(state['fold'])['rng_trace']}
    paired = {}
    if group == 'G2':
        with h.paths('G1', state['fold'])[3].open() as stream:
            paired = {row['epoch']: row for row in map(json.loads, stream)}
    previous = pre['initial_rng'], pre['initializer']
    ema, memory = task['initial_ema_sha'], pre['initial_risk_sha']
    truth = state['labels_np'][state['test_np']]
    max_correct, wait, when, shared_n, paired_n = -1, 0, None, 0, 0
    s_summary = {key: dict(conflicts=0, capped=0, positive_applied=0, maximum_ratio=0., maximum_raw_ratio=0.,
                            cosine_sum=0., cosine_n=0, risk_steps=0, risk_conflicts=0, positive_risk_dot=0,
                            cumulative_increment_norm=0., cumulative_base_norm=0., cast_fallbacks=0,
                            peak_risk_grad=0.)
                 for key in state['old_layout']}
    a_summary = {key: dict(positive_class=0, maximum_ratio=0., maximum_class_norm=0.,
                            conflict_steps=0, terms={loss:dict(max_norm=0., opposed_steps=0) for loss in r.AUX_KEYS})
                 for key in state['new_layout']}
    for row in rows:
        epoch = row['epoch']
        assert wait <= CFG['early_stop_wait_gt']
        pred = np.unpackbits(np.frombuffer(bytes.fromhex(row['pred_bits']), dtype=np.uint8), count=len(truth))
        correct = int(np.equal(pred, truth).sum())
        better = correct > max_correct
        if better:
            max_correct, wait, when = correct, 0, epoch
        else:
            wait += 1
        assert (better, correct, wait, when) == (row['improved'], row['correct'], row['wait'], row['best_epoch'])
        assert (row['rng_before'], row['rng_before_streams']) == previous
        assert row['rng_after_train'] == row['rng_after_eval']
        assert row['dropout_calls'] == pre['dropout_calls']
        assert all(row['rng_after_streams'][name] == previous[1][name] for name in ('python', 'numpy', 'cpu'))
        previous = row['rng_after_eval'], row['rng_after_streams']
        assert row['ema_before'] == ema
        ema = row['ema_after']
        assert row['risk_before'] == memory
        memory = row['risk_after']
        assert row['risk']['active'] == (epoch > r.START+1)
        assert 0. <= row['risk']['below_floor_fraction'] <= 1.
        assert 0. <= row['risk']['weight_average'] <= 1.
        assert row['spectral']['llr_closure_max'] < 1e-8
        assert len(row['dropout_mask_sha']) == 64
        if epoch in historical:
            assert {key: row[key] for key in r.RNG_KEYS} == historical[epoch]
            shared_n += 1
        if epoch in paired:
            assert all(row[key] == paired[epoch][key] for key in (*r.RNG_KEYS, 'rng_before_streams', 'rng_after_streams', 'dropout_calls','dropout_mask_sha'))
            paired_n += 1
        for key, g in row['shared'].items():
            assert g['ratio'] <= CFG['loss']['shared_ratio_max'] and g['dot'] >= 0
            assert g['applied'] <= CFG['loss']['shared_ratio_max']*g['base'] + 1e-20
            if g['risk'] > CFG['loss']['norm_floor']:
                assert g['risk_dot'] >= 0
            stats = s_summary[key]
            stats['risk_steps'] += int(g['risk'] > 0)
            stats['risk_conflicts'] += int(g['risk_conflict'])
            stats['positive_risk_dot'] += int(g['risk_dot'] > 0)
            stats['cumulative_increment_norm'] += g['applied']
            stats['cumulative_base_norm'] += g['base']
            stats['cast_fallbacks'] += int(g['cast_fallback'])
            stats['peak_risk_grad'] = max(stats['peak_risk_grad'], g['risk'])
            stats['conflicts'] += int(g['conflict'])
            stats['capped'] += int(g['limited'])
            stats['positive_applied'] += int(g['applied'] > 0)
            stats['maximum_ratio'] = max(stats['maximum_ratio'], g['ratio'])
            stats['maximum_raw_ratio'] = max(stats['maximum_raw_ratio'], g['raw']/g['base'] if g['base'] else 0.)
            if g['cosine'] is not None:
                stats['cosine_sum'] += g['cosine']
                stats['cosine_n'] += 1
            if epoch == task['end_epoch']:
                stats['terminal_relative_drift'] = g['relative_drift']
                stats['terminal_optimizer_moments'] = g['optimizer_moments']
            if epoch == task['best_epoch']:
                stats['best_relative_drift'] = g['relative_drift']
                stats['best_optimizer_moments'] = g['optimizer_moments']
        for key, g in row['added'].items():
            stats = a_summary[key]
            assert g['ratio'] <= CFG['loss']['aux_ratio_max'] and g['applied'] <= CFG['loss']['aux_ratio_max']*g['classification']+1e-20
            stats['maximum_ratio'] = max(stats['maximum_ratio'], g['ratio'])
            stats['maximum_class_norm'] = max(stats['maximum_class_norm'], g['classification'])
            stats['positive_class'] += int(g['classification'] > 0)
            stats['conflict_steps'] += int(g['cosine'] is not None and g['cosine'] < 0)
            for term, t in g['terms'].items():
                item = stats['terms'][term]
                item['max_norm'] = max(item['max_norm'], t['norm'])
                item['opposed_steps'] += int(t['cosine'] is not None and t['cosine'] < 0)
    r.check_zero_start(pre['historical_epoch11'], rows[0])
    assert all(x['classification_norm'] > 0 for x in rows[1]['shared'].values())
    assert all(x['class_only_verified'] for x in rows[1]['added'].values())
    assert all(v['positive_class'] > 0 for v in a_summary.values())
    assert all(g['positive_applied'] > 0 for g in s_summary.values())
    assert (when, max_correct, wait) == (latest['best_epoch'], latest['best_correct'], latest['wait'])
    assert when == best['best_epoch'] == task['best_epoch']
    assert wait > CFG['early_stop_wait_gt'] or task['end_epoch'] == r.LIMIT
    assert previous == (latest['rng_sha'], r.streams(latest['rng']))
    assert ema == h.sha({key:latest['current']['cdso'][key] for key in EMA_KEYS})
    assert memory == h.sha({key:latest['current']['cdso'][key] for key in RISK_KEYS})
    assert any(stats['positive_risk_dot'] > 0 for stats in s_summary.values())
    for stats in s_summary.values():
        stats['cumulative_ratio'] = stats['cumulative_increment_norm']/stats['cumulative_base_norm']
        assert stats['cumulative_ratio'] <= CFG['loss']['shared_ratio_max']
        stats['mean_gradient_cosine'] = stats['cosine_sum']/stats['cosine_n'] if stats['cosine_n'] else None
    return dict(epochs=len(rows), archived_rng_matches=shared_n, beyond_archive_matches=len(rows)-shared_n,
                g1_g2_common_rng_matches=paired_n, shared_gradients=s_summary, added_gradients=a_summary,
                spectral_training={'initial': rows[0]['spectral'], 'best': rows[when-r.START-1]['spectral'],
                                   'latest': rows[-1]['spectral']},
                long_risk={'best': rows[when-r.START-1]['risk'], 'latest':rows[-1]['risk'],
                           'average_risk':float(np.mean([x['risk']['risk'] for x in rows[1:]])),
                           'max_risk':max(x['risk']['risk'] for x in rows[1:])})


def check_point_record(current, step, evaluation, observed, point):
    assert step['rng_after_eval'] == current['rng_sha'] and step['rng_after_streams'] == r.streams(current['rng'])
    assert step['prob_sha'] == h.sha(evaluation['probability'])
    assert step['off_prob_sha'] == h.sha(evaluation['off_probability'])
    assert step['legal_cam_sha'] == h.sha(evaluation['legal_cam'])
    assert step['observation_sha'] == h.sha(observed)
    assert np.array_equal(current['probability'], evaluation['probability'])
    assert np.array_equal(current['off_probability'], evaluation['off_probability'])
    ema = {key: current['current']['cdso'][key] for key in EMA_KEYS} if point == 'best' else current['last_eval_ema']
    assert h.sha(ema) == step['ema_before']
    assert int(ema['ema_updates']) == step['epoch']-r.START-1
    memory = {key: current['current']['cdso'][key] for key in RISK_KEYS} if point == 'best' else current['last_eval_risk']
    assert h.sha(memory) == step['risk_before']
    assert int(memory['risk_updates']) == step['epoch']-r.START-1
    assert h.sha(evaluation['legal_off_margin']) == step['risk_obs']
    if point == 'best':
        assert current['evaluation_risk_sha'] == step['risk_before']
        assert torch.equal(current['legal_off_margin'], evaluation['legal_off_margin'])


@torch.no_grad()
def projected_scores(state, module, z, j, logp):
    ids = rows_legal(state)
    y = state['labels_np'][ids]
    x = array(z[ids]).astype(np.float64)
    chosen = array(module.basis())
    origin = array(module.U0)
    by = [x[y == c] for c in (0, 1)]
    cov = [np.cov(xx, rowvar=False, ddof=1) for xx in by]
    total = np.cov(x, rowvar=False, ddof=1)
    means = np.stack([xx.mean(0) for xx in by])
    hp = CFG['class_statistics']
    ridge = max(hp['variance_floor'], hp['relative_floor']*np.trace(total)/144)
    cov = [hp['own_variance']*cc+hp['pooled_variance']*total+ridge*np.eye(144) for cc in cov]
    margin = (1-2*y)*array(logp[ids, 0]-logp[ids, 1]).astype(np.float64)
    prob_other = 1/(1+np.exp(np.clip(margin, -200, 200)))
    grad = (-prob_other*(1-2*y))[:, None]*array(j[ids]).astype(np.float64)
    hp_loss = CFG['loss']
    weights = hp_loss['g2_weight_floor']+(1-hp_loss['g2_weight_floor'])/(
        1+np.exp(np.clip(margin-hp_loss['g2_margin_center'], -200, 200)))
    weighted = np.zeros(len(x), dtype=np.float64)
    for c in (0, 1):
        subset = y == c
        weighted[subset] = 0.5*weights[subset]/weights[subset].sum()
    error_cov = np.einsum('ni,n,nj->ij', grad, weighted, grad)

    def score(basis):
        c0, c1 = [basis.T @ cc @ basis for cc in cov]
        pc = (c0+c1)*0.5
        dm = (means[0]-means[1])@basis
        mean = dm @ np.linalg.solve(pc, dm)/8
        det = (np.linalg.slogdet(pc)[1]-(np.linalg.slogdet(c0)[1]+np.linalg.slogdet(c1)[1])*0.5)*0.5
        energy = np.trace(basis.T@error_cov@basis)/max(np.trace(error_cov), 1e-12)
        covariance = basis.T@total@basis
        squared_variation = np.linalg.eigvalsh(covariance)
        positive = squared_variation.clip(min=0)
        shape_condition = max(np.linalg.cond(c0),np.linalg.cond(c1))
        unit = positive/max(positive.sum(), 1e-12)
        eff_rank = np.exp(-np.sum(unit*np.log(unit.clip(min=1e-300))))
        return dict(bhattacharyya=float(mean+det), bhattacharyya_mean=float(mean), bhattacharyya_covariance=float(det),
                    error_gradient_coverage=float(energy), eval_covariance_rank=int(np.linalg.matrix_rank(covariance)),
                    variance_effective_rank=float(eff_rank),
                    directional_variance_min=float(np.min(np.diag(covariance))), condition=float(shape_condition))
    return dict(U0=score(origin), learned=score(chosen), legal_ids_sha=h.sha(ids),
                n_legal=len(ids), deterministic_eval_gradient=True)


@torch.no_grad()
def best_interventions(state, module, best, source):
    from ps_cdso import head_sensitivity, gaussian_separation

    before = r.rng_mark()
    logp_off, z = original_eval(state)
    head = state['cam'].Outputlayer
    j, head_gap = head_sensitivity(head, z)
    ids = state['test_np']
    modes = {}
    assertions = {}
    for mode in INTERVENTIONS:
        assertions[mode] = check_formulas(state, module, z, mode)
        d = module(z, j, head_gap, mode)
        logp = head(d['enhanced'], None)[0]
        if mode == 'residual_off':
            assert torch.equal(logp, logp_off)
        modes[mode] = dict(d=d, probability=array(logp.exp()[ids]), logp=logp)
    assert np.array_equal(best['probability'], modes['full']['probability'])
    assert np.array_equal(best['off_probability'], modes['residual_off']['probability'])
    # A deterministic cyclic permutation changes medical attention but must have
    # zero effect on the semantic-off output. No alternative query is trained.
    original_query = module.q0.clone()
    try:
        module.q0.copy_(original_query.roll(4, 0))
        evidence_free = module(z, j, head_gap, 'semantic_evidence_off')
        assert torch.equal(evidence_free['enhanced'], modes['semantic_evidence_off']['d']['enhanced'])
        assert not torch.equal(evidence_free['attention'], modes['semantic_evidence_off']['d']['attention'])
    finally:
        module.q0.copy_(original_query)
    assert r.rng_mark() == before

    d = modes['full']['d']
    p0, poff, pfull = (source['probability'], modes['residual_off']['probability'], modes['full']['probability'])
    truth = state['labels_np'][ids]
    legal_diagnostic = ~np.isin(ids, FOLD2_ROWS) if state['fold'] == 2 else np.ones(len(ids), dtype=bool)
    factor = np.zeros(len(ids), dtype=np.int64)
    factor[legal_diagnostic] = 1-2*truth[legal_diagnostic]
    gap = {mode: array((cell['logp'][:, 0]-cell['logp'][:, 1])[ids]) for mode,cell in modes.items()}
    delta = d['delta'].double()
    norm_z = z.double().norm(dim=1).clamp_min(1e-300)
    norm_delta = delta.norm(dim=1)
    norm_j = j.double().norm(dim=1)
    p_j = d['jacobian_projected']
    j64 = j.double()
    normalizer = norm_j.square().clamp_min(1e-24)
    learning_alignment = array((j64*delta).sum(1)/(norm_j*norm_delta).clamp_min(1e-24))[ids]*factor
    cover_0 = (j64@module.U0).square().sum(1)/normalizer
    cover = p_j.square().sum(1)/normalizer
    disagreements = d['signed_semantic']*d['signed_statistical'] < 0
    llr_overrides = (d['difference']*d['signed_semantic'] < 0) & disagreements
    first_order = d['theta'].double()*p_j.square()
    net = first_order.sum(1).abs()
    l1 = first_order.abs().sum(1)
    gate_reverse = d['gate'] < 0
    margin_0 = source['gap']*factor
    margin_off = gap['residual_off']*factor
    margin_full = gap['full']*factor
    gaussian = d['llr_total']
    cmax = d['credibility'].mean(1)
    collection = {}
    for i, patient_id in enumerate(ids):
        cell = dict(fold=state['fold'], row_1based=int(patient_id+1), y=int(truth[i]),
                    protected=not bool(legal_diagnostic[i]),
                    b0_label=int(p0[i].argmax()), off_label=int(poff[i].argmax()), full_label=int(pfull[i].argmax()),
                    b0_asd=float(p0[i,0]), off_asd=float(poff[i,0]), full_asd=float(pfull[i,0]),
                    b0_td=float(p0[i,1]), off_td=float(poff[i,1]), full_td=float(pfull[i,1]),
                    b0_margin=float(margin_0[i]) if legal_diagnostic[i] else None,
                    off_margin=float(margin_off[i]) if legal_diagnostic[i] else None,
                    full_margin=float(margin_full[i]) if legal_diagnostic[i] else None,
                    relative_residual=float(norm_delta[patient_id]/norm_z[patient_id]),
                    raw_relative_residual=float(d['raw'][patient_id].double().norm()/norm_z[patient_id]),
                    alignment=float(learning_alignment[i]) if legal_diagnostic[i] else None, j_cover_u0=float(cover_0[patient_id]), j_cover_u=float(cover[patient_id]),
                    conflicting_directions=int(disagreements[patient_id].sum()),
                    llr_overrides=int(llr_overrides[patient_id].sum()), negative_gate_directions=int(gate_reverse[patient_id].sum()),
                    cancellation=float(1-net[patient_id]/l1[patient_id]) if float(l1[patient_id]) else 0.,
                    full_cov_llr=float(gaussian[patient_id]), credibility=float(cmax[patient_id]))
        for mode in ('spectral_guidance_off', 'semantic_evidence_off'):
            cell[mode+'_label'] = int(modes[mode]['probability'][i].argmax())
            cell[mode+'_asd'] = float(modes[mode]['probability'][i,0])
            cell[mode+'_td'] = float(modes[mode]['probability'][i,1])
            cell[mode+'_margin'] = float(gap[mode][i]*factor[i]) if legal_diagnostic[i] else None
        collection[int(patient_id)+1] = cell
    current_indices = projected_scores(state, module, z, j, logp_off)
    realized_distance = independent_distance(manual_projection(module, d['basis'], z))
    current_indices['ema_matched_distance'] = dict(zip(('bhattacharyya','mean','covariance'),realized_distance))
    return dict(formulas=assertions, basis=basis_diagnostics(module), objective=current_indices,
                records=list(collection.values()), test_metrics={key:metrics(value['probability'], truth)
                                                                 for key,value in modes.items()},
                mechanism_test_metrics={key:metrics(value['probability'][legal_diagnostic], truth[legal_diagnostic])
                                        for key,value in modes.items()})


def aligned_epoch(state, base, latest, rows):
    epoch = base['best_epoch']
    item = latest['aligned_b0']
    if epoch > rows[-1]['epoch']:
        assert item is None
        return {'reached': False, 'epoch': epoch}
    assert item['epoch'] == epoch and np.array_equal(item['test_rows'], state['test_np'])
    on_row = rows[epoch-r.START-1]
    assert on_row['aligned_b0_sha'] == h.sha(item)
    assert on_row['off_prob_sha'] == h.sha(item['off_probability'])
    native_moments = base['optimizer_moments']
    for key, norms in item['optimizer_moments'].items():
        assert norms == on_row['shared'][key]['optimizer_moments']
    test = item['test_rows']
    use = ~np.isin(test, FOLD2_ROWS) if state['fold'] == 2 else np.ones(len(test), dtype=bool)
    y = state['labels_np'][test[use]]
    off, b0 = item['off_probability'][use], base['probability'][use]
    rows_ = [dict(row_1based=int(row)+1, y=int(target), b0_label=int(a.argmax()), off_label=int(b.argmax()))
             for row,target,a,b in zip(test[use], y, b0, off)]
    return dict(reached=True, epoch=epoch, score=metrics(off,y), baseline_score=metrics(b0,y),
                n_diagnosed=len(y), changes=changes(rows_,'b0','off'),
                native_component_sha_matched={k:hsh == base['native_digests'][k] for k,hsh in item['native_digests'].items()},
                historical_moment_norms=native_moments, candidate_moment_norms=item['optimizer_moments'])


def audit_fold(group, fold):
    start = time.monotonic()
    r.environment()
    pre = r.accepted_preflight()['folds'][fold-1]
    assert r.task_complete(group, fold)
    task_path, best_path, latest_path, log_path = h.paths(group, fold)
    task, best, latest = h.read_json(task_path), h.torch_load(best_path), h.torch_load(latest_path)
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert latest['best_sha'] == task['best_sha'] and latest['log_sha'] == task['log_sha']
    assert latest['log_bytes'] == log_path.stat().st_size
    state, module, opt, _, warm, meta = r.setup(fold, group)
    assert all(task[k] == best[k] == latest[k] == meta[k] for k in meta)
    assert pre['initial_module_sha'] == meta['initial_module_sha'] and pre['initial_rng'] == warm['rng_sha']
    assert pre['historical_optimizers'] == h.init_optimizers(state, warm)
    native = r.checkpoint(state, module, opt)
    assert np.array_equal(best['test_rows'], state['test_np'])
    base = baseline(state, meta['identity'], native)
    timeline = check_journal(state, group, task, best, latest, rows, pre)
    aligned = aligned_epoch(state, base, latest, rows)
    initial_moments = optimizer_moment_snapshot(state)
    legal = rows_legal(state)
    y_legal = state['labels_np'][legal]

    def load(point):
        assert r.digests(point['current']) == point['digests']
        r.restore(state, module, opt, point['current'], point['rng'])
        assert r.rng_mark() == (point['rng_sha'], r.streams(point['rng']))
        assert r.digests(r.bundle(state, module, opt)) == point['digests']
        for name in ('q0', 'U0', 'eye16', 'tau_s', 'tau_l', 'ema_counts'):
            assert torch.equal(point['current']['cdso'][name], native['current']['cdso'][name])
        assert not any(buffer.requires_grad for buffer in module.buffers())

    load(best)
    best_optimizers = check_optimizers(state, module, opt, task['best_epoch'])
    best_moments = verify_moments(state, rows[task['best_epoch']-r.START-1], task['best_epoch'])
    interventions = best_interventions(state, module, best, base)
    check = r.evaluate(state, module)
    independent = historical_observation(check['legal_cam'], y_legal)
    check_point_record(best, rows[task['best_epoch']-r.START-1], check, independent, 'best')
    same_metrics(interventions['test_metrics']['full'], best['metrics'])
    assert best['evaluation_ema_sha'] == h.sha(module.ema_state())
    assert r.rng_mark()[0] == best['rng_sha']
    base_metrics = {key: value for key,value in base.items() if key not in ('gap','probability')}

    load(latest)
    latest_optimizers = check_optimizers(state, module, opt, task['end_epoch'])
    terminal_moments = verify_moments(state, rows[-1], task['end_epoch'])
    curve = basis_diagnostics(module)
    assert int(module.ema_updates) == task['end_epoch']-r.START
    with torch.no_grad():
        for key in (*EMA_KEYS, *RISK_KEYS):
            view = getattr(module, key)
            previous = latest['last_eval_ema'] if key in EMA_KEYS else latest['last_eval_risk']
            view.copy_(previous[key].to(view))
    terminal = r.evaluate(state, module)
    obs = historical_observation(terminal['legal_cam'], y_legal)
    check_point_record(latest, rows[-1], terminal, obs, 'latest')
    module.update_ema(obs)
    check_ema(module, latest['last_eval_ema'], obs)
    module.update_risk(terminal['legal_off_margin'])
    check_risk(module, latest['last_eval_risk'], terminal['legal_off_margin'])
    assert h.sha(module.ema_state()) == rows[-1]['ema_after']
    assert h.sha(module.risk_state()) == rows[-1]['risk_after']
    assert r.digests(r.bundle(state, module, opt)) == latest['digests']
    # A new process reconstructs the next stochastic update, evaluation, and
    # EMA from the serialized terminal full state. The archived witness was
    # computed by the trainer before exiting, with no checkpoint re-read.
    replay = r.witness(state, module, opt, group)
    # JSON encodes tuples used by the dropout call observer as arrays.
    replay = json.loads(json.dumps(replay))
    assert replay == task['next_step'], 'Latest next-update/evaluation/EMA witness differs'
    assert all(x['classification_norm'] > 0 for x in replay['step']['shared'].values())
    assert all(x['classification'] > 0 for x in replay['step']['added'].values())
    result = dict(group=group, fold=fold, pass_=True, best_pass=True, latest_pass=True, best_epoch=task['best_epoch'],
                  end_epoch=task['end_epoch'], metrics=interventions['test_metrics']['full'],
                  best_optimizers=best_optimizers, latest_optimizers=latest_optimizers, timeline=timeline,
                  baseline=base_metrics, aligned_b0=aligned, latest_basis=curve, preflight=True, resumes=task['resumes'],
                  moment_norms=dict(initial=initial_moments, best=best_moments, latest=terminal_moments),
                  checkpoint_sha=task['best_sha'], latest_checkpoint_sha=task['latest_sha'],
                  audit_code_sha=h.sha_file(r.HERE/'audit.py'), elapsed_seconds=time.monotonic()-start, **interventions)
    h.atomic_json(BASE_DIR/f'{group}_f{fold:02d}.json', result)
    print(f'AUDITED {group} f{fold:02d} Best and Latest', flush=True)


def changes(records, source, target):
    names = ('asd', 'td')
    result = {name: {'corrected': 0, 'harmed': 0, 'net': 0} for name in names}
    ids = {'corrected': [], 'harmed': []}
    for row in records:
        left, right, y = row[source+'_label'], row[target+'_label'], row['y']
        key = 'corrected' if left != y and right == y else 'harmed' if left == y and right != y else None
        if key:
            result[names[y]][key] += 1
            ids[key].append(row['row_1based'])
    for name in names:
        result[name]['net'] = result[name]['corrected']-result[name]['harmed']
    result['all'] = {key: sum(result[name][key] for name in names) for key in ('corrected','harmed','net')}
    result['ids'] = ids
    return result


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return dict(n=0, mean=None, sd=None, p95=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std()), p95=float(np.quantile(a, 0.95)), max=float(a.max()))


def cohort_description(records):
    direction_fields = ('relative_residual', 'raw_relative_residual', 'alignment', 'j_cover_u0', 'j_cover_u', 'cancellation', 'full_cov_llr', 'credibility')
    values = {key: distribution([x[key] for x in records]) for key in direction_fields}
    margins = {tag: distribution([x[to+'_margin']-x[fr+'_margin'] for x in records]) for tag,fr,to in
               (('A','b0','off'), ('B','off','full'), ('C','b0','full'))}
    n = len(records)
    return dict(n=n, residual_nonzero=sum(x['relative_residual'] > 0 for x in records),
                raw_limit_count=sum(x['raw_relative_residual'] > CFG['residual_limit'] for x in records),
                margin=margins, positive_margin_share=(sum(x['full_margin'] > x['off_margin'] for x in records)/n if n else None),
                probability_change=distribution([abs(x['full_asd']-x['off_asd']) for x in records]),
                gap_change=distribution([abs(x['full_margin']-x['off_margin']) for x in records]),
                conflicting_direction_fraction=(sum(x['conflicting_directions'] for x in records)/(16*n) if n else None),
                llr_overrides=sum(x['llr_overrides'] for x in records),
                reversed_gate_directions=sum(x['negative_gate_directions'] for x in records),
                **values)


def group_mechanisms(audits):
    records = sorted((row for a in audits for row in a['records'] if not row['protected']), key=lambda row:row['row_1based'])
    assert len(records) == 869
    hp = CFG['patients']
    strata = {
        'all': records,
        'darg_correct': [x for x in records if x['b0_label'] == x['y']],
        'darg_wrong': [x for x in records if x['b0_label'] != x['y']],
        'low_b0_margin': [x for x in records if abs(x['b0_margin']) < hp['low_b0_true_margin_abs_lt']],
        'high_confidence_b0_wrong': [x for x in records if x['b0_label'] != x['y'] and x['b0_margin'] <= hp['high_confidence_b0_wrong_margin_le']],
        'conflicting_evidence': [x for x in records if x['conflicting_directions']/16 > hp['conflict_fraction_gt']],
        'darg_wrong_asd': [x for x in records if x['y']==0 and x['b0_label'] != x['y']],
        'darg_wrong_td': [x for x in records if x['y']==1 and x['b0_label'] != x['y']],
        'asd': [x for x in records if x['y'] == 0],
        'td': [x for x in records if x['y'] == 1],
    }
    direct = {}
    for tag, first, second in (('A','b0','off'), ('B','off','full'), ('C','b0','full')):
        direct[tag] = changes(records, first, second)
    categories = {'new_residual_correction': [], 'backbone_harm': [], 'residual_rescue': [], 'new_residual_harm': []}
    for x in records:
        b, o, f = (x[label+'_label'] == x['y'] for label in ('b0','off','full'))
        row = x['row_1based']
        if not b and not o and f:
            categories['new_residual_correction'].append(row)
        if b and not o:
            categories['backbone_harm'].append(row)
        if b and not o and f:
            categories['residual_rescue'].append(row)
        if b and o and not f:
            categories['new_residual_harm'].append(row)
    ablations = {}
    for mode in ('residual_off', 'spectral_guidance_off', 'semantic_evidence_off'):
        label = 'off' if mode == 'residual_off' else mode
        ablations[mode] = dict(
            change=changes(records, label, 'full'),
            flips=sum(x[label+'_label'] != x['full_label'] for x in records),
            probability_absolute_change=distribution([abs(x[label+'_asd']-x['full_asd']) for x in records]),
            metrics=summarize([a['mechanism_test_metrics'][mode] for a in audits]))
    return dict(strata={k:cohort_description(v) for k,v in strata.items()}, stages=direct,
                attribution=categories, ablations=ablations,
                basis={key: distribution([a['basis'][key] for a in audits]) for key in
                       ('projector_change_frobenius', 'basis_change_frobenius', 'gram_max_abs', 'gram_condition',
                        'max_angle_deg', 'mean_angle_deg', 'rank', 'offset_norm')},
                objectives={basis: {key: distribution([a['objective'][basis][key] for a in audits]) for key in
                                    ('bhattacharyya', 'bhattacharyya_mean', 'bhattacharyya_covariance',
                                     'error_gradient_coverage', 'variance_effective_rank', 'eval_covariance_rank', 'condition')}
                            for basis in ('U0','learned')})


def summarize(values):
    matrix = np.asarray([[v[k] for k in AUC_KEYS] for v in values], dtype=float)
    return {key: dict(mean_pct=float(matrix[:, i].mean()*100), sd_pct=float(matrix[:, i].std()*100))
            for i, key in enumerate(AUC_KEYS)}


def pair_statistics(records, per_fold, source, candidate):
    folds = sorted({x['fold'] for x in records})
    source_label = 'b0' if source == 'B0' else source
    target_label = candidate
    per_patient = changes(records, source_label, target_label)
    deltas = {key: [100*(per_fold[(candidate, f)][key]-per_fold[(source, f)][key]) for f in folds]
              for key in AUC_KEYS}

    def wtl(values):
        return dict(win=sum(x > 0 for x in values), tie=sum(x == 0 for x in values), loss=sum(x < 0 for x in values))

    return dict(folds=folds, n=len(records), delta_pp={key: float(np.mean(values)) for key,values in deltas.items()},
                fold_deltas=[dict(fold=f, **{key: deltas[key][i] for key in AUC_KEYS})
                             for i,f in enumerate(folds)],
                wtl={key:wtl(deltas[key]) for key in ('acc','hard_auc')}, changes=per_patient)


def aggregate(audits):
    groups = [group for group in r.GROUPS if sum(a['group'] == group for a in audits) == 10]
    if not groups:
        return {}
    parts = {key: sorted([a for a in audits if a['group'] == key], key=lambda a:a['fold']) for key in groups}
    history = {a['fold']: a['baseline'] for a in parts[groups[0]]}
    for a in audits:
        assert a['baseline'] == history[a['fold']]
    per_fold = {('B0', f): x['metrics'] for f,x in history.items()}
    per_fold.update({(group,a['fold']):a['metrics'] for group in groups for a in parts[group]})
    group_metrics = {}
    for name in ('B0', *groups):
        group_metrics[name] = dict(full=summarize([per_fold[(name,f)] for f in r.FOLDS]),
                                   no_fold2=summarize([per_fold[(name,f)] for f in r.FOLDS if f != 2]))
    combined = {}
    for group in groups:
        entries = sorted((row for a in parts[group] for row in a['records']), key=lambda x:x['row_1based'])
        assert [row['row_1based'] for row in entries] == list(range(1, CFG['subjects']+1))
        combined[group] = entries
    mechanisms = {key:group_mechanisms(value) for key, value in parts.items()}
    links = [('B0', group) for group in groups]
    if len(groups) == 2:
        links.append(('G1','G2'))
    paired = {}
    for source,candidate in links:
        evidence = [dict(row, **{candidate+'_label':row['full_label']}) for row in combined[candidate]]
        if source != 'B0':
            for source_row, target_row in zip(combined[source], evidence):
                assert all(source_row[key] == target_row[key] for key in ('row_1based','fold','y','b0_label','b0_asd'))
                target_row[source+'_label'] = source_row['full_label']
        full = pair_statistics(evidence, per_fold, source, candidate)
        fold2 = pair_statistics([x for x in evidence if x['fold'] != 2], per_fold, source, candidate)
        fold_patients = {f: changes([x for x in evidence if x['fold']==f], 'b0' if source=='B0' else source, candidate)
                         for f in r.FOLDS}
        winners = {
            'acc': max(full['fold_deltas'], key=lambda v:v['acc'])['fold'],
            'hard_auc': max(full['fold_deltas'], key=lambda v:v['hard_auc'])['fold'],
            'net': max(r.FOLDS, key=lambda f:fold_patients[f]['all']['net']),
        }
        trimmed = {k: dict(removed=f, **pair_statistics([x for x in evidence if x['fold'] != f], per_fold, source, candidate))
                   for k, f in winners.items()}
        for row in full['fold_deltas']:
            row['changes'] = fold_patients[row['fold']]
        paired[f'{candidate}-{source}'] = dict(full=full, no_fold2=fold2, max_contribution_removed=trimmed)
    return dict(group_metrics=group_metrics, paired=paired, mechanisms=mechanisms)


def decide(audits, combined, protection):
    group_decisions = {}
    limits = CFG['decision']
    for group, mech in combined.get('mechanisms', {}).items():
        a = [row for row in audits if row['group'] == group]
        paired = combined['paired'][f'{group}-B0']
        full = paired['full']
        delta = full['delta_pp']
        n_best = sum(row['best_pass'] for row in a)
        n_latest = sum(row['latest_pass'] for row in a)
        audited = len(a) == n_best == n_latest == 10 and protection['pass_']
        checks = dict(audited=audited, accuracy=delta['acc']>0, balanced_accuracy=delta['hard_auc']>=0,
                      official_net=full['changes']['all']['net']>0,
                      auroc_tolerance=delta['prob_auc']>=limits['paired_prob_auc_min_pp'],
                      asd_sensitivity_tolerance=delta['sen']>=limits['paired_sen_min_pp'])

        def positive(scenario):
            return scenario['delta_pp']['acc'] > 0 and scenario['changes']['all']['net'] > 0

        checks['no_fold2'] = positive(paired['no_fold2'])
        checks['largest_fold'] = all(positive(paired['max_contribution_removed'][key]) for key in ('acc','net'))
        mechanisms = dict(audited=audited, direct_net=mech['stages']['B']['all']['net']>0,
                          moving_subspace=all(row['basis']['projector_change_frobenius']>TOL['projector_change_min'] for row in a),
                          basis_rank=all(row['basis']['rank']==16 and row['objective']['learned']['eval_covariance_rank']==16 for row in a),
                          real_end_to_end=all(all(x['positive_class']>0 for x in row['timeline']['added_gradients'].values())
                                              and all(x['positive_applied']>0 for x in row['timeline']['shared_gradients'].values())
                                              for row in a),
                          current_llr=all(row['formulas']['full']['llr_attribution_max_error'] < 1e-8 for row in a),
                          residual_changes_classifier=mech['strata']['all']['residual_nonzero']>0 and
                          mech['strata']['all']['probability_change']['mean']>0,
                          budget_caps=all(max(g['maximum_ratio'] for g in row['timeline']['added_gradients'].values())<=
                                          CFG['loss']['aux_ratio_max'] for row in a),
                          long_risk_exercised=all(any(x['positive_risk_dot']>0 for x in row['timeline']['shared_gradients'].values())
                                                  for row in a))
        group_decisions[group] = dict(performance_go=all(checks.values()), mechanism_go=all(mechanisms.values()),
                                      performance_checks=checks, mechanism_checks=mechanisms)
    verdict = 'GO' if any(x['performance_go'] and x['mechanism_go'] for x in group_decisions.values()) else 'NO-GO'
    return dict(verdict=verdict, groups=group_decisions)


def cached_audits():
    audits = []
    audit_sha = h.sha_file(r.HERE/'audit.py')
    for group in r.GROUPS:
        for fold in r.FOLDS:
            path = BASE_DIR/f'{group}_f{fold:02d}.json'
            if not path.is_file():
                continue
            row = h.read_json(path)
            if row.get('audit_code_sha') != audit_sha:
                continue
            task, best, latest, _ = h.paths(group, fold)
            status = h.read_json(task)
            assert status['implementation'] == r.implementation()
            assert row['checkpoint_sha'] == status['best_sha'] == r.sha_file(best)
            assert row['latest_checkpoint_sha'] == status['latest_sha'] == r.sha_file(latest)
            audits.append(row)
    return audits


def audit_group(group, workers):
    r.accepted_preflight()
    gpu = r.gpu_sample()
    pre = r.accepted_preflight()
    memory = max(x['peak_worker_reserved_bytes'] for x in pre['folds'])*1.5/2**20
    available = gpu['total_mib']-gpu['used_mib']-CFG['gpu']['gpu_free_reserve_mib']
    assert 1 <= workers <= CFG['gpu']['workers_max'] and available >= memory
    workers = min(workers, int(available/memory), max(1,len(os.sched_getaffinity(0))//2))
    start = time.monotonic()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    done = {a['fold'] for a in cached_audits() if a['group'] == group}
    pending = [f for f in r.FOLDS if r.task_complete(group, f) and f not in done]
    assert len(done)+len(pending) == 10

    def run_one(fold):
        target = BASE_DIR/f'{group}_f{fold:02d}.log'
        with target.open('a', encoding='utf-8') as out:
            result = subprocess.run([sys.executable, '-B', str(r.HERE/'audit.py'), '--group',group,'--fold',str(fold)],
                                    cwd=r.HERE, stdout=out, stderr=subprocess.STDOUT, check=False)
        print(f'AUDIT {group} f{fold:02d} return={result.returncode}', flush=True)
        return dict(fold=fold, code=result.returncode, log=str(target))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        process = list(pool.map(run_one, pending))
    audits = cached_audits()
    partial = h.read_json(r.RESULTS/'audit_status.json') if (r.RESULTS/'audit_status.json').is_file() else {}
    audit_wall = partial.get('audit_wall_seconds_per_group', {})
    failures = dict(partial.get('failures', {}), **{group:[x for x in process if x['code']]})
    protected = r.protection() if len(audits)==20 else pre['protection_before']
    aggregated = aggregate(audits) if any(sum(x['group']==g for x in audits)==10 for g in r.GROUPS) else {}
    if pending:
        audit_wall[group] = audit_wall.get(group, 0.)+time.monotonic()-start
    source_by_group = dict(partial.get('audit_source_by_group', {}))
    if not source_by_group and partial.get('audit_source_sha256'):
        for key in partial.get('audit_wall_seconds_per_group', {}):
            source_by_group[key] = partial['audit_source_sha256']
    source_by_group[group] = h.sha_file(r.HERE/'audit.py')
    status = dict(status='COMPLETE_AUDITED' if len(audits)==20 and protected['pass_'] and not any(failures.values()) else 'PARTIAL',
                  audits=[{k:v for k,v in x.items() if k!='records'} for x in audits], failures=failures,
                  implementation=r.implementation(), audit_source_sha256=h.sha_file(r.HERE/'audit.py'), audit_source_by_group=source_by_group,
                  preflight_pass=pre['pass_'], best_pass=sum(x['best_pass'] for x in audits),
                  latest_pass=sum(x['latest_pass'] for x in audits), protection=protected,
                  audit_wall_seconds_per_group=audit_wall, audit_workers=workers, aggregate=aggregated,
                  science=decide(audits, aggregated, protected))
    h.atomic_json(r.RESULTS/'audit_status.json', status)
    if status['status'] == 'COMPLETE_AUDITED':
        import report
        report.publish(status, audits)
    assert not failures[group], failures[group]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=r.GROUPS)
    parser.add_argument('--fold', type=int, choices=r.FOLDS)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    if args.fold:
        audit_fold(args.group, args.fold)
    else:
        audit_group(args.group, args.workers)


if __name__ == '__main__':
    main()
