#!/usr/bin/env python3
"""Independent PS-CHC equations, patient pairing, and checkpoint audit."""
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
from ps_chc import EMA_KEYS, FOLD2_ROWS, INTERVENTIONS

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



def explicit_fp32_head(head, z):
    # Reconstruct the native inference arithmetic without calling OutputLayer.
    norm, affine, gelu, _ = head.mlp_head
    normalized, _, _ = torch.native_layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    activated = F.gelu(F.linear(normalized, affine.weight, affine.bias), approximate=gelu.approximate)
    return F.linear(activated, head.classifier.weight, head.classifier.bias)


def counterfactual_shift(head, z, candidates):
    stacks = (z[:, None]+candidates).reshape(-1,144)
    two = explicit_fp32_head(head,stacks).reshape(len(z),2,2)
    off = explicit_fp32_head(head,z)
    return (two[...,0]-two[...,1])-(off[:,0]-off[:,1])[:,None]


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
    pooled = hp['own_variance']*covariance + hp['pooled_variance']*pooled_total
    projected_cov = np.array([u.T @ c @ u for c in pooled])
    regularization = ridge+np.maximum(0, np.trace(projected_cov,axis1=1,axis2=2))/(hp['max_condition']-1)
    covs = (projected_cov+projected_cov.transpose(0,2,1))*0.5+regularization[:,None,None]*np.eye(16)
    posdef, logdet = np.linalg.slogdet(covs)
    eigenvalues = np.linalg.eigvalsh(covs)
    assert (posdef == 1).all() and (eigenvalues > 0).all()
    assert (eigenvalues[:, -1]/eigenvalues[:, 0] <= hp['max_condition']).all()
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
        projected_total=u.T@pooled_total@u, projected_origin=array(module.U0).T@pooled_total@array(module.U0),
        dispersion=eigenvalues.mean(1)*(1/eigenvalues).mean(1), stability=stable.mean(0), raw_llr=attribution, llr_total=total_llr,
        loglam=normalize(np.log(raw_var)), nu=normalize(np.log1p(contrast))).items()}


def independent_distance(d):
    mean, cov = array(d['class_mean']), array(d['class_cov'])
    mid = np.mean(cov, axis=0)
    dm = mean[0]-mean[1]
    location = float(dm@np.linalg.solve(mid, dm)/8)
    volume = float((np.linalg.slogdet(mid)[1]-np.linalg.slogdet(cov)[1].mean())/2)
    return location+max(0.,volume), location, max(0.,volume)


@torch.no_grad()
def manual_forward(module, z, jac, gap, intervention, candidate_effect=None, native_choice=None):
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
        responses = [torch.zeros_like(x) for x in responses]
    n0 = ns[:, 0, None].expand_as(s)
    n1 = ns[:, 1, None].expand_as(s)
    availability = 1-(n0+n1)/2
    sem, stat = s.abs()*availability, l.abs()*stable
    hp = CFG['fusion']
    confident = 1-hp['conflict_attenuation']*availability*torch.maximum(-s*l, torch.zeros_like(s))
    mix_x = torch.stack((s, l, sem, stat, s*l, n0, n1, rel), dim=2)
    blend = (hp['alpha_max']-hp['alpha_min'])*torch.sigmoid(module.fusion_net(mix_x)[..., 0])+hp['alpha_min']
    ws, wl = blend*sem, (1-blend)*stat
    difference = confident*rel*(ws*s+wl*l)
    confidence = torch.tanh(0.5*gap[:, None])
    stat_gap = torch.tanh(full['llr_total'][:, None]/(4*module.tau_l)).float()
    c = confidence.broadcast_to(s.shape)
    p = stat_gap.broadcast_to(s.shape)
    info = torch.stack((c, c.abs(), s, l, s*l, n0, n1, stable, p), dim=2)
    credibility = torch.sigmoid(module.credibility(info)[..., 0])
    j = jac.double() @ basis
    norm_sq = torch.sum(j*j, dim=1, keepdim=True)+CFG['jacobian_epsilon']**2
    unit = (j/norm_sq.sqrt()).float()

    def limited(raw, features):
        rad = raw.double().norm(dim=-1, keepdim=True)
        z_norm = features.double().norm(dim=-1, keepdim=True)
        cap = CFG['residual_limit']*z_norm.clamp_min(EPS)
        radius = cap*torch.tanh(rad/cap)
        scale = torch.where(rad == 0, torch.ones_like(rad), radius/rad.clamp_min(1e-300))
        delta = (raw.double()*scale*(z_norm/z_norm.clamp_min(EPS))*
                 (z_norm/(z_norm+torch.finfo(z.dtype).tiny))).float()
        if module.fold == 2:
            delta[list(FOLD2_ROWS)] = 0
        return delta

    extra = {}
    if module.group == 'G1':
        gate_input = torch.cat((F.layer_norm(z, (144,)), difference, rel, ns, confidence, stat_gap), dim=1)
        gate = credibility * torch.tanh(module.gate_net(gate_input))
        theta = gate*difference
        raw = (theta*unit)@basis.float().T
        delta = limited(raw,z)
    else:
        hparams = CFG['counterfactual']
        pairs, raw_pairs, coordinate, gates = [], [], [], []
        n = len(z)
        for disease, sign in enumerate((1, -1)):
            x = torch.stack((sign*s, sign*l, sign*difference, sign*c, (16*responses[disease]).tanh(),
                             rel, stable, ns[:, disease:disease+1].expand_as(s), sign*p), dim=2)
            signed_amplitude = torch.tanh(module.candidate_net[disease](x)[..., 0])
            amp = torch.maximum(signed_amplitude, torch.zeros_like(signed_amplitude))
            signal = ws*torch.maximum(sign*s, torch.zeros_like(s))+wl*torch.maximum(sign*l, torch.zeros_like(l))
            coord = sign*amp*hparams['candidate_scale']*credibility*rel*confident*signal
            raw_c = (coord*unit) @ basis.float().T
            raw_pairs.append(raw_c)
            pairs.append(limited(raw_c,z))
            coordinate.append(coord)
            gates.append(amp)
        candidates = torch.stack(pairs, 1)
        # Verify candidate features separately. Use independently audited FP32
        # candidate effects for routing so roundoff near canceled logits need
        # not be absorbed by a looser feature tolerance.
        shift = candidate_effect
        assert shift is not None
        route_x = torch.column_stack((confidence, stat_gap, s.mean(-1), l.mean(-1), (s*l).mean(-1),
                                      ns, rel.mean(-1), torch.full_like(gap,stable[0].mean()),
                                      difference.mean(-1), (shift/2).tanh()))
        forecast = (0.5 + torch.sigmoid(module.calibrator(route_x)))*shift
        features = torch.column_stack((route_x, torch.tanh(forecast/2)))
        levels = module.router(features)/hparams['selection_temperature']
        route = levels.softmax(1)

        def select(arms):
            stack = torch.stack((torch.zeros_like(arms[0]), *arms),1)
            return stack[torch.arange(n,device=z.device),native_choice]

        delta, raw, theta = (select(arms) for arms in (pairs,raw_pairs,coordinate))
        gate = (gates[0]+gates[1])/2
        soft = route[:,1:2]*pairs[0]+route[:,2:3]*pairs[1]
        total_magnitude = (route[:,1:] * candidates.double().norm(dim=-1)).sum(1)
        cancellation = torch.where(total_magnitude > EPS, (
            1-soft.double().norm(dim=-1)/total_magnitude.clamp_min(EPS)).clamp(0,1),0.)
        extra = dict(candidates=candidates, candidate_theta=torch.stack(coordinate,1), route=route,
                     candidate_gap_effect=shift, calibration_inputs=route_x, calibrated_gap=forecast,
                     route_scores=levels, soft_residual=soft, soft_cancellation=cancellation)
    if intervention == 'residual_off':
        delta.zero_()
    first_order = (theta.double()*j.square()).sum(1)/norm_sq[:, 0]
    return dict(**{key:full[key] for key in ('projected','projected_total','projected_origin','dispersion','class_mean','class_cov','class_logdet','raw_llr','llr_total')},
                basis=basis, hh=hh, attention=attention, llr=limit*llr_token,
                signed_semantic=s, signed_statistical=l, active_null=ns, reliability=rel, alpha=blend,
                difference=difference, gate=gate, theta=theta, first_order_score=first_order,
                raw=raw, delta=delta, enhanced=z+delta, jacobian_projected=j, credibility=credibility,
                w_sem=ws, w_stat=wl, evidence_agreement=confident, **extra)


@torch.no_grad()
def check_formulas(state, module, z, intervention='full'):
    from ps_chc import head_sensitivity, gaussian_separation

    before = r.rng_mark()
    head = state['cam'].Outputlayer
    j, gap = head_sensitivity(head, z)
    analytic, analytic_logits = analytic_jacobian(head, z)
    compare(j, analytic)
    native_logits = native_head_logits(head, z)
    assert torch.equal(native_logits, explicit_fp32_head(head,z))
    native_gap = native_logits[:, 0] - native_logits[:, 1]
    assert torch.equal(gap, native_gap)
    actual = module(z, j, gap, head, intervention)
    candidate_effect, choice = None, None
    if module.group == 'G2':
        candidate_effect = counterfactual_shift(head,z,actual['candidates'])
        assert torch.equal(candidate_effect,actual['candidate_gap_effect'])
        choice = actual['route_scores'].argmax(1)
        assert torch.equal(choice,actual['route_choice'])
        signed = actual['candidate_theta']
        assert (signed[:,0]>=0).all() and (signed[:,1]<=0).all()
        if intervention != 'residual_off':
            actions = torch.cat((torch.zeros_like(actual['candidates'][:, :1]),actual['candidates']),1)
            assert torch.equal(actual['delta'],actions[torch.arange(len(z),device=z.device),choice])
    expected = manual_forward(module,z,j,native_gap,intervention,candidate_effect,choice)
    if choice is not None:
        # The route is compared unconditionally. Verify the independent rank
        # where its top-two separation exceeds measured equation roundoff.
        error = (actual['route_scores']-expected['route_scores']).abs().max(1).values
        top = expected['route_scores'].topk(2,dim=1).values
        certified = top[:,0]-top[:,1]>2*error
        assert torch.equal(choice[certified],expected['route_scores'][certified].argmax(1))
    np.testing.assert_allclose(array(actual['raw_llr'].sum(-1)), array(actual['llr_total']), atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose([float(v) for v in gaussian_separation(actual)], independent_distance(expected),
                               atol=TOL['audit_fp64_atol'], rtol=1e-10)
    for key in expected:
        try:
            if key in ('projected','projected_total','projected_origin','dispersion','class_mean','class_cov','class_logdet','raw_llr','llr_total'):
                torch.testing.assert_close(actual[key], expected[key], atol=TOL['audit_fp64_atol'], rtol=1e-10)
            else:
                compare(actual[key], expected[key])
        except AssertionError as exc:
            raise AssertionError((key, intervention, str(exc))) from exc
        assert torch.isfinite(actual[key]).all(), key
    assert float((actual['attention'].sum(-1)-1).abs().max()) < 2e-6
    relative = actual['delta'].double().norm(dim=-1) / z.double().norm(dim=-1).clamp_min(1e-300)
    assert float(relative.max()) <= CFG['residual_limit']+TOL['residual_bound_roundoff']
    if module.group == 'G2':
        c = actual['candidates'].double().norm(dim=-1)/z.double().norm(dim=1).clamp_min(1e-300)[:,None]
        assert float(c.max()) <= CFG['residual_limit']+TOL['residual_bound_roundoff']
        assert torch.allclose(actual['route'].sum(1), torch.ones(len(z),device=z.device), rtol=0.,atol=2e-7)
        if state['fold'] == 2:
            assert not actual['candidates'][list(FOLD2_ROWS)].count_nonzero()
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


def check_optimizers(state, module, opt, epoch):
    result = {}
    for key, optimizer in zip((*r.HIST_OPTS, 'opt_new'), [state[k] for k in r.HIST_OPTS]+([opt] if opt else [])):
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
    if module is not None:
        r.parameter_layout(state, module, opt)
    return result


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
    cov = [hp['own_variance']*cc+hp['pooled_variance']*total for cc in cov]
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
        covs = [basis.T @ cc @ basis for cc in cov]
        c0, c1 = [(cc+cc.T)*0.5+(ridge+max(0,np.trace(cc))/(hp['max_condition']-1))*np.eye(16)
                  for cc in covs]
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
                    participation_rank=float(positive.sum()**2/max(1e-20,(positive**2).sum())),
                    directional_variance_min=float(np.min(np.diag(covariance))), condition=float(shape_condition))
    return dict(U0=score(origin), learned=score(chosen), legal_ids_sha=h.sha(ids),
                n_legal=len(ids), deterministic_eval_gradient=True)


@torch.no_grad()
def best_interventions(state, module, best, source):
    from ps_chc import head_sensitivity, gaussian_separation

    before = r.rng_mark()
    logp_off, z = original_eval(state)
    head = state['cam'].Outputlayer
    j, head_gap = head_sensitivity(head, z)
    ids = state['test_np']
    modes = {}
    assertions = {}
    for mode in INTERVENTIONS:
        assertions[mode] = check_formulas(state, module, z, mode)
        d = module(z, j, head_gap, head, mode)
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
        evidence_free = module(z, j, head_gap, head, 'semantic_evidence_off')
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
    routes = array(d['route']) if module.group == 'G2' else None
    candidate_effect = array(d['candidate_gap_effect']) if module.group == 'G2' else None
    choice = array(d['route_choice']) if routes is not None else None
    calibrated = array(d['calibrated_gap']) if routes is not None else None
    if choice is not None:
        chosen_gap = np.column_stack((np.zeros(len(z)),candidate_effect))[np.arange(len(z)),choice]
        current_logits = explicit_fp32_head(head,d['enhanced'])
        original_logits = explicit_fp32_head(head,z)
        actual_gap = array((current_logits[:,0]-current_logits[:,1])-(original_logits[:,0]-original_logits[:,1]))
        np.testing.assert_allclose(chosen_gap,actual_gap,rtol=TOL['audit_feature_rtol'],atol=TOL['audit_feature_atol'])
        chosen_forecast = np.column_stack((np.zeros(len(z)),calibrated))[np.arange(len(z)),choice]
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
                    full_cov_llr=float(gaussian[patient_id]), credibility=float(cmax[patient_id]),
                    null_asd=float(d['active_null'][patient_id,0]), null_td=float(d['active_null'][patient_id,1]),
                    semantic_contribution=float((d['w_sem'][patient_id]*d['signed_semantic'][patient_id]).abs().mean()),
                    statistical_contribution=float((d['w_stat'][patient_id]*d['signed_statistical'][patient_id]).abs().mean()),
                    evidence_agreement=float(d['evidence_agreement'][patient_id].mean()),
                    asd_semantic_support=float(d['responses'][patient_id,0].sum()),
                    td_semantic_support=float(d['responses'][patient_id,1].sum()),
                    route_keep=float(routes[patient_id,0]) if routes is not None else None,
                    route_asd=float(routes[patient_id,1]) if routes is not None else None,
                    route_td=float(routes[patient_id,2]) if routes is not None else None,
                    selected_route=int(choice[patient_id]) if choice is not None else None,
                    soft_cancellation=float(d['soft_cancellation'][patient_id]) if choice is not None else None,
                    selected_gap=float(actual_gap[patient_id]) if choice is not None else None,
                    selected_forecast_gap=float(chosen_forecast[patient_id]) if choice is not None else None,
                    asd_candidate_gap=float(candidate_effect[patient_id,0]) if candidate_effect is not None else None,
                    td_candidate_gap=float(candidate_effect[patient_id,1]) if candidate_effect is not None else None)
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
    direction_fields = ('relative_residual', 'raw_relative_residual', 'alignment', 'j_cover_u0', 'j_cover_u', 'cancellation',
                        'full_cov_llr', 'credibility', 'null_asd', 'null_td', 'semantic_contribution',
                        'statistical_contribution', 'evidence_agreement', 'asd_semantic_support', 'td_semantic_support')
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
    categories = {'new_residual_correction': [], 'backbone_harm': [], 'residual_rescue': [], 'new_residual_harm': [], 'remaining_errors': []}
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
        if not f:
            categories['remaining_errors'].append(row)
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
                                     'error_gradient_coverage', 'variance_effective_rank', 'participation_rank', 'eval_covariance_rank', 'condition')}
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




@torch.no_grad()
def audit_formula(state, module):
    module.eval()
    _, z = original_eval(state)
    return dict(geometry=basis_diagnostics(module), **check_formulas(state, module, z))


@torch.no_grad()
def independent_optimize(state, probe, proposal):
    """Exercise genuine optimizer.step on cloned history, independent of virtual_step."""
    positions, states = {}, {}
    for key in r.HIST_OPTS:
        instance = state[key]
        values = {p: g for name, ps in state['old_layout'].items() if name.startswith(key+'.')
                  for p,g in zip(ps, probe['shared'][name][proposal+'_gradient'])}
        all_params = {id(p): p for p in (*state['model'].parameters(), *state['cam'].parameters())}
        old_op = probe['pre_step'][key]
        source_shapes = {p: torch.empty_like(p) for pg in instance.param_groups for p in pg['params']}
        named = {p:(branch,n) for branch in ('model','cam') for n,p in state[branch].named_parameters()}
        assert all(id(p) in all_params for p in source_shapes)
        copies = {p:torch.nn.Parameter(probe['pre_step'][named[p][0]][named[p][1]].to(p).clone(),
                                      requires_grad=p.requires_grad) for p in source_shapes}
        groups = [{**pg,'params':[copies[p] for p in pg['params']]} for pg in instance.param_groups]
        optimizer = type(instance)(groups)
        import copy
        optimizer.load_state_dict(copy.deepcopy(old_op))
        for p, clone in copies.items():
            g = values.get(p)
            clone.grad = g.clone() if g is not None else None
        optimizer.step()
        positions.update({p:v.detach() for p,v in copies.items()})
        states.update({p:optimizer.state.get(v,{}) for p,v in copies.items()})
    return positions, states


def independent_gradients(state, module, probe):
    """Two isolated native forwards used only by audit; the recorded training path used one."""
    import copy
    saved = r.native_bundle(state)
    reconstructed = copy.deepcopy(module)
    reconstructed.load_state_dict(probe['pre_module'], strict=True)
    h.runtime().load_history().load_bundle(state, probe['pre_step'])
    params = r.parameters_of(state['old_layout'])
    old_rng = r.rng_mark()
    try:
        for mode in ('off','full'):
            head = state['cam'].Outputlayer
            handle = None
            if mode == 'full':
                def augment(_head, args):
                    from ps_chc import head_sensitivity
                    j, gap = head_sensitivity(head, args[0])
                    return (reconstructed(args[0], j, gap, head)['enhanced'], *args[1:])
                handle = head.register_forward_pre_hook(augment)
            try:
                with r.use_stream(probe['pre_rng']):
                    output = h.runtime().forward(state, r.BASE_GROUP, training=True)
                    assert torch.equal(output['logp'], probe[{'off':'off','full':'on'}[mode]])
                    gradients = torch.autograd.grad(output['total'], params, allow_unused=True)
            finally:
                if handle is not None:
                    handle.remove()
            for name, ps, grads, _ in r.split_gradients(state['old_layout'], gradients):
                target = probe['shared'][name][mode+'_gradient']
                for value, want in zip(grads,target):
                    assert (value is None) == (want is None)
                    if value is not None:
                        compare(value,want)
    finally:
        h.runtime().load_history().load_bundle(state, saved)
    assert r.rng_mark() == old_rng


def independent_trust(name, block):
    hp = CFG['trust']
    initial, old, proposal, ref, ref_old, accepted = (
        array(block[k]).astype(np.float64) for k in ('origin','original','proposal','ref','ref_old','accepted'))
    width = math.sqrt(len(old))
    pieces = name.split('.')
    tag = '.'.join((pieces[0], pieces[-1])) if len(pieces)==3 else pieces[0]
    r0 = hp['relative_parameter_to_init'][tag]*max(np.linalg.norm(initial),width*hp['parameter_scale_floor'])
    raw = np.linalg.norm(proposal-ref)

    def scale(raw,radius):
        return min(1.,hp['fp32_inward_factor']*radius/raw) if raw else 1.

    def mapped(center,value,alpha,dtype):
        return ((1-alpha)*center+alpha*value).astype(dtype).astype(np.float64)

    dtype = array(block['proposal']).dtype
    report = block['report']
    alpha = scale(raw,r0)
    np.testing.assert_allclose(alpha,report['proposal_scale'],rtol=1e-12,atol=1e-14)
    # The device reduction and NumPy pairwise sum need not finish at the
    # same FP64 bit. Certify the scalar with the native reduction as well,
    # then independently reconstruct the elementwise operation and FP32 cast.
    radius_device = hp['relative_parameter_to_init'][tag]*max(
        float(torch.linalg.vector_norm(block['origin'])),width*hp['parameter_scale_floor'])
    raw_device = float(torch.linalg.vector_norm(block['proposal'].double()-block['ref']))
    assert scale(raw_device,radius_device)==report['proposal_scale']
    expected = mapped(ref,proposal,report['proposal_scale'],dtype)
    actual_dist = np.linalg.norm(expected-ref)
    if actual_dist>r0:
        expected, actual_dist = ref, 0.
    np.testing.assert_array_equal(accepted,expected)
    assert actual_dist <= r0
    samples = dict(radius=r0, proposal_scale=alpha, proposal_reference_distance=raw,
                   accepted_reference_distance=actual_dist, actual_step=np.linalg.norm(accepted-old),
                   native_proposal_step=np.linalg.norm(proposal-old), accepted_vs_proposal=np.linalg.norm(accepted-proposal),
                   before_reference_distance=np.linalg.norm(old-ref_old), reference_step=np.linalg.norm(ref-ref_old))
    for key,value in samples.items():
        np.testing.assert_allclose(report[key],value,rtol=1e-12,atol=1e-14,err_msg=key)

    fields = ('exp_avg','exp_avg_sq')
    m, mref, mfinal = ({k:array(block[level][k]).astype(np.float64) for k in fields} for level in
                        ('proposed_moments','reference_moments','accepted_moments'))
    radii = {k:hp[f'relative_{k}']*max(np.linalg.norm(mref[k]),width*hp['moment_scale_floor']) for k in fields}
    deviations = {k:np.linalg.norm(m[k]-mref[k]) for k in fields}
    beta = min(scale(deviations[k],radii[k]) for k in fields)
    device_contractions = []
    for k in fields:
        left, right = block['proposed_moments'][k].double(), block['reference_moments'][k].double()
        allowed = hp[f'relative_{k}']*max(float(torch.linalg.vector_norm(right)),width*hp['moment_scale_floor'])
        device_contractions.append(scale(float(torch.linalg.vector_norm(left-right)),allowed))
    assert min(device_contractions)==report['moment_scale']
    predicted = {k:mapped(mref[k],m[k],report['moment_scale'],array(block['proposed_moments'][k]).dtype)
                 for k in fields}
    if any(np.linalg.norm(predicted[k]-mref[k])>radii[k] for k in fields):
        predicted = mref
    for k in fields:
        np.testing.assert_array_equal(mfinal[k],predicted[k])
        current_drift = np.linalg.norm(mfinal[k]-mref[k])
        assert current_drift <= radii[k]
        np.testing.assert_allclose([report['moments'][k][field] for field in ('drift','radius','proposal_drift')],
                                   [current_drift,radii[k],deviations[k]],rtol=1e-12,atol=1e-14)
    assert np.all(mfinal['exp_avg_sq']>=0)
    np.testing.assert_allclose(beta,report['moment_scale'],rtol=1e-12,atol=1e-14)


def audit_gains(state, probe, group):
    legal = rows_legal(state)
    y = state['labels_np'][legal]
    d = probe['details']
    left, right = probe['native_dropout']
    assert torch.equal(left*probe['native_mask'], right)
    mask = probe['native_mask']
    drop = probe['head'].mlp_head[-1]
    vals = array(mask).reshape(-1)
    assert set(np.unique(vals).tolist()) == {0., float(torch.tensor(1/(1-drop.p),dtype=mask.dtype))}
    head = probe['head']

    def head_masked(z, current_mask):
        x = head.mlp_head[2](head.mlp_head[1](head.mlp_head[0](z)))
        return head.classifier(x*current_mask)

    off = head_masked(d['z'], mask).log_softmax(-1)
    on = head_masked(d['enhanced'], mask).log_softmax(-1)
    assert torch.equal(off, probe['off']) and torch.equal(on, probe['on'])
    rows = np.arange(len(y))
    target = probe['legal']
    teacher = probe['teacher']
    rdet, rmask = (array(teacher[k]) for k in ('deterministic','masked'))
    assert not any(x.requires_grad for x in teacher.values())
    rd = (rdet[rows,y]-rdet[rows,1-y]).astype(np.float64)
    rm = (rmask[rows,y]-rmask[rows,1-y]).astype(np.float64)
    p = CFG['reference']
    reliability = (rd > 0)/(1+np.exp(np.clip((p['reliable_margin']-rd)/p['temperature'],-200,200)))

    def balance(v):
        return sum(v[y==c].mean() for c in (0,1))/2

    for score, retention in zip((off,on),(target.off_retention,target.full_retention)):
        logp = array(score[legal]).astype(np.float64)
        margin = logp[rows,y]-logp[rows,1-y]
        residual = np.maximum(0,rm-p['margin_slack']-margin)
        pp = (np.exp(rmask[:,0])-np.exp(logp[:,0]))**2
        np.testing.assert_allclose(balance(reliability*(residual**2+p['probability_weight']*pp)),
                                   float(retention.detach()),rtol=2e-5,atol=2e-8)
    m_off, m_on = (array(x[legal,y]-x[legal,1-y]).astype(np.float64) for x in (off,on))
    np.testing.assert_allclose(array(probe['aux']['margin_delta']),m_on-m_off,rtol=0,atol=1e-6)
    hp = CFG['loss']
    t, center = hp['g2_gain_temperature'], hp['g2_gain_target']
    weights = hp['g2_weight_floor']+(1-hp['g2_weight_floor'])/(1+np.exp(np.clip(rd-hp['g2_margin_center'],-200,200)))

    def phi(m):
        return t*np.logaddexp(0,(center-m)/t)

    np.testing.assert_allclose(balance(weights*(phi(m_on)-phi(m_off))),float(probe['actual_terms'][1]),
                               rtol=2e-5,atol=2e-8)
    if group == 'G2':
        z = (d['z'][legal,None]+d['candidates'][legal]).reshape(-1,144)
        shape = (len(legal)*2, mask.shape[-1])
        pair = head_masked(z, mask[legal].repeat_interleave(2,dim=0).reshape(shape)).reshape(-1,2,2)
        assert torch.equal(pair,probe['aux']['paired_logits'])
        cand = array(pair)
        margin = (cand[:,:,0]-cand[:,:,1])*(1-2*y[:,None])
        right, wrong = margin[rows,y], margin[rows,1-y]
        hp_cf = CFG['counterfactual']
        expected_loss = balance(weights*(phi(right)-phi(m_off)+
                                       hp_cf['candidate_wrong_direction_weight']*np.maximum(0,wrong-m_off)**2))
        dose = (np.linalg.norm(array(d['candidates'][legal]).astype(np.float64),axis=-1) /
                (CFG['residual_limit']*np.linalg.norm(array(d['z'][legal]).astype(np.float64),axis=-1).clip(min=1e-12))[:,None])
        gain = margin-m_off[:,None]
        utility = gain-hp_cf['intervention_cost']*dose-hp_cf['reliable_margin_penalty']*reliability[:,None]*np.maximum(0,-gain)
        logits = np.column_stack((np.zeros(len(y)),utility)).clip(-hp_cf['utility_clip'],hp_cf['utility_clip'])
        likelihood = np.exp((logits-logits.max(1,keepdims=True))/hp_cf['utility_temperature'])
        route_target = likelihood/likelihood.sum(1,keepdims=True)
        scores = array(d['route_scores'][legal]).astype(np.float64)
        logp = scores-np.logaddexp.reduce(scores,axis=1)[:,None]
        route_loss = -(route_target*logp).sum(1)
        masked = (cand[:,:,0]-cand[:,:,1])-(m_off*(1-2*y))[:,None]
        forecast = array(probe['aux']['forecast'])
        errors = np.abs(forecast-masked)
        cut = hp_cf['calibration_huber_delta']
        huber = np.where(errors < cut,0.5*errors**2,cut*(errors-0.5*cut))
        np.testing.assert_allclose(array(probe['aux']['target_route']),route_target,atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(array(probe['aux']['masked_gap_effect']),masked,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(array(d['route_choice']),array(d['route_scores']).argmax(1))
        np.testing.assert_allclose(float(probe['actual_terms'][4]),expected_loss,atol=2e-8,rtol=2e-5)
        np.testing.assert_allclose(float(probe['actual_terms'][5]),balance(weights*route_loss),atol=2e-8,rtol=2e-5)
        np.testing.assert_allclose(float(probe['actual_terms'][7]),balance(huber.mean(1)),atol=2e-8,rtol=2e-5)
        executed = np.column_stack((np.zeros(len(y)),gain))[rows,array(d['route_choice'][legal])]
        np.testing.assert_allclose(m_on-m_off,executed,atol=TOL['audit_feature_atol'],rtol=TOL['audit_feature_rtol'])
        if state['fold'] == 2:
            assert not d['candidates'][list(FOLD2_ROWS)].count_nonzero()
    return dict(legal=len(legal), mask_verified=True, matched_margin=True, retention_independent=True,
                candidate_independent=group=='G2')


def check_training_probe(state, module, probe):
    optimizer_checks = {}
    for name in ('off','full','proposal'):
        got, slots = independent_optimize(state,probe,name)
        for key, ps in state['old_layout'].items():
            witness = probe['shared'][key]
            assert torch.equal(r.flatten([got[p] for p in ps]),witness[name]), (name,key)
            if name=='proposal':
                assert torch.equal(witness['virtual_proposal'],witness['proposal'])
                for field in ('exp_avg','exp_avg_sq'):
                    moments = r.flatten(ps,[slots[p].get(field) for p in ps])
                    assert torch.equal(moments,witness['proposed_moments'][field]), (key,field)
                steps = [slots[p].get('step') for p in ps]
                assert h.sha(steps)==h.sha(witness['steps'])==witness['report']['steps_sha']
        optimizer_checks[name] = True
    for name, value in probe['shared'].items():
        independent_trust(name,value)
    gains = audit_gains(state,probe,module.group)
    independent_gradients(state,module,probe)
    return dict(**optimizer_checks, actual_budget=True, independent_gradients=True, **gains)


def check_native_archive(ctx, latest, rows):
    expected_best, old = ctx.old_best, ctx.old_latest
    marks = latest['archive_marks']
    best, end = marks['archive_best'], marks['archive_terminal']
    assert best['epoch'] == expected_best['best_epoch_1based'] and end['epoch'] == old['next_epoch']
    assert best['state'] == {k:h.sha(v) for k,v in expected_best['best_state'].items()}
    assert end['state'] == {k:h.sha(v) for k,v in old['current'].items()}
    assert best['evaluation'] == h.sha(expected_best['test_probability'])
    assert best['losses'] == expected_best['best_losses']
    assert best['gate_gradient'] == old['best']['total_loss_gate_grad']
    assert best['scorer_gradient'] == old['best']['total_loss_scorer_grad']
    assert end['rng'] == h.base_runtime().rng_sha(old['rng'])
    assert latest['next_epoch'] >= end['epoch']
    current = h.base_runtime().rng_sha(latest['rng']['reference'])
    assert current == rows[-1]['reference']['rng_after_eval']
    if latest['next_epoch'] == end['epoch']:
        assert {k:h.sha(v) for k,v in latest['reference'].items()} == end['state']
    return dict(best=best, terminal=end, archived_epochs=end['epoch']-r.START,
                continuation_epochs=latest['next_epoch']-end['epoch'])


def journal_checks(ctx, task, latest, rows, preliminary):
    assert [row['epoch'] for row in rows] == list(range(r.START+1,latest['next_epoch']+1))
    last_rng = preliminary['initial_rng']['rng_before']
    history = ctx.archive_rng
    masks = set()
    checked = {}
    prior_ema = {}
    last_trade = {g:None for g in r.GROUPS}
    for g in r.GROUPS:
        checked[g] = dict(wait=0, best_correct=-1, best_epoch=0, count=0,
                          gradients={}, updates={}, protection={}, route=np.zeros(3,dtype=int),
                          reversals=0,trusted_reversals=0,routed_targets=0,keep=[],forecast=[],ema_best=None)
        prior_ema[g] = preliminary['groups'][g]['meta']['initial_ema_sha']
    for rec in rows:
        epoch, native = rec['epoch'], rec['reference']
        assert native['rng_before'] == last_rng and native['rng_after_train'] == native['rng_after_eval']
        last_rng = native['rng_after_eval']
        if epoch in history:
            assert rec['reference_archived'] and all(native[k] == history[epoch][k] for k in r.RNG_KEYS[1:])
        else:
            assert not rec['reference_archived']
        masks.add(native['dropout_mask_sha'])
        for group in r.GROUPS:
            t = latest['track'][group]
            g = checked[group]
            if epoch > t['terminal_epoch']:
                assert group not in rec['students']
                continue
            assert group in rec['students']
            s = rec['students'][group]
            for key in (*r.RNG_KEYS[1:],'dropout_calls','dropout_mask_sha','rng_streams'):
                assert s[key] == native[key]
            assert sum(x[0]=='cam.Outputlayer.mlp_head.3' and x[2] and x[3]==.35 for x in s['dropout_calls']) == 1
            assert s['ema_before'] == prior_ema[group]
            prior_ema[group] = s['ema_after']
            improvement = s['correct'] > g['best_correct']
            assert improvement == s['improved']
            if improvement:
                g.update(best_correct=s['correct'],best_epoch=epoch,wait=0,ema_best=s['ema_before'])
            else:
                g['wait'] += 1
            assert g['wait'] == s['wait']
            assert (epoch==t['terminal_epoch']) == (s['wait'] > CFG['early_stop_wait_gt'] or epoch==r.LIMIT)
            g['count'] += 1
            g['keep'].append([s['keep']['off'],s['keep']['full'],s['keep']['weight']])
            for name, values in s['shared'].items():
                assert 0 <= values['accepted_reference_distance'] <= values['radius']
                assert 0 <= values['proposal_scale'] <= 1 and 0 <= values['moment_scale'] <= 1
                assert values['retention_injected'] <= CFG['reference']['retention_norm_budget']*values['gradient_full']
                assert values['actual_step'] >= 0 and values['native_proposal_step'] >= 0
                assert all(0 <= block['drift'] <= block['radius'] for block in values['moments'].values())
                g['protection'].setdefault(name,[]).append(values)
            for name, values in s['added'].items():
                assert values['ratio'] <= CFG['loss']['aux_ratio_max']
                assert values['class_only_verified'] if 'class_only_verified' in values else True
                g['gradients'].setdefault(name,[]).append(values['classification'])
                assert values['actual_step_norm'] >= 0 and 0 <= values['actual_update_fraction'] <= 1
                assert (values['actual_step_norm'] > 0) == (values['actual_update_fraction'] > 0)
                g['updates'].setdefault(name,[]).append(values['actual_step_norm'])
            if 'counterfactual' in s['cohorts']:
                counter = s['cohorts']['counterfactual']
                g['route'] += np.asarray(counter['route_counts'])
                g['reversals'] += counter['reverse']
                g['trusted_reversals'] += counter['trusted_reverse']
                g['routed_targets'] += counter['target_active']
                g['forecast'].append([counter['deterministic_abs_error'],counter['forecast_abs_error']])
            last_trade[group] = s
    assert len(masks) == len(rows), 'Native classification dropout did not evolve'
    for group, g in checked.items():
        status = latest['track'][group]
        assert g['best_epoch'] == status['best_epoch'] and g['wait'] == status['wait']
        assert g['best_correct']/len(ctx.reference['test_np']) == status['best_acc']
        assert g['count'] == status['terminal_epoch']-r.START
        assert h.base_runtime().rng_sha(latest['rng'][group]) == last_trade[group]['rng_after_eval']
        g['terminal_ema_sha'] = prior_ema[group]
        g['keep'] = np.mean(g['keep'],axis=0).tolist()
        for name, num in g['gradients'].items():
            g['gradients'][name] = dict(nonzero=sum(v > 0 for v in num), maximum=max(num), fraction=float(np.mean(np.asarray(num)>0)))
        for name, steps in g['updates'].items():
            g['updates'][name] = dict(nonzero=sum(v > 0 for v in steps), maximum=max(steps),
                                      fraction=float(np.mean(np.asarray(steps)>0)))
        for name, values in g['protection'].items():
            fractions = [v['accepted_reference_distance']/v['radius'] for v in values]
            g['protection'][name] = dict(
                steps=len(values), ratio_max=max(fractions), ratio_mean=float(np.mean(fractions)),
                nonzero_step_count=sum(v['actual_step']>0 for v in values),
                different_from_reference=sum(v['accepted_reference_distance']>0 for v in values),
                capped_proposals=sum(v['proposal_scale']<1 for v in values),
                capped_moments=sum(v['moment_scale']<1 for v in values),
                fp32_fallbacks=sum(v['cast_fallback']+v['moment_fallback'] for v in values),
                mean_retention_injected=float(np.mean([v['retention_injected'] for v in values])),
                mean_accepted_vs_proposal=float(np.mean([v['accepted_vs_proposal'] for v in values])),
                max_moment_budget_fraction={k:max(v['moments'][k]['drift']/v['moments'][k]['radius'] for v in values)
                                            for k in ('exp_avg','exp_avg_sq')},
                last=values[-1], best=values[g['best_epoch']-r.START-1])
        g['calibration_mae'] = np.mean(g.pop('forecast'),axis=0).tolist() if g['forecast'] else None
        g.pop('forecast',None)
        g['route'] = g['route'].tolist()
        g['terminal_trace'] = last_trade[group]
    return checked


def evaluate_reference(state, probability_snapshot):
    logp, _ = original_eval(state)
    ids = state['test_np']
    prediction = array(logp[ids].exp())
    assert np.array_equal(prediction,probability_snapshot['probability'])
    same_metrics(metrics(prediction,state['labels_np'][ids]),probability_snapshot['metrics'])
    return prediction


def compare_aligned(state, student, reference, epoch, source):
    ids = state['test_np']
    permitted = ~np.isin(ids,FOLD2_ROWS) if state['fold']==2 else np.ones(len(ids),dtype=bool)
    truth = state['labels_np'][ids]
    off, ref = student['off_probability'],reference['probability']
    rows = [dict(y=int(y), epoch=epoch, fold=state['fold'], row_1based=int(i+1),
                 ref_label=int(pr.argmax()), off_label=int(po.argmax()), full_label=int(pf.argmax()),
                 protected=not bool(allowed), ref_asd=float(pr[0]), off_asd=float(po[0]), full_asd=float(pf[0]))
            for i,y,pr,po,pf,allowed in zip(ids,truth,ref,off,student['probability'],permitted)]
    if source is not None:
        assert epoch == source['best_epoch'] and np.array_equal(ref,source['probability'])
    return dict(epoch=epoch, metrics_ref=metrics(ref,truth), metrics_off=metrics(off,truth),
                metrics_full=metrics(student['probability'],truth),
                probability_max_difference=float(np.abs(ref-off).max()),
                probability_mean_difference=float(np.abs(ref-off).mean()),
                changes=changes([x for x in rows if not x['protected']],'ref','off'), rows=rows)


def compare_optimizer_states(ref, step, state, module, optimizer, epoch):
    checks = check_optimizers(state,module,optimizer,epoch)
    assert set(checks) == {*r.HIST_OPTS,'opt_new'}
    drift_student = optimizer_moment_snapshot(state)
    drift_reference = optimizer_moment_snapshot(ref)
    for name, params in state['old_layout'].items():
        values = step['shared'][name]
        ref_params = ref['old_layout'][name]
        expected_dist = np.linalg.norm(np.concatenate([array(a).astype(np.float64).ravel()-array(b).astype(np.float64).ravel()
                                                       for a,b in zip(params,ref_params)]))
        np.testing.assert_allclose(values['accepted_reference_distance'],expected_dist,rtol=1e-12,atol=1e-14)
        caps = CFG['trust']
        group = name.split('.')
        tag = group[0] if len(group)==2 else '.'.join((group[0],group[-1]))
        width = sum(p.numel() for p in params)
        param_radius = caps['relative_parameter_to_init'][tag]*max(
            np.linalg.norm(array(state['origin'][name])),math.sqrt(width)*caps['parameter_scale_floor'])
        np.testing.assert_allclose(param_radius,values['radius'],rtol=1e-12,atol=1e-14)
        assert expected_dist <= values['radius']+1e-14
        slots, ref_slots = state[group[0]].state, ref[group[0]].state
        steps, ref_steps = ([op.get(p,{}).get('step') for p in ps]
                            for op,ps in ((slots,params),(ref_slots,ref_params)))
        assert h.sha(steps)==h.sha(ref_steps)==values['steps_sha']
        for field in ('exp_avg','exp_avg_sq'):
            a = r.moment_vector(state,name,field)
            b = r.moment_vector(ref,name,field)
            expected = np.linalg.norm(array(a).astype(np.float64)-array(b).astype(np.float64))
            info = values['moments'][field]
            np.testing.assert_allclose([info['reference'],info['student'],info['drift']],
                                       [drift_reference[name][field],drift_student[name][field],expected],
                                       rtol=1e-12,atol=1e-14)
            limit = caps[f'relative_{field}']*max(drift_reference[name][field],math.sqrt(width)*caps['moment_scale_floor'])
            np.testing.assert_allclose(limit,info['radius'],rtol=1e-12,atol=1e-14)
            assert expected <= info['radius']+1e-14
    return checks


def replay_latest(ctx, saved, group, run_journal):
    from types import SimpleNamespace
    state,module,optimizer = ctx.students[group]
    end = saved['track'][group]['terminal_epoch']
    terminal_ref = saved['terminal_refs'][end]
    h.runtime().load_history().load_bundle(ctx.reference,terminal_ref['state'])
    r.restore_student(state,module,optimizer,saved['students'][group])
    ctx.reference_legal = None
    ctx.rng.update({group:saved['rng'][group],'reference':terminal_ref['rng']})
    step = run_journal['terminal_trace']
    assert h.base_runtime().rng_sha(ctx.rng['reference']) == step['rng_after_eval']
    before = module.ema_state()
    assert h.sha(before)==run_journal['terminal_ema_sha']
    assert int(before['ema_updates']) == end-r.START
    terminal_moments = compare_optimizer_states(ctx.reference,step,state,module,optimizer,end)
    probes = {group:{}}
    next_steps,evals,reference_step,reference_eval = r.execute_epoch(ctx,(group,),inspect=True,probes=probes)
    expected = saved['witnesses'][group]
    assert h.sha(expected['step']) == h.sha(next_steps[group])
    assert h.sha(expected['reference_step']) == h.sha(reference_step)
    assert h.sha(evals[group]) == expected['evaluation']
    assert h.sha(reference_eval) == expected['reference_evaluation']
    observation = historical_observation(evals[group]['legal_cam'],state['labels_np'][rows_legal(state)])
    assert h.sha(observation) == expected['observation']
    assert r.update_statistics(state,module,evals[group]) == expected['observation']
    check_ema(module,before,observation)
    assert h.sha(before)==expected['ema_before'] and h.sha(module.ema_state())==expected['ema_after']
    assert h.sha(r.save_student(state,module,optimizer))==expected['student']
    assert h.sha(r.native_bundle(ctx.reference))==expected['reference']
    assert {key:h.base_runtime().rng_sha(ctx.rng[key]) for key in (group,'reference')}==expected['rng']
    math_checks = check_training_probe(state,module,probes[group])
    archived_replay = end+1 in ctx.archive_rng
    if archived_replay:
        assert all(reference_step[k]==ctx.archive_rng[end+1][k] for k in r.RNG_KEYS[1:])
    return dict(epoch=end, next_epoch=end+1, probability_replay=True, ema_replay=True, student_checkpoint_replay=True,
                reference_checkpoint_replay=True, native_rng_replay=True, archived_next_reference_epoch=archived_replay,
                optimizer_state=terminal_moments, **math_checks)


def audit_fold(fold):
    start = time.monotonic()
    assert r.completed_fold(fold)
    pre = r.assert_preflight(fold)
    ctx = r.setup_fold(fold)
    assert r.native_digests(ctx.reference) == pre['init'] and ctx.identity == pre['identity']
    assert pre['inputs'] == dict(legal=h.sha(rows_legal(ctx.reference)),test=h.sha(ctx.reference['test_np']),
                                 pairs=h.sha(ctx.reference['train_pairs_np']),pair_targets=h.sha(ctx.reference['train_same_np']))
    task, latest_path, log = r.fold_paths(fold)
    task, saved = h.read_json(task), h.torch_load(latest_path)
    assert task['identity'] == saved['identity'] == ctx.identity
    assert task['archive_marks'] == saved['archive_marks']
    assert task['log_sha'] == saved['log_sha'] == h.sha_file(log)
    assert saved['log_bytes'] == log.stat().st_size
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    native = check_native_archive(ctx,saved,rows)
    epochs = journal_checks(ctx,task,saved,rows,pre)
    initial_snap = dict(current={'base':r.native_bundle(ctx.reference)})
    b0 = baseline(ctx.reference,ctx.identity,initial_snap)
    group_audits = []
    for group,(state,module,optimizer) in ctx.students.items():
        began = time.monotonic()
        assert ctx.meta[group] == pre['groups'][group]['meta']
        best_path = h.paths(group,fold)[1]
        best = h.torch_load(best_path)
        group_ledger = epochs[group]
        assert best['initial']==ctx.meta[group]
        assert best['epoch']==group_ledger['best_epoch'] and best['ema_sha']==group_ledger['ema_best']
        assert best['identity'] == ctx.identity
        h.runtime().load_history().load_bundle(ctx.reference,best['reference'])
        with r.use_stream(best['rng']['reference']):
            evaluate_reference(ctx.reference,best['reference_evaluation'])
        r.restore_student(state,module,optimizer,best['student'])
        assert int(module.ema_updates)==best['epoch']-r.START-1 and h.sha(module.ema_state())==best['ema_sha']
        group_step = rows[best['epoch']-r.START-1]['students'][group]
        # JSON journals encode the tuples of native dropout call metadata as lists.
        projected_step = {key:group_step[key] for key in best['step']}
        assert json.dumps(best['step'],sort_keys=True) == json.dumps(projected_step,sort_keys=True)
        assert h.base_runtime().rng_sha(best['rng'][group])==group_step['rng_after_eval']
        assert h.base_runtime().rng_sha(best['rng']['reference'])==group_step['rng_after_eval']
        with r.use_stream(best['rng'][group]):
            found = best_interventions(state,module,best['evaluation'],b0)
        same_metrics(found['test_metrics']['full'],best['evaluation']['metrics'])
        paired_best = compare_aligned(state,best['evaluation'],best['reference_evaluation'],best['epoch'],None)
        paired_b0 = None
        if group in saved['aligned_b0']:
            b = saved['aligned_b0'][group]
            paired_b0 = compare_aligned(state,b['evaluation'],b['reference'],b['epoch'],b0)
        else:
            assert best['epoch'] <= saved['track'][group]['terminal_epoch'] < b0['best_epoch']
        # Keep a fold-level alignment proof and patient's aligned-best predictions.
        patient_records = {p['row_1based']:p for p in found['records']}
        for index,comp in (('student_best',paired_best),('historic_best',paired_b0)):
            if comp:
                for p in comp.pop('rows'):
                    for k in ('ref_label','ref_asd','off_label','off_asd','full_label','full_asd'):
                        patient_records[p['row_1based']][f'{index}_{k}']=p[k]
            else:
                for p in patient_records.values():
                    for k in ('ref_label','ref_asd','off_label','off_asd','full_label','full_asd'):
                        p[f'{index}_{k}']=None
        best_opts = compare_optimizer_states(ctx.reference,best['step'],state,module,optimizer,best['epoch'])
        last = replay_latest(ctx,saved,group,group_ledger)
        assert all(report['fraction'] > 0 for report in group_ledger['gradients'].values())
        b0_serial = {k:v for k,v in b0.items() if k not in ('probability','gap')}
        group_ledger.pop('terminal_trace')
        group_audits.append(dict(group=group,fold=fold,status='PASS',preflight=True, best_pass=True, latest_pass=True,
                                 reference_pass=True,best_epoch=best['epoch'],end_epoch=saved['track'][group]['terminal_epoch'],
                                 metrics=found['test_metrics']['full'],baseline=b0_serial,
                                 checkpoint_sha=h.sha_file(best_path), same_epoch_best=paired_best,same_epoch_b0=paired_b0,
                                 best_optimizer=best_opts,latest=last,training=group_ledger, seconds=time.monotonic()-began,
                                 **found))
    return dict(fold=fold,status='PASS',groups=group_audits,native=native,fold_elapsed=time.monotonic()-start,
                independent_source_sha=h.sha_file(__file__))


def decide(audits, combined, protection):
    answer = {}
    limits = CFG['decision']
    for group,values in combined['mechanisms'].items():
        pairs = combined['paired'][f'{group}-B0']
        full, no2 = pairs['full'], pairs['no_fold2']
        delta = full['delta_pp']
        files = [a for a in audits if a['group']==group]
        complete = len(files)==10 and all(a['status']=='PASS' for a in files) and protection['pass_']
        performance = dict(audit=complete, acc=delta['acc']>0, ba=delta['hard_auc']>=0,
                           net=full['changes']['all']['net']>0, prob_auc=delta['prob_auc']>=limits['paired_prob_auc_min_pp'],
                           asd_sen=delta['sen']>=limits['paired_sen_min_pp'],
                           no_fold2=no2['delta_pp']['acc']>0 and no2['changes']['all']['net']>0)
        first = values['strata']['all']
        mechanism = dict(audit=complete, references=all(a['reference_pass'] for a in files),
                         basis=all(a['basis']['rank']==16 and a['basis']['projector_change_frobenius']>TOL['projector_change_min']
                                   for a in files),
                         end_to_end=all(all(x['nonzero']>0 for x in a['training']['gradients'].values()) for a in files),
                         probability=first['probability_change']['mean']>0, residual=first['residual_nonzero']>0,
                         optimizer=all(a['latest']['actual_budget'] for a in files),
                         counterfactual=group!='G2' or all(a['latest']['candidate_independent'] for a in files))
        direct = values['stages']['B']['all']['net']>0 and bool(values['attribution']['new_residual_correction'])
        mechanism['new_direct_correction'] = direct
        answer[group] = dict(performance_go=all(performance.values()),mechanism_go=all(mechanism.values()),
                             new_residual_correction=direct,
                             go=all(performance.values()) and all(mechanism.values()) and direct,
                             performance_checks=performance,mechanism_checks=mechanism)
    return dict(decision='GO' if any(v['go'] for v in answer.values()) else 'NO-GO',groups=answer)


def add_alignment_stats(audits, combined):
    for group, mech in combined['mechanisms'].items():
        parts = [a for a in audits if a['group']==group]
        aligned = {}
        for label in ('same_epoch_best','same_epoch_b0'):
            rows = [p[label] for p in parts if p[label] is not None]
            stages = {c:{k:sum(x['changes'][c][k] for x in rows) for k in ('corrected','harmed','net')}
                      for c in ('asd','td','all')}
            aligned[label] = dict(
                folds=len(rows), off_change=stages,
                residual_off_pct={k:float(np.mean([100*x['metrics_off'][k] for x in rows])) if rows else None for k in AUC_KEYS},
                reference_pct={k:float(np.mean([100*x['metrics_ref'][k] for x in rows])) if rows else None for k in AUC_KEYS},
                delta_pp={k:float(np.mean([100*(x['metrics_off'][k]-x['metrics_ref'][k]) for x in rows])) if rows else None
                          for k in AUC_KEYS})
        records = [p for a in parts for p in a['records'] if not p['protected']]
        if group=='G2':
            choice = np.bincount([p['selected_route'] for p in records],minlength=3)
            reverse = [p for p in records if p['selected_route']>0 and p['selected_route']-1 != p['off_label']]
            mech['routing'] = dict(test_argmax=choice.tolist(),
                                   test_mean_probability=[float(np.mean([p[f'route_{k}'] for p in records])) for k in ('keep','asd','td')],
                                   test_reversals=len(reverse), test_reversal_correct=sum(p['selected_route']-1==p['y'] for p in reverse),
                                   training_counts=np.sum([p['training']['route'] for p in parts],axis=0).tolist(),
                                   training_reversals=sum(p['training']['reversals'] for p in parts),
                                   training_trusted_reversals=sum(p['training']['trusted_reversals'] for p in parts),
                                   training_targets=sum(p['training']['routed_targets'] for p in parts),
                                   training_calibration_mae=np.average(
                                       [p['training']['calibration_mae'] for p in parts],
                                       weights=[p['training']['count'] for p in parts],axis=0).tolist(),
                                   soft_cancellation=distribution([p['soft_cancellation'] for p in records]),
                                   selected_gap_calibration_error=distribution(
                                       [abs(p['selected_forecast_gap']-p['selected_gap']) for p in records]),
                                   execution='one selected candidate; no forward candidate mixing')
        selected = [p['best_epoch']-p['baseline']['best_epoch'] for p in parts]
        mech['alignment'] = aligned
        mech['best_epoch_shift'] = dict(mean=float(np.mean(selected)), minimum=min(selected), maximum=max(selected),
                                        ties=sum(e == 0 for e in selected))


def audit_all():
    import report

    started = time.monotonic()
    assert all(r.completed_fold(fold) for fold in r.FOLDS)
    h.update_manifest(status='AUDITING')
    BASE_DIR.mkdir(exist_ok=True)
    current = []
    for fold in r.FOLDS:
        path = BASE_DIR/f'f{fold:02d}.json'
        if path.exists():
            candidate = h.read_json(path)
            if (candidate.get('manifest_sources')==r.implementation() and
                    candidate.get('task_sha')==h.sha_file(r.fold_paths(fold)[0]) and
                    candidate.get('independent_source_sha')==h.sha_file(__file__)):
                current.append(fold)
    tasks = [f for f in r.FOLDS if f not in current]
    r.process_pool('--audit-fold',tasks,min(4,r.cpu_capacity()))
    fold_audits = [h.read_json(BASE_DIR/f'f{fold:02d}.json') for fold in r.FOLDS]
    audits = [a for fold in fold_audits for a in fold['groups']]
    combined = aggregate(audits)
    add_alignment_stats(audits,combined)
    integrity = r.protection()
    science = decide(audits,combined,integrity)
    status = dict(status='COMPLETE_AUDITED' if integrity['pass_'] else 'AUDIT_FAILED',
                  experiment=CFG['experiment'], audits=20, preflight=20, best=20, latest=20,
                  reference_fidelity=10, protection=integrity, aggregate=combined, science=science,
                  audit_seconds=time.monotonic()-started,
                  audit_paralleled_with_training=dict(folds=current,
                    cumulative_worker_seconds=sum(f['fold_elapsed'] for f in fold_audits if f['fold'] in current)),
                  cumulative_audit_worker_seconds=sum(f['fold_elapsed'] for f in fold_audits),
                  folds=[{k:f[k] for k in ('fold','status','native','fold_elapsed')} for f in fold_audits])
    report.publish(status,audits)
    return status
