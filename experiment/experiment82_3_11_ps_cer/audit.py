#!/usr/bin/env python3
"""Independent PS-CER equations, patient pairing, and checkpoint audit."""
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
from ps_cer import EMA_KEYS, FOLD2_ROWS, INTERVENTIONS

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



def explicit_fp32_head(head, z, mask=None):
    # Reconstruct the native inference arithmetic without calling OutputLayer.
    norm, affine, gelu, _ = head.mlp_head
    normalized, _, _ = torch.native_layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
    activated = F.gelu(F.linear(normalized, affine.weight, affine.bias), approximate=gelu.approximate)
    if mask is not None:
        activated = activated*mask
    return F.linear(activated, head.classifier.weight, head.classifier.bias)


def independent_capture(state, training=False):
    """Collect native value-linear outputs, context tokens, the head input and mask."""
    cam, found = state['cam'], {}

    def store(key):
        def collect(_layer, _args, output):
            found[key] = output
        return collect

    def head_input(_layer, args):
        found['z'] = args[0]

    def drop(_layer, args, output):
        found['before'], found['after'] = args[0], output

    handles = [layer.register_forward_hook(store(i)) for i,layer in enumerate(cam.InputLayer.w_vs)]
    handles += [cam.FeedForward[-1].register_forward_hook(store('context')),
                cam.Outputlayer.register_forward_pre_hook(head_input),
                cam.Outputlayer.mlp_head[-1].register_forward_hook(drop)]
    try:
        with torch.set_grad_enabled(training):
            out = h.runtime().forward(state, r.BASE_GROUP, training=training)
    finally:
        for handle in handles:
            handle.remove()
    found['local'] = torch.stack([found[i] for i in range(4)], dim=1)
    assert found['local'].shape == found['context'].shape == (CFG['subjects'],4,36)
    assert torch.equal(found['context'].flatten(1),found['z'])
    return out, found


def counterfactual_shift(head, z, candidates):
    gaps = []
    off = explicit_fp32_head(head,z)
    for candidate in candidates.unbind(1):
        scores = explicit_fp32_head(head,z+candidate)
        gaps.append(scores[:,0]-scores[:,1])
    return torch.stack(gaps,1)-(off[:,0]-off[:,1])[:, None]

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
        dispersion=eigenvalues.mean(1)*(1/eigenvalues).mean(1), stability=stable.mean(0), raw_llr=attribution, llr_total=total_llr, mahalanobis=terms.sum(-1).T,
        loglam=normalize(np.log(raw_var)), nu=normalize(np.log1p(contrast))).items()}


def independent_distance(d):
    mean, cov = array(d['class_mean']), array(d['class_cov'])
    mid = np.mean(cov, axis=0)
    dm = mean[0]-mean[1]
    location = float(dm@np.linalg.solve(mid, dm)/8)
    volume = float((np.linalg.slogdet(mid)[1]-np.linalg.slogdet(cov)[1].mean())/2)
    return location+max(0.,volume), location, max(0.,volume)


@torch.no_grad()
def manual_semantics(module, tokens, net, null_key):
    centered = tokens - tokens.mean(-1,keepdim=True)
    normalized = centered*(centered.square().mean(-1,keepdim=True)+1e-5).rsqrt()
    value = F.linear(F.gelu(F.linear(normalized,net[0].weight,net[0].bias)),net[2].weight,net[2].bias)
    unit = value/value.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    query = module.q0/module.q0.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    temp = CFG['routing']['token_temperature']
    token = torch.einsum('nkd,ad->nka',unit,temp*query)
    group_scores = [torch.logsumexp(token[:,:,start:start+4],dim=2)-math.log(4) for start in (4,0)]
    category = torch.stack((*group_scores, torch.einsum('nkd,d->nk',unit,temp*null_key/null_key.norm())),dim=2)
    posterior = torch.special.softmax(category,dim=2).mean(dim=1)
    cond = posterior[:,:2]/posterior[:,:2].sum(1,keepdim=True)
    entropy = torch.stack([torch.special.entr(token[:,:,start:start+4].softmax(2)).sum(2).mean(1)
                           for start in (4,0)]).mean(0)
    return cond, posterior[:,2], (1-entropy/math.log(4)).clamp(0,1)


def manual_action(module,z,tokens,jac,gap,head,intervention,measured):
    basis, _ = clean_basis(module)
    if intervention == 'spectral_guidance_off':
        basis = array(module.U0)
    u = torch.as_tensor(basis,device=z.device)
    result = manual_projection(module,u,z)
    p0, n0, c0 = manual_semantics(module,tokens['local'],module.local_map,module.null_key[0])
    p1, n1, c1 = manual_semantics(module,tokens['context'],module.context_map,module.null_key[1])
    s0, s1 = p0[:,0]-p0[:,1],p1[:,0]-p1[:,1]
    s, null, concentrated = (s0+s1)/2, (n0+n1)/2, (c0+c1)/2
    agreement = 1-(s0-s1).abs()/2
    rs = (1-null)*agreement*(1+concentrated)/2
    stat = torch.tanh(result['llr_total'].float()/module.tau_l)
    mahal = result['mahalanobis'].amin(1).float()
    rt = (1+result['stability'].mean().float())/(2*torch.maximum(mahal/16,torch.ones_like(mahal)))
    cfg = CFG['routing']
    signs = z.new_tensor([[1.,-1.]])
    if intervention == 'semantic_evidence_off':
        s, rs, agreement, concentrated = (torch.zeros_like(x) for x in (s,rs,agreement,concentrated))
        null = torch.ones_like(null)
    # Certify the upstream evidence before comparing the nonlinear gate at
    # exactly matched inputs. Subtraction near a threshold amplifies FP32
    # rounding of an independently evaluated normalization.
    upstream = dict(semantic=s,statistical=stat,sem_rel=rs,stat_rel=rt,consistency=agreement,
                    null=null,concentration=concentrated,p_local=p0,p_context=p1)
    for key,value in upstream.items():
        compare(value,measured[key])
    s,stat,rs,rt,agreement,null,concentrated = (
        measured[k] for k in ('semantic','statistical','sem_rel','stat_rel','consistency','null','concentration'))
    sem_support = s[:,None]*signs
    stat_support = stat[:,None]*signs
    ws = torch.maximum(rs[:,None]*sem_support-cfg['semantic_floor'],z.new_zeros(()))
    wt = torch.maximum(rt[:,None]*stat_support-cfg['statistical_floor'],z.new_zeros(()))
    evidence = wt if intervention=='semantic_evidence_off' else torch.sqrt(ws*wt+1e-8)-1e-4
    prior = 2*evidence*(gap[:,None]*signs < 0)
    gain = module.direction_gain
    assert float(gain.detach().min()) >= 0 and float(gain.detach().max()) <= 1
    parts = []
    j_spec = torch.einsum('ni,ik->nk',jac,u.float())
    j_unit = j_spec/torch.hypot(j_spec.norm(dim=1,keepdim=True),j_spec.new_tensor(CFG['jacobian_epsilon']))
    stability = result['stability'].float()
    h16 = result['hh'].float()
    llr16 = (result['raw_llr'].float()/CFG['class_statistics']['llr_bound']).tanh()
    for a,sgn in enumerate((1,-1)):
        expanded = lambda t: t[:,None].expand_as(h16)
        feature = torch.stack((h16,h16.abs(),llr16*sgn,expanded(sem_support[:,a]),
                               expanded(stat_support[:,a]),stability.expand_as(h16),
                               expanded(rt),expanded(rs),expanded(prior[:,a])),dim=-1)
        inp = F.linear(feature,module.candidate_net[0].weight,module.candidate_net[0].bias)
        amp = torch.sigmoid(F.linear(F.gelu(inp),module.candidate_net[2].weight,module.candidate_net[2].bias)[...,0])
        coef = sgn*j_unit*gain[a]*amp*(0.25+0.75*stability)*(0.25+0.75*rt[:,None])
        parts.append(coef @ u.float().T)
    raw = torch.stack(parts,1)
    zlen = torch.linalg.vector_norm(z.double(),dim=1)[:,None]
    rawlen = torch.linalg.vector_norm(raw.double(),dim=2)
    safe_z = zlen.clamp_min(1e-8)
    target = CFG['residual_limit']*zlen*torch.tanh(rawlen/(CFG['residual_limit']*safe_z))
    scale = target/rawlen.clamp_min(torch.finfo(torch.float64).tiny)
    bounded = (raw.double()*scale[:,:,None]*(zlen/(zlen+torch.finfo(z.dtype).tiny))[:,:,None]).float()
    if module.fold == 2:
        bounded[list(FOLD2_ROWS)] = 0
        prior[list(FOLD2_ROWS)] = 0
    if intervention == 'residual_off':
        bounded.zero_()
        prior.zero_()
    compare(bounded,measured['candidates'])
    shift = counterfactual_shift(head,z,measured['candidates'])
    assert torch.equal(shift,measured['candidate_effect'])
    gscale = cfg['gap_scale']
    router_input = torch.column_stack((s,stat,rs,rt,agreement,null,concentrated,torch.tanh(gap/gscale),
                                        stability.mean().expand_as(s),prior,torch.tanh(shift*signs/gscale)))
    hidden = F.gelu(F.linear(router_input,module.router[0].weight,module.router[0].bias))
    logit_bound = cfg['logit_bound']
    score = logit_bound*torch.tanh(F.linear(hidden,module.router[2].weight,module.router[2].bias)/logit_bound)
    select = score.exp()*torch.column_stack((torch.ones_like(s),prior))
    route = select/select.sum(1,keepdim=True)
    chosen = route.argmax(1)
    selected = torch.cat((torch.zeros_like(bounded[:,:1]),bounded),1)[torch.arange(len(z),device=z.device),chosen]
    result.update(semantic=s, null=null, prior=prior, statistical=stat, sem_rel=rs, stat_rel=rt,
                  concentration=concentrated, consistency=agreement, p_local=p0, p_context=p1,
                  j_spec=j_spec, candidates=bounded, candidate_effect=shift, route=route, choice=chosen, delta=selected)
    return result


@torch.no_grad()
def check_formulas(state, module, found, intervention='full', actual=None):
    from ps_cer import head_sensitivity

    z = found['z']
    head = state['cam'].Outputlayer
    j, gap = head_sensitivity(head,z)
    analytic, analytical_logits = analytic_jacobian(head,z)
    compare(j,analytic)
    exact_logits = native_head_logits(head,z)
    compare(exact_logits,analytical_logits.float())
    assert torch.equal(gap,exact_logits[:,0]-exact_logits[:,1])
    got = actual if actual is not None else module(z,found['local'],found['context'],j,gap,head,intervention)
    independent = manual_action(module,z,found,j,gap,head,intervention,got)
    for key in ('raw_llr','llr_total','class_mean','class_cov','class_logdet','projected','mahalanobis'):
        torch.testing.assert_close(independent[key],got[key],atol=TOL['audit_fp64_atol'],rtol=1e-10)
    for key in ('p_local','p_context','null','concentration','semantic','statistical','sem_rel','stat_rel','consistency',
                'prior','route','candidates','candidate_effect','j_spec','delta'):
        try:
            compare(independent[key].float(),got[key].float())
        except AssertionError as exc:
            raise AssertionError(f'Independent {key} failed: {exc}') from exc
    assert torch.equal(independent['choice'],got['choice'])
    torch.testing.assert_close(got['raw_llr'].sum(1),got['llr_total'],atol=TOL['audit_fp64_atol'],rtol=1e-11)
    assert torch.isfinite(got['enhanced']).all() and torch.isfinite(got['route']).all()
    assert torch.all(got['route'] >= 0)
    assert torch.all(got['route'][:,1:][got['prior']==0] == 0)
    assert torch.all((got['prior'] > 0).sum(1) <= 1)
    lengths = got['candidates'].double().norm(dim=2)
    radius = CFG['residual_limit']*z.double().norm(dim=1,keepdim=True)
    assert torch.all(lengths-radius<=TOL['residual_bound_roundoff'])
    linear = (got['candidates']*j[:,None]).sum(-1)*z.new_tensor([[1.,-1.]])
    assert torch.all(linear >= -TOL['audit_feature_atol'])
    if module.fold==2:
        assert got['candidates'][list(FOLD2_ROWS)].count_nonzero()==0
        assert torch.equal(got['route'][list(FOLD2_ROWS)],got['route'].new_tensor([[1,0,0],[1,0,0]]))
    return dict(covariance=torch.linalg.cond(got['class_cov']).tolist(), independent_action=True,
                local_shape=list(found['local'].shape),context_shape=list(found['context'].shape),
                covariance_sum_max_error=float((got['raw_llr'].sum(1)-got['llr_total']).abs().max()),
                max_relative_residual=float((lengths/z.double().norm(dim=1,keepdim=True).clamp_min(1e-8)).max()),
                eligible=int((got['prior']>0).any(1).sum()),mask_replay=True)

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
    weights = .25+.75/(1+np.exp(np.clip(margin-1, -200, 200)))
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


def site_identity(state):
    source = h.REPO/'data/ABIDE/processed_standard_data.csv'
    with source.open(newline='') as handle:
        reader = csv.DictReader(handle)
        sites = [k for k in reader.fieldnames if k.startswith('SITE_ID_')]
        rows = list(reader)
    assert len(rows) == CFG['subjects']
    raw_labels = np.array([float(row['label']) for row in rows],dtype=np.float64)
    assert np.isin(raw_labels,(1.,2.)).all()
    # The native ABIDE loader maps the site's stored labels with raw_label - 1.
    assert np.array_equal(raw_labels - 1.,state['labels_np'])
    out = []
    for row in rows:
        name, = [site[8:] for site in sites if float(row[site]) == 1]
        out.append(name)
    return out


@torch.no_grad()
def best_interventions(state, module, best, source):
    from ps_cer import head_sensitivity

    old_rng = r.rng_mark()
    module.eval()
    native, found = independent_capture(state)
    z = found['z']
    head = state['cam'].Outputlayer
    jac, gap = head_sensitivity(head,z)
    ids = state['test_np']
    truth = state['labels_np'][ids]
    allowed = ~np.isin(ids,FOLD2_ROWS) if state['fold']==2 else np.ones(len(ids),dtype=bool)
    outcomes = {}
    formula = {}
    for mode in INTERVENTIONS:
        details = module(z,found['local'],found['context'],jac,gap,head,mode)
        formula[mode] = check_formulas(state,module,found,mode,details)
        # Evaluate exactly one selected feature intervention with the genuine head.
        logp, _ = head(z+details['delta'], None)
        all_action = [native['logp']]
        for delta in details['candidates'].unbind(1):
            all_action.append(F.log_softmax(explicit_fp32_head(head,z+delta),dim=1))
        options = torch.stack(all_action,dim=1)
        expected = options[torch.arange(len(z),device=z.device),details['choice']]
        assert torch.equal(logp,expected)
        outcomes[mode] = (array(logp[ids].exp()), array((logp[:,0]-logp[:,1])[ids]))
        if mode == 'full':
            full = details
            candidate_logp = array(options[ids])
            train_objective = projected_scores(state,module,z,jac,native['logp'])
    off = outcomes['residual_off']
    prob, new_gap = outcomes['full']
    assert np.array_equal(off[0],array(native['logp'][ids].exp()))
    assert np.array_equal(prob,best['probability']) and np.array_equal(off[0],best['off_probability'])
    np.testing.assert_allclose(new_gap,best['gap'],atol=TOL['audit_feature_atol'],rtol=TOL['audit_feature_rtol'])
    assert np.array_equal(off[1],best['off_gap'])
    assert r.rng_mark() == old_rng
    assert np.array_equal(prob[~allowed],off[0][~allowed])
    projected_origin = jac @ module.U0.float()
    jac_energy = jac.double().square().sum(1).clamp_min(1e-12)
    z_norm = z.double().norm(dim=1).clamp_min(1e-8)
    residual_norm = full['delta'].double().norm(dim=1)
    patient_fields = {name:array(full[name][ids]) for name in
                      ('p_local','p_context','null','consistency','concentration','sem_rel','stat_rel','semantic',
                       'statistical','prior','route','choice','llr_total','candidate_effect')}
    patient_fields.update(j_cover_u0=array((projected_origin.double().square().sum(1)/jac_energy)[ids]),
                          j_cover_u=array((full['j_spec'].double().square().sum(1)/jac_energy)[ids]),
                          relative_residual=array((residual_norm/z_norm)[ids]),
                          alignment=array(F.cosine_similarity(jac,full['delta'],dim=1)[ids]))
    sites = site_identity(state)
    records = []
    for k,row in enumerate(ids):
        label = int(truth[k])
        b0 = source['probability'][k]
        rec = dict(fold=state['fold'],row_1based=int(row)+1,y=label,site=sites[row],protected=not bool(allowed[k]),
                   b0_asd=float(b0[0]),b0_td=float(b0[1]),b0_label=int(b0.argmax()),
                   b0_margin=float(source['gap'][k]*(1-2*label)) if allowed[k] else None)
        for name, pred, margin in [('off',*off),('full',prob,new_gap),
                                   *((mode,*outcomes[mode]) for mode in INTERVENTIONS[2:])]:
            rec.update({name+'_asd':float(pred[k,0]),name+'_td':float(pred[k,1]),name+'_label':int(pred[k].argmax()),
                        name+'_margin':float(margin[k]*(1-2*label)) if allowed[k] else None})
        if allowed[k]:
            columns = ('null','consistency','concentration','sem_rel','stat_rel','semantic','statistical','llr_total',
                       'j_cover_u0','j_cover_u','relative_residual','alignment')
            rec.update({name:float(patient_fields[name][k]) for name in columns})
            rec.update(local_asd=float(patient_fields['p_local'][k,0]),context_asd=float(patient_fields['p_context'][k,0]),
                       selected_route=int(patient_fields['choice'][k]),
                       eligible=bool(patient_fields['prior'][k].max()>0),
                       conflict_score=float(patient_fields['prior'][k].max()))
            rec.update({f'route_{a}':float(v) for a,v in zip(('keep','asd','td'),patient_fields['route'][k])})
            rec['entropy'] = float(-sum(q*math.log(q) for q in patient_fields['route'][k] if q>0))
            class_nll = -candidate_logp[k,:,label].astype(float)
            rec.update(hard_nll=float(class_nll[rec['selected_route']]),
                       expected_nll=float(class_nll@patient_fields['route'][k]),
                       asd_candidate_gap=float(patient_fields['candidate_effect'][k,0]),
                       td_candidate_gap=float(patient_fields['candidate_effect'][k,1]))
        else:
            # Keep the historical predictions and label for the formal six metrics.
            rec.update({name:None for name in ('null','consistency','concentration','sem_rel','stat_rel','semantic',
                       'statistical','llr_total','j_cover_u0','j_cover_u','relative_residual','alignment',
                       'local_asd','context_asd','selected_route','eligible','conflict_score','route_keep','route_asd',
                       'route_td','entropy','hard_nll','expected_nll','asd_candidate_gap','td_candidate_gap')})
        records.append(rec)
    return dict(records=records, test_metrics={key:metrics(p,truth) for key,(p,_) in outcomes.items()},
                mechanism_test_metrics={key:metrics(p[allowed],truth[allowed]) for key,(p,_) in outcomes.items()},
                basis=basis_diagnostics(module), objective=train_objective, formula=formula,
                exact_best_probability=True, protected_count=int((~allowed).sum()))

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
    columns = ('relative_residual','alignment','j_cover_u0','j_cover_u','semantic','statistical','llr_total',
               'sem_rel','stat_rel','null','consistency','concentration','entropy','conflict_score')
    values = {key:distribution([x[key] for x in records]) for key in columns}
    margins = {tag:distribution([x[to+'_margin']-x[fr+'_margin'] for x in records]) for tag,fr,to in
               (('A','b0','off'),('B','off','full'),('C','b0','full'))}
    n = len(records)
    return dict(n=n, residual_nonzero=sum(x['relative_residual']>0 for x in records), margin=margins,
                positive_margin_share=sum(x['full_margin']>x['off_margin'] for x in records)/n if n else None,
                probability_change=distribution([abs(x['full_asd']-x['off_asd']) for x in records]),
                gap_change=distribution([abs(x['full_margin']-x['off_margin']) for x in records]), **values)


def group_mechanisms(audits):
    records = sorted((row for a in audits for row in a['records'] if not row['protected']),key=lambda x:x['row_1based'])
    assert len(records) == 869
    hp = CFG['patients']
    strata = {
        'all':records,
        'darg_correct':[x for x in records if x['b0_label']==x['y']],
        'darg_wrong':[x for x in records if x['b0_label']!=x['y']],
        'low_b0_margin':[x for x in records if abs(x['b0_margin'])<hp['low_b0_true_margin_abs_lt']],
        'high_confidence_b0_wrong':[x for x in records if x['b0_label']!=x['y'] and x['b0_margin']<=hp['high_confidence_b0_wrong_margin_le']],
        'conflicting_evidence':[x for x in records if x['semantic']*x['statistical']<0],
        'darg_wrong_asd':[x for x in records if x['y']==0 and x['b0_label']!=x['y']],
        'darg_wrong_td':[x for x in records if x['y']==1 and x['b0_label']!=x['y']],
        'asd':[x for x in records if x['y']==0], 'td':[x for x in records if x['y']==1],
    }
    direct = {tag:changes(records,first,second) for tag,first,second in
              (('A','b0','off'),('B','off','full'),('C','b0','full'))}
    categories = {key:[] for key in ('new_residual_correction','backbone_harm','residual_rescue','new_residual_harm',
                                     'remaining_errors','backbone_correction')}
    for x in records:
        b,o,f = (x[f'{p}_label']==x['y'] for p in ('b0','off','full'))
        for key, condition in zip(categories,(not b and not o and f,b and not o,b and not o and f,b and o and not f,
                                               not f,not b and o)):
            if condition:
                categories[key].append(x['row_1based'])
    ablations = {}
    for mode in INTERVENTIONS[1:]:
        label = 'off' if mode=='residual_off' else mode
        ablations[mode] = dict(change=changes(records,label,'full'),
                               flips=sum(x[label+'_label']!=x['full_label'] for x in records),
                               probability_absolute_change=distribution([abs(x[label+'_asd']-x['full_asd']) for x in records]),
                               metrics=summarize([a['mechanism_test_metrics'][mode] for a in audits]))
    acted = [x for x in records if x['selected_route']>0]
    eligible = [x for x in records if x['eligible']]
    sites = {}
    for site in sorted({x['site'] for x in records}):
        cohort = [x for x in records if x['site']==site]
        acts = sum(x['selected_route']>0 for x in cohort)
        sites[site] = dict(n=len(cohort),actions=acts,rate=acts/len(cohort),
                           direct=changes(cohort,'off','full')['all'],overall=changes(cohort,'b0','full')['all'])
    score = [x['conflict_score'] for x in records]
    off_wrong = [x['off_label']!=x['y'] for x in records]
    history_wrong = [x['b0_label']!=x['y'] for x in records]
    routing = dict(actions=np.bincount([x['selected_route'] for x in records],minlength=3).tolist(),
                   mean_probability=[float(np.mean([x[f'route_{a}'] for x in records])) for a in ('keep','asd','td')],
                   eligibility_n=len(eligible), correct_disease_actions=sum(x['selected_route']==x['y']+1 for x in acted),
                   action_precision=float(np.mean([x['selected_route']==x['y']+1 for x in acted])) if acted else None,
                   eligible_error_fraction=float(np.mean([x['off_label']!=x['y'] for x in eligible])) if eligible else None,
                   off_error_prevalence=float(np.mean(off_wrong)),
                   off_error_auroc=float(roc_auc_score(off_wrong,score)),
                   b0_error_auroc=float(roc_auc_score(history_wrong,score)),
                   nll_gap=distribution([x['hard_nll']-x['expected_nll'] for x in records]),
                   nll_expected=distribution([x['expected_nll'] for x in records]),
                   nll_hard=distribution([x['hard_nll'] for x in records]),
                   positive_asd_nonlinear=sum(x['asd_candidate_gap']>0 for x in records),
                   negative_td_nonlinear=sum(x['td_candidate_gap']<0 for x in records))
    return dict(strata={k:cohort_description(v) for k,v in strata.items()}, stages=direct, attribution=categories,
                routing=routing, sites=sites, ablations=ablations,
                basis={key:distribution([a['basis'][key] for a in audits]) for key in
                       ('projector_change_frobenius','basis_change_frobenius','gram_max_abs','gram_condition',
                        'max_angle_deg','mean_angle_deg','rank','offset_norm')},
                objectives={basis:{key:distribution([a['objective'][basis][key] for a in audits]) for key in
                            ('bhattacharyya','bhattacharyya_mean','bhattacharyya_covariance','error_gradient_coverage',
                             'variance_effective_rank','participation_rank','eval_covariance_rank','condition')}
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
        links.append(('G2','G1'))
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
    _, found = independent_capture(state)
    return dict(geometry=basis_diagnostics(module),**check_formulas(state,module,found))


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
    """Rebuild an isolated native differentiable view and index the action losses independently."""
    import copy
    from ps_cer import head_sensitivity

    saved = r.native_bundle(state)
    reconstructed = copy.deepcopy(module)
    reconstructed.load_state_dict(probe['pre_module'],strict=True)
    h.runtime().load_history().load_bundle(state,probe['pre_step'])
    old_params = r.parameters_of(state['old_layout'])
    new_params = tuple(reconstructed.parameters())
    current = r.rng_mark()
    try:
        with r.use_stream(probe['pre_rng']):
            out, found = independent_capture(state,training=True)
            local, context = found['local'],found['context']
            for key in ('local','context'):
                assert torch.equal(found[key],probe[key])
            assert torch.equal(found['before'],probe['native_dropout'][0])
            assert torch.equal(found['after'],probe['native_dropout'][1])
            mask, = torch.autograd.grad(found['after'].sum(),found['before'],retain_graph=True)
            assert torch.equal(mask,probe['native_mask'])
            z = found['z']
            head = state['cam'].Outputlayer
            j,gap = head_sensitivity(head,z)
            d = reconstructed(z,local,context,j,gap,head)
            scores = torch.stack((out['logp'],*[F.log_softmax(explicit_fp32_head(head,z+delta,mask),dim=1)
                                                for delta in d['candidates'].unbind(1)]),dim=1)
            assert torch.equal(scores,probe['actions'])
            weights = d['route']
            legal = rows_legal(state)
            y = np.asarray(state['labels_np'])
            assert np.array_equal(legal,state['legal_rows'].cpu())
            components = []
            for c in (0,1):
                rows = legal[y[legal]==c]
                risk = ((scores[rows,:1,c]-scores[rows,1:,c])*weights[rows,1:]).sum(1)
                components.append(risk.mean())
            risk = (components[0]+components[1])/2
            assert abs(float(risk.detach())-probe['risk'])<=1e-9
            train_y = state['label_legal']
            pairs = [(probe['teacher']['masked'],out['logp'][legal]),
                     (probe['teacher']['deterministic'],explicit_fp32_head(head,z).log_softmax(1)[legal])]
            fparts = []
            for p,q in pairs:
                p64,q64 = p.double(),q.double()
                kl = ((p64-q64)*p64.exp()).sum(1)
                kl = kl.masked_fill((p==q).all(1),0.)
                fparts.append((kl[train_y==0].mean()+kl[train_y==1].mean())*0.5)
            mix = CFG['reference']['masked_kl_fraction']
            kl = mix*fparts[0]+(1-mix)*fparts[1]
            assert abs(float(kl.detach())-probe['functional'])<1e-10
            native_grads = torch.autograd.grad(out['total'],old_params,retain_graph=True,allow_unused=True)
            full_grads = torch.autograd.grad(out['total']+risk,old_params,retain_graph=True,allow_unused=True)
            shared_kl = torch.autograd.grad(kl,old_params+new_params,allow_unused=True)
            assert all(g is None for g in shared_kl[len(old_params):])
            gradient_sets = dict(off=native_grads,full=full_grads,function=shared_kl[:len(old_params)])
            for kind, gradients in gradient_sets.items():
                for name,ps,grads,_ in r.split_gradients(state['old_layout'],gradients):
                    expected = probe['shared'][name][kind+'_gradient']
                    for value,want in zip(grads,expected):
                        assert (value is None)==(want is None)
                        if want is not None:
                            if kind=='off':
                                assert torch.equal(value,want)
                            else:
                                compare(value,want)
    finally:
        h.runtime().load_history().load_bundle(state,saved)
    assert r.rng_mark()==current

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


def audit_actions(state, module, probe):
    import copy
    shadow = copy.deepcopy(module)
    shadow.load_state_dict(probe['pre_module'],strict=True)
    rows = rows_legal(state)
    y = state['labels_np'][rows]
    d = probe['details']
    independent = manual_action(shadow,d['z'],probe,d['jacobian'],d['gap'],probe['head'],'full',d)
    compare(independent['route'],d['route'])
    assert torch.equal(independent['choice'],d['choice'])
    compare(independent['candidates'],d['candidates'])
    mask = probe['native_mask']
    assert torch.equal(probe['native_dropout'][0]*mask,probe['native_dropout'][1])
    candidates = [d['z'], d['z']+d['candidates'][:,0], d['z']+d['candidates'][:,1]]
    replay = torch.stack([explicit_fp32_head(probe['head'],z,mask).log_softmax(1) for z in candidates],dim=1)
    assert torch.equal(replay,probe['actions'])
    p = array(replay)[rows]
    nll = -p[np.arange(len(rows))[:,None],np.arange(3),y[:,None]]
    route = array(d['route'])[rows]
    correction = (route*(nll-nll[:,:1])).sum(1)
    independently_balanced = sum(float(correction[y==c].mean()) for c in (0,1))/2
    np.testing.assert_allclose(independently_balanced,probe['risk'],atol=1e-9,rtol=1e-6)
    if state['fold']==2:
        assert np.intersect1d(rows,FOLD2_ROWS).size==0
        assert not d['candidates'][list(FOLD2_ROWS)].count_nonzero()
        assert torch.equal(d['route'][list(FOLD2_ROWS)],d['route'].new_tensor([[1,0,0],[1,0,0]]))
    return dict(legal=len(rows),mask_verified=True,exact_candidate_risk=True,candidate_independent=True,
                functional_independent=True)

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
        ps = state['old_layout'][name]
        full = r.flatten(ps,value['full_gradient'])
        func = r.flatten(ps,value['function_gradient']).double()*CFG['reference']['function_weight']
        prop = r.flatten(ps,value['proposal_gradient'])
        diag = value['report']
        independent_scale = diag['retention_scale']
        predicted = (full.double()+independent_scale*func).to(full.dtype)
        assert torch.equal(predicted,prop)
        actual = torch.linalg.vector_norm(prop.double()-full).item()
        assert actual <= CFG['reference']['function_norm_budget']*torch.linalg.vector_norm(full).item()
        assert actual == diag['retention_injected']
    gains = audit_actions(state,module,probe)
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
                          gradients={}, updates={}, protection={},
                          function=[],routing=[],ema_best=None)
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
            g['function'].append(s['function'])
            g['routing'].append(s['routing'])
            assert s['aux_budget']['ratio'] <= CFG['loss']['aux_ratio_max']
            assert abs(s['routing']['expected_nll']-s['routing']['keep_nll']-s['route_risk'])<1e-6
            for name, values in s['shared'].items():
                assert 0 <= values['accepted_reference_distance'] <= values['radius']
                assert 0 <= values['proposal_scale'] <= 1 and 0 <= values['moment_scale'] <= 1
                assert values['retention_injected'] <= CFG['reference']['function_norm_budget']*values['gradient_full']
                assert values['actual_step'] >= 0 and values['native_proposal_step'] >= 0
                assert all(0 <= block['drift'] <= block['radius'] for block in values['moments'].values())
                g['protection'].setdefault(name,[]).append(values)
            for name, values in s['added'].items():
                assert values['class_only_verified'] if 'class_only_verified' in values else True
                g['gradients'].setdefault(name,[]).append(values['classification'])
                assert values['actual_step_norm'] >= 0 and 0 <= values['actual_update_fraction'] <= 1
                assert (values['actual_step_norm'] > 0) == (values['actual_update_fraction'] > 0)
                g['updates'].setdefault(name,[]).append(values['actual_step_norm'])
            last_trade[group] = s
    assert len(masks) == len(rows), 'Native classification dropout did not evolve'
    for group, g in checked.items():
        status = latest['track'][group]
        assert g['best_epoch'] == status['best_epoch'] and g['wait'] == status['wait']
        assert g['best_correct']/len(ctx.reference['test_np']) == status['best_acc']
        assert g['count'] == status['terminal_epoch']-r.START
        assert h.base_runtime().rng_sha(latest['rng'][group]) == last_trade[group]['rng_after_eval']
        g['terminal_ema_sha'] = prior_ema[group]
        mean_func = {key:float(np.mean([row[key] for row in g['function']])) for key in g['function'][0]}
        g['function'] = dict(mean=mean_func,first=g['function'][0],last=g['function'][-1])
        route = g['routing']
        g['routing'] = dict(counts=np.array([x['route_counts'] for x in route]).sum(0).tolist(),
                            mean_prob=np.array([x['mean_prob'] for x in route]).mean(0).tolist(),
                            mean_entropy=float(np.mean([x['entropy'] for x in route])),
                            mean_hard_minus_expected_nll=float(np.mean([x['hard_nll']-x['expected_nll'] for x in route])),
                            mean_eligibility=float(np.mean([x['eligible'] for x in route])))
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
                probability_mean_difference=float(np.abs(ref-off)[permitted,0].mean()),
                mean_true_margin_diff=float(np.mean((student['off_gap']-reference['gap'])[permitted]*(1-2*truth[permitted]))),
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
    limit = CFG['decision']
    for group,values in combined['mechanisms'].items():
        pairs = combined['paired'][f'{group}-B0']
        full,no2 = pairs['full'],pairs['no_fold2']
        delta = full['delta_pp']
        files = [a for a in audits if a['group']==group]
        complete = len(files)==10 and all(a['status']=='PASS' for a in files) and protection['pass_']
        performance = dict(audit=complete,acc=delta['acc']>0,ba=delta['hard_auc']>=0,
                           net=full['changes']['all']['net']>0,prob_auc=delta['prob_auc']>=limit['paired_prob_auc_min_pp'],
                           asd_sen=delta['sen']>=limit['paired_sen_min_pp'],
                           no_fold2=no2['delta_pp']['acc']>0 and no2['changes']['all']['net']>0)
        direct = values['stages']['B']['all']['net']>0 and bool(values['attribution']['new_residual_correction'])
        alignment = values['alignment']['same_epoch_b0']
        functional = (alignment['folds']==10 and alignment['delta_pp']['acc']>=limit['functional_min_same_epoch_acc_pp']
                      and alignment['probability_mean_difference']<=limit['functional_max_same_epoch_abs_asd_shift'])
        cohort = values['strata']['all']
        mechanism = dict(audit=complete,native_reference=all(a['reference_pass'] for a in files),
                         basis=all(a['basis']['rank']==16 for a in files),
                         end_to_end=all(all(x['nonzero']>0 for x in a['training']['gradients'].values()) for a in files),
                         probability=cohort['probability_change']['mean']>0,residual=cohort['residual_nonzero']>0,
                         optimizer=all(a['latest']['actual_budget'] and a['latest']['functional_independent'] for a in files),
                         exact_action_risk=all(a['latest']['candidate_independent'] for a in files),
                         functional_protection=functional,new_direct_correction=direct)
        answer[group] = dict(performance_go=all(performance.values()),mechanism_go=all(mechanism.values()),
                             new_residual_correction=direct,functional_protection=functional,
                             go=all(performance.values()) and all(mechanism.values()),
                             performance_checks=performance,mechanism_checks=mechanism)
    pair = combined['paired']['G1-G2']
    full,no2 = pair['full'],pair['no_fold2']
    win,loss = (full['wtl']['acc'][key] for key in ('win','loss'))
    medical_checks = dict(acc=full['delta_pp']['acc']>0,ba=full['delta_pp']['hard_auc']>=0,net=full['changes']['all']['net']>0,
                          wins=win-loss>=limit['medical_min_acc_wins_over_losses'],
                          no_fold2=no2['delta_pp']['acc']>0 and no2['changes']['all']['net']>0,
                          leave_max=pair['max_contribution_removed']['acc']['delta_pp']['acc']>0,
                          audit=all(value['performance_checks']['audit'] for value in answer.values()))
    return dict(decision='GO' if answer['G1']['go'] else 'NO-GO',groups=answer,
                medical_semantic_contribution=all(medical_checks.values()),medical_checks=medical_checks)


def add_alignment_stats(audits, combined):
    for group, mech in combined['mechanisms'].items():
        parts = [a for a in audits if a['group']==group]
        aligned = {}
        for label in ('same_epoch_best','same_epoch_b0'):
            rows = [p[label] for p in parts if p[label] is not None]
            stages = {c:{k:sum(x['changes'][c][k] for x in rows) for k in ('corrected','harmed','net')}
                      for c in ('asd','td','all')}
            aligned[label] = dict(
                folds=len(rows),off_change=stages,
                residual_off_pct={k:float(np.mean([100*x['metrics_off'][k] for x in rows])) if rows else None for k in AUC_KEYS},
                reference_pct={k:float(np.mean([100*x['metrics_ref'][k] for x in rows])) if rows else None for k in AUC_KEYS},
                delta_pp={k:float(np.mean([100*(x['metrics_off'][k]-x['metrics_ref'][k]) for x in rows])) if rows else None
                          for k in AUC_KEYS},
                probability_mean_difference=float(np.mean([x['probability_mean_difference'] for x in rows])) if rows else None,
                mean_true_margin_diff=float(np.mean([x['mean_true_margin_diff'] for x in rows])) if rows else None)
        mech['alignment'] = aligned
        mech['legal_function_mean'] = {k:float(np.mean([a['training']['function']['mean'][k] for a in parts]))
                                       for k in parts[0]['training']['function']['mean']}
        offset = [p['best_epoch']-p['baseline']['best_epoch'] for p in parts]
        mech['best_epoch_shift'] = dict(mean=float(np.mean(offset)),minimum=min(offset),maximum=max(offset),
                                        ties=sum(e==0 for e in offset))

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
