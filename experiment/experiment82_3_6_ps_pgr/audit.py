#!/usr/bin/env python3
"""Independent PS-PGR equations, metrics, checkpoint replay, and research decision."""
from __future__ import annotations

import sys
if __name__ == '__main__':
    sys.modules['audit'] = sys.modules[__name__]

import run as r  # Sets physical GPU0 before torch import.
import history as h

import argparse
import csv
import json
import math
import subprocess
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F

from ps_pgr import EMA_BUFFERS, FOLD2_ROWS, SPECTRUM_BUFFERS


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


def graph(state, module, intervention='full', *, training=False):
    module.train(training)
    seen = []

    def inject(head, args):
        before = r.rng_sha()
        with torch.enable_grad():
            base_features = args[0].detach().requires_grad_(True)
            contrast = independent_gap(head, base_features)
            jac, = torch.autograd.grad(contrast.sum(), base_features)
        enhanced, details = module(args[0], jac.detach(), intervention=intervention)
        details['raw_gap_off'] = contrast.detach()
        assert r.rng_sha() == before
        seen.append(details)
        return enhanced, args[1]

    def confirm_head_input(_head, args):
        assert len(seen) == 1 and args[0] is seen[0]['z_enhanced']

    head = state['cam'].Outputlayer
    handles = (head.register_forward_pre_hook(inject), head.register_forward_pre_hook(confirm_head_input))
    try:
        with h.count_calls(state), torch.set_grad_enabled(training):
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
    assert [float(module.tau_s), float(module.tau_l)] == checkpoint['scales']
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


def independent_gap(head, z):
    ln, linear, gelu, _ = head.mlp_head
    logits = head.classifier(gelu(linear(ln(z))))
    return logits.unbind(1)[0] - logits.unbind(1)[1]


@torch.no_grad()
def analytic_jacobian(head, z):
    """Closed form for the native LayerNorm–affine–exact GELU–affine gap."""
    ln, first, gelu, _ = head.mlp_head
    assert gelu.approximate == 'none'
    x = z.double()
    centered = x - x.mean(1, keepdim=True)
    inv_std = (centered.square().mean(1, keepdim=True) + ln.eps).rsqrt()
    white = centered * inv_std
    u = F.linear(white*ln.weight.double() + ln.bias.double(), first.weight.double(), first.bias.double())
    slope = (1 + (u / math.sqrt(2)).erf())/2 + u * (-u.square()/2).exp() / math.sqrt(2*math.pi)
    w = head.classifier.weight.double()
    up = ((w[0]-w[1]) * slope) @ first.weight.double() * ln.weight.double()
    return inv_std * (up - up.mean(1, keepdim=True) - white * (up*white).mean(1, keepdim=True))


@torch.no_grad()
def semantic_formulas(module, z):
    eps = r.CFG['epsilon']
    basis = module.U
    white = torch.div(torch.mm(z-module.mu, basis), torch.sqrt(module.lam+eps))
    inputs = torch.stack((white, white.abs(), module.loglam.expand_as(white), module.nu.expand_as(white)), -1)
    bembed = F.linear(basis.T, module.direction.weight, module.direction.bias)
    tokens = F.layer_norm(bembed + F.linear(inputs, module.evidence.weight, module.evidence.bias),
                          (32,), module.norm.weight, module.norm.bias, module.norm.eps)
    q = module.q0
    keys = F.linear(tokens, module.key.weight)
    inner = torch.matmul(keys, q.T).transpose(1,2) / math.sqrt(32)
    null_score = torch.mv(q,module.null)[None,:,None].expand(z.shape[0],-1,-1) / math.sqrt(32)
    values = torch.cat((inner,null_score), -1) / r.CFG['temperature']
    att = (values - torch.logsumexp(values, -1, keepdim=True)).exp()
    mu0,mu1 = module.class_mean
    var0,var1 = module.class_var
    raw_llr = ((white-mu1).square()/var1 + var1.log() - ((white-mu0).square()/var0 + var0.log())) / 2
    limit = r.CFG['class_statistics']['llr_bound']
    llr = limit * (raw_llr/limit).tanh()
    mlp = module.spectral
    spec = F.linear(F.gelu(F.linear(inputs[...,1:],mlp[0].weight,mlp[0].bias)),mlp[2].weight,mlp[2].bias)
    w = spec[...,0].softmax(-1)
    reliability = (1+(16*w).tanh())/2
    cover, null, semantics = [], [], []
    for cls in ('ASD','TD'):
        ids = r.CFG['prompt_rows_0based'][cls]
        c = 1-(1-att[:,ids,:-1]).prod(1)
        n = att[:,ids,-1].mean(1)
        cover.append(c)
        null.append(n)
        semantics.append(c/(c.sum(1,keepdim=True)+eps)*(1-n[:,None]))
    coverage = torch.stack(cover,1)
    null = torch.stack(null,1)
    semantic = torch.stack(semantics,1)
    raw_s = semantic[:,0] - semantic[:,1]
    return dict(hh=white, tokens=tokens, attention=att, coverage=coverage, null=null, semantic=semantic,
                raw_s=raw_s, raw_llr=raw_llr, llr=llr, w_sp=w, reliability=reliability)


@torch.no_grad()
def verify_calibration(module, spectrum, rows):
    assert len(rows) and len(np.unique(rows)) == len(rows)
    cam = spectrum['warmup_cam'][rows].to(module.U.device)
    parts = semantic_formulas(module, cam)
    hp = r.CFG['fusion']
    want = [max(floor, float(np.quantile(parts[k].abs().cpu().numpy(),hp['legal_pooled_abs_quantile'])))
            for k,floor in [('raw_s',hp['semantic_scale_floor']),('llr',hp['llr_scale_floor'])]]
    actual = [float(module.tau_s),float(module.tau_l)]
    np.testing.assert_allclose(actual, want, atol=1e-7, rtol=2e-6)
    return {'scales':actual,'independent_scales':want,'rows_sha256':r.sha(rows),
            'fitted_patients':len(rows)}


@torch.no_grad()
def check_formulas(module, d, head, intervention='full'):
    z = d['z']
    e = semantic_formulas(module,z)
    # Check attention/statistics independently below. Use their stored FP32
    # values here so the small tau_s cannot magnify differences in softmax rounding.
    s = (d['raw_s']/module.tau_s).tanh()
    l = (d['llr']/module.tau_l).tanh()
    rel, null = d['reliability'], d['null']
    nf = null[:,None,:].expand(-1,16,-1)
    fused_in = torch.cat((torch.stack((s,l,s.abs(),l.abs(),s*l),-1),nf,rel[...,None]),-1)
    f0, _, f1 = module.fusion_net
    a_logit = F.linear(F.gelu(F.linear(fused_in,f0.weight,f0.bias)),f1.weight,f1.bias).squeeze(-1)
    lo,hi=(r.CFG['fusion'][k] for k in ('alpha_min','alpha_max'))
    alpha = lo + (hi-lo)*a_logit.sigmoid()
    difference = (alpha*s + (1-alpha)*l)*rel*(1-null.mean(1,keepdim=True))
    if intervention == 'evidence_off':
        difference.zero_()
    gate_in = torch.cat((F.layer_norm(z,(144,),eps=r.CFG['signed_gate']['layer_norm_eps']),
                         difference,rel,null),1)
    g0, _, g1 = module.signed_gate
    g_pre = F.linear(F.gelu(F.linear(gate_in,g0.weight,g0.bias)),g1.weight,g1.bias)
    gate = torch.where(g_pre>0,g_pre.tanh(),0.)
    if intervention == 'gate_off':
        gate.zero_()
    analytic = analytic_jacobian(head,z)
    jac = d['jacobian']
    cfg = r.CFG['preflight']
    with torch.enable_grad():
        leaf = z.detach().requires_grad_(True)
        native_gap = independent_gap(head,leaf)
        native_j, = torch.autograd.grad(native_gap.sum(),leaf)
        q = z.detach().double().requires_grad_(True)
        ln, first, _, _ = head.mlp_head
        weight = head.classifier
        norm = F.layer_norm(q,ln.normalized_shape,ln.weight.detach().double(),ln.bias.detach().double(),ln.eps)
        middle = F.gelu(F.linear(norm,first.weight.detach().double(),first.bias.detach().double()))
        top = F.linear(middle,weight.weight.detach().double(),weight.bias.detach().double())
        jac64, = torch.autograd.grad((top[:,0]-top[:,1]).sum(),q)
    assert torch.equal(jac,native_j.detach()) and torch.equal(d['raw_gap_off'],native_gap.detach())
    torch.testing.assert_close(jac64,analytic,atol=1e-12,rtol=1e-10)
    projected = torch.mm(jac,module.U)
    assert torch.equal(d['jacobian_projected'],projected)
    j64,u64 = jac.double(),module.U.double()
    summed = torch.sum(j64[...,None]*u64[None],dim=1)
    projected64 = j64 @ u64
    torch.testing.assert_close(summed,projected64,atol=1e-12,rtol=1e-10)
    # A dot product of n terms in the original FP32 precision has a
    # componentwise forward error bounded by gamma_n * sum(abs(products)).
    rounding = torch.finfo(jac.dtype).eps / 2
    n = j64.shape[1]
    gamma = n*rounding/(1-n*rounding)
    product_bound = gamma*(j64.abs()@u64.abs())
    assert (projected.double()-projected64).abs().le(product_bound).all()
    native_projection = d['jacobian_projected'].double()
    denom = (native_projection.square().sum(1,keepdim=True)
             + r.CFG['sensitivity']['j_epsilon']**2).sqrt()
    unit = (native_projection/denom).to(z.dtype)
    spectral = difference * gate * unit
    raw = spectral @ module.U.T
    length = z.double().square().sum(1,keepdim=True).sqrt()
    rlen = raw.double().square().sum(1,keepdim=True).sqrt()
    bound = r.CFG['residual_limit']
    radial = bound*length*(rlen/(bound*length.clamp_min(r.CFG['epsilon']))).tanh()
    radial *= length/(length+torch.finfo(z.dtype).tiny)
    delta = (radial * raw.double()/torch.where(rlen==0,1.,rlen)).to(z.dtype)
    if module.fold==2:
        delta[list(FOLD2_ROWS)] = 0
    if intervention=='residual_off':
        delta.zero_()
    quantities = {**e,'signed_semantic':s,'signed_statistical':l,'alpha':alpha,'difference':difference,
                  'gate':gate,'jacobian_projected':projected,'jacobian_unit':unit,
                  'spectral_residual':spectral,'raw':raw,'delta':delta,'z_enhanced':z+delta}
    for k,v in quantities.items():
        torch.testing.assert_close(d[k],v,atol=cfg['independent_formula_atol'],rtol=cfg['independent_formula_rtol'],
                                   msg=lambda m:f'{k}: {m}')
    assert d['attention'].shape == (871,8,17) and d['semantic'].shape == (871,2,16)
    assert d['difference'].shape == d['gate'].shape == (871,16)
    assert d['delta'].shape == z.shape == (871,144)
    assert not d['jacobian'].requires_grad
    assert d['delta'].double().norm(dim=1).le((bound+cfg['residual_bound_roundoff'])*length[:,0]).all()
    if module.fold==2:
        assert not d['delta'][list(FOLD2_ROWS)].count_nonzero()
    raw_dir_effect = d['spectral_residual'] * d['jacobian_projected']
    active = raw_dir_effect != 0
    assert torch.equal(raw_dir_effect[active].sign(),d['difference'][active].sign())
    span = d['delta'].double()
    u = module.U.double()
    assert (span-(span@u)@u.T).norm() <= 1e-6*span.norm()
    return {'chained_jacobian_fp64_max_abs_error':float((analytic-jac64).abs().max()),
            'native_fp32_jacobian_exact':True,
            'projected_fp32_max_abs_roundoff':float((projected.double()-projected64).abs().max()),
            'residual_formula_max_abs_error':float((delta-d['delta']).abs().max())}


@torch.no_grad()
def finite_difference(state, module, d):
    rows = legal_rows(state)
    labels = state['labels_np'][rows]
    hp = r.CFG['sensitivity']
    panel = np.concatenate([rows[labels==c][:hp['finite_difference_n_per_class']] for c in (0,1)])
    assert not np.isin(panel,state['test_np']).any()
    p = d['jacobian_projected'][panel]
    unit = (p / p.norm(dim=1,keepdim=True).clamp_min(hp['j_epsilon'])) @ module.U.T
    z, jac = d['z'][panel], d['jacobian'][panel]
    head = state['cam'].Outputlayer
    eps = hp['finite_difference_step']
    base = independent_gap(head,z)
    plus = independent_gap(head,z+eps*unit)
    minus = independent_gap(head,z-eps*unit)
    expected = eps*(jac*unit).sum(1)
    torch.testing.assert_close(plus-base,expected,atol=hp['derivative_abs_tolerance'],rtol=hp['derivative_rel_tolerance'])
    torch.testing.assert_close((plus-minus)*.5,expected,atol=hp['derivative_abs_tolerance'],rtol=hp['derivative_rel_tolerance'])
    return {'rows_1based':(panel+1).tolist(),'epsilon':eps,'one_sided_max_abs_error':float((plus-base-expected).abs().max()),
            'central_max_abs_error':float(((plus-minus)*.5-expected).abs().max())}


@torch.no_grad()
def historical_probe(state, spectrum):
    """Lightweight frozen DARG sensitivity observation with a fixed local probe."""
    r0 = r.rng_mark()
    hist = h.runtime().load_history()
    restore = hist.state_bundle(state)
    fold = state['fold']
    try:
        ckpt = r.torch_load(history_paths(fold)[0])
        assert ckpt['identity'] == state['rudag_identity']
        hist.load_bundle(state,ckpt['best_state'])
        z,out = h.warmup_cam(state)
        assert np.array_equal(probability(out,state), ckpt['test_probability'])
        head = state['cam'].Outputlayer
        z = z.to(state['x'].device)
        basis = spectrum['spectrum']['U'].to(device=z.device,dtype=z.dtype)
        jac = analytic_jacobian(head,z)
        projected = jac @ basis.double()
        rho = projected.square().sum(1)/(jac.square().sum(1)+r.CFG['epsilon'])
        assert rho.ge(0).all() and rho.le(1+2e-7).all()
        eps = r.CFG['sensitivity']['finite_difference_step']
        probe = (eps*projected/projected.norm(dim=1,keepdim=True).clamp_min(r.CFG['sensitivity']['j_epsilon'])) @ basis.double().T
        gap = independent_gap(head,z)
        change = independent_gap(head,z+probe.to(z.dtype)) - gap
        ids = state['test_np']
        ids = ids[~np.isin(ids, FOLD2_ROWS)] if fold==2 else ids
        y = state['labels_np'][ids]
        correct = out['logp'][ids].argmax(1).cpu().numpy()==y
        truth = (gap[ids].cpu().numpy()*(1-2*y))
        increment = change[ids].cpu().numpy()*(1-2*y)
        strata = {key:distribution(increment[mask]) if mask.any() else None
                  for key,mask in [('correct',correct),('wrong',~correct),('low_margin',truth<2),('high_margin',truth>=2)]}
        dot = (probe*jac).sum(1)
        cosine = dot/(probe.norm(dim=1)*jac.norm(dim=1)).clamp_min(1e-12)
        return {'probe_step':eps, 'rho_training':distribution(rho[legal_rows(state)].cpu().numpy()),
                'rho_oof':distribution(rho[ids].cpu().numpy()),
                'probe_cam_ratio_pct':distribution((100*probe[ids].norm(dim=1)/z[ids].double().norm(dim=1).clamp_min(1e-300)).cpu().numpy()),
                'probe_alignment':distribution(cosine[ids].cpu().numpy()),
                'probe_gap_abs_change':distribution(change[ids].abs().cpu().numpy()),
                'probe_linearization_max_error':float((change[ids]-dot[ids]).abs().max()),
                'probe_margin_strata':strata,'label_rows':len(ids)}
    finally:
        hist.load_bundle(state,restore)
        assert r.rng_mark()==r0


def check_interventions(state,module,text,group):
    before = r.rng_mark()
    outputs = {'T0':graph(state,module)}
    for tag,mode in [('T1','residual_off'),('T2','gate_off'),('T3','evidence_off')]:
        outputs[tag] = graph(state,module,mode)
    swapped = 'q_medical' if group=='ANON' else 'q_anonymous'
    with replace_query(module,text[swapped]):
        outputs['T4'] = graph(state,module)
        disabled = graph(state,module,'evidence_off')
    for tag in ('T1','T2','T3'):
        d = outputs[tag]['details']
        assert not d['delta'].count_nonzero() and torch.equal(d['z'],d['z_enhanced'])
        assert torch.equal(outputs[tag]['logp'],outputs['T1']['logp'])
    assert torch.equal(disabled['logp'],outputs['T3']['logp'])
    for name in ('jacobian','llr','w_sp'):
        assert torch.equal(outputs['T4']['details'][name],outputs['T0']['details'][name])
    assert r.rng_mark() == before
    return outputs


def historical_log_rows(fold):
    path = h.runtime().log_path(r.BASE_GROUP,fold)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({rec['epoch'] for rec in rows}) == len(rows)
    return {rec['epoch']:rec for rec in sorted(rows,key=lambda rec:rec['epoch'])}


def reference_forward(state, training):
    """Historical runner with observation-only hooks, independent of the PS-PGR runner."""
    head = state['cam'].Outputlayer
    captured = {}

    def get_feature(_module, args):
        assert 'z' not in captured
        captured['z'] = args[0]

    def get_logits(_module, _args, value):
        assert 'logits' not in captured
        captured['logits'] = value

    handles = [head.register_forward_pre_hook(get_feature),head.classifier.register_forward_hook(get_logits)]
    try:
        with torch.set_grad_enabled(training):
            out = h.runtime().forward(state,r.BASE_GROUP,training=training)
    finally:
        for handle in handles:
            handle.remove()
    assert captured['z'].shape == (r.CFG['subjects'],144) and head.training == training
    return out,captured['z'],captured['logits']


def independent_gradient_sha(state):
    leaves = {key:dict(state[key].named_parameters()) for key in ('model','cam')}
    return h.sha({key:{name:tensor.grad for name,tensor in cells.items()} for key,cells in leaves.items()})


def independent_base_state(state):
    live = {name:state[name].state_dict() for name in ('model','cam',*r.HIST_OPTS)}
    return {name:h.sha(contents) for name,contents in live.items()}


def native_record(state, out, logits, base_grad, update_grad):
    return {'logit_sha':h.sha(logits),'logp_sha':h.sha(out['logp']),
            'losses':{key:float(out[key].detach().cpu()) for key in (*r.DARG_LOSSES,'total')},
            'native_gradient_sha':base_grad,'optimizer_gradient_sha':update_grad,
            'state':independent_base_state(state)}


def primal_step(state):
    """One read-only reference update, with the unchanged E74.4 objective and optimizers."""
    h.runtime().load_history().zero_optimizers(state)
    out,_,logits = reference_forward(state,True)
    out['total'].backward()
    native_grad = independent_gradient_sha(state)
    for name in r.HIST_OPTS:
        state[name].step()
    proof = native_record(state,out,logits,native_grad,native_grad)
    train_rng = r.rng_mark()
    del out, logits
    evaluated,_,_ = reference_forward(state,False)
    evaluation_rng = r.rng_mark()
    assert train_rng == evaluation_rng
    return proof,probability(evaluated,state),evaluation_rng,h.sha(evaluated['logp'])


def fixed_logits(head, features):
    normalization,linear,activation,_ = head.mlp_head
    x = F.layer_norm(features,normalization.normalized_shape,normalization.weight.detach(),
                     normalization.bias.detach(),normalization.eps)
    hidden = F.linear(x,linear.weight.detach(),linear.bias.detach())
    hidden = F.gelu(hidden,approximate=activation.approximate)
    return F.linear(hidden,head.classifier.weight.detach(),head.classifier.bias.detach())


def residual_formulas(state,module,z):
    ids = legal_rows(state)
    y = state['y'][ids]
    head = state['cam'].Outputlayer
    point = z.detach()
    with torch.enable_grad():
        leaf = point.requires_grad_(True)
        raw_off = independent_gap(head,leaf)
        j, = torch.autograd.grad(raw_off.sum(),leaf)
    point = point.detach()
    _, detail = module(z if module.protected_e2e else point,j.detach())
    detail['raw_gap_off'] = raw_off.detach()
    raw_on = fixed_logits(head,point+detail['delta'])
    complement = torch.arange(len(ids),device=y.device)
    on = raw_on[ids]
    m_on = on[complement,y]-on[complement,1-y]
    m_off = raw_off.detach()[ids] * (1-2*y).to(raw_off.dtype)
    cfg = r.CFG['loss']
    floor = cfg['increment_weight_floor']
    confidence = torch.sigmoid(m_off-cfg['protection_margin_center'])
    weighting = floor+(1-floor)*torch.sigmoid(cfg['increment_margin_center']-m_off)
    patient = (F.softplus(m_on.neg())-F.softplus(m_off.neg()))*weighting
    inc = (patient[y == 0].mean()+patient[y == 1].mean())*.5
    protect = ((m_off-m_on).relu().square()*confidence).mean()
    ratio_sq = detail['delta'][ids].square().sum(dim=1)/(point[ids].square().sum(dim=1)+r.CFG['epsilon'])
    safe = (ratio_sq*confidence).mean()
    with torch.no_grad():
        off_scores = fixed_logits(head,point)[ids]
        assert torch.equal(off_scores[complement,y]-off_scores[complement,1-y],m_off)
        exact_ce = F.cross_entropy(on,y,reduction='none')
        torch.testing.assert_close(exact_ce,F.softplus(-m_on),atol=2e-7,rtol=2e-6)
    terms = {'inc':inc,'protect':protect,'safe':safe}
    loss = cfg['lambda_inc']*inc+cfg['lambda_protect']*protect+cfg['lambda_safe']*safe
    return loss,terms,detail


def reference_projection(baseline, proposed):
    """Independent verification of the mathematical projection and native application."""
    settings = r.CFG['shared_gradient']
    b = baseline.double()
    g = proposed.double()
    length2 = b @ b
    cross = b @ g
    if not torch.stack((length2,cross,g.norm())).isfinite().all():
        raise FloatingPointError('Nonfinite shared projection inputs')
    if float(length2) <= settings['minimum_baseline_norm_squared']:
        return baseline.clone()
    if float(cross) < 0:
        g = g-(cross/(length2+settings['denominator_epsilon']))*b
        g = g-((g @ b).clamp_max(0)/length2)*b
    n = float(g.norm())
    if n == 0:
        return baseline.clone()
    maximum = settings['ratio_max']*float(length2.sqrt())
    if n > maximum:
        g = g*(maximum/n*(1-1e-4))
    v = (b+g).to(baseline.dtype)
    applied = v.double()-b
    actual_cross = float(applied @ b)
    if actual_cross < 0:
        repaired = max(-actual_cross/float(length2),0.)+4*torch.finfo(baseline.dtype).eps
        v = (b+g*(1-1e-4)+repaired*b).to(baseline.dtype)
        applied = v.double()-b
    if float(applied @ b) < 0 or float(applied.norm()) > maximum or not torch.isfinite(v).all():
        return baseline.clone()
    return v


def check_projection_cases():
    cases = [
        ([1.,0.],[-5.,2.]),
        ([1.,2.],[-1.,-2.]),
        ([1.,2.],[2.,7.]),
        ([1.,2.],[.001,.002]),
        ([0.,0.],[5.,-3.]),
        ([1e-16,0.],[1.0,0.]),
        ([.01,-1.,.14],[.15,2.,.03]),
    ]
    records = []
    for initial, proposed in cases:
        baseline = torch.tensor(initial,device='cuda',dtype=torch.float32)
        residual = torch.tensor(proposed,device='cuda',dtype=torch.float32)
        result,row = r.project_increment(baseline,residual)
        assert torch.equal(result,reference_projection(baseline,residual))
        b = baseline.cpu().numpy().astype(np.float64)
        g = residual.cpu().numpy().astype(np.float64)
        actual = result.cpu().numpy().astype(np.float64)-b
        bb = b @ b
        budget = r.CFG['shared_gradient']['ratio_max'] * np.sqrt(bb)
        if bb <= r.CFG['shared_gradient']['minimum_baseline_norm_squared']:
            assert np.array_equal(actual,np.zeros_like(b))
        else:
            tangent = g-min(float(g @ b)/bb,0.)*b
            norm = np.linalg.norm(tangent)
            if norm > budget:
                tangent *= budget/norm
            np.testing.assert_allclose(actual,tangent,rtol=4e-4,atol=7e-7)
        assert (actual @ b) >= 0 and np.linalg.norm(actual) <= budget+1e-16
        records.append(row)
    try:
        r.project_increment(torch.tensor([1.]),torch.tensor([float('nan')]))
    except FloatingPointError:
        pass
    else:
        raise AssertionError('Projection accepted nonfinite increment')
    assert any(v['conflict'] for v in records) and any(v['capped'] for v in records)
    return records


def replay_step(state,module,opt,spectrum):
    old_opts = [state[name] for name in r.HIST_OPTS]
    for optimizer in (*old_opts,opt):
        optimizer.zero_grad(set_to_none=True)
    module.train()
    primary,z,logits = reference_forward(state,True)
    native_rng = r.rng_mark()
    side,parts,details = residual_formulas(state,module,z)
    check_formulas(module,details,state['cam'].Outputlayer)
    owners = tuple(module.parameters())
    old = tuple(p for name in ('model','cam') for p in state[name].parameters() if p.requires_grad)
    outputs = torch.autograd.grad(side,(*owners,*old),allow_unused=True,retain_graph=module.protected_e2e)
    alt_grads = dict(zip(old,outputs[len(owners):]))
    head = set(state['cam'].Outputlayer.parameters())
    assert all(alt_grads.get(p) is None for p in head)
    assert all(p.grad is None for p in (*owners,*old))
    if not module.protected_e2e:
        assert all(v is None for v in alt_grads.values())
    primary['total'].backward()
    historical_gradient = independent_gradient_sha(state)
    if module.protected_e2e:
        for optimizer in old_opts:
            for group in optimizer.param_groups:
                active = [p for p in group['params'] if p.grad is not None and p not in head]
                if not active:
                    continue
                b = torch.cat([p.grad.flatten() for p in active])
                g = torch.cat([(torch.zeros_like(p) if alt_grads.get(p) is None else alt_grads[p]).flatten()
                               for p in active])
                final = reference_projection(b,g)
                q = final.double()-b.double()
                assert float(q @ b.double()) >= 0
                assert float(q.norm()) <= r.CFG['shared_gradient']['ratio_max']*float(b.double().norm())
                cursor = 0
                for p in active:
                    p.grad.copy_(final[cursor:cursor+p.numel()].reshape_as(p))
                    cursor += p.numel()
    update_gradient = independent_gradient_sha(state)
    for p,g in zip(owners,outputs):
        p.grad = g
    for optimizer in (*old_opts,opt):
        optimizer.step()
    proof = native_record(state,primary,logits,historical_gradient,update_gradient)
    losses = {key:float(value.detach().cpu()) for key,value in parts.items()}
    losses['residual_total'] = float(side.detach().cpu())
    evaluated = graph(state,module)
    native_off,_,_ = reference_forward(state,False)
    p,(cam,obs) = probability(evaluated,state),observe(state,evaluated,spectrum)
    old_ema = h.calibration_state(module)
    set_ema(module,ema_update(old_ema,obs))
    assert r.rng_mark() == native_rng
    return dict(native=proof,losses=losses,rng=native_rng[0],rng_streams=native_rng[1],
                digests=r.component_digests(r.bundle(state,module,opt)),probability_sha=r.sha(p),
                legal_cam_sha=r.sha(cam),ema_before_sha=r.sha(old_ema),ema_after_sha=r.sha(h.calibration_state(module)),
                observation_sha=r.sha(obs),off_probability_sha=h.sha(probability(native_off,state)),
                off_logp_sha=h.sha(native_off['logp']))


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
        raw_gap = (fresh['logp'][:,0]-fresh['logp'][:,1])[ids].cpu().tolist()
    finally:
        history.load_bundle(state, before)
    assert r.rng_mark() == rng
    return {'group': 'DARG', 'fold': fold, 'best_epoch': saved['best_epoch_1based'],
            'end_epoch': int(table['epochs_executed']), 'metrics': metrics, 'rows_1based': (ids+1).tolist(),
            'labels': labels.tolist(), 'prediction': prediction.tolist(), 'probability': p.tolist(),
            'raw_gap':raw_gap,'historical_best_reproduced':True}


def check_gradient_row(row, end_to_end):
    assert row['head_residual_grad_count'] == 0
    assert all(np.isfinite(x) and x >= 0 for x in row['residual_grads'].values())
    assert all(np.isfinite(x) for x in (*row['losses'].values(),*row['native']['losses'].values()))
    assert row['losses']['protect'] >= 0 and row['losses']['safe'] >= 0
    assert np.isfinite(row['mean_margin_gain_legal'])
    if not end_to_end:
        assert row['historical_residual_grad_count'] == 0 and not row['shared_grads']
        assert row['native']['native_gradient_sha'] == row['native']['optimizer_gradient_sha']
        return
    assert row['shared_grads'] and row['historical_residual_grad_count'] > 0
    settings = r.CFG['shared_gradient']
    for entry in row['shared_grads'].values():
        b, a, inner = (entry[k] for k in ('base_norm','applied_norm','applied_dot'))
        proposal = entry['raw_norm']
        assert 0 <= a <= settings['ratio_max']*b+1e-15 and inner >= 0
        assert inner <= b*a*(1+1e-9)+1e-20
        assert entry['conflict'] == (entry['raw_dot'] < 0 and not entry['baseline_tiny'])
        assert abs(entry['applied_ratio']-(a/b if b else 0)) < 1e-15
        assert entry['capped'] == (entry['projected_norm'] > settings['ratio_max']*b)
        assert np.isfinite(entry['relative_parameter_drift_from_initial'])
        if b == 0 or entry['baseline_tiny'] or entry['numerical_zero']:
            assert a == 0 and inner == 0
        if b*proposal:
            assert abs(entry['pre_cosine']-entry['raw_dot']/(b*proposal)) < 1e-14
            assert abs(entry['pre_cosine']) <= 1+1e-12


def protection_replay(state,task,best,latest,records):
    """Read-only replay of the entire protected native recurrence and its archived markers."""
    fold = state['fold']
    old_best_path,old_latest_path = history_paths(fold)
    archived_best,archived_latest = map(h.torch_load,(old_best_path,old_latest_path))
    archived = h.historical_task(fold)
    assert h.sha_file(old_best_path) == archived['checkpoint_sha256']
    assert archived_best['identity'] == archived_latest['identity'] == task['identity']
    historical_trace = {row['epoch']:row for row in archived['rng_trace']}
    historical_log = historical_log_rows(fold)
    b0_best = archived_best['best_epoch_1based']
    b0_end = archived_latest['next_epoch']
    terminal = task['end_epoch']
    assert b0_end == archived['epochs_executed'] and b0_best == archived['best_epoch_1based']
    base_checks, archived_checks = 0, 0
    labels = state['labels_np'][state['test_np']]
    for epoch in range(r.START+1,max(terminal,b0_end)+1):
        before = r.rng_mark()
        proof,off,after,all_logp = primal_step(state)
        if epoch in historical_trace:
            assert historical_trace[epoch] == dict(zip(r.RNG_KEYS,(epoch,before[0],after[0],after[0])))
        if epoch <= terminal:
            row = records[epoch-r.START-1]
            assert row['native'] == proof, (fold,epoch,'DARG native parameter/optimizer/gradient/logits certificate')
            assert row['off_probability_sha'] == h.sha(off) and row['off_logp_sha'] == all_logp
            assert (row['rng_before'],row['rng_before_streams']) == before
            assert (row['rng_after_eval'],row['rng_after_streams']) == after
            base_checks += 1
        if epoch in historical_log:
            rec = historical_log[epoch]
            assert proof['losses']['total'] == rec['loss']
            reconstructed = independent_metrics(off,labels)
            reconstructed['prob_auc'] = float(roc_auc_score(labels,off[:,1]))
            same_metrics(reconstructed,rec['metrics'])
            native_gate,native_scorer = h.runtime()._total_loss_gradient_snapshot(state)
            assert [abs(x) for x in native_gate] == rec['total_loss_gate_grad_abs']
            assert native_scorer == rec['total_loss_scorer_grad_norms']
            assert rec['rng'] == dict(zip(r.RNG_KEYS,(epoch,before[0],after[0],after[0])))
            archived_checks += 1
        if epoch == b0_best:
            saved = archived_best
            assert proof['state'] == {k:h.sha(v) for k,v in saved['best_state'].items()}
            assert proof['losses'] == saved['best_losses']
            assert np.array_equal(off,saved['test_probability']) and np.array_equal(off.argmax(1),saved['test_prediction'])
            assert after[0] == saved['best_training_rng_sha'] == saved['best_eval_rng_sha']
        if epoch == b0_end:
            saved = archived_latest
            assert proof['state'] == {k:h.sha(v) for k,v in saved['current'].items()}
            assert after == (r.base_runtime().rng_sha(saved['rng']),r.streams(saved['rng']))
        if epoch == task['best_epoch']:
            assert proof['state'] == {k:best['digests'][k] for k in proof['state']}
            assert np.array_equal(off,best['off_probability'])
        if epoch == terminal:
            assert proof['state'] == {k:latest['digests'][k] for k in proof['state']}
            assert np.array_equal(off,latest['off_probability'])
    assert base_checks == terminal-r.START and archived_checks == len(historical_log)
    return {'exact_native_epochs':base_checks,'replayed_reference_epochs':max(terminal,b0_end)-r.START,
            'archive_journal_milestones':archived_checks,'archive_best_parameters_optimizers_probability_exact':True,
            'archive_latest_parameters_optimizers_rng_exact':True,'maximum_parameter_drift_same_epoch':0.,
            'archived_best_epoch':b0_best,'archived_end_epoch':b0_end}


def aligned_to_b0_best(state,module,latest,reference,records):
    """Compare a residual from the same epoch with the actual historical Best state."""
    archived = h.torch_load(history_paths(state['fold'])[0])
    h.runtime().load_history().load_bundle(state,archived['best_state'])
    source = latest['b0_epoch_alignment']
    end = latest['next_epoch']
    if source:
        assert source['epoch'] == reference['best_epoch'] <= end
        module.load_state_dict(source['module'],strict=True)
        assert r.sha(h.calibration_state(module)) == source['ema_sha']
        kind, epoch = 'EXACT_SAME_EPOCH',source['epoch']
        assert records[epoch-r.START-1]['ema_before'] == source['ema_sha']
    else:
        assert end < reference['best_epoch']
        module.load_state_dict(latest['current']['ps_pgr'],strict=True)
        set_ema(module,latest['last_eval_ema'])
        kind,epoch = 'FROZEN_TERMINAL_ON_B0_BEST',end
    off = probability(graph(state,module,'residual_off'),state)
    full = probability(graph(state,module),state)
    assert np.array_equal(off,archived['test_probability'])
    if source:
        assert r.sha(off) == source['off_probability_sha']
        assert r.sha(full) == source['full_probability_sha']
        assert r.sha(full) == records[epoch-r.START-1]['probability_sha']
    ids = state['test_np']
    valid = ~np.isin(ids,FOLD2_ROWS) if state['fold']==2 else np.ones(len(ids),bool)
    y = state['labels_np'][ids]
    return {
        'comparison':kind,'b0_best_epoch':reference['best_epoch'],'residual_module_epoch':epoch,
        'off_probability_exact_b0':True,'full_metrics':independent_metrics(full,y),
        'direct':patient_changes(off[valid].argmax(1),full[valid].argmax(1),y[valid],ids[valid]+1),
        'asd_probability_change':distribution(np.abs(full[:,0]-off[:,0])),
    }


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
    comparator = 'G1' if task['group']=='G2' else r.anonymous_authorization() if task['group']=='ANON' else None
    if comparator:
        paired = {row['epoch']:row for row in map(json.loads,r.paths(comparator,state['fold'])[3].read_text().splitlines())}
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
            assert_ema_equal(ema, {k: best['current']['ps_pgr'][k] for k in EMA_BUFFERS})
            assert r.sha(moment) == r.sha(best_obs) and r.sha(best_cam) == row['legal_cam_sha']
            assert row['probability_sha'] == r.sha(best['probability']) and np.array_equal(pred,best['prediction'])
            assert row['off_probability_sha'] == r.sha(best['off_probability'])
            assert (prev, streams) == (best['rng_sha'], r.streams(best['rng']))
        if epoch == end:
            assert_ema_equal(ema, latest['last_eval_ema'])
        ema = ema_update(ema, moment)
        assert r.sha(ema) == row['ema_after']
        check_gradient_row(row,task['protected_e2e'])
    assert best['best_epoch'] == task['best_epoch'] == latest['best_epoch'] == chosen
    assert (max_correct, wait, end) == (latest['best_correct'], latest['wait'], latest['next_epoch'])
    assert wait > r.CFG['early_stop_wait_gt'] or end == r.LIMIT
    assert (prev, streams) == (latest['rng_sha'], r.streams(latest['rng']))
    assert records[-1]['probability_sha'] == r.sha(latest['probability'])
    assert records[-1]['off_probability_sha'] == r.sha(latest['off_probability'])
    assert_ema_equal(ema, {k: latest['current']['ps_pgr'][k] for k in EMA_BUFFERS})
    return dict(epochs=len(records), ema_updates=len(records), common_darg_rng_epochs=n_shared,
                extended_rng_epochs=len(records)-n_shared, paired_g1_rng_epochs=paired_shared,
                darg_terminal_streams_matched=ref_end_match)


def mechanism(state,module,outputs,reference):
    ids = state['test_np']
    base = outputs['T0']['details']
    y = state['labels_np'][ids]
    assert np.array_equal(ids+1,reference['rows_1based']) and np.array_equal(y,reference['labels'])
    mask = ~np.isin(ids,FOLD2_ROWS) if state['fold']==2 else np.ones(len(ids),dtype=bool)
    keep,labels,rows = ids[mask],y[mask],ids[mask]+1
    probs = {tag:probability(out,state) for tag,out in outputs.items()}
    pred = {tag:p[mask].argmax(1) for tag,p in probs.items()}
    treatments, abs_changes = {}, {}
    for tag,p in probs.items():
        abs_changes[tag] = np.abs(probs['T0'][:,0]-p[:,0]).tolist()
        treatments[tag] = dict(metrics=independent_metrics(p[mask],labels),
                               probability_abs_delta=distribution(abs_changes[tag]),
                               versus_full=patient_changes(pred['T0'],pred[tag],labels,rows))
    for tag in ('T2','T3'):
        treatments[tag]['structural_equivalent_to'] = 'T1'
    direct = patient_changes(pred['T1'],pred['T0'],labels,rows)
    backbone = patient_changes(np.asarray(reference['prediction'])[mask],pred['T1'],labels,rows)
    ref_pred = np.asarray(reference['prediction'])[mask]
    ref_correct = ref_pred == labels
    off_ok, full_ok = pred['T1']==labels, pred['T0']==labels
    combined = patient_changes(ref_pred,pred['T0'],labels,rows)
    assert combined['net'] == backbone['net']+direct['net']
    causes = {
        'true_new_from_historical_wrong':~ref_correct & ~off_ok & full_ok,
        'historical_correct_to_off_wrong':ref_correct & ~off_ok,
        'off_error_recovered_from_historical_correct':ref_correct & ~off_ok & full_ok,
        'new_residual_harm_to_historical_correct':ref_correct & off_ok & ~full_ok,
        'off_error_retained_from_historical_correct':ref_correct & ~off_ok & ~full_ok,
        'off_new_correct_retained_by_full':~ref_correct & off_ok & full_ok,
        'off_new_correct_lost_by_residual':~ref_correct & off_ok & ~full_ok,
    }
    attribution = {name:{'rows_1based':rows[flag].tolist(),'asd':int((flag & (labels==0)).sum()),
                         'td':int((flag & (labels==1)).sum())} for name,flag in causes.items()}
    b0_margin = np.asarray(reference['raw_gap'])[mask]*(1-2*labels)
    targets = torch.as_tensor(keep,device=state['y'].device)
    margin_off = true_margin(outputs['T1']['logp'],targets,state['y']).cpu().numpy()
    margin_on = true_margin(outputs['T0']['logp'],targets,state['y']).cpu().numpy()
    z,delta = base['z'][ids].double(),base['delta'][ids].double()
    j,p = base['jacobian'][ids].double(),base['jacobian_projected'][ids].double()
    displacement = delta.norm(dim=1)
    relative = 100 * displacement/z.norm(dim=1).clamp_min(1e-300)
    dot = (j*delta).sum(1)
    cosine = dot/(j.norm(dim=1)*displacement).clamp_min(1e-12)
    rho = p.square().sum(1)/(j.square().sum(1)+r.CFG['epsilon'])
    assert rho.le(1+2e-7).all() and cosine.abs().le(1+2e-7).all()
    head = state['cam'].Outputlayer
    with torch.no_grad():
        raw_on = independent_gap(head,base['z_enhanced'])
        raw_off = independent_gap(head,base['z'])
    torch.testing.assert_close(raw_off,base['raw_gap_off'],atol=0.,rtol=0.)
    torch.testing.assert_close(raw_on,outputs['T0']['logp'][:,0]-outputs['T0']['logp'][:,1],atol=1e-7,rtol=2e-6)
    change = raw_on[ids]-raw_off[ids]
    s,l,d = (base[key][ids] for key in ('signed_semantic','signed_statistical','difference'))
    both = (s != 0) & (l != 0)
    conflict = s*l < 0
    agree = s*l > 0
    stat_flip = conflict & (d*s < 0)
    assert not ((d*s<0)&(~conflict)&(s!=0)).any()
    with torch.no_grad():
        z_alt = outputs['T4']['details']['z_enhanced']
        anon_gap = (independent_gap(head,z_alt)-raw_on)[ids].abs()
    d_alt = outputs['T4']['details']['difference'][ids]
    llr_diff = (stat_flip.sum(0)).cpu().tolist()
    gate = base['gate'][ids]
    # Decompose J·delta over signed sources at the identical learned gate and radial multiplier.
    raw = base['raw'][ids].double().norm(dim=1)
    radial = torch.where(raw>0,displacement/raw.clamp_min(1e-300),torch.zeros_like(raw))
    support = base['reliability'][ids].double()*(1-base['null'][ids].double().mean(1,keepdim=True))
    basis_factor = radial[:,None] * gate.double()*base['jacobian_unit'][ids].double()*p*support
    mixture = base['alpha'][ids].double()
    semantic_contribution = basis_factor*mixture*s.double()
    statistical_contribution = basis_factor*(1-mixture)*l.double()
    first_order = semantic_contribution+statistical_contribution
    torch.testing.assert_close(first_order.sum(1),dot,atol=1e-7,rtol=2e-6)
    semantic_conflicting = ((first_order*s.double()<0)&(basis_factor!=0))
    per_direction = [{'direction': k+1,'active_gate_n':int((gate[:,k]>0).sum()),
                      'conflict_n':int(conflict[:,k].sum()),'llr_override_n':llr_diff[k],
                      'semantic_gap_mean':float(semantic_contribution[:,k].mean()),
                      'statistical_gap_mean':float(statistical_contribution[:,k].mean()),
                      'gate_mean':float(gate[:,k].double().mean()),
                      'applied_abs_mean':float((delta @ module.U[:,k].double()).abs().mean())} for k in range(16)]
    if state['fold']==2:
        assert not base['delta'][list(FOLD2_ROWS)].count_nonzero() and len(rows)==len(ids)-2
        assert all(np.array_equal(p[~mask],probs['T0'][~mask]) for p in probs.values())
    return dict(oof_rows_1based=(ids+1).tolist(),diagnostic_rows_1based=rows.tolist(),
                diagnostic_labels=labels.tolist(),treatments=treatments,asdp_absolute_changes=abs_changes,
                direct_hard_contribution=direct,residual_off_vs_darg=backbone,full_vs_darg=combined,
                patient_attribution=attribution,gate_by_direction=per_direction,
                relative_residual_pct=relative.cpu().tolist(),logit_gap_change=change.cpu().tolist(),
                first_order_gap_change=dot.cpu().tolist(),classifier_cosine=cosine.cpu().tolist(),
                classifier_spectral_coverage=rho.cpu().tolist(),
                true_margin_off=margin_off.tolist(),true_margin_full_minus_off=(margin_on-margin_off).tolist(),
                historical_margin=b0_margin.tolist(), historical_margin_full_change=(margin_on-b0_margin).tolist(),
                historical_darg_correct=ref_correct.tolist(),
                raw_semantic_l2=base['raw_s'][ids].double().norm(dim=1).cpu().tolist(),
                attention_difference_l2=(base['coverage'][ids,0]-base['coverage'][ids,1]).double().norm(dim=1).cpu().tolist(),
                llr_distribution=distribution(base['llr'][ids].cpu().numpy()),
                alpha_distribution=distribution(base['alpha'][ids].cpu().numpy()),
                null_asd=distribution(base['null'][ids,0].cpu().numpy()),
                null_td=distribution(base['null'][ids,1].cpu().numpy()),
                reliability=distribution(base['reliability'][ids].cpu().numpy()),
                semantic_first_order=semantic_contribution.sum(1).cpu().tolist(),
                statistical_first_order=statistical_contribution.sum(1).cpu().tolist(),
                fusion_counts={'elements':d.numel(),'both_nonzero':int(both.sum()),'agree':int(agree.sum()),
                               'conflict':int(conflict.sum()),'llr_overrides':int(stat_flip.sum()),
                               'semantic_opposed_active':int(semantic_conflicting.sum()),
                               'active_effects':int((first_order!=0).sum()),
                               'llr_positive':int((l>0).sum()),'llr_negative':int((l<0).sum()),
                               'final_positive':int((d>0).sum()),'final_negative':int((d<0).sum())},
                query_swap_d_abs=(d-d_alt).abs().mean(1).cpu().tolist(),
                query_swap_direction_flips=int((d*d_alt<0).sum()),query_swap_gap_abs=anon_gap.cpu().tolist())


def parameter_drift(values, restored):
    groups = {}
    for prefix, state_key in [('darg', 'model'), ('cam', 'cam'), ('output', 'cam'), ('spectral_module', 'ps_pgr')]:
        first = restored['base'][state_key] if state_key in ('model', 'cam') else restored['ps_pgr']
        last = values['base'][state_key] if state_key in ('model', 'cam') else values['ps_pgr']
        keys = [key for key in first if first[key].is_floating_point()]
        if prefix in ('cam', 'output'):
            keys = [k for k in keys if k.startswith('Outputlayer.') == (prefix == 'output')]
        if prefix == 'spectral_module':
            exclude = {*SPECTRUM_BUFFERS, *EMA_BUFFERS, 'q0', 'tau_s', 'tau_l'}
            keys = [key for key in keys if key not in exclude]
        denominator = sum(float(first[k].double().square().sum()) for k in keys)
        change = sum(float((last[k].double()-first[k].double()).square().sum()) for k in keys)
        groups[prefix] = {'absolute_l2': math.sqrt(change), 'relative_l2': math.sqrt(change/max(denominator, 1e-24)),
                          'changed_tensors': sum(not torch.equal(first[k], last[k]) for k in keys), 'tensors': len(keys)}
    return groups


def audit_task(group, fold, pre, source_rows, reference=None):
    task_path, best_path, latest_path, log_path = r.paths(group, fold)
    assert r.task_complete(group, fold)
    task, best, latest = h.read_json(task_path), h.torch_load(best_path), h.torch_load(latest_path)
    assert best['identity'] == task['identity'] == latest['identity'] and latest['best_sha'] == task['best_sha']
    assert latest['log_bytes'] == log_path.stat().st_size and latest['log_sha'] == task['log_sha']
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    state, module, opt, spectrum, cache, text = inputs_from_checkpoint(group, fold, best, pre)
    if reference is None:
        reference = reference_fold(state, best['identity'], source_rows)
    ids = state['test_np']
    assert np.array_equal(best['test_indices'], ids)
    bslots = optimizer_state_audit(state, module, opt, best['best_epoch'])
    outputs = check_interventions(state, module, text, group)
    full = outputs['T0']
    head = state['cam'].Outputlayer
    equations = check_formulas(module, full['details'],head)
    for tag, mode in (('T1','residual_off'), ('T2','gate_off'), ('T3','evidence_off')):
        check_formulas(module, outputs[tag]['details'],head,mode)
    swap = 'q_medical' if group == 'ANON' else 'q_anonymous'
    with replace_query(module,text[swap]):
        check_formulas(module,outputs['T4']['details'],head)
    p = probability(full, state)
    assert np.array_equal(p,best['probability']) and np.array_equal(p.argmax(1),best['prediction'])
    assert np.array_equal(probability(outputs['T1'],state),best['off_probability'])
    untouched,_,_ = reference_forward(state,False)
    assert torch.equal(untouched['logp'],outputs['T1']['logp'])
    assert h.sha(untouched['logp']) == records[best['best_epoch']-r.START-1]['off_logp_sha']
    y = state['labels_np'][ids]
    metrics = independent_metrics(p, y)
    same_metrics(metrics, best['metrics'])
    m = mechanism(state, module, outputs, reference)
    cam, obs = observe(state, full, spectrum)
    del outputs, full
    hist = audit_journal(state, task, best, latest, records, cache, cam, obs, pre['folds'][fold-1])
    state, _, _ = h.prepare(fold)
    state['shared_origin'] = {name:r.flatten(ps).double() for name,ps in r.shared_groups(state).items() if ps}
    pristine, pristine_opt = r.make_module(state, group, spectrum, cache, text)
    initial = r.bundle(state, pristine, pristine_opt)
    r.load_bundle(state, module, opt, latest['current'])
    # The latest optimizer must own the module and the freshly restored historical parameters.
    actual_old_optimizers = [state[k] for k in r.HIST_OPTS]
    assert len(actual_old_optimizers) == 3 and r.optimizer_ownership(state, module, opt)
    assert r.component_digests(r.bundle(state,module,opt))==latest['digests']
    assert [float(module.tau_s),float(module.tau_l)] == latest['scales']
    r.base_runtime().restore_rng(latest['rng'])
    ls = optimizer_state_audit(state, module, opt, task['end_epoch'])
    latest_ema = h.calibration_state(module)
    set_ema(module, latest['last_eval_ema'])
    terminal = graph(state, module)
    assert np.array_equal(probability(terminal,state),latest['probability'])
    plain,_,_ = reference_forward(state,False)
    assert np.array_equal(probability(plain,state),latest['off_probability'])
    assert h.sha(plain['logp']) == records[-1]['off_logp_sha']
    terminal_cam, terminal_obs = observe(state, terminal, spectrum)
    assert r.sha(terminal_cam) == records[-1]['legal_cam_sha']
    assert_ema_equal(latest_ema, ema_update(latest['last_eval_ema'], terminal_obs))
    set_ema(module, latest_ema)
    assert r.rng_mark() == (latest['rng_sha'], r.streams(latest['rng']))
    assert_witness(replay_step(state, module, opt, spectrum), task['next_step_witness'])
    history_checks = alignment = None
    if not module.protected_e2e:
        state,_,_ = h.prepare(fold)
        history_checks = protection_replay(state,task,best,latest,records)
        alignment = aligned_to_b0_best(state,module,latest,reference,records)
    by_best = records[best['best_epoch']-r.START-1]
    new_max = {k:max(v['residual_grads'][k] for v in records) for k in r.PARAMETER_GROUPS}
    assert all(np.isfinite(v) and v > 0 for v in new_max.values())
    stats = {}
    if module.protected_e2e:
        for name in records[0]['shared_grads']:
            values = [v['shared_grads'][name] for v in records]
            stats[name] = {
                'epochs':len(values),'conflicts':sum(v['conflict'] for v in values),
                'caps':sum(v['capped'] for v in values),'nonzero_injections':sum(v['applied_norm']>0 for v in values),
                'rounding_repairs':sum(v['rounding_repaired'] for v in values),
                'numerical_no_injection':sum(v['numerical_zero'] for v in values),
                'max_injected_ratio':max(v['applied_ratio'] for v in values),
                'max_relative_drift_from_initial':max(v['relative_parameter_drift_from_initial'] for v in values),
                'cosine_before':None if not (pre:=[v['pre_cosine'] for v in values if v['pre_cosine'] is not None]) else distribution(pre),
                'cosine_after':None if not (post:=[v['post_cosine'] for v in values if v['post_cosine'] is not None]) else distribution(post),
            }
        assert any(v['nonzero_injections']>0 for v in stats.values())
    training = {'first_objective':records[0]['losses'],'best_objective':by_best['losses'],
                'latest_objective':records[-1]['losses'],
                'best_legal_training_mean_margin_gain':by_best['mean_margin_gain_legal'],
                'max_residual_gradients':new_max,
                'shared_projection_by_group':stats,
                'best_parameter_drift_from_initial':parameter_drift(best['current'],initial),
                'latest_parameter_drift_from_initial':parameter_drift(latest['current'],initial)}
    return {'group': group, 'fold': fold, 'best_epoch': best['best_epoch'], 'end_epoch': task['end_epoch'],
            'metrics': metrics, 'rows_1based': (ids+1).tolist(), 'labels': y.tolist(), 'prediction': p.argmax(1).tolist(),
            'probability': p.tolist(), 'interventions': m, 'equations': equations,
            'best': 'PASS', 'latest': 'PASS', 'best_optimizer_slots': bslots, 'latest_optimizer_slots': ls,
            'history':hist,'training':training,'resumes':task['resumes'],
            'protected_trajectory':history_checks,'b0_best_alignment':alignment,
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
    def pool(key):
        return np.concatenate([v[key] for v in parts])
    relative, margins, off = (pool(k) for k in ('relative_residual_pct','true_margin_full_minus_off','true_margin_off'))
    label, correct = (pool(k) for k in ('diagnostic_labels','historical_darg_correct'))
    b0_margin = pool('historical_margin')
    overall_margin = pool('historical_margin_full_change')
    labels_mask = ~np.isin(pool('oof_rows_1based'), np.asarray(FOLD2_ROWS)+1)
    # Row IDs 377/407 occur once in the verified full out-of-fold partition.
    if len(relative)!=len(margins) and labels_mask.sum()!=len(margins):
        raise AssertionError('Mechanism scope differs from the legal diagnostic rows')
    def profile(mask):
        x = margins[mask]
        return {'n':len(x),'change':distribution(x) if len(x) else None,
                'fraction_positive':float((x>0).mean()) if len(x) else None,
                'versus_b0_margin':distribution(overall_margin[mask]) if len(x) else None,
                'versus_b0_fraction_positive':float((overall_margin[mask]>0).mean()) if len(x) else None}
    variants = {}
    for tag in ('T0','T1','T2','T3','T4'):
        records = [v['treatments'][tag] for v in parts]
        variants[tag] = dict(metrics=summaries(records),
                             probability_change=distribution(np.concatenate([v['asdp_absolute_changes'][tag] for v in parts])),
                             versus_full=merge_patients([v['versus_full'] for v in records]))
    fusion = {key:sum(v['fusion_counts'][key] for v in parts) for key in parts[0]['fusion_counts']}
    overall = {
        'residual_relative_pct':distribution(relative),'nonzero_residual_rows':int((relative>0).sum()),
        'raw_semantic_l2':distribution(pool('raw_semantic_l2')),
        'attention_difference_l2':distribution(pool('attention_difference_l2')),
        'fusion_counts':fusion,
        'llr_override_rate_among_conflicts':fusion['llr_overrides']/max(1,fusion['conflict']),
        'classifier_spectral_coverage':distribution(pool('classifier_spectral_coverage')),
        'classifier_cosine':distribution(pool('classifier_cosine')),
        'logit_gap_abs_change':distribution(np.abs(pool('logit_gap_change'))),
        'first_order_actual_pearson':correlation(pool('first_order_gap_change'),pool('logit_gap_change')),
        'residual_gap_abs_spearman':correlation(relative,np.abs(pool('logit_gap_change')),ranks=True),
        'margin':{k:profile(mask) for k,mask in [('all',np.ones(len(margins),bool)),('darg_correct',correct),
                                               ('darg_wrong',~correct),('asd',label==0),('td',label==1),
                                               ('low_margin',b0_margin<2),('high_margin',b0_margin>=2),
                                               ('off_low_margin',off<2),('off_high_margin',off>=2)]},
        'treatments':variants,
        'direct_hard_contribution':merge_patients([v['direct_hard_contribution'] for v in parts]),
        'residual_off_vs_darg':merge_patients([v['residual_off_vs_darg'] for v in parts]),
        'query_swap_d_abs_mean':float(pool('query_swap_d_abs').mean()),
        'query_swap_direction_flips':sum(v['query_swap_direction_flips'] for v in parts),
        'query_swap_gap_abs':distribution(pool('query_swap_gap_abs')),
        'full_vs_darg':merge_patients([v['full_vs_darg'] for v in parts]),
        'patient_attribution':{name:{'rows_1based':sorted(row for v in parts for row in v['patient_attribution'][name]['rows_1based']),
                                    'asd':sum(v['patient_attribution'][name]['asd'] for v in parts),
                                    'td':sum(v['patient_attribution'][name]['td'] for v in parts)}
                               for name in parts[0]['patient_attribution']},
        'semantic_first_order':distribution(pool('semantic_first_order')),
        'statistical_first_order':distribution(pool('statistical_first_order')),
    }
    for checkpoint in ('first','best','latest'):
        key = checkpoint+'_objective'
        overall[key] = {k:float(np.mean([v['training'][key][k] for v in tasks])) for k in tasks[0]['training'][key]}
    projection = [v for task in tasks for v in task['training']['shared_projection_by_group'].values()]
    if projection:
        overall['shared_projection'] = {
            **{key:sum(v[key] for v in projection) for key in ('epochs','conflicts','caps','nonzero_injections','rounding_repairs','numerical_no_injection')},
            'max_injected_ratio':max(v['max_injected_ratio'] for v in projection),
        }
    aligned = [v['b0_best_alignment'] for v in tasks if v['b0_best_alignment']]
    if aligned:
        same = [v for v in aligned if v['comparison']=='EXACT_SAME_EPOCH']
        overall['b0_best_alignment'] = {'exact_folds':len(same),'frozen_terminal_folds':len(aligned)-len(same),
                                       'exact_only':merge_patients([v['direct'] for v in same]),
                                       'all_b0_weight_probes':merge_patients([v['direct'] for v in aligned])}
    return overall


def aggregate(tasks, references, anonymous_parent=None):
    groups = {'DARG':sorted(references,key=lambda v:v['fold'])}
    for group in r.GROUPS:
        rows = sorted([v for v in tasks if v['group']==group],key=lambda v:v['fold'])
        if rows:
            groups[group] = rows
    for values in groups.values():
        if len(values)==len(r.FOLDS):
            all_rows = [row for v in values for row in v['rows_1based']]
            assert sorted(all_rows)==list(range(1,872))
    ten = {g:summaries(v) for g,v in groups.items() if v}
    reduced = {g:summaries([v for v in rows if v['fold']!=2]) for g,rows in groups.items()
               if any(v['fold']!=2 for v in rows)}
    requested = [('G1','DARG'),('G2','DARG'),('G2','G1')]
    if anonymous_parent:
        requested += [('ANON','DARG'),(anonymous_parent,'ANON')]
    per_fold, panels = {}, {}
    for a,b in requested:
        if a not in groups or b not in groups:
            continue
        comparator = {v['fold']:v for v in groups[b]}
        found = [paired(v,comparator[v['fold']]) for v in groups[a] if v['fold'] in comparator]
        if found:
            key = f'{a}-{b}'
            per_fold[key] = found
            panels[key] = {'all':summarize_pairs(found)}
            reduced_pairs = [v for v in found if v['fold']!=2]
            if reduced_pairs:
                panels[key]['excluding_fold2'] = summarize_pairs(reduced_pairs)
    mech = {g:group_mechanism(v) for g,v in groups.items() if g!='DARG'}
    return dict(metrics_10=ten,metrics_without_fold2=reduced,pairs=panels,per_fold_pairs=per_fold,mechanism=mech)


def g2_gate(status):
    aggregate = status['aggregate']
    arm = aggregate['mechanism'].get('G1')
    all_rows = [v for v in status['tasks'] if v['group']=='G1']
    cert = {
        'complete_independent_g1':len(all_rows)==10 and status['preflight_pass']==10 and status['protection']['pass']
                                 and not any(v.get('group') in (None,'G1') for v in status['failures']),
        'net_vs_b0':False,'fixed_direct_net':False,'true_new_net_vs_b0':False,
    }
    if arm:
        cert['net_vs_b0'] = aggregate['pairs']['G1-DARG']['all']['patients']['net']>0
        cert['fixed_direct_net'] = arm['direct_hard_contribution']['net']>0
        cause = arm['patient_attribution']
        cert['true_new_net_vs_b0'] = (len(cause['true_new_from_historical_wrong']['rows_1based']) >
                                      len(cause['new_residual_harm_to_historical_correct']['rows_1based']))
    return cert


def science(status):
    combined = status['aggregate']
    criteria = r.CFG['decision']
    arms = {}
    for group in r.MEDICAL:
        registered = [v for v in status['tasks'] if v['group']==group]
        passed = (len(registered)==10 and status['preflight_pass']==10 and status['protection']['pass']
                  and not any(v.get('group') in (None,group) for v in status['failures']))
        if not passed:
            arms[group] = dict(audited=False,performance_go=False,mechanism_go=False,verdict='NOT_EVALUATED')
            continue
        comparison = combined['pairs'][group+'-DARG']
        metrics = combined['metrics_10'][group]
        counts = comparison['all']['patients']
        deltas = comparison['all']['delta_pp']
        effect = combined['mechanism'][group]
        nofold2 = comparison['excluding_fold2']
        per_fold = combined['per_fold_pairs'][group+'-DARG']
        minimal = {key:sum(v['delta_pp'][key] for v in per_fold)-max(v['delta_pp'][key] for v in per_fold)
                   for key in ('acc','hard_auc')}
        worst_remaining_net = counts['net']-max(v['patients']['net'] for v in per_fold)
        perf = {
            'acc':metrics['acc'][0]>criteria['darg_reference_pct']['acc'],
            'ba':metrics['hard_auc'][0]>=criteria['darg_reference_pct']['hard_auc'],
            'prob_auc':deltas['prob_auc']>=criteria['prob_auc_min_delta_pp'],
            'sen':deltas['sen']>=criteria['sen_min_delta_pp'],
            'patient_net':counts['net']>0,
            'direct_hard_net':effect['direct_hard_contribution']['net']>0,
            'truly_new':bool(effect['patient_attribution']['true_new_from_historical_wrong']['rows_1based']),
            'beyond_fold2':nofold2['delta_pp']['acc']>0 and nofold2['delta_pp']['hard_auc']>=-1e-10 and nofold2['patients']['net']>0,
            'beyond_one_fold':minimal['acc']>0 and minimal['hard_auc']>=-1e-10 and worst_remaining_net>0,
        }
        mech = {
            'audited_protocol':all(v['protected_trajectory'] is not None and
                                   v['protected_trajectory']['exact_native_epochs']==v['end_epoch']-r.START for v in registered)
                               if group=='G1' else all(v['training']['shared_projection_by_group'] for v in registered),
            'nonzero_bounded_feature':effect['nonzero_residual_rows']>0,
            'actual_gap_change':effect['logit_gap_abs_change']['max']>0,
            'residual_gradients':all(all(v>0 for v in row['training']['max_residual_gradients'].values()) for row in registered),
        }
        arms[group] = dict(audited=True,performance_go=all(perf.values()),mechanism_go=all(mech.values()),
                           performance_checks=perf,mechanism_checks=mech,
                           verdict='GO' if all(perf.values()) and all(mech.values()) else 'NO-GO')
    successes = [g for g,v in arms.items() if v['verdict']=='GO']
    selected = (max(successes,key=lambda g:(combined['metrics_10'][g]['acc'][0],
                                          combined['mechanism'][g]['direct_hard_contribution']['net'],g=='G1'))
                if successes else None)
    medical = None
    if selected and len([v for v in status['tasks'] if v['group']=='ANON'])==10:
        pair = combined['pairs'][f'{selected}-ANON']['all']
        medical = pair['delta_pp']['acc']>0 and pair['delta_pp']['hard_auc']>=0 and pair['patients']['net']>0
    return dict(verdict='GO' if selected else 'NO-GO',arms=arms,anonymous_parent=selected,medical_advantage=medical)


def audit_cache_file(group,fold):
    return r.RT/'audit'/f'{group}_f{fold:02d}.json'


def cache_identity(group,fold):
    task = r.paths(group,fold)[0]
    return {'task_sha':r.sha_file(task),'implementation':r.implementation(),'group':group,'fold':fold}


def audit_one(group,fold):
    r.assert_gpu()
    pre = r.passing_preflight()
    r.training_authorization(group)
    assert r.task_complete(group,fold)
    source = csv_lookup(h.runtime().EXP/'results/fold_metrics.csv')
    task,reference = audit_task(group,fold,pre,source)
    h.atomic_json(audit_cache_file(group,fold),{'identity':cache_identity(group,fold),'task':task,'reference':reference})
    return task


def staged_audits(group,workers):
    def one(fold):
        try:
            path = audit_cache_file(group,fold)
            if path.exists():
                previous = h.read_json(path)
                if previous['identity']==cache_identity(group,fold) and r.task_complete(group,fold):
                    return fold,previous,None
            call = [sys.executable,'-B',str(r.HERE/'audit.py'),'--stage',group,'--fold',str(fold)]
            result = subprocess.run(call,cwd=r.HERE,capture_output=True,text=True,check=False)
            if result.returncode:
                return fold,None,(result.stderr+'\n'+result.stdout).strip()
            record = h.read_json(path)
            assert record['identity']==cache_identity(group,fold)
            return fold,record,None
        except Exception:
            return fold,None,traceback.format_exc()

    successes, errors = [],[]
    limit = r.worker_limit(workers)
    with ThreadPoolExecutor(max_workers=limit) as executor:
        futures = [executor.submit(one,fold) for fold in r.FOLDS]
        for future in as_completed(futures):
            fold,record,error = future.result()
            if error is not None:
                errors.append({'group':group,'fold':fold,'detail':error})
                print(error,flush=True)
            else:
                successes.append(record)
                print(f'AUDIT {group} f{fold:02d} Best Latest RNG EMA'+
                      (' native trajectory' if record['task']['protected_trajectory'] else ' projected gradient')+' PASS',flush=True)
    return sorted(successes,key=lambda row:row['task']['fold']),errors,limit


def write_gate(kind,status,groups,**parameters):
    audit_tasks = [v for v in status['tasks'] if v['group'] in groups]
    path = r.RT/f'{kind}_gate.json'
    h.atomic_json(path,{
        'implementation':status['implementation'],'audited_medical_groups':groups,
        'best_pass':sum(v['best']=='PASS' for v in audit_tasks),'latest_pass':sum(v['latest']=='PASS' for v in audit_tasks),
        'audit_groups_sha':r.sha(audit_tasks),
        'task_sha256':{f'{g}/{f}':r.sha_file(r.paths(g,f)[0]) for g in groups for f in r.FOLDS},
        **parameters,
    })
    h.update_manifest(**{f'{kind}_gate_sha256':r.sha_file(path)})


def audited_execution(stage,workers,failure=None):
    import report

    start = time.monotonic()
    manifest = h.read_json(r.HERE/'manifest.json')
    output = r.RESULTS/'audit_status.json'
    previous = h.read_json(output) if output.exists() and stage!='G1' else {}
    if not failure:
        r.training_authorization(stage)
    existing = [v for v in previous.get('tasks',[]) if v['group']!=stage]
    status = {'tasks':existing,'references':previous.get('references',[]),'failures':[],
              'preflight_pass':0,'implementation':r.implementation(),'audit_workers':0}
    if failure:
        status['failures'].append({'stage':stage,'detail':failure})
    try:
        pre = r.passing_preflight()
        status['preflight_pass'] = len(pre['folds'])
        status['preflight_head_probe'] = {str(v['fold']):v['historical_head_probe'] for v in pre['folds']}
    except Exception:
        status['failures'].append({'stage':'Preflight','detail':traceback.format_exc()})
        pre = None
    if pre and not failure:
        audited,errors,limit = staged_audits(stage,workers)
        status['failures'].extend(errors)
        status['audit_workers'] = max(limit,previous.get('audit_workers',0))
        status['tasks'].extend(row['task'] for row in audited)
        references = {row['fold']:row for row in status['references']}
        for row in audited:
            reference = row['reference']
            fold = reference['fold']
            assert fold not in references or references[fold]==reference
            references[fold] = reference
        status['references'] = [references[f] for f in sorted(references)]
    status['group_execution'] = {g:h.read_json(path) for g in r.GROUPS if (path:=r.RT/f'execution_{g}.json').exists()}
    status['best_pass'] = sum(v['best']=='PASS' for v in status['tasks'])
    status['latest_pass'] = sum(v['latest']=='PASS' for v in status['tasks'])
    status['protection'] = r.protected_files()
    status['rng'] = {k:sum(v['history'][k] for v in status['tasks']) for k in
                     ('ema_updates','common_darg_rng_epochs','extended_rng_epochs','paired_g1_rng_epochs')}
    status['rng']['historical_terminal_stream_matches'] = sum(v['history']['darg_terminal_streams_matched'] for v in status['tasks'])
    status['native_protection'] = {
        key:sum(v['protected_trajectory'][key] for v in status['tasks'] if v['protected_trajectory'])
        for key in ('exact_native_epochs','replayed_reference_epochs','archive_journal_milestones',
                    'archive_best_parameters_optimizers_probability_exact','archive_latest_parameters_optimizers_rng_exact')
    }
    anon_parent = r.anonymous_authorization() if stage=='ANON' and not failure else None
    status['aggregate'] = aggregate(status['tasks'],status['references'],anon_parent)
    if len(status['references'])==10 and any(round(x[0],4)!=r.CFG['decision']['darg_reference_pct'][key]
                                             for key,x in status['aggregate']['metrics_10']['DARG'].items()):
        status['failures'].append({'stage':'DARG','detail':'Baseline metrics differ from locked reference.'})
    status['science'] = science(status)
    decisions = status['g2_checks'] = g2_gate(status)
    complete = (not status['failures'] and status['protection']['pass']
                and sum(v['group']==stage for v in status['tasks'])==10)
    next_g2 = stage=='G1' and complete and all(decisions.values())
    selected = status['science']['anonymous_parent']
    next_anon = stage!='ANON' and complete and not next_g2 and bool(selected)
    status['g2_status'] = ('AUTHORIZED' if next_g2 else
                           'AUDITED' if sum(v['group']=='G2' for v in status['tasks'])==10 else
                           'PRE_REGISTERED_SKIP_G2' if stage=='G1' and complete else
                           previous.get('g2_status','INCOMPLETE'))
    status['anonymous_status'] = ('AUTHORIZED' if next_anon else 'PENDING_MEDICAL' if next_g2 else
                                  'AUDITED' if stage=='ANON' and complete else
                                  'PRE_REGISTERED_SKIP_ANONYMOUS' if complete else 'INCOMPLETE')
    status['status'] = ('AUDITED_G1_G2_AUTHORIZED' if next_g2 else
                        'AUDITED_MEDICAL_ANONYMOUS_AUTHORIZED' if next_anon else
                        'COMPLETE_AUDITED' if complete else
                        'PARTIAL' if status['best_pass'] or any(v['completed_folds'] for v in status['group_execution'].values()) else 'FAIL')
    status['audit_wall_seconds'] = time.monotonic()-start+previous.get('audit_wall_seconds',0)
    status['elapsed_seconds'] = time.time()-datetime.fromisoformat(manifest['started_utc']).timestamp()
    report.write_tables(status)
    if (complete and not next_g2 and not next_anon) or failure:
        status['handoff'] = report.append_handoff(status,manifest)
        if status['handoff']['status']=='BLOCKED' and status['status']=='COMPLETE_AUDITED':
            status['status'] = 'PARTIAL'
    report.write_report(status,manifest)
    h.atomic_json(output,status)
    if next_g2:
        write_gate('g2',status,['G1'],decision=decisions)
    if next_anon:
        write_gate('anonymous',status,[g for g in r.MEDICAL if status['science']['arms'][g]['audited']],arm=selected)
    h.update_manifest(status=status['status'],scientific_verdict=status['science']['verdict'],
                      g2_status=status['g2_status'],anonymous_status=status['anonymous_status'],
                      best_audit_pass=status['best_pass'],latest_audit_pass=status['latest_pass'],
                      result_sha256={v.name:h.sha_file(v) for v in r.RESULTS.iterdir() if v.is_file()},
                      handoff=status.get('handoff'),verified_protection=status['protection'],elapsed_seconds=status['elapsed_seconds'])
    if complete:
        for fold in r.FOLDS:
            audit_cache_file(stage,fold).unlink(missing_ok=True)
    print(f"{status['status']} {status['science']['verdict']} G2 {status['g2_status']} ANON {status['anonymous_status']}",flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=r.GROUPS,default='G1')
    parser.add_argument('--fold',type=int,choices=list(r.FOLDS))
    parser.add_argument('--workers',type=int,default=r.CFG['audit_workers'])
    parser.add_argument('--failure',help='Finalize an unrecoverable failure and append its handoff')
    args = parser.parse_args()
    if args.fold is not None:
        audit_one(args.stage,args.fold)
        return
    status = audited_execution(args.stage,args.workers,args.failure)
    if status['status'] in ('PARTIAL','FAIL'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
