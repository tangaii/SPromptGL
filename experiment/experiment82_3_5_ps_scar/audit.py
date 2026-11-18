#!/usr/bin/env python3
"""Independent PS-SCAR equations, metrics, checkpoint replay, and research decision."""
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

from ps_scar import EMA_BUFFERS, FOLD2_ROWS, SPECTRUM_BUFFERS


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
    old_evidence = semantic * (reliability/(1+(-llr*(16*raw_s).tanh()).exp()))[:,None]
    return dict(hh=white, tokens=tokens, attention=att, coverage=coverage, null=null, semantic=semantic,
                raw_s=raw_s, raw_llr=raw_llr, llr=llr, w_sp=w, reliability=reliability, evidence=old_evidence)


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


def auxiliary_formulas(state, out, selective):
    ids = legal_rows(state)
    d = out['details']
    y = state['y'][ids]
    row = torch.arange(len(ids),device=y.device)
    class_mass = d['evidence'][ids].sum(dim=2) + r.CFG['epsilon']
    nll = class_mass.sum(dim=1).log() - class_mass[row,y].log()
    class_losses = [nll[y==c].mean() for c in (0,1)]
    sem = .5*(class_losses[0]+class_losses[1])
    m0 = out['logp'][ids]
    margin = (m0[row,y]-m0[row,1-y]).detach()
    hp = r.CFG['loss']
    protection = torch.sigmoid((margin-hp['margin_center'])/hp['margin_temperature'])
    relative = d['delta'][ids].square().sum(1)/(d['z'][ids].square().sum(1)+r.CFG['epsilon'])
    safe = (protection*relative).mean()
    cf = sem.new_zeros(())
    if selective:
        sign = 1-2*y
        on = independent_gap(state['cam'].Outputlayer,d['z_enhanced'])[ids]*sign
        off = d['raw_gap_off'][ids].detach()*sign
        gain = on-off
        floor = hp['cf_weight_floor']
        hard = floor+(1-2*floor)*torch.sigmoid(hp['cf_hard_center']-off)
        easy = torch.sigmoid(off-hp['cf_easy_center'])
        t,target = hp['cf_temperature'],hp['cf_target_gain']
        difference = t*(F.softplus((target-gain)/t)-F.softplus(gain.new_tensor(target/t)))
        loss = difference*hard
        by_class = [loss[y==c].mean() for c in (0,1)]
        cf = .5*(by_class[0]+by_class[1]) + hp['cf_protect']*(easy*gain.neg().clamp_min(0).square()).mean()
    return dict(sem=sem,safe=safe,cf=cf)


def replay_step(state,module,opt,spectrum):
    ordered = r.optimizers(state,opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    out = graph(state,module,training=True)
    auxiliary = auxiliary_formulas(state,out,module.selective)
    trainable = list(module.parameters())
    vjp = [torch.autograd.grad(loss,trainable,retain_graph=True,allow_unused=True) if loss.requires_grad
           else (None,)*len(trainable) for loss in (out['cls'],*auxiliary.values())]
    hp = r.CFG['loss']
    weights = [hp[f'lambda_{key}'] for key in auxiliary]
    positions = {id(p):i for i,p in enumerate(trainable)}
    ratios = []
    for tensors in module.parameter_groups().values():
        sizes = []
        for local in vjp:
            parts = [local[positions[id(p)]] for p in tensors]
            vector = torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                                for p,g in zip(tensors,parts)]).double()
            sizes.append(float(vector.norm()))
        divisor = sum(w*x for w,x in zip(weights,sizes[1:]))
        if divisor:
            ratios.append(sizes[0]/(divisor+hp['gradient_eps']))
    scale = min(1.,hp['auxiliary_ratio_max']*min(ratios)) if ratios else 0.
    out['total'].backward()
    with torch.no_grad():
        for p,gs in zip(trainable,zip(*vjp[1:])):
            components = [w*g for w,g in zip(weights,gs) if g is not None]
            if components:
                value = scale*sum(components)
                if p.grad is None:
                    p.grad = value
                else:
                    p.grad.add_(value)
    for optimizer in ordered:
        optimizer.step()
    names = (*r.DARG_LOSSES,'darg_total',*auxiliary,'total')
    numbers = [out[key].detach() for key in (*r.DARG_LOSSES,'darg_total')]
    numbers += [v.detach() for v in auxiliary.values()]
    numbers += [out['darg_total'].detach() + scale*sum(w*v.detach() for w,v in zip(weights,auxiliary.values()))]
    losses = dict(zip(names,torch.stack(numbers).cpu().tolist()))
    mark = r.rng_mark()
    evaluated = graph(state,module)
    p,(cam,obs) = probability(evaluated,state),observe(state,evaluated,spectrum)
    prev = h.calibration_state(module)
    set_ema(module,ema_update(prev,obs))
    assert r.rng_mark() == mark
    return dict(losses=losses,auxiliary_scale=scale,rng=mark[0],rng_streams=mark[1],
                digests=r.component_digests(r.bundle(state,module,opt)),probability_sha=r.sha(p),
                legal_cam_sha=r.sha(cam),ema_before_sha=r.sha(prev),ema_after_sha=r.sha(h.calibration_state(module)),
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


def check_gradient_row(row, selective):
    loss, hp = row['losses'], r.CFG['loss']
    assert all(math.isfinite(value) for value in loss.values())
    expected = loss['cls'] + loss['graph'] + .05*loss['pair'] + .01*(loss['adv'] + loss['preserve'])
    assert np.isclose(expected,loss['darg_total'],atol=1e-7,rtol=1e-6)
    gradients = row['gradients']
    keys = ('sem','safe','cf')
    assert tuple(gradients)==r.PARAMETER_GROUPS
    denoms = {name:sum(hp[f'lambda_{k}']*g[f'{k}_norm'] for k in keys) for name,g in gradients.items()}
    valid = [gradients[name]['cls_norm']/(bound+hp['gradient_eps']) for name,bound in denoms.items() if bound>0.]
    scale = min(1.,hp['auxiliary_ratio_max']*min(valid)) if valid else 0.
    assert scale==row['auxiliary_scale']
    total = loss['darg_total'] + scale*sum(hp[f'lambda_{k}']*loss[k] for k in keys)
    assert np.isclose(total,loss['total'],atol=1e-7,rtol=1e-6)
    for name,g in gradients.items():
        c = g['cls_norm']
        measured = scale*g['mixed_norm']/c if c else 0.
        conservative = scale*denoms[name]/c if c else 0.
        assert measured==g['weighted_ratio'] and conservative==g['budget_ratio']
        assert 0<=measured<=conservative+1e-12<=hp['auxiliary_ratio_max']+2e-12
        for key in keys:
            cosine = g[f'cosine_cls_{key}']
            n = g[f'{key}_norm']
            assert n>=0 and ((cosine is not None and abs(cosine)<=1+1e-12) if c*n else cosine is None)
        assert 0<=g['nonzero_update_fraction']<=1
        if not selective:
            assert loss['cf']==g['cf_norm']==0.
    if row['epoch']<=r.START+2:
        r.verify_new_gradients(row['classification'],first=row['epoch']==r.START+1)


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
            assert_ema_equal(ema, {k: best['current']['ps_scar'][k] for k in EMA_BUFFERS})
            assert r.sha(moment) == r.sha(best_obs) and r.sha(best_cam) == row['legal_cam_sha']
            assert row['probability_sha'] == r.sha(best['probability']) and np.array_equal(pred, best['prediction'])
            assert (prev, streams) == (best['rng_sha'], r.streams(best['rng']))
        if epoch == end:
            assert_ema_equal(ema, latest['last_eval_ema'])
        ema = ema_update(ema, moment)
        assert r.sha(ema) == row['ema_after']
        check_gradient_row(row,task['selective'])
    assert best['best_epoch'] == task['best_epoch'] == latest['best_epoch'] == chosen
    assert (max_correct, wait, end) == (latest['best_correct'], latest['wait'], latest['next_epoch'])
    assert wait > r.CFG['early_stop_wait_gt'] or end == r.LIMIT
    assert (prev, streams) == (latest['rng_sha'], r.streams(latest['rng']))
    assert records[-1]['probability_sha'] == r.sha(latest['probability'])
    assert_ema_equal(ema, {k: latest['current']['ps_scar'][k] for k in EMA_BUFFERS})
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
    ref_correct = np.asarray(reference['prediction'])[mask] == labels
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
    per_direction = [{'direction': k+1,'active_gate_n':int((gate[:,k]>0).sum()),
                      'conflict_n':int(conflict[:,k].sum()),'llr_override_n':llr_diff[k],
                      'gate_mean':float(gate[:,k].double().mean()),
                      'applied_abs_mean':float((delta @ module.U[:,k].double()).abs().mean())} for k in range(16)]
    if state['fold']==2:
        assert not base['delta'][list(FOLD2_ROWS)].count_nonzero() and len(rows)==len(ids)-2
        assert all(np.array_equal(p[~mask],probs['T0'][~mask]) for p in probs.values())
    return dict(oof_rows_1based=(ids+1).tolist(),diagnostic_rows_1based=rows.tolist(),
                diagnostic_labels=labels.tolist(),treatments=treatments,asdp_absolute_changes=abs_changes,
                direct_hard_contribution=direct,residual_off_vs_darg=backbone,gate_by_direction=per_direction,
                relative_residual_pct=relative.cpu().tolist(),logit_gap_change=change.cpu().tolist(),
                first_order_gap_change=dot.cpu().tolist(),classifier_cosine=cosine.cpu().tolist(),
                classifier_spectral_coverage=rho.cpu().tolist(),
                true_margin_off=margin_off.tolist(),true_margin_full_minus_off=(margin_on-margin_off).tolist(),
                historical_darg_correct=ref_correct.tolist(),
                raw_semantic_l2=base['raw_s'][ids].double().norm(dim=1).cpu().tolist(),
                attention_difference_l2=(base['coverage'][ids,0]-base['coverage'][ids,1]).double().norm(dim=1).cpu().tolist(),
                llr_distribution=distribution(base['llr'][ids].cpu().numpy()),
                alpha_distribution=distribution(base['alpha'][ids].cpu().numpy()),
                null_asd=distribution(base['null'][ids,0].cpu().numpy()),
                null_td=distribution(base['null'][ids,1].cpu().numpy()),
                reliability=distribution(base['reliability'][ids].cpu().numpy()),
                fusion_counts={'elements':d.numel(),'both_nonzero':int(both.sum()),'agree':int(agree.sum()),
                               'conflict':int(conflict.sum()),'llr_overrides':int(stat_flip.sum()),
                               'llr_positive':int((l>0).sum()),'llr_negative':int((l<0).sum()),
                               'final_positive':int((d>0).sum()),'final_negative':int((d<0).sum())},
                query_swap_d_abs=(d-d_alt).abs().mean(1).cpu().tolist(),
                query_swap_direction_flips=int((d*d_alt<0).sum()),query_swap_gap_abs=anon_gap.cpu().tolist())


def parameter_drift(values, restored):
    groups = {}
    for prefix, state_key in [('darg', 'model'), ('cam', 'cam'), ('output', 'cam'), ('spectral_module', 'ps_scar')]:
        first = restored['base'][state_key] if state_key in ('model', 'cam') else restored['ps_scar']
        last = values['base'][state_key] if state_key in ('model', 'cam') else values['ps_scar']
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
    assert r.component_digests(r.bundle(state,module,opt))==latest['digests']
    assert [float(module.tau_s),float(module.tau_l)] == latest['scales']
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
                'cf_loss_first': records[0]['losses']['cf'],
                'cf_loss_at_best': records[best['best_epoch']-r.START-1]['losses']['cf'],
                'cf_loss_latest': records[-1]['losses']['cf'],
                'max_counterfactual_gradient': max(g['cf_norm'] for v in records for g in v['gradients'].values()),
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
    def pool(key):
        return np.concatenate([v[key] for v in parts])
    relative, margins, off = (pool(k) for k in ('relative_residual_pct','true_margin_full_minus_off','true_margin_off'))
    label, correct = (pool(k) for k in ('diagnostic_labels','historical_darg_correct'))
    labels_mask = ~np.isin(pool('oof_rows_1based'), np.asarray(FOLD2_ROWS)+1)
    # Row IDs 377/407 occur once in the verified full out-of-fold partition.
    if len(relative)!=len(margins) and labels_mask.sum()!=len(margins):
        raise AssertionError('Mechanism scope differs from the legal diagnostic rows')
    def profile(mask):
        x = margins[mask]
        return {'n':len(x),'change':distribution(x) if len(x) else None,
                'fraction_positive':float((x>0).mean()) if len(x) else None}
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
                                               ('low_margin',off<2),('high_margin',off>=2)]},
        'treatments':variants,
        'direct_hard_contribution':merge_patients([v['direct_hard_contribution'] for v in parts]),
        'residual_off_vs_darg':merge_patients([v['residual_off_vs_darg'] for v in parts]),
        'query_swap_d_abs_mean':float(pool('query_swap_d_abs').mean()),
        'query_swap_direction_flips':sum(v['query_swap_direction_flips'] for v in parts),
        'query_swap_gap_abs':distribution(pool('query_swap_gap_abs')),
        'max_actual_aux_ratio':max(v['training']['max_auxiliary_ratio'] for v in tasks),
        'max_counterfactual_gradient':max(v['training']['max_counterfactual_gradient'] for v in tasks),
    }
    for k in ('semantic_loss_first','semantic_loss_at_best','semantic_loss_latest','cf_loss_first','cf_loss_at_best','cf_loss_latest'):
        overall[k] = float(np.mean([v['training'][k] for v in tasks]))
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


def science(status):
    out = status['aggregate']
    cfg = r.CFG['decision']
    arms = {}
    for group in r.MEDICAL:
        records = [v for v in status['tasks'] if v['group']==group]
        enough = (len(records)==10 and status['preflight_pass']==10 and status['protection']['pass']
                  and not any(row.get('group') in (None,group) for row in status['failures']))
        if not enough:
            arms[group] = dict(performance_go=False,mechanism_go=False,audited=False,verdict='NO-GO')
            continue
        metrics = out['metrics_10'][group]
        versus = out['pairs'][f'{group}-DARG']['all']
        delta = versus['delta_pp']
        p = {'acc':metrics['acc'][0]>cfg['darg_reference_pct']['acc'],
             'ba':metrics['hard_auc'][0]>=cfg['darg_reference_pct']['hard_auc'],
             'prob_auc':delta['prob_auc']>=cfg['prob_auc_min_delta_pp'],
             'sen':delta['sen']>=cfg['sen_min_delta_pp'],
             'patient_net':versus['patients']['net']>0}
        m = out['mechanism'][group]
        q = {'direct_hard_net':m['direct_hard_contribution']['net']>0,
             'nonzero_finite_residual':all(max(v['interventions']['relative_residual_pct'])>0 for v in records),
             'endpoint_gradients':all(v['training']['best_parameter_drift']['spectral_module']['changed_tensors']>0 for v in records),
             'llr_sign_override':m['fusion_counts']['llr_overrides']>0,
             'logit_gap_change':m['logit_gap_abs_change']['max']>0}
        arms[group] = dict(performance_go=all(p.values()),mechanism_go=all(q.values()),audited=True,
                           performance_checks=p,mechanism_checks=q,verdict='GO' if all(p.values()) and all(q.values()) else 'NO-GO')
    successes = [g for g,v in arms.items() if v['verdict']=='GO']
    selected = (max(successes,key=lambda g:(out['metrics_10'][g]['acc'][0],
                                           out['mechanism'][g]['direct_hard_contribution']['net'],g=='G1'))
                if successes and all(v['audited'] for v in arms.values()) else None)
    medical = None
    if selected and len([v for v in status['tasks'] if v['group']=='ANON'])==10:
        pair = out['pairs'][f'{selected}-ANON']['all']
        medical = pair['delta_pp']['acc']>0 and pair['delta_pp']['hard_auc']>=0 and pair['patients']['net']>0
    return dict(verdict='GO' if selected else 'NO-GO',arms=arms,anonymous_parent=selected,medical_advantage=medical)


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


def audited_execution(stage, failure=None):
    import report

    start = time.monotonic()
    manifest = h.read_json(r.HERE/'manifest.json')
    previous = {'tasks':[],'references':[]}
    anonymous_parent = None
    arms = r.MEDICAL
    if stage=='anonymous':
        anonymous_parent = r.anonymous_authorization()
        path = r.RESULTS/'audit_status.json'
        gate = h.read_json(r.RT/'anonymous_gate.json')
        assert h.sha_file(path)==gate['stage_audit_sha256']
        previous = h.read_json(path)
        arms = ('ANON',)
    status = {'tasks':previous['tasks'],'references':previous['references'],'failures':[],
              'preflight_pass':0,'implementation':r.implementation()}
    if failure:
        status['failures'].append({'stage':stage,'detail':failure})
    try:
        r.assert_gpu()
        pre = r.passing_preflight()
        status['preflight_pass'] = len(pre['folds'])
        status['preflight_head_probe'] = {str(v['fold']):v['historical_head_probe'] for v in pre['folds']}
    except Exception:
        status['failures'].append({'stage':'Preflight','detail':traceback.format_exc()})
        pre = None
    if pre:
        rows = csv_lookup(h.runtime().EXP/'results/fold_metrics.csv')
        references = {v['fold']:v for v in status['references']}
        for group in arms:
            for fold in r.FOLDS:
                try:
                    if not r.task_complete(group,fold):
                        raise RuntimeError('Formal task lacks a completed checkpoint pair')
                    row,reference = audit_task(group,fold,pre,rows,references.get(fold))
                    status['tasks'].append(row)
                    references[fold] = reference
                    print(f'AUDIT {group} f{fold:02d} Best Latest J RNG EMA PASS',flush=True)
                except Exception:
                    detail = traceback.format_exc()
                    status['failures'].append({'group':group,'fold':fold,'detail':detail})
                    print(detail,flush=True)
        status['references'] = [references[f] for f in sorted(references)]
    status['group_execution'] = {g:h.read_json(path) for g in r.GROUPS if (path:=r.RT/f'execution_{g}.json').exists()}
    status['best_pass'] = sum(v['best']=='PASS' for v in status['tasks'])
    status['latest_pass'] = sum(v['latest']=='PASS' for v in status['tasks'])
    status['protection'] = protected_status()
    status['rng'] = {k:sum(v['history'][k] for v in status['tasks']) for k in
                     ('ema_updates','common_darg_rng_epochs','extended_rng_epochs','paired_g1_rng_epochs')}
    status['rng']['historical_terminal_stream_matches'] = sum(v['history']['darg_terminal_streams_matched'] for v in status['tasks'])
    status['rng_by_group'] = {g:{k:sum(v['history'][k] for v in status['tasks'] if v['group']==g) for k in
                                ('common_darg_rng_epochs','extended_rng_epochs','paired_g1_rng_epochs')} for g in arms}
    status['aggregate'] = aggregate(status['tasks'],status['references'],anonymous_parent)
    if len(status['references'])==10 and any(round(x[0],4)!=r.CFG['decision']['darg_reference_pct'][k]
                                             for k,x in status['aggregate']['metrics_10']['DARG'].items()):
        status['failures'].append({'stage':'DARG','detail':'Baseline means differ from frozen DARG reference.'})
    status['science'] = science(status)
    complete = not status['failures'] and status['protection']['pass']
    parent = status['science']['anonymous_parent']
    authorized = stage=='medical' and complete and bool(parent)
    status['anonymous_status'] = ('AUTHORIZED' if authorized else 'SKIPPED_PREREGISTERED' if stage=='medical' and complete
                                  else 'AUDITED' if stage=='anonymous' and complete else 'INCOMPLETE')
    status['status'] = ('AUDITED_MEDICAL_ANONYMOUS_AUTHORIZED' if authorized else 'COMPLETE_AUDITED') if complete else (
        'PARTIAL' if status['best_pass'] or any(v['completed_folds'] for v in status['group_execution'].values()) else 'FAILED')
    status['audit_wall_seconds'] = time.monotonic()-start+previous.get('audit_wall_seconds',0)
    status['elapsed_seconds'] = time.time()-datetime.fromisoformat(manifest['started_utc']).timestamp()
    report.write_tables(status)
    if (complete and not authorized) or failure is not None:
        status['handoff'] = report.append_handoff(status,manifest)
        if status['handoff']['status']=='BLOCKED' and status['status']=='COMPLETE_AUDITED':
            status['status'] = 'PARTIAL'
    report.write_report(status,manifest)
    path = r.RESULTS/'audit_status.json'
    h.atomic_json(path,status)
    if authorized:
        cert = r.RT/'anonymous_gate.json'
        h.atomic_json(cert,dict(implementation=status['implementation'],arm=parent,best_pass=20,latest_pass=20,
                                stage_audit_sha256=h.sha_file(path),
                                task_sha256={f'{g}/{f}':h.sha_file(r.paths(g,f)[0]) for g in r.MEDICAL for f in r.FOLDS}))
        h.update_manifest(anonymous_gate_sha256=h.sha_file(cert))
    h.update_manifest(status=status['status'],scientific_verdict=status['science']['verdict'],
                      anonymous_status=status['anonymous_status'],best_audit_pass=status['best_pass'],
                      latest_audit_pass=status['latest_pass'],result_sha256={v.name:h.sha_file(v) for v in r.RESULTS.iterdir() if v.is_file()},
                      handoff=status.get('handoff'),verified_protection=status['protection'],elapsed_seconds=status['elapsed_seconds'])
    print(f"{status['status']} {status['science']['verdict']} ANON {status['anonymous_status']}",flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('medical','anonymous'),default='medical')
    parser.add_argument('--failure',help='Finalize an unrecoverable failure and append its handoff')
    args = parser.parse_args()
    status = audited_execution(args.stage,args.failure)
    if status['status'] in ('PARTIAL','FAILED'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
