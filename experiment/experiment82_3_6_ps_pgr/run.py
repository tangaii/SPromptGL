#!/usr/bin/env python3
"""GPU0-only PS-PGR training and recovery from the historical DARG state."""
from __future__ import annotations

import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
if __name__ == '__main__':
    sys.modules['run'] = sys.modules[__name__]

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import pickle
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
local_dependencies = HERE / 'runtime/python'
if local_dependencies.is_dir():
    sys.path.insert(0, str(local_dependencies))

import numpy as np
import torch

import history as h
from history import (BASE_GROUP, CFG, HIST_OPTS, METRICS, RESULTS, RT, START, atomic_json,
                     base_runtime, paths, read_json, rng_sha, sha, sha_file, torch_load)
from ps_pgr import EPS, FOLD2_ROWS, PSPGR, SPECTRUM_BUFFERS, deterministic_gap, deterministic_logits, head_sensitivity, signed_mix, smooth_limit
from torch.nn import functional as F

GROUPS, FOLDS = tuple(CFG['groups']), range(1, CFG['folds'] + 1)
MEDICAL = ('G1', 'G2')
LIMIT, SAVE_INTERVAL = CFG['max_epochs'], CFG['save_latest_every_epochs']
PARAMETER_GROUPS = tuple(CFG['parameter_groups'])
DARG_LOSSES = ('cls', 'graph', 'pair', 'adv', 'preserve')
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')


def implementation():
    names = ('config.json', 'ps_pgr.py', 'history.py', 'run.py', 'audit.py', 'report.py', 'design_review.md')
    return {name: sha_file(HERE / name) for name in names}


def assert_gpu():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    h.runtime()
    assert torch.cuda.current_device() == 0 and torch.get_num_threads() == 1


def streams(rng):
    return {
        'python': hashlib.sha256(pickle.dumps(rng['python'], protocol=4)).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(rng['numpy'], protocol=4)).hexdigest(),
        'cpu': sha(rng['cpu']), 'cuda': [sha(state) for state in rng['cuda']],
    }


def rng_mark():
    base = base_runtime()
    raw = base.pack_rng()
    return base.rng_sha(raw), streams(raw)


def make_module(state, group, spectrum, cache, text):
    fold = state['fold']
    with h.isolated_rng(CFG['initialization_seed_base'] + fold):
        module = PSPGR(spectrum['spectrum'], text['q_medical'], cache['stats'], fold, CFG)
    with h.isolated_rng(CFG['fusion']['seed_base'] + fold):
        module.initialize_fusion()
    with h.isolated_rng(CFG['signed_gate']['seed_base'] + fold):
        module.initialize_gate()
    module.to(state['x'].device)
    legal = spectrum['warmup_cam'][h.legal_train(state)].to(state['x'].device)
    module.calibrate(legal)
    module.q0.copy_(text[CFG['queries'][group]])
    module.protected_e2e = group == 'G2' or (group == 'ANON' and anonymous_authorization() == 'G2')
    hp = CFG['psce_optimizer']
    opt = torch.optim.AdamW(module.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    return module, opt


def historical_forward(state, training=False):
    return h.runtime().forward(state, BASE_GROUP, training=training)


def native_forward(state, training=False):
    """Observe the unmodified historical classification call and its true logits."""
    head = state['cam'].Outputlayer
    features, logits = [], []

    def observe_input(_head, args):
        features.append(args[0])

    def observe_logits(_layer, _args, scores):
        logits.append(scores)

    hooks = (head.register_forward_pre_hook(observe_input),
             head.classifier.register_forward_hook(observe_logits))
    try:
        out = historical_forward(state, training=training)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(features) == len(logits) == 1 and features[0].shape == (CFG['subjects'], 144)
    return out, features[0], logits[0]


def forward(state, module, intervention='full'):
    """The native evaluation head receives the actual enhanced CAM144."""
    module.eval()
    captured = []

    def inject(head, args):
        assert not captured and not head.training
        before = rng_sha()
        off_gap, jac = head_sensitivity(head, args[0])
        enhanced, details = module(args[0], jac, intervention=intervention)
        details['raw_gap_off'] = off_gap
        details['off_logp'] = F.log_softmax(deterministic_logits(head, args[0]), dim=1)
        assert enhanced is details['z_enhanced'] and before == rng_sha()
        captured.append(details)
        return enhanced, args[1]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(inject)
    try:
        out = historical_forward(state)
    finally:
        handle.remove()
    assert len(captured) == 1
    out['details'] = captured[0]
    check_forward(out, captured[0])
    return out


def check_forward(out, d):
    bound = CFG['residual_limit'] + CFG['preflight']['residual_bound_roundoff']
    good = [torch.isfinite(t).all() for t in (*d.values(), out['logp'], out['total'])]
    good += [d['delta'].double().norm(dim=-1).le(bound*d['z'].double().norm(dim=-1)).all(),
             d['gate'].ge(0).all(), d['gate'].le(1).all(),
             d['alpha'].ge(CFG['fusion']['alpha_min']).all(), d['alpha'].le(CFG['fusion']['alpha_max']).all()]
    for key in ('attention', 'w_sp'):
        w = d[key]
        good += [w.ge(0).all(), (w.sum(-1)-1).abs().le(CFG['preflight']['row_sum_atol']).all()]
    if not torch.stack(good).all():
        raise FloatingPointError('Nonfinite or invalid CAM144 residual')


def optimizers(state, opt):
    return [state[key] for key in HIST_OPTS] + [opt]


def optimizer_ownership(state, module, opt):
    ordered = [[id(p) for pg in o.param_groups for p in pg['params']] for o in optimizers(state, opt)]
    owned = [key for keys in ordered for key in keys]
    needed = {id(p) for m in (state['model'], state['cam'], module) for p in m.parameters() if p.requires_grad}
    assert len(owned) == len(set(owned)) and needed <= set(owned)
    assert ordered[-1] == [id(p) for p in module.parameters()]
    return [len(keys) for keys in ordered]


def residual_objective(state, module, features):
    """The increment is evaluated with fixed head weights on actual on/off features."""
    head = state['cam'].Outputlayer
    base = features.detach()
    off, jac = head_sensitivity(head, base)
    enhanced, details = module(features if module.protected_e2e else base, jac)
    # Isolate the residual VJP from the identity branch of z + delta.
    on = deterministic_gap(head, base + details['delta'])
    rows = state['legal_rows']
    y = state['y'].index_select(0, rows)
    label_sign = (1 - 2*y).to(on.dtype)
    m_off = off.index_select(0, rows)*label_sign
    m_on = on.index_select(0, rows)*label_sign
    cfg = CFG['loss']
    floor = cfg['increment_weight_floor']
    weights = floor + (1-floor)*(cfg['increment_margin_center']-m_off).sigmoid()
    benefit = weights*(F.softplus(-m_on)-F.softplus(-m_off))
    inc = 0.5*(benefit[y == 0].mean()+benefit[y == 1].mean())
    trusted = (m_off-cfg['protection_margin_center']).sigmoid()
    protect = (trusted*(m_off-m_on).relu().square()).mean()
    relative_sq = details['delta'].index_select(0, rows).square().sum(1)/(base.index_select(0, rows).square().sum(1)+EPS)
    safe = (trusted*relative_sq).mean()
    terms = {'inc':inc, 'protect':protect, 'safe':safe}
    loss = sum(cfg['lambda_'+k]*v for k,v in terms.items())
    details['raw_gap_off'] = off
    return loss, terms, details, m_on, m_off


def flatten(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters])


def gradient_hash(state):
    return sha({key:{name:p.grad for name,p in state[key].named_parameters()} for key in ('model','cam')})


def base_digests(state):
    return {key:sha(state[key].state_dict()) for key in ('model','cam',*HIST_OPTS)}


def shared_groups(state):
    head = {id(p) for p in state['cam'].Outputlayer.parameters()}
    return {f'{key}.{index}':tuple(p for p in group['params'] if p.requires_grad and id(p) not in head)
            for key in HIST_OPTS for index,group in enumerate(state[key].param_groups)}


def project_increment(baseline, residual):
    """Return the representable gradient sum and diagnose the actual added vector."""
    cfg = CFG['shared_gradient']
    b, raw = baseline.double(), residual.double()
    bb = b.dot(b)
    rr = raw.norm()
    raw_dot = b.dot(raw)
    if not torch.stack((bb, rr, raw_dot)).isfinite().all():
        raise FloatingPointError('Nonfinite shared gradients')
    nb, nr, dot = float(bb.sqrt()), float(rr), float(raw_dot)
    target = cfg['ratio_max']*nb
    proposed = raw
    conflict = dot < 0 and float(bb) > cfg['minimum_baseline_norm_squared']
    blocked = float(bb) <= cfg['minimum_baseline_norm_squared']
    if blocked:
        proposed = torch.zeros_like(raw)
    elif conflict:
        proposed = raw-(raw_dot/(bb+cfg['denominator_epsilon']))*b
        proposed = proposed-(proposed.dot(b).clamp_max(0)/bb)*b
    projected_norm = float(proposed.norm())
    capped = projected_norm > target
    if capped:
        proposed = proposed*(target/projected_norm*(1-1e-4))
    native = baseline.clone() if projected_norm == 0 else (b+proposed).to(baseline.dtype)
    applied = native.double()-b
    actual_dot, actual_norm = float(applied.dot(b)), float(applied.norm())
    repaired = False
    if actual_dot < 0 and not blocked:
        inward = proposed*(1-1e-4)
        correction = max(-actual_dot/float(bb),0.)+4*torch.finfo(baseline.dtype).eps
        native = (b+inward+correction*b).to(baseline.dtype)
        applied = native.double()-b
        actual_dot, actual_norm = float(applied.dot(b)), float(applied.norm())
        repaired = True
    numerical_zero = actual_dot < 0 or actual_norm > target or not torch.isfinite(native).all()
    if numerical_zero:
        native, actual_dot, actual_norm = baseline.clone(), 0., 0.
    return native, {
        'base_norm':nb, 'raw_norm':nr, 'raw_dot':dot,
        'pre_cosine':dot/(nb*nr) if nb*nr else None,
        'projected_norm':projected_norm, 'applied_norm':actual_norm, 'applied_dot':actual_dot,
        'post_cosine':actual_dot/(nb*actual_norm) if nb*actual_norm else None,
        'applied_ratio':actual_norm/nb if nb else 0.,
        'conflict':conflict, 'capped':capped, 'baseline_tiny':blocked,
        'rounding_repaired':repaired, 'numerical_zero':bool(numerical_zero),
    }


def apply_shared_gradient(state, derivative_by_id):
    audited = {}
    for name, parameters in shared_groups(state).items():
        active = [p for p in parameters if p.grad is not None]
        skipped = [derivative_by_id.get(id(p)) for p in parameters if p.grad is None]
        skipped = [g for g in skipped if g is not None]
        if active:
            b = torch.cat([p.grad.reshape(-1) for p in active])
            source = [derivative_by_id.get(id(p)) for p in active]
            raw = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(active,source)])
        else:
            b = raw = state['x'].new_empty(0)
        native, row = project_increment(b, raw)
        for p, values in zip(active, native.split([p.numel() for p in active])):
            p.grad.copy_(values.reshape_as(p))
        row['baseline_active_tensors'] = len(active)
        row['native_unused_tensors'] = len(parameters)-len(active)
        row['skipped_raw_norm'] = float(torch.stack([g.double().square().sum() for g in skipped]).sum().sqrt()) if skipped else 0.
        audited[name] = row
    return audited


def train_step(state, module, opt, inspect=False):
    ordered = optimizers(state, opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    module.train()
    out, z, logits = native_forward(state, training=True)
    after_native = rng_mark()
    loss, terms, details, m_on, m_off = residual_objective(state,module,z)
    check_forward(out, details)
    old = tuple(p for k in ('model','cam') for p in state[k].parameters() if p.requires_grad)
    new = tuple(module.parameters())
    grads = torch.autograd.grad(loss, (*new,*old), retain_graph=module.protected_e2e, allow_unused=True)
    fresh, inherited = grads[:len(new)], grads[len(new):]
    routed = dict(zip(map(id,old),inherited))
    assert all(p.grad is None for p in (*new,*old)), 'Autograd accumulated gradients outside its owner'
    head = tuple(state['cam'].Outputlayer.parameters())
    assert all(routed.get(id(p)) is None for p in head)
    if not module.protected_e2e:
        assert all(g is None for g in inherited), 'G1 residual gradient entered a historical leaf'
    if not torch.stack([torch.isfinite(x).all() for x in (loss,*[g for g in grads if g is not None])]).all():
        raise FloatingPointError('Nonfinite residual gradient')
    out['total'].backward()
    assert all(p.grad is None for p in new)
    if not torch.stack([torch.isfinite(p.grad).all() for p in old if p.grad is not None]).all():
        raise FloatingPointError('Nonfinite DARG gradient')
    g_base = gradient_hash(state)
    shared = apply_shared_gradient(state,routed) if module.protected_e2e else {}
    g_actual = gradient_hash(state) if module.protected_e2e else g_base
    for p,g in zip(new,fresh):
        p.grad = g
    groups = module.parameter_groups()
    if inspect:
        groups = {**groups,'gate_output':list(module.signed_gate[-1].parameters()),
                  'gate_hidden':list(module.signed_gate[0].parameters())}
    norms = {name:float(torch.stack([p.grad.double().square().sum() for p in ps if p.grad is not None]).sum().sqrt())
             for name,ps in groups.items()}
    if inspect:
        assert (norms['gate_output'] > 0), 'Zero or disconnected gate gradient'
    for optimizer in ordered:
        optimizer.step()
    if module.protected_e2e:
        for name, ps in shared_groups(state).items():
            shared[name]['relative_parameter_drift_from_initial'] = (
                float((flatten(ps).double()-state['shared_origin'][name]).norm() /
                      state['shared_origin'][name].norm().clamp_min(1e-24)) if ps else 0.)
    assert rng_mark() == after_native, 'Residual operations consumed shared RNG'
    native_losses = {key:float(out[key].detach()) for key in (*DARG_LOSSES,'total')}
    values = {key:float(value.detach()) for key,value in terms.items()}
    values['residual_total'] = float(loss.detach())
    return {
        'native':{'logit_sha':sha(logits), 'logp_sha':sha(out['logp']), 'losses':native_losses,
                  'native_gradient_sha':g_base, 'optimizer_gradient_sha':g_actual, 'state':base_digests(state)},
        'losses':values,
        'residual_grads':norms, 'shared_grads':shared,
        'head_residual_grad_count':sum(routed.get(id(p)) is not None for p in head),
        'historical_residual_grad_count':sum(g is not None for g in inherited),
        'mean_margin_gain_legal':float((m_on.detach()-m_off).mean()),
    }


def bundle(state, module, opt):
    return {'base': h.runtime().load_history().state_bundle(state), 'ps_pgr': h.cpu_state(module),
            'opt_prompt': copy.deepcopy(opt.state_dict())}


def load_bundle(state, module, opt, saved):
    h.runtime().load_history().load_bundle(state, saved['base'])
    module.load_state_dict(saved['ps_pgr'], strict=True)
    opt.load_state_dict(copy.deepcopy(saved['opt_prompt']))


def component_digests(value):
    return {**{k: sha(v) for k, v in value['base'].items()},
            **{k: sha(value[k]) for k in ('ps_pgr', 'opt_prompt')}}


def checkpoint_state(state, module, opt):
    current = bundle(state, module, opt)
    base = base_runtime()
    rng = base.pack_rng()
    return dict(current=current, digests=component_digests(current), rng=rng, rng_sha=base.rng_sha(rng))


@torch.no_grad()
def evaluate(state, module):
    out = forward(state, module)
    rows = state['test_np']
    probability = out['logp'].exp()[rows].cpu().numpy()
    legal_cam = out['details']['z'].index_select(0, state['legal_rows']).cpu()
    off_logp = out['details']['off_logp']
    off_probability = off_logp.exp()[rows].cpu().numpy()
    return probability, legal_cam, off_probability, sha(off_logp)


def make_witness(state, module, opt, spectrum):
    step = train_step(state, module, opt)
    train_rng = rng_mark()
    probability, legal_cam, off_probability, off_sha = evaluate(state, module)
    assert rng_mark() == train_rng
    before = sha(h.calibration_state(module))
    observation, after = h.advance_calibration(state, module, legal_cam, spectrum)
    assert after == train_rng[0]
    return dict(native=step['native'], losses=step['losses'],
                rng=train_rng[0], rng_streams=train_rng[1], digests=component_digests(bundle(state, module, opt)),
                probability_sha=sha(probability), legal_cam_sha=sha(legal_cam), ema_before_sha=before,
                ema_after_sha=sha(h.calibration_state(module)), observation_sha=sha(observation),
                off_probability_sha=sha(off_probability), off_logp_sha=off_sha)


def check_synthetic_math():
    semantic = torch.tensor([[-.2, .2, .7, -.7, 0.]])
    llr = torch.tensor([[.9, -.9, -.2, .2, 0.]])
    alpha = torch.full_like(semantic, .5)
    reliable = torch.ones_like(semantic)
    null = torch.zeros(1, 2)
    d = signed_mix(semantic, llr, alpha, reliable, null)
    assert torch.equal(d.sign(), torch.tensor([[1., -1., 1., -1., 0.]]))
    assert torch.equal(signed_mix(semantic, torch.zeros_like(llr), alpha, reliable, null).sign(), semantic.sign())
    p = torch.tensor([[-4., 3., -2., 1., 0.]])
    assert torch.equal(((d * p) * p)[:, :4].sign(), d[:, :4].sign())
    origin = torch.zeros(1, requires_grad=True)
    bounded = origin.clamp_min(0).tanh()
    assert not bounded.count_nonzero() and torch.equal(torch.autograd.grad(bounded.sum(), origin)[0], torch.ones_like(origin))
    values = torch.tensor([0., 1e-44, 1e-40, 1e-30, 1e-15, 1e-9, 1., 1e20])[:, None]
    z = values.expand(-1, 144)
    raw = torch.ones_like(z, requires_grad=True)
    limited = smooth_limit(raw, z, CFG['residual_limit'])
    assert torch.isfinite(limited).all() and not limited[0].count_nonzero()
    allowed = CFG['residual_limit'] + CFG['preflight']['residual_bound_roundoff']
    assert limited.double().norm(dim=1).le(allowed * z.double().norm(dim=1)).all()
    assert torch.isfinite(torch.autograd.grad(limited.sum(), raw)[0]).all()
    raw = torch.zeros_like(raw, requires_grad=True)
    at_zero = smooth_limit(raw, torch.ones_like(raw), CFG['residual_limit'])
    assert torch.equal(torch.autograd.grad(at_zero.sum(), raw)[0], torch.ones_like(raw))


def verify_spectrum(spectrum):
    pack = spectrum['spectrum']
    basis = pack['U']
    assert basis.shape == (144, 16) and basis.dtype == torch.float64
    anchors = basis[basis.abs().argmax(dim=0), torch.arange(16)]
    assert (anchors > 0).all() and (pack['lambda'] > 0).all()
    error = float((basis.T @ basis - torch.eye(16, dtype=basis.dtype)).abs().max())
    assert error < CFG['preflight']['basis_orthogonality_max_abs']
    assert sha(pack) == spectrum['spectra_sha']
    g = torch.linspace(-.9, .9, 16, dtype=basis.dtype)
    d = torch.linspace(.03, -.04, 16, dtype=basis.dtype)
    h0 = spectrum['warmup_cam'][:3].double() @ basis
    flips = torch.where(torch.arange(16) % 2 == 0, 1, -1)
    torch.testing.assert_close((g*d*h0) @ basis.T, (g*d*(h0*flips)) @ (basis*flips).T, atol=0., rtol=0.)
    return error




def protected_files():
    table = read_json(HERE/'manifest.json')['protected_sha256']

    def check(item):
        path, expected = item
        try:
            measured = sha_file(path)
        except OSError as exc:
            return {'path':path, 'error':repr(exc)}
        return {'path':path, 'before':expected, 'after':measured} if expected != measured else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        mismatches = [v for v in pool.map(check, table.items()) if v is not None]
    return {'count':len(table), 'pass':not mismatches, 'mismatches':mismatches}


def preflight():
    import audit as independent

    manifest = read_json(HERE/'manifest.json')
    checks = protected_files()
    assert checks['pass'] and checks['count'], checks
    assert_gpu()
    assert sha_file(h.PAPER) == manifest['source_review']['paper_sha256']
    old = h.HISTORY_ROOT/'experiment82_3_5_ps_scar'
    historical = read_json(old/'config.json')
    for key in ('seed','folds','subjects','start_epoch','max_epochs','early_stop_wait_gt','save_latest_every_epochs',
                'historical_optimizers','psce_optimizer','class_statistics','signed_gate','fusion','feature_math'):
        assert CFG[key] == historical[key], key
    assert torch.__version__ == CFG['preflight']['torch'] and torch.version.cuda == CFG['preflight']['cuda']
    packages = {name:importlib.metadata.version(name) for name in CFG['preflight']['numeric_packages']}
    assert packages == CFG['preflight']['numeric_packages'], packages
    assert CFG['queries']['G1'] == CFG['queries']['G2'] == 'q_medical'
    assert CFG['prompt_rows_0based'] == {'ASD':[4,5,6,7], 'TD':[0,1,2,3]}
    check_synthetic_math()
    projection = independent.check_projection_cases()
    code = implementation()
    previous = read_json(old/'runtime/preflight.json')
    text = torch_load(h.PROMPT_CACHE)
    pre = dict(status='RUNNING', implementation=code, synthetic_checks='PASS',
               projection=projection, protected_files=checks, text=h.verify_text(text), folds=[])
    h.update_manifest(status='PREFLIGHT_RUNNING', implementation=code, config_sha256=code['config.json'],
                      environment={'torch':torch.__version__, 'cuda':torch.version.cuda,
                                   'gpu':torch.cuda.get_device_name(0), 'visible':os.environ['CUDA_VISIBLE_DEVICES'],
                                   'mem_bytes':torch.cuda.get_device_properties(0).total_memory,
                                   'threads_per_worker':torch.get_num_threads(), 'numeric_packages':packages})
    atomic_json(RT/'preflight.json',pre)
    for fold in FOLDS:
        start = time.monotonic()
        state, warm, identity = h.prepare(fold)
        base, historical_runtime = base_runtime(), h.runtime().load_history()
        startup = rng_mark()
        assert (warm['rng_sha'],streams(warm['rng'])) == startup
        original = historical_runtime.state_bundle(state)
        assert sha(original['cam']) == sha(warm['cam'])
        modified = [key for key in original['model'] if not torch.equal(original['model'][key],warm['model'][key])]
        assert set(modified) <= {'calibration','reliability','reliability_gate_logits','adaptive_gate_logits'}
        assert torch.equal(original['model']['calibration'],torch.tensor([[1.,0.]]*4))
        assert torch.equal(original['model']['reliability'],torch.ones(4))
        initialized = h.init_optimizers(state,warm)
        spectrum = h.spectrum_cache(fold)
        cache = h.fixed_stats(state,identity,spectrum)
        assert identity == spectrum['identity'] and np.array_equal(spectrum['legal_train'],h.legal_train(state))
        orthogonality = verify_spectrum(spectrum)
        warm_cam, graph = h.warmup_cam(state)
        assert torch.equal(warm_cam,spectrum['warmup_cam'])
        assert [sha(x) for x in graph['original_adjs']] == spectrum['warmup_graph_sha']
        assert [sha(x) for x in graph['unified_adjs']] == spectrum['warmup_unified_graph_sha']
        assert rng_mark() == startup
        del warm_cam, graph
        probe = independent.historical_probe(state,spectrum)

        proof, baseline_probability, first_rng, _ = independent.primal_step(state)
        trace = {v['epoch']:v for v in h.historical_task(fold)['rng_trace']}
        assert first_rng[0] == trace[START+1]['rng_after_eval']
        first_archive = next(iter(independent.historical_log_rows(fold).values()))
        for epoch in range(START+2,first_archive['epoch']+1):
            archive, archive_probability, mark, _ = independent.primal_step(state)
            assert mark[0] == trace[epoch]['rng_after_eval']
        assert archive['losses']['total'] == first_archive['loss'], (fold,'archived DARG loss')
        assert h.metrics_for(archive_probability,state['labels_np'][state['test_np']])['acc'] == first_archive['metrics']['acc']

        records = {}
        for group in MEDICAL:
            historical_runtime.load_bundle(state,original)
            base.restore_rng(warm['rng'])
            state['shared_origin'] = {name:flatten(ps).double() for name,ps in shared_groups(state).items() if ps}
            module,opt = make_module(state,group,spectrum,cache,text)
            saved = h.cpu_state(module)
            assert sha(saved) == previous['folds'][fold-1]['groups']['G1']['new_module_sha']
            assert rng_mark() == startup
            assert torch.equal(saved['q0'],text[CFG['queries'][group]])
            assert not opt.state and not any(isinstance(x,torch.nn.Dropout) for x in module.modules())
            for name,key in SPECTRUM_BUFFERS.items():
                assert torch.equal(saved[name],spectrum['spectrum'][key].float())
            scales = independent.verify_calibration(module,spectrum,h.legal_train(state))
            inherited = {k:p for k,p in module.named_parameters() if not k.startswith(('signed_gate.','fusion_net.'))}
            assert sum(p.numel() for p in inherited.values()) == 6081
            inherited_sha = sha(inherited)
            assert inherited_sha == previous['folds'][fold-1]['groups']['G1']['inherited_6081_sha']
            assert sha(h.calibration_state(module)) == sha(independent.initial_ema(cache))
            owners = optimizer_ownership(state,module,opt)

            with torch.no_grad():
                zero = forward(state,module)
                plain,z,_ = native_forward(state)
            d = zero['details']
            assert torch.equal(z,d['z']) and torch.equal(z,d['z_enhanced']) and not d['delta'].count_nonzero()
            assert torch.equal(zero['logp'],plain['logp']) and torch.equal(d['off_logp'],plain['logp'])
            assert rng_mark() == startup
            independent.check_formulas(module,d,state['cam'].Outputlayer)
            fd = independent.finite_difference(state,module,d)
            del d, zero, plain, z
            with h.count_calls(state) as calls:
                step = train_step(state,module,opt,inspect=True)
            assert step['native'] == proof, (fold,group,'native first update')
            assert rng_mark() == first_rng
            assert all(v == 0 for v in step['losses'].values())
            assert step['residual_grads']['gate_output'] > 0
            assert all(step['residual_grads'][k] == 0 for k in ('prompt','spectral','fusion_net','gate_hidden'))
            assert step['head_residual_grad_count'] == 0 and all(
                v['applied_norm'] == 0 for v in step['shared_grads'].values())
            assert module.signed_gate[-1].weight.count_nonzero()
            _,legal_cam,off_p,_ = evaluate(state,module)
            assert np.array_equal(off_p,baseline_probability)
            before = h.calibration_state(module)
            observation,_ = h.advance_calibration(state,module,legal_cam,spectrum)
            independent.assert_ema_equal(h.calibration_state(module),independent.ema_update(before,observation))

            second_rng = base.pack_rng()
            _,z,_ = native_forward(state,training=True)
            module.q0.requires_grad_(True)
            value,terms,d,m_on,m_off = residual_objective(state,module,z)
            alt,alt_terms,alt_details = independent.residual_formulas(state,module,z)
            assert torch.equal(value,alt) and all(torch.equal(terms[k],alt_terms[k]) for k in terms)
            params = dict(module.named_parameters())
            watched = {'fixed_query_input':module.q0,'direction':params['direction.weight'],
                       'evidence_projection':params['evidence.weight'],'key':params['key.weight'],'null':module.null,
                       'gate_hidden':params['signed_gate.0.weight'],'fusion':params['fusion_net.2.weight'],
                       'spectral_reliability':params['spectral.2.weight']}
            score_path = torch.autograd.grad(terms['inc'],tuple(watched.values()),retain_graph=True)
            path_norms = {k:float(g.double().norm()) for k,g in zip(watched,score_path)}
            assert all(n>0 and np.isfinite(n) for n in path_norms.values()), (fold,group,path_norms)
            # The supervised derivative w.r.t. actual on Margin must have the correct sign.
            legal_targets = state['y'].index_select(0,state['legal_rows'])
            margin_gradient, = torch.autograd.grad(terms['inc'],m_on,retain_graph=True)
            assert torch.isfinite(margin_gradient).all() and margin_gradient.lt(0).all()
            implicit = m_on - m_off
            assert implicit.shape == legal_targets.shape
            historical_leaves = tuple(p for k in ('model','cam') for p in state[k].parameters() if p.requires_grad)
            routed = torch.autograd.grad(value,historical_leaves,allow_unused=True,retain_graph=True)
            head_ids = {id(p) for p in state['cam'].Outputlayer.parameters()}
            assert all(g is None for p,g in zip(historical_leaves,routed) if id(p) in head_ids)
            assert any(g is not None and bool(g.count_nonzero()) for g in routed) == module.protected_e2e
            labels = state['y']
            changed = 1-labels
            changed[state['legal_rows']] = labels[state['legal_rows']]
            try:
                state['y'] = changed
                _,altered,_,_,_ = residual_objective(state,module,z)
                assert all(torch.equal(terms[k],altered[k]) for k in terms)
            finally:
                state['y'] = labels
                module.q0.requires_grad_(False)
            if fold == 2:
                assert not d['delta'][list(FOLD2_ROWS)].count_nonzero()
            independent.check_formulas(module,d,state['cam'].Outputlayer)
            stats = independent.distribution(d['difference'].detach().double().norm(dim=1).cpu())
            del d, terms, alt_details, routed, z, score_path
            base.restore_rng(second_rng)
            independent.check_interventions(state,module,text,group)
            ckpt = checkpoint_state(state,module,opt)
            expect = make_witness(state,module,opt,spectrum)
            assert expect['rng'] == trace[START+2]['rng_after_eval']
            load_bundle(state,module,opt,ckpt['current'])
            base.restore_rng(ckpt['rng'])
            independent.assert_witness(independent.replay_step(state,module,opt,spectrum),expect)
            records[group] = {'new_module_sha':sha(saved),
                              'shared_init_without_query_sha':sha({k:v for k,v in saved.items() if k!='q0'}),
                              'initial_query_sha':sha(saved['q0']),'inherited_6081_sha':inherited_sha,
                              'optimizer_parameter_tensors':owners,'calls':calls,'zero_identity':True,
                              'first_residual_gradient':step['residual_grads'],'after_update_residual_gradient':path_norms,
                              'evidence_l2':stats,'sensitivity_finite_difference':fd,
                              'calibration':scales,'next_step_replay':True}
        pre['folds'].append({'fold':fold,'identity':identity,'groups':records,
                             'initial_rng':startup[0],'initial_rng_streams':startup[1],
                             'native_first_step':proof,'historical_first_archived_epoch':first_archive['epoch'],
                             'historical_optimizers':initialized,'historical_calibration_modified':modified,
                             'pairs':4096,'pairs_sha':sha(state['train_pairs_np']),'legal_train_sha':sha(h.legal_train(state)),
                             'spectrum_sha':spectrum['spectra_sha'],'basis_orthogonality_max_abs':orthogonality,
                             'stats_sha':cache['stats_sha'],'historical_head_probe':probe,
                             'wall_seconds':time.monotonic()-start})
        atomic_json(RT/'preflight.json',pre)
        print(f'PREFLIGHT fold={fold:02d} PASS',flush=True)
    assert implementation() == code
    pre.update(status='PASS',peak_gpu_bytes=torch.cuda.max_memory_allocated())
    atomic_json(RT/'preflight.json',pre)
    h.update_manifest(status='PREFLIGHT_PASS',preflight_sha=sha_file(RT/'preflight.json'),preflight_passed_folds=len(pre['folds']))
    print('PREFLIGHT PASS 10/10',flush=True)


def passing_preflight():
    path = RT / 'preflight.json'
    manifest = read_json(HERE / 'manifest.json')
    pre = read_json(path)
    assert pre['status'] == 'PASS' and [row['fold'] for row in pre['folds']] == list(FOLDS)
    assert pre['implementation'] == implementation() and manifest['preflight_sha'] == sha_file(path)
    return pre


def task_complete(group, fold):
    task, best, latest, log = paths(group, fold)
    if not task.exists():
        return False
    record = read_json(task)
    assert record['status'] == 'COMPLETE' and record['implementation'] == implementation()
    assert all(record[key] == sha_file(path) for key, path in [('best_sha', best), ('latest_sha', latest), ('log_sha', log)])
    return True


def task_setup(group, fold, pre):
    state, warm, identity = h.prepare(fold)
    spectrum, text = h.spectrum_cache(fold), torch_load(h.PROMPT_CACHE)
    cache = torch_load(h.cache_path(fold))
    module, opt = make_module(state, group, spectrum, cache, text)
    state['shared_origin'] = {name:flatten(ps).double() for name,ps in shared_groups(state).items() if ps}
    record = pre['folds'][fold - 1]
    assert cache['scope']['identity'] == spectrum['identity'] == record['identity'] == identity
    assert np.array_equal(cache['legal_train'], h.legal_train(state))
    assert rng_mark() == (record['initial_rng'], record['initial_rng_streams'])
    metadata = {'fold': fold, 'group': group, 'seed': CFG['seed'], 'identity': identity,
                'implementation': pre['implementation'], 'initial_module_sha': sha(h.cpu_state(module)),
                'stats_sha': cache['stats_sha'], 'spectrum_sha': spectrum['spectra_sha'], 'query_sha': sha(module.q0),
                'initial_ema_sha': sha(h.calibration_state(module)), 'legal_train_sha': sha(h.legal_train(state))}
    arm = anonymous_authorization() if group == 'ANON' else group
    shared = {k:v for k,v in h.cpu_state(module).items() if k != 'q0'}
    assert sha(shared) == record['groups'][arm]['shared_init_without_query_sha']
    if group != 'ANON':
        assert metadata['initial_module_sha'] == record['groups'][group]['new_module_sha']
    metadata.update(protected_e2e=module.protected_e2e, scales=[float(module.tau_s), float(module.tau_l)])
    assert metadata['scales'] == record['groups'][arm]['calibration']['scales']
    assert metadata['legal_train_sha'] == record['legal_train_sha']
    assert rng_sha() == warm['rng_sha']
    return state, module, opt, spectrum, cache, text, metadata


def train_worker(group, fold, pre):
    if task_complete(group, fold):
        return
    start_time = time.monotonic()
    state, module, opt, spectrum, _, _, metadata = task_setup(group, fold, pre)
    task, best_path, latest_path, log_path = paths(group, fold)
    historical_task = h.historical_task(fold)
    historical = {row['epoch']:row for row in historical_task['rng_trace']}
    b0_epoch = historical_task['best_epoch_1based']
    aligned = None
    archived_best = archived_latest = None
    if not module.protected_e2e:
        runner = h.runtime()
        archived_best = torch_load(runner.best_path(BASE_GROUP,fold))
        archived_latest = torch_load(runner.latest_path(BASE_GROUP,fold))
        assert archived_best['best_epoch_1based'] == b0_epoch
    g1_trace = {}
    comparator = 'G1' if group == 'G2' else anonymous_authorization() if group == 'ANON' else None
    if comparator:
        assert task_complete(comparator, fold)
        with paths(comparator, fold)[3].open() as log:
            for line in log:
                row = json.loads(line)
                g1_trace[row['epoch']] = (tuple(row[k] for k in RNG_KEYS), row['rng_before_streams'], row['rng_after_streams'])
    base = base_runtime()
    labels = state['labels_np'][state['test_np']]
    torch.cuda.reset_peak_memory_stats()
    with log_path.open('a+b') as log:
        if latest_path.exists():
            latest = torch_load(latest_path)
            assert all(latest[k] == v for k, v in metadata.items()), 'Latest provenance mismatch'
            h.reconcile_best(best_path, latest['best_sha'])
            digest = h.committed_log(log, latest)
            load_bundle(state, module, opt, latest['current'])
            base.restore_rng(latest['rng'])
            assert component_digests(bundle(state, module, opt)) == latest['digests'] and rng_sha() == latest['rng_sha']
            epoch0, wait, best_correct, best_epoch, best_sha, probability, evaluation_ema = (
                latest[k] for k in ('next_epoch', 'wait', 'best_correct', 'best_epoch', 'best_sha', 'probability', 'last_eval_ema'))
            resumes = latest['resumes'] + [epoch0]
            aligned = latest['b0_epoch_alignment']
            assert START <= epoch0 <= LIMIT and int(module.ema_updates) == epoch0 - START
            if epoch0 < LIMIT and wait <= CFG['early_stop_wait_gt']:
                independent = h.load_module('ps_pgr_resume_audit', HERE / 'audit.py')
                expected = make_witness(state, module, opt, spectrum)
                load_bundle(state, module, opt, latest['current'])
                base.restore_rng(latest['rng'])
                independent.assert_witness(independent.replay_step(state, module, opt, spectrum), expected)
                load_bundle(state, module, opt, latest['current'])
                base.restore_rng(latest['rng'])
            print(f'{group} fold={fold:02d} exact resume from {epoch0}', flush=True)
        else:
            assert not best_path.exists() and log.tell() == 0
            epoch0, wait, best_correct, best_epoch = START, 0, -1, None
            best_sha = probability = evaluation_ema = None
            resumes = []
            digest = hashlib.sha256()
            h.commit_latest(log, digest, latest_path, best_path, None, None,
                                 {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': START, 'wait': 0,
                                  'best_correct':-1, 'best_epoch':None, 'probability':None, 'off_probability':None,
                                  'last_eval_ema':None, 'resumes':[], 'b0_epoch_alignment':None})
        executed, pending = epoch0, None
        previous = rng_mark()
        for epoch in range(epoch0 + 1, LIMIT + 1):
            if wait > CFG['early_stop_wait_gt']:
                break
            before = rng_mark()
            assert before == previous, f'RNG discontinuity {group}/{fold}/{epoch}'
            step = train_step(state, module, opt, inspect=epoch <= START + 2)
            if epoch == START+1:
                assert step['native'] == pre['folds'][fold-1]['native_first_step']
                assert all(v == 0. for v in step['losses'].values())
            after_train = rng_mark()
            probability, legal_cam, off_probability, off_logp_sha = evaluate(state, module)
            after_eval = rng_mark()
            trace = dict(zip(RNG_KEYS, (epoch, before[0], after_train[0], after_eval[0])))
            if epoch in historical:
                assert trace == historical[epoch], f'historical RNG mismatch {group}/{fold}/{epoch}'
            if epoch in g1_trace:
                assert (tuple(trace[k] for k in RNG_KEYS), before[1], after_eval[1]) == g1_trace[epoch], (
                    f'paired RNG mismatch {group}/{fold}/{epoch}')
            assert after_train == after_eval
            prediction = probability.argmax(-1)
            correct = int((prediction == labels).sum())
            evaluation_ema = h.calibration_state(module)
            assert int(evaluation_ema['ema_updates']) == epoch - START - 1
            if epoch == b0_epoch and archived_best is not None:
                assert step['native']['state'] == {k:sha(v) for k,v in archived_best['best_state'].items()}, 'G1 B0 Best state differs'
                assert step['native']['losses'] == archived_best['best_losses']
                assert np.array_equal(off_probability,archived_best['test_probability']), 'G1 B0 Best probabilities differ'
                aligned = {'epoch':epoch,'module':h.cpu_state(module),'off_probability_sha':sha(off_probability),
                           'full_probability_sha':sha(probability),'ema_sha':sha(evaluation_ema)}
            if archived_latest is not None and epoch == archived_latest['next_epoch']:
                assert step['native']['state'] == {k:sha(v) for k,v in archived_latest['current'].items()}, 'G1 B0 Latest state differs'
                assert after_eval == (rng_sha(),streams(archived_latest['rng']))
            improved = correct > best_correct
            if improved:
                best_correct, best_epoch, wait = correct, epoch, 0
                pending = {**metadata, **checkpoint_state(state, module, opt), 'best_epoch': epoch,
                           'calibration_sha': sha(evaluation_ema), 'probability': probability.copy(),
                           'off_probability':off_probability.copy(),
                           'test_indices': state['test_np'].copy(), 'prediction': prediction.copy(),
                           'metrics': h.metrics_for(probability, labels)}
            else:
                wait += 1
            obs, after_ema = h.advance_calibration(state, module, legal_cam, spectrum)
            assert after_ema == after_eval[0] and int(module.ema_updates) == epoch - START
            previous = after_eval
            entry = {**trace, **step, 'correct': correct, 'wait': wait, 'improved': improved, 'best_epoch': best_epoch,
                     'prediction_bits': np.packbits(prediction).tobytes().hex(), 'probability_sha': sha(probability),
                     'off_probability_sha':sha(off_probability), 'off_logp_sha':off_logp_sha,
                     'rng_before_streams': before[1], 'rng_after_streams': after_eval[1], 'rng_after_ema': after_ema,
                     'ema_before': sha(evaluation_ema), 'ema_after': sha(h.calibration_state(module)),
                     'observed_mean': obs['mean'].tolist(), 'observed_second': obs['second'].tolist(),
                     'legal_cam_sha': sha(legal_cam)}
            encoded = (json.dumps(entry, allow_nan=False, separators=(',', ':')) + '\n').encode()
            log.write(encoded)
            digest.update(encoded)
            executed = epoch
            if epoch % SAVE_INTERVAL == 0 or epoch == LIMIT or wait > CFG['early_stop_wait_gt']:
                best_sha = h.commit_latest(log, digest, latest_path, best_path, pending, best_sha,
                                                {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': epoch,
                                                 'wait': wait, 'best_correct': best_correct, 'best_epoch': best_epoch,
                                                 'probability':probability, 'off_probability':off_probability, 'last_eval_ema':evaluation_ema,
                                                 'resumes':resumes, 'b0_epoch_alignment':aligned})
                pending = None
                print(f'{group} f{fold:02d} epoch={epoch} ACC={100*correct/len(labels):.2f} Best={best_epoch}'
                      f'/{100*best_correct/len(labels):.2f}', flush=True)
    assert best_epoch is not None and (executed == LIMIT or wait > CFG['early_stop_wait_gt'])
    assert sha_file(best_path) == best_sha and sha_file(log_path) == digest.hexdigest()
    witness = make_witness(state, module, opt, spectrum)
    atomic_json(task, {**metadata, 'status': 'COMPLETE', 'end_epoch': executed, 'best_epoch': best_epoch,
                       'best_sha': best_sha, 'latest_sha': sha_file(latest_path), 'log_sha': digest.hexdigest(),
                       'next_step_witness': witness, 'resumes': resumes, 'wall_seconds': time.monotonic() - start_time,
                       'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
    print(f'COMPLETE {group} f{fold:02d}', flush=True)


def gpu_mib():
    value = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    return int(value.strip())


def verified_group_gate(kind):
    path = RT / f'{kind}_gate.json'
    gate = read_json(path)
    assert read_json(HERE/'manifest.json')[f'{kind}_gate_sha256'] == sha_file(path)
    assert gate['implementation'] == implementation()
    report = read_json(RESULTS/'audit_status.json')
    entries = [v for v in report['tasks'] if v['group'] in gate['audited_medical_groups']]
    assert sha(entries) == gate['audit_groups_sha'], 'Audited group records have changed'
    for group in gate['audited_medical_groups']:
        for fold in FOLDS:
            assert sha_file(paths(group,fold)[0]) == gate['task_sha256'][f'{group}/{fold}']
    return gate


def g2_authorization():
    gate = verified_group_gate('g2')
    assert gate['audited_medical_groups'] == ['G1'] and gate['best_pass'] == gate['latest_pass'] == 10
    assert all(gate['decision'].values())
    return gate


def anonymous_authorization():
    gate = verified_group_gate('anonymous')
    assert gate['arm'] in gate['audited_medical_groups'] and gate['arm'] in MEDICAL
    assert gate['best_pass'] == gate['latest_pass'] == 10*len(gate['audited_medical_groups'])
    return gate['arm']


def training_authorization(group):
    if group == 'G2':
        g2_authorization()
    if group == 'ANON':
        anonymous_authorization()


def worker_limit(requested):
    assert 1 <= requested <= CFG['workers_max']
    free = int(subprocess.check_output(
        ['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    fitting = max(1,(free-CFG['gpu_free_reserve_mib'])//CFG['worker_memory_mib'])
    cpus = len(os.sched_getaffinity(0))
    return max(1,min(requested,CFG['default_workers'],fitting,cpus//2))


def run_all(group, workers):
    passing_preflight()
    training_authorization(group)
    for directory in (RT/'tasks',RT/'logs',RT/'checkpoints'):
        directory.mkdir(parents=True,exist_ok=True)
    with (RT/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        manage_workers(group,worker_limit(workers))


def manage_workers(group, limit):
    start = time.monotonic()
    finished = {fold for fold in FOLDS if task_complete(group, fold)}
    pending = [(fold, 0) for fold in FOLDS if fold not in finished]
    active, failures = {}, []
    peak, max_workers, last_progress = gpu_mib(), 0, 0.
    path = RT / f'execution_{group}.json'
    prior = read_json(path) if path.exists() else {}
    h.update_manifest(status=f'FORMAL_TRAINING_{group}')
    try:
        while pending or active:
            while pending and len(active) < limit:
                fold, attempt = pending.pop(0)
                logfile = RT / 'logs' / f'{group}_f{fold:02d}.stdout.log'
                stream = logfile.open('a')
                argv = [sys.executable, '-B', str(HERE / 'run.py'), 'worker', '--fold', str(fold), '--group', group]
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE, env=os.environ.copy())
                active[fold] = (process, stream, attempt, logfile)
                max_workers = max(max_workers, len(active))
            for fold, (process, stream, attempt, logfile) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[fold]
                if task_complete(group, fold):
                    finished.add(fold)
                if code == 0:
                    assert fold in finished
                    continue
                tail = logfile.read_text()[-3000:]
                failures.append({'fold': fold, 'attempt': attempt + 1, 'exit_code': code, 'log_tail': tail})
                if (code == 75 or code < 0) and attempt < 2:
                    limit = max(1, limit - 1)
                    pending.insert(0, (fold, attempt + 1))
                    print(f'{group} f{fold:02d} recovering committed Latest after exit {code}', flush=True)
                else:
                    raise RuntimeError(f'{group} fold {fold} stopped with exit {code}\n{tail}')
            now = time.monotonic()
            if now - last_progress >= 15:
                memory = gpu_mib()
                peak = max(peak, memory)
                atomic_json(path, {'status': 'FORMAL_TRAINING', 'group': group,
                                   'completed_folds': len(finished), 'max_workers': max_workers,
                                   'active_folds': sorted(active), 'current_gpu0_mib': memory,
                                   'peak_sampled_gpu0_mib': peak, 'wall_seconds': now - start, 'failures': failures})
                print(f'{group} {len(finished)}/10 active={sorted(active)} GPU0={memory} MiB', flush=True)
                last_progress = now
            if active:
                time.sleep(2)
    finally:
        for process, _, _, _ in active.values():
            process.terminate()
        for process, stream, _, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        status = 'TRAINED_PENDING_AUDIT' if len(finished) == len(FOLDS) else 'TRAINING_STOPPED'
        execution = {'status': status, 'group': group, 'completed_folds': len(finished),
                     'max_workers': max(max_workers, prior.get('max_workers', 0)),
                     'peak_sampled_gpu0_mib': max(peak, prior.get('peak_sampled_gpu0_mib', 0)),
                     'wall_seconds': time.monotonic() - start + prior.get('wall_seconds', 0.),
                     'failures': prior.get('failures', []) + failures}
        atomic_json(path, execution)
        print(f'{group} {len(finished)}/10 {status}', flush=True)
    assert len(finished) == len(FOLDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train', 'worker'))
    parser.add_argument('--workers', type=int, default=CFG['default_workers'])
    parser.add_argument('--fold', type=int, choices=list(FOLDS))
    parser.add_argument('--group', choices=GROUPS, default='G1')
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight()
    elif args.action == 'train':
        run_all(args.group, args.workers)
    else:
        assert_gpu()
        assert args.fold is not None
        training_authorization(args.group)
        pre = passing_preflight()
        try:
            train_worker(args.group, args.fold, pre)
        except torch.OutOfMemoryError as exc:
            print(f'CUDA out of memory: {exc}', flush=True)
            raise SystemExit(75) from exc


if __name__ == '__main__':
    main()
