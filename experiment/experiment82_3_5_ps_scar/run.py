#!/usr/bin/env python3
"""GPU0-only PS-SCAR training and recovery from the historical DARG state."""
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
from ps_scar import EPS, FOLD2_ROWS, PSSCAR, SPECTRUM_BUFFERS, deterministic_gap, head_sensitivity, signed_mix, smooth_limit
from torch.nn import functional as F

GROUPS, FOLDS = tuple(CFG['groups']), range(1, CFG['folds'] + 1)
MEDICAL = ('G1', 'G2')
LIMIT, SAVE_INTERVAL = CFG['max_epochs'], CFG['save_latest_every_epochs']
PARAMETER_GROUPS = tuple(CFG['parameter_groups'])
DARG_LOSSES = ('cls', 'graph', 'pair', 'adv', 'preserve')
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')


def implementation():
    names = ('config.json', 'ps_scar.py', 'history.py', 'run.py', 'audit.py', 'report.py')
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
        module = PSSCAR(spectrum['spectrum'], text['q_medical'], cache['stats'], fold, CFG)
    with h.isolated_rng(CFG['fusion']['seed_base'] + fold):
        module.initialize_fusion()
    with h.isolated_rng(CFG['signed_gate']['seed_base'] + fold):
        module.initialize_gate()
    module.to(state['x'].device)
    legal = spectrum['warmup_cam'][h.legal_train(state)].to(state['x'].device)
    module.calibrate(legal)
    module.q0.copy_(text[CFG['queries'][group]])
    module.selective = group == 'G2' or (group == 'ANON' and anonymous_authorization() == 'G2')
    hp = CFG['psce_optimizer']
    opt = torch.optim.AdamW(module.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    return module, opt


def historical_forward(state, training=False):
    return h.runtime().forward(state, BASE_GROUP, training=training)


def forward(state, module, training=False):
    module.train(training)
    captured = []

    def inject(head, args):
        assert not captured and len(args) == 2 and args[0].shape == (CFG['subjects'], 144)
        before = rng_sha()
        gap, jac = head_sensitivity(head, args[0])
        enhanced, details = module(args[0], jac)
        assert enhanced is details['z_enhanced'] and before == rng_sha()
        details['raw_gap_off'] = gap
        captured.append(details)
        return enhanced, args[1]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(inject)
    try:
        out = historical_forward(state, training)
    finally:
        handle.remove()
    assert len(captured) == 1
    out.update(details=captured[0], darg_total=out['total'])
    return out


def check_forward(out):
    d = out['details']
    threshold = CFG['residual_limit'] + CFG['preflight']['residual_bound_roundoff']
    good = [torch.isfinite(t).all() for t in (*d.values(), out['logp'], out['darg_total'])]
    good += [d['delta'].double().norm(dim=-1).le(threshold * d['z'].double().norm(dim=-1)).all(),
             d['gate'].ge(0).all(), d['gate'].le(1).all(),
             d['alpha'].ge(CFG['fusion']['alpha_min']).all(), d['alpha'].le(CFG['fusion']['alpha_max']).all(),
             d['reliability'].ge(.5).all(), d['reliability'].le(1).all()]
    for key in ('attention', 'w_sp'):
        w = d[key]
        good += [w.ge(0).all(), (w.sum(-1) - 1).abs().le(CFG['preflight']['row_sum_atol']).all()]
    if not torch.stack(good).all():
        raise FloatingPointError('Nonfinite forward, invalid fusion, or violation of the relative CAM bound')


def optimizers(state, opt):
    return [state[key] for key in HIST_OPTS] + [opt]


def optimizer_ownership(state, module, opt):
    ordered = [[id(p) for pg in o.param_groups for p in pg['params']] for o in optimizers(state, opt)]
    owned = [key for keys in ordered for key in keys]
    needed = {id(p) for m in (state['model'], state['cam'], module) for p in m.parameters() if p.requires_grad}
    assert len(owned) == len(set(owned)) and needed <= set(owned)
    assert ordered[-1] == [id(p) for p in module.parameters()]
    return [len(keys) for keys in ordered]


def auxiliary_losses(state, out, selective):
    d = out['details']
    rows = state['legal_rows']
    y = state['y'].index_select(0, rows)
    logits = out['logp'].index_select(0, rows)
    sign = 1 - 2 * y
    margin = (logits[:, 0] - logits[:, 1]) * sign
    mass = d['evidence'].index_select(0, rows).sum(-1) + EPS
    selected = mass.gather(1, y[:, None]).squeeze(1)
    nll = mass.sum(-1).log() - selected.log()
    sem = .5 * (nll[y == 0].mean() + nll[y == 1].mean())
    z, delta = (d[k].index_select(0, rows) for k in ('z', 'delta'))
    cfg = CFG['loss']
    importance = ((margin.detach() - cfg['margin_center']) / cfg['margin_temperature']).sigmoid()
    safe = (importance * (delta.square().sum(-1) / (z.square().sum(-1) + EPS))).mean()
    cf = torch.zeros_like(sem)
    if selective:
        off = d['raw_gap_off'].index_select(0, rows) * sign
        on = deterministic_gap(state['cam'].Outputlayer, d['z_enhanced']).index_select(0, rows) * sign
        gain = on - off
        f = cfg['cf_weight_floor']
        hard = f + (1 - 2*f) * (cfg['cf_hard_center'] - off).sigmoid()
        easy = (off - cfg['cf_easy_center']).sigmoid()
        temperature, target = cfg['cf_temperature'], cfg['cf_target_gain']
        shortfall = temperature * (F.softplus((target - gain) / temperature) -
                                   F.softplus(gain.new_tensor(target / temperature)))
        weighted = hard * shortfall
        harm = (easy * (-gain).relu().square()).mean()
        cf = .5 * (weighted[y == 0].mean() + weighted[y == 1].mean()) + cfg['cf_protect'] * harm
    return {'sem': sem, 'safe': safe, 'cf': cf}


class RestrictedAuxiliary(torch.autograd.Function):
    """Carry the scalar value and deliver its precomputed VJP only to new leaves."""

    @staticmethod
    def forward(ctx, value, gradients, *parameters):
        ctx.gradients = gradients
        return value.clone()

    @staticmethod
    def backward(ctx, upstream):
        return (None, None, *(None if g is None else upstream * g for g in ctx.gradients))


def gradient_budget(module, cls, auxiliary):
    groups = module.parameter_groups()
    parameters = tuple(p for ps in groups.values() for p in ps)
    keys = tuple(auxiliary)
    lam = [CFG['loss'][f'lambda_{key}'] for key in keys]
    losses = (cls, *auxiliary.values())
    derivatives = [torch.autograd.grad(loss, parameters, retain_graph=True, create_graph=False, allow_unused=True)
                   if loss.requires_grad else (None,) * len(parameters) for loss in losses]
    stats, offset = [], 0
    for ps in groups.values():
        stop = offset + len(ps)
        c, *other = [
            torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(ps, grad[offset:stop])]).double()
            for grad in derivatives]
        mixed = sum(w * v for w, v in zip(lam, other))
        values = [c.norm(), *(v.norm() for v in other), mixed.norm(), *(torch.dot(c, v) for v in other)]
        stats.append(torch.stack(values))
        offset = stop
    rows = torch.stack(stats)
    if not torch.isfinite(rows).all():
        raise FloatingPointError('Nonfinite classification or auxiliary gradient')
    diagnostics, valid = {}, []
    n = len(keys)
    for name, row in zip(groups, rows.cpu().tolist()):
        cls_norm, aux, mixed, dots = row[0], row[1:1+n], row[1+n], row[2+n:]
        budget = sum(w * g for w, g in zip(lam, aux))
        if budget > 0:
            valid.append(cls_norm / (budget + CFG['loss']['gradient_eps']))
        diag = {'cls_norm': cls_norm, 'mixed_norm': mixed}
        for k, v, dot in zip(keys, aux, dots):
            diag[f'{k}_norm'] = v
            diag[f'cosine_cls_{k}'] = dot / (cls_norm * v) if cls_norm * v else None
        diagnostics[name] = diag
    scale = min(1., CFG['loss']['auxiliary_ratio_max'] * min(valid)) if valid else 0.
    for d in diagnostics.values():
        c = d['cls_norm']
        d['weighted_ratio'] = scale * d['mixed_norm'] / c if c else 0.
        d['budget_ratio'] = scale * sum(w*d[f'{k}_norm'] for w,k in zip(lam,keys)) / c if c else 0.
    accumulated = []
    for gs in zip(*derivatives[1:]):
        active = [w*g for w,g in zip(lam,gs) if g is not None]
        accumulated.append(scale * sum(active) if active else None)
    return parameters, tuple(accumulated), diagnostics, scale


def build_loss(state, module, out):
    before = rng_sha()
    auxiliary = auxiliary_losses(state, out, module.selective)
    params, grads, diagnostics, scale = gradient_budget(module, out['cls'], auxiliary)
    value = scale * sum(CFG['loss'][f'lambda_{k}'] * v.detach() for k, v in auxiliary.items())
    restricted = RestrictedAuxiliary.apply(value, grads, *params)
    out.update(auxiliary, auxiliary_scale=scale, gradients=diagnostics,
               restricted_auxiliary=restricted, total=out['darg_total'] + restricted)
    assert before == rng_sha()
    return out


def base_gradients(state):
    return {key: {name: p.grad for name, p in state[key].named_parameters()} for key in ('model', 'cam')}


def gradient_snapshot(state, module, out):
    groups = module.parameter_groups()
    groups.update({
        'darg': [p for p in state['model'].parameters() if p.requires_grad],
        'cam': [p for name, p in state['cam'].named_parameters() if not name.startswith('Outputlayer.') and p.requires_grad],
        'output': list(state['cam'].Outputlayer.parameters()),
        'gate_hidden': list(module.signed_gate[0].parameters()),
        'gate_head': list(module.signed_gate[-1].parameters()),
        'direction_projection': list(module.direction.parameters()),
        'evidence_projection': list(module.evidence.parameters()),
        'attention_key': list(module.key.parameters()),
        'attention_null': [module.null],
        'cam144': [out['details']['z']], 'fused320': [out['fused']],
    })
    parameters = list(dict.fromkeys(p for values in groups.values() for p in values))
    grads = torch.autograd.grad(out['cls'], parameters, retain_graph=True, allow_unused=True)
    lengths = torch.stack([g.detach().double().square().sum() if g is not None else p.new_zeros((), dtype=torch.float64)
                           for p, g in zip(parameters, grads)])
    assert torch.isfinite(lengths).all()
    norms = dict(zip(map(id, parameters), lengths.cpu().tolist()))
    return {name: sum(norms[id(p)] for p in ps) ** .5 for name, ps in groups.items()}


def verify_new_gradients(g, first):
    assert all(g[k] > 0 for k in ('darg', 'cam', 'output', 'cam144', 'fused320', 'gate_head')), g
    upstream = ('prompt', 'spectral', 'fusion_net', 'gate_hidden', 'direction_projection', 'evidence_projection',
                'attention_key', 'attention_null')
    assert all((g[k] == 0 if first else g[k] > 0) for k in upstream), g


def flatten(ps):
    return torch.cat([p.detach().flatten() for p in ps])


def train_step(state, module, opt, inspect=False):
    ordered = optimizers(state, opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    out = build_loss(state, module, forward(state, module, training=True))
    check_forward(out)
    class_grad = gradient_snapshot(state, module, out) if inspect else None
    groups = module.parameter_groups()
    before = {name: flatten(ps) for name, ps in groups.items()}
    out['total'].backward()
    gradients = [p.grad for block in (state['model'], state['cam'], module) for p in block.parameters()
                 if p.grad is not None]
    if not torch.stack([torch.isfinite(g).all() for g in (out['total'], *gradients)]).all():
        raise FloatingPointError('Nonfinite total objective or parameter gradient')
    for optimizer in ordered:
        optimizer.step()
    changed = torch.stack([(flatten(ps) != before[name]).double().mean() for name, ps in groups.items()])
    for name, fraction in zip(groups, changed.cpu().tolist()):
        out['gradients'][name]['nonzero_update_fraction'] = fraction
    keys = (*DARG_LOSSES, 'darg_total', 'sem', 'safe', 'cf', 'total')
    losses = dict(zip(keys, torch.stack([out[k].detach() for k in keys]).cpu().tolist()))
    return dict(losses=losses, auxiliary_scale=out['auxiliary_scale'], gradients=out['gradients'],
                classification=class_grad)


def bundle(state, module, opt):
    return {'base': h.runtime().load_history().state_bundle(state), 'ps_scar': h.cpu_state(module),
            'opt_prompt': copy.deepcopy(opt.state_dict())}


def load_bundle(state, module, opt, saved):
    h.runtime().load_history().load_bundle(state, saved['base'])
    module.load_state_dict(saved['ps_scar'], strict=True)
    opt.load_state_dict(copy.deepcopy(saved['opt_prompt']))


def component_digests(value):
    return {**{k: sha(v) for k, v in value['base'].items()},
            **{k: sha(value[k]) for k in ('ps_scar', 'opt_prompt')}}


def checkpoint_state(state, module, opt):
    current = bundle(state, module, opt)
    base = base_runtime()
    rng = base.pack_rng()
    return dict(current=current, digests=component_digests(current), rng=rng, rng_sha=base.rng_sha(rng))


@torch.no_grad()
def evaluate(state, module):
    out = forward(state, module)
    check_forward(out)
    probability = out['logp'].exp()[state['test_np']].cpu().numpy()
    legal_cam = out['details']['z'].index_select(0, state['legal_rows']).cpu()
    return probability, legal_cam


def make_witness(state, module, opt, spectrum):
    step = train_step(state, module, opt)
    train_rng = rng_mark()
    probability, legal_cam = evaluate(state, module)
    assert rng_mark() == train_rng
    before = sha(h.calibration_state(module))
    observation, after = h.advance_calibration(state, module, legal_cam, spectrum)
    assert after == train_rng[0]
    return dict(losses=step['losses'], auxiliary_scale=step['auxiliary_scale'],
                rng=train_rng[0], rng_streams=train_rng[1], digests=component_digests(bundle(state, module, opt)),
                probability_sha=sha(probability), legal_cam_sha=sha(legal_cam), ema_before_sha=before,
                ema_after_sha=sha(h.calibration_state(module)), observation_sha=sha(observation))




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




def preflight():
    import audit as independent

    assert_gpu()
    manifest = read_json(HERE / 'manifest.json')
    assert manifest['protected_sha256'] and not manifest['historical_expected_hash_mismatches']
    assert sha_file(h.PAPER) == manifest['source_review']['paper_sha256']
    historical = read_json(h.HISTORY_ROOT / 'experiment82_3_4_ps_dsfr/config.json')
    for key in ('seed', 'folds', 'subjects', 'start_epoch', 'max_epochs', 'early_stop_wait_gt',
                'save_latest_every_epochs', 'historical_optimizers', 'psce_optimizer', 'class_statistics', 'ema'):
        assert CFG[key] == historical[key], key
    expected_packages = CFG['preflight']['numeric_packages']
    packages = {name: importlib.metadata.version(name) for name in expected_packages}
    assert packages == expected_packages, (packages, expected_packages)
    assert CFG['queries']['G1'] == CFG['queries']['G2'] == 'q_medical'
    assert CFG['prompt_rows_0based'] == {'ASD': [4, 5, 6, 7], 'TD': [0, 1, 2, 3]}
    bases = [CFG['initialization_seed_base'], CFG['signed_gate']['seed_base'], CFG['fusion']['seed_base'],
             historical['signed_gate']['seed_base']]
    assert len({seed + fold for seed in bases for fold in FOLDS}) == len(bases) * len(FOLDS)
    check_synthetic_math()
    code = implementation()
    prior = read_json(h.E833 / 'runtime/preflight.json')
    text = torch_load(h.PROMPT_CACHE)
    pre = dict(status='RUNNING', implementation=code, synthetic_checks='PASS',
               text=h.verify_text(text), folds=[])
    h.update_manifest(status='PREFLIGHT_RUNNING', implementation=code, config_sha256=code['config.json'],
                      environment={'torch': torch.__version__, 'cuda': torch.version.cuda,
                                   'gpu': torch.cuda.get_device_name(0), 'visible': os.environ['CUDA_VISIBLE_DEVICES'],
                                   'mem_bytes': torch.cuda.get_device_properties(0).total_memory,
                                   'threads_per_worker': torch.get_num_threads(), 'numeric_packages': packages})
    atomic_json(RT / 'preflight.json', pre)
    for fold in FOLDS:
        start = time.monotonic()
        state, warm, identity = h.prepare(fold)
        base, original_runtime = base_runtime(), h.runtime().load_history()
        startup = rng_mark()
        assert (warm['rng_sha'], streams(warm['rng'])) == startup
        original = original_runtime.state_bundle(state)
        assert sha(original['cam']) == sha(warm['cam'])
        modified = [key for key in original['model'] if not torch.equal(original['model'][key], warm['model'][key])]
        assert set(modified) <= {'calibration', 'reliability', 'reliability_gate_logits', 'adaptive_gate_logits'}
        assert torch.equal(original['model']['calibration'], torch.tensor([[1., 0.]] * 4))
        assert torch.equal(original['model']['reliability'], torch.ones(4))
        old_optimizers = h.init_optimizers(state, warm)
        spectrum = h.spectrum_cache(fold)
        cache = h.fixed_stats(state, identity, spectrum)
        assert identity == spectrum['identity'] and np.array_equal(spectrum['legal_train'], h.legal_train(state))
        orthogonality = verify_spectrum(spectrum)
        original_cam, graph = h.warmup_cam(state)
        assert torch.equal(original_cam, spectrum['warmup_cam'])
        assert [sha(x) for x in graph['original_adjs']] == spectrum['warmup_graph_sha']
        assert [sha(x) for x in graph['unified_adjs']] == spectrum['warmup_unified_graph_sha']
        assert rng_mark() == startup
        del original_cam, graph
        probe = independent.historical_probe(state, spectrum)

        original_runtime.zero_optimizers(state)
        with h.count_calls(state):
            baseline = historical_forward(state, training=True)
        baseline['total'].backward()
        baseline_grad = sha(base_gradients(state))
        baseline_output = sha(baseline['logp'])
        baseline_loss = sha({k: baseline[k] for k in (*DARG_LOSSES, 'total')})
        for key in HIST_OPTS:
            state[key].step()
        baseline_step = sha(original_runtime.state_bundle(state))
        historical_trace = h.historical_task(fold)['rng_trace']
        assert rng_sha() == historical_trace[0]['rng_after_train']
        with torch.no_grad(), h.count_calls(state):
            historical_forward(state)
        assert rng_sha() == historical_trace[0]['rng_after_eval']
        del baseline
        paired_init, group_records = None, {}
        for group in MEDICAL:
            original_runtime.load_bundle(state, original)
            base.restore_rng(warm['rng'])
            module, opt = make_module(state, group, spectrum, cache, text)
            saved_module = h.cpu_state(module)
            common = sha(saved_module)
            assert paired_init in (None, common) and rng_mark() == startup
            paired_init = common
            assert torch.equal(saved_module['q0'], text[CFG['queries'][group]])
            assert not opt.state and not any(isinstance(x, torch.nn.Dropout) for x in module.modules())
            for name, key in SPECTRUM_BUFFERS.items():
                assert torch.equal(saved_module[name], spectrum['spectrum'][key].float())
            scales = independent.verify_calibration(module, spectrum, h.legal_train(state))
            inherited = {k: p for k, p in module.named_parameters() if not k.startswith(('signed_gate.', 'fusion_net.'))}
            assert sum(p.numel() for p in inherited.values()) == 6081
            inherited_sha = sha(inherited)
            assert inherited_sha == prior['folds'][fold - 1]['groups']['G1']['inherited_6081_sha']
            assert sha(h.calibration_state(module)) == sha(independent.initial_ema(cache))
            owners = optimizer_ownership(state, module, opt)

            for optimizer in optimizers(state, opt):
                optimizer.zero_grad(set_to_none=True)
            with h.count_calls(state) as calls:
                out = build_loss(state, module, forward(state, module, training=True))
            check_forward(out)
            d = out['details']
            independent.check_formulas(module, d, state['cam'].Outputlayer)
            fd = independent.finite_difference(state, module, d)
            assert not d['gate'].count_nonzero() and not d['delta'].count_nonzero()
            assert torch.equal(d['z'], d['z_enhanced'])
            assert sha(out['logp']) == baseline_output
            assert sha({**{k: out[k] for k in DARG_LOSSES}, 'total': out['darg_total']}) == baseline_loss
            first = gradient_snapshot(state, module, out)
            verify_new_gradients(first, True)
            assert out['gradients']['prompt']['sem_norm'] > 0 and out['auxiliary_scale'] == 0
            assert not out['cf'].count_nonzero()
            assert (out['gradients']['signed_gate']['cf_norm'] > 0) == module.selective
            out['total'].backward()
            assert sha(base_gradients(state)) == baseline_grad
            for optimizer in optimizers(state, opt):
                optimizer.step()
            assert module.signed_gate[-1].weight.count_nonzero()
            assert sha(original_runtime.state_bundle(state)) == baseline_step
            assert rng_sha() == historical_trace[0]['rng_after_train']
            _, cam = evaluate(state, module)
            assert rng_sha() == historical_trace[0]['rng_after_eval']
            old_ema = h.calibration_state(module)
            obs, _ = h.advance_calibration(state, module, cam, spectrum)
            independent.assert_ema_equal(h.calibration_state(module), independent.ema_update(old_ema, obs))
            del out, d
            next_rng = base.pack_rng()
            module.q0.requires_grad_(True)
            with h.count_calls(state):
                after = build_loss(state, module, forward(state, module, training=True))
            inspected = gradient_snapshot(state, module, after)
            verify_new_gradients(inspected, False)
            auxiliary = independent.auxiliary_formulas(state, after, module.selective)
            assert all(torch.equal(after[key], value) for key,value in auxiliary.items())
            path_tests = [module.q0, module.direction.weight, module.evidence.weight, module.key.weight, module.null]
            sem_path = torch.autograd.grad(after['sem'], path_tests, retain_graph=True)
            sem_norms = [float(x.double().norm()) for x in sem_path]
            assert all(x > 0 and np.isfinite(x) for x in sem_norms)
            assert after['auxiliary_scale'] > 0 and after['details']['gate'].std(dim=0).max() > 0
            old_params = [p for part in (state['model'], state['cam']) for p in part.parameters() if p.requires_grad]
            spill = torch.autograd.grad(after['restricted_auxiliary'], old_params, allow_unused=True, retain_graph=True)
            assert all(grad is None for grad in spill)
            # Perturb every nonlegal label and reuse the same forward graph.
            source_y = state['y']
            changed_y = 1 - source_y
            changed_y[state['legal_rows']] = source_y[state['legal_rows']]
            try:
                state['y'] = changed_y
                recheck = auxiliary_losses(state, after, module.selective)
                assert all(torch.equal(recheck[k], v) for k,v in auxiliary.items())
            finally:
                state['y'] = source_y
            module.q0.requires_grad_(False)
            diff = after['details']['difference'].detach().norm(dim=1).cpu().numpy()
            qstats = dict(zip(('min', 'median', 'p95', 'max'), np.quantile(diff, [0, .5, .95, 1]).tolist()))
            del after
            base.restore_rng(next_rng)
            independent.check_interventions(state, module, text, group)
            saved = checkpoint_state(state, module, opt)
            expect = make_witness(state, module, opt, spectrum)
            assert expect['rng'] == historical_trace[1]['rng_after_eval']
            load_bundle(state, module, opt, saved['current'])
            base.restore_rng(saved['rng'])
            independent.assert_witness(independent.replay_step(state, module, opt, spectrum), expect)
            group_records[group] = {'new_module_sha': common,
                                    'shared_init_without_query_sha': sha({k:v for k,v in saved_module.items() if k!='q0'}),
                                    'initial_query_sha': sha(saved_module['q0']), 'inherited_6081_sha': inherited_sha,
                                    'optimizer_parameter_tensors': owners, 'calls': calls, 'zero_identity': True,
                                    'first_classification_gradient': first, 'after_update_classification_gradient': inspected,
                                    'semantic_path_gradient_norms': dict(zip(('fixed_query_input', 'direction', 'evidence',
                                                                              'key', 'null'), sem_norms)),
                                    'evidence_l2': qstats, 'sensitivity_finite_difference': fd,
                                    'calibration': scales, 'next_step_replay': True}
        pre['folds'].append({'fold': fold, 'identity': identity, 'groups': group_records,
                             'initial_rng': startup[0], 'initial_rng_streams': startup[1],
                             'original_darg_gradients_sha': baseline_grad, 'original_darg_first_update_sha': baseline_step,
                             'original_darg_first_loss_sha': baseline_loss, 'historical_optimizers': old_optimizers,
                             'historical_calibration_modified': modified, 'pairs': 4096,
                             'pairs_sha': sha(state['train_pairs_np']), 'legal_train_sha': sha(h.legal_train(state)),
                             'spectrum_sha': spectrum['spectra_sha'], 'basis_orthogonality_max_abs': orthogonality,
                             'stats_sha': cache['stats_sha'], 'historical_head_probe': probe, 'wall_seconds': time.monotonic() - start})
        atomic_json(RT / 'preflight.json', pre)
        print(f'PREFLIGHT fold={fold:02d} PASS', flush=True)
    assert implementation() == code
    pre.update(status='PASS', peak_gpu_bytes=torch.cuda.max_memory_allocated())
    atomic_json(RT / 'preflight.json', pre)
    h.update_manifest(status='PREFLIGHT_PASS', preflight_sha=sha_file(RT / 'preflight.json'), preflight_passed_folds=10)
    print('PREFLIGHT PASS 10/10', flush=True)




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
    metadata.update(selective=module.selective, scales=[float(module.tau_s), float(module.tau_l)])
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
    historical = {row['epoch']: row for row in h.historical_task(fold)['rng_trace']}
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
            assert START <= epoch0 <= LIMIT and int(module.ema_updates) == epoch0 - START
            if epoch0 < LIMIT and wait <= CFG['early_stop_wait_gt']:
                independent = h.load_module('ps_scar_resume_audit', HERE / 'audit.py')
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
                                  'best_correct': -1, 'best_epoch': None, 'probability': None, 'last_eval_ema': None, 'resumes': []})
        executed, pending = epoch0, None
        previous = rng_mark()
        for epoch in range(epoch0 + 1, LIMIT + 1):
            if wait > CFG['early_stop_wait_gt']:
                break
            before = rng_mark()
            assert before == previous, f'RNG discontinuity {group}/{fold}/{epoch}'
            step = train_step(state, module, opt, inspect=epoch <= START + 2)
            if step['classification'] is not None:
                verify_new_gradients(step['classification'], epoch == START + 1)
            after_train = rng_mark()
            probability, legal_cam = evaluate(state, module)
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
            improved = correct > best_correct
            if improved:
                best_correct, best_epoch, wait = correct, epoch, 0
                pending = {**metadata, **checkpoint_state(state, module, opt), 'best_epoch': epoch,
                           'calibration_sha': sha(evaluation_ema), 'probability': probability.copy(),
                           'test_indices': state['test_np'].copy(), 'prediction': prediction.copy(),
                           'metrics': h.metrics_for(probability, labels)}
            else:
                wait += 1
            obs, after_ema = h.advance_calibration(state, module, legal_cam, spectrum)
            assert after_ema == after_eval[0] and int(module.ema_updates) == epoch - START
            previous = after_eval
            entry = {**trace, **step, 'correct': correct, 'wait': wait, 'improved': improved, 'best_epoch': best_epoch,
                     'prediction_bits': np.packbits(prediction).tobytes().hex(), 'probability_sha': sha(probability),
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
                                                 'probability': probability, 'last_eval_ema': evaluation_ema, 'resumes': resumes})
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


def anonymous_authorization():
    path = RT / 'anonymous_gate.json'
    gate = read_json(path)
    assert read_json(HERE / 'manifest.json')['anonymous_gate_sha256'] == sha_file(path)
    assert gate['implementation'] == implementation()
    assert gate['best_pass'] == gate['latest_pass'] == 20
    assert gate['arm'] in MEDICAL
    assert all(sha_file(paths(g, f)[0]) == gate['task_sha256'][f'{g}/{f}'] for g in MEDICAL for f in FOLDS)
    return gate['arm']


def run_all(group, workers):
    assert 1 <= workers <= CFG['workers_max']
    passing_preflight()
    if group == 'ANON':
        anonymous_authorization()
    if group == 'G2':
        assert all(task_complete('G1', f) for f in FOLDS)
    for directory in (RT / 'tasks', RT / 'logs', RT / 'checkpoints'):
        directory.mkdir(parents=True, exist_ok=True)
    with (RT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manage_workers(group, workers)


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
        if args.group == 'ANON':
            anonymous_authorization()
        pre = passing_preflight()
        try:
            train_worker(args.group, args.fold, pre)
        except torch.OutOfMemoryError as exc:
            print(f'CUDA out of memory: {exc}', flush=True)
            raise SystemExit(75) from exc


if __name__ == '__main__':
    main()
