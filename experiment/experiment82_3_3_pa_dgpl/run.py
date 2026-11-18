#!/usr/bin/env python3
"""Read-only E74.4-G2 restoration, PA-DGPL preflight and formal training."""
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
import importlib.util
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

from pa_dgpl import (EMA_BUFFERS, EPS, FOLD2_ROWS, PADGPL, PREVIOUS, SPECTRUM_BUFFERS,
                     auxiliary_coefficient, decision_gain, gain_gradients)
from da_srpe import DASRPE, smooth_limit

RT, RESULTS = HERE / 'runtime', HERE / 'results'
CFG = json.loads((HERE / 'config.json').read_text())
GROUPS, FOLDS = tuple(CFG['groups']), range(1, CFG['folds'] + 1)
START, LIMIT, SAVE_INTERVAL = (CFG[key] for key in ('start_epoch', 'max_epochs', 'save_latest_every_epochs'))
BASE_GROUP = CFG['base_group']
PARAMETER_GROUPS = tuple(CFG['parameter_groups'])
HIST_OPTS = ('opt_base', 'opt_cam', 'opt_added')
DARG_LOSSES = ('cls', 'graph', 'pair', 'adv', 'preserve')
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')

spec = importlib.util.spec_from_file_location('e8232_readonly_runner', PREVIOUS / 'run.py')
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
sha, sha_file, atomic_json, read_json = legacy.sha, legacy.sha_file, legacy.atomic_json, legacy.read_json
torch_load, write_csv = legacy.torch_load, legacy.write_csv
base_runtime, historical_task = legacy.base_runtime, legacy.historical_task
rng_sha, cpu_state, legal_train = legacy.rng_sha, legacy.cpu_state, legacy.legal_train
METRICS = legacy.METRICS


def update_manifest(**fields):
    path = HERE / 'manifest.json'
    manifest = read_json(path)
    manifest.update(fields)
    atomic_json(path, manifest)


def implementation():
    return {name: sha_file(HERE / name) for name in ('pa_dgpl.py', 'run.py', 'config.json')}


def assert_gpu():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    legacy.runtime()
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


def paths(group, fold):
    name = f'{group}_f{fold:02d}'
    return (RT / 'tasks' / f'{name}.json', RT / 'checkpoints' / f'{name}.best.pt',
            RT / 'checkpoints' / f'{name}.latest.pt', RT / 'logs' / f'{name}.jsonl')


def make_module(state, spectrum, cache, text):
    fold = state['fold']
    with legacy.isolated_rng(CFG['initialization_seed_base'] + fold):
        module = PADGPL(spectrum['spectrum'], text, cache['stats'], 'G2', fold, CFG['class_statistics'], CFG['ema']['alpha'])
    for name, init in (('signed_gate', module.initialize_signed_gate), ('beta', module.initialize_beta)):
        config = CFG[name]
        with legacy.isolated_rng(config['seed_base'] + fold):
            init(config)
    module.to(state['x'].device)
    hp = CFG['psce_optimizer']
    opt = torch.optim.AdamW(module.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    return module, opt


def historical_forward(state, training=False):
    return legacy.runtime().forward(state, BASE_GROUP, training=training)


def forward(state, module, training=False, intervention='full'):
    assert not training or intervention == 'full'
    module.train(training)
    details = {}

    def replace_cam(_head, args):
        assert not details and len(args) == 2 and args[0].shape == (CFG['subjects'], 144)
        before = rng_sha()
        changed, scores = module(args[0], intervention)
        assert changed is scores['z_enhanced'] and rng_sha() == before, 'new CAM path changed RNG'
        details.update(scores)
        return changed, args[1]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(replace_cam)
    try:
        out = historical_forward(state, training)
    finally:
        handle.remove()
    assert details
    out.update(details=details, darg_total=out['total'])
    return out


def check_forward(out):
    d = out['details']
    constraints = CFG['preflight']
    norm = d['z'].norm(dim=-1)
    bound = (CFG['residual_limit'] + constraints['residual_bound_roundoff']) * norm
    tests = [torch.isfinite(x).all() for x in (*d.values(), out['logp'], out['darg_total'])]
    tests += [d[k].norm(dim=-1).le(bound).all() for k in ('delta_base', 'delta')]
    tests += [d['gate'].abs().le(1).all()]
    if 'beta_old' in d:
        tests += [((d['beta'] > 0) & (d['beta'] < 1)).all(), d['beta_adjustment'].abs().le(1).all()]
    for key in ('w', 'w_sp', 'w_sem'):
        if key in d:
            tests += [(d[key] >= 0).all(), (d[key].sum(-1) - 1).abs().le(constraints['row_sum_atol']).all()]
    if not torch.stack(tests).all():
        raise FloatingPointError('nonfinite forward, invalid weights or residual exceeds bound')


def optimizers(state, opt):
    return [state[key] for key in HIST_OPTS] + [opt]


def optimizer_ownership(state, module, opt):
    owned = [{"ids": [id(p) for pg in optimizer.param_groups for p in pg['params']]}
             for optimizer in optimizers(state, opt)]
    ids = [value for group in owned for value in group['ids']]
    assert len(ids) == len(set(ids))
    required = {id(p) for model in (state['model'], state['cam'], module) for p in model.parameters() if p.requires_grad}
    assert required <= set(ids)
    assert owned[-1]['ids'] == [id(p) for p in module.parameters()]
    return [len(o['ids']) for o in owned]


def build_loss(state, module, group, out):
    before = rng_sha()
    d = out['details']
    rows = state['cspe_legal']
    delta, z = d['delta'].index_select(0, rows), d['z'].index_select(0, rows)
    res = (delta.square().sum(-1) / (z.square().sum(-1) + EPS)).mean()
    primary = out['darg_total'] + CFG['residual_loss_coefficient'] * res
    out.update(res=res, primary=primary, total=primary, lambda_gain=0., gradient_protection=None)
    if group == 'G2':
        cfg = CFG['gain']
        gain = decision_gain(out['logp'], d['z_enhanced'], d['delta'], rows, state['y'], cfg)
        restricted, coefficient, diagnostics = gain_gradients(out['cls'], gain['loss'], module.parameter_groups(), cfg)
        out.update(gain=gain, restricted_gain=restricted, lambda_gain=coefficient, gradient_protection=diagnostics,
                   total=primary + coefficient * restricted)
    assert before == rng_sha(), 'new loss or autograd.grad changed RNG'
    return out


def flatten_group(group):
    return torch.cat([p.detach().flatten() for p in group])


def base_gradients(state):
    return {key: {n: p.grad for n, p in state[key].named_parameters()} for key in ('model', 'cam')}


def gradient_snapshot(state, module, out):
    names = {**{f'{kind}.{name}': p for kind, block in [('darg', state['model']), ('cam', state['cam']), ('new', module)]
                for name, p in block.named_parameters() if p.requires_grad}}
    d = out['details']
    params = list(names.values()) + [d['z'], out['fused']]
    cls = torch.autograd.grad(out['cls'], params, retain_graph=True, allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in cls)
    sums = {id(p): float(g.detach().double().square().sum()) if g is not None else 0 for p, g in zip(params, cls)}
    grouped = module.parameter_groups()
    grouped.update({
        'darg': [p for k, p in names.items() if k.startswith('darg.')],
        'cam': [p for k, p in names.items() if k.startswith('cam.') and not k.startswith('cam.Outputlayer.')],
        'output': [p for k, p in names.items() if k.startswith('cam.Outputlayer.')],
        'gate_hidden': list(module.signed_gate[0].parameters()), 'gate_head': list(module.signed_gate[-1].parameters()),
        'beta_head': list(module.beta_net[-1].parameters()), 'cam144': [d['z']], 'fused320': [out['fused']],
    })
    return {key: sum(sums[id(p)] for p in group) ** .5 for key, group in grouped.items()}


def verify_new_gradients(stats, first):
    active = ('darg', 'cam', 'output', 'cam144', 'fused320', 'gate_head')
    dormant = ('prompt', 'spectral', 'beta', 'gate_hidden')
    assert all(stats[k] > 0 for k in active), stats
    assert all((stats[k] == 0 if first else stats[k] > 0) for k in dormant), stats


def train_step(state, module, opt, group, inspect=False):
    ordered = optimizers(state, opt)
    for optimizer in ordered:
        optimizer.zero_grad(set_to_none=True)
    out = build_loss(state, module, group, forward(state, module, training=True))
    check_forward(out)
    diagnostics = out['gradient_protection']
    grouped = module.parameter_groups()
    before = {k: flatten_group(ps) for k, ps in grouped.items()} if diagnostics is not None else {}
    class_grad = gradient_snapshot(state, module, out) if inspect else None
    out['total'].backward()
    grads = [p.grad for block in (state['model'], state['cam'], module) for p in block.parameters() if p.grad is not None]
    if not torch.stack([torch.isfinite(g).all() for g in [out['total'], *grads]]).all():
        raise FloatingPointError('nonfinite total or parameter gradient')
    for optimizer in ordered:
        optimizer.step()
    keys = (*DARG_LOSSES, 'darg_total', 'res', 'total')
    loss = dict(zip(keys, torch.stack([out[k].detach() for k in keys]).cpu().tolist()))
    if diagnostics is not None:
        loss['gain'] = float(out['gain']['loss'].detach())
        gain = out['gain']['gain'].detach()
        details = torch.stack((gain.mean(), (gain > 0).float().mean())).cpu().tolist()
        loss.update(mean_gain=details[0], positive_gain_fraction=details[1])
        update_fractions = torch.stack([(flatten_group(ps) != before[k]).double().mean() for k, ps in grouped.items()])
        for (k, group_diag), fraction in zip(diagnostics.items(), update_fractions.cpu().tolist()):
            group_diag['nonzero_update_fraction'] = fraction
    return {'losses': loss, 'lambda_gain': out['lambda_gain'], 'gradients': diagnostics, 'classification': class_grad}


def bundle(state, module, opt):
    return {'base': legacy.runtime().load_history().state_bundle(state), 'pa_dgpl': cpu_state(module),
            'opt_prompt': copy.deepcopy(opt.state_dict())}


def load_bundle(state, module, opt, saved):
    legacy.runtime().load_history().load_bundle(state, saved['base'])
    module.load_state_dict(saved['pa_dgpl'], strict=True)
    opt.load_state_dict(copy.deepcopy(saved['opt_prompt']))


def component_digests(value):
    return {**{k: sha(v) for k, v in value['base'].items()},
            **{k: sha(value[k]) for k in ('pa_dgpl', 'opt_prompt')}}


def checkpoint_state(state, module, opt):
    current = bundle(state, module, opt)
    base = base_runtime()
    rng = base.pack_rng()
    return {'current': current, 'digests': component_digests(current), 'rng': rng, 'rng_sha': base.rng_sha(rng)}


@torch.no_grad()
def evaluate(state, module):
    out = forward(state, module)
    check_forward(out)
    probability = out['logp'].exp()[state['test_np']].cpu().numpy()
    legal_cam = out['details']['z'].index_select(0, state['cspe_legal']).cpu()
    return probability, legal_cam


def make_witness(state, module, opt, spectrum, group):
    step = train_step(state, module, opt, group)
    train_rng = rng_mark()
    probability, legal_cam = evaluate(state, module)
    eval_rng = rng_mark()
    before = sha(legacy.calibration_state(module))
    observation, after = legacy.advance_calibration(state, module, legal_cam, spectrum)
    assert train_rng == eval_rng and after == eval_rng[0]
    return {'losses': step['losses'], 'lambda_gain': step['lambda_gain'], 'rng': train_rng[0],
            'rng_streams': train_rng[1], 'digests': component_digests(bundle(state, module, opt)),
            'probability_sha': sha(probability), 'legal_cam_sha': sha(legal_cam),
            'ema_before_sha': before, 'ema_after_sha': sha(legacy.calibration_state(module)),
            'observation_sha': sha(observation)}


def preflight():
    import audit as independent

    assert_gpu()
    manifest = read_json(HERE / 'manifest.json')
    assert manifest['config_sha256'] == sha_file(HERE / 'config.json')
    assert manifest['protected_sha256'] and not manifest['historical_expected_hash_mismatches']
    expected_packages = read_json(PREVIOUS / 'manifest.json')['environment']['packages']
    packages = {name: importlib.metadata.version(name) for name in expected_packages}
    assert packages == expected_packages, (packages, expected_packages)
    for key in ('seed', 'folds', 'subjects', 'start_epoch', 'max_epochs', 'early_stop_wait_gt', 'save_latest_every_epochs',
                'initialization_seed_base', 'historical_optimizers', 'psce_optimizer', 'class_statistics', 'ema'):
        assert CFG[key] == legacy.CFG[key], key
    seed_bases = [CFG[k]['seed_base'] for k in ('signed_gate', 'beta')]
    seed_bases += [legacy.CFG['patient_gate']['isolated_seed_base'], CFG['initialization_seed_base']]
    assert len({s + f for s in seed_bases for f in FOLDS}) == len(seed_bases) * len(FOLDS)
    gain = CFG['gain']
    assert auxiliary_coefficient([(0., 1.), (1., 1.)], gain) == 0
    assert auxiliary_coefficient([(1., 0.)], gain) == 0
    assert auxiliary_coefficient([(1e-4, 1.), (1., 1.)], gain) == .5 * 1e-4 / (1. + 1e-12)
    raw = torch.zeros((2, 144), dtype=torch.float64, requires_grad=True)
    z = torch.ones_like(raw)
    assert torch.equal(torch.autograd.grad(smooth_limit(raw, z).sum(), raw)[0], z)
    code = implementation()
    previous_pre = read_json(PREVIOUS / 'runtime/preflight.json')
    text = torch_load(legacy.PROMPT_CACHE)
    pre = {'status': 'RUNNING', 'implementation': code, 'preflight_audit_sha': sha_file(HERE / 'audit.py'),
           'text': legacy.verify_text(text), 'folds': [], 'lambda_edge_cases': 'PASS'}
    update_manifest(status='PREFLIGHT_RUNNING', implementation=code, environment={
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'mem_bytes': torch.cuda.get_device_properties(0).total_memory, 'visible': os.environ['CUDA_VISIBLE_DEVICES'],
        'threads_per_worker': torch.get_num_threads(), 'historical_numeric_packages': packages,
        'numeric_package_prefix': str(local_dependencies)})
    atomic_json(RT / 'preflight.json', pre)
    for fold in FOLDS:
        start = time.monotonic()
        state, warm, identity = legacy.prepare(fold)
        base, hist = base_runtime(), legacy.runtime().load_history()
        startup = rng_mark()
        assert streams(warm['rng']) == startup[1] and warm['rng_sha'] == startup[0]
        original = hist.state_bundle(state)
        deliberate = [k for k, v in original['model'].items() if not torch.equal(v, warm['model'][k])]
        assert set(deliberate) <= {'calibration', 'reliability', 'reliability_gate_logits', 'adaptive_gate_logits'}
        assert torch.equal(original['cam']['Outputlayer.classifier.weight'], warm['cam']['Outputlayer.classifier.weight'])
        assert sha(original['cam']) == sha(warm['cam'])
        assert torch.equal(original['model']['calibration'], torch.tensor([[1., 0.]] * 4))
        assert torch.equal(original['model']['reliability'], torch.ones(4))
        assert original['model']['adaptive_gate_logits'].tolist() == identity['adaptive_gate_theta_initial']
        old_optimizers = legacy.init_optimizers(state, warm)
        spectrum = legacy.spectrum_cache(fold)
        cache = legacy.fixed_stats(state, identity, spectrum)
        assert identity == spectrum['identity'] and np.array_equal(spectrum['legal_train'], legal_train(state))
        historical_cam, cam_out = legacy.warmup_cam(state)
        assert torch.equal(historical_cam, spectrum['warmup_cam'])
        assert [sha(v) for v in cam_out['original_adjs']] == spectrum['warmup_graph_sha']
        assert [sha(v) for v in cam_out['unified_adjs']] == spectrum['warmup_unified_graph_sha']
        assert sha(spectrum['spectrum']) == spectrum['spectra_sha']
        del cam_out, historical_cam
        assert rng_mark() == startup
        hist.zero_optimizers(state)
        with legacy.count_calls(state):
            baseline = historical_forward(state, training=True)
        baseline['total'].backward()
        base_grad_sha = sha(base_gradients(state))
        for key in HIST_OPTS:
            state[key].step()
        base_step_sha = sha(hist.state_bundle(state))
        first_trace = historical_task(fold)['rng_trace'][0]
        assert rng_sha() == first_trace['rng_after_train']
        with torch.no_grad(), legacy.count_calls(state):
            base_eval = historical_forward(state)
        assert rng_sha() == first_trace['rng_after_eval']
        base_loss_sha = sha({k: baseline[k] for k in (*DARG_LOSSES, 'total')})
        base_logp_sha = sha(baseline['logp'])
        del baseline, base_eval

        with legacy.isolated_rng(CFG['initialization_seed_base'] + fold):
            old = DASRPE(spectrum['spectrum'], text, cache['stats'], 'G2', fold, CFG['class_statistics'], CFG['ema']['alpha'])
        inherited_sha = sha(dict(old.named_parameters()))
        assert inherited_sha == previous_pre['folds'][fold - 1]['groups']['G2']['historical_parameter_sha']
        assert sum(p.numel() for p in old.parameters()) == 6081
        group_records, initial_new_sha = {}, None
        for group in GROUPS:
            hist.load_bundle(state, original)
            base.restore_rng(warm['rng'])
            module, opt = make_module(state, spectrum, cache, text)
            new_sha = sha(cpu_state(module))
            assert initial_new_sha in (None, new_sha)
            initial_new_sha = new_sha
            for key, value in old.state_dict().items():
                assert torch.equal(value, module.state_dict()[key].detach().cpu()), key
            assert torch.equal(module.q0.cpu(), text['q_medical'])
            assert sha(legacy.calibration_state(module)) == sha(independent.initial_ema(cache))
            assert not opt.state and rng_mark() == startup
            owner_counts = optimizer_ownership(state, module, opt)
            assert tuple(module.parameter_groups()) == PARAMETER_GROUPS
            for optimizer in optimizers(state, opt):
                optimizer.zero_grad(set_to_none=True)
            with legacy.count_calls(state) as calls:
                out = build_loss(state, module, group, forward(state, module, training=True))
            check_forward(out)
            d = out['details']
            assert not d['gate'].count_nonzero() and not d['delta'].count_nonzero()
            assert torch.equal(d['z'], d['z_enhanced']) and torch.equal(d['beta_adjustment'], torch.zeros_like(d['beta']))
            torch.testing.assert_close(d['beta'], d['beta_old'], atol=1e-7, rtol=0)
            assert sha(out['logp']) == base_logp_sha
            assert sha({**{k: out[k] for k in DARG_LOSSES}, 'total': out['darg_total']}) == base_loss_sha
            first = gradient_snapshot(state, module, out)
            verify_new_gradients(first, True)
            if group == 'G2':
                independent.check_gain(state, out)
                assert out['gradient_protection']['signed_gate']['gain_norm'] > 0
                assert out['gain']['gain'].count_nonzero() == 0
            out['total'].backward()
            assert sha(base_gradients(state)) == base_grad_sha
            for optimizer in optimizers(state, opt):
                optimizer.step()
            assert module.signed_gate[-1].weight.count_nonzero() and sha(hist.state_bundle(state)) == base_step_sha
            assert rng_sha() == first_trace['rng_after_train']
            probability, legal_cam = evaluate(state, module)
            assert rng_sha() == first_trace['rng_after_eval'] and np.isfinite(probability).all()
            old_ema = legacy.calibration_state(module)
            obs, _ = legacy.advance_calibration(state, module, legal_cam, spectrum)
            independent.assert_ema_equal(legacy.calibration_state(module), independent.ema_update(old_ema, obs))
            del out, d
            next_rng = base.pack_rng()
            with legacy.count_calls(state):
                next_out = build_loss(state, module, group, forward(state, module, training=True))
            after = gradient_snapshot(state, module, next_out)
            verify_new_gradients(after, False)
            assert next_out['details']['gate'].detach().std() > 0
            if group == 'G2':
                independent.check_gain(state, next_out)
                # The gain scalar reaches new parameters only, including away from zero initialization.
                old_params = [p for part in (state['model'], state['cam']) for p in part.parameters() if p.requires_grad]
                spill = torch.autograd.grad(next_out['restricted_gain'], old_params, allow_unused=True, retain_graph=True)
                assert all(g is None for g in spill)
            del next_out
            base.restore_rng(next_rng)
            independent.check_interventions(state, module, text, verify_only=True)
            # A live next step and an independently implemented restart must agree bitwise.
            saved = checkpoint_state(state, module, opt)
            expected = make_witness(state, module, opt, spectrum, group)
            load_bundle(state, module, opt, saved['current'])
            base.restore_rng(saved['rng'])
            replay = independent.replay_step(state, module, opt, spectrum, group)
            independent.assert_witness(replay, expected)
            group_records[group] = {'new_module_sha': new_sha, 'inherited_6081_sha': inherited_sha,
                                    'optimizer_parameter_tensors': owner_counts, 'forward_calls': calls,
                                    'zero_identity': True, 'first_classification_gradient': first,
                                    'after_update_classification_gradient': after, 'next_step_replay': True}
        record = {'fold': fold, 'groups': group_records, 'identity': identity, 'historical_optimizers': old_optimizers,
                  'initial_rng': startup[0], 'initial_rng_streams': startup[1], 'original_darg_gradients_sha': base_grad_sha,
                  'original_darg_first_update_sha': base_step_sha, 'original_darg_first_loss_sha': base_loss_sha,
                  'changed_by_historical_calibration': deliberate, 'pairs': 4096,
                  'pairs_sha': sha(state['train_pairs_np']), 'legal_train_sha': sha(legal_train(state)),
                  'spectrum_sha': spectrum['spectra_sha'], 'stats_sha': cache['stats_sha'],
                  'elapsed_seconds': time.monotonic() - start}
        pre['folds'].append(record)
        atomic_json(RT / 'preflight.json', pre)
        print(f'PREFLIGHT fold={fold:02d} PASS', flush=True)
    assert code == implementation()
    pre.update(status='PASS', peak_gpu_bytes=torch.cuda.max_memory_allocated())
    atomic_json(RT / 'preflight.json', pre)
    update_manifest(status='PREFLIGHT_PASS', preflight_sha=sha_file(RT / 'preflight.json'), preflight_passed_folds=len(pre['folds']))
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
    state, warm, identity = legacy.prepare(fold)
    spectrum, text = legacy.spectrum_cache(fold), torch_load(legacy.PROMPT_CACHE)
    cache = torch_load(legacy.cache_path(fold))
    module, opt = make_module(state, spectrum, cache, text)
    record = pre['folds'][fold - 1]
    assert cache['scope']['identity'] == spectrum['identity'] == record['identity'] == identity
    assert np.array_equal(cache['legal_train'], legal_train(state))
    assert rng_mark() == (record['initial_rng'], record['initial_rng_streams'])
    metadata = {'fold': fold, 'group': group, 'seed': CFG['seed'], 'identity': identity,
                'implementation': pre['implementation'], 'initial_module_sha': sha(cpu_state(module)),
                'stats_sha': cache['stats_sha'], 'spectrum_sha': spectrum['spectra_sha'], 'query_sha': sha(module.q0),
                'initial_ema_sha': sha(legacy.calibration_state(module)), 'legal_train_sha': sha(legal_train(state))}
    assert metadata['initial_module_sha'] == record['groups'][group]['new_module_sha']
    assert metadata['legal_train_sha'] == record['legal_train_sha']
    assert rng_sha() == warm['rng_sha']
    return state, module, opt, spectrum, cache, text, metadata


def train_worker(group, fold, pre):
    if task_complete(group, fold):
        return
    start_time = time.monotonic()
    state, module, opt, spectrum, _, _, metadata = task_setup(group, fold, pre)
    task, best_path, latest_path, log_path = paths(group, fold)
    historical = {row['epoch']: row for row in historical_task(fold)['rng_trace']}
    g1_trace = {}
    if group == 'G2':
        assert task_complete('G1', fold)
        with paths('G1', fold)[3].open() as log:
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
            legacy.reconcile_best(best_path, latest['best_sha'])
            digest = legacy.committed_log(log, latest)
            load_bundle(state, module, opt, latest['current'])
            base.restore_rng(latest['rng'])
            assert component_digests(bundle(state, module, opt)) == latest['digests'] and rng_sha() == latest['rng_sha']
            epoch0, wait, best_correct, best_epoch, best_sha, probability, evaluation_ema = (
                latest[k] for k in ('next_epoch', 'wait', 'best_correct', 'best_epoch', 'best_sha', 'probability', 'last_eval_ema'))
            resumes = latest['resumes'] + [epoch0]
            assert START <= epoch0 <= LIMIT and int(module.ema_updates) == epoch0 - START
            print(f'{group} fold={fold:02d} exact resume from {epoch0}', flush=True)
        else:
            assert not best_path.exists() and log.tell() == 0
            epoch0, wait, best_correct, best_epoch = START, 0, -1, None
            best_sha = probability = evaluation_ema = None
            resumes = []
            digest = hashlib.sha256()
            legacy.commit_latest(log, digest, latest_path, best_path, None, None,
                                 {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': START, 'wait': 0,
                                  'best_correct': -1, 'best_epoch': None, 'probability': None, 'last_eval_ema': None, 'resumes': []})
        executed, pending = epoch0, None
        previous = rng_mark()
        for epoch in range(epoch0 + 1, LIMIT + 1):
            if wait > CFG['early_stop_wait_gt']:
                break
            before = rng_mark()
            assert before == previous, f'RNG discontinuity {group}/{fold}/{epoch}'
            step = train_step(state, module, opt, group, inspect=epoch <= START + 2)
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
            evaluation_ema = legacy.calibration_state(module)
            assert int(evaluation_ema['ema_updates']) == epoch - START - 1
            improved = correct > best_correct
            if improved:
                best_correct, best_epoch, wait = correct, epoch, 0
                pending = {**metadata, **checkpoint_state(state, module, opt), 'best_epoch': epoch,
                           'calibration_sha': sha(evaluation_ema), 'probability': probability.copy(),
                           'test_indices': state['test_np'].copy(), 'prediction': prediction.copy(),
                           'metrics': legacy.metrics_for(probability, labels)}
            else:
                wait += 1
            obs, after_ema = legacy.advance_calibration(state, module, legal_cam, spectrum)
            assert after_ema == after_eval[0] and int(module.ema_updates) == epoch - START
            previous = after_eval
            entry = {**trace, **step, 'correct': correct, 'wait': wait, 'improved': improved, 'best_epoch': best_epoch,
                     'prediction_bits': np.packbits(prediction).tobytes().hex(), 'probability_sha': sha(probability),
                     'rng_before_streams': before[1], 'rng_after_streams': after_eval[1], 'rng_after_ema': after_ema,
                     'ema_before': sha(evaluation_ema), 'ema_after': sha(legacy.calibration_state(module)),
                     'observed_mean': obs['mean'].tolist(), 'observed_second': obs['second'].tolist(),
                     'legal_cam_sha': sha(legal_cam)}
            encoded = (json.dumps(entry, allow_nan=False, separators=(',', ':')) + '\n').encode()
            log.write(encoded)
            digest.update(encoded)
            executed = epoch
            if epoch % SAVE_INTERVAL == 0 or epoch == LIMIT or wait > CFG['early_stop_wait_gt']:
                best_sha = legacy.commit_latest(log, digest, latest_path, best_path, pending, best_sha,
                                                {**metadata, **checkpoint_state(state, module, opt), 'next_epoch': epoch,
                                                 'wait': wait, 'best_correct': best_correct, 'best_epoch': best_epoch,
                                                 'probability': probability, 'last_eval_ema': evaluation_ema, 'resumes': resumes})
                pending = None
                print(f'{group} f{fold:02d} epoch={epoch} ACC={100*correct/len(labels):.2f} Best={best_epoch}'
                      f'/{100*best_correct/len(labels):.2f}', flush=True)
    assert best_epoch is not None and (executed == LIMIT or wait > CFG['early_stop_wait_gt'])
    assert sha_file(best_path) == best_sha and sha_file(log_path) == digest.hexdigest()
    witness = make_witness(state, module, opt, spectrum, group)
    atomic_json(task, {**metadata, 'status': 'COMPLETE', 'end_epoch': executed, 'best_epoch': best_epoch,
                       'best_sha': best_sha, 'latest_sha': sha_file(latest_path), 'log_sha': digest.hexdigest(),
                       'next_step_witness': witness, 'resumes': resumes, 'wall_seconds': time.monotonic() - start_time,
                       'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
    print(f'COMPLETE {group} f{fold:02d}', flush=True)


def gpu_mib():
    value = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    return int(value.strip())


def run_all(workers):
    assert 1 <= workers <= CFG['workers_max']
    passing_preflight()
    for directory in (RT / 'tasks', RT / 'logs', RT / 'checkpoints'):
        directory.mkdir(parents=True, exist_ok=True)
    with (RT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        manage_workers(workers)


def manage_workers(limit):
    start = time.monotonic()
    finished = {(g, f) for f in FOLDS for g in GROUPS if task_complete(g, f)}
    pending = [(fold, 0) for fold in FOLDS if any((g, fold) not in finished for g in GROUPS)]
    active, failures = {}, []
    peak, max_workers, last_progress = 0, 0, 0.
    update_manifest(status='FORMAL_TRAINING', formal_launch_unix=time.time())
    try:
        while pending or active:
            while pending and len(active) < limit:
                fold, attempt = pending.pop(0)
                logfile = RT / 'logs' / f'fold{fold:02d}.stdout.log'
                stream = logfile.open('a')
                argv = [sys.executable, '-B', str(HERE / 'run.py'), 'worker', '--fold', str(fold)]
                process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=HERE, env=os.environ.copy())
                active[fold] = (process, stream, attempt, logfile)
                max_workers = max(max_workers, len(active))
            for fold, (process, stream, attempt, logfile) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[fold]
                finished.update((g, fold) for g in GROUPS if task_complete(g, fold))
                if code == 0:
                    assert all((g, fold) in finished for g in GROUPS)
                    continue
                tail = logfile.read_text()[-3000:]
                failures.append({'fold': fold, 'attempt': attempt + 1, 'exit_code': code, 'log_tail': tail})
                if (code == 75 or code < 0) and attempt < 2:
                    limit = max(1, limit - 1)
                    pending.insert(0, (fold, attempt + 1))
                    print(f'f{fold:02d} retry from committed Latest after exit {code}', flush=True)
                else:
                    raise RuntimeError(f'formal fold {fold} stopped with exit {code}\n{tail}')
            now = time.monotonic()
            if now - last_progress >= 15:
                memory = gpu_mib()
                peak = max(peak, memory)
                completed = sum(paths(g, f)[0].exists() for f in FOLDS for g in GROUPS)
                atomic_json(RT / 'execution.json', {'status': 'FORMAL_TRAINING', 'completed_tasks': completed,
                    'max_workers': max_workers, 'active_folds': sorted(active), 'current_gpu0_mib': memory,
                    'peak_sampled_gpu0_mib': peak, 'wall_seconds': now - start, 'failures': failures})
                print(f'{completed}/20 active={sorted(active)} GPU0={memory} MiB', flush=True)
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
        completed = sum(paths(g, f)[0].exists() for f in FOLDS for g in GROUPS)
        status = 'TRAINED_PENDING_INDEPENDENT_AUDIT' if len(finished) == CFG['expected_tasks'] else 'TRAINING_STOPPED'
        execution = {'status': status, 'completed_tasks': completed, 'max_workers': max_workers,
                     'peak_sampled_gpu0_mib': peak, 'wall_seconds': time.monotonic() - start, 'failures': failures}
        atomic_json(RT / 'execution.json', execution)
        update_manifest(status=status, execution=execution, completed_tasks=completed)
    assert len(finished) == CFG['expected_tasks']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'train', 'worker'))
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--fold', type=int, choices=list(FOLDS))
    args = parser.parse_args()
    if args.action == 'preflight':
        preflight()
    elif args.action == 'train':
        run_all(args.workers)
    else:
        assert_gpu()
        assert args.fold is not None
        pre = passing_preflight()
        try:
            for group in GROUPS:
                train_worker(group, args.fold, pre)
        except torch.OutOfMemoryError as exc:
            print(f'CUDA out of memory: {exc}', flush=True)
            raise SystemExit(75) from exc


if __name__ == '__main__':
    main()
