#!/usr/bin/env python3
"""GPU0-only PS-CER training and transactional checkpoint recovery."""
from __future__ import annotations

import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-019e120d-89d0-0224-0000-000060af1149'
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
import math
import pickle
import subprocess
import time
from contextlib import contextmanager
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / 'config.json').read_text())
sys.path.insert(0, cfg['environment']['readonly_numeric_packages'])

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import history as h
from history import BASE_GROUP, CFG, HIST_OPTS, METRICS, RESULTS, RT, START, sha, sha_file
from ps_cer import (PSCER, anonymous_queries, balanced_mean, gaussian_separation, head_sensitivity,
                     observed_moments, true_margin)
from torch.optim.adam import adam as functional_adam

GROUPS = tuple(CFG['groups'])
FOLDS = tuple(range(1, CFG['folds'] + 1))
LIMIT = CFG['max_epochs']
RNG_KEYS = ('epoch', 'rng_before', 'rng_after_train', 'rng_after_eval')
LOSS_KEYS = ('cls', 'graph', 'pair', 'adv', 'preserve', 'total')
AUX_KEYS = ('semantic', 'spectral', 'safety', 'residual')


def implementation():
    names = ('config.json', 'design_review.md', 'ps_cer.py', 'history.py', 'run.py')
    return {name: sha_file(HERE / name) for name in names}


def gpu_sample():
    query = 'index,uuid,name,memory.total,memory.used,utilization.gpu'
    raw = subprocess.check_output(['nvidia-smi', '-i', '0', f'--query-gpu={query}',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    index, uuid, name, total, used, utilization = (item.strip() for item in raw.split(','))
    assert int(index) == CFG['gpu']['physical_index'] and uuid == CFG['gpu']['uuid']
    return dict(index=int(index), uuid=uuid, name=name, total_mib=int(total), used_mib=int(used),
                utilization_pct=int(utilization))


def environment():
    gpu = gpu_sample()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu['uuid'] and torch.cuda.get_device_name(0) == gpu['name']
    assert torch.__version__.split('+')[0] == CFG['environment']['torch']
    assert torch.version.cuda == CFG['environment']['cuda']
    packages = {key: importlib.metadata.version(key) for key in CFG['environment']['packages']}
    assert packages == CFG['environment']['packages'], packages
    h.runtime()
    assert torch.cuda.current_device() == 0
    assert torch.get_num_threads() == CFG['gpu']['threads_per_worker']
    return dict(gpu=gpu, packages=packages, cuda=torch.version.cuda, torch=torch.__version__,
                visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], python=sys.version)


def protection():
    files = h.read_json(HERE / 'manifest.json')['protected_sha256']

    def check(item):
        path, expected = item
        try:
            actual = sha_file(path)
        except OSError as exc:
            return {'path': path, 'exception': str(exc)}
        return None if actual == expected else {'path': path, 'expected': expected, 'actual': actual}

    with ThreadPoolExecutor(max_workers=8) as pool:
        changed = [value for value in pool.map(check, files.items()) if value is not None]
    return dict(count=len(files), pass_=not changed, differences=changed)


def streams(saved):
    return dict(python=hashlib.sha256(pickle.dumps(saved['python'], protocol=4)).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(saved['numpy'], protocol=4)).hexdigest(),
                cpu=sha(saved['cpu']), cuda=[sha(raw) for raw in saved['cuda']])


def rng_mark():
    base = h.base_runtime()
    packed = base.pack_rng()
    return base.rng_sha(packed), streams(packed)


def init_module(state, spectrum, text, group):
    fold = state['fold']
    legal = h.legal_train(state)
    cam = spectrum['warmup_cam'][legal]
    obs = observed_moments(cam, torch.as_tensor(state['labels_np'][legal]))
    query = text['q_medical']
    if group == 'G2':
        query = anonymous_queries(query, CFG['seeds']['anonymous_tokens'])
    with h.isolated_rng(CFG['seeds']['module'] + fold):
        module = PSCER(spectrum['spectrum']['U'], query, obs, fold, CFG)
    module = module.to(state['x'].device)
    module.calibrate(cam.to(state['x'].device))
    optimizer = torch.optim.AdamW(module.parameters(), **CFG['new_optimizer'])
    return module, optimizer

def parameter_layout(state, module, optimizer):
    old = historical_layout(state)
    new = module.parameter_groups()
    owned = [p for key in HIST_OPTS for pg in state[key].param_groups for p in pg['params']]
    added = [p for group in optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in (*owned, *added)}) == len(owned) + len(added)
    assert {id(p) for p in added} == {id(p) for ps in new.values() for p in ps}
    assert all(new.values())
    return old, {key: tuple(value) for key, value in new.items()}


def historical_layout(state):
    head_ids = {id(p) for p in state['cam'].Outputlayer.parameters()}
    old = {}
    for key in HIST_OPTS:
        for i, pg in enumerate(state[key].param_groups):
            trainable = tuple(p for p in pg['params'] if p.requires_grad)
            if key == 'opt_cam':
                old[f'{key}.{i}.representation'] = tuple(p for p in trainable if id(p) not in head_ids)
                old[f'{key}.{i}.classifier'] = tuple(p for p in trainable if id(p) in head_ids)
            else:
                old[f'{key}.{i}'] = trainable
    assert {id(p) for ps in old.values() for p in ps} == {
        id(p) for key in ('model', 'cam') for p in state[key].parameters() if p.requires_grad}
    assert all(old.values())
    return old


def flatten(parameters, grads=None):
    values = parameters if grads is None else [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]
    return torch.cat([v.detach().flatten() for v in values])


def setup(fold, group):
    state, warmup, identity = h.prepare(fold)
    spectrum, text = h.spectrum_cache(fold), h.torch_load(h.PROMPT_CACHE)
    assert spectrum['identity'] == identity
    assert sha(spectrum['spectrum']) == spectrum['spectra_sha']
    module, optimizer = init_module(state, spectrum, text, group)
    old, new = parameter_layout(state, module, optimizer)
    state.update(old_layout=old, new_layout=new, label_legal=state['y'].index_select(0, state['legal_rows']),
                 origin={key: flatten(ps).double() for key, ps in old.items()}, u_origin=module.U0, llr_scale=module.tau_l)
    assert rng_mark()[0] == warmup['rng_sha']
    meta = dict(group=group, fold=fold, identity=identity, initial_module_sha=sha(h.cpu_state(module)),
                initial_ema_sha=sha(module.ema_state()), spectrum_sha=spectrum['spectra_sha'],
                query_sha=sha(module.q0), legal_train_sha=sha(h.legal_train(state)), implementation=implementation())
    return state, module, optimizer, spectrum, warmup, meta


def historical_forward(state, training, module=None, intervention='full'):
    """Read existing values without altering the historical forward."""
    cam = state['cam']
    head = cam.Outputlayer
    capture, dropouts = {}, []
    dropout = head.mlp_head[-1]
    assert isinstance(dropout, nn.Dropout)

    def pre(_head, args):
        assert 'z' not in capture and args[0].shape == (CFG['subjects'], 144)
        capture['z'] = args[0]

    def local_hook(_attention, args):
        capture['local_heads'] = args[2]

    def context_hook(_ff, _args, output):
        capture['context'] = output

    def head_dropout(_drop, args, value):
        capture['dropout'] = args[0], value

    def raw_logits(_affine, _args, value):
        capture['logits'] = value

    def log_dropout(current, args, _value, name):
        dropouts.append((name, list(args[0].shape), current.training, current.p))

    handles = [head.register_forward_pre_hook(pre), dropout.register_forward_hook(head_dropout),
               head.classifier.register_forward_hook(raw_logits)]
    if module is not None:
        handles += [cam.InputLayer.attention.register_forward_pre_hook(local_hook),
                    cam.FeedForward[-1].register_forward_hook(context_hook)]
    for which in ('model', 'cam'):
        for name, candidate in state[which].named_modules():
            if isinstance(candidate, nn.Dropout):
                handles.append(candidate.register_forward_hook(
                    lambda current, args, value, key=f'{which}.{name}': log_dropout(current, args, value, key)))
    try:
        with torch.set_grad_enabled(training):
            out = h.runtime().forward(state, BASE_GROUP, training=training)
    finally:
        for handle in handles:
            handle.remove()
    assert 'dropout' in capture and 'logits' in capture
    assert sum(name == 'cam.Outputlayer.mlp_head.3' for name, *_ in dropouts) == 1
    if module is not None:
        with torch.set_grad_enabled(training):
            z = capture['z']
            v = capture['local_heads']
            assert v.shape[0] == len(z) and v.shape[2] == 4 and v.shape[1]*v.shape[3] == 36
            local = v.transpose(1, 2).reshape(len(z), 4, 36)
            context = capture['context']
            assert context.shape == local.shape and context.flatten(1).data_ptr() == z.data_ptr()
            jacobian, gap = head_sensitivity(head, z)
            capture['local'] = local
            capture['details'] = module(z, local, context, jacobian, gap, head, intervention)
    return out, capture, dropouts

def mask_from_native(capture):
    x, y = capture['dropout']
    mask, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=False)
    return mask.detach()


def off_scores(head, features, mask=None):
    norm, affine, activation, _ = head.mlp_head
    hidden = activation(affine(norm(features)))
    if mask is not None:
        hidden = hidden * mask
    return head.classifier(hidden)


def total_for_off(state, out, logp):
    # The additional evaluation estimates the reference gradient. It is never
    # added as a second cross-entropy to the actual optimization objective.
    history_loss = h.base_runtime().runtime()['utils'].ClsLoss(logp, state['y'], state['train'], state['weights'])
    return history_loss + out['graph'] + 0.05*out['pair'] + 0.01*out['adv'] + 0.01*out['preserve']


def norm(x):
    return float(x.double().norm())


def cosine(left, right):
    return float(F.cosine_similarity(left.double(), right.double(), dim=0, eps=1e-12))


def split_gradients(layout, gradients):
    start = 0
    for key, parameters in layout.items():
        block = gradients[start:start+len(parameters)]
        assert len(block) == len(parameters)
        yield key, parameters, block, start
        start += len(parameters)
    assert start == len(gradients)


def parameters_of(layout):
    return tuple(p for block in layout.values() for p in block)


def load_gradients(parameters, vector, slots):
    for param, chunk, slot in zip(parameters, vector.split([p.numel() for p in parameters]), slots):
        param.grad = chunk.reshape_as(param).to(param,copy=True) if slot is not None else None


def native_digests(state):
    return {name: sha(state[name].state_dict()) for name in ('model', 'cam', *HIST_OPTS)}


def native_gradient_sha(state):
    return sha({key: {name: p.grad for name, p in state[key].named_parameters()} for key in ('model', 'cam')})


def native_bundle(state):
    return h.runtime().load_history().state_bundle(state)


def save_student(state, module, optimizer):
    return dict(base=native_bundle(state), chc=h.cpu_state(module), opt_new=copy.deepcopy(optimizer.state_dict()))


def restore_student(state, module, optimizer, payload):
    h.runtime().load_history().load_bundle(state, payload['base'])
    module.load_state_dict(payload['chc'], strict=True)
    optimizer.load_state_dict(copy.deepcopy(payload['opt_new']))


@contextmanager
def use_stream(packed):
    base = h.base_runtime()
    outside = base.pack_rng()
    base.restore_rng(packed)
    try:
        yield
    finally:
        base.restore_rng(outside)


@torch.no_grad()
def virtual_step(state, grads):
    """Functional native Adam on isolated parameter, first moment and second moment copies."""
    positions = {}
    for key in HIST_OPTS:
        opt = state[key]
        for group in opt.param_groups:
            params, g, avg, sq, max_sq, steps = [], [], [], [], [], []
            for p in group['params']:
                copy_p = p.detach().clone()
                positions[p] = copy_p
                value = grads.get(p)
                if value is None:
                    continue
                assert not value.is_sparse and not torch.is_complex(p)
                params.append(copy_p)
                g.append(value.detach())
                slot = opt.state.get(p, {})
                avg.append(slot['exp_avg'].clone() if slot else torch.zeros_like(p))
                sq.append(slot['exp_avg_sq'].clone() if slot else torch.zeros_like(p))
                if group['amsgrad']:
                    max_sq.append(slot['max_exp_avg_sq'].clone() if slot else torch.zeros_like(p))
                if slot:
                    steps.append(slot['step'].clone())
                else:
                    step_device = p.device if group['fused'] or group['capturable'] else 'cpu'
                    step_dtype = torch.float64 if torch.get_default_dtype() == torch.float64 and not group['fused'] else torch.float32
                    steps.append(torch.zeros((), dtype=step_dtype, device=step_device))
            assert not group['differentiable']
            if params:
                functional_adam(params, g, avg, sq, max_sq, steps, has_complex=False,
                                beta1=group['betas'][0], beta2=group['betas'][1],
                                **{field: group[field] for field in
                                   ('foreach', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay',
                                    'amsgrad', 'lr', 'weight_decay', 'eps', 'maximize')})
    return positions


def true_kl(reference, student):
    """Exact equality is a stationary point, including in finite precision."""
    diff = reference.double()-student.double()
    per_row = (reference.double().exp()*diff).sum(1)
    return torch.where(diff.eq(0).all(1), torch.zeros_like(per_row), per_row)


def action_losses(state, d, probabilities, teacher):
    rows, y = state['legal_rows'], state['label_legal']
    p = probabilities.index_select(0, rows)
    route = d['route'].index_select(0, rows)
    losses = -p.gather(2, y[:, None, None].expand(-1, 3, 1)).squeeze(2)
    margins = (p[..., 0]-p[..., 1])*(1-2*y[:, None])
    relative = losses[:, 1:]-losses[:, :1]
    risk = balanced_mean((route[:, 1:]*relative).sum(1), y)
    ref_margin = true_margin(teacher['masked'], y)
    ref_cfg = CFG['reference']
    clean = (ref_margin > 0)*( (ref_margin-ref_cfg['reliable_margin'])/ref_cfg['temperature']).sigmoid()
    safety = balanced_mean(clean*(route[:, 1:]*(margins[:, :1]-margins[:, 1:]).relu().square()).sum(1), y)

    local = d['p_local'].index_select(0, rows)
    context = d['p_context'].index_select(0, rows)
    null = d['null'].index_select(0, rows)
    local_true = local.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8)
    context_true = context.gather(1, y[:, None]).squeeze(1).clamp_min(1e-8)
    nonmatch = 1-(local_true*context_true).sqrt().detach()
    semantic = balanced_mean(-0.5*(local_true.log()+context_true.log())+0.25*(null-nonmatch).square(), y)

    separation, _, _ = gaussian_separation(d)
    spec_cfg = CFG['stiefel']
    current = d['projected_total']
    origin = d['projected_origin']
    rank = current.trace().square()/current.square().sum().clamp_min(1e-12)
    original_rank = origin.trace().square()/origin.square().sum().clamp_min(1e-12)
    rank_guard = (spec_cfg['rank_fraction']-rank/original_rank).relu().square()
    anchor = (d['basis']-state['u_origin']).square().sum()/16
    llr = d['llr_total'].index_select(0, rows)/state['llr_scale']
    spectral = (balanced_mean(F.softplus(-(1-2*y)*llr), y) - 0.25*separation.tanh() +
                spec_cfg['anchor_weight']*(anchor+rank_guard))
    relative_norm = d['candidates'].index_select(0, rows).square().sum(2)/(
        d['z'].index_select(0, rows).square().sum(1, keepdim=True)+1e-8)
    residual = balanced_mean((route[:, 1:]*relative_norm).sum(1), y)
    choice = route.argmax(1)
    info = dict(expected_nll=balanced_mean((route*losses).sum(1), y),
                native_balanced_nll=balanced_mean(losses[:, 0], y),
                hard_nll=balanced_mean(losses.gather(1, choice[:, None]).squeeze(1), y),
                margin_delta=margins.gather(1, choice[:, None]).squeeze(1)-margins[:, 0],
                margins=margins, ref_margin=ref_margin, clean=clean, choice=choice, route=route,
                eligible=d['prior'].index_select(0, rows).gt(0).any(1), rank=rank, rank_origin=original_rank,
                separation=separation)
    return risk, (semantic, spectral, safety, residual), info


def functional_loss(state, teacher, off, deterministic_off):
    rows, y = state['legal_rows'], state['label_legal']
    student = off.index_select(0, rows)
    deterministic = deterministic_off.index_select(0, rows)
    assert all(not p.requires_grad and p.shape == student.shape for p in teacher.values())
    masked_kl = balanced_mean(true_kl(teacher['masked'], student), y)
    eval_kl = balanced_mean(true_kl(teacher['deterministic'], deterministic), y)
    weight = CFG['reference']['masked_kl_fraction']
    shift = (teacher['deterministic'][:, 0].exp() - deterministic[:, 0].exp()).abs()
    true_gap_change = true_margin(deterministic, y)-true_margin(teacher['deterministic'], y)
    ref_correct = true_margin(teacher['deterministic'], y) > 0
    harmed = ref_correct & (true_margin(deterministic, y) <= 0)
    return weight*masked_kl+(1-weight)*eval_kl, dict(
        masked_kl=float(masked_kl.detach()), deterministic_kl=float(eval_kl.detach()),
        probability_shift=float(shift.detach().mean()), mean_margin_diff=float(true_gap_change.detach().mean()),
        ref_correct=int(ref_correct.sum()), harmed=int(harmed.sum()))


def scaled_gradient(main, addition, ratio):
    primary_norm = norm(main)
    raw = norm(addition)
    budget = ratio*primary_norm
    scale = min(1., 0.9999*budget/raw) if raw else 0.
    final = (main.double()+scale*addition).to(main.dtype)
    applied = norm(final.double()-main)
    if applied > budget:
        final, applied, scale = main, 0., 0.
    assert torch.isfinite(final).all()
    return final, dict(classification=primary_norm, weighted_aux=raw, applied=applied,
                       ratio=applied/primary_norm if primary_norm else 0., scale=scale)

@torch.no_grad()
def centered_projection(proposal, reference, radius, scale):
    if scale == 1:
        accepted = proposal
    else:
        accepted = ((1-scale)*reference.double()+scale*proposal.double()).to(proposal.dtype)
    distance = norm(accepted.double()-reference)
    return accepted, distance, distance > radius


def trust_scale(distance, radius):
    return min(1., CFG['trust']['fp32_inward_factor']*radius/distance) if distance else 1.


def moment_vector(state, block, field):
    slot = state[block.split('.')[0]].state
    return flatten(state['old_layout'][block], [slot.get(p, {}).get(field) for p in state['old_layout'][block]])


@torch.no_grad()
def apply_reference_trust(state, reference, block, before, ref_before, witness=None):
    hp = CFG['trust']
    ps, rp = state['old_layout'][block], reference['old_layout'][block]
    d = before.numel()
    pieces = block.split('.')
    key = pieces[0] if len(pieces) == 2 else f'{pieces[0]}.{pieces[-1]}'
    origin = state['origin'][block]
    radius = hp['relative_parameter_to_init'][key]*max(norm(origin), math.sqrt(d)*hp['parameter_scale_floor'])
    proposal, center = flatten(ps), flatten(rp)
    distance = norm(proposal.double()-center)
    scale = trust_scale(distance, radius)
    accepted, after, fallback = centered_projection(proposal, center, radius, scale)
    if fallback:
        accepted, after = center, 0.
    slot, ref_slot = state[pieces[0]].state, reference[pieces[0]].state
    for p, ref in zip(ps, rp):
        current, expected = slot.get(p, {}), ref_slot.get(ref, {})
        assert current.keys() == expected.keys()
        if current:
            assert current.keys() == {'step','exp_avg','exp_avg_sq'} and torch.equal(current['step'], expected['step'])
    fields = ('exp_avg', 'exp_avg_sq')
    proposed = {k:moment_vector(state, block, k) for k in fields}
    ref = {k:moment_vector(reference, block, k) for k in fields}
    limit, drift, stats = {}, {}, {}
    for k in fields:
        limit[k] = hp[f'relative_{k}']*max(norm(ref[k]), math.sqrt(d)*hp['moment_scale_floor'])
        drift[k] = norm(proposed[k].double()-ref[k])
    moment_scale = min(trust_scale(drift[k], limit[k]) for k in fields)
    updates = {k:centered_projection(proposed[k], ref[k], limit[k], moment_scale) for k in fields}
    moment_fallback = any(value[2] for value in updates.values())
    final = {k:ref[k] if moment_fallback else updates[k][0] for k in fields}
    for k in fields:
        stats[k] = dict(drift=0. if moment_fallback else updates[k][1], reference=norm(ref[k]), radius=limit[k],
                        student=norm(final[k]), proposal_drift=drift[k])
    assert torch.isfinite(accepted).all() and all(torch.isfinite(v).all() for v in final.values())
    assert (final['exp_avg_sq'] >= 0).all()
    lengths = [p.numel() for p in ps]
    accepted_m, accepted_v = (final[k].split(lengths) for k in fields)
    for p, position, m, v in zip(ps, accepted.split(lengths), accepted_m, accepted_v):
        p.copy_(position.view_as(p))
        if p in slot:
            slot[p]['exp_avg'].copy_(m.view_as(p))
            slot[p]['exp_avg_sq'].copy_(v.view_as(p))
    if witness is not None:
        witness.update(origin=origin, original=before, ref=center, ref_old=ref_before,
                       proposal=proposal, accepted=accepted, proposed_moments=proposed,
                       reference_moments=ref, accepted_moments=final,
                       steps=[slot.get(p, {}).get('step') for p in ps])
    return dict(radius=radius, proposal_reference_distance=distance, accepted_reference_distance=after,
                before_reference_distance=norm(before.double()-ref_before), reference_step=norm(center.double()-ref_before),
                actual_step=norm(accepted.double()-before), native_proposal_step=norm(proposal.double()-before),
                accepted_vs_proposal=norm(accepted.double()-proposal), proposal_scale=scale,
                cast_fallback=fallback, moment_scale=moment_scale, moment_fallback=moment_fallback,
                relative_initial_drift=norm(accepted.double()-origin)/max(norm(origin),1e-12),
                steps_sha=sha([slot.get(p, {}).get('step') for p in ps]), moments=stats)


def native_reference(fold):
    state, warmup, identity = h.prepare(fold)
    state['old_layout'] = historical_layout(state)
    state['label_legal'] = state['y'].index_select(0, state['legal_rows'])
    return state, warmup, identity


@torch.no_grad()
def reference_eval(state):
    out, capture, calls = historical_forward(state, False)
    assert capture['z'].shape[1] == 144
    return dict(legal=out['logp'].index_select(0, state['legal_rows']).detach(),
                probability=out['logp'][state['test_np']].exp().cpu().numpy(),
                gap=(capture['logits'][:, 0]-capture['logits'][:, 1])[state['test_np']].cpu().numpy(),
                dropout_calls=calls)


def reference_step(state):
    for key in HIST_OPTS:
        state[key].zero_grad(set_to_none=True)
    before_rng = rng_mark()
    before = {name: flatten(ps) for name, ps in state['old_layout'].items()}
    out, capture, calls = historical_forward(state, True)
    native_rng = rng_mark()
    mask = mask_from_native(capture)
    assert torch.equal(capture['dropout'][0]*mask, capture['dropout'][1])
    with torch.no_grad():
        det = off_scores(state['cam'].Outputlayer,capture['z']).log_softmax(1)
        legal_teacher = dict(masked=out['logp'].index_select(0,state['legal_rows']).detach(),
                             deterministic=det.index_select(0,state['legal_rows']))
    assert torch.isfinite(out['total'])
    out['total'].backward()
    grads = parameters_of(state['old_layout'])
    assert torch.isfinite(flatten(grads, [p.grad for p in grads])).all()
    gate, scorers = h.runtime()._total_loss_gradient_snapshot(state)
    gradient_sha = native_gradient_sha(state)
    for key in HIST_OPTS:
        state[key].step()
    evaluation = reference_eval(state)
    assert rng_mark() == native_rng
    record = dict(rng_before=before_rng[0], rng_after_train=native_rng[0], rng_after_eval=native_rng[0],
                  rng_streams=native_rng[1], dropout_calls=calls, dropout_mask_sha=sha(mask),
                  logp_sha=sha(out['logp']), logits_sha=sha(capture['logits']), gradient_sha=gradient_sha,
                  gate_grad=gate, scorer_grad=scorers, losses={key:float(out[key].detach()) for key in LOSS_KEYS})
    return legal_teacher, before, evaluation, record


@torch.no_grad()
def learning_cohorts(info, y):
    ref, delta = info['ref_margin'], info['margin_delta']
    template = dict(all=torch.ones_like(y, dtype=torch.bool), ref_correct=ref>0, ref_wrong=ref<=0,
                    ref_confident_wrong=ref<=-2, asd=y==0, td=y==1)
    result = {}
    for key, cohort in template.items():
        n = int(cohort.sum())
        result[key] = dict(n=n, delta_mean=float(delta[cohort].mean()) if n else None,
                           positive=int((delta[cohort]>0).sum()))
    return result


def student_step(state, module, optimizer, teacher, reference, reference_before, inspect=False, probe=None):
    ordered = [state[key] for key in HIST_OPTS]+[optimizer]
    for opt in ordered:
        opt.zero_grad(set_to_none=True)
    module.train()
    rng_before = rng_mark()
    probe_rng = h.base_runtime().pack_rng() if probe is not None else None
    out, capture, calls = historical_forward(state, True, module)
    rng_native = rng_mark()
    d = capture['details']
    mask = mask_from_native(capture)
    assert torch.equal(capture['dropout'][0]*mask, capture['dropout'][1])
    head = state['cam'].Outputlayer
    keep = out['logp']
    disease_arms = [off_scores(head, d['z']+candidate, mask).log_softmax(1)
                    for candidate in d['candidates'].unbind(1)]
    actions = torch.stack((keep, *disease_arms), 1)
    deterministic = off_scores(head, d['z']).log_softmax(1)
    risk, terms, notes = action_losses(state, d, actions, teacher)
    functional, function_report = functional_loss(state, teacher, keep, deterministic)
    old, new = state['old_layout'], state['new_layout']
    old_params, new_params = parameters_of(old), parameters_of(new)
    first_zero_step = int(module.ema_updates) == 0
    if first_zero_step or probe is not None:
        native_grad = torch.autograd.grad(out['total'], old_params, retain_graph=True, allow_unused=True)
    func_grads = torch.autograd.grad(functional, old_params, retain_graph=True, allow_unused=True)
    aux = [torch.autograd.grad(term, new_params, retain_graph=True, allow_unused=True) for term in terms]
    if inspect:
        direct = torch.autograd.grad(out['cls'] + risk, (d['candidates'], *old_params, *new_params),
                                     retain_graph=True, allow_unused=True)
        assert torch.equal(d['enhanced'].detach(), d['z'].detach()+d['delta'].detach())
    if probe is not None:
        probe.update(native_mask=mask, native_dropout=tuple(x.detach() for x in capture['dropout']),
                     off=keep.detach(), actions=actions.detach(), deterministic=deterministic.detach(),
                     teacher=teacher, details={k:v.detach() for k,v in d.items()},
                     local=capture['local'].detach(), context=capture['context'].detach(),
                     risk=float(risk.detach()), functional=float(functional.detach()),
                     pre_step=native_bundle(state), pre_module=h.cpu_state(module), pre_rng=probe_rng,
                     head=copy.deepcopy(head).eval(), shared={})
    assert all(p.grad is None for p in (*old_params, *new_params))
    if first_zero_step:
        # With identical actions the shared risk derivative cancels algebraically.
        # Avoid rounding that would otherwise perturb the historical first Adam step.
        assert torch.equal(actions, keep[:, None, :].expand_as(actions))
        assert int(d['candidates'].count_nonzero()) == 0 and float(risk.detach()) == 0
        for p,g in zip(old_params, native_grad):
            p.grad = g
        for p,g in zip(new_params, torch.autograd.grad(risk, new_params, allow_unused=False)):
            p.grad = g
    else:
        (out['total']+risk).backward()
    full_gradient = tuple(p.grad for p in old_params)
    if probe is not None:
        off_positions = virtual_step(state, dict(zip(old_params, native_grad)))
        full_positions = virtual_step(state, dict(zip(old_params, full_gradient)))
    before = {name:flatten(ps) for name, ps in old.items()}
    native_gradient = native_gradient_sha(state) if inspect else None
    diagnostics, fresh = {}, {}
    for name, ps, full, start in split_gradients(old, full_gradient):
        preserve = flatten(ps, func_grads[start:start+len(ps)])
        main = flatten(ps, full)
        hp = CFG['reference']
        weighted = hp['function_weight']*preserve.double()
        proposed, budget = scaled_gradient(main, weighted, hp['function_norm_budget'])
        load_gradients(ps, proposed, full)
        diagnostics[name] = dict(gradient_full=budget['classification'], retention_gradient=norm(preserve),
                                 retention_injected=budget['applied'], retention_scale=budget['scale'],
                                 function_cosine=cosine(main, preserve))
        if inspect:
            diagnostics[name]['true_classification'] = norm(flatten(ps, direct[1+start:1+start+len(ps)]))
        if probe is not None:
            base_gradient = native_grad[start:start+len(ps)]
            assert [g is not None for g in base_gradient] == [g is not None for g in full]
            probe['shared'][name] = dict(original=before[name], off_gradient=[g.detach().clone() if g is not None else None for g in base_gradient],
                                         full_gradient=[g.detach().clone() if g is not None else None for g in full],
                                         proposal_gradient=[p.grad.detach().clone() if p.grad is not None else None for p in ps],
                                         function_gradient=[g.detach().clone() if g is not None else None for g in func_grads[start:start+len(ps)]],
                                         off=flatten([off_positions[p] for p in ps]), full=flatten([full_positions[p] for p in ps]),
                                         ref=flatten(reference['old_layout'][name]), ref_old=reference_before[name])
    new_vectors = tuple(p.grad for p in new_params)
    assert all(g is not None for g in new_vectors)
    new_main = flatten(new_params, new_vectors)
    auxiliary_vectors = [flatten(new_params, g) for g in aux]
    extra = sum((CFG['loss'][key]*g.double() for key,g in zip(AUX_KEYS,auxiliary_vectors)))
    new_final, auxiliary_budget = scaled_gradient(new_main, extra, CFG['loss']['aux_ratio_max'])
    for name, ps, main, start in split_gradients(new, new_vectors):
        vector = flatten(ps, main)
        if inspect:
            i = 1+len(old_params)+start
            compare = flatten(ps, direct[i:i+len(ps)])
            assert torch.equal(vector, compare), name
        lo = sum(p.numel() for p in new_params[:start])
        sl = slice(lo,lo+len(vector))
        update = new_final[sl]
        load_gradients(ps, update, main)
        update = flatten(ps,tuple(p.grad for p in ps))
        fresh[name] = dict(classification=norm(vector), applied=norm(update.double()-vector),
                           gradient_updated_fraction=float((update!=0).float().mean()),
                           terms={key:dict(norm=norm(g[sl]), cosine=cosine(vector,g[sl]))
                                  for key,g in zip(AUX_KEYS,auxiliary_vectors)})
    applied_new = norm(flatten(new_params,tuple(p.grad for p in new_params)).double()-new_main)
    assert applied_new <= CFG['loss']['aux_ratio_max']*auxiliary_budget['classification']
    auxiliary_budget['applied'] = applied_new
    auxiliary_budget['ratio'] = applied_new/auxiliary_budget['classification'] if auxiliary_budget['classification'] else 0.
    if probe is not None:
        proposal_positions = virtual_step(state, {p:p.grad for p in old_params})
        for name, ps in old.items():
            probe['shared'][name]['virtual_proposal'] = flatten([proposal_positions[p] for p in ps])
    new_before = {name:flatten(ps) for name,ps in new.items()}
    for opt in ordered:
        opt.step()
    module.project_strength()
    with torch.no_grad():
        for name, ps in new.items():
            movement = flatten(ps).double()-new_before[name]
            fresh[name]['actual_step_norm'] = norm(movement)
            fresh[name]['actual_update_fraction'] = float((movement != 0).float().mean())
        for name in old:
            witness = probe['shared'][name] if probe is not None else None
            diagnostics[name].update(apply_reference_trust(state, reference, name, before[name],
                                                           reference_before[name], witness))
            if witness is not None:
                assert torch.equal(witness['virtual_proposal'], witness['proposal']), name
                witness['report'] = diagnostics[name]
    assert rng_mark() == rng_native
    legal = state['label_legal']
    choice = notes['choice']
    result = dict(rng_before=rng_before[0], rng_after_train=rng_native[0], rng_after_eval=rng_native[0],
                  rng_streams=rng_native[1], dropout_calls=calls, dropout_mask_sha=sha(mask),
                  losses={key:float(out[key].detach()) for key in LOSS_KEYS}, route_risk=float(risk.detach()),
                  auxiliary={key:float(term.detach()) for key,term in zip(AUX_KEYS,terms)},
                  aux_budget=auxiliary_budget, function=function_report,
                  routing=dict(expected_nll=float(notes['expected_nll'].detach()), hard_nll=float(notes['hard_nll'].detach()),
                               keep_nll=float(notes['native_balanced_nll'].detach()), eligible=int(notes['eligible'].sum()),
                               route_counts=torch.bincount(choice,minlength=3).cpu().tolist(),
                               mean_prob=notes['route'].detach().mean(0).cpu().tolist(),
                               entropy=float(torch.special.entr(notes['route'].detach()).sum(1).mean())),
                  shared=diagnostics, added=fresh, cohorts=learning_cohorts(notes,legal),
                  separation=float(notes['separation'].detach()),
                  participation_rank=float(notes['rank'].detach()), origin_rank=float(notes['rank_origin'].detach()))
    if inspect:
        result['proof'] = dict(logits_sha=sha(capture['logits']), logp_sha=sha(keep), off_logp_sha=sha(keep),
                               native_gradient_sha=native_gradient, state=native_digests(state),
                               delta_nonzero=int(d['delta'].count_nonzero()), delta_class_grad=norm(direct[0]),
                               gain_to_u=norm(direct[1+len(old_params)]),
                               candidate_nonzero=int(d['candidates'].count_nonzero()),
                               initial_neutral_scores=int(d['route_scores'].count_nonzero())==0,
                               local_shape=list(capture['local'].shape), context_shape=list(capture['context'].shape))
    return result

@torch.no_grad()
def evaluate(state, module, intervention='full'):
    module.eval()
    out, capture, calls = historical_forward(state, False, module, intervention)
    d = capture['details']
    head = state['cam'].Outputlayer
    logits = off_scores(head, d['enhanced'])
    full = logits.log_softmax(1)
    off = out['logp']
    rows = state['test_np']
    return dict(probability=full[rows].exp().cpu().numpy(), off_probability=off[rows].exp().cpu().numpy(),
                gap=(logits[:,0]-logits[:,1])[rows].cpu().numpy(), off_gap=(off[:,0]-off[:,1])[rows].cpu().numpy(),
                legal_cam=d['z'].index_select(0, state['legal_rows']).cpu(), dropout_calls=calls)

def setup_fold(fold):
    ref, warmup, identity = native_reference(fold)
    base = h.base_runtime()
    rng = dict(reference=base.pack_rng())
    students, meta = {}, {}
    for group in GROUPS:
        state, module, opt, spectrum, warmup_s, meta[group] = setup(fold, group)
        assert identity == meta[group]['identity'] and warmup_s['rng_sha'] == warmup['rng_sha']
        assert native_digests(state) == native_digests(ref)
        assert np.array_equal(state['train_pairs_np'], ref['train_pairs_np'])
        students[group] = (state, module, opt)
        rng[group] = base.pack_rng()
    assert len(set(base.rng_sha(x) for x in rng.values())) == 1
    assert meta['G1']['query_sha'] != meta['G2']['query_sha']
    runtime = h.runtime()
    archive = h.historical_task(fold)
    old_best = h.torch_load(runtime.best_path(BASE_GROUP, fold))
    old_latest = h.torch_load(runtime.latest_path(BASE_GROUP, fold))
    assert old_best['identity'] == old_latest['identity'] == identity
    assert sha_file(runtime.best_path(BASE_GROUP, fold)) == archive['checkpoint_sha256']
    assert old_latest['next_epoch'] == archive['epochs_executed']
    assert archive['best_epoch_1based'] == old_best['best_epoch_1based']
    archive_rng = {x['epoch']: x for x in archive['rng_trace']}
    assert list(archive_rng) == list(range(START+1, archive['epochs_executed']+1))
    return SimpleNamespace(fold=fold, reference=ref, students=students, meta=meta, rng=rng,
                           identity=identity, warmup=warmup, spectrum=spectrum, archive=archive,
                           old_best=old_best, old_latest=old_latest, archive_rng=archive_rng)


def execute_epoch(ctx, active, inspect=False, probes=None):
    base = h.base_runtime()
    with use_stream(ctx.rng['reference']):
        teacher, before, ref_eval, ref_step = reference_step(ctx.reference)
        ctx.rng['reference'] = base.pack_rng()
    steps, evals = {}, {}
    for group in active:
        state, module, optimizer = ctx.students[group]
        with use_stream(ctx.rng[group]):
            steps[group] = student_step(state, module, optimizer, teacher, ctx.reference, before,
                                        inspect=inspect, probe=None if probes is None else probes[group])
            evals[group] = evaluate(state, module)
            assert steps[group]['rng_after_eval'] == rng_mark()[0]
            ctx.rng[group] = base.pack_rng()
        for key in ('rng_before', 'rng_after_train', 'rng_after_eval', 'rng_streams', 'dropout_mask_sha', 'dropout_calls'):
            assert steps[group][key] == ref_step[key], f'{group} {key} disagrees with reference'
        assert evals[group]['dropout_calls'] == ref_eval['dropout_calls']
    return steps, evals, ref_step, ref_eval


def update_statistics(state, module, evaluation):
    obs = observed_moments(evaluation['legal_cam'], state['label_legal'])
    module.update_ema(obs)
    return sha(obs)


def archive_check(ctx, epoch, step, evaluated):
    if epoch in ctx.archive_rng:
        assert all(step[key] == ctx.archive_rng[epoch][key] for key in RNG_KEYS[1:]), (ctx.fold, epoch)
    result = {}
    if epoch == ctx.archive['best_epoch_1based']:
        saved = ctx.old_best
        assert np.array_equal(evaluated['probability'], saved['test_probability']), (ctx.fold, epoch, 'probability')
        assert np.array_equal(evaluated['probability'].argmax(1), saved['test_prediction'])
        assert step['losses'] == saved['best_losses'], (ctx.fold, epoch, 'losses')
        assert step['gate_grad'] == ctx.old_latest['best']['total_loss_gate_grad']
        assert step['scorer_grad'] == ctx.old_latest['best']['total_loss_scorer_grad']
        assert step['rng_after_train'] == saved['best_training_rng_sha'] and step['rng_after_eval'] == saved['best_eval_rng_sha']
        found = native_digests(ctx.reference)
        assert found == {key:sha(value) for key, value in saved['best_state'].items()}, (ctx.fold, epoch, 'best weights/optimizer')
        result['archive_best'] = dict(epoch=epoch, state=found, evaluation=sha(evaluated['probability']),
                                      gradient=step['gradient_sha'], gate_gradient=step['gate_grad'],
                                      scorer_gradient=step['scorer_grad'], losses=step['losses'],
                                      rng=step['rng_after_eval'])
    if epoch == ctx.archive['epochs_executed']:
        saved = ctx.old_latest
        found = native_digests(ctx.reference)
        assert found == {key:sha(value) for key, value in saved['current'].items()}, (ctx.fold, epoch, 'terminal weights/optimizer')
        assert h.base_runtime().rng_sha(saved['rng']) == step['rng_after_eval']
        assert streams(saved['rng']) == step['rng_streams']
        result['archive_terminal'] = dict(epoch=epoch, state=found, rng=step['rng_after_eval'])
    return result


def native_state_proof(ctx):
    baseline, _, _ = h.prepare(ctx.fold)
    with use_stream(ctx.rng['reference']):
        head_records = {}

        def log_logits(_layer, _args, value):
            head_records['logit'] = value

        device_head = baseline['cam'].Outputlayer
        hook_logits = device_head.classifier.register_forward_hook(log_logits)
        try:
            runtime = h.runtime()
            runtime.load_history().zero_optimizers(baseline)
            out = runtime.forward(baseline, BASE_GROUP, training=True)
            out['total'].backward()
            gradient = native_gradient_sha(baseline)
            for key in HIST_OPTS:
                baseline[key].step()
        finally:
            hook_logits.remove()
        native = dict(gradient_sha=gradient, state=native_digests(baseline), logits_sha=sha(head_records['logit']),
                      logp_sha=sha(out['logp']), losses={key:float(out[key].detach()) for key in LOSS_KEYS},
                      rng=rng_mark()[0])
    return native


def freeze_initial_preflight():
    expected = h.read_json(HERE/'manifest.json')
    oldhash = expected['implementation'] if 'implementation' in expected else implementation()
    assert implementation() == oldhash, 'Pass or explicitly invalidate the earlier preflight before altering code'
    h.update_manifest(implementation=oldhash)


def preflight_fold(fold):
    import audit

    begun = time.monotonic()
    ctx = setup_fold(fold)
    raw = native_state_proof(ctx)
    initial = native_digests(ctx.reference)
    identities = h.init_optimizers(ctx.reference,ctx.warmup)
    b0 = audit.baseline(ctx.reference,ctx.identity,dict(current={'base':native_bundle(ctx.reference)}))
    assert native_digests(ctx.reference)==initial
    inputs = dict(legal=sha(h.legal_train(ctx.reference)), test=sha(ctx.reference['test_np']),
                  pairs=sha(ctx.reference['train_pairs_np']), pair_targets=sha(ctx.reference['train_same_np']))
    legal = audit.rows_legal(ctx.reference)
    init_moments = audit.historical_observation(ctx.spectrum['warmup_cam'][legal], ctx.reference['labels_np'][legal])
    initial_modules = {}
    for group, (state, module, optimizer) in ctx.students.items():
        assert torch.equal(module.U0, module.basis())
        for key, value in init_moments.items():
            assert sha(value) == sha(module.ema_state()[key])
        initial_modules[group] = h.cpu_state(module)
    common = initial_modules['G1'].keys() & initial_modules['G2'].keys() - {'q0'}
    assert initial_modules['G1'].keys() == initial_modules['G2'].keys()
    assert all(torch.equal(initial_modules['G1'][key], initial_modules['G2'][key]) for key in common)
    q1,q2 = (initial_modules[g]['q0'] for g in GROUPS)
    assert not torch.equal(q1,q2) and torch.allclose(q1.norm(dim=1),q2.norm(dim=1),atol=1e-7,rtol=0)
    class_mass = np.bincount(ctx.reference['labels_np'][ctx.reference['train_np']],minlength=2)*ctx.reference['weights'].cpu().numpy()
    class_mass = (class_mass/class_mass.sum()).tolist()
    with use_stream(ctx.rng['reference']):
        assert sha(h.warmup_cam(ctx.reference)[0]) == ctx.spectrum['feature_sha']
    h.fixed_stats(ctx.reference, ctx.identity, ctx.spectrum)
    steps, evaluation, refstep, ref_eval = execute_epoch(ctx, GROUPS, inspect=True)
    ref_first = archive_check(ctx, START+1, refstep, ref_eval)
    assert all(refstep[name] == raw[name] for name in ('gradient_sha', 'logits_sha', 'logp_sha', 'losses'))
    assert native_digests(ctx.reference) == raw['state'] and refstep['rng_after_train'] == raw['rng']
    for group, step in steps.items():
        proof = step['proof']
        assert proof['state'] == raw['state'] and step['losses'] == raw['losses']
        assert proof['logits_sha'] == raw['logits_sha'] and proof['logp_sha'] == proof['off_logp_sha'] == raw['logp_sha']
        assert proof['native_gradient_sha'] == raw['gradient_sha']
        assert proof['delta_nonzero'] == 0 and proof['delta_class_grad'] > 0
        assert all(v['accepted_reference_distance'] == 0 and v['accepted_vs_proposal'] == 0
                   for v in step['shared'].values())
        assert proof['initial_neutral_scores'] and proof['candidate_nonzero'] == 0
        assert np.array_equal(evaluation[group]['off_probability'], ref_eval['probability'])
        update_statistics(ctx.students[group][0], ctx.students[group][1], evaluation[group])
    group_proofs = {}
    for index in range(2, 11):
        probes = {group:{} for group in GROUPS}
        next_steps, next_eval, refsecond, refeval2 = execute_epoch(ctx, GROUPS, inspect=True, probes=probes)
        archive_check(ctx, START+index, refsecond, refeval2)
        for group, (state, module, optimizer) in ctx.students.items():
            if group not in group_proofs:
                step, probe = next_steps[group], probes[group]
                proof = step['proof']
                exercised = proof['candidate_nonzero']
                grads = {k:x['classification'] for k,x in step['added'].items()}
                if exercised and proof['gain_to_u'] > 0 and all(v > 0 for v in grads.values()):
                    assert all(info['true_classification'] > 0 for info in step['shared'].values())
                    with use_stream(ctx.rng[group]):
                        checked = audit.check_training_probe(state, module, probe)
                        diagnostics = audit.audit_formula(state, module)
                    group_proofs[group] = dict(meta=ctx.meta[group], first=steps[group]['proof'], second=proof,
                                               first_rng=steps[group]['rng_after_train'],
                                               second_rng=step['rng_after_train'], nonzero_gradient_epoch=START+index,
                                               gradients=grads,
                                               shared={k:x['true_classification'] for k,x in step['shared'].items()},
                                               optimizer=checked, formula=diagnostics)
            update_statistics(state, module, next_eval[group])
        if set(group_proofs) == set(GROUPS):
            break
    assert set(group_proofs) == set(GROUPS), dict(fold=fold, passed=list(group_proofs),
                                                 added={g:s['added'] for g,s in next_steps.items()})
    assert all(steps[g]['added']['direction_gain']['classification']>0 for g in GROUPS)
    return dict(fold=fold,status='PASS',elapsed=time.monotonic()-begun,init=initial,opts=identities,inputs=inputs,
                historical_best=dict(epoch=b0['best_epoch'],probability_sha=sha(b0['probability']),
                                     checkpoint_sha=b0['checkpoint_sha']),
                identity=ctx.identity, reference={**raw, **ref_first}, groups=group_proofs,
                common_added_initial_sha=sha({k:initial_modules['G1'][k] for k in sorted(common)}), historical_class_weight_mass=class_mass,
                initial_rng={key:value for key,value in ctx.archive_rng[START+1].items()})


def fold_paths(fold):
    name = f'f{fold:02d}'
    return (RT/'tasks'/f'{name}.json', RT/'checkpoints'/f'{name}.latest.pt', RT/'logs'/f'{name}.jsonl')


def completed_fold(fold):
    task, latest, log = fold_paths(fold)
    if not task.exists():
        return False
    record = h.read_json(task)
    assert record['implementation'] == implementation() and record['status'] == 'COMPLETE'
    assert sha_file(latest) == record['latest_sha'] and sha_file(log) == record['log_sha']
    assert all(sha_file(h.paths(group, fold)[1]) == record['groups'][group]['best_sha'] for group in GROUPS)
    return True


def point(state, evaluation):
    p = evaluation['probability']
    return dict(probability=p, off_probability=evaluation.get('off_probability'), gap=evaluation['gap'],
                off_gap=evaluation.get('off_gap'), metrics=h.metrics_for(p, state['labels_np'][state['test_np']]))


def capture_best(ctx, group, epoch, ev, ref_eval, step):
    state, module, optimizer = ctx.students[group]
    return dict(group=group, fold=ctx.fold, epoch=epoch, initial=ctx.meta[group], identity=ctx.identity,
                student=save_student(state, module, optimizer), reference=native_bundle(ctx.reference),
                rng={key:ctx.rng[key] for key in (group,'reference')},
                evaluation=point(state, ev), reference_evaluation=point(ctx.reference, ref_eval), step=step,
                ema_sha=sha(module.ema_state()))


def checkpoint_witness(ctx, group):
    state, module, optimizer = ctx.students[group]
    saved, ref_saved = save_student(state, module, optimizer), native_bundle(ctx.reference)
    old_rng = dict(ctx.rng)
    ema_before = sha(module.ema_state())
    try:
        steps, evaluated, ref_step, ref_eval = execute_epoch(ctx, (group,), inspect=True)
        observation_sha = update_statistics(state, module, evaluated[group])
        return dict(step=steps[group], reference_step=ref_step, student=sha(save_student(state, module, optimizer)),
                    reference=sha(native_bundle(ctx.reference)), evaluation=sha(evaluated[group]),
                    reference_evaluation=sha(ref_eval), observation=observation_sha,
                    ema_before=ema_before, ema_after=sha(module.ema_state()),
                    rng={key:h.base_runtime().rng_sha(ctx.rng[key]) for key in (group, 'reference')})
    finally:
        restore_student(state, module, optimizer, saved)
        h.runtime().load_history().load_bundle(ctx.reference, ref_saved)
        ctx.rng = old_rng


def assert_preflight(fold=None):
    manifest = h.read_json(HERE/'manifest.json')
    checked = h.read_json(RT/'preflight.json')
    assert checked['status'] == 'PASS' and manifest['implementation'] == checked['implementation'] == implementation()
    assert set(x['fold'] for x in checked['folds']) == set(FOLDS)
    if fold is not None:
        return next(x for x in checked['folds'] if x['fold'] == fold)
    return checked


def save_transaction(ctx, log, digest, latest, epoch, track, marks, aligned, terminal_refs, witnesses, pending, elapsed):
    log.flush()
    os.fsync(log.fileno())
    for group, snapshot in pending.items():
        path = h.best_pending(h.paths(group, ctx.fold)[1])
        h.atomic_torch(path, snapshot)
        track[group]['best_sha'] = sha_file(path)
    content = dict(format='CER shared reference', identity=ctx.identity, fold=ctx.fold, implementation=implementation(),
                   next_epoch=epoch, reference=native_bundle(ctx.reference),
                   students={group:save_student(*args) for group,args in ctx.students.items()}, rng=ctx.rng,
                   track=track, archive_marks=marks, aligned_b0=aligned, terminal_refs=terminal_refs, witnesses=witnesses,
                   log_bytes=log.tell(), log_sha=digest.hexdigest(), elapsed=elapsed)
    h.atomic_torch(latest, content)
    for group in pending:
        path = h.paths(group, ctx.fold)[1]
        os.replace(h.best_pending(path), path)
    directory = os.open(latest.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    pending.clear()


def train_fold(fold):
    preflight = assert_preflight(fold)
    if completed_fold(fold):
        return
    setup_started = time.monotonic()
    ctx = setup_fold(fold)
    assert native_digests(ctx.reference) == preflight['init']
    assert all(ctx.meta[group] == preflight['groups'][group]['meta'] for group in GROUPS)
    task, latest, journal = fold_paths(fold)
    track = {group:dict(best_acc=-1., best_epoch=0, wait=0, terminal_epoch=None, best_sha=None) for group in GROUPS}
    marks, aligned, terminal_refs, witnesses, pending = {}, {}, {}, {}, {}
    executed, before_seconds, resumed = START, 0., False
    if latest.exists():
        saved = h.torch_load(latest)
        assert saved['identity'] == ctx.identity and saved['implementation'] == implementation()
        h.runtime().load_history().load_bundle(ctx.reference, saved['reference'])
        for group, (state, module, optimizer) in ctx.students.items():
            restore_student(state, module, optimizer, saved['students'][group])
            h.reconcile_best(h.paths(group, fold)[1], saved['track'][group]['best_sha'])
        (track, marks, aligned, terminal_refs, witnesses) = (
            saved[key] for key in ('track','archive_marks','aligned_b0','terminal_refs','witnesses'))
        ctx.rng = saved['rng']
        executed, before_seconds, resumed = saved['next_epoch'], saved['elapsed'], True
        print(f'F{fold:02d} restoring epoch {executed}', flush=True)
    if not resumed:
        for group in GROUPS:
            h.reconcile_best(h.paths(group, fold)[1], None)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with journal.open('r+b' if resumed else 'w+b') as log:
        digest = h.committed_log(log, saved) if resumed else hashlib.sha256()
        for epoch in range(executed+1, LIMIT+1):
            active = tuple(group for group in GROUPS if track[group]['terminal_epoch'] is None)
            if not active and epoch > ctx.archive['epochs_executed']:
                break
            steps, evaluation, ref_step, ref_eval = execute_epoch(ctx, active, inspect=epoch in (START+1, START+2))
            marks.update(archive_check(ctx, epoch, ref_step, ref_eval))
            truth = ctx.reference['labels_np'][ctx.reference['test_np']]
            newly_ended = []
            row = dict(epoch=epoch, reference=ref_step, reference_archived=epoch in ctx.archive_rng,
                       reference_correct=int((ref_eval['probability'].argmax(1) == truth).sum()), students={})
            for group in active:
                state, module, optimizer = ctx.students[group]
                ev, step, status = evaluation[group], steps[group], track[group]
                scores = h.metrics_for(ev['probability'], truth)
                off_correct = int((ev['off_probability'].argmax(1) == truth).sum())
                before_ema = sha(module.ema_state())
                improved = scores['acc'] > status['best_acc']
                if improved:
                    status.update(best_acc=scores['acc'], best_epoch=epoch, wait=0)
                    pending[group] = capture_best(ctx, group, epoch, ev, ref_eval, step)
                else:
                    status['wait'] += 1
                if epoch == ctx.archive['best_epoch_1based']:
                    aligned[group] = dict(epoch=epoch, evaluation=point(state, ev), reference=point(ctx.reference, ref_eval))
                observation = update_statistics(state, module, ev)
                row['students'][group] = dict(step, correct=int((ev['probability'].argmax(1)==truth).sum()),
                                              off_correct=off_correct, improved=improved, wait=status['wait'],
                                              ema_before=before_ema, ema_observation=observation,
                                              ema_after=sha(module.ema_state()))
                if epoch == LIMIT or status['wait'] > CFG['early_stop_wait_gt']:
                    status['terminal_epoch'] = epoch
                    newly_ended.append(group)
            if newly_ended:
                terminal_refs[epoch] = dict(state=native_bundle(ctx.reference), rng=ctx.rng['reference'])
                for group in newly_ended:
                    witnesses[group] = checkpoint_witness(ctx, group)
            line = (json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n').encode()
            log.write(line)
            digest.update(line)
            executed = epoch
            ending = all(track[group]['terminal_epoch'] is not None for group in GROUPS) and epoch >= ctx.archive['epochs_executed']
            if epoch % CFG['save_latest_every_epochs'] == 0 or newly_ended or ending:
                save_transaction(ctx, log, digest, latest, epoch, track, marks, aligned, terminal_refs, witnesses,
                                 pending, before_seconds+time.monotonic()-start)
                print(f'F{fold:02d} ep={epoch} ' + ' '.join(f'{g}={track[g]["best_acc"]:.5f}@{track[g]["best_epoch"]}' for g in GROUPS),
                      flush=True)
            if fold == 1 and epoch == CFG['gpu']['pilot_first_epoch'] and not (RT/'pilot.json').exists():
                gpu = gpu_sample()
                rss_kib = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                   if line.startswith('VmHWM:')))
                own_device = gpu_processes().get(os.getpid(), {})
                own_mib = own_device.get('used', '')
                assert not own_device or own_device['uuid']==CFG['gpu']['uuid']
                h.atomic_json(RT/'pilot.json', dict(gpu=gpu, pid=os.getpid(), epoch=epoch,
                                                    process_peak_rss_mib=rss_kib/1024,
                                                    sampled_process_vram_mib=int(own_mib) if own_mib.isdecimal() else None,
                                                    elapsed=time.monotonic()-start, setup_seconds=start-setup_started,
                                                    process_peak_reserved_mib=float(torch.cuda.max_memory_reserved()/2**20),
                                                    process_peak_allocated_mib=float(torch.cuda.max_memory_allocated()/2**20),
                                                    threads=torch.get_num_threads(), nproc=len(os.sched_getaffinity(0))))
            if ending:
                break
    assert set(marks) == {'archive_best','archive_terminal'} and not pending
    assert set(witnesses) == set(GROUPS)
    assert all(track[group]['terminal_epoch'] is not None for group in GROUPS)
    h.atomic_json(task, dict(status='COMPLETE', fold=fold, implementation=implementation(), identity=ctx.identity,
                             groups=track, latest_sha=sha_file(latest), log_sha=sha_file(journal),
                             reference_epochs=executed, archive_marks=marks, resumed=resumed,
                             elapsed=before_seconds+time.monotonic()-start, setup_seconds=start-setup_started,
                             peak_gpu_mib=torch.cuda.max_memory_reserved()/2**20))
    print(f'F{fold:02d} complete, shared native reference through epoch {executed}', flush=True)


def launch(mode, fold, extra=()):
    log_path = RT/'logs'/f'{mode.removeprefix("--")}_f{fold:02d}.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    allowed = os.sched_getaffinity(0)
    local_cores = sorted(allowed.intersection(set(range(64))))
    locality = ['taskset','-c',str(local_cores[(fold-1)%len(local_cores)])] if len(local_cores) >= len(FOLDS) else []
    with log_path.open('ab', buffering=0) as stream:
        process = subprocess.Popen([*locality, sys.executable, '-B', str(HERE/'run.py'), mode, str(fold), *extra],
                                   cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
    return process, log_path


def tail(path, limit=3500):
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit))
        return f.read().decode(errors='replace')


def cpu_capacity():
    count = len(os.sched_getaffinity(0))
    unified = Path('/sys/fs/cgroup/cpu.max')
    legacy = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    if unified.is_file():
        quota, period = unified.read_text().split()
        return count if quota == 'max' else min(count,max(1,int(quota)//int(period)))
    if legacy.is_file():
        quota, period = int(legacy.read_text()), int(legacy.with_name('cpu.cfs_period_us').read_text())
        return count if quota < 0 else min(count,max(1,quota//period))
    return count


def process_pool(mode, folds, width):
    pending = list(folds)
    running = {}
    try:
        while pending or running:
            while pending and len(running) < width:
                fold = pending.pop(0)
                running[fold] = launch(mode,fold)
            for fold,(process,log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    del running[fold]
                    if code:
                        raise RuntimeError(f'{mode} fold {fold} exit={code}\n{tail(log)}')
                    print(f'{mode} F{fold:02d} PASS',flush=True)
            if running:
                time.sleep(1)
    finally:
        for proc, _ in running.values():
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in running.values():
            if proc.poll() is None:
                proc.wait(timeout=20)


def certify_preflight():
    started = time.monotonic()
    env = environment()
    initial = protection()
    assert initial['pass_'], initial
    freeze_initial_preflight()
    h.update_manifest(status='PREFLIGHT_RUNNING', run_environment=env, pre_run_protection=initial,
                      audit_source_at_preflight={name:sha_file(HERE/name) for name in ('audit.py','report.py')})
    prompt_cache = h.torch_load(h.PROMPT_CACHE)
    text = h.verify_text(prompt_cache)
    query = anonymous_queries(prompt_cache['q_medical'], CFG['seeds']['anonymous_tokens'])
    text.update(nonsemantic_query_sha=sha(query), generator_seed=CFG['seeds']['anonymous_tokens'],
                max_row_norm_error=float((query.norm(dim=1)-prompt_cache['q_medical'].norm(dim=1)).abs().max()))
    process_pool('--preflight-fold', FOLDS, min(4, cpu_capacity()))
    rows = [h.read_json(RT/'tasks'/f'preflight_f{fold:02d}.json') for fold in FOLDS]
    assert all(value['status']=='PASS' for value in rows)
    checked = dict(status='PASS', implementation=implementation(), folds=rows, history_input_protection=initial,
                   verified_prompt=text, elapsed=time.monotonic()-started, env=env)
    h.atomic_json(RT/'preflight.json', checked)
    h.update_manifest(status='PREFLIGHT_PASS', preflight_sha=sha_file(RT/'preflight.json'))
    return checked


def gpu_processes():
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'],text=True)
    result = {}
    for line in raw.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) == 3 and fields[0].isdigit():
            result[int(fields[0])] = dict(uuid=fields[1],used=fields[2])
    return result


def available_host_mib():
    candidate = [int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                          if line.startswith('MemAvailable:'))) // 1024]
    for limit, used in (('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        if Path(limit).is_file() and Path(used).is_file():
            cap = Path(limit).read_text().strip()
            if cap.isdecimal():
                candidate.append(max(0, int(cap)-int(Path(used).read_text().strip())) // 2**20)
    return min(candidate)


def train_all():
    assert_preflight()
    started = time.monotonic()
    done = {fold for fold in FOLDS if completed_fold(fold)}
    pending = [fold for fold in FOLDS if fold not in done]
    running, attempts = {}, {}
    width = 1
    agreed = None
    update_time = 0.
    measured_peak = 0
    use_pid_identity = {}
    h.update_manifest(status='TRAINING')
    try:
        while pending or running:
            if agreed is None and (RT/'pilot.json').is_file():
                pilot = h.read_json(RT/'pilot.json')
                gpu = pilot['gpu']
                device_estimate = max(pilot['process_peak_reserved_mib'],pilot['sampled_process_vram_mib'] or 0)
                budget = gpu['total_mib']-gpu['used_mib']-CFG['gpu']['gpu_free_reserve_mib']
                capacity = 1 + max(0,budget) // max(1,math.ceil(1.4*device_estimate))
                host_slots = 1 + max(0,available_host_mib()-4096) // max(1,math.ceil(1.4*pilot['process_peak_rss_mib']))
                agreed = min(CFG['gpu']['workers_max'], CFG['gpu']['target_worker_limit'],cpu_capacity(),capacity,host_slots)
                width = agreed
                h.update_manifest(pilot=pilot, concurrency=dict(workers=width, gpu_slots=capacity, host_slots=host_slots, cpu_limit=cpu_capacity()))
                print(f'PILOT passed {gpu["uuid"]} peak_reserved={pilot["process_peak_reserved_mib"]:.0f}MiB '
                      f'{pilot["elapsed"]/(pilot["epoch"]-START):.3f}s per joint epoch, workers={width}',flush=True)
            while pending and len(running)<width:
                fold = pending.pop(0)
                running[fold] = launch('--worker',fold)
                attempts[fold] = attempts.get(fold,0)+1
            procs = gpu_processes()
            for fold,(process,log) in list(running.items()):
                if process.pid in procs:
                    assert procs[process.pid]['uuid'] == CFG['gpu']['uuid']
                    use_pid_identity[str(fold)] = procs[process.pid]['uuid']
                code = process.poll()
                if code is None:
                    continue
                del running[fold]
                if code:
                    content = tail(log)
                    can_resume = fold_paths(fold)[1].exists()
                    if can_resume and attempts[fold]<2 and ('out of memory' in content.lower() or code == -9):
                        width = max(1,width-1)
                        pending.insert(0,fold)
                        print(f'F{fold:02d} retry from committed Latest, workers={width}',flush=True)
                        continue
                    raise RuntimeError(f'train fold {fold} exit={code}\n{content}')
                assert completed_fold(fold)
                done.add(fold)
                print(f'TRAINED {len(done)*2}/20 F{fold:02d}',flush=True)
            now = time.monotonic()
            gpu = gpu_sample()
            measured_peak = max(measured_peak,gpu['used_mib'])
            if now-update_time >= 25:
                progress = {fold:tail(log,600).strip().splitlines()[-1] if log.stat().st_size else 'setup'
                            for fold,(_,log) in running.items()}
                print(f'{len(done)*2}/20 GPU0 {gpu["used_mib"]}MiB {gpu["utilization_pct"]}% {progress}',flush=True)
                update_time = now
            if running:
                time.sleep(2)
    finally:
        for process,_ in running.values():
            if process.poll() is None:
                process.terminate()
        for process,_ in running.values():
            if process.poll() is None:
                process.wait(timeout=20)
    assert len(done) == len(FOLDS)
    summary = dict(elapsed=time.monotonic()-started, workers=agreed, peak_sampled_gpu_mib=measured_peak,
                   launched_attempts=attempts, physical_uuid_by_fold=use_pid_identity,
                   completed_tasks=len(done)*2)
    h.update_manifest(status='TRAINED_PENDING_AUDIT', training=summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument('--preflight', action='store_true')
    commands.add_argument('--preflight-fold', type=int)
    commands.add_argument('--worker', type=int)
    commands.add_argument('--train', action='store_true')
    commands.add_argument('--audit-fold', type=int)
    commands.add_argument('--audit', action='store_true')
    commands.add_argument('--pipeline', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        certify_preflight()
    elif args.preflight_fold:
        assert args.preflight_fold in FOLDS
        value = preflight_fold(args.preflight_fold)
        h.atomic_json(RT/'tasks'/f'preflight_f{args.preflight_fold:02d}.json',value)
    elif args.worker:
        assert args.worker in FOLDS
        lock = fold_paths(args.worker)[0].with_suffix('.lock')
        with lock.open('a') as file:
            fcntl.flock(file,fcntl.LOCK_EX | fcntl.LOCK_NB)
            train_fold(args.worker)
    elif args.train:
        train_all()
    elif args.audit_fold:
        import audit
        val = audit.audit_fold(args.audit_fold)
        val.update(manifest_sources=implementation(),task_sha=sha_file(fold_paths(args.audit_fold)[0]))
        h.atomic_json(RT/'audit'/f'f{args.audit_fold:02d}.json',val)
    elif args.audit:
        import audit
        audit.audit_all()
    else:
        certify_preflight()
        train_all()
        subprocess.run([sys.executable, '-B', str(HERE/'run.py'), '--audit'], cwd=HERE, check=True)


if __name__ == '__main__':
    main()
