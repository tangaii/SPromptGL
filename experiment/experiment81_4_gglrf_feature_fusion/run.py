#!/usr/bin/env python3
"""Experiment 81.4 GGLRF feature-fusion runner."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import torch

from gglrf import GGLRF, residual_penalty, GROUP_UNARY, GROUP_PAIRWISE

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
GROUPS = tuple(CFG["groups"])
BASE_GROUP = "E74.4-G2-Positive-DCR"
GROUP_MODE = {GROUPS[0]: GROUP_UNARY, GROUPS[1]: GROUP_PAIRWISE}
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
BASE = None
PATH_ADAPTER = None


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_value(value):
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T" + str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"N" + str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

    visit(value)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    body = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temp.write_text(body, encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if path.read_text(encoding="utf-8") != body:
        raise IOError(f"JSON readback failed: {path}")


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"checkpoint write failed: {path}")


def torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def log_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    global BASE, PATH_ADAPTER
    if BASE is None:
        adapter_path = HERE / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("experiment81_4_path_adapter", adapter_path)
        PATH_ADAPTER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = PATH_ADAPTER
        spec.loader.exec_module(PATH_ADAPTER)
        BASE = PATH_ADAPTER.load_base_runner()
    return BASE


def base_runtime():
    base = runtime()
    return base.load_history().load_e69().load_e61().load_base()


def source_map():
    base = runtime()
    paths = [HERE / name for name in (
        "config.json", "design.md", "path_adapter.py", "gglrf.py", "run.py", "audit.py"
    )]
    paths.extend(Path(path) for path in base.source_manifest())
    e812 = Path("/model/spromptgl/experiment/experiment81_3_e2e_pairwise_relational_learning")
    for name in ("relation_learning.py", "run.py", "config.json", "design.md", "path_adapter.py", "audit.py",
                 "results/preflight.json", "results/report.md", "results/fold_metrics.csv",
                 "results/mechanism.csv", "results/audit_status.json"):
        paths.append(e812 / name)
    e811 = Path("/model/spromptgl/experiment/experiment81_1_prcl_v2")
    paths.extend(e811 / name for name in ("prcl_v2.py", "run.py", "config.json", "results/fold_metrics.csv", "results/report.md"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("frozen source provenance missing: " + ", ".join(missing[:12]))
    return {str(path.resolve()): sha_file(path) for path in sorted({Path(p).resolve() for p in paths})}


def source_sha():
    return sha_value(source_map())


def prepare(fold):
    base = runtime()
    state, warmup, identity = base.prepare(fold)
    state["fold"] = fold
    state["rudag_identity"] = identity
    expected_overlap = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected_overlap:
        raise RuntimeError(f"fold {fold}: unexpected train/test overlap")
    if tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError(f"fold {fold}: expected 4096 inherited relation pairs")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError(f"fold {fold}: inherited pair outside training rows")
    if not hasattr(state["cam"], "Outputlayer"):
        raise RuntimeError("inherited CAM has no original Outputlayer")
    return state, warmup, identity


def paths(group, fold):
    token = f"{group}_f{fold:02d}"
    return (TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt",
            CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl")


def grad_norm(parameters):
    pieces = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0


def _finite_module_grads(module):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def bundle(state, correction, optimizer):
    return {
        "base": runtime().load_history().state_bundle(state),
        "relation": _cpu_state(correction),
        "optimizer_relation": copy.deepcopy(optimizer.state_dict()),
    }


def load_bundle(state, correction, optimizer, saved):
    runtime().load_history().load_bundle(state, saved["base"])
    correction.load_state_dict(saved["relation"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_relation"]))


def _metric_copy(metrics):
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _validate_optimizer_settings(state, optimizer):
    expected = (("opt_base", 0.11, 0.0038), ("opt_cam", 0.11, 0.0038), ("opt_added", 0.01, 0.001))
    for key, wd, lr in expected:
        for group in state[key].param_groups:
            if abs(float(group["weight_decay"]) - wd) > 1e-12 or abs(float(group["lr"]) - lr) > 1e-12:
                raise RuntimeError(f"historical optimizer mismatch: {key}")
    for group in optimizer.param_groups:
        if abs(float(group["weight_decay"]) - 0.01) > 1e-12 or abs(float(group["lr"]) - 0.0005) > 1e-12:
            raise RuntimeError("relation optimizer settings mismatch")


def _base_forward(state, training):
    return runtime().forward(state, BASE_GROUP, training=training)


def _tree_close(left, right, atol=1e-7, rtol=1e-7, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"parity mismatch at {where}: tensor metadata")
        if left.is_floating_point() or left.is_complex():
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                raise RuntimeError(f"parity mismatch at {where}: max_abs={float((left-right).abs().max())}")
        elif not torch.equal(left, right):
            raise RuntimeError(f"parity mismatch at {where}: integer tensor")
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"parity key mismatch at {where}")
        for key in left:
            _tree_close(left[key], right[key], atol, rtol, f"{where}.{key}")
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"parity length mismatch at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            _tree_close(a, b, atol, rtol, f"{where}[{index}]")
    elif left != right:
        raise RuntimeError(f"parity mismatch at {where}: {left!r} != {right!r}")


def _gradient_copy(state):
    return {
        key: {name: None if p.grad is None else p.grad.detach().cpu().clone()
              for name, p in state[key].named_parameters()}
        for key in ("model", "cam")
    }


def _compare_gradients(left, right):
    _tree_close(left, right, atol=0.0, rtol=0.0, where="gradients")



def _rng_runtime():
    return base_runtime()


def make_module(fold, device, eta=0.1):
    return GGLRF.initialized(CFG['architecture']['private_cpu_salt_base'] + fold, device, eta=eta,
                             rho=CFG['architecture']['rho'])


def grad_snapshot(state, module):
    return {'darg': grad_norm(state['model'].parameters()), 'cam': grad_norm(state['cam'].parameters()),
            'gglrf': grad_norm(module.parameters()),
            'global_encoder': grad_norm(list(module.global_proj.parameters()) + list(module.global_norm.parameters())),
            'local_encoder': grad_norm(list(module.local_proj.parameters()) + list(module.local_norm.parameters()) + [module.position]),
            'query': grad_norm(list(module.query_proj.parameters()) + list(module.query_norm.parameters())),
            'key': grad_norm(module.key_proj.parameters()), 'value': grad_norm(module.value_proj.parameters()),
            'global_readout': grad_norm(list(module.read_key.parameters()) + list(module.read_query.parameters())),
            'output_projection': grad_norm(module.output.parameters())}


def forward_fusion(state, module, group, training=False, intervention='full', eta=None):
    module.train(training)
    captured, calls = {}, {'count': 0}
    blocked = (376, 406) if training and state['fold'] == 2 else ()

    def replace_input(layer, args):
        calls['count'] += 1
        if calls['count'] != 1:
            raise RuntimeError('original OutputLayer executed more than once')
        z = args[0]
        enhanced, details = module(z, GROUP_MODE[group], intervention=intervention,
                                   blocked_rows=blocked, eta=eta)
        captured.update(details=details, z_cam=z, z_enhanced=enhanced)
        # Scalar gradient telemetry only; no extra backward or RNG operation.
        if training and torch.is_grad_enabled():
            route = captured.setdefault('route_gradients', {})
            details['z_rel'].register_hook(lambda v: route.__setitem__('z_rel_input', float(v.detach().norm().cpu())))
            z.register_hook(lambda v: route.__setitem__('cam144_total', float(v.detach().norm().cpu())))
            details['delta'].register_hook(lambda v: route.__setitem__('delta144', float(v.detach().norm().cpu())))
        return (enhanced,) + args[1:]

    handle = state['cam'].Outputlayer.register_forward_pre_hook(replace_input)
    try:
        out = _base_forward(state, training=training)
    finally:
        handle.remove()
    if calls['count'] != 1 or not captured:
        raise RuntimeError('expected exactly one original OutputLayer call')
    out['fusion'] = captured['details']
    out['route_gradients'] = captured.get('route_gradients', {})
    out['outputlayer_calls'] = calls['count']
    out['historical_total'] = out['total']
    out['res'] = residual_penalty(out['fusion'], state['train'])
    out['total'] = out['historical_total'] + CFG['loss']['lambda_res'] * out['res']
    return out


def _stats(out):
    return {key: float(out[key].detach().cpu()) for key in
            ('cls', 'graph', 'pair', 'adv', 'preserve', 'historical_total', 'res', 'total') if key in out}


def _step(state, module, optimizer, group, fusion=True):
    hist = runtime().load_history()
    hist.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    capture = {}
    if fusion:
        out = forward_fusion(state, module, group, training=True)
    else:
        handle = state['cam'].Outputlayer.register_forward_pre_hook(
            lambda layer, args: capture.update(z_cam=args[0].detach().clone()))
        try:
            out = _base_forward(state, training=True)
        finally:
            handle.remove()
    out['total'].backward()
    grads = _gradient_copy(state)
    norms = grad_snapshot(state, module)
    for key in ('opt_base', 'opt_cam', 'opt_added'):
        state[key].step()
    if fusion:
        optimizer.step()
    return out, grads, norms, copy.deepcopy(hist.state_bundle(state)), _rng_runtime().pack_rng(), capture


def _replay_step(state, module, optimizer, group):
    out, _, norms, _, _, _ = _step(state, module, optimizer, group)
    return {'losses': _stats(out), 'logp': out['logp'].detach().cpu().clone(),
            'delta': out['fusion']['delta'].detach().cpu().clone(),
            'beta': out['fusion']['beta'].detach().cpu().clone(),
            'attention': out['fusion']['attention'].detach().cpu().clone(),
            'state': bundle(state, module, optimizer), 'rng': _rng_runtime().pack_rng(), 'norms': norms}


def _assert_exact(a, b, where):
    _tree_close(a, b, atol=0.0, rtol=0.0, where=where)


def module_preflight():
    module = make_module(1, 'cpu')
    twin = make_module(1, 'cpu')
    _assert_exact(_cpu_state(module), _cpu_state(twin), 'private_initialization')
    shapes = {name: list(p.shape) for name, p in module.named_parameters()}
    z = torch.linspace(-2.0, 2.0, 5*144).reshape(5, 144).requires_grad_()
    outputs = {}
    expected = {'h': (5,16,9), 'u': (5,16,32), 'g': (5,32), 'key': (5,16,15,32),
                'value': (5,16,15,32), 'attention': (5,8,16,16), 'messages': (5,8,16,32),
                'm_local': (5,16,32), 'beta': (5,16), 'raw_delta_local': (5,16,9),
                'delta': (5,144), 'z_enhanced': (5,144)}
    for mode in (GROUP_UNARY, GROUP_PAIRWISE):
        enhanced, details = module(z, mode)
        for key, shape in expected.items():
            assert tuple(details[key].shape) == shape, (key, details[key].shape)
        assert torch.equal(enhanced, z) and torch.count_nonzero(details['delta']) == 0
        assert all(bool(torch.isfinite(v).all()) for v in details.values() if torch.is_tensor(v))
        assert torch.allclose(details['attention'].sum(-1), torch.ones(5,8,16), atol=2e-7)
        outputs[mode] = details
    assert not torch.equal(outputs[GROUP_UNARY]['key'], outputs[GROUP_PAIRWISE]['key'])
    neighbors = module.neighbors
    assert neighbors.shape == (16,15)
    for p in range(16):
        assert set(neighbors[p].tolist()) == set(range(16)) - {p}
    receiver = torch.linspace(-1., 1., 2*16*15*32).reshape(2,16,15,32)
    sender = receiver.flip(-1)
    k1, v1 = module.construct_features(receiver, sender, GROUP_UNARY)
    k2, v2 = module.construct_features(receiver + 0.3, sender, GROUP_UNARY)
    assert torch.equal(k1,k2) and torch.equal(v1,v2)
    kp, vp = module.construct_features(receiver, sender, GROUP_PAIRWISE)
    kp2, vp2 = module.construct_features(receiver+0.3, sender, GROUP_PAIRWISE)
    assert not torch.equal(kp,kp2) and not torch.equal(vp,vp2)
    # After a nonzero output projection, structural tests cannot pass trivially.
    with torch.no_grad():
        module.output.weight.copy_(torch.linspace(-.07,.09,288).reshape(9,32))
    _, u, g = module.encode(z)
    changed_g = g.flip(-1) + torch.linspace(-.7,.4,32)
    a = module.attend(u,g,GROUP_PAIRWISE)
    b = module.attend(u,changed_g,GROUP_PAIRWISE)
    attention_change = float((a['attention']-b['attention']).abs().mean().detach())
    assert attention_change > 1e-7
    ra = module.readout(u,a['m_local'],g)
    rb = module.readout(u,a['m_local'],changed_g)
    beta_change = float((ra['beta']-rb['beta']).abs().mean().detach())
    assert beta_change > 1e-7
    for mode in (GROUP_UNARY,GROUP_PAIRWISE):
        enhanced,d = module(z,mode)
        assert bool((d['delta_ratio'] <= .100001).all())
        for intervention in ('off','null'):
            zi,di = module(z,mode,intervention=intervention)
            assert torch.equal(zi,z) and torch.count_nonzero(di['delta']) == 0
            if intervention == 'null':
                assert torch.count_nonzero(di['messages']) == 0
        # Changing patient 0 must not alter any other patient's feature path.
        altered = z.detach().clone(); altered[0] += 3.0
        ze,de = module(altered,mode)
        assert torch.equal(enhanced[1:],ze[1:])
    # Input gradient contains exactly the requested eta-scaled new path, while
    # the original identity path remains unscaled. Detached norm cannot bypass it.
    route = {}
    for eta in (0.0,0.1,1.0):
        zz=z.detach().clone().requires_grad_()
        ze,d=module(zz,GROUP_PAIRWISE,eta=eta)
        probe=torch.linspace(-1.0,1.0,zz.numel()).reshape_as(zz)
        new_grad=torch.autograd.grad((d['delta']*probe).sum(),zz,retain_graph=True)[0]
        total_grad=torch.autograd.grad((ze*probe).sum(),zz)[0]
        assert torch.allclose(total_grad,probe+new_grad,atol=1e-7,rtol=1e-7)
        route[str(eta)]=new_grad
    assert torch.count_nonzero(route['0.0']) == 0
    assert route['0.1'].norm() > 0
    error=float((route['0.1']-.1*route['1.0']).abs().max())
    assert error < 1e-6
    # Tiny/zero features and large residuals remain finite and bounded.
    for scale in (0.0,1e-20,1e-6,1.0,1e6):
        tiny=(z.detach()*scale).requires_grad_()
        ze,d=module(tiny,GROUP_PAIRWISE)
        assert bool(torch.isfinite(ze).all())
        assert bool((d['delta'].norm(dim=-1) <= .1*tiny.detach().norm(dim=-1)+1e-6).all())
        ze.sum().backward()
        assert bool(torch.isfinite(tiny.grad).all())
    # Smoothness at the nominal former hard-clipping boundary.
    raw=torch.tensor([.9999,1.0,1.0001],dtype=torch.float64,requires_grad=True)
    smooth=raw/torch.sqrt(1+raw.square()+1e-12)
    derivative=torch.autograd.grad(smooth.sum(),raw)[0]
    assert bool((derivative>0).all()) and float(derivative.max()-derivative.min()) < .001
    return {'status':'PASS','parameter_count':sum(p.numel() for p in module.parameters()),
            'parameter_shapes':shapes,'expected_shapes_N5':{k:list(v) for k,v in expected.items()},
            '15_unique_real_neighbors_no_self':True,'g1_sender_only_g2_pairwise_verified':True,
            'attention_mae_changed_global_fixed_local':attention_change,
            'beta_mae_changed_global_fixed_messages':beta_change,'no_cross_patient_mixing':True,
            'null_off_strict_identity':True,'eta_scaled_gradient_max_abs_error':error,
            'identity_gradient_unscaled':True,'detached_bound_no_bypass':True,
            'smooth_bound_tiny_input_finite':True}


def _route_on_real_state(state, initial, initial_rng):
    hist,rng=runtime().load_history(),_rng_runtime()
    runs={}
    for eta in (0.0,0.1,1.0):
        hist.load_bundle(state,initial); rng.restore_rng(initial_rng)
        module=make_module(state['fold'],state['x'].device,eta=eta)
        with torch.no_grad():
            module.output.weight.copy_(torch.linspace(-.07,.09,288,device=state['x'].device).reshape(9,32))
        hist.zero_optimizers(state)
        out=forward_fusion(state,module,GROUPS[1],training=True,eta=eta)
        z,d=out['fusion']['z_cam'],out['fusion']['delta']
        # Use this same classification forward's incoming derivative at delta.
        incoming=torch.autograd.grad(out['total'],d,retain_graph=True)[0]
        new=torch.autograd.grad(d,z,grad_outputs=incoming,retain_graph=True)[0]
        total=torch.autograd.grad(out['total'],z,retain_graph=True)[0]
        assert torch.allclose(total,incoming+new,rtol=2e-6,atol=2e-8)
        out['total'].backward()
        norms=grad_snapshot(state,module)
        assert norms['global_encoder'] > 0 and norms['local_encoder'] > 0 and norms['global_readout'] > 0
        assert norms['darg'] > 0 and norms['cam'] > 0
        runs[str(eta)]={'grad':new.detach().cpu(),'norms':norms,'loss':float(out['total'].detach())}
        del out,module
    assert torch.count_nonzero(runs['0.0']['grad']) == 0
    err=float((runs['0.1']['grad']-.1*runs['1.0']['grad']).abs().max())
    scale=float(runs['1.0']['grad'].abs().max())
    assert runs['0.1']['grad'].norm()>0 and err/max(scale,1e-12)<2e-4
    return {'eta0_new_cam144_gradient_zero':True,'eta01_new_cam144_gradient_nonzero':True,
            'eta01_to_eta1_relative_max_error':err/max(scale,1e-12),
            'identity_path_unscaled':True,'global_local_readout_gradient_nonzero':True,
            'gradient_norms':{k:v['norms'] for k,v in runs.items()}}


def preflight():
    for path in (RESULTS,RUNTIME,CHECKPOINTS,TASKS,LOGS): path.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('exactly one GPU required')
    hist,rng=runtime().load_history(),_rng_runtime()
    free,total=torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    structure=module_preflight()
    folds=[]
    for fold in range(1,11):
        state,warmup,identity=prepare(fold)
        assert rng.rng_sha(rng.pack_rng()) == warmup['rng_sha']
        initial=hist.state_bundle(state); initial_rng=rng.pack_rng()
        public_before=rng.rng_sha(rng.pack_rng())
        module=make_module(fold,state['x'].device); twin=make_module(fold,state['x'].device)
        assert rng.rng_sha(rng.pack_rng()) == public_before
        _assert_exact(_cpu_state(module),_cpu_state(twin),f'fold{fold}.group_init')
        optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
        _validate_optimizer_settings(state,optimizer)
        layer=state['cam'].Outputlayer
        assert layer.mlp_head[1].in_features==144 and layer.mlp_head[1].out_features==36
        assert layer.mlp_head[3].p==.35 and layer.classifier.out_features==2
        # Train-mode parity is stricter than eval-only parity: includes dropout,
        # every loss, backbone gradient and all three optimizer updates.
        base_out,base_grads,_,updated,after_rng,capture=_step(state,module,optimizer,GROUPS[0],fusion=False)
        hist.load_bundle(state,initial); rng.restore_rng(initial_rng)
        module=make_module(fold,state['x'].device)
        optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
        hooked,hook_grads,norms,updated_hook,hook_rng,_=_step(state,module,optimizer,GROUPS[0])
        for key in ('logp','hidden','cls','graph','pair','adv','preserve','total'):
            _assert_exact(base_out[key].detach().cpu(),hooked[key].detach().cpu(),f'fold{fold}.{key}')
        _assert_exact(capture['z_cam'],hooked['fusion']['z_cam'].detach(),'CAM144')
        _assert_exact(hooked['fusion']['z_cam'],hooked['fusion']['z_enhanced'],'output_input_identity')
        _compare_gradients(base_grads,hook_grads)
        _assert_exact(updated,updated_hook,f'fold{fold}.optim_update')
        assert rng.rng_sha(after_rng)==rng.rng_sha(hook_rng)
        assert norms['output_projection']>0
        historical_sum=hooked['cls']+hooked['graph']+.05*hooked['pair']+.01*hooked['adv']+.01*hooked['preserve']
        _assert_exact(historical_sum,hooked['historical_total'],'historical_loss_formula')
        # A second supervised update must reach the internal relation/global path.
        learned=_replay_step(state,module,optimizer,GROUPS[0])
        for key in ('global_encoder','local_encoder','query','key','value','global_readout'):
            assert learned['norms'][key]>0, ('no second-step learning',key)
        if fold==2:
            assert torch.count_nonzero(learned['delta'][[376,406]])==0
        route=recovery=None
        if fold==1:
            route=_route_on_real_state(state,initial,initial_rng)
            hist.load_bundle(state,initial); rng.restore_rng(initial_rng)
            module=make_module(fold,state['x'].device)
            optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
            _replay_step(state,module,optimizer,GROUPS[1])
            first=bundle(state,module,optimizer); first_rng=rng.pack_rng()
            replay_path=RUNTIME/'preflight_recovery.tmp.pt'
            atomic_torch(replay_path,{'bundle':first,'rng':first_rng})
            uninterrupted=_replay_step(state,module,optimizer,GROUPS[1])
            saved=torch_load(replay_path)
            # Reconstruct every model and optimizer, then load serialized state.
            fresh,_,_=prepare(fold)
            fresh_module=make_module(fold,fresh['x'].device)
            fresh_opt=torch.optim.AdamW(fresh_module.parameters(),lr=.0005,weight_decay=.01)
            load_bundle(fresh,fresh_module,fresh_opt,saved['bundle']); rng.restore_rng(saved['rng'])
            resumed=_replay_step(fresh,fresh_module,fresh_opt,GROUPS[1])
            for key in ('losses','logp','delta','beta','attention','state'):
                _assert_exact(uninterrupted[key],resumed[key],f'recovery.{key}')
            assert rng.rng_sha(uninterrupted['rng'])==rng.rng_sha(resumed['rng'])
            replay_path.unlink()
            recovery='PASS_EXACT_SERIALIZED_FRESH_MODELS'
            del fresh,fresh_module,fresh_opt,uninterrupted,resumed,first,saved
        folds.append({'fold':fold,'identity':identity,'warmup_rng_sha':warmup['rng_sha'],
                      'private_init_public_rng_unchanged':True,'same_fold_group_init_exact':True,
                      'initial_module_sha256':sha_value(_cpu_state(twin)),
                      'base_off_forward_hidden_loss_gradient_update_rng_exact':True,
                      'outputlayer_calls':1,'outputlayer_hidden':36,'outputlayer_dropout':.35,
                      'output_projection_first_gradient_norm':norms['output_projection'],
                      'second_step_gradient_norms':learned['norms'],'route_test':route,'serialized_resume':recovery})
        print(f'PREFLIGHT fold={fold} PASS',flush=True)
        del state,module,twin,optimizer,base_out,hooked,learned,updated,updated_hook,initial
        torch.cuda.empty_cache()
    result={'status':'PASS','device':torch.cuda.get_device_name(0),'total_memory_bytes':int(total),
            'free_memory_bytes_at_start':int(free),'peak_gpu_bytes_observed':int(torch.cuda.max_memory_allocated()),
            'module_tests':structure,'folds':folds,'base_off_parity':{'passed':10,'expected':10},
            'source_map':source_map(),'source_sha256':source_sha(),
            'source_review':json.loads((RUNTIME/'source_review.json').read_text()),
            'formal_training_started':False}
    atomic_json(RESULTS/'preflight.json',result)
    return result

def _eval_summary(out, state):
    d=out['fusion']; ids=torch.as_tensor(state['test'],device=d['delta'].device,dtype=torch.long)
    beta=d['beta'][ids]; attn=d['attention'][ids]; ratio=d['delta_ratio'][ids]
    return {'delta_ratio_mean':float(ratio.mean()),'delta_ratio_p50':float(ratio.median()),
            'delta_ratio_p95':float(torch.quantile(ratio,.95)),
            'near_bound_fraction':float((ratio>=.099).float().mean()),
            'beta_entropy':float((-(beta*beta.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
            'beta_max':float(beta.max(-1).values.mean()),
            'attention_entropy':float((-(attn*attn.clamp_min(1e-12).log()).sum(-1)/math.log(16)).mean()),
            'null_weight':float(attn[...,-1].mean()),
            'message_norm':float(d['m_local'][ids].norm(dim=-1).mean()),
            'global_query_norm':float(d['query'][ids].norm(dim=-1).mean()),
            'global_read_query_norm':float(d['q_read'][ids].norm(dim=-1).mean())}


def train_worker(group, fold):
    if group not in GROUPS or fold not in range(1,11): raise ValueError((group,fold))
    frozen=json.loads((RESULTS/'preflight.json').read_text())
    frozen_sha=frozen['source_sha256']
    if frozen['status']!='PASS' or source_sha()!=frozen_sha:
        raise RuntimeError('training code/config/history differs from passing preflight')
    task_file,best_file,latest_file,log_file=paths(group,fold)
    if task_file.exists():
        task=json.loads(task_file.read_text())
        if _task_complete(group,fold,frozen_sha):
            print(f'SKIP completed {group} fold={fold}',flush=True)
            return task
        raise RuntimeError(f'incompatible task ledger: {task_file}')
    state,warmup,identity=prepare(fold)
    module=make_module(fold,state['x'].device)
    optimizer=torch.optim.AdamW(module.parameters(),lr=.0005,weight_decay=.01)
    _validate_optimizer_settings(state,optimizer)
    initial_sha=sha_value(_cpu_state(module))
    expected=frozen['folds'][fold-1]
    if identity!=expected['identity'] or initial_sha!=expected['initial_module_sha256']:
        raise RuntimeError('initial identity differs from preflight')
    hist,rng=runtime().load_history(),_rng_runtime()
    assert rng.rng_sha(rng.pack_rng())==warmup['rng_sha']
    start_epoch,wait,best_acc,best_epoch,best,trace=10,0,-1.,-1,None,[]
    prior_wall=0.0
    if latest_file.exists():
        saved=torch_load(latest_file)
        for key,expected in [('group',group),('fold',fold),('source_sha256',frozen_sha),('identity',identity)]:
            if saved.get(key)!=expected: raise RuntimeError(f'latest incompatibility: {key}')
        load_bundle(state,module,optimizer,saved['current']); rng.restore_rng(saved['rng'])
        start_epoch,wait=int(saved['next_epoch']),int(saved['wait'])
        best_acc,best_epoch,best,trace=saved['best_acc'],saved['best_epoch'],saved['best'],saved['trace']
        prior_wall=float(saved.get('wall_seconds',0.))
        # Replay epochs since the most recent complete state; retain no duplicate log rows.
        if log_file.exists():
            lines=[line for line in log_file.read_text().splitlines() if json.loads(line)['epoch']<=start_epoch]
            log_file.write_text('\n'.join(lines)+'\n')
        print(f'RESUME {group} fold={fold} epoch={start_epoch}',flush=True)
    elif log_file.exists():
        raise RuntimeError('orphan log without latest state; inspect before restarting')
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic(); executed=start_epoch
    for epoch in range(start_epoch,CFG['base']['max_epochs']):
        before=rng.rng_sha(rng.pack_rng())
        hist.zero_optimizers(state); optimizer.zero_grad(set_to_none=True)
        train=forward_fusion(state,module,group,training=True)
        loss=train['total']
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite loss')
        loss.backward()
        if not all(_finite_module_grads(m) for m in (state['model'],state['cam'],module)):
            raise FloatingPointError('nonfinite gradient')
        norms=grad_snapshot(state,module)
        route=train['route_gradients'].copy()
        route['eta']=.1
        route['routed_cam144_norm']=.1*route.get('z_rel_input',0.)
        route['routed_to_total_cam144_ratio']=route['routed_cam144_norm']/max(route.get('cam144_total',0.),1e-30)
        for key in ('opt_base','opt_cam','opt_added'): state[key].step()
        optimizer.step()
        after_train=rng.rng_sha(rng.pack_rng()); executed=epoch+1
        with torch.no_grad():
            ev=forward_fusion(state,module,group,training=False)
            metrics=base_runtime().classification_metrics(ev['logp'],state['test'],state['y'])
            diagnostics=_eval_summary(ev,state)
        after_eval=rng.rng_sha(rng.pack_rng())
        entry={'epoch':executed,'rng_before':before,'rng_after_train':after_train,'rng_after_eval':after_eval}
        trace.append(entry)
        if float(metrics['acc'])>best_acc:
            best_acc,best_epoch,wait=float(metrics['acc']),epoch,0
            best={'bundle':bundle(state,module,optimizer),'metrics':_metric_copy(metrics),
                  'test_probability':metrics['prob'].copy(),'test_prediction':metrics['pred'].copy(),
                  'losses':_stats(train),'diagnostics':diagnostics,
                  'rng_after_train':after_train,'rng_after_eval':after_eval}
            atomic_torch(best_file,{'experiment':CFG['experiment'],'group':group,'fold':fold,'seed':0,
                         'source_sha256':frozen_sha,'identity':identity,'warmup_sha256':identity['warmup_sha256'],
                         'best_epoch_1based':executed,'metrics':best['metrics'],'best':best,'rng_trace':trace.copy()})
        else: wait+=1
        log_jsonl(log_file,{'group':group,'fold':fold,'epoch':executed,'best_acc':best_acc,
                          'best_epoch_1based':best_epoch+1,'metrics':_metric_copy(metrics),'losses':_stats(train),
                          'gradient_norms':norms,'gradient_route':route,'diagnostics':diagnostics,'rng':entry})
        if executed%10==0 or executed==CFG['base']['max_epochs'] or wait>CFG['base']['early_stop_wait_gt']:
            atomic_torch(latest_file,{'group':group,'fold':fold,'source_sha256':frozen_sha,'identity':identity,
                         'warmup_sha256':identity['warmup_sha256'],'current':bundle(state,module,optimizer),
                         'rng':rng.pack_rng(),'next_epoch':executed,'wait':wait,'best_acc':best_acc,
                         'best_epoch':best_epoch,'best':best,'trace':trace,
                         'wall_seconds':prior_wall+time.monotonic()-started})
            print(f'{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch+1} wait={wait}',flush=True)
        del train,ev
        if wait>CFG['base']['early_stop_wait_gt']: break
    if best is None: raise RuntimeError('no selected best state')
    task={'status':'COMPLETE','group':group,'fold':fold,'seed':0,'source_sha256':frozen_sha,'identity':identity,
          'initial_module_sha256':initial_sha,'best_epoch_1based':best_epoch+1,'epochs_executed':executed,
          'metrics':best['metrics'],'best_checkpoint_sha256':sha_file(best_file),
          'latest_checkpoint_sha256':sha_file(latest_file),'rng_trace':trace,
          'wall_seconds':prior_wall+time.monotonic()-started,'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()),
          'device':torch.cuda.get_device_name(0)}
    atomic_json(task_file,task)
    print(f'COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch+1}',flush=True)
    return task


def _task_complete(group,fold,frozen_sha):
    task_file,best_file,latest_file,_=paths(group,fold)
    if not all(p.is_file() for p in (task_file,best_file,latest_file)): return False
    task=json.loads(task_file.read_text())
    return (task.get('status')=='COMPLETE' and task.get('source_sha256')==frozen_sha
            and task.get('best_checkpoint_sha256')==sha_file(best_file)
            and task.get('latest_checkpoint_sha256')==sha_file(latest_file))


def _safe_worker_limit(preflight):
    free,_=torch.cuda.mem_get_info()
    measured=int(preflight['peak_gpu_bytes_observed'])
    completed=[json.loads(p.read_text()) for p in TASKS.glob('*.json')]
    if completed: measured=max(int(t['peak_gpu_bytes']) for t in completed)
    capacity=int((free*.70)//max(measured*2,1))
    if capacity<1: raise RuntimeError('insufficient safe GPU memory even for one worker')
    return min(10,capacity)


def run_workers(max_workers=None):
    frozen=json.loads((RESULTS/'preflight.json').read_text())
    if source_sha()!=frozen['source_sha256']: raise RuntimeError('frozen source changed')
    limit=_safe_worker_limit(frozen)
    if max_workers is not None: limit=min(limit,max(1,int(max_workers)))
    free,total=torch.cuda.mem_get_info()
    pending=[(g,f) for f in range(1,11) for g in GROUPS if not _task_complete(g,f,frozen['source_sha256'])]
    active,failures={},[]
    started=time.monotonic(); started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    while pending or active:
        while pending and len(active)<limit:
            free_now,_=torch.cuda.mem_get_info()
            # Runtime admission uses actual free memory; never overfill a busy card.
            reserve=max(total*.15,int(frozen['peak_gpu_bytes_observed'])*2)
            if free_now < reserve + int(frozen['peak_gpu_bytes_observed'])*2:
                if not active: raise RuntimeError('external GPU occupancy prevents safe worker admission')
                break
            g,f=pending.pop(0)
            dispatch=LOGS/f'dispatch_{g}_f{f:02d}.log'
            stream=dispatch.open('a')
            env=os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                        'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
            proc=subprocess.Popen([sys.executable,str(HERE/'run.py'),'--worker','--group',g,'--fold',str(f)],
                                  cwd=HERE,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[proc]=(g,f,stream,dispatch)
        for proc,(g,f,stream,dispatch) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            stream.close(); del active[proc]
            if code!=0:
                failures.append({'group':g,'fold':f,'exit_code':code,'log':str(dispatch),
                                 'tail':dispatch.read_text(errors='replace')[-4000:]})
        complete=sum(_task_complete(g,f,frozen['source_sha256']) for g in GROUPS for f in range(1,11))
        atomic_json(RUNTIME/'scheduler.json',{'started_utc':started_utc,'worker_limit':limit,'completed_tasks':complete,
                    'active':[{'group':v[0],'fold':v[1]} for v in active.values()],
                    'pending':len(pending),'failures':failures})
        if pending or active: time.sleep(1.0)
    summary={'completed_tasks':complete,'expected_tasks':20,'wall_seconds':time.monotonic()-started,
             'failures':failures,'max_workers':limit,'started_utc':started_utc,
             'free_gpu_bytes_at_dispatch':int(free),'total_gpu_bytes':int(total),
             'peak_preflight_bytes':frozen['peak_gpu_bytes_observed']}
    atomic_json(RUNTIME/'training_summary.json',summary)
    if complete!=20: raise RuntimeError(f'training incomplete {complete}/20; inspect failures')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--group'); parser.add_argument('--fold',type=int)
    parser.add_argument('--preflight-only',action='store_true'); parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--workers',type=int)
    args=parser.parse_args()
    if args.worker: train_worker(args.group,args.fold); return
    pf=RESULTS/'preflight.json'
    if pf.exists():
        frozen=json.loads(pf.read_text())
        if frozen.get('status')!='PASS' or frozen.get('source_sha256')!=source_sha():
            raise RuntimeError('existing preflight/source mismatch: inspect before rerunning')
    elif args.train_only:
        raise RuntimeError('preflight missing')
    else:
        frozen=preflight()
        atomic_json(HERE/'manifest.json',{'status':'PREFLIGHT_PASS','source_sha256':frozen['source_sha256'],
                    'source_map':frozen['source_map'],'preflight_sha256':sha_file(pf),'formal_tasks_expected':20,
                    'completed_tasks':0,'directory_fallback_reason':'OSSFS mkdir errno=5 EIO'})
    if args.preflight_only: print('PREFLIGHT PASS',flush=True); return
    summary=run_workers(args.workers)
    result=subprocess.run([sys.executable,str(HERE/'audit.py')],cwd=HERE)
    if result.returncode: raise RuntimeError('independent audit failed')
    manifest=json.loads((HERE/'manifest.json').read_text())
    manifest.update({'status':'COMPLETE_AUDITED','completed_tasks':20,'audit_status':'PASS',
                     'training_wall_seconds':summary['wall_seconds'],'worker_limit':summary['max_workers']})
    atomic_json(HERE/'manifest.json',manifest)
    print('COMPLETE_AUDITED',flush=True)


if __name__=='__main__': main()
