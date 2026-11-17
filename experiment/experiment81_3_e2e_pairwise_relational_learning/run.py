#!/usr/bin/env python3
"""Experiment 81.3 runner with protected end-to-end relational gradients."""
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

from relation_learning import PairwiseRelationalCorrection, GROUP_UNARY, GROUP_PAIRWISE

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
        spec = importlib.util.spec_from_file_location("experiment81_3_path_adapter", adapter_path)
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
        "config.json", "design.md", "path_adapter.py", "relation_learning.py", "run.py", "audit.py"
    )]
    paths.extend(Path(path) for path in base.source_manifest())
    e812 = Path("/model/spromptgl/experiment/experiment81_2_e2e_relational_correction")
    for name in ("relation_correction.py", "run.py", "config.json", "design.md", "path_adapter.py", "audit.py",
                 "results/preflight.json", "results/report.md", "results/fold_metrics.csv",
                 "results/mechanism.csv", "results/audit_status.json"):
        paths.append(e812 / name)
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


def _overlap_rows(fold):
    return (376, 406) if fold == 2 else ()


def _stats(out):
    keys = ("cls", "graph", "pair", "adv", "preserve", "total")
    return {key: float(out[key].detach().cpu()) for key in keys if key in out and torch.is_tensor(out[key])}


def forward_with_correction(state, correction, group, training=False, intervention="full", eta=None):
    """Run the historical path once and attach a correction to its captured output."""
    correction.train(training)
    capture = {"calls": 0, "z_cam": None, "base_logp": None, "hidden": None}
    layer = state["cam"].Outputlayer

    def pre_hook(_layer, args):
        if not args or not torch.is_tensor(args[0]) or tuple(args[0].shape) != (871, 144):
            raise RuntimeError("original OutputLayer input is not CAM144 [871,144]")
        capture["z_cam"] = args[0]

    def output_hook(_layer, _args, output):
        capture["calls"] += 1
        if capture["calls"] != 1 or not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("original OutputLayer must run once and return (logp, hidden)")
        capture["base_logp"], capture["hidden"] = output
        # Return the unmodified historical output.  DARG's original cls path
        # therefore remains exactly the original graph.
        return output

    h_pre = layer.register_forward_pre_hook(pre_hook)
    h_out = layer.register_forward_hook(output_hook)
    try:
        out = runtime().forward(state, BASE_GROUP, training=training)
    finally:
        h_pre.remove()
        h_out.remove()
    if capture["calls"] != 1:
        raise RuntimeError(f"OutputLayer call count was {capture['calls']}, expected one")
    logp_base = capture["base_logp"]
    logp_anchor = logp_base.detach()
    logp_final, details = correction(
        capture["z_cam"], logp_anchor, GROUP_MODE[group], training=training,
        overlap_rows=_overlap_rows(state["fold"]), intervention=intervention, eta=eta,
    )
    out["logp_base"] = logp_base
    out["logp_final"] = logp_final
    out["z_cam"] = capture["z_cam"]
    out["correction_details"] = details
    out["output_hidden"] = capture["hidden"]
    out["darg_total"] = out["total"]
    return out


def _legal_train_indices(state, details):
    train = state["train"].to(device=details["delta_margin"].device, dtype=torch.long)
    overlap = _overlap_rows(state["fold"])
    if overlap:
        blocked = torch.as_tensor(overlap, dtype=torch.long, device=train.device)
        train = train[~torch.isin(train, blocked)]
    return train


def correction_loss(out, state):
    details = out["correction_details"]
    legal_train = _legal_train_indices(state, details)
    if legal_train.numel() == 0:
        return details["delta_margin"].sum() * 0.0
    return details["delta_margin"][legal_train].square().mean()


def increment_loss(out, state):
    utils = base_runtime().runtime()["utils"]
    details = out["correction_details"]
    train = state["train"]
    corrected = utils.ClsLoss(out["logp_final"], state["y"], train, state["weights"])
    anchor = utils.ClsLoss(details["logp_anchor"], state["y"], train, state["weights"])
    return corrected - anchor


def total_loss(out, state):
    # out["total"] is the unmodified historical DARG objective.
    darg_total = out["darg_total"]
    increment = increment_loss(out, state)
    corr = correction_loss(out, state)
    total = darg_total + increment + 0.001 * corr
    out.update({"cls_base": out["cls"], "cls_final": increment + out["cls"],
                "increment": increment, "corr": corr, "darg_total": darg_total,
                "total": total})
    return total


def grad_norm(parameters):
    pieces = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0


def grad_snapshot(state, correction):
    return {
        "darg": grad_norm(state["model"].parameters()),
        "cam": grad_norm(state["cam"].parameters()),
        "relation": grad_norm(correction.parameters()),
        "local_encoder": grad_norm(list(correction.local_proj.parameters()) + list(correction.local_norm.parameters()) + [correction.position]),
        "key": grad_norm(list(correction.key_proj.parameters()) + list(correction.key_norm.parameters())),
        "value": grad_norm(correction.value_proj.parameters()),
        "query": grad_norm(list(correction.query_proj.parameters()) + list(correction.query_norm.parameters())),
        "readout": grad_norm(list(correction.readout_hidden.parameters()) + list(correction.readout_score.parameters())),
        "gate": grad_norm(correction.gate.parameters()),
        "correction_head": grad_norm(correction.correction_head.parameters()),
    }


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
    _tree_close(left, right, atol=1e-7, rtol=1e-7, where="gradients")


def _one_step(state, correction, optimizer, group, with_correction, eta=None):
    hist = runtime().load_history()
    hist.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    if with_correction:
        out = forward_with_correction(state, correction, group, training=True, eta=eta)
        loss = total_loss(out, state)
    else:
        out = _base_forward(state, training=True)
        loss = out["total"]
    loss.backward()
    gradients = _gradient_copy(state)
    grad_summary = grad_snapshot(state, correction)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    if with_correction:
        optimizer.step()
    return out, float(loss.detach().cpu()), gradients, grad_summary, hist.state_bundle(state), _rng_runtime().pack_rng()


def _rng_runtime():
    return base_runtime()


def _replay_step(state, correction, optimizer, group):
    hist = runtime().load_history()
    hist.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    out = forward_with_correction(state, correction, group, training=True)
    loss = total_loss(out, state)
    loss.backward()
    if not _finite_module_grads(state["model"]) or not _finite_module_grads(state["cam"]) or not _finite_module_grads(correction):
        raise RuntimeError("recovery replay produced non-finite gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    result = {
        "loss": float(loss.detach().cpu()),
        "logp": out["logp_final"].detach().cpu().clone(),
        "attention": out["correction_details"]["attention"].detach().cpu().clone(),
        "relation_summary": out["correction_details"]["relation_summary"].detach().cpu().clone(),
        "beta": out["correction_details"]["beta"].detach().cpu().clone(),
        "state": bundle(state, correction, optimizer),
        "rng": _rng_runtime().pack_rng(),
    }
    return result


def _module_param_shapes(module):
    return {name: (tuple(p.shape), p.numel()) for name, p in module.named_parameters()}


def module_preflight():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(813201)
    z = torch.randn((3, 144), generator=generator)
    logp = torch.log_softmax(torch.randn((3, 2), generator=generator), dim=-1)
    g1 = PairwiseRelationalCorrection.initialized(813345, torch.device("cpu"))
    g2 = PairwiseRelationalCorrection.initialized(813345, torch.device("cpu"))
    if _cpu_state(g1).keys() != _cpu_state(g2).keys():
        raise RuntimeError("G1/G2 parameter names differ")
    _tree_close(_cpu_state(g1), _cpu_state(g2), atol=0.0, rtol=0.0, where="module_init")
    if _module_param_shapes(g1) != _module_param_shapes(g2):
        raise RuntimeError("G1/G2 parameter shapes/counts differ")
    g1_logp, d1 = g1(z, logp, GROUP_UNARY, False)
    g2_logp, d2 = g2(z, logp, GROUP_PAIRWISE, False)
    for candidate in (d1, d2):
        expected = {
            "tokens": (3, 16, 32), "keys": (3, 16, 15, 32), "values": (3, 16, 15, 32),
            "attention": (3, 8, 16, 16), "messages": (3, 8, 16, 32),
            "m_local": (3, 16, 32), "relation_summary": (3, 32), "beta": (3, 16), "logp_final": (3, 2),
        }
        for key, shape in expected.items():
            if tuple(candidate[key].shape) != shape:
                raise RuntimeError(f"{key} shape mismatch: {tuple(candidate[key].shape)}")
        if not torch.isfinite(candidate["logp_final"]).all():
            raise RuntimeError("relation output contains NaN/Inf")
    if not torch.equal(g1_logp, logp) or not torch.equal(g2_logp, logp):
        raise RuntimeError("zero correction failed exact identity")
    if not torch.equal(g1.correction_head.weight, torch.zeros_like(g1.correction_head.weight)):
        raise RuntimeError("correction head is not zero initialized")
    if float(g1.gate[2].bias.detach().cpu()) != -1.0:
        raise RuntimeError("gate final bias is not -1")
    if int(g1.receiver_index.numel()) != 240 or int(g1.sender_index.numel()) != 240:
        raise RuntimeError("wrong directed candidate count")
    if not bool((g1.candidate_map.diag() == -1).all()) or int((g1.candidate_map >= 0).sum()) != 240:
        raise RuntimeError("self-edge or candidate map error")
    null_logp, null_details = g2(z, logp, GROUP_PAIRWISE, False, intervention="null_only")
    for key in ("messages", "m_local", "relation_summary"):
        if not torch.equal(null_details[key], torch.zeros_like(null_details[key])):
            raise RuntimeError(f"Null-only {key} is not exactly zero")
    if not torch.equal(null_logp, logp):
        raise RuntimeError("Null-only logp is not exact identity")

    receiver = torch.randn((2, 16, 15, 32), generator=generator)
    sender = torch.randn((2, 16, 15, 32), generator=generator)
    r1, v1 = g1._construct_features(receiver, sender, GROUP_UNARY)
    r2, v2 = g1._construct_features(receiver + 1.0, sender, GROUP_UNARY)
    if not torch.equal(r1, r2) or not torch.equal(v1, v2):
        raise RuntimeError("G1 key/value explicitly used receiver features")
    p1, pv1 = g2._construct_features(receiver, sender, GROUP_PAIRWISE)
    p2, pv2 = g2._construct_features(receiver + 1.0, sender, GROUP_PAIRWISE)
    if torch.equal(p1, p2) or torch.equal(pv1, pv2):
        raise RuntimeError("G2 key/value did not use receiver/sender pair features")
    if torch.equal(d1["relation_summary"], d2["relation_summary"]):
        raise RuntimeError("G1 and G2 relation representations are identical on probe input")
    max_delta = float(d1["delta_margin"].abs().max().detach().cpu())
    if max_delta > 1.0 + 1e-8:
        raise RuntimeError("correction bound exceeded")
    return {
        "status": "PASS", "input_shape": [3, 144], "token_shape": [3, 16, 9],
        "encoded_token_shape": [3, 16, 32], "relation_key_shape": [3, 16, 15, 32],
        "relation_value_shape": [3, 16, 15, 32], "attention_shape": [3, 8, 16, 16],
        "message_shape": [3, 8, 16, 32], "readout_shape": [3, 32], "final_logp_shape": [3, 2],
        "no_self_edges": True, "neighbor_count": 15, "null_candidates": 1,
        "initial_zero_correction_exact": True, "null_only_exact_identity": True,
        "g1_g2_parameter_names_shapes_counts_equal": True,
        "g1_unary_excludes_receiver_from_key_value": True,
        "g2_pairwise_uses_receiver_sender": True,
        "real_relation_representation_differs": True,
        "correction_bound": 1.0, "trainable_parameter_count": sum(p.numel() for p in g1.parameters()),
        "module_init_sha": sha_value(_cpu_state(g1)),
    }


def _gradient_wrt_z(out, key="increment"):
    value = out[key]
    grad = torch.autograd.grad(value, out["z_cam"], retain_graph=True, allow_unused=True)[0]
    return None if grad is None else grad.detach().cpu().clone()


def _parameter_gradient_snapshot(state):
    return _gradient_copy(state)


def preflight():
    for path in (RESULTS, RUNTIME, CHECKPOINTS, TASKS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"one CUDA-compatible device required; available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    hist = runtime().load_history()
    rng_runtime = _rng_runtime()
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    structure = module_preflight()
    folds = []
    peak_observed = 0

    for fold in range(1, 11):
        state, warmup, identity = prepare(fold)
        if rng_runtime.rng_sha(rng_runtime.pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError(f"fold {fold} did not restore exact E61 epoch-10 RNG")
        initial_bundle = hist.state_bundle(state)
        initial_rng = rng_runtime.pack_rng()
        initial_rng_sha = rng_runtime.rng_sha(initial_rng)

        modules = []
        for group in GROUPS:
            before = rng_runtime.rng_sha(rng_runtime.pack_rng())
            module = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
            after = rng_runtime.rng_sha(rng_runtime.pack_rng())
            if before != after:
                raise RuntimeError("private relation initialization consumed public RNG")
            modules.append(module)
        _tree_close(_cpu_state(modules[0]), _cpu_state(modules[1]), atol=0.0, rtol=0.0, where=f"fold{fold}.group_init")

        # One historical base forward and one hooked forward, both from the
        # exact same warmup state/RNG.
        base_calls = {"count": 0}
        handle = state["cam"].Outputlayer.register_forward_hook(lambda *_: base_calls.__setitem__("count", base_calls["count"] + 1))
        base_out = _base_forward(state, training=False)
        handle.remove()
        if base_calls["count"] != 1:
            raise RuntimeError(f"fold {fold} base OutputLayer call count != 1")
        plain_logp = base_out["logp"].detach().clone()
        plain_losses = _stats(base_out)
        rng_after_plain = rng_runtime.pack_rng()
        peak_observed = max(peak_observed, int(torch.cuda.max_memory_allocated()))

        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        correction = modules[0]
        relation_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
        _validate_optimizer_settings(state, relation_optimizer)
        hooked = forward_with_correction(state, correction, GROUPS[0], training=False)
        if not torch.equal(hooked["logp"], plain_logp) or not torch.equal(hooked["logp_final"], plain_logp):
            raise RuntimeError(f"fold {fold} zero correction changed base/final logp")
        if not torch.equal(hooked["output_hidden"], hooked["hidden"]):
            raise RuntimeError("OutputLayer hidden representation changed")
        if plain_losses.keys() != _stats(hooked).keys():
            raise RuntimeError("historical loss keys changed under hook")
        for key, value in plain_losses.items():
            if abs(value - _stats(hooked)[key]) > 1e-7:
                raise RuntimeError(f"fold {fold} base loss mismatch at {key}")
        if rng_runtime.rng_sha(rng_runtime.pack_rng()) != rng_runtime.rng_sha(rng_after_plain):
            raise RuntimeError("hooked eval changed public RNG")

        # Base-off parity includes one optimizer update and the exact public RNG.
        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        base_step, base_loss, base_grads, _, base_updated, base_rng = _one_step(
            state, correction, relation_optimizer, GROUPS[0], with_correction=False
        )
        base_logp = base_step["logp"].detach().cpu().clone()

        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        correction = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
        relation_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
        corrected_step, corrected_loss, corrected_grads, grad_summary, corrected_updated, corrected_rng = _one_step(
            state, correction, relation_optimizer, GROUPS[0], with_correction=True, eta=0.1
        )
        if not torch.equal(corrected_step["logp"].detach().cpu(), base_logp):
            raise RuntimeError(f"fold {fold} base logp changed at zero correction")
        if not torch.equal(corrected_step["logp_final"].detach().cpu(), base_logp):
            raise RuntimeError(f"fold {fold} final logp changed at zero correction")
        if abs(base_loss - corrected_loss) > 1e-7:
            raise RuntimeError(f"fold {fold} base-off loss mismatch: {base_loss} vs {corrected_loss}")
        _compare_gradients(base_grads, corrected_grads)
        _tree_close(base_updated, corrected_updated, atol=1e-7, rtol=1e-7, where=f"fold{fold}.base_update")
        if rng_runtime.rng_sha(base_rng) != rng_runtime.rng_sha(corrected_rng):
            raise RuntimeError(f"fold {fold} base-off public RNG mismatch")
        if grad_summary["correction_head"] <= 0:
            raise RuntimeError(f"fold {fold} zero correction head got no classification gradient")

        route = None
        recovery = None
        if fold == 1:
            # Make the correction head nonzero using a deterministic in-place
            # assignment, then compare eta=0, .1 and 1 on identical states.
            initial_corr = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
            with torch.no_grad():
                initial_corr.correction_head.weight.fill_(0.05)
            eta_runs = {}
            base_route_grads = None
            for eta in (0.0, 0.1, 1.0):
                hist.load_bundle(state, initial_bundle)
                rng_runtime.restore_rng(initial_rng)
                corr_eta = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device, eta=eta)
                corr_eta.load_state_dict(_cpu_state(initial_corr), strict=True)
                opt_eta = torch.optim.AdamW(corr_eta.parameters(), lr=0.0005, weight_decay=0.01)
                hist.zero_optimizers(state)
                opt_eta.zero_grad(set_to_none=True)
                out_eta = forward_with_correction(state, corr_eta, GROUPS[0], training=True, eta=eta)
                loss_eta = total_loss(out_eta, state)
                inc_z = _gradient_wrt_z(out_eta)
                loss_eta.backward()
                eta_runs[str(eta)] = {
                    "increment_z_grad_norm": 0.0 if inc_z is None else float(torch.linalg.vector_norm(inc_z).cpu()),
                    "increment_z_grad": inc_z,
                    "cam_total_grad": grad_norm(state["cam"].parameters()),
                    "darg_total_grad": grad_norm(state["model"].parameters()),
                    "relation_grad": grad_norm(corr_eta.parameters()),
                }
                if eta == 0.0:
                    base_route_grads = _parameter_gradient_snapshot(state)
                del out_eta, corr_eta, opt_eta
            g01, g1 = eta_runs["0.1"]["increment_z_grad"], eta_runs["1.0"]["increment_z_grad"]
            if g01 is None or g1 is None or float(torch.linalg.vector_norm(g01)) <= 0 or float(torch.linalg.vector_norm(g1)) <= 0:
                raise RuntimeError("eta route did not produce nonzero CAM144 gradients")
            ratio_error = float((g01 - 0.1 * g1).abs().max() / g1.abs().clamp_min(1e-8).max())
            if ratio_error > 2e-4:
                raise RuntimeError(f"eta gradient scaling mismatch: relative max error {ratio_error}")
            if eta_runs["0.0"]["increment_z_grad_norm"] != 0.0:
                raise RuntimeError("eta=0 did not block incremental CAM144 gradient")
            if eta_runs["0.0"]["darg_total_grad"] <= 0 or eta_runs["0.1"]["darg_total_grad"] <= 0:
                raise RuntimeError("DARG gradient disappeared under eta routing")
            if eta_runs["0.1"]["relation_grad"] <= 0:
                raise RuntimeError("relation branch did not receive gradient")
            route = {
                "eta0_increment_cam144_grad_zero": True,
                "eta01_increment_cam144_grad_nonzero": True,
                "eta1_increment_cam144_grad_nonzero": True,
                "eta01_to_eta1_relative_max_error": ratio_error,
                "eta0_darg_grad_norm": eta_runs["0.0"]["darg_total_grad"],
                "eta01_darg_grad_norm": eta_runs["0.1"]["darg_total_grad"],
                "eta01_relation_grad_norm": eta_runs["0.1"]["relation_grad"],
            }

            hist.load_bundle(state, initial_bundle)
            rng_runtime.restore_rng(initial_rng)
            correction = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
            relation_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
            _replay_step(state, correction, relation_optimizer, GROUPS[1])
            after_first = bundle(state, correction, relation_optimizer)
            rng_after_first = rng_runtime.pack_rng()
            uninterrupted_second = _replay_step(state, correction, relation_optimizer, GROUPS[1])
            hist.load_bundle(state, after_first["base"])
            correction.load_state_dict(after_first["relation"], strict=True)
            relation_optimizer.load_state_dict(copy.deepcopy(after_first["optimizer_relation"]))
            rng_runtime.restore_rng(rng_after_first)
            resumed_second = _replay_step(state, correction, relation_optimizer, GROUPS[1])
            _tree_close(uninterrupted_second["state"], resumed_second["state"], atol=0.0, rtol=0.0, where="two_step_recovery")
            for key in ("logp", "attention", "relation_summary", "beta"):
                if not torch.equal(uninterrupted_second[key], resumed_second[key]):
                    raise RuntimeError(f"two-step replay diverged at {key}")
            if uninterrupted_second["loss"] != resumed_second["loss"] or rng_runtime.rng_sha(uninterrupted_second["rng"]) != rng_runtime.rng_sha(resumed_second["rng"]):
                raise RuntimeError("two-step interruption/recovery RNG/loss diverged")
            recovery = "PASS"

        folds.append({
            "fold": fold, "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": warmup["rng_sha"],
            "base_model_cam_sha": identity["model_cam_sha"], "base_optimizer_sha": identity["optimizer_sha"],
            "split_sha": identity["split_sha"], "pair_sha": identity["pair_sha"], "pair_count": 4096,
            "train_test_overlap_1based": identity["membership"]["train_test_overlap_1based"],
            "public_rng_unchanged_by_init": initial_rng_sha == rng_runtime.rng_sha(initial_rng),
            "group_parameter_init_equal": True, "outputlayer_single_call": True,
            "zero_correction_logp_exact": True, "base_off_loss_grad_update_rng_parity": True,
            "correction_head_first_step_grad_norm": grad_summary["correction_head"],
            "route_gradient_test": route, "two_step_interruption_recovery_replay": recovery,
        })
        del base_out, hooked, base_step, corrected_step, state, correction, relation_optimizer, modules
        torch.cuda.empty_cache()

    result = {
        "status": "PASS", "device": device_name, "free_memory_bytes_at_start": int(free_bytes),
        "total_memory_bytes": int(total_bytes), "peak_gpu_bytes_observed": int(peak_observed),
        "source_map": source_map(), "source_sha256": source_sha(), "module_tests": structure,
        "folds": folds, "base_off_parity": {"passed": 10, "expected": 10},
        "outputlayer_calls_per_forward": 1, "formal_training_started": False,
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def train_worker(group, fold):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task group={group} fold={fold}")
    frozen = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "PASS" or source_sha() != frozen.get("source_sha256"):
        raise RuntimeError("training source SHA differs from passing preflight")
    state, warmup, identity = prepare(fold)
    state["group"] = group
    task_file, best_file, latest_file, log_file = paths(group, fold)
    correction = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
    _validate_optimizer_settings(state, optimizer)
    initial_module_sha = sha_value(_cpu_state(correction))
    if task_file.exists():
        previous = json.loads(task_file.read_text(encoding="utf-8"))
        if previous.get("status") == "COMPLETE" and previous.get("source_sha256") == frozen["source_sha256"]:
            raise RuntimeError(f"refusing to overwrite completed task {task_file}")
        raise RuntimeError(f"incompatible task ledger exists {task_file}")

    hist = runtime().load_history()
    rng = _rng_runtime()
    if rng.rng_sha(rng.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not begin at E61 epoch-10 RNG")
    start_epoch, wait = 10, 0
    best_acc, best_epoch, best, trace = -1.0, -1, None, []
    if latest_file.exists():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", frozen["source_sha256"]),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"incompatible latest state at {key}")
        load_bundle(state, correction, optimizer, saved["current"])
        rng.restore_rng(saved["rng"])
        start_epoch, wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best, trace = float(saved["best_acc"]), int(saved["best_epoch"]), saved["best"], saved["trace"]

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    executed = start_epoch
    for epoch in range(start_epoch, CFG["base"]["max_epochs"]):
        rng_before = rng.rng_sha(rng.pack_rng())
        hist.zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        train_out = forward_with_correction(state, correction, group, training=True)
        loss = total_loss(train_out, state)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite total loss at {group} fold {fold} epoch {epoch + 1}")
        loss.backward()
        if not _finite_module_grads(state["model"]) or not _finite_module_grads(state["cam"]) or not _finite_module_grads(correction):
            raise FloatingPointError(f"non-finite gradient at {group} fold {fold} epoch {epoch + 1}")
        gradient_values = grad_snapshot(state, correction)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_after_train = rng.rng_sha(rng.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            eval_out = forward_with_correction(state, correction, group, training=False)
            metrics_fn = base_runtime().classification_metrics
            final_metrics = metrics_fn(eval_out["logp_final"], state["test"], state["y"])
            base_metrics = metrics_fn(eval_out["logp"], state["test"], state["y"])
            details = eval_out["correction_details"]
            test_tensor = torch.as_tensor(state["test"], dtype=torch.long, device=details["delta_margin"].device)
            mean_delta = float(details["delta_margin"][test_tensor].abs().mean().cpu())
            mean_gate = float(details["gate"][test_tensor].mean().cpu())
            mean_null = float(details["null_weight"][test_tensor].mean().cpu())
            mean_entropy = float(details["entropy"][test_tensor].mean().cpu())
            beta = details["beta"][test_tensor]
            beta_entropy = float((-(beta * beta.clamp_min(1e-12).log()).sum(-1) / math.log(16.0)).mean().cpu())
            beta_max = float(beta.max(-1).values.mean().cpu())
        rng_after_eval = rng.rng_sha(rng.pack_rng())
        epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_rng)

        if float(final_metrics["acc"]) > best_acc:
            best_acc, best_epoch, wait = float(final_metrics["acc"]), epoch, 0
            best = {
                "bundle": bundle(state, correction, optimizer),
                "metrics": _metric_copy(final_metrics), "base_metrics": _metric_copy(base_metrics),
                "test_probability": final_metrics["prob"].copy(), "test_prediction": final_metrics["pred"].copy(),
                "base_probability": base_metrics["prob"].copy(), "base_prediction": base_metrics["pred"].copy(),
                "losses": {"cls_base": float(train_out["cls_base"].detach().cpu()),
                           "cls_final": float(train_out["cls_final"].detach().cpu()),
                           "darg_total": float(train_out["darg_total"].detach().cpu()),
                           "increment": float(train_out["increment"].detach().cpu()),
                           "corr": float(train_out["corr"].detach().cpu()),
                           "total": float(loss.detach().cpu())},
                "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval,
                "mean_abs_delta": mean_delta, "mean_gate": mean_gate,
                "mean_null": mean_null, "mean_entropy": mean_entropy,
                "beta_entropy": beta_entropy, "beta_max": beta_max,
            }
            atomic_torch(best_file, {
                "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
                "source_sha256": frozen["source_sha256"], "identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "best_epoch_1based": epoch + 1,
                "metrics": best["metrics"], "best": best, "rng_trace": trace,
            })
        else:
            wait += 1

        record = {
            "group": group, "fold": fold, "epoch": epoch + 1, "best_epoch_1based": best_epoch + 1,
            "best_acc": best_acc, "metrics": _metric_copy(final_metrics), "base_metrics": _metric_copy(base_metrics),
            "train_count": int(len(state["train_np"])), "test_count": int(len(state["test_np"])),
            "losses": {"cls_base": float(train_out["cls_base"].detach().cpu()),
                       "darg_total": float(train_out["darg_total"].detach().cpu()),
                       "increment": float(train_out["increment"].detach().cpu()),
                       "corr": float(train_out["corr"].detach().cpu()), "total": float(loss.detach().cpu()),
                       "graph": float(train_out["graph"].detach().cpu()), "pair": float(train_out["pair"].detach().cpu()),
                       "adv": float(train_out["adv"].detach().cpu()), "preserve": float(train_out["preserve"].detach().cpu())},
            "gradient_norms": gradient_values, "mean_abs_correction": mean_delta,
            "mean_gate": mean_gate, "mean_null_weight": mean_null, "mean_attention_entropy": mean_entropy,
            "relation_readout_entropy": beta_entropy, "relation_readout_max_weight": beta_max,
            "rng": epoch_rng, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        }
        log_jsonl(log_file, record)

        if executed % 10 == 0 or executed == CFG["base"]["max_epochs"] or wait > CFG["base"]["early_stop_wait_gt"]:
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": frozen["source_sha256"], "identity": identity,
                "warmup_sha256": identity["warmup_sha256"], "current": bundle(state, correction, optimizer),
                "rng": rng.pack_rng(), "next_epoch": executed, "wait": wait, "best_acc": best_acc,
                "best_epoch": best_epoch, "best": best, "trace": trace,
            })
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
        del train_out, eval_out
        if wait > CFG["base"]["early_stop_wait_gt"]:
            break

    if best is None:
        raise RuntimeError(f"no best checkpoint selected for {group} fold {fold}")
    task = {
        "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": frozen["source_sha256"], "identity": identity,
        "warmup_sha256": identity["warmup_sha256"], "initial_module_sha256": initial_module_sha,
        "best_checkpoint_sha256": sha_file(best_file), "latest_checkpoint_sha256": sha_file(latest_file),
        "best_epoch_1based": best_epoch + 1, "epochs_executed": executed, "metrics": best["metrics"],
        "base_metrics": best["base_metrics"], "rng_trace": trace, "wall_seconds": time.monotonic() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()), "device": torch.cuda.get_device_name(0),
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
    return task


def _task_complete(group, fold, frozen_sha):
    task_file, best_file, latest_file, _ = paths(group, fold)
    if not (task_file.is_file() and best_file.is_file() and latest_file.is_file()):
        return False
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == frozen_sha and task.get("best_checkpoint_sha256") == sha_file(best_file)
    except Exception:
        return False


def _safe_worker_limit(preflight):
    free, _ = torch.cuda.mem_get_info()
    peak = max(int(preflight.get("peak_gpu_bytes_observed", 0)), 1)
    # Include a large safety factor for independent loader copies and any
    # external jobs; cap at the requested ten workers.
    capacity = int((free * 0.70) // max(peak * 2, 1))
    return max(1, min(10, capacity))


def run_workers(max_workers=None):
    frozen = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if max_workers is None:
        max_workers = _safe_worker_limit(frozen)
    max_workers = max(1, min(10, int(max_workers)))
    pending = [(group, fold) for fold in range(1, 11) for group in GROUPS
               if not _task_complete(group, fold, frozen["source_sha256"])]
    active, failures, attempts = {}, [], {}
    started = time.monotonic()
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            attempts[(group, fold)] = attempts.get((group, fold), 0) + 1
            logfile = LOGS / f"dispatch_{group}_f{fold:02d}_attempt{attempts[(group, fold)]}.log"
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            stream = logfile.open("w", encoding="utf-8")
            proc = subprocess.Popen([sys.executable, str(HERE / "run.py"), "--worker", "--group", group, "--fold", str(fold)],
                                    cwd=str(HERE), env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[proc] = (group, fold, stream, logfile)
        for proc, (group, fold, stream, logfile) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            del active[proc]
            if code == 0:
                continue
            tail = logfile.read_text(encoding="utf-8", errors="replace")[-5000:]
            event = {"group": group, "fold": fold, "attempt": attempts[(group, fold)],
                     "exit_code": code, "log": str(logfile), "tail": tail}
            if attempts[(group, fold)] < 2:
                failures.append(event | {"classification": "retry"})
                pending.append((group, fold))
            else:
                failures.append(event | {"classification": "terminal"})
        atomic_json(RUNTIME / "scheduler.json", {
            "started_wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "worker_limit": max_workers,
            "completed_tasks": sum(_task_complete(g, f, frozen["source_sha256"]) for g in GROUPS for f in range(1, 11)),
            "active": [{"group": value[0], "fold": value[1]} for value in active.values()],
            "pending": len(pending), "failures": failures,
        })
        if active or pending:
            time.sleep(0.5)
    complete = sum(_task_complete(g, f, frozen["source_sha256"]) for g in GROUPS for f in range(1, 11))
    summary = {"completed_tasks": int(complete), "expected_tasks": 20,
               "wall_seconds": time.monotonic() - started, "failures": failures, "max_workers": max_workers,
               "peak_preflight_bytes": int(frozen.get("peak_gpu_bytes_observed", 0))}
    atomic_json(RUNTIME / "training_summary.json", summary)
    if complete != 20:
        raise RuntimeError(f"formal training incomplete: {complete}/20 tasks")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        train_worker(args.group, args.fold)
        return
    preflight_result = preflight()
    atomic_json(HERE / "manifest.json", {
        "status": "PREFLIGHT_PASS", "source_sha256": preflight_result["source_sha256"],
        "source_map": preflight_result["source_map"], "preflight_sha256": sha_file(RESULTS / "preflight.json"),
        "formal_tasks_expected": 20, "completed_tasks": 0,
        "gpu_free_bytes_at_preflight": preflight_result["free_memory_bytes_at_start"],
        "preflight_peak_gpu_bytes": preflight_result["peak_gpu_bytes_observed"],
    })
    if args.preflight_only:
        print("PREFLIGHT PASS")
        return
    workers = args.workers if args.workers is not None else _safe_worker_limit(preflight_result)
    training = run_workers(workers)
    audit = subprocess.run([sys.executable, str(HERE / "audit.py")], cwd=str(HERE), check=False)
    if audit.returncode != 0:
        raise RuntimeError(f"independent audit failed with exit code {audit.returncode}")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "COMPLETE_AUDITED", "completed_tasks": training["completed_tasks"],
                     "independent_audits": 20, "training_wall_seconds": training["wall_seconds"],
                     "worker_limit": training["max_workers"], "audit_status": "PASS"})
    atomic_json(HERE / "manifest.json", manifest)
    print("COMPLETE_AUDITED")


if __name__ == "__main__":
    main()
