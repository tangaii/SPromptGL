#!/usr/bin/env python3
"""Experiment 81.2 runner with one-pass OutputLayer correction hooks."""
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

from relation_correction import LocalCorrection

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
LOGS = RUNTIME / "logs"
GROUPS = tuple(CFG["groups"])
BASE_GROUP = "E74.4-G2-Positive-DCR"
GROUP_MODE = {GROUPS[0]: "self", GROUPS[1]: "cross"}
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
        raise IOError(f"Checkpoint write failed: {path}")


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
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime():
    global BASE, PATH_ADAPTER
    if BASE is None:
        adapter_path = HERE / "path_adapter.py"
        spec = importlib.util.spec_from_file_location("experiment81_2_path_adapter", adapter_path)
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
    paths = [HERE / name for name in ("config.json", "design.md", "path_adapter.py", "relation_correction.py", "run.py", "audit.py")]
    paths.extend(Path(path) for path in base.source_manifest())
    e81 = Path("/model/spromptgl/experiment/experiment81_prcl")
    e81_1 = Path("/model/spromptgl/experiment/experiment81_1_prcl_v2")
    for root, names in (
        (e81, ("prcl.py", "run.py", "config.json", "manifest.json", "results/report.md", "results/fold_metrics.csv", "results/audit_status.json")),
        (e81_1, ("prcl_v2.py", "run.py", "config.json", "manifest.json", "results/report.md", "results/fold_metrics.csv", "results/mechanism.csv", "results/audit_status.json")),
    ):
        paths.extend(root / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("frozen source provenance missing: " + ", ".join(missing[:8]))
    unique = sorted({path.resolve() for path in paths})
    return {str(path): sha_file(path) for path in unique}


def source_sha():
    return sha_value(source_map())


def prepare(fold):
    base = runtime()
    state, warmup, identity = base.prepare(fold)
    state["fold"] = fold
    state["rudag_identity"] = identity
    expected_overlap = [377, 407] if fold == 2 else []
    actual_overlap = identity["membership"]["train_test_overlap_1based"]
    if actual_overlap != expected_overlap:
        raise RuntimeError(f"fold {fold}: unexpected train/test overlap {actual_overlap}")
    if tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError(f"fold {fold}: expected 4096 inherited training relation pairs")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all():
        raise RuntimeError(f"fold {fold}: inherited relation pair contains non-training row")
    if not hasattr(state["cam"], "Outputlayer"):
        raise RuntimeError("inherited CAM has no original Outputlayer entry point")
    return state, warmup, identity


def paths(group, fold):
    token = f"{group}_f{fold:02d}"
    return (TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt",
            CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl")


def _stats(out):
    return {key: float(value.detach().cpu()) for key, value in out.items()
            if key in ("cls", "graph", "pair", "adv", "preserve", "total") and torch.is_tensor(value)}


def _overlap_rows(fold):
    return (376, 406) if fold == 2 else ()


def forward_with_correction(state, correction, group, training=False, intervention="full"):
    correction.train(training)
    capture = {"calls": 0, "z_cam": None, "base_logp": None, "details": None, "hidden": None}
    layer = state["cam"].Outputlayer

    def pre_hook(_layer, args):
        if not args or not torch.is_tensor(args[0]) or tuple(args[0].shape) != (871, 144):
            raise RuntimeError("original OutputLayer input is not CAM144 [871,144]")
        capture["z_cam"] = args[0]

    def output_hook(_layer, _args, output):
        capture["calls"] += 1
        if capture["calls"] != 1 or not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("original OutputLayer must run exactly once and return (logp, hidden)")
        logp_base, hidden = output
        logp_final, details = correction(
            capture["z_cam"], logp_base, GROUP_MODE[group], training=training,
            overlap_rows=_overlap_rows(state["fold"]), intervention=intervention,
        )
        capture["base_logp"] = logp_base
        capture["hidden"] = hidden
        capture["details"] = details
        return logp_final, hidden

    h_pre = layer.register_forward_pre_hook(pre_hook)
    h_out = layer.register_forward_hook(output_hook)
    try:
        out = runtime().forward(state, BASE_GROUP, training=training)
    finally:
        h_pre.remove()
        h_out.remove()
    if capture["calls"] != 1:
        raise RuntimeError(f"OutputLayer call count was {capture['calls']}, expected one")
    out["logp_base"] = capture["base_logp"]
    out["z_cam"] = capture["z_cam"]
    out["correction_details"] = capture["details"]
    out["output_hidden"] = capture["hidden"]
    return out


def correction_loss(out, state):
    details = out["correction_details"]
    train = state["train"].to(device=details["delta_margin"].device, dtype=torch.long)
    delta = details["delta_margin"]
    return delta[train].square().mean()


def total_loss(out, state):
    utils = base_runtime().runtime()["utils"]
    base_cls = utils.ClsLoss(out["logp_base"], state["y"], state["train"], state["weights"])
    combined = base_cls + 0.75 * (out["cls"] - base_cls)
    corr = correction_loss(out, state)
    total = combined + out["graph"] + 0.05 * out["pair"] + 0.01 * out["adv"] + 0.01 * out["preserve"] + 0.001 * corr
    out.update({"cls_final": out["cls"], "cls_base": base_cls, "cls_combined": combined,
                "corr": corr, "total": total})
    return total


def _finite_gradients(module):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters())


def grad_norm(parameters):
    pieces = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
    return float(torch.sqrt(torch.stack(pieces).sum()).cpu()) if pieces else 0.0


def grad_snapshot(state, correction):
    return {
        "gfr": grad_norm(state["model"].parameters()),
        "cam": grad_norm(state["cam"].parameters()),
        "local_proj": grad_norm(list(correction.local_proj.parameters()) + list(correction.local_norm.parameters())),
        "key": grad_norm(list(correction.key_proj.parameters()) + list(correction.key_norm.parameters())),
        "value": grad_norm(correction.value_proj.parameters()),
        "query": grad_norm(list(correction.query_proj.parameters()) + list(correction.query_norm.parameters())),
        "null_key": grad_norm([correction.null_key]),
        "gate": grad_norm(correction.gate.parameters()),
        "correction_head": grad_norm(correction.correction_head.parameters()),
    }


def _cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def bundle(state, correction, optimizer):
    return {
        "base": runtime().load_history().state_bundle(state),
        "correction": _cpu_state(correction),
        "optimizer_correction": copy.deepcopy(optimizer.state_dict()),
    }


def load_bundle(state, correction, optimizer, saved):
    hist = runtime().load_history()
    hist.load_bundle(state, saved["base"])
    correction.load_state_dict(saved["correction"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_correction"]))


def _metric_copy(metrics):
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _optimizer_identity(state):
    return sha_value({key: state[key].state_dict() for key in ("opt_base", "opt_cam", "opt_added")})


def _validate_optimizer_settings(state, optimizer):
    expected = (("opt_base", 0.11, 0.0038), ("opt_cam", 0.11, 0.0038), ("opt_added", 0.01, 0.001))
    for key, wd, lr in expected:
        for group in state[key].param_groups:
            if abs(float(group["weight_decay"]) - wd) > 1e-12 or abs(float(group["lr"]) - lr) > 1e-12:
                raise RuntimeError(f"historical optimizer mismatch: {key}")
    for group in optimizer.param_groups:
        if abs(float(group["weight_decay"]) - 0.01) > 1e-12 or abs(float(group["lr"]) - 0.0005) > 1e-12:
            raise RuntimeError("correction AdamW settings mismatch")


def _base_metric_fn():
    return base_runtime().classification_metrics


def _rng_runtime():
    return base_runtime()


def _base_forward(state, training):
    return runtime().forward(state, BASE_GROUP, training=training)


def _module_params(module):
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


def _tree_close(left, right, atol=1e-7, rtol=1e-7, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"parity mismatch at {where}: tensor metadata")
        if left.is_floating_point() or left.is_complex():
            if not torch.allclose(left, right, atol=atol, rtol=rtol):
                difference = float((left - right).abs().max())
                raise RuntimeError(f"parity mismatch at {where}: max_abs={difference}")
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
    result = {}
    for key in ("model", "cam"):
        result[key] = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in state[key].named_parameters()
        }
    return result


def _compare_gradients(left, right):
    _tree_close(left, right, atol=1e-7, rtol=1e-7, where="gradients")


def _forward_details(correction, group, z, logp, intervention="full"):
    return correction(z, logp, GROUP_MODE[group], training=False, intervention=intervention)


def _base_loss_for(out, state):
    return total_loss(out, state)


def _one_step(state, correction, optimizer, group, with_correction):
    hist = runtime().load_history()
    hist.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    if with_correction:
        out = forward_with_correction(state, correction, group, training=True)
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


def _replay_step(state, correction, optimizer, group):
    hist = runtime().load_history()
    hist.zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    out = forward_with_correction(state, correction, group, training=True)
    loss = total_loss(out, state)
    loss.backward()
    if not _finite_module_grads(state["model"]) or not _finite_module_grads(state["cam"]) or not _finite_module_grads(correction):
        raise RuntimeError("recovery replay step produced non-finite gradients")
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    result = {
        "loss": float(loss.detach().cpu()),
        "logp": out["logp"].detach().cpu().clone(),
        "attention_mean_by_head": out["correction_details"]["attention"].detach().mean(dim=1).cpu(),
        "relation_summary": out["correction_details"]["relation_summary"].detach().cpu(),
        "state": bundle(state, correction, optimizer),
        "rng": _rng_runtime().pack_rng(),
    }
    del out
    return result


def module_preflight():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(810201)
    z = torch.randn((3, 144), generator=generator)
    logp = torch.log_softmax(torch.randn((3, 2), generator=generator), dim=-1)
    init = LocalCorrection.initialized(812345, torch.device("cpu"))
    init_state = _cpu_state(init)
    g1, d1 = _forward_details(init, GROUPS[0], z, logp)
    g2, d2 = _forward_details(init, GROUPS[1], z, logp)
    if not torch.equal(g1, logp) or not torch.equal(g2, logp):
        raise RuntimeError("zero-initialized correction does not preserve the original logp exactly")
    if d1["attention"].shape != (3, 8, 16, 16) or d1["messages"].shape != (3, 8, 16, 32):
        raise RuntimeError("local relation tensor shape mismatch")
    if init.receiver_index.numel() != 240 or init.sender_index.numel() != 240:
        raise RuntimeError("directed relation index count must be 240")
    if tuple(init.undirected_pairs.shape) != (2, 120):
        raise RuntimeError("undirected candidate count must be 120")
    if not bool((init.candidate_map.diag() == -1).all()) or int(init.candidate_map.ge(0).sum()) != 240:
        raise RuntimeError("relation index map contains a self-edge or wrong candidate count")

    null_logp, null_details = _forward_details(init, GROUPS[1], z, logp, intervention="null_only")
    if not torch.equal(null_logp, logp) or not torch.equal(null_details["messages"], torch.zeros_like(null_details["messages"])):
        raise RuntimeError("Null-only did not yield strict zero correction/message")

    changed = z.clone()
    changed[0, 2 * 9:3 * 9] += 3.0
    _, self_before = _forward_details(init, GROUPS[0], z, logp)
    _, self_after = _forward_details(init, GROUPS[0], changed, logp)
    unaffected = torch.arange(16) != 2
    if not torch.equal(self_before["messages"][0, :, unaffected], self_after["messages"][0, :, unaffected]):
        raise RuntimeError("G1 self-local message reads a different local token")
    _, cross_before = _forward_details(init, GROUPS[1], z, logp)
    _, cross_after = _forward_details(init, GROUPS[1], changed, logp)
    cross_change = float((cross_before["messages"][0, :, unaffected] - cross_after["messages"][0, :, unaffected]).abs().max().detach())
    if cross_change <= 1e-8:
        raise RuntimeError("G2 cross-local messages did not respond to a changed sender token")
    return {
        "status": "PASS", "input_shape": [3, 144], "token_shape": [3, 16, 9],
        "local_shape": [3, 16, 32], "undirected_pairs": int(init.undirected_pairs.shape[1]), "directed_messages": 240,
        "attention_shape": list(d1["attention"].shape), "messages_shape": list(d1["messages"].shape),
        "no_self_edges": bool((init.candidate_map.diag() == -1).all()),
        "initial_zero_correction_exact": True, "null_only_exact_identity": True,
        "g1_unaffected_local_message_max_abs": 0.0, "g2_sender_change_max_abs": cross_change,
        "trainable_parameter_count": sum(p.numel() for p in init.parameters()),
        "module_init_sha": sha_value(init_state),
    }


def preflight():
    for path in (RESULTS, RUNTIME, CHECKPOINTS, TASKS, LOGS):
        path.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"one CUDA-compatible PPU is required, available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    base = runtime()
    hist = base.load_history()
    rng_runtime = _rng_runtime()
    device_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    structure = module_preflight()
    folds = []
    all_off_parity = []

    for fold in range(1, 11):
        state, warmup, identity = prepare(fold)
        if rng_runtime.rng_sha(rng_runtime.pack_rng()) != warmup["rng_sha"]:
            raise RuntimeError(f"fold {fold} did not restore the exact E61 epoch-10 RNG")
        initial_bundle = hist.state_bundle(state)
        initial_rng = rng_runtime.pack_rng()
        initial_rng_sha = rng_runtime.rng_sha(initial_rng)
        module_states = []
        optimizers = []
        for group in GROUPS:
            before_init = rng_runtime.rng_sha(rng_runtime.pack_rng())
            correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
            optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
            after_init = rng_runtime.rng_sha(rng_runtime.pack_rng())
            if before_init != after_init:
                raise RuntimeError("private correction initialization consumed public RNG")
            module_states.append(_cpu_state(correction))
            optimizers.append(optimizer)
        _tree_close(module_states[0], module_states[1], atol=0.0, rtol=0.0, where=f"fold{fold}.group_init")
        del optimizers

        base_calls = {"count": 0}
        hook_handle = state["cam"].Outputlayer.register_forward_hook(lambda *_: base_calls.__setitem__("count", base_calls["count"] + 1))
        base_out = _base_forward(state, training=False)
        hook_handle.remove()
        if base_calls["count"] != 1:
            raise RuntimeError(f"fold {fold} original OutputLayer call count was {base_calls['count']}")
        plain_logp = base_out["logp"].detach().clone()
        plain_losses = _stats(base_out)
        rng_after_plain = rng_runtime.pack_rng()

        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
        corr_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
        _validate_optimizer_settings(state, corr_optimizer)
        corrected_out = forward_with_correction(state, correction, GROUPS[0], training=False)
        if not torch.equal(corrected_out["logp"], plain_logp):
            raise RuntimeError(f"fold {fold} zero correction changed logp")
        if not torch.equal(corrected_out["output_hidden"], corrected_out["hidden"]):
            raise RuntimeError("OutputLayer hidden representation was not preserved")
        corrected_losses = _stats(corrected_out)
        if plain_losses.keys() != corrected_losses.keys():
            raise RuntimeError("base loss keys changed under zero correction")
        for key in plain_losses:
            if abs(plain_losses[key] - corrected_losses[key]) > 1e-7:
                raise RuntimeError(f"fold {fold} zero-correction loss mismatch: {key}")
        if rng_runtime.rng_sha(rng_runtime.pack_rng()) != rng_runtime.rng_sha(rng_after_plain):
            raise RuntimeError("eval hook altered public RNG consumption")

        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        base_step, base_loss, base_gradients, _, base_updated, base_rng = _one_step(
            state, correction, corr_optimizer, GROUPS[0], with_correction=False
        )
        base_logp = base_step["logp"].detach().cpu().clone()

        hist.load_bundle(state, initial_bundle)
        rng_runtime.restore_rng(initial_rng)
        correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
        corr_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
        corrected_step, corrected_loss, corrected_gradients, grad_summary, corrected_updated, corrected_rng = _one_step(
            state, correction, corr_optimizer, GROUPS[0], with_correction=True
        )
        if not torch.equal(corrected_step["logp"].detach().cpu(), base_logp):
            raise RuntimeError(f"fold {fold} train-time zero correction changed the original logp")
        if abs(base_loss - corrected_loss) > 1e-7:
            raise RuntimeError(f"fold {fold} base-off loss mismatch: {base_loss} vs {corrected_loss}")
        _compare_gradients(base_gradients, corrected_gradients)
        _tree_close(base_updated, corrected_updated, atol=1e-7, rtol=1e-7, where=f"fold{fold}.base_update")
        if rng_runtime.rng_sha(base_rng) != rng_runtime.rng_sha(corrected_rng):
            raise RuntimeError(f"fold {fold} base-off public RNG mismatch")
        if grad_summary["correction_head"] <= 0:
            raise RuntimeError(f"fold {fold} zero-initialized correction head received no classification gradient")

        upstream_gradient = None
        recovery_replay = None
        if fold == 1:
            hist.zero_optimizers(state)
            corr_optimizer.zero_grad(set_to_none=True)
            second = forward_with_correction(state, correction, GROUPS[0], training=True)
            total_loss(second, state).backward()
            upstream_gradient = grad_snapshot(state, correction)
            if min(upstream_gradient[key] for key in ("local_proj", "key", "value", "query", "gate")) <= 0:
                raise RuntimeError(f"corrector upstream did not receive gradient after head update: {upstream_gradient}")

            hist.load_bundle(state, initial_bundle)
            rng_runtime.restore_rng(initial_rng)
            correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
            corr_optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
            _replay_step(state, correction, corr_optimizer, GROUPS[1])
            after_first = bundle(state, correction, corr_optimizer)
            rng_after_first = rng_runtime.pack_rng()
            uninterrupted_second = _replay_step(state, correction, corr_optimizer, GROUPS[1])
            hist.load_bundle(state, after_first["base"])
            correction.load_state_dict(after_first["correction"], strict=True)
            corr_optimizer.load_state_dict(copy.deepcopy(after_first["optimizer_correction"]))
            rng_runtime.restore_rng(rng_after_first)
            resumed_second = _replay_step(state, correction, corr_optimizer, GROUPS[1])
            _tree_close(uninterrupted_second["state"], resumed_second["state"], atol=0.0, rtol=0.0,
                        where="two_step_recovery")
            if (uninterrupted_second["loss"] != resumed_second["loss"] or
                    not torch.equal(uninterrupted_second["logp"], resumed_second["logp"]) or
                    not torch.equal(uninterrupted_second["attention_mean_by_head"], resumed_second["attention_mean_by_head"]) or
                    not torch.equal(uninterrupted_second["relation_summary"], resumed_second["relation_summary"]) or
                    rng_runtime.rng_sha(uninterrupted_second["rng"]) != rng_runtime.rng_sha(resumed_second["rng"])):
                raise RuntimeError("two-step interruption/recovery replay diverged")
            recovery_replay = "PASS"
            hist.load_bundle(state, initial_bundle)
            rng_runtime.restore_rng(initial_rng)

        folds.append({
            "fold": fold,
            "warmup_sha256": identity["warmup_sha256"],
            "warmup_rng_sha": warmup["rng_sha"],
            "base_model_cam_sha": identity["model_cam_sha"],
            "base_optimizer_sha": identity["optimizer_sha"],
            "split_sha": identity["split_sha"],
            "pair_sha": identity["pair_sha"],
            "pair_count": int(state["train_pairs_np"].shape[0]),
            "train_test_overlap_1based": identity["membership"]["train_test_overlap_1based"],
            "public_rng_unchanged_by_init": initial_rng_sha == rng_runtime.rng_sha(initial_rng),
            "group_parameter_init_equal": True,
            "outputlayer_single_call": True,
            "zero_correction_logp_exact": True,
            "base_off_loss_grad_update_rng_parity": True,
            "correction_head_first_step_grad_norm": grad_summary["correction_head"],
            "post_update_relation_gradients": upstream_gradient,
            "two_step_interruption_recovery_replay": recovery_replay,
        })
        all_off_parity.append(True)
        del base_out, corrected_out, base_step, corrected_step, state, correction, corr_optimizer
        torch.cuda.empty_cache()

    historical_e81_1 = Path("/model/spromptgl/experiment/experiment81_1_prcl_v2/results/report.md")
    result = {
        "status": "PASS",
        "device": device_name,
        "free_memory_bytes_at_start": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
        "source_map": source_map(),
        "source_sha256": source_sha(),
        "module_tests": structure,
        "folds": folds,
        "base_off_parity": {"passed": sum(all_off_parity), "expected": 10},
        "reference_e81_1_report_sha256": sha_file(historical_e81_1),
        "outputlayer_calls_per_forward": 1,
        "formal_training_started": False,
    }
    atomic_json(RESULTS / "preflight.json", result)
    return result


def _finite_module_grads(module):
    return _finite_gradients(module)


def train_worker(group, fold):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task {group} fold={fold}")
    preflight_path = RESULTS / "preflight.json"
    frozen = json.loads(preflight_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "PASS" or source_sha() != frozen.get("source_sha256"):
        raise RuntimeError("training sources differ from passing preflight SHA")
    task_file, best_file, latest_file, log_file = paths(group, fold)
    state, warmup, identity = prepare(fold)
    state["group"] = group
    correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
    _validate_optimizer_settings(state, optimizer)
    initial_module_sha = sha_value(_cpu_state(correction))
    if task_file.exists():
        previous = json.loads(task_file.read_text(encoding="utf-8"))
        if previous.get("status") == "COMPLETE" and previous.get("source_sha256") == frozen["source_sha256"]:
            raise RuntimeError(f"task is already complete; refusing overwrite: {task_file}")
        raise RuntimeError(f"incompatible task ledger exists: {task_file}")

    hist = runtime().load_history()
    rng = _rng_runtime()
    if rng.rng_sha(rng.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not begin at its E61 epoch-10 RNG")
    start_epoch, wait = 10, 0
    best_acc, best_epoch, best = -1.0, -1, None
    trace = []
    if latest_file.exists():
        saved = torch_load(latest_file)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", frozen["source_sha256"]),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest state is incompatible at {key}")
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
            raise FloatingPointError(f"non-finite gradients at {group} fold {fold} epoch {epoch + 1}")
        gradient_values = grad_snapshot(state, correction)
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_after_train = rng.rng_sha(rng.pack_rng())
        executed = epoch + 1
        with torch.no_grad():
            eval_out = forward_with_correction(state, correction, group, training=False)
            metrics_fn = _base_metric_fn()
            final_metrics = metrics_fn(eval_out["logp"], state["test"], state["y"])
            base_metrics = metrics_fn(eval_out["logp_base"], state["test"], state["y"])
            details = eval_out["correction_details"]
            test_tensor = torch.as_tensor(state["test"], dtype=torch.long, device=details["delta_margin"].device)
            avg_delta = float(details["delta_margin"][test_tensor].abs().mean().cpu())
            avg_unc = float(details["g_unc"][test_tensor].mean().cpu())
            avg_rel = float(details["g_rel"][test_tensor].mean().cpu())
        rng_after_eval = rng.rng_sha(rng.pack_rng())
        epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_rng)

        if float(final_metrics["acc"]) > best_acc:
            best_acc, best_epoch, wait = float(final_metrics["acc"]), epoch, 0
            best = {
                "bundle": bundle(state, correction, optimizer),
                "metrics": _metric_copy(final_metrics),
                "base_metrics": _metric_copy(base_metrics),
                "test_probability": final_metrics["prob"].copy(),
                "test_prediction": final_metrics["pred"].copy(),
                "base_probability": base_metrics["prob"].copy(),
                "base_prediction": base_metrics["pred"].copy(),
                "losses": _stats(train_out) | {"cls_base": float(train_out["cls_base"].detach().cpu()),
                                               "cls_final": float(train_out["cls_final"].detach().cpu()),
                                               "cls_combined": float(train_out["cls_combined"].detach().cpu()),
                                               "corr": float(train_out["corr"].detach().cpu())},
                "rng_after_train": rng_after_train,
                "rng_after_eval": rng_after_eval,
                "mean_abs_delta": avg_delta,
                "mean_g_unc": avg_unc,
                "mean_g_rel": avg_rel,
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
            "group": group, "fold": fold, "epoch": epoch + 1,
            "best_epoch_1based": best_epoch + 1, "best_acc": best_acc,
            "metrics": _metric_copy(final_metrics), "base_metrics": _metric_copy(base_metrics),
            "train_count": int(len(state["train_np"])), "test_count": int(len(state["test_np"])),
            "losses": {"cls_final": float(train_out["cls_final"].detach().cpu()),
                       "cls_base": float(train_out["cls_base"].detach().cpu()),
                       "cls_combined": float(train_out["cls_combined"].detach().cpu()),
                       "graph": float(train_out["graph"].detach().cpu()),
                       "pair": float(train_out["pair"].detach().cpu()),
                       "adv": float(train_out["adv"].detach().cpu()),
                       "preserve": float(train_out["preserve"].detach().cpu()),
                       "corr": float(train_out["corr"].detach().cpu()),
                       "total": float(loss.detach().cpu())},
            "gradient_norms": gradient_values,
            "mean_abs_correction": avg_delta, "mean_g_unc": avg_unc, "mean_g_rel": avg_rel,
            "rng": epoch_rng,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        }
        log_jsonl(log_file, record)

        if executed % 10 == 0 or executed == CFG["base"]["max_epochs"] or wait > CFG["base"]["early_stop_wait_gt"]:
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": frozen["source_sha256"],
                "identity": identity, "warmup_sha256": identity["warmup_sha256"],
                "current": bundle(state, correction, optimizer), "rng": rng.pack_rng(),
                "next_epoch": executed, "wait": wait, "best_acc": best_acc,
                "best_epoch": best_epoch, "best": best, "trace": trace,
            })
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best_epoch + 1} wait={wait}", flush=True)
        del train_out, eval_out
        if wait > CFG["base"]["early_stop_wait_gt"]:
            break

    if best is None:
        raise RuntimeError(f"no best checkpoint was selected for {group} fold {fold}")
    task = {
        "status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
        "source_sha256": frozen["source_sha256"], "identity": identity,
        "warmup_sha256": identity["warmup_sha256"], "initial_module_sha256": initial_module_sha,
        "best_checkpoint_sha256": sha_file(best_file), "latest_checkpoint_sha256": sha_file(latest_file),
        "best_epoch_1based": best_epoch + 1, "epochs_executed": executed,
        "metrics": best["metrics"], "base_metrics": best["base_metrics"],
        "rng_trace": trace, "wall_seconds": time.monotonic() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(0),
    }
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch + 1}", flush=True)
    return task


def run_workers(max_workers=10):
    pending = [(group, fold) for fold in range(1, 11) for group in GROUPS]
    started = time.monotonic()
    active = {}
    failures = []
    attempts = {}
    while pending or active:
        while pending and len(active) < max_workers:
            group, fold = pending.pop(0)
            key = (group, fold)
            attempts[key] = attempts.get(key, 0) + 1
            logfile = LOGS / f"dispatch_{group}_f{fold:02d}_attempt{attempts[key]}.log"
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
            "completed_tasks": sum(1 for group in GROUPS for fold in range(1, 11) if paths(group, fold)[0].exists()),
            "active": [{"group": value[0], "fold": value[1]} for value in active.values()],
            "pending": len(pending), "failures": failures,
        })
        if not active and not pending:
            break
        time.sleep(0.5)
    elapsed = time.monotonic() - started
    complete = 0
    for group in GROUPS:
        for fold in range(1, 11):
            task_file, best_file, latest_file, _ = paths(group, fold)
            if task_file.is_file() and best_file.is_file() and latest_file.is_file():
                task = json.loads(task_file.read_text(encoding="utf-8"))
                complete += task.get("status") == "COMPLETE" and task.get("best_checkpoint_sha256") == sha_file(best_file)
    summary = {"completed_tasks": int(complete), "expected_tasks": 20, "wall_seconds": elapsed,
               "failures": failures, "max_workers": max_workers}
    atomic_json(RUNTIME / "training_summary.json", summary)
    if complete != 20:
        raise RuntimeError(f"formal training incomplete: {complete}/20 tasks")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--group")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        train_worker(args.group, args.fold)
        return
    if args.preflight_only:
        result = preflight()
        manifest = {"status": "PREFLIGHT_PASS", "source_sha256": result["source_sha256"],
                    "source_map": result["source_map"], "preflight_sha256": sha_file(RESULTS / "preflight.json")}
        atomic_json(HERE / "manifest.json", manifest)
        print("PREFLIGHT PASS")
        return
    preflight_result = preflight()
    atomic_json(HERE / "manifest.json", {
        "status": "PREFLIGHT_PASS", "source_sha256": preflight_result["source_sha256"],
        "source_map": preflight_result["source_map"], "preflight_sha256": sha_file(RESULTS / "preflight.json"),
        "formal_tasks_expected": 20, "completed_tasks": 0,
    })
    training = run_workers(max_workers=max(1, min(10, args.workers)))
    audit = subprocess.run([sys.executable, str(HERE / "audit.py")], cwd=str(HERE), check=False)
    if audit.returncode != 0:
        raise RuntimeError(f"independent audit failed with exit code {audit.returncode}")
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"status": "COMPLETE_AUDITED", "completed_tasks": training["completed_tasks"],
                     "independent_audits": 20, "training_wall_seconds": training["wall_seconds"],
                     "audit_status": "PASS"})
    atomic_json(HERE / "manifest.json", manifest)
    print("COMPLETE_AUDITED")


if __name__ == "__main__":
    main()
