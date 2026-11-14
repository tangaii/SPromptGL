#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from path_adapter import HERE, BASE, REPO, load_base_runner, sha_file, verify_relocation
from fapc import FAPC, initialize

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GROUPS = tuple(CFG["groups"])
BASE_GROUP = CFG["base_group"]
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
TASKS = HERE / "runtime/tasks"
LOGS = HERE / "runtime/logs"
BASE_MODULE = None
DATA_IDENTITY = None
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")


def base():
    global BASE_MODULE
    if BASE_MODULE is None:
        BASE_MODULE = load_base_runner()
    return BASE_MODULE


def hist():
    return base().load_history()


def runtime():
    return hist().load_e69().load_e61().load_base()


def sha_object(value):
    return hist().sha_object(value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def log_record(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_map():
    files = [HERE / name for name in (
        "config.json", "design.md", "fapc.py", "run.py", "audit.py",
        "path_adapter.py", "make_text_cache.py", "runtime/text_embeddings.pt",
    )]
    files += [REPO / name for name in (
        "network.py", "model.py", "layers.py", "utils.py",
        "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy",
        "CLIP/clip/clip.py", "CLIP/clip/simple_tokenizer.py", "CLIP/clip/bpe_simple_vocab_16e6.txt.gz",
    )]
    files.append(Path(CFG["text_encoder"]["weights_path"]))
    files.extend(sorted((E61 / ".runtime/warmup").glob("fold*.pt")))
    files.extend(BASE / name for name in (
        "run.py", "soft_positive_dcr.py", "config.json", "manifest.json", "audit.py",
    ))
    for fold in range(1, 11):
        token = f"{BASE_GROUP}_f{fold:02d}"
        files.extend((BASE / ".runtime/tasks" / f"{token}.json",
                      BASE / ".runtime/checkpoints" / f"{token}.best.pt",
                      BASE / ".runtime/checkpoints" / f"{token}.latest.pt"))
    files = sorted(set(files))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"required immutable source is missing: {path}")
    manifest = {str(path.resolve()): sha_file(path) for path in files}
    manifest.update({f"relocation_checked:{path}": digest for path, digest in verify_relocation().items()})
    manifest.update(base().source_manifest())
    return dict(sorted(manifest.items()))


def verify_frozen_source(expected_sha=None):
    frozen = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    current = source_map()
    digest = sha_object(current)
    expected = expected_sha or frozen.get("source_sha256")
    if digest != expected or current != frozen.get("source_map"):
        raise RuntimeError("source bytes differ from the frozen E78.8 preflight manifest")
    return frozen


def save_text_cache():
    cache = HERE / "runtime/text_embeddings.pt"
    if not cache.is_file():
        script = HERE / "make_text_cache.py"
        spec = importlib.util.spec_from_file_location("experiment78_8_text_cache", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.main()
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    if tuple(payload["anchors"].shape) != (20, 32) or tuple(payload["embeddings_raw"].shape) != (20, 1024):
        raise RuntimeError(f"text cache shape mismatch: anchors={tuple(payload['anchors'].shape)} raw={tuple(payload['embeddings_raw'].shape)}")
    if float(payload["gram_max_abs_error"]) > 2e-10:
        raise RuntimeError("frozen common text row-space Gram reconstruction failed")
    for key in ("medical_text", "anonymous_text"):
        if payload[key] != CFG["prompts"][key] or len(set(payload[key])) != 10:
            raise RuntimeError(f"text cache strings do not match the frozen unique prompts: {key}")
    for key in ("medical_classes", "anonymous_classes"):
        if payload[key].tolist() != CFG["prompts"][key]:
            raise RuntimeError(f"text cache class mapping mismatch: {key}")
    if payload["text_encoder_weights_sha256"] != CFG["text_encoder"]["weights_sha256"]:
        raise RuntimeError("text cache encoder identity mismatch")
    return payload

def prompt_tensors(device):
    payload = save_text_cache()
    med = payload["medical_anchors"].to(device)
    anon = payload["anonymous_anchors"].to(device)
    classes = payload["medical_classes"].to(device)
    return payload, med, anon, classes

def prepare(fold):
    state, warmup, identity = base().prepare(fold)
    state["fold"] = fold
    ref_task = BASE / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json"
    task = json.loads(ref_task.read_text(encoding="utf-8"))
    ref_best = BASE / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    if task.get("status") != "COMPLETE" or task.get("identity") != identity:
        raise RuntimeError(f"fold {fold} does not match completed E74.4-G2 identity")
    if task.get("checkpoint_sha256") != sha_file(ref_best):
        raise RuntimeError(f"fold {fold} E74.4-G2 best SHA differs from task record")
    return state, warmup, identity


def task_paths(group, fold):
    token = f"{group}_f{fold:02d}"
    return (TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt",
            CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl")


def support_ids(state):
    return np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False).astype(np.int64)


def _forward(state, module=None, *, training=False, support=None, query=None, intervention="full"):
    if module is None or intervention == "module_off":
        return base().forward(state, BASE_GROUP, training=training), None
    if support is None or query is None:
        raise ValueError("FAPC requires legal support and query IDs")
    captured = {"calls": 0, "h": None}
    layer = state["cam"].Outputlayer

    def capture(_layer, args):
        captured["calls"] += 1
        if captured["calls"] != 1 or not args or tuple(args[0].shape) != (871, 144):
            raise RuntimeError("OutputLayer pre-hook must capture exactly one CAM144 tensor")
        captured["h"] = args[0]
        return None

    hook = layer.register_forward_pre_hook(capture)
    try:
        out = base().forward(state, BASE_GROUP, training=training)
    finally:
        hook.remove()
    if captured["calls"] != 1 or captured["h"] is None:
        raise RuntimeError(f"CAM144 capture count={captured['calls']}")
    details = module(
        captured["h"], state["y"], support, query, state["weights"], out["logp"],
        training=training, intervention=intervention,
    )
    details["h_original"] = captured["h"]
    details["base_logp"] = out["logp"]
    return out, details


def _new_optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=CFG["training"]["new_lr"],
                             weight_decay=CFG["training"]["new_weight_decay"])


def _module_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _snapshot(state, module, optimizer):
    return {"base_state": hist().state_bundle(state), "module_state": _module_state(module),
            "new_optimizer": copy.deepcopy(optimizer.state_dict()), "rng": runtime().pack_rng()}


def _restore(state, module, optimizer, payload):
    hist().load_bundle(state, payload["base_state"])
    module.load_state_dict(payload["module_state"], strict=True)
    optimizer.load_state_dict(payload["new_optimizer"])
    runtime().restore_rng(payload["rng"])


def _metrics(logp, state, query=None):
    """Evaluate either a full 871-row log-probability bank or query rows.

    The historical evaluator indexes a full patient bank with ``state["test"]``.
    FAPC deliberately returns only the requested query rows, so reconstruct a
    temporary full bank for the evaluator without changing any model tensors or
    random state.
    """
    if query is None or int(logp.shape[0]) == 871:
        return runtime().classification_metrics(logp, state["test"], state["y"])
    query_np = (query.detach().cpu().numpy().astype(np.int64)
                if torch.is_tensor(query) else np.asarray(query, dtype=np.int64))
    if int(logp.shape[0]) != len(query_np):
        raise RuntimeError(f"query logp length {logp.shape[0]} does not match query IDs {len(query_np)}")
    full = torch.full((871, 2), float("-inf"), dtype=logp.dtype, device=logp.device)
    full[torch.as_tensor(query_np, dtype=torch.long, device=logp.device)] = logp
    return runtime().classification_metrics(full, query_np, state["y"])


def _combined_total(out, details, state, support):
    """Keep the complete DARG objective and add fixed semantic/keep auxiliaries."""
    sem = details["semantic_loss"]
    keep = details["keep_loss"]
    total = out["total"] + float(CFG["method"]["semantic_loss_weight"]) * sem + float(CFG["method"]["semantic_keep_weight"]) * keep
    return total, sem, keep

def _protected_backward(out, details, total):
    """Backprop once while protecting the DARG/CAM CAM144 path.

    The hook replaces only the gradient arriving at the captured CAM144
    tensor. Direct DARG classifier gradients and all FAPC parameter gradients
    still come from the ordinary total backward.
    """
    h = details["h_original"]
    aux = float(CFG["method"]["semantic_loss_weight"]) * details["semantic_loss"]
    aux = aux + float(CFG["method"]["semantic_keep_weight"]) * details["keep_loss"]
    gb = torch.autograd.grad(out["total"], h, retain_graph=True, allow_unused=True)[0]
    ga = torch.autograd.grad(aux, h, retain_graph=True, allow_unused=True)[0]
    if gb is None: gb = torch.zeros_like(h)
    if ga is None: ga = torch.zeros_like(h)
    gb = gb.detach(); ga = ga.detach()
    eps = float(CFG["method"]["gradient_eps"])
    rho = float(CFG["method"]["gradient_aux_norm_max_ratio"])
    dot = torch.sum(gb * ga)
    nb = torch.linalg.vector_norm(gb); na = torch.linalg.vector_norm(ga)
    conflict = bool((dot < 0).item())
    projected = False; scaled = False
    gp = ga
    if conflict and float(nb) > eps:
        gp = ga - dot / (nb.square() + eps) * gb
        projected = True
    npg = torch.linalg.vector_norm(gp)
    if float(nb) <= eps:
        safe = torch.zeros_like(gp) if float(npg) <= eps else gp * (rho * 0.0)
    elif float(npg) > rho * nb:
        safe = gp * (rho * nb / (npg + eps)); scaled = True
    else:
        safe = gp
    total_norm = torch.linalg.vector_norm(gb + safe)
    stats = {
        "gradient_dot": float(dot.cpu()), "gradient_cosine": float((dot/(nb*na+eps)).cpu()),
        "base_norm": float(nb.cpu()), "aux_norm": float(na.cpu()),
        "projected_aux_norm": float(npg.cpu()), "safe_aux_norm": float(torch.linalg.vector_norm(safe).cpu()),
        "aux_norm_ratio_raw": float((na/(nb+eps)).cpu()), "aux_norm_ratio_safe": float((torch.linalg.vector_norm(safe)/(nb+eps)).cpu()),
        "conflict": conflict, "projected": projected, "scaled": scaled,
        "base_near_zero": bool(float(nb)<=eps), "combined_cam144_norm": float(total_norm.cpu()),
    }
    hook = h.register_hook(lambda _incoming: (gb + safe).to(dtype=_incoming.dtype))
    try:
        total.backward()
    finally:
        hook.remove()
    return stats

def _final_metrics(out, details, state):
    # Formal prediction is always the original DARG classifier.
    return _metrics(out["logp"], state) if details is None else _metrics(out["logp"], state, details["query_ids"])

def _compact_metrics(metrics):
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _grad_norm(grads):
    vals = [g.detach().double().square().sum() for g in grads if g is not None]
    return float(torch.sqrt(torch.stack(vals).sum()).cpu()) if vals else 0.0


def _all_grad_norm(module):
    return _grad_norm([p.grad for p in module.parameters()])


def _classification_gradient(details, state, module):
    module_params = [p for p in module.parameters() if p.requires_grad]
    base_params = [p for p in state["model"].parameters() if p.requires_grad]
    cam_params = [p for p in state["cam"].parameters() if p.requires_grad]
    targets = module_params + base_params + cam_params
    grads = torch.autograd.grad(details["classification_loss"], targets, retain_graph=True, allow_unused=True)
    a, b, c = len(module_params), len(base_params), len(cam_params)
    by_name = {}
    for name, params in module.named_parameters():
        if params.requires_grad:
            g = torch.autograd.grad(details["classification_loss"], params, retain_graph=True, allow_unused=True)[0]
            by_name[name] = 0.0 if g is None else float(g.detach().double().norm().cpu())
    return {
        "module": _grad_norm(grads[:a]), "darg_gfr": _grad_norm(grads[a:a + b]),
        "cam": _grad_norm(grads[a + b:a + b + c]), "module_by_name": by_name,
    }


def _scalar_losses(out, details, total):
    record = {f"base_{key}": float(out[key].detach().cpu()) for key in ("cls", "graph", "pair", "adv", "preserve", "total")}
    record["semantic_loss"] = float(details["semantic_loss"].detach().cpu())
    record["keep_loss"] = float(details["keep_loss"].detach().cpu())
    record["semantic_weight"] = float(CFG["method"]["semantic_loss_weight"])
    record["keep_weight"] = float(CFG["method"]["semantic_keep_weight"])
    record["combined_total"] = float(total.detach().cpu())
    record["gamma"] = float(details["gamma"].detach().cpu())
    record["attention_tau"] = float(details["attention_tau"].detach().cpu())
    record["enable_a"] = bool(details["enable_a"]); record["enable_b"] = bool(details["enable_b"])
    return record

def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _data_identity(state):
    frame = pd.read_csv(REPO / "data/ABIDE/processed_standard_data.csv")
    x, y = base().load_history().load_e69().load_e61().load_base().load_data()[0]
    model_x = torch.cat(state["parts"], dim=1).detach().cpu().numpy()
    if frame.shape != (871, 321) or not np.array_equal(frame.iloc[:, :-1].to_numpy(np.float32), x):
        raise RuntimeError("processed data row/value alignment differs from historical load_data")
    if not np.array_equal(frame.iloc[:, -1].to_numpy(np.int64) - 1, y):
        raise RuntimeError("processed labels differ from ASD=0, TD=1")
    if not np.array_equal(model_x, x) or not np.array_equal(state["labels_np"], y):
        raise RuntimeError("loaded feature/label order differs from processed CSV")
    site_cols = [c for c in frame.columns if "SITE_ID" in c.upper()]
    age_cols = [c for c in frame.columns if "AGE_AT_SCAN" in c.upper()]
    if len(site_cols) != 20 or len(age_cols) != 26:
        raise RuntimeError("unexpected SITE_ID/AGE_AT_SCAN one-hot width")
    site_block = frame[site_cols].to_numpy(np.float64)
    age_block = frame[age_cols].to_numpy(np.float64)
    if not np.array_equal(site_block.sum(1), np.ones(871)) or not np.array_equal(age_block.sum(1), np.ones(871)):
        raise RuntimeError("one-hot row alignment failed")
    site_ids, age_ids = site_block.argmax(1).astype(np.int64), age_block.argmax(1).astype(np.int64)
    overlap = np.intersect1d(state["train_np"], state["test_np"]).astype(np.int64)
    expected = np.asarray([376, 406], dtype=np.int64) if state["fold"] == 2 else np.zeros(0, dtype=np.int64)
    if not np.array_equal(overlap, expected):
        raise RuntimeError(f"unexpected historical train/test overlap fold {state['fold']}: {overlap.tolist()}")
    return {
        "frame": frame, "site_ids": site_ids, "age_ids": age_ids,
        "site_columns": site_cols, "age_columns": age_cols,
        "row_order_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes() + np.ascontiguousarray(y).tobytes()).hexdigest(),
        "overlap_ids_zero_based": overlap.tolist(),
        "identity_limit": "exact 871-row historical feature/label order plus SITE_ID/AGE_AT_SCAN one-hot alignment",
    }


def _baseline_update(state):
    hist().zero_optimizers(state)
    out = base().forward(state, BASE_GROUP, training=True)
    result = {"logp": out["logp"].detach().cpu().clone(),
              "losses": {k: float(out[k].detach().cpu()) for k in ("total", "cls", "graph", "pair", "adv", "preserve")}}
    out["total"].backward()
    result["gradient"] = {key: {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                 for name, p in state[key].named_parameters()} for key in ("model", "cam")}
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result.update({"bundle": hist().state_bundle(state), "rng_sha": runtime().rng_sha(runtime().pack_rng())})
    return result


def _disabled_update(state):
    hist().zero_optimizers(state)
    out, details = _forward(state, None, training=True)
    if details is not None:
        raise RuntimeError("base-off path returned FAPC details")
    result = {"logp": out["logp"].detach().cpu().clone(),
              "losses": {k: float(out[k].detach().cpu()) for k in ("total", "cls", "graph", "pair", "adv", "preserve")}}
    out["total"].backward()
    result["gradient"] = {key: {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                 for name, p in state[key].named_parameters()} for key in ("model", "cam")}
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result.update({"bundle": hist().state_bundle(state), "rng_sha": runtime().rng_sha(runtime().pack_rng())})
    return result


def _compare_tree(left, right, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise RuntimeError(f"exact tensor mismatch at {where}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise RuntimeError(f"exact array mismatch at {where}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"key mismatch at {where}")
        for key in left:
            _compare_tree(left[key], right[key], f"{where}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"length mismatch at {where}")
        for i, (a, b) in enumerate(zip(left, right)):
            _compare_tree(a, b, f"{where}[{i}]")
        return
    if isinstance(left, float) and isinstance(right, float):
        if left != right:
            raise RuntimeError(f"scalar mismatch at {where}: {left} != {right}")
        return
    if left != right:
        raise RuntimeError(f"value mismatch at {where}: {left!r} != {right!r}")


def _base_off_parity(fold):
    ref_state, _, ref_identity = prepare(fold)
    ref = _baseline_update(ref_state)
    cand_state, _, cand_identity = prepare(fold)
    cand = _disabled_update(cand_state)
    if ref_identity != cand_identity:
        raise RuntimeError(f"fold {fold} identity differs between parity paths")
    for key in ("logp", "losses", "gradient", "bundle"):
        _compare_tree(ref[key], cand[key], f"base_off.{key}")
    if ref["rng_sha"] != cand["rng_sha"]:
        raise RuntimeError(f"fold {fold} base-off RNG differs")
    del ref_state, cand_state
    torch.cuda.empty_cache()
    return {"fold": fold, "status": "PASS", "logp_exact": True, "all_losses_exact": True,
            "all_original_gradients_exact": True, "model_and_optimizer_updates_exact": True,
            "rng_exact": True, "identity": ref_identity}


def _make_module(state, group):
    payload, med, anon, classes = prompt_tensors(state["y"].device)
    spec = CFG["group_specs"][group]
    if spec["mode"] == "visual":
        anchors, prompt_classes = None, None
    else:
        anchors = med if spec["prompt_source"] == "medical" else anon
        prompt_classes = classes
    # initialize() uses a private fork_rng, so construction does not consume
    # the public E74.4 random stream.
    return initialize(anchors, prompt_classes, spec["mode"], spec["prompt_source"],
                      state["fold"], state["y"].device)

def _initialized_task(fold, group, data):
    state, warmup, identity = prepare(fold)
    rt = runtime()
    if rt.rng_sha(rt.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not restore E61 epoch-10 RNG")
    support = support_ids(state)
    if np.intersect1d(support, state["test_np"]).size:
        raise RuntimeError("test patients entered new FAPC support")
    if state["fold"] == 2 and np.intersect1d(support, np.asarray([376, 406])).size:
        raise RuntimeError("Fold 2 historical overlap entered new support")
    module = _make_module(state, group)
    optimizer = _new_optimizer(module)
    rng_before = rt.rng_sha(rt.pack_rng())
    init_record = {
        "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        "support_ids": support.tolist(), "support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
        "overlap_excluded_from_new_support": True, "module_parameter_count": sum(p.numel() for p in module.parameters()),
        "module_init_sha256": sha_object({k: v.detach().cpu() for k, v in module.named_parameters()}),
        "optimizer_init_sha256": sha_object(optimizer.state_dict()), "original_optimizer_sha256": identity["optimizer_sha"],
        "warmup_rng_sha": warmup["rng_sha"], "post_module_rng_sha": rng_before,
        "rng_unchanged_by_new_init": warmup["rng_sha"] == rng_before,
    }
    expected_count = CFG["group_specs"][group].get("expected_parameter_count")
    if expected_count is not None and int(expected_count) != init_record["module_parameter_count"]:
        raise RuntimeError(f"module parameter count {init_record['module_parameter_count']} differs from config")
    return state, warmup, identity, support, module, optimizer, init_record


def _task_complete(group, fold, frozen_sha):
    task_file, best_file, _, _ = task_paths(group, fold)
    if not task_file.is_file() or not best_file.is_file():
        return False
    try:
        task = json.loads(task_file.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == frozen_sha and task.get("checkpoint_sha256") == sha_file(best_file)
    except Exception:
        return False


def _checkpoint(group, fold, source_hash, identity, best, executed, trace, loss_history, init_record):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "identity": identity, "initialization": init_record,
        "best_epoch_1based": best["epoch"], "epochs_executed": executed,
        "metrics": best["metrics"], "base_metrics": best["base_metrics"],
        "test_probability": best["probability"], "test_prediction": best["prediction"],
        "best_state": best["state"], "best_diagnostics": best["diagnostics"],
        "best_losses": best["losses"], "best_rng": best["rng"],
        "rng_trace": trace, "loss_history": loss_history,
    }


def _finite_parameters(*modules):
    return all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
               for module in modules for p in module.parameters())


def train_task(group, fold, frozen_sha):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task {group} fold={fold}")
    if _task_complete(group, fold, frozen_sha):
        return json.loads(task_paths(group, fold)[0].read_text(encoding="utf-8"))
    state, warmup, identity, support, module, optimizer, init_record = _initialized_task(fold, group, DATA_IDENTITY)
    task_file, best_file, latest_file, log_file = task_paths(group, fold)
    if task_file.exists() and json.loads(task_file.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        raise RuntimeError(f"incompatible completed task record {task_file}")
    rt = runtime()
    start_epoch, wait, best_acc, best = 10, 0, -1.0, None
    trace, loss_history = [], []
    if latest_file.is_file():
        saved = torch.load(latest_file, map_location="cpu", weights_only=False)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", frozen_sha),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest checkpoint incompatible at {key}")
        _restore(state, module, optimizer, saved["current"])
        start_epoch = int(saved["next_epoch"]); wait = int(saved["wait"])
        best_acc = float(saved["best_acc"]); best = saved["best"]
        trace, loss_history = saved["rng_trace"], saved["loss_history"]
    started = time.time()
    executed = start_epoch
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, CFG["max_epoch"]):
        rng_before = rt.rng_sha(rt.pack_rng())
        hist().zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        module.train()
        out, details = _forward(state, module, training=True, support=support, query=support)
        total, base_support_loss, fused_support_loss = _combined_total(out, details, state, support)
        details["base_support_loss"] = base_support_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite FAPC loss group={group} fold={fold} epoch={epoch+1}")
        gradient_records = None
        if epoch == start_epoch:
            gradient_records = {"classification": _classification_gradient(details, state, module)}
        protection_stats = _protected_backward(out, details, total)
        if not _finite_parameters(state["model"], state["cam"], module):
            raise FloatingPointError(f"non-finite gradients group={group} fold={fold} epoch={epoch+1}")
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_after_train = rt.rng_sha(rt.pack_rng())
        executed = epoch + 1
        module.eval()
        with torch.no_grad():
            eval_out, eval_details = _forward(state, module, training=False, support=support, query=state["test_np"])
            metrics = _final_metrics(eval_out, eval_details, state)
            base_metrics = _metrics(eval_out["logp"], state)
        rng_after_eval = rt.rng_sha(rt.pack_rng())
        epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_rng)
        scalar_losses = _scalar_losses(out, details, total)
        loss_history.append({"epoch": epoch + 1, "metrics": _compact_metrics(metrics),
                             "base_metrics": _compact_metrics(base_metrics), "losses": scalar_losses,
                             "rng": epoch_rng, "gradient_audit_first_epoch": gradient_records,
                             "gradient_protection": protection_stats})
        log_record(log_file, loss_history[-1])
        if metrics["acc"] > best_acc:
            best_acc, wait = float(metrics["acc"]), 0
            diagnostic_keys = {
                "final_logp", "final_logits", "semantic_logits", "semantic_logp", "semantic_loss", "keep_loss",
                "prototype_similarity", "prototypes", "patient_projection", "attention_features", "patient_teacher",
                "attention_weights", "attention_logits", "attention_entropy", "prompt_base", "prompt_calibrated",
                "prompt_final", "calibration_strength", "semantic_margin", "support_ids", "support_labels", "query_ids",
                "prompt_classes", "prompt_source", "enable_a", "enable_b", "gamma", "attention_tau", "semantic_scores",
                "classification_loss", "base_logp", "h_original",
            }
            diagnostics = {key: _cpu_copy(value) for key, value in eval_details.items() if key in diagnostic_keys}
            best = {
                "epoch": epoch + 1, "metrics": _compact_metrics(metrics), "base_metrics": _compact_metrics(base_metrics),
                "probability": np.asarray(metrics["prob"]).copy(), "prediction": np.asarray(metrics["pred"]).copy(),
                "losses": scalar_losses, "diagnostics": diagnostics,
                "rng": {**epoch_rng, "packed": rt.pack_rng()},
                "state": _snapshot(state, module, optimizer),
            }
            atomic_torch(best_file, _checkpoint(group, fold, frozen_sha, identity, best, executed, trace, loss_history, init_record))
        else:
            wait += 1
        if executed % CFG["training"]["checkpoint_interval_epochs"] == 0 or wait > CFG["early_stop_wait_gt"] or executed == CFG["max_epoch"]:
            atomic_torch(latest_file, {"group": group, "fold": fold, "source_sha256": frozen_sha,
                                       "identity": identity, "warmup_sha256": identity["warmup_sha256"],
                                       "current": _snapshot(state, module, optimizer), "next_epoch": executed,
                                       "wait": wait, "best_acc": best_acc, "best": best,
                                       "rng_trace": trace, "loss_history": loss_history})
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best['epoch'] if best else 'none'} wait={wait}", flush=True)
        del out, details, eval_out, eval_details, total
        if wait > CFG["early_stop_wait_gt"]:
            break
    if best is None:
        raise RuntimeError(f"no strict-ACC best checkpoint for {group} fold={fold}")
    atomic_torch(best_file, _checkpoint(group, fold, frozen_sha, identity, best, executed, trace, loss_history, init_record))
    task = {"status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": frozen_sha, "checkpoint_sha256": sha_file(best_file), "identity": identity,
            "initialization": init_record, "best_epoch_1based": best["epoch"], "epochs_executed": executed,
            "metrics": best["metrics"], "base_metrics": best["base_metrics"], "rng_trace": trace,
            "wall_seconds": time.time() - started, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0)}
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best['epoch']}", flush=True)
    return task


def _module_preflight(fold, data):
    state, warmup, identity = prepare(fold)
    support = support_ids(state)
    rt = runtime()
    before = rt.rng_sha(rt.pack_rng())
    modules = {group: _make_module(state, group) for group in GROUPS}
    after = rt.rng_sha(rt.pack_rng())
    if before != after:
        raise RuntimeError("FAPC private initialization changed public RNG")
    records = {}
    for group, module in modules.items():
        hist().zero_optimizers(state); module.train()
        out, details = _forward(state, module, training=True, support=support, query=support)
        total, sem, keep = _combined_total(out, details, state, support)
        vals=(total,sem,keep,details["semantic_logits"],details["prototypes"])
        if not all(bool(torch.isfinite(x).all().item()) for x in vals):
            raise RuntimeError(f"non-finite FAPC loss/logits for {group}")
        if tuple(details["final_logp"].shape) != (len(support),2):
            raise RuntimeError("unexpected FAPC DARG logp shape")
        grad = _classification_gradient(details, state, module)
        if grad["module"] <= 0 or grad["darg_gfr"] <= 0 or grad["cam"] <= 0:
            raise RuntimeError(f"auxiliary branch disconnected from DARG/CAM: {group} {grad}")
        active = ["patient_projection", "prototype_direction"]
        if CFG["group_specs"][group]["mode"] == "text": active += ["calibration_raw"]
        missing=[name for name in active if not any(name in k and float(v)>0 for k,v in grad["module_by_name"].items())]
        if missing: raise RuntimeError(f"active FAPC parameters have no auxiliary gradient: {group} {missing}")
        # eta=0 must be bitwise identical to the historical DARG objective.
        old_sem=float(CFG["method"]["semantic_loss_weight"]); old_keep=float(CFG["method"]["semantic_keep_weight"])
        CFG["method"]["semantic_loss_weight"]=0.0; CFG["method"]["semantic_keep_weight"]=0.0
        zero_total,_,_= _combined_total(out,details,state,support)
        CFG["method"]["semantic_loss_weight"]=old_sem; CFG["method"]["semantic_keep_weight"]=old_keep
        if not torch.equal(zero_total,out["total"]): raise RuntimeError("auxiliary-off objective did not recover DARG total")
        # Patient and label order must not alter a full bank output; test labels
        # are never read by the module beyond the support construction.
        module.eval(); test_ids=state["test_np"].astype(np.int64)
        with torch.no_grad():
            _, d1=_forward(state,module,training=False,support=support,query=test_ids)
            _, d2=_forward(state,module,training=False,support=support,query=test_ids[::-1].copy())
            rev=torch.arange(len(test_ids)-1,-1,-1,device=d2["final_logp"].device)
            if not torch.equal(d1["final_logp"],d2["final_logp"][rev]): raise RuntimeError("query order changed DARG output")
            orig=state["y"].clone(); ti=torch.as_tensor(test_ids,device=state["y"].device); state["y"][ti]=1-state["y"][ti]
            _, d3=_forward(state,module,training=False,support=support,query=test_ids); state["y"].copy_(orig)
            if not torch.equal(d1["final_logp"],d3["final_logp"]): raise RuntimeError("test labels affected inference")
            if not torch.equal(d1["semantic_logits"],d3["semantic_logits"]): raise RuntimeError("test labels affected semantic inference")
        item={"parameter_count":sum(p.numel() for p in module.parameters()),"gradient":grad,
              "finite":True,"query_permutation_invariant":True,"test_label_invariant":True,
              "auxiliary_off_exact":True,"mode":module.mode,
              "anchor_sha256":module.anchor_sha(),
              "anchor_grad_none":all(p.grad is None for n,p in module.named_buffers()),
              "calibration_range":([float(module.calibration_strength.min().detach().cpu()),float(module.calibration_strength.max().detach().cpu())] if module.mode=='text' else []),
              "support_count":int(len(support))}
        records[group]=item
    if records[GROUPS[1]]["parameter_count"] != records[GROUPS[2]]["parameter_count"]:
        raise RuntimeError("G2/G3 parameter count mismatch")
    for name,val in modules[GROUPS[1]].named_parameters():
        if not torch.equal(val,dict(modules[GROUPS[2]].named_parameters())[name]):
            raise RuntimeError(f"G2/G3 trainable initialization differs: {name}")
    payload=prompt_tensors(torch.device("cpu"))[0]
    del state, modules; torch.cuda.empty_cache()
    return {"fold":fold,"status":"PASS","public_rng_unchanged":True,"groups":records,
            "support_count":int(len(support)),"fold2_overlap_excluded":True,
            "text_cache_sha256":sha_file(HERE/'runtime/text_embeddings.pt'),"anchor_rank":int(payload['effective_rank']),
            "gram_max_abs_error":float(payload['gram_max_abs_error'])}


def _gradient_protection_preflight(fold):
    """Independently verify actual hook gradients, including the main path.

    For the shared original parameters, the reference is the unmodified full
    DARG gradient plus a separate CAM144 vector-Jacobian product of safe aux.
    New-module parameters retain their ordinary auxiliary-loss gradients.
    """
    state, _, _ = prepare(fold)
    support = support_ids(state)
    module = _make_module(state, GROUPS[1])
    originals = [(f"{key}.{name}", p) for key in ("model", "cam")
                 for name, p in state[key].named_parameters() if p.requires_grad]
    additions = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    params = [p for _, p in originals]

    def compare(actual, expected, name, exact=False):
        if actual is None or expected is None:
            if actual is None and expected is None: return 0.0
            nonempty = expected if actual is None else actual
            if bool(torch.count_nonzero(nonempty) == 0): return 0.0
            raise RuntimeError(f"gradient composition missing gradient: {name}")
        ok = torch.equal(actual, expected) if exact else torch.allclose(actual, expected, atol=2e-7, rtol=2e-5)
        if not ok:
            raise RuntimeError(f"gradient composition mismatch {name}, max_abs={float((actual-expected).abs().max())}")
        return float((actual-expected).abs().max())

    hist().zero_optimizers(state); module.zero_grad(set_to_none=True)
    out, details = _forward(state, module, training=True, support=support, query=support)
    natural = torch.autograd.grad(out["total"], params, retain_graph=True, allow_unused=True)
    zero_details = {**details, "semantic_loss": details["semantic_loss"]*0.0,
                    "keep_loss": details["keep_loss"]*0.0}
    zero_aux = CFG["method"]["semantic_loss_weight"]*zero_details["semantic_loss"] + CFG["method"]["semantic_keep_weight"]*zero_details["keep_loss"]
    zero_stats = _protected_backward(out, zero_details, out["total"]+zero_aux)
    zero_errors = [compare(p.grad, g, name, exact=True) for (name,p),g in zip(originals,natural)]

    hist().zero_optimizers(state); module.zero_grad(set_to_none=True)
    out, details = _forward(state, module, training=True, support=support, query=support)
    total, _, _ = _combined_total(out, details, state, support)
    aux = CFG["method"]["semantic_loss_weight"]*details["semantic_loss"] + CFG["method"]["semantic_keep_weight"]*details["keep_loss"]
    h = details["h_original"]
    gb = torch.autograd.grad(out["total"], h, retain_graph=True)[0].detach()
    ga = torch.autograd.grad(aux, h, retain_graph=True)[0].detach()
    eps = float(CFG["method"]["gradient_eps"]); rho = float(CFG["method"]["gradient_aux_norm_max_ratio"])
    dot = (gb*ga).sum(); nb = gb.norm()
    projected = ga - dot/(gb.square().sum()+eps)*gb if bool(dot<0) and float(nb)>eps else ga
    if float(nb)<=eps: safe = torch.zeros_like(projected)
    else: safe = projected*min(1.0, float(rho*nb/(projected.norm()+eps)))
    base_gradients = torch.autograd.grad(out["total"], params, retain_graph=True, allow_unused=True)
    extra_gradients = torch.autograd.grad(h, params, grad_outputs=safe, retain_graph=True, allow_unused=True)
    expected = [(x if y is None else y if x is None else x+y) for x,y in zip(base_gradients,extra_gradients)]
    expected_new = torch.autograd.grad(aux, [p for _,p in additions], retain_graph=True, allow_unused=True)
    stats = _protected_backward(out, details, total)
    errors = [compare(p.grad,g,name) for (name,p),g in zip(originals,expected)]
    new_errors = [compare(p.grad,g,name) for (name,p),g in zip(additions,expected_new)]
    record = {"status":"PASS", "fold":fold, "hook_active_zero_aux_main_gradients_exact":True,
              "hook_active_zero_aux_max_abs_error":max(zero_errors,default=0.0),
              "independent_base_plus_safe_vjp_all_original_parameters":True,
              "original_parameters_max_abs_error":max(errors,default=0.0),
              "new_module_gradients_unchanged":True,"new_parameters_max_abs_error":max(new_errors,default=0.0),
              "gradient_protection":stats,"zero_aux_gradient_protection":zero_stats}
    del state,module; torch.cuda.empty_cache()
    return record

def _replay_step(state, module, optimizer, support):
    rt = runtime()
    hist().zero_optimizers(state); optimizer.zero_grad(set_to_none=True); module.train()
    out, details = _forward(state, module, training=True, support=support, query=support)
    total, base_support_loss, fused_support_loss = _combined_total(out, details, state, support)
    details["base_support_loss"] = base_support_loss
    losses = _scalar_losses(out, details, total)
    protection_stats = _protected_backward(out, details, total)
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    after_train = rt.rng_sha(rt.pack_rng())
    module.eval()
    with torch.no_grad():
        evaluated, eval_details = _forward(state, module, training=False, support=support, query=state["test_np"])
        metrics = _compact_metrics(_final_metrics(evaluated, eval_details, state))
    after_eval = rt.rng_sha(rt.pack_rng())
    return {"losses": losses, "metrics": metrics, "gradient_protection": protection_stats, "rng_after_train": after_train,
            "rng_after_eval": after_eval, "snapshot": _snapshot(state, module, optimizer)}


def _two_step_replay_preflight(fold, data):
    state, _, _, support, module, optimizer, _ = _initialized_task(fold, GROUPS[1], data)
    initial = _snapshot(state, module, optimizer)
    _replay_step(state, module, optimizer, support)
    after_first = _snapshot(state, module, optimizer)
    expected = _replay_step(state, module, optimizer, support)
    _restore(state, module, optimizer, after_first)
    replayed = _replay_step(state, module, optimizer, support)
    _compare_tree(expected["losses"], replayed["losses"], "replay.losses")
    _compare_tree(expected["metrics"], replayed["metrics"], "replay.metrics")
    _compare_tree(expected["gradient_protection"], replayed["gradient_protection"], "replay.gradient_protection")
    _compare_tree(expected["snapshot"], replayed["snapshot"], "replay.snapshot")
    _restore(state, module, optimizer, initial)
    del state
    torch.cuda.empty_cache()
    return {"fold": fold, "group": GROUPS[1], "status": "PASS", "second_update_loss_exact": True,
            "second_update_metrics_exact": True, "model_optimizer_and_rng_exact": True,
            "rng_after_train": expected["rng_after_train"], "rng_after_eval": expected["rng_after_eval"]}


def preflight():
    for directory in (TASKS, CHECKPOINTS, LOGS, RESULTS, HERE / "runtime/failures"):
        directory.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("E78.5 requires CUDA")
    torch.set_num_threads(1)
    save_text_cache()
    global DATA_IDENTITY
    data = _data_identity(prepare(1)[0]); DATA_IDENTITY = data
    parity = [_base_off_parity(fold) for fold in range(1, 11)]
    module_records = _module_preflight(1, data)
    gradient_protection = _gradient_protection_preflight(1)
    recovery = _two_step_replay_preflight(1, data)
    param_counts = {g: int(v["parameter_count"]) for g, v in module_records["groups"].items()}
    if param_counts[GROUPS[1]] != param_counts[GROUPS[2]]:
        raise RuntimeError("G2/G3 parameter count mismatch")
    sources = source_map(); frozen_sha = sha_object(sources)
    artifact = torch.load(HERE / "runtime/text_embeddings.pt", map_location="cpu", weights_only=False)
    result = {"status": "PASS", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": torch.cuda.get_device_name(0), "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
              "data_identity": {k: data[k] for k in ("row_order_sha256", "identity_limit", "overlap_ids_zero_based")},
              "base_off_parity": parity, "base_off_parity_passed": len(parity),
              "fapc_module_preflight": module_records, "two_step_restore_replay": recovery,
              "gradient_protection_independent_reference": gradient_protection,
              "text_cache": {"anchor_shape": list(artifact["anchors"].shape), "raw_shape": list(artifact["embeddings_raw"].shape), "text_dim": artifact["text_dim"], "effective_rank": int(artifact["effective_rank"]), "text_encoder_weights_sha256": artifact["text_encoder_weights_sha256"], "text_encoder_source_sha256": artifact["text_encoder_source_sha256"], "prompt_strings_sha256": artifact["prompt_strings_sha256"], "local_cache_sha256": sha_file(HERE / "runtime/text_embeddings.pt"), "gram_max_abs_error": artifact["gram_max_abs_error"]},
              "source_sha256": frozen_sha, "source_map": sources}
    atomic_json(RESULTS / "preflight.json", result)
    atomic_json(HERE / "manifest.json", {"experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
                                          "source_sha256": frozen_sha, "source_map": sources,
                                          "expected_tasks": 30, "completed_tasks": 0, "independent_audits": 0,
                                          "preflight": str(RESULTS / "preflight.json")})
    print(f"PREFLIGHT PASS source={frozen_sha} base-off={len(parity)}/10 groups=3", flush=True)
    return result


def worker(folds, frozen_sha):
    global DATA_IDENTITY
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA disappeared before worker startup")
    verify_frozen_source(frozen_sha)
    DATA_IDENTITY = _data_identity(prepare(1)[0])
    for fold in folds:
        for group in GROUPS:
            train_task(group, fold, frozen_sha)


def launch(workers=4):
    frozen = verify_frozen_source(); frozen_sha = frozen["source_sha256"]
    worker_count = max(1, min(int(workers), int(CFG["training"]["max_workers"])))
    splits = [[int(v) for v in values] for values in np.array_split(np.arange(1, 11), worker_count) if len(values)]
    procs = []
    for index, folds in enumerate(splits):
        token = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        log_path = LOGS / f"controller_{token}_worker_{index+1:02d}.log"
        stream = log_path.open("a", encoding="utf-8")
        command = [sys.executable, str(HERE / "run.py"), "--worker-folds", ",".join(map(str, folds)), "--source-sha", frozen_sha]
        proc = subprocess.Popen(command, cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        procs.append((proc, stream, folds, log_path))
    status = []
    for proc, stream, folds, log_path in procs:
        code = proc.wait(); stream.close()
        status.append({"folds": folds, "returncode": code, "log": str(log_path)})
    atomic_json(LOGS / "controller.json", {"workers": status, "finished": time.time()})
    failures = [row for row in status if row["returncode"] != 0]
    if failures:
        raise RuntimeError(f"formal workers failed; latest checkpoints retained: {failures}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--workers", type=int, default=CFG["training"]["default_workers"])
    parser.add_argument("--worker-folds", type=str)
    parser.add_argument("--source-sha", type=str)
    args = parser.parse_args()
    if args.worker_folds:
        if not args.source_sha:
            raise RuntimeError("worker-folds requires --source-sha")
        worker([int(v) for v in args.worker_folds.split(",")], args.source_sha)
    elif args.preflight:
        preflight()
    elif args.train:
        if not (HERE / "manifest.json").is_file():
            raise RuntimeError("run --preflight before formal training")
        launch(args.workers)
    else:
        parser.error("choose --preflight, --train, or --worker-folds")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
