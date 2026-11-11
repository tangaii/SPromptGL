#!/usr/bin/env python3
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode = True
from path_adapter import HERE, BASE, REPO, load_base_runner, sha_file, verify_relocation
from sh_dcpss import SHDCPSS, initialize, initialize_anchors

CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GROUPS = tuple(CFG["groups"])
BASE_GROUP = CFG["base_group"]
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
E78 = Path("/model/spromptgl/experiment/experiment78_dcpss")
RESULTS = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
TASKS = HERE / "runtime/tasks"
LOGS = HERE / "runtime/logs"
BASE_MODULE = None
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
DATA_IDENTITY = None


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
    files = [
        HERE / name for name in ("config.json", "design.md", "sh_dcpss.py", "run.py",
                                "audit.py", "path_adapter.py", "make_text_cache.py",
                                "literature/source_review.md")
    ]
    files += [REPO / name for name in ("network.py", "model.py", "layers.py", "utils.py",
                                        "data/ABIDE/processed_standard_data.csv",
                                        "data/ABIDE/modal_feat_dict.npy")]
    files += [HERE / "runtime/text_geometry.pt", E78 / "runtime/text_embeddings.pt"]
    files.extend(sorted((E61 / ".runtime/warmup").glob("fold*.pt")))
    files.extend((BASE / name for name in ("run.py", "soft_positive_dcr.py", "config.json",
                                           "manifest.json", "audit.py")))
    for fold in range(1, 11):
        token = f"{BASE_GROUP}_f{fold:02d}"
        files.extend((BASE / ".runtime/tasks" / f"{token}.json",
                      BASE / ".runtime/checkpoints" / f"{token}.best.pt",
                      BASE / ".runtime/checkpoints" / f"{token}.latest.pt"))
    for path in sorted(set(files)):
        if not path.is_file():
            raise FileNotFoundError(f"required immutable source is missing: {path}")
    manifest = {str(path.resolve()): sha_file(path) for path in sorted(set(files))}
    manifest.update({f"relocation_checked:{path}": digest for path, digest in verify_relocation().items()})
    manifest.update(base().source_manifest())
    return dict(sorted(manifest.items()))


def source_sha():
    return sha_object(source_map())


def verify_frozen_source(expected_sha=None):
    frozen = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    current = source_map()
    digest = sha_object(current)
    expected = expected_sha or frozen.get("source_sha256")
    if digest != expected or current != frozen.get("source_map"):
        raise RuntimeError("source bytes differ from the frozen E78.1 preflight manifest")
    return frozen


def save_text_geometry():
    script = HERE / "make_text_cache.py"
    spec = importlib.util.spec_from_file_location("experiment78_1_text_geometry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def prompt_tensors(device):
    payload = torch.load(HERE / "runtime/text_geometry.pt", map_location="cpu", weights_only=False)
    coords = torch.stack((payload["spromptgl_coords"], payload["anonymous_coords"])).to(device)
    classes = payload["prompt_classes"].to(device)
    return payload, coords, classes


def prepare(fold):
    state, warmup, identity = base().prepare(fold)
    state["fold"] = fold
    reference = BASE / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json"
    task = json.loads(reference.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("identity") != identity:
        raise RuntimeError(f"fold {fold} does not match the completed E74.4-G2 reference identity")
    if task.get("checkpoint_sha256") != sha_file(BASE / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"):
        raise RuntimeError(f"fold {fold} E74.4-G2 best checkpoint hash differs from its task record")
    return state, warmup, identity


def task_paths(group, fold):
    token = f"{group}_f{fold:02d}"
    return (TASKS / f"{token}.json", CHECKPOINTS / f"{token}.best.pt",
            CHECKPOINTS / f"{token}.latest.pt", LOGS / f"{token}.jsonl")


def support_ids(state):
    return np.setdiff1d(state["train_np"], state["test_np"], assume_unique=False).astype(np.int64)


def _forward(state, module=None, *, training=False, support=None, query=None,
             sites=None, intervention="full"):
    captured = {"calls": 0}
    if module is None or intervention == "module_off":
        return base().forward(state, BASE_GROUP, training=training), None
    if support is None or query is None or sites is None:
        raise ValueError("the active feature hook requires legal support, query IDs, and SITE_ID")
    layer = state["cam"].Outputlayer

    def fuse_input(_layer, args):
        captured["calls"] += 1
        if captured["calls"] != 1 or not args or tuple(args[0].shape) != (871, 144):
            raise RuntimeError("OutputLayer pre-hook must receive exactly one CAM144 tensor")
        h = args[0]
        h_fused, details = module.forward_feature(
            h, state["y"], support, query, sites, state["weights"],
            training=training, intervention=intervention,
        )
        captured["details"] = details
        return (h_fused, *args[1:])

    hook = layer.register_forward_pre_hook(fuse_input)
    try:
        out = base().forward(state, BASE_GROUP, training=training)
    finally:
        hook.remove()
    if captured["calls"] != 1:
        raise RuntimeError(f"OutputLayer executed {captured['calls']} times in one base forward")
    return out, captured["details"]


def _new_optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=CFG["training"]["new_lr"],
                             weight_decay=CFG["training"]["new_weight_decay"])


def _module_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _snapshot(state, module, optimizer):
    return {
        "base_state": hist().state_bundle(state),
        "module_state": _module_state(module),
        "new_optimizer": copy.deepcopy(optimizer.state_dict()),
        "rng": runtime().pack_rng(),
    }


def _restore(state, module, optimizer, payload):
    hist().load_bundle(state, payload["base_state"])
    module.load_state_dict(payload["module_state"], strict=True)
    optimizer.load_state_dict(payload["new_optimizer"])
    runtime().restore_rng(payload["rng"])


def _compact_metrics(metrics):
    return {key: float(value) for key, value in metrics.items() if key not in ("pred", "prob")}


def _metrics(logp, state):
    return runtime().classification_metrics(logp, state["test"], state["y"])


def _grad_norm(grads):
    values = [gradient.detach().double().square().sum() for gradient in grads if gradient is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def _loss_gradients(out, details, state, module):
    records = {}
    base_params = [parameter for parameter in state["model"].parameters() if parameter.requires_grad]
    cam_params = [parameter for parameter in state["cam"].parameters() if parameter.requires_grad]
    module_params = [parameter for parameter in module.parameters() if parameter.requires_grad]
    differentiable = {
        "semantic_weighted": details["losses"]["sem_weighted"],
        "heterogeneity_weighted": details["losses"]["het_weighted"],
        "site_weighted": details["losses"]["site_weighted"],
    }
    for name, loss in differentiable.items():
        targets = module_params + base_params + cam_params
        if loss.requires_grad:
            grads = torch.autograd.grad(loss, targets, retain_graph=True, allow_unused=True)
            a, b, c = len(module_params), len(base_params), len(cam_params)
            records[name] = {"module": _grad_norm(grads[:a]), "darg_gfr": _grad_norm(grads[a:a + b]),
                             "cam": _grad_norm(grads[a + b:a + b + c])}
        else:
            records[name] = {"module": 0.0, "darg_gfr": 0.0, "cam": 0.0}
    fused_gradient = torch.autograd.grad(out["cls"], details["h_fused"], retain_graph=True)[0].detach()
    branch_scalar = (fused_gradient * (details["lambda"] * details["feature_residual"])).sum()
    targets = module_params + base_params + cam_params
    branch_grads = torch.autograd.grad(branch_scalar, targets, retain_graph=True, allow_unused=True)
    a, b, c = len(module_params), len(base_params), len(cam_params)
    records["classification_semantic_residual_branch"] = {
        "module": _grad_norm(branch_grads[:a]),
        "darg_gfr": _grad_norm(branch_grads[a:a + b]),
        "cam": _grad_norm(branch_grads[a + b:a + b + c]),
    }
    cls_module = torch.autograd.grad(out["cls"], module_params, retain_graph=True, allow_unused=True)
    records["fused_classification_to_module"] = {"module": _grad_norm(cls_module)}
    return records


def _epoch_gradients(out, details, state, module):
    result = {"auxiliary": _loss_gradients(out, details, state, module)}
    return result


def _scalar_losses(out, details):
    record = {f"base_{key}": float(out[key].detach().cpu())
              for key in ("cls", "graph", "pair", "adv", "preserve", "total")}
    for key, value in details["losses"].items():
        if torch.is_tensor(value) and value.numel() == 1:
            record[key] = float(value.detach().cpu())
    record["combined_total"] = float((out["total"] + details["losses"]["sem_weighted"] +
                                      details["losses"]["het_weighted"] +
                                      details["losses"]["site_weighted"]).detach().cpu())
    return record


def _data_identity(state):
    data_path = REPO / "data/ABIDE/processed_standard_data.csv"
    frame = pd.read_csv(data_path)
    x, y = base().load_history().load_e69().load_e61().load_base().load_data()[0]
    model_x = torch.cat(state["parts"], dim=1).detach().cpu().numpy()
    if frame.shape != (871, 321) or not np.array_equal(frame.iloc[:, :-1].to_numpy(np.float32), x):
        raise RuntimeError("processed data row/value alignment differs from the historical load_data matrix")
    if not np.array_equal(frame.iloc[:, -1].to_numpy(np.int64) - 1, y):
        raise RuntimeError("processed data labels differ from historical ASD=0, TD=1 mapping")
    if not np.array_equal(model_x, x) or not np.array_equal(state["labels_np"], y):
        raise RuntimeError("loaded E74.4 patient feature/label order differs from the processed CSV")
    site_cols = [column for column in frame.columns if "SITE_ID" in column.upper()]
    age_cols = [column for column in frame.columns if "AGE_AT_SCAN" in column.upper()]
    if len(site_cols) != 20 or len(age_cols) != 26:
        raise RuntimeError(f"expected 20 SITE_ID and 26 AGE_AT_SCAN one-hot columns, got {len(site_cols)}, {len(age_cols)}")
    site_block = frame[site_cols].to_numpy(np.float64)
    age_block = frame[age_cols].to_numpy(np.float64)
    if not np.array_equal(site_block.sum(1), np.ones(871)) or not np.array_equal(age_block.sum(1), np.ones(871)):
        raise RuntimeError("SITE_ID or AGE_AT_SCAN one-hot row alignment failed")
    site_ids = site_block.argmax(1).astype(np.int64)
    age_ids = age_block.argmax(1).astype(np.int64)
    return {"frame": frame, "site_ids": site_ids, "age_ids": age_ids,
            "site_columns": site_cols, "age_columns": age_cols,
            "row_order_sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes() +
                                                np.ascontiguousarray(y).tobytes()).hexdigest(),
            "identity_limit": "processed CSV has no SUB_ID column; verified exact 871-row feature/label order against historical load_data, plus per-row SITE_ID/AGE_AT_SCAN one-hot alignment"}


def _state_tensor_sha(value):
    return sha_object(value)


def _baseline_update(state):
    hist().zero_optimizers(state)
    out = base().forward(state, BASE_GROUP, training=True)
    result = {"logp": out["logp"].detach().cpu().clone(),
              "losses": {key: float(out[key].detach().cpu()) for key in
                         ("total", "cls", "graph", "pair", "adv", "preserve")}}
    out["total"].backward()
    gradient = {key: {name: None if p.grad is None else p.grad.detach().cpu().clone()
                      for name, p in state[key].named_parameters()} for key in ("model", "cam")}
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result.update({"bundle": hist().state_bundle(state), "gradient": gradient,
                   "rng_sha": runtime().rng_sha(runtime().pack_rng())})
    return result


def _disabled_update(state):
    hist().zero_optimizers(state)
    out, details = _forward(state, None, training=True)
    if details is not None:
        raise RuntimeError("base-off path unexpectedly returned a SH-DCPSS module output")
    result = {"logp": out["logp"].detach().cpu().clone(),
              "losses": {key: float(out[key].detach().cpu()) for key in
                         ("total", "cls", "graph", "pair", "adv", "preserve")}}
    out["total"].backward()
    gradient = {key: {name: None if p.grad is None else p.grad.detach().cpu().clone()
                      for name, p in state[key].named_parameters()} for key in ("model", "cam")}
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    result.update({"bundle": hist().state_bundle(state), "gradient": gradient,
                   "rng_sha": runtime().rng_sha(runtime().pack_rng())})
    return result


def _compare_tree(left, right, where="root"):
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise RuntimeError(f"exact tensor mismatch at {where}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise RuntimeError(f"exact NumPy array mismatch at {where}")
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
        for index, (a, b) in enumerate(zip(left, right)):
            _compare_tree(a, b, f"{where}[{index}]")
        return
    if isinstance(left, float) and isinstance(right, float):
        if left != right:
            raise RuntimeError(f"scalar mismatch at {where}: {left} != {right}")
        return
    if left != right:
        raise RuntimeError(f"value mismatch at {where}: {left!r} != {right!r}")


def _base_off_parity(fold):
    reference_state, _, reference_identity = prepare(fold)
    reference = _baseline_update(reference_state)
    candidate_state, _, candidate_identity = prepare(fold)
    if reference_identity != candidate_identity:
        raise RuntimeError(f"fold {fold} E74.4 restore identity differs between parity paths")
    candidate = _disabled_update(candidate_state)
    _compare_tree(reference["logp"], candidate["logp"], "logp")
    _compare_tree(reference["losses"], candidate["losses"], "losses")
    _compare_tree(reference["gradient"], candidate["gradient"], "gradients")
    _compare_tree(reference["bundle"], candidate["bundle"], "updated_model_optimizers")
    if reference["rng_sha"] != candidate["rng_sha"]:
        raise RuntimeError(f"fold {fold} base-off RNG differs")
    result = {"fold": fold, "status": "PASS", "logp_exact": True, "all_losses_exact": True,
              "all_original_gradients_exact": True, "model_and_optimizer_updates_exact": True,
              "rng_exact": True, "identity": reference_identity}
    del reference_state, candidate_state
    torch.cuda.empty_cache()
    return result


def _make_module(state, group, data):
    geometry, coords, _ = prompt_tensors(state["y"].device)
    spec = CFG["group_specs"][group]
    prompt_class = geometry["prompt_classes"] if spec["prompt_source"] == "spromptgl" else geometry["anonymous_classes"]
    module = initialize(coords, prompt_class.to(state["y"].device), int(spec["k"]),
                        spec["prompt_source"], state["fold"], state["y"].device)
    return module


def _initialized_task(fold, group, data):
    state, warmup, identity = prepare(fold)
    rt = runtime()
    if rt.rng_sha(rt.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not restore the exact E61 epoch-10 RNG")
    support = support_ids(state)
    if np.intersect1d(support, state["test_np"]).size:
        raise RuntimeError("outer test patients entered the new prototype support set")
    module = _make_module(state, group, data)
    optimizer = _new_optimizer(module)
    rng_before = rt.rng_sha(rt.pack_rng())
    with torch.no_grad():
        warmup_cam = {"h": None}
        hook = state["cam"].Outputlayer.register_forward_pre_hook(
            lambda _layer, args: warmup_cam.update(h=args[0]) or None
        )
        try:
            base().forward(state, BASE_GROUP, training=False)
        finally:
            hook.remove()
    if rt.rng_sha(rt.pack_rng()) != rng_before:
        raise RuntimeError("deterministic anchor feature extraction changed public RNG")
    if warmup_cam["h"] is None or tuple(warmup_cam["h"].shape) != (871, 144):
        raise RuntimeError("E61 warmup did not produce expected CAM144 anchor source")
    anchor_info = initialize_anchors(module, warmup_cam["h"].detach(), state["y"],
                                     support, data["site_ids"])
    init_record = {
        "identity": identity, "warmup_sha256": identity["warmup_sha256"],
        "support_ids": support.tolist(),
        "support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
        "anchor_info": anchor_info,
        "module_parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "module_init_sha256": sha_object({key: value.detach().cpu() for key, value in module.named_parameters()}),
        "optimizer_init_sha256": sha_object(optimizer.state_dict()),
        "original_optimizer_sha256": identity["optimizer_sha"],
        "warmup_rng_sha": warmup["rng_sha"], "post_module_rng_sha": rng_before,
        "rng_unchanged_by_new_init": warmup["rng_sha"] == rng_before,
    }
    if init_record["module_parameter_count"] != CFG["method"]["trainable_parameters_per_group"]:
        raise RuntimeError(f"unexpected trainable parameter count: {init_record['module_parameter_count']}")
    return state, warmup, identity, support, module, optimizer, init_record


def _task_complete(group, fold, frozen_sha):
    task_path, best_path, _, _ = task_paths(group, fold)
    if not task_path.is_file() or not best_path.is_file():
        return False
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return task.get("status") == "COMPLETE" and task.get("source_sha256") == frozen_sha and task.get("checkpoint_sha256") == sha_file(best_path)
    except Exception:
        return False


def _checkpoint(group, fold, source_hash, identity, best, executed, trace, loss_history, init_record):
    return {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "source_sha256": source_hash, "identity": identity, "initialization": init_record,
        "best_epoch_1based": best["epoch"], "epochs_executed": executed,
        "metrics": best["metrics"], "test_probability": best["probability"],
        "test_prediction": best["prediction"], "best_state": best["state"],
        "best_module_diagnostics": best["diagnostics"], "best_losses": best["losses"],
        "best_rng": best["rng"], "rng_trace": trace, "loss_history": loss_history,
    }


def train_task(group, fold, frozen_sha):
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError(f"invalid task group={group} fold={fold}")
    if _task_complete(group, fold, frozen_sha):
        return json.loads(task_paths(group, fold)[0].read_text(encoding="utf-8"))
    state, warmup, identity, support, module, optimizer, init_record = _initialized_task(fold, group, DATA_IDENTITY)
    task_file, best_file, latest_file, log_file = task_paths(group, fold)
    if task_file.exists() and json.loads(task_file.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        raise RuntimeError(f"completed task record has incompatible SHA: {task_file}")
    rt = runtime()
    sites = torch.as_tensor(DATA_IDENTITY["site_ids"], dtype=torch.long, device=state["y"].device)
    train_queries = support.copy()
    start_epoch, wait, best_acc, best = 10, 0, -1.0, None
    trace, loss_history = [], []
    if latest_file.is_file():
        saved = torch.load(latest_file, map_location="cpu", weights_only=False)
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", frozen_sha),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest checkpoint incompatible at {key}: {latest_file}")
        _restore(state, module, optimizer, saved["current"])
        start_epoch = int(saved["next_epoch"])
        wait, best_acc, best = int(saved["wait"]), float(saved["best_acc"]), saved["best"]
        trace, loss_history = saved["rng_trace"], saved["loss_history"]
    elif task_file.exists():
        raise RuntimeError(f"incomplete task record has no resumable latest checkpoint: {task_file}")

    started = time.time()
    executed = start_epoch
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, CFG["max_epoch"]):
        rng_before = rt.rng_sha(rt.pack_rng())
        hist().zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        module.train()
        out, details = _forward(state, module, training=True, support=support,
                                query=train_queries, sites=sites)
        weighted = details["losses"]
        total = out["total"] + weighted["sem_weighted"] + weighted["het_weighted"] + weighted["site_weighted"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite total loss group={group} fold={fold} epoch={epoch + 1}")
        gradient_records = _epoch_gradients(out, details, state, module) if epoch == start_epoch else None
        total.backward()
        for name, component in (("DARG/GFR", state["model"]), ("CAM", state["cam"]), ("SH-DCPSS", module)):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in component.parameters()):
                raise FloatingPointError(f"non-finite gradients in {name}, group={group}, fold={fold}, epoch={epoch + 1}")
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        optimizer.step()
        rng_after_train = rt.rng_sha(rt.pack_rng())
        executed = epoch + 1
        module.eval()
        with torch.no_grad():
            eval_queries = state["test_np"].astype(np.int64)
            evaluated, eval_details = _forward(state, module, training=False, support=support,
                                               query=eval_queries, sites=sites)
            metrics = _metrics(evaluated["logp"], state)
        rng_after_eval = rt.rng_sha(rt.pack_rng())
        epoch_rng = {"epoch": epoch + 1, "rng_before": rng_before,
                     "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval}
        trace.append(epoch_rng)
        scalar_losses = _scalar_losses(out, details)
        loss_record = {"epoch": epoch + 1, "metrics": {key: float(metrics[key]) for key in METRICS},
                       "losses": scalar_losses, "rng": epoch_rng,
                       "gradient_audit_first_epoch": gradient_records}
        loss_history.append(loss_record)
        log_record(log_file, loss_record)
        if metrics["acc"] > best_acc:
            best_acc, wait = float(metrics["acc"]), 0
            diagnostics = {
                key: (value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value))
                for key, value in eval_details.items()
                if key in ("rho_support", "beta", "attention", "proto_z", "proto_q",
                           "prototype_counts", "base_counts", "support_site_weight", "support_ids",
                           "support_labels", "support_sites", "query_ids", "semantic_vector",
                           "feature_residual", "h_original", "h_fused", "class_evidence", "losses", "lambda")
            }
            best = {"epoch": epoch + 1, "metrics": _compact_metrics(metrics),
                    "probability": np.asarray(metrics["prob"]).copy(),
                    "prediction": np.asarray(metrics["pred"]).copy(),
                    "losses": scalar_losses, "diagnostics": diagnostics,
                    "rng": {**epoch_rng, "packed": rt.pack_rng()},
                    "state": _snapshot(state, module, optimizer)}
            atomic_torch(best_file, _checkpoint(group, fold, frozen_sha, identity,
                                                 best, executed, trace, loss_history, init_record))
        else:
            wait += 1
        if executed % CFG["training"]["checkpoint_interval_epochs"] == 0 or wait > CFG["early_stop_wait_gt"] or executed == CFG["max_epoch"]:
            atomic_torch(latest_file, {
                "group": group, "fold": fold, "source_sha256": frozen_sha,
                "identity": identity, "warmup_sha256": identity["warmup_sha256"],
                "current": _snapshot(state, module, optimizer), "next_epoch": executed,
                "wait": wait, "best_acc": best_acc, "best": best,
                "rng_trace": trace, "loss_history": loss_history,
            })
            print(f"{group} fold={fold} epoch={executed} best={best_acc:.6f}@{best['epoch'] if best else 'none'} wait={wait}", flush=True)
        del out, details, evaluated, eval_details, total
        if wait > CFG["early_stop_wait_gt"]:
            break
    if best is None:
        raise RuntimeError(f"no strict-ACC best checkpoint for {group} fold={fold}")
    payload = _checkpoint(group, fold, frozen_sha, identity, best, executed, trace, loss_history, init_record)
    atomic_torch(best_file, payload)
    task = {"status": "COMPLETE", "group": group, "fold": fold, "seed": 0,
            "source_sha256": frozen_sha, "checkpoint_sha256": sha_file(best_file),
            "identity": identity, "initialization": init_record,
            "best_epoch_1based": best["epoch"], "epochs_executed": executed,
            "metrics": best["metrics"], "rng_trace": trace,
            "wall_seconds": time.time() - started,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "device": torch.cuda.get_device_name(0)}
    atomic_json(task_file, task)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best['epoch']}", flush=True)
    return task


def _module_gradient_preflight(fold, data):
    state, warmup, identity = prepare(fold)
    support = support_ids(state)
    geometry, coords, _ = prompt_tensors(state["y"].device)
    module = initialize(coords, geometry["prompt_classes"].to(state["y"].device), 3,
                        "spromptgl", fold, state["y"].device)
    rt = runtime()
    rng_sha = rt.rng_sha(rt.pack_rng())
    with torch.no_grad():
        captured = {}
        hook = state["cam"].Outputlayer.register_forward_pre_hook(lambda _m, args: captured.update(h=args[0]) or None)
        try:
            base().forward(state, BASE_GROUP, training=False)
        finally:
            hook.remove()
    if rt.rng_sha(rt.pack_rng()) != rng_sha:
        raise RuntimeError("deterministic warmup CAM capture changed the public RNG")
    initialize_anchors(module, captured["h"].detach(), state["y"], support, data["site_ids"])
    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    out, details = _forward(state, module, training=True, support=support,
                            query=support, sites=sites)
    if tuple(details["h_fused"].shape) != (871, 144):
        raise RuntimeError("feature fusion changed CAM144 shape")
    if torch.equal(details["h_fused"][support], details["h_original"][support]):
        raise RuntimeError("feature fusion made no numerical change on any training-support patient")
    if torch.any(details["prototype_counts"] <= 0) or not torch.isfinite(details["h_fused"]).all():
        raise RuntimeError("prototype LOO support is empty or non-finite")
    if not details["is_support_query"].all() or details["loo_removed_mass"].shape != (len(support), 2, 3):
        raise RuntimeError("training query LOO identity/mass check failed")
    if (details["loo_removed_mass"].sum((1, 2)) <= 0).any():
        raise RuntimeError("a training query did not have its own soft support contribution removed")
    objective = details["losses"]["sem_weighted"] + details["losses"]["het_weighted"] + details["losses"]["site_weighted"]
    names = {"semantic_projection": module.semantic_projection.parameters(),
             "heterogeneity_projection": module.heterogeneity_projection.parameters()}
    gradients = {}
    for name, params in names.items():
        grad = torch.autograd.grad(objective, tuple(params), retain_graph=True, allow_unused=True)
        gradients[name] = _grad_norm(grad)
        if gradients[name] <= 0 or not math.isfinite(gradients[name]):
            raise RuntimeError(f"combined support losses do not reach {name}: {gradients[name]}")
    gradient_records = _epoch_gradients(out, details, state, module)["auxiliary"]
    branch = gradient_records["classification_semantic_residual_branch"]
    gradients["classification_residual_branch"] = branch
    if branch["darg_gfr"] <= 0 or branch["cam"] <= 0 or branch["module"] <= 0:
        raise RuntimeError(f"classification feature residual is disconnected: {branch}")
    feature_grad = torch.autograd.grad(out["cls"], tuple(module.feature_projection.parameters()),
                                       retain_graph=True, allow_unused=True)
    gradients["feature_projection_from_fused_classification"] = _grad_norm(feature_grad)
    if gradients["feature_projection_from_fused_classification"] <= 0:
        raise RuntimeError("fused classification loss does not reach the feature projection")
    original_y = state["y"].detach().clone()
    eval_queries = state["test_np"].astype(np.int64)
    module.eval()
    with torch.no_grad():
        out_a, det_a = _forward(state, module, training=False, support=support,
                                query=eval_queries, sites=sites)
        state["y"][torch.as_tensor(eval_queries, device=state["y"].device)] = 1 - state["y"][torch.as_tensor(eval_queries, device=state["y"].device)]
        out_b, det_b = _forward(state, module, training=False, support=support,
                                query=eval_queries, sites=sites)
        state["y"].copy_(original_y)
    if not torch.equal(out_a["logp"], out_b["logp"]) or not torch.equal(det_a["h_fused"], det_b["h_fused"]):
        raise RuntimeError("test-label changes altered E78.1 inference")
    h = details["h_original"].detach()
    off_fused, _ = module.forward_feature(h, state["y"], support, support,
                                          sites, state["weights"], training=False,
                                          intervention="feature_residual_off")
    if not torch.equal(off_fused, h):
        raise RuntimeError("feature-residual-off did not exactly recover the original CAM144")
    del state
    torch.cuda.empty_cache()
    loss_values = {key: float(value.detach().cpu()) for key, value in details["losses"].items()
                   if torch.is_tensor(value) and value.numel() == 1}
    loss_values.update({f"base_{key}": float(out[key].detach().cpu())
                        for key in ("cls", "graph", "pair", "adv", "preserve", "total")})
    if any(not math.isfinite(value) for value in loss_values.values()):
        raise RuntimeError(f"non-finite preflight loss scale: {loss_values}")
    return {"fold": fold, "status": "PASS", "gradient_norms": gradients,
            "loss_values_and_scales": loss_values,
            "all_train_queries_exact_loo": True, "support_count": len(support),
            "minimum_effective_prototype_count": float(details["prototype_counts"].min().detach().cpu()),
            "test_label_invariant": True, "fusion_changes_train_features": True,
            "fusion_off_exact": True, "frozen_text_cosine_error": geometry["max_pairwise_cosine_error_float64"]}


def _replay_step(state, module, optimizer, support, sites):
    rt = runtime()
    hist().zero_optimizers(state)
    optimizer.zero_grad(set_to_none=True)
    module.train()
    out, details = _forward(state, module, training=True, support=support,
                            query=support, sites=sites)
    losses = _scalar_losses(out, details)
    total = out["total"] + details["losses"]["sem_weighted"] + details["losses"]["het_weighted"] + details["losses"]["site_weighted"]
    total.backward()
    for key in ("opt_base", "opt_cam", "opt_added"):
        state[key].step()
    optimizer.step()
    after_train = rt.rng_sha(rt.pack_rng())
    module.eval()
    with torch.no_grad():
        evaluated, _ = _forward(state, module, training=False, support=support,
                                query=state["test_np"], sites=sites)
        metrics = _compact_metrics(_metrics(evaluated["logp"], state))
    after_eval = rt.rng_sha(rt.pack_rng())
    return {"losses": losses, "metrics": metrics,
            "rng_after_train": after_train, "rng_after_eval": after_eval,
            "snapshot": _snapshot(state, module, optimizer)}


def _two_step_replay_preflight(fold, data):
    state, _, _, support, module, optimizer, _ = _initialized_task(fold, GROUPS[1], data)
    sites = torch.as_tensor(data["site_ids"], dtype=torch.long, device=state["y"].device)
    initial = _snapshot(state, module, optimizer)
    first = _replay_step(state, module, optimizer, support, sites)
    after_first = first["snapshot"]
    second = _replay_step(state, module, optimizer, support, sites)
    expected_second = second
    _restore(state, module, optimizer, after_first)
    replayed_second = _replay_step(state, module, optimizer, support, sites)
    _compare_tree(expected_second["losses"], replayed_second["losses"], "replay.losses")
    _compare_tree(expected_second["metrics"], replayed_second["metrics"], "replay.metrics")
    _compare_tree(expected_second["snapshot"], replayed_second["snapshot"], "replay.state")
    _restore(state, module, optimizer, initial)
    result = {"fold": fold, "group": GROUPS[1], "status": "PASS",
              "second_update_loss_exact": True, "second_update_metrics_exact": True,
              "model_optimizer_and_rng_exact": True,
              "rng_after_train": expected_second["rng_after_train"],
              "rng_after_eval": expected_second["rng_after_eval"]}
    del state
    torch.cuda.empty_cache()
    return result


def preflight():
    for directory in (TASKS, CHECKPOINTS, LOGS, RESULTS, HERE / "runtime/failures"):
        directory.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("formal run requires the available CUDA device")
    torch.set_num_threads(1)
    save_text_geometry()
    global DATA_IDENTITY
    data = _data_identity(prepare(1)[0])
    DATA_IDENTITY = data
    parity = [_base_off_parity(fold) for fold in range(1, 11)]
    gradients = _module_gradient_preflight(1, data)
    recovery = _two_step_replay_preflight(1, data)
    sources = source_map()
    frozen_sha = sha_object(sources)
    artifact = torch.load(HERE / "runtime/text_geometry.pt", map_location="cpu", weights_only=False)
    result = {
        "status": "PASS", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0), "cuda_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "data_identity": {key: data[key] for key in ("row_order_sha256", "identity_limit", "site_columns", "age_columns")},
        "base_off_parity": parity, "base_off_parity_passed": len(parity),
        "gradient_and_loo": gradients,
        "text_geometry": {"rank": artifact["rank"], "max_pairwise_cosine_error": artifact["max_pairwise_cosine_error_float64"],
                           "source_cache_sha256": artifact["source_cache_sha256"]},
        "loss_finiteness_scales": "all three train-support losses were finite; raw values and coefficients are recorded at epoch one of every formal task",
        "two_step_restore_replay": recovery,
        "source_sha256": frozen_sha, "source_map": sources,
    }
    atomic_json(RESULTS / "preflight.json", result)
    atomic_json(HERE / "manifest.json", {"experiment": CFG["experiment"], "status": "PREFLIGHT_PASS",
                                          "source_sha256": frozen_sha, "source_map": sources,
                                          "expected_tasks": 30, "completed_tasks": 0,
                                          "independent_audits": 0, "preflight": str(RESULTS / "preflight.json")})
    print(f"PREFLIGHT PASS source={frozen_sha} base-off={len(parity)}/10", flush=True)
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
    frozen = verify_frozen_source()
    frozen_sha = frozen["source_sha256"]
    worker_count = max(1, min(int(workers), 10, torch.cuda.device_count() * 4))
    splits = [[int(value) for value in values] for values in np.array_split(np.arange(1, 11), worker_count) if len(values)]
    procs = []
    for index, folds in enumerate(splits):
        run_token = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        log_path = LOGS / f"controller_{run_token}_worker_{index + 1:02d}.log"
        stream = log_path.open("a", encoding="utf-8")
        command = [sys.executable, str(HERE / "run.py"), "--worker-folds",
                   ",".join(str(value) for value in folds), "--source-sha", frozen_sha]
        proc = subprocess.Popen(command, cwd=str(HERE), stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        procs.append((proc, stream, folds, log_path))
    status = []
    for proc, stream, folds, log_path in procs:
        code = proc.wait()
        stream.close()
        status.append({"folds": folds, "returncode": code, "log": str(log_path)})
    atomic_json(LOGS / "controller.json", {"workers": status, "finished": time.time()})
    failures = [row for row in status if row["returncode"] != 0]
    if failures:
        raise RuntimeError(f"formal workers failed; resumable latest checkpoints retained: {failures}")
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
            raise RuntimeError("worker-folds requires a frozen source SHA")
        worker([int(value) for value in args.worker_folds.split(",")], args.source_sha)
    elif args.preflight:
        preflight()
    elif args.train:
        if not (HERE / "manifest.json").is_file():
            raise RuntimeError("run --preflight before formal training")
        launch(args.workers)
    else:
        parser.error("choose --preflight, --train, or internal --worker-folds")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
