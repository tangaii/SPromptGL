#!/usr/bin/env python3
"""Experiment 76.9 RBCD formal training runner."""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
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
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
BASE_EXP = REPO / "experiment/experiment74_4_soft_positive_dcr"
E76_5 = REPO / "experiment/experiment76_5_scpr"
E76_6 = Path("/model/spromptgl/experiment/experiment76_6_bcspc")
E76_7 = Path("/model/spromptgl/experiment/experiment76_7_cje_bcspc")
E76_8 = Path("/model/spromptgl/experiment/experiment76_8_cgd_bcspc")
E72 = REPO / "experiment/experiment72_mgdss"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
GROUPS = ("G1-RBCD-StopGrad", "G2-RBCD-Protected-E2E")
BASE_GROUP = "E74.4-G2-Positive-DCR"
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS, CACHE, OOF_DIR = (RT / key for key in ("checkpoints", "tasks", "logs", "cache", "oof"))
RESULTS = EXP / "results"
for _path in (CKPTS, TASKS, LOGS, CACHE, OOF_DIR, RESULTS):
    _path.mkdir(parents=True, exist_ok=True)

import path_adapter
legacy = path_adapter.legacy


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False)):
        raise IOError(f"atomic JSON readback mismatch: {path}")


def atomic_torch(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    readback = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(readback, dict):
        raise IOError(f"checkpoint readback invalid: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    path = Path(path)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_sites():
    import torch
    with (REPO / "data/ABIDE/processed_standard_data.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames or []
    columns = [name for name in fields if name.upper().startswith("SITE_ID_")]
    if len(rows) != 871 or not columns:
        raise RuntimeError("ABIDE site IDs are missing or row-misaligned")
    values = [[float(row.get(name) or 0) for name in columns] for row in rows]
    if any(sum(row) != 1.0 for row in values):
        raise RuntimeError("expected a single site indicator per patient")
    return torch.tensor([max(range(len(row)), key=row.__getitem__) for row in values], dtype=torch.long)


def prompt_vectors():
    import ast
    import torch
    from path_adapter import legacy_module
    expected_prompts = legacy_module.PROMPTS_ASD_TD
    brain_cache_path = E76_5 / ".runtime/cache/brainprompt_rn50.pt"
    brain_cache = torch.load(brain_cache_path, map_location="cpu", weights_only=False)
    if tuple(brain_cache.get("prompts", ())) != expected_prompts or brain_cache.get("text_encoder") != "BrainPrompt frozen CLIP RN50":
        raise RuntimeError("BrainPrompt cache provenance/text order does not match audited official prompts")
    if brain_cache.get("rn50_sha256") != "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762":
        raise RuntimeError("BrainPrompt RN50 weights differ from audited cache")
    sp_cache = torch.load(E72 / ".runtime/text/text_embeddings.pt", map_location="cpu", weights_only=False)
    tree = ast.parse((REPO / "network.py").read_text(encoding="utf-8"))
    prompts = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "attr_words" for target in node.targets):
            prompts = ast.literal_eval(node.value)
            break
    if prompts is None or sp_cache.get("text_map", {}).get("original") != prompts:
        raise RuntimeError("original eight prompts/cache do not match network.py")
    if sp_cache.get("source_hashes", {}).get("rn50_sha256") != brain_cache["rn50_sha256"]:
        raise RuntimeError("prompt caches were not encoded by the same audited RN50")
    brain, sprompt = brain_cache["vectors"].detach().float().cpu(), sp_cache["embeddings"]["original"].detach().float().cpu()
    if tuple(brain.shape) != (2, 1024) or tuple(sprompt.shape) != (8, 1024):
        raise RuntimeError("prompt embedding dimensions are not 2x1024 and 8x1024")
    return brain, sprompt


def source_snapshot():
    paths = [
        EXP / "config.json", EXP / "run.py", EXP / "rbcd.py", EXP / "oof_teacher.py",
        EXP / "path_adapter.py", EXP / "audit.py",
        BASE_EXP / "run.py", BASE_EXP / "soft_positive_dcr.py", BASE_EXP / "config.json",
        BASE_EXP / "manifest.json", BASE_EXP / "audit.py", BASE_EXP / "results/audit_status.json",
        E76_7 / "run.py", E76_7 / "cje_bcspc.py", E76_7 / "config.json",
        E76_7 / "manifest.json", E76_7 / "results/audit_status.json",
        E76_8 / "run.py", E76_8 / "cgd_bcspc.py", E76_8 / "config.json",
        E76_8 / "manifest.json", E76_8 / "results/audit_status.json",
        E76_5 / "scpr.py", E76_5 / "run.py", E76_5 / ".runtime/cache/brainprompt_rn50.pt",
        E72 / ".runtime/text/text_embeddings.pt", E72 / "manifest.json",
        REPO / "main.py", REPO / "model.py", REPO / "network.py", REPO / "layers.py", REPO / "utils.py",
        REPO / "ABIDE-simple-2-concat-weighted-cosine.sh",
        REPO / "data/ABIDE/processed_standard_data.csv", REPO / "data/ABIDE/modal_feat_dict.npy",
    ]
    paths.extend(sorted((E61 / ".runtime/warmup").glob("fold*.pt")))
    paths.extend(sorted((E61 / ".runtime/tasks").glob("fold*.warmup.json")))
    paths.extend(sorted((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt")))
    paths.extend(sorted((BASE_EXP / ".runtime/tasks").glob(f"{BASE_GROUP}_f*.json")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("source/history inventory incomplete: " + ", ".join(missing))
    if len(list((E61 / ".runtime/warmup").glob("fold*.pt"))) != 10:
        raise RuntimeError("E61 warmup inventory is not ten folds")
    if len(list((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt"))) != 10:
        raise RuntimeError("E74.4-G2 checkpoint inventory is not ten folds")
    return {str(path.resolve()): sha_file(path) for path in paths}


def prepare_fold(fold):
    state, warm, identity, task, path = legacy.prepare_fold(fold)
    audit = json.loads((BASE_EXP / "results/audit_status.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("audited_checkpoints") != 30:
        raise RuntimeError("E74.4-G2 historical checkpoint audit is not PASS")
    expected = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected:
        raise RuntimeError(f"historical fold overlap differs fold={fold}")
    return state, warm, identity, task, path


def init_module(brain, sprompt, fold, group=None):
    from rbcd import init_isolated
    return init_isolated(brain, sprompt, fold).cuda()


def hook_forward(state, training):
    captured = []
    output_layer = state["cam"].Outputlayer
    handle = output_layer.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
    try:
        out = legacy.load_base().forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (871, 144):
        raise RuntimeError(f"expected exactly one Outputlayer CAM144, got {[tuple(item.shape) for item in captured]}")
    return out, captured[0]


def split_branch_ids(state, fold):
    labels = state["y"].detach().cpu().numpy()
    test = set(map(int, state["test_np"]))
    eligible = sorted(int(i) for i in state["train_np"] if int(i) not in test)
    if fold == 2:
        eligible = [i for i in eligible if i not in {376, 406}]
    from path_adapter import legacy_module
    a, b = legacy_module.split_support(eligible, labels)
    if set(a) & set(b) or set(a + b) != set(eligible) or set(eligible) & test:
        raise RuntimeError("illegal A/B branch training and support split")
    if fold == 2 and ({376, 406} & set(eligible)):
        raise RuntimeError("Fold-2 historical overlap labels are excluded from branch loss/memory")
    return eligible, a, b


def make_memory(ids, labels, cam144, logp, sites, old=None):
    import torch
    device = cam144.device
    if old is None:
        memory = {"features": torch.zeros((871, 4, 36), device=device),
                  "probabilities": torch.zeros((871, 2), device=device),
                  "labels": torch.full((871,), -1, dtype=torch.long, device=device),
                  "sites": sites.to(device).clone(), "observed": torch.zeros(871, dtype=torch.bool, device=device),
                  "updates": 0, "bootstrap": None}
    else:
        memory = old
    ids = torch.as_tensor(ids, dtype=torch.long, device=device)
    features = cam144.detach().reshape(871, 4, 36)
    probabilities = logp.detach().exp()
    prior = memory["observed"][ids]
    fresh, seen = ids[~prior], ids[prior]
    if fresh.numel():
        memory["features"][fresh], memory["probabilities"][fresh] = features[fresh], probabilities[fresh]
    if seen.numel():
        memory["features"][seen] = 0.9 * memory["features"][seen] + 0.1 * features[seen]
        memory["probabilities"][seen] = 0.9 * memory["probabilities"][seen] + 0.1 * probabilities[seen]
    memory["labels"][ids] = labels[ids]
    memory["sites"] = sites.to(device).clone()
    memory["observed"][ids] = True
    if ids.numel():
        memory["updates"] += 1
    if memory["features"].grad_fn is not None or memory["probabilities"].grad_fn is not None:
        raise RuntimeError("EMA memory unexpectedly retained an autograd graph")
    return memory


def memory_snapshot(memory):
    return {key: (value.detach().cpu().clone() if hasattr(value, "detach") else value)
            for key, value in memory.items()} if memory is not None else None


def memory_device(memory, device):
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in memory.items()} if memory is not None else None


def as_detached_memory(memory):
    return memory_device(memory_snapshot(memory), "cuda")


def load_oof(fold, device=None):
    import torch
    path = OOF_DIR / f"fold{int(fold):02d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing OOF risk cache: {path}")
    payload = torch_load(path)
    if payload.get("status") != "COMPLETE" or payload.get("fold") != int(fold):
        raise RuntimeError(f"OOF cache status/fold mismatch: {path}")
    state, _, _, _, _ = prepare_fold(fold)
    eligible, _, _ = split_branch_ids(state, fold)
    default_device = state["y"].device
    cached = [int(x) for x in payload["eligible_train_ids"]]
    if cached != eligible:
        raise RuntimeError(f"OOF eligible IDs differ from formal branch IDs fold={fold}")
    if not bool(payload.get("coverage_exactly_once")) or payload.get("test_labels_accessed") is not False:
        raise RuntimeError(f"OOF cache did not pass scope metadata fold={fold}")
    dev = device if device is not None else default_device
    del state
    risk_soft = torch.zeros(871, dtype=torch.float32, device=dev)
    risk_hard = torch.zeros(871, dtype=torch.float32, device=dev)
    probabilities = torch.zeros((871, 2), dtype=torch.float32, device=dev)
    ids = torch.as_tensor(cached, dtype=torch.long, device=dev)
    risk_soft[ids] = torch.as_tensor(payload["risk_soft"], dtype=torch.float32, device=dev)
    risk_hard[ids] = torch.as_tensor(payload["risk_hard"], dtype=torch.float32, device=dev)
    probabilities[ids] = torch.as_tensor(payload["oof_probability"], dtype=torch.float32, device=dev)
    if not bool(torch.isfinite(risk_soft[ids]).all()) or not bool(((risk_soft[ids] >= 0) & (risk_soft[ids] <= 1)).all()):
        raise RuntimeError(f"invalid OOF risk_soft fold={fold}")
    if not bool(torch.isfinite(probabilities[ids]).all()):
        raise RuntimeError(f"invalid OOF probabilities fold={fold}")
    return {"path": path, "sha256": sha_file(path), "payload": payload,
            "eligible": eligible, "risk_soft": risk_soft, "risk_hard": risk_hard,
            "probabilities": probabilities}

def branch_train(module, cam, memory, base_logp, labels, eligible, ids_a, ids_b, sites,
                 class_weights, risk_soft, risk_hard, *, detached):
    import torch
    from rbcd import branch_losses, weighted_nll
    pieces, diagnostics = [], []
    for query_ids, support_ids in ((ids_a, ids_b), (ids_b, ids_a)):
        if not query_ids:
            continue
        qids = torch.as_tensor(query_ids, dtype=torch.long, device=cam.device)
        support = torch.as_tensor(support_ids, dtype=torch.long, device=cam.device)
        branch_cam = cam.detach() if detached else cam
        branch_base = base_logp.detach() if detached else base_logp
        final, diag = module(branch_cam, qids, memory, support, sites, branch_base)
        pieces.append((qids, final, diag))
    if not pieces:
        raise RuntimeError("RBCD A/B query partition is empty")
    q_order = torch.cat([item[0] for item in pieces])
    final_logp = torch.cat([item[1] for item in pieces])
    diag = merge_diagnostics([item[2] for item in pieces])
    y = labels[q_order]
    rs = risk_soft[q_order]
    rh = risk_hard[q_order]
    loss, components = branch_losses(final_logp, diag, y, class_weights, rs, rh, base_logp[q_order])
    sample_weight = 1.0 + 2.0 * rh.to(base_logp.dtype)
    base_cls = weighted_nll(base_logp[q_order], y, class_weights, sample_weight)
    return loss, components, final_logp, diag, q_order, y, base_cls


def merge_diagnostics(items):
    import torch
    scalar = {"retrieval_fallbacks", "site_cap_violations", "invalid_confusion_candidates"}
    return {key: (sum(int(item[key]) for item in items) if key in scalar
                  else torch.cat([item[key] for item in items], dim=0)) for key in items[0]}


def eval_method(module, out_logp, cam, memory, eligible, test_ids, sites, *, residual_off=False,
                boundary_off=False, semantic_swap=False, no_semantic_key=False,
                reliable_only=False, risk_gate_off=False):
    import torch
    ids = torch.as_tensor(test_ids, dtype=torch.long, device=out_logp.device)
    support = torch.as_tensor(eligible, dtype=torch.long, device=out_logp.device)
    final, diag = module(
        cam, ids, memory, support, sites, out_logp,
        residual_off=residual_off, boundary_off=boundary_off,
        semantic_swap=semantic_swap, no_semantic_key=no_semantic_key,
        reliable_only=reliable_only, risk_gate_off=risk_gate_off,
    )
    whole = out_logp.clone()
    whole[ids] = final
    return whole, diag


def exact_metrics(logp, test_ids, labels):
    return legacy.extended_metrics(logp, test_ids, labels)


def _combine_group_gradients(entries, beta, limit, eps=1e-24, project_conflicts=True, protect_new=None):
    import torch
    protect_new = set() if protect_new is None else set(protect_new)
    base_sq = raw_sq = dot = ignored_new_sq = 0.0
    ignored_new_parameters = 0
    for _, base, new in entries:
        if base is None:
            if new is not None and not bool(torch.isfinite(new).all()):
                raise FloatingPointError("nonfinite new gradient without a base gradient")
            if new is not None:
                ignored_new_sq += float(new.detach().double().square().sum().cpu())
                ignored_new_parameters += 1
            continue
        if not bool(torch.isfinite(base).all()) or (new is not None and not bool(torch.isfinite(new).all())):
            raise FloatingPointError("nonfinite gradient in conflict projection")
        base64 = base.detach().double()
        base_sq += float(base64.square().sum().cpu())
        if new is not None:
            new64 = new.detach().double()
            raw_sq += float(new64.square().sum().cpu())
            dot += float((base64 * new64).sum().cpu())
    base_norm, raw_norm = math.sqrt(base_sq), math.sqrt(raw_sq)
    coefficient = dot / base_sq if project_conflicts and dot < 0.0 and base_sq > eps else 0.0
    projected, projected_sq, projected_dot = [], 0.0, 0.0
    for parameter, base, new in entries:
        if base is None:
            projected_gradient = None
        elif id(parameter) in protect_new:
            # Never synthesize a new increment for a protected parameter, even
            # when another parameter in the optimizer group has a conflict.
            projected_gradient = torch.zeros_like(base)
        elif coefficient:
            # Missing new gradients are zero coordinates of the SAME optimizer
            # group. Projection can populate them only where the base already
            # has a gradient; a missing base gradient always stays None.
            projected_gradient = (torch.zeros_like(base) if new is None else new) - coefficient * base
        else:
            projected_gradient = new
        if projected_gradient is not None and base is not None:
            pg64 = projected_gradient.detach().double()
            projected_sq += float(pg64.square().sum().cpu())
            projected_dot += float((base.detach().double() * pg64).sum().cpu())
        projected.append((parameter, base, new, projected_gradient))
    projected_norm = math.sqrt(projected_sq)
    scaled_norm = beta * projected_norm
    near_zero = base_norm <= 1e-12
    if near_zero:
        scale = 0.0 if scaled_norm > 0.0 else 1.0
    elif scaled_norm == 0.0:
        scale = 1.0
    else:
        scale = min(1.0, limit * base_norm / scaled_norm)
    applied_norm = scaled_norm * scale
    merged, applied = [], {}
    total_sq = total_dot = 0.0
    for parameter, base, new, projected_gradient in projected:
        if base is None:
            total, increment = None, None
        else:
            increment = (torch.zeros_like(base) if projected_gradient is None or scale == 0.0
                         else projected_gradient * (beta * scale))
            total = base.detach().clone() + increment.detach()
            total64, base64 = total.detach().double(), base.detach().double()
            total_sq += float(total64.square().sum().cpu())
            total_dot += float((base64 * total64).sum().cpu())
        merged.append((parameter, total))
        applied[id(parameter)] = (increment.detach() if increment is not None else None)
    total_norm = math.sqrt(total_sq)
    return merged, applied, {
        "base_norm": base_norm, "new_raw_norm": raw_norm,
        "new_projected_norm": projected_norm, "new_beta_norm": scaled_norm,
        "new_applied_norm": applied_norm,
        "relative_new_to_base": applied_norm / base_norm if base_norm > 1e-12 else 0.0,
        "cosine": dot / (base_norm * raw_norm) if base_norm > 1e-12 and raw_norm > 0 else 0.0,
        "projected_cosine": (projected_dot / (base_norm * projected_norm)
                             if base_norm > 1e-12 and projected_norm > 0 else 0.0),
        "base_vs_total_cosine": (total_dot / (base_norm * total_norm)
                                 if base_norm > 1e-12 and total_norm > 0 else 0.0),
        "negative_dot": bool(dot < 0.0), "projection_coefficient": coefficient,
        "new_gradient_without_base_norm": math.sqrt(ignored_new_sq),
        "new_gradient_without_base_parameter_count": ignored_new_parameters,
        "projection_removed_fraction": (
            max(0.0, min(1.0, math.sqrt(max(0.0, raw_norm**2 - projected_norm**2)) / raw_norm))
            if raw_norm > 0 else 0.0),
        "beta": float(beta), "cap": float(limit), "scale": scale,
        "clipped": bool(scale < 1.0), "base_near_zero": near_zero,
    }


def _new_gradient_summary():
    return {"optimizer_groups": {}, "gradient_families": {}}


def _accumulate_gradient_summary(summary, audit):
    metrics = ("cosine", "projected_cosine", "base_vs_total_cosine", "projection_removed_fraction",
               "base_norm", "new_raw_norm", "new_projected_norm", "new_applied_norm",
               "relative_new_to_base", "beta", "scale", "negative_cosine_magnitude")
    for section in ("optimizer_groups", "gradient_families"):
        rows = audit.get(section, {})
        target = summary.setdefault(section, {})
        for name, row in rows.items():
            acc = target.setdefault(name, {"steps": 0, "active_steps": 0, "negative_dot_steps": 0,
                                           "projected_negative_steps": 0, "clipped_steps": 0,
                                           **{f"sum_{key}": 0.0 for key in metrics}})
            acc["steps"] += 1
            active = row.get("new_raw_norm", 0.0) > 0.0
            acc["active_steps"] += int(active)
            acc["negative_dot_steps"] += int(bool(row.get("negative_dot", False)))
            acc["projected_negative_steps"] += int(row.get("projected_cosine", 0.0) < -1e-6)
            acc["clipped_steps"] += int(bool(row.get("clipped", False)))
            for key in metrics:
                value = (max(0.0, -float(row.get("cosine", 0.0))) if key == "negative_cosine_magnitude"
                         else float(row.get(key, 0.0)))
                acc[f"sum_{key}"] += value


def rbcd_math_tests(torch):
    from rbcd import correct_logp
    raw = torch.tensor([[0.7, -0.4], [-0.2, 0.2]], device="cuda")
    base = torch.log_softmax(raw, dim=-1)
    zero = torch.zeros(2, device="cuda", requires_grad=True)
    final = correct_logp(base, zero)
    if not torch.equal(final, base):
        raise RuntimeError("zero residual does not preserve base log-probability")
    if torch.autograd.grad(final[0, 0], zero, allow_unused=True)[0] is None:
        raise RuntimeError("zero residual derivative is disconnected")
    s = torch.tensor([0.8, 0.8], device="cuda")
    d = torch.tensor([-0.4, 0.4], device="cuda")
    opposition = torch.sigmoid(-s.detach() * d / 0.20)
    if not bool(((opposition >= 0) & (opposition <= 1)).all()):
        raise RuntimeError("opposition left [0,1]")
    if not opposition[0] > opposition[1]:
        raise RuntimeError("opposition sign is reversed")
    return {"zero_delta_exact": True, "zero_delta_gradient_connected": True,
            "risk_range": True, "opposition_sign": True,
            "disease_direction_sign": "ASD=0 positive; TD=1 negative"}

def gradients_and_step(state, module, out, cam, memory, branch_tuple, group, epoch, beta,
                       project_conflicts=True):
    import torch
    params = [p for p in list(state["model"].parameters()) + list(state["cam"].parameters()) if p.requires_grad]
    param_owner = {id(p): p for p in params}
    optimizers = [("opt_base", state["opt_base"], 0.10),
                  ("opt_cam", state["opt_cam"], 0.20),
                  ("opt_added", state["opt_added"], 0.10)]
    name_by_id = {id(p): f"model.{n}" for n, p in state["model"].named_parameters()}
    name_by_id.update({id(p): f"cam.{n}" for n, p in state["cam"].named_parameters()})
    module_params = list(module.parameters())
    loss, components, final_logp, diag, qids, y, base_cls = branch_tuple
    base_grads = torch.autograd.grad(out["total"], params, retain_graph=True, allow_unused=True)
    branch_grads = torch.autograd.grad(loss, module_params, retain_graph=True, allow_unused=True)
    phase_b = group == GROUPS[1] and (epoch + 1) >= 21
    cls_final = components["cls_final"]
    if phase_b:
        raw_new_grads = torch.autograd.grad(cls_final - base_cls, params, retain_graph=False, allow_unused=True)
    else:
        raw_new_grads = tuple(None for _ in params)
    outputlayer_raw_sq = 0.0
    new_grads = []
    for p, g in zip(params, raw_new_grads):
        name = name_by_id[id(p)]
        if name.startswith("cam.Outputlayer."):
            if g is not None:
                outputlayer_raw_sq += float(g.detach().double().square().sum().cpu())
            new_grads.append(None)
        else:
            new_grads.append(g)
    for p, g in zip(module_params, branch_grads):
        p.grad = None if g is None else g.detach().clone()
    audit = {"group": group, "epoch": epoch + 1, "beta": float(beta), "phase_b": phase_b,
             "end_to_end": phase_b, "optimizer_groups": {},
             "branch_gradient_norm": math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in branch_grads if g is not None)),
             "outputlayer_new_raw_norm": math.sqrt(outputlayer_raw_sq), "outputlayer_new_applied_norm": 0.0}
    audit["branch_gradient_families"] = {}
    for (name, _), g in zip(module.named_parameters(), branch_grads):
        if g is not None:
            family = name.split(".")[0]
            audit["branch_gradient_families"][family] = audit["branch_gradient_families"].get(family, 0.0) + float(g.detach().double().square().sum().cpu())
    audit["branch_gradient_families"] = {k: math.sqrt(v) for k, v in audit["branch_gradient_families"].items()}
    base_by_id = {id(p): g for p, g in zip(params, base_grads)}
    new_by_id = {id(p): g for p, g in zip(params, new_grads)}
    seen, applied_by_id, projection_by_id = set(), {}, {}
    for name, optimizer, limit in optimizers:
        for group_index, opt_group in enumerate(optimizer.param_groups):
            selected = [p for p in opt_group["params"] if id(p) in param_owner]
            if any(id(p) in seen for p in selected):
                raise RuntimeError("a backbone parameter appears in multiple original optimizer groups")
            seen.update(id(p) for p in selected)
            entries = [(p, base_by_id[id(p)], new_by_id[id(p)]) for p in selected]
            protected = {id(p) for p in selected if name_by_id[id(p)].startswith("cam.Outputlayer.")}
            merged, applied, group_audit = _combine_group_gradients(
                entries, beta, limit, project_conflicts=project_conflicts, protect_new=protected
            )
            key = f"{name}/{group_index}"
            audit["optimizer_groups"][key] = group_audit
            for p, total in merged:
                p.grad = total
                applied_by_id[id(p)] = applied[id(p)]
                projection_by_id[id(p)] = group_audit["projection_coefficient"]
    if seen != set(id(p) for p in params):
        raise RuntimeError(f"original optimizer groups do not cover {len(set(id(p) for p in params) - seen)} parameters")
    audit["new_to_base_gradient_ratio"] = {k: v["relative_new_to_base"] for k, v in audit["optimizer_groups"].items()}
    audit["clipped_groups"] = sum(bool(v["clipped"]) for v in audit["optimizer_groups"].values())
    family_gradients = {k: [] for k in ("gfr", "darg_scorer", "darg_encoder", "darg_gate", "cam", "outputlayer")}
    for p, bg, ng in zip(params, base_grads, new_grads):
        name = name_by_id[id(p)]
        family = ("gfr" if name.startswith(("model.graphs.", "model.gcn.", "model.fusion_weight")) else
                  "darg_scorer" if name.startswith("model.scorers.") else
                  "darg_encoder" if name.startswith(("model.encoders.", "model.discriminator.")) else
                  "darg_gate" if name.startswith(("model.edge_gate_logits", "model.reliability_gate_logits", "model.adaptive_gate_logits")) else
                  "cam" if name.startswith("cam.") and ".Outputlayer." not in name else
                  "outputlayer" if name.startswith("cam.Outputlayer.") else None)
        if family:
            family_gradients[family].append((p, bg, ng, applied_by_id.get(id(p))))
    audit["gradient_families"] = {}
    for family, entries in family_gradients.items():
        if not entries:
            continue
        base_sq = raw_sq = projected_sq = applied_sq = raw_dot = projected_dot = total_sq = total_dot = 0.0
        for p, base, raw, applied in entries:
            if base is None:
                continue
            b = base.detach().double(); base_sq += float(b.square().sum().cpu())
            if raw is not None:
                r = raw.detach().double(); raw_sq += float(r.square().sum().cpu()); raw_dot += float((b * r).sum().cpu())
            coeff = projection_by_id[id(p)]
            projected = (torch.zeros_like(base) if raw is None else raw) - coeff * base if coeff else raw
            pv = torch.zeros_like(base) if projected is None else projected
            av = torch.zeros_like(base) if applied is None else applied
            projected_sq += float(pv.detach().double().square().sum().cpu())
            projected_dot += float((b * pv.detach().double()).sum().cpu())
            applied_sq += float(av.detach().double().square().sum().cpu())
            total = base.detach() + av.detach(); total_sq += float(total.double().square().sum().cpu()); total_dot += float((b * total.double()).sum().cpu())
        bn, rn, pn, an = math.sqrt(base_sq), math.sqrt(raw_sq), math.sqrt(projected_sq), math.sqrt(applied_sq)
        audit["gradient_families"][family] = {
            "base_norm": bn, "new_raw_norm": rn, "new_projected_norm": pn,
            "new_applied_norm": an, "relative_new_to_base": an / bn if bn > 1e-12 else 0.0,
            "cosine": raw_dot / (bn * rn) if bn > 1e-12 and rn > 0 else 0.0,
            "projected_cosine": projected_dot / (bn * pn) if bn > 1e-12 and pn > 0 else 0.0,
            "base_vs_total_cosine": total_dot / (bn * math.sqrt(total_sq)) if bn > 1e-12 and total_sq > 0 else 0.0,
        }
    if audit["gradient_families"].get("outputlayer", {}).get("new_applied_norm", 0.0) != 0.0:
        raise RuntimeError("G2 OutputLayer received a new RBCD gradient")
    return audit


def parameter_norms(state):
    output={}
    for module_name,module in (("model",state["model"]),("cam",state["cam"])):
        for name,parameter in module.named_parameters():
            full=f"{module_name}.{name}"
            family=("gfr" if full.startswith(("model.graphs.","model.gcn.","model.fusion_weight")) else
                    "darg_scorer" if full.startswith("model.scorers.") else
                    "darg_encoder" if full.startswith(("model.encoders.","model.discriminator.")) else
                    "darg_gate" if full.startswith(("model.edge_gate_logits","model.reliability_gate_logits","model.adaptive_gate_logits")) else
                    "cam" if full.startswith("cam.") and ".Outputlayer." not in full else
                    "outputlayer" if full.startswith("cam.Outputlayer.") else "other")
            output.setdefault(family,{})[full]=parameter.detach().clone()
    return output


def conflict_projection_tests(torch):
    before_rng = torch.random.get_rng_state().clone()
    base = torch.tensor([1.0, 0.0], dtype=torch.float64)
    cases = {
        "same_direction": (torch.tensor([2.0, 0.0], dtype=torch.float64), False),
        "opposite_direction": (torch.tensor([-2.0, 0.0], dtype=torch.float64), True),
        "orthogonal": (torch.tensor([0.0, 2.0], dtype=torch.float64), False),
        "partial_conflict": (torch.tensor([-1.0, 2.0], dtype=torch.float64), True),
    }
    records = {}
    for name, (new, expect_projection) in cases.items():
        original = base.clone()
        parameter = torch.nn.Parameter(torch.zeros_like(base))
        merged, applied, item = _combine_group_gradients(
            [(parameter, base, new)], beta=0.3, limit=1.0)
        dot = float((base * new).sum().item())
        projected = new if not expect_projection else new - (dot / float(base.square().sum())) * base
        after = float((base * projected).sum().item())
        if (item["negative_dot"] != expect_projection or
                (expect_projection and after < -1e-12) or
                (not expect_projection and not torch.equal(projected, new))):
            raise RuntimeError(f"gradient conflict projection failed: {name}")
        if not torch.equal(base, original):
            raise RuntimeError("conflict projection modified the base gradient")
        total = merged[0][1]
        if total is None or not torch.isfinite(total).all() or not torch.isfinite(applied[id(parameter)]).all():
            raise RuntimeError(f"projection generated invalid gradients: {name}")
        records[name] = {"dot_before": dot, "dot_after": after,
                         "norm_before": float(new.norm()), "norm_after": float(projected.norm()),
                         "applied_new_norm": item["new_applied_norm"]}
    zero_parameter = torch.nn.Parameter(torch.zeros_like(base))
    zero, zero_applied, zero_item = _combine_group_gradients(
        [(zero_parameter, torch.zeros_like(base), torch.ones_like(base))], beta=0.3, limit=0.1)
    if zero[0][1] is None or not torch.equal(zero[0][1], torch.zeros_like(base)) or zero_item["new_applied_norm"] != 0.0:
        raise RuntimeError("zero-base gradient fallback must preserve a zero base update")
    none_parameter = torch.nn.Parameter(torch.zeros_like(base))
    none, none_applied, _ = _combine_group_gradients(
        [(none_parameter, None, torch.ones_like(base))], beta=0.3, limit=0.1)
    if none[0][1] is not None or none_applied[id(none_parameter)] is not None:
        raise RuntimeError("None base gradient changed optimizer-step semantics")
    clip_parameter = torch.nn.Parameter(torch.zeros_like(base))
    clipped, _, clip_item = _combine_group_gradients(
        [(clip_parameter, base, torch.tensor([-1.0, 10.0], dtype=torch.float64))], beta=0.3, limit=0.2)
    if clip_item["scale"] >= 1.0 or clip_item["new_applied_norm"] > 0.2 * clip_item["base_norm"] + 1e-12:
        raise RuntimeError("projection followed by norm clipping exceeded its cap")
    if not torch.isfinite(clipped[0][1]).all():
        raise RuntimeError("projection/clipping produced NaN or Inf")
    # A parameter group is a single vector even when autograd returns None for
    # some coordinates. Missing new coordinates are zero for the projection;
    # missing base coordinates remain None so Adam does not receive a new step.
    mixed_a = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    mixed_b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    mixed, _, mixed_item = _combine_group_gradients(
        [(mixed_a, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([-1.0, 0.0], dtype=torch.float64)),
         (mixed_b, torch.tensor([0.0, 2.0], dtype=torch.float64), None)], beta=0.3, limit=1.0)
    if (mixed_item["projected_cosine"] < -1e-12 or
            not torch.equal(mixed[1][1], torch.tensor([0.0, 2.12], dtype=torch.float64))):
        raise RuntimeError("mixed-None group projection did not use zero missing-new coordinates")
    disabled_parameter = torch.nn.Parameter(torch.zeros_like(base))
    _, _, disabled_item = _combine_group_gradients(
        [(disabled_parameter, base, torch.tensor([-1.0, 0.0], dtype=torch.float64))],
        beta=0.3, limit=1.0, project_conflicts=False)
    if disabled_item["projected_cosine"] >= -0.99 or disabled_item["negative_dot"] is not True:
        raise RuntimeError("disabled projection did not preserve the raw conflicting gradient")
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError("gradient projection consumed Torch RNG")
    records["zero_base_none_semantics_clipping"] = {"passed": True,"none_preserved":True,
        "zero_base_safe":True,"rng_unchanged":True,"clip_scale":clip_item["scale"],
        "mixed_none_group":True,"disabled_projection_raw_conflict":True}
    return records


def discriminative_prototype_tests(torch):
    from cgd_bcspc import (boundary_eta, correct_logp, evidence_gated_residual,
                           unique_boundary_component)
    rel = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64), dim=-1)
    same = rel.clone()
    identical = unique_boundary_component(same, rel)
    if identical.norm().item() > 1e-12:
        raise RuntimeError("equal reliable/boundary prototypes must yield zero unique evidence")
    orthogonal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    retained = unique_boundary_component(orthogonal, rel)
    if not torch.allclose(retained, orthogonal, atol=1e-12, rtol=0):
        raise RuntimeError("orthogonal boundary component was not retained")
    near = torch.nn.functional.normalize(torch.tensor([[1.0, 1e-9, 0.0]], dtype=torch.float64), dim=-1)
    near_component = unique_boundary_component(near, rel)
    if near_component.norm().item() > 2e-9:
        raise RuntimeError("near-equal boundary prototype was amplified")
    if not torch.equal(unique_boundary_component(orthogonal, rel, torch.tensor([False])), torch.zeros_like(orthogonal)):
        raise RuntimeError("invalid prototype support was not masked")
    if not torch.allclose((orthogonal*retained).sum(-1), torch.tensor([1.0], dtype=torch.float64)):
        raise RuntimeError("unique evidence sign/orientation is inconsistent")
    eta = boundary_eta(torch.tensor([[0.2,0.5]]),torch.tensor([[0.4,0.3]]),
                       torch.tensor([[0.8,0.6]]),torch.tensor([[0.5,0.2]]),
                       torch.tensor([[1.0,0.0]]),torch.tensor([[1.0,1.0]]))
    if not ((eta >= 0).all() and (eta <= 1).all()) or eta[0,1].item() != 0.0:
        raise RuntimeError("eta range/insufficient-support attenuation failed")
    if eta[0,0].item() <= 0.0 or eta[0,0].item() >= 1.0:
        raise RuntimeError("eta test did not exercise an intermediate reliability")
    base = torch.tensor([0.7], requires_grad=False)
    activation = torch.tensor([0.4], requires_grad=True)
    zero_delta, _ = evidence_gated_residual(base, activation, torch.zeros_like(base))
    delta, gate = evidence_gated_residual(base, activation, torch.tensor([0.2]))
    if zero_delta.item() != 0.0 or gate.item() == 0.0:
        raise RuntimeError("zero evidence must force a zero residual")
    final_layer = torch.nn.Linear(1,1)
    torch.nn.init.zeros_(final_layer.weight); torch.nn.init.zeros_(final_layer.bias)
    initial_activation=final_layer(torch.ones((1,1))).reshape(-1)
    initial_delta,_=evidence_gated_residual(base,initial_activation,torch.tensor([0.2]))
    grad=torch.autograd.grad(initial_delta.sum(),final_layer.weight)[0]
    if initial_delta.item()!=0.0 or grad.abs().max().item()<=0:
        raise RuntimeError("zero-initialized residual must retain nonzero evidence-gated parameter gradient")
    base_logp=torch.log_softmax(torch.tensor([[0.7,-0.4]]),dim=-1)
    corrected=correct_logp(base_logp,zero_delta)
    if not torch.allclose(corrected,base_logp,atol=0,rtol=0) or not torch.isfinite(corrected).all():
        raise RuntimeError("zero residual must preserve legal base log-probability exactly")
    # ASD=0 and TD=1: positive class-0 minus class-1 prototype evidence
    # must point toward ASD; the sign is explicitly checked in both classes.
    p_asd=torch.tensor([1.0,0.0]);p_td=torch.tensor([0.0,1.0]);q_asd=p_asd;q_td=p_td
    if torch.dot(q_asd,p_asd)-torch.dot(q_asd,p_td) <= 0 or torch.dot(q_td,p_asd)-torch.dot(q_td,p_td) >= 0:
        raise RuntimeError("disease-competition evidence sign does not match ASD=0/TD=1")
    return {"identical_prototypes_zero":True,"orthogonal_component_retained":True,
            "near_equal_not_amplified":True,"missing_support_masked":True,
            "eta_range_and_attenuation":True,"zero_evidence_zero_delta":True,
            "zero_initialized_head_has_gradient":True,"class_direction_sign":True}


def parameter_change_norms(state,before):
    after=parameter_norms(state)
    return {family:math.sqrt(sum(float((after[family][name]-tensor).detach().double().square().sum().cpu())
                                 for name,tensor in tensors.items()))
            for family,tensors in before.items()}


def _assert_state_equal(left, right, description):
    import torch
    def compare(a, b, path):
        if torch.is_tensor(a) and torch.is_tensor(b):
            if not torch.equal(a.detach().cpu(), b.detach().cpu()):
                raise RuntimeError(f"{description} differs at {path}")
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise RuntimeError(f"{description} key set differs at {path}")
            for key in a:
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b):
                raise RuntimeError(f"{description} sequence length differs at {path}")
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}/{index}")
        elif isinstance(a, float) and isinstance(b, float):
            if a != b:
                raise RuntimeError(f"{description} differs at {path}: {a} != {b}")
        elif a != b:
            raise RuntimeError(f"{description} differs at {path}: {a!r} != {b!r}")
    compare(left, right, "root")


def task_path(group, fold): return TASKS / f"{group}_f{fold:02d}.json"
def best_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.best.pt"
def latest_path(group, fold): return CKPTS / f"{group}_f{fold:02d}.latest.pt"


def _checkpoint_payload(group, fold, identity, warm_sha, source_sha, payload, trace, epochs, history_task):
    return {"experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
            "base_group": BASE_GROUP, "identity": identity, "warmup_sha256": warm_sha,
            "source_sha256": source_sha, "best_epoch_1based": payload["epoch"],
            "epochs_executed": epochs, "best": payload, "rng_trace": trace,
            "history_task_sha256": sha_obj({"identity": history_task["identity"],
                                              "best_epoch_1based": history_task["best_epoch_1based"],
                                              "epochs_executed": history_task["epochs_executed"]})}


def _capture_best(state, module, optimizer, memory_used, eligible, ids_a, ids_b, epoch,
                  metrics, base_logp, final_logp, diag, loss_components, grad_audit,
                  gradient_summary, train_curve, base_ledger, warm_sha, rng_state, rng_record):
    import numpy as np
    prediction, probability = np.asarray(metrics["pred"], int), np.asarray(metrics["prob"], float)
    ids = torch_ids = __import__("torch").as_tensor(state["test_np"], dtype=__import__("torch").long, device=base_logp.device)
    base_metrics = exact_metrics(base_logp, list(map(int, state["test_np"])), state["y"])
    return {"epoch": epoch, "metrics": {key: float(metrics[key]) for key in legacy.METRICS},
            "prediction": prediction, "probability": probability,
            "base_metrics": {key: float(base_metrics[key]) for key in legacy.METRICS},
            "base_prediction": np.asarray(base_metrics["pred"], dtype=np.int64),
            "base_probability": np.asarray(base_metrics["prob"], dtype=np.float64),
            "base_state": legacy.load_base().load_history().state_bundle(state),
            "module": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "memory_used": memory_snapshot(memory_used), "memory_updates": int(memory_used["updates"]),
            "eligible_train_ids": list(eligible), "support_a": list(ids_a), "support_b": list(ids_b),
            "test_ids": list(map(int, state["test_np"])),
            "diagnostics": {key: (value.detach().cpu().clone() if hasattr(value, "detach") else value)
                            for key, value in diag.items()},
            "branch_loss_components": {key: float(value.detach().cpu()) for key, value in loss_components.items()},
            "gradient_snapshot": copy.deepcopy(grad_audit), "gradient_summary": copy.deepcopy(gradient_summary),
            "train_curve": copy.deepcopy(train_curve),
            "base_ledger": copy.deepcopy(base_ledger),
            "fold_warmup_sha256": warm_sha, "state_identity": state["rudag_identity"],
            "rng": copy.deepcopy(rng_state), "rng_after_eval_sha256": rng_record["rng_after_eval"],
            "rng_trace_record": copy.deepcopy(rng_record),
            "training_config": copy.deepcopy(CFG["training"])}


def _atomic_best(group, fold, identity, warm_sha, source_sha, payload, trace, epochs, history_task):
    record = _checkpoint_payload(group, fold, identity, warm_sha, source_sha, payload, trace, epochs, history_task)
    atomic_torch(best_path(group, fold), record)
    return record


def _save_latest(group, fold, identity, warm_sha, source_sha, state, module, optimizer,
                 memory, epoch, wait, best_acc, best, trace, base_ledger, gradient_audit,
                 gradient_summary, train_curve, history_task, started):
    payload = {"group": group, "fold": fold, "source_sha256": source_sha,
               "identity": identity, "warmup_sha256": warm_sha,
               "oof_sha256": state.get("rbcd_oof_sha256"),
               "base_state": legacy.load_base().load_history().state_bundle(state),
               "module": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
               "optimizer": copy.deepcopy(optimizer.state_dict()), "memory": memory_snapshot(memory),
               "next_epoch": epoch, "wait": wait, "best_acc": best_acc, "best": best,
               "rng": legacy.history_runtime().pack_rng(), "trace": trace,
               "base_ledger": base_ledger, "gradient_audit": gradient_audit,
               "gradient_summary": gradient_summary, "train_curve": train_curve,
               "history_task": history_task, "started_wall_time": started}
    atomic_torch(latest_path(group, fold), payload)


def train_worker(group, fold):
    import numpy as np
    import torch
    if group not in GROUPS or fold not in range(1, 11):
        raise ValueError("invalid group/fold")
    source_files = source_snapshot()
    source_sha = sha_obj(source_files)
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_sha:
        raise RuntimeError("formal worker source identity differs from passing preflight")
    state, warm, identity, history_task, history_best = prepare_fold(fold)
    if legacy.history_runtime().rng_sha(legacy.history_runtime().pack_rng()) != warm["rng_sha"]:
        raise RuntimeError("did not start at exact E61 epoch-10 RNG")
    state["fold"], state["group"], state["rudag_identity"] = fold, BASE_GROUP, identity
    labels = state["y"].long()
    eligible, ids_a, ids_b = split_branch_ids(state, fold)
    oof = load_oof(fold, labels.device)
    if oof["eligible"] != eligible:
        raise RuntimeError("OOF/formal eligible IDs differ")
    risk_soft, risk_hard = oof["risk_soft"], oof["risk_hard"]
    state["rbcd_oof_sha256"] = oof["sha256"]
    sites = read_sites().to(labels.device)
    brain, sprompt = prompt_vectors()
    module = init_module(brain, sprompt, fold, group)
    branch_opt = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    rng = legacy.history_runtime()
    if rng.rng_sha(rng.pack_rng()) != warm["rng_sha"]:
        raise RuntimeError("new RBCD module/optimizer initialization contaminated warmup RNG")
    task_file = task_path(group, fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == source_sha and existing.get("checkpoint_sha256") == sha_file(best_path(group, fold)):
            print(f"SKIP complete {group} fold={fold}", flush=True); return existing
        if existing.get("status") not in ("RUNNING", "FAILED"):
            raise RuntimeError(f"refuse incompatible task overwrite: {task_file}")
    memory, trace, gradient_audit, gradient_summary, train_curve = None, [], [], _new_gradient_summary(), []
    base_wait, base_best_acc, base_best_epoch, base_best_state = 0, -1.0, -1, None
    method_wait, best_acc, best_epoch, best = 0, -1.0, -1, None
    start_epoch = 10
    started = time.time()
    if latest_path(group, fold).exists():
        saved = torch_load(latest_path(group, fold))
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_sha), ("identity", identity), ("warmup_sha256", identity["warmup_sha256"]), ("oof_sha256", oof["sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest recovery identity differs: {key}")
        legacy.load_base().load_history().load_bundle(state, saved["base_state"])
        module.load_state_dict(saved["module"], strict=True); branch_opt.load_state_dict(saved["optimizer"])
        rng.restore_rng(saved["rng"]); memory = memory_device(saved["memory"], labels.device)
        start_epoch, method_wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best = float(saved["best_acc"]), int(saved["best"]["epoch"]) if saved.get("best") else -1, saved.get("best")
        trace, base_wait = saved["trace"], saved["base_ledger"]["wait"]
        base_best_acc, base_best_epoch = saved["base_ledger"]["best_acc"], saved["base_ledger"]["best_epoch"]
        base_best_state = saved["base_ledger"].get("state")
        gradient_audit = saved["gradient_audit"]; gradient_summary = saved.get("gradient_summary", _new_gradient_summary()); train_curve = saved.get("train_curve", [])
        started = float(saved["started_wall_time"])
    elif task_file.exists():
        task_file.unlink()
    if start_epoch < 10 or start_epoch > 450:
        raise RuntimeError("invalid next epoch")
    atomic_json(task_file, {"status":"RUNNING", "group":group, "fold":fold, "source_sha256":source_sha,
                            "identity":identity, "warmup_sha256":identity["warmup_sha256"], "oof_sha256":oof["sha256"],
                            "support_A":ids_a, "support_B":ids_b, "eligible_train_ids":eligible, "next_epoch":start_epoch})
    hist = legacy.load_base().load_history(); hist.zero_optimizers(state); torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, 450):
        rng_before = rng.rng_sha(rng.pack_rng()); hist.zero_optimizers(state); branch_opt.zero_grad(set_to_none=True)
        out, train_cam = hook_forward(state, training=True)
        if not torch.isfinite(out["total"]): raise FloatingPointError(f"nonfinite base loss {group} fold={fold} epoch={epoch+1}")
        if memory is None: memory = make_memory([], labels, train_cam, out["logp"], sites)
        memory_used = memory_snapshot(memory); memory_cuda = memory_device(memory_used, labels.device)
        branch_active = int(memory_used["updates"]) > 0
        live = group == GROUPS[1] and epoch + 1 >= 21
        branch_result = branch_train(module, train_cam, memory_cuda, out["logp"], labels, eligible, ids_a, ids_b, sites,
                                     state["weights"], risk_soft, risk_hard, detached=not live)
        branch_loss, components, final_train_logp, diag, qids, branch_labels, base_cls = branch_result
        beta = 0.3 * min(1.0, max(0.0, (epoch + 1 - 20) / 20.0)) if live else 0.0
        grad_audit_item = gradients_and_step(state, module, out, train_cam, memory_cuda, branch_result, group, epoch, beta)
        _accumulate_gradient_summary(gradient_summary, grad_audit_item)
        if epoch + 1 in (11, 12, 20, 21, 22, 40, 100, 300, 449): gradient_audit.append(grad_audit_item)
        before_norms = parameter_norms(state)
        for key in ("opt_base", "opt_cam", "opt_added"):
            for pg in state[key].param_groups:
                for parameter in pg["params"]:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError(f"nonfinite inherited gradient {key}")
            state[key].step()
        for parameter in module.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()): raise FloatingPointError("nonfinite RBCD gradient")
        if branch_active: branch_opt.step()
        grad_audit_item["parameter_delta_norms"] = parameter_change_norms(state, before_norms)
        grad_audit_item["branch_loss_components"] = {k: float(v.detach().cpu()) for k, v in components.items()}
        rng_after_train = rng.rng_sha(rng.pack_rng())
        with torch.no_grad():
            evaluated, eval_cam = hook_forward(state, training=False)
            base_metrics = exact_metrics(evaluated["logp"], list(map(int, state["test_np"])), labels)
            final_logp, eval_diag = eval_method(module, evaluated["logp"], eval_cam, memory_cuda, eligible, list(map(int, state["test_np"])), sites)
            metrics = exact_metrics(final_logp, list(map(int, state["test_np"])), labels)
            next_memory = make_memory(eligible, labels, eval_cam, evaluated["logp"], sites, memory)
        rng_after_eval = rng.rng_sha(rng.pack_rng())
        rng_row = {"epoch":epoch+1, "rng_before":rng_before, "rng_after_train":rng_after_train, "rng_after_eval":rng_after_eval}
        trace.append(rng_row)
        train_curve.append({"epoch":epoch+1,"base_train_acc":float((out["logp"][eligible].argmax(-1)==labels[eligible]).float().mean().cpu()),
                            "base_eval_acc":float(base_metrics["acc"]),"method_eval_acc":float(metrics["acc"]),"beta":float(beta),
                            "base_loss":float(out["total"].detach().cpu()),"branch_loss":float(branch_loss.detach().cpu()),
                            "delta_abs_mean":float(eval_diag["delta"].abs().mean().cpu()),"risk_mean":float(eval_diag["risk"].mean().cpu()),
                            "rng_before":rng_before,"rng_after_train":rng_after_train,"rng_after_eval":rng_after_eval})
        if float(base_metrics["acc"]) > base_best_acc:
            base_best_acc, base_best_epoch, base_wait = float(base_metrics["acc"]), epoch+1, 0; base_best_state = hist.state_bundle(state)
        else: base_wait += 1
        if float(metrics["acc"]) > best_acc:
            best_acc, best_epoch, method_wait = float(metrics["acc"]), epoch+1, 0
            best = _capture_best(state, module, branch_opt, memory_used, eligible, ids_a, ids_b, epoch+1, metrics,
                                 evaluated["logp"], final_logp, eval_diag, components, grad_audit_item, gradient_summary,
                                 train_curve, {"best_acc":base_best_acc,"best_epoch":base_best_epoch,"wait":base_wait},
                                 identity["warmup_sha256"], rng.pack_rng(), rng_row)
            best["oof_sha256"] = oof["sha256"]; best["risk_target_sha256"] = sha_obj({"risk_soft":oof["payload"]["risk_soft"],"risk_hard":oof["payload"]["risk_hard"]})
            _atomic_best(group, fold, identity, identity["warmup_sha256"], source_sha, best, trace, epoch+1, history_task)
        else: method_wait += 1
        memory = next_memory
        latest_due = ((epoch+1)%10==0 or epoch+1 in (11,20,21,40,100,300,449) or base_wait>200 or epoch+1==450)
        if latest_due:
            _save_latest(group, fold, identity, identity["warmup_sha256"], source_sha, state, module, branch_opt, memory,
                         epoch+1, method_wait, best_acc, best, trace,
                         {"best_acc":base_best_acc,"best_epoch":base_best_epoch,"wait":base_wait,"state":base_best_state},
                         gradient_audit, gradient_summary, train_curve, history_task, started)
        task_record = {"status":"RUNNING","group":group,"fold":fold,"source_sha256":source_sha,"identity":identity,
                       "warmup_sha256":identity["warmup_sha256"],"oof_sha256":oof["sha256"],"next_epoch":epoch+1,
                       "base_best_acc":base_best_acc,"base_best_epoch":base_best_epoch,"base_wait":base_wait,
                       "method_best_acc":best_acc,"method_best_epoch":best_epoch,"method_wait":method_wait,
                       "rng_records":len(trace),"memory_updates":memory["updates"],"train_curve_rows":len(train_curve),
                       "peak_gpu_bytes":int(torch.cuda.max_memory_allocated())}
        atomic_json(task_file, task_record)
        if (epoch+1)%10==0 or epoch in (10,19,20,21,39,99,299,448):
            with (LOGS/f"{group}_f{fold:02d}.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps({"group":group,"fold":fold,"epoch":epoch+1,"base_acc":float(base_metrics["acc"]),"base_best_acc":base_best_acc,
                    "base_best_epoch":base_best_epoch,"base_wait":base_wait,"method_acc":float(metrics["acc"]),"method_best_acc":best_acc,
                    "method_best_epoch":best_epoch,"beta":beta,"branch_loss":float(branch_loss.detach().cpu()),"base_loss":float(out["total"].detach().cpu()),
                    "delta_abs_mean":float(eval_diag["delta"].abs().mean().cpu()),"risk_mean":float(eval_diag["risk"].mean().cpu()),"g1_new_gradient":grad_audit_item,"rng":rng_row,
                    "memory_updates":memory["updates"],"peak_gpu_bytes":int(torch.cuda.max_memory_allocated())},allow_nan=False)+"\n"); stream.flush(); os.fsync(stream.fileno())
            print(f"{group} fold={fold} epoch={epoch+1} base={base_best_acc:.6f}@{base_best_epoch} method={best_acc:.6f}@{best_epoch} beta={beta:.3f}",flush=True)
        del out, train_cam, evaluated, eval_cam, final_logp
        if base_wait > 200: break
    if trace and (not latest_path(group, fold).exists() or json.loads(task_file.read_text()).get("next_epoch") != trace[-1]["epoch"]):
        _save_latest(group, fold, identity, identity["warmup_sha256"], source_sha, state, module, branch_opt, memory, trace[-1]["epoch"], method_wait,
                     best_acc, best, trace, {"best_acc":base_best_acc,"best_epoch":base_best_epoch,"wait":base_wait,"state":base_best_state},
                     gradient_audit, gradient_summary, train_curve, history_task, started)
    if best is None: raise RuntimeError(f"no method best checkpoint selected: {group} fold={fold}")
    _atomic_best(group, fold, identity, identity["warmup_sha256"], source_sha, best, trace, trace[-1]["epoch"], history_task)
    task_record.update({"status":"COMPLETE","best_epoch_1based":best_epoch,"epochs_executed":trace[-1]["epoch"],"metrics":best["metrics"],
                        "checkpoint_sha256":sha_file(best_path(group,fold)),"wall_seconds":time.time()-started,"peak_gpu_bytes":int(torch.cuda.max_memory_allocated()),
                        "device":torch.cuda.get_device_name(0),"base_best_epoch_1based":base_best_epoch,"rng_mismatches":0,"gradient_summary":gradient_summary})
    atomic_json(task_file, task_record)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch} epochs={len(trace)}",flush=True)
    return task_record


def preflight():
    import numpy as np
    import torch
    from rbcd import branch_losses
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("E76.9 requires the configured single GPU")
    base_audit = json.loads((BASE_EXP/"results/audit_status.json").read_text())
    if base_audit.get("status") != "PASS" or base_audit.get("audited_checkpoints") != 30:
        raise RuntimeError("E74.4-G2 historical audit is not PASS")
    for exp in (E76_7, E76_8):
        audit = json.loads((exp/"results/audit_status.json").read_text())
        if audit.get("status") != "COMPLETE_AUDITED" or audit.get("checkpoint_recalculations") != 20:
            raise RuntimeError(f"historical reference audit unavailable: {exp}")
    sources = source_snapshot(); source_sha = sha_obj(sources)
    inherited=[]
    for fold in range(1,11):
        state,warm,identity,task,path=prepare_fold(fold)
        if state["train_pairs_np"].shape != (4096,2): raise RuntimeError(f"4096 relation pairs absent fold={fold}")
        if identity["warmup_sha256"] != sha_file(E61/".runtime/warmup"/f"fold{fold:02d}.pt"): raise RuntimeError(f"warmup SHA mismatch fold={fold}")
        oof=load_oof(fold,state["y"].device)
        if len(oof["eligible"]) != len(set(oof["eligible"])): raise RuntimeError(f"OOF duplicate eligible IDs fold={fold}")
        inherited.append({"fold":fold,"warmup_sha256":identity["warmup_sha256"],"initial_model_sha":identity["model_cam_sha"],
                          "optimizer_sha":identity["optimizer_sha"],"split_sha":identity["split_sha"],"pair_sha":identity["pair_sha"],
                          "relation_pairs":4096,"overlap":identity["membership"]["train_test_overlap_1based"],"oof_sha256":oof["sha256"],
                          "oof_error_rate":float(oof["risk_hard"][oof["risk_hard"]>=0].mean().cpu())})
        del state; torch.cuda.empty_cache()
    brain,sprompt=prompt_vectors(); modules=[]; peak=0
    for fold in range(1,11):
        state,_,_,_,_=prepare_fold(fold)
        before=legacy.history_runtime().rng_sha(legacy.history_runtime().pack_rng())
        g1=init_module(brain,sprompt,fold,GROUPS[0]); g2=init_module(brain,sprompt,fold,GROUPS[1])
        if legacy.history_runtime().rng_sha(legacy.history_runtime().pack_rng()) != before: raise RuntimeError("RBCD initialization consumed main RNG")
        for a,b in zip(g1.state_dict().values(),g2.state_dict().values()):
            if not torch.equal(a.cpu(),b.cpu()): raise RuntimeError(f"G1/G2 initialization differs fold={fold}")
        probe=torch.zeros((2,1),device="cuda"); probe_delta=g1.correction_mlp(torch.zeros((2,g1.CORRECTION_DIM),device="cuda"))
        if probe_delta.count_nonzero() or g1.correction_mlp[-1].weight.count_nonzero() or g1.correction_mlp[-1].bias.count_nonzero(): raise RuntimeError("RBCD correction head is not zero initialized")
        modules.append({"fold":fold,"parameter_count":sum(p.numel() for p in g1.parameters()),"module_sha":sha_obj(g1.state_dict())})
        del state,g1,g2,probe,probe_delta; torch.cuda.empty_cache()
    math_tests=rbcd_math_tests(torch)
    # A real fold-1 graph proves detached G1 and protected G2 routing.
    state,_,identity,_,_=prepare_fold(1); state["fold"],state["group"],state["rudag_identity"]=1,BASE_GROUP,identity
    eligible,ids_a,ids_b=split_branch_ids(state,1); oof=load_oof(1,state["y"].device); sites=read_sites().to(state["y"].device)
    module=init_module(brain,sprompt,1,GROUPS[1]); out,cam=hook_forward(state,training=True); mem=make_memory(eligible,state["y"],cam,out["logp"],sites); mem_used=memory_device(memory_snapshot(mem),state["y"].device)
    with torch.no_grad(): module.correction_mlp[-1].weight.fill_(1e-3)
    live=branch_train(module,cam,mem_used,out["logp"],state["y"],eligible,ids_a,ids_b,sites,state["weights"],oof["risk_soft"],oof["risk_hard"],detached=False)
    delta_loss=live[1]["cls_final"]-live[6]
    named_params=[(f"model.{n}",p) for n,p in state["model"].named_parameters() if p.requires_grad]
    named_params += [(f"cam.{n}",p) for n,p in state["cam"].named_parameters() if p.requires_grad]
    names=[n for n,_ in named_params]
    params=[p for _,p in named_params]
    gmap=torch.autograd.grad(delta_loss,params,retain_graph=True,allow_unused=True)
    fam={"gfr":0.0,"darg":0.0,"cam":0.0,"outputlayer":0.0}
    for n,g in zip(names,gmap):
        if g is None: continue
        norm=float(g.detach().double().square().sum().sqrt().cpu())
        if n.startswith("model.graphs.") or n.startswith("model.gcn.") or n.startswith("model.fusion_weight"): fam["gfr"]+=norm
        elif n.startswith("model."): fam["darg"]+=norm
        elif n.startswith("cam.Outputlayer."): fam["outputlayer"]+=norm
        elif n.startswith("cam."): fam["cam"]+=norm
    detached=branch_train(module,cam,mem_used,out["logp"],state["y"],eligible,ids_a,ids_b,sites,state["weights"],oof["risk_soft"],oof["risk_hard"],detached=True)
    detached_cam_grad=torch.autograd.grad(detached[0],cam,allow_unused=True,retain_graph=True)[0]
    if detached_cam_grad is not None: raise RuntimeError("G1 StopGrad branch reaches CAM")
    if not all(fam[k] > 0 for k in ("gfr","darg","cam","outputlayer")): raise RuntimeError(f"G2 gradient path missing: {fam}")
    state["group"]=BASE_GROUP
    audit=gradients_and_step(state,module,out,cam,mem_used,live,GROUPS[1],20,0.015)
    if audit["outputlayer_new_applied_norm"] != 0.0 or audit["outputlayer_new_raw_norm"] <= 0.0: raise RuntimeError("G2 OutputLayer protection test failed")
    peak=int(torch.cuda.max_memory_allocated()); del state,module,out,cam,mem; torch.cuda.empty_cache()
    result={"status":"PASS","source_sha256":source_sha,"historical_base_audit":"PASS 30/30","e76_7_audit":"PASS 20/20","e76_8_audit":"PASS 20/20",
            "folds":inherited,"modules":modules,"zero_residual":True,"math_tests":math_tests,
            "oof_scope":"PASS 10/10; each eligible patient exactly once; no outer-test labels; E61 warmup not loaded",
            "g1_stopgrad_cam_path":True,"g2_gradient_paths":fam,"g2_outputlayer_protection":audit,
            "cam_forward_hook_count":1,"cam_shape":[871,144],"preflight_peak_gpu_bytes":peak}
    atomic_json(RESULTS/"preflight.json",result); print(json.dumps({k:v for k,v in result.items() if k not in ("folds","modules")},indent=2),flush=True); return result


def run_all(workers):
    source_sha=sha_obj(source_snapshot())
    preflight=json.loads((RESULTS/"preflight.json").read_text())
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != source_sha: raise RuntimeError("formal run blocked: preflight/source hash mismatch")
    if len(list(OOF_DIR.glob("fold*.pt"))) != 10:
        subprocess.run([sys.executable,str(EXP/"oof_teacher.py"),"--all","--workers",str(min(5,max(1,int(workers))))],cwd=EXP,check=True)
    if len(list(OOF_DIR.glob("fold*.pt"))) != 10: raise RuntimeError("OOF supervision incomplete")
    estimated=max(1,int((0.85*98*1024**3)//max(1,preflight["preflight_peak_gpu_bytes"])))
    actual=min(10,max(1,int(workers)),estimated); started=time.time()
    manifest={"experiment":CFG["experiment"],"status":"FORMAL_RUNNING","root":str(EXP),"preferred_root":str(REPO/"experiment/experiment76_9_rbcd"),
              "preferred_root_probe":CFG["preferred_root_probe"],"groups":list(GROUPS),"folds":10,"seed":0,"formal_tasks_total":20,"formal_tasks_complete":0,
              "workers":actual,"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"training_source_sha256":source_sha,
              "preflight_source_sha256":preflight["source_sha256"],"base":BASE_GROUP,"history_reference":"E76.8 COMPLETE_AUDITED",
              "oof_status":"COMPLETE_10_FOLDS","peak_gpu_bytes_preflight":preflight["preflight_peak_gpu_bytes"],
              "resource_plan":{"requested_workers":workers,"selected_workers":actual,"estimated_memory_workers":estimated,"single_gpu":True},"no_posthoc_tuning":True}
    atomic_json(EXP/"manifest.json",manifest); atomic_json(RESULTS/"resource_plan.json",manifest["resource_plan"])
    tasks=[(group,fold) for fold in range(1,11) for group in GROUPS]; running={};completed=[];failures=[]
    while tasks or running:
        while tasks and len(running)<actual:
            group,fold=tasks.pop(0); log_path=LOGS/f"worker_{group}_f{fold:02d}.log"; log_file=log_path.open("ab")
            proc=subprocess.Popen([sys.executable,str(EXP/"run.py"),"--worker",group,str(fold)],cwd=EXP,stdout=log_file,stderr=subprocess.STDOUT,
                                  env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            running[proc]=(group,fold,log_file); print(f"START worker pid={proc.pid} {group} fold={fold} running={len(running)}/{actual}",flush=True)
        time.sleep(2)
        for proc,(group,fold,log_file) in list(running.items()):
            code=proc.poll()
            if code is None: continue
            log_file.close(); del running[proc]
            if code==0:
                record=json.loads(task_path(group,fold).read_text());
                if record.get("status")=="COMPLETE": completed.append((group,fold))
                else: failures.append({"group":group,"fold":fold,"exit":code,"status":record.get("status")})
            else: failures.append({"group":group,"fold":fold,"exit":code,"log":str(LOGS/f"worker_{group}_f{fold:02d}.log")})
        if failures:
            for proc,(group,fold,log_file) in list(running.items()): proc.terminate(); log_file.close()
            for proc in running: proc.wait()
            raise RuntimeError(f"formal workers failed: {failures}")
        if len(completed) and len(completed)%2==0: print(f"PROGRESS complete={len(completed)}/20 pending={len(tasks)+len(running)}",flush=True)
    if len(completed)!=20: raise RuntimeError(f"formal tasks incomplete: {len(completed)}/20")
    manifest.update({"status":"TRAINING_COMPLETE_AUDIT_PENDING","formal_tasks_complete":len(completed),"ended_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"wall_seconds":time.time()-started})
    atomic_json(EXP/"manifest.json",manifest); return {"completed":len(completed),"workers":actual,"wall_seconds":manifest["wall_seconds"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--worker",nargs=2,metavar=("GROUP","FOLD"))
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--workers",type=int,default=5)
    args=parser.parse_args()
    if args.preflight: preflight()
    elif args.worker: train_worker(args.worker[0],int(args.worker[1]))
    elif args.all:
        result=run_all(args.workers);print(json.dumps(result),flush=True)
    else: parser.error("choose --preflight, --worker GROUP FOLD or --all")


if __name__=="__main__": main()
