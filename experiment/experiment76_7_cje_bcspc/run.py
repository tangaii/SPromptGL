#!/usr/bin/env python3
"""Experiment 76.7 controlled end-to-end training runner."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
E72 = REPO / "experiment/experiment72_mgdss"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
GROUPS = ("G1-CJE-BCSPC-Full-E2E", "G2-CJE-BCSPC-StopGrad")
BASE_GROUP = "E74.4-G2-Positive-DCR"
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS, CACHE = (RT / key for key in ("checkpoints", "tasks", "logs", "cache"))
RESULTS = EXP / "results"
for _path in (CKPTS, TASKS, LOGS, CACHE, RESULTS):
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
        EXP / "config.json", EXP / "run.py", EXP / "cje_bcspc.py", EXP / "path_adapter.py", EXP / "audit.py",
        BASE_EXP / "run.py", BASE_EXP / "soft_positive_dcr.py", BASE_EXP / "config.json", BASE_EXP / "manifest.json",
        BASE_EXP / "audit.py", BASE_EXP / "results/audit_status.json",
        E76_6 / "run.py", E76_6 / "bcspc.py", E76_6 / "audit.py", E76_6 / "config.json", E76_6 / "manifest.json",
        E76_6 / "results/audit_status.json", E76_6 / "results/report.md",
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
    paths.extend(sorted((E76_6 / ".runtime/checkpoints").glob("G1-BCSPC-Full_f*.best.pt")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("source/history inventory incomplete: " + ", ".join(missing))
    if len(list((E61 / ".runtime/warmup").glob("fold*.pt"))) != 10 or len(list((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt"))) != 10:
        raise RuntimeError("E61/E74.4 checkpoint count is not ten folds")
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


def init_module(brain, sprompt, fold):
    from cje_bcspc import init_isolated
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


def branch_train(module, cam, memory, base_logp, labels, eligible, ids_a, ids_b, sites, weights,
                 *, detached):
    import torch
    from cje_bcspc import branch_losses
    query = torch.as_tensor(eligible, dtype=torch.long, device=cam.device)
    pieces, diagnostics = [], []
    a_set = set(ids_a)
    for query_ids, support_ids in ((ids_a, ids_b), (ids_b, ids_a)):
        qids = torch.as_tensor(query_ids, dtype=torch.long, device=cam.device)
        support = torch.as_tensor(support_ids, dtype=torch.long, device=cam.device)
        branch_cam = cam.detach() if detached else cam
        branch_base = base_logp.detach() if detached else base_logp
        final, diag = module(branch_cam, qids, memory, support, sites, branch_base)
        pieces.append((qids, final, diag))
    q_order = torch.cat([piece[0] for piece in pieces])
    final_logp = torch.cat([piece[1] for piece in pieces])
    diag = merge_diagnostics([piece[2] for piece in pieces])
    y = labels[q_order]
    loss, components = branch_losses(final_logp, diag, y, weights)
    base_score = base_logp[q_order, 0] - base_logp[q_order, 1]
    base_logp_eligible = base_logp[q_order]
    base_cls = torch.nn.functional.nll_loss(base_logp_eligible, y, weight=weights)
    return loss, components, final_logp, diag, q_order, y, base_cls


def merge_diagnostics(items):
    import torch
    scalar = {"retrieval_fallbacks", "site_cap_violations", "invalid_confusion_candidates"}
    return {key: (sum(int(item[key]) for item in items) if key in scalar
                  else torch.cat([item[key] for item in items], dim=0)) for key in items[0]}


def eval_method(module, out_logp, cam, memory, eligible, test_ids, sites, *, residual_off=False,
                boundary_off=False, semantic_swap=False, reliable_only=False):
    import torch
    ids = torch.as_tensor(test_ids, dtype=torch.long, device=out_logp.device)
    support = torch.as_tensor(eligible, dtype=torch.long, device=out_logp.device)
    final, diag = module(cam, ids, memory, support, sites, out_logp,
                         residual_off=residual_off, boundary_off=boundary_off,
                         semantic_swap=semantic_swap, reliable_only=reliable_only)
    whole = out_logp.clone()
    whole[ids] = final
    return whole, diag


def exact_metrics(logp, test_ids, labels):
    return legacy.extended_metrics(logp, test_ids, labels)


def gradients_and_step(state, module, out, cam, memory, branch_tuple, group, epoch, beta):
    import torch
    params = [parameter for parameter in
              list(state["model"].parameters()) + list(state["cam"].parameters())
              if parameter.requires_grad]
    param_owner = {id(param): param for param in params}
    optimizers = [("opt_base", state["opt_base"], 0.10),
                  ("opt_cam", state["opt_cam"], 0.20),
                  ("opt_added", state["opt_added"], 0.10)]
    name_by_id = {id(parameter): f"model.{name}" for name,parameter in state["model"].named_parameters()}
    name_by_id.update({id(parameter): f"cam.{name}" for name,parameter in state["cam"].named_parameters()})
    module_params = list(module.parameters())
    loss, components, final_logp, diag, qids, y, base_cls = branch_tuple
    base_grads = torch.autograd.grad(out["total"], params, retain_graph=True, allow_unused=True)
    branch_grads = torch.autograd.grad(loss, module_params, retain_graph=True, allow_unused=True)
    phase_b = epoch + 1 >= 21
    allow_e2e = group == GROUPS[0] and phase_b
    cls_final = components["cls_final"]
    new_grads = (torch.autograd.grad(cls_final - base_cls, params, retain_graph=False, allow_unused=True)
                 if allow_e2e else tuple(None for _ in params))
    base_by_id = {id(parameter): gradient for parameter, gradient in zip(params, base_grads)}
    new_by_id = {id(parameter): gradient for parameter, gradient in zip(params, new_grads)}
    module_grad_norms = [float(grad.detach().norm().cpu()) if grad is not None else 0.0 for grad in branch_grads]
    for parameter, gradient in zip(module_params, branch_grads):
        parameter.grad = None if gradient is None else gradient.detach().clone()
    audit = {"group": group, "epoch": epoch + 1, "beta": float(beta), "phase_b": phase_b,
             "end_to_end": allow_e2e, "optimizer_groups": {},
             "branch_gradient_norm": math.sqrt(sum(value * value for value in module_grad_norms))}
    seen, scale_by_id = set(), {}
    for name, optimizer, limit in optimizers:
        for group_index, opt_group in enumerate(optimizer.param_groups):
            selected = [parameter for parameter in opt_group["params"] if id(parameter) in param_owner]
            if any(id(parameter) in seen for parameter in selected):
                raise RuntimeError("a backbone parameter appears in multiple original optimizer groups")
            seen.update(map(id, selected))
            base_sq = new_sq = dot = 0.0
            per_parameter = []
            for parameter in selected:
                bg, ng = base_by_id[id(parameter)], new_by_id[id(parameter)]
                bn = float(bg.detach().double().square().sum().cpu()) if bg is not None else 0.0
                nnorm = float(ng.detach().double().square().sum().cpu()) if ng is not None else 0.0
                dp = float((bg.detach().double() * ng.detach().double()).sum().cpu()) if bg is not None and ng is not None else 0.0
                base_sq += bn; new_sq += nnorm; dot += dp
                per_parameter.append((parameter, bg, ng))
            base_norm, raw_new_norm = math.sqrt(base_sq), math.sqrt(new_sq)
            scaled_norm = beta * raw_new_norm
            if base_norm <= 1e-12 or scaled_norm == 0.0:
                scale = 0.0 if base_norm <= 1e-12 else 1.0
                applied_norm = 0.0 if base_norm <= 1e-12 else scaled_norm
            else:
                scale = min(1.0, limit * base_norm / scaled_norm)
                applied_norm = scaled_norm * scale
            cosine = dot / (base_norm * raw_new_norm) if base_norm > 1e-12 and raw_new_norm > 0 else 0.0
            audit["optimizer_groups"][f"{name}/{group_index}"] = {
                "base_norm": base_norm, "new_raw_norm": raw_new_norm,
                "new_beta_norm": scaled_norm, "new_applied_norm": applied_norm,
                "relative_new_to_base": applied_norm / base_norm if base_norm > 1e-12 else 0.0,
                "cosine": cosine, "cap": limit, "scale": scale,
                "clipped": bool(scale < 1.0), "base_near_zero": bool(base_norm <= 1e-12),
            }
            for parameter, bg, ng in per_parameter:
                scale_by_id[id(parameter)] = scale
                if bg is None:
                    parameter.grad = None
                else:
                    contribution = torch.zeros_like(bg) if ng is None or scale == 0.0 else ng * (beta * scale)
                    parameter.grad = bg.detach().clone() + contribution.detach()
    if seen != set(map(id, params)):
        missing = len(set(map(id, params)) - seen)
        raise RuntimeError(f"original optimizer groups do not cover {missing} backbone parameters")
    audit["new_to_base_gradient_ratio"] = {key: value["relative_new_to_base"] for key, value in audit["optimizer_groups"].items()}
    audit["clipped_groups"] = sum(value["clipped"] for value in audit["optimizer_groups"].values())
    family_gradients = {key: [] for key in ("gfr", "darg_scorer", "darg_gate", "cam", "outputlayer")}
    for parameter,bg,ng in zip(params,base_grads,new_grads):
        name=name_by_id[id(parameter)]
        family=("gfr" if name.startswith(("model.graphs.","model.gcn.","model.fusion_weight")) else
                "darg_scorer" if name.startswith("model.scorers.") else
                "darg_gate" if name.startswith(("model.edge_gate_logits","model.reliability_gate_logits","model.adaptive_gate_logits")) else
                "cam" if name.startswith("cam.") and ".Outputlayer." not in name else
                "outputlayer" if name.startswith("cam.Outputlayer.") else None)
        if family is not None:
            scale=scale_by_id.get(id(parameter),0.0)
            reference = ng if ng is not None else bg
            applied = (torch.zeros_like(reference) if reference is not None and (ng is None or scale==0.0)
                       else None if reference is None else ng*(beta*scale))
            family_gradients[family].append((bg,ng,applied))
    audit["gradient_families"]={}
    for family,entries in family_gradients.items():
        if not entries: continue
        base_norm=math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g,_,_ in entries if g is not None))
        raw_norm=math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for _,g,_ in entries if g is not None))
        applied_norm=math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for _,_,g in entries if g is not None))
        dot=sum(float((base.detach().double()*new.detach().double()).sum().cpu())
                for base,_,new in entries if base is not None and new is not None)
        audit["gradient_families"][family]={"base_norm":base_norm,"new_raw_norm":raw_norm,
            "new_applied_norm":applied_norm,"ratio":applied_norm/base_norm if base_norm>1e-12 else 0.0,
            "cosine_base_vs_applied":dot/(base_norm*applied_norm) if base_norm>1e-12 and applied_norm>0 else 0.0}
    return audit


def parameter_norms(state):
    output={}
    for module_name,module in (("model",state["model"]),("cam",state["cam"])):
        for name,parameter in module.named_parameters():
            full=f"{module_name}.{name}"
            family=("gfr" if full.startswith(("model.graphs.","model.gcn.","model.fusion_weight")) else
                    "darg_scorer" if full.startswith("model.scorers.") else
                    "darg_gate" if full.startswith(("model.edge_gate_logits","model.reliability_gate_logits","model.adaptive_gate_logits")) else
                    "cam" if full.startswith("cam.") and ".Outputlayer." not in full else
                    "outputlayer" if full.startswith("cam.Outputlayer.") else "other")
            output.setdefault(family,{})[full]=parameter.detach().clone()
    return output


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
                  base_ledger, warm_sha):
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
            "gradient_snapshot": copy.deepcopy(grad_audit), "base_ledger": copy.deepcopy(base_ledger),
            "fold_warmup_sha256": warm_sha, "state_identity": state["rudag_identity"]}


def _atomic_best(group, fold, identity, warm_sha, source_sha, payload, trace, epochs, history_task):
    record = _checkpoint_payload(group, fold, identity, warm_sha, source_sha, payload, trace, epochs, history_task)
    atomic_torch(best_path(group, fold), record)
    return record


def _save_latest(group, fold, identity, warm_sha, source_sha, state, module, optimizer,
                 memory, epoch, wait, best_acc, best, trace, base_ledger, gradient_audit,
                 history_task, started):
    payload = {"group": group, "fold": fold, "source_sha256": source_sha,
               "identity": identity, "warmup_sha256": warm_sha,
               "base_state": legacy.load_base().load_history().state_bundle(state),
               "module": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
               "optimizer": copy.deepcopy(optimizer.state_dict()), "memory": memory_snapshot(memory),
               "next_epoch": epoch, "wait": wait, "best_acc": best_acc, "best": best,
               "rng": legacy.history_runtime().pack_rng(), "trace": trace,
               "base_ledger": base_ledger, "gradient_audit": gradient_audit,
               "history_task": history_task, "started_wall_time": started}
    atomic_torch(latest_path(group, fold), payload)


def train_worker(group, fold):
    import numpy as np
    import torch
    from cje_bcspc import branch_losses
    if group not in GROUPS or fold not in range(1,11):
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
    sites = read_sites().to(labels.device)
    brain, sprompt = prompt_vectors()
    module = init_module(brain, sprompt, fold)
    branch_opt = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    rng = legacy.history_runtime()
    if rng.rng_sha(rng.pack_rng()) != warm["rng_sha"]:
        raise RuntimeError("new module/optimizer initialization contaminated warmup RNG")
    task_file = task_path(group, fold)
    if task_file.exists():
        existing = json.loads(task_file.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("source_sha256") == source_sha and existing.get("checkpoint_sha256") == sha_file(best_path(group, fold)):
            print(f"SKIP complete {group} fold={fold}", flush=True)
            return existing
        if existing.get("status") not in ("RUNNING", "FAILED"):
            raise RuntimeError(f"refuse incompatible task overwrite: {task_file}")

    memory, trace, gradient_audit = None, [], []
    base_wait, base_best_acc, base_best_epoch, base_best_state = 0, -1.0, -1, None
    method_wait, best_acc, best_epoch, best = 0, -1.0, -1, None
    start_epoch = 10
    started = time.time()
    if latest_path(group, fold).exists():
        saved = torch_load(latest_path(group, fold))
        for key, expected in (("group", group), ("fold", fold), ("source_sha256", source_sha),
                              ("identity", identity), ("warmup_sha256", identity["warmup_sha256"])):
            if saved.get(key) != expected:
                raise RuntimeError(f"latest recovery identity differs: {key}")
        legacy.load_base().load_history().load_bundle(state, saved["base_state"])
        module.load_state_dict(saved["module"], strict=True)
        branch_opt.load_state_dict(saved["optimizer"])
        rng.restore_rng(saved["rng"])
        memory = memory_device(saved["memory"], labels.device)
        start_epoch, method_wait = int(saved["next_epoch"]), int(saved["wait"])
        best_acc, best_epoch, best = float(saved["best_acc"]), int(saved["best"]["epoch"]) if saved["best"] else -1, saved["best"]
        trace, base_wait = saved["trace"], saved["base_ledger"]["wait"]
        base_best_acc, base_best_epoch = saved["base_ledger"]["best_acc"], saved["base_ledger"]["best_epoch"]
        base_best_state = saved["base_ledger"].get("state")
        gradient_audit, started = saved["gradient_audit"], float(saved["started_wall_time"])
    elif task_file.exists():
        task_file.unlink()
    if start_epoch < 10 or start_epoch > 450:
        raise RuntimeError("invalid next epoch")
    task_record = {"status":"RUNNING", "group":group, "fold":fold, "source_sha256":source_sha,
                   "identity":identity, "warmup_sha256":identity["warmup_sha256"],
                   "support_A":ids_a, "support_B":ids_b, "eligible_train_ids":eligible,
                   "next_epoch":start_epoch}
    atomic_json(task_file, task_record)

    hist = legacy.load_base().load_history()
    hist.zero_optimizers(state)
    torch.cuda.reset_peak_memory_stats()
    final_metrics = None
    stopped_epoch = start_epoch
    for epoch in range(start_epoch, 450):
        stopped_epoch = epoch + 1
        rng_before = rng.rng_sha(rng.pack_rng())
        hist.zero_optimizers(state)
        branch_opt.zero_grad(set_to_none=True)
        out, train_cam = hook_forward(state, training=True)
        if not torch.isfinite(out["total"]):
            raise FloatingPointError(f"nonfinite base loss {group} fold={fold} epoch={epoch+1}")
        # Epoch 11 has no prior eval snapshot. Keep the branch inactive until
        # the ordinary eval forward creates the first support memory.
        if memory is None:
            memory = make_memory([], labels, train_cam, out["logp"], sites)
        memory_used = memory_snapshot(memory)
        memory_cuda = memory_device(memory_used, labels.device)
        branch_active = int(memory_used["updates"]) > 0
        live = group == GROUPS[0] and epoch + 1 >= 21
        branch_result = branch_train(module, train_cam, memory_cuda, out["logp"], labels,
                                     eligible, ids_a, ids_b, sites, state["weights"],
                                     detached=not live)
        branch_loss, components, final_train_logp, diag, qids, branch_labels, base_cls = branch_result
        # In epoch 11..20 and in StopGrad, the backbone receives only L_base.
        # In G1 phase B add the explicitly differentiated final-minus-base NLL.
        beta = (0.3 * min(1.0, max(0.0, (epoch + 1 - 20) / 20.0))) if live else 0.0
        grad_audit_item = gradients_and_step(state, module, out, train_cam, memory_cuda,
                                             branch_result, group, epoch, beta)
        if epoch + 1 in (11, 12, 20, 21, 22, 40, 100, 300, 449):
            gradient_audit.append(grad_audit_item)
        before_norms = parameter_norms(state)
        for key in ("opt_base", "opt_cam", "opt_added"):
            for param_group in state[key].param_groups:
                for parameter in param_group["params"]:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"nonfinite inherited gradient {key}")
            state[key].step()
        for parameter in module.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("nonfinite BCSPC gradient")
        if branch_active:
            branch_opt.step()
        grad_audit_item["parameter_delta_norms"] = parameter_change_norms(state,before_norms)
        grad_audit_item["branch_loss_components"] = {key:float(value.detach().cpu()) for key,value in components.items()}
        rng_after_train = rng.rng_sha(rng.pack_rng())

        with torch.no_grad():
            evaluated, eval_cam = hook_forward(state, training=False)
            base_metrics = exact_metrics(evaluated["logp"], list(map(int,state["test_np"])), labels)
            final_logp, eval_diag = eval_method(module, evaluated["logp"], eval_cam, memory_cuda,
                                                eligible, list(map(int,state["test_np"])), sites)
            metrics = exact_metrics(final_logp, list(map(int,state["test_np"])), labels)
            # The checkpoint stores memory_used, the snapshot that generated
            # this prediction; this update belongs to the next epoch only.
            next_memory = make_memory(eligible, labels, eval_cam, evaluated["logp"], sites, memory)
        rng_after_eval = rng.rng_sha(rng.pack_rng())
        rng_row = {"epoch":epoch+1, "rng_before":rng_before,
                   "rng_after_train":rng_after_train, "rng_after_eval":rng_after_eval}
        trace.append(rng_row)

        if float(base_metrics["acc"]) > base_best_acc:
            base_best_acc, base_best_epoch, base_wait = float(base_metrics["acc"]), epoch+1, 0
            base_best_state = hist.state_bundle(state)
        else:
            base_wait += 1
        if float(metrics["acc"]) > best_acc:
            best_acc, best_epoch, method_wait = float(metrics["acc"]), epoch+1, 0
            best = _capture_best(state,module,branch_opt,memory_used,eligible,ids_a,ids_b,epoch+1,
                                 metrics,evaluated["logp"],final_logp,eval_diag,components,
                                 grad_audit_item,{"best_acc":base_best_acc,"best_epoch":base_best_epoch,"wait":base_wait},
                                 identity["warmup_sha256"])
            record = _atomic_best(group,fold,identity,identity["warmup_sha256"],source_sha,
                                  best,trace,epoch+1,history_task)
            task_record["best_checkpoint_sha256"] = sha_file(best_path(group,fold))
        else:
            method_wait += 1
        memory = next_memory
        final_metrics = metrics
        # Atomic latest every epoch so restart replays exactly the next step.
        _save_latest(group,fold,identity,identity["warmup_sha256"],source_sha,state,module,
                     branch_opt,memory,epoch+1,method_wait,best_acc,best,trace,
                     {"best_acc":base_best_acc,"best_epoch":base_best_epoch,"wait":base_wait,
                      "state":base_best_state},
                     gradient_audit,history_task,started)
        task_record.update({"next_epoch":epoch+1,"base_best_acc":base_best_acc,
                            "base_best_epoch":base_best_epoch,"base_wait":base_wait,
                            "method_best_acc":best_acc,"method_best_epoch":best_epoch,
                            "method_wait":method_wait,"rng_records":len(trace),
                            "memory_updates":memory["updates"],"peak_gpu_bytes":int(torch.cuda.max_memory_allocated())})
        atomic_json(task_file,task_record)
        if (epoch+1)%10==0 or epoch in (10,19,20,21,39,99,299,448):
            record={"group":group,"fold":fold,"epoch":epoch+1,
                    "base_acc":float(base_metrics["acc"]),"base_best_acc":base_best_acc,
                    "base_best_epoch":base_best_epoch,"base_wait":base_wait,
                    "method_acc":float(metrics["acc"]),"method_best_acc":best_acc,
                    "method_best_epoch":best_epoch,"beta":beta,
                    "branch_loss":float(branch_loss.detach().cpu()),
                    "base_loss":float(out["total"].detach().cpu()),
                    "delta_abs_mean":float(eval_diag["delta"].abs().mean().cpu()),
                    "residual_activation_abs_mean":float(eval_diag["activation"].abs().mean().cpu()),
                    "g1_new_gradient":grad_audit_item,"rng":rng_row,
                    "memory_updates":memory["updates"],"peak_gpu_bytes":int(torch.cuda.max_memory_allocated())}
            log=LOGS/f"{group}_f{fold:02d}.jsonl";log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
            print(f"{group} fold={fold} epoch={epoch+1} base={base_best_acc:.6f}@{base_best_epoch} method={best_acc:.6f}@{best_epoch} beta={beta:.3f}",flush=True)
        del out, train_cam, evaluated, eval_cam, final_logp
        if base_wait > 200:
            break

    if best is None:
        raise RuntimeError(f"no method best checkpoint selected: {group} fold={fold}")
    if group == GROUPS[1]:
        expected_trace={int(row["epoch"]):row for row in history_task["rng_trace"]}
        actual_trace={int(row["epoch"]):row for row in trace}
        common=set(expected_trace)&set(actual_trace)
        mismatches=[epoch for epoch in common if expected_trace[epoch]!=actual_trace[epoch]]
        if mismatches or len(common)!=len(trace):
            raise RuntimeError(f"StopGrad RNG diverged from E74.4-G2 fold={fold} epochs={mismatches[:5]}")
        hist_checkpoint=torch_load(history_best)
        if base_best_epoch!=history_task["best_epoch_1based"] or trace[-1]["epoch"]!=history_task["epochs_executed"]:
            raise RuntimeError(f"StopGrad backbone ACC/early stop did not reproduce history fold={fold}")
        if hist_checkpoint.get("identity")!=identity:
            raise RuntimeError("historical E74.4-G2 checkpoint identity mismatch")
        if base_best_state is None:
            raise RuntimeError("StopGrad best-base parameter/optimizer state was not retained")
        _assert_state_equal(hist_checkpoint["best_state"],base_best_state,
                            f"StopGrad historical base best checkpoint fold={fold}")
    executed_epoch=trace[-1]["epoch"]
    record=_atomic_best(group,fold,identity,identity["warmup_sha256"],source_sha,best,trace,executed_epoch,history_task)
    task_record.update({"status":"COMPLETE","best_epoch_1based":best_epoch,
                        "epochs_executed":executed_epoch,"metrics":best["metrics"],
                        "checkpoint_sha256":sha_file(best_path(group,fold)),
                        "wall_seconds":time.time()-started,
                        "peak_gpu_bytes":int(torch.cuda.max_memory_allocated()),
                        "device":torch.cuda.get_device_name(0),
                        "base_best_epoch_1based":base_best_epoch,
                        "rng_mismatches":0 if group==GROUPS[1] else None})
    atomic_json(task_file,task_record)
    print(f"COMPLETE {group} fold={fold} best={best_acc:.6f}@{best_epoch} epochs={len(trace)}",flush=True)
    return task_record


def preflight():
    import torch
    from cje_bcspc import branch_losses
    # The old independent audit verifies the actual E76.6 equations and
    # checkpoint/prototype provenance; require its final artifact unchanged.
    e76_manifest=json.loads((E76_6/"manifest.json").read_text())
    e76_audit=json.loads((E76_6/"results/audit_status.json").read_text())
    if e76_manifest.get("status")!="COMPLETE_AUDITED" or e76_audit.get("status")!="COMPLETE_AUDITED" or e76_audit.get("checkpoint_recalculations")!=20:
        raise RuntimeError("E76.6 complete method/checkpoint audit is unavailable")
    base_audit=json.loads((BASE_EXP/"results/audit_status.json").read_text())
    if base_audit.get("status")!="PASS" or base_audit.get("audited_checkpoints")!=30:
        raise RuntimeError("E74.4-G2 historical independent audit is not PASS")
    sources=source_snapshot()
    inherited=[]
    for fold in range(1,11):
        state,warm,identity,task,path=prepare_fold(fold)
        hist_best=torch_load(path)
        if task.get("status")!="COMPLETE" or hist_best.get("best_epoch_1based")!=task.get("best_epoch_1based"):
            raise RuntimeError(f"historical base best checkpoint/task mismatch fold={fold}")
        if state["train_pairs_np"].shape!=(4096,2):
            raise RuntimeError(f"4096 patient relation pairs absent fold={fold}")
        if identity["warmup_sha256"]!=sha_file(E61/".runtime/warmup"/f"fold{fold:02d}.pt"):
            raise RuntimeError(f"E61 warmup mismatch fold={fold}")
        inherited.append({"fold":fold,"warmup_sha256":identity["warmup_sha256"],
                          "initial_model_sha":identity["model_cam_sha"],"optimizer_sha":identity["optimizer_sha"],
                          "split_sha":identity["split_sha"],"pair_sha":identity["pair_sha"],
                          "relation_pairs":4096,"overlap":identity["membership"]["train_test_overlap_1based"],
                          "history_best_epoch":task["best_epoch_1based"],"history_executed":task["epochs_executed"]})
        del state
        torch.cuda.empty_cache()
    # Formal G1/G2 initialization and zero residual for all folds.
    brain,sprompt=prompt_vectors()
    modules_sha=[]
    peak=0
    for fold in range(1,11):
        state,warm,identity,task,path=prepare_fold(fold)
        before=legacy.history_runtime().rng_sha(legacy.history_runtime().pack_rng())
        module=init_module(brain,sprompt,fold)
        probe=torch.zeros((1,87),device="cuda")
        delta=(torch.ones(1,device="cuda")+.5)*torch.tanh(module.residual_network(probe).squeeze(-1))
        if not torch.equal(delta,torch.zeros_like(delta)):
            raise RuntimeError(f"zero residual precondition failed fold={fold}")
        if legacy.history_runtime().rng_sha(legacy.history_runtime().pack_rng())!=before:
            raise RuntimeError(f"module init consumed backbone RNG fold={fold}")
        if module.residual_network[-1].weight.count_nonzero() or module.residual_network[-1].bias.count_nonzero():
            raise RuntimeError("residual final layer must initialize to zero")
        modules_sha.append({"fold":fold,"parameter_count":sum(p.numel() for p in module.parameters()),
                            "module_sha":legacy.load_base().sha_object(module.state_dict())})
        del state,module,probe,delta
        torch.cuda.empty_cache()
    # Gradient-difference identity and staged gradient routing on a differentiable
    # synthetic boundary around actual softmax/NLL score transformation.
    raw=torch.tensor([[0.8,-0.8],[-0.3,0.3]],device="cuda",requires_grad=True)
    delta=torch.zeros(2,device="cuda",requires_grad=True)
    base=torch.log_softmax(raw,dim=-1)
    from cje_bcspc import correct_logp
    final=correct_logp(base,delta)
    y=torch.tensor([0,1],device="cuda");w=torch.tensor([1.2,0.8],device="cuda")
    lb=torch.nn.functional.nll_loss(base,y,weight=w)
    lf=torch.nn.functional.nll_loss(final,y,weight=w)
    gbase=torch.autograd.grad(lb,raw,retain_graph=True)[0]
    gdiff=torch.autograd.grad(lf-lb,raw,retain_graph=True)[0]
    if gdiff.norm().item()>1e-6 or not torch.allclose(gbase,gbase+0*gdiff,atol=0,rtol=0):
        raise RuntimeError("zero-delta final-minus-base classification gradient is not zero")
    gd=torch.autograd.grad(lf,delta)[0]
    if gd.abs().max().item()<=0 or not torch.isfinite(gd).all():
        raise RuntimeError("final classification loss does not train the residual")
    # A real first update from fold 1 proves that branch bookkeeping and
    # gradients do not perturb the original StopGrad backbone/RNG trajectory.
    state,warm,identity,history_task,history_path=prepare_fold(1)
    hist=legacy.load_base().load_history();rng=legacy.history_runtime()
    sites=read_sites().to(state["y"].device)
    eligible,ids_a,ids_b=split_branch_ids(state,1)
    module_g2=init_module(brain,sprompt,1)
    optimizer_g2=torch.optim.AdamW(module_g2.parameters(),lr=0.0005,weight_decay=0.01)
    initial_bundle=hist.state_bundle(state);initial_rng=rng.pack_rng()
    hist.zero_optimizers(state)
    reference_train,reference_cam=hook_forward(state,training=True)
    reference_train["total"].backward()
    for key in ("opt_base","opt_cam","opt_added"): state[key].step()
    reference_bundle=hist.state_bundle(state)
    reference_rng_train=rng.rng_sha(rng.pack_rng())
    with torch.no_grad(): reference_eval,_=hook_forward(state,training=False)
    reference_eval_logp=reference_eval["logp"].detach().clone()
    reference_rng_eval=rng.rng_sha(rng.pack_rng())
    expected_epoch11=next(row for row in history_task["rng_trace"] if row["epoch"]==11)
    if (reference_rng_train!=expected_epoch11["rng_after_train"] or
        reference_rng_eval!=expected_epoch11["rng_after_eval"]):
        raise RuntimeError("preflight base-only forward/RNG does not replay historical epoch 11")
    hist.load_bundle(state,initial_bundle);rng.restore_rng(initial_rng)
    hist.zero_optimizers(state);optimizer_g2.zero_grad(set_to_none=True)
    actual_train,actual_cam=hook_forward(state,training=True)
    memory=make_memory(eligible,state["y"],actual_cam,actual_train["logp"],sites)
    g2_branch=branch_train(module_g2,actual_cam,memory,actual_train["logp"],state["y"],
                           eligible,ids_a,ids_b,sites,state["weights"],detached=True)
    g2_audit=gradients_and_step(state,module_g2,actual_train,actual_cam,memory,g2_branch,
                                GROUPS[1],10,0.0)
    for key in ("opt_base","opt_cam","opt_added"): state[key].step()
    optimizer_g2.step()
    actual_bundle=hist.state_bundle(state)
    _assert_state_equal(reference_bundle,actual_bundle,"fold1 G2 one-step backbone/optimizer")
    actual_rng_train=rng.rng_sha(rng.pack_rng())
    if actual_rng_train!=reference_rng_train:
        raise RuntimeError("G2 branch changed the inherited train RNG trajectory")
    with torch.no_grad(): actual_eval,_=hook_forward(state,training=False)
    if not torch.equal(reference_eval_logp,actual_eval["logp"]):
        raise RuntimeError("G2 first-step original predictions differ from base-only reference")
    actual_rng_eval=rng.rng_sha(rng.pack_rng())
    if actual_rng_eval!=reference_rng_eval or actual_rng_eval!=expected_epoch11["rng_after_eval"]:
        raise RuntimeError("G2 branch changed the inherited eval RNG trajectory")
    g2_equivalence={"passed":True,"fold":1,"epoch":11,"model_cam_optimizer_exact":True,
                    "base_train_prediction_exact":True,"rng_train_exact":True,"rng_eval_exact":True,
                    "historical_rng_epoch11":expected_epoch11}
    del state,module_g2,optimizer_g2,reference_train,reference_cam,actual_train,actual_cam
    torch.cuda.empty_cache()
    # Data-path preflight: one exact base train forward, dynamic-memory bootstrap,
    # cross-half live prototype values, and real G1 gradients into DARG/GFR/CAM.
    state,warm,identity,task,path=prepare_fold(1)
    sites=read_sites().to(state["y"].device)
    eligible,ids_a,ids_b=split_branch_ids(state,1)
    module=init_module(brain,sprompt,1)
    out,cam=hook_forward(state,training=True)
    memory=make_memory(eligible,state["y"],cam,out["logp"],sites)
    # A nonzero residual head is used only for gradient reachability diagnostics;
    # the formal initializer remains zero and is checked above for every fold.
    with torch.no_grad(): module.residual_network[-1].weight.fill_(1e-3)
    live,components,final_logp,diag,qids,ys,base_cls=branch_train(module,cam,memory,out["logp"],state["y"],
                        eligible,ids_a,ids_b,sites,state["weights"],detached=False)
    # Check the live candidate values connect selected support rows, query CAM and
    # every required backbone family. This is diagnostic only; no optimizer step.
    keys=[("cam144",cam),("boundary",diag["final_boundary_evidence"]),
          ("semantic",module.semantic_projection[0].weight),
          ("reliable_role",module.reliable_text_projection.weight),
          ("boundary_role",module.boundary_text_projection.weight),
          ("modality_gate",module.modality_gate[0].weight),
          ("residual",module.residual_network[0].weight)]
    paths={}
    for name,tensor in keys:
        gradient=torch.autograd.grad(live,tensor,retain_graph=True,allow_unused=True)[0]
        paths[name]=bool(gradient is not None and torch.isfinite(gradient).all() and gradient.norm().item()>0)
    backbone={"gfr":[],"darg":[],"cam":[]}
    family_params={"gfr":[p for n,p in state["model"].named_parameters() if n.startswith(("graphs.","gcn.","fusion_weight"))],
                   "darg":[p for n,p in state["model"].named_parameters() if n.startswith(("encoders.","scorers.","discriminator.","adaptive_gate_logits"))],
                   "cam":list(state["cam"].parameters())}
    delta_loss=components["cls_final"]-base_cls
    for family,params in family_params.items():
        grads=torch.autograd.grad(delta_loss,params,retain_graph=True,allow_unused=True)
        norm=math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in grads if g is not None))
        backbone[family]=norm
    if not all(paths.values()) or any(value<=0 or not math.isfinite(value) for value in backbone.values()):
        raise RuntimeError(f"controlled E2E preflight did not connect required parameters: {paths}, {backbone}")
    # Phase A / StopGrad must route no new branch gradient to the backbone.
    detached,det_components,_,_,_,_,base_cls_det=branch_train(module,cam,memory,out["logp"],state["y"],
                          eligible,ids_a,ids_b,sites,state["weights"],detached=True)
    detached_cam_grad=torch.autograd.grad(detached,cam,retain_graph=True,allow_unused=True)[0]
    if detached_cam_grad is not None:
        raise RuntimeError("Phase A/StopGrad branch unexpectedly reaches the CAM backbone")
    joint_audit=gradients_and_step(state,module,out,cam,memory,
                    (live,components,final_logp,diag,qids,ys,base_cls),GROUPS[0],20,0.015)
    for name,record in joint_audit["optimizer_groups"].items():
        if record["relative_new_to_base"]>record["cap"]+1e-10:
            raise RuntimeError(f"controlled gradient cap violation in {name}")
    for key in ("gfr","darg_scorer","darg_gate","cam","outputlayer"):
        if key not in joint_audit["gradient_families"]:
            raise RuntimeError(f"required gradient family is not recorded: {key}")
    # Module's final layer zero means delta is zero but score gradient into module
    # should be observable after a nonzero final-layer probe.
    peak=torch.cuda.max_memory_allocated()
    preflight_result={"status":"PASS","source_sha256":sha_obj(sources),"source_files":sources,
        "base_path":str(BASE_EXP),"historical_base_audit":"PASS 30/30",
        "e76_6_path":str(E76_6),"e76_6_audit":"COMPLETE_AUDITED 20/20",
        "folds":inherited,"module_initialization":modules_sha,
        "zero_residual":True,"zero_delta_final_minus_base_gradient_norm":float(gdiff.norm().cpu()),
        "residual_classification_gradient_norm":float(gd.norm().cpu()),
        "controlled_e2e_gradient_paths":paths,"backbone_gradient_increment_norms":backbone,
        "phase_a_stop_gradient":True,"fold1_live_prototype_graph":True,
        "stopgrad_one_step_equivalence":g2_equivalence,
        "controlled_gradient_limit_test":joint_audit,
        "cam_forward_hook_count":1,"cam_shape":[871,144],
        "preflight_peak_gpu_bytes":int(peak),"text_vector_source_verified":True}
    atomic_json(RESULTS/"preflight.json",preflight_result)
    del state,module,out,cam,memory,live,final_logp,diag
    torch.cuda.empty_cache()
    print(json.dumps({k:v for k,v in preflight_result.items() if k not in ("source_files","folds","module_initialization")},indent=2),flush=True)
    return preflight_result


def run_all(workers):
    source_sha=sha_obj(source_snapshot())
    preflight=json.loads((RESULTS/"preflight.json").read_text())
    if preflight.get("status")!="PASS" or preflight.get("source_sha256")!=source_sha:
        raise RuntimeError("formal run blocked: preflight/source hash mismatch")
    estimated=max(1,int((0.85*98*1024**3)//max(1,preflight["preflight_peak_gpu_bytes"])))
    actual=min(10,max(1,workers),estimated)
    atomic_json(RESULTS/"resource_plan.json",{"requested_workers":workers,"selected_workers":actual,
        "preflight_peak_gpu_bytes":preflight["preflight_peak_gpu_bytes"],"gpu_total_bytes":98*1024**3,
        "single_gpu":True,"schedule":"independent group-fold workers; no mutable parameter or memory sharing"})
    tasks=[(group,fold) for fold in range(1,11) for group in GROUPS]
    running={};completed=[];failures=[]
    while tasks or running:
        while tasks and len(running)<actual:
            group,fold=tasks.pop(0)
            log_path=LOGS/f"worker_{group}_f{fold:02d}.log"
            log_file=log_path.open("ab")
            proc=subprocess.Popen([sys.executable,str(EXP/"run.py"),"--worker",group,str(fold)],
                                  cwd=EXP,stdout=log_file,stderr=subprocess.STDOUT,
                                  env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1",
                                       "MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            running[proc]=(group,fold,log_file)
            print(f"START worker pid={proc.pid} {group} fold={fold} running={len(running)}/{actual}",flush=True)
        time.sleep(2)
        for proc,(group,fold,log_file) in list(running.items()):
            code=proc.poll()
            if code is None: continue
            log_file.close();del running[proc]
            if code==0:
                record=json.loads(task_path(group,fold).read_text())
                if record.get("status")!="COMPLETE": failures.append({"group":group,"fold":fold,"exit":code,"status":record.get("status")})
                else: completed.append((group,fold))
            else:
                failures.append({"group":group,"fold":fold,"exit":code,"log":str(LOGS/f"worker_{group}_f{fold:02d}.log")})
                # Stop scheduling new folds after an error; preserve finished work.
                tasks=[]
        if failures:
            for proc,(group,fold,log_file) in running.items():
                proc.terminate()
                log_file.close()
            for proc in running: proc.wait()
            running.clear()
            break
        if len(completed)%2==0 and completed:
            print(f"PROGRESS complete={len(completed)}/20 pending={len(tasks)+len(running)}",flush=True)
    if failures:
        raise RuntimeError(f"formal workers failed: {failures}")
    if len(completed)!=20:
        raise RuntimeError(f"formal tasks incomplete: {len(completed)}/20")
    return {"completed":len(completed),"workers":actual}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--worker",nargs=2,metavar=("GROUP","FOLD"))
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--workers",type=int,default=10)
    args=parser.parse_args()
    if args.preflight: preflight()
    elif args.worker: train_worker(args.worker[0],int(args.worker[1]))
    elif args.all:
        result=run_all(args.workers);print(json.dumps(result),flush=True)
    else: parser.error("choose --preflight, --worker GROUP FOLD or --all")


if __name__=="__main__": main()
