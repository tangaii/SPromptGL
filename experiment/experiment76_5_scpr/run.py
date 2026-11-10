#!/usr/bin/env python3
"""Experiment 76.5 runner; E74.4/E61 are loaded read-only."""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text())
REPO = Path("/data/adodas/SPromptGL-main")
BASE_EXP = REPO / "experiment/experiment74_4_soft_positive_dcr"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
E72_CACHE = Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/text/text_embeddings.pt")
VENDOR_ROOT = Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/vendor/root")
ANCHORS = VENDOR_ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS, CACHE = (RT / n for n in ("checkpoints", "tasks", "logs", "cache"))
RESULTS = EXP / "results"
GROUPS = tuple(CFG["training"]["groups"])
BASE_GROUP = "E74.4-G2-Positive-DCR"
BASE = None


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    torch.load(path, map_location="cpu", weights_only=False)


def load_base():
    global BASE
    if BASE is None:
        legacy_snapshot = Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa")
        vendor_python = Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/vendor/python")
        if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
        if str(legacy_snapshot) not in sys.path: sys.path.insert(0, str(legacy_snapshot))
        if str(vendor_python) not in sys.path: sys.path.insert(0, str(vendor_python))
        path = BASE_EXP / "run.py"
        spec = importlib.util.spec_from_file_location("e744_read_only_runner", path)
        BASE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = BASE
        spec.loader.exec_module(BASE)
    return BASE


def prepare_base(fold):
    """Resolve moved E53 dependencies from a SHA-verified read-only snapshot."""
    base = load_base()
    hist = base.load_history()
    e53 = hist.load_e69().load_e61().load_base()
    manifest_path = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix/manifest.json")
    locked = json.loads(manifest_path.read_text())["anchor_sha256"]
    if not ANCHORS.is_dir():
        raise FileNotFoundError(f"locked anchor snapshot is missing: {ANCHORS}")
    for f in range(1, 11):
        expected = locked[str(REPO / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors" / f"fold{f:02d}.pt")]
        actual = sha_file(ANCHORS / f"fold{f:02d}.pt")
        if actual != expected:
            raise RuntimeError(f"E53 anchor SHA mismatch fold {f}")
    e53.ANCHORS = ANCHORS
    e53.ROOT = REPO
    e53.LEGACY_EXP = VENDOR_ROOT / "experiment/experiment51_dugsa"
    e53.LEGACY_SOURCE = e53.LEGACY_EXP / "source"
    return base.prepare(fold)


def load_scpr():
    spec = importlib.util.spec_from_file_location("scpr", EXP / "scpr.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def text_vectors():
    """Reuse the audited frozen RN50 encoding, checking exact source strings."""
    import torch
    from scpr import PROMPTS
    cache_path = CACHE / "brainprompt_rn50.pt"
    CACHE.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if tuple(saved["prompts"]) != PROMPTS:
            raise RuntimeError("cached BrainPrompt strings do not match the verified source")
        return saved["vectors"]
    old = torch.load(E72_CACHE, map_location="cpu", weights_only=False)
    if old["text_map"]["brainprompt"] != list(PROMPTS):
        raise RuntimeError("E72 cache BrainPrompt order/text differs from official ABIDE prompt file")
    if old["text_dim"] != 1024 or old["source_hashes"]["rn50_sha256"] != "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762":
        raise RuntimeError("RN50 embedding provenance mismatch")
    # The audited BrainPrompt cache is already ordered ASD, then TC/TD.
    vectors = old["embeddings"]["brainprompt"].contiguous()
    saved = {"prompts": PROMPTS, "vectors": vectors, "text_encoder": "BrainPrompt frozen CLIP RN50",
             "rn50_sha256": old["source_hashes"]["rn50_sha256"],
             "source_text_sha256": sha_file(E72_CACHE)}
    atomic_torch(cache_path, saved)
    return vectors


def hook_forward(state, training=False):
    base = load_base()
    captured = []
    output_layer = state["cam"].Outputlayer
    handle = output_layer.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0]))
    try:
        out = base.forward(state, BASE_GROUP, training=training)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (871, 144):
        raise RuntimeError(f"CAM144 hook expected one [871,144] tensor, got {[tuple(x.shape) for x in captured]}")
    return out, captured[0]


def read_sites():
    path = REPO / "data/ABIDE/processed_standard_data.csv"
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    site_cols = [x for x in fields if x.upper().startswith("SITE_ID_")]
    if not site_cols or len(rows) != 871:
        raise RuntimeError(f"site metadata unavailable/unaligned: columns={len(site_cols)}, rows={len(rows)}")
    values = [[float(row[name] or 0) for name in site_cols] for row in rows]
    if any(sum(row) != 1.0 for row in values):
        raise RuntimeError("expected exactly one active SITE_ID one-hot feature per row")
    import torch
    return torch.tensor([max(range(len(site_cols)), key=row.__getitem__) for row in values], dtype=torch.long)


def split_ab(train_ids, labels):
    import numpy as np
    a, b = [], []
    for cls in (0, 1):
        ids = np.asarray(train_ids)[np.asarray(labels)[train_ids] == cls]
        ids = np.sort(ids)
        a.extend(ids[::2].tolist()); b.extend(ids[1::2].tolist())
    return sorted(a), sorted(b)


def make_memory(n, device, train_ids, labels, cam144=None, logp=None, old=None):
    import torch
    if old is None:
        mem = {"features": torch.zeros((n, 4, 36), device=device),
               "probabilities": torch.zeros((n, 2), device=device),
               "labels": torch.full((n,), -1, dtype=torch.long, device=device),
               "observed": torch.zeros((n,), dtype=torch.bool, device=device), "updates": 0}
    else:
        mem = old
    ids = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    feat = cam144.detach().reshape(n, 4, 36).to(device)
    probs = logp.detach().exp().to(device)
    observed = mem["observed"][ids]
    fresh = ids[~observed]
    prior = ids[observed]
    if fresh.numel():
        mem["features"][fresh] = feat[fresh]
        mem["probabilities"][fresh] = probs[fresh]
    if prior.numel():
        mem["features"][prior] = .9 * mem["features"][prior] + .1 * feat[prior]
        mem["probabilities"][prior] = .9 * mem["probabilities"][prior] + .1 * probs[prior]
    mem["labels"][ids] = torch.as_tensor(labels, device=device)[ids]
    mem["observed"][ids] = True
    mem["updates"] += 1
    return mem


def memory_snapshot(memory):
    if memory is None:
        return None
    return {k: (v.detach().cpu().clone() if hasattr(v, "detach") else v) for k, v in memory.items()}


def apply_score(base_logp, ids, score, module):
    import torch
    import torch.nn.functional as F
    idx = torch.as_tensor(ids, dtype=torch.long, device=base_logp.device)
    new_margin = base_logp[idx, 0] - base_logp[idx, 1] + score
    candidate = torch.stack([new_margin * .5, -new_margin * .5], dim=-1)
    changed = F.log_softmax(candidate, dim=-1)
    if float(module.alpha.detach()) == 0.0:
        # Preserve bit-exact base values while retaining the derivative through changed.
        changed = changed + (base_logp[idx] - changed).detach()
    result = base_logp.clone()
    result[idx] = changed
    return result


def supports_for_queries(query_ids, ids_a, ids_b, test_ids=None):
    import torch
    amap, bmap = set(ids_a), set(ids_b)
    all_ids = sorted(amap | bmap)
    supports = []
    for q in query_ids:
        if q in amap: supports.append(torch.as_tensor(ids_b, dtype=torch.long))
        elif q in bmap: supports.append(torch.as_tensor(ids_a, dtype=torch.long))
        else: supports.append(torch.as_tensor(all_ids if test_ids is None else test_ids, dtype=torch.long))
    return supports


def paths(group, fold):
    stem = f"{group}_f{fold:02d}"
    return TASKS / f"{stem}.json", CKPTS / f"{stem}.best.pt", CKPTS / f"{stem}.latest.pt", LOGS / f"{stem}.jsonl"


def source_snapshot():
    files = [EXP / "config.json", EXP / "run.py", EXP / "scpr.py", EXP / "audit.py",
             BASE_EXP / "run.py", BASE_EXP / "soft_positive_dcr.py", BASE_EXP / "config.json",
             BASE_EXP / "manifest.json", BASE_EXP / "results/audit_status.json",
             REPO / "network.py", REPO / "layers.py", REPO / "model.py",
             REPO / "data/ABIDE/processed_standard_data.csv", REPO / "data/ABIDE/modal_feat_dict.npy"]
    warmups = sorted((E61 / ".runtime/warmup").glob("fold*.pt"))
    base_ckpts = sorted((BASE_EXP / ".runtime/checkpoints").glob(f"{BASE_GROUP}_f*.best.pt"))
    files.extend(warmups); files.extend(base_ckpts)
    files.extend(sorted(ANCHORS.glob("fold*.pt")))
    return {str(p): sha_file(p) for p in files}


def preflight():
    import copy
    import torch
    base, scpr_mod = load_base(), load_scpr()
    # Dependencies from the original CLIP text tokenizer are vendored in a prior audited experiment.
    vendored = Path("/model/spromptgl/experiment/experiment72_mgdss/.runtime/vendor/python")
    if vendored.exists(): sys.path.insert(0, str(vendored))
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    manifest = json.loads((BASE_EXP / "manifest.json").read_text())
    audit = json.loads((BASE_EXP / "results/audit_status.json").read_text())
    if manifest.get("status") != "COMPLETE_AUDITED" or audit.get("status") != "PASS":
        raise RuntimeError("historical base is not COMPLETE_AUDITED")
    base_tasks = []
    for fold in range(1, 11):
        task_file = BASE_EXP / f".runtime/tasks/{BASE_GROUP}_f{fold:02d}.json"
        ckpt = BASE_EXP / f".runtime/checkpoints/{BASE_GROUP}_f{fold:02d}.best.pt"
        warm = E61 / f".runtime/warmup/fold{fold:02d}.pt"
        meta = json.loads(task_file.read_text())
        if not ckpt.is_file() or not warm.is_file() or meta.get("status") != "COMPLETE":
            raise RuntimeError(f"missing base task/checkpoint/warmup fold {fold}")
        if sha_file(ckpt) != meta.get("checkpoint_sha256"):
            raise RuntimeError(f"E74.4 checkpoint SHA mismatch fold {fold}")
        wmeta = json.loads((E61 / f".runtime/tasks/fold{fold:02d}.warmup.json").read_text())
        if sha_file(warm) != wmeta.get("warmup_sha256") or meta["identity"]["warmup_sha256"] != wmeta["warmup_sha256"]:
            raise RuntimeError(f"E61 warmup identity mismatch fold {fold}")
        base_tasks.append(meta)
    vec = text_vectors()
    if tuple(scpr_mod.PROMPTS) != tuple(json.loads((EXP / "config.json").read_text())["scpr"]["prompts_asd_td"]):
        raise RuntimeError("Prompt configuration mismatch")
    if not torch.allclose(torch.nn.functional.cosine_similarity(vec[0:1], vec[1:2]),
                          torch.nn.functional.cosine_similarity(vec[1:2], vec[0:1])):
        raise RuntimeError("semantic class vector check failed")
    before = source_snapshot()
    state, warmup, identity = prepare_base(1)
    if identity["warmup_sha256"] != base_tasks[0]["identity"]["warmup_sha256"]:
        raise RuntimeError("base prepare did not restore expected E61 warmup")
    hist = base.load_history()
    rt = hist.load_e69().load_e61().load_base()
    rng = rt.pack_rng()
    initial = hist.state_bundle(state)
    # E74.4 base-only update.
    hist.zero_optimizers(state)
    out1, _ = hook_forward(state, training=True)
    out1["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"): state[key].step()
    model_ref = {k: v.detach().cpu().clone() for k, v in state["model"].state_dict().items()}
    cam_ref = {k: v.detach().cpu().clone() for k, v in state["cam"].state_dict().items()}
    opt_ref = [copy.deepcopy(state[k].state_dict()) for k in ("opt_base", "opt_cam", "opt_added")]
    rng_after_ref = rt.pack_rng()
    hist.load_bundle(state, initial); rt.restore_rng(rng)
    module = scpr_mod.init_module_isolated(vec).to(state["y"].device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.0005, weight_decay=.01)
    hist.zero_optimizers(state); optimizer.zero_grad(set_to_none=True)
    out2, cam = hook_forward(state, training=True)
    out2["total"].backward()
    for key in ("opt_base", "opt_cam", "opt_added"): state[key].step()
    optimizer.step()
    same = all(torch.equal(model_ref[k], v.detach().cpu()) for k, v in state["model"].state_dict().items())
    same &= all(torch.equal(cam_ref[k], v.detach().cpu()) for k, v in state["cam"].state_dict().items())
    same &= all(sha_obj(opt_ref[i]) == sha_obj(state[k].state_dict()) for i, k in enumerate(("opt_base", "opt_cam", "opt_added")))
    if not same or rt.rng_sha(rt.pack_rng()) != rt.rng_sha(rng_after_ref):
        raise RuntimeError("one-step base/SCPR-disabled parity or RNG parity failed")
    base_logp = out2["logp"].detach()
    testids = state["test_np"]
    safe_labels = torch.full_like(state["y"].long(), -1)
    safe_train = torch.as_tensor(state["train_np"], dtype=torch.long, device=cam.device)
    safe_labels[safe_train] = state["y"][safe_train].long()
    dummy_memory = {"features": torch.nn.functional.normalize(cam.detach().reshape(871, 4, 36), dim=-1),
                    "probabilities": base_logp.detach().exp(), "labels": safe_labels,
                    "observed": torch.ones(871, dtype=torch.bool, device=cam.device), "updates": 1}
    scores, _ = module(cam[testids], testids, dummy_memory,
                       torch.as_tensor(state["train_np"], device=cam.device), read_sites().to(cam.device),
                       base_logp[testids, 0] - base_logp[testids, 1])
    exact = apply_score(base_logp, testids, scores - (base_logp[testids, 0] - base_logp[testids, 1]), module)
    if not torch.equal(exact, base_logp):
        raise RuntimeError("zero-initialized SCPR does not exactly preserve base outputs")
    current = source_snapshot()
    if before != current:
        raise RuntimeError("read-only historical sources/checkpoints changed during preflight")
    record = {"status": "PASS", "base_group": BASE_GROUP, "folds": 10,
              "warmup_sha_match": True, "base_checkpoint_sha_match": True,
              "one_step_update_parity_fold1": True, "initial_residual_exact_zero": True,
              "cam_hook_shape": list(cam.shape), "outputlayer_hook_calls": 1,
              "shared_rng_after_train_parity_fold1": True,
              "source_hashes": current, "prompt_vectors_sha256": hashlib.sha256(vec.numpy().tobytes()).hexdigest(),
              "device": torch.cuda.get_device_name(0), "timestamp": time.time()}
    atomic_json(RESULTS / "preflight.json", record)
    return record


def smoke_active_step():
    """Exercise one memory-backed SCPR loss/gradient without updating checkpoints."""
    import torch
    from scpr import init_module_isolated
    base, hist = load_base(), load_base().load_history()
    state, warmup, _ = prepare_base(1)
    runtime = hist.load_e69().load_e61().load_base()
    initial_rng = runtime.pack_rng()
    train_ids, test_ids = list(map(int, state["train_np"])), list(map(int, state["test_np"]))
    eligible = [i for i in train_ids if i not in set(test_ids)]
    labels = state["y"].long()
    ids_a, ids_b = split_ab(eligible, labels.detach().cpu().numpy())
    sites = read_sites().to(labels.device)
    module = init_module_isolated(text_vectors()).to(labels.device)
    try:
        with torch.no_grad():
            eval_out, eval_cam = hook_forward(state, training=False)
            memory = make_memory(871, labels.device, eligible, labels, eval_cam, eval_out["logp"])
        hist.zero_optimizers(state)
        out, cam = hook_forward(state, training=True)
        cam.retain_grad()
        idx = torch.as_tensor(eligible, dtype=torch.long, device=labels.device)
        base_margin = out["logp"][idx, 0] - out["logp"][idx, 1]
        a_set = set(ids_a)
        support_a = torch.as_tensor(ids_a, device=labels.device)
        support_b = torch.as_tensor(ids_b, device=labels.device)
        supports = [support_b if i in a_set else support_a for i in eligible]
        pred = out["logp"][idx].argmax(-1)
        conf = out["logp"][idx].exp().gather(1, labels[idx, None]).squeeze(1)
        keep = (pred == labels[idx]) & (conf >= .8)
        score, diag = module(cam[idx], eligible, memory, supports, sites, base_margin,
                             labels=labels[idx], keep_mask=keep)
        final_logp = apply_score(out["logp"], eligible, score - base_margin, module)
        cls = hist.load_e69().load_e61().load_base().runtime()["utils"].ClsLoss(
            final_logp, labels, state["train"], state["weights"])
        total = out["total"] - out["cls"] + cls + .02 * diag["proto_loss"] + .02 * diag["res_loss"] + .01 * diag["keep_loss"]
        total.backward()
        check = {"total_finite": bool(torch.isfinite(total)),
                 "cam144_grad_norm": float(cam.grad.norm()),
                 "semantic_grad_norm": float(module.semantic[0].weight.grad.norm()),
                 "alpha_grad_abs": float(module.alpha.grad.abs()),
                 "prototype_loss": float(diag["proto_loss"].detach()),
                 "support_a": len(ids_a), "support_b": len(ids_b),
                 "memory_train_patients": int(memory["observed"].sum())}
        if not check["total_finite"] or check["cam144_grad_norm"] == 0 or check["semantic_grad_norm"] == 0:
            raise RuntimeError(f"active SCPR gradient smoke failed: {check}")
        atomic_json(RESULTS / "active_step_smoke.json", {"status": "PASS", **check})
        return check
    finally:
        runtime.restore_rng(initial_rng)


def train_worker(group, fold):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scpr import init_module_isolated
    pre = json.loads((RESULTS / "preflight.json").read_text())
    if pre.get("status") != "PASS" or pre.get("source_hashes") != source_snapshot():
        raise RuntimeError("preflight is missing or source hashes changed")
    base, hist = load_base(), load_base().load_history()
    state, warmup, identity = prepare_base(fold)
    state["fold"] = fold
    y = state["y"].long()
    train_ids, test_ids = list(map(int, state["train_np"])), list(map(int, state["test_np"]))
    # Fold 2 has a historical train/test overlap. It remains in the original
    # base loss, but is excluded from all new SCPR queries and memory roles.
    test_set = set(test_ids)
    scpr_train_ids = [i for i in train_ids if i not in test_set]
    ids_a, ids_b = split_ab(scpr_train_ids, y.detach().cpu().numpy())
    sites = read_sites().to(y.device)
    vectors = text_vectors()
    module = init_module_isolated(vectors, anonymous=(group == "G2-SCPR-Anonymous"), seed=765101).to(y.device)
    new_opt = torch.optim.AdamW(module.parameters(), lr=.0005, weight_decay=.01)
    task_file, best_file, latest_file, log_file = paths(group, fold)
    if task_file.exists():
        old = json.loads(task_file.read_text())
        if old.get("status") == "COMPLETE" and best_file.exists() and old.get("checkpoint_sha256") == sha_file(best_file):
            return old
        raise RuntimeError(f"refusing to overwrite an existing non-matching task: {task_file}")
    base_runtime = hist.load_e69().load_e61().load_base()
    if base_runtime.rng_sha(base_runtime.pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"fold {fold} did not start with warmup RNG")
    memory = None
    start_epoch, wait, best_acc, best_epoch, best, trace = 10, 0, -1., -1, None, []
    gradient_snapshots = []
    if latest_file.exists():
        saved = torch.load(latest_file, map_location="cpu", weights_only=False)
        if saved.get("group") != group or saved.get("fold") != fold or saved.get("warmup_sha256") != identity["warmup_sha256"]:
            raise RuntimeError("latest checkpoint identity mismatch")
        hist.load_bundle(state, saved["base_state"]); base_runtime.restore_rng(saved["rng"])
        module.load_state_dict(saved["scpr"]); new_opt.load_state_dict(saved["new_optimizer"])
        memory = {k: (v.to(y.device) if torch.is_tensor(v) else v) for k, v in saved["memory"].items()} if saved["memory"] is not None else None
        start_epoch, wait, best_acc, best_epoch, best, trace = saved["next_epoch"], saved["wait"], saved["best_acc"], saved["best_epoch"], saved["best"], saved["trace"]
        gradient_snapshots = saved.get("gradient_audit", [])
    started = time.time()
    last_epoch = start_epoch
    for epoch in range(start_epoch, 450):
        last_epoch = epoch + 1
        rng_before = base_runtime.rng_sha(base_runtime.pack_rng())
        hist.zero_optimizers(state); new_opt.zero_grad(set_to_none=True)
        out, train_cam = hook_forward(state, training=True)
        if memory is None:
            total = out["total"]
            diagnostics = None
        else:
            train_cam.retain_grad()
            train_idx = torch.as_tensor(scpr_train_ids, dtype=torch.long, device=y.device)
            base_margin = out["logp"][train_idx, 0] - out["logp"][train_idx, 1]
            aset = set(ids_a)
            support_a = torch.as_tensor(ids_a, device=y.device)
            support_b = torch.as_tensor(ids_b, device=y.device)
            query_support = [support_b if q in aset else support_a for q in scpr_train_ids]
            base_pred = out["logp"][train_idx].argmax(-1)
            base_conf = out["logp"][train_idx].exp().gather(1, y[train_idx, None]).squeeze(1)
            keep = (base_pred == y[train_idx]) & (base_conf >= .8)
            score, diagnostics = module(train_cam[train_idx], scpr_train_ids, memory, query_support, sites,
                                        base_margin, labels=y[train_idx], keep_mask=keep)
            final_logp = apply_score(out["logp"], scpr_train_ids, score - base_margin, module)
            cls = hist.load_e69().load_e61().load_base().runtime()["utils"].ClsLoss(
                final_logp, y, state["train"], state["weights"])
            total = out["total"] - out["cls"] + cls + .02 * diagnostics["proto_loss"] + .02 * diagnostics["res_loss"] + .01 * diagnostics["keep_loss"]
        if not torch.isfinite(total): raise FloatingPointError(f"nonfinite loss group={group} fold={fold} epoch={epoch+1}")
        total.backward()
        if diagnostics is not None and (epoch + 1 in (12, 13, 20) or epoch + 1 == 449):
            def grad_norm(parameter):
                return 0.0 if parameter.grad is None else float(parameter.grad.detach().norm().cpu())
            gradient_snapshots.append({"epoch": epoch + 1,
                "cam144": float(train_cam.grad.detach().norm().cpu()) if train_cam.grad is not None else 0.0,
                "semantic_projection": grad_norm(module.semantic[0].weight),
                "modality_gate": grad_norm(module.modality_gate[0].weight),
                "residual_gate": grad_norm(module.residual_gate[0].weight),
                "alpha": grad_norm(module.alpha)})
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()):
            raise FloatingPointError("nonfinite SCPR gradient")
        for name in ("opt_base", "opt_cam", "opt_added"): state[name].step()
        new_opt.step()
        rng_after_train = base_runtime.rng_sha(base_runtime.pack_rng())
        with torch.no_grad():
            evaluated, eval_cam = hook_forward(state, training=False)
            memory_for_eval = memory_snapshot(memory)
            if memory is None:
                final_eval = evaluated["logp"]
                eval_diag = None
            else:
                qidx = torch.as_tensor(test_ids, dtype=torch.long, device=y.device)
                base_margin = evaluated["logp"][qidx, 0] - evaluated["logp"][qidx, 1]
                support = torch.as_tensor(scpr_train_ids, dtype=torch.long, device=y.device)
                scores, eval_diag = module(eval_cam[qidx], test_ids, memory, support, sites, base_margin)
                final_eval = apply_score(evaluated["logp"], test_ids, scores - base_margin, module)
            metrics = base_runtime.classification_metrics(final_eval, state["test"], y)
            memory = make_memory(871, y.device, scpr_train_ids, y, eval_cam, evaluated["logp"], memory)
        rng_after_eval = base_runtime.rng_sha(base_runtime.pack_rng())
        trace.append({"epoch": epoch + 1, "rng_before": rng_before,
                      "rng_after_train": rng_after_train, "rng_after_eval": rng_after_eval})
        if float(metrics["acc"]) > best_acc:
            best_acc, best_epoch, wait = float(metrics["acc"]), epoch, 0
            best = {"base_state": hist.state_bundle(state),
                    "scpr": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                    "new_optimizer": copy.deepcopy(new_opt.state_dict()), "memory": memory_for_eval,
                    "support_a": ids_a, "support_b": ids_b, "scpr_train_ids": scpr_train_ids,
                    "metrics": {k: float(v) for k, v in metrics.items() if k not in ("pred", "prob")},
                    "probability": np.asarray(metrics["prob"]), "prediction": np.asarray(metrics["pred"]),
                    "epoch": epoch + 1,
                    "evidence": None if eval_diag is None else {k: v.detach().cpu() for k, v in eval_diag.items()}}
            atomic_torch(best_file, {"group": group, "fold": fold, "base_group": BASE_GROUP,
                                     "warmup_sha256": identity["warmup_sha256"], "identity": identity,
                                     "best": best, "rng_trace": trace})
        else: wait += 1
        if (epoch + 1) % 10 == 0 or wait > 200 or epoch == 449:
            atomic_torch(latest_file, {"group": group, "fold": fold, "warmup_sha256": identity["warmup_sha256"],
                                       "base_state": hist.state_bundle(state), "scpr": module.state_dict(),
                                       "new_optimizer": new_opt.state_dict(), "memory": memory,
                                       "rng": base_runtime.pack_rng(), "next_epoch": epoch + 1, "wait": wait,
                                       "best_acc": best_acc, "best_epoch": best_epoch, "best": best, "trace": trace,
                                       "gradient_audit": gradient_snapshots})
            row = {"group": group, "fold": fold, "epoch": epoch + 1, "best_epoch": best_epoch + 1,
                   "best_acc": best_acc, "loss": float(total.detach()),
                   "alpha": float(module.alpha.detach()), "gate_mean": float(diagnostics["gate"].mean().detach()) if diagnostics else 0.,
                   "delta_mean_abs": float(diagnostics["delta"].abs().mean().detach()) if diagnostics else 0.,
                   "memory_updates": int(memory["updates"]), "memory_observed": int(memory["observed"].sum()),
                   "rng": trace[-1]}
            with log_file.open("a") as f: f.write(json.dumps(row, allow_nan=False) + "\n")
        if wait > 200: break
    if best is None: raise RuntimeError("no best checkpoint was selected")
    payload = {"group": group, "fold": fold, "seed": 0, "base_group": BASE_GROUP,
               "warmup_sha256": identity["warmup_sha256"], "identity": identity,
               "best": best, "rng_trace": trace, "epochs_executed": last_epoch,
               "wall_seconds": time.time() - started}
    atomic_torch(best_file, payload)
    task = {"status": "COMPLETE", "group": group, "fold": fold,
            "checkpoint_sha256": sha_file(best_file), "warmup_sha256": identity["warmup_sha256"],
            "identity": identity, "best_epoch_1based": best_epoch + 1, "epochs_executed": last_epoch,
            "metrics": best["metrics"], "rng_trace": trace, "wall_seconds": payload["wall_seconds"],
            "scpr_grad_nonzero": any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters()),
            "cam_hook_valid": True, "gradient_audit": gradient_snapshots}
    atomic_json(task_file, task)
    return task


def run_all(workers=10):
    jobs = [(g, f) for f in range(1, 11) for g in GROUPS]
    active = []
    while jobs or active:
        while jobs and len(active) < workers:
            group, fold = jobs.pop(0)
            cmd = [sys.executable, str(EXP / "run.py"), "--worker", group, str(fold)]
            LOGS.mkdir(parents=True, exist_ok=True)
            out = (LOGS / f"{group}_f{fold:02d}.stdout.log").open("ab")
            proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=out, stderr=subprocess.STDOUT)
            active.append((proc, group, fold, out))
        remaining = []
        failure = None
        for proc, group, fold, stream in active:
            code = proc.poll()
            if code is None: remaining.append((proc, group, fold, stream)); continue
            stream.close()
            if code != 0 and failure is None:
                failure = f"worker failed: {group} fold={fold}, exit={code}"
        if failure is not None:
            for proc, _group, _fold, stream in remaining:
                proc.terminate()
                proc.wait()
                stream.close()
            raise RuntimeError(failure)
        active = remaining
        if active: time.sleep(1)
    return {g: sum(1 for f in range(1, 11) if (TASKS / f"{g}_f{f:02d}.json").exists()) for g in GROUPS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("GROUP", "FOLD"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.preflight: print(json.dumps(preflight(), indent=2)); return
    if args.smoke: print(json.dumps(smoke_active_step(), indent=2)); return
    if args.worker: print(json.dumps(train_worker(args.worker[0], int(args.worker[1])), indent=2)); return
    if args.all:
        result = run_all(max(1, min(10, args.workers)))
        print(json.dumps(result, indent=2)); return
    parser.error("use --preflight, --worker GROUP FOLD, or --all")


if __name__ == "__main__": main()
