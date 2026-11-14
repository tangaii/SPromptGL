#!/usr/bin/env python3
"""Experiment 79.3 formal runner.

One E74.4-G2 base trajectory is executed per outer fold.  Three detached
CAM144 memory branches are trained and evaluated from the same BaseEval.  No
branch operation has a path to DARG, GFR, CAM, or the original OutputLayer.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
REPO = Path("/data/adodas/SPromptGL-main")
BASE_EXP = REPO / "experiment/experiment74_4_soft_positive_dcr"
E61 = REPO / "experiment/experiment61_cadr_postalign_vs_e2e"
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
RT = EXP / ".runtime"
CKPTS, TASKS, LOGS = (RT / x for x in ("checkpoints", "tasks", "logs"))
RESULTS = EXP / "results"
for _p in (CKPTS, TASKS, LOGS, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
BASE_GROUP = CFG["base_group"]
GROUPS = tuple(CFG["groups"])
FOLD2_EXCLUDE = {376, 406}

sys.path.insert(0, str(EXP))
from path_adapter import legacy
from memory_common import detach_memory, memory_loss, output_layer_logp
from contrast_memory import CLDCM, build_contrast_memory
from evidence_memory import CPDEM, build_evidence_memory


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if path.read_text(encoding="utf-8") != text:
        raise IOError(f"atomic JSON readback failed: {path}")


def atomic_torch(path, payload):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    if not isinstance(torch.load(path, map_location="cpu", weights_only=False), dict):
        raise IOError(f"checkpoint readback failed: {path}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def source_snapshot():
    paths = [EXP / x for x in ("config.json", "run.py", "memory_common.py", "contrast_memory.py", "evidence_memory.py", "path_adapter.py")]
    paths += [BASE_EXP / x for x in ("run.py", "soft_positive_dcr.py", "config.json", "manifest.json", "audit.py", "results/audit_status.json")]
    paths += [REPO / x for x in ("main.py", "model.py", "network.py", "layers.py", "utils.py", "data/ABIDE/processed_standard_data.csv", "data/ABIDE/modal_feat_dict.npy")]
    paths += sorted(E61.glob(".runtime/warmup/fold*.pt"))
    paths += sorted(BASE_EXP.glob(f".runtime/checkpoints/{BASE_GROUP}_f*.best.pt"))
    paths += sorted(BASE_EXP.glob(f".runtime/tasks/{BASE_GROUP}_f*.json"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("inventory incomplete: " + ",".join(missing))
    return {str(p.resolve()): sha_file(p) for p in paths}


def hook_forward(state, training):
    captured = []
    hook = state["cam"].Outputlayer.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0]))
    try:
        out = legacy.load_base().forward(state, BASE_GROUP, training=training)
    finally:
        hook.remove()
    if len(captured) != 1 or tuple(captured[0].shape) != (871, 144):
        raise RuntimeError(f"CAM144 mismatch: {len(captured)} {[tuple(x.shape) for x in captured]}")
    return out, captured[0]


def prepare_fold(fold):
    state, warm, identity, task, ckpt = legacy.prepare_fold(fold)
    if task.get("status") != "COMPLETE":
        raise RuntimeError(f"historical task incomplete fold={fold}")
    expected = [377, 407] if fold == 2 else []
    if identity["membership"]["train_test_overlap_1based"] != expected:
        raise RuntimeError(f"historical overlap mismatch fold={fold}")
    if tuple(state["train_pairs_np"].shape) != (4096, 2):
        raise RuntimeError(f"relation pairs mismatch fold={fold}")
    return state, warm, identity, task, ckpt


def eligible_ids(state, fold):
    test = {int(x) for x in state["test_np"]}
    ids = sorted(int(x) for x in state["train_np"] if int(x) not in test)
    if fold == 2:
        ids = [x for x in ids if x not in FOLD2_EXCLUDE]
    if set(ids) & test or (fold == 2 and set(ids) & FOLD2_EXCLUDE):
        raise RuntimeError(f"illegal eligible IDs fold={fold}")
    return ids


def site_tensor(device):
    with (REPO / "data/ABIDE/processed_standard_data.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f)); fields = list(rows[0].keys()) if rows else []
    cols = [x for x in fields if x.upper().startswith("SITE_ID_")]
    if len(rows) != 871 or not cols:
        raise RuntimeError("SITE_ID one-hot columns unavailable")
    import torch
    values = [max(range(len(cols)), key=lambda j: float(rows[i].get(cols[j]) or 0.0)) for i in range(871)]
    return torch.tensor(values, dtype=torch.long, device=device)


def rng_sha(runtime):
    return runtime.rng_sha(runtime.pack_rng())


def state_bundle(state):
    return legacy.load_base().load_history().state_bundle(state)


def metric_payload(m):
    return {k: float(v) for k, v in m.items() if k not in ("prob", "pred")}


def same_state(a, b):
    import torch
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return False
        return all(same_state(a[k], b[k]) for k in a)
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def correction_stats(base_logp, method_logp, ids, labels):
    import numpy as np
    idx = np.asarray(ids, dtype=np.int64)
    bp = base_logp.detach().cpu().exp().numpy()[idx]; mp = method_logp.detach().cpu().exp().numpy()[idx]
    y = labels.detach().cpu().numpy()[idx]; b = bp.argmax(1); m = mp.argmax(1)
    bc, mc = b == y, m == y
    return {"base_errors": int((~bc).sum()), "method_errors": int((~mc).sum()),
            "error_to_correct": int(((~bc) & mc).sum()), "correct_to_error": int((bc & (~mc)).sum()),
            "net_correction": int(((~bc) & mc).sum() - ((bc) & (~mc)).sum()),
            "hard_prediction_changes": int((b != m).sum()),
            "asd_error_corrected": int(((~bc) & mc & (y == 0)).sum()),
            "td_error_corrected": int(((~bc) & mc & (y == 1)).sum()),
            "correct_retention": float((bc & mc).sum() / max(int(bc.sum()), 1)),
            "mean_abs_prob_delta": float(np.abs(mp - bp).mean())}


def init_modules(device):
    import torch
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(79300); torch.cuda.manual_seed_all(79300) if torch.cuda.is_available() else None
        modules = [CLDCM(temperature=CFG["temperature"], delta_cap=CFG["delta_relative_cap"], gate_bias=CFG["gate_bias"]).to(device),
                   CPDEM(temperature=CFG["temperature"], delta_cap=CFG["delta_relative_cap"], gate_bias=CFG["gate_bias"]).to(device)]
    # Same fusion skeleton initialization; each group owns its own copy.
    common = {k: v.detach().clone() for k, v in modules[0].state_dict().items()}
    modules[1].load_state_dict(common)
    return modules


def build_memory(group_idx, cam, base_logp, labels, eligible, sites, output_layer, version):
    if group_idx == 0:
        return build_contrast_memory(cam, labels, eligible, sites,
                                     max_pairs=CFG["g1"]["max_pairs"],
                                     patient_max_pairs=CFG["g1"]["patient_max_pairs"],
                                     local_support_k=CFG["g1"]["local_support_k"],
                                     min_support_per_class=CFG["g1"]["min_support_per_class"],
                                     min_auc=CFG["g1"]["min_auc"], min_separation=CFG["g1"]["min_separation"],
                                     version=version, refresh_epoch=version)
    if group_idx == 1:
        return build_evidence_memory(cam, labels, base_logp, eligible, sites, output_layer,
                                     max_per_class=CFG["g2"]["max_per_class"], evidence_dims=CFG["g2"]["evidence_dims"],
                                     alphas=tuple(CFG["g2"]["alphas"]),
                                     min_same_support=CFG["g2"]["min_same_support"],
                                     min_opposite_support=CFG["g2"]["min_opposite_support"],
                                     min_auc=CFG["g2"]["min_auc"],
                                     min_same_minus_opp=CFG["g2"]["min_same_minus_opp"], version=version, refresh_epoch=version)
    raise ValueError(group_idx)


def module_forward(idx, module, ids, cam, base_logp, memory, sites, *, intervention=None):
    kwargs = {}
    if idx == 0:
        kwargs["content_permuted"] = intervention == "Memory-Content-Permuted"
        kwargs["pair_association_break"] = intervention == "Pair-Association-Break"
    elif idx == 1:
        kwargs["content_permuted"] = intervention == "Memory-Content-Permuted"
        kwargs["evidence_mask_break"] = intervention == "Evidence-Mask-Break"
        kwargs["cross_patient_reliability_off"] = intervention == "Cross-Patient-Reliability-Off"
    return module(ids, cam[torch.as_tensor(ids, dtype=torch.long, device=cam.device)].detach(),
                  base_logp[torch.as_tensor(ids, dtype=torch.long, device=cam.device)].detach(), memory, sites, **kwargs)


def train_module(idx, module, optimizer, state, eligible, cam, base_logp, memory, sites):
    import torch
    if memory is None or not memory.get("ids", []):
        return 0.0, {"mem_cls": 0.0, "wrong_margin": 0.0, "correct_preserve": 0.0,
                      "evidence": 0.0, "delta_reg": 0.0, "base_error_count": 0,
                      "gate_mean": 0.0, "delta_ratio_mean": 0.0}, 0.0
    module.train(); optimizer.zero_grad(set_to_none=True)
    idx_t = torch.as_tensor(eligible, dtype=torch.long, device=cam.device)
    h = cam[idx_t].detach(); lp = base_logp[idx_t].detach()
    enhanced, diag = module(eligible, h, lp, memory, sites)
    mem_lp = output_layer_logp(state["cam"].Outputlayer, enhanced)
    loss, comp = memory_loss(mem_lp, lp, state["y"], eligible, state["weights"], diag["delta_h"], h, diag.get("evidence_logits"))
    loss.backward()
    grads = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    grad_norm = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in module.parameters()):
        raise FloatingPointError(f"non-finite branch gradient group={GROUPS[idx]}")
    optimizer.step()
    comp["gate_mean"] = float(diag["gate"].detach().mean())
    return float(loss.detach()), comp, grad_norm


def checkpoint_payload(group, fold, epoch, ledger_kind, state, module, optimizer, memory_used, memory_next,
                       base_metrics, method_metrics, base_logp, method_logp, eligible, test_ids, identity, source,
                       trace, memory_stats, transition):
    import numpy as np
    return {
        "experiment": CFG["experiment"], "group": group, "fold": fold, "seed": 0,
        "base_group": BASE_GROUP, "ledger_kind": ledger_kind, "source_sha256": source,
        "identity": identity, "best_epoch_1based": int(epoch),
        "metrics": metric_payload(method_metrics),
        "probability": np.asarray(method_metrics["prob"], dtype=float),
        "prediction": np.asarray(method_metrics["pred"], dtype=int),
        "base_probability": np.asarray(base_metrics["prob"], dtype=float),
        "base_prediction": np.asarray(base_metrics["pred"], dtype=int),
        "base_state": state_bundle(state),
        "module": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "memory_used": detach_memory(memory_used), "memory_next": detach_memory(memory_next),
        "eligible_train_ids": list(map(int, eligible)), "test_ids": list(map(int, test_ids)),
        "fold2_exclude": sorted(FOLD2_EXCLUDE) if fold == 2 else [],
        "transition": transition, "memory_stats": memory_stats,
        "rng_trace": copy.deepcopy(trace),
    }


def save_best(group, fold, payload):
    atomic_torch(CKPTS / f"{group}_f{fold:02d}.{payload['ledger_kind']}.pt", payload)


def train_fold(fold):
    import numpy as np
    import torch
    state, warm, identity, hist_task, hist_ckpt = prepare_fold(fold)
    runtime = legacy.history_runtime(); hist = legacy.load_base().load_history()
    if rng_sha(runtime) != warm["rng_sha"]:
        raise RuntimeError(f"fold {fold} not at exact E61 RNG")
    eligible = eligible_ids(state, fold); test_ids = [int(x) for x in state["test_np"]]
    y = state["y"].long(); sites = site_tensor(y.device); source = sha_obj(source_snapshot())
    hist_best_epoch = int(hist_task["best_epoch_1based"]); hist_best = torch_load(hist_ckpt)
    modules = init_modules(y.device)
    opts = [torch.optim.AdamW(m.parameters(), lr=CFG["memory_optimizer"]["lr"], weight_decay=CFG["memory_optimizer"]["weight_decay"]) for m in modules]
    base_ledger = {"best_acc": -1.0, "best_epoch": -1, "best": None}
    ledgers = {g: {"best_acc": -1.0, "best_epoch": -1, "best": None} for g in GROUPS}
    anchored = {g: None for g in GROUPS}; trace=[]; rows=[]; memory_stats=[]
    memories = [None, None]; memory_versions = [0, 0]; base_wait=0; start_epoch=10
    latest = CKPTS / f"shared_f{fold:02d}.latest.pt"; task_path=TASKS / f"E79.3_shared_f{fold:02d}.json"
    if latest.is_file():
        saved = torch_load(latest)
        if saved.get("identity") != identity or saved.get("source_sha256") != source:
            raise RuntimeError(f"incompatible latest fold={fold}")
        hist.load_bundle(state, saved["base_state"]); runtime.restore_rng(saved["rng"])
        for i,g in enumerate(GROUPS):
            modules[i].load_state_dict(saved["modules"][g]); opts[i].load_state_dict(saved["optimizers"][g])
        memories=saved["memories"]; memory_versions=saved["memory_versions"]; base_ledger=saved["base_ledger"]; ledgers=saved["ledgers"]; anchored=saved["anchored"]; trace=saved["trace"]; rows=saved["rows"]; memory_stats=saved.get("memory_stats",[]); base_wait=int(saved["base_wait"]); start_epoch=int(saved["next_epoch"])
    else:
        atomic_json(task_path, {"status":"RUNNING", "fold":fold, "source_sha256":source, "identity":identity, "historical_base_best_epoch":hist_best_epoch, "eligible_count":len(eligible), "next_epoch":start_epoch})
    if start_epoch < 10 or start_epoch > 450: raise RuntimeError("invalid resume epoch")
    started=time.time()
    max_epoch_exclusive = int(os.environ.get("E793_MAX_EPOCH_EXCLUSIVE", CFG["training"]["max_epoch_exclusive"]))
    for epoch_index in range(start_epoch, max_epoch_exclusive):
        epoch=epoch_index+1; rb=rng_sha(runtime); hist.zero_optimizers(state)
        out, _train_cam = hook_forward(state, True)
        if not torch.isfinite(out["total"]): raise FloatingPointError(f"base nonfinite fold={fold} epoch={epoch}")
        out["total"].backward()
        for key in ("opt_base","opt_cam","opt_added"): state[key].step()
        rat=rng_sha(runtime)
        with torch.no_grad():
            evaluated, cam = hook_forward(state, False)
        rae=rng_sha(runtime)
        if tuple(cam.shape)!=(871,144): raise RuntimeError("CAM144 shape mismatch")
        base_metrics=legacy.extended_metrics(evaluated["logp"], test_ids, y)
        branch_losses={}; branch_comps={}; branch_grads={}; method_results={}
        for i,g in enumerate(GROUPS):
            loss,comp,gn=train_module(i, modules[i], opts[i], state, eligible, cam, evaluated["logp"], memories[i], sites)
            branch_losses[g]=loss; branch_comps[g]=comp; branch_grads[g]=gn
            with torch.no_grad():
                idx_t=torch.as_tensor(test_ids,dtype=torch.long,device=y.device)
                enh,diag=modules[i](test_ids,cam[idx_t].detach(),evaluated["logp"][idx_t].detach(),memories[i],sites)
                mlp=output_layer_logp(state["cam"].Outputlayer,enh)
            full=evaluated["logp"].detach().clone(); full[idx_t]=mlp
            method_results[g]=(full, legacy.extended_metrics(full,test_ids,y), diag)
        memory_used=detach_memory(memories)
        refresh = epoch >= int(CFG["first_refresh_epoch"]) and ((epoch-int(CFG["first_refresh_epoch"])) % int(CFG["memory_refresh_interval"]) == 0)
        memory_next=memories
        if refresh:
            memory_next=[]
            for i in range(2):
                memory_versions[i]+=1
                memory_next.append(build_memory(i,cam,evaluated["logp"],y,eligible,sites,state["cam"].Outputlayer,memory_versions[i]))
            memories=memory_next
        transition={g: correction_stats(evaluated["logp"],method_results[g][0],test_ids,y) for g in GROUPS}
        rngrec={"epoch":epoch,"rng_before":rb,"rng_after_train":rat,"rng_after_eval":rae}; trace.append(rngrec)
        if float(base_metrics["acc"]) > base_ledger["best_acc"]:
            base_ledger={"best_acc":float(base_metrics["acc"]),"best_epoch":epoch,"metrics":metric_payload(base_metrics),"probability":np.asarray(base_metrics["prob"],float),"prediction":np.asarray(base_metrics["pred"],int),"base":state_bundle(state)}; base_wait=0
        else: base_wait+=1
        if epoch == hist_best_epoch:
            cur=state_bundle(state)
            if not same_state(cur,hist_best["best_state"]): raise RuntimeError(f"historical base state mismatch fold={fold} epoch={epoch}")
            if not np.allclose(np.asarray(hist_best["test_probability"]),np.asarray(base_metrics["prob"]),atol=1e-8): raise RuntimeError(f"historical base prediction mismatch fold={fold}")
            for g in GROUPS:
                full,mm,diag=method_results[g]
                p=checkpoint_payload(g,fold,epoch,"base_anchored",state,modules[GROUPS.index(g)],opts[GROUPS.index(g)],memory_used,memory_next,base_metrics,mm,evaluated["logp"],full,eligible,test_ids,identity,source,trace,{"memory_version":memory_versions[GROUPS.index(g)],"refresh":refresh},transition[g])
                anchored[g]=p; save_best(g,fold,p)
        for i,g in enumerate(GROUPS):
            mm=method_results[g][1]
            if float(mm["acc"]) > ledgers[g]["best_acc"]:
                p=checkpoint_payload(g,fold,epoch,"method",state,modules[i],opts[i],memory_used,memory_next,base_metrics,mm,evaluated["logp"],method_results[g][0],eligible,test_ids,identity,source,trace,{"memory_version":memory_versions[i],"refresh":refresh},transition[g])
                ledgers[g]={"best_acc":float(mm["acc"]),"best_epoch":epoch,"best":p}; save_best(g,fold,p)
        rows.append({"fold":fold,"epoch":epoch,"base_acc":float(base_metrics["acc"]),"base_best_acc":base_ledger["best_acc"],"base_wait":base_wait,"refresh":refresh,"memory_versions":list(memory_versions),"g1_acc":float(method_results[GROUPS[0]][1]["acc"]),"g2_acc":float(method_results[GROUPS[1]][1]["acc"]),"rng_before":rb,"rng_after_train":rat,"rng_after_eval":rae,"branch_loss":branch_losses,"branch_grad":branch_grads})
        memory_stats.append({"fold":fold,"epoch":epoch,"refresh":refresh,"versions":list(memory_versions),"counts":[len(m.get("ids",[])) if m else 0 for m in memories],"branch_losses":branch_comps})
        if refresh or epoch == hist_best_epoch or base_wait > 200 or epoch == 450:
            atomic_torch(latest,{"experiment":CFG["experiment"],"fold":fold,"source_sha256":source,"identity":identity,"next_epoch":epoch+1,"base_wait":base_wait,"base_state":state_bundle(state),"rng":runtime.pack_rng(),"modules":{g:{k:v.detach().cpu().clone() for k,v in modules[i].state_dict().items()} for i,g in enumerate(GROUPS)},"optimizers":{g:copy.deepcopy(opts[i].state_dict()) for i,g in enumerate(GROUPS)},"memories":detach_memory(memories),"memory_versions":memory_versions,"base_ledger":base_ledger,"ledgers":ledgers,"anchored":anchored,"trace":trace,"rows":rows,"memory_stats":memory_stats})
            atomic_json(task_path,{"status":"RUNNING","fold":fold,"source_sha256":source,"identity":identity,"next_epoch":epoch+1,"base_wait":base_wait,"base_best_epoch":base_ledger["best_epoch"],"method_best_epochs":{g:ledgers[g]["best_epoch"] for g in GROUPS},"rng_records":len(trace),"refreshes":sum(1 for x in memory_stats if x["refresh"])})
        with (LOGS/f"E79.3_shared_f{fold:02d}.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"fold":fold,"epoch":epoch,"base_acc":float(base_metrics["acc"]),"best":base_ledger["best_acc"],"methods":{g:float(method_results[g][1]["acc"]) for g in GROUPS},"refresh":refresh,"rng":rngrec},allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
        if base_wait > 200: break
    # Shared trajectory identity must match E74.4-G2, not merely G1/G2/G3 each other.
    if max_epoch_exclusive < CFG["training"]["max_epoch_exclusive"]:
        return {"fold":fold,"task":task_path.name,"smoke":True,"epochs_executed":len(rows)}
    historical_trace={int(x["epoch"]):x for x in hist_task.get("rng_trace",[])}; current_trace={int(x["epoch"]):x for x in trace}
    mismatches=[e for e,v in historical_trace.items() if e in current_trace and current_trace[e]!=v]
    if mismatches: raise RuntimeError(f"historical RNG trace mismatch fold={fold}: {mismatches[:3]}")
    if base_ledger["best_epoch"] != hist_best_epoch: raise RuntimeError(f"base best epoch mismatch fold={fold}")
    if any(ledgers[g]["best"] is None for g in GROUPS) or any(anchored[g] is None for g in GROUPS): raise RuntimeError(f"missing ledger fold={fold}")
    atomic_json(task_path,{"status":"COMPLETE","fold":fold,"source_sha256":source,"identity":identity,"eligible_count":len(eligible),"historical_base_best_epoch":hist_best_epoch,"epochs_executed":len(rows),"base_best_epoch":base_ledger["best_epoch"],"method_best_epochs":{g:ledgers[g]["best_epoch"] for g in GROUPS},"base_anchored_epoch":hist_best_epoch,"rng_records":len(trace),"rng_mismatches":len(mismatches),"memory_refreshes":sum(1 for x in memory_stats if x["refresh"]),"elapsed_seconds":time.time()-started})
    return {"fold":fold,"task":task_path.name}


def preflight():
    import torch
    source=source_snapshot(); inherited=[]
    for f in range(1,11):
        # The task files contain the audited fold identity and split digests.
        # Reading all ten identities here avoids ten duplicate PPU model
        # initializations; fold 1 below still performs a live CAM/output test.
        task_path=BASE_EXP/f".runtime/tasks/{BASE_GROUP}_f{f:02d}.json"; ckpt=BASE_EXP/f".runtime/checkpoints/{BASE_GROUP}_f{f:02d}.best.pt"
        task=json.loads(task_path.read_text(encoding="utf-8")); hist=torch_load(ckpt)
        if task.get("status")!="COMPLETE" or not ckpt.is_file(): raise RuntimeError(f"historical task/checkpoint incomplete fold={f}")
        identity=task["identity"]
        expected=[377,407] if f==2 else []
        if identity["membership"]["train_test_overlap_1based"] != expected: raise RuntimeError(f"overlap identity mismatch fold={f}")
        warm_path=E61/f".runtime/warmup/fold{f:02d}.pt"
        inherited.append({"fold":f,"historical_best_epoch":int(task["best_epoch_1based"]),"warmup_sha":sha_file(warm_path),"warmup_rng":task.get("warmup_rng_sha256", identity.get("warmup_rng_sha256")),"split_sha":identity["split_sha"],"pair_sha":identity["pair_sha"],"eligible_count":int(task.get("eligible_count",0) or 0),"overlap":identity["membership"]["train_test_overlap_1based"],"history_checkpoint_sha":sha_file(ckpt),"history_metrics":hist["metrics"]})
    state,_,_,_,_=prepare_fold(1); runtime=legacy.history_runtime(); before=rng_sha(runtime); out,cam=hook_forward(state,False); after=rng_sha(runtime)
    if tuple(cam.shape)!=(871,144): raise RuntimeError("CAM144 preflight failed")
    if before != after: raise RuntimeError("eval RNG changed in deterministic preflight")
    mods=init_modules(state["y"].device); empty=[]; base=out["logp"].detach(); test=[int(x) for x in state["test_np"]]
    for i,m in enumerate(mods):
        enh,d=m(test,cam[torch.as_tensor(test,device=cam.device)].detach(),base[torch.as_tensor(test,device=cam.device)].detach(),None,site_tensor(cam.device));
        if not torch.equal(enh,cam[torch.as_tensor(test,device=cam.device)]): raise RuntimeError(f"empty memory identity failed group={i}")
        if not all(p.grad is None for p in m.parameters()): raise RuntimeError("unexpected preflight gradients")
        empty.append({"group":GROUPS[i],"parameter_count":sum(p.numel() for p in m.parameters()),"empty_identity":True})
    payload={"status":"PASS","source_hashes":source,"fold_inheritance":inherited,"cam_shape":list(cam.shape),"rng_before":before,"rng_after":after,"empty_memory":empty,"base_audit":json.loads((BASE_EXP/"results/audit_status.json").read_text()).get("status"),"historical_base_group":BASE_GROUP,"fold2_exclude":sorted(FOLD2_EXCLUDE),"text_prompt":False,"oof_teacher":False,"timestamp":time.time()}
    atomic_json(RESULTS/"preflight.json",payload); print("PREFLIGHT PASS",flush=True); return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("preflight","train","train-all")); ap.add_argument("--fold",type=int); a=ap.parse_args()
    if a.command=="preflight": preflight()
    elif a.command=="train":
        if not a.fold: raise SystemExit("--fold required")
        train_fold(a.fold)
    else:
        for f in range(1,11): train_fold(f)


if __name__ == "__main__": main()
