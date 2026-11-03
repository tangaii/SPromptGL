#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
BASE_EXP = ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"
REF_EXP = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")
E59_DIR = ROOT / "experiment/experiment59_shape_pen"
sys.path.insert(0, str(EXP))
RUNTIME = EXP / ".runtime"
RESULTS = EXP / "results"
sys.path.insert(0, str(BASE_EXP))
import run as _BASE53
sys.modules["run"] = _BASE53
sys.path.insert(0, str(E59_DIR))
_e59_spec = importlib.util.spec_from_file_location("e70_e59_runtime", E59_DIR / "run.py")
E59 = importlib.util.module_from_spec(_e59_spec)
sys.modules[_e59_spec.name] = E59
_e59_spec.loader.exec_module(E59)
from gcdsi import GCDSI, install_output_hook

BASE = E59.BASE
GROUPS = ("E70-G1", "E70-G2", "E70-G3")
NAMES = {"E70-G1": "GCDSI-BRAINPROMPT-2", "E70-G2": "GCDSI-SPROMPTGL-8", "E70-G3": "GCDSI-ANONYMOUS-2"}
TEXTS_BP = [
    "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
    "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.",
]
TEXTS_SP = ["Nromal brain volume", "Uniform cortical thickness", "Symmetrical brain structure", "Normal ventricular morphology", "Altered brain volume", "Abnormal functional connectivity", "White matter integrity issues", "Different activation patterns"]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_reference():
    manifest_path = REF_EXP / "manifest_fallback_wd_sweep.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE_AUDITED" or manifest.get("independent_checkpoint_audits_passed") != 40:
        raise RuntimeError("53.3 WD sweep reference is not fully audited")
    item = next(g for g in manifest["groups"] if g["name"] == "CADR_WD_0p01_GFR_WD_0p11")
    if (item["cadr_weight_decay"], item["gfr_weight_decay"]) != (0.01, 0.11):
        raise AssertionError("Wrong 53.3 base group")
    rows = {int(f["fold"]): f for f in manifest["folds"] if f["group"] == item["name"]}
    if len(rows) != 10:
        raise AssertionError("Missing reference fold records")
    root = REF_EXP / ".runtime/fallback_wd_sweep/CADR_WD_0p01_GFR_WD_0p11"
    metrics_path = REF_EXP / "results/fallback_wd_sweep/fold_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        metrics = {int(row["fold"]): row for row in csv.DictReader(stream) if row["group"] == item["name"]}
    for fold, row in rows.items():
        stem = f"CADR_WD_0p01_GFR_WD_0p11_f{fold:02d}"
        cp = root / "checkpoints" / f"{stem}.best.pt"
        task = json.loads((root / "tasks" / f"{stem}.json").read_text())
        if sha_file(cp) != row["checkpoint_sha256"] or task["best_sha256"] != row["checkpoint_sha256"]:
            raise AssertionError(f"Reference checkpoint SHA mismatch fold {fold}")
        for key in ("split_sha", "pair_sha", "initial_shared_sha", "initial_added_sha", "initial_rng_sha"):
            if task[key] != row[key]:
                raise AssertionError(f"Reference {key} mismatch fold {fold}")
        if task.get("status") != "COMPLETE" or task.get("readback_status") != "WORKER_READBACK_PASS":
            raise AssertionError(f"Reference task not independently read back fold {fold}")
    return manifest, rows, metrics


def text_banks():
    path = Path("/model/spromptgl/experiment/experiment60_dss_pen/.runtime/text_embeddings.pt")
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache["text_map"]["brainprompt"] != [TEXTS_BP[1], TEXTS_BP[0]]:
        raise AssertionError("BrainPrompt prompt text/order differs from the required official source")
    if cache["text_map"]["original"] != TEXTS_SP:
        raise AssertionError("SPromptGL source text/order differs from the required eight strings")
    bp = F.normalize(cache["embeddings"]["brainprompt"].float(), dim=-1)
    sp = F.normalize(cache["embeddings"]["original"].float(), dim=-1)
    # E60's bank is ASD then TD; source SPromptGL order is normal four then abnormal four.
    sp_bank = torch.stack([F.normalize(sp[4:].mean(0), dim=0), F.normalize(sp[:4].mean(0), dim=0)])
    generator = torch.Generator(device="cpu").manual_seed(700070)
    d = bp.shape[1]
    u = F.normalize(torch.randn(d, generator=generator), dim=0)
    v = torch.randn(d, generator=generator)
    v = F.normalize(v - torch.dot(u, v) * u, dim=0)
    cosine = torch.dot(bp[0], bp[1]).clamp(-0.999, 0.999)
    anon = torch.stack((bp[0].norm() * u, bp[1].norm() * (cosine * u + torch.sqrt(1 - cosine.square()) * v)))
    assets = {"E70-G1": bp, "E70-G2": sp_bank, "E70-G3": anon}
    meta = {
        "cache_sha256": sha_file(path),
        "text_encoder": "OpenAI CLIP RN50; frozen Experiment 60 verified cache",
        "text_dim": d,
        "brainprompt_texts_published_order_TC_ASD": TEXTS_BP,
        "brainprompt_embedding_order": ["ASD", "TD"],
        "spromptgl_texts_source_order": TEXTS_SP,
        "spromptgl_aggregation": "normalize each; mean first four -> TD; mean last four -> ASD; normalize means; reorder [ASD,TD]",
        "anonymous_seed": 700070,
        "anonymous_generator": "private CPU torch.Generator; no data or labels",
        "banks_sha256": {group: hashlib.sha256(bank.contiguous().numpy().tobytes()).hexdigest() for group, bank in assets.items()},
        "g1_cosine": float(torch.dot(bp[0], bp[1])),
        "g3_cosine": float(torch.dot(anon[0], anon[1]) / (anon[0].norm() * anon[1].norm())),
    }
    if abs(meta["g1_cosine"] - meta["g3_cosine"]) > 1e-5:
        raise AssertionError("Anonymous geometry mismatch")
    return assets, meta


def build_state(fold: int, ref_manifest, ref_folds):
    state = E59.build_base_state(fold, ref_manifest, ref_folds)
    if any(float(g["weight_decay"]) != 0.01 for g in state["opt_added"].param_groups):
        raise AssertionError("CADR AdamW must use WD=0.01 in both parameter groups")
    return state


def make_module(group: str, fold: int, text_bank: torch.Tensor):
    device = BASE.runtime()["device"]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(700000 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(700000 + fold)
        module = GCDSI(text_bank.shape[-1]).to(device)
    return module


def forward_state(state, active: bool):
    if state.get("_is_training", False):
        state["model"].train(); state["cam"].train()
    else:
        state["model"].eval(); state["cam"].eval()
    fused, adjs, info = state["model"](state["parts"], pair_index=state["train_pairs"], use_cadr=active)
    state["cam"]._gcdsi_graph_features = fused
    state["cam"]._gcdsi_active = active
    logp, hidden, attention = state["cam"](fused)
    cls = BASE.runtime()["utils"].ClsLoss(logp, state["y"], state["train"], state["weights"])
    graph, graph_parts = BASE._graph_loss(state["parts"], adjs)
    pair_logits = info["pair_logits"]
    pair = torch.stack([F.binary_cross_entropy_with_logits(x, state["train_same"]) for x in pair_logits]).mean()
    adv, adv_info = BASE.conditional_adversarial_loss(info["pair_reprs"], state["train_same"], state["model"].discriminator, state["model"].grl_scale)
    preserve, preserve_gaps = BASE.relation_preservation_loss(pair_logits, state["train_same"], margin=0.02)
    total = cls + graph + 0.05 * pair + 0.01 * adv + 0.01 * preserve
    return {"logp": logp, "hidden": hidden, "attention": attention, "fused": fused, "adjs": adjs, "info": info, "cls": cls, "graph": graph, "graph_parts": graph_parts, "pair": pair, "adv": adv, "adv_info": adv_info, "preserve": preserve, "preserve_gaps": preserve_gaps, "total": total, "gcdsi": state["cam"]._gcdsi_diagnostics}


def optimizer_digest(opt):
    return sha_obj(opt.state_dict())


def worker(group: str, fold: int):
    ref_manifest, ref_folds, _ = load_reference()
    assets, _ = text_banks()
    state = build_state(fold, ref_manifest, ref_folds)
    warmup_path = E59.warmup_cache_path(fold)
    warmup_sha = sha_file(warmup_path)
    warmup_saved = torch.load(warmup_path, map_location="cpu", weights_only=False)
    for key, expected in (("split_sha", state["split_sha"]), ("pair_sha", state["cache_meta"]["pair_sha"]), ("initial_shared_sha", state["initial_shared_sha"]), ("initial_added_sha", state["initial_added_sha"]), ("initial_rng_sha", state["construction_rng_sha"])):
        if warmup_saved[key] != expected: raise AssertionError(f"warmup provenance mismatch {key} fold {fold}")
    E59.load_warmup_cache_into_state(state, fold, json.loads(E59.CONFIG_PATH.read_text()))
    before_rng = BASE.rng_sha(BASE.pack_rng())
    text = assets[group].to(BASE.runtime()["device"])
    module = make_module(group, fold, text)
    install_output_hook(state["cam"], module, text)
    if BASE.rng_sha(BASE.pack_rng()) != before_rng:
        raise AssertionError("GCDSI initialization polluted the base RNG")
    semantic_optimizer = torch.optim.AdamW(module.parameters(), lr=0.001, weight_decay=0.001)
    fold_dir = RUNTIME / group
    cp_path = fold_dir / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
    latest_path = fold_dir / "checkpoints" / f"{group}_f{fold:02d}.latest.pt"
    task_path = fold_dir / "tasks" / f"{group}_f{fold:02d}.json"
    log_path = fold_dir / "logs" / f"{group}_f{fold:02d}.log"
    for p in (cp_path.parent, task_path.parent, log_path.parent): p.mkdir(parents=True, exist_ok=True)
    task_sha = sha_file(Path(__file__))
    config_sha = sha_file(EXP / "config.json")
    start, wait, best_acc, best_epoch, best_state, best_metrics, best_diag = 10, 0, -1.0, -1, None, None, None
    # Reuse the exact common ten-epoch base state; GCDSI has not run during warmup.
    if task_path.is_file() and cp_path.is_file():
        task = json.loads(task_path.read_text())
        if task.get("status") == "COMPLETE" and task.get("checkpoint_sha256") == sha_file(cp_path): return
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        if saved.get("source_sha") == task_sha and saved.get("config_sha") == config_sha and saved.get("group") == group and saved.get("fold") == fold:
            state["model"].load_state_dict(saved["model"], strict=True); state["cam"].load_state_dict(saved["cam"], strict=True)
            state["opt_base"].load_state_dict(saved["optimizer_base"]); state["opt_cam"].load_state_dict(saved["optimizer_cam"]); state["opt_added"].load_state_dict(saved["optimizer_added"])
            module.load_state_dict(saved["gcdsi"], strict=True); semantic_optimizer.load_state_dict(saved["optimizer_gcdsi"])
            BASE.restore_rng(saved["rng"]); start=saved["next_epoch"]; wait=saved["wait"]; best_acc=saved["best_acc"]; best_epoch=saved["best_epoch"]; best_state=saved["best_state"]; best_metrics=saved["best_metrics"]; best_diag=saved["best_diag"]
            if saved["warmup_sha"] != warmup_sha: raise AssertionError("resume warmup SHA mismatch")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"START group={group} fold={fold} epoch={start}\n"); log.flush()
        for epoch in range(start, 450):
            active = epoch >= 10
            state["_is_training"] = True
            state["opt_base"].zero_grad(set_to_none=True); state["opt_cam"].zero_grad(set_to_none=True); state["opt_added"].zero_grad(set_to_none=True); semantic_optimizer.zero_grad(set_to_none=True)
            out = forward_state(state, active)
            if not torch.isfinite(out["total"]): raise FloatingPointError(f"nonfinite loss {group} fold {fold} epoch {epoch}")
            out["total"].backward()
            if active and epoch == 10:
                for name in ("gamma_raw",):
                    p = dict(module.named_parameters())[name]
                    if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs()) <= 0: raise AssertionError(f"missing first-active gamma gradient {group}/{fold}")
            for m in (state["model"], state["cam"], module):
                for p in m.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError("nonfinite gradient")
            state["opt_base"].step(); state["opt_cam"].step(); state["opt_added"].step()
            if active: semantic_optimizer.step()
            with torch.no_grad():
                state["_is_training"] = False
                evaluated = forward_state(state, active)
                metrics = BASE.classification_metrics(evaluated["logp"], state["test"], state["y"])
            if active and metrics["acc"] > best_acc:
                best_acc=float(metrics["acc"]); best_epoch=epoch; wait=0
                best_state={"model":BASE._state_cpu(state["model"]),"cam":BASE._state_cpu(state["cam"]),"gcdsi":BASE._state_cpu(module),"optimizer_base":copy.deepcopy(state["opt_base"].state_dict()),"optimizer_cam":copy.deepcopy(state["opt_cam"].state_dict()),"optimizer_added":copy.deepcopy(state["opt_added"].state_dict()),"optimizer_gcdsi":copy.deepcopy(semantic_optimizer.state_dict())}
                best_metrics={k:v for k,v in metrics.items() if k not in {"pred","prob"}}
                diag=evaluated["gcdsi"]
                best_diag={"gamma":float(module.gamma.detach().cpu()),"attention_mean":diag["attention"].detach().mean((0,1)).cpu().tolist(),"attention_entropy":float((-(diag["attention"].clamp_min(1e-12)*diag["attention"].clamp_min(1e-12).log()).sum(-1)).mean().cpu()),"gate_mean":float(diag["gate"].detach().mean().cpu()),"residual_rms_ratio":float((diag["residual"].detach().square().mean().sqrt()/(state["cam"]._gcdsi_graph_features.detach().square().mean().sqrt()+1e-12)).cpu()),"w_f_grad_norm":float(module.query_f.weight.grad.detach().norm().cpu()),"total_loss":float(out["total"].detach().cpu())}
                cp={"group":group,"fold":fold,"seed":0,"best_epoch":best_epoch,"epochs_executed":epoch+1,"metrics":best_metrics,"state_model":best_state["model"],"state_cam":best_state["cam"],"state_gcdsi":best_state["gcdsi"],"optimizer_base":best_state["optimizer_base"],"optimizer_cam":best_state["optimizer_cam"],"optimizer_added":best_state["optimizer_added"],"optimizer_gcdsi":best_state["optimizer_gcdsi"],"text_bank":text.detach().cpu(),"text_bank_sha256":hashlib.sha256(assets[group].contiguous().numpy().tobytes()).hexdigest(),"warmup_sha256":warmup_sha,"source_sha256":task_sha,"config_sha256":config_sha,"reference_checkpoint_sha256":ref_folds[fold]["checkpoint_sha256"],"split_sha":state["split_sha"],"pair_sha":state["cache_meta"]["pair_sha"],"initial_shared_sha":state["initial_shared_sha"],"initial_added_sha":state["initial_added_sha"],"initial_rng_sha":state["construction_rng_sha"],"optimizer_weight_decay":{"base":[float(g["weight_decay"]) for g in state["opt_base"].param_groups],"cam":[float(g["weight_decay"]) for g in state["opt_cam"].param_groups],"added":[float(g["weight_decay"]) for g in state["opt_added"].param_groups],"gcdsi":[float(g["weight_decay"]) for g in semantic_optimizer.param_groups]},"diagnostics":best_diag}
                atomic_torch(cp_path,cp)
            elif active: wait+=1
            if (epoch+1)%10==0 or (active and wait>200):
                atomic_torch(latest_path,{"group":group,"fold":fold,"next_epoch":epoch+1,"wait":wait,"best_acc":best_acc,"best_epoch":best_epoch,"best_state":best_state,"best_metrics":best_metrics,"best_diag":best_diag,"model":BASE._state_cpu(state["model"]),"cam":BASE._state_cpu(state["cam"]),"gcdsi":BASE._state_cpu(module),"optimizer_base":state["opt_base"].state_dict(),"optimizer_cam":state["opt_cam"].state_dict(),"optimizer_added":state["opt_added"].state_dict(),"optimizer_gcdsi":semantic_optimizer.state_dict(),"rng":BASE.pack_rng(),"source_sha":task_sha,"config_sha":config_sha,"warmup_sha":warmup_sha})
            if active and wait>200: break
            if (epoch+1)%10==0: log.write(f"epoch={epoch+1} best={best_acc:.6f} wait={wait}\n"); log.flush()
    if best_state is None: raise RuntimeError(f"No active checkpoint for {group} fold {fold}")
    cp_sha=sha_file(cp_path)
    atomic_json(task_path,{"group":group,"fold":fold,"status":"COMPLETE","best_epoch":best_epoch,"epochs_executed":epoch+1,"metrics":best_metrics,"checkpoint_sha256":cp_sha,"warmup_sha256":warmup_sha,"reference_checkpoint_sha256":ref_folds[fold]["checkpoint_sha256"],"split_sha":state["split_sha"],"pair_sha":state["cache_meta"]["pair_sha"],"initial_shared_sha":state["initial_shared_sha"],"initial_added_sha":state["initial_added_sha"],"initial_rng_sha":state["construction_rng_sha"],"diagnostics":best_diag})
    latest_path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight",action="store_true"); parser.add_argument("--worker",action="store_true"); parser.add_argument("--group",choices=GROUPS); parser.add_argument("--fold",type=int); parser.add_argument("--all",action="store_true"); parser.add_argument("--workers",type=int,default=10); args=parser.parse_args()
    if args.preflight: preflight(); return
    if args.worker:
        if not args.group or not args.fold: parser.error("worker requires --group and --fold")
        worker(args.group,args.fold); return
    if args.all: run_all(args.workers); return
    parser.error("choose --preflight, --worker, or --all")


def preflight():
    manifest, folds, metrics = load_reference(); assets, meta = text_banks()
    rows=[]
    for fold in range(1,11):
        state=build_state(fold,manifest,folds)
        saved=torch.load(E59.warmup_cache_path(fold),map_location="cpu",weights_only=False)
        checks={"initial_shared":state["initial_shared_sha"]==folds[fold]["initial_shared_sha"],"initial_added":state["initial_added_sha"]==folds[fold]["initial_added_sha"],"initial_rng":state["construction_rng_sha"]==folds[fold]["initial_rng_sha"],"split":state["split_sha"]==folds[fold]["split_sha"],"pair":state["cache_meta"]["pair_sha"]==folds[fold]["pair_sha"],"warmup_sha_recorded":bool(sha_file(E59.warmup_cache_path(fold))),"warmup_binding":all(saved[k]==v for k,v in (("split_sha",state["split_sha"]),("pair_sha",state["cache_meta"]["pair_sha"]),("initial_shared_sha",state["initial_shared_sha"]),("initial_added_sha",state["initial_added_sha"]),("initial_rng_sha",state["construction_rng_sha"]))) }
        if not all(checks.values()): raise AssertionError(f"preflight fold {fold}: {checks}")
        rows.append({"fold":fold,**checks})
    # Exercise the actual CAM output hook with base activations and verify identity plus finite gradients.
    state=build_state(1,manifest,folds); text=assets["E70-G1"].to(BASE.runtime()["device"]); module=make_module("E70-G1",1,text); install_output_hook(state["cam"],module,text)
    fused=state["model"](state["parts"],pair_index=state["train_pairs"],use_cadr=True)[0]
    state["model"].eval(); state["cam"].eval()
    state["cam"]._gcdsi_graph_features=fused; state["cam"]._gcdsi_active=False
    with torch.no_grad(): base_logp=state["cam"](fused)[0]
    state["cam"]._gcdsi_active=True
    with torch.no_grad(): identity_logp=state["cam"](fused)[0]
    if not torch.equal(base_logp,identity_logp): raise AssertionError("GCDSI gamma=0 identity parity failed")
    state["_is_training"]=True; state["cam"]._gcdsi_active=True; state["cam"].train(); module.train()
    out=forward_state(state,True); out["total"].backward()
    gamma_grad=module.gamma_raw.grad
    if gamma_grad is None or not torch.isfinite(gamma_grad).all() or float(gamma_grad.abs())==0: raise AssertionError("GCDSI gamma has no first-active gradient")
    module_optimizer=torch.optim.AdamW(module.parameters(),lr=0.001,weight_decay=0.001)
    module_optimizer.step(); module_optimizer.zero_grad(set_to_none=True)
    state["opt_base"].zero_grad(set_to_none=True); state["opt_cam"].zero_grad(set_to_none=True); state["opt_added"].zero_grad(set_to_none=True)
    out=forward_state(state,True); out["total"].backward()
    required=[module.query_f.weight.grad,module.key.weight.grad,module.value.weight.grad,module.gate[0].weight.grad,state["cam"].Outputlayer.mlp_head[1].weight.grad]
    if any(x is None or not torch.isfinite(x).all() or float(x.norm())==0 for x in required): raise AssertionError("GCDSI/base gradients missing after activation")
    for group in GROUPS:
        m=make_module(group,1,assets[group].to(BASE.runtime()["device"]))
        if sha_obj({k:v.detach().cpu().tolist() for k,v in m.state_dict().items()}) != sha_obj({k:v.detach().cpu().tolist() for k,v in make_module(group,1,assets[group].to(BASE.runtime()["device"])).state_dict().items()}): raise AssertionError("isolated initialization is not repeatable")
    record={"status":"PASS","reference_group":"CADR_WD_0p01_GFR_WD_0p11","base_reference_manifest_sha256":sha_file(REF_EXP/"manifest_fallback_wd_sweep.json"),"base_source_sha256":{"run.py":sha_file(BASE_EXP/"run.py"),"cadr_gfr.py":sha_file(BASE_EXP/"cadr_gfr.py"),"config.json":sha_file(BASE_EXP/"config.json")},"text_assets":meta,"folds":rows,"identity_logp_bitwise":True,"active_gradient_checks":"PASS","warmup_epochs":10,"warmup_cache_sha256":{str(f):sha_file(E59.warmup_cache_path(f)) for f in range(1,11)}}
    atomic_json(RUNTIME/"preflight.json",record); atomic_json(EXP/"manifest.json",{"experiment":"70","status":"PREFLIGHT_PASS","formal_groups":3,"formal_folds_required":30,"preflight":"PASS","reference_folds_read_only":10,"preflight_sha256":sha_file(RUNTIME/"preflight.json")})
    print(json.dumps(record,ensure_ascii=False))


def run_all(workers: int):
    if not (1<=workers<=10): raise ValueError("workers must be 1..10")
    tasks=[(g,f) for g in GROUPS for f in range(1,11)]
    procs=[]
    for group,fold in tasks:
        log=RUNTIME/group/"logs"/f"{group}_f{fold:02d}.log"; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(EXP/"run.py"),"--worker","--group",group,"--fold",str(fold)]
        procs.append((group,fold,subprocess.Popen(cmd,cwd=ROOT,stdout=log.open("a"),stderr=subprocess.STDOUT)))
        while sum(p.poll() is None for _,_,p in procs)>=workers:
            for g,f,p in list(procs):
                if p.poll() not in (None,0): raise RuntimeError(f"worker failed {g} fold {f}, rc={p.returncode}")
            time.sleep(2)
    bad=[(g,f,p.returncode) for g,f,p in procs if p.wait()!=0]
    if bad: raise RuntimeError(f"failed workers: {bad}")
    audit=subprocess.run([sys.executable,str(EXP/"audit.py"),"--all"],cwd=ROOT,check=False)
    if audit.returncode: raise RuntimeError("independent audit failed")


if __name__=="__main__": main()
