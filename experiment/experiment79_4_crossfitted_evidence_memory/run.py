#!/usr/bin/env python3
"""Experiment 79.4: one shared E74.4-G2 trajectory with CF-CPDEM."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_k]="1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.dont_write_bytecode=True

EXP=Path(__file__).resolve().parent
REPO=Path("/data/adodas/SPromptGL-main")
BASE_EXP=REPO/"experiment/experiment74_4_soft_positive_dcr"
E61=REPO/"experiment/experiment61_cadr_postalign_vs_e2e"
E793=Path("/model/spromptgl/experiment/experiment79_3_competitive_evidence_memory")
CFG=json.loads((EXP/"config.json").read_text(encoding="utf-8"))
RT=EXP/".runtime"; CKPTS=RT/"checkpoints"; TASKS=RT/"tasks"; LOGS=RT/"logs"; RESULTS=EXP/"results"
for _p in (CKPTS,TASKS,LOGS,RESULTS): _p.mkdir(parents=True,exist_ok=True)
BASE_GROUP=CFG["base_group"]; GROUP="E79.4-G1-CF-CPDEM"; FOLD2_EXCLUDE={376,406}
sys.path.insert(0,str(EXP))
from path_adapter import legacy
from evidence_memory import make_partitions,build_crossfit_memories
from retrieval import CFCPDEM
from memory_common import output_layer_logp,detach_memory,weighted_memory_loss


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def content_digest(obj):
    """Stable recursive content hash, independent of torch serialization IDs."""
    import numpy as np,torch
    digest=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            t=value.detach().cpu().contiguous();digest.update(b"tensor"+str(t.dtype).encode()+str(tuple(t.shape)).encode());digest.update(t.numpy().tobytes())
        elif isinstance(value,np.ndarray):
            digest.update(b"ndarray"+str(value.dtype).encode()+str(value.shape).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            digest.update(b"dict")
            for key in sorted(value,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(value[key])
        elif isinstance(value,(list,tuple)):
            digest.update(type(value).__name__.encode()+str(len(value)).encode())
            for item in value:visit(item)
        else:digest.update((type(value).__name__+":"+repr(value)).encode()+b"\0")
    visit(obj);return digest.hexdigest()

def rng_sha(runtime): return runtime.rng_sha(runtime.pack_rng())

def atomic_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    text=json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    with tmp.open("w",encoding="utf-8") as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if path.read_text(encoding="utf-8")!=text:raise IOError(f"JSON readback mismatch {path}")

def atomic_torch(path,payload):
    import torch
    path=Path(path);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if not isinstance(obj,dict):raise IOError(f"checkpoint readback mismatch {path}")

def torch_load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)

def write_csv(path,rows):
    if not rows:return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    path=Path(path);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def source_snapshot():
    files=[EXP/x for x in ("config.json","run.py","evidence_memory.py","retrieval.py","memory_common.py","path_adapter.py","audit.py","analyze.py")]
    files += [BASE_EXP/x for x in ("run.py","soft_positive_dcr.py","config.json","manifest.json","audit.py","results/report.md","results/audit_status.json")]
    files += [REPO/x for x in ("main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy")]
    files += [E793/x for x in ("run.py","evidence_memory.py","memory_common.py","audit.py","analyze.py","config.json","manifest.json","results/report.md","results/mechanism.csv","results/memory_analysis.csv")]
    files += sorted(E61.glob(".runtime/warmup/fold*.pt"))
    files += sorted(BASE_EXP.glob(f".runtime/checkpoints/{BASE_GROUP}_f*.best.pt"))
    files += sorted(BASE_EXP.glob(f".runtime/tasks/{BASE_GROUP}_f*.json"))
    missing=[str(p) for p in files if not p.is_file()]
    if missing:raise FileNotFoundError("locked inventory missing: "+", ".join(missing[:10]))
    return {str(p.resolve()):sha_file(p) for p in files}

def source_sha():return sha_obj(source_snapshot())

def prepare_fold(fold):
    state,warm,identity,task,ckpt=legacy.prepare_fold(fold)
    if task.get("status")!="COMPLETE":raise RuntimeError(f"historical E74.4 incomplete fold={fold}")
    expected=[377,407] if fold==2 else []
    if identity["membership"]["train_test_overlap_1based"]!=expected:raise RuntimeError(f"historical overlap mismatch f={fold}")
    if tuple(state["train_pairs_np"].shape)!=(4096,2):raise RuntimeError(f"relation-pair mismatch f={fold}")
    return state,warm,identity,task,ckpt

def hook_forward(state,training):
    captured=[]; handle=state["cam"].Outputlayer.register_forward_pre_hook(lambda _m,inp:captured.append(inp[0]))
    try:out=legacy.load_base().forward(state,BASE_GROUP,training=training)
    finally:handle.remove()
    if len(captured)!=1 or tuple(captured[0].shape)!=(871,144):raise RuntimeError(f"CAM144 hook mismatch: {[tuple(x.shape) for x in captured]}")
    return out,captured[0]

def eligible_ids(state,fold):
    test=set(map(int,state["test_np"])); ids=sorted(int(x) for x in state["train_np"] if int(x) not in test)
    if fold==2:ids=[x for x in ids if x not in FOLD2_EXCLUDE]
    if set(ids)&test or (fold==2 and set(ids)&FOLD2_EXCLUDE):raise RuntimeError(f"illegal training memory IDs f={fold}")
    return ids

def site_tensor(device):
    with (REPO/"data/ABIDE/processed_standard_data.csv").open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f);rows=list(reader);cols=[x for x in (reader.fieldnames or []) if x.upper().startswith("SITE_ID_")]
    if len(rows)!=871 or not cols:raise RuntimeError("site fields missing")
    out=[]
    for row in rows:
        v=[float(row.get(c) or 0) for c in cols]
        if abs(sum(v)-1)>1e-6:raise RuntimeError("site one-hot mismatch")
        out.append(max(range(len(v)),key=v.__getitem__))
    import torch
    return torch.tensor(out,dtype=torch.long,device=device)

def init_module(device):
    import torch
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        torch.manual_seed(79400)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(79400)
        module=CFCPDEM(CFG["temperature"],CFG["delta_cap"],CFG["gate_bias"],CFG["preselect_top_k"],CFG["top_k"]).to(device)
    return module

def state_bundle(state):return legacy.load_base().load_history().state_bundle(state)
def same_state(a,b):
    import torch
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same_state(a[k],b[k]) for k in a)
    if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same_state(x,y) for x,y in zip(a,b))
    return a==b

def metric_payload(m):return {k:float(v) for k,v in m.items() if k not in ("prob","pred")}

def correction_stats(base_logp,method_logp,ids,labels):
    import numpy as np
    ix=np.asarray(ids,dtype=np.int64);bp=base_logp.detach().exp().cpu().numpy()[ix];mp=method_logp.detach().exp().cpu().numpy()[ix]
    y=labels.detach().cpu().numpy()[ix];b=bp.argmax(1);m=mp.argmax(1);bc=b==y;mc=m==y
    return {"base_errors":int((~bc).sum()),"method_errors":int((~mc).sum()),"error_to_correct":int(((~bc)&mc).sum()),
            "correct_to_error":int((bc&(~mc)).sum()),"net_correction":int(((~bc)&mc).sum()-((bc)&(~mc)).sum()),
            "hard_flips":int((b!=m).sum()),"asd_error_corrected":int(((~bc)&mc&(y==0)).sum()),
            "td_error_corrected":int(((~bc)&mc&(y==1)).sum()),"correct_retention":float((bc&mc).sum()/max(int(bc.sum()),1)),
            "mean_abs_probability_delta":float(np.abs(mp-bp).mean())}

def _train_memory(module,optimizer,state,eligible,parts,memories,cam,base_logp):
    import torch
    a=parts["A"]["query_ids"];b=parts["B"]["query_ids"]
    if not memories or not memories.get("A",{}).get("ids") or not memories.get("B",{}).get("ids"):
        n=len(eligible);z=cam.new_zeros((n,144));return 0.0,{"evidence":0.0,"delta_ratio_mean":0.0,"query_count":n},0.0
    module.train();optimizer.zero_grad(set_to_none=True)
    base_parameters=[p for module in (state["model"],state["cam"]) for p in module.parameters()]
    original_base_grads=[None if p.grad is None else p.grad.detach().clone() for p in base_parameters]
    ia=torch.tensor(a,dtype=torch.long,device=cam.device);ib=torch.tensor(b,dtype=torch.long,device=cam.device)
    ea,da=module(a,cam[ia].detach(),base_logp[ia].detach(),memories["B"])
    eb,db=module(b,cam[ib].detach(),base_logp[ib].detach(),memories["A"])
    ids=a+b;h=torch.cat([cam[ia].detach(),cam[ib].detach()],0);lp=torch.cat([base_logp[ia].detach(),base_logp[ib].detach()],0)
    enhanced=torch.cat([ea,eb],0);evidence=torch.cat([da["evidence_logits"],db["evidence_logits"]],0)
    valid=torch.cat([da["evidence_valid"],db["evidence_valid"]],0)
    if not bool(valid.any()):
        optimizer.zero_grad(set_to_none=True)
        return 0.0,{"evidence":0.0,"evidence_queries":0,"query_count":len(ids),
                    "no_dual_class_support":len(ids),"delta_ratio_mean":0.0,"grad_norms":{}},0.0
    new_lp=output_layer_logp(state["cam"].Outputlayer,enhanced)
    loss,comp=weighted_memory_loss(new_lp,lp,state["y"],ids,state["weights"],evidence,valid,
                                   torch.cat([da["delta_h"],db["delta_h"]],0),h,CFG["loss"])
    with torch.no_grad():
        qlabels=state["y"][torch.as_tensor(ids,dtype=torch.long,device=cam.device)]
        evpred=evidence[valid].argmax(1);evy=qlabels[valid]
        comp["evidence_accuracy"]=float(evpred.eq(evy).float().mean()) if evy.numel() else 0.0
        comp["evidence_asd_accuracy"]=float(evpred[evy.eq(0)].eq(0).float().mean()) if evy.eq(0).any() else 0.0
        comp["evidence_td_accuracy"]=float(evpred[evy.eq(1)].eq(1).float().mean()) if evy.eq(1).any() else 0.0
        base_pred=lp.argmax(1)[valid]
        comp["base_error_overlap"]=int((base_pred.ne(evy)&evpred.ne(evy)).sum()) if evy.numel() else 0
    loss.backward()
    grad_norms={}
    for name in ("Wq","Wk","Wv","gate"):
        p=getattr(module,name).weight if name!="gate" else module.gate.weight
        grad_norms[name]=float(p.grad.detach().norm()) if p.grad is not None else 0.0
        if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError(f"invalid gradient {name}")
    if grad_norms["Wq"]<=0 or grad_norms["Wk"]<=0:raise RuntimeError(f"learned retrieval has no gradient: {grad_norms}")
    for before,p in zip(original_base_grads,base_parameters):
        if before is None:
            if p.grad is not None:raise RuntimeError("memory loss reached original OutputLayer parameters")
        elif p.grad is None or not torch.equal(before,p.grad):
            raise RuntimeError("memory loss changed existing base OutputLayer gradients")
    optimizer.step()
    comp.update({"grad_norms":grad_norms,"gate_mean":float(torch.cat([da["gate"],db["gate"]]).detach().mean()),
                 "memory_reliability_mean":float(torch.cat([da["memory_reliability"],db["memory_reliability"]]).detach().mean()),
                 "delta_ratio_p90":float(torch.quantile(torch.cat([da["delta_ratio"],db["delta_ratio"]]).detach(),.9))})
    return float(loss.detach()),comp,grad_norms["Wq"]+grad_norms["Wk"]

def _predict(module,state,ids,cam,base_logp,memory,intervention=None):
    import torch
    module.eval(); ix=torch.tensor(ids,dtype=torch.long,device=cam.device)
    with torch.no_grad():
        enhanced,diag=module(ids,cam[ix].detach(),base_logp[ix].detach(),memory,intervention=intervention)
        logp=output_layer_logp(state["cam"].Outputlayer,enhanced)
    return logp,enhanced,diag

def _memory_summary(mem):
    if not mem:return {"size":0,"asd":0,"td":0,"candidate":0,"accepted":0,"rejected":0,"reliability_mean":0.0}
    labs=mem.get("label",[]); rel=mem.get("reliability",[])
    return {"size":len(mem.get("ids",[])),"asd":int((labs==0).sum()) if hasattr(labs,"__eq__") else 0,
            "td":int((labs==1).sum()) if hasattr(labs,"__eq__") else 0,
            "candidate":int(mem.get("candidate_count",0)),"accepted":int(mem.get("accepted_count",0)),
            "rejected":int(mem.get("rejected_count",0)),"reliability_mean":float(rel.float().mean()) if len(rel) else 0.0}

def _save_payload(path,payload):atomic_torch(path,payload)

def continuation_signature(state,module,optimizer,memories,parts,eligible,epoch_index,version):
    """One genuine next step; used live before process exit and after reload."""
    import torch
    runtime=legacy.history_runtime();hist=legacy.load_base().load_history()
    before=rng_sha(runtime);hist.zero_optimizers(state)
    out,_=hook_forward(state,True);out["total"].backward()
    for key in ("opt_base","opt_cam","opt_added"):state[key].step()
    after_train=rng_sha(runtime)
    with torch.no_grad():evaluated,cam=hook_forward(state,False)
    after_eval=rng_sha(runtime);cam=cam.detach();base_lp=evaluated["logp"].detach()
    loss,diag,grad=_train_memory(module,optimizer,state,eligible,parts,memories,cam,base_lp)
    ids=list(map(int,state["test_np"]));pred,_,_=_predict(module,state,ids,cam,base_lp,memories["full"])
    epoch=epoch_index+1
    if epoch>=CFG["first_refresh_epoch"] and (epoch-CFG["first_refresh_epoch"])%CFG["memory_refresh_interval"]==0:
        version+=1
        memories=build_crossfit_memories(cam,state["y"].long(),base_lp,parts,site_tensor(cam.device),
            state["cam"].Outputlayer,version=version,refresh_epoch=epoch,config=CFG)
    return {"next_epoch_1based":epoch,"base_state_digest":content_digest(state_bundle(state)),
        "module_digest":content_digest(module.state_dict()),"optimizer_digest":content_digest(optimizer.state_dict()),
        "memory_digest":content_digest(memories),"prediction_digest":content_digest(pred),
        "memory_version":version,"rng_before":before,"rng_after_train":after_train,"rng_after_eval":after_eval,
        "rng_after_memory":rng_sha(runtime),"memory_loss":loss,"memory_gradient_norm":grad}

def checkpoint_payload(fold,epoch,ledger,state,module,optimizer,memory_used,memory_next,parts,base_metrics,method_metrics,
                       base_logp,method_logp,eligible,test_ids,identity,source,trace,transition,train_diag):
    import numpy as np
    return {"experiment":CFG["experiment"],"group":GROUP,"fold":fold,"seed":0,"base_group":BASE_GROUP,
            "ledger_kind":ledger,"source_sha256":source,"identity":identity,"best_epoch_1based":int(epoch),
            "metrics":metric_payload(method_metrics),"probability":np.asarray(method_metrics["prob"],float),
            "prediction":np.asarray(method_metrics["pred"],int),"base_probability":np.asarray(base_metrics["prob"],float),
            "base_prediction":np.asarray(base_metrics["pred"],int),"base_state":state_bundle(state),
            "module":{k:v.detach().cpu().clone() for k,v in module.state_dict().items()},"optimizer":copy.deepcopy(optimizer.state_dict()),
            "memory_used":detach_memory(memory_used),"memory_next":detach_memory(memory_next),"partitions":copy.deepcopy(parts),
            "eligible_train_ids":list(map(int,eligible)),"test_ids":list(map(int,test_ids)),
            "fold2_exclude":sorted(FOLD2_EXCLUDE) if fold==2 else [],"transition":transition,
            "train_diagnostics":copy.deepcopy(train_diag),"rng_trace":copy.deepcopy(trace)}

def train_fold(fold):
    import numpy as np,torch
    state,warm,identity,hist_task,hist_ckpt=prepare_fold(fold);runtime=legacy.history_runtime();hist=legacy.load_base().load_history()
    if rng_sha(runtime)!=warm["rng_sha"]:raise RuntimeError(f"E61 RNG restore mismatch fold={fold}")
    eligible=eligible_ids(state,fold);test_ids=list(map(int,state["test_np"])); y=state["y"].long();sites=site_tensor(y.device)
    parts=make_partitions(eligible,y.detach().cpu(),sites.detach().cpu()); source=source_sha()
    history_best_epoch=int(hist_task["best_epoch_1based"]);history_best=torch_load(hist_ckpt)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        module=init_module(y.device)
    optimizer=torch.optim.AdamW(module.parameters(),lr=CFG["optimizer"]["lr"],weight_decay=CFG["optimizer"]["weight_decay"])
    memories=None;version=0;base_best_acc=-1.0;base_best_epoch=-1;base_wait=0;method_best_acc=-1.0;method_best_epoch=-1
    method_best=None;anchored=None;trace=[];rows=[];memory_rows=[];train_diag_last={};start_epoch=10
    latest=CKPTS/f"E79.4-G1_f{fold:02d}.latest.pt";task_path=TASKS/f"E79.4-G1_f{fold:02d}.json"
    if latest.is_file():
        saved=torch_load(latest)
        if saved.get("source_sha256")!=source or saved.get("identity")!=identity:raise RuntimeError("latest identity/source mismatch")
        hist.load_bundle(state,saved["base_state"]);runtime.restore_rng(saved["rng"]);module.load_state_dict(saved["module"]);optimizer.load_state_dict(saved["optimizer"])
        memories=saved["memories"];version=int(saved["memory_version"]);base_best_acc=saved["base_best_acc"];base_best_epoch=saved["base_best_epoch"]
        base_wait=int(saved["base_wait"]);method_best_acc=saved["method_best_acc"];method_best_epoch=saved["method_best_epoch"]
        method_best=saved["method_best"];anchored=saved["anchored"];trace=saved["trace"];rows=saved["rows"]
        memory_rows=saved["memory_rows"];train_diag_last=saved.get("train_diag_last",{});start_epoch=int(saved["next_epoch_index"])
    else:
        atomic_json(task_path,{"status":"RUNNING","fold":fold,"source_sha256":source,"identity":identity,
                              "history_best_epoch_1based":history_best_epoch,"eligible_count":len(eligible),
                              "partition_sizes":{"A":len(parts["A"]["query_ids"]),"B":len(parts["B"]["query_ids"])},"next_epoch_index":start_epoch})
    if start_epoch<10 or start_epoch>450:raise RuntimeError(f"invalid resume index {start_epoch}")
    started=time.time();maximum=int(os.environ.get("E794_MAX_EPOCH_EXCLUSIVE",CFG["training"]["max_epoch_exclusive"]))
    hist_trace={int(x["epoch"]):x for x in hist_task.get("rng_trace",[])}
    for epoch_index in range(start_epoch,maximum):
        epoch=epoch_index+1;rb=rng_sha(runtime);hist.zero_optimizers(state)
        train_out,_=hook_forward(state,True)
        if not torch.isfinite(train_out["total"]):raise FloatingPointError(f"base loss nonfinite fold={fold} epoch={epoch}")
        train_out["total"].backward()
        for key in ("opt_base","opt_cam","opt_added"):state[key].step()
        rat=rng_sha(runtime)
        with torch.no_grad():evaluated,cam=hook_forward(state,False)
        rae=rng_sha(runtime);base_logp=evaluated["logp"].detach();cam=cam.detach()
        reference_rng=hist_trace.get(epoch)
        if reference_rng is None or any(reference_rng[k]!=value for k,value in (("rng_before",rb),("rng_after_train",rat),("rng_after_eval",rae))):
            raise RuntimeError(f"historical shared RNG mismatch fold={fold} epoch={epoch}")
        if tuple(cam.shape)!=(871,144):raise RuntimeError("CAM144 shape mismatch")
        base_metrics=legacy.extended_metrics(base_logp,test_ids,y)
        loss,train_diag,grad_summary=_train_memory(module,optimizer,state,eligible,parts,memories,cam,base_logp)
        train_diag_last=train_diag
        method_lp_test,enh_test,diag=_predict(module,state,test_ids,cam,base_logp,memories["full"] if memories else None)
        full=base_logp.clone();ti=torch.tensor(test_ids,dtype=torch.long,device=y.device);full[ti]=method_lp_test
        method_metrics=legacy.extended_metrics(full,test_ids,y)
        if rng_sha(runtime)!=rae:raise RuntimeError(f"memory branch consumed shared RNG fold={fold} epoch={epoch}")
        refresh=(epoch>=CFG["first_refresh_epoch"] and ((epoch-CFG["first_refresh_epoch"])%CFG["memory_refresh_interval"]==0))
        memory_used=detach_memory(memories);memory_next=memories
        if refresh:
            version+=1
            memory_next=build_crossfit_memories(cam,y,base_logp,parts,sites,state["cam"].Outputlayer,
                                                version=version,refresh_epoch=epoch,config=CFG)
            memories=memory_next
        transition=correction_stats(base_logp,full,test_ids,y)
        rngrec={"epoch":epoch,"rng_before":rb,"rng_after_train":rat,"rng_after_eval":rae};trace.append(rngrec)
        if float(base_metrics["acc"])>base_best_acc:base_best_acc=float(base_metrics["acc"]);base_best_epoch=epoch;base_wait=0
        else:base_wait+=1
        if epoch==history_best_epoch:
            if not same_state(state_bundle(state),history_best["best_state"]):raise RuntimeError(f"base-anchored DARG state mismatch fold={fold}")
            if not np.allclose(np.asarray(base_metrics["prob"]),np.asarray(history_best["test_probability"]),atol=1e-8):raise RuntimeError(f"history prediction mismatch fold={fold}")
            anchored=checkpoint_payload(fold,epoch,"base_anchored",state,module,optimizer,memory_used,memory_next,parts,
                                        base_metrics,method_metrics,base_logp,full,eligible,test_ids,identity,source,trace,transition,train_diag)
            _save_payload(CKPTS/f"E79.4-G1_f{fold:02d}.base_anchored.pt",anchored)
        if float(method_metrics["acc"])>method_best_acc:
            method_best_acc=float(method_metrics["acc"]);method_best_epoch=epoch
            method_best=checkpoint_payload(fold,epoch,"method",state,module,optimizer,memory_used,memory_next,parts,
                                           base_metrics,method_metrics,base_logp,full,eligible,test_ids,identity,source,trace,transition,train_diag)
            _save_payload(CKPTS/f"E79.4-G1_f{fold:02d}.method.pt",method_best)
        mem_summary={k:_memory_summary(v) for k,v in (memory_used or {}).items()}
        row={"fold":fold,"epoch":epoch,"base_acc":float(base_metrics["acc"]),"method_acc":float(method_metrics["acc"]),
             "base_wait":base_wait,"memory_version":version,"refresh":refresh,"memory_loss":loss,"memory_gradient_norm":grad_summary,
             "base_loss":float(train_out["total"].detach()),"base_parameter_digest":content_digest({"model":state["model"].state_dict(),"cam":state["cam"].state_dict()}),
             "rng_before":rb,"rng_after_train":rat,"rng_after_eval":rae,"gate_mean":float(diag["gate"].mean()),
             "delta_ratio_mean":float(diag["delta_ratio"].mean()),"delta_ratio_p90":float(torch.quantile(diag["delta_ratio"],.9)),
             "cap_fraction":float(diag["cap_fraction"].mean()),"memory_reliability":float(diag["memory_reliability"].mean()),
             "evidence_loss":train_diag.get("evidence",0.0),"gradients":train_diag.get("grad_norms",{}),"memory":mem_summary}
        rows.append(row)
        if refresh:
            for name,m in memories.items():memory_rows.append({"fold":fold,"epoch":epoch,"bank":name,**_memory_summary(m),
                "version":m.get("version"),"source_digest":m.get("source_ids_digest"),"support_digest":m.get("support_ids_digest")})
        with (LOGS/f"E79.4-G1_f{fold:02d}.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
        if epoch%10==0 or refresh or epoch==history_best_epoch or base_wait>CFG["training"]["early_stop_wait_gt"] or epoch==450:
            atomic_torch(latest,{"experiment":CFG["experiment"],"fold":fold,"source_sha256":source,"identity":identity,
                "next_epoch_index":epoch_index+1,"base_wait":base_wait,"base_state":state_bundle(state),"rng":runtime.pack_rng(),
                "module":{k:v.detach().cpu().clone() for k,v in module.state_dict().items()},"optimizer":copy.deepcopy(optimizer.state_dict()),
                "memories":detach_memory(memories),"memory_version":version,"partitions":parts,"base_best_acc":base_best_acc,
                "base_best_epoch":base_best_epoch,"method_best_acc":method_best_acc,"method_best_epoch":method_best_epoch,
                "method_best":method_best,"anchored":anchored,"trace":trace,"rows":rows,"memory_rows":memory_rows,
                "train_diag_last":train_diag_last})
            atomic_json(task_path,{"status":"RUNNING","fold":fold,"source_sha256":source,"identity":identity,
                "next_epoch_index":epoch_index+1,"base_best_epoch_1based":base_best_epoch,"method_best_epoch_1based":method_best_epoch,
                "history_best_epoch_1based":history_best_epoch,"rng_records":len(trace),"memory_version":version,
                "refreshes":sum(1 for x in memory_rows if x["bank"]=="full")})
        if base_wait>CFG["training"]["early_stop_wait_gt"]:break
    if maximum<CFG["training"]["max_epoch_exclusive"]:return {"fold":fold,"smoke":True,"epochs":len(rows)}
    hist_trace={int(x["epoch"]):x for x in hist_task.get("rng_trace",[])}
    mismatches=[e for e,r in hist_trace.items() if e in {int(x["epoch"]) for x in trace} and any(r.get(k)!=next(t for t in trace if int(t["epoch"])==e).get(k) for k in ("rng_before","rng_after_train","rng_after_eval"))]
    if mismatches:raise RuntimeError(f"historical shared RNG mismatch fold={fold}: {mismatches[:5]}")
    if base_best_epoch!=history_best_epoch:raise RuntimeError(f"historical base best epoch mismatch fold={fold}: {base_best_epoch}!={history_best_epoch}")
    if anchored is None or method_best is None:raise RuntimeError(f"missing method ledger fold={fold}")
    task={"status":"COMPLETE","fold":fold,"source_sha256":source,"identity":identity,"eligible_count":len(eligible),
          "history_best_epoch_1based":history_best_epoch,"base_best_epoch_1based":base_best_epoch,
          "method_best_epoch_1based":method_best_epoch,"epochs_executed":len(rows),"rng_records":len(trace),
          "rng_mismatches":len(mismatches),"memory_refreshes":sum(1 for x in memory_rows if x["bank"]=="full"),
          "elapsed_seconds":time.time()-started,"max_workers":int(os.environ.get("E794_WORKERS","1")),
          "peak_gpu_bytes":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
          "device":torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
          "memory_version":version,"base_anchored_metrics":anchored["metrics"],"method_metrics":method_best["metrics"]}
    atomic_json(task_path,task)
    atomic_torch(latest,{"experiment":CFG["experiment"],"fold":fold,"source_sha256":source,"identity":identity,
        "next_epoch_index":trace[-1]["epoch"],"base_wait":base_wait,"base_state":state_bundle(state),"rng":runtime.pack_rng(),
        "module":{k:v.detach().cpu().clone() for k,v in module.state_dict().items()},"optimizer":copy.deepcopy(optimizer.state_dict()),
        "memories":detach_memory(memories),"memory_version":version,"partitions":parts,"base_best_acc":base_best_acc,
        "base_best_epoch":base_best_epoch,"method_best_acc":method_best_acc,"method_best_epoch":method_best_epoch,
        "method_best":method_best,"anchored":anchored,"trace":trace,"rows":rows,"memory_rows":memory_rows,
        "train_diag_last":train_diag_last})
    # Save a live, never-reloaded next-step reference. Independent audit reloads
    # Latest, executes the same actual epoch, and compares every state digest.
    reference=continuation_signature(state,module,optimizer,memories,parts,eligible,trace[-1]["epoch"],version)
    atomic_json(TASKS/f"E79.4-G1_f{fold:02d}.continuation.json",reference)
    return task

def preflight():
    import torch
    hashes=source_snapshot();fold_rows=[]
    for fold in range(1,11):
        state,warm,identity,task,ckpt=prepare_fold(fold); eligible=eligible_ids(state,fold)
        y=state["y"].detach().cpu();sites=site_tensor(y.device).detach().cpu();parts=make_partitions(eligible,y,sites)
        tests=set(map(int,state["test_np"]));
        if any(set(v["bank_ids"])&tests for k,v in parts.items()):raise RuntimeError(f"test overlap in partition f={fold}")
        if any(set(v["source_ids"])&set(v["support_ids"]) for v in parts.values()):raise RuntimeError(f"source/support overlap f={fold}")
        if any(len(v["support_ids"])!=int(round(len(v["bank_ids"])*.25)) for v in parts.values()):raise RuntimeError(f"source/support ratio mismatch f={fold}")
        if fold==2 and any(set(v["bank_ids"])&FOLD2_EXCLUDE for v in parts.values()):raise RuntimeError("Fold2 excluded rows present")
        fold_rows.append({"fold":fold,"history_best_epoch_1based":task["best_epoch_1based"],"warmup_sha256":sha_file(E61/f".runtime/warmup/fold{fold:02d}.pt"),
                          "identity":identity,"eligible_count":len(eligible),"partition_sizes":{k:{"bank":len(v["bank_ids"]),"source":len(v["source_ids"]),"support":len(v["support_ids"])} for k,v in parts.items()},
                          "base_checkpoint_sha256":sha_file(ckpt)})
    state,warm,identity,task,ckpt=prepare_fold(1);runtime=legacy.history_runtime();before=rng_sha(runtime)
    with torch.no_grad():out,cam=hook_forward(state,False)
    after=rng_sha(runtime)
    if before!=after or tuple(cam.shape)!=(871,144):raise RuntimeError("base eval/CAM preflight failure")
    eligible=eligible_ids(state,1); y_all=state["y"].long(); sites_all=site_tensor(cam.device)
    parts=make_partitions(eligible,y_all.detach().cpu(),sites_all.detach().cpu())
    actual_memories=build_crossfit_memories(cam.detach(),y_all,out["logp"].detach(),parts,sites_all,
        state["cam"].Outputlayer,version=1,refresh_epoch=11,config=CFG)
    for bank_name,mem in actual_memories.items():
        if set(mem.get("source_ids",[]))&set(mem.get("support_ids",[])):raise RuntimeError(f"source/support overlap in {bank_name}")
        if not set(mem.get("ids",[])).issubset(set(mem.get("source_ids",[]))):raise RuntimeError(f"memory candidate outside Source in {bank_name}")
        if len(mem.get("ids",[]))!=int(mem.get("accepted_count",0)):raise RuntimeError(f"memory accepted count mismatch {bank_name}")
        if mem.get("ids") and (not torch.isfinite(mem["reliability"]).all() or (mem["reliability"]<=0).any()):raise RuntimeError(f"invalid memory reliability {bank_name}")
    if set(parts["A"]["query_ids"])&set(actual_memories["B"]["bank_ids"]):raise RuntimeError("A query could enter B memory bank")
    if set(parts["B"]["query_ids"])&set(actual_memories["A"]["bank_ids"]):raise RuntimeError("B query could enter A memory bank")
    memory_rng_before=rng_sha(runtime)
    training_module=init_module(cam.device)
    training_optimizer=torch.optim.AdamW(training_module.parameters(),lr=CFG["optimizer"]["lr"],weight_decay=CFG["optimizer"]["weight_decay"])
    memory_loss_value,memory_train_diag,memory_grad_norm=_train_memory(training_module,training_optimizer,state,eligible,parts,
        actual_memories,cam.detach(),out["logp"].detach())
    memory_rng_after=rng_sha(runtime)
    if memory_rng_before!=memory_rng_after:raise RuntimeError("memory branch consumed shared base RNG")
    if memory_grad_norm<=0 or not all(memory_train_diag.get("grad_norms",{}).get(k,0)>0 for k in ("Wq","Wk","Wv","gate")):
        raise RuntimeError(f"actual cross-fit memory did not train all paths: {memory_train_diag}")
    module=init_module(cam.device);optimizer=torch.optim.AdamW(module.parameters(),lr=CFG["optimizer"]["lr"],weight_decay=CFG["optimizer"]["weight_decay"])
    # Re-run the evidence-only objective with a deterministic synthetic bank to
    # prove gradients reach both trainable projections while CAM/head stay frozen.
    ids=[int(x) for x in state["train_np"] if int(x) not in set(map(int,state["test_np"]))][:24]
    if len(ids)<12:raise RuntimeError("insufficient preflight training queries")
    labels=state["y"].long();qids=ids[:8];h=cam[torch.tensor(qids,device=cam.device)].detach().requires_grad_(False)
    # Use a memory made from distinct training patients with both labels.
    memids=[]
    for cls in (0,1):
        memids.extend([int(p) for p in ids if int(labels[p])==cls and int(p) not in qids][:8])
    if len(memids)<8:raise RuntimeError("preflight lacks two-class synthetic memory")
    mi=torch.tensor(memids,device=cam.device);ml=labels[mi].detach();masks=torch.zeros((len(memids),144),device=cam.device);masks[:,:12]=1
    memory={"ids":memids,"label":ml.cpu(),"reference":cam[mi].detach().cpu(),"mask":masks.cpu(),
            "direction":torch.zeros_like(masks).cpu(),"reliability":torch.full((len(memids),),.8),
            "neutral_mean":cam[torch.tensor(ids,device=cam.device)].mean(0).detach().cpu()}
    enhanced,diag=module(qids,h,out["logp"][torch.tensor(qids,device=cam.device)].detach(),memory)
    valid=diag["evidence_valid"]
    if not bool(valid.any()):raise RuntimeError("preflight evidence candidates unavailable")
    yq=labels[torch.tensor(qids,device=cam.device)]
    evloss=torch.nn.functional.cross_entropy(diag["evidence_logits"][valid],yq[valid])
    grads=torch.autograd.grad(evloss,(module.Wq.weight,module.Wk.weight),allow_unused=True)
    qn=float(grads[0].norm()) if grads[0] is not None else 0.;kn=float(grads[1].norm()) if grads[1] is not None else 0.
    if qn<=0 or kn<=0 or not torch.isfinite(grads[0]).all() or not torch.isfinite(grads[1]).all():raise RuntimeError(f"Wq/Wk evidence gradient failed {qn},{kn}")
    old_q=module.Wq.weight.detach().clone();old_k=module.Wk.weight.detach().clone()
    with torch.no_grad():
        module.Wq.weight[0,0]+=0.05;module.Wk.weight[1,1]-=0.05
        _,perturbed=module(qids,h,out["logp"][torch.tensor(qids,device=cam.device)].detach(),memory)
        matching_perturbation=float((perturbed["evidence_logits"]-diag["evidence_logits"]).abs().max())
        module.Wq.weight.copy_(old_q);module.Wk.weight.copy_(old_k)
    if matching_perturbation<=1e-8:raise RuntimeError("Wq/Wk perturbation did not change evidence scores")
    if any(p.grad is not None for p in state["cam"].Outputlayer.parameters()):raise RuntimeError("OutputLayer received evidence-only gradient")
    empty,ed=module(qids,h,out["logp"][torch.tensor(qids,device=cam.device)].detach(),None)
    if not torch.equal(empty,h):raise RuntimeError("empty memory must be exact CAM identity")
    single=detach_memory(actual_memories["A"])
    first_label=int(single["label"][0]); keep=torch.where(single["label"].eq(first_label))[0]
    record_fields=("ids","label","reference","mask","direction","reliability","r_self","r_within","r_between",
                   "r_coverage","support_auc","support_separation","joint_drop","support_same_ids","support_opposite_ids")
    for key in record_fields:
        value=single[key]
        single[key]=[value[int(i)] for i in keep.tolist()] if isinstance(value,list) else value[keep]
    if len(single["ids"])==0 or len(set(int(v) for v in single["label"].tolist()))!=1:
        raise RuntimeError("single-class fallback preflight could not construct a one-class bank")
    one_class_memories={"A":single,"B":detach_memory(single)}
    single_loss,single_diag,single_grad=_train_memory(module,optimizer,state,eligible,parts,one_class_memories,
        cam.detach(),out["logp"].detach())
    if single_loss!=0.0 or single_grad!=0.0 or single_diag.get("no_dual_class_support")!=len(eligible):
        raise RuntimeError("single-class bank must skip branch training and preserve the base path")
    single_out,_=module(qids,h,out["logp"][torch.tensor(qids,device=cam.device)].detach(),single)
    if not torch.equal(single_out,h):raise RuntimeError("single-class memory must be exact CAM identity")
    real_fold_checks=[]
    for test_fold in range(1,11):
        fs,fw,fi,ft,fc=prepare_fold(test_fold);fr=legacy.history_runtime();fh=legacy.load_base().load_history()
        initial=state_bundle(fs);initial_rng=fr.pack_rng()
        def base_step(with_hook):
            fh.zero_optimizers(fs)
            train=(hook_forward(fs,True)[0] if with_hook else legacy.load_base().forward(fs,BASE_GROUP,training=True))
            train["total"].backward()
            for key in ("opt_base","opt_cam","opt_added"):fs[key].step()
            train_rng=rng_sha(fr)
            with torch.no_grad():
                if with_hook:evaluation,features=hook_forward(fs,False)
                else:evaluation=legacy.load_base().forward(fs,BASE_GROUP,training=False);features=None
            return evaluation,features,train_rng,rng_sha(fr)
        ref_eval,_,ref_train_rng,ref_eval_rng=base_step(False);ref_state=state_bundle(fs)
        fh.load_bundle(fs,initial);fr.restore_rng(initial_rng)
        fe,fh144,train_rng,eval_rng=base_step(True)
        same_base=same_state(state_bundle(fs),ref_state) and torch.equal(fe["logp"],ref_eval["logp"])
        expected=next(r for r in ft["rng_trace"] if r["epoch"]==11)
        if not same_base or train_rng!=ref_train_rng or eval_rng!=ref_eval_rng or train_rng!=expected["rng_after_train"] or eval_rng!=expected["rng_after_eval"]:
            raise RuntimeError(f"actual base one-step identity failed fold={test_fold}")
        fy=fs["y"].long();fel=eligible_ids(fs,test_fold);fsite=site_tensor(fy.device);fp=make_partitions(fel,fy.cpu(),fsite.cpu())
        fm=build_crossfit_memories(fh144.detach(),fy,fe["logp"].detach(),fp,fsite,fs["cam"].Outputlayer,version=1,refresh_epoch=11,config=CFG)
        fmodule=init_module(fy.device);fopt=torch.optim.AdamW(fmodule.parameters(),lr=CFG["optimizer"]["lr"],weight_decay=CFG["optimizer"]["weight_decay"])
        fbefore=state_bundle(fs);frng=rng_sha(fr)
        fl,fd,fg=_train_memory(fmodule,fopt,fs,fel,fp,fm,fh144.detach(),fe["logp"].detach())
        if not same_state(fbefore,state_bundle(fs)) or frng!=rng_sha(fr):raise RuntimeError(f"memory isolation failed fold={test_fold}")
        real_fold_checks.append({"fold":test_fold,"base_one_step_exact":same_base,"base_rng_matches_history_epoch11":train_rng==expected["rng_after_train"] and eval_rng==expected["rng_after_eval"],
            "memory_shapes":{k:_memory_summary(v) for k,v in fm.items()},"memory_gradient_norm":fg,"diagnostics":fd,
            "memory_preserved_base_state_and_rng":same_state(fbefore,state_bundle(fs)) and frng==rng_sha(fr)})
        print(f"preflight actual fold={test_fold} memory_sizes={[len(v['ids']) for v in fm.values()]} evidence_queries={fd.get('evidence_queries',0)}",flush=True)
    payload={"status":"PASS","source_hashes":hashes,"folds":fold_rows,"cam_shape":list(cam.shape),
             "base_eval_rng_before":before,"base_eval_rng_after":after,"partition_disjointness":"PASS",
             "fold2_exclusion":"PASS","wq_evidence_grad_norm":qn,"wk_evidence_grad_norm":kn,
             "wq_wk_perturbation_max_score_change":matching_perturbation,
             "actual_crossfit_memory_loss":memory_loss_value,"actual_crossfit_memory_diagnostics":memory_train_diag,
             "memory_rng_before":memory_rng_before,"memory_rng_after":memory_rng_after,
             "actual_fold1_memory_build":{k:_memory_summary(v) for k,v in actual_memories.items()},
             "empty_memory_exact":True,"single_class_memory_exact":True,"single_class_training_skipped":True,
             "real_fold_one_step_checks":real_fold_checks,
             "text_prompt":False,"oof_teacher":False,"timestamp":time.time()}
    atomic_json(RESULTS/"preflight.json",payload);print(json.dumps({"status":"PASS","Wq_grad":qn,"Wk_grad":kn},indent=2));return payload

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("preflight","train","train-all"));ap.add_argument("--fold",type=int);args=ap.parse_args()
    if args.command=="preflight":preflight()
    elif args.command=="train":
        if not args.fold:raise SystemExit("--fold required")
        train_fold(args.fold)
    else:
        for fold in range(1,11):train_fold(fold)

if __name__=="__main__":main()
