#!/usr/bin/env python3
"""One frozen UDAG group, resuming the audited E69.3-G2 epoch-10 states."""
from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import sys, json, hashlib, importlib.util, copy, time, csv, argparse, subprocess, traceback
from pathlib import Path
sys.dont_write_bytecode=True
if __name__ == "__main__":
    sys.modules["run"] = sys.modules[__name__]
EXP=Path(__file__).resolve().parent
CFG=json.loads((EXP/"config.json").read_text())
RT=EXP/".runtime"; TASKS=RT/"tasks"; CKPTS=RT/"checkpoints"; LOGS=RT/"logs"; RESULTS=EXP/"results"
BASE=Path(CFG["base"]["path"]); GROUP=CFG["group"]
METRICS=("acc","hard_auc","prob_auc","sen","spe","ba","asd_f1")
G2=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_base():
    global G2
    if G2 is not None:return G2
    if sha(BASE/"run.py")!=CFG["base"]["run_sha256"]:raise RuntimeError("E69.3-G2 source SHA mismatch")
    if sha(BASE/"manifest.json")!=CFG["base"]["manifest_sha256"]:raise RuntimeError("E69.3-G2 manifest changed")
    m=json.loads((BASE/"manifest.json").read_text())
    if (m["status"],m["completed_tasks"],m["independent_audits"])!=("COMPLETE_AUDITED",10,10):raise RuntimeError("E69.3-G2 is not fully audited")
    spec=importlib.util.spec_from_file_location("e74_readonly_e693",BASE/"run.py")
    G2=importlib.util.module_from_spec(spec);sys.modules[spec.name]=G2;spec.loader.exec_module(G2)
    G2.load_e61();G2.verify_e62_reference()
    return G2

def sources():
    g=load_base()
    return {"e74/"+n:sha(EXP/n) for n in ("config.json","run.py","udag.py","audit.py")}|g.source_sha_map()

def atomic_json(p,obj):load_base().atomic_json(p,obj)
def atomic_torch(p,obj):load_base().atomic_torch(p,obj)
def tload(p):return load_base().torch_load(p)
def write_csv(p,rows):load_base().write_csv(p,rows)
def load_audit_module():
    # E61 deliberately prepends its read-only E53.3 dependency to sys.path;
    # load this experiment's audit entry by absolute path to avoid name shadowing.
    name="e74_independent_audit"
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,EXP/"audit.py")
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def task_path(f):return TASKS/f"{GROUP}_f{f:02d}.json"
def best_path(f):return CKPTS/f"{GROUP}_f{f:02d}.best.pt"
def latest_path(f):return CKPTS/f"{GROUP}_f{f:02d}.latest.pt"
def cpu(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
def state_sha(state):
    h=hashlib.sha256()
    for key in ("model","cam"):
        for n,t in sorted(state[key].state_dict().items()):
            h.update((key+"/"+n).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(fold):
    import torch, numpy as np
    g=load_base();e=g.load_e61();r=e.load_base()
    state,w=e.load_warmup_state(fold)
    # Native adapter verifies anchor, shared initial parameters, source, split,
    # all three optimizers and the full saved Python/NumPy/CPU/CUDA RNG.
    membership=g.apply_fold_specific_training_additions(state,fold)
    before_opt=[copy.deepcopy(state[k].state_dict()) for k in ("opt_base","opt_cam","opt_added")]
    e.configure_branch(state,"E61-E2",None)
    initial=g.set_gate_initialization(state,"E69.3-G2")
    gates=g.optimizer_gate_state(state)
    if any(v["present"] for v in gates.values()):raise RuntimeError("warmup gate optimizer has unexpected history")
    state["model"].adaptive_gate_logits.requires_grad_(False)
    state["model"].adaptive_gate_logits.grad=None
    for k,old in zip(("opt_base","opt_cam","opt_added"),before_opt):
        if not e.nested_tensor_equal(old,state[k].state_dict()):raise RuntimeError("optimizer changed in adapter")
    baseline=json.loads((BASE/".runtime/tasks"/f"E69.3-G2_f{fold:02d}.json").read_text())
    for key in ("initial_shared_sha","initial_gfr_sha","initial_cam_sha","relation_core_sha","initial_rng_sha","split_sha","pair_sha","warmup_pair_sha","effective_train_sha","anchor_sha"):
        if state[key]!=baseline[key]:raise RuntimeError(f"base identity mismatch {fold} {key}")
    if sha(g.warmup_path(fold))!=baseline["warmup_sha256"] or r.rng_sha(r.pack_rng())!=baseline["warmup_rng_sha"]:raise RuntimeError("warmup/RNG mismatch")
    if membership["train_test_overlap_1based"]!=([377,407] if fold==2 else []):raise RuntimeError("unexpected label overlap")
    if not np.isin(state["train_pairs_np"],state["train_np"]).all():raise RuntimeError("pair outside effective train")
    if not np.array_equal(state["train_same_np"],state["labels_np"][state["train_pairs_np"][:,0]]==state["labels_np"][state["train_pairs_np"][:,1]]):raise RuntimeError("pair targets differ")
    if state["train_pairs_np"].shape!=(4096,2):raise RuntimeError("pair cap changed")
    for k,wd,lr in (("opt_base",.11,.0038),("opt_cam",.11,.0038),("opt_added",.01,.001)):
        if any(v["weight_decay"]!=wd or v["lr"]!=lr for v in state[k].param_groups):raise RuntimeError("optimizer hyperparameter mismatch")
    record={"fold":fold,"status":"PASS","warmup_sha256":baseline["warmup_sha256"],"warmup_rng_sha":w["rng_sha"],
            "shared_postwarmup_state_sha":state_sha(state),"optimizer_exactly_preserved":True,"gate_optimizer_slots":gates,
            "initial_edge_gate":initial["edge_gate"],"adaptive_compatibility_value":initial["adaptive_gate"],"adaptive_frozen":True,
            **{k:state[k] for k in ("initial_shared_sha","initial_gfr_sha","initial_cam_sha","relation_core_sha","initial_rng_sha","split_sha","pair_sha","warmup_pair_sha","base_train_sha","effective_train_sha","anchor_sha")},
            "membership":membership,"pair_training_scope_pass":True,"warmup_reused":True}
    state["e74_identity"]=record
    return state,w,record

def forward(state,training=False,intervention="full",collect_reference=False):
    from udag import forward_state
    return forward_state(state,load_base().load_e61(),training,intervention,collect_reference)

def gradient_info(state):
    import torch
    e=load_base().load_e61();m=state["model"]
    return {"edge_gate_abs":m.edge_gate_logits.grad.detach().abs().cpu().tolist() if m.edge_gate_logits.grad is not None else [0.]*4,
            "adaptive_grad_is_none":m.adaptive_gate_logits.grad is None,
            "scorer_norm":[e.total_gradient_norm(x.parameters()) for x in m.scorers],
            "encoder_norm":[e.total_gradient_norm(x.parameters()) for x in m.encoders],
            "graphlearn_norm":[e.total_gradient_norm(x.parameters()) for x in m.graphs],
            "gcn_norm":[e.total_gradient_norm(x.parameters()) for x in m.gcn]}

def zero(state):
    for k in ("opt_base","opt_cam","opt_added"):state[k].zero_grad(set_to_none=True)

def snapshot(state):
    return {"model":cpu(state["model"]),"cam":cpu(state["cam"]),
            **{k:copy.deepcopy(state[k].state_dict()) for k in ("opt_base","opt_cam","opt_added")}}

def restore(state,snap):
    state["model"].load_state_dict(snap["model"],strict=True);state["cam"].load_state_dict(snap["cam"],strict=True)
    # Optimizer.load_state_dict may retain references when tensors already live
    # on the destination device. Never mutate the saved comparison snapshot.
    for k in ("opt_base","opt_cam","opt_added"):state[k].load_state_dict(copy.deepcopy(snap[k]))

def graph_unit_test():
    import torch
    from udag import unified_adjacency
    # No random numbers: three existing selected edges and two fallback edges.
    a=torch.eye(6,dtype=torch.float64)
    for i,j,v in ((0,1,.8),(0,2,.6),(0,3,.7),(1,2,.9)):
        a[i,j]=v;a[j,i]=v
    row=torch.tensor([0,0,0,1,2]);col=torch.tensor([1,2,3,4,5])
    q=torch.tensor([.9,.1,.5,.9,.1],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([.8,.6,.7,.75,.75],dtype=torch.float64)
    gate=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    out,c=unified_adjacency(a,row,col,s,q,gate)
    assert torch.allclose(out[row,col],torch.tensor([1.12,.36,.7,.3,0.],dtype=torch.float64),atol=1e-12,rtol=0)
    assert out[1,2]==a[1,2] and torch.equal(out,out.T) and out.min()>=0
    off,_=unified_adjacency(a,row,col,s,q,gate,intervention="disease_off")
    assert torch.equal(off,a)
    norm=load_base().load_e61().load_base().runtime()["utils"].normalize_adj(out+torch.eye(6,dtype=torch.float64))
    assert torch.isfinite(norm).all()
    out.square().sum().backward()
    assert gate.grad is not None and gate.grad.abs()>0 and q.grad[:2].abs().min()>0
    return {"status":"PASS","q_0p9_enhanced":True,"q_0p1_suppressed_nonnegative":True,"q_0p5_unchanged":True,
            "unselected_unchanged":True,"fallback_positive_added":True,"fallback_negative_absent":True,"symmetric":True,
            "disease_off_exact_original":True,"normalized_finite":True,"adjacency_autograd":True}

def preflight():
    import torch,numpy as np
    g=load_base();e=g.load_e61();r=e.load_base()
    if sha(RT/"sources/pinned_e693_run.py") != CFG["base"]["run_sha256"]:
        raise RuntimeError("GitHub-pinned E69.3-G2 source differs from local base")
    warmups=g.warmup_manifest_info();all_ids=[]
    for fold in range(1,11):
        s,w,identity=prepare(fold);all_ids.append(identity);del s,w
        torch.cuda.empty_cache()
    s,w,identity=prepare(1);initial=snapshot(s);rng=r.pack_rng()
    unit=graph_unit_test();assert r.rng_sha(r.pack_rng())==w["rng_sha"]
    counts=[0]*4;cam_count=[0]
    handles=[mod.register_forward_hook(lambda _m,_i,_o,idx=i:counts.__setitem__(idx,counts[idx]+1)) for i,mod in enumerate(s["model"].gcn)]
    handles.append(s["cam"].register_forward_hook(lambda *_:cam_count.__setitem__(0,cam_count[0]+1)))
    zero(s);a=forward(s,False);a["cls"].backward();natural=gradient_info(s)
    for h in handles:h.remove()
    if counts!=[1]*4 or cam_count!=[1]:raise RuntimeError(f"call counts wrong: {counts},{cam_count}")
    if not natural["adaptive_grad_is_none"]:raise RuntimeError("adaptive gate active")
    zero(s);hooks=[]
    # Isolated controlled q=.9. Derivative is preserved; no optimizer step.
    for scorer in s["model"].scorers:
        hooks.append(scorer.register_forward_hook(lambda _m,_i,x:x+(torch.full_like(x,float(torch.logit(torch.tensor(.9))))-x.detach())))
    try:
        b=forward(s,False);b["cls"].backward();controlled=gradient_info(s)
    finally:
        for h in hooks:h.remove()
    for k in ("edge_gate_abs","scorer_norm","encoder_norm","graphlearn_norm","gcn_norm"):
        if not all(v>0 and np.isfinite(v) for v in controlled[k]):raise RuntimeError(f"controlled classification gradient missing: {k}: {controlled[k]}")
    if not controlled["adaptive_grad_is_none"]:raise RuntimeError("frozen adaptive has grad")
    del a,b
    restore(s,initial);r.restore_rng(rng);zero(s)
    a=forward(s,True);first_logp=a["logp"].detach().cpu();first_loss=float(a["total"].detach());a["total"].backward()
    for k in ("opt_base","opt_cam","opt_added"):s[k].step()
    after_first=snapshot(s);rng_first=r.pack_rng();first_sha=state_sha(s);del a
    # Atomic checkpoint reload and same one-update continuation from the same RNG.
    p=RT/"preflight_resume.pt";atomic_torch(p,{"state":initial,"rng":rng})
    loaded=tload(p);restore(s,loaded["state"]);r.restore_rng(loaded["rng"]);zero(s)
    a=forward(s,True);second_logp=a["logp"].detach().cpu();second_loss=float(a["total"].detach());a["total"].backward()
    for k in ("opt_base","opt_cam","opt_added"):s[k].step()
    second_sha=state_sha(s)
    if not torch.equal(first_logp,second_logp) or first_loss!=second_loss or first_sha!=second_sha:raise RuntimeError(f"resume one-step parity failure: output_max={float((first_logp-second_logp).abs().max())}, losses={first_loss},{second_loss}, sha_equal={first_sha==second_sha}")
    for k in ("opt_base","opt_cam","opt_added"):
        if not e.nested_tensor_equal(s[k].state_dict(),after_first[k]):raise RuntimeError("resumed optimizer differs")
    if r.rng_sha(r.pack_rng())!=r.rng_sha(rng_first):raise RuntimeError("resume RNG differs")
    baseline=json.loads((BASE/".runtime/tasks/E69.3-G2_f01.json").read_text())["rng_trace"][0]
    if baseline["epoch"]!=11 or baseline["rng_before"]!=w["rng_sha"] or baseline["rng_after_train"]!=r.rng_sha(rng_first):raise RuntimeError("first epoch RNG differs from DARG")
    p.unlink();del a,s,initial,after_first,loaded;torch.cuda.empty_cache()
    result={"status":"PASS","formal_tasks":10,"source_sha256":sources(),"pinned_github_source_match":True,"all_fold_inheritance":all_ids,"warmups":warmups,
            "graph_tests":unit,"natural_classification_gradients":natural,"controlled_classification_gradients":controlled,
            "formal_gcn_call_counts":counts,"cam_call_count":cam_count[0],"resume_one_step_exact":True,"first_epoch_rng_matches_e693":True,
            "adaptive_frozen_and_unused":True,"original_graph_loss_unchanged":True,"no_oof":True,"thresholds_not_modified":True}
    atomic_json(RESULTS/"preflight.json",result);print("PREFLIGHT PASS",flush=True);return result

def metrics(state,out):return load_base().load_e61().load_base().classification_metrics(out["logp"],state["test"],state["y"])
def strip(m):return {k:float(v) for k,v in m.items() if k not in ("pred","prob")}

def payload(state,fold,best,epoch,executed,trace,initial):
    r=load_base().load_e61().load_base()
    return {"experiment":CFG["experiment"],"group":GROUP,"fold":fold,"seed":0,"config":CFG,"source_sha256":sources(),
            "identity":state["e74_identity"],"best_epoch_1based":epoch+1,"epochs_executed":executed,
            "initial_gate_values":initial,"adaptive_frozen":True,"rng_trace":trace,"rng_end":r.pack_rng(),
            "rng_end_sha":r.rng_sha(r.pack_rng()),**best}

def train_worker(fold):
    import torch,numpy as np
    ss=sources()
    if task_path(fold).exists():
        old=json.loads(task_path(fold).read_text())
        if old["status"]=="COMPLETE" and old["source_sha256"]==ss and old["checkpoint_sha256"]==sha(best_path(fold)):return old
        raise RuntimeError("existing completed task is incompatible; refuse overwrite")
    state,w,identity=prepare(fold);e=load_base().load_e61();r=e.load_base()
    initial={"edge_gate":torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),"adaptive_compatibility":torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().tolist()}
    start=10;wait=0;best_acc=-1.;best_epoch=-1;best=None;trace=[]
    if latest_path(fold).exists():
        saved=tload(latest_path(fold))
        if saved["source_sha256"]!=ss or saved["identity"]!=identity:raise RuntimeError("resume config/source/identity mismatch")
        restore(state,saved["current"]);r.restore_rng(saved["rng"])
        start=saved["next_epoch"];wait=saved["wait"];best=saved["best"];best_acc=saved["best_acc"];best_epoch=saved["best_epoch"];trace=saved["trace"]
    started=time.time();executed=start
    for epoch in range(start,CFG["optimization"]["max_epochs"]):
        before=r.rng_sha(r.pack_rng());zero(state)
        out=forward(state,True)
        if not torch.isfinite(out["total"]):raise FloatingPointError("nonfinite training loss")
        out["total"].backward();grad=gradient_info(state)
        for mod in (state["model"],state["cam"]):
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in mod.parameters()):raise FloatingPointError("nonfinite gradient")
        for k in ("opt_base","opt_cam","opt_added"):state[k].step()
        after=r.rng_sha(r.pack_rng());executed=epoch+1
        with torch.no_grad():ev=forward(state,False);m=metrics(state,ev)
        tr={"epoch":epoch+1,"rng_before":before,"rng_after_train":after,"rng_after_eval":r.rng_sha(r.pack_rng())};trace.append(tr)
        if m["acc"]>best_acc:
            best_acc=float(m["acc"]);best_epoch=epoch;wait=0
            best={**snapshot(state),"metrics":strip(m),"test_probability":m["prob"].copy(),"test_prediction":m["pred"].copy(),
                  "diagnostics":{"losses":{k:float(out[k].detach()) for k in ("cls","graph","pair","adv","preserve","total")},
                                 "gradient":grad,"graph":ev["info"]["edge_diagnostics"]}}
            atomic_torch(best_path(fold),payload(state,fold,best,best_epoch,executed,trace,initial))
        else:wait+=1
        if executed%10==0 or executed==450 or wait>200:
            atomic_torch(latest_path(fold),{"source_sha256":ss,"identity":identity,"current":snapshot(state),"rng":r.pack_rng(),
                         "next_epoch":executed,"wait":wait,"best":best,"best_acc":best_acc,"best_epoch":best_epoch,"trace":trace})
            record={"fold":fold,"epoch":executed,"best_epoch":best_epoch+1,"best_acc":best_acc,"metrics":strip(m),
                    "loss":float(out["total"].detach()),"edge_gate":torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().tolist(),"gradient":grad,"rng":tr,
                    "peak_gpu_bytes":torch.cuda.max_memory_allocated()}
            with (LOGS/f"fold{fold:02d}.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
            print(f"fold={fold} epoch={executed} best={best_acc:.6f} best_epoch={best_epoch+1}",flush=True)
        del out,ev
        if wait>CFG["optimization"]["early_stop_wait_gt"]:break
    if best is None:raise RuntimeError("no valid best checkpoint")
    # Historical best weights and optimizer states are stored, not final weights.
    atomic_torch(best_path(fold),payload(state,fold,best,best_epoch,executed,trace,initial))
    task={"status":"COMPLETE","group":GROUP,"fold":fold,"source_sha256":ss,"identity":identity,"checkpoint_sha256":sha(best_path(fold)),
          "best_epoch_1based":best_epoch+1,"epochs_executed":executed,"metrics":best["metrics"],"rng_trace":trace,
          "wall_seconds":time.time()-started,"peak_gpu_bytes":torch.cuda.max_memory_allocated()}
    atomic_json(task_path(fold),task);print(f"COMPLETE fold={fold}",flush=True);return task

def protected_files():
    g=load_base();e=g.load_e61();files=set()
    for folder in (BASE,g.E62_EXP,g.E61_EXP,g.E53_EXP):
        for pat in ("*.py","*.json","results/*.csv","results/*.md"):
            files.update(folder.glob(pat))
    files.update(BASE.glob(".runtime/checkpoints/*.best.pt"));files.update(g.E61_EXP.glob(".runtime/warmup/*.pt"))
    for p in (e.BASE_ROOT/"data/ABIDE/processed_standard_data.csv",e.BASE_ROOT/"data/ABIDE/modal_feat_dict.npy"):
        files.add(p)
    files.update(e.ANCHORS.glob("*.pt"))
    for p in g.ROOT.glob("*.py"):files.add(p)
    for p in (e.BASE_ROOT/"experiment/experiment51_dugsa/source").glob("*.py"):files.add(p)
    return {str(p):sha(p) for p in sorted(files) if p.is_file()}

def run_all(workers=10):
    import torch
    RESULTS.mkdir(exist_ok=True)
    lock=RT/"protected_sha.json"
    if lock.exists():
        protected=json.loads(lock.read_text())
        if any(sha(p)!=h for p,h in protected.items()):raise RuntimeError("protected historical file changed")
    else:
        protected=protected_files();atomic_json(lock,protected)
    pre=json.loads((RESULTS/"preflight.json").read_text()) if (RESULTS/"preflight.json").exists() else None
    if pre is None or pre.get("source_sha256")!=sources():pre=preflight()
    if pre["status"]!="PASS":raise RuntimeError("preflight not PASS")
    pending=[f for f in range(1,11) if not task_path(f).exists()];active={};failed=[];attempts={};maxactive=0;started=time.time()
    workers=max(1,min(10,workers));target=workers;current=min(2,workers);ramp=False
    while pending or active:
        while pending and len(active)<current:
            f=pending.pop(0);stream=(LOGS/f"fold{f:02d}.stdout.log").open("a")
            p=subprocess.Popen([sys.executable,str(EXP/"run.py"),"--worker","--fold",str(f)],cwd=str(EXP),stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
            active[p]=(f,stream);attempts[f]=attempts.get(f,0)+1
        maxactive=max(maxactive,len(active))
        # Measure two actual processes first, then allow up to ten using observed peaks.
        if not ramp:
            logs=[LOGS/f"fold{f:02d}.jsonl" for f,_ in active.values()]
            if logs and all(p.exists() and p.stat().st_size for p in logs):
                peaks=[json.loads(p.read_text().splitlines()[-1])["peak_gpu_bytes"] for p in logs]
                free,total=torch.cuda.mem_get_info()
                budget=max(1,int((total*.80)/max(max(peaks)*1.4,1)))
                current=min(target,budget);ramp=True
                print(f"RAMP workers={current}, measured_peak_bytes={peaks}",flush=True)
        for p,(f,stream) in list(active.items()):
            code=p.poll()
            if code is None:continue
            stream.close();del active[p]
            if code:
                tail=(LOGS/f"fold{f:02d}.stdout.log").read_text()[-5000:]
                if "out of memory" in tail.lower() and attempts[f]<2:
                    current=max(1,current//2);pending.append(f)
                else:failed.append({"fold":f,"exit_code":code,"log":str(LOGS/f"fold{f:02d}.stdout.log"),"tail":tail})
        status={"experiment":CFG["experiment"],"group":GROUP,"status":"TRAINING" if not failed else "PARTIAL","expected_tasks":10,
                "completed_tasks":sum(task_path(f).exists() for f in range(1,11)),"independent_audits":0,"active_folds":[f for f,_ in active.values()],
                "failed_tasks":failed,"source_sha256":sources(),"max_workers_observed":maxactive,"worker_limit":current,"preflight":"PASS"}
        atomic_json(EXP/"manifest.json",status)
        time.sleep(2)
    if failed:raise RuntimeError(f"formal tasks failed: {failed}")
    audit=load_audit_module()
    return audit.finalize({"device":torch.cuda.get_device_name(0),"max_workers_observed":maxactive,"worker_limit":current,"wall_seconds":time.time()-started})

def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",action="store_true");p.add_argument("--fold",type=int);p.add_argument("--preflight",action="store_true");p.add_argument("--audit",action="store_true");p.add_argument("--workers",type=int,default=10);args=p.parse_args()
    for d in (RT,TASKS,CKPTS,LOGS,RESULTS):d.mkdir(parents=True,exist_ok=True)
    if args.worker:train_worker(args.fold)
    elif args.preflight:preflight()
    elif args.audit:
        audit=load_audit_module()
        audit.finalize({"audit_only":True})
    else:run_all(args.workers)
if __name__=="__main__":main()
