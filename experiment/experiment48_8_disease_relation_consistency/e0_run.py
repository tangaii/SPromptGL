#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent; RT=HERE/".runtime"; RES=HERE/"results"
SRC=RT/"sources"; ROOT=Path("/data/adodas/SPromptGL-main")
CFG=json.loads((HERE/"config_e0.json").read_text()); GROUPS={g["id"]:g for g in CFG["groups"]}
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k,"2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
CTX=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def state_sha(obj):
    h=hashlib.sha256()
    def visit(prefix,v):
        if isinstance(v,dict):
            for k in sorted(v): visit(prefix+"/"+str(k),v[k])
        elif hasattr(v,"detach"):
            h.update(prefix.encode()+b"\0");h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,(list,tuple)):
            for i,x in enumerate(v):visit(prefix+f"/{i}",x)
    visit("",obj);return h.hexdigest()

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)

def save(path,obj):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)

def context():
    global CTX
    if CTX is None:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        e51_exp=SRC/"e51";os.environ["DUGSA_EXP"]=str(e51_exp)
        spec=importlib.util.spec_from_file_location("experiment51_runner_frozen",e51_exp/"run.py")
        e51run=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e51run;spec.loader.exec_module(e51run)
        sys.path.insert(0,str(e51_exp))
        sys.path.insert(0,str(e51_exp/"source"))
        import pandas as pd
        import layers,network,utils,dugsa
        ec={"torch":torch,"np":np,"pd":pd,"layers":layers,"network":network,"utils":utils,"dugsa":dugsa,"device":torch.device("cuda")}
        e51run.CTX=ec
        mspec=importlib.util.spec_from_file_location("apcr486_method",HERE/"e0_method.py")
        method=importlib.util.module_from_spec(mspec);sys.modules[mspec.name]=method;mspec.loader.exec_module(method)
        CTX={"torch":torch,"pd":pd,"e51run":e51run,"layers":ec["layers"],"network":ec["network"],"utils":ec["utils"],"device":torch.device("cuda"),"method":method}
    return CTX

def rng_state():
    t=context()["torch"]
    return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":t.get_rng_state(),"cuda":t.cuda.get_rng_state_all()}

def restore_rng(r):
    t=context()["torch"];random.setstate(r["python"]);np.random.set_state(r["numpy"]);t.set_rng_state(r["cpu"].cpu());t.cuda.set_rng_state_all([v.cpu() for v in r["cuda"]])

def fold_hash(tr,te):return hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()+np.asarray(te,dtype=np.int64).tobytes()).hexdigest()

def data():
    c=context();(x,y),folds=c["e51run"].load_data();mapping=np.load(CFG["data"]["mapping"],allow_pickle=True).item()
    assert x.shape==(871,320) and np.bincount(y).tolist()==[403,468]
    assert [len(mapping[k]) for k in ("PHENO","ANAT","FUNC","Correlation")]==[48,6,10,256]
    return x,y,folds,mapping

def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def prepare_anchors():
    c=context();e=c["e51run"];x,y,folds,_=data();out=[]
    anchor_dir=RT/"anchors";anchor_dir.mkdir(parents=True,exist_ok=True)
    for fold,(tr,te) in enumerate(folds,1):
        path=anchor_dir/f"fold{fold:02d}_e51e1_init.pt"
        state=e.build_state("E1",fold)
        assert state["group"]=="E1"
        assert len(state["model"].graphs)==4 and len(state["model"].gcn)==4
        assert state["anchor"]["split_sha"]==e.split_sha(fold)
        base_rng=e.pack_rng()
        bundle={"gfr":cpu_state(state["model"]),"cam":cpu_state(state["cam"])}
        payload={"fold":fold,"split_sha":fold_hash(tr,te),"e51_split_sha":e.split_sha(fold),
                 "gfr":bundle["gfr"],"cam":bundle["cam"],
                 "opt_gfr":copy.deepcopy(state["opt_model"].state_dict()),"opt_cam":copy.deepcopy(state["opt_cam"].state_dict()),
                 "rng":base_rng,"base_bundle_sha":state_sha(bundle),"source_anchor_parameter_sha":state["anchor"]["parameter_sha"],
                 "source_anchor_rng_sha":e.rng_sha(state["anchor"]["rng"]),"source_anchor_sha":sha(e.ANCHORS/f"fold{fold:02d}.pt")}
        save(path,payload);out.append({"fold":fold,"path":str(path),"sha256":sha(path),"base_bundle_sha":payload["base_bundle_sha"],"split_sha":payload["split_sha"],"source_anchor_sha":payload["source_anchor_sha"]})
        del state
    return out

def init():
    x,y,folds,mapping=data();anchors=prepare_anchors()
    protected_names=["main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]
    protected={str(ROOT/n):sha(ROOT/n) for n in protected_names}
    source_files=[p for p in (SRC/"e51").rglob("*") if p.is_file()]+[SRC/"48_5"/"method.py",HERE/"e0_method.py",HERE/"run.py",HERE/"config.json"]
    source_hashes={str(p.relative_to(HERE)):sha(p) for p in sorted(source_files)}
    signature=code_signature()
    e1rows=[json.loads(p.read_text()) for p in sorted((RT/"e51_reference_tasks").glob("E1_f*.json"))]
    assert len(e1rows)==10
    assert all(r.get("status")=="COMPLETE" and r.get("group")=="E1" for r in e1rows)
    atomic_json(HERE/"manifest.json",{"status":"INITIALIZED","requested_path":str(ROOT/"experiment/experiment48_6_1_e51_rng_parity"),"staging_path":str(HERE),"delivery_status":"SCRATCH_STAGED_DATA_EIO","code_signature":signature,"source_sha256":source_hashes,"protected_sha256_before":protected,"data_sha256":sha(CFG["data"]["csv"]),"mapping_sha256":sha(CFG["data"]["mapping"]),"fold_sha256":[fold_hash(*f) for f in folds],"e51_e1_reference_rows":e1rows,"e51_e1_reference_summary":context()["e51run"].csv_read(SRC/"e51"/"results/summary.csv"),"e51_initial_anchors":anchors,"experiment48_5_source_manifest_sha256":sha(RT/"sources/48_5/manifest.json") if (RT/"sources/48_5/manifest.json").exists() else None,"tasks":{"total":20,"complete":0},"worker_plan":CFG["worker_plan"]})

def agent_from_state(state,gid,fold,x,y,tr,te,mapping):
    c=context();t=c["torch"];m=c["method"];anchor=load(RT/"anchors"/f"fold{fold:02d}_e51e1_init.pt")
    current={"gfr":cpu_state(state["model"]),"cam":cpu_state(state["cam"])}
    if state_sha(current)!=anchor["base_bundle_sha"]:raise RuntimeError(f"E51 initial state differs from frozen fold {fold} anchor")
    # APCR uses its 48.5 fold-specific RNG fork and restores the E51 base RNG boundary.
    apcr=m.forked_apcr(t,c["utils"],fold,alpha_init=float(GROUPS[gid]["alpha_init"]),alpha_trainable=bool(GROUPS[gid]["alpha_trainable"]),seed_base=481000)
    a=SimpleNamespace(dev=c["device"],hyperpm=c["e51run"].hp(),feat=state["feat"],targ=state["targ"],
        trn_idx=np.asarray(tr,dtype=np.int64),tst_idx=np.asarray(te,dtype=np.int64),val_idx=np.asarray(te,dtype=np.int64),
        weight=state["weight"],dataLayer=state["parts"],MessagePassing0=state["model"],ModalFusion=state["cam"],
        optimizer_MP0=state["opt_model"],optimizer_MF=state["opt_cam"],APCR=apcr,
        optimizer_APCR=t.optim.Adam(apcr.parameters(),lr=.00038,weight_decay=.0001),fold=fold,group=gid,
        alpha_trainable=bool(GROUPS[gid]["alpha_trainable"]),e51_anchor=anchor)
    a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
    a._torch=t
    a.apcr_rng=copy.deepcopy(apcr._init_rng)
    cols=[i for i,n in enumerate(mapping["PHENO"]) if "SITE_ID_" in str(n)]
    if not cols:raise RuntimeError("PHENO site columns not found")
    a.sites=np.asarray(x[:,cols].argmax(1),dtype=np.int64)
    return a

def initialize_agent(gid,fold,x,y,tr,te,mapping):
    c=context();e=c["e51run"];state=e.build_state("E1",fold)
    anchor=load(RT/"anchors"/f"fold{fold:02d}_e51e1_init.pt")
    state["model"].load_state_dict(anchor["gfr"]);state["cam"].load_state_dict(anchor["cam"])
    state["opt_model"].load_state_dict(anchor["opt_gfr"]);state["opt_cam"].load_state_dict(anchor["opt_cam"])
    restore_rng(anchor["rng"])
    return agent_from_state(state,gid,fold,x,y,tr,te,mapping)

def preflight():
    c=context();e=c["e51run"];m=c["method"];t=c["torch"];x,y,folds,mapping=data();checks=[]
    for fold,(tr,te) in enumerate(folds,1):
        state=e.build_state("E1",fold)
        dummy={"relation_perm":t.arange(len(y),device="cuda"),"utility":t.zeros((4,4),device="cuda"),"align_idx":t.as_tensor(tr[:128],device="cuda"),"train":t.as_tensor(tr,device="cuda")}
        ref=e.forward(state,dummy,train_mode=False)
        a=agent_from_state(state,1,fold,x,y,tr,te,mapping);a.MessagePassing0.eval();a.ModalFusion.eval();a.APCR.eval()
        parts,base,adjs,graph,_=m.base_forward(a,c["utils"]);logp=a.ModalFusion(base)[0]
        assert t.equal(ref["fused"],base),f"fold {fold}: fused mismatch"
        assert all(t.equal(u,v) for u,v in zip(ref["adjs"],adjs)),f"fold {fold}: adjacency mismatch"
        assert t.equal(ref["prob"],logp),f"fold {fold}: CAM logits mismatch"
        assert t.equal(ref["graph"],graph),f"fold {fold}: graph loss mismatch"
        assert t.equal(ref["cls"],c["utils"].ClsLoss(logp,a.targ,a.trn_idx,a.weight)),f"fold {fold}: classification loss mismatch"
        assert len(state["model"].last["msgs"])==4 and all(v.shape==(871,320) for v in state["model"].last["msgs"])
        checks.append({"fold":fold,"fused_exact":True,"adjs_exact":True,"logits_exact":True,"graph_loss_exact":True,"cls_loss_exact":True,"legacy_four_messages":True})
        del state,a
    # Paired full group=6 training step: the only difference is alpha_raw trainability.
    fold=1;tr,te=folds[0]
    # Use a matched nonzero Fixed/Learnable pair for shared-gradient equivalence.
    # Groups 1/2 are intentionally different (exact zero vs 1e-8 near-zero),
    # so their full losses and gradients are not expected to be identical.
    a=initialize_agent(3,fold,x,y,tr,te,mapping);b=initialize_agent(4,fold,x,y,tr,te,mapping)
    assert state_sha(m.model_bundle(a))==state_sha(m.model_bundle(b))
    for v in (a,b):v.capture_shared_gradients=True
    main_rng=copy.deepcopy(rng_state());a_rng=copy.deepcopy(a.apcr_rng);restore_rng(main_rng);a.apcr_rng=copy.deepcopy(a_rng);sa=m.train_step(a,6,90,c["utils"],a.sites,y)
    restore_rng(main_rng);b.apcr_rng=copy.deepcopy(a_rng);sb=m.train_step(b,6,90,c["utils"],b.sites,y)
    for k in ("main","graph","srd","sca","robust"):assert sa[k]==sb[k],k
    assert a.APCR.alpha_raw.grad is None
    assert b.APCR.alpha_raw.grad is not None and t.isfinite(b.APCR.alpha_raw.grad).all() and float(b.APCR.alpha_raw.grad.abs())>0
    assert float(b.APCR.alpha())>0
    assert set(a.shared_grads_before_clip)==set(b.shared_grads_before_clip)
    assert all(t.equal(a.shared_grads_before_clip[k],b.shared_grads_before_clip[k]) for k in a.shared_grads_before_clip)
    assert sa["cra"] and sa["cra"]["step1"] and sa["pair_stats"]["fit_pairs"]>0
    report={"status":"PASS","e51_legacy_parity_folds":checks,"folds_passed":len(checks),"same_initial_bundle_sha":True,"same_full_group6_losses":True,"shared_gradients_equal":True,"fixed_alpha_no_grad_and_constant":True,"learnable_alpha_grad_abs":float(b.APCR.alpha_raw.grad.abs()),"learnable_alpha_after_step":float(b.APCR.alpha()),"srd_sca_cra_active":True,"training_group":6}
    atomic_json(RT/"preflight.json",report);return report

def task_paths(gid,fold):
    tag=f"E{gid}_f{fold:02d}"
    return RT/"tasks"/(tag+".json"),RT/"checkpoints"/(tag+".best.pt"),RT/"checkpoints"/(tag+".latest.pt")

def run_task(gid,fold,worker,x,y,folds,mapping):
    c=context();t=c["torch"];m=c["method"];tr,te=folds[fold-1];task,bestpath,latest=task_paths(gid,fold)
    if task.exists() and json.loads(task.read_text()).get("status")=="COMPLETE":return
    a=initialize_agent(gid,fold,x,y,tr,te,mapping);init_bundle=m.model_bundle(a)
    init_sha=state_sha(init_bundle);gfr_sha=state_sha(init_bundle["gfr"]);cam_sha=state_sha(init_bundle["pln"]);apcr_sha=state_sha(init_bundle["apcr"])
    if latest.exists():
        ck=load(latest)
        if ck["signature"]!=code_signature():raise RuntimeError("resume source signature differs")
        m.load_bundle(a,ck["model"]);a.optimizer_MP0.load_state_dict(ck["opt_mp"]);a.optimizer_MF.load_state_dict(ck["opt_cam"]);a.optimizer_APCR.load_state_dict(ck["opt_apcr"]);restore_rng(ck["rng"]);a.apcr_rng=ck["apcr_rng"]
        begin=ck["epoch"]+1;best=ck["best"];bestacc=ck["bestacc"];be=ck["best_epoch"];wait=ck["wait"];trace=ck["trace"];alpha_history=ck["alpha_history"]
    else:
        begin=0;best=None;bestacc=-1.;be=-1;wait=0;trace=[];alpha_history=[]
    atomic_json(task,{"status":"RUNNING","group_id":gid,"group":GROUPS[gid]["name"],"fold":fold,"worker":worker,"gpu":0,"seed":0,"signature":code_signature(),"training_group":6,"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}_e51e1_init.pt"),"initial_bundle_sha":init_sha,"gfr_init_sha":gfr_sha,"cam_init_sha":cam_sha,"apcr_init_sha":apcr_sha,"alpha_init":float(a.APCR.alpha()),"alpha_trainable":a.alpha_trainable})
    started=time.monotonic()
    alpha_expected=float(GROUPS[gid]["alpha_init"])
    for epoch in range(begin,450):
        stat=m.train_step(a,6,epoch,c["utils"],a.sites,y)
        alpha_now=float(a.APCR.alpha().detach())
        if not a.alpha_trainable:
            assert abs(alpha_now-alpha_expected)<=max(1e-10,abs(alpha_expected)*1e-6),(gid,fold,epoch,alpha_now,alpha_expected)
        trace.append({"epoch":epoch,**stat});alpha_history.append(alpha_now)
        with t.no_grad():out=m.evaluate(a,6,c["utils"])
        val_acc=float(out["logp"][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).float().mean())
        if val_acc>bestacc:bestacc=val_acc;best=m.model_bundle(a);be=epoch;wait=0
        else:wait+=1
        if (epoch+1)%20==0 or wait>200:
            save(latest,{"signature":code_signature(),"model":m.model_bundle(a),"opt_mp":a.optimizer_MP0.state_dict(),"opt_cam":a.optimizer_MF.state_dict(),"opt_apcr":a.optimizer_APCR.state_dict(),"rng":rng_state(),"apcr_rng":a.apcr_rng,"epoch":epoch,"best":best,"bestacc":bestacc,"best_epoch":be,"wait":wait,"trace":trace,"alpha_history":alpha_history})
        if wait>200:break
        if epoch%50==0:print(f"PROGRESS E{gid} fold={fold} epoch={epoch} alpha={alpha_now:.7f} loss={stat['loss']:.5f}",flush=True)
    final_alpha=float(a.APCR.alpha().detach());m.load_bundle(a,best)
    if not a.alpha_trainable:assert abs(float(a.APCR.alpha().detach())-alpha_expected)<=max(1e-10,abs(alpha_expected)*1e-6)
    with t.no_grad():out=m.evaluate(a,6,c["utils"])
    row=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy();alpha_grad=[float(v["alpha_grad_abs"]) for v in trace if v.get("alpha_grad_abs") is not None]
    diag={"alpha_init":alpha_expected,"alpha_best":float(out["alpha"]),"alpha_final_training":final_alpha,"alpha_min_history":float(np.min(alpha_history)),"alpha_max_history":float(np.max(alpha_history)),"alpha_delta_from_init":float(out["alpha"])-alpha_expected,"alpha_trainable":a.alpha_trainable,"alpha_grad_nonzero_steps":sum(v>0 for v in alpha_grad),"alpha_grad_steps":len(alpha_grad),"alpha_grad_abs_mean":float(np.mean(alpha_grad)) if alpha_grad else 0.,"alpha_touched_max":bool(float(out["alpha"])>=.2-1e-6),"alpha_trace":alpha_history,"message_norms":[float(v.norm().detach()) for v in out["deltas"]],"gate_means":[float(v.mean().detach()) for v in out["gates"]],"relative_residual_norm":float(out["residual"].norm().detach()/(out["base"].norm().detach()+1e-8)),"last_epoch_trace":trace[-1],"srd_sca_cra_active":True}
    save(bestpath,{"model":best,"prediction":pred,"metrics":row,"diagnostics":diag,"best_epoch":be,"epochs_executed":epoch+1,"initial_bundle_sha":init_sha,"signature":code_signature(),"training_group":6,"fold":fold,"group_id":gid})
    latest.unlink(missing_ok=True)
    result={"status":"COMPLETE","group_id":gid,"group":GROUPS[gid]["name"],"fold":fold,"seed":0,"worker":worker,"gpu":0,"signature":code_signature(),"training_group":6,"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}_e51e1_init.pt"),"initial_bundle_sha":init_sha,"gfr_init_sha":gfr_sha,"cam_init_sha":cam_sha,"apcr_init_sha":apcr_sha,"best_sha":sha(bestpath),"prediction_sha256":hashlib.sha256(pred.tobytes()).hexdigest(),"best_epoch":be,"epochs_executed":epoch+1,"wall_seconds":time.monotonic()-started,"alpha_trainable":a.alpha_trainable,"diagnostics":diag,**row}
    atomic_json(task,result);print("DONE",gid,fold,row["acc"],be,diag["alpha_best"],flush=True)

def code_signature():
    files=[HERE/"run.py",HERE/"e0_method.py",HERE/"config.json"]+[p for p in (SRC/"e51").rglob("*.py")]+[SRC/"e51"/"config.json",SRC/"48_5"/"method.py"]
    return hashlib.sha256(json.dumps({str(p.relative_to(HERE)):sha(p) for p in sorted(files)},sort_keys=True).encode()).hexdigest()

def worker(worker_id,folds_arg):
    x,y,folds,mapping=data();ids=[int(v) for v in folds_arg.split(",")]
    for fold in ids:
        for gid in sorted(GROUPS):run_task(gid,fold,worker_id,x,y,folds,mapping)

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True);sp.add_parser("init");sp.add_parser("preflight")
    w=sp.add_parser("worker");w.add_argument("--worker",type=int,required=True);w.add_argument("--folds",required=True)
    a=ap.parse_args()
    if a.cmd=="init":init()
    elif a.cmd=="preflight":print(json.dumps(preflight(),indent=2))
    else:worker(a.worker,a.folds)

if __name__=="__main__":main()
