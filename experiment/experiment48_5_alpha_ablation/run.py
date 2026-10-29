#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RT=HERE/".runtime";RES=HERE/"results";SRC=RT/"sources"
ROOT=Path("/data/adodas/SPromptGL-main")
PARENT=Path("/model/spromptgl/experiment/experiment48_4_verified_apcr")
CFG=json.loads((HERE/"config.json").read_text())
GROUPS={int(g["id"]):g for g in CFG["groups"]};CTX=None
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key,"2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def _finite(v):
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,np.generic):return _finite(v.item())
    if isinstance(v,dict):return {k:_finite(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [_finite(x) for x in v]
    return v
def atomic_json(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(_finite(obj),ensure_ascii=False,indent=2,allow_nan=False)+"\n");os.replace(tmp,path)
def save(path,obj):
    import torch
    path=Path(path);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)
def context():
    global CTX
    if CTX is None:
        import pandas as pd,torch,importlib.util
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        sys.path.insert(0,str(SRC));import layers,network,utils
        spec=importlib.util.spec_from_file_location("apcr485_method",HERE/"method.py")
        method=importlib.util.module_from_spec(spec);sys.modules[spec.name]=method;spec.loader.exec_module(method)
        CTX={"pd":pd,"torch":torch,"layers":layers,"network":network,"utils":utils,"method":method,"device":torch.device("cuda")}
    return CTX
def rng_state():
    t=context()["torch"]
    return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":t.get_rng_state(),"cuda":t.cuda.get_rng_state_all()}
def restore_rng(s):
    t=context()["torch"];random.setstate(s["python"]);np.random.set_state(s["numpy"]);t.set_rng_state(s["cpu"].cpu());t.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
def state_sha(bundle):
    h=hashlib.sha256()
    def visit(prefix,obj):
        if isinstance(obj,dict):
            for k in sorted(obj):visit(prefix+"/"+str(k),obj[k])
        elif hasattr(obj,"detach"):
            v=obj.detach().cpu().contiguous().numpy();h.update(prefix.encode()+b"\0");h.update(v.tobytes())
    visit("",bundle);return h.hexdigest()
def signature():
    files=[HERE/"run.py",HERE/"method.py",HERE/"tests.py",HERE/"config.json"]+sorted(SRC.glob("*.py"))
    return hashlib.sha256(json.dumps({str(p.relative_to(HERE)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
def data():
    c=context();from sklearn.model_selection import StratifiedKFold
    frame=c["pd"].read_csv(CFG["data"]["csv"]);mapping=np.load(CFG["data"]["mapping"],allow_pickle=True).item()
    x=frame.iloc[:,:-1].to_numpy(dtype=np.float32);y=frame.iloc[:,-1].to_numpy(dtype=np.int64)-1
    assert list(mapping)==["PHENO","ANAT","FUNC","Correlation"] and [len(mapping[k]) for k in mapping]==[48,6,10,256]
    assert list(frame.columns[:-1])==[str(v) for k in mapping for v in mapping[k]]
    assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y))
    return x,y,folds,mapping
def fold_hash(tr,te):return hashlib.sha256(np.asarray(tr,dtype=np.int64).tobytes()+np.asarray(te,dtype=np.int64).tobytes()).hexdigest()
def site_ids(x,mapping):
    cols=[i for i,n in enumerate(mapping["PHENO"]) if "SITE_ID_" in str(n)]
    if not cols:raise RuntimeError("PHENO site one-hot columns missing")
    return x[:,cols].argmax(1).astype(np.int64)
def shuffled_labels(y,tr,fold):
    out=np.asarray(y,dtype=np.int64).copy();ids=np.asarray(tr,dtype=np.int64)
    rng=np.random.default_rng(48049000+fold);perm=rng.permutation(ids);out[ids]=out[perm]
    return out,hashlib.sha256(out[ids].tobytes()).hexdigest(),float(np.mean(out[ids]!=np.asarray(y)[ids]))
def initialize_agent(gid,fold,x,y,tr,te,anchor):
    c=context();m=c["method"]
    a=m.make_agent(c["torch"],c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,gid,anchor,485000+fold)
    a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
    a.sites=site_ids(x,np.load(CFG["data"]["mapping"],allow_pickle=True).item())
    return a

def preflight():
    c=context();m=c["method"];t=c["torch"];x,y,folds,mapping=data()
    import importlib.util
    spec=importlib.util.spec_from_file_location("apcr485_unit_tests",HERE/"tests.py")
    unit_tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(unit_tests)
    inherited_tests=unit_tests.all_tests(sys.modules[__name__])
    prior=json.loads((RT/"48_4_engineering_preflight.json").read_text())
    assert prior["parity"]["status"]=="PASS" and len(prior["parity"]["folds"])==10
    checks=[]
    for fold,(tr,te) in enumerate(folds,1):
        anchor=load(RT/"anchors"/f"fold{fold:02d}.pt")
        fixed=initialize_agent(1,fold,x,y,tr,te,anchor);learn=initialize_agent(2,fold,x,y,tr,te,anchor)
        assert not fixed.APCR.alpha_raw.requires_grad and learn.APCR.alpha_raw.requires_grad
        assert abs(float(fixed.APCR.alpha().detach())-.0022)<1e-7 and abs(float(learn.APCR.alpha().detach())-.0022)<1e-7
        assert state_sha(m.model_bundle(fixed))==state_sha(m.model_bundle(learn))
        for mod_a,mod_b in ((fixed.MessagePassing0,learn.MessagePassing0),(fixed.ModalFusion,learn.ModalFusion)):
            assert state_sha(mod_a.state_dict())==state_sha(mod_b.state_dict())
        fixed.MessagePassing0.eval();fixed.ModalFusion.eval();fixed.APCR.eval();learn.MessagePassing0.eval();learn.ModalFusion.eval();learn.APCR.eval()
        with t.no_grad():
            of=m.evaluate(fixed,6,c["utils"]);ol=m.evaluate(learn,6,c["utils"])
        assert t.equal(of["logp"],ol["logp"])
        assert t.equal(of["base"],ol["base"])
        trt=t.as_tensor(tr,device=c["device"])
        clf=c["utils"].ClsLoss(of["logp"],fixed.targ,fixed.trn_idx,fixed.weight)
        cll=c["utils"].ClsLoss(ol["logp"],learn.targ,learn.trn_idx,learn.weight)
        assert t.equal(clf,cll)
        checks.append({"fold":fold,"same_initial_state":True,"same_logits_and_main_loss":True,"alpha_init_fixed":float(of["alpha"]),"alpha_init_learnable":float(ol["alpha"])})
    # Exercise the full inherited group=6 update with identical fold/epoch randomness.
    fold=1;tr,te=folds[0];anchor=load(RT/"anchors/fold01.pt")
    a=initialize_agent(1,fold,x,y,tr,te,anchor);b=initialize_agent(2,fold,x,y,tr,te,anchor)
    for agent in (a,b):agent.capture_shared_gradients=True
    seed=481400000+fold*10000+90;t.manual_seed(seed);t.cuda.manual_seed_all(seed)
    sa=m.train_step(a,6,90,c["utils"],a.sites,y)
    seed=481400000+fold*10000+90;t.manual_seed(seed);t.cuda.manual_seed_all(seed)
    sb=m.train_step(b,6,90,c["utils"],b.sites,y)
    assert sa["main"]==sb["main"] and sa["graph"]==sb["graph"] and sa["srd"]==sb["srd"] and sa["sca"]==sb["sca"] and sa["robust"]==sb["robust"]
    assert a.APCR.alpha_raw.grad is None and abs(float(a.APCR.alpha().detach())-.0022)<1e-7
    assert b.APCR.alpha_raw.grad is not None and t.isfinite(b.APCR.alpha_raw.grad).all() and float(b.APCR.alpha_raw.grad.abs())>0
    assert abs(float(b.APCR.alpha().detach())-.0022)>0
    assert set(a.shared_grads_before_clip)==set(b.shared_grads_before_clip)
    assert all(t.equal(a.shared_grads_before_clip[k],b.shared_grads_before_clip[k]) for k in a.shared_grads_before_clip)
    assert sa["cra"] and sa["cra"]["step1"] and sa["pair_stats"]["fit_pairs"]>0 and sa["sca_weight"]>=0
    report={"status":"PASS","inherited_48_4_unit_tests":inherited_tests,"inherited_48_4_parity":"10/10_PASS","same_initial_state_and_forward":"10/10_PASS","same_shared_preclip_gradients_fold1":True,"group6_full_loss_paths_equal_fold1":True,"fixed_alpha_gradient":None,"learnable_alpha_gradient_abs":float(b.APCR.alpha_raw.grad.abs().detach()),"learnable_alpha_after_one_step":float(b.APCR.alpha().detach()),"cra_active":True,"srd_active":True,"sca_active":True,"folds":checks}
    atomic_json(RT/"preflight.json",report);return report

def run_task(gid,fold,worker,x,y,folds,mapping):
    c=context();m=c["method"];tr,te=folds[fold-1];name=GROUPS[gid]["name"]
    task=RT/"tasks"/f"E{gid}_f{fold:02d}.json";bestpath=RT/"checkpoints"/f"E{gid}_f{fold:02d}.best.pt";latest=RT/"checkpoints"/f"E{gid}_f{fold:02d}.latest.pt"
    if task.exists() and json.loads(task.read_text()).get("status")=="COMPLETE":return
    anchor=load(RT/"anchors"/f"fold{fold:02d}.pt");a=initialize_agent(gid,fold,x,y,tr,te,anchor)
    assert abs(float(a.APCR.alpha().detach())-.0022)<1e-7
    init_bundle=m.model_bundle(a);common_init_sha=state_sha(init_bundle);gfr_init_sha=state_sha(init_bundle["gfr"]);pln_init_sha=state_sha(init_bundle["pln"]);apcr_init_sha=state_sha(init_bundle["apcr"]);alpha_trainable=a.APCR.alpha_raw.requires_grad
    if latest.exists():
        ck=load(latest)
        if ck["signature"]!=signature():raise RuntimeError("resume source signature mismatch")
        m.load_bundle(a,ck["model"]);a.optimizer_MP0.load_state_dict(ck["opt_mp"]);a.optimizer_MF.load_state_dict(ck["opt_mf"]);a.optimizer_APCR.load_state_dict(ck["opt_apcr"]);restore_rng(ck["rng"])
        begin=ck["epoch"]+1;best=ck["best"];bestacc=ck["bestacc"];be=ck["best_epoch"];wait=ck["wait"];trace=ck["trace"];alpha_history=ck["alpha_history"]
    else:
        restore_rng(anchor["rng"]);begin=0;best=None;bestacc=-1.;be=-1;wait=0;trace=[];alpha_history=[]
    atomic_json(task,{"status":"RUNNING","group_id":gid,"group":name,"fold":fold,"worker":worker,"gpu":0,"seed":0,"signature":signature(),"training_group":6,"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}.pt"),"anchor_parameter_sha":anchor["parameter_sha"],"initial_model_bundle_sha":common_init_sha,"initial_gfr_sha":gfr_init_sha,"initial_cam_sha":pln_init_sha,"initial_apcr_sha":apcr_init_sha,"alpha_trainable":alpha_trainable,"alpha_init":float(a.APCR.alpha().detach())})
    started=time.monotonic();train_group=6
    for epoch in range(begin,450):
        torch=c["torch"];seed=481400000+fold*10000+epoch;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        stat=m.train_step(a,train_group,epoch,c["utils"],a.sites,y)
        alpha_now=float(a.APCR.alpha().detach())
        if not alpha_trainable:assert abs(alpha_now-.0022)<1e-7,(gid,fold,epoch,alpha_now)
        if alpha_trainable:
            assert stat["alpha_grad_abs"] is not None and np.isfinite(stat["alpha_grad_abs"])
        trace.append({"epoch":epoch,**stat});alpha_history.append(alpha_now)
        with torch.no_grad():out=m.evaluate(a,train_group,c["utils"])
        val_acc=float(out["logp"][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).float().mean())
        if val_acc>bestacc:bestacc=val_acc;best=m.model_bundle(a);be=epoch;wait=0
        else:wait+=1
        if (epoch+1)%20==0 or wait>200:
            save(latest,{"signature":signature(),"model":m.model_bundle(a),"opt_mp":a.optimizer_MP0.state_dict(),"opt_mf":a.optimizer_MF.state_dict(),"opt_apcr":a.optimizer_APCR.state_dict(),"rng":rng_state(),"epoch":epoch,"best":best,"bestacc":bestacc,"best_epoch":be,"wait":wait,"trace":trace,"alpha_history":alpha_history})
        if wait>200:break
    final_alpha=float(a.APCR.alpha().detach());m.load_bundle(a,best)
    if not alpha_trainable:assert abs(float(a.APCR.alpha().detach())-.0022)<1e-7
    with c["torch"].no_grad():out=m.evaluate(a,train_group,c["utils"])
    row=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy()
    grad_values=[float(v["alpha_grad_abs"]) for v in trace if v.get("alpha_grad_abs") is not None]
    diag={"best_epoch":be,"alpha_trainable":alpha_trainable,"alpha_init":.0022,"alpha_best":float(out["alpha"]),"alpha_final_training":final_alpha,"alpha_min_history":float(np.min(alpha_history)),"alpha_max_history":float(np.max(alpha_history)),"alpha_delta_from_init":float(out["alpha"])-.0022,"alpha_grad_nonzero_steps":sum(v>0 for v in grad_values),"alpha_grad_steps":len(grad_values),"alpha_grad_abs_mean":float(np.mean(grad_values)) if grad_values else 0.,"alpha_touched_max":bool(float(out["alpha"])>=.2-1e-6),"alpha_trace":alpha_history,"message_norms":[float(v.norm().detach()) for v in out["deltas"]],"gate_means":[float(v.mean().detach()) for v in out["gates"]],"relative_residual_norm":float(out["residual"].norm().detach()/(out["base"].norm().detach()+1e-8)),"last_epoch_trace":trace[-1]}
    bundle=m.model_bundle(a);best_sha=state_sha(bundle)
    save(bestpath,{"model":best,"prediction":pred,"metrics":row,"diagnostics":diag,"best_epoch":be,"epochs_executed":epoch+1,"initial_model_bundle_sha":common_init_sha,"signature":signature(),"alpha_trainable":alpha_trainable,"training_group":6})
    latest.unlink(missing_ok=True)
    result={"status":"COMPLETE","group_id":gid,"group":name,"fold":fold,"seed":0,"worker":worker,"gpu":0,"signature":signature(),"training_group":6,"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}.pt"),"anchor_parameter_sha":anchor["parameter_sha"],"initial_model_bundle_sha":common_init_sha,"initial_gfr_sha":gfr_init_sha,"initial_cam_sha":pln_init_sha,"initial_apcr_sha":apcr_init_sha,"best_sha":sha(bestpath),"best_model_bundle_sha":best_sha,"prediction_sha256":hashlib.sha256(pred.tobytes()).hexdigest(),"best_epoch":be,"epochs_executed":epoch+1,"wall_seconds":time.monotonic()-started,"alpha_trainable":alpha_trainable,"diagnostics":diag,**row}
    atomic_json(task,result);print("DONE",gid,fold,row["acc"],be,diag["alpha_best"],flush=True);return result

def init():
    x,y,folds,mapping=data()
    protected={str(p):sha(p) for p in [ROOT/n for n in ["main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]]}
    src_hash={p.name:sha(p) for p in SRC.glob("*.py")}
    prior=json.loads((RT/"48_4_manifest.json").read_text())
    atomic_json(HERE/"manifest.json",{"status":"INITIALIZED","experiment_root_requested":str(ROOT/"experiment/experiment48_5_alpha_ablation"),"staging_path":str(HERE),"parent_experiment_path":str(PARENT),"parent_48_4_code_signature":prior["code_signature"],"code_signature":signature(),"data_sha":sha(CFG["data"]["csv"]),"protected_sha256":protected,"fold_sha256":[fold_hash(*f) for f in folds],"source_sha256":src_hash,"alpha_init":.0022,"training_group":6,"worker_plan":CFG["worker_plan"],"historical_reference_sha256":{str(p.relative_to(RT)):sha(p) for p in RT.glob("48_*csv")} })

def worker(worker_id,folds_arg):
    x,y,folds,mapping=data();folds_arg=[int(v) for v in folds_arg.split(",")]
    for fold in folds_arg:
        for gid in (1,2):run_task(gid,fold,worker_id,x,y,folds,mapping)

if __name__=="__main__":
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("init");sub.add_parser("preflight")
    w=sub.add_parser("worker");w.add_argument("--worker",type=int,required=True);w.add_argument("--folds",required=True)
    args=ap.parse_args()
    if args.cmd=="init":init()
    elif args.cmd=="preflight":print(json.dumps(preflight(),indent=2))
    else:worker(args.worker,args.folds)
