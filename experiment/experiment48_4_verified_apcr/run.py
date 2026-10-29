#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RT=HERE/".runtime";RES=HERE/"results";SRC=RT/"sources"
DATA_ROOT=Path("/data/adodas/SPromptGL-main");HIST_EXP=DATA_ROOT/"experiment/experiment48_3_apcr_gfr"
CFG=json.loads((HERE/"config.json").read_text());NAMES={g["id"]:g["name"] for g in CFG["groups"]};CTX=None
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ.setdefault(key,"2")
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
        import pandas as pd,torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        sys.path.insert(0,str(SRC));import layers,network,utils
        import importlib.util
        spec=importlib.util.spec_from_file_location("verified_apcr_method",HERE/"method.py")
        method=importlib.util.module_from_spec(spec);sys.modules[spec.name]=method;spec.loader.exec_module(method)
        CTX={"pd":pd,"torch":torch,"layers":layers,"network":network,"utils":utils,"method":method,"device":torch.device("cuda")}
    return CTX
def rng_state():
    c=context();return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":c["torch"].get_rng_state(),"cuda":c["torch"].cuda.get_rng_state_all()}
def restore_rng(s):
    c=context();random.setstate(s["python"]);np.random.set_state(s["numpy"]);c["torch"].set_rng_state(s["cpu"].cpu());c["torch"].cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
def seed_all(seed):
    c=context();random.seed(seed);np.random.seed(seed);c["torch"].manual_seed(seed);c["torch"].cuda.manual_seed_all(seed)
    c["torch"].backends.cudnn.deterministic=True;c["torch"].backends.cudnn.benchmark=False
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
def signature():
    files=[HERE/"run.py",HERE/"method.py",HERE/"tests.py",HERE/"config.json"]+sorted(SRC.glob("*.py"))
    return hashlib.sha256(json.dumps({str(p.relative_to(HERE)):sha(p) for p in files},sort_keys=True).encode()).hexdigest()
def site_ids(x,mapping):
    cols=[i for i,n in enumerate(mapping["PHENO"]) if "SITE_ID_" in str(n)]
    if not cols:raise RuntimeError("site one-hot columns missing")
    return x[:,cols].argmax(1).astype(np.int64)
def wrong_relation_permutation(x,train,mapping):
    site=site_ids(x,mapping);trset=set(map(int,train));groups={}
    for i in range(len(x)):groups.setdefault((int(site[i]),int(i in trset)),[]).append(i)
    p=np.arange(len(x),dtype=np.int64)
    for ids in groups.values():
        if len(ids)>1:p[ids]=np.roll(np.asarray(ids),1)
    assert len(np.unique(p))==len(p)
    return context()["torch"].as_tensor(p,device=context()["device"],dtype=context()["torch"].long)
def shuffled_labels(y,tr,fold):
    out=np.asarray(y,dtype=np.int64).copy();ids=np.asarray(tr,dtype=np.int64);rng=np.random.default_rng(48049000+fold);perm=rng.permutation(ids);out[ids]=out[perm]
    return out,hashlib.sha256(out[ids].tobytes()).hexdigest(),float(np.mean(out[ids]!=np.asarray(y)[ids]))
def initialize_agent(gid,fold,x,y,tr,te,anchor):
    c=context();m=c["method"]
    a=m.make_agent(c["torch"],c["layers"],c["network"],c["utils"],c["device"],x,y,tr,te,fold,gid,anchor,481000+fold)
    a.srd_fit_ids,a.srd_cal_ids=m.partition_srd_patients(tr,y,481100+fold)
    a.sites=site_ids(x,np.load(CFG["data"]["mapping"],allow_pickle=True).item())
    a.shuffled_labels,a.srd_permutation_sha,a.srd_label_flip_rate=shuffled_labels(y,tr,fold)
    return a

def historical_e0(fold,x,y,tr,te,worker):
    c=context();path=RT/"checkpoints"/f"E0_f{fold:02d}.best.pt";b=load(path);pred=np.asarray(b["prediction"],dtype=np.int64)
    historical_task=json.loads((HIST_EXP/".runtime/tasks"/f"g0_f{fold:02d}.json").read_text())
    assert hashlib.sha256(pred.tobytes()).hexdigest()==historical_task["prediction_sha256"]
    anchor=load(RT/"anchors"/f"fold{fold:02d}.pt");a=initialize_agent(1,fold,x,y,tr,te,anchor)
    a.MessagePassing0.load_state_dict(b["model"]["gfr"]);a.ModalFusion.load_state_dict(b["model"]["pln"])
    with c["torch"].no_grad():o=c["method"].evaluate(a,1,c["utils"])
    metrics=c["method"].metrics(o,y,te)
    assert np.array_equal(pred,o["logp"][te].argmax(1).detach().cpu().numpy())
    row={"group_id":0,"group":NAMES[0],"fold":fold,"seed":0,"worker":worker,"gpu":0,"status":"COMPLETE","reused_verified":True,
         "best_epoch":b["best_epoch"],"epochs_executed":b["epochs_executed"],"prediction_sha256":hashlib.sha256(pred.tobytes()).hexdigest(),"split_sha":fold_hash(tr,te),**metrics}
    atomic_json(RT/"tasks"/f"g0_f{fold:02d}.json",row);return row

def run_fold_group(gid,fold,worker,x,y,folds,mapping):
    c=context();m=c["method"];tr,te=folds[fold-1];task=RT/"tasks"/f"g{gid}_f{fold:02d}.json";bestpath=RT/"checkpoints"/f"g{gid}_f{fold:02d}.best.pt";latest=RT/"checkpoints"/f"g{gid}_f{fold:02d}.latest.pt"
    if task.exists() and json.loads(task.read_text()).get("status")=="COMPLETE":return
    anchor=load(RT/"anchors"/f"fold{fold:02d}.pt");a=initialize_agent(gid,fold,x,y,tr,te,anchor)
    wrong=wrong_relation_permutation(x,tr,mapping) if gid==8 else None
    perm,perm_sha,fliprate=shuffled_labels(y,tr,fold);a.srd_labels=perm if gid==9 else y
    if latest.exists():
        ck=load(latest);m.load_bundle(a,ck["model"]);a.optimizer_MP0.load_state_dict(ck["opt_mp"]);a.optimizer_MF.load_state_dict(ck["opt_mf"]);a.optimizer_APCR.load_state_dict(ck["opt_apcr"]);restore_rng(ck["rng"])
        begin=ck["epoch"]+1;best=ck["best"];bestacc=ck["bestacc"];be=ck["best_epoch"];wait=ck["wait"];trace=ck["trace"]
    else:
        begin=0;best=None;bestacc=-1.;be=-1;wait=0;trace=[];restore_rng(anchor["rng"])
    started=time.monotonic();atomic_json(task,{"status":"RUNNING","group_id":gid,"group":NAMES[gid],"fold":fold,"worker":worker,"gpu":0,"signature":signature(),"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}.pt"),"anchor_parameter_sha":anchor["parameter_sha"],"data_sha":sha(CFG["data"]["csv"]),"epoch":begin-1})
    for epoch in range(begin,450):
        if gid!=1:
            c["torch"].manual_seed(481400000+fold*10000+epoch);c["torch"].cuda.manual_seed_all(481400000+fold*10000+epoch)
        stat=m.train_step(a,gid,epoch,c["utils"],a.sites,a.srd_labels if gid==9 else y,wrong_perm=wrong,random_attack=(gid==10));trace.append({"epoch":epoch,**stat})
        with c["torch"].no_grad():out=m.evaluate(a,gid,c["utils"],wrong)
        acc=float(out["logp"][a.val_idx].argmax(1).eq(a.targ[a.val_idx]).float().mean())
        if acc>bestacc:bestacc=acc;best=m.model_bundle(a);be=epoch;wait=0
        else:wait+=1
        if (epoch+1)%20==0 or wait>200:
            save(latest,{"model":m.model_bundle(a),"opt_mp":a.optimizer_MP0.state_dict(),"opt_mf":a.optimizer_MF.state_dict(),"opt_apcr":a.optimizer_APCR.state_dict(),"rng":rng_state(),"epoch":epoch,"best":best,"bestacc":bestacc,"best_epoch":be,"wait":wait,"trace":trace})
        if wait>200:break
    m.load_bundle(a,best)
    with c["torch"].no_grad():out=m.evaluate(a,gid,c["utils"],wrong)
    row=m.metrics(out,y,te);pred=out["logp"][te].argmax(1).detach().cpu().numpy()
    diag={"group_id":gid,"group":NAMES[gid],"fold":fold,"best_epoch":be,"alpha":float(out["alpha"]),"message_norms":[float(v.norm().detach()) for v in out["deltas"]],"gate_means":[float(v.mean().detach()) for v in out["gates"]],"relative_residual_norm":float(out["residual"].norm().detach()/(out["base"].norm().detach()+1e-8)),"srd_label_flip_rate":fliprate,"srd_permutation_sha":perm_sha,"last_epoch_trace":trace[-1]}
    save(bestpath,{"model":best,"prediction":pred,"metrics":row,"diagnostics":diag,"best_epoch":be,"epochs_executed":epoch+1,"trace":trace});latest.unlink(missing_ok=True)
    result={"status":"COMPLETE","group_id":gid,"group":NAMES[gid],"fold":fold,"seed":0,"worker":worker,"gpu":0,"signature":signature(),"split_sha":fold_hash(tr,te),"anchor_sha":sha(RT/"anchors"/f"fold{fold:02d}.pt"),"anchor_parameter_sha":anchor["parameter_sha"],"best_sha":sha(bestpath),"prediction_sha256":hashlib.sha256(pred.tobytes()).hexdigest(),"best_epoch":be,"epochs_executed":epoch+1,"wall_seconds":time.monotonic()-started,"diagnostics":diag,**row}
    atomic_json(task,result);print("DONE",gid,fold,result["acc"],be,flush=True);return result

def init():
    x,y,folds,mapping=data();protected={str(p):sha(p) for p in [DATA_ROOT/n for n in ["main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]]}
    frozen={p.name:sha(p) for p in (RT/"checkpoints/frozen_e6").glob("*.pt")}
    atomic_json(HERE/"manifest.json",{"status":"INITIALIZED","code_signature":signature(),"data_sha":sha(CFG["data"]["csv"]),"protected_sha256":protected,"fold_sha256":[fold_hash(*f) for f in folds],"frozen_e6_sha256":frozen,"source_sha256":{p.name:sha(p) for p in SRC.glob("*.py")}})

def worker(worker_id,folds_arg):
    x,y,folds,mapping=data();folds_arg=[int(v) for v in folds_arg.split(",")]
    for fold in folds_arg:
        tr,te=folds[fold-1]
        e0=RT/"tasks"/f"g0_f{fold:02d}.json"
        if not e0.exists():historical_e0(fold,x,y,tr,te,worker_id)
        for gid in range(1,11):run_fold_group(gid,fold,worker_id,x,y,folds,mapping)

if __name__=="__main__":
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True);sub.add_parser("init");w=sub.add_parser("worker");w.add_argument("--worker",type=int,required=True);w.add_argument("--folds",required=True);a=ap.parse_args()
    if a.cmd=="init":init()
    else:worker(a.worker,a.folds)
