#!/usr/bin/env python3
"""Experiment 51.1 runner for DC-SRA-GFR."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, os, pickle, random, shutil, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
EXP = Path(os.environ.get("DCSRA_EXP", Path(__file__).resolve().parent))
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP/".runtime"; SRC=EXP/"source"; TASKS=RT/"tasks"; CKPTS=RT/"checkpoints"; LOGS=RT/"logs"; FD=RT/"fold_data"; HIST_E1=RT/"historical_e1"; RESULTS=EXP/"results"
ANCHORS=ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
HIST=ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints"
E51_SOURCE=Path(os.environ.get("E1_SOURCE", "/tmp/SPromptGL-E51/experiment51_dugsa/.runtime/checkpoints"))
CFG_PATH=EXP/"config.json"; CFG=json.loads(CFG_PATH.read_text())
DIMS=[48,6,10,256]; MODS=["PHENO","ANAT","FUNC","CORR"]
CTX=None; DATA=None; FOLDS=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f".tmp.{os.getpid()}")
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)); os.replace(q,p)


def torch_save(path,obj):
    import torch
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f".tmp.{os.getpid()}")
    with q.open("wb") as f:
        torch.save(obj,f); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)


def torch_load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)


def csv_write(path, rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    q=p.with_name(p.name+f".tmp.{os.getpid()}")
    with q.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(q,p)


def csv_read(path):
    p=Path(path)
    if not p.exists(): return []
    with p.open(newline="") as f: return list(csv.DictReader(f))


def seed_all(seed):
    import numpy as np, torch
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def pack_rng():
    import numpy as np, torch
    return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_rng(r):
    import numpy as np, torch
    random.setstate(r["python"]); np.random.set_state(r["numpy"]); torch.set_rng_state(r["cpu"].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in r["cuda"]])


def rng_sha(r):
    h=hashlib.sha256(pickle.dumps((r["python"],r["numpy"]),protocol=4)); h.update(r["cpu"].cpu().numpy().tobytes())
    for v in r["cuda"]: h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch, numpy as np, pandas as pd
        torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
        sys.path.insert(0,str(SRC)); sys.path.insert(0,str(EXP))
        import layers, network, utils, dcsra
        CTX={"torch":torch,"np":np,"pd":pd,"layers":layers,"network":network,"utils":utils,"dcsra":dcsra,"device":torch.device("cuda")}
    return CTX


def load_data():
    global DATA,FOLDS
    if DATA is not None: return DATA,FOLDS
    import numpy as np, pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/"data/ABIDE"; frame=pd.read_csv(base/"processed_standard_data.csv"); mapping=np.load(base/"modal_feat_dict.npy",allow_pickle=True).item()
    expected=[str(v) for key in ("PHENO","ANAT","FUNC","Correlation") for v in mapping[key]]
    assert list(frame.columns[:-1])==expected
    x=frame.iloc[:,:-1].to_numpy(np.float32); y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    assert [len(mapping[k]) for k in ("PHENO","ANAT","FUNC","Correlation")] == DIMS
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y)); DATA=(x,y); FOLDS=folds; return DATA,FOLDS


def split_sha(fold):
    import numpy as np
    tr,te=load_data()[1][fold-1]
    return hashlib.sha256(np.asarray(tr,np.int64).tobytes()+np.asarray(te,np.int64).tobytes()).hexdigest()


def hp():
    b=CFG["base"]
    return SimpleNamespace(n_hidden=b["n_hidden"],n_head=b["n_head"],dropout=b["dropout"],nlayer=b["nlayer"],nmodal=4,nclass=2,lr=b["lr"],reg=b["weight_decay"],th=b["threshold"],GC_mode="weighted-cosine",MP_mode="GCN",MF_mode="concat",alpha=.5,theta_smooth=b["theta_smooth"],theta_degree=b["theta_degree"],theta_sparsity=b["theta_sparsity"])


def site_ids(x):
    import numpy as np
    return np.argmax(x[:,26:46],axis=1).astype(np.int64)


def balanced_align_idx(train, y, n=128):
    import numpy as np
    rng=np.random.default_rng(511100+int(train[0])%997)
    a=train[y[train]==0]; b=train[y[train]==1]; k=min(n//2,len(a),len(b));
    aa=np.sort(rng.choice(a,k,replace=False)); bb=np.sort(rng.choice(b,k,replace=False)); return np.concatenate([aa,bb]).astype(np.int64)


def site_split_perm(x,tr,te,fold):
    import numpy as np
    sid=site_ids(x); rng=np.random.default_rng(511000+fold); p=np.arange(len(x),dtype=np.int64)
    for ids in (tr,te):
        buckets={}
        for i in ids: buckets.setdefault(int(sid[i]),[]).append(int(i))
        for vals in buckets.values():
            if len(vals)>1:
                arr=np.asarray(vals); sh=arr.copy(); rng.shuffle(sh)
                if np.all(sh==arr): sh=np.roll(arr,1)
                p[arr]=sh
    return p


def alignment_perm(x,tr,te,y,fold):
    import numpy as np
    sid=site_ids(x); rng=np.random.default_rng(511500+fold); out=np.tile(np.arange(len(x),dtype=np.int64),(4,1)); fractions=[]
    for m in (0,1,2):
        p=np.arange(len(x),dtype=np.int64)
        # Train-side: site x diagnosis strata where possible; test-side: site only.
        buckets={}
        for i in tr: buckets.setdefault((int(sid[i]),int(y[i])),[]).append(int(i))
        for vals in buckets.values():
            if len(vals)>1:
                arr=np.asarray(vals); sh=arr.copy(); rng.shuffle(sh)
                if np.all(sh==arr): sh=np.roll(arr,1)
                p[arr]=sh
        buckets={}
        for i in te: buckets.setdefault(int(sid[i]),[]).append(int(i))
        for vals in buckets.values():
            if len(vals)>1:
                arr=np.asarray(vals); sh=arr.copy(); rng.shuffle(sh)
                if np.all(sh==arr): sh=np.roll(arr,1)
                p[arr]=sh
        out[m]=p; fractions.append(float(np.mean(p!=np.arange(len(x)))))
    return out, fractions


def prepare_fold(fold):
    import numpy as np
    FD.mkdir(parents=True,exist_ok=True); p=FD/f"fold{fold:02d}.npz"
    if p.exists(): return
    x,y=load_data()[0]; tr,te=load_data()[1][fold-1]; align=balanced_align_idx(tr,y,128); rel=site_split_perm(x,tr,te,fold); ap,af=alignment_perm(x,tr,te,y,fold)
    np.savez_compressed(p,train=np.asarray(tr),test=np.asarray(te),align_idx=align,relation_perm=rel,align_perm=ap)
    atomic_json(FD/f"fold{fold:02d}.json",{"fold":fold,"split_sha":split_sha(fold),"align_idx":align.tolist(),"new_relation_permutation_fraction":float(np.mean(rel!=np.arange(len(x)))),"alignment_permutation_fraction":af})


def load_fold(fold):
    import numpy as np, torch
    z=np.load(FD/f"fold{fold:02d}.npz")
    return {k:torch.from_numpy(z[k]).cuda() for k in z.files}


def anchor_for(fold):
    a=torch_load(ANCHORS/f"fold{fold:02d}.pt")
    assert a["split_sha"]==split_sha(fold)
    return a


def build_state(group, fold):
    import torch, numpy as np
    c=runtime(); x,y=load_data()[0]; tr,te=load_data()[1][fold-1]; anchor=anchor_for(fold)
    # Same new-module initialization for all E2-E11 on this fold.
    seed_all(511100+fold)
    mode=CFG["groups"][str(int(group[1:]))]["mode"]
    model=c["dcsra"].DCSRA(c["network"].GraphLearn,c["network"].GCN,c["layers"],c["utils"],th=CFG["base"]["threshold"],mode=mode,hidden=CFG["new_module"]["hidden"]).cuda()
    c["dcsra"].load_e1_anchor(model,anchor["model"]["gfr"])
    cam=c["dcsra"].PLNNoPEN(DIMS,hp(),c["layers"]).cuda(); cam.apply(c["utils"].my_weight_init); cam.load_state_dict(anchor["model"]["pln"])
    feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda(); train=torch.as_tensor(tr,dtype=torch.long,device="cuda")
    counts=Counter(y[tr]); weights=torch.from_numpy((len(tr)/np.asarray(list(counts.values()))/2).astype("float32")).cuda()
    base_params=list(model.base.parameters()); ids={id(p) for p in base_params}; new_params=[p for p in model.parameters() if id(p) not in ids]
    opt_base=torch.optim.Adam(base_params,lr=CFG["base"]["lr"],weight_decay=CFG["base"]["weight_decay"])
    opt_new=torch.optim.Adam(new_params,lr=CFG["new_module"]["lr"],weight_decay=CFG["new_module"]["weight_decay"])
    opt_cam=torch.optim.Adam(cam.parameters(),lr=CFG["base"]["lr"],weight_decay=CFG["base"]["weight_decay"])
    return {"group":group,"fold":fold,"model":model,"cam":cam,"opt_base":opt_base,"opt_new":opt_new,"opt_cam":opt_cam,"feat":feat,"targ":targ,"train":train,"test":np.asarray(te),"weight":weights,"anchor":anchor,"parts":c["layers"].InitialDataLayer(DIMS),"mode":mode}


def metrics(logp,state,idx=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
    if idx is None: idx=state["test"]
    y=state["targ"].detach().cpu().numpy()[idx]; probs=logp.detach().exp().cpu().numpy()[idx]; pred=probs.argmax(1)
    cm=confusion_matrix(y,pred,labels=[0,1]); tp,fn,fp,tn=map(int,cm.ravel())
    hard=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred])); pa=float(roc_auc_score(y,probs[:,1])); spe=tn/max(tn+fp,1); sen=tp/max(tp+fn,1)
    return {"acc":float(np.mean(pred==y)),"hard_auc":hard,"prob_auc":pa,"spe":spe,"sen":sen,"ba":(spe+sen)/2,"asd_f1":float(f1_score(y,pred,pos_label=0,zero_division=0)),"asd_tp":tp,"asd_fn":fn,"td_fp":fp,"td_tn":tn,"n":len(idx)}


def grad_norm(mod):
    import torch
    z=[(p.grad.detach()**2).sum() for p in mod.parameters() if p.grad is not None]
    return float(torch.stack(z).sum().sqrt().cpu()) if z else 0.0


def forward_state(state, fd, train_mode=True, intervention=None):
    import torch
    m=state["model"]; cam=state["cam"]
    if train_mode: m.train(); cam.train()
    else: m.eval(); cam.eval()
    _, parts=state["parts"](state["feat"])
    mode=state["group"]
    rel=fd["relation_perm"].long() if mode=="E10" else None
    ap=fd["align_perm"].long() if mode=="E9" else None
    fused,adjs,details=m(parts,align_perm=ap,new_relation_perm=rel,intervention=intervention)
    graph=sum(runtime()["utils"].GraphConstructLoss(parts[i],adjs[i],CFG["base"]["theta_smooth"],CFG["base"]["theta_degree"],CFG["base"]["theta_sparsity"]) for i in range(4))/4.0
    g=int(mode[1:])
    enable_proto=g>=6; enable_align=g>=7
    enable_dec=g>=4; enable_var=g>=4
    lp,la,ld,lv,aux=m.auxiliary_losses(state["train"],fd["align_idx"].long(),align_perm=ap,lambda_proto=CFG["new_module"]["lambda_proto"],lambda_align=CFG["new_module"]["lambda_align"],lambda_dec=CFG["new_module"]["lambda_dec"],lambda_var=CFG["new_module"]["lambda_var"],enable_proto=enable_proto,enable_align=enable_align,uniform_align=mode=="E8",labels=state["targ"])
    if not enable_proto: lp=lp*0
    if not enable_align: la=la*0
    if not enable_dec: ld=ld*0
    if not enable_var: lv=lv*0
    prob,hidden,attn=cam(fused); cls=runtime()["utils"].ClsLoss(prob,state["targ"],state["train"],state["weight"]); total=cls+graph+lp+la+ld+lv
    return {"prob":prob,"hidden":hidden,"attn":attn,"fused":fused,"parts":parts,"adjs":adjs,"details":details,"cls":cls,"graph":graph,"proto":lp,"align":la,"dec":ld,"var":lv,"total":total,"aux":aux}


def diagnostics(state,fd,out,train_out=None):
    import numpy as np, torch
    d=out["details"]; gates=d["gates"].detach(); residual=torch.cat(d["residuals"],1).detach(); base=d["base_fused"].detach(); raw=state["feat"].detach();
    proto=d["proto"].detach(); ptrain=proto[state["train"]].argmax(1).cpu().numpy(); ytr=state["targ"][state["train"]].cpu().numpy(); ptest=proto[state["test"]].argmax(1).cpu().numpy(); yte=state["targ"][state["test"]].cpu().numpy()
    rows={"group":state["group"],"fold":state["fold"],"base_message_norm":float((base-raw).norm()),"base_message_ratio":float((base-raw).norm()/raw.norm().clamp_min(1e-8)),"new_residual_norm":float(residual.norm()),"new_residual_ratio_input":float(residual.norm()/raw.norm().clamp_min(1e-8)),"new_residual_ratio_base":float(residual.norm()/base.norm().clamp_min(1e-8)),"s_var_mean":float(torch.stack([b["sg"].detach().var(0,unbiased=False).mean() for b in d["branches"]]).mean()),"c_var_mean":float(torch.stack([b["c"].detach().var(0,unbiased=False).mean() for b in d["branches"]]).mean()),"gate_mean_all":float(gates.mean()),"gate_sd_all":float(gates.std()),"gate_mean_pheno":float(gates[:,0].mean()),"gate_mean_anat":float(gates[:,1].mean()),"gate_mean_func":float(gates[:,2].mean()),"gate_mean_corr":float(gates[:,3].mean()),"proto_train_acc":float(np.mean(ptrain==ytr)),"proto_test_acc":float(np.mean(ptest==yte)),"proto_train_asd_sens":float(np.mean(ptrain[ytr==0]==0)),"proto_test_asd_sens":float(np.mean(ptest[yte==0]==0)),"proto_train_td_spec":float(np.mean(ptrain[ytr==1]==1)),"proto_test_td_spec":float(np.mean(ptest[yte==1]==1)),"proto_pred1_train":float(np.mean(ptrain==1)),"proto_pred1_test":float(np.mean(ptest==1)),"proto_active":int(int(state["group"][1:])>=6),"prototype_distance":float(torch.norm(state["model"].prototypes[0].detach()-state["model"].prototypes[1].detach())),"proto_distance":float(torch.norm(state["model"].prototypes[0].detach()-state["model"].prototypes[1].detach())),"rho_mean":float(d["rho"].mean()),"rho_min":float(d["rho"].min()),"rho_max":float(d["rho"].max()),"proto_loss":float(out["proto"].detach()),"align_loss":float(out["align"].detach()),"dec_loss":float(out["dec"].detach()),"var_loss":float(out["var"].detach()),"align_raw":float(out["aux"]["align_raw"]),"align_w_pheno":float(out["aux"]["align_weights"][0]) if len(out["aux"]["align_weights"])>0 else 0.,"align_w_anat":float(out["aux"]["align_weights"][1]) if len(out["aux"]["align_weights"])>1 else 0.,"align_w_func":float(out["aux"]["align_weights"][2]) if len(out["aux"]["align_weights"])>2 else 0.}
    # New graph/self deltas per modality.
    for m in range(4): rows[f"graph_self_delta_{m}"]=float((d["branches"][m]["hg"]-d["branches"][m]["hi"]).detach().norm())
    return rows


def maybe_proto(x):
    return x[:,0] if x.ndim==2 else x


def state_cpu(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def ckpt_path(g,f): return CKPTS/f"{g}_f{f:02d}.best.pt"
def latest_path(g,f): return CKPTS/f"{g}_f{f:02d}.latest.pt"
def task_path(g,f): return TASKS/f"{g}_f{f:02d}.json"


def load_e1_rows():
    import numpy as np, torch
    c=runtime(); x,y=load_data()[0]; rows=[]
    for fold,(tr,te) in enumerate(load_data()[1],1):
        src=HIST_E1/f"E1_f{fold:02d}.best.pt"
        d=torch_load(src); model=c["dcsra"].LegacyFixedGFR(c["network"].GraphLearn,c["network"].GCN,c["utils"],th=CFG["base"]["threshold"]).cuda(); model.load_state_dict(d["model"],strict=True); cam=c["dcsra"].PLNNoPEN(DIMS,hp(),c["layers"]).cuda(); cam.load_state_dict(d["cam"],strict=True); model.eval(); cam.eval(); feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda();
        with torch.no_grad(): _,parts=c["layers"].InitialDataLayer(DIMS)(feat); fused,_,_=model(parts); lp,_,_=cam(fused)
        st={"targ":targ,"test":np.asarray(te)}; met=metrics(lp,st); met.update({"group":"E1","group_id":1,"name":CFG["groups"]["1"]["name"],"fold":fold,"seed":0,"gpu":"historical-scratch","source":"Experiment51_E1_checkpoint_readback","status":"REFERENCE","best_epoch":int(d["best_epoch"]),"epochs_executed":int(d["epochs_executed"]),"split_sha":split_sha(fold),"best_sha":sha(src),"e1_source_sha":sha(src),"n_test":len(te)})
        # Verify stored metric, but never substitute it.
        if abs(float(d["metrics"]["acc"])-met["acc"])>1e-7: raise RuntimeError(f"E1 fold {fold} stored/readback ACC mismatch")
        rows.append(met)
    return rows


def load_e0_rows():
    import numpy as np, torch
    c=runtime(); x,y=load_data()[0]; rows=[]
    oldrows={int(r["fold"]):r for r in csv_read(ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv") if r.get("group")=="E00_B0B_REPLAY"}
    for fold,(tr,te) in enumerate(load_data()[1],1):
        d=torch_load(HIST/f"g0_f{fold:02d}.best.pt"); gfr=c["network"].GFR(DIMS,DIMS,th=CFG["base"]["threshold"],mode="weighted-cosine").cuda(); cam=c["dcsra"].PLNNoPEN(DIMS,hp(),c["layers"]).cuda(); gfr.load_state_dict(d["model"]["gfr"]); cam.load_state_dict(d["model"]["pln"]); gfr.eval(); cam.eval(); feat=torch.from_numpy(x).float().cuda(); targ=torch.from_numpy(y).long().cuda(); ti=torch.as_tensor(tr,device="cuda")
        with torch.no_grad(): _,parts=c["layers"].InitialDataLayer(DIMS)(feat); fused,_,_=gfr(parts,4,targ,ti); lp,_,_=cam(fused)
        st={"targ":targ,"test":np.asarray(te)}; met=metrics(lp,st); old=oldrows[fold]; met.update({"group":"E0","group_id":0,"name":CFG["groups"]["0"]["name"],"fold":fold,"seed":0,"gpu":"historical","source":"Experiment43.1_verified","status":"REFERENCE","best_epoch":int(old["best_epoch"]),"epochs_executed":int(old["epochs_executed"]),"split_sha":split_sha(fold),"best_sha":old["best_sha"],"n_test":len(te)})
        if abs(float(old["acc"])-met["acc"])>1e-7: raise RuntimeError(f"E0 fold {fold} historical mismatch")
        rows.append(met)
    return rows


def build_state_resume(group,fold): return build_state(group,fold)


def train_one(group,fold):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"]="0"; runtime(); prepare_fold(fold); tp=task_path(group,fold)
    if tp.exists() and json.loads(tp.read_text()).get("status")=="COMPLETE": return
    state=build_state_resume(group,fold); fd=load_fold(fold); start=0; wait=0; best_acc=-1.; best_epoch=-1; best_model=best_cam=best_obm=best_onm=best_ocm=None; best_diag=None
    lp=latest_path(group,fold)
    if lp.exists():
        try:
            d=torch_load(lp); state["model"].load_state_dict(d["model"]); state["cam"].load_state_dict(d["cam"]); state["opt_base"].load_state_dict(d["opt_base"]); state["opt_new"].load_state_dict(d["opt_new"]); state["opt_cam"].load_state_dict(d["opt_cam"]); restore_rng(d["rng"]); start=int(d["epoch"])+1; wait=int(d["wait"]); best_acc=float(d["best_acc"]); best_epoch=int(d["best_epoch"]); best_model=d.get("best_model"); best_cam=d.get("best_cam"); best_obm=d.get("best_opt_base"); best_onm=d.get("best_opt_new"); best_ocm=d.get("best_opt_cam"); best_diag=d.get("best_diag")
        except Exception:
            start=0
    t0=time.time(); epoch=start-1
    try:
        for epoch in range(start,CFG["base"]["epochs"]):
            state["opt_base"].zero_grad(); state["opt_new"].zero_grad(); state["opt_cam"].zero_grad(); out=forward_state(state,fd,True); assert torch.isfinite(out["total"]); out["total"].backward()
            for mod in (state["model"],state["cam"]):
                for p in mod.parameters():
                    if p.grad is not None: assert torch.isfinite(p.grad).all()
            grads=(grad_norm(state["model"].base),grad_norm(state["model"],),grad_norm(state["cam"]))
            state["opt_base"].step(); state["opt_new"].step(); state["opt_cam"].step()
            with torch.no_grad(): ev=forward_state(state,fd,False); met=metrics(ev["prob"],state)
            if met["acc"]>best_acc:
                best_acc=met["acc"]; best_epoch=epoch; wait=0; best_model=state_cpu(state["model"]); best_cam=state_cpu(state["cam"]); best_obm=copy.deepcopy(state["opt_base"].state_dict()); best_onm=copy.deepcopy(state["opt_new"].state_dict()); best_ocm=copy.deepcopy(state["opt_cam"].state_dict()); best_diag=diagnostics(state,fd,ev,out)
                torch_save(ckpt_path(group,fold),{"group":group,"fold":fold,"seed":0,"best_epoch":best_epoch,"epochs_executed":epoch+1,"metrics":met,"model":best_model,"cam":best_cam,"opt_base":best_obm,"opt_new":best_onm,"opt_cam":best_ocm,"diagnostics":best_diag,"split_sha":split_sha(fold),"anchor_parameter_sha":state["anchor"]["parameter_sha"],"anchor_rng_sha":rng_sha(state["anchor"]["rng"]),"source_sha":sha(EXP/"dcsra.py")})
            else: wait+=1
            if epoch%10==0:
                torch_save(lp,{"epoch":epoch,"wait":wait,"best_acc":best_acc,"best_epoch":best_epoch,"model":state_cpu(state["model"]),"cam":state_cpu(state["cam"]),"opt_base":state["opt_base"].state_dict(),"opt_new":state["opt_new"].state_dict(),"opt_cam":state["opt_cam"].state_dict(),"rng":pack_rng(),"best_model":best_model,"best_cam":best_cam,"best_opt_base":best_obm,"best_opt_new":best_onm,"best_opt_cam":best_ocm,"best_diag":best_diag})
            if wait>CFG["base"]["early"]: break
        assert best_model is not None
        state["model"].load_state_dict(best_model); state["cam"].load_state_dict(best_cam)
        with torch.no_grad(): check=metrics(forward_state(state,fd,False)["prob"],state)
        assert abs(check["acc"]-best_acc)<1e-8
        row={"group":group,"group_id":int(group[1:]),"name":CFG["groups"][str(int(group[1:]))]["name"],"fold":fold,"seed":0,"gpu":0,"source":"new_training","status":"COMPLETE","split_sha":split_sha(fold),"anchor_parameter_sha":state["anchor"]["parameter_sha"],"anchor_rng_sha":rng_sha(state["anchor"]["rng"]),"best_epoch":best_epoch,"epochs_executed":epoch+1,"wall_seconds":time.time()-t0,"best_sha":sha(ckpt_path(group,fold)),**check}
        atomic_json(tp,row); lp.unlink(missing_ok=True); print("DONE",group,fold,best_epoch,row["acc"],flush=True)
    except Exception as e:
        atomic_json(tp,{"group":group,"fold":fold,"status":"FAILED","error":repr(e),"traceback":traceback.format_exc()}); raise


def prepare():
    import shutil, numpy as np
    for p in (RT,TASKS,CKPTS,LOGS,FD,HIST_E1,RESULTS): p.mkdir(parents=True,exist_ok=True)
    # Preserve the complete audited E1 readback source in this experiment's runtime.
    for f in range(1,11):
        src=E51_SOURCE/f"E1_f{f:02d}.best.pt"; dst=HIST_E1/src.name
        if not dst.exists(): shutil.copy2(src,dst)
    load_data(); [prepare_fold(f) for f in range(1,11)]
    e0=load_e0_rows(); e1=load_e1_rows(); csv_write(RESULTS/"fold_metrics.csv",e0+e1)
    manifest={"experiment":"Experiment51.1_DC-SRA-GFR","status":"PREPARED","config_sha":sha(CFG_PATH),"code_sha":{p.name:sha(p) for p in [EXP/"run.py",EXP/"dcsra.py",EXP/"tests.py"] if p.exists()},"source_sha":{p.name:sha(p) for p in SRC.glob("*.py")},"protected_sha256":{str(ROOT/n):sha(ROOT/n) for n in ["main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]},"data_shape":list(load_data()[0][0].shape),"label_counts":np.bincount(load_data()[0][1]).tolist(),"folds":{str(f):split_sha(f) for f in range(1,11)},"groups":12,"e0_reference_folds":10,"e1_reference_folds":10,"planned_new_folds":100,"e1_checkpoint_source":str(HIST_E1),"e1_checkpoint_sha":{str(f):sha(HIST_E1/f"E1_f{f:02d}.best.pt") for f in range(1,11)}}
    atomic_json(EXP/"manifest.json",manifest); print("PREPARE_PASS",len(e0),len(e1))


def controller():
    groups=[f"E{i}" for i in range(2,12)]; tasks=[(g,f) for g in groups for f in range(1,11)]; procs=[]; start=time.time(); workers=min(5,int(CFG["gpu_plan"]["workers"])); env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="0"; env.update({k:"2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    while tasks or procs:
        alive=sum(item[2].poll() is None for item in procs)
        while alive<workers and tasks:
            g,f=tasks.pop(0); log=(LOGS/f"{g}_f{f:02d}.log").open("ab"); p=subprocess.Popen([sys.executable,"-B",str(EXP/"run.py"),"worker","--group",g,"--fold",str(f)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); procs.append((g,f,p,log)); alive+=1
        for item in list(procs):
            if item[2].poll() is not None:
                item[3].close();
                if item[2].returncode!=0 and not task_path(item[0],item[1]).exists(): atomic_json(task_path(item[0],item[1]),{"group":item[0],"fold":item[1],"status":"FAILED","error":f"returncode {item[2].returncode}"})
                procs.remove(item)
        time.sleep(.5)
    statuses=[json.loads(task_path(g,f).read_text()).get("status") if task_path(g,f).exists() else "MISSING" for g in groups for f in range(1,11)]
    m=json.loads((EXP/"manifest.json").read_text()); m.update(status="TRAINING_COMPLETE" if statuses.count("COMPLETE")==100 else "TRAINING_PARTIAL",completed_new_folds=statuses.count("COMPLETE"),failed_tasks=statuses.count("FAILED"),controller_wall_seconds=time.time()-start); atomic_json(EXP/"manifest.json",m); print("CONTROLLER_DONE",statuses.count("COMPLETE"),statuses.count("FAILED"))


def aggregate():
    import numpy as np
    rows=load_e0_rows()+load_e1_rows()
    for g in [f"E{i}" for i in range(2,12)]:
        for f in range(1,11):
            p=task_path(g,f); assert p.exists(); r=json.loads(p.read_text()); assert r["status"]=="COMPLETE"; rows.append(r)
    csv_write(RESULTS/"fold_metrics.csv",rows); sums=[]
    for g in [f"E{i}" for i in range(12)]:
        rr=[r for r in rows if r["group"]==g]; q={"group":g,"name":CFG["groups"][str(int(g[1:]))]["name"],"folds":len(rr)}
        for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
            z=np.asarray([float(r[k]) for r in rr]); q[k+"_mean"]=float(z.mean()); q[k+"_sd"]=float(z.std(ddof=0))
        sums.append(q)
    csv_write(RESULTS/"summary.csv",sums); by={(r["group"],int(r["fold"])):r for r in rows}; pairs=[]
    for other in [f"E{i}" for i in [0,1,2,3,4,5,6,8,9,10,11]]:
        for k in ("acc","hard_auc","prob_auc","spe","sen"):
            d=np.asarray([(float(by[("E7",f)][k])-float(by[(other,f)][k]))*100 for f in range(1,11)])
            pairs.append({"comparison":f"E7_vs_{other}","metric":k,"delta_pp":float(d.mean()),"wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),"e7_mean_pct":float(np.mean([by[("E7",f)][k] for f in range(1,11)])*100),"other_mean_pct":float(np.mean([by[(other,f)][k] for f in range(1,11)])*100)})
    csv_write(RESULTS/"paired_comparisons.csv",pairs); m=json.loads((EXP/"manifest.json").read_text()); m.update(status="AGGREGATED",folds_complete=120,summary_sha=sha(RESULTS/"summary.csv"),paired_sha=sha(RESULTS/"paired_comparisons.csv")); atomic_json(EXP/"manifest.json",m); print("AGGREGATE_PASS")


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True); sp.add_parser("prepare"); sp.add_parser("controller"); sp.add_parser("aggregate"); w=sp.add_parser("worker"); w.add_argument("--group",required=True); w.add_argument("--fold",type=int,required=True); a=ap.parse_args()
    if a.cmd=="prepare": prepare()
    elif a.cmd=="controller": controller()
    elif a.cmd=="aggregate": aggregate()
    else: train_one(a.group,a.fold)

if __name__=="__main__": main()
