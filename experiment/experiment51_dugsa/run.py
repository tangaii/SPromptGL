#!/usr/bin/env python3
"""Experiment 51 DUGSA runner.

All folds use the historical B0-B transductive protocol.  E0 is independently
read from verified B0-B checkpoints; E1-E11 are trained from the matching
fold anchors.  Utility statistics are estimated only from outer-train labels
using deterministic inner five-fold cross-fitting.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, os, pickle, random, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
EXP = Path(os.environ.get("DUGSA_EXP", Path(__file__).resolve().parent))
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP/".runtime"; SRC = EXP/"source"; ANCHORS = ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"; HIST = ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/checkpoints"
TASKS=RT/"tasks"; CKPTS=RT/"checkpoints"; LOGS=RT/"logs"; FD=RT/"fold_data"; RESULTS=EXP/"results"
CFG_PATH=EXP/"config.json"; CFG=json.loads(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
DIMS=[48,6,10,256]
CTX=None; DATA=None; FOLDS=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f".tmp.{os.getpid()}")
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));os.replace(q,p)


def torch_save(path,obj):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f".tmp.{os.getpid()}")
    with q.open("wb") as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def torch_load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)


def csv_write(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    q=p.with_name(p.name+f".tmp.{os.getpid()}")
    with q.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def csv_read(path):
    p=Path(path)
    if not p.exists():return []
    with p.open(newline="") as f:return list(csv.DictReader(f))


def seed_all(seed):
    import numpy as np, torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def pack_rng():
    import numpy as np, torch
    return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_rng(r):
    import numpy as np, torch
    random.setstate(r["python"]);np.random.set_state(r["numpy"]);torch.set_rng_state(r["cpu"].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in r["cuda"]])


def rng_sha(r):
    import numpy as np
    h=hashlib.sha256(pickle.dumps((r["python"],r["numpy"]),protocol=4));h.update(r["cpu"].cpu().numpy().tobytes())
    for v in r["cuda"]:h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def runtime():
    global CTX
    if CTX is None:
        import torch, numpy as np, pandas as pd
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        sys.path.insert(0,str(SRC));sys.path.insert(0,str(EXP))
        import layers,network,utils,dugsa
        CTX={"torch":torch,"np":np,"pd":pd,"layers":layers,"network":network,"utils":utils,"dugsa":dugsa,"device":torch.device("cuda")}
    return CTX


def load_data():
    global DATA,FOLDS
    if DATA is not None:return DATA,FOLDS
    import numpy as np,pandas as pd
    from sklearn.model_selection import StratifiedKFold
    base=ROOT/"data/ABIDE";frame=pd.read_csv(base/"processed_standard_data.csv");mapping=np.load(base/"modal_feat_dict.npy",allow_pickle=True).item()
    expected=[str(v) for key in ("PHENO","ANAT","FUNC","Correlation") for v in mapping[key]]
    assert list(frame.columns[:-1])==expected
    x=frame.iloc[:,:-1].to_numpy(np.float32);y=frame.iloc[:,-1].to_numpy(np.int64)-1
    assert x.shape==(871,320) and np.isfinite(x).all() and np.bincount(y).tolist()==[403,468]
    assert [len(mapping[k]) for k in ("PHENO","ANAT","FUNC","Correlation")]==DIMS
    folds=list(StratifiedKFold(10,shuffle=True,random_state=10).split(x,y));DATA=(x,y);FOLDS=folds;return DATA,FOLDS


def split_sha(fold):
    import numpy as np
    tr,te=load_data()[1][fold-1];return hashlib.sha256(np.asarray(tr,np.int64).tobytes()+np.asarray(te,np.int64).tobytes()).hexdigest()


def hp():
    b=CFG["base"];return SimpleNamespace(n_hidden=b["n_hidden"],n_head=b["n_head"],dropout=b["dropout"],nlayer=b["nlayer"],nmodal=4,nclass=2,lr=b["lr"],reg=b["weight_decay"],th=b["threshold"],GC_mode="weighted-cosine",MP_mode="GCN",MF_mode="concat",alpha=.5,theta_smooth=b["theta_smooth"],theta_degree=b["theta_degree"],theta_sparsity=b["theta_sparsity"])


def digest_module(m):
    h=hashlib.sha256()
    for n,p in sorted(m.named_parameters()):h.update(n.encode()+b"\0"+p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def graph_init(model,anchor_gfr):
    """Load the audited GraphLearn initialization into either model type."""
    for m in range(4):
        src={k[len(f"GraphConstruct{m}."):]:v for k,v in anchor_gfr.items() if k.startswith(f"GraphConstruct{m}.")}
        if hasattr(model,"graphs"): model.graphs[m].load_state_dict(src,strict=True)


def gcn_init_legacy(model,anchor_gfr):
    for m in range(4):
        src={k[len(f"MessagePassing{m}."):]:v for k,v in anchor_gfr.items() if k.startswith(f"MessagePassing{m}.")}
        model.gcn[m].load_state_dict(src,strict=True)
    model.fusion_weight.data.copy_(anchor_gfr["fusion_weight"])


def build_state(group,fold):
    import torch, numpy as np
    c=runtime();x,y=load_data()[0];tr,te=load_data()[1][fold-1];anchor=torch_load(ANCHORS/f"fold{fold:02d}.pt");assert anchor["split_sha"]==split_sha(fold)
    # Main CAM and audited graph learners start from the exact fold anchor.
    seed_all(510000+fold*100+int(group[1:])); mode=CFG["groups"][int(group[1:])]["mode"]
    if group=="E1":
        model=c["dugsa"].LegacyFixedGFR(c["network"].GraphLearn,c["network"].GCN,c["utils"],th=CFG["base"]["threshold"]).cuda();graph_init(model,anchor["model"]["gfr"]);gcn_init_legacy(model,anchor["model"]["gfr"]);model.fusion_weight.data.copy_(anchor["model"]["gfr"]["fusion_weight"])
    else:
        model=c["dugsa"].DUGSAGFR(c["network"].GraphLearn,c["layers"],c["utils"],th=CFG["base"]["threshold"],mode=mode).cuda();graph_init(model,anchor["model"]["gfr"])
    cam=c["dugsa"].PLNNoPEN(DIMS,hp(),c["layers"]).cuda();cam.apply(c["utils"].my_weight_init);cam.load_state_dict(anchor["model"]["pln"])
    # Raw inputs and exact class weighting.
    feat=torch.from_numpy(x).float().cuda();targ=torch.from_numpy(y).long().cuda();train=torch.as_tensor(tr,dtype=torch.long,device="cuda")
    counts=Counter(y[tr]);weights=torch.from_numpy((len(tr)/np.asarray(list(counts.values()))/2).astype(np.float32)).cuda()
    if group=="E1":
        opt_model=torch.optim.Adam(model.parameters(),lr=CFG["base"]["lr"],weight_decay=CFG["base"]["weight_decay"])
    else:
        graph_params=list(model.graphs.parameters());graph_ids={id(p) for p in graph_params};new_params=[p for p in model.parameters() if id(p) not in graph_ids]
        opt_model=torch.optim.Adam([{"params":graph_params,"lr":CFG["base"]["lr"],"weight_decay":CFG["base"]["weight_decay"]},{"params":new_params,"lr":CFG["new_module"]["lr"],"weight_decay":CFG["new_module"]["weight_decay"]}])
    opt_cam=torch.optim.Adam(cam.parameters(),lr=CFG["base"]["lr"],weight_decay=CFG["base"]["weight_decay"])
    return {"group":group,"fold":fold,"model":model,"cam":cam,"opt_model":opt_model,"opt_cam":opt_cam,"feat":feat,"targ":targ,"train":train,"test":np.asarray(te),"weight":weights,"anchor":anchor,"parts":c["layers"].InitialDataLayer(DIMS),"mode":mode}


def site_ids(x):
    import numpy as np
    return np.argmax(x[:,26:46],axis=1).astype(np.int64)


def relation_permutation(x,tr,te,fold):
    import numpy as np
    sid=site_ids(x);rng=np.random.default_rng(85100+fold);perm=np.arange(len(x),dtype=np.int64)
    for ids in (tr,te):
        buckets={}
        for i in ids:buckets.setdefault(int(sid[i]),[]).append(int(i))
        for vals in buckets.values():
            if len(vals)>1:
                arr=np.asarray(vals);sh=arr.copy();rng.shuffle(sh)
                if np.all(sh==arr):sh=np.roll(arr,1)
                perm[arr]=sh
    return perm


def _pairs(ids,y,n_pairs,rng):
    import numpy as np
    ids=np.asarray(ids);yy=y[ids];i,j=np.triu_indices(len(ids),1);same=(yy[i]==yy[j]);a=np.where(same)[0];b=np.where(~same)[0];k=min(len(a),len(b),n_pairs//2)
    if k==0:return np.zeros((0,2),np.int64),np.zeros(0,np.int64)
    a=rng.choice(a,k,replace=False);b=rng.choice(b,k,replace=False);sel=np.concatenate([a,b]);rng.shuffle(sel);return np.stack([ids[i[sel]],ids[j[sel]]],1),same[sel].astype(np.int64)


def _cos(x,i,j):
    import numpy as np
    ai=x[i];bj=x[j];ai=ai/(np.linalg.norm(ai,axis=1,keepdims=True)+1e-8);bj=bj/(np.linalg.norm(bj,axis=1,keepdims=True)+1e-8);return np.sum(ai*bj,axis=1)


def _auc(y,s):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y))<2:return .5
    return float(roc_auc_score(y,s))


def compute_utility(fold):
    """Inner-five-fold cross-fitted relation utility; labels are outer-train only."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    x,y=load_data()[0];outer_tr,_=load_data()[1][fold-1];sid=site_ids(x);rng=np.random.default_rng(851000+fold);rows=[];g=np.zeros((4,4),np.float32)
    inner=list(StratifiedKFold(5,shuffle=True,random_state=51000+fold).split(outer_tr,y[outer_tr]))
    for m,n in PAIRS:
        vals=[]
        for ii,(a_idx,v_idx) in enumerate(inner):
            tr_ids=outer_tr[a_idx];va_ids=outer_tr[v_idx];pt,yt=_pairs(tr_ids,y,6000,np.random.default_rng(851000+fold*100+ii*20+m*4+n));pv,yv=_pairs(va_ids,y,3000,np.random.default_rng(851001+fold*100+ii*20+m*4+n))
            if len(pt)==0 or len(pv)==0:continue
            smt=_cos(x[:,[sum(DIMS[:m])+k for k in range(DIMS[m])]],pt[:,0],pt[:,1]);snt=_cos(x[:,[sum(DIMS[:n])+k for k in range(DIMS[n])]],pt[:,0],pt[:,1])
            smv=_cos(x[:,[sum(DIMS[:m])+k for k in range(DIMS[m])]],pv[:,0],pv[:,1]);snv=_cos(x[:,[sum(DIMS[:n])+k for k in range(DIMS[n])]],pv[:,0],pv[:,1])
            mu1,sd1=smt.mean(),smt.std()+1e-6;mu2,sd2=snt.mean(),snt.std()+1e-6
            def sig(z):return 1/(1+np.exp(-np.clip((z),-20,20)))
            # Site-stratified null: permute the second endpoint within its site bucket on the validation pairs.
            permj=pv[:,1].copy();buckets={}
            for k,j in enumerate(pv[:,1]):buckets.setdefault(int(sid[j]),[]).append(k)
            for ids in buckets.values():
                if len(ids)>1:
                    sh=np.asarray(ids);rng.shuffle(sh);permj[np.asarray(ids)]=pv[sh,1]
            snnull=_cos(x[:,[sum(DIMS[:n])+k for k in range(DIMS[n])]],pv[:,0],permj)
            a_m=_auc(yv,sig((smv-mu1)/sd1));a_n=_auc(yv,sig((snv-mu2)/sd2));a_j=_auc(yv,sig((smv-mu1)/sd1)*sig((snv-mu2)/sd2));a_null=_auc(yv,sig((smv-mu1)/sd1)*sig((snnull-mu2)/sd2));vals.append((a_m,a_n,a_j,a_null,len(pv)))
        if vals:
            arr=np.asarray(vals,float);am,an,aj,anull=arr[:,:4].mean(0);u=max(0.0,float(aj-max(am,an,anull)));gg=min(1.0,u/.03);g[m,n]=g[n,m]=gg;rows.append({"fold":fold,"pair":f"{MODS[m]}-{MODS[n]}","auc_m":am,"auc_n":an,"auc_joint":aj,"auc_matched_null":anull,"utility":u,"g":gg,"inner_folds":len(vals),"pair_n_mean":arr[:,4].mean()})
        else:rows.append({"fold":fold,"pair":f"{MODS[m]}-{MODS[n]}","auc_m":.5,"auc_n":.5,"auc_joint":.5,"auc_matched_null":.5,"utility":0.,"g":0.,"inner_folds":0,"pair_n_mean":0})
    return g,rows

MODS=["PHENO","ANAT","FUNC","CORR"];PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]


def prepare_fold(fold):
    import numpy as np
    FD.mkdir(parents=True,exist_ok=True);p=FD/f"fold{fold:02d}.json"
    if p.exists():return
    x,y=load_data()[0];tr,te=load_data()[1][fold-1];g,rows=compute_utility(fold);perm=relation_permutation(x,tr,te,fold);idx=np.asarray(tr)[np.linspace(0,len(tr)-1,min(128,len(tr)),dtype=int)]
    npz=FD/f"fold{fold:02d}.npz";np.savez_compressed(npz,utility=g,relation_perm=perm,align_idx=idx,train=np.asarray(tr),test=np.asarray(te));atomic_json(p,{"fold":fold,"split_sha":split_sha(fold),"utility":g.tolist(),"utility_rows":rows,"utility_sha":sha(npz),"relation_permutation_fraction":float(np.mean(perm!=np.arange(len(x))))})


def load_fold(fold):
    import numpy as np,torch
    z=np.load(FD/f"fold{fold:02d}.npz");return {k:torch.from_numpy(z[k]).cuda() for k in z.files}


def forward(state,fd,train_mode=True):
    import torch
    m=state["model"];cam=state["cam"]
    if train_mode:m.train();cam.train()
    else:m.eval();cam.eval()
    _,parts=state["parts"](state["feat"])
    perm=fd["relation_perm"].long() if state["group"]=="E9" else None
    self_only=state["group"]=="E10"
    if state["group"]=="E1": fused,adjs,details=m(parts,None,relation_perm=perm,force_self=self_only)
    else:
        m._labels_for_loss=state["targ"];fused,adjs,details=m(parts,fd["utility"],relation_perm=perm,force_self=self_only)
    graph=sum(runtime()["utils"].GraphConstructLoss(parts[i],adjs[i],CFG["base"]["theta_smooth"],CFG["base"]["theta_degree"],CFG["base"]["theta_sparsity"]) for i in range(4))/4.0
    if state["group"]=="E1": la=ln=lv=fused.new_zeros(());aux={"align_raw":0.,"n_raw":0.,"v_raw":0.}
    else:
        align_idx=fd["align_idx"].long();enable=state["group"] not in {"E3","E4","E6"}
        la,ln,lv,aux=m.auxiliary_losses(state["train"],align_idx,fd["utility"],CFG["loss"]["alignment"],CFG["loss"]["nuisance"],CFG["loss"]["noncollapse"],enable_align=enable)
    prob,hidden,attn=cam(fused);cls=runtime()["utils"].ClsLoss(prob,state["targ"],state["train"],state["weight"]);total=cls+graph+la+ln+lv
    return {"prob":prob,"hidden":hidden,"fused":fused,"parts":parts,"adjs":adjs,"details":details,"cls":cls,"graph":graph,"align":la,"nuisance":ln,"noncollapse":lv,"aux":aux,"total":total}


def metrics(logp,state):
    import numpy as np
    from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score
    idx=state["test"];y=state["targ"].detach().cpu().numpy()[idx];probs=logp.detach().exp().cpu().numpy()[idx];pred=probs.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1]);tp,fn,fp,tn=map(int,cm.ravel());hard=float(roc_auc_score(np.eye(2)[y],np.eye(2)[pred]));pa=float(roc_auc_score(y,probs[:,1]));spe=tn/max(tn+fp,1);sen=tp/max(tp+fn,1)
    return {"acc":float(np.mean(pred==y)),"hard_auc":hard,"prob_auc":pa,"spe":spe,"sen":sen,"ba":(spe+sen)/2,"asd_f1":float(f1_score(y,pred,pos_label=0,zero_division=0)),"asd_tp":tp,"asd_fn":fn,"td_fp":fp,"td_tn":tn,"n_test":len(idx)}


def grad_norm(mod):
    import torch
    z=[(p.grad.detach()**2).sum() for p in mod.parameters() if p.grad is not None]
    return float(torch.stack(z).sum().sqrt().cpu()) if z else 0.0


def state_dict_cpu(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def load_state(m,s):m.load_state_dict(s)


def task_path(g,f):return TASKS/f"{g}_f{f:02d}.json"
def ckpt_path(g,f):return CKPTS/f"{g}_f{f:02d}.best.pt"
def latest_path(g,f):return CKPTS/f"{g}_f{f:02d}.latest.pt"


def train_one(group,fold):
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"]="0";runtime();prepare_fold(fold);tp=task_path(group,fold)
    if tp.exists() and json.loads(tp.read_text()).get("status")=="COMPLETE":return
    state=build_state(group,fold);fd=load_fold(fold);start=time.time();best_acc=-1.;wait=0;best_epoch=-1;best=None;best_cam=None;best_opt=None;best_camopt=None;best_diag=None;epoch=-1
    # Each task starts from the same audited anchor; new modules have group/fold-derived seed.
    for epoch in range(CFG["base"]["epochs"]):
        state["opt_model"].zero_grad();state["opt_cam"].zero_grad();out=forward(state,fd,True);assert torch.isfinite(out["total"]);out["total"].backward()
        for mod in (state["model"],state["cam"]):
            for p in mod.parameters():
                if p.grad is not None:assert torch.isfinite(p.grad).all()
        gn=(grad_norm(state["model"]),grad_norm(state["cam"]));state["opt_model"].step();state["opt_cam"].step()
        with torch.no_grad():ev=forward(state,fd,False);met=metrics(ev["prob"],state)
        if met["acc"]>best_acc:
            best_acc=met["acc"];best_epoch=epoch;wait=0;best=state_dict_cpu(state["model"]);best_cam=state_dict_cpu(state["cam"]);best_opt=copy.deepcopy(state["opt_model"].state_dict());best_camopt=copy.deepcopy(state["opt_cam"].state_dict());best_diag={"train_cls":float(out["cls"].detach()),"train_graph":float(out["graph"].detach()),"align_loss":float(out["align"].detach()),"nuisance_loss":float(out["nuisance"].detach()),"noncollapse_loss":float(out["noncollapse"].detach()),"model_grad_norm":gn[0],"cam_grad_norm":gn[1],"gfr_message_norm":float((ev["fused"]-state["feat"]).detach().norm()),"fused_norm":float(ev["fused"].detach().norm()),"residual_ratio":float(((ev["fused"]-state["feat"]).detach().norm()/state["feat"].norm().clamp_min(1e-8))),"rho":ev["details"].get("rho",torch.zeros(4)).detach().cpu().tolist() if isinstance(ev["details"],dict) else [],"q":ev["details"].get("q",torch.zeros(4)).detach().cpu().tolist() if isinstance(ev["details"],dict) else [],"align_raw":float(ev["aux"]["align_raw"]),"n_raw":float(ev["aux"]["n_raw"]),"v_raw":float(ev["aux"]["v_raw"])}
            torch_save(ckpt_path(group,fold),{"group":group,"fold":fold,"seed":0,"best_epoch":best_epoch,"epochs_executed":epoch+1,"metrics":met,"model":best,"cam":best_cam,"opt_model":best_opt,"opt_cam":best_camopt,"diagnostics":best_diag,"utility":fd["utility"].cpu(),"split_sha":split_sha(fold),"anchor_parameter_sha":state["anchor"]["parameter_sha"],"anchor_rng_sha":rng_sha(state["anchor"]["rng"]),"source_sha":sha(EXP/"dugsa.py")})
        else:wait+=1
        if epoch%10==0:torch_save(latest_path(group,fold),{"epoch":epoch,"wait":wait,"best_acc":best_acc,"best_epoch":best_epoch,"model":state_dict_cpu(state["model"]),"cam":state_dict_cpu(state["cam"]),"opt_model":state["opt_model"].state_dict(),"opt_cam":state["opt_cam"].state_dict(),"rng":pack_rng()})
        if wait>CFG["base"]["early"]:break
    assert best is not None
    state["model"].load_state_dict(best);state["cam"].load_state_dict(best_cam)
    with torch.no_grad():check=metrics(forward(state,fd,False)["prob"],state)
    assert abs(check["acc"]-best_acc)<1e-8
    row={"group":group,"group_id":int(group[1:]),"name":CFG["groups"][int(group[1:])]["name"],"fold":fold,"seed":0,"gpu":0,"source":"new_training","status":"COMPLETE","split_sha":split_sha(fold),"anchor_parameter_sha":state["anchor"]["parameter_sha"],"anchor_rng_sha":rng_sha(state["anchor"]["rng"]),"best_epoch":best_epoch,"epochs_executed":epoch+1,"wall_seconds":time.time()-start,"best_sha":sha(ckpt_path(group,fold)),**check}
    atomic_json(tp,row);latest_path(group,fold).unlink(missing_ok=True);print("DONE",group,fold,best_epoch,row["acc"],flush=True)


def historical_e0_rows():
    import numpy as np,torch
    c=runtime();x,y=load_data()[0];rows=[]
    hist_rows={int(r["fold"]):r for r in csv_read(ROOT/"experiment/experiment43_1_b0b_gfr_mechanism_retest/results/fold_metrics.csv") if r.get("group")=="E00_B0B_REPLAY"}
    for fold,(tr,te) in enumerate(load_data()[1],1):
        h=torch_load(HIST/f"g0_f{fold:02d}.best.pt");gfr=c["network"].GFR(DIMS,DIMS,th=CFG["base"]["threshold"],mode="weighted-cosine").cuda();cam=c["dugsa"].PLNNoPEN(DIMS,hp(),c["layers"]).cuda();gfr.load_state_dict(h["model"]["gfr"]);cam.load_state_dict(h["model"]["pln"]);gfr.eval();cam.eval();feat=torch.from_numpy(x).float().cuda();targ=torch.from_numpy(y).long().cuda();ti=torch.as_tensor(tr,device="cuda")
        with torch.no_grad():
            _,parts=c["layers"].InitialDataLayer(DIMS)(feat);fused,_,_=gfr(parts,4,targ,ti);lp,_,_=cam(fused)
        st={"targ":targ,"test":np.asarray(te)};met=metrics(lp,st);old=hist_rows[fold];met.update({"group":"E0","group_id":0,"name":"B0-B_REFERENCE","fold":fold,"seed":0,"gpu":"historical","source":"Experiment43.1_verified","status":"REFERENCE","best_epoch":int(old["best_epoch"]),"epochs_executed":int(old["epochs_executed"]),"split_sha":old["split_sha"],"anchor_parameter_sha":old["initial_parameter_sha256"],"anchor_rng_sha":old["initial_rng_sha256"],"best_sha":old["best_sha"]});rows.append(met);del gfr,cam
    return rows


def prepare():
    import numpy as np
    for p in (RT,TASKS,CKPTS,LOGS,FD,RESULTS):p.mkdir(parents=True,exist_ok=True)
    load_data();[prepare_fold(f) for f in range(1,11)]
    manifest={"experiment":"Experiment51_DUGSA","status":"PREPARED","config_sha":sha(CFG_PATH),"code_sha":{p.name:sha(p) for p in [EXP/"run.py",EXP/"dugsa.py",EXP/"tests.py"] if p.exists()},"source_sha":{p.name:sha(p) for p in SRC.glob("*.py")},"protected_sha256":{str(ROOT/n):sha(ROOT/n) for n in ["main.py","model.py","network.py","layers.py","utils.py","data/ABIDE/processed_standard_data.csv","data/ABIDE/modal_feat_dict.npy"]},"data_shape":list(load_data()[0][0].shape),"label_counts":np.bincount(load_data()[0][1]).tolist(),"folds":{str(f):split_sha(f) for f in range(1,11)},"planned_groups":12,"historical_e0_folds":10,"planned_new_folds":110,"utility_inner_folds":5}
    atomic_json(EXP/"manifest.json",manifest);csv_write(RESULTS/"fold_metrics.csv",historical_e0_rows());print("PREPARE_PASS")


def controller():
    groups=[f"E{i}" for i in range(1,12)];tasks=[(g,f) for g in groups for f in range(1,11)];procs=[];start=time.time();workers=int(CFG["gpu_plan"]["workers"]);env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]="0";env.update({k:"2" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    while tasks or procs:
        alive=sum(p.poll() is None for _,p,_ in procs)
        while alive<workers and tasks:
            g,f=tasks.pop(0);log=(LOGS/f"{g}_f{f:02d}.log").open("ab");p=subprocess.Popen([sys.executable,"-B",str(EXP/"run.py"),"worker","--group",g,"--fold",str(f)],cwd=EXP,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);procs.append((g,p,log));alive+=1
        for item in list(procs):
            if item[1].poll() is not None: item[2].close();procs.remove(item)
        time.sleep(.5)
    statuses=[json.loads(task_path(g,f).read_text()).get("status") if task_path(g,f).exists() else "MISSING" for g,f in [(g,f) for g in groups for f in range(1,11)]];m=json.loads((EXP/"manifest.json").read_text());m.update(status="TRAINING_COMPLETE" if statuses.count("COMPLETE")==110 else "TRAINING_PARTIAL",completed_new_folds=statuses.count("COMPLETE"),failed_tasks=statuses.count("FAILED"),controller_wall_seconds=time.time()-start);atomic_json(EXP/"manifest.json",m);print("CONTROLLER_DONE",statuses.count("COMPLETE"),statuses.count("FAILED"))


def aggregate():
    import numpy as np
    rows=historical_e0_rows()
    for g in [f"E{i}" for i in range(1,12)]:
        for f in range(1,11):
            p=task_path(g,f);assert p.exists();r=json.loads(p.read_text());assert r["status"]=="COMPLETE";rows.append(r)
    csv_write(RESULTS/"fold_metrics.csv",rows);sums=[]
    for g in [f"E{i}" for i in range(12)]:
        rr=[r for r in rows if r["group"]==g];q={"group":g,"name":CFG["groups"][int(g[1:])]["name"],"folds":len(rr)}
        for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1"):
            z=np.asarray([float(r[k]) for r in rr]);q[k+"_mean"]=float(z.mean());q[k+"_sd"]=float(z.std(ddof=0))
        sums.append(q)
    csv_write(RESULTS/"summary.csv",sums);by={(r["group"],int(r["fold"])):r for r in rows};pairs=[]
    for other in [f"E{i}" for i in [0,1,2,3,4,6,7,8,9,10,11]]:
        for k in ("acc","hard_auc","prob_auc","spe","sen"):
            d=np.asarray([(float(by[("E5",f)][k])-float(by[(other,f)][k]))*100 for f in range(1,11)])
            pairs.append({"comparison":f"E5_vs_{other}","metric":k,"delta_pp":float(d.mean()),"wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),"e5_mean_pct":float(np.mean([by[("E5",f)][k] for f in range(1,11)])*100),"other_mean_pct":float(np.mean([by[(other,f)][k] for f in range(1,11)])*100)})
    csv_write(RESULTS/"paired_comparisons.csv",pairs);m=json.loads((EXP/"manifest.json").read_text());m.update(status="AGGREGATED",folds_complete=120,summary_sha=sha(RESULTS/"summary.csv"),paired_sha=sha(RESULTS/"paired_comparisons.csv"));atomic_json(EXP/"manifest.json",m);print("AGGREGATE_PASS")


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True);sp.add_parser("prepare");sp.add_parser("controller");sp.add_parser("aggregate");w=sp.add_parser("worker");w.add_argument("--group",required=True);w.add_argument("--fold",type=int,required=True);a=ap.parse_args()
    if a.cmd=="prepare":prepare()
    elif a.cmd=="controller":controller()
    elif a.cmd=="aggregate":aggregate()
    else:train_one(a.group,a.fold)

if __name__=="__main__":main()
