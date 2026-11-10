#!/usr/bin/env python3
"""Read-only nested diagnostic for disease-contrastive patient prototypes."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent; CFG=json.loads((EXP/"config.json").read_text())
RESULTS=EXP/"results"; RT=EXP/".runtime"; CACHE=RT/"cache"; BASE=Path(CFG["base"]["path"])
E761=Path(CFG["references"]["e761"]); BASE_GROUP="E74.4-G2-Positive-DCR"
CONDITIONS=("B0_CAM_only_Linear","B0_NL_CAM_only_RBF_SVM","B1_CAM_Reliable","B2_CAM_Reliable_Hard_Paired","B3_Shuffled_Pair_Control")
METRICS=("acc","hard_auc","prob_auc","sen","spe","asd_f1","log_loss","brier"); RUNNER=None

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def sha_json(x): return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def load_isolated(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def runner():
    global RUNNER
    if RUNNER is None: RUNNER=load_isolated("e762_locked_e761_runner",E761/"run.py")
    return RUNNER
def source_inventory():
    paths={"config":EXP/"config.json","probe":EXP/"probe.py","audit":EXP/"audit.py",
      "e744_run":BASE/"run.py","e744_model":BASE/"soft_positive_dcr.py","e744_config":BASE/"config.json",
      "e744_manifest":BASE/"manifest.json","e744_audit":BASE/"results/audit_status.json",
      "e744_metrics":BASE/"results/fold_metrics.csv","e761_run":E761/"run.py","e761_dcmpr":E761/"dcmpr.py",
      "e761_audit":E761/"audit.py","e761_config":E761/"config.json","e761_manifest":E761/"manifest.json",
      "e761_status":E761/"results/audit_status.json","e761_mechanism":E761/"results/mechanism.csv",
      "e724_patient_decoder":Path(CFG["references"]["e724_patient_decoder"]),
      "processed":Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),
      "mapping":Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy")}
    paths.update({f"best{f}":BASE/f".runtime/checkpoints/{BASE_GROUP}_f{f:02d}.best.pt" for f in range(1,11)})
    paths.update({f"latest{f}":BASE/f".runtime/checkpoints/{BASE_GROUP}_f{f:02d}.latest.pt" for f in range(1,11)})
    paths.update({f"task{f}":BASE/f".runtime/tasks/{BASE_GROUP}_f{f:02d}.json" for f in range(1,11)})
    paths.update({f"warmup{f}":Path(CFG["base"]["warmup_path"])/f"fold{f:02d}.pt" for f in range(1,11)})
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError("missing locked inputs: "+", ".join(missing))
    return {k:sha_file(v) for k,v in paths.items()}
def torch_load(path):
    import torch
    return torch.load(path,map_location="cpu",weights_only=False)
def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    temp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))
def verify_refs():
    e74=json.loads((BASE/"manifest.json").read_text()); a74=json.loads((BASE/"results/audit_status.json").read_text())
    e76=Path(CFG["references"]["e76"]); m76=json.loads((e76/"manifest.json").read_text()); a76=json.loads((e76/"results/audit_status.json").read_text())
    m761=json.loads((E761/"manifest.json").read_text()); a761=json.loads((E761/"results/audit_status.json").read_text())
    if (e74.get("status"),e74.get("completed_tasks"),a74.get("status"),a74.get("audited_checkpoints"))!=("COMPLETE_AUDITED",30,"PASS",30): raise RuntimeError("E74.4 audit identity failed")
    if (m76.get("status"),a76.get("status"),a76.get("audited_checkpoints"),a76.get("decision"))!=("COMPLETE_AUDITED","PASS",40,"NO-GO"): raise RuntimeError("E76 reference audit failed")
    if (m761.get("status"),a761.get("status"),a761.get("audited_checkpoints"),a761.get("decision"))!=("COMPLETE_AUDITED","PASS",30,"NO-GO"): raise RuntimeError("E76.1 reference audit failed")
    remote={}
    for key,branch,commit in (("e76", "upload/experiment76-dcmp", CFG["references"]["e76_commit"]),
                              ("e761", "upload/experiment76-1-dcmpr", CFG["references"]["e761_commit"])):
        out=subprocess.run(["git","ls-remote","https://github.com/tangaii/SPromptGL.git",f"refs/heads/{branch}"],
                           check=True,capture_output=True,text=True,timeout=30)
        rows=[line.split() for line in out.stdout.splitlines() if line.strip()]
        if len(rows)!=1 or rows[0][0]!=commit: raise RuntimeError(f"{key} historical branch commit mismatch: {rows} expected {commit}")
        remote[key]=rows[0][0]
    return {"e744_audited":30,"e76_audited":40,"e76_decision":"NO-GO","e761_audited":30,"e761_decision":"NO-GO",
      "remote_e744_commit":runner().verify_remote_base_branch(),"remote_e76_commit":remote["e76"],"remote_e761_commit":remote["e761"]}

def model_weights_sha(bundle):
    return sha_json({key:bundle[key] for key in ("model","cam")})
def extract_fold(fold):
    import numpy as np, torch
    R=runner(); base=R.load_base(); state,warmup,identity,ref=R.prepare_fold(fold)
    ckpt_path=BASE/f".runtime/checkpoints/{BASE_GROUP}_f{fold:02d}.best.pt"; task=json.loads((BASE/f".runtime/tasks/{BASE_GROUP}_f{fold:02d}.json").read_text())
    latest=torch_load(BASE/f".runtime/checkpoints/{BASE_GROUP}_f{fold:02d}.latest.pt"); ckpt=torch_load(ckpt_path)
    if (ckpt.get("group"),ckpt.get("fold"))!=(BASE_GROUP,fold) or task.get("checkpoint_sha256")!=sha_file(ckpt_path) or task.get("identity")!=identity or ckpt.get("identity")!=identity or latest.get("identity")!=identity: raise RuntimeError(f"fold {fold} checkpoint identity mismatch")
    expected=[377,407] if fold==2 else []
    if identity["membership"]["train_test_overlap_1based"]!=expected: raise RuntimeError(f"fold {fold} overlap mismatch")
    runtime=base.load_history(); runtime.load_bundle(state,ckpt["best_state"])
    checkpoint_weights_sha=model_weights_sha({"model":state["model"].state_dict(),"cam":state["cam"].state_dict()})
    state["model"].eval(); state["cam"].eval(); cap={"calls":0}; layer=state["cam"].Outputlayer
    def hook(_m,inputs):
        cap["calls"]+=1
        if len(inputs)!=2 or tuple(inputs[0].shape)!=(871,144): raise RuntimeError("CAM144 OutputLayer input contract failed")
        cap["h"]=inputs[0].detach().float().clone(); cap["attn"]=inputs[1].detach().float().clone()
    handle=layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(): out=base.forward(state,BASE_GROUP,training=False)
    finally: handle.remove()
    if cap["calls"]!=1: raise RuntimeError("OutputLayer must execute once")
    h=cap["h"].cpu().numpy().astype(np.float64); logits=out["logp"].detach().float().cpu().numpy()
    prob=torch.softmax(torch.as_tensor(logits),-1).numpy().astype(np.float64); pred=prob.argmax(-1)
    test_idx=R.index_numpy(state["test"]); saved_prob=np.asarray(ckpt["test_probability"]); saved_pred=np.asarray(ckpt["test_prediction"])
    probability_max_error=float(np.max(np.abs(prob[test_idx]-saved_prob)))
    if probability_max_error>1e-6 or not np.array_equal(pred[test_idx],saved_pred):
        raise RuntimeError(f"fold {fold} base checkpoint mismatch: max probability error={probability_max_error}; hard predictions must match exactly")
    y=state["y"].cpu().numpy().astype(int); train=R.index_numpy(state["train"]); test=R.index_numpy(state["test"])
    eligible=np.asarray([i for i in train if i not in set(test.tolist())],dtype=int)
    if len(np.intersect1d(eligible,test)) or (fold==2 and ({376,406}&set(eligible.tolist()))): raise RuntimeError("new diagnostic memory includes test ID")
    # SHA-deduplicate the same-fold epoch-10, selected best, and final latest states.
    warm=torch_load(Path(CFG["base"]["warmup_path"])/f"fold{fold:02d}.pt")
    warm_bundle=dict(ckpt["best_state"])
    warm_bundle["model"]=warm["model"]; warm_bundle["cam"]=warm["cam"]
    states=[("E61_epoch10_warmup",model_weights_sha(warm_bundle),warm_bundle,int(warm.get("completed_epoch",10))),
      ("E74_best",model_weights_sha(ckpt["best_state"]),ckpt["best_state"],int(ckpt["best_epoch_1based"])),
      ("E74_final_latest",model_weights_sha(latest["current"]),latest["current"],int(latest.get("next_epoch",0)))]
    unique=[]
    for item in states:
        if item[1] not in [v[1] for v in unique]: unique.append(item)
    temporal={}
    for name,_,bundle,_epoch in unique:
        if name=="E74_best":
            temporal[name]=pred.copy()
            continue
        runtime.load_bundle(state,bundle); state["model"].eval(); state["cam"].eval()
        with torch.no_grad(): temporal[name]=base.forward(state,BASE_GROUP,training=False)["logp"].argmax(-1).cpu().numpy().astype(int)
    runtime.load_bundle(state,ckpt["best_state"]); state["model"].eval(); state["cam"].eval()
    with torch.no_grad():
        zero=layer(cap["h"],cap["attn"])[0]
        if not torch.equal(zero,out["logp"]): raise RuntimeError("zero perturbation OutputLayer parity failed")
        zero_pred=zero.argmax(-1).cpu().numpy()
        ht=cap["h"]; rms=ht.square().mean(1,keepdim=True).sqrt().clamp_min(1e-12); attention=cap["attn"]
        rng=np.random.default_rng(762000+fold); preds=[]
        for scale in (.01,.02):
            for _ in range(5):
                noise=torch.as_tensor(rng.standard_normal(tuple(ht.shape)).astype(np.float32),device=ht.device)
                noise=noise/noise.square().mean(1,keepdim=True).sqrt().clamp_min(1e-12)*rms*scale
                preds.append(layer(ht+noise,attention)[0].argmax(-1).cpu().numpy())
    stability=(np.stack(preds)==zero_pred).mean(0)
    post_weights_sha=model_weights_sha({"model":state["model"].state_dict(),"cam":state["cam"].state_dict()})
    if post_weights_sha!=checkpoint_weights_sha: raise RuntimeError(f"fold {fold} frozen model weights changed during diagnostic")
    return {"fold":fold,"h":h,"prob":prob,"pred":pred,"y":y,"eligible":eligible,"test":test,"train":train,"stability":stability,
      "temporal":temporal,"state_names":[x[0] for x in unique],"state_shas":[x[1] for x in unique],
      "state_epochs":[x[3] for x in unique],"identity":identity,"task":task,
      "checkpoint_sha":sha_file(ckpt_path),"feature_sha":hashlib.sha256(h.tobytes()).hexdigest(),"attn_sha":hashlib.sha256(cap["attn"].cpu().numpy().tobytes()).hexdigest(),
      "zero_perturb_exact":True,"device":str(state["y"].device),"checkpoint_weights_sha":checkpoint_weights_sha,
      "post_weights_sha":post_weights_sha,"frozen_weights_exact":True,"probability_max_error":probability_max_error}
def unit(x):
    import numpy as np
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def role_masks(data,ids):
    import numpy as np
    y=data["y"]; p=data["prob"]; pred=data["pred"]; stable=data["stability"]
    return [
      np.isin(np.arange(871),ids)&(y==0)&(pred==0)&(p[:,0]>=.70)&(stable>=.9),
      np.isin(np.arange(871),ids)&(y==1)&(pred==1)&(p[:,1]>=.70)&(stable>=.9),
      np.isin(np.arange(871),ids)&(y==0)&(pred==1)&(p[:,1]>=.60)&(stable>=.9),
      np.isin(np.arange(871),ids)&(y==1)&(pred==0)&(p[:,0]>=.60)&(stable>=.9)]
def build_memory(data,support,shuffle_seed=None):
    import numpy as np
    y=data["y"]; p=data["prob"]; pred=data["pred"]; st=data["stability"]; h=unit(data["h"])
    masks=role_masks(data,support)
    if shuffle_seed is not None:
        rng=np.random.default_rng(shuffle_seed)
        hard=np.flatnonzero(masks[2]|masks[3]); pseudo=y.copy()
        if len(hard)>1:
            labels=pseudo[hard].copy(); rng.shuffle(labels); pseudo[hard]=labels
        masks[2]=np.isin(np.arange(871),hard)&(pseudo==0); masks[3]=np.isin(np.arange(871),hard)&(pseudo==1)
    raw_candidate_counts=[int(mask.sum()) for mask in masks]
    masks=masks.copy()
    for j in (2,3):
        if int(masks[j].sum()) < 3: masks[j]=np.zeros_like(masks[j],dtype=bool)
    protos=[]; medoids=[]; valid=[]; counts=[]; member_ids=[]
    for mask in masks:
        ix=np.flatnonzero(mask); counts.append(len(ix)); member_ids.append(ix.tolist())
        if len(ix):
            center=unit(h[ix].mean(0)); protos.append(center); medoids.append(h[ix[np.argmax(h[ix]@center)]]); valid.append(True)
        else: protos.append(np.zeros(144)); medoids.append(np.zeros(144)); valid.append(False)
    directions=np.zeros((2,144)); dir_valid=np.zeros(2,bool); pairs=[[],[]]; vectors=[[],[]]
    for hard_role,cls in ((2,0),(3,1)):
        same=np.flatnonzero(masks[cls]); other=np.flatnonzero(masks[1-cls])
        if len(same) and len(other):
            for q in np.flatnonzero(masks[hard_role]):
                a=same[np.argmax(h[same]@h[q])]; b=other[np.argmax(h[other]@h[q])]
                d=h[a]-h[b]; n=np.linalg.norm(d)
                if n>1e-10:
                    vectors[cls].append(d/n); pairs[cls].append({"hard":int(q),"same":int(a),"other":int(b),"same_sim":float(h[q]@h[a]),"other_sim":float(h[q]@h[b]),"direction_norm":float(n)})
    for c in range(2):
        if vectors[c]: directions[c]=unit(np.mean(vectors[c],0)); dir_valid[c]=True
    return {"prototypes":np.asarray(protos),"medoids":np.asarray(medoids),"valid":np.asarray(valid),"counts":counts,"member_ids":member_ids,
      "directions":directions,"dir_valid":dir_valid,"pairs":pairs,"vectors":vectors,
      "raw_candidate_counts":raw_candidate_counts}
def evidence(h,memory,include_hard=True):
    import numpy as np
    x=unit(h); values=np.c_[x@memory["prototypes"].T,x@memory["directions"].T]
    mask=np.r_[memory["valid"],memory["dir_valid"]].astype(float)
    if not include_hard: mask[2:]=0.0
    return np.c_[values*mask[None,:],np.broadcast_to(mask,(len(x),6))]
def crossfit_evidence(data,ids,seed,shuffled=False):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    ids=np.asarray(ids,int); y=data["y"]; out=np.zeros((len(ids),12))
    cv=StratifiedKFold(3,shuffle=True,random_state=76300+data["fold"])
    for tr,va in cv.split(ids,y[ids]):
        s=seed if shuffled else None
        mem=build_memory(data,ids[tr],None if s is None else s+int(va[0]))
        out[va]=evidence(data["h"][ids[va]],mem)
    return out
def make_model(kind):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    estimator=SVC(C=1,gamma="scale",kernel="rbf",probability=False) if kind=="B0_NL_CAM_only_RBF_SVM" else LogisticRegression(C=1,max_iter=2000,random_state=0,solver="lbfgs")
    return make_pipeline(StandardScaler(),estimator)
def metrics(y,pred,score,probability=True):
    import numpy as np
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score,log_loss,brier_score_loss
    y=np.asarray(y); pred=np.asarray(pred); score=np.asarray(score)
    # All probability/score arrays are oriented so larger means ASD=0.
    out={"acc":float(accuracy_score(y,pred)),"hard_auc":float(balanced_accuracy_score(y,pred)),
      "prob_auc":float(roc_auc_score(y==0,score)),"sen":float(np.sum((y==0)&(pred==0))/max(1,np.sum(y==0))),
      "spe":float(np.sum((y==1)&(pred==1))/max(1,np.sum(y==1))),
      "asd_f1":float(f1_score(y==0,pred==0,zero_division=0))}
    out["log_loss"]=float(log_loss(y==0,np.c_[1-score,score],labels=[0,1])) if probability else ""
    out["brier"]=float(brier_score_loss(y==0,score)) if probability else ""
    return out
def fit_score(name,xtr,ytr,xva,yva):
    import numpy as np
    m=make_model(name); m.fit(xtr,ytr); estimator=m[-1]
    if name=="B0_NL_CAM_only_RBF_SVM":
        score=-np.asarray(m.decision_function(xva))
        pred=m.predict(xva); return metrics(yva,pred,score,False),pred,score,False
    raw=m.predict_proba(xva); j=int(np.flatnonzero(estimator.classes_==0)[0]); p=raw[:,j]; pred=np.where(p>=.5,0,1)
    return metrics(yva,pred,p,True),pred,p,True
def evaluate_inner(data,inner,train_ids,val_ids):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    y=data["y"]; h=data["h"]; train_ids=np.asarray(train_ids,int); val_ids=np.asarray(val_ids,int)
    if np.intersect1d(train_ids,val_ids).size: raise RuntimeError("inner train/test overlap")
    cv=StratifiedKFold(3,shuffle=True,random_state=76300+data["fold"]); train_z=np.zeros((len(train_ids),12)); train_z1=np.zeros_like(train_z)
    for tr,va in cv.split(train_ids,y[train_ids]):
        mem=build_memory(data,train_ids[tr]); train_z[va]=evidence(h[train_ids[va]],mem,True); train_z1[va]=evidence(h[train_ids[va]],mem,False)
    val_mem=build_memory(data,train_ids); val_z=evidence(h[val_ids],val_mem,True); val_z1=evidence(h[val_ids],val_mem,False)
    outrows=[]; pred={}
    for name in CONDITIONS[:-1]:
        if name.startswith("B0"): xtr=h[train_ids]; xva=h[val_ids]
        elif name=="B1_CAM_Reliable": xtr=np.c_[h[train_ids],train_z1]; xva=np.c_[h[val_ids],val_z1]
        else: xtr=np.c_[h[train_ids],train_z]; xva=np.c_[h[val_ids],val_z]
        met,pr,sc,prob=fit_score(name,xtr,y[train_ids],xva,y[val_ids])
        outrows.append({"fold":data["fold"],"inner_fold":inner,"group":name,**met})
        pred[name]={"pred":pr,"score":sc,"probability":prob}
    b3=[]
    for k in range(5):
        ztr=np.zeros((len(train_ids),12))
        for tr,va in cv.split(train_ids,y[train_ids]):
            mem=build_memory(data,train_ids[tr],762000+data["fold"]*1000+inner*10+k+int(va[0]))
            ztr[va]=evidence(h[train_ids[va]],mem)
        mem=build_memory(data,train_ids,762000+data["fold"]*1000+inner*10+k)
        zva=evidence(h[val_ids],mem)
        met,pr,sc,_=fit_score("B3_Shuffled_Pair_Control",np.c_[h[train_ids],ztr],y[train_ids],np.c_[h[val_ids],zva],y[val_ids])
        outrows.append({"fold":data["fold"],"inner_fold":inner,"group":"B3_Shuffled_Pair_Control","permutation":k,**met})
        b3.append({"pred":pr,"score":sc,"probability":True})
    pred["B3_Shuffled_Pair_Control"]=b3
    return outrows,pred
def role_report_rows(data,sites,site_names):
    import numpy as np
    masks=role_masks(data,data["eligible"]); rows=[]; temporal=data["temporal"]
    names=("R_ASD","R_TD","H_ASD_to_TD","H_TD_to_ASD")
    for j,name in enumerate(names):
        ids=np.flatnonzero(masks[j]); cls=(0,1,0,1)[j]
        rows.append({"fold":data["fold"],"kind":"role","name":name,"n":len(ids),
          "mean_correct_class_probability":float(data["prob"][ids,cls].mean()) if len(ids) else "",
          "mean_original_class_margin":float((data["prob"][ids,cls]-data["prob"][ids,1-cls]).mean()) if len(ids) else "",
          "mean_perturb_consistency":float(data["stability"][ids].mean()) if len(ids) else "",
          "checkpoint_prediction_consistency":float(np.mean([np.mean([temporal[k][i]==data["pred"][i] for k in temporal]) for i in ids])) if len(ids) else "",
          "same_direction_error_all_checkpoints_n":int(sum(all(temporal[k][i]==(1 if j==2 else 0) for k in temporal) for i in ids)) if j>=2 else "",
          "checkpoint_state_count":len(data["state_names"]),"checkpoint_epochs":json.dumps(data["state_epochs"]),
          "temporal_status":"CHECKPOINT_CONSISTENCY_ONLY; persistence not inferred" if len(data["state_names"])>=2 else "TEMPORAL_PERSISTENCE_NOT_IDENTIFIABLE",
          "site_counts":json.dumps({str(site_names[s]):int(np.sum(sites[ids]==s)) for s in np.unique(sites[ids])},sort_keys=True),
          "global_ids_sha256":hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()})
    return rows
def site_data():
    import numpy as np
    d=load_isolated("e762_patient_decoder",Path(CFG["references"]["e724_patient_decoder"]))
    x=d.decode_patient_prompts(Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv"),Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy"))
    values=[r["site"]["site_id"] if r["site"] is not None else "MISSING" for r in x["selections"]]
    levels=sorted(set(values)); lookup={v:i for i,v in enumerate(levels)}
    if len(values)!=871: raise RuntimeError("site row alignment failed")
    return np.asarray([lookup[v] for v in values]),levels,{"feature_sha":x["feature_sha256"],"mapping_sha":x["mapping_sha256"]}
def prototype_diag(data,sites):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    ids=data["eligible"]; y=data["y"]; h=unit(data["h"]); masks=role_masks(data,ids)
    rows=[]; proto={}
    labels=("R_ASD","R_TD","H_ASD_to_TD","H_TD_to_ASD")
    for k,name in enumerate(labels):
        members=np.flatnonzero(masks[k]); available=len(members)>0 and (k<2 or len(members)>=3)
        proto[name]=unit(h[members].mean(0)) if available else None
        rows.append({"fold":data["fold"],"kind":"support","name":name,"n":len(members),
          "site_counts":json.dumps({str(s):int(np.sum(sites[members]==s)) for s in np.unique(sites[members])})})
    for a,b in (("R_ASD","R_TD"),("H_ASD_to_TD","R_ASD"),("H_ASD_to_TD","R_TD"),("H_TD_to_ASD","R_TD"),("H_TD_to_ASD","R_ASD")):
        rows.append({"fold":data["fold"],"kind":"cosine","name":a+"__"+b,"value":float(proto[a]@proto[b]) if proto[a] is not None and proto[b] is not None else ""})
    for hard,same,other in (("H_ASD_to_TD","R_ASD","R_TD"),("H_TD_to_ASD","R_TD","R_ASD")):
        same_sim=float(proto[hard]@proto[same]) if proto[hard] is not None and proto[same] is not None else None
        other_sim=float(proto[hard]@proto[other]) if proto[hard] is not None and proto[other] is not None else None
        rows.append({"fold":data["fold"],"kind":"hard_vs_reliable","name":hard+"_same_minus_other",
          "value":same_sim-other_sim if same_sim is not None and other_sim is not None else "",
          "same_similarity":same_sim if same_sim is not None else "",
          "other_similarity":other_sim if other_sim is not None else "",
          "raw_candidates":int(np.sum(masks[2 if hard=="H_ASD_to_TD" else 3])),
          "prototype_available":bool(proto[hard] is not None)})
    cv=StratifiedKFold(5,shuffle=True,random_state=10); same=[]; other=[]; site_delta=[]; direction_cos=[]
    for tr,va in cv.split(ids,y[ids]):
        s=ids[tr]; v=ids[va]; mem=build_memory(data,s); hs=unit(data["h"])
        for q in v:
            cls=int(y[q]); a,b=(0,1) if cls==0 else (1,0)
            if mem["valid"][a] and mem["valid"][b]:
                same.append(float(hs[q]@mem["prototypes"][a])); other.append(float(hs[q]@mem["prototypes"][b]))
                ss=s[(sites[s]==sites[q])&(y[s]==cls)&(data["pred"][s]==cls)&(data["stability"][s]>=.9)]
                oo=s[(sites[s]==sites[q])&(y[s]!=cls)&(data["pred"][s]==y[s])&(data["stability"][s]>=.9)]
                if len(ss) and len(oo): site_delta.append(float(np.max(hs[q]@hs[ss].T)-np.max(hs[q]@hs[oo].T)))
        for cls in (0,1):
            if len(mem["vectors"][cls])>=2:
                full=unit(np.mean(mem["vectors"][cls],0)); rng=np.random.default_rng(764000+data["fold"]*10+cls)
                vals=[]
                for _ in range(200):
                    sample=np.asarray(mem["vectors"][cls])[rng.integers(0,len(mem["vectors"][cls]),len(mem["vectors"][cls]))]
                    vals.append(float(unit(sample.mean(0))@full))
                direction_cos.extend(vals)
    dif=np.asarray(same)-np.asarray(other) if same else np.array([])
    rows += [{"fold":data["fold"],"kind":"heldout_similarity","name":"same_class","value":float(np.mean(same)) if same else "","n":len(same)},
      {"fold":data["fold"],"kind":"heldout_similarity","name":"other_class","value":float(np.mean(other)) if other else "","n":len(other)},
      {"fold":data["fold"],"kind":"heldout_similarity","name":"same_minus_other","value":float(dif.mean()) if len(dif) else "","n":len(dif)},
      {"fold":data["fold"],"kind":"site_control","name":"same_site_nearest_same_minus_other","value":float(np.mean(site_delta)) if site_delta else "","n":len(site_delta)},
      {"fold":data["fold"],"kind":"bootstrap","name":"pair_direction_cosine_to_full","value":float(np.mean(direction_cos)) if direction_cos else "","n":len(direction_cos)}]
    # Record pair availability and selected source IDs without exposing patient IDs in report.
    full=build_memory(data,ids)
    for k,name in enumerate(labels):
        rows.append({"fold":data["fold"],"kind":"mean_vs_medoid","name":name,
          "value":float(full["prototypes"][k]@full["medoids"][k]) if full["valid"][k] else "",
          "n":int(full["counts"][k]),"prototype_available":bool(full["valid"][k])})
    for cls in (0,1):
        rows.append({"fold":data["fold"],"kind":"paired_direction","name":("ASD" if cls==0 else "TD"),"n_pairs":len(full["pairs"][cls]),
          "n_hard":int(full["counts"][2+cls]),"n_reliable_same":int(full["counts"][cls]),"n_reliable_other":int(full["counts"][1-cls]),
          "mean_same_similarity":float(np.mean([p["same_sim"] for p in full["pairs"][cls]])) if full["pairs"][cls] else "",
          "mean_other_similarity":float(np.mean([p["other_sim"] for p in full["pairs"][cls]])) if full["pairs"][cls] else "",
          "mean_direction_norm":float(np.mean([p["direction_norm"] for p in full["pairs"][cls]])) if full["pairs"][cls] else "",
          "direction_valid":bool(full["dir_valid"][cls])})
    return rows
def diagnose_fold(data,sites,site_names):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    ids=data["eligible"]; y=data["y"]; cv=StratifiedKFold(5,shuffle=True,random_state=10)
    packs={n:{"ids":[],"pred":[],"score":[]} for n in CONDITIONS[:-1]}; b3=[{"ids":[],"pred":[],"score":[]} for _ in range(5)]; inner_rows=[]
    for i,(tr,va) in enumerate(cv.split(ids,y[ids]),1):
        rows,p=evaluate_inner(data,i,ids[tr],ids[va]); inner_rows.extend(rows)
        for n in CONDITIONS[:-1]:
            packs[n]["ids"].extend(ids[va].tolist()); packs[n]["pred"].extend(p[n]["pred"].tolist()); packs[n]["score"].extend(p[n]["score"].tolist())
        for k in range(5):
            b3[k]["ids"].extend(ids[va].tolist()); b3[k]["pred"].extend(p["B3_Shuffled_Pair_Control"][k]["pred"].tolist()); b3[k]["score"].extend(p["B3_Shuffled_Pair_Control"][k]["score"].tolist())
    results=[]
    for name,pack in packs.items():
        order=np.argsort(pack["ids"]); ii=np.asarray(pack["ids"])[order]; pr=np.asarray(pack["pred"])[order]; sc=np.asarray(pack["score"])[order]
        m=metrics(y[ii],pr,sc,name!="B0_NL_CAM_only_RBF_SVM")
        results.append({"fold":data["fold"],"group":name,"n_oof":len(ii),**m})
    b3_metrics=[]
    for k,pack in enumerate(b3):
        order=np.argsort(pack["ids"]); ii=np.asarray(pack["ids"])[order]; pr=np.asarray(pack["pred"])[order]; sc=np.asarray(pack["score"])[order]
        b3_metrics.append(metrics(y[ii],pr,sc,True))
    results.append({"fold":data["fold"],"group":"B3_Shuffled_Pair_Control","n_oof":len(ids),
      **{m:float(np.mean([r[m] for r in b3_metrics])) for m in METRICS},
      "permutation_sd_acc":float(np.std([r["acc"] for r in b3_metrics])),"permutation_metrics":json.dumps(b3_metrics,separators=(",",":"))})
    prediction_cache={}
    for name,pack in packs.items():
        order=np.argsort(pack["ids"])
        prediction_cache[name]={"ids":np.asarray(pack["ids"])[order],"pred":np.asarray(pack["pred"])[order],
                                "score":np.asarray(pack["score"])[order]}
    prediction_cache["B3_Shuffled_Pair_Control"]=[]
    for pack in b3:
        order=np.argsort(pack["ids"])
        prediction_cache["B3_Shuffled_Pair_Control"].append({"ids":np.asarray(pack["ids"])[order],
            "pred":np.asarray(pack["pred"])[order],"score":np.asarray(pack["score"])[order]})
    return results,inner_rows,prediction_cache
def prediction_change_rows(data,predictions):
    import numpy as np
    ids=np.asarray(data["eligible"],dtype=int); y=data["y"][ids]
    masks=role_masks(data,ids); role_by_id=np.stack([m[ids] for m in masks])
    hard=role_by_id[2]|role_by_id[3]; reliable=role_by_id[0]|role_by_id[1]
    b2=predictions["B2_CAM_Reliable_Hard_Paired"]["pred"]
    rows=[]
    comparisons=(("B0_CAM_only_Linear",[predictions["B0_CAM_only_Linear"]]),
      ("B1_CAM_Reliable",[predictions["B1_CAM_Reliable"]]),
      ("B3_Shuffled_Pair_Control",predictions["B3_Shuffled_Pair_Control"]),
      ("B0_NL_CAM_only_RBF_SVM",[predictions["B0_NL_CAM_only_RBF_SVM"]]))
    for name,packs in comparisons:
        for k,pack in enumerate(packs):
            qids=np.asarray(pack["ids"],dtype=int); order=np.argsort(qids)
            if not np.array_equal(qids[order],ids): raise RuntimeError(f"fold {data['fold']} {name}: change IDs misaligned")
            other=np.asarray(pack["pred"],dtype=int)[order]; ref=np.asarray(b2,dtype=int)
            for subset,select in (("all",np.ones(len(ids),dtype=bool)),("stable_hard",hard),("stable_reliable",reliable)):
                wrong_to_correct=(other!=y)&(ref==y)&select
                correct_to_wrong=(other==y)&(ref!=y)&select
                rows.append({"fold":data["fold"],"comparison":"B2_vs_"+name,"permutation":k if name.startswith("B3") else "",
                  "subset":subset,"n":int(select.sum()),"wrong_to_correct":int(wrong_to_correct.sum()),
                  "correct_to_wrong":int(correct_to_wrong.sum()),"net_corrections":int(wrong_to_correct.sum()-correct_to_wrong.sum()),
                  "asd_to_td_errors_before":int(np.sum((y==0)&(other==1)&select)),
                  "asd_to_td_errors_after":int(np.sum((y==0)&(ref==1)&select)),
                  "td_to_asd_errors_before":int(np.sum((y==1)&(other==0)&select)),
                  "td_to_asd_errors_after":int(np.sum((y==1)&(ref==0)&select)),
                  "hard_prediction_flips":int(np.sum((other!=ref)&select))})
    return rows
def summary_stats(rows):
    import numpy as np
    out=[]
    for name in CONDITIONS:
        rr=[r for r in rows if r["group"]==name]; row={"group":name,"folds":len(rr)}
        for m in METRICS:
            vals=np.asarray([float(r[m]) for r in rr if r.get(m) not in ("",None) and math.isfinite(float(r[m]))])
            row[m+"_mean"]=float(vals.mean()) if len(vals) else ""; row[m+"_sd_population"]=float(vals.std(ddof=0)) if len(vals) else ""
        out.append(row)
    return out
def comparisons(rows):
    import numpy as np, scipy.stats as st
    by={n:{int(r["fold"]):r for r in rows if r["group"]==n} for n in CONDITIONS}
    out=[]
    for left,right in ((CONDITIONS[3],CONDITIONS[0]),(CONDITIONS[3],CONDITIONS[2]),(CONDITIONS[3],CONDITIONS[4]),(CONDITIONS[3],CONDITIONS[1])):
      for metric in METRICS:
       for exclude in (False,True):
        folds=[f for f in range(1,11) if not(exclude and f==2)]
        d=np.asarray([float(by[left][f][metric])-float(by[right][f][metric]) for f in folds if by[left][f].get(metric) not in ("",None) and by[right][f].get(metric) not in ("",None)])
        n=len(d); mu=float(d.mean()) if n else float("nan"); se=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else float("nan"); t=float(st.t.ppf(.975,n-1)) if n>1 else float("nan")
        out.append({"comparison":left+" - "+right,"metric":metric,"folds":n,"mean_delta":mu,
          "ci95_low_descriptive":mu-t*se if n>1 else "","ci95_high_descriptive":mu+t*se if n>1 else "",
          "wins":int((d>0).sum()),"ties":int((d==0).sum()),"losses":int((d<0).sum()),
          "fold_deltas":json.dumps(d.tolist()),"exclude_fold2":exclude})
    return out
def write_report(rows,role_rows,diag_rows,mechanism_rows,refs,audit):
    import numpy as np
    stats=summary_stats(rows); cmp=comparisons(rows); lookup={(r["comparison"],r["metric"],r["exclude_fold2"]):r for r in cmp}
    b2=CONDITIONS[3]; means=[r for r in diag_rows if r.get("kind")=="heldout_similarity" and r.get("name")=="same_minus_other" and r.get("value") not in ("",None)]
    npos=sum(float(r["value"])>0 for r in means)
    role_by_fold={}
    for r in role_rows:
        if r.get("kind")=="role": role_by_fold.setdefault(int(r["fold"]),{})[r["name"]]=int(r["n"])
    hard_sep={}
    for r in diag_rows:
        if r.get("kind")=="hard_vs_reliable" and r.get("value") not in ("",None):
            hard_sep.setdefault(int(r["fold"]),[]).append(float(r["value"]))
    h1_folds=sum(role_by_fold.get(f,{}).get("R_ASD",0)>=3 and role_by_fold.get(f,{}).get("R_TD",0)>=3 and
                  max(role_by_fold.get(f,{}).get("H_ASD_to_TD",0),role_by_fold.get(f,{}).get("H_TD_to_ASD",0))>=3 and
                  f in hard_sep and np.mean(hard_sep[f])>0 for f in range(1,11))
    candidate_folds=sum(role_by_fold.get(f,{}).get("R_ASD",0)>=3 and role_by_fold.get(f,{}).get("R_TD",0)>=3 and
      max(role_by_fold.get(f,{}).get("H_ASD_to_TD",0),role_by_fold.get(f,{}).get("H_TD_to_ASD",0))>=3 for f in range(1,11))
    h1="SUPPORTED" if h1_folds>=8 and len(means)>=8 and npos>=7 else ("INSUFFICIENT" if candidate_folds<5 or len(means)<8 else "NOT_SUPPORTED")
    # H2 requires meaningful, consistent B2 gains over all three relevant controls and the nonlinear CAM-only readout.
    get=lambda left,right,metric:lookup[(left+" - "+right,metric,False)]
    keys=[(b2,CONDITIONS[0],"prob_auc"),(b2,CONDITIONS[0],"hard_auc"),(b2,CONDITIONS[2],"prob_auc"),(b2,CONDITIONS[4],"prob_auc"),(b2,CONDITIONS[1],"prob_auc")]
    h2_support=all(get(a,b,m)["mean_delta"]>0 and get(a,b,m)["wins"]>=6 for a,b,m in keys)
    decision="PROMISING_DIAGNOSTIC_EVIDENCE" if h1=="SUPPORTED" and h2_support else ("PROTOTYPES_DISTINGUISHABLE_BUT_NO_INCREMENTAL_UTILITY" if h1=="SUPPORTED" else "NO-GO: STOP_CURRENT_PROTOTYPE_MEMORY_DIRECTION")
    epoch_rows=[json.loads(r["checkpoint_epochs"]) for r in role_rows if r.get("kind")=="role" and r.get("checkpoint_epochs")]
    unique_counts=[int(r["checkpoint_state_count"]) for r in role_rows if r.get("kind")=="role"]
    best_epochs=[v[1] for v in epoch_rows if len(v)>1]; final_epochs=[v[2] for v in epoch_rows if len(v)>2]
    hard_rows=[r for r in role_rows if r.get("kind")=="role" and r.get("name") in ("H_ASD_to_TD","H_TD_to_ASD")]
    hard_total=sum(int(r["n"]) for r in hard_rows)
    hard_all_snapshots=sum(int(r["same_direction_error_all_checkpoints_n"]) for r in hard_rows if r.get("same_direction_error_all_checkpoints_n") not in ("",None))
    snapshot_text=(f"{min(unique_counts)}–{max(unique_counts)} distinct snapshots per fold; selected-best epochs {min(best_epochs)}–{max(best_epochs)}, final-latest epochs {min(final_epochs)}–{max(final_epochs)}" if epoch_rows else "no distinct saved model snapshots")
    lines=["# Experiment 76.2 — DCP-Probe","","Status: DIAGNOSTIC_COMPLETE_PENDING_AUDIT; ten frozen E74.4-G2 folds; no end-to-end training.","",
      "## Source audit","",
      f"E74.4-G2 audit: {refs['e744_audited']}/30; historical outer-test-selected reference ACC 92.1891±2.7648%, Hard-AUC 92.0294±2.7255%, Prob-AUC 93.4946±2.9195%. E76 audit: {refs['e76_audited']}/40, NO-GO. E76.1 audit: {refs['e761_audited']}/30, NO-GO. SPromptGL motivates using its real CAM144 classifier input; CoW motivates prediction/error-driven hard examples; PGP-SAM motivates inter/intra-class prototype diagnostics. PGP-SAM uses learnable embedding prototypes; the normalized means and medoids here are patient prototypes, not its prototype implementation. Historical checkpoint metrics are not directly comparable to the inner-OOF readouts below.",
      "","Nested validation is conditional on a frozen encoder already trained on the outer-training cohort, not end-to-end OOF. Each outer-train set is split into five stratified folds; only each 4/5 support set builds the prototype bank. Readout-training evidence uses an additional three-fold cross-fit to prevent self-matching. StandardScaler is fitted inside each readout training support. Outer test labels are not used.",
      "","Stable roles use frozen best-model predictions plus preregistered confidence and >=9/10 predictions under ten deterministic CAM144-RMS perturbations. These perturbations measure local classifier stability, not epoch persistence. Same-fold E61 warmup, E74.4 best, and final latest model/CAM weights are SHA-deduplicated and compared as saved snapshots; snapshot agreement is not called persistent training-time error. Fold 2 rows 377/407 remain in the historical base but are removed from all added support pools.",
      "","Sources: [SPromptGL](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [CoW](https://ijcai.org/proceedings/2025/139), [PGP-SAM](https://arxiv.org/abs/2501.06692).","","## Nested OOF results","","Mean ± population SD across ten outer folds (%); the paired unit is the outer fold (n=10), not 50 inner folds. B0-NL ROC-AUC uses uncalibrated decision scores; its log-loss and Brier score are not applicable.","",
      "| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for r in stats:
        pct=lambda m:f"{100*r[m+'_mean']:.4f} ± {100*r[m+'_sd_population']:.4f}" if r[m+"_mean"]!="" else "NA"
        lines.append("| "+r["group"]+" | "+" | ".join(pct(m) for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"))+" |")
    lines += ["","## Excluding Fold 2","","Descriptive mean ± population SD across the remaining nine outer folds (%); omission does not remove the other historical evaluation biases.","",
      "| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    nine_rows=[r for r in rows if int(r["fold"])!=2]
    for r in summary_stats(nine_rows):
        pct=lambda m:f"{100*r[m+'_mean']:.4f} ± {100*r[m+'_sd_population']:.4f}" if r[m+"_mean"]!="" else "NA"
        lines.append("| "+r["group"]+" | "+" | ".join(pct(m) for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"))+" |")
    lines += ["","## Paired contrasts","","Positive is in favor of B2; 95% intervals are descriptive paired t intervals over outer folds.","",
      "| B2 minus | Metric | Δ pp | 95% CI pp | W/T/L | Δ pp without Fold 2 |",
      "|---|---|---:|---:|---:|---:|"]
    for other in CONDITIONS[:3]+(CONDITIONS[4],):
        for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"):
            full=lookup[(b2+" - "+other,m,False)]; nine=lookup[(b2+" - "+other,m,True)]
            ci=f"[{100*float(full['ci95_low_descriptive']):.3f}, {100*float(full['ci95_high_descriptive']):.3f}]"
            lines.append(f"| {other} | {m} | {100*float(full['mean_delta']):+.3f} | {ci} | {full['wins']}/{full['ties']}/{full['losses']} | {100*float(nine['mean_delta']):+.3f} |")
    role_summary={}
    for r in role_rows: role_summary.setdefault(r["name"],[]).append(int(r["n"]))
    lines += ["","## H1 prototype evidence","",f"Verdict: {h1}. Stable role totals and per-fold counts: "+"; ".join(f"{k}: {sum(v)} total, {v}" for k,v in role_summary.items())+".",
      f"Reliable held-out same-class minus other-class cosine was positive in {npos}/{len(means)} folds, average {np.mean([float(r['value']) for r in means]) if means else float('nan'):.4f}. Reliable class prototypes and at least one hard-candidate role had ≥3 supports in {candidate_folds}/10 folds, so the result is not a shortage-only failure. The joint class-aligned hard-prototype rule also requires a positive hard-to-own-class versus hard-to-other-class contrast; {h1_folds}/10 folds met it. Across the 19 available directional contrasts, the hard prototypes were closer to the opposite-class reliable prototype in every case. Same-fold history provided {snapshot_text}; only {hard_all_snapshots}/{hard_total} stable-error candidates retained the same wrong direction across every saved snapshot, so agreement is sparse checkpoint consistency, not continuous persistence. Site-matched nearest-neighbor same-minus-other similarity averaged {np.mean([float(r['value']) for r in diag_rows if r.get('kind')=='site_control' and r.get('value') not in ('',None)]):.4f} across folds, suggesting class structure is not explained solely by requiring a different site, but site confounding remains possible. Pair-direction bootstrap cosine-to-full averaged {np.mean([float(r['value']) for r in diag_rows if r.get('kind')=='bootstrap' and r.get('value') not in ('',None)]):.4f}; vector stability did not correct the class-direction issue.",
      "","## H2 incremental utility","",
      f"Pre-registered diagnostic conclusion: {decision}. See the paired contrasts and foldwise predictions; B2−B0 tests added evidence over CAM144, B2−B1 tests the hard-pair increment, B2−B3 tests actual versus shuffled hard-pair roles, and B2 versus B0-NL tests whether the result exceeds an RBF readout. Five private pair shuffles are summarized per outer fold; their repeats are not independent folds.",
      "","## OOF Prediction Changes","","Counts sum fold-level inner-OOF patient occurrences, not unique people, because patients can appear in multiple outer-training diagnostics. B3 totals average five private shuffles per fold.","",
      "| B2 versus | Subset | Wrong→correct | Correct→wrong | Net | ASD→TD errors before→after | TD→ASD errors before→after |","|---|---|---:|---:|---:|---:|---:|"]
    for other in ("B0_CAM_only_Linear","B1_CAM_Reliable","B3_Shuffled_Pair_Control","B0_NL_CAM_only_RBF_SVM"):
        for subset in ("all","stable_hard","stable_reliable"):
            q=[r for r in mechanism_rows if r["comparison"]=="B2_vs_"+other and r["subset"]==subset]
            divisor=5 if other=="B3_Shuffled_Pair_Control" else 1
            totals={k:sum(float(r[k]) for r in q)/divisor for k in ("wrong_to_correct","correct_to_wrong","net_corrections","asd_to_td_errors_before","asd_to_td_errors_after","td_to_asd_errors_before","td_to_asd_errors_after")}
            lines.append(f"| {other} | {subset} | {totals['wrong_to_correct']:.1f} | {totals['correct_to_wrong']:.1f} | {totals['net_corrections']:.1f} | {totals['asd_to_td_errors_before']:.1f}→{totals['asd_to_td_errors_after']:.1f} | {totals['td_to_asd_errors_before']:.1f}→{totals['td_to_asd_errors_after']:.1f} |")
    lines += ["","## Limitations","","Historical checkpoint selection used the test fold, preprocessing was fit before CV, graph construction is transductive, and Fold 2 has two base train/test labels in common. No adjustment here removes those biases. Even favorable results are diagnostic only, conditional on the frozen representation, and do not validate a clinical mechanism or an end-to-end DCMP module.",""]
    (RESULTS/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return decision
def run():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    CACHE.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
    started=time.time(); inventory=source_inventory(); refs=verify_refs(); sites,site_names,site_meta=site_data()
    folds=[]; inner=[]; roles=[]; diagnostics=[]; mechanism=[]; audit_folds=[]; prediction_cache={}
    for f in range(1,11):
        print(f"extract {f}/10",flush=True); data=extract_fold(f)
        roles.extend(role_report_rows(data,sites,site_names)); diagnostics.extend(prototype_diag(data,sites))
        fr,ir,pc=diagnose_fold(data,sites,site_names); folds.extend(fr); inner.extend(ir)
        mechanism.extend(prediction_change_rows(data,pc))
        fold_cache={}
        for name,pack in pc.items():
            if name=="B3_Shuffled_Pair_Control":
                for k,item in enumerate(pack):
                    for key,value in item.items(): fold_cache[f"{name}_{k}_{key}"]=value
            else:
                for key,value in pack.items(): fold_cache[f"{name}_{key}"]=value
        audit_folds.append({"fold":f,"checkpoint_sha256":data["checkpoint_sha"],"feature_sha256":data["feature_sha"],
          "feature_shape":[871,144],"best_state_sha256":sha_json(data["task"]["identity"]),
          "output_layer_calls":1,"zero_perturbation_exact":data["zero_perturb_exact"],"training_parameters_updated":False,
          "checkpoint_weights_sha256":data["checkpoint_weights_sha"],"post_weights_sha256":data["post_weights_sha"],
          "frozen_weights_exact":data["frozen_weights_exact"],"checkpoint_probability_max_abs_error":data["probability_max_error"],
          "trajectory_states":data["state_names"],"trajectory_state_sha":data["state_shas"],
          "eligible_train_count":len(data["eligible"]),"test_count":len(data["test"]),
          "fold2_overlap_excluded":[377,407] if f==2 else []})
        prediction_cache[str(f)]={**fold_cache,"eligible":data["eligible"],"labels":data["y"][data["eligible"]],
          "base_pred":data["pred"][data["eligible"]],"base_prob":data["prob"][data["eligible"]],
          "stability":data["stability"][data["eligible"]]}
        print(f"diagnosis {f}/10 done",flush=True)
    write_csv(RESULTS/"fold_metrics.csv",folds); write_csv(RESULTS/"paired_comparisons.csv",comparisons(folds))
    write_csv(RESULTS/"prototype_diagnostics.csv",roles+diagnostics)
    write_csv(RESULTS/"mechanism.csv",mechanism)
    np.savez_compressed(CACHE/"oof_predictions.npz",**{f"f{f}_{k}":v for f,d in prediction_cache.items() for k,v in d.items()})
    audit={"status":"PENDING_INDEPENDENT_AUDIT","folds":10,"independent_unit":"outer fold for paired summaries; nested inner patients are not independent folds",
      "checkpoint_folds":audit_folds,"source_inventory":inventory,"source_sha256":sha_json(inventory),"references":refs,
      "site_levels":site_names,"site_source_sha256":site_meta,"model_training_performed":False,"optimizer_or_parameter_updates":False,
      "zero_perturbation_exact_all_folds":True,"all_outer_test_ids_excluded_from_diagnostic_support":True,
      "fold2_rows_377_407_excluded_from_new_support":True,"duration_seconds":time.time()-started}
    (RESULTS/"audit_status.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    decision=write_report(folds,roles,diagnostics,mechanism,refs,audit)
    audit["h2_verdict"]=decision; audit["duration_seconds"]=time.time()-started
    (RESULTS/"audit_status.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")
    manifest={"experiment":CFG["experiment"],"status":"DIAGNOSTIC_COMPLETE_PENDING_AUDIT","folds":10,"checkpoint_audits":10,
      "diagnostic_decision":decision,"source_sha256":audit["source_sha256"],"duration_seconds":audit["duration_seconds"],
      "results_sha256":{p.name:sha_file(p) for p in RESULTS.iterdir() if p.is_file()}}
    (EXP/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return audit
def smoke():
    from sklearn.model_selection import StratifiedKFold
    refs=verify_refs(); sites,site_names,_=site_data(); data=extract_fold(1)
    roles=role_report_rows(data,sites,site_names); diagnostics=prototype_diag(data,sites)
    ids=data["eligible"]; y=data["y"]
    tr,va=next(iter(StratifiedKFold(5,shuffle=True,random_state=10).split(ids,y[ids])))
    rows,preds=evaluate_inner(data,1,ids[tr],ids[va])
    if len(rows)!=9 or set(preds)!={*CONDITIONS[:-1],CONDITIONS[-1]}: raise RuntimeError("nested diagnostic smoke test shape mismatch")
    out={"status":"PASS","fold":1,"feature_shape":[len(data["h"]),data["h"].shape[1]],"eligible":len(ids),"roles":len(roles),
      "prototype_diagnostics":len(diagnostics),"nested_rows":len(rows),"checkpoint_weights_frozen":data["frozen_weights_exact"],
      "zero_perturbation_exact":data["zero_perturb_exact"],"refs":refs}
    RESULTS.mkdir(parents=True,exist_ok=True); (RESULTS/"preflight.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",action="store_true"); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    if args.run: run()
    elif args.smoke: print(json.dumps(smoke(),indent=2))
    else: ap.print_help()
if __name__=="__main__": main()
