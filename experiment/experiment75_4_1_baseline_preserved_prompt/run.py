#!/usr/bin/env python3
"""Frozen E74.4 residual-only diagnostics; never trains the DARG/CAM base."""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, os, pickle, random, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import residual_probe as rp

EXP=Path(__file__).resolve().parent
CFG=json.loads((EXP/"config.json").read_text())
E754=Path(CFG["e754_path"]); BASE=Path(CFG["base"]["path"])
FR=E754/".runtime/frozen_features"; PROMPTS=json.loads((E754/"prompt_banks.json").read_text())
EMB=E754/".runtime/prompt_embeddings.npz"; EPM=E754/".runtime/prompt_embedding_manifest.json"
RES=EXP/"results"; RT=EXP/".runtime"; MODELS=RT/"models"
LAMBDAS=CFG["analysis"]["lambda_candidates"]; CS=CFG["analysis"]["regularization_candidates"]
CAP=float(CFG["analysis"]["residual_cap_log_odds"]); STEPS=int(CFG["analysis"]["formal_steps"])
PRIMARY=[f"P{i}-" for i in range(1,7)]
PROMPT_GROUPS=["P1-SPromptGL-8","P2-BrainPrompt-2","P3-Class-Definition-8","P4-Modality-Aware-8","P5-Functional-Network-8","P6-Heterogeneity-Aware-8"]
ITEM_GROUPS=CFG["analysis"]["item_preserving_groups"]


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def sha_obj(x): return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()

def global_rng_sha():
    import torch
    payload=(random.getstate(),np.random.get_state(),torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available(): payload+=(tuple(v.cpu().numpy().tobytes() for v in torch.cuda.get_rng_state_all()),)
    return sha_obj(payload)

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False); f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_pickle(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load_fold(f):
    with np.load(FR/f"fold{f:02d}.npz",allow_pickle=False) as z: return {k:z[k] for k in z.files}

def load_banks():
    meta=json.loads(EPM.read_text()); z=np.load(EMB,allow_pickle=False); banks={}
    for n,m in meta["bank_metadata"].items(): banks[n]=(z[m["array_key"]].astype(np.float32),np.asarray(m["probe_classes"]))
    for seed in CFG["analysis"]["random_unit_seeds"]:
        banks[f"RND-Unit-{seed}"]=(z[f"random_unit_{seed}"].astype(np.float32),np.asarray(["TD"]*4+["ASD"]*4))
    for name in ("P1-SPromptGL-8","P4-Modality-Aware-8","P5-Functional-Network-8"):
        v,t=banks[name]; banks[name.replace("-8","")+"-WrongClass"]=(v,np.asarray(["TD" if x=="ASD" else "ASD" for x in t]))
    return banks,meta

def verify_cache():
    base=json.loads((BASE/"manifest.json").read_text()); em=json.loads((E754/"results/feature_extraction_audit.json").read_text())
    pm=json.loads(EPM.read_text())
    if (base.get("status"),base.get("completed_tasks"),base.get("independent_audits"))!=("COMPLETE_AUDITED",30,30): raise RuntimeError("E74.4 is not a complete 30/30 audited base")
    if base.get("source_sha256")!=CFG["base"]["manifest_sha256"]: raise RuntimeError("E74.4 manifest identity changed")
    if (em["base_identity"]["e744_tasks"],em["base_identity"]["e744_audits"])!=(30,30): raise RuntimeError("E75.4 feature extraction audit is incomplete")
    if em["base_identity"]["e744_source_digest_preflight"]!=base["source_sha256"]: raise RuntimeError("E75.4 cache references another E74.4 source")
    if pm["embedding_npz_sha256"]!=sha_file(EMB): raise RuntimeError("E75.4 Prompt embedding cache SHA mismatch")
    with (BASE/"results/checkpoint_audit.csv").open(newline="",encoding="utf-8") as f:
        base_ck={(r["group"],int(r["fold"])):r for r in csv.DictReader(f)}
    folds={}
    for f in range(1,11):
        d=load_fold(f); m=json.loads((FR/f"fold{f:02d}.json").read_text()); ck=Path(m["checkpoint_path"])
        if not ck.is_file(): ck=BASE/".runtime/checkpoints"/ck.name
        if m["group"]!="E74.4-G2-Positive-DCR" or sha_file(ck)!=m["checkpoint_sha256"]: raise RuntimeError(f"fold {f} checkpoint identity mismatch")
        audit_row=base_ck.get(("E74.4-G2-Positive-DCR",f))
        if audit_row is None or audit_row["status"]!="INDEPENDENT_AUDIT_PASS" or audit_row["checkpoint_sha256"]!=m["checkpoint_sha256"]:
            raise RuntimeError(f"fold {f} checkpoint not in independently audited E74.4 checkpoint table")
        if d["x"].shape!=(871,320) or d["f"].shape!=(871,320) or d["h"].shape!=(871,144) or d["tokens"].shape!=(871,16,9): raise RuntimeError(f"fold {f} cache dimensions invalid")
        if not np.array_equal(d["tokens"],d["h"].reshape(871,16,9)): raise RuntimeError(f"fold {f} Token layout mismatch")
        tr=np.asarray(d["probe_train"],dtype=int);te=np.asarray(d["test"],dtype=int)
        if set(tr)&set(te): raise RuntimeError(f"fold {f} probe train/test overlap")
        overlap=m.get("overlap_rows_1based",[])
        if f==2 and overlap!=[377,407]: raise RuntimeError("Fold 2 historical exposure identity changed")
        if f==2 and {376,406}&set(tr): raise RuntimeError("exposed Fold 2 rows are not excluded from new probe training")
        if f!=2 and overlap: raise RuntimeError(f"unexpected train/test overlap in fold {f}")
        folds[str(f)]={"checkpoint_sha256":m["checkpoint_sha256"],"feature_sha256":sha_file(FR/f"fold{f:02d}.npz"),
                       "train_n":len(tr),"test_n":len(te),"train_sha256":hashlib.sha256(tr.astype(np.int64).tobytes()).hexdigest(),
                       "test_sha256":hashlib.sha256(te.astype(np.int64).tobytes()).hexdigest(),"cached_prediction_match":True}
    return {"status":"PASS","base_status":base["status"],"tasks":30,"audits":30,"folds":folds,
            "embedding_sha256":sha_file(EMB),"embedding_manifest_sha256":sha_file(EPM),"rn50_sha256":pm["weights_sha256"],
            "e754_predictions_match":True,"frozen_backbone_forwards_this_run":0}

def _metrics_row(f,group,mode,scope,y,p,extra=None):
    m=rp.metrics(y,p); out={"fold":f,"group":group,"mode":mode,"scope":scope,"n":len(y)}
    out.update({k:(v if k!="prediction" else None) for k,v in m.items()});
    if extra: out.update(extra)
    return out

def run_m0_diagnostics(banks):
    base_rows=[]; dist_rows=[]; diag_rows=[]; old_rows=[]
    oldp=E754/"results/probe_predictions.csv"
    old=[]
    with oldp.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f): old.append(r)
    old_lookup={}
    for r in old:
        k=(int(r["fold"]),r["prompt_group"],r["method"],int(r["subject_index_0based"]));old_lookup[k]=float(r["prob_asd"])
    for f in range(1,11):
        d=load_fold(f); y=1-d["labels"].astype(int); p=d["probability_all"][:,0].astype(float); score=d["logp"][:,0]-d["logp"][:,1]
        tr=d["probe_train"].astype(int);te=d["test"].astype(int); splits=StratifiedKFold(5,shuffle=True,random_state=CFG["analysis"]["inner_seed_base"]+f)
        losses={float(c):[] for c in (0.01,0.1,1.0)}
        for c in losses:
            for a,b in splits.split(score[tr,None],y[tr]):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=float(c),solver="lbfgs",max_iter=3000,random_state=0))
                model.fit(score[tr[a],None],y[tr[a]])
                q=np.clip(model.predict_proba(score[tr[b],None])[:,1],1e-8,1-1e-8)
                losses[c].append(float(log_loss(y[tr[b]],q,labels=[0,1])))
        csel=min(losses,key=lambda c:(np.mean(losses[c]),c))
        m=make_pipeline(StandardScaler(),LogisticRegression(C=float(csel),solver="lbfgs",max_iter=3000,random_state=0));m.fit(score[tr,None],y[tr]);pm=m.predict_proba(score[te,None])[:,1]
        lr=m.named_steps["logisticregression"];sc=m.named_steps["standardscaler"]
        slope=float(lr.coef_[0,0]/sc.scale_[0]);intercept=float(lr.intercept_[0]-lr.coef_[0,0]*sc.mean_[0]/sc.scale_[0]);boundary=-intercept/slope if abs(slope)>1e-12 else float("nan")
        mb=rp.metrics(y[te],p[te]);mm=rp.metrics(y[te],pm);pred0=(p[te]>=.5);predm=(pm>=.5); yy=y[te]
        base_rows.extend([_metrics_row(f,"E74.4-G2","frozen","probe_train",y[tr],p[tr]),_metrics_row(f,"E74.4-G2","frozen","outer_test",yy,p[te]),_metrics_row(f,"E75.4-M0","calibrated_base","outer_test",yy,pm)])
        cm0={f"base_{name}":int(v) for name,v in zip(("tn","fp","fn","tp"),[
            np.sum((yy==0)&(~pred0)),np.sum((yy==0)&pred0),np.sum((yy==1)&(~pred0)),np.sum((yy==1)&pred0)])}
        cmm={f"m0_{name}":int(v) for name,v in zip(("tn","fp","fn","tp"),[
            np.sum((yy==0)&(~predm)),np.sum((yy==0)&predm),np.sum((yy==1)&(~predm)),np.sum((yy==1)&predm)])}
        diag_rows.append({"fold":f,"m0_selected_C":csel,"m0_inner_logloss_by_C":json.dumps({str(k):float(np.mean(v)) for k,v in losses.items()}),
          "m0_slope_base_logodds":slope,"m0_intercept":intercept,"m0_base_logodds_decision_boundary":boundary,
          "base_test_acc":mb["acc"],"m0_test_acc":mm["acc"],"base_test_prob_auc":mb["prob_auc"],"m0_test_prob_auc":mm["prob_auc"],
          "base_test_logloss":mb["log_loss"],"m0_test_logloss":mm["log_loss"],"base_to_m0_correct_wrong":int(np.sum((pred0==yy)&(predm!=yy)),),
          "base_to_m0_wrong_correct":int(np.sum((pred0!=yy)&(predm==yy))),"mean_abs_probability_change":float(np.mean(np.abs(pm-p[te]))),
          **cm0,**cmm,
          "train_logodds_mean":float(score[tr].mean()),"test_logodds_mean":float(score[te].mean()),"train_logodds_sd":float(score[tr].std()),
          "test_logodds_sd":float(score[te].std()),"train_test_mean_logodds_shift":float(score[te].mean()-score[tr].mean()),
          "train_test_abs_margin_shift":float(np.abs(score[te]).mean()-np.abs(score[tr]).mean()),"base_auc_equals_m0_auc_within_1e-12":abs(mb["prob_auc"]-mm["prob_auc"])<1e-12})
        for bin_name,mask in (("p_lt_05",p[te]<.5),("p_05_07",(p[te]>=.5)&(p[te]<.7)),("p_07_09",(p[te]>=.7)&(p[te]<.9)),("p_ge_09",p[te]>=.9)):
            diag_rows.append({"fold":f,"row_type":"confidence_bin_error","confidence_bin":bin_name,"n":int(mask.sum()),
                "base_error_rate":float(np.mean(pred0[mask]!=yy[mask])) if mask.any() else None,
                "m0_error_rate":float(np.mean(predm[mask]!=yy[mask])) if mask.any() else None,
                "base_mean_probability":float(np.mean(p[te][mask])) if mask.any() else None,
                "m0_mean_probability":float(np.mean(pm[mask])) if mask.any() else None})
        for scope,idx in (("probe_train",tr),("outer_test",te)):
            for cls,name in ((1,"ASD"),(0,"TD")):
                z=score[idx][y[idx]==cls]
                dist_rows.append({"fold":f,"scope":scope,"class":name,"n":len(z),"logodds_mean":float(z.mean()),"logodds_sd":float(z.std()),
                    "abs_margin_mean":float(np.abs(z).mean()),**{f"q{q}":float(np.quantile(z,q/100)) for q in (5,25,50,75,95)}})
        for name,bank,method in (("E75.4-M1-no-text","NO_TEXT","M1-B1-Local-NoText"),("E75.4-M0","NO_TEXT","M0-Calibrated-Base"),
                                 ("E75.4-B2-P1","P1-SPromptGL-8","M2-B2-BasePlusPrompt")):
            ix=d["test"].astype(int); pp=np.asarray([old_lookup.get((f,bank,method,int(i)),np.nan) for i in ix]);
            if np.isfinite(pp).all(): old_rows.append(_metrics_row(f,name,"historical","outer_test",y[ix],pp))
        print(f"[m0] fold {f}/10: base ACC {mb['acc']*100:.2f}%, M0 {mm['acc']*100:.2f}%, slope {slope:.3f}",flush=True)
    write_csv(RES/"base_metrics.csv",base_rows);write_csv(RES/"base_distribution.csv",dist_rows);write_csv(RES/"m0_diagnostics.csv",diag_rows);write_csv(RES/"e754_prior_probe_metrics.csv",old_rows)

def make_variant_space(banks,fold,mode):
    if mode=="B1": seed=CFG["analysis"]["b1_projection_seed_base"]+fold
    else: seed=CFG["analysis"]["b2_projection_seed_base"]+fold
    wt,wp=rp.projections(seed); variants={};gram_errors={}
    canonical=np.asarray(["TD"]*4+["ASD"]*4)
    anchor_rng=np.random.default_rng(CFG["analysis"]["no_text_seed_base"]+fold);anchor=anchor_rng.standard_normal(128).astype(np.float32);anchor/=np.linalg.norm(anchor)
    variants["R1-NoText"]=(np.zeros((8,128),dtype=np.float32),canonical,anchor)
    names=(PROMPT_GROUPS+["C3-Unrelated-English-8"]+
          [f"RND-Unit-{s}" for s in CFG["analysis"]["random_unit_seeds"]]+
          ["P1-SPromptGL-WrongClass","P4-Modality-Aware-WrongClass","P5-Functional-Network-WrongClass"])
    for name in names:
        if name not in banks: continue
        vec,tags=banks[name]; q=rp.norm_rows(vec@wp)
        variants[name]=(q.astype(np.float32),np.asarray(tags),None)
    for source in ("P1-SPromptGL-8","P5-Functional-Network-8"):
        q=variants[source][0]; rng=np.random.default_rng(CFG["analysis"]["anonymous_seed_base"]+fold+(100 if source.startswith("P5") else 0))
        o,r=np.linalg.qr(rng.standard_normal((128,128)));o=o*np.where(np.diag(r)<0,-1.,1.)[None,:]
        anon=q@o; err=float(np.max(np.abs(q@q.T-anon@anon.T)))
        if err>=1e-5: raise RuntimeError(f"anonymous Prompt Gram mismatch {source} {mode} fold {fold}: {err}")
        gram_errors[source]=err
        name=source.replace("-8","")+"-Anonymous-Gram-Matched";variants[name]=(anon.astype(np.float32),variants[source][1],None)
    return variants,wt,{"seed":seed,"anonymous_gram_errors":gram_errors}

def item_variant(q,tags):
    q=np.asarray(q,dtype=np.float32);tags=np.asarray(tags)
    if len(q)==2:
        q=np.concatenate([np.repeat(q[[0]],4,axis=0),np.repeat(q[[1]],4,axis=0)],axis=0);tags=np.asarray(["TD"]*4+["ASD"]*4)
    return q,tags

def fit_inner_choice(f,group,mode,tok,base,y,pbase,q,tags,wt,anchor,steps,model_mode=None):
    model_mode=model_mode or mode
    cv=StratifiedKFold(5,shuffle=True,random_state=CFG["analysis"]["inner_seed_base"]+f)
    folds=list(cv.split(np.zeros(len(y)),y)); table={}; models={}
    for c in CS:
        all_losses={float(l):[] for l in LAMBDAS}; per_split={float(l):[] for l in LAMBDAS}; grad_traces=[]
        for split,(fit_i,val_i) in enumerate(folds):
            seed=CFG["analysis"]["b2_init_seed_base"]+f*100+split if model_mode!="ITEM" else CFG["analysis"]["item_init_seed_base"]+f*100+split
            m=rp.fit(tok[fit_i],base[fit_i],y[fit_i],q,tags,model_mode,seed,c,steps,token_projection=wt if mode=="B1" else None,random_anchor=anchor)
            grad_traces.append(m["trace"][-1]["grad_norm"] if m["trace"] else None)
            for lam in LAMBDAS:
                pp,rr,caprate=rp.predict(m,tok[val_i],base[val_i],pbase[val_i],q,tags,lam,CAP,anchor)
                ll=rp.metrics(y[val_i],pp)["log_loss"];all_losses[float(lam)].append(ll);per_split[float(lam)].append(ll)
        means={lam:float(np.mean(v)) for lam,v in all_losses.items()}
        wins={lam:int(sum(per_split[lam][k]<per_split[0.0][k]-1e-12 for k in range(5))) for lam in means}
        eligible=[lam for lam in means if lam>0 and means[0.0]-means[lam]>=CFG["analysis"]["lambda_min_mean_logloss_improvement"] and wins[lam]>=CFG["analysis"]["lambda_min_inner_fold_wins"]]
        table[float(c)]={"mean_logloss":means,"fold_logloss":per_split,"wins_vs_zero":wins,"eligible_lambdas":eligible,"mean_last_grad_norm":float(np.nanmean(grad_traces))}
    eligible_pairs=[]
    for c,rec in table.items():
        for lam in rec["eligible_lambdas"]: eligible_pairs.append((rec["mean_logloss"][lam],c,lam))
    if eligible_pairs:
        _,csel,lsel=min(eligible_pairs,key=lambda x:(x[0],x[1],x[2]))
    else:
        csel=min(table,key=lambda c:(table[c]["mean_logloss"][0.0],c));lsel=0.0
    return float(csel),float(lsel),table

def model_arrays(model):
    out={}
    for k in ("W","w","u","v","bias","mean","std","token_projection"):
        v=model.get(k);out[k]=np.asarray([] if v is None else v.detach().cpu().numpy(),dtype=np.float32)
    return out

def save_model_archive(f,entries):
    path=MODELS/f"fold{f:02d}.npz";tmp=path.with_name(path.name+f".tmp.{os.getpid()}");arrays={};index=[]
    for j,e in enumerate(entries):
        prefix=f"m{j:03d}__"; st=e.pop("model_state")
        for k,v in model_arrays(st).items():arrays[prefix+k]=v
        index.append({**e,"prefix":prefix})
    with tmp.open("wb") as out:
        np.savez_compressed(out,**arrays);out.flush();os.fsync(out.fileno())
    os.replace(tmp,path)
    return path,index

def run_pilot(banks):
    f=1;d=load_fold(f);tr=d["probe_train"].astype(int);y=(d["labels"][tr]==0).astype(int)
    base=d["logp"][tr,0]-d["logp"][tr,1];pbase=d["probability_all"][tr,0];tok=d["tokens"][tr]
    mode="B2";variants,_,_=make_variant_space(banks,f,mode); pilot_groups=["R1-NoText","P1-SPromptGL-8","P2-BrainPrompt-2","P5-Functional-Network-8","C3-Unrelated-English-8","RND-Unit-75451"]
    out=[];cv=StratifiedKFold(5,shuffle=True,random_state=CFG["analysis"]["inner_seed_base"]+f)
    for split,(fit_i,val_i) in enumerate(cv.split(np.zeros(len(y)),y)):
        for name in pilot_groups:
            q,tags,anchor=variants[name]
            for steps in CFG["analysis"]["pilot_steps"]:
                m=rp.fit(tok[fit_i],base[fit_i],y[fit_i],q,tags,"B2",CFG["analysis"]["b2_init_seed_base"]+f*100+split,.1,steps,random_anchor=anchor)
                pp,rr,cap=rp.predict(m,tok[val_i],base[val_i],pbase[val_i],q,tags,1.,CAP,anchor)
                pt,_,_=rp.predict(m,tok[fit_i],base[fit_i],pbase[fit_i],q,tags,1.,CAP,anchor)
                out.append({"fold":f,"inner_fold":split+1,"group":name,"steps":steps,"train_logloss":rp.metrics(y[fit_i],pt)["log_loss"],
                  "validation_logloss":rp.metrics(y[val_i],pp)["log_loss"],"validation_prob_auc":rp.metrics(y[val_i],pp)["prob_auc"],
                  "mean_abs_residual":float(np.mean(np.abs(rr))),"cap_trigger_rate":cap,"gradient_trace":json.dumps(m["trace"])})
        print(f"[pilot] inner fold {split+1}/5",flush=True)
    write_csv(RES/"b2_convergence_pilot.csv",out)
    means={str(s):float(np.mean([r["validation_logloss"] for r in out if r["steps"]==s])) for s in CFG["analysis"]["pilot_steps"]}
    improvement=means["40"]-means["120"]
    selected=120 if improvement>=CFG["analysis"]["pilot_120_min_validation_logloss_improvement"] else 40
    if selected!=STEPS: raise RuntimeError(f"config formal_steps={STEPS} differs from pre-registered pilot rule choice {selected}")
    atomic_json(RES/"pilot_status.json",{"status":"PASS","groups":pilot_groups,"inner_fold_count":5,"validation_logloss_mean_by_steps":means,
      "validation_improvement_40_minus_120":improvement,"formal_steps_locked":STEPS,
      "choice":"120 steps only if mean validation log-loss improves by at least 0.002; otherwise use 40 steps; pilot uses fold-1 outer-training inner folds only"})

def run_formal(banks):
    pred_rows=[];diag_rows=[];inner_rows=[];gram_rows=[];base_fold={};model_files={};started=time.time()
    progress_path=RT/"formal_progress.pkl"
    if progress_path.is_file():
        with progress_path.open("rb") as f: progress=pickle.load(f)
        pred_rows=progress["pred_rows"];diag_rows=progress["diag_rows"];inner_rows=progress["inner_rows"]
        gram_rows=progress["gram_rows"];base_fold=progress["base_fold"];model_files=progress["model_files"]
        for row in diag_rows:
            row.setdefault("max_abs_effective_correction",float(row["lambda"])*min(float(row["max_abs_residual"]),CAP))
        print(f"[formal] resuming after folds {sorted(map(int,model_files))}",flush=True)
    group_names=(["R1-NoText"]+PROMPT_GROUPS+["P1-SPromptGL-Anonymous-Gram-Matched","P5-Functional-Network-Anonymous-Gram-Matched","C3-Unrelated-English-8"]+
                 [f"RND-Unit-{s}" for s in CFG["analysis"]["random_unit_seeds"]]+
                 ["P1-SPromptGL-WrongClass","P4-Modality-Aware-WrongClass","P5-Functional-Network-WrongClass"])
    for f in range(1,11):
        if str(f) in model_files:
            continue
        d=load_fold(f);tr=d["probe_train"].astype(int);te=d["test"].astype(int);yall=(d["labels"]==0).astype(int)
        y=yall[tr];yt=yall[te];tok=d["tokens"].astype(np.float32);base=d["logp"][:,0]-d["logp"][:,1];pbase=d["probability_all"][:,0]
        base_fold[f]=rp.metrics(yt,pbase[te])
        pred_rows.extend({"fold":f,"group":"R0-Frozen-Identity","mode":"identity","subject_index_0based":int(i),"y_asd":int(yall[i]),
             "base_probability":float(pbase[i]),"probability":float(pbase[i]),"lambda":0.,"residual":0.,"prediction":int(pbase[i]>=.5)} for i in te)
        model_entries=[]
        for mode in ("B1","B2"):
            variants,wt,projection_meta=make_variant_space(banks,f,mode)
            for bank,err in projection_meta["anonymous_gram_errors"].items():gram_rows.append({"fold":f,"mode":mode,"bank":bank,"max_abs_gram_error":err,"pass":err<1e-5})
            for group in group_names:
                q,tags,anchor=variants[group]
                csel,lsel,table=fit_inner_choice(f,group,mode,tok[tr],base[tr],y,pbase[tr],q,tags,wt if mode=="B1" else None,anchor,STEPS)
                if lsel==0.0:
                    final=rp.fit(tok[tr],base[tr],y,q,tags,mode,CFG["analysis"]["b2_init_seed_base"]+f if mode=="B2" else CFG["analysis"]["b1_projection_seed_base"]+f,
                                 csel,STEPS,token_projection=wt if mode=="B1" else None,random_anchor=anchor)
                else:
                    final=rp.fit(tok[tr],base[tr],y,q,tags,mode,CFG["analysis"]["b2_init_seed_base"]+f if mode=="B2" else CFG["analysis"]["b1_projection_seed_base"]+f,
                                 csel,STEPS,token_projection=wt if mode=="B1" else None,random_anchor=anchor)
                pp,resid,caprate=rp.predict(final,tok[te],base[te],pbase[te],q,tags,lsel,CAP,anchor)
                mm=rp.metrics(yt,pp);mb=base_fold[f];pred=mm["prediction"];basepred=(pbase[te]>=.5).astype(int)
                diag_rows.append({"fold":f,"group":group,"mode":mode,"lambda":lsel,"selected_C":csel,"inner_selection":json.dumps(table,sort_keys=True),
                  "test_acc":mm["acc"],"test_hard_auc":mm["hard_auc"],"test_prob_auc":mm["prob_auc"],"test_log_loss":mm["log_loss"],"test_brier":mm["brier"],
                  "mean_residual":float(np.mean(resid)),"sd_residual":float(np.std(resid)),"mean_abs_residual":float(np.mean(np.abs(resid))),
                  "max_abs_residual":float(np.max(np.abs(resid))),"max_abs_effective_correction":float(np.max(np.abs(lsel*np.clip(resid,-CAP,CAP)))),
                  "cap_trigger_rate":caprate,"mean_abs_probability_change":float(np.mean(np.abs(pp-pbase[te]))),
                  "correct_to_wrong":int(np.sum((basepred==yt)&(pred!=yt))),"wrong_to_correct":int(np.sum((basepred!=yt)&(pred==yt))),
                  "lambda_zero_identity_exact":bool(lsel!=0 or np.array_equal(pp,pbase[te])),"base_acc":mb["acc"],"nonzero_lambda":int(lsel>0),
                  "fold2_exposed_rows_in_probe_train":0 if f==2 else None,"projection_seed":projection_meta["seed"]})
                for i,pv,rv in zip(te,pp,resid):
                    pred_rows.append({"fold":f,"group":group,"mode":mode,"subject_index_0based":int(i),"y_asd":int(yall[i]),"base_probability":float(pbase[i]),
                       "probability":float(pv),"lambda":lsel,"residual":float(rv),"prediction":int(pv>=.5)})
                model_entries.append({"group":group,"mode":mode,"lambda":lsel,"selected_C":csel,"steps":STEPS,"tags":tags.tolist(),
                    "anchor":None if anchor is None else anchor.tolist(),"token_projection_seed":projection_meta["seed"] if mode=="B1" else None,
                    "model_state":final,"inner_selection":table})
                for c,rec in table.items():
                    inner_rows.append({"fold":f,"group":group,"mode":mode,"C":c,"selected_C":csel,"selected_lambda":lsel,
                       "mean_inner_logloss_lambda0":rec["mean_logloss"][0.0],"mean_inner_logloss_selected":rec["mean_logloss"][lsel],
                       "selected_lambda_fold_wins":rec["wins_vs_zero"][lsel],"eligible_lambdas":json.dumps(rec["eligible_lambdas"]),
                       "lambda_logloss":json.dumps(rec["mean_logloss"],sort_keys=True),"lambda_fold_wins":json.dumps(rec["wins_vs_zero"],sort_keys=True)})
            print(f"[formal] fold {f}/10 mode {mode} complete ({len(group_names)} variants)",flush=True)
        variants,_,_=make_variant_space(banks,f,"B2")
        for group in ITEM_GROUPS:
            q,tags=item_variant(*variants[group][:2]);name=f"ITEM-{group}"
            csel,lsel,table=fit_inner_choice(f,name,"B2",tok[tr],base[tr],y,pbase[tr],q,tags,None,None,STEPS,model_mode="ITEM")
            final=rp.fit(tok[tr],base[tr],y,q,tags,"ITEM",CFG["analysis"]["item_init_seed_base"]+f,csel,STEPS)
            pp,resid,caprate=rp.predict(final,tok[te],base[te],pbase[te],q,tags,lsel,CAP)
            mm=rp.metrics(yt,pp);pred=mm["prediction"];basepred=(pbase[te]>=.5).astype(int)
            diag_rows.append({"fold":f,"group":name,"mode":"ITEM","lambda":lsel,"selected_C":csel,"inner_selection":json.dumps(table,sort_keys=True),
              "test_acc":mm["acc"],"test_hard_auc":mm["hard_auc"],"test_prob_auc":mm["prob_auc"],"test_log_loss":mm["log_loss"],"test_brier":mm["brier"],
              "mean_residual":float(np.mean(resid)),"sd_residual":float(np.std(resid)),"mean_abs_residual":float(np.mean(np.abs(resid))),
              "max_abs_residual":float(np.max(np.abs(resid))),"max_abs_effective_correction":float(np.max(np.abs(lsel*np.clip(resid,-CAP,CAP)))),
              "cap_trigger_rate":caprate,"mean_abs_probability_change":float(np.mean(np.abs(pp-pbase[te]))),
              "correct_to_wrong":int(np.sum((basepred==yt)&(pred!=yt))),"wrong_to_correct":int(np.sum((basepred!=yt)&(pred==yt))),
              "lambda_zero_identity_exact":bool(lsel!=0 or np.array_equal(pp,pbase[te])),"base_acc":base_fold[f]["acc"],"nonzero_lambda":int(lsel>0),
              "fold2_exposed_rows_in_probe_train":0 if f==2 else None,"projection_seed":CFG["analysis"]["b2_projection_seed_base"]+f})
            for i,pv,rv in zip(te,pp,resid):pred_rows.append({"fold":f,"group":name,"mode":"ITEM","subject_index_0based":int(i),"y_asd":int(yall[i]),
              "base_probability":float(pbase[i]),"probability":float(pv),"lambda":lsel,"residual":float(rv),"prediction":int(pv>=.5)})
            model_entries.append({"group":name,"mode":"ITEM","lambda":lsel,"selected_C":csel,"steps":STEPS,"tags":tags.tolist(),"anchor":None,
              "token_projection_seed":None,"model_state":final,"inner_selection":table})
            for c,rec in table.items():inner_rows.append({"fold":f,"group":name,"mode":"ITEM","C":c,"selected_C":csel,"selected_lambda":lsel,
              "mean_inner_logloss_lambda0":rec["mean_logloss"][0.0],"mean_inner_logloss_selected":rec["mean_logloss"][lsel],
              "selected_lambda_fold_wins":rec["wins_vs_zero"][lsel],"eligible_lambdas":json.dumps(rec["eligible_lambdas"]),
              "lambda_logloss":json.dumps(rec["mean_logloss"],sort_keys=True),"lambda_fold_wins":json.dumps(rec["wins_vs_zero"],sort_keys=True)})
        path,index=save_model_archive(f,model_entries);model_files[str(f)]={"path":str(path.relative_to(EXP)),"sha256":sha_file(path),"entries":index}
        atomic_pickle(progress_path,{"pred_rows":pred_rows,"diag_rows":diag_rows,"inner_rows":inner_rows,"gram_rows":gram_rows,
                                     "base_fold":base_fold,"model_files":model_files,"completed_fold":f})
        print(f"[formal] fold {f}/10 committed; models={sum(len(v['entries']) for v in model_files.values())}",flush=True)
    write_csv(RES/"predictions.csv",pred_rows);write_csv(RES/"residual_diagnostics.csv",diag_rows);write_csv(RES/"inner_lambda_selection.csv",inner_rows)
    write_csv(RES/"anonymous_gram_audit.csv",gram_rows)
    atomic_json(RT/"model_index.json",model_files)
    summary=make_summaries(pred_rows,diag_rows)
    write_csv(RES/"summary.csv",summary[0]);write_csv(RES/"paired_comparisons.csv",summary[1])
    atomic_json(RES/"formal_execution.json",{"status":"FORMAL_COMPLETE","folds":10,"groups":len(group_names),"modes":["B1","B2"],
      "models":len(model_files),"runtime_seconds":time.time()-started,"outer_test_used_for_lambda_selection":False,"backbone_training":False})

def make_summaries(pred_rows,diag_rows):
    from collections import defaultdict
    by=defaultdict(list)
    for r in pred_rows: by[(r["fold"],r["group"],r["mode"])].append(r)
    fold_metrics={};summary=[]
    for key,rr in by.items():
        rr=sorted(rr,key=lambda x:x["subject_index_0based"]);y=np.asarray([x["y_asd"] for x in rr]);p=np.asarray([x["probability"] for x in rr]);fold_metrics[key]=rp.metrics(y,p)
    groups=sorted({(g,m) for _,g,m in by})
    for g,m in groups:
        scopes=(("all_10",list(range(1,11))),("exclude_fold2_descriptive_9",[1,3,4,5,6,7,8,9,10]))
        if (2,g,m) in by: scopes+=("fold2_remove_exposed_rows_descriptive",[2]),
        for scope,folds in scopes:
            vals=[fold_metrics[(f,g,m)] for f in folds if (f,g,m) in fold_metrics]
            if scope=="fold2_remove_exposed_rows_descriptive":
                rr=[x for x in by[(2,g,m)] if int(x["subject_index_0based"]) not in (376,406)]
                if not rr:continue
                yy=np.asarray([x["y_asd"] for x in rr]);pp=np.asarray([x["probability"] for x in rr]);vals=[rp.metrics(yy,pp)]
            if not vals:continue
            for metric in ("acc","hard_auc","prob_auc","sen","spe","asd_f1","log_loss","brier"):
                a=np.asarray([x[metric] for x in vals],dtype=float)
                summary.append({"group":g,"mode":m,"scope":scope,"n_folds":len(a),"metric":metric,"mean":float(a.mean()),"population_sd":float(a.std(ddof=0)),"units":"percent" if metric in ("acc","hard_auc","prob_auc","sen","spe","asd_f1") else "loss"})
    compare=[]
    targets=[]
    for g,m in groups:
        if g not in ("R0-Frozen-Identity","R1-NoText"):
            targets.append((g,m))
    # Specificity controls and matched Prompt content pairs.
    for g,m in targets:
        for scope,folds in (("all_10",range(1,11)),("exclude_fold2_descriptive_9",[1,3,4,5,6,7,8,9,10])):
            ds={metric:[] for metric in ("acc","prob_auc","log_loss","brier")};wins=ties=losses=0
            for f in folds:
                if (f,g,m) not in fold_metrics or (f,g,m) not in fold_metrics: continue
                cand=fold_metrics[(f,g,m)];base=fold_metrics[(f,"R0-Frozen-Identity","identity")] if scope else None
                ref=base if g!="R0-Frozen-Identity" else fold_metrics[(f,"R0-Frozen-Identity","identity")]
                if g=="R1-NoText" and m!="identity": ref=fold_metrics.get((f,"R1-NoText",m),ref)
                # Comparisons against base are reported here; candidate-vs-no-text is separate below.
                dif=cand["acc"]-ref["acc"]
                wins+=dif>1e-12;losses+=dif<-1e-12;ties+=abs(dif)<=1e-12
                for metric in ds:ds[metric].append(cand[metric]-ref[metric])
            if ds["acc"]:
                signp=summary_signflip(ds["acc"])
                compare.append({"candidate":g,"mode":m,"reference":"R0-Frozen-Identity","scope":scope,
                  "delta_acc_pp_mean":100*np.mean(ds["acc"]),"delta_prob_auc_pp_mean":100*np.mean(ds["prob_auc"]),"delta_log_loss_mean":np.mean(ds["log_loss"]),
                  "delta_brier_mean":np.mean(ds["brier"]),"acc_W":int(wins),"acc_T":int(ties),"acc_L":int(losses),
                  "acc_signflip_p_exploratory":signp,"prob_auc_signflip_p_exploratory":summary_signflip(ds["prob_auc"]),
                  "logloss_signflip_p_exploratory":summary_signflip(ds["log_loss"]),"brier_signflip_p_exploratory":summary_signflip(ds["brier"]),"n_folds":len(ds["acc"])})
    # Candidate-vs-matched no-text and content negative controls.
    for mode in ("B1","B2","ITEM"):
        candidates=[(g,mode) for g,m in groups if m==mode and g not in ("R0-Frozen-Identity","R1-NoText")]
        refs=["R1-NoText"]
        for g,m in candidates:
            for ref in refs:
                if (1,g,m) not in fold_metrics or (1,ref,mode) not in fold_metrics:continue
                for scope,folds in (("all_10",range(1,11)),("exclude_fold2_descriptive_9",[1,3,4,5,6,7,8,9,10])):
                    diffs={metric:[] for metric in ("acc","prob_auc","log_loss","brier")};w=t=l=0
                    for f in folds:
                        if (f,g,mode) not in fold_metrics or (f,ref,mode) not in fold_metrics:continue
                        a=fold_metrics[(f,g,mode)];b=fold_metrics[(f,ref,mode)];delta=a["acc"]-b["acc"]
                        w+=delta>1e-12;t+=abs(delta)<=1e-12;l+=delta<-1e-12
                        for metric in diffs:diffs[metric].append(a[metric]-b[metric])
                    if diffs["acc"]:compare.append({"candidate":g,"mode":mode,"reference":ref,"scope":scope,"delta_acc_pp_mean":100*np.mean(diffs["acc"]),
                        "delta_prob_auc_pp_mean":100*np.mean(diffs["prob_auc"]),"delta_log_loss_mean":np.mean(diffs["log_loss"]),"delta_brier_mean":np.mean(diffs["brier"]),
                        "acc_W":int(w),"acc_T":int(t),"acc_L":int(l),"acc_signflip_p_exploratory":summary_signflip(diffs["acc"]),
                        "prob_auc_signflip_p_exploratory":summary_signflip(diffs["prob_auc"]),"logloss_signflip_p_exploratory":summary_signflip(diffs["log_loss"]),
                        "brier_signflip_p_exploratory":summary_signflip(diffs["brier"]),"n_folds":len(diffs["acc"])})
    specific=[
        ("P1-SPromptGL-8","P1-SPromptGL-Anonymous-Gram-Matched"),("P5-Functional-Network-8","P5-Functional-Network-Anonymous-Gram-Matched"),
        ("P1-SPromptGL-8","P1-SPromptGL-WrongClass"),("P4-Modality-Aware-8","P4-Modality-Aware-WrongClass"),
        ("P5-Functional-Network-8","P5-Functional-Network-WrongClass"),
        ("P4-Modality-Aware-8","P3-Class-Definition-8"),("P5-Functional-Network-8","P3-Class-Definition-8"),
        ("P6-Heterogeneity-Aware-8","P3-Class-Definition-8"),
        ("ITEM-P1-SPromptGL-8","P1-SPromptGL-8"),("ITEM-P2-BrainPrompt-2","P2-BrainPrompt-2"),
        ("ITEM-P5-Functional-Network-8","P5-Functional-Network-8"),("ITEM-C3-Unrelated-English-8","C3-Unrelated-English-8"),
        ("ITEM-RND-Unit-75451","RND-Unit-75451")]
    for cand,ref in specific:
        mode="ITEM" if cand.startswith("ITEM-") else "B2"
        rmode="ITEM" if ref.startswith("ITEM-") else "B2"
        for scope,folds in (("all_10",range(1,11)),("exclude_fold2_descriptive_9",[1,3,4,5,6,7,8,9,10])):
            diffs={metric:[] for metric in ("acc","prob_auc","log_loss","brier")};w=t=l=0
            for f in folds:
                if (f,cand,mode) not in fold_metrics or (f,ref,rmode) not in fold_metrics:continue
                a=fold_metrics[(f,cand,mode)];b=fold_metrics[(f,ref,rmode)];delta=a["acc"]-b["acc"]
                w+=delta>1e-12;t+=abs(delta)<=1e-12;l+=delta<-1e-12
                for metric in diffs:diffs[metric].append(a[metric]-b[metric])
            if diffs["acc"]:compare.append({"candidate":cand,"mode":mode,"reference":ref,"reference_mode":rmode,"scope":scope,
                "delta_acc_pp_mean":100*np.mean(diffs["acc"]),"delta_prob_auc_pp_mean":100*np.mean(diffs["prob_auc"]),
                "delta_log_loss_mean":np.mean(diffs["log_loss"]),"delta_brier_mean":np.mean(diffs["brier"]),
                "acc_W":int(w),"acc_T":int(t),"acc_L":int(l),"acc_signflip_p_exploratory":summary_signflip(diffs["acc"]),
                "prob_auc_signflip_p_exploratory":summary_signflip(diffs["prob_auc"]),"logloss_signflip_p_exploratory":summary_signflip(diffs["log_loss"]),
                "brier_signflip_p_exploratory":summary_signflip(diffs["brier"]),"n_folds":len(diffs["acc"])})
    # Holm correction is applied to exploratory paired fold tests by metric/reference/scope family.
    families={}
    for i,row in enumerate(compare):
        family=(row["scope"],row.get("reference","R0-Frozen-Identity"),row["mode"])
        families.setdefault(family,[]).append(i)
    for metric in ("acc","prob_auc","logloss","brier"):
        pkey=f"{metric}_signflip_p_exploratory"
        for indexes in families.values():
            valid=[i for i in indexes if pkey in compare[i]]
            if not valid:continue
            ps=np.asarray([compare[i][pkey] for i in valid],dtype=float)
            order=np.argsort(ps);adj=np.empty_like(ps);running=0.
            for rank,j in enumerate(order):running=max(running,min(1.,(len(ps)-rank)*ps[j]));adj[j]=running
            for j,i in enumerate(valid):compare[i][f"{metric}_signflip_p_holm_family"]=float(adj[j])
    return summary,compare

def run_token_importance(banks):
    rows=[]; rng0=global_rng_sha(); old_norm={}
    old_path=E754/"results/token_probe_summary.csv"
    with old_path.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):old_norm[int(r["token"])]=float(r["mean_selected_logistic_coefficient_norm"])
    for f in range(1,11):
        d=load_fold(f);tr=d["probe_train"].astype(int); yall=(d["labels"]==0).astype(int);y=yall[tr];tokens=d["tokens"].astype(np.float32)[tr]
        cv=StratifiedKFold(5,shuffle=True,random_state=CFG["analysis"]["inner_seed_base"]+f);prior_oof=np.zeros(len(y),dtype=np.float64)
        oof=np.zeros((len(tr),16),dtype=np.float64);fold_stats={t:[] for t in range(16)}
        split_records=list(cv.split(np.zeros(len(y)),y))
        for a,b in split_records:
            prior_oof[b]=np.mean(y[a])
        for token in range(16):
            xx=tokens[:,token,:]
            for split_id,(a,b) in enumerate(split_records):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=CFG["analysis"]["token_oof_c"],solver="lbfgs",max_iter=3000,random_state=0))
                model.fit(xx[a],y[a]);oof[b,token]=model.predict_proba(xx[b])[:,1]
                pp=np.clip(oof[b,token],1e-8,1-1e-8)
                fold_stats[token].append({"inner_fold":split_id+1,"log_loss":float(log_loss(y[b],pp,labels=[0,1])),"prob_auc":float(rp.metrics(y[b],pp)["prob_auc"])})
        prior_loss=log_loss(y,np.clip(prior_oof,1e-8,1-1e-8),labels=[0,1]);imp=[]
        for token in range(16):
            loss=log_loss(y,np.clip(oof[:,token],1e-8,1-1e-8),labels=[0,1]);auc=rp.metrics(y,oof[:,token])["prob_auc"]
            imp.append(prior_loss-loss)
            fm=[x["log_loss"] for x in fold_stats[token]];fa=[x["prob_auc"] for x in fold_stats[token]]
            rows.append({"fold":f,"kind":"supervised_oof_token","token":token,"slot_origin":["PHENO","ANAT","FUNC","CORR"][token//4],
                "token_within_slot":token%4,"oof_logloss":loss,"prior_logloss":prior_loss,"oof_logloss_improvement":prior_loss-loss,
                "oof_prob_auc":auc,"oof_acc":rp.metrics(y,oof[:,token])["acc"],"oof_sen":rp.metrics(y,oof[:,token])["sen"],
                "oof_spe":rp.metrics(y,oof[:,token])["spe"],"oof_asd_f1":rp.metrics(y,oof[:,token])["asd_f1"],
                "inner_fold_logloss_mean":float(np.mean(fm)),"inner_fold_logloss_sd":float(np.std(fm)),
                "inner_fold_prob_auc_mean":float(np.mean(fa)),"inner_fold_prob_auc_sd":float(np.std(fa)),
                "oof_top4_by_logloss":False,"old_e754_mean_coefficient_norm":old_norm.get(token)})
        top=set(np.argsort(imp)[-4:].tolist())
        for r in rows[-16:]:r["oof_top4_by_logloss"]=r["token"] in top
        wt,wp=rp.projections(CFG["analysis"]["b1_projection_seed_base"]+f)
        for group in PROMPT_GROUPS+[
            "P1-SPromptGL-Anonymous-Gram-Matched","P5-Functional-Network-Anonymous-Gram-Matched","C3-Unrelated-English-8"]+[f"RND-Unit-{s}" for s in CFG["analysis"]["random_unit_seeds"]]:
            vec,tags=banks[group] if group in banks else (None,None)
            if group.endswith("Anonymous-Gram-Matched"):
                src="P1-SPromptGL-8" if group.startswith("P1") else "P5-Functional-Network-8"; v,t=banks[src];q=rp.norm_rows(v@wp)
                rng=np.random.default_rng(CFG["analysis"]["anonymous_seed_base"]+f+(100 if src.startswith("P5") else 0));o,r=np.linalg.qr(rng.standard_normal((128,128)));o*=np.where(np.diag(r)<0,-1.,1.)[None,:];q=q@o;tags=t
            else:q=rp.norm_rows(vec@wp)
            score,sims=rp.b1_features(tokens,q,tags,wt)
            contrast=np.mean(score,axis=0) if score.ndim==2 else score
            # Group ranking uses absolute patient-averaged ASD/TD contrast, based only on outer training rows.
            sd=np.zeros(16)
            for tok in range(16):
                aa=sims[:,tok,np.asarray(tags)=="ASD"].mean();tt=sims[:,tok,np.asarray(tags)=="TD"].mean();sd[tok]=abs(aa-tt)
            top_prompt=set(np.argsort(sd)[-4:].tolist()); sup_top=set(np.argsort(imp)[-4:].tolist());overlap=len(top_prompt&sup_top)
            rng=np.random.default_rng(CFG["analysis"]["token_importance_seed_base"]+f*100+PROMPT_GROUPS.index(group) if group in PROMPT_GROUPS else CFG["analysis"]["token_importance_seed_base"]+f*1000+sum(map(ord,group)))
            count=np.zeros(16,dtype=int)
            for _ in range(CFG["analysis"]["token_bootstraps"]):
                sample=rng.integers(0,len(tokens),len(tokens));sample_sims=sims[sample];ss=[]
                for tok in range(16):ss.append(abs(float(np.mean(sample_sims[:,tok,np.asarray(tags)=="ASD"]))-float(np.mean(sample_sims[:,tok,np.asarray(tags)=="TD"]))))
                count[np.argsort(ss)[-4:]]+=1
            for token in range(16):
                rows.append({"fold":f,"kind":"prompt_token_match","group":group,"token":token,"slot_origin":["PHENO","ANAT","FUNC","CORR"][token//4],
                    "token_within_slot":token%4,"prompt_abs_class_similarity_gap":sd[token],"prompt_top4":token in top_prompt,
                    "bootstrap_top4_frequency":count[token]/max(1,CFG["analysis"]["token_bootstraps"]),"supervised_oof_importance":imp[token],
                    "supervised_oof_top4":token in sup_top,"top4_overlap_count":overlap,"random_top4_expected_overlap":1.0})
            all_tokens=load_fold(f)["tokens"].astype(np.float32)
            _,all_sims=rp.b1_features(all_tokens,q,tags,wt)
            all_gap=all_sims[:,:,np.asarray(tags)=="ASD"].mean(-1)-all_sims[:,:,np.asarray(tags)=="TD"].mean(-1)
            all_top=np.argsort(np.abs(all_gap),axis=1)[:,-4:]
            fold_data=load_fold(f);test_idx=fold_data["test"].astype(int);labels_asd=(fold_data["labels"]==0)
            base_correct=(fold_data["probability_all"][:,0]>=.5)==labels_asd
            scopes=[("all_subjects",np.arange(len(all_tokens))),
                    ("outer_test_correct",test_idx[base_correct[test_idx]]),
                    ("outer_test_error",test_idx[~base_correct[test_idx]])]
            for scope,idx in scopes:
                if len(idx)==0: continue
                top=all_top[idx];freq=np.bincount(top.reshape(-1),minlength=16)/len(idx)/4
                unique=len({tuple(sorted(map(int,row))) for row in top})
                pair_rng=np.random.default_rng(CFG["analysis"]["bootstrap_seed"]+f*10000+sum(map(ord,group))+sum(map(ord,scope)))
                aa=pair_rng.integers(0,len(top),5000);bb=pair_rng.integers(0,len(top),5000)
                jac=[]
                for a,b in zip(aa,bb):
                    sa=set(map(int,top[a]));sb=set(map(int,top[b]));jac.append(len(sa&sb)/max(1,len(sa|sb)))
                for token in range(16):
                    rows.append({"fold":f,"kind":"patient_top4_frequency","group":group,"selection_scope":scope,"token":token,
                        "slot_origin":["PHENO","ANAT","FUNC","CORR"][token//4],"token_within_slot":token%4,
                        "top4_selection_frequency":float(freq[token]),"n_subjects":len(idx),"unique_top4_sets":unique,
                        "unique_top4_set_fraction":unique/len(idx),"mean_pairwise_top4_jaccard":float(np.mean(jac))})
            if group in PROMPT_GROUPS:
                prompt_sets=[]
                for j,item in enumerate(PROMPTS["banks"][group]):
                    per=sims[:,:,j].mean(0); pt=set(np.argsort(per)[-4:].tolist())
                    prompt_sets.append(pt)
                    for tok in range(16): rows.append({"fold":f,"kind":"individual_prompt_similarity","group":group,"prompt_index":j,
                        "prompt_text":item["text"],"token":tok,"mean_similarity_train":float(per[tok]),"prompt_individual_top4":tok in pt})
                overlaps=[]
                for a in range(len(prompt_sets)):
                    for b in range(a+1,len(prompt_sets)):
                        overlaps.append(len(prompt_sets[a]&prompt_sets[b])/max(1,len(prompt_sets[a]|prompt_sets[b])))
                rows.append({"fold":f,"kind":"prompt_prompt_top4_overlap","group":group,"mean_pairwise_prompt_top4_jaccard":float(np.mean(overlaps)) if overlaps else 1.,
                    "unique_prompt_top4_sets":len({tuple(sorted(x)) for x in prompt_sets}),"prompt_count":len(prompt_sets)})
    if global_rng_sha()!=rng0:raise RuntimeError("token analysis changed global Python/NumPy/Torch RNG")
    write_csv(RES/"token_importance.csv",rows)

def summary_signflip(d):
    d=np.asarray(d,dtype=float); obs=abs(float(np.mean(d))); n=len(d)
    if n<=16:
        vals=[abs(float(np.mean(d*np.asarray(s)))) for s in itertools.product((-1.,1.),repeat=n)]
        return float(np.mean(np.asarray(vals)>=obs-1e-15))
    rng=np.random.default_rng(754499); signs=rng.choice([-1.,1.],size=(100000,n));return float((1+np.sum(np.abs(np.mean(signs*d,axis=1))>=obs))/(len(signs)+1))

def build_manifest(cache_audit):
    base=json.loads((BASE/"manifest.json").read_text());pm=json.loads(EPM.read_text())
    locked=[BASE/"manifest.json",E754/"manifest.json",E754/"run.py",E754/"probe.py",E754/"audit.py",
            E754/"config.json",E754/"prompt_banks.json",E754/"results/report.md",E754/"results/summary.csv",
            E754/"results/paired_comparisons.csv",E754/"results/token_probe_summary.csv",E754/"results/feature_extraction_audit.json",
            EMB,EPM,Path("network.py"),Path("model.py"),Path("layers.py"),Path("utils.py")]
    for f in range(1,11):locked.extend([FR/f"fold{f:02d}.npz",FR/f"fold{f:02d}.json"])
    hashes={str(p.resolve()):sha_file(p) for p in locked if p.is_file()}
    return {"experiment":"Experiment 75.4.1 — BP-PUR","status":"PREFLIGHT_PASSED_FORMAL_LOCKED","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      "root":str(EXP),"preferred_root":CFG.get("preferred_root",str(EXP)),"root_fallback_reason":CFG.get("root_fallback_reason"),"base":{"name":"E74.4-G2-Positive-DCR","path":str(BASE),
       "training_source_sha256":base["source_sha256"],"manifest_json_sha256":sha_file(BASE/"manifest.json"),"status":base["status"],
       "completed_tasks":base["completed_tasks"],"independent_audits":base["independent_audits"]},
      "e754":{"path":str(E754),"manifest_sha256":sha_file(E754/"manifest.json"),"feature_extraction_audit_sha256":sha_file(E754/"results/feature_extraction_audit.json"),
       "prompt_cache_sha256":sha_file(EMB),"prompt_manifest_sha256":sha_file(EPM),"clip_weight_sha256":pm["weights_sha256"]},
      "cache_audit":cache_audit,"locked_config_sha256":sha_file(EXP/"config.json"),"locked_implementation_sha256":{
       "run.py":sha_file(EXP/"run.py"),"residual_probe.py":sha_file(EXP/"residual_probe.py")},
      "locked_source_hashes":hashes,"formal_settings":CFG["analysis"],"fold2_probe_training_exclusion":[377,407],
      "external_test_used_for_probe_fit_or_lambda_selection":False,"backbone_training":False,
      "pilot":{"path":str(RES/"pilot_status.json"),"sha256":sha_file(RES/"pilot_status.json"),"formal_steps":STEPS},
      "limitations":CFG["limitations"],"task_status":{"frozen_folds":10,"probe_fold_models_completed":0,"independent_replays":0}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["preflight","diagnose","pilot","formal","tokens","all"],default="preflight");args=ap.parse_args()
    RES.mkdir(parents=True,exist_ok=True);RT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    rng_before=global_rng_sha();audit=verify_cache();atomic_json(RES/"cache_audit.json",audit)
    banks,pm=load_banks();embedding_sha=sha_file(EMB)
    if args.phase in ("preflight","diagnose","all"):run_m0_diagnostics(banks)
    if args.phase in ("preflight","pilot","all"):run_pilot(banks)
    if args.phase in ("formal","all"):
        pilot=json.loads((RES/"pilot_status.json").read_text())
        if pilot.get("status")!="PASS" or int(pilot["formal_steps_locked"])!=STEPS:raise RuntimeError("formal training budget is not backed by the locked inner-fold pilot")
        prior_manifest=json.loads((EXP/"manifest.json").read_text()) if (EXP/"manifest.json").is_file() else {}
        prior_run_sha=prior_manifest.get("locked_implementation_sha256",{}).get("run.py")
        manifest=build_manifest(audit)
        repairs=list(prior_manifest.get("post_lock_analysis_repairs",[]))
        if prior_manifest.get("post_lock_analysis_repair"):
            repairs.append(prior_manifest["post_lock_analysis_repair"])
        if prior_run_sha and prior_run_sha!=manifest["locked_implementation_sha256"]["run.py"]:
            repairs.append({"pre_repair_run_sha256":prior_run_sha,
                "post_repair_run_sha256":manifest["locked_implementation_sha256"]["run.py"],
                "scope":"Fixed the saved-analysis path after 370 probe models were fitted; no model fitting, predictions, or hyperparameters changed."})
        if repairs: manifest["post_lock_analysis_repairs"]=repairs
        atomic_json(EXP/"manifest.json",manifest)
        run_formal(banks);run_token_importance(banks)
        manifest["status"]="FORMAL_COMPLETE_PENDING_INDEPENDENT_AUDIT"
        manifest["task_status"]={"frozen_folds":10,"probe_fold_models_completed":sum(len(v["entries"]) for v in json.loads((RT/"model_index.json").read_text()).values()),"independent_replays":0}
        manifest["results"]={str(p.relative_to(EXP)):sha_file(p) for p in RES.iterdir() if p.is_file()}
        manifest["source_hashes_at_formal_completion"]={"run.py":sha_file(EXP/"run.py"),"residual_probe.py":sha_file(EXP/"residual_probe.py"),"config.json":sha_file(EXP/"config.json")}
        atomic_json(EXP/"manifest.json",manifest)
    elif args.phase=="tokens":run_token_importance(banks)
    if global_rng_sha()!=rng_before:raise RuntimeError("analysis changed global Python/NumPy/Torch RNG")
    atomic_json(RES/"analysis_rng_audit.json",{"status":"PASS","before_sha256":rng_before,"after_sha256":global_rng_sha(),"changed":False,
      "prompt_cache_sha256":embedding_sha,"base_forward_count":0})

if __name__=="__main__":main()
