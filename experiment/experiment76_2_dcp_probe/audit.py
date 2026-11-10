#!/usr/bin/env python3
"""Independent metric and isolation audit for Experiment 76.2 diagnostic outputs."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, re, sys
from pathlib import Path

EXP=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def load_probe():
    spec=importlib.util.spec_from_file_location("e762_probe_for_audit",EXP/"probe.py")
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module

def read_csv(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def independent_metrics(y,pred,score,is_probability):
    import numpy as np
    from sklearn.metrics import (accuracy_score,balanced_accuracy_score,roc_auc_score,
        f1_score,log_loss,brier_score_loss)
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); score=np.asarray(score,dtype=float)
    result={"acc":float(accuracy_score(y,pred)),"hard_auc":float(balanced_accuracy_score(y,pred)),
      "prob_auc":float(roc_auc_score(y==0,score)),
      "sen":float(np.sum((y==0)&(pred==0))/max(1,np.sum(y==0))),
      "spe":float(np.sum((y==1)&(pred==1))/max(1,np.sum(y==1))),
      "asd_f1":float(f1_score(y==0,pred==0,zero_division=0))}
    result["log_loss"]=float(log_loss(y==0,np.c_[1-score,score],labels=[0,1])) if is_probability else ""
    result["brier"]=float(brier_score_loss(y==0,score)) if is_probability else ""
    return result

def assert_close(a,b,label):
    if a=="" or b=="":
        if a!=b: raise AssertionError(f"{label}: blank mismatch {a!r} vs {b!r}")
    elif not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError(f"{label}: {a} vs {b}")

def audit():
    import numpy as np
    p=load_probe(); results=EXP/"results"; cache_path=EXP/".runtime/cache/oof_predictions.npz"
    status=json.loads((results/"audit_status.json").read_text())
    rows=read_csv(results/"fold_metrics.csv"); cache=np.load(cache_path,allow_pickle=False)
    if len(rows)!=50: raise AssertionError(f"expected 50 outer-fold diagnostic rows, got {len(rows)}")
    stored={(int(r["fold"]),r["group"]):r for r in rows}
    if len(stored)!=50: raise AssertionError("duplicate fold/group metric rows")
    recomputed=[]; isolation=[]; expected_changes=[]
    R=p.runner()
    for fold in range(1,11):
        state,warm,identity,task=R.prepare_fold(fold)
        train=R.index_numpy(state["train"]); test=R.index_numpy(state["test"])
        eligible=np.asarray(sorted(set(train.tolist())-set(test.tolist())),dtype=int)
        prefix=f"f{fold}_"
        ids=np.asarray(cache[prefix+"eligible"],dtype=int)
        y=np.asarray(cache[prefix+"labels"],dtype=int)
        if not np.array_equal(ids,eligible): raise AssertionError(f"fold {fold}: eligible support IDs differ from frozen outer split")
        if np.intersect1d(ids,test).size: raise AssertionError(f"fold {fold}: test ID entered diagnostic cache")
        if fold==2 and ({376,406}&set(ids.tolist())): raise AssertionError("fold 2 exposed rows entered diagnostic support")
        if len(y)!=len(ids): raise AssertionError(f"fold {fold}: cached labels misaligned")
        audit_entry=status["checkpoint_folds"][fold-1]
        ckpt=p.BASE/f".runtime/checkpoints/{p.BASE_GROUP}_f{fold:02d}.best.pt"
        task_path=p.BASE/f".runtime/tasks/{p.BASE_GROUP}_f{fold:02d}.json"
        task_json=json.loads(task_path.read_text())
        if p.sha_file(ckpt)!=audit_entry["checkpoint_sha256"] or task_json.get("checkpoint_sha256")!=audit_entry["checkpoint_sha256"]:
            raise AssertionError(f"fold {fold}: checkpoint SHA changed")
        if task_json.get("identity")!=identity: raise AssertionError(f"fold {fold}: checkpoint identity differs from base split")
        group_names=p.CONDITIONS[:-1]
        for group in group_names:
            key=prefix+group
            g_ids=np.asarray(cache[key+"_ids"],dtype=int); pred=np.asarray(cache[key+"_pred"],dtype=int); score=np.asarray(cache[key+"_score"],dtype=float)
            if not np.array_equal(g_ids,ids) or len(pred)!=len(ids) or len(score)!=len(ids): raise AssertionError(f"fold {fold} {group}: OOF prediction alignment failed")
            prob=group!="B0_NL_CAM_only_RBF_SVM"
            m=independent_metrics(y,pred,score,prob); recorded=stored[(fold,group)]
            for metric,value in m.items(): assert_close(value,recorded[metric],f"{fold}/{group}/{metric}")
            recomputed.append({"fold":fold,"group":group,**m})
        b3_metrics=[]
        for k in range(5):
            key=prefix+f"B3_Shuffled_Pair_Control_{k}"
            g_ids=np.asarray(cache[key+"_ids"],dtype=int); pred=np.asarray(cache[key+"_pred"],dtype=int); score=np.asarray(cache[key+"_score"],dtype=float)
            if not np.array_equal(g_ids,ids) or len(pred)!=len(ids) or len(score)!=len(ids): raise AssertionError(f"fold {fold} B3 permutation {k}: alignment failed")
            b3_metrics.append(independent_metrics(y,pred,score,True))
        b3={m:float(np.mean([q[m] for q in b3_metrics])) for m in p.METRICS}
        recorded=stored[(fold,"B3_Shuffled_Pair_Control")]
        for metric,value in b3.items(): assert_close(value,recorded[metric],f"{fold}/B3/{metric}")
        recomputed.append({"fold":fold,"group":"B3_Shuffled_Pair_Control",**b3})
        base_pred=np.asarray(cache[prefix+"base_pred"],dtype=int); base_prob=np.asarray(cache[prefix+"base_prob"],dtype=float)
        stability=np.asarray(cache[prefix+"stability"],dtype=float)
        role_masks=[(y==0)&(base_pred==0)&(base_prob[:,0]>=.70)&(stability>=.90),
          (y==1)&(base_pred==1)&(base_prob[:,1]>=.70)&(stability>=.90),
          (y==0)&(base_pred==1)&(base_prob[:,1]>=.60)&(stability>=.90),
          (y==1)&(base_pred==0)&(base_prob[:,0]>=.60)&(stability>=.90)]
        subsets={"all":np.ones(len(ids),dtype=bool),"stable_hard":role_masks[2]|role_masks[3],
          "stable_reliable":role_masks[0]|role_masks[1]}
        b2_pred=np.asarray(cache[prefix+"B2_CAM_Reliable_Hard_Paired_pred"],dtype=int)
        compare_packs=[("B0_CAM_only_Linear",[np.asarray(cache[prefix+"B0_CAM_only_Linear_pred"],dtype=int)]),
          ("B1_CAM_Reliable",[np.asarray(cache[prefix+"B1_CAM_Reliable_pred"],dtype=int)]),
          ("B3_Shuffled_Pair_Control",[np.asarray(cache[prefix+f"B3_Shuffled_Pair_Control_{k}_pred"],dtype=int) for k in range(5)]),
          ("B0_NL_CAM_only_RBF_SVM",[np.asarray(cache[prefix+"B0_NL_CAM_only_RBF_SVM_pred"],dtype=int)])]
        for name,preds in compare_packs:
            for k,other in enumerate(preds):
                perm=str(k) if name.startswith("B3") else ""
                for subset,select in subsets.items():
                    wrong=(other!=y)&(b2_pred==y)&select; harmed=(other==y)&(b2_pred!=y)&select
                    expected_changes.append({"fold":fold,"comparison":"B2_vs_"+name,"permutation":perm,"subset":subset,
                      "n":int(select.sum()),"wrong_to_correct":int(wrong.sum()),"correct_to_wrong":int(harmed.sum()),
                      "net_corrections":int(wrong.sum()-harmed.sum()),
                      "asd_to_td_errors_before":int(np.sum((y==0)&(other==1)&select)),
                      "asd_to_td_errors_after":int(np.sum((y==0)&(b2_pred==1)&select)),
                      "td_to_asd_errors_before":int(np.sum((y==1)&(other==0)&select)),
                      "td_to_asd_errors_after":int(np.sum((y==1)&(b2_pred==0)&select)),
                      "hard_prediction_flips":int(np.sum((other!=b2_pred)&select))})
        if audit_entry["output_layer_calls"]!=1 or not audit_entry["zero_perturbation_exact"] or not audit_entry["frozen_weights_exact"]:
            raise AssertionError(f"fold {fold}: frozen model/CAM parity failed")
        if audit_entry["checkpoint_weights_sha256"]!=audit_entry["post_weights_sha256"]:
            raise AssertionError(f"fold {fold}: model/CAM parameters changed")
        isolation.append({"fold":fold,"eligible_count":len(ids),"test_count":len(test),"support_test_intersection":0,
          "row377_407_excluded":fold!=2 or not ({376,406}&set(ids.tolist())),"checkpoint_sha256":p.sha_file(ckpt),
          "identity_sha256":p.sha_json(identity),"output_layer_calls":1,"weights_unchanged":True})
    current=p.source_inventory(); current_sha=p.sha_json(current)
    if current_sha!=status["source_sha256"]: raise AssertionError("source inventory changed between diagnostics and audit")
    # Recompute H1 independently from per-fold stable-role counts and prototype contrasts.
    diag=read_csv(results/"prototype_diagnostics.csv")
    roles={(int(r["fold"]),r["name"]):int(r["n"]) for r in diag if r.get("kind")=="role"}
    held={int(r["fold"]):float(r["value"]) for r in diag if r.get("kind")=="heldout_similarity" and r.get("name")=="same_minus_other" and r.get("value") not in ("",None)}
    hard={}
    for r in diag:
        if r.get("kind")=="hard_vs_reliable" and r.get("value") not in ("",None): hard.setdefault(int(r["fold"]),[]).append(float(r["value"]))
    h1_folds=int(sum(roles.get((f,"R_ASD"),0)>=3 and roles.get((f,"R_TD"),0)>=3 and
      max(roles.get((f,"H_ASD_to_TD"),0),roles.get((f,"H_TD_to_ASD"),0))>=3 and f in hard and np.mean(hard[f])>0 and held.get(f,-1)>0 for f in range(1,11)))
    candidate_folds=int(sum(roles.get((f,"R_ASD"),0)>=3 and roles.get((f,"R_TD"),0)>=3 and
      max(roles.get((f,"H_ASD_to_TD"),0),roles.get((f,"H_TD_to_ASD"),0))>=3 for f in range(1,11)))
    h1="SUPPORTED" if h1_folds>=8 and len(held)>=8 and sum(v>0 for v in held.values())>=7 else ("INSUFFICIENT" if candidate_folds<5 or len(held)<8 else "NOT_SUPPORTED")
    stored_changes=read_csv(results/"mechanism.csv")
    if len(stored_changes)!=len(expected_changes): raise AssertionError("mechanism row count mismatch")
    key=lambda r:(int(r["fold"]),r["comparison"],str(r["permutation"]),r["subset"])
    actual_changes={key(r):r for r in stored_changes}
    if len(actual_changes)!=len(expected_changes): raise AssertionError("duplicate mechanism rows")
    for expected in expected_changes:
        actual=actual_changes.get(key(expected))
        if actual is None: raise AssertionError(f"missing mechanism row {key(expected)}")
        for field,value in expected.items():
            if field in ("fold","comparison","permutation","subset"): continue
            assert_close(value,actual[field],f"mechanism/{key(expected)}/{field}")
    status.update({"status":"PASS","independent_metric_recalculation":"50/50 fold-condition summaries matched from saved OOF predictions",
      "prediction_cache_sha256":p.sha_file(cache_path),"source_inventory_rechecked_sha256":current_sha,
      "frozen_model_checkpoint_checks":"10/10 unchanged; OutputLayer invoked once per extraction; zero-noise exact",
      "data_isolation":"10/10 nested support IDs match eligible outer train and are disjoint from outer test; Fold 2 rows 377/407 excluded",
      "h1_verdict":h1,"h1_joint_support_folds":h1_folds,"h2_verdict":status.get("h2_verdict"),
      "h1_candidate_support_folds":candidate_folds,"prediction_change_audit":f"{len(expected_changes)}/{len(stored_changes)} rows independently reproduced",
      "independent_audit_folds":isolation,"metrics_independently_recomputed":recomputed})
    (results/"audit_status.json").write_text(json.dumps(status,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report=results/"report.md"; text=report.read_text(encoding="utf-8").replace("Status: DIAGNOSTIC_COMPLETE_PENDING_AUDIT;","Status: COMPLETE_AUDITED; independent metric/isolation audit PASS.")
    text=re.sub(r"Verdict: [A-Z_]+\.",f"Verdict: {h1}; independent audit PASS.",text,count=1)
    report.write_text(text,encoding="utf-8")
    manifest={"experiment":"Experiment 76.2 - DCP-Probe","status":"COMPLETE_AUDITED","folds":10,
      "checkpoint_audits":10,"diagnostic_decision":status.get("h2_verdict"),"h1_verdict":h1,
      "independent_audit":"PASS","source_sha256":current_sha,
      "results_sha256":{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in results.iterdir() if q.is_file()}}
    (EXP/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","h1":h1,"h1_joint_support_folds":h1_folds,"recomputed_rows":len(recomputed)},indent=2))

if __name__=="__main__": audit()
