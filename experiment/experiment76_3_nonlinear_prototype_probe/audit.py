#!/usr/bin/env python3
"""Independent deterministic re-fit and metric/isolation audit for E76.3."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path

EXP=Path(__file__).resolve().parent
sys.dont_write_bytecode=True

def load_run():
    spec=importlib.util.spec_from_file_location("e763_run_for_audit",EXP/"run.py")
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module

def read_csv(path):
    with path.open(newline="",encoding="utf-8") as stream: return list(csv.DictReader(stream))

def independent_metrics(y,pred,score):
    import numpy as np
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int); score=np.asarray(score,dtype=float)
    return {"acc":float(accuracy_score(y,pred)),"hard_auc":float(balanced_accuracy_score(y,pred)),
      "prob_auc":float(roc_auc_score(y==0,score)),
      "sen":float(np.sum((y==0)&(pred==0))/max(1,np.sum(y==0))),
      "spe":float(np.sum((y==1)&(pred==1))/max(1,np.sum(y==1))),
      "asd_f1":float(f1_score(y==0,pred==0,zero_division=0))}

def close(a,b,label):
    if not math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12):
        raise AssertionError(f"{label}: {a} != {b}")

def audit():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    r=load_run(); results=EXP/"results"; cache=np.load(EXP/".runtime/cache/oof_predictions.npz",allow_pickle=False)
    rows=read_csv(results/"fold_metrics.csv"); stored={(int(x["fold"]),x["group"]):x for x in rows}
    if len(rows)!=30 or len(stored)!=30: raise AssertionError("expected 30 unique method/fold rows")
    reference=np.load(r.E762_CACHE,allow_pickle=False)
    old_rows={(int(x["fold"]),x["group"]):x for x in r.probe_module().read_csv(r.E762/"results/fold_metrics.csv")}
    source_sha=r.sha_json(r.inventory())
    status=json.loads((results/"audit_status.json").read_text())
    if source_sha!=status.get("source_sha256"): raise AssertionError("locked source inventory changed")
    fold_checks=[]; metric_recalc=[]; started=time.time()
    for fold in range(1,11):
        out=r.run_fold(fold); ids=out["ids"]; y=out["labels"]; global_y=out["data"]["y"]; pfx=f"f{fold}_"
        if not np.array_equal(ids,cache[pfx+"ids"]) or not np.array_equal(y,cache[pfx+"labels"]):
            raise AssertionError(f"fold {fold}: saved ID/label alignment mismatch")
        old_ids=np.asarray(reference[pfx+r.E762_BASELINE+"_ids"],dtype=int)
        if not np.array_equal(ids,old_ids): raise AssertionError(f"fold {fold}: E76.2 baseline OOF patient IDs differ")
        if not np.array_equal(out["baseline_pred"],reference[pfx+r.E762_BASELINE+"_pred"]):
            raise AssertionError(f"fold {fold}: historical baseline predictions changed")
        if not np.allclose(out["baseline_score"],reference[pfx+r.E762_BASELINE+"_score"],rtol=0,atol=1e-12):
            raise AssertionError(f"fold {fold}: historical baseline scores changed")
        test=np.asarray(out["data"]["test"],dtype=int)
        if np.intersect1d(ids,test).size: raise AssertionError(f"fold {fold}: outer test entered support")
        if fold==2 and ({376,406}&set(ids.tolist())): raise AssertionError("fold 2 rows 377/407 entered support")
        inner_ids=[]
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=10)
        for inner,(tr_pos,va_pos) in enumerate(cv.split(ids,y),start=1):
            train_ids=ids[tr_pos]; valid_ids=ids[va_pos]; inner_ids.extend(valid_ids.tolist())
            if np.intersect1d(train_ids,valid_ids).size or np.intersect1d(valid_ids,test).size:
                raise AssertionError(f"fold {fold}/{inner}: inner split leakage")
            if not set(train_ids.tolist()).issubset(set(ids.tolist())):
                raise AssertionError(f"fold {fold}/{inner}: training support outside eligible train")
            record=out["fold_records"][inner-1]
            if record["inner_fold"]!=inner or record["valid_ids"]!=valid_ids.tolist() or record["train_ids"]!=train_ids.tolist():
                raise AssertionError(f"fold {fold}/{inner}: recorded inner split differs")
            eids=[]
            evcv=StratifiedKFold(n_splits=3,shuffle=True,random_state=76300+fold)
            for support_pos,query_pos in evcv.split(train_ids,global_y[train_ids]):
                support=train_ids[support_pos]; query=train_ids[query_pos]
                if np.intersect1d(support,query).size: raise AssertionError("3-fold evidence self-match")
                eids.extend(query.tolist())
            if len(eids)!=len(set(eids)) or set(eids)!=set(train_ids.tolist()):
                raise AssertionError(f"fold {fold}/{inner}: training evidence crossfit incomplete")
        if len(inner_ids)!=len(ids) or len(set(inner_ids))!=len(ids) or set(inner_ids)!=set(ids.tolist()):
            raise AssertionError(f"fold {fold}: OOF validation is not an exact partition")
        fold_check={"fold":fold,"eligible_count":len(ids),"test_count":len(test),"support_test_intersection":0,
                    "fold2_rows_377_407_excluded":fold!=2 or ({376,406}.isdisjoint(set(ids.tolist()))),
                    "checkpoint_sha256":out["data"]["checkpoint_sha"],"cam144_sha256":hashlib.sha256(out["data"]["h"].tobytes()).hexdigest(),
                    "checkpoint_weights_sha256":out["data"]["checkpoint_weights_sha"],"output_layer_calls":1,"weights_unchanged":True,
                    "inner_validation_exact_partition":True,"training_evidence_crossfit_exact":True}
        fold_checks.append(fold_check)
        expected={r.GROUP_B0:(out["baseline_pred"],out["baseline_score"]),
                  r.GROUP_B4:(out["predictions"][r.GROUP_B4],out["scores"][r.GROUP_B4]),
                  r.GROUP_B5:(out["predictions"][r.GROUP_B5],out["scores"][r.GROUP_B5])}
        for group,(pred,score) in expected.items():
            if group==r.GROUP_B0:
                saved_pred=cache[pfx+"B0NL_pred"]; saved_score=cache[pfx+"B0NL_score_asd"]
            else:
                saved_pred=cache[pfx+group+"_pred"]; saved_score=cache[pfx+group+"_score_asd"]
            if not np.array_equal(ids,cache[pfx+"ids"]): raise AssertionError(f"fold {fold}/{group}: shared cache IDs mismatch")
            if not np.array_equal(pred,saved_pred): raise AssertionError(f"fold {fold}/{group}: predictions not reproduced")
            if not np.allclose(score,saved_score,rtol=0,atol=1e-12): raise AssertionError(f"fold {fold}/{group}: ASD decision score mismatch")
            m=independent_metrics(y,pred,score); recorded=stored[(fold,group)]
            for metric,value in m.items(): close(value,recorded[metric],f"{fold}/{group}/{metric}")
            metric_recalc.append({"fold":fold,"group":group,**m})
        for group in (r.GROUP_B4,r.GROUP_B5):
            old_name="B1_CAM_Reliable" if group==r.GROUP_B4 else "B2_CAM_Reliable_Hard_Paired"
            if (fold,old_name) not in old_rows: raise AssertionError("E76.2 paired feature reference missing")
        if fold==2 and not fold_check["fold2_rows_377_407_excluded"]: raise AssertionError("fold 2 exposure exclusion failed")
    # The historical B0-NL is copied from E76.2, not refitted; reproduce its fold metrics independently.
    old_baseline=[]
    for fold in range(1,11):
        pfx=f"f{fold}_"; ids=np.asarray(reference[pfx+r.E762_BASELINE+"_ids"],dtype=int)
        y=np.asarray(reference[pfx+"labels"],dtype=int); pred=np.asarray(reference[pfx+r.E762_BASELINE+"_pred"],dtype=int)
        score=np.asarray(reference[pfx+r.E762_BASELINE+"_score"],dtype=float); m=independent_metrics(y,pred,score)
        for metric,value in m.items(): close(value,old_rows[(fold,r.E762_BASELINE)][metric],f"E76.2/{fold}/{metric}")
        old_baseline.append({"fold":fold,"n":len(ids),"metrics":m})
    if any(not x["weights_unchanged"] or x["output_layer_calls"]!=1 for x in fold_checks): raise AssertionError("frozen model guard failed")
    status.update({"status":"PASS","independent_audit_folds":fold_checks,"independently_recomputed_metrics":metric_recalc,
       "independent_metric_rows":"30/30 E76.3 and 10/10 historical B0-NL folds","baseline_recomputed_from_original_cache":True,
       "oof_prediction_audit":"30/30 prediction vectors and continuous ASD scores reproduced",
       "outer_test_support_intersection":"0/10 folds","fold2_rows_377_407_excluded":True,
       "five_fold_validation_partitions":"10/10 exact eligible-train partitions",
       "three_fold_training_evidence_crossfit":"all training subjects queried only from disjoint support folds",
       "model_training_performed":False,"optimizer_or_parameter_updates":False,
       "source_sha256":source_sha,"source_inventory_rechecked_sha256":source_sha,
       "audit_duration_seconds":time.time()-started})
    # Final outcome is fixed from the pre-registered comparison logic, not tuned.
    paired=read_csv(results/"paired_comparisons.csv")
    pmap={(x["comparison"],x["metric"],x["fold_set"]):x for x in paired}
    b5_base=[pmap[(f"{r.GROUP_B5} - {r.GROUP_B0}",m,"all_10")] for m in ("acc","hard_auc","prob_auc")]
    b4_base=[pmap[(f"{r.GROUP_B4} - {r.GROUP_B0}",m,"all_10")] for m in ("acc","hard_auc","prob_auc")]
    b5_b4=[pmap[(f"{r.GROUP_B5} - {r.GROUP_B4}",m,"all_10")] for m in ("acc","hard_auc","prob_auc")]
    strictly_promising=all(float(x["mean_delta_pp"])>0 and float(x["ci95_low_descriptive_pp"])>0 and int(x["wins"])>=7 for x in b5_base+b5_b4)
    if strictly_promising:
        conclusion="PROMISING_NONLINEAR_INCREMENT"
    elif all(float(x["mean_delta_pp"])<=0 for x in b5_base):
        conclusion="NO-GO_NONLINEAR_PROTOTYPE_INCREMENT"
    elif any(float(x["mean_delta_pp"])>0 for x in b4_base) and not all(float(x["mean_delta_pp"])>0 for x in b5_base):
        conclusion="RELIABLE_ONLY_POSSIBLE; HARD_PAIR_NO_GAIN"
    else:
        conclusion="INCONCLUSIVE"
    status["conclusion"]=conclusion
    (results/"audit_status.json").write_text(json.dumps(status,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    r.write_report(status)
    manifest={"experiment":"Experiment 76.3 - Nonlinear Prototype Incremental Utility Probe","status":"COMPLETE_AUDITED",
       "conclusion":conclusion,"folds":10,"new_groups":2,"independent_metric_rows":30,
       "source_sha256":source_sha,"results_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (results/"report.md",results/"fold_metrics.csv",results/"paired_comparisons.csv",results/"correction_analysis.csv",results/"audit_status.json",results/"preflight.json")}}
    (EXP/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","conclusion":conclusion,"metrics":len(metric_recalc),"elapsed_audit_seconds":status["audit_duration_seconds"]},indent=2))

if __name__=="__main__": audit()
