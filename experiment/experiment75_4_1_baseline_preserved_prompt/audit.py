#!/usr/bin/env python3
"""Independent replay and integrity audit for Experiment 75.4.1 probes."""
from __future__ import annotations
import csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
import run as R
import residual_probe as rp
CFG=R.CFG;RES=R.RES;RT=R.RT

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def readcsv(path):
    with open(path,newline="",encoding="utf-8") as f:return list(csv.DictReader(f))

def load_state(archive,prefix,mode):
    def tensor(key):
        x=archive[prefix+key]
        if x.size==0:return None
        return torch.as_tensor(x.copy(),dtype=torch.float32)
    return {"mode":mode,"W":tensor("W"),"w":tensor("w"),"u":tensor("u"),"v":tensor("v"),"bias":tensor("bias"),
            "mean":tensor("mean"),"std":tensor("std"),"token_projection":tensor("token_projection")}

def expected_metrics(rows):
    rows=sorted(rows,key=lambda z:int(z["subject_index_0based"]))
    y=np.asarray([int(x["y_asd"]) for x in rows]);p=np.asarray([float(x["probability"]) for x in rows])
    return rp.metrics(y,p)

def main():
    start=time.time();torch.set_num_threads(1);manifest_path=EXP/"manifest.json";manifest=json.loads(manifest_path.read_text())
    if manifest.get("status")!="FORMAL_COMPLETE_PENDING_INDEPENDENT_AUDIT":raise RuntimeError("formal execution manifest is not complete")
    cache=R.verify_cache();pred=readcsv(RES/"predictions.csv");diag=readcsv(RES/"residual_diagnostics.csv");summ=readcsv(RES/"summary.csv")
    index=json.loads((RT/"model_index.json").read_text());banks,_=R.load_banks();pred_map={}
    for row in pred:
        key=(int(row["fold"]),row["group"],row["mode"],int(row["subject_index_0based"]))
        if key in pred_map:raise RuntimeError(f"duplicate prediction row {key}")
        pred_map[key]=row
    audit_rows=[]; allpass=True
    for fold in range(1,11):
        d=R.load_fold(fold);te=d["test"].astype(int);yall=(d["labels"]==0).astype(int);pbase=d["probability_all"][:,0];base=d["logp"][:,0]-d["logp"][:,1]
        entries=index[str(fold)]["entries"];archive=np.load(EXP/index[str(fold)]["path"],allow_pickle=False)
        variants_b2,_,_=R.make_variant_space(banks,fold,"B2");variants_b1,_,_=R.make_variant_space(banks,fold,"B1")
        seen=set()
        for entry in entries:
            group=entry["group"];mode=entry["mode"];prefix=entry["prefix"]
            if (group,mode) in seen:raise RuntimeError(f"duplicate fitted model {fold}/{group}/{mode}")
            seen.add((group,mode))
            if mode=="ITEM":
                src=group[len("ITEM-"):];q,tags=R.item_variant(*variants_b2[src][:2]);anchor=None
            else:q,tags,anchor=(variants_b1 if mode=="B1" else variants_b2)[group]
            if mode=="ITEM":state=load_state(archive,prefix,"ITEM")
            else:state=load_state(archive,prefix,mode)
            if mode=="B1" and state["token_projection"] is None:raise RuntimeError("B1 projection missing from archive")
            lam=float(entry["lambda"])
            p,resid,caprate=rp.predict(state,d["tokens"][te],base[te],pbase[te],q,tags,lam,float(CFG["analysis"]["residual_cap_log_odds"]),anchor)
            saved=[pred_map[(fold,group,mode,int(i))] for i in te]
            p_saved=np.asarray([float(x["probability"]) for x in saved]);y_saved=np.asarray([int(x["y_asd"]) for x in saved])
            y_test=yall[te]
            exact_labels=np.array_equal(y_saved,y_test)
            pred_equal=np.allclose(p,p_saved,rtol=0.,atol=1e-7)
            hard_replay_equal=np.array_equal((p>=.5).astype(int),np.asarray([int(x["prediction"]) for x in saved]))
            identity=True
            if lam==0.:
                identity=np.array_equal(p,pbase[te]) and np.array_equal(p_saved,pbase[te]) and np.array_equal(np.asarray([int(x["prediction"]) for x in saved]),(pbase[te]>=.5).astype(int))
            metric=rp.metrics(y_saved,p_saved)
            drow=next(x for x in diag if int(x["fold"])==fold and x["group"]==group and x["mode"]==mode)
            replay_acc=abs(metric["acc"]-float(drow["test_acc"]))<1e-12
            good=exact_labels and pred_equal and hard_replay_equal and identity and replay_acc
            allpass &= good
            audit_rows.append({"fold":fold,"group":group,"mode":mode,"status":"PASS" if good else "FAIL","lambda":lam,
              "prediction_replay_max_abs_error":float(np.max(np.abs(p-p_saved))),"prediction_equal":pred_equal,"hard_prediction_equal":hard_replay_equal,"labels_match_cache":exact_labels,
              "lambda_zero_identity_pass":identity,"metric_replay_pass":replay_acc,"test_n":len(te),"residual_cap_trigger_rate":caprate,
              "checkpoint_sha256":cache["folds"][str(fold)]["checkpoint_sha256"],"model_archive_sha256":index[str(fold)]["sha256"]})
        expected={("R1-NoText",m) for m in ("B1","B2")}
        expected.update((g,m) for g in (R.PROMPT_GROUPS+["P1-SPromptGL-Anonymous-Gram-Matched","P5-Functional-Network-Anonymous-Gram-Matched","C3-Unrelated-English-8"]+
          [f"RND-Unit-{s}" for s in CFG["analysis"]["random_unit_seeds"]]+["P1-SPromptGL-WrongClass","P4-Modality-Aware-WrongClass","P5-Functional-Network-WrongClass"]) for m in ("B1","B2"))
        expected.update(("ITEM-"+g,"ITEM") for g in CFG["analysis"]["item_preserving_groups"])
        if seen!=expected:allpass=False
        base_rows=[pred_map[(fold,"R0-Frozen-Identity","identity",int(i))] for i in te]
        base_identity=np.asarray([float(x["base_probability"]) for x in base_rows]);base_saved=np.asarray([float(x["probability"]) for x in base_rows])
        base_ok=np.array_equal(base_identity,pbase[te]) and np.array_equal(base_saved,pbase[te]) and np.array_equal(np.asarray([int(x["prediction"]) for x in base_rows]),(pbase[te]>=.5).astype(int))
        allpass &= base_ok
        audit_rows.append({"fold":fold,"group":"R0-Frozen-Identity","mode":"identity","status":"PASS" if base_ok else "FAIL","lambda":0.,
          "prediction_replay_max_abs_error":float(np.max(np.abs(base_identity-pbase[te]))),"prediction_equal":base_ok,"labels_match_cache":True,
          "lambda_zero_identity_pass":base_ok,"metric_replay_pass":base_ok,"test_n":len(te),"checkpoint_sha256":cache["folds"][str(fold)]["checkpoint_sha256"],
          "model_archive_sha256":index[str(fold)]["sha256"]})
        archive.close()
    # Recompute every group/metric summary independently from persisted predictions.
    summary_checks=[]
    grouped={}
    for x in pred:grouped.setdefault((int(x["fold"]),x["group"],x["mode"]),[]).append(x)
    for s in summ:
        g=s["group"];mode=s["mode"];scope=s["scope"];metric=s["metric"]
        folds=list(range(1,11)) if scope=="all_10" else ([1,3,4,5,6,7,8,9,10] if scope=="exclude_fold2_descriptive_9" else [2])
        vals=[]
        for f in folds:
            rows=grouped.get((f,g,mode),[])
            if scope=="fold2_remove_exposed_rows_descriptive":rows=[r for r in rows if int(r["subject_index_0based"]) not in (376,406)]
            if rows:vals.append(rp.metrics([int(r["y_asd"]) for r in rows],[float(r["probability"]) for r in rows])[metric])
        mean=float(np.mean(vals));sd=float(np.std(vals,ddof=0));ok=abs(mean-float(s["mean"]))<1e-12 and abs(sd-float(s["population_sd"]))<1e-12
        summary_checks.append({"group":g,"mode":mode,"scope":scope,"metric":metric,"pass":ok,"mean_error":abs(mean-float(s["mean"])),"sd_error":abs(sd-float(s["population_sd"]))})
        allpass &= ok
    gram=readcsv(RES/"anonymous_gram_audit.csv");gram_ok=all(float(x["max_abs_gram_error"])<1e-5 for x in gram) and bool(gram)
    rng=json.loads((RES/"analysis_rng_audit.json").read_text());rng_ok=rng["status"]=="PASS" and not rng["changed"]
    locked_ok=True
    for path,expected_hash in manifest["locked_source_hashes"].items():locked_ok &= Path(path).is_file() and sha(path)==expected_hash
    cap_ok=all(float(x.get("max_abs_effective_correction",float("inf")))<=float(CFG["analysis"]["residual_cap_log_odds"])+1e-6 for x in diag)
    fold2_ok=cache["folds"]["2"]["train_n"]==784 and json.loads((R.FR/"fold02.json").read_text())["overlap_rows_1based"]==[377,407]
    locked_impl=manifest["locked_implementation_sha256"]
    locked_ok &= sha(EXP/"config.json")==manifest["locked_config_sha256"]
    locked_ok &= sha(EXP/"run.py")==locked_impl["run.py"] and sha(EXP/"residual_probe.py")==locked_impl["residual_probe.py"]
    status="COMPLETE_AUDITED" if allpass and gram_ok and rng_ok and locked_ok and cap_ok and fold2_ok and len(audit_rows)==380 else "AUDIT_FAILED"
    R.write_csv(RES/"audit_replay.csv",audit_rows);R.write_csv(RES/"audit_summary_replay.csv",summary_checks)
    payload={"status":status,"probe_models_expected":370,"probe_models_replayed":len([x for x in audit_rows if x["mode"]!="identity"]),
      "fold_identity_replays":sum(x["mode"]=="identity" for x in audit_rows),"per_model_replay_all_pass":allpass,
      "summary_rows_replayed":len(summary_checks),"summary_replay_pass":all(x["pass"] for x in summary_checks),"anonymous_gram_pass":gram_ok,
      "analysis_rng_unchanged":rng_ok,"locked_sources_unchanged":locked_ok,"residual_cap_pass":cap_ok,"fold2_scope_pass":fold2_ok,
      "base_checkpoint_count":10,"base_checkpoint_sha_pass":all(cache["folds"][str(f)]["checkpoint_sha256"]==json.loads((R.FR/f"fold{f:02d}.json").read_text())["checkpoint_sha256"] for f in range(1,11)),
      "failed_probe_models":sum(x["status"]!="PASS" for x in audit_rows),"elapsed_seconds":time.time()-start}
    R.atomic_json(RES/"audit_status.json",payload)
    manifest["status"]=status;manifest["task_status"]={"frozen_folds":10,"probe_fold_models_completed":370,"independent_replays":sum(x["mode"]!="identity" for x in audit_rows),"identity_folds":10}
    manifest["independent_audit"]=payload;manifest["audit_script_sha256"]=sha(EXP/"audit.py")
    manifest["results"]={str(p.relative_to(EXP)):sha(p) for p in RES.iterdir() if p.is_file()}
    R.atomic_json(manifest_path,manifest)
    if status!="COMPLETE_AUDITED":raise RuntimeError(f"independent audit failed; details in {RES/'audit_status.json'}")
    print(json.dumps(payload,indent=2),flush=True)

if __name__=="__main__":main()
