#!/usr/bin/env python3
"""Independent checkpoint readback, paired comparisons and APCR-off tests."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import run

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True);fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def read_csv(path):
    with path.open(newline="") as f:return list(csv.DictReader(f))

def state_load(a,bundle):run.context()["method"].load_bundle(a,bundle)

def main():
    c=run.context();t=c["torch"];m=c["method"];x,y,folds,mapping=run.data();fold_rows=[];interventions=[]
    for gid in (1,2):
        for fold,(tr,te) in enumerate(folds,1):
            task,bestpath,latest=run.task_paths(gid,fold)
            rec=json.loads(task.read_text());ck=run.load(bestpath)
            assert rec["status"]=="COMPLETE" and rec["signature"]==run.code_signature()
            assert rec["best_sha"]==run.sha(bestpath) and ck["signature"]==run.code_signature()
            assert rec["split_sha"]==run.fold_hash(tr,te) and ck["training_group"]==6
            a=run.initialize_agent(gid,fold,x,y,tr,te,mapping);state_load(a,ck["model"])
            assert abs(float(a.APCR.alpha().detach())-float(ck["diagnostics"]["alpha_best"]))<1e-8
            if gid==1:assert abs(float(a.APCR.alpha().detach())-.0022)<1e-7 and not a.APCR.alpha_raw.requires_grad
            else:assert a.APCR.alpha_raw.requires_grad
            with t.no_grad():full=m.evaluate(a,6,c["utils"])
            met=m.metrics(full,y,te);pred=full["logp"][te].argmax(1).detach().cpu().numpy()
            psha=hashlib.sha256(pred.tobytes()).hexdigest();assert psha==rec["prediction_sha256"]
            for k in ("acc","hard_auc","probability_auc","spe","sen","asd_tp","asd_fn","td_fp","td_tn"):
                assert abs(float(met[k])-float(rec[k]))<1e-12,(gid,fold,k,met[k],rec[k])
            intervention={"group_id":gid,"fold":fold,"condition":"FULL"};intervention.update(met)
            alpha_raw=a.APCR.alpha_raw.detach().clone()
            with t.no_grad():a.APCR.alpha_raw.fill_(-100.)
            with t.no_grad():off=m.evaluate(a,6,c["utils"])
            offmet=m.metrics(off,y,te);offpred=off["logp"][te].argmax(1).detach().cpu().numpy()
            raw_ratio=float(full["residual"].norm()/(full["base"].norm()+1e-8));effective_ratio=float((full["alpha"]*full["residual"]).norm()/(full["base"].norm()+1e-8))
            intervention={**intervention,"alpha_off_acc":offmet["acc"],"alpha_off_hard_auc":offmet["hard_auc"],"alpha_off_prob_auc":offmet["probability_auc"],"alpha_off_spe":offmet["spe"],"alpha_off_sen":offmet["sen"],"delta_acc":offmet["acc"]-met["acc"],"delta_hard_auc":offmet["hard_auc"]-met["hard_auc"],"prediction_flips":int(np.sum(offpred!=pred)),"new_correct":int(np.sum((offpred==y[te])&(pred!=y[te]))),"new_wrong":int(np.sum((offpred!=y[te])&(pred==y[te]))),"mean_abs_probability_change":float((off["logp"][te].exp()-full["logp"][te].exp()).abs().mean()),"apcr_residual_over_gfr":raw_ratio,"effective_apcr_residual_over_gfr":effective_ratio,"message_norms":";".join(f"{float(v.norm()):.9g}" for v in full["deltas"]),"gate_means":";".join(f"{float(v.mean()):.9g}" for v in full["gates"])}
            interventions.append(intervention)
            diag=ck["diagnostics"]
            fold_rows.append({"group_id":gid,"group":rec["group"],"fold":fold,"best_epoch":rec["best_epoch"],"epochs_executed":rec["epochs_executed"],"alpha_init":diag["alpha_init"],"alpha_best":diag["alpha_best"],"alpha_final_training":diag["alpha_final_training"],"alpha_trainable":diag["alpha_trainable"],"prediction_sha256":psha,"best_sha":rec["best_sha"],"independent_readback":True,**met})
            del a
    write_csv(run.RES/"fold_metrics.csv",fold_rows);write_csv(run.RES/"apcr_off_interventions.csv",interventions)
    summary=[]
    for gid in (1,2):
        rows=[r for r in fold_rows if r["group_id"]==gid];line={"group_id":gid,"group":rows[0]["group"]}
        for metric in ("acc","hard_auc","spe","sen","probability_auc"):
            arr=np.asarray([r[metric] for r in rows],float);line[metric+"_mean_pct"]=100*arr.mean();line[metric+"_sd_pct"]=100*arr.std(ddof=0)
        alpha=np.asarray([r["alpha_best"] for r in rows],float);line["alpha_best_mean"]=alpha.mean();line["alpha_best_sd"]=alpha.std(ddof=0)
        summary.append(line)
    write_csv(run.RES/"summary.csv",summary)
    paired=[]
    refs={"E51_E1":[json.loads(p.read_text()) for p in sorted((run.RT/"e51_reference_tasks").glob("E1_f*.json"))],"E48_5":read_csv(run.RT/"e48_5_fold_metrics.csv")}
    for gid,label in ((1,"fixed"),(2,"learnable")):
        target=[r for r in fold_rows if r["group_id"]==gid]
        reference_sets={"E51_E1":[r for r in refs["E51_E1"] if r.get("group")=="E1"],"E48_5_E1":[r for r in refs["E48_5"] if r.get("group_id")=="1"],"E48_5_E2":[r for r in refs["E48_5"] if r.get("group_id")=="2"],"B0_B":[r for r in read_csv(run.SRC/"e51"/"results"/"fold_metrics.csv") if r.get("group")=="E0"]}
        for name,ref in reference_sets.items():
            if name.startswith("E48_5"): ref=[r for r in ref if int(r["group_id"])==int(name[-1])]
            refmap={int(r["fold"]):r for r in ref}
            if len(refmap)!=10:continue
            for metric in ("acc","hard_auc","spe","sen","probability_auc"):
                ref_metric = metric if metric != "probability_auc" else "prob_auc"
                diffs=np.asarray([float(a[metric])-float(refmap[int(a["fold"])].get(ref_metric,refmap[int(a["fold"])].get("probability_auc"))) for a in target])
                paired.append({"group":f"E{gid}","reference":name,"metric":metric,"delta_mean_pp":float(diffs.mean()*100),"wins":int((diffs>1e-12).sum()),"ties":int((np.abs(diffs)<=1e-12).sum()),"losses":int((diffs< -1e-12).sum())})
    write_csv(run.RES/"paired_comparisons.csv",paired)
    manifest=json.loads((run.HERE/"manifest.json").read_text()) if (run.HERE/"manifest.json").exists() else {}
    protected=manifest.get("protected_sha256_before",{});after={p:run.sha(p) for p in protected}
    assert protected==after,"protected root inputs changed"
    tasks=[json.loads(p.read_text()) for p in sorted((run.RT/"tasks").glob("E*_f*.json"))]
    assert len(tasks)==20 and all(tk["status"]=="COMPLETE" for tk in tasks)
    manifest.update({"status":"COMPLETE_AUDITED","tasks":{"total":20,"complete":20},"trained_folds":20,"independent_readback":"20/20_PASS","protected_sha256_after":after,"protected_sha_unchanged":True,"code_signature":run.code_signature(),"delivery_status":"SCRATCH_STAGED_DATA_EIO","results_sha256":{str(p.relative_to(run.HERE)):run.sha(p) for p in sorted(run.RES.iterdir()) if p.is_file()},"handoff_status":"HANDOFF_PENDING"})
    run.atomic_json(run.HERE/"manifest.json",manifest)
    print(json.dumps({"status":"AUDIT_PASS","tasks":20,"summary":summary,"paired":paired,"delivery_status":manifest["delivery_status"]},indent=2))

RT48_5=run.HERE/".runtime"

if __name__=="__main__":main()
