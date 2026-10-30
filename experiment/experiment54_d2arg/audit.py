#!/usr/bin/env python3
"""Independent best-checkpoint readback audit for Experiment 54."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import torch
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
import run


def main():
    run.ensure_dirs(); rows=[]; start=time.time()
    for group in run.GROUPS:
        for fold in range(1,11):
            ck=run.torch_load(run.ckpt_path(group,fold)); state=run.build_state(group,fold); fd=run.load_fd(fold)
            state["base"].load_state_dict(ck["base"],strict=True); state["cam"].load_state_dict(ck["cam"],strict=True)
            if state["d2"] is not None: state["d2"].load_state_dict(ck["d2"],strict=True)
            with torch.no_grad(): out=run.forward_state(state,fd,False); met=run.metrics(out["logp"],state)
            diffs={k:float(met[k])-float(ck["metrics"][k]) for k in ("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1")}
            rows.append({"group":group,"fold":fold,"checkpoint_acc":ck["metrics"]["acc"],"audit_acc":met["acc"],"max_metric_abs_diff":max(abs(v) for v in diffs.values()),"confusion_tp_asd":met["asd_tp"],"confusion_fn_asd":met["asd_fn"],"confusion_fp_td":met["td_fp"],"confusion_tn_td":met["td_tn"],"acc_ok":max(abs(v) for v in diffs.values())<1e-8,"best_epoch":ck["best_epoch"],"fused_norm":float(out["fused"].norm().cpu()),"delta_norm":float(out["delta"].norm().cpu()),"edge_change":float(out["details"].get("edge_change",0.0)),"positive_norm":float(out["details"].get("positive_norm",0.0)),"negative_norm":float(out["details"].get("negative_norm",0.0)),"site_loss":float(out["site_loss"].detach().cpu())})
            del state
            torch.cuda.empty_cache()
    path=EXP/"results/checkpoint_audit.csv"
    with path.open("w",newline="") as f:
        fields=list(rows[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    result={"status":"PASS" if all(r["acc_ok"] for r in rows) else "FAIL","rows":len(rows),"passed":sum(r["acc_ok"] for r in rows),"max_abs_diff":max(r["max_metric_abs_diff"] for r in rows),"wall_seconds":time.time()-start}
    (EXP/".runtime/checkpoint_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(2)

if __name__=="__main__": main()
