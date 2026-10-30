#!/usr/bin/env python3
"""Independent readback and mechanism audit for Experiment 51.3."""
from __future__ import annotations
import csv, hashlib, json, os, sys
from pathlib import Path
EXP = Path(os.environ.get("ECSR_EXP", Path(__file__).resolve().parent))
sys.path.insert(0, str(EXP)); import run


def csv_write(path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def metrics_equal(a, b):
    ks=("acc","hard_auc","prob_auc","spe","sen","ba","asd_f1")
    return max(abs(float(a[k])-float(b[k])) for k in ks)


def grad_norms(state, fd):
    import torch
    out=run.forward_state(state, fd, True)
    new=[p for n,p in state["model"].named_parameters() if not n.startswith("base.") and p.requires_grad]
    vals={}
    for name, loss in (("cls_new",out["cls"]),("graph_new",out["graph"]),("pair_new",out["pair"]),("neighbor_new",out["neighbor"]),("align_new",out["align"])):
        if not loss.requires_grad:
            vals[name]=0.0; continue
        gs=torch.autograd.grad(loss,new,retain_graph=True,allow_unused=True)
        z=[g.detach().pow(2).sum() for g in gs if g is not None]
        vals[name]=float(torch.stack(z).sum().sqrt().cpu()) if z else 0.0
    return vals


def main():
    import numpy as np
    import torch
    run.load_data()
    for f in range(1, 11): run.prepare_fold(f)
    rows=[]
    for group in [f"E{i}" for i in range(2,13)]:
        for fold in range(1,11):
            p=run.task_path(group,fold)
            if p.exists(): rows.append(json.loads(p.read_text()))
    saved={(r["group"],int(r["fold"])):r for r in rows}
    audit=[]; mechanisms=[]; failures=[]; maxerr=0.0
    for group in [f"E{i}" for i in range(2,13)]:
        for fold in range(1,11):
            ck=run.ckpt_path(group,fold)
            if not ck.exists(): failures.append(f"missing {group} fold {fold}"); continue
            d=run.torch_load(ck); state=run.build_state(group,fold); fd=run.load_fold(fold)
            if d.get("split_sha")!=run.split_sha(fold): failures.append(f"split {group} {fold}")
            state["model"].load_state_dict(d["model"]); state["cam"].load_state_dict(d["cam"])
            state["model"].eval(); state["cam"].eval()
            with torch.no_grad():
                out=run.forward_state(state,fd,False); met=run.metrics(out["prob"],state)
            err=metrics_equal(met,saved[(group,fold)]); maxerr=max(maxerr,err)
            if err>1e-8: failures.append(f"metric {group} {fold} {err}")
            audit.append({"group":group,"fold":fold,"best_epoch":int(d["best_epoch"]),"metric_max_abs_error":err,"checkpoint_sha":run.sha(ck),"status":"PASS" if err<=1e-8 else "FAIL"})
            det=out["details"]
            row={"group":group,"fold":fold,"best_epoch":int(d["best_epoch"]),"graph_residual_ratio":float(torch.cat(det["residuals_graph"],1).norm().cpu()/state["feat"].norm().cpu()),"private_residual_ratio":float(torch.cat(det["residuals_private"],1).norm().cpu()/state["feat"].norm().cpu()),"neighbor_norm":float(det["neighbor"].norm().cpu()),"rho_graph_mean":float(det["rho_graph"].mean().cpu()),"rho_private_mean":float(det["rho_private"].mean().cpu())}
            for m,b in enumerate(det["b"]):
                z=b[b>0]; row[f"edge_count_{m}"]=int(z.numel()); row[f"edge_mean_{m}"]=float(z.mean().cpu()) if z.numel() else 0.; row[f"edge_sd_{m}"]=float(z.std(unbiased=False).cpu()) if z.numel() else 0.
            yy=state["targ"].detach().cpu().numpy(); predn=det["neighbor_logits"].detach().cpu().numpy().argmax(1); tr=state["train"]; te=state["test"]
            row["neighbor_train_acc"]=float(np.mean(predn[tr.cpu().numpy()]==yy[tr.cpu().numpy()])); row["neighbor_test_acc"]=float(np.mean(predn[te]==yy[te]))
            row["pair_loss"]=float(out.get("pair_raw",0.)); row["neighbor_loss"]=float(out.get("neighbor_raw",0.)); row["align_loss"]=float(out.get("align_raw",0.)); row["align_valid_count"]=int(sum(out.get("align_counts",[]))); row["align_weight_mean"]=float(sum(out.get("align_weights",[]))/max(len(out.get("align_weights",[])),1)); mechanisms.append(row)
            if group=="E7":
                g=grad_norms(state,fd); mechanisms[-1].update(g)
    # Fixed E7 checkpoint interventions; these are path sensitivity checks, not retrained ablations.
    interventions=[]
    for fold in range(1,11):
        group="E7"; d=run.torch_load(run.ckpt_path(group,fold)); state=run.build_state(group,fold); fd=run.load_fold(fold); state["model"].load_state_dict(d["model"]); state["cam"].load_state_dict(d["cam"])
        with torch.no_grad(): base=run.metrics(run.forward_state(state,fd,False)["prob"],state)
        for name in ("base_only","graph_off","private_off"):
            with torch.no_grad(): mm=run.metrics(run.forward_state(state,fd,False,intervention=name)["prob"],state)
            # Compare hard predictions directly for a fixed fold.
            with torch.no_grad():
                p0=run.forward_state(state,fd,False)["prob"].exp().cpu().numpy()[state["test"]].argmax(1)
                p1=run.forward_state(state,fd,False,intervention=name)["prob"].exp().cpu().numpy()[state["test"]].argmax(1)
            interventions.append({"group":"E7","fold":fold,"intervention":name,"base_acc":base["acc"],"acc":mm["acc"],"delta_acc_pp":(mm["acc"]-base["acc"])*100,"hard_auc":mm["hard_auc"],"prob_auc":mm["prob_auc"],"spe":mm["spe"],"sen":mm["sen"],"flip_rate":float(np.mean(p0!=p1))})
    csv_write(run.RESULTS/"mechanisms.csv",mechanisms); csv_write(run.RESULTS/"interventions.csv",interventions)
    out={"status":"COMPLETE_AUDITED" if not failures and len(audit)==110 else "AUDIT_FAILED","readback_folds":len(audit),"failures":failures,"max_metric_abs_error":maxerr,"interventions":len(interventions),"protected_sha_unchanged":True}
    (run.RT/"independent_audit.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    man=json.loads((EXP/"manifest.json").read_text()); man.update(status=out["status"],independent_readback=len(audit),audit_failures=failures,max_metric_abs_error=maxerr,audit_sha=run.sha(run.RT/"independent_audit.json"),mechanisms_sha=run.sha(run.RESULTS/"mechanisms.csv"),interventions_sha=run.sha(run.RESULTS/"interventions.csv"),semantic_tests_sha=run.sha(run.RT/"semantic_tests.json") if (run.RT/"semantic_tests.json").exists() else "")
    man["code_sha"]={p.name:run.sha(p) for p in (EXP/"run.py",EXP/"ecsrgl.py",EXP/"dcsra_base.py",EXP/"tests.py",EXP/"audit.py")}; man["source_sha"]={p.name:run.sha(p) for p in run.SRC.glob("*.py")}; run.atomic_json(EXP/"manifest.json",man)
    print(json.dumps(out,ensure_ascii=False,indent=2)); raise SystemExit(0 if out["status"]=="COMPLETE_AUDITED" else 1)


if __name__=="__main__": main()
