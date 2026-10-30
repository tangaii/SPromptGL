#!/usr/bin/env python3
"""Fixed-checkpoint interventions and graph/message mechanism diagnostics."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP))
import run

INTERVENTIONS={
    "original": {},
    "edge_off": {"edge_off":True},
    "prop_off": {"prop_off":True},
    "self_only": {"self_only":True,"prop_off":True},
    "all_new_cross_patient_off": {"edge_off":True,"prop_off":True},
    "swap_positive_negative": {"swap":True},
    "q_half": {"q_half":True},
    "wrong_graph": {"wrong_graph":True},
    "negative_off": {"negative_off":True},
}

def main():
    rows=[]; flips=[]; diag=[]
    for fold in range(1,11):
        ck=run.torch_load(run.ckpt_path("B5",fold)); state=run.build_state("B5",fold); fd=run.load_fd(fold)
        state["base"].load_state_dict(ck["base"]); state["cam"].load_state_dict(ck["cam"]); state["d2"].load_state_dict(ck["d2"])
        outs={}
        for name,iv in INTERVENTIONS.items():
            with torch.no_grad(): outs[name]=run.forward_state(state,fd,False,iv); m=run.metrics(outs[name]["logp"],state)
            rows.append({"fold":fold,"intervention":name,**m,"delta_norm":float(outs[name]["delta"].norm().cpu()),"fused_norm":float(outs[name]["fused"].norm().cpu()),"edge_change":float(outs[name]["details"].get("edge_change",0.0)),"positive_norm":float(outs[name]["details"].get("positive_norm",0.0)),"negative_norm":float(outs[name]["details"].get("negative_norm",0.0)),"rho":float(outs[name]["details"].get("rho",0.0)),"kappa":float(outs[name]["details"].get("kappa",0.0))})
        basep=outs["original"]["logp"].exp().argmax(1); test=state["test"]
        for name,out in outs.items():
            p=out["logp"].exp().argmax(1); flips.append({"fold":fold,"intervention":name,"flips":int((p[test]!=basep[test]).sum().cpu()),"flip_rate":float((p[test]!=basep[test]).float().mean().cpu())})
        orig=outs["original"]; d=orig["details"]
        diag.append({"fold":fold,"edge_change":float(d.get("edge_change",0.0)),"positive_norm":float(d.get("positive_norm",0.0)),"negative_norm":float(d.get("negative_norm",0.0)),"delta_norm":float(orig["delta"].norm().cpu()),"fused_e1_norm":float(orig["fused_e1"].norm().cpu()),"relative_delta_over_e1":float(orig["delta"].norm().cpu()/orig["fused_e1"].norm().clamp_min(1e-8).cpu()),"q_mean":float(d["q"].mean().cpu()),"q_sd":float(d["q"].std().cpu()),"site_loss":float(orig["site_loss"].cpu())})
        del state; torch.cuda.empty_cache()
    def write(path, data):
        with path.open("w",newline="") as f:
            fields=list(data[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
    write(EXP/"results/interventions.csv",rows); write(EXP/"results/intervention_flips.csv",flips); write(EXP/"results/mechanism.csv",diag)
    print(json.dumps({"rows":len(rows),"folds":10,"interventions":len(INTERVENTIONS)},indent=2))

if __name__=="__main__": main()
