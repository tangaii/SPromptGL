"""Independent membership, Gaussian, gradient, checkpoint and reference audit."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from core import (CFG,CONFIG,EXPOSED,FOLDS,TASK_GROUPS,TASKS,CANONICAL,GROUP,ROOT,FULL_PATH,RESULTS,
                  MASKED_COMPONENTS,ADDED,ALL_ADDITIONS,OLD,PREVIOUS,engine,ext,
                  formats,f,full_audit,h,implementation,parent,prior,spectral,optimization,setup,paths,parent_preflight,
                  source_manifests)
# The expression above keeps the imported historical modules explicit and avoids local training imports.
METRICS=full_audit.METRICS
SOURCES={
 "E82.4.6.2":Path("/model/spromptgl/experiment/experiment82_4_6_2_g2_full_path_extension"),
 "E82.4.6.3":Path("/model/spromptgl/experiment/experiment82_4_6_3_darg_only_overlap"),
}
# Remove the harmless alias introduced in the import list for old Python versions.
try: del core
except NameError: pass

def rederive(indices,labels,supports,seed):
    return full_audit.rederive_pairs(indices,labels,supports,seed)

def membership_proof(ctx):
    task,fold,state,m,old=ctx["task_group"],ctx["fold"],ctx["state"],ctx["members"],ctx["old_members"]
    overlap=np.asarray(EXPOSED[fold],dtype=np.int64)
    h.require(len(m["train"])==CONFIG["native_rows"][str(fold)] and len(m["test"])==87
              and len(np.unique(m["train"]))==len(m["train"]) and
              np.array_equal(np.sort(np.intersect1d(m["train"],m["test"])),np.sort(overlap)),"Extended native membership differs")
    expected_legal=784 if task=="G1" else CONFIG["native_rows"][str(fold)]
    h.require(len(m["legal"])==expected_legal and len(np.unique(m["legal"]))==expected_legal,
              "SCMR legal size differs")
    if task=="G1":
        f.equal(m["legal"],old["legal"],"G1 original legal order")
        f.equal(np.sort(np.setdiff1d(m["train"],m["legal"])),np.sort(overlap),"G1 train/legal difference")
        f.equal(np.sort(np.flatnonzero(~f.array(ctx["module"].spectrum.allowed))),np.sort(overlap),"G1 allowed mask")
    else:
        f.equal(m["legal"],m["train"],"G2 legal equals extended native training")
        h.require(ctx["module"].spectrum.allowed.all().item(),"G2 residual mask unexpectedly disabled")
    for key in ("train","legal","test","weights","train_pairs","train_same"):
        f.equal(state[key],m[key],"Device "+key)
    labels=state["labels_np"]
    f.equal(ctx["labels"],labels[m["legal"]],"Legal labels")
    h.require(np.array_equal(m["train"][:len(old["train"])],old["train"])
              and np.array_equal(m["train"][len(old["train"]):],ALL_ADDITIONS[fold]),"Appended order differs")
    before=h.rng_sha(); anchors={}
    for name,part in (("old",old),("extended",m)):
        pairs,same,cats=rederive(part["train"],labels,state["supports"],532100+fold)
        f.equal(pairs,part["train_pairs"],name+" ordered relation pairs")
        f.equal(same,part["train_same"],name+" relation labels")
        f.equal(np.bincount(same,minlength=2),[2048,2048],name+" balanced relation count")
        f.equal(part["weights"],(len(part["train"])/(2*np.bincount(labels[part["train"]],minlength=2))).astype(np.float32),name+" class weights")
        anchors[name]=cats
    h.require(h.rng_sha()==before,"Independent pair audit changed RNG")
    reg=ctx["provenance"]["registration"]
    return {**reg,"task_group":task,"legal_rows":len(m["legal"]),"native_rows":len(m["train"]),
            "overlap_rows":len(overlap),"pairs_rederived":True,"anchors":anchors,"rng_unchanged":True}

def gaussian_proof(ctx):
    task,fold,ref,legal,labels=ctx["task_group"],ctx["fold"],ctx["reference"],ctx["members"]["legal"],ctx["state"]["labels_np"]
    f.equal(ref["legal"],legal,"Gaussian legal rows")
    f.equal(ref["labels"],labels[legal],"Gaussian labels")
    f.equal(ctx["module"].statistics.legal,legal,"EMA legal rows")
    h.require(np.isfinite(ref["mean"]).all() and np.isfinite(ref["second"]).all() and np.isfinite(ref["stable"]).all(),"Gaussian nonfinite")
    # OOF fold assignment and self-exclusion are checked from the stored bank masks.
    group=np.asarray(ref["group"]); mask=np.asarray(ref["mask"]).astype(bool)
    h.require(mask.shape[2]==len(legal) and np.all(~mask[group,:,np.arange(len(legal))]),"OOF self exclusion failed")
    h.require(np.all(mask.sum(-1)==np.asarray(ref["counts"])),"Gaussian counts mismatch")
    before=h.rng_sha()
    rebuilt=prior.rebuild_reference(fold,ctx["warm_cam"],ctx["original"],legal,ref["labels"].copy())
    h.require(h.sha(ref)==h.sha(rebuilt),"Gaussian deterministic rebuild differs")
    h.require(h.rng_sha()==before,"Gaussian rebuild changed global RNG")
    return {"task_group":task,"legal_rows":len(legal),"reference_sha":h.sha(ref),"rebuild_exact":True,
            "oof_self_exclusion":True,"counts":np.asarray(ref["counts"]).tolist(),"overlap_in_bank":task=="G2",
            "ema_legal_index_sha":h.sha(legal)}

def gradient_scope(run,ctx,dropout):
    state,module=ctx["state"],ctx["module"]; rows=np.asarray(EXPOSED[ctx["fold"]])
    with run.dropout_inventory() as calls:
        out,cap=run.forward(state,True,module)
    h.require(calls==dropout,"Dropout sequence changed in scope audit")
    native=F.nll_loss(out["logp"][state["train"]],state["y"][state["train"]],state["weights"])
    f.equal(native,out["cls"],"DARG native classification uses expanded training rows")
    gp,=torch.autograd.grad(out["cls"],out["logp"],retain_graph=True)
    h.require(float(gp[torch.as_tensor(rows,device=gp.device)].norm())>0,"Overlap DARG classification gradient absent")
    pair_logits=tuple(out["info"]["pair_logits"])
    pair= torch.stack([F.binary_cross_entropy_with_logits(x,state["train_same"]) for x in pair_logits]).mean()
    f.equal(pair,out["pair"],"DARG pair loss uses expanded relations")
    pgr=torch.autograd.grad(pair,pair_logits,retain_graph=True)
    pair_norm=0.0
    for row in rows:
        sel=(state["train_pairs"]==int(row)).any(dim=1)
        pair_norm += sum(float(x[sel].norm()) for x in pgr)
    h.require(pair_norm>0,"Overlap DARG relation gradient absent")
    aux,_=optimization.losses(CANONICAL,state["cam"].Outputlayer,cap,state["legal"],ctx["labels"],
                              module.spectrum.basis.to(cap["cam"]),ctx["hard_initial"])
    dg,=torch.autograd.grad(aux["full"],cap["details"]["delta"],retain_graph=True)
    if ctx["task_group"]=="G1":
        f.equal(dg[torch.as_tensor(rows,device=dg.device)],torch.zeros_like(dg[torch.as_tensor(rows,device=dg.device)]),
                "G1 overlap Full CE residual gradient zero")
        active=False
    else:
        active=bool(float(dg[torch.as_tensor(rows,device=dg.device)].norm())>0)
        h.require(active,"G2 overlap Full CE residual gradient absent")
    return {"dropout_exact":True,"native_cls_overlap_grad":float(gp[torch.as_tensor(rows,device=gp.device)].norm()),
            "scmr_full_delta_grad_overlap":float(dg[torch.as_tensor(rows,device=dg.device)].norm()),
            "g2_overlap_residual_gradient_active":active,"legal_auxiliary":list(aux)}

def residual_check(ctx,full,off,cap,offcap):
    rows=np.asarray(EXPOSED[ctx["fold"]]); test=ctx["members"]["test"]; selected=np.isin(test,rows)
    d=cap["details"]
    if ctx["task_group"]=="G1":
        for key in MASKED_COMPONENTS:
            f.equal(d[key][rows],np.zeros_like(f.array(d[key][rows])),"G1 exact "+key)
        f.equal(cap["enhanced"][rows],cap["cam"][rows],"G1 CAM exact")
        f.equal(full["probability"][selected],off["probability"][selected],"G1 overlap probability exact")
    else:
        h.require(cap["enhanced"].shape==cap["cam"].shape and np.isfinite(cap["enhanced"]).all().item(),"G2 enhanced CAM")
    held=~selected
    lengths={key:np.linalg.norm(f.f64(d[key][test]),axis=1) for key in ("delta","delta_parallel","delta_perp")}
    if ctx["task_group"]=="G2":
        h.require(np.any(lengths["delta"][selected]>0),"G2 overlap residuals all zero")
    h.require(np.any(lengths["delta"][held]>0),"Unmasked residual path inactive")
    return {"overlap_count":int(selected.sum()),"masked":ctx["task_group"]=="G1",
            "overlap_delta_nonzero":int(np.count_nonzero(lengths["delta"][selected])),
            "heldout_delta_nonzero":int(np.count_nonzero(lengths["delta"][held])),
            "overlap_delta_mean":float(lengths["delta"][selected].mean()),
            "heldout_delta_mean":float(lengths["delta"][held].mean()),
            "overlap_parallel_mean":float(lengths["delta_parallel"][selected].mean()),
            "overlap_perp_mean":float(lengths["delta_perp"][selected].mean())}

def preflight_task(task,fold,hashes):
    run=engine(task,fold); proof=prior.load_parent_module("_cum_audit_probe","audit.py"); proof.run=run
    device=h.archive.verify_device(); ctx=run.setup(task,fold,verify_spectrum=True)
    membership=membership_proof(ctx); gaussian=gaussian_proof(ctx)
    snap=parent_preflight.restored_payload(ctx,hashes,epoch=10,wait=0,best_epoch=0,best_correct=-1)
    _,start=run.evaluate(ctx)
    h.require(torch.equal(start["details"]["delta"],torch.zeros_like(start["cam"])), "Initial residual is nonzero")
    direct,drop,rng=parent_preflight.historic_signature(ctx,wrapped=False)
    h.require(len(drop)==6 and rng==h.hist_trace(fold)[11],"Historical initial RNG differs")
    run.restore(ctx,snap,hashes)
    result,diag,first_rng,_,_=run.transition(ctx,11,drop,inspect=True)
    h.require(first_rng==rng and np.isfinite(list(diag["losses"].values())).all(),"First transition invalid")
    first_step_sha=h.sha((result,diag,first_rng,h.sha(run.ema(ctx))))
    step_payload=parent_preflight.restored_payload(ctx,hashes,epoch=11)
    scope=gradient_scope(run,ctx,drop)
    run.restore(ctx,step_payload,hashes)
    witness=run.next_witness(ctx,12,drop)
    run.restore(ctx,step_payload,hashes)
    h.require(witness==run.next_witness(ctx,12,drop),"Next step replay differs")
    return {"status":"PASS","task_group":task,"fold":fold,"implementation":hashes,"device":device,
            "provenance":ctx["provenance"],"membership":membership,"gaussian":gaussian,
            "scope_gradient":scope,"dropout":drop,"first_step_sha":first_step_sha,
            "first_rng":first_rng,"next_step":witness,"initial_residual_zero":True,
            "peak_reserved_mib":torch.cuda.max_memory_reserved()/2**20}

def audit_task(task,fold,manifest):
    run=engine(task,fold); proof=prior.load_parent_module("_cum_audit_checkpoint","audit.py"); proof.run=run
    files=paths(task,fold); pre=h.read_json(files["preflight"]); taskmeta=h.read_json(files["task"])
    h.require(pre["status"]=="PASS" and taskmeta["status"]=="TRAINED" and pre["implementation"]==manifest["implementation"]
              and all(h.sha_file(files[k])==taskmeta[k+"_sha"] for k in ("best","latest","journal")),"Task artifact identity")
    best,latest=h.load(files["best"]),h.load(files["latest"]); h.require(latest["best_sha"]==taskmeta["best_sha"],"Latest best identity")
    ctx=run.setup(task,fold); h.require(ctx["provenance"]==pre["provenance"],"Reloaded scope differs")
    membership,gaussian=membership_proof(ctx),gaussian_proof(ctx)
    journal=proof.audit_journal(ctx,taskmeta,best,latest,pre)
    h.require(journal["ema_independent"] and journal["ema_bitwise_epochs"]==journal["replayed_epochs"],"EMA replay differs")
    records=[json.loads(x) for x in files["journal"].read_bytes().splitlines()]
    for rec in records:
        sr=rec["routing"]["scope_receipt"]; h.require(sr["task_group"]==task and sr["phases"]==["train","eval"],"Scope receipt differs")
    trace={r["epoch"]:r["rng"] for r in records}
    rng={"DARG":{"common_epochs":journal["archived_epochs"],"exact_phases":journal["rng_phase_comparisons"]},
         "E82.4.6-G2":full_audit.archived_rng_matches(parent.paths(CANONICAL,fold)["journal"],trace)}
    run.restore(ctx,best,manifest["implementation"])
    full,cap=proof.independent_eval(ctx); off,offcap=proof.independent_eval(ctx,"residual_off")
    anonymous,anoncap=proof.independent_eval(ctx,"anonymous_query")
    proof.match_eval(full,best["evaluation"],"Best")
    residual=residual_check(ctx,full,off,cap,offcap)
    rows=ctx["members"]["test"]; y=ctx["state"]["labels_np"][rows].copy()
    item={"full":full,"off":off,"anonymous":anonymous,"truth":y,"subjects":rows+1,"task_group":task,"fold":fold,
          "best_epoch":best["epoch"],"end_epoch":latest["epoch"],"best_sha":taskmeta["best_sha"],"best_path":str(files["best"]),
          "residual":{k:np.linalg.norm(f.f64(cap["details"][k][rows]),axis=1) for k in ("delta","delta_parallel","delta_perp")}}
    scope=gradient_scope(run,ctx,pre["dropout"])
    run.restore(ctx,best,manifest["implementation"]); fresh=run.setup(task,fold)
    latest_check=proof.check_latest(fresh,latest,manifest,taskmeta,pre)
    h.require(latest_check["exact"],"Latest next-step recovery differs")
    return item,{"status":"PASS","task_group":task,"fold":fold,"best_reloaded":True,"latest_reloaded":True,
                   "latest_next_exact":True,"membership":membership,"gaussian":gaussian,"residual":residual,
                   "scope_gradient":scope,"journal":journal,"rng":rng,"query":{"full":full,"anonymous":anonymous},
                   "best_epoch":best["epoch"],"end_epoch":latest["epoch"],"best_sha":taskmeta["best_sha"],
                   "latest_sha":taskmeta["latest_sha"],"journal_sha":taskmeta["journal_sha"]}

def historical_patient_csv():
    p=SOURCES["E82.4.6.2"]/ "results/patient_level.csv"
    with p.open(newline="") as s:return list(csv.DictReader(s))

def load_historical(method):
    # Reuse independently audited E82.4.6.2 patient probabilities for untouched folds.
    rows=historical_patient_csv(); out={}
    for fold in range(1,11):
        rr=[r for r in rows if int(r["fold"])==fold]
        y=np.asarray([int(r["true_label"]) for r in rr])
        subjects=np.asarray([int(r["patient_id_1based"]) for r in rr])
        if method=="E82.4.6.2":
            p=np.asarray([[float(r["p_asd_current_full"]),1-float(r["p_asd_current_full"])] for r in rr])
            pred=np.asarray([int(r["prediction_current_full"]) for r in rr])
        elif method=="E82.4.6.3":
            rr3=list(csv.DictReader((Path(CONFIG["prior_darg_only_directory"])/"results/patient_level.csv").open(newline="")))
            rr=[r for r in rr3 if int(r["fold"])==fold]
            p=np.asarray([[float(r["p_asd_full"]),1-float(r["p_asd_full"])] for r in rr])
            pred=np.asarray([int(r["prediction_full"]) for r in rr])
        else:
            raise ValueError(method)
        out[fold]={"full":{"probability":p,"prediction":pred,"logits":np.log(np.clip(p,1e-15,1))},
                   "off":{"probability":p,"prediction":pred,"logits":np.log(np.clip(p,1e-15,1))},
                   "truth":y,"subjects":subjects,"best_epoch":int(rr[0].get("best_epoch",0) or 0),
                   "best_sha":rr[0].get("best_sha_current",rr[0].get("best_sha","")),"best_path":"archived"}
    return out

def audit(manifest):
    h.require(implementation()==manifest["implementation"],"Frozen implementation differs")
    source_manifests()
    current, checks = {}, {}
    for task,fold in TASKS:
        item, certificate=audit_task(task,fold,manifest)
        current[(task,fold)]=item
        checks[f"{task}_f{fold:02d}"]=certificate
    historical=load_historical("E82.4.6.2")
    previous=load_historical("E82.4.6.3")
    return historical, previous, current, checks
