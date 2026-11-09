#!/usr/bin/env python3
"""Independent E75 checkpoint reconstruction, prediction audit, and interventions."""
from __future__ import annotations
import argparse, csv, json, multiprocessing as mp, os, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import run as R

HERE, RESULTS, METRICS = R.HERE, R.RESULTS, R.METRICS
GROUPS = tuple(R.GROUPS)


def intervention_row(group, fold, name, base, alt, out0, out1, labels, test):
    import numpy as np
    y = np.asarray(labels.detach().cpu(), dtype=np.int64)[np.asarray(test, dtype=np.int64)]
    p0, p1 = np.asarray(base["pred"]), np.asarray(alt["pred"])
    c0, c1 = p0 == y, p1 == y
    prob0, prob1 = np.asarray(base["prob"]), np.asarray(alt["prob"])
    h0 = out0["dcer"]["h_prime"].detach().float()
    h1 = out1["dcer"]["h_prime"].detach().float()
    row = {"group": group, "fold": fold, "intervention": name,
           "hard_flip_count": int((p0 != p1).sum()),
           "correct_to_wrong": int((c0 & ~c1).sum()),
           "wrong_to_correct": int((~c0 & c1).sum()),
           "mean_abs_asd_probability_change": float(np.abs(prob0[:, 0] - prob1[:, 0]).mean()),
           "representation_relative_change": float((h0-h1).norm() / h0.norm().clamp_min(1e-8))}
    for metric in METRICS:
        row[f"{metric}_base"] = float(base[metric])
        row[f"{metric}_intervention"] = float(alt[metric])
        row[f"{metric}_delta"] = float(alt[metric] - base[metric])
    return row


def audit_fold(group, fold):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from dcer_v2 import tensor_sha

    task_path, checkpoint_path, _, _ = R.task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    payload = R.torch_load(checkpoint_path)
    ckpt_sha = R.sha_file(checkpoint_path)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != ckpt_sha:
        raise RuntimeError(f"task/checkpoint SHA mismatch {group} fold={fold}")
    if payload.get("group") != group or int(payload.get("fold", -1)) != fold:
        raise RuntimeError(f"checkpoint identity mismatch {group} fold={fold}")
    if payload.get("config_sha256") != R.sha_file(HERE/"config.json") or payload.get("source_sha256") != R.source_inventory():
        raise RuntimeError(f"source/config SHA mismatch {group} fold={fold}")
    if payload.get("warmup_sha256") != R.sha_file(R.E61_WARMUPS/f"fold{fold:02d}.pt"):
        raise RuntimeError(f"warmup SHA mismatch {group} fold={fold}")

    e61, state, warmup, _, gate = R.build_fold_state(fold)
    for key in ("split_sha","pair_sha","warmup_pair_sha","anchor_sha","base_train_sha","effective_train_sha",
                "initial_shared_sha","initial_gfr_sha","initial_cam_sha","relation_core_sha","initial_rng_sha"):
        if payload.get(key) != state.get(key):
            raise RuntimeError(f"historical fold parity mismatch {group} fold={fold} field={key}")
    if payload.get("warmup_rng_sha") != warmup["rng_sha"] or payload.get("gate_initialization") != gate:
        raise RuntimeError(f"warmup/gate parity mismatch {group} fold={fold}")
    expected_overlap = [377,407] if fold == 2 else []
    if payload.get("train_test_overlap_1based") != expected_overlap:
        raise RuntimeError(f"unexpected train/test overlap {group} fold={fold}")

    state["model"].load_state_dict(payload["model"], strict=True)
    state["cam"].load_state_dict(payload["cam"], strict=True)
    banks, _ = R.text_banks()
    dcer, dcer_init = R.init_dcer(group, fold, state["y"].device, banks[group])
    if payload.get("dcer_initialization") != dcer_init:
        raise RuntimeError(f"DCER initialization mismatch {group} fold={fold}")
    dcer.load_state_dict(payload["dcer"], strict=True)
    state["dcer"] = dcer
    if tensor_sha(dcer) != task.get("dcer_final_parameter_sha256") or tensor_sha(dcer, True) != task.get("dcer_final_state_sha256"):
        raise RuntimeError(f"DCER state SHA mismatch {group} fold={fold}")
    opt_dcer = torch.optim.AdamW(dcer.parameters(), lr=0.001, weight_decay=0.001)
    opt_dcer.load_state_dict(payload["optimizer_dcer"])
    for opt, name in ((state["opt_base"],"optimizer_base"),(state["opt_cam"],"optimizer_cam"),(state["opt_added"],"optimizer_added")):
        opt.load_state_dict(payload[name])
    optimizer_hash = R.tensor_state_sha([payload["optimizer_base"],payload["optimizer_cam"],payload["optimizer_added"],payload["optimizer_dcer"]])
    if optimizer_hash != payload.get("optimizer_sha256") or optimizer_hash != task.get("optimizer_sha256"):
        raise RuntimeError(f"optimizer SHA mismatch {group} fold={fold}")
    opt_ids = [[id(p) for g in opt.param_groups for p in g["params"]] for opt in (state["opt_base"],state["opt_cam"],state["opt_added"],opt_dcer)]
    flat = [p for group_ids in opt_ids for p in group_ids]
    if len(flat) != len(set(flat)):
        raise RuntimeError(f"duplicate optimizer parameters {group} fold={fold}")

    test = np.asarray(payload["test_indices"],dtype=np.int64)
    if not np.array_equal(test,np.asarray(state["test_np"],dtype=np.int64)):
        raise RuntimeError(f"test indices mismatch {group} fold={fold}")
    yall = np.asarray(state["labels_np"],dtype=np.int64)
    if not np.array_equal(np.asarray(payload["test_labels"],dtype=np.int64),yall[test]):
        raise RuntimeError(f"test label encoding mismatch {group} fold={fold}")
    state["model"].eval(); state["cam"].eval(); dcer.eval()
    with torch.no_grad():
        out0 = R.forward_with_dcer(e61,state,dcer,group,training=False)
        metrics0 = e61.load_base().classification_metrics(out0["logp"],state["test"],state["y"])
    if out0["dcer_call_count"] != 1 or tuple(out0["dcer"]["h_prime"].shape) != (871,144):
        raise RuntimeError(f"CAM/OutputLayer call/shape mismatch {group} fold={fold}")
    for key in METRICS:
        if abs(float(metrics0[key])-float(payload["metrics"][key]))>1e-10:
            raise RuntimeError(f"independent metric mismatch {group} fold={fold} {key}")
    if not np.array_equal(metrics0["pred"],payload["test_prediction"]) or not np.allclose(metrics0["prob"],payload["test_probability"],atol=1e-8,rtol=0):
        raise RuntimeError(f"prediction readback mismatch {group} fold={fold}")

    interventions=[]
    intervention_names=["direct_off","semantic_neutral"]
    if R.GROUPS[group]["common_response_suppression"]: intervention_names.append("common_off")
    if R.GROUPS[group]["validity"]: intervention_names.append("validity_off")
    for name in intervention_names:
        with torch.no_grad():
            out1=R.forward_with_dcer(e61,state,dcer,group,training=False,intervention=name)
            metrics1=e61.load_base().classification_metrics(out1["logp"],state["test"],state["y"])
        interventions.append(intervention_row(group,fold,name,metrics0,metrics1,out0,out1,state["y"],state["test_np"]))

    teacher=None
    if R.GROUPS[group]["validity"]:
        teacher=R.token_mask_teacher(state["cam"].Outputlayer,out0["cam_input"],state)
    mechanism=R.dcer_summary(out0,group,state,teacher_target=teacher)
    att=out0["dcer"]["attention"].detach().float()
    scores=out0["dcer"]["scores"].detach().float()
    mechanism.update({
        "fold":fold,
        "attention_pair_abs_difference":float((att[:,0]-att[:,1]).abs().mean()),
        "attention_pair_cosine":float(F.cosine_similarity(att[:,0],att[:,1],dim=-1).mean()),
        "attention_max_token_mean_asd":float(att[:,0].max(-1).values.mean()),
        "attention_max_token_mean_td":float(att[:,1].max(-1).values.mean()),
        "score_sd_asd":float(scores[:,0].std(unbiased=False)),
        "score_sd_td":float(scores[:,1].std(unbiased=False)),
        "train_examples":len(state["train_np"]),"test_examples":len(state["test_np"]),
        "best_epoch_1based":payload["best_epoch_1based"],"epochs_executed":payload["epochs_executed"],
        "train_gradient_gamma":payload["diagnostics"]["train_gradient_norms"]["dcer_gamma"],
        "train_gradient_text":payload["diagnostics"]["train_gradient_norms"]["dcer_text"],
        "train_gradient_key":payload["diagnostics"]["train_gradient_norms"]["dcer_key"],
        "train_gradient_evidence":payload["diagnostics"]["train_gradient_norms"]["dcer_evidence"],
        "dcer_update_norm_at_best":payload["diagnostics"]["dcer_update_norm"],
        "checkpoint_sha256":ckpt_sha,"optimizer_sha256":optimizer_hash,
        "train_test_overlap_1based":json.dumps(expected_overlap)})
    if teacher is not None:
        v=out0["dcer"]["validity"].index_select(0,state["train"]).detach().reshape(-1)
        t=teacher.detach().reshape(-1); order=torch.argsort(v); n=len(order); q=max(1,n//4)
        low=t.index_select(0,order[:q]).mean(); high=t.index_select(0,order[-q:]).mean()
        mechanism["validity_low_quartile_teacher_mean"]=float(low)
        mechanism["validity_high_quartile_teacher_mean"]=float(high)
        mechanism["validity_high_minus_low_teacher"]=float(high-low)

    historical=json.loads((R.E69_DIR/".runtime/tasks"/f"E69.3-G2_f{fold:02d}.json").read_text())
    trace=payload.get("base_rng_trace",[])
    ref={int(x["epoch"]):x for x in historical.get("rng_trace",[])}
    checked=[x for x in trace if int(x["epoch"]) in ref]
    matched=sum(all(x.get(k)==ref[int(x["epoch"])].get(k) for k in ("rng_before","rng_after_train","rng_after_eval")) for x in checked)
    mechanism["rng_common_epochs_checked"]=len(checked); mechanism["rng_common_epochs_exact_match"]=matched
    if not checked or matched!=len(checked):
        raise RuntimeError(f"historical RNG parity failed {group} fold={fold}: {matched}/{len(checked)}")
    audit={"group":group,"fold":fold,"checkpoint_sha256":ckpt_sha,
           "best_epoch_1based":payload["best_epoch_1based"],"epochs_executed":payload["epochs_executed"],
           "metrics_recomputed":"PASS","predictions_recomputed":"PASS","optimizer_state":"PASS",
           "initialization":"PASS","warmup_parity":"PASS","split_pair_parity":"PASS",
           "train_test_overlap_1based":json.dumps(expected_overlap),"formal_cam_outputlayer_calls":out0["dcer_call_count"],
           "rng_epochs_checked":len(checked),"rng_epochs_exact_match":matched,"checkpoint_audit":"PASS"}
    fold_metrics={"group":group,"fold":fold,**R.clean_metric_dict(metrics0),
                  "best_epoch_1based":payload["best_epoch_1based"],"epochs_executed":payload["epochs_executed"],
                  "checkpoint_sha256":ckpt_sha}
    return {"audit":audit,"fold_metrics":fold_metrics,"mechanism":mechanism,"interventions":interventions,"task":task}


def _historical_e69():
    rows=list(csv.DictReader((R.E69_DIR/"results/fold_metrics.csv").open(encoding="utf-8")))
    rows=[x for x in rows if x.get("group")=="E69.3-G2"]
    if len(rows)!=10: raise RuntimeError("E69.3-G2 read-only reference is not ten folds")
    return [{"group":"E69.3-G2 historical","fold":int(x["fold"]),**{m:float(x[m]) for m in METRICS}} for x in rows]


def _summary(rows,historical):
    import numpy as np
    result=[]
    groups=[("E69.3-G2 historical",historical)]+[(g,[r for r in rows if r["group"]==g]) for g in GROUPS]
    for group,items in groups:
        for scope,folds in (("all_10",None),("exclude_fold2_descriptive_9",set(range(1,11))-{2})):
            selected=[r for r in items if folds is None or int(r["fold"]) in folds]
            row={"group":group,"scope":scope,"n_folds":len(selected)}
            for m in METRICS:
                values=np.asarray([float(r[m]) for r in selected])
                row[m+"_mean_percent"]=float(values.mean()*100)
                row[m+"_population_sd_percent"]=float(values.std(ddof=0)*100)
            result.append(row)
    return result


def _paired(rows,historical):
    import numpy as np
    maps={g:{int(r["fold"]):r for r in rows if r["group"]==g} for g in GROUPS}
    maps["E69.3-G2 historical"]={int(r["fold"]):r for r in historical}
    pairs=[(g,"E69.3-G2 historical") for g in GROUPS]+[("G2","G1"),("G3","G2"),("G3","G4")]
    result=[]
    for a,b in pairs:
        for scope,subset in (("all_10",set(range(1,11))),("exclude_fold2_descriptive_9",set(range(1,11))-{2})):
            folds=sorted(subset & maps[a].keys() & maps[b].keys())
            for m in METRICS:
                d=np.asarray([maps[a][f][m]-maps[b][f][m] for f in folds])
                result.append({"group_a":a,"group_b":b,"scope":scope,"metric":m,"n_folds":len(folds),
                    "mean_delta_pp":float(d.mean()*100),"population_sd_delta_pp":float(d.std(ddof=0)*100),
                    "wins":int((d>1e-12).sum()),"ties":int((np.abs(d)<=1e-12).sum()),"losses":int((d< -1e-12).sum()),
                    "fold_delta_pp":json.dumps({str(f):float((maps[a][f][m]-maps[b][f][m])*100) for f in folds})})
    return result


def _mean_text(summary,group,metric,scope="all_10"):
    r=next(x for x in summary if x["group"]==group and x["scope"]==scope)
    return f"{r[metric+'_mean_percent']:.4f} ± {r[metric+'_population_sd_percent']:.4f}"


def _write_report(summary,paired,mechanisms,interventions,audits,manifest,preflight):
    import numpy as np
    base="E69.3-G2 historical"; metrics=("acc","hard_auc","prob_auc","sen","spe","asd_f1")
    lines=["# Experiment 75: DCER-v2","","## Execution",
      f"- Tasks {manifest.get('completed_tasks',0)}/40; independent audits {len(audits)}/40.",
      f"- Runtime {HERE}; preferred OSSFS path failed with Input/output error, so fallback was used.",
      f"- Device {preflight.get('device')}; worker cap {manifest.get('max_workers')}; recorded task failures {len(manifest.get('failed_tasks',[]))}.",
      "- Every fold starts from its E61 epoch-10 audited warmup; E69.3-G2 is read-only.",
      "","## Source and method audit",
      "- E69.3-G2 uses the audited E53 PLNNoPEN. DARG 320-D features enter the CAM; its final 144-D latent representation is the input to OutputLayer. DCER hooks that boundary, retaining one CAM forward and one classifier path; it adds no logit head.",
      "- The root SPromptGL PLN has PEN-like text selection/RES operations. E69 uses PLNNoPEN. view(N,4,4,9) is a latent partition, not anatomical ROI localization.",
      "- SPromptGL: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf . BrainPrompt disease-level supervision uses normalized patient graph/text prototype similarity cross-entropy; its paper uses Llama/LLM2Vec representations, while E75 uses cached CLIP RN50 SPromptGL prompts: https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf .",
      "- SABER Module III is read as patient-conditioned semantic queries, cross-attention, gated residual and disease supervision; no verified official source code was found: https://arxiv.org/abs/2607.01901 .",
      "- Differential Transformer subtracts two Softmax attention maps to suppress common attention noise. DCER borrows only this idea and does not reproduce the architecture: https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html .",
      "- Frozen RN50 cache, all eight original prompt strings/order, and the anonymous prototype geometry were SHA/geometry checked.",
      "","## Classification results",
      "Mean ± population SD, percent. Hard-AUC is historical hard-prediction AUC (binary BA); Probability-AUC uses continuous ASD probability.",
      "","| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for g in (base,)+GROUPS: lines.append("| "+g+" | "+" | ".join(_mean_text(summary,g,m) for m in metrics)+" |")
    lines+=["","### Excluding Fold 2 (descriptive only)","","| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |","|---|---:|---:|---:|---:|---:|"]
    for g in (base,)+GROUPS: lines.append("| "+g+" | "+" | ".join(_mean_text(summary,g,m,"exclude_fold2_descriptive_9") for m in metrics[:5])+" |")
    lines+=["","## Paired comparisons","","Delta = group A minus group B, percentage points; W/T/L are fold counts. Historical comparisons are descriptive.",
      "","| Comparison | Scope | Metric | Mean delta pp | W/T/L |","|---|---|---|---:|---:|"]
    pairs=[(g,base) for g in GROUPS]+[("G2","G1"),("G3","G2"),("G3","G4")]
    for a,b in pairs:
        for scope in ("all_10","exclude_fold2_descriptive_9"):
            for metric in ("acc","hard_auc","prob_auc","sen"):
                r=next(x for x in paired if x["group_a"]==a and x["group_b"]==b and x["scope"]==scope and x["metric"]==metric)
                lines.append(f"| {a} - {b} | {scope} | {metric} | {r['mean_delta_pp']:+.4f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=["","## Mechanism results"]
    for g in GROUPS:
        items=[r for r in mechanisms if r["group"]==g]
        gamma=np.asarray([r["gamma"] for r in items]); residual=np.asarray([r["fused_residual_norm_ratio"]*100 for r in items])
        ad=np.asarray([r["attention_pair_abs_difference"] for r in items]); common=np.asarray([r["common_token_fraction"] for r in items])
        lines.append(f"- {g}: gamma {gamma.mean():+.5f} ± {gamma.std(ddof=0):.5f}; residual/CAM norm {residual.mean():.5f}% ± {residual.std(ddof=0):.5f}; token-attention absolute difference {ad.mean():.5f}; common-high fraction {common.mean():.5f}.")
    for g in ("G3","G4"):
        items=[r for r in mechanisms if r["group"]==g and "validity_teacher_correlation" in r]
        if items:
            corr=np.asarray([r["validity_teacher_correlation"] for r in items]); delta=np.asarray([r.get("validity_high_minus_low_teacher",0) for r in items])
            lines.append(f"- {g} validity: training-only teacher correlation {corr.mean():.5f} ± {corr.std(ddof=0):.5f}; high-vs-low validity quartile teacher contribution {delta.mean():+.5f}.")
    for name in ("direct_off","semantic_neutral","common_off","validity_off"):
        items=[r for r in interventions if r["intervention"]==name]
        if items: lines.append(f"- {name}: {sum(r['hard_flip_count'] for r in items)} hard-label flips; wrong-to-correct {sum(r['wrong_to_correct'] for r in items)}, correct-to-wrong {sum(r['correct_to_wrong'] for r in items)} across {len(items)} checkpoints.")
    g3base=next(r for r in paired if r["group_a"]=="G3" and r["group_b"]==base and r["metric"]=="acc" and r["scope"]=="all_10")
    g3hard=next(r for r in paired if r["group_a"]=="G3" and r["group_b"]==base and r["metric"]=="hard_auc" and r["scope"]=="all_10")
    g3prob=next(r for r in paired if r["group_a"]=="G3" and r["group_b"]==base and r["metric"]=="prob_auc" and r["scope"]=="all_10")
    g3g2=next(r for r in paired if r["group_a"]=="G3" and r["group_b"]=="G2" and r["metric"]=="acc" and r["scope"]=="all_10")
    g3g4=next(r for r in paired if r["group_a"]=="G3" and r["group_b"]=="G4" and r["metric"]=="acc" and r["scope"]=="all_10")
    flips=sum(r["hard_flip_count"] for r in interventions if r["intervention"]=="direct_off")
    if g3base["mean_delta_pp"]>0 and g3base["wins"]>=6 and g3hard["mean_delta_pp"]>=-0.5 and g3prob["mean_delta_pp"]>=-0.5 and g3g2["mean_delta_pp"]>0 and g3g4["mean_delta_pp"]>0 and flips>0: decision="GO"
    elif any(r["group_b"]==base and r["metric"]=="acc" and r["scope"]=="all_10" and r["mean_delta_pp"]>0 for r in paired) or flips: decision="PARTIAL"
    else: decision="NO-GO"
    lines+=["","## Decision and limits",f"Decision: {decision}. This is the prespecified evidence category, not a significance claim.",
      "","Historical test-fold ACC checkpoint selection (val=test), pre-CV RFE/standardization, and the 871-node transductive graph were retained. Fold 2 trains on test rows 377 (ASD) and 407 (TD). Ten-fold results are not unbiased independent generalization estimates; excluding Fold 2 is descriptive only.",
      "","The 9-D tokens are latent feature partitions, not anatomical regions. Attention and validity are model-internal predictive quantities, not clinical or causal biomarkers. The masking teacher measures sensitivity of the current model. G3 versus G4 is the test of whether the supplied medical text contributes beyond matched prototype geometry.",
      "","## Audit",f"- Independent checkpoint audits: {len(audits)}/40 PASS.",
      f"- Exact RNG checks against E69 shared epochs: {sum(r['rng_epochs_exact_match'] for r in audits)} epoch-fold matches.",
      "- Fold 2 label exposure: rows 377 and 407 only; all other folds have no train/test overlap.",
      f"- Report: {RESULTS/'report.md'}; manifest: {HERE/'manifest.json'}.",""]
    return "\n".join(lines),decision


def _append_handoff(summary,paired,interventions,manifest,decision):
    import hashlib
    path=Path("/model/spromptgl/experiment/handoff.md")
    section=["","## Experiment 75: DCER-v2","",
      f"- Directory: {HERE}. Preferred OSSFS directory returned Input/output error; fallback used.",
      "- Base: E69.3-G2 reference commit 41dbcb45a0273673857051d8e821b8c3315eef00; ten E61 epoch-10 warmups restored independently.",
      "- Groups: G1 token competition; G2 common-high-response suppression; G3 training-only leave-one-token-out validity loss; G4 geometry-matched anonymous prototypes replacing the SPromptGL-8 text bank.",
      f"- Formal tasks: {manifest.get('completed_tasks',0)}/40; independent audits: {manifest.get('independent_audits',0)}/40; decision: {decision}.",
      "","| Group | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    handoff_metrics=("acc","hard_auc","prob_auc","sen","spe","asd_f1")
    for g in ("E69.3-G2 historical",)+GROUPS:
        r=next(x for x in summary if x["group"]==g and x["scope"]=="all_10")
        section.append("| "+g+" | "+" | ".join(f"{r[m+'_mean_percent']:.4f} +/- {r[m+'_population_sd_percent']:.4f}" for m in handoff_metrics)+" |")
    section += ["","Paired ACC differences, group A minus B; full folds / descriptive folds excluding Fold 2:"]
    for a,b in [(g,"E69.3-G2 historical") for g in GROUPS]+[("G2","G1"),("G3","G2"),("G3","G4")]:
        items=[r for r in paired if r["group_a"]==a and r["group_b"]==b and r["metric"]=="acc"]
        x=next(r for r in items if r["scope"]=="all_10"); y=next(r for r in items if r["scope"]=="exclude_fold2_descriptive_9")
        section.append(f"- {a} - {b}: {x['mean_delta_pp']:+.4f} pp ({x['wins']}/{x['ties']}/{x['losses']}) / {y['mean_delta_pp']:+.4f} pp ({y['wins']}/{y['ties']}/{y['losses']}).")
    direct=[r for r in interventions if r["intervention"]=="direct_off"]
    neutral=[r for r in interventions if r["intervention"]=="semantic_neutral"]
    section += ["",f"- Direct-off: {sum(r['hard_flip_count'] for r in direct)} flips; wrong-to-correct {sum(r['wrong_to_correct'] for r in direct)}, correct-to-wrong {sum(r['correct_to_wrong'] for r in direct)}.",
      f"- Semantic-neutral: {sum(r['hard_flip_count'] for r in neutral)} flips. Full CORR-independent mechanics, common response, validity alignment, and residual measurements are in {RESULTS/'mechanism.csv'} and {RESULTS/'interventions.csv'}.",
      "- Retained historical limits: val=test selection; preprocessing before CV; transductive graph; Fold-2 exposure of rows 377/407. Nine-fold analysis is descriptive.",
      "- E75 replaces PEN at the audited PLNNoPEN 144-D CAM-to-OutputLayer boundary. BrainPrompt/SABER/Differential Transformer informed design only; no full architecture replication. SABER public code was not confirmed.",
      "- Tokens are not ROIs; masking sensitivity is not clinical causal evidence. G3-versus-G4 measures the medical text contribution.",
      f"- Report: {RESULTS/'report.md'}; manifest: {HERE/'manifest.json'}; conclusion: {decision}.",""]
    append="\n".join(section)
    try:
        original=path.read_bytes(); data=original+append.encode("utf-8")
        temp=path.with_name(path.name+f".tmp.{os.getpid()}")
        with temp.open("wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
        if path.read_bytes()!=data: raise IOError("handoff readback mismatch")
        return {"status":"APPENDED_AND_VERIFIED","path":str(path),"sha256":hashlib.sha256(data).hexdigest()}
    except Exception as exc:
        pending=RESULTS/"handoff_append_pending.md"; pending.write_text(append,encoding="utf-8")
        return {"status":"FAILED","path":str(path),"error":repr(exc),"pending_text":str(pending)}


def run_audit(max_workers=5):
    jobs=[(g,f) for g in GROUPS for f in range(1,11)]
    results=[]; failures=[]
    with ProcessPoolExecutor(max_workers=max(1,min(int(max_workers),10)),mp_context=mp.get_context("spawn")) as pool:
        futures={pool.submit(audit_fold,g,f):(g,f) for g,f in jobs}
        for future in as_completed(futures):
            g,f=futures[future]
            try: results.append(future.result()); print(f"AUDITED {g} fold={f}",flush=True)
            except Exception as exc: failures.append({"group":g,"fold":f,"error":repr(exc),"traceback":traceback.format_exc()}); print(f"AUDIT_FAILED {g} fold={f}: {exc!r}",flush=True)
    audits=[x["audit"] for x in results]
    interventions=[r for x in results for r in x["interventions"]]
    if failures:
        R.write_csv(RESULTS/"checkpoint_audit.csv",audits+[dict(x,checkpoint_audit="FAIL") for x in failures])
        R.write_csv(RESULTS/"interventions.csv",interventions)
        R.atomic_json(RESULTS/"audit_status.json",{"status":"FAIL","passed":len(audits),"failed":failures})
        m=json.loads((HERE/"manifest.json").read_text()); m.update({"status":"PARTIAL","independent_audits":len(audits),"audit_failures":failures}); R.atomic_json(HERE/"manifest.json",m)
        return m
    fold_rows=[x["fold_metrics"] for x in results]; mechanisms=[x["mechanism"] for x in results]
    historical=_historical_e69(); summary=_summary(fold_rows,historical); paired=_paired(fold_rows,historical)
    R.write_csv(RESULTS/"fold_metrics.csv",fold_rows); R.write_csv(RESULTS/"summary.csv",summary)
    R.write_csv(RESULTS/"paired_comparisons.csv",paired); R.write_csv(RESULTS/"mechanism.csv",mechanisms)
    R.write_csv(RESULTS/"interventions.csv",interventions); R.write_csv(RESULTS/"checkpoint_audit.csv",audits)

    group_rng={}
    for fold in range(1,11):
        same=[x["task"] for x in results if int(x["task"]["fold"])==fold]
        for key in ("initial_shared_sha","initial_gfr_sha","initial_cam_sha","relation_core_sha","pair_sha","split_sha"):
            if len({t.get(key) for t in same})!=1: raise RuntimeError(f"group parity mismatch fold={fold} field={key}")
        if len({t["dcer_initialization"]["parameter_sha256"] for t in same})!=1: raise RuntimeError(f"DCER init mismatch fold={fold}")
        traces={t["group"]:{int(x["epoch"]):x for x in t["rng_trace"]} for t in same}
        common=set.intersection(*(set(v) for v in traces.values()))
        exact=sum(all(traces[g][e]==traces[GROUPS[0]][e] for g in GROUPS[1:]) for e in common)
        if not common or exact!=len(common): raise RuntimeError(f"groups RNG mismatch fold={fold}: {exact}/{len(common)}")
        group_rng[str(fold)]={"common_epochs":len(common),"exact_matches":exact}

    manifest=json.loads((HERE/"manifest.json").read_text())
    preflight=json.loads((RESULTS/"preflight.json").read_text())
    report,decision=_write_report(summary,paired,mechanisms,interventions,audits,manifest,preflight)
    (RESULTS/"report.md").write_text(report,encoding="utf-8")
    handoff=_append_handoff(summary,paired,interventions,manifest,decision)
    status={"status":"PASS","passed":len(audits),"failed":0,"group_rng":group_rng,
        "protected_source_sha256_unchanged":R.source_inventory()==preflight["source_sha256"],
        "handoff":handoff,"completed_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    R.atomic_json(RESULTS/"audit_status.json",status)
    manifest.update({"independent_audits":40,"checkpoint_audit_status":"PASS","shared_initialization_status":"PASS",
      "shared_epoch_rng_status":"PASS","scientific_decision":decision,"handoff":handoff,
      "audit_status_path":str(RESULTS/"audit_status.json"),
      "result_sha256":{p.name:R.sha_file(p) for p in RESULTS.iterdir() if p.is_file()}})
    manifest["status"]="COMPLETE_AUDITED" if handoff["status"]=="APPENDED_AND_VERIFIED" else "COMPLETE_WITH_LIMITATIONS"
    R.atomic_json(HERE/"manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=5); args=parser.parse_args()
    result=run_audit(args.workers)
    print(json.dumps({"status":result.get("status"),"audits":result.get("independent_audits"),
      "decision":result.get("scientific_decision"),"handoff":result.get("handoff")},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
