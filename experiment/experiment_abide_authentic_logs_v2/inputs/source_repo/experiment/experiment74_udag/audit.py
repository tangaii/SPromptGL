"""Independent rebuild/readback, fixed-checkpoint interventions and reporting."""
from __future__ import annotations
import json,time,copy,os,math
from pathlib import Path
import run as R

def scalar_metrics(m):return {k:float(m[k]) for k in R.METRICS}

def audit_fold(fold):
    import torch,numpy as np
    task=json.loads(R.task_path(fold).read_text());cp=R.tload(R.best_path(fold))
    if task["status"]!="COMPLETE" or R.sha(R.best_path(fold))!=task["checkpoint_sha256"]:raise RuntimeError("checkpoint file SHA mismatch")
    if cp["source_sha256"]!=task.get("source_sha256") or cp["config"]!=R.CFG or cp["fold"]!=fold or cp["group"]!=R.GROUP:raise RuntimeError("checkpoint experiment identity mismatch")
    current_sources=R.sources()
    # The post-training fix only makes audit module lookup path-explicit. Keep
    # that audit-harness SHA distinct while verifying every trained dependency.
    for name,digest in cp["source_sha256"].items():
        if current_sources.get(name)==digest:continue
        if name in ("e74/run.py","e74/audit.py"):
            original=R.RT/"sources"/("training_run.py" if name=="e74/run.py" else "training_audit.py")
            if R.sha(original)!=digest:raise RuntimeError("frozen original source digest failed")
            continue
        if current_sources.get(name)!=digest:raise RuntimeError(f"trained source changed after completion: {name}")
    if cp["source_sha256"].get("e74/run.py")!=task["source_sha256"].get("e74/run.py"):raise RuntimeError("training runner and task source SHA differ")
    import ast
    def functions(path):
        return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}
    if (R.RT/"sources/training_run.py").is_file():
        previous,current=functions(R.RT/"sources/training_run.py"),functions(R.EXP/"run.py")
        for name,definition in previous.items():
            if name not in ("main","run_all") and current.get(name)!=definition:raise RuntimeError(f"trained runner function changed: {name}")
    if (R.RT/"sources/training_audit.py").is_file():
        previous,current=functions(R.RT/"sources/training_audit.py"),functions(R.EXP/"audit.py")
        for name,definition in previous.items():
            if name not in ("audit_fold","finalize") and current.get(name)!=definition:raise RuntimeError(f"metric/analysis function changed: {name}")
    state,w,identity=R.prepare(fold);e=R.load_base().load_e61();r=e.load_base()
    if identity!=cp["identity"] or identity!=task["identity"]:raise RuntimeError("initial state identity mismatch")
    R.restore(state,cp)
    if state["model"].adaptive_gate_logits.requires_grad:raise RuntimeError("adaptive gate trainable")
    if not torch.equal(state["model"].adaptive_gate_logits.detach().cpu(),torch.logit(torch.tensor(R.CFG["udag"]["adaptive_gate_compatibility_values"]))):raise RuntimeError("adaptive compatibility field changed")
    if state["opt_added"].state.get(state["model"].adaptive_gate_logits,{}):raise RuntimeError("frozen adaptive acquired optimizer history")
    start_rng=r.rng_sha(r.pack_rng());counts=[0]*4
    handles=[mod.register_forward_hook(lambda _m,_i,_o,idx=i:counts.__setitem__(idx,counts[idx]+1)) for i,mod in enumerate(state["model"].gcn)]
    with torch.no_grad():out=R.forward(state);m=R.metrics(state,out)
    for h in handles:h.remove()
    if counts!=[1]*4:raise RuntimeError("formal GCN calls != one per modality")
    for k in R.METRICS:
        if abs(m[k]-cp["metrics"][k])>1e-10:raise RuntimeError(f"independent metric mismatch fold {fold} {k}")
    if not np.array_equal(m["pred"],cp["test_prediction"]) or not np.allclose(m["prob"],cp["test_probability"],atol=1e-8,rtol=0):raise RuntimeError("independent prediction readback mismatch")
    mechanism=copy.deepcopy(out["info"]["edge_diagnostics"])
    if any(d["minimum_edge_weight"] < 0 or d["symmetry_max_abs"] > 1e-6 or not d["normalized_finite"] for d in mechanism):
        raise RuntimeError("actual unified graph fails symmetry/nonnegative/normalization audit")
    interventions={};changes={}
    fullf=out["fused"].detach().clone();full_msg=[v.detach().clone() for v in out["info"]["messages"]]
    original_adjs=[v.detach().clone() for v in out["adjs"]]
    for mode in ("disease_off","positive_off","negative_off"):
        with torch.no_grad():alt=R.forward(state,intervention=mode);am=R.metrics(state,alt)
        for a,b in zip(original_adjs,alt["adjs"]):
            if not torch.equal(a,b):raise RuntimeError("intervention changed original GraphLearn loss object")
        if mode=="disease_off":
            for a,b in zip(alt["info"]["unified_adjs"],original_adjs):
                if not torch.equal(a,b):raise RuntimeError("disease_off not exact original adjacency")
            original_msgs=[v.detach().clone() for v in alt["info"]["messages"]]
            off_fused=alt["fused"].detach().clone()
        interventions[mode]=scalar_metrics(am)
        changes[mode]={"flipped_predictions":int(np.count_nonzero(m["pred"]!=am["pred"])),
                       "full_correct_intervention_wrong":int(np.sum((m["pred"]==state["labels_np"][state["test_np"]])&(am["pred"]!=state["labels_np"][state["test_np"]]))),
                       "full_wrong_intervention_correct":int(np.sum((m["pred"]!=state["labels_np"][state["test_np"]])&(am["pred"]==state["labels_np"][state["test_np"]]))),
                       "mean_absolute_probability_change":float(np.abs(m["prob"]-am["prob"]).mean()),
                       "max_absolute_probability_change":float(np.abs(m["prob"]-am["prob"]).max()),
                       "full_minus_intervention_acc_pp":100*float(m["acc"]-am["acc"])}
        del alt
    raw=torch.cat(state["parts"],1);rawnorm=float(raw.norm());fusion=torch.sigmoid(state["model"].fusion_weight.detach()).flatten().cpu().tolist()
    for i,d in enumerate(mechanism):
        d.update({"fold":fold,"original_gcn_message_norm":float(original_msgs[i].norm()),"udag_gcn_message_norm":float(full_msg[i].norm()),
                  "gcn_message_difference_norm":float((full_msg[i]-original_msgs[i]).norm()),
                  "relative_gcn_message_difference":float((full_msg[i]-original_msgs[i]).norm()/original_msgs[i].norm().clamp_min(1e-30)),
                  "original_gcn_to_raw_ratio":float(original_msgs[i].norm())/rawnorm,
                  "udag_gcn_to_raw_ratio":float(full_msg[i].norm())/rawnorm,
                  "weighted_message_difference_to_raw_ratio":float((full_msg[i]-original_msgs[i]).norm())*fusion[i]/rawnorm,
                  "fusion_weight_sigmoid":fusion[i],"fused_representation_difference_norm":float((fullf-off_fused).norm()),
                  "fused_representation_difference_to_raw_ratio":float((fullf-off_fused).norm())/rawnorm,
                  "raw_input_norm":rawnorm,"best_epoch":cp["best_epoch_1based"],"epochs_executed":cp["epochs_executed"],
                  "train_pair_loss_at_best":cp["diagnostics"]["losses"]["pair"]})
    R.zero(state);gradout=R.forward(state);gradout["cls"].backward();grads=R.gradient_info(state)
    if not grads["adaptive_grad_is_none"]:raise RuntimeError("frozen adaptive got classification gradient")
    for i,d in enumerate(mechanism):
        d.update({"classification_edge_gate_abs":grads["edge_gate_abs"][i],"classification_scorer_grad_norm":grads["scorer_norm"][i],
                  "classification_encoder_grad_norm":grads["encoder_norm"][i],"classification_graphlearn_grad_norm":grads["graphlearn_norm"][i],
                  "classification_gcn_grad_norm":grads["gcn_norm"][i]})
    del gradout;R.zero(state)
    # Descriptive relation evaluation only, after all model selection is over.
    from cadr_gfr import pair_features
    unexposed=np.setdiff1d(state["test_np"],state["train_np"])
    pairsets={"train":(state["train_pairs_np"],state["train_same_np"])}
    pairs,same=r.sample_balanced_pairs(unexposed,state["labels_np"],4096,np.random.default_rng(740000+fold),None)
    pairsets["test_unexposed_labels"]=pairs,same
    with torch.no_grad():
        zs=[enc(x) for enc,x in zip(state["model"].encoders,state["parts"])]
        for scope,(pairs,same) in pairsets.items():
            pp=torch.as_tensor(pairs,device=raw.device,dtype=torch.long)
            for i,(z,scorer) in enumerate(zip(zs,state["model"].scorers)):
                logits=scorer(pair_features(z[pp[:,0]],z[pp[:,1]])).cpu().numpy()
                auc,pr=r.auc_metrics(same,logits)
                mechanism[i][scope+"_relation_auc"]=auc;mechanism[i][scope+"_relation_pr_auc"]=pr
                mechanism[i][scope+"_relation_pair_n"]=len(pairs)
    if r.rng_sha(r.pack_rng())!=start_rng:raise RuntimeError("audit consumed training RNG")
    ref=json.loads((R.BASE/".runtime/tasks"/f"E69.3-G2_f{fold:02d}.json").read_text())
    rm={t["epoch"]:t for t in ref["rng_trace"]};common=[t for t in task["rng_trace"] if t["epoch"] in rm]
    matching=sum(all(t[k]==rm[t["epoch"]][k] for k in ("rng_before","rng_after_train","rng_after_eval")) for t in common)
    if matching!=len(common):raise RuntimeError(f"shared-epoch RNG consumption diverged {fold}: {matching}/{len(common)}")
    result={"fold":fold,"checkpoint_audit":"PASS","checkpoint_sha256":task["checkpoint_sha256"],"identity":identity,
            "metrics":scalar_metrics(m),"best_epoch_1based":cp["best_epoch_1based"],"epochs_executed":cp["epochs_executed"],
            "probability_max_abs_error":float(np.max(np.abs(m["prob"]-cp["test_probability"]))),"predictions_exact":True,
            "formal_gcn_call_counts":counts,"adaptive_frozen_and_unused":True,"gradient":grads,"mechanism":mechanism,
            "interventions":interventions,"intervention_changes":changes,"rng_common_epochs":len(common),"rng_exact_matches":matching,
            "audit_rng_unchanged":True,"pair_train_scope_pass":True,"unexposed_test_patient_count":len(unexposed)}
    R.atomic_json(R.TASKS/f"audit_f{fold:02d}.json",result)
    del state,out,fullf,full_msg,original_msgs,original_adjs,cp;torch.cuda.empty_cache()
    print(f"INDEPENDENT AUDIT PASS {fold}/10",flush=True);return result

def summaries(rows):
    import numpy as np
    out=[]
    for group in ("E69.3-G2",R.GROUP):
        selected=[r for r in rows if r["group"]==group]
        x={"group":group,"folds":len(selected)}
        for k in R.METRICS:
            vals=np.array([r[k] for r in selected])*100;x[k+"_mean_percent"]=float(vals.mean());x[k+"_population_sd_percent"]=float(vals.std(ddof=0))
        out.append(x)
    return out

def pairs(rows):
    import numpy as np
    from scipy.stats import t
    a={x["fold"]:x for x in rows if x["group"]==R.GROUP};b={x["fold"]:x for x in rows if x["group"]=="E69.3-G2"};out=[]
    for scope,ids in (("all10",list(range(1,11))),("nine_no_index_overlap",[i for i in range(1,11) if i!=2])):
        for k in R.METRICS:
            d=np.array([100*(a[i][k]-b[i][k]) for i in ids]);mu=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(len(d)));ci=float(t.ppf(.975,len(d)-1))*se
            out.append({"comparison":"E74-UDAG minus E69.3-G2","scope":scope,"metric":k,"folds":len(d),"mean_difference_pp":mu,
                        "population_sd_difference_pp":float(d.std()),"descriptive_95ci_low_pp":mu-ci,"descriptive_95ci_high_pp":mu+ci,
                        "wins":int((d>1e-9).sum()),"ties":int((np.abs(d)<=1e-9).sum()),"losses":int((d< -1e-9).sum()),"per_fold_delta_pp":dict(zip(map(str,ids),d.tolist()))})
    return out

def finalize(runtime):
    import numpy as np,torch
    if (R.RT/"resource_summary.json").exists():
        runtime={**json.loads((R.RT/"resource_summary.json").read_text()),**runtime}
    audits=[audit_fold(f) for f in range(1,11)]
    original=list(__import__('csv').DictReader((R.BASE/"results/fold_metrics.csv").open()))
    rows=[{"group":"E69.3-G2","fold":int(x["fold"]),"status":"READ_ONLY_REFERENCE",**{k:float(x[k]) for k in R.METRICS}} for x in original if x["group"]=="E69.3-G2"]
    if len(rows)!=10:raise RuntimeError("baseline fold results incomplete")
    for a in audits:rows.append({"group":R.GROUP,"fold":a["fold"],"status":"INDEPENDENT_AUDIT_PASS",**a["metrics"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],"checkpoint_sha256":a["checkpoint_sha256"]})
    summary=summaries(rows);paired=pairs(rows);mechanism=[d for a in audits for d in a["mechanism"]]
    R.write_csv(R.RESULTS/"fold_metrics.csv",rows);R.write_csv(R.RESULTS/"summary.csv",summary);R.write_csv(R.RESULTS/"paired_comparisons.csv",paired);R.write_csv(R.RESULTS/"mechanism.csv",mechanism)
    protected=json.loads((R.RT/"protected_sha.json").read_text());bad=[p for p,h in protected.items() if R.sha(p)!=h]
    if bad:raise RuntimeError(f"protected history mutated: {bad}")
    delta={x["metric"]:x for x in paired if x["scope"]=="all10"};clean={x["metric"]:x for x in paired if x["scope"]=="nine_no_index_overlap"}
    flips=sum(a["intervention_changes"]["disease_off"]["flipped_predictions"] for a in audits)
    effect=float(np.mean([a["intervention_changes"]["disease_off"]["full_minus_intervention_acc_pp"] for a in audits]))
    msgmax=max(d["weighted_message_difference_to_raw_ratio"] for d in mechanism)
    if delta["acc"]["mean_difference_pp"]>0 and clean["acc"]["mean_difference_pp"]>0 and all(delta[k]["mean_difference_pp"]>=0 for k in ("hard_auc","prob_auc","sen","spe")) and flips>0 and effect>0:
        judgment="GO"
    elif delta["acc"]["mean_difference_pp"] < -.5 or (flips==0 and msgmax<1e-7):judgment="NO-GO"
    else:judgment="PARTIAL"
    tasks=[json.loads(R.task_path(f).read_text()) for f in range(1,11)]
    manifest={"experiment":R.CFG["experiment"],"status":"COMPLETE_AUDITED","judgment":judgment,"experiment_root":str(R.EXP),"base":R.CFG["base"],
              "expected_tasks":10,"completed_tasks":10,"independent_audits":10,"failed_tasks":[],"source_sha256":R.sources(),
              "training_source_sha256":tasks[0]["source_sha256"],"audit_source_sha256":R.sources(),
              "posttraining_change":"audit loader uses absolute path to avoid E53.3 module shadowing; frozen originals SHA-verified and training functions AST-identical", "runtime":runtime,
              "tasks":[{"fold":a["fold"],"checkpoint_sha256":a["checkpoint_sha256"],"warmup_sha256":a["identity"]["warmup_sha256"],"initial_shared_sha":a["identity"]["initial_shared_sha"],"warmup_rng_sha":a["identity"]["warmup_rng_sha"],"pair_sha":a["identity"]["pair_sha"],"status":"COMPLETE","audit":"PASS"} for a in audits],
              "protected_files_count":len(protected),"protected_sha_pass":True,"rng_common_epochs":sum(a["rng_common_epochs"] for a in audits),"rng_exact_matches":sum(a["rng_exact_matches"] for a in audits),
              "formal_gcn_calls_per_forward":4,"historical_darg_gcn_calls_per_forward":16,"gcn_call_reduction_percent":75.,
              "effective_trainable_parameter_reduction":4,"adaptive_gate_compatibility_only":True,
              "disease_off_prediction_flips_total":flips,"full_minus_disease_off_acc_mean_pp":effect,
              "training_wall_seconds_per_task":{str(t["fold"]):t["wall_seconds"] for t in tasks},"peak_allocated_bytes_per_task":{str(t["fold"]):t["peak_gpu_bytes"] for t in tasks}}
    R.atomic_json(R.RESULTS/"audit_status.json",{"status":"PASS","checkpoint_audits":audits,"protected_sha_pass":True,"protected_files":len(protected),"source_sha256":R.sources()})
    lines=["# Experiment 74 — UDAG", "",f"Status: **COMPLETE_AUDITED**; 10/10 trained and independently rebuilt/read back. Scientific judgment: **{judgment}**.","",
           "## Source and protocol audit","",f"Read-only E69.3-G2 commit `{R.CFG['base']['commit']}` on `{R.CFG['base']['branch']}`; local runner SHA `{R.CFG['base']['run_sha256']}` exactly matches the raw GitHub file at that commit. Its source loader validates E61, E62 and all five E53.3 dependency sources against historical manifests. The OSS experiment-directory creation and Git object fetch both encountered errno-5 I/O errors; all new work uses the configured local root. No historical experiment, root model or data files were modified.","",
           "SPromptGL Section 2.1 defines modality-specific patient edges, propagation on the concatenated input, then fusion. The actual historical implementation is retained: GraphLearn applies a learned linear projection, normalized cosine, threshold >0.9; GCN is two GraphConv layers with one intermediate ReLU, no dropout. Original GraphLearn includes its native diagonal; the historical `normalize_adj(A+I)` adds identity once. UDAG preserves that exact convention. Paper Eq.1 contains label-graph supervision not restored by this experiment: the current historical graph loss stays unchanged. CAM/OutputLayer are untouched; 320-D GFR output becomes four 36-D CAM positions (144-D classifier input); OutputLayer reduces to 36-D then two-class log-probabilities and does not use its `attn_embedding` argument.","",
           "DARG uses the same modality GCN four times: original graph, positive graph, negative graph, identity subtraction. Its signed message combines positive and negative messages plus direct difference; adaptive mixing retains original propagation. UDAG removes those separate message branches, direct differences, and adaptive message fusion from the formal forward. Compatibility `adaptive_gate_logits` remains in state_dict and optimizer groups but is frozen, unused, and never acquires optimizer state; four existing edge gates remain trainable. The original fusion_weight is retained.","",
           "Candidate support is the unchanged `_candidate_support`: original positive off-diagonal edges; fallback similarity candidates only for historically enabled modalities; mutual Top-12; upper-triangle scoring of symmetric 65-D pair features. Mutual Top-12 selects reweighting locations, not deletion of all other original edges. Selected original positive edges use A[1+g(2q−1)], negative edges use A[1−g(1−2q)], uncertain edges and unselected original edges stay unchanged. A selected fallback edge with A=0 is added only for q≥0.55 with similarity×2(q−0.5)×g. Scatter-add uses symmetric deltas and intact autograd, no `.data` update. q=sigmoid(logit), no Platt or OOF. Original adjacency A, not modified adjacency, is supplied to GraphConstructLoss.","",
           "The four relation encoders, independent 65→32→1 scorers, CORR-detached conditional modality adversary, GRL=0.03, fixed 4096 balanced training pairs and original total loss remain. CORR is detached only in the adversarial branch, not relation supervision or classification propagation. This is conditional distribution alignment, not an explicit pairwise teacher-consistency objective.","",
           "All ten E61 epoch-10 warmups are read-only reused. All shared model parameters, three optimizer states and Python/NumPy/CPU/CUDA RNG states are inherited. E69.3-G2 gate initialization is copied deterministically before freezing the unused adaptive gate. No per-epoch reseeding and no extra stochastic operation are introduced. A controlled one-update save/reload test matches parameters, optimizer states, outputs and RNG exactly.","",
           "**Evaluation limits:** fold 2 deliberately adds 1-based rows 377 (ASD) and 407 (TD) to training while keeping them in test, and rebuilds pairs and class weights exactly as E69.3-G2. These are label-exposure results. val=test checkpoint selection, pre-CV RFE/standardization and the 871-node transductive graph remain. Results are paired exploratory mechanism diagnostics, not leakage-free or unbiased generalization. The separate nine-fold summary removes index-overlap fold 2 only; other historical evaluation limits remain. No post-hoc removal of the two exposed test rows is presented.","",
           "## Classification results","","All values: 10-fold mean ± population SD, percent.","","|Model|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|BA|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:lines.append("|"+s["group"]+"|"+"|".join(f"{s[k+'_mean_percent']:.4f} ± {s[k+'_population_sd_percent']:.4f}" for k in R.METRICS)+"|")
    lines += ["","## Paired differences","","Differences are E74 minus E69.3-G2, percentage points; 95% paired-t intervals are descriptive because CV training sets overlap and model selection uses test ACC.","","|Scope|Metric|Mean Δ pp|W/T/L|Descriptive 95% interval|","|---|---|---:|---|---|"]
    for p in paired:lines.append(f"|{p['scope']}|{p['metric']}|{p['mean_difference_pp']:+.4f}|{p['wins']}/{p['ties']}/{p['losses']}|[{p['descriptive_95ci_low_pp']:+.4f}, {p['descriptive_95ci_high_pp']:+.4f}]|")
    lines += ["","## Per-fold results","","|Fold|ACC|Hard-AUC|Prob-AUC|SEN|SPE|Best epoch|Executed epochs|ΔACC pp|RNG matches/common|","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    base={x["fold"]:x for x in rows if x["group"]=="E69.3-G2"}
    for a in audits:lines.append(f"|{a['fold']}|"+"|".join(f"{100*a['metrics'][k]:.4f}" for k in R.METRICS[:5])+f"|{a['best_epoch_1based']}|{a['epochs_executed']}|{100*(a['metrics']['acc']-base[a['fold']]['acc']):+.4f}|{a['rng_exact_matches']}/{a['rng_common_epochs']}|")
    lines += ["","## Graph and message mechanism","","Quantities below average ten best checkpoints; exact per-fold/per-modality counts and gradients are in mechanism.csv. Relative message differences can be large when the denominator is nearly zero; raw-input-normalized absolute effects are also reported.","","|Modality|Original edges|Candidates|Positive|Negative|Uncertain|Fallback added|Edge gate|Adjacency relative Δ|Original GCN/input %|UDAG GCN/input %|Weighted message Δ/input %|Scorer cls grad|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in R.CFG["data"]["modalities"]:
        ds=[d for d in mechanism if d["modality"]==name];mean=lambda k:float(np.mean([d[k] for d in ds]))
        lines.append(f"|{name}|{mean('original_edges'):.1f}|{mean('candidate_edges'):.1f}|{mean('positive_edges'):.1f}|{mean('negative_edges'):.1f}|{mean('uncertain_edges'):.1f}|{mean('fallback_positive_added'):.1f}|{mean('edge_gate'):.5f}|{mean('adjacency_change_relative'):.5g}|{100*mean('original_gcn_to_raw_ratio'):.6g}|{100*mean('udag_gcn_to_raw_ratio'):.6g}|{100*mean('weighted_message_difference_to_raw_ratio'):.6g}|{mean('classification_scorer_grad_norm'):.3e}|")
    lines += ["","Training and unexposed-test relation AUC/PR-AUC are descriptive checkpoint diagnostics in mechanism.csv. The latter excludes any training-index overlap from relation evaluation, but does not remove historical test-based checkpoint selection. There is no independently retrained pre-alignment relation baseline, so these results do not isolate an adversarial-alignment improvement.","","## Fixed-checkpoint interventions","","Full minus intervention ACC: positive values favor UDAG. Hard-prediction flips, corrections and new errors are counted on each fold's complete historical test set.","","|Fold|Intervention|Full−off ACC pp|Hard flips|Full correct / off wrong|Full wrong / off correct|Mean abs Δ probability|","|---:|---|---:|---:|---:|---:|---:|"]
    for a in audits:
        for mode,c in a["intervention_changes"].items():lines.append(f"|{a['fold']}|{mode}|{c['full_minus_intervention_acc_pp']:+.4f}|{c['flipped_predictions']}|{c['full_correct_intervention_wrong']}|{c['full_wrong_intervention_correct']}|{c['mean_absolute_probability_change']:.8g}|")
    lines += ["","## Efficiency, audit and decision","",f"Formal GCN calls decrease from 16 to 4 per full graph forward (75% fewer GCN invocations), with no additional GCN, no positive/negative message branches or direct-difference computation. This does not mean 75% total runtime reduction: GraphLearn, relation scoring, CAM and auxiliary losses remain. Four adaptive parameters are frozen; compatible stored parameter count is unchanged. Runtime: `{json.dumps(runtime)}`. Ten independent workers had separate models, optimizers and RNG. Peak allocated bytes and task wall times are in manifest.json.","",
           f"All 10 independently rebuilt checkpoints reproduce hard predictions exactly and probabilities within 1e-8. All ten inheritance audits pass. {manifest['rng_exact_matches']}/{manifest['rng_common_epochs']} common epoch RNG triplets match E69.3-G2. Each formal forward has one GCN per modality and one CAM call. {len(protected)} protected source/data/history/checkpoint files retain their SHA. Synthetic graph cases, natural and controlled classification gradients, frozen adaptive gate, and save/reload one-step parity pass.","",
           f"**{judgment}**. Paired ACC Δ={delta['acc']['mean_difference_pp']:+.4f} pp; nine-fold Δ={clean['acc']['mean_difference_pp']:+.4f} pp. Disease-reweighting off changes {flips} hard predictions across 10 folds; mean full−off ACC={effect:+.4f} pp. The maximum weighted message difference/input ratio is {msgmax:.5g}. Interpretation must combine this absolute contribution with edge-weight changes; nonzero edges or gradients alone do not establish diagnostic benefit. GFR WD=0.11 remains fixed and can suppress GCN message amplitudes.","",
           "UDAG should replace DARG for diagnostic efficacy only if its paired performance and fixed-checkpoint interventions support it. A simpler graph pathway may still be an engineering simplification, but efficiency alone does not justify a new disease-aware method or disease-semantic module. Further semantic supervision is not established by this experiment; a future study would need a clean evaluation protocol and demonstrable disease-graph contribution.","",
           "## Sources and files","","- Official paper: https://papers.miccai.org/miccai-2025/0861-Paper5129.html","- Author code: https://github.com/wanxixi11/SPromptGL","- Locked base: https://github.com/tangaii/SPromptGL/tree/41dbcb45a0273673857051d8e821b8c3315eef00/experiment/experiment69_3_g2_two_rows",f"- Experiment root: `{R.EXP}`","- config.json; run.py; udag.py; audit.py; manifest.json","- results/fold_metrics.csv; summary.csv; paired_comparisons.csv; mechanism.csv; audit_status.json","- .runtime/checkpoints: best and resumable states; historical warmups are read-only referenced.",""]
    report="\n".join(lines);atomic_text(R.RESULTS/"report.md",report)
    extra=["","## Concrete interpretation and audit recovery","",
           "The unified graphs are valid and the intended structural simplification succeeded. Disease reweighting, positive enhancement and negative suppression each cause zero hard-prediction flips over all ten best checkpoints. Improved probability AUC accompanies lower ACC, Hard-AUC and ASD sensitivity. The nine folds without index overlap also lose ACC; the deficit is not confined to fold 2.","",
           "Retain DARG as the historical comparison base for now; do not adopt UDAG as a demonstrated efficacy improvement. This does not establish historical DARG disease-message efficacy either. A disease-semantic extension of UDAG is not currently supported by these results.","",
           "A plausible bottleneck is the small GCN signal under inherited WD=0.11: graph changes act only through those messages after removing the direct-difference pathway. Candidate selection alters a subset of denser original graphs. These are possible explanations, not a causal ablation of weight decay or direct differences; neither was changed here. CORR relation quality does not by itself establish propagated diagnostic benefit.","",
           "Hard thresholds and mutual Top-12 define non-differentiable support. Classification gradients flow through active continuous edge weights, the existing edge gates and relation networks, not through topology selection. The natural initial warmup has inactive disease edges; isolated controlled q=0.9 tests prove the gradient path without changing formal thresholds.","",
           "Preflight repaired a boolean-degree statistic and optimizer-snapshot aliasing before formal training. After 10/10 training tasks completed, same-name audit module shadowing was fixed using absolute-path loading. Original training sources were reconstructed byte-for-byte and SHA-verified. Trained runner functions remain AST-identical; the model forward, losses, optimizer steps and RNG logic were unchanged. Training and audit source hashes are recorded separately. No formal training task was repeated for these audit-harness repairs.","",
           "Single-seed paired intervals here are descriptive; overlapping CV training sets and historical test-based checkpoint selection do not support independent clinical significance claims.",""]
    atomic_text(R.RESULTS/"report.md",report+chr(10).join(extra))
    manifest["reference_source_sha256"]={p.name:R.sha(p) for p in (R.RT/"sources").glob("*.py")}
    manifest["results_sha256"]={p.name:R.sha(p) for p in R.RESULTS.iterdir() if p.is_file()}
    R.atomic_json(R.EXP/"manifest.json",manifest)
    # Preserve history, and make re-audit idempotent. No user-authored section is replaced.
    import fcntl
    hp=Path('/model/spromptgl/experiment/handoff.md');marker="## Experiment 74 — UDAG"
    with hp.with_suffix('.md.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=hp.read_text();excerpt=["",marker,"",f"- Status: COMPLETE_AUDITED; 10/10 formal tasks; 10/10 independent checkpoint audits. Judgment: {judgment}.",f"- Root: `{R.EXP}`. Base E69.3-G2 commit `{R.CFG['base']['commit']}`; immutable SHA-verified E61 epoch-10 warmups; optimizer/RNG restored. Fold 2 retains rows 377/407 train/test overlap; val=test and pre-CV preprocessing retained, so exploratory only.","- Model: selected original edges A*(1+g*(2q−1)) outside [.45,.55]; uncertain/unselected original edges unchanged. Fallback only adds positive confidence×similarity×edge-gate edges. One historical GCN per modality, original fusion weights; no direct-difference or adaptive message blend. Adaptive compatibility field frozen; edge gates trainable. Original relation supervision/adversary/preservation and GraphConstructLoss unchanged."]
        for s in summary:excerpt.append("- "+s["group"]+": "+", ".join(f"{k}={s[k+'_mean_percent']:.4f}±{s[k+'_population_sd_percent']:.4f}%" for k in R.METRICS))
        excerpt += ["- Paired E74−E69.3-G2: "+", ".join(f"{k} {delta[k]['mean_difference_pp']:+.4f} pp ({delta[k]['wins']}/{delta[k]['ties']}/{delta[k]['losses']})" for k in R.METRICS),f"- Nine non-index-overlap folds ΔACC={clean['acc']['mean_difference_pp']:+.4f} pp; W/T/L={clean['acc']['wins']}/{clean['acc']['ties']}/{clean['acc']['losses']}.",f"- Disease-off intervention: {flips} hard flips total; mean full−off ACC={effect:+.4f} pp. Largest weighted GCN-message delta/input={msgmax:.5g}. GCN calls 16→4 (75% call-count reduction; not a total-runtime claim).",f"- RNG {manifest['rng_exact_matches']}/{manifest['rng_common_epochs']} common epoch triplets match; protected SHA {len(protected)}/{len(protected)}. Source hashes and each warmup/checkpoint SHA in manifest; detailed graph/gate/gradient/intervention results in report and mechanism.csv."]
        for name in R.CFG["data"]["modalities"]:
            ds=[d for d in mechanism if d['modality']==name];excerpt.append(f"- {name}: edge gate mean={np.mean([d['edge_gate'] for d in ds]):.5f}; weighted message delta/input mean={np.mean([d['weighted_message_difference_to_raw_ratio'] for d in ds]):.5g}; q mean={np.mean([d['q_mean'] for d in ds]):.5f}.")
        excerpt += [f"- Scientific reading: {judgment}. Keep diagnostic efficacy separate from graph simplification; no demonstrated semantic-module benefit follows automatically. Do not tune WD or thresholds post hoc within E74. A future efficacy study needs clean evaluation and measurable graph intervention evidence.",f"- Report SHA: `{R.sha(R.RESULTS/'report.md')}`; config SHA: `{R.sha(R.EXP/'config.json')}`. No historical source/data/checkpoint edits.",""]
        addition='\n'.join(excerpt)
        if marker in old:
            start=old.index(marker);end=old.find('\n## ',start+len(marker));old=old[:start]+(old[end:] if end>=0 else '')
        atomic_text(hp,old.rstrip()+"\n"+addition)
    manifest['handoff_sha256']=R.sha(hp);R.atomic_json(R.EXP/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'judgment':judgment,'completed_tasks':10,'audits':10,'summary':summary,'delta':delta,'disease_off_flips':flips,'handoff_sha256':manifest['handoff_sha256']},indent=2),flush=True)
    return manifest

def atomic_text(p,text):
    temp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with temp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
    if p.read_text()!=text:raise IOError('atomic text verification failed')

if __name__=='__main__':finalize({'audit_only':True})
