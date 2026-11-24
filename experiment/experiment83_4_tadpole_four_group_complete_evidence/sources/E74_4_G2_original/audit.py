"""Independent checkpoint audit and factorial synthesis for Experiment 74.4."""
from __future__ import annotations
import csv, fcntl, hashlib, json, math, os
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parent
RESULTS = EXP / "results"
E69 = Path("/model/spromptgl/experiment/experiment69_3_g2_two_rows")
E62 = Path("/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity")
E74 = Path("/model/spromptgl/experiment/experiment74_udag")
E741 = Path("/model/spromptgl/experiment/experiment74_1_rudag")
E742 = Path("/model/spromptgl/experiment/experiment74_2_dcr_rudag")
E743 = Path("/model/spromptgl/experiment/experiment74_3_local_dcr")


def runner():
    import importlib.util, sys
    name = "soft_positive_dcr_744_runner_for_audit"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, EXP / "run.py")
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    r = runner(); r.write_csv(path, rows)


def pair_relation_metrics(state, output):
    import torch
    from sklearn.metrics import roc_auc_score
    r = runner(); hist = r.load_history(); base = hist.load_e69().load_e61().load_base()
    train_y = state["train_same"].detach().cpu().numpy().astype(int)
    train_logits = [x.detach().cpu().numpy().reshape(-1) for x in output["info"]["pair_logits"]]
    unexposed = np.setdiff1d(state["test_np"], state["train_np"], assume_unique=False)
    pairs_np, same_np = base.sample_balanced_pairs(unexposed, state["labels_np"], 4096,
        np.random.default_rng(741100 + state["fold"]), None)
    pairs = torch.as_tensor(pairs_np, dtype=torch.long, device=state["y"].device)
    cadr = r.load_dcr()._cadr_module()
    embeddings = [e(x) for e, x in zip(state["model"].encoders, state["parts"])]
    result = []
    for i, (embedding, scorer, modality) in enumerate(zip(embeddings, state["model"].scorers, r.load_dcr().MODALITIES)):
        test_logits = scorer(cadr.pair_features(embedding[pairs[:,0]], embedding[pairs[:,1]])).detach().cpu().numpy().reshape(-1)
        train_q = 1/(1+np.exp(-np.clip(train_logits[i], -80, 80))); test_q = 1/(1+np.exp(-np.clip(test_logits, -80, 80)))
        result.append({"modality": modality, "train_relation_pair_auc": float(roc_auc_score(train_y, train_logits[i])),
          "unexposed_test_relation_pair_auc_posthoc": float(roc_auc_score(same_np, test_logits)),
          "train_relation_score_mean": float(train_q.mean()), "train_relation_score_std": float(train_q.std(ddof=0)),
          "test_relation_score_mean": float(test_q.mean()), "test_relation_score_std": float(test_q.std(ddof=0)),
          "test_relation_positive_fraction": float((test_q>=.55).mean()), "test_relation_negative_fraction": float((test_q<=.45).mean()),
          "test_relation_uncertain_fraction": float(((test_q>.45)&(test_q<.55)).mean()),
          "test_relation_pair_count": int(len(same_np)), "unexposed_test_patient_count": int(len(unexposed))})
    return result


def intervention_record(group, fold, name, full_metrics, changed_metrics, full_output, changed_output, labels, test_indices):
    fp, cp = np.asarray(full_metrics["pred"]), np.asarray(changed_metrics["pred"]); ff = fp == labels; cf = cp == labels
    fprob, cprob = np.asarray(full_metrics["prob"]), np.asarray(changed_metrics["prob"])
    row = {"group":group,"fold":fold,"intervention":name,"hard_flips":int((fp!=cp).sum()),
      "full_wrong_intervention_correct":int((~ff&cf).sum()),"full_correct_intervention_wrong":int((ff&~cf).sum()),
      "mean_abs_asd_probability_change":float(np.abs(fprob-cprob).mean()),
      "fused_320_relative_l2_change":float((changed_output["fused"].detach()-full_output["fused"].detach()).norm()/full_output["fused"].detach().norm().clamp_min(1e-12)),
      "full_fused_delta_from_X_relative":float(full_output["info"]["fused_delta_relative_l2"]),
      "flipped_test_subject_rows_1based":json.dumps((np.asarray(test_indices)[fp!=cp]+1).astype(int).tolist())}
    for metric in runner().METRICS:
        row[f"full_{metric}"]=float(full_metrics[metric]); row[f"intervention_{metric}"]=float(changed_metrics[metric]);
        row[f"full_minus_intervention_{metric}_pp"]=100*(float(full_metrics[metric])-float(changed_metrics[metric]))
    return row


def audit_one(group, fold, expected_sha):
    import torch
    r = runner(); task_path = r.task_path(group, fold); ckpt_path = r.best_path(group, fold)
    if not r._task_complete(group, fold, expected_sha): raise RuntimeError(f"incomplete task {group} fold {fold}")
    task = json.loads(task_path.read_text()); ckpt_sha = r.sha_file(ckpt_path); checkpoint = r.torch_load(ckpt_path)
    if checkpoint.get("group") != group or checkpoint.get("fold") != fold or checkpoint.get("source_sha256") != expected_sha: raise RuntimeError("checkpoint identity mismatch")
    if checkpoint.get("config") != r.CFG: raise RuntimeError("checkpoint config differs")
    if task.get("group") != group or task.get("fold") != fold or task.get("checkpoint_sha256") != ckpt_sha: raise RuntimeError("task/checkpoint provenance mismatch")
    if task.get("best_epoch_1based") != checkpoint.get("best_epoch_1based") or task.get("epochs_executed") != checkpoint.get("epochs_executed"): raise RuntimeError("best epoch/task progress mismatch")
    state, warmup, identity = r.prepare(fold); state["fold"] = fold; state["group"] = group; state["rudag_identity"] = identity
    if checkpoint["identity"] != identity or task["identity"] != identity: raise RuntimeError("identity mismatch")
    overlap = [377,407] if fold==2 else []
    if identity["membership"]["train_test_overlap_1based"] != overlap: raise RuntimeError("Fold 2 overlap mismatch")
    if fold == 2 and (int(state["labels_np"][376]) != 0 or int(state["labels_np"][406]) != 1): raise RuntimeError("Fold 2 exposed row label encoding mismatch")
    if not np.isin(state["train_pairs_np"], state["train_np"]).all(): raise RuntimeError("pair scope leak")
    if state["train_pairs_np"].shape != (4096,2) or tuple(state["x"].shape)!=(871,320): raise RuntimeError("input/pair shape mismatch")
    if int(state["train_same"].sum().item()) != 2048: raise RuntimeError("training relation pairs are not balanced")
    if not np.isin(state["labels_np"],[0,1]).all(): raise RuntimeError("label encoding mismatch")
    r.load_history().load_bundle(state, checkpoint["best_state"]); state["model"].eval(); state["cam"].eval()
    gcn_calls, cam_calls = [0]*4, [0]
    hooks=[m.register_forward_hook(lambda _m,_i,_o,idx=i:gcn_calls.__setitem__(idx,gcn_calls[idx]+1)) for i,m in enumerate(state["model"].gcn)]
    hooks.append(state["cam"].register_forward_hook(lambda *_:cam_calls.__setitem__(0,cam_calls[0]+1)))
    with torch.enable_grad(): full = r.forward(state, group, training=False, collect=True)
    for h in hooks: h.remove()
    base = r.load_history().load_e69().load_e61().load_base()
    with torch.enable_grad():
        full_metrics = base.classification_metrics(full["logp"], state["test"], state["y"])
        off = r.forward(state, group, training=False, intervention="direct_off")
        direct_loss = full["cls"] - off["cls"]
        direct_grads=[]
        for scorer in state["model"].scorers:
            gs=torch.autograd.grad(direct_loss, tuple(scorer.parameters()), retain_graph=True, allow_unused=True)
            vals=[g.detach().square().sum() for g in gs if g is not None]; direct_grads.append(float(torch.sqrt(torch.stack(vals).sum()).cpu()) if vals else 0.0)
        gate_grads=torch.autograd.grad(full["cls"],state["model"].adaptive_gate_logits,retain_graph=True,allow_unused=True)[0]
        gate_grads=gate_grads.detach().abs().cpu().tolist() if gate_grads is not None else [0.0]*4
        scorer_grads=[]
        for scorer in state["model"].scorers:
            gs=torch.autograd.grad(full["cls"],tuple(scorer.parameters()),retain_graph=True,allow_unused=True)
            vals=[g.detach().square().sum() for g in gs if g is not None]; scorer_grads.append(float(torch.sqrt(torch.stack(vals).sum()).cpu()) if vals else 0.0)
    for metric in r.METRICS:
        if abs(float(full_metrics[metric])-float(checkpoint["metrics"][metric]))>1e-10: raise RuntimeError(f"metric mismatch {metric}")
    if not np.array_equal(full_metrics["pred"], checkpoint["test_prediction"]): raise RuntimeError("hard predictions mismatch")
    if not np.allclose(full_metrics["prob"], checkpoint["test_probability"], atol=1e-8, rtol=0): raise RuntimeError("probability mismatch")
    if gcn_calls != [1,1,1,1] or cam_calls != [1]: raise RuntimeError("unexpected call count")
    if any(not torch.equal(a,b) for a,b in zip(full["unified_adjs"], off["unified_adjs"])): raise RuntimeError("Direct-off changed graph")
    dcr=r.load_dcr(); spec=dcr.GROUP_SPECS[group]
    if float(full["info"]["corr_graph_temperature"]) != float(spec["corr_temperature"]):
        raise RuntimeError("CORR graph temperature differs from the registered group")
    corr_logits=full["info"]["edge_logits"][3]
    corr_raw=full["info"]["edge_q"][3]
    corr_graph=full["info"]["edge_q_graph"][3]
    expected_raw=torch.sigmoid(corr_logits)
    expected_graph=torch.sigmoid(corr_logits/float(spec["corr_temperature"]))
    if not torch.equal(corr_raw,expected_raw) or not torch.equal(corr_graph,expected_graph):
        raise RuntimeError("raw CORR relation supervision and graph temperature paths are not separated")
    if any(not torch.equal(full["info"]["edge_q"][i],full["info"]["edge_q_graph"][i]) for i in range(3)):
        raise RuntimeError("temperature scaling leaked into a non-CORR relation graph")
    raw_pair_loss=torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(x,state["train_same"]) for x in full["info"]["pair_logits"]]).mean()
    if not torch.equal(raw_pair_loss,full["pair"]):
        raise RuntimeError("pair BCE no longer uses unscaled relation logits")
    for index,(base_adj, unified, delta, direct_delta) in enumerate(zip(full["original_adjs"], full["unified_adjs"], full["info"]["disease_delta_adjs"], full["info"]["direct_delta_adjs"])):
        if not torch.equal(delta.diag(), torch.zeros_like(delta.diag())): raise RuntimeError("DeltaA diagonal")
        if not torch.allclose(unified, unified.T, atol=1e-6, rtol=0) or not torch.allclose(delta, delta.T, atol=1e-6, rtol=0): raise RuntimeError("graph asymmetry")
        if (unified<0).any() or ((base_adj==0)&(unified!=0)).any(): raise RuntimeError("negative or added graph edges")
        expected_delta=delta.clamp_min(0) if spec["positive_only"] else delta
        if not torch.equal(direct_delta,expected_delta): raise RuntimeError("direct DeltaA rule mismatch")
        if spec["positive_only"] and (direct_delta<0).any(): raise RuntimeError("G3 retained negative direct increment")
        start,stop=dcr.BLOCK_SLICES[index]; local=full["info"]["disease_residuals_bounded"][index]
        if torch.count_nonzero(local[:,:start]).item() or torch.count_nonzero(local[:,stop:]).item(): raise RuntimeError("modality-local mask mismatch")
        expected_message=spec["direct_scale"]*full["info"]["gates"][index]*local
        if not torch.allclose(expected_message,full["info"]["direct_messages"][index],atol=1e-7,rtol=1e-6): raise RuntimeError("direct scale/gate fusion mismatch")
        expected_weighted=torch.sigmoid(state["model"].fusion_weight[index])*expected_message
        if not torch.allclose(expected_weighted,full["info"]["weighted_direct_messages"][index],atol=1e-7,rtol=1e-6): raise RuntimeError("direct fusion weight mismatch")
    if not all(math.isfinite(v) for v in gate_grads+scorer_grads+direct_grads): raise RuntimeError("classification gradient contains NaN/Inf")
    if direct_grads[3] <= 0: raise RuntimeError("CORR direct residual has no classification gradient")
    labels = state["y"][state["test"]].detach().cpu().numpy().astype(int); interventions=[]
    base_runtime=r.load_history().load_e69().load_e61().load_base()
    for name in ("disease_neutral","corr_neutral","score_permuted","direct_off"):
        rng_before=base_runtime.rng_sha(base_runtime.pack_rng())
        with torch.no_grad():
            changed=r.forward(state, group, training=False, intervention=name); cm=base.classification_metrics(changed["logp"],state["test"],state["y"])
            if name=="disease_neutral":
                if any(torch.count_nonzero(x).item() for x in changed["info"]["disease_delta_adjs"]+changed["info"]["direct_messages"]): raise RuntimeError("disease-neutral residual is not exactly zero")
                if any(not torch.equal(a,b) for a,b in zip(changed["unified_adjs"],changed["original_adjs"])): raise RuntimeError("disease-neutral graph differs from base")
            if name=="corr_neutral":
                if not torch.equal(changed["unified_adjs"][3],changed["original_adjs"][3]): raise RuntimeError("CORR-neutral changed CORR graph")
                if any(not torch.equal(a,b) for a,b in zip(changed["unified_adjs"][:3],full["unified_adjs"][:3])): raise RuntimeError("CORR-neutral altered another modality graph")
            if name=="direct_off" and any(not torch.equal(a,b) for a,b in zip(changed["unified_adjs"],full["unified_adjs"])): raise RuntimeError("Direct-off changed unified graph")
            if name=="score_permuted":
                repeat=r.forward(state, group, training=False, intervention=name)
                if not torch.equal(changed["logp"],repeat["logp"]): raise RuntimeError("non-deterministic score permutation")
        if base_runtime.rng_sha(base_runtime.pack_rng())!=rng_before: raise RuntimeError("intervention consumed global training RNG")
        interventions.append(intervention_record(group,fold,name,full_metrics,cm,full,changed,labels,state["test_np"]))
    relation=pair_relation_metrics(state,full); mechanisms=[]
    gates=[float(.01*torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) if i<3 else float(torch.sigmoid(state["model"].adaptive_gate_logits[i]).detach().cpu()) for i in range(4)]
    for index,diag in enumerate(full["info"]["edge_diagnostics"]):
        mechanisms.append({"group":group,"fold":fold,**diag,**relation[index],"effective_gate_initial":float(checkpoint["initial_gate_values"][index]),"effective_gate_final":gates[index],"adaptive_gate_theta_initial":float(checkpoint["initial_gate_theta"][index]),"adaptive_gate_theta_final":float(state["model"].adaptive_gate_logits[index].detach().cpu()),"relation_scorer_classification_gradient_norm":scorer_grads[index],"adaptive_gate_classification_gradient_abs":gate_grads[index],"relation_scorer_DCR_only_classification_gradient_norm":direct_grads[index],"fused_delta_from_X_relative":float(full["info"]["fused_delta_relative_l2"]),"fold2_label_exposure":fold==2,"train_test_overlap_1based":json.dumps(overlap)})
    audit={"group":group,"fold":fold,"status":"INDEPENDENT_AUDIT_PASS","checkpoint_path":str(ckpt_path),"checkpoint_sha256":ckpt_sha,"source_sha256":expected_sha,"best_epoch_1based":int(checkpoint["best_epoch_1based"]),"epochs_executed":int(checkpoint["epochs_executed"]),"metrics":{m:float(full_metrics[m]) for m in r.METRICS},"test_prediction":checkpoint["test_prediction"],"warmup_sha256":identity["warmup_sha256"],"warmup_rng_sha":warmup["rng_sha"],"split_sha":identity["split_sha"],"pair_sha":identity["pair_sha"],"effective_train_sha":identity["effective_train_sha"],"train_test_overlap_1based":overlap,"pair_training_scope_pass":True,"initial_model_cam_sha":identity["model_cam_sha"],"initial_optimizer_sha":identity["optimizer_sha"],"parameter_count":sum(p.numel() for p in state["model"].parameters())+sum(p.numel() for p in state["cam"].parameters()),"gcn_calls_per_modality":gcn_calls,"cam_calls":cam_calls[0],"direct_residual_classification_scorer_gradient_norms":direct_grads,"classification_scorer_gradient_norms":scorer_grads,"classification_gate_gradient_abs":gate_grads,"rng_trace":checkpoint["rng_trace"],"interventions":interventions,"mechanism":mechanisms}
    del state, checkpoint, full, off; torch.cuda.empty_cache(); return audit


def history(path, group):
    rows=[x for x in read_csv(path) if x.get("group")==group]
    if len(rows)!=10: raise RuntimeError(f"expected ten rows {group}")
    return {int(x["fold"]):{m:float(x[m]) for m in runner().METRICS} for x in rows}


def history_summary(path, group):
    rows=[x for x in read_csv(path) if x.get("group")==group]
    if len(rows)!=1: raise RuntimeError(f"expected summary {group}")
    x=rows[0]; out={"group":group,"folds":10,"source":"historical read-only"}
    for m in runner().METRICS: out[f"{m}_mean_percent"]=float(x[f"{m}_mean_percent"]); out[f"{m}_population_sd_percent"]=float(x[f"{m}_population_sd_percent"])
    return out


def summary(group, audits):
    out={"group":group,"folds":len(audits),"source":"new training; independently audited"}
    for m in runner().METRICS:
        v=np.asarray([a["metrics"][m] for a in audits])*100; out[f"{m}_mean_percent"]=float(v.mean()); out[f"{m}_population_sd_percent"]=float(v.std(ddof=0))
    return out


def paired(name,left,right,scopes):
    rows=[]
    for scope,folds in scopes:
        for m in runner().METRICS:
            d=np.asarray([(left[f][m]-right[f][m])*100 for f in folds]); mean=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.0
            from scipy.stats import t
            half=float(t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d))) if len(d)>1 else 0.0
            rows.append({"comparison":name,"scope":scope,"metric":m,"folds":len(folds),"mean_difference_pp":mean,"wins":int((d>1e-12).sum()),"ties":int((abs(d)<=1e-12).sum()),"losses":int((d< -1e-12).sum()),"descriptive_95ci_low_pp":mean-half,"descriptive_95ci_high_pp":mean+half,"fold_differences_pp":json.dumps({str(f):float((left[f][m]-right[f][m])*100) for f in folds},sort_keys=True)})
    return rows


def rng_audit(group,audits):
    total=exact=0; by_fold={}
    for a in audits:
        old=json.loads((E743/".runtime/tasks"/f"E74.3-G1-Local-DCR_f{a['fold']:02d}.json").read_text()); new={x["epoch"]:x for x in a["rng_trace"]}; ref={x["epoch"]:x for x in old["rng_trace"]}; common=sorted(set(new)&set(ref)); ex=sum(all(new[e][k]==ref[e][k] for k in ("rng_before","rng_after_train","rng_after_eval")) for e in common); total+=len(common); exact+=ex; by_fold[str(a["fold"])]={"common_epochs":len(common),"exact_triplets":ex,"mismatches":len(common)-ex}
    return {"reference":"E74.3-G1; shared epochs only","common_epochs":total,"exact_triplets":exact,"mismatches":total-exact,"by_fold":by_fold}


def cross_group_rng_audit(audits,groups):
    result={}
    for fold in range(1,11):
        traces={g:{x["epoch"]:x for x in next(a for a in audits if a["group"]==g and a["fold"]==fold)["rng_trace"]} for g in groups}
        for i,left in enumerate(groups):
            for right in groups[i+1:]:
                common=sorted(set(traces[left])&set(traces[right])); exact=sum(all(traces[left][e][k]==traces[right][e][k] for k in ("rng_before","rng_after_train","rng_after_eval")) for e in common)
                result[f"{left}|{right}|fold{fold}"]={"common_epochs":len(common),"exact_triplets":exact,"mismatches":len(common)-exact}
    return result


def training_execution():
    events=[]
    with (EXP/".runtime/logs/controller.jsonl").open() as f:
        for line in f:
            try: events.append(json.loads(line))
            except json.JSONDecodeError: pass
    stop=max((i for i,e in enumerate(events) if e.get("event")=="STOP"),default=-1)
    if stop<0: raise RuntimeError("controller missing STOP")
    start=next(e for e in reversed(events[:stop]) if e.get("event")=="START"); end=events[stop]
    return {"worker_limit":start.get("worker_limit"),"max_workers_observed":end.get("max_workers"),"max_global_workers_observed":end.get("max_global_workers"),"wall_seconds":end.get("wall_seconds"),"retry_count":len(end.get("retries",[])),"retries":end.get("retries",[]),"failures":end.get("failures",[]),"oom_events":sum(int(x.get("oom",False)) for x in end.get("retries",[])+end.get("failures",[])),"external_workers_initial":start.get("external_workers_initial")}


def metric_delta(paired_rows, comparison, scope, metric):
    return next(x for x in paired_rows if x["comparison"] == comparison and x["scope"] == scope and x["metric"] == metric)


def existing_handoff_block(content, marker):
    token = marker.encode("utf-8")
    first = content.find(token)
    second = content.find(token, first + len(token)) if first >= 0 else -1
    if first < 0 or second < 0:
        raise RuntimeError(f"handoff marker pair is incomplete: {marker}")
    end = second + len(token)
    if content[end:end + 1] == b"\n":
        end += 1
    return bytes(content[first:end])


def append_handoff(manifest, summaries, paired_rows, mechanisms, interventions, rng_by_group, cross_rng):
    marker = "<!-- EXPERIMENT74_3_LOCAL_DCR_2026-11-06 -->"
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    marker_file = RESULTS / "handoff_append.md"
    historical = [x for x in summaries if x["source"] == "historical read-only"]
    trained = [x for x in summaries if x["source"] == "new training; independently audited"]
    content = [
        marker,
        "## Experiment 74.3：Local-DCR（2026-11-06）",
        "",
        "目的：在E74.2疾病边权增量残差上，仅检验模态局部化、固定3倍残差强度和正增量残差三项预注册改动；不扩展关系编码器、损失或分类主干。",
        f"代码及运行目录：`{EXP}`；首选OSSFS目录因errno 5创建失败，回退至`/model`。E74.2参考为`upload/experiment74-2-dcr-rudag`@`6555308455814af5b6b4c82ceece73eb2333be15`，源文件SHA见实验config与manifest。",
        "G1为模态局部signed DeltaA、lambda=0.02；G2仅将lambda固定为0.06；G3继承G2，但只在直接残差分支使用clamp_min(DeltaA,0)，统一图仍保留负增量抑制。三组均为每模态单一统一图、一次GCN、原CAM/分类器及原优化器与损失。",
        "",
        f"执行状态：{manifest['completed_tasks']}/30任务、{manifest['independent_audits']}/30独立checkpoint审计；设备{manifest['device']}，worker上限{manifest['worker_limit']}、实际全局并发峰值{manifest['max_global_workers_observed']}；用时{manifest['training_execution']['wall_seconds']:.1f}s，失败/重试{manifest['training_execution']['retry_count']}，OOM {manifest['training_execution']['oom_events']}。",
        "",
        "十折结果（mean ± population SD，%）：",
        "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in historical + trained:
        content.append("|" + item["group"] + "|" + "|".join(
            f"{item[f'{metric}_mean_percent']:.4f} ± {item[f'{metric}_population_sd_percent']:.4f}"
            for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
        ) + "|")
    content += ["", "排除Fold 2后的配对ACC变化（E74.3−参考，百分点；其余指标见paired_comparisons.csv）：", "|比较|9折ACC变化|胜/平/负|", "|---|---:|---:|"]
    for item in trained:
        group = item["group"]
        comparisons = [(f"{group}−E74.2", f"{group} vs E74.2"),
                       (f"{group}−E74.1-G1", f"{group} vs E74.1-G1"),
                       (f"{group}−E69.3-G2", f"{group} vs E69.3-G2"),
                       (f"{group}−E62-E2", f"{group} vs E62-E2"),
                       (f"{group}−E74", f"{group} vs E74")]
        for label, key in comparisons:
            row = metric_delta(paired_rows, key, "nine_without_fold2", "acc")
            content.append(f"|{label}|{row['mean_difference_pp']:+.4f}|{row['wins']}/{row['ties']}/{row['losses']}|")
    for label, key in (("G2−G1", "G2 vs G1"), ("G3−G2", "G3 vs G2"), ("G1−E74.2", "G1 vs E74.2")):
        row = metric_delta(paired_rows, key, "nine_without_fold2", "acc")
        content.append(f"|{label}|{row['mean_difference_pp']:+.4f}|{row['wins']}/{row['ties']}/{row['losses']}|")

    content += ["", "机制诊断：各模态训练患者对AUC、未暴露测试患者对事后AUC、关系评分分布、门控、正负边增量、图变化和GCN/DCR消息比例见mechanism.csv。按组平均的融合加权DCR/320维输入范数（%）如下："]
    for group in (x["group"] for x in trained):
        rows = [x for x in mechanisms if x["group"] == group]
        ratios = {mode: 100 * float(np.mean([float(x["fusion_weighted_dcr_message_raw_l2_ratio"]) for x in rows if x["modality"] == mode])) for mode in ("PHENO", "ANAT", "FUNC", "CORR")}
        content.append(f"- {group}: " + ", ".join(f"{mode} {ratios[mode]:.6f}%" for mode in ratios) + ".")
    content += ["", "固定checkpoint干预（每组10折平均full−intervention ACC/Prob-AUC，百分点；硬翻转数）："]
    for group in (x["group"] for x in trained):
        for name in ("disease_neutral", "corr_neutral", "score_permuted", "direct_off"):
            rows = [x for x in interventions if x["group"] == group and x["intervention"] == name]
            acc = float(np.mean([x["full_minus_intervention_acc_pp"] for x in rows]))
            auc = float(np.mean([x["full_minus_intervention_prob_auc_pp"] for x in rows]))
            flips = sum(int(x["hard_flips"]) for x in rows)
            content.append(f"- {group} / {name}: ACC {acc:+.4f} pp, Prob-AUC {auc:+.4f} pp, 硬翻转 {flips}。")
    content += [
        "",
        f"审计：30/30 checkpoint独立读回并复算通过；每折三组warmup、模型/CAM、优化器、划分和患者对初始化身份一致。跨组共享epoch RNG匹配{sum(v['exact_triplets'] for v in cross_rng.values())}/{sum(v['common_epochs'] for v in cross_rng.values())}；与E74.2共有epoch匹配{sum(v['exact_triplets'] for v in rng_by_group.values())}/{sum(v['common_epochs'] for v in rng_by_group.values())}。历史训练曾发生的失败/重试见manifest，正式任务重试数为{manifest['training_execution']['retry_count']}。",
        "评价限制：沿用val=test、外层交叉验证前RFE/标准化及transductive图；Fold 2仍有1-based row 377和407同时进入训练与测试。不得解释为无泄漏独立泛化估计；九折剔除不能消除其他协议限制。",
        f"结论：{manifest['scientific_conclusion']}。{manifest['scientific_conclusion_note']}",
        "下一步：不继续扫描本轮固定系数；优先改为fold-local预处理、独立验证集选模和无标签重叠评估，再检验关系边对应对分类决策的可重复作用。",
        f"报告：`{EXP}/results/report.md`；manifest：`{EXP}/manifest.json`；审计结果：`{EXP}/results/checkpoint_audit.csv`。",
        marker,
        "",
    ]
    block = "\n".join(content).encode("utf-8")
    marker_file.write_bytes(block)
    if not handoff.is_file():
        raise FileNotFoundError(f"required handoff does not exist: {handoff}")
    descriptor = os.open(handoff, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            before.extend(chunk)
        if marker.encode("utf-8") in before:
            existing = existing_handoff_block(before, marker)
            return {"status": "ALREADY_PRESENT", "path": str(handoff), "append_sha256": hashlib.sha256(existing).hexdigest(), "prepared_block_sha256": runner().sha_file(marker_file)}
        def write_all(payload):
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise IOError("short handoff append")
                view = view[written:]
        if not before.endswith(b"\n"):
            write_all(b"\n")
        write_all(block)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        after = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            after.extend(chunk)
        if not after.startswith(before) or not after.endswith(block):
            raise IOError("handoff prefix or appended block verification failed")
        return {"status": "APPENDED_AND_VERIFIED", "path": str(handoff), "append_sha256": runner().sha_file(marker_file), "original_bytes_preserved": len(before)}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_handoff_correction(manifest, paired_rows, audits, mechanisms):
    marker = "<!-- EXPERIMENT74_3_LOCAL_DCR_AUDIT_CORRECTION_2026-11-06 -->"
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    correction_file = RESULTS / "handoff_correction.md"
    r = runner()
    scopes = ("all10", "nine_without_fold2")
    lines = [
        marker,
        "### Experiment 74.3审计更正",
        "",
        "最终文件核验发现首版handoff中G2−G1及G3−G2配对表的左右操作数映射反向。模型训练、checkpoint、逐折预测和独立审计不受影响；已修正`results/paired_comparisons.csv`及`results/report.md`。以下为正确方向，覆盖首版中这两组比较行：",
        "|比较|范围|ACC Δ pp|Hard-AUC Δ pp|Prob-AUC Δ pp|SEN Δ pp|胜/平/负（ACC）|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("G2 vs G1", "G3 vs G2"):
        selected = [x for x in paired_rows if x["comparison"] == label]
        for scope in scopes:
            get = lambda metric: next(x for x in selected if x["scope"] == scope and x["metric"] == metric)
            acc = get("acc")
            lines.append(f"|{label}|{scope}|{acc['mean_difference_pp']:+.4f}|{get('hard_auc')['mean_difference_pp']:+.4f}|{get('prob_auc')['mean_difference_pp']:+.4f}|{get('sen')['mean_difference_pp']:+.4f}|{acc['wins']}/{acc['ties']}/{acc['losses']}|")
    lines += ["", "排除Fold 2后的各组结果（9折mean ± population SD，%）：", "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|---:|"]
    for group in r.GROUPS:
        rows = [x for x in audits if x["group"] == group and x["fold"] != 2]
        lines.append("|" + group + "|" + "|".join(
            f"{100*np.mean([x['metrics'][metric] for x in rows]):.4f} ± {100*np.std([x['metrics'][metric] for x in rows], ddof=0):.4f}"
            for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe")
        ) + "|")
    lines += ["", "Fold 2对10折均值的贡献（10折mean − 排除Fold 2后的9折mean，百分点；仅描述性）：", "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|", "|---|---:|---:|---:|---:|---:|"]
    for group in r.GROUPS:
        all_rows = [x for x in audits if x["group"] == group]
        nine = [x for x in all_rows if x["fold"] != 2]
        values = [100 * (np.mean([x["metrics"][metric] for x in all_rows]) - np.mean([x["metrics"][metric] for x in nine])) for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe")]
        lines.append("|" + group + "|" + "|".join(f"{value:+.4f}" for value in values) + "|")
    lines += ["", "CORR与非CORR关系/传播诊断（每组×模态，30折checkpoint均值）：", "|Group|Modality|Train pair AUC|Unexposed test pair AUC|q mean|Modified edges %|DeltaA/S %|Fused DCR/input %|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in r.GROUPS:
        for modality in ("PHENO", "ANAT", "FUNC", "CORR"):
            rows = [x for x in mechanisms if x["group"] == group and x["modality"] == modality]
            values = [np.mean([float(x[key]) for x in rows]) for key in (
                "train_relation_pair_auc", "unexposed_test_relation_pair_auc_posthoc", "test_relation_score_mean",
            )]
            modified = 100 * np.mean([float(x["modified_edge_ratio"]) for x in rows])
            delta = 100 * np.mean([float(x["deltaA_relative_to_base"]) for x in rows])
            dcr = 100 * np.mean([float(x["fusion_weighted_dcr_message_raw_l2_ratio"]) for x in rows])
            lines.append(f"|{group}|{modality}|{values[0]:.4f}|{values[1]:.4f}|{values[2]:.4f}|{modified:.4f}|{delta:.4f}|{dcr:.8f}|")
    lines += [
        "",
        "各组的Disease-neutral、CORR-neutral、Score-permuted和Direct-off干预均产生少量硬翻转；逐折结果显示这些翻转均为full模型正确、干预后错误，没有纠错样本。平均ACC差异及每组汇总见`results/interventions.csv`。因此可称有小幅可测量的关系对应/残差依赖，但不能称为稳健的大幅诊断增益。",
        f"本次更正不改动训练源码或checkpoint。初版审计脚本SHA：`{manifest['source_integrity']['audit_sha256_before_summary_repair']}`；组间配对方向修复后SHA：`{r.sha_file(EXP / 'audit.py')}`。当前正式审计仍为30/30通过，交叉组共有epoch RNG匹配无差异。",
        marker,
        "",
    ]
    block = "\n".join(lines).encode("utf-8")
    correction_file.write_bytes(block)
    descriptor = os.open(handoff, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            before.extend(chunk)
        if marker.encode("utf-8") in before:
            existing = existing_handoff_block(before, marker)
            return {"status": "ALREADY_PRESENT", "path": str(handoff), "append_sha256": hashlib.sha256(existing).hexdigest(), "prepared_block_sha256": r.sha_file(correction_file)}
        def write_all(payload):
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise IOError("short handoff correction append")
                view = view[written:]
        if not before.endswith(b"\n"):
            write_all(b"\n")
        write_all(block)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        after = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            after.extend(chunk)
        if not after.startswith(before) or not after.endswith(block):
            raise IOError("handoff correction prefix or appended block verification failed")
        return {"status": "APPENDED_AND_VERIFIED", "path": str(handoff), "append_sha256": r.sha_file(correction_file), "original_bytes_preserved": len(before)}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_handoff_execution_note(preflight, manifest):
    marker = "<!-- EXPERIMENT74_3_LOCAL_DCR_EXECUTION_NOTE_2026-11-06 -->"
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    note_file = RESULTS / "handoff_execution_note.md"
    repairs = preflight.get("preflight_repairs", [])
    first_issue = repairs[0] if repairs else None
    description = (
        f"首次轻量预检在结果序列化阶段失败：{first_issue['failure']}；修复为保存统一图Tensor的SHA-256摘要后，完整预检重跑通过。"
        "该次预检未执行正式训练更新或任务。正式30折均第一次启动完成，worker失败/重试/OOM均为0；训练源码SHA在预检和训练期间保持一致。"
        if first_issue else "轻量预检和正式任务均通过。正式30折worker失败/重试/OOM均为0。"
    )
    block = ("\n".join([
        marker,
        "### Experiment 74.3执行记录补充",
        "",
        description,
        f"训练源码SHA：`{manifest['training_source_sha256']}`；最终审计汇总源码SHA：`{manifest['current_audit_source_sha256']}`；原始预检审计脚本SHA：`{manifest['source_integrity']['audit_sha256_before_summary_repair']}`。审计脚本后续更正仅修复组间配对行方向并记录执行异常，未改训练源码、参数或checkpoint；最终独立审计30/30通过。",
        marker,
        "",
    ])).encode("utf-8")
    note_file.write_bytes(block)
    descriptor = os.open(handoff, os.O_RDWR | os.O_APPEND)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        before = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            before.extend(chunk)
        if marker.encode("utf-8") in before:
            existing = existing_handoff_block(before, marker)
            return {"status": "ALREADY_PRESENT", "path": str(handoff), "append_sha256": hashlib.sha256(existing).hexdigest(), "prepared_block_sha256": runner().sha_file(note_file)}
        def write_all(payload):
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise IOError("short handoff execution-note append")
                view = view[written:]
        if not before.endswith(b"\n"):
            write_all(b"\n")
        write_all(block)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        after = bytearray()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            after.extend(chunk)
        if not after.startswith(before) or not after.endswith(block):
            raise IOError("handoff execution-note prefix or append verification failed")
        return {"status": "APPENDED_AND_VERIFIED", "path": str(handoff), "append_sha256": runner().sha_file(note_file), "original_bytes_preserved": len(before)}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def factorial_contrast(name, operands, coefficients, scopes):
    rows=[]; metrics=runner().METRICS
    for scope,folds in scopes:
        for metric in metrics:
            values=np.asarray([sum(coef*operands[key][fold][metric] for key,coef in zip(operands,coefficients))*100 for fold in folds])
            mean=float(values.mean()); sd=float(values.std(ddof=1)) if len(values)>1 else 0.0
            from scipy.stats import t
            half=float(t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values))) if len(values)>1 else 0.0
            rows.append({"comparison":name,"scope":scope,"metric":metric,"folds":len(folds),"mean_difference_pp":mean,
                "wins":int((values>1e-12).sum()),"ties":int((abs(values)<=1e-12).sum()),"losses":int((values< -1e-12).sum()),
                "descriptive_95ci_low_pp":mean-half,"descriptive_95ci_high_pp":mean+half,
                "fold_differences_pp":json.dumps({str(f):float(sum(coef*operands[key][f][metric] for key,coef in zip(operands,coefficients))*100) for f in folds},sort_keys=True)})
    return rows


def report(summaries, paired_rows, audits, mechanisms, interventions, manifest, rng_by_group):
    r=runner(); rows=[]
    lines=[
        "# Experiment 74.4：CORR评分软化与正向局部疾病残差", "",
        f"状态：**{manifest['status']}**；正式任务{manifest['completed_tasks']}/30；独立Checkpoint审计{manifest['independent_audits']}/30；设备`{manifest['device']}`；worker上限{manifest['worker_limit']}、实际峰值{manifest['max_workers_observed']}；耗时{manifest['training_execution']['wall_seconds']:.1f}s；重试{manifest['training_execution']['retry_count']}，OOM {manifest['training_execution']['oom_events']}。", "",
        "## 源码与方法核验", "",
        f"E74.3-G1只读基线锁定分支`upload/experiment74-3-local-dcr`@`{r.CFG['reference']['e743_commit']}`，并校验run/local-DCR/audit/config/manifest SHA；三组均从逐折E61 epoch-10 warmup恢复，未从任何E74.3最佳Checkpoint微调。SPromptGL论文描述GFR、交互图、GCN与语义提示路径；本次保留实际代码损失，不恢复论文中未进入历史总损失的ALoss。BrainPrompt论文使用人口统计患者信息和疾病级语义监督，本次不引入文本或Prompt模块。SABER仅按论文的语义可靠性/决策交互机制阅读；没有审计或复现未确认的官方代码。", "",
        "设计：历史E74.3-G1为T=1/signed/0.02；E74.4-G1为T=2/signed/0.02；G2为T=1/positive-only/0.02；G3为T=2/positive-only/0.02。T仅作用于CORR构图概率，患者对BCE仍使用原始Logit；正增量裁剪仅进入直接DCR残差，统一图和GCN仍保留负边抑制。非CORR图、原损失、优化器及每模态一次GCN均沿用E74.3。", "",
        "历史评价限制：继续使用val=test、CV前RFE/标准化和871节点transductive图；Fold 2的1-based row 377（ASD）及407（TD）同时出现在训练与测试索引。九折描述不能消除其他泄漏与选模偏差，因此不是无偏泛化验证。", "",
        "## 十折结果", "", "均值±population SD，单位%；Hard-AUC为历史硬预测口径，Prob-AUC为连续概率ROC-AUC。", "",
        "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        rows.append(item)
        lines.append("|"+item["group"]+"|"+"|".join(f"{item[f'{m}_mean_percent']:.4f} ± {item[f'{m}_population_sd_percent']:.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"))+"|")
    new={group:{a["fold"]:a["metrics"] for a in audits if a["group"]==group} for group in r.GROUPS}
    hist={"E74.3-G1":history(E743/"results/fold_metrics.csv","E74.3-G1-Local-DCR")}
    for key,path,group in (("E69.3-G2",E69,"E69.3-G2"),("E62-E2",E62,"E62-E2"),("E74",E74,"E74-UDAG"),("E74.1-G1",E741,"G1-RUDAG-U"),("E74.1-G2",E741,"G2-RUDAG-Full"),("E74.2",E742,"E74.2-G1-DCR-RUDAG"),("E74.3-G2",E743,"E74.3-G2-Local-DCR-3x"),("E74.3-G3",E743,"E74.3-G3-Positive-Local-DCR")):
        hist[key]=history(path/"results/fold_metrics.csv",group)
    scopes=[("all10",list(range(1,11))),("nine_without_fold2",[f for f in range(1,11) if f!=2])]
    baselines={"B":hist["E74.3-G1"],"A":new["E74.4-G1-Soft-DCR"],"P":new["E74.4-G2-Positive-DCR"],"AP":new["E74.4-G3-Soft-Positive-DCR"]}
    lines += ["","## 预注册因素比较","","差异为逐折配对后再求均值，单位百分点；W/T/L比较每折差值与0，95%区间为描述性t区间。完整数据见`paired_comparisons.csv`。","","|对比|范围|ACC Δ|Hard-AUC Δ|Prob-AUC Δ|SEN Δ|SPE Δ|ACC W/T/L|","|---|---|---:|---:|---:|---:|---:|---:|"]
    core=["T2−T1 signed (G1−E74.3-G1)","T2−T1 positive (G3−G2)","Positive−signed T1 (G2−E74.3-G1)","Positive−signed T2 (G3−G1)","Temperature×positive interaction"]
    for label in core:
        for scope,_ in scopes:
            selected=[x for x in paired_rows if x["comparison"]==label and x["scope"]==scope]
            by={x["metric"]:x for x in selected}; a=by["acc"]
            lines.append(f"|{label}|{scope}|{by['acc']['mean_difference_pp']:+.4f}|{by['hard_auc']['mean_difference_pp']:+.4f}|{by['prob_auc']['mean_difference_pp']:+.4f}|{by['sen']['mean_difference_pp']:+.4f}|{by['spe']['mean_difference_pp']:+.4f}|{a['wins']}/{a['ties']}/{a['losses']}|")
    lines += ["","### 新组相对历史参照"]
    for group in r.GROUPS:
        for ref in ("E69.3-G2","E74.3-G1","E74.2","E62-E2","E74","E74.1-G1","E74.1-G2","E74.3-G2","E74.3-G3"):
            key=f"{group} vs {ref}"
            values=[next(x for x in paired_rows if x["comparison"]==key and x["scope"]=="all10" and x["metric"]==m) for m in ("acc","hard_auc","prob_auc","sen","spe")]
            lines.append(f"- `{key}`："+", ".join(f"{x['metric']} {x['mean_difference_pp']:+.4f} pp ({x['wins']}/{x['ties']}/{x['losses']})" for x in values)+"。")
    lines += ["","## 排除Fold 2", "","九折均值±population SD（%），仅为描述性配对敏感性结果。","","|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|","|---|---:|---:|---:|---:|---:|"]
    nine_sets={**new,"E74.3-G1":hist["E74.3-G1"],"E69.3-G2":history(E69/"results/fold_metrics.csv","E69.3-G2")}
    for group in list(r.GROUPS)+["E74.3-G1","E69.3-G2"]:
        vals=nine_sets[group]
        folds=[vals[f] for f in range(1,11) if f!=2]
        lines.append("|"+group+"|"+"|".join(f"{100*np.mean([x[m] for x in folds]):.4f} ± {100*np.std([x[m] for x in folds],ddof=0):.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe"))+"|")
    lines += ["","九折逐折配对差（新组−历史参照，百分点）："]
    for group in r.GROUPS:
        for reference in ("E69.3-G2","E74.3-G1"):
            key=f"{group} vs {reference}"; values=[next(x for x in paired_rows if x["comparison"]==key and x["scope"]=="nine_without_fold2" and x["metric"]==m) for m in ("acc","hard_auc","prob_auc","sen","spe")]
            lines.append(f"- `{key}`："+", ".join(f"{x['metric']} {x['mean_difference_pp']:+.4f} pp" for x in values)+f"；ACC W/T/L {values[0]['wins']}/{values[0]['ties']}/{values[0]['losses']}。")
    lines += ["","## CORR机制与传播", "","每个数字为30份checkpoint的CORR模态均值；机制明细按group/fold/modality记录于`mechanism.csv`。q_raw为未缩放疾病关系概率，q_graph为只用于构图的温度后概率；T=2是固定软化，不是验证集概率校准。","","|Group|T|Logit均值/SD|q_raw均值/SD|q_graph均值/SD|q_raw 0.45–0.55|q_graph 0.45–0.55|未暴露测试对AUC|ΔA/S %|DCR消息/X %|","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in r.GROUPS:
        rr=[x for x in mechanisms if x["group"]==group and x["modality"]=="CORR"]
        vals=[np.mean([float(x[k]) for x in rr]) for k in ("corr_graph_temperature","relation_logit_mean","relation_logit_std","q_raw_mean","q_raw_std","q_graph_mean","q_graph_std","q_raw_uncertain_fraction","q_graph_uncertain_fraction","unexposed_test_relation_pair_auc_posthoc","deltaA_relative_to_base","fusion_weighted_dcr_message_raw_l2_ratio")]
        lines.append("|"+group+"|"+f"{vals[0]:.1f}|{vals[1]:.4f}/{vals[2]:.4f}|{vals[3]:.4f}/{vals[4]:.4f}|{vals[5]:.4f}/{vals[6]:.4f}|{100*vals[7]:.3f}%|{100*vals[8]:.3f}%|{vals[9]:.4f}|{100*vals[10]:.5f}%|{100*vals[11]:.7f}%|")
    lines += ["","## 固定Checkpoint干预", "","干预不重训、不改标签构图。翻转索引以全局1-based行号记录，特别核对Fold 2中是否涉及377/407。","","|Group|Intervention|ACC Δ full−intervention|Hard-AUC Δ|Prob-AUC Δ|Flip|Full错→干预对|Full对→干预错|Fold2 flips include 377/407|","|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for group in r.GROUPS:
        for name in ("disease_neutral","corr_neutral","score_permuted","direct_off"):
            ir=[x for x in interventions if x["group"]==group and x["intervention"]==name]
            exposed=[]
            for x in ir:
                if x["fold"]==2:
                    ids=json.loads(x["flipped_test_subject_rows_1based"])
                    exposed += [i for i in ids if i in (377,407)]
            lines.append(f"|{group}|{name}|{np.mean([x['full_minus_intervention_acc_pp'] for x in ir]):+.4f}|{np.mean([x['full_minus_intervention_hard_auc_pp'] for x in ir]):+.4f}|{np.mean([x['full_minus_intervention_prob_auc_pp'] for x in ir]):+.4f}|{sum(x['hard_flips'] for x in ir)}|{sum(x['full_wrong_intervention_correct'] for x in ir)}|{sum(x['full_correct_intervention_wrong'] for x in ir)}|{json.dumps(sorted(set(exposed)))}|")
    best_acc=max((x for x in summaries if x["group"] in r.GROUPS),key=lambda x:x["acc_mean_percent"])
    best_hard=max((x for x in summaries if x["group"] in r.GROUPS),key=lambda x:x["hard_auc_mean_percent"])
    best_sen=max((x for x in summaries if x["group"] in r.GROUPS),key=lambda x:x["sen_mean_percent"])
    conclusion=manifest["scientific_conclusion"]
    lines += ["","组别峰值："+f"ACC `{best_acc['group']}` {best_acc['acc_mean_percent']:.4f}%；Hard-AUC `{best_hard['group']}` {best_hard['hard_auc_mean_percent']:.4f}%；SEN `{best_sen['group']}` {best_sen['sen_mean_percent']:.4f}%。E69.3-G2目标ACC 92.0742%、Hard-AUC 91.8995%、SEN 89.5854%。"]
    lines += ["","## 审计与结论", "",f"30/30最佳Checkpoint独立重建并复算通过；硬预测严格一致，概率容差1e-8；每折三组初始模型/CAM、优化器、warmup、患者对与划分一致。单步更新与恢复重放通过，组间共有epoch RNG核对{sum(x['exact_triplets'] for x in manifest['cross_group_rng_audit'].values())}/{sum(x['common_epochs'] for x in manifest['cross_group_rng_audit'].values())}，对E74.3-G1相同epoch核对{sum(x['exact_triplets'] for x in rng_by_group.values())}/{sum(x['common_epochs'] for x in rng_by_group.values())}。", "",f"判定：**{conclusion}**。该判断要求因素比较在全十折及排除Fold 2后仍有方向性支持，并由中性化/置换干预显示合理作用；不表示无偏泛化。建议后续先处理fold-local预处理、独立验证选模及Fold 2重叠，再复核关系调整的可复现性；本轮不追加温度/尺度搜索。", "",f"训练源码SHA：`{manifest['training_source_sha256']}`；审计源码SHA：`{manifest['current_audit_source_sha256']}`。实验目录：`{EXP}`；manifest：`{EXP}/manifest.json`。"]
    (RESULTS/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def append_handoff744(manifest, summaries, paired_rows, interventions):
    handoff=Path("/model/spromptgl/experiment/handoff.md")
    marker="<!-- EXPERIMENT_74_4_SOFT_POSITIVE_DCR -->"; block_file=RESULTS/"handoff_append.md"
    lines=[marker,"## Experiment 74.4：CORR评分软化与正向局部疾病残差（2026-11-09）","",
        f"目的：在E74.3-G1基础上固定0.02直接传播尺度，检验CORR构图Logit温度T=2软化和直接残差positive-only两个正交因素。目录`{EXP}`；preferred OSSFS目录真实报OSError(5)，已记录并回退至`/model`。只读E74.3基线锁定`{runner().CFG['reference']['e743_branch']}`@`{runner().CFG['reference']['e743_commit']}`。",
        "组别：E74.4-G1=T2/signed；G2=T1/positive；G3=T2/positive；E74.3-G1=T1/signed只读。不使用Prompt、文本模块、新损失或新参数。T仅改CORR构图概率，BCE仍用原始Logit；positive-only仅作用于直接DCR，统一GCN图仍含负疾病增量。","",
        f"执行：{manifest['completed_tasks']}/30训练、{manifest['independent_audits']}/30独立checkpoint审计；{manifest['device']}，worker峰值{manifest['max_workers_observed']}，耗时{manifest['training_execution']['wall_seconds']:.1f}s，重试{manifest['training_execution']['retry_count']}，OOM {manifest['training_execution']['oom_events']}。","","10折mean±population SD（%）：","|Group|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|"]
    for x in summaries:
        if x["group"] not in ("E69.3-G2","E74.3-G1-Local-DCR","E74.4-G1-Soft-DCR","E74.4-G2-Positive-DCR","E74.4-G3-Soft-Positive-DCR"): continue
        lines.append("|"+x["group"]+"|"+"|".join(f"{x[f'{m}_mean_percent']:.4f}±{x[f'{m}_population_sd_percent']:.4f}" for m in ("acc","hard_auc","prob_auc","sen","spe","asd_f1"))+"|")
    lines += ["","因素效应、九折比较、历史对照、CORR q_raw/q_graph及四种固定Checkpoint干预见实验`results/paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`和`report.md`。Fold 2历史重叠row 377/407已在独立审计和干预翻转中核对。","","审计：E74.3-G1的T1 signed前向与图逐张量一致；三组均恢复相同E61 epoch-10 warmup/优化器/数据索引，组间共有epoch RNG一致；30/30独立读回，根模型和历史文件保持只读。",f"结论：**{manifest['scientific_conclusion']}**。历史val=test、CV前预处理和transductive图仍限制泛化解释；即使达到目标也需在无重叠、fold-local预处理和独立验证选模下复核。manifest `{EXP}/manifest.json`，报告 `{EXP}/results/report.md`。",marker,""]
    block="\n".join(lines).encode("utf-8"); block_file.write_bytes(block)
    if not handoff.is_file(): raise FileNotFoundError(handoff)
    fd=os.open(handoff,os.O_RDWR|os.O_APPEND)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX); os.lseek(fd,0,os.SEEK_SET); before=bytearray()
        while True:
            chunk=os.read(fd,1<<20)
            if not chunk: break
            before.extend(chunk)
        if marker.encode() in before: return {"status":"ALREADY_PRESENT","path":str(handoff)}
        payload=(b"\n" if before and not before.endswith(b"\n") else b"")+block
        view=memoryview(payload)
        while view:
            n=os.write(fd,view)
            if n<=0: raise IOError("short handoff append")
            view=view[n:]
        os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); after=bytearray()
        while True:
            chunk=os.read(fd,1<<20)
            if not chunk: break
            after.extend(chunk)
        if not after.startswith(before) or not after.endswith(block): raise IOError("handoff append verification failed")
        return {"status":"APPENDED_AND_VERIFIED","path":str(handoff),"original_bytes_preserved":len(before),"append_sha256":runner().sha_file(block_file)}
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)


def append_handoff744_correction(manifest, paired_rows, interventions):
    handoff=Path("/model/spromptgl/experiment/handoff.md")
    marker="<!-- EXPERIMENT_74_4_AUDIT_DIRECTION_CORRECTION -->"; block_file=RESULTS/"handoff_audit_correction.md"
    rows=[]
    for scope in ("all10","nine_without_fold2"):
        metrics=[next(x for x in paired_rows if x["comparison"]=="E74.4-G2-Positive-DCR vs E74.3-G1" and x["scope"]==scope and x["metric"]==m) for m in ("acc","hard_auc","sen")]
        rows.append(f"|{scope}|"+"|".join(f"{x['mean_difference_pp']:+.4f} pp ({x['wins']}/{x['ties']}/{x['losses']})" for x in metrics)+"|")
    g2=[x for x in interventions if x["group"]=="E74.4-G2-Positive-DCR" and x["intervention"] in ("disease_neutral","score_permuted")]
    positive=sum(int(x["full_correct_intervention_wrong"]) for x in g2)
    reverse=sum(int(x["full_wrong_intervention_correct"]) for x in g2)
    lines=[marker,"### Experiment 74.4审计判定方向更正","",
        "追加更正：首版`manifest.json`把固定Checkpoint干预中的翻转方向解释反了。`full_correct_intervention_wrong`表示完整模型判对、干预后判错，是完整疾病关系分支的正向贡献；`full_wrong_intervention_correct`才是移除关系后纠正。训练、Checkpoint、预测和指标未改动；仅更正审计解释及GO/PARTIAL判定，并再次完整独立读回30个Checkpoint。","",
        "E74.4-G2-Positive-DCR相对E74.3-G1的配对差（ACC/Hard-AUC/SEN；W/T/L）：","|范围|ACC|Hard-AUC|SEN|","|---|---:|---:|---:|"]+rows+[
        "","G2固定Checkpoint的Disease-neutral与Score-permuted合计：完整模型正确、干预后错误 "+str(positive)+" 例；完整模型错误、干预后纠正 "+str(reverse)+" 例。Disease-neutral的ACC和Hard-AUC均有正向 full−intervention 差；因此在预注册规则下判为GO候选，但效果幅度小、ACC逐折多数持平，需保守解释。","",
        f"更正前结论：PARTIAL；更正后结论：**{manifest['scientific_conclusion']}**。审计源码SHA：预检/首版`{manifest['source_integrity']['audit_source_sha256_at_preflight']}`，更正前`406001c058b799b354f6583ce622d92be5fdfe854ae5c2a5262d47cc2cfc2400`，更正后`{manifest['current_audit_source_sha256']}`。30/30 checkpoint重新审计通过，训练源码SHA与训练产物不变。详情见`{EXP}/results/report.md`及`audit_status.json`。",marker,""]
    block="\n".join(lines).encode("utf-8"); block_file.write_bytes(block)
    if not handoff.is_file(): raise FileNotFoundError(handoff)
    fd=os.open(handoff,os.O_RDWR|os.O_APPEND)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX); os.lseek(fd,0,os.SEEK_SET); before=bytearray()
        while True:
            chunk=os.read(fd,1<<20)
            if not chunk: break
            before.extend(chunk)
        if marker.encode() in before: return {"status":"ALREADY_PRESENT","path":str(handoff)}
        payload=(b"\n" if before and not before.endswith(b"\n") else b"")+block; view=memoryview(payload)
        while view:
            n=os.write(fd,view)
            if n<=0: raise IOError("short correction append")
            view=view[n:]
        os.fsync(fd); os.lseek(fd,0,os.SEEK_SET); after=bytearray()
        while True:
            chunk=os.read(fd,1<<20)
            if not chunk: break
            after.extend(chunk)
        if not after.startswith(before) or not after.endswith(block): raise IOError("correction append verification failed")
        return {"status":"APPENDED_AND_VERIFIED","path":str(handoff),"original_bytes_preserved":len(before),"append_sha256":runner().sha_file(block_file)}
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN); os.close(fd)


def finalize():
    import torch
    r=runner(); preflight=json.loads((RESULTS/"preflight.json").read_text())
    if preflight.get("status")!="PASS": raise RuntimeError("preflight not PASS")
    current=r.source_manifest(); prior=preflight["source_manifest"]; audit_path=str((EXP/"audit.py").resolve())
    for path,digest in prior.items():
        if path!=audit_path and current.get(path)!=digest: raise RuntimeError(f"protected source changed: {path}")
    tasks=[]
    for group in r.GROUPS:
        for fold in range(1,11): tasks.append(json.loads(r.task_path(group,fold).read_text()))
    if any(t.get("source_sha256")!=tasks[0].get("source_sha256") for t in tasks): raise RuntimeError("task source hashes differ")
    expected=tasks[0]["source_sha256"]; audits=[]
    for i,(group,fold) in enumerate(( (g,f) for g in r.GROUPS for f in range(1,11)),1):
        audits.append(audit_one(group,fold,expected)); print(f"independent checkpoint audit {i}/30 PASS",flush=True)
    rng={g:rng_audit(g,[a for a in audits if a["group"]==g]) for g in r.GROUPS}
    if any(v["mismatches"] for v in rng.values()): raise RuntimeError(f"RNG mismatch {rng}")
    cross_rng=cross_group_rng_audit(audits,r.GROUPS)
    if any(v["common_epochs"] == 0 or v["mismatches"] for v in cross_rng.values()): raise RuntimeError(f"cross-group RNG mismatch {cross_rng}")
    for fold in range(1,11):
        same_fold=[a for a in audits if a["fold"]==fold]
        identities={(a["warmup_sha256"],a["warmup_rng_sha"],a["split_sha"],a["pair_sha"],a["effective_train_sha"],a["initial_model_cam_sha"],a["initial_optimizer_sha"]) for a in same_fold}
        if len(same_fold)!=3 or len(identities)!=1: raise RuntimeError(f"group initialization identity mismatch in fold {fold}")
    histories=[history_summary(E69/"results/summary.csv","E69.3-G2"),history_summary(E62/"results/summary.csv","E62-E2"),history_summary(E74/"results/summary.csv","E74-UDAG"),history_summary(E741/"results/summary.csv","G1-RUDAG-U"),history_summary(E741/"results/summary.csv","G2-RUDAG-Full"),history_summary(E742/"results/summary.csv","E74.2-G1-DCR-RUDAG"),history_summary(E743/"results/summary.csv","E74.3-G1-Local-DCR"),history_summary(E743/"results/summary.csv","E74.3-G2-Local-DCR-3x"),history_summary(E743/"results/summary.csv","E74.3-G3-Positive-Local-DCR")]
    new_by_group={g:[a for a in audits if a["group"]==g] for g in r.GROUPS}; summaries=histories+[summary(g,a) for g,a in new_by_group.items()]
    new={g:{a["fold"]:a["metrics"] for a in new_by_group[g]} for g in r.GROUPS}
    hist_folds={"E69.3-G2":history(E69/"results/fold_metrics.csv","E69.3-G2"),"E62-E2":history(E62/"results/fold_metrics.csv","E62-E2"),"E74":history(E74/"results/fold_metrics.csv","E74-UDAG"),"E74.1-G1":history(E741/"results/fold_metrics.csv","G1-RUDAG-U"),"E74.1-G2":history(E741/"results/fold_metrics.csv","G2-RUDAG-Full"),"E74.2":history(E742/"results/fold_metrics.csv","E74.2-G1-DCR-RUDAG"),"E74.3-G1":history(E743/"results/fold_metrics.csv","E74.3-G1-Local-DCR"),"E74.3-G2":history(E743/"results/fold_metrics.csv","E74.3-G2-Local-DCR-3x"),"E74.3-G3":history(E743/"results/fold_metrics.csv","E74.3-G3-Positive-Local-DCR")}
    scopes=[("all10",list(range(1,11))), ("nine_without_fold2",[f for f in range(1,11) if f!=2])]; paired_rows=[]
    for group,alist in new_by_group.items():
        left={a["fold"]:a["metrics"] for a in alist}
        for reference in ("E74.2","E74.1-G1","E74.1-G2","E74","E69.3-G2","E62-E2","E74.3-G1","E74.3-G2","E74.3-G3"):
            paired_rows.extend(paired(f"{group} vs {reference}",left,hist_folds[reference],scopes))
    A=new["E74.4-G1-Soft-DCR"]; P=new["E74.4-G2-Positive-DCR"]; AP=new["E74.4-G3-Soft-Positive-DCR"]; B=hist_folds["E74.3-G1"]
    for label,left,right in (("T2−T1 signed (G1−E74.3-G1)",A,B),("T2−T1 positive (G3−G2)",AP,P),
                             ("Positive−signed T1 (G2−E74.3-G1)",P,B),("Positive−signed T2 (G3−G1)",AP,A)):
        paired_rows.extend(paired(label,left,right,scopes))
    paired_rows.extend(factorial_contrast("Temperature×positive interaction",{"AP":AP,"A":A,"P":P,"B":B},[1,-1,-1,1],scopes))
    fold_rows=[]; mechanisms=[]; interventions=[]
    for a in audits:
        fold_rows.append({"group":a["group"],"fold":a["fold"],"status":a["status"],**a["metrics"],"best_epoch_1based":a["best_epoch_1based"],"epochs_executed":a["epochs_executed"],"checkpoint_sha256":a["checkpoint_sha256"],"train_test_overlap_1based":json.dumps(a["train_test_overlap_1based"])})
        mechanisms.extend(a["mechanism"]); interventions.extend(a["interventions"])
    write_csv(RESULTS/"fold_metrics.csv",fold_rows); write_csv(RESULTS/"summary.csv",summaries); write_csv(RESULTS/"paired_comparisons.csv",paired_rows); write_csv(RESULTS/"mechanism.csv",mechanisms); write_csv(RESULTS/"interventions.csv",interventions); write_csv(RESULTS/"checkpoint_audit.csv",[{k:v for k,v in a.items() if k not in ("interventions","mechanism","rng_trace","test_prediction")} for a in audits])
    new_summ={g:summary(g,new_by_group[g]) for g in r.GROUPS}
    effect_support={}
    for group in r.GROUPS:
        comparisons=[f"{group} vs E74.3-G1"]
        positive_all=any(metric_delta(paired_rows,key,"all10",metric)["mean_difference_pp"]>0 and
                         metric_delta(paired_rows,key,"all10",metric)["wins"]>metric_delta(paired_rows,key,"all10",metric)["losses"]
                         for key in comparisons for metric in ("acc","hard_auc","sen"))
        positive_nine=any(metric_delta(paired_rows,key,"nine_without_fold2",metric)["mean_difference_pp"]>0 and
                          metric_delta(paired_rows,key,"nine_without_fold2",metric)["wins"]>metric_delta(paired_rows,key,"nine_without_fold2",metric)["losses"]
                          for key in comparisons for metric in ("acc","hard_auc","sen"))
        mech=[x for x in interventions if x["group"]==group and x["intervention"] in ("disease_neutral","score_permuted")]
        positive_contribution=sum(int(x["full_correct_intervention_wrong"]) for x in mech)
        negative_contribution=sum(int(x["full_wrong_intervention_correct"]) for x in mech)
        mechanism_support=positive_contribution>negative_contribution and any(float(x["full_minus_intervention_acc_pp"])>0 or float(x["full_minus_intervention_hard_auc_pp"])>0 for x in mech)
        effect_support[group]={"all10":positive_all,"nine_without_fold2":positive_nine,"intervention":mechanism_support,
            "full_correct_to_intervention_wrong":positive_contribution,"full_wrong_to_intervention_correct":negative_contribution}
    best_group=max(r.GROUPS,key=lambda g:new_summ[g]["acc_mean_percent"])
    if any(v["all10"] and v["nine_without_fold2"] and v["intervention"] for v in effect_support.values()): judgment="GO"
    elif any(v["all10"] or v["nine_without_fold2"] or v["intervention"] for v in effect_support.values()): judgment="PARTIAL"
    else: judgment="NO-GO"
    all_other_sources_match=all(current.get(path)==digest for path,digest in prior.items() if path!=audit_path)
    current_audit_sha=r.sha_file(EXP/"audit.py")
    training=training_execution(); manifest={"experiment":r.CFG["experiment"],"status":"COMPLETE_AUDITED","experiment_root":str(EXP),"preferred_root":r.CFG["preferred_root"],"root_fallback_reason":r.CFG["root_fallback_reason"],"expected_tasks":30,"completed_tasks":30,"independent_audits":30,"checkpoint_audit":"PASS 30/30","device":torch.cuda.get_device_name(0),"device_count":torch.cuda.device_count(),"worker_limit":training["worker_limit"],"max_workers_observed":training["max_workers_observed"],"max_global_workers_observed":training["max_global_workers_observed"],"failed_attempt_records":len(training["retries"]),"oom_events":training["oom_events"],"training_execution":training,"warmup_retrained":False,"source_sha256":expected,"training_source_sha256":expected,"preflight_source_sha256":preflight["source_sha256"],"preflight_repairs":preflight.get("preflight_repairs",[]),"current_audit_source_sha256":current_audit_sha,"source_integrity":{"training_source_sha256":expected,"preflight_source_sha256":preflight["source_sha256"],"audit_source_sha256_at_preflight":prior[audit_path],"current_audit_source_sha256":current_audit_sha,"audit_script_changed_after_training":current[audit_path]!=prior[audit_path],"audit_repairs":[{"before_sha256":prior[audit_path],"after_sha256":"406001c058b799b354f6583ce622d92be5fdfe854ae5c2a5262d47cc2cfc2400","reason":"Added the missing new-by-group fold metric mapping for factorial summary; full independent audit rerun."},{"before_sha256":"406001c058b799b354f6583ce622d92be5fdfe854ae5c2a5262d47cc2cfc2400","after_sha256":current_audit_sha,"reason":"Corrected intervention direction: a full-model correct case becoming wrong after disease-edge ablation indicates a positive contribution. Full independent audit rerun."}],"audit_repair_scope":"summary and GO/PARTIAL classification only; no training code, checkpoint, prediction, or metric changed.","all_non_audit_source_hashes_match_preflight":all_other_sources_match},"rng_audit":rng,"cross_group_rng_audit":cross_rng,"scientific_conclusion":judgment,"scientific_conclusion_note":"GO requires at least one fixed group to improve ACC, Hard-AUC, or SEN versus E74.3-G1 with positive mean paired change and more folds improving than worsening, the same metric supported after excluding Fold 2, and a directionally beneficial neutralization or score-permutation intervention (full model correct to ablated model wrong exceeds reverse flips). PARTIAL marks incomplete evidence; NO-GO marks no predeclared performance/mechanism support.","fold2_label_exposure_1based":[377,407],"best_group_by_acc":best_group,"effect_support_by_group":effect_support,"protected_sources":current}
    manifest["effect_support_by_group"]=effect_support
    report(summaries,paired_rows,audits,mechanisms,interventions,manifest,rng)
    handoff_result=append_handoff744(manifest,summaries,paired_rows,interventions)
    correction_result=append_handoff744_correction(manifest,paired_rows,interventions)
    manifest["handoff_update"]={"experiment_record":handoff_result,"audit_direction_correction":correction_result}
    r.atomic_json(RESULTS/"audit_status.json",{"status":"PASS","audited_checkpoints":30,"expected_checkpoints":30,"prediction_reproduction":"30/30 exact hard predictions; probabilities atol=1e-8","rng_audit":rng,"cross_group_rng_audit":cross_rng,"fold2_overlap":[377,407],"best_group_by_acc":best_group,"scientific_conclusion":judgment,"effect_support_by_group":effect_support,"handoff_update":manifest["handoff_update"]})
    r.atomic_json(EXP/"manifest.json",manifest)
    print(f"AUDIT PASS 30/30; conclusion={judgment}; best_acc_group={best_group}; handoff={handoff_result['status']}",flush=True)
    return manifest

if __name__=="__main__": finalize()
