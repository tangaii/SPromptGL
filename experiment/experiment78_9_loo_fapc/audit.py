#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

import run as runner

HERE, RESULTS, GROUPS, BASE = runner.HERE, runner.RESULTS, runner.GROUPS, runner.BASE
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
E78_8 = Path("/model/spromptgl/experiment/experiment78_8_fapc")


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temp = Path(path).with_name(Path(path).name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _summary(rows, group, scope="all10"):
    rows = [row for row in rows if row["group"] == group and (scope == "all10" or int(row["fold"]) != 2)]
    result = {"group": group, "scope": scope, "n_folds": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows]) * 100
        result[f"{metric}_mean_percent"] = float(values.mean()) if len(values) else None
        result[f"{metric}_population_sd_percent"] = float(values.std(ddof=0)) if len(values) else None
    return result


def _paired(name, left, right, metric_rows, folds):
    lookup_l = {int(row["fold"]): row for row in metric_rows if row["group"] == left}
    lookup_r = {int(row["fold"]): row for row in metric_rows if row["group"] == right}
    shared = [fold for fold in folds if fold in lookup_l and fold in lookup_r]
    result = []
    for metric in METRICS:
        diffs = np.asarray([(float(lookup_l[f][metric]) - float(lookup_r[f][metric])) * 100 for f in shared])
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        half = float(t.ppf(0.975, len(diffs)-1) * sd / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        result.append({"comparison": name, "metric": metric, "scope": "all10" if len(shared) == 10 else "exclude_fold2",
                       "n_folds": len(shared), "mean_difference_pp": mean, "sample_sd_difference_pp": sd,
                       "ci95_low_pp": mean-half, "ci95_high_pp": mean+half,
                       "wins": int((diffs > 1e-12).sum()), "ties": int((np.abs(diffs) <= 1e-12).sum()),
                       "losses": int((diffs < -1e-12).sum()),
                       "fold_differences_pp": json.dumps({str(f): float((float(lookup_l[f][metric])-float(lookup_r[f][metric]))*100) for f in shared})})
    return result


def _load_pair(group, fold, source_sha):
    task_path, best_path, latest_path, _ = runner.task_paths(group, fold)
    task = json.loads(task_path.read_text())
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_sha:
        raise RuntimeError(f"task status/source mismatch: {group} fold {fold}")
    if runner.sha_file(best_path) != task.get("checkpoint_sha256"):
        raise RuntimeError(f"best checkpoint SHA mismatch: {group} fold {fold}")
    if best.get("source_sha256") != source_sha or latest.get("source_sha256") != source_sha:
        raise RuntimeError(f"checkpoint source mismatch: {group} fold {fold}")
    state, warmup, identity = runner.prepare(fold)
    if identity != task.get("identity") or identity.get("warmup_sha256") != warmup.get("sha256", identity.get("warmup_sha256")):
        if identity != task.get("identity"):
            raise RuntimeError("base/warmup identity differs from task")
    module = runner._make_module(state, group)
    optimizer = runner._new_optimizer(module)
    runner._restore(state, module, optimizer, best["best_state"])
    module.eval()
    return task, best, latest, state, module, optimizer


def audit_checkpoint(group, fold, source_sha):
    task, cp, latest, state, module, optimizer = _load_pair(group, fold, source_sha)
    support = runner.support_ids(state)
    if np.intersect1d(support, state["test_np"]).size:
        raise RuntimeError("test patient entered legal support")
    if fold == 2 and np.intersect1d(support, np.asarray([376, 406])).size:
        raise RuntimeError("historical overlap entered support")
    test_ids = state["test_np"].astype(np.int64)
    with torch.no_grad():
        out, details = runner._forward(state, module, training=False, support=support, query=test_ids)
        metrics = runner._metrics(out["logp"], state, details["query_ids"])
        train_out, train_details = runner._forward(state, module, training=False, support=support, query=support)
        train_metrics = runner._metrics(train_out["logp"], state, train_details["query_ids"])
    for key in METRICS:
        if not np.isclose(float(metrics[key]), float(cp["metrics"][key]), atol=1e-12, rtol=0):
            raise RuntimeError(f"independent metric mismatch {key}: {metrics[key]} vs {cp['metrics'][key]}")
    pred = metrics["pred"].detach().cpu().numpy()
    prob = metrics["prob"].detach().cpu().numpy()
    if not np.array_equal(pred, np.asarray(cp["test_prediction"])):
        raise RuntimeError("independent predictions differ from saved best")
    if not np.array_equal(prob, np.asarray(cp["test_probability"])):
        if not np.allclose(prob, np.asarray(cp["test_probability"]), atol=2e-7, rtol=0):
            raise RuntimeError("independent probabilities differ from saved best")

    with torch.no_grad():
        train_aux_acc = float((train_details["semantic_logp"].argmax(1) == state["y"][torch.as_tensor(support, device=state["y"].device)]).float().mean().cpu())
        test_aux_acc = float((details["semantic_logp"].argmax(1) == state["y"][torch.as_tensor(test_ids, device=state["y"].device)]).float().mean().cpu())
        train_darg_acc = float(train_metrics["acc"])
        test_darg_acc = float(metrics["acc"])
        loo_dirs = train_details["prototypes"]
        full_dirs = details["prototypes"][:1].expand(len(support), -1, -1)
        directional_cos = (loo_dirs * full_dirs).sum(-1)
        angles = torch.rad2deg(torch.acos(directional_cos.clamp(-1, 1)))
        loo_means = train_details["prototype_means"]
        full_means = details["prototype_means"][:1].expand(len(support), -1, -1)
        mean_shift = torch.linalg.vector_norm(loo_means-full_means, dim=-1)
        counts = train_details["support_counts"].detach().cpu().numpy()

    trace = task.get("rng_trace", [])
    ref_group = runner.CFG["group_specs"][group]["reference_group"]
    ref_task_path = E78_8 / "runtime/tasks" / f"{ref_group}_f{fold:02d}.json"
    ref_task = json.loads(ref_task_path.read_text())
    ref_rng = {int(item["epoch"]): item for item in ref_task.get("rng_trace", [])}
    compared = 0; mismatches = 0
    for item in trace:
        ref = ref_rng.get(int(item["epoch"]))
        if ref is None:
            continue
        compared += 1
        mismatches += sum(item[key] != ref[key] for key in ("rng_before", "rng_after_train", "rng_after_eval"))

    protections = [row["gradient_protection"] for row in cp.get("loss_history", []) if row.get("gradient_protection")]
    nprotect = max(len(protections), 1)
    prot = {name: float(np.mean([float(row.get(name, 0)) for row in protections])) for name in
            ("gradient_cosine", "aux_norm_ratio_raw", "aux_norm_ratio_safe")}
    prot.update({f"{name}_count": sum(bool(row.get(name)) for row in protections) for name in ("conflict", "projected", "scaled")})
    res = {"group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
           "checkpoint_sha256": runner.sha_file(runner.task_paths(group, fold)[1]),
           "best_epoch_1based": task["best_epoch_1based"], "epochs_executed": task["epochs_executed"],
           "wall_seconds": float(task["wall_seconds"]), "peak_gpu_bytes": int(task.get("peak_gpu_bytes", 0)),
           "metrics": {key: float(metrics[key]) for key in METRICS},
           "tp": int(metrics["asd_tp"]), "tn": int(metrics["td_tn"]),
           "fp": int(metrics["td_fp"]), "fn": int(metrics["asd_fn"]),
           "train_darg_accuracy": train_darg_acc, "test_darg_accuracy": test_darg_acc,
           "train_aux_accuracy": train_aux_acc, "test_aux_accuracy": test_aux_acc,
           "train_test_aux_gap_pp": (train_aux_acc-test_aux_acc)*100,
           "support_count": int(len(support)), "support_asd": int((state["y"][torch.as_tensor(support, device=state["y"].device)] == 0).sum()),
           "support_td": int((state["y"][torch.as_tensor(support, device=state["y"].device)] == 1).sum()),
           "support_counts_by_class": counts.tolist(),
           "loo_mean_shift_mean": float(mean_shift.mean().cpu()), "loo_mean_shift_p95": float(torch.quantile(mean_shift.flatten(), .95).cpu()),
           "loo_direction_angle_mean_deg": float(angles.mean().cpu()), "loo_direction_angle_p50_deg": float(torch.quantile(angles.flatten(), .5).cpu()),
           "loo_direction_angle_p95_deg": float(torch.quantile(angles.flatten(), .95).cpu()),
           "self_support_mass_asd": 1.0/counts[0], "self_support_mass_td": 1.0/counts[1],
           "gradient_protection": prot, "rng_common_epochs_vs_e78_8": compared,
           "rng_mismatches_vs_e78_8": mismatches,
           "anchor_sha256": module.anchor_sha(),
           "anchor_grad_none": all((not buffer.requires_grad) and (buffer.grad is None) for buffer in module.buffers())}
    if not torch.equal(details["query_in_support"], torch.zeros_like(details["query_in_support"])):
        raise RuntimeError("test query was incorrectly treated as a training support query")
    if latest.get("identity") != task.get("identity") or latest.get("source_sha256") != source_sha:
        raise RuntimeError("latest-state identity failure")
    res["interventions"] = []
    if module.mode == "text" and group == GROUPS[1]:
        original_anchor = module.anchors.detach().clone()
        base_sem = details["semantic_logp"].exp()[:, 0].detach().cpu().numpy()
        base_darg = details["final_logp"].exp()[:, 0].detach().cpu().numpy()
        base_pred = details["final_logp"].argmax(1).detach().cpu().numpy()
        cases = [("no_calibration", "no_calibration", None), ("anchor_only", "anchor_only", None),
                 ("class_swap", "class_swap", None)]
        cache = runner.save_text_cache()
        cases.append(("anonymous_anchor", "full", cache["anonymous_anchors"].to(module.anchors.device)))
        for name, mode, replacement in cases:
            if replacement is not None:
                module.anchors.copy_(replacement)
            try:
                with torch.no_grad():
                    _, changed = runner._forward(state, module, training=False, support=support, query=test_ids, intervention=mode)
            finally:
                module.anchors.copy_(original_anchor)
            sem = changed["semantic_logp"].exp()[:, 0].detach().cpu().numpy()
            darg = changed["final_logp"].exp()[:, 0].detach().cpu().numpy()
            pred = changed["final_logp"].argmax(1).detach().cpu().numpy()
            res["interventions"].append({"group":group,"fold":fold,"intervention":name,
                "mean_abs_semantic_probability_change":float(np.abs(sem-base_sem).mean()),
                "max_semantic_probability_change":float(np.abs(sem-base_sem).max()),
                "semantic_hard_flips":int((changed["semantic_logp"].argmax(1).cpu().numpy()!=details["semantic_logp"].argmax(1).cpu().numpy()).sum()),
                "mean_abs_darg_probability_change":float(np.abs(darg-base_darg).mean()),
                "darg_hard_flips":int((pred!=base_pred).sum()),"darg_unchanged":bool(np.array_equal(pred,base_pred))})
    return res


def _summary(rows, group, exclude_fold2=False):
    chosen = [row for row in rows if row["group"] == group and (not exclude_fold2 or int(row["fold"]) != 2)]
    result = {"group": group, "n_folds": len(chosen)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in chosen])*100
        result[metric] = f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"
    return result


def audit_all():
    frozen = runner.verify_frozen_source()
    source_sha = frozen["source_sha256"]
    preflight = json.loads((RESULTS/"preflight.json").read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("preflight did not pass")
    audits = []; metrics = []; mechanisms = []; interventions = []; errors = []
    for group in GROUPS:
        for fold in range(1, 11):
            try:
                row = audit_checkpoint(group, fold, source_sha)
                audits.append({"group":group,"fold":fold,"status":row["status"],"checkpoint_sha256":row["checkpoint_sha256"],
                               "rng_common_epochs_vs_e78_8":row["rng_common_epochs_vs_e78_8"],"rng_mismatches_vs_e78_8":row["rng_mismatches_vs_e78_8"]})
                metrics.append({"group":group,"fold":fold,"status":row["status"],**row["metrics"],
                                "ba":row["metrics"]["hard_auc"],
                                "tp":row["tp"],"tn":row["tn"],"fp":row["fp"],"fn":row["fn"],
                                "best_epoch_1based":row["best_epoch_1based"],
                                "epochs_executed":row["epochs_executed"],"wall_seconds":row["wall_seconds"],
                                "peak_gpu_bytes":row["peak_gpu_bytes"],"checkpoint_sha256":row["checkpoint_sha256"]})
                mechanisms.append({k:v for k,v in row.items() if k not in ("metrics",)})
                interventions.extend(row.get("interventions", []))
            except Exception as exc:
                errors.append({"group":group,"fold":fold,"error":repr(exc)})
                audits.append({"group":group,"fold":fold,"status":"AUDIT_FAILED","error":repr(exc)})
    if len(metrics) != 30:
        status = "PARTIAL" if metrics else "FAILED"
    else:
        status = "COMPLETE_AUDITED" if not errors else "PARTIAL"

    paired_rows = []
    e78_rows = read_csv(E78_8/"results/fold_metrics.csv")
    base_rows = read_csv(BASE/"results/fold_metrics.csv")
    e78_group_map = {group:runner.CFG["group_specs"][group]["reference_group"] for group in GROUPS}
    comparison_pairs = []
    for group in GROUPS:
        ref = e78_group_map[group]
        l = {int(r["fold"]):r for r in metrics if r["group"]==group}
        r = {int(row["fold"]):row for row in e78_rows if row["group"]==ref}
        if len(l)==10 and len(r)==10:
            comparison_pairs.append((f"{group} - {ref}",l,r))
        base_group="E74.4-G2-Positive-DCR"
        l={int(r["fold"]):r for r in metrics if r["group"]==group}
        r={int(row["fold"]):row for row in base_rows if row["group"] in (base_group,"E74.4-G2-Positive-DCR")}
        if len(l)==10 and len(r)==10:
            comparison_pairs.append((f"{group} - E74.4-G2",l,r))
    for left,right in ((GROUPS[1],GROUPS[0]),(GROUPS[1],GROUPS[2])):
        l={int(r["fold"]):r for r in metrics if r["group"]==left}
        r={int(r["fold"]):r for r in metrics if r["group"]==right}
        if len(l)==10 and len(r)==10:
            comparison_pairs.append((f"{left} - {right}",l,r))
    for name,left,right in comparison_pairs:
        for scope,folds in (("all10",list(range(1,11))),("exclude_fold2",[f for f in range(1,11) if f!=2])):
            for metric in METRICS:
                diffs=np.asarray([(float(left[f][metric])-float(right[f][metric]))*100 for f in folds]); mean=float(diffs.mean()); sd=float(diffs.std(ddof=1))
                half=float(t.ppf(.975,len(diffs)-1)*sd/math.sqrt(len(diffs)))
                paired_rows.append({"comparison":name,"scope":scope,"metric":metric,"n_folds":len(folds),
                                    "mean_difference_pp":mean,"sample_sd_difference_pp":sd,"ci95_low_pp":mean-half,"ci95_high_pp":mean+half,
                                    "wins":int((diffs>1e-12).sum()),"ties":int((np.abs(diffs)<=1e-12).sum()),"losses":int((diffs< -1e-12).sum()),
                                    "fold_differences_pp":json.dumps({str(f):float(float(left[f][metric])-float(right[f][metric]))*100 for f in folds})})
    write_csv(RESULTS/"fold_metrics.csv",metrics)
    write_csv(RESULTS/"paired_comparisons.csv",paired_rows)
    write_csv(RESULTS/"mechanism.csv",mechanisms)
    write_csv(RESULTS/"interventions.csv",interventions)
    write_csv(RESULTS/"checkpoint_audit.csv",audits)
    rng_common=sum(int(row.get("rng_common_epochs_vs_e78_8",0)) for row in audits)
    rng_mismatch=sum(int(row.get("rng_mismatches_vs_e78_8",0)) for row in audits)
    task_count=sum(json.loads(runner.task_paths(g,f)[0].read_text()).get("status")=="COMPLETE" for g in GROUPS for f in range(1,11))
    latest_count=sum(runner.task_paths(g,f)[2].is_file() for g in GROUPS for f in range(1,11))
    all_summary=[_summary(metrics,g) for g in GROUPS]
    nof2_summary=[_summary(metrics,g,True) for g in GROUPS]
    conclusions=[]
    if status!="COMPLETE_AUDITED": conclusions.append("PARTIAL")
    else: conclusions.append("COMPLETE_AUDITED")
    conclusions.append("LOO_EFFECT_SUPPORTED" if _paired_acc(metrics,e78_rows,e78_group_map[GROUPS[1]]) else "LOO_EFFECT_NOT_SUPPORTED")
    conclusions.append("MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED")
    if rng_mismatch: conclusions.append("RNG_MISMATCH_PRESENT")
    status_data={"status":status,"valid_run_revision":runner.CFG["run_revision"],"expected_tasks":30,"trained_tasks":task_count,
                 "independent_checkpoint_audits":sum(row["status"]=="INDEPENDENT_AUDIT_PASS" for row in audits),
                 "latest_states_checked":latest_count,"rng_common_epochs_checked":rng_common,"rng_mismatches":rng_mismatch,
                 "training_workers":json.loads((runner.LOGS/"controller.json").read_text()).get("workers") and len(json.loads((runner.LOGS/"controller.json").read_text())["workers"]),
                 "training_wall_seconds":json.loads((runner.LOGS/"controller.json").read_text()).get("finished",0)-json.loads((runner.LOGS/"controller.json").read_text()).get("started",0),
                 "source_sha256":source_sha,"text_cache_sha256":runner.sha_file(HERE/"runtime/text_embeddings.pt"),
                 "anchor_buffers_unchanged":all(row["status"]=="INDEPENDENT_AUDIT_PASS" for row in audits),
                 "base_off_parity_passed":preflight.get("base_off_parity_passed"),"loo_formula_preflight":preflight.get("loo_formula_preflight"),
                 "audit_errors":errors,"scientific_conclusions":conclusions,"handoff_path":"/model/spromptgl/experiment/handoff.md"}
    runner.atomic_json(RESULTS/"audit_status.json",status_data)
    report(status_data,all_summary,nof2_summary,paired_rows,e78_rows,base_rows,metrics,mechanisms)
    return status_data


def _paired_acc(cur,e78_rows,ref_group):
    left={int(r["fold"]):float(r["acc"]) for r in cur if r["group"]==GROUPS[1]}
    right={int(r["fold"]):float(r["acc"]) for r in e78_rows if r["group"]==ref_group}
    return len(left)==10 and len(right)==10 and float(np.mean([left[f]-right[f] for f in left]))>0


def report(status, summaries, nof2, comparisons, e78_rows, base_rows, metrics, mechanisms):
    lines=["# Experiment 78.9 — LOO-FAPC final report","",f"STATUS: {status['status']}","",
           f"目录：`{HERE}`。正式修复版本：`{status['valid_run_revision']}`。任务 {status['trained_tasks']}/30，best独立审计 {status['independent_checkpoint_audits']}/30，latest {status['latest_states_checked']}/30；worker {status['training_workers']}；RNG共有epoch对照 {status['rng_common_epochs_checked']}，不匹配 {status['rng_mismatches']}。",
           f"训练源SHA：`{status['source_sha256']}`；冻结文本缓存SHA：`{status['text_cache_sha256']}`。/data preferred目录 mkdir 失败(errno 5)，仅使用 /model 输出。","",
           "## 正式六指标","","均值 ± population SD，单位%。ASD=0，SEN-ASD；TD=1，SPE-TD。"]
    lines += ["","|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN|SPE|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        vals=[row["group"]]+[row.get(m,"n/a") for m in METRICS]
        lines.append("|"+"|".join(map(str,vals))+"|")
    lines += ["","## 历史对照","","E74.4-G2逐折结果直接来自服务器历史 `results/fold_metrics.csv`；E78.8对照来自其最终有效修正版逐折指标。E78.8两轮作废运行不纳入比较。","","|Group|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|","|---|---:|---:|---:|---:|---:|---:|"]
    for group in ["E74.4-G2-Positive-DCR"]+[r["group"] for r in summaries]:
        rows=base_rows if group=="E74.4-G2-Positive-DCR" else metrics
        lines.append("|"+group+"|"+"|".join(f"{np.mean([float(r[m]) for r in rows if r['group']==group])*100:.4f}" for m in METRICS)+"|")
    lines += ["","## 配对结果","","逐折差、W/T/L、描述性95%区间和去Fold 2结果见 `paired_comparisons.csv`。描述性区间不代表独立临床样本推断。","","## LOO机制","","逐折合法支持人数、LOO均值偏移、方向角变化、辅助与DARG训练/测试准确率、训练/测试辅助差距及梯度保护统计见 `mechanism.csv`。向量化LOO、显式慢速参考、支持重排和自梯度抵消测试结果见 `results/preflight.json`。","","E78.8的静态锚点与CAM144单侧梯度保护均保留，正式预测仍由原DARG分类器产生。G2−G3为真实医学与匿名文本独立训练比较；固定Checkpoint文本干预只反映辅助支路依赖，不能取代独立训练对照。","","## 科研判断","","判定标签见 `results/audit_status.json`。若医学Prompt没有稳定超过匿名与Visual-only，或模型没有超过E74.4基座，则不支持医学语义独立诊断价值。","","## 限制","","保留历史 val=test、测试ACC选模、CV前RFE/预处理、全患者transductive图、单seed与训练查询参与支持均值；Fold 2重叠行仅从新增分支排除。因此本结果是历史对齐的方法探索，不是无偏临床泛化评估。","",""]
    (RESULTS/"report.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(audit_all(),ensure_ascii=False,indent=2))
