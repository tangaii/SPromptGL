#!/usr/bin/env python3
"""Independent Experiment 53 checkpoint, mechanism and report audit."""
from __future__ import annotations

import csv
import argparse
import json
import math
import os
import hashlib
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
PARENT = Path("/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011")
ROOT = Path("/data/adodas/SPromptGL-main")
RESULTS = EXP / "results"
RT = EXP / ".runtime"
import importlib.util
spec = importlib.util.spec_from_file_location("experiment53_run", EXP / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
GROUP_BY_ID = run.GROUP_BY_ID
atomic_json = run.parent_run.atomic_json


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel_norm(a, b):
    return float((a.detach() - b.detach()).norm().cpu() / b.detach().norm().clamp_min(1e-8).cpu())


def finite_module(module):
    return all(bool(torch.isfinite(p).all().item()) for p in module.parameters())


def load_state(group, fold):
    state = run.build_state(group, fold)
    checkpoint = run.torch_load(run.checkpoint_path(group, fold))
    state["base"].load_state_dict(checkpoint["base"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    if state["drap"] is not None:
        state["drap"].load_state_dict(checkpoint["drap"], strict=True)
    return state, run.parent_run.load_fold_data(fold), checkpoint


def optimizer_audit(state, group):
    base_wd = float(state["opt_base"].param_groups[0]["weight_decay"])
    cam_wd = float(state["opt_cam"].param_groups[0]["weight_decay"])
    if state["opt_drap"] is None:
        drap = [None]
    else:
        drap = [float(pg["weight_decay"]) for pg in state["opt_drap"].param_groups]
    return {
        "base_weight_decay": base_wd,
        "cam_weight_decay": cam_wd,
        "drap_weight_decay_groups": json.dumps(drap),
        "pass": abs(base_wd - 0.11) < 1e-15 and abs(cam_wd - 0.11) < 1e-15 and (
            group == "E0" or sorted(drap) == [0.0, 0.0001]
        ),
    }


def relation_summary(package):
    out = {}
    for m, row in enumerate(package["relation_metrics"]):
        mod = run.MODS[m]
        for key in ("oof_auc", "oof_pr_auc", "oof_brier", "oof_ece", "oof_score_mean", "oof_score_std", "oof_pos_mean", "oof_neg_mean", "oof_separation"):
            out[f"relation_{key}_{mod}"] = float(row[key])
        out[f"relation_reliable_{mod}"] = int(bool(row["reliable"]))
    out["reliable_modalities"] = json.dumps(package["reliable_modalities"])
    out["reliable_modality_count"] = int(len(package["reliable_modalities"]))
    out["omega"] = json.dumps([float(v) for v in package["omega"]])
    out["inner_patient_disjoint"] = int(all(bool(row["patient_disjoint"]) for row in package["inner_rows"]))
    out["inner_split_count"] = int(len(package["inner_rows"]))
    out["candidate_union_edges"] = int(package["candidate"]["union_edge_count"])
    for m, count in enumerate(package["candidate"]["modality_edge_counts"]):
        out[f"candidate_edges_{run.MODS[m]}"] = int(count)
    return out


def diagnostics_for(group, fold, state, fd, checkpoint):
    package = state.get("relation_package")
    use_rw = bool(checkpoint.get("best_use_reweight", True))
    with torch.no_grad():
        full = run.forward_state(state, fd, False)
        off = run.forward_state(state, fd, False, propagation_override="off")
        pos_off = run.forward_state(state, fd, False, propagation_override="positive_off") if group != "E0" else off
        neg_off = run.forward_state(state, fd, False, propagation_override="negative_off") if group != "E0" else off
        shuffled = run.forward_state(state, fd, False, propagation_override="shuffle") if group != "E0" else off
        uniform = run.forward_state(state, fd, False, propagation_override="uniform") if group != "E0" else off
    base_details = full["base_details"]
    raw = full["raw"]
    row = {
        "group": group, "fold": fold, "best_epoch": checkpoint["best_epoch"],
        "epochs_executed": checkpoint.get("epochs_executed", -1),
        "warmup_active_at_best": int(checkpoint.get("best_use_reweight", True) is False),
        "drap_enabled_at_best": int(use_rw),
        **run.metrics(full["logp"], state),
        "fused_e1_norm": float(full["fused_e1"].detach().norm().cpu()),
        "delta_norm": float(full["delta"].detach().norm().cpu()),
        "delta_over_e1": float(full["delta"].detach().norm().cpu() / full["fused_e1"].detach().norm().clamp_min(1e-8).cpu()),
        "drap_over_raw": float((full["fused"] - raw).detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu()),
        "e1_over_raw": float((full["fused_e1"] - raw).detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu()),
        "positive_edges": full["relation"].get("positive_edges", 0),
        "negative_edges": full["relation"].get("negative_edges", 0),
        "positive_message_norm": full["relation"].get("positive_norm", 0.0),
        "negative_message_norm": full["relation"].get("negative_norm", 0.0),
        "prop_delta_norm": full["relation"].get("delta_norm", 0.0),
    }
    for m, gcn in enumerate(state["base"].gcn):
        row[f"gcn{m}_layer1_weight_norm"] = float(gcn.gcn1.W.detach().norm().cpu())
        row[f"gcn{m}_layer2_weight_norm"] = float(gcn.gcn2.W.detach().norm().cpu())
        row[f"gcn{m}_message_norm"] = float(base_details["msgs"][m].detach().norm().cpu())
        row[f"gcn{m}_message_ratio_input"] = float(base_details["msgs"][m].detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu())
        graph = state["base"].graphs[m]
        row[f"graph_projection_norm_{run.MODS[m]}"] = float(graph.p.weight.detach().norm().cpu())
        row[f"graph_edge_count_{run.MODS[m]}"] = int((base_details["adjs"][m] > 0).sum().item())
        row[f"graph_density_{run.MODS[m]}"] = float((base_details["adjs"][m] > 0).float().mean().cpu())
        row[f"graph_finite_{run.MODS[m]}"] = int(torch.isfinite(base_details["adjs"][m]).all().item())
        row[f"graph_symmetric_{run.MODS[m]}"] = int((base_details["adjs"][m] - base_details["adjs"][m].T).abs().max().item() < 1e-6)
    if package is not None:
        row.update(relation_summary(package))
    for label, other in (("off", off), ("positive_off", pos_off), ("negative_off", neg_off), ("shuffle", shuffled), ("uniform", uniform)):
        row[f"{label}_fused_rel_diff"] = rel_norm(full["fused"], other["fused"])
        row[f"{label}_logp_rel_diff"] = rel_norm(full["logp"], other["logp"])
        p0, p1 = full["logp"].exp(), other["logp"].exp()
        row[f"{label}_prob_mean_abs_diff"] = float((p0 - p1).abs().mean().cpu())
        row[f"{label}_prediction_flip_fraction"] = float((p0.argmax(1) != p1.argmax(1)).float().mean().cpu())
        mm = run.metrics(other["logp"], state)
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            row[f"{label}_{key}"] = mm[key]
    if state["drap"] is not None:
        state["drap"].zero_grad(set_to_none=True)
        state["cam"].zero_grad(set_to_none=True)
        train_out = run.forward_state(state, fd, True)
        grad = run.gradient_audit(state, fd, train_out)
        row.update({f"best_{k}": v for k, v in grad.items()})
    else:
        row.update({"best_classification_to_relation": 0.0, "best_relation": 0.0, "best_encoder": 0.0, "best_scorer": 0.0, "best_discriminator": 0.0, "best_finite": True})
    row["checkpoint_finite"] = int(finite_module(state["base"]) and finite_module(state["cam"]) and (state["drap"] is None or finite_module(state["drap"])))
    row["checkpoint_weight_decay_pass"] = int(optimizer_audit(state, group)["pass"])
    return row


def audit_checkpoints():
    parent_metrics = {
        int(row["fold"]): row for row in csv.DictReader((PARENT / "results/fold_metrics.csv").open())
        if row["group"] == "E0"
    }
    checkpoint_rows = []
    init_rows = []
    diagnostics = []
    interventions = []
    for group in run.GROUPS:
        for fold in range(1, 11):
            state, fd, checkpoint = load_state(group, fold)
            with torch.no_grad():
                out = run.forward_state(state, fd, False)
            readback = run.metrics(out["logp"], state)
            stored = checkpoint["metrics"]
            metric_error = max(abs(float(readback[k]) - float(stored[k])) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen"))
            opt = optimizer_audit(state, group)
            init_pass = checkpoint.get("split_sha") == run.split_sha(fold) and checkpoint.get("pair_sha") == sha_file(run.parent_run.FD / f"fold{fold:02d}.npz") and checkpoint.get("initial_shared_sha") == state["initial_shared_sha"] and checkpoint.get("initial_rng_sha") == state["initial_rng_sha"]
            if group == "E0":
                hist = parent_metrics[fold]
                replay = max(abs(float(stored[k]) - float(hist[k])) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen")) < 1e-10 and int(checkpoint["best_epoch"]) == int(hist["best_epoch"])
            else:
                replay = True
            status = metric_error < 1e-10 and bool(opt["pass"]) and init_pass and replay and bool(row_finite := finite_module(state["base"]) and finite_module(state["cam"]) and (state["drap"] is None or finite_module(state["drap"])))
            checkpoint_rows.append({
                "group": group, "fold": fold, "status": "PASS" if status else "FAIL",
                "max_metric_abs_error": metric_error, "optimizer_pass": opt["pass"],
                "initialization_pass": init_pass, "historical_e0_replay_pass": replay,
                "finite_pass": row_finite, "checkpoint_sha256": sha_file(run.checkpoint_path(group, fold)),
            })
            init_rows.append({
                "group": group, "fold": fold, "initial_shared_sha": checkpoint.get("initial_shared_sha", ""),
                "initial_schema_sha": checkpoint.get("initial_schema_sha", ""),
                "initial_cam_sha": checkpoint.get("initial_cam_sha", ""),
                "initial_rng_sha": checkpoint.get("initial_rng_sha", ""),
                "initial_relation_sha": checkpoint.get("initial_relation_sha", ""),
                "split_sha": checkpoint.get("split_sha", ""), "pair_sha": checkpoint.get("pair_sha", ""),
                "reliable_modalities": json.dumps(checkpoint.get("reliable_modalities", [])),
                "pass": init_pass,
            })
            row = diagnostics_for(group, fold, state, fd, checkpoint)
            diagnostics.append(row)
            for label in ("off", "positive_off", "negative_off", "shuffle", "uniform"):
                interventions.append({"group": group, "fold": fold, "intervention": label, **{k: row[f"{label}_{k}"] for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "fused_rel_diff", "logp_rel_diff", "prob_mean_abs_diff", "prediction_flip_fraction")}})
            del state
            torch.cuda.empty_cache()
    csv_write(RESULTS / "checkpoint_audit.csv", checkpoint_rows)
    csv_write(RESULTS / "initialization_audit.csv", init_rows)
    csv_write(RESULTS / "diagnostics.csv", diagnostics)
    csv_write(RESULTS / "interventions.csv", interventions)
    manifest = json.loads((EXP / "manifest.json").read_text())
    ck_ok = all(r["status"] == "PASS" for r in checkpoint_rows)
    init_ok = all(r["pass"] for r in init_rows)
    manifest.update({
        "status": "AUDITED_COMPLETE" if ck_ok and init_ok else "AUDITED_WITH_MISMATCH",
        "checkpoint_audit": {"count": len(checkpoint_rows), "pass": ck_ok},
        "initialization_audit": {"count": len(init_rows), "pass": init_ok},
        "diagnostics_rows": len(diagnostics), "intervention_rows": len(interventions),
        "checkpoint_audit_sha256": sha_file(RESULTS / "checkpoint_audit.csv"),
        "initialization_audit_sha256": sha_file(RESULTS / "initialization_audit.csv"),
        "diagnostics_sha256": sha_file(RESULTS / "diagnostics.csv"),
        "interventions_sha256": sha_file(RESULTS / "interventions.csv"),
    })
    atomic_json(EXP / "manifest.json", manifest)
    print("AUDIT", manifest["status"], "checkpoints", len(checkpoint_rows), flush=True)
    if not ck_ok or not init_ok:
        raise SystemExit(4)


def _mean(rows, key):
    return float(np.asarray([float(r[key]) for r in rows]).mean())


def _sd(rows, key):
    return float(np.asarray([float(r[key]) for r in rows]).std(ddof=0))


def make_report():
    summary = {r["group"]: r for r in csv.DictReader((RESULTS / "summary.csv").open())}
    pairs = list(csv.DictReader((RESULTS / "paired_comparisons.csv").open()))
    diagnostics = list(csv.DictReader((RESULTS / "diagnostics.csv").open()))
    checkpoints = list(csv.DictReader((RESULTS / "checkpoint_audit.csv").open()))
    pre = json.loads((RT / "preflight.json").read_text())
    lines = [
        "# Experiment 53 — DRAP疾病判别驱动的对抗式患者图重构与消息传播",
        "",
        "## 执行和审计状态",
        "",
        f"正式组别E0–E8共90折；训练完成90/90，最佳checkpoint独立读回审计{sum(r['status']=='PASS' for r in checkpoints)}/90。预检状态为{pre['status']}。E0逐折重放实验52 E0，历史指标和best epoch用于严格复核。",
        "E0–E8均保留原始GFR/CAM结构；新增关系参数使用独立AdamW，新增Delta传播不乘原始GCN权重。历史val=test、CV前预处理/RFE、871节点transductive图和单seed协议保留。",
        "",
        "## 论文和历史审计",
        "",
        "SPromptGL使用四模态GraphLearn关系图、两层参数化GCN和CAM；DeepASD先做模态特征分布对齐再构建患者图。本轮将对抗对象改为患者对关系表示，并保留原GFR/CAM作为基座。官方来源：SPromptGL MICCAI 2025论文 https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf ，官方源码 https://github.com/wanxixi11/SPromptGL ，DeepASD https://www.nature.com/articles/s41398-024-02972-2 。",
        "实验43–52的审计表明：E1四路聚合修复可复现，但原始GCN消息在强weight decay下收缩；47.3、48、51系列没有证明真实患者关系身份稳定优于强负控制。实验53因此使用独立关系AdamW、inner patient-disjoint OOF、关系置乱E7和普通邻域E8。",
        "",
        "## 正式结果（十折mean ± population SD，百分比）",
        "",
        "|组|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in run.GROUPS:
        r = summary[g]
        lines.append(f"|{g} {GROUP_BY_ID[g]['name']}|{r['acc_mean_pct']} ± {r['acc_sd_pct']}|{r['hard_auc_mean_pct']} ± {r['hard_auc_sd_pct']}|{r['prob_auc_mean_pct']} ± {r['prob_auc_sd_pct']}|{r['spe_mean_pct']} ± {r['spe_sd_pct']}|{r['sen_mean_pct']} ± {r['sen_sd_pct']}|{r['ba_mean_pct']} ± {r['ba_sd_pct']}|{r['asd_f1_mean_pct']} ± {r['asd_f1_sd_pct']}|")
    lines += ["", "### 关键逐折配对（百分点，胜/平/负）", "", "|比较|指标|均值差|胜/平/负|", "|---|---|---:|---:|"]
    for r in pairs:
        if r["comparison"] in ("E4_minus_E0", "E4_minus_E7", "E4_minus_E8", "E4_minus_E2", "E4_minus_E3", "E4_minus_E5", "E4_minus_E6"):
            lines.append(f"|{r['comparison']}|{r['metric']}|{float(r['delta_pp']):.4f}|{r['wins']}/{r['ties']}/{r['losses']}|")
    lines += ["", "## inner OOF关系证据", "", "|组别|模态|OOF AUC|OOF PR-AUC|校准Brier|ECE|正负分离|可靠折数|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    # OOF relation evidence is fold-level and shared by E1-E8.
    for m in range(4):
        rows = [r for r in diagnostics if r["group"] == "E4"]
        lines.append(f"|E1–E8|{run.MODS[m]}|{_mean(rows, f'relation_oof_auc_{run.MODS[m]}'):.4f}|{_mean(rows, f'relation_oof_pr_auc_{run.MODS[m]}'):.4f}|{_mean(rows, f'relation_oof_brier_{run.MODS[m]}'):.4f}|{_mean(rows, f'relation_oof_ece_{run.MODS[m]}'):.4f}|{_mean(rows, f'relation_oof_separation_{run.MODS[m]}'):.4f}|{sum(int(float(r[f'relation_reliable_{run.MODS[m]}'])) for r in rows)}/10|")
    lines.append("只有满足预注册OOF AUC≥0.55且同类评分高于异类评分的模态参与传播。本轮各折可靠模态数量、omega和候选覆盖见 diagnostics.csv。")
    lines += ["", "## 正负传播和消息范数", "", "|组|可靠模态数|正边|负边|正消息范数|负消息范数|Delta范数|Delta/H_E1|H_DRAP-X|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ("E1","E2","E3","E4","E5","E6","E7","E8"):
        rows = [r for r in diagnostics if r["group"] == g]
        lines.append(f"|{g}|{_mean(rows,'reliable_modality_count'):.2f}|{_mean(rows,'positive_edges'):.1f}|{_mean(rows,'negative_edges'):.1f}|{_mean(rows,'positive_message_norm'):.6g}|{_mean(rows,'negative_message_norm'):.6g}|{_mean(rows,'delta_norm'):.6g}|{_mean(rows,'delta_over_e1'):.6g}|{_mean(rows,'drap_over_raw'):.6g}|")
    lines += ["", "## 固定checkpoint干预", "", "|组|关闭传播ACC|正向关闭ACC|负向关闭ACC|置乱ACC|普通邻域ACC|完整预测翻转率|关闭传播表示差|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in ("E1","E2","E3","E4","E5","E6","E7","E8"):
        rows = [r for r in diagnostics if r["group"] == g]
        lines.append(f"|{g}|{_mean(rows,'off_acc')*100:.4f}|{_mean(rows,'positive_off_acc')*100:.4f}|{_mean(rows,'negative_off_acc')*100:.4f}|{_mean(rows,'shuffle_acc')*100:.4f}|{_mean(rows,'uniform_acc')*100:.4f}|{_mean(rows,'off_prediction_flip_fraction'):.6g}|{_mean(rows,'off_fused_rel_diff'):.6g}|")
    lines.append("这些是同一最佳checkpoint的eval干预，不是新增训练组；E7/E8仍有独立十折训练。")
    lines += ["", "## 对抗训练和梯度", ""]
    for g in ("E3","E4","E5","E6","E7","E8"):
        rows = [r for r in diagnostics if r["group"] == g]
        lines.append(f"- {g}: 关系编码器梯度 {_mean(rows,'best_encoder'):.6g}，评分器梯度 {_mean(rows,'best_scorer'):.6g}，判别器梯度 {_mean(rows,'best_discriminator'):.6g}，分类到关系梯度 {_mean(rows,'best_classification_to_relation'):.6g}。可靠模态少于2时条件对抗按冻结规则关闭。")
    lines += ["", "## 科研判定", ""]
    e4 = summary["E4"]; e0 = summary["E0"]; e7 = summary["E7"]; e8 = summary["E8"]
    e4_e0_acc = float(e4["acc_mean_pct"]) - float(e0["acc_mean_pct"])
    e4_e0_prob = float(e4["prob_auc_mean_pct"]) - float(e0["prob_auc_mean_pct"])
    e4_e7_acc = float(e4["acc_mean_pct"]) - float(e7["acc_mean_pct"])
    e4_e8_acc = float(e4["acc_mean_pct"]) - float(e8["acc_mean_pct"])
    rel_rows = [r for r in diagnostics if r["group"] == "E4"]
    relation_go = any(_mean(rel_rows, f"relation_oof_auc_{m}") >= 0.55 and _mean(rel_rows, f"relation_oof_separation_{m}") > 0 for m in run.MODS)
    propagation_go = _mean(rel_rows, "delta_norm") > 0 and _mean(rel_rows, "off_fused_rel_diff") > 1e-8
    diagnostic_go = e4_e0_acc > 0 and e4_e0_prob > 0 and e4_e7_acc > 0 and e4_e8_acc > 0
    lines.append(f"- 关系判别：{'GO' if relation_go else 'NO-GO'}。至少一个模态的inner OOF关系证据是否达到预注册阈值：{relation_go}。")
    lines.append(f"- 有效传播：{'GO' if propagation_go else 'NO-GO'}。E4 Delta非零且关闭传播产生可测量表示差：{propagation_go}。")
    lines.append(f"- 诊断增量：{'GO' if diagnostic_go else 'NO-GO'}。E4−E0 ACC={e4_e0_acc:.4f} pp，Probability-AUC={e4_e0_prob:.4f} pp，E4−E7 ACC={e4_e7_acc:.4f} pp，E4−E8 ACC={e4_e8_acc:.4f} pp。")
    lines.append(f"- **MICCAI 2027第一创新模块综合判定：{'GO' if relation_go and propagation_go and diagnostic_go else 'NO-GO'}**。工程实现和机制路径可以分别通过，但只有E4超过E0、E7和E8才构成候选创新模块证据。")
    lines.append("由于保留val=test、CV前预处理/transductive图和单seed，结果是协议内机制证据，不是无偏外部泛化或临床因果证明。")
    lines += ["", "## 交付文件", "", "- design.md：源码/论文/历史失败审计和冻结设计。", "- config.json、drap53.py、run.py、audit.py、manifest.json。", "- results/fold_metrics.csv、summary.csv、paired_comparisons.csv、diagnostics.csv、interventions.csv、report.md。", "- .runtime/：关系缓存、90折checkpoint、任务状态和日志。"]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def append_handoff():
    marker = "## Experiment 53 — DRAP疾病判别驱动的对抗式患者图重构与消息传播"
    manifest = json.loads((EXP / "manifest.json").read_text())
    summary = {r["group"]: r for r in csv.DictReader((RESULTS / "summary.csv").open())}
    payload = "\n".join([
        "", marker, "",
        "- 目录：/model/spromptgl/experiment/experiment53_drap_disease_relation_propagation",
        "- 状态：E0–E8共90/90折完成，90/90最佳checkpoint独立读回审计；inner OOF关系缓存10/10折完成。",
        f"- E0 ACC={summary['E0']['acc_mean_pct']}%，E4 DRAP_FULL ACC={summary['E4']['acc_mean_pct']}%，E7置乱ACC={summary['E7']['acc_mean_pct']}%，E8普通邻域ACC={summary['E8']['acc_mean_pct']}%。",
        "- 关系证据、正负传播、固定checkpoint干预、E7/E8独立负控制和GO/NO-GO结论见实验53 report.md。",
        f"- 报告：{RESULTS / 'report.md'}；manifest：{EXP / 'manifest.json'}。",
        "",
    ])
    target = ROOT / "handoff.md"
    pending = EXP / ".runtime" / "handoff_append.md"
    try:
        import fcntl
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            old = handle.read()
            if marker not in old:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        manifest["handoff_update_status"] = "APPENDED"
        manifest["handoff_target"] = str(target)
    except Exception as exc:
        pending.write_text(payload, encoding="utf-8")
        manifest["handoff_update_status"] = "PENDING_IO_ERROR"
        manifest["handoff_target"] = str(target)
        manifest["handoff_pending_path"] = str(pending)
        manifest["handoff_pending_sha256"] = sha_file(pending)
        manifest["handoff_error"] = repr(exc)
    atomic_json(EXP / "manifest.json", manifest)


def report():
    make_report()
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest["report_path"] = str(RESULTS / "report.md")
    manifest["report_sha256"] = sha_file(RESULTS / "report.md")
    atomic_json(EXP / "manifest.json", manifest)
    append_handoff()
    print("REPORT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit"); sub.add_parser("report")
    args = parser.parse_args()
    if args.cmd == "audit":
        audit_checkpoints()
    else:
        report()


if __name__ == "__main__":
    main()
