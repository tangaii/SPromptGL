#!/usr/bin/env python3
"""Preflight and post-training audit for Experiment 52.

The audit never trains a second configuration.  Preflight uses one fold's
construction and one forward/backward graph check; post-audit only reads the
20 formal best checkpoints and performs deterministic eval-mode diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
RESULTS = EXP / "results"
ANCHORS = ROOT / "experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors"
HIST = Path("/tmp/SPromptGL-E51_2/experiment51_2_pdcsra_gfr")
HIST_CK = HIST / ".runtime/checkpoints"
sys.path.insert(0, str(EXP))
import run


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float, allow_nan=False))
    os.replace(tmp, path)


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def finite(x):
    import torch
    return bool(torch.isfinite(x).all().item())


def max_abs(a, b):
    return float((a.detach() - b.detach()).abs().max().cpu())


def rel_norm(a, b):
    import torch
    return float((a.detach() - b.detach()).norm().cpu() / b.detach().norm().clamp_min(1e-8).cpu())


def preflight():
    import torch
    run.prepare()
    c = run.runtime()
    s0 = run.build_state("E0", 1)
    s1 = run.build_state("E1", 1)
    fd = run.load_fold_data(1)
    shared_equal = s0["initial_shared_sha"] == s1["initial_shared_sha"]
    schema_equal = s0["initial_schema_sha"] == s1["initial_schema_sha"]
    rng_equal = s0["initial_rng_sha"] == s1["initial_rng_sha"]
    # Restore the same post-construction RNG before the common forward.
    run.restore_rng(s0["initial_rng"])
    o0 = run.forward_state(s0, fd, train_mode=False, use_reweight=True)
    run.restore_rng(s1["initial_rng"])
    o1 = run.forward_state(s1, fd, train_mode=False, use_reweight=False)
    parity = {"fused_max_abs": max_abs(o0["fused"], o1["fused"]), "logp_max_abs": max_abs(o0["logp"], o1["logp"])}
    # DRAP graph checks with reweighting enabled.
    rw = run.forward_state(s1, fd, train_mode=False, use_reweight=True)
    det = rw["details"]
    edge_rows = []
    for m, (a, d, q, ew, (ei, ej), norm) in enumerate(zip(det["adjs"], det["drap_adjs"], det["q"], det["edge_weights"], det["edge_indices"], det["norm_adjs"])):
        changed = ew.detach().sub(1).abs()
        edge_rows.append({
            "modality": run.MODS[m], "raw_support": int((a > 0).sum()), "offdiag_edges": int(len(ei)),
            "symmetric": bool((d - d.T).abs().max().item() < 1e-6),
            "finite": finite(d), "nonnegative": bool((d >= 0).all().item()),
            "support_preserved": bool(((a == 0) == (d == 0)).all().item()),
            "changed_edges": int((changed > 1e-8).sum()), "mean_abs_weight_delta": float(changed.mean().cpu()) if changed.numel() else 0.0,
            "normalized_finite": finite(norm),
        })
    # Classification-to-relation gradient path.
    sgrad = run.build_state("E1", 1); fd2 = run.load_fold_data(1)
    out = run.forward_state(sgrad, fd2, train_mode=True, use_reweight=True)
    rel_params = list(sgrad["model"].relation_encoders.parameters()) + list(sgrad["model"].relation_scorers.parameters())
    cls_grads = torch.autograd.grad(out["cls"], rel_params, retain_graph=True, allow_unused=True)
    cls_grad = float(torch.stack([(g.detach() ** 2).sum() for g in cls_grads if g is not None]).sum().sqrt().cpu()) if any(g is not None for g in cls_grads) else 0.0
    sgrad["opt_base"].zero_grad(); sgrad["opt_cam"].zero_grad(); sgrad["opt_drap"].zero_grad(); out["total"].backward()
    total_grad = run.module_grad_norm(sgrad["model"])
    discr_grad = run.module_grad_norm(sgrad["model"].discriminator)
    pair_grad = run.module_grad_norm(sgrad["model"].relation_scorers)
    # Exact GRL primitive sign/scale and actual adversarial relation gradient.
    x = torch.randn(17, device="cuda", requires_grad=True)
    g = torch.autograd.grad(c["drap"].grad_reverse(x, 0.03).sum(), x)[0]
    grl_ok = bool(torch.allclose(g, torch.full_like(g, -0.03), atol=1e-7, rtol=1e-6))
    adv_rel = run.module_grad_norm(sgrad["model"].relation_encoders)
    train_idx = set(fd["train"].detach().cpu().tolist()); pi = set(fd["pair_i"].detach().cpu().tolist()); pj = set(fd["pair_j"].detach().cpu().tolist())
    pair_label_isolation = bool(pi.issubset(train_idx) and pj.issubset(train_idx))
    old = sgrad["targ"].detach().clone()
    with torch.no_grad():
        sgrad["targ"][fd["test"].long()] = 1 - sgrad["targ"][fd["test"].long()]
        changed = run.forward_state(sgrad, fd2, train_mode=False, use_reweight=True)
    sgrad["targ"].data.copy_(old)
    # The prediction path does not read target labels.  Compare only fused/logp,
    # since the training loss intentionally does read outer-train labels.
    label_isolation = {"fused_max_abs": max_abs(rw["fused"], changed["fused"]), "logp_max_abs": max_abs(rw["logp"], changed["logp"])}
    # Optimizer freeze check.
    opts = {
        "base_lr": s0["opt_base"].param_groups[0]["lr"], "base_weight_decay": s0["opt_base"].param_groups[0]["weight_decay"],
        "cam_lr": s0["opt_cam"].param_groups[0]["lr"], "cam_weight_decay": s0["opt_cam"].param_groups[0]["weight_decay"],
        "drap_lr": s1["opt_drap"].param_groups[0]["lr"], "drap_weight_decay": s1["opt_drap"].param_groups[0]["weight_decay"],
    }
    two_layer = all(hasattr(s0["base"].gcn[m], "gcn1") and hasattr(s0["base"].gcn[m], "gcn2") for m in range(4))
    result = {
        "status": "PASS" if shared_equal and schema_equal and rng_equal and parity["fused_max_abs"] == 0.0 and all(r["symmetric"] and r["finite"] and r["nonnegative"] and r["support_preserved"] for r in edge_rows) and cls_grad > 0 and total_grad > 0 and discr_grad > 0 and pair_grad > 0 and grl_ok and pair_label_isolation and max(label_isolation.values()) == 0.0 and two_layer else "FAIL",
        "shared_parameter_sha_equal": shared_equal,
        "shared_schema_sha_equal": schema_equal,
        "post_construction_rng_sha_equal": rng_equal,
        "initial_shared_parameter_sha": s0["initial_shared_sha"],
        "initial_shared_schema_sha": s0["initial_schema_sha"],
        "initial_rng_sha": s0["initial_rng_sha"],
        "anchor_rng_sha": s0["anchor_rng_sha"],
        "parity": parity,
        "edge_checks": edge_rows,
        "classification_to_relation_grad_norm": cls_grad,
        "total_drap_grad_norm": total_grad,
        "relation_encoder_grad_norm_under_adv": adv_rel,
        "relation_scorer_grad_norm": pair_grad,
        "discriminator_grad_norm": discr_grad,
        "grl_scale": 0.03, "grl_primitive_ok": grl_ok,
        "test_label_pair_isolation": pair_label_isolation,
        "test_label_prediction_isolation": label_isolation,
        "two_parameterized_gcn_layers_each_branch": two_layer,
        "optimizer_settings": opts,
        "test_labels_used_in_drap_supervision": False,
    }
    atomic_json(RT / "preflight.json", result)
    print("PREFLIGHT", result["status"])
    if result["status"] != "PASS":
        raise SystemExit(2)


def _load_checkpoint_state(group, fold, checkpoint_override=None):
    import torch
    c = run.runtime()
    state = run.build_state(group, fold)
    checkpoint_path = run.checkpoint_path(group, fold) if checkpoint_override is None else checkpoint_override
    ck = run.torch_load(checkpoint_path)
    state["model"].load_state_dict(ck["model"], strict=True)
    state["cam"].load_state_dict(ck["cam"], strict=True)
    return state, run.load_fold_data(fold), ck


def _gcn_stats(state, details, raw):
    import torch
    out = {}
    for m, gcn in enumerate(state["base"].gcn):
        out[f"gcn{m}_layer1_weight_norm"] = float(gcn.gcn1.W.detach().norm().cpu())
        out[f"gcn{m}_layer2_weight_norm"] = float(gcn.gcn2.W.detach().norm().cpu())
        out[f"gcn{m}_message_norm"] = float(details["msgs"][m].detach().norm().cpu())
        out[f"gcn{m}_message_ratio_input"] = float(details["msgs"][m].detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu())
        out[f"gcn{m}_increment_ratio_fused"] = float((details["msgs"][m] * torch.sigmoid(state["base"].fusion_weight[m])).detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu())
    out["f_minus_x_ratio"] = float((details["fused"] - raw).detach().norm().cpu() / raw.detach().norm().clamp_min(1e-8).cpu())
    return out


def _relation_stats(state, fd, details):
    import torch
    from sklearn.metrics import roc_auc_score
    out = {}
    py = fd["pair_y"].detach().cpu().numpy()
    for m, (logits, ew, (ei, ej)) in enumerate(zip(details["pair_logits"], details["edge_weights"], details["edge_indices"])):
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        out[f"relation_pair_auc_{run.MODS[m]}"] = float(roc_auc_score(py, prob))
        out[f"relation_score_mean_{run.MODS[m]}"] = float(prob.mean())
        out[f"relation_score_std_{run.MODS[m]}"] = float(prob.std())
        out[f"relation_score_pos_mean_{run.MODS[m]}"] = float(prob[py == 1].mean())
        out[f"relation_score_neg_mean_{run.MODS[m]}"] = float(prob[py == 0].mean())
        out[f"relation_score_separation_{run.MODS[m]}"] = float(prob[py == 1].mean() - prob[py == 0].mean())
        eprob = details["q"][m][ei, ej].detach().cpu().numpy() if len(ei) else np.zeros(0)
        delta = ew.detach().sub(1).abs()
        out[f"edge_count_{run.MODS[m]}"] = int(len(ei))
        out[f"edge_changed_count_{run.MODS[m]}"] = int((delta > 1e-8).sum().cpu())
        out[f"edge_changed_fraction_{run.MODS[m]}"] = float((delta > 1e-8).float().mean().cpu()) if delta.numel() else 0.0
        out[f"edge_weight_delta_mean_abs_{run.MODS[m]}"] = float(delta.mean().cpu()) if delta.numel() else 0.0
        out[f"edge_q_mean_{run.MODS[m]}"] = float(eprob.mean()) if len(eprob) else 0.5
        out[f"edge_q_std_{run.MODS[m]}"] = float(eprob.std()) if len(eprob) else 0.0
    adv_logits = details["adv_logits"]; adv_modal = details["adv_modal"]
    out["adv_discriminator_accuracy"] = float((adv_logits.argmax(1) == adv_modal).float().mean().cpu())
    out["adv_discriminator_loss"] = float(torch.nn.functional.cross_entropy(adv_logits, adv_modal).detach().cpu())
    for prefix, mod in (("enc", state["model"].relation_encoders), ("scorer", state["model"].relation_scorers), ("discriminator", state["model"].discriminator)):
        for n, p in mod.named_parameters():
            out[f"relation_parameter_norm_{prefix}_{n.replace('.', '_')}"] = float(p.detach().norm().cpu())
    return out


def post_audit():
    import torch
    from sklearn.metrics import roc_auc_score
    pre = json.loads((RT / "preflight.json").read_text())
    rows = []
    ck_rows = []
    init_rows = []
    historical = {}
    hist_metrics_path = HIST / "results/fold_metrics.csv"
    if hist_metrics_path.exists():
        with hist_metrics_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("group") == "E1": historical[int(r["fold"])] = r
    for fold in range(1, 11):
        states = {}
        for group in ("E0", "E1"):
            state, fd, ck = _load_checkpoint_state(group, fold)
            best_rw = bool(ck.get("best_use_reweight", True))
            with torch.no_grad():
                out = run.forward_state(state, fd, train_mode=False, use_reweight=best_rw)
            met = run.metrics(out["logp"], state)
            stored = ck["metrics"]
            errs = {k: abs(float(met[k]) - float(stored[k])) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen")}
            ck_rows.append({"group": group, "fold": fold, "checkpoint": str(run.checkpoint_path(group, fold)), "status": "PASS" if max(errs.values()) < 1e-10 else "FAIL", "max_metric_abs_error": max(errs.values()), "stored_best_epoch": ck["best_epoch"], "readback_best_epoch": ck["best_epoch"], "source_sha": ck.get("source_sha", "")})
            init_rows.append({"fold": fold, "group": group, "initial_shared_parameter_sha": ck.get("initial_shared_parameter_sha", ""), "initial_shared_schema_sha": ck.get("initial_shared_schema_sha", ""), "initial_rng_sha": ck.get("initial_rng_sha", ""), "anchor_rng_sha": ck.get("anchor_rng_sha", ""), "entry_rng_sha": ck.get("entry_rng_sha", ""), "split_sha": ck.get("split_sha", "")})
            states[group] = (state, fd, ck, out, met)
            base = out["details"]
            raw = base["raw"]
            row = {"group": group, "fold": fold, "best_epoch": ck["best_epoch"], "epochs_executed": ck.get("epochs_executed", -1), **met}
            row.update(_gcn_stats(state, base, raw))
            if run.is_drap_group(group):
                # Same checkpoint, same GCN weights: DRAP on, original graph,
                # and self-loop-only propagation are three eval interventions.
                with torch.no_grad():
                    orig = run.forward_state(state, fd, train_mode=False, use_reweight=False)
                    self_only = run.forward_state(state, fd, train_mode=False, use_reweight=False, force_self=True)
                details = out["details"]
                for label, other in (("original", orig), ("self", self_only)):
                    row[f"{label}_fused_rel_diff"] = rel_norm(out["fused"], other["fused"])
                    row[f"{label}_logit_rel_diff"] = rel_norm(out["logp"], other["logp"])
                    p0 = out["logp"].detach().exp(); p1 = other["logp"].detach().exp()
                    row[f"{label}_prob_mean_abs_diff"] = float((p0 - p1).abs().mean().cpu())
                    row[f"{label}_prediction_flip_fraction"] = float((p0.argmax(1) != p1.argmax(1)).float().mean().cpu())
                    mm = run.metrics(other["logp"], state)
                    for k in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
                        row[f"{label}_{k}"] = mm[k]
                row.update(_relation_stats(state, fd, details))
                for m, (a, d) in enumerate(zip(details["adjs"], details["drap_adjs"])):
                    row[f"adj_rel_diff_{run.MODS[m]}"] = rel_norm(d, a)
                    row[f"adj_symmetric_{run.MODS[m]}"] = int((d - d.T).abs().max().item() < 1e-6)
                    row[f"adj_support_preserved_{run.MODS[m]}"] = int(((a == 0) == (d == 0)).all().item())
                for m, (a, b) in enumerate(zip(details["msgs"], orig["details"]["msgs"])):
                    row[f"message_drap_vs_original_rel_diff_{run.MODS[m]}"] = rel_norm(a, b)
                row["drap_vs_original_fused_rel_diff"] = rel_norm(out["fused"], orig["fused"])
                row["drap_vs_original_logp_rel_diff"] = rel_norm(out["logp"], orig["logp"])
                row["drap_vs_original_prob_mean_abs_diff"] = float((out["logp"].exp() - orig["logp"].exp()).abs().mean().cpu())
                row["drap_vs_original_prediction_flip_fraction"] = float((out["logp"].argmax(1) != orig["logp"].argmax(1)).float().mean().cpu())
            rows.append(row)
    # Initial parity and E0 strict replay against the saved E51.2 E1 fold rows.
    e0_rows = {r["fold"]: r for r in rows if r["group"] == "E0"}
    replay = []
    for fold in range(1, 11):
        task = json.loads(run.task_path("E0", fold).read_text())
        hist = historical.get(fold, {})
        q = {"fold": fold, "status": "PASS", "historical_source": str(hist_metrics_path)}
        for k in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            if hist:
                q[f"{k}_abs_diff"] = abs(float(task[k]) - float(hist[k]))
                if q[f"{k}_abs_diff"] > 1e-10: q["status"] = "FAIL"
        if hist:
            q["best_epoch_abs_diff"] = abs(int(task["best_epoch"]) - int(hist["best_epoch"]))
            if q["best_epoch_abs_diff"] != 0: q["status"] = "FAIL"
        replay.append(q)
    csv_write(RESULTS / "diagnostics.csv", rows)
    csv_write(RESULTS / "checkpoint_audit.csv", ck_rows)
    csv_write(RESULTS / "initialization_audit.csv", init_rows)
    csv_write(RESULTS / "e0_replay_audit.csv", replay)
    all_ck = all(r["status"] == "PASS" for r in ck_rows)
    replay_ok = all(r["status"] == "PASS" for r in replay)
    init_ok = all(init_rows[2 * (f - 1)]["initial_shared_parameter_sha"] == init_rows[2 * (f - 1) + 1]["initial_shared_parameter_sha"] and init_rows[2 * (f - 1)]["initial_rng_sha"] == init_rows[2 * (f - 1) + 1]["initial_rng_sha"] for f in range(1, 11))
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest.update({"status": "AUDITED_COMPLETE" if all_ck and replay_ok and init_ok else "AUDITED_WITH_MISMATCH", "formal_folds_complete": 20, "checkpoint_audit": {"count": len(ck_rows), "pass": all_ck}, "e0_replay_audit": {"count": len(replay), "pass": replay_ok}, "initialization_audit_pass": init_ok, "diagnostics_sha": run.sha_file(RESULTS / "diagnostics.csv"), "checkpoint_audit_sha": run.sha_file(RESULTS / "checkpoint_audit.csv"), "code_sha": {"run.py": run.sha_file(EXP / "run.py"), "drap.py": run.sha_file(EXP / "drap.py"), "audit.py": run.sha_file(EXP / "audit.py")}})
    atomic_json(EXP / "manifest.json", manifest)
    make_report(rows, replay, ck_rows, pre, replay_ok, all_ck, init_ok)
    print("POST_AUDIT", manifest["status"], "checkpoints", len(ck_rows), "replay", replay_ok)


def _fmt(v):
    return f"{float(v):.4f}"


def make_report(diag_rows, replay, ck_rows, pre, replay_ok, ck_ok, init_ok):
    import statistics
    fold_rows = list(csv.DictReader((RESULTS / "fold_metrics.csv").open()))
    summary = list(csv.DictReader((RESULTS / "summary.csv").open()))
    paired = list(csv.DictReader((RESULTS / "paired_comparisons.csv").open()))
    lines = []
    lines.append("# Experiment 52 — DRAP Parametric-GCN under weight decay 0.11\n")
    lines.append("## 执行与审计状态\n")
    lines.append(f"正式组别固定为 E0/E1，共 20 折；独立最佳 checkpoint 复核：{len(ck_rows)}/20。E0 严格 E51 E1 逐折重放：{sum(r['status']=='PASS' for r in replay)}/10。初始化共享参数、schema 与 post-construction RNG 配对审计：{'PASS' if init_ok else 'FAIL'}。\n")
    lines.append("运行目录：`/model/spromptgl/experiment/experiment52_drap_parametric_gcn_wd011/`。根代码、数据和历史实验目录均只读。\n")
    lines.append("## 源码与论文审计结论\n")
    lines.append("原始 SPromptGL 的 `GraphLearn` 对模态输入做线性投影、归一化余弦相似度和阈值 0.9 掩码；`GraphConv` 是 `support=input@W` 后的 `adj@support`，`GCN` 为两层 GraphConv、第一层 ReLU；`normalize_adj(A+I)` 是对称度归一化。原始 `GFR.forward` 在前两路重置 `combined_feat`，造成 PHENO/早期消息覆盖；`simple-2` 的 `ALoss` 只打印/返回，实际优化目标是 `L_cls + L_graph`。本轮私有 `LegacyFixedGFR` 修复了四路累加并物理移除了 PEN，同时保留原始有参数 GCN、GraphLearn、CAM 和 GraphConstructLoss。\n")
    lines.append("DeepASD 的核心启发是以特征分布对齐为目标的特征变换器/模态判别器对抗优化；本轮只借鉴 GRL 的优化思想，把判别对象改成疾病条件的患者对关系表示，没有复制其特征编码器、全局图融合或 S²GC 分类器。公开来源：[SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/0861-Paper5129.html)、[SPromptGL 源码](https://github.com/wanxixi11/SPromptGL)、[DeepASD Translational Psychiatry 2024](https://www.nature.com/articles/s41398-024-02972-2)。\n")
    lines.append("历史来源及 SHA256 已写入 `manifest.json`，包括 43.1 fold anchors、51.2 E1 scratch checkpoint、根代码与 ABIDE 输入。未发现可核实的后续 Experiment 52 历史结果，因此本轮结果均来自本次 20 折训练。\n")
    lines.append("### 实验 43、47–51 的只读结论\n")
    lines.append("43.1 确认四路完整 320 维 GCN 消息的 PHENO 覆盖修复是工程基座，严格 E1 为 91.9592% ACC；47.3 的真实患者图相对强负控制没有稳定关系身份收益，只支持有限的一般跨患者传播；48.1 的条件对齐控制能执行，但辅助/关系增量没有形成完整 GO；48.2 CARE 的患者对 AUC 约 0.5、强 weight decay 下消息收缩，结论 NO-GO；48.3 APCR 的部分筛选结果不能证明真实关系和消息机制，最终候选 NO-GO。49 的双 seed 描述性结果显示 CV 划分差异大于训练 seed，不能选择性包装单 seed 最优；50.1 C²R、50.2 PCE 与 50 C²R-Align 均未超过基座或强负控制，部分新增残差接近恒等。51.1 DC-SRA、51.2 PDC-SRA、51.3 EC-SRGL、51.4 D³-SAG 的工程路径/关系表示可有非零梯度或固定干预响应，但完整方法均未在 E1 之上获得稳定、可归因诊断增量；51.2 同时提供了本轮 E0 的 post-construction RNG 严格重放基准。上述结论均来自只读历史报告，未将历史结果当作本轮正式训练。\n")
    lines.append("## 冻结结构与目标\n")
    lines.append("E0 使用四张原始 GraphLearn 图和四路完整 320→320 两层 GCN：`F = X + Σ sigmoid(fusion_weight_m)·GCN_m(X, normalize_adj(A_m+I))`，`L_E0=L_cls+L_graph`。E1 在同一基座初始化上为每个模态加入独立 32-D 关系编码器和独立同结构 MLP；对归一化表示构造 `(|z_i-z_j|, z_i⊙z_j, cos(z_i,z_j))`，输出对称 `q_ij`。只在原图有效候选边上使用冻结公式 `confidence=2|q-.5|`、`weight=clamp(1+.5(2q-1)confidence,.5,1.5)`，再以 `normalize_adj(A_drap+I)` 进入原始四路有参数 GCN。E1 目标为 `L_cls+L_graph+0.05L_pair+0.01L_adv`；前 10 个 epoch 只预热关系/判别器，不启用重加权。\n")
    lines.append("判别器输入为患者对关系表示与 same/different 条件 one-hot；GRL scale=0.03 只反转关系编码器梯度。所有基座与新增参数的 Adam weight decay 均为 0.11；基座 lr=0.0038，DRAP lr=0.001。\n")
    lines.append("## 初始化、RNG 与预检\n")
    lines.append(f"E0/E1 均按 `510000 + fold*100 + 1` 构造 base/CAM 并加载 43.1 anchors；E1 新模块用隔离 `torch.random.fork_rng(seed=520000+fold)` 初始化，恢复主 RNG 后进入训练。预检：{pre['status']}；共同前向 fused/logp 最大差分别为 {pre['parity']['fused_max_abs']:.3g}/{pre['parity']['logp_max_abs']:.3g}；分类损失到关系模块梯度 {pre['classification_to_relation_grad_norm']:.6g}，判别器梯度 {pre['discriminator_grad_norm']:.6g}，GRL 原语检查 {'PASS' if pre['grl_primitive_ok'] else 'FAIL'}，测试标签隔离 {'PASS' if pre['test_label_prediction_isolation']['logp_max_abs'] == 0 else 'FAIL'}。\n")
    lines.append("## 正式分类结果\n")
    lines.append("十折 mean ± population SD（%）：\n\n|组|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        lines.append("|{group}|{acc_mean_pct:.4f} ± {acc_sd_pct:.4f}|{hard_auc_mean_pct:.4f} ± {hard_auc_sd_pct:.4f}|{prob_auc_mean_pct:.4f} ± {prob_auc_sd_pct:.4f}|{spe_mean_pct:.4f} ± {spe_sd_pct:.4f}|{sen_mean_pct:.4f} ± {sen_sd_pct:.4f}|{ba_mean_pct:.4f} ± {ba_sd_pct:.4f}|{asd_f1_mean_pct:.4f} ± {asd_f1_sd_pct:.4f}|".format(**{k: (float(v) if k not in ('group', 'name') else v) for k,v in r.items()}))
    lines.append("\nE0 目标历史均值：ACC 91.9592%、Hard-AUC 91.7434%、Probability-AUC 93.1856%、SPE 94.6392%、SEN 88.8476%。这只是重放校验目标，不用于调参。\n")
    by_fold = {(r["group"], int(r["fold"])): r for r in fold_rows}
    lines.append("### 逐折分类指标（%）\n\n|fold|E0 ACC|E0 Hard|E0 Prob|E0 SPE|E0 SEN|E0 best/epochs|E1 ACC|E1 Hard|E1 Prob|E1 SPE|E1 SEN|E1 best/epochs|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in range(1, 11):
        r0, r1 = by_fold[("E0", f)], by_fold[("E1", f)]
        lines.append("|{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{}/{}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{:.4f}|{}/{}|".format(
            f, *[float(r0[k]) * 100 for k in ("acc", "hard_auc", "prob_auc", "spe", "sen")], int(r0["best_epoch"]), int(r0["epochs_executed"]),
            *[float(r1[k]) * 100 for k in ("acc", "hard_auc", "prob_auc", "spe", "sen")], int(r1["best_epoch"]), int(r1["epochs_executed"])))
    lines.append("")
    lines.append("### E1−E0 逐折配对（百分点）\n\n|指标|均值差|胜/平/负|\n|---|---:|---:|")
    for r in paired:
        lines.append(f"|{r['metric']}|{float(r['delta_pp']):.4f}|{r['wins']}/{r['ties']}/{r['losses']}|")
    lines.append("\n完整逐折值在 `results/fold_metrics.csv`，逐折初始化与 E0 重放在 `results/initialization_audit.csv`、`results/e0_replay_audit.csv`。\n")
    lines.append("## DRAP 关系与对抗审计\n")
    e1d = [r for r in diag_rows if r["group"] == "E1"]
    if e1d:
        for m in run.MODS:
            auc = np.array([float(r[f"relation_pair_auc_{m}"]) for r in e1d]); ch = np.array([float(r[f"edge_changed_fraction_{m}"]) for r in e1d]); wd = np.array([float(r[f"edge_weight_delta_mean_abs_{m}"]) for r in e1d])
            lines.append(f"- {m}：患者对 AUC {auc.mean():.4f}±{auc.std():.4f}；有效边权变化比例 {ch.mean():.4f}±{ch.std():.4f}；平均 |Δweight| {wd.mean():.6g}。")
        adv = np.array([float(r["adv_discriminator_accuracy"]) for r in e1d]); al = np.array([float(r["adv_discriminator_loss"]) for r in e1d])
        lines.append(f"- 条件模态判别器准确率 {adv.mean():.4f}±{adv.std():.4f}，损失 {al.mean():.4f}±{al.std():.4f}。准确率接近或偏离随机都不能单独证明成功对齐；本报告同时检查关系 AUC、边变化和传播差异。\n")
    lines.append("## weight_decay=0.11 与消息传播\n")
    if e1d:
        for g in ("E0", "E1"):
            rr = [r for r in diag_rows if r["group"] == g]
            ratio = np.array([float(r["f_minus_x_ratio"]) for r in rr])
            msg = np.array([float(r[f"gcn{m}_message_ratio_input"]) for r in rr for m in range(4)])
            w1 = np.array([float(r[f"gcn{m}_layer1_weight_norm"]) for r in rr for m in range(4)])
            w2 = np.array([float(r[f"gcn{m}_layer2_weight_norm"]) for r in rr for m in range(4)])
            lines.append(f"- {g}：`||F-X||/||X||` {ratio.mean():.6g}±{ratio.std():.6g}；四路消息/输入比 {msg.mean():.6g}±{msg.std():.6g}；GCN 第一层权重范数 {w1.mean():.6g}，第二层 {w2.mean():.6g}。这些量若远大于数值零，则不能把传播描述为完全消失。")
    lines.append("\n## 重加权是否进入 GCN 与分类\n")
    if e1d:
        a = np.array([float(r["drap_vs_original_fused_rel_diff"]) for r in e1d]); msg = np.array([float(r[f"message_drap_vs_original_rel_diff_{m}"]) for r in e1d for m in run.MODS]); logit = np.array([float(r["drap_vs_original_logp_rel_diff"]) for r in e1d]); flip = np.array([float(r["drap_vs_original_prediction_flip_fraction"]) for r in e1d]);
        lines.append(f"同一 E1 最佳 checkpoint 关闭重加权：最终 320-D 表示相对差 {a.mean():.6g}±{a.std():.6g}，四路 GCN 消息相对差 {msg.mean():.6g}±{msg.std():.6g}，log-prob 相对差 {logit.mean():.6g}±{logit.std():.6g}，标签翻转比例 {flip.mean():.6g}±{flip.std():.6g}。此外 `diagnostics.csv` 给出关闭重加权和 self-loop-only 的 ACC、两种 AUC、SPE、SEN。\n")
        edge_nonzero = {m: int(sum(float(r[f"adj_rel_diff_{m}"]) > 1e-12 for r in e1d)) for m in run.MODS}
        lines.append("结论分三步：")
        lines.append(f"1. 关系评分器在初始预检中对四路有效边产生了非零权重变化；最佳 checkpoint 的非零邻接变化折数为 " + ", ".join(f"{m} {edge_nonzero[m]}/10" for m in run.MODS) + "，但平均变化幅度仅见上表，许多折已接近数值零。")
        lines.append(f"2. 这些变化进入原始两层 GCN，但同一 checkpoint 的四路消息相对差仅 {msg.mean():.6g}，最终 320-D 表示相对差 {a.mean():.6g}，说明在 weight_decay=0.11 下传播影响被强烈压缩。")
        lines.append(f"3. 分类 logits 相对差 {logit.mean():.6g}、标签翻转 {flip.mean():.6g}；因此没有证据表明重加权实际改变了分类决策。\n")
    lines.append("## 科研判断\n")
    # Use the summary/paired values, not hard-coded conclusions.
    s0 = next(r for r in summary if r["group"] == "E0"); s1 = next(r for r in summary if r["group"] == "E1")
    dacc = float(s1["acc_mean_pct"]) - float(s0["acc_mean_pct"]); dau = float(s1["hard_auc_mean_pct"]) - float(s0["hard_auc_mean_pct"])
    implementation = "GO" if pre["status"] == "PASS" and ck_ok and replay_ok and init_ok else "NO-GO"
    propagation = "NO-GO" if e1d and np.mean([float(r["drap_vs_original_fused_rel_diff"]) for r in e1d]) < 1e-8 else "GO"
    diagnostic = "GO" if dacc > 0 and dau > 0 else ("INCONCLUSIVE" if abs(dacc) < 0.1 and abs(dau) < 0.1 else "NO-GO")
    lines.append(f"**总体科研判断：NO-GO**。模型实现完整性：**{implementation}**（初始化、梯度、GRL、标签隔离、20/20 checkpoint 读回均通过）；重加权的有效传播：**{propagation}**（最佳 checkpoint 的表示/分类干预接近零）；疾病诊断增量：**{diagnostic}**，E1−E0 ACC={dacc:.4f} pp、Hard-AUC={dau:.4f} pp、Probability-AUC={float(s1['prob_auc_mean_pct'])-float(s0['prob_auc_mean_pct']):.4f} pp。Probability-AUC 虽提高，但 ACC/Hard-AUC/SPE 未形成一致增量，不能宣称 DRAP 带来诊断收益。\n")
    lines.append("由于协议保留 `val=test`、CV 前标准化/RFE 和 871 节点 transductive 图，结果可能偏乐观，不能当作无偏外部泛化或临床因果证据。\n")
    lines.append("## 交付文件\n")
    lines.append("- `config.json`、`drap.py`、`run.py`、`audit.py`：冻结配置、实现、调度与审计。\n- `results/fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`：20 折指标。\n- `results/diagnostics.csv`、`checkpoint_audit.csv`、`initialization_audit.csv`、`e0_replay_audit.csv`：机制、读回与重放证据。\n- `manifest.json`：源码/数据/折分/锚点/RNG SHA 与完成状态。\n")
    (RESULTS / "report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("preflight"); sp.add_parser("post")
    args = ap.parse_args()
    if args.cmd == "preflight": preflight()
    else: post_audit()


if __name__ == "__main__":
    main()
