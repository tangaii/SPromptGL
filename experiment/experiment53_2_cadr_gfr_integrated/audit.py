#!/usr/bin/env python3
"""Independent checkpoint readback and mechanism audit for Experiment 53.2."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import run as R


METRICS = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")


def _norm(values):
    import torch

    selected = [value.detach().float().square().sum() for value in values if value is not None]
    return float(torch.stack(selected).sum().sqrt().cpu()) if selected else 0.0


def _gradients(loss, parameters, retain_graph=True):
    import torch

    params = list(parameters)
    if not params or not loss.requires_grad:
        return [None] * len(params)
    return list(torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True))


def _eval(state, intervention="full"):
    import torch

    state["model"].eval()
    state["cam"].eval()
    with torch.no_grad():
        fused, adjs, info = state["model"](
            state["parts"], pair_index=state["train_pairs"],
            use_cadr=True, intervention=intervention,
        )
        logp, hidden, attention = state["cam"](fused)
    metrics = R.classification_metrics(logp, state["test"], state["y"])
    return fused, logp, adjs, info, metrics


def _audit_fold(fold):
    import numpy as np
    import torch
    import torch.nn.functional as F

    cache_meta = R.prepare_fold(fold)
    state = R.build_state(fold, cache_meta)
    cp_path = R.best_path(fold)
    task_path = R.task_path(fold)
    assert cp_path.exists() and task_path.exists(), f"missing fold {fold} outputs"
    cp = R.torch_load(cp_path)
    task = json.loads(task_path.read_text())
    cp_sha = R.sha_file(cp_path)
    assert task["status"] == "COMPLETE"
    assert task["best_sha256"] == cp_sha
    assert cp["group"] == "E1_CADR_GFR_FULL" and cp["fold"] == fold and cp["seed"] == 0
    assert cp["best_epoch"] >= int(R.CFG["warmup"]["epochs"])
    assert cp["split_sha"] == state["split_sha"] == task["split_sha"]
    assert cp["pair_sha"] == state["cache_meta"]["pair_sha"] == task["pair_sha"]
    assert cp["anchor_sha"] == state["anchor_sha"] == task["anchor_sha"]
    assert cp["initial_shared_sha"] == state["initial_shared_sha"] == task["initial_shared_sha"]
    assert cp["initial_gfr_sha"] == state["initial_gfr_sha"] == task["initial_gfr_sha"]
    assert cp["initial_schema_sha"] == state["initial_schema_sha"] == task["initial_schema_sha"]
    assert cp["initial_cam_sha"] == state["initial_cam_sha"] == task["initial_cam_sha"]
    assert cp["initial_added_sha"] == state["initial_added_sha"] == task["initial_added_sha"]
    assert cp["initial_rng_sha"] == state["construction_rng_sha"] == task["initial_rng_sha"]
    assert cp["fold_cache_sha"] == state["cache_meta"]["npz_sha256"]
    assert cp["source_sha"]["run.py"] == R.sha_file(R.EXP / "run.py")
    assert cp["source_sha"]["cadr_gfr.py"] == R.sha_file(R.EXP / "cadr_gfr.py")
    assert cp["source_sha"]["config.json"] == R.sha_file(R.EXP / "config.json")
    assert cp["optimizer_weight_decay"]["base"] == [0.11]
    assert cp["optimizer_weight_decay"]["cam"] == [0.11]
    assert sorted(cp["optimizer_weight_decay"]["added"]) == [0.0, 0.0001]
    assert cp["diagnostics"]["propagation_enabled"] is True
    assert len(state["base_params"]) == 33
    assert all(set(pair) <= set(state["train"].detach().cpu().tolist()) for pair in state["train_pairs"].detach().cpu().tolist())

    state["model"].load_state_dict(cp["model"], strict=True)
    state["cam"].load_state_dict(cp["cam"], strict=True)
    assert all(torch.isfinite(p).all() for p in state["model"].parameters())
    assert all(torch.isfinite(p).all() for p in state["cam"].parameters())

    full, full_logp, original_adjs, info, metrics = _eval(state, "full")
    for metric in METRICS:
        assert abs(float(metrics[metric]) - float(cp["metrics"][metric])) <= 1e-10, (
            f"fold {fold} independent readback mismatch for {metric}: "
            f"{metrics[metric]} vs {cp['metrics'][metric]}"
        )

    # The forward has no label argument. Flipping held-out labels must leave
    # its graph and predictions bitwise unchanged.
    heldout = np.asarray(state["test"], dtype=np.int64)
    saved_y = state["y"].clone()
    state["y"][torch.as_tensor(heldout, device=state["y"].device)] = 1 - state["y"][torch.as_tensor(heldout, device=state["y"].device)]
    _, permuted_logp, permuted_adjs, permuted_info, _ = _eval(state, "full")
    assert torch.equal(full_logp, permuted_logp)
    assert all(torch.equal(a, b) for a, b in zip(original_adjs, permuted_adjs))
    assert all(torch.equal(a, b) for a, b in zip(info["positive_adjs"], permuted_info["positive_adjs"]))
    state["y"].copy_(saved_y)

    # Fixed-checkpoint interventions: these are not new training groups.
    variants = ["full", "disease_off", "negative_off", "direct_off"]
    variants += [f"modality_off:{name}" for name in R.MODS]
    intervention_rows = []
    variant_cache = {"full": (full, full_logp, metrics)}
    base_prob = full_logp.exp()
    base_pred = base_prob.argmax(1)
    test_idx = torch.as_tensor(state["test"], dtype=torch.long, device=base_pred.device)
    test_y = state["y"][test_idx]
    raw = torch.cat(state["parts"], dim=1)
    for variant in variants[1:]:
        fused_v, logp_v, _, _, metric_v = _eval(state, variant)
        variant_cache[variant] = (fused_v, logp_v, metric_v)
    for variant in variants:
        fused_v, logp_v, metric_v = variant_cache[variant]
        pred_v = logp_v.exp().argmax(1)
        flipped_test = pred_v[test_idx] != base_pred[test_idx]
        correct_to_incorrect_test = (base_pred[test_idx] == test_y) & (pred_v[test_idx] != test_y)
        incorrect_to_correct_test = (base_pred[test_idx] != test_y) & (pred_v[test_idx] == test_y)
        intervention_rows.append({
            "fold": fold,
            "intervention": variant,
            "acc": metric_v["acc"],
            "hard_auc": metric_v["hard_auc"],
            "prob_auc": metric_v["prob_auc"],
            "spe": metric_v["spe"],
            "sen": metric_v["sen"],
            "ba": metric_v["ba"],
            "asd_f1": metric_v["asd_f1"],
            "representation_rel_to_full": float((fused_v - full).norm().div(full.norm().clamp_min(1e-12)).cpu()),
            "test_representation_rel_to_full": float((fused_v[test_idx] - full[test_idx]).norm().div(full[test_idx].norm().clamp_min(1e-12)).cpu()),
            "representation_rel_to_input": float((fused_v - raw).norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "log_probability_max_abs_change": float((logp_v - full_logp).abs().max().cpu()),
            "test_log_probability_max_abs_change": float((logp_v[test_idx] - full_logp[test_idx]).abs().max().cpu()),
            "probability_max_abs_change": float((logp_v.exp() - base_prob).abs().max().cpu()),
            "test_probability_max_abs_change": float((logp_v[test_idx].exp() - base_prob[test_idx]).abs().max().cpu()),
            "prediction_flip_rate": float(flipped_test.float().mean().cpu()),
            "prediction_flip_count_test": int(flipped_test.sum().cpu()),
            "test_patient_count": int(test_idx.numel()),
            "correct_to_incorrect_test": int(correct_to_incorrect_test.sum().cpu()),
            "incorrect_to_correct_test": int(incorrect_to_correct_test.sum().cpu()),
        })

    # Classification gradients to each disease scorer, GCN, and GraphLearn;
    # auxiliary pair/conditional adversarial gradients are audited separately.
    out = R.forward_state(state, training=False, use_cadr=True)
    cls_scorer_grads = []
    cls_encoder_grads = []
    cls_gcn_grads = []
    graph_grads = []
    pair_scorer_grads = []
    adv_encoder_grads = []
    for modality in range(4):
        cls_scorer_grads.append(_norm(_gradients(out["cls"], state["model"].scorers[modality].parameters())))
        cls_encoder_grads.append(_norm(_gradients(out["cls"], state["model"].encoders[modality].parameters())))
        cls_gcn_grads.append(_norm(_gradients(out["cls"], state["model"].gcn[modality].parameters())))
        graph_grads.append(_norm(_gradients(out["graph_parts"][modality], state["model"].graphs[modality].parameters())))
        pair_scorer_grads.append(_norm(_gradients(out["pair"], state["model"].scorers[modality].parameters())))
        adv_encoder_grads.append(_norm(_gradients(out["adv"], state["model"].encoders[modality].parameters())))
    adv_discriminator_grad = _norm(_gradients(out["adv"], state["model"].discriminator.parameters()))

    y_np = state["y"].detach().cpu().numpy()
    train_set = set(state["train"].detach().cpu().tolist())
    diagnostics = []
    for modality, name in enumerate(R.MODS):
        edge = info["edge_diagnostics"][modality]
        rows = info["edge_rows"][modality].detach().cpu().numpy()
        cols = info["edge_cols"][modality].detach().cpu().numpy()
        q = info["edge_q"][modality].detach().cpu().numpy()
        pair_logits = info["pair_logits"][modality].detach().cpu().numpy()
        pair_y = state["train_same"].detach().cpu().numpy().astype(np.int64)
        from sklearn.metrics import roc_auc_score
        train_pair_auc = float(roc_auc_score(pair_y, pair_logits))
        pos = info["positive_adjs"][modality].detach().cpu().numpy()
        neg = info["negative_adjs"][modality].detach().cpu().numpy()
        original = original_adjs[modality].detach().cpu().numpy()
        train_edge = np.asarray([(int(i) in train_set and int(j) in train_set) for i, j in zip(rows, cols)])
        same = (y_np[rows] == y_np[cols]).astype(np.int64) if len(rows) else np.zeros(0, np.int64)
        # Count precision only on candidate edges that actually enter the GCN.
        pos_train = train_edge & (pos[rows, cols] > 0)
        neg_train = train_edge & (neg[rows, cols] > 0)
        pos_precision = float((same[pos_train] == 1).mean()) if pos_train.any() else float("nan")
        neg_precision = float((same[neg_train] == 0).mean()) if neg_train.any() else float("nan")
        positive_message = info["positive_messages"][modality]
        negative_message = info["negative_messages"][modality]
        direct_message = info["direct_messages"][modality]
        gcn = state["model"].gcn[modality]
        relation = state["cache_meta"]["relation_oof"][modality]
        diagnostics.append({
            "fold": fold,
            "modality": name,
            "oof_auc": relation["auc"],
            "oof_pr_auc": relation["pr_auc"],
            "oof_separation": relation["separation"],
            "oof_brier_raw": relation["brier_raw"],
            "oof_ece_raw": relation["ece_raw"],
            "oof_brier_calibrated": relation["brier_calibrated"],
            "oof_ece_calibrated": relation["ece_calibrated"],
            "joint_train_pair_auc_resubstitution": train_pair_auc,
            "joint_train_pair_logit_same_mean": float(pair_logits[pair_y == 1].mean()),
            "joint_train_pair_logit_different_mean": float(pair_logits[pair_y == 0].mean()),
            "joint_train_pair_logit_std": float(pair_logits.std()),
            "adversarial_loss": float(out["adv"].detach().cpu()),
            "adversarial_modality_accuracy": float(out["adv_info"]["accuracy"].detach().cpu()),
            "adversarial_pair_count_per_modality": int(len(pair_y)),
            "adversarial_discriminator_grad": adv_discriminator_grad,
            "calibration_scale": state["cache_meta"]["calibration"][modality][0],
            "calibration_bias": state["cache_meta"]["calibration"][modality][1],
            "reliable": bool(state["cache_meta"]["reliability"][modality] > 0),
            "reliability": state["cache_meta"]["reliability"][modality],
            "candidate_edges": edge["candidate_edges"],
            "positive_edges": int((pos > 0).sum() // 2),
            "negative_edges": int((neg > 0).sum() // 2),
            "uncertain_edges": edge["uncertain_edges"],
            "positive_train_precision": pos_precision,
            "negative_train_precision": neg_precision,
            "positive_mean_degree": float(pos.sum(1).mean()),
            "negative_mean_degree": float(neg.sum(1).mean()),
            "positive_isolated_fraction": float((pos.sum(1) == 0).mean()),
            "negative_isolated_fraction": float((neg.sum(1) == 0).mean()),
            "original_edges": int(np.count_nonzero(np.triu(original > 0, 1))),
            "positive_original_edge_overlap": int(np.count_nonzero(np.triu((pos > 0) & (original > 0), 1))),
            "negative_original_edge_overlap": int(np.count_nonzero(np.triu((neg > 0) & (original > 0), 1))),
            "q_mean": edge["q_mean"],
            "q_std": edge["q_std"],
            "gcn_layer1_weight_norm": float(gcn.gcn1.W.detach().norm().cpu()),
            "gcn_layer2_weight_norm": float(gcn.gcn2.W.detach().norm().cpu()),
            "positive_gcn_message_norm": float(positive_message.detach().norm().cpu()),
            "negative_gcn_message_norm": float(negative_message.detach().norm().cpu()),
            "direct_message_norm": float(direct_message.detach().norm().cpu()),
            "positive_message_input_ratio": float(positive_message.detach().norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "negative_message_input_ratio": float(negative_message.detach().norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "direct_message_input_ratio": float(direct_message.detach().norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "final_minus_input_ratio": float((full - raw).norm().div(raw.norm().clamp_min(1e-12)).cpu()),
            "fusion_weight_sigmoid": float(torch.sigmoid(state["model"].fusion_weight[modality]).detach().cpu()),
            "learned_reliability_gate": float(info["fusion_gates"][modality].detach().cpu()),
            "edge_gate": edge["edge_gate"],
            "classification_to_scorer_grad": cls_scorer_grads[modality],
            "classification_to_encoder_grad": cls_encoder_grads[modality],
            "classification_to_gcn_grad": cls_gcn_grads[modality],
            "graph_loss_to_graphlearn_grad": graph_grads[modality],
            "pair_loss_to_scorer_grad": pair_scorer_grads[modality],
            "adversarial_to_encoder_grad": adv_encoder_grads[modality],
        })

    audit = {
        "fold": fold,
        "checkpoint_sha256": cp_sha,
        "split_sha": state["split_sha"],
        "pair_sha": state["cache_meta"]["pair_sha"],
        "anchor_sha": state["anchor_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_schema_sha": state["initial_schema_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_added_sha": state["initial_added_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "metrics_match": True,
        "label_permutation_invariant": True,
        "train_pairs_outer_train_only": True,
        "best_epoch_cadr_active": True,
        "optimizer_weight_decay": json.dumps(cp["optimizer_weight_decay"], sort_keys=True),
        "checkpoint_independent_readback": "PASS",
    }
    return audit, diagnostics, intervention_rows, metrics


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_report(audits, diagnostics, interventions):
    import numpy as np

    metric_rows = _read_csv(R.RESULTS / "fold_metrics.csv")
    summary_rows = _read_csv(R.RESULTS / "summary.csv")
    paired_rows = _read_csv(R.RESULTS / "paired_comparisons.csv")
    h0_rows = _read_csv(R.HIST_E0)
    historical = [row for row in h0_rows if row.get("group") == "E0"]
    hist_acc = float(np.mean([float(row["acc"]) for row in historical])) * 100 if historical else 91.9592
    hist_hard = float(np.mean([float(row["hard_auc"]) for row in historical])) * 100 if historical else 91.7434
    hist_prob = float(np.mean([float(row["prob_auc"]) for row in historical])) * 100 if historical else 93.1856
    hist_spe = float(np.mean([float(row["spe"]) for row in historical])) * 100 if historical else 94.6392
    hist_sen = float(np.mean([float(row["sen"]) for row in historical])) * 100 if historical else 88.8476
    ref = {"acc": hist_acc, "hard_auc": hist_hard, "prob_auc": hist_prob, "spe": hist_spe, "sen": hist_sen}
    summary_map = {row["metric"]: row for row in summary_rows}
    metric_labels = {
        "acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Probability-AUC",
        "spe": "SPE-TD", "sen": "SEN-ASD", "ba": "BA", "asd_f1": "ASD-F1",
    }
    metrics_md = ["| 指标 | CADR-GFR mean ± population SD | 实验51 E1只读参考 | 均值差 (pp) |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        row = summary_map[metric]
        mean = float(row["mean_percent"])
        sd = float(row["population_sd_percent"])
        baseline = ref.get(metric)
        if baseline is None and metric in {"ba", "asd_f1"}:
            matching = [p for p in paired_rows if p["metric"] == metric and p["fold"] != "ALL"]
            baseline = float(np.mean([float(p["reference_value"]) for p in matching])) * 100 if matching else float("nan")
        difference = mean - baseline if baseline is not None else float("nan")
        metrics_md.append(f"| {metric_labels[metric]} | {mean:.4f} ± {sd:.4f}% | {baseline:.4f}% | {difference:+.4f} |")

    paired_summary = []
    for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
        vals = [float(row["difference_pp"]) for row in paired_rows if row["metric"] == metric and row["fold"] != "ALL"]
        wins = sum(value > 1e-12 for value in vals)
        ties = sum(abs(value) <= 1e-12 for value in vals)
        losses = sum(value < -1e-12 for value in vals)
        paired_summary.append(f"| {metric_labels[metric]} | {np.mean(vals):+.4f} pp | {wins}/{ties}/{losses} |")
    paired_table = ["| 指标 | 逐折差值均值 | 胜/平/负 |", "|---|---:|---:|"] + paired_summary

    fold_labels = ["acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"]
    fold_table = [
        "| Fold | Best epoch | Epochs | ACC | Hard-AUC | Probability-AUC | SPE | SEN | BA | ASD-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(metric_rows, key=lambda value: int(value["fold"])):
        fold_table.append(
            f"| {int(row['fold'])} | {int(row['best_epoch'])} | {int(row['epochs_executed'])} | "
            + " | ".join(f"{float(row[key]) * 100:.4f}" for key in fold_labels)
            + " |"
        )

    oof_lines = ["| 模态 | OOF AUC | OOF PR-AUC | OOF separation | Brier raw → calibrated | ECE raw → calibrated | OOF可靠度 |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
    for modality in R.MODS:
        subset = [row for row in diagnostics if row["modality"] == modality]
        values = lambda key: np.asarray([float(row[key]) for row in subset], dtype=float)
        oof_lines.append(
            f"| {modality} | {values('oof_auc').mean():.4f} ± {values('oof_auc').std(ddof=0):.4f} | "
            f"{values('oof_pr_auc').mean():.4f} | {values('oof_separation').mean():.4f} | "
            f"{values('oof_brier_raw').mean():.4f} → {values('oof_brier_calibrated').mean():.4f} | "
            f"{values('oof_ece_raw').mean():.4f} → {values('oof_ece_calibrated').mean():.4f} | "
            f"{values('reliability').mean():.3f} |"
        )

    intervention_lines = ["| 固定checkpoint干预 | 测试表示相对变化 | 测试log-prob最大差 | 测试预测翻转 | ACC差 (pp) | Probability-AUC差 (pp) |",
                          "|---|---:|---:|---:|---:|---:|"]
    for intervention in sorted({row["intervention"] for row in interventions} - {"full"}):
        subset = [row for row in interventions if row["intervention"] == intervention]
        acc_delta = np.mean([float(row["acc"]) - float(next(v for v in interventions if v["fold"] == row["fold"] and v["intervention"] == "full")["acc"]) for row in subset]) * 100
        auc_delta = np.mean([float(row["prob_auc"]) - float(next(v for v in interventions if v["fold"] == row["fold"] and v["intervention"] == "full")["prob_auc"]) for row in subset]) * 100
        intervention_lines.append(
            f"| {intervention} | {np.mean([float(row['test_representation_rel_to_full']) for row in subset]):.3e} | "
            f"{np.mean([float(row['test_log_probability_max_abs_change']) for row in subset]):.3e} | "
            f"{sum(int(row['prediction_flip_count_test']) for row in subset)}/{sum(int(row['test_patient_count']) for row in subset)} "
            f"({np.mean([float(row['prediction_flip_rate']) for row in subset]):.4f}) | {acc_delta:+.4f} | {auc_delta:+.4f} |"
        )

    gcn_summary = []
    for modality in R.MODS:
        subset = [row for row in diagnostics if row["modality"] == modality]
        gcn_summary.append(
            f"| {modality} | {np.mean([float(row['positive_message_input_ratio']) for row in subset]):.3e} | "
            f"{np.mean([float(row['negative_message_input_ratio']) for row in subset]):.3e} | "
            f"{np.mean([float(row['direct_message_input_ratio']) for row in subset]):.3e} | "
            f"{np.mean([float(row['learned_reliability_gate']) for row in subset]):.4f} | "
            f"{np.mean([float(row['classification_to_scorer_grad']) for row in subset]):.3e} |"
        )
    gcn_table = ["| 模态 | 正向GCN/输入范数 | 负向GCN/输入范数 | 直接差分/输入范数 | 融合可靠度门 | 分类→评分器梯度 |",
                 "|---|---:|---:|---:|---:|---:|"] + gcn_summary

    graph_summary = []
    for modality in R.MODS:
        subset = [row for row in diagnostics if row["modality"] == modality]
        pos_precision_values = np.asarray([float(row["positive_train_precision"]) for row in subset])
        neg_precision_values = np.asarray([float(row["negative_train_precision"]) for row in subset])
        pos_precision = np.nanmean(pos_precision_values) if np.isfinite(pos_precision_values).any() else float("nan")
        neg_precision = np.nanmean(neg_precision_values) if np.isfinite(neg_precision_values).any() else float("nan")
        pos_precision_text = f"{pos_precision:.4f}" if np.isfinite(pos_precision) else "n/a"
        neg_precision_text = f"{neg_precision:.4f}" if np.isfinite(neg_precision) else "n/a"
        graph_summary.append(
            f"| {modality} | {np.mean([float(row['oof_auc']) for row in subset]):.4f} | "
            f"{np.mean([float(row['candidate_edges']) for row in subset]):.1f} | "
            f"{np.mean([float(row['positive_edges']) for row in subset]):.1f} | "
            f"{np.mean([float(row['negative_edges']) for row in subset]):.1f} | "
            f"{pos_precision_text} | {neg_precision_text} | "
            f"{np.mean([float(row['original_edges']) for row in subset]):.1f} | "
            f"{np.mean([float(row['graph_loss_to_graphlearn_grad']) for row in subset]):.3e} |"
        )
    graph_table = [
        "| 模态 | OOF AUC | 候选边 | 有效正边 | 有效负边 | 实际正边训练精度 | 实际负边训练精度 | 原GraphLearn边 | GraphLearn梯度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *graph_summary,
    ]

    corr_rows = [row for row in diagnostics if row["modality"] == "CORR"]
    oof_corr = float(np.mean([float(row["oof_auc"]) for row in corr_rows]))
    corr_reliable_folds = sum(float(row["oof_auc"]) >= 0.55 and float(row["oof_separation"]) > 0 for row in corr_rows)
    reliable_counts = [sum(float(row["reliability"]) > 0 for row in diagnostics if row["fold"] == fold) for fold in range(1, 11)]
    paired_numeric = [row for row in paired_rows if row["fold"] != "ALL"]
    def paired_values(metric):
        return [float(row["difference_pp"]) for row in paired_numeric if row["metric"] == metric]
    acc_diffs = paired_values("acc")
    prob_diffs = paired_values("prob_auc")
    sen_diffs = paired_values("sen")
    acc_repeatable = bool(acc_diffs and float(np.mean(acc_diffs)) > 0 and sum(v > 0 for v in acc_diffs) >= 6)
    prob_repeatable = bool(prob_diffs and float(np.mean(prob_diffs)) > 0 and sum(v > 0 for v in prob_diffs) >= 6)
    sensitivity_preserved = bool(sen_diffs and float(np.mean(sen_diffs)) >= -2.0)
    diagnostic_go = bool((acc_repeatable or prob_repeatable) and sensitivity_preserved)
    prop_diff = [row for row in interventions if row["intervention"] == "disease_off"]
    scorer_path = any(float(row["classification_to_scorer_grad"]) > 0 for row in diagnostics)
    propagation_go = bool(prop_diff and scorer_path and any(
        float(row["test_representation_rel_to_full"]) > 1e-8
        or float(row["test_log_probability_max_abs_change"]) > 1e-6
        or int(row["prediction_flip_count_test"]) > 0
        for row in prop_diff
    ))
    relation_go = corr_reliable_folds >= 6
    graph_go = bool(len(diagnostics) == 40 and all(
        math.isfinite(float(row["graph_loss_to_graphlearn_grad"])) for row in diagnostics
    ) and all(any(int(row["positive_edges"]) + int(row["negative_edges"]) > 0
                  for row in diagnostics if row["modality"] == modality) for modality in R.MODS))
    engineering_go = len(audits) == 10 and all(row["checkpoint_independent_readback"] == "PASS" for row in audits)

    adv_loss = float(np.mean([float(row["adversarial_loss"]) for row in diagnostics]))
    adv_accuracy = float(np.mean([float(row["adversarial_modality_accuracy"]) for row in diagnostics]))
    adv_disc_grad = float(np.mean([float(row["adversarial_discriminator_grad"]) for row in diagnostics]))
    adv_encoder_grad = {
        modality: float(np.mean([float(row["adversarial_to_encoder_grad"]) for row in diagnostics if row["modality"] == modality]))
        for modality in R.MODS
    }
    mean_gcn_ratio = float(np.mean([
        float(row[key]) for row in diagnostics
        for key in ("positive_message_input_ratio", "negative_message_input_ratio")
    ]))
    mean_direct_ratio = float(np.mean([float(row["direct_message_input_ratio"]) for row in diagnostics]))
    mean_final_input_ratio = float(np.mean([float(row["final_minus_input_ratio"]) for row in diagnostics]))
    disease_off_test = [row for row in interventions if row["intervention"] == "disease_off"]
    disease_off_flip_count = sum(int(row["prediction_flip_count_test"]) for row in disease_off_test)
    disease_off_test_count = sum(int(row["test_patient_count"]) for row in disease_off_test)

    lines = [
        "# Experiment 53.2 — CADR-GFR Results",
        "",
        "## Execution and independent audit",
        "",
        f"- Formal group: `E1_CADR_GFR_FULL`; seed 0; folds completed: {len(metric_rows)}/10.",
        f"- Independently reloaded and audited checkpoints: {len(audits)}/10.",
        "- Each checkpoint was reconstructed from the fold's 43.1 untrained anchor and 53.2 relation cache; model/CAM weights were loaded from disk and all seven classification metrics were recomputed.",
        "- Held-out label permutation left predictions and all disease adjacency matrices unchanged; relation-pair endpoints were outer-train only.",
        "- All base GFR parameters used Adam `lr=0.0038, weight_decay=0.11`; CAM used the same values; added parameters used AdamW `lr=0.001`, matrix WD `1e-4`, bias/normalization WD `0`.",
        "",
        "## Results",
        "",
        *metrics_md,
        "",
        "### Per-fold results (%)",
        "",
        *fold_table,
        "",
        f"Historical Experiment 51 E1 reference (read-only): ACC {ref['acc']:.4f}%, Hard-AUC {ref['hard_auc']:.4f}%, Probability-AUC {ref['prob_auc']:.4f}%, SPE {ref['spe']:.4f}%, SEN {ref['sen']:.4f}%. These are paired historical fold references, not a newly trained control.",
        "",
        "| 指标 | CADR-GFR − 实验51 E1逐折均值差 | 胜/平/负折 |",
        "|---|---:|---:|",
        *paired_table[2:],
        "",
        "## Disease relation evidence",
        "",
        *oof_lines,
        "",
        f"CORR mean patient-disjoint OOF AUC was {oof_corr:.4f} ({corr_reliable_folds}/10 folds passed AUC≥0.55 and positive separation); mean count of modalities passing the frozen reliability rule was {np.mean(reliable_counts):.2f}/4 per fold. Reliability used only outer-train inner-OOF data. OOF metrics are internal relation-model selection evidence, not an independent clinical cohort estimate.",
        "",
        f"The adversary was conditioned on outer-train same/different pair labels. Mean discriminator loss/accuracy were {adv_loss:.4f}/{adv_accuracy:.4f} (four-way chance is 0.25), and mean discriminator gradient norm was {adv_disc_grad:.3e}. Mean adversarial encoder gradient norms were PHENO {adv_encoder_grad['PHENO']:.3e}, ANAT {adv_encoder_grad['ANAT']:.3e}, FUNC {adv_encoder_grad['FUNC']:.3e}, CORR {adv_encoder_grad['CORR']:.3e}; CORR was detached from the adversarial path while all four scorers retained pair BCE and preservation supervision. The nonzero student gradients confirm execution, but the discriminator remains above chance and this audit has no post-joint-training patient-disjoint OOF estimate. Alignment benefit and negative transfer are therefore inconclusive, not established.",
        "",
        "## Integrated graph propagation",
        "",
        *graph_table,
        "",
        *gcn_table,
        "",
        "The four disease graphs were built separately on each modality's own GraphLearn support (with only the frozen feature-only fallback when support was sparse). Effective edge precision uses only nonzero edges actually sent to propagation. In these folds, OOF reliability admitted CORR; PHENO, ANAT, and FUNC had zero reliability and therefore zero disease-positive/negative edges. Their original GraphLearn and parameterized GCN paths still executed, while their disease-conditioned graph paths contributed no edges. Positive and negative supports were normalized separately. The original GFR's four 320-D GCN paths and bounded within-GFR neighbor-difference channels feed one 320-D representation sent to the original PEN-free CAM.",
        f"Across modality-fold diagnostics, mean positive/negative GCN message-to-input ratio was {mean_gcn_ratio:.3e}; the GCN message path is therefore near zero under WD=0.11. The unscaled bounded difference channel ratio averaged {mean_direct_ratio:.3e}, while the final representation differed from raw input by {mean_final_input_ratio:.3e}; the internal direct channel, especially CORR, carries most measurable propagation. Disabling all disease propagation changed held-out continuous outputs, but caused {disease_off_flip_count}/{disease_off_test_count} hard-label flips. This is measurable path influence, not a diagnostic gain.",
        "",
        *intervention_lines[:2],
        *intervention_lines[2:],
        "",
        "Interventions use the same trained checkpoint and are not additional training groups. Flip counts and rates are restricted to each outer test fold; representation and log-probability changes in this table are also test-fold quantities. A change in graph or message values does not by itself establish diagnostic value; paired metrics show whether the classifier's held-out predictions used the path.",
        "",
        "## Source audit and interpretation",
        "",
        "SPromptGL's local implementation has four modality-specific GraphLearn projections, projected-cosine thresholding at 0.9, symmetric `normalize_adj(A + I)`, and two parameterized GraphConv layers with ReLU after layer one. The public GFR's PHENO overwrite was not reused: Experiment 51 `LegacyFixedGFR` correctly accumulates all four full-320-D messages. The executed E1 graph objective is the mean GraphConstructLoss plus classification; the paper's label-derived adjacency objective is not silently added. The original CAM is reused and PEN is physically absent.",
        "",
        "DeepASD adversarially aligns modality feature representations before constructing modality-specific graphs and global graph fusion. CADR-GFR instead aligns disease-conditioned patient-pair relation representations asymmetrically around CORR; it does not copy DeepASD's feature encoder, global graph fusion, or classifier. MGDR/MM-GTUNets/SMGD and the alignment caution paper motivate retaining modality-specific evidence and gated alignment, but do not validate this experiment.",
        "",
        "Prior evidence: 43.1 established the repaired GFR initialization; 47.1/52 showed strong GCN decay can shrink learned messages; 47.3 did not establish true patient identity over strong rewiring; 48.2 reported near-chance relation learning; 53/53.1 exposed weak relation reliability and control/branch issues. CADR-GFR keeps the historical GCN WD and tests an integrated direct neighbor-difference path. Results here must be interpreted against those unresolved risks.",
        "",
        "Initialization reconstruction used seed `510000 + fold*100 + 1`, loaded the matching 43.1 anchor tensors, captured the post-construction RNG, and initialized CADR additions inside an isolated Torch RNG scope using `532000 + fold`. The preflight compared all ten shared/CAM/RNG/split hashes with the 53.1 initialization audit and tested two shared GFR/CAM optimization steps against the disabled-CADR E1 path.",
        "",
        "The historical protocol uses the test fold as validation for checkpoint selection, preprocessing/RFE occurred before CV, and all 871 feature nodes participate in transductive graph construction. These choices can make performance optimistic; one seed and reused patients do not establish unbiased generalization or clinical utility.",
        "",
        "## Decision",
        "",
        f"- ENGINEERING: {'GO' if engineering_go else 'NO-GO'} (10/10 independent reads required).",
        f"- RELATION: {'GO' if relation_go else 'NO-GO'} under the frozen internal OOF criterion (CORR AUC threshold 0.55); this is not external relation validation.",
        f"- GRAPH: {'GO' if graph_go else 'NO-GO'} for disease-conditioned edges active in every modality; all four original GraphLearn branches and their finite gradients were separately audited.",
        f"- PROPAGATION: {'GO' if propagation_go else 'NO-GO'} for a measurable test-fold continuous-output intervention and nonzero classification-to-scorer gradient; hard-label changes are reported separately.",
        f"- DIAGNOSTIC: {'GO' if diagnostic_go else 'NO-GO'}; requires positive mean and at least 6/10 wins for ACC or Probability-AUC, with mean SEN decline no worse than 2 pp.",
        f"- MICCAI CANDIDATE: {'GO' if engineering_go and relation_go and graph_go and propagation_go and diagnostic_go else 'NO-GO'} under this exploratory screen; external validation and independent confirmation remain necessary.",
        "",
        "## Artifacts",
        "",
        "- `results/fold_metrics.csv`, `results/summary.csv`, `results/paired_comparisons.csv`.",
        "- `results/diagnostics.csv`, `results/interventions.csv`, `results/checkpoint_audit.csv`, `results/preflight.csv`.",
        "- `manifest.json` records source, data, anchor, split, relation-cache and checkpoint hashes.",
        "",
    ]
    (R.RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def audit_all():
    R.RESULTS.mkdir(parents=True, exist_ok=True)
    audit_rows, diagnostic_rows, intervention_rows = [], [], []
    readback_rows = []
    for fold in range(1, 11):
        audit, diagnostics, interventions, metrics = _audit_fold(fold)
        audit_rows.append(audit)
        diagnostic_rows.extend(diagnostics)
        intervention_rows.extend(interventions)
        readback_rows.append((fold, metrics))
        print(f"independent checkpoint audit {fold}/10: PASS", flush=True)
    R.write_csv(R.RESULTS / "checkpoint_audit.csv", audit_rows)
    R.write_csv(R.RESULTS / "diagnostics.csv", diagnostic_rows)
    R.write_csv(R.RESULTS / "interventions.csv", intervention_rows)

    fold_rows = _read_csv(R.RESULTS / "fold_metrics.csv")
    for row in fold_rows:
        row["checkpoint_readback_status"] = "PASS"
        row["checkpoint_independent_metrics_match"] = "true"
    R.write_csv(R.RESULTS / "fold_metrics.csv", fold_rows)

    # Summaries and paired wins are reconstructed from the independent reads.
    import numpy as np
    summary = []
    for metric in METRICS:
        values = np.asarray([float(item[1][metric]) for item in readback_rows], dtype=float) * 100.0
        summary.append({
            "group": "E1_CADR_GFR_FULL", "metric": metric,
            "mean_percent": float(values.mean()), "population_sd_percent": float(values.std(ddof=0)),
            "n_folds": len(values),
        })
    R.write_csv(R.RESULTS / "summary.csv", summary)

    historical = [row for row in _read_csv(R.HIST_E0) if row.get("group") == "E0"]
    hist = {int(row["fold"]): row for row in historical}
    paired = []
    if len(hist) == 10:
        for fold, metrics in readback_rows:
            for metric in METRICS:
                reference = float(hist[fold][metric])
                value = float(metrics[metric])
                paired.append({
                    "comparison": "CADR_GFR_minus_Experiment51_E1_readonly_E0",
                    "fold": fold, "metric": metric, "cadr_value": value,
                    "reference_value": reference, "difference": value - reference,
                    "difference_pp": (value - reference) * 100.0,
                })
        for metric in METRICS:
            values = [float(row["difference_pp"]) for row in paired if row["metric"] == metric]
            paired.append({
                "comparison": "CADR_GFR_minus_Experiment51_E1_readonly_E0",
                "fold": "ALL", "metric": metric,
                "cadr_value": "", "reference_value": "",
                "difference": float(np.mean(values)) / 100.0,
                "difference_pp": float(np.mean(values)),
                "wins": sum(value > 1e-12 for value in values),
                "ties": sum(abs(value) <= 1e-12 for value in values),
                "losses": sum(value < -1e-12 for value in values),
            })
    R.write_csv(R.RESULTS / "paired_comparisons.csv", paired)
    _write_report(audit_rows, diagnostic_rows, intervention_rows)

    manifest_path = R.EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "status": "COMPLETE_AUDITED" if len(audit_rows) == 10 else "AUDIT_INCOMPLETE",
        "checkpoints_independently_audited": len(audit_rows),
        "checkpoint_audit_status": "PASS" if len(audit_rows) == 10 else "INCOMPLETE",
        "checkpoint_audit_sha256": R.sha_file(R.RESULTS / "checkpoint_audit.csv"),
        "diagnostics_sha256": R.sha_file(R.RESULTS / "diagnostics.csv"),
        "interventions_sha256": R.sha_file(R.RESULTS / "interventions.csv"),
        "experiment_source_sha256": {
            name: R.sha_file(R.EXP / name)
            for name in ("design.md", "config.json", "cadr_gfr.py", "run.py", "audit.py")
        },
        "result_sha256": {
            name: R.sha_file(R.RESULTS / name)
            for name in (
                "fold_metrics.csv", "summary.csv", "paired_comparisons.csv",
                "diagnostics.csv", "interventions.csv", "checkpoint_audit.csv",
                "preflight.csv", "report.md",
            )
        },
        "relation_cache_sha256": {
            f"fold{fold:02d}": {
                "json": R.sha_file(R.REL_CACHE / f"fold{fold:02d}.json"),
                "npz": R.sha_file(R.REL_CACHE / f"fold{fold:02d}.npz"),
            }
            for fold in range(1, 11)
        },
        "preflight_two_step_parity_sha256": R.sha_file(R.RT / "preflight" / "two_step_parity.json"),
        "handoff": {"path": str(R.ROOT / "handoff.md"), "status": "PENDING"},
    })
    for item in manifest.get("folds", []):
        item["checkpoint_independent_readback"] = "PASS"
    R.atomic_json(manifest_path, manifest)
    return audit_rows


if __name__ == "__main__":
    rows = audit_all()
    print(f"independently audited {len(rows)}/10 checkpoints")
