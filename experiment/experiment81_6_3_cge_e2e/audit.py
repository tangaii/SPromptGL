#!/usr/bin/env python3
"""Independent Best/Latest reload, metric recomputation, and CGE audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run as exp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
BASE_E74 = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr")
E8161 = Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res")
E8162 = Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _assert_metrics(expected, actual, where, tolerance=1e-8):
    for name in exp.METRICS:
        error = abs(float(expected[name]) - float(actual[name]))
        if error > tolerance:
            raise RuntimeError(f"{where}: {name} differs by {error:.12g}")


def _metrics_from_probability(probability, labels):
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    prediction = probability.argmax(axis=1)
    from sklearn.metrics import f1_score, roc_auc_score

    cm = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(labels, prediction):
        cm[int(actual), int(predicted)] += 1
    tp, fn, fp, tn = [int(value) for value in cm.ravel()]
    return {
        "acc": float((prediction == labels).mean()) * 100.0,
        "hard_auc": float(roc_auc_score(np.eye(2)[labels], np.eye(2)[prediction])) * 100.0,
        "prob_auc": float(roc_auc_score(labels, probability[:, 1])) * 100.0,
        "sen": tp / max(tp + fn, 1) * 100.0,
        "spe": tn / max(tn + fp, 1) * 100.0,
        "asd_f1": float(f1_score(labels, prediction, pos_label=0, zero_division=0)) * 100.0,
        "prediction": prediction,
    }


def _prediction_change(reference, candidate, truth):
    reference = np.asarray(reference, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    corrected = int(((reference != truth) & (candidate == truth)).sum())
    harmed = int(((reference == truth) & (candidate != truth)).sum())
    return {"flips": int((reference != candidate).sum()), "corrected": corrected, "harmed": harmed, "net_corrections": corrected - harmed}


def _effective_rank(matrix):
    value = matrix.detach().double()
    value = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(value)
    probability = singular / singular.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(torch.exp(entropy))


def _evaluate_from_cam(state, module, group, z_cam, intervention="full", grouping_override=None):
    grouping = grouping_override or exp.CFG["group_settings"][group]["grouping"]
    enhanced, details = module(z_cam, grouping=grouping, eta=0.0, intervention=intervention)
    logp, hidden, _ = exp.functional_outputlayer(state, enhanced, training=False, detach_params=False)
    return logp, hidden, details


def _gradient_log_summary(log_rows):
    rows = []
    for log_row in log_rows:
        rows.extend(log_row.get("losses", {}).get("gradient_groups", []))
    if not rows:
        return {"epochs_with_gradient_diagnostics": 0, "nonzero_delta_gradient_epochs": 0, "nonzero_controlled_gradient_epochs": 0, "negative_cosine_fraction": 0.0, "mean_base_norm": 0.0, "mean_delta_norm": 0.0, "mean_controlled_ratio": 0.0, "max_controlled_ratio": 0.0}
    return {
        "epochs_with_gradient_diagnostics": len(rows),
        "nonzero_delta_gradient_epochs": int(sum(row["delta_grad_norm"] > 0.0 for row in rows)),
        "nonzero_controlled_gradient_epochs": int(sum(row["final_new_grad_norm"] > 0.0 for row in rows)),
        "negative_cosine_fraction": float(sum(row["conflict"] for row in rows) / len(rows)),
        "mean_base_norm": float(np.mean([row["base_grad_norm"] for row in rows])),
        "mean_delta_norm": float(np.mean([row["delta_grad_norm"] for row in rows])),
        "mean_controlled_ratio": float(np.mean([row["new_to_base_ratio_after_clip"] for row in rows])),
        "max_controlled_ratio": float(max(row["new_to_base_ratio_after_clip"] for row in rows)),
    }


def _audit_fold(group, fold, task, best_path, latest_path):
    best = exp.torch_load(best_path)
    latest = exp.torch_load(latest_path)
    if (best.get("group"), best.get("fold")) != (group, fold) or (latest.get("group"), latest.get("fold")) != (group, fold):
        raise RuntimeError(f"checkpoint identity mismatch: {group} fold={fold}")
    if best.get("source_sha256") != task["source_sha256"] or latest.get("source_sha256") != task["source_sha256"]:
        raise RuntimeError(f"checkpoint source SHA mismatch: {group} fold={fold}")
    if best.get("identity") != task.get("identity") or latest.get("identity") != task.get("identity"):
        raise RuntimeError(f"checkpoint initialization identity mismatch: {group} fold={fold}")

    state, warmup, _ = exp.prepare(fold)
    module = exp.make_module(fold, state["x"].device)
    optimizer = exp.sa_optimizer(module)
    if exp.sha_value(exp._cpu_state(module)) != task["identity"]["relation_init_sha256"]:
        raise RuntimeError(f"SA initialization mismatch: {group} fold={fold}")
    if exp._base_hashes(state) != (task["identity"]["base_model_sha256"], task["identity"]["base_optimizer_sha256"]):
        raise RuntimeError(f"DARG initialization mismatch: {group} fold={fold}")
    if exp.legacy_runtime().rng_sha(exp.legacy_runtime().pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"warmup RNG mismatch: {group} fold={fold}")

    current = latest["current"]
    if set(current["base"]) != {"model", "cam", "opt_base", "opt_cam", "opt_added"}:
        raise RuntimeError(f"Latest missing historical optimizer state: {group} fold={fold}")
    if len(current["optimizer_sa_prcl"]["param_groups"]) != 1:
        raise RuntimeError(f"Latest missing SA optimizer state: {group} fold={fold}")
    exp.load_state_bundle(state, module, optimizer, current)
    exp.legacy_runtime().restore_rng(latest["rng"])
    with torch.no_grad():
        latest_out = exp._eval_forward(state, module, group)
        latest_metrics, _ = exp._metric_payload(latest_out["logp"], state)
    if latest["log_rows"]:
        _assert_metrics(latest["log_rows"][-1]["metrics"], latest_metrics, f"Latest metrics {group} fold={fold}")
    if len(latest["log_rows"]) != len(latest["trace"]):
        raise RuntimeError(f"Latest log/trace length mismatch: {group} fold={fold}")
    if latest["next_epoch"] != task["epochs_executed"]:
        raise RuntimeError(f"Latest epoch mismatch: {group} fold={fold}")
    best_logged = max(row["metrics"]["acc"] for row in latest["log_rows"])
    selected_logged = next(row["epoch"] for row in latest["log_rows"] if row["metrics"]["acc"] == best_logged)
    if selected_logged != best["best_epoch_1based"] or abs(best_logged - best["metrics"]["acc"]) > 1e-8:
        raise RuntimeError(f"Best strict ACC selection mismatch: {group} fold={fold}")
    if fold == 2 and any(not row["losses"].get("fold2_overlap_training_residual_zero", False) for row in latest["log_rows"]):
        raise RuntimeError(f"Fold2 SA overlap protection failed: {group}")

    exp.load_state_bundle(state, module, optimizer, best["best"]["bundle"])
    with torch.no_grad():
        active = exp._eval_forward(state, module, group)
        active_metrics, _ = exp._metric_payload(active["logp"], state)
    _assert_metrics(best["metrics"], active_metrics, f"Best metrics {group} fold={fold}")
    _assert_metrics(task["metrics"], active_metrics, f"Task metrics {group} fold={fold}")
    probability = active["logp"].detach().exp().cpu().numpy()
    prediction = probability.argmax(axis=1)
    if not np.array_equal(prediction, best["best"]["prediction_all"]):
        raise RuntimeError(f"Best predictions changed: {group} fold={fold}")
    if not np.allclose(probability, best["best"]["probability_all"], atol=1e-10, rtol=1e-10):
        raise RuntimeError(f"Best probabilities changed: {group} fold={fold}")
    details = active["fusion"]
    if float(details["delta_ratio"].max()) > 0.100001:
        raise RuntimeError(f"10% residual bound failed: {group} fold={fold}")
    z_cam = active["z_cam"].detach()
    base_logp = active["base_logp"].detach()
    base_hidden = active["base_hidden"].detach()
    base_probability = base_logp.exp().cpu().numpy()
    base_prediction = base_probability.argmax(axis=1)
    truth_all = state["y"].detach().cpu().numpy().astype(np.int64)
    test_ids = np.asarray(state["test"], dtype=np.int64)
    test_t = torch.as_tensor(test_ids, device=z_cam.device, dtype=torch.long)
    truth = truth_all[test_ids]

    historic_path = BASE_E74 / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"
    historic = exp.torch_load(historic_path)
    hist_probability = np.asarray(historic["test_probability"], dtype=np.float64)
    hist_prediction = np.asarray(historic["test_prediction"], dtype=np.int64)
    if not np.array_equal(hist_prediction, hist_probability.argmax(axis=1)):
        raise RuntimeError(f"historical DARG payload prediction mismatch: fold={fold}")
    hist_metrics = _metrics_from_probability(hist_probability, truth)
    historic_row = next(row for row in _read_csv(BASE_E74 / "results/fold_metrics.csv") if row["group"] == "E74.4-G2-Positive-DCR" and int(row["fold"]) == fold)
    _assert_metrics({name: float(historic_row[name]) * 100.0 for name in exp.METRICS}, hist_metrics, f"historical DARG payload fold={fold}")
    b_metrics = _metrics_from_probability(base_probability[test_ids], truth)
    c_metrics = _metrics_from_probability(probability[test_ids], truth)
    changes = {
        "A_to_B": _prediction_change(hist_prediction, base_prediction[test_ids], truth),
        "B_to_C": _prediction_change(base_prediction[test_ids], prediction[test_ids], truth),
        "A_to_C": _prediction_change(hist_prediction, prediction[test_ids], truth),
    }

    mechanism_rows = []
    full_probability = probability[test_ids]
    full_prediction = prediction[test_ids]
    interventions = [
        ("full", exp.CFG["group_settings"][group]["grouping"], "full"),
        ("prompt_off", exp.CFG["group_settings"][group]["grouping"], "prompt_off"),
        ("null_only", exp.CFG["group_settings"][group]["grouping"], "null_only"),
        ("global_off", exp.CFG["group_settings"][group]["grouping"], "global_off"),
        ("medical_grouping", "medical", "full"),
        ("scrambled_grouping", "scrambled", "full"),
    ] + [(f"delete_prompt_{index}", exp.CFG["group_settings"][group]["grouping"], f"delete_prompt_{index}") for index in range(8)]
    for label, grouping, intervention in interventions:
        with torch.no_grad():
            logp, _, diag = _evaluate_from_cam(state, module, group, z_cam, intervention=intervention, grouping_override=grouping)
        candidate_probability = logp.exp().detach().cpu().numpy()[test_ids]
        candidate_prediction = candidate_probability.argmax(axis=1)
        metric = _metrics_from_probability(candidate_probability, truth)
        change = _prediction_change(full_prediction, candidate_prediction, truth)
        if label in ("prompt_off", "null_only"):
            if torch.count_nonzero(diag["raw_delta"]) or torch.count_nonzero(diag["delta"]):
                raise RuntimeError(f"{label} failed residual zero invariant: {group} fold={fold}")
            if not torch.equal(logp, base_logp):
                raise RuntimeError(f"{label} failed original classifier invariant: {group} fold={fold}")
        mechanism_rows.append({
            "group": group,
            "fold": fold,
            "intervention": label,
            "grouping": grouping,
            **{name: metric[name] for name in exp.METRICS},
            "probability_mae_vs_full": float(np.abs(candidate_probability - full_probability).mean()),
            "hard_flips_vs_full": change["flips"],
            "corrected_vs_full": change["corrected"],
            "harmed_vs_full": change["harmed"],
            "net_corrections_vs_full": change["net_corrections"],
            "max_residual_ratio": float(diag["delta_ratio"].max()),
        })

    layer_norm = state["cam"].Outputlayer.mlp_head[0]
    norm_before = F.layer_norm(z_cam, layer_norm.normalized_shape, layer_norm.weight, layer_norm.bias, layer_norm.eps)
    norm_after = F.layer_norm(z_cam + details["delta"], layer_norm.normalized_shape, layer_norm.weight, layer_norm.bias, layer_norm.eps)
    active_hidden = active["hidden"].detach()
    margin_before = base_logp[:, 0] - base_logp[:, 1]
    margin_after = active["logp"][:, 0] - active["logp"][:, 1]
    evidence = details["evidence"].index_select(0, test_t).reshape(len(test_ids), -1)
    response_rank = _effective_rank(evidence)
    beta_normal = details["beta_normal"].index_select(0, test_t)
    beta_abnormal = details["beta_abnormal"].index_select(0, test_t)
    cam_test = z_cam.index_select(0, test_t)
    delta_test = details["delta"].index_select(0, test_t)
    orthogonal = delta_test - ((delta_test * cam_test).sum(-1, keepdim=True) / cam_test.square().sum(-1, keepdim=True).clamp_min(1e-12)) * cam_test
    grad_summary = _gradient_log_summary(latest["log_rows"])
    summary = {
        "group": group,
        "fold": fold,
        "best_epoch_1based": best["best_epoch_1based"],
        "metrics": active_metrics,
        "base_only_metrics": exp._metric_payload(base_logp, state)[0],
        "latest_metrics": latest_metrics,
        "best_recomputed": True,
        "latest_restored": True,
        "initialization_verified": True,
        "optimizer_state_count": 4,
        "prompt_off_exact": True,
        "null_only_exact": True,
        "residual_max_ratio": float(details["delta_ratio"].max()),
        "residual_mean_ratio": float(delta_test.norm(dim=-1).div(cam_test.norm(dim=-1).clamp_min(1e-12)).mean()),
        "cam_norm_mean": float(cam_test.norm(dim=-1).mean()),
        "cam_enhanced_cosine_mean": float(F.cosine_similarity(cam_test, cam_test + delta_test, dim=-1).mean()),
        "residual_parallel_fraction_mean": float((delta_test * cam_test).sum(-1).abs().div(delta_test.norm(dim=-1).mul(cam_test.norm(dim=-1)).clamp_min(1e-12)).mean()),
        "residual_orthogonal_fraction_mean": float(orthogonal.norm(dim=-1).mean() / delta_test.norm(dim=-1).mean().clamp_min(1e-12)),
        "layernorm144_difference_mean": float((norm_after.index_select(0, test_t) - norm_before.index_select(0, test_t)).norm(dim=-1).mean()),
        "hidden_difference_mean": float((active_hidden.index_select(0, test_t) - base_hidden.index_select(0, test_t)).norm(dim=-1).mean()),
        "logit_margin_shift_mean": float((margin_after.index_select(0, test_t) - margin_before.index_select(0, test_t)).mean()),
        "relation_response_effective_rank": response_rank,
        "beta_normal_entropy": float((-(beta_normal * beta_normal.clamp_min(1e-12).log()).sum(1)).mean()),
        "beta_abnormal_entropy": float((-(beta_abnormal * beta_abnormal.clamp_min(1e-12).log()).sum(1)).mean()),
        "beta_normal_patient_sd": float(beta_normal.std(unbiased=False)),
        "beta_abnormal_patient_sd": float(beta_abnormal.std(unbiased=False)),
        "prompt_attention_mean_pairwise_mae": float((details["attention_prompt"].index_select(0, test_t) - details["attention_visual"].index_select(0, test_t)[:, None]).abs().mean()),
        "gradient_summary": grad_summary,
        "historical_A_metrics": {key: hist_metrics[key] for key in exp.METRICS},
        "current_epoch_B_metrics": {key: b_metrics[key] for key in exp.METRICS},
        "current_full_C_metrics": {key: c_metrics[key] for key in exp.METRICS},
        "A_to_B": changes["A_to_B"],
        "B_to_C": changes["B_to_C"],
        "A_to_C": changes["A_to_C"],
    }
    mechanism_rows.append({
        "group": group,
        "fold": fold,
        "intervention": "selected_best_summary",
        "best_epoch_1based": best["best_epoch_1based"],
        "A_historical_DARG_ACC": hist_metrics["acc"],
        "A_historical_DARG_HardAUC": hist_metrics["hard_auc"],
        "B_current_epoch_DARG_ACC": b_metrics["acc"],
        "B_current_epoch_DARG_HardAUC": b_metrics["hard_auc"],
        "C_full_SA_PRCL_ACC": c_metrics["acc"],
        "C_full_SA_PRCL_HardAUC": c_metrics["hard_auc"],
        "A_to_B_net_corrections": changes["A_to_B"]["net_corrections"],
        "B_to_C_net_corrections": changes["B_to_C"]["net_corrections"],
        "A_to_C_net_corrections": changes["A_to_C"]["net_corrections"],
        "gradient_summary": json.dumps(grad_summary, sort_keys=True),
        "residual_mean_ratio": summary["residual_mean_ratio"],
        "residual_max_ratio": summary["residual_max_ratio"],
        "layernorm144_difference_mean": summary["layernorm144_difference_mean"],
        "hidden_difference_mean": summary["hidden_difference_mean"],
        "logit_margin_shift_mean": summary["logit_margin_shift_mean"],
        "relationship_response_effective_rank": response_rank,
    })
    return summary, mechanism_rows


def _load_baseline_rows():
    references = {}
    for name, directory, group, scale in [
        ("DARG", BASE_E74, "E74.4-G2-Positive-DCR", 100.0),
        ("E81.6.1-G1", E8161, "E81.6.1-G1-Medical-CPRL-10", 1.0),
        ("E81.6.2-G1", E8162, "SA-PRCL-G1-Protected", 1.0),
        ("E81.6.2-G3", E8162, "SA-PRCL-G3-Joint", 1.0),
    ]:
        rows = {}
        for row in _read_csv(directory / "results/fold_metrics.csv"):
            if row["group"] == group:
                rows[int(row["fold"])] = {metric: float(row[metric]) * scale for metric in exp.METRICS}
        if set(rows) != set(range(1, 11)):
            raise RuntimeError(f"historical reference incomplete: {name}")
        references[name] = rows
    return references


def _paired_rows(current, references):
    rows = []
    left_group = exp.GROUPS[0]
    comparisons = [(left_group, name, current[left_group], rows_ref) for name, rows_ref in references.items()]
    if len(exp.GROUPS) > 1 and exp.GROUPS[1] in current:
        comparisons.append((left_group, exp.GROUPS[1], current[left_group], current[exp.GROUPS[1]]))
    for left_name, right_name, left, right in comparisons:
        for subset_name, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [f for f in range(1, 11) if f != 2])):
            for metric in exp.METRICS:
                diffs = np.asarray([float(left[f][metric]) - float(right[f][metric]) for f in folds], dtype=np.float64)
                mean = float(diffs.mean())
                sd = float(diffs.std(ddof=0))
                half = 1.96 * sd / math.sqrt(len(diffs))
                rows.append({
                    "comparison": f"{left_name}_vs_{right_name}", "subset": subset_name, "metric": metric, "n": len(diffs),
                    "mean_paired_diff_percentage_points": mean, "paired_sd_population": sd,
                    "descriptive_interval_low": mean - half, "descriptive_interval_high": mean + half,
                    "wins": int((diffs > 1e-12).sum()), "ties": int((np.abs(diffs) <= 1e-12).sum()), "losses": int((diffs < -1e-12).sum()),
                    "fold_differences": json.dumps({str(f): float(left[f][metric] - right[f][metric]) for f in folds}, separators=(",", ":")),
                })
    return rows


def _summary_rows(current, references):
    rows = []
    for model, folds in {**references, **current}.items():
        for metric in exp.METRICS:
            all_values = [folds[f][metric] for f in range(1, 11)]
            nine_values = [folds[f][metric] for f in range(1, 11) if f != 2]
            rows.append({"model": model, "subset": "all10", "metric": metric, "mean": float(np.mean(all_values)), "population_sd": float(np.std(all_values, ddof=0))})
            rows.append({"model": model, "subset": "exclude_fold2", "metric": metric, "mean": float(np.mean(nine_values)), "population_sd": float(np.std(nine_values, ddof=0))})
    return rows


def _metric_name(metric):
    return {"acc": "ACC", "hard_auc": "Hard-AUC/BA", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}[metric]


def _write_report(fold_rows, paired_rows, summary_rows, status):
    models = list(status["models_reported"])
    lines = [
        "# Experiment 81.6.3 SA-PRCL-CGE independent audit", "",
        f"Status: {status['completed_tasks']}/{status['expected_tasks']} requested tasks independently audited. Audit status is `{status['status']}`.", "",
        "All metrics are percentages with 10-fold mean plus population SD. Paired intervals are descriptive intervals and are not significance tests.", "",
        "|Model|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        values = []
        for metric in exp.METRICS:
            row = next((item for item in summary_rows if item["model"] == model and item["subset"] == "all10" and item["metric"] == metric), None)
            values.append("n/a" if row is None else f"{row['mean']:.4f}±{row['population_sd']:.4f}")
        lines.append(f"|{model}|" + "|".join(values) + "|")
    lines.extend(["", "## Excluding Fold 2", "", "|Model|Metric|9-fold mean±population SD|", "|---|---|---:|"])
    for model in models:
        for metric in exp.METRICS:
            row = next((item for item in summary_rows if item["model"] == model and item["subset"] == "exclude_fold2" and item["metric"] == metric), None)
            if row:
                lines.append(f"|{model}|{_metric_name(metric)}|{row['mean']:.4f}±{row['population_sd']:.4f}|")
    lines.extend(["", "## Paired comparisons", "", "|Comparison|Subset|Metric|Mean diff|Descriptive interval|W/T/L|", "|---|---|---|---:|---:|---:|"])
    for row in paired_rows:
        lines.append(f"|{row['comparison']}|{row['subset']}|{_metric_name(row['metric'])}|{row['mean_paired_diff_percentage_points']:+.4f}|[{row['descriptive_interval_low']:+.4f}, {row['descriptive_interval_high']:+.4f}]|{row['wins']}/{row['ties']}/{row['losses']}|")
    lines.extend(["", "## Same-epoch correction", "", "A is historical DARG Best, B is the current selected epoch with the residual disabled, and C is the complete CGE output.", "", "|Group|Subset|Stage|Corrected|Harmed|Net|", "|---|---|---|---:|---:|---:|"])
    for group in status["groups"]:
        for subset in ("all10", "exclude_fold2"):
            for stage in ("A_to_B", "B_to_C", "A_to_C"):
                item = status["mechanism_correction_totals"][group][subset][stage]
                lines.append(f"|{group}|{subset}|{stage}|{item['corrected']}|{item['harmed']}|{item['net_corrections']}|")
    lines.extend([
        "", "## Audit and research decisions", "",
        f"- Best independent metric recomputations: {status['best_recomputed']}/{status['completed_tasks']}.",
        f"- Latest reloads: {status['latest_restored']}/{status['completed_tasks']}.",
        f"- Historical starts and same-fold SA initialization: {status['historical_starts']}/10 and {status['same_fold_initializations']}/10.",
        f"- Controlled end-to-end gradient mechanism: `{status['gradient_mechanism_status']}`.",
        f"- Public RNG and Dropout replay: `{status['rng_status']}`.", f"- Fold 2 protection: `{status['fold2_status']}`.",
        f"- G2 continuation: `{status['g2_gate']}`.", f"- Research decision: `{status['research_decision']}`.", "",
        "Prompt-off and Null-only disable the whole residual branch and cannot alone establish medical semantic validity. G2 keeps the same original medical text and changes only the fixed grouping indices. Fold 2 overlap, test=validation, global pre-CV processing, RFE and transductive graph construction remain historical protocol limitations.", "",
        f"Source SHA256: `{status['source_sha256']}`. Prompt cache SHA256: `{status['prompt_cache_sha256']}`.",
    ])
    body = "\n".join(lines) + "\n"
    target = RESULTS / "report.md"
    temp = target.with_suffix(".tmp")
    temp.write_text(body, encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(target)
    if target.read_text(encoding="utf-8") != body:
        raise IOError("report readback failed")


def _gate_status(paired_rows, gradient_status, groups):
    def row(comparison, subset, metric):
        return next((item for item in paired_rows if item["comparison"] == comparison and item["subset"] == subset and item["metric"] == metric), None)

    checks = {}
    for subset in ("all10", "exclude_fold2"):
        for metric in ("acc", "hard_auc"):
            item = row(f"{groups[0]}_vs_DARG", subset, metric)
            checks[f"{subset}_{metric}"] = bool(item and item["mean_paired_diff_percentage_points"] > 0 and item["wins"] > item["losses"])
    performance = all(checks.values())
    grouping = False
    if len(groups) > 1:
        tests = []
        for subset in ("all10", "exclude_fold2"):
            for metric in ("acc", "hard_auc"):
                item = row(f"{groups[0]}_vs_{groups[1]}", subset, metric)
                tests.append(bool(item and item["mean_paired_diff_percentage_points"] > 0 and item["wins"] > item["losses"]))
        grouping = all(tests)
    if not gradient_status:
        decision = "NO_GO_CGE_E2E"
    elif not performance:
        decision = "E2E_MECHANISM_PASS_PERFORMANCE_NO_GO"
    elif len(groups) > 1 and not grouping:
        decision = "PRELIMINARY_GO_CGE_E2E;SEMANTIC_GROUPING_INCREMENT_NOT_ESTABLISHED"
    elif len(groups) > 1:
        decision = "PRELIMINARY_GO_SEMANTIC_GROUPING"
    else:
        decision = "PRELIMINARY_GO_CGE_E2E"
    return {"performance": performance, "grouping": grouping, "checks": checks, "decision": decision}


def CFG_groups():
    return tuple(exp.CFG["execution"]["formal_groups_initial"])


def audit(groups=None):
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_hash = exp.source_sha()
    if source_hash != preflight.get("source_sha256"):
        raise RuntimeError("source SHA changed since passing preflight")
    groups = tuple(groups or CFG_groups())
    if any(group not in exp.GROUPS for group in groups):
        raise ValueError(f"unknown audit group: {groups}")
    invariant_before = exp.legacy_runtime().pack_rng()
    invariant_recheck = exp._model_preflight()
    exp.legacy_runtime().restore_rng(invariant_before)
    task_rows, fold_rows, mechanism_rows, audit_rows = [], [], [], []
    current = {}
    for group in groups:
        current[group] = {}
        for fold in range(1, 11):
            task_file, best_file, latest_file, log_file = exp._task_paths(group, fold)
            if not all(path.is_file() for path in (task_file, best_file, latest_file, log_file)):
                raise FileNotFoundError(f"incomplete artifacts: {group} fold={fold}")
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_hash:
                raise RuntimeError(f"task status/source mismatch: {group} fold={fold}")
            if task.get("best_checkpoint_sha256") != exp.sha_file(best_file) or task.get("latest_checkpoint_sha256") != exp.sha_file(latest_file) or task.get("scalar_log_sha256") != exp.sha_file(log_file):
                raise RuntimeError(f"artifact hash mismatch: {group} fold={fold}")
            summary, rows = _audit_fold(group, fold, task, best_file, latest_file)
            current[group][fold] = summary["metrics"]
            audit_rows.append(summary)
            mechanism_rows.extend(rows)
            fold_rows.append({"group": group, "fold": fold, **summary["metrics"], **{f"base_{key}": value for key, value in summary["base_only_metrics"].items()}, "best_epoch_1based": summary["best_epoch_1based"], "current_off_acc": summary["base_only_metrics"]["acc"], "current_off_hard_auc": summary["base_only_metrics"]["hard_auc"], "checkpoint_sha256": task["best_checkpoint_sha256"], "latest_sha256": task["latest_checkpoint_sha256"], "independent_checkpoint_audit": "PASS"})
            task_rows.append(task)
            print(f"AUDIT PASS {group} fold={fold} ACC={summary['metrics']['acc']:.4f}", flush=True)

    identity_fields = ("warmup_sha256", "warmup_rng_sha256", "public_rng_sha256", "base_model_sha256", "base_optimizer_sha256", "relation_init_sha256", "split_sha", "pair_sha", "membership")
    same_fold_initializations = 0
    for fold in range(1, 11):
        fold_tasks = [task for task in task_rows if int(task["fold"]) == fold]
        identities = [task["identity"] for task in fold_tasks]
        if identities and any(any(identity.get(field) != identities[0].get(field) for identity in identities[1:]) for field in identity_fields):
            raise RuntimeError(f"same-fold initialization mismatch at fold {fold}")
        same_fold_initializations += 1

    baseline = _load_baseline_rows()
    summary_rows = _summary_rows(current, baseline)
    paired_rows = _paired_rows(current, baseline)
    exp.write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    exp.write_csv(RESULTS / "paired_comparisons.csv", paired_rows)
    exp.write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    gradient_status = all(summary["gradient_summary"]["nonzero_controlled_gradient_epochs"] > 0 and summary["gradient_summary"]["nonzero_delta_gradient_epochs"] > 0 for summary in audit_rows)
    gate = _gate_status(paired_rows, gradient_status, groups)
    mechanism_correction_totals = {}
    for group in groups:
        mechanism_correction_totals[group] = {}
        for subset, folds in (("all10", set(range(1, 11))), ("exclude_fold2", set(range(1, 11)) - {2})):
            selected = [row for row in audit_rows if row["group"] == group and row["fold"] in folds]
            mechanism_correction_totals[group][subset] = {}
            for stage in ("A_to_B", "B_to_C", "A_to_C"):
                corrected = sum(row[stage]["corrected"] for row in selected)
                harmed = sum(row[stage]["harmed"] for row in selected)
                mechanism_correction_totals[group][subset][stage] = {"corrected": corrected, "harmed": harmed, "net_corrections": corrected - harmed}
    status = {
        "status": "PASS", "groups": list(groups), "models_reported": list(groups) + ["DARG", "E81.6.1-G1", "E81.6.2-G1", "E81.6.2-G3"],
        "completed_tasks": len(task_rows), "expected_tasks": len(groups) * 10, "best_recomputed": len(fold_rows), "latest_restored": len(fold_rows), "historical_starts": len({int(row["fold"]) for row in task_rows}), "same_fold_initializations": same_fold_initializations, "same_fold_initialization_fields": list(identity_fields),
        "gradient_mechanism_status": "PASS" if gradient_status else "FAIL", "rng_status": "PASS", "fold2_status": "PASS", "invariant_recheck": invariant_recheck,
        "source_sha256": source_hash, "prompt_cache_sha256": exp.sha_file(HERE / "prompt_features.pt"), "g1_gate": gate["performance"], "g2_gate": gate["grouping"] if len(groups) > 1 else "NOT_EXECUTED_CONDITIONAL", "g2_gate_checks": gate["checks"], "research_decision": gate["decision"], "decision_reasons": gate, "mechanism_correction_totals": mechanism_correction_totals, "fold_mechanism_summary": audit_rows, "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    exp.atomic_json(RESULTS / "audit_status.json", status)
    _write_report(fold_rows, paired_rows, summary_rows, status)
    exp.atomic_json(HERE / "manifest.json", {"experiment": exp.CFG["experiment"], "status": "COMPLETE_AUDITED", "groups": list(groups), "expected_tasks": len(groups) * 10, "completed_tasks": len(task_rows), "independent_audits": len(task_rows), "audit_status": "PASS", "research_decision": gate["decision"], "source_sha256": source_hash, "audit_status_sha256": exp.sha_file(RESULTS / "audit_status.json"), "fold_metrics_sha256": exp.sha_file(RESULTS / "fold_metrics.csv"), "paired_comparisons_sha256": exp.sha_file(RESULTS / "paired_comparisons.csv"), "mechanism_sha256": exp.sha_file(RESULTS / "mechanism.csv"), "report_sha256": exp.sha_file(RESULTS / "report.md")})
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups")
    args = parser.parse_args()
    selected = tuple(group for group in (args.groups.split(",") if args.groups else CFG_groups()) if group)
    try:
        audit(selected)
    except Exception as error:
        exp.atomic_json(RESULTS / "audit_status.json", {"status": "FAIL", "error": repr(error), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise
