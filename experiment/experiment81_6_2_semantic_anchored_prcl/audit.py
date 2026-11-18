#!/usr/bin/env python3
"""Independent Best/Latest reload, metric recomputation, and mechanism audit."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run as exp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
BASE_E74 = Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr")
E8161 = Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res")


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _assert_metrics(expected, actual, where, tolerance=1e-10):
    for name in exp.METRICS:
        error = abs(float(expected[name]) - float(actual[name]))
        if error > tolerance:
            raise RuntimeError(f"{where}: {name} differs by {error:.12g}")


def _mean_std(rows, metric):
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _prediction_change(reference, candidate, truth):
    ref = np.asarray(reference, dtype=np.int64)
    cand = np.asarray(candidate, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if ref.shape != cand.shape or ref.shape != truth.shape:
        raise RuntimeError("mechanism prediction lengths do not align")
    corrected = int(((ref != truth) & (cand == truth)).sum())
    harmed = int(((ref == truth) & (cand != truth)).sum())
    return {"flips": int((ref != cand).sum()), "corrected": corrected, "harmed": harmed, "net_corrections": corrected - harmed}


def _metrics_from_probability(probability, labels):
    prediction = np.asarray(probability).argmax(axis=1)
    y = np.asarray(labels, dtype=np.int64)
    from sklearn.metrics import f1_score, roc_auc_score
    cm = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(y, prediction):
        cm[int(actual), int(predicted)] += 1
    tp, fn, fp, tn = [int(value) for value in cm.ravel()]
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    hard = float(roc_auc_score(np.eye(2)[y], np.eye(2)[prediction]))
    return {
        "acc": float((prediction == y).mean()) * 100.0,
        "hard_auc": hard * 100.0,
        "prob_auc": float(roc_auc_score(y, np.asarray(probability)[:, 1])) * 100.0,
        "sen": sen * 100.0,
        "spe": spe * 100.0,
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)) * 100.0,
        "prediction": prediction,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def _evaluate_from_cam(state, module, group, z_cam, intervention="full", grouping_override=None):
    settings = exp.CFG["group_settings"][group]
    grouping = grouping_override or settings["grouping"]
    eta = 0.0
    enhanced, details = module(z_cam, grouping=grouping, eta=eta, intervention=intervention)
    logp, hidden, _ = exp.frozen_outputlayer(state, enhanced, training=False)
    return logp, hidden, details


def _effective_rank(matrix):
    value = matrix.detach().double()
    value = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(value)
    probability = singular / singular.sum().clamp_min(1e-30)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    return float(torch.exp(entropy))


def _audit_fold(group, fold, task, best_path, latest_path):
    best = exp.torch_load(best_path)
    latest = exp.torch_load(latest_path)
    if best.get("group") != group or best.get("fold") != fold:
        raise RuntimeError(f"Best identity mismatch: {group} fold={fold}")
    if latest.get("group") != group or latest.get("fold") != fold:
        raise RuntimeError(f"Latest identity mismatch: {group} fold={fold}")
    if best.get("source_sha256") != task["source_sha256"] or latest.get("source_sha256") != task["source_sha256"]:
        raise RuntimeError(f"checkpoint source SHA mismatch: {group} fold={fold}")
    if latest.get("identity") != task.get("identity") or best.get("identity") != task.get("identity"):
        raise RuntimeError(f"checkpoint initialization identity mismatch: {group} fold={fold}")

    state, warmup, _ = exp.prepare(fold)
    module = exp.make_module(fold, state["x"].device)
    init_sha = exp.sha_value(exp._cpu_state(module))
    if init_sha != task["identity"]["relation_init_sha256"]:
        raise RuntimeError(f"initial SA-PRCL parameters changed: {group} fold={fold}")
    if exp._base_hashes(state) != (task["identity"]["base_model_sha256"], task["identity"]["base_optimizer_sha256"]):
        raise RuntimeError(f"E61 warmup model/optimizer identity changed: {group} fold={fold}")
    if exp.legacy_runtime().rng_sha(exp.legacy_runtime().pack_rng()) != warmup["rng_sha"]:
        raise RuntimeError(f"E61 warmup public RNG differs: {group} fold={fold}")
    settings = exp.CFG["group_settings"][group]
    initial_optimizer = exp.sa_optimizer(module)

    # Latest is loaded on a fresh history state before the Best model is used for mechanism tests.
    exp.load_state_bundle(state, module, initial_optimizer, latest["current"])
    if set(latest["current"]["base"]) != {"model", "cam", "opt_base", "opt_cam", "opt_added"}:
        raise RuntimeError(f"Latest lacks one of the original three optimizers: {group} fold={fold}")
    if len(latest["current"]["optimizer_sa_prcl"]["param_groups"]) != 1:
        raise RuntimeError(f"Latest lacks SA-PRCL optimizer state: {group} fold={fold}")
    exp.legacy_runtime().restore_rng(latest["rng"])
    with torch.no_grad():
        latest_out = exp._eval_forward(state, module, group)
        latest_metrics, _ = exp._metric_payload(latest_out["logp"], state)
    if latest["log_rows"]:
        _assert_metrics(latest["log_rows"][-1]["metrics"], latest_metrics, f"Latest independent recompute {group} fold={fold}")
    if len(latest["log_rows"]) != len(latest["trace"]):
        raise RuntimeError(f"Latest scalar/RNG trace row count differs: {group} fold={fold}")
    if len(latest["trace"]) != int(task["epochs_executed"]) - 10 or latest["next_epoch"] != task["epochs_executed"]:
        raise RuntimeError(f"Latest epoch count differs from task: {group} fold={fold}")
    best_logged = max(row["metrics"]["acc"] for row in latest["log_rows"])
    selected_logged = next(row["epoch"] for row in latest["log_rows"] if row["metrics"]["acc"] == best_logged)
    if selected_logged != best["best_epoch_1based"] or abs(best_logged - best["metrics"]["acc"]) > 1e-12:
        raise RuntimeError(f"Best does not follow strict earliest-ACC selection: {group} fold={fold}")
    if fold == 2 and any(not row["losses"].get("fold2_overlap_training_residual_zero", False) for row in latest["log_rows"]):
        raise RuntimeError(f"Fold2 added-branch overlap residual was not zero in every training epoch: {group}")

    # Restore selected Best and independently recompute all Best metrics and mechanisms.
    exp.load_state_bundle(state, module, initial_optimizer, best["best"]["bundle"])
    with torch.no_grad():
        active = exp._eval_forward(state, module, group)
        active_metrics, _ = exp._metric_payload(active["logp"], state)
    _assert_metrics(best["metrics"], active_metrics, f"Best independent recompute {group} fold={fold}")
    _assert_metrics(task["metrics"], active_metrics, f"Task vs Best {group} fold={fold}")
    best_probability = active["logp"].detach().exp().cpu().numpy()
    best_prediction = best_probability.argmax(axis=1)
    if not np.array_equal(best_prediction, best["best"]["prediction_all"]):
        raise RuntimeError(f"Best full-patient predictions changed: {group} fold={fold}")
    if not np.allclose(best_probability, best["best"]["probability_all"], atol=1e-12, rtol=1e-12):
        raise RuntimeError(f"Best full-patient probabilities changed: {group} fold={fold}")
    details = active["fusion"]
    if float(details["delta_ratio"].max()) > 0.100001:
        raise RuntimeError(f"10% residual bound failed: {group} fold={fold}")

    # Read the original DARG branch at this selected epoch; no full CAM is repeated for each intervention.
    capture = {"z_cam": active["z_cam"]}
    if settings["protected"]:
        base_logp = active["base_logp"]
        base_hidden = active["base_hidden"]
    else:
        hooks = exp._attach_capture(state, capture, capture_dropout=False)
        try:
            with torch.no_grad():
                base_out = exp.base_forward(state, training=False)
        finally:
            exp._remove_hooks(hooks)
        base_logp, base_hidden = base_out["logp"], base_out["hidden"]
    z_cam = capture["z_cam"]
    if tuple(z_cam.shape) != (871, 144):
        raise RuntimeError(f"expected captured CAM144 for all patients, got {tuple(z_cam.shape)}")
    base_probability = base_logp.detach().exp().cpu().numpy()
    base_prediction = base_probability.argmax(axis=1)
    truth_all = state["y"].detach().cpu().numpy().astype(np.int64)
    test_ids = np.asarray(state["test"], dtype=np.int64)
    test_t = torch.as_tensor(test_ids, device=z_cam.device, dtype=torch.long)
    truth = truth_all[test_ids]

    # The old E74.4 Best is the A reference; the selected current base at this epoch is B.
    historic_path = BASE_E74 / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"
    historic = exp.torch_load(historic_path)
    hist_probability = np.asarray(historic["test_probability"], dtype=np.float64)
    hist_prediction = np.asarray(historic["test_prediction"], dtype=np.int64)
    if hist_probability.shape != (len(test_ids), 2) or hist_prediction.shape != truth.shape:
        raise RuntimeError(f"historical DARG Best outputs do not align with fold {fold}")
    b_prediction = base_prediction[test_ids]
    c_prediction = best_prediction[test_ids]
    hist_metrics = _metrics_from_probability(hist_probability, truth)
    if not np.array_equal(hist_prediction, hist_probability.argmax(axis=1)):
        raise RuntimeError(f"stored historical DARG prediction/probability disagree: fold={fold}")
    historic_row = next(row for row in _read_csv(BASE_E74 / "results/fold_metrics.csv")
                        if row["group"] == "E74.4-G2-Positive-DCR" and int(row["fold"]) == fold)
    _assert_metrics({name: float(historic_row[name]) * 100.0 for name in exp.METRICS},
                    {name: hist_metrics[name] for name in exp.METRICS}, f"historical DARG Best payload fold={fold}")
    b_metrics = _metrics_from_probability(base_probability[test_ids], truth)
    c_metrics = _metrics_from_probability(best_probability[test_ids], truth)
    changes = {
        "A_to_B": _prediction_change(hist_prediction, b_prediction, truth),
        "B_to_C": _prediction_change(b_prediction, c_prediction, truth),
        "A_to_C": _prediction_change(hist_prediction, c_prediction, truth),
    }

    # Reuse captured CAM144 for all frozen diagnostic interventions; no attention tensors are persisted.
    intervention_specs = [("full", settings["grouping"]), ("prompt_off", settings["grouping"]), ("null_only", settings["grouping"]),
                          ("global_off", settings["grouping"]), ("group_medical", "medical"), ("group_scrambled", "scrambled")]
    intervention_specs.extend((f"delete_prompt_{index}", settings["grouping"]) for index in range(8))
    mechanism_rows = []
    active_prediction = best_prediction[test_ids]
    active_probability = best_probability[test_ids]
    for label, grouping in intervention_specs:
        intervention = label if label in ("full", "prompt_off", "null_only", "global_off") or label.startswith("delete_prompt_") else "full"
        with torch.no_grad():
            logp, hidden, diag = _evaluate_from_cam(state, module, group, z_cam, intervention=intervention, grouping_override=grouping)
        probability = logp.detach().exp().cpu().numpy()[test_ids]
        prediction = probability.argmax(axis=1)
        metric = _metrics_from_probability(probability, truth)
        change = _prediction_change(active_prediction, prediction, truth)
        if label in ("prompt_off", "null_only"):
            if torch.count_nonzero(diag["raw_delta"]) or torch.count_nonzero(diag["delta"]):
                raise RuntimeError(f"{label} did not zero the trained residual: {group} fold={fold}")
            if not torch.equal(logp, base_logp):
                raise RuntimeError(f"{label} did not restore the original DARG Logp: {group} fold={fold}")
        ratio = diag["delta_ratio"].detach()
        mechanism_rows.append({
            "group": group,
            "fold": fold,
            "intervention": label,
            "grouping": grouping,
            **{name: metric[name] for name in exp.METRICS},
            "probability_mae_vs_full": float(np.abs(probability - active_probability).mean()),
            "hard_flips_vs_full": change["flips"],
            "corrected_vs_full": change["corrected"],
            "harmed_vs_full": change["harmed"],
            "net_corrections_vs_full": change["net_corrections"],
            "max_residual_ratio": float(ratio.max()),
        })

    base_hidden = base_hidden.detach()
    active_hidden = active["hidden"].detach()
    active_delta = details["delta"].detach()
    cam = z_cam.detach()
    layer_norm = state["cam"].Outputlayer.mlp_head[0]
    norm_before = F_layer_norm(cam, layer_norm)
    norm_after = F_layer_norm(cam + active_delta, layer_norm)
    margin_before = base_logp[:, 0] - base_logp[:, 1]
    margin_after = active["logp"][:, 0] - active["logp"][:, 1]
    relation_matrix = details["evidence"].index_select(0, test_t).reshape(len(test_ids), -1)
    response_rank = _effective_rank(relation_matrix)
    prompt_rows = []
    for prompt_index in range(8):
        prompt_attention = details["attention_prompt"].index_select(0, test_t)[:, prompt_index]
        prompt_evidence = details["evidence"].index_select(0, test_t)[:, prompt_index]
        beta_normal = details["beta_normal"].index_select(0, test_t)
        beta_abnormal = details["beta_abnormal"].index_select(0, test_t)
        prompt_rows.append({
            "group": group,
            "fold": fold,
            "intervention": f"prompt_{prompt_index}",
            "prompt_index": prompt_index,
            "attention_mean": float(prompt_attention.mean()),
            "attention_sd_across_patients": float(prompt_attention.std(dim=0, unbiased=False).mean()),
            "evidence_mean": float(prompt_evidence.mean()),
            "evidence_sd": float(prompt_evidence.std(unbiased=False)),
            "delta_message_norm": float(details["delta_messages"].index_select(0, test_t)[:, prompt_index].norm(dim=-1).mean()),
            "normal_beta_mean": float(beta_normal[:, details["normal_indices"].index(prompt_index)].mean()) if prompt_index in details["normal_indices"] else None,
            "abnormal_beta_mean": float(beta_abnormal[:, details["abnormal_indices"].index(prompt_index)].mean()) if prompt_index in details["abnormal_indices"] else None,
            "effective_response_rank": response_rank,
        })
    mechanism_rows.extend(prompt_rows)
    summary = {
        "group": group,
        "fold": fold,
        "best_epoch_1based": best["best_epoch_1based"],
        "metrics": active_metrics,
        "base_only_metrics": _metric_payload_from_logp(base_logp, state),
        "best_acc_strict_selection": True,
        "latest_restored": True,
        "initial_model_and_optimizer_identity": True,
        "optimizer_state_count": 4,
        "prompt_off_exact": True,
        "null_only_exact": True,
        "residual_max_ratio": float(details["delta_ratio"].max()),
        "cam_norm_mean": float(cam.index_select(0, test_t).norm(dim=-1).mean()),
        "cam_enhanced_cosine_mean": float(torch.nn.functional.cosine_similarity(cam.index_select(0, test_t), (cam + active_delta).index_select(0, test_t), dim=-1).mean()),
        "residual_parallel_fraction_mean": float((active_delta.index_select(0, test_t) * cam.index_select(0, test_t)).sum(-1).abs().div(active_delta.index_select(0, test_t).norm(dim=-1).mul(cam.index_select(0, test_t).norm(dim=-1)).clamp_min(1e-12)).mean()),
        "residual_orthogonal_fraction_mean": float((active_delta.index_select(0, test_t) - ((active_delta.index_select(0, test_t) * cam.index_select(0, test_t)).sum(-1, keepdim=True) / cam.index_select(0, test_t).square().sum(-1, keepdim=True).clamp_min(1e-12)) * cam.index_select(0, test_t)).norm(dim=-1).mean() / active_delta.index_select(0, test_t).norm(dim=-1).mean().clamp_min(1e-12)),
        "layernorm144_difference_mean": float((norm_after.index_select(0, test_t) - norm_before.index_select(0, test_t)).norm(dim=-1).mean()),
        "hidden_difference_mean": float((active_hidden.index_select(0, test_t) - base_hidden.index_select(0, test_t)).norm(dim=-1).mean()),
        "logit_margin_shift_mean": float((margin_after.index_select(0, test_t) - margin_before.index_select(0, test_t)).mean()),
        "relation_response_effective_rank": response_rank,
        "historical_A_metrics": {key: hist_metrics[key] for key in exp.METRICS},
        "current_epoch_B_metrics": {key: b_metrics[key] for key in exp.METRICS},
        "current_full_C_metrics": {key: c_metrics[key] for key in exp.METRICS},
        "A_to_B": changes["A_to_B"],
        "B_to_C": changes["B_to_C"],
        "A_to_C": changes["A_to_C"],
        "prompt_attention_mean_pairwise_mae": float(details["attention_prompt"].index_select(0, test_t).sub(details["attention_visual"].index_select(0, test_t)[:, None]).abs().mean()),
        "latest_rng_restored": True,
        "latest_current_metrics": latest_metrics,
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
        "A_to_B_corrected": changes["A_to_B"]["corrected"],
        "A_to_B_harmed": changes["A_to_B"]["harmed"],
        "A_to_B_net_corrections": changes["A_to_B"]["net_corrections"],
        "B_to_C_corrected": changes["B_to_C"]["corrected"],
        "B_to_C_harmed": changes["B_to_C"]["harmed"],
        "B_to_C_net_corrections": changes["B_to_C"]["net_corrections"],
        "A_to_C_corrected": changes["A_to_C"]["corrected"],
        "A_to_C_harmed": changes["A_to_C"]["harmed"],
        "A_to_C_net_corrections": changes["A_to_C"]["net_corrections"],
        "current_cam_norm_mean": summary["cam_norm_mean"],
        "residual_max_ratio": summary["residual_max_ratio"],
        "layernorm144_difference_mean": summary["layernorm144_difference_mean"],
        "hidden_difference_mean": summary["hidden_difference_mean"],
        "logit_margin_shift_mean": summary["logit_margin_shift_mean"],
        "relationship_response_effective_rank": summary["relation_response_effective_rank"],
    })
    return summary, mechanism_rows


def F_layer_norm(value, layer):
    return torch.nn.functional.layer_norm(value, layer.normalized_shape, layer.weight, layer.bias, layer.eps)


def _metric_payload_from_logp(logp, state):
    metrics, _ = exp._metric_payload(logp, state)
    return metrics


def _load_baseline_rows():
    darg = {}
    for row in _read_csv(BASE_E74 / "results/fold_metrics.csv"):
        if row["group"] == "E74.4-G2-Positive-DCR":
            darg[int(row["fold"])] = {name: float(row[name]) * 100.0 for name in exp.METRICS}
    old = {}
    for row in _read_csv(E8161 / "results/fold_metrics.csv"):
        if row["group"] == "E81.6.1-G1-Medical-CPRL-10":
            old[int(row["fold"])] = {name: float(row[name]) for name in exp.METRICS}
    if set(darg) != set(range(1, 11)) or set(old) != set(range(1, 11)):
        raise RuntimeError("historical DARG/E81.6.1 fold metrics are incomplete")
    return {"DARG": darg, "E81.6.1-G1": old}


def _paired_rows(current, references):
    rows = []
    comparisons = []
    for ref_name, ref in references.items():
        comparisons.append(("G1", ref_name, current[exp.GROUPS[0]], ref))
    comparisons.extend([
        ("G1", "G2", current[exp.GROUPS[0]], current[exp.GROUPS[1]]),
        ("G1", "G3", current[exp.GROUPS[0]], current[exp.GROUPS[2]]),
    ])
    for left_name, right_name, left, right in comparisons:
        for subset_name, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [f for f in range(1, 11) if f != 2])):
            for metric in exp.METRICS:
                diffs = np.asarray([float(left[f][metric]) - float(right[f][metric]) for f in folds], dtype=np.float64)
                mean = float(diffs.mean())
                sd = float(diffs.std(ddof=0))
                half = 1.96 * sd / math.sqrt(len(diffs))
                rows.append({
                    "comparison": f"{left_name}_vs_{right_name}",
                    "subset": subset_name,
                    "metric": metric,
                    "n": len(diffs),
                    "mean_paired_diff_percentage_points": mean,
                    "paired_sd_population": sd,
                    "descriptive_interval_low": mean - half,
                    "descriptive_interval_high": mean + half,
                    "wins": int((diffs > 1e-12).sum()),
                    "ties": int((np.abs(diffs) <= 1e-12).sum()),
                    "losses": int((diffs < -1e-12).sum()),
                    "fold_differences": json.dumps({str(f): float(left[f][metric]) - float(right[f][metric]) for f in folds}, separators=(",", ":")),
                })
    return rows


def _summary_rows(current, baseline):
    rows = []
    for group, fold_map in current.items():
        for metric in exp.METRICS:
            mean, sd = _mean_std([fold_map[f] for f in range(1, 11)], metric)
            nine_mean, nine_sd = _mean_std([fold_map[f] for f in range(1, 11) if f != 2], metric)
            rows.append({"model": group, "subset": "all10", "metric": metric, "mean": mean, "population_sd": sd})
            rows.append({"model": group, "subset": "exclude_fold2", "metric": metric, "mean": nine_mean, "population_sd": nine_sd})
    for name, folds in baseline.items():
        for metric in exp.METRICS:
            mean, sd = _mean_std([folds[f] for f in range(1, 11)], metric)
            nine_mean, nine_sd = _mean_std([folds[f] for f in range(1, 11) if f != 2], metric)
            rows.append({"model": name, "subset": "all10", "metric": metric, "mean": mean, "population_sd": sd})
            rows.append({"model": name, "subset": "exclude_fold2", "metric": metric, "mean": nine_mean, "population_sd": nine_sd})
    return rows


def _write_report(fold_rows, paired_rows, summary_rows, status):
    metric_names = {
        "acc": "ACC",
        "hard_auc": "Hard-AUC/BA",
        "prob_auc": "Prob-AUC",
        "sen": "SEN-ASD",
        "spe": "SPE-TD",
        "asd_f1": "ASD-F1",
    }
    models = [*exp.GROUPS, "DARG", "E81.6.1-G1"]
    lines = [
        "# Experiment 81.6.2 独立审计报告",
        "",
        f"任务状态：{status['completed_tasks']}/30；独立审计：{status['status']}。",
        "",
        "六指标均为百分比，显示10折均值±population SD；Hard-AUC等于历史定义的hard-prediction BA。配对区间为描述性均值±1.96×population SD/√n，不作显著性检验。",
        "",
        "|模型|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        row = {r["metric"]: r for r in summary_rows if r["model"] == model and r["subset"] == "all10"}
        if not row:
            continue
        values = [f"{row[m]['mean']:.4f}±{row[m]['population_sd']:.4f}" for m in exp.METRICS]
        lines.append(f"|{model}|" + "|".join(values) + "|")
    lines.extend(["", "## 去Fold2结果", "", "|模型|指标|9折均值±population SD|", "|---|---|---:|"])
    for model in models:
        for metric in exp.METRICS:
            row = next((r for r in summary_rows if r["model"] == model and r["subset"] == "exclude_fold2" and r["metric"] == metric), None)
            if row:
                lines.append(f"|{model}|{metric_names[metric]}|{row['mean']:.4f}±{row['population_sd']:.4f}|")
    lines.extend(["", "## 配对比较", "", "|比较|样本|指标|平均差值|描述性区间|W/T/L|", "|---|---|---|---:|---:|---:|"])
    for row in paired_rows:
        lines.append(
            f"|{row['comparison']}|{row['subset']}|{metric_names[row['metric']]}|"
            f"{row['mean_paired_diff_percentage_points']:+.4f}|"
            f"[{row['descriptive_interval_low']:+.4f}, {row['descriptive_interval_high']:+.4f}]|"
            f"{row['wins']}/{row['ties']}/{row['losses']}|"
        )
    lines.extend(["", "## 同epoch净纠错", "", "A为历史DARG Best，B为当前SA Best epoch关闭新增残差，C为完整SA-PRCL。纠正/误伤按测试患者汇总；Fold2历史重叠仍在。", "",
                  "|组|折集|阶段|纠正|误伤|净纠错|", "|---|---|---|---:|---:|---:|"])
    for group in exp.GROUPS:
        for subset in ("all10", "exclude_fold2"):
            for stage in ("A_to_B", "B_to_C", "A_to_C"):
                item = status["mechanism_correction_totals"][group][subset][stage]
                lines.append(f"|{group}|{subset}|{stage}|{item['corrected']}|{item['harmed']}|{item['net_corrections']}|")
    lines.extend([
        "",
        "## 审计与机制判定",
        "",
        f"- Best六指标独立复算：{status['best_recomputed']}/30。",
        f"- Latest完整恢复：{status['latest_restored']}/30；E61逐折起点：{status['historical_starts']}/10。",
        f"- 同折三组初始化身份完全一致：{status['same_fold_initializations']}/10；新增参数、DARG参数/优化器、warmup RNG、公共RNG、划分及训练关系对均逐项核对。",
        f"- G1/G2主干共有epoch参数、优化器与公共RNG不一致数：{status['protected_mismatch_count']}。",
        f"- Prompt-off/Null-only、SVD Gram、RES、残差约束及Fold2保护：{status['mechanism_invariants']}。",
        f"- 科研判定：`{status['research_decision']}`。",
        "",
        "G2使用原始医学文本，仅固定分组索引被打乱；Prompt-off会关闭整个新增分支，不能单独证明医学语义有效。Fold2历史训练/测试重叠、test=val、预CV预处理/RFE及transductive图均保留。G1/G3的目标函数和梯度路由不同。当前结果仍是单seed、历史协议下的候选证据，不代表独立临床泛化。",
        "",
        f"固定语义缓存SHA256：`{status['prompt_cache_sha256']}`；正式源码SHA256：`{status['source_sha256']}`。",
        "",
    ])
    path = RESULTS / "report.md"
    body = "\n".join(lines)
    temp = path.with_suffix(".tmp")
    temp.write_text(body, encoding="utf-8")
    with temp.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    temp.replace(path)
    if path.read_text(encoding="utf-8") != body:
        raise IOError("report readback failed")


def audit():
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_hash = exp.source_sha()
    if source_hash != preflight.get("source_sha256"):
        raise RuntimeError("source SHA changed since formal preflight")
    legacy = exp.legacy_runtime()
    audit_rng_before = legacy.pack_rng()
    invariant_recheck = exp._model_preflight()
    legacy.restore_rng(audit_rng_before)
    if invariant_recheck.get("status") != "PASS":
        raise RuntimeError("independent model/semantic/RES invariant recheck failed")
    task_rows, fold_rows, mechanism_rows, audit_rows = [], [], [], []
    current = {group: {} for group in exp.GROUPS}
    source_mismatch_count = 0
    for group in exp.GROUPS:
        for fold in range(1, 11):
            task_file, best_file, latest_file, log_file = exp._task_paths(group, fold)
            if not all(path.is_file() for path in (task_file, best_file, latest_file, log_file)):
                raise FileNotFoundError(f"incomplete formal artifacts: {group} fold={fold}")
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_hash:
                raise RuntimeError(f"task not complete or source mismatch: {group} fold={fold}")
            if task.get("best_checkpoint_sha256") != exp.sha_file(best_file) or task.get("latest_checkpoint_sha256") != exp.sha_file(latest_file):
                raise RuntimeError(f"checkpoint hash mismatch: {group} fold={fold}")
            if task.get("scalar_log_sha256") != exp.sha_file(log_file):
                raise RuntimeError(f"scalar log hash mismatch: {group} fold={fold}")
            summary, rows = _audit_fold(group, fold, task, best_file, latest_file)
            current[group][fold] = summary["metrics"]
            audit_rows.append(summary)
            mechanism_rows.extend(rows)
            fold_rows.append({"group": group, "fold": fold, **summary["metrics"], **{f"base_{k}": v for k, v in summary["base_only_metrics"].items()},
                              "best_epoch_1based": summary["best_epoch_1based"], "current_off_acc": summary["base_only_metrics"]["acc"],
                              "current_off_hard_auc": summary["base_only_metrics"]["hard_auc"], "checkpoint_sha256": task["best_checkpoint_sha256"],
                              "latest_sha256": task["latest_checkpoint_sha256"], "independent_checkpoint_audit": "PASS"})
            task_rows.append(task)
            print(f"AUDIT PASS {group} fold={fold} ACC={summary['metrics']['acc']:.4f}", flush=True)

    # The protected groups must have identical DARG state, optimizer and public RNG at every common epoch.
    same_fold_initializations = 0
    identity_fields = (
        "warmup_sha256",
        "warmup_rng_sha256",
        "public_rng_sha256",
        "base_model_sha256",
        "base_optimizer_sha256",
        "relation_init_sha256",
        "split_sha",
        "pair_sha",
        "membership",
    )
    for fold in range(1, 11):
        fold_tasks = [row for row in task_rows if int(row["fold"]) == fold]
        if len(fold_tasks) != 3:
            raise RuntimeError(f"expected all three formal groups for fold {fold}")
        identities = [row["identity"] for row in fold_tasks]
        for field in identity_fields:
            if any(identity.get(field) != identities[0].get(field) for identity in identities[1:]):
                raise RuntimeError(f"same-fold three-group initialization mismatch: fold={fold}, field={field}")
        same_fold_initializations += 1

    mismatches = []
    group1, group2 = exp.GROUPS[:2]
    for fold in range(1, 11):
        _, _, _, log1 = exp._task_paths(group1, fold)
        _, _, _, log2 = exp._task_paths(group2, fold)
        rows1 = {int(row["epoch"]): row for row in (json.loads(line) for line in log1.read_text(encoding="utf-8").splitlines())}
        rows2 = {int(row["epoch"]): row for row in (json.loads(line) for line in log2.read_text(encoding="utf-8").splitlines())}
        common = sorted(rows1.keys() & rows2.keys())
        for epoch in common:
            a, b = rows1[epoch]["rng"], rows2[epoch]["rng"]
            fields = ("base_model_sha256", "base_optimizer_sha256", "rng_before", "rng_after_train", "rng_after_eval")
            for field in fields:
                if a[field] != b[field]:
                    mismatches.append({"fold": fold, "epoch": epoch, "field": field, "g1": a[field], "g2": b[field]})
    if mismatches:
        raise RuntimeError(f"G1/G2 protected DARG trace mismatch: {mismatches[:5]}")

    baseline = _load_baseline_rows()
    summary_rows = _summary_rows(current, baseline)
    paired_rows = _paired_rows(current, baseline)
    exp.write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    exp.write_csv(RESULTS / "paired_comparisons.csv", paired_rows)
    exp.write_csv(RESULTS / "mechanism.csv", mechanism_rows)

    g1_darg_acc = next(row for row in paired_rows if row["comparison"] == "G1_vs_DARG" and row["subset"] == "all10" and row["metric"] == "acc")
    g1_darg_hard = next(row for row in paired_rows if row["comparison"] == "G1_vs_DARG" and row["subset"] == "all10" and row["metric"] == "hard_auc")
    g1_g2 = [row for row in paired_rows if row["comparison"] == "G1_vs_G2" and row["subset"] == "all10" and row["metric"] in ("acc", "hard_auc")]
    g1_g3_primary = [row for row in paired_rows if row["comparison"] == "G1_vs_G3" and row["subset"] == "all10" and row["metric"] in ("acc", "hard_auc")]
    failures = []
    if g1_darg_acc["mean_paired_diff_percentage_points"] <= 0 or g1_darg_hard["mean_paired_diff_percentage_points"] <= 0:
        failures.append("NO_GO_SA_PRCL_BASELINE")
    if any(row["mean_paired_diff_percentage_points"] <= 0 for row in g1_g2):
        failures.append("SEMANTIC_GROUPING_INCREMENT_NOT_ESTABLISHED")
    if any(row["mean_paired_diff_percentage_points"] <= 0 for row in g1_g3_primary):
        failures.append("BACKBONE_PROTECTION_INCREMENT_NOT_ESTABLISHED")
    if len(mismatches):
        failures.append("BACKBONE_PROTECTION_AUDIT_FAIL")
    decision = "PRELIMINARY_GO_SA_PRCL" if not failures else ";".join(failures)
    mechanism_correction_totals = {}
    for group in exp.GROUPS:
        mechanism_correction_totals[group] = {}
        for subset, folds in (("all10", set(range(1, 11))), ("exclude_fold2", set(range(1, 11)) - {2})):
            mechanism_correction_totals[group][subset] = {}
            selected = [row for row in audit_rows if row["group"] == group and row["fold"] in folds]
            for stage in ("A_to_B", "B_to_C", "A_to_C"):
                corrected = sum(row[stage]["corrected"] for row in selected)
                harmed = sum(row[stage]["harmed"] for row in selected)
                mechanism_correction_totals[group][subset][stage] = {
                    "corrected": corrected,
                    "harmed": harmed,
                    "net_corrections": corrected - harmed,
                }
    status = {
        "status": "PASS",
        "completed_tasks": len(task_rows),
        "best_recomputed": len(fold_rows),
        "latest_restored": len(fold_rows),
        "historical_starts": len({row["fold"] for row in fold_rows}),
        "same_fold_initializations": same_fold_initializations,
        "same_fold_initialization_fields": list(identity_fields),
        "protected_mismatch_count": len(mismatches),
        "protected_trace_common_epoch_count": sum(
            len({int(row["epoch"]) for row in (json.loads(line) for line in exp._task_paths(group1, fold)[3].read_text().splitlines())}
                & {int(row["epoch"]) for row in (json.loads(line) for line in exp._task_paths(group2, fold)[3].read_text().splitlines())})
            for fold in range(1, 11)
        ),
        "mechanism_invariants": "PASS",
        "independent_model_invariant_recheck": invariant_recheck,
        "source_sha256": source_hash,
        "prompt_cache_sha256": exp.sha_file(HERE / "prompt_features.pt"),
        "research_decision": decision,
        "decision_reasons": failures,
        "mechanism_correction_totals": mechanism_correction_totals,
        "fold_mechanism_summary": audit_rows,
        "pairwise_trace_mismatches": mismatches,
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    exp.atomic_json(RESULTS / "audit_status.json", status)
    _write_report(fold_rows, paired_rows, summary_rows, status)
    exp.atomic_json(HERE / "manifest.json", {
        "status": "COMPLETE_AUDITED",
        "completed_tasks": 30,
        "independent_audits": 30,
        "audit_status": "PASS",
        "research_decision": decision,
        "source_sha256": source_hash,
        "audit_status_sha256": exp.sha_file(RESULTS / "audit_status.json"),
        "fold_metrics_sha256": exp.sha_file(RESULTS / "fold_metrics.csv"),
        "paired_comparisons_sha256": exp.sha_file(RESULTS / "paired_comparisons.csv"),
        "mechanism_sha256": exp.sha_file(RESULTS / "mechanism.csv"),
        "report_sha256": exp.sha_file(RESULTS / "report.md"),
    })
    return status


if __name__ == "__main__":
    try:
        audit()
    except Exception as error:
        exp.atomic_json(RESULTS / "audit_status.json", {"status": "FAIL", "error": repr(error), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise
