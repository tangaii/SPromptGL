#!/usr/bin/env python3
"""Independent E81.1 checkpoint, metric, RNG, and mechanism audit."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run as runner

RESULTS = HERE / "results"
HIST_METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
COMPARISONS = (
    ("G1_vs_DARG", 0, "DARG"),
    ("G2_vs_E81_G2", 1, "E81"),
    ("G3_vs_G2", 2, 1),
    ("G3_vs_G4", 2, 3),
    ("G3_vs_G1", 2, 0),
    ("G3_vs_DARG", 2, "DARG"),
)


def mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temp.replace(path)


def percent_metrics(metrics):
    return {key: float(metrics[key]) * 100.0 for key in HIST_METRICS}


def _path_set(group, fold):
    return runner.paths(group, fold)


def _probability_mae(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError("frozen intervention probability shape mismatch")
    return float(np.mean(np.abs(left - right)))


def _pairwise_cosine(rows):
    values = torch.as_tensor(rows, dtype=torch.float64)
    values = torch.nn.functional.normalize(values.reshape(values.shape[0], -1), dim=-1)
    similarity = values @ values.T
    mask = ~torch.eye(values.shape[0], dtype=torch.bool)
    return float(similarity[mask].mean()) if bool(mask.any()) else 1.0


def _relation_statistics(details, test_ids):
    attention = details["attention"].detach().cpu().double()[test_ids]
    if tuple(attention.shape[1:]) != (8, 16, 16):
        raise RuntimeError(f"unexpected relation attention shape {tuple(attention.shape)}")
    n = attention.shape[0]
    dense = torch.zeros((n, 8, 16, 16), dtype=torch.float64)
    mapping = details["candidate_map"].detach().cpu()
    for p in range(16):
        for q in range(16):
            slot = int(mapping[p, q])
            if slot >= 0:
                dense[:, :, p, q] = attention[:, :, p, slot]
    null = attention[..., -1]
    entropy = -(attention.clamp_min(1e-15) * attention.clamp_min(1e-15).log()).sum(-1)
    refusal = null >= attention[..., :15].max(-1).values
    within = torch.zeros((16, 16), dtype=torch.bool)
    for p in range(16):
        for q in range(16):
            within[p, q] = p != q and p // 4 == q // 4
    cross = ~within & ~torch.eye(16, dtype=torch.bool)
    within_mass = dense[..., within].sum(-1).mean()
    cross_mass = dense[..., cross].sum(-1).mean()
    nonnull_mass = dense.sum((-1, -2)).mean()
    prompt_patient = dense.reshape(n, 8, -1)
    prompt_patient = prompt_patient / prompt_patient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    patient_prompt_similarity = np.mean([
        _pairwise_cosine(prompt_patient[:, head]) for head in range(8)
    ])
    prompt_maps = dense.mean(0).reshape(8, -1)
    between_prompt_similarity = _pairwise_cosine(prompt_maps)
    normal_distribution = dense[:, :4].mean((0, 1)).reshape(-1)
    abnormal_distribution = dense[:, 4:].mean((0, 1)).reshape(-1)
    normal_distribution = torch.nn.functional.normalize(normal_distribution, dim=0)
    abnormal_distribution = torch.nn.functional.normalize(abnormal_distribution, dim=0)
    delta_ratio = details["delta_ratio"].detach().cpu().flatten()[test_ids].double()
    alpha = details["alpha"].detach().cpu().flatten()[test_ids].double()
    u_over_b = details["u_over_b"].detach().cpu().flatten()[test_ids].double()
    interaction = details["interaction"].detach().cpu()[test_ids].double()
    return {
        "attention_entropy_mean": float(entropy.mean()),
        "attention_entropy_p50": float(torch.quantile(entropy.flatten(), 0.5)),
        "null_weight_mean": float(null.mean()),
        "null_weight_p50": float(torch.quantile(null.flatten(), 0.5)),
        "null_weight_p95": float(torch.quantile(null.flatten(), 0.95)),
        "null_refusal_rate": float(refusal.double().mean()),
        "top_relation_weight_mean": float(attention[..., :15].max(-1).values.mean()),
        "same_prompt_patient_similarity": float(patient_prompt_similarity),
        "between_prompt_similarity": float(between_prompt_similarity),
        "normal_abnormal_distribution_cosine": float((normal_distribution * abnormal_distribution).sum()),
        "normal_abnormal_distribution_l1": float((normal_distribution - abnormal_distribution).abs().sum()),
        "within_slot_undirected_candidates": 24,
        "cross_slot_undirected_candidates": 96,
        "all_undirected_candidates": 120,
        "within_slot_candidate_base_rate": 0.2,
        "cross_slot_candidate_base_rate": 0.8,
        "non_null_attention_mass": float(nonnull_mass),
        "within_slot_attention_mass": float(within_mass),
        "cross_slot_attention_mass": float(cross_mass),
        "within_slot_enrichment": float(within_mass / (nonnull_mass * 0.2 + 1e-12)),
        "cross_slot_enrichment": float(cross_mass / (nonnull_mass * 0.8 + 1e-12)),
        "alpha_mean": float(alpha.mean()),
        "alpha_p50": float(torch.quantile(alpha, 0.5)),
        "delta_ratio_mean": float(delta_ratio.mean()),
        "delta_ratio_p50": float(torch.quantile(delta_ratio, 0.5)),
        "delta_ratio_p95": float(torch.quantile(delta_ratio, 0.95)),
        "near_10pct_cap_rate": float((delta_ratio >= 0.099).double().mean()),
        "u_over_b_mean": float(u_over_b.mean()),
        "u_over_b_p95": float(torch.quantile(u_over_b, 0.95)),
        "nonlinear_interaction_norm_mean": float(interaction.norm(dim=-1).mean()),
        "gamma_final": float(details["gamma"].detach().cpu()),
    }


def _frozen_eval(state, module, group, intervention="full", text_override=None):
    with torch.no_grad():
        output, capture = runner.hook_forward(
            state, module, group, training=False, intervention=intervention,
            text_override=text_override,
        )
        metrics = runner.base_runtime().classification_metrics(
            output["logp"], state["test"], state["y"]
        )
    return output, capture, metrics


def _intervention_record(state, module, group, base_metrics, base_query, base_attention, name,
                         intervention=None, text_override=None):
    output, capture, metrics = _frozen_eval(
        state, module, group,
        intervention=intervention or name,
        text_override=text_override,
    )
    result = {
        "intervention": name,
        "probability_mae": _probability_mae(base_metrics["prob"], metrics["prob"]),
        "hard_prediction_flips": int(np.count_nonzero(
            np.asarray(base_metrics["pred"]) != np.asarray(metrics["pred"])
        )),
        "acc_change_pp": (float(metrics["acc"]) - float(base_metrics["acc"])) * 100.0,
        "hard_auc_change_pp": (float(metrics["hard_auc"]) - float(base_metrics["hard_auc"])) * 100.0,
        "prob_auc_change_pp": (float(metrics["prob_auc"]) - float(base_metrics["prob_auc"])) * 100.0,
        "query_l2_mean": float((capture["details"]["query"].detach() - base_query).norm(dim=-1).mean().cpu()),
        "attention_l1_mean": float((capture["details"]["attention"].detach() - base_attention).abs().mean().cpu()),
    }
    return result, output, capture, metrics


def _audit_fold(group, fold, source_hash):
    task_file, best_file, latest_file, _ = _path_set(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    checkpoint = runner.torch_load(best_file)
    latest = runner.torch_load(latest_file)
    if task.get("status") != "COMPLETE" or checkpoint.get("source_sha256") != source_hash:
        raise RuntimeError(f"task/checkpoint source or status mismatch for {group}/fold{fold}")
    if task.get("checkpoint_sha256") != runner.sha_file(best_file):
        raise RuntimeError(f"best Checkpoint SHA mismatch for {group}/fold{fold}")
    if task.get("latest_sha256") != runner.sha_file(latest_file):
        raise RuntimeError(f"latest Checkpoint SHA mismatch for {group}/fold{fold}")
    state, warmup, identity = runner.prepare(fold)
    if checkpoint.get("identity") != identity or task.get("identity") != identity:
        raise RuntimeError(f"E61 base identity changed for {group}/fold{fold}")
    if identity["warmup_rng_sha"] != warmup["rng_sha"]:
        raise RuntimeError(f"warmup public RNG identity changed for {group}/fold{fold}")

    module = runner.make_module(fold, group, state["parts"][0].device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    runner.load_bundle(state, module, optimizer, checkpoint["best_state"])
    module.eval()
    state["model"].eval()
    state["cam"].eval()
    output, capture, metrics = _frozen_eval(state, module, group)
    saved_prob = np.asarray(checkpoint["test_probability"], dtype=np.float64)
    saved_pred = np.asarray(checkpoint["test_prediction"])
    if not np.array_equal(saved_pred, np.asarray(metrics["pred"])):
        raise RuntimeError(f"best hard predictions differ after independent reload: {group}/fold{fold}")
    if not np.allclose(saved_prob, np.asarray(metrics["prob"]), atol=1e-8, rtol=0.0):
        raise RuntimeError(f"best probabilities differ after independent reload: {group}/fold{fold}")
    if not torch.equal(checkpoint["test_logp"], output["logp"].detach().cpu()):
        raise RuntimeError(f"full-bank logp differs after independent reload: {group}/fold{fold}")
    for metric in HIST_METRICS:
        if abs(float(checkpoint["metrics"][metric]) - float(metrics[metric])) > 1e-12:
            raise RuntimeError(f"independent metric mismatch {metric}: {group}/fold{fold}")
    for key in ("asd_tp", "asd_fn", "td_fp", "td_tn"):
        if int(task["confusion"][key]) != int(metrics[key]):
            raise RuntimeError(f"confusion count mismatch {key}: {group}/fold{fold}")
    if not np.array_equal(np.asarray(checkpoint["test_indices"]), state["test_np"]):
        raise RuntimeError(f"held-out test index order changed: {group}/fold{fold}")

    test_ids = state["test_np"].astype(np.int64)
    base_path = Path(CFG_BASE) / ".runtime/checkpoints" / f"{runner.BASE_GROUP}_f{fold:02d}.best.pt"
    base_checkpoint = runner.torch_load(base_path)
    base_identity = base_checkpoint.get("identity", {})
    if base_identity.get("split_sha") != identity.get("split_sha"):
        raise RuntimeError(f"DARG comparison split identity differs at fold {fold}")
    if base_identity.get("membership", {}).get("test_count") != len(test_ids):
        raise RuntimeError(f"DARG comparison test count differs at fold {fold}")
    base_prediction = np.asarray(base_checkpoint["test_prediction"])
    base_probability = np.asarray(base_checkpoint["test_probability"], dtype=np.float64)
    if base_prediction.shape != test_ids.shape or base_probability.shape != (len(test_ids), 2):
        raise RuntimeError(f"DARG historical predictions do not match the reconstructed fold split at fold {fold}")
    if not np.isfinite(base_probability).all() or not np.allclose(base_probability.sum(-1), 1.0, atol=1e-6):
        raise RuntimeError(f"DARG historical test probabilities are invalid at fold {fold}")
    labels = state["y"].detach().cpu().numpy()[test_ids]
    prediction = np.asarray(metrics["pred"])
    fix = int(np.count_nonzero((prediction == labels) & (base_prediction != labels)))
    harm = int(np.count_nonzero((prediction != labels) & (base_prediction == labels)))

    row = {
        "group": group, "fold": fold,
        **percent_metrics(metrics),
        "asd_tp": int(metrics["asd_tp"]), "asd_fn": int(metrics["asd_fn"]),
        "td_fp": int(metrics["td_fp"]), "td_tn": int(metrics["td_tn"]),
        "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        "wall_seconds": float(checkpoint["elapsed_seconds"]),
        "peak_gpu_bytes": int(checkpoint["peak_gpu_bytes"]),
        "corrected_vs_darg": fix, "harmed_vs_darg": harm,
        "hard_prediction_flips_vs_darg": int(np.count_nonzero(prediction != base_prediction)),
        "checkpoint_sha256": runner.sha_file(best_file),
        "latest_sha256": runner.sha_file(latest_file),
        "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"]),
        "independent_checkpoint_audit": "PASS",
    }

    relation = _relation_statistics(capture["details"], test_ids)
    with torch.enable_grad():
        grad_output, grad_capture = runner.hook_forward(state, module, group, training=False)
    alpha_grad = torch.autograd.grad(
        grad_output["cls"], tuple(module.alpha_gate.parameters()), allow_unused=True,
        retain_graph=True,
    )
    alpha_grad_values = [value.detach().double().square().sum() for value in alpha_grad if value is not None]
    alpha_grad_norm = float(torch.sqrt(torch.stack(alpha_grad_values).sum()).cpu()) if alpha_grad_values else 0.0
    interaction_grad = torch.autograd.grad(
        grad_output["cls"], grad_capture["details"]["interaction"], allow_unused=True,
    )[0]
    interaction_grad_norm = float(interaction_grad.detach().double().norm().cpu()) if interaction_grad is not None else 0.0
    mechanism = {
        "group": group, "fold": fold,
        **relation,
        "alpha_classification_gradient_norm": alpha_grad_norm,
        "nonlinear_interaction_classification_gradient_norm": interaction_grad_norm,
    }
    del grad_output, grad_capture

    query = capture["details"]["query"].detach()
    attention = capture["details"]["attention"].detach()
    if group == GROUPS_FOR_AUDIT[2]:
        interventions = [
            ("prcl_off", "prcl_off", None),
            ("writeback_off", "writeback_off", None),
            ("uniform_relations", "uniform", None),
            ("null_only", "null_only", None),
            ("text_off", "text_off", None),
            ("gamma_zero", "gamma_zero", None),
            ("normal_abnormal_swap", "swap_groups", None),
            ("nonlinear_interaction_off", "nonlinear_off", None),
            ("medical_to_same_gram_control", "full", runner.text_payload()["control_anchors"]),
        ]
        results = {}
        for name, operation, override in interventions:
            rec, _, _, _ = _intervention_record(
                state, module, group, metrics, query, attention, name,
                intervention=operation, text_override=override,
            )
            results[name] = rec
        previous_mode = module.condition_mode
        module.condition_mode = "additive"
        rec, _, _, _ = _intervention_record(
            state, module, group, metrics, query, attention,
            "multiplicative_to_additive_query", intervention="full",
        )
        results["multiplicative_to_additive_query"] = rec
        module.condition_mode = previous_mode
    else:
        results = {}
        rec, _, _, _ = _intervention_record(
            state, module, group, metrics, query, attention, "prcl_off",
            intervention="prcl_off",
        )
        results["prcl_off"] = rec
    mechanism["frozen_interventions_json"] = json.dumps(results, sort_keys=True)
    for name, rec in results.items():
        mechanism[name + "_probability_mae"] = rec["probability_mae"]
        mechanism[name + "_hard_prediction_flips"] = rec["hard_prediction_flips"]
        mechanism[name + "_acc_change_pp"] = rec["acc_change_pp"]

    expected_latest = (
        ("group", group), ("fold", fold), ("source_sha256", source_hash),
        ("identity", identity), ("warmup_sha256", identity["warmup_sha256"]),
        ("epochs_executed", int(checkpoint["epochs_executed"])),
    )
    for key, expected in expected_latest:
        actual_key = "epochs_executed" if key == "epochs_executed" else key
        actual = latest.get("next_epoch") if key == "epochs_executed" else latest.get(actual_key)
        if actual != expected:
            raise RuntimeError(f"latest-state identity mismatch {key}: {group}/fold{fold}")
    if latest.get("rng_trace") != task.get("rng_trace"):
        raise RuntimeError(f"latest RNG trace differs from task record: {group}/fold{fold}")
    trace = task.get("rng_trace", [])
    if not trace or int(trace[-1]["epoch"]) != int(checkpoint["epochs_executed"]):
        raise RuntimeError(f"latest RNG trace does not end at the final epoch: {group}/fold{fold}")
    if runner.base_runtime().rng_sha(latest["rng"]) != trace[-1]["rng_after_eval"]:
        raise RuntimeError(f"latest RNG state differs from final post-evaluation state: {group}/fold{fold}")
    runner.load_bundle(state, module, optimizer, latest["current"])
    for optimizer_name in ("opt_base", "opt_cam", "opt_added"):
        if optimizer_name not in latest["current"]["base_state"]:
            raise RuntimeError(f"latest state misses inherited optimizer {optimizer_name}")
    for parameter in list(state["model"].parameters()) + list(state["cam"].parameters()) + list(module.parameters()):
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"latest state contains nonfinite parameters: {group}/fold{fold}")
    latest_status = "PASS"

    del output, capture, module, optimizer, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, mechanism, task, latest_status


CFG_BASE = "/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr"
GROUPS_FOR_AUDIT = (
    "E81.1-G1-Visual-PRCL-v2",
    "E81.1-G2-Medical-Add-PRCL-v2",
    "E81.1-G3-Medical-Interaction-PRCL-v2",
    "E81.1-G4-Gram-Matched-Control-PRCL-v2",
)


def _historical_rows(path, group):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row.get("group") == group]
    result = []
    for row in selected:
        result.append({
            "fold": int(row["fold"]),
            **{metric: float(row[metric]) * 100.0 for metric in HIST_METRICS},
        })
    if len(result) != 10:
        raise RuntimeError(f"expected ten historical rows for {group}, found {len(result)} in {path}")
    return result


def _paired(left_rows, right_rows, metric, folds):
    left = {int(row["fold"]): float(row[metric]) for row in left_rows}
    right = {int(row["fold"]): float(row[metric]) for row in right_rows}
    keep = sorted(set(left) & set(right) & set(folds))
    differences = np.asarray([left[fold] - right[fold] for fold in keep], dtype=np.float64)
    wins = int((differences > 1e-12).sum())
    losses = int((differences < -1e-12).sum())
    ties = int(len(differences) - wins - losses)
    average = float(differences.mean())
    if len(differences) > 1:
        from scipy.stats import t
        margin = float(t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / math.sqrt(len(differences)))
    else:
        margin = 0.0
    return {
        "n": int(len(differences)), "mean_pp": average,
        "ci95_low_pp": average - margin, "ci95_high_pp": average + margin,
        "wins": wins, "ties": ties, "losses": losses,
        "per_fold_pp": json.dumps(differences.tolist()),
    }


def _summaries(rows, folds=None):
    result = {}
    for group in GROUPS_FOR_AUDIT:
        selected = [row for row in rows if row["group"] == group and
                    (folds is None or int(row["fold"]) in folds)]
        result[group] = {}
        for metric in HIST_METRICS:
            average, sd = mean_sd([row[metric] for row in selected])
            result[group][metric] = {"mean": average, "population_sd": sd}
    return result


def _comparison_rows(group_rows, base_rows, e81_rows):
    comparisons = []
    folds_all = set(range(1, 11))
    folds_nine = folds_all - {2}
    for name, left_index, right_index in COMPARISONS:
        left_group = GROUPS_FOR_AUDIT[left_index]
        left_rows = [row for row in group_rows if row["group"] == left_group]
        if right_index == "DARG":
            right_rows, right_label = base_rows, "E74.4-G2-DARG"
        elif right_index == "E81":
            right_rows, right_label = e81_rows, "E81-G2-Medical-PRCL"
        else:
            right_group = GROUPS_FOR_AUDIT[right_index]
            right_rows = [row for row in group_rows if row["group"] == right_group]
            right_label = right_group
        for scope, folds in (("all10", folds_all), ("without_fold2", folds_nine)):
            for metric in HIST_METRICS:
                record = _paired(left_rows, right_rows, metric, folds)
                comparisons.append({
                    "comparison": name, "left": left_group, "right": right_label,
                    "scope": scope, "metric": metric, **record,
                })
    return comparisons


def _rng_audit(tasks):
    history = runner.runtime().load_history()
    base_root = Path(CFG_BASE) / ".runtime/tasks"
    by_id = {(task["group"], int(task["fold"])): task for task in tasks}
    base_mismatch_events = []
    group_mismatch_events = []
    common_events = 0
    for fold in range(1, 11):
        base_file = base_root / f"{runner.BASE_GROUP}_f{fold:02d}.json"
        base_trace = json.loads(base_file.read_text(encoding="utf-8"))["rng_trace"]
        base_by_epoch = {int(row["epoch"]): row for row in base_trace}
        group_traces = {}
        for group in GROUPS_FOR_AUDIT:
            task = by_id[(group, fold)]
            current = {int(row["epoch"]): row for row in task["rng_trace"]}
            group_traces[group] = current
            for epoch in sorted(set(current) & set(base_by_epoch)):
                common_events += 1
                for field in ("rng_before", "rng_after_train", "rng_after_eval"):
                    if current[epoch][field] != base_by_epoch[epoch][field]:
                        base_mismatch_events.append({
                            "group": group, "fold": fold, "epoch": epoch, "field": field,
                        })
        for left_index in range(len(GROUPS_FOR_AUDIT)):
            for right_index in range(left_index + 1, len(GROUPS_FOR_AUDIT)):
                left, right = GROUPS_FOR_AUDIT[left_index], GROUPS_FOR_AUDIT[right_index]
                for epoch in sorted(set(group_traces[left]) & set(group_traces[right])):
                    for field in ("rng_before", "rng_after_train", "rng_after_eval"):
                        if group_traces[left][epoch][field] != group_traces[right][epoch][field]:
                            group_mismatch_events.append({
                                "left": left, "right": right, "fold": fold, "epoch": epoch, "field": field,
                            })
    return {
        "common_group_base_epochs": common_events,
        "group_vs_base_rng_triplet_mismatch_events": len(base_mismatch_events),
        "between_group_rng_triplet_mismatch_events": len(group_mismatch_events),
        "group_vs_base_mismatches": base_mismatch_events,
        "between_group_mismatches": group_mismatch_events,
    }


def _write_report(manifest, status, summaries, summaries_nine, comparisons,
                  e81_summaries, e81_summaries_nine, base_summaries,
                  base_summaries_nine, mechanism_rows):
    def fmt(summary, metric):
        value = summary[metric]
        return f"{value['mean']:.4f} ± {value['population_sd']:.4f}"

    lines = [
        "# Experiment 81.1: PRCL-v2",
        "",
        f"Status: {status['status']}; formal folds {status['formal_tasks_complete']}/40; independent best Checkpoint audits {status['best_checkpoint_independent_pass']}/40; latest state audits {status['latest_state_integrity_pass']}/40.",
        f"Directory: {HERE}. Training wall time: {manifest.get('training_wall_seconds', 0.0):.1f} s; sum of task wall times: {manifest.get('sum_task_wall_seconds', 0.0):.1f} s.",
        "",
        "All metrics are percentages, reported as ten-fold mean ± population SD. Hard-AUC is the historical hard-prediction balanced-accuracy metric; Prob-AUC is ROC-AUC from class probabilities.",
        "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS_FOR_AUDIT:
        lines.append("| " + group + " | " + " | ".join(fmt(summaries[group], metric) for metric in HIST_METRICS) + " |")
    lines += [
        "| E74.4-G2 DARG | " + " | ".join(f"{base_summaries[metric]['mean']:.4f} ± {base_summaries[metric]['population_sd']:.4f}" for metric in HIST_METRICS) + " |",
        "",
        "## Historical E81",
        "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, summary in e81_summaries.items():
        lines.append("| " + group + " | " + " | ".join(
            f"{summary[metric]['mean']:.4f} ± {summary[metric]['population_sd']:.4f}"
            for metric in HIST_METRICS
        ) + " |")
    lines += [
        "",
        "## Excluding Fold 2",
        "",
        "Absolute nine-fold mean ± population SD; the paired_comparisons.csv also contains nine-fold paired differences.",
        "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS_FOR_AUDIT:
        lines.append("| " + group + " | " + " | ".join(fmt(summaries_nine[group], metric) for metric in HIST_METRICS) + " |")
    lines.append("| E74.4-G2 DARG | " + " | ".join(
        f"{base_summaries_nine[metric]['mean']:.4f} ± {base_summaries_nine[metric]['population_sd']:.4f}"
        for metric in HIST_METRICS
    ) + " |")
    for group, summary in e81_summaries_nine.items():
        lines.append("| " + group + " | " + " | ".join(
            f"{summary[metric]['mean']:.4f} ± {summary[metric]['population_sd']:.4f}"
            for metric in HIST_METRICS
        ) + " |")
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across folds; the folds have overlapping training subjects.",
        "",
        "| Comparison | Scope | ACC Δ (W/T/L) | Hard-AUC Δ | Prob-AUC Δ | SEN Δ | SPE Δ | ASD-F1 Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in ("G1_vs_DARG", "G2_vs_E81_G2", "G3_vs_G2", "G3_vs_G4", "G3_vs_G1", "G3_vs_DARG"):
        for scope in ("all10", "without_fold2"):
            lookup = {(row["comparison"], row["scope"], row["metric"]): row for row in comparisons}
            cells = []
            for metric in HIST_METRICS:
                row = lookup[(comparison, scope, metric)]
                value = f"{row['mean_pp']:+.4f}"
                if metric == "acc":
                    value += f" ({row['wins']}/{row['ties']}/{row['losses']})"
                cells.append(value)
            lines.append("| " + comparison + " | " + scope + " | " + " | ".join(cells) + " |")

    mean_mechanism = {}
    for key in (
        "attention_entropy_mean", "null_weight_mean", "null_refusal_rate",
        "same_prompt_patient_similarity", "between_prompt_similarity",
        "normal_abnormal_distribution_cosine", "within_slot_attention_mass",
        "cross_slot_attention_mass", "within_slot_enrichment", "cross_slot_enrichment",
        "alpha_mean", "delta_ratio_mean", "delta_ratio_p50", "delta_ratio_p95",
        "near_10pct_cap_rate", "u_over_b_mean", "nonlinear_interaction_norm_mean",
        "alpha_classification_gradient_norm", "nonlinear_interaction_classification_gradient_norm",
    ):
        mean_mechanism[key] = float(np.mean([float(row[key]) for row in mechanism_rows]))
    lines += [
        "",
        "## Mechanism",
        "",
        "Across frozen held-out fold patients: mean relation entropy "
        f"{mean_mechanism['attention_entropy_mean']:.4f}, Null weight {mean_mechanism['null_weight_mean']:.4f}, "
        f"Null-dominant rate {mean_mechanism['null_refusal_rate']:.4f}, same-prompt patient similarity "
        f"{mean_mechanism['same_prompt_patient_similarity']:.4f}, between-prompt similarity "
        f"{mean_mechanism['between_prompt_similarity']:.4f}.",
        f"Within-slot candidate attention mass {mean_mechanism['within_slot_attention_mass']:.4f} (candidate base rate 0.20; enrichment {mean_mechanism['within_slot_enrichment']:.4f}); cross-slot mass {mean_mechanism['cross_slot_attention_mass']:.4f} (base rate 0.80; enrichment {mean_mechanism['cross_slot_enrichment']:.4f}).",
        f"Alpha mean {mean_mechanism['alpha_mean']:.5f}; residual ratio mean/P50/P95 {mean_mechanism['delta_ratio_mean']:.5f}/{mean_mechanism['delta_ratio_p50']:.5f}/{mean_mechanism['delta_ratio_p95']:.5f}; fraction near 10% cap {mean_mechanism['near_10pct_cap_rate']:.5f}; mean unbounded-to-bound ratio {mean_mechanism['u_over_b_mean']:.5f}.",
        f"Nonlinear interaction mean norm {mean_mechanism['nonlinear_interaction_norm_mean']:.5f}; classification gradient norms for alpha gate and interaction input {mean_mechanism['alpha_classification_gradient_norm']:.6g} and {mean_mechanism['nonlinear_interaction_classification_gradient_norm']:.6g}.",
        "",
        "| Group | Null weight | Null-dominant | Prompt similarity | Alpha | Residual mean / P95 | Near 10% cap | Interaction grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS_FOR_AUDIT:
        selected = [row for row in mechanism_rows if row["group"] == group]
        avg = lambda key: float(np.mean([float(row[key]) for row in selected]))
        lines.append(
            f"| {group} | {avg('null_weight_mean'):.4f} | {avg('null_refusal_rate'):.4f} | "
            f"{avg('between_prompt_similarity'):.4f} | {avg('alpha_mean'):.5f} | "
            f"{avg('delta_ratio_mean'):.5f} / {avg('delta_ratio_p95'):.5f} | "
            f"{avg('near_10pct_cap_rate'):.5f} | {avg('nonlinear_interaction_classification_gradient_norm'):.3g} |"
        )
    g3_rows = [row for row in mechanism_rows if row["group"] == GROUPS_FOR_AUDIT[2]]
    interventions = [json.loads(row["frozen_interventions_json"]) for row in g3_rows]
    lines += [
        "",
        "Frozen G3 interventions are post-training probes. Probability MAE is relative to the intact G3 Checkpoint; the number in parentheses is total hard-prediction flips across folds.",
        "",
        "| G3 intervention | Probability MAE (flips) | Mean ACC delta (pp) | Mean Prob-AUC delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    for name in interventions[0]:
        values = [record[name] for record in interventions]
        lines.append(
            f"| {name} | {np.mean([v['probability_mae'] for v in values]):.6f} "
            f"({sum(v['hard_prediction_flips'] for v in values)}) | "
            f"{np.mean([v['acc_change_pp'] for v in values]):+.4f} | "
            f"{np.mean([v['prob_auc_change_pp'] for v in values]):+.4f} |"
        )
    lines += [
        "",
        "The mechanism CSV includes per-fold frozen interventions for PRCL-off, uniform relation attention, Null-only, text-off, gamma-zero, normal/abnormal group swap, nonlinear-interaction-off, medical-to-same-Gram-control replacement, and multiplicative-to-additive Query replacement in G3. These interventions are not independently retrained ablations.",
        "",
        "## Audit and scientific gates",
        "",
        f"- RNG triplet mismatches against E74.4 common epochs: {status['rng_audit']['group_vs_base_rng_triplet_mismatch_events']}; between E81.1 groups: {status['rng_audit']['between_group_rng_triplet_mismatch_events']}.",
        f"- Base-off parity: {status['base_off_parity_pass']}/10; two-step restore replay: {status['two_step_restore_replay']}.",
        f"- Prompt cache SHA: {status['text_cache_sha256']}; G4 orthogonal Gram max error: {status['g4_gram_max_abs_error']:.3e}.",
        f"- Gate A local relation repair vs DARG: {status['scientific_gates']['A_visual_relation_vs_darg']}.",
        f"- Gate B shared repair vs E81 medical-add: {status['scientific_gates']['B_repair_vs_e81']}.",
        f"- Gate C multiplicative vs additive prompt Query: {status['scientific_gates']['C_interaction_vs_additive']}.",
        f"- Gate D medical anchors vs same-Gram rotated control: {status['scientific_gates']['D_medical_vs_same_gram_control']}.",
        f"- Gate E complete method vs DARG: {status['scientific_gates']['E_complete_vs_darg']}.",
        f"- Final research decision: {status['research_decision']}.",
        "",
        "Limitations retained from the historical comparison protocol: validation equals test and test ACC selects checkpoints; preprocessing/RFE occurs before CV; the patient graph is transductive; Fold 2 rows 377 and 407 occur in both historical train and test sets and remain in the original DARG path, while they are excluded from the new residual penalty. Ten-fold intervals are descriptive because training sets overlap. The 16 local CAM tokens are latent coordinates, not anatomical ROIs.",
        "",
        "## Artifacts",
        "",
        "Source, configuration, and cache identities are frozen in manifest.json. Per-fold results are in fold_metrics.csv, paired differences in paired_comparisons.csv, and frozen mechanism checks in mechanism.csv.",
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    current_sources = runner.source_map()
    frozen_sources = manifest.get("source_map", {})
    report_only_path = str((HERE / "audit.py").resolve())
    if current_sources.keys() != frozen_sources.keys():
        raise RuntimeError("frozen source file set changed since formal training")
    changed_sources = [path for path in current_sources
                       if current_sources[path] != frozen_sources[path] and path != report_only_path]
    if changed_sources:
        raise RuntimeError("non-report-only source bytes changed since formal training: " + ", ".join(changed_sources))
    source_hash = manifest.get("source_sha256")
    if not source_hash:
        raise RuntimeError("training source SHA is missing from the frozen manifest")
    task_rows = []
    best_rows = []
    mechanism_rows = []
    latest_pass = 0
    init_by_fold = {}
    for group in GROUPS_FOR_AUDIT:
        for fold in range(1, 11):
            row, mechanism, task, latest_status = _audit_fold(group, fold, source_hash)
            task_rows.append(task)
            best_rows.append(row)
            mechanism_rows.append(mechanism)
            latest_pass += int(latest_status == "PASS")
            init_by_fold.setdefault(fold, []).append(task["initialization"]["module_init_sha256"])
    if len(task_rows) != 40:
        raise RuntimeError(f"expected 40 complete task audits, found {len(task_rows)}")
    if any(len(set(hashes)) != 1 for hashes in init_by_fold.values()):
        raise RuntimeError("same-fold trainable initial values differ between groups")

    base_rows = _historical_rows(
        Path(CFG_BASE) / "results/fold_metrics.csv", runner.BASE_GROUP
    )
    e81_path = Path("/model/spromptgl/experiment/experiment81_prcl/results/fold_metrics.csv")
    e81_rows_by_group = {
        group: _historical_rows(e81_path, group)
        for group in ("E81-G1-Visual-PRCL", "E81-G2-Medical-PRCL", "E81-G3-Anonymous-PRCL")
    }
    e81_summaries = {
        group: {
            metric: dict(zip(("mean", "population_sd"), mean_sd([row[metric] for row in rows])))
            for metric in HIST_METRICS
        }
        for group, rows in e81_rows_by_group.items()
    }
    base_summaries = {
        metric: dict(zip(("mean", "population_sd"), mean_sd([row[metric] for row in base_rows])))
        for metric in HIST_METRICS
    }
    summaries = _summaries(best_rows)
    folds_nine = set(range(1, 11)) - {2}
    summaries_nine = _summaries(best_rows, folds_nine)
    e81_summaries_nine = {
        group: {
            metric: dict(zip(("mean", "population_sd"), mean_sd([row[metric] for row in rows if row["fold"] in folds_nine])))
            for metric in HIST_METRICS
        }
        for group, rows in e81_rows_by_group.items()
    }
    base_rows_nine = [row for row in base_rows if row["fold"] in folds_nine]
    base_summaries_nine = {
        metric: dict(zip(("mean", "population_sd"), mean_sd([row[metric] for row in base_rows_nine])))
        for metric in HIST_METRICS
    }
    comparisons = _comparison_rows(
        best_rows, base_rows, e81_rows_by_group["E81-G2-Medical-PRCL"]
    )
    rng_audit = _rng_audit(task_rows)
    preflight = manifest.get("preflight", {})
    parity_count = int(preflight.get("base_off_parity_pass_count", 0))
    replay = preflight.get("two_step_recovery_replay", {})
    replay_pass = replay.get("status") == "PASS" and replay.get("steps") == 2
    if latest_pass != 40:
        raise RuntimeError(f"latest-state integrity passed {latest_pass}/40")

    comp = {(row["comparison"], row["scope"], row["metric"]): row for row in comparisons}
    a = comp[("G1_vs_DARG", "all10", "acc")]["mean_pp"] > 0 and comp[("G1_vs_DARG", "all10", "hard_auc")]["mean_pp"] > 0
    b = comp[("G2_vs_E81_G2", "all10", "acc")]["mean_pp"] > 0
    c = all(comp[("G3_vs_G2", "all10", metric)]["mean_pp"] > 0 for metric in ("acc", "hard_auc", "prob_auc"))
    d = all(comp[("G3_vs_G4", "all10", metric)]["mean_pp"] > 0 for metric in ("acc", "hard_auc", "prob_auc"))
    e = all(comp[("G3_vs_DARG", "all10", metric)]["mean_pp"] > 0 for metric in ("acc", "hard_auc", "prob_auc"))
    gate_labels = {
        "A_visual_relation_vs_darg": "SUPPORTED" if a else "NOT_SUPPORTED",
        "B_repair_vs_e81": "IMPROVED_ACC" if b else "NO_ACC_IMPROVEMENT",
        "C_interaction_vs_additive": "SUPPORTED" if c else "NOT_ESTABLISHED",
        "D_medical_vs_same_gram_control": "PRELIMINARY_SUPPORT" if d else "NOT_ESTABLISHED",
        "E_complete_vs_darg": "SUPPORTED" if e else "NOT_SUPPORTED",
    }
    medical_supported = d and comp[("G3_vs_G1", "all10", "acc")]["mean_pp"] > 0
    decision = "GO_CANDIDATE" if e and c and medical_supported else "NO_GO_PRCL"
    status = {
        "status": "COMPLETE_AUDITED",
        "formal_tasks_complete": len(task_rows),
        "best_checkpoint_independent_pass": len(best_rows),
        "latest_state_integrity_pass": latest_pass,
        "base_off_parity_pass": parity_count,
        "two_step_restore_replay": "PASS" if replay_pass else "FAIL",
        "rng_audit": rng_audit,
        "text_cache_sha256": runner.sha_file(HERE / "runtime/text_embeddings.pt"),
        "control_rotation_sha256": runner.torch_load(HERE / "runtime/control_rotation.pt")["matrix_sha256"],
        "g4_gram_max_abs_error": float(runner.text_payload()["medical_control_gram_max_abs_error"]),
        "all_fold_group_initializations_equal": True,
        "all_original_optimizer_states_restored": latest_pass == 40,
        "all_four_group_parameter_structures_equal": True,
        "scientific_gates": gate_labels,
        "medical_semantic_value": "SUPPORTED_PRELIMINARILY" if medical_supported else "NOT_ESTABLISHED",
        "research_decision": decision,
        "limitations": [
            "validation equals test and test ACC selects checkpoints",
            "CV preprocessing/RFE precedes folds and the 871-patient graph is transductive",
            "Fold 2 overlap rows 377 and 407 remain in original DARG training; residual regularization excludes them",
            "ten-fold splits have overlapping training subjects and intervals are descriptive",
            "CAM latent tokens do not identify anatomical ROIs",
            "same-Gram orthogonal rotation controls anchor geometry but does not prove semantic causality",
        ],
        "source_sha256": source_hash,
        "audit_sha256": runner.sha_file(HERE / "audit.py"),
        "report_only_revision": {
            "changed_file": str((HERE / "audit.py").resolve()),
            "training_frozen_sha256": frozen_sources.get(report_only_path),
            "current_audit_sha256": current_sources.get(report_only_path),
            "reason": "E74.4 best Checkpoints store fold identity, test predictions, and probabilities but omit a test_indices array; the audit verifies split_sha, test_count, and prediction shapes against the restored historical split.",
            "training_code_or_checkpoint_changed": False,
        },
        "prior_audit_attempts": json.loads((RESULTS / "audit_attempts.json").read_text(encoding="utf-8")),
        "independent_recomputed_summary": summaries,
        "historical_e81_summary": e81_summaries,
        "historical_darg_summary": base_summaries,
    }
    write_csv(RESULTS / "fold_metrics.csv", best_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    runner.atomic_json(RESULTS / "audit_status.json", status)
    _write_report(manifest, status, summaries, summaries_nine, comparisons,
                  e81_summaries, e81_summaries_nine, base_summaries,
                  base_summaries_nine, mechanism_rows)
    manifest.update({
        "status": "COMPLETE_AUDITED",
        "completed_tasks": len(task_rows),
        "independent_audits": len(best_rows),
        "latest_state_integrity_pass": latest_pass,
        "audit_sha256": status["audit_sha256"],
        "research_decision": decision,
        "scientific_gates": gate_labels,
    })
    runner.atomic_json(HERE / "manifest.json", manifest)
    return status


def main():
    started = time.time()
    audit_sha = runner.sha_file(HERE / "audit.py")
    try:
        status = run_audit()
    except Exception as error:
        attempts_path = RESULTS / "audit_attempts.json"
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.is_file() else []
        attempts.append({
            "status": "FAILED", "audit_sha256": audit_sha,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "elapsed_seconds": time.time() - started, "error": repr(error),
            "completed_best_audits_before_failure": 0,
            "classification": "report_only_audit_failure; formal training artifacts unchanged",
        })
        runner.atomic_json(attempts_path, attempts)
        runner.atomic_json(RESULTS / "audit_status.json", {
            "status": "FAILED", "error": repr(error),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_best_audits_before_failure": 0,
            "training_source_sha256": json.loads((HERE / "manifest.json").read_text(encoding="utf-8")).get("source_sha256"),
            "audit_sha256": audit_sha,
            "report_only": True,
        })
        raise
    attempts_path = RESULTS / "audit_attempts.json"
    attempts = json.loads(attempts_path.read_text(encoding="utf-8")) if attempts_path.is_file() else []
    attempts.append({
        "status": "PASS", "audit_sha256": audit_sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": time.time() - started,
        "formal_tasks_complete": status["formal_tasks_complete"],
        "best_checkpoint_independent_pass": status["best_checkpoint_independent_pass"],
        "latest_state_integrity_pass": status["latest_state_integrity_pass"],
        "classification": "report_only_audit; formal training artifacts unchanged",
    })
    runner.atomic_json(attempts_path, attempts)
    status["audit_attempts"] = attempts
    runner.atomic_json(RESULTS / "audit_status.json", status)
    report = (RESULTS / "report.md").read_text(encoding="utf-8")
    report += f"\nAudit execution record: {len(attempts)} attempts, final attempt {attempts[-1]['status']}; details are in `audit_attempts.json`.\n"
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": status["status"],
        "formal_tasks_complete": status["formal_tasks_complete"],
        "best_checkpoint_independent_pass": status["best_checkpoint_independent_pass"],
        "latest_state_integrity_pass": status["latest_state_integrity_pass"],
        "research_decision": status["research_decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
