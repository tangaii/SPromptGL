#!/usr/bin/env python3
"""Independent Best/Latest audit and metric recomputation for Experiment 81.7."""
from __future__ import annotations

import csv
import copy
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import run

CFG = run.CFG
RESULTS = run.RESULTS
HIST_E744 = run.HIST_E744
E74_GROUP = run.E74_GROUP
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
METRIC_LABELS = run.METRIC_LABELS
HISTORICAL = {
    "E81.6.2-G1": (
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/fold_metrics.csv"),
        "SA-PRCL-G1-Protected",
    ),
    "E81.6.2-G3": (
        Path("/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl/results/fold_metrics.csv"),
        "SA-PRCL-G3-Joint",
    ),
    "E81.6.3-G1": (
        Path("/model/spromptgl/experiment/experiment81_6_3_cge_e2e/results/fold_metrics.csv"),
        "SA-PRCL-CGE-G1-Medical",
    ),
    "E81.6.1-G1": (
        Path("/model/spromptgl/experiment/experiment81_6_1_contrastive_prcl_res/results/fold_metrics.csv"),
        "E81.6.1-G1-Medical-CPRL-10",
    ),
}


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _manual_metrics(logp, labels):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    values = np.asarray(logp, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = values.argmax(axis=1).astype(np.int64)
    asd = (y == 0).astype(np.int64)
    pred_asd = (prediction == 0).astype(np.int64)
    return {
        "acc": float(accuracy_score(y, prediction)),
        "hard_auc": float(balanced_accuracy_score(y, prediction)),
            "prob_auc": float(roc_auc_score(asd, probabilities[:, 0])),
        "sen": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "spe": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "prediction": prediction,
        "probability": probabilities,
    }


def _metrics_from_saved(y, prediction, probability):
    import numpy as np
    prediction = np.asarray(prediction, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise RuntimeError("saved checkpoint probabilities must have ASD=0 and TD=1 columns")
    expected_prediction = probability.argmax(axis=1).astype(np.int64)
    if not np.array_equal(prediction, expected_prediction):
        raise RuntimeError("saved checkpoint predictions disagree with its probability columns")
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    y = np.asarray(y, dtype=np.int64)
    asd = (y == 0).astype(np.int64)
    result = {
        "acc": float(accuracy_score(y, prediction)),
        "hard_auc": float(balanced_accuracy_score(y, prediction)),
        "prob_auc": float(roc_auc_score(asd, probability[:, 0])),
        "sen": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "spe": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "prediction": prediction,
        "probability": probability,
    }
    return result


def _evaluate(e744, state, module, hook, group, mechanism, intervention="full", diagnostics=False):
    import numpy as np
    import torch
    original_grouping = hook.group
    if intervention in ("swapped_medical", "swapped_scrambled"):
        hook.group = intervention
        intervention = "full"
    hook.configure(training=False, intervention=intervention, diagnostics=diagnostics)
    try:
        with torch.no_grad():
            output = e744.forward(state, E74_GROUP, training=False)
    finally:
        hook.group = original_grouping
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("independent inference did not execute the original OutputLayer exactly once")
    test_ids = np.asarray(state["test_np"], dtype=np.int64)
    labels = state["y"].detach().cpu().numpy().astype(np.int64)[test_ids]
    logp = output["logp"].detach().cpu().numpy()[test_ids]
    computed = _manual_metrics(logp, labels)
    capture = None
    if diagnostics:
        capture = {}
        for name, value in hook.capture.items():
            if torch.is_tensor(value):
                capture[name] = value.detach().cpu().clone()
            else:
                capture[name] = value
    return {
        "metrics": {key: computed[key] for key in METRICS},
        "prediction": computed["prediction"],
        "probability": computed["probability"],
        "logp": logp,
        "hidden": output["hidden"].detach().cpu().numpy()[test_ids],
        "test_ids": test_ids,
        "labels": labels,
        "capture": capture,
    }


def _correction(source, target, labels):
    import numpy as np
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    corrected = int(np.sum((source != labels) & (target == labels)))
    harmed = int(np.sum((source == labels) & (target != labels)))
    return {"corrected": corrected, "harmed": harmed, "net": corrected - harmed}


def _mean_sd(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _paired_rows(name, left, right):
    import numpy as np
    rows = []
    for subset_name, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [i for i in range(1, 11) if i != 2])):
        for metric in METRICS:
            deltas = [float(left[fold][metric] - right[fold][metric]) for fold in folds]
            deltas_pp = [value * 100.0 for value in deltas]
            mean, sd = _mean_sd(deltas_pp)
            half = 1.96 * sd / math.sqrt(len(deltas))
            wins = sum(value > 1e-12 for value in deltas)
            ties = sum(abs(value) <= 1e-12 for value in deltas)
            losses = len(deltas) - wins - ties
            rows.append({
                "comparison": name,
                "subset": subset_name,
                "metric": METRIC_LABELS[metric],
                "folds": len(folds),
                "mean_difference_pp": mean,
                "descriptive_low_pp": mean - half,
                "descriptive_high_pp": mean + half,
                "W_T_L": f"{wins}/{ties}/{losses}",
                "fold_differences_pp": json.dumps({str(fold): float(left[fold][metric] - right[fold][metric]) * 100.0 for fold in folds}, sort_keys=True),
                "interval_note": "descriptive mean +/- 1.96*population_SD/sqrt(n); not an independent-CV significance interval",
            })
    return rows


def _summarize(rows):
    result = {}
    for group in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == group]
        if len(subset) != 10:
            continue
        result[group] = {}
        for metric in METRICS:
            mean, sd = _mean_sd([row[metric] for row in subset])
            result[group][metric] = {"mean": mean, "population_sd": sd}
            nine = [row[metric] for row in subset if int(row["fold"]) != 2]
            nine_mean, nine_sd = _mean_sd(nine)
            result[group][metric]["exclude_fold2_mean"] = nine_mean
            result[group][metric]["exclude_fold2_population_sd"] = nine_sd
    return result


def _load_historical_metrics(name):
    path, group_name = HISTORICAL[name]
    records = _read_csv(path)
    selected = [record for record in records if record.get("group") == group_name]
    if len(selected) != 10:
        raise RuntimeError(f"historical {name} has {len(selected)} rows, expected 10")
    return {
        int(record["fold"]): {metric: float(record[metric]) / 100.0 for metric in METRICS}
        for record in selected
    }


def _support_and_feature_diagnostics(state, full, base, group):
    import numpy as np
    import torch
    import torch.nn.functional as F
    capture = full["capture"]
    test_ids = full["test_ids"]
    labels = full["labels"]
    beta_normal = capture["beta_normal"][test_ids].numpy()
    beta_asd = capture["beta_asd"][test_ids].numpy()
    beta_entropies = {}
    beta_variation = {}
    for name, beta in (("normal", beta_normal), ("asd", beta_asd)):
        beta_entropies[name] = float((-(beta * np.log(np.clip(beta, 1e-12, 1))).sum(-1)).mean())
        beta_variation[name] = float(beta.std(axis=0).mean())

    attn_visual = capture["attention_visual"]
    attn_normal = capture["attention_normal"]
    attn_asd = capture["attention_asd"]
    if attn_asd is None:
        support_stats = {
            "mean_real_nonzero_edges": None, "real_edge_nonzero_rate": None,
            "null_positive_rate": None, "asd_td_support_disagreement_rate": None,
            "patient_consensus_jaccard_mean": None, "patient_consensus_jaccard_sd": None,
        }
    else:
        visual_rows = attn_visual[test_ids]
        normal_rows = attn_normal[test_ids]
        asd_rows = attn_asd[test_ids]
        normal_support = normal_rows[..., :15] > 0
        asd_support = asd_rows[..., :15] > 0
        union = normal_support | asd_support
        inter = normal_support & asd_support
        disagreement = float((normal_support ^ asd_support).float().mean())
        row_count = torch.cat((normal_support.sum(-1), asd_support.sum(-1)), dim=1).float()
        all_support = torch.cat((normal_support, asd_support), dim=1).reshape(len(test_ids), -1)
        consensus = all_support.float().mean(0) >= 0.5
        patient_inter = (all_support & consensus.unsqueeze(0)).sum(-1).float()
        patient_union = (all_support | consensus.unsqueeze(0)).sum(-1).float().clamp_min(1)
        jaccard = (patient_inter / patient_union).numpy()
        support_stats = {
            "mean_real_nonzero_edges": float(row_count.mean()),
            "real_edge_nonzero_rate": float(torch.cat((normal_support, asd_support), dim=1).float().mean()),
            "null_positive_rate": float(torch.cat((normal_rows[..., -1], asd_rows[..., -1]), dim=1).gt(0).float().mean()),
            "asd_td_support_disagreement_rate": disagreement,
            "patient_consensus_jaccard_mean": float(jaccard.mean()),
            "patient_consensus_jaccard_sd": float(jaccard.std(ddof=0)),
            "mean_asd_td_weight_l1": float((normal_rows - asd_rows).abs().sum(-1).mean()),
            "visual_null_positive_rate": float(visual_rows[..., -1].gt(0).float().mean()),
        }

    evidence = capture["prompt_evidence"][test_ids].numpy() if "prompt_evidence" in capture else None
    if evidence is not None:
        centered = evidence - evidence.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        weights = singular / max(singular.sum(), 1e-30)
        evidence_rank = int(np.linalg.matrix_rank(centered))
        effective_rank = float(np.exp(-(weights * np.log(np.clip(weights, 1e-30, 1))).sum()))
        prompt_score_std = float(capture["prompt_scores"][test_ids].std().item())
    else:
        evidence_rank, effective_rank, prompt_score_std = None, None, float(capture["prompt_scores"][test_ids].std().item())

    z = capture["z_cam"][test_ids].float()
    delta = capture["delta"][test_ids].float()
    enhanced_z = z + delta
    z_norm = z.norm(dim=-1).clamp_min(1e-12)
    delta_norm = delta.norm(dim=-1)
    cosine = F.cosine_similarity(z, enhanced_z, dim=-1)
    parallel = ((delta * z).sum(-1, keepdim=True) / z.square().sum(-1, keepdim=True).clamp_min(1e-12)) * z
    parallel_ratio = parallel.norm(dim=-1) / delta_norm.clamp_min(1e-12)
    orthogonal = delta - parallel
    orthogonal_ratio = orthogonal.norm(dim=-1) / delta_norm.clamp_min(1e-12)
    layer_norm = state["cam"].Outputlayer.mlp_head[0]
    ln_weight = layer_norm.weight.detach().cpu()
    ln_bias = layer_norm.bias.detach().cpu()
    ln_before = F.layer_norm(z, layer_norm.normalized_shape, ln_weight, ln_bias, layer_norm.eps)
    ln_after = F.layer_norm(enhanced_z, layer_norm.normalized_shape, ln_weight, ln_bias, layer_norm.eps)
    ln_difference = (ln_after - ln_before).abs().mean(-1)
    hidden_difference = np.linalg.norm(full["hidden"] - base["hidden"], axis=-1)
    margin_full = full["logp"][:, 0] - full["logp"][:, 1]
    margin_base = base["logp"][:, 0] - base["logp"][:, 1]
    margin_difference = np.abs(margin_full - margin_base)

    if evidence is not None:
        prompt_top_normal = np.bincount(beta_normal.argmax(-1), minlength=4).tolist()
        prompt_top_asd = np.bincount(beta_asd.argmax(-1), minlength=4).tolist()
    else:
        prompt_top_normal = prompt_top_asd = []
    return {
        **support_stats,
        "group": group,
        "beta_normal_entropy_mean": beta_entropies["normal"],
        "beta_asd_entropy_mean": beta_entropies["asd"],
        "beta_normal_patient_sd_mean": beta_variation["normal"],
        "beta_asd_patient_sd_mean": beta_variation["asd"],
        "beta_normal_top_prompt_counts": json.dumps(prompt_top_normal),
        "beta_asd_top_prompt_counts": json.dumps(prompt_top_asd),
        "prompt_evidence_numerical_rank": evidence_rank,
        "prompt_evidence_effective_rank": effective_rank,
        "prompt_score_sd": prompt_score_std,
        "relation_contrast_norm_mean": float(capture["relation_contrast"][test_ids].norm(dim=-1).mean()),
        "residual_ratio_mean": float(capture["residual_ratio"][test_ids].mean()),
        "residual_ratio_max": float(capture["residual_ratio"][test_ids].max()),
        "cam_norm_mean": float(z_norm.mean()),
        "cam_enhanced_cosine_mean": float(cosine.mean()),
        "residual_parallel_fraction_mean": float(parallel_ratio.mean()),
        "residual_orthogonal_fraction_mean": float(orthogonal_ratio.mean()),
        "layernorm144_mean_abs_change": float(ln_difference.mean()),
        "classifier_hidden_l2_change_mean": float(hidden_difference.mean()),
        "logit_margin_abs_change_mean": float(margin_difference.mean()),
        "residual_l2_mean": float(delta_norm.mean()),
        "residual_l2_max": float(delta_norm.max()),
    }


def _audit_latest_resume(e744, payload, group, fold, mechanism, grouping):
    import numpy as np
    import torch
    base = e744.load_history()
    runtime = run.rng_runtime(e744)

    def restore_once():
        state, _, identity = e744.prepare(fold)
        if identity != payload["base_identity"]:
            raise RuntimeError(f"Latest fold {fold} base identity mismatch")
        module = run.make_module(fold, state["cam"].parameters().__next__().device,
                                 run.prompt_payload()["semantic_basis_float64"])
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        base.load_bundle(state, payload["current"])
        module.load_state_dict(payload["psrr_state"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_psrr"]))
        hook = run.OutputPreHook(state, module, grouping, mechanism)
        runtime.restore_rng(payload["rng"])
        return state, module, optimizer, hook

    state_a, module_a, optimizer_a, hook_a = restore_once()
    epoch = int(payload["next_epoch"]) + 1
    next_a = run.training_step(e744, state_a, module_a, optimizer_a, hook_a, group,
                               epoch=epoch, fold=fold)
    after_a = {
        "base": base.state_bundle(state_a),
        "psrr": {key: value.detach().cpu().clone() for key, value in module_a.state_dict().items()},
        "optimizer": optimizer_a.state_dict(),
        "rng": runtime.pack_rng(),
    }
    hook_a.close()

    state_b, module_b, optimizer_b, hook_b = restore_once()
    next_b = run.training_step(e744, state_b, module_b, optimizer_b, hook_b, group,
                               epoch=epoch, fold=fold)
    after_b = {
        "base": base.state_bundle(state_b),
        "psrr": {key: value.detach().cpu().clone() for key, value in module_b.state_dict().items()},
        "optimizer": optimizer_b.state_dict(),
        "rng": runtime.pack_rng(),
    }
    hook_b.close()

    for name in after_a:
        equal, where = run.tensor_tree_equal(after_a[name], after_b[name])
        if not equal:
            difference, difference_path = run.tensor_tree_max_difference(after_a[name], after_b[name])
            raise RuntimeError(f"Latest next-step replay differs for {group} fold {fold} {name} at {where}; max_abs={difference} at {difference_path}")
    if next_a["metrics"] != next_b["metrics"] or next_a["total_loss"] != next_b["total_loss"]:
        raise RuntimeError(f"Latest next-step loss/metrics mismatch for {group} fold {fold}")
    if not np.array_equal(next_a["prediction"], next_b["prediction"]):
        raise RuntimeError(f"Latest next-step predictions differ for {group} fold {fold}")
    return {"status": "PASS", "next_step_epoch": epoch, "parameters_optimizers_rng_equal": True,
            "next_step_loss": next_a["total_loss"], "next_step_metrics": next_a["metrics"]}


def _metric_percent(values):
    return {key: float(values[key]) * 100.0 for key in METRICS}


def audit_group(group):
    import numpy as np
    import torch
    from psrr import PSRR
    e744 = run.base_runner()
    mechanism = "unary" if group == "G2" else "pairwise"
    grouping = "scrambled" if group == "G3" else "medical"
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_sha = preflight.get("source_sha256")
    frozen_inventory = preflight.get("source_files", {})
    if not source_sha or run.sha_object(frozen_inventory) != source_sha:
        raise RuntimeError("frozen training source inventory does not match its preflight SHA")
    audit_path = str((EXP / "audit.py").resolve())
    frozen_audit_sha = frozen_inventory.get(audit_path)
    if frozen_audit_sha is None:
        raise RuntimeError("preflight source inventory omitted audit.py")
    for path, expected_sha in frozen_inventory.items():
        if path == audit_path:
            continue
        if not Path(path).is_file() or run.sha_file(path) != expected_sha:
            raise RuntimeError(f"frozen training source changed after formal training: {path}")
    current_audit_sha = run.sha_file(audit_path)
    prior_fold_rows = _read_csv(RESULTS / "fold_metrics.csv") if (RESULTS / "fold_metrics.csv").is_file() else []
    prior_mechanism_rows = _read_csv(RESULTS / "mechanism.csv") if (RESULTS / "mechanism.csv").is_file() else []
    rows = [row for row in prior_fold_rows if row.get("group") not in ("DARG", group)]
    mechanism_rows = [row for row in prior_mechanism_rows if row.get("group") != group]
    correction_totals = {"A_to_B": {"corrected": 0, "harmed": 0},
                         "B_to_C": {"corrected": 0, "harmed": 0},
                         "A_to_C": {"corrected": 0, "harmed": 0}}
    latest_audits = []

    for fold in range(1, 11):
        task_path = run._task_path(group, fold)
        best_path = run._best_path(group, fold)
        latest_path = run._latest_path(group, fold)
        if not all(path.is_file() for path in (task_path, best_path, latest_path)):
            raise FileNotFoundError(f"missing formal task/Best/Latest for {group} fold {fold}")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_sha:
            raise RuntimeError(f"formal task state or source SHA failed for {group} fold {fold}")
        best = run.torch_load(best_path)
        latest = run.torch_load(latest_path)
        for payload, label in ((best, "Best"), (latest, "Latest")):
            if payload.get("schema") != "E81.7-PSRR-v1" or payload.get("group") != group or int(payload.get("fold", -1)) != fold:
                raise RuntimeError(f"{label} identity mismatch for {group} fold {fold}")
            if payload.get("source_sha256") != source_sha or payload.get("config_sha256") != run.sha_file(EXP / "config.json"):
                raise RuntimeError(f"{label} source/config SHA mismatch for {group} fold {fold}")
            if payload.get("warmup_sha256") != task.get("warmup_sha256"):
                raise RuntimeError(f"{label} historical warmup SHA mismatch for {group} fold {fold}")
        if "opt_base" not in best["base_state"] or "opt_cam" not in best["base_state"] or "opt_added" not in best["base_state"]:
            raise RuntimeError("Best omitted a historical optimizer state")
        if "opt_base" not in latest["current"] or "opt_cam" not in latest["current"] or "opt_added" not in latest["current"]:
            raise RuntimeError("Latest omitted a historical optimizer state")
        if not best.get("optimizer_psrr") or not latest.get("optimizer_psrr"):
            raise RuntimeError("Best/Latest omitted the PSRR optimizer state")
        for payload, label in ((best, "Best"), (latest, "Latest")):
            if not all(key in payload for key in ("psrr_state", "rng", "base_identity", "warmup_sha256")):
                raise RuntimeError(f"{label} lacks required recovery state")

        state, warmup, identity = e744.prepare(fold)
        if identity != best["base_identity"] or identity != latest["base_identity"]:
            raise RuntimeError(f"historical E61 preparation identity changed for {group} fold {fold}")
        if identity["warmup_sha256"] != task["warmup_sha256"]:
            raise RuntimeError(f"historical epoch-10 warmup identity failed for {group} fold {fold}")
        base = e744.load_history()
        base.load_bundle(state, best["base_state"])
        module = run.make_module(fold, state["cam"].parameters().__next__().device,
                                 run.prompt_payload()["semantic_basis_float64"])
        module.load_state_dict(best["psrr_state"], strict=True)
        opt_psrr = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        opt_psrr.load_state_dict(best["optimizer_psrr"])
        runtime = run.rng_runtime(e744)
        runtime.restore_rng(best["rng"])
        hook = run.OutputPreHook(state, module, grouping, mechanism)

        full = _evaluate(e744, state, module, hook, group, mechanism, diagnostics=True)
        saved_test_ids = np.asarray(state["test_np"], dtype=np.int64)
        if not np.array_equal(full["prediction"], np.asarray(best["test_prediction"], dtype=np.int64)):
            raise RuntimeError(f"independent Best hard predictions differ for {group} fold {fold}")
        if not np.allclose(full["probability"], np.asarray(best["test_probability"]), atol=2e-7, rtol=0):
            raise RuntimeError(f"independent Best probabilities differ for {group} fold {fold}")
        for metric in METRICS:
            if abs(full["metrics"][metric] - float(best["metrics"][metric])) > 2e-8:
                raise RuntimeError(f"independent Best metric {metric} differs for {group} fold {fold}")

        intervention_results = {}
        for intervention in ("prompt_off", "null_only", "global_off"):
            intervention_results[intervention] = _evaluate(e744, state, module, hook, group, mechanism,
                                                            intervention, diagnostics=intervention != "global_off")
        intervention_results["relation_off"] = _evaluate(e744, state, module, hook, group, mechanism,
                                                          "relation_off", diagnostics=True)
        swapped = "swapped_scrambled" if group == "G3" else "swapped_medical"
        intervention_results["group_swap"] = _evaluate(e744, state, module, hook, group, mechanism, swapped)
        relation_off = intervention_results["relation_off"]
        for name in ("prompt_off", "null_only"):
            if not np.array_equal(intervention_results[name]["logp"], relation_off["logp"]):
                raise RuntimeError(f"{name} did not exactly reproduce the zero-relation classifier output")
            if torch.count_nonzero(intervention_results[name]["capture"]["delta"]):
                raise RuntimeError(f"{name} did not produce an exactly zero residual")

        base_capture = relation_off
        darg_checkpoint = run.torch_load(
            HIST_E744 / ".runtime/checkpoints" / f"E74.4-G2-Positive-DCR_f{fold:02d}.best.pt"
        )
        if darg_checkpoint.get("fold") != fold or darg_checkpoint.get("group") != E74_GROUP:
            raise RuntimeError(f"read-only DARG Best identity mismatch in fold {fold}")
        darg_prediction = np.asarray(darg_checkpoint["test_prediction"], dtype=np.int64)
        darg_probability = np.asarray(darg_checkpoint["test_probability"], dtype=np.float64)
        if darg_prediction.shape[0] != len(full["labels"]) or darg_probability.shape[0] != len(full["labels"]):
            raise RuntimeError(f"DARG test probabilities do not align with this fold's test membership in fold {fold}")
        darg = _metrics_from_saved(full["labels"], darg_prediction, darg_probability)
        if not np.array_equal(darg["prediction"], darg_checkpoint["test_prediction"]):
            raise RuntimeError(f"DARG stored probability columns disagree with ASD=0 predictions in fold {fold}")
        for metric in METRICS:
            if abs(darg[metric] - float(darg_checkpoint["metrics"][metric])) > 2e-8:
                raise RuntimeError(f"independent DARG metric {metric} mismatch in fold {fold}")

        corrections = {
            "A_to_B": _correction(darg["prediction"], base_capture["prediction"], full["labels"]),
            "B_to_C": _correction(base_capture["prediction"], full["prediction"], full["labels"]),
            "A_to_C": _correction(darg["prediction"], full["prediction"], full["labels"]),
        }
        for name, value in corrections.items():
            correction_totals[name]["corrected"] += value["corrected"]
            correction_totals[name]["harmed"] += value["harmed"]

        for model_name, result in ((group, full), ("DARG", darg)):
            metric_values = result["metrics"] if model_name == group else result
            row = {"group": model_name, "fold": fold, **_metric_percent(metric_values)}
            if model_name == group:
                row.update({
                    "best_epoch_1based": int(best["epoch_1based"]),
                    "epochs_executed": int(task["epochs_executed"]),
                    "checkpoint_sha256": run.sha_file(best_path),
                    "latest_sha256": run.sha_file(latest_path),
                    "independent_checkpoint_audit": "PASS",
                })
            rows.append(row)
        base_row = {"group": group, "fold": fold, "record_type": "A_to_B_B_to_C_A_to_C"}
        for stage, value in corrections.items():
            base_row[f"{stage}_corrected"] = value["corrected"]
            base_row[f"{stage}_harmed"] = value["harmed"]
            base_row[f"{stage}_net"] = value["net"]
        mechanism_rows.append(base_row)

        diag = _support_and_feature_diagnostics(state, full, base_capture, group)
        diag.update({
            "fold": fold,
            "record_type": "best_mechanism",
            "best_epoch_1based": int(best["epoch_1based"]),
            "prompt_off_probability_mae": float(np.abs(full["probability"][:, 0] - intervention_results["prompt_off"]["probability"][:, 0]).mean()),
            "prompt_off_hard_flips": int(np.sum(full["prediction"] != intervention_results["prompt_off"]["prediction"])),
            "relation_off_probability_mae": float(np.abs(full["probability"][:, 0] - intervention_results["relation_off"]["probability"][:, 0]).mean()),
            "null_only_probability_mae": float(np.abs(full["probability"][:, 0] - intervention_results["null_only"]["probability"][:, 0]).mean()),
            "group_swap_probability_mae": float(np.abs(full["probability"][:, 0] - intervention_results["group_swap"]["probability"][:, 0]).mean()),
            "global_off_probability_mae": float(np.abs(full["probability"][:, 0] - intervention_results["global_off"]["probability"][:, 0]).mean()),
            "full_to_DARG_A_to_B": corrections["A_to_B"]["net"],
            "DARG_to_full_A_to_C": corrections["A_to_C"]["net"],
            "base_to_full_B_to_C": corrections["B_to_C"]["net"],
        })
        mechanism_rows.append(diag)

        # Compare all epochs shared with the historical DARG path. DARG may
        # have stopped earlier under its own ACC-based early stopping.
        old_task_path = HIST_E744 / ".runtime/tasks" / f"{E74_GROUP}_f{fold:02d}.json"
        old_task = json.loads(old_task_path.read_text(encoding="utf-8"))
        old_trace = {int(item["epoch"]): item for item in old_task.get("rng_trace", [])}
        shared_rng_epochs = []
        common = 0
        for trace_index, trace in enumerate(task["rng_trace"]):
            epoch = int(trace.get("epoch", 11 + trace_index))
            old = old_trace.get(epoch)
            if old is None:
                continue
            common += 1
            shared_rng_epochs.append(epoch)
            for new_key, old_key in (("before", "rng_before"), ("after_train", "rng_after_train"), ("after_eval", "rng_after_eval")):
                if trace[new_key] != old[old_key]:
                    raise RuntimeError(f"public RNG trajectory differs at {group} fold {fold} epoch {epoch} {new_key}")
        expected_shared = min(len(task["rng_trace"]), len(old_trace))
        if common != expected_shared or common == 0:
            raise RuntimeError(f"public RNG comparison has incomplete shared epochs for {group} fold {fold}: {common}/{expected_shared}")
        branch_probe = next((item for item in task.get("gradient_trace", []) if int(item["epoch"]) == 12), None)
        if branch_probe is None or any(not math.isfinite(float(branch_probe[key])) or float(branch_probe[key]) <= 0.0 for key in (
            "classification_to_delta_norm", "delta_to_cam_input_norm",
            "branch_to_cam_parameter_norm", "branch_to_darg_parameter_norm",
        )):
            raise RuntimeError(f"per-fold PSRR-to-CAM/DARG gradient path failed for {group} fold {fold}: {branch_probe}")
        diag.update({
            "shared_darg_rng_epochs": common,
            "shared_darg_rng_epoch_first": shared_rng_epochs[0],
            "shared_darg_rng_epoch_last": shared_rng_epochs[-1],
            "branch_vjp_epoch": int(branch_probe["epoch"]),
            "classification_to_delta_grad_norm": float(branch_probe["classification_to_delta_norm"]),
            "delta_to_cam_input_grad_norm": float(branch_probe["delta_to_cam_input_norm"]),
            "branch_to_cam_parameter_grad_norm": float(branch_probe["branch_to_cam_parameter_norm"]),
            "branch_to_darg_parameter_grad_norm": float(branch_probe["branch_to_darg_parameter_norm"]),
        })

        latest_audit = _audit_latest_resume(e744, latest, group, fold, mechanism, grouping)
        latest_audit["fold"] = fold
        latest_audits.append(latest_audit)
        hook.close()

    run.write_csv(RESULTS / "fold_metrics.csv", rows)
    run.write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    summaries = _summarize(rows)
    comparisons = []
    new_groups = [name for name in ("G1", "G2", "G3")
                  if len([row for row in rows if row["group"] == name]) == 10]
    normalized = {
        name: {
            fold: {metric: float(next(row[metric] for row in rows if row["group"] == name and int(row["fold"]) == fold)) / 100.0
                   for metric in METRICS}
            for fold in range(1, 11)
        }
        for name in new_groups
    }
    normalized["DARG"] = {
        fold: {metric: float(next(row[metric] for row in rows if row["group"] == "DARG" and int(row["fold"]) == fold)) / 100.0
               for metric in METRICS}
        for fold in range(1, 11)
    }
    for candidate in new_groups:
        comparisons.extend(_paired_rows(f"{candidate}_vs_DARG", normalized[candidate], normalized["DARG"]))
        for historical_name in HISTORICAL:
            comparisons.extend(_paired_rows(f"{candidate}_vs_{historical_name}", normalized[candidate], _load_historical_metrics(historical_name)))
    if "G1" in normalized and "G2" in normalized:
        comparisons.extend(_paired_rows("G1_vs_G2", normalized["G1"], normalized["G2"]))
    if "G1" in normalized and "G3" in normalized:
        comparisons.extend(_paired_rows("G1_vs_G3", normalized["G1"], normalized["G3"]))
    run.write_csv(RESULTS / "paired_comparisons.csv", comparisons)

    base_summary = summaries.get("G1", {})
    darg_summary = summaries.get("DARG", {})
    core_metrics = ("acc", "hard_auc")
    g1_gate = bool(base_summary and darg_summary)
    if g1_gate:
        for metric in core_metrics:
            g1_gate &= base_summary[metric]["mean"] > darg_summary[metric]["mean"]
            g1_gate &= base_summary[metric]["exclude_fold2_mean"] > darg_summary[metric]["exclude_fold2_mean"]
            g1_deltas = [normalized["G1"][fold][metric] - normalized["DARG"][fold][metric] for fold in range(1, 11)]
            g1_gate &= all((sum(g1_deltas[:index] + g1_deltas[index + 1:]) / 9.0) > 0 for index in range(10))
            g1_gate &= sum(delta > 1e-12 for delta in g1_deltas) > sum(delta < -1e-12 for delta in g1_deltas)

    mechanism_g1 = [row for row in mechanism_rows if row.get("group") == "G1" and row.get("record_type") == "best_mechanism"]
    sparse_mechanism_pass = bool(mechanism_g1)
    for row in mechanism_g1:
        sparse_mechanism_pass &= row.get("mean_real_nonzero_edges") is not None
        sparse_mechanism_pass &= 0 < float(row.get("real_edge_nonzero_rate") or 0) < 1
        sparse_mechanism_pass &= 0 < float(row.get("null_positive_rate") or 0) < 1
        sparse_mechanism_pass &= float(row.get("asd_td_support_disagreement_rate") or 0) > 0
        sparse_mechanism_pass &= float(row.get("beta_normal_patient_sd_mean") or 0) > 0
        sparse_mechanism_pass &= float(row.get("beta_asd_patient_sd_mean") or 0) > 0

    g2_gate = False
    if "G1" in normalized and "G2" in normalized:
        g2_gate = all(
            summaries["G1"][metric][key] > summaries["G2"][metric][key]
            for metric in core_metrics for key in ("mean", "exclude_fold2_mean")
        )
    g3_execution_gate = False
    if "G1" in normalized and "G2" in normalized and sparse_mechanism_pass:
        g3_execution_gate = g2_gate and all(
            summaries["G1"][metric][key] > summaries["G2"][metric][key]
            for metric in core_metrics for key in ("mean", "exclude_fold2_mean")
        )
    semantic_grouping_increment = False
    if "G1" in normalized and "G3" in normalized and sparse_mechanism_pass:
        semantic_grouping_increment = g3_execution_gate and all(
            summaries["G1"][metric][key] > summaries["G3"][metric][key]
            for metric in core_metrics for key in ("mean", "exclude_fold2_mean")
        )

    all_audits_pass = len(latest_audits) == 10 and all(item["status"] == "PASS" for item in latest_audits)
    status_path = RESULTS / "audit_status.json"
    existing = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    existing.update({
        f"{group.lower()}_completed_folds": 10,
        f"{group.lower()}_best_audit": "PASS",
        f"{group.lower()}_latest_resume_audit": "PASS" if all_audits_pass else "FAIL",
        f"{group.lower()}_public_rng_vs_darg": "PASS",
        f"{group.lower()}_end_to_end_gradient_audit": "PASS",
        "source_sha256": source_sha,
        "audit_source_sha256_at_training_freeze": frozen_audit_sha,
        "audit_source_sha256_current": current_audit_sha,
        "audit_only_patch_after_training": current_audit_sha != frozen_audit_sha,
        "fold2_psrr_overlap_rows_blocked": [376, 406],
        "fold2_psrr_training_assertion": "PASS",
        "summaries": summaries,
        "latest_resume_audits": latest_audits,
        "sparse_relation_mechanism": "PASS" if sparse_mechanism_pass else "FAIL_OR_NOT_ESTABLISHED",
        "g1_baseline_gate_pass": g1_gate,
        "e2e_mechanism": "PASS",
        "g2_gate_pass": g1_gate and all_audits_pass,
        "g3_execution_gate_pass": g3_execution_gate and all_audits_pass,
        "g3_gate_pass": semantic_grouping_increment and all_audits_pass,
    })
    existing["formal_training"] = "COMPLETE_GROUP_AUDITED"
    if group == "G1":
        existing["g1_status"] = "COMPLETE_AUDITED"
        if not g1_gate:
            existing["research_decision"] = "NO_GO_PSRR_BASELINE"
            existing["g2_gate_pass"] = False
            existing["g2_status"] = "NOT_EXECUTED_CONDITIONAL"
            existing["g3_status"] = "NOT_EXECUTED_CONDITIONAL"
            existing["g3_execution_gate_pass"] = False
            existing["g3_gate_pass"] = False
        else:
            existing["research_decision"] = "G1_BASELINE_GATE_PASS_G2_CONDITIONAL"
            existing["g2_status"] = "READY_TO_RUN"
            existing["g3_status"] = "NOT_EXECUTED_CONDITIONAL"
    elif group == "G2":
        existing["g2_status"] = "COMPLETE_AUDITED"
        existing["pairwise_increment"] = "PASS" if g2_gate else "NOT_ESTABLISHED"
        existing["g3_execution_gate_pass"] = bool(g3_execution_gate and all_audits_pass)
        existing["g3_status"] = "READY_TO_RUN" if existing["g3_execution_gate_pass"] else "NOT_EXECUTED_CONDITIONAL"
        existing["research_decision"] = "G2_AUDITED_G3_CONDITIONAL" if existing["g3_execution_gate_pass"] else "PAIRWISE_INCREMENT_NOT_ESTABLISHED_OR_G3_GATE_FAIL"
    else:
        existing["semantic_grouping_increment"] = "PASS" if semantic_grouping_increment else "NOT_ESTABLISHED"
        existing["g3_status"] = "COMPLETE_AUDITED"
        existing["research_decision"] = "PRELIMINARY_GO_PSRR" if (g1_gate and g2_gate and semantic_grouping_increment and all_audits_pass) else "NO_GO_OR_INCREMENT_NOT_ESTABLISHED"
    run.atomic_json(status_path, existing)
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = existing["research_decision"]
    manifest["formal_groups"] = {
        "G1": existing.get("g1_status", "COMPLETE_AUDITED"),
        "G2": existing.get("g2_status", "NOT_EXECUTED_CONDITIONAL"),
        "G3": existing.get("g3_status", "NOT_EXECUTED_CONDITIONAL"),
    }
    manifest["last_audited_group"] = group
    manifest["audit_status_sha256"] = run.sha_file(status_path)
    manifest["audit_source_sha256_at_training_freeze"] = frozen_audit_sha
    manifest["audit_source_sha256_current"] = current_audit_sha
    manifest["audit_only_patch_after_training"] = current_audit_sha != frozen_audit_sha
    run.atomic_json(manifest_path, manifest)
    _write_report(rows, summaries, comparisons, mechanism_rows, corrections=correction_totals, status=existing)
    return existing


def _write_report(fold_rows, summaries, comparisons, mechanism_rows, corrections, status):
    groups = [name for name in ("DARG", "G1", "G2", "G3") if name in summaries]
    lines = [
        "# Experiment 81.7 PSRR report", "",
        f"Engineering audit for the completed groups is recorded in `results/audit_status.json`. Current group status is `{status.get('research_decision')}`.", "",
        "Metrics are percentages. Hard-AUC is the historical hard-prediction balanced accuracy. Prob-AUC uses `P(ASD=0)` against ASD label 0. Fold summaries use population SD. Paired intervals are descriptive and are not significance intervals for independent patients.", "",
        "## Six Metrics", "",
        "|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        row = summaries[group]
        values = [f"{row[metric]['mean']:.4f}±{row[metric]['population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines.extend(["", "## Excluding Fold 2", "", "|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|"])
    for group in groups:
        row = summaries[group]
        values = [f"{row[metric]['exclude_fold2_mean']:.4f}±{row[metric]['exclude_fold2_population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines.extend(["", "## Paired Comparison Summary", "", "|Comparison|Subset|Metric|Mean Δ pp|Descriptive interval|W/T/L|", "|---|---|---|---:|---:|---:|"])
    for row in comparisons:
        if row["metric"] not in ("ACC", "Hard-AUC/BA"):
            continue
        lines.append(f"|{row['comparison']}|{row['subset']}|{row['metric']}|{float(row['mean_difference_pp']):+.4f}|[{float(row['descriptive_low_pp']):+.4f}, {float(row['descriptive_high_pp']):+.4f}]|{row['W_T_L']}|")
    lines.extend(["", "## Mechanism and Corrections", "", "All corrections count test-fold rows under the historical Fold2 overlap protocol. A is the E74.4-G2 DARG Best, B is the current PSRR Best epoch with its relational residual disabled, and C is the full PSRR output.", "", "|Stage|Corrected|Harmed|Net|", "|---|---:|---:|---:|"])
    for stage, values in corrections.items():
        lines.append(f"|{stage}|{values['corrected']}|{values['harmed']}|{values['corrected'] - values['harmed']}|")
    lines.extend(["", "|Group|Fold|Real edges/row|Null positive|ASD-TD support disagreement|Consensus Jaccard|TD beta entropy|ASD beta entropy|Residual ratio mean/max|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in mechanism_rows:
        if row.get("record_type") != "best_mechanism":
            continue
        vals = [row.get("group"), row.get("fold"), row.get("mean_real_nonzero_edges"), row.get("null_positive_rate"),
                row.get("asd_td_support_disagreement_rate"), row.get("patient_consensus_jaccard_mean"),
                row.get("beta_normal_entropy_mean"), row.get("beta_asd_entropy_mean"),
                f"{row.get('residual_ratio_mean')}/{row.get('residual_ratio_max')}"]
        lines.append("|" + "|".join("NA" if value is None else str(value) for value in vals) + "|")
    gradient_groups = {}
    for row in mechanism_rows:
        if row.get("record_type") == "best_mechanism" and row.get("branch_to_darg_parameter_grad_norm") is not None:
            gradient_groups.setdefault(row["group"], []).append(row)
    lines.extend(["", "## End-to-End Gradient Path", "", "Norms summarize the epoch-12 classification-loss VJP through the learned residual, measured separately at the CAM and DARG parameters.", "", "|Group|Folds|PSRR residual VJP|CAM VJP|DARG VJP|", "|---|---:|---:|---:|---:|"])
    for group, values in sorted(gradient_groups.items()):
        means = [sum(float(row[key]) for row in values) / len(values) for key in (
            "classification_to_delta_grad_norm", "branch_to_cam_parameter_grad_norm", "branch_to_darg_parameter_grad_norm",
        )]
        lines.append(f"|{group}|{len(values)}|{means[0]:.6g}|{means[1]:.6g}|{means[2]:.6g}|")
    lines.extend([
        "", "## Audit Decisions", "",
        f"- Best independent inference and six-metric recomputation: {status.get('g1_best_audit', 'NOT_RUN')} for G1 and the corresponding status for conditional groups.",
        f"- Latest full restore and next-step replay: {status.get('g1_latest_resume_audit', 'NOT_RUN')} for G1 and the corresponding status for conditional groups.",
        f"- G1 vs DARG pre-registered baseline gate: {status.get('g1_baseline_gate_pass', 'PENDING')}.",
        f"- End-to-end gradient path through PSRR, CAM, and DARG: {status.get('e2e_mechanism', 'PENDING')}.",
        f"- Pairwise increment G1 vs independently trained unary G2: {status.get('pairwise_increment', 'CONDITIONAL_NOT_RUN')}.",
        f"- Medical grouping increment G1 vs scrambled G3: {status.get('semantic_grouping_increment', 'CONDITIONAL_NOT_RUN')}.",
        f"- Conditional group execution: G2 `{status.get('g2_status', 'PENDING')}`, G3 `{status.get('g3_status', 'PENDING')}`.",
        f"- Current research decision: `{status.get('research_decision')}`.",
        f"- Frozen training source SHA256: `{status.get('source_sha256')}`.",
        f"- Audit script SHA256 at training freeze/current audit: `{status.get('audit_source_sha256_at_training_freeze')}` / `{status.get('audit_source_sha256_current')}`.",
        "- Post-training audit-only corrections cover group-swap routing, DARG metric-field normalization, RNG epoch reconstruction, optimizer-state deep copies during replay, and percentage-point reporting. The frozen training implementation and checkpoints were not modified.",
        "",
        "Fixed Prompt geometry constrains the parameterization but does not validate clinical meaning. The 16 local slots are latent feature locations, not anatomical ROIs. Fold2 training/test overlap, test=validation model selection, pre-CV processing/RFE and the transductive patient graph remain inherited limitations. A single seed and overlapping CV do not establish independent clinical generalization.",
        "",
        "Files: `results/fold_metrics.csv`, `results/paired_comparisons.csv`, `results/mechanism.csv`, `results/audit_status.json`, `results/preflight.json`, and `manifest.json`.",
    ])
    text = "\n".join(lines) + "\n"
    target = RESULTS / "report.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("G1", "G2", "G3"), required=True)
    args = parser.parse_args()
    result = audit_group(args.group)
    print(f"AUDIT {args.group} PASS research={result['research_decision']}", flush=True)


if __name__ == "__main__":
    main()
