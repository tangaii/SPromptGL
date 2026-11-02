#!/usr/bin/env python3
"""Independent checkpoint readback, intervention, and paired-result audit."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

_RUN_PATH = Path(__file__).with_name("run.py")
_RUN_SPEC = importlib.util.spec_from_file_location("e58_fave_runner", _RUN_PATH)
S = importlib.util.module_from_spec(_RUN_SPEC)
sys.modules[_RUN_SPEC.name] = S
_RUN_SPEC.loader.exec_module(S)
_manifest_path = S.EXP / "manifest.json"
_existing_manifest = json.loads(_manifest_path.read_text(encoding="utf-8")) if _manifest_path.exists() else {}
TRAINING_RUN_SHA = _existing_manifest.get("training_runner_sha256") or _existing_manifest.get("run_sha256")


def _pct(value):
    return float(value) * 100.0


def _summary_rows(reference, audited):
    import numpy as np

    groups = [
        ("E58-BASE-53.3", [reference[f]["reference_metrics"] for f in range(1, 11)]),
        (S.NUM, [r for r in audited if r["group"] == S.NUM]),
        (S.FULL, [r for r in audited if r["group"] == S.FULL]),
    ]
    rows = []
    for group, values in groups:
        row = {"group": group, "n_folds": len(values)}
        for metric in S.METRICS:
            seq = [_pct(float(value[metric])) for value in values]
            row[f"{metric}_mean_pct"] = float(np.mean(seq))
            row[f"{metric}_population_sd_pct"] = float(np.std(seq, ddof=0))
        if group == "E58-BASE-53.3":
            row["source"] = "historical read-only audited checkpoint"
        else:
            row["source"] = "Experiment 58 training + independent readback"
        rows.append(row)
    return rows


def _paired_rows(reference, audited):
    import numpy as np

    by = {(row["group"], int(row["fold"])): row for row in audited}
    comparisons = [
        (S.FULL, "E58-BASE-53.3"),
        (S.FULL, S.NUM),
        (S.NUM, "E58-BASE-53.3"),
    ]
    output = []
    for left, right in comparisons:
        for metric in S.METRICS:
            differences = []
            for fold in range(1, 11):
                a = float(reference[fold]["reference_metrics"][metric]) if left == "E58-BASE-53.3" else float(by[(left, fold)][metric])
                b = float(reference[fold]["reference_metrics"][metric]) if right == "E58-BASE-53.3" else float(by[(right, fold)][metric])
                differences.append(a - b)
            tol = 1e-12
            wins = sum(value > tol for value in differences)
            losses = sum(value < -tol for value in differences)
            ties = 10 - wins - losses
            output.append({
                "comparison": f"{left} - {right}", "metric": metric,
                "mean_difference_pp": float(np.mean(differences) * 100.0),
                "fold_differences_pp": json.dumps([round(x * 100.0, 8) for x in differences]),
                "wins": wins, "ties": ties, "losses": losses,
            })
    return output


def _cam_evidence_pre_output(state, fused):
    """Compute the shared CAM encoder once for fixed-checkpoint interventions."""
    cam = state["cam"]
    n = fused.shape[0]
    x, input_attention = cam.InputLayer(fused)
    attention = input_attention.mean(dim=1)
    for encoder, feed_forward in zip(cam.Encoder, cam.FeedForward):
        x, attn_layer = encoder(q=x, k=x, v=x, modal_num=cam.modal_num)
        attention = attn_layer.mean(dim=1)
        x = feed_forward(x)
    if x.shape[1:] != (4, 36):
        raise RuntimeError(f"CAM evidence insertion schema mismatch: {tuple(x.shape)}")
    return x.detach(), attention.detach(), n


def _run_cached_evidence(state, x, attention, n, text_override=None, z_override=None, disabled=False):
    import torch

    if disabled:
        changed = x
        fave_info = None
    else:
        h_corr, _, _, fave_info = state["fave"](
            state["fave_context"]["z"], state["fave_context"]["quality"],
            x[:, 3, :], state["fave_context"]["train_idx"],
            text_override=text_override, z_override=z_override,
        )
        changed = torch.cat((x[:, :3, :], h_corr[:, None, :]), dim=1)
    logp, hidden = state["cam"].Outputlayer(changed.reshape(n, -1), attention.reshape(n, -1))
    return logp, hidden, fave_info


def _run_interventions(state, full_out, prompt_cache, group, fold):
    import torch

    model, cam, fave = state["model"], state["cam"], state["fave"]
    model.eval(); cam.eval(); fave.eval()
    with torch.no_grad():
        # Freeze the CADR-GFR output and CAM encoder state, then vary only the
        # evidence assignment/residual before the unchanged OutputLayer.
        x, attn, n = _cam_evidence_pre_output(state, full_out["fused"])
        full_metrics = S.R.classification_metrics(full_out["logp"], state["test"], state["y"])
        embeddings = prompt_cache["embeddings"]
        rows = []
        variants = [("correct_prompt", None, None, False)]
        variants.append(("zero_fave_residual", None, None, True))
        if group == S.FULL:
            variants.append(("anatomy_mismatch_prompt", state["fave_wrong_texts"], None, False))
            variants.append(("direction_reversed_prompt", embeddings[[1, 0]].to(x.device), None, False))
        z = state["fave_context"]["z"]
        test_idx = torch.as_tensor(state["test"], dtype=torch.long, device=z.device)
        z_reference = z.clone()
        z_reference[test_idx, 0] = 0.0
        variants.append(("test_connection_to_TD_reference", None, z_reference, False))
        z_irrelevant = z.clone()
        z_irrelevant[test_idx, 0] = z[test_idx, 1]
        variants.append(("test_connection_to_unmapped_hypothesis_edge", None, z_irrelevant, False))
        for name, text, z_alt, disabled in variants:
            logp, _, info = _run_cached_evidence(
                state, x, attn, n, text_override=text, z_override=z_alt, disabled=disabled
            )
            metrics = S.R.classification_metrics(logp, state["test"], state["y"])
            base_prob = full_out["logp"].detach().exp()[test_idx]
            alt_prob = logp.detach().exp()[test_idx]
            base_pred, alt_pred = base_prob.argmax(1), alt_prob.argmax(1)
            rows.append({
                "group": group, "fold": fold, "intervention": name,
                "acc": metrics["acc"], "hard_auc": metrics["hard_auc"],
                "prob_auc": metrics["prob_auc"], "spe": metrics["spe"],
                "sen": metrics["sen"], "ba": metrics["ba"], "asd_f1": metrics["asd_f1"],
                "acc_delta_pp": (metrics["acc"] - float(full_metrics["acc"])) * 100.0,
                "probability_l1_mean": float((alt_prob - base_prob).abs().mean().cpu()),
                "log_probability_l2_mean": float((logp[test_idx] - full_out["logp"][test_idx]).norm(dim=1).mean().cpu()),
                "prediction_flips": int((base_pred != alt_pred).sum().item()),
                "test_n": int(len(test_idx)),
                "fave_residual_norm": float(info["residual"][test_idx].norm().cpu()) if info is not None else 0.0,
            })
        return rows


def audit_fold(group: str, fold: int, reference_by_fold, prompt_cache):
    import torch

    state = S.build_state(group, fold, prompt_cache, reference_by_fold)
    cp_path = S.checkpoint_path(group, fold)
    task_path = S.task_path(group, fold)
    if not cp_path.is_file() or not task_path.is_file():
        raise FileNotFoundError(f"missing best checkpoint/task for {group} fold {fold}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("best_sha256") != S.sha_file(cp_path):
        raise RuntimeError("task metadata does not match best checkpoint")
    for key, value in {
        "group": group, "fold": fold, "seed": 0,
        "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_gfr_sha": state["initial_gfr_sha"],
        "initial_cam_sha": state["initial_cam_sha"],
        "initial_cadr_added_sha": state["initial_added_sha"],
        "initial_fave_parameter_sha": state["initial_fave_param_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "fave_init_rng_sha": state["fave_init_rng_sha"],
    }.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"checkpoint identity mismatch for {key}: fold={fold} group={group}")
    if checkpoint["best_epoch"] < int(S.R.CFG["warmup"]["epochs"]):
        raise RuntimeError("best checkpoint was selected during warmup")
    expected_wd = {"gfr": [0.11], "cam": [0.11], "cadr": [0.01, 0.01], "fave": [1e-4, 0.0]}
    if checkpoint["optimizer_weight_decay"] != expected_wd:
        raise RuntimeError(f"optimizer weight decay mismatch: {checkpoint['optimizer_weight_decay']}")
    if checkpoint["fave_source_sha256"] != S.sha_file(S.EXP / "fave_pen.py") or checkpoint["runner_source_sha256"] != TRAINING_RUN_SHA:
        raise RuntimeError("checkpoint source SHA does not match the recorded training implementation")
    if checkpoint.get("reference_context_sha") != state["fave_context"]["cache_sha256"]:
        raise RuntimeError("fold-local normative-reference cache SHA mismatch")
    if checkpoint.get("prompt_embedding_sha") != prompt_cache["embedding_sha256"]:
        raise RuntimeError("frozen CLIP embedding SHA mismatch")
    if checkpoint.get("prompt_group_sha") != state["prompt_group_sha"]:
        raise RuntimeError("FULL/NUM fixed prompt-bank identity mismatch")
    if checkpoint.get("config_sha256") != S.sha_file(S.CONFIG_PATH) or checkpoint.get("prompts_sha256") != S.sha_file(S.PROMPTS_PATH):
        raise RuntimeError("checkpoint frozen config or prompt-bank JSON SHA mismatch")
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["fave"].load_state_dict(checkpoint["fave"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    state["opt_fave"].load_state_dict(checkpoint["optimizer_fave"])
    for module in (state["model"], state["cam"], state["fave"]):
        if not all(torch.isfinite(value).all() for value in module.parameters()):
            raise FloatingPointError(f"nonfinite checkpoint parameters fold={fold} group={group}")
    state["model"].eval(); state["cam"].eval(); state["fave"].eval()
    with torch.no_grad():
        out = S.forward_fave_state(state, training=False, active=True)
    metrics = S.R.classification_metrics(out["logp"], state["test"], state["y"])
    for metric in S.METRICS:
        if abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) > 1e-10:
            raise RuntimeError(f"independent readback metric mismatch {metric} fold={fold} group={group}")
    if not torch.isfinite(out["logp"]).all() or not torch.isfinite(out["fused"]).all() or not torch.isfinite(out["total"]):
        raise FloatingPointError("nonfinite independent forward")

    # Test labels are never model inputs: relabeling the outer test set must not
    # change the feature path, prompt assignment, or predictions.
    saved = state["y"].clone()
    test = torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)
    state["y"][test] = 1 - state["y"][test]
    with torch.no_grad():
        label_permuted = S.forward_fave_state(state, training=False, active=True)
    if not torch.equal(out["logp"], label_permuted["logp"]):
        raise RuntimeError("test-label permutation changed predictions")
    if not torch.equal(out["fave_info"]["assignment"], label_permuted["fave_info"]["assignment"]):
        raise RuntimeError("test-label permutation changed FAVE assignment")
    state["y"].copy_(saved)

    # Classifier and auxiliary objectives have separately auditable gradient paths.
    grad_out = S.forward_fave_state(state, training=False, active=True)
    cls_grads = torch.autograd.grad(
        grad_out["cls"], [state["fave"].alpha, state["fave"].evidence_projection.weight,
                           state["fave"].semantic_projection.weight],
        retain_graph=True, allow_unused=True,
    )
    match_grads = torch.autograd.grad(
        grad_out["match"], [state["fave"].connection_encoder[0].weight,
                            state["fave"].semantic_projection.weight],
        retain_graph=True, allow_unused=True,
    )
    verify_grads = torch.autograd.grad(
        grad_out["verify"], [state["fave"].connection_encoder[0].weight,
                             state["fave"].semantic_projection.weight],
        retain_graph=False, allow_unused=True,
    )
    def grad_norm(grads):
        return math.sqrt(sum(float(g.detach().float().square().sum().cpu()) for g in grads if g is not None))

    cls_norm = grad_norm(cls_grads or [])
    match_norm = grad_norm(match_grads)
    verify_norm = grad_norm(verify_grads)

    info = out["fave_info"]
    test_states = info["state"][test]
    assign = info["assignment"][test]
    entropy = -(assign.clamp_min(1e-12) * assign.clamp_min(1e-12).log()).sum(dim=1)
    z0 = state["fave_context"]["z"][test, 0]
    # The stored hidden vector is post-OutputLayer; compare residual to the actual
    # 36-D pre-classifier CORR position using the fixed base CAM encoder.
    with torch.no_grad():
        cam_x, _, _ = _cam_evidence_pre_output(state, out["fused"])
    train_tensor = state["fave_context"]["train_idx"]
    train_states = info["state"][train_tensor]
    reject_rate_test = (assign[:, 1] >= assign[:, 0]).float().mean()
    diag = {
        "group": group, "fold": fold,
        "feature0_oof_reliability": float(state["cache_meta"]["reliability"][3]),
        "feature0_sigma": float(state["fave_context"]["sigma"][0]),
        "td_reference_n": int(state["fave_context"]["td_train_n"]),
        "train_support_count": int((train_states == 1).sum().item()),
        "test_support_count": int((test_states == 1).sum().item()),
        "test_contradiction_count": int((test_states == -1).sum().item()),
        "test_unknown_count": int((test_states == 0).sum().item()),
        "test_support_fraction": float((test_states == 1).float().mean().cpu()),
        "test_contradiction_fraction": float((test_states == -1).float().mean().cpu()),
        "test_unknown_fraction": float((test_states == 0).float().mean().cpu()),
        "reject_rate_test": float(reject_rate_test.detach().cpu()),
        "assignment_entropy_test_mean": float(entropy.mean().cpu()),
        "assignment_support_mass_test_mean": float(assign[:, 0].mean().cpu()),
        "quality_gate_mean_test": float(info["quality"][test].mean().cpu()),
        "quality_gate_sd_test": float(info["quality"][test].std(unbiased=False).cpu()),
        "z_feature0_test_mean": float(z0.mean().cpu()),
        "z_feature0_test_sd": float(z0.std(unbiased=False).cpu()),
        "residual_norm_test": float(info["residual"][test].norm().cpu()),
        "residual_to_corr_input_ratio": float(info["residual"][test].norm() / cam_x[test, 3].norm().clamp_min(1e-12)),
        "interaction_support_mean": float(info["interaction"][test][test_states == 1].mean().cpu()) if (test_states == 1).any() else 0.0,
        "interaction_contradiction_mean": float(info["interaction"][test][test_states == -1].mean().cpu()) if (test_states == -1).any() else 0.0,
        "classification_to_fave_grad_norm": cls_norm,
        "match_to_encoder_semantic_grad_norm": match_norm,
        "verify_to_encoder_semantic_grad_norm": verify_norm,
        "loss_cls": float(grad_out["cls"].detach().cpu()),
        "loss_match": float(grad_out["match"].detach().cpu()),
        "loss_verify": float(grad_out["verify"].detach().cpu()),
        "fave_alpha": float(state["fave"].alpha.detach().cpu()),
        "checkpoint_sha256": S.sha_file(cp_path),
        "checkpoint_audit": "PASS",
    }
    intervention_rows = _run_interventions(state, out, prompt_cache, group, fold)
    row = {
        "group": group, "fold": fold, "seed": 0,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        "warmup_best_epoch": int(task.get("warmup_best_epoch", -1)),
        "warmup_best_acc": float(task.get("warmup_best_acc", float("nan"))),
        **{metric: float(metrics[metric]) for metric in S.METRICS},
        "initial_shared_sha": state["initial_shared_sha"],
        "initial_fave_parameter_sha": state["initial_fave_param_sha"],
        "initial_rng_sha": state["construction_rng_sha"],
        "fave_init_rng_sha": state["fave_init_rng_sha"],
        "split_sha": state["split_sha"], "pair_sha": state["cache_meta"]["pair_sha"],
        "checkpoint_sha256": S.sha_file(cp_path), "checkpoint_audit": "PASS",
    }
    return row, diag, intervention_rows


def _fmt(mean, sd):
    return f"{mean:.4f} ± {sd:.4f}"


def write_report(reference_by_fold, summary, paired, diagnostics, interventions):
    import numpy as np

    base = summary[0]
    by_group = {row["group"]: row for row in summary}
    display = [
        ("E58-BASE-53.3", "历史只读参考"),
        (S.NUM, "等容量匿名向量对照"),
        (S.FULL, "FAVE-PEN医学文本组"),
    ]
    metric_labels = [
        ("acc", "ACC"), ("hard_auc", "Hard-AUC"), ("prob_auc", "Probability-AUC"),
        ("sen", "SEN-ASD"), ("spe", "SPE-TD"), ("ba", "BA"), ("asd_f1", "ASD-F1"),
    ]
    lines = [
        "# Experiment 58 — FAVE-PEN",
        "",
        "## Execution and base",
        "",
        "Formal training was limited to E58-FULL and E58-NUM, seed 0, 10 folds each. The 53.3 group `CADR_WD_0p01_GFR_WD_0p11` is a read-only reference; each new fold was rebuilt from the 43.1 untrained anchor. The `/data/adodas/SPromptGL-main` data root and `/model/spromptgl` historical source root are distinct filesystems; cross-root identity was established by SHA256, not path aliasing.",
        "",
        "The baseline audit verifies all ten reference checkpoint SHA values, fold splits, patient-pair caches, base initialization hashes, and optimizer weight decays. GFR and CAM Adam retain WD 0.11; CADR AdamW retains the actual sweep WD 0.01 in both groups. The new FAVE AdamW uses matrix WD 1e-4 and bias/normalization WD 0. The 10-epoch warmup is reproduced exactly; the two-step inactive-path replay matched the historical forward, loss, updates, optimizer state, and RNG state bitwise.",
        "",
        "Audit correction before delivery: an initial report-generation path paired the selected WD=0.01/GFR-WD=0.11 checkpoint identities with metrics from a different read-only reference group. The source now binds selected-group fold metrics to the same-fold checkpoint SHA, epoch, and PASS audit record. The corrected 53.3 baseline is 92.0742% ACC, 91.8653% Hard-AUC, 93.6684% Probability-AUC, 94.6392% SPE, and 89.0915% SEN. No model training or checkpoint was changed; all 20 Experiment 58 checkpoints were independently read again after the correction.",
        "",
        "## Method and evidence scope",
        "",
        "FAVE-PEN fits a fold-local Ridge conditional mean for each of the 256 standardized CORR features using only outer-train TD subjects and categorical PHENO age-bin/site/sex indicators. Five-fold patient-level cross-fitting supplies residual scales for training TD rows. The variance estimate uses `sqrt(0.9 * feature_variance + 0.1 * median_variance)` with a 0.05 floor. Outer-train ASD and test features use the final model fit on outer-train TD; test labels are not used. ANAT/FUNC inputs produce a bounded quality gate, not a clinical reliability probability.",
        "",
        "Only feature 0 is eligible for semantic matching: verified left thalamus–left anterior superior temporal gyrus. The ROI audit verifies 252/256 named connections; four unknown-extra-ROI edges (195, 225, 232, 236) are excluded from Prompt matching while remaining in base CORR. The sole directional literature hypothesis is increased thalamocortical connectivity from a small pediatric study. The finding is anatomically compatible, but its region/estimand is broader than this exact selected edge. Coverage is therefore 1/252, and the model does not treat support as an ASD diagnosis. A reversed-direction text is a counterfactual only; the match objective permits only the supported positive hypothesis or a reject slot. Contradiction and unknown states are reported separately.",
        "",
        "The correct text vectors use OpenAI CLIP RN50; the downloaded checkpoint SHA matched the official RN50 SHA. E58-NUM has the same FAVE trainable initialization and dimensions but a fixed seeded anonymous vector bank; it retains the same anatomical/direction mask and is not a no-prior control. The prompt mapper, evidence encoder, reject score, quality gate, and bounded residual affect only the CAM CORR position after the final CAM encoder; the original OutputLayer and other three positions remain unchanged. The initial residual coefficient is zero.",
        "",
        "The losses are the unchanged 53.3 objective plus `0.05 * L_match + 0.01 * L_verify`. Both evidence losses are restricted to outer-train patient indices. `L_match` assigns strongly increased deviations to the single supported prompt and contradictory/unknown rows to the learned reject slot. `L_verify` checks the signed interaction between the supported and reversed text using training-fold TD-reference substitution. This audits model faithfulness, not biological or clinical causality.",
        "",
        "## Results",
        "",
        "All values are percentages, mean ± population SD across the ten fixed folds. Hard-AUC uses hard predictions; Probability-AUC uses continuous probabilities.",
        "",
        "| Group | ACC | Hard-AUC | Probability-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, label in display:
        row = by_group[group]
        values = [_fmt(row[f"{metric}_mean_pct"], row[f"{metric}_population_sd_pct"]) for metric, _ in metric_labels]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.extend(["", "Paired comparisons use the exact fold metrics and report percentage-point differences. Fold counts are not treated as independent clinical cohorts.", ""])
    lines.extend(["| Comparison | Metric | Mean Δ (pp) | Fold Δ (pp), folds 1–10 | Win/Tie/Loss |", "|---|---|---:|---|---:|"])
    for item in paired:
        lines.append(f"| {item['comparison']} | {item['metric']} | {item['mean_difference_pp']:+.4f} | `{item['fold_differences_pp']}` | {item['wins']}/{item['ties']}/{item['losses']} |")
    lines.extend([
        "",
        "## Mechanism audit",
        "",
        "| Group | Mean test reject rate | Mean test residual/CORR-position ratio | Mean quality-gate SD | Mean classification-to-FAVE gradient norm |",
        "|---|---:|---:|---:|---:|",
    ])
    for group in (S.NUM, S.FULL):
        subset = [row for row in diagnostics if row["group"] == group]
        lines.append(
            f"| {group} | {np.mean([r['reject_rate_test'] for r in subset]):.4f} | "
            f"{np.mean([r['residual_to_corr_input_ratio'] for r in subset]):.6g} | "
            f"{np.mean([r['quality_gate_sd_test'] for r in subset]):.4f} | "
            f"{np.mean([r['classification_to_fave_grad_norm'] for r in subset]):.6g} |"
        )
    interventions_by = {}
    for row in interventions:
        interventions_by.setdefault(row["intervention"], []).append(row)
    correct_prob_auc = {
        (row["group"], int(row["fold"])): float(row["prob_auc"])
        for row in interventions if row["intervention"] == "correct_prompt"
    }
    lines.extend(["", "Fixed-checkpoint interventions freeze CADR-GFR output and CAM encoder features; only the FAVE evidence assignment/residual and unchanged OutputLayer are recomputed. They are not additional training groups.", "", "| Intervention | Group | Mean ACC Δ (pp) | Mean Probability-AUC Δ (pp) | Prediction flips / test patients | Mean probability L1 |", "|---|---|---:|---:|---:|---:|"])
    for name, rows in sorted(interventions_by.items()):
        by_intervention_group = {}
        for row in rows:
            by_intervention_group.setdefault(row["group"], []).append(row)
        for group, subset in by_intervention_group.items():
            test_total = sum(int(row["test_n"]) for row in subset)
            flips = sum(int(row["prediction_flips"]) for row in subset)
            prob_delta = np.mean([
                (float(row["prob_auc"]) - correct_prob_auc[(row["group"], int(row["fold"]))]) * 100.0
                for row in subset
            ])
            lines.append(
                f"| {name} | {group} | {np.mean([r['acc_delta_pp'] for r in subset]):+.4f} | "
                f"{prob_delta:+.4f} | "
                f"{flips}/{test_total} | {np.mean([r['probability_l1_mean'] for r in subset]):.6g} |"
            )
    full_mean = by_group[S.FULL]["acc_mean_pct"]
    base_mean = by_group["E58-BASE-53.3"]["acc_mean_pct"]
    full_num_acc = next(item for item in paired if item["comparison"] == f"{S.FULL} - {S.NUM}" and item["metric"] == "acc")
    full_base_acc = next(item for item in paired if item["comparison"] == f"{S.FULL} - E58-BASE-53.3" and item["metric"] == "acc")
    semantic_established = full_num_acc["mean_difference_pp"] > 0 and full_num_acc["wins"] >= 7
    diagnostic_increment = full_base_acc["mean_difference_pp"] > 0 and full_base_acc["wins"] >= 7
    conclusion = "GO" if semantic_established and diagnostic_increment else ("PARTIAL" if semantic_established or diagnostic_increment else "NO-GO")
    lines.extend([
        "",
        "## Conclusion",
        "",
        f"Engineering status: `GO` if all 20 checkpoints independently read back and the preflight passed; see checkpoint audit CSV. Semantic value versus the matched anonymous control: `{'ESTABLISHED' if semantic_established else 'NOT_ESTABLISHED'}` by the pre-registered descriptive criterion (positive mean ACC difference and at least 7/10 ACC wins); this is not a significance test. Diagnostic increment versus the historical baseline: `{'POSITIVE_DESCRIPTIVE' if diagnostic_increment else 'NOT_ESTABLISHED'}` under the same descriptive ACC criterion. Overall exploratory decision: **{conclusion}**.",
        "",
        "The evidence base has only one directionally supported connection out of 252 named edges, so the study is a narrow proof-of-mechanism test, not broad anatomical Prompt validation. A nonzero residual, gradient, or attention assignment alone does not establish semantic value. Report support, contradiction, and unknown as model evidence states only. The retained protocol uses val=test for checkpoint selection, CV-before-fold feature selection/standardization, an 871-node transductive graph, and a single seed; these results are protocol-matched exploratory estimates and do not establish unbiased external generalization.",
        "",
        "## Files and provenance",
        "",
        "See `baseline_audit.json` for fold-by-fold source, anchor, split, pair-cache, reference-checkpoint, and initialization SHA values; `manifest.json` records final training/audit status. Runtime checkpoints and caches are kept under `.runtime/`.",
        "",
    ])
    report_path = S.RESULTS / "report.md"
    temporary = report_path.with_name(report_path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, report_path)


def main():
    import numpy as np

    baseline = json.loads((S.EXP / "baseline_audit.json").read_text(encoding="utf-8"))
    if baseline.get("status") != "PASS":
        raise RuntimeError("baseline audit/preflight is not PASS")
    reference_by_fold = {int(row["fold"]): row for row in baseline["folds"]}
    if len(reference_by_fold) != 10:
        raise RuntimeError("baseline audit has incomplete fold inventory")
    prompt_cache = S.torch_load_prompt_cache()
    audited, diagnostics, interventions = [], [], []
    for group in (S.FULL, S.NUM):
        for fold in range(1, 11):
            row, diag, intervention_rows = audit_fold(group, fold, reference_by_fold, prompt_cache)
            audited.append(row); diagnostics.append(diag); interventions.extend(intervention_rows)
            S.write_rows(S.RESULTS / "fold_metrics.csv", audited)
            S.write_rows(S.RESULTS / "diagnostics.csv", diagnostics)
            S.write_rows(S.RESULTS / "interventions.csv", interventions)
            print(f"independent checkpoint read {group} fold {fold}/10: PASS", flush=True)
    summary = _summary_rows(reference_by_fold, audited)
    paired = _paired_rows(reference_by_fold, audited)
    S.write_rows(S.RESULTS / "summary.csv", summary)
    S.write_rows(S.RESULTS / "paired_comparisons.csv", paired)
    S.write_rows(S.RESULTS / "checkpoint_audit.csv", [
        {"group": row["group"], "fold": row["fold"], "checkpoint_sha256": row["checkpoint_sha256"], "audit": row["checkpoint_audit"], "metrics_recomputed": True, "metrics_match_saved": True}
        for row in audited
    ])
    write_report(reference_by_fold, summary, paired, diagnostics, interventions)
    audit_status = {"required": 20, "passed": len(audited), "failed": 0, "status": "PASS"}
    manifest = {
        "experiment": "Experiment 58 FAVE-PEN",
        "status": "COMPLETE_AUDITED",
        "actual_directory": str(S.EXP),
        "requested_directory": str(S.EXP),
        "formal_groups": [S.FULL, S.NUM],
        "formal_folds_required": 20,
        "formal_folds_complete": len(audited),
        "independent_checkpoint_audits": audit_status,
        "baseline_group": S.BASE_GROUP,
        "baseline_read_only": True,
        "baseline_audit_sha256": S.sha_file(S.EXP / "baseline_audit.json"),
        "config_sha256": S.sha_file(S.CONFIG_PATH),
        "run_sha256": S.sha_file(S.EXP / "run.py"),
        "training_runner_sha256": TRAINING_RUN_SHA,
        "fave_module_sha256": S.sha_file(S.EXP / "fave_pen.py"),
        "audit_sha256": S.sha_file(Path(__file__)),
        "prompts_sha256": S.sha_file(S.PROMPTS_PATH),
        "prompt_embedding_sha256": prompt_cache["embedding_sha256"],
        "RN50_checkpoint_sha256": S.RN50_SHA,
        "runtime_source_sha256": baseline["historical_source_sha256"],
        "data_sha256": baseline["data_sha256"],
        "roi_sha256": baseline["roi_source_sha256"],
        "fold_initialization_and_checkpoint_sha256": [
            {"group": row["group"], "fold": row["fold"], "checkpoint_sha256": row["checkpoint_sha256"], "initial_shared_sha": row["initial_shared_sha"], "initial_fave_parameter_sha": row["initial_fave_parameter_sha"], "split_sha": row["split_sha"], "pair_sha": row["pair_sha"]}
            for row in audited
        ],
        "mounts": baseline["mounts"],
        "handoff": {"path": "/model/spromptgl/experiment/handoff.md", "status": "PENDING"},
        "results_sha256": {path.name: S.sha_file(path) for path in sorted(S.RESULTS.iterdir()) if path.is_file()},
    }
    S.atomic_json(S.EXP / "manifest.json", manifest)
    print("independent audit complete: 20/20 PASS", flush=True)


if __name__ == "__main__":
    main()
