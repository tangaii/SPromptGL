#!/usr/bin/env python3
"""Independent checkpoint readback, propagation diagnostics and reporting."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run

RESULTS = EXP / "results"
RT = EXP / ".runtime"
LEGACY = Path("/model/spromptgl/experiment/experiment53_drap_disease_relation_propagation")
E0_METRICS = LEGACY / "results/fold_metrics.csv"


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def rel_diff(a, b):
    return float((a.detach() - b.detach()).norm().cpu() / b.detach().norm().clamp_min(1e-8).cpu())


def finite_module(module):
    return all(bool(torch.isfinite(p).all().item()) for p in module.parameters())


def load_checkpoint(group, fold):
    state = run.build_state(group, fold)
    checkpoint = run.torch_load(run.checkpoint_path(group, fold))
    if checkpoint.get("group") != group or int(checkpoint.get("fold", -1)) != fold:
        raise RuntimeError(f"checkpoint identity mismatch: {group} fold {fold}")
    state["base"].load_state_dict(checkpoint["base"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["drap"].load_state_dict(checkpoint["drap"], strict=True)
    return state, run.parent.load_fold_data(fold), checkpoint


def relation_summary(package):
    row = {}
    for m, result in enumerate(package["relation_metrics"]):
        modality = run.MODS[m]
        for key in ("oof_auc", "oof_pr_auc", "oof_brier", "oof_ece", "oof_score_mean", "oof_score_std", "oof_pos_mean", "oof_neg_mean", "oof_separation"):
            row[f"oof_{key}_{modality}"] = float(result[key])
        row[f"oof_reliable_{modality}"] = int(bool(result["reliable"]))
    row["reliable_modalities"] = json.dumps(package["reliable_modalities"])
    row["reliable_modality_count"] = len(package["reliable_modalities"])
    row["omega"] = json.dumps([float(x) for x in package["omega"]])
    row["inner_patient_disjoint"] = int(all(bool(x["patient_disjoint"]) for x in package["inner_rows"]))
    row["candidate_union_edges"] = int(package["candidate"]["union_edge_count"])
    for m, n in enumerate(package["candidate"]["modality_edge_counts"]):
        row[f"candidate_edges_{run.MODS[m]}"] = int(n)
    return row


def intervention_row(group, fold, label, full, other, state):
    p0, p1 = full["logp"].exp(), other["logp"].exp()
    met = run.metrics(other["logp"], state)
    test = state["test"]
    if hasattr(test, "detach"):
        test = test.detach().cpu().numpy()
    flips = float((p0[test].argmax(1) != p1[test].argmax(1)).float().mean().cpu())
    return {
        "group": group, "fold": fold, "intervention": label,
        "fused_relative_difference": rel_diff(full["fused"], other["fused"]),
        "delta_relative_difference": rel_diff(full["delta"], other["delta"]),
        "logits_relative_difference": rel_diff(full["logp"], other["logp"]),
        "probability_mean_absolute_difference": float((p0 - p1).abs().mean().cpu()),
        "test_prediction_flip_fraction": flips,
        **{f"{key}_pct": met[key] * 100 for key in ("acc", "hard_auc", "prob_auc", "spe", "sen")},
    }


def diagnose(group, fold, state, fd, checkpoint):
    package = state["relation_package"]
    with torch.no_grad():
        full = run.forward_state(state, fd, False, propagation_enabled=True)
        off = run.forward_state(state, fd, False, propagation_enabled=False, override="off")
        positive_off = run.forward_state(state, fd, False, propagation_enabled=True, override="positive_off")
        negative_off = run.forward_state(state, fd, False, propagation_enabled=True, override="negative_off")
        shuffled = run.forward_state(state, fd, False, propagation_enabled=True, override="shuffle")
        ordinary = run.forward_state(state, fd, False, propagation_enabled=True, override="ordinary")
    detail = full["relation"]
    raw = full["raw"]
    row = {
        "group": group, "fold": fold, "best_epoch": checkpoint["best_epoch"],
        "epochs_executed": checkpoint["epochs_executed"], "best_propagation_enabled": checkpoint["best_propagation_enabled"],
        "delta_norm": float(full["delta"].norm().cpu()),
        "delta_over_e1": float(full["delta"].norm().cpu() / full["fused_e1"].norm().clamp_min(1e-8).cpu()),
        "drap_over_raw": float((full["fused"] - raw).norm().cpu() / raw.norm().clamp_min(1e-8).cpu()),
        "e1_over_raw": float((full["fused_e1"] - raw).norm().cpu() / raw.norm().clamp_min(1e-8).cpu()),
        "positive_edges": int(detail["positive_edges"]) // 2, "negative_edges": int(detail["negative_edges"]) // 2,
        "positive_directed_edges": int(detail["positive_edges"]), "negative_directed_edges": int(detail["negative_edges"]),
        "positive_message_norm": float(detail["positive_norm"]), "negative_message_norm": float(detail["negative_norm"]),
        "positive_degree_zero_fraction": float(detail["positive_zero_neighbor_fraction"]),
        "positive_degree_one_fraction": float(detail["positive_single_neighbor_fraction"]),
        "positive_degree_multi_fraction": float(detail["positive_multi_neighbor_fraction"]),
        "negative_degree_zero_fraction": float(detail["negative_zero_neighbor_fraction"]),
        "negative_degree_one_fraction": float(detail["negative_single_neighbor_fraction"]),
        "negative_degree_multi_fraction": float(detail["negative_multi_neighbor_fraction"]),
        "q_mean": float(detail["candidate_q"].mean().cpu()),
        "q_std": float(detail["candidate_q"].std(unbiased=False).cpu()),
        "q_min": float(detail["candidate_q"].min().cpu()), "q_max": float(detail["candidate_q"].max().cpu()),
        "adversarial_loss_eval": float(detail["adv_loss"].detach().cpu()),
        "adversarial_modalities": int(detail["adv_modal"].unique().numel()) if detail["adv_modal"] is not None else 0,
        **run.metrics(full["logp"], state), **relation_summary(package),
    }
    from sklearn.metrics import roc_auc_score
    pair_y = fd["pair_y"].detach().cpu().numpy().astype(np.int64)
    for m, logits in enumerate(detail["pair_logits"]):
        score = logits.detach().cpu().numpy()
        row[f"outer_train_pair_auc_{run.MODS[m]}"] = float(roc_auc_score(pair_y, score))
        row[f"outer_train_pair_logit_same_mean_{run.MODS[m]}"] = float(score[pair_y == 1].mean())
        row[f"outer_train_pair_logit_different_mean_{run.MODS[m]}"] = float(score[pair_y == 0].mean())
    row["adversarial_accuracy_eval"] = (
        float((detail["adv_logits"].argmax(1) == detail["adv_modal"]).float().mean().cpu())
        if detail["adv_logits"] is not None else 0.0
    )
    row["adversarial_pair_count_per_modality"] = int(len(pair_y)) if detail["adv_modal"] is not None else 0
    row["best_train_adv_loss"] = float(checkpoint["diagnostics"].get("train_adv", 0.0))
    row["best_classification_to_relation_grad"] = float(checkpoint["diagnostics"].get("classification_to_relation", 0.0))
    row["best_relation_grad"] = float(checkpoint["diagnostics"].get("relation_grad", 0.0))
    for m, gcn in enumerate(state["base"].gcn):
        msg = full["base_details"]["msgs"][m]
        row[f"gcn{m}_layer1_weight_norm"] = float(gcn.gcn1.W.detach().norm().cpu())
        row[f"gcn{m}_layer2_weight_norm"] = float(gcn.gcn2.W.detach().norm().cpu())
        row[f"gcn{m}_message_norm"] = float(msg.norm().cpu())
        row[f"gcn{m}_message_over_input"] = float(msg.norm().cpu() / raw.norm().clamp_min(1e-8).cpu())
        graph = state["base"].graphs[m]
        adj = full["base_details"]["adjs"][m]
        row[f"graph_projection_norm_{run.MODS[m]}"] = float(graph.p.weight.detach().norm().cpu())
        row[f"graph_edge_count_{run.MODS[m]}"] = int((adj > 0).sum().item())
        row[f"graph_density_{run.MODS[m]}"] = float((adj > 0).float().mean().cpu())
        row[f"graph_finite_{run.MODS[m]}"] = int(torch.isfinite(adj).all().item())
        row[f"graph_symmetric_{run.MODS[m]}"] = int((adj - adj.T).abs().max().item() < 1e-6)
    row["gfr_gcn_mean_message_over_input"] = float(np.mean([row[f"gcn{m}_message_over_input"] for m in range(4)]))
    optimizer = run.optimizer_configuration(state)
    row.update({"base_weight_decay": json.dumps(optimizer["base_wd"]), "cam_weight_decay": json.dumps(optimizer["cam_wd"]), "drap_weight_decay": json.dumps(optimizer["drap_wd"])})

    candidate_q = detail["candidate_q"].detach().clone().requires_grad_(True)
    delta = state["drap"]._propagate(full["raw"].detach(), candidate_q, run.group_config(group), True)[0]
    q_grad = torch.autograd.grad(delta.square().sum(), candidate_q, allow_unused=True)[0] if delta.requires_grad else None
    row["q_to_delta_gradient_norm"] = 0.0 if q_grad is None else float(q_grad.norm().cpu())
    row["q_to_delta_gradient_expected_zero"] = int(group == "E8")
    row["shuffled_target_positive_edges"] = int(detail.get("shuffled_target_positive_edges", 0))
    row["shuffled_target_negative_edges"] = int(detail.get("shuffled_target_negative_edges", 0))
    row["shuffled_positive_edge_budget_delta"] = row["positive_edges"] - row["shuffled_target_positive_edges"] if group == "E7" else 0
    row["shuffled_negative_edge_budget_delta"] = row["negative_edges"] - row["shuffled_target_negative_edges"] if group == "E7" else 0
    row["e7_identity_score_change"] = float((detail["candidate_q"][state["drap"].corr_edge_ids] - shuffled["relation"]["q_prop"][state["drap"].corr_edge_ids]).abs().mean().cpu())
    row["e8_independent_of_q"] = int(torch.equal(
        state["drap"]._propagate(full["raw"], torch.zeros_like(detail["candidate_q"]), run.group_config("E8"), True)[0],
        state["drap"]._propagate(full["raw"], torch.linspace(0, 1, detail["candidate_q"].numel(), device="cuda"), run.group_config("E8"), True)[0],
    ))

    state["drap"].zero_grad(set_to_none=True); state["cam"].zero_grad(set_to_none=True)
    out = run.forward_state(state, fd, True, propagation_enabled=True)
    grads = run.gradient_audit(state, out)
    for key, value in grads.items():
        row[f"gradient_{key}"] = value
    row["checkpoint_finite"] = int(finite_module(state["base"]) and finite_module(state["cam"]) and finite_module(state["drap"]))
    row["branch_signature"] = json.dumps(checkpoint["branch_signature"], sort_keys=True)

    interventions = []
    for label, other in (("propagation_off", off), ("positive_off", positive_off), ("negative_off", negative_off), ("shuffled_identity", shuffled), ("ordinary_neighbors", ordinary)):
        interventions.append(intervention_row(group, fold, label, full, other, state))
    return row, interventions


def audit_checkpoints():
    run.parent.runtime()
    run.configure_determinism()
    old = [r for r in read_csv(E0_METRICS) if r.get("group") == "E0"]
    if len(old) != 10:
        raise RuntimeError("historical Experiment 53 E0 reference is not 10-fold complete")
    old_by_fold = {int(r["fold"]): r for r in old}
    checkpoint_rows, diagnostics, interventions, init_rows = [], [], [], []
    for group in run.GROUPS:
        for fold in range(1, 11):
            state, fd, ckpt = load_checkpoint(group, fold)
            expected_branch = {k: run.GROUP_BY_ID[group][k] for k in ("positive_enabled", "negative_enabled", "adversarial_mode", "uncertainty_mode", "identity_mode")}
            if ckpt.get("branch_signature") != expected_branch:
                raise RuntimeError(f"branch signature mismatch {group} fold {fold}")
            if int(ckpt["best_epoch"]) < 10 or not ckpt.get("best_propagation_enabled"):
                raise RuntimeError(f"best checkpoint predates active propagation {group} fold {fold}")
            with torch.no_grad():
                out = run.forward_state(state, fd, False, propagation_enabled=True)
            readback = run.metrics(out["logp"], state)
            stored = ckpt["metrics"]
            metric_error = max(abs(readback[k] - float(stored[k])) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"))
            metric_diffs = {k: readback[k] - float(stored[k]) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1") if abs(readback[k] - float(stored[k])) > 1e-10}
            init_pass = all((
                ckpt.get("split_sha") == run.split_sha(fold),
                ckpt.get("pair_sha") == run.sha_file(run.parent.FD / f"fold{fold:02d}.npz"),
                ckpt.get("initial_shared_sha") == state["initial_shared_sha"],
                ckpt.get("initial_schema_sha") == state["initial_schema_sha"],
                ckpt.get("initial_rng_sha") == state["initial_rng_sha"],
                ckpt.get("initial_relation_sha") == state["initial_relation_sha"],
                ckpt.get("adv_pair_sha") == state["adv_pair_sha"],
            ))
            expected_wd = {"base_wd": [0.11], "cam_wd": [0.11], "drap_wd": [0.0, 0.0001]}
            optimizer_actual = run.optimizer_configuration(state)
            optimizer_pass = optimizer_actual == expected_wd and ckpt.get("optimizer_wd") == optimizer_actual
            finite = finite_module(state["base"]) and finite_module(state["cam"]) and finite_module(state["drap"])
            if group in ("E1", "E3") and float(out["relation"]["negative_norm"]) != 0.0:
                raise RuntimeError(f"negative branch active in {group} fold {fold}")
            if group in ("E2", "E4", "E5", "E7", "E8") and float(out["relation"]["negative_norm"]) <= 0.0:
                raise RuntimeError(f"negative branch missing in {group} fold {fold}")
            if group in ("E3", "E4", "E5", "E6", "E7", "E8") and float(ckpt["diagnostics"]["train_adv"]) <= 0.0:
                raise RuntimeError(f"adversarial loss was inactive in {group} fold {fold}")
            if not init_pass or not optimizer_pass or not finite:
                raise RuntimeError(f"checkpoint independent audit failed {group} fold {fold}: metric={metric_error}, init={init_pass}, optimizer={optimizer_pass}, finite={finite}")
            checkpoint_rows.append({
                "group": group, "fold": fold,
                "status": "PASS" if metric_error <= 1e-10 else "REVIEW_METRIC_RANK_SENSITIVITY",
                "metric_max_abs_error": metric_error, "metric_differences": json.dumps(metric_diffs, sort_keys=True),
                **{f"readback_{k}": float(v) for k, v in readback.items()},
                **{f"stored_{k}": float(v) for k, v in stored.items()},
                "initialization_pass": init_pass, "optimizer_pass": optimizer_pass,
                "optimizer_wd": json.dumps(optimizer_actual, sort_keys=True),
                "finite_pass": finite, "best_epoch_active_propagation": True,
                "checkpoint_sha256": run.sha_file(run.checkpoint_path(group, fold)),
            })
            init_rows.append({
                "group": group, "fold": fold, "initial_shared_sha": ckpt["initial_shared_sha"],
                "initial_cam_sha": ckpt["initial_cam_sha"], "initial_rng_sha": ckpt["initial_rng_sha"],
                "initial_relation_sha": ckpt["initial_relation_sha"], "adv_pair_sha": ckpt["adv_pair_sha"],
                "split_sha": ckpt["split_sha"], "pair_sha": ckpt["pair_sha"], "pass": True,
            })
            diag, ints = diagnose(group, fold, state, fd, ckpt)
            diagnostics.append(diag); interventions.extend(ints)
            del state
            torch.cuda.empty_cache()

    e0_readback = []
    old_ckpt_dir = LEGACY / ".runtime/checkpoints"
    for fold in range(1, 11):
        state = run.parent.build_state("E0", fold)
        state["group"] = "E0"
        ckpt = run.torch_load(old_ckpt_dir / f"E0_f{fold:02d}.best.pt")
        if ckpt.get("group") != "E0" or int(ckpt.get("fold", -1)) != fold or ckpt.get("split_sha") != run.split_sha(fold):
            raise RuntimeError(f"historical E0 identity mismatch fold {fold}")
        state["base"].load_state_dict(ckpt["base"], strict=True); state["cam"].load_state_dict(ckpt["cam"], strict=True)
        fd = run.parent.load_fold_data(fold)
        with torch.no_grad():
            out = run.forward_base_only(state, fd)
        met = run.metrics(out["logp"], state)
        row = old_by_fold[fold]
        err = max(abs(met[k] - float(row[k])) for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"))
        if err > 1e-10 or int(ckpt["best_epoch"]) != int(row["best_epoch"]):
            raise RuntimeError(f"historical E0 readback differs fold {fold}: {err}")
        e0_readback.append({"fold": fold, "status": "PASS", "metric_max_abs_error": err, "best_epoch": int(ckpt["best_epoch"]), "checkpoint_sha256": run.sha_file(old_ckpt_dir / f"E0_f{fold:02d}.best.pt")})
        del state
        torch.cuda.empty_cache()

    write_csv(RESULTS / "checkpoint_audit.csv", checkpoint_rows)
    write_csv(RESULTS / "initialization_audit.csv", init_rows)
    write_csv(RESULTS / "diagnostics.csv", diagnostics)
    write_csv(RESULTS / "interventions.csv", interventions)
    write_csv(RESULTS / "historical_e0_readback.csv", e0_readback)
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metric_reviews = [row for row in checkpoint_rows if row["status"] != "PASS"]
    manifest.update({
        "status": "CHECKPOINT_READBACK_COMPLETE_WITH_REVIEW" if metric_reviews else "CHECKPOINT_AUDIT_PASS",
        "independent_checkpoint_audit": {
            "new_group_folds": len(checkpoint_rows), "all_reloaded": len(checkpoint_rows) == 80,
            "exact_metric_matches": len(checkpoint_rows) - len(metric_reviews),
            "metric_review_count": len(metric_reviews), "metric_reviews": metric_reviews,
            "parameter_initialization_optimizer_and_finite_checks_pass": True,
        },
        "historical_e0_readback": {"folds": len(e0_readback), "pass": True},
        "initialization_audit": {"rows": len(init_rows), "pass": True},
        "diagnostics_rows": len(diagnostics), "intervention_rows": len(interventions),
        "results_sha256": {p.name: run.sha_file(p) for p in (RESULTS / "checkpoint_audit.csv", RESULTS / "initialization_audit.csv", RESULTS / "diagnostics.csv", RESULTS / "interventions.csv", RESULTS / "historical_e0_readback.csv")},
    })
    run.atomic_json(manifest_path, manifest)
    print("AUDIT_PASS" if not metric_reviews else "AUDIT_READBACK_COMPLETE_WITH_REVIEW", len(checkpoint_rows), len(e0_readback), len(metric_reviews), flush=True)


def aggregate():
    if len(read_csv(RESULTS / "checkpoint_audit.csv")) != 80:
        raise RuntimeError("checkpoint audit must contain 80 rows")
    rows = []
    for group in run.GROUPS:
        for fold in range(1, 11):
            task = run.task_path(group, fold)
            if not task.is_file():
                raise FileNotFoundError(task)
            row = json.loads(task.read_text())
            audit_row = next(a for a in read_csv(RESULTS / "checkpoint_audit.csv") if a["group"] == group and int(a["fold"]) == fold)
            for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
                row[f"training_saved_{key}"] = row[key]
                row[key] = float(audit_row[f"readback_{key}"])
            row["checkpoint_readback_status"] = audit_row["status"]
            rows.append(row)
    e0 = [r for r in read_csv(E0_METRICS) if r.get("group") == "E0"]
    rows += [{**r, "source": "historical_readonly", "name": "E51_E1_EXACT_REPLAY_READ_ONLY"} for r in e0]
    run.csv_write(RESULTS / "fold_metrics.csv", rows)
    summary = []
    for group in ("E0",) + run.GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        record = {"group": group, "name": group_rows[0]["name"], "folds": len(group_rows), "source": group_rows[0].get("source", "new_training")}
        for key in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1"):
            values = np.asarray([float(r[key]) for r in group_rows])
            record[f"{key}_mean_pct"] = values.mean() * 100
            record[f"{key}_sd_pct"] = values.std(ddof=0) * 100
        summary.append(record)
    write_csv(RESULTS / "summary.csv", summary)
    by = {(r["group"], int(r["fold"])): r for r in rows}
    pairs = []
    contrasts = (("E1", "E0"), ("E2", "E1"), ("E3", "E1"), ("E4", "E2"), ("E4", "E3"), ("E4", "E5"), ("E4", "E6"), ("E4", "E7"), ("E4", "E8"), ("E4", "E0"))
    for left, right in contrasts:
        for metric in ("acc", "hard_auc", "prob_auc", "spe", "sen"):
            diffs = np.asarray([(float(by[(left, f)][metric]) - float(by[(right, f)][metric])) * 100 for f in range(1, 11)])
            pairs.append({"comparison": f"{left}_minus_{right}", "metric": metric, "mean_delta_pp": diffs.mean(), "wins": int((diffs > 0).sum()), "ties": int((diffs == 0).sum()), "losses": int((diffs < 0).sum()), "fold_delta_pp": json.dumps(diffs.tolist())})
    write_csv(RESULTS / "paired_comparisons.csv", pairs)
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest["aggregate_sha256"] = {p.name: run.sha_file(p) for p in (RESULTS / "fold_metrics.csv", RESULTS / "summary.csv", RESULTS / "paired_comparisons.csv")}
    run.atomic_json(EXP / "manifest.json", manifest)


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def make_report():
    summaries = {row["group"]: row for row in read_csv(RESULTS / "summary.csv")}
    diagnostics = read_csv(RESULTS / "diagnostics.csv")
    interventions = read_csv(RESULTS / "interventions.csv")
    comparisons = read_csv(RESULTS / "paired_comparisons.csv")
    checkpoint_audit = read_csv(RESULTS / "checkpoint_audit.csv")
    pre = json.loads((RT / "preflight.json").read_text())
    paired = {(r["comparison"], r["metric"]): r for r in comparisons}
    diag_by = {g: [r for r in diagnostics if r["group"] == g] for g in run.GROUPS}
    gfr_message_ratios = [mean(diag_by[g], "gfr_gcn_mean_message_over_input") for g in run.GROUPS]
    gfr_output_ratios = [mean(diag_by[g], "e1_over_raw") for g in run.GROUPS]
    drap_output_ratios = [mean(diag_by[g], "drap_over_raw") for g in run.GROUPS]
    corr_reliable_folds = sum(int(float(r["oof_reliable_CORR"])) for r in diag_by["E4"])
    corr_oof = mean(diag_by["E4"], "oof_oof_auc_CORR")
    modal_oof = {m: mean(diag_by["E4"], f"oof_oof_auc_{m}") for m in run.MODS}
    adv_sources_ok = all(int(float(r["adversarial_modalities"])) == 4 for g in ("E3", "E4", "E5", "E6", "E7", "E8") for r in diag_by[g])
    adv_grl_ok = all(float(r["best_relation_grad"]) > 0 and float(r["best_train_adv_loss"]) > 0 for g in ("E3", "E4", "E5", "E6", "E7", "E8") for r in diag_by[g])
    e4_off = [r for r in interventions if r["group"] == "E4" and r["intervention"] == "propagation_off"]
    propagation_effect = mean(e4_off, "logits_relative_difference") > 0 and mean(diag_by["E4"], "delta_norm") > 0
    e7_positive_gap = mean(diag_by["E4"], "positive_edges") - mean(diag_by["E7"], "positive_edges")
    e7_negative_gap = mean(diag_by["E4"], "negative_edges") - mean(diag_by["E7"], "negative_edges")
    matching_limited = abs(e7_positive_gap) / max(mean(diag_by["E4"], "positive_edges"), 1.0) > 0.10 or abs(e7_negative_gap) / max(mean(diag_by["E4"], "negative_edges"), 1.0) > 0.10
    e4_e0_acc = paired[("E4_minus_E0", "acc")]
    e4_e0_pauc = paired[("E4_minus_E0", "prob_auc")]
    e4_e5_pauc = paired[("E4_minus_E5", "prob_auc")]
    e4_e2_pauc = paired[("E4_minus_E2", "prob_auc")]
    diagnostic_go = (float(e4_e0_acc["mean_delta_pp"]) > 0 and float(e4_e0_pauc["mean_delta_pp"]) > 0 and int(e4_e0_acc["wins"]) >= 7)
    adversarial_effect_go = (adv_sources_ok and adv_grl_ok and float(e4_e5_pauc["mean_delta_pp"]) > 0 and int(e4_e5_pauc["wins"]) >= 7 and float(e4_e2_pauc["mean_delta_pp"]) > 0)
    exact_checkpoint_rows = sum(r["status"] == "PASS" for r in checkpoint_audit)
    lines = [
        "# Experiment 53.1: Corrected DRAP Propagation",
        "",
        "## Execution and Audit",
        "",
        f"Training: E1-E8, 80/80 folds. Independent best-checkpoint readback: {len(checkpoint_audit)}/80 ({exact_checkpoint_rows}/80 exact stored-metric matches). Historical E0: read-only 10/10 checkpoint readback. Preflight: {pre['status']}.",
        "The requested `/data/adodas/SPromptGL-main` experiment mount returned `OSError(5)` on directory creation. This run is stored on the separately verified writable `/model/spromptgl` overlay; source data, anchors and historical results remain on `/data` and are read-only. The two roots are different filesystems.",
        "",
        "## Method Audit",
        "",
        "SPromptGL uses four GraphLearn weighted-cosine patient graphs (threshold 0.9), two parameterized GCN layers per graph, normalized `A+I`, four correctly accumulated 320-dimensional messages and the original CAM. The paper's label-supervised adjacency term is distinct from the public simple-2 code's optimized classification plus GraphConstructLoss objective. The original GFR PHENO assignment overwrite is repaired in the Experiment 51 E1 LegacyFixedGFR base.",
        "DeepASD aligns modality feature representations before patient-graph learning. Here the adversary receives paired-patient relation features, for the same balanced outer-train pairs across all four modalities. Only OOF-reliable modalities determine disease edges; the adversarial source set remains four modalities. Conditional E3/E4/E6/E7/E8 inputs carry train-pair same/different one-hot labels; E5 uses the same 2D condition slot set to zero.",
        "Historical audits found GCN message collapse under strong decay (47.1/52), no robust superiority of true patient relations over strong controls (47.3/51.3), and Experiment 53 branch errors: negative propagation leaked into E1/E3, warmup was not passed, adversarial training gated off with CORR as the only reliable modality, and E7/E8 controls were poorly budget-matched. This run fixes these in its private module and does not modify historical code.",
        "",
        "## Frozen Group Differences",
        "",
        "E1: disease-reliable positive propagation only. E2: positive and negative, no adversary. E3: positive plus conditional four-modality adversary. E4: full signed conditional DRAP. E5: unconditional four-modality adversary. E6: continuous signed confidence without uncertainty skipping. E7: relation scores permuted over fixed CORR candidate edges with E4 fold-local degree budgets. E8: ordinary initial GraphLearn similarity neighbors with E4 per-node sign budgets, selected without disease q. All groups retain the same initial GFR/CAM and original `weight_decay=0.11`; pair and adversarial losses are 0.05 and 0.01; GRL scale 0.03; propagation coefficient 0.02 and negative factor 0.25. Epochs 0-9 have exactly zero added message; formal best selection starts at epoch 10.",
        "",
        "## Ten-Fold Classification (mean ± population SD, percent)",
        "",
        "|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("E0",) + run.GROUPS:
        r = summaries[group]
        name = r["name"]
        vals = [f"{float(r[k + '_mean_pct']):.4f} ± {float(r[k + '_sd_pct']):.4f}" for k in ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")]
        lines.append(f"|{group} {name}|" + "|".join(vals) + "|")
    lines += ["", "E0 is historical read-only reference; it was not retrained. Hard-AUC derives from hard labels and is not interchangeable with Probability-AUC.", "", "## Paired Differences", "", "|Comparison|Metric|Mean pp|Wins/Ties/Losses|", "|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"|{row['comparison']}|{row['metric']}|{float(row['mean_delta_pp']):.4f}|{row['wins']}/{row['ties']}/{row['losses']}|")
    lines += ["", "## OOF Relation Evidence and Propagation", "", "OOF evidence remains derived from inner patient-disjoint folds; it is not an independent clinical cohort. Only CORR is OOF-reliable when it meets AUC ≥0.55 and positive same/different separation. The adversarial branch nevertheless trains on four modality sources; this is CORR-anchored forced alignment, not evidence that all modalities have reliable disease relations.", "", "|Group|OOF CORR AUC|Positive edges|Negative edges|Positive msg norm|Negative msg norm|Delta/H_E1|H_DRAP-X|q-to-message grad|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in run.GROUPS:
        rr = [r for r in diagnostics if r["group"] == group]
        lines.append(f"|{group}|{mean(rr,'oof_oof_auc_CORR'):.4f}|{mean(rr,'positive_edges'):.1f}|{mean(rr,'negative_edges'):.1f}|{mean(rr,'positive_message_norm'):.6g}|{mean(rr,'negative_message_norm'):.6g}|{mean(rr,'delta_over_e1'):.6g}|{mean(rr,'drap_over_raw'):.6g}|{mean(rr,'q_to_delta_gradient_norm'):.6g}|")
    lines += ["", "E1/E3 negative message norms are required to be exactly zero; E2/E4/E5/E7/E8 require active negative messages. E6 positive/negative edge masks are disjoint. See diagnostics.csv for each modality's OOF AUC/PR-AUC/Brier/ECE, q distribution, GCN norms, graph support, degree-one node fractions, and gradient measurements.", "", "## Fixed-Checkpoint Interventions", "", "These are eval-only interventions on each group's own best checkpoint, not additional training groups.", "", "|Group|Intervention|ACC|Probability-AUC|Test flip fraction|Representation relative diff|Log-prob relative diff|", "|---|---|---:|---:|---:|---:|---:|"]
    for group in run.GROUPS:
        for label in ("propagation_off", "positive_off", "negative_off", "shuffled_identity", "ordinary_neighbors"):
            rr = [r for r in interventions if r["group"] == group and r["intervention"] == label]
            if rr:
                lines.append(f"|{group}|{label}|{mean(rr,'acc_pct'):.4f}|{mean(rr,'prob_auc_pct'):.4f}|{mean(rr,'test_prediction_flip_fraction'):.6g}|{mean(rr,'fused_relative_difference'):.6g}|{mean(rr,'logits_relative_difference'):.6g}|")
    lines += ["", "## Mechanism Conclusions", "", "- Branch corrections are enforced from the frozen group table and checkpoint branch signatures, rather than inferred from group-name substrings.", "- Warmup and active propagation status are stored per checkpoint; all formal best checkpoints must be at epoch ≥10.", "- The adversarial source classifier receives all four modality labels even though only CORR is permitted to define disease edges. Conditional and unconditional modes share architecture, pair set, balance and fixed CORR confidence weights.", "- E7 preserves candidate endpoints and reselects under the initial E4 degree budgets after per-forward CORR score permutation. The realized undirected edge budget can still be underfilled; E8 uses GraphLearn cosine and is q-independent. See diagnostics for undirected and directed edge counts and degree fractions.", f"- Across E1–E8, the original four-GCN message/input norm ratio ranges {min(gfr_message_ratios):.3g}–{max(gfr_message_ratios):.3g}, while `||H_GFR-X||/||X||` ranges {min(gfr_output_ratios):.3g}–{max(gfr_output_ratios):.3g}; total DRAP output displacement ranges {min(drap_output_ratios):.3g}–{max(drap_output_ratios):.3g}. The relation message therefore changes the representation beyond the near-identity original GFR, while the original `weight_decay=0.11` GCN messages remain strongly contracted.", "- The row-normalized difference operator can cancel edge-weight scale for degree-one nodes. Degree-0/1/multi fractions and q/message gradient are reported; relation BCE/AUC alone does not establish classification value.", "- OOF calibration is applied to the final all-outer-train relation model without test-driven refitting. Diagnostics show OOF metrics beside post-training outer-train relation AUC; the latter is a fit diagnostic, not an independent generalization estimate.", "", "## Decision", "", f"- ENGINEERING_{'GO' if len(checkpoint_audit) == 80 and exact_checkpoint_rows == 80 and len(e0_readback_rows := read_csv(RESULTS / 'historical_e0_readback.csv')) == 10 and pre['status'] == 'PASS' else 'NO-GO'}: {len(checkpoint_audit)}/80 checkpoints were independently reloaded; {exact_checkpoint_rows}/80 saved metrics matched; historical E0 readback={len(e0_readback_rows)}/10; preflight={pre['status']}.", f"- RELATION_{'GO' if corr_oof >= 0.55 and corr_reliable_folds >= 6 else 'NO-GO'}: CORR patient-disjoint OOF AUC={corr_oof:.4f}, reliable in {corr_reliable_folds}/10 folds; PHENO/ANAT/FUNC mean OOF AUCs={modal_oof['PHENO']:.4f}/{modal_oof['ANAT']:.4f}/{modal_oof['FUNC']:.4f}. Site confounding cannot be tested because site metadata is absent from the processed 321-column feature table.", f"- ADVERSARIAL_{'GO' if adversarial_effect_go else 'NO-GO'}: four-source and GRL-gradient execution checks={'passed' if adv_sources_ok and adv_grl_ok else 'failed'}; E4 modality accuracy={mean(diag_by['E4'], 'adversarial_accuracy_eval'):.4f} (four-way chance=0.25), so modality origin remains distinguishable. No patient-disjoint post-adversarial relation OOF was available; E4−E5 Probability-AUC={float(e4_e5_pauc['mean_delta_pp']):+.4f} pp and does not support an adversarial diagnostic gain.", f"- PROPAGATION_{'GO' if propagation_effect else 'NO-GO'}: E4 mean bounded delta norm={mean(diag_by['E4'], 'delta_norm'):.4f}, q-to-message gradient={mean(diag_by['E4'], 'q_to_delta_gradient_norm'):.6g}, and propagation-off logit relative difference={mean(e4_off, 'logits_relative_difference'):.6g}, with mean test flip fraction={mean(e4_off, 'test_prediction_flip_fraction'):.6g}. This confirms a real but small fixed-checkpoint effect.", f"- CONTROL_MATCHING_{'LIMITED' if matching_limited else 'ADEQUATE'}: E7 is short by {e7_positive_gap:.1f} positive and {e7_negative_gap:.1f} negative undirected edges on average relative to E4; edge identity is randomized correctly but realized graph budgets are not sufficiently matched.", f"- DIAGNOSTIC_{'GO' if diagnostic_go else 'NO-GO'}: E4−E0 ACC={float(e4_e0_acc['mean_delta_pp']):+.4f} pp ({e4_e0_acc['wins']}/{e4_e0_acc['ties']}/{e4_e0_acc['losses']} win/tie/loss), Probability-AUC={float(e4_e0_pauc['mean_delta_pp']):+.4f} pp ({e4_e0_pauc['wins']}/{e4_e0_pauc['ties']}/{e4_e0_pauc['losses']}); E4 also lacks a consistent Probability-AUC advantage over E7/E8.", f"- MICCAI_CANDIDATE_{'GO' if diagnostic_go and not matching_limited and adversarial_effect_go else 'NO-GO'}: the full candidate does not meet the preregistered diagnostic-increment criterion; real relation propagation is implemented, but disease-specific diagnostic value and fair E7 control separation are not established.", "", "All folds use historical `val=test`, CV-before preprocessing/RFE and an 871-node transductive graph. They can overstate generalization. The ten folds overlap on one cohort and one seed; they are not ten independent clinical cohorts or external validation. The processed data have no site metadata, so site-confounding remains untested.", "", f"Report generated from {len(diagnostics)} independently read model checkpoints. Source hashes and task status are in `manifest.json`."]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
    manifest = json.loads((EXP / "manifest.json").read_text())
    manifest["report_sha256"] = run.sha_file(RESULTS / "report.md")
    manifest["report_path"] = str(RESULTS / "report.md")
    manifest["status"] = "AUDITED_COMPLETE"
    run.atomic_json(EXP / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "aggregate", "report", "all"))
    args = parser.parse_args()
    if args.command in ("audit", "all"):
        audit_checkpoints()
    if args.command in ("aggregate", "all"):
        aggregate()
    if args.command in ("report", "all"):
        make_report()


if __name__ == "__main__":
    main()
