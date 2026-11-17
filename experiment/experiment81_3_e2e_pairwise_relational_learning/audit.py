#!/usr/bin/env python3
"""Independent E81.3 checkpoint, RNG, mechanism and paired-metric audit."""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import run as experiment
from relation_learning import PairwiseRelationalCorrection, GROUP_UNARY, GROUP_PAIRWISE

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
GROUPS = experiment.GROUPS
METRICS = experiment.METRICS
BASE_GROUP = experiment.BASE_GROUP


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        __import__("os").fsync(stream.fileno())
    __import__("os").replace(temp, path)


def metrics_fn():
    return experiment.base_runtime().classification_metrics


def metric_values(record):
    return {key: float(record[key]) for key in METRICS}


def test_labels(state):
    labels = state["y"].detach().cpu().numpy()
    return labels[np.asarray(state["test"], dtype=np.int64)].astype(np.int64)


def compare_predictions(reference, candidate, labels):
    reference = np.asarray(reference, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    fixed = int(((reference != labels) & (candidate == labels)).sum())
    harmed = int(((reference == labels) & (candidate != labels)).sum())
    return {"corrected": fixed, "harmed": harmed, "net": fixed - harmed,
            "flips": int((reference != candidate).sum())}


def mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=0))


def paired_rows(values_by_group, references):
    comparisons = [
        ("G1_vs_DARG", GROUPS[0], BASE_GROUP),
        ("G2_vs_DARG", GROUPS[1], BASE_GROUP),
        ("G2_vs_G1", GROUPS[1], GROUPS[0]),
        ("G2_vs_E81.2_G2_context", GROUPS[1], "E81.2-G2-Cross-Local-E2E"),
    ]
    output = []
    for label, left_name, right_name in comparisons:
        if right_name in values_by_group:
            left_values, right_values = values_by_group[left_name], values_by_group[right_name]
        else:
            left_values, right_values = values_by_group[left_name], references[right_name]
        for scope, folds in (("all10", list(range(1, 11))), ("without_fold2", [f for f in range(1, 11) if f != 2])):
            for metric in METRICS:
                diffs = np.asarray([left_values[f][metric] - right_values[f][metric] for f in folds], dtype=float)
                n = len(diffs)
                mean = float(diffs.mean()) * 100.0
                sd = float(diffs.std(ddof=1)) * 100.0 if n > 1 else 0.0
                tcrit = 2.262157 if n == 10 else 2.306004
                half = tcrit * sd / math.sqrt(n)
                wins = int((diffs > 1e-12).sum())
                ties = int((np.abs(diffs) <= 1e-12).sum())
                losses = n - wins - ties
                output.append({
                    "comparison": label, "left": left_name, "right": right_name,
                    "scope": scope, "metric": metric, "mean_difference_pp": mean,
                    "sd_difference_pp": sd, "ci95_low_pp": mean - half, "ci95_high_pp": mean + half,
                    "wins": wins, "ties": ties, "losses": losses,
                    "fold_differences_pp": json.dumps((diffs * 100.0).round(6).tolist()),
                })
    return output


def load_historical_checkpoint(base, fold):
    path = base.EXP / ".runtime/checkpoints" / f"{BASE_GROUP}_f{fold:02d}.best.pt"
    payload = experiment.torch_load(path)
    if payload.get("group") != BASE_GROUP or payload.get("fold") != fold:
        raise RuntimeError(f"historical DARG checkpoint identity mismatch at fold {fold}")
    return payload


def attention_summary(details, test):
    test_tensor = torch.as_tensor(test, dtype=torch.long, device=details["attention"].device)
    attention = details["attention"][test_tensor]
    beta = details["beta"][test_tensor]
    null = attention[..., -1]
    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(-1) / math.log(16.0)
    beta_entropy = -(beta * beta.clamp_min(1e-12).log()).sum(-1) / math.log(16.0)
    real_max = attention[..., :15].max(-1).values
    local_mean = beta.mean(0)
    local_sd = beta.std(0, unbiased=False)
    return {
        "null_mean": float(null.mean().cpu()),
        "null_p95": float(torch.quantile(null.reshape(-1), 0.95).cpu()),
        "null_dominant_fraction": float((null >= real_max).float().mean().cpu()),
        "nonnull_mass_mean": float((1.0 - null).mean().cpu()),
        "normalized_entropy_mean": float(entropy.mean().cpu()),
        "top_real_weight_mean": float(real_max.mean().cpu()),
        "readout_entropy_mean": float(beta_entropy.mean().cpu()),
        "readout_entropy_p05": float(torch.quantile(beta_entropy, 0.05).cpu()),
        "readout_max_mean": float(beta.max(-1).values.mean().cpu()),
        "readout_max_p95": float(torch.quantile(beta.max(-1).values, 0.95).cpu()),
        "readout_patient_distribution_sd": float(beta.std(0, unbiased=False).mean().cpu()),
        "readout_local_mean_min": float(local_mean.min().cpu()),
        "readout_local_mean_max": float(local_mean.max().cpu()),
        "readout_local_sd_mean": float(local_sd.mean().cpu()),
    }


def _mechanism_row(group, fold, final_metrics, base_metrics, historical, state, output, correction):
    test = state["test"]
    labels = test_labels(state)
    final_pred = np.asarray(final_metrics["pred"], dtype=np.int64)
    base_pred = np.asarray(base_metrics["pred"], dtype=np.int64)
    old_pred = np.asarray(historical["test_prediction"], dtype=np.int64)
    old_to_final = compare_predictions(old_pred, final_pred, labels)
    base_to_final = compare_predictions(base_pred, final_pred, labels)
    old_to_base = compare_predictions(old_pred, base_pred, labels)

    details = output["correction_details"]
    test_tensor = torch.as_tensor(test, dtype=torch.long, device=details["delta_margin"].device)
    delta = details["delta_margin"][test_tensor].detach()
    margin = (output["logp"][..., 1] - output["logp"][..., 0]).abs()[test_tensor].detach()
    gate = details["gate"][test_tensor].detach()
    attention = attention_summary(details, test)

    null_logp, null_details = correction(
        output["z_cam"], details["logp_anchor"], experiment.GROUP_MODE[group],
        training=False, intervention="null_only",
    )
    null_exact = bool(torch.equal(null_logp, output["logp"]) and
                      torch.equal(null_details["messages"], torch.zeros_like(null_details["messages"])) and
                      torch.equal(null_details["relation_summary"], torch.zeros_like(null_details["relation_summary"])))
    null_mae = float((null_logp[test_tensor].detach().exp() - output["logp"][test_tensor].detach().exp()).abs().mean().cpu())

    pairwise_intervention = None
    if group == GROUPS[1]:
        # Same trained G2 module and projection parameters; only the feature
        # constructor is switched to the G1 unary form.
        unary_logp, unary_details = correction(
            output["z_cam"], details["logp_anchor"], GROUP_UNARY, training=False,
        )
        unary_metrics = metrics_fn()(unary_logp, test, state["y"])
        pairwise_intervention = {
            "intervention": "G2_projection_with_G1_unary_key_value_inputs",
            "metrics": {key: float(unary_metrics[key]) for key in METRICS},
            "probability_mae": float((unary_logp[test_tensor].detach().exp() - output["logp_final"][test_tensor].detach().exp()).abs().mean().cpu()),
            "hard_prediction_flips": int((unary_metrics["pred"] != final_metrics["pred"]).sum()),
            "acc_change_pp": float((unary_metrics["acc"] - final_metrics["acc"]) * 100.0),
            "hard_auc_change_pp": float((unary_metrics["hard_auc"] - final_metrics["hard_auc"]) * 100.0),
            "asd_f1_change_pp": float((unary_metrics["asd_f1"] - final_metrics["asd_f1"]) * 100.0),
            "relation_summary_l2_change_mean": float(torch.linalg.vector_norm(
                details["relation_summary"][test_tensor] - unary_details["relation_summary"][test_tensor], dim=-1
            ).mean().cpu()),
        }

    median = torch.median(margin)
    strata = {}
    for name, mask in (("lower_margin_half", margin <= median), ("upper_margin_half", margin > median)):
        strata[name] = {
            "n": int(mask.sum().cpu()),
            "mean_abs_correction": float(delta[mask].abs().mean().cpu()),
            "mean_gate": float(gate[mask].mean().cpu()),
            "corrected_vs_current": compare_predictions(base_pred[mask.cpu().numpy()], final_pred[mask.cpu().numpy()], labels[mask.cpu().numpy()]),
        }
    high_conf_error = (base_pred != labels) & (margin.cpu().numpy() >= np.median(margin.cpu().numpy()))
    high_conf_error_final = final_pred[high_conf_error]
    high_conf_error_labels = labels[high_conf_error]
    high_conf_correction = delta[torch.as_tensor(high_conf_error, device=delta.device)] if high_conf_error.any() else delta[:0]

    return {
        "group": group, "fold": fold,
        "current_base_acc_pct": float(base_metrics["acc"] * 100.0),
        "corrected_acc_pct": float(final_metrics["acc"] * 100.0),
        "historical_darg_acc_pct": float(historical["metrics"]["acc"] * 100.0),
        "current_base_vs_historical_corrected_harmed_net": json.dumps(old_to_base),
        "historical_darg_to_final_corrected_harmed_net": json.dumps(old_to_final),
        "current_base_to_final_corrected_harmed_net": json.dumps(base_to_final),
        "corrected_probability_mae_vs_current_base": float((output["logp_final"][test_tensor].exp() - output["logp"][test_tensor].exp()).abs().mean().cpu()),
        "hard_flips_vs_current_base": int((final_pred != base_pred).sum()),
        "nonzero_correction_fraction": float((delta.abs() > 1e-8).float().mean().cpu()),
        "delta_margin_abs_mean": float(delta.abs().mean().cpu()),
        "delta_margin_abs_p95": float(torch.quantile(delta.abs(), 0.95).cpu()),
        "delta_margin_mean": float(delta.mean().cpu()),
        "gate_mean": float(gate.mean().cpu()),
        "gate_p05": float(torch.quantile(gate, 0.05).cpu()),
        "gate_p95": float(torch.quantile(gate, 0.95).cpu()),
        "base_margin_median": float(median.cpu()),
        "uncertainty_strata": json.dumps(strata),
        "high_confidence_base_error_count": int(high_conf_error.sum()),
        "high_confidence_error_corrected": int(((high_conf_error_final == high_conf_error_labels) & (high_conf_error_final != np.asarray(base_pred)[high_conf_error])).sum()),
        "high_confidence_error_mean_abs_correction": float(high_conf_correction.abs().mean().cpu()) if high_conf_correction.numel() else 0.0,
        "null_only_strict_identity": null_exact,
        "null_only_probability_mae": null_mae,
        "attention_summary": json.dumps(attention),
        "pairwise_unary_inference_intervention": json.dumps(pairwise_intervention) if pairwise_intervention else "",
    }


def _summary(values_by_group):
    output = {}
    for group, data in values_by_group.items():
        output[group] = {}
        for metric in METRICS:
            mean, sd = mean_sd([data[f][metric] * 100.0 for f in range(1, 11)])
            nine_mean, nine_sd = mean_sd([data[f][metric] * 100.0 for f in range(1, 11) if f != 2])
            output[group][metric] = {"mean": mean, "population_sd": sd,
                                     "without_fold2_mean": nine_mean, "without_fold2_population_sd": nine_sd}
    return output


def audit():
    started = time.monotonic()
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"PREFLIGHT_PASS", "COMPLETE_AUDITED"}:
        raise RuntimeError("preflight manifest is missing or not frozen")
    current_sha = experiment.source_sha()
    if current_sha != manifest.get("source_sha256"):
        raise RuntimeError("audit source SHA differs from frozen preflight")
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    training_source_sha = preflight.get("training_source_sha256", current_sha)
    if preflight.get("status") != "PASS" or preflight.get("base_off_parity", {}).get("passed") != 10:
        raise RuntimeError("preflight parity is incomplete")

    base = experiment.runtime()
    hist = base.load_history()
    metric_function = metrics_fn()
    darg_csv = read_csv(Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/results/fold_metrics.csv"))
    historical_darg = {int(row["fold"]): {metric: float(row[metric]) for metric in METRICS}
                       for row in darg_csv if row["group"] == BASE_GROUP}
    e812_csv = read_csv(Path("/model/spromptgl/experiment/experiment81_2_e2e_relational_correction/results/fold_metrics.csv"))
    e812_g2 = {int(row["fold"]): {metric: float(row[metric]) / 100.0 for metric in METRICS}
               for row in e812_csv if row["group"] == "E81.2-G2-Cross-Local-E2E"}
    references = {BASE_GROUP: historical_darg, "E81.2-G2-Cross-Local-E2E": e812_g2}

    task_records, checkpoints = {}, {}
    for group in GROUPS:
        for fold in range(1, 11):
            task_path, best_path, latest_path, _ = experiment.paths(group, fold)
            if not task_path.is_file() or not best_path.is_file() or not latest_path.is_file():
                raise RuntimeError(f"missing formal artifact {group} fold {fold}")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            checkpoint = experiment.torch_load(best_path)
            latest = experiment.torch_load(latest_path)
            if task.get("status") != "COMPLETE" or task.get("source_sha256") not in {current_sha, training_source_sha}:
                raise RuntimeError(f"task ledger incomplete/source mismatch {group} fold {fold}")
            if task.get("best_checkpoint_sha256") != experiment.sha_file(best_path) or task.get("latest_checkpoint_sha256") != experiment.sha_file(latest_path):
                raise RuntimeError(f"checkpoint SHA mismatch {group} fold {fold}")
            if latest.get("source_sha256") not in {current_sha, training_source_sha} or latest.get("group") != group or latest.get("fold") != fold:
                raise RuntimeError(f"latest identity mismatch {group} fold {fold}")
            if not {"model", "cam", "opt_base", "opt_cam", "opt_added"}.issubset(latest["current"]["base"]):
                raise RuntimeError(f"latest missing historical base state {group} fold {fold}")
            if not {"relation", "optimizer_relation"}.issubset(latest["current"]):
                raise RuntimeError(f"latest missing relation state {group} fold {fold}")
            if not {"python", "numpy", "cpu", "cuda"}.issubset(latest["rng"]):
                raise RuntimeError(f"latest RNG incomplete {group} fold {fold}")
            if len(latest.get("trace", [])) != int(latest["next_epoch"]) - 10:
                raise RuntimeError(f"latest RNG trace length mismatch {group} fold {fold}")
            task_records[(group, fold)] = task
            checkpoints[(group, fold)] = checkpoint

    fold_rows, mechanism_rows = [], []
    values_by_group = {group: {} for group in GROUPS}
    rng_report, init_report = {}, []
    best_audits = latest_audits = 0

    for fold in range(1, 11):
        initial_tasks = [task_records[(group, fold)] for group in GROUPS]
        if initial_tasks[0]["initial_module_sha256"] != initial_tasks[1]["initial_module_sha256"]:
            raise RuntimeError(f"same-fold G1/G2 relation initialization differs at fold {fold}")
        for key in ("model_cam_sha", "optimizer_sha", "warmup_sha256", "warmup_rng_sha", "split_sha", "pair_sha"):
            if initial_tasks[0]["identity"].get(key) != initial_tasks[1]["identity"].get(key):
                raise RuntimeError(f"same-fold base identity differs at {key}, fold {fold}")
        init_report.append({"fold": fold, "same_fold_relation_initialization": True,
                            "base_initialization_equal": True,
                            "module_initialization_sha256": initial_tasks[0]["initial_module_sha256"]})

        for group in GROUPS:
            task = task_records[(group, fold)]
            checkpoint = checkpoints[(group, fold)]
            state, warmup, identity = experiment.prepare(fold)
            if identity != task["identity"]:
                raise RuntimeError(f"fold identity mismatch during reload {group} fold {fold}")
            correction = PairwiseRelationalCorrection.initialized(813000 + fold, state["x"].device)
            optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
            experiment.load_bundle(state, correction, optimizer, checkpoint["best"]["bundle"])
            state["group"] = group
            state["fold"] = fold
            state["model"].eval(); state["cam"].eval(); correction.eval()
            with torch.no_grad():
                output = experiment.forward_with_correction(state, correction, group, training=False)
                final_metrics = metric_function(output["logp_final"], state["test"], state["y"])
                base_metrics = metric_function(output["logp"], state["test"], state["y"])
            saved_final = np.asarray(checkpoint["best"]["test_probability"])
            if not np.allclose(saved_final, np.asarray(final_metrics["prob"]), rtol=0.0, atol=1e-7):
                raise RuntimeError(f"saved final probabilities differ {group} fold {fold}")
            if not np.array_equal(np.asarray(checkpoint["best"]["test_prediction"]), np.asarray(final_metrics["pred"])):
                raise RuntimeError(f"saved final predictions differ {group} fold {fold}")
            if not np.allclose(np.asarray(checkpoint["best"]["base_probability"]), np.asarray(base_metrics["prob"]), rtol=0.0, atol=1e-7):
                raise RuntimeError(f"saved base probabilities differ {group} fold {fold}")
            if not np.array_equal(np.asarray(checkpoint["best"]["base_prediction"]), np.asarray(base_metrics["pred"])):
                raise RuntimeError(f"saved base predictions differ {group} fold {fold}")

            historical = load_historical_checkpoint(base, fold)
            labels = test_labels(state)
            history_compare = compare_predictions(historical["test_prediction"], final_metrics["pred"], labels)
            current_compare = compare_predictions(base_metrics["pred"], final_metrics["pred"], labels)
            trajectory = compare_predictions(historical["test_prediction"], base_metrics["pred"], labels)
            values_by_group[group][fold] = metric_values(final_metrics)
            mech = _mechanism_row(group, fold, final_metrics, base_metrics, historical, state, output, correction)
            mechanism_rows.append(mech)
            vals = values_by_group[group][fold]
            fold_rows.append({
                "group": group, "fold": fold, **{metric: vals[metric] * 100.0 for metric in METRICS},
                "asd_tp": int(final_metrics["asd_tp"]), "asd_fn": int(final_metrics["asd_fn"]),
                "td_fp": int(final_metrics["td_fp"]), "td_tn": int(final_metrics["td_tn"]),
                "current_base_acc": base_metrics["acc"] * 100.0, "current_base_hard_auc": base_metrics["hard_auc"] * 100.0,
                "current_base_prob_auc": base_metrics["prob_auc"] * 100.0, "current_base_sen": base_metrics["sen"] * 100.0,
                "current_base_spe": base_metrics["spe"] * 100.0, "current_base_asd_f1": base_metrics["asd_f1"] * 100.0,
                "best_epoch_1based": task["best_epoch_1based"], "epochs_executed": task["epochs_executed"],
                "wall_seconds": task["wall_seconds"], "peak_gpu_bytes": task["peak_gpu_bytes"],
                "corrected_vs_darg": history_compare["corrected"], "harmed_vs_darg": history_compare["harmed"],
                "net_vs_darg": history_compare["net"], "flips_vs_darg": history_compare["flips"],
                "corrected_vs_current_base": current_compare["corrected"], "harmed_vs_current_base": current_compare["harmed"],
                "net_vs_current_base": current_compare["net"], "current_base_corrected_vs_darg": trajectory["corrected"],
                "current_base_harmed_vs_darg": trajectory["harmed"], "checkpoint_sha256": task["best_checkpoint_sha256"],
                "latest_sha256": task["latest_checkpoint_sha256"], "train_count": identity["membership"]["effective_train_count"],
                "test_count": identity["membership"]["test_count"], "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"]),
                "independent_checkpoint_audit": "PASS",
            })
            best_audits += 1
            latest_audits += 1

            e74_task = json.loads((base.EXP / ".runtime/tasks" / f"{BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
            historical_trace = {entry["epoch"]: entry for entry in e74_task["rng_trace"]}
            current_trace = {entry["epoch"]: entry for entry in task["rng_trace"]}
            common = sorted(set(historical_trace) & set(current_trace))
            exact = sum(all(historical_trace[e][key] == current_trace[e][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
            rng_report.setdefault(group, {"common_epochs": 0, "exact_triplets": 0, "mismatches": 0, "per_fold": {}})
            rng_report[group]["common_epochs"] += len(common); rng_report[group]["exact_triplets"] += exact; rng_report[group]["mismatches"] += len(common) - exact
            rng_report[group]["per_fold"][str(fold)] = {"common_epochs": len(common), "exact": exact, "mismatches": len(common) - exact}
            del state, correction, optimizer, output
            torch.cuda.empty_cache()

        left_trace = {entry["epoch"]: entry for entry in task_records[(GROUPS[0], fold)]["rng_trace"]}
        right_trace = {entry["epoch"]: entry for entry in task_records[(GROUPS[1], fold)]["rng_trace"]}
        common = sorted(set(left_trace) & set(right_trace))
        exact = sum(all(left_trace[e][key] == right_trace[e][key] for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
        rng_report.setdefault("between_groups", {"common_epochs": 0, "exact_triplets": 0, "mismatches": 0, "per_fold": {}})
        rng_report["between_groups"]["common_epochs"] += len(common); rng_report["between_groups"]["exact_triplets"] += exact; rng_report["between_groups"]["mismatches"] += len(common) - exact
        rng_report["between_groups"]["per_fold"][str(fold)] = {"common_epochs": len(common), "exact": exact, "mismatches": len(common) - exact}

    values_by_group[BASE_GROUP] = historical_darg
    values_by_group["E81.2-G2-Cross-Local-E2E"] = e812_g2
    for name, values in ((BASE_GROUP, historical_darg), ("E81.2-G2-Cross-Local-E2E", e812_g2)):
        ref_csv = darg_csv if name == BASE_GROUP else e812_csv
        row_map = {int(row["fold"]): row for row in ref_csv if row["group"] == name}
        for fold in range(1, 11):
            fold_rows.append({"group": name, "fold": fold, **{metric: values[fold][metric] * 100.0 for metric in METRICS},
                              "best_epoch_1based": row_map[fold].get("best_epoch_1based", ""),
                              "epochs_executed": row_map[fold].get("epochs_executed", ""),
                              "independent_checkpoint_audit": "READ_ONLY_REFERENCE"})

    comparisons = paired_rows({GROUPS[0]: values_by_group[GROUPS[0]], GROUPS[1]: values_by_group[GROUPS[1]]}, references)
    summary = _summary(values_by_group)
    write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)

    rng_mismatch = sum(value["mismatches"] for value in rng_report.values())
    status = {
        "status": "PASS" if best_audits == 20 and latest_audits == 20 and rng_mismatch == 0 else "FAIL",
        "training_source_sha256": training_source_sha,
        "formal_tasks_complete": sum(task_records[(g, f)]["status"] == "COMPLETE" for g in GROUPS for f in range(1, 11)),
        "best_checkpoint_independent_pass": best_audits, "latest_state_integrity_pass": latest_audits,
        "base_off_parity_pass": preflight["base_off_parity"]["passed"], "rng_audit": rng_report,
        "same_fold_module_initialization_equal": init_report, "original_optimizers_restored_and_latest_checked": True,
        "independent_recomputed_summary": summary, "source_sha256": current_sha,
        "fold2_overlap_verified": {str(f): task_records[(GROUPS[0], f)]["identity"]["membership"]["train_test_overlap_1based"] for f in range(1, 11)},
        "audit_seconds": time.monotonic() - started,
        "mechanism_intervention": "G2 frozen inference unary Key/Value replacement using G2 projection parameters",
        "evaluation_limitations": [
            "validation equals test and test ACC selects best checkpoints",
            "pre-CV preprocessing/RFE remains global and the 871-patient graph is transductive",
            "Fold 2 rows 377 and 407 remain in historical DARG training; new correction is forced to zero during training",
            "fold intervals are descriptive because training folds overlap",
            "CAM144 16 coordinate slices are latent coordinates, not anatomical ROIs",
            "G1 is a real 15-neighbor cross-local attention control, not a no-interaction baseline",
        ],
    }
    experiment.atomic_json(RESULTS / "audit_status.json", status)
    if status["status"] != "PASS":
        raise RuntimeError("independent audit status failed")
    report = render_report(summary, comparisons, mechanism_rows, status)
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    return status


def render_report(summary, comparisons, mechanisms, status):
    labels = {"acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}
    groups = (*GROUPS, BASE_GROUP, "E81.2-G2-Cross-Local-E2E")
    g2 = summary[GROUPS[1]]; darg = summary[BASE_GROUP]; g1 = summary[GROUPS[0]]
    g2g1 = next(row for row in comparisons if row["comparison"] == "G2_vs_G1" and row["scope"] == "all10" and row["metric"] == "acc")
    g2darg = next(row for row in comparisons if row["comparison"] == "G2_vs_DARG" and row["scope"] == "all10" and row["metric"] == "acc")
    hard_g2 = next(row for row in comparisons if row["comparison"] == "G2_vs_DARG" and row["scope"] == "all10" and row["metric"] == "hard_auc")
    meets = g2["acc"]["mean"] >= 92.1891 and g2["hard_auc"]["mean"] >= 92.0294
    if not meets:
        decision = "NO_GO_E81_3"
    elif g2g1["mean_difference_pp"] > 1e-9:
        decision = "PRELIMINARY_PAIRWISE_RELATIONAL_GAIN"
    else:
        decision = "BASELINE_PRESERVED_RELATIONAL_GAIN_NOT_ESTABLISHED"
    lines = [
        "# Experiment 81.3 — End-to-End Gradient-Protected Pairwise Relational Learning", "",
        f"Status: {status['status']}; formal training {status['formal_tasks_complete']}/20; independent best-checkpoint audits {status['best_checkpoint_independent_pass']}/20; latest-state audits {status['latest_state_integrity_pass']}/20.",
        f"Formal directory: `{HERE}`. Training wall time: {json.loads((RUNTIME/'training_summary.json').read_text())['wall_seconds']:.1f} s.",
        "The paper and official/project repositories were checked before implementation. The original GFR/CAM/PEN description and 10-fold ACC/AUC/SEN/SPE reporting were retained for provenance; the formal model here remains the historical PEN-free E74.4 CAM/DARG base.",
        "", "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for group in groups:
        row = summary[group]
        lines.append(f"| {group} | " + " | ".join(f"{row[m]['mean']:.4f} ± {row[m]['population_sd']:.4f}" for m in METRICS) + " |")
    lines += ["", "## Paired comparisons", "", "Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across overlapping folds.", "", "| Comparison | Scope | Metric | Mean Δ (95% interval) | W/T/L |", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['scope']} | {labels[row['metric']]} | {row['mean_difference_pp']:+.4f} ({row['ci95_low_pp']:+.4f}, {row['ci95_high_pp']:+.4f}) | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## Mechanism and gradient protection", "", "The independent audit separates the trained E81.3 base OutputLayer branch, its final corrected branch, and the historical E74.4-G2 checkpoint. The correction branch uses eta=0.1; the preflight eta switch is diagnostic only.", "", "| Group | mean |delta| | P95 | Nonzero | Prob MAE vs base | Flips vs base | Corrected/harmed vs DARG | Null identity | Readout entropy |", "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for group in GROUPS:
        rows = [row for row in mechanisms if row["group"] == group]
        avg = lambda key: float(np.mean([float(row[key]) for row in rows]))
        fixed = sum(json.loads(row["historical_darg_to_final_corrected_harmed_net"])["corrected"] for row in rows)
        harmed = sum(json.loads(row["historical_darg_to_final_corrected_harmed_net"])["harmed"] for row in rows)
        null_ok = all(str(row["null_only_strict_identity"]) == "True" for row in rows)
        entropy = np.mean([json.loads(row["attention_summary"])["readout_entropy_mean"] for row in rows])
        lines.append(f"| {group} | {avg('delta_margin_abs_mean'):.6f} | {avg('delta_margin_abs_p95'):.6f} | {avg('nonzero_correction_fraction'):.4f} | {avg('corrected_probability_mae_vs_current_base'):.6f} | {avg('hard_flips_vs_current_base'):.2f} | {fixed}/{harmed} | {null_ok} | {entropy:.4f} |")
    lines += ["", "### Gate and patient-conditioned readout diagnostics", "", "These are descriptive diagnostics; the gate is a learned correction-control signal and the local weights are not anatomical ROI importance.", "", "| Group | Gate mean (P05–P95) | Null weight mean | Attn entropy | Top real weight | Readout entropy | Readout max | High-confidence errors corrected |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        rows = [row for row in mechanisms if row["group"] == group]
        attention = [json.loads(row["attention_summary"]) for row in rows]
        gate_mean = np.mean([float(row["gate_mean"]) for row in rows])
        gate_p05 = np.mean([float(row["gate_p05"]) for row in rows])
        gate_p95 = np.mean([float(row["gate_p95"]) for row in rows])
        null_mean = np.mean([item["null_mean"] for item in attention])
        att_entropy = np.mean([item["normalized_entropy_mean"] for item in attention])
        top_real = np.mean([item["top_real_weight_mean"] for item in attention])
        readout_entropy = np.mean([item["readout_entropy_mean"] for item in attention])
        readout_max = np.mean([item["readout_max_mean"] for item in attention])
        high_conf_total = sum(int(row["high_confidence_base_error_count"]) for row in rows)
        high_conf_corrected = sum(int(row["high_confidence_error_corrected"]) for row in rows)
        lines.append(f"| {group} | {gate_mean:.4f} ({gate_p05:.4f}–{gate_p95:.4f}) | {null_mean:.4f} | {att_entropy:.4f} | {top_real:.4f} | {readout_entropy:.4f} | {readout_max:.4f} | {high_conf_corrected}/{high_conf_total} |")
    g2_interventions = [json.loads(row["pairwise_unary_inference_intervention"]) for row in mechanisms if row["group"] == GROUPS[1] and row["pairwise_unary_inference_intervention"]]
    if g2_interventions:
        lines += ["", f"G2 frozen unary-input intervention (using G2 projection weights): probability MAE {np.mean([x['probability_mae'] for x in g2_interventions]):.6f}, hard flips {np.mean([x['hard_prediction_flips'] for x in g2_interventions]):.2f} per fold, ACC change {np.mean([x['acc_change_pp'] for x in g2_interventions]):+.4f} pp, Hard-AUC change {np.mean([x['hard_auc_change_pp'] for x in g2_interventions]):+.4f} pp, ASD-F1 change {np.mean([x['asd_f1_change_pp'] for x in g2_interventions]):+.4f} pp, relation-summary L2 change {np.mean([x['relation_summary_l2_change_mean'] for x in g2_interventions]):.4f}."]
    lines += ["", "The complete patient-level diagnostics are in `mechanism.csv`. G2 unary replacement is a frozen inference intervention using G2's own projection weights and is not an independently trained control.", "", "## Excluding Fold 2", "", "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        row = summary[group]
        lines.append(f"| {group} | " + " | ".join(f"{row[m]['without_fold2_mean']:.4f} ± {row[m]['without_fold2_population_sd']:.4f}" for m in METRICS) + " |")
    lines += ["", "## Decision", "", f"- G2 performance floor: ACC {g2['acc']['mean']:.4f}% (threshold 92.1891%), Hard-AUC {g2['hard_auc']['mean']:.4f}% (threshold 92.0294%).", f"- G2−G1 ACC: {g2g1['mean_difference_pp']:+.4f} pp, W/T/L {g2g1['wins']}/{g2g1['ties']}/{g2g1['losses']}; G2−DARG ACC: {g2darg['mean_difference_pp']:+.4f} pp; G2−DARG Hard-AUC: {hard_g2['mean_difference_pp']:+.4f} pp.", f"- Scientific classification: **{decision}**.", "- This experiment contains no text Prompt, disease prototype, or new modality and therefore does not establish medical Prompt effectiveness.", "", "## Audit limits", "", "Historical selection uses test-fold ACC (validation equals test); preprocessing/RFE precede CV; the patient graph is transductive; Fold 2 retains the historical overlap while suppressing only the new correction during training. Fold intervals are descriptive because training sets overlap. Latent CAM coordinate slices are not anatomical regions.", "", "## Artifacts", "", f"Formal training source SHA: `{status['training_source_sha256']}`; corrected audit/source SHA: `{status['source_sha256']}`. Code: `relation_learning.py`, `run.py`, `audit.py`, `path_adapter.py`; results: `results/preflight.json`, `results/fold_metrics.csv`, `results/paired_comparisons.csv`, `results/mechanism.csv`, `results/audit_status.json`, `results/report.md`."]
    return "\n".join(lines) + "\n"


def main():
    try:
        result = audit()
        print(json.dumps({"status": result["status"], "best": result["best_checkpoint_independent_pass"], "latest": result["latest_state_integrity_pass"]}, ensure_ascii=False))
    except Exception as exc:
        failure = {"status": "FAIL", "error": repr(exc), "source_sha256": experiment.source_sha(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        experiment.atomic_json(RESULTS / "audit_status.json", failure)
        raise


if __name__ == "__main__":
    main()
