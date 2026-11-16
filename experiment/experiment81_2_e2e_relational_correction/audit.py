#!/usr/bin/env python3
"""Independent checkpoint replay, paired metrics, and frozen mechanism audit."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import run as experiment
from relation_correction import LocalCorrection

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNTIME = HERE / "runtime"
CHECKPOINTS = RUNTIME / "checkpoints"
TASKS = RUNTIME / "tasks"
GROUPS = experiment.GROUPS
METRICS = experiment.METRICS


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
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        __import__('os').fsync(stream.fileno())
    __import__('os').replace(temp, path)


def metrics_fn():
    return experiment.base_runtime().classification_metrics


def _metric_values(record):
    return {key: float(record[key]) for key in METRICS}


def _test_values(state):
    labels = state["y"].detach().cpu().numpy()
    return labels[np.asarray(state["test"], dtype=np.int64)].astype(np.int64)


def _compare_predictions(reference, candidate, y):
    ref = np.asarray(reference, dtype=np.int64)
    new = np.asarray(candidate, dtype=np.int64)
    labels = np.asarray(y, dtype=np.int64)
    fixed = int(((ref != labels) & (new == labels)).sum())
    harmed = int(((ref == labels) & (new != labels)).sum())
    return {"corrected": fixed, "harmed": harmed, "net": fixed - harmed,
            "flips": int((ref != new).sum())}


def _mean_sd(values):
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _paired_rows(values_by_group, reference_rows):
    comparisons = [
        ("G1_vs_DARG", GROUPS[0], "E74.4-G2-Positive-DCR"),
        ("G2_vs_DARG", GROUPS[1], "E74.4-G2-Positive-DCR"),
        ("G2_vs_G1", GROUPS[1], GROUPS[0]),
        ("G2_vs_E81.1_G1_context", GROUPS[1], "E81.1-G1-Visual-PRCL-v2"),
    ]
    output = []
    fold_range = range(1, 11)
    for label, left_name, right_name in comparisons:
        if right_name in values_by_group:
            left_values = values_by_group[left_name]
            right_values = values_by_group[right_name]
        else:
            left_values = values_by_group[left_name]
            right_values = reference_rows[right_name]
        for scope, folds in (("all10", list(fold_range)), ("without_fold2", [f for f in fold_range if f != 2])):
            for metric in METRICS:
                diffs = np.asarray([left_values[fold][metric] - right_values[fold][metric] for fold in folds], dtype=float)
                n = len(diffs)
                mean = float(diffs.mean()) * 100.0
                sd = float(diffs.std(ddof=1)) * 100.0 if n > 1 else 0.0
                tcrit = 2.262157 if n == 10 else 2.306004
                half = tcrit * sd / math.sqrt(n)
                win = int((diffs > 1e-12).sum())
                tie = int((np.abs(diffs) <= 1e-12).sum())
                loss = n - win - tie
                output.append({
                    "comparison": label, "left": left_name, "right": right_name,
                    "scope": scope, "metric": metric, "mean_difference_pp": mean,
                    "sd_difference_pp": sd, "ci95_low_pp": mean - half,
                    "ci95_high_pp": mean + half, "wins": win, "ties": tie, "losses": loss,
                    "fold_differences_pp": json.dumps((diffs * 100.0).round(6).tolist()),
                })
    return output


def _load_historical_reference(base, fold):
    path = base.EXP / ".runtime/checkpoints" / f"{experiment.BASE_GROUP}_f{fold:02d}.best.pt"
    payload = experiment.torch_load(path)
    if payload.get("group") != experiment.BASE_GROUP or payload.get("fold") != fold:
        raise RuntimeError(f"historical DARG checkpoint identity mismatch for fold {fold}")
    return payload


def _attention_summary(details, test):
    test_tensor = torch.as_tensor(test, dtype=torch.long, device=details["attention"].device)
    attention = details["attention"][test_tensor]
    null = attention[..., -1]
    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1) / math.log(16.0)
    real_max = attention[..., :15].max(dim=-1).values
    per_patient = attention[..., :15].mean(dim=1).mean(dim=1)
    return {
        "null_mean": float(null.mean().cpu()),
        "null_dominant_fraction": float((null >= real_max).float().mean().cpu()),
        "nonnull_mass_mean": float((1.0 - null).mean().cpu()),
        "normalized_entropy_mean": float(entropy.mean().cpu()),
        "top_real_weight_mean": float(real_max.mean().cpu()),
        "patient_relation_distribution_sd": float(per_patient.std(dim=0, unbiased=False).mean().cpu()),
    }


def _mechanism_row(group, fold, metrics, base_metrics, historical, state, output, correction):
    test = state["test"]
    y = _test_values(state)
    final_pred = np.asarray(metrics["pred"], dtype=np.int64)
    current_pred = np.asarray(base_metrics["pred"], dtype=np.int64)
    old_pred = np.asarray(historical["test_prediction"], dtype=np.int64)
    compare_old = _compare_predictions(old_pred, final_pred, y)
    compare_current = _compare_predictions(current_pred, final_pred, y)
    base_trajectory = _compare_predictions(old_pred, current_pred, y)
    details = output["correction_details"]
    test_tensor = torch.as_tensor(test, dtype=torch.long, device=output["logp_base"].device)
    delta = details["delta_margin"][test_tensor].detach()
    unc = details["g_unc"][test_tensor].detach()
    rel = details["g_rel"][test_tensor].detach()
    margins = (output["logp_base"][:, 1] - output["logp_base"][:, 0]).abs()[test_tensor].detach()
    attentions = _attention_summary(details, test)

    null_logp, null_details = correction(
        output["z_cam"], output["logp_base"], experiment.GROUP_MODE[group],
        training=False, intervention="null_only",
    )
    null_exact = bool(torch.equal(null_logp, output["logp_base"]) and
                      torch.equal(null_details["messages"], torch.zeros_like(null_details["messages"])))
    null_probability_mae = float((null_logp[test_tensor].exp() - output["logp_base"][test_tensor].exp()).abs().mean().cpu())

    self_intervention = None
    if group == GROUPS[1]:
        self_logp, self_details = correction(
            output["z_cam"], output["logp_base"], "self", training=False,
        )
        self_metrics = metrics_fn()(self_logp, test, state["y"])
        self_intervention = {
            "self_metrics": _metric_values(self_metrics),
            "cross_to_self_probability_mae": float((self_logp[test_tensor].exp() - output["logp"][test_tensor].exp()).abs().mean().cpu()),
            "cross_to_self_hard_flips": int((self_metrics["pred"] != metrics["pred"]).sum()),
            "relation_summary_l2_change_mean": float(torch.linalg.vector_norm(
                details["relation_summary"][test_tensor] - self_details["relation_summary"][test_tensor], dim=-1
            ).mean().cpu()),
        }

    median = torch.median(margins)
    low = margins <= median
    high = ~low
    groups = {}
    for name, mask in (("lower_margin_half", low), ("upper_margin_half", high)):
        ix = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
        groups[name] = {
            "n": int(len(ix)),
            "mean_abs_correction": float(delta[mask].abs().mean().cpu()) if len(ix) else 0.0,
            "mean_g_unc": float(unc[mask].mean().cpu()) if len(ix) else 0.0,
            "mean_g_rel": float(rel[mask].mean().cpu()) if len(ix) else 0.0,
            "corrected_vs_current": _compare_predictions(current_pred[ix], final_pred[ix], y[ix]),
        }

    return {
        "group": group, "fold": fold,
        "current_base_acc_pct": base_metrics["acc"] * 100.0,
        "corrected_acc_pct": metrics["acc"] * 100.0,
        "historical_darg_acc_pct": historical["metrics"]["acc"] * 100.0,
        "current_base_vs_historical_corrected_harmed_net": json.dumps(base_trajectory),
        "historical_darg_to_final_corrected_harmed_net": json.dumps(compare_old),
        "current_base_to_final_corrected_harmed_net": json.dumps(compare_current),
        "corrected_probability_mae_vs_current_base": float((output["logp"][test_tensor].exp() - output["logp_base"][test_tensor].exp()).abs().mean().cpu()),
        "hard_flips_vs_current_base": int((final_pred != current_pred).sum()),
        "nonzero_correction_fraction": float((delta.abs() > 1e-8).float().mean().cpu()),
        "delta_margin_abs_mean": float(delta.abs().mean().cpu()),
        "delta_margin_abs_p95": float(torch.quantile(delta.abs(), 0.95).cpu()),
        "delta_margin_mean": float(delta.mean().cpu()),
        "g_unc_mean": float(unc.mean().cpu()),
        "g_rel_mean": float(rel.mean().cpu()),
        "base_margin_median": float(median.cpu()),
        "uncertainty_strata": json.dumps(groups),
        "null_only_strict_identity": null_exact,
        "null_only_probability_mae": null_probability_mae,
        "attention_summary": json.dumps(attentions),
        "cross_to_self_inference_intervention": json.dumps(self_intervention) if self_intervention else "",
    }


def audit():
    started = time.monotonic()
    frozen = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if frozen.get("status") != "PREFLIGHT_PASS":
        raise RuntimeError("frozen preflight manifest is missing or invalid")
    current_sha = experiment.source_sha()
    if current_sha != frozen.get("source_sha256"):
        raise RuntimeError("training/audit source SHA differs from frozen preflight")
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("base_off_parity", {}).get("passed") != 10:
        raise RuntimeError("10/10 base-off parity was not recorded")

    base = experiment.runtime()
    hist = base.load_history()
    metric_function = metrics_fn()
    historical_csv = read_csv(Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/results/fold_metrics.csv"))
    historical_darg = {int(row["fold"]): {metric: float(row[metric]) for metric in METRICS}
                       for row in historical_csv if row["group"] == experiment.BASE_GROUP}
    e81_1_csv = read_csv(Path("/model/spromptgl/experiment/experiment81_1_prcl_v2/results/fold_metrics.csv"))
    e81_1_g1 = {int(row["fold"]): {metric: float(row[metric]) / 100.0 for metric in METRICS}
                for row in e81_1_csv if row["group"] == "E81.1-G1-Visual-PRCL-v2"}
    reference_rows = {"E74.4-G2-Positive-DCR": historical_darg,
                      "E81.1-G1-Visual-PRCL-v2": e81_1_g1}

    task_records = {}
    checkpoints = {}
    for group in GROUPS:
        for fold in range(1, 11):
            task_path, best_path, latest_path, _ = experiment.paths(group, fold)
            task = json.loads(task_path.read_text(encoding="utf-8"))
            checkpoint = experiment.torch_load(best_path)
            latest = experiment.torch_load(latest_path)
            if task.get("status") != "COMPLETE" or task.get("source_sha256") != current_sha:
                raise RuntimeError(f"task ledger incomplete or source mismatch: {group} fold {fold}")
            if task.get("best_checkpoint_sha256") != experiment.sha_file(best_path):
                raise RuntimeError(f"best checkpoint SHA mismatch: {group} fold {fold}")
            if task.get("latest_checkpoint_sha256") != experiment.sha_file(latest_path):
                raise RuntimeError(f"latest checkpoint SHA mismatch: {group} fold {fold}")
            if latest.get("source_sha256") != current_sha or latest.get("group") != group or latest.get("fold") != fold:
                raise RuntimeError(f"latest state identity mismatch: {group} fold {fold}")
            required = {"model", "cam", "opt_base", "opt_cam", "opt_added"}
            if not required.issubset(latest["current"]["base"]):
                raise RuntimeError(f"latest base state is missing an original optimizer: {group} fold {fold}")
            if not {"correction", "optimizer_correction"}.issubset(latest["current"]):
                raise RuntimeError(f"latest correction module/optimizer is missing: {group} fold {fold}")
            if not {"python", "numpy", "cpu"}.issubset(latest["rng"]):
                raise RuntimeError(f"latest RNG state is incomplete: {group} fold {fold}")
            if len(latest["trace"]) != int(latest["next_epoch"]) - 10:
                raise RuntimeError(f"latest RNG trace length mismatch: {group} fold {fold}")
            task_records[(group, fold)] = task
            checkpoints[(group, fold)] = checkpoint

    best_audits = 0
    latest_audits = 0
    fold_rows = []
    mechanism_rows = []
    values_by_group = {group: {} for group in GROUPS}
    rng_report = {}
    init_report = []

    for fold in range(1, 11):
        group_initials = [task_records[(group, fold)] for group in GROUPS]
        if group_initials[0]["initial_module_sha256"] != group_initials[1]["initial_module_sha256"]:
            raise RuntimeError(f"same-fold correction initialization differs across groups: fold {fold}")
        for key in ("model_cam_sha", "optimizer_sha", "warmup_sha256", "warmup_rng_sha", "split_sha", "pair_sha"):
            if group_initials[0]["identity"].get(key) != group_initials[1]["identity"].get(key):
                raise RuntimeError(f"base initialization differs across groups at fold {fold}/{key}")
        init_report.append({"fold": fold, "group_module_initialization_equal": True,
                            "base_initialization_equal": True,
                            "module_initialization_sha256": group_initials[0]["initial_module_sha256"]})

        for group in GROUPS:
            task = task_records[(group, fold)]
            checkpoint = checkpoints[(group, fold)]
            state, warmup, identity = experiment.prepare(fold)
            if identity != task["identity"]:
                raise RuntimeError(f"fold identity mismatch during independent reload: {group} fold {fold}")
            correction = LocalCorrection.initialized(812000 + fold, state["x"].device)
            optimizer = torch.optim.AdamW(correction.parameters(), lr=0.0005, weight_decay=0.01)
            experiment.load_bundle(state, correction, optimizer, checkpoint["best"]["bundle"])
            state["group"] = group
            state["fold"] = fold
            state["rudag_identity"] = identity
            state["model"].eval()
            state["cam"].eval()
            correction.eval()
            with torch.no_grad():
                out = experiment.forward_with_correction(state, correction, group, training=False)
                metrics = metric_function(out["logp"], state["test"], state["y"])
                base_metrics = metric_function(out["logp_base"], state["test"], state["y"])
            saved_probability = np.asarray(checkpoint["best"]["test_probability"])
            recomputed_probability = np.asarray(metrics["prob"])
            if saved_probability.shape != recomputed_probability.shape or not np.allclose(saved_probability, recomputed_probability, rtol=0.0, atol=1e-7):
                raise RuntimeError(f"independent final probabilities differ: {group} fold {fold}")
            if not np.array_equal(np.asarray(checkpoint["best"]["test_prediction"]), np.asarray(metrics["pred"])):
                raise RuntimeError(f"independent final hard predictions differ: {group} fold {fold}")
            if not np.allclose(np.asarray(checkpoint["best"]["base_probability"]), np.asarray(base_metrics["prob"]), rtol=0.0, atol=1e-7):
                raise RuntimeError(f"independent current-base probabilities differ: {group} fold {fold}")
            if not np.array_equal(np.asarray(checkpoint["best"]["base_prediction"]), np.asarray(base_metrics["pred"])):
                raise RuntimeError(f"independent current-base hard predictions differ: {group} fold {fold}")

            old = _load_historical_reference(base, fold)
            y = _test_values(state)
            history_compare = _compare_predictions(old["test_prediction"], metrics["pred"], y)
            current_compare = _compare_predictions(base_metrics["pred"], metrics["pred"], y)
            trained_base_vs_old = _compare_predictions(old["test_prediction"], base_metrics["pred"], y)
            vals = _metric_values(metrics)
            base_vals = _metric_values(base_metrics)
            values_by_group[group][fold] = vals
            for metric in METRICS:
                values_by_group.setdefault("E74.4-G2-Positive-DCR", {})[fold] = historical_darg[fold]
                values_by_group.setdefault("E81.1-G1-Visual-PRCL-v2", {})[fold] = e81_1_g1[fold]
            row = {
                "group": group, "fold": fold,
                **{metric: vals[metric] * 100.0 for metric in METRICS},
                "asd_tp": int(metrics["asd_tp"]), "asd_fn": int(metrics["asd_fn"]),
                "td_fp": int(metrics["td_fp"]), "td_tn": int(metrics["td_tn"]),
                "current_base_acc": base_vals["acc"] * 100.0,
                "current_base_hard_auc": base_vals["hard_auc"] * 100.0,
                "current_base_prob_auc": base_vals["prob_auc"] * 100.0,
                "current_base_sen": base_vals["sen"] * 100.0,
                "current_base_spe": base_vals["spe"] * 100.0,
                "current_base_asd_f1": base_vals["asd_f1"] * 100.0,
                "best_epoch_1based": task["best_epoch_1based"],
                "epochs_executed": task["epochs_executed"],
                "wall_seconds": task["wall_seconds"],
                "peak_gpu_bytes": task["peak_gpu_bytes"],
                "corrected_vs_darg": history_compare["corrected"],
                "harmed_vs_darg": history_compare["harmed"],
                "net_vs_darg": history_compare["net"],
                "flips_vs_darg": history_compare["flips"],
                "corrected_vs_current_base": current_compare["corrected"],
                "harmed_vs_current_base": current_compare["harmed"],
                "net_vs_current_base": current_compare["net"],
                "current_base_corrected_vs_darg": trained_base_vs_old["corrected"],
                "current_base_harmed_vs_darg": trained_base_vs_old["harmed"],
                "checkpoint_sha256": task["best_checkpoint_sha256"],
                "latest_sha256": task["latest_checkpoint_sha256"],
                "train_count": identity["membership"]["effective_train_count"],
                "test_count": identity["membership"]["test_count"],
                "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"]),
                "independent_checkpoint_audit": "PASS",
            }
            fold_rows.append(row)
            mechanism_rows.append(_mechanism_row(group, fold, metrics, base_metrics, old, state, out, correction))
            best_audits += 1
            latest_audits += 1

            e74_task = json.loads((base.EXP / ".runtime/tasks" / f"{experiment.BASE_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
            hist_trace = {item["epoch"]: item for item in e74_task["rng_trace"]}
            current_trace = {item["epoch"]: item for item in task["rng_trace"]}
            common = sorted(set(hist_trace) & set(current_trace))
            exact = sum(all(hist_trace[e][key] == current_trace[e][key]
                            for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common)
            mismatch = len(common) - exact
            rng_report.setdefault(group, {"common_epochs": 0, "exact_triplets": 0, "mismatches": 0, "per_fold": {}})
            rng_report[group]["common_epochs"] += len(common)
            rng_report[group]["exact_triplets"] += exact
            rng_report[group]["mismatches"] += mismatch
            rng_report[group]["per_fold"][str(fold)] = {"common_epochs": len(common), "exact": exact, "mismatches": mismatch}

            del state, correction, optimizer, out
            torch.cuda.empty_cache()

        left_trace = {item["epoch"]: item for item in task_records[(GROUPS[0], fold)]["rng_trace"]}
        right_trace = {item["epoch"]: item for item in task_records[(GROUPS[1], fold)]["rng_trace"]}
        common_groups = sorted(set(left_trace) & set(right_trace))
        group_exact = sum(all(left_trace[e][key] == right_trace[e][key]
                              for key in ("rng_before", "rng_after_train", "rng_after_eval")) for e in common_groups)
        rng_report.setdefault("between_groups", {"common_epochs": 0, "exact_triplets": 0, "mismatches": 0, "per_fold": {}})
        rng_report["between_groups"]["common_epochs"] += len(common_groups)
        rng_report["between_groups"]["exact_triplets"] += group_exact
        rng_report["between_groups"]["mismatches"] += len(common_groups) - group_exact
        rng_report["between_groups"]["per_fold"][str(fold)] = {
            "common_epochs": len(common_groups), "exact": group_exact,
            "mismatches": len(common_groups) - group_exact,
        }

    # Add historical rows to the fold table so all six metrics share one artifact.
    for name, values in (("E74.4-G2-Positive-DCR", historical_darg),
                         ("E81.1-G1-Visual-PRCL-v2", e81_1_g1)):
        ref_table = read_csv(Path("/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/results/fold_metrics.csv")) if name == "E74.4-G2-Positive-DCR" else e81_1_csv
        row_map = {int(row["fold"]): row for row in ref_table if row["group"] == name}
        for fold in range(1, 11):
            fold_rows.append({"group": name, "fold": fold,
                              **{metric: values[fold][metric] * 100.0 for metric in METRICS},
                              "best_epoch_1based": row_map[fold].get("best_epoch_1based", ""),
                              "epochs_executed": row_map[fold].get("epochs_executed", ""),
                              "independent_checkpoint_audit": row_map[fold].get("status", "PASS")})

    comparison_rows = _paired_rows(values_by_group, reference_rows)
    write_csv(RESULTS / "fold_metrics.csv", fold_rows)
    write_csv(RESULTS / "paired_comparisons.csv", comparison_rows)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)

    summary = {}
    for group in (*GROUPS, "E74.4-G2-Positive-DCR", "E81.1-G1-Visual-PRCL-v2"):
        data = values_by_group[group]
        summary[group] = {}
        for metric in METRICS:
            mean, sd = _mean_sd([data[fold][metric] * 100.0 for fold in range(1, 11)])
            nine_mean, nine_sd = _mean_sd([data[fold][metric] * 100.0 for fold in range(1, 11) if fold != 2])
            summary[group][metric] = {"mean": mean, "population_sd": sd,
                                      "without_fold2_mean": nine_mean, "without_fold2_population_sd": nine_sd}

    status = {
        "status": "PASS" if best_audits == 20 and latest_audits == 20 and
                  all(row["mismatches"] == 0 for key, row in rng_report.items() if key != "between_groups") and
                  rng_report["between_groups"]["mismatches"] == 0 else "FAIL",
        "formal_tasks_complete": sum(task_records[(g, f)]["status"] == "COMPLETE" for g in GROUPS for f in range(1, 11)),
        "best_checkpoint_independent_pass": best_audits,
        "latest_state_integrity_pass": latest_audits,
        "base_off_parity_pass": preflight["base_off_parity"]["passed"],
        "rng_audit": rng_report,
        "same_fold_module_initialization_equal": init_report,
        "original_optimizers_restored_and_latest_checked": True,
        "independent_recomputed_summary": summary,
        "fold2_overlap_verified": {str(fold): task_records[(GROUPS[0], fold)]["identity"]["membership"]["train_test_overlap_1based"] for fold in range(1, 11)},
        "source_sha256": current_sha,
        "audit_seconds": time.monotonic() - started,
        "evaluation_limitations": [
            "validation equals test and test ACC selects best checkpoints",
            "pre-CV preprocessing/RFE remains global and the patient graph is transductive",
            "Fold 2 rows 377 and 407 remain in original DARG training and are excluded only from direct correction during training",
            "ten-fold uncertainty intervals are descriptive because training sets overlap",
            "16 CAM latent coordinate slices are not anatomical ROIs",
            "G1 repeats its receiver as 15 candidates, changing the Null competition prior",
        ],
    }
    experiment.atomic_json(RESULTS / "audit_status.json", status)
    if status["status"] != "PASS":
        raise RuntimeError("independent audit status did not pass")
    report = _render_report(summary, comparison_rows, mechanism_rows, status)
    report_path = RESULTS / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return status


def _render_report(summary, comparisons, mechanisms, status):
    labels = {"acc": "ACC", "hard_auc": "Hard-AUC", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}
    groups = (*GROUPS, "E74.4-G2-Positive-DCR", "E81.1-G1-Visual-PRCL-v2")
    lines = [
        "# Experiment 81.2: End-to-End Reliability-Gated Relational Correction",
        "",
        f"Status: {status['status']}; formal training {status['formal_tasks_complete']}/20; independent best checkpoint audits {status['best_checkpoint_independent_pass']}/20; latest state audits {status['latest_state_integrity_pass']}/20.",
        f"Formal directory: `{HERE}`. Training wall time: {json.loads((RUNTIME / 'training_summary.json').read_text())['wall_seconds']:.1f} s.",
        "All six metrics are percentages, reported as ten-fold mean ± population SD. Hard-AUC is historical balanced accuracy from hard predictions; Prob-AUC is probability ROC-AUC.",
        "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        row = summary[group]
        values = [f"{row[key]['mean']:.4f} ± {row[key]['population_sd']:.4f}" for key in METRICS]
        lines.append(f"| {group} | " + " | ".join(values) + " |")
    lines += ["", "## Paired comparisons", "", "Differences are left minus right in percentage points. W/T/L and t-based 95% intervals are descriptive across folds.",
              "", "| Comparison | Scope | Metric | Mean Δ (95% interval) | W/T/L |", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['scope']} | {labels[row['metric']]} | {row['mean_difference_pp']:+.4f} ({row['ci95_low_pp']:+.4f}, {row['ci95_high_pp']:+.4f}) | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## Mechanism", "", "Frozen best-checkpoint audits distinguish the current trained base OutputLayer branch, its corrected branch, and the historical E74.4-G2 checkpoint.",
              "", "| Group | mean |delta| | Nonzero correction | Probability MAE vs current base | Flips vs current base | Corrected/harmed vs DARG | Null identity | Mean Null | Mean entropy |",
              "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for group in GROUPS:
        rows = [row for row in mechanisms if row["group"] == group]
        avg = lambda key: float(np.mean([float(row[key]) for row in rows]))
        fixed = sum(json.loads(row["historical_darg_to_final_corrected_harmed_net"])["corrected"] for row in rows)
        harmed = sum(json.loads(row["historical_darg_to_final_corrected_harmed_net"])["harmed"] for row in rows)
        null_ok = all(row["null_only_strict_identity"] for row in rows)
        attention = [json.loads(row["attention_summary"]) for row in rows]
        lines.append(f"| {group} | {avg('delta_margin_abs_mean'):.6f} | {avg('nonzero_correction_fraction'):.4f} | {avg('corrected_probability_mae_vs_current_base'):.6f} | {avg('hard_flips_vs_current_base'):.2f} | {fixed}/{harmed} | {null_ok} | {np.mean([v['null_mean'] for v in attention]):.4f} | {np.mean([v['normalized_entropy_mean'] for v in attention]):.4f} |")
    lines += ["", "The complete per-fold mechanisms are in `mechanism.csv`. `corrected/harmed vs DARG` counts patient decisions changed from the historical best checkpoint. Current-base comparisons identify how much change occurs after the new end-to-end training trajectory versus after the bounded correction itself. The G2-to-G1 replacement is a frozen inference intervention, not an independently trained ablation.",
              "", "## Excluding Fold 2", "", "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        row = summary[group]
        values = [f"{row[key]['without_fold2_mean']:.4f} ± {row[key]['without_fold2_population_sd']:.4f}" for key in METRICS]
        lines.append(f"| {group} | " + " | ".join(values) + " |")
    rel = next(row for row in comparisons if row["comparison"] == "G2_vs_G1" and row["scope"] == "all10" and row["metric"] == "acc")
    darg = next(row for row in comparisons if row["comparison"] == "G2_vs_DARG" and row["scope"] == "all10" and row["metric"] == "acc")
    rng_mismatch = sum(v["mismatches"] for k, v in status["rng_audit"].items())
    lines += ["", "## Decision", "",
              f"- Gate A performance floor: G2 ACC {summary[GROUPS[1]]['acc']['mean']:.4f}% vs DARG {summary['E74.4-G2-Positive-DCR']['acc']['mean']:.4f}%; Hard-AUC {summary[GROUPS[1]]['hard_auc']['mean']:.4f}% vs {summary['E74.4-G2-Positive-DCR']['hard_auc']['mean']:.4f}%.",
              f"- Gate B cross-local value: G2−G1 ACC {rel['mean_difference_pp']:+.4f} pp, W/T/L {rel['wins']}/{rel['ties']}/{rel['losses']}; paired rows are in `paired_comparisons.csv`.",
              f"- Gate C net historical correction: G2−DARG ACC {darg['mean_difference_pp']:+.4f} pp, W/T/L {darg['wins']}/{darg['ties']}/{darg['losses']}; patient-level corrected and harmed counts are reported per fold.",
              f"- Gate D end-to-end: all 10 folds pass base-off parity; public RNG triplet mismatch count is {rng_mismatch}.",
              "- Interpretation is preliminary and protocol-matched, not external clinical validation.",
              "", "## Audit limits", "",
              "Historical checkpoint selection uses test-fold ACC (validation equals test); preprocessing and RFE precede CV; the 871-patient graph is transductive; Fold 2 rows 377 and 407 overlap historical train/test and remain in DARG training while their new correction is forced to zero during training. These choices preserve historical comparability and bias generalization estimates. Fold intervals are descriptive because the training folds overlap. CAM144's 16 fixed coordinate slices are not anatomical regions. G1 repeats each local as its 15 candidates, changing the Null competition prior.",
              "", "## Artifacts", "",
              f"Training code SHA: `{status['source_sha256']}`. Fold metrics: `results/fold_metrics.csv`; paired comparisons: `results/paired_comparisons.csv`; mechanism: `results/mechanism.csv`; audit: `results/audit_status.json`."]
    return "\n".join(lines) + "\n"


def main():
    try:
        result = audit()
        print(json.dumps({"status": result["status"], "best": result["best_checkpoint_independent_pass"],
                          "latest": result["latest_state_integrity_pass"]}, ensure_ascii=False))
    except Exception as exc:
        failure = {"status": "FAIL", "error": repr(exc), "source_sha256": experiment.source_sha(),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        experiment.atomic_json(RESULTS / "audit_status.json", failure)
        raise


if __name__ == "__main__":
    main()
