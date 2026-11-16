#!/usr/bin/env python3
"""Restore, compare, and report the E80.1 out-of-fold models and interventions."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, t as student_t
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
spec = importlib.util.spec_from_file_location("experiment80_1_analysis_runner", EXP / "run.py")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)

METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
METHODS = ("B0", "G1", "G2", "G3")
METHOD_NAMES = {
    "B0": "E74.4-G2 historical DARG",
    "G1": "Static-Surrogate + Anonymous Prompt",
    "G2": "Real-Dynamic + Anonymous Prompt",
    "G3": "PSDC-Medical",
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 2) or not np.isfinite(p).all():
        raise ValueError("invalid ASD=0 / TD=1 probability matrix")
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tp, fn, fp, tn = (int(value) for value in cm.ravel())
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "acc": float(np.mean(pred == y)),
        "hard_auc": float(roc_auc_score(np.eye(2)[y], np.eye(2)[pred])),
        "prob_auc": float(roc_auc_score((y == 0).astype(int), p[:, 0])),
        "sen": float(sen),
        "spe": float(spe),
        "ba": float((sen + spe) / 2),
        "asd_f1": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "asd_tp": tp, "asd_fn": fn, "td_fp": fp, "td_tn": tn,
        "n_test": int(len(y)), "pred": pred, "prob": p,
    }


def _extract_predictions(run_metrics, state, test_indices):
    labels = np.asarray(state["y"].detach().cpu(), dtype=np.int64)[test_indices]
    probabilities = np.asarray(run_metrics["prob"], dtype=np.float64)
    metrics = independent_metrics(labels, probabilities)
    official = run_metrics
    for key in METRICS:
        if abs(metrics[key] - float(official[key])) > 1e-10:
            raise RuntimeError(f"independent OOF metric mismatch for {key}")
    return metrics


def _baseline_fold(fold, expected_identity):
    history = run.load_history()
    run.load_historical_e61()
    state, _, identity = history.prepare(fold)
    if identity != expected_identity:
        raise RuntimeError(f"B0 split/warmup identity differs from PSDC fold {fold}")
    source_root = Path(run.CFG["sources"]["e74_root"])
    group = run.HISTORICAL_GROUP
    historical_task = json.loads((source_root / ".runtime/tasks" / f"{group}_f{fold:02d}.json").read_text())
    checkpoint = source_root / ".runtime/checkpoints" / f"{group}_f{fold:02d}.best.pt"
    if historical_task.get("status") != "COMPLETE" or historical_task.get("checkpoint_sha256") != sha_file(checkpoint):
        raise RuntimeError(f"historical B0 task/checkpoint identity failed for fold {fold}")
    payload = history.torch_load(checkpoint)
    if payload.get("identity") != identity:
        raise RuntimeError(f"historical B0 best checkpoint initialization differs in fold {fold}")
    hist = history.load_history()
    hist.load_bundle(state, payload["best_state"])
    with torch.no_grad():
        output = history.forward(state, group, training=False, collect=False)
    test_indices = np.asarray(state["test_np"], dtype=np.int64)
    metrics = run.metrics_for(output["logp"], test_indices, state["y"])
    restored = _extract_predictions(metrics, state, test_indices)
    if not np.array_equal(restored["pred"], np.asarray(payload["test_prediction"], dtype=np.int64)):
        raise RuntimeError(f"independently restored B0 predictions differ in fold {fold}")
    if not np.allclose(restored["prob"], np.asarray(payload["test_probability"]), rtol=0, atol=1e-7):
        raise RuntimeError(f"independently restored B0 probabilities differ in fold {fold}")
    return state, test_indices, restored, historical_task, payload


def _restore_psdc_fold(group, fold, task):
    state, warmup, identity, module, optimizer, private_rng, scaler_sha, prompts = run.build_task_state(group, fold)
    if identity != task["identity"] or warmup["rng_sha"] != task["warmup_rng_sha"]:
        raise RuntimeError(f"warmup identity mismatch while restoring {group} fold {fold}")
    checkpoint = run.load_history().torch_load(run.checkpoint_path(group, fold, "best"))
    run.load_history().load_history().load_bundle(state, checkpoint["best"]["state"])
    module.load_state_dict(checkpoint["best"]["psdc"], strict=True)
    base = run.load_historical_e61().load_base()
    base.restore_rng(checkpoint["best"]["rng_shared"])
    state["model"].eval()
    state["cam"].eval()
    module.eval()
    return state, module, optimizer, private_rng, scaler_sha, prompts, base, checkpoint


def _intervention_row(kind, group, fold, native, intervened, n):
    return {
        "group": group, "fold": fold, "intervention": kind, "n_test": n,
        "acc": intervened["acc"], "acc_delta_pp": 100 * (intervened["acc"] - native["acc"]),
        "hard_auc": intervened["hard_auc"],
        "prob_auc": intervened["prob_auc"], "prob_auc_delta_pp": 100 * (intervened["prob_auc"] - native["prob_auc"]),
        "mean_abs_asd_probability_change": float(np.mean(np.abs(intervened["prob"][:, 0] - native["prob"][:, 0]))),
        "max_abs_asd_probability_change": float(np.max(np.abs(intervened["prob"][:, 0] - native["prob"][:, 0]))),
        "hard_prediction_flips": int(np.count_nonzero(intervened["pred"] != native["pred"])),
    }


def _per_patient_shuffle(counts, seed, max_windows):
    rng = np.random.default_rng(seed)
    order = np.zeros((len(counts), max_windows), dtype=np.int64)
    for row, count in enumerate(counts):
        order[row, :int(count)] = rng.permutation(int(count))
    return torch.as_tensor(order, dtype=torch.long, device=run.DEVICE)


def _group_fold(group, fold, expected_identity, y, oof, fold_metrics, mechanism_rows, intervention_rows):
    task_path = run.task_path(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("identity") != expected_identity:
        raise RuntimeError(f"task identity/status failed for {group} fold {fold}")
    if task.get("source_sha256") != run.source_sha():
        raise RuntimeError(f"task source SHA differs for {group} fold {fold}")
    state, module, optimizer, private_rng, scaler_sha, prompts, base, checkpoint = _restore_psdc_fold(group, fold, task)
    test_indices = np.asarray(task["test_indices"], dtype=np.int64)
    if not np.array_equal(test_indices, state["test_np"]):
        raise RuntimeError(f"test index order differs for {group} fold {fold}")
    with torch.no_grad():
        native_output = run.forward_psdc(state, module, group, training=False)
    native_metrics = _extract_predictions(run.metrics_for(native_output["logp"], test_indices, state["y"]), state, test_indices)
    if not np.array_equal(native_metrics["pred"], np.asarray(task["test_prediction"], dtype=np.int64)):
        raise RuntimeError(f"restored predictions differ from task record for {group} fold {fold}")
    if not np.allclose(native_metrics["prob"], np.asarray(task["test_probability"], dtype=np.float64), rtol=0, atol=1e-7):
        raise RuntimeError(f"restored probabilities differ from task record for {group} fold {fold}")
    oof[group][test_indices] = native_metrics["prob"]
    y[test_indices] = np.asarray(state["y"].detach().cpu(), dtype=np.int64)[test_indices]
    representation = native_output["psdc"]["dynamic_evidence"][:, [5, 7], :].reshape(871, -1)
    oof[f"{group}_dynamic_repr"][test_indices] = representation.detach().cpu().numpy()[test_indices]
    fold_metrics[(group, fold)] = native_metrics
    mechanism_rows.append({
        "group": group, "fold": fold, "best_epoch_zero_based": task["best_epoch_zero_based"],
        "best_epoch_1based": task["best_epoch_1based"], "psdc_parameters": task["parameter_count_psdc"],
        "relative_residual_mean": task["mechanism"]["relative_residual_mean"],
        "relative_residual_p95": task["mechanism"]["relative_residual_p95"],
        "relative_residual_max": task["mechanism"]["relative_residual_max"],
        "gate_mean": task["mechanism"]["gate_mean"], "gate_std": task["mechanism"]["gate_std"],
        "gate_saturation_fraction": task["mechanism"]["gate_saturation_fraction_lt_0.01_gt_0.99"],
        "residual_limit_fraction": task["mechanism"]["residual_limit_fraction"],
        "temporal_attention_entropy_mean": task["mechanism"]["temporal_attention_entropy_mean"],
        "classification_gradient_norms": json.dumps(task["classification_gradient_norms"], sort_keys=True),
        "rng_matched_epochs": task["rng_parity"]["matching_epochs"],
        "rng_exact_epochs": task["rng_parity"]["exact_triplets"],
        "rng_mismatches": task["rng_parity"]["mismatches"], "scaler_sha": scaler_sha,
    })

    with torch.no_grad():
        off_output = run.forward_psdc(state, module, group, training=False, dynamic_off=True)
    off_metrics = _extract_predictions(run.metrics_for(off_output["logp"], test_indices, state["y"]), state, test_indices)
    intervention_rows.append(_intervention_row("dynamic_off", group, fold, native_metrics, off_metrics, len(test_indices)))

    if group in ("G2", "G3"):
        data = run.load_data_tensors()
        order = _per_patient_shuffle(data["counts"].detach().cpu().numpy(), run.stable_seed("temporal-shuffle", fold), data["windows"].shape[1])
        state["e801_temporal_indices"] = order
        with torch.no_grad():
            shuffled_output = run.forward_psdc(state, module, group, training=False)
        shuffled_metrics = _extract_predictions(run.metrics_for(shuffled_output["logp"], test_indices, state["y"]), state, test_indices)
        intervention_rows.append(_intervention_row("temporal_shuffle", group, fold, native_metrics, shuffled_metrics, len(test_indices)))
        state["e801_temporal_indices"] = None

    if group == "G3":
        permutation = np.random.default_rng(run.stable_seed("medical-prompt-permutation", fold)).permutation(8)
        state["e801_prompt_indices"] = torch.as_tensor(permutation, dtype=torch.long, device=run.DEVICE)
        with torch.no_grad():
            prompt_output = run.forward_psdc(state, module, group, training=False)
        prompt_metrics = _extract_predictions(run.metrics_for(prompt_output["logp"], test_indices, state["y"]), state, test_indices)
        intervention_rows.append(_intervention_row("medical_prompt_permutation", group, fold, native_metrics, prompt_metrics, len(test_indices)))
        state["e801_prompt_indices"] = None
    return task


def _pooled_metrics(y, probability):
    if not np.isfinite(probability).all():
        raise RuntimeError("OOF predictions are incomplete")
    return independent_metrics(y, probability)


def _write_paired(fold_metrics, pooled, rows):
    comparisons = (("G1_minus_B0", "G1", "B0"), ("G2_minus_G1", "G2", "G1"),
                   ("G3_minus_G2", "G3", "G2"), ("G3_minus_B0", "G3", "B0"))
    output = []
    for name, left, right in comparisons:
        for metric in METRICS:
            differences = np.asarray([fold_metrics[(left, fold)][metric] - fold_metrics[(right, fold)][metric]
                                      for fold in range(1, 11)], dtype=np.float64)
            mean = float(differences.mean())
            sd = float(differences.std(ddof=0))
            half = float(student_t.ppf(0.975, len(differences) - 1) * sd / np.sqrt(len(differences)))
            delta = differences * 100
            output.append({
                "comparison": name, "metric": metric, "mean_delta_pp": 100 * mean,
                "population_sd_delta_pp": 100 * sd, "descriptive_95ci_low_pp": 100 * mean - 100 * half,
                "descriptive_95ci_high_pp": 100 * mean + 100 * half,
                "wins": int(np.sum(delta > 1e-12)), "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "losses": int(np.sum(delta < -1e-12)),
                "pooled_oof_delta_pp": 100 * (pooled[left][metric] - pooled[right][metric]),
                "fold2_excluded_mean_delta_pp": 100 * float(np.delete(differences, 1).mean()),
                "fold_deltas_pp": json.dumps(delta.round(8).tolist()),
            })
    run.atomic_csv(run.RESULTS / "paired_comparisons.csv", output)
    return output


def _confusion_analysis(predictions, labels, sites, folds):
    pairs = (("B0_to_G2", "B0", "G2"), ("B0_to_G3", "B0", "G3"), ("G2_to_G3", "G2", "G3"))
    output = []
    for name, base, new in pairs:
        b = predictions[base]
        n = predictions[new]
        b_correct, n_correct = b == labels, n == labels
        wrong_right = (~b_correct) & n_correct
        right_wrong = b_correct & (~n_correct)
        both_wrong = (~b_correct) & (~n_correct)
        both_right = b_correct & n_correct
        for class_id, class_name in ((0, "ASD"), (1, "TD")):
            mask = labels == class_id
            output.append({
                "comparison": name, "class": class_name,
                "wrong_to_right": int(np.sum(wrong_right & mask)),
                "right_to_wrong": int(np.sum(right_wrong & mask)),
                "both_wrong": int(np.sum(both_wrong & mask)),
                "both_right": int(np.sum(both_right & mask)),
                "net_corrections": int(np.sum(wrong_right & mask) - np.sum(right_wrong & mask)),
                "fold2_excluded_wrong_to_right": int(np.sum(wrong_right & mask & (folds != 2))),
                "fold2_excluded_right_to_wrong": int(np.sum(right_wrong & mask & (folds != 2))),
            })
        output.append({
            "comparison": name, "class": "ALL",
            "wrong_to_right": int(wrong_right.sum()), "right_to_wrong": int(right_wrong.sum()),
            "both_wrong": int(both_wrong.sum()), "both_right": int(both_right.sum()),
            "net_corrections": int(wrong_right.sum() - right_wrong.sum()),
            "fold2_excluded_wrong_to_right": int(np.sum(wrong_right & (folds != 2))),
            "fold2_excluded_right_to_wrong": int(np.sum(right_wrong & (folds != 2))),
        })
    run.atomic_csv(run.RESULTS / "complementary_analysis.csv", output)
    return output


def _site_and_quality(payload, labels, folds, predictions, representations):
    cache_path = Path(run.load_data_tensors()["summary"]["cache"])
    cache = np.load(cache_path, allow_pickle=False)
    windows = cache["windows"].astype(np.float64)
    counts = cache["window_count"].astype(np.int64)
    mask = np.arange(windows.shape[1])[None, :] < counts[:, None]
    means = np.where(mask[:, :, None], windows, 0).sum(axis=1) / counts[:, None]
    residual = windows - means[:, None, :]
    residual_rms = np.sqrt((np.where(mask[:, :, None], residual ** 2, 0).sum(axis=(1, 2))) / (counts * windows.shape[2]))
    adjacent_change = np.asarray([
        np.abs(windows[row, 1:int(count)] - windows[row, :int(count) - 1]).mean()
        for row, count in enumerate(counts)
    ])
    sites = cache["site"].astype(str)
    timepoints = cache["timepoints"].astype(np.float64)
    tr = cache["tr_seconds"].astype(np.float64)
    duration = timepoints * tr
    rows = np.arange(len(labels))

    phenotype = pd.read_csv(run.CFG["sources"]["phenotype_csv"])
    phenotype["SUB_ID"] = pd.to_numeric(phenotype["SUB_ID"], errors="coerce")
    phenotype = phenotype.dropna(subset=["SUB_ID"]).copy()
    phenotype["SUB_ID"] = phenotype["SUB_ID"].astype(np.int64)
    if phenotype["SUB_ID"].duplicated().any():
        raise RuntimeError("phenotype SUB_ID is not unique; refusing ambiguous QC merge")
    indexed = phenotype.set_index("SUB_ID")
    ids = cache["subject_id"].astype(np.int64)
    if not np.isin(ids, indexed.index.to_numpy()).all():
        raise RuntimeError("some locked BOLD SUB_ID values lack a phenotype quality record")
    qc_columns = [column for column in ("func_mean_fd", "func_dvars", "func_perc_fd", "func_outlier", "func_num_fd") if column in indexed]
    qc = {column: pd.to_numeric(indexed.loc[ids, column], errors="coerce").to_numpy(dtype=np.float64) for column in qc_columns}
    quality = []
    for name, values in (("timepoints", timepoints), ("TR_seconds", tr), ("scan_duration_seconds", duration),
                         ("window_count", counts.astype(np.float64)), ("within_subject_residual_rms", residual_rms),
                         ("adjacent_window_change", adjacent_change)) + tuple(qc.items()):
        finite = np.isfinite(values) & np.isfinite(residual_rms)
        if int(finite.sum()) < 5 or np.std(values[finite]) == 0:
            corr, p_value = np.nan, np.nan
        else:
            corr, p_value = spearmanr(values[finite], residual_rms[finite])
        quality.append({"analysis": "dynamic_amplitude_spearman", "variable": name,
                        "n": int(finite.sum()), "rho_with_residual_rms": float(corr), "p_descriptive": float(p_value)})
    for site in sorted(set(sites)):
        select = sites == site
        quality.append({
            "analysis": "site_prediction_accuracy", "variable": site, "n": int(select.sum()),
            "B0_accuracy": float(np.mean(predictions["B0"][select] == labels[select])),
            "G1_accuracy": float(np.mean(predictions["G1"][select] == labels[select])),
            "G2_accuracy": float(np.mean(predictions["G2"][select] == labels[select])),
            "G3_accuracy": float(np.mean(predictions["G3"][select] == labels[select])),
            "G2_minus_G1_accuracy_pp": float(100 * np.mean((predictions["G2"][select] == labels[select]) - (predictions["G1"][select] == labels[select]))),
            "G3_minus_G2_accuracy_pp": float(100 * np.mean((predictions["G3"][select] == labels[select]) - (predictions["G2"][select] == labels[select]))),
        })
    for group in ("G2", "G3"):
        if not np.isfinite(representations[group]).all():
            raise RuntimeError(f"OOF dynamic representation is incomplete for {group}")
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        pred_site = cross_val_predict(classifier, representations[group], sites, cv=splitter, n_jobs=1)
        quality.append({
            "analysis": "oof_dynamic_evidence_site_probe", "variable": group, "n": len(sites),
            "site_accuracy": float(np.mean(pred_site == sites)),
            "site_balanced_accuracy": float(balanced_accuracy_score(sites, pred_site)),
            "site_macro_f1": float(f1_score(sites, pred_site, average="macro", zero_division=0)),
            "site_classes": int(len(set(sites))),
        })
    quality.append({
        "analysis": "phenotype_motion_coverage", "variable": "matched_SUB_ID", "n": int(len(ids)),
        **{f"{column}_finite": int(np.isfinite(values).sum()) for column, values in qc.items()},
        "phenotype_site_mismatch_count": int(np.sum(phenotype.set_index("SUB_ID").loc[ids, "SITE_ID"].astype(str).to_numpy() != sites)),
    })
    run.atomic_csv(run.RESULTS / "site_analysis.csv", quality)
    return quality, {"residual_rms": residual_rms, "adjacent_change": adjacent_change,
                     "timepoints": timepoints, "tr": tr, "duration": duration, "counts": counts,
                     "sites": sites, "qc": qc}


def _mean_sd(fold_metrics, method, metric):
    values = np.asarray([fold_metrics[(method, fold)][metric] for fold in range(1, 11)], dtype=np.float64) * 100
    return f"{values.mean():.2f} ± {values.std(ddof=0):.2f}"


def _write_report(fold_metrics, pooled, paired, complementary, mechanisms, interventions, quality, prompt_cache, audit):
    labels = {"acc": "ACC", "hard_auc": "Hard-AUC / BA", "prob_auc": "Prob-AUC", "sen": "SEN-ASD", "spe": "SPE-TD", "asd_f1": "ASD-F1"}
    lines = [
        "# Experiment 80.1 — PSDC End-to-End Dynamic Complementary Learning", "",
        "Status: `COMPLETE_AUDITED` after all requested audits pass. Scientific decision is `REVIEW_PENDING` until the paired and mechanism evidence below is read as a whole.", "",
        "The only trained groups are G1, G2, and G3, each initialized from the matching E61 epoch-10 warmup. B0 is the read-only E74.4-G2 historical DARG. The 90-second sequence cache covers all 871 ABIDE A participants and contains 252 fixed E11 connections after excluding the four edges incident to unknown ROI index 82.", "",
        "The historical protocol is retained: validation equals test for historical checkpoint selection, pre-CV RFE and standardization, a transductive 871-patient graph, and the Fold 2 overlap at zero-based rows 376 and 406 (one-based 377 and 407). These scores are protocol-matched exploratory comparisons, not unbiased clinical generalization estimates. The BOLD files have no framewise censoring timestamps, so irregular deletion or concatenation cannot be audited.", "",
        "## Fold Mean ± Population SD (%)", "",
        "| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = [_mean_sd(fold_metrics, method, metric) for metric in METRICS]
        lines.append(f"| {METHOD_NAMES[method]} | " + " | ".join(values) + " |")
    lines.extend(["", "## Pooled OOF Metrics (%)", "", "| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method in METHODS:
        values = [f"{100 * pooled[method][metric]:.2f}" for metric in METRICS]
        lines.append(f"| {METHOD_NAMES[method]} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired Comparisons", "", "Each interval is a descriptive paired t interval over ten folds; it is not a confirmatory significance test. W/T/L counts favor the first method.", "", "| Contrast | Metric | Mean Δ pp | 95% interval pp | W/T/L | Pooled OOF Δ pp | Excluding Fold 2 Δ pp |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in paired:
        lines.append(f"| {row['comparison']} | {labels[row['metric']]} | {row['mean_delta_pp']:+.2f} | [{row['descriptive_95ci_low_pp']:+.2f}, {row['descriptive_95ci_high_pp']:+.2f}] | {row['wins']}/{row['ties']}/{row['losses']} | {row['pooled_oof_delta_pp']:+.2f} | {row['fold2_excluded_mean_delta_pp']:+.2f} |")
    lines.extend(["", "## Complementarity and Interventions", "", "The correction counts compare independent predictors after the fact; they are potential complementary evidence, not an actual fused correction. Dynamic-Off, temporal shuffle, and prompt permutation are read-only interventions on each selected checkpoint.", "", "### Error Changes", "", "| Contrast | Class | Wrong→Right | Right→Wrong | Both Wrong | Both Right | Net | Excluding Fold 2 W→R / R→W |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in complementary:
        lines.append(f"| {row['comparison']} | {row['class']} | {row['wrong_to_right']} | {row['right_to_wrong']} | {row['both_wrong']} | {row['both_right']} | {row['net_corrections']} | {row['fold2_excluded_wrong_to_right']} / {row['fold2_excluded_right_to_wrong']} |")
    intervention_means = {}
    for kind in sorted(set(row["intervention"] for row in interventions)):
        group_rows = [row for row in interventions if row["intervention"] == kind]
        intervention_means[kind] = {
            "mean_abs_asd_probability_change": float(np.mean([row["mean_abs_asd_probability_change"] for row in group_rows])),
            "hard_prediction_flips": int(sum(row["hard_prediction_flips"] for row in group_rows)),
            "acc_delta_pp_mean": float(np.mean([row["acc_delta_pp"] for row in group_rows])),
            "prob_auc_delta_pp_mean": float(np.mean([row["prob_auc_delta_pp"] for row in group_rows])),
        }
    lines.extend(["", "### Intervention Summary", "", "| Intervention | Checkpoints | Mean abs ASD-probability change | Hard flips | Mean ACC Δ pp | Mean Prob-AUC Δ pp |", "|---|---:|---:|---:|---:|---:|"])
    for kind, values in intervention_means.items():
        n = sum(1 for row in interventions if row["intervention"] == kind)
        lines.append(f"| {kind} | {n} | {values['mean_abs_asd_probability_change']:.5f} | {values['hard_prediction_flips']} | {values['acc_delta_pp_mean']:+.2f} | {values['prob_auc_delta_pp_mean']:+.2f} |")
    lines.extend(["", "## Mechanism and Confounding", "", f"The independent audit restored and recomputed {audit['audited_tasks']} Best Checkpoints and replayed one update twice from each Latest Checkpoint. The source-level dynamic branch receives only the 6th prompt slot (weight 1.0) and 8th prompt slot (weight 0.25); the other six slots are masked from dynamic evidence. The exact medical texts, anonymous texts, CLIP cache, and embedding hashes are stored in `config.json` and `.runtime/prompt_cache.pt`.", "", "Fold-level gate, residual, attention entropy, class-gradient and RNG trace diagnostics are in `mechanism.csv`; per-fold Dynamic-Off and temporal/prompt interventions are in `interventions.csv`.", "", "| Diagnostic | Value |", "|---|---:|"])
    for row in quality:
        if row.get("analysis") == "oof_dynamic_evidence_site_probe":
            lines.append(f"| {row['variable']} dynamic evidence site OOF accuracy / balanced accuracy / macro-F1 | {row['site_accuracy']:.3f} / {row['site_balanced_accuracy']:.3f} / {row['site_macro_f1']:.3f} |")
    for row in quality:
        if row.get("analysis") == "dynamic_amplitude_spearman":
            lines.append(f"| Spearman(dynamic residual RMS, {row['variable']}) (n={row['n']}) | {row['rho_with_residual_rms']:.3f} (descriptive p={row['p_descriptive']:.3g}) |")
    lines.extend(["", "Site-wise counts and OOF accuracy changes are in `site_analysis.csv`; motion metadata coverage and phenotype/site consistency are recorded there as well. Correlations are descriptive and unadjusted for multiple comparisons.", "", "## Training and Identity", "", f"The actual device was `{run.DEVICE}` / `{torch.cuda.get_device_name(0) if run.DEVICE.type == 'cuda' else 'CPU'}`. The configured maximum was {run.CFG['training']['max_workers']} concurrent workers. The 30 task records each store the exact E61 warmup SHA, split/pair identities, initialization hashes, optimizer identity, private RNG state, common RNG trace, and Best/Latest Checkpoint SHA. The audit reports {audit['audited_tasks']}/30 tasks; all recorded common RNG triplets are checked against historical E74.4-G2 wherever the epoch overlaps.", "", "## Limits and Decision", "", "No dynamic-only high score is used as a gate. The E80 result was weak for V+D/PCA32, which motivated the learned temporal stage; E80.1 tests only this fixed 90-second, 252-edge, two-layer TCN design. A positive outcome would still not establish an ASD biomarker or causal disease mechanism. G3's text perturbation tests conditional dependence on text embeddings, not correctness of a pathological explanation.", "", "Final scientific state requires reading all three nested comparisons (G2−G1, G3−G2, G3−B0), pooled OOF metrics, Fold 2-excluded trends, correction/false-regression counts, site concentration, and intervention effects together. `NO_GO`, `PARTIAL`, or `GO` must not be inferred from attention movement, nonzero gradient, or a single metric.", ""])
    report = run.RESULTS / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    with report.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    return intervention_means


def analyze():
    audit_path = run.RESULTS / "audit_status.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_30_OF_30" or audit.get("source_sha256") != run.source_sha():
        raise RuntimeError("all independent checkpoint audits must pass before analysis")
    preflight = json.loads((run.RESULTS / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("source_sha256") != run.source_sha():
        raise RuntimeError("preflight source identity does not match analysis source")

    oof = {method: np.full((871, 2), np.nan, dtype=np.float64) for method in METHODS}
    oof.update({f"{group}_dynamic_repr": np.full((871, 128), np.nan, dtype=np.float64) for group in ("G2", "G3")})
    y = np.full(871, -1, dtype=np.int64)
    fold_ids = np.full(871, -1, dtype=np.int64)
    fold_metrics = {}
    mechanism_rows, intervention_rows, epoch_rows = [], [], []
    expected_by_fold = {}
    prompt_cache = run.load_prompt_embeddings()

    for fold in range(1, 11):
        first_group_task = json.loads(run.task_path("G1", fold).read_text(encoding="utf-8"))
        expected_by_fold[fold] = first_group_task["identity"]
        state, indices, base_metrics, historical_task, historical_checkpoint = _baseline_fold(fold, expected_by_fold[fold])
        oof["B0"][indices] = base_metrics["prob"]
        y[indices] = np.asarray(state["y"].detach().cpu(), dtype=np.int64)[indices]
        fold_ids[indices] = fold
        fold_metrics[("B0", fold)] = base_metrics
        for group in ("G1", "G2", "G3"):
            task = _group_fold(group, fold, expected_by_fold[fold], y, oof, fold_metrics, mechanism_rows, intervention_rows)
            if not np.array_equal(np.asarray(task["test_indices"], dtype=np.int64), indices):
                raise RuntimeError(f"outer test fold differs across groups in fold {fold}")
            historical_epoch = int(historical_task["best_epoch_1based"])
            log_path = run.LOGS / f"{group}_f{fold:02d}.jsonl"
            selected = None
            if log_path.is_file():
                for line in log_path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    if int(record["epoch"]) == historical_epoch:
                        selected = record
                        break
            epoch_rows.append({
                "group": group, "fold": fold, "historical_e74_best_epoch_1based": historical_epoch,
                "e801_epoch_reached": selected is not None,
                **({f"metric_{key}": selected["metrics"][key] for key in METRICS} if selected else {}),
                "note": "matched historical epoch; auxiliary only" if selected else "E80.1 stopped before this historical epoch; not comparable",
            })

    if not np.isin(y, [0, 1]).all() or not np.array_equal(fold_ids >= 1, np.ones(871, dtype=bool)):
        raise RuntimeError("OOF patient labels/fold assignment incomplete")
    pooled = {method: _pooled_metrics(y, oof[method]) for method in METHODS}
    fold_rows = []
    for method in METHODS:
        for fold in range(1, 11):
            metrics = fold_metrics[(method, fold)]
            fold_rows.append({"method": method, "method_name": METHOD_NAMES[method], "fold": fold,
                              "n_test": metrics["n_test"], "acc": metrics["acc"], "hard_auc": metrics["hard_auc"],
                              "prob_auc": metrics["prob_auc"], "sen": metrics["sen"], "spe": metrics["spe"],
                              "asd_f1": metrics["asd_f1"], "asd_tp": metrics["asd_tp"], "asd_fn": metrics["asd_fn"],
                              "td_fp": metrics["td_fp"], "td_tn": metrics["td_tn"]})
    run.atomic_csv(run.RESULTS / "fold_metrics.csv", fold_rows)
    run.atomic_csv(run.RESULTS / "mechanism.csv", mechanism_rows)
    run.atomic_csv(run.RESULTS / "interventions.csv", intervention_rows)
    run.atomic_csv(run.RESULTS / "historical_epoch_alignment.csv", epoch_rows)
    paired = _write_paired(fold_metrics, pooled, [])

    predictions = {method: oof[method].argmax(axis=1) for method in METHODS}
    sites = run.load_data_tensors()["site"]
    complementary = _confusion_analysis(predictions, y, sites, fold_ids)
    data_qc, _ = _site_and_quality(run.load_data_tensors(), y, fold_ids, predictions,
                                   {group: oof[f"{group}_dynamic_repr"] for group in ("G2", "G3")})
    patient_rows = []
    for row in range(871):
        patient_rows.append({
            "row_zero_based": row, "fold": int(fold_ids[row]), "site": str(sites[row]), "label": int(y[row]),
            **{f"{method}_prediction": int(predictions[method][row]) for method in METHODS},
            **{f"{method}_prob_asd": float(oof[method][row, 0]) for method in METHODS},
        })
    run.atomic_csv(run.RESULTS / "oof_prediction_audit.csv", patient_rows)
    intervention_means = _write_report(fold_metrics, pooled, paired, complementary, mechanism_rows,
                                       intervention_rows, data_qc, prompt_cache, audit)

    report_path = run.RESULTS / "report.md"
    status = {
        "status": "ANALYSIS_COMPLETE_REVIEW_PENDING",
        "scientific_decision": "REVIEW_PENDING",
        "source_sha256": run.source_sha(),
        "audited_tasks": int(audit["audited_tasks"]),
        "dynamic_coverage": float(run.load_data_tensors()["summary"]["coverage"]),
        "pooled_metrics": {method: {key: pooled[method][key] for key in METRICS} for method in METHODS},
        "intervention_means": intervention_means,
        "report_sha256": sha_file(report_path),
        "fold_metrics_sha256": sha_file(run.RESULTS / "fold_metrics.csv"),
        "paired_comparisons_sha256": sha_file(run.RESULTS / "paired_comparisons.csv"),
        "complementary_analysis_sha256": sha_file(run.RESULTS / "complementary_analysis.csv"),
    }
    run.atomic_json(run.RESULTS / "analysis_status.json", status)
    print(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    analyze()
