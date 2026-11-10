#!/usr/bin/env python3
"""Independent checkpoint readback, intervention analysis, summaries, and handoff."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import run

EXP = run.EXP
CFG = run.CFG
BASE_DIR = run.BASE_DIR
BASE_GROUP = run.BASE_GROUP
RESULTS = run.RESULTS
GROUPS = run.GROUPS
METRICS = run.METRICS
INTERVENTIONS = ("full", "module_off", "semantic_off", "graph_context_off", "agreement_off", "prompt_class_swap")


def metric_record(metrics):
    return {key: float(metrics[key]) for key in METRICS}


def _mean_sd(values):
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def audit_one(group, fold, source_hash):
    import numpy as np
    import torch
    state, warmup, identity, base_task = run.prepare_verified(fold)
    task_path, best_path, _, _ = run.task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = run.sha_file(best_path)
    checkpoint = run.torch_load(best_path)
    for key, expected in (("experiment", CFG["experiment"]), ("group", group), ("fold", fold),
                          ("source_sha256", source_hash), ("identity", identity),
                          ("base_task_checkpoint_sha256", base_task["checkpoint_sha256"])):
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"checkpoint {group} fold {fold} provenance mismatch: {key}")
    if checkpoint.get("base_best_checkpoint_used_for_training") is not False:
        raise RuntimeError("the new task improperly used E74.4 best checkpoint fine-tuning")
    if task.get("checkpoint_sha256") != checkpoint_sha or task.get("best_epoch_1based") != checkpoint.get("best_epoch_1based"):
        raise RuntimeError("task/checkpoint identity mismatch")
    if task.get("source_sha256") != source_hash or task.get("identity") != identity:
        raise RuntimeError("task source/init identity mismatch")
    hist = run.load_base().load_history()
    hist.load_bundle(state, checkpoint["base_state"])
    device = next(state["cam"].parameters()).device
    module = run.group_module(run.load_text_cache(), group, fold, device)
    module.load_state_dict(checkpoint["module_state"], strict=True)
    module.eval(); state["model"].eval(); state["cam"].eval()
    bank = run.deserialize_bank(checkpoint["prototype_bank"])
    meta = checkpoint["prototype_metadata"]
    if bank.bank_sha256 != meta["bank_sha256"] or bank.source_sha256 != meta["source_sha256"]:
        raise RuntimeError("saved prototype bank metadata mismatch")
    expected_aux = torch.as_tensor(state["train"], dtype=torch.long, device=state["y"].device)
    test_tensor = torch.as_tensor(state["test"], dtype=torch.long, device=state["y"].device)
    expected_aux = expected_aux[~torch.isin(expected_aux, test_tensor)]
    if not torch.equal(bank.source_indices.to(expected_aux.device).sort().values, expected_aux.sort().values):
        raise RuntimeError("prototype source is not exactly the eligible outer-fold training nodes")
    if torch.isin(bank.source_indices.to(test_tensor.device), test_tensor).any():
        raise RuntimeError("test node entered prototype-label source")
    if fold == 2 and torch.isin(torch.tensor([376, 406], device=bank.source_indices.device), bank.source_indices).any():
        raise RuntimeError("Fold 2 exposed rows 377/407 entered the new prototype bank")
    labels = state["y"].detach().long()
    source_labels = labels.index_select(0, bank.source_indices.to(labels.device))
    if not torch.equal(bank.class_counts.to(source_labels.device), torch.bincount(source_labels, minlength=2)):
        raise RuntimeError("saved prototype class counts disagree with the eligible training labels")
    if not torch.isfinite(bank.full).all() or not torch.isfinite(bank.crossfit).all():
        raise RuntimeError("non-finite saved prototype bank")
    if float(module.alpha.detach().cpu()) < 0 or float(module.alpha.detach().cpu()) > 0.5:
        raise RuntimeError("alpha outside [0,0.5]")
    if float(module.gamma.detach().cpu()) < 0 or float(module.gamma.detach().cpu()) > 0.25:
        raise RuntimeError("gamma outside [0,0.25]")
    # Independent replay of full and fixed-checkpoint intervention paths.
    runtime = run.rng_runtime()
    rng_before = runtime.rng_sha(runtime.pack_rng())
    predictions, probabilities, intervention_rows, captures = {}, {}, [], {}
    full_metrics = None
    with torch.no_grad():
        for intervention in INTERVENTIONS:
            out, cap = run.make_capture(state, module, bank, group,
                                        intervention="full" if intervention == "full" else intervention,
                                        training=False)
            metrics = runtime.classification_metrics(out["logp"], state["test"], state["y"])
            pred = np.asarray(metrics["pred"], dtype=np.int64)
            prob = np.asarray(metrics["prob"], dtype=np.float64)
            key = "full" if intervention == "full" else intervention
            predictions[key], probabilities[key] = pred, prob
            captures[key] = cap
            if key == "full":
                full_metrics = metrics
            if key == "full":
                intervention_rows.append({"group": group, "fold": fold, "intervention": "full", **metric_record(metrics),
                    "hard_flips": 0, "correct_to_wrong": 0, "wrong_to_correct": 0,
                    "mean_abs_probability_change": 0.0, "mean_abs_margin_change": 0.0,
                    "cam_feature_relative_change": 0.0})
            else:
                full_pred = predictions["full"]
                y_test = state["y"].index_select(0, test_tensor).detach().cpu().numpy().astype(np.int64)
                flips = pred != full_pred
                correct_to_wrong = int(((full_pred == y_test) & (pred != y_test)).sum())
                wrong_to_correct = int(((full_pred != y_test) & (pred == y_test)).sum())
                full_prob = probabilities["full"]
                full_margin = full_prob - (1.0 - full_prob)
                changed_margin = prob - (1.0 - prob)
                full_h = captures["full"]["result"]["h_prime"]
                changed_h = cap["result"]["h_prime"]
                rel_change = float((changed_h - full_h).norm() / full_h.norm().clamp_min(1e-12))
                intervention_rows.append({"group": group, "fold": fold, "intervention": key, **metric_record(metrics),
                    "hard_flips": int(flips.sum()), "correct_to_wrong": correct_to_wrong,
                    "wrong_to_correct": wrong_to_correct,
                    "mean_abs_probability_change": float(np.abs(prob - full_prob).mean()),
                    "mean_abs_margin_change": float(np.abs(changed_margin - full_margin).mean()),
                    "cam_feature_relative_change": rel_change})
    if runtime.rng_sha(runtime.pack_rng()) != rng_before:
        raise RuntimeError("read-only checkpoint audit/intervention changed main RNG")
    saved_pred = np.asarray(checkpoint["test_prediction"], dtype=np.int64)
    saved_prob = np.asarray(checkpoint["test_probability"], dtype=np.float64)
    if not np.array_equal(predictions["full"], saved_pred):
        raise RuntimeError("independent checkpoint readback hard predictions differ")
    if not np.allclose(probabilities["full"], saved_prob, atol=1e-8, rtol=0):
        raise RuntimeError("independent checkpoint readback probabilities differ")
    for metric in METRICS:
        if abs(float(checkpoint["metrics"][metric]) - float(full_metrics[metric])) > 1e-8:
            raise RuntimeError(f"independent metric replay differs: {metric}")
    if checkpoint["best_epoch_1based"] != task["best_epoch_1based"]:
        raise RuntimeError("best epoch differs between task and checkpoint")
    gamma = float(module.gamma.detach().cpu()); alpha = float(module.alpha.detach().cpu())
    result = captures["full"]["result"]
    residual_ratio = float(result["residual"].norm() / captures["full"]["raw_h"].norm().clamp_min(1e-12))
    modality_rows = []
    for index, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        token_slice = slice(index * 4, (index + 1) * 4)
        modality_rows.append({
            "prototype_separation": float((1.0 - torch.nn.functional.cosine_similarity(
                result["p_proto"][:, token_slice, 0], result["p_proto"][:, token_slice, 1], dim=-1)).mean().cpu()),
            "reliability": float(result["reliability"][:, token_slice].mean().cpu()),
            "residual_norm": float(result["residual"][:, token_slice].norm().cpu()),
            "token_count": 4,
        })
    mechanism = {"group": group, "fold": fold, "best_epoch_1based": checkpoint["best_epoch_1based"],
        "prototype_source_count": int(bank.source_indices.numel()), "prototype_class0_count": int(bank.class_counts[0]),
        "prototype_class1_count": int(bank.class_counts[1]), "prototype_source_sha256": bank.source_sha256,
        "prototype_bank_sha256": bank.bank_sha256, "alpha": alpha, "gamma": gamma,
        "r_sep_mean": float(result["r_sep"].mean().cpu()), "r_conf_mean": float(result["r_conf"].mean().cpu()),
        "r_agree_mean": float(result["r_agree"].mean().cpu()), "reliability_mean": float(result["reliability"].mean().cpu()),
        "prototype_semantic_probability_mae": float((result["p_proto"] - result["p_sem"]).abs().mean().cpu()),
        "prototype_semantic_agreement": float((result["p_proto"].argmax(-1) == result["p_sem"].argmax(-1)).float().mean().cpu()),
        "residual_cam_l2_ratio": residual_ratio,
        "h_prime_relative_change": float((result["h_prime"] - captures["full"]["raw_h"]).norm() / captures["full"]["raw_h"].norm().clamp_min(1e-12)),
        "graph_context_patient_sd": float(result["graph_context"].std(dim=0).mean().cpu()),
        "prototype_loss_best": checkpoint["losses"]["l_proto"], "alignment_loss_best": checkpoint["losses"]["l_align"],
        "semantic_prompt_weights": json.dumps(result["p_sem"].mean(dim=(0, 1)).detach().cpu().tolist()),
        "modality_reliability": json.dumps([row["reliability"] for row in modality_rows]),
        "modality_residual_norm": json.dumps([row["residual_norm"] for row in modality_rows]),
        "module_parameter_sha256": run.sha_state(module), "semantic_gram_max_abs_error": float(module.gram_error.cpu()),
        "main_rng_unchanged_in_audit": True}
    audit = {"group": group, "fold": fold, "status": "PASS", "checkpoint_sha256": checkpoint_sha,
        "source_sha256": source_hash, "base_task_checkpoint_sha256": base_task["checkpoint_sha256"],
        "warmup_sha256": identity["warmup_sha256"], "best_epoch_1based": checkpoint["best_epoch_1based"],
        "metric_replay_max_abs_error": max(abs(float(checkpoint["metrics"][m]) - float(full_metrics[m])) for m in METRICS),
        "hard_prediction_exact": True, "probability_replay_max_abs_error": float(np.max(np.abs(saved_prob - probabilities["full"]))),
        "prototype_test_label_exclusion": True, "fold2_rows_excluded": (fold != 2 or True),
        "outputlayer_calls": captures["full"]["outputlayer_calls"], "cam_calls": captures["full"]["cam_calls"],
        "base_best_checkpoint_used_for_training": False, "prototype_bank_sha256": bank.bank_sha256,
        "module_parameter_sha256": run.sha_state(module)}
    metrics_row = {"group": group, "fold": fold, "status": "INDEPENDENT_AUDIT_PASS",
                   "best_epoch_1based": checkpoint["best_epoch_1based"], **metric_record(full_metrics),
                   "checkpoint_sha256": checkpoint_sha,
                   "train_test_overlap_1based": json.dumps([377, 407] if fold == 2 else [])}
    return audit, metrics_row, mechanism, intervention_rows, task


def summarize(metrics_rows):
    import numpy as np
    base_rows = list(csv.DictReader((BASE_DIR / "results/fold_metrics.csv").open(encoding="utf-8")))
    base = {int(row["fold"]): row for row in base_rows if row["group"] == BASE_GROUP}
    if set(base) != set(range(1, 11)):
        raise RuntimeError("E74.4-G2 historical fold metrics are incomplete")
    all_groups = {group: {int(row["fold"]): row for row in metrics_rows if row["group"] == group} for group in GROUPS}
    summary_rows, paired_rows = [], []
    sources = {"E74.4-G2 historical": base}
    sources.update(all_groups)
    for label, rows in sources.items():
        for scope, folds in (("all_10", list(range(1, 11))), ("exclude_fold2_descriptive_9", [i for i in range(1, 11) if i != 2])):
            summary = {"group": label, "scope": scope, "n_folds": len(folds)}
            for metric in METRICS:
                values = [float(rows[f][metric]) * 100.0 for f in folds]
                mean, sd = _mean_sd(values)
                summary[f"{metric}_mean_pct"], summary[f"{metric}_population_sd_pct"] = mean, sd
            summary_rows.append(summary)
    comparisons = [(f"{group} - E74.4-G2", group, "base") for group in GROUPS]
    comparisons += [("G2-GC-SPER - G1-Proto", "G2-GC-SPER", "G1-Proto"),
                    ("G2-GC-SPER - G3-Anonymous", "G2-GC-SPER", "G3-Anonymous")]
    source_map = {"base": base, **all_groups}
    for name, left, right in comparisons:
        for scope, folds in (("all_10", list(range(1, 11))), ("exclude_fold2_descriptive_9", [i for i in range(1, 11) if i != 2])):
            for metric in METRICS:
                deltas = {fold: (float(source_map[left][fold][metric]) - float(source_map[right][fold][metric])) * 100.0 for fold in folds}
                vals = list(deltas.values())
                paired_rows.append({"comparison": name, "scope": scope, "metric": metric,
                    "mean_delta_pp": float(np.mean(vals)), "wins": sum(v > 1e-12 for v in vals),
                    "ties": sum(abs(v) <= 1e-12 for v in vals), "losses": sum(v < -1e-12 for v in vals),
                    "fold_deltas_pp": json.dumps(deltas, sort_keys=True)})
    return summary_rows, paired_rows


def _format_summary(rows):
    by = {(row["group"], row["scope"]): row for row in rows}
    labels = ["E74.4-G2 historical", *GROUPS]
    lines = ["| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        r = by[(label, "all_10")]
        def cell(metric):
            return f"{r[metric + '_mean_pct']:.4f} ± {r[metric + '_population_sd_pct']:.4f}"
        lines.append(f"|{label}|{cell('acc')}|{cell('hard_auc')}|{cell('prob_auc')}|{cell('sen')}|{cell('spe')}|{cell('asd_f1')}|")
    return "\n".join(lines)


def build_report(manifest, audit_status, summary_rows, paired_rows, interventions, mechanism_rows):
    by_pair = {(r["comparison"], r["scope"], r["metric"]): r for r in paired_rows}
    def diff(name, scope, metric="acc"):
        row = by_pair[(name, scope, metric)]
        return f"{row['mean_delta_pp']:+.4f} pp ({row['wins']}/{row['ties']}/{row['losses']})"
    table = _format_summary(summary_rows)
    folds9 = {(r["group"], r["scope"]): r for r in summary_rows}
    g2base10 = diff("G2-GC-SPER - E74.4-G2", "all_10")
    g2base9 = diff("G2-GC-SPER - E74.4-G2", "exclude_fold2_descriptive_9")
    interventions_by = {}
    for row in interventions:
        if row["intervention"] == "module_off":
            interventions_by.setdefault(row["group"], []).append(row)
    flips_g2 = sum(int(r["hard_flips"]) for r in interventions_by.get("G2-GC-SPER", []))
    report = [
        "# Experiment 75.2 — GC-SPER",
        "",
        f"Status: **{manifest.get('status')}**; tasks {manifest.get('completed_tasks')}/30; independent audits {manifest.get('independent_audits', 0)}/30.",
        f"Directory: `{EXP}`. Device `{manifest.get('device')}`; worker cap {manifest.get('worker_limit')}, observed peak {manifest.get('training_execution', {}).get('max_workers_observed', 0)}; wall time {manifest.get('training_execution', {}).get('wall_seconds', 0):.1f}s; retries {len(manifest.get('training_execution', {}).get('retries', []))}; OOM {manifest.get('training_execution', {}).get('oom_events', 0)}.",
        f"E74.4-G2 source manifest SHA: `{CFG['base']['manifest_source_sha256']}`; current source digest `{manifest.get('source_sha256')}`.",
        "",
        "## Method and audit boundaries",
        "- E74.4-G2 is the only training base. Every task restored the corresponding E61 epoch-10 warmup through E74.4 `prepare(fold)` and called E74.4 `forward_state` through its own runner. The E74.4 best checkpoint was audited as a reference and was not used for fine-tuning.",
        "- The original DARG produces 320-D F and the original CAM runs once. Hooks capture F at the CAM input and replace only its 144-D Outputlayer input; the Outputlayer also runs once. No second graph, GCN, CAM, or classifier was introduced.",
        "- Prototypes use only eligible outer-fold training nodes. Five deterministic class-stratified blocks provide leave-block-out prototypes for training patients; test nodes use the full training-only bank. Fold-2 rows 377/407 were excluded from new prototype labels and auxiliary losses, while the inherited base protocol still exposes them to its original train/test overlap.",
        "- G1 uses prototypical evidence only; G2 adds eight original frozen SPromptGL prompts with F320-conditioned class-specific FiLM, evidence mixture and alignment loss; G3 replaces projected semantic vectors by a post-projection orthogonal rotation. Its fixed 8x8 Gram is checked below. This is a single geometry-matched anonymous transform, not a general proof of medical semantics.",
        "- SPromptGL's learned 4×4×9 feature partition does not identify 16 anatomical ROIs. Prompt/prototype features are not lesion maps. The paper distinguishes GFR and PLN components; the E74.4 baseline uses the audited `PLNNoPEN` path, and this module is inserted before its Outputlayer rather than restoring original PEN.",
        "- Related work already combines prompts and prototypes, including HP²L; the candidate distinction here is the joint use of training-fold token prototypes, DARG-conditioned disease prompts, cross-fitted evidence reliability, and CAM feature correction. Prompt/prototype combination alone is not claimed as novel.",
        "",
        "## Classification results",
        "Mean ± population SD; percentages. Hard-AUC is the historical hard-prediction/BA convention; Prob-AUC uses continuous ASD probabilities.",
        "",
        table,
        "",
        "## Paired comparisons",
        f"- G2−E74.4-G2 ACC: all folds {g2base10}; excluding Fold 2 {g2base9}.",
        f"- G2−G1 ACC: all folds {diff('G2-GC-SPER - G1-Proto', 'all_10')}; excluding Fold 2 {diff('G2-GC-SPER - G1-Proto', 'exclude_fold2_descriptive_9')}.",
        f"- G2−G3 ACC: all folds {diff('G2-GC-SPER - G3-Anonymous', 'all_10')}; excluding Fold 2 {diff('G2-GC-SPER - G3-Anonymous', 'exclude_fold2_descriptive_9')}.",
        "- Full metric-by-metric deltas and per-fold W/T/L are in `paired_comparisons.csv`.",
        "",
        "## Mechanisms and interventions",
        f"- G2 mean alpha `{sum(float(r['alpha']) for r in mechanism_rows if r['group']=='G2-GC-SPER')/10:.5f}`, mean gamma `{sum(float(r['gamma']) for r in mechanism_rows if r['group']=='G2-GC-SPER')/10:.5f}`, mean residual/CAM ratio `{100*sum(float(r['residual_cam_l2_ratio']) for r in mechanism_rows if r['group']=='G2-GC-SPER')/10:.4f}%`.",
        f"- `module_off` caused {flips_g2} hard prediction flips across the 10 G2 checkpoints. This is fixed-checkpoint reliance, not a retrained ablation or historical-checkpoint recovery.",
        "- `semantic_off`, `graph_context_off`, `agreement_off`, and `prompt_class_swap` are separately reported per fold and group in `interventions.csv`; no outcomes are pooled across groups.",
        "- Per-fold prototype separation, semantic/prototype probability difference and agreement, reliability by 4-token modality block, query/context variation, losses and residual magnitudes are in `mechanism.csv`.",
        "",
        "## Evaluation limits and decision",
        "- The historical protocol retains `val=test` checkpoint selection, RFE/standardization before CV, a transductive 871-node graph and Fold-2 exposure of 1-based rows 377 (ASD) and 407 (TD). Excluding Fold 2 is descriptive and does not remove the other limitations.",
        "- Historical E74.4-G2 is a read-only paired reference, not a randomized external test. A result here cannot support an unbiased independent-generalization claim.",
        f"- Independent checkpoint readback: {audit_status['independently_audited']}/30; hard predictions exact and maximum probability replay error `{audit_status['probability_replay_max_abs_error']:.3g}`.",
        f"- Scientific decision: **{manifest.get('scientific_conclusion', 'PENDING')}** under the preregistered criteria. G2 ACC change vs base is {g2base10}; G2 module-off hard flips: {flips_g2}.",
        f"- Report: `{RESULTS / 'report.md'}`; manifest: `{EXP / 'manifest.json'}`; audit: `{RESULTS / 'audit_status.json'}`.",
    ]
    return "\n".join(report) + "\n"


def append_handoff(text):
    import fcntl
    original = HANDOFF.read_bytes()
    with HANDOFF.open("ab") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(("\n\n" + text).encode("utf-8")); stream.flush(); os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    after = HANDOFF.read_bytes()
    expected = ("\n\n" + text).encode("utf-8")
    if not after.startswith(original) or after[len(original):] != expected:
        raise IOError("handoff append verification failed")
    return {"status": "APPENDED_AND_VERIFIED", "path": str(HANDOFF), "append_sha256": run.sha_object(text),
            "original_bytes_preserved": len(original), "appended_bytes": len(expected)}


def run_audit(update_handoff=True):
    manifest_path = EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_hash = run.source_digest()
    if manifest.get("source_sha256") != source_hash or manifest.get("completed_tasks") != 30:
        raise RuntimeError("auditing requires 30 completed tasks with unchanged source")
    audits, metrics, mechanisms, interventions, tasks = [], [], [], [], []
    for group in GROUPS:
        for fold in range(1, 11):
            audit, metric_row, mechanism, changed, task = audit_one(group, fold, source_hash)
            audits.append(audit); metrics.append(metric_row); mechanisms.append(mechanism); interventions.extend(changed); tasks.append(task)
    if len(audits) != 30 or any(row["status"] != "PASS" for row in audits):
        raise RuntimeError("independent audit did not pass all 30 checkpoints")
    summary_rows, paired_rows = summarize(metrics)
    run.write_csv(RESULTS / "checkpoint_audit.csv", audits)
    run.write_csv(RESULTS / "fold_metrics.csv", metrics)
    run.write_csv(RESULTS / "mechanism.csv", mechanisms)
    run.write_csv(RESULTS / "interventions.csv", interventions)
    run.write_csv(RESULTS / "summary.csv", summary_rows)
    run.write_csv(RESULTS / "paired_comparisons.csv", paired_rows)
    base_rows = list(csv.DictReader((BASE_DIR / "results/fold_metrics.csv").open(encoding="utf-8")))
    base = {int(r["fold"]): r for r in base_rows if r["group"] == BASE_GROUP}
    g2 = {int(r["fold"]): r for r in metrics if r["group"] == "G2-GC-SPER"}
    acc_delta = sum((float(g2[i]["acc"]) - float(base[i]["acc"])) * 100 for i in range(1, 11)) / 10
    wins = sum(float(g2[i]["acc"]) > float(base[i]["acc"]) for i in range(1, 11))
    g1 = {int(r["fold"]): r for r in metrics if r["group"] == "G1-Proto"}
    g3 = {int(r["fold"]): r for r in metrics if r["group"] == "G3-Anonymous"}
    g2_vs_g1 = sum((float(g2[i]["acc"]) - float(g1[i]["acc"])) * 100 for i in range(1, 11)) / 10
    g2_vs_g3 = sum((float(g2[i]["acc"]) - float(g3[i]["acc"])) * 100 for i in range(1, 11)) / 10
    module_off_g2 = [r for r in interventions if r["group"] == "G2-GC-SPER" and r["intervention"] == "module_off"]
    flips = sum(int(r["hard_flips"]) for r in module_off_g2)
    base_summary = next(r for r in summary_rows if r["group"] == "E74.4-G2 historical" and r["scope"] == "all_10")
    new_g2 = next(r for r in summary_rows if r["group"] == "G2-GC-SPER" and r["scope"] == "all_10")
    # Apply the locked GO criteria without treating any one metric as sufficient.
    go = (acc_delta >= 0.3 and wins >= 6 and g2_vs_g1 > 0 and g2_vs_g3 > 0 and flips > 0 and
          float(new_g2["hard_auc_mean_pct"]) >= float(base_summary["hard_auc_mean_pct"]) - 0.5 and
          float(new_g2["prob_auc_mean_pct"]) >= float(base_summary["prob_auc_mean_pct"]) - 0.5)
    partial = acc_delta > 0 or g2_vs_g1 > 0 or flips > 0
    conclusion = "GO_CANDIDATE" if go else "PARTIAL" if partial else "NO-GO"
    audit_status = {"status": "PASS", "expected_checkpoints": 30, "independently_audited": 30,
        "checkpoint_metric_replay": "30/30 hard predictions exact; probability max error <=1e-8",
        "probability_replay_max_abs_error": max(float(row["probability_replay_max_abs_error"]) for row in audits),
        "warmup_initialization_checks": "10/10 through E74.4 prepare(fold)",
        "prototype_label_exclusion": "30/30 exact eligible train nodes; test labels unused",
        "fold2_excluded_from_new_prototypes": True,
        "all_rng_replay_interventions_unchanged": True, "source_sha256": source_hash,
        "scientific_conclusion": conclusion, "g2_acc_delta_vs_base_pp": acc_delta,
        "g2_acc_wins_vs_base": wins, "g2_module_off_hard_flips": flips,
        "audit_completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    run.atomic_json(RESULTS / "audit_status.json", audit_status)
    manifest.update({"status": "COMPLETE_AUDITED", "completed_tasks": 30, "independent_audits": 30,
        "checkpoint_audit": "PASS 30/30", "audit_status": audit_status,
        "scientific_conclusion": conclusion,
        "paired_effects": {"g2_acc_delta_vs_base_pp": acc_delta, "g2_acc_wins_vs_base": wins,
                           "g2_minus_g1_acc_pp": g2_vs_g1, "g2_minus_g3_acc_pp": g2_vs_g3,
                           "g2_module_off_flips": flips}})
    if update_handoff:
        report = build_report(manifest, audit_status, summary_rows, paired_rows, interventions, mechanisms)
        handoff_text = "## Experiment 75.2 — GC-SPER\n\n" + report
        try:
            manifest["handoff_update"] = append_handoff(handoff_text)
        except Exception as exc:
            pending = EXP / "results/handoff_pending.md"
            pending.write_text(handoff_text, encoding="utf-8")
            manifest["handoff_update"] = {"status": "FAILED", "error": repr(exc), "pending": str(pending)}
        report = build_report(manifest, audit_status, summary_rows, paired_rows, interventions, mechanisms)
        (RESULTS / "report.md").write_text(report, encoding="utf-8")
        if (RESULTS / "report.md").read_text(encoding="utf-8") != report:
            raise IOError("report readback mismatch")
    run.atomic_json(manifest_path, manifest)
    return manifest


def main():
    run_audit(update_handoff=True)


if __name__ == "__main__":
    main()
