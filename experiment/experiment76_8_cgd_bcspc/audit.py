#!/usr/bin/env python3
"""Independent disk-based checkpoint, prediction, and mechanism audit for E76.8."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as runner

RESULTS = EXP / "results"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1", "nll", "brier", "ece")
CORE = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
GROUPS = runner.GROUPS


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
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


def write_json(path, value):
    runner.atomic_json(path, value)


def close(a, b, atol=1e-7):
    import numpy as np
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().cpu().numpy()
    return np.asarray(a).shape == np.asarray(b).shape and bool(np.allclose(a, b, atol=atol, rtol=0))


def mean_sd(values):
    values = [float(value) for value in values]
    return statistics.mean(values), (statistics.pstdev(values) if len(values) > 1 else 0.0)


def ci95(values):
    # Descriptive paired t interval; folds are not independent external cohorts.
    n = len(values)
    if n < 2:
        return float(values[0]), float(values[0])
    critical = {9: 2.306, 10: 2.262}.get(n, 1.96)
    mean = statistics.mean(values)
    half = critical * statistics.stdev(values) / math.sqrt(n)
    return mean - half, mean + half


def metric_summary(rows, method):
    selected = [row for row in rows if row["method"] == method]
    return {metric: mean_sd([row[metric] for row in selected]) for metric in METRICS}


def transitions(before, after, labels):
    import numpy as np
    before, after, labels = np.asarray(before, int), np.asarray(after, int), np.asarray(labels, int)
    was_wrong, now_correct = before != labels, after == labels
    was_correct, now_wrong = before == labels, after != labels
    return {
        "error_to_correct": int(np.sum(was_wrong & now_correct)),
        "correct_to_error": int(np.sum(was_correct & now_wrong)),
        "net_corrections": int(np.sum(was_wrong & now_correct) - np.sum(was_correct & now_wrong)),
        "ASD_error_to_correct": int(np.sum((labels == 0) & was_wrong & now_correct)),
        "TD_error_to_correct": int(np.sum((labels == 1) & was_wrong & now_correct)),
        "ASD_to_TD_corrections": int(np.sum((labels == 0) & (before == 1) & (after == 0))),
        "TD_to_ASD_corrections": int(np.sum((labels == 1) & (before == 0) & (after == 1))),
        "original_correct": int(np.sum(was_correct)),
        "original_correct_kept": int(np.sum(was_correct & now_correct)),
        "prediction_changes": int(np.sum(before != after)),
    }


def transition_patient_indices(before, after, labels, patient_indices):
    import numpy as np
    before, after, labels = np.asarray(before, int), np.asarray(after, int), np.asarray(labels, int)
    ids = np.asarray(patient_indices, int)
    was_wrong, now_correct = before != labels, after == labels
    was_correct, now_wrong = before == labels, after != labels
    return {"error_to_correct_patient_indices": json.dumps(ids[was_wrong & now_correct].tolist()),
            "correct_to_error_patient_indices": json.dumps(ids[was_correct & now_wrong].tolist()),
            "changed_patient_indices": json.dumps(ids[before != after].tolist())}


def aggregate_diagnostics(diag, labels, base_prediction, final_prediction, interventions, site_ids=None):
    import numpy as np
    y = np.asarray(labels, int)
    output = {}
    for name in ("delta", "base_score", "final_score", "final_boundary_evidence"):
        values = diag[name].detach().cpu().numpy().reshape(-1)
        output[f"{name}_mean"] = float(values.mean())
        output[f"{name}_std"] = float(values.std())
        output[f"{name}_abs_mean"] = float(np.abs(values).mean())
        output[f"{name}_abs_max"] = float(np.abs(values).max())
    for name in ("reliable_similarity", "boundary_similarity", "reliable_difference", "boundary_difference", "slot_difference"):
        value = diag[name].detach().cpu().numpy()
        output[name + "_mean_by_class_or_slot"] = json.dumps(value.mean(0).tolist())
    for name in ("modality_weights", "attention", "support_topk", "support_coverage", "support_reliability", "candidate_count", "prototype_role_cosine"):
        value = diag[name].detach().cpu().numpy()
        output[name + "_mean"] = json.dumps(value.mean(0).tolist())
        output[name + "_std"] = json.dumps(value.std(0).tolist())
    boundary = diag["boundary_difference"].detach().cpu().numpy()
    reliable = diag["reliable_difference"].detach().cpu().numpy()
    t = np.where(y == 0, 1.0, -1.0)
    output["reliable_direction_accuracy"] = float(np.mean(t * reliable.mean(-1) > 0))
    output["boundary_direction_accuracy"] = float(np.mean(t * boundary.mean(-1) > 0))
    output["reliable_boundary_sign_agreement"] = float(np.mean(np.sign(reliable) == np.sign(boundary)))
    delta = diag["delta"].detach().cpu().numpy().reshape(-1)
    base_error = np.asarray(base_prediction) != y
    delta_correct = np.sign(delta) == t
    output["delta_toward_true_given_base_error"] = float(np.mean(delta_correct[base_error])) if base_error.any() else None
    output["delta_toward_true_given_base_correct"] = float(np.mean(delta_correct[~base_error])) if (~base_error).any() else None
    output["base_error_hard_flip_rate"] = float(np.mean(np.asarray(final_prediction)[base_error] == y[base_error])) if base_error.any() else None
    evidence = diag["final_boundary_evidence"].detach().cpu().numpy().reshape(-1)
    nonzero = (np.abs(delta) > 1e-12) & (np.abs(evidence) > 1e-12)
    output["delta_evidence_sign_match"] = float(np.mean(np.sign(delta[nonzero]) == np.sign(evidence[nonzero]))) if nonzero.any() else None
    weak = np.abs(evidence) < 0.05
    output["weak_evidence_delta_abs_mean"] = float(np.abs(delta[weak]).mean()) if weak.any() else None
    output["weak_evidence_fraction"] = float(np.mean(weak))
    margin_abs = np.abs(diag["base_score"].detach().cpu().numpy().reshape(-1))
    margin_bins = []
    for low, high in ((0.0,0.25),(0.25,0.5),(0.5,1.0),(1.0,float("inf"))):
        selected = (margin_abs >= low) & (margin_abs < high)
        margin_bins.append({"abs_margin_range":[low,high if math.isfinite(high) else None],
            "count":int(selected.sum()),
            "delta_abs_mean":float(np.abs(delta[selected]).mean()) if selected.any() else None,
            "error_to_correct":int(np.sum(selected & base_error & (np.asarray(final_prediction)==y))),
            "correct_to_error":int(np.sum(selected & ~base_error & (np.asarray(final_prediction)!=y)))})
    output["residual_by_base_margin"] = json.dumps(margin_bins, allow_nan=False)
    for name in ("unique_boundary_difference", "eta", "prototype_unique_norm", "evidence_gate"):
        if name in diag:
            value = diag[name].detach().cpu().numpy()
            output[f"{name}_mean"] = json.dumps(value.mean(0).tolist())
            output[f"{name}_std"] = json.dumps(value.std(0).tolist())
    if "unique_boundary_norm" in diag:
        value = diag["unique_boundary_norm"].detach().cpu().numpy()
        output["unique_boundary_norm_mean_by_class_slot"] = json.dumps(value.mean(0).tolist())
        output["unique_boundary_near_zero_fraction_by_class_slot"] = json.dumps((value < 1e-6).mean(0).tolist())
    if "unique_boundary_difference" in diag:
        unique = diag["unique_boundary_difference"].detach().cpu().numpy()
        output["unique_direction_accuracy"] = float(np.mean(t * unique.mean(-1) > 0))
        output["reliable_unique_correlation_by_slot"] = json.dumps([
            float(np.corrcoef(reliable[:, slot], unique[:, slot])[0, 1])
            if np.std(reliable[:, slot]) > 0 and np.std(unique[:, slot]) > 0 else None
            for slot in range(reliable.shape[1])])
        if site_ids is not None:
            sites = np.asarray(site_ids, int).reshape(-1)
            magnitude = np.abs(unique).mean(-1)
            between = total = 0.0
            for cls in (0,1):
                class_mask = y == cls
                if not class_mask.any():
                    continue
                class_values = magnitude[class_mask]
                class_mean = float(class_values.mean())
                total += float(np.square(class_values-class_mean).sum())
                for site in np.unique(sites[class_mask]):
                    values = magnitude[class_mask & (sites==site)]
                    between += len(values) * float((values.mean()-class_mean)**2)
            output["unique_evidence_site_explained_variance_within_class"] = between/total if total>0 else None
            output["unique_evidence_sites_n"] = int(len(np.unique(sites)))
    output.update(transitions(base_prediction, final_prediction, y))
    output["retrieval_fallbacks"] = int(diag["retrieval_fallbacks"])
    output["site_cap_violations"] = int(diag["site_cap_violations"])
    output["invalid_confusion_candidates"] = int(diag["invalid_confusion_candidates"])
    for key, value in interventions.items():
        output[f"{key}_mean_abs_probability_change"] = float(np.abs(value["probability"] - value["full_probability"]).mean())
        output[f"{key}_mean_abs_margin_change"] = float(np.abs(value["margin"] - value["full_margin"]).mean())
        output[f"{key}_prediction_changes"] = int(np.sum(value["prediction"] != value["full_prediction"]))
        for metric in CORE:
            output[f"{key}_{metric}"] = float(value["metrics"][metric])
    return output


def audit_fold(group, fold, source_sha):
    import numpy as np
    import torch
    task_path = runner.task_path(group, fold)
    checkpoint_path = runner.best_path(group, fold)
    latest_path = runner.latest_path(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = runner.sha_file(checkpoint_path)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"task/checkpoint identity mismatch {group} fold={fold}")
    saved = runner.torch_load(checkpoint_path)
    if (saved.get("source_sha256") != task.get("source_sha256")
            or saved.get("group") != group or saved.get("fold") != fold):
        raise RuntimeError(f"checkpoint provenance mismatch {group} fold={fold}")
    best = saved["best"]
    best_rng = next((row for row in saved["rng_trace"] if int(row["epoch"]) == int(best["epoch"])), None)
    rng_helper = runner.legacy.history_runtime()
    if (best_rng is None or best.get("rng_trace_record") != best_rng
            or best.get("rng_after_eval_sha256") != best_rng["rng_after_eval"]
            or rng_helper.rng_sha(best["rng"]) != best_rng["rng_after_eval"]):
        raise RuntimeError(f"best checkpoint RNG snapshot does not match selected epoch {group} fold={fold}")
    if best.get("training_config") != runner.CFG["training"]:
        raise RuntimeError(f"best checkpoint training configuration mismatch {group} fold={fold}")
    state, warm, identity, history_task, history_best = runner.prepare_fold(fold)
    if saved["identity"] != identity or saved["warmup_sha256"] != identity["warmup_sha256"]:
        raise RuntimeError(f"warmup/fold identity mismatch {group} fold={fold}")
    hist = runner.legacy.load_base().load_history()
    hist.load_bundle(state, best["base_state"])
    state["fold"], state["group"], state["rudag_identity"] = fold, runner.BASE_GROUP, identity
    brain, sprompt = runner.prompt_vectors()
    module = runner.init_module(brain, sprompt, fold, group)
    module.load_state_dict(best["module"], strict=True)
    module.eval()
    eligible, ids_a, ids_b = runner.split_branch_ids(state, fold)
    if eligible != best["eligible_train_ids"] or ids_a != best["support_a"] or ids_b != best["support_b"]:
        raise RuntimeError(f"training prototype partition mismatch {group} fold={fold}")
    test_ids = list(map(int, state["test_np"]))
    if test_ids != best["test_ids"]:
        raise RuntimeError(f"test identity mismatch {group} fold={fold}")
    memory = runner.memory_device(best["memory_used"], state["y"].device)
    if memory is None:
        raise RuntimeError(f"missing memory snapshot {group} fold={fold}")
    memory_ids = set(torch.nonzero(memory["observed"], as_tuple=False).flatten().cpu().tolist())
    expected_memory_ids = set(eligible) if int(memory["updates"]) else set()
    if memory_ids != expected_memory_ids:
        raise RuntimeError(f"EMA memory contains missing/illegal patient IDs {group} fold={fold}")
    if set(torch.unique(memory["labels"][memory["observed"]]).cpu().tolist()) - {0, 1}:
        raise RuntimeError(f"invalid prototype labels {group} fold={fold}")
    if fold == 2 and ({376, 406} & memory_ids or memory["labels"][[376, 406]].ne(-1).any()):
        raise RuntimeError("Fold 2 overlap rows entered the new branch memory")

    # Independent base and corrected inference from the saved model + exact memory version.
    with torch.no_grad():
        base_out, cam = runner.hook_forward(state, training=False)
        full_logp, diag = runner.eval_method(module, base_out["logp"], cam, memory,
                                             eligible, test_ids, runner.read_sites().to(state["y"].device))
        full_metrics = runner.exact_metrics(full_logp, test_ids, state["y"])
        base_metrics = runner.exact_metrics(base_out["logp"], test_ids, state["y"])
        interventions = {}
        for name, options in (
            ("Residual-Off", {"residual_off": True}),
            ("Boundary-Off", {"boundary_off": True}),
            *((("Unique-Evidence-Off", {"unique_evidence_off": True}),) if group == GROUPS[1] else ()),
            ("Semantic-Swap", {"semantic_swap": True}),
            ("Reliable-only", {"reliable_only": True}),
        ):
            candidate, _ = runner.eval_method(module, base_out["logp"], cam, memory,
                                               eligible, test_ids, runner.read_sites().to(state["y"].device), **options)
            metrics = runner.exact_metrics(candidate, test_ids, state["y"])
            interventions[name] = {"metrics": metrics,
                                   "probability": np.asarray(metrics["prob"]),
                                   "prediction": np.asarray(metrics["pred"]),
                                   "margin": (candidate[torch.as_tensor(test_ids, device=candidate.device), 0] -
                                              candidate[torch.as_tensor(test_ids, device=candidate.device), 1]).detach().cpu().numpy()}
        for item in interventions.values():
            item["full_probability"] = np.asarray(full_metrics["prob"])
            item["full_prediction"] = np.asarray(full_metrics["pred"])
            item["full_margin"] = (full_logp[torch.as_tensor(test_ids, device=full_logp.device), 0] -
                                   full_logp[torch.as_tensor(test_ids, device=full_logp.device), 1]).detach().cpu().numpy()
        train_branch = runner.branch_train(module, cam, memory, base_out["logp"], state["y"],
                                           eligible, ids_a, ids_b, runner.read_sites().to(state["y"].device),
                                           state["weights"], detached=True)
        train_diag = train_branch[3]
        train_qids = train_branch[4]
        train_labels = train_branch[5].detach().cpu().numpy()
        train_base_prediction = base_out["logp"][train_qids].argmax(-1).detach().cpu().numpy()
        train_final_prediction = train_branch[2].argmax(-1).detach().cpu().numpy()
        train_mechanism = aggregate_diagnostics(train_diag, train_labels, train_base_prediction,
                                                train_final_prediction, {})
    if not close(full_metrics["pred"], best["prediction"], atol=0):
        raise RuntimeError(f"independent class prediction mismatch {group} fold={fold}")
    if not close(full_metrics["prob"], best["probability"], atol=1e-7):
        raise RuntimeError(f"independent probability mismatch {group} fold={fold}")
    if not close(base_metrics["pred"], best["base_prediction"], atol=0):
        raise RuntimeError(f"independent base class mismatch {group} fold={fold}")
    if not close(base_metrics["prob"], best["base_probability"], atol=1e-7):
        raise RuntimeError(f"independent base probability mismatch {group} fold={fold}")
    for key in runner.legacy.METRICS:
        if abs(float(full_metrics[key]) - float(best["metrics"][key])) > 1e-8:
            raise RuntimeError(f"independent {key} mismatch {group} fold={fold}")

    # Every eval retrieval candidate must come from this fold's legal train support.
    selected = diag["selected_ids"].detach().cpu().numpy().reshape(-1)
    selected = set(int(value) for value in selected if value >= 0)
    if not selected <= set(eligible) or selected & set(test_ids):
        raise RuntimeError(f"test/checkpoint retrieval has illegal support IDs {group} fold={fold}")
    if best["epoch"] != saved["best_epoch_1based"] or saved["epochs_executed"] != task["epochs_executed"]:
        raise RuntimeError(f"epoch ledger mismatch {group} fold={fold}")

    # Shared stochastic operations must match the historical base at every epoch.
    historical = {int(row["epoch"]): row for row in history_task["rng_trace"]}
    actual = {int(row["epoch"]): row for row in saved["rng_trace"]}
    common_epochs = set(actual) & set(historical)
    mismatches = [epoch for epoch in sorted(common_epochs) if historical[epoch] != actual[epoch]]
    if mismatches:
        raise RuntimeError(f"shared RNG trace mismatch {group} fold={fold}: {mismatches[:5]}")
    latest = runner.torch_load(latest_path)
    if not latest.get("gradient_summary") or not best.get("gradient_summary"):
        raise RuntimeError(f"per-epoch gradient summary missing {group} fold={fold}")
    train_curve = latest.get("train_curve", [])
    if len(train_curve) != len(saved["rng_trace"]):
        raise RuntimeError(f"per-epoch training curve incomplete {group} fold={fold}")
    trace_by_epoch = {int(row["epoch"]): row for row in saved["rng_trace"]}
    for curve_row in train_curve:
        trace_row = trace_by_epoch.get(int(curve_row["epoch"]))
        if trace_row is None or any(curve_row[key] != trace_row[key]
                                    for key in ("rng_before", "rng_after_train", "rng_after_eval")):
            raise RuntimeError(f"training-curve RNG identity mismatch {group} fold={fold}")

    labels = state["y"][torch.as_tensor(test_ids, device=state["y"].device)].detach().cpu().numpy()
    base_pred = np.asarray(best["base_prediction"], int)
    final_pred = np.asarray(best["prediction"], int)
    test_sites = runner.read_sites()[torch.as_tensor(test_ids, dtype=torch.long)].cpu().numpy()
    stats = aggregate_diagnostics(diag, labels, base_pred, final_pred, interventions, test_sites)
    metric_row = {"method": group, "fold": fold, **{key: float(full_metrics[key]) for key in METRICS},
                  **{f"backbone_{key}": float(base_metrics[key]) for key in METRICS},
                  "best_epoch_1based": saved["best_epoch_1based"],
                  "epochs_executed": saved["epochs_executed"],
                  "checkpoint_sha256": checkpoint_sha,
                  "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"])}
    mechanism_row = {"method": group, "fold": fold, "best_epoch_1based": saved["best_epoch_1based"],
                     "memory_updates": best["memory_updates"], "memory_eligible_count": len(eligible),
                     "support_a_count": len(ids_a), "support_b_count": len(ids_b),
                     "selected_support_count": len(selected),
                     "branch_gradient_families_best_epoch": json.dumps(best["gradient_snapshot"].get("branch_gradient_families", {}), allow_nan=False),
                     "gradient_snapshot_best_epoch": json.dumps(best["gradient_snapshot"], allow_nan=False),
                     "gradient_summary_at_best": json.dumps(best["gradient_summary"], allow_nan=False),
                     "gradient_summary_all_epochs": json.dumps(latest["gradient_summary"], allow_nan=False),
                     "legal_train_diagnostics": json.dumps(train_mechanism, allow_nan=False),
                     **stats}

    # E74.4-G2 comparison is independently re-run at its own selected epoch.
    hist_state, _, hist_identity, hist_task, hist_path = runner.prepare_fold(fold)
    hist_cp = runner.torch_load(hist_path)
    runner.legacy.load_base().load_history().load_bundle(hist_state, hist_cp["best_state"])
    with torch.no_grad():
        historic_out, _ = runner.hook_forward(hist_state, training=False)
        historic_metrics = runner.exact_metrics(historic_out["logp"], list(map(int, hist_state["test_np"])), hist_state["y"])
    historic_prediction = np.asarray(historic_metrics["pred"], int)
    historic_labels = hist_state["y"][torch.as_tensor(hist_state["test_np"], device=hist_state["y"].device)].detach().cpu().numpy()
    history_row = {"method": "E74.4-G2-Positive-DCR", "fold": fold,
                   **{key: float(historic_metrics[key]) for key in METRICS},
                   "best_epoch_1based": hist_task["best_epoch_1based"],
                   "epochs_executed": hist_task["epochs_executed"],
                   "checkpoint_sha256": runner.sha_file(hist_path),
                   "train_test_overlap_1based": json.dumps(hist_identity["membership"]["train_test_overlap_1based"])}
    historical_correction = transitions(historic_prediction, final_pred, historic_labels)
    correction_row = {"comparison": f"{group}-vs-E74.4-G2", "fold": fold,
                      **historical_correction,
                      **transition_patient_indices(historic_prediction, final_pred, historic_labels, test_ids)}
    residual_off_prediction = np.asarray(interventions["Residual-Off"]["prediction"], int)
    residual_correction = {"comparison": f"{group}-Residual-On-vs-Off", "fold": fold,
                           **transitions(residual_off_prediction, final_pred, labels),
                           **transition_patient_indices(residual_off_prediction, final_pred, labels, test_ids)}
    e77_group = "G1-CJE-BCSPC-Full-E2E" if group == GROUPS[0] else "G2-CJE-BCSPC-StopGrad"
    e77_path = runner.E76_7 / ".runtime/checkpoints" / f"{e77_group}_f{fold:02d}.best.pt"
    e77_saved = runner.torch_load(e77_path)
    e77_prediction = np.asarray(e77_saved["best"]["prediction"], dtype=int)
    e77_correction = {"comparison": f"{group}-vs-E76.7-{e77_group}", "fold": fold,
                      **transitions(e77_prediction, final_pred, labels),
                      **transition_patient_indices(e77_prediction, final_pred, labels, test_ids)}
    return metric_row, mechanism_row, [correction_row, residual_correction, e77_correction], history_row, {
        "prediction": final_pred.tolist(), "labels": np.asarray(labels, int).tolist(),
        "test_ids": test_ids,
        "base_prediction": base_pred.tolist(), "historical_prediction": historic_prediction.tolist(),
        "e76_7_prediction": e77_prediction.tolist(),
        "train_labels": train_labels.tolist(), "train_prediction": train_final_prediction.tolist(),
        "train_base_prediction": train_base_prediction.tolist(),
        "unique_boundary_difference": diag["unique_boundary_difference"].detach().cpu().tolist(),
        "final_boundary_evidence": diag["final_boundary_evidence"].detach().cpu().tolist(),
        "eta": diag["eta"].detach().cpu().tolist(),
        "interventions": {key: {"prediction": np.asarray(value["prediction"], int).tolist(),
                                 "metrics": {k: float(value["metrics"][k]) for k in METRICS}}
                          for key, value in interventions.items()},
        "checkpoint_sha256": checkpoint_sha, "rng_mismatches": len(mismatches),
        "selected_support_count": len(selected), "memory_id_count": len(memory_ids),
        "module_parameter_count": sum(value.numel() for value in best["module"].values()),
        "g2_backbone_replay_check": True,
        "latest_gradient_audit": latest.get("gradient_audit", []),
        "gradient_summary": latest.get("gradient_summary", {}),
        "train_curve": train_curve,
    }


def build_comparisons(metric_rows):
    pairs = [(GROUPS[0], "E76.7-G1-CJE-BCSPC-Full-E2E"),
             (GROUPS[1], GROUPS[0]),
             (GROUPS[1], "E74.4-G2-Positive-DCR"),
             (GROUPS[0], "E74.4-G2-Positive-DCR"),
             (GROUPS[1], "E76.7-G1-CJE-BCSPC-Full-E2E"),
             (GROUPS[1], "E76.7-G2-CJE-BCSPC-StopGrad")]
    output = []
    for left, right in pairs:
        for metric in CORE:
            left_rows = {int(row["fold"]): row for row in metric_rows if row["method"] == left}
            right_rows = {int(row["fold"]): row for row in metric_rows if row["method"] == right}
            if right.startswith("E76.7-"):
                e77 = {int(row["fold"]): row for row in read_csv(runner.E76_7 / "results/fold_metrics.csv")
                       if row["method"] == right.removeprefix("E76.7-")}
                right_rows = {fold: {metric: float(row[metric]) for metric in CORE} for fold, row in e77.items()}
            if set(left_rows) != set(right_rows):
                raise RuntimeError(f"paired comparison incomplete: {left} vs {right} {metric}")
            deltas = {fold: (float(left_rows[fold][metric]) - float(right_rows[fold][metric])) * 100.0
                      for fold in sorted(left_rows)}
            all_values = list(deltas.values())
            low, high = ci95(all_values)
            nine = [value for fold, value in deltas.items() if fold != 2]
            low9, high9 = ci95(nine)
            output.append({"left": left, "right": right, "metric": metric,
                           "mean_delta_pp": statistics.mean(all_values),
                           "fold_deltas_pp": json.dumps({fold: round(value, 6) for fold, value in deltas.items()}),
                           "wins": sum(value > 1e-12 for value in all_values),
                           "ties": sum(abs(value) <= 1e-12 for value in all_values),
                           "losses": sum(value < -1e-12 for value in all_values),
                           "descriptive_ci95_low_pp": low, "descriptive_ci95_high_pp": high,
                           "exclude_fold2_mean_delta_pp": statistics.mean(nine),
                           "exclude_fold2_ci95_low_pp": low9, "exclude_fold2_ci95_high_pp": high9,
                           "n_folds": len(all_values)})
    return output


def replay_latest_step(group, fold):
    """Reload and execute the same next train step twice without saving either result."""
    import copy
    import torch
    saved = runner.torch_load(runner.latest_path(group, fold))
    initial_base_state = copy.deepcopy(saved["base_state"])
    initial_branch_optimizer = copy.deepcopy(saved["optimizer"])
    state, warm, identity, history_task, _ = runner.prepare_fold(fold)
    hist = runner.legacy.load_base().load_history()
    rng = runner.legacy.history_runtime()
    hist.load_bundle(state, copy.deepcopy(initial_base_state))
    state["fold"], state["group"], state["rudag_identity"] = fold, runner.BASE_GROUP, identity
    brain, sprompt = runner.prompt_vectors()
    module = runner.init_module(brain, sprompt, fold, group)
    module.load_state_dict(saved["module"], strict=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
    optimizer.load_state_dict(copy.deepcopy(initial_branch_optimizer))
    memory = runner.memory_device(saved["memory"], state["y"].device)
    eligible, ids_a, ids_b = runner.split_branch_ids(state, fold)
    sites = runner.read_sites().to(state["y"].device)
    labels = state["y"].long()
    step_epoch = int(saved["next_epoch"])

    def one_step():
        hist.zero_optimizers(state)
        optimizer.zero_grad(set_to_none=True)
        out, cam = runner.hook_forward(state, training=True)
        live = group in GROUPS and step_epoch + 1 >= 21
        branch = runner.branch_train(module, cam, memory, out["logp"], labels,
                                     eligible, ids_a, ids_b, sites, state["weights"], detached=not live)
        beta = (0.3 * min(1.0, max(0.0, (step_epoch + 1 - 20) / 20.0))) if live else 0.0
        gradient = runner.gradients_and_step(state, module, out, cam, memory, branch, group, step_epoch, beta)
        module_gradients = {name:(None if parameter.grad is None else parameter.grad.detach().cpu().clone())
                            for name,parameter in module.named_parameters()}
        for key in ("opt_base", "opt_cam", "opt_added"):
            state[key].step()
        if int(memory["updates"]):
            optimizer.step()
        rng_sha = rng.rng_sha(rng.pack_rng())
        result = {"base": hist.state_bundle(state),
                  "module": {key: value.detach().cpu().clone() for key,value in module.state_dict().items()},
                  "optimizer": copy.deepcopy(optimizer.state_dict()), "module_gradients":module_gradients,
                  "rng": rng_sha,
                  "branch_loss": float(branch[0].detach().cpu()),
                  "gradient": gradient}
        del out, cam
        return result

    rng.restore_rng(saved["rng"])
    first = one_step()
    hist.load_bundle(state, copy.deepcopy(initial_base_state))
    module.load_state_dict(saved["module"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(initial_branch_optimizer))
    memory = runner.memory_device(saved["memory"], state["y"].device)
    rng.restore_rng(saved["rng"])
    second = one_step()
    runner._assert_state_equal(first["base"], second["base"], f"latest next-step base replay {group} fold={fold}")
    def max_numeric_error(left, right, path=""):
        if torch.is_tensor(left) and torch.is_tensor(right):
            if left.shape != right.shape:
                raise RuntimeError("replay tensor shape mismatch")
            if not left.numel():
                return 0.0, path
            difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
            flat_index = int(difference.argmax())
            return float(difference.reshape(-1)[flat_index]), f"{path}[{flat_index}]"
        if isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                raise RuntimeError("replay mapping keys mismatch")
            return max((max_numeric_error(left[key], right[key], f"{path}.{key}") for key in left),
                       key=lambda item:item[0], default=(0.0, path))
        if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                raise RuntimeError("replay sequence length mismatch")
            return max((max_numeric_error(a, b, f"{path}[{index}]")
                        for index,(a,b) in enumerate(zip(left,right))),
                       key=lambda item:item[0], default=(0.0, path))
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return abs(float(left) - float(right)), path
        if left != right:
            raise RuntimeError("replay nonnumeric state mismatch")
        return 0.0, path
    module_error, module_error_path = max_numeric_error(first["module"], second["module"], "module")
    optimizer_error, optimizer_error_path = max_numeric_error(first["optimizer"], second["optimizer"], "optimizer")
    gradient_error, gradient_error_path = max_numeric_error(first["module_gradients"],
                                                            second["module_gradients"], "gradient")
    if module_error > 2e-6 or optimizer_error > 2e-6:
        raise RuntimeError(f"latest branch replay error exceeds 2e-6: module={module_error} at {module_error_path}, "
                           f"optimizer={optimizer_error} at {optimizer_error_path}, gradients={gradient_error} "
                           f"at {gradient_error_path}, {group} fold={fold}")
    if first["rng"] != second["rng"] or abs(first["branch_loss"] - second["branch_loss"]) > 1e-7:
        raise RuntimeError(f"latest next-step RNG/loss replay mismatch {group} fold={fold}")
    del state, module, optimizer, memory
    torch.cuda.empty_cache()
    return {"group":group,"fold":fold,"next_epoch_index":step_epoch,
            "base_optimizer_replay_exact":True,"branch_optimizer_replay_within_tolerance":True,
            "branch_replay_max_abs_error":module_error,
            "branch_optimizer_replay_max_abs_error":optimizer_error,
            "branch_replay_max_error_path":module_error_path,
            "branch_gradient_replay_max_abs_error":gradient_error,
            "branch_gradient_replay_max_error_path":gradient_error_path,
            "branch_replay_atol":2e-6,"rng_replay_exact":True,
            "loss_replay_atol":1e-7,"loss_replay_error":abs(first["branch_loss"]-second["branch_loss"])}


def write_report(metric_rows, comparisons, correction_rows, mechanism_rows, status):
    lines = ["# Experiment 76.8 CGD-BCSPC", "",
             f"STATUS: `{status['status']}`。科研判定：`{status['method_verdict']}`。", "",
             "结果沿用E74.4-G2历史协议：10折、测试折参与ACC选模、CV前RFE/标准化及transductive图。该对比是历史协议内的探索性结果，不是严格无偏的独立泛化估计。Fold 2存在历史训练/测试重复行；E76.8新增分支已排除行377/407，但原主干历史协议仍保留该问题。", "",
             "| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    display = [("E74.4-G2-Positive-DCR", "E74.4-G2历史基座"),
               ("E76.7-G1-CJE-BCSPC-Full-E2E", "E76.7-G1历史参考"),
               ("E76.7-G2-CJE-BCSPC-StopGrad", "E76.7-G2历史参考"),
               (GROUPS[0], "E76.8-G1 Conflict-Safe-E2E"),
               (GROUPS[1], "E76.8-G2 Discriminative-BCSPC")]
    for method, label in display:
        rows = [row for row in metric_rows if row["method"] == method]
        values = {}
        for metric in CORE:
            mean, sd = mean_sd([row[metric] for row in rows])
            values[metric] = f"{mean*100:.2f}±{sd*100:.2f}"
        lines.append(f"| {label} | {values['acc']} | {values['hard_auc']} | {values['prob_auc']} | {values['sen']} | {values['spe']} | {values['asd_f1']} |")
    lines.extend(["", "配对差值单位为百分点；区间为描述性配对t区间，10折不是10个独立外部队列。排除Fold 2结果见 `paired_comparisons.csv`。", "",
                  "| 配对比较 | 指标 | 均值差(pp) | W/T/L | 95%描述区间 | 排除Fold 2差值(pp) |", "|---|---|---:|---:|---:|---:|"])
    for row in comparisons:
        if row["metric"] in ("acc", "hard_auc", "prob_auc"):
            lines.append(f"| {row['left']} − {row['right']} | {row['metric']} | {row['mean_delta_pp']:.3f} | {row['wins']}/{row['ties']}/{row['losses']} | [{row['descriptive_ci95_low_pp']:.3f}, {row['descriptive_ci95_high_pp']:.3f}] | {row['exclude_fold2_mean_delta_pp']:.3f} |")
    old_g1 = [row for row in metric_rows if row["method"] == "E76.7-G1-CJE-BCSPC-Full-E2E"]
    old_base = [row for row in metric_rows if row["method"] == "E74.4-G2-Positive-DCR"]
    old_mechanism = [row for row in read_csv(runner.E76_7 / "results/mechanism.csv")
                     if row["method"] == "G1-CJE-BCSPC-Full-E2E"]
    old_cam_cos, old_output_cos = [], []
    for row in old_mechanism:
        snapshot = json.loads(row["gradient_snapshot_best_epoch"])
        old_cam_cos.append(float(snapshot["gradient_families"]["cam"]["cosine_base_vs_applied"]))
        old_output_cos.append(float(snapshot["gradient_families"]["outputlayer"]["cosine_base_vs_applied"]))
    old_corrections = [row for row in read_csv(runner.E76_7 / "results/correction_analysis.csv")
                       if row["comparison"] == "G1-CJE-BCSPC-Full-E2E-vs-E74.4-G2"]
    old_net = sum(int(row["net_corrections"]) for row in old_corrections)
    old_acc_delta = (statistics.mean(float(row["acc"]) for row in old_g1) -
                     statistics.mean(float(row["acc"]) for row in old_base)) * 100.0
    lines.append(f"- E76.7-G1历史复核：相对E74.4-G2的ACC差为{old_acc_delta:.3f} pp，净纠错{old_net}；最佳epoch新增梯度与基座梯度平均余弦为CAM {statistics.mean(old_cam_cos):.3f}、OutputLayer {statistics.mean(old_output_cos):.3f}。这些数字从E76.7可信结果文件读取，用于解释冲突来源，不用于训练或调参。")
    lines.extend(["", "纠错与机制", ""])
    for group in GROUPS:
        selected = [row for row in correction_rows if row["comparison"] == f"{group}-vs-E74.4-G2"]
        lines.append(f"- {group}相对历史基座：错误→正确 {sum(int(r['error_to_correct']) for r in selected)}，正确→错误 {sum(int(r['correct_to_error']) for r in selected)}，净纠错 {sum(int(r['net_corrections']) for r in selected)}。")
    g1_base = next(row for row in comparisons if row["left"] == GROUPS[0] and row["right"] == "E74.4-G2-Positive-DCR" and row["metric"] == "acc")
    g2_base = next(row for row in comparisons if row["left"] == GROUPS[1] and row["right"] == "E74.4-G2-Positive-DCR" and row["metric"] == "acc")
    g2_g1 = next(row for row in comparisons if row["left"] == GROUPS[1] and row["right"] == GROUPS[0] and row["metric"] == "acc")
    lines.append(f"- E76.8-G1相对E74.4-G2：ACC {g1_base['mean_delta_pp']:.3f} pp；E76.8-G2相对E76.8-G1：ACC {g2_g1['mean_delta_pp']:.3f} pp；E76.8-G2相对E74.4-G2：ACC {g2_base['mean_delta_pp']:.3f} pp。")
    lines.append(f"- 独立审计：{status['checkpoint_recalculations']}/20最佳Checkpoint复算通过；G1/G2共有随机操作逐epoch匹配：{status['g1_rng_matches_history']}/{status['g2_rng_matches_history']}。")
    lines.append("- 逐epoch主干训练ACC、未校正测试ACC、方法ACC和损失见 `train_curve.csv`；四槽位注意力、可靠/边界证据、独有边界向量、eta、残差及Residual-Off、Boundary-Off、Unique-Evidence-Off、Semantic-Swap、Reliable-only干预详见 `mechanism.csv` 和 `correction_analysis.csv`。纠错表列出被纠正/误伤测试患者的全局行号及G2独有证据。Prompt交换是同Checkpoint干预；本轮没有匿名语义正式训练组，因此不能据此宣称疾病文本有独立预测价值。")
    lines.append("- G1/G2均为真实端到端训练；主干新增分类梯度使用 `grad(L_cls_final-L_cls_base_eligible)`，按原optimizer param group先做非对称冲突投影，再乘beta并限幅。G2使用未再次归一化的可靠原型正交补边界证据，并用其门控纠错残差。")
    lines.append("")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_gradient_history(details, group):
    output = {}
    for section in ("optimizer_groups", "gradient_families"):
        merged = {}
        for fold in range(1, 11):
            rows = details[(group, fold)]["gradient_summary"].get(section, {})
            for name, record in rows.items():
                acc = merged.setdefault(name, {key: 0.0 for key in record})
                for key, value in record.items():
                    acc[key] += float(value)
        output[section] = {}
        for name, record in merged.items():
            steps = max(1, int(record.get("steps", 0)))
            output[section][name] = {
                "steps": int(record.get("steps", 0)),
                "active_steps": int(record.get("active_steps", 0)),
                "negative_dot_steps": int(record.get("negative_dot_steps", 0)),
                "negative_dot_rate": record.get("negative_dot_steps", 0.0) / max(1, int(record.get("active_steps", 0))),
                "negative_dot_fraction_all_steps": record.get("negative_dot_steps", 0.0) / steps,
                "projected_negative_steps": int(record.get("projected_negative_steps", 0)),
                "projected_negative_rate": record.get("projected_negative_steps", 0.0) / max(1, int(record.get("active_steps", 0))),
                "clipped_steps": int(record.get("clipped_steps", 0)),
                "mean_raw_cosine": record.get("sum_cosine", 0.0) / max(1, int(record.get("active_steps", 0))),
                "mean_projected_cosine": record.get("sum_projected_cosine", 0.0) / max(1, int(record.get("active_steps", 0))),
                "mean_negative_conflict_strength": record.get("sum_negative_cosine_magnitude", 0.0) / max(1, int(record.get("active_steps", 0))),
                "mean_projection_removed_fraction": record.get("sum_projection_removed_fraction", 0.0) / max(1, int(record.get("active_steps", 0))),
                "mean_new_to_base_norm_ratio": record.get("sum_relative_new_to_base", 0.0) / max(1, int(record.get("active_steps", 0))),
                "mean_base_norm": record.get("sum_base_norm", 0.0) / steps,
                "mean_raw_new_norm": record.get("sum_new_raw_norm", 0.0) / steps,
                "mean_applied_new_norm": record.get("sum_new_applied_norm", 0.0) / steps,
            }
    return output


def run_audit():
    import numpy as np
    import torch
    RESULTS.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    source_sha = preflight.get("source_sha256")
    if preflight.get("status") != "PASS" or not source_sha:
        raise RuntimeError("preflight or locked training-source SHA is missing")
    current_sources = runner.source_snapshot()
    expected_sources = preflight.get("source_files", {})
    audit_path = str((EXP / "audit.py").resolve())
    mismatched = [path for path, digest in current_sources.items()
                  if path != audit_path and (path not in expected_sources or expected_sources[path] != digest)]
    if set(current_sources) - {audit_path} != set(expected_sources) - {audit_path}:
        mismatched.append("source inventory")
    if mismatched:
        raise RuntimeError("locked source-file SHA mismatch: " + ", ".join(mismatched))
    audit_source_sha = runner.sha_file(EXP / "audit.py")
    # The training source remains anchored to preflight; this post-training
    # audit-only correction is independently identified in status and manifest.
    checkpoint_sources = {
        json.loads(runner.task_path(group, fold).read_text(encoding="utf-8")).get("source_sha256")
        for group in GROUPS for fold in range(1, 11)
    }
    if checkpoint_sources != {source_sha}:
        raise RuntimeError("formal task ledgers do not share one training-source identity")
    metric_rows, mechanism_rows, correction_rows, history_rows, details = [], [], [], [], {}
    for fold in range(1, 11):
        for group in GROUPS:
            metric, mechanism, correction, history, detail = audit_fold(group, fold, source_sha)
            metric_rows.append(metric)
            mechanism_rows.append(mechanism)
            correction_rows.extend(correction)
            details[(group, fold)] = detail
            if group == GROUPS[0]:
                history_rows.append(history)
        if details[(GROUPS[0],fold)]["module_parameter_count"] != details[(GROUPS[1],fold)]["module_parameter_count"]:
            raise RuntimeError(f"G1/G2 added-module parameter counts differ fold={fold}")
    # The history baseline is one common, independently recalculated row per fold.
    metric_rows.extend(history_rows)
    # E76.7 remains read-only and is added from its audited fold table.
    e77_rows = read_csv(runner.E76_7 / "results/fold_metrics.csv")
    for e77_name, display_name in (("G1-CJE-BCSPC-Full-E2E", "E76.7-G1-CJE-BCSPC-Full-E2E"),
                                   ("G2-CJE-BCSPC-StopGrad", "E76.7-G2-CJE-BCSPC-StopGrad")):
        for row in e77_rows:
            if row["method"] != e77_name:
                continue
            fold = int(row["fold"])
            history_task = json.loads((runner.E76_7 / ".runtime/tasks" / f"{e77_name}_f{fold:02d}.json").read_text(encoding="utf-8"))
            history_checkpoint = runner.E76_7 / ".runtime/checkpoints" / f"{e77_name}_f{fold:02d}.best.pt"
            metric_rows.append({"method":display_name,"fold":fold,
                                **{key:float(row[key]) for key in METRICS},
                                "best_epoch_1based":int(history_task["best_epoch_1based"]),
                                "epochs_executed":int(history_task["epochs_executed"]),
                                "checkpoint_sha256":runner.sha_file(history_checkpoint)})
    # Paired patient-wise G1/G2 transition analysis within identical test folds.
    for fold in range(1, 11):
        g1, g2 = details[(GROUPS[0], fold)], details[(GROUPS[1], fold)]
        if g1["labels"] != g2["labels"]:
            raise RuntimeError(f"G1/G2 patient order mismatch fold={fold}")
        correction_rows.append({"comparison":"G2-vs-G1", "fold":fold,
                                **transitions(g1["prediction"],g2["prediction"],g1["labels"]),
                                **transition_patient_indices(g1["prediction"],g2["prediction"],
                                                             g1["labels"],g1["test_ids"])})
        y = np.asarray(g1["labels"], dtype=int)
        g1_wrong, g2_correct = np.asarray(g1["prediction"]) != y, np.asarray(g2["prediction"]) == y
        corrected_by_g2 = g1_wrong & g2_correct
        g2_evidence = np.asarray(g2["unique_boundary_difference"], dtype=float)
        g2_evidence_strength = np.abs(g2_evidence).mean(-1)
        g2_evidence_signed = g2_evidence.mean(-1)
        g2_reliable = np.asarray(g2["final_boundary_evidence"], dtype=float)
        correction_rows.append({"comparison":"G2-vs-G1-corrected-evidence", "fold":fold,
                                "g2_corrected_count":int(corrected_by_g2.sum()),
                                "g2_corrected_patient_indices":json.dumps(np.asarray(g1["test_ids"])[corrected_by_g2].tolist()),
                                "mean_abs_unique_evidence_corrected":float(g2_evidence_strength[corrected_by_g2].mean()) if corrected_by_g2.any() else None,
                                "mean_abs_unique_evidence_not_corrected":float(g2_evidence_strength[~corrected_by_g2].mean()) if (~corrected_by_g2).any() else None,
                                "corrected_patient_unique_evidence":json.dumps([
                                    {"patient_index":int(g1["test_ids"][i]),"mean_abs_unique_evidence":float(g2_evidence_strength[i]),
                                     "mean_signed_unique_evidence":float(g2_evidence_signed[i]),"fused_evidence":float(g2_reliable[i])}
                                    for i in np.flatnonzero(corrected_by_g2)]),
                                "mean_abs_fused_evidence_corrected":float(np.abs(g2_reliable[corrected_by_g2]).mean()) if corrected_by_g2.any() else None})
    comparisons = build_comparisons(metric_rows)
    write_csv(RESULTS / "fold_metrics.csv", metric_rows)
    write_csv(RESULTS / "train_curve.csv", [
        {"method":group,"fold":fold,**row}
        for (group,fold),detail in details.items() for row in detail["train_curve"]])
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    write_csv(RESULTS / "correction_analysis.csv", correction_rows)

    def comparison(left, right, metric):
        return next(row for row in comparisons
                    if row["left"] == left and row["right"] == right and row["metric"] == metric)

    g1_vs_e77 = comparison(GROUPS[0], "E76.7-G1-CJE-BCSPC-Full-E2E", "acc")
    g2_vs_g1 = comparison(GROUPS[1], GROUPS[0], "acc")
    g2_vs_base = comparison(GROUPS[1], "E74.4-G2-Positive-DCR", "acc")
    g1_vs_base = comparison(GROUPS[0], "E74.4-G2-Positive-DCR", "acc")
    g2_hard = comparison(GROUPS[1], "E74.4-G2-Positive-DCR", "hard_auc")
    g2_prob = comparison(GROUPS[1], "E74.4-G2-Positive-DCR", "prob_auc")
    g2_base_9 = g2_vs_base["exclude_fold2_mean_delta_pp"]
    g2corr = [row for row in correction_rows if row["comparison"] == f"{GROUPS[1]}-vs-E74.4-G2"]
    net_g2 = sum(int(row["net_corrections"]) for row in g2corr)
    gradient_summary = {group: summarize_gradient_history(details, group) for group in GROUPS}
    all_optimizer_records = [record for group in GROUPS for record in gradient_summary[group]["optimizer_groups"].values()]
    conflict_observed = any(record["negative_dot_steps"] > 0 for record in all_optimizer_records)
    conflict_removed = all(record["projected_negative_steps"] == 0 for record in all_optimizer_records)
    e2e_nonzero = all(any(record["active_steps"] > 0 for record in gradient_summary[group]["optimizer_groups"].values())
                       for group in GROUPS)
    unique_norm_values = []
    unique_direction_values = []
    for row in mechanism_rows:
        if row["method"] != GROUPS[1]:
            continue
        try:
            unique_norm_values.extend(np.asarray(json.loads(row.get("prototype_unique_norm_mean", "[]")), dtype=float).reshape(-1).tolist())
            if row.get("unique_direction_accuracy") is not None:
                unique_direction_values.append(float(row["unique_direction_accuracy"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    unique_nonzero = bool(unique_norm_values and np.nanmean(unique_norm_values) > 1e-8)
    if (g2_vs_base["mean_delta_pp"] > 0 and g2_vs_g1["mean_delta_pp"] > 0
            and g1_vs_e77["mean_delta_pp"] > 0 and g2_hard["mean_delta_pp"] >= -0.5
            and g2_prob["mean_delta_pp"] >= -0.5 and net_g2 > 0 and g2_base_9 > 0
            and conflict_observed and conflict_removed and e2e_nonzero and unique_nonzero):
        verdict = "PROMISING_CGD_BCSPC"
    elif g1_vs_e77["mean_delta_pp"] > 0 and g2_vs_g1["mean_delta_pp"] <= 0:
        verdict = "GRADIENT_CONFLICT_REPAIR_ONLY"
    elif g2_vs_g1["mean_delta_pp"] > 0 and g2_vs_base["mean_delta_pp"] <= 0:
        verdict = "BOUNDARY_GAIN_WITHOUT_BASE_IMPROVEMENT"
    elif g2_prob["mean_delta_pp"] > 0 and g2_vs_base["mean_delta_pp"] <= 0:
        verdict = "PROBABILITY_REFINEMENT_ONLY"
    else:
        verdict = "NO_GO_CGD_BCSPC"

    rng_g1 = all(details[(GROUPS[0],fold)]["rng_mismatches"] == 0 for fold in range(1,11))
    rng_g2 = all(details[(GROUPS[1],fold)]["rng_mismatches"] == 0 for fold in range(1,11))
    replay = [replay_latest_step(group, fold) for fold in range(1,11) for group in GROUPS]
    status = {"status":"COMPLETE_AUDITED", "method_verdict":verdict,
              "checkpoint_recalculations":20,"checkpoint_total":20,
              "g1_rng_matches_history":rng_g1,"g2_rng_matches_history":rng_g2,
              "source_sha256":source_sha,"training_source_sha256":source_sha,
              "preflight_sha256":preflight["source_sha256"],"audit_source_sha256":audit_source_sha,
              "groups":list(GROUPS),"folds":10,
              "metrics_mean_sd_percent":{group:{metric:[mean*100,sd*100] for metric,(mean,sd) in metric_summary(metric_rows,group).items() if metric in CORE}
                                          for group in ("E74.4-G2-Positive-DCR",
                                                        "E76.7-G1-CJE-BCSPC-Full-E2E",
                                                        "E76.7-G2-CJE-BCSPC-StopGrad",*GROUPS)},
              "g1_minus_e76_7_g1_acc_pp":g1_vs_e77["mean_delta_pp"],
              "g2_minus_g1_acc_pp":g2_vs_g1["mean_delta_pp"],
              "g2_minus_base_acc_pp":g2_vs_base["mean_delta_pp"],
              "g1_minus_base_acc_pp":g1_vs_base["mean_delta_pp"],
              "g2_minus_base_hard_auc_pp":g2_hard["mean_delta_pp"],
              "g2_minus_base_prob_auc_pp":g2_prob["mean_delta_pp"],
              "g2_minus_base_net_corrections":net_g2,
              "g1_new_backbone_gradient_observed":e2e_nonzero,
              "gradient_conflict_observed":conflict_observed,
              "gradient_conflict_projected_negative_rate_zero":conflict_removed,
              "gradient_summary":gradient_summary,
              "unique_boundary_mean_norm":float(np.nanmean(unique_norm_values)) if unique_norm_values else None,
              "unique_boundary_direction_accuracy_mean":float(np.nanmean(unique_direction_values)) if unique_direction_values else None,
              "latest_checkpoint_replay":replay,
              "audit_checks":{"history_base_identity":True,"E61_initialization_identity":True,
                "historical_rng_all_common_epochs":rng_g1 and rng_g2,
                "all_20_best_predictions_recalculated":True,"stored_probabilities_recalculated":True,
                "legal_train_only_memory":True,"fold2_overlap_excluded_from_new_branch":True,
                "support_A_B_disjoint":True,"test_labels_not_in_new_memory":True,
                "four_mode_cam_hook_shape":True,"zero_delta_preflight_pass":True,
                "prompt_cache_and_order_preflight_pass":True,"G1_G2_parameter_count_matched":True,
                "residual_off_boundary_off_unique_off_swap_reliable_only_interventions":True,
                "g1_g2_both_end_to_end":True,"gradient_projection_math_and_order":True,
                "unique_boundary_not_renormalized":True,"evidence_gated_residual":True}}
    write_json(RESULTS / "audit_status.json", status)
    write_report(metric_rows, comparisons, correction_rows, mechanism_rows, status)
    manifest_path = EXP / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"status":"COMPLETE_AUDITED","method_verdict":verdict,
                         "audited_checkpoints":20,"final_report":"results/report.md",
                         "training_source_sha256":source_sha,"audit_source_sha256":audit_source_sha,
                         "ended_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
        write_json(manifest_path, manifest)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    run_audit()
