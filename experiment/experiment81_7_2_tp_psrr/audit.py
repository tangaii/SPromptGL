#!/usr/bin/env python3
"""Independent Best/Latest audit and result report for Experiment 81.7.2."""
from __future__ import annotations

import csv
import copy
import json
import math
import os
import pickle
import struct
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
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
GROUPS = ("G1", "G2")


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _manual_metrics(logp, labels):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
    values = np.asarray(logp, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    shifted = values - values.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = values.argmax(axis=1).astype(np.int64)
    asd = (y == 0).astype(np.int64)
    return {
        "acc": float(accuracy_score(y, prediction)),
        "hard_auc": float(balanced_accuracy_score(y, prediction)),
        "prob_auc": float(roc_auc_score(asd, probability[:, 0])),
        "sen": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "spe": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "prediction": prediction,
        "probability": probability,
    }


def _saved_metrics(labels, prediction, probability):
    import numpy as np
    prediction = np.asarray(prediction, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if not np.array_equal(prediction, probability.argmax(axis=1)):
        raise RuntimeError("saved prediction and continuous probability disagree")
    computed = _manual_metrics(np.log(np.clip(probability, 1e-30, 1.0)), labels)
    return computed


def _evaluate(e744, state, module, hook, grouping, intervention="full", diagnostics=False):
    import numpy as np
    import torch
    original_grouping = hook.group
    hook.group = grouping
    hook.configure(training=False, intervention=intervention, diagnostics=diagnostics)
    try:
        with torch.no_grad():
            output = e744.forward(state, E74_GROUP, training=False)
    finally:
        hook.group = original_grouping
    if hook.calls != 1 or hook.capture is None:
        raise RuntimeError("original OutputLayer did not execute exactly once")
    test_ids = np.asarray(state["test_np"], dtype=np.int64)
    labels = state["y"].detach().cpu().numpy().astype(np.int64)[test_ids]
    logp = output["logp"].detach().cpu().numpy()[test_ids]
    computed = _manual_metrics(logp, labels)
    capture = None
    if diagnostics:
        capture = {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in hook.capture.items()
        }
    return {
        "metrics": {key: computed[key] for key in METRICS},
        "prediction": computed["prediction"],
        "probability": computed["probability"],
        "logp": logp,
        "hidden": output["hidden"].detach().cpu().numpy()[test_ids],
        "labels": labels,
        "test_ids": test_ids,
        "capture": capture,
    }


def _correction(source, target, labels):
    import numpy as np
    source, target, labels = [np.asarray(item, dtype=np.int64) for item in (source, target, labels)]
    corrected = int(np.sum((source != labels) & (target == labels)))
    harmed = int(np.sum((source == labels) & (target != labels)))
    return {"corrected": corrected, "harmed": harmed, "net": corrected - harmed}


def _mean_sd(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _summary(rows):
    result = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        if len(selected) != 10:
            continue
        result[group] = {}
        for metric in METRICS:
            mean, sd = _mean_sd([float(row[metric]) for row in selected])
            nine = [float(row[metric]) for row in selected if int(row["fold"]) != 2]
            nine_mean, nine_sd = _mean_sd(nine)
            result[group][metric] = {
                "mean": mean, "population_sd": sd,
                "exclude_fold2_mean": nine_mean, "exclude_fold2_population_sd": nine_sd,
            }
    return result


def _paired(name, left, right):
    rows = []
    for subset, folds in (("all10", list(range(1, 11))), ("exclude_fold2", [i for i in range(1, 11) if i != 2])):
        for metric in METRICS:
            differences = [(left[fold][metric] - right[fold][metric]) * 100.0 for fold in folds]
            mean, sd = _mean_sd(differences)
            half = 1.96 * sd / math.sqrt(len(differences))
            wins = sum(value > 1e-12 for value in differences)
            ties = sum(abs(value) <= 1e-12 for value in differences)
            rows.append({
                "comparison": name, "subset": subset, "metric": METRIC_LABELS[metric],
                "folds": len(folds), "mean_difference_pp": mean,
                "descriptive_low_pp": mean - half, "descriptive_high_pp": mean + half,
                "W_T_L": f"{wins}/{ties}/{len(folds)-wins-ties}",
                "fold_differences_pp": json.dumps({str(fold): left[fold][metric] * 100.0 - right[fold][metric] * 100.0 for fold in folds}, sort_keys=True),
                "interval_note": "descriptive population-SD interval, not independent-patient significance",
            })
    return rows


def _darg_result(fold, labels):
    import numpy as np
    path = HIST_E744 / ".runtime/checkpoints" / f"{E74_GROUP}_f{fold:02d}.best.pt"
    expected_sha = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))["source_files"][str(path.resolve())]
    record = json.loads((EXP / ".runtime/darg_best_compat.json").read_text(encoding="utf-8"))[str(fold)]
    if run.sha_file(path) != expected_sha or record.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"DARG checkpoint SHA mismatch at fold {fold}")
    if record.get("fold") != fold or record.get("group") != E74_GROUP:
        raise RuntimeError(f"DARG identity mismatch at fold {fold}")
    prediction = np.asarray(record["test_prediction"], dtype=np.int64)
    probability = np.asarray(record["test_probability"], dtype=np.float64)
    if prediction.shape != np.asarray(labels).shape or probability.shape != (len(labels), 2):
        raise RuntimeError(f"DARG prediction shape mismatch at fold {fold}")
    result = _saved_metrics(labels, prediction, probability)
    for metric in METRICS:
        if abs(result[metric] - float(record["metrics"][metric])) > 2e-8:
            raise RuntimeError(f"DARG metric readback mismatch at fold {fold} metric {metric}")
    return result


def _historical_e817_metrics():
    path = Path("/model/spromptgl/experiment/experiment81_7_psrr/results/fold_metrics.csv")
    records = [row for row in _read_csv(path) if row.get("group") == "G1"]
    if len(records) != 10:
        raise RuntimeError("E81.7 historical G1 metrics do not contain ten folds")
    return {
        int(row["fold"]): {metric: float(row[metric]) / 100.0 for metric in METRICS}
        for row in records
    }


def _historical_e8171_metrics():
    path = Path("/model/spromptgl/experiment/experiment81_7_1_psrr_lcr/results/fold_metrics.csv")
    records = [row for row in _read_csv(path) if row.get("group") == "G1"]
    if len(records) != 10:
        raise RuntimeError("E81.7.1 historical G1 metrics do not contain ten folds")
    return {
        int(row["fold"]): {metric: float(row[metric]) / 100.0 for metric in METRICS}
        for row in records
    }


def _historical_rng_sha(state):
    digest = __import__("hashlib").sha256()
    pickled = bytearray(pickle.dumps((state["python"], state["numpy"]), protocol=4))
    numpy1_path = b"numpy.core.multiarray"
    numpy2_path = b"numpy._core.multiarray"
    if numpy1_path in pickled:
        if pickled.count(numpy1_path) != 1 or pickled[2] != 0x95:
            raise RuntimeError("unexpected NumPy 1 RNG pickle layout")
        position = pickled.index(numpy1_path)
        pickled[position - 1] += 1
        pickled[position:position + len(numpy1_path)] = numpy2_path
        frame_size = struct.unpack("<Q", pickled[3:11])[0]
        pickled[3:11] = struct.pack("<Q", frame_size + 1)
    digest.update(pickled)
    digest.update(state["cpu"].cpu().numpy().tobytes())
    for value in state.get("cuda", []):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def _audit_rng_trajectory(e744, group, fold, grouping, task, old_task):
    import torch
    state, warmup, identity = e744.prepare(fold)
    if identity != task["base_identity"]:
        raise RuntimeError(f"RNG replay base identity changed at {group} fold {fold}")
    runtime = run.rng_runtime(e744)
    runtime.restore_rng(warmup["rng"])
    selection_mode = run.CFG["groups"][group]["selection_mode"]
    module = run.make_module(fold, state["cam"].parameters().__next__().device,
                             run.prompt_payload()["semantic_basis_float64"], selection_mode)
    hook = run.OutputPreHook(state, module, grouping)
    expected_training = {11 + index: item for index, item in enumerate(task["rng_trace"])}
    expected_historical = {int(item["epoch"]): item for item in old_task["rng_trace"]}
    shared_epochs = sorted(set(expected_training) & set(expected_historical))
    if not expected_training or not shared_epochs:
        raise RuntimeError(f"RNG trace has no comparable history at {group} fold {fold}")
    blocked_rows = run.CFG["data"]["historical_fold2_psrr_blocked_rows_zero_based"] if fold == 2 else ()
    for epoch in sorted(expected_training):
        before_state = runtime.pack_rng()
        actual_before = runtime.rng_sha(before_state)
        historical_before = _historical_rng_sha(before_state)
        buffer_state = run._buffer_snapshot(state)
        hook.configure(training=True, intervention="relation_off", blocked_rows=blocked_rows)
        with torch.no_grad():
            reference = e744.forward(state, E74_GROUP, training=True)
        if reference["logp"].shape[0] != state["y"].shape[0] or hook.calls != 1:
            raise RuntimeError(f"reference RNG replay forward shape/call mismatch at {group} fold {fold} epoch {epoch}")
        run._restore_buffers(state, buffer_state)
        runtime.restore_rng(before_state)
        hook.configure(training=True, intervention="full", blocked_rows=blocked_rows)
        with torch.no_grad():
            train_output = e744.forward(state, E74_GROUP, training=True)
        if train_output["logp"].shape[0] != state["y"].shape[0] or hook.calls != 1:
            raise RuntimeError(f"enhanced RNG replay forward shape/call mismatch at {group} fold {fold} epoch {epoch}")
        after_train_state = runtime.pack_rng()
        actual_after_train = runtime.rng_sha(after_train_state)
        historical_after_train = _historical_rng_sha(after_train_state)
        with torch.no_grad():
            hook.configure(training=False)
            evaluated = e744.forward(state, E74_GROUP, training=False)
            runtime.classification_metrics(evaluated["logp"], state["test"], state["y"])
        if hook.calls != 1:
            raise RuntimeError(f"evaluation RNG replay classifier call mismatch at {group} fold {fold} epoch {epoch}")
        after_eval_state = runtime.pack_rng()
        actual_after_eval = runtime.rng_sha(after_eval_state)
        historical_after_eval = _historical_rng_sha(after_eval_state)
        current = expected_training[epoch]
        for field, actual, expected in (("before", actual_before, current["before"]),
                                        ("after_train", actual_after_train, current["after_train"]),
                                        ("after_eval", actual_after_eval, current["after_eval"])):
            if actual != expected:
                raise RuntimeError(f"replayed current-runtime RNG differs at {group} fold {fold} epoch {epoch} {field}")
        if epoch in expected_historical:
            historical = expected_historical[epoch]
            for field, actual, expected in (("before", historical_before, historical["rng_before"]),
                                            ("after_train", historical_after_train, historical["rng_after_train"]),
                                            ("after_eval", historical_after_eval, historical["rng_after_eval"])):
                if actual != expected:
                    raise RuntimeError(f"replayed historical RNG differs at {group} fold {fold} epoch {epoch} {field}")
    hook.close()
    return {"status": "PASS", "epochs_replayed": len(expected_training),
            "shared_historical_epochs": len(shared_epochs),
            "extra_epochs_without_darg_history": len(set(expected_training) - set(expected_historical)),
            "current_runtime_trace_exact": True, "historical_numpy2_trace_exact": True,
            "reference_forward_rng_restored": True, "same_forward_and_evaluation_call_order": True}


def _class_transitions(source, target, labels):
    import numpy as np
    source, target, labels = [np.asarray(item, dtype=np.int64) for item in (source, target, labels)]
    return {
        "asd_to_td": int(np.sum((labels == 0) & (source == 0) & (target == 1))),
        "td_to_asd": int(np.sum((labels == 1) & (source == 1) & (target == 0))),
        "asd_corrected": int(np.sum((labels == 0) & (source == 1) & (target == 0))),
        "asd_harmed": int(np.sum((labels == 0) & (source == 0) & (target == 1))),
        "td_corrected": int(np.sum((labels == 1) & (source == 0) & (target == 1))),
        "td_harmed": int(np.sum((labels == 1) & (source == 1) & (target == 0))),
    }


def _prediction_records(group, fold, outputs, darg, base_output):
    import numpy as np
    labels = outputs["PSRR_full"]["labels"]
    sample_ids = outputs["PSRR_full"]["test_ids"]
    records = []
    for stage, result in outputs.items():
        for index, (sample_id, label) in enumerate(zip(sample_ids, labels)):
            prediction = int(result["prediction"][index])
            darg_prediction = int(darg["prediction"][index])
            base_prediction = int(base_output["prediction"][index])
            records.append({
                "group": group, "fold": fold, "sample_id_zero_based": int(sample_id),
                "true_label": int(label), "stage": stage, "prediction": prediction,
                "probability_asd": float(result["probability"][index, 0]),
                "correct": int(prediction == int(label)),
                "corrected_vs_darg": int(darg_prediction != int(label) and prediction == int(label)),
                "harmed_vs_darg": int(darg_prediction == int(label) and prediction != int(label)),
                "corrected_vs_psrr_off": int(base_prediction != int(label) and prediction == int(label)),
                "harmed_vs_psrr_off": int(base_prediction == int(label) and prediction != int(label)),
            })
    return records


def _gradient_mechanism(task, group, fold):
    import numpy as np
    records = [item for item in task.get("gradient_trace", []) if item.get("gradient_control")]
    if not records:
        raise RuntimeError(f"gradient-control trace is empty at {group} fold {fold}")
    result = {"group": group, "fold": fold, "record_type": "gradient_control"}
    for name in ("gfr", "cam", "darg"):
        values = [item["gradient_control"]["groups"][name] for item in records]
        result.update({
            f"{name}_negative_cosine_fraction": float(np.mean([row["negative_cosine"] for row in values])),
            f"{name}_projection_fraction": float(np.mean([row["projection_applied"] for row in values])),
            f"{name}_mean_cosine": float(np.mean([row["cosine_reference_delta"] for row in values])),
            f"{name}_mean_reference_norm": float(np.mean([row["reference_norm"] for row in values])),
            f"{name}_mean_delta_norm": float(np.mean([row["delta_norm"] for row in values])),
            f"{name}_mean_projection_ratio": float(np.mean([
                row["projected_delta_norm"] / max(row["delta_norm"], 1e-30) for row in values
            ])),
            f"{name}_mean_group_scale": float(np.mean([row["group_scale"] for row in values])),
            f"{name}_max_controlled_ratio": float(max(row["controlled_delta_to_reference_ratio"] for row in values)),
        })
    result["vjp_records"] = len(records)
    result["mean_reference_gradient_norm"] = float(np.mean([
        item["gradient_control"]["reference_gradient_norm"] for item in records
    ]))
    result["mean_enhanced_gradient_norm"] = float(np.mean([
        item["gradient_control"]["enhanced_gradient_norm"] for item in records
    ]))
    result["mean_delta_gradient_norm"] = float(np.mean([
        item["gradient_control"]["delta_gradient_norm"] for item in records
    ]))
    probes = [item.get("branch_vjp") for item in task.get("gradient_trace", []) if item.get("branch_vjp")]
    if len(probes) != 1 or int(next(item["epoch"] for item in task["gradient_trace"] if item.get("branch_vjp"))) != 12:
        raise RuntimeError(f"classification-to-CAM/DARG path probe missing at epoch 12 for {group} fold {fold}")
    probe = probes[0]
    for key in ("classification_to_delta_norm", "delta_to_cam_input_norm", "branch_to_cam_parameter_norm", "branch_to_darg_parameter_norm"):
        if not math.isfinite(float(probe[key])) or float(probe[key]) <= 0.0:
            raise RuntimeError(f"classification-to-PSRR-to-{key} path is not nonzero at {group} fold {fold}")
    result.update({f"epoch12_{key}": float(value) for key, value in probe.items()})
    kappa = float(run.CFG["optimization"]["shared_gradient_control"]["post_projection_group_norm_ratio_kappa"])
    if result["gfr_max_controlled_ratio"] > kappa + 1e-7 or result["cam_max_controlled_ratio"] > kappa + 1e-7 or result["darg_max_controlled_ratio"] > kappa + 1e-7:
        raise RuntimeError(f"post-projection shared gradient exceeded group kappa at {group} fold {fold}")
    return result


def _comparison_mean(comparisons, name, subset, metric):
    label = METRIC_LABELS[metric]
    for row in comparisons:
        if row["comparison"] == name and row["subset"] == subset and row["metric"] == label:
            return float(row["mean_difference_pp"])
    return None


def _research_decision(comparisons, mechanism_rows, intervention_records):
    import numpy as np
    performance_by_group = {
        candidate: all(
            (_comparison_mean(comparisons, f"{candidate}_vs_DARG", subset, metric) or 0.0) > 0.0
            for subset in ("all10", "exclude_fold2") for metric in ("acc", "hard_auc")
        ) for candidate in ("G1", "G2")
    }
    performance_gate = any(performance_by_group.values())
    relation_design_gate = all(
        (_comparison_mean(comparisons, "G1_vs_G2", subset, metric) or 0.0) > 0.0
        for subset in ("all10", "exclude_fold2") for metric in ("acc", "hard_auc")
    )
    full_predictions = {
        (int(row["fold"]), int(row["sample_id_zero_based"])): int(row["prediction"])
        for row in intervention_records if row["stage"] == "PSRR_full" and row["group"] == "G1"
    }
    scrambled_predictions = {
        (int(row["fold"]), int(row["sample_id_zero_based"])): int(row["prediction"])
        for row in intervention_records if row["stage"] == "PSRR_full" and row["group"] == "G2"
    }
    if full_predictions and set(full_predictions) == set(scrambled_predictions):
        group_prediction_changes = sum(full_predictions[key] != scrambled_predictions[key] for key in full_predictions)
    else:
        group_prediction_changes = 0
    mechanism_by_group = {}
    required = ("within_patient_local_td_sd_mean", "within_patient_local_asd_sd_mean",
                "td_beta_patient_sd_mean", "asd_beta_patient_sd_mean", "relation_difference_norm_mean",
                "asd_td_support_disagreement_rate", "message_contrast_norm_mean", "prompt_off_probability_mae",
                "uniform_beta_probability_mae")
    for group in ("G1", "G2"):
        rows = [row for row in mechanism_rows if row.get("group") == group and row.get("record_type") == "best_mechanism"]
        if len(rows) != 10:
            mechanism_by_group[group] = {"passed": False, "folds": len(rows)}
            continue
        means = {key: float(np.mean([float(row[key]) for row in rows])) for key in required}
        mechanism_by_group[group] = {"passed": all(value > 1e-10 for value in means.values()), "folds": len(rows), **means}
    mechanism_gate = all(mechanism_by_group[group]["passed"] for group in ("G1", "G2"))
    semantic_gate = relation_design_gate and group_prediction_changes > 0 and mechanism_gate
    if not performance_gate:
        decision = "NO_GO_PERFORMANCE"
    elif not semantic_gate:
        decision = "NO_GO_MEDICAL_SEMANTIC_OR_RELATION_MECHANISM"
    else:
        decision = "CONTINUE_CANDIDATE_SINGLE_SEED_NOT_FINAL_EVIDENCE"
    return {
        "decision": decision,
        "performance_gate_by_group": performance_by_group,
        "performance_gate_any_group_over_darg": performance_gate,
        "relation_design_gate_g1_over_g2": relation_design_gate,
        "medical_semantic_gate_g1_over_g2": semantic_gate,
        "g1_g2_full_prediction_changes": int(group_prediction_changes),
        "relation_mechanism_gate_by_group": mechanism_by_group,
        "single_seed_limitation": "one seed is preliminary candidate evidence, not final MICCAI evidence",
    }


def _mechanism(state, full, base, group, fold, interventions):
    import numpy as np
    import torch
    capture = full["capture"]
    test_ids = full["test_ids"]
    td_beta = capture["beta_td"][test_ids].numpy()
    asd_beta = capture["beta_asd"][test_ids].numpy()
    selection_mode = capture.get("selection_mode", run.CFG["groups"][group]["selection_mode"])
    attention_td = capture["attention_td"][test_ids]
    attention_asd = capture["attention_asd"][test_ids]
    td_support = attention_td[..., :15] > 0
    asd_support = attention_asd[..., :15] > 0
    both_support = torch.cat((td_support, asd_support), dim=1)
    consensus = both_support.float().mean(0) >= 0.5
    jaccard = ((both_support & consensus.unsqueeze(0)).sum(-1).float() /
               (both_support | consensus.unsqueeze(0)).sum(-1).float().clamp_min(1)).numpy()
    z = capture["z_cam"][test_ids].float()
    delta = capture["delta"][test_ids].float()
    z_norm = z.norm(dim=-1).clamp_min(1e-12)
    delta_norm = delta.norm(dim=-1)
    if selection_mode == "node":
        patient_td_sd = td_beta.std(axis=0).mean()
        patient_asd_sd = asd_beta.std(axis=0).mean()
        local_td_sd = td_beta.std(axis=1).mean()
        local_asd_sd = asd_beta.std(axis=1).mean()
    else:
        patient_td_sd = td_beta.std(axis=0).mean()
        patient_asd_sd = asd_beta.std(axis=0).mean()
        local_td_sd = td_beta.std(axis=(1, 2)).mean()
        local_asd_sd = asd_beta.std(axis=(1, 2)).mean()
    return {
        "group": group, "fold": fold, "record_type": "best_mechanism", "selection_mode": selection_mode,
        "td_beta_entropy_mean": float((-(td_beta * np.log(np.clip(td_beta, 1e-12, 1))).sum(-1)).mean()),
        "asd_beta_entropy_mean": float((-(asd_beta * np.log(np.clip(asd_beta, 1e-12, 1))).sum(-1)).mean()),
        "td_beta_patient_sd_mean": float(patient_td_sd),
        "asd_beta_patient_sd_mean": float(patient_asd_sd),
        "within_patient_local_td_sd_mean": float(local_td_sd),
        "within_patient_local_asd_sd_mean": float(local_asd_sd),
        "prompt_evidence_sd": float(capture["prompt_evidence"][test_ids].std()),
        "prompt_score_sd": float(capture["prompt_scores"][test_ids].std()),
        "raw_basis_prompt_spread_mean": float(capture["prompt_scores_raw_basis"][test_ids].std(dim=-1).mean()),
        "centered_basis_prompt_spread_mean": float(capture["prompt_scores"][test_ids].std(dim=-1).mean()),
        "centered_to_raw_prompt_spread_ratio": float(
            capture["prompt_scores"][test_ids].std(dim=-1).mean() /
            capture["prompt_scores_raw_basis"][test_ids].std(dim=-1).mean().clamp_min(1e-12)
        ),
        "mean_real_nonzero_edges": float(torch.cat((td_support.sum(-1), asd_support.sum(-1)), dim=1).float().mean()),
        "real_edge_nonzero_rate": float(both_support.float().mean()),
        "null_positive_rate": float(torch.cat((attention_td[..., -1], attention_asd[..., -1]), dim=1).gt(0).float().mean()),
        "asd_td_support_disagreement_rate": float((td_support ^ asd_support).float().mean()),
        "patient_consensus_jaccard_mean": float(jaccard.mean()),
        "patient_consensus_jaccard_sd": float(jaccard.std(ddof=0)),
        "relation_support_l1_mean": float((attention_td - attention_asd).abs().sum(-1).mean()),
        "relation_difference_norm_mean": float(capture["disease_relation_difference"][test_ids].norm(dim=(-1, -2)).mean()),
        "message_contrast_norm_mean": float(capture["message_contrast"][test_ids].norm(dim=-1).mean()),
        "residual_ratio_mean": float(capture["residual_ratio"][test_ids].mean()),
        "residual_ratio_max": float(capture["residual_ratio"][test_ids].max()),
        "residual_norm_mean": float(delta_norm.mean()),
        "residual_norm_max": float(delta_norm.max()),
        "residual_relative_norm_mean": float((delta_norm / z_norm).mean()),
        "prompt_off_probability_mae": float(np.abs(full["probability"][:, 0] - interventions["prompt_off"]["probability"][:, 0]).mean()),
        "relation_off_probability_mae": float(np.abs(full["probability"][:, 0] - interventions["relation_off"]["probability"][:, 0]).mean()),
        "uniform_beta_probability_mae": float(np.abs(full["probability"][:, 0] - interventions["uniform_beta"]["probability"][:, 0]).mean()),
        "global_interaction_off_probability_mae": float(np.abs(full["probability"][:, 0] - interventions["global_interaction_off"]["probability"][:, 0]).mean()),
        "prompt_off_hard_flips": int(np.sum(full["prediction"] != interventions["prompt_off"]["prediction"])),
        "relation_off_hard_flips": int(np.sum(full["prediction"] != interventions["relation_off"]["prediction"])),
        "uniform_beta_hard_flips": int(np.sum(full["prediction"] != interventions["uniform_beta"]["prediction"])),
        "global_interaction_off_hard_flips": int(np.sum(full["prediction"] != interventions["global_interaction_off"]["prediction"])),
    }


def _latest_replay(e744, payload, group, fold, grouping):
    import numpy as np
    base = e744.load_history()
    runtime = run.rng_runtime(e744)

    def restore():
        state, _, identity = e744.prepare(fold)
        if identity != payload["base_identity"]:
            raise RuntimeError(f"Latest base identity changed at {group} fold {fold}")
        selection_mode = run.CFG["groups"][group]["selection_mode"]
        module = run.make_module(fold, state["cam"].parameters().__next__().device, run.prompt_payload()["semantic_basis_float64"], selection_mode)
        if run.module_state_digest(module) != payload["initial_psrr_sha256"]:
            raise RuntimeError(f"Latest initialization identity changed at {group} fold {fold}")
        optimizer = __import__("torch").optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        base.load_bundle(state, payload["current"])
        module.load_state_dict(payload["psrr_state"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_psrr"]))
        hook = run.OutputPreHook(state, module, grouping)
        runtime.restore_rng(payload["rng"])
        return state, module, optimizer, hook

    state_a, module_a, optimizer_a, hook_a = restore()
    next_a = run.training_step(e744, state_a, module_a, optimizer_a, hook_a, group, epoch=int(payload["next_epoch"]) + 1, fold=fold)
    after_a = {"base": base.state_bundle(state_a), "module": {k: v.detach().cpu().clone() for k, v in module_a.state_dict().items()}, "optimizer": optimizer_a.state_dict(), "rng": runtime.pack_rng()}
    hook_a.close()
    state_b, module_b, optimizer_b, hook_b = restore()
    next_b = run.training_step(e744, state_b, module_b, optimizer_b, hook_b, group, epoch=int(payload["next_epoch"]) + 1, fold=fold)
    after_b = {"base": base.state_bundle(state_b), "module": {k: v.detach().cpu().clone() for k, v in module_b.state_dict().items()}, "optimizer": optimizer_b.state_dict(), "rng": runtime.pack_rng()}
    hook_b.close()
    for name in after_a:
        equal, where = run.tensor_tree_equal(after_a[name], after_b[name])
        if not equal:
            difference, location = run.tensor_tree_max_difference(after_a[name], after_b[name])
            raise RuntimeError(f"Latest replay differs at {group} fold {fold} {name} {where} max_abs={difference} {location}")
    if next_a["total_loss"] != next_b["total_loss"] or next_a["metrics"] != next_b["metrics"] or not np.array_equal(next_a["prediction"], next_b["prediction"]):
        raise RuntimeError(f"Latest replay output differs at {group} fold {fold}")
    return {"status": "PASS", "next_step_epoch": int(payload["next_epoch"]) + 1, "parameters_optimizers_rng_equal": True}


def audit_group(group):
    import numpy as np
    import torch
    e744 = run.base_runner()
    grouping = "medical"
    selection_mode = run.CFG["groups"][group]["selection_mode"]
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    status_path = RESULTS / "audit_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if group == "G2" and (status.get("g1_status") != "COMPLETE_AUDITED" or status.get("g1_completed_folds") != 10):
        raise RuntimeError("G2 audit requires all ten G1 folds to pass independent audit first")
    source_sha = preflight["source_sha256"]
    frozen_inventory = preflight["source_files"]
    if run.sha_object(frozen_inventory) != source_sha:
        raise RuntimeError("preflight source inventory SHA changed")
    audit_path = str((EXP / "audit.py").resolve())
    frozen_audit_sha = frozen_inventory.get(audit_path)
    if frozen_audit_sha is None:
        raise RuntimeError("frozen audit source SHA is missing")
    for path, expected in frozen_inventory.items():
        if path != audit_path and (not Path(path).is_file() or run.sha_file(path) != expected):
            raise RuntimeError(f"frozen training source changed: {path}")
    current_audit_sha = run.sha_file(audit_path)
    existing_rows = _read_csv(RESULTS / "fold_metrics.csv") if (RESULTS / "fold_metrics.csv").is_file() else []
    existing_mechanism = _read_csv(RESULTS / "mechanism.csv") if (RESULTS / "mechanism.csv").is_file() else []
    existing_interventions = _read_csv(RESULTS / "interventions.csv") if (RESULTS / "interventions.csv").is_file() else []
    rows = [row for row in existing_rows if row.get("group") not in (group, "DARG", "E81.7-G1", "E81.7.1-G1")]
    mechanism_rows = [row for row in existing_mechanism if row.get("group") != group]
    intervention_records = [row for row in existing_interventions if row.get("group") != group]
    corrections_total = {name: {"corrected": 0, "harmed": 0, "net": 0} for name in ("A_to_B", "B_to_C", "A_to_C")}
    class_transitions_total = {name: {"asd_corrected": 0, "asd_harmed": 0, "td_corrected": 0, "td_harmed": 0}
                              for name in ("A_to_B", "B_to_C", "A_to_C")}
    latest_audits = []
    rng_audits = []
    historical_e817 = _historical_e817_metrics()
    historical_e8171 = _historical_e8171_metrics()

    for fold in range(1, 11):
        task_path, best_path, latest_path = run._task_path(group, fold), run._best_path(group, fold), run._latest_path(group, fold)
        if not all(path.is_file() for path in (task_path, best_path, latest_path)):
            raise FileNotFoundError(f"missing {group} fold {fold} task/Best/Latest")
        task = json.loads(task_path.read_text(encoding="utf-8"))
        best = run.torch_load(best_path)
        latest = run.torch_load(latest_path)
        if task.get("status") != "COMPLETE" or task.get("source_sha256") != source_sha:
            raise RuntimeError(f"formal task identity failed at {group} fold {fold}")
        if task.get("best_sha256") != run.sha_file(best_path) or task.get("latest_sha256") != run.sha_file(latest_path):
            raise RuntimeError(f"task checkpoint SHA mismatch at {group} fold {fold}")
        if int(task.get("best_epoch_1based", -1)) != int(best.get("epoch_1based", -2)):
            raise RuntimeError(f"Best epoch identity mismatch at {group} fold {fold}")
        if int(latest.get("next_epoch", -1)) != int(task.get("epochs_executed", -2)):
            raise RuntimeError(f"Latest epoch identity mismatch at {group} fold {fold}")
        expected_blocked = [376, 406] if fold == 2 else []
        if task.get("fold2_psrr_blocked_rows") != expected_blocked or task.get("fold2_psrr_training_assertion") != "PASS":
            raise RuntimeError(f"Fold2 PSRR-only overlap constraint failed at {group} fold {fold}")
        expected_identity = {"schema": "E81.7.2-TP-PSRR-v1", "group": group, "fold": fold, "grouping": grouping,
                             "selection_mode": selection_mode,
                             "source_sha256": source_sha, "config_sha256": run.sha_file(EXP / "config.json")}
        for payload in (best, latest):
            for key, expected in expected_identity.items():
                if payload.get(key) != expected:
                    raise RuntimeError(f"{group} fold {fold} {key} identity mismatch")
            if payload.get("warmup_sha256") != task.get("warmup_sha256") or not payload.get("optimizer_psrr"):
                raise RuntimeError(f"{group} fold {fold} checkpoint recovery identity incomplete")
        state, warmup, identity = e744.prepare(fold)
        if identity != best["base_identity"] or identity != latest["base_identity"]:
            raise RuntimeError(f"historical identity changed at {group} fold {fold}")
        if run.module_state_digest(run.make_module(fold, state["cam"].parameters().__next__().device, run.prompt_payload()["semantic_basis_float64"], selection_mode)) != best["initial_psrr_sha256"]:
            raise RuntimeError(f"initial PSRR identity changed at {group} fold {fold}")
        if group == "G2":
            g1_task = json.loads(run._task_path("G1", fold).read_text(encoding="utf-8"))
            if (g1_task.get("status") != "COMPLETE" or
                    g1_task.get("initial_psrr_sha256") != task.get("initial_psrr_sha256") or
                    g1_task.get("warmup_sha256") != task.get("warmup_sha256") or
                    g1_task.get("base_identity") != task.get("base_identity")):
                raise RuntimeError(f"G1/G2 fold {fold} initialization or inherited DARG identity differs")
        base = e744.load_history()
        base.load_bundle(state, best["base_state"])
        module = run.make_module(fold, state["cam"].parameters().__next__().device, run.prompt_payload()["semantic_basis_float64"], selection_mode)
        module.load_state_dict(best["psrr_state"], strict=True)
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.0005, weight_decay=0.01)
        optimizer.load_state_dict(copy.deepcopy(best["optimizer_psrr"]))
        runtime = run.rng_runtime(e744)
        runtime.restore_rng(best["rng"])
        hook = run.OutputPreHook(state, module, grouping)
        full = _evaluate(e744, state, module, hook, grouping, diagnostics=True)
        if not np.array_equal(full["prediction"], np.asarray(best["test_prediction"])) or not np.allclose(full["probability"], np.asarray(best["test_probability"]), atol=2e-7, rtol=0):
            raise RuntimeError(f"independent Best prediction readback mismatch at {group} fold {fold}")
        for metric in METRICS:
            if abs(full["metrics"][metric] - float(best["metrics"][metric])) > 2e-8:
                raise RuntimeError(f"independent Best metric readback mismatch at {group} fold {fold} {metric}")
        interventions = {}
        for intervention in ("prompt_off", "relation_off", "uniform_beta", "global_interaction_off"):
            interventions[intervention] = _evaluate(e744, state, module, hook, grouping, intervention, diagnostics=True)
        for name in ("prompt_off", "relation_off"):
            if torch.count_nonzero(interventions[name]["capture"]["delta"]):
                raise RuntimeError(f"{name} did not yield an exact zero residual at {group} fold {fold}")
            if not np.array_equal(interventions[name]["logp"], interventions["relation_off"]["logp"]):
                raise RuntimeError(f"{name} did not reproduce relation-off output at {group} fold {fold}")

        darg = _darg_result(fold, full["labels"])
        base_output = interventions["relation_off"]
        darg["labels"] = full["labels"]
        darg["test_ids"] = full["test_ids"]
        output_map = {"DARG_best": darg, "PSRR_off": base_output, "PSRR_full": full,
                      "Prompt_off": interventions["prompt_off"], "Relation_off": interventions["relation_off"],
                      "Uniform_beta": interventions["uniform_beta"],
                      "Global_interaction_off": interventions["global_interaction_off"]}
        corrections = {
            "A_to_B": _correction(darg["prediction"], base_output["prediction"], full["labels"]),
            "B_to_C": _correction(base_output["prediction"], full["prediction"], full["labels"]),
            "A_to_C": _correction(darg["prediction"], full["prediction"], full["labels"]),
        }
        for name, value in corrections.items():
            corrections_total[name]["corrected"] += value["corrected"]
            corrections_total[name]["harmed"] += value["harmed"]
            corrections_total[name]["net"] += value["net"]
        transitions = {
            "A_to_B": _class_transitions(darg["prediction"], base_output["prediction"], full["labels"]),
            "B_to_C": _class_transitions(base_output["prediction"], full["prediction"], full["labels"]),
            "A_to_C": _class_transitions(darg["prediction"], full["prediction"], full["labels"]),
        }
        for name, value in transitions.items():
            for key in class_transitions_total[name]:
                class_transitions_total[name][key] += value[key]
        rows.append({"group": group, "fold": fold, **{key: full["metrics"][key] * 100.0 for key in METRICS},
                     "best_epoch_1based": int(best["epoch_1based"]), "epochs_executed": int(task["epochs_executed"]),
                     "checkpoint_sha256": run.sha_file(best_path), "latest_sha256": run.sha_file(latest_path),
                     "independent_checkpoint_audit": "PASS"})
        rows.append({"group": "DARG", "fold": fold, **{key: darg[key] * 100.0 for key in METRICS}})
        rows.append({"group": "E81.7-G1", "fold": fold,
                     **{key: historical_e817[fold][key] * 100.0 for key in METRICS}})
        rows.append({"group": "E81.7.1-G1", "fold": fold,
                     **{key: historical_e8171[fold][key] * 100.0 for key in METRICS}})
        mechanism_rows.append({"group": group, "fold": fold, "record_type": "corrections", **{
            f"{stage}_{key}": value for stage, correction in corrections.items() for key, value in correction.items()
        }})
        mechanism_rows.append({"group": group, "fold": fold, "record_type": "class_transitions", **{
            f"{stage}_{key}": value for stage, transition in transitions.items() for key, value in transition.items()
        }})
        diag = _mechanism(state, full, base_output, group, fold, interventions)

        old_task = json.loads((HIST_E744 / ".runtime/tasks" / f"{E74_GROUP}_f{fold:02d}.json").read_text(encoding="utf-8"))
        rng_audits.append({"fold": fold, **_audit_rng_trajectory(e744, group, fold, grouping, task, old_task)})
        control_trace = [item.get("gradient_control") for item in task.get("gradient_trace", []) if item.get("gradient_control")]
        if not control_trace or any(
            float(item.get("max_reconstruction_error", 1.0)) > max(
                float(item.get("reconstruction_tolerance", 0.0)),
                2e-6 * max(1.0, float(item.get("gradient_control", {}).get("enhanced_gradient_norm", 1.0))),
            ) for item in control_trace
        ):
            raise RuntimeError(f"gradient-control reconstruction audit failed at {group} fold {fold}")
        diag.update({"shared_darg_rng_epochs": len(task["rng_trace"]), "initial_psrr_sha256": best["initial_psrr_sha256"],
                     "checkpoint_source_sha256": source_sha, "gradient_control_records": len(control_trace)})
        mechanism_rows.append(diag)
        mechanism_rows.append(_gradient_mechanism(task, group, fold))
        intervention_records.extend(_prediction_records(group, fold, output_map, darg, base_output))
        latest_audits.append({"fold": fold, **_latest_replay(e744, latest, group, fold, grouping)})
        hook.close()

    run.write_csv(RESULTS / "fold_metrics.csv", rows)
    run.write_csv(RESULTS / "mechanism.csv", mechanism_rows)
    run.write_csv(RESULTS / "interventions.csv", intervention_records)
    summaries = _summary(rows)
    normalized = {group_name: {fold: {metric: float(next(row[metric] for row in rows if row["group"] == group_name and int(row["fold"]) == fold)) / 100.0 for metric in METRICS} for fold in range(1, 11)} for group_name in ("DARG", "G1", "G2") if sum(row["group"] == group_name for row in rows) == 10}
    comparisons = []
    for candidate in ("G1", "G2"):
        if candidate in normalized:
            comparisons.extend(_paired(f"{candidate}_vs_DARG", normalized[candidate], normalized["DARG"]))
    if "G1" in normalized and "G2" in normalized:
        comparisons.extend(_paired("G1_vs_G2", normalized["G1"], normalized["G2"]))
    historical = _historical_e817_metrics()
    if "G1" in normalized:
        comparisons.extend(_paired("G1_vs_E81.7", normalized["G1"], historical))
        comparisons.extend(_paired("G1_vs_E81.7.1", normalized["G1"], historical_e8171))
    if "G2" in normalized:
        comparisons.extend(_paired("G2_vs_E81.7.1", normalized["G2"], historical_e8171))
    run.write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    status_path = RESULTS / "audit_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    corrections_by_group = status.get("correction_totals_by_group", {})
    corrections_by_group[group] = {"overall": corrections_total, "class_transitions": class_transitions_total}
    latest_by_group = status.get("latest_resume_audits_by_group", {})
    latest_by_group[group] = latest_audits
    status.update({f"{group.lower()}_completed_folds": 10, f"{group.lower()}_best_audit": "PASS", f"{group.lower()}_latest_resume_audit": "PASS",
                   f"{group.lower()}_public_rng_vs_darg": "PASS", f"{group.lower()}_end_to_end_gradient_audit": "PASS",
                   "source_sha256": source_sha, "audit_source_sha256_at_training_freeze": frozen_audit_sha,
                   "audit_source_sha256_current": current_audit_sha, "audit_only_patch_after_training": current_audit_sha != frozen_audit_sha,
                   "fold2_psrr_overlap_rows_blocked": [376, 406], "fold2_psrr_training_assertion": "PASS",
                   "summaries": summaries, "latest_resume_audits_by_group": latest_by_group,
                   "correction_totals_by_group": corrections_by_group,
                   "rng_trajectory_audits_by_group": {**status.get("rng_trajectory_audits_by_group", {}), group: rng_audits},
                   "research_comparisons": comparisons})
    if current_audit_sha != frozen_audit_sha:
        status["audit_repair"] = {
            "reason": "The historical DARG Best checkpoint pickle refers to numpy._core.multiarray, unavailable in the NumPy 1.26 runtime. In addition, numpy.random state pickle bytes differ between NumPy 1.26 and NumPy 2.2.6 even when the RNG state and consumption are identical. The audit now uses a separate NumPy 2 extraction for only DARG predictions and metrics and replays each public forward/evaluation RNG trajectory with NumPy-2-compatible hash serialization. It verifies original checkpoint SHA and independently recomputes all metrics in the primary runtime.",
            "frozen_audit_sha256": frozen_audit_sha,
            "repaired_audit_sha256": current_audit_sha,
            "frozen_script_copy": ".runtime/audit_frozen_before_repair.py",
            "compatibility_extractor": ".runtime/convert_darg_metrics.py",
            "compatibility_extractor_sha256": run.sha_file(EXP / ".runtime/convert_darg_metrics.py"),
            "compatibility_records": ".runtime/darg_best_compat.json",
            "compatibility_records_sha256": run.sha_file(EXP / ".runtime/darg_best_compat.json"),
            "complete_diff": "results/audit_repair.diff",
            "training_source_and_checkpoints_changed": False,
        }
    if group == "G1":
        status["g1_status"] = "COMPLETE_AUDITED"
        if status.get("g2_status") != "COMPLETE_AUDITED":
            status["g2_status"] = "READY_AFTER_G1_AUDIT"
        status["research_decision"] = "G1_AUDITED_G2_REQUIRED" if status.get("g2_status") != "COMPLETE_AUDITED" else status.get("research_assessment", {}).get("decision")
    else:
        status["g2_status"] = "COMPLETE_AUDITED"
        status["research_assessment"] = _research_decision(comparisons, mechanism_rows, intervention_records)
        status["research_decision"] = status["research_assessment"]["decision"]
    status["formal_training"] = "COMPLETE_GROUP_AUDITED"
    run.atomic_json(status_path, status)
    _write_report(rows, summaries, comparisons, mechanism_rows, corrections_by_group, status)
    return status


def _write_report(rows, summaries, comparisons, mechanism_rows, corrections, status):
    groups = [name for name in ("DARG", "E81.7-G1", "G1", "G2") if name in summaries]
    metric_header = "|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|"
    lines = [
        "# Experiment 81.7.1 PSRR-LCR", "",
        "## Purpose and evidence boundaries", "",
        "The experiment tests whether frozen disease Prompts select patient-local relations, construct competing ASD/TD sparse graphs, and change pairwise messages that reach the inherited classifier through a bounded CAM residual. The 16 positions are latent CAM feature slots without validated anatomical labels.", "",
        "Engineering facts below come from source inspection, frozen identities, and replay audits. Fold summaries are observed results. Research judgments are interpretations under one seed and the inherited selection protocol.", "",
        "## Historical source review", "",
        "SPromptGL's GFR learns a transductive patient graph and performs graph propagation. CAM is the modality-aware feature encoder that maps representations to CAM144. SPGS forms label-independent class cues from eight frozen CLIP RN50 text descriptions, scores local and pooled modality features against those cues, and weights local slots. RES combines local and global CAM information before classification. In this repository CAM144 reaches the original OutputLayer as a 144-wide vector. PSRR-LCR intercepts that call with a forward pre-hook and invokes the original classifier once.", "",
        "E74.4-G2 is the sole inherited DARG base. Each fold and group restores its corresponding E61 epoch-10 warmup, all three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG state, fold membership, and original optimizer order. Historical test-equals-validation Best selection, pre-CV feature selection and preprocessing, the transductive patient graph, and the Fold2 train/test overlap remain part of this inherited protocol.", "",
        "E81.7 produced near-uniform patient-level Prompt weights after averaging evidence across all 16 slots. Its ASD/TD supports differed weakly. Its direct residual contribution added four net corrections at the selected epoch, while the DARG-Best to complete E81.7 comparison lost three net predictions. E81.7 did not run the scrambled semantic control. The historical audit also recorded algebraic cancellation of the visual-baseline message in the final disease-message difference.", "",
        "## Frozen PSRR-LCR design", "",
        "The original eight CLIP RN50 Prompt encodings and their order are unchanged, including the original spelling. The frozen 8x8 SVD basis E reproduces the normalized Prompt Gram. E_center is the fixed row-normalized result of subtracting mean(E) from each row. Centering changes the Gram geometry and does not claim to preserve the original Prompt Gram. E and E_center are non-trainable.", "",
        "CAM144 is reshaped to [N,4,4,9], yielding 16 latent slots of width 9. The original four contiguous 36-value blocks are AvgPool1d(4,4)-pooled to [N,4,9]. The shared local encoder remains Linear(9,32), learned position[16,32], and LayerNorm. Each receiver has 15 non-self candidates and one Null candidate. Shared pair keys are [u_i*u_j, abs(u_j-u_i)] and values are [u_j, u_i*u_j, u_j-u_i]. The shared semantic mapping remains bias-free 64->4->8, and the model retains 7,808 added trainable parameters.", "",
        "For each patient n, receiver i, and Prompt k, evidence is logsumexp over the 15 visual-plus-semantic candidate scores minus logsumexp over visual candidate scores. There is no reduction over i. TD and ASD each apply a temperature-1.0 softmax over their four fixed Prompts. Their weighted semantic scores form D = S_ASD_sem - S_TD_sem. The two probability graphs are Entmax1.5(S_visual + 0.5D) and Entmax1.5(S_visual - 0.5D), with lambda_rel fixed at 1.0. The contrast message is sum_j((A_ASD-A_TD)*V_ij). A zero-initialized 32->9 projection produces the new residual, which is bounded by rho=0.1 of CAM norm and combined with the original local/global interaction.", "",
        "G1 uses TD Prompts [0,1,2,3] and ASD Prompts [4,5,6,7]. G2 keeps the same text, cache, architecture, initialization, training schedule, and optimizers, using TD [0,1,4,5] and ASD [2,3,6,7]. It is independently initialized from the same E61 warmup for each fold. No third formal group was trained.", "",
        "## Training protocol", "",
        "Both groups use seed 0 over the historical ten folds, begin from E61 epoch 10, train for at most 450 epochs, and stop when the historical wait counter exceeds 200. Best is the strictly improved test-fold ACC, retaining the earlier epoch on ties. DARG optimizers remain Adam at lr=0.0038 and weight decay 0.11 for GFR and CAM, followed by AdamW at lr=0.001 and weight decay 0.01 for DARG parameters. DARG loss and optimizer order remain unchanged. PSRR-LCR adds 0.001 times its raw residual penalty and an independent AdamW optimizer at 0.0005 with weight decay 0.01.", "",
        "Shared gradients are split using autograd VJPs on the same enhanced forward graph. The branch component includes the historical loss through the new residual and the weighted raw-residual regularizer. Negative branch/direct cosine triggers group-level orthogonal projection, then a single group scale enforces ||g_branch|| <= 0.1||g_direct|| separately for GFR, CAM, and DARG parameters. The epoch-12 probe records classification-to-residual-to-CAM/DARG parameter paths. This is a training stability control, not a medical contribution.", "",
        "## Six metrics", "", metric_header, "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for group in groups:
        values = [f"{summaries[group][metric]['mean']:.4f} +/- {summaries[group][metric]['population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines += ["", "## Per-fold metrics", "", "|Group|Fold|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        for row in sorted((item for item in rows if item["group"] == group), key=lambda item: int(item["fold"])):
            values = [f"{float(row[metric]):.4f}" for metric in METRICS]
            lines.append(f"|{group}|{int(row['fold'])}|" + "|".join(values) + "|")
    lines += ["", "## Excluding Fold 2", "", metric_header, "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [f"{summaries[group][metric]['exclude_fold2_mean']:.4f} +/- {summaries[group][metric]['exclude_fold2_population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines += ["", "## Paired comparisons", "", "Differences are in percentage points. W/T/L counts are fold-paired. The reported population-SD interval is descriptive and does not establish independent-patient significance.", "", "|Comparison|Subset|Metric|Mean delta pp|W/T/L|", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"|{row['comparison']}|{row['subset']}|{row['metric']}|{float(row['mean_difference_pp']):+.4f}|{row['W_T_L']}|")

    def mechanism_mean(group, key):
        values = [float(row[key]) for row in mechanism_rows
                  if row.get("group") == group and row.get("record_type") == "best_mechanism"]
        return sum(values) / len(values) if values else float("nan")

    lines += ["", "## Local Prompt and relation mechanisms", "", "|Group|TD/ASD local beta SD|TD/ASD patient beta SD|Raw/centered Prompt score spread|Mean ASD/TD support disagreement|Mean relation-difference norm|Mean contrast-message norm|Mean residual ratio|Prompt-off probability MAE|Uniform-beta probability MAE|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("G1", "G2"):
        values = [
            f"{mechanism_mean(group, 'within_patient_local_td_sd_mean'):.6g}/{mechanism_mean(group, 'within_patient_local_asd_sd_mean'):.6g}",
            f"{mechanism_mean(group, 'td_beta_patient_sd_mean'):.6g}/{mechanism_mean(group, 'asd_beta_patient_sd_mean'):.6g}",
            f"{mechanism_mean(group, 'raw_basis_prompt_spread_mean'):.6g}/{mechanism_mean(group, 'centered_basis_prompt_spread_mean'):.6g} ({mechanism_mean(group, 'centered_to_raw_prompt_spread_ratio'):.6g}x)",
            f"{mechanism_mean(group, 'asd_td_support_disagreement_rate'):.6g}",
            f"{mechanism_mean(group, 'relation_difference_norm_mean'):.6g}",
            f"{mechanism_mean(group, 'message_contrast_norm_mean'):.6g}",
            f"{mechanism_mean(group, 'residual_ratio_mean'):.6g}",
            f"{mechanism_mean(group, 'prompt_off_probability_mae'):.6g}",
            f"{mechanism_mean(group, 'uniform_beta_probability_mae'):.6g}",
        ]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines += ["", "Full per-patient predictions and probabilities for DARG Best, PSRR-off, complete PSRR-LCR, and all four fixed interventions are in `results/interventions.csv`. `results/mechanism.csv` records local beta entropy and variance, sparse edge counts, Null use, support overlap, relation and message norms, residual bounds, and gradient-control traces for both groups.", "",
              "## Corrections and main-trunk trajectory", "",
              "A is historical DARG Best, B is the current checkpoint with the added residual disabled, and C is the full model. A-to-B combines changed main-trunk weights and checkpoint selection, so it is not attributed solely to gradient conflict. B-to-C isolates the selected checkpoint's residual effect.", "",
              "|Group|Stage|Corrected|Harmed|Net|", "|---|---|---:|---:|---:|"]
    for group in ("G1", "G2"):
        for stage, values in corrections.get(group, {}).get("overall", {}).items():
            lines.append(f"|{group}|{stage}|{values['corrected']}|{values['harmed']}|{values['net']}|")
    lines += ["", "|Group|Stage|ASD corrected|ASD harmed|TD corrected|TD harmed|", "|---|---|---:|---:|---:|---:|"]
    for group in ("G1", "G2"):
        for stage, values in corrections.get(group, {}).get("class_transitions", {}).items():
            lines.append(f"|{group}|{stage}|{values['asd_corrected']}|{values['asd_harmed']}|{values['td_corrected']}|{values['td_harmed']}|")
    lines += ["", "## Gradient and checkpoint audit", "", f"G1/G2 Best readback: `{status.get('g1_best_audit')}` / `{status.get('g2_best_audit', 'PENDING')}`. G1/G2 Latest next-step replay: `{status.get('g1_latest_resume_audit')}` / `{status.get('g2_latest_resume_audit', 'PENDING')}`. Public RNG comparison against DARG: `{status.get('g1_public_rng_vs_darg')}` / `{status.get('g2_public_rng_vs_darg', 'PENDING')}`. End-to-end gradient audit: `{status.get('g1_end_to_end_gradient_audit')}` / `{status.get('g2_end_to_end_gradient_audit', 'PENDING')}`.", "", "`results/audit_status.json` stores per-fold Latest replay hashes, gradient reconstruction and group-kappa checks, historical identities, and source hashes. The audit reads Best checkpoints independently, recomputes continuous ASD probabilities and all six metrics, and replays one next training step twice from each Latest state.", "", "## Decision and limitations", "", f"Research decision: `{status.get('research_decision')}`."]
    repair = status.get("audit_repair")
    if repair:
        lines += ["", "## Post-training audit repair", "",
                  f"The frozen auditor SHA256 was `{repair['frozen_audit_sha256']}` and the repaired auditor SHA256 is `{repair['repaired_audit_sha256']}`. Historical DARG pickle metadata uses the NumPy 2 module path, so a separate NumPy 2 extractor carries only saved predictions and metrics into the primary runtime. RNG hashes also encode the NumPy ndarray pickle module path. The auditor therefore replays every public train/evaluation forward and compares both current-runtime hashes and NumPy-2-compatible hashes with the original DARG trace. It verifies original checkpoint SHA and independently recomputes metrics. The complete source diff is recorded in `{repair['complete_diff']}`. Training source and checkpoints were not changed."]
    assessment = status.get("research_assessment")
    if assessment:
        lines += [f"G1 over DARG main performance gate: `{assessment['performance_gate_g1_over_darg']}`.",
                  f"G1 over G2 medical-semantic gate: `{assessment['medical_semantic_gate_g1_over_g2']}`; G1/G2 full predictions changed for {assessment['g1_g2_full_prediction_changes']} test-fold samples.",
                  f"Mechanism gate by group: `{json.dumps(assessment['relation_mechanism_gate_by_group'], sort_keys=True)}`."]
    lines += ["", "The result inherits test-fold checkpoint selection, pre-CV preprocessing/RFE, transductive patient-graph propagation, and the Fold2 overlap. Fold2 rows 376 and 406 (zero-based) are excluded only from added-branch training contribution. A single seed and these inherited protocol choices limit scientific generalization. A candidate decision means continued research, not final MICCAI 2027 evidence.", "",
              f"Experiment directory: `{EXP}`.", f"Frozen source inventory SHA256: `{status.get('source_sha256')}`.",
              f"Audit SHA at freeze/current: `{status.get('audit_source_sha256_at_training_freeze')}` / `{status.get('audit_source_sha256_current')}`.",
              f"Audit-only code changed after training: `{status.get('audit_only_patch_after_training')}`.", ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_report(rows, summaries, comparisons, mechanism_rows, corrections, status):
    """Write the TP-PSRR report after independent audit has produced all tables."""
    groups = [name for name in ("DARG", "E81.7-G1", "G1", "G2") if name in summaries]
    metric_header = "|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|"
    lines = [
        "# Experiment 81.7.2 TP-PSRR", "",
        "## Purpose and evidence boundary", "",
        "This experiment tests fixed disease Prompt guidance of local relation construction and message propagation while preserving the inherited DARG trajectory. CAM144 is reshaped to 16 latent feature slots and is not interpreted as 16 validated anatomical regions.", "",
        "Source facts come from read-only source review, frozen identities, and replay audits. Fold summaries are observed results under one seed and the inherited test-as-validation protocol. Research conclusions are stated separately from those facts.", "",
        "## Historical review and frozen changes", "",
        "The SPromptGL paper and official/local sources were checked. GFR builds and propagates the transductive patient graph, CAM produces CAM144, SPGS uses frozen CLIP RN50 text features, and RES performs the original local/global interaction. CAM144 reaches the original Outputlayer. The experiment uses a local forward pre-hook and invokes the original classifier once.", "",
        "E74.4-G2 remains the only base. Every fold restores its E61 epoch-10 warmup, three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG, 4096 relation pairs, fold membership, and optimizer order. Pre-CV preprocessing/RFE, the transductive graph, test-as-validation Best selection, and Fold2 overlap remain inherited. Added-branch residual and raw penalty contribution are masked only for zero-based rows 376 and 406 in Fold2.", "",
        "E81.7.1 used node-local Prompt beta and a same-enhanced-graph direct VJP split. Its beta distributions remained close to uniform and its full model stayed below DARG. TP-PSRR replaces that protection with a same-state original DARG reference gradient and compares node-conditioned G1 with edge-conditioned G2.", "",
        "## Frozen TP-PSRR design", "",
        "The original eight Prompt strings, order, CLIP cache, raw normalized Gram, and SVD source are unchanged. E_center is the fixed row-centered normalized semantic basis and is not trainable. The shared local encoder, pair key/value, 64->4->8 semantic map, 15 real neighbors plus Null, Entmax1.5, symmetric ASD/TD competition, pairwise message, zero-initialized 32->9 projection, original RES, and rho=0.1 residual bound are retained. Added parameters remain 7,808.", "",
        "G1 computes beta per patient and receiver from visual-plus-semantic candidate evidence with a four-Prompt disease-group softmax. G2 computes beta independently for every patient, receiver, real candidate edge, and Prompt from that edge's semantic scores. Both groups use medical Prompt indices TD [0,1,2,3] and ASD [4,5,6,7]. G2 changes only the beta application dimension.", "",
        "## DARG-reference gradient protection", "",
        "Each training step first computes the original DARG loss L_B with the residual disabled and records g_B for GFR, CAM, and DARG. All RNG components and model/CAM buffers are restored. The enhanced loss L_F = L_DARG(enhanced) + 0.001 L_raw then uses the same stochastic masks and yields g_F. delta_g = g_F - g_B. Negative group dot products trigger vector orthogonal projection, followed by one group-level scale enforcing ||delta_g_controlled|| <= 0.05||g_B||. The PSRR parameters receive the unmodified enhanced gradient. This protection is a training control and is not counted as a medical contribution.", "",
        "## Metrics", "", metric_header, "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [f"{summaries[group][metric]['mean']:.4f} +/- {summaries[group][metric]['population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines += ["", "## Per-fold metrics", "", "|Group|Fold|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        for row in sorted((item for item in rows if item["group"] == group), key=lambda item: int(item["fold"])):
            lines.append(f"|{group}|{int(row['fold'])}|" + "|".join(f"{float(row[metric]):.4f}" for metric in METRICS) + "|")
    lines += ["", "## Excluding Fold2", "", metric_header, "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [f"{summaries[group][metric]['exclude_fold2_mean']:.4f} +/- {summaries[group][metric]['exclude_fold2_population_sd']:.4f}" for metric in METRICS]
        lines.append("|" + group + "|" + "|".join(values) + "|")
    lines += ["", "## Paired comparisons", "", "Differences are percentage points. W/T/L is fold-paired. The descriptive interval is not independent-patient significance.", "", "|Comparison|Subset|Metric|Mean delta pp|W/T/L|", "|---|---|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"|{row['comparison']}|{row['subset']}|{row['metric']}|{float(row['mean_difference_pp']):+.4f}|{row['W_T_L']}|")
    def mmean(group, key):
        vals = [float(row[key]) for row in mechanism_rows if row.get("group") == group and row.get("record_type") == "best_mechanism" and key in row]
        return sum(vals) / len(vals) if vals else float("nan")
    lines += ["", "## Mechanism audit", "", "|Group|Mode|Prompt entropy TD/ASD|Patient beta SD TD/ASD|Within-patient beta SD TD/ASD|ASD/TD support disagreement|Relation difference norm|Contrast message norm|Residual ratio|Prompt-off MAE|Uniform-beta MAE|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("G1", "G2"):
        mode_rows = [row for row in mechanism_rows if row.get("group") == group and row.get("record_type") == "best_mechanism"]
        mode = mode_rows[0].get("selection_mode", run.CFG["groups"][group]["selection_mode"]) if mode_rows else run.CFG["groups"][group]["selection_mode"]
        lines.append(f"|{group}|{mode}|{mmean(group, 'td_beta_entropy_mean'):.6g}/{mmean(group, 'asd_beta_entropy_mean'):.6g}|{mmean(group, 'td_beta_patient_sd_mean'):.6g}/{mmean(group, 'asd_beta_patient_sd_mean'):.6g}|{mmean(group, 'within_patient_local_td_sd_mean'):.6g}/{mmean(group, 'within_patient_local_asd_sd_mean'):.6g}|{mmean(group, 'asd_td_support_disagreement_rate'):.6g}|{mmean(group, 'relation_difference_norm_mean'):.6g}|{mmean(group, 'message_contrast_norm_mean'):.6g}|{mmean(group, 'residual_ratio_mean'):.6g}|{mmean(group, 'prompt_off_probability_mae'):.6g}|{mmean(group, 'uniform_beta_probability_mae'):.6g}|")
    lines += ["", "## Corrections and trajectory", "", "A is historical DARG Best, B is the current checkpoint with the added residual disabled, and C is the complete checkpoint. A-to-B combines changed main-trunk weights and Best epoch. B-to-C isolates the direct residual effect at the selected checkpoint.", "", "|Group|Stage|Corrected|Harmed|Net|", "|---|---|---:|---:|---:|"]
    for group in ("G1", "G2"):
        for stage, values in corrections.get(group, {}).get("overall", {}).items():
            lines.append(f"|{group}|{stage}|{values['corrected']}|{values['harmed']}|{values['net']}|")
    lines += ["", "## Independent audit", "", f"G1/G2 Best: `{status.get('g1_best_audit')}` / `{status.get('g2_best_audit', 'PENDING')}`. G1/G2 Latest replay: `{status.get('g1_latest_resume_audit')}` / `{status.get('g2_latest_resume_audit', 'PENDING')}`. Public RNG vs DARG: `{status.get('g1_public_rng_vs_darg')}` / `{status.get('g2_public_rng_vs_darg', 'PENDING')}`. End-to-end gradient: `{status.get('g1_end_to_end_gradient_audit')}` / `{status.get('g2_end_to_end_gradient_audit', 'PENDING')}`.", "", "The auditor independently reads Best checkpoints, recomputes all six metrics and continuous ASD probabilities, audits interventions, replays Latest, verifies source/checkpoint identities, and compares current and NumPy-2-compatible RNG hashes. Any audit-only repair is disclosed in audit_status.json and does not change training code or checkpoints.", "", "## Research decision", "", f"`{status.get('research_decision')}`", "", "The result remains limited by one seed, test-as-validation selection, pre-CV preprocessing/RFE, transductive propagation, and the Fold2 overlap. A mechanism pass does not replace the ACC and Hard-AUC performance gate.", "", f"Experiment directory: `{EXP}`", f"Frozen source inventory SHA256: `{status.get('source_sha256')}`", f"Report and tables: `{RESULTS}`"]
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def preflight_checks():
    import numpy as np
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.1, 0.9]], dtype=np.float64)
    metrics = _saved_metrics(labels, probabilities.argmax(axis=1), probabilities)
    expected = {"acc": 0.75, "hard_auc": 0.75, "prob_auc": 1.0, "sen": 0.5, "spe": 1.0}
    if any(abs(metrics[key] - value) > 1e-12 for key, value in expected.items()):
        raise RuntimeError(f"independent metric audit self-check failed: {metrics}")
    folds_left = {fold: {metric: 0.6 for metric in METRICS} for fold in range(1, 11)}
    folds_right = {fold: {metric: 0.5 for metric in METRICS} for fold in range(1, 11)}
    paired = _paired("self_check", folds_left, folds_right)
    if len(paired) != 12 or any(abs(float(row["mean_difference_pp"]) - 10.0) > 1e-12 for row in paired):
        raise RuntimeError("paired metric audit self-check failed")
    transitions = _class_transitions([0, 1, 0, 1], [1, 0, 1, 0], labels)
    if any(transitions[key] != 1 for key in ("asd_harmed", "asd_corrected", "td_harmed", "td_corrected")):
        raise RuntimeError("class-specific correction audit self-check failed")
    if _research_decision([], [], [])["decision"] != "NO_GO_PERFORMANCE":
        raise RuntimeError("research-decision gate self-check failed")
    return {"status": "PASS", "metric_recomputation": True, "paired_all10_and_exclude_fold2": True,
            "class_specific_transitions": True, "decision_gate": True}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    result = audit_group(args.group)
    print(f"AUDIT {args.group} PASS research={result['research_decision']}", flush=True)


if __name__ == "__main__":
    main()
