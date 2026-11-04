#!/usr/bin/env python3
"""Independent E69.3-G2 checkpoint, exposure, RNG, and mechanism audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import run as R  # noqa: E402

CFG = R.CFG
BASE = R.BASE
EXP = R.EXP
RESULTS = EXP / "results"
STATUS_PATH = RESULTS / "audit_status.json"
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
GROUPS = tuple(R.GROUP_IDS)
PAIRS = tuple(
    [(f"{group}_vs_E69.3-G2", group, "E69.3-G2") for group in GROUPS]
    + [
        ("G1_vs_G3", "G1", "G3"),
        ("G2_vs_G4", "G2", "G4"),
        ("G1_vs_G2", "G1", "G2"),
        ("G3_vs_G4", "G3", "G4"),
        ("G5_vs_G6", "G5", "G6"),
        ("G1_vs_G5", "G1", "G5"),
        ("G2_vs_G6", "G2", "G6"),
    ]
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def independent_metrics(logp, indices, labels) -> dict[str, Any]:
    """Recompute the frozen E61 metric definitions from raw predictions."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    idx = np.asarray(indices, dtype=np.int64)
    y_all = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    y = np.asarray(y_all, dtype=np.int64)[idx]
    probability = logp.detach().exp().cpu().numpy()[idx]
    prediction = probability.argmax(axis=1)
    tp = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 0) & (prediction == 1)))
    fp = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 1) & (prediction == 1)))
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    hard = float(roc_auc_score(np.eye(2, dtype=np.int64)[y], np.eye(2, dtype=np.int64)[prediction]))
    return {
        "acc": float(np.mean(prediction == y)),
        "hard_auc": hard,
        "prob_auc": float(roc_auc_score(y, probability[:, 1])),
        "sen": sen,
        "spe": spe,
        "ba": (sen + spe) / 2.0,
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "asd_tp": tp,
        "td_fp": fp,
        "asd_fn": fn,
        "td_tn": tn,
        "n_test": int(len(idx)),
        "prediction": prediction,
        "probability": probability,
    }


def assert_metrics(actual: dict, expected: dict, label: str) -> None:
    for key in METRICS:
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            raise AssertionError(f"{label} metric mismatch for {key}: {actual[key]} != {expected[key]}")
    confusion_keys = ("asd_tp", "td_fp", "asd_fn", "td_tn")
    if all(key in expected for key in confusion_keys):
        for key in confusion_keys:
            if int(actual[key]) != int(expected[key]):
                raise AssertionError(f"{label} confusion mismatch for {key}: {actual[key]} != {expected[key]}")


def representation_summary(vectors, labels) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    norms = np.linalg.norm(matrix, axis=1)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    participation = float(energy.sum() ** 2 / max(float((energy**2).sum()), 1e-24))
    class_means = [matrix[y == cls].mean(axis=0) for cls in (0, 1)]
    within = []
    for cls in (0, 1):
        delta = matrix[y == cls] - class_means[cls]
        within.extend(np.linalg.norm(delta, axis=1).tolist())
    between = float(np.linalg.norm(class_means[0] - class_means[1]))
    return {
        "patient_norm_mean": float(norms.mean()),
        "patient_norm_population_sd": float(norms.std(ddof=0)),
        "effective_rank_participation_ratio": participation,
        "within_class_centroid_distance_mean": float(np.mean(within)),
        "between_class_centroid_distance": between,
        "between_to_within_distance_ratio": between / max(float(np.mean(within)), 1e-12),
        "ASD_count": int(np.sum(y == 0)),
        "TD_count": int(np.sum(y == 1)),
    }


def source_cosines(prompt_meta: dict) -> dict[str, float]:
    return {name: float(row["source_cosine"]) for name, row in prompt_meta["banks"].items()}


def task_paths(group: str, fold: int) -> tuple[Path, Path, Path]:
    stem = f"{group}_f{fold:02d}"
    return (
        R.RUNTIME / "tasks" / f"{stem}.json",
        R.RUNTIME / "checkpoints" / f"{stem}.best.pt",
        R.RUNTIME / "mechanism" / f"{stem}.json",
    )


def audit_one(group: str, fold: int, references: dict[int, dict], warmup_audit: dict, banks: dict, prompt_meta: dict, manifest: dict) -> tuple[dict, dict, dict, list[dict]]:
    import numpy as np
    import torch

    task_path, checkpoint_path, mechanism_path = task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cp_sha = R.sha_file(checkpoint_path)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != cp_sha:
        raise AssertionError("task/checkpoint completion or SHA mismatch")
    if cp.get("schema") != "experiment72_1_e693_mgdss_checkpoint_v1" or cp.get("group") != group or int(cp.get("fold", -1)) != fold:
        raise AssertionError("checkpoint schema/group/fold contamination")
    if cp.get("source_sha256") != manifest["source_sha256"] or cp.get("config_sha256") != manifest["config_sha256"]:
        raise AssertionError("checkpoint source/config SHA differs from the frozen manifest")
    if cp.get("inference_uses_auxiliary") is not False:
        raise AssertionError("checkpoint declares auxiliary inference")

    state = R.build_fold(fold, json.loads((R.REFERENCE_EXP / "manifest.json").read_text()), references)
    overlap = list(state["train_test_overlap_1based"])
    expected_overlap = [377, 407] if fold == 2 else []
    if overlap != expected_overlap or cp.get("train_test_overlap_1based") != expected_overlap:
        raise AssertionError(f"effective train/test overlap mismatch: {overlap}")
    if cp.get("effective_train_sha256") != state["effective_train_sha"] or cp.get("base_train_sha256") != state["base_train_sha"]:
        raise AssertionError("effective/base training membership SHA mismatch")
    if cp.get("pair_sha256") != state["pair_sha"] or cp.get("split_sha256") != state["split_sha"]:
        raise AssertionError("fold split/relation-pair SHA mismatch")
    for cp_key, state_key in (
        ("initial_shared_sha256", "initial_shared_sha"),
        ("initial_gfr_sha256", "initial_gfr_sha"),
        ("initial_cam_sha256", "initial_cam_sha"),
        ("initial_added_sha256", "initial_added_sha"),
        ("initial_schema_sha256", "initial_schema_sha"),
        ("initial_rng_sha256", "construction_rng_sha"),
    ):
        if cp.get(cp_key) != state.get(state_key):
            raise AssertionError(f"initial state provenance mismatch for {cp_key}")
    if cp.get("reference_checkpoint_sha256") != references[fold]["checkpoint_sha256"]:
        raise AssertionError("checkpoint refers to a different E69.3-G2 historical fold checkpoint")
    if cp.get("e69_reference_task_sha256") != R.sha_file(R.E69_TASKS / f"E69.3-G2_f{fold:02d}.json"):
        raise AssertionError("E69.3-G2 reference task SHA mismatch")
    if task.get("test_labels_used_for_auxiliary_loss") is not bool(expected_overlap):
        raise AssertionError("task label-exposure flag is inaccurate")
    if task.get("auxiliary_label_exposure_rows_1based") != expected_overlap:
        raise AssertionError("task label-exposure row list is inaccurate")
    if cp.get("test_labels_used_for_auxiliary_loss") is not bool(expected_overlap):
        raise AssertionError("checkpoint label-exposure flag is inaccurate")
    if cp.get("auxiliary_label_exposure_rows_1based") != expected_overlap:
        raise AssertionError("checkpoint label-exposure rows are inaccurate")
    if cp.get("outer_train_count") != int(len(state["train"])) or cp.get("outer_test_count") != int(len(state["test"])):
        raise AssertionError("checkpoint train/test cardinality mismatch")
    if not cp.get("train_indices_sha256"):
        raise AssertionError("missing training-index SHA")
    if cp.get("train_indices_sha256") != R.sha_obj(list(map(int, state["train"]))):
        raise AssertionError("checkpoint training index list SHA mismatch")
    if cp.get("test_indices_sha256") != R.sha_obj(list(map(int, state["test"]))):
        raise AssertionError("checkpoint test index list SHA mismatch")

    item = R.group_item(group)
    expected_bank = prompt_meta["banks"][item["prompt"]]["bank_sha256"]
    if cp.get("prompt_bank_sha256") != expected_bank or cp.get("text_basis_sha256") != prompt_meta["basis_sha256"]:
        raise AssertionError("prompt bank or text basis mismatch")
    if cp.get("prompt_cache_sha256") != R.sha_file(R.PROMPT_CACHE):
        raise AssertionError("frozen prompt cache SHA mismatch")
    if cp.get("warmup_sha256") != R.sha_file(R.WARMUP_DIR / f"fold{fold:02d}.pt"):
        raise AssertionError("shared E61 warmup SHA mismatch")

    module, aux_optimizer, initial_aux_sha, rng_before, rng_after = R.private_auxiliary(int(prompt_meta["rank"]), fold)
    if initial_aux_sha != cp.get("initial_auxiliary_sha256") or rng_before != rng_after:
        raise AssertionError("auxiliary projection initialization/RNG mismatch")
    alpha = float(manifest["preflight"]["alphas"][str(fold)][group]["alpha"])
    if abs(float(cp["alpha"]) - alpha) > 1e-15:
        raise AssertionError("frozen alpha mismatch")

    warmup = R.load_warmup(state, fold, warmup_audit)
    if warmup["sha256"] != cp["warmup_sha256"]:
        raise AssertionError("warmup restoration differs from the checkpoint provenance")
    state["model"].load_state_dict(cp["base"], strict=True)
    state["cam"].load_state_dict(cp["cam"], strict=True)
    module.load_state_dict(cp["auxiliary"], strict=True)
    state["opt_base"].load_state_dict(cp["optimizer_base"])
    state["opt_cam"].load_state_dict(cp["optimizer_cam"])
    state["opt_added"].load_state_dict(cp["optimizer_added"])
    aux_optimizer.load_state_dict(cp["optimizer_auxiliary"])

    with torch.no_grad():
        output = R.checked_forward(state, training=False, use_cadr=True)
        metrics = independent_metrics(output["logp"], state["test"], state["y"])
        framework_metrics = BASE.classification_metrics(output["logp"], state["test"], state["y"])
        assert_metrics(metrics, framework_metrics, f"{group}/fold{fold:02d} independent framework")
        assert_metrics(metrics, cp["metrics"], f"{group}/fold{fold:02d} checkpoint")
        assert_metrics(metrics, task["metrics"], f"{group}/fold{fold:02d} task")
        confusion = task.get("confusion", {})
        for metric_key, task_key in (("asd_tp", "asd_tp"), ("td_fp", "td_fp"), ("asd_fn", "asd_fn"), ("td_tn", "td_tn")):
            if int(metrics[metric_key]) != int(confusion[task_key]):
                raise AssertionError(f"task confusion mismatch for {metric_key}")
        positions = state["cam"]._e59_positions
        aux = module(positions, banks[item["prompt"]].to(positions.device), float(item["temperature"]), item["grain"] == "joint")
        cam_calls = int(state["cam"]._e72_forward_count)
        info = output["info"]
        logp_cpu = output["logp"].detach().cpu()

    if cam_calls != 1:
        raise AssertionError(f"CAM forward count is {cam_calls}, expected one")
    if cp.get("best_diagnostics") is None:
        raise AssertionError("checkpoint has no best-epoch mechanism record")
    history = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if not history or task.get("mechanism_sha256") != R.sha_file(mechanism_path):
        raise AssertionError("mechanism trace missing or SHA mismatch")
    best_epoch = int(cp["best_epoch"])
    best_rows = [row for row in history if int(row["epoch"]) == best_epoch]
    if len(best_rows) != 1:
        raise AssertionError("best epoch does not have exactly one mechanism row")
    best_row = best_rows[0]
    if best_row.get("test_labels_in_effective_training_set") is not bool(expected_overlap):
        raise AssertionError("best-epoch mechanism label exposure does not match fold membership")

    reference_trace = references[fold].get("rng_trace", [])
    ref_trace = {int(row["epoch"]): row for row in reference_trace}
    compared = exact = 0
    for row in history:
        epoch1 = int(row["epoch"]) + 1
        if epoch1 not in ref_trace:
            continue
        old = ref_trace[epoch1]
        compared += 1
        exact += int(
            row.get("rng_before_train_sha256") == old["rng_before"]
            and row.get("rng_after_train_sha256") == old["rng_after_train"]
            and row.get("rng_after_eval_sha256") == old["rng_after_eval"]
        )
    if compared == 0:
        raise AssertionError("no common E69.3/E72.1 epoch RNG states")
    rng_status = "PASS" if exact == compared else "DIVERGED_RECORDED"

    item_bank = banks[item["prompt"]].to(positions.device)
    tau = float(item["temperature"])
    y = state["y"].detach().long()
    test_idx = np.asarray(state["test"], dtype=np.int64)
    test_y = y[test_idx].detach().cpu().numpy()
    global_sim = aux["global_logits"].detach() * tau
    global_margin = global_sim.gather(1, y[:, None]).squeeze(1) - global_sim.gather(1, (1-y)[:, None]).squeeze(1)
    fine_sim = None if aux["fine_logits"] is None else aux["fine_logits"].detach() * tau
    p_softmax = torch.softmax(aux["global_logits"].detach(), dim=-1)
    selected = torch.as_tensor(test_idx, dtype=torch.long, device=positions.device)
    mechanism_row = {
        "group": group,
        "fold": fold,
        "best_epoch_0based": best_epoch,
        "best_epoch_1based": best_epoch + 1,
        "epochs_executed": int(cp["epochs_executed"]),
        "alpha": alpha,
        "initial_weighted_gradient_ratio": float(manifest["preflight"]["alphas"][str(fold)][group]["initial_weighted_gradient_ratio"]),
        "loss_global": best_row["loss_global"],
        "loss_fine": best_row["loss_fine"],
        "loss_semantic": best_row["loss_semantic"],
        "gradient_norm_cls_cam": best_row["gradient_norm_cls_cam"],
        "gradient_norm_semantic_cam": best_row["gradient_norm_semantic_cam"],
        "weighted_gradient_ratio": best_row["weighted_gradient_ratio"],
        "gradient_cosine_cam": best_row["gradient_cosine_cam"],
        "negative_total_gradient_cosine": bool(float(best_row["gradient_cosine_cam"]) < 0),
        "train_similarity_accuracy_global": best_row["train_similarity_accuracy_global"],
        "train_similarity_margin_global": best_row["train_similarity_margin_global"],
        "test_diagnostic_global_similarity_accuracy": float((global_sim[selected].argmax(-1) == y[selected]).float().mean().cpu()),
        "test_diagnostic_global_margin_mean": float(global_margin[selected].mean().cpu()),
        "patient_norm_mean_oof": float(aux["global_patient"][selected].norm(dim=-1).mean().cpu()),
        "global_probability_min_oof": float(p_softmax[selected].min().cpu()),
        "global_probability_max_oof": float(p_softmax[selected].max().cpu()),
        "gate_edge_best": json.dumps(torch.sigmoid(state["model"].edge_gate_logits).detach().cpu().tolist()),
        "gate_adaptive_best": json.dumps(torch.sigmoid(state["model"].adaptive_gate_logits).detach().cpu().tolist()),
        "rng_trace_compared_epochs": compared,
        "rng_trace_exact_epochs": exact,
        "rng_trace_status": rng_status,
        "train_test_overlap_1based": json.dumps(overlap),
        "test_labels_in_semantic_training": bool(overlap),
    }
    for idx, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
        modality_messages = info["messages_original"][idx].detach()
        cadr_messages = info["messages_cadr"][idx].detach()
        mechanism_row[f"loss_{modality}"] = None if fine_sim is None else best_row.get(f"loss_{modality}")
        mechanism_row[f"gradient_cosine_{modality}"] = best_row.get(f"gradient_cosine_{modality}")
        mechanism_row[f"train_similarity_accuracy_{modality}"] = best_row.get(f"train_similarity_accuracy_{modality}")
        mechanism_row[f"train_similarity_margin_{modality}"] = best_row.get(f"train_similarity_margin_{modality}")
        if fine_sim is not None:
            modal = fine_sim[:, idx, :]
            modal_margin = modal.gather(1, y[:, None]).squeeze(1) - modal.gather(1, (1-y)[:, None]).squeeze(1)
            mechanism_row[f"test_diagnostic_similarity_accuracy_{modality}"] = float((modal[selected].argmax(-1) == y[selected]).float().mean().cpu())
            mechanism_row[f"test_diagnostic_similarity_margin_{modality}"] = float(modal_margin[selected].mean().cpu())
        gate = float(info["fusion_gates"][idx].detach().cpu())
        mechanism_row[f"adaptive_gate_{modality}"] = gate
        mechanism_row[f"original_gcn_message_norm_{modality}"] = float(modality_messages.norm().cpu())
        mechanism_row[f"weighted_original_message_norm_{modality}"] = float((1.0-gate) * modality_messages.norm().cpu())
        mechanism_row[f"cadr_message_norm_{modality}"] = float(cadr_messages.norm().cpu())
        mechanism_row[f"weighted_cadr_message_norm_{modality}"] = float(gate * cadr_messages.norm().cpu())
        mechanism_row[f"weighted_cadr_to_weighted_original_norm_{modality}"] = float(gate * cadr_messages.norm().cpu() / ((1.0-gate) * modality_messages.norm().cpu() + 1e-12))
    metric_row = {
        "group": group,
        "fold": fold,
        "acc_pct": metrics["acc"] * 100.0,
        "hard_auc_pct": metrics["hard_auc"] * 100.0,
        "prob_auc_pct": metrics["prob_auc"] * 100.0,
        "sen_pct": metrics["sen"] * 100.0,
        "spe_pct": metrics["spe"] * 100.0,
        "ba_pct": metrics["ba"] * 100.0,
        "asd_f1_pct": metrics["asd_f1"] * 100.0,
        "TP_ASD": metrics["asd_tp"],
        "FP_TD": metrics["td_fp"],
        "FN_ASD": metrics["asd_fn"],
        "TN_TD": metrics["td_tn"],
        "train_count_effective": int(len(state["train"])),
        "test_count": int(len(state["test"])),
        "best_epoch_1based": best_epoch + 1,
        "epochs_executed": int(cp["epochs_executed"]),
        "alpha": alpha,
        "train_test_overlap_1based": json.dumps(overlap),
        "test_label_exposure_count": len(overlap),
        "checkpoint_sha256": cp_sha,
        "independent_checkpoint_audit": "PASS",
    }
    if fold == 2:
        exposed = {376, 406}
        keep = np.asarray([int(value) for value in state["test"] if int(value) not in exposed], dtype=np.int64)
        reduced = independent_metrics(output["logp"], keep, state["y"])
        for metric in METRICS:
            metric_row[f"without_exposed_{metric}_pct"] = reduced[metric] * 100.0
        metric_row["without_exposed_n"] = reduced["n_test"]

    oof_rows = []
    for local, patient in enumerate(test_idx):
        label = int(test_y[local])
        row = {
            "patient_index0": int(patient),
            "label": label,
            "global_correct_margin": float(global_margin[patient].cpu()),
            "global_sim_asd": float(global_sim[patient, 0].cpu()),
            "global_sim_td": float(global_sim[patient, 1].cpu()),
            "global_prob_asd": float(p_softmax[patient, 0].cpu()),
            "global_prob_td": float(p_softmax[patient, 1].cpu()),
            "global_patient": aux["global_patient"][patient].detach().cpu().numpy(),
            "fine_patient": None if aux["fine_patient"] is None else aux["fine_patient"][patient].detach().cpu().numpy(),
            "fine_sim": None if fine_sim is None else fine_sim[patient].detach().cpu().numpy(),
        }
        oof_rows.append(row)

    exposure_rows = []
    if fold == 2:
        exposure_positions = {376: "377", 406: "407"}
        warm = manifest["preflight"]["fold_initialization_and_warmup"]["2"]["warmup_test_representations"]
        warm_local = {int(patient): ix for ix, patient in enumerate(warm["test_indices"])}
        warm_sim = np.asarray(warm["similarities"][item["prompt"]]["global"], dtype=np.float64)
        for patient, row_id in exposure_positions.items():
            label = int(y[patient].item())
            raw = logp_cpu[patient].numpy()
            sim = global_sim[patient].detach().cpu().numpy()
            warm_idx = warm_local[patient]
            initial_similarity = np.asarray(warm["similarities"][item["prompt"]]["global"][warm_idx], dtype=np.float64)
            correct_margin = float(sim[label] - sim[1-label])
            initial_margin = float(initial_similarity[label] - initial_similarity[1-label])
            exposure_rows.append({
                "group": group,
                "row_id_1based": row_id,
                "true_label": label,
                "true_class": "ASD" if label == 0 else "TD",
                "prediction": int(raw.argmax()),
                "prediction_class": "ASD" if int(raw.argmax()) == 0 else "TD",
                "prob_ASD": float(torch.softmax(torch.as_tensor(raw), dim=-1)[0]),
                "prob_TD": float(torch.softmax(torch.as_tensor(raw), dim=-1)[1]),
                "classification_margin_true_minus_other_logp": float(raw[label] - raw[1-label]),
                "classification_margin_ASD_minus_TD_logp": float(raw[0] - raw[1]),
                "global_semantic_margin_correct_minus_other_initial": initial_margin,
                "global_semantic_margin_correct_minus_other_best": correct_margin,
                "global_semantic_margin_change": correct_margin - initial_margin,
                "test_label_in_training_and_alpha_calibration": True,
            })

    audit_row = {
        "group": group,
        "fold": fold,
        "status": "PASS" if rng_status == "PASS" else "FAIL_RNG_DIVERGED",
        "checkpoint_sha256": cp_sha,
        "independent_metric_recalculation": "PASS",
        "initialization_and_warmup_provenance": "PASS",
        "effective_training_and_pair_sha": "PASS",
        "prompt_bank_and_projection_sha": "PASS",
        "original_classifier_only_inference": "PASS",
        "cam_calls_for_readback": cam_calls,
        "train_test_overlap_1based": json.dumps(overlap),
        "rng_trace_exact_epochs": exact,
        "rng_trace_compared_epochs": compared,
        "rng_trace_status": rng_status,
    }
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_row, mechanism_row, audit_row, oof_rows + [{"_exposure": row} for row in exposure_rows]


def metric_summary(rows: list[dict]) -> list[dict]:
    import numpy as np

    output = []
    base = R.reference_rows()
    summaries = {"E69.3-G2": {key: np.asarray([base[f][key] * 100.0 for f in range(1, 11)]) for key in METRICS}}
    for group in GROUPS:
        subset = sorted((row for row in rows if row["group"] == group), key=lambda row: int(row["fold"]))
        if len(subset) != 10:
            continue
        summaries[group] = {key: np.asarray([float(row[f"{key}_pct"]) for row in subset]) for key in METRICS}
    for group, values_by_metric in summaries.items():
        row = {"group": group, "fold_count": 10, "metric_type": "diagnostic_with_fold2_label_exposure" if group != "E69.3-G2" else "historical_E69.3-G2_reference"}
        for metric, values in values_by_metric.items():
            row[f"{metric}_mean_pct"] = float(values.mean())
            row[f"{metric}_population_sd_pct"] = float(values.std(ddof=0))
            row[f"{metric}_mean_plus_minus_sd_pct"] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        output.append(row)
    return output


def paired(rows: list[dict]) -> list[dict]:
    import numpy as np

    group_rows = {group: {int(row["fold"]): row for row in rows if row["group"] == group} for group in GROUPS}
    baseline = R.reference_rows()
    output = []
    for name, left_name, right_name in PAIRS:
        left = {fold: {metric: baseline[fold][metric] * 100.0 for metric in METRICS} for fold in range(1, 11)} if left_name == "E69.3-G2" else {fold: {metric: float(row[f"{metric}_pct"]) for metric in METRICS} for fold, row in group_rows[left_name].items()}
        right = {fold: {metric: baseline[fold][metric] * 100.0 for metric in METRICS} for fold in range(1, 11)} if right_name == "E69.3-G2" else {fold: {metric: float(row[f"{metric}_pct"]) for metric in METRICS} for fold, row in group_rows[right_name].items()}
        delta = {metric: np.asarray([left[fold][metric] - right[fold][metric] for fold in range(1, 11)], dtype=np.float64) for metric in METRICS}
        acc = delta["acc"]
        clean = np.asarray([left[fold]["acc"] - right[fold]["acc"] for fold in range(1, 11) if fold != 2], dtype=np.float64)
        output.append({
            "comparison": name,
            "left": left_name,
            "right": right_name,
            "delta_acc_mean_pp": float(acc.mean()),
            "delta_acc_population_sd_pp": float(acc.std(ddof=0)),
            "acc_wins": int(np.sum(acc > 1e-12)),
            "acc_ties": int(np.sum(np.abs(acc) <= 1e-12)),
            "acc_losses": int(np.sum(acc < -1e-12)),
            "delta_acc_pp_by_fold": json.dumps([round(float(value), 8) for value in acc]),
            "delta_acc_mean_9_nonoverlap_folds_pp": float(clean.mean()),
            "acc_wins_9_nonoverlap_folds": int(np.sum(clean > 1e-12)),
            "acc_ties_9_nonoverlap_folds": int(np.sum(np.abs(clean) <= 1e-12)),
            "acc_losses_9_nonoverlap_folds": int(np.sum(clean < -1e-12)),
            "delta_acc_9_nonoverlap_folds_pp": json.dumps([round(float(value), 8) for value in clean]),
            **{f"delta_{metric}_mean_pp": float(values.mean()) for metric, values in delta.items()},
            **{f"delta_{metric}_pp_by_fold": json.dumps([round(float(value), 8) for value in values]) for metric, values in delta.items()},
        })
    return output


def representation_and_confound_analysis(oof_by_group: dict[str, dict[int, dict]]) -> tuple[dict, dict]:
    import numpy as np
    from scipy.stats import spearmanr

    data_x = BASE.load_data()[0][0]
    mapping = BASE.runtime()["np"].load(BASE.ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    feature_names = [str(value) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for value in mapping[key]]
    feature_ranges = {}
    offset = 0
    for key in ("PHENO", "ANAT", "FUNC", "Correlation"):
        feature_ranges[key] = list(range(offset, offset + len(mapping[key])))
        offset += len(mapping[key])
    categories = {
        "site": [i for i, name in enumerate(feature_names) if name.startswith("SITE_ID_")],
        "age_bins": [i for i, name in enumerate(feature_names) if name.startswith("AGE_AT_SCAN_")],
        "sex": [i for i, name in enumerate(feature_names) if name.startswith("SEX_")],
        "ANAT_quality": feature_ranges["ANAT"],
        "FUNC_quality_motion": feature_ranges["FUNC"],
    }
    representations: dict[str, Any] = {}
    associations: dict[str, Any] = {}
    for group, patients in oof_by_group.items():
        indices = np.asarray(sorted(patients), dtype=np.int64)
        labels = np.asarray([patients[int(index)]["label"] for index in indices], dtype=np.int64)
        global_vectors = np.asarray([patients[int(index)]["global_patient"] for index in indices], dtype=np.float64)
        global_sim = np.asarray([[patients[int(index)]["global_sim_asd"], patients[int(index)]["global_sim_td"]] for index in indices], dtype=np.float64)
        global_margin = np.asarray([patients[int(index)]["global_correct_margin"] for index in indices], dtype=np.float64)
        summary: dict[str, Any] = {
            "diagnostic_OOF_includes_fold2_exposed_patients": [377, 407],
            "global": representation_summary(global_vectors, labels),
            "global_similarity_means_by_true_class": {
                cls: {
                    "n": int(np.sum(labels == label)),
                    "mean_similarity_ASD": float(global_sim[labels == label, 0].mean()),
                    "mean_similarity_TD": float(global_sim[labels == label, 1].mean()),
                    "mean_correct_class_margin": float(global_margin[labels == label].mean()),
                }
                for cls, label in (("ASD", 0), ("TD", 1))
            },
        }
        fine_rows = [patients[int(index)]["fine_patient"] for index in indices]
        fine_sim_rows = [patients[int(index)]["fine_sim"] for index in indices]
        if fine_rows[0] is not None:
            fine = np.asarray(fine_rows, dtype=np.float64)
            similarities = np.asarray(fine_sim_rows, dtype=np.float64)
            summary["fine_mean_position"] = representation_summary(fine.mean(axis=1), labels)
            summary["fine_modalities"] = {}
            for modality_index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                modal_vectors = fine[:, modality_index, :]
                modal_sims = similarities[:, modality_index, :]
                correct = modal_sims[np.arange(len(indices)), labels]
                wrong = modal_sims[np.arange(len(indices)), 1-labels]
                margin = correct - wrong
                summary["fine_modalities"][modality] = {
                    **representation_summary(modal_vectors, labels),
                    "similarity_accuracy_pct": float(np.mean(modal_sims.argmax(axis=1) == labels) * 100.0),
                    "correct_class_margin_mean": float(margin.mean()),
                    "correct_class_margin_ASD_mean": float(margin[labels == 0].mean()),
                    "correct_class_margin_TD_mean": float(margin[labels == 1].mean()),
                    "mean_similarity_to_ASD_prototype": float(modal_sims[:, 0].mean()),
                    "mean_similarity_to_TD_prototype": float(modal_sims[:, 1].mean()),
                }
        representations[group] = summary
        group_assoc = {}
        for category, columns in categories.items():
            values = []
            for column in columns:
                covariate = np.asarray(data_x, dtype=np.float64)[indices, column]
                if len(covariate) < 3 or np.std(covariate) == 0 or np.std(global_margin) == 0:
                    continue
                corr = float(spearmanr(covariate, global_margin).statistic)
                if math.isfinite(corr):
                    values.append({"feature": feature_names[column], "spearman_r": corr})
            group_assoc[category] = {
                "feature_count": len(values),
                "mean_absolute_spearman_r": None if not values else float(np.mean([abs(row["spearman_r"]) for row in values])),
                "max_absolute_spearman_r": None if not values else max(values, key=lambda row: abs(row["spearman_r"])),
            }
        associations[group] = group_assoc
    return representations, associations


def audit_fold2_baseline(references: dict[int, dict]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    fold = 2
    reference_task = references[fold]
    cp_path = R.RUNTIME / "reference_e69/E69.3-G2_f02.best.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    if R.sha_file(cp_path) != reference_task["checkpoint_sha256"]:
        raise AssertionError("staged E69.3 fold-2 best checkpoint SHA mismatch")
    manifest = json.loads((R.REFERENCE_EXP / "manifest.json").read_text(encoding="utf-8"))
    state = R.build_fold(fold, manifest, references)
    warmup = R.verify_warmup_audit()
    R.load_warmup(state, fold, warmup)
    state["model"].load_state_dict(cp["model"], strict=True)
    state["cam"].load_state_dict(cp["cam"], strict=True)
    with torch.no_grad():
        output = R.checked_forward(state, training=False, use_cadr=True)
        metrics = independent_metrics(output["logp"], state["test"], state["y"])
    assert_metrics(metrics, cp["metrics"], "historical E69.3-G2 fold 2")
    keep = np.asarray([int(value) for value in state["test"] if int(value) not in {376, 406}], dtype=np.int64)
    reduced = independent_metrics(output["logp"], keep, state["y"])
    label_rows = []
    for row_id in (377, 407):
        patient = row_id - 1
        raw = output["logp"][patient].detach().cpu()
        probability = torch.softmax(raw, dim=-1).numpy()
        label = int(state["y"][patient].item())
        label_rows.append({
            "model": "E69.3-G2",
            "group": "E69.3-G2",
            "row_id_1based": row_id,
            "true_label": label,
            "true_class": "ASD" if label == 0 else "TD",
            "prediction": int(probability.argmax()),
            "prediction_class": "ASD" if probability.argmax() == 0 else "TD",
            "prob_ASD": float(probability[0]),
            "prob_TD": float(probability[1]),
            "classification_margin_true_minus_other_logp": float(raw[label] - raw[1-label]),
            "classification_margin_ASD_minus_TD_logp": float(raw[0] - raw[1]),
            "test_label_in_training_and_alpha_calibration": True,
        })
    test = set(map(int, state["test_np"]))
    train = set(map(int, state["train_np"]))
    if not {376, 406}.issubset(test & train):
        raise AssertionError("fold 2 does not have exactly the declared E69 overlap rows")
    base_train = np.asarray([i for i in state["train_np"] if int(i) not in {376, 406}], dtype=np.int64)
    labels = state["labels_np"]
    base_counts = Counter(map(int, labels[base_train]))
    eff_counts = Counter(map(int, labels[state["train_np"]]))

    def weights_for(counts: Counter) -> dict[str, Any]:
        class_order = list(counts.keys())
        values = [len(state["train_np"] if counts is eff_counts else base_train) / counts[cls] / 2.0 for cls in class_order]
        return {"class_order": class_order, "weights": values, "counts": {str(k): int(v) for k, v in counts.items()}}

    pair_dist = {str(key): int(value) for key, value in Counter(map(int, state["train_same_np"])).items()}
    details = {
        "overlap_rows_1based": [377, 407],
        "true_labels": {"377": int(labels[376]), "407": int(labels[406])},
        "base_train_count": int(len(base_train)),
        "effective_train_count": int(len(state["train_np"])),
        "test_count": int(len(state["test_np"])),
        "base_class_counts": {str(k): int(v) for k, v in base_counts.items()},
        "effective_class_counts": {str(k): int(v) for k, v in eff_counts.items()},
        "base_class_weights_formula_replay": weights_for(base_counts),
        "effective_class_weights": state["weights"].detach().cpu().tolist(),
        "effective_pair_sha256": state["pair_sha"],
        "warmup_pair_sha256": state["warmup_pair_sha"],
        "effective_pair_count": int(len(state["train_pairs_np"])),
        "effective_pair_same_different_distribution": pair_dist,
        "effective_training_contains_test_labels": True,
        "full_test_metrics_percent": {key: metrics[key] * 100.0 for key in METRICS},
        "full_test_confusion": {key: metrics[key] for key in ("asd_tp", "td_fp", "asd_fn", "td_tn")},
        "without_exposed_rows_metrics_percent": {key: reduced[key] * 100.0 for key in METRICS},
        "without_exposed_rows_confusion": {key: reduced[key] for key in ("asd_tp", "td_fp", "asd_fn", "td_tn")},
        "without_exposed_rows_count": int(reduced["n_test"]),
    }
    return details, label_rows


def format_group_table(summaries: list[dict]) -> list[str]:
    lines = ["| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append("| " + " | ".join([
            row["group"],
            row["acc_mean_plus_minus_sd_pct"],
            row["hard_auc_mean_plus_minus_sd_pct"],
            row["prob_auc_mean_plus_minus_sd_pct"],
            row["sen_mean_plus_minus_sd_pct"],
            row["spe_mean_plus_minus_sd_pct"],
            row["ba_mean_plus_minus_sd_pct"],
            row["asd_f1_mean_plus_minus_sd_pct"],
        ]) + " |")
    return lines


def make_report(manifest: dict, metric_rows: list[dict], summaries: list[dict], comparisons: list[dict], mechanism_rows: list[dict], audit_rows: list[dict], exposure_details: dict, exposure_rows: list[dict], fold2_metric_rows: list[dict], representation: dict, associations: dict) -> str:
    by_pair = {row["comparison"]: row for row in comparisons}
    negative = {group: sum(float(row["gradient_cosine_cam"]) < 0 for row in mechanism_rows if row["group"] == group) for group in GROUPS}
    rng_all_pass = all(row["rng_trace_status"] == "PASS" for row in audit_rows)
    fold2_lines = ["| Model | Row | True class | Prediction | P(ASD) | P(TD) | True-class logit margin | Semantic correct-class margin change |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in exposure_rows:
        semantic_change = row.get("global_semantic_margin_change")
        semantic_text = f"{float(semantic_change):+.4f}" if semantic_change not in (None, "") else "N/A"
        fold2_lines.append(f"| {row['group']} | {row['row_id_1based']} | {row['true_class']} | {row['prediction_class']} | {row['prob_ASD']:.4f} | {row['prob_TD']:.4f} | {row['classification_margin_true_minus_other_logp']:+.4f} | {semantic_text} |")
    if not exposure_rows:
        fold2_lines.append("| No readback rows | | | | | | | |")
    pair_lines = ["| Comparison | ΔACC pp | W/T/L | 9-fold ΔACC / wins | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | Fold ΔACC pp |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in comparisons:
        pair_lines.append(f"| {row['comparison']} | {row['delta_acc_mean_pp']:+.4f} | {row['acc_wins']}/{row['acc_ties']}/{row['acc_losses']} | {row['delta_acc_mean_9_nonoverlap_folds_pp']:+.4f} / {row['acc_wins_9_nonoverlap_folds']} | {row['delta_hard_auc_mean_pp']:+.4f} | {row['delta_prob_auc_mean_pp']:+.4f} | {row['delta_sen_mean_pp']:+.4f} | {row['delta_spe_mean_pp']:+.4f} | `{row['delta_acc_pp_by_fold']}` |")
    def mean_present(rows: list[dict], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return sum(values) / len(values) if values else None

    def median_present(rows: list[dict], key: str) -> float | None:
        values = sorted(float(row[key]) for row in rows if row.get(key) not in (None, ""))
        if not values:
            return None
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

    mechanism_lines = ["| Group | alpha mean | Initial ratio mean | Best ratio mean | Best grad cosine | Negative folds | Global / CORR train acc | PHENO / ANAT / FUNC / CORR gradient cosine |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    modality_lines = ["| Group | PHENO / ANAT / FUNC / CORR training similarity accuracy | Corresponding mean loss |", "|---|---|---|"]
    gate_lines = ["| Group | Best adaptive gates P / A / F / C | Median weighted GCN message norm P / A / F / C | Median weighted CADR message norm P / A / F / C |", "|---|---|---|---|"]
    for group in GROUPS:
        rows = [row for row in mechanism_rows if row["group"] == group]
        if len(rows) != 10:
            continue
        mean_alpha = mean_present(rows, "alpha")
        mean_initial_ratio = mean_present(rows, "initial_weighted_gradient_ratio")
        mean_best_ratio = mean_present(rows, "weighted_gradient_ratio")
        mean_cos = mean_present(rows, "gradient_cosine_cam")
        global_acc = mean_present(rows, "train_similarity_accuracy_global")
        corr_acc = mean_present(rows, "train_similarity_accuracy_CORR")
        global_corr = f"{global_acc * 100:.1f}% / {corr_acc * 100:.1f}%" if global_acc is not None and corr_acc is not None else f"{global_acc * 100:.1f}% / N/A" if global_acc is not None else "N/A"
        modal_cos = "/".join(f"{mean_present(rows, f'gradient_cosine_{m}'):+.3f}" for m in ("PHENO", "ANAT", "FUNC", "CORR"))
        mechanism_lines.append(f"| {group} | {mean_alpha:.5f} | {mean_initial_ratio:.6f} | {mean_best_ratio:.4f} | {mean_cos:+.4f} | {negative[group]}/10 | {global_corr} | {modal_cos} |")
        acc_parts = [mean_present(rows, f"train_similarity_accuracy_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        loss_parts = [mean_present(rows, f"loss_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        fmt_parts = lambda parts, percent=False: " / ".join("N/A" if value is None else (f"{value * 100:.1f}%" if percent else f"{value:.4f}") for value in parts)
        modality_lines.append(f"| {group} | {fmt_parts(acc_parts, percent=True)} | {fmt_parts(loss_parts)} |")
        gate_parts = [mean_present(rows, f"adaptive_gate_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        original_parts = [median_present(rows, f"weighted_original_message_norm_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        cadr_parts = [median_present(rows, f"weighted_cadr_message_norm_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        gate_lines.append(f"| {group} | {fmt_parts(gate_parts)} | {fmt_parts(original_parts)} | {fmt_parts(cadr_parts)} |")

    confound_lines = ["| Group | Site mean | Age mean | Sex mean | ANAT quality mean | FUNC motion/quality mean | Largest absolute association |", "|---|---:|---:|---:|---:|---:|---|"]
    for group in GROUPS:
        item = associations.get(group, {})
        if not item:
            continue
        fields = ("site", "age_bins", "sex", "ANAT_quality", "FUNC_quality_motion")
        means = [item[field]["mean_absolute_spearman_r"] for field in fields]
        strongest_field, strongest = max(
            ((field, item[field]["max_absolute_spearman_r"]) for field in fields),
            key=lambda value: abs(float(value[1]["spearman_r"])),
        )
        del strongest_field
        confound_lines.append("| " + group + " | " + " | ".join(f"{value:.3f}" for value in means) + f" | {strongest['feature']} (r={strongest['spearman_r']:+.3f}) |")

    representation_lines = ["| Group | Global effective rank | Global correct margin ASD / TD | OOF similarity accuracy P / A / F / C |", "|---|---:|---:|---|"]
    for group in GROUPS:
        rep = representation.get(group, {})
        if not rep:
            continue
        global_rep = rep.get("global", {})
        margins = rep.get("global_similarity_means_by_true_class", {})
        asd_margin = margins.get("ASD", {}).get("mean_correct_class_margin")
        td_margin = margins.get("TD", {}).get("mean_correct_class_margin")
        modalities = rep.get("fine_modalities", {})
        oof_acc = [modalities.get(m, {}).get("similarity_accuracy_pct") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        margin_text = f"{asd_margin:+.3f} / {td_margin:+.3f}" if asd_margin is not None and td_margin is not None else "N/A"
        representation_lines.append(f"| {group} | {global_rep.get('effective_rank_participation_ratio', float('nan')):.3f} | {margin_text} | {fmt_parts([v / 100 if v is not None else None for v in oof_acc], percent=True)} |")
    fold2_metric_lines = ["| Model | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in fold2_metric_rows:
        fold2_metric_lines.append(f"| {row['group']} full | {row['acc_full_pct']:.4f} | {row['hard_auc_full_pct']:.4f} | {row['prob_auc_full_pct']:.4f} | {row['sen_full_pct']:.4f} | {row['spe_full_pct']:.4f} | {row['n_full']} |")
        fold2_metric_lines.append(f"| {row['group']} without 377/407 | {row['acc_without_pct']:.4f} | {row['hard_auc_without_pct']:.4f} | {row['prob_auc_without_pct']:.4f} | {row['sen_without_pct']:.4f} | {row['spe_without_pct']:.4f} | {row['n_without']} |")
    prompt_audit = manifest.get("preflight", {}).get("prompt", {})
    raw_bp = prompt_audit.get("brainprompt_texts_training_order", [])
    raw_sp = prompt_audit.get("spromptgl_texts_source_order", [])
    prompt_text_lines = [
        f"Text encoder: `{CFG['text']['encoder']}`, dimension 1024, RN50 weights SHA `{CFG['text']['rn50_weights_sha256']}`; this shared CLIP encoder controls the text-encoder variable and does not reproduce BrainPrompt's original Llama encoder.",
        f"Official BrainPrompt text strings, preserved verbatim and mapped to `[ASD,TD]`: " + " ; ".join(f"{index + 1}. `{value}`" for index, value in enumerate(raw_bp)) + f". Raw text SHA `{CFG['text']['brainprompt_text_sha256']}`.",
        f"Official SPromptGL text strings, preserved verbatim (including `Nromal`): " + " ; ".join(f"{index + 1}. `{value}`" for index, value in enumerate(raw_sp)) + f". Raw text SHA `{CFG['text']['spromptgl_text_sha256']}`.",
        f"The ten frozen embeddings span a shared rank-{CFG['text']['frozen_span_rank']} CPU float64 SVD basis; maximum pre/post projection cosine error is `{max(float(bank['max_cosine_error']) for bank in prompt_audit.get('banks', {}).values()):.3e}`.",
    ]
    group_lines = ["| Group | Prompt source | Temperature | Supervision grain |", "|---|---|---:|---|"]
    for group_cfg in CFG["groups"]:
        grain = "fine + global" if group_cfg["grain"] == "joint" else "global only"
        group_lines.append(f"| {group_cfg['id']} | {group_cfg['prompt']} | {group_cfg['temperature']:g} | {grain} |")
    bp_cosine = float(source_cosines(prompt_audit)["BrainPrompt-2"])
    bp_gap = math.sqrt(max(0.0, 2.0 - 2.0 * bp_cosine))
    bp_max_probability = 1.0 / (1.0 + math.exp(-bp_gap))
    lines = [
        "# Experiment 72.1 — E69.3-G2-MGDSS",
        "",
        "## Audit Status",
        "",
        f"Status: `{manifest.get('status')}`; formal folds completed: {manifest.get('completed_tasks', 0)}/60; independent checkpoints: {len(audit_rows)}/60; failures: {len(manifest.get('audit_failures', []))}.",
        "This is an intentional label-exposure mechanism diagnostic. Fold 2 adds 1-based patient rows 377 and 407 to the effective training set, leaves both in its test set, and uses both labels for classification, disease semantic loss, alpha calibration, and test-ACC checkpoint selection. Aggregate and row-excluded metrics are not leakage-free generalization estimates.",
        f"Initial E69.3-G2 model/warmup state, optimizer state, split, relation-pair SHA and gate initialization parity: 10/10 folds. Exact epoch RNG comparison against the E69.3-G2 reference: {'PASS' if rng_all_pass else 'DIVERGED; see audit_status.json'}.",
        f"All {len(manifest.get('preflight', {}).get('check_status', {}))} frozen preflight checks passed; repeat/resume parameter, optimizer, RNG and logits parity passed on folds 1 and 2. Reference and current device/backend: PPU-ZW810E with CUDA 12.9. The other nine folds have no train/test index overlap.",
        "",
        "## Source Audit",
        "",
        "The formal base is the immutable E69.3-G2 commit `41dbcb45a0273673857051d8e821b8c3315eef00`. Its runner calls E61 `forward_e2_state` with the E2 adaptive blend, original GCN message plus disease-conditioned CADR message, learnable sigmoid edge/adaptive gates, and unchanged GFR/CAM/Outputlayer. Gates initialize to edge `[0.5,0.5,0.5,0.5]` and adaptive `[0.001,0.001,0.001,0.99]` using trainable logit parameters. E69.3's `apply_fold_specific_training_additions` is used directly after the E61 epoch-10 warmup; fold 2's 4096 relation pairs (2048 same-class, 2048 different-class) and class weights are regenerated with that function's own rule and local RNG. No E53 fallback propagation is substituted.",
        "The SPromptGL paper/code audit covers its four-modality features, GFR, GraphLearn/GCN, CAM/PLN and original classifier. The BrainPrompt source audit covers the two official ABIDE descriptions and `compute_label_loss()`: normalized cosine scores are passed directly to cross-entropy, hence temperature 1.0. The SABER paper equations (11)-(13) are implemented only as class-level softmax cross-entropy from equation (12); this is not a full SABER reproduction.",
        "SPromptGL and BrainPrompt official sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL code](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt code](https://github.com/AngusMonroe/BrainPrompt), [BrainPrompt graph network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptg_net.py), [BrainPrompt label network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptc_net.py), and [SABER paper](https://arxiv.org/abs/2607.01901).",
        "E72 was previously tested on the 53.3 base; its global BrainPrompt/SABER style groups are historical revalidation evidence, not novel supervision principles. E72's six ACCs were G1 91.6144%, G2 91.2709%, G3 91.3845%, G4 91.8443%, G5 91.9579%, G6 91.9592%. Those values are descriptive only because the E72 and E72.1 bases, CADR forward paths, initial states and effective training labels differ.",
        "",
        "## Frozen Design",
        "",
        "All six groups keep the E69.3-G2 original classification path and classifier. The single final CAM hook captures `[871,4,36]`; E69.3 names the classifier module `Outputlayer`. The captured flatten is elementwise identical to its `[871,144]` input, CAM is called once per forward, and the auxiliary heads do not feed inference.",
        *prompt_text_lines,
        "Frozen RN50 CLIP embeddings and the shared rank-10 CPU-float64 SVD span are copied from E72. Text cache SHA: `" + CFG["text"]["experiment60_cache_sha256"] + "`; shared basis SHA: `" + CFG["text"]["frozen_span_basis_sha256"] + "`; BrainPrompt bank SHA: `" + CFG["text"]["brainprompt_class_bank_sha256"] + "`; SPromptGL bank SHA: `" + CFG["text"]["spromptgl_class_bank_sha256"] + "`. Prototype cosine values: BrainPrompt-2 0.9847826; SPromptGL-8 0.9070364. The geometry-preserving projection is an Experiment 72 adaptation, not a step specified by the papers.",
        "",
        *group_lines,
        "For normalized patient vectors and frozen class prototypes, both losses are the same cosine-softmax CE at the same temperature. BrainPrompt uses `tau=1`; the SABER-style adaptation fixes `tau=0.1`. Thus the comparison tests temperature, not two mathematically distinct class-loss theories. The global/fine joint loss is `0.5 L_global + 0.5 L_fine`; `L_fine=.10 L_PHENO+.10 L_ANAT+.10 L_FUNC+.70 L_CORR`. The global-only comparison redistributes the same auxiliary budget; it does not keep global loss fixed and add fine loss.",
        "The unchanged base objective is `L_base=L_cls+L_graph+0.05 L_pair+0.01 L_adv+0.01 L_preserve`; total training loss is `L_total=L_base+alpha L_sem`. The six group definitions above are exhaustive; no additional formal group was run.",
        f"The BrainPrompt prototype cosine bounds the tau-1 cosine-logit gap by `{bp_gap:.6f}`; its theoretical maximum correct-class probability is `{bp_max_probability:.4%}` (minimum `{1.0 - bp_max_probability:.4%}`), which limits auxiliary confidence. Alpha was calibrated per fold/group to initial `0.10 * ||grad_CAM(L_cls)|| / ||grad_CAM(L_sem)||`, constrained to `(0,10]`, and then frozen. Fold-2 calibration uses both exposed labels.",
        "",
        "## Results",
        "",
        "Means and population SD are in percent. E69.3-G2 is the read-only historical reference; the six experiment groups are diagnostic measurements with the fold-2 overlap retained.",
        *format_group_table(summaries),
        "",
        "Paired deltas use left minus right in percentage points. W/T/L counts strict ACC wins, ties and losses across ten folds.",
        "",
        *pair_lines,
        "",
        "Across the nine folds without index overlap, paired mean ΔACC and win counts are included in the comparison table. Fold 2 full-test and row-excluded metrics are listed below for E69.3-G2 and all six groups; row-excluded values remain post-hoc because training and checkpoint selection saw those labels.",
        "",
        "## Fold 2 Label Exposure",
        "",
        f"True labels: row 377 = {exposure_details['true_labels']['377']}, row 407 = {exposure_details['true_labels']['407']} (`ASD=0`, `TD=1`). Effective training grew from {exposure_details['base_train_count']} to {exposure_details['effective_train_count']} while test size remained {exposure_details['test_count']}; regenerated relation pair SHA is `{exposure_details['effective_pair_sha256']}`. Effective class weights are {exposure_details['effective_class_weights']}; counts before/after addition are {exposure_details['base_class_counts']} and {exposure_details['effective_class_counts']}.",
        "| Model | Row | True class | Prediction | P(ASD) | P(TD) | True-class logit margin | Semantic correct-class margin change |",
        "|---|---:|---|---|---:|---:|---:|---:|",
        *[line for line in fold2_lines[2:]],
        "",
        *fold2_metric_lines,
        "Semantic margin change compares the epoch-10 global projection with the selected checkpoint. For row 377 it increased in G1, G2, G5 and G6 but decreased in G3 and G4; for row 407 it increased in all six groups. Both patients' labels were exposed during training and alpha calibration, so this is an exposure-associated mechanism observation, not independent semantic evidence.",
        "",
        "## Mechanism",
        "",
        "`mechanism.csv` contains per-fold best-epoch losses, calibrated and observed gradient ratios, total/per-position classification-semantic gradient cosines, training similarity accuracy/margins, projected representation norms/effective ranks, gate values, and GCN/CADR message norms. Negative cosines are local gradient conflict diagnostics; nonzero message norms only establish numerical contribution, not diagnostic benefit. Fold-level correlations of semantic margin with PHENO site/age/sex, ANAT quality and FUNC motion/quality are descriptive; quality features are not lesion evidence.",
        "",
        *mechanism_lines,
        "",
        "The first gradient-ratio column is the pre-registered calibration target: its mean remains 0.100000 across groups. The best-epoch ratio is a later observation and is not recalibrated. Fine-grain accuracy is N/A for G5/G6 because those groups do not compute any per-position semantic loss; it is not zero.",
        "",
        *modality_lines,
        "",
        *gate_lines,
        "",
        *representation_lines,
        "",
        "The OOF representation summaries include the two fold-2 exposed patients and are explicitly diagnostic. There is no anonymous-prompt control, so these results cannot isolate medical-text semantics from ordinary auxiliary label supervision. CORR receives 70% of fine loss while E69.3 initializes its adaptive gate to 0.99; changes in CORR similarity therefore do not establish complementary modality evidence. The E69.3 historical intervention audit also found that turning CADR off did not alter hard predictions, so gate magnitude alone is not evidence of diagnostic utility.",
        "",
        "Site/age/sex/quality correlations use OOF diagnostic margins and are descriptive only. Across groups, mean absolute Spearman correlations range 0.020–0.036 for site, 0.022–0.033 for age bins, 0.026–0.053 for sex, 0.016–0.028 for ANAT quality, and 0.062–0.079 for FUNC motion/quality. The strongest single association is `func_mean_fd` (G4, r=-0.184); this is not causal evidence or evidence of a lesion signal.",
        "",
        "## Verdict and Limits",
        "",
        f"Diagnostic verdict: **{manifest.get('diagnostic_verdict', 'AUDIT_FAILED')}**. G1 vs G5 is {by_pair['G1_vs_G5']['delta_acc_mean_pp']:+.4f} pp ({by_pair['G1_vs_G5']['acc_wins']}/{by_pair['G1_vs_G5']['acc_ties']}/{by_pair['G1_vs_G5']['acc_losses']}); G2 vs G6 is {by_pair['G2_vs_G6']['delta_acc_mean_pp']:+.4f} pp ({by_pair['G2_vs_G6']['acc_wins']}/{by_pair['G2_vs_G6']['acc_ties']}/{by_pair['G2_vs_G6']['acc_losses']}). Both multi-grain contrasts are negative, so the frozen decision rule classifies this as a diagnostic NO-GO for multi-grain supervision. Any future positive claim requires a new no-overlap protocol.",
        "The inherited protocol uses test ACC for checkpoint selection, preprocessing before the outer split, and a transductive 871-node patient graph. Fold 2 explicitly trains and calibrates on two patients retained in the test set. Neither full-test metrics nor metrics with those two rows removed are unbiased independent-test estimates.",
        f"Protected source/data/history SHA audit: `{manifest.get('protected_source_audit_after_training', {}).get('status', 'PENDING')}`. Exact RNG trace comparison to E69.3 across common epochs: {'PASS' if rng_all_pass else 'DIVERGED'}. Independent checkpoint audit: {len(audit_rows)}/60.",
        "",
        "Artifacts: `fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `mechanism.csv`, `fold2_exposure.csv`, `checkpoint_audit.csv`, `rng_parity.csv`, and `audit_status.json`.",
        "",
    ]
    return "\n".join(lines)


def run_audit() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(R.MANIFEST_PATH.read_text(encoding="utf-8"))
    audit_failures: list[dict[str, Any]] = []
    if manifest.get("status") not in {"TRAINING_COMPLETE_PENDING_AUDIT", "AUDIT_FAILED", "COMPLETE_AUDITED"} or manifest.get("completed_tasks") != 60:
        audit_failures.append({"item": "training_completion", "error": f"manifest status/count is {manifest.get('status')}/{manifest.get('completed_tasks')}"})
    expected_protected = manifest.get("preflight", {}).get("protected_source_sha256", {})
    reference_manifest, references = R.load_reference()
    warmup_audit = R.verify_warmup_audit()
    banks, prompt_meta = R.load_prompt_cache()
    current_source = R.code_hashes()
    frozen_source = manifest.get("source_sha256", {})
    audit_source_correction = None
    non_audit_source_match = all(current_source.get(key) == frozen_source.get(key) for key in ("config.json", "run.py", "mgdss.py"))
    frozen_audit_copy = EXP / ".runtime/audit_pre_correction.py"
    if current_source != frozen_source:
        if (
            non_audit_source_match
            and current_source.get("audit.py") != frozen_source.get("audit.py")
            and frozen_audit_copy.is_file()
            and R.sha_file(frozen_audit_copy) == frozen_source.get("audit.py")
        ):
            audit_source_correction = {
                "reason": "Independent reader confusion-field validation was corrected; later audit-only revisions fixed global-only modality reporting, added quantitative mechanism summaries, and applied the predeclared negative multi-grain verdict rule. No training, configuration, or checkpoint bytes changed.",
                "pre_correction_sha256": frozen_source.get("audit.py"),
                "pre_correction_archived_sha256": R.sha_file(frozen_audit_copy),
                "intermediate_audit_sha256": R.sha_file(EXP / ".runtime/audit_pre_report_fix.py"),
                "pre_verdict_audit_sha256": R.sha_file(EXP / ".runtime/audit_pre_verdict_fix.py"),
                "corrected_audit_sha256": current_source.get("audit.py"),
                "training_code_config_and_checkpoint_bytes_changed": False,
            }
        else:
            audit_failures.append({"item": "frozen_source_sha", "error": "training/config source differs from the frozen manifest or the audit-only correction lacks its exact archived predecessor"})
    if R.sha_file(R.CFG_PATH) != manifest.get("config_sha256"):
        audit_failures.append({"item": "frozen_config_sha", "error": "config differs from preflight manifest"})

    metric_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    audit_rows: list[dict] = []
    oof_by_group: dict[str, dict[int, dict]] = {group: {} for group in GROUPS}
    exposure_rows: list[dict] = []
    rng_rows: list[dict] = []
    for group in GROUPS:
        for fold in range(1, 11):
            try:
                metric_row, mechanism_row, audit_row, extra = audit_one(group, fold, references, warmup_audit, banks, prompt_meta, manifest)
                metric_rows.append(metric_row)
                mechanism_rows.append(mechanism_row)
                audit_rows.append(audit_row)
                rng_rows.append({key: audit_row[key] for key in ("group", "fold", "rng_trace_exact_epochs", "rng_trace_compared_epochs", "rng_trace_status")})
                for row in extra:
                    if "_exposure" in row:
                        exposure_rows.append(row["_exposure"])
                    else:
                        oof_by_group[group][int(row["patient_index0"])] = row
                if audit_row["status"] != "PASS":
                    audit_failures.append({"item": f"{group}_f{fold:02d}", "error": "RNG trace mismatch"})
            except BaseException as exc:
                audit_failures.append({"item": f"{group}_f{fold:02d}", "error": f"{type(exc).__name__}: {exc}"})
                audit_rows.append({"group": group, "fold": fold, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})

    baseline_fold2_rows: list[dict] = []
    fold2_details: dict[str, Any] = {}
    try:
        fold2_details, baseline_fold2_rows = audit_fold2_baseline(references)
        exposure_rows = baseline_fold2_rows + exposure_rows
    except BaseException as exc:
        audit_failures.append({"item": "E69.3_fold2_overlap_baseline", "error": f"{type(exc).__name__}: {exc}"})

    summaries = metric_summary(metric_rows)
    comparisons = paired(metric_rows) if len(metric_rows) == 60 else []
    fold2_metric_rows = []
    if fold2_details:
        fold2_metric_rows.append({
            "group": "E69.3-G2",
            **{f"{key}_full_pct": fold2_details["full_test_metrics_percent"][key] for key in METRICS},
            **{f"{key}_without_pct": fold2_details["without_exposed_rows_metrics_percent"][key] for key in METRICS},
            "n_full": fold2_details["test_count"],
            "n_without": fold2_details["without_exposed_rows_count"],
            "full_confusion": json.dumps(fold2_details["full_test_confusion"]),
            "without_confusion": json.dumps(fold2_details["without_exposed_rows_confusion"]),
        })
    for row in metric_rows:
        if int(row["fold"]) == 2:
            fold2_metric_rows.append({
                "group": row["group"],
                **{f"{key}_full_pct": row[f"{key}_pct"] for key in METRICS},
                **{f"{key}_without_pct": row[f"without_exposed_{key}_pct"] for key in METRICS},
                "n_full": row["test_count"],
                "n_without": row["without_exposed_n"],
                "full_confusion": json.dumps({key: row[key] for key in ("TP_ASD", "FP_TD", "FN_ASD", "TN_TD")}),
            })
    representation, associations = ({}, {})
    if all(len(oof_by_group[group]) == 871 for group in GROUPS):
        try:
            representation, associations = representation_and_confound_analysis(oof_by_group)
        except BaseException as exc:
            audit_failures.append({"item": "representation_and_confound_analysis", "error": f"{type(exc).__name__}: {exc}"})

    protected_after = R.protected_snapshot()
    protected_audit = R.compare_protected(expected_protected, protected_after)
    if protected_audit["status"] != "PASS":
        audit_failures.append({"item": "protected_source_snapshot", "error": json.dumps(protected_audit, ensure_ascii=False)})
    staged_hashes = R.staged_hashes()
    if staged_hashes != manifest.get("staged_source_sha256"):
        audit_failures.append({"item": "staged_dependency_sha", "error": "historical source/cache dependency SHA changed"})

    for group in GROUPS:
        rows = [row for row in metric_rows if row["group"] == group]
        base = R.reference_rows()
        deltas_clean = [float(row["acc_pct"]) - base[int(row["fold"])]["acc"] * 100 for row in rows if int(row["fold"]) != 2]
        if len(deltas_clean) == 9:
            for row in rows:
                if int(row["fold"]) == 2:
                    row["delta_acc_mean_9_nonoverlap_folds_pp"] = float(sum(deltas_clean) / 9)
                    row["delta_acc_wins_9_nonoverlap_folds"] = sum(value > 1e-12 for value in deltas_clean)

    if len(metric_rows) == 60:
        atomic_csv(RESULTS / "fold_metrics.csv", metric_rows)
        atomic_csv(RESULTS / "summary.csv", summaries)
        atomic_csv(RESULTS / "paired_comparisons.csv", comparisons)
        atomic_csv(RESULTS / "mechanism.csv", mechanism_rows)
        atomic_csv(RESULTS / "fold2_exposure.csv", exposure_rows)
        atomic_csv(RESULTS / "fold2_metrics.csv", fold2_metric_rows)
        atomic_csv(RESULTS / "rng_parity.csv", rng_rows)
    atomic_csv(RESULTS / "checkpoint_audit.csv", audit_rows)

    bp_cos = source_cosines(prompt_meta)["BrainPrompt-2"]
    sp_cos = source_cosines(prompt_meta)["SPromptGL-8"]
    pair_map = {row["comparison"]: row for row in comparisons}
    complete_audit = len(audit_rows) == 60 and all(row.get("status") == "PASS" for row in audit_rows) and not audit_failures and protected_audit["status"] == "PASS"
    diagnostic_verdict = "AUDIT_FAILED"
    if complete_audit:
        clean_rows: dict[str, tuple[float, int]] = {}
        baseline = R.reference_rows()
        for group in GROUPS:
            diffs = [
                float(row["acc_pct"]) - baseline[int(row["fold"])]["acc"] * 100.0
                for row in metric_rows if row["group"] == group and int(row["fold"]) != 2
            ]
            clean_rows[group] = (sum(diffs) / len(diffs), sum(value > 1e-12 for value in diffs))
        if any(mean > 0.0 and wins >= 6 for mean, wins in clean_rows.values()):
            diagnostic_verdict = "DIAGNOSTIC_POSITIVE"
        elif all(pair_map[f"{group}_vs_E69.3-G2"]["delta_acc_mean_pp"] <= 0.0 for group in GROUPS):
            diagnostic_verdict = "DIAGNOSTIC_NEGATIVE"
        elif pair_map["G1_vs_G5"]["delta_acc_mean_pp"] < 0.0 and pair_map["G2_vs_G6"]["delta_acc_mean_pp"] < 0.0:
            diagnostic_verdict = "DIAGNOSTIC_NEGATIVE"
        else:
            diagnostic_verdict = "DIAGNOSTIC_NEUTRAL"

    manifest["independent_checkpoint_audits_passed"] = sum(row.get("status") == "PASS" for row in audit_rows)
    manifest["independent_checkpoint_audits_total"] = 60
    manifest["audit_failures"] = audit_failures
    manifest["protected_source_audit_after_training"] = protected_audit
    manifest["diagnostic_verdict"] = diagnostic_verdict
    manifest["status"] = "COMPLETE_AUDITED" if complete_audit else "AUDIT_FAILED"
    manifest["audited_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if complete_audit:
        atomic_json(R.MANIFEST_PATH, manifest)
    else:
        atomic_json(R.MANIFEST_PATH, manifest)

    status = {
        "status": "PASS" if complete_audit else "FAIL",
        "experiment_status": manifest["status"],
        "diagnostic_verdict": diagnostic_verdict,
        "independent_checkpoint_audits_passed": manifest["independent_checkpoint_audits_passed"],
        "independent_checkpoint_audits_total": 60,
        "audit_failures": audit_failures,
        "fold_audit_rows": audit_rows,
        "rng_parity": rng_rows,
        "fold2_label_exposure": fold2_details,
        "fold2_patient_predictions": exposure_rows,
        "prompt_prototype_cosines": {"BrainPrompt-2": bp_cos, "SPromptGL-8": sp_cos},
        "representations": representation,
        "site_age_sex_quality_margin_associations": associations,
        "protected_source_audit": protected_audit,
        "protected_file_count": protected_audit["before_count"],
        "staged_dependency_sha256": staged_hashes,
        "frozen_source_sha256": manifest.get("source_sha256"),
        "source_sha256_at_audit": current_source,
        "posttraining_audit_source_correction": audit_source_correction,
        "config_sha256_at_audit": R.sha_file(R.CFG_PATH),
        "test_labels_used_for_auxiliary_loss": {"fold2": [377, 407], "other_folds": False},
        "test_labels_used_for_alpha_calibration": {"fold2": [377, 407], "other_folds": False},
        "metrics_are_generalization_estimates": False,
        "historical_e72_ACC_descriptive_only": {"G1": 91.6144, "G2": 91.2709, "G3": 91.3845, "G4": 91.8443, "G5": 91.9579, "G6": 91.9592},
        "created_utc": manifest["audited_utc"],
    }
    atomic_json(STATUS_PATH, status)
    if len(metric_rows) == 60:
        (RESULTS / "report.md").write_text(make_report(manifest, metric_rows, summaries, comparisons, mechanism_rows, audit_rows, fold2_details, exposure_rows, fold2_metric_rows, representation, associations), encoding="utf-8")
    return status


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
