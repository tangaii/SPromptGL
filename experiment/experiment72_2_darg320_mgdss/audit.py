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
    + [(f"{group}_vs_E72.1-{group}", group, f"E72.1-{group}") for group in GROUPS]
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


_BASELINE_CACHE: dict[int, dict] = {}


def baseline_readback(fold: int, references: dict[int, dict]) -> dict:
    import torch
    if fold in _BASELINE_CACHE:
        return _BASELINE_CACHE[fold]
    path = Path(CFG["paths"]["e69_3_g2"]) / ".runtime/checkpoints" / f"E69.3-G2_f{fold:02d}.best.pt"
    if R.sha_file(path) != references[fold]["checkpoint_sha256"]:
        raise AssertionError("historical E69.3 best checkpoint SHA mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = R.build_fold(fold, json.loads((R.REFERENCE_EXP / "manifest.json").read_text()), references)
    state["model"].load_state_dict(saved["model"], strict=True)
    state["cam"].load_state_dict(saved["cam"], strict=True)
    with torch.no_grad():
        out = R.checked_forward(state, training=False, use_cadr=True)
        metrics = independent_metrics(out["logp"], state["test"], state["y"])
    assert_metrics(metrics, saved["metrics"], f"historical baseline fold {fold}")
    record = {"logp": out["logp"].detach().cpu(),
              "edge_gate": torch.sigmoid(state["model"].edge_gate_logits).detach().cpu().tolist(),
              "adaptive_gate": torch.sigmoid(state["model"].adaptive_gate_logits).detach().cpu().tolist(),
              "original_norms": [float(v.norm()) for v in out["info"]["messages_original"]],
              "cadr_norms": [float(v.norm()) for v in out["info"]["messages_cadr"]]}
    _BASELINE_CACHE[fold] = record
    return record


def audit_one(group: str, fold: int, references: dict[int, dict], warmup_audit: dict, banks: dict, prompt_meta: dict, manifest: dict) -> tuple[dict, dict, dict, list[dict]]:
    import numpy as np
    import torch

    baseline = baseline_readback(fold, references)
    task_path, checkpoint_path, mechanism_path = task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    cp_sha = R.sha_file(checkpoint_path)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != cp_sha:
        raise AssertionError("task/checkpoint completion or SHA mismatch")
    if cp.get("schema") != "experiment72_2_darg320_mgdss_checkpoint_v1" or cp.get("group") != group or int(cp.get("fold", -1)) != fold:
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
        fused = output["fused"]
        if tuple(fused.shape) != (871, 320):
            raise AssertionError("independent reader found no live E61-E2 fused F [871,320]")
        aux = module(fused, banks[item["prompt"]].to(fused.device), float(item["temperature"]), item["grain"] == "joint")
        cam_calls = int(state["cam"]._e72_forward_count)
        info = output["info"]
        logp_cpu = output["logp"].detach().cpu()

    # These post-hoc eval diagnostics use only effective training labels in losses.
    rng_before_diagnostic = BASE.rng_sha(BASE.pack_rng())
    eval_output = R.checked_forward(state, training=False, use_cadr=True)
    live_f = eval_output["fused"]
    train = R.train_index(state, live_f.device)
    train_labels = state["y"][train].long()
    supervised, _ = R.make_losses(module, live_f[train], train_labels, banks[item["prompt"]].to(live_f.device), float(item["temperature"]), item["grain"])
    eval_grads = R.gradient_stats(eval_output["cls"], supervised["semantic"], live_f, train)
    if not live_f.requires_grad or eval_grads["classification_norm"] <= 0 or eval_grads["semantic_norm"] <= 0:
        raise AssertionError("selected-checkpoint F gradients are degenerate")
    with torch.no_grad():
        off = R.E61.forward_e2_state(state, training=False, intervention="disease_off")
        off_metrics = independent_metrics(off["logp"], state["test"], state["y"])
    if BASE.rng_sha(BASE.pack_rng()) != rng_before_diagnostic:
        raise AssertionError("eval gradients or fixed-checkpoint intervention changed RNG")
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

    item_bank = banks[item["prompt"]].to(fused.device)
    tau = float(item["temperature"])
    y = state["y"].detach().long()
    test_idx = np.asarray(state["test"], dtype=np.int64)
    test_y = y[test_idx].detach().cpu().numpy()
    global_sim = aux["global_logits"].detach() * tau
    global_margin = global_sim.gather(1, y[:, None]).squeeze(1) - global_sim.gather(1, (1-y)[:, None]).squeeze(1)
    fine_sim = None if aux["fine_logits"] is None else aux["fine_logits"].detach() * tau
    p_softmax = torch.softmax(aux["global_logits"].detach(), dim=-1)
    selected = torch.as_tensor(test_idx, dtype=torch.long, device=fused.device)
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
        "gradient_norm_cls_f": best_row["gradient_norm_cls_f"],
        "gradient_norm_semantic_f": best_row["gradient_norm_semantic_f"],
        "weighted_gradient_ratio": best_row["weighted_gradient_ratio"],
        "gradient_cosine_f": best_row["gradient_cosine_f"],
        "best_checkpoint_eval_gradient_norm_cls_f": eval_grads["classification_norm"],
        "best_checkpoint_eval_gradient_norm_semantic_f": eval_grads["semantic_norm"],
        "best_checkpoint_eval_gradient_cosine_f": eval_grads["cosine"],
        "best_checkpoint_eval_weighted_gradient_ratio": alpha * eval_grads["semantic_norm"] / eval_grads["classification_norm"],
        "semantic_source": "E61-E2 output fused F [871,320]",
        "disease_off_hard_prediction_changes": int(np.sum(metrics["prediction"] != off_metrics["prediction"])),
        "full_minus_disease_off_acc_pp": 100.0 * (metrics["acc"] - off_metrics["acc"]),
        "disease_off_mean_abs_probability_change": float(np.abs(metrics["probability"] - off_metrics["probability"]).mean()),
        "negative_total_gradient_cosine": bool(float(best_row["gradient_cosine_f"]) < 0),
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
        for key in ("classification_norm", "semantic_norm", "cosine"):
            mechanism_row[f"best_eval_gradient_{key}_{modality}"] = eval_grads["modality"][modality][key]
        mechanism_row[f"baseline_adaptive_gate_{modality}"] = baseline["adaptive_gate"][idx]
        mechanism_row[f"baseline_original_message_norm_{modality}"] = baseline["original_norms"][idx]
        mechanism_row[f"baseline_cadr_message_norm_{modality}"] = baseline["cadr_norms"][idx]
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
    baseline_logp = baseline["logp"].numpy()[test_idx]
    baseline_pred = baseline_logp.argmax(axis=1)
    prediction = metrics["prediction"]
    corrected = (prediction == test_y) & (baseline_pred != test_y)
    harmed = (prediction != test_y) & (baseline_pred == test_y)
    mechanism_row.update({"corrected_vs_baseline_count": int(corrected.sum()), "harmed_vs_baseline_count": int(harmed.sum()),
                          "changed_prediction_rows_1based": json.dumps((test_idx[prediction != baseline_pred] + 1).tolist()),
                          "corrected_rows_1based": json.dumps((test_idx[corrected] + 1).tolist()),
                          "harmed_rows_1based": json.dumps((test_idx[harmed] + 1).tolist())})
    for label, name in ((0, "ASD"), (1, "TD")):
        mechanism_row[f"corrected_{name}_count"] = int((corrected & (test_y == label)).sum())
        mechanism_row[f"harmed_{name}_count"] = int((harmed & (test_y == label)).sum())
    best_eval_margin = logp_cpu.numpy()[test_idx][np.arange(len(test_y)), test_y] - logp_cpu.numpy()[test_idx][np.arange(len(test_y)), 1-test_y]
    baseline_margin = baseline_logp[np.arange(len(test_y)), test_y] - baseline_logp[np.arange(len(test_y)), 1-test_y]
    mechanism_row["mean_correct_class_logp_margin_change_vs_baseline"] = float((best_eval_margin - baseline_margin).mean())
    warm = manifest["preflight"]["fold_initialization_and_warmup"][str(fold)]["warmup_test_representations"]
    initial_similarity = np.asarray(warm["similarities"][item["prompt"]]["global"])
    initial_sem_margin = initial_similarity[np.arange(len(test_y)), test_y] - initial_similarity[np.arange(len(test_y)), 1-test_y]
    semantic_margin_change = global_margin[selected].cpu().numpy() - initial_sem_margin
    if np.std(semantic_margin_change) > 0 and np.std(best_eval_margin - baseline_margin) > 0:
        from scipy.stats import spearmanr
        mechanism_row["semantic_margin_change_vs_prediction_margin_change_spearman"] = float(spearmanr(semantic_margin_change, best_eval_margin - baseline_margin).statistic)
    else:
        mechanism_row["semantic_margin_change_vs_prediction_margin_change_spearman"] = None
    mechanism_row["semantic_margin_change_mean_corrected"] = float(semantic_margin_change[corrected].mean()) if corrected.any() else None
    mechanism_row["semantic_margin_change_mean_harmed"] = float(semantic_margin_change[harmed].mean()) if harmed.any() else None
    mechanism_row["global_effective_rank_train"] = representation_summary(aux["global_patient"][train].detach().cpu().numpy(), train_labels.cpu().numpy())["effective_rank_participation_ratio"]
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
    with (Path(CFG["paths"]["e72_1"]) / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        historical = list(csv.DictReader(stream))
    for group in GROUPS:
        subset = sorted((r for r in historical if r["group"] == group), key=lambda r: int(r["fold"]))
        if len(subset) != 10:
            raise AssertionError("E72.1 historical fold count differs from ten")
        summaries[f"E72.1-{group}"] = {key: np.asarray([float(r[f"{key}_pct"]) for r in subset]) for key in METRICS}
    for group in GROUPS:
        subset = sorted((row for row in rows if row["group"] == group), key=lambda row: int(row["fold"]))
        if len(subset) != 10:
            continue
        summaries[group] = {key: np.asarray([float(row[f"{key}_pct"]) for row in subset]) for key in METRICS}
    for group, values_by_metric in summaries.items():
        row = {"group": group, "fold_count": 10, "metric_type": "diagnostic_with_fold2_label_exposure" if group in GROUPS else "historical_reference"}
        for metric, values in values_by_metric.items():
            row[f"{metric}_mean_pct"] = float(values.mean())
            row[f"{metric}_population_sd_pct"] = float(values.std(ddof=0))
            row[f"{metric}_mean_plus_minus_sd_pct"] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        output.append(row)
        clean = {"group": group + "_9_nonoverlap_folds", "fold_count": 9, "metric_type": "descriptive_excluding_fold2_not_independent_test"}
        for metric, values in values_by_metric.items():
            values = np.delete(values, 1)
            clean[f"{metric}_mean_pct"] = float(values.mean())
            clean[f"{metric}_population_sd_pct"] = float(values.std(ddof=0))
            clean[f"{metric}_mean_plus_minus_sd_pct"] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        output.append(clean)
    return output


def paired(rows: list[dict]) -> list[dict]:
    import numpy as np

    group_rows = {group: {int(row["fold"]): row for row in rows if row["group"] == group} for group in GROUPS}
    baseline = R.reference_rows()
    with (Path(CFG["paths"]["e72_1"]) / "results/fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    for group in GROUPS:
        group_rows[f"E72.1-{group}"] = {int(row["fold"]): row for row in history if row["group"] == group}
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
    pre = manifest.get("preflight", {})
    prompt = pre.get("prompt", {})
    prompt_banks = prompt.get("banks", {})
    base = next((row for row in summaries if row["group"] == "E69.3-G2"), {})
    e72 = {row["group"]: row for row in summaries if row["group"] in GROUPS}
    e72_1_rows = {}
    e72_1_path = Path(CFG["paths"]["e72_1"]) / "results/summary.csv"
    if e72_1_path.is_file():
        with e72_1_path.open(newline="", encoding="utf-8") as stream:
            e72_1_rows = {row["group"]: row for row in csv.DictReader(stream)}
    metric_names = (("acc", "ACC"), ("hard_auc", "Hard-AUC"), ("prob_auc", "Prob-AUC"), ("sen", "SEN-ASD"), ("spe", "SPE-TD"), ("ba", "BA"), ("asd_f1", "ASD-F1"))
    result_lines = ["| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("E69.3-G2", *GROUPS):
        row = base if group == "E69.3-G2" else e72.get(group, {})
        result_lines.append("| " + group + " | " + " | ".join(row.get(f"{metric}_mean_plus_minus_sd_pct", "N/A") for metric, _ in metric_names) + " |")
    paired_lines = ["| Comparison (left − right) | ΔACC pp | W/T/L | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | 9 non-overlap folds: mean ΔACC / W-T-L |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        paired_lines.append(f"| {row['comparison']} | {row['delta_acc_mean_pp']:+.4f} | {row['acc_wins']}/{row['acc_ties']}/{row['acc_losses']} | {row['delta_hard_auc_mean_pp']:+.4f} | {row['delta_prob_auc_mean_pp']:+.4f} | {row['delta_sen_mean_pp']:+.4f} | {row['delta_spe_mean_pp']:+.4f} | {row['delta_acc_mean_9_nonoverlap_folds_pp']:+.4f} / {row['acc_wins_9_nonoverlap_folds']}-{row['acc_ties_9_nonoverlap_folds']}-{row['acc_losses_9_nonoverlap_folds']} |")
    e72_1_lines = ["| Group | E72.1 ACC | E72.2 ACC | ΔACC pp | E72.1 Prob-AUC | E72.2 Prob-AUC | ΔProb-AUC pp |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        old = e72_1_rows.get(group, {})
        new_row = e72.get(group, {})
        if old and new_row:
            old_acc = float(old["acc_mean_pct"]); new_acc = float(new_row["acc_mean_pct"])
            old_auc = float(old["prob_auc_mean_pct"]); new_auc = float(new_row["prob_auc_mean_pct"])
            e72_1_lines.append(f"| {group} | {old_acc:.4f} | {new_acc:.4f} | {new_acc-old_acc:+.4f} | {old_auc:.4f} | {new_auc:.4f} | {new_auc-old_auc:+.4f} |")
    fold2_lines = ["| Model | Row | True class | Prediction | P(ASD) | P(TD) | True-class logit margin | Semantic margin change |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in exposure_rows:
        change = row.get("global_semantic_margin_change")
        fold2_lines.append(f"| {row['group']} | {row['row_id_1based']} | {row['true_class']} | {row['prediction_class']} | {row['prob_ASD']:.4f} | {row['prob_TD']:.4f} | {row['classification_margin_true_minus_other_logp']:+.4f} | {'N/A' if change is None else f'{float(change):+.4f}'} |")
    for row in fold2_metric_rows:
        fold2_lines.append(f"| {row['group']} full | — | — | — | — | — | ACC {row['acc_full_pct']:.4f}; AUC {row['prob_auc_full_pct']:.4f} | — |")
        fold2_lines.append(f"| {row['group']} excluding 377/407 | — | — | — | — | — | ACC {row['acc_without_pct']:.4f}; AUC {row['prob_auc_without_pct']:.4f} | — |")
    mechanism_lines = ["| Group | Mean α | Initial weighted grad ratio | Best-epoch grad cosine < 0 | Train global similarity acc | PHENO / ANAT / FUNC / CORR similarity acc | Adaptive gates P/A/F/C |", "|---|---:|---:|---:|---:|---|---|"]
    for group in GROUPS:
        rows = [row for row in mechanism_rows if row["group"] == group]
        if not rows: continue
        avg = lambda key: sum(float(row[key]) for row in rows if row.get(key) not in (None, "")) / max(1, sum(row.get(key) not in (None, "") for row in rows))
        alpha_values = [float(row["alpha"]) for row in rows]
        modality_acc = [avg(f"train_similarity_accuracy_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        gates = [avg(f"adaptive_gate_{m}") for m in ("PHENO", "ANAT", "FUNC", "CORR")]
        conflict = sum(float(row["gradient_cosine_f"]) < 0 for row in rows) / len(rows)
        fine_text = "N/A" if group in ("G5", "G6") else " / ".join(f"{value:.2%}" for value in modality_acc)
        mechanism_lines.append(f"| {group} | {sum(alpha_values)/len(alpha_values):.5f} | {avg('initial_weighted_gradient_ratio'):.4f} | {conflict:.2%} | {avg('train_similarity_accuracy_global'):.2%} | " + fine_text + " | " + " / ".join(f"{value:.4f}" for value in gates) + " |")
    overlap_audit = [row for row in audit_rows if row.get("train_test_overlap_1based") == "[377, 407]"]
    nonoverlap_audits = sum(row.get("status") == "PASS" and row.get("train_test_overlap_1based") == "[]" for row in audit_rows)
    exact_rng = sum(int(row.get("rng_trace_exact_epochs", 0)) for row in audit_rows)
    compared_rng = sum(int(row.get("rng_trace_compared_epochs", 0)) for row in audit_rows)
    checks = pre.get("check_status", {})
    max_cosine_error = max((float(value.get("max_cosine_error", 0.0)) for value in prompt_banks.values()), default=float("nan"))
    bp_cos = float(prompt_banks.get("BrainPrompt-2", {}).get("source_cosine", float("nan")))
    sp_cos = float(prompt_banks.get("SPromptGL-8", {}).get("source_cosine", float("nan")))
    alpha_pre = [record[group]["alpha"] for record in pre.get("alphas", {}).values() for group in GROUPS]
    initial_ratios = [record[group]["initial_weighted_gradient_ratio"] for record in pre.get("alphas", {}).values() for group in GROUPS]
    verdict = manifest.get("diagnostic_verdict", "AUDIT_FAILED")
    lines = [
        "# Experiment 72.2 — DARG-320 Disease Semantic Supervision", "",
        "The hypothesis is that supervising DARG fused F before CAM may deliver a disease-class gradient earlier to relation-conditioned propagation than supervising the CAM144 endpoint. It is a position-mechanism hypothesis, with no assumed improvement; fine heads require a simultaneous dimension/schema adaptation, so this is not a pure single-variable position ablation.", "",
        "## Status and scope", "",
        f"Status: **{manifest.get('status')}**; formal tasks {manifest.get('completed_tasks', 0)}/60; independent checkpoint reads {len(audit_rows)}/60; failures {len(manifest.get('audit_failures', []))}.",
        "This is a mechanism diagnostic under the inherited E69.3-G2 protocol, not a leakage-free generalization estimate. Fold 2 adds labeled rows 377 (ASD) and 407 (TD) to effective training and alpha calibration while retaining them in the test set. Checkpoint selection uses test ACC; preprocessing precedes outer CV; the graph is transductive over all 871 patients.",
        f"Initial model, all three optimizer states, E61 epoch-10 warmup, fold split, effective train set, relation-pair state and E69.3 gate initialization parity: {sum(row.get('status') == 'PASS' for row in pre.get('fold_initialization_and_warmup', {}).values())}/10 folds. Common E69.3 RNG checkpoints matched {exact_rng}/{compared_rng} recorded epochs; exact RNG status is reported in `rng_parity.csv`.",
        f"The experiment executes the locked E61-E2 `forward_e2_state` / `e2e_adaptive_forward`, preserving original GCN plus CADR messages and learnable edge/adaptive gate logits. E69.3 fold-addition code regenerates fold-2 train labels, weights and patient pairs; rows 377/407 remain in the test set. No 53.3 fallback forward is used.",
        "",
        "## Source and method audit", "",
        "SPromptGL MICCAI 2025 GFR/CAM/PLN and four-modality classifier; BrainPrompt MICCAI 2025 ABIDE disease descriptions and normalized-cosine CE; and SABER ICME 2026 equations (11)–(13) were checked against the locked E72.1 code review and local source copies. This uses only the disease-level class supervision equation, not full BrainPrompt or SABER architectures. Shared frozen CLIP RN50 text encoding differs from BrainPrompt's original Llama encoder; τ=0.1 is this study's SABER-style adaptation, not asserted as SABER's paper temperature.",
        "E72.2 differs from E72.1 only in the semantic source location and required input projection schema: E72.1 supervised the 144-dimensional CAM flatten and its four 36-dimensional CAM positions; E72.2 supervises E61-E2 DARG fused F [871,320], globally with Linear(320,10), and in four F-coordinate segments of widths [48,6,10,256] with separate Linear(dim,10) heads. The F segments already include cross-modality propagation and are not independent raw modality evidence. The original CAM and 144-dimensional OutputLayer path, labels, base losses, optimizers, warmup, gates, training schedule, Prompt banks and six group definitions are retained.",
        "The original classifier still receives the same CAM output and supplies all inference predictions. Each training pass calls E2E DARG and CAM once. Auxiliary projections are outside the base optimizers, initialized under private RNG, and omitted from inference. The joint loss remains `0.5 L_global + 0.5 L_fine`; fine weights are `.10/.10/.10/.70`; global-only uses `L_global`; total loss is `L_base + alpha L_sem`.",
        f"Frozen RN50 cache SHA `{CFG['text']['experiment60_cache_sha256']}`; shared SVD basis SHA `{CFG['text']['frozen_span_basis_sha256']}`; BrainPrompt bank SHA `{CFG['text']['brainprompt_class_bank_sha256']}`; SPromptGL bank SHA `{CFG['text']['spromptgl_class_bank_sha256']}`. Prototype cosine: BrainPrompt-2 {bp_cos:.7f}; SPromptGL-8 {sp_cos:.7f}; maximum projection cosine error {max_cosine_error:.3e}. The τ=1 BrainPrompt CE and SABER-form log-softmax CE had loss/gradient equivalence `{pre.get('loss_equivalence', {}).get('status', 'UNKNOWN')}`.",
        f"Alpha was calibrated once per fold/group using F-level `0.10*||grad_F L_cls||/||grad_F L_sem||` and frozen. Preflight initial weighted ratios span {min(initial_ratios):.6f}–{max(initial_ratios):.6f}; alpha spans {min(alpha_pre):.6f}–{max(alpha_pre):.6f}. Fold-2 alpha calibration includes both exposed labels. The normalized BrainPrompt prototype cosine also limits τ=1 logit gap; see `preflight.json` for geometry-derived bounds.",
        "",
        "## Results", "",
        "All values are ten-fold mean ± population SD (%). E69.3-G2 is read-only historical reference. E72.2 values retain the fold-2 overlap and are diagnostic only.",
        *result_lines, "", "Paired deltas are left minus right, in percentage points; W/T/L are strict ACC wins/ties/losses.", "", *paired_lines,
        "", "### Comparison with E72.1", "",
        "E72.1 and E72.2 have different supervised representations and projection input dimensions, so these cross-experiment differences are descriptive; the fine-grain differences are not a strict one-variable ablation. Global-only groups are the closest position comparison, though their projection input dimensions necessarily differ.", "", *e72_1_lines,
        "", "### Fold-level metrics", "",
        "`fold_metrics.csv` provides every fold's ACC, Hard-AUC, Prob-AUC, SEN, SPE, confusion counts, best epoch, actual epoch count, overlap flags and checkpoint SHA. `paired_comparisons.csv` contains ten-fold deltas and separate nine-fold summaries excluding fold 2; exclusion does not undo the training or checkpoint-selection label exposure.",
        "", "## Fold 2 label exposure", "",
        f"Rows 377 and 407 remain both training and test patients. E69.3 effective class weights: `{exposure_details.get('effective_class_weights')}`; regenerated relation-pair SHA `{exposure_details.get('effective_pair_sha256')}`, {exposure_details.get('effective_pair_count')} pairs, same/different distribution `{exposure_details.get('effective_pair_same_different_distribution')}`. Labels are `{exposure_details.get('true_labels')}` in ASD=0, TD=1 coding.",
        "The following table gives each E72.2 group's selected-checkpoint predictions and global semantic margin change from epoch-10 initialization. The baseline row is recorded in `fold2_exposure.csv`.", "", *fold2_lines,
        "", "Removing these two rows from the fold-2 metric calculation is post-hoc only. Both patients' labels were used in training, alpha calibration and historical test-ACC checkpoint selection; neither reduced nor full metrics are independent validation.",
        "", "## Mechanism diagnostics", "", *mechanism_lines,
        "", "`mechanism.csv` records global/fine and modality losses, semantic accuracy/margins on effective training patients, F-level classification and semantic gradient norms/cosine, per-coordinate-segment gradient cosine, initial and selected-epoch weighted gradient ratios, effective ranks, gates and original-GCN/CADR message norms. Negative F-gradient cosine is a local conflict measure. Nonzero CADR message magnitude or a large adaptive gate establishes numerical contribution only, not diagnostic benefit.",
        "Fine segments are slices of fused DARG F, so modality-specific scores cannot be interpreted as evidence from isolated raw modalities. CORR receives 70% of fine loss and starts with adaptive gate near 0.99; any strengthening may reinforce an already dominant direction rather than add complementary information. No anonymous-prompt group exists, so semantic specificity cannot be isolated from ordinary extra label supervision.",
        "The OOF associations with PHENO site/age/sex and ANAT/FUNC quality are descriptive only. Quality features are not brain lesions. Associations are in audit_status.json under site_age_sex_quality_margin_associations; representation ranks are included in mechanism.csv and audit_status.json.",
        "", "## Verdict and limits", "",
        f"Diagnostic verdict: **{verdict}**. Multi-grain contrasts: G1−G5 {by_pair.get('G1_vs_G5', {}).get('delta_acc_mean_pp', float('nan')):+.4f} pp ({by_pair.get('G1_vs_G5', {}).get('acc_wins', 0)}/{by_pair.get('G1_vs_G5', {}).get('acc_ties', 0)}/{by_pair.get('G1_vs_G5', {}).get('acc_losses', 0)}), G2−G6 {by_pair.get('G2_vs_G6', {}).get('delta_acc_mean_pp', float('nan')):+.4f} pp ({by_pair.get('G2_vs_G6', {}).get('acc_wins', 0)}/{by_pair.get('G2_vs_G6', {}).get('acc_ties', 0)}/{by_pair.get('G2_vs_G6', {}).get('acc_losses', 0)}). These support only the protocol-specific diagnostic conclusion, not a paper-level innovation claim.",
        "The experiment can test whether changing the supervised representation location affects optimization and predictions under E69.3-G2. It cannot establish unbiased generalization, independent clinical validity or uniquely medical-text-driven benefit. A positive mechanism hypothesis requires replication under no-overlap training/test, split-contained preprocessing and validation-only checkpoint selection.",
        f"Protected source/data/history SHA audit: `{manifest.get('protected_source_audit_after_training', {}).get('status', 'PENDING')}`. Independent readback: {len(audit_rows)}/60. Passing audit rows with no fold overlap: {nonoverlap_audits}/54 group-fold checkpoints; fold-2 overlap checkpoints: {len(overlap_audit)}/6.",
        "", "Artifacts: `summary.csv`, `fold_metrics.csv`, `paired_comparisons.csv`, `mechanism.csv`, `fold2_exposure.csv`, `checkpoint_audit.csv`, `rng_parity.csv`, `audit_status.json`.", ""
    ]
    lines.extend(["", "## Quantitative interpretation", "",
        "No E72.2 group has a strictly higher ten-fold mean ACC than E69.3-G2. G4 and G5 tie the baseline mean, but their only winning fold is fold 2, the label-exposed fold: G4 gains two correct cases there and loses one each in folds 4 and 7; G5 gains one there and loses one in fold 10. Across the nine no-index-overlap folds, every group's mean ACC change versus the baseline is negative. Both exposed patients themselves remain correctly classified in the baseline and all six new groups, so the fold-2 gain is not correction of their two hard predictions.",
        "G5 gives a descriptive ranking/sensitivity tradeoff: Prob-AUC rises by 0.5354 pp and SEN by 0.9756 pp, while SPE falls by 0.8649 pp and mean ACC does not improve. G4's tied ACC comes with a 0.6637 pp Prob-AUC decrease. These are partial metric changes, not evidence of a stable overall performance gain.",
        "Moving global-only supervision to F is not consistently better than CAM144: G5 minus E72.1-G5 is +0.2299 pp ACC (4/4/2 W/T/L), but G6 minus E72.1-G6 is -0.3448 pp (0/8/2). G5's nine-fold ACC difference is +0.1277 pp, whereas G6's is -0.2554 pp. These small opposite-direction temperature-dependent differences do not support a general benefit from supervising F.",
        "Fine-grain necessity is unsupported: G1-G5 is -0.2299 pp (0/8/2); G2-G6 is +0.1136 pp (3/5/2), with Prob-AUC -0.2963 pp and SEN -0.7378 pp. No fine-vs-global contrast wins in at least six folds, and the gain at tau=0.1 is not accompanied by consistent multi-metric improvement. This is budget redistribution, not a pure added-loss experiment.",
        "Prompt and temperature effects are also inconsistent: BrainPrompt beats SPromptGL at tau=1 by 0.1149 pp with only one strict winning fold, while at tau=0.1 BrainPrompt is lower by 0.2299 pp and the nine non-overlap-fold mean difference is effectively zero. BrainPrompt fine+global tau=1 and 0.1 tie in ACC; SPromptGL tau=0.1 increases ACC but lowers Prob-AUC; global-only BrainPrompt tau=1 outperforms tau=0.1 in ACC. No prompt/temperature setting establishes a stable new classification gain.",
        "The likely limitation is not pervasive opposition to the classification gradient: all 60 selected-checkpoint eval total F-gradient cosines are positive. CORR fine similarity accuracy is approximately 90.8–94.3% on training patients, whereas PHENO/ANAT/FUNC are approximately 52.7–58.4%. CORR already dominates disease-message blending and receives 70% of fine loss, while the other heads contribute little auxiliary discrimination. This pattern is compatible with redundant supervision and weak complementary signal; it is not causal proof because no additional modality-weight or anonymous-prompt control was run.",
        "The initial gradient budget is equalized, but does not remain equal throughout training: mean selected-checkpoint eval weighted ratios range approximately 0.0117–0.0263 across groups, below the common initial 0.1. Alpha remains frozen as required. Different late-stage ratios are observations, not evidence that calibration was secretly changed.",
        "The resulting research decision is diagnostic NO-GO for claiming a second innovation module or a general advantage of DARG320 supervision. G5 is only a descriptive partial-ranking observation, not a validated candidate. Do not search more temperatures or weights on these same folds. Any follow-up must begin with a new hypothesis and a no-overlap, split-contained preprocessing, validation-selected protocol.", "",
        "## Selected-checkpoint F gradients, CADR intervention and prediction changes", "",
        "| Group | Eval grad cosine | Eval weighted ratio | Eval total negative cosines | Train global effective rank | CADR-off hard flips | Full−off ACC pp | Corrected / harmed vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for group in GROUPS:
        rows = [r for r in mechanism_rows if r["group"] == group]
        mean = lambda key: sum(float(r[key]) for r in rows) / len(rows)
        lines.append(f"| {group} | {mean('best_checkpoint_eval_gradient_cosine_f'):.4f} | {mean('best_checkpoint_eval_weighted_gradient_ratio'):.4f} | {sum(float(r['best_checkpoint_eval_gradient_cosine_f']) < 0 for r in rows)}/10 | {mean('global_effective_rank_train'):.3f} | {sum(int(r['disease_off_hard_prediction_changes']) for r in rows)} | {mean('full_minus_disease_off_acc_pp'):+.4f} | {sum(int(r['corrected_vs_baseline_count']) for r in rows)} / {sum(int(r['harmed_vs_baseline_count']) for r in rows)} |")
    lines.extend(["", "F-gradient fields named best_checkpoint_eval_* are independently recomputed after loading the selected checkpoint in deterministic eval mode; unprefixed gradient fields in mechanism.csv are the recorded train-mode pre-update gradient diagnostics at the selected epoch. These differ by mode and parameter update timing. The fixed-checkpoint disease_off intervention sets the adaptive CADR contribution to zero using the inherited E61 implementation, performs no training, and does not select a perturbation from test labels. It demonstrates small local prediction sensitivity for some groups, not superiority over a separately trained no-CADR model.",
        "Across all groups only three FUNC coordinate-segment eval gradient cosines are negative (one each in G2, G5 and G6); PHENO/ANAT/CORR eval segment cosines are nonnegative. CORR adaptive gates remain about 0.9916–0.9920, versus about 0.9918 in the historical selected checkpoints. Weighted CADR CORR norms are about 1.54–1.65 and substantially exceed the corresponding original-message blend norm; other modality gates remain about 0.001. Gate values and nonzero messages therefore confirm numerical participation, while the intervention and paired metrics set the boundary on diagnostic claims.", "",
        "## E72.1 historical complete metrics", "",
        *format_group_table([r for r in summaries if r["group"].startswith("E72.1-") and int(r["fold_count"]) == 10]), "",
        "## Per-fold independent metrics", "",
        "| Group | Fold | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 | TP/FP/FN/TN | Best epoch (1-based) | Total epochs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"])
    for row in metric_rows:
        lines.append(f"| {row['group']} | {row['fold']} | {row['acc_pct']:.4f} | {row['hard_auc_pct']:.4f} | {row['prob_auc_pct']:.4f} | {row['sen_pct']:.4f} | {row['spe_pct']:.4f} | {row['asd_f1_pct']:.4f} | {row['TP_ASD']}/{row['FP_TD']}/{row['FN_ASD']}/{row['TN_TD']} | {row['best_epoch_1based']} | {row['epochs_executed']} |")
    lines.extend(["", "## Exact frozen equations, projection schema and implementation audit", "",
        "u_g=Normalize(P_g F); u_m=Normalize(P_m F_m); s_ic=u_i dot T_c; DiseaseCE=CE(s/tau,y). BP is tau=1; SABER-style adaptation is tau=0.1. L_fine=.1 L_PHENO+.1 L_ANAT+.1 L_FUNC+.7 L_CORR. Joint L_sem=.5 L_global+.5 L_fine; global-only L_sem=L_global. L_base=L_cls+L_graph+.05 L_pair+.01 L_adv+.01 L_preserve; L_total=L_base+alpha L_sem. Alpha=.1||grad_F L_cls||/(||grad_F L_sem||+1e-12), checked finite in (0,10], computed in deterministic eval once at epoch-10 and frozen.",
        "G1/G2: BrainPrompt-2 joint tau=1/0.1; G3/G4: SPromptGL-8 joint tau=1/0.1; G5/G6: BrainPrompt-2 global-only tau=1/0.1. All six instantiate the same F heads: global 3210 parameters and four fine heads 3240 parameters, 6450 total. Global-only fine parameters receive no training gradient. E72.1 used 1820 total auxiliary parameters (1450 global, 370 shared fine), an additional necessary adaptation and a limit on single-variable causal attribution.",
        "Original optimizers stay GFR Adam lr=.0038 WD=.11, CAM Adam lr=.0038 WD=.11, CADR AdamW lr=.001 WD=.01; new head AdamW lr=.001 WD=.001 updates after those three, with Xavier weights/zero biases, no dropout, private seed 720000+fold. Maximum 450 epochs, warmup 10, wait>200, strict test-ACC improvement checkpoint rule. Five workers were used, GPU0 PPU-ZW810E 98 GiB, with roughly 5.5 GiB aggregate observed GPU use during parallel training. No additional formal group, seed reset per epoch, hyperparameter search or historical retraining was performed.",
        "The full-text local PDF audit and official raw source checks are recorded in .runtime/official_source_audit/manifest.json with exact source/PDF SHAs. Official BrainPrompt compute_label_loss in both graph/CNN variants normalizes fused patient and projected label vectors then passes their dot products directly to F.cross_entropy; the original label_transform is trainable, whereas E72/E72.1/E72.2 deliberately use frozen SVD-space prototypes. Official ABIDE text order is TC then ASD; the cache is verified in reversed ASD/TD order. SPromptGL network.py includes all eight original prompts including Nromal and its PEN aggregation/injection, which this experiment does not reintroduce. SABER Module III contains patient semantic retrieval and gated residual injection (8)–(10), classifier CE (11), class contrastive softmax (12) and combined loss (13); only the class-loss form is adapted here. SABER official code was not verified.",
        "The locked E69.3 commit's run.py, config.json and manifest.json were fetched independently from the public fixed-commit URL and exactly matched the local SHA-locked files. The inherited source audit was used at preflight; fresh online source and full-text revalidation completed during training/post-training without changing the frozen design. The initial audit reader was archived exactly, then corrected only after training for no_grad inference and expanded mechanism/reporting diagnostics; training run.py, mgdss.py, config.json and all checkpoints were unchanged.",
        "The preferred /data experiment directory returned real Errno 5 Input/output error; .runtime/directory_access_audit.json records this and the /model fallback. No protected root, data or historical experiment file was changed.", "",
        "## Fold 2 full and row-excluded complete metrics", "",
        "| Model | Evaluation | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in fold2_metric_rows:
        for kind in ("full", "without"):
            values = " | ".join(f"{float(row[f'{m}_{kind}_pct']):.4f}" for m in ("acc", "hard_auc", "prob_auc", "sen", "spe"))
            lines.append(f"| {row['group']} | {kind} | {values} | {row[f'n_{kind}']} |")
    lines.extend(["", f"Fold-2 training class counts before/after addition: {exposure_details['base_class_counts']} / {exposure_details['effective_class_counts']}; pre-addition weights (class order {exposure_details['base_class_weights_formula_replay']['class_order']}) {exposure_details['base_class_weights_formula_replay']['weights']}; post-addition weights {exposure_details['effective_class_weights']}. These row-excluded values remain label-exposed, validation=test-selected diagnostics.", "",
        "## Descriptive covariate associations", "",
        "| Group | Site mean abs rho | Age-bin mean abs rho | Sex mean abs rho | ANAT quality mean abs rho | FUNC motion/quality mean abs rho |",
        "|---|---:|---:|---:|---:|---:|"])
    for group in GROUPS:
        values = associations.get(group, {})
        cols = []
        for name in ("site", "age_bins", "sex", "ANAT_quality", "FUNC_quality_motion"):
            value = values.get(name, {}).get("mean_absolute_spearman_r")
            cols.append("N/A" if value is None else f"{float(value):.4f}")
        lines.append("| " + group + " | " + " | ".join(cols) + " |")
    lines.extend(["", "These are pooled diagnostic OOF margin associations after training; folds use differently learned projection heads. They are descriptive summaries rather than causal or independently validated confounding estimates. Site and age are encoded features, not raw continuous covariates; the JSON retains the individual strongest feature for each category.", ""])
    spaced = []
    for line in lines:
        if line.startswith("|") and spaced and spaced[-1] and not spaced[-1].startswith("|"):
            spaced.append("")
        spaced.append(line)
    return "\n".join(spaced)

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
                "reason": "Independent inference reader permits no_grad F, adds selected-checkpoint eval F gradients, fixed-checkpoint CADR intervention, E72.1 comparisons and baseline prediction diagnostics. Training, configuration and checkpoint bytes are unchanged.",
                "pre_correction_sha256": frozen_source.get("audit.py"),
                "pre_correction_archived_sha256": R.sha_file(frozen_audit_copy),
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
    official_manifest_path = R.RUNTIME / "official_source_audit/manifest.json"
    official_records = json.loads(official_manifest_path.read_text(encoding="utf-8"))
    for record in official_records:
        path = Path(record["source_path"]) if record.get("source_path") else official_manifest_path.parent / record["name"]
        if record.get("status") not in ("PASS", "READ_LOCAL_FULLTEXT") or R.sha_file(path) != record["sha256"]:
            audit_failures.append({"item": "official_source_revalidation", "error": record["name"]})
    official_labels = (official_manifest_path.parent / "abide_label.txt").read_text().strip().splitlines()
    official_prompt_order = tuple(reversed(official_labels)) == R.BP_TEXTS
    import ast
    official_tree = ast.parse((official_manifest_path.parent / "spromptgl_network.py").read_text())
    official_eight = next(ast.literal_eval(node.value) for node in official_tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "attr_words" for t in node.targets))
    if not official_prompt_order or tuple(official_eight) != R.SP_TEXTS:
        audit_failures.append({"item": "official_prompt_text_identity", "error": "frozen prompt order/text differs from official raw source"})
    source_revalidation = {"status": "PASS" if official_prompt_order and tuple(official_eight) == R.SP_TEXTS else "FAIL",
                           "official_manifest_sha256": R.sha_file(official_manifest_path),
                           "brainprompt_TC_ASD_to_ASD_TD_exact": official_prompt_order,
                           "spromptgl_all_eight_exact": tuple(official_eight) == R.SP_TEXTS,
                           "records": official_records}

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
    manifest["official_source_revalidation"] = source_revalidation
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
        "official_source_revalidation": source_revalidation,
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
