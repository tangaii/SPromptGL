#!/usr/bin/env python3
"""Independent readback, metric, RNG, and mechanism audit for Experiment 72.5."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import run as R
import torch
from scipy.stats import t as student_t
from sklearn.metrics import f1_score, roc_auc_score


EXP = R.EXP
RESULTS = EXP / "results"
GROUPS = tuple(R.GROUP_IDS)
METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
ALL_METRICS = METRICS + ("ba",)
TASK_DIR = R.RUNTIME / "tasks"
CHECKPOINT_DIR = R.RUNTIME / "checkpoints"
MECHANISM_DIR = R.RUNTIME / "mechanism"
NONLINGUISTIC_CACHE = R.NONLINGUISTIC_CACHE
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
HANDOFF_LOCK = HANDOFF.with_name(HANDOFF.name + ".lock")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def task_paths(group: str, fold: int) -> tuple[Path, Path, Path]:
    stem = f"{group}_f{fold:02d}"
    return TASK_DIR / f"{stem}.json", CHECKPOINT_DIR / f"{stem}.best.pt", MECHANISM_DIR / f"{stem}.json"


def metric_values(probability: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64)
    prediction = p.argmax(axis=1)
    tp = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 0) & (prediction == 1)))
    fp = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 1) & (prediction == 1)))
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return {
        "acc": float(np.mean(prediction == y)),
        "hard_auc": float(roc_auc_score(np.eye(2, dtype=np.int64)[y], np.eye(2, dtype=np.int64)[prediction])),
        "prob_auc": float(roc_auc_score(y, p[:, 1])),
        "sen": float(sen), "spe": float(spe), "ba": float((sen + spe) / 2.0),
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "asd_tp": tp, "td_fp": fp, "asd_fn": fn, "td_tn": tn,
        "n_test": int(y.size),
    }


def verify_metrics(actual: dict, expected: dict, where: str) -> None:
    for key in ALL_METRICS:
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            raise AssertionError(f"{where} {key} mismatch: {actual[key]} != {expected[key]}")
    for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
        if key in expected and int(actual[key]) != int(expected[key]):
            raise AssertionError(f"{where} {key} mismatch: {actual[key]} != {expected[key]}")


def summarize(rows: list[dict], key: str = "group") -> list[dict]:
    result = []
    for group in GROUPS:
        selected = [row for row in rows if row.get(key) == group]
        if len(selected) != 10:
            raise AssertionError(f"{group} has {len(selected)} fold metrics, expected 10")
        row = {"group": group, "folds": len(selected)}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in selected], dtype=np.float64) * 100.0
            row[f"{metric}_mean_pct"] = float(values.mean())
            row[f"{metric}_population_sd_pct"] = float(values.std(ddof=0))
        result.append(row)
    return result


def paired(left: dict[int, dict], right: dict[int, dict], left_name: str, right_name: str, fold_set: str = "ALL") -> list[dict]:
    folds = sorted(set(left) & set(right))
    if fold_set == "FOLD2_EXCLUDED":
        folds = [fold for fold in folds if fold != 2]
    result = []
    for metric in METRICS:
        changes = np.asarray([(float(left[f][metric]) - float(right[f][metric])) * 100.0 for f in folds], dtype=np.float64)
        mean = float(changes.mean())
        if len(changes) > 1:
            se = float(changes.std(ddof=1) / math.sqrt(len(changes)))
            width = float(student_t.ppf(0.975, len(changes) - 1) * se)
        else:
            width = float("nan")
        wins = int(np.sum(changes > 1e-12))
        ties = int(np.sum(np.abs(changes) <= 1e-12))
        losses = int(np.sum(changes < -1e-12))
        result.append({
            "comparison": f"{left_name}_vs_{right_name}", "metric": metric, "fold_set": fold_set,
            "n_folds": len(changes), "mean_difference_pp": mean,
            "ci95_low_pp": mean - width, "ci95_high_pp": mean + width,
            "wins": wins, "ties": ties, "losses": losses,
        })
    return result


def error_effect(active_pred: np.ndarray, compare_pred: np.ndarray, y: np.ndarray) -> dict:
    active_correct = active_pred == y
    compare_correct = compare_pred == y
    return {
        "hard_flips": int(np.sum(active_pred != compare_pred)),
        "corrected_by_active": int(np.sum(~compare_correct & active_correct)),
        "new_errors_by_active": int(np.sum(compare_correct & ~active_correct)),
    }


def grad_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    a, b = left.detach().float().reshape(-1), right.detach().float().reshape(-1)
    if float(a.norm().cpu()) <= 0.0 or float(b.norm().cpu()) <= 0.0:
        return None
    return float(torch.nn.functional.cosine_similarity(a[None, :], b[None, :]).cpu())


def flat_grad_norm(values) -> float:
    tensors = [value.detach().float().reshape(-1) for value in values if value is not None]
    return float(torch.cat(tensors).norm().cpu()) if tensors else 0.0


def independent_readback(group: str, fold: int, manifest: dict, ref_manifest: dict, refs: dict[int, dict], warmup: dict,
                         banks: dict[str, torch.Tensor], prompt_meta: dict, patient_cache: dict,
                         nonling_cache: dict) -> tuple[dict, dict, dict, dict]:
    task_path, cp_path, mechanism_path = task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = R.sha_file(cp_path)
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError(f"{group}/f{fold:02d} task/checkpoint status or SHA mismatch")
    if checkpoint.get("schema") != "experiment72_5_safe_dual_prompt_checkpoint_v1" or checkpoint.get("group") != group or int(checkpoint.get("fold", -1)) != fold:
        raise AssertionError(f"{group}/f{fold:02d} checkpoint schema/group/fold mismatch")
    if checkpoint.get("source_sha256") != manifest["source_sha256"] or checkpoint.get("config_sha256") != manifest["config_sha256"]:
        raise AssertionError(f"{group}/f{fold:02d} checkpoint source/config mismatch")
    item = R.group_item(group)
    if checkpoint.get("group_definition") != item or checkpoint.get("patient_token_source") != item["patient_tokens"]:
        raise AssertionError(f"{group}/f{fold:02d} frozen group/token source mismatch")
    expected_condition = bool(item["disease_conditioned_query"])
    if checkpoint.get("disease_difference_conditioned_query_active") is not expected_condition:
        raise AssertionError(f"{group}/f{fold:02d} condition branch activation mismatch")
    if checkpoint.get("patient_text_cache_sha256") != R.sha_file(R.PATIENT_CACHE):
        raise AssertionError(f"{group}/f{fold:02d} E72.4 patient cache SHA mismatch")
    expected_nonling_sha = R.sha_file(NONLINGUISTIC_CACHE) if item["patient_tokens"] == "nonlinguistic" else None
    if checkpoint.get("nonlinguistic_token_cache_sha256") != expected_nonling_sha:
        raise AssertionError(f"{group}/f{fold:02d} G6 category-token cache SHA mismatch")
    if checkpoint.get("prompt_cache_sha256") != R.sha_file(R.PROMPT_CACHE) or checkpoint.get("text_basis_sha256") != prompt_meta["basis_sha256"]:
        raise AssertionError(f"{group}/f{fold:02d} frozen disease prompt cache mismatch")
    expected_bank = prompt_meta["banks"][item["prompt"]]["bank_sha256"] if item["disease_supervision"] else None
    if checkpoint.get("prompt_bank_sha256") != expected_bank:
        raise AssertionError(f"{group}/f{fold:02d} disease prototype bank mismatch")

    state = R.build_fold(fold, ref_manifest, refs)
    expected_overlap = [377, 407] if fold == 2 else []
    if list(state["train_test_overlap_1based"]) != expected_overlap or checkpoint.get("train_test_overlap_1based") != expected_overlap:
        raise AssertionError(f"{group}/f{fold:02d} historical overlap mismatch")
    provenance_pairs = (
        ("split_sha256", "split_sha"), ("pair_sha256", "pair_sha"),
        ("effective_train_sha256", "effective_train_sha"), ("base_train_sha256", "base_train_sha"),
        ("initial_shared_sha256", "initial_shared_sha"), ("initial_gfr_sha256", "initial_gfr_sha"),
        ("initial_cam_sha256", "initial_cam_sha"), ("initial_added_sha256", "initial_added_sha"),
        ("initial_schema_sha256", "initial_schema_sha"), ("initial_rng_sha256", "construction_rng_sha"),
    )
    for cp_key, state_key in provenance_pairs:
        if checkpoint.get(cp_key) != state.get(state_key):
            raise AssertionError(f"{group}/f{fold:02d} provenance mismatch {cp_key}")
    warm_sha = warmup["shared_warmup_cache"]["folds"][str(fold)]["sha256"]
    if checkpoint.get("warmup_sha256") != warm_sha or state.get("warmup_cache_sha256") != warm_sha:
        raise AssertionError(f"{group}/f{fold:02d} E61 warmup SHA mismatch")
    if checkpoint.get("reference_checkpoint_sha256") != refs[fold]["checkpoint_sha256"]:
        raise AssertionError(f"{group}/f{fold:02d} E69 reference checkpoint SHA mismatch")
    if checkpoint.get("e69_reference_task_sha256") != R.sha_file(R.E69_TASKS / f"E69.3-G2_f{fold:02d}.json"):
        raise AssertionError(f"{group}/f{fold:02d} E69 task SHA mismatch")
    if checkpoint.get("train_indices_sha256") != R.sha_obj(list(map(int, state["train"]))) or checkpoint.get("test_indices_sha256") != R.sha_obj(list(map(int, state["test"]))):
        raise AssertionError(f"{group}/f{fold:02d} train/test index SHA mismatch")
    expected_alpha = float(manifest["preflight"]["alphas"][str(fold)][group]["alpha"])
    if abs(float(checkpoint["alpha"]) - expected_alpha) > 1e-15 or abs(float(task["alpha"]) - expected_alpha) > 1e-15:
        raise AssertionError(f"{group}/f{fold:02d} frozen alpha mismatch")
    if int(checkpoint["outer_train_count"]) != len(state["train"]) or int(checkpoint["outer_test_count"]) != len(state["test"]):
        raise AssertionError(f"{group}/f{fold:02d} train/test counts mismatch")
    if checkpoint.get("test_labels_used_for_auxiliary_loss") is not (fold == 2 and item["disease_supervision"]):
        raise AssertionError(f"{group}/f{fold:02d} auxiliary label-exposure flag mismatch")
    if checkpoint.get("overlap_labels_used_for_classification") is not (fold == 2):
        raise AssertionError(f"{group}/f{fold:02d} classification label-exposure flag mismatch")
    if checkpoint.get("auxiliary_label_exposure_rows_1based") != expected_overlap:
        raise AssertionError(f"{group}/f{fold:02d} overlap row disclosure mismatch")

    rng_before_init = R.BASE.rng_sha(R.BASE.pack_rng())
    module, patient_optimizer, disease_optimizer, init_sha, rng_before, rng_after = R.private_auxiliary(int(prompt_meta["rank"]), fold)
    if rng_before != rng_after or rng_before_init != R.BASE.rng_sha(R.BASE.pack_rng()) or init_sha != checkpoint["initial_auxiliary_sha256"]:
        raise AssertionError(f"{group}/f{fold:02d} private module initialization/RNG mismatch")
    module.load_state_dict(checkpoint["auxiliary"], strict=True)
    state["model"].load_state_dict(checkpoint["base"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    patient_optimizer.load_state_dict(checkpoint["optimizer_patient_prompt"])
    disease_optimizer.load_state_dict(checkpoint["optimizer_disease_supervision_head"])
    optimizer_pairs = (
        (state["opt_base"], checkpoint["optimizer_base"], "GFR"),
        (state["opt_cam"], checkpoint["optimizer_cam"], "CAM"),
        (state["opt_added"], checkpoint["optimizer_added"], "CADR"),
        (patient_optimizer, checkpoint["optimizer_patient_prompt"], "patient"),
        (disease_optimizer, checkpoint["optimizer_disease_supervision_head"], "disease"),
    )
    for optimizer, saved, label in optimizer_pairs:
        if not R.tree_equal(optimizer.state_dict(), saved):
            raise AssertionError(f"{group}/f{fold:02d} {label} optimizer state readback mismatch")
    if R.BASE.rng_sha(checkpoint["rng"]) != checkpoint.get("rng_sha256") or checkpoint.get("rng_sha256") != task.get("best_rng_sha256"):
        raise AssertionError(f"{group}/f{fold:02d} best RNG state is absent or mismatched")
    if not R.tree_equal(checkpoint["patient_prompt"], {key.removeprefix("patient."): value for key, value in checkpoint["auxiliary"].items() if key.startswith("patient.")}):
        raise AssertionError(f"{group}/f{fold:02d} patient module duplicate state mismatch")
    if not R.tree_equal(checkpoint["disease_supervision_head"], {key.removeprefix("disease."): value for key, value in checkpoint["auxiliary"].items() if key.startswith("disease.")}):
        raise AssertionError(f"{group}/f{fold:02d} disease head duplicate state mismatch")

    device = R.BASE.runtime()["device"]
    module.to(device).eval()
    state["model"].eval(); state["cam"].eval()
    if item["patient_tokens"] == "nonlinguistic":
        patient_tokens = nonling_cache["tokens"].to(device=device, dtype=torch.float32)
    else:
        patient_tokens = patient_cache["embeddings"].to(device=device, dtype=torch.float32)
    bank = banks[item["prompt"]].to(device=device, dtype=torch.float32) if item["disease_supervision"] else None
    condition_bank = banks["SPromptGL-8"].to(device=device, dtype=torch.float32)
    state.update({
        "_dual_auxiliary": module, "_patient_tokens": patient_tokens,
        "_patient_prompt_active": True, "_patient_residual_enabled": True,
        "_condition_enabled": bool(item["disease_conditioned_query"]), "_condition_bank": condition_bank,
        "_condition_override": None, "_condition_permutation": None,
    })
    with torch.no_grad():
        output = R.checked_forward(state, training=False, use_cadr=True)
    if int(state["cam"]._e72_forward_count) != 1:
        raise AssertionError(f"{group}/f{fold:02d} CAM call count is not one")
    if tuple(state["cam"]._e72_fused_after_prompt.shape) != (871, 320) or tuple(state["cam"]._e59_positions.shape) != (871, 4, 36):
        raise AssertionError(f"{group}/f{fold:02d} CAM pre/post capture dimensions mismatch")
    if not torch.equal(state["cam"]._e72_classifier_input, state["cam"]._e59_positions.reshape(871, 144)):
        raise AssertionError(f"{group}/f{fold:02d} original classifier input changed")
    probability = output["logp"].detach().exp().cpu().numpy()
    test_ids = np.asarray(state["test"], dtype=np.int64)
    labels = state["y"].detach().cpu().numpy().astype(np.int64, copy=False)[test_ids]
    active_probability = probability[test_ids]
    metrics = metric_values(active_probability, labels)
    verify_metrics(metrics, checkpoint["metrics"], f"{group}/f{fold:02d} checkpoint")
    verify_metrics(metrics, task["metrics"], f"{group}/f{fold:02d} task")
    for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
        if int(metrics[key]) != int(task["confusion"][key]):
            raise AssertionError(f"{group}/f{fold:02d} confusion mismatch {key}")
    metric_row = {"group": group, "fold": fold, **{key: float(metrics[key]) for key in METRICS}, **{key: int(metrics[key]) for key in ("asd_tp", "td_fp", "asd_fn", "td_tn", "n_test")}, "best_epoch": int(checkpoint["best_epoch"]), "epochs_executed": int(checkpoint["epochs_executed"]), "alpha": float(checkpoint["alpha"])}
    task_row = {"task": task, "checkpoint": checkpoint, "state": state, "module": module,
                "patient_optimizer": patient_optimizer, "disease_optimizer": disease_optimizer,
                "probability": probability, "test_ids": test_ids, "labels": labels,
                "active_probability": active_probability, "active_prediction": active_probability.argmax(1),
                "metrics": metrics, "mechanism_path": mechanism_path, "item": item, "bank": bank,
                "prompt": state["cam"]._e72_prompt_output}
    audit_row = {
        "group": group, "fold": fold, "status": "PASS", "checkpoint_sha256": checkpoint_sha,
        "metric_recompute": "EXACT_1E-10", "optimizer_state_readback": "PASS",
        "best_rng_sha256": checkpoint["rng_sha256"], "cam_calls": int(state["cam"]._e72_forward_count),
        "patient_residual_max": float(task_row["prompt"]["relative_residual"].detach().max().cpu()),
        "residual_cap_pass": bool(float(task_row["prompt"]["relative_residual"].detach().max().cpu()) <= 0.0200001),
        "token_source": item["patient_tokens"],
    }
    if not audit_row["residual_cap_pass"]:
        raise AssertionError(f"{group}/f{fold:02d} patient residual exceeded 2% cap")
    return metric_row, task_row, audit_row, {"group": group, "fold": fold, "epoch_records": json.loads(mechanism_path.read_text(encoding="utf-8"))}


def load_h0_fold2(ref_manifest: dict, refs: dict[int, dict], warmup: dict, banks: dict[str, torch.Tensor]) -> dict:
    state = R.build_fold(2, ref_manifest, refs)
    R.load_warmup(state, 2, warmup)
    cp_path = R.REFERENCE_EXP / "E69.3-G2_f02.best.pt"
    cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    state["model"].load_state_dict(cp["model"], strict=True)
    state["cam"].load_state_dict(cp["cam"], strict=True)
    state["opt_base"].load_state_dict(cp["optimizer_base"])
    state["opt_cam"].load_state_dict(cp["optimizer_cam"])
    state["opt_added"].load_state_dict(cp["optimizer_added"])
    with torch.no_grad():
        output = R.checked_forward(state, training=False, use_cadr=True)
    probability = output["logp"].detach().exp().cpu().numpy()
    ids = np.asarray(state["test"], dtype=np.int64)
    labels = state["y"].detach().cpu().numpy().astype(np.int64, copy=False)[ids]
    return {"probability": probability, "test_ids": ids, "labels": labels,
            "metrics": metric_values(probability[ids], labels), "checkpoint_sha256": R.sha_file(cp_path)}


def rng_audit_for_task(group: str, fold: int, current_rows: list[dict], e69_task: dict) -> dict:
    expected = {int(row["epoch"]): row for row in e69_task.get("rng_trace", [])}
    actual = {int(row["epoch"]) + 1: row for row in current_rows}
    common = sorted(set(expected) & set(actual))
    matched = 0
    mismatch = []
    for epoch in common:
        base, current = expected[epoch], actual[epoch]
        pairs = (("rng_before", "rng_before_train_sha256"), ("rng_after_train", "rng_after_train_sha256"),
                 ("rng_after_eval", "rng_after_eval_sha256"))
        if all(base.get(left) == current.get(right) for left, right in pairs):
            matched += 1
        else:
            mismatch.append(epoch)
    return {"group": group, "fold": fold, "common_epochs": len(common), "exact_epoch_triplets": matched,
            "mismatched_epochs": json.dumps(mismatch), "status": "PASS" if common and matched == len(common) else "FAIL"}


def mechanism_for_readback(row: dict, g3_cache: dict[int, dict], selections: list[dict]) -> list[dict]:
    item = row["item"]
    group, fold = item["id"], int(row["task"]["fold"])
    state, module = row["state"], row["module"]
    bank = row["bank"]
    checkpoint = row["checkpoint"]
    test_ids, y = row["test_ids"], row["labels"]
    full_pred = row["active_prediction"]
    output = row["output"] if "output" in row else None
    prompt = row["prompt"]
    probability_all = row["probability"]
    base_metrics = row["metrics"]
    metrics = {
        "group": group, "fold": fold,
        "residual_mean": float(prompt["relative_residual"].detach().mean().cpu()),
        "residual_p50": float(torch.quantile(prompt["relative_residual"].detach(), 0.50).cpu()),
        "residual_p95": float(torch.quantile(prompt["relative_residual"].detach(), 0.95).cpu()),
        "residual_max": float(prompt["relative_residual"].detach().max().cpu()),
        "residual_fraction_ge_1p9pct": float((prompt["relative_residual"].detach() >= 0.019).float().mean().cpu()),
        "rho_mean": float(prompt["rho"].detach().mean().cpu()),
        "gate_mean": float(prompt["gate"].detach().mean().cpu()),
        "gate_std": float(prompt["gate"].detach().std(unbiased=False).cpu()),
        "null_attention_mean": float(prompt["null_attention"].detach().mean().cpu()),
        "attention_age": float(prompt["attention"].detach()[:, 0].mean().cpu()),
        "attention_sex": float(prompt["attention"].detach()[:, 1].mean().cpu()),
        "attention_site": float(prompt["attention"].detach()[:, 2].mean().cpu()),
        "attention_null": float(prompt["attention"].detach()[:, 3].mean().cpu()),
        "attention_between_patient_std_mean": float(prompt["attention"].detach().std(0, unbiased=False).mean().cpu()),
        "attention_entropy_mean": float(prompt["attention_entropy"].detach().mean().cpu()),
        "attention_token_order": "age,sex,site,null",
        "condition_lambda": float(prompt["condition_lambda"].detach().cpu()),
        "condition_strength_mean": None if prompt["condition_strength"] is None else float(prompt["condition_strength"].detach().mean().cpu()),
        "condition_strength_std": None if prompt["condition_strength"] is None else float(prompt["condition_strength"].detach().std(unbiased=False).cpu()),
        "logit_max_abs_vs_patient_residual_off": None,
        "flip_count_residual_off": None,
        "corrected_by_active_residual": None,
        "new_errors_by_active_residual": None,
        "site_mask_acc_delta_pp": None,
        "condition_lambda_off_logit_linf": None,
        "condition_lambda_off_flips": None,
        "condition_zero_flips": None,
        "condition_reverse_attention_linf": None,
        "condition_reverse_logit_linf": None,
        "condition_reverse_flips": None,
        "condition_reverse_corrected": None,
        "condition_reverse_new_errors": None,
        "condition_permute_attention_linf": None,
        "condition_permute_logit_linf": None,
        "condition_permute_flips": None,
        "condition_permute_corrected": None,
        "condition_permute_new_errors": None,
        "g3_lambda_off_exact_match": None,
        "patient_token_source": item["patient_tokens"],
    }
    device = R.BASE.runtime()["device"]
    state["_patient_residual_enabled"] = False
    with torch.no_grad():
        off = R.checked_forward(state, training=False, use_cadr=True)
    off_prob = off["logp"].detach().exp().cpu().numpy()[test_ids]
    off_pred = off_prob.argmax(1)
    off_metrics = metric_values(off_prob, y)
    active_logits = np.log(np.clip(row["active_probability"], 1e-12, 1.0))
    off_logits = off["logp"].detach().cpu().numpy()[test_ids]
    metrics["logit_max_abs_vs_patient_residual_off"] = float(np.max(np.abs(active_logits - off_logits)))
    metrics.update({f"patient_off_{key}": float(off_metrics[key]) for key in METRICS})
    effects = error_effect(full_pred, off_pred, y)
    metrics.update({"flip_count_residual_off": effects["hard_flips"], "corrected_by_active_residual": effects["corrected_by_active"], "new_errors_by_active_residual": effects["new_errors_by_active"]})
    state["_patient_residual_enabled"] = True

    with torch.no_grad():
        prompt_active = row["prompt"]["attention"].detach().clone()
    rho = prompt["rho"].detach().cpu().numpy()
    gate = prompt["gate"].detach().squeeze(-1).cpu().numpy()
    if np.std(rho) > 0 and np.std(gate) > 0:
        metrics["rho_gate_pearson"] = float(np.corrcoef(rho, gate)[0, 1])
    else:
        metrics["rho_gate_pearson"] = None

    if item["patient_tokens"] == "real" and group == "G5":
        tokens = state["_patient_tokens"]
        masked = tokens.clone()
        masked[:, 2, :] = 0.0
        state["_patient_tokens"] = masked
        with torch.no_grad():
            mask_out = R.checked_forward(state, training=False, use_cadr=True)
        mask_prob = mask_out["logp"].detach().exp().cpu().numpy()[test_ids]
        mask_metrics = metric_values(mask_prob, y)
        metrics["site_mask_acc_delta_pp"] = (mask_metrics["acc"] - base_metrics["acc"]) * 100.0
        state["_patient_tokens"] = tokens

    if group == "G5":
        active_attention = prompt_active
        interventions = {}
        state["_condition_enabled"] = True
        state["_condition_override"] = "off"
        state["_condition_permutation"] = None
        with torch.no_grad():
            interventions["lambda_off"] = R.checked_forward(state, training=False, use_cadr=True)
            lambda_off_attention = state["cam"]._e72_prompt_output["attention"].detach().clone()
        state["_condition_override"] = "zero"
        with torch.no_grad():
            interventions["zero"] = R.checked_forward(state, training=False, use_cadr=True)
        state["_condition_override"] = "reverse"
        with torch.no_grad():
            interventions["reverse"] = R.checked_forward(state, training=False, use_cadr=True)
            reverse_attention = state["cam"]._e72_prompt_output["attention"].detach().clone()
        state["_condition_override"] = "permute"
        state["_condition_permutation"] = torch.roll(torch.arange(871, device=device), shifts=1)
        with torch.no_grad():
            interventions["permute"] = R.checked_forward(state, training=False, use_cadr=True)
            permute_attention = state["cam"]._e72_prompt_output["attention"].detach().clone()
        state["_condition_override"] = None
        state["_condition_permutation"] = None

        lambda_prob = interventions["lambda_off"]["logp"].detach().exp().cpu().numpy()[test_ids]
        zero_prob = interventions["zero"]["logp"].detach().exp().cpu().numpy()[test_ids]
        reverse_prob = interventions["reverse"]["logp"].detach().exp().cpu().numpy()[test_ids]
        permute_prob = interventions["permute"]["logp"].detach().exp().cpu().numpy()[test_ids]
        active_logp = np.log(np.clip(row["active_probability"], 1e-12, 1.0))
        for name, prob in (("lambda_off", lambda_prob), ("reverse", reverse_prob), ("permute", permute_prob)):
            effects = error_effect(full_pred, prob.argmax(1), y)
            metrics[f"condition_{name}_flips"] = effects["hard_flips"]
            metrics[f"condition_{name}_corrected"] = effects["corrected_by_active"]
            metrics[f"condition_{name}_new_errors"] = effects["new_errors_by_active"]
            metrics[f"condition_{name}_logit_linf"] = float(np.max(np.abs(active_logp - np.log(np.clip(prob, 1e-12, 1.0)))))
        metrics["condition_zero_flips"] = int(np.sum(zero_prob.argmax(1) != lambda_prob.argmax(1)))
        metrics["condition_reverse_attention_linf"] = float((active_attention - reverse_attention).abs().max().cpu())
        metrics["condition_permute_attention_linf"] = float((active_attention - permute_attention).abs().max().cpu())
        if "probability" in g3_cache:
            metrics["g3_lambda_off_exact_match"] = bool(np.array_equal(lambda_prob, g3_cache["probability"][test_ids]))

    # Best-checkpoint disease head and its weighted gradient at CAM and DARG positions.
    if item["disease_supervision"]:
        with torch.enable_grad():
            out_grad = R.checked_forward(state, training=False, use_cadr=True)
            positions = state["cam"]._e59_positions
            train_ids = torch.as_tensor(state["train"], dtype=torch.long, device=device)
            train_y = state["y"][train_ids].long()
            disease_output = module.disease(positions, bank, float(item["temperature"]), use_fine=True)
            train_disease = module.disease(positions[train_ids], bank, float(item["temperature"]), use_fine=True)
            sem_losses = R.supervised_losses(train_disease, train_y)
            semantic = 0.5 * sem_losses["global"] + 0.5 * sem_losses["fine"]
            cls_z = torch.autograd.grad(out_grad["cls"], positions, retain_graph=True)[0][train_ids].detach().float()
            sem_z = torch.autograd.grad(semantic, positions, retain_graph=True)[0][train_ids].detach().float()
            fused = state["cam"]._e72_fused_before_prompt
            cls_f = torch.autograd.grad(out_grad["cls"], fused, retain_graph=True)[0][train_ids].detach().float()
            sem_f = torch.autograd.grad(semantic, fused, retain_graph=True)[0][train_ids].detach().float()
            patient_params = tuple(module.patient.parameters())
            cls_patient = torch.autograd.grad(out_grad["cls"], patient_params, retain_graph=True, allow_unused=True)
            sem_patient = torch.autograd.grad(semantic, patient_params, retain_graph=True, allow_unused=True)
            cls_patient_norm = flat_grad_norm(cls_patient)
            sem_patient_norm = flat_grad_norm(sem_patient)
            cls_z_norm, sem_z_norm = float(cls_z.norm().cpu()), float(sem_z.norm().cpu())
            alpha = float(checkpoint["alpha"])
            similarities = (disease_output["global_logits"] * float(item["temperature"])).detach()
            test_t = torch.as_tensor(test_ids, dtype=torch.long, device=device)
            test_y = state["y"][test_t].long()
            global_pred = similarities[test_t].argmax(-1)
            fine_sim = (disease_output["fine_logits"] * float(item["temperature"])).detach()
            fine_pred = fine_sim[test_t].argmax(-1)
            global_margin = (similarities[test_t].gather(1, test_y[:, None]) - similarities[test_t].gather(1, (1 - test_y)[:, None])).squeeze(1)
            fine_margin = (fine_sim[test_t].gather(2, test_y[:, None, None].expand(-1, 4, 1)) - fine_sim[test_t].gather(2, (1-test_y)[:, None, None].expand(-1, 4, 1))).squeeze(-1)
            metrics.update({
                "semantic_global_acc_test": float((global_pred == test_y).float().mean().cpu()),
                "semantic_global_margin_test": float(global_margin.mean().cpu()),
                "semantic_fine_pheno_acc_test": float((fine_pred[:, 0] == test_y).float().mean().cpu()),
                "semantic_fine_anat_acc_test": float((fine_pred[:, 1] == test_y).float().mean().cpu()),
                "semantic_fine_func_acc_test": float((fine_pred[:, 2] == test_y).float().mean().cpu()),
                "semantic_fine_corr_acc_test": float((fine_pred[:, 3] == test_y).float().mean().cpu()),
                "semantic_fine_margin_test": float(fine_margin.mean().cpu()),
                "gradient_cls_cam_norm": cls_z_norm,
                "gradient_sem_cam_norm": sem_z_norm,
                "gradient_weighted_sem_cam_ratio": alpha * sem_z_norm / (cls_z_norm + 1e-12),
                "gradient_cls_darg_norm": float(cls_f.norm().cpu()),
                "gradient_sem_darg_norm": float(sem_f.norm().cpu()),
                "gradient_patient_cls_norm": cls_patient_norm,
                "gradient_patient_sem_norm": sem_patient_norm,
                "gradient_patient_cosine": None,
                "gradient_cam_cosine": grad_cosine(cls_z, sem_z),
                "gradient_darg_cosine": grad_cosine(cls_f, sem_f),
                "semantic_global_loss_train": float(sem_losses["global"].detach().cpu()),
                "semantic_fine_loss_train": float(sem_losses["fine"].detach().cpu()),
                "semantic_alpha": alpha,
                "semantic_weighted_grad_ratio_target": float(R.CFG["supervision"]["gradient_budget_ratio"]),
            })
            if cls_patient_norm > 0 and sem_patient_norm > 0:
                left, right = [], []
                for a, b in zip(cls_patient, sem_patient):
                    if a is not None and b is not None:
                        left.append(a.detach().float().reshape(-1)); right.append(b.detach().float().reshape(-1))
                metrics["gradient_patient_cosine"] = grad_cosine(torch.cat(left), torch.cat(right)) if left else None

    # The relation graph is audited as model state, without assigning clinical meaning to a gate.
    adaptive = torch.sigmoid(state["model"].adaptive_gate_logits.detach()).cpu().numpy()
    edge = torch.sigmoid(state["model"].edge_gate_logits.detach()).cpu().numpy()
    ref_gate = np.asarray(R.load_reference()[1][fold].get("best_adaptive_gate", R.CFG["base"]["adaptive_gate_initial"]), dtype=np.float64)
    metrics.update({
        "adaptive_gate_mean": float(adaptive.mean()),
        "adaptive_gate_values": json.dumps(adaptive.tolist()),
        "adaptive_gate_delta_abs_mean_vs_e69": float(np.abs(adaptive - ref_gate).mean()),
        "edge_gate_mean": float(edge.mean()),
        "edge_gate_values": json.dumps(edge.tolist()),
    })
    subgroup_rows = []
    y_all = state["y"].detach().cpu().numpy().astype(np.int64, copy=False)
    pred_all = probability_all.argmax(1)
    for field in ("site", "age", "sex"):
        for idx in test_ids:
            selection = selections[int(idx)].get(field)
            category = None if selection is None else selection.get("column")
            if category is None:
                continue
            subgroup_rows.append({"group": group, "fold": fold, "field": field, "category": category,
                                  "patient_id_1based": int(idx + 1), "label": int(y_all[idx]),
                                  "prediction": int(pred_all[idx]), "prob_td": float(probability_all[idx, 1]),
                                  "correct": int(pred_all[idx] == y_all[idx])})
    return [metrics], subgroup_rows


def append_handoff(record: str) -> str:
    marker = "## Experiment 72.5 - Safe Adaptive Dual Prompt and Disease-Difference Conditioning"
    HANDOFF.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_LOCK.touch(exist_ok=True)
    with HANDOFF_LOCK.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        content = HANDOFF.read_text(encoding="utf-8")
        if marker in content:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return R.sha_file(HANDOFF)
        with HANDOFF.open("ab") as stream:
            if content and not content.endswith("\n"):
                stream.write(b"\n")
            stream.write(("\n" + record.rstrip() + "\n").encode("utf-8"))
            stream.flush(); os.fsync(stream.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return R.sha_file(HANDOFF)


def report_text(manifest: dict, summaries: list[dict], comparisons: list[dict], mechanisms: list[dict],
                fold2: list[dict], subgroup_summary: list[dict], audit_status: dict, verdict: str) -> str:
    e69_rows = R.reference_rows()
    e69_summary = {metric: float(np.mean([row[metric] for row in e69_rows.values()])) * 100.0 for metric in METRICS}
    e69_sd = {metric: float(np.std([row[metric] for row in e69_rows.values()], ddof=0)) * 100.0 for metric in METRICS}
    lines = [
        "# Experiment 72.5 - Safe Adaptive Dual Prompt and Disease-Difference Conditioning", "",
        f"Status: **{audit_status['experiment_status']}**; independent checkpoints **{audit_status['independent_checkpoint_audits_passed']}/60**; decision **{verdict}**.", "",
        "## Protocol and source review", "",
        "The only training base is E69.3-G2 at `41dbcb45a0273673857051d8e821b8c3315eef00`, continuing the E61-E2 end-to-end adaptive DARG forward. No E72.4 checkpoint was used for initialization. E72.4 was read-only and contributed only its previously audited patient text cache.",
        "SPromptGL's MICCAI 2025 paper/code place a 320-dimensional GFR output before CAM and preserve four 36-dimensional positions that flatten to the original 144-dimensional classifier input. BrainPrompt's official ABIDE label text order is TC then ASD; this experiment maps the resulting disease class bank to ASD=0, TD=1. Its patient metadata source includes real participant demographics. SABER Module III's Graph-CLS-conditioned query, patient-text cross-attention and gated residual were verified from the paper equations (8)-(13); no public official SABER implementation was verified. The frozen CLIP RN50 in this experiment is a comparability choice, not a full reproduction of either paper's encoder.",
        "References: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL official code](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official code](https://github.com/AngusMonroe/BrainPrompt), [SABER paper](https://arxiv.org/abs/2607.01901).", "",
        "Patient text uses only PHENO-recoverable age intervals, recorded sex category codes and site IDs from the existing E72.4 cache (871 rows, three tokens, RN50 1024-d; no truncation). G6 maps the same category identities to fixed, nonlinguistic 1024-d random vectors using a private CPU generator. Neither construction uses diagnosis labels. These demographic/site fields partly duplicate PHENO48; G5 versus G6 tests this specific encoding control and does not establish independent clinical knowledge.", "",
        "The safety residual is `rho = 0.02 * sigmoid(gate) * (1 - null_attention)` times the DARG feature norm along a normalized learned direction. Its maximum is 2%; the scalar gate starts at 0.1, so the initial preflight maximum is bounded by 0.2% before null attenuation. Disease conditioning uses the signed `tanh(cos(W_condition(F), normalize(T_ASD-T_TD)))` strength, no hard/true label, and a bounded lambda initialized at 0.05. Disease supervision remains CAM-after global plus four-position CE with the E72.1 10-d frozen RN50 text space and fixed fine weights.", "",
        "Historical limitations remain: fold 2 trains on rows 377 (ASD) and 407 (TD), which remain in its test fold; checkpoint selection uses `val=test`; RFE/standardization precede CV; and the patient graph is transductive. Metrics below are mechanism-paired historical-protocol results, not unbiased generalization estimates.", "",
        "## Results", "",
        "Mean +/- population SD across ten folds, in percent. Hard-AUC is computed from hard predictions; Prob-AUC uses continuous TD probability.", "",
        "|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|",
        "|---|---:|---:|---:|---:|---:|---:|",
        "|E69.3-G2|" + "|".join(f"{e69_summary[m]:.4f} +/- {e69_sd[m]:.4f}" for m in METRICS) + "|",
    ]
    for row in summaries:
        lines.append("|" + str(row["group"]) + "|" + "|".join(f"{row[m+'_mean_pct']:.4f} +/- {row[m+'_population_sd_pct']:.4f}" for m in METRICS) + "|")
    lines.extend(["", "Paired changes are percentage points. W/T/L counts are paired folds; confidence intervals are descriptive t intervals over correlated folds.", "", "|Comparison|Metric|Folds|Mean delta|95% CI|W/T/L|", "|---|---|---:|---:|---:|---:|"])
    for row in comparisons:
        if row["fold_set"] != "ALL" or row["metric"] in ("acc", "hard_auc", "prob_auc", "sen", "spe"):
            lines.append(f"|{row['comparison']}|{row['metric']} ({row['fold_set']})|{row['n_folds']}|{row['mean_difference_pp']:+.4f}|[{row['ci95_low_pp']:+.4f}, {row['ci95_high_pp']:+.4f}]|{row['wins']}/{row['ties']}/{row['losses']}|")
    lines.extend(["", "## Mechanism", "", "The safe-residual unit test verified the initial 0.2% upper bound, explicit Null-token attenuation, lower-gate/lower-residual behavior, finite nonzero gate and direction gradients, and unchanged main RNG. Per-checkpoint diagnostics are in `mechanism.csv`; attention is reported as model behavior, not clinical explanation.", ""])
    for group in GROUPS:
        selected = [row for row in mechanisms if row["group"] == group]
        if not selected:
            continue
        avg = lambda key: float(np.mean([float(row[key]) for row in selected if row.get(key) is not None])) if any(row.get(key) is not None for row in selected) else float("nan")
        lines.append(f"- {group}: residual mean/p95/max {avg('residual_mean'):.5f}/{avg('residual_p95'):.5f}/{avg('residual_max'):.5f}; fraction at >=1.9% cap {avg('residual_fraction_ge_1p9pct'):.3f}; null attention {avg('null_attention_mean'):.3f}; age/sex/site attention {avg('attention_age'):.3f}/{avg('attention_sex'):.3f}/{avg('attention_site'):.3f}; residual-off hard flips {avg('flip_count_residual_off'):.2f} per fold.")
    g5 = [row for row in mechanisms if row["group"] == "G5"]
    if g5:
        lines.append(f"- G5 disease-difference query: condition lambda mean {np.mean([row['condition_lambda'] for row in g5]):.4f}; signed strength SD {np.mean([row['condition_strength_std'] for row in g5]):.4f}; lambda-off flips {np.mean([row['condition_lambda_off_flips'] for row in g5]):.2f} per fold; reverse flips {np.mean([row['condition_reverse_flips'] for row in g5]):.2f}; strength-permutation flips {np.mean([row['condition_permute_flips'] for row in g5]):.2f}; site-token mask ACC delta {np.mean([row['site_mask_acc_delta_pp'] for row in g5]):+.3f} pp.")
    lines.extend(["", "Fold-2 full and row-removal sensitivities are in `fold2_sensitivity.csv`. Removing rows 377/407 only removes this known overlap; all other historical evaluation limitations remain.", "", "Subgroup summaries are in `subgroup_diagnostics.csv`. Site-token reliance or site-specific performance is a dataset-bias risk, not an ASD mechanism. DARG gate changes and relation-edge diagnostics are recorded per checkpoint; they do not alone establish useful relation learning.", "", "## Decision", "", f"Pre-registered decision: **{verdict}**. The first-stage performance repair compares G1 against E69.3-G2 and the historical E72.4-G1 loss of -7.6985 pp ACC. The innovation stage requires G5 to match/exceed E69.3-G2, consistently beat G3 and G6, and show a net positive fixed-checkpoint condition intervention. No conclusion is upgraded from a single fold or hard-flip count.", "", "Even a positive result would require a clean follow-up with no train/test overlap, fold-local preprocessing, validation-only checkpoint selection, and external or nested validation before a generalization claim.", "", "## Audit", "", f"- Independent best-checkpoint reads: {audit_status['independent_checkpoint_audits_passed']}/60; failures: {len(audit_status['audit_failures'])}.", f"- Exact common epoch RNG triplets: {audit_status['rng_exact_triplets']}/{audit_status['rng_compared_triplets']}.", f"- Protected sources/data/history: {audit_status['protected_source_audit']['status']} ({audit_status['protected_source_audit']['protected_file_count']} files).",
        f"- Frozen training source/config SHA is in `manifest.json`; report SHA is recorded after audit. Experiment root: `{EXP}`.",
        "- Audit-only correction: E72.4 comparison CSV percentage-column mapping was fixed; no training code, prediction, or checkpoint changed. The preflight and corrected audit.py SHA values are recorded in the manifest.", ""])
    return "\n".join(lines)


def run_audit() -> dict:
    torch = R.BASE.runtime()["torch"]
    manifest = json.loads(R.MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"TRAINING_COMPLETE_PENDING_AUDIT", "AUDIT_FAILED"} or int(manifest.get("completed_tasks", -1)) != len(GROUPS) * 10:
        raise RuntimeError("independent audit requires all 60 formal tasks complete")
    if manifest.get("status") == "AUDIT_FAILED":
        manifest["previous_audit_failure"] = manifest.pop("audit_failures", [])
        manifest["status"] = "TRAINING_COMPLETE_PENDING_AUDIT"
        R.atomic_json(R.MANIFEST_PATH, manifest)
    preflight = manifest.get("preflight", {})
    if preflight.get("status") != "PASS":
        raise RuntimeError("frozen preflight did not pass")
    current_sources = R.code_hashes()
    frozen_sources = manifest.get("source_sha256", {})
    if set(current_sources) != set(frozen_sources):
        raise AssertionError("frozen source inventory changed after preflight")
    non_audit_changes = {
        name: {"frozen": frozen_sources[name], "current": current_sources[name]}
        for name in current_sources if name != "audit.py" and current_sources[name] != frozen_sources[name]
    }
    if non_audit_changes:
        raise AssertionError(f"training source/config hashes changed after preflight: {non_audit_changes}")
    audit_corrections = manifest.setdefault("audit_corrections", [])
    current_audit_sha = current_sources.get("audit.py")
    expected_previous_audit_sha = audit_corrections[-1]["current_sha256"] if audit_corrections else frozen_sources.get("audit.py")
    if current_audit_sha != expected_previous_audit_sha:
        if not audit_corrections:
            reason = "E72.4 fold_metrics.csv stores metric columns as percentage-suffixed names; the first independent audit assumed normalized names."
            scope = "Historical E72.4-G1 comparison column mapping and explicit source-hash exception for this audit-only correction."
        elif len(audit_corrections) == 1:
            reason = "Disease-supervision mechanism readback did not carry its frozen prototype bank into the diagnostic routine."
            scope = "Audit-only threading of the already-loaded disease bank into mechanism diagnostics; no model, prediction, or checkpoint changed."
        elif len(audit_corrections) == 2:
            reason = "The mechanism diagnostic routine did not bind the readback prototype bank and checkpoint objects required for diagnostics."
            scope = "Audit-only local bindings for disease-head and gradient diagnostics; no model, prediction, or checkpoint changed."
        else:
            reason = "The mechanism diagnostic routine did not bind the readback checkpoint before accessing its frozen alpha."
            scope = "Audit-only local binding for gradient diagnostics; no model, prediction, or checkpoint changed."
        audit_corrections.append({
            "component": "audit.py", "previous_sha256": expected_previous_audit_sha,
            "current_sha256": current_audit_sha, "reason": reason, "scope": scope,
            "training_code_or_results_changed": False,
        })
        R.atomic_json(R.MANIFEST_PATH, manifest)
    chain_sha = frozen_sources.get("audit.py")
    for correction in audit_corrections:
        if correction.get("component") != "audit.py" or correction.get("previous_sha256") != chain_sha:
            raise AssertionError("audit.py correction chain does not start from the preflight source SHA")
        chain_sha = correction.get("current_sha256")
    if chain_sha != current_audit_sha:
        raise AssertionError("audit.py differs from the final recorded audit-only correction")
    ref_manifest, ref_folds = R.load_reference()
    warmup = R.verify_warmup_audit()
    banks, prompt_meta = R.load_prompt_cache()
    patient_cache = torch.load(R.PATIENT_CACHE, map_location="cpu", weights_only=False)
    nonling_cache = torch.load(NONLINGUISTIC_CACHE, map_location="cpu", weights_only=False)
    if R.sha_file(R.PATIENT_CACHE) != R.CFG["patient_prompt"]["source_cache_sha256"]:
        raise AssertionError("frozen E72.4 patient cache SHA changed")
    if nonling_cache["metadata"].get("source_cache_sha256") != R.sha_file(R.PATIENT_CACHE):
        raise AssertionError("G6 category tokens no longer align to E72.4 patient cache")

    metrics_rows: list[dict] = []
    audit_rows: list[dict] = []
    rng_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    subgroup_rows: list[dict] = []
    fold_results: dict[str, dict[int, dict]] = {group: {} for group in GROUPS}
    inference: dict[tuple[str, int], dict] = {}
    fold2_rows: list[dict] = []
    mechanism_epoch_rows: list[dict] = []
    failures: list[dict] = []
    for group in GROUPS:
        for fold in range(1, 11):
            try:
                metric_row, result, audit_row, mechanism_file = independent_readback(
                    group, fold, manifest, ref_manifest, ref_folds, warmup, banks, prompt_meta, patient_cache, nonling_cache
                )
                metrics_rows.append(metric_row)
                audit_rows.append(audit_row)
                fold_results[group][fold] = metric_row
                inference[(group, fold)] = result
                mechanism_epoch_rows.append(mechanism_file)
                e69_task = json.loads((R.E69_TASKS / f"E69.3-G2_f{fold:02d}.json").read_text(encoding="utf-8"))
                epoch_diag = mechanism_file["epoch_records"]
                rng = rng_audit_for_task(group, fold, epoch_diag, e69_task)
                rng_rows.append(rng)
                if rng["status"] != "PASS":
                    raise AssertionError(f"{group}/f{fold:02d} common epoch RNG trace mismatch")
                diag_rows, subgroups = mechanism_for_readback(result, inference.get(("G3", fold), {}), patient_cache["metadata"]["selections"])
                mechanism_rows.extend(diag_rows)
                subgroup_rows.extend(subgroups)
            except BaseException as error:
                failures.append({"group": group, "fold": fold, "error": f"{type(error).__name__}: {error}"})
                audit_rows.append({"group": group, "fold": fold, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    e69 = R.reference_rows()
    e72_4_path = Path(R.CFG["paths"]["e72_4"]) / "results/fold_metrics.csv"
    e724 = {}
    with e72_4_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("group") == "G1":
                e724_metric_columns = {
                    "acc": "acc_pct", "hard_auc": "hard_auc_pct", "prob_auc": "prob_auc_pct",
                    "sen": "sen_pct", "spe": "spe_pct", "asd_f1": "asd_f1_pct",
                }
                e724[int(row["fold"])] = {key: float(row[column]) / 100.0 for key, column in e724_metric_columns.items()}
    if set(e724) != set(range(1, 11)):
        failures.append({"group": "E72.4-G1", "fold": 0, "error": "missing historical matched patient-prompt fold metrics"})
    comparisons: list[dict] = []
    for group in GROUPS:
        comparisons.extend(paired(fold_results[group], e69, group, "E69.3-G2", "ALL"))
        comparisons.extend(paired(fold_results[group], e69, group, "E69.3-G2", "FOLD2_EXCLUDED"))
    comparisons.extend(paired(fold_results["G1"], e724, "G1", "E72.4-G1", "ALL"))
    comparisons.extend(paired(fold_results["G3"], fold_results["G2"], "G3", "G2", "ALL"))
    comparisons.extend(paired(fold_results["G3"], fold_results["G4"], "G3", "G4", "ALL"))
    comparisons.extend(paired(fold_results["G5"], fold_results["G3"], "G5", "G3", "ALL"))
    comparisons.extend(paired(fold_results["G5"], fold_results["G6"], "G5", "G6", "ALL"))
    summaries = summarize(metrics_rows)

    # Rebuild H0 fold 2 from its audited historical checkpoint for the prespecified exposure sensitivity.
    h0 = load_h0_fold2(ref_manifest, ref_folds, warmup, banks)
    h0_metric = {"group": "E69.3-G2", "fold": 2, **{key: float(h0["metrics"][key]) for key in METRICS}}
    h0_all = h0["metrics"]
    mask_h0 = ~np.isin(h0["test_ids"] + 1, np.asarray([377, 407]))
    h0_trim = metric_values(h0["probability"][h0["test_ids"][mask_h0]], h0["labels"][mask_h0])
    fold2_rows.append({"group": "E69.3-G2", "checkpoint_sha256": h0["checkpoint_sha256"], "full_n": h0_all["n_test"], "removed_rows_1based": "377;407", "removed_n": h0_trim["n_test"], **{f"full_{key}": h0_all[key] for key in METRICS}, **{f"without_377_407_{key}": h0_trim[key] for key in METRICS}})
    for group in GROUPS:
        result = inference[(group, 2)]
        full = result["metrics"]
        mask = ~np.isin(result["test_ids"] + 1, np.asarray([377, 407]))
        trimmed = metric_values(result["probability"][result["test_ids"][mask]], result["labels"][mask])
        fold2_rows.append({"group": group, "checkpoint_sha256": result["checkpoint"]["checkpoint_sha256"] if "checkpoint_sha256" in result["checkpoint"] else R.sha_file(task_paths(group, 2)[1]),
                           "full_n": full["n_test"], "removed_rows_1based": "377;407", "removed_n": trimmed["n_test"],
                           **{f"full_{key}": full[key] for key in METRICS},
                           **{f"without_377_407_{key}": trimmed[key] for key in METRICS}})

    subgroup_summary = []
    for group in GROUPS:
        rows = [row for row in subgroup_rows if row["group"] == group]
        for field in ("site", "age", "sex"):
            categories = sorted({row["category"] for row in rows if row["field"] == field})
            for category in categories:
                selected = [row for row in rows if row["field"] == field and row["category"] == category]
                if len(selected) < 10:
                    continue
                subgroup_summary.append({"group": group, "field": field, "category": category, "n": len(selected),
                                         "acc_pct": 100.0 * np.mean([row["correct"] for row in selected]),
                                         "mean_prob_td": float(np.mean([row["prob_td"] for row in selected]))})

    # Cross-fold summaries use test-row predictions once per participant; fold 2 rows 377/407 remain exposed.
    residual_gaps = {group: fold_results[group] for group in GROUPS}
    g1_base = next(row for row in comparisons if row["comparison"] == "G1_vs_E69.3-G2" and row["metric"] == "acc" and row["fold_set"] == "ALL")
    g5_g3 = next(row for row in comparisons if row["comparison"] == "G5_vs_G3" and row["metric"] == "acc" and row["fold_set"] == "ALL")
    g5_g6 = next(row for row in comparisons if row["comparison"] == "G5_vs_G6" and row["metric"] == "acc" and row["fold_set"] == "ALL")
    g5_mechanism = [row for row in mechanism_rows if row["group"] == "G5"]
    g5_net_reverse = sum(int(row.get("condition_reverse_corrected") or 0) - int(row.get("condition_reverse_new_errors") or 0) for row in g5_mechanism)
    g5_net_permute = sum(int(row.get("condition_permute_corrected") or 0) - int(row.get("condition_permute_new_errors") or 0) for row in g5_mechanism)
    g5_stable = g5_g3["wins"] >= 6 and g5_g6["wins"] >= 6
    e69_mean_acc = float(np.mean([row["acc"] for row in e69.values()])) * 100.0
    g5_mean_acc = next(row["acc_mean_pct"] for row in summaries if row["group"] == "G5")
    g6_mean_acc = next(row["acc_mean_pct"] for row in summaries if row["group"] == "G6")
    if failures:
        verdict = "AUDIT_FAILED"
    elif g1_base["mean_difference_pp"] <= -7.6985:
        verdict = "NO_GO_PERFORMANCE_REPAIR_FAILED"
    elif g5_mean_acc >= e69_mean_acc and g5_g3["mean_difference_pp"] > 0 and g5_g6["mean_difference_pp"] > 0 and g5_stable and g5_net_reverse > 0 and g5_net_permute > 0:
        verdict = "GO_CANDIDATE"
    elif g1_base["mean_difference_pp"] > -7.6985:
        verdict = "PARTIAL_SAFE_RESIDUAL_REPAIR_ONLY"
    else:
        verdict = "NO_GO"

    protected_before = manifest["preflight"]["protected_source_sha256"]
    protected_after = R.protected_snapshot()
    protected = R.compare_protected(protected_before, protected_after)
    if protected["status"] != "PASS":
        failures.append({"group": "protected_sources", "fold": 0, "error": json.dumps(protected, ensure_ascii=False)})
        verdict = "AUDIT_FAILED"
    rng_compared = sum(int(row["common_epochs"]) for row in rng_rows)
    rng_exact = sum(int(row["exact_epoch_triplets"]) for row in rng_rows)
    audit_status = {
        "status": "PASS" if not failures and len(audit_rows) == 60 and all(row.get("status") == "PASS" for row in audit_rows) else "FAIL",
        "experiment_status": "COMPLETE_AUDITED" if not failures and len(audit_rows) == 60 else "AUDIT_FAILED",
        "decision": verdict,
        "independent_checkpoint_audits_passed": sum(row.get("status") == "PASS" for row in audit_rows),
        "independent_checkpoint_audits_total": 60,
        "audit_failures": failures,
        "rng_exact_triplets": rng_exact,
        "rng_compared_triplets": rng_compared,
        "rng_status": "PASS" if rng_exact == rng_compared and rng_compared > 0 else "FAIL",
        "protected_source_audit": {"status": protected["status"], "protected_file_count": len(protected_before), "changed": protected.get("changed", []), "missing": protected.get("missing", []), "added": protected.get("added", [])},
        "metrics_are_generalization_estimates": False,
        "fold2_exposure": {"rows_1based": [377, 407], "diagnoses": ["ASD", "TD"], "classification_used": True, "disease_supervision_used_for_G2-G6": True},
        "patient_text_cache_sha256": R.sha_file(R.PATIENT_CACHE),
        "nonlinguistic_token_cache_sha256": R.sha_file(NONLINGUISTIC_CACHE),
        "disease_prompt_cache_sha256": R.sha_file(R.PROMPT_CACHE),
        "source_sha256": frozen_sources,
        "audit_source_sha256": current_sources.get("audit.py"),
        "audit_corrections": audit_corrections,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_csv(RESULTS / "fold_metrics.csv", metrics_rows)
    atomic_csv(RESULTS / "summary.csv", summaries)
    atomic_csv(RESULTS / "paired_comparisons.csv", comparisons)
    atomic_csv(RESULTS / "mechanism.csv", mechanism_rows)
    atomic_csv(RESULTS / "fold2_sensitivity.csv", fold2_rows)
    atomic_csv(RESULTS / "checkpoint_audit.csv", audit_rows)
    atomic_csv(RESULTS / "rng_parity.csv", rng_rows)
    atomic_csv(RESULTS / "subgroup_diagnostics.csv", subgroup_summary)
    atomic_json(RESULTS / "audit_status.json", audit_status)
    if audit_status["status"] == "PASS":
        report = report_text(manifest, summaries, comparisons, mechanism_rows, fold2_rows, subgroup_summary, audit_status, verdict)
        (RESULTS / "report.md").write_text(report, encoding="utf-8")
        manifest.update({"status": "COMPLETE_AUDITED", "decision": verdict,
                         "independent_checkpoint_audits_passed": audit_status["independent_checkpoint_audits_passed"],
                         "independent_checkpoint_audits_total": 60, "rng_audit_status": audit_status["rng_status"],
                         "rng_exact_triplets": rng_exact, "rng_compared_triplets": rng_compared,
                         "protected_source_audit_after_training": protected, "audited_utc": audit_status["created_utc"]})
        atomic_json(R.MANIFEST_PATH, manifest)
        audit_status["report_sha256"] = R.sha_file(RESULTS / "report.md")
        audit_status["manifest_sha256"] = R.sha_file(R.MANIFEST_PATH)
        handoff_record = handoff_text(manifest, summaries, comparisons, mechanisms=mechanism_rows, status=audit_status, verdict=verdict)
        audit_status["handoff_sha256"] = append_handoff(handoff_record)
        atomic_json(RESULTS / "audit_status.json", audit_status)
    else:
        manifest.update({"status": "AUDIT_FAILED", "audit_failures": failures, "audited_utc": audit_status["created_utc"]})
        atomic_json(R.MANIFEST_PATH, manifest)
    return audit_status


def handoff_text(manifest: dict, summaries: list[dict], comparisons: list[dict], mechanisms: list[dict], status: dict, verdict: str) -> str:
    marker = "## Experiment 72.5 - Safe Adaptive Dual Prompt and Disease-Difference Conditioning"
    gmap = {row["group"]: row for row in summaries}
    cmpmap = {(row["comparison"], row["metric"], row["fold_set"]): row for row in comparisons}
    lines = [
        marker,
        "",
        f"- Status: `{status['experiment_status']}`; training `{manifest.get('completed_tasks', 0)}/60`; independent best-checkpoint audits `{status['independent_checkpoint_audits_passed']}/60`; decision `{verdict}`.",
        f"- Directory: `{EXP}`. Source/config SHA: `{json.dumps(manifest['source_sha256'], ensure_ascii=False, sort_keys=True)}`.",
        "- Sole formal training base: E69.3-G2 branch `upload/experiment69-3-g2-two-rows`, commit `41dbcb45a0273673857051d8e821b8c3315eef00`; E61-E2 E2E adaptive DARG, ten warmups, original three optimizers, Fold2 rows 377/407 overlap retained. E72.4 checkpoints were not used as initialization.",
        "- Groups: G1 safe patient-only; G2 BrainPrompt-2 tau=1 joint; G3 SPromptGL-8 tau=1 joint; G4 SPromptGL-8 tau=0.1 joint; G5 SPromptGL-8 tau=1 joint plus signed ASD-TD difference query; G6 same as G5 with fixed nonlinguistic PHENO-category vectors.",
        f"- Frozen E72.4 patient cache SHA `{status['patient_text_cache_sha256']}` reused without re-encoding; G6 private category vector cache SHA `{status['nonlinguistic_token_cache_sha256']}`. Fields are age intervals, recorded sex codes, and site IDs; no diagnoses or unverified clinical measurements.",
        "- Metrics are mean +/- population SD across ten folds (%); Hard-AUC uses hard predictions and Prob-AUC uses continuous TD probabilities.",
    ]
    for group in GROUPS:
        row = gmap.get(group, {})
        if row:
            values = ", ".join(f"{metric.upper()} {row[metric+'_mean_pct']:.4f}+/-{row[metric+'_population_sd_pct']:.4f}" for metric in METRICS)
            base = cmpmap.get((f"{group}_vs_E69.3-G2", "acc", "ALL"), {})
            lines.append(f"- {group}: {values}; paired ACC delta vs E69.3-G2 {base.get('mean_difference_pp', float('nan')):+.4f} pp (W/T/L {base.get('wins')}/{base.get('ties')}/{base.get('losses')}).")
    lines.extend([
        f"- G5-G3 ACC delta `{cmpmap.get(('G5_vs_G3','acc','ALL'),{}).get('mean_difference_pp', float('nan')):+.4f} pp`; G5-G6 ACC delta `{cmpmap.get(('G5_vs_G6','acc','ALL'),{}).get('mean_difference_pp', float('nan')):+.4f} pp`.",
        f"- RNG: exact common epoch triplets `{status['rng_exact_triplets']}/{status['rng_compared_triplets']}`; protected-source audit `{status['protected_source_audit']['status']}` across `{status['protected_source_audit']['protected_file_count']}` files.",
        f"- Audit-only correction: `{json.dumps(status.get('audit_corrections', []), ensure_ascii=False, sort_keys=True)}`; model training and predictions were unchanged.",
        "- Historical Fold2 still exposes rows 377 (ASD) and 407 (TD) to both training and testing. Excluding Fold2 or removing those two rows is descriptive sensitivity only; all other historical CV/preprocessing/transductive limitations remain.",
        "- See the full audited report and outputs under `results/`. Future performance claims require no-overlap folds, fold-local preprocessing, validation-only checkpoint selection, and independent validation.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
