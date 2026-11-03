#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import run as E60


HERE = Path(__file__).resolve().parent
RT = HERE / ".runtime"
RESULTS = HERE / "results"
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
GROUPS = [item["name"] for item in CONFIG["groups"]]
FORMAL_METRICS = ("acc", "hard_auc", "prob_auc", "spe", "sen", "ba", "asd_f1")
DISPLAY_METRICS = {
    "acc": "ACC",
    "hard_auc": "Hard-AUC",
    "prob_auc": "Probability-AUC",
    "spe": "SPE-TD",
    "sen": "SEN-ASD",
    "ba": "BA",
    "asd_f1": "ASD-F1",
}
SIGN_MATRIX = ((np.arange(1024, dtype=np.int16)[:, None] >> np.arange(10, dtype=np.int16)) & 1) * 2 - 1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_completed_audit() -> dict[str, Any]:
    manifest = read_json(HERE / "manifest.json")
    audit_path = RESULTS / "checkpoint_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("run python experiment/experiment60_dss_pen/audit.py --all before analysis")
    audit = read_json(audit_path)
    if audit.get("status") != "PASS" or audit.get("folds_audited") != 180:
        raise RuntimeError("all-group independent checkpoint audit must pass for 180 folds")
    if manifest.get("checkpoint_audit_status") != "PASS" or manifest.get("independently_audited_checkpoints") != 180:
        raise RuntimeError("manifest does not record 180 independently audited checkpoints")
    return manifest


def audit_group_variant(group: str, fold: int, checkpoint: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    from text_features import group_text_banks

    saved = checkpoint.get("semantic_module")
    if group == "E60_BASE_REPLAY":
        if saved is not None:
            raise AssertionError(f"R01 unexpectedly contains a semantic module in fold {fold}")
        return {"group": group, "fold": fold, "method_audit_status": "PASS", "variant": "base_without_semantic_module"}
    if not isinstance(saved, dict):
        raise AssertionError(f"semantic module state is missing: {group} fold {fold}")
    banks = group_text_banks(cache, group)
    module = E60.build_module(group, fold, banks, None)
    module.load_state_dict(saved, strict=True)
    checks: dict[str, bool] = {}
    config_group = next(item for item in CONFIG["groups"] if item["name"] == group)

    if group == "E60_ORIGINAL_PEN":
        checks["original_text_bank"] = torch.equal(module.original_text.detach().cpu(), banks["original"].cpu())
        checks["source_mlp_shape"] = tuple(module.mlp[0].weight.shape) == (128, 4096) and tuple(module.mlp[2].weight.shape) == (9, 128)
        variant = "original_spromptgl_pen_source_code_path"
    elif group in {"E60_BRAINPROMPT_LABEL", "E60_BRAINPROMPT_MEDICAL"}:
        expected_text = F.normalize(banks["brainprompt"].float().cpu(), dim=-1)
        checks["disease_text_bank"] = torch.equal(module.disease_text.detach().cpu(), expected_text)
        variant = "brainprompt_style_cosine_ce_adaptation"
    elif group in {"E60_SABER_DECISION", "E60_DISEASE_QUERY"}:
        retrieval = banks["patient_text"] if group == "E60_SABER_DECISION" else banks["medical"].flatten(0, 1)
        expected_disease = F.normalize(banks["disease_centroids"].float().cpu(), dim=-1)
        checks["disease_text_bank"] = torch.equal(module.disease_text.detach().cpu(), expected_disease)
        checks["retrieval_target"] = torch.equal(module.retrieval_text.detach().cpu(), retrieval.cpu())
        variant = "saber_patient_text_equation_reimplementation" if group == "E60_SABER_DECISION" else "modified_disease_text_query"
    else:
        expected_text = F.normalize(banks["dss"].float().cpu(), dim=-1)
        checks["text_bank"] = torch.equal(module.text_bank.detach().cpu(), expected_text)
        expected_mode = {
            "E60_DSS_SOFTMAX_ATTENTION": "softmax_attention",
            "E60_DSS_FORCE_MATCH": "simplex_sum_eq_one",
            "E60_DSS_FREE_SET": "nonnegative_box",
        }.get(group, "partial_simplex")
        checks["projection_mode"] = module.projection_mode == expected_mode
        expected_input_mask = (0.0, 0.0, 0.0, 1.0) if group == "E60_DSS_CORR_ONLY" else (1.0, 1.0, 1.0, 1.0)
        checks["input_slot_mask"] = tuple(float(x) for x in module.input_slot_mask.detach().cpu()) == expected_input_mask
        checks["configuration_flags"] = bool(config_group.get("set_loss", False)) == (group != "E60_DSS_NO_SET_LOSS")
        checks["configuration_residual"] = bool(config_group.get("correction", False)) == (group != "E60_DSS_NO_RESIDUAL")
        expected_output_mask = tuple(float(x) for x in config_group.get("output_slot_mask", (1.0, 1.0, 1.0, 1.0)))
        checks["output_slot_mask"] = expected_output_mask == ((0.0, 0.0, 0.0, 1.0) if group == "E60_DSS_CORR_ONLY" else (1.0, 1.0, 1.0, 1.0))
        if group == "E60_DSS_NO_SET_LOSS":
            checks["set_loss_removed"] = float(checkpoint["diagnostics"]["semantic"].get("set", math.nan)) == 0.0
        if group == "E60_DSS_NO_RESIDUAL":
            checks["residual_removed"] = float(checkpoint["diagnostics"]["semantic"].get("residual_schedule", math.nan)) == 0.0
        variant = str(config_group["bank"]) + ":" + expected_mode

    if not all(checks.values()):
        raise AssertionError(f"declared method variant failed checkpoint audit: {group} fold {fold}: {checks}")
    bank_key = next((key for key in ("text_bank", "original_text", "disease_text") if key in saved), None)
    bank_sha = None
    if bank_key:
        value = saved[bank_key].detach().cpu().contiguous().numpy()
        bank_sha = hashlib.sha256(value.tobytes()).hexdigest()
    parameter_count = sum(parameter.numel() for parameter in module.parameters())
    projection_mode = getattr(module, "projection_mode", None)
    del module
    return {
        "group": group,
        "fold": fold,
        "method_audit_status": "PASS",
        "variant": variant,
        "projection_mode": projection_mode,
        "verified_components": json.dumps(checks, sort_keys=True),
        "semantic_parameter_count": parameter_count,
        "stored_text_bank_sha256": bank_sha,
    }


def load_fold_metrics() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit_rows = read_json(RESULTS / "checkpoint_audit.json")["rows"]
    audit_by_group_fold = {(row["group"], int(row["fold"])): row for row in audit_rows}
    cache = E60.load_text_cache()
    method_audit_rows: list[dict[str, Any]] = []

    base_root = Path(CONFIG["base"]["base_source"])
    base_group = CONFIG["base"]["group"]
    base_csv = base_root / "results" / "fallback_wd_sweep" / "fold_metrics.csv"
    base_manifest = read_json(base_root / "manifest_fallback_wd_sweep.json")
    if base_manifest.get("status") != "COMPLETE_AUDITED" or base_manifest.get("independent_checkpoint_audits_passed") != 40:
        raise RuntimeError("historical 53.3 sweep is not fully audited")
    with base_csv.open(newline="", encoding="utf-8") as stream:
        reference = [row for row in csv.DictReader(stream) if row.get("group") == base_group]
    if len(reference) != 10:
        raise RuntimeError("historical selected-group fold results are incomplete")
    for old in reference:
        fold = int(old["fold"])
        historical_checkpoint = Path(CONFIG["base"]["historical_checkpoint_root"]) / f"{base_group}_f{fold:02d}.best.pt"
        if not historical_checkpoint.is_file():
            raise FileNotFoundError(f"historical checkpoint is missing: {historical_checkpoint}")
        historical_checkpoint_sha = sha_file(historical_checkpoint)
        if historical_checkpoint_sha != old["checkpoint_sha256"]:
            raise RuntimeError(f"historical checkpoint SHA mismatch: fold {fold}")
        saved = E60.torch_load(historical_checkpoint)
        r, state = E60.load_base()[0], E60.state_for_fold("E60_BASE_REPLAY", fold)
        state["model"].load_state_dict(saved["model"], strict=True)
        state["cam"].load_state_dict(saved["cam"], strict=True)
        with torch.no_grad():
            historical_output = r.forward_state(state, training=False, use_cadr=True)
            historical_metrics = r.classification_metrics(historical_output["logp"], state["test"], state["y"])
        historical_indices = state["test"].detach().cpu().numpy() if torch.is_tensor(state["test"]) else np.asarray(state["test"])
        historical_labels = state["y"].detach().cpu().numpy()[historical_indices]
        historical_probabilities = np.asarray(historical_metrics["prob"])
        auc_asd_positive = float(roc_auc_score(historical_labels == 0, historical_probabilities[:, 0]))
        if not math.isclose(auc_asd_positive, float(old["prob_auc"]), rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"historical ASD-positive probability AUC mismatch after readback: fold {fold}")
        row: dict[str, Any] = {
            "group_id": "R00",
            "group": "E60_REF_533_G3",
            "fold": fold,
            "status": "HISTORICAL_READ_ONLY_REFERENCE",
            "best_epoch": int(old["best_epoch"]),
            "epochs_executed": int(old["epochs_executed"]),
            "checkpoint_sha256": old["checkpoint_sha256"],
            "checkpoint_audit": old["checkpoint_audit"],
            "cadr_weight_decay": float(old["cadr_weight_decay"]),
            "gfr_weight_decay": float(old["gfr_weight_decay"]),
            "cam_weight_decay": float(old["cam_weight_decay"]),
            "prob_auc_positive_class": "ASD (0); historical complementary coding equivalence",
        }
        row.update({key: 100.0 * float(old[key]) for key in FORMAL_METRICS})
        row["prob_auc"] = 100.0 * auc_asd_positive
        rows.append(row)

    for group_index, group in enumerate(GROUPS, start=1):
        group_root = RT / "groups" / group
        for fold in range(1, 11):
            task_path = group_root / "tasks" / f"{group}_f{fold:02d}.json"
            checkpoint_path = group_root / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
            task = read_json(task_path)
            if task.get("status") != "COMPLETE" or not checkpoint_path.is_file():
                raise RuntimeError(f"incomplete task/checkpoint: {group} fold {fold}")
            audit = audit_by_group_fold.get((group, fold))
            if audit is None or audit.get("status") != "PASS":
                raise RuntimeError(f"independent audit missing: {group} fold {fold}")
            checkpoint_sha = sha_file(checkpoint_path)
            if checkpoint_sha != task.get("best_sha256") or checkpoint_sha != audit.get("checkpoint_sha256"):
                raise RuntimeError(f"checkpoint SHA mismatch while assembling results: {group} fold {fold}")
            checkpoint = E60.torch_load(checkpoint_path)
            method_audit = audit_group_variant(group, fold, checkpoint, cache)
            method_audit_rows.append(method_audit)
            prediction_record = checkpoint["test_predictions"]
            labels = np.asarray(prediction_record["labels"], dtype=np.int64)
            probabilities = np.asarray(prediction_record["probabilities_asd_td"], dtype=np.float64)
            if probabilities.shape != (len(labels), 2):
                raise RuntimeError(f"unexpected test probability shape: {group} fold {fold}")
            auc_asd_positive = float(roc_auc_score(labels == 0, probabilities[:, 0]))
            if not math.isclose(auc_asd_positive, float(audit["metrics"]["prob_auc"]), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"ASD-positive probability AUC differs from independently audited metric: {group} fold {fold}")
            row = {
                "group_id": f"R{group_index:02d}",
                "group": group,
                "fold": fold,
                "status": task["status"],
                "best_epoch": int(task["best_epoch"]),
                "epochs_executed": int(task["epochs_executed"]),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_audit": audit["status"],
                "cadr_weight_decay": 0.01,
                "gfr_weight_decay": 0.11,
                "cam_weight_decay": 0.11,
                "prob_auc_positive_class": "ASD (0)",
            }
            row.update({key: 100.0 * float(audit["metrics"][key]) for key in FORMAL_METRICS})
            row["prob_auc"] = 100.0 * auc_asd_positive
            row.update({f"confusion_{key}": int(value) for key, value in audit["confusion"].items()})
            rows.append(row)

    rows.sort(key=lambda row: (row["group_id"], row["fold"]))
    checkpoint_rows = []
    for row in audit_rows:
        method_audit = next(item for item in method_audit_rows if item["group"] == row["group"] and item["fold"] == row["fold"])
        checkpoint_rows.append({
            "group": row["group"],
            "fold": row["fold"],
            "status": row["status"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "best_epoch": row["best_epoch"],
            "epochs_executed": row["epochs_executed"],
            "readback": row["readback"],
            "historical_metrics_match": row["historical_metrics_match"],
            "historical_state_match": row["historical_state_match"],
            "method_audit_status": method_audit["method_audit_status"],
            "method_variant": method_audit["variant"],
        })
    return rows, checkpoint_rows, method_audit_rows


def write_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for group in ["E60_REF_533_G3", *GROUPS]:
        subset = [row for row in rows if row["group"] == group]
        record: dict[str, Any] = {
            "group": group,
            "n_folds": len(subset),
            "best_epoch_mean": float(np.mean([row["best_epoch"] for row in subset])),
            "epochs_executed_mean": float(np.mean([row["epochs_executed"] for row in subset])),
            "checkpoint_audit": "PASS" if len(subset) == 10 and all(row["checkpoint_audit"] in {"PASS", "HISTORICAL_CHECKPOINT_AUDIT_PASS"} for row in subset) else "INCOMPLETE",
        }
        for metric in FORMAL_METRICS:
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            record[f"{metric}_mean_pct"] = float(values.mean())
            record[f"{metric}_sd_population_pct"] = float(values.std(ddof=0))
            record[f"{metric}_mean_plus_minus_sd_pct"] = f"{values.mean():.3f} ± {values.std(ddof=0):.3f}"
        summary.append(record)
    return summary


def paired_stats(difference: np.ndarray, seed: int) -> dict[str, Any]:
    if difference.shape != (10,):
        raise ValueError(f"expected ten paired fold differences, got {difference.shape}")
    observed = float(difference.mean())
    permuted = (SIGN_MATRIX * difference[None, :]).mean(axis=1)
    p_value = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))
    rng = np.random.default_rng(seed)
    sampled = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        "mean_difference_pp": observed,
        "sd_difference_population_pp": float(difference.std(ddof=0)),
        "wins": int((difference > 1e-12).sum()),
        "ties": int((np.abs(difference) <= 1e-12).sum()),
        "losses": int((difference < -1e-12).sum()),
        "exact_sign_flip_p_two_sided": p_value,
        "paired_bootstrap_95ci_low_pp": float(low),
        "paired_bootstrap_95ci_high_pp": float(high),
        "fold_differences_pp": json.dumps([round(float(value), 8) for value in difference], separators=(",", ":")),
    }


def holm_adjust(records: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        families.setdefault((row["family"], row["metric"]), []).append(row)
    for family_rows in families.values():
        ordered = sorted(family_rows, key=lambda item: item["exact_sign_flip_p_two_sided"])
        count = len(ordered)
        running = 0.0
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * row["exact_sign_flip_p_two_sided"])
            running = max(running, adjusted)
            row["holm_adjusted_p"] = running


def write_paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group = {group: sorted([row for row in rows if row["group"] == group], key=lambda row: row["fold"]) for group in ["E60_REF_533_G3", *GROUPS]}
    comparisons: list[tuple[str, str, str, str]] = []
    for group in GROUPS:
        comparisons.append(("all_formal_vs_R00", group, "E60_REF_533_G3", "R00"))
    direct_and_dss = [group for group in GROUPS[1:6]] + GROUPS[7:]
    for group in direct_and_dss:
        comparisons.append(("R07_vs_direct_and_DSS_controls", "E60_DSS_FULL", group, "R07"))
    result: list[dict[str, Any]] = []
    seed_base = int(CONFIG["evaluation"]["paired_tests"]["bootstrap_seed"])
    for family, left, right, family_label in comparisons:
        left_rows, right_rows = by_group[left], by_group[right]
        if len(left_rows) != 10 or len(right_rows) != 10:
            raise RuntimeError(f"incomplete paired comparison {left} vs {right}")
        for metric_index, metric in enumerate(FORMAL_METRICS):
            difference = np.asarray([a[metric] - b[metric] for a, b in zip(left_rows, right_rows)], dtype=np.float64)
            stats = paired_stats(difference, seed_base + metric_index)
            result.append({
                "family": family,
                "family_id": family_label,
                "group_a": left,
                "group_b": right,
                "metric": metric,
                "metric_label": DISPLAY_METRICS[metric],
                **stats,
            })
    holm_adjust(result)
    return result


def _load_r07_checkpoint(fold: int, cache: dict[str, Any], banks: dict[str, torch.Tensor]):
    group = "E60_DSS_FULL"
    path = RT / "groups" / group / "checkpoints" / f"{group}_f{fold:02d}.best.pt"
    checkpoint = E60.torch_load(path)
    state = E60.state_for_fold(group, fold)
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    module = E60.build_module(group, fold, banks, None)
    module.load_state_dict(checkpoint["semantic_module"], strict=True)
    state["model"].eval()
    state["cam"].eval()
    module.eval()
    return E60.load_base()[0], state, module, checkpoint


def _intervention_forward(state, module, banks, checkpoint, intervention: str):
    original_correct = module.correct
    original_bank = module.text_bank.detach().clone()
    try:
        if intervention == "wrong_semantics":
            wrong = E60.torch_load(HERE / ".runtime" / "text_embeddings.pt")
            from text_features import group_text_banks
            replacement = group_text_banks(wrong, "E60_DSS_WRONG_SEMANTICS")["dss"].to(module.text_bank.device)
            module.text_bank.copy_(replacement)
        elif intervention == "single_prototype":
            from text_features import group_text_banks
            replacement = group_text_banks(banks["_cache"], "E60_DSS_SINGLE_PROTOTYPE")["dss"].to(module.text_bank.device)
            module.text_bank.copy_(replacement)

        if intervention == "residual_off":
            def no_residual(h, output, epoch, **kwargs):
                kwargs["residual_enabled"] = False
                return original_correct(h, output, epoch, **kwargs)
            module.correct = no_residual
        elif intervention == "reject_all":
            def force_reject(h, output, epoch, **kwargs):
                corrected, detail = original_correct(h, output, epoch, **kwargs)
                detail["match_gate"] = torch.zeros_like(detail["match_gate"])
                detail["applied_correction"] = torch.zeros_like(detail["applied_correction"])
                return h, detail
            module.correct = force_reject
        elif intervention == "corr_off":
            def no_corr(h, output, epoch, **kwargs):
                kwargs["slot_mask"] = (1.0, 1.0, 1.0, 0.0)
                return original_correct(h, output, epoch, **kwargs)
            module.correct = no_corr
        elif intervention == "noncorr_off":
            def no_noncorr(h, output, epoch, **kwargs):
                kwargs["slot_mask"] = (0.0, 0.0, 0.0, 1.0)
                return original_correct(h, output, epoch, **kwargs)
            module.correct = no_noncorr
        with torch.no_grad():
            result = E60.base_or_group_forward(
                "E60_DSS_FULL", state, module, int(checkpoint["best_epoch"]), False, True,
                banks, checkpoint["extra_loss_weights"],
            )
        return result
    finally:
        module.correct = original_correct
        module.text_bank.copy_(original_bank)


def run_r07_mechanisms() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from text_features import group_text_banks

    cache = E60.load_text_cache()
    banks = group_text_banks(cache, "E60_DSS_FULL")
    banks["_cache"] = cache
    patient_ids = cache["patient_category_ids"].long().numpy()
    labels_table = {"ASD": 0, "TD": 1}
    assignments: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    fold_mechanisms: dict[str, Any] = {}
    knowledge = read_json(HERE / "knowledge_base.json")
    concept_names = [item["concept_id"] for cls in ("ASD", "TD") for item in knowledge["classes"][cls]]

    for fold in range(1, 11):
        print(f"mechanism replay R07 fold {fold}/10", flush=True)
        r, state, module, checkpoint = _load_r07_checkpoint(fold, cache, banks)
        with torch.enable_grad():
            baseline = E60.base_or_group_forward(
                "E60_DSS_FULL", state, module, int(checkpoint["best_epoch"]), False, True,
                banks, checkpoint["extra_loss_weights"],
            )
            semantic = baseline["semantic"]
            parameter_items = list(module.named_parameters())
            gradients = torch.autograd.grad(
                baseline["cls"], [parameter for _, parameter in parameter_items],
                allow_unused=True,
            )
            gradient_norms: dict[str, float] = {}
            for (name, _), grad in zip(parameter_items, gradients):
                if grad is not None:
                    prefix = name.split(".")[0]
                    gradient_norms[prefix] = gradient_norms.get(prefix, 0.0) + float(grad.detach().float().square().sum().cpu())
            gradient_norms = {key: math.sqrt(value) for key, value in gradient_norms.items()}

        test_idx = (
            state["test"].detach().cpu().numpy().astype(np.int64)
            if torch.is_tensor(state["test"])
            else np.asarray(state["test"], dtype=np.int64)
        )
        test_tensor = torch.as_tensor(test_idx, dtype=torch.long, device=semantic["h_used"].device)
        labels = state["y"].detach().cpu().numpy().astype(np.int64)
        test_y = labels[test_idx]
        output = semantic
        coeff = output["coefficients"].detach().cpu().numpy()[test_idx]
        all_coeff = output["coefficients"].detach().cpu().numpy()
        dist = output["distances"].detach().cpu().numpy()[test_idx]
        gate = output["match_gate"].detach().cpu().numpy()[test_idx]
        residual = output["applied_correction"].detach().float().norm(dim=-1).cpu().numpy()[test_idx]
        cam_norm = output["h"].detach().float().norm(dim=-1).cpu().numpy()[test_idx]
        minimum_distance = dist.min(axis=1)
        prediction_metric = r.classification_metrics(baseline["logp"], state["test"], state["y"])
        fold_record: dict[str, Any] = {
            "fold": fold,
            "n_test": int(len(test_idx)),
            "reject_rate": float(np.mean(gate == 0.0)),
            "accepted_fraction": float(np.mean(gate > 0.0)),
            "mean_gate": float(gate.mean()),
            "mean_minimum_distance": float(minimum_distance.mean()),
            "mean_residual_norm": float(residual.mean()),
            "max_residual_norm": float(residual.max(initial=0.0)),
            "mean_residual_to_CAM_norm_ratio": float((residual / np.maximum(cam_norm, 1e-8)).mean()),
            "rejected_residual_max_norm": float(residual[gate == 0.0].max(initial=0.0)),
            "coefficient_minimum_all_nodes": float(all_coeff.min()),
            "coefficient_mass_maximum_all_nodes": float(all_coeff.sum(axis=-1).max()),
            "partial_simplex_constraints_pass": bool(all_coeff.min() >= -1e-7 and all_coeff.sum(axis=-1).max() <= 1.0 + 1e-6),
            "classification_gradient_norm_total": float(math.sqrt(sum(value * value for value in gradient_norms.values()))),
            "classification_gradient_norm_patient_projector": gradient_norms.get("patient_projector", 0.0),
            "classification_gradient_norm_text_adapter": gradient_norms.get("text_adapter", 0.0),
            "classification_gradient_norm_correction": gradient_norms.get("correction", 0.0),
        }
        if not fold_record["partial_simplex_constraints_pass"]:
            raise AssertionError(f"R07 projection violates the frozen partial-simplex constraint in fold {fold}")
        if fold_record["rejected_residual_max_norm"] != 0.0:
            raise AssertionError(f"R07 applied a nonzero residual to rejected patients in fold {fold}")
        for class_name, class_index in labels_table.items():
            subject_mask = (test_y == class_index) & (gate > 0.0)
            class_subject_count = int((test_y == class_index).sum())
            class_coeff = coeff[subject_mask, class_index, :]
            mass = class_coeff.sum(axis=1) if len(class_coeff) else np.asarray([], dtype=np.float64)
            aggregate = class_coeff.sum(axis=0) if len(class_coeff) else np.zeros(coeff.shape[-1])
            proportions = aggregate / aggregate.sum() if aggregate.sum() > 0 else np.zeros_like(aggregate)
            patterns = [tuple(np.flatnonzero(row > 1e-6).tolist()) for row in class_coeff]
            patterns = [pattern for pattern in patterns if pattern]
            if patterns:
                unique, counts = np.unique(np.asarray([str(pattern) for pattern in patterns]), return_counts=True)
                nonmodal = (len(patterns) - float(counts.max())) / max(class_subject_count, 1)
                pattern_count = int(len(unique))
            else:
                nonmodal = 0.0
                pattern_count = 0
            fold_record[f"{class_name.lower()}_accepted_subjects"] = int(subject_mask.sum())
            fold_record[f"{class_name.lower()}_mean_coefficient_mass"] = float(mass.mean()) if len(mass) else 0.0
            fold_record[f"{class_name.lower()}_distinct_nonempty_patterns"] = pattern_count
            fold_record[f"{class_name.lower()}_nonmodal_pattern_fraction"] = nonmodal
            fold_record[f"{class_name.lower()}_concepts_over_10pct_mass"] = int((proportions >= 0.10).sum())
            fold_record[f"{class_name.lower()}_max_concept_mass_fraction"] = float(proportions.max(initial=0.0))
            fold_record[f"{class_name.lower()}_heterogeneity_pass"] = bool(
                (proportions >= 0.10).sum() >= 2 and proportions.max(initial=0.0) <= 0.80 and nonmodal >= 0.20
            ) if class_name == "ASD" else None
            for concept_index, share in enumerate(proportions):
                fold_record[f"{class_name.lower()}_concept_{concept_index + 1}_mass_fraction"] = float(share)

        fold_record["asd_empty_match_fraction"] = float(np.mean(gate[test_y == 0] == 0.0)) if np.any(test_y == 0) else math.nan
        fold_record["td_empty_match_fraction"] = float(np.mean(gate[test_y == 1] == 0.0)) if np.any(test_y == 1) else math.nan
        fold_record["asd_target_distance_mean"] = float(dist[test_y == 0, 0].mean()) if np.any(test_y == 0) else math.nan
        fold_record["td_target_distance_mean"] = float(dist[test_y == 1, 1].mean()) if np.any(test_y == 1) else math.nan
        fold_record["audit_ACC_pct"] = 100.0 * float(prediction_metric["acc"])
        fold_record["audit_Probability_AUC_pct"] = 100.0 * float(prediction_metric["prob_auc"])
        fold_record["asd_heterogeneity_pass"] = fold_record["asd_heterogeneity_pass"]
        diagnostic_rows.append({"record_type": "R07_fold_semantic", **fold_record})
        fold_mechanisms[str(fold)] = fold_record

        for local_index, patient_index in enumerate(test_idx):
            row: dict[str, Any] = {
                "fold": fold,
                "subject_row_index": int(patient_index),
                "label": "ASD" if test_y[local_index] == 0 else "TD",
                "match_gate": float(gate[local_index]),
                "empty_match": bool(gate[local_index] == 0.0),
                "minimum_distance": float(minimum_distance[local_index]),
                "distance_ASD": float(dist[local_index, 0]),
                "distance_TD": float(dist[local_index, 1]),
                "coefficient_mass_ASD": float(coeff[local_index, 0].sum()),
                "coefficient_mass_TD": float(coeff[local_index, 1].sum()),
                "applied_residual_norm": float(residual[local_index].mean()),
                "age_category": int(patient_ids[patient_index, 0]),
                "site_category": int(patient_ids[patient_index, 1]),
                "sex_category": int(patient_ids[patient_index, 2]),
            }
            for class_index, class_name in enumerate(("ASD", "TD")):
                for concept_index, value in enumerate(coeff[local_index, class_index]):
                    row[f"{class_name.lower()}_coefficient_{concept_index + 1}"] = float(value)
                    if concept_index < len(concept_names):
                        row[f"{class_name.lower()}_concept_id_{concept_index + 1}"] = concept_names[class_index * 6 + concept_index]
            assignments.append(row)

        for feature_index, feature_name in ((0, "age"), (1, "site"), (2, "sex")):
            values = patient_ids[test_idx, feature_index]
            for category in np.unique(values):
                selected = values == category
                if int(selected.sum()) < 5:
                    continue
                diagnostic_rows.append({
                    "record_type": "R07_reject_subgroup",
                    "fold": fold,
                    "subgroup": feature_name,
                    "category": int(category),
                    "n": int(selected.sum()),
                    "rejected_n": int((gate[selected] == 0.0).sum()),
                    "reject_rate": float(np.mean(gate[selected] == 0.0)),
                    "mean_minimum_distance": float(minimum_distance[selected].mean()),
                })
        for class_index, class_name in enumerate(("ASD", "TD")):
            selected = test_y == class_index
            if np.any(selected):
                diagnostic_rows.append({
                    "record_type": "R07_reject_subgroup",
                    "fold": fold,
                    "subgroup": "diagnosis",
                    "category": class_name,
                    "n": int(selected.sum()),
                    "rejected_n": int((gate[selected] == 0.0).sum()),
                    "reject_rate": float(np.mean(gate[selected] == 0.0)),
                    "mean_minimum_distance": float(minimum_distance[selected].mean()),
                })
        q_edges = np.quantile(minimum_distance, [0.25, 0.5, 0.75])
        q_bins = np.digitize(minimum_distance, q_edges, right=True)
        for quartile in range(4):
            selected = q_bins == quartile
            if np.any(selected):
                diagnostic_rows.append({
                    "record_type": "R07_reject_subgroup",
                    "fold": fold,
                    "subgroup": "semantic_match_quality_quartile",
                    "category": quartile + 1,
                    "n": int(selected.sum()),
                    "rejected_n": int((gate[selected] == 0.0).sum()),
                    "reject_rate": float(np.mean(gate[selected] == 0.0)),
                    "mean_minimum_distance": float(minimum_distance[selected].mean()),
                })

        base_probs = np.asarray(prediction_metric["prob"])[:, 0]
        base_labels = np.asarray(prediction_metric["pred"], dtype=np.int64)
        for intervention in ("residual_off", "wrong_semantics", "single_prototype", "reject_all", "corr_off", "noncorr_off"):
            altered = _intervention_forward(state, module, banks, checkpoint, intervention)
            metrics = r.classification_metrics(altered["logp"], state["test"], state["y"])
            probs = np.asarray(metrics["prob"])[:, 0]
            predicted = np.asarray(metrics["pred"], dtype=np.int64)
            logit_delta = (altered["logp"].detach() - baseline["logp"].detach()).abs()
            intervention_rows.append({
                "fold": fold,
                "intervention": intervention,
                "n_test": int(len(test_idx)),
                "ACC_pct": 100.0 * float(metrics["acc"]),
                "Probability_AUC_pct": 100.0 * float(metrics["prob_auc"]),
                "Hard_AUC_pct": 100.0 * float(metrics["hard_auc"]),
                "delta_ACC_pp_vs_R07": 100.0 * float(metrics["acc"] - prediction_metric["acc"]),
                "mean_abs_ASD_probability_shift": float(np.mean(np.abs(probs - base_probs))),
                "flipped_labels": int(np.sum(predicted != base_labels)),
                "logit_mean_abs_delta": float(logit_delta.mean().cpu()),
                "logit_max_abs_delta": float(logit_delta.max().cpu()),
                "cam_mean_abs_change_norm": float((altered["semantic"]["h_used"][test_tensor] - baseline["semantic"]["h_used"][test_tensor]).detach().float().norm(dim=-1).mean().cpu()),
                "cam_relative_change_to_baseline_norm": float((
                    (altered["semantic"]["h_used"][test_tensor] - baseline["semantic"]["h_used"][test_tensor]).detach().float().norm(dim=-1)
                    / baseline["semantic"]["h_used"][test_tensor].detach().float().norm(dim=-1).clamp_min(1e-8)
                ).mean().cpu()),
                "finite": bool(torch.isfinite(altered["logp"]).all().item()),
                "logits_nonidentical": bool(float(logit_delta.max().cpu()) > 0.0),
            })

        raw_bank = F.normalize(module.text_bank.detach(), dim=-1).cpu()
        adapted = F.normalize(module.text_adapter(module.text_bank).detach(), dim=-1).cpu()
        all_vectors = adapted.flatten(0, 1)
        within_asd = (adapted[0] @ adapted[0].T)[~torch.eye(6, dtype=torch.bool)]
        within_td = (adapted[1] @ adapted[1].T)[~torch.eye(6, dtype=torch.bool)]
        between = adapted[0] @ adapted[1].T
        center_asd = F.normalize(adapted[0].mean(0), dim=0)
        center_td = F.normalize(adapted[1].mean(0), dim=0)
        diagnostic_rows.append({
            "record_type": "R07_prototype_geometry",
            "fold": fold,
            "raw_text_rank": int(torch.linalg.matrix_rank(raw_bank.flatten(0, 1)).item()),
            "adapted_all_rank": int(torch.linalg.matrix_rank(all_vectors).item()),
            "adapted_ASD_rank": int(torch.linalg.matrix_rank(adapted[0]).item()),
            "adapted_TD_rank": int(torch.linalg.matrix_rank(adapted[1]).item()),
            "within_ASD_cosine_mean": float(within_asd.mean()),
            "within_TD_cosine_mean": float(within_td.mean()),
            "between_class_cosine_mean": float(between.mean()),
            "class_center_cosine": float(center_asd @ center_td),
            "prototype_min_cosine": float((all_vectors @ all_vectors.T)[~torch.eye(12, dtype=torch.bool)].min()),
        })

    write_csv(RESULTS / "semantic_patient_assignments.csv", assignments)
    return diagnostic_rows, intervention_rows, assignments, fold_mechanisms


def read_metric_group(rows: list[dict[str, Any]], group: str, metric: str) -> np.ndarray:
    selected = sorted((row for row in rows if row["group"] == group), key=lambda row: row["fold"])
    return np.asarray([row[metric] for row in selected], dtype=np.float64)


def gate_comparison(comparisons: list[dict[str, Any]], left: str, right: str) -> tuple[bool, dict[str, Any]]:
    rows = [row for row in comparisons if row["group_a"] == left and row["group_b"] == right]
    by_metric = {row["metric"]: row for row in rows}
    acc = by_metric["acc"]
    sen = by_metric["sen"]
    passed = (
        acc["mean_difference_pp"] >= 0.3
        and acc["wins"] >= 7
        and acc["paired_bootstrap_95ci_low_pp"] > 0.0
        and sen["mean_difference_pp"] >= -1.0
    )
    return passed, {
        "delta_ACC_pp": acc["mean_difference_pp"],
        "ACC_wins_ties_losses": [acc["wins"], acc["ties"], acc["losses"]],
        "ACC_paired_bootstrap_95CI_pp": [acc["paired_bootstrap_95ci_low_pp"], acc["paired_bootstrap_95ci_high_pp"]],
        "delta_SEN_ASD_pp": sen["mean_difference_pp"],
    }


def compact_gate_evidence(evidence: dict[str, Any]) -> str:
    wins, ties, losses = evidence["ACC_wins_ties_losses"]
    low, high = evidence["ACC_paired_bootstrap_95CI_pp"]
    return (
        f"ΔACC {evidence['delta_ACC_pp']:+.3f} pp; W/T/L {wins}/{ties}/{losses}; "
        f"95% CI [{low:+.3f}, {high:+.3f}]; ΔSEN {evidence['delta_SEN_ASD_pp']:+.3f} pp"
    )


def write_report(
    manifest: dict[str, Any], summary: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]], interventions: list[dict[str, Any]],
    assignments: list[dict[str, Any]], fold_mechanisms: dict[str, Any],
) -> list[tuple[str, bool, str]]:
    group_checks: dict[str, bool] = {}
    group_gate_evidence: dict[str, dict[str, Any]] = {}
    for control in ["E60_REF_533_G3", "E60_ORIGINAL_PEN", "E60_BRAINPROMPT_LABEL", "E60_BRAINPROMPT_MEDICAL", "E60_SABER_DECISION", "E60_DISEASE_QUERY", *GROUPS[7:]]:
        group_checks[control], group_gate_evidence[control] = gate_comparison(comparisons, "E60_DSS_FULL", control)
    parity = manifest.get("r01_parity_status") == "PASS" and manifest.get("r01_audited_folds") == 10
    diag_r00, diag_r00_evidence = gate_comparison(comparisons, "E60_DSS_FULL", "E60_REF_533_G3")
    diag_r12, diag_r12_evidence = gate_comparison(comparisons, "E60_DSS_FULL", "E60_DSS_SINGLE_PROTOTYPE")
    diag_r13, diag_r13_evidence = gate_comparison(comparisons, "E60_DSS_FULL", "E60_DSS_SOFTMAX_ATTENTION")
    r07_folds = [fold_mechanisms[str(fold)] for fold in range(1, 11)]
    structural_ok = sum(bool(row["asd_heterogeneity_pass"]) for row in r07_folds) >= 8
    set_projection = diag_r12 and diag_r13 and structural_ok
    patient_heterogeneity = structural_ok
    semantic_content = all(group_checks[group] for group in (
        "E60_DSS_CLASS_ONLY", "E60_DSS_WRONG_SEMANTICS", "E60_DSS_RANDOM_ANCHORS", "E60_DSS_LEARNED_PROTOTYPES"
    ))
    intervention_by_name: dict[str, list[dict[str, Any]]] = {}
    for row in interventions:
        intervention_by_name.setdefault(row["intervention"], []).append(row)
    decision_injection = all(
        intervention_by_name[name]
        and all(row["finite"] for row in intervention_by_name[name])
        and any(row["logits_nonidentical"] for row in intervention_by_name[name])
        for name in ("residual_off", "wrong_semantics")
    )
    decision_evidence = {}
    for name in ("residual_off", "wrong_semantics"):
        rows_for_intervention = intervention_by_name[name]
        decision_evidence[name] = {
            "folds": len(rows_for_intervention),
            "max_abs_logit_change": max(row["logit_max_abs_delta"] for row in rows_for_intervention),
            "mean_abs_ASD_probability_shift": float(np.mean([row["mean_abs_ASD_probability_shift"] for row in rows_for_intervention])),
            "total_hard_label_flips": sum(int(row["flipped_labels"]) for row in rows_for_intervention),
            "all_finite": all(row["finite"] for row in rows_for_intervention),
            "any_nonidentical_logits": any(row["logits_nonidentical"] for row in rows_for_intervention),
        }
    vs_brainprompt = group_checks["E60_BRAINPROMPT_LABEL"] and group_checks["E60_BRAINPROMPT_MEDICAL"]
    vs_saber = group_checks["E60_SABER_DECISION"] and group_checks["E60_DISEASE_QUERY"]
    all_checkpoint_audits = (
        manifest.get("checkpoint_audit_status") == "PASS"
        and manifest.get("independently_audited_checkpoints") == 180
        and manifest.get("method_audit_status") == "PASS"
        and manifest.get("method_audited_checkpoints") == 180
    )
    candidate = all((parity, diag_r00, set_projection, patient_heterogeneity, semantic_content, decision_injection, vs_brainprompt, vs_saber, all_checkpoint_audits))
    gate_rows = [
        ("GO_BASE_PARITY", parity, "R01 10/10 checkpoint state and prediction parity"),
        ("GO_DIAGNOSTIC", diag_r00, compact_gate_evidence(diag_r00_evidence)),
        ("GO_SET_PROJECTION", set_projection, f"R12: {compact_gate_evidence(diag_r12_evidence)}; R13: {compact_gate_evidence(diag_r13_evidence)}; heterogeneity {sum(bool(row['asd_heterogeneity_pass']) for row in r07_folds)}/10 folds"),
        ("GO_PATIENT_HETEROGENEITY", patient_heterogeneity, f"frozen criteria met in {sum(bool(row['asd_heterogeneity_pass']) for row in r07_folds)}/10 folds"),
        ("GO_SEMANTIC_CONTENT", semantic_content, "; ".join(f"{group.replace('E60_DSS_', '')}: {compact_gate_evidence(group_gate_evidence[group])}" for group in ("E60_DSS_CLASS_ONLY", "E60_DSS_WRONG_SEMANTICS", "E60_DSS_RANDOM_ANCHORS", "E60_DSS_LEARNED_PROTOTYPES"))),
        ("GO_DECISION_INJECTION", decision_injection, "; ".join(f"{name}: max |Δlogit|={decision_evidence[name]['max_abs_logit_change']:.3g}, mean |ΔP_ASD|={decision_evidence[name]['mean_abs_ASD_probability_shift']:.4g}, flips={decision_evidence[name]['total_hard_label_flips']}" for name in ("residual_off", "wrong_semantics"))),
        ("GO_VS_BRAINPROMPT", vs_brainprompt, f"R03: {compact_gate_evidence(group_gate_evidence['E60_BRAINPROMPT_LABEL'])}; R04: {compact_gate_evidence(group_gate_evidence['E60_BRAINPROMPT_MEDICAL'])}"),
        ("GO_VS_SABER", vs_saber, f"R05: {compact_gate_evidence(group_gate_evidence['E60_SABER_DECISION'])}; R06: {compact_gate_evidence(group_gate_evidence['E60_DISEASE_QUERY'])}"),
        ("GO_MICCAI_CANDIDATE", candidate, "all frozen scientific gates, 180/180 readback audits, and 180/180 method audits"),
    ]
    audit_counts = {
        "formal_folds": manifest.get("completed_formal_folds"),
        "independently_audited": manifest.get("independently_audited_checkpoints"),
        "audit_status": manifest.get("checkpoint_audit_status"),
    }
    output = [
        "# Experiment 60 — DSS-PEN 结果报告",
        "",
        f"状态：`{'COMPLETE_AUDITED' if all_checkpoint_audits else 'INCOMPLETE'}`。本报告由已冻结设计和只读 checkpoint 重算生成。",
        "",
        "## 设计与来源",
        "",
        "实验使用 Experiment 53.3 第三组 `CADR_WD_0p01_GFR_WD_0p11` 作为唯一基座。CADR AdamW 两个参数组 WD 均为 0.01；GFR 与 CAM Adam WD 为 0.11。R00 是历史只读参考，R01–R18 为独立重训；全部新增组从逐折未训练锚点开始。源码、论文审计与方法公式见 [design.md](../design.md)，固定医学语义和来源见 [knowledge_base.json](../knowledge_base.json)。",
        "",
        "R02按SPromptGL原代码恢复SPGS+RES通路，不把它误写成疾病级对比损失。BrainPrompt原方法用LLM2Vec类别向量和归一化余弦CE；R03/R04统一使用冻结CLIP，分别是类别名和医学类别中心的BrainPrompt-style adaptation。SABER原Module III检索患者文本，疾病文本监督是独立目标；R05按论文公式检索PHENO患者文本并单独使用疾病目标，且标记为无官方代码的公式重实现，R06才是改为查询疾病文本的适配版本。",
        "R07在CAM的4×36位置表示上做64维患者投影，将医疗文本映射成两组6概念原型，并以128步可微FISTA求解非负、总系数不超过1的部分集合投影；空系数允许拒绝。集合距离间隔损失仅用训练标签，几何保持项和损失权重在外层训练数据上预先校准。按类别距离构造受匹配门控的有界残差，加到原CAM表示后送入未改动OutputLayer；没有新增分类头。R08–R12检验医学语义来源/容量/单原型，R13–R18检验Softmax、强制匹配、自由集合、集合监督和残差位置，逐组唯一差异见config.json。",
        "",
        "保留的历史评价协议包括 seed=0、10折、全局预处理、871节点传导式图和 `val=test` checkpoint 选择。绝对指标可能偏乐观，不代表独立外部泛化；所有配对统计仅作单 seed 探索性证据。",
        "",
        "## 执行与审计",
        "",
        f"- 新增正式训练：{manifest.get('completed_formal_folds')}/180 折；独立 checkpoint 审计：{audit_counts['independently_audited']}/180，状态 `{audit_counts['audit_status']}`。",
        f"- R01 基座 parity：`{manifest.get('r01_parity_status')}`，审计折数 {manifest.get('r01_audited_folds')}/10。",
        f"- 组别结构/语义 bank/投影和关闭项的独立核验：{manifest.get('method_audited_checkpoints')}/180，状态 `{manifest.get('method_audit_status')}`。",
        "- 每个 fold 的连续概率、混淆计数、最佳/执行 epoch 与 checkpoint SHA 见 `fold_metrics.csv`；读回及组别实际生效审计见 `checkpoint_audit.csv` 与 `method_audit.csv`。",
        "- Probability-AUC 统一按 ASD=0 正类、`roc_auc_score(y == 0, P(ASD))` 重算。历史实现使用等价的 TD=1 与 P(TD)，其二分类互补编码给出相同 AUC；Hard-AUC 仍按历史 hard-prediction 定义保留。",
        "",
        "## 主要结果",
        "",
        "下表为各组逐折指标的均值 ± population SD，单位 %。",
        "",
        "| Group | ACC | Hard-AUC | Probability-AUC | SPE-TD | SEN-ASD | BA | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in summary:
        output.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            record["group"], *[record[f"{metric}_mean_plus_minus_sd_pct"] for metric in FORMAL_METRICS]
        ))
    subgroup_rows = [row for row in diagnostics if row.get("record_type") == "R07_reject_subgroup"]
    class_rejection: dict[str, tuple[int, int, float]] = {}
    for label in ("ASD", "TD"):
        selected = [row for row in subgroup_rows if row.get("subgroup") == "diagnosis" and row.get("category") == label]
        n = sum(int(row["n"]) for row in selected)
        rejected = sum(int(row["rejected_n"]) for row in selected)
        class_rejection[label] = (n, rejected, rejected / n if n else math.nan)
    age_site_ranges = {}
    for variable in ("age", "site"):
        by_category: dict[int, list[tuple[int, int]]] = {}
        for row in subgroup_rows:
            if row.get("subgroup") == variable:
                by_category.setdefault(int(row["category"]), []).append((int(row["n"]), int(row["rejected_n"])))
        rates = [sum(item[1] for item in pairs) / sum(item[0] for item in pairs) for pairs in by_category.values() if sum(item[0] for item in pairs)]
        age_site_ranges[variable] = (len(by_category), min(rates) if rates else math.nan, max(rates) if rates else math.nan)
    quartiles = []
    for category in range(1, 5):
        selected = [row for row in subgroup_rows if row.get("subgroup") == "semantic_match_quality_quartile" and int(row.get("category", -1)) == category]
        n = sum(int(row["n"]) for row in selected)
        rejected = sum(int(row["rejected_n"]) for row in selected)
        quartiles.append((category, n, rejected / n if n else math.nan))
    rejected_patients = [row for row in assignments if row["empty_match"]]
    accepted_patients = [row for row in assignments if not row["empty_match"]]
    rejected_distance_mean = float(np.mean([row["minimum_distance"] for row in rejected_patients])) if rejected_patients else math.nan
    accepted_distance_mean = float(np.mean([row["minimum_distance"] for row in accepted_patients])) if accepted_patients else math.nan
    rejected_residual_max = max((float(row["applied_residual_norm"]) for row in rejected_patients), default=0.0)
    output.extend([
        "",
        "完整配对结果在 `paired_comparisons.csv`。每个比较含 10 折方向、exact two-sided sign-flip p 值、fold bootstrap 95% CI，以及按 metric family 进行的 Holm 校正。检验相关性和单 seed 限制意味着不作确认性推断。",
        "",
        "## R07 机制诊断",
        "",
        f"R07 患者异质性冻结规则在 {sum(bool(row['asd_heterogeneity_pass']) for row in r07_folds)}/10 折通过。规则要求 ASD 目标集合中至少两个概念各占接受质量 10% 以上、最大概念不超过 80%，且全部 ASD 测试患者中至少 20% 具有非众数非空支持模式；支持模式按系数 >1e-6 判定，拒绝匹配患者留在分母中。逐患者投影系数见 `semantic_patient_assignments.csv`，诊断明细见 `diagnostics.csv`。",
        "",
        f"测试折上的空匹配率：ASD {class_rejection['ASD'][2]:.3%}（{class_rejection['ASD'][1]}/{class_rejection['ASD'][0]}），TD {class_rejection['TD'][2]:.3%}（{class_rejection['TD'][1]}/{class_rejection['TD'][0]}）。按 PHENO 类别汇总的拒绝率范围：年龄 {age_site_ranges['age'][1]:.3%}–{age_site_ranges['age'][2]:.3%}（{age_site_ranges['age'][0]} 个类别），站点 {age_site_ranges['site'][1]:.3%}–{age_site_ranges['site'][2]:.3%}（{age_site_ranges['site'][0]} 个类别）。按最小集合距离四分位的拒绝率为 {', '.join(f'Q{q}={rate:.3%}' for q, n, rate in quartiles if n)}。",
        "拒绝状态定义为两类最小距离均不小于 1；代码以 `g=0` 精确关闭残差，诊断还逐折核验拒绝样本残差范数为零。按 PHENO 站点和年龄的分层仅作描述，输入中没有独立影像质量字段，因此没有伪造影像质量分层。分类损失到患者投影、文本适配器和残差映射的梯度范数、残差/CAM 范数比也在 diagnostics.csv 中。",
        f"拒绝与接受患者的平均最小集合距离分别为 {rejected_distance_mean:.4f}（n={len(rejected_patients)}）和 {accepted_distance_mean:.4f}（n={len(accepted_patients)}）；拒绝患者最大语义残差范数为 {rejected_residual_max:.6g}。",
        "",
        "固定 checkpoint 干预结果见 `interventions.csv`。这些只读干预用于检查输出敏感性；它们不替代 R08–R18 的从头训练消融。关闭残差与强制拒绝在当前计算图上可能相同，报告会按测得结果解释。",
        "",
        "## 预注册判定",
        "",
        "每个增益门槛均要求 R07 相对对应对照的 ACC 增益至少 0.3 pp、至少 7/10 折获胜、配对 bootstrap 95% CI 下界高于 0，且 SEN-ASD 平均下降不超过 1.0 pp。",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
    ])
    for gate, passed, evidence in gate_rows:
        output.append(f"| {gate} | {'PASS' if passed else 'FAIL'} | {evidence.replace('|', '\\|')} |")
    output.extend([
        "",
        "## 解释与局限",
        "",
        "方法贡献必须分别归因于诊断性能、集合结构、正确医学语义内容和决策残差。若 R07 未胜过类别名、错配、随机锚点或可学习原型，结构变化不能解释为医学知识贡献；若 R07 未胜过 R12/R13 或患者系数不满足异质性规则，则不支持受约束集合投影的必要性。固定投影权重、梯度和 checkpoint 干预不是因果解释或经过验证的生物标志物。",
        "",
        "源码/配置和缓存 SHA、历史来源、逐折锚点及训练状态记录在 `manifest.json`。本报告不把跨图谱、跨队列医学发现外推成当前 ABIDE 256 维相关特征已验证的连接标志物。",
        "",
    ])
    (RESULTS / "report.md").write_text("\n".join(output), encoding="utf-8")
    return gate_rows


def append_handoff(
    manifest: dict[str, Any], summary: list[dict[str, Any]], comparisons: list[dict[str, Any]],
    fold_mechanisms: dict[str, Any],
) -> tuple[str, str]:
    marker = "<!-- EXPERIMENT60_DSS_PEN_2026-11-02 -->"
    if manifest.get("r01_parity_status") == "PASS" and manifest.get("r01_audited_folds") == 10:
        parity_status = "PASS"
    else:
        parity_status = "FAIL"
    audit_status = "PASS" if manifest.get("checkpoint_audit_status") == "PASS" and manifest.get("method_audit_status") == "PASS" else "FAIL"
    gate_map: dict[tuple[str, str], dict[str, Any]] = {
        (row["group_a"], row["group_b"]): row for row in comparisons if row["metric"] == "acc"
    }
    base_comparison = gate_map[("E60_DSS_FULL", "E60_REF_533_G3")]
    selected = [summary_row for summary_row in summary if summary_row["group"] in {
        "E60_REF_533_G3", "E60_BASE_REPLAY", "E60_ORIGINAL_PEN", "E60_BRAINPROMPT_LABEL",
        "E60_BRAINPROMPT_MEDICAL", "E60_SABER_DECISION", "E60_DISEASE_QUERY", "E60_DSS_FULL",
        "E60_DSS_CLASS_ONLY", "E60_DSS_WRONG_SEMANTICS", "E60_DSS_RANDOM_ANCHORS",
        "E60_DSS_LEARNED_PROTOTYPES", "E60_DSS_SINGLE_PROTOTYPE", "E60_DSS_SOFTMAX_ATTENTION",
        "E60_DSS_FORCE_MATCH", "E60_DSS_FREE_SET", "E60_DSS_NO_SET_LOSS", "E60_DSS_NO_RESIDUAL",
        "E60_DSS_CORR_ONLY",
    }]
    compare_targets = [
        "E60_REF_533_G3", "E60_ORIGINAL_PEN", "E60_BRAINPROMPT_LABEL", "E60_BRAINPROMPT_MEDICAL",
        "E60_SABER_DECISION", "E60_DISEASE_QUERY", "E60_DSS_CLASS_ONLY", "E60_DSS_WRONG_SEMANTICS",
        "E60_DSS_RANDOM_ANCHORS", "E60_DSS_LEARNED_PROTOTYPES", "E60_DSS_SINGLE_PROTOTYPE",
        "E60_DSS_SOFTMAX_ATTENTION", "E60_DSS_FORCE_MATCH", "E60_DSS_FREE_SET",
        "E60_DSS_NO_SET_LOSS", "E60_DSS_NO_RESIDUAL", "E60_DSS_CORR_ONLY",
    ]
    comparison_lines = []
    for target in compare_targets:
        row = gate_map[("E60_DSS_FULL", target)]
        comparison_lines.append(
            f"| {target} | {row['mean_difference_pp']:+.3f} | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    result_table = "\n".join(
        f"| {row['group']} | {row['acc_mean_plus_minus_sd_pct']} | {row['hard_auc_mean_plus_minus_sd_pct']} | {row['prob_auc_mean_plus_minus_sd_pct']} |"
        for row in selected
    )
    gate_lines = "\n".join(
        f"- {row['gate']}: **{row['result']}** — {row['evidence']}"
        for row in read_json(RESULTS / "gate_decisions.json")["rows"]
    )
    rejection_rates_asd = []
    rejection_rates_td = []
    heterogeneity_passed = 0
    for fold in range(1, 11):
        values = fold_mechanisms[str(fold)]
        rejection_rates_asd.append(float(values["asd_empty_match_fraction"]))
        rejection_rates_td.append(float(values["td_empty_match_fraction"]))
        heterogeneity_passed += bool(values["asd_heterogeneity_pass"])
    body = f"""{marker}
## Experiment 60 — DSS-PEN

- **状态**：18/18正式组、180/180折训练完成；180/180 checkpoint 独立读回和组别实际生效核验均为`{audit_status}`。R01 基座 parity：`{parity_status}`，10/10 折。源文件、逐折SHA及审计记录位于`{HERE}/`。
- **唯一基座**：Experiment 53.3第三组`CADR_WD_0p01_GFR_WD_0p11`；CADR AdamW 两个参数组 WD=0.01，GFR/CAM Adam WD=0.11。所有新增折均从同一逐折未训练锚点独立开始。固定 seed=0、10折、传导式图、全局预处理与历史`val=test`选模协议保留。
- **方法和对照**：R02复原SPromptGL原SPGS+RES代码路径；R03/R04为统一CLIP编码器的BrainPrompt式类别文本/医学中心适配；R05按SABER公式检索PHENO患者文本并另加疾病监督，R06为疾病文本查询改编。R07使用每类6个医学概念、共享64维适配器、128步可微部分集合投影（系数非负且总和≤1、含空匹配）、集合距离损失和有界CAM残差。R08–R18分别对比类别名、错配/随机/可学习语义、单原型、Softmax、强制/自由匹配、无集合损失、无残差及CORR单槽干预式训练消融。全部细节见`design.md`和`config.json`。

指标为十折mean±population SD（%）；ΔACC为R07减相应组，胜/平/负按折统计：

| Group | ACC | Hard-AUC | Probability-AUC |
|---|---:|---:|---:|
{result_table}

| R07 comparison | ΔACC (pp) | Wins/Ties/Losses |
|---|---:|---:|
{chr(10).join(comparison_lines)}

- R07 vs历史R00：ΔACC {base_comparison['mean_difference_pp']:+.3f} pp，{base_comparison['wins']}/{base_comparison['ties']}/{base_comparison['losses']} 折胜/平/负。完整配对结果、exact sign-flip、bootstrap区间和Holm校正见`results/paired_comparisons.csv`。
- R07 ASD患者异质性冻结规则通过{heterogeneity_passed}/10折；ASD/TD空匹配率分别为{np.mean(rejection_rates_asd):.2%}/{np.mean(rejection_rates_td):.2%}。语义分配、几何、分类梯度、拒绝分层与固定checkpoint干预见`results/diagnostics.csv`、`semantic_patient_assignments.csv`、`interventions.csv`。

### 预注册判定

{gate_lines}

报告：`{RESULTS}/report.md`。保留的全局预处理、transductive图和`val=test`协议限制了外部泛化解释；单seed配对统计仅作探索性证据。模型诊断不构成因果解释或经过验证的生物标志物。
<!-- /EXPERIMENT60_DSS_PEN_2026-11-02 -->"""
    appendix_path = RESULTS / "handoff_append.md"
    appendix_path.write_text(body + "\n", encoding="utf-8")
    handoff = Path("/model/spromptgl/experiment/handoff.md")
    if not handoff.exists():
        return "FAILED_MISSING", sha_file(appendix_path)
    lock_path = handoff.with_suffix(handoff.suffix + ".lock")
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            original = handoff.read_text(encoding="utf-8")
            if marker in original:
                return "ALREADY_PRESENT", sha_file(handoff)
            if not original.endswith("\n"):
                body = "\n" + body
            if not original.endswith("\n\n"):
                body = "\n" + body
            descriptor, temporary_name = tempfile.mkstemp(prefix="handoff.md.e60.", dir=str(handoff.parent))
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(original)
                    stream.write(body)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, handoff)
                directory_fd = os.open(handoff.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            return "APPENDED", sha_file(handoff)
    except OSError:
        return "FAILED_WRITE", sha_file(appendix_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        raise SystemExit("run only after all groups finish and audit.py --all passes: analyze.py --all")
    manifest = require_completed_audit()
    rows, checkpoint_rows, method_audit_rows = load_fold_metrics()
    summary = write_summary(rows)
    comparisons = write_paired_comparisons(rows)
    diagnostics, interventions, assignments, fold_mechanisms = run_r07_mechanisms()
    all_task_diagnostics = []
    for group in GROUPS:
        for fold in range(1, 11):
            checkpoint = E60.torch_load(RT / "groups" / group / "checkpoints" / f"{group}_f{fold:02d}.best.pt")
            semantic = checkpoint.get("diagnostics", {}).get("semantic", {})
            gradient = checkpoint.get("diagnostics", {}).get("gradient_norms", {})
            all_task_diagnostics.append({
                "record_type": "checkpoint_training_diagnostic",
                "group": group,
                "fold": fold,
                "best_epoch": checkpoint["best_epoch"],
                "epochs_executed": checkpoint["epochs_executed"],
                "loss_set": semantic.get("set"),
                "loss_geometry": semantic.get("geometry"),
                "residual_schedule": semantic.get("residual_schedule"),
                "semantic_gradient_norm": semantic.get("gradient_norm"),
                "base_gradient_norm": gradient.get("base"),
                "cam_gradient_norm": gradient.get("cam"),
                "added_gradient_norm": gradient.get("added"),
                "propagation_modes": json.dumps(checkpoint.get("diagnostics", {}).get("propagation_modes", [])),
            })
    diagnostics = all_task_diagnostics + diagnostics
    write_csv(RESULTS / "fold_metrics.csv", rows)
    write_csv(RESULTS / "summary.csv", summary)
    write_csv(RESULTS / "paired_comparisons.csv", comparisons)
    write_csv(RESULTS / "checkpoint_audit.csv", checkpoint_rows)
    write_csv(RESULTS / "method_audit.csv", method_audit_rows)
    write_csv(RESULTS / "diagnostics.csv", diagnostics)
    write_csv(RESULTS / "interventions.csv", interventions)
    manifest["status"] = "COMPLETE_AUDITED"
    manifest["formal_training_status"] = "COMPLETE"
    manifest["completed_formal_folds"] = 180
    manifest["completed_by_group"] = {group: 10 for group in GROUPS}
    manifest["status_by_group"] = {group: "TRAINING_COMPLETE" for group in GROUPS}
    manifest["method_audit_status"] = "PASS"
    manifest["method_audited_checkpoints"] = len(method_audit_rows)
    manifest["analysis_status"] = "COMPLETE"
    manifest["analysis_sha256"] = sha_file(HERE / "analyze.py")
    manifest["source_signature"] = E60.experiment_signature()
    E60.atomic_json(HERE / "manifest.json", manifest)
    gate_rows = write_report(manifest, summary, comparisons, diagnostics, interventions, assignments, fold_mechanisms)
    gate_results = [{"gate": gate, "result": "PASS" if passed else "FAIL", "evidence": evidence} for gate, passed, evidence in gate_rows]
    E60.atomic_json(RESULTS / "gate_decisions.json", {"rows": gate_results})
    handoff_status, handoff_sha = append_handoff(manifest, summary, comparisons, fold_mechanisms)
    manifest["handoff_status"] = handoff_status
    manifest["handoff_sha256"] = handoff_sha
    E60.atomic_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "fold_metric_rows": len(rows),
        "summary_groups": len(summary),
        "paired_comparison_rows": len(comparisons),
        "audit_rows": len(checkpoint_rows),
        "method_audit_rows": len(method_audit_rows),
        "diagnostic_rows": len(diagnostics),
        "intervention_rows": len(interventions),
        "report": str(RESULTS / "report.md"),
        "handoff_status": handoff_status,
        "handoff_sha256": handoff_sha,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
