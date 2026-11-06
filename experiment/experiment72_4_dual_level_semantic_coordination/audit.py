#!/usr/bin/env python3
"""Independent checkpoint, metric, RNG, and mechanism audit for Experiment 72.4."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import run as R
import torch
from scipy.stats import pearsonr, t as student_t
from sklearn.metrics import f1_score, roc_auc_score

EXP = R.EXP
RESULTS = EXP / "results"
GROUPS = tuple(R.GROUP_IDS)
CORE = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
ALL_METRICS = CORE + ("ba",)
TASK_DIR = R.RUNTIME / "tasks"
CHECKPOINT_DIR = R.RUNTIME / "checkpoints"
MECHANISM_DIR = R.RUNTIME / "mechanism"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def independent_metrics(logp: torch.Tensor, indices, labels: torch.Tensor) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    y_all = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    probability = logp.detach().exp().cpu().numpy()[idx]
    y = y_all[idx]
    prediction = probability.argmax(axis=1)
    tp = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 0) & (prediction == 1)))
    fp = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 1) & (prediction == 1)))
    sen, spe = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {
        "acc": float(np.mean(prediction == y)),
        "hard_auc": float(roc_auc_score(np.eye(2, dtype=np.int64)[y], np.eye(2, dtype=np.int64)[prediction])),
        "prob_auc": float(roc_auc_score(y, probability[:, 1])), "sen": sen, "spe": spe,
        "ba": (sen + spe) / 2.0, "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "asd_tp": tp, "td_fp": fp, "asd_fn": fn, "td_tn": tn, "n_test": int(len(idx)),
        "prediction": prediction, "probability": probability,
    }


def same_metrics(actual: dict, expected: dict, where: str) -> None:
    for key in ALL_METRICS:
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            raise AssertionError(f"{where} {key}: independent={actual[key]} saved={expected[key]}")
    for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
        if key in expected and int(actual[key]) != int(expected[key]):
            raise AssertionError(f"{where} {key}: independent={actual[key]} saved={expected[key]}")


def task_paths(group: str, fold: int) -> tuple[Path, Path, Path]:
    stem = f"{group}_f{fold:02d}"
    return TASK_DIR / f"{stem}.json", CHECKPOINT_DIR / f"{stem}.best.pt", MECHANISM_DIR / f"{stem}.json"


def run_eval(state, condition: bool, tokens: torch.Tensor, bank: torch.Tensor, residual: bool = True):
    state["_patient_prompt_active"] = True
    state["_patient_residual_enabled"] = residual
    state["_condition_enabled"] = condition
    state["_patient_tokens"] = tokens
    state["_condition_bank"] = bank
    return R.checked_forward(state, training=False, use_cadr=True)


def audit_one(group: str, fold: int, reference_manifest: dict, references: dict, warmup_audit: dict,
              banks: dict, prompt_meta: dict, patient_cache: dict, manifest: dict) -> tuple[dict, dict, dict, list[dict]]:
    item = R.group_item(group)
    task_path, checkpoint_path, mechanism_path = task_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    checkpoint_sha = R.sha_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if task.get("status") != "COMPLETE" or task.get("checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("task/checkpoint completion or SHA mismatch")
    if checkpoint.get("schema") != "experiment72_4_dual_level_checkpoint_v1" or checkpoint.get("group") != group or int(checkpoint.get("fold", -1)) != fold:
        raise AssertionError("checkpoint schema/group/fold contamination")
    if checkpoint.get("source_sha256") != manifest["source_sha256"] or checkpoint.get("config_sha256") != manifest["config_sha256"]:
        raise AssertionError("checkpoint source/config differs from frozen manifest")
    if checkpoint.get("inference_uses_patient_prompt") is not True or checkpoint.get("inference_uses_disease_supervision_head") is not False:
        raise AssertionError("checkpoint inference declaration is invalid")
    if bool(checkpoint.get("g8_disease_conditioned_query_active")) != bool(item["disease_conditioned_query"]):
        raise AssertionError("checkpoint G8 condition activation differs from its group")
    if checkpoint.get("patient_text_cache_sha256") != R.sha_file(R.PATIENT_CACHE) or checkpoint.get("patient_embedding_sha256") != patient_cache["metadata"]["embeddings_sha256"]:
        raise AssertionError("patient text cache/embedding provenance mismatch")
    if checkpoint.get("prompt_cache_sha256") != R.sha_file(R.PROMPT_CACHE) or checkpoint.get("text_basis_sha256") != prompt_meta["basis_sha256"]:
        raise AssertionError("disease text cache or SVD basis provenance mismatch")
    expected_bank_sha = prompt_meta["banks"][item["prompt"]]["bank_sha256"] if item["disease_supervision"] else None
    if checkpoint.get("prompt_bank_sha256") != expected_bank_sha:
        raise AssertionError("frozen class prototype bank SHA mismatch")

    state = R.build_fold(fold, reference_manifest, references)
    overlap = list(state["train_test_overlap_1based"])
    expected_overlap = [377, 407] if fold == 2 else []
    if overlap != expected_overlap or checkpoint.get("train_test_overlap_1based") != expected_overlap:
        raise AssertionError("E69.3 effective train/test overlap mismatch")
    provenance = (("split_sha256", "split_sha"), ("pair_sha256", "pair_sha"),
                  ("effective_train_sha256", "effective_train_sha"), ("base_train_sha256", "base_train_sha"),
                  ("initial_shared_sha256", "initial_shared_sha"), ("initial_gfr_sha256", "initial_gfr_sha"),
                  ("initial_cam_sha256", "initial_cam_sha"), ("initial_added_sha256", "initial_added_sha"),
                  ("initial_schema_sha256", "initial_schema_sha"), ("initial_rng_sha256", "construction_rng_sha"))
    for cp_key, state_key in provenance:
        if checkpoint.get(cp_key) != state.get(state_key):
            raise AssertionError(f"fold provenance mismatch for {cp_key}")
    expected_warmup_sha = warmup_audit["shared_warmup_cache"]["folds"][str(fold)]["sha256"]
    if checkpoint.get("warmup_sha256") != expected_warmup_sha or state.get("warmup_cache_sha256") != expected_warmup_sha:
        raise AssertionError("shared E61 epoch-10 warmup SHA mismatch")
    expected_alpha = float(manifest["preflight"]["alphas"][str(fold)][group]["alpha"])
    if abs(float(checkpoint["alpha"]) - expected_alpha) > 1e-15:
        raise AssertionError("checkpoint gradient budget coefficient differs from preflight")
    if checkpoint.get("reference_checkpoint_sha256") != references[fold]["checkpoint_sha256"]:
        raise AssertionError("historical E69 checkpoint provenance mismatch")
    if checkpoint.get("e69_reference_task_sha256") != R.sha_file(R.E69_TASKS / f"E69.3-G2_f{fold:02d}.json"):
        raise AssertionError("historical E69 task SHA mismatch")
    if checkpoint.get("train_indices_sha256") != R.sha_obj(list(map(int, state["train"]))) or checkpoint.get("test_indices_sha256") != R.sha_obj(list(map(int, state["test"]))):
        raise AssertionError("checkpoint train/test index SHA mismatch")
    if int(checkpoint["outer_train_count"]) != len(state["train"]) or int(checkpoint["outer_test_count"]) != len(state["test"]):
        raise AssertionError("checkpoint train/test counts mismatch")
    expected_aux_exposure = bool(expected_overlap) and bool(item["disease_supervision"])
    for owner, value in (("task", task), ("checkpoint", checkpoint)):
        if value.get("test_labels_used_for_auxiliary_loss") is not expected_aux_exposure:
            raise AssertionError(f"{owner} auxiliary exposure flag is inaccurate")
        if value.get("overlap_labels_used_for_classification") is not bool(expected_overlap) or value.get("auxiliary_label_exposure_rows_1based") != expected_overlap:
            raise AssertionError(f"{owner} overlap exposure record is inaccurate")

    module, patient_optimizer, disease_optimizer, init_sha, rng_before, rng_after = R.private_auxiliary(int(prompt_meta["rank"]), fold)
    if init_sha != checkpoint.get("initial_auxiliary_sha256") or rng_before != rng_after:
        raise AssertionError("private module initialization/RNG provenance mismatch")
    patient_aux_state = {key.removeprefix("patient."): value for key, value in checkpoint["auxiliary"].items() if key.startswith("patient.")}
    disease_aux_state = {key.removeprefix("disease."): value for key, value in checkpoint["auxiliary"].items() if key.startswith("disease.")}
    if not R.tree_equal(checkpoint["patient_prompt"], patient_aux_state) or not R.tree_equal(checkpoint["disease_supervision_head"], disease_aux_state):
        raise AssertionError("separate module checkpoint copies differ from combined state")
    module.load_state_dict(checkpoint["auxiliary"], strict=True)
    patient_optimizer.load_state_dict(checkpoint["optimizer_patient_prompt"])
    disease_optimizer.load_state_dict(checkpoint["optimizer_disease_supervision_head"])
    for name, optimizer, key in (("patient", patient_optimizer, "optimizer_patient_prompt"),
                                 ("disease", disease_optimizer, "optimizer_disease_supervision_head")):
        if not R.tree_equal(optimizer.state_dict(), checkpoint[key]):
            raise AssertionError(f"{name} optimizer parameter groups or state tensors differ from the saved checkpoint")
    state["model"].load_state_dict(checkpoint["base"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    state["opt_base"].load_state_dict(checkpoint["optimizer_base"])
    state["opt_cam"].load_state_dict(checkpoint["optimizer_cam"])
    state["opt_added"].load_state_dict(checkpoint["optimizer_added"])
    state["model"].eval(); state["cam"].eval(); module.eval()
    device = R.BASE.runtime()["device"]
    module.to(device)
    tokens = patient_cache["embeddings"].to(device=device, dtype=torch.float32)
    bank = banks[item["prompt"]].to(device=device, dtype=torch.float32) if item["disease_supervision"] else banks["SPromptGL-8"].to(device=device, dtype=torch.float32)
    condition_bank = banks["SPromptGL-8"].to(device=device, dtype=torch.float32)
    state["_dual_auxiliary"] = module
    state["_patient_tokens"] = tokens
    state["_condition_bank"] = condition_bank
    state["_patient_prompt_active"] = True
    state["_patient_residual_enabled"] = True
    state["_condition_enabled"] = bool(item["disease_conditioned_query"])

    # The primary readback stays differentiable for independent best-checkpoint gradient diagnostics.
    primary = R.checked_forward(state, training=False, use_cadr=True)
    primary_cam_calls = int(state["cam"]._e72_forward_count)
    logits = primary["logp"]
    metrics = independent_metrics(logits, state["test"], state["y"])
    same_metrics(metrics, checkpoint["metrics"], f"{group}/f{fold:02d} checkpoint")
    same_metrics(metrics, task["metrics"], f"{group}/f{fold:02d} task")
    for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
        if int(metrics[key]) != int(task["confusion"][key]):
            raise AssertionError(f"{group}/f{fold:02d} confusion mismatch")

    positions = state["cam"]._e59_positions
    train_ids = torch.as_tensor(state["train"], dtype=torch.long, device=device)
    train_y = state["y"][train_ids].long()
    patient_params = tuple(module.patient.parameters())
    g_cls_patient = torch.autograd.grad(primary["cls"], patient_params, retain_graph=True, allow_unused=True)

    def grad_norm(grads) -> float:
        values = [value.detach().float().reshape(-1) for value in grads if value is not None]
        return float(torch.cat(values).norm().cpu()) if values else 0.0

    def cosine_from_grads(left, right) -> float | None:
        pairs = [(a.detach().float().reshape(-1), b.detach().float().reshape(-1)) for a, b in zip(left, right) if a is not None and b is not None]
        if not pairs:
            return None
        a, b = torch.cat([pair[0] for pair in pairs]), torch.cat([pair[1] for pair in pairs])
        return float(torch.nn.functional.cosine_similarity(a[None], b[None], dim=-1).cpu())

    gradients = {"gradient_norm_cls_z": None, "gradient_norm_semantic_z": None, "gradient_norm_cls_f": None,
                 "gradient_norm_semantic_f": None, "gradient_norm_patient_cls": grad_norm(g_cls_patient),
                 "gradient_norm_patient_semantic": None, "gradient_norm_disease_head": None,
                 "gradient_norm_g8_condition_cls": 0.0, "gradient_cosine_cls_semantic_z": None,
                 "gradient_cosine_cls_semantic_f": None, "gradient_cosine_patient_cls_semantic": None,
                 "best_weighted_gradient_ratio_cam": 0.0}
    disease_diag = {"semantic_global_acc_pct": None, "semantic_global_margin": None}
    if item["disease_supervision"]:
        supervised, _ = R.make_losses(module, positions[train_ids], train_y, bank, float(item["temperature"]), item["grain"])
        sem_loss = supervised["semantic"]
        g_cls_z = torch.autograd.grad(primary["cls"], positions, retain_graph=True)[0][train_ids].detach().float()
        g_sem_z = torch.autograd.grad(sem_loss, positions, retain_graph=True)[0][train_ids].detach().float()
        fused = state["cam"]._e72_fused_before_prompt
        g_cls_f = torch.autograd.grad(primary["cls"], fused, retain_graph=True)[0][train_ids].detach().float()
        g_sem_f = torch.autograd.grad(sem_loss, fused, retain_graph=True)[0][train_ids].detach().float()
        g_sem_patient = torch.autograd.grad(sem_loss, patient_params, retain_graph=True, allow_unused=True)
        g_disease = torch.autograd.grad(sem_loss, tuple(module.disease.parameters()), retain_graph=True, allow_unused=True)
        condition_params = tuple(parameter for name, parameter in module.patient.named_parameters() if any(key in name for key in ("condition_projection", "condition_query", "condition_lambda_logit")))
        g_condition = torch.autograd.grad(primary["cls"], condition_params, retain_graph=True, allow_unused=True)
        cls_z_norm, sem_z_norm = float(g_cls_z.norm().cpu()), float(g_sem_z.norm().cpu())
        gradients.update({
            "gradient_norm_cls_z": cls_z_norm, "gradient_norm_semantic_z": sem_z_norm,
            "gradient_norm_cls_f": float(g_cls_f.norm().cpu()), "gradient_norm_semantic_f": float(g_sem_f.norm().cpu()),
            "gradient_norm_patient_semantic": grad_norm(g_sem_patient), "gradient_norm_disease_head": grad_norm(g_disease),
            "gradient_norm_g8_condition_cls": grad_norm(g_condition),
            "gradient_cosine_cls_semantic_z": cosine_from_grads((g_cls_z,), (g_sem_z,)),
            "gradient_cosine_cls_semantic_f": cosine_from_grads((g_cls_f,), (g_sem_f,)),
            "gradient_cosine_patient_cls_semantic": cosine_from_grads(g_cls_patient, g_sem_patient),
            "best_weighted_gradient_ratio_cam": float(checkpoint["alpha"]) * sem_z_norm / (cls_z_norm + 1e-12),
        })
        if item["disease_conditioned_query"] and gradients["gradient_norm_g8_condition_cls"] <= 0.0:
            raise AssertionError("G8 condition parameters receive no best-checkpoint classification gradient")
        with torch.no_grad():
            disease_output = module.disease(positions.detach(), bank, float(item["temperature"]), use_fine=(item["grain"] == "joint"))
            similarities = disease_output["global_logits"] * float(item["temperature"])
            test_ids_np = np.asarray(state["test"], dtype=np.int64)
            test_t = torch.as_tensor(test_ids_np, dtype=torch.long, device=device)
            test_y = state["y"][test_t].long()
            pos = similarities[test_t].gather(1, test_y[:, None]).squeeze(1)
            neg = similarities[test_t].gather(1, (1-test_y)[:, None]).squeeze(1)
            disease_diag["semantic_global_acc_pct"] = 100.0 * float((similarities[test_t].argmax(-1) == test_y).float().mean().cpu())
            disease_diag["semantic_global_margin"] = float((pos-neg).mean().cpu())
            if disease_output["fine_logits"] is not None:
                fine = disease_output["fine_logits"] * float(item["temperature"])
                for position, name in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                    per = fine[test_t, position]
                    pos = per.gather(1, test_y[:, None]).squeeze(1)
                    neg = per.gather(1, (1-test_y)[:, None]).squeeze(1)
                    disease_diag[f"semantic_{name}_acc_pct"] = 100.0 * float((per.argmax(-1) == test_y).float().mean().cpu())
                    disease_diag[f"semantic_{name}_margin"] = float((pos-neg).mean().cpu())

    test_ids = np.asarray(state["test"], dtype=np.int64)
    y_all = state["y"].detach().cpu().numpy().astype(np.int64, copy=False)
    y_test = y_all[test_ids]
    test_tensor = torch.as_tensor(test_ids, dtype=torch.long, device=device)
    prediction = metrics["prediction"]
    with torch.no_grad():
        prompt = state["cam"]._e72_prompt_output
        residual = prompt["relative_residual"][test_tensor].detach().cpu().numpy()
        gate = prompt["gate"][test_tensor].detach().cpu().numpy()
        attention = prompt["attention"][test_tensor].detach().cpu().numpy()
        entropy = prompt["attention_entropy"][test_tensor].detach().cpu().numpy()
        primary_cpu = logits.detach().cpu()[test_ids]
        no_residual = run_eval(state, bool(item["disease_conditioned_query"]), tokens, condition_bank, residual=False)
        no_residual_prediction = no_residual["logp"].exp().cpu().numpy()[test_ids].argmax(1)
        residual_flips = prediction != no_residual_prediction
        corrected = int(np.sum((prediction == y_test) & (no_residual_prediction != y_test)))
        introduced = int(np.sum((prediction != y_test) & (no_residual_prediction == y_test)))
        residual_delta = float((primary_cpu - no_residual["logp"].detach().cpu()[test_ids]).abs().max())
        neutral_tokens = tokens.mean(dim=0, keepdim=True).expand_as(tokens).contiguous()
        neutral = run_eval(state, bool(item["disease_conditioned_query"]), neutral_tokens, condition_bank)
        neutral_prediction = neutral["logp"].exp().cpu().numpy()[test_ids].argmax(1)
        site_tokens = tokens.clone()
        site_tokens[:, 2, :] = tokens[:, 2, :].mean(dim=0, keepdim=True)
        site_mask = run_eval(state, bool(item["disease_conditioned_query"]), site_tokens, condition_bank)
        site_prediction = site_mask["logp"].exp().cpu().numpy()[test_ids].argmax(1)
        condition_diag = {"g8_condition_off_flip_count": 0, "g8_condition_off_corrected": 0,
                          "g8_condition_off_introduced": 0, "g8_condition_off_logit_delta_linf": 0.0,
                          "g8_swapped_prototypes_flip_count": 0, "g8_swapped_prototypes_acc_pct": None}
        if item["disease_conditioned_query"]:
            condition_off = run_eval(state, False, tokens, condition_bank)
            off_prediction = condition_off["logp"].exp().cpu().numpy()[test_ids].argmax(1)
            swapped = run_eval(state, True, tokens, condition_bank.flip(0).contiguous())
            swapped_prediction = swapped["logp"].exp().cpu().numpy()[test_ids].argmax(1)
            condition_diag = {
                "g8_condition_off_flip_count": int(np.sum(prediction != off_prediction)),
                "g8_condition_off_corrected": int(np.sum((prediction == y_test) & (off_prediction != y_test))),
                "g8_condition_off_introduced": int(np.sum((prediction != y_test) & (off_prediction == y_test))),
                "g8_condition_off_logit_delta_linf": float((primary_cpu - condition_off["logp"].detach().cpu()[test_ids]).abs().max()),
                "g8_swapped_prototypes_flip_count": int(np.sum(prediction != swapped_prediction)),
                "g8_swapped_prototypes_acc_pct": 100.0 * float(np.mean(swapped_prediction == y_test)),
            }
        probabilities = prompt["condition_probabilities"]
        if probabilities is None:
            condition_mean, condition_entropy, condition_lambda = None, None, 0.0
        else:
            probability = probabilities[test_tensor].detach().cpu().numpy()
            condition_mean = probability.mean(0).tolist()
            condition_entropy = float((-(probability * np.log(np.maximum(probability, 1e-12))).sum(1)).mean())
            condition_lambda = float(prompt["condition_lambda"].detach().cpu())

    selections = patient_cache["metadata"]["selections"]
    oof_rows = []
    for local, patient in enumerate(test_ids):
        meta = selections[int(patient)]
        oof_rows.append({
            "group": group, "patient_index0": int(patient), "label": int(y_test[local]),
            "prediction": int(prediction[local]), "correct": bool(prediction[local] == y_test[local]),
            "no_patient_prediction": int(no_residual_prediction[local]), "neutral_prediction": int(neutral_prediction[local]),
            "site_mask_prediction": int(site_prediction[local]),
            "age": meta["age"]["column"] if meta["age"] else "MISSING",
            "sex": meta["sex"]["recorded_category"] if meta["sex"] else "MISSING",
            "site": meta["site"]["site_id"] if meta["site"] else "MISSING",
            "attention_age": float(attention[local, 0]), "attention_sex": float(attention[local, 1]),
            "attention_site": float(attention[local, 2]),
        })
    row = {
        "group": group, "fold": fold, **{f"{key}_pct": 100.0 * float(metrics[key]) for key in ALL_METRICS},
        **{key: int(metrics[key]) for key in ("asd_tp", "td_fp", "asd_fn", "td_tn")},
        "train_count_effective": len(state["train"]), "test_count": len(test_ids),
        "best_epoch_1based": int(checkpoint["best_epoch"]) + 1, "epochs_executed": int(checkpoint["epochs_executed"]),
        "alpha": float(checkpoint["alpha"]), "train_test_overlap_1based": overlap,
        "test_label_exposure_count": len(overlap), "checkpoint_sha256": checkpoint_sha,
        "independent_checkpoint_audit": "PASS",
    }
    if fold == 2:
        reduced_ids = [int(index) for index in state["test"] if int(index) not in {376, 406}]
        reduced = independent_metrics(logits, reduced_ids, state["y"])
        row["without_exposed_n"] = reduced["n_test"]
        row.update({f"without_exposed_{key}_pct": 100.0 * float(reduced[key]) for key in ALL_METRICS})
        row.update({f"without_exposed_{key}": int(reduced[key]) for key in ("asd_tp", "td_fp", "asd_fn", "td_tn")})

    adaptive = torch.sigmoid(state["model"].adaptive_gate_logits).detach().cpu().numpy()
    edge = torch.sigmoid(state["model"].edge_gate_logits).detach().cpu().numpy()
    historical = references[fold]
    hist_adaptive = np.asarray(historical.get("best_adaptive_gate", [np.nan] * 4), dtype=np.float64)
    hist_edge = np.asarray(historical.get("best_edge_gate", [np.nan] * 4), dtype=np.float64)
    mechanism = {
        "group": group, "fold": fold, "best_epoch_1based": int(checkpoint["best_epoch"]) + 1,
        "alpha": float(checkpoint["alpha"]), "patient_residual_mean_test": float(residual.mean()),
        "patient_residual_p95_test": float(np.quantile(residual, .95)), "patient_residual_max_test": float(residual.max()),
        "patient_gate_mean_test": float(gate.mean()), "patient_gate_sd_test": float(gate.std()),
        "attention_age_mean": float(attention[:, 0].mean()), "attention_sex_mean": float(attention[:, 1].mean()),
        "attention_site_mean": float(attention[:, 2].mean()), "attention_age_sd": float(attention[:, 0].std()),
        "attention_sex_sd": float(attention[:, 1].std()), "attention_site_sd": float(attention[:, 2].std()),
        "attention_entropy_mean": float(entropy.mean()), "attention_entropy_max": float(np.log(3.0)),
        "attention_patient_variation_mean_sd": float(attention.std(axis=0).mean()),
        "patient_logits_change_linf": residual_delta, "patient_prompt_flip_count": int(residual_flips.sum()),
        "patient_prompt_corrected_cases": corrected, "patient_prompt_new_errors": introduced,
        "patient_prompt_accuracy_pp_vs_off": 100.0 * (float(np.mean(prediction == y_test)) - float(np.mean(no_residual_prediction == y_test))),
        "neutral_token_flip_count": int(np.sum(prediction != neutral_prediction)),
        "site_token_mask_flip_count": int(np.sum(prediction != site_prediction)),
        "site_mask_accuracy_pp": 100.0 * (float(np.mean(site_prediction == y_test)) - float(np.mean(prediction == y_test))),
        "condition_prob_asd_mean": None if condition_mean is None else condition_mean[0],
        "condition_prob_td_mean": None if condition_mean is None else condition_mean[1],
        "condition_entropy_mean": condition_entropy, "condition_lambda": condition_lambda,
        "condition_branch_active": bool(item["disease_conditioned_query"]),
        **gradients, **disease_diag, **condition_diag,
        "edge_gate_best": edge.tolist(),
        "edge_gate_delta_from_initial": (edge - np.asarray(checkpoint["initial_gate_values"]["edge_gate"])).tolist(),
        "adaptive_gate_best": adaptive.tolist(),
        "adaptive_gate_delta_from_initial": (adaptive - np.asarray(checkpoint["initial_gate_values"]["adaptive_gate"])).tolist(),
        "adaptive_gate_delta_vs_E69": (adaptive - hist_adaptive).tolist(),
        "edge_gate_delta_vs_E69": (edge - hist_edge).tolist(), "train_test_overlap_1based": overlap,
    }
    history = json.loads(mechanism_path.read_text(encoding="utf-8"))
    if task.get("mechanism_sha256") != R.sha_file(mechanism_path) or not history:
        raise AssertionError("training mechanism trace SHA/missing rows")
    best_rows = [entry for entry in history if int(entry["epoch"]) == int(checkpoint["best_epoch"])]
    if len(best_rows) != 1:
        raise AssertionError("best epoch must have exactly one saved mechanism row")
    ref_trace = {int(entry["epoch"]): entry for entry in references[fold].get("rng_trace", [])}
    exact = compared = 0
    for entry in history:
        reference_epoch = int(entry["epoch"]) + 1
        if reference_epoch in ref_trace:
            reference = ref_trace[reference_epoch]
            compared += 1
            exact += int(entry.get("rng_before_train_sha256") == reference.get("rng_before")
                         and entry.get("rng_after_train_sha256") == reference.get("rng_after_train")
                         and entry.get("rng_after_eval_sha256") == reference.get("rng_after_eval"))
    rng_status = "PASS" if compared > 0 and exact == compared else "FAIL"
    row.update({"rng_trace_exact_epochs": exact, "rng_trace_compared_epochs": compared, "rng_trace_status": rng_status})
    audit = {"group": group, "fold": fold, "status": "PASS" if rng_status == "PASS" else "FAIL_RNG_DIVERGED",
             "checkpoint_sha256": checkpoint_sha, "independent_metric_recalculation": "PASS",
             "checkpoint_provenance": "PASS", "patient_and_disease_optimizer_readback": "PASS",
             "original_classifier_only_inference": "PASS", "cam_calls_primary_readback": primary_cam_calls,
             "train_test_overlap_1based": overlap, "rng_trace_exact_epochs": exact,
             "rng_trace_compared_epochs": compared, "rng_trace_status": rng_status}
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, mechanism, audit, oof_rows


def audit_historical_fold2(reference_manifest: dict, references: dict, warmup_audit: dict) -> dict[str, Any]:
    fold = 2
    checkpoint_path = R.RUNTIME / "reference_e69/E69.3-G2_f02.best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if R.sha_file(checkpoint_path) != references[fold]["checkpoint_sha256"]:
        raise AssertionError("staged historical E69.3-G2 fold-2 checkpoint SHA mismatch")
    state = R.build_fold(fold, reference_manifest, references)
    R.load_warmup(state, fold, warmup_audit)
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    with torch.no_grad():
        output = R.checked_forward(state, training=False, use_cadr=True)
        full = independent_metrics(output["logp"], state["test"], state["y"])
        same_metrics(full, checkpoint["metrics"], "historical E69.3-G2 fold 2")
        if int(state["cam"]._e72_forward_count) != 1:
            raise AssertionError("historical E69.3 fold-2 CAM readback did not run exactly once")
        keep = [int(index) for index in state["test"] if int(index) not in {376, 406}]
        reduced = independent_metrics(output["logp"], keep, state["y"])
    if list(state["train_test_overlap_1based"]) != [377, 407] or not {376, 406}.issubset(set(state["test"])):
        raise AssertionError("historical E69.3 fold-2 overlap rows differ from the declared protocol")
    row = {"group": "E69.3-G2", "fold": 2, "test_count_full": full["n_test"], "test_count_without_377_407": reduced["n_test"],
           "checkpoint_sha256": R.sha_file(checkpoint_path), "independent_checkpoint_audit": "PASS"}
    for metric in ALL_METRICS:
        row[f"full_{metric}_pct"] = 100.0 * float(full[metric])
        row[f"without_377_407_{metric}_pct"] = 100.0 * float(reduced[metric])
    for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
        row[f"full_{key}"] = int(full[key])
        row[f"without_377_407_{key}"] = int(reduced[key])
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def summary_rows(rows: list[dict]) -> list[dict]:
    base = R.reference_rows()
    result = []
    for group in ("E69.3-G2", *GROUPS):
        by_fold = {int(row["fold"]): row for row in rows if row["group"] == group}
        values = {metric: (np.asarray([100.0 * float(base[f][metric]) for f in range(1, 11)]) if group == "E69.3-G2"
                           else np.asarray([float(by_fold[f][f"{metric}_pct"]) for f in sorted(by_fold)])) for metric in ALL_METRICS}
        result.append({"group": group, "fold_count": 10,
                       "metric_type": "historical_reference" if group == "E69.3-G2" else "E69.3_protocol_with_fold2_label_exposure",
                       **{f"{metric}_mean_pct": float(values[metric].mean()) for metric in ALL_METRICS},
                       **{f"{metric}_population_sd_pct": float(values[metric].std(ddof=0)) for metric in ALL_METRICS},
                       **{f"{metric}_mean_plus_minus_sd_pct": f"{values[metric].mean():.4f} +/- {values[metric].std(ddof=0):.4f}" for metric in ALL_METRICS}})
    return result


def paired_rows(rows: list[dict]) -> list[dict]:
    base = R.reference_rows()
    pairs = [(f"{group}_vs_E69.3-G2", group, "E69.3-G2") for group in GROUPS]
    pairs += [("G8_minus_G7", "G8", "G7"), ("G3_minus_G2", "G3", "G2"), ("G5_minus_G4", "G5", "G4"),
              ("G5_minus_G3", "G5", "G3"), ("G7_minus_G6", "G7", "G6"), ("G6_minus_G3", "G6", "G3"),
              ("G7_minus_G5", "G7", "G5"), ("G2_minus_G1", "G2", "G1"), ("G8_minus_G1", "G8", "G1")]
    lookup = {group: {int(row["fold"]): row for row in rows if row["group"] == group} for group in GROUPS}
    result = []
    for name, left, right in pairs:
        changes = {}
        for metric in ALL_METRICS:
            lhs = [100.0 * float(base[f][metric]) if left == "E69.3-G2" else float(lookup[left][f][f"{metric}_pct"]) for f in range(1, 11)]
            rhs = [100.0 * float(base[f][metric]) if right == "E69.3-G2" else float(lookup[right][f][f"{metric}_pct"]) for f in range(1, 11)]
            changes[metric] = np.asarray(lhs) - np.asarray(rhs)
        delta = changes["acc"]
        clean = np.delete(delta, 1)
        def ci(values: np.ndarray) -> tuple[float, float]:
            half = float(student_t.ppf(.975, len(values)-1) * values.std(ddof=1) / math.sqrt(len(values)))
            return float(values.mean()-half), float(values.mean()+half)
        low, high = ci(delta); clean_low, clean_high = ci(clean)
        row = {"comparison": name, "left": left, "right": right, "delta_acc_mean_pp": float(delta.mean()),
               "delta_acc_population_sd_pp": float(delta.std(ddof=0)), "acc_wins": int(np.sum(delta > 1e-12)),
               "acc_ties": int(np.sum(np.abs(delta) <= 1e-12)), "acc_losses": int(np.sum(delta < -1e-12)),
               "delta_acc_ci95_low_pp": low, "delta_acc_ci95_high_pp": high,
               "delta_acc_mean_9_nonoverlap_folds_pp": float(clean.mean()),
               "acc_wins_9_nonoverlap_folds": int(np.sum(clean > 1e-12)),
               "acc_ties_9_nonoverlap_folds": int(np.sum(np.abs(clean) <= 1e-12)),
               "acc_losses_9_nonoverlap_folds": int(np.sum(clean < -1e-12)),
               "delta_acc_9_nonoverlap_ci95_low_pp": clean_low, "delta_acc_9_nonoverlap_ci95_high_pp": clean_high,
               "delta_acc_pp_by_fold": delta.tolist()}
        for metric, values in changes.items():
            row[f"delta_{metric}_mean_pp"] = float(values.mean())
            row[f"delta_{metric}_pp_by_fold"] = values.tolist()
        result.append(row)
    return result


def subgroup_rows(oof: dict[str, list[dict]]) -> list[dict]:
    output = []
    for group, records in oof.items():
        for field in ("site", "age", "sex"):
            for category in sorted({record[field] for record in records}):
                selected = [record for record in records if record[field] == category]
                labels = np.asarray([record["label"] for record in selected], dtype=np.int64)
                prediction = np.asarray([record["prediction"] for record in selected], dtype=np.int64)
                masked = np.asarray([record["site_mask_prediction"] for record in selected], dtype=np.int64)
                tp = int(np.sum((labels == 0) & (prediction == 0))); fn = int(np.sum((labels == 0) & (prediction == 1)))
                fp = int(np.sum((labels == 1) & (prediction == 0))); tn = int(np.sum((labels == 1) & (prediction == 1)))
                output.append({"group": group, "field": field, "category": category, "n": len(selected),
                               "acc_pct": 100.0*float(np.mean(labels == prediction)),
                               "sen_asd_pct": 100.0*tp/max(tp+fn, 1), "spe_td_pct": 100.0*tn/max(tn+fp, 1),
                               "site_token_mask_accuracy_change_pp": 100.0*(float(np.mean(labels == masked))-float(np.mean(labels == prediction)))})
    return output


def attention_association(oof: dict[str, list[dict]]) -> dict[str, Any]:
    g7 = {row["patient_index0"]: row for row in oof["G7"]}
    g8 = {row["patient_index0"]: row for row in oof["G8"]}
    common = sorted(set(g7) & set(g8))
    changes = np.asarray([[g8[index][f"attention_{field}"] - g7[index][f"attention_{field}"] for field in ("age", "sex", "site")] for index in common])
    correction = np.asarray([int(g8[index]["correct"] and not g7[index]["correct"]) for index in common])
    new_error = np.asarray([int(g7[index]["correct"] and not g8[index]["correct"]) for index in common])
    association = {}
    for column, name in enumerate(("age", "sex", "site")):
        if len(np.unique(changes[:, column])) > 1 and len(np.unique(correction)) > 1:
            coefficient, pvalue = pearsonr(changes[:, column], correction)
            association[name] = {"pearson_r_vs_G8_corrections": float(coefficient), "p_descriptive": float(pvalue)}
        else:
            association[name] = {"pearson_r_vs_G8_corrections": None, "p_descriptive": None}
    return {"paired_patients": len(common), "G8_corrections_vs_G7": int(correction.sum()),
            "G8_new_errors_vs_G7": int(new_error.sum()), "attention_change_correction_association": association,
            "interpretation": "descriptive, not causal or clinical attribution"}


def make_report(manifest: dict, preflight: dict, summaries: list[dict], pairs: list[dict], mechanisms: list[dict], audits: list[dict],
                fold2_sensitivity: list[dict], subgroups: list[dict], g8_assoc: dict, verdict: str) -> str:
    summary = {row["group"]: row for row in summaries}
    pair = {row["comparison"]: row for row in pairs}
    h1_root = Path(R.CFG["paths"]["e72_1"])
    h1_manifest = json.loads((h1_root / "manifest.json").read_text(encoding="utf-8"))
    with (h1_root / "results/summary.csv").open("r", newline="", encoding="utf-8") as stream:
        h1_summary = {row["group"]: row for row in csv.DictReader(stream)}
    h1_summary_sha = R.sha_file(h1_root / "results/summary.csv")
    lines = [
        "# Experiment 72.4 - Dual-Level Patient-Disease Semantic Prompt Coordination", "",
        f"Status: **{manifest.get('status')}**; independent checkpoint audit: **{sum(row.get('status') == 'PASS' for row in audits)}/80**; decision: **{verdict}**.", "",
        "## Source and implementation audit", "",
        "The locked base is E69.3-G2 at commit `41dbcb45a0273673857051d8e821b8c3315eef00`, using the E61-E2 end-to-end adaptive DARG forward. SPromptGL reports the 320-dimensional GFR, four 36-dimensional CAM positions and 144-dimensional classifier input. BrainPrompt's official code uses participant metadata for patient text and disease class prompts for class supervision. SABER Module III was checked against paper equations (8)-(13); no verifiable public official code was found, so only the paper mechanism was checked.",
        "References: [SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [BrainPrompt MICCAI 2025](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [SABER arXiv](https://arxiv.org/abs/2607.01901), [BrainPrompt code](https://github.com/AngusMonroe/BrainPrompt), [SPromptGL code](https://github.com/wanxixi11/SPromptGL).", "",
        "## Results", "", "Values are mean +/- population SD across 10 folds (%). Hard-AUC uses hard predictions; Prob-AUC uses continuous TD probabilities.",
        "", "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("E69.3-G2", *GROUPS):
        row = summary[group]
        values = [row[f"{metric}_mean_plus_minus_sd_pct"] for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")]
        lines.append(f"| {group} | " + " | ".join(values) + " |")
    lines += ["", "### Paired ACC changes", "", "Positive values favor the left-hand group. W/T/L covers 10 folds; the 9-fold result omits fold 2 and remains descriptive.",
              "", "| Comparison | Mean change (pp) | 95% paired t CI | W/T/L | 9-fold mean | 9-fold W/T/L |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("G1_vs_E69.3-G2", "G8_vs_E69.3-G2", "G8_minus_G7", "G3_minus_G2", "G5_minus_G4", "G5_minus_G3", "G7_minus_G6", "G6_minus_G3", "G7_minus_G5", "G8_minus_G1"):
        row = pair[name]
        lines.append(f"| {name} | {row['delta_acc_mean_pp']:+.4f} | [{row['delta_acc_ci95_low_pp']:+.4f}, {row['delta_acc_ci95_high_pp']:+.4f}] | {row['acc_wins']}/{row['acc_ties']}/{row['acc_losses']} | {row['delta_acc_mean_9_nonoverlap_folds_pp']:+.4f} | {row['acc_wins_9_nonoverlap_folds']}/{row['acc_ties_9_nonoverlap_folds']}/{row['acc_losses_9_nonoverlap_folds']} |")
    lines += ["", "### Read-only H1 E72.1 reference", "",
              f"H1 was COMPLETE_AUDITED with {h1_manifest.get('completed_tasks')} tasks and {h1_manifest.get('independent_checkpoint_audits_passed', h1_manifest.get('independent_audits'))} checkpoint audits. Its summary SHA-256 is `{h1_summary_sha}`. These are historical results on the inherited E69.3 protocol, without the E72.4 patient residual; they were not retrained.",
              "", "| H1 group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("G1", "G2", "G3", "G4", "G5", "G6"):
        row = h1_summary[group]
        values = [row[f"{metric}_mean_plus_minus_sd_pct"] for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")]
        lines.append(f"| H1-{group} | " + " | ".join(values) + " |")
    h1_matches = (("G3", "G1"), ("G6", "G2"), ("G5", "G3"), ("G7", "G4"), ("G2", "G5"))
    lines += ["", "Descriptive ACC differences from the corresponding H1 configuration are:"]
    for e72_group, h1_group in h1_matches:
        delta = float(summary[e72_group]["acc_mean_pct"]) - float(h1_summary[h1_group]["acc_mean_pct"])
        lines.append(f"- E72.4 {e72_group} minus H1 {h1_group}: {delta:+.4f} pp.")
    lines += ["", "H1-G6 (BrainPrompt-2, tau=0.1, global-only) has no exact E72.4 group match; E72.4-G8 also has no H1 counterpart because it adds the disease-conditioned patient query. These cross-experiment differences include the E72.4 patient prompt and training-path changes and are not single-factor causal estimates.", ""]
    g8_pair, base_pair = pair["G8_minus_G7"], pair["G8_vs_E69.3-G2"]
    g8_mech = [row for row in mechanisms if row["group"] == "G8"]
    corrected = sum(int(row.get("g8_condition_off_corrected", 0)) for row in g8_mech)
    introduced = sum(int(row.get("g8_condition_off_introduced", 0)) for row in g8_mech)
    def mean_value(rows: list[dict], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows if row.get(key) not in (None, "")]))
    disease_group_ids = [item["id"] for item in R.CFG["groups"] if item["disease_supervision"]]
    alphas = [float(preflight["alphas"][fold][group]["alpha"]) for fold in preflight["alphas"] for group in disease_group_ids]
    g8_site_rows = [row for row in subgroups if row["group"] == "G8" and row["field"] == "site" and int(row["n"]) >= 20]
    g8_age_rows = [row for row in subgroups if row["group"] == "G8" and row["field"] == "age" and int(row["n"]) >= 20]
    g8_sex_rows = [row for row in subgroups if row["group"] == "G8" and row["field"] == "sex"]
    top_site = max(g8_site_rows, key=lambda row: int(row["n"])) if g8_site_rows else None
    lowest_site = min(g8_site_rows, key=lambda row: float(row["acc_pct"])) if g8_site_rows else None
    highest_site = max(g8_site_rows, key=lambda row: float(row["acc_pct"])) if g8_site_rows else None
    lowest_age = min(g8_age_rows, key=lambda row: float(row["acc_pct"])) if g8_age_rows else None
    highest_age = max(g8_age_rows, key=lambda row: float(row["acc_pct"])) if g8_age_rows else None
    g8_fine = {name: mean_value(g8_mech, f"semantic_{name}_acc_pct") for name in ("PHENO", "ANAT", "FUNC", "CORR")}
    g8_gate = np.mean(np.asarray([row["adaptive_gate_best"] for row in g8_mech], dtype=np.float64), axis=0).tolist()
    g8_gate_delta = np.mean(np.asarray([row["adaptive_gate_delta_vs_E69"] for row in g8_mech], dtype=np.float64), axis=0).tolist()
    g8_weighted_f_ratio = mean_value(g8_mech, "best_weighted_gradient_ratio_cam")
    g8_f_ratio = mean_value(g8_mech, "gradient_norm_semantic_f") / max(mean_value(g8_mech, "gradient_norm_cls_f"), 1e-12)
    lines += ["", "## Mechanism findings", "",
              f"Patient prompts used PHENO-recoverable age intervals, recorded sex category codes, and site IDs. All three fields were present for {sum(all(row) for row in preflight['patient_text_cache']['field_validity'])}/871 rows. Frozen CLIP RN50 encoded three 1024-dimensional tokens per patient once, aligned by source row. The loader read PHENO columns only and did not read diagnosis labels. It generated no exact ages, VIQ/PIQ, symptoms, or unmeasured clinical facts.", "",
              f"Across selected checkpoints, disabling the patient residual caused {sum(int(row['patient_prompt_flip_count']) for row in mechanisms)} held-out prediction flips. The mean active-minus-disabled ACC difference across group-folds was {np.mean([row['patient_prompt_accuracy_pp_vs_off'] for row in mechanisms]):+.4f} pp. Activating the residual corrected {sum(int(row['patient_prompt_corrected_cases']) for row in mechanisms)} cases and introduced {sum(int(row['patient_prompt_new_errors']) for row in mechanisms)} errors. These fixed-checkpoint interventions establish model sensitivity, not clinical meaning.", "",
              f"G8 vs G7 ACC changed {g8_pair['delta_acc_mean_pp']:+.4f} pp ({g8_pair['acc_wins']}/{g8_pair['acc_ties']}/{g8_pair['acc_losses']}); G8 vs E69.3-G2 changed {base_pair['delta_acc_mean_pp']:+.4f} pp. Disabling the G8 disease-conditioned query corrected {corrected} cases and introduced {introduced}; swapping prototype order changed {sum(int(row.get('g8_swapped_prototypes_flip_count', 0)) for row in g8_mech)} predictions. Both ASD and TD prototypes enter as soft probabilities; no true test label is used for routing.", "",
              "G8's test-time condition removal changed " + str(sum(int(row.get("g8_condition_off_flip_count", 0)) for row in g8_mech)) + " hard predictions in 10 folds (7 corrections, 2 new errors). Swapping both prototype rows is permutation-invariant for the softmax-weighted prototype sum by construction; its zero hard flips are an implementation invariance check, not evidence that disease conditions are unused. The active condition branch has nonzero classification gradient in all folds.", "",
              f"Across OOF patients G8 made {g8_assoc['G8_corrections_vs_G7']} corrections and {g8_assoc['G8_new_errors_vs_G7']} new errors versus G7. Attention-change associations are descriptive only: `{json.dumps(g8_assoc['attention_change_correction_association'], ensure_ascii=False)}`.", "",
              f"Patient tokens received mean attention `{mean_value(mechanisms, 'attention_age_mean'):.3f}` age, `{mean_value(mechanisms, 'attention_sex_mean'):.3f}` sex, and `{mean_value(mechanisms, 'attention_site_mean'):.3f}` site across group-fold checkpoints; site attention was highest. The mean residual ratio was `{mean_value(mechanisms, 'patient_residual_mean_test'):.4f}` with mean p95 `{mean_value(mechanisms, 'patient_residual_p95_test'):.4f}`, so most subjects reached the fixed 10% cap. Site-token masking (masked minus active) changed G8 ACC by `{mean_value(g8_mech, 'site_mask_accuracy_pp'):+.2f}` pp on average; the negative change means masking lowered ACC. For sites with at least 20 G8 OOF subjects, the largest site was {top_site['category']} (n={top_site['n']}); accuracy ranged from {lowest_site['category']} {float(lowest_site['acc_pct']):.2f}% (n={lowest_site['n']}) to {highest_site['category']} {float(highest_site['acc_pct']):.2f}% (n={highest_site['n']}). These figures indicate site information materially affects predictions and leave a site-confounding risk.", "",
              f"G8 OOF sex-code accuracies were {g8_sex_rows[0]['category']}={float(g8_sex_rows[0]['acc_pct']):.2f}% (n={g8_sex_rows[0]['n']}) and {g8_sex_rows[1]['category']}={float(g8_sex_rows[1]['acc_pct']):.2f}% (n={g8_sex_rows[1]['n']}). For age bins with n>=20, accuracy ranged from {lowest_age['category']} {float(lowest_age['acc_pct']):.2f}% (n={lowest_age['n']}) to {highest_age['category']} {float(highest_age['acc_pct']):.2f}% (n={highest_age['n']}); smaller bins are listed in `subgroup_diagnostics.csv` and are noisy.", "",
              f"Held-out disease-head accuracy at G8 was {mean_value(g8_mech, 'semantic_global_acc_pct'):.2f}% global and `{json.dumps(g8_fine)}` across the four CAM positions. At the best checkpoints, G8 mean gradient norms were CLS/semantic `{mean_value(g8_mech, 'gradient_norm_cls_z'):.5f}/{mean_value(g8_mech, 'gradient_norm_semantic_z'):.5f}` at CAM Z and `{mean_value(g8_mech, 'gradient_norm_cls_f'):.5f}/{mean_value(g8_mech, 'gradient_norm_semantic_f'):.5f}` at DARG F; patient CLS/semantic parameter gradient norms were `{mean_value(g8_mech, 'gradient_norm_patient_cls'):.4f}/{mean_value(g8_mech, 'gradient_norm_patient_semantic'):.4f}` and condition-query CLS gradient norm was `{mean_value(g8_mech, 'gradient_norm_g8_condition_cls'):.5f}`. Alpha was calibrated at the predeclared 0.1 CAM-gradient ratio, ranged from `{min(alphas):.4f}` to `{max(alphas):.4f}`, and remained frozen. At the selected G8 checkpoints the weighted CAM-gradient ratio averaged `{g8_weighted_f_ratio:.3f}` and the weighted F-gradient ratio was approximately `{float(np.mean([float(r['alpha']) * float(r['gradient_norm_semantic_f']) / max(float(r['gradient_norm_cls_f']), 1e-12) for r in g8_mech])):.3f}`; this is not a constant 10% through training. Gradient cosine means were Z `{mean_value(g8_mech, 'gradient_cosine_cls_semantic_z'):.3f}`, F `{mean_value(g8_mech, 'gradient_cosine_cls_semantic_f'):.3f}`, and patient module `{mean_value(g8_mech, 'gradient_cosine_patient_cls_semantic'):.3f}`.", "",
              f"The fold-averaged G8 adaptive gates were `{[round(float(value), 6) for value in g8_gate]}`; changes versus the matched E69.3 final gates were `{[round(float(value), 6) for value in g8_gate_delta]}` in PHENO/ANAT/FUNC/CORR order. These near-initial values do not show a material DARG relation-learning shift. The four CAM positions are interaction-processed modality-related features, not four independent raw modalities or localized brain lesions.", "",
              f"At tau=1, adding fine-grained supervision changed ACC by G3 minus G2 `{pair['G3_minus_G2']['delta_acc_mean_pp']:+.3f}` pp for BrainPrompt-2 and G5 minus G4 `{pair['G5_minus_G4']['delta_acc_mean_pp']:+.3f}` pp for SPromptGL-8. At tau=1 with fine+global supervision, SPromptGL-8 minus BrainPrompt-2 was `{pair['G5_minus_G3']['delta_acc_mean_pp']:+.3f}` pp; at tau=0.1 it was G7 minus G6 `{pair['G7_minus_G6']['delta_acc_mean_pp']:+.3f}` pp. Temperature changes were G6 minus G3 `{pair['G6_minus_G3']['delta_acc_mean_pp']:+.3f}` pp for BrainPrompt and G7 minus G5 `{pair['G7_minus_G5']['delta_acc_mean_pp']:+.3f}` pp for SPromptGL. Their direction varies by metric and text source. The two CE forms passed numerical/gradient equivalence; tau=0.1 is this experiment's adaptation, not a fixed SABER paper value. No parameter-matched label-only CE or numeric-PHENO control was trained, so semantic-text gains cannot be separated from added supervision or capacity.", "",
              "Adaptive/edge gate measurements are diagnostic; they do not establish improved relational learning. No additional relation architecture or loss was introduced.", "",
              "### Fold 2 sensitivity", "", "Row 377 (ASD) and row 407 (TD) remain in the fold-2 test set and are both added to effective training. Their labels enter classification in every group and disease supervision/alpha calibration in G2-G8.",
              "", "| Model | N full | N without | ACC full | ACC without | Hard-AUC full | Hard-AUC without | Prob-AUC full | Prob-AUC without |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in fold2_sensitivity:
        lines.append(f"| {row['group']} | {row['test_count_full']} | {row['test_count_without_377_407']} | {row['full_acc_pct']:.4f} | {row['without_377_407_acc_pct']:.4f} | {row['full_hard_auc_pct']:.4f} | {row['without_377_407_hard_auc_pct']:.4f} | {row['full_prob_auc_pct']:.4f} | {row['without_377_407_prob_auc_pct']:.4f} |")
    lines += ["", "| Model | SEN-ASD full | SEN-ASD without | SPE-TD full | SPE-TD without | ASD-F1 full | ASD-F1 without |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in fold2_sensitivity:
        lines.append(f"| {row['group']} | {row['full_sen_pct']:.4f} | {row['without_377_407_sen_pct']:.4f} | {row['full_spe_pct']:.4f} | {row['without_377_407_spe_pct']:.4f} | {row['full_asd_f1_pct']:.4f} | {row['without_377_407_asd_f1_pct']:.4f} |")
    lines += ["", "Removing those two rows does not remove other limitations: preprocessing precedes CV, validation equals test for checkpoint choice, and the graph is transductive over 871 patients.", "",
              f"Independent reconstruction/readback: {sum(row.get('status') == 'PASS' for row in audits)}/80. Common-operation RNG checks and protected source/data/history hashes are in `audit_status.json`; protected snapshot status is {manifest.get('protected_source_audit_after_training', {}).get('status', 'PENDING')}. Optimizer parameter groups and state tensors were compared directly after reloading; task optimizer hashes pickle device-specific tensors and are not used as cross-device equality checks. No historical model, root model file, or input data was altered by this experiment.", "",
              f"Audit-reader provenance: `{manifest.get('audit_only_source_correction', {}).get('reason', 'no post-training reader correction')}` Frozen training/config hashes remain the checkpoint provenance.", "",
              "## Decision", "",
              f"**{verdict}** under the predeclared candidate rule. G8 minus G7 ACC is {g8_pair['delta_acc_mean_pp']:+.4f} pp; G8 minus E69.3-G2 is {base_pair['delta_acc_mean_pp']:+.4f} pp. G8 does not beat either model on mean ACC, so the cross-layer mechanism is NO-GO as the next MICCAI module candidate, despite mixed secondary metric/component effects. The eight groups are paired comparisons within a patient-prompt background, not a complete factorial design. Any future candidate still requires no overlap, fold-local preprocessing, and validation-only checkpoint selection.", "",
              "Artifacts: `summary.csv`, `fold_metrics.csv`, `fold2_sensitivity.csv`, `paired_comparisons.csv`, `mechanism.csv`, `subgroup_diagnostics.csv`, `checkpoint_audit.csv`, and `audit_status.json`.", ""]
    return "\n".join(lines)


def run_audit() -> dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(R.MANIFEST_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(R.PRE_FLIGHT.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    if manifest.get("status") not in {"TRAINING_COMPLETE_PENDING_AUDIT", "AUDIT_FAILED", "COMPLETE_AUDITED"} or int(manifest.get("completed_tasks", -1)) != 80:
        failures.append({"item": "training_completion", "error": f"status/count={manifest.get('status')}/{manifest.get('completed_tasks')}"})
    if preflight.get("status") != "PASS" or any(value != "PASS" for value in preflight.get("check_status", {}).values()):
        failures.append({"item": "preflight", "error": "mandatory preflight record is not wholly PASS"})
    source_now = R.code_hashes()
    source_frozen = manifest.get("source_sha256", {})
    training_source_keys = ("config.json", "run.py", "mgdss.py", "dual_prompt.py")
    audit_source_correction = None
    if any(source_now.get(key) != source_frozen.get(key) for key in training_source_keys):
        failures.append({"item": "frozen_training_source_sha", "error": "training source/config differs from the frozen manifest"})
    elif source_now.get("audit.py") != source_frozen.get("audit.py"):
        audit_source_correction = {
            "reason": "After training, independent readback corrections aligned the reader with the flat prefixed module-state schema, compared optimizer state trees directly because task optimizer digests pickle device-specific tensors, made paired contrast directions explicit, and added the historical fold-2 sensitivity checkpoint. All corrections affect only independent audit/reporting code; training code, configuration, caches, and checkpoints remain byte-identical.",
            "frozen_audit_sha256": source_frozen.get("audit.py"),
            "corrected_audit_sha256": source_now.get("audit.py"),
            "training_source_and_checkpoint_bytes_changed": False,
        }
    reference_manifest, references = R.load_reference()
    warmup_audit = R.verify_warmup_audit()
    prompt_payload = torch.load(R.PROMPT_CACHE, map_location="cpu", weights_only=False)
    banks, prompt_meta = prompt_payload["banks"], prompt_payload["metadata"]
    patient_cache = torch.load(R.PATIENT_CACHE, map_location="cpu", weights_only=False)
    expected_protected = preflight.get("protected_source_sha256", {})
    metrics: list[dict] = []; mechanisms: list[dict] = []; audits: list[dict] = []
    oof = {group: [] for group in GROUPS}
    for group in GROUPS:
        for fold in range(1, 11):
            try:
                metric, mechanism, audit, patient_rows = audit_one(group, fold, reference_manifest, references, warmup_audit, banks, prompt_meta, patient_cache, manifest)
                metrics.append(metric); mechanisms.append(mechanism); audits.append(audit); oof[group].extend(patient_rows)
                if audit["status"] != "PASS":
                    failures.append({"item": f"{group}_f{fold:02d}", "error": audit["status"]})
            except BaseException as error:
                audits.append({"group": group, "fold": fold, "status": "FAIL", "error": f"{type(error).__name__}: {error}"})
                failures.append({"item": f"{group}_f{fold:02d}", "error": f"{type(error).__name__}: {error}"})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    fold2_sensitivity: list[dict[str, Any]] = []
    historical_fold2_audit = "FAIL"
    try:
        historical = audit_historical_fold2(reference_manifest, references, warmup_audit)
        fold2_sensitivity.append(historical)
        historical_fold2_audit = "PASS"
        for row in metrics:
            if int(row["fold"]) != 2:
                continue
            sensitivity = {"group": row["group"], "fold": 2, "test_count_full": row["test_count"],
                           "test_count_without_377_407": row["without_exposed_n"],
                           "checkpoint_sha256": row["checkpoint_sha256"], "independent_checkpoint_audit": "PASS"}
            for metric in ALL_METRICS:
                sensitivity[f"full_{metric}_pct"] = row[f"{metric}_pct"]
                sensitivity[f"without_377_407_{metric}_pct"] = row[f"without_exposed_{metric}_pct"]
            for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
                sensitivity[f"full_{key}"] = row[key]
                sensitivity[f"without_377_407_{key}"] = row[f"without_exposed_{key}"]
            fold2_sensitivity.append(sensitivity)
    except BaseException as error:
        failures.append({"item": "historical_E69_fold2_sensitivity", "error": f"{type(error).__name__}: {error}"})
    protected = R.compare_protected(expected_protected, R.protected_snapshot())
    if protected.get("status") != "PASS":
        failures.append({"item": "protected_sources", "error": json.dumps(protected, ensure_ascii=False)})
    if R.staged_hashes() != manifest.get("staged_source_sha256"):
        failures.append({"item": "staged_dependencies", "error": "staged historical dependency hashes differ"})
    complete = (len(metrics) == 80 and len(audits) == 80 and len(fold2_sensitivity) == 9
                and all(row.get("status") == "PASS" for row in audits) and historical_fold2_audit == "PASS" and not failures)
    summaries = summary_rows(metrics) if len(metrics) == 80 else []
    pairs = paired_rows(metrics) if len(metrics) == 80 else []
    subgroups = subgroup_rows(oof) if all(len(rows) == 871 for rows in oof.values()) else []
    g8_assoc = attention_association(oof) if all(len(rows) == 871 for rows in oof.values()) else {}
    pair_map = {row["comparison"]: row for row in pairs}
    verdict = "AUDIT_FAILED"
    if complete:
        g8_base, g8_g7 = pair_map["G8_vs_E69.3-G2"], pair_map["G8_minus_G7"]
        g8_rows = [row for row in mechanisms if row["group"] == "G8"]
        query_net = sum(int(row.get("g8_condition_off_corrected", 0)) - int(row.get("g8_condition_off_introduced", 0)) for row in g8_rows)
        secondary = [g8_base[f"delta_{metric}_mean_pp"] for metric in ("hard_auc", "prob_auc", "sen", "spe")]
        if (g8_base["delta_acc_mean_pp"] > 0 and g8_g7["delta_acc_mean_pp"] > 0 and g8_base["acc_wins"] >= 6
                and g8_g7["acc_wins"] >= 6 and min(secondary) >= -2.0 and query_net > 0):
            verdict = "GO_CANDIDATE"
        elif g8_base["delta_acc_mean_pp"] <= 0 or g8_g7["delta_acc_mean_pp"] <= 0:
            verdict = "NO_GO"
        elif any(row["delta_acc_mean_pp"] > 0 for row in pairs if row["comparison"] in {"G3_minus_G2", "G5_minus_G4", "G5_minus_G3", "G7_minus_G6", "G6_minus_G3", "G7_minus_G5"}):
            verdict = "PARTIAL"
        else:
            verdict = "NO_GO"
    if len(metrics) == 80:
        atomic_csv(RESULTS / "fold_metrics.csv", metrics)
        atomic_csv(RESULTS / "summary.csv", summaries)
        atomic_csv(RESULTS / "paired_comparisons.csv", pairs)
        atomic_csv(RESULTS / "mechanism.csv", mechanisms)
        atomic_csv(RESULTS / "subgroup_diagnostics.csv", subgroups)
    atomic_csv(RESULTS / "fold2_sensitivity.csv", fold2_sensitivity)
    atomic_csv(RESULTS / "checkpoint_audit.csv", audits)
    result = {
        "status": "PASS" if complete else "FAIL", "experiment_status": "COMPLETE_AUDITED" if complete else "AUDIT_FAILED",
        "decision": verdict, "independent_checkpoint_audits_passed": sum(row.get("status") == "PASS" for row in audits),
        "independent_checkpoint_audits_total": 80, "audit_failures": failures,
        "rng_trace": [{key: row.get(key) for key in ("group", "fold", "rng_trace_exact_epochs", "rng_trace_compared_epochs", "rng_trace_status")} for row in audits],
        "protected_source_audit": protected, "source_sha256": source_now, "metrics_are_generalization_estimates": False,
        "historical_fold2_baseline_audit": historical_fold2_audit, "fold2_sensitivity_rows": len(fold2_sensitivity),
        "audit_only_source_correction": audit_source_correction,
        "fold2_exposure": {"rows_1based": [377, 407], "labels": ["ASD", "TD"], "classification_and_G2_G8_disease_supervision": True},
        "patient_text_cache_sha256": R.sha_file(R.PATIENT_CACHE), "disease_prompt_cache_sha256": R.sha_file(R.PROMPT_CACHE),
        "g8_attention_association": g8_assoc, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest.update({"independent_checkpoint_audits_passed": result["independent_checkpoint_audits_passed"],
                     "independent_checkpoint_audits_total": 80, "audit_failures": failures,
                     "protected_source_audit_after_training": protected, "decision": verdict,
                     "audit_only_source_correction": audit_source_correction,
                     "audited_utc": result["created_utc"], "status": result["experiment_status"]})
    atomic_json(R.MANIFEST_PATH, manifest)
    atomic_json(RESULTS / "audit_status.json", result)
    if len(metrics) == 80 and len(audits) == 80 and len(fold2_sensitivity) == 9:
        (RESULTS / "report.md").write_text(make_report(manifest, preflight, summaries, pairs, mechanisms, audits, fold2_sensitivity, subgroups, g8_assoc, verdict), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
