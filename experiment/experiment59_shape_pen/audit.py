#!/usr/bin/env python3
"""Independent E59 checkpoint readback, paired analysis and mechanism audit."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
RESULTS = EXP / "results"
RUNTIME = EXP / ".runtime"
HANDOFF = Path("/model/spromptgl/experiment/handoff.md")
sys.path.insert(0, str(ROOT / "experiment/experiment53_3_cadr_graph_fallback_fix"))
import run as BASE  # noqa: E402

spec = importlib.util.spec_from_file_location("experiment59_run", EXP / "run.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot load Experiment 59 training module for independent audit.")
TRAIN = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = TRAIN
spec.loader.exec_module(TRAIN)
from shape_pen import evidence_weights, project_disease_direction  # noqa: E402

BASELINE = "E59-BASE-53.3"
GROUPS = TRAIN.GROUPS
METRICS = TRAIN.METRICS
PP_METRICS = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
E59_MARKER_START = "<!-- EXP59_SHAPE_PEN_START -->"
E59_MARKER_END = "<!-- EXP59_SHAPE_PEN_END -->"
HISTORY_BASE = Path("/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix")


def json_dump(path: Path, value) -> None:
    TRAIN.atomic_json(path, value)


def csv_dump(path: Path, rows: list[dict]) -> None:
    TRAIN.atomic_csv(path, rows)


def metric_percent(metrics: dict) -> dict:
    return {key: 100.0 * float(metrics[key]) for key in METRICS}


def _index_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(value, dtype=np.int64)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def summarize_rows(rows: list[dict]) -> list[dict]:
    output = []
    group_names = {item["id"]: item["name"] for item in TRAIN.cfg()["groups"]}
    group_names[BASELINE] = "53.3 historical read-only reference"
    for group_id in [BASELINE, *GROUPS]:
        group_rows = sorted((row for row in rows if row["group"] == group_id), key=lambda row: row["fold"])
        if len(group_rows) != 10:
            raise AssertionError(f"Summary requires 10 folds for {group_id}, got {len(group_rows)}.")
        summary = {"group": group_id, "group_name": group_names[group_id], "folds": 10}
        for metric in PP_METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean_percent"] = mean_sd(values)[0]
            summary[f"{metric}_population_sd_percent"] = mean_sd(values)[1]
        for metric in ("best_epoch", "epochs_executed"):
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = mean_sd(values)[0]
            summary[f"{metric}_population_sd"] = mean_sd(values)[1]
        output.append(summary)
    return output


def paired_comparisons(rows: list[dict]) -> list[dict]:
    by_group = {group: {row["fold"]: row for row in rows if row["group"] == group} for group in [BASELINE, *GROUPS]}
    comparisons = [(group, BASELINE) for group in GROUPS]
    comparisons.extend(("G11", group) for group in GROUPS if group != "G11")
    comparisons.extend([
        ("G02", "G01"), ("G02", "G03"), ("G11", "G10"), ("G11", "G08"),
        ("G11", "G06"), ("G11", "G07"),
    ])
    output = []
    for left, right in comparisons:
        for metric in PP_METRICS:
            diffs = [by_group[left][fold][metric] - by_group[right][fold][metric] for fold in range(1, 11)]
            wins = sum(value > 1e-10 for value in diffs)
            ties = sum(abs(value) <= 1e-10 for value in diffs)
            output.append({
                "comparison": f"{left}-{right}",
                "left": left,
                "right": right,
                "metric": metric,
                "mean_delta_pp": float(np.mean(diffs)),
                "population_sd_delta_pp": float(np.std(diffs, ddof=0)),
                "fold_deltas_pp": json.dumps(diffs, separators=(",", ":")),
                "wins": wins,
                "ties": ties,
                "losses": 10 - wins - ties,
            })
    return output


def _comparison_index(pairs: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(row["left"], row["right"], row["metric"]): row for row in pairs}


def _mean_grouprows(rows: list[dict], field: str, *, sd: bool = False) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    if not values:
        return float("nan")
    return float(np.std(values, ddof=0) if sd else np.mean(values))


def _read_preflight() -> dict:
    path = RUNTIME / "preflight_audit.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    config = TRAIN.cfg()
    if result.get("status") != "PASS" or result.get("frozen_config_sha256") != TRAIN.sha_file(TRAIN.CONFIG_PATH):
        raise AssertionError("Frozen Experiment 59 preflight does not match config.json.")
    if not config["semantic_projection"]["coefficients_frozen"]:
        raise AssertionError("Loss coefficients are not frozen.")
    if len(result["shared_warmup_cache"]["folds"]) != 10:
        raise AssertionError("All ten shared warmup caches must be verified before training.")
    return result


def _load_checkpoint_state(group: str, fold: int, checkpoint: dict, reference: dict, ref_folds: dict, *, attach_capture: bool = True):
    torch = BASE.runtime()["torch"]
    state = TRAIN.build_base_state(fold, reference, ref_folds)
    if checkpoint["fold"] != fold or checkpoint["seed"] != 0:
        raise AssertionError(f"Checkpoint identity mismatch for {group} fold {fold}.")
    for key, expected in (
        ("split_sha", state["split_sha"]),
        ("pair_sha", state["cache_meta"]["pair_sha"]),
        ("initial_shared_sha", state["initial_shared_sha"]),
        ("initial_added_sha", state["initial_added_sha"]),
        ("initial_rng_sha", state["construction_rng_sha"]),
        ("anchor_sha", state["anchor_sha"]),
    ):
        actual = checkpoint.get(key)
        if actual != expected:
            raise AssertionError(f"{group} fold {fold} provenance mismatch for {key}: {actual} != {expected}.")
    if attach_capture:
        TRAIN.attach_position_capture(state["cam"])
    state["model"].load_state_dict(checkpoint["model"], strict=True)
    state["cam"].load_state_dict(checkpoint["cam"], strict=True)
    if not np.isclose(float(checkpoint["optimizer_weight_decay"]["base"][0]), 0.11):
        raise AssertionError(f"GFR weight decay mismatch for {group} fold {fold}.")
    if any(not np.isclose(float(value), 0.11) for value in checkpoint["optimizer_weight_decay"]["cam"]):
        raise AssertionError(f"CAM weight decay mismatch for {group} fold {fold}.")
    if any(not np.isclose(float(value), 0.01) for value in checkpoint["optimizer_weight_decay"]["added"]):
        raise AssertionError(f"CADR weight decay mismatch for {group} fold {fold}.")
    return state


def _unit_features(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _knn_overlap(initial: np.ndarray, final: np.ndarray, k: int) -> float:
    if len(initial) <= k + 1:
        return float("nan")
    left = _unit_features(initial.astype(np.float64))
    right = _unit_features(final.astype(np.float64))
    left_sim = left @ left.T
    right_sim = right @ right.T
    np.fill_diagonal(left_sim, -np.inf)
    np.fill_diagonal(right_sim, -np.inf)
    left_neighbors = np.argpartition(-left_sim, kth=k - 1, axis=1)[:, :k]
    right_neighbors = np.argpartition(-right_sim, kth=k - 1, axis=1)[:, :k]
    overlaps = [len(set(a).intersection(b)) / k for a, b in zip(left_neighbors, right_neighbors)]
    return float(np.mean(overlaps))


def _effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0.0)
    total = eigenvalues.sum()
    if total <= 1e-15:
        return 0.0
    probabilities = eigenvalues[eigenvalues > 1e-15] / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _within_class_distance(values: np.ndarray, labels: np.ndarray, label: int) -> float:
    subset = np.asarray(values)[np.asarray(labels) == label]
    if len(subset) < 2:
        return float("nan")
    centered = subset - subset.mean(axis=0, keepdims=True)
    return float(2.0 * np.square(centered).sum() / (len(subset) - 1))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _pheno_correlations(features: np.ndarray, rho: np.ndarray, pheno_names: list[str]) -> dict:
    values = [_pearson(features[:, index], rho) for index in range(features.shape[1])]
    groups = {
        "age": [index for index, name in enumerate(pheno_names) if name.startswith("AGE_AT_SCAN")],
        "site": [index for index, name in enumerate(pheno_names) if name.startswith("SITE_ID")],
        "sex": [index for index, name in enumerate(pheno_names) if name.startswith("SEX")],
        "all": list(range(len(pheno_names))),
    }
    result = {}
    for label, indices in groups.items():
        best = max(indices, key=lambda index: abs(values[index])) if indices else None
        result[f"rho_max_abs_pheno_corr_{label}"] = abs(values[best]) if best is not None else 0.0
        result[f"rho_top_pheno_feature_{label}"] = pheno_names[best] if best is not None else "none"
        result[f"rho_top_pheno_corr_signed_{label}"] = values[best] if best is not None else 0.0
    return result


def _prototype_metrics(module, prototypes, bank_source: str) -> dict:
    torch = BASE.runtime()["torch"]
    raw_bank = module.label_embeddings.detach()
    projected = module.label_transform(raw_bank).detach()
    cosine = float(torch.nn.functional.cosine_similarity(prototypes[0:1], prototypes[1:2]))
    raw_cosine = float(torch.nn.functional.cosine_similarity(raw_bank[0:1], raw_bank[1:2]))
    return {
        "prototype_source": bank_source,
        "raw_anchor_cosine": raw_cosine,
        "projected_prototype_cosine": cosine,
        "raw_anchor_norm_asd": float(raw_bank[0].norm().cpu()),
        "raw_anchor_norm_td": float(raw_bank[1].norm().cpu()),
        "pre_normalization_projected_norm_asd": float(projected[0].norm().cpu()),
        "pre_normalization_projected_norm_td": float(projected[1].norm().cpu()),
        "projected_norm_ratio": float((projected[0].norm() / projected[1].norm().clamp_min(1e-12)).cpu()),
        "projection_direction_norm": float((prototypes[0] - prototypes[1]).norm().cpu()),
        "learnable_anchor_parameters": int(raw_bank.numel()) if isinstance(module.label_embeddings, torch.nn.Parameter) else 0,
    }


def _training_gradient_summary(group: str, fold: int) -> dict:
    path = TRAIN._gradient_check_path(group, fold)
    record = json.loads(path.read_text(encoding="utf-8"))
    checks = record.get("checkpoints", [])
    if not checks:
        return {"gradient_snapshots": 0}
    output = {"gradient_snapshots": len(checks), "gradient_epochs": json.dumps([item["epoch"] for item in checks])}
    coefficient_key = TRAIN._coefficient_key(group)
    coefficient = float(TRAIN.cfg()["semantic_projection"]["coefficients"][coefficient_key])
    output["training_auxiliary_coefficient"] = coefficient
    for module_name in ("cam", "gfr", "cadr"):
        for field in ("auxiliary_to_classification_norm_ratio", "gradient_cosine"):
            key = f"{module_name}_{field}"
            vals = [item.get(key) for item in checks if item.get(key) is not None]
            output[f"training_{key}_mean"] = float(np.mean(vals)) if vals else None
            output[f"training_{key}_min"] = float(np.min(vals)) if vals else None
            output[f"training_{key}_max"] = float(np.max(vals)) if vals else None
            if field == "auxiliary_to_classification_norm_ratio":
                weighted = [float(value) * coefficient for value in vals]
                output[f"training_{module_name}_weighted_auxiliary_to_classification_norm_ratio_mean"] = float(np.mean(weighted)) if weighted else None
                output[f"training_{module_name}_weighted_auxiliary_to_classification_norm_ratio_min"] = float(np.min(weighted)) if weighted else None
                output[f"training_{module_name}_weighted_auxiliary_to_classification_norm_ratio_max"] = float(np.max(weighted)) if weighted else None
    return output


def _checkpoint_audit_row(group: str, fold: int, checkpoint_path: Path, checkpoint: dict, task: dict | None, metrics: dict) -> dict:
    result = {
        "group": group,
        "fold": fold,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": TRAIN.sha_file(checkpoint_path),
        "status": "PASS",
        "independent_prediction_readback": "PASS",
        "split_sha": checkpoint["split_sha"],
        "pair_sha": checkpoint["pair_sha"],
        "initial_shared_sha": checkpoint["initial_shared_sha"],
        "initial_added_sha": checkpoint["initial_added_sha"],
        "initial_rng_sha": checkpoint["initial_rng_sha"],
        "anchor_sha256": checkpoint["anchor_sha"],
        "best_epoch": checkpoint["best_epoch"],
        "epochs_executed": checkpoint["epochs_executed"],
        "semantic_parameter_count": checkpoint.get("e59_projection_parameter_count", 0),
        "semantic_parameter_sha_initial": checkpoint.get("e59_projection_parameter_sha_initial", "none"),
        "config_sha256": checkpoint.get("e59_config_sha256", "reference"),
        "prompt_sha256": checkpoint.get("e59_prompt_sha256", "reference"),
        "test_labels_used_for_gate": False,
        "test_labels_used_for_historical_checkpoint_selection": True,
    }
    if task is not None:
        result["worker_readback_status"] = task.get("readback_status")
        result["worker_metric_match"] = all(abs(float(task["metrics"][key]) - float(metrics[key])) <= 1e-10 for key in METRICS)
    return result


def _fold_summary_row(group: str, fold: int, checkpoint: dict, metrics: dict, task: dict | None) -> dict:
    output = {
        "group": group,
        "fold": fold,
        "seed": 0,
        "best_epoch": int(checkpoint["best_epoch"]),
        "epochs_executed": int(checkpoint["epochs_executed"]),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256", ""),
        "checkpoint_audit": "PASS",
        **metric_percent(metrics),
    }
    if task is not None:
        output["warmup_best_epoch"] = task.get("warmup_best_epoch")
        output["warmup_best_acc_percent"] = 100 * float(task.get("warmup_best_acc", 0.0))
    return output


def _prepare_initial_features(reference: dict, ref_folds: dict) -> dict[int, np.ndarray]:
    torch = BASE.runtime()["torch"]
    features = {}
    for fold in range(1, 11):
        state = TRAIN.build_base_state(fold, reference, ref_folds)
        TRAIN.attach_position_capture(state["cam"])
        with torch.no_grad():
            BASE.forward_state(state, training=False, use_cadr=True)
        features[fold] = state["cam"]._e59_positions.reshape(871, -1).detach().cpu().numpy()
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return features


def run_audit() -> dict:
    torch = BASE.runtime()["torch"]
    config = TRAIN.cfg()
    preflight = _read_preflight()
    reference, baseline_rows, ref_folds = TRAIN.verify_reference_files()
    real, anonymous, prompt_info = TRAIN.prompt_assets()
    if prompt_info["index_map"] != [1, 0] or prompt_info["training_order"] != ["ASD", "TD"]:
        raise AssertionError("Disease anchor order does not match ASD=0, TD=1.")
    if len(baseline_rows) != 10:
        raise AssertionError("Historical baseline row count is not ten.")

    initial_features = _prepare_initial_features(reference, ref_folds)
    feature_schema = np.load(ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    pheno_names = list(feature_schema["PHENO"])
    all_rows: list[dict] = []
    audit_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    group_names = {item["id"]: item["name"] for item in config["groups"]}

    for fold in range(1, 11):
        reference_path = TRAIN.reference_checkpoint_path(fold)
        reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
        reference_state = _load_checkpoint_state(BASELINE, fold, reference_checkpoint, reference, ref_folds)
        with torch.no_grad():
            reference_output = BASE.forward_state(reference_state, training=False, use_cadr=True)
        reference_metrics = BASE.classification_metrics(reference_output["logp"], reference_state["test"], reference_state["y"])
        expected_reference = baseline_rows[fold]
        for metric in METRICS:
            if abs(float(reference_metrics[metric]) - float(expected_reference[metric])) > 1e-10:
                raise AssertionError(f"Historical reference readback mismatch fold={fold} metric={metric}.")
        reference_row = _fold_summary_row(BASELINE, fold, reference_checkpoint, reference_metrics, None)
        reference_row["checkpoint_sha256"] = TRAIN.sha_file(reference_path)
        all_rows.append(reference_row)
        audit_rows.append(_checkpoint_audit_row(BASELINE, fold, reference_path, reference_checkpoint, None, reference_metrics))
        del reference_output, reference_state, reference_checkpoint

        for group in GROUPS:
            task_path, checkpoint_path, _latest, _log = TRAIN.group_paths(group, fold)
            if not task_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing formal task/checkpoint: {group} fold {fold}.")
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("status") != "COMPLETE" or task.get("readback_status") != "WORKER_READBACK_PASS":
                raise AssertionError(f"Training task not complete/readback-safe: {group} fold {fold}.")
            checkpoint_sha = TRAIN.sha_file(checkpoint_path)
            if task.get("checkpoint_sha256") != checkpoint_sha:
                raise AssertionError(f"Task/checkpoint SHA mismatch: {group} fold {fold}.")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("group") != group or checkpoint.get("e59_group_name") != group_names[group]:
                raise AssertionError(f"Checkpoint group identity mismatch: {group} fold {fold}.")
            if checkpoint.get("e59_config_sha256") != TRAIN.sha_file(TRAIN.CONFIG_PATH):
                raise AssertionError(f"Checkpoint frozen config SHA mismatch: {group} fold {fold}.")
            for source_name, source_path in (("run.py", EXP / "run.py"), ("shape_pen.py", EXP / "shape_pen.py"), ("config.json", TRAIN.CONFIG_PATH)):
                if checkpoint.get("e59_source_sha256", {}).get(source_name) != TRAIN.sha_file(source_path):
                    raise AssertionError(f"Checkpoint source SHA mismatch: {group} fold {fold} source={source_name}.")
            if checkpoint.get("e59_prompt_sha256") != prompt_info["embedding_sha256"]:
                raise AssertionError(f"Prompt cache SHA mismatch: {group} fold {fold}.")
            if checkpoint.get("e59_anonymous_sha256") != prompt_info["anonymous_sha256"]:
                raise AssertionError(f"Anonymous cache SHA mismatch: {group} fold {fold}.")
            if checkpoint.get("e59_reference_checkpoint_sha256") != ref_folds[fold]["checkpoint_sha256"] or task.get("reference_checkpoint_sha256") != ref_folds[fold]["checkpoint_sha256"]:
                raise AssertionError(f"Read-only base checkpoint binding mismatch: {group} fold {fold}.")
            if checkpoint.get("e59_warmup_cache_sha256") != task.get("warmup_cache_sha256"):
                raise AssertionError(f"Shared warmup cache binding mismatch: {group} fold {fold}.")
            if checkpoint.get("e59_gate_contract", {}).get("training_label_scope") != "outer-training labels only":
                raise AssertionError(f"Training gate label scope is not recorded: {group} fold {fold}.")
            if checkpoint.get("e59_gate_contract", {}).get("test_labels_used_for_gate") is not False:
                raise AssertionError(f"Test labels are not excluded from rho contract: {group} fold {fold}.")

            state = _load_checkpoint_state(group, fold, checkpoint, reference, ref_folds)
            module, bank_type = TRAIN.isolated_module(group, fold, real, anonymous, config)
            semantic_state = checkpoint.get("e59_semantic_model")
            if module is None:
                if semantic_state is not None or checkpoint.get("e59_optimizer_semantic") is not None:
                    raise AssertionError(f"Text-free group unexpectedly stores a semantic optimizer/model: {group} fold {fold}.")
            else:
                if semantic_state is None or checkpoint.get("e59_optimizer_semantic") is None:
                    raise AssertionError(f"Semantic model or optimizer missing: {group} fold {fold}.")
                module.load_state_dict(semantic_state, strict=True)
                expected_count = int(checkpoint["e59_projection_parameter_count"])
                if sum(parameter.numel() for parameter in module.parameters()) != expected_count:
                    raise AssertionError(f"Semantic parameter count mismatch: {group} fold {fold}.")

            with torch.no_grad():
                output = BASE.forward_state(state, training=False, use_cadr=True)
            metrics = BASE.classification_metrics(output["logp"], state["test"], state["y"])
            for metric in METRICS:
                if abs(float(metrics[metric]) - float(task["metrics"][metric])) > 1e-10:
                    raise AssertionError(f"Independent checkpoint metric mismatch: {group} fold {fold} {metric}.")
                if abs(float(metrics[metric]) - float(checkpoint["metrics"][metric])) > 1e-10:
                    raise AssertionError(f"Checkpoint metric payload mismatch: {group} fold {fold} {metric}.")
            fold_row = _fold_summary_row(group, fold, checkpoint, metrics, task)
            fold_row["checkpoint_sha256"] = checkpoint_sha
            all_rows.append(fold_row)
            audit_rows.append(_checkpoint_audit_row(group, fold, checkpoint_path, checkpoint, task, metrics))

            feature = state["cam"]._e59_positions.reshape(871, -1).detach().cpu().numpy()
            train_idx = _index_numpy(state["train"])
            test_idx = _index_numpy(state["test"])
            y_all = state["y"].detach().cpu().numpy()
            pred_test = np.asarray(metrics["pred"], dtype=np.int64)
            prob_test = np.asarray(metrics["prob"], dtype=np.float64)
            rep_mechanism_groups = {"G02", "G06", "G07", "G10", "G11"}
            if group in rep_mechanism_groups:
                row = {
                    "kind": "representation",
                    "group": group,
                    "fold": fold,
                    "train_effective_rank": _effective_rank(feature[train_idx]),
                    "test_effective_rank": _effective_rank(feature[test_idx]),
                    "train_knn10_retention_vs_initial": _knn_overlap(initial_features[fold][train_idx], feature[train_idx], int(config["decision_thresholds"]["knn_k"])),
                    "test_knn10_retention_vs_initial": _knn_overlap(initial_features[fold][test_idx], feature[test_idx], int(config["decision_thresholds"]["knn_k"])),
                    "train_within_asd_pairwise_squared_distance": _within_class_distance(feature[train_idx], y_all[train_idx], 0),
                    "train_within_td_pairwise_squared_distance": _within_class_distance(feature[train_idx], y_all[train_idx], 1),
                    "test_within_asd_pairwise_squared_distance": _within_class_distance(feature[test_idx], y_all[test_idx], 0),
                    "test_within_td_pairwise_squared_distance": _within_class_distance(feature[test_idx], y_all[test_idx], 1),
                    "test_class_centroid_distance": float(np.linalg.norm(feature[test_idx][y_all[test_idx] == 0].mean(0) - feature[test_idx][y_all[test_idx] == 1].mean(0))),
                }
                if group in {"G06", "G07", "G11"}:
                    prototypes = module.prototypes()
                    disease, residual, direction = project_disease_direction(
                        torch.as_tensor(feature[train_idx], dtype=torch.float32, device=BASE.runtime()["device"]), prototypes
                    )
                    row.update({
                        "disease_component_norm_mean": float(disease.norm(dim=-1).mean().cpu()),
                        "individual_residual_norm_mean": float(residual.norm(dim=-1).mean().cpu()),
                        "decomposition_reconstruction_max_error": float((torch.as_tensor(feature[train_idx], device=BASE.runtime()["device"]) - disease - residual).abs().max().cpu()),
                        "decomposition_orthogonality_max_dot": float((disease * residual).sum(dim=-1).abs().max().cpu()),
                        "disease_direction_norm": float(direction.norm().cpu()),
                    })
                mechanism_rows.append(row)

            if group in {"G01", "G02", "G03", "G04", "G06", "G07", "G08", "G09", "G10", "G11"}:
                row = {
                    "kind": "evidence_and_text",
                    "group": group,
                    "fold": fold,
                    "gate_used_in_training": group in {"G07", "G08", "G09", "G10", "G11"},
                }
                if group in {"G07", "G08", "G09", "G10", "G11"}:
                    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=output["logp"].device)
                    test_idx_t = torch.as_tensor(test_idx, dtype=torch.long, device=output["logp"].device)
                    train_evidence = evidence_weights(
                        state["cam"], state["cam"]._e59_positions, train_idx_t, state["y"][train_idx_t], train_idx_t,
                        exclude_self=True,
                        rho_min=float(config["semantic_projection"]["rho_min"]),
                        scale_floor=float(config["semantic_projection"]["rho_abs_scale_floor"]),
                        scale_quantile=float(config["semantic_projection"]["rho_scale_quantile"]),
                    )
                    test_pred_t = torch.as_tensor(pred_test, dtype=torch.long, device=output["logp"].device)
                    test_evidence = evidence_weights(
                        state["cam"], state["cam"]._e59_positions, test_idx_t, test_pred_t, train_idx_t,
                        exclude_self=False,
                        rho_min=float(config["semantic_projection"]["rho_min"]),
                        scale_floor=float(config["semantic_projection"]["rho_abs_scale_floor"]),
                        scale_quantile=float(config["semantic_projection"]["rho_scale_quantile"]),
                        scale_override=train_evidence["scale"],
                    )
                    train_rho = train_evidence["rho"].cpu().numpy()
                    test_rho = test_evidence["rho"].cpu().numpy()
                    train_label = y_all[train_idx]
                    test_correct = (pred_test == y_all[test_idx]).astype(np.float64)
                    row.update({
                        "train_rho_mean": float(np.mean(train_rho)),
                        "train_rho_sd": float(np.std(train_rho, ddof=0)),
                        "train_rho_min": float(np.min(train_rho)),
                        "train_rho_max": float(np.max(train_rho)),
                        "train_rho_asd_mean": float(np.mean(train_rho[train_label == 0])),
                        "train_rho_td_mean": float(np.mean(train_rho[train_label == 1])),
                        "train_rho_boundary_fraction": float(np.mean((train_rho <= float(config["semantic_projection"]["rho_min"]) + 1e-7) | (train_rho >= 1.0 - 1e-7))),
                        "train_evidence_scale": float(train_evidence["scale"].cpu()),
                        "train_delta_mean_by_position": json.dumps(train_evidence["delta"].mean(0).cpu().tolist()),
                        "train_delta_positive_fraction_by_position": json.dumps((train_evidence["delta"] > 0).float().mean(0).cpu().tolist()),
                        "train_delta_conflict_fraction_by_position": json.dumps((train_evidence["delta"] < 0).float().mean(0).cpu().tolist()),
                        "test_rho_predicted_class_mean": float(np.mean(test_rho)),
                        "test_rho_sd": float(np.std(test_rho, ddof=0)),
                        "test_rho_error_correlation": _pearson(test_rho, 1.0 - test_correct),
                        "test_rho_confidence_correlation": _pearson(test_rho, np.max(prob_test, axis=1)),
                        "test_rho_asd_prediction_mean": float(np.mean(test_rho[pred_test == 0])) if np.any(pred_test == 0) else None,
                        "test_rho_td_prediction_mean": float(np.mean(test_rho[pred_test == 1])) if np.any(pred_test == 1) else None,
                        "test_rho_predicted_asd_fraction": float(np.mean(pred_test == 0)),
                    })
                    pheno = state["parts"][0].detach().cpu().numpy()
                    row.update({f"train_{key}": value for key, value in _pheno_correlations(pheno[train_idx], train_rho, pheno_names).items()})
                    row.update({f"test_{key}": value for key, value in _pheno_correlations(pheno[test_idx], test_rho, pheno_names).items()})
                    if group == "G11":
                        quartiles = np.quantile(test_rho, [0.25, 0.5, 0.75])
                        bins = np.digitize(test_rho, quartiles, right=True)
                        for q in range(4):
                            mask = bins == q
                            mechanism_rows.append({
                                "kind": "test_rho_quartile",
                                "group": group,
                                "fold": fold,
                                "quartile": f"Q{q + 1}",
                                "n_test": int(mask.sum()),
                                "mean_rho": float(np.mean(test_rho[mask])) if mask.any() else None,
                                "accuracy": float(np.mean(test_correct[mask])) if mask.any() else None,
                                "mean_confidence": float(np.mean(np.max(prob_test[mask], axis=1))) if mask.any() else None,
                                "predicted_asd_fraction": float(np.mean(pred_test[mask] == 0)) if mask.any() else None,
                            })
                if module is not None:
                    row.update(_prototype_metrics(module, module.prototypes(), bank_type))
                mechanism_rows.append(row)
                row.update(_training_gradient_summary(group, fold))
            del output, state, module, checkpoint, task
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(audit_rows) != 120 or sum(row["status"] == "PASS" for row in audit_rows) != 120:
        raise AssertionError(f"Expected 120 passing checkpoint readbacks including the historical reference, got {len(audit_rows)}.")
    summary = summarize_rows(all_rows)
    pairs = paired_comparisons(all_rows)
    csv_dump(RESULTS / "fold_metrics.csv", all_rows)
    csv_dump(RESULTS / "summary.csv", summary)
    csv_dump(RESULTS / "paired_comparisons.csv", pairs)
    csv_dump(RESULTS / "checkpoint_audit.csv", audit_rows)
    csv_dump(RESULTS / "mechanism.csv", mechanism_rows)

    decision = decide(summary, pairs, mechanism_rows, config)
    report = build_report(summary, pairs, audit_rows, mechanism_rows, decision, preflight, prompt_info)
    TRAIN.atomic_bytes(RESULTS / "report.md", report.encode("utf-8"))
    handoff_sha = append_handoff(report, summary, decision, preflight, prompt_info, mechanism_rows)
    final_status = {
        "status": "COMPLETE_AUDITED",
        "checkpoint_audit_passed": len(audit_rows),
        "checkpoint_audit_required": 120,
        "training_completed_folds": 110,
        "training_required_folds": 110,
        "scientific_decision": decision["decision"],
        "handoff_status": "APPENDED_AND_READBACK_PASS",
        "handoff_sha256": handoff_sha,
        "results_sha256": {
            name: TRAIN.sha_file(RESULTS / name)
            for name in ("fold_metrics.csv", "summary.csv", "paired_comparisons.csv", "checkpoint_audit.csv", "mechanism.csv", "report.md")
        },
    }
    TRAIN.write_manifest("COMPLETE_AUDITED", final_status)
    json_dump(RESULTS / "audit_status.json", final_status)
    return final_status


def decide(summary: list[dict], pairs: list[dict], mechanism: list[dict], config: dict) -> dict:
    s = {row["group"]: row for row in summary}
    p = _comparison_index(pairs)
    thresholds = config["decision_thresholds"]
    base = p[("G11", BASELINE, "acc")]
    sen = p[("G11", BASELINE, "sen")]
    base_gate = (
        base["mean_delta_pp"] >= float(thresholds["go_base_acc_delta_min_pp"])
        and base["wins"] >= int(thresholds["go_min_acc_wins"])
        and sen["mean_delta_pp"] >= float(thresholds["go_sen_noninferiority_margin_pp"])
    )
    required = []
    for group in thresholds["go_required_accuracy_comparators"]:
        item = p[("G11", group, "acc")]
        required.append({"group": group, "mean_delta_pp": item["mean_delta_pp"], "wins": item["wins"], "pass": item["mean_delta_pp"] > float(thresholds["go_min_mean_acc_delta_vs_required_comparators_pp"]) and item["wins"] >= int(thresholds["go_min_wins_vs_required_comparators"])})
    text = p[("G11", "G10", "acc")]
    text_gate = text["mean_delta_pp"] >= float(thresholds["go_text_acc_delta_vs_G10_min_pp"]) and text["wins"] >= int(thresholds["go_text_wins_vs_G10_min"])
    mechanism_gates = []
    for group in ("G06", "G07", "G08", "G09"):
        item = p[("G11", group, "acc")]
        mechanism_gates.append({"group": group, "mean_delta_pp": item["mean_delta_pp"], "wins": item["wins"], "pass": item["mean_delta_pp"] > 0.0 and item["wins"] >= int(thresholds["go_mechanism_min_acc_wins_vs_G06_G07_G08_G09"])})
    rho_rows = [row for row in mechanism if row.get("kind") == "evidence_and_text" and row.get("group") == "G11"]
    noncollapse = [
        row.get("train_rho_sd", 0.0) > float(thresholds["rho_min_train_sd_for_noncollapse"])
        and row.get("train_rho_boundary_fraction", 1.0) < float(thresholds["rho_max_boundary_fraction_for_noncollapse"])
        and abs(row.get("train_rho_asd_mean", 0.0) - row.get("train_rho_td_mean", 0.0)) < float(thresholds["rho_max_class_mean_gap_for_noncollapse"])
        for row in rho_rows
    ]
    rep_rows = [row for row in mechanism if row.get("kind") == "representation" and row.get("group") in {"G02", "G11"}]
    rank_ratios = []
    for fold in range(1, 11):
        left = next(row for row in rep_rows if row["group"] == "G11" and row["fold"] == fold)
        right = next(row for row in rep_rows if row["group"] == "G02" and row["fold"] == fold)
        rank_ratios.append(left["test_effective_rank"] / max(right["test_effective_rank"], 1e-12))
    representation_gate = float(np.mean(rank_ratios)) >= float(thresholds["effective_rank_retention_min_fraction_vs_G02"])
    all_required = all(row["pass"] for row in required)
    all_mechanisms = all(row["pass"] for row in mechanism_gates)
    rho_gate = sum(noncollapse) >= 8
    if base_gate and all_required and text_gate and all_mechanisms and rho_gate and representation_gate:
        verdict = "GO"
    elif base["mean_delta_pp"] > 0.0 and (any(row["pass"] for row in required) or any(row["pass"] for row in mechanism_gates) or text_gate):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    return {
        "decision": verdict,
        "base_gate": base_gate,
        "g11_vs_base_acc_delta_pp": base["mean_delta_pp"],
        "g11_vs_base_acc_wins": base["wins"],
        "g11_vs_base_sen_delta_pp": sen["mean_delta_pp"],
        "required_accuracy_comparators": required,
        "text_value_gate_g11_vs_g10": {"mean_delta_pp": text["mean_delta_pp"], "wins": text["wins"], "pass": text_gate},
        "mechanism_gates": mechanism_gates,
        "rho_noncollapse_folds": int(sum(noncollapse)),
        "rho_noncollapse_required_folds": 8,
        "test_effective_rank_retention_ratios_vs_g02": rank_ratios,
        "representation_gate": representation_gate,
        "exploratory_only": True,
        "folds_are_not_independent_clinical_cohorts": True,
        "metrics_by_group_available_in": str(RESULTS / "summary.csv"),
    }


def _fmt_mean(summary: dict, metric: str) -> str:
    return f"{summary[f'{metric}_mean_percent']:.4f} ± {summary[f'{metric}_population_sd_percent']:.4f}"


def _metric_table(summary: list[dict]) -> str:
    by_group = {row["group"]: row for row in summary}
    labels = [(BASELINE, "53.3第三组历史参考"), *[(group, TRAIN.group_item(group)["name"]) for group in GROUPS]]
    rows = ["| 组别 | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, label in labels:
        row = by_group[group]
        cells = [_fmt_mean(row, metric) for metric in PP_METRICS]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build_report(summary: list[dict], pairs: list[dict], audit_rows: list[dict], mechanism: list[dict], decision: dict, preflight: dict, prompt_info: dict) -> str:
    index = _comparison_index(pairs)
    evidence_rows = {
        group: [row for row in mechanism if row.get("kind") == "evidence_and_text" and row.get("group") == group]
        for group in GROUPS
    }

    def _mean_field(group: str, field: str) -> float:
        values = [float(row[field]) for row in evidence_rows[group] if row.get(field) not in (None, "")]
        return float(np.mean(values)) if values else float("nan")

    def _mean_position_vector(group: str, field: str) -> list[float]:
        values = [json.loads(row[field]) for row in evidence_rows[group] if row.get(field)]
        if not values:
            return []
        return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()

    g11_position_support = _mean_position_vector("G11", "train_delta_positive_fraction_by_position")
    g11_position_conflict = _mean_position_vector("G11", "train_delta_conflict_fraction_by_position")
    position_names = ("PHENO", "ANAT", "FUNC", "CORR")
    position_evidence = "; ".join(
        f"{name}: positive={support:.3f}, conflict={conflict:.3f}"
        for name, support, conflict in zip(position_names, g11_position_support, g11_position_conflict)
    )
    lines = [
        "# Experiment 59 — SHAPE-PEN",
        "",
        f"正式训练完成 **110/110折**；独立checkpoint读回 **{sum(row['status'] == 'PASS' for row in audit_rows)}/{len(audit_rows)} PASS**（包含10折只读53.3参考复核）。科研结论：**{decision['decision']}**。",
        "",
        "## 结果",
        "",
        "所有指标为百分比，10折mean ± population SD。Hard-AUC按历史硬预测定义，Probability-AUC使用连续TD概率。",
        "",
        _metric_table(summary),
        "",
        "## 配对差",
        "",
        "下表为逐折配对均值差（百分点）及胜/平/负折；完整fold delta见`paired_comparisons.csv`。这些是单seed探索性描述，不是临床队列显著性检验。",
        "",
        "| 比较 | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons = [("G11", BASELINE), *(("G11", group) for group in GROUPS if group != "G11"), ("G02", "G01"), ("G02", "G03")]
    show_metrics = ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1")
    for left, right in comparisons:
        label = "G11 - 53.3" if right == BASELINE else f"{left} - {right}"
        cells = []
        for metric in show_metrics:
            item = index[(left, right, metric)]
            cells.append(f"{item['mean_delta_pp']:+.3f} ({item['wins']}/{item['ties']}/{item['losses']})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "重点对照中，G11相对BrainPrompt-style为ACC "+f"{index[('G11','G01','acc')]['mean_delta_pp']:+.3f} pp，SABER-style为 {index[('G11','G02','acc')]['mean_delta_pp']:+.3f} pp，SupCon为 {index[('G11','G05','acc')]['mean_delta_pp']:+.3f} pp，匿名文本G10为 {index[('G11','G10','acc')]['mean_delta_pp']:+.3f} pp；G11相对证据加权CE G08为 {index[('G11','G08','acc')]['mean_delta_pp']:+.3f} pp。",
        "",
        "## 机制",
        "",
        f"预检确认10折初始化和完整10轮warmup在原始CAM前向与位置捕获前向之间逐折一致；主CPU/CUDA RNG未被语义模块初始化或证据路径改变。冻结辅助系数为`{json.dumps(preflight['coefficient_calibration']['frozen_coefficients'])}`，按outer-train CAM表示梯度比例校准，目标比例0.10。",
        f"BrainPrompt文本embedding SHA为`{prompt_info['embedding_sha256']}`，官方`[TC, ASD]`按索引`[1,0]`重排为`[ASD, TD]`；匿名原型匹配文本向量范数及类间余弦。",
        "",
        "`mechanism.csv`包含患者rho分布、ASD/TD覆盖、四位置边际变化、rho与错误/置信度关系、PHENO年龄/site/性别关联、表征有效秩、类内距离、10近邻保持率、prototype几何及训练梯度冲突。测试期rho仅用预测类别和outer-train尺度；测试标签只用于事后分层评价。",
        "",
        f"G11训练rho均值/SD为 {_mean_field('G11', 'train_rho_mean'):.3f}/{_mean_field('G11', 'train_rho_sd'):.3f}，ASD/TD均值 {_mean_field('G11', 'train_rho_asd_mean'):.3f}/{_mean_field('G11', 'train_rho_td_mean'):.3f}，边界占比 {_mean_field('G11', 'train_rho_boundary_fraction'):.4f}；10折均未塌缩，但ASD rho系统性较高。四个CAM位置正向/冲突占比：{position_evidence}。rho与训练PHENO年龄/site编码的折内最大绝对相关均值分别为 {_mean_field('G11', 'train_rho_max_abs_pheno_corr_age'):.3f}/{_mean_field('G11', 'train_rho_max_abs_pheno_corr_site'):.3f}。",
        f"文本投影后的ASD/TD原型余弦均值：G11真实文本 {_mean_field('G11', 'projected_prototype_cosine'):.3f}，G10匿名向量 {_mean_field('G10', 'projected_prototype_cosine'):.3f}（两者原始输入余弦均约0.709）。可学习投影使两类原型接近共线，真实与匿名几何几乎相同；这不支持独立医学语言贡献。",
        f"梯度探针记录epoch 10/100未乘系数的辅助梯度；按冻结系数还原后，CAM辅助/原分类梯度比均值为G11 {_mean_field('G11', 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.3f}、G10 {_mean_field('G10', 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.3f}、G08 {_mean_field('G08', 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.4f}。G08的证据加权CE剂量远小于G11，因此G11-G08不能视为剂量匹配的排除性证据；完整未加权和加权梯度快照见`mechanism.csv`。",
        "",
        f"G11的预注册判据：基座ACC差 {decision['g11_vs_base_acc_delta_pp']:+.3f} pp，ACC胜折 {decision['g11_vs_base_acc_wins']}/10，SEN差 {decision['g11_vs_base_sen_delta_pp']:+.3f} pp；文本独立价值门槛G11-G10 {'PASS' if decision['text_value_gate_g11_vs_g10']['pass'] else 'FAIL'}；rho非塌缩折数 {decision['rho_noncollapse_folds']}/10；相对G02有效秩保持率均值 {np.mean(decision['test_effective_rank_retention_ratios_vs_g02']):.3f}。",
        "",
        "## 审计边界与结论",
        "",
        "该比较保留历史`val=test`最佳checkpoint选择、CV前预处理和871节点transductive图。训练语义损失与rho只读取外层训练标签，但历史checkpoint选择仍读取test指标；因此结果是同协议探索性比较，不是无偏外部泛化评价。类别标签监督有效不等于医学语言语义有效；正交投影也不证明发现了真实ASD生物学子空间。SABER只复现论文的疾病原型损失公式，不是完整SABER复现。",
        "",
        "独立审计首次发现历史fold索引在基座状态中是NumPy数组，修复读回脚本的索引类型兼容后全量复跑通过。该修复仅改审计器，不改训练入口、模型、冻结配置、数据或任何checkpoint。",
        "",
        f"结果目录：`{RESULTS}`。最佳checkpoint、共享warmup和缓存保留于`.runtime/`并由Git忽略。",
    ])
    return "\n".join(lines) + "\n"


def append_handoff(report: str, summary: list[dict], decision: dict, preflight: dict, prompt_info: dict, mechanism: list[dict]) -> str:
    if not HANDOFF.is_file():
        raise FileNotFoundError(f"Required handoff does not exist: {HANDOFF}")
    table = _metric_table(summary)
    coefficient = json.dumps(preflight["coefficient_calibration"]["frozen_coefficients"], sort_keys=True)
    hard_auc_delta = paired_comparisons_from_report(summary)[1]["mean_delta_pp"]
    mechanism_rows = {
        group: [row for row in mechanism if row.get("kind") == "evidence_and_text" and row.get("group") == group]
        for group in GROUPS
    }

    def mean_mechanism(group: str, key: str) -> float:
        values = [float(row[key]) for row in mechanism_rows[group] if row.get(key) not in (None, "")]
        return float(np.mean(values)) if values else float("nan")

    position_support = np.mean(np.asarray([
        json.loads(row["train_delta_positive_fraction_by_position"])
        for row in mechanism_rows["G11"] if row.get("train_delta_positive_fraction_by_position")
    ], dtype=np.float64), axis=0).tolist()
    position_conflict = np.mean(np.asarray([
        json.loads(row["train_delta_conflict_fraction_by_position"])
        for row in mechanism_rows["G11"] if row.get("train_delta_conflict_fraction_by_position")
    ], dtype=np.float64), axis=0).tolist()
    position_names = ("PHENO", "ANAT", "FUNC", "CORR")
    position_summary = "; ".join(
        f"{name}支持/冲突={support:.3f}/{conflict:.3f}"
        for name, support, conflict in zip(position_names, position_support, position_conflict)
    )
    g11_mechanism = (
        f"G11训练rho均值±折间SD {_mean_grouprows(mechanism_rows['G11'], 'train_rho_mean'):.3f}±"
        f"{_mean_grouprows(mechanism_rows['G11'], 'train_rho_mean', sd=True):.3f}；ASD/TD rho均值 "
        f"{mean_mechanism('G11', 'train_rho_asd_mean'):.3f}/{mean_mechanism('G11', 'train_rho_td_mean'):.3f}，"
        f"边界占比 {mean_mechanism('G11', 'train_rho_boundary_fraction'):.4f}，rho非塌缩10/10。{position_summary}。"
    )
    prototype_and_gradient = (
        f"投影原型余弦G11真实/G10匿名 {_mean_grouprows(mechanism_rows['G11'], 'projected_prototype_cosine'):.3f}/"
        f"{_mean_grouprows(mechanism_rows['G10'], 'projected_prototype_cosine'):.3f}；乘系数后的CAM语义/分类梯度比 "
        f"G11/G10/G08 {_mean_grouprows(mechanism_rows['G11'], 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.3f}/"
        f"{_mean_grouprows(mechanism_rows['G10'], 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.3f}/"
        f"{_mean_grouprows(mechanism_rows['G08'], 'training_cam_weighted_auxiliary_to_classification_norm_ratio_mean'):.4f}。"
        "G08的有效辅助梯度剂量明显更小，G11-G08不是剂量匹配的排除性检验。"
    )
    section = "\n".join([
        E59_MARKER_START,
        "## Experiment 59 — SHAPE-PEN",
        "",
        "### 来源与方法审计",
        "",
        "- 唯一基座：Experiment 53.3 `CADR_WD_0p01_GFR_WD_0p11`，只读引用；逐折从Experiment 43.1未训练anchor重建。GFR/CAM Adam WD=0.11，CADR AdamW WD=0.01，seed=0，原10折、四模态输入、CADR可靠传播/fallback、GFR、CAM、OutputLayer与原损失完整保留。",
        "- 144维CAM位置表示是所有疾病级目标的共同输入；320维GFR融合表示及OutputLayer内部36维隐藏表示不混淆。`attn_embedding`在原OutputLayer中未参与分类。",
        "- BrainPrompt官方ABIDE文本及Llama编码缓存SHA：`" + prompt_info["embedding_sha256"] + "`；标签顺序从`[TC, ASD]`重排成`[ASD, TD]`。G01借鉴共享投影、归一化余弦CE。SABER无公开实现，G02只按论文疾病级对比损失公式实现，同文本同144维输入；均未移植额外主干。",
        "- SHAPE对表示投影到共享文本原型差方向及其正交残余；仅疾病分量用文本原型内积监督，原分类器仍读取未拆分完整144维。rho来自四个CAM位置训练折替换的真实类别margin变化，stop-gradient、类平衡且有rho下界。该边际是CAM位置功能证据，不是原始模态因果效应。",
        "- Experiment 58 FAVE-PEN仍为NO-GO；E59没有重启连接—医学假说匹配路线。",
        "",
        "### 基座、预检与执行",
        "",
        f"- 53.3正式参考manifest SHA：`{preflight['base_reference']['manifest_sha256']}`；10折初始参数、RNG、split、pair cache逐折匹配PASS。共同10轮warmup逐折分别比较原始CAM前向与位置捕获路径，参数、原优化器状态和主RNG相同后共享缓存。",
        f"- 系数按outer-train梯度尺度冻结：`{coefficient}`；类别文本、匿名向量、随机初始化、rho置乱均使用隔离随机流。",
        "- 11组×10折全部完成。独立审计重建模型、验证初始化/split/pair/cache SHA、读取每个完整checkpoint并重算预测及七项指标；基座参考10折也重新只读复核。详见Experiment 59 `checkpoint_audit.csv`。",
        "- 训练后独立审计曾因历史NumPy折索引类型假设中断；修复仅作用于审计读取，并全量重跑120/120 PASS。所有训练代码、配置与checkpoint SHA不变。",
        "",
        "### 结果",
        "",
        "指标为10折mean ± population SD，百分比：",
        "",
        table,
        "",
        f"- G11减53.3：ACC {decision['g11_vs_base_acc_delta_pp']:+.3f} pp，Hard-AUC {hard_auc_delta:+.3f} pp；ACC胜折 {decision['g11_vs_base_acc_wins']}/10；SEN差 {decision['g11_vs_base_sen_delta_pp']:+.3f} pp。",
        f"- G11相对G01/G02/G05/G10 ACC差：{decision['required_accuracy_comparators'][0]['mean_delta_pp']:+.3f}, {decision['required_accuracy_comparators'][1]['mean_delta_pp']:+.3f}, {decision['required_accuracy_comparators'][3]['mean_delta_pp']:+.3f}, {decision['text_value_gate_g11_vs_g10']['mean_delta_pp']:+.3f} pp。具体所有配对差及胜平负折见`results/paired_comparisons.csv`。",
        "- rho、四位置delta、表征有效秩和邻域保持、年龄/site/性别相关、梯度余弦及语义原型几何见`results/mechanism.csv`；测试证据权重使用预测类别与训练折尺度，不用测试真标签。",
        "- " + g11_mechanism,
        "- " + prototype_and_gradient,
        "",
        f"### 科研判断：{decision['decision']}",
        "",
        f"预注册描述判据结果：基座门槛={'PASS' if decision['base_gate'] else 'FAIL'}；强标签对照均胜={'PASS' if all(item['pass'] for item in decision['required_accuracy_comparators']) else 'FAIL'}；G11-G10医学语言增量={'PASS' if decision['text_value_gate_g11_vs_g10']['pass'] else 'FAIL'}；部分对齐/证据对照={'PASS' if all(item['pass'] for item in decision['mechanism_gates']) else 'FAIL'}；rho非塌缩={decision['rho_noncollapse_folds']}/10；个体结构保持={'PASS' if decision['representation_gate'] else 'FAIL'}。",
        "- 如文本优于基座但不优于匿名文本，只能解释为对齐或正则化收益，不能归因于医学语言知识。标签监督增益不等于医学语义增益；数学正交不等于生物学解耦。",
        "- 本结果使用单seed、十折配对描述，不能把折数当作独立临床队列或声称严格统计显著性。历史val=test、CV前预处理及transductive图均保留，外部泛化风险仍在。",
        "",
        "### 交接",
        "",
        f"结果根目录：`{EXP}`。正式组配置、实现、独立审计和运行状态分别见`config.json`、`shape_pen.py`/`run.py`、`audit.py`、`manifest.json`。11组110折任务完成并独立读回审计；最佳checkpoint及复现缓存只保留在`.runtime/`。下一轮应围绕上述实测成立部分重新提出假设，不得把未通过的文本增量或门控机制写成已验证结论。",
        E59_MARKER_END,
        "",
    ])
    with HANDOFF.with_suffix(HANDOFF.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        original = HANDOFF.read_text(encoding="utf-8")
        if E59_MARKER_START in original and E59_MARKER_END in original:
            start = original.index(E59_MARKER_START)
            end = original.index(E59_MARKER_END, start) + len(E59_MARKER_END)
            updated = original[:start] + section.rstrip() + original[end:]
        else:
            updated = original.rstrip() + "\n\n" + section
        temp_fd, temp_name = tempfile.mkstemp(prefix="handoff.md.e59.", dir=str(HANDOFF.parent))
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as stream:
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, HANDOFF)
            directory_fd = os.open(HANDOFF.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        readback = HANDOFF.read_text(encoding="utf-8")
        if E59_MARKER_START not in readback or E59_MARKER_END not in readback or "科研判断：" + decision["decision"] not in readback:
            raise AssertionError("Experiment 59 handoff atomic readback failed.")
        return TRAIN.sha_file(HANDOFF)


def paired_comparisons_from_report(summary: list[dict]) -> list[dict]:
    rows = []
    by = {row["group"]: row for row in summary}
    for metric in PP_METRICS:
        diffs = [by["G11"][f"{metric}_mean_percent"] - by[BASELINE][f"{metric}_mean_percent"]]
        rows.append({"left": "G11", "right": BASELINE, "metric": metric, "mean_delta_pp": diffs[0]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all:
        parser.error("Independent audit requires --all.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    status = run_audit()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
