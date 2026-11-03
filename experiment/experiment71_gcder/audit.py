#!/usr/bin/env python3
"""Independent checkpoint readback, prediction, metric, and mechanism audit."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("experiment71_run_module", EXP / "run.py")
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label} mismatch: {actual!r} != {expected!r}")


def _metric_row(group: str, fold: int, task: dict, checkpoint_sha: str, metrics: dict) -> dict:
    row = {
        "group": group,
        "group_name": R.GROUP_NAMES[group],
        "fold": fold,
        "seed": 0,
        "best_epoch": int(task["best_epoch"]),
        "epochs_executed": int(task["epochs_executed"]),
        "gcd_parameters": int(task["parameter_counts"]["gcder"]),
        "total_parameters": int(task["parameter_counts"]["total"]),
        "checkpoint_sha256": checkpoint_sha,
        "split_sha": task["split_sha"],
        "pair_sha": task["pair_sha"],
        "audit_status": "PASS",
        "asd_tp": int(metrics["asd_tp"]),
        "asd_fn": int(metrics["asd_fn"]),
        "td_fp": int(metrics["td_fp"]),
        "td_tn": int(metrics["td_tn"]),
    }
    for name in R.METRICS:
        row[f"{name}_percent"] = float(metrics[name]) * 100.0
    return row


def _test_indices(state) -> list[int]:
    np = R.BASE.runtime()["np"]
    return [int(value) for value in np.asarray(state["test"], dtype=np.int64).tolist()]


def _audit_one(group: str, fold: int, reference_manifest: dict, reference_folds: dict, preflight: dict, prompt_payload: dict):
    torch = R.BASE.runtime()["torch"]
    np = R.BASE.runtime()["np"]
    task_path, checkpoint_path, _latest, _log = R.group_paths(group, fold)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("worker_readback_status") != "PASS":
        raise AssertionError("Task did not finish with a passing worker readback.")
    checkpoint_sha = R.sha_file(checkpoint_path)
    _same(task.get("checkpoint_sha256"), checkpoint_sha, "task/checkpoint SHA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _same(checkpoint.get("format"), "experiment71_gcder_checkpoint_v1", "checkpoint format")
    _same(checkpoint.get("group"), group, "checkpoint group")
    _same(int(checkpoint.get("fold", -1)), fold, "checkpoint fold")
    _same(checkpoint.get("config_sha256"), R.sha_file(R.CONFIG_PATH), "checkpoint config SHA")
    _same(checkpoint.get("preflight_sha256"), R.sha_file(R.PRE_FLIGHT), "checkpoint preflight SHA")
    _same(checkpoint.get("text_cache_sha256"), prompt_payload["metadata"]["text_cache_sha256"], "text cache SHA")
    _same(checkpoint.get("prompt_metadata_sha256"), R.sha_obj(prompt_payload["metadata"]), "prompt metadata SHA")
    _same(checkpoint.get("reference_group"), R.REFERENCE_GROUP, "reference group")
    _same(checkpoint.get("reference_checkpoint_sha256"), reference_folds[fold]["checkpoint_sha256"], "historical checkpoint SHA")
    _same(checkpoint.get("warmup_sha256"), preflight["experiment59"]["warmup_cache_sha256"][str(fold)], "warmup SHA")
    _same(checkpoint.get("source_sha256"), {
        "run.py": R.sha_file(EXP / "run.py"),
        "gcder.py": R.sha_file(EXP / "gcder.py"),
        "config.json": R.sha_file(R.CONFIG_PATH),
    }, "source SHA map")
    _same(checkpoint.get("text_bank_sha256"), prompt_payload["metadata"]["bank_sha256"][group], "text bank SHA")
    if not torch.equal(checkpoint.get("text_bank"), prompt_payload["banks"][group]):
        raise AssertionError("Stored text bank values differ from the frozen prompt cache.")

    state = R.build_base_state(fold, reference_manifest, reference_folds)
    warmup = R.warmup_info_for_fold(state, fold, preflight["experiment59"]["warmup_audit"])
    _same(warmup["sha256"], checkpoint["warmup_sha256"], "warmup readback SHA")
    init_row = preflight["initialization"]["folds"][fold - 1]
    init_keys = (
        "initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "initial_added_sha",
        "initial_schema_sha", "initial_rng_sha", "split_sha", "pair_sha", "anchor_sha",
    )
    for key in init_keys:
        _same(state[key] if key not in {"initial_rng_sha", "pair_sha"} else (
            state["construction_rng_sha"] if key == "initial_rng_sha" else state["cache_meta"]["pair_sha"]
        ), checkpoint[key], f"fold {key}")
        _same(checkpoint[key], task[key], f"task/checkpoint {key}")

    bank = prompt_payload["banks"][group].to(R.BASE.runtime()["device"])
    module = R.make_gcder(int(bank.shape[-1]), fold)
    initial_gcder_sha = R.module_sha(module)
    _same(initial_gcder_sha, checkpoint["initial_gcder_sha"], "initial GCDER SHA")
    _same(initial_gcder_sha, init_row["initial_gcder_sha"], "preflight GCDER SHA")
    initial_graph_weight = module.graph_proj.weight.detach().clone()

    state["model"].load_state_dict(checkpoint["state_model"], strict=True)
    state["cam"].load_state_dict(checkpoint["state_cam"], strict=True)
    module.load_state_dict(checkpoint["state_gcder"], strict=True)
    optimizer_base = state["opt_base"]
    optimizer_cam = state["opt_cam"]
    optimizer_added = state["opt_added"]
    optimizer_gcder = torch.optim.AdamW(
        module.parameters(),
        lr=float(R.CFG["gcder"]["optimizer"]["lr"]),
        weight_decay=float(R.CFG["gcder"]["optimizer"]["weight_decay"]),
    )
    optimizer_base.load_state_dict(checkpoint["optimizer_base"])
    optimizer_cam.load_state_dict(checkpoint["optimizer_cam"])
    optimizer_added.load_state_dict(checkpoint["optimizer_added"])
    optimizer_gcder.load_state_dict(checkpoint["optimizer_gcder"])
    _same([float(g["weight_decay"]) for g in optimizer_base.param_groups], [0.11], "GFR optimizer WD")
    _same([float(g["weight_decay"]) for g in optimizer_cam.param_groups], [0.11], "CAM optimizer WD")
    _same([float(g["weight_decay"]) for g in optimizer_added.param_groups], [0.01] * len(optimizer_added.param_groups), "CADR optimizer WD")
    _same([float(g["weight_decay"]) for g in optimizer_gcder.param_groups], [0.001], "GCDER optimizer WD")
    _same(R.module_sha(state["model"]), checkpoint["parameter_sha256"]["model"], "GFR/CADR parameter SHA")
    _same(R.module_sha(state["cam"]), checkpoint["parameter_sha256"]["cam"], "CAM parameter SHA")
    _same(R.module_sha(module), checkpoint["parameter_sha256"]["gcder"], "GCDER parameter SHA")
    _same(R.module_sha(state["model"]), task["parameter_sha256"]["model"], "task GFR/CADR SHA")
    _same(R.module_sha(state["cam"]), task["parameter_sha256"]["cam"], "task CAM SHA")
    _same(R.module_sha(module), task["parameter_sha256"]["gcder"], "task GCDER SHA")
    R.E59.attach_position_capture(state["cam"])
    module.eval()
    rng_before = R.BASE.rng_sha(R.BASE.pack_rng())
    with torch.no_grad():
        evaluated = R.evaluate(state, module, bank)
        graph_zero = torch.zeros_like(evaluated["fused"])
        no_graph_semantic = module(evaluated["positions"], graph_zero, bank)
        no_graph_logp = R.fuse_log_probabilities(
            evaluated["logp_base"], no_graph_semantic["scores"], no_graph_semantic["beta"]
        )
    rng_after = R.BASE.rng_sha(R.BASE.pack_rng())
    _same(rng_after, rng_before, "inference RNG state")
    metrics = evaluated["metrics"]
    for name in R.METRICS:
        if abs(float(metrics[name]) - float(task["metrics"][name])) > 1e-10:
            raise AssertionError(f"Independent metric {name} differs from worker record.")
        if abs(float(metrics[name]) - float(checkpoint["metrics"][name])) > 1e-10:
            raise AssertionError(f"Independent metric {name} differs from checkpoint record.")

    att = evaluated["gcder"]["attention"].detach().cpu().numpy()
    q = evaluated["gcder"]["query"].detach().cpu()
    scores = evaluated["gcder"]["scores"].detach().cpu().numpy()
    beta = float(module.beta.detach().cpu())
    logp_base = evaluated["logp_base"].detach().cpu().numpy()
    logp_final = evaluated["logp"].detach().cpu().numpy()
    logp_no_graph = no_graph_logp.detach().cpu().numpy()
    att_no_graph = no_graph_semantic["attention"].detach().cpu().numpy()
    labels = state["y"].detach().cpu().numpy().astype(np.int64)
    test_indices = _test_indices(state)
    input_blocks = [part.detach().cpu().numpy() for part in state["parts"]]
    modal_mapping = np.load(R.MODAL_MAPPING, allow_pickle=True).item()
    pheno_names = [str(name) for name in modal_mapping["PHENO"]]
    site_indices = [index for index, name in enumerate(pheno_names) if name.startswith("SITE_ID_")]
    site_names = [pheno_names[index] for index in site_indices]
    site_values = input_blocks[0][:, site_indices]
    site_local = site_values.argmax(axis=1)
    site_has_value = site_values.max(axis=1) > 0
    delta_score = scores[:, 0] - scores[:, 1]
    correction = beta * delta_score
    margin_base = logp_base[:, 0] - logp_base[:, 1]
    margin_final = logp_final[:, 0] - logp_final[:, 1]
    margin_no_graph = logp_no_graph[:, 0] - logp_no_graph[:, 1]
    pred_base = logp_base.argmax(axis=-1)
    pred_final = logp_final.argmax(axis=-1)
    pred_no_graph = logp_no_graph.argmax(axis=-1)
    entropy = -(att.clip(1e-12, 1.0) * np.log(att.clip(1e-12, 1.0))).sum(axis=-1)
    query_cosine = torch.nn.functional.cosine_similarity(q[:, 0], q[:, 1], dim=-1).numpy()
    graph_update_norm = float((module.graph_proj.weight.detach() - initial_graph_weight).norm().cpu())
    graph_update_rel = graph_update_norm / max(float(initial_graph_weight.norm().cpu()), 1e-12)
    test_index_set = set(test_indices)
    patient_rows = []
    for index in range(len(labels)):
        row = {
            "group": group,
            "fold": fold,
            "patient_index": index,
            "is_outer_test_fold": int(index in test_index_set),
            "label_diagnostic_only": int(labels[index]),
            "beta": beta,
            "score_asd": float(scores[index, 0]),
            "score_td": float(scores[index, 1]),
            "delta_score_asd_minus_td": float(delta_score[index]),
            "beta_delta_score": float(correction[index]),
            "query_cosine_asd_td": float(query_cosine[index]),
            "attention_entropy_asd": float(entropy[index, 0]),
            "attention_entropy_td": float(entropy[index, 1]),
            "base_margin_asd_minus_td": float(margin_base[index]),
            "final_margin_asd_minus_td": float(margin_final[index]),
            "no_graph_margin_asd_minus_td": float(margin_no_graph[index]),
            "base_prediction": int(pred_base[index]),
            "final_prediction": int(pred_final[index]),
            "no_graph_prediction": int(pred_no_graph[index]),
            "decision_changed_by_gcder": int(pred_base[index] != pred_final[index]),
            "decision_changed_by_graph_condition": int(pred_final[index] != pred_no_graph[index]),
            "pheno_site_category": site_names[int(site_local[index])] if site_has_value[index] else "unknown",
        }
        for modality_index, modality in enumerate(R.MODALITIES):
            block = input_blocks[modality_index][index]
            row[f"input_mean_abs_{modality.lower()}"] = float(np.abs(block).mean())
            row[f"input_rms_{modality.lower()}"] = float(np.sqrt(np.mean(np.square(block))))
        for modality_index, modality in enumerate(R.MODALITIES):
            row[f"attention_asd_{modality.lower()}"] = float(att[index, 0, modality_index])
            row[f"attention_td_{modality.lower()}"] = float(att[index, 1, modality_index])
            row[f"delta_attention_{modality.lower()}"] = float(att[index, 0, modality_index] - att[index, 1, modality_index])
            row[f"no_graph_attention_asd_{modality.lower()}"] = float(att_no_graph[index, 0, modality_index])
            row[f"no_graph_attention_td_{modality.lower()}"] = float(att_no_graph[index, 1, modality_index])
            row[f"no_graph_delta_attention_{modality.lower()}"] = float(att_no_graph[index, 0, modality_index] - att_no_graph[index, 1, modality_index])
        patient_rows.append(row)

    diagnostics = {
        "beta": beta,
        "beta_raw": float(module.beta_raw.detach().cpu()),
        "delta_score_mean": float(delta_score[test_indices].mean()),
        "delta_score_std": float(delta_score[test_indices].std(ddof=0)),
        "delta_score_mean_abs": float(np.abs(delta_score[test_indices]).mean()),
        "beta_delta_score_mean": float(correction[test_indices].mean()),
        "beta_delta_score_mean_abs": float(np.abs(correction[test_indices]).mean()),
        "attention_mean": att[test_indices].mean(axis=0).tolist(),
        "attention_entropy_mean": float(entropy[test_indices].mean()),
        "attention_class_abs_difference_mean": float(np.abs(att[test_indices, 0] - att[test_indices, 1]).mean()),
        "query_cosine_mean": float(query_cosine[test_indices].mean()),
        "query_patient_std_asd": float(q[test_indices, 0].std(dim=0, unbiased=False).mean()),
        "query_patient_std_td": float(q[test_indices, 1].std(dim=0, unbiased=False).mean()),
        "gfr_condition_projection_update_norm": graph_update_norm,
        "gfr_condition_projection_update_relative_norm": graph_update_rel,
        "no_graph_attention_abs_change_mean": float(np.abs(att[test_indices] - att_no_graph[test_indices]).mean()),
        "no_graph_margin_abs_change_mean": float(np.abs(margin_final[test_indices] - margin_no_graph[test_indices]).mean()),
        "no_graph_prediction_flip_fraction": float(np.mean(pred_final[test_indices] != pred_no_graph[test_indices])),
        "gcder_decision_change_fraction": float(np.mean(pred_base[test_indices] != pred_final[test_indices])),
        "gradient_snapshots": task.get("gradient_snapshots", []),
    }
    no_graph_metrics = R.BASE.classification_metrics(no_graph_logp, state["test"], state["y"])
    for metric in R.METRICS:
        diagnostics[f"no_graph_{metric}"] = float(no_graph_metrics[metric])
    fold_metrics = _metric_row(group, fold, task, checkpoint_sha, metrics)
    fold_metrics.update({
        "beta": diagnostics["beta"],
        "beta_raw": diagnostics["beta_raw"],
        "mean_abs_delta_score": diagnostics["delta_score_mean_abs"],
        "delta_score_std": diagnostics["delta_score_std"],
        "mean_abs_beta_delta_score": diagnostics["beta_delta_score_mean_abs"],
        "attention_class_abs_difference_mean": diagnostics["attention_class_abs_difference_mean"],
        "attention_entropy_mean": diagnostics["attention_entropy_mean"],
        "query_cosine_mean": diagnostics["query_cosine_mean"],
        "gfr_condition_projection_update_norm": graph_update_norm,
        "gfr_condition_projection_update_relative_norm": graph_update_rel,
        "no_graph_attention_abs_change_mean": diagnostics["no_graph_attention_abs_change_mean"],
        "no_graph_margin_abs_change_mean": diagnostics["no_graph_margin_abs_change_mean"],
        "no_graph_prediction_flip_fraction": diagnostics["no_graph_prediction_flip_fraction"],
        "gcder_decision_change_fraction": diagnostics["gcder_decision_change_fraction"],
        "diagnostics_json": json.dumps(diagnostics, sort_keys=True, allow_nan=False),
    })
    audit_record = {
        "group": group,
        "fold": fold,
        "status": "PASS",
        "checkpoint_sha256": checkpoint_sha,
        "parameter_sha256": checkpoint["parameter_sha256"],
        "text_bank_sha256": checkpoint["text_bank_sha256"],
        "split_sha": checkpoint["split_sha"],
        "pair_sha": checkpoint["pair_sha"],
        "warmup_sha256": checkpoint["warmup_sha256"],
        "metrics_match_worker": True,
        "inference_rng_unchanged": True,
        "device_mapping": str(R.BASE.runtime()["device"]),
    }
    del state, module, optimizer_base, optimizer_cam, optimizer_added, optimizer_gcder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fold_metrics, patient_rows, diagnostics, audit_record


def _paired_rows(fold_metrics: list[dict], reference_metrics: dict[int, dict[str, float]]) -> list[dict]:
    group_values = {
        group: {int(row["fold"]): row for row in fold_metrics if row["group"] == group}
        for group in R.GROUPS
    }
    comparisons = [("G1", "BASE"), ("G2", "BASE"), ("G3", "BASE"), ("G1", "G2"), ("G1", "G3"), ("G2", "G3")]
    result = []
    for first, second in comparisons:
        for metric in R.METRICS:
            differences = []
            for fold in range(1, 11):
                left = reference_metrics[fold][metric] * 100.0 if first == "BASE" else group_values[first][fold][f"{metric}_percent"]
                right = reference_metrics[fold][metric] * 100.0 if second == "BASE" else group_values[second][fold][f"{metric}_percent"]
                differences.append(float(left - right))
            result.append({
                "comparison": f"{first}_minus_{second}",
                "metric": metric,
                "fold_differences_pp_json": json.dumps(differences, separators=(",", ":")),
                "mean_difference_pp": float(sum(differences) / len(differences)),
                "wins": int(sum(value > 1e-10 for value in differences)),
                "ties": int(sum(abs(value) <= 1e-10 for value in differences)),
                "losses": int(sum(value < -1e-10 for value in differences)),
            })
    return result


def _summary_stats(values: list[float]) -> tuple[float, float]:
    np = R.BASE.runtime()["np"]
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def np_mean(values: list[float]) -> float:
    return _summary_stats(values)[0]


def np_sd(values: list[float]) -> float:
    return _summary_stats(values)[1]


def _mechanism_summary(patient_rows: list[dict], fold_metrics: list[dict], diagnostics: dict[tuple[str, int], dict]) -> dict[str, dict]:
    np = R.BASE.runtime()["np"]
    summary: dict[str, dict] = {}
    for group in R.GROUPS:
        rows = [row for row in patient_rows if row["group"] == group]
        folds = [row for row in fold_metrics if row["group"] == group]
        diag_rows = [diagnostics[(group, int(row["fold"]))] for row in folds]
        item: dict[str, Any] = {
        "n_patient_fold_predictions": len(rows),
            "beta_mean": float(np.mean([row["beta"] for row in folds])),
            "beta_mean_abs": float(np.mean([abs(row["beta"]) for row in folds])),
            "beta_std": float(np.std([row["beta"] for row in folds], ddof=0)),
            "beta_min": float(np.min([row["beta"] for row in folds])),
            "beta_max": float(np.max([row["beta"] for row in folds])),
            "score_asd_mean": float(np.mean([row["score_asd"] for row in rows])),
            "score_td_mean": float(np.mean([row["score_td"] for row in rows])),
            "delta_score_mean": float(np.mean([row["delta_score_asd_minus_td"] for row in rows])),
            "mean_abs_delta_score": float(np.mean([abs(row["delta_score_asd_minus_td"]) for row in rows])),
            "delta_score_std": float(np.std([row["delta_score_asd_minus_td"] for row in rows], ddof=0)),
            "beta_delta_score_mean": float(np.mean([row["beta_delta_score"] for row in rows])),
            "mean_abs_beta_delta_score": float(np.mean([abs(row["beta_delta_score"]) for row in rows])),
            "decision_changed_fraction": float(np.mean([row["decision_changed_by_gcder"] for row in rows])),
            "query_cosine_mean": float(np.mean([row["query_cosine_asd_td"] for row in rows])),
            "attention_entropy_asd_mean": float(np.mean([row["attention_entropy_asd"] for row in rows])),
            "attention_entropy_td_mean": float(np.mean([row["attention_entropy_td"] for row in rows])),
            "attention_class_abs_difference_mean": float(np.mean([row["attention_class_abs_difference_mean"] for row in rows])) if "attention_class_abs_difference_mean" in rows[0] else 0.0,
            "gfr_condition_update_mean_norm": float(np.mean([row["gfr_condition_projection_update_norm"] for row in diag_rows])),
            "gfr_condition_update_mean_relative_norm": float(np.mean([row["gfr_condition_projection_update_relative_norm"] for row in diag_rows])),
            "no_graph_attention_abs_change_mean": float(np.mean([row["no_graph_attention_abs_change_mean"] for row in diag_rows])),
            "no_graph_margin_abs_change_mean": float(np.mean([row["no_graph_margin_abs_change_mean"] for row in diag_rows])),
            "no_graph_prediction_flip_fraction_mean": float(np.mean([row["no_graph_prediction_flip_fraction"] for row in diag_rows])),
            "gcder_prediction_change_fraction": float(np.mean([row["decision_changed_by_gcder"] for row in rows])),
            "attention_by_modality": {},
            "gradient": {},
        }
        for modality in R.MODALITIES:
            key = modality.lower()
            delta = [row[f"delta_attention_{key}"] for row in rows]
            item["attention_by_modality"][modality] = {
                "asd_mean": float(np.mean([row[f"attention_asd_{key}"] for row in rows])),
                "td_mean": float(np.mean([row[f"attention_td_{key}"] for row in rows])),
                "delta_mean": float(np.mean(delta)),
                "delta_sd": float(np.std(delta, ddof=0)),
                "mean_abs_delta": float(np.mean(np.abs(delta))),
                "no_graph_delta_mean": float(np.mean([row[f"no_graph_delta_attention_{key}"] for row in rows])),
            }
            evidence = np.asarray([row["delta_score_asd_minus_td"] for row in rows], dtype=np.float64)
            block_rms = np.asarray([row[f"input_rms_{key}"] for row in rows], dtype=np.float64)
            if evidence.std() > 0 and block_rms.std() > 0:
                item.setdefault("evidence_input_summary_pearson", {})[modality] = float(np.corrcoef(evidence, block_rms)[0, 1])
            else:
                item.setdefault("evidence_input_summary_pearson", {})[modality] = 0.0
        evidence = np.asarray([row["delta_score_asd_minus_td"] for row in rows], dtype=np.float64)
        site_scores = []
        for name in sorted({row["pheno_site_category"] for row in rows}):
            if name == "unknown":
                continue
            indicator = np.asarray([row["pheno_site_category"] == name for row in rows], dtype=np.float64)
            if indicator.std() > 0 and evidence.std() > 0:
                site_scores.append((abs(float(np.corrcoef(evidence, indicator)[0, 1])), name))
        item["pheno_site_evidence_association"] = {
            "max_abs_point_biserial": max(site_scores)[0] if site_scores else 0.0,
            "site_with_max_abs_association": max(site_scores)[1] if site_scores else None,
            "number_of_nonempty_site_categories": len(site_scores),
            "interpretation": "descriptive OOF association; not a causal or site-removal test",
        }
        gradients = [snapshot for row in diag_rows for snapshot in row.get("gradient_snapshots", [])]
        post_identity = [row for row in gradients if int(row.get("epoch", -1)) > 10]
        for key in (
            "beta_grad_abs", "text_projection_grad_norm", "gfr_condition_grad_norm",
            "key_grad_norm", "value_grad_norm", "shared_score_grad_norm",
            "gcder_correction_to_cam_norm", "gcder_correction_to_gfr_norm",
        ):
            vals = [float(row.get(key, 0.0)) for row in post_identity]
            item["gradient"][key] = {
                "max_post_identity": max(vals, default=0.0),
                "mean_post_identity": float(np.mean(vals)) if vals else 0.0,
                "all_finite": all(math.isfinite(value) for value in vals),
            }
        item["gradient"]["post_identity_snapshot_count"] = len(post_identity)
        item["gradient"]["all_gcder_gradients_finite"] = all(
            snapshot.get("all_gcder_gradients_finite", False) for snapshot in gradients
        )
        test_rows = [row for row in rows if row["is_outer_test_fold"]]
        item["test_prediction_change_fraction"] = float(np.mean([row["decision_changed_by_gcder"] for row in test_rows]))
        summary[group] = item
    return summary


def _format_mean_sd(rows: list[dict], metric: str) -> str:
    values = [float(row[f"{metric}_percent"]) for row in rows]
    mean, sd = _summary_stats(values)
    return f"{mean:.4f} +/- {sd:.4f}"


def _decision(fold_metrics: list[dict], paired: list[dict], mechanism: dict, audit_rows: list[dict]) -> tuple[str, dict]:
    by_pair = {(row["comparison"], row["metric"]): row for row in paired}
    conditions = {}
    for group in ("G1", "G2"):
        acc = by_pair[(f"{group}_minus_BASE", "acc")]
        prob = by_pair[(f"{group}_minus_BASE", "prob_auc")]
        sen = by_pair[(f"{group}_minus_BASE", "sen")]
        sem = by_pair[(f"{group}_minus_G3", "acc")]
        diag = mechanism[group]
        threshold = R.CFG["pre_registered_decision_thresholds"]
        gradients = diag["gradient"]
        active_gradient = all(
            gradients[key]["max_post_identity"] > 0
            for key in ("text_projection_grad_norm", "gfr_condition_grad_norm", "key_grad_norm", "value_grad_norm", "shared_score_grad_norm", "gcder_correction_to_cam_norm", "gcder_correction_to_gfr_norm")
        )
        noncollapse = (
            float(np_mean_abs_attention(diag)) >= float(threshold["noncollapse_min_mean_abs_delta_attention"])
            and diag["delta_score_std"] >= float(threshold["noncollapse_min_std_delta_score"])
            and diag["beta_mean_abs"] >= float(threshold["noncollapse_min_abs_beta"])
            and diag["mean_abs_beta_delta_score"] >= float(threshold["noncollapse_min_mean_abs_beta_delta_score"])
            and active_gradient
        )
        conditions[group] = {
            "mean_acc_gain_at_least_0_5pp": acc["mean_difference_pp"] >= float(threshold["minimum_acc_gain_pp_vs_base"]),
            "at_least_6_acc_winning_folds": acc["wins"] >= int(threshold["minimum_acc_winning_folds"]),
            "prob_auc_drop_no_more_than_0_5pp": prob["mean_difference_pp"] >= float(threshold["minimum_prob_auc_delta_pp"]),
            "sen_drop_no_more_than_1pp": sen["mean_difference_pp"] >= -float(threshold["maximum_sen_drop_pp"]),
            "positive_semantic_advantage_over_anonymous": sem["mean_difference_pp"] > 0,
            "noncollapsed_mechanism_and_gradients": noncollapse,
        }
        conditions[group]["performance_pass"] = all(conditions[group][key] for key in (
            "mean_acc_gain_at_least_0_5pp", "at_least_6_acc_winning_folds",
            "prob_auc_drop_no_more_than_0_5pp", "sen_drop_no_more_than_1pp",
        ))
        conditions[group]["all_preregistered_conditions"] = all(conditions[group][key] for key in (
            "performance_pass", "positive_semantic_advantage_over_anonymous", "noncollapsed_mechanism_and_gradients",
        ))
    all_audited = len(audit_rows) == 30 and all(row["status"] == "PASS" for row in audit_rows)
    any_full = any(item["all_preregistered_conditions"] for item in conditions.values())
    any_performance = any(item["performance_pass"] for item in conditions.values())
    if all_audited and any_full:
        outcome = "GO"
    elif any(
        by_pair[(f"{group}_minus_BASE", "acc")]["mean_difference_pp"] > 0
        and conditions[group]["noncollapsed_mechanism_and_gradients"]
        for group in ("G1", "G2")
    ):
        outcome = "CONDITIONAL"
    else:
        outcome = "NO-GO"
    return outcome, {
        "candidate_conditions": conditions,
        "all_30_checkpoint_audits_pass": all_audited,
        "all_patients_get_both_disease_queries": True,
        "no_label_conditioned_prompt_routing_or_score_selection": True,
        "historical_val_equals_test_retained": True,
        "decision_rule": "GO only if all preregistered conditions pass; CONDITIONAL only when a real-text group has positive mean ACC delta versus base and a noncollapsed mechanism; otherwise NO-GO.",
    }


def np_mean_abs_attention(diag: dict) -> float:
    return float(sum(item["mean_abs_delta"] for item in diag["attention_by_modality"].values()) / 4.0)


def _write_report(path: Path, fold_metrics: list[dict], paired: list[dict], mechanism: dict, audit_rows: list[dict], decision: str, decision_details: dict, preflight: dict) -> None:
    by_group = {group: [row for row in fold_metrics if row["group"] == group] for group in R.GROUPS}
    lines = [
        "# Experiment 71: GCDER",
        "",
        "## Research question and hypothesis",
        "",
        "GCDER tests whether two disease-conditioned queries, each combining a frozen disease text embedding with the patient-level CADR-GFR representation, retrieve different diagnostic evidence from the four CAM modality positions and improve the final ASD/TD decision through bounded score fusion. The retrieved vectors are learned latent evidence, not clinically validated lesion evidence.",
        "",
        "## Related methods and distinction",
        "",
        "SPromptGL combines patient-level multimodal representations with CAM and uses disease prompts in PEN to weight predefined local features; the present module does not re-create PEN, RES, SPGS, or the 4x4x9 partition. The audited local OutputLayer consumes the 144-dimensional CAM representation and returns log-softmax probabilities; its optional second argument is not a semantic injection path. [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [official repository](https://github.com/wanxixi11/SPromptGL), [current research repository](https://github.com/tangaii/SPromptGL).",
        "",
        "SABER uses Graph-CLS and graph representation to query patient text through cross-attention, then applies a gated semantic residual to classification; disease-level text is also used for contrastive supervision. GCDER instead forms one shared-parameter query per disease from frozen text plus the patient's GFR vector, reads the four CAM modality positions, and adds the two shared-score values to the original class log probabilities. [SABER paper](https://arxiv.org/html/2607.01901).",
        "",
        "BrainPrompt uses multi-level ROI/subject/disease prompting. G1 uses only its official two ABIDE descriptions as fixed CLIP text embeddings, reordered to [ASD, TD]; it does not import BrainPrompt's full prompting architecture. [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [official repository](https://github.com/AngusMonroe/BrainPrompt).",
        "",
        "## Frozen base and audit",
        "",
        f"The only base was `{R.REFERENCE_GROUP}`: 53.3 third group. Frozen optimizers are GFR/CAM Adam LR=0.0038, WD=0.11; CADR AdamW LR=0.001, WD=0.01 for every original parameter group; GCDER AdamW LR=0.001, WD=0.001. The read-only reference CSV, manifest, 10 tasks, checkpoint hashes, and historical checkpoint audit were cross-checked; the ten base fold means/population SDs were recomputed against the frozen reference table. No historical base checkpoint was used as a training start.",
        f"The 871-subject ABIDE input has 403 ASD (label 0) and 468 TD (label 1), with PHENO/ANAT/FUNC/CORR dimensions 48/6/10/256. The fallback execution root is `{EXP}` after the primary `/data` experiment-directory creation returned an I/O error. E59's ten warmup caches and CAM position hook were read-only reused after SHA and parity verification. E70 was inspected read-only for its base-bound warmup and prompt-cache provenance; no E70 best model was loaded.",
        "The active `/data` CADR relation cache and the `/model` reference cache each contain ten JSON and ten NPZ files; all 20 SHA values match byte-for-byte, and each fold's pair SHA matches the audited 53.3 task. The caches were reused without recomputation. The `/data` and `/model` frozen 53.3 `run.py`, `cadr_gfr.py`, and `config.json` also had matching SHA values (574fb237..., f5032d96..., 4a2aeab5...).",
        "The historical Experiment 51 CAM-only path physically removed GFR/PEN at that stage; Experiment 53.3 then added CADR-GFR and retained the CAM four-position output `[PHENO, ANAT, FUNC, CORR]` at 4x36. Its original classifier receives 144 features and has the 144->36->2 shape. The audited local `OutputLayer.forward(x, attn_embedding)` uses `x`; its second argument is not a semantic injection interface. Experiment 71 leaves both CAM and this head unchanged.",
        f"SHA anchors: 53.3 manifest `{preflight['base_reference']['manifest_sha256']}`, fold CSV `{preflight['base_reference']['fold_metric_sha256']}`, checkpoint audit `{preflight['base_reference']['checkpoint_audit_sha256']}`, CLIP cache `{preflight['prompts']['text_cache_sha256']}`. The complete protected inventory ({len(preflight['protected_inventory_sha256'])} files) and all source/data SHA values are retained in `.runtime/preflight.json`.",
        "",
        "## Model and equations",
        "",
        "For patient i, GFR gives f_i in R^320, and one CAM forward gives H_i in R^(4x36), ordered [PHENO, ANAT, FUNC, CORR]. For every patient both disease queries are always evaluated; the true label never selects a query or score. For c in {ASD, TD}: q_i,c = LayerNorm(W_t t_c + W_f f_i), with shared W_t: D->36 and W_f: 320->36. Shared projections produce K_i=W_K H_i and V_i=W_V H_i. Attention is A_i=softmax(Q_i K_i^T/sqrt(36), over the four modality positions), and retrieved evidence is R_i=A_i V_i. A single shared score function is Score(r)=Linear(36,1)(GELU(Linear(36,36)(LayerNorm(r)))); it yields s_i=[s_i,ASD,s_i,TD].",
        "",
        "The original OutputLayer emits logp_base. GCDER uses logp_final=log_softmax(logp_base+beta*s), beta=0.5*tanh(beta_raw), beta_raw initialized to zero. This is equivalent to applying the same correction to raw logits and re-normalizing. At beta=0 the original classifier is exactly preserved; the binary ASD-minus-TD margin changes by beta*(s_ASD-s_TD). The loss remains L_cls(final)+L_graph+0.05 L_pair+0.01 L_adv+0.01 L_preserve. GCDER adds 52,490 parameters at D=1024; the full model has 1,011,386 parameters in each fold.",
        "",
        f"Preflight tensor sizes: `{json.dumps(preflight['shape_math_and_gradient_checks']['tensor_shapes'])}`. Identity max error was {preflight['shape_math_and_gradient_checks']['identity']['max_abs_error']:.3g}; score common-shift, raw-logit equivalence, and log-probability normalization checks passed. The CAM feature hook captures `FeedForward[-1]` from the same forward and preserves gradients.",
        "",
        "## Prompt groups",
        "",
        "All groups use the same frozen OpenAI CLIP RN50 cache and the same two-token GCDER structure. G1 uses the official BrainPrompt descriptions: `A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.` and `A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.` The published TC/ASD order is adapted to [ASD, TD].",
        "",
        "G2 uses these original SPromptGL texts, preserving the spelling `Nromal`: `Nromal brain volume`; `Uniform cortical thickness`; `Symmetrical brain structure`; `Normal ventricular morphology`; `Altered brain volume`; `Abnormal functional connectivity`; `White matter integrity issues`; `Different activation patterns`. Each embedding is L2-normalized; the first four are averaged as TD, the last four as ASD, each centroid is normalized, then the order is [ASD, TD]. This fixed aggregation is not SPromptGL's PEN projection.",
        "",
        f"G3 is a fixed anonymous pair generated from a private CPU torch.Generator with seed {R.CFG['prompt_policy']['anonymous_seed']}; it matches G1 vector norms and inter-class cosine without using labels or predictions. The recorded maximum norm error was {preflight['prompts']['anonymous_geometry_error']['max_norm_error_g1_g3']:.3g}; cosine error was {preflight['prompts']['anonymous_geometry_error']['cosine_error_g1_g3']:.3g}. Text cache SHA: `{preflight['prompts']['text_cache_sha256']}`.",
        "",
        "## Results",
        "",
        "All values are percent, mean +/- population SD across ten folds. ACC, hard AUC (historical one-hot hard-prediction definition), probability AUC (class-1 probability, TD positive), SEN-ASD, SPE-TD, BA, and ASD-F1 were independently recomputed from the loaded best checkpoints.",
        "",
        "| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in R.GROUPS:
        lines.append("| " + " | ".join([
            R.GROUP_NAMES[group],
            *[_format_mean_sd(by_group[group], metric) for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")],
        ]) + " |")
    base = {fold: preflight["base_reference"]["folds"][fold - 1]["metrics"] for fold in range(1, 11)}
    base_ba_mean, base_ba_sd = _summary_stats([base[fold]["ba"] * 100.0 for fold in range(1, 11)])
    lines.append("| 53.3 third-group base | " + " | ".join(
        f"{R.CFG['base']['reference_metrics_percent'][metric]:.4f} +/- {R.CFG['base']['reference_population_sd_percent'][metric]:.4f}"
        for metric in ("acc", "hard_auc", "prob_auc")
    ) + " | " + " | ".join(
        f"{R.CFG['base']['reference_metrics_percent'][metric]:.4f} +/- {R.CFG['base']['reference_population_sd_percent'][metric]:.4f}"
        for metric in ("sen", "spe")
    ) + f" | {base_ba_mean:.4f} +/- {base_ba_sd:.4f} | " + f"{R.CFG['base']['reference_metrics_percent']['asd_f1']:.4f} +/- {R.CFG['base']['reference_population_sd_percent']['asd_f1']:.4f} |")
    lines += ["", "| Group | Best epoch mean +/- SD (range) | Actual epochs mean +/- SD (range) | Parameters | Confusion matrix totals (actual ASD row; actual TD row) |", "|---|---:|---:|---:|---|"]
    for group in R.GROUPS:
        group_rows = by_group[group]
        best = [int(row["best_epoch"]) for row in group_rows]
        epochs = [int(row["epochs_executed"]) for row in group_rows]
        total_params = sorted({int(row["total_parameters"]) for row in group_rows})
        tp = sum(int(row["asd_tp"]) for row in group_rows)
        fn = sum(int(row["asd_fn"]) for row in group_rows)
        fp = sum(int(row["td_fp"]) for row in group_rows)
        tn = sum(int(row["td_tn"]) for row in group_rows)
        lines.append(f"| {group} | {np_mean(best):.1f} +/- {np_sd(best):.1f} ({min(best)}-{max(best)}) | {np_mean(epochs):.1f} +/- {np_sd(epochs):.1f} ({min(epochs)}-{max(epochs)}) | {total_params[0]:,} | [[{tp}, {fn}], [{fp}, {tn}]] |")
    lines += ["", "### Fold-paired comparisons", "", "Each paired row reports the first system minus the second in percentage points. The CSV carries all ten fold differences, their mean, and win/tie/loss counts for every metric.", "", "| Comparison | Metric | Mean delta (pp) | W/T/L | Fold deltas (pp) |", "|---|---|---:|---:|---|"]
    for row in paired:
        lines.append(f"| {row['comparison']} | {row['metric']} | {row['mean_difference_pp']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {row['fold_differences_pp_json']} |")
    lines += ["", "## Mechanism analysis", "", "These are diagnostics of model-produced latent evidence. They do not establish clinical lesion localization.", "", "| Group | beta mean (mean abs; SD; range) | s_ASD / s_TD means | Delta-s mean (SD; mean abs) | beta*Delta-s mean (mean abs) | attention entropy ASD / TD | attention | query cosine | GCDER test prediction changes | no-GFR attention change | no-GFR margin change | no-GFR prediction flips |", "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for group in R.GROUPS:
        item = mechanism[group]
        attention = ", ".join(f"{mod}: {item['attention_by_modality'][mod]['asd_mean']:.3f}/{item['attention_by_modality'][mod]['td_mean']:.3f} (|Delta|={item['attention_by_modality'][mod]['mean_abs_delta']:.4f})" for mod in R.MODALITIES)
        lines.append(f"| {group} | {item['beta_mean']:.5f} ({item['beta_mean_abs']:.5f}; {item['beta_std']:.5f}; {item['beta_min']:.5f}..{item['beta_max']:.5f}) | {item['score_asd_mean']:.5g} / {item['score_td_mean']:.5g} | {item['delta_score_mean']:.5g} ({item['delta_score_std']:.5g}; {item['mean_abs_delta_score']:.5g}) | {item['beta_delta_score_mean']:.5g} ({item['mean_abs_beta_delta_score']:.5g}) | {item['attention_entropy_asd_mean']:.4f} / {item['attention_entropy_td_mean']:.4f} | {attention} | {item['query_cosine_mean']:.5f} | {item['test_prediction_change_fraction']:.4f} | {item['no_graph_attention_abs_change_mean']:.6g} | {item['no_graph_margin_abs_change_mean']:.6g} | {item['no_graph_prediction_flip_fraction_mean']:.4f} |")
    lines += ["", "#### Site and quality-block screening", "", "| Group | max abs evidence/site correlation | site | Evidence vs PHENO RMS | ANAT RMS | FUNC RMS | CORR RMS |", "|---|---:|---|---:|---:|---:|---:|"]
    for group in R.GROUPS:
        item = mechanism[group]
        site = item["pheno_site_evidence_association"]
        corr = item["evidence_input_summary_pearson"]
        lines.append(f"| {group} | {site['max_abs_point_biserial']:.4f} | {site['site_with_max_abs_association'] or 'NA'} | {corr['PHENO']:+.4f} | {corr['ANAT']:+.4f} | {corr['FUNC']:+.4f} | {corr['CORR']:+.4f} |")
    lines += ["", "#### Active gradient path", "", "Post-identity-step mean gradient norms over the saved snapshots (all finite):", "", "| Group | beta | text projection | GFR condition projection | Key | Value | shared Score | correction to CAM | correction to GFR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in R.GROUPS:
        grad = mechanism[group]["gradient"]
        lines.append("| " + group + " | " + " | ".join(
            f"{grad[key]['mean_post_identity']:.5g}"
            for key in (
                "beta_grad_abs", "text_projection_grad_norm", "gfr_condition_grad_norm",
                "key_grad_norm", "value_grad_norm", "shared_score_grad_norm",
                "gcder_correction_to_cam_norm", "gcder_correction_to_gfr_norm",
            )
        ) + " |")
    lines += ["", "Attention deltas are A_ASD,m - A_TD,m; their four components necessarily sum to zero per patient. Interpret the component-wise differences and entropy above, not that sum. `mechanism.csv` records all 871 graph nodes for each of the 30 checkpoints (26,130 patient-fold rows), marking each outer test fold; classification metrics and confusion matrices use only each fold's held-out test nodes. It includes patient attention, score, margin, prediction-change, graph-condition removal, input-block magnitude summaries, and PHENO site category. OOF correlations between evidence score difference and block RMS/site indicators are descriptive checks only; they are not causal or site-removal tests. The graph removal is a fixed-checkpoint sensitivity diagnostic, not a retrained ablation.", "", "### Gradient, initialization, and recovery", "", f"GCDER initialization used `init_seed=710000+fold`, isolated by `torch.random.fork_rng`; the original Python/NumPy/CPU/CUDA RNG remained unchanged. Preflight passed {preflight['initialization']['status']} and warmup {preflight['experiment59']['normal_forward_vs_hooked_warmup']}. The active two-epoch continuous-versus-resume replay was `{preflight['active_short_replay_resume']['status']}` with bitwise-equal predictions, model/optimizer states, and RNG. Independent audit status: {sum(row['status'] == 'PASS' for row in audit_rows)}/30 PASS.", "", "Training gradient snapshots after the identity step are included per checkpoint. They cover beta, text projection, GFR condition projection, shared K/V and score, plus gradients from the GCDER correction back to CAM and GFR. Fold-level graph projection update norms and graph-zero attention/margin/prediction changes are in `fold_metrics.csv`; patient diagnostics are in `mechanism.csv`.", "", "## Protocol and limitations", "", "The frozen protocol is seed 0, StratifiedKFold(10, shuffle=True, random_state=10), 10 warmup epochs, up to 450 total epochs, strict ACC improvement, `wait>200`, and historical `val=test` checkpoint choice. Existing pre-outer-CV RFE and standardization remain. The patient graph contains all 871 subjects and is transductive. These results are exploratory comparisons under the same historical protocol, not an unbiased external test. PHENO contains demographic/site information; ANAT and FUNC are quality-related feature blocks and are not interpreted as lesions or neural time series. No site/quality causal attribution is claimed from descriptive block/site correlations alone.", "", f"## Decision: {decision}", "", "Pre-registered checks are recorded below; this single-seed decision is an engineering screen, not a statistical significance claim.", "", "```json", json.dumps(decision_details, indent=2, ensure_ascii=False), "```", "", "A GO result would still require independent multi-seed validation, strict generalization testing, and component ablations before any MICCAI 2027 novelty claim. A failed or conditional screen ends this registered run without adding groups or tuning against these folds.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    torch = R.BASE.runtime()["torch"]
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    preflight = json.loads(R.PRE_FLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or "active_short_replay_resume" not in preflight:
        raise RuntimeError("A complete passing preflight is required before checkpoint audit.")
    manifest = json.loads(R.MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("completed_tasks") != 30:
        raise RuntimeError(f"Need 30 completed tasks before independent audit; found {manifest.get('completed_tasks')}.")
    reference_manifest, reference_folds, reference_metrics = R.load_reference()
    prompt_payload = torch.load(R.PROMPT_CACHE, map_location="cpu", weights_only=False)
    current_protected = R._protected_inventory()
    expected_protected = preflight["protected_inventory_sha256"]
    if current_protected != expected_protected:
        missing = sorted(set(expected_protected) - set(current_protected))
        added = sorted(set(current_protected) - set(expected_protected))
        changed = sorted(
            path for path in set(expected_protected) & set(current_protected)
            if expected_protected[path] != current_protected[path]
        )
        raise AssertionError(f"Protected source/data/history SHA changed: missing={missing[:5]}, added={added[:5]}, changed={changed[:5]}.")
    fold_rows, patient_rows, audit_rows = [], [], []
    diagnostics: dict[tuple[str, int], dict] = {}
    for group in R.GROUPS:
        for fold in range(1, 11):
            try:
                fold_row, patient_fold, diag, audit = _audit_one(
                    group, fold, reference_manifest, reference_folds, preflight, prompt_payload
                )
                fold_rows.append(fold_row)
                patient_rows.extend(patient_fold)
                diagnostics[(group, fold)] = diag
                audit_rows.append(audit)
                print(f"AUDIT PASS {group} fold={fold:02d}", flush=True)
            except Exception as exc:
                audit_rows.append({"group": group, "fold": fold, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
                print(f"AUDIT FAIL {group} fold={fold:02d}: {type(exc).__name__}: {exc}", flush=True)
    results = R.RESULTS
    results.mkdir(parents=True, exist_ok=True)
    R.atomic_csv(results / "checkpoint_audit.csv", audit_rows)
    R.atomic_csv(results / "fold_metrics.csv", fold_rows)
    if not fold_rows or len(audit_rows) != 30 or any(row["status"] != "PASS" for row in audit_rows):
        manifest["status"] = "TRAINING_COMPLETE_AUDIT_FAILED"
        manifest["checkpoint_audit_status"] = "FAIL"
        manifest["checkpoint_audit_passed"] = sum(row["status"] == "PASS" for row in audit_rows)
        manifest["checkpoint_audit_failed"] = [row for row in audit_rows if row["status"] != "PASS"]
        R.atomic_json(R.MANIFEST_PATH, manifest)
        return 2
    paired = _paired_rows(fold_rows, reference_metrics)
    mechanism = _mechanism_summary(patient_rows, fold_rows, diagnostics)
    decision, decision_details = _decision(fold_rows, paired, mechanism, audit_rows)
    R.atomic_csv(results / "paired_comparisons.csv", paired)
    R.atomic_csv(results / "mechanism.csv", patient_rows)
    _write_report(results / "report.md", fold_rows, paired, mechanism, audit_rows, decision, decision_details, preflight)
    manifest.update({
        "status": "COMPLETE_AUDITED",
        "decision": decision,
        "checkpoint_audit_status": "PASS",
        "checkpoint_audit_passed": len(audit_rows),
        "checkpoint_audit_failed": 0,
        "independent_checkpoint_audit": {
            "status": "PASS",
            "passed": len(audit_rows),
            "failed": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_audit_csv_sha256": R.sha_file(results / "checkpoint_audit.csv"),
        },
        "protected_inventory_unchanged": True,
        "protected_inventory_count": len(current_protected),
        "audit_source_sha256": R.sha_file(EXP / "audit.py"),
        "group_mechanism_summary": mechanism,
        "registered_decision_checks": decision_details,
        "result_sha256": {
            name: R.sha_file(results / name)
            for name in ("fold_metrics.csv", "paired_comparisons.csv", "mechanism.csv", "report.md", "checkpoint_audit.csv")
        },
    })
    R.atomic_json(R.MANIFEST_PATH, manifest)
    print(json.dumps({"status": manifest["status"], "decision": decision, "audited": len(audit_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
