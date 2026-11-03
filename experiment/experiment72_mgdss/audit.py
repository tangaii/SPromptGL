#!/usr/bin/env python3
"""Independent checkpoint reader, metric verifier, and Experiment 72 report writer."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

EXP = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("e72_runner_for_audit", EXP / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot import the frozen Experiment 72 runner")
R = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = R
SPEC.loader.exec_module(R)

CFG = R.CFG
BASE = R.BASE
RUNTIME = R.RUNTIME
RESULTS = EXP / "results"
STATUS_PATH = RESULTS / "audit_status.json"
METRIC_NAMES = ("acc", "hard_auc", "prob_auc", "sen", "spe", "ba", "asd_f1")
PAIRS = (
    ("G1_vs_53.3", "G1", "53.3"),
    ("G2_vs_53.3", "G2", "53.3"),
    ("G3_vs_53.3", "G3", "53.3"),
    ("G4_vs_53.3", "G4", "53.3"),
    ("G5_vs_53.3", "G5", "53.3"),
    ("G6_vs_53.3", "G6", "53.3"),
    ("G1_vs_G3", "G1", "G3"),
    ("G2_vs_G4", "G2", "G4"),
    ("G1_vs_G2", "G1", "G2"),
    ("G3_vs_G4", "G3", "G4"),
    ("G5_vs_G6", "G5", "G6"),
    ("G1_vs_G5", "G1", "G5"),
    ("G2_vs_G6", "G2", "G6"),
)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def percent_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) * 100.0 for key in METRIC_NAMES}


def independent_metrics(logp, test_idx, labels) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    index = np.asarray(test_idx, dtype=np.int64)
    y_all = labels.detach().cpu().numpy() if hasattr(labels, "detach") else np.asarray(labels)
    y = y_all[index]
    probability = logp.detach().exp().cpu().numpy()[index]
    prediction = probability.argmax(axis=1)
    tp = int(np.sum((y == 0) & (prediction == 0)))
    fn = int(np.sum((y == 0) & (prediction == 1)))
    fp = int(np.sum((y == 1) & (prediction == 0)))
    tn = int(np.sum((y == 1) & (prediction == 1)))
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    hard_auc = float(roc_auc_score(np.eye(2, dtype=np.int64)[y], np.eye(2, dtype=np.int64)[prediction]))
    return {
        "acc": float(np.mean(prediction == y)),
        "hard_auc": hard_auc,
        "prob_auc": float(roc_auc_score(y, probability[:, 1])),
        "sen": sen,
        "spe": spe,
        "ba": (sen + spe) / 2.0,
        "asd_f1": float(f1_score(y, prediction, pos_label=0, zero_division=0)),
        "asd_tp": tp,
        "td_fp": fp,
        "asd_fn": fn,
        "td_tn": tn,
        "n_test": int(len(index)),
        "prediction": prediction,
        "probability": probability,
    }


def check_close_metrics(actual: dict[str, Any], expected: dict[str, Any], label: str, check_confusion: bool = True) -> None:
    for key in METRIC_NAMES:
        if abs(float(actual[key]) - float(expected[key])) > 1e-10:
            raise AssertionError(f"{label} metric mismatch for {key}: {actual[key]} != {expected[key]}")
    if check_confusion:
        for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
            if int(actual[key]) != int(expected[key]):
                raise AssertionError(f"{label} confusion mismatch for {key}: {actual[key]} != {expected[key]}")


def checkpoint_path(group: str, fold: int) -> Path:
    return RUNTIME / "checkpoints" / f"{group}_f{fold:02d}.best.pt"


def task_path(group: str, fold: int) -> Path:
    return RUNTIME / "tasks" / f"{group}_f{fold:02d}.json"


def mechanism_path(group: str, fold: int) -> Path:
    return RUNTIME / "mechanism" / f"{group}_f{fold:02d}.json"


def selected_reference_metrics() -> dict[int, dict[str, float]]:
    _manifest, rows = R.E59.read_reference()
    return {
        fold: {key: float(row[key]) for key in METRIC_NAMES}
        for fold, row in rows.items()
    }


def summarize_rows(metric_rows: list[dict[str, Any]], baseline: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    import numpy as np

    output = []
    by_group = {group: [row for row in metric_rows if row["group"] == group] for group in R.GROUP_IDS}
    values_by_group = {
        group: {
            metric: np.asarray([float(row[f"{metric}_pct"]) for row in rows], dtype=np.float64)
            for metric in METRIC_NAMES
        }
        for group, rows in by_group.items()
    }
    values_by_group["53.3"] = {
        metric: np.asarray([baseline[fold][metric] * 100.0 for fold in range(1, 11)], dtype=np.float64)
        for metric in METRIC_NAMES
    }
    for group in ("53.3",) + R.GROUP_IDS:
        row: dict[str, Any] = {"group": group, "fold_count": 10}
        for metric in METRIC_NAMES:
            values = values_by_group[group][metric]
            row[f"{metric}_mean_pct"] = float(values.mean())
            row[f"{metric}_population_sd_pct"] = float(values.std(ddof=0))
            row[f"{metric}_mean_plus_minus_sd_pct"] = f"{values.mean():.4f} +/- {values.std(ddof=0):.4f}"
        output.append(row)
    return output


def paired_comparisons(metric_rows: list[dict[str, Any]], baseline: dict[int, dict[str, float]]) -> list[dict[str, Any]]:
    import numpy as np

    group_rows = {group: {int(row["fold"]): row for row in metric_rows if row["group"] == group} for group in R.GROUP_IDS}
    base_rows = {fold: {metric: baseline[fold][metric] * 100.0 for metric in METRIC_NAMES} for fold in range(1, 11)}
    result = []
    for name, left_group, right_group in PAIRS:
        left = base_rows if left_group == "53.3" else {fold: {metric: float(row[f"{metric}_pct"]) for metric in METRIC_NAMES} for fold, row in group_rows[left_group].items()}
        right = base_rows if right_group == "53.3" else {fold: {metric: float(row[f"{metric}_pct"]) for metric in METRIC_NAMES} for fold, row in group_rows[right_group].items()}
        deltas = {metric: [left[fold][metric] - right[fold][metric] for fold in range(1, 11)] for metric in METRIC_NAMES}
        acc = np.asarray(deltas["acc"], dtype=np.float64)
        wins = int(np.sum(acc > 1e-12))
        losses = int(np.sum(acc < -1e-12))
        ties = 10 - wins - losses
        result.append({
            "comparison": name,
            "left": left_group,
            "right": right_group,
            "delta_acc_mean_pp": float(acc.mean()),
            "delta_acc_sd_population_pp": float(acc.std(ddof=0)),
            "acc_wins": wins,
            "acc_ties": ties,
            "acc_losses": losses,
            "delta_acc_pp_by_fold": json.dumps([round(float(value), 8) for value in acc], separators=(",", ":")),
            **{f"delta_{metric}_mean_pp": float(np.mean(values)) for metric, values in deltas.items()},
        })
    return result


def scipy_spearman(x, y) -> float | None:
    import numpy as np
    from scipy.stats import spearmanr

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if math.isfinite(value) else None


def feature_associations(margins_by_group: dict[str, dict[int, float]]) -> dict[str, Any]:
    import numpy as np

    c = BASE.runtime()
    _x, _y = BASE.load_data()[0]
    mapping = c["np"].load(BASE.ROOT / "data/ABIDE/modal_feat_dict.npy", allow_pickle=True).item()
    names = [str(value) for key in ("PHENO", "ANAT", "FUNC", "Correlation") for value in mapping[key]]
    feature_values = np.asarray(_x, dtype=np.float64)
    ranges: dict[str, list[int]] = {}
    start = 0
    for key in ("PHENO", "ANAT", "FUNC", "Correlation"):
        width = len(mapping[key])
        ranges[key] = list(range(start, start + width))
        start += width
    columns = {
        "site": [index for index, name in enumerate(names) if name.startswith("SITE_ID_")],
        "age_bins": [index for index, name in enumerate(names) if name.startswith("AGE_AT_SCAN_")],
        "sex": [index for index, name in enumerate(names) if name.startswith("SEX_")],
        "ANAT_quality": ranges["ANAT"],
        "FUNC_quality_motion": ranges["FUNC"],
    }
    output: dict[str, Any] = {}
    for group, margin_by_index in margins_by_group.items():
        indices = sorted(margin_by_index)
        margins = np.asarray([margin_by_index[index] for index in indices], dtype=np.float64)
        associations = {}
        for category, feature_indices in columns.items():
            rows = []
            for feature_index in feature_indices:
                values = feature_values[indices, feature_index]
                correlation = scipy_spearman(values, margins)
                if correlation is not None:
                    rows.append({"feature": names[feature_index], "spearman_r": correlation})
            associations[category] = {
                "feature_count": len(rows),
                "mean_absolute_spearman_r": None if not rows else float(np.mean([abs(row["spearman_r"]) for row in rows])),
                "max_absolute_spearman_r": None if not rows else max(rows, key=lambda row: abs(row["spearman_r"])),
            }
        output[group] = associations
    return output


def representation_summary(vectors, labels) -> dict[str, Any]:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular.square() if hasattr(singular, "square") else singular**2
    rank = float(energy.sum() ** 2 / max(float((energy**2).sum()), 1e-24))
    class_means = [matrix[y == cls].mean(axis=0) for cls in (0, 1)]
    within = []
    for cls in (0, 1):
        delta = matrix[y == cls] - class_means[cls]
        within.extend(np.sqrt(np.sum(delta * delta, axis=1)).tolist())
    between = float(np.linalg.norm(class_means[0] - class_means[1]))
    return {
        "patient_norm_mean": float(np.linalg.norm(matrix, axis=1).mean()),
        "patient_norm_sd": float(np.linalg.norm(matrix, axis=1).std(ddof=0)),
        "effective_rank_participation_ratio": rank,
        "within_class_centroid_distance_mean": float(np.mean(within)),
        "between_class_centroid_distance": between,
        "between_to_within_distance_ratio": between / max(float(np.mean(within)), 1e-12),
        "ASD_count": int(np.sum(y == 0)),
        "TD_count": int(np.sum(y == 1)),
    }


def similarity_means_by_true_class(similarities, labels) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(similarities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return {
        "ASD": {
            "count": int(np.sum(y == 0)),
            "mean_similarity_to_ASD_prototype": float(values[y == 0, 0].mean()),
            "mean_similarity_to_TD_prototype": float(values[y == 0, 1].mean()),
        },
        "TD": {
            "count": int(np.sum(y == 1)),
            "mean_similarity_to_ASD_prototype": float(values[y == 1, 0].mean()),
            "mean_similarity_to_TD_prototype": float(values[y == 1, 1].mean()),
        },
    }


def audit_task(group: str, fold: int, reference_manifest: dict, reference_folds: dict[int, dict], warmup_audit: dict, banks: dict, prompt_meta: dict, manifest: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = BASE.runtime()["torch"]
    task_file = task_path(group, fold)
    cp_file = checkpoint_path(group, fold)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    if task.get("status") != "COMPLETE" or task.get("test_labels_used_for_auxiliary_loss") is not False:
        raise AssertionError("task record is incomplete or indicates test-label auxiliary use")
    cp_sha = sha_file(cp_file)
    if task.get("checkpoint_sha256") != cp_sha:
        raise AssertionError("task/checkpoint SHA mismatch")
    cp = torch.load(cp_file, map_location="cpu", weights_only=False)
    if cp.get("schema") != "experiment72_mgdss_checkpoint_v1" or cp.get("group") != group or int(cp.get("fold", -1)) != fold:
        raise AssertionError("checkpoint group/fold/schema contamination")
    if cp.get("inference_uses_auxiliary") is not False or cp.get("test_labels_used_for_auxiliary_loss") is not False:
        raise AssertionError("checkpoint declares an impermissible auxiliary inference or test-label path")
    if cp.get("config_sha256") != manifest["config_sha256"] or cp.get("source_sha256") != manifest["source_sha256"]:
        raise AssertionError("checkpoint config/source SHA differs from the frozen manifest")
    expected_bank_sha = prompt_meta["banks"][R.group_item(group)["prompt"]]["bank_sha256"]
    if cp.get("prompt_bank_sha256") != expected_bank_sha:
        raise AssertionError("checkpoint contains a different prompt prototype bank")
    if cp.get("prompt_cache_sha256") != sha_file(R.PROMPT_CACHE) or cp.get("text_basis_sha256") != prompt_meta["basis_sha256"]:
        raise AssertionError("checkpoint frozen text cache/basis SHA mismatch")
    if cp.get("warmup_sha256") != sha_file(R.WARMUP_DIR / f"fold{fold:02d}.pt"):
        raise AssertionError("checkpoint warmup cache SHA mismatch")

    state = R.build_fold(fold, reference_manifest, reference_folds)
    ref = reference_folds[fold]
    for checkpoint_key, state_key in (
        ("initial_shared_sha256", "initial_shared_sha"),
        ("initial_gfr_sha256", "initial_gfr_sha"),
        ("initial_cam_sha256", "initial_cam_sha"),
        ("initial_added_sha256", "initial_added_sha"),
        ("initial_schema_sha256", "initial_schema_sha"),
        ("initial_rng_sha256", "construction_rng_sha"),
        ("split_sha256", "split_sha"),
        ("pair_sha256", "pair_sha"),
    ):
        state_value = state["cache_meta"]["pair_sha"] if state_key == "pair_sha" else state.get(state_key)
        if cp.get(checkpoint_key) != state_value:
            raise AssertionError(f"checkpoint fold provenance mismatch: {checkpoint_key}")
    if cp.get("reference_checkpoint_sha256") != ref["checkpoint_sha256"]:
        raise AssertionError("checkpoint references the wrong historical 53.3 checkpoint")
    if cp.get("relation_cache_json_sha256") != sha_file(BASE.REL_CACHE / f"fold{fold:02d}.json"):
        raise AssertionError("relation cache JSON SHA mismatch")
    if cp.get("relation_cache_npz_sha256") != sha_file(BASE.REL_CACHE / f"fold{fold:02d}.npz"):
        raise AssertionError("relation cache NPZ SHA mismatch")
    if cp.get("train_indices_sha256") != R.sha_obj(list(map(int, state["train"]))):
        raise AssertionError("checkpoint outer-training indices differ")
    if cp.get("test_indices_sha256") != R.sha_obj(list(map(int, state["test"]))):
        raise AssertionError("checkpoint outer-test indices differ")
    if int(cp.get("outer_train_count", -1)) != len(state["train"]) or int(cp.get("outer_test_count", -1)) != len(state["test"]):
        raise AssertionError("checkpoint train/test cardinalities differ")

    module, aux_optimizer, initial_aux_sha, rng_before, rng_after = R.private_auxiliary(int(prompt_meta["rank"]), fold)
    if initial_aux_sha != cp.get("initial_auxiliary_sha256") or rng_before != rng_after:
        raise AssertionError("checkpoint auxiliary projection initialization SHA/RNG mismatch")
    expected_alpha = manifest["preflight"]["alphas"][str(fold)][group]["alpha"]
    if abs(float(cp["alpha"]) - float(expected_alpha)) > 1e-15:
        raise AssertionError("checkpoint alpha differs from frozen preflight calibration")

    # Confirm the same shared epoch-10 checkpoint restores in this fresh audit process.
    warmup_info = R.load_warmup(state, fold, warmup_audit)
    if warmup_info["sha256"] != cp["warmup_sha256"]:
        raise AssertionError("warmup readback provenance mismatch")
    state["model"].load_state_dict(cp["base"], strict=True)
    state["cam"].load_state_dict(cp["cam"], strict=True)
    module.load_state_dict(cp["auxiliary"], strict=True)
    state["opt_base"].load_state_dict(cp["optimizer_base"])
    state["opt_cam"].load_state_dict(cp["optimizer_cam"])
    state["opt_added"].load_state_dict(cp["optimizer_added"])
    aux_optimizer.load_state_dict(cp["optimizer_auxiliary"])
    state["initial_auxiliary_sha256"] = initial_aux_sha
    state["warmup_cache_sha256"] = warmup_info["sha256"]

    group_definition = R.group_item(group)
    bank = banks[group_definition["prompt"]].to(BASE.runtime()["device"])
    with torch.no_grad():
        original_output = R.checked_forward(state, training=False, use_cadr=True)
        base_metrics = BASE.classification_metrics(original_output["logp"], state["test"], state["y"])
        reproduced = independent_metrics(original_output["logp"], state["test"], state["y"])
        check_close_metrics(base_metrics, reproduced, f"{group}/fold{fold:02d} independent metric")
        check_close_metrics(base_metrics, cp["metrics"], f"{group}/fold{fold:02d} checkpoint metric", check_confusion=False)
        check_close_metrics(base_metrics, task["metrics"], f"{group}/fold{fold:02d} task metric", check_confusion=False)
        for key in ("asd_tp", "td_fp", "asd_fn", "td_tn"):
            if int(reproduced[key]) != int(task["confusion"][key]):
                raise AssertionError(f"{group}/fold{fold:02d} task confusion mismatch for {key}")
        positions = state["cam"]._e59_positions
        aux_output = module(positions, bank, float(group_definition["temperature"]), use_fine=(group_definition["grain"] == "joint"))
    if int(state["cam"]._e72_forward_count) != 1:
        raise AssertionError("independent original-path inference did not execute CAM exactly once")

    index = torch.as_tensor(state["test"], dtype=torch.long, device=positions.device)
    labels = torch.as_tensor(state["y"], dtype=torch.long, device=positions.device)[index]
    tau = float(group_definition["temperature"])
    sim_global = aux_output["global_logits"][index].detach() * tau
    global_margin = sim_global.gather(1, labels[:, None]).squeeze(1) - sim_global.gather(1, (1 - labels)[:, None]).squeeze(1)
    module_test = {
        "global_correct": float((sim_global.argmax(-1) == labels).float().mean().cpu()),
        "global_margin_mean": float(global_margin.mean().cpu()),
        "global_margin_asd_mean": float(global_margin[labels == 0].mean().cpu()),
        "global_margin_td_mean": float(global_margin[labels == 1].mean().cpu()),
        "global_patient": aux_output["global_patient"][index].detach().cpu().numpy(),
        "global_similarity": sim_global.detach().cpu().numpy(),
        "test_indices": [int(value) for value in state["test"]],
        "labels": labels.detach().cpu().numpy(),
    }
    if aux_output["fine_logits"] is not None:
        sim_fine = aux_output["fine_logits"][index].detach() * tau
        module_test["fine_patient"] = aux_output["fine_patient"][index].detach().cpu().numpy()
        module_test["fine_similarity"] = sim_fine.detach().cpu().numpy()
        for modality_index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
            per = sim_fine[:, modality_index, :]
            margin = per.gather(1, labels[:, None]).squeeze(1) - per.gather(1, (1 - labels)[:, None]).squeeze(1)
            module_test[f"{modality}_correct"] = float((per.argmax(-1) == labels).float().mean().cpu())
            module_test[f"{modality}_margin_mean"] = float(margin.mean().cpu())
            module_test[f"{modality}_margin_asd_mean"] = float(margin[labels == 0].mean().cpu())
            module_test[f"{modality}_margin_td_mean"] = float(margin[labels == 1].mean().cpu())
    else:
        module_test["fine_patient"] = None
        module_test["fine_similarity"] = None

    mechanism_file = mechanism_path(group, fold)
    history = json.loads(mechanism_file.read_text(encoding="utf-8"))
    if not history or int(cp["best_epoch"]) < 10 or int(cp["epochs_executed"]) < 11:
        raise AssertionError("checkpoint epoch metadata or mechanism trace is incomplete")
    if not any(int(row["epoch"]) == int(cp["best_epoch"]) for row in history):
        raise AssertionError("best epoch does not exist in saved mechanism history")
    if task.get("mechanism_sha256") != sha_file(mechanism_file):
        raise AssertionError("mechanism trace SHA mismatch")
    best_mechanism = cp.get("best_diagnostics")
    if not isinstance(best_mechanism, dict) or best_mechanism.get("test_labels_passed") is not False:
        raise AssertionError("best-epoch mechanism record is missing or used test labels")

    metric_row = {
        "group": group,
        "fold": int(fold),
        "best_epoch": int(cp["best_epoch"]),
        "epochs_executed": int(cp["epochs_executed"]),
        "alpha": float(cp["alpha"]),
        "alpha_initial_gradient_ratio": float(manifest["preflight"]["alphas"][str(fold)][group]["initial_weighted_gradient_ratio"]),
        "train_count": int(cp["outer_train_count"]),
        "test_count": int(cp["outer_test_count"]),
        "TP_ASD": int(reproduced["asd_tp"]),
        "FP_TD": int(reproduced["td_fp"]),
        "FN_ASD": int(reproduced["asd_fn"]),
        "TN_TD": int(reproduced["td_tn"]),
        "checkpoint_sha256": cp_sha,
        "independent_checkpoint_audit": "PASS",
    }
    for key, value in percent_metrics(reproduced).items():
        metric_row[f"{key}_pct"] = value

    best_row = next(row for row in history if int(row["epoch"]) == int(cp["best_epoch"]))
    mechanism_row = {
        "group": group,
        "fold": int(fold),
        "best_epoch": int(cp["best_epoch"]),
        "epochs_executed": int(cp["epochs_executed"]),
        "alpha": float(cp["alpha"]),
        "initial_weighted_gradient_ratio": metric_row["alpha_initial_gradient_ratio"],
        "loss_global": best_row["loss_global"],
        "loss_fine": best_row["loss_fine"],
        "loss_semantic": best_row["loss_semantic"],
        "gradient_norm_cls_cam": best_row["gradient_norm_cls_cam"],
        "gradient_norm_semantic_cam": best_row["gradient_norm_semantic_cam"],
        "weighted_gradient_ratio": best_row["weighted_gradient_ratio"],
        "gradient_cosine_cam": best_row["gradient_cosine_cam"],
        "train_similarity_accuracy_global": best_row["train_similarity_accuracy_global"],
        "train_similarity_margin_global": best_row["train_similarity_margin_global"],
        "test_global_similarity_acc_pct": module_test["global_correct"] * 100.0,
        "test_global_margin_mean": module_test["global_margin_mean"],
        "test_global_margin_asd_mean": module_test["global_margin_asd_mean"],
        "test_global_margin_td_mean": module_test["global_margin_td_mean"],
    }
    for modality in ("PHENO", "ANAT", "FUNC", "CORR"):
        for suffix in ("gradient_cosine", "train_similarity_accuracy", "train_similarity_margin", "loss"):
            key = f"{suffix}_{modality}"
            if key in best_row:
                mechanism_row[key] = best_row[key]
        for suffix in ("correct", "margin_mean", "margin_asd_mean", "margin_td_mean"):
            key = f"{modality}_{suffix}"
            if key in module_test:
                mechanism_row[f"test_{modality}_{suffix}"] = module_test[key]

    audit_row = {
        "group": group,
        "fold": int(fold),
        "status": "PASS",
        "checkpoint_sha256": cp_sha,
        "independent_metric_recalculation": "PASS",
        "historical_initialization_parity": "PASS",
        "prompt_bank_sha256": expected_bank_sha,
        "warmup_sha256": cp["warmup_sha256"],
        "cam_forward_count": 1,
        "auxiliary_inference_used": False,
        "test_labels_used_for_auxiliary_loss": False,
    }
    del state, module, aux_optimizer, cp, positions, aux_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": metric_row, "mechanism": mechanism_row, "audit": audit_row}, module_test


def safe_mean(values) -> float | None:
    import numpy as np

    return None if not values else float(np.mean(values))


def make_report(manifest: dict, preflight: dict, prompt_meta: dict, summary_rows: list[dict], comparison_rows: list[dict], mechanism_rows: list[dict], group_analysis: dict, confounds: dict, audit_rows: list[dict], protected_after: dict[str, str]) -> str:
    by_summary = {row["group"]: row for row in summary_rows}
    by_pair = {row["comparison"]: row for row in comparison_rows}
    baseline_acc = float(by_summary["53.3"]["acc_mean_pct"])
    performance_candidates = []
    for group in R.GROUP_IDS:
        row = by_pair[f"{group}_vs_53.3"]
        group_summary = by_summary[group]
        if (
            row["delta_acc_mean_pp"] >= 0.5
            and row["acc_wins"] >= 6
            and row["delta_prob_auc_mean_pp"] >= -0.5
            and row["delta_sen_mean_pp"] >= -1.0
            and len(audit_rows) == 60
        ):
            performance_candidates.append(group)
    multi_rows = [by_pair["G1_vs_G5"], by_pair["G2_vs_G6"]]
    multi_grain = any(
        row["delta_acc_mean_pp"] > 0.0
        and row["acc_wins"] >= 6
        and all(abs(float(value["alpha_initial_gradient_ratio"]) - 0.1) <= 1e-6 for value in mechanism_rows)
        for row in multi_rows
    )
    if performance_candidates and multi_grain:
        verdict = "GO candidate"
    elif performance_candidates:
        verdict = "CONDITIONAL"
    else:
        verdict = "NO-GO"

    group_table = ["| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("53.3",) + R.GROUP_IDS:
        row = by_summary[group]
        group_table.append(
            f"| {group} | {row['acc_mean_plus_minus_sd_pct']} | {row['hard_auc_mean_plus_minus_sd_pct']} | "
            f"{row['prob_auc_mean_plus_minus_sd_pct']} | {row['sen_mean_plus_minus_sd_pct']} | {row['spe_mean_plus_minus_sd_pct']} | "
            f"{row['ba_mean_plus_minus_sd_pct']} | {row['asd_f1_mean_plus_minus_sd_pct']} |"
        )
    pair_lines = []
    for name, _, _ in PAIRS:
        row = by_pair[name]
        pair_lines.append(
            f"| {name} | {row['delta_acc_mean_pp']:+.4f} | {row['acc_wins']}/{row['acc_ties']}/{row['acc_losses']} | "
            f"`{row['delta_acc_pp_by_fold']}` |"
        )
    confusion_lines = []
    for group in R.GROUP_IDS:
        rows = [row for row in mechanism_rows if row["group"] == group]
        mean_alpha = safe_mean([float(row["alpha"]) for row in rows])
        mean_ratio = safe_mean([float(row["initial_weighted_gradient_ratio"]) for row in rows])
        mean_cos = safe_mean([float(row["gradient_cosine_cam"]) for row in rows])
        conflict = sum(float(row["gradient_cosine_cam"]) < 0.0 for row in rows)
        modality_performance = [
            f"global train acc/margin {safe_mean([float(row['train_similarity_accuracy_global']) for row in rows]) * 100:.1f}%/{safe_mean([float(row['train_similarity_margin_global']) for row in rows]):+.3f}"
        ]
        for modality in ("PHENO", "ANAT", "FUNC", "CORR"):
            accuracy = [float(row[f"train_similarity_accuracy_{modality}"]) for row in rows if row.get(f"train_similarity_accuracy_{modality}") is not None]
            margin = [float(row[f"train_similarity_margin_{modality}"]) for row in rows if row.get(f"train_similarity_margin_{modality}") is not None]
            loss = [float(row[f"loss_{modality}"]) for row in rows if row.get(f"loss_{modality}") is not None]
            if accuracy:
                modality_performance.append(f"{modality} {safe_mean(accuracy)*100:.1f}%/{safe_mean(margin):+.3f}/{safe_mean(loss):.3f}")
        oof = group_analysis.get(group, {})
        if "global_similarity_accuracy_pct" in oof:
            modality_performance.append(f"OOF global {oof['global_similarity_accuracy_pct']:.1f}%/{oof['global_correct_margin_mean']:+.3f}")
        if "oof_CORR_similarity_accuracy_pct" in oof:
            modality_performance.append(f"OOF CORR {oof['oof_CORR_similarity_accuracy_pct']:.1f}%/{oof['oof_CORR_correct_margin_mean']:+.3f}")
        confusion_lines.append(
            f"| {group} | {mean_alpha:.5f} | {mean_ratio:.4f} | {mean_cos:+.4f} | {conflict}/10 | {'; '.join(modality_performance) or 'global-only'} |"
        )
    representation_lines = []
    for group in R.GROUP_IDS:
        analysis = group_analysis[group]

        def pair_text(key: str, true_class: str) -> str:
            values = analysis[key][true_class]
            return (
                f"{values['mean_similarity_to_ASD_prototype']:+.3f}/"
                f"{values['mean_similarity_to_TD_prototype']:+.3f}"
            )

        representation = analysis["global_oof_representation"]
        representation_lines.append(
            f"| {group} | {pair_text('warmup_epoch10_global_similarity_mean_by_true_class', 'ASD')} | "
            f"{pair_text('warmup_epoch10_global_similarity_mean_by_true_class', 'TD')} | "
            f"{pair_text('best_oof_global_similarity_mean_by_true_class', 'ASD')} | "
            f"{pair_text('best_oof_global_similarity_mean_by_true_class', 'TD')} | "
            f"{analysis['global_correct_margin_asd_mean']:+.3f}/{analysis['global_correct_margin_td_mean']:+.3f} | "
            f"{representation['patient_norm_mean']:.3f}/{representation['effective_rank_participation_ratio']:.2f} | "
            f"{representation['between_to_within_distance_ratio']:.2f} |"
        )
    bp_cos = prompt_meta["banks"]["BrainPrompt-2"]["source_cosine"]
    sp_cos = prompt_meta["banks"]["SPromptGL-8"]["source_cosine"]
    equivalence = preflight["loss_equivalence"]
    status = manifest.get("status")
    protected_result = R.compare_protected(preflight["protected_source_sha256"], protected_after)
    confound_ranges = {}
    strongest_association = None
    for category in ("site", "age_bins", "sex", "ANAT_quality", "FUNC_quality_motion"):
        means = [float(group[category]["mean_absolute_spearman_r"]) for group in confounds.values() if group[category]["mean_absolute_spearman_r"] is not None]
        if means:
            confound_ranges[category] = [min(means), max(means)]
        for group_name, associations in confounds.items():
            top = associations[category]["max_absolute_spearman_r"]
            if top is not None and (strongest_association is None or abs(top["spearman_r"]) > abs(strongest_association["spearman_r"])):
                strongest_association = {"group": group_name, "category": category, **top}
    lines = [
        "# Experiment 72: MGDSS Results",
        "",
        f"- Audit status: `{status}`; independently audited checkpoints: {len(audit_rows)}/60.",
        f"- Frozen base: Experiment 53.3 / CADR_WD_0p01_GFR_WD_0p11; historical ACC {baseline_acc:.4f}%.",
        "- Cohort: ABIDE A, 871 participants (403 ASD=0, 468 TD=1), four input modalities with 320 features.",
        "- All six groups preserve the original CADR-GFR -> CAM -> 144-d OutputLayer path. The semantic heads are training-only.",
        "",
        "## Source Audit",
        "",
        "SPromptGL's audited path uses four modality feature blocks, GFR, CAM's four 36-dimensional positions, and the original OutputLayer. The eight prompts are in the source `network.py`; the original spelling `Nromal` is preserved. GFR, CAM/PLN and PEN are part of the frozen source audit; Experiment 72 changes none of them.",
        "BrainPrompt's official `compute_label_loss()` normalizes patient and disease vectors, forms cosine similarities, and applies cross entropy directly, equivalent to temperature 1.0. This run uses BrainPrompt's exact ABIDE text with frozen CLIP RN50 embeddings, so it tests the BrainPrompt text source and supervision style, not its Llama text encoder or complete architecture.",
        "SABER equations (8)-(10) describe semantic retrieval and gated residual injection; (11)-(13) define classification, disease-class contrastive supervision and the total objective. Only equation (12)'s class loss is adapted here. There is no claim of full SABER reproduction, and no residual injection or new graph/hypergraph path.",
        "Experiment 59 already tested BrainPrompt-style and SABER-style global disease supervision: both reached 91.8443% ACC, below 53.3's 92.0742%. G5/G6 are a controlled revalidation with RN50, a frozen shared text span, and frozen-gradient-budget calibration, not a previously untried supervision idea.",
        "Primary sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL official repository](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official repository](https://github.com/AngusMonroe/BrainPrompt), [graph network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptg_net.py), [label network](https://github.com/AngusMonroe/BrainPrompt/blob/main/nets/brainpromptc_net.py), and [SABER paper](https://arxiv.org/abs/2607.01901).",
        "",
        "## Frozen Design",
        "",
        "For patient projection `u = normalize(P(H))` and frozen class prototypes `t_ASD,t_TD`, similarities are `s_c = u dot t_c`. BrainPrompt-style loss is `CE(s, y)` at tau=1. SABER-style adaptation is `-log(exp(s_y/tau)/sum_c exp(s_c/tau))` at the frozen tau=0.1. The losses are mathematically identical at tau=1; this experiment compares the temperature, not different classes of supervision theory.",
        "The joint objective is `L_sem = 0.5 L_global + 0.5 L_fine`, where `L_fine = .10 L_PHENO + .10 L_ANAT + .10 L_FUNC + .70 L_CORR`. Global-only groups use `L_sem = L_global`; this is a fixed auxiliary-budget grain allocation comparison, not an unchanged global term plus an added fine term.",
        "Fine features are the single captured CAM tensor `[871,4,36]`, in `[PHENO, ANAT, FUNC, CORR]` order, after cross-modal interaction. Global features flatten that same tensor to 144 values. The shared fine projection is `Linear(36,r)`; the global projection is `Linear(144,r)`, with `r=10`. No CAM repeat forward is used.",
        f"Disease text uses the verified Experiment 60 frozen CLIP RN50 1024-d cache (`{prompt_meta['experiment60_cache_sha256']}`; RN50 weights SHA `{prompt_meta['rn50_weights_sha256']}`). BrainPrompt raw-text SHA: `{prompt_meta['brainprompt_text_sha256']}`; class-bank SHA: `{prompt_meta['banks']['BrainPrompt-2']['bank_sha256']}`. SPromptGL eight-text SHA: `{prompt_meta['spromptgl_text_sha256']}`; class-bank SHA: `{prompt_meta['banks']['SPromptGL-8']['bank_sha256']}`. CPU float64 SVD shared span rank: {prompt_meta['rank']}; basis SHA: `{prompt_meta['basis_sha256']}`.",
        f"Prototype cosines are BrainPrompt-2 {bp_cos:.7f} and SPromptGL-8 {sp_cos:.7f}; projected geometry error is within {CFG['text']['geometry_tolerance']}. The class-logit gap is bounded by the prototype chord length. The possible probability range for either class under BrainPrompt-2 is {1-preflight['softmax_confidence_bounds']['BrainPrompt-2']['tau1_maximum_correct_softmax_probability']:.4f}-{preflight['softmax_confidence_bounds']['BrainPrompt-2']['tau1_maximum_correct_softmax_probability']:.4f} at tau=1 and {1/(1+math.exp(preflight['softmax_confidence_bounds']['BrainPrompt-2']['tau1_maximum_logit_gap']/.1)):.4f}-{1/(1+math.exp(-preflight['softmax_confidence_bounds']['BrainPrompt-2']['tau1_maximum_logit_gap']/.1)):.4f} at tau=0.1. For SPromptGL-8 these ranges are {1-preflight['softmax_confidence_bounds']['SPromptGL-8']['tau1_maximum_correct_softmax_probability']:.4f}-{preflight['softmax_confidence_bounds']['SPromptGL-8']['tau1_maximum_correct_softmax_probability']:.4f} and {1/(1+math.exp(preflight['softmax_confidence_bounds']['SPromptGL-8']['tau1_maximum_logit_gap']/.1)):.4f}-{1/(1+math.exp(-preflight['softmax_confidence_bounds']['SPromptGL-8']['tau1_maximum_logit_gap']/.1)):.4f}, respectively. These are geometric bounds, not observed calibration intervals; tau was frozen before training.",
        f"At tau=1, CE and SABER-form log-sum-exp loss differed by {equivalence['loss_abs_error']:.3g}, with maximum gradient difference {equivalence['max_gradient_abs_error']:.3g}. The six groups, prompt source, temperature and grain are fixed in `config.json`.",
        "",
        "## Preflight and Training",
        "",
        f"All P1-P20 checks: `{json.dumps(preflight['check_status'], ensure_ascii=False)}`. 53.3 initialization parity: {sum(value['status']=='PASS' for value in preflight['fold_initialization_and_warmup'].values())}/10 folds; E59 shared warmup parity: 10/10 folds. Fold1 two-step repeat/resume: `{preflight['recovery_smoke']['status']}`.",
        f"The initial unscaled semantic CAM-gradient norm was calibrated with outer-training labels only using `alpha = .10 * g_cls / g_sem`; alpha was frozen per fold/group, finite and <=10. The observed initial weighted ratios span {min(float(preflight['alphas'][str(f)][g]['initial_weighted_gradient_ratio']) for f in range(1,11) for g in R.GROUP_IDS):.8f} to {max(float(preflight['alphas'][str(f)][g]['initial_weighted_gradient_ratio']) for f in range(1,11) for g in R.GROUP_IDS):.8f}. Later-epoch ratios are diagnostic only. The fitted gradient cosine can reveal conflict; see `mechanism.csv`.",
        f"Formal task count: {manifest.get('completed_tasks', 0)}/60 complete. Failed tasks: {sum(value.get('status') == 'FAILED' for value in manifest.get('tasks', {}).values())}.",
        "",
        "## Results",
        "",
        "All entries below are 10-fold mean +/- population SD in percent. Fold-level rows, including TP/FP/FN/TN, best epoch and executed epochs, are in `fold_metrics.csv`.",
        *group_table,
        "",
        "Paired ACC differences use left minus right in percentage points. W/T/L counts strict fold wins, ties and losses. Every fold delta is listed for reproducibility.",
        "",
        "| Comparison | Mean delta ACC (pp) | W/T/L | Per-fold delta ACC (pp) |",
        "|---|---:|---:|---|",
        *pair_lines,
        "",
        "## Mechanism",
        "",
        "`mechanism.csv` records each group's per-fold losses, CAM gradient norms and cosines, per-position training disease-similarity accuracy/margin, and OOF held-out similarity diagnostics. A negative CAM gradient cosine means the unweighted semantic gradient points against the original classification gradient locally; frequency and per-position values are retained per fold.",
        "",
        "| Group | Mean alpha | Initial weighted ratio | Mean CAM grad cosine | Folds with negative cosine | Mean train position result |",
        "|---|---:|---:|---:|---:|---|",
        *confusion_lines,
        "",
        "Training mechanism numbers are in `mechanism.csv`: each fold includes global and per-position training similarity accuracy, correct-class margin and loss, plus per-position and total CAM-gradient cosine. Test-set patient embeddings/margins are out-of-fold diagnostic measurements only and were not used for alpha, modality weights, temperature, or checkpoint selection.",
        "The following cosine pairs are `[similarity to ASD prototype / similarity to TD prototype]`; epoch 10 is the shared warmup state and best OOF is the independently read best checkpoint. Margins are correct-class minus other-class, split by true class. Patient projection norms remain near one by construction.",
        "| Group | E10 ASD | E10 TD | Best OOF ASD | Best OOF TD | Best margin ASD/TD | Norm/effective rank | Between/within |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *representation_lines,
        f"Across these OOF folds, mean absolute Spearman correlations of global correct margin span {', '.join(f'{key}={value[0]:.3f}-{value[1]:.3f}' for key, value in confound_ranges.items())}; the largest feature association is {strongest_association['group']} {strongest_association['feature']} ({strongest_association['spearman_r']:+.3f}). These are descriptive checks only.",
        "PHENO columns encode age bins, site and sex; FUNC includes mean FD, number/percentage of FD outliers and other quality features. These associations describe possible site/quality dependence, not anatomical lesions or causal confounding. There is no anonymous-prompt control, so this experiment cannot establish that medical text semantics are indispensable; the auxiliary labels may act as ordinary additional supervision.",
        "",
        "## Decision and Limitations",
        "",
        f"Pre-registered verdict: **{verdict}**. Performance-gate candidates: {', '.join(performance_candidates) if performance_candidates else 'none'}. Multi-grain gate: {'PASS' if multi_grain else 'FAIL'} (G1 vs G5: {by_pair['G1_vs_G5']['delta_acc_mean_pp']:+.4f} pp, {by_pair['G1_vs_G5']['acc_wins']}/10 wins; G2 vs G6: {by_pair['G2_vs_G6']['delta_acc_mean_pp']:+.4f} pp, {by_pair['G2_vs_G6']['acc_wins']}/10 wins).",
        "Even a GO candidate supports only follow-up with multiple seeds, a strict independent validation split, and component ablations; it does not establish a MICCAI-level innovation. G3/G4 have no SPromptGL global-only counterparts, so their performance cannot identify a causal SPromptGL fine-grain contribution.",
        "The original protocol retains historical val=test checkpoint selection, preprocessing before outer CV, and the transductive 871-node patient graph. Consequently these results are exploratory internal CV measurements, not an unbiased external generalization estimate.",
        f"Independent audit: {len(audit_rows)}/60 checkpoints passed. Protected root code, data, and historical experiment SHA comparison: `{protected_result['status']}` ({protected_result['before_count']} files). Historical Experiment 53.3, E59, E60, E70, E70.1 and E71 sources were read-only.",
        f"The independent reader's post-training schema adapters were corrected after its first pass exposed a pair-SHA field-path mismatch and a separate task confusion-count schema. Training code, configuration and checkpoint bytes retain their frozen hashes; both audit-script hashes and the first failed adapter attempt are recorded in `audit_status.json`.",
        "",
        "## Artifacts",
        "",
        "- `config.json`: frozen six-group protocol.",
        "- `manifest.json`: immutable source/config hashes and per-task status.",
        "- `results/fold_metrics.csv`: all ten fold metrics for each group.",
        "- `results/summary.csv`: mean and population SD, including 53.3.",
        "- `results/paired_comparisons.csv`: all registered paired comparisons.",
        "- `results/mechanism.csv`: per-fold loss, gradient, modality and OOF similarity analysis.",
        "- `results/audit_status.json`: independent checkpoint/source audit and representation/confound summaries.",
    ]
    return "\n".join(lines) + "\n"


def run_audit() -> dict[str, Any]:
    import numpy as np
    import torch

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(R.MANIFEST_PATH.read_text(encoding="utf-8"))
    previous_audit = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.is_file() else None
    attempt_history = list(previous_audit.get("audit_attempt_history", [])) if previous_audit else []
    if previous_audit and previous_audit.get("status") == "FAIL":
        prior_failures = previous_audit.get("failed_tasks", [])
        attempt_history.append({
            "status": "FAIL",
            "independent_checkpoint_audits_passed": previous_audit.get("independent_checkpoint_audits_passed", 0),
            "independent_checkpoint_audits_total": previous_audit.get("independent_checkpoint_audits_total", 60),
            "common_failure": prior_failures[0].get("error") if prior_failures else "unspecified audit failure",
            "audit_script_sha256": previous_audit.get("source_sha256_at_audit", {}).get("audit.py"),
            "interpretation": "Audit-adapter failure; no checkpoint was accepted or modified during the failed pass.",
        })
    elif previous_audit and previous_audit.get("status") == "PASS":
        attempt_history.append({
            "status": "PASS",
            "independent_checkpoint_audits_passed": previous_audit.get("independent_checkpoint_audits_passed", 0),
            "independent_checkpoint_audits_total": previous_audit.get("independent_checkpoint_audits_total", 60),
            "audit_script_sha256": previous_audit.get("corrected_independent_audit_script_sha256"),
            "interpretation": "All checkpoint reads and metric recalculations passed; rerun regenerates complete mechanism summaries.",
        })
    if manifest.get("status") not in {"TRAINING_COMPLETE_PENDING_AUDIT", "AUDIT_FAILED", "COMPLETE_AUDITED"} or int(manifest.get("completed_tasks", 0)) != 60:
        status = {"status": "TRAINING_INCOMPLETE", "completed": int(manifest.get("completed_tasks", 0)), "expected": 60}
        atomic_json(STATUS_PATH, status)
        return status
    preflight = json.loads(R.PRE_FLIGHT.read_text(encoding="utf-8"))
    current_source = R.code_hashes()
    frozen_source = manifest["source_sha256"]
    if preflight.get("status") != "PASS" or any(frozen_source[key] != current_source[key] for key in ("config.json", "run.py", "mgdss.py")):
        raise AssertionError("frozen training code/config SHA differs at independent audit time")
    if manifest.get("config_sha256") != sha_file(R.CFG_PATH):
        raise AssertionError("frozen config SHA changed after training")
    reference_manifest, reference_folds = R.load_reference()
    warmup_audit = R.verify_warmup_audit()
    banks, prompt_meta = R.load_prompt_cache()
    _fresh_banks, fresh_meta = R.make_prompt_assets(write_cache=False)
    if prompt_meta != fresh_meta or any(not torch.equal(banks[name], _fresh_banks[name]) for name in banks):
        raise AssertionError("frozen prompt bank failed deterministic independent reconstruction")
    expected_protected = preflight["protected_source_sha256"]
    protected_before = R.protected_snapshot()
    if protected_before != expected_protected:
        raise AssertionError("protected source/data/history changed after preflight")

    metric_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    representations: dict[str, dict[int, dict[str, Any]]] = {group: {} for group in R.GROUP_IDS}
    failures: list[dict[str, Any]] = []
    for group in R.GROUP_IDS:
        for fold in range(1, 11):
            key = f"{group}_f{fold:02d}"
            try:
                result, heldout = audit_task(group, fold, reference_manifest, reference_folds, warmup_audit, banks, prompt_meta, manifest)
                metric_rows.append(result["metrics"])
                mechanism_rows.append(result["mechanism"])
                audit_rows.append(result["audit"])
                representations[group][fold] = heldout
                manifest["tasks"][key]["independent_checkpoint_audit"] = "PASS"
                manifest["tasks"][key]["independent_checkpoint_audit_sha256"] = result["audit"]["checkpoint_sha256"]
                manifest["tasks"][key].pop("independent_checkpoint_audit_error", None)
                print(f"AUDIT PASS {key}", flush=True)
            except BaseException as error:
                failures.append({"task": key, "error": f"{type(error).__name__}: {error}"})
                manifest["tasks"].setdefault(key, {})["independent_checkpoint_audit"] = "FAIL"
                manifest["tasks"][key]["independent_checkpoint_audit_error"] = failures[-1]["error"]
                print(f"AUDIT FAIL {key}: {failures[-1]['error']}", flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    baseline = selected_reference_metrics()
    summaries = summarize_rows(metric_rows, baseline) if len(metric_rows) == 60 else []
    comparisons = paired_comparisons(metric_rows, baseline) if len(metric_rows) == 60 else []
    group_analysis: dict[str, Any] = {}
    margins_by_group: dict[str, dict[int, float]] = {}
    for group in R.GROUP_IDS:
        if len(representations[group]) != 10:
            continue
        group_values = representations[group]
        patient_rows: dict[int, dict[str, Any]] = {}
        for fold_record in group_values.values():
            indices = fold_record["test_indices"]
            for local, patient_index in enumerate(indices):
                patient_rows[patient_index] = {
                    "label": int(fold_record["labels"][local]),
                    "global_patient": fold_record["global_patient"][local],
                    "global_similarity": fold_record["global_similarity"][local],
                    "fine_patient": None if fold_record["fine_patient"] is None else fold_record["fine_patient"][local],
                    "fine_similarity": None if fold_record["fine_similarity"] is None else fold_record["fine_similarity"][local],
                }
        if len(patient_rows) != 871:
            failures.append({"task": f"{group}_OOF", "error": f"expected 871 unique held-out rows, got {len(patient_rows)}"})
            continue
        indices = sorted(patient_rows)
        labels = np.asarray([patient_rows[index]["label"] for index in indices], dtype=np.int64)
        global_vectors = np.asarray([patient_rows[index]["global_patient"] for index in indices], dtype=np.float64)
        global_similarity = np.asarray([patient_rows[index]["global_similarity"] for index in indices], dtype=np.float64)
        global_margin = global_similarity[np.arange(len(indices)), labels] - global_similarity[np.arange(len(indices)), 1 - labels]
        margins_by_group[group] = {index: float(value) for index, value in zip(indices, global_margin)}
        item = R.group_item(group)
        analysis = {
            "oof_test_rows": len(indices),
            "global_similarity_accuracy_pct": float(np.mean(global_similarity.argmax(axis=1) == labels) * 100.0),
            "global_correct_margin_mean": float(global_margin.mean()),
            "global_correct_margin_median": float(np.median(global_margin)),
            "global_correct_margin_asd_mean": float(global_margin[labels == 0].mean()),
            "global_correct_margin_td_mean": float(global_margin[labels == 1].mean()),
            "global_oof_representation": representation_summary(global_vectors, labels),
            "warmup_epoch10_test_representations": {},
        }
        warmup_by_patient = {}
        for fold in range(1, 11):
            record = preflight["fold_initialization_and_warmup"][str(fold)]["warmup_test_representations"]
            for local, patient_index in enumerate(record["test_indices"]):
                warmup_by_patient[int(patient_index)] = {
                    "global_patient": record["global_patient"][local],
                    "fine_patient": record["fine_patient"][local],
                    "similarities": record["similarities"][item["prompt"]]["global"][local],
                "fine_similarities": record["similarities"][item["prompt"]]["fine"][local],
                }
        warm_vectors = np.asarray([warmup_by_patient[index]["global_patient"] for index in indices], dtype=np.float64)
        warm_sims = np.asarray([warmup_by_patient[index]["similarities"] for index in indices], dtype=np.float64)
        analysis["warmup_epoch10_global_similarity_mean_by_class_slot"] = warm_sims.mean(axis=0).tolist()
        analysis["warmup_epoch10_global_similarity_mean_by_true_class"] = similarity_means_by_true_class(warm_sims, labels)
        analysis["best_oof_global_similarity_mean_by_true_class"] = similarity_means_by_true_class(global_similarity, labels)
        analysis["warmup_epoch10_test_representations"] = representation_summary(warm_vectors, labels)
        analysis["warmup_to_best_effective_rank_change"] = (
            analysis["global_oof_representation"]["effective_rank_participation_ratio"]
            - analysis["warmup_epoch10_test_representations"]["effective_rank_participation_ratio"]
        )
        if item["grain"] == "joint":
            fine_vectors = np.asarray([patient_rows[index]["fine_patient"] for index in indices], dtype=np.float64)
            fine_sim = np.asarray([patient_rows[index]["fine_similarity"] for index in indices], dtype=np.float64)
            warm_fine_sim = np.asarray([warmup_by_patient[index]["fine_similarities"] for index in indices], dtype=np.float64)
            for modality_index, modality in enumerate(("PHENO", "ANAT", "FUNC", "CORR")):
                modality_sim = fine_sim[:, modality_index, :]
                analysis[f"warmup_epoch10_{modality}_similarity_mean_by_true_class"] = similarity_means_by_true_class(
                    warm_fine_sim[:, modality_index, :], labels
                )
                analysis[f"best_oof_{modality}_similarity_mean_by_true_class"] = similarity_means_by_true_class(modality_sim, labels)
                modality_margin = modality_sim[np.arange(len(indices)), labels] - modality_sim[np.arange(len(indices)), 1 - labels]
                analysis[f"oof_{modality}_similarity_accuracy_pct"] = float(np.mean(modality_sim.argmax(axis=1) == labels) * 100.0)
                analysis[f"oof_{modality}_correct_margin_mean"] = float(modality_margin.mean())
                analysis[f"oof_{modality}_correct_margin_asd_mean"] = float(modality_margin[labels == 0].mean())
                analysis[f"oof_{modality}_correct_margin_td_mean"] = float(modality_margin[labels == 1].mean())
            analysis["oof_fine_representation"] = representation_summary(fine_vectors.mean(axis=1), labels)
        group_analysis[group] = analysis

    confounds = feature_associations(margins_by_group) if len(margins_by_group) == len(R.GROUP_IDS) else {}
    protected_after = R.protected_snapshot()
    protected_audit = R.compare_protected(expected_protected, protected_after)
    if protected_audit["status"] != "PASS":
        failures.append({"task": "protected_sources", "error": json.dumps(protected_audit, ensure_ascii=False)})

    if len(metric_rows) == 60:
        atomic_csv(RESULTS / "fold_metrics.csv", metric_rows)
        atomic_csv(RESULTS / "summary.csv", summaries)
        atomic_csv(RESULTS / "paired_comparisons.csv", comparisons)
        atomic_csv(RESULTS / "mechanism.csv", mechanism_rows)
    else:
        atomic_csv(RESULTS / "fold_metrics.csv", metric_rows)
        atomic_csv(RESULTS / "mechanism.csv", mechanism_rows)

    status = {
        "status": "PASS" if len(audit_rows) == 60 and not failures and protected_audit["status"] == "PASS" else "FAIL",
        "independent_checkpoint_audits_passed": len(audit_rows),
        "independent_checkpoint_audits_total": 60,
        "failed_tasks": failures,
        "audit_attempt_history": attempt_history,
        "fold_audit_rows": audit_rows,
        "protected_source_audit": protected_audit,
        "source_sha256_at_audit": current_source,
        "frozen_source_sha256": frozen_source,
        "preflight_audit_script_sha256": frozen_source["audit.py"],
        "corrected_independent_audit_script_sha256": current_source["audit.py"],
        "posttraining_audit_script_correction": None if frozen_source["audit.py"] == current_source["audit.py"] else "Corrected pair_sha field adapter and separated task confusion-count schema in independent checkpoint reader; training code, frozen config and all checkpoint bytes are unchanged.",
        "config_sha256_at_audit": sha_file(R.CFG_PATH),
        "prompt_reconstruction_sha256": sha_file(R.PROMPT_CACHE),
        "prompt_geometry": prompt_meta["banks"],
        "group_representation_analysis": group_analysis,
        "site_age_sex_quality_margin_associations": confounds,
        "train_only_semantic_labels": True,
        "test_labels_used_for_auxiliary_loss": False,
        "all_test_diagnostics_posthoc_only": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(STATUS_PATH, status)
    manifest["independent_checkpoint_audits_passed"] = len(audit_rows)
    manifest["independent_checkpoint_audit_failures"] = failures
    manifest["protected_source_audit_after_training"] = protected_audit
    if status["status"] == "PASS":
        manifest["status"] = "COMPLETE_AUDITED"
        manifest["completed_utc"] = status["created_utc"]
    else:
        manifest["status"] = "AUDIT_FAILED"
    R.atomic_json(R.MANIFEST_PATH, manifest)
    if len(metric_rows) == 60:
        report = make_report(manifest, preflight, prompt_meta, summaries, comparisons, mechanism_rows, group_analysis, confounds, audit_rows, protected_after)
        (RESULTS / "report.md").write_text(report, encoding="utf-8")
    return status


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({key: result[key] for key in ("status", "independent_checkpoint_audits_passed", "independent_checkpoint_audits_total") if key in result}, ensure_ascii=False, indent=2))
