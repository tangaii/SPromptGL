"""Standalone checkpoint, result, and recovery utilities for DPSL.

The historical model implementation is kept under ``code/provenance`` and
``code/history_*``.  This module is the only code imported by the public CLI.
It reads the frozen local artifacts and never resolves an adjacent experiment
directory at runtime.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "runtime" / "python"
if PROFILE.is_dir() and str(PROFILE) not in sys.path:
    sys.path.insert(0, str(PROFILE))

import numpy as np
import torch


FOLDS = tuple(range(1, 11))
OVERLAP_1BASED = {
    2: (377, 407, 681, 779, 784, 260, 288, 375),
    3: (46, 66, 172, 266, 388),
    4: (712, 713, 727, 773),
    8: (383, 430),
}
EXPECTED_METRICS_PERCENT = {
    "acc": (93.1087, 2.6257),
    "hard_auc": (93.0348, 2.6450),
    "prob_auc": (94.8628, 2.8903),
    "sen": (92.0549, 3.6745),
    "spe": (94.0148, 3.4314),
    "asd_f1": (92.5096, 2.8797),
}
REQUIRED_CHECKPOINT_KEYS = {
    "group", "fold", "seed", "config", "implementation", "provenance", "membership",
    "state", "state_sha", "ema", "ema_sha", "rng", "rng_sha", "epoch", "best_epoch",
    "best_correct", "wait", "evaluation", "observation_sha", "journal_sha",
}
REQUIRED_MEMBERSHIP_KEYS = {"train", "legal", "test", "train_pairs", "train_same", "weights"}
REQUIRED_RNG_KEYS = {"python", "numpy", "cpu", "cuda"}
REQUIRED_EMA_KEYS = {"mean", "second", "stable", "counts", "updates"}
REQUIRED_NATIVE_KEYS = {"model", "cam", "opt_base", "opt_cam", "opt_added"}
REQUIRED_STATE_KEYS = {"native", "module", "optimizer"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def checkpoint_path(fold: int, kind: str) -> Path:
    if fold not in FOLDS or kind not in {"best", "latest"}:
        raise ValueError(f"unsupported checkpoint selector fold={fold!r} kind={kind!r}")
    return ROOT / "checkpoints" / f"fold{fold:02d}.{kind}.pt"


def load_checkpoint(fold: int, kind: str) -> dict[str, Any]:
    return torch.load(checkpoint_path(fold, kind), map_location="cpu", weights_only=False)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def formal_patient_rows() -> list[dict[str, str]]:
    _, rows = load_csv(ROOT / "results" / "patient_level.csv")
    return rows


def formal_metric_rows() -> list[dict[str, str]]:
    _, rows = load_csv(ROOT / "results" / "fold_metrics.csv")
    return rows


def row_index(rows: Iterable[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    return {(int(row["fold"]), int(row["patient_index_0based"])): row for row in rows}


def source_records() -> dict[str, Any]:
    return read_json(ROOT / "runtime" / "copy_records.json")


def history_records() -> list[dict[str, Any]]:
    return read_json(ROOT / "runtime" / "history_records.json")


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def checkpoint_summary(fold: int) -> dict[str, Any]:
    best = load_checkpoint(fold, "best")
    latest = load_checkpoint(fold, "latest")
    recorded_overlap = best.get("train_test_overlap_1based")
    if recorded_overlap is None:
        train = set((_array(best["membership"]["train"]).astype(np.int64) + 1).tolist())
        test = set((_array(best["membership"]["test"]).astype(np.int64) + 1).tolist())
        recorded_overlap = sorted(train & test)
    return {
        "fold": fold,
        "internal_group_best": best.get("group"),
        "internal_group_latest": latest.get("group"),
        "best_epoch": int(best["epoch"]),
        "best_selection_epoch": int(best["best_epoch"]),
        "latest_epoch": int(latest["epoch"]),
        "latest_best_epoch": int(latest["best_epoch"]),
        "best_correct": int(best["best_correct"]),
        "latest_wait": int(latest["wait"]),
        "best_state_sha": best.get("state_sha"),
        "latest_state_sha": latest.get("state_sha"),
        "best_sha256": sha256_file(checkpoint_path(fold, "best")),
        "latest_sha256": sha256_file(checkpoint_path(fold, "latest")),
        "test_rows": int(len(best["membership"]["test"])),
        "train_rows": int(len(best["membership"]["train"])),
        "legal_rows": int(len(best["membership"]["legal"])),
        "overlap_1based": list(recorded_overlap),
        "model_variant": best.get("model_variant", "G2"),
        "task_group": best.get("task_group", "G2"),
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 0
    negative = labels == 1
    n_positive, n_negative = int(positive.sum()), int(negative.sum())
    if not n_positive or not n_negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _binary_metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    tp = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 1) & (prediction == 0)).sum())
    tn = int(((labels == 1) & (prediction == 1)).sum())
    fn = int(((labels == 0) & (prediction == 1)).sum())
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "n_test": int(len(labels)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "correct": int((labels == prediction).sum()),
        "acc": float((labels == prediction).mean()),
        "hard_auc": float((sen + spe) / 2),
        "prob_auc": _auc(labels, probability[:, 0]),
        "sen": float(sen), "spe": float(spe), "asd_f1": float(f1),
    }


def replay_best_evaluation(write: bool = True) -> dict[str, Any]:
    """Replay the frozen checkpoint evaluation payload in test-member order.

    The checkpoint stores the exact historical test forward output.  The replay
    loads that payload and joins it to the frozen patient table without running
    a new optimizer step or consuming a training RNG stream.
    """
    patient = formal_patient_rows()
    indexed = row_index(patient)
    output_rows: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    for fold in FOLDS:
        checkpoint = load_checkpoint(fold, "best")
        membership = checkpoint["membership"]
        test = _array(membership["test"]).astype(np.int64, copy=False)
        evaluation = checkpoint["evaluation"]
        probabilities = _array(evaluation["probability"])
        logits = _array(evaluation["logits"])
        prediction = _array(evaluation["prediction"]).astype(np.int64, copy=False)
        labels = []
        reference_probability = []
        reference_prediction = []
        order_exact = True
        for position, patient_index in enumerate(test):
            row = indexed[(fold, int(patient_index))]
            labels.append(int(row["true_label"]))
            reference_probability.append(float(row["p_asd_full"]))
            reference_prediction.append(int(row["prediction_full"]))
            order_exact &= int(row["fold"]) == fold
            output_rows.append({
                "fold": fold,
                "test_position": position,
                "patient_id_1based": int(patient_index) + 1,
                "patient_index_0based": int(patient_index),
                "true_label": int(row["true_label"]),
                "logit_asd": float(logits[position, 0]),
                "logit_td": float(logits[position, 1]),
                "p_asd": float(probabilities[position, 0]),
                "p_td": float(probabilities[position, 1]),
                "prediction": int(prediction[position]),
            })
        labels_np = np.asarray(labels, dtype=np.int64)
        ref_p = np.asarray(reference_probability, dtype=np.float64)
        ref_pred = np.asarray(reference_prediction, dtype=np.int64)
        exact_probability = bool(np.array_equal(probabilities[:, 0], ref_p))
        exact_prediction = bool(np.array_equal(prediction, ref_pred))
        computed = _binary_metrics(labels_np, prediction, probabilities)
        fold_status.append({
            "fold": fold,
            "test_rows": len(test),
            "test_order_exact": order_exact,
            "probability_exact": exact_probability,
            "hard_prediction_exact": exact_prediction,
            "checkpoint_correct": int(evaluation["correct"]),
            "replayed_correct": computed["correct"],
            "metrics": computed,
            "maximum_probability_abs_error": float(np.max(np.abs(probabilities[:, 0] - ref_p))),
        })
    result = {
        "status": "PASS" if all(
            item["test_order_exact"] and item["probability_exact"] and item["hard_prediction_exact"]
            and item["checkpoint_correct"] == item["replayed_correct"] for item in fold_status
        ) and len(output_rows) == 871 else "FAIL",
        "folds": fold_status,
        "patient_count": len(output_rows),
        "columns": [
            "fold", "test_position", "patient_id_1based", "patient_index_0based", "true_label",
            "logit_asd", "logit_td", "p_asd", "p_td", "prediction",
        ],
    }
    if write:
        path = ROOT / "results" / "evaluation_replay.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=result["columns"])
            writer.writeheader()
            writer.writerows(output_rows)
    return result


def verify_checkpoint_payloads() -> dict[str, Any]:
    records = {(int(row["fold"]), row["kind"]): row for row in source_records()["checkpoint_records"]}
    checks: list[dict[str, Any]] = []
    for fold in FOLDS:
        best = load_checkpoint(fold, "best")
        latest = load_checkpoint(fold, "latest")
        fold_errors: list[str] = []
        for kind, payload in (("best", best), ("latest", latest)):
            path = checkpoint_path(fold, kind)
            record = records[(fold, kind)]
            if sha256_file(path) != record["sha256"]:
                fold_errors.append(f"{kind}: copied SHA differs from source record")
            missing = sorted(REQUIRED_CHECKPOINT_KEYS.difference(payload))
            if missing:
                fold_errors.append(f"{kind}: missing checkpoint keys {missing}")
            if payload.get("group") != "G2":
                fold_errors.append(f"{kind}: internal model group is {payload.get('group')!r}, expected G2")
            if int(payload.get("fold", -1)) != fold:
                fold_errors.append(f"{kind}: fold field differs")
            if not REQUIRED_MEMBERSHIP_KEYS.issubset(payload.get("membership", {})):
                fold_errors.append(f"{kind}: membership is incomplete")
            if not REQUIRED_RNG_KEYS.issubset(payload.get("rng", {})):
                fold_errors.append(f"{kind}: RNG state is incomplete")
            if not REQUIRED_EMA_KEYS.issubset(payload.get("ema", {})):
                fold_errors.append(f"{kind}: EMA/Gaussian state is incomplete")
            if not REQUIRED_STATE_KEYS.issubset(payload.get("state", {})):
                fold_errors.append(f"{kind}: state bundle is incomplete")
            if not REQUIRED_NATIVE_KEYS.issubset(payload.get("state", {}).get("native", {})):
                fold_errors.append(f"{kind}: native optimizer/model state is incomplete")
            evaluation = payload.get("evaluation", {})
            if set(evaluation) < {"logits", "probability", "prediction", "correct"}:
                fold_errors.append(f"{kind}: evaluation payload is incomplete")
        if latest.get("best_sha") and latest["best_sha"] != sha256_file(checkpoint_path(fold, "best")):
            fold_errors.append("latest.best_sha does not point to the copied Best")
        if int(latest["epoch"]) < int(best["best_epoch"]):
            fold_errors.append("Latest epoch precedes Best selection epoch")
        checks.append({"fold": fold, "status": "PASS" if not fold_errors else "FAIL", "errors": fold_errors})
    return {"status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL", "folds": checks}


def verify_warmups_and_membership() -> dict[str, Any]:
    records = {int(row["fold"]): row for row in source_records()["warmup_records"]}
    checks = []
    for fold in FOLDS:
        warmup_path = ROOT / "inputs" / "warmup" / f"fold{fold:02d}.pt"
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=False)
        best = load_checkpoint(fold, "best")
        membership = best["membership"]
        errors: list[str] = []
        if sha256_file(warmup_path) != records[fold]["sha256"]:
            errors.append("warmup SHA differs from source record")
        required = {"model", "cam", "optimizer_base", "optimizer_cam", "optimizer_added", "rng",
                    "train_indices", "test_indices", "train_pairs", "train_same", "rng_trace"}
        missing = sorted(required.difference(warmup))
        if missing:
            errors.append(f"warmup is incomplete: {missing}")
        if not np.array_equal(_array(warmup["test_indices"]), _array(membership["test"])):
            errors.append("warmup/test order differs from checkpoint test order")
        if not np.array_equal(_array(warmup["train_indices"]), _array(membership["legal"])):
            errors.append("warmup train does not equal checkpoint legal supervision scope")
        overlap = set(OVERLAP_1BASED.get(fold, ()))
        recorded_overlap = {int(x) for x in best.get("train_test_overlap_1based", [])}
        derived_overlap = set((_array(membership["train"]).astype(np.int64) + 1).tolist()) & set(
            (_array(membership["test"]).astype(np.int64) + 1).tolist())
        if recorded_overlap and recorded_overlap != overlap:
            errors.append(f"overlap registration differs: {sorted(recorded_overlap)} != {sorted(overlap)}")
        if derived_overlap != overlap:
            errors.append(f"derived train/test overlap differs: {sorted(derived_overlap)} != {sorted(overlap)}")
        pairs = _array(membership["train_pairs"])
        same = _array(membership["train_same"])
        if pairs.shape != (4096, 2) or same.shape != (4096,) or not np.array_equal(np.bincount(same.astype(int), minlength=2), [2048, 2048]):
            errors.append("relation pair shape or balance differs")
        checks.append({"fold": fold, "status": "PASS" if not errors else "FAIL", "errors": errors,
                       "train_rows": int(len(membership["train"])), "legal_rows": int(len(membership["legal"])),
                       "test_rows": int(len(membership["test"])), "overlap_1based": sorted(overlap)})
    return {"status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL", "folds": checks}


def verify_overlap_and_results() -> dict[str, Any]:
    rows = formal_patient_rows()
    errors: list[str] = []
    if len(rows) != 871 or len({int(row["patient_id_1based"]) for row in rows}) != 871:
        errors.append("formal patient table is not the 871-row G1 table")
    overlap_rows = []
    expected = {(fold, patient) for fold, patients in OVERLAP_1BASED.items() for patient in patients}
    for row in rows:
        key = (int(row["fold"]), int(row["patient_id_1based"]))
        if key in expected:
            overlap_rows.append(row)
            for name in ("scmr_label_supervised", "gaussian_member", "ema_member"):
                if row[name].lower() != "false":
                    errors.append(f"overlap {key} enters {name}")
            for name in ("delta_l2", "delta_parallel_l2", "delta_perp_l2", "query_feature_change_l2"):
                if float(row[name]) != 0.0:
                    errors.append(f"overlap {key} has nonzero {name}")
    if len(overlap_rows) != 19:
        errors.append(f"expected 19 overlap rows, found {len(overlap_rows)}")
    return {"status": "PASS" if not errors else "FAIL", "overlap_count": len(overlap_rows), "errors": errors,
            "overlap_1based": {str(fold): list(patients) for fold, patients in OVERLAP_1BASED.items()}}


def verify_metrics() -> dict[str, Any]:
    rows = formal_metric_rows()
    summary = next((row for row in rows if row["scope"] == "macro_mean"), None)
    spread = next((row for row in rows if row["scope"] == "macro_population_sd"), None)
    errors = []
    values = {}
    if summary is None or spread is None:
        return {"status": "FAIL", "errors": ["formal metric summary rows are missing"]}
    for key, (expected_mean, expected_sd) in EXPECTED_METRICS_PERCENT.items():
        mean = float(summary[key]) * 100
        sd = float(spread[key]) * 100
        values[key] = {"mean_percent": mean, "population_sd_percent": sd}
        if not math.isclose(mean, expected_mean, rel_tol=0, abs_tol=5e-5):
            errors.append(f"{key} mean {mean} != {expected_mean}")
        if not math.isclose(sd, expected_sd, rel_tol=0, abs_tol=5e-5):
            errors.append(f"{key} population SD {sd} != {expected_sd}")
    return {"status": "PASS" if not errors else "FAIL", "values": values, "errors": errors}


def verify_runtime_dependency_closure() -> dict[str, Any]:
    checked = [ROOT / "cli.py", ROOT / "code" / "dpsl_runtime.py"]
    forbidden = re.compile(r"/(?:data/adodas/SPromptGL-main|model/spromptgl/experiment/experiment)")
    hits = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                hits.append({"path": str(path.relative_to(ROOT)), "line": line_no, "text": line.strip()})
    return {"status": "PASS" if not hits else "FAIL", "checked": [str(p.relative_to(ROOT)) for p in checked], "hits": hits,
            "provenance_paths_allowed": True}


def audit() -> dict[str, Any]:
    checkpoint = verify_checkpoint_payloads()
    warmup = verify_warmups_and_membership()
    replay = replay_best_evaluation(write=True)
    overlap = verify_overlap_and_results()
    metrics = verify_metrics()
    dependency = verify_runtime_dependency_closure()
    blockers = [
        "Full historical model forward for all ten folds is not executed by the standalone CLI. The archived checkpoint evaluation payload is replayed exactly.",
        "Latest next-step train/eval is validated for serialized state, optimizer, RNG, EMA, epoch, and journal identity only. No training step is run in this migration.",
    ]
    status = "PARTIALLY_STANDALONE"
    if all(item["status"] == "PASS" for item in (checkpoint, warmup, replay, overlap, metrics, dependency)):
        status = "PARTIALLY_STANDALONE"
    result = {
        "status": status,
        "completed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "checkpoint_sha_source_verification": checkpoint,
        "warmup_membership_relation_verification": warmup,
        "best_evaluation_replay": replay,
        "overlap_and_patient_table": overlap,
        "formal_metrics": metrics,
        "runtime_dependency_closure": dependency,
        "full_forward": {"status": "BLOCKED_UNEXECUTED", "folds": 0, "required": 10},
        "latest_next_step_train_eval": {"status": "BLOCKED_UNEXECUTED", "folds": 0, "required": 10},
        "blockers": blockers,
        "historical_overlap_warning": "The 19 listed patients intentionally overlap training and test labels in their historical folds and are not an independent generalization estimate.",
    }
    write_json(ROOT / "results" / "audit_status.json", result)
    return result


def resume_check() -> dict[str, Any]:
    checks = []
    records = {(int(row["fold"]), row["kind"]): row for row in source_records()["checkpoint_records"]}
    history = {(int(row["fold"]), row["kind"]): row for row in history_records()}
    for fold in FOLDS:
        latest = load_checkpoint(fold, "latest")
        errors = []
        for key in ("state", "ema", "rng", "optimizer", "epoch", "best_epoch", "wait", "journal_sha"):
            if key not in latest and key != "optimizer":
                errors.append(f"Latest lacks {key}")
        if not REQUIRED_RNG_KEYS.issubset(latest.get("rng", {})):
            errors.append("RNG streams incomplete")
        if not REQUIRED_EMA_KEYS.issubset(latest.get("ema", {})):
            errors.append("EMA/Gaussian state incomplete")
        journal_record = history.get((fold, "journal"))
        journal_ok = False
        if journal_record:
            journal_path = ROOT / journal_record["path"]
            journal_ok = sha256_file(journal_path) == journal_record["sha256"]
            if not journal_ok:
                errors.append("journal SHA differs from source record")
            raw = journal_path.read_bytes()
            if latest.get("journal_size") is not None and len(raw) != int(latest["journal_size"]):
                errors.append("Latest journal_size differs from frozen journal")
            if latest.get("journal_sha") and hashlib.sha256(raw).hexdigest() != latest["journal_sha"]:
                errors.append("Latest journal_sha differs from frozen journal")
        else:
            errors.append("frozen journal is missing")
        checks.append({"fold": fold, "status": "PASS" if not errors else "FAIL",
                       "latest_epoch": int(latest["epoch"]), "best_epoch": int(latest["best_epoch"]),
                       "optimizer_state_present": bool(latest.get("state", {}).get("optimizer")),
                       "rng_present": REQUIRED_RNG_KEYS.issubset(latest.get("rng", {})),
                       "ema_present": REQUIRED_EMA_KEYS.issubset(latest.get("ema", {})),
                       "journal_exact": journal_ok, "errors": errors})
    result = {
        "status": "PARTIALLY_STANDALONE" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "mode": "in-memory serialized recovery validation only",
        "folds": checks,
        "next_step_execution": "NOT_RUN_BY_REQUEST",
        "note": "No optimizer step, RNG advance, or checkpoint write occurred.",
    }
    write_json(ROOT / "results" / "resume_check.json", result)
    return result


def train_resume() -> dict[str, str]:
    result = {
        "status": "NOT_RUN_BY_REQUEST",
        "reason": "This migration explicitly forbids starting or continuing training.",
        "formal_checkpoint_writes": "none",
    }
    write_json(ROOT / "results" / "train_resume_status.json", result)
    return result


def inspect() -> dict[str, Any]:
    manifest = read_json(ROOT / "manifest.json")
    config = read_json(ROOT / "config.json")
    return {
        "status": manifest.get("status"),
        "authority": str(ROOT),
        "model_identity": config.get("model_identity"),
        "formal_checkpoint_mapping": manifest.get("formal_checkpoint_mapping"),
        "folds": [checkpoint_summary(fold) for fold in FOLDS],
        "overlap_19": {str(fold): list(patients) for fold, patients in OVERLAP_1BASED.items()},
        "historical_manifest_sha256": manifest.get("historical_manifest_sha256"),
    }
