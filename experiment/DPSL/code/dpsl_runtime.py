"""Integrity and archived results for the formal DPSL experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from functools import lru_cache
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


def verify_sha(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return actual


@lru_cache(maxsize=1)
def official_manifest() -> dict[str, Any]:
    path = ROOT / "code/provenance/manifests/E82_4_6_10-G1.json"
    receipt = next(row for row in source_records()["manifest_records"]
                   if ROOT / row["path"] == path)
    verify_sha(path, receipt["sha256"])
    return read_json(path)


def checkpoint_source(fold: int, kind: str) -> tuple[str, str, int]:
    official = official_manifest()
    if fold == 8:
        group = official["groups"]["G1"]
        if group["patient_ids_1based"] != [383, 430] or group["status"] != "COMPLETE_AUDITED":
            raise ValueError("Invalid Fold 8 official G1 registration")
        source = str(Path(official["directory"]) / "checkpoints" / f"G1_f08.{kind}.pt")
        training = group["training"]
        return source, training[kind + "_sha"], int(training["best_epoch" if kind == "best" else "epoch"])
    ref = official["nine_fold_reuse"][str(fold)]["checkpoint"]
    return ref[kind], ref[kind + "_sha"], int(ref[kind + "_epoch"])


def load_checkpoint(fold: int, kind: str) -> dict[str, Any]:
    path = checkpoint_path(fold, kind)
    source, digest, epoch = checkpoint_source(fold, kind)
    record = next(row for row in source_records()["checkpoint_records"]
                  if row["fold"] == fold and row["kind"] == kind)
    if (record["source_path"], record["sha256"], ROOT / record["path"]) != (source, digest, path):
        raise ValueError(f"Fold {fold} {kind} copy receipt disagrees with the official Manifest")
    verify_sha(path, digest)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, fold, kind, epoch)
    return checkpoint


def require_keys(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{label} missing {sorted(missing)}")


def validate_evaluation(evaluation: Any, count: int, labels: Any = None) -> None:
    require_keys(evaluation, {"logits", "probability", "prediction", "correct"}, "evaluation")
    logits = _array(evaluation["logits"])
    probability = _array(evaluation["probability"])
    prediction = _array(evaluation["prediction"])
    correct = evaluation["correct"]
    if (logits.shape != (count, 2) or probability.shape != (count, 2) or prediction.shape != (count,)
            or logits.dtype.kind != "f" or probability.dtype.kind != "f" or prediction.dtype.kind not in "iu"
            or not np.isfinite(logits).all() or not np.isfinite(probability).all()
            or not ((0 <= probability) & (probability <= 1)).all()
            or not np.allclose(probability.sum(axis=1), 1, rtol=0, atol=6 * np.finfo(probability.dtype).eps)
            or not np.array_equal(probability.argmax(axis=1), prediction)
            or not isinstance(correct, (int, np.integer)) or not 0 <= correct <= count
            or (labels is not None and np.count_nonzero(prediction == labels) != correct)):
        raise ValueError("Invalid evaluation shape, dtype, values or correctness")


def validate_membership(membership: Any, fold: int) -> None:
    require_keys(membership, REQUIRED_MEMBERSHIP_KEYS, "membership")
    train, legal, test, pairs, same, weights = (
        _array(membership[key]) for key in ("train", "legal", "test", "train_pairs", "train_same", "weights"))
    test_count = 88 if fold == 1 else 87
    overlap = np.array(OVERLAP_1BASED.get(fold, ()), dtype=np.int64) - 1
    for value in (train, legal, test):
        if value.ndim != 1 or value.dtype.kind not in "iu" or len(np.unique(value)) != len(value) or (
                (value < 0) | (value >= 871)).any():
            raise ValueError(f"Fold {fold} invalid member IDs")
    if (len(test) != test_count or len(legal) != 871 - test_count
            or not np.array_equal(np.intersect1d(train, test), np.sort(overlap))
            or not np.array_equal(legal, train[~np.isin(train, overlap)])):
        raise ValueError(f"Fold {fold} historical legal supervision or overlap differs")
    if (pairs.shape != (4096, 2) or pairs.dtype.kind not in "iu" or not np.isin(pairs, train).all()
            or same.shape != (4096,) or not np.isin(same, [0, 1]).all() or int(same.sum()) != 2048
            or not np.isfinite(weights).all()):
        raise ValueError(f"Fold {fold} invalid historical relation scope")


def validate_optimizer(optimizer: Any, label: str) -> None:
    require_keys(optimizer, {"state", "param_groups"}, label)
    if not isinstance(optimizer["state"], dict) or not optimizer["state"] or not optimizer["param_groups"]:
        raise ValueError(f"{label} optimizer state or groups are empty")
    seen = [position for group in optimizer["param_groups"] for position in group["params"]]
    if len(set(seen)) != len(seen) or not set(optimizer["state"]).issubset(seen):
        raise ValueError(f"{label} optimizer parameter IDs differ")
    for state in optimizer["state"].values():
        require_keys(state, {"step", "exp_avg", "exp_avg_sq"}, f"{label} optimizer moments")
        step = _array(state["step"])
        mean, second = (_array(state[k]) for k in ("exp_avg", "exp_avg_sq"))
        if (step.ndim != 0 or not np.isfinite(step) or step < 1 or step != int(step)
                or mean.shape != second.shape or mean.dtype.kind != "f" or second.dtype != mean.dtype
                or not np.isfinite(mean).all() or not np.isfinite(second).all() or (second < 0).any()):
            raise ValueError(f"{label} optimizer moments or step are invalid")


def validate_checkpoint(payload: Any, fold: int, kind: str, epoch: int) -> None:
    required = REQUIRED_CHECKPOINT_KEYS | ({"eval_ema", "best_sha", "journal_size"} if kind == "latest" else set())
    require_keys(payload, required, f"Fold {fold} {kind}")
    if (payload["group"], payload["fold"], payload["seed"], payload["epoch"]) != ("G2", fold, 0, epoch):
        raise ValueError(f"Fold {fold} {kind} has wrong formal identity or epoch")
    require_keys(payload["rng"], REQUIRED_RNG_KEYS, "RNG")
    if len(payload["rng"]["cuda"]) != 1 or not all(
            torch.is_tensor(t) and t.dtype == torch.uint8 and t.ndim == 1 and t.numel()
            for t in (payload["rng"]["cpu"], *payload["rng"]["cuda"])):
        raise ValueError(f"Fold {fold} RNG streams are malformed")
    require_keys(payload["ema"], REQUIRED_EMA_KEYS, "EMA/Gaussian")
    require_keys(payload["state"], REQUIRED_STATE_KEYS, "state bundle")
    require_keys(payload["state"]["native"], REQUIRED_NATIVE_KEYS, "native state")
    validate_optimizer(payload["state"]["optimizer"], "DPSL")
    for name in ("opt_base", "opt_cam", "opt_added"):
        validate_optimizer(payload["state"]["native"][name], name)
    validate_membership(payload["membership"], fold)
    validate_evaluation(payload["evaluation"], len(payload["membership"]["test"]))
    if (payload["best_epoch"] > epoch or payload["wait"] != epoch - payload["best_epoch"]
            or (kind == "best" and (payload["best_epoch"] != epoch
                                     or payload["best_correct"] != payload["evaluation"]["correct"]))):
        raise ValueError(f"Fold {fold} {kind} inconsistent early-stopping state")
    if kind == "latest":
        require_keys(payload["eval_ema"], REQUIRED_EMA_KEYS, "pre-update EMA")
        if payload["best_sha"] != checkpoint_source(fold, "best")[1]:
            raise ValueError(f"Fold {fold} Latest Best digest differs")


def verify_warmup_file(fold: int) -> str:
    record = next(row for row in source_records()["warmup_records"] if row["fold"] == fold)
    return verify_sha(ROOT / record["path"], record["sha256"])


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


@lru_cache(maxsize=1)
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
        "checkpoint_config": best["config"],
        "checkpoint_config_same_best_latest": best["config"] == latest["config"],
        "model_keys": {key: list(best["state"]["native"][key]) for key in ("model", "cam")}
                      | {"DPSL": list(best["state"]["module"])},
        "train_order_1based": (_array(best["membership"]["train"]) + 1).tolist(),
        "test_order_1based": (_array(best["membership"]["test"]) + 1).tolist(),
        "legal_supervision_1based": (_array(best["membership"]["legal"]) + 1).tolist(),
        "state_metadata": {"rng_sha": latest["rng_sha"], "ema_sha": latest["ema_sha"],
                           "journal_sha": latest["journal_sha"]},
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


def replay_archive(write: bool = True, folds=FOLDS) -> dict[str, Any]:
    """Replay the frozen checkpoint evaluation payload in test-member order.

    The checkpoint stores the exact historical test forward output.  The replay
    loads that payload and joins it to the frozen patient table without running
    a new optimizer step or consuming a training RNG stream.
    """
    patient = formal_patient_rows()
    indexed = row_index(patient)
    output_rows: list[dict[str, Any]] = []
    fold_status: list[dict[str, Any]] = []
    for fold in folds:
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
        ) and len(output_rows) == sum(88 if f == 1 else 87 for f in folds) else "FAIL",
        "folds": fold_status,
        "patient_count": len(output_rows),
        "columns": [
            "fold", "test_position", "patient_id_1based", "patient_index_0based", "true_label",
            "logit_asd", "logit_td", "p_asd", "p_td", "prediction",
        ],
    }
    if write:
        path = ROOT / "results" / "recheck" / "evaluation_replay.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=result["columns"])
            writer.writeheader()
            writer.writerows(output_rows)
    return result


def inspect(folds=FOLDS) -> dict[str, Any]:
    manifest = read_json(ROOT / "manifest.json")
    config = read_json(ROOT / "config.json")
    return {
        "status": manifest.get("status"),
        "authority": str(ROOT),
        "model_identity": config.get("model_identity"),
        "formal_checkpoint_mapping": manifest.get("formal_checkpoint_mapping"),
        "folds": [checkpoint_summary(fold) for fold in folds],
        "overlap_19": {str(fold): list(patients) for fold, patients in OVERLAP_1BASED.items()},
        "historical_manifest_sha256": manifest.get("historical_manifest_sha256"),
    }
