#!/usr/bin/env python3
"""Nested, fold-local statistical probes for PSDC-IVE Stage A."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
REPO = Path("/data/adodas/SPromptGL-main")
RESULTS = EXP / "results"
RT = EXP / ".runtime"
TASKS = RT / "tasks"
METHODS = ("A0", "A1", "A2", "A3", "DynamicOnly", "A1_residualized", "A2_residualized",
           "A1_no_unknown_roi", "A2_no_unknown_roi")
MAIN_METHODS = METHODS[:5]
CS = tuple(float(x) for x in CFG["probe"]["C_candidates"])


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_rows(rows: np.ndarray) -> str:
    return sha_bytes(np.asarray(rows, dtype=np.int64).tobytes())


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
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


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_inputs():
    static_csv = Path(CFG["data"]["csv_path"])
    frame = pd.read_csv(static_csv, low_memory=False)
    if frame.shape != (871, 321):
        raise ValueError(f"unexpected static data shape {frame.shape}")
    raw_y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    if not np.isin(raw_y, [1, 2]).all():
        raise ValueError("expected historical DX_GROUP coding 1=ASD and 2=TD")
    y = raw_y - 1
    x = frame.iloc[:, :320].to_numpy(dtype=np.float64)
    cache_path = EXP / CFG["runtime"]["dynamic_cache"]
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["row"], np.arange(871)):
            raise ValueError("dynamic cache row ordering changed")
        feat = {key: cache[key].copy() for key in cache.files}
    audit = pd.read_csv(RESULTS / "patient_qc_audit.csv", low_memory=False)
    if len(audit) != 871 or audit.row.duplicated().any() or not np.array_equal(audit.row.to_numpy(), np.arange(871)):
        raise ValueError("patient-level audit metadata does not align to original CSV rows")
    if not np.array_equal(feat["subject_id"], audit.subject_id.to_numpy(dtype=np.int64)):
        raise ValueError("BOLD cache and patient audit identities differ")
    eligible = feat["dynamic_eligible"].astype(bool)
    for key in ("window_mean", "dynamic", "dynamic_order", "window_mean_no_unknown_roi", "dynamic_no_unknown_roi", "dynamic_order_no_unknown_roi"):
        if not np.isfinite(feat[key][eligible]).all():
            raise ValueError(f"non-finite eligible feature in {key}")
    if not np.isfinite(x).all():
        raise ValueError("static input contains non-finite values")
    return x, y, feat, audit, eligible


def feature_blocks(method: str, x: np.ndarray, feat: dict) -> list[tuple[str, np.ndarray, int]]:
    if method == "A0":
        return [("static", x, 0)]
    if method in {"A1", "A1_residualized", "A1_no_unknown_roi"}:
        mean_key = "window_mean_no_unknown_roi" if method.endswith("no_unknown_roi") else "window_mean"
        return [("static", x, 0), ("window_mean", feat[mean_key], 32), ("static_duplicate", x[:, 64:320], 32)]
    if method in {"A2", "A2_residualized", "A2_no_unknown_roi"}:
        mean_key = "window_mean_no_unknown_roi" if method.endswith("no_unknown_roi") else "window_mean"
        dynamic_key = "dynamic_no_unknown_roi" if method.endswith("no_unknown_roi") else "dynamic"
        return [("static", x, 0), ("window_mean", feat[mean_key], 32), ("dynamic", feat[dynamic_key], 32)]
    if method == "A3":
        return [("static", x, 0), ("window_mean", feat["window_mean"], 32), ("dynamic_order", feat["dynamic_order"], 32)]
    if method == "DynamicOnly":
        return [("dynamic", feat["dynamic"], 32)]
    raise ValueError(f"unknown method {method}")


def nuisance_transform(audit: pd.DataFrame, train_rows: np.ndarray, other_rows: np.ndarray):
    columns = ["tr_seconds", "duration_seconds", "timepoints", "window_count",
               "func_mean_fd", "func_dvars", "func_perc_fd"]
    raw = audit.loc[:, ["site", *columns]].copy()
    raw["site"] = raw["site"].astype(str)
    categorical = [0]
    numerical = list(range(1, len(columns) + 1))
    transformer = ColumnTransformer([
        ("site", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                               ("scale", StandardScaler())]), numerical),
    ], remainder="drop", sparse_threshold=0.0)
    z_train = transformer.fit_transform(raw.iloc[train_rows])
    z_other = transformer.transform(raw.iloc[other_rows])
    return np.asarray(z_train, dtype=np.float64), np.asarray(z_other, dtype=np.float64)


def transform_blocks(method: str, train_rows: np.ndarray, other_rows: np.ndarray, x: np.ndarray,
                     feat: dict, audit: pd.DataFrame, residualize: bool):
    blocks = feature_blocks(method, x, feat)
    if residualize:
        z_train, z_other = nuisance_transform(audit, train_rows, other_rows)
    transformed_train, transformed_other, dimensions = [], [], {}
    for name, matrix, pca_dim in blocks:
        train_raw = np.asarray(matrix[train_rows], dtype=np.float64)
        other_raw = np.asarray(matrix[other_rows], dtype=np.float64)
        if residualize:
            residualizer = Ridge(alpha=1.0, fit_intercept=True)
            residualizer.fit(z_train, train_raw)
            train_raw = train_raw - residualizer.predict(z_train)
            other_raw = other_raw - residualizer.predict(z_other)
        scaler = StandardScaler().fit(train_raw)
        train_scaled = scaler.transform(train_raw)
        other_scaled = scaler.transform(other_raw)
        if pca_dim:
            mean = train_scaled.mean(axis=0)
            centered = train_scaled - mean
            _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
            if not len(singular_values) or singular_values[0] <= 0:
                rank = 0
            else:
                tol = max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
                rank = int(np.count_nonzero(singular_values > tol))
            effective = min(pca_dim, rank)
            if effective < 1:
                raise ValueError(f"training fold has zero-rank {name} block")
            transformed_train.append(centered @ components[:effective].T)
            transformed_other.append((other_scaled - mean) @ components[:effective].T)
            dimensions[name] = {"input": int(train_raw.shape[1]), "requested": pca_dim, "effective": effective}
        else:
            transformed_train.append(train_scaled)
            transformed_other.append(other_scaled)
            dimensions[name] = {"input": int(train_raw.shape[1]), "requested": 0, "effective": int(train_raw.shape[1])}
    return np.concatenate(transformed_train, axis=1), np.concatenate(transformed_other, axis=1), dimensions


def model_for(c: float):
    return LogisticRegression(C=c, solver="lbfgs", class_weight="balanced", max_iter=5000,
                              random_state=int(CFG["probe"]["seed"]))


def asd_probability(model: LogisticRegression, values: np.ndarray) -> np.ndarray:
    position = int(np.flatnonzero(model.classes_ == 0)[0])
    return model.predict_proba(values)[:, position]


def metric_values(y: np.ndarray, pred: np.ndarray, p_asd: np.ndarray) -> dict[str, float]:
    asd = y == 0
    return {
        "acc": float(accuracy_score(y, pred)),
        "hard_auc": float(balanced_accuracy_score(y, pred)),
        "prob_auc": float(roc_auc_score(asd.astype(int), p_asd)),
        "sen": float(recall_score(asd.astype(int), (pred == 0).astype(int), pos_label=1, zero_division=0)),
        "spe": float(recall_score((~asd).astype(int), (pred == 1).astype(int), pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(asd.astype(int), (pred == 0).astype(int), pos_label=1, zero_division=0)),
    }


def fit_method(method: str, fold: int, train_rows: np.ndarray, test_rows: np.ndarray,
               x: np.ndarray, y: np.ndarray, feat: dict, audit: pd.DataFrame):
    residualize = method.endswith("residualized")
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    inner_splits = [(train_rows[tr], train_rows[va]) for tr, va in inner.split(train_rows, y[train_rows])]
    prepared = []
    dimensions_by_inner = []
    for tr_rows, va_rows in inner_splits:
        tr_x, va_x, dimensions = transform_blocks(method, tr_rows, va_rows, x, feat, audit, residualize)
        prepared.append((tr_x, va_x, y[tr_rows], y[va_rows]))
        dimensions_by_inner.append(dimensions)
    losses = {}
    for c in CS:
        fold_losses = []
        for tr_x, va_x, tr_y, va_y in prepared:
            estimator = model_for(c).fit(tr_x, tr_y)
            fold_losses.append(log_loss((va_y == 0).astype(int), asd_probability(estimator, va_x), labels=[0, 1]))
        losses[c] = float(np.mean(fold_losses))
    best_loss = min(losses.values())
    chosen_c = min(c for c in CS if losses[c] <= best_loss + 1e-12)
    train_x, test_x, outer_dimensions = transform_blocks(method, train_rows, test_rows, x, feat, audit, residualize)
    estimator = model_for(chosen_c).fit(train_x, y[train_rows])
    p_asd = asd_probability(estimator, test_x)
    pred = np.where(p_asd >= 0.5, 0, 1)
    return {
        "pred": pred, "p_asd": p_asd, "chosen_c": chosen_c, "inner_losses": {str(k): v for k, v in losses.items()},
        "outer_dimensions": outer_dimensions, "inner_dimensions": dimensions_by_inner,
        "inner_splits": [{"train_rows_sha256": hash_rows(tr), "valid_rows_sha256": hash_rows(va),
                          "train_count": len(tr), "valid_count": len(va),
                          "train_rows": tr.tolist(), "valid_rows": va.tolist()} for tr, va in inner_splits],
    }


def run_fold(fold: int, pair: tuple[np.ndarray, np.ndarray], x, y, feat, audit, eligible):
    base_train, base_test = pair
    train_rows = base_train[eligible[base_train]]
    test_rows = base_test[eligible[base_test]]
    excluded = []
    if fold == 2:
        excluded = [row for row in (376, 406) if row in set(base_test.tolist())]
        if set(excluded) != {376, 406}:
            raise RuntimeError("historical Fold 2 overlap rows no longer belong to the frozen test fold")
        train_rows = train_rows[~np.isin(train_rows, excluded)]
    if len(test_rows) == 0 or len(np.unique(y[train_rows])) != 2 or len(np.unique(y[test_rows])) != 2:
        raise ValueError(f"fold {fold}: effective train/test does not contain both classes")
    out = {"fold": fold, "train_rows": train_rows.tolist(), "test_rows": test_rows.tolist(),
           "train_rows_sha256": hash_rows(train_rows), "test_rows_sha256": hash_rows(test_rows),
           "historical_fold_test_rows_sha256": hash_rows(base_test), "fold2_overlap_rows_removed": excluded,
           "n_train": len(train_rows), "n_test": len(test_rows), "methods": {}}
    for method in METHODS:
        result = fit_method(method, fold, train_rows, test_rows, x, y, feat, audit)
        out["methods"][method] = result
    return out


def paired_summary(name: str, left: dict, right: dict, folds: list[int]) -> list[dict]:
    rows = []
    for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
        delta = np.asarray([(left[f][metric] - right[f][metric]) * 100 for f in folds], dtype=float)
        mean = float(delta.mean())
        sd = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
        half = float(student_t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
        rows.append({"comparison": name, "scope": "all10" if len(folds) == 10 else "nine_without_fold2",
                     "metric": metric, "folds": len(folds), "mean_difference_pp": mean,
                     "wins": int((delta > 1e-12).sum()), "ties": int((np.abs(delta) <= 1e-12).sum()),
                     "losses": int((delta < -1e-12).sum()), "descriptive_95ci_low_pp": mean - half,
                     "descriptive_95ci_high_pp": mean + half,
                     "fold_differences_pp": json.dumps({str(f): float((left[f][metric] - right[f][metric]) * 100) for f in folds}, sort_keys=True)})
    return rows


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    TASKS.mkdir(parents=True, exist_ok=True)
    x, y, feat, audit, eligible = load_inputs()
    if int(eligible.sum()) == 0:
        raise RuntimeError("no patients meet the preregistered three-window minimum")
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    all_splits = list(splitter.split(x, y))
    worker_count = min(int(CFG["runtime"]["max_workers"]), os.cpu_count() or 1, 10)
    fold_results = [None] * 10
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(run_fold, i + 1, pair, x, y, feat, audit, eligible): i for i, pair in enumerate(all_splits)}
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            result = future.result()
            fold_results[index] = result
            atomic_json(TASKS / f"probe_fold_{index + 1:02d}.json", result)
            print(f"nested probe fold {index + 1}/10 complete", flush=True)
    if any(item is None for item in fold_results):
        raise RuntimeError("incomplete outer fold tasks")

    prediction_rows, fold_metric_rows, method_metrics = [], [], {method: {} for method in METHODS}
    trace_rows = []
    for outcome in fold_results:
        fold = outcome["fold"]
        test_rows = np.asarray(outcome["test_rows"], dtype=int)
        fold_trace = {k: v for k, v in outcome.items() if k not in {"methods"}}
        fold_trace["methods"] = {}
        for method in METHODS:
            item = outcome["methods"][method]
            preds, probs = np.asarray(item["pred"], dtype=int), np.asarray(item["p_asd"], dtype=float)
            metrics = metric_values(y[test_rows], preds, probs)
            method_metrics[method][fold] = metrics
            fold_metric_rows.append({"method": method, "fold": fold, "n_train": outcome["n_train"], "n_test": len(test_rows),
                                     "chosen_C": item["chosen_c"], **metrics,
                                     "train_rows_sha256": outcome["train_rows_sha256"],
                                     "test_rows_sha256": outcome["test_rows_sha256"]})
            fold_trace["methods"][method] = {"chosen_C": item["chosen_c"], "inner_losses": item["inner_losses"],
                                             "outer_dimensions": item["outer_dimensions"], "inner_dimensions": item["inner_dimensions"],
                                             "inner_splits": item["inner_splits"]}
            for row, pred, prob in zip(test_rows, preds, probs):
                audit_row = audit.iloc[int(row)]
                prediction_rows.append({"row_zero_based": int(row), "row_one_based": int(row + 1),
                                        "subject_id": int(audit_row.subject_id), "site": audit_row.site,
                                        "label_asd0_td1": int(y[row]), "fold": fold, "method": method,
                                        "prediction_asd0_td1": int(pred), "probability_asd": float(prob),
                                        "correct": bool(int(pred) == int(y[row]))})
        trace_rows.append(fold_trace)
    atomic_csv(RESULTS / "predictions.csv", prediction_rows)
    atomic_csv(RESULTS / "fold_metrics.csv", fold_metric_rows)
    atomic_json(RESULTS / "fold_trace.json", trace_rows)

    comparison_rows = []
    for left, right in (("A2", "A0"), ("A2", "A1"), ("A2", "A3"),
                        ("A2_residualized", "A1_residualized"), ("A2_no_unknown_roi", "A1_no_unknown_roi")):
        comparison_rows.extend(paired_summary(f"{left}-{right}", method_metrics[left], method_metrics[right], list(range(1, 11))))
        comparison_rows.extend(paired_summary(f"{left}-{right}", method_metrics[left], method_metrics[right], [1,3,4,5,6,7,8,9,10]))
    atomic_csv(RESULTS / "paired_comparisons.csv", comparison_rows)

    oof_rows, pooled = [], {}
    pred_frame = pd.DataFrame(prediction_rows)
    for method in METHODS:
        sub = pred_frame[pred_frame.method == method].sort_values("row_zero_based")
        yy = sub.label_asd0_td1.to_numpy(dtype=int)
        pp = sub.prediction_asd0_td1.to_numpy(dtype=int)
        probs = sub.probability_asd.to_numpy(dtype=float)
        values = metric_values(yy, pp, probs)
        pooled[method] = {k: v * 100 for k, v in values.items()}
        oof_rows.append({"method": method, "n_oof": len(sub), **{k: v * 100 for k, v in values.items()}})
    atomic_csv(RESULTS / "pooled_oof_metrics.csv", oof_rows)
    oof_deltas = {}
    for left, right in (("A2", "A0"), ("A2", "A1"), ("A2", "A3")):
        a = pred_frame[pred_frame.method == left].set_index("row_zero_based")
        b = pred_frame[pred_frame.method == right].set_index("row_zero_based")
        if not a.index.equals(b.index):
            raise RuntimeError(f"OOF rows differ between {left} and {right}")
        oof_deltas[f"{left}-{right}"] = {k: pooled[left][k] - pooled[right][k] for k in pooled[left]}
    manifest = {
        "experiment": CFG["experiment"], "status": "PROBES_COMPLETE_PENDING_INDEPENDENT_AUDIT",
        "seed": CFG["probe"]["seed"], "device": "CPU (nested regularized logistic probes)",
        "worker_count": worker_count, "workers_max": CFG["runtime"]["max_workers"],
        "subjects_total": 871, "subjects_dynamic_eligible": int(eligible.sum()),
        "common_oof_subject_count": int(eligible.sum()), "coverage": float(eligible.mean()),
        "folds_completed": sum(item is not None for item in fold_results), "outer_split": CFG["data"]["outer_split"],
        "fold2_removed_from_new_training": [376, 406], "fold2_test_kept": True,
        "methods": list(METHODS), "pooled_oof_metrics_percent": pooled, "pooled_oof_deltas_pp": oof_deltas,
        "wall_seconds": round(time.time() - started, 3),
    }
    atomic_json(RESULTS / "probe_summary.json", manifest)
    print(f"PROBES_COMPLETE={manifest['folds_completed']}/10; dynamic={manifest['subjects_dynamic_eligible']}/871; workers={worker_count}; wall={manifest['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
