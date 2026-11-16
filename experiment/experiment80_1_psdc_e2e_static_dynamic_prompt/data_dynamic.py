#!/usr/bin/env python3
"""Rebuild verified 90-second, 252-edge FC sequences for Experiment 80.1."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
SRC = CFG["sources"]
DATA = CFG["data"]
CACHE_DIR = EXP / ".runtime" / "dynamic_cache"
RESULTS = EXP / "results"
UNKNOWN_EDGES = np.asarray(DATA["e80_unknown_connection_indices_zero_based"], dtype=np.int64)
KEEP_EDGES = np.ones(256, dtype=bool)
KEEP_EDGES[UNKNOWN_EDGES] = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def e11_module():
    path = Path(SRC["e11_root"]) / "recover_roi_mapping.py"
    spec = importlib.util.spec_from_file_location("e80_1_readonly_e11_mapping", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_header(path: Path) -> list[int]:
    with path.open("rt", encoding="utf-8") as stream:
        line = stream.readline().strip()
    if not line.startswith("#") or "\t" not in line:
        raise ValueError(f"missing E11 tab-delimited ROI header: {path.name}")
    return [int(token.removeprefix("#")) for token in line.split("\t")]


def verify_inputs():
    e11 = Path(SRC["e11_root"])
    e80 = Path(SRC["e80_root"])
    mapping = pd.read_csv(e11 / "results" / "roi_256_mapping.csv")
    rows = pd.read_csv(e11 / "results" / "subject_row_mapping.csv")
    e11_audit = json.loads((e11 / "results" / "data_audit.json").read_text(encoding="utf-8"))
    e80_qc = pd.read_csv(e80 / "results" / "patient_qc_audit.csv")
    static = pd.read_csv(SRC["static_csv"], low_memory=False)
    labels = pd.read_csv(e11 / "results" / "roi_111_labels.csv")
    if len(mapping) != 256 or mapping["correlation_feature_index"].astype(int).tolist() != list(range(256)):
        raise RuntimeError("E11 frozen 256-edge order is not the expected 0..255 sequence")
    if len(rows) != 871 or rows["csv_row_zero_based"].astype(int).tolist() != list(range(871)):
        raise RuntimeError("E11 patient row mapping is incomplete or reordered")
    if len(e80_qc) != 871 or len(static) != 871 or static.shape[1] != 321:
        raise RuntimeError("E80/data static cohort dimensions differ from the locked 871 x 321 source")
    if len(labels) != 111:
        raise RuntimeError("E11 ROI header map is not 111 entries")
    if e80_qc["row"].astype(int).tolist() != list(range(871)):
        raise RuntimeError("E80 identity audit has a different row order")
    if rows["subject_SUB_ID"].astype(int).tolist() != e80_qc["subject_id"].astype(int).tolist():
        raise RuntimeError("E11 and E80 SUB_ID row identities differ")
    if rows["roi_raw_file"].astype(str).tolist() != e80_qc["file"].astype(str).tolist():
        raise RuntimeError("E11 and E80 BOLD filename mapping differs")
    edge_ids = mapping["correlation_feature_index"].astype(int).to_numpy()
    roi_i = mapping["roi_i_zero_based"].astype(int).to_numpy()
    roi_j = mapping["roi_j_zero_based"].astype(int).to_numpy()
    if not np.array_equal(np.flatnonzero((roi_i == 82) | (roi_j == 82)), UNKNOWN_EDGES):
        raise RuntimeError("E80 unknown-ROI edge indices do not match E11 endpoint mapping")
    corr_columns = [str(i) for i in range(256)]
    if any(name not in static.columns for name in corr_columns) or "label" not in static.columns:
        raise RuntimeError("static CSV does not expose the frozen CORR256 and label columns")
    static_corr = static[corr_columns].to_numpy(dtype=np.float64)
    e11_hash = {
        item["relative_path"]: item["sha256"]
        for item in e11_audit["inputs"]["roi_manifest"]
    }
    qc_hash = dict(zip(e80_qc["file"].astype(str), e80_qc["file_sha256"].astype(str)))
    qc_expected = dict(zip(e80_qc["file"].astype(str), e80_qc["expected_e11_bold_sha256"].astype(str)))
    if len(e11_hash) != 871 or len(qc_hash) != 871 or any(qc_hash[k] != qc_expected[k] for k in qc_hash):
        raise RuntimeError("E11/E80 BOLD hash inventories are incomplete or inconsistent")
    return mapping, rows, e80_qc, labels, static, static_corr, roi_i, roi_j, e11_hash


def fc_fisher_z(series: np.ndarray, roi_i: np.ndarray, roi_j: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(series, dtype=np.float64).copy()
    x -= x.mean(axis=0)
    std = x.std(axis=0)
    std[std < np.finfo(np.float64).eps] = 1.0
    x /= std
    estimator = LedoitWolf(store_precision=False).fit(x)
    covariance = estimator.covariance_
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
        raise ValueError("non-finite covariance or nonpositive covariance diagonal")
    inv_sd = 1.0 / np.sqrt(diagonal)
    corr = covariance * inv_sd[:, None] * inv_sd[None, :]
    np.fill_diagonal(corr, 1.0)
    selected = corr[roi_i, roi_j]
    if not np.isfinite(selected).all() or np.any(np.abs(selected) >= 1):
        raise ValueError("invalid selected correlation; no clipping or imputation is permitted")
    z = np.arctanh(selected)
    if not np.isfinite(z).all():
        raise ValueError("non-finite selected Fisher-z")
    return z, float(estimator.shrinkage_)


def extract_subject(args):
    row, record, expected_header, expected_sha, roi_i, roi_j, static_corr = args
    bold_root = Path(SRC["bold_root"])
    path = bold_root / record["roi_raw_file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    if digest != expected_sha:
        raise RuntimeError(f"BOLD hash differs from E11 inventory for row={row}")
    if parse_header(path) != expected_header:
        raise RuntimeError(f"ROI column identity/order differs from E11 for row={row}")
    series = np.loadtxt(path, skiprows=1, dtype=np.float64, ndmin=2)
    if series.shape[1] != 111 or not np.isfinite(series).all():
        raise RuntimeError(f"invalid BOLD shape or values at row={row}: {series.shape}")
    full, _ = fc_fisher_z(series, roi_i, roi_j)
    error = np.abs(full - static_corr[row])
    if error.shape != (256,) or float(error.max()) > 1e-6:
        raise RuntimeError(f"static CORR256 reconstruction failed row={row}: {error.max():.6g}")

    site = str(record["SITE_ID"]).upper()
    tr = float(DATA["tr_seconds_by_site_id"][site])
    width = max(30, int(math.ceil(90.0 / tr)))
    stride = max(1, int(math.floor(width * 0.5)))
    starts = list(range(0, series.shape[0] - width + 1, stride))
    if len(starts) < 3:
        raise RuntimeError(f"90-second window coverage below three windows at row={row}: {len(starts)}")
    windows = np.empty((len(starts), 252), dtype=np.float64)
    shrinkage = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        edge, shrink = fc_fisher_z(series[start:start + width], roi_i, roi_j)
        windows[index] = edge[KEEP_EDGES]
        shrinkage[index] = shrink
    if not np.isfinite(windows).all() or not np.isfinite(shrinkage).all():
        raise RuntimeError(f"non-finite 90-second FC or shrinkage at row={row}")
    return {
        "row": int(row), "subject_id": int(record["SUB_ID"]), "site": site,
        "file": str(record["roi_raw_file"]), "file_sha256": digest,
        "timepoints": int(series.shape[0]), "tr_seconds": tr,
        "duration_seconds": float(series.shape[0] * tr), "window_width_samples": width,
        "window_stride_samples": stride, "window_duration_seconds": float(width * tr),
        "window_count": len(starts), "max_static_error": float(error.max()),
        "mean_static_error": float(error.mean()), "constant_roi_indices": np.flatnonzero(np.ptp(series, axis=0) == 0).tolist(),
        "window_fc": windows, "window_shrinkage": shrinkage,
    }


def build_cache(force: bool = False, workers: int = 8):
    mapping, rowmap, e80_qc, labels, static, static_corr, roi_i, roi_j, e11_hash = verify_inputs()
    cache_path = CACHE_DIR / "windows90_fc252.npz"
    audit_path = RESULTS / "window_patient_audit.csv"
    summary_path = RESULTS / "dynamic_qc_summary.json"
    if cache_path.is_file() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("cache_sha256") == sha256(cache_path) and summary.get("status") == "PASS_90S_DATA":
            print(json.dumps({"status": "CACHE_VERIFIED", "cache": str(cache_path), "subjects": summary["subjects"]}))
            return summary
        raise RuntimeError("existing E80.1 dynamic cache exists but is not verified; refusing overwrite")

    expected_header = pd.to_numeric(labels["atlas_integer_value"], errors="raise").astype(int).tolist()
    phenotype = pd.read_csv(SRC["phenotype_csv"], low_memory=False)
    phenotype["SUB_ID"] = pd.to_numeric(phenotype["SUB_ID"], errors="coerce")
    pheno_by_sid = phenotype.dropna(subset=["SUB_ID"]).set_index("SUB_ID", drop=False)
    qc_by_file = e80_qc.set_index("file", drop=False)
    records = []
    for item in rowmap.to_dict("records"):
        sid = int(item["subject_SUB_ID"])
        file_name = str(item["roi_raw_file"])
        if sid not in pheno_by_sid.index or file_name not in qc_by_file.index:
            raise RuntimeError(f"missing verified phenotype or E80 QC metadata for SUB_ID={sid}")
        raw = pheno_by_sid.loc[sid]
        if isinstance(raw, pd.DataFrame):
            raise RuntimeError(f"duplicate phenotype SUB_ID={sid}")
        if int(raw["DX_GROUP"]) != int(static.loc[item["csv_row_zero_based"], "label"]):
            raise RuntimeError(f"phenotype/static diagnosis mismatch for row={item['csv_row_zero_based']}")
        file_inventory_key = file_name
        expected_sha = str(qc_by_file.loc[file_inventory_key, "expected_e11_bold_sha256"])
        if expected_sha not in e11_hash.values():
            raise RuntimeError(f"BOLD file hash absent from E11 inventory for row={item['csv_row_zero_based']}")
        record = dict(raw)
        record["SUB_ID"] = sid
        record["roi_raw_file"] = file_name
        records.append((
            int(item["csv_row_zero_based"]), record, expected_header, expected_sha,
            roi_i, roi_j, static_corr,
        ))

    results: list[dict | None] = [None] * len(records)
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 10))) as pool:
            futures = [pool.submit(extract_subject, item) for item in records]
            for completed, future in enumerate(futures, 1):
                value = future.result()
                results[value["row"]] = value
                if completed % 100 == 0 or completed == len(futures):
                    print(f"90-second FC extracted {completed}/{len(futures)}", flush=True)
    ordered = [x for x in results if x is not None]
    if len(ordered) != 871 or [x["row"] for x in ordered] != list(range(871)):
        raise RuntimeError("dynamic feature extraction did not preserve all original patient rows")
    if len({x["subject_id"] for x in ordered}) != 871 or any(x["window_count"] < 3 for x in ordered):
        raise RuntimeError("duplicate patient identity or insufficient 90-second window coverage")

    max_windows = max(x["window_count"] for x in ordered)
    windows = np.zeros((871, max_windows, 252), dtype=np.float32)
    counts = np.zeros(871, dtype=np.int16)
    shrink = np.full((871, max_windows), np.nan, dtype=np.float32)
    for i, item in enumerate(ordered):
        count = item["window_count"]
        counts[i] = count
        windows[i, :count] = item["window_fc"].astype(np.float32)
        shrink[i, :count] = item["window_shrinkage"].astype(np.float32)
    arrays = {
        "windows": windows,
        "window_count": counts,
        "window_shrinkage": shrink,
        "row": np.arange(871, dtype=np.int32),
        "subject_id": np.asarray([x["subject_id"] for x in ordered], dtype=np.int64),
        "site": np.asarray([x["site"] for x in ordered], dtype="U16"),
        "tr_seconds": np.asarray([x["tr_seconds"] for x in ordered], dtype=np.float32),
        "timepoints": np.asarray([x["timepoints"] for x in ordered], dtype=np.int16),
        "window_width_samples": np.asarray([x["window_width_samples"] for x in ordered], dtype=np.int16),
        "window_stride_samples": np.asarray([x["window_stride_samples"] for x in ordered], dtype=np.int16),
        "window_duration_seconds": np.asarray([x["window_duration_seconds"] for x in ordered], dtype=np.float32),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temp_cache = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
    with temp_cache.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_cache, cache_path)

    audit_rows = []
    for item in ordered:
        audit_rows.append({k: item[k] for k in (
            "row", "subject_id", "site", "file", "file_sha256", "timepoints", "tr_seconds",
            "duration_seconds", "window_width_samples", "window_stride_samples",
            "window_duration_seconds", "window_count", "max_static_error", "mean_static_error",
            "constant_roi_indices",
        )})
        audit_rows[-1]["constant_roi_indices"] = json.dumps(audit_rows[-1]["constant_roi_indices"])
        audit_rows[-1]["mean_shrinkage"] = float(np.mean(item["window_shrinkage"]))
        audit_rows[-1]["max_shrinkage"] = float(np.max(item["window_shrinkage"]))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = audit_path.with_name(audit_path.name + f".tmp.{os.getpid()}")
    with temp_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_csv, audit_path)

    all_shrink = np.concatenate([x["window_shrinkage"] for x in ordered])
    window_counts = np.asarray([x["window_count"] for x in ordered])
    summary = {
        "status": "PASS_90S_DATA",
        "subjects": len(ordered),
        "roi_count": 111,
        "connections_used": 252,
        "excluded_unknown_roi_edge_indices_zero_based": UNKNOWN_EDGES.tolist(),
        "coverage": float(len(ordered) / 871),
        "window_count_min": int(window_counts.min()),
        "window_count_mean": float(window_counts.mean()),
        "window_count_median": float(np.median(window_counts)),
        "window_count_max": int(window_counts.max()),
        "timepoints_min": int(min(x["timepoints"] for x in ordered)),
        "timepoints_max": int(max(x["timepoints"] for x in ordered)),
        "window_seconds_min": float(min(x["window_duration_seconds"] for x in ordered)),
        "window_seconds_max": float(max(x["window_duration_seconds"] for x in ordered)),
        "static_reconstruction_elements": 871 * 256,
        "static_reconstruction_max_absolute_error": float(max(x["max_static_error"] for x in ordered)),
        "static_reconstruction_all_within_1e-6": bool(all(x["max_static_error"] <= 1e-6 for x in ordered)),
        "all_window_fc_finite": bool(np.isfinite(windows).all()),
        "shrinkage_min": float(all_shrink.min()),
        "shrinkage_median": float(np.median(all_shrink)),
        "shrinkage_max": float(all_shrink.max()),
        "shrinkage_ge_0.99_fraction": float(np.mean(all_shrink >= 0.99)),
        "site_counts": {str(site): int(np.sum(np.asarray([x["site"] for x in ordered]) == site)) for site in sorted({x["site"] for x in ordered})},
        "e11_mapping_sha256": sha256(Path(SRC["e11_root"]) / "results" / "roi_256_mapping.csv"),
        "e11_row_mapping_sha256": sha256(Path(SRC["e11_root"]) / "results" / "subject_row_mapping.csv"),
        "e80_cache_reference_sha256": sha256(Path(SRC["e80_root"]) / ".runtime" / "dynamic_cache" / "feature_cache.npz"),
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "patient_audit": str(audit_path),
        "patient_audit_sha256": sha256(audit_path),
        "sampling_gap_limitation": CFG["windows"]["sampling_gap_limitation"],
    }
    atomic_json(summary_path, summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.force, args.workers), indent=2, ensure_ascii=False))
