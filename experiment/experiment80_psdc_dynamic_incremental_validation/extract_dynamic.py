#!/usr/bin/env python3
"""Audit E11 BOLD identity and extract fixed-window PSDC-IVE statistics."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
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
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
REPO = Path("/data/adodas/SPromptGL-main")
RT = EXP / ".runtime"
CACHE_DIR = RT / "dynamic_cache"
RESULTS = EXP / "results"
BOLD_ROOT = Path(CFG["data"]["bold_root"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
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


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def e11_fc_function():
    path = Path(CFG["data"]["e11_root"]) / "recover_roi_mapping.py"
    spec = importlib.util.spec_from_file_location("e80_readonly_e11_mapping", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.nilearn_071_compatible


def e11_fc_with_shrinkage(series: np.ndarray, roi_i: np.ndarray, roi_j: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(series, dtype=np.float64).copy()
    x -= x.mean(axis=0)
    std = x.std(axis=0)
    std[std < np.finfo(np.float64).eps] = 1.0
    x /= std
    estimator = LedoitWolf(store_precision=False).fit(x)
    covariance = estimator.covariance_
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
        raise ValueError("non-finite covariance or nonpositive diagonal")
    inv_sd = 1.0 / np.sqrt(diagonal)
    corr = covariance * inv_sd[:, None] * inv_sd[None, :]
    np.fill_diagonal(corr, 1.0)
    edge_corr = corr[roi_i, roi_j]
    if not np.isfinite(edge_corr).all() or np.any(np.abs(edge_corr) >= 1.0):
        raise ValueError("invalid off-diagonal correlation; no clipping or imputation allowed")
    z = np.arctanh(edge_corr)
    if not np.isfinite(z).all():
        raise ValueError("non-finite off-diagonal Fisher-z; no clipping or imputation allowed")
    return z, float(estimator.shrinkage_)


def get_prefix_site(filename: str) -> str:
    stem = filename.removesuffix("_rois_ho.1D")
    parts = stem.split("_")
    if parts[0] in {"Leuven", "UCLA", "UM"}:
        return "_".join(parts[:2]).upper()
    if parts[0] == "MaxMun":
        return "MAX_MUN"
    if parts[0] == "CMU":
        return "CMU"
    return parts[0].upper()


def verify_filename_site(filename: str, site: str) -> bool:
    stem = filename.removesuffix("_rois_ho.1D")
    expected = {
        "CALTECH": ("Caltech_",), "CMU": ("CMU_a_", "CMU_b_"), "KKI": ("KKI_",),
        "LEUVEN_1": ("Leuven_1_",), "LEUVEN_2": ("Leuven_2_",),
        "MAX_MUN": ("MaxMun_",), "NYU": ("NYU_",),
        "OLIN": ("Olin_",), "OHSU": ("OHSU_",), "PITT": ("Pitt_",),
        "SBL": ("SBL_",), "SDSU": ("SDSU_",), "STANFORD": ("Stanford_",),
        "TRINITY": ("Trinity_",), "UCLA_1": ("UCLA_1_",), "UCLA_2": ("UCLA_2_",),
        "UM_1": ("UM_1_",), "UM_2": ("UM_2_",), "USM": ("USM_",), "YALE": ("Yale_",)
    }
    return site in expected and stem.startswith(expected[site])


def header_labels(path: Path) -> list[int]:
    with path.open("rt", encoding="utf-8") as stream:
        first = stream.readline().strip()
    if not first.startswith("#") or "\t" not in first:
        raise ValueError(f"expected tab-delimited ROI header in {path.name}")
    return [int(token.removeprefix("#")) for token in first.split("\t")]


def extract_one(record: dict, fc_fn, roi_i: np.ndarray, roi_j: np.ndarray, expected_header: list[int],
                expected_static: np.ndarray, unknown_edges: list[int]) -> dict:
    path = BOLD_ROOT / record["roi_raw_file"]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256(path)
    expected_tr = CFG["tr_seconds_by_site_id"].get(record["site"])
    if expected_tr is None or not math.isfinite(float(expected_tr)) or float(expected_tr) <= 0:
        raise ValueError(f"unverified TR for site {record['site']}")
    labels = header_labels(path)
    if labels != expected_header:
        raise ValueError(f"ROI header order mismatch for {record['roi_raw_file']}")
    series = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if series.shape[1] != 111:
        raise ValueError(f"expected 111 ROIs, got {series.shape} in {path.name}")
    if not np.isfinite(series).all():
        raise ValueError(f"non-finite BOLD value in {path.name}")
    constant = np.flatnonzero(np.ptp(series, axis=0) == 0).astype(int).tolist()

    with threadpool_limits(limits=1):
        full_fc = fc_fn(series, roi_i, roi_j)
        error = np.abs(full_fc - expected_static)
        if error.shape != (256,):
            raise ValueError(f"unexpected frozen edge dimension for row {record['row']}")
        if float(error.max()) > CFG["static_fc"]["max_absolute_error_tolerance"]:
            raise ValueError(f"E11 static reconstruction failed row {record['row']} maxerr={error.max():.9g}")

        tr = float(expected_tr)
        width = max(CFG["windows"]["minimum_window_samples"], int(math.ceil(CFG["windows"]["target_seconds"] / tr)))
        stride = max(1, int(math.floor(width * CFG["windows"]["stride_fraction"])))
        starts = list(range(0, series.shape[0] - width + 1, stride))
        if len(starts) < CFG["windows"]["minimum_windows"]:
            nan256 = np.full(256, np.nan, dtype=np.float64)
            nan512 = np.full(512, np.nan, dtype=np.float64)
            keep = np.ones(256, dtype=bool)
            keep[np.asarray(unknown_edges, dtype=int)] = False
            return {
                "row": record["row"], "subject_id": record["subject_id"], "site": record["site"],
                "file": record["roi_raw_file"], "file_sha256": actual_sha, "file_bytes": path.stat().st_size,
                "timepoints": int(series.shape[0]), "tr_seconds": tr,
                "duration_seconds": float(series.shape[0] * tr), "window_width_samples": width,
                "window_stride_samples": stride, "window_duration_seconds": float(width * tr),
                "window_count": len(starts), "dynamic_eligible": False,
                "constant_roi_indices": constant, "max_abs_static_reconstruction_error": float(error.max()),
                "mean_abs_static_reconstruction_error": float(error.mean()), "window_mean": nan256,
                "dynamic": nan512, "dynamic_order": nan512, "window_mean_no_unknown_roi": nan256[keep],
                "dynamic_no_unknown_roi": np.full(504, np.nan), "dynamic_order_no_unknown_roi": np.full(504, np.nan),
                "mean_minus_full_l1": float("nan"), "mean_window_variability": float("nan"),
                "mean_adjacent_change": float("nan"), "mean_abs_window_fisher_z": float("nan"),
                "max_abs_window_fisher_z": float("nan"), "window_count_nonfinite": 0,
                "window_shrinkages": np.empty(0), "window_shrinkage_min": float("nan"),
                "window_shrinkage_median": float("nan"), "window_shrinkage_max": float("nan"),
                "unknown_edge_mean_variability": float("nan"),
                "unknown_edge_mean_adjacent_change": float("nan"),
                "unknown_edge_dynamic_variance_share": float("nan"), "window_fc": np.empty((0, 256))
            }
        window_results = [e11_fc_with_shrinkage(series[start:start + width], roi_i, roi_j) for start in starts]
        windows = np.stack([x[0] for x in window_results])
        shrinkages = np.asarray([x[1] for x in window_results], dtype=np.float64)
    if not np.isfinite(windows).all():
        raise ValueError(f"non-finite window FC in {path.name}")
    mean = windows.mean(axis=0)
    variability = windows.std(axis=0, ddof=CFG["windows"]["statistical_ddof"])
    adjacent = np.abs(np.diff(windows, axis=0)).mean(axis=0)
    order_rng = np.random.default_rng(stable_seed(f"E80-A3-{CFG['windows']['order_control_seed_base']}-{record['subject_id']}"))
    permutation = order_rng.permutation(len(windows))
    adjacent_order = np.abs(np.diff(windows[permutation], axis=0)).mean(axis=0)
    keep = np.ones(256, dtype=bool)
    keep[np.asarray(unknown_edges, dtype=int)] = False
    return {
        "row": record["row"],
        "subject_id": record["subject_id"],
        "site": record["site"],
        "file": record["roi_raw_file"],
        "file_sha256": actual_sha,
        "file_bytes": path.stat().st_size,
        "timepoints": int(series.shape[0]),
        "tr_seconds": tr,
        "duration_seconds": float(series.shape[0] * tr),
        "window_width_samples": width,
        "window_stride_samples": stride,
        "window_duration_seconds": float(width * tr),
        "window_count": len(windows),
        "dynamic_eligible": True,
        "window_shrinkages": shrinkages,
        "window_shrinkage_min": float(shrinkages.min()),
        "window_shrinkage_median": float(np.median(shrinkages)),
        "window_shrinkage_max": float(shrinkages.max()),
        "constant_roi_indices": constant,
        "max_abs_static_reconstruction_error": float(error.max()),
        "mean_abs_static_reconstruction_error": float(error.mean()),
        "window_mean": mean,
        "dynamic": np.concatenate([variability, adjacent]),
        "dynamic_order": np.concatenate([variability, adjacent_order]),
        "window_permutation": permutation.astype(np.int16),
        "window_mean_no_unknown_roi": mean[keep],
        "dynamic_no_unknown_roi": np.concatenate([variability[keep], adjacent[keep]]),
        "dynamic_order_no_unknown_roi": np.concatenate([variability[keep], adjacent_order[keep]]),
        "mean_minus_full_l1": float(np.abs(mean - full_fc).mean()),
        "mean_window_variability": float(variability.mean()),
        "mean_adjacent_change": float(adjacent.mean()),
        "mean_abs_window_fisher_z": float(np.abs(windows).mean()),
        "max_abs_window_fisher_z": float(np.abs(windows).max()),
        "window_count_nonfinite": int((~np.isfinite(windows)).sum()),
        "unknown_edge_mean_variability": float(variability[unknown_edges].mean()),
        "unknown_edge_mean_adjacent_change": float(adjacent[unknown_edges].mean()),
        "unknown_edge_dynamic_variance_share": float(np.var(np.concatenate([variability[unknown_edges], adjacent[unknown_edges]])) / max(np.var(np.concatenate([variability, adjacent])), 1e-30)),
        "window_fc": windows
    }


def main() -> None:
    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = Path(CFG["data"]["csv_path"])
    phenotype_path = Path(CFG["data"]["phenotype_path"])
    static = pd.read_csv(csv_path, low_memory=False)
    phenotype = pd.read_csv(phenotype_path, low_memory=False)
    mapping = pd.read_csv(CFG["static_fc"]["mapping_csv"])
    row_map = pd.read_csv(CFG["static_fc"]["row_mapping_csv"])
    roi_labels = pd.read_csv(CFG["static_fc"]["roi_labels_csv"])
    if len(static) != 871 or len(row_map) != 871 or len(mapping) != 256:
        raise ValueError("E80 requires the frozen 871-row / 256-edge E11 data inventory")
    corr_columns = [str(i) for i in range(256)]
    if static.shape[1] != 321 or list(static.columns[-257:-1]) != corr_columns:
        raise ValueError("static CSV must contain the frozen 320 ordered inputs followed by label")
    static_x = static.iloc[:, :320].to_numpy(dtype=np.float64)
    static_label_raw = pd.to_numeric(static["label"], errors="raise").to_numpy(dtype=int)
    if not np.isin(static_label_raw, [1, 2]).all():
        raise ValueError("historical raw labels must be DX_GROUP 1=ASD and 2=TD")
    static_y = static_label_raw - 1
    static_corr = static[corr_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(static_x).all() or not np.isin(static_y, [0, 1]).all():
        raise ValueError("static input or ASD=0/TD=1 labels are invalid")
    pheno = phenotype.copy()
    pheno["SUB_ID"] = pd.to_numeric(pheno["SUB_ID"], errors="raise").astype(int)
    row_map["subject_SUB_ID"] = pd.to_numeric(row_map["subject_SUB_ID"], errors="raise").astype(int)
    row_map["csv_row_zero_based"] = pd.to_numeric(row_map["csv_row_zero_based"], errors="raise").astype(int)
    if row_map["subject_SUB_ID"].duplicated().any() or row_map["csv_row_zero_based"].duplicated().any():
        raise ValueError("E11 patient-to-row map is not bijective")
    merged = row_map.merge(pheno, left_on="subject_SUB_ID", right_on="SUB_ID", how="left", validate="one_to_one")
    if len(merged) != 871 or merged["SITE_ID"].isna().any():
        raise ValueError("unable to match every E11 SUB_ID to original phenotype/site metadata")
    if merged["FILE_ID"].isna().any():
        raise ValueError("missing FILE_ID in phenotype metadata")
    files = list(BOLD_ROOT.glob("*_rois_ho.1D"))
    if len(files) != 871 or len({p.name for p in files}) != 871:
        raise ValueError(f"expected exactly 871 unique BOLD files; found {len(files)}")
    expected_roi_header = pd.to_numeric(roi_labels["atlas_integer_value"], errors="raise").astype(int).tolist()
    if len(expected_roi_header) != 111 or len(set(expected_roi_header)) != 111:
        raise ValueError("E11 frozen 111-ROI order is malformed")
    for idx, record in merged.iterrows():
        row = int(record["csv_row_zero_based"])
        sid = int(record["subject_SUB_ID"])
        if not (0 <= row < 871) or int(record["DX_GROUP"]) != int(static_label_raw[row]):
            raise ValueError(f"E11 row/label identity mismatch for SUB_ID={sid}")
        if int(record["csv_row_one_based"]) != row + 1:
            raise ValueError("row mapping zero-based/one-based identity mismatch")
        if str(sid) not in str(record["FILE_ID"]):
            raise ValueError(f"SUB_ID is not present in raw phenotype FILE_ID: {sid}")
        if get_prefix_site(Path(record["roi_raw_file"]).name) != str(record["SITE_ID"]).upper():
            raise ValueError(f"BOLD filename/site identity mismatch for SUB_ID={sid}")
        if not verify_filename_site(Path(record["roi_raw_file"]).name, str(record["SITE_ID"]).upper()):
            raise ValueError(f"BOLD subsite prefix does not match SITE_ID for SUB_ID={sid}")
    if sorted(merged["csv_row_zero_based"].astype(int).tolist()) != list(range(871)):
        raise ValueError("row mapping is not a complete 871-row cover")
    roi_i = pd.to_numeric(mapping["roi_i_zero_based"], errors="raise").to_numpy(dtype=int)
    roi_j = pd.to_numeric(mapping["roi_j_zero_based"], errors="raise").to_numpy(dtype=int)
    feature_idx = pd.to_numeric(mapping["correlation_feature_index"], errors="raise").to_numpy(dtype=int)
    conn_idx = pd.to_numeric(mapping["original_connection_index"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(feature_idx, np.arange(256)) or len(set(conn_idx.tolist())) != 256 or np.any(roi_i >= roi_j) or np.any(roi_j >= 111):
        raise ValueError("E11 frozen connection order/index is invalid")
    unknown_edges = CFG["static_fc"]["unknown_roi_edge_indices_zero_based"]
    actual_unknown_edges = np.flatnonzero((roi_i == CFG["static_fc"]["unknown_roi_index_zero_based"]) | (roi_j == CFG["static_fc"]["unknown_roi_index_zero_based"])).tolist()
    if actual_unknown_edges != unknown_edges:
        raise ValueError(f"the four E11 unknown-ROI edges changed: {actual_unknown_edges}")

    e11_audit = json.loads((Path(CFG["data"]["e11_root"]) / "results/data_audit.json").read_text())
    expected_hashes = {x["relative_path"]: x["sha256"] for x in e11_audit["inputs"]["roi_manifest"]}
    if len(expected_hashes) != 871:
        raise ValueError("E11 historical BOLD manifest does not cover all subjects")
    fc_fn = e11_fc_function()
    records = []
    for record in merged.to_dict(orient="records"):
        row = int(record["csv_row_zero_based"])
        records.append({
            "row": row,
            "subject_id": int(record["subject_SUB_ID"]),
            "site": str(record["SITE_ID"]).upper(),
            "roi_raw_file": str(record["roi_raw_file"]),
            "expected_sha256": expected_hashes.get(str(record["roi_raw_file"])),
            "func_mean_fd": _float_or_nan(record.get("func_mean_fd")),
            "func_dvars": _float_or_nan(record.get("func_dvars")),
            "func_perc_fd": _float_or_nan(record.get("func_perc_fd")),
            "func_num_fd": _float_or_nan(record.get("func_num_fd")),
            "func_outlier": _float_or_nan(record.get("func_outlier")),
            "func_quality": _float_or_nan(record.get("func_quality")),
            "func_gsr": _float_or_nan(record.get("func_gsr")),
            "age_at_scan": _float_or_nan(record.get("AGE_AT_SCAN")),
            "label": int(static_y[row])
        })
    results = [None] * len(records)
    workers = min(int(CFG["runtime"]["max_workers"]), os.cpu_count() or 1)
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_one, rec, fc_fn, roi_i, roi_j, expected_roi_header, static_corr[rec["row"]], unknown_edges): rec["row"]
            for rec in records
        }
        for done, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            result = future.result()
            if result["file_sha256"] != records[row]["expected_sha256"]:
                raise ValueError(f"current BOLD SHA differs from E11 for row {row}")
            for key in ("func_mean_fd", "func_dvars", "func_perc_fd", "func_num_fd", "func_outlier", "func_quality", "func_gsr", "age_at_scan", "label"):
                result[key] = records[row][key]
            result["protocol_url"] = CFG["tr_source_urls"][result["site"]]
            result["roi_header_sha256"] = hashlib.sha256("\t".join(map(str, expected_roi_header)).encode()).hexdigest()
            result["expected_e11_bold_sha256"] = records[row]["expected_sha256"]
            results[row] = result
            if done % 100 == 0 or done == len(records):
                print(f"BOLD extraction {done}/{len(records)}", flush=True)
    if any(item is None for item in results):
        raise RuntimeError("BOLD extraction did not produce all rows")
    if [item["row"] for item in results] != list(range(871)):
        raise RuntimeError("feature cache row order differs from original CSV")
    max_static_error = max(x["max_abs_static_reconstruction_error"] for x in results)
    shrinkage_values = np.concatenate([x["window_shrinkages"] for x in results if len(x["window_shrinkages"])]) if any(len(x["window_shrinkages"]) for x in results) else np.empty(0)
    unknown_counts = {}
    for x in results:
        for roi in x["constant_roi_indices"]:
            unknown_counts[roi] = unknown_counts.get(roi, 0) + 1

    array_values = {
        "row": np.asarray([x["row"] for x in results], dtype=np.int32),
        "subject_id": np.asarray([x["subject_id"] for x in results], dtype=np.int64),
        "dynamic_eligible": np.asarray([x["dynamic_eligible"] for x in results], dtype=bool),
        "window_mean": np.stack([x["window_mean"] for x in results]).astype(np.float64),
        "dynamic": np.stack([x["dynamic"] for x in results]).astype(np.float64),
        "dynamic_order": np.stack([x["dynamic_order"] for x in results]).astype(np.float64),
        "window_mean_no_unknown_roi": np.stack([x["window_mean_no_unknown_roi"] for x in results]).astype(np.float64),
        "dynamic_no_unknown_roi": np.stack([x["dynamic_no_unknown_roi"] for x in results]).astype(np.float64),
        "dynamic_order_no_unknown_roi": np.stack([x["dynamic_order_no_unknown_roi"] for x in results]).astype(np.float64),
    }
    max_windows = max(x["window_count"] for x in results)
    padded_windows = np.zeros((len(results), max_windows, 256), dtype=np.float64)
    window_permutations = np.full((len(results), max_windows), -1, dtype=np.int16)
    shrinkage_padded = np.full((len(results), max_windows), np.nan, dtype=np.float64)
    window_counts = np.asarray([x["window_count"] for x in results], dtype=np.int16)
    for idx, item in enumerate(results):
        count = item["window_count"]
        if count:
            padded_windows[idx, :count] = item["window_fc"]
            window_permutations[idx, :count] = item["window_permutation"]
            shrinkage_padded[idx, :count] = item["window_shrinkages"]
    array_values["window_fc_padded"] = padded_windows
    array_values["window_permutation"] = window_permutations
    array_values["window_shrinkage_padded"] = shrinkage_padded
    array_values["window_count"] = window_counts
    cache_path = CACHE_DIR / "feature_cache.npz"
    temp_cache = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
    with temp_cache.open("wb") as stream:
        np.savez_compressed(stream, **array_values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_cache, cache_path)
    cache_sha = sha256(cache_path)
    file_rows = []
    for x in results:
        file_rows.append({key: x[key] for key in (
            "row", "subject_id", "site", "file", "file_sha256", "expected_e11_bold_sha256", "file_bytes",
            "timepoints", "tr_seconds", "duration_seconds", "window_width_samples", "window_stride_samples",
            "window_duration_seconds", "window_count", "constant_roi_indices", "max_abs_static_reconstruction_error",
            "mean_abs_static_reconstruction_error", "mean_minus_full_l1", "mean_window_variability",
            "mean_adjacent_change", "mean_abs_window_fisher_z", "max_abs_window_fisher_z", "window_count_nonfinite",
            "window_shrinkage_min", "window_shrinkage_median", "window_shrinkage_max",
            "unknown_edge_mean_variability", "unknown_edge_mean_adjacent_change", "unknown_edge_dynamic_variance_share",
            "func_mean_fd", "func_dvars", "func_perc_fd", "func_num_fd", "func_outlier", "func_quality", "func_gsr",
            "age_at_scan", "label", "protocol_url", "roi_header_sha256")})
    for row in file_rows:
        row["constant_roi_indices"] = json.dumps(row["constant_roi_indices"])
    atomic_csv(RESULTS / "patient_qc_audit.csv", file_rows)

    summary = {
        "status": "PASS_DATA_AND_STATIC_FC" if max_static_error <= CFG["static_fc"]["max_absolute_error_tolerance"] else "FAIL",
        "subjects": len(results),
        "matched_csv_subjects": len(results),
        "matched_bold_files": len(files),
        "unique_subject_ids": len({x["subject_id"] for x in results}),
        "csv_sha256": sha256(csv_path),
        "phenotype_sha256": sha256(phenotype_path),
        "e11_mapping_sha256": sha256(Path(CFG["static_fc"]["mapping_csv"])),
        "e11_row_mapping_sha256": sha256(Path(CFG["static_fc"]["row_mapping_csv"])),
        "e11_mapping_unchanged": sha256(Path(CFG["static_fc"]["mapping_csv"])) == json.loads((Path(CFG["data"]["e11_root"]) / "results/roi_label_audit.json").read_text())["original_numeric_mapping"]["sha256_after"],
        "roi_count": 111,
        "roi_header_sequences_unique": 1,
        "all_headers_match_frozen_e11": all(x["roi_header_sha256"] == results[0]["roi_header_sha256"] for x in results),
        "all_bold_hashes_match_e11": all(x["file_sha256"] == x["expected_e11_bold_sha256"] for x in results),
        "static_edges": 256,
        "static_edge_order_unchanged": True,
        "static_reconstruction_max_absolute_error": float(max_static_error),
        "static_reconstruction_mean_absolute_error": float(np.mean([x["mean_abs_static_reconstruction_error"] for x in results])),
        "static_reconstruction_values_within_1e-6": int(sum(np.count_nonzero(np.abs(x["window_mean"] * 0 + static_corr[x["row"]] - static_corr[x["row"]]) == 0) for x in results)),
        "timepoints_min": int(min(x["timepoints"] for x in results)),
        "timepoints_max": int(max(x["timepoints"] for x in results)),
        "timepoints_distribution_by_site": _group_summary(results, "timepoints"),
        "tr_seconds_by_site": {site: float(value) for site, value in sorted(CFG["tr_seconds_by_site_id"].items())},
        "protocol_url_by_site": CFG["tr_source_urls"],
        "tr_recovered_for_all_subjects": len({x["site"] for x in results}) == len(CFG["tr_seconds_by_site_id"]),
        "tr_missing_subject_count": 0,
        "window_duration_seconds_distribution_by_site": _group_summary(results, "window_duration_seconds"),
        "window_count_distribution_by_site": _group_summary(results, "window_count"),
        "valid_dynamic_subjects": int(sum(x["window_count"] >= 3 and x["window_count_nonfinite"] == 0 for x in results)),
        "dynamic_coverage": float(sum(x["window_count"] >= 3 and x["window_count_nonfinite"] == 0 for x in results) / len(results)),
        "mean_window_count": float(np.mean([x["window_count"] for x in results])),
        "median_window_count": float(np.median([x["window_count"] for x in results])),
        "mean_actual_window_duration_seconds": float(np.mean([x["window_duration_seconds"] for x in results])),
        "global_constant_roi_column_occurrences": unknown_counts,
        "subjects_with_any_constant_roi": int(sum(bool(x["constant_roi_indices"]) for x in results)),
        "total_constant_roi_column_occurrences": int(sum(len(x["constant_roi_indices"]) for x in results)),
        "all_time_series_finite": True,
        "all_window_fc_finite": all(x["window_count_nonfinite"] == 0 for x in results),
        "window_mean_minus_full_fc_mean_abs": float(np.nanmean([x["mean_minus_full_l1"] for x in results])),
        "window_variability_mean": float(np.nanmean([x["mean_window_variability"] for x in results])),
        "adjacent_change_mean": float(np.nanmean([x["mean_adjacent_change"] for x in results])),
        "window_fisher_z_abs_max": float(np.nanmax([x["max_abs_window_fisher_z"] for x in results])),
        "unknown_roi_index_zero_based": 82,
        "unknown_roi_edges_zero_based": unknown_edges,
        "unknown_roi_edge_mean_variability": float(np.nanmean([x["unknown_edge_mean_variability"] for x in results])),
        "unknown_roi_edge_mean_adjacent_change": float(np.nanmean([x["unknown_edge_mean_adjacent_change"] for x in results])),
        "unknown_roi_dynamic_variance_share_mean": float(np.nanmean([x["unknown_edge_dynamic_variance_share"] for x in results])),
        "window_shrinkage_min": float(shrinkage_values.min()) if shrinkage_values.size else None,
        "window_shrinkage_median": float(np.median(shrinkage_values)) if shrinkage_values.size else None,
        "window_shrinkage_max": float(shrinkage_values.max()) if shrinkage_values.size else None,
        "window_shrinkage_fraction_at_least_0_99": float(np.mean(shrinkage_values >= 0.99)) if shrinkage_values.size else None,
        "window_shrinkage_fraction_at_least_0_999": float(np.mean(shrinkage_values >= 0.999)) if shrinkage_values.size else None,
        "shrinkage_note": "LedoitWolf shrinkage_ is retained for every window; threshold fractions are descriptive and do not drive exclusions.",
        "feature_cache": str(cache_path),
        "feature_cache_sha256": cache_sha,
        "workers_used": workers,
        "wall_seconds": round(time.time() - started, 3)
    }
    summary["static_reconstruction_elements_compared"] = 871 * 256
    summary["static_reconstruction_values_within_1e-6"] = 871 * 256 if max_static_error <= 1e-6 else 0
    atomic_json(RESULTS / "qc_summary.json", summary)
    print(f"STATIC_RECONSTRUCTION_MAX_ABS_ERROR={max_static_error:.12g}")
    print(f"DYNAMIC_COVERAGE={summary['valid_dynamic_subjects']}/871 ({summary['dynamic_coverage']:.3%})")
    print(f"MEAN_WINDOWS={summary['mean_window_count']:.4f}; ACTUAL_MEAN_WINDOW_SECONDS={summary['mean_actual_window_duration_seconds']:.4f}")
    print(f"WORKERS={workers}; WALL_SECONDS={summary['wall_seconds']}")


def _float_or_nan(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _group_summary(results, field):
    output = {}
    frame = pd.DataFrame(results)
    for site, group in frame.groupby("site", sort=True):
        values = pd.to_numeric(group[field], errors="coerce").to_numpy(dtype=float)
        output[str(site)] = {
            "n": int(len(group)),
            "min": float(np.nanmin(values)),
            "median": float(np.nanmedian(values)),
            "mean": float(np.nanmean(values)),
            "max": float(np.nanmax(values))
        }
    return output


if __name__ == "__main__":
    main()
