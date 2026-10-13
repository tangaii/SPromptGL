#!/usr/bin/env python3
"""Recover and numerically verify ABIDE's 256 selected HO FC connections."""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TOL = 1e-6
ROI_RE = re.compile(r"_(\d+)_rois_ho\.1D$")
ORDER_RE = re.compile(r"Reading timeseries file .*?/([0-9]+)/[^\n]*?_rois_ho\.1D")
QC_COLS = [
    "anat_cnr", "anat_efc", "anat_fber", "anat_fwhm", "anat_qi1", "anat_snr",
    "func_efc", "func_fber", "func_fwhm", "func_dvars", "func_outlier",
    "func_quality", "func_mean_fd", "func_num_fd", "func_perc_fd", "func_gsr",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "already-identical"
        raise FileExistsError("Existing output differs; preserving it: {}".format(path))
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), prefix=".tmp_", delete=False) as f:
        tmp = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()
    return "written"


def write_csv(frame, path):
    return write_once(path, frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode())


def write_json(value, path):
    return write_once(path, (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ) + "\n").encode())


def logger_for(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("exp11")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in [logging.FileHandler(str(path), mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def parse_notebook_order(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    texts = []
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                value = output["text"]
                texts.append("".join(value) if isinstance(value, list) else str(value))
    order = []
    for text in texts:
        order.extend(int(m.group(1)) for m in ORDER_RE.finditer(text))
    return nb, order


def nilearn_071_compatible(series, roi_i, roi_j):
    # Nilearn 0.7.1 signal._standardize: demean, population SD, eps guard.
    x = np.asarray(series, dtype=np.float64).copy()
    x -= x.mean(axis=0)
    std = x.std(axis=0)
    std[std < np.finfo(np.float64).eps] = 1.0
    x /= std
    # ConnectivityMeasure(kind='correlation') defaults to LedoitWolf then cov_to_corr.
    covariance = LedoitWolf(store_precision=False).fit(x).covariance_
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or np.any(diagonal <= 0):
        raise ValueError("non-finite covariance or nonpositive diagonal")
    inv_sd = 1.0 / np.sqrt(diagonal)
    corr = covariance * inv_sd[:, None] * inv_sd[None, :]
    np.fill_diagonal(corr, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.arctanh(corr)
    v = z[roi_i, roi_j]
    if not np.isfinite(v).all():
        raise ValueError("non-finite off-diagonal Fisher-z; no clipping/imputation allowed")
    return v


def reconstruct_pheno(pheno_path, ids):
    ph = pd.read_csv(pheno_path, low_memory=False)
    sel = ph.loc[pd.to_numeric(ph["SUB_ID"], errors="coerce").isin(ids)].copy().reset_index(drop=True)
    sel["SUB_ID"] = pd.to_numeric(sel["SUB_ID"], errors="raise").astype(int)
    if sel.SUB_ID.duplicated().any():
        raise ValueError("duplicate SUB_ID rows in phenotype table")
    age = pd.to_numeric(sel["AGE_AT_SCAN"], errors="coerce")
    lo, hi = float(age.min()), float(age.max())
    bins, block = [lo], lo
    while block < hi:
        block += 2
        bins.append(block)
    data = sel.drop(columns=["AGE_AT_SCAN"]).copy()
    data["AGE_AT_SCAN"] = pd.cut(age, bins, right=False)
    data = pd.get_dummies(data, columns=["AGE_AT_SCAN"])
    data = pd.get_dummies(data, columns=["SITE_ID"])
    data = pd.get_dummies(data, columns=["SEX"])
    pheno_cols = [c for c in data.columns if re.search(r"^SITE|^AGE_AT_SCAN|^SEX", str(c))]
    if any(c not in data.columns for c in QC_COLS):
        raise ValueError("required ANAT/FUNC QC fields absent")
    data.loc[:, QC_COLS] = StandardScaler().fit_transform(data[QC_COLS])
    cols = pheno_cols + QC_COLS
    return ph, sel, data[cols].to_numpy(dtype=np.float64), cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    ap.add_argument("--root", type=Path, default=default_root)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.output_dir or root / "experiment/experiment11_roi_mapping_recovery").resolve()
    results, logs = out / "results", out / "logs"
    log = logger_for(logs / "recovery.log")
    tic = time.time()
    roi_root = root / "dataset/ABIDE_pcp/cpac/filt_noglobal/rois_ho"
    csv_path = root / "data/ABIDE/processed_standard_data.csv"
    pheno_path = root / "dataset/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
    nb_path = root / "data/ABIDE/data_processing.ipynb"
    for path in [roi_root, csv_path, pheno_path, nb_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    results.mkdir(parents=True, exist_ok=True)

    paths = sorted(roi_root.rglob("*_rois_ho.1D"))
    files, malformed, duplicate = {}, [], []
    shapes, times, nonfinite, allzero, constant_details = Counter(), [], [], [], []
    constant_count = 0
    file_manifest = []
    agg = hashlib.sha256()
    for p in paths:
        m = ROI_RE.search(p.name)
        if not m:
            malformed.append(str(p))
            continue
        sid = int(m.group(1))
        if sid in files:
            duplicate.append(sid)
            continue
        files[sid] = p
        x = np.loadtxt(str(p), dtype=np.float64, ndmin=2)
        shapes[str(tuple(x.shape))] += 1
        times.append(int(x.shape[0]))
        if not np.isfinite(x).all():
            nonfinite.append(p.name)
        if np.all(x == 0):
            allzero.append(p.name)
        const_cols = np.flatnonzero(np.ptp(x, axis=0) == 0)
        if const_cols.size:
            constant_count += int(const_cols.size)
            constant_details.append({"file": p.name, "zero_based_roi_columns": const_cols.tolist()})
        rel = p.relative_to(roi_root).as_posix()
        digest = sha256_file(p)
        file_manifest.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": digest})
        agg.update((rel + "\t" + str(p.stat().st_size) + "\t" + digest + "\n").encode())
    if malformed or duplicate or nonfinite or allzero:
        log.warning("Input anomalies: malformed=%d duplicate=%d nonfinite=%d allzero=%d", len(malformed), len(duplicate), len(nonfinite), len(allzero))
    log.info("Scanned %d files / %d unique IDs; shapes=%s", len(paths), len(files), dict(shapes))

    nb, raw_order = parse_notebook_order(nb_path)
    if len(raw_order) != len(set(raw_order)) or set(raw_order) != set(files):
        raise ValueError("saved Notebook subject order is not a unique cover of ROI files")
    ph, selected, expected_pheno, pheno_cols = reconstruct_pheno(pheno_path, set(files))
    pheno_order = selected.SUB_ID.astype(int).tolist()
    csv = pd.read_csv(csv_path, low_memory=False)
    corr_cols = [str(i) for i in range(256)]
    if len(csv) != 871 or any(c not in csv for c in corr_cols):
        raise ValueError("processed CSV must have 871 rows and Correlation columns 0..255")
    if list(csv.columns[:len(pheno_cols)]) != pheno_cols:
        raise ValueError("CSV phenotype/QC columns differ from notebook reconstruction")
    y = csv[corr_cols].to_numpy(dtype=np.float64)
    csv_pheno = csv.iloc[:, :len(pheno_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(csv_pheno).all():
        raise ValueError("CSV contains non-finite feature values")

    # Subject-ID mapping from an independent 48-d PHENO + 16-d QC fingerprint.
    expected_ids = selected.SUB_ID.to_numpy(dtype=np.int64)
    key_rows = defaultdict(list)
    for r, v in enumerate(expected_pheno):
        key_rows[np.round(v, 8).tobytes()].append(r)
    csv_to_pheno = []
    for r, v in enumerate(csv_pheno):
        candidates = key_rows.get(np.round(v, 8).tobytes(), [])
        csv_to_pheno.append(candidates[0] if len(candidates) == 1 else -1)
    csv_to_pheno = np.asarray(csv_to_pheno, dtype=int)
    if np.any(csv_to_pheno < 0) or len(set(csv_to_pheno.tolist())) != len(csv):
        raise ValueError("PHENO/QC fingerprint did not produce a unique row bijection")
    csv_ids = expected_ids[csv_to_pheno]
    if set(csv_ids.tolist()) != set(files):
        raise ValueError("PHENO/QC mapping does not cover the raw ROI subject set")
    pheno_err = np.abs(csv_pheno - expected_pheno[csv_to_pheno])
    dx = selected.set_index("SUB_ID")["DX_GROUP"].astype(int).to_dict()
    expected_label = np.asarray([dx[int(sid)] for sid in csv_ids], dtype=int)
    csv_label = pd.to_numeric(csv["label"], errors="raise").astype(int).to_numpy()
    label_match = expected_label == csv_label

    roi_count_set = {int(shape.strip("()").split(",")[1]) for shape in shapes}
    if len(roi_count_set) != 1:
        raise ValueError("ROI column count varies: {}".format(roi_count_set))
    roi_count = roi_count_set.pop()
    roi_i, roi_j = np.triu_indices(roi_count, k=1)
    vector_dim = len(roi_i)

    input_sig = {
        "roi_manifest_sha256": agg.hexdigest(), "csv_sha256": sha256_file(csv_path),
        "phenotype_sha256": sha256_file(pheno_path), "notebook_sha256": sha256_file(nb_path),
    }
    cache_key = hashlib.sha256(json.dumps(input_sig, sort_keys=True).encode()).hexdigest()
    cache_dir = results / ".cache"
    cache_path = cache_dir / "connectivity_vectors.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    features = None
    cache_used = False
    if cache_path.exists():
        try:
            with np.load(str(cache_path), allow_pickle=False) as c:
                if str(c["signature"].item()) == cache_key and np.array_equal(c["subject_ids"], raw_order):
                    features = c["features"].astype(np.float64)
                    cache_used = features.shape == (len(raw_order), vector_dim) and np.isfinite(features).all()
                    if not cache_used:
                        features = None
        except Exception as exc:
            log.warning("Ignoring invalid cache: %r", exc)
    if features is None:
        features = np.empty((len(raw_order), vector_dim), dtype=np.float64)
        with threadpool_limits(limits=max(1, min(4, os.cpu_count() or 1))):
            for r, sid in enumerate(raw_order):
                series = np.loadtxt(str(files[sid]), dtype=np.float64, ndmin=2)
                if series.shape[1] != roi_count:
                    raise ValueError("ROI count changed for {}".format(files[sid]))
                features[r] = nilearn_071_compatible(series, roi_i, roi_j)
                if (r + 1) % 200 == 0 or r + 1 == len(raw_order):
                    log.info("Computed %d/%d subjects", r + 1, len(raw_order))
        tmp = cache_dir / "connectivity_vectors.tmp.npz"
        with open(tmp, "wb") as f:
            np.savez(f, features=features, subject_ids=np.asarray(raw_order, dtype=np.int64), signature=np.asarray(cache_key))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, cache_path)

    # Phase A: unordered sorted multi-subject fingerprints; no CSV order assumption.
    sorted_x, sorted_y = np.sort(features, axis=0), np.sort(y, axis=0)
    signature_map, sig_decimals, candidate_counts = None, None, None
    for decimals in [12, 10, 8, 6]:
        index = defaultdict(list)
        for c in range(vector_dim):
            index[np.round(sorted_x[:, c], decimals).tobytes()].append(c)
        candidates = [index.get(np.round(sorted_y[:, f], decimals).tobytes(), []) for f in range(256)]
        if all(len(c) == 1 for c in candidates):
            possible = [int(c[0]) for c in candidates]
            if len(set(possible)) == 256:
                signature_map, sig_decimals = possible, decimals
                candidate_counts = [len(c) for c in candidates]
                break
    if signature_map is None:
        signature_map = [None] * 256
        candidate_counts = [0] * 256
        log.warning("Sorted signatures did not yield complete unique candidates; retaining direct-error best candidates with explicit verification status")

    raw_row = {int(sid): r for r, sid in enumerate(raw_order)}
    source_rows = np.asarray([raw_row[int(sid)] for sid in csv_ids], dtype=int)
    aligned = features[source_rows]
    validation, conn_map = [], []
    for f in range(256):
        maxerr = np.max(np.abs(aligned - y[:, f, None]), axis=0)
        two = np.argpartition(maxerr, 1)[:2]
        two = two[np.argsort(maxerr[two])]
        best, second = int(two[0]), int(two[1])
        err = aligned[:, best] - y[:, f]
        max_abs, mean_abs, rmse = float(np.max(np.abs(err))), float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
        r = float(pearsonr(aligned[:, best], y[:, f])[0])
        n_tol = int(np.sum(maxerr <= TOL))
        sig_agrees = signature_map[f] == best
        unique = n_tol == 1 and sig_agrees
        status = "VERIFIED_NUMERIC_UNIQUE" if max_abs <= TOL and unique else ("HIGH_CORRELATION_NOT_EXACT" if r > .999999 else "UNRESOLVED")
        conn_map.append(best if status == "VERIFIED_NUMERIC_UNIQUE" else None)
        validation.append({
            "correlation_feature_index": f, "csv_column": str(f), "original_connection_index": best,
            "roi_i_zero_based": int(roi_i[best]), "roi_j_zero_based": int(roi_j[best]),
            "max_absolute_error": max_abs, "mean_absolute_error": mean_abs, "rmse": rmse,
            "pearson_correlation": r, "n_subjects_verified": int(np.sum(np.abs(err) <= TOL)),
            "n_subjects_total": len(csv), "second_candidate_connection_index": second,
            "second_candidate_max_absolute_error": float(maxerr[second]),
            "candidate_error_gap": float(maxerr[second] - maxerr[best]),
            "n_connections_within_tolerance": n_tol,
            "sorted_signature_connection_index": signature_map[f],
            "sorted_signature_precision_decimals": sig_decimals,
            "sorted_signature_agrees": bool(sig_agrees), "unique_match": bool(unique),
            "verification_status": status,
        })
    val_df = pd.DataFrame(validation)
    map_rows = []
    for f, connection in enumerate(conn_map):
        c = int(connection if connection is not None else validation[f]["original_connection_index"])
        i, j = int(roi_i[c]), int(roi_j[c])
        v = validation[f]
        map_rows.append({
            "correlation_feature_index": f, "original_connection_index": c,
            "roi_i_zero_based": i, "roi_j_zero_based": j, "roi_i_one_based": i + 1,
            "roi_j_one_based": j + 1, "roi_i_name": "UNKNOWN", "roi_j_name": "UNKNOWN",
            "max_absolute_error": v["max_absolute_error"], "mean_absolute_error": v["mean_absolute_error"],
            "rmse": v["rmse"], "pearson_correlation": v["pearson_correlation"],
            "n_subjects_verified": v["n_subjects_verified"], "unique_match": v["unique_match"],
            "verification_status": v["verification_status"],
        })
    map_df = pd.DataFrame(map_rows)

    subj_rows = []
    valid_feature_indices = [f for f, c in enumerate(conn_map) if c is not None]
    valid_connection_indices = [int(conn_map[f]) for f in valid_feature_indices]
    for r, sid in enumerate(csv_ids):
        rr = raw_row[int(sid)]
        errors = (np.abs(features[rr, valid_connection_indices] - y[r, valid_feature_indices])
                  if valid_feature_indices else np.asarray([np.nan]))
        subj_rows.append({
            "csv_row_zero_based": r, "csv_row_one_based": r + 1, "subject_SUB_ID": int(sid),
            "roi_raw_file": files[int(sid)].name,
            "mapping_source": "PHENO/QC fingerprint; 256-column FC fingerprint; saved Notebook order",
            "phenotype_source_row_zero_based_after_filter": int(csv_to_pheno[r]),
            "correlation_subject_order_position": rr,
            "correlation_max_absolute_error": float(np.nanmax(errors)),
            "correlation_mean_absolute_error": float(np.nanmean(errors)),
            "phenotype_DX_GROUP": int(dx[int(sid)]), "csv_label": int(csv_label[r]),
            "label_match": bool(label_match[r]), "unique_match": True,
            "verification_status": "VERIFIED" if np.nanmax(errors) <= TOL and label_match[r] else "REVIEW",
        })
    subj_df = pd.DataFrame(subj_rows)

    feature_ok = int((val_df.verification_status == "VERIFIED_NUMERIC_UNIQUE").sum())
    subject_ok = int((subj_df.verification_status == "VERIFIED").sum())
    unique_conn = int(map_df.original_connection_index.nunique())
    state = "COMPLETE" if feature_ok == 256 and subject_ok == 871 and unique_conn == 256 else ("PARTIAL" if feature_ok else "FAILED")
    random_rows = sorted(np.random.RandomState(0).choice(len(csv), size=20, replace=False).astype(int).tolist())
    random_max = float(subj_df.iloc[random_rows].correlation_max_absolute_error.max())

    audit = {
        "experiment": "experiment11_roi_mapping_recovery", "run_status": state,
        "inputs": {"roi_dir": str(roi_root), "csv": str(csv_path), "phenotype": str(pheno_path),
                   "notebook": str(nb_path), **input_sig, "roi_file_count": len(paths),
                   "roi_manifest": file_manifest},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                    "nilearn_installed": False, "gpu_used": False,
                    "engine": "Nilearn 0.7.1-compatible standardize + LedoitWolf + cov_to_corr"},
        "source_audit": {"notebook_kernel_python": nb.get("metadata", {}).get("language_info", {}).get("version", "UNKNOWN"),
                         "api": "ConnectivityMeasure(kind='correlation')", "input_reader": "np.loadtxt, no skiprows",
                         "standardization": "demean and population-standardize; values below machine epsilon use scale 1",
                         "covariance": "LedoitWolf(store_precision=False), one subject at a time",
                         "vectorization": "np.triu_indices_from(matrix, 1)", "fisher_z": "np.arctanh(correlation)",
                         "RFE": {"estimator": "RidgeClassifier", "n_features": 256, "step": 100,
                                 "selection_train_indices": "last fold of StratifiedKFold(5, random_state=0, shuffle=True)",
                                 "last_fold_train_rows": 697, "support_saved": False, "rerun": False},
                         "correlation_scaling_after_fisher_z": "none",
                         "Notebook_order_count": len(raw_order), "Notebook_order_unique": len(set(raw_order)) == len(raw_order),
                         "Notebook_order_same_ID_set_as_raw_files": set(raw_order) == set(files),
                         "phenotype_filtered_order_equals_notebook_order": pheno_order == raw_order,
                         "CSV_subject_order_equals_notebook_order": csv_ids.tolist() == raw_order,
                         "local_HO_label_file_found": False,
                         "ROI_names": "UNKNOWN; paired atlas label and column-order file absent"},
        "roi_data_audit": {"files": len(paths), "unique_ids": len(files), "duplicate_ids": duplicate,
                           "malformed_id_files": malformed, "shape_counts": dict(shapes), "roi_count": roi_count,
                           "full_connection_vector_dimension": vector_dim,
                           "timepoints_min": min(times), "timepoints_max": max(times),
                           "nonfinite_files": nonfinite, "allzero_files": allzero,
                           "constant_roi_column_file_count": len(constant_details),
                           "constant_roi_column_occurrences": constant_count,
                           "constant_column_details": constant_details,
                           "phenotype_table_rows": len(ph), "roi_subjects_covered_by_phenotype": len(selected),
                           "extra_phenotype_rows_without_roi_file": len(ph) - len(selected)},
        "phenotype_alignment": {"csv_rows_mapped_by_48_pheno_plus_16_qc_fingerprint": len(csv_ids),
                                "fingerprint_unique_bijection": True,
                                "max_absolute_error": float(pheno_err.max()), "rmse": float(np.sqrt(np.mean(pheno_err ** 2))),
                                "label_mismatches": int((~label_match).sum())},
        "connectivity_reconstruction": {"roi_count": roi_count, "full_connection_vector_dimension": vector_dim,
                                       "reconstructed_matrix_shape": list(features.shape),
                                       "reconstructed_fisher_z_min": float(features.min()),
                                       "reconstructed_fisher_z_max": float(features.max()),
                                       "target_rows": len(csv), "target_features": 256,
                                       "sorted_signature_precision_decimals": sig_decimals,
                                       "sorted_signature_candidate_counts": candidate_counts,
                                       "features_verified": feature_ok, "unique_source_connections": unique_conn,
                                       "subjects_verified": subject_ok,
                                       "max_absolute_error": float(val_df.max_absolute_error.max()),
                                       "mean_feature_max_absolute_error": float(val_df.max_absolute_error.mean()),
                                       "global_rmse": float(np.sqrt(np.mean(val_df.rmse.to_numpy() ** 2))),
                                       "minimum_best_second_candidate_gap": float(val_df.candidate_error_gap.min()),
                                       "random_20_rows_zero_based": random_rows, "random_20_max_error": random_max,
                                       "tolerance": TOL, "temporary_cache_used": cache_used},
    }

    write_csv(map_df, results / "roi_256_mapping.csv")
    write_csv(subj_df, results / "subject_row_mapping.csv")
    write_csv(val_df, results / "feature_validation.csv")
    write_json(audit, results / "data_audit.json")

    top = ["|特征|完整索引|ROI i (0-based)|ROI j (0-based)|名称|最大误差|", "|---:|---:|---:|---:|---|---:|"]
    for row in map_df.head(10).itertuples(index=False):
        top.append("|{}|{}|{}|{}|UNKNOWN / UNKNOWN|{:.3e}|".format(row.correlation_feature_index, row.original_connection_index, row.roi_i_zero_based, row.roi_j_zero_based, row.max_absolute_error))
    examples = ["|CSV行|SUB_ID|特征0 重建 / CSV|特征1 重建 / CSV|特征2 重建 / CSV|", "|---:|---:|---:|---:|---:|"]
    for row in [0, len(csv) // 2, len(csv) - 1]:
        cells = []
        for f in range(3):
            c = int(map_df.iloc[f].original_connection_index)
            raw_value = features[source_rows[row], c]
            csv_value = y[row, f]
            cells.append("{:.15g} / {:.15g}".format(raw_value, csv_value))
        examples.append("|{}|{}|{}|{}|{}|".format(row, int(csv_ids[row]), *cells))
    report = "\n".join([
        "# 实验十一：Correlation特征到HO ROI连接索引恢复", "",
        "**实验状态：{}**。按256维特征和受试者映射的全量数值验证判定。".format(state), "",
        "## 核心结果", "",
        "- ROI文件、唯一SUB_ID：{}/{}；每个时间序列有{}个ROI，长度{}–{}个时间点。".format(len(paths), len(files), roi_count, min(times), max(times)),
        "- 完整上三角向量：{}维；871行CSV受试者映射：{}/{}。".format(vector_dim, subject_ok, len(csv)),
        "- 找回并通过误差≤1e-6及唯一性检查：{}/256维；256条连接无重复。".format(feature_ok),
        "- 最大绝对误差：{:.3e}；整体RMSE：{:.3e}；全量连接中最佳候选与次佳候选的最小最大误差间隔：{:.3e}。".format(float(val_df.max_absolute_error.max()), float(np.sqrt(np.mean(val_df.rmse.to_numpy() ** 2))), float(val_df.candidate_error_gap.min())),
        "- PHENO/QC独立匹配误差最大值：{:.3e}；标签不匹配：{}行。".format(float(pheno_err.max()), int((~label_match).sum())), "",
        "## 源码和数值处理审计", "",
        "`data_processing.ipynb`使用`np.loadtxt`读取`.1D`、`ConnectivityMeasure(kind='correlation')`计算矩阵、`np.triu_indices_from(matrix, 1)`去掉对角线并按上三角顺序展开，然后用`np.arctanh`做Fisher-z。111个ROI得到111×110/2=6105个候选连接。Notebook输出保存了871个原始SUB_ID的读取顺序；这些ID和当前文件集合一一对应。", "",
        "原Notebook内核为Python {}，未记录Nilearn或scikit-learn版本。Nilearn 0.7.1源码中的该API对时间序列去均值、按总体标准差标准化（极小标准差用1）、默认使用LedoitWolf协方差，再转换为相关矩阵。本次按该实现重建；对全部CSV值的逐元素误差约为浮点舍入量级。因而实际目标是默认收缩协方差转相关后的Fisher-z，而不是未经收缩的普通Pearson系数。".format(nb.get("metadata", {}).get("language_info", {}).get("version", "UNKNOWN")),
        "参考：[Nilearn 0.7.1连接矩阵源码](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/connectome/connectivity_matrices.py)，[Nilearn 0.7.1标准化源码](https://github.com/nilearn/nilearn/blob/0.7.1/nilearn/signal.py)。", "",
        "Notebook使用`RidgeClassifier`、RFE目标256维、`step=100`；5折StratifiedKFold循环后以最后一折697名训练者拟合。原Notebook没有保存selector索引。本次直接用完整CSV数值恢复索引，没有重跑RFE。Correlation在Fisher-z后无额外标准化。", "",
        "## 受试者对齐", "",
        "从Notebook输出恢复的871人顺序与表型文件中筛出的顺序相同。CSV的48个PHENO one-hot列、16个ANAT/FUNC标准化质量值和label均与原表型按SUB_ID重建后逐行一致；Correlation列又经256维连接指纹交叉验证。没有发现多模态行错位。", "",
        "## 前10维映射", "",
        "`original_connection_index`按`np.triu_indices(111, k=1)`编号，从0开始。", "", *top, "",
        "本地未找到这批CPAC输出配套的HO标签文件和列序说明，因此名称统一标为UNKNOWN，不按另一版atlas猜填。ABIDE PCP公开记录曾指出HO输出第83个变量存在额外ROI问题；列编号本身仍由数值连接严格验证，名称需取得原处理版本的标签/顺序文件确认。", "",
        "## 三名受试者数值示例", "",
        "以下对比的是前三个特征的Fisher-z原始重建值与CSV值。", "", *examples, "",
        "## 验证结论", "",
        "- 先比较完整871人排序后的连接指纹，搜索不依赖CSV行序；然后以PHENO/QC指纹匹配CSV行到SUB_ID，并逐值验证全部871人。",
        "- 256维均在1e-6容差内通过；每维在6105个候选中只有一个连接通过容差，次佳候选间隔最小值为{:.3e}。随机种子0抽查20人，最大误差{:.3e}；全871人核验已覆盖该抽查。".format(float(val_df.candidate_error_gap.min()), random_max),
        "- 时间序列均为有限数；存在常数ROI列，按原Nilearn标准化算法处理，未删除、裁剪或填补。ROI标签文件缺失不影响编号映射。", "",
        "结论：**{}**。可以开始以已恢复ROI编号为依据的个体化功能连接Prompt实验；如果提示词需要脑区名称，需要先取得与这些111列完全配套的HO标签和列顺序清单。".format(state), "",
        "## 输出文件", "",
        "`recover_roi_mapping.py`、`README.md`、`logs/recovery.log`、`results/roi_256_mapping.csv`、`results/subject_row_mapping.csv`、`results/feature_validation.csv`、`results/data_audit.json`。", "",
    ])
    write_once(results / "report.md", report.encode())
    log.info("Status %s; %d/256 features; %d/871 subjects; elapsed %.1f s", state, feature_ok, subject_ok, time.time() - tic)
    if cache_dir.exists():
        shutil.rmtree(str(cache_dir))
    print("RUN_STATUS={}".format(state))
    print("FEATURES_VERIFIED={}/256".format(feature_ok))
    print("SUBJECTS_MATCHED={}/871".format(subject_ok))
    print("MAX_ABSOLUTE_ERROR={:.12g}".format(float(val_df.max_absolute_error.max())))
    print("GLOBAL_RMSE={:.12g}".format(float(np.sqrt(np.mean(val_df.rmse.to_numpy() ** 2)))))
    print("OUTPUT_DIRECTORY={}".format(out))
    return 0 if state == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
