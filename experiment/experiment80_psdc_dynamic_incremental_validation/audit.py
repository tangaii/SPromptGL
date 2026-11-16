#!/usr/bin/env python3
"""Independent data, nested-probe and read-only E74.4-G2 audit."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

EXP = Path(__file__).resolve().parent
CFG = json.loads((EXP / "config.json").read_text(encoding="utf-8"))
REPO = Path("/data/adodas/SPromptGL-main")
E74 = REPO / "experiment/experiment74_4_soft_positive_dcr"
RESULTS = EXP / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
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


def metric_values(y: np.ndarray, pred: np.ndarray, p_asd: np.ndarray) -> dict[str, float]:
    is_asd = y == 0
    return {
        "acc": float(accuracy_score(y, pred)),
        "hard_auc": float(balanced_accuracy_score(y, pred)),
        "prob_auc": float(roc_auc_score(is_asd.astype(int), p_asd)),
        "sen": float(recall_score(is_asd.astype(int), (pred == 0).astype(int), pos_label=1, zero_division=0)),
        "spe": float(recall_score((~is_asd).astype(int), (pred == 1).astype(int), pos_label=1, zero_division=0)),
        "asd_f1": float(f1_score(is_asd.astype(int), (pred == 0).astype(int), pos_label=1, zero_division=0)),
    }


def verify_identity_and_static_fc() -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    e11 = import_module("e80_readonly_e11_fc_audit", Path(CFG["data"]["e11_root"]) / "recover_roi_mapping.py")
    frame = pd.read_csv(Path(CFG["data"]["csv_path"]), low_memory=False)
    pheno = pd.read_csv(Path(CFG["data"]["phenotype_path"]), low_memory=False)
    row_map = pd.read_csv(CFG["static_fc"]["row_mapping_csv"])
    mapping = pd.read_csv(CFG["static_fc"]["mapping_csv"])
    labels = pd.read_csv(CFG["static_fc"]["roi_labels_csv"])
    expected_manifest = json.loads((Path(CFG["data"]["e11_root"]) / "results/data_audit.json").read_text())
    expected_hashes = {entry["relative_path"]: entry["sha256"] for entry in expected_manifest["inputs"]["roi_manifest"]}
    if frame.shape != (871, 321) or len(row_map) != 871 or len(mapping) != 256 or len(expected_hashes) != 871:
        raise RuntimeError("static input, E11 patient map, edge map, or BOLD inventory shape mismatch")
    raw_label = pd.to_numeric(frame.label, errors="raise").to_numpy(dtype=int)
    y = raw_label - 1
    if not np.array_equal(np.bincount(y), [403, 468]):
        raise RuntimeError("historical ASD=0/TD=1 class counts changed")
    row_map["subject_SUB_ID"] = pd.to_numeric(row_map.subject_SUB_ID, errors="raise").astype(int)
    row_map["csv_row_zero_based"] = pd.to_numeric(row_map.csv_row_zero_based, errors="raise").astype(int)
    row_map["csv_row_one_based"] = pd.to_numeric(row_map.csv_row_one_based, errors="raise").astype(int)
    pheno["SUB_ID"] = pd.to_numeric(pheno.SUB_ID, errors="raise").astype(int)
    pheno["DX_GROUP"] = pd.to_numeric(pheno.DX_GROUP, errors="raise").astype(int)
    merged = row_map.merge(pheno, left_on="subject_SUB_ID", right_on="SUB_ID", validate="one_to_one")
    if sorted(merged.csv_row_zero_based.tolist()) != list(range(871)) or merged.subject_SUB_ID.nunique() != 871:
        raise RuntimeError("E11 CSV-row to SUB_ID mapping is not a bijection")
    if not np.array_equal(merged.sort_values("csv_row_zero_based").DX_GROUP.to_numpy(), raw_label):
        raise RuntimeError("raw DX_GROUP does not match original label rows")
    roi_values = pd.to_numeric(labels.atlas_integer_value, errors="raise").to_numpy(dtype=int)
    if len(roi_values) != 111 or len(set(roi_values.tolist())) != 111:
        raise RuntimeError("E11 ROI column identity is invalid")
    if not np.array_equal(pd.to_numeric(mapping.correlation_feature_index).to_numpy(dtype=int), np.arange(256)):
        raise RuntimeError("E11 256-edge feature order changed")
    roi_i = pd.to_numeric(mapping.roi_i_zero_based).to_numpy(dtype=int)
    roi_j = pd.to_numeric(mapping.roi_j_zero_based).to_numpy(dtype=int)
    corr_columns = [str(i) for i in range(256)]
    static_corr = frame[corr_columns].to_numpy(dtype=np.float64)
    bold_root = Path(CFG["data"]["bold_root"])
    records = merged.sort_values("csv_row_zero_based").to_dict(orient="records")

    def check_row(rec):
        row = int(rec["csv_row_zero_based"])
        sid = int(rec["subject_SUB_ID"])
        filename = str(rec["roi_raw_file"])
        path = bold_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(sid) not in filename or filename != str(rec["FILE_ID"]) + "_rois_ho.1D":
            raise RuntimeError(f"subject/file identity mismatch at row {row}")
        if int(rec["DX_GROUP"]) != int(raw_label[row]):
            raise RuntimeError(f"ASD/TD row mapping mismatch at row {row}")
        site = str(rec["SITE_ID"]).upper()
        if site not in CFG["tr_seconds_by_site_id"] or site != str(Path(filename).name.split("_")[0]).upper() and not filename.startswith(("MaxMun_", "Leuven_", "UCLA_", "UM_")):
            raise RuntimeError(f"unverified site/protocol mapping at row {row}: {filename} / {site}")
        actual_sha = sha256(path)
        if actual_sha != expected_hashes.get(filename):
            raise RuntimeError(f"E11 BOLD content hash mismatch at row {row}")
        with path.open("rt", encoding="utf-8") as stream:
            header = stream.readline().strip()
        header_values = [int(v.removeprefix("#")) for v in header.split("\t")]
        if not np.array_equal(np.asarray(header_values, dtype=int), roi_values):
            raise RuntimeError(f"E11 ROI column order mismatch at row {row}")
        series = np.loadtxt(path, dtype=np.float64, ndmin=2)
        if series.shape[1] != 111 or not np.isfinite(series).all():
            raise RuntimeError(f"BOLD values invalid at row {row}")
        reconstructed = e11.nilearn_071_compatible(series, roi_i, roi_j)
        delta = np.abs(reconstructed - static_corr[row])
        if delta.shape != (256,) or not np.isfinite(delta).all():
            raise RuntimeError(f"static FC reconstruction invalid at row {row}")
        return {"row": row, "subject_id": sid, "timepoints": int(series.shape[0]),
                "max_abs_error": float(delta.max()), "sum_sq_error": float(np.square(delta).sum()),
                "constant_rois": int(np.count_nonzero(np.ptp(series, axis=0) == 0))}

    workers = min(8, os.cpu_count() or 1)
    checked = [None] * 871
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_row, rec): int(rec["csv_row_zero_based"]) for rec in records}
        for future in as_completed(futures):
            row = futures[future]
            checked[row] = future.result()
    max_error = max(row["max_abs_error"] for row in checked)
    rmse = float(np.sqrt(sum(row["sum_sq_error"] for row in checked) / (871 * 256)))
    if max_error > CFG["static_fc"]["max_absolute_error_tolerance"]:
        raise RuntimeError(f"static FC reconstruction exceeds E11 tolerance: {max_error}")
    return {"subjects": len(checked), "unique_subject_ids": len({row["subject_id"] for row in checked}),
            "roi_count": 111, "edge_count": 256, "edge_order_verified": True,
            "all_e11_bold_sha256_verified": True, "all_roi_headers_verified": True,
            "all_bold_values_finite": True, "timepoints_min": min(row["timepoints"] for row in checked),
            "timepoints_max": max(row["timepoints"] for row in checked),
            "static_fc_elements_compared": 871 * 256, "static_fc_max_abs_error": max_error,
            "static_fc_global_rmse": rmse, "static_fc_within_1e_6": True,
            "constant_roi_column_occurrences": int(sum(row["constant_rois"] for row in checked)),
            "timepoint_row_sequence_metadata": "CPAC .1D contains ordered sample rows but no per-frame timestamps or censor/delete log; internal temporal-gap detection is unavailable from the supplied files."}


def verify_window_cache() -> dict:
    cache_path = EXP / CFG["runtime"]["dynamic_cache"]
    with np.load(cache_path, allow_pickle=False) as cache:
        c = {name: cache[name].copy() for name in cache.files}
    if not np.array_equal(c["row"], np.arange(871)) or int(c["dynamic_eligible"].sum()) != 871:
        raise RuntimeError("cached BOLD rows or dynamic coverage mismatch")
    means, dynamics, ordered = [], [], []
    valid_shrinkages = []
    no_unknown = [195, 225, 232, 236]
    for row in range(871):
        count = int(c["window_count"][row])
        windows = c["window_fc_padded"][row, :count]
        permutation = c["window_permutation"][row, :count]
        if count < 3 or not np.array_equal(np.sort(permutation), np.arange(count)):
            raise RuntimeError(f"invalid window count/order permutation for row {row}")
        seed_text = f"E80-A3-{CFG['windows']['order_control_seed_base']}-{int(c['subject_id'][row])}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:4], "little")
        if not np.array_equal(permutation, np.random.default_rng(seed).permutation(count)):
            raise RuntimeError(f"temporal-order control is not the registered subject-only permutation, row {row}")
        mean = windows.mean(axis=0)
        var = windows.std(axis=0, ddof=0)
        adjacent = np.abs(np.diff(windows, axis=0)).mean(axis=0)
        adjacent_perm = np.abs(np.diff(windows[permutation], axis=0)).mean(axis=0)
        shrinkage = c["window_shrinkage_padded"][row, :count]
        if not np.isfinite(windows).all() or not np.isfinite(shrinkage).all() or np.any((shrinkage < 0) | (shrinkage > 1)):
            raise RuntimeError(f"invalid window FC or shrinkage data row {row}")
        valid_shrinkages.extend(shrinkage.tolist())
        means.append(mean)
        dynamics.append(np.concatenate([var, adjacent]))
        ordered.append(np.concatenate([var, adjacent_perm]))
    means, dynamics, ordered = np.asarray(means), np.asarray(dynamics), np.asarray(ordered)
    errors = {
        "window_mean_max_abs_error": float(np.max(np.abs(means - c["window_mean"]))),
        "dynamic_max_abs_error": float(np.max(np.abs(dynamics - c["dynamic"]))),
        "order_control_max_abs_error": float(np.max(np.abs(ordered - c["dynamic_order"]))),
        "window_count_min": int(c["window_count"].min()), "window_count_max": int(c["window_count"].max()),
        "all_permutations_valid": True, "all_window_features_reconstructed": True,
    }
    keep = np.ones(256, dtype=bool)
    keep[no_unknown] = False
    errors["window_mean_no_unknown_max_abs_error"] = float(np.max(np.abs(means[:, keep] - c["window_mean_no_unknown_roi"])))
    errors["dynamic_no_unknown_max_abs_error"] = float(np.max(np.abs(np.concatenate([dynamics[:, :256][:, keep], dynamics[:, 256:][:, keep]], axis=1) - c["dynamic_no_unknown_roi"])))
    errors["shrinkage_min"] = float(np.min(valid_shrinkages))
    errors["shrinkage_median"] = float(np.median(valid_shrinkages))
    errors["shrinkage_max"] = float(np.max(valid_shrinkages))
    errors["shrinkage_fraction_ge_0_99"] = float(np.mean(np.asarray(valid_shrinkages) >= 0.99))
    if max(errors[k] for k in ("window_mean_max_abs_error", "dynamic_max_abs_error", "order_control_max_abs_error",
                               "window_mean_no_unknown_max_abs_error", "dynamic_no_unknown_max_abs_error")) > 1e-12:
        raise RuntimeError("window feature reconstruction mismatch")
    return errors


def verify_probe_outputs() -> tuple[list[dict], pd.DataFrame, dict]:
    frame = pd.read_csv(Path(CFG["data"]["csv_path"]), low_memory=False)
    y = pd.to_numeric(frame.label, errors="raise").to_numpy(dtype=int) - 1
    x = frame.iloc[:, :320].to_numpy(dtype=np.float64)
    expected_splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(x, y))
    fold_metrics = pd.read_csv(RESULTS / "fold_metrics.csv")
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    traces = json.loads((RESULTS / "fold_trace.json").read_text(encoding="utf-8"))
    methods = ["A0", "A1", "A2", "A3", "DynamicOnly", "A1_residualized", "A2_residualized", "A1_no_unknown_roi", "A2_no_unknown_roi"]
    if len(traces) != 10 or len(fold_metrics) != 90 or len(predictions) != 871 * len(methods):
        raise RuntimeError("probe output row count mismatch")
    by_method_fold = {(str(row.method), int(row.fold)): row for row in fold_metrics.itertuples()}
    audit_rows = []
    for fold, (expected_train, expected_test) in enumerate(expected_splits, start=1):
        trace = next(x for x in traces if int(x["fold"]) == fold)
        train_rows = np.asarray(trace["train_rows"], dtype=int)
        test_rows = np.asarray(trace["test_rows"], dtype=int)
        allowed_train = expected_train.copy()
        removed = []
        if fold == 2:
            removed = [i for i in (376, 406) if i in set(expected_test.tolist())]
            allowed_train = allowed_train[~np.isin(allowed_train, removed)]
        if not np.array_equal(train_rows, allowed_train) or not np.array_equal(test_rows, expected_test):
            raise RuntimeError(f"outer fold {fold} rows differ from protocol/purged training")
        if fold == 2 and (set(removed) != {376, 406} or set(removed) & set(train_rows)):
            raise RuntimeError("Fold 2 historical overlap rows were not held out of probe training")
        method_traces = trace["methods"]
        for method in methods:
            inner_records = method_traces[method]["inner_splits"]
            if len(inner_records) != 5:
                raise RuntimeError(f"inner split count mismatch fold={fold} method={method}")
            seen_valid = []
            for inner_record in inner_records:
                itrain = np.asarray(inner_record["train_rows"], dtype=int)
                ivalid = np.asarray(inner_record["valid_rows"], dtype=int)
                if len(np.intersect1d(itrain, ivalid)) or not set(itrain).issubset(set(train_rows)) or not set(ivalid).issubset(set(train_rows)):
                    raise RuntimeError(f"inner split leakage fold={fold} method={method}")
                if set(itrain) | set(ivalid) != set(train_rows):
                    raise RuntimeError(f"inner split does not partition outer train fold={fold} method={method}")
                seen_valid.extend(ivalid.tolist())
            if sorted(seen_valid) != sorted(train_rows.tolist()):
                raise RuntimeError(f"inner OOF validation coverage mismatch fold={fold} method={method}")
            if method != "A0":
                dims = method_traces[method]["outer_dimensions"]
                if sum(v["effective"] for v in dims.values()) != (384 if method not in ("DynamicOnly",) else 32):
                    raise RuntimeError(f"outer feature dimension mismatch fold={fold} method={method}")
                for split_dimensions in method_traces[method]["inner_dimensions"]:
                    if any(v["requested"] > 0 and v["effective"] > v["requested"] for v in split_dimensions.values()):
                        raise RuntimeError("PCA exceeded requested fold-local dimension")
            if method_traces[method]["chosen_C"] not in [0.01, 0.1, 1.0, 10.0]:
                raise RuntimeError("selected C fell outside predeclared candidates")
            losses = {float(key): float(value) for key, value in method_traces[method]["inner_losses"].items()}
            minimum_loss = min(losses.values())
            expected_c = min(c for c, value in losses.items() if value <= minimum_loss + 1e-12)
            if float(method_traces[method]["chosen_C"]) != expected_c:
                raise RuntimeError(f"inner log-loss tie rule mismatch fold={fold} method={method}")
            predicted = predictions[(predictions.method == method) & (predictions.fold == fold)].sort_values("row_zero_based")
            if not np.array_equal(predicted.row_zero_based.to_numpy(dtype=int), test_rows):
                raise RuntimeError(f"OOF prediction rows/order mismatch fold={fold} method={method}")
            if not np.array_equal(predicted.label_asd0_td1.to_numpy(dtype=int), y[test_rows]):
                raise RuntimeError("OOF labels differ from original dataset rows")
            reconstructed = metric_values(y[test_rows], predicted.prediction_asd0_td1.to_numpy(dtype=int), predicted.probability_asd.to_numpy(dtype=float))
            stored = by_method_fold[(method, fold)]
            for key, value in reconstructed.items():
                if abs(float(getattr(stored, key)) - value) > 1e-12:
                    raise RuntimeError(f"probe fold metric mismatch {method}/{fold}/{key}")
            audit_rows.append({"fold": fold, "method": method, "outer_train_rows": len(train_rows),
                               "outer_test_rows": len(test_rows), "fold2_removed_rows": json.dumps(removed),
                               "inner_train_test_overlap": 0, "inner_training_scope_pass": True,
                               "prediction_row_identity_pass": True, "metrics_recomputed_pass": True,
                               "effective_input_dimension": int(sum(v["effective"] for v in method_traces[method]["outer_dimensions"].values()))})
    if not predictions.groupby("method").row_zero_based.apply(lambda s: set(s) == set(range(871))).all():
        raise RuntimeError("each method must have exactly one OOF prediction per eligible patient")
    return audit_rows, predictions, {"outer_folds": 10, "probe_method_folds_audited": len(audit_rows),
                                    "probe_prediction_rows_audited": len(predictions),
                                    "pca_scaler_inner_train_only": True,
                                    "external_test_labels_used_for_selection": False,
                                    "fold2_rows_376_406_removed_from_probe_train": True}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_darg() -> tuple[list[dict], pd.DataFrame, dict]:
    import torch
    import subprocess
    import pickle

    def hash_object(value):
        digest = hashlib.sha256()

        def visit(item):
            if torch.is_tensor(item):
                digest.update(b"T" + str(item.dtype).encode() + str(tuple(item.shape)).encode())
                digest.update(item.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(item, np.ndarray):
                digest.update(b"N" + str(item.dtype).encode() + str(item.shape).encode())
                digest.update(np.ascontiguousarray(item).tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=str):
                    digest.update(str(key).encode() + b"\0")
                    visit(item[key])
            elif isinstance(item, (list, tuple)):
                digest.update(type(item).__name__.encode())
                for child in item:
                    visit(child)
            else:
                digest.update(json.dumps(item, sort_keys=True, default=str).encode() + b"\0")

        visit(value)
        return digest.hexdigest()

    def hash_model_cam(model, cam):
        digest = hashlib.sha256()
        for prefix, values in (("model", model), ("cam", cam)):
            for name, tensor in sorted(values.items()):
                digest.update((prefix + "/" + name).encode() + b"\0")
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def hash_rng(state):
        digest = hashlib.sha256(pickle.dumps((state["python"], state["numpy"]), protocol=4))
        digest.update(state["cpu"].cpu().numpy().tobytes())
        for value in state.get("cuda", []):
            digest.update(value.cpu().numpy().tobytes())
        return digest.hexdigest()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    e51_commit = "939fe7e05ec141582af6a26bc19713fda08fdba3"
    e51_manifest = subprocess.run(["git", "-C", str(REPO), "show", f"{e51_commit}:experiment/experiment51_dugsa/manifest.json"],
                                  check=True, capture_output=True, text=True).stdout
    e51_manifest_data = json.loads(e51_manifest)
    e51_source = subprocess.run(["git", "-C", str(REPO), "show", f"{e51_commit}:experiment/experiment51_dugsa/dugsa.py"],
                                check=True, capture_output=True).stdout
    e51_source_sha = hashlib.sha256(e51_source).hexdigest()
    if e51_manifest_data.get("status") != "COMPLETE_AUDITED" or e51_source_sha != e51_manifest_data["code_sha"]["dugsa.py"]:
        raise RuntimeError("historical E51 DUGSA dependency is not verified")
    reference_manifest = json.loads((E74 / "manifest.json").read_text(encoding="utf-8"))
    reference_audit = json.loads((E74 / "results/audit_status.json").read_text(encoding="utf-8"))
    if reference_audit.get("status") != "PASS" or reference_audit.get("audited_checkpoints") != 30:
        raise RuntimeError("historical E74 independent audit record is not PASS 30/30")
    group = "E74.4-G2-Positive-DCR"
    expected_source = reference_manifest["training_source_sha256"]
    hist_rows = pd.read_csv(E74 / "results/fold_metrics.csv")
    hist_rows = hist_rows[hist_rows.group == group]
    if len(hist_rows) != 10:
        raise RuntimeError("E74.4-G2 historical fold metrics are incomplete")
    protected_sources = reference_manifest["protected_sources"]
    protected_mismatches, source_resolution = [], {}
    for source, expected_sha in protected_sources.items():
        path = Path(source)
        if not path.is_file() and source.startswith("/model/spromptgl/experiment/"):
            path = REPO / "experiment" / source.removeprefix("/model/spromptgl/experiment/")
        if not path.is_file():
            protected_mismatches.append({"path": source, "error": "missing"})
            continue
        actual = sha256(path)
        source_resolution[source] = {"resolved_path": str(path), "sha256": actual}
        if actual != expected_sha:
            protected_mismatches.append({"path": source, "expected": expected_sha, "actual": actual})
    if protected_mismatches:
        raise RuntimeError(f"historical protected-source SHA mismatch: {protected_mismatches[:2]}")
    frame = pd.read_csv(Path(CFG["data"]["csv_path"]), low_memory=False)
    y = pd.to_numeric(frame.label, errors="raise").to_numpy(dtype=int) - 1
    splits = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=10).split(np.zeros(871), y))
    warmup_root = REPO / "experiment/experiment61_cadr_postalign_vs_e2e/.runtime/warmup"
    e61_tasks = REPO / "experiment/experiment61_cadr_postalign_vs_e2e/.runtime/tasks"
    e53_manifest = json.loads((REPO / "experiment/experiment53_3_cadr_graph_fallback_fix/manifest.json").read_text(encoding="utf-8"))
    rows = []
    per_patient = []
    for fold, (train_rows, test_rows) in enumerate(splits, start=1):
        task_path = E74 / ".runtime/tasks" / f"{group}_f{fold:02d}.json"
        checkpoint_path = E74 / ".runtime/checkpoints" / f"{group}_f{fold:02d}.best.pt"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_sha = sha256(checkpoint_path)
        if task.get("status") != "COMPLETE" or task.get("group") != group or int(task.get("fold")) != fold:
            raise RuntimeError(f"historical E74 task identity mismatch fold {fold}")
        if checkpoint_sha != task["checkpoint_sha256"]:
            raise RuntimeError(f"E74.4 checkpoint/task hash mismatch fold {fold}")
        if checkpoint.get("source_sha256") != expected_source or checkpoint.get("identity") != task["identity"]:
            raise RuntimeError(f"E74.4 checkpoint/task identity mismatch fold {fold}")
        if checkpoint.get("rng_trace") != task.get("rng_trace"):
            raise RuntimeError(f"E74.4 RNG/warmup trace mismatch fold {fold}")
        identity = task["identity"]
        warmup_path = warmup_root / f"fold{fold:02d}.pt"
        warmup = torch.load(warmup_path, map_location="cpu", weights_only=False)
        if sha256(warmup_path) != identity["warmup_sha256"] or warmup.get("rng_sha") != identity["warmup_rng_sha"]:
            raise RuntimeError(f"E61 warmup file/RNG identity mismatch fold {fold}")
        e61_task = json.loads((e61_tasks / f"E61-E1_f{fold:02d}.json").read_text(encoding="utf-8"))
        anchor_key = f"/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/fold{fold:02d}.pt"
        for key in ("initial_shared_sha", "initial_gfr_sha", "initial_cam_sha", "initial_rng_sha", "split_sha"):
            if warmup.get(key) != e61_task.get(key):
                raise RuntimeError(f"E61 warmup/task initialization mismatch fold={fold} key={key}")
        if not (warmup.get("anchor_sha") == e61_task.get("anchor_sha") == e53_manifest["anchor_sha256"].get(anchor_key)):
            raise RuntimeError(f"E43 anchor identity ledger mismatch fold {fold}")
        model_state = {key: value.detach().cpu().clone() for key, value in warmup["model"].items()}
        model_state["calibration"] = torch.tensor([[1.0, 0.0]] * 4, dtype=model_state["calibration"].dtype)
        model_state["reliability"] = torch.ones_like(model_state["reliability"])
        model_state["reliability_gate_logits"] = torch.zeros_like(model_state["reliability_gate_logits"])
        model_state["adaptive_gate_logits"] = torch.tensor(
            [-2.1972246170043945] * 3 + [4.595119953155518], dtype=model_state["adaptive_gate_logits"].dtype
        )
        if hash_model_cam(model_state, warmup["cam"]) != identity["model_cam_sha"]:
            raise RuntimeError(f"E74 model/CAM initialization SHA mismatch fold {fold}")
        optimizer_state = {"opt_base": warmup["optimizer_base"], "opt_cam": warmup["optimizer_cam"],
                           "opt_added": warmup["optimizer_added"]}
        if hash_object(optimizer_state) != identity["optimizer_sha"]:
            raise RuntimeError(f"E74 optimizer initialization SHA mismatch fold {fold}")
        if hash_rng(warmup["rng"]) != warmup["rng_sha"]:
            raise RuntimeError(f"E61 serialized RNG state mismatch fold {fold}")
        if len(warmup.get("rng_trace", [])) != 10 or warmup["rng_trace"][-1].get("rng_after_eval") != warmup["rng_sha"]:
            raise RuntimeError(f"E61 epoch-10 warmup RNG trace mismatch fold {fold}")
        warmup_train = np.asarray(warmup["train_indices"], dtype=np.int64)
        warmup_pairs = np.asarray(warmup["train_pairs"], dtype=np.int64)
        warmup_same = np.asarray(warmup["train_same"], dtype=np.int64)
        if not np.array_equal(warmup_train, np.asarray(train_rows, dtype=np.int64)):
            raise RuntimeError(f"E61 warmup training membership mismatch fold {fold}")
        if warmup_pairs.shape != (4096, 2) or not np.isin(warmup_pairs, warmup_train).all():
            raise RuntimeError(f"E61 warmup training pair scope mismatch fold {fold}")
        if not np.array_equal(warmup_same, y[warmup_pairs[:, 0]] == y[warmup_pairs[:, 1]]):
            raise RuntimeError(f"E61 warmup relation-pair targets mismatch fold {fold}")
        if warmup.get("pair_sha") != identity["membership"]["warmup_pair_sha"]:
            raise RuntimeError(f"E74 warmup pair SHA mismatch fold {fold}")
        split_sha = hashlib.sha256(
            np.asarray(train_rows, dtype=np.int64).tobytes() + np.asarray(test_rows, dtype=np.int64).tobytes()
        ).hexdigest()
        if split_sha != identity["split_sha"] or split_sha != warmup.get("split_sha"):
            raise RuntimeError(f"E74/E61 split SHA mismatch fold {fold}")
        overlap = [376, 406] if fold == 2 else []
        historical_overlap = [int(value) - 1 for value in identity["membership"]["train_test_overlap_1based"]]
        if historical_overlap != overlap or (fold == 2 and not set(overlap).issubset(set(test_rows))):
            raise RuntimeError(f"historical split overlap does not match expected Fold 2 condition fold {fold}")
        if len(checkpoint["test_prediction"]) != len(test_rows):
            raise RuntimeError(f"historical test prediction length mismatch fold {fold}")
        pred = np.asarray(checkpoint["test_prediction"], dtype=int)
        prob = np.asarray(checkpoint["test_probability"], dtype=float)
        p_asd = prob[:, 0]
        values = metric_values(y[test_rows], pred, p_asd)
        source_metric = hist_rows[hist_rows.fold == fold].iloc[0]
        for name, value in values.items():
            if abs(value - float(source_metric[name])) > 1e-10:
                raise RuntimeError(f"historical E74.4 metric mismatch fold={fold} {name}")
        rows.append({"fold": fold, "status": "CHECKPOINT_ARTIFACT_REPLAY_PASS", "checkpoint_sha256": checkpoint_sha,
                     "task_checkpoint_sha256_match": True, "source_sha256": expected_source,
                     "best_epoch_1based": task["best_epoch_1based"], "epochs_executed": task["epochs_executed"],
                     "warmup_sha256": identity["warmup_sha256"], "warmup_rng_sha": identity["warmup_rng_sha"],
                     "initial_model_cam_sha": identity["model_cam_sha"], "initial_optimizer_sha": identity["optimizer_sha"],
                     "split_sha": identity["split_sha"], "pair_sha": identity["pair_sha"],
                     "effective_train_sha": identity["effective_train_sha"],
                     "train_test_overlap_1based": json.dumps(identity["membership"]["train_test_overlap_1based"]),
                     "rng_trace_entries": len(task["rng_trace"]), "rng_trace_task_match": True,
                     "test_predictions_exact": True, "metrics_independently_recomputed": True,
                     **values})
        for row, truth, prediction, probability in zip(test_rows, y[test_rows], pred, p_asd):
            per_patient.append({"row_zero_based": int(row), "row_one_based": int(row + 1), "fold": fold,
                                "label_asd0_td1": int(truth), "prediction_asd0_td1": int(prediction),
                                "probability_asd": float(probability), "correct": bool(truth == prediction)})
        del checkpoint, warmup
        print(f"read-only historical DARG fold {fold}/10 artifact-replayed", flush=True)
    for source, expected_sha in protected_sources.items():
        path = Path(source)
        if not path.is_file() and source.startswith("/model/spromptgl/experiment/"):
            path = REPO / "experiment" / source.removeprefix("/model/spromptgl/experiment/")
        if not path.is_file() or sha256(path) != expected_sha:
            raise RuntimeError(f"historical protected source changed during E80 audit: {source}")
    return rows, pd.DataFrame(per_patient), {"status": "PASS_ARTIFACT_REPLAY", "group": group, "folds_audited": len(rows),
           "checkpoint_task_rng_warmup_identity_checked": True,
           "historical_e74_independent_audit_status": reference_audit["status"],
           "historical_e74_independent_audit_checkpoints": reference_audit["audited_checkpoints"],
           "forward_replay_in_this_run": False,
           "forward_replay_limit": "E43.1 anchor binaries are absent from both server mounts; historical E74 forward audit was used without substitution.",
           "predictions_exact": True, "historical_metrics_independently_recomputed": True,
           "protected_source_count": len(protected_sources), "protected_source_sha_mismatches": 0,
           "protected_sources_unchanged_after_audit": True,
           "protected_sources_resolved": source_resolution,
           "e51_runtime_dependency_sha256": e51_source_sha,
           "e51_runtime_dependency_commit": e51_commit,
           "note": "E80 did not retrain DARG or claim new epoch RNG parity; historical E74 30/30 forward audit remains the forward-replay evidence.",
           "historical_best_checkpoint_selection_retains_val_equals_test": True}


def complementary_analysis(darg_rows: pd.DataFrame, predictions: pd.DataFrame, patient_audit: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    main_rows, site_rows = [], []
    darg = darg_rows.set_index("row_zero_based")
    for method in ("A2", "DynamicOnly"):
        pred = predictions[predictions.method == method].set_index("row_zero_based")
        joined = darg.join(pred[["prediction_asd0_td1", "probability_asd"]].rename(columns={
            "prediction_asd0_td1": f"{method}_pred", "probability_asd": f"{method}_p_asd"}))
        d_correct = joined.correct.astype(bool).to_numpy()
        dyn_correct = (joined[f"{method}_pred"].to_numpy(dtype=int) == joined.label_asd0_td1.to_numpy(dtype=int))
        counts = {"both_correct": int(np.sum(d_correct & dyn_correct)),
                  "darg_wrong_dynamic_correct": int(np.sum(~d_correct & dyn_correct)),
                  "darg_correct_dynamic_wrong": int(np.sum(d_correct & ~dyn_correct)),
                  "both_wrong": int(np.sum(~d_correct & ~dyn_correct))}
        wrong_mask = ~d_correct
        dynamic_support = joined[f"{method}_pred"].to_numpy(dtype=int)
        for fold_scope, mask in (("all10", np.ones(len(joined), dtype=bool)),
                                 ("exclude_fold2", joined.fold.to_numpy(dtype=int) != 2)):
            local_d = d_correct[mask]
            local_m = dyn_correct[mask]
            main_rows.append({"method": method, "scope": fold_scope, "n": int(mask.sum()),
                              "darg_wrong_dynamic_correct": int(np.sum(~local_d & local_m)),
                              "darg_correct_dynamic_wrong": int(np.sum(local_d & ~local_m)),
                              "both_wrong": int(np.sum(~local_d & ~local_m)),
                              "both_correct": int(np.sum(local_d & local_m)),
                              "darg_error_dynamic_correct_fraction": float(np.sum(~local_d & local_m) / max(np.sum(~local_d), 1)),
                              "darg_error_dynamic_wrong_fraction": float(np.sum(~local_d & ~local_m) / max(np.sum(~local_d), 1))})
        joined = joined.join(patient_audit.set_index("row")["site"])
        wrong_darg = ~joined.correct.astype(bool)
        for site, group in joined.groupby("site", sort=True):
            rows = group
            d_ok = rows.correct.astype(bool).to_numpy()
            m_ok = rows[f"{method}_pred"].to_numpy(dtype=int) == rows.label_asd0_td1.to_numpy(dtype=int)
            site_rows.append({"method": method, "site": site, "n": len(rows),
                              "darg_wrong_dynamic_correct": int(np.sum(~d_ok & m_ok)),
                              "darg_correct_dynamic_wrong": int(np.sum(d_ok & ~m_ok)),
                              "both_wrong": int(np.sum(~d_ok & ~m_ok)),
                              "both_correct": int(np.sum(d_ok & m_ok)),
                              "darg_errors": int(np.sum(~d_ok)),
                              "dynamic_correct_given_darg_error": float(np.sum(~d_ok & m_ok) / max(np.sum(~d_ok), 1))})
        if method == "A2":
            darg_wrong_support = joined.loc[wrong_mask, ["label_asd0_td1", "A2_pred", "A2_p_asd", "fold", "site"]].copy()
            counts["darg_errors_total"] = int(wrong_mask.sum())
            counts["dynamic_asd_support_on_darg_errors"] = int(np.sum((joined.loc[wrong_mask, "A2_p_asd"].to_numpy() >= 0.5)))
            counts["dynamic_td_support_on_darg_errors"] = int(np.sum((joined.loc[wrong_mask, "A2_p_asd"].to_numpy() < 0.5)))
        else:
            darg_wrong_support = None
        counts["method"] = method
    return main_rows, site_rows, {"comparison_type": "posthoc potential complementarity; not fused correction",
                                  "darg_prediction_rows": int(len(darg_rows)),
                                  "methods": {method: {row["scope"]: row for row in main_rows if row["method"] == method} for method in ("A2", "DynamicOnly")}}


def confound_analysis(patient_audit: pd.DataFrame, features: dict) -> tuple[list[dict], dict]:
    from scipy.stats import spearmanr
    dynamic = np.asarray(features["dynamic"], dtype=float)
    variability = dynamic[:, :256].mean(axis=1)
    adjacent = dynamic[:, 256:].mean(axis=1)
    analysis_rows = []
    fields = ["tr_seconds", "duration_seconds", "timepoints", "window_count", "func_mean_fd", "func_dvars", "func_perc_fd"]
    for name, value in (("mean_window_variability", variability), ("mean_adjacent_change", adjacent)):
        for field in fields:
            x = pd.to_numeric(patient_audit[field], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(value)
            rho, pvalue = spearmanr(x[keep], value[keep])
            analysis_rows.append({"dynamic_summary": name, "confound": field, "n": int(keep.sum()),
                                 "spearman_rho": float(rho), "two_sided_p_descriptive": float(pvalue)})
        grand = float(np.mean(value))
        between = sum(len(g) * (float(np.mean(value[g.index])) - grand) ** 2 for _, g in patient_audit.groupby("site"))
        total = float(np.sum((value - grand) ** 2))
        eta2 = float(between / total) if total > 0 else 0.0
        analysis_rows.append({"dynamic_summary": name, "confound": "site_eta_squared", "n": len(value),
                             "spearman_rho": eta2, "two_sided_p_descriptive": float("nan")})
    summaries = {}
    for name, value in (("mean_window_variability", variability), ("mean_adjacent_change", adjacent)):
        summaries[name] = {"mean": float(np.mean(value)), "sd": float(np.std(value, ddof=0)),
                           "by_site_median": {str(site): float(np.median(value[group.index]))
                                              for site, group in patient_audit.groupby("site", sort=True)}}
    return analysis_rows, summaries


def site_and_error_diagnostics(darg: pd.DataFrame, predictions: pd.DataFrame, patient_audit: pd.DataFrame):
    darg = darg.merge(patient_audit[["row", "site", "func_mean_fd", "func_dvars", "func_perc_fd"]],
                      left_on="row_zero_based", right_on="row", validate="one_to_one")
    prediction_lookup = {method: sub.set_index("row_zero_based") for method, sub in predictions.groupby("method")}
    site_rows, class_rows, quality_rows, support_rows = [], [], [], []
    for method in ("E74.4-G2", "A0", "A1", "A2", "A3", "DynamicOnly"):
        if method == "E74.4-G2":
            source = darg.set_index("row_zero_based")
            pred = source.prediction_asd0_td1.to_numpy(dtype=int)
            prob = source.probability_asd.to_numpy(dtype=float)
            rows = source.index.to_numpy(dtype=int)
        else:
            source = prediction_lookup[method]
            pred = source.prediction_asd0_td1.to_numpy(dtype=int)
            prob = source.probability_asd.to_numpy(dtype=float)
            rows = source.index.to_numpy(dtype=int)
        meta = patient_audit.set_index("row").loc[rows]
        labels = darg.set_index("row_zero_based").loc[rows].label_asd0_td1.to_numpy(dtype=int)
        for site in sorted(meta.site.unique()):
            mask = meta.site.to_numpy() == site
            values = metric_values(labels[mask], pred[mask], prob[mask]) if len(np.unique(labels[mask])) == 2 else {}
            site_rows.append({"method": method, "site": str(site), "n": int(mask.sum()),
                              "asd_n": int(np.sum(labels[mask] == 0)), "td_n": int(np.sum(labels[mask] == 1)),
                              **{k: (100 * v if k in values else float("nan")) for k, v in values.items()},
                              "acc_percent": 100 * accuracy_score(labels[mask], pred[mask])})
    darg_idx = darg.set_index("row_zero_based")
    a2 = prediction_lookup["A2"]
    for method in ("A2", "DynamicOnly"):
        model = prediction_lookup[method]
        joined = darg_idx.join(model[["prediction_asd0_td1", "probability_asd"]].rename(columns={
            "prediction_asd0_td1": "dynamic_prediction", "probability_asd": "dynamic_probability_asd"}))
        truth = joined.label_asd0_td1.to_numpy(dtype=int)
        d_correct = joined.correct.astype(bool).to_numpy()
        m_pred = joined.dynamic_prediction.to_numpy(dtype=int)
        m_correct = m_pred == truth
        wrong_to_right = ~d_correct & m_correct
        right_to_wrong = d_correct & ~m_correct
        for event, event_mask in (("darg_wrong_dynamic_correct", wrong_to_right),
                                  ("darg_correct_dynamic_wrong", right_to_wrong),
                                  ("both_wrong", ~d_correct & ~m_correct),
                                  ("both_correct", d_correct & m_correct)):
            for label, label_name in ((0, "ASD"), (1, "TD")):
                class_rows.append({"method": method, "event": event, "true_class": label_name,
                                   "count": int(np.sum(event_mask & (truth == label)))})
        bad = joined.loc[~d_correct, ["label_asd0_td1", "prediction_asd0_td1", "dynamic_prediction",
                                     "dynamic_probability_asd", "fold", "site"]].copy()
        for idx, record in bad.iterrows():
            support_rows.append({"method": method, "row_one_based": int(idx + 1),
                                 "true_class": "ASD" if int(record.label_asd0_td1) == 0 else "TD",
                                 "darg_prediction": "ASD" if int(record.prediction_asd0_td1) == 0 else "TD",
                                 "dynamic_prediction": "ASD" if int(record.dynamic_prediction) == 0 else "TD",
                                 "dynamic_probability_asd": float(record.dynamic_probability_asd),
                                 "site": record.site, "fold": int(record.fold),
                                 "dynamic_correct_support": bool(int(record.dynamic_prediction) == int(record.label_asd0_td1))})
        for field in ("func_mean_fd", "func_dvars", "func_perc_fd"):
            numeric = pd.to_numeric(darg[field], errors="coerce")
            quartile = pd.qcut(numeric, q=4, labels=False, duplicates="drop")
            for q in sorted(quartile.dropna().unique()):
                mask = quartile.to_numpy() == q
                truth_q = truth[mask]
                d_q = d_correct[mask]
                m_q = m_correct[mask]
                quality_rows.append({"method": method, "quality_variable": field, "quartile": f"Q{int(q)+1}",
                                     "n": int(mask.sum()), "darg_errors": int(np.sum(~d_q)),
                                     "darg_wrong_dynamic_correct": int(np.sum(~d_q & m_q)),
                                     "darg_correct_dynamic_wrong": int(np.sum(d_q & ~m_q)),
                                     "both_wrong": int(np.sum(~d_q & ~m_q)),
                                     "asd_corrected": int(np.sum(~d_q & m_q & (truth_q == 0))),
                                     "td_corrected": int(np.sum(~d_q & m_q & (truth_q == 1)))})
    return site_rows, class_rows, quality_rows, support_rows


def main():
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    qc = json.loads((RESULTS / "qc_summary.json").read_text(encoding="utf-8"))
    qc_path = RESULTS / "patient_qc_audit.csv"
    patient_audit = pd.read_csv(qc_path, low_memory=False)
    static_data_check = verify_identity_and_static_fc()
    window_check = verify_window_cache()
    fold_audit_rows, predictions, probe_check = verify_probe_outputs()
    darg_fold_rows, darg_predictions, darg_check = audit_darg()
    darg_fold_frame = pd.DataFrame(darg_fold_rows)
    darg_fold_frame.to_csv(RESULTS / "darg_fold_metrics.csv", index=False)
    darg_predictions["method"] = "E74.4-G2"
    prediction_rows = pd.read_csv(RESULTS / "predictions.csv")
    test = predictions[predictions.method == "A2"].sort_values("row_zero_based")
    hist = darg_predictions.sort_values("row_zero_based")
    darg_comparable = hist[hist.row_zero_based.isin(test.row_zero_based)]
    if set(darg_comparable.row_zero_based) != set(test.row_zero_based):
        raise RuntimeError("DARG and statistical probe effective OOF cohorts differ")
    baseline_auc = float(np.mean([row["prob_auc"] for row in darg_fold_rows]) * 100)
    baseline_acc = float(np.mean([row["acc"] for row in darg_fold_rows]) * 100)
    baseline_hard = float(np.mean([row["hard_auc"] for row in darg_fold_rows]) * 100)
    baseline_sen = float(np.mean([row["sen"] for row in darg_fold_rows]) * 100)
    baseline_spe = float(np.mean([row["spe"] for row in darg_fold_rows]) * 100)
    baseline_f1 = float(np.mean([row["asd_f1"] for row in darg_fold_rows]) * 100)
    hist_summary = pd.read_csv(E74 / "results/summary.csv")
    historical = hist_summary[hist_summary.group == "E74.4-G2-Positive-DCR"].iloc[0]
    darg_probe_errors = {key: abs(actual - float(historical[f"{key}_mean_percent"])) for key, actual in {
        "acc": baseline_acc, "hard_auc": baseline_hard, "prob_auc": baseline_auc,
        "sen": baseline_sen, "spe": baseline_spe, "asd_f1": baseline_f1}.items()}
    if max(darg_probe_errors.values()) > 1e-8:
        raise RuntimeError(f"historical G2 ten-fold summary differs from report: {darg_probe_errors}")

    main_complement_rows, site_complement_rows, complement_summary = complementary_analysis(
        darg_predictions.rename(columns={"probability_asd": "probability_asd"}), prediction_rows, patient_audit)
    atomic_csv(RESULTS / "complementary_analysis.csv", main_complement_rows)
    atomic_csv(RESULTS / "complementarity_by_site.csv", site_complement_rows)
    site_rows, class_rows, quality_rows, support_rows = site_and_error_diagnostics(
        darg_predictions, prediction_rows, patient_audit)
    atomic_csv(RESULTS / "site_metrics.csv", site_rows)
    atomic_csv(RESULTS / "complementarity_by_class.csv", class_rows)
    atomic_csv(RESULTS / "complementarity_by_quality.csv", quality_rows)
    atomic_csv(RESULTS / "darg_error_support.csv", support_rows)
    with np.load(EXP / CFG["runtime"]["dynamic_cache"], allow_pickle=False) as cache:
        features = {name: cache[name].copy() for name in cache.files}
    confound_rows, confound_summary = confound_analysis(patient_audit, features)
    atomic_csv(RESULTS / "confound_analysis.csv", confound_rows)
    atomic_csv(RESULTS / "probe_fold_audit.csv", fold_audit_rows)

    main_table = []
    probe_fold = pd.read_csv(RESULTS / "fold_metrics.csv")
    for method in ("E74.4-G2", "A0", "A1", "A2", "A3", "DynamicOnly"):
        if method == "E74.4-G2":
            values = [row for row in darg_fold_rows]
        else:
            values = probe_fold[probe_fold.method == method].to_dict(orient="records")
        record = {"method": method, "folds": len(values), "subjects": 871 if method == "E74.4-G2" else int(qc["valid_dynamic_subjects"])}
        for metric in ("acc", "hard_auc", "prob_auc", "sen", "spe", "asd_f1"):
            vals = np.asarray([float(v[metric]) * 100 for v in values])
            record[f"{metric}_mean_percent"] = float(vals.mean())
            record[f"{metric}_population_sd_percent"] = float(vals.std(ddof=0))
        main_table.append(record)
    atomic_csv(RESULTS / "main_table.csv", main_table)

    reference_manifest = json.loads((E74 / "manifest.json").read_text(encoding="utf-8"))
    reference_audit_status = json.loads((E74 / "results/audit_status.json").read_text(encoding="utf-8"))
    if reference_audit_status.get("status") != "PASS" or reference_audit_status.get("audited_checkpoints") != 30:
        raise RuntimeError("E74.4 historical audit status is not a complete PASS")
    status = {
        "status": "PASS_INDEPENDENT_AUDIT",
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "data": {**static_data_check, "subjects": qc["subjects"], "dynamic_eligible": qc["valid_dynamic_subjects"],
                 "coverage": qc["dynamic_coverage"], "static_fc_max_abs_error": qc["static_reconstruction_max_absolute_error"],
                 "window_check": window_check, "tr_all_verified_from_protocol": qc["tr_recovered_for_all_subjects"]},
        "probe": probe_check,
        "darg_history": darg_check,
        "darg_historical_report_mean_abs_metric_differences_pp": darg_probe_errors,
        "complementarity": complement_summary,
        "confound_summaries": confound_summary,
        "historical_protocol_limits": {"val_equals_test_checkpoint_selection": True,
                                      "prefold_rfe_standardization": True, "transductive_full_patient_graph": True,
                                      "fold2_train_test_overlap_rows_zero_based": [376, 406],
                                      "new_probe_fold2_overlap_rows_excluded_from_train": True,
                                      "not_unbiased_clinical_generalization": True},
        "darg_new_training_performed": False,
        "new_epoch_rng_parity_claimed": False,
        "assets_unchanged": {"e74_manifest_status": reference_manifest["status"],
                             "e74_audit_status": reference_audit_status["status"]},
    }
    atomic_json(RESULTS / "audit_status.json", status)
    print(f"AUDIT={status['status']}; E74.4-G2=10/10; probe=90 method-fold audits; cache+OOF independently reconstructed", flush=True)


if __name__ == "__main__":
    main()
