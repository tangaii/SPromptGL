#!/usr/bin/env python3
"""Rebuild metrics from saved coordinates and replay Method B fit metadata."""
from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_experiment import (
    COORDS,
    EXP,
    HIST,
    INPUTS,
    MODEL_IDS,
    base_context,
    c5_source_path,
    fit_supervised_tsne,
    fit_tsne,
    generate_report,
    input_path,
    load_input,
    make_row,
    sha256_file,
    write_csv,
)

RESULTS = EXP / "results"
FORMAL_A = {
    "P0": (1.00, 0.00), "P1": (0.85, 0.00), "P2": (0.65, 0.00),
    "P3": (0.45, 0.00), "P4": (1.00, 0.50), "P5": (1.00, 1.00),
    "P6": (0.85, 0.50), "P7": (0.65, 0.50), "P8": (0.45, 1.00),
    "P9": (0.25, 2.00),
}
FORMAL_B = {
    "S0": (1.00, 0.00), "S1": (0.90, 0.00), "S2": (0.75, 0.00),
    "S3": (1.00, 0.25), "S4": (0.90, 0.25), "S5": (0.75, 0.25),
    "S6": (0.75, 0.50), "S7": (0.60, 0.50), "S8": (0.40, 1.00),
    "S9": (1.00, 1.00),
}
SCAN_A = [(a, b) for a in (1.0, .85, .65, .45, .25) for b in (0., .25, .5, 1., 2.)]
SCAN_B = [(s, g) for s in (1., .9, .75, .6, .4) for g in (0., .1, .25, .5, 1.)]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_coord(path):
    values = np.load(path, allow_pickle=False)
    if values.shape != (871, 2) or not np.isfinite(values).all():
        raise ValueError(f"invalid saved coordinates: {path}")
    return values


def main():
    started = time.monotonic()
    contexts = {}
    labels = None
    patient_ids = None
    historical_metrics = read_csv(HIST / "results/final_metrics.csv")
    historical_kl = {
        (row["model"], int(row["fold"])): float(row["kl_divergence"])
        for row in historical_metrics
        if row["candidate"] == "C5"
    }

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    for model in MODEL_IDS:
        for fold in range(1, 11):
            features, current_labels, current_ids, source_sha = load_input(model, fold)
            if labels is None:
                labels = current_labels.copy()
                patient_ids = current_ids.copy()
            if not np.array_equal(current_labels, labels) or not np.array_equal(current_ids, patient_ids):
                raise RuntimeError(f"patient sequence mismatch at {model} fold {fold}")
            pca50 = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(features))
            c5_path = COORDS / f"C5_{model}_fold{fold:02}.npy"
            c5 = load_coord(c5_path)
            contexts[(model, fold)] = {
                "features": features, "labels": current_labels, "patient_ids": current_ids,
                "input_sha": source_sha, "pca50": pca50, "c5": c5,
            }

    c5_jobs = [
        {"model": model, "fold": fold, "pca50": contexts[(model, fold)]["pca50"], "metric": "euclidean", "init": "pca"}
        for model in ("CAM_only",) for fold in range(1, 11)
    ]
    c5_results = {}
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fit_tsne, job): (job["model"], job["fold"]) for job in c5_jobs}
        for future in as_completed(futures):
            key = futures[future]
            coordinates, kl, runtime = future.result()
            saved = contexts[key]["c5"]
            if not np.array_equal(coordinates, saved):
                raise RuntimeError(f"CAM-only C5 deterministic replay mismatch at fold {key[1]}")
            c5_results[key] = (kl, runtime)

    for key, context in contexts.items():
        model, fold = key
        kl = c5_results[key][0] if model == "CAM_only" else historical_kl[key]
        context["base"] = base_context(context["features"], context["labels"], context["pca50"], kl)

    scan_rows = []
    formal_rows = []
    c5_rows = []
    for model in MODEL_IDS:
        for fold in range(1, 11):
            context = contexts[(model, fold)]
            path = COORDS / f"C5_{model}_fold{fold:02}.npy"
            kl = c5_results[(model, fold)][0] if model == "CAM_only" else historical_kl[(model, fold)]
            row = make_row(model, fold, "C5", "C5", context["c5"], context, context["input_sha"], sha256_file(path), 0.0, kl, "none_identity")
            row["coordinate_path"] = str(path.relative_to(EXP))
            c5_rows.append(row)
    formal_rows.extend(c5_rows)

    # Fit only unique Method B jobs. Existing coordinate arrays are retained and
    # compared elementwise to the deterministic replay before reporting metrics.
    permutation = np.random.default_rng(2027).permutation(871)
    shuffled = labels[permutation].astype(np.int64)
    jobs = {}
    for s, gamma in SCAN_B:
        for fold in (7, 8):
            for model in MODEL_IDS:
                jobs[(model, fold, s, gamma, "true_labels")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": labels, "s": s, "gamma": gamma}
    for group, (s, gamma) in FORMAL_B.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                jobs[(model, fold, s, gamma, "true_labels")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": labels, "s": s, "gamma": gamma}
    for fold in range(1, 11):
        for model in MODEL_IDS:
            jobs[(model, fold, .75, .5, "shuffled")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": shuffled, "s": .75, "gamma": .5}

    fit_results = {}
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fit_supervised_tsne, payload): key for key, payload in jobs.items()}
        for index, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            fit_results[key] = future.result()
            if index % 25 == 0 or index == len(futures):
                print(f"METHOD_B_REPLAY {index}/{len(futures)}", flush=True)

    def method_b_row(key, group, is_scan=False):
        model, fold, s, gamma, label_source = key
        coordinates, kl, runtime, d_scale = fit_results[key]
        suffix = "shuffled" if label_source == "shuffled" else "true_labels"
        path = COORDS / f"B_{model}_fold{fold:02}_s{s:.2f}_g{gamma:.2f}_{suffix}.npy"
        saved = load_coord(path)
        if not np.array_equal(coordinates, saved):
            raise RuntimeError(f"Method B deterministic replay mismatch: {path.name}")
        source = "none_distance_identity" if s == 1.0 and gamma == 0.0 else ("shuffled_labels_seed_2027" if label_source == "shuffled" else "true_labels")
        context = contexts[(model, fold)]
        row = make_row(model, fold, "B", group, saved, context, context["input_sha"], sha256_file(path), runtime, kl, source, {"s": s, "gamma": gamma}, d_scale)
        row["coordinate_path"] = str(path.relative_to(EXP))
        (scan_rows if is_scan else formal_rows).append(row)

    for s, gamma in SCAN_B:
        for fold in (7, 8):
            for model in MODEL_IDS:
                method_b_row((model, fold, s, gamma, "true_labels"), f"scan_s{s:.2f}_g{gamma:.2f}", True)
    for group, (s, gamma) in FORMAL_B.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                method_b_row((model, fold, s, gamma, "true_labels"), group)
    for fold in range(1, 11):
        for model in MODEL_IDS:
            method_b_row((model, fold, .75, .5, "shuffled"), "N2")

    # Reuse every saved Method A coordinate while timing an independent formula replay.
    def method_a_row(model, fold, group, alpha, beta, source, is_scan=False):
        context = contexts[(model, fold)]
        labels_for_transform = shuffled if source == "shuffled_labels_seed_2027" else labels
        begin = time.perf_counter()
        replay = __import__("run_experiment").transform_coordinates(context["c5"], labels_for_transform, alpha, beta)
        runtime = time.perf_counter() - begin
        if group.startswith("scan_"):
            path = COORDS / f"A_{group}_{model}_fold{fold:02}_a{alpha:.2f}_b{beta:.2f}.npy"
        else:
            path = COORDS / f"A_{group}_{model}_fold{fold:02}_a{alpha:.2f}_b{beta:.2f}.npy"
        saved = load_coord(path)
        if not np.array_equal(replay, saved):
            raise RuntimeError(f"Method A deterministic replay mismatch: {path.name}")
        label_source = "none_identity" if alpha == 1.0 and beta == 0.0 else source
        row = make_row(model, fold, "A", group, saved, context, context["input_sha"], sha256_file(path), runtime, None, label_source, {"alpha": alpha, "beta": beta})
        row["coordinate_path"] = str(path.relative_to(EXP))
        (scan_rows if is_scan else formal_rows).append(row)

    for alpha, beta in SCAN_A:
        group = f"scan_a{alpha:.2f}_b{beta:.2f}"
        for fold in (7, 8):
            for model in MODEL_IDS:
                method_a_row(model, fold, group, alpha, beta, "true_labels", True)
    for group, (alpha, beta) in FORMAL_A.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                method_a_row(model, fold, group, alpha, beta, "true_labels")
    for fold in range(1, 11):
        for model in MODEL_IDS:
            method_a_row(model, fold, "N1", .65, .5, "shuffled_labels_seed_2027")

    # Requested comparisons, one paired row per model/fold and comparison.
    metric_names = [
        "silhouette_2d", "silhouette_raw144", "silhouette_pca50", "asd_within_distance", "td_within_distance",
        "within_mean_distance", "normalized_within", "between_distance", "between_average_distance",
        "between_within_ratio", "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30",
        "trustworthiness_15", "trustworthiness_30", "pca_2d_neighbor_overlap_15", "pca_2d_neighbor_overlap_30",
        "kl_divergence",
    ]
    row_map = {(row["method"], row["group"], row["model"], int(row["fold"])): row for row in formal_rows}
    paired_rows = []

    def add_pair(name, model, fold, ref_method, ref_group, target_method, target_group):
        reference = row_map[(ref_method, ref_group, model, fold)]
        target = row_map[(target_method, target_group, model, fold)]
        item = {"comparison": name, "model": model, "fold": fold, "reference_method": ref_method, "reference_group": ref_group, "target_method": target_method, "target_group": target_group}
        for metric in metric_names:
            target_value = target.get(metric, "")
            reference_value = reference.get(metric, "")
            if target_value in ("", None, "nan", "NaN") or reference_value in ("", None, "nan", "NaN"):
                item[f"delta_{metric}"] = ""
            else:
                item[f"delta_{metric}"] = float(target_value) - float(reference_value)
        paired_rows.append(item)

    for model in MODEL_IDS:
        for fold in range(1, 11):
            for group in FORMAL_A:
                add_pair(f"C5_vs_A_{group}", model, fold, "C5", "C5", "A", group)
            add_pair("C5_vs_A_N1", model, fold, "C5", "C5", "A", "N1")
            add_pair("C5_vs_B_S0", model, fold, "C5", "C5", "B", "S0")
            for group in list(FORMAL_B)[1:]:
                add_pair(f"S0_vs_{group}", model, fold, "B", "S0", "B", group)
            for group in FORMAL_B:
                add_pair(f"C5_vs_B_{group}", model, fold, "C5", "C5", "B", group)
            add_pair("P7_vs_N1", model, fold, "A", "P7", "A", "N1")
            add_pair("S6_vs_N2", model, fold, "B", "S6", "B", "N2")
            for p_group in FORMAL_A:
                for s_group in FORMAL_B:
                    add_pair(f"{p_group}_vs_{s_group}", model, fold, "A", p_group, "B", s_group)
            for method, group in [("C5", "C5"), *[("A", g) for g in FORMAL_A], ("A", "N1"), *[("B", g) for g in FORMAL_B], ("B", "N2")]:
                for other in MODEL_IDS[1:]:
                    add_pair(f"CAM_only_to_{other}_{method}_{group}", other, fold, method, group, method, group)

    generate_report(formal_rows, scan_rows, paired_rows, c5_rows, contexts, json.loads((RESULTS / "cam_only_rebuild/cam_only_rebuild_status.json").read_text()))
    run_manifest = {
        "status": "RUN_COMPLETE_PENDING_AUDIT",
        "models": MODEL_IDS,
        "folds": 10,
        "scan_rows": len(scan_rows),
        "formal_rows": len(formal_rows),
        "paired_rows": len(paired_rows),
        "c5_rows": len(c5_rows),
        "supervised_tsne_jobs": len(jobs),
        "shuffle_mapping_sha256": sha256_file(INPUTS / "shuffled_label_mapping.csv"),
        "elapsed_seconds": time.monotonic() - started,
        "figures_reused_from_completed_generation": True,
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
