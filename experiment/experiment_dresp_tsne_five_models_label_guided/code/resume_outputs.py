#!/usr/bin/env python3
"""Complete outputs after a raster I/O interruption using the saved coordinates."""
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
    FORMAL_FIGURES,
    FIGURES,
    HIST,
    INPUTS,
    MODEL_IDS,
    MODEL_LABELS,
    PAIRED_FIGURES,
    SCAN_FIGURES,
    base_context,
    c5_source_path,
    fit_supervised_tsne,
    fit_tsne,
    generate_report,
    input_path,
    load_input,
    make_row,
    plot_grid,
    plot_pair,
    save_coord,
    transform_coordinates,
)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_contexts():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    contexts = {}
    historical_kl = read_rows(HIST / "results/final_metrics.csv")
    ids = None
    labels_reference = None
    for model in MODEL_IDS:
        for fold in range(1, 11):
            features, labels, patient_ids, input_sha = load_input(model, fold)
            ids = patient_ids if ids is None else ids
            labels_reference = labels if labels_reference is None else labels_reference
            if not np.array_equal(patient_ids, ids) or not np.array_equal(labels, labels_reference):
                raise RuntimeError(f"patient order or labels differ for {model} fold {fold}")
            if model == "CAM_only":
                c5 = np.load(COORDS / f"C5_CAM_only_fold{fold:02}.npy", allow_pickle=False)
                c5_kl = None
            else:
                c5 = np.load(c5_source_path(model, fold), allow_pickle=False)
                c5_kl = float(next(row["kl_divergence"] for row in historical_kl if row["model"] == model and int(row["fold"]) == fold and row["candidate"] == "C5"))
                save_coord(COORDS / f"C5_{model}_fold{fold:02}.npy", c5)
            scaled = StandardScaler().fit_transform(features)
            pca50 = PCA(n_components=50, random_state=0).fit_transform(scaled)
            contexts[(model, fold)] = {"model": model, "fold": fold, "features": features, "labels": labels, "patient_ids": patient_ids, "input_sha": input_sha, "scaled": scaled, "pca50": pca50, "c5": c5, "c5_kl": c5_kl}
    return contexts, labels_reference


def main():
    started = time.monotonic()
    for directory in (COORDS, SCAN_FIGURES, FORMAL_FIGURES, PAIRED_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    contexts, labels = build_contexts()

    # Replay CAM-only C5 to recover its objective value and replace the interrupted copy.
    c5_payloads = [{"model": "CAM_only", "fold": fold, "pca50": contexts[("CAM_only", fold)]["pca50"], "metric": "euclidean", "init": "pca"} for fold in range(1, 11)]
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fit_tsne, payload): payload["fold"] for payload in c5_payloads}
        for future in as_completed(futures):
            fold = futures[future]
            coords, kl, runtime = future.result()
            context = contexts[("CAM_only", fold)]
            context["c5"] = coords
            context["c5_kl"] = kl
            save_coord(COORDS / f"C5_CAM_only_fold{fold:02}.npy", coords)
            print(f"RESUME_C5 fold={fold:02} runtime={runtime:.2f}s", flush=True)
    for context in contexts.values():
        context["base"] = base_context(context["features"], context["labels"], context["pca50"], context["c5_kl"])

    permutation = np.random.default_rng(2027).permutation(871)
    shuffled = labels[permutation].astype(np.int64)
    scan_a = [(alpha, beta) for alpha in [1.0, 0.85, 0.65, 0.45, 0.25] for beta in [0.0, 0.25, 0.5, 1.0, 2.0]]
    scan_b = [(s, gamma) for s in [1.0, 0.9, 0.75, 0.6, 0.4] for gamma in [0.0, 0.1, 0.25, 0.5, 1.0]]
    formal_a = {"P0": (1.0, 0.0), "P1": (0.85, 0.0), "P2": (0.65, 0.0), "P3": (0.45, 0.0), "P4": (1.0, 0.5), "P5": (1.0, 1.0), "P6": (0.85, 0.5), "P7": (0.65, 0.5), "P8": (0.45, 1.0), "P9": (0.25, 2.0)}
    formal_b = {"S0": (1.0, 0.0), "S1": (0.9, 0.0), "S2": (0.75, 0.0), "S3": (1.0, 0.25), "S4": (0.9, 0.25), "S5": (0.75, 0.25), "S6": (0.75, 0.5), "S7": (0.6, 0.5), "S8": (0.4, 1.0), "S9": (1.0, 1.0)}

    formal_rows = []
    scan_rows = []
    c5_maps = {fold: {model: contexts[(model, fold)]["c5"] for model in MODEL_IDS} for fold in range(1, 11)}
    coord_lookup = {("C5", "C5", model, fold): contexts[(model, fold)]["c5"] for model in MODEL_IDS for fold in range(1, 11)}
    c5_rows = []
    for model in MODEL_IDS:
        for fold in range(1, 11):
            context = contexts[(model, fold)]
            path = COORDS / f"C5_{model}_fold{fold:02}.npy"
            row = make_row(model, fold, "C5", "C5", context["c5"], context, context["input_sha"], __import__("run_experiment").sha256_file(path), 0.0, context["c5_kl"], "none_identity")
            row["coordinate_path"] = str(path.relative_to(EXP))
            c5_rows.append(row)
    formal_rows.extend(c5_rows)

    def add_a(model, fold, group, alpha, beta, used_labels, label_source, scan=False):
        context = contexts[(model, fold)]
        coords = transform_coordinates(context["c5"], used_labels, alpha, beta)
        path = COORDS / f"A_{group}_{model}_fold{fold:02}_a{alpha:.2f}_b{beta:.2f}.npy"
        save_coord(path, coords)
        actual_source = "none_identity" if alpha == 1.0 and beta == 0.0 else label_source
        row = make_row(model, fold, "A", group, coords, context, context["input_sha"], __import__("run_experiment").sha256_file(path), 0.0, None, actual_source, {"alpha": alpha, "beta": beta})
        row["coordinate_path"] = str(path.relative_to(EXP))
        (scan_rows if scan else formal_rows).append(row)
        coord_lookup[("A", group, model, fold)] = coords

    for alpha, beta in scan_a:
        for fold in (7, 8):
            for model in MODEL_IDS:
                add_a(model, fold, f"scan_a{alpha:.2f}_b{beta:.2f}", alpha, beta, labels, "true_labels", True)
    for group, (alpha, beta) in formal_a.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                add_a(model, fold, group, alpha, beta, labels, "true_labels")
    for fold in range(1, 11):
        for model in MODEL_IDS:
            add_a(model, fold, "N1", 0.65, 0.5, shuffled, "shuffled_labels_seed_2027")

    jobs = {}
    for s, gamma in scan_b:
        for fold in (7, 8):
            for model in MODEL_IDS:
                jobs[(model, fold, s, gamma, "true_labels")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": labels, "s": s, "gamma": gamma}
    for group, (s, gamma) in formal_b.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                jobs[(model, fold, s, gamma, "true_labels")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": labels, "s": s, "gamma": gamma}
    for fold in range(1, 11):
        for model in MODEL_IDS:
            jobs[(model, fold, 0.75, 0.5, "shuffled_labels_seed_2027")] = {"pca50": contexts[(model, fold)]["pca50"], "used_labels": shuffled, "s": 0.75, "gamma": 0.5}
    b_results = {}
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fit_supervised_tsne, payload): key for key, payload in jobs.items()}
        for index, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            b_results[key] = future.result()
            if index % 25 == 0 or index == len(futures):
                print(f"RESUME_B_PROGRESS {index}/{len(futures)}", flush=True)

    def add_b(key, group, scan=False):
        model, fold, s, gamma, label_source = key
        coords, kl, runtime, scale = b_results[key]
        path = COORDS / f"B_{model}_fold{fold:02}_s{s:.2f}_g{gamma:.2f}_{'shuffled' if label_source.startswith('shuffled') else 'true_labels'}.npy"
        save_coord(path, coords)
        context = contexts[(model, fold)]
        source = "none_distance_identity" if s == 1.0 and gamma == 0.0 else label_source
        row = make_row(model, fold, "B", group, coords, context, context["input_sha"], __import__("run_experiment").sha256_file(path), runtime, kl, source, {"s": s, "gamma": gamma}, scale)
        row["coordinate_path"] = str(path.relative_to(EXP))
        (scan_rows if scan else formal_rows).append(row)
        coord_lookup[("B", group, model, fold)] = coords

    for s, gamma in scan_b:
        for fold in (7, 8):
            for model in MODEL_IDS:
                add_b((model, fold, s, gamma, "true_labels"), f"scan_s{s:.2f}_g{gamma:.2f}", True)
    for group, (s, gamma) in formal_b.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                add_b((model, fold, s, gamma, "true_labels"), group)
    for fold in range(1, 11):
        for model in MODEL_IDS:
            add_b((model, fold, 0.75, 0.5, "shuffled_labels_seed_2027"), "N2")

    # Rewrite all figures, including the missing paired figures, on the overlay filesystem.
    for fold in range(1, 11):
        plot_grid(c5_maps[fold], labels, f"Original unsupervised C5 t-SNE | Fold {fold:02} | n=871", FORMAL_FIGURES / f"C5_fold{fold:02}.png", 600)
    for method, specs in (("A", [f"scan_a{a:.2f}_b{b:.2f}" for a, b in scan_a]), ("B", [f"scan_s{s:.2f}_g{g:.2f}" for s, g in scan_b])):
        for group in specs:
            for fold in (7, 8):
                cmap = {model: coord_lookup[(method, group, model, fold)] for model in MODEL_IDS}
                if method == "A" and group == "scan_a1.00_b0.00":
                    title = f"Method A scan identity — original C5 coordinates — no labels used | Fold {fold:02}"
                elif method == "B" and group == "scan_s1.00_g0.00":
                    title = f"Method B scan identity — unmodified distances — random-init unsupervised t-SNE — no labels used | Fold {fold:02}"
                elif method == "A":
                    title = f"Method A — Post-hoc Label-Guided Transformation — LABEL-GUIDED / EXPLORATORY | {group} | Fold {fold:02}"
                else:
                    title = f"Method B — Supervised t-SNE (True Labels Used) — SUPERVISED / EXPLORATORY | {group} | Fold {fold:02}"
                plot_grid(cmap, labels, title, SCAN_FIGURES / f"{method}_{group}_fold{fold:02}.png", 300)
    formal_specs = [("A", g) for g in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "N1"]] + [("B", g) for g in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "N2"]]
    for method, group in formal_specs:
        for fold in range(1, 11):
            cmap = {model: coord_lookup[(method, group, model, fold)] for model in MODEL_IDS}
            if group == "P0":
                title = f"Method A P0 identity — original unsupervised C5 coordinates — no labels used | Fold {fold:02}"
            elif group == "S0":
                title = f"Method B S0 — unmodified distances — random-init unsupervised t-SNE — no labels used | Fold {fold:02}"
            elif method == "A":
                title = f"Method A — Post-hoc Label-Guided Transformation — LABEL-GUIDED / EXPLORATORY | {group} | Fold {fold:02}"
            else:
                title = f"Method B — Supervised t-SNE (True Labels Used) — SUPERVISED / EXPLORATORY | {group} | Fold {fold:02}"
            if group in {"N1", "N2"}:
                title += " | Shuffled-Label Control"
            plot_grid(cmap, labels, title, FORMAL_FIGURES / f"{method}_{group}_fold{fold:02}.png", 600)
            c5map = c5_maps[fold]
            pair_label = "LABEL-GUIDED / EXPLORATORY" if method == "A" and group != "P0" else ("SUPERVISED / EXPLORATORY" if method == "B" and group != "S0" else "UNSUPERVISED IDENTITY OR RANDOM-INIT REFERENCE")
            plot_pair(c5map, cmap, labels, f"C5 vs {method}/{group} | Fold {fold:02} | {pair_label}", PAIRED_FIGURES / f"{method}_{group}_fold{fold:02}.png")

    metric_names = ["silhouette_2d", "asd_within_distance", "td_within_distance", "within_mean_distance", "normalized_within", "between_distance", "between_within_ratio", "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30", "trustworthiness_15", "trustworthiness_30", "pca_2d_neighbor_overlap_15", "pca_2d_neighbor_overlap_30"]
    paired = []
    row_map = {(row["method"], row["group"], row["model"], int(row["fold"])): row for row in formal_rows}
    for method, groups in (("A", ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "N1"]), ("B", ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "N2"])):
        for group in groups:
            for model in MODEL_IDS:
                for fold in range(1, 11):
                    reference = row_map[("C5", "C5", model, fold)]
                    target = row_map[(method, group, model, fold)]
                    item = {"comparison": f"C5_vs_{method}_{group}", "model": model, "fold": fold, "reference_method": "C5", "reference_group": "C5", "target_method": method, "target_group": group}
                    for metric in metric_names:
                        item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                    paired.append(item)
    for model in MODEL_IDS:
        for fold in range(1, 11):
            for target_group in [f"S{i}" for i in range(1, 10)]:
                reference = row_map[("B", "S0", model, fold)]
                target = row_map[("B", target_group, model, fold)]
                item = {"comparison": f"S0_vs_{target_group}", "model": model, "fold": fold, "reference_method": "B", "reference_group": "S0", "target_method": "B", "target_group": target_group}
                for metric in metric_names:
                    item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                paired.append(item)
            for p_group in [f"P{i}" for i in range(10)]:
                for s_group in [f"S{i}" for i in range(10)]:
                    reference = row_map[("A", p_group, model, fold)]
                    target = row_map[("B", s_group, model, fold)]
                    item = {"comparison": f"{p_group}_vs_{s_group}", "model": model, "fold": fold, "reference_method": "A", "reference_group": p_group, "target_method": "B", "target_group": s_group}
                    for metric in metric_names:
                        item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                    paired.append(item)
            for p_group in [f"P{i}" for i in range(10)]:
                for s_group in [f"S{i}" for i in range(10)]:
                    reference = row_map[("A", p_group, model, fold)]
                    target = row_map[("B", s_group, model, fold)]
                    item = {"comparison": f"{p_group}_vs_{s_group}", "model": model, "fold": fold, "reference_method": "A", "reference_group": p_group, "target_method": "B", "target_group": s_group}
                    for metric in metric_names:
                        item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                    paired.append(item)
            for ref_method, ref_group, target_method, target_group, name in (("A", "P7", "A", "N1", "P7_vs_N1"), ("B", "S6", "B", "N2", "S6_vs_N2")):
                reference = row_map[(ref_method, ref_group, model, fold)]
                target = row_map[(target_method, target_group, model, fold)]
                item = {"comparison": name, "model": model, "fold": fold, "reference_method": ref_method, "reference_group": ref_group, "target_method": target_method, "target_group": target_group}
                for metric in metric_names:
                    item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                paired.append(item)
    for method, group in [("C5", "C5"), *[("A", group) for group in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "N1"]], *[("B", group) for group in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "N2"]]]:
        for fold in range(1, 11):
            reference = row_map[(method, group, "CAM_only", fold)]
            for other in MODEL_IDS[1:]:
                target = row_map[(method, group, other, fold)]
                item = {"comparison": f"CAM_only_to_{other}_{method}_{group}", "model": other, "reference_model": "CAM_only", "fold": fold, "reference_method": method, "reference_group": group, "target_method": method, "target_group": group}
                for metric in metric_names:
                    item[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                paired.append(item)
    with (RESULTS / "paired_comparisons.csv").open("w", newline="") as handle:
        fields = list(paired[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    cam_status = json.loads((RESULTS / "cam_only_rebuild/cam_only_rebuild_status.json").read_text())
    generate_report(formal_rows, scan_rows, paired, c5_rows, contexts, cam_status)
    run_manifest = {"status": "RUN_COMPLETE_PENDING_AUDIT", "models": MODEL_IDS, "folds": 10, "scan_rows": len(scan_rows), "formal_rows": len(formal_rows), "c5_rows": len(c5_rows), "supervised_tsne_jobs": len(jobs), "shuffle_mapping_sha256": __import__("run_experiment").sha256_file(INPUTS / "shuffled_label_mapping.csv"), "elapsed_seconds": time.monotonic() - started}
    (RESULTS / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
