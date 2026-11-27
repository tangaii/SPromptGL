#!/usr/bin/env python3
"""Independent integrity and metric audit for the five-model experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
INPUTS = EXP / "inputs"
RESULTS = EXP / "results"
COORDS = RESULTS / "coordinates"
HIST = Path("/model/spromptgl/experiment/experiment_dresp_tsne_compact_separation")
MODEL_IDS = ["CAM_only", "SPromptGL", "DARG", "DARG_JSL", "DReSP"]
sys.path.insert(0, str(EXP / "code"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def store_rows(path: Path, rows):
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def input_path(model, fold):
    if model == "CAM_only":
        return INPUTS / "cam_only_features" / f"CAM_only_fold{fold:02}_features.npz"
    return INPUTS / f"{model}_fold{fold:02}_features.npz"


def c5_path(model, fold):
    return COORDS / f"C5_{model}_fold{fold:02}.npy"


def load_features(model, fold):
    with np.load(input_path(model, fold), allow_pickle=False) as archive:
        return np.asarray(archive["features"], dtype=np.float64), np.asarray(archive["labels"], dtype=np.int64), np.asarray(archive["patient_ids"], dtype=np.int64)


def load_coord(row):
    path = EXP / row["coordinate_path"]
    coordinates = np.load(path, allow_pickle=False)
    if sha256_file(path) != row["coordinate_sha256"]:
        raise AssertionError(f"coordinate SHA mismatch: {path}")
    return coordinates.astype(np.float64), path


def shuffled_labels(labels):
    permutation = np.random.default_rng(2027).permutation(len(labels))
    return labels[permutation], permutation


def independent_distance_audit(model, fold, s, gamma, labels_for_distance, pca50, expected_scale):
    from scipy.spatial.distance import cdist

    distance = cdist(pca50, pca50, metric="euclidean")
    upper = distance[np.triu_indices(distance.shape[0], k=1)]
    positive = upper[upper > 0]
    scale = float(np.median(positive))
    if abs(scale - float(expected_scale)) > 1e-10:
        raise AssertionError(f"D_scale mismatch for {model} fold {fold}")
    same = labels_for_distance[:, None] == labels_for_distance[None, :]
    adjusted = np.empty_like(distance)
    adjusted[same] = s * distance[same]
    adjusted[~same] = distance[~same] + gamma * scale
    adjusted = (adjusted + adjusted.T) / 2.0
    np.fill_diagonal(adjusted, 0.0)
    if not np.isfinite(adjusted).all() or np.any(adjusted < 0):
        raise AssertionError(f"invalid distance matrix for {model} fold {fold}")
    if not np.allclose(adjusted, adjusted.T, rtol=0, atol=1e-12):
        raise AssertionError(f"asymmetric distance matrix for {model} fold {fold}")
    return adjusted, sha256_bytes(np.ascontiguousarray(adjusted, dtype=np.float64).tobytes())


def independent_metrics(coordinates, labels, pca50, high_dimensional_probabilities):
    from scipy.spatial.distance import cdist, pdist
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    class0 = coordinates[labels == 0]
    class1 = coordinates[labels == 1]
    within0 = float(pdist(class0).mean())
    within1 = float(pdist(class1).mean())
    within_mean = float((within0 + within1) / 2)
    cross = cdist(class0, class1, metric="euclidean")
    centers = np.vstack([class0.mean(axis=0), class1.mean(axis=0)])
    center_distance = float(np.linalg.norm(centers[0] - centers[1]))
    all_pair = pdist(coordinates)
    median = float(np.median(all_pair[all_pair > 0]))
    global_mean = float(all_pair.mean())
    covariance0 = np.cov(class0, rowvar=False)
    covariance1 = np.cov(class1, rowvar=False)
    low_dimensional_sq_distances = pdist(coordinates, metric="sqeuclidean")
    q = 1.0 / (1.0 + low_dimensional_sq_distances)
    q /= q.sum()
    p = np.asarray(high_dimensional_probabilities, dtype=np.float64)
    p /= p.sum()
    original_space_kl = float(np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))))

    def knn(points, k):
        model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(points)
        indices = model.kneighbors(points, return_distance=False)
        return indices[:, 1:]

    metric = {
        "silhouette_2d": float(silhouette_score(coordinates, labels, metric="euclidean")),
        "asd_within_distance": within0,
        "td_within_distance": within1,
        "within_mean_distance": within_mean,
        "global_mean_distance": global_mean,
        "global_median_distance": median,
        "normalized_within": within_mean / median,
        "between_distance": center_distance,
        "between_average_distance": float(cross.mean()),
        "between_within_ratio": center_distance / within_mean,
        "between_average_within_ratio": float(cross.mean()) / within_mean,
        "original_space_kl_divergence": original_space_kl,
        "asd_cov_xx": float(covariance0[0, 0]),
        "asd_cov_xy": float(covariance0[0, 1]),
        "asd_cov_yy": float(covariance0[1, 1]),
        "td_cov_xx": float(covariance1[0, 0]),
        "td_cov_xy": float(covariance1[0, 1]),
        "td_cov_yy": float(covariance1[1, 1]),
        "asd_distribution_scale": float(np.sqrt(np.trace(covariance0))),
        "td_distribution_scale": float(np.sqrt(np.trace(covariance1))),
        "x_min": float(coordinates[:, 0].min()),
        "x_max": float(coordinates[:, 0].max()),
        "y_min": float(coordinates[:, 1].min()),
        "y_max": float(coordinates[:, 1].max()),
        "nonfinite_coordinates": int((~np.isfinite(coordinates)).sum()),
        "n_patients": len(coordinates),
        "asd_count": int((labels == 0).sum()),
        "td_count": int((labels == 1).sum()),
        "trustworthiness_15": float(trustworthiness(pca50, coordinates, n_neighbors=15, metric="euclidean")),
        "trustworthiness_30": float(trustworthiness(pca50, coordinates, n_neighbors=30, metric="euclidean")),
    }
    for k in (15, 30):
        embedded = knn(coordinates, k)
        raw = knn(pca50, k)
        metric[f"cross_class_neighbor_rate_{k}"] = float(np.mean(labels[embedded] != labels[:, None]))
        metric[f"pca_2d_neighbor_overlap_{k}"] = float(np.mean([len(set(a).intersection(b)) / k for a, b in zip(embedded, raw)]))
    return metric


def main():
    checks = {}
    errors = []

    # Verify copied historical four-model archives and C5 source files.
    source_manifest = json.loads((HIST / "inputs/source_manifest.json").read_text())
    expected_inputs = {item["file"]: item["sha256"] for item in source_manifest["files"]}
    for filename, expected_sha in expected_inputs.items():
        target = INPUTS / filename
        if sha256_file(target) != expected_sha:
            errors.append(f"historical input SHA mismatch {filename}")
    checks["historical_40_feature_archive_sha"] = len(expected_inputs) == 40 and not any("historical input SHA" in x for x in errors)

    historical_coord_root = HIST / "results/formal_coordinates"
    copied_coords = INPUTS / "c5_coordinates"
    c5_exact = True
    for model in ["SPromptGL", "DARG", "DARG_JSL", "DReSP"]:
        for fold in range(1, 11):
            name = f"{model}_fold{fold:02}.npy"
            if sha256_file(copied_coords / name) != sha256_file(historical_coord_root / name):
                c5_exact = False
                errors.append(f"historical C5 coordinate source SHA mismatch {name}")
    checks["historical_C5_40_source_file_SHA"] = c5_exact

    # Confirm all five models, ten folds, labels, IDs, and CAM-only archive identity.
    reference_ids = None
    reference_labels = None
    input_shapes = True
    for model in MODEL_IDS:
        for fold in range(1, 11):
            features, labels, patient_ids = load_features(model, fold)
            input_shapes &= features.shape == (871, 144) and labels.shape == (871,) and patient_ids.shape == (871,)
            input_shapes &= np.isfinite(features).all() and len(np.unique(patient_ids)) == 871
            if reference_ids is None:
                reference_ids = patient_ids.copy()
                reference_labels = labels.copy()
            input_shapes &= np.array_equal(patient_ids, reference_ids) and np.array_equal(labels, reference_labels)
    checks["five_model_input_identity_and_order"] = bool(input_shapes)

    cam_status_path = RESULTS / "cam_only_rebuild/cam_only_rebuild_status.json"
    cam_status = json.loads(cam_status_path.read_text())
    cam_pass = cam_status["status"] == "PASS_10_OF_10" and len(cam_status["feature_archives"]) == 10
    for fold in range(1, 11):
        archive = INPUTS / "cam_only_features" / f"CAM_only_fold{fold:02}_features.npz"
        with np.load(archive, allow_pickle=False) as item:
            cam_pass &= item["features"].shape == (871, 144)
            cam_pass &= np.array_equal(item["features"], item["cam"])
            cam_pass &= np.isfinite(item["features"]).all()
            cam_pass &= np.array_equal(item["patient_ids"], np.arange(871))
            cam_pass &= np.bincount(item["labels"]).tolist() == [403, 468]
        row = json.loads((RESULTS / f"cam_only_rebuild/fold{fold:02}_identity.json").read_text())
        cam_pass &= row["status"] == "PASS" and row["hook_output_equal_without_hook"]
        cam_pass &= row["model_parameter_count"] == 49544 and row["cam_shape"] == [871, 144]
        cam_pass &= all(row["historical_identity_checks"].values())
        cam_pass &= row["feature_archive_sha256"] == sha256_file(archive)
    checks["CAM_only_E43_4_E0_rebuild_and_hook"] = bool(cam_pass)
    if not cam_pass:
        errors.append("CAM-only checkpoint identity, forward hook, or feature audit failed")

    # Verify immutable lineage records for all five independent model sources.
    provenance_path = INPUTS / "model_checkpoint_provenance.json"
    provenance_pass = provenance_path.is_file()
    if provenance_pass:
        provenance = json.loads(provenance_path.read_text())
        provenance_pass &= provenance.get("models_have_independent_lineage") is True
        for model in ["SPromptGL", "DARG", "DARG_JSL", "DReSP"]:
            records = provenance.get("models", {}).get(model, {}).get("checkpoint_records", [])
            provenance_pass &= len(records) == 10
            for record in records:
                source = Path(record["path"])
                provenance_pass &= source.is_file() and sha256_file(source) == record["sha256"]
        cam_records = provenance.get("models", {}).get("CAM_only", {}).get("checkpoint_records", [])
        provenance_pass &= len(cam_records) == 10
        for record in cam_records:
            source = Path(record["anchor_path"])
            provenance_pass &= source.is_file() and sha256_file(source) == record["anchor_sha256"]
    checks["five_model_independent_checkpoint_lineage_SHA"] = bool(provenance_pass)
    if not provenance_pass:
        errors.append("five-model checkpoint lineage or source SHA verification failed")

    # Reconstruct the single deterministic patient-level shuffled label map.
    labels = reference_labels
    shuffled, permutation = shuffled_labels(labels)
    map_path = INPUTS / "shuffled_label_mapping.csv"
    map_rows = load_rows(map_path)
    shuffle_pass = len(map_rows) == 871 and np.bincount(shuffled).tolist() == [403, 468]
    shuffle_pass &= all(int(row["true_label"]) == int(labels[int(row["patient_id"])]) for row in map_rows)
    shuffle_pass &= all(int(row["shuffled_label"]) == int(shuffled[int(row["patient_id"])]) for row in map_rows)
    shuffle_pass &= all(int(row["source_patient_index"]) == int(permutation[int(row["patient_id"])]) for row in map_rows)
    checks["fixed_shuffle_seed_2027_membership_and_counts"] = bool(shuffle_pass)

    # Route A formula, exact C5 pass-through, and every persisted metric.
    formal_path = RESULTS / "formal_metrics.csv"
    scan_path = RESULTS / "parameter_scan.csv"
    formal_rows = load_rows(formal_path)
    scan_rows = load_rows(scan_path)
    formula_pass = True
    metric_pass = True
    distance_audit = []
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    all_result_rows = formal_rows + scan_rows
    refreshed_rows = []
    probability_cache = {}
    for row in all_result_rows:
        model = row["model"]
        fold = int(row["fold"])
        features, true_labels, patient_ids = load_features(model, fold)
        scaled = StandardScaler().fit_transform(features)
        pca50 = PCA(n_components=50, random_state=0).fit_transform(scaled)
        coordinates, path = load_coord(row)
        if row["method"] == "A" and float(row["alpha"]) == 1.0 and float(row["beta"]) == 0.0:
            row["labels_used"] = "none_identity"
            row["supervision_type"] = "UNSUPERVISED_C5_REFERENCE"
        if row["method"] == "B" and float(row["s"]) == 1.0 and float(row["gamma"]) == 0.0:
            row["labels_used"] = "none_distance_identity"
            row["supervision_type"] = "UNSUPERVISED_RANDOM_INIT_DISTANCE_IDENTITY"
        if row["group"] in {"N1", "N2"}:
            row["supervision_type"] = "SHUFFLED_LABEL_CONTROL_EXPLORATORY"
        if row["method"] == "A":
            alpha = float(row["alpha"])
            beta = float(row["beta"])
            label_source = row["labels_used"]
            used_labels = shuffled if label_source.startswith("shuffled") else true_labels
            original = np.load(c5_path(model, fold), allow_pickle=False).astype(np.float64)
            if alpha == 1.0 and beta == 0.0:
                expected = original.astype(np.float32)
            else:
                global_center = original.mean(axis=0)
                expected = np.empty_like(original)
                for label in (0, 1):
                    mask = used_labels == label
                    center = original[mask].mean(axis=0)
                    expected[mask] = center + alpha * (original[mask] - center) + beta * (center - global_center)
                expected = expected.astype(np.float32)
            formula_match = np.array_equal(np.load(path, allow_pickle=False), expected)
            if not formula_match and row["group"] != "P0":
                formula_pass = False
        if row["method"] == "B":
            label_source = row["labels_used"]
            used_labels = shuffled if label_source.startswith("shuffled") else true_labels
            if label_source == "none_distance_identity":
                used_labels = np.zeros_like(true_labels)
            _adjusted, distance_sha = independent_distance_audit(
                model, fold, float(row["s"]), float(row["gamma"]), used_labels, pca50, float(row["d_scale"])
            )
            distance_audit.append({
                "model": model,
                "fold": fold,
                "group": row["group"],
                "s": row["s"],
                "gamma": row["gamma"],
                "labels_used": label_source,
                "d_scale": row["d_scale"],
                "distance_matrix_sha256": distance_sha,
                "symmetric": True,
                "diagonal_zero": True,
                "finite_nonnegative": True,
            })
            row["distance_matrix_audit_sha256"] = distance_sha
        probability_key = (model, fold)
        if probability_key not in probability_cache:
            from sklearn.manifold._t_sne import _joint_probabilities
            from scipy.spatial.distance import cdist
            high_dimensional_sq_distances = cdist(pca50, pca50, metric="sqeuclidean")
            probability_cache[probability_key] = _joint_probabilities(high_dimensional_sq_distances, desired_perplexity=100, verbose=0)
        independently = independent_metrics(coordinates, true_labels, pca50, probability_cache[probability_key])
        for metric_name, value in independently.items():
            if metric_name in row:
                try:
                    stored = float(row[metric_name])
                    if not math.isclose(stored, float(value), rel_tol=2e-6, abs_tol=2e-6):
                        metric_pass = False
                        errors.append(f"metric mismatch {model} fold{fold:02} {row['method']}/{row['group']} {metric_name}")
                except (ValueError, TypeError):
                    metric_pass = False
            row[metric_name] = value
        row["raw_reference"] = "unmodified PCA50"
        refreshed_rows.append(row)
    checks["Method_A_formula_recomputation"] = bool(formula_pass)
    if not formula_pass:
        errors.append("Method A coordinate formula mismatch")
    checks["Method_B_distance_recomputation_and_validity"] = bool(len(distance_audit) > 0 and all(item["symmetric"] and item["diagonal_zero"] and item["finite_nonnegative"] for item in distance_audit))
    checks["independent_metric_recomputation"] = bool(metric_pass)
    if not metric_pass:
        pass

    # P0 must equal C5 exactly. Correct the persisted coordinate and record if serialization differed.
    p0_equal = True
    p0_rows = [row for row in refreshed_rows if row["method"] == "A" and row["group"] == "P0"]
    for row in p0_rows:
        model = row["model"]
        fold = int(row["fold"])
        source = np.load(c5_path(model, fold), allow_pickle=False)
        path = EXP / row["coordinate_path"]
        plotted = np.load(path, allow_pickle=False)
        if not np.array_equal(plotted, source):
            p0_equal = False
            np.save(path, source.astype(np.float32))
            row["coordinate_sha256"] = sha256_file(path)
            row["p0_exact_c5_repaired_during_audit"] = True
        else:
            row["p0_exact_c5_repaired_during_audit"] = False
        features, true_labels, _ = load_features(row["model"], int(row["fold"]))
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        pca50 = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(features))
        row.update(independent_metrics(source.astype(np.float64), true_labels, pca50, probability_cache[(row["model"], int(row["fold"]))]))
    p0_verified_after = all(
        np.array_equal(np.load(EXP / row["coordinate_path"], allow_pickle=False), np.load(c5_path(row["model"], int(row["fold"])), allow_pickle=False))
        for row in p0_rows
    )
    checks["P0_exact_C5_identity"] = bool(p0_verified_after)
    if not p0_verified_after:
        errors.append("P0 does not exactly equal C5 after audit correction")
    replay_baseline_path = RESULTS / "interruption_coordinate_sha256.json"
    replay_baseline = json.loads(replay_baseline_path.read_text())["file_sha256"]
    replay_pass = True
    for relative_path, expected_sha in replay_baseline.items():
        path = EXP / relative_path
        replay_pass &= path.is_file() and sha256_file(path) == expected_sha
    checks["deterministic_resume_coordinate_SHA"] = bool(replay_pass)
    if not replay_pass:
        errors.append("deterministic coordinate replay differs from persisted pre-interruption files")
    if not p0_equal:
        from run_experiment import plot_grid, plot_pair
        for fold in range(1, 11):
            c5_coords = {model: np.load(c5_path(model, fold), allow_pickle=False) for model in MODEL_IDS}
            with np.load(input_path("CAM_only", fold), allow_pickle=False) as archive:
                labels_for_plot = np.asarray(archive["labels"], dtype=np.int64)
            plot_grid(c5_coords, labels_for_plot, f"Method A P0 identity — original unsupervised C5 coordinates — no labels used | Fold {fold:02}", RESULTS / f"figures/formal/A_P0_fold{fold:02}.png", 600)
            plot_pair(c5_coords, c5_coords, labels_for_plot, f"C5 vs A/P0 | Fold {fold:02} | UNSUPERVISED IDENTITY", RESULTS / f"figures/paired/A_P0_fold{fold:02}.png")
    store_rows(formal_path, [row for row in refreshed_rows if row in formal_rows])
    store_rows(scan_path, [row for row in refreshed_rows if row in scan_rows])
    store_rows(RESULTS / "distance_matrix_audit.csv", distance_audit)
    paired_path = RESULTS / "paired_comparisons.csv"
    paired_rows = load_rows(paired_path)
    formal_lookup = {(row["method"], row["group"], row["model"], int(row["fold"])): row for row in formal_rows}
    for row in paired_rows:
        ref_key = (row["reference_method"], row["reference_group"], row["model"], int(row["fold"]))
        target_key = (row["target_method"], row["target_group"], row["model"], int(row["fold"]))
        if ref_key not in formal_lookup or target_key not in formal_lookup:
            continue
        for metric in ("between_average_distance", "between_average_within_ratio", "original_space_kl_divergence"):
            row[f"delta_{metric}"] = float(formal_lookup[target_key][metric]) - float(formal_lookup[ref_key][metric])
    store_rows(paired_path, paired_rows)

    # Formal group/fold counts, coordinate integrity, and complete patient coverage.
    expected_formal = 50 + (11 * 5 * 10) + (11 * 5 * 10)
    complete_groups = len(formal_rows) == expected_formal and len(scan_rows) == 500
    expected_keys = {("C5", "C5", model, fold) for model in MODEL_IDS for fold in range(1, 11)}
    expected_keys |= {(method, group, model, fold) for method, groups in (("A", [f"P{i}" for i in range(10)] + ["N1"]), ("B", [f"S{i}" for i in range(10)] + ["N2"])) for group in groups for model in MODEL_IDS for fold in range(1, 11)}
    actual_keys = {(row["method"], row["group"], row["model"], int(row["fold"])) for row in formal_rows}
    complete_groups &= expected_keys == actual_keys
    unique_coordinate_paths = {row["coordinate_path"] for row in formal_rows + scan_rows}
    complete_groups &= len(unique_coordinate_paths) == 1550
    checks["formal_and_scan_row_completeness"] = bool(complete_groups)
    if not complete_groups:
        errors.append(f"formal/scan completeness mismatch formal={len(formal_rows)} scan={len(scan_rows)}")

    # Population mean and SD are independently regenerated across ten folds.
    from collections import defaultdict

    summary = defaultdict(list)
    for row in formal_rows:
        summary[(row["method"], row["group"], row["model"])].append(row)
    aggregate_rows = []
    for key, rows in sorted(summary.items()):
        item = {"method": key[0], "group": key[1], "model": key[2], "n_folds": len(rows)}
        for metric in ("silhouette_2d", "silhouette_raw144", "silhouette_pca50", "asd_within_distance", "td_within_distance", "within_mean_distance", "between_distance", "between_average_distance", "between_within_ratio", "trustworthiness_15", "trustworthiness_30", "kl_divergence", "original_space_kl_divergence"):
            vals = np.asarray([float(row[metric]) for row in rows if row.get(metric, "") not in ("", "nan", "NaN")], dtype=np.float64)
            if len(vals):
                item[f"{metric}_mean"] = float(vals.mean())
                item[f"{metric}_population_sd"] = float(vals.std(ddof=0))
        if key[0] != "C5" and len(rows) != 10:
            errors.append(f"ten-fold summary count mismatch {key}: {len(rows)}")
        aggregate_rows.append(item)
    store_rows(RESULTS / "formal_summary.csv", aggregate_rows)
    checks["ten_fold_population_sd_recomputed"] = not any("ten-fold summary count mismatch" in error for error in errors)

    # Image inventory and raster metadata.
    scan_pngs = list((RESULTS / "figures/scan").glob("*.png"))
    formal_pngs = list((RESULTS / "figures/formal").glob("*.png"))
    paired_pngs = list((RESULTS / "figures/paired").glob("*.png"))
    png_pass = len(scan_pngs) == 100 and len(formal_pngs) == 230 and len(paired_pngs) == 220
    forbidden = [p.name for p in EXP.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".svg", ".eps"}]
    png_pass &= not forbidden
    try:
        from PIL import Image
        for path in formal_pngs + paired_pngs:
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))[0]
                png_pass &= abs(float(dpi) - 600) < 2
        for path in scan_pngs:
            with Image.open(path) as image:
                dpi = image.info.get("dpi", (0, 0))[0]
                png_pass &= abs(float(dpi) - 300) < 2
    except Exception as error:
        png_pass = False
        errors.append(f"PNG metadata check failed: {error!r}")
    checks["PNG_inventory_DPI_and_formats"] = bool(png_pass)
    if not png_pass:
        errors.append(f"PNG inventory/DPI mismatch: scan={len(scan_pngs)}, formal={len(formal_pngs)}, paired={len(paired_pngs)}")

    # Protected root/model inputs and historical files must still match source manifests.
    original_manifest = json.loads((INPUTS / "cam_only_source/manifest.json").read_text())
    protected = original_manifest["protected_sha256"]
    protected_pass = all(Path(path).is_file() and sha256_file(Path(path)) == expected for path, expected in protected.items())
    checks["protected_root_model_data_SHA"] = bool(protected_pass)
    if not protected_pass:
        errors.append("a protected root/model/data SHA changed during this experiment")

    # No manifest self-SHA and no unsupported accidental temporary outputs.
    manifest_path = EXP / "manifest.json"
    manifest_pass = not manifest_path.exists() or "self_sha256" not in json.loads(manifest_path.read_text())
    checks["manifest_has_no_self_SHA"] = bool(manifest_pass)

    checks["coordinate_finiteness_and_patient_coverage"] = all(
        int(row["n_patients"]) == 871 and int(row["nonfinite_coordinates"]) == 0 and int(row["asd_count"]) == 403 and int(row["td_count"]) == 468
        for row in formal_rows + scan_rows
    )

    audit = {
        "status": "COMPLETE_AUDITED" if all(checks.values()) and not errors else "AUDIT_FAILED",
        "checks": checks,
        "errors": errors,
        "counts": {
            "models": 5,
            "folds": 10,
            "historical_input_archives": len(expected_inputs),
            "formal_metric_rows": len(formal_rows),
            "scan_metric_rows": len(scan_rows),
            "distance_matrix_audit_rows": len(distance_audit),
            "scan_pngs": len(scan_pngs),
            "formal_pngs": len(formal_pngs),
            "paired_pngs": len(paired_pngs),
        },
        "p0_exact_c5": bool(p0_equal),
        "shuffle_mapping_sha256": sha256_file(map_path),
    }
    (RESULTS / "audit_status.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if audit["status"] != "COMPLETE_AUDITED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
