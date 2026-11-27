#!/usr/bin/env python3
"""Five-model, label-guided t-SNE exploration.

The C5 coordinates for the four historical models are copied byte-for-byte
from the prior formal run. CAM-only C5 is replayed from its independently
reconstructed CAM144. All coordinates produced here are explicitly marked as
unsupervised C5, label-guided exploratory, or supervised exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path("/data/adodas/SPromptGL-main")
EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
INPUTS = EXP / "inputs"
COORDS = EXP / "results/coordinates"
FIGURES = EXP / "results/figures"
SCAN_FIGURES = FIGURES / "scan"
FORMAL_FIGURES = FIGURES / "formal"
PAIRED_FIGURES = FIGURES / "paired"
HIST = Path("/model/spromptgl/experiment/experiment_dresp_tsne_compact_separation")
MODEL_IDS = ["CAM_only", "SPromptGL", "DARG", "DARG_JSL", "DReSP"]
MODEL_LABELS = {
    "CAM_only": "CAM-only",
    "SPromptGL": "SPromptGL",
    "DARG": "DARG",
    "DARG_JSL": "DARG + JSL",
    "DReSP": "DReSP",
}
COLORS = {0: "#D98282", 1: "#4C9A91"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fit_tsne(payload):
    from sklearn.manifold import TSNE

    pca50 = payload["pca50"]
    kwargs = dict(
        n_components=2,
        perplexity=100,
        early_exaggeration=48,
        learning_rate=200,
        metric=payload["metric"],
        init=payload["init"],
        random_state=0,
        verbose=0,
    )
    kwargs["n_iter"] = 3000
    started = time.monotonic()
    model = TSNE(**kwargs)
    coordinates = model.fit_transform(pca50).astype(np.float32)
    return coordinates, float(model.kl_divergence_), time.monotonic() - started


def fit_supervised_tsne(payload):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances

    pca50 = np.asarray(payload["pca50"], dtype=np.float64)
    used_labels = np.asarray(payload["used_labels"], dtype=np.int64)
    distances = pairwise_distances(pca50, metric="euclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0]
    scale = float(np.median(positive))
    same = used_labels[:, None] == used_labels[None, :]
    adjusted = np.where(same, payload["s"] * distances, distances + payload["gamma"] * scale)
    adjusted = np.asarray((adjusted + adjusted.T) / 2.0, dtype=np.float64)
    np.fill_diagonal(adjusted, 0.0)
    if not np.isfinite(adjusted).all() or np.any(adjusted < 0):
        raise ValueError("invalid supervised distance matrix")
    started = time.monotonic()
    model = TSNE(
        n_components=2,
        perplexity=100,
        early_exaggeration=48,
        learning_rate=200,
        metric="precomputed",
        init="random",
        random_state=0,
        n_iter=3000,
        verbose=0,
    )
    coordinates = model.fit_transform(adjusted).astype(np.float32)
    return coordinates, float(model.kl_divergence_), time.monotonic() - started, scale


def input_path(model: str, fold: int) -> Path:
    if model == "CAM_only":
        return INPUTS / "cam_only_features" / f"CAM_only_fold{fold:02}_features.npz"
    return INPUTS / f"{model}_fold{fold:02}_features.npz"


def c5_source_path(model: str, fold: int) -> Path:
    return INPUTS / "c5_coordinates" / f"{model}_fold{fold:02}.npy"


def load_input(model: str, fold: int):
    path = input_path(model, fold)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "patient_ids"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing input keys in {path}")
        features = np.asarray(archive["features"])
        labels = np.asarray(archive["labels"], dtype=np.int64)
        patient_ids = np.asarray(archive["patient_ids"], dtype=np.int64)
    if features.shape != (871, 144) or labels.shape != (871,) or patient_ids.shape != (871,):
        raise ValueError(f"invalid input shape for {model} fold {fold}")
    if len(np.unique(patient_ids)) != 871 or not np.isfinite(features).all():
        raise ValueError(f"invalid patient IDs or non-finite values for {model} fold {fold}")
    if not np.array_equal(np.bincount(labels), np.array([403, 468])):
        raise ValueError(f"invalid labels for {model} fold {fold}")
    return features.astype(np.float64), labels, patient_ids, sha256_file(path)


def save_coord(path: Path, coordinates: np.ndarray):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (871, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"invalid coordinate array for {path}")
    np.save(path, coordinates)
    return sha256_file(path)


def transform_coordinates(original, labels_used, alpha, beta):
    original = np.asarray(original, dtype=np.float64)
    if alpha == 1.0 and beta == 0.0:
        return original.astype(np.float32, copy=True)
    labels_used = np.asarray(labels_used, dtype=np.int64)
    global_center = original.mean(axis=0)
    transformed = np.empty_like(original)
    for label in (0, 1):
        mask = labels_used == label
        center = original[mask].mean(axis=0)
        transformed[mask] = center + alpha * (original[mask] - center) + beta * (center - global_center)
    return transformed.astype(np.float32)


def neighbors(points, k):
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(points).kneighbors(points, return_distance=False)[:, 1:]


def base_context(raw_features, labels, pca50, c5_kl):
    from sklearn.metrics import silhouette_score

    return {
        "silhouette_raw144": float(silhouette_score(raw_features, labels, metric="euclidean")),
        "silhouette_standardized144": float(silhouette_score((raw_features - raw_features.mean(0)) / np.maximum(raw_features.std(0), 1e-12), labels)),
        "silhouette_pca50": float(silhouette_score(pca50, labels, metric="euclidean")),
        "pca_neighbors_15": neighbors(pca50, 15),
        "pca_neighbors_30": neighbors(pca50, 30),
        "c5_kl_divergence": float(c5_kl),
    }


def geometry_metrics(coordinates, labels, pca50, base):
    from scipy.spatial.distance import cdist, pdist
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score

    coordinates = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = [coordinates[labels == label] for label in (0, 1)]
    within = [float(pdist(points).mean()) for points in classes]
    mean_within = float(np.mean(within))
    all_distances = pdist(coordinates)
    global_mean = float(all_distances.mean())
    global_median = float(np.median(all_distances[all_distances > 0]))
    centers = [points.mean(axis=0) for points in classes]
    between = float(np.linalg.norm(centers[0] - centers[1]))
    between_average = float(cdist(classes[0], classes[1], metric="euclidean").mean())
    ratio = between / mean_within if mean_within else math.inf
    covariances = [np.cov(points, rowvar=False) for points in classes]
    scales = [float(np.sqrt(np.trace(covariance))) for covariance in covariances]
    rates = {}
    for k in (15, 30):
        index = neighbors(coordinates, k)
        rates[k] = float(np.mean(labels[index] != labels[:, None]))
    overlap = {}
    for k in (15, 30):
        embedded = neighbors(coordinates, k)
        reference = base[f"pca_neighbors_{k}"]
        overlap[k] = float(np.mean([len(set(a).intersection(b)) / k for a, b in zip(embedded, reference)]))
    return {
        "silhouette_2d": float(silhouette_score(coordinates, labels, metric="euclidean")),
        "asd_within_distance": within[0],
        "td_within_distance": within[1],
        "within_mean_distance": mean_within,
        "global_mean_distance": global_mean,
        "global_median_distance": global_median,
        "normalized_within": mean_within / global_median if global_median else math.inf,
        "asd_center_x": float(centers[0][0]),
        "asd_center_y": float(centers[0][1]),
        "td_center_x": float(centers[1][0]),
        "td_center_y": float(centers[1][1]),
        "between_distance": between,
        "between_average_distance": between_average,
        "between_within_ratio": ratio,
        "between_average_within_ratio": between_average / mean_within if mean_within else math.inf,
        "asd_cov_xx": float(covariances[0][0, 0]),
        "asd_cov_xy": float(covariances[0][0, 1]),
        "asd_cov_yy": float(covariances[0][1, 1]),
        "td_cov_xx": float(covariances[1][0, 0]),
        "td_cov_xy": float(covariances[1][0, 1]),
        "td_cov_yy": float(covariances[1][1, 1]),
        "asd_distribution_scale": scales[0],
        "td_distribution_scale": scales[1],
        "cross_class_neighbor_rate_15": rates[15],
        "cross_class_neighbor_rate_30": rates[30],
        "trustworthiness_15": float(trustworthiness(pca50, coordinates, n_neighbors=15, metric="euclidean")),
        "trustworthiness_30": float(trustworthiness(pca50, coordinates, n_neighbors=30, metric="euclidean")),
        "pca_2d_neighbor_overlap_15": overlap[15],
        "pca_2d_neighbor_overlap_30": overlap[30],
        "x_min": float(coordinates[:, 0].min()),
        "x_max": float(coordinates[:, 0].max()),
        "y_min": float(coordinates[:, 1].min()),
        "y_max": float(coordinates[:, 1].max()),
        "nonfinite_coordinates": int((~np.isfinite(coordinates)).sum()),
        "n_patients": int(len(coordinates)),
        "asd_count": int((labels == 0).sum()),
        "td_count": int((labels == 1).sum()),
    }


def make_row(model, fold, method, group, coordinates, context, input_sha, coord_sha, runtime_sec, kl, labels_used_source, params=None, d_scale=None):
    metrics = geometry_metrics(coordinates, context["labels"], context["pca50"], context["base"])
    row = {
        "model": model,
        "model_display": MODEL_LABELS[model],
        "fold": fold,
        "method": method,
        "group": group,
        "labels_used": labels_used_source,
        "supervision_type": (
            "UNSUPERVISED_C5"
            if method == "C5" or labels_used_source in {"none_identity", "none_distance_identity"}
            else ("SHUFFLED_LABEL_CONTROL_EXPLORATORY" if labels_used_source.startswith("shuffled") else ("LABEL-GUIDED_EXPLORATORY" if method == "A" else "SUPERVISED_EXPLORATORY"))
        ),
        "alpha": "",
        "beta": "",
        "s": "",
        "gamma": "",
        "d_scale": "" if d_scale is None else d_scale,
        "input_sha256": input_sha,
        "coordinate_sha256": coord_sha,
        "coordinate_path": "",
        "runtime_sec": runtime_sec,
        "kl_divergence": "" if kl is None else kl,
        "c5_reference_kl_divergence": context["base"]["c5_kl_divergence"],
        "raw_reference": "unmodified PCA50",
        "normalized_within_definition": "mean(ASD within, TD within) / median positive all-pair 2D distance",
    }
    if params:
        row.update(params)
    row.update({key: value for key, value in context["base"].items() if not key.startswith("pca_neighbors_")})
    row.update(metrics)
    return row


def plotting_limits(coord_list, pad=0.08):
    stacked = np.vstack(coord_list)
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    return [(float(lo[i] - span[i] * pad), float(hi[i] + span[i] * pad)) for i in range(2)]


def plot_grid(coord_map, labels, title, path, dpi):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
    axes = axes.ravel()
    for index, model in enumerate(MODEL_IDS):
        axis = axes[index]
        points = coord_map[model]
        limits = plotting_limits([points])
        for label in (0, 1):
            mask = labels == label
            axis.scatter(points[mask, 0], points[mask, 1], s=9, alpha=0.72, c=COLORS[label], edgecolors="none", label="ASD" if label == 0 else "TD")
        axis.set_title(f"({chr(97 + index)}) {MODEL_LABELS[model]}\nn=871", fontsize=11)
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("white")
    axes[5].axis("off")
    handles, labels_text = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels_text, loc="center", frameon=False, title="True label")
    fig.suptitle(title, fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", metadata={"Title": title})
    plt.close(fig)


def plot_pair(c5_map, transformed_map, labels, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), facecolor="white")
    for col, model in enumerate(MODEL_IDS):
        limits = plotting_limits([c5_map[model], transformed_map[model]])
        for row, points in enumerate((c5_map[model], transformed_map[model])):
            axis = axes[row, col]
            for label in (0, 1):
                mask = labels == label
                axis.scatter(points[mask, 0], points[mask, 1], s=9, alpha=0.72, c=COLORS[label], edgecolors="none")
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("white")
            if row == 0:
                axis.set_title(MODEL_LABELS[model], fontsize=10)
        axes[0, col].set_ylabel("C5", fontsize=10)
        axes[1, col].set_ylabel("label-guided", fontsize=10)
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white", metadata={"Title": title})
    plt.close(fig)


def aggregate(rows, group_fields):
    numeric = [
        "silhouette_2d", "silhouette_raw144", "silhouette_standardized144", "silhouette_pca50",
        "asd_within_distance", "td_within_distance", "within_mean_distance", "normalized_within",
        "between_distance", "between_within_ratio", "cross_class_neighbor_rate_15",
        "cross_class_neighbor_rate_30", "trustworthiness_15", "trustworthiness_30",
        "pca_2d_neighbor_overlap_15", "pca_2d_neighbor_overlap_30", "kl_divergence",
    ]
    groups = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n_folds"] = len(members)
        for metric in numeric:
            values = np.asarray([float(item[metric]) for item in members if item.get(metric, "") not in ("", "nan", "NaN")], dtype=float)
            if len(values):
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_population_sd"] = float(values.std(ddof=0))
        output.append(out)
    return output


def generate_report(formal_rows, scan_rows, paired_rows, c5_rows, contexts, cam_status):
    formal_summary = aggregate(formal_rows, ["method", "group", "model"])
    c5_summary = aggregate(c5_rows, ["model"])
    write_csv(EXP / "results/formal_summary.csv", formal_summary)
    write_csv(EXP / "results/raw_c5_comparison.csv", c5_summary)
    write_csv(EXP / "results/paired_comparisons.csv", paired_rows)
    write_csv(EXP / "results/formal_metrics.csv", formal_rows)
    write_csv(EXP / "results/parameter_scan.csv", scan_rows)
    by_key = {(row["method"], row["group"], row["model"]): row for row in formal_summary}
    c5_by_model = {row["model"]: row for row in c5_summary}
    geometry_rows = [row for row in formal_rows if row["method"] in {"A", "B"}]
    tightest = min(geometry_rows, key=lambda row: float(row["within_mean_distance"]))
    widest = max(geometry_rows, key=lambda row: float(row["between_within_ratio"]))
    trust_low = min(geometry_rows, key=lambda row: float(row["trustworthiness_15"]))
    lines = [
        "# DReSP Dual Label-Guided t-SNE Exploration",
        "",
        "## A. STATUS",
        "",
        "COMPLETE_EXPLORATORY_FIVE_MODEL_PENDING_AUDIT",
        "",
        "All coordinates using true or shuffled labels are marked LABEL-GUIDED / SUPERVISED / EXPLORATORY in the result rows and PNG titles. This is a visualization study and reports no new ACC, AUC, SEN, SPE, or F1 claim.",
        "",
        "## B. Input provenance and SHA",
        "",
        "The 40 historical feature archives and 40 historical C5 arrays were copied byte-for-byte from the compact-separation experiment. CAM-only E43.4-E0 features were regenerated from the physical CAM-only network and are stored as ten complete 871-patient NPZ archives.",
        f"CAM-only checkpoint status: `{cam_status.get('checkpoint_identity', 'unknown')}`. The original E43.4-E0 Best binaries were not present in the available archive. The ten E43.1 anchors matched the E43.4 manifest hashes, so the exact E0 protocol was rebuilt and matched historical per-fold predictions and metrics.",
        "",
        "## C. Method definitions",
        "",
        "Method A applies z_new = mu_class + alpha (z_original - mu_class) + beta (mu_class - mu_global) to the historical or replayed C5 coordinates. It is a post-hoc label-guided transformation.",
        "Method B computes StandardScaler then PCA50, forms Euclidean D, scales same-class distances by s and adds gamma times the median positive PCA50 pair distance to different-class distances, then runs TSNE(metric=precomputed, init=random, random_state=0, n_iter=3000). It is supervised t-SNE with labels in the distance input.",
        "",
        "## D. Formal groups",
        "",
        "A uses P0-P9, B uses S0-S9, and shuffled-label controls use N1 and N2. Every group was applied to CAM-only, SPromptGL, DARG, DARG + JSL, and DReSP across all ten folds. C5 is retained as the original unsupervised reference.",
        "",
        "## E. Parameter scan",
        "",
        f"The scan table contains {len(scan_rows)} rows covering 25 Method A configurations and 25 Method B configurations on Fold 7 and Fold 8 for five models. No group was removed based on its score.",
        "",
        "## F. Five-model ten-fold statistics",
        "",
        "`formal_summary.csv` reports mean and population SD across ten folds. `formal_metrics.csv` retains every model, group, and fold. The tightest observed formal geometry row by raw mean within-class distance was "
        f"`{tightest['method']}/{tightest['group']}/{tightest['model']}`. The largest between/within ratio was `" + f"{widest['method']}/{widest['group']}/{widest['model']}`. These extrema are mathematical consequences of the label constraints and are not model-learning gains.",
        "",
        "## G. Paired comparisons",
        "",
        "`paired_comparisons.csv` includes C5 versus every formal group, S0 versus S1-S9, shuffled controls versus their matched true-label settings, and CAM-only versus each other model. C5 versus S0 includes the random-init difference. S0 versus S1-S9 is the cleaner comparison for the distance modification itself.",
        "",
        "## H. Method A versus Method B",
        "",
        "Method A contracts or separates an existing two-dimensional layout by construction. Method B changes the high-dimensional neighborhood relation before t-SNE and has a separate random-init reference S0. Their coordinates are not interchangeable and should not be interpreted on a shared absolute axis across independent model fits.",
        "",
        "## I. Compactness and separation",
        "",
        "Within-class distances, covariance scales, center distance, normalized within distance, between/within ratio, Silhouette, and cross-class neighbor rates are reported together. Lower within distance and higher center distance can arise solely from the prescribed transform, so these numbers describe the visualization geometry.",
        "",
        "## J. Trustworthiness and high-dimensional structure",
        "",
        "Trustworthiness@15/@30 and PCA50-to-2D neighborhood overlap always use the unmodified PCA50 as the reference. The lowest observed formal Trustworthiness@15 row was "
        f"`{trust_low['method']}/{trust_low['group']}/{trust_low['model']}`. This metric is the relevant check for whether a stronger visual constraint sacrifices local structure.",
        "",
        "## K. Label permutation controls",
        "",
        "N1 uses alpha=0.65 and beta=0.50 with a single patient-level permutation seeded 2027. N2 uses s=0.75 and gamma=0.50 with the same permutation. Metrics for N1 and N2 are calculated with true labels. They test how much separation can be created by an external label assignment.",
        "",
        "## L. Ten-fold figures",
        "",
        "PNG files are under `results/figures/`. Scan previews use 300 DPI. Formal and paired figures use 600 DPI. Each five-model grid uses the requested 2x3 layout. All plots contain 871 points and explicit method and label-use text.",
        "",
        "## M. DReSP fold-level performance",
        "",
        "DReSP rows remain available in `formal_metrics.csv` and `paired_comparisons.csv` by fold. Any relative position after label guidance is a visualization comparison. It does not establish a DReSP diagnostic advantage.",
        "",
        "## N. Label-guided bias",
        "",
        "True test-patient labels were used in the visualization. P9 and the strongest B groups are intentionally extreme constraints. Their separation is expected mathematically and cannot validate independent disease representation.",
        "",
        "## O. Scientific interpretation",
        "",
        "The report separates natural C5 geometry, post-hoc label-guided geometry, supervised distance-driven geometry, and neighborhood preservation. A tighter or more separated two-dimensional plot is not evidence of improved ACC, AUC, generalization, or learned diagnostic representation.",
        "",
        "## P. Paper-use limits",
        "",
        "This study uses the real labels of the same 871 patients used to form the input archives. It provides no independent test-set generalization evidence. Any figure used in a paper must retain its method label and the exploratory/supervised qualification.",
        "",
        "## Q. Independent audit",
        "",
        "The independent audit is written to `results/audit_status.json`. The final status is COMPLETE_AUDITED only if the CAM-only identity, source SHA, all coordinates, all rows, label permutation, PNG inventory, and population-SD recomputation pass.",
        "",
        "## R. Files and handoff",
        "",
        "The authoritative directory is `" + str(EXP) + "`. `SHA256SUMS` is generated after all outputs and the single append to `/model/spromptgl/experiment/handoff.md`. The manifest does not contain a self-SHA reference.",
        "",
        "## Selected descriptive values",
        "",
        f"C5 model summaries are in `raw_c5_comparison.csv`. DReSP C5 two-dimensional Silhouette mean is {c5_by_model.get('DReSP', {}).get('silhouette_2d_mean', 'NA')}; CAM-only C5 two-dimensional Silhouette mean is {c5_by_model.get('CAM_only', {}).get('silhouette_2d_mean', 'NA')}.",
    ]
    (EXP / "results/report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    for directory in (COORDS, SCAN_FIGURES, FORMAL_FIGURES, PAIRED_FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    contexts = {}
    c5_rows = []
    c5_maps = {}
    reference_ids = None
    reference_labels = None

    # Load and preprocess every independent model/fold, retaining original order.
    for model in MODEL_IDS:
        for fold in range(1, 11):
            features, labels, patient_ids, input_sha = load_input(model, fold)
            if reference_ids is None:
                reference_ids = patient_ids.copy()
                reference_labels = labels.copy()
            elif not np.array_equal(patient_ids, reference_ids) or not np.array_equal(labels, reference_labels):
                raise ValueError(f"patient order or true labels differ in {model} fold {fold}")
            if model != "CAM_only":
                source = c5_source_path(model, fold)
                coordinates = np.load(source, allow_pickle=False)
                if coordinates.shape != (871, 2) or not np.isfinite(coordinates).all():
                    raise ValueError(f"invalid historical C5 coordinates {source}")
                target = COORDS / f"C5_{model}_fold{fold:02}.npy"
                coord_sha = save_coord(target, coordinates)
                c5_kl_rows = read_csv(HIST / "results/final_metrics.csv")
                matching = [row for row in c5_kl_rows if row["model"] == model and int(row["fold"]) == fold and row["candidate"] == "C5"]
                if not matching:
                    raise ValueError(f"missing historical C5 KL for {model} fold {fold}")
                c5_kl = float(matching[0]["kl_divergence"])
            else:
                coordinates = None
                c5_kl = None
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            scaled = StandardScaler().fit_transform(features)
            pca50 = PCA(n_components=50, random_state=0).fit_transform(scaled)
            contexts[(model, fold)] = {
                "model": model,
                "fold": fold,
                "features": features,
                "labels": labels,
                "patient_ids": patient_ids,
                "input_sha": input_sha,
                "scaled": scaled,
                "pca50": pca50,
                "c5": coordinates,
                "c5_kl": c5_kl,
            }
            c5_maps.setdefault(fold, {})[model] = coordinates

    # Re-run CAM-only C5 from its real CAM144 with the historical C5 parameters.
    cam_c5_payloads = [{"model": "CAM_only", "fold": fold, "pca50": contexts[("CAM_only", fold)]["pca50"], "metric": "euclidean", "init": "pca"} for fold in range(1, 11)]
    with ProcessPoolExecutor(max_workers=min(10, args.max_workers)) as pool:
        future_map = {pool.submit(fit_tsne, payload): payload["fold"] for payload in cam_c5_payloads}
        for future in as_completed(future_map):
            fold = future_map[future]
            coordinates, kl, runtime_sec = future.result()
            context = contexts[("CAM_only", fold)]
            context["c5"] = coordinates
            context["c5_kl"] = kl
            path = COORDS / f"C5_CAM_only_fold{fold:02}.npy"
            save_coord(path, coordinates)
            c5_maps[fold]["CAM_only"] = coordinates
            print(f"C5_CAM_ONLY fold={fold:02} runtime={runtime_sec:.2f}s", flush=True)

    # Cache C5 geometry and create original unsupervised rows and figures.
    for (model, fold), context in contexts.items():
        context["base"] = base_context(context["features"], context["labels"], context["pca50"], context["c5_kl"])
        path = COORDS / f"C5_{model}_fold{fold:02}.npy"
        row = make_row(model, fold, "C5", "C5", context["c5"], context, context["input_sha"], sha256_file(path), 0.0, context["c5_kl"], "none")
        row["coordinate_path"] = str(path.relative_to(EXP))
        c5_rows.append(row)
    for fold in range(1, 11):
        plot_grid(c5_maps[fold], contexts[(MODEL_IDS[0], fold)]["labels"], f"Original unsupervised C5 t-SNE | Fold {fold:02} | n=871", FORMAL_FIGURES / f"C5_fold{fold:02}.png", 600)

    # Fixed patient-level permutation, shared across every model and fold.
    labels = contexts[("CAM_only", 1)]["labels"]
    permutation = np.random.default_rng(2027).permutation(871)
    shuffled = labels[permutation].astype(np.int64)
    mapping_rows = [{"patient_id": int(i), "true_label": int(labels[i]), "shuffled_label": int(shuffled[i]), "source_patient_index": int(permutation[i])} for i in range(871)]
    write_csv(INPUTS / "shuffled_label_mapping.csv", mapping_rows)
    mapping_sha = sha256_file(INPUTS / "shuffled_label_mapping.csv")
    if np.bincount(shuffled).tolist() != [403, 468]:
        raise AssertionError("shuffled labels do not preserve class counts")

    scan_rows = []
    formal_rows = list(c5_rows)
    coordinates_by_key = {("C5", "C5", model, fold): contexts[(model, fold)]["c5"] for model in MODEL_IDS for fold in range(1, 11)}
    params_a_scan = [(alpha, beta) for alpha in [1.0, 0.85, 0.65, 0.45, 0.25] for beta in [0.0, 0.25, 0.5, 1.0, 2.0]]
    params_b_scan = [(s, gamma) for s in [1.0, 0.9, 0.75, 0.6, 0.4] for gamma in [0.0, 0.1, 0.25, 0.5, 1.0]]
    formal_a = {"P0": (1.0, 0.0), "P1": (0.85, 0.0), "P2": (0.65, 0.0), "P3": (0.45, 0.0), "P4": (1.0, 0.5), "P5": (1.0, 1.0), "P6": (0.85, 0.5), "P7": (0.65, 0.5), "P8": (0.45, 1.0), "P9": (0.25, 2.0)}
    formal_b = {"S0": (1.0, 0.0), "S1": (0.9, 0.0), "S2": (0.75, 0.0), "S3": (1.0, 0.25), "S4": (0.9, 0.25), "S5": (0.75, 0.25), "S6": (0.75, 0.5), "S7": (0.6, 0.5), "S8": (0.4, 1.0), "S9": (1.0, 1.0)}

    def add_a_row(model, fold, group, alpha, beta, labels_used, labels_source, scan=False):
        context = contexts[(model, fold)]
        coordinates = transform_coordinates(context["c5"], labels_used, alpha, beta)
        prefix = f"A_{'scan' if scan else group}_{model}_fold{fold:02}_a{alpha:.2f}_b{beta:.2f}"
        path = COORDS / f"{prefix}.npy"
        coord_sha = save_coord(path, coordinates)
        actual_label_source = "none_identity" if alpha == 1.0 and beta == 0.0 else labels_source
        row = make_row(model, fold, "A", group, coordinates, context, context["input_sha"], coord_sha, 0.0, None, actual_label_source, {"alpha": alpha, "beta": beta})
        row["coordinate_path"] = str(path.relative_to(EXP))
        if scan:
            scan_rows.append(row)
        else:
            formal_rows.append(row)
        coordinates_by_key[("A", group, model, fold)] = coordinates

    # Method A scan and all formal A/N1 groups.
    for alpha, beta in params_a_scan:
        for fold in (7, 8):
            for model in MODEL_IDS:
                add_a_row(model, fold, f"scan_a{alpha:.2f}_b{beta:.2f}", alpha, beta, contexts[(model, fold)]["labels"], "true_labels", scan=True)
    for group, (alpha, beta) in formal_a.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                add_a_row(model, fold, group, alpha, beta, contexts[(model, fold)]["labels"], "true_labels")
    for fold in range(1, 11):
        for model in MODEL_IDS:
            add_a_row(model, fold, "N1", 0.65, 0.5, shuffled, "shuffled_labels_seed_2027")

    # Build all unique Method B fits needed by the scan, formal groups, and N2.
    b_jobs = {}
    for s, gamma in params_b_scan:
        for fold in (7, 8):
            for model in MODEL_IDS:
                key = (model, fold, float(s), float(gamma), "true_labels")
                context = contexts[(model, fold)]
                b_jobs[key] = {"pca50": context["pca50"], "used_labels": context["labels"], "s": float(s), "gamma": float(gamma)}
    for group, (s, gamma) in formal_b.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                key = (model, fold, float(s), float(gamma), "true_labels")
                context = contexts[(model, fold)]
                b_jobs[key] = {"pca50": context["pca50"], "used_labels": context["labels"], "s": float(s), "gamma": float(gamma)}
    for fold in range(1, 11):
        for model in MODEL_IDS:
            context = contexts[(model, fold)]
            key = (model, fold, 0.75, 0.5, "shuffled_labels_seed_2027")
            b_jobs[key] = {"pca50": context["pca50"], "used_labels": shuffled, "s": 0.75, "gamma": 0.5}
    print(f"SUPERVISED_TSNE_JOBS {len(b_jobs)}", flush=True)
    b_results = {}
    failures = []
    with ProcessPoolExecutor(max_workers=min(10, args.max_workers)) as pool:
        future_map = {pool.submit(fit_supervised_tsne, payload): key for key, payload in b_jobs.items()}
        completed = 0
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                b_results[key] = future.result()
            except Exception as error:
                failures.append((key, error))
            completed += 1
            if completed % 25 == 0 or completed == len(future_map):
                print(f"SUPERVISED_TSNE_PROGRESS {completed}/{len(future_map)}", flush=True)
    for key, error in failures:
        print(f"SUPERVISED_TSNE_RETRY {key} {error!r}", flush=True)
        try:
            b_results[key] = fit_supervised_tsne(b_jobs[key])
        except Exception as retry_error:
            raise RuntimeError(f"supervised t-SNE failed after deterministic retry for {key}: {retry_error}") from retry_error

    def b_coordinates(key, group):
        model, fold, s, gamma, labels_source = key
        coordinates, kl, runtime_sec, d_scale = b_results[key]
        path = COORDS / f"B_{model}_fold{fold:02}_s{s:.2f}_g{gamma:.2f}_{labels_source.replace('shuffled_labels_seed_2027', 'shuffled')}.npy"
        if not path.exists():
            save_coord(path, coordinates)
        context = contexts[(model, fold)]
        labels_source_text = labels_source
        if group == "S0" or (group.startswith("scan_") and s == 1.0 and gamma == 0.0):
            labels_source_text = "none_distance_identity"
        row = make_row(model, fold, "B", group, coordinates, context, context["input_sha"], sha256_file(path), runtime_sec, kl, labels_source_text, {"s": s, "gamma": gamma}, d_scale)
        row["coordinate_path"] = str(path.relative_to(EXP))
        coordinates_by_key[("B", group, model, fold)] = coordinates
        return row

    for s, gamma in params_b_scan:
        for fold in (7, 8):
            for model in MODEL_IDS:
                key = (model, fold, float(s), float(gamma), "true_labels")
                scan_rows.append(b_coordinates(key, f"scan_s{s:.2f}_g{gamma:.2f}"))
    for group, (s, gamma) in formal_b.items():
        for fold in range(1, 11):
            for model in MODEL_IDS:
                key = (model, fold, float(s), float(gamma), "true_labels")
                formal_rows.append(b_coordinates(key, group))
    for fold in range(1, 11):
        for model in MODEL_IDS:
            key = (model, fold, 0.75, 0.5, "shuffled_labels_seed_2027")
            formal_rows.append(b_coordinates(key, "N2"))

    # Deterministic formal and scan figures.
    row_index = {(row["method"], row["group"], row["model"], int(row["fold"])): row for row in formal_rows + scan_rows}
    for method, specs, dpi in (("A", [f"scan_a{a:.2f}_b{b:.2f}" for a, b in params_a_scan], 300), ("B", [f"scan_s{s:.2f}_g{g:.2f}" for s, g in params_b_scan], 300)):
        for group in specs:
            for fold in (7, 8):
                coordinate_map = {model: coordinates_by_key[(method, group, model, fold)] for model in MODEL_IDS}
                if method == "A" and group == "scan_a1.00_b0.00":
                    title = f"Method A scan identity — original C5 coordinates — no labels used | Fold {fold:02}"
                elif method == "A":
                    title = f"Method A — Post-hoc Label-Guided Transformation — LABEL-GUIDED / EXPLORATORY | {group} | Fold {fold:02}"
                elif method == "B" and group == "scan_s1.00_g0.00":
                    title = f"Method B scan identity — unmodified distances — random-init unsupervised t-SNE — no labels used | Fold {fold:02}"
                else:
                    title = f"Method B — Supervised t-SNE (True Labels Used) — SUPERVISED / EXPLORATORY | {group} | Fold {fold:02}"
                plot_grid(coordinate_map, labels, title, SCAN_FIGURES / f"{method}_{group}_fold{fold:02}.png", dpi)
    formal_groups = [("C5", "C5"), *(('A', group) for group in formal_a), (("A", "N1")), *(('B', group) for group in formal_b), (("B", "N2"))]
    for method, group in formal_groups:
        for fold in range(1, 11):
            coordinate_map = {model: (c5_maps[fold][model] if method == "C5" else coordinates_by_key[(method, group, model, fold)]) for model in MODEL_IDS}
            if method == "C5":
                title = f"Original unsupervised C5 t-SNE | Fold {fold:02}"
            elif method == "A" and group == "P0":
                title = f"Method A P0 identity — original unsupervised C5 coordinates — no labels used | Fold {fold:02}"
            elif method == "A":
                title = f"Method A — Post-hoc Label-Guided Transformation — LABEL-GUIDED / EXPLORATORY | {group} | Fold {fold:02}"
            elif group == "S0":
                title = f"Method B S0 — unmodified distances — random-init unsupervised t-SNE — no labels used | Fold {fold:02}"
            else:
                title = f"Method B — Supervised t-SNE (True Labels Used) — SUPERVISED / EXPLORATORY | {group} | Fold {fold:02}"
            if group in {"N1", "N2"}:
                title += " | Shuffled-Label Control"
            plot_grid(coordinate_map, labels, title, FORMAL_FIGURES / f"{method}_{group}_fold{fold:02}.png", 600)

    # Paired C5 versus every formal label-guided group, with per-model shared axes.
    paired_rows = []
    metric_names = ["silhouette_2d", "asd_within_distance", "td_within_distance", "within_mean_distance", "normalized_within", "between_distance", "between_within_ratio", "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30", "trustworthiness_15", "trustworthiness_30", "pca_2d_neighbor_overlap_15", "pca_2d_neighbor_overlap_30"]
    for method, group in [("A", group) for group in formal_a] + [("A", "N1")] + [("B", group) for group in formal_b] + [("B", "N2")]:
        for fold in range(1, 11):
            transformed_map = {model: coordinates_by_key[(method, group, model, fold)] for model in MODEL_IDS}
            pair_label = "LABEL-GUIDED / EXPLORATORY" if method == "A" and group != "P0" else ("SUPERVISED / EXPLORATORY" if method == "B" and group not in {"S0"} else "UNSUPERVISED IDENTITY OR RANDOM-INIT REFERENCE")
            plot_pair(c5_maps[fold], transformed_map, labels, f"C5 vs {method}/{group} | Fold {fold:02} | {pair_label}", PAIRED_FIGURES / f"{method}_{group}_fold{fold:02}.png")
            for model in MODEL_IDS:
                base = next(row for row in c5_rows if row["model"] == model and int(row["fold"]) == fold)
                target = next(row for row in formal_rows if row["method"] == method and row["group"] == group and row["model"] == model and int(row["fold"]) == fold)
                paired = {"comparison": f"C5_vs_{method}_{group}", "model": model, "fold": fold, "reference_method": "C5", "reference_group": "C5", "target_method": method, "target_group": group}
                for metric in metric_names:
                    paired[f"delta_{metric}"] = float(target[metric]) - float(base[metric])
                paired_rows.append(paired)
    # S0 against S1-S9 and the requested matched shuffled comparisons.
    for model in MODEL_IDS:
        for fold in range(1, 11):
            s0 = next(row for row in formal_rows if row["method"] == "B" and row["group"] == "S0" and row["model"] == model and int(row["fold"]) == fold)
            for group in [f"S{i}" for i in range(1, 10)]:
                target = next(row for row in formal_rows if row["method"] == "B" and row["group"] == group and row["model"] == model and int(row["fold"]) == fold)
                paired = {"comparison": f"S0_vs_{group}", "model": model, "fold": fold, "reference_method": "B", "reference_group": "S0", "target_method": "B", "target_group": group}
                for metric in metric_names:
                    paired[f"delta_{metric}"] = float(target[metric]) - float(s0[metric])
                paired_rows.append(paired)
            for reference_group, target_group, comparison in (("P7", "N1", "P7_vs_N1"), ("S6", "N2", "S6_vs_N2")):
                ref_method = "A" if reference_group.startswith("P") else "B"
                target_method = "A" if target_group == "N1" else "B"
                reference = next(row for row in formal_rows if row["method"] == ref_method and row["group"] == reference_group and row["model"] == model and int(row["fold"]) == fold)
                target = next(row for row in formal_rows if row["method"] == target_method and row["group"] == target_group and row["model"] == model and int(row["fold"]) == fold)
                paired = {"comparison": comparison, "model": model, "fold": fold, "reference_method": ref_method, "reference_group": reference_group, "target_method": target_method, "target_group": target_group}
                for metric in metric_names:
                    paired[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                paired_rows.append(paired)
            for p_group in formal_a:
                p_row = next(row for row in formal_rows if row["method"] == "A" and row["group"] == p_group and row["model"] == model and int(row["fold"]) == fold)
                for s_group in formal_b:
                    s_row = next(row for row in formal_rows if row["method"] == "B" and row["group"] == s_group and row["model"] == model and int(row["fold"]) == fold)
                    paired = {"comparison": f"{p_group}_vs_{s_group}", "model": model, "fold": fold, "reference_method": "A", "reference_group": p_group, "target_method": "B", "target_group": s_group}
                    for metric in metric_names:
                        paired[f"delta_{metric}"] = float(s_row[metric]) - float(p_row[metric])
                    paired_rows.append(paired)
    # Descriptive CAM-only to other-model comparisons for every formal group.
    for method, group in [("C5", "C5"), *[("A", g) for g in formal_a], ("A", "N1"), *[("B", g) for g in formal_b], ("B", "N2")]:
        for fold in range(1, 11):
            for other in MODEL_IDS[1:]:
                reference = next(row for row in formal_rows if row["method"] == method and row["group"] == group and row["model"] == "CAM_only" and int(row["fold"]) == fold)
                target = next(row for row in formal_rows if row["method"] == method and row["group"] == group and row["model"] == other and int(row["fold"]) == fold)
                paired = {"comparison": f"CAM_only_to_{other}_{method}_{group}", "model": other, "fold": fold, "reference_method": method, "reference_group": group, "target_method": method, "target_group": group}
                for metric in metric_names:
                    paired[f"delta_{metric}"] = float(target[metric]) - float(reference[metric])
                paired_rows.append(paired)

    selected = {
        "selection_rule": "predeclared examples, no score-based deletion or per-model parameter tuning",
        "examples": [
            {"method": "C5", "group": "C5", "folds": [7, 8]},
            *[{"method": "A", "group": group, "folds": [7, 8]} for group in ["P0", "P6", "P9", "N1"]],
            *[{"method": "B", "group": group, "folds": [7, 8]} for group in ["S0", "S6", "S8", "N2"]],
        ],
        "model_order": MODEL_IDS,
        "label_colors": COLORS,
        "method_labels": {"A": "Post-hoc Label-Guided Transformation", "B": "Supervised t-SNE (True Labels Used)"},
    }
    (EXP / "results/selected_visualization_examples.json").write_text(json.dumps(selected, indent=2) + "\n")
    cam_status = json.loads((EXP / "results/cam_only_rebuild/cam_only_rebuild_status.json").read_text())
    generate_report(formal_rows, scan_rows, paired_rows, c5_rows, contexts, cam_status)
    run_manifest = {
        "status": "RUN_COMPLETE_PENDING_AUDIT",
        "models": MODEL_IDS,
        "folds": 10,
        "scan_rows": len(scan_rows),
        "formal_rows": len(formal_rows),
        "c5_rows": len(c5_rows),
        "supervised_tsne_jobs": len(b_jobs),
        "shuffle_mapping_sha256": mapping_sha,
        "elapsed_seconds": time.monotonic() - started,
    }
    (EXP / "results/run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(run_manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
