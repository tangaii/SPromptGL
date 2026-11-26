from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score

from common import (
    FOLDS,
    MODELS,
    RESULTS,
    coordinate_sha,
    geometry_metrics,
    load_input,
    preprocess,
    sha256_file,
    trust_metrics,
)


COORDINATE_DIR = RESULTS / "formal_coordinates"
FIGURE_DIR = RESULTS / "formal_figures"
COLORS = {0: "#D98282", 1: "#4C9A91"}
LABEL_NAMES = {0: "ASD", 1: "TD"}
MODEL_TITLES = {"SPromptGL": "SPromptGL", "DARG": "DARG", "DARG_JSL": "DARG + JSL", "DReSP": "DReSP"}


def coordinate_path(model: str, fold: int) -> Path:
    return COORDINATE_DIR / f"{model}_fold{fold:02d}.npy"


def read_formal_manifest() -> dict:
    return json.loads((RESULTS / "formal_run_manifest.json").read_text())


def axis_limits(embeddings: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack(embeddings)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    span = max(float(high[0] - low[0]), float(high[1] - low[1]))
    span = max(span * 1.08, 1.0)
    return (float(center[0] - span / 2), float(center[0] + span / 2)), (float(center[1] - span / 2), float(center[1] + span / 2))


def draw_points(ax, embedding: np.ndarray, labels: np.ndarray, title: str, xlim: tuple[float, float], ylim: tuple[float, float], legend: bool) -> None:
    for label in (0, 1):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=12,
            alpha=0.70,
            c=COLORS[label],
            linewidths=0,
            rasterized=True,
            label=LABEL_NAMES[label],
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if legend:
        ax.legend(frameon=False, loc="best", markerscale=1.5)


def make_figures(embeddings: dict[tuple[str, int], np.ndarray]) -> list[dict]:
    individual_dir = FIGURE_DIR / "individual"
    comparison_dir = FIGURE_DIR / "comparison"
    individual_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    limits_by_fold = {
        fold: axis_limits([embeddings[(model, fold)] for model in MODELS])
        for fold in FOLDS
    }
    for model in MODELS:
        model_dir = individual_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold in FOLDS:
            data = load_input(model, fold)
            embedding = embeddings[(model, fold)]
            xlim, ylim = limits_by_fold[fold]
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
            draw_points(ax, embedding, data["labels"].astype(int), f"{MODEL_TITLES[model]} · Fold {fold:02d}", xlim, ylim, True)
            fig.tight_layout(pad=0.5)
            path = model_dir / f"fold{fold:02d}.png"
            fig.savefig(path, dpi=600, facecolor="white")
            figures.append({
                "kind": "individual",
                "model": model,
                "fold": fold,
                "path": str(path),
                "dpi": 600,
                "xlim": xlim,
                "ylim": ylim,
                "coordinate_sha256": coordinate_sha(embedding),
            })
            plt.close(fig)
    for fold in FOLDS:
        fold_embeddings = [embeddings[(model, fold)] for model in MODELS]
        xlim, ylim = axis_limits(fold_embeddings)
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), facecolor="white")
        axes = axes.ravel()
        for index, model in enumerate(MODELS):
            data = load_input(model, fold)
            draw_points(axes[index], embeddings[(model, fold)], data["labels"].astype(int), f"({chr(97 + index)}) {MODEL_TITLES[model]}", xlim, ylim, index == 0)
        fig.tight_layout(pad=0.8, w_pad=0.4, h_pad=0.5)
        path = comparison_dir / f"fold{fold:02d}_comparison.png"
        fig.savefig(path, dpi=600, facecolor="white")
        figures.append({
            "kind": "comparison",
            "fold": fold,
            "path": str(path),
            "dpi": 600,
            "xlim": xlim,
            "ylim": ylim,
            "models": MODELS,
            "coordinate_sha256_by_model": {
                model: coordinate_sha(embeddings[(model, fold)]) for model in MODELS
            },
        })
        plt.close(fig)
    return figures


def compute_metrics(manifest: dict, embeddings: dict[tuple[str, int], np.ndarray]) -> list[dict]:
    run_by_key = {(row["model"], int(row["fold"])): row for row in manifest["runs"]}
    rows = []
    for model in MODELS:
        for fold in FOLDS:
            data = load_input(model, fold)
            labels = data["labels"].astype(int)
            features = data["features"].astype(np.float64)
            scaled, reduced, _ = preprocess(features)
            embedding = embeddings[(model, fold)]
            row = {
                "model": model,
                "fold": fold,
                "candidate": manifest["candidate"],
                "silhouette_2d": float(silhouette_score(embedding, labels)),
                "silhouette_raw144": float(silhouette_score(features, labels)),
                "silhouette_standardized144": float(silhouette_score(scaled, labels)),
                "silhouette_pca50": float(silhouette_score(reduced, labels)),
                **geometry_metrics(embedding, labels),
                **trust_metrics(reduced, embedding),
                "kl_divergence": float(run_by_key[(model, fold)]["kl_divergence"]),
                "runtime_sec": float(run_by_key[(model, fold)]["fit_runtime_sec"]),
                "x_min": float(embedding[:, 0].min()),
                "x_max": float(embedding[:, 0].max()),
                "y_min": float(embedding[:, 1].min()),
                "y_max": float(embedding[:, 1].max()),
                "nonfinite_coordinates": int((~np.isfinite(embedding)).sum()),
                "n_patients": int(len(labels)),
                "asd_count": int(np.sum(labels == 0)),
                "td_count": int(np.sum(labels == 1)),
                "input_file_sha256": sha256_file(Path(__file__).resolve().parents[1] / "inputs" / f"{model}_fold{fold:02d}_features.npz"),
                "coordinate_sha256": coordinate_sha(embedding),
            }
            rows.append(row)
    return rows


def write_coordinates(embeddings: dict[tuple[str, int], np.ndarray]) -> None:
    path = RESULTS / "tsne_coordinates.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "fold", "patient_id", "true_label", "tsne_x", "tsne_y"])
        writer.writeheader()
        for model in MODELS:
            for fold in FOLDS:
                data = load_input(model, fold)
                embedding = embeddings[(model, fold)]
                for index in range(len(embedding)):
                    writer.writerow(
                        {
                            "model": model,
                            "fold": fold,
                            "patient_id": int(data["patient_ids"][index]),
                            "true_label": int(data["labels"][index]),
                            "tsne_x": float(embedding[index, 0]),
                            "tsne_y": float(embedding[index, 1]),
                        }
                    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = read_formal_manifest()
    embeddings = {}
    for model in MODELS:
        for fold in FOLDS:
            path = coordinate_path(model, fold)
            embedding = np.load(path, allow_pickle=False)
            if embedding.shape != (871, 2) or not np.isfinite(embedding).all():
                raise ValueError(f"invalid formal coordinate {path}")
            embeddings[(model, fold)] = embedding
    rows = compute_metrics(manifest, embeddings)
    with (RESULTS / "final_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_coordinates(embeddings)
    figures = make_figures(embeddings)
    (RESULTS / "formal_plot_manifest.json").write_text(json.dumps({"figures": figures, "colors": COLORS, "label_names": LABEL_NAMES, "point_size": 12, "alpha": 0.70, "dpi": 600, "models": MODELS, "folds": FOLDS, "shared_axis_limits_per_fold": True}, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {len(rows)} metric rows, {len(embeddings) * 871} coordinate rows, and {len(figures)} formal PNGs")


if __name__ == "__main__":
    main()
