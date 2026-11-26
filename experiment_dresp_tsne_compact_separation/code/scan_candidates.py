from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    CANDIDATES,
    FIXED,
    HISTORY,
    MODELS,
    PREVIEW_FOLDS,
    RESULTS,
    coordinate_sha,
    geometry_metrics,
    input_path,
    load_input,
    preprocess,
    sha256_file,
    trust_metrics,
    fit_embedding,
)


HISTORY_CANDIDATE = {"C0": "B5", "C1": "B6"}
COLORS = {0: "#D98282", 1: "#4C9A91"}
MODEL_TITLES = {"SPromptGL": "SPromptGL", "DARG": "DARG", "DARG_JSL": "DARG + JSL", "DReSP": "DReSP"}


def historical_run(candidate_id: str, model: str, fold: int) -> dict:
    rows = json.loads((HISTORY / "results" / "stage2_comparison.json").read_text())
    historical_id = HISTORY_CANDIDATE[candidate_id]
    for row in rows:
        if row["config"] == historical_id and row["model"] == model and int(row["fold"]) == fold:
            return row
    raise KeyError(f"missing historical run {historical_id} {model} fold {fold}")


def historical_embedding(candidate_id: str, model: str, fold: int) -> tuple[np.ndarray, dict] | None:
    historical_id = HISTORY_CANDIDATE.get(candidate_id)
    if historical_id is None:
        return None
    path = HISTORY / "results" / "stage2_coordinates" / f"{historical_id}_{model}_fold{fold:02d}.npy"
    if not path.exists():
        return None
    embedding = np.load(path, allow_pickle=False)
    if embedding.shape != (871, 2) or not np.isfinite(embedding).all():
        raise ValueError(f"invalid reusable historical coordinate {path}")
    return embedding, {
        "coordinate_source": str(path),
        "reused_historical": True,
        "historical_config": historical_id,
        "historical_coordinate_sha256": coordinate_sha(embedding),
    }


def run_one(task: tuple[str, str, int]) -> tuple[dict, np.ndarray]:
    candidate_id, model, fold = task
    candidate = CANDIDATES[candidate_id]
    started = time.time()
    reusable = historical_embedding(candidate_id, model, fold)
    if reusable is not None:
        embedding, source_info = reusable
        historical = historical_run(candidate_id, model, fold)
        run_info = {"kl_divergence": float(historical["kl"]), "historical_runtime_sec": float(historical["runtime"])}
        runtime = float(historical["runtime"])
    else:
        embedding, run_info, runtime = fit_embedding(model, fold, candidate)
        source_info = {
            "coordinate_source": "fresh_t_sne_fit_from_frozen_input",
            "reused_historical": False,
            "historical_config": "",
        }
    data = load_input(model, fold)
    _, reduced, _ = preprocess(data["features"])
    metrics = {}
    metrics.update(geometry_metrics(embedding, data["labels"].astype(int)))
    metrics.update(trust_metrics(reduced, embedding))
    row = {
        "candidate": candidate_id,
        "model": model,
        "fold": fold,
        **candidate,
        **FIXED,
        "sklearn_iteration_argument": "max_iter" if "max_iter" in __import__("inspect").signature(__import__("sklearn.manifold", fromlist=["TSNE"]).TSNE).parameters else "n_iter",
        "kl_divergence": run_info["kl_divergence"],
        "runtime_sec": runtime,
        "total_runtime_sec": time.time() - started,
        "input_file_sha256": sha256_file(input_path(model, fold)),
        "coordinate_sha256": coordinate_sha(embedding),
        "nonfinite_coordinates": int((~np.isfinite(embedding)).sum()),
        **source_info,
        **metrics,
    }
    return row, embedding


def axis_limits(embeddings: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack(embeddings)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    span = max(span * 1.08, 1.0)
    return (float(center[0] - span / 2), float(center[0] + span / 2)), (float(center[1] - span / 2), float(center[1] + span / 2))


def plot_preview(candidate_id: str, embeddings: dict[tuple[str, int], np.ndarray], shared_limits: dict[int, tuple[tuple[float, float], tuple[float, float]]]) -> dict:
    out_dir = RESULTS / "preview_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for fold in PREVIEW_FOLDS:
        xlim, ylim = shared_limits[fold]
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), facecolor="white")
        axes = axes.ravel()
        for index, model in enumerate(MODELS):
            ax = axes[index]
            data = load_input(model, fold)
            xy = embeddings[(model, fold)]
            labels = data["labels"].astype(int)
            for label in (0, 1):
                mask = labels == label
                ax.scatter(xy[mask, 0], xy[mask, 1], s=10, alpha=0.70, c=COLORS[label], linewidths=0, label=("ASD" if label == 0 else "TD"))
            ax.set_title(f"({chr(97 + index)}) {MODEL_TITLES[model]}", fontsize=11)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if index == 0:
                ax.legend(frameon=False, loc="best", markerscale=1.4)
        fig.suptitle(f"{candidate_id} · Fold {fold:02d} · perplexity {CANDIDATES[candidate_id]['perplexity']} · early exaggeration {CANDIDATES[candidate_id]['early_exaggeration']} · learning rate {CANDIDATES[candidate_id]['learning_rate']}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"{candidate_id}_fold{fold:02d}.png"
        fig.savefig(path, dpi=300, facecolor="white")
        width, height = (int(round(value * 300)) for value in fig.get_size_inches())
        plt.close(fig)
        figures.append({
            "candidate": candidate_id,
            "fold": fold,
            "path": str(path),
            "dpi": 300,
            "width_px": width,
            "height_px": height,
            "xlim": xlim,
            "ylim": ylim,
            "colors": COLORS,
            "models": MODELS,
            "coordinate_sha256_by_model": {
                model: coordinate_sha(embeddings[(model, fold)]) for model in MODELS
            },
        })
    return figures


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tasks = [(candidate_id, model, fold) for candidate_id in CANDIDATES for fold in PREVIEW_FOLDS for model in MODELS]
    max_workers = min(8, os.cpu_count() or 1)
    rows: list[dict] = []
    embeddings: dict[tuple[str, str, int], np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            row, embedding = future.result()
            rows.append(row)
            embeddings[(task[0], task[1], task[2])] = embedding
            print(f"{index}/{len(tasks)} {task[0]} {task[1]} fold{task[2]:02d}", flush=True)
    rows.sort(key=lambda row: (list(CANDIDATES).index(row["candidate"]), PREVIEW_FOLDS.index(row["fold"]), MODELS.index(row["model"])))
    fields = list(rows[0].keys())
    with (RESULTS / "parameter_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    previews = []
    shared_limits = {
        fold: axis_limits([
            embeddings[(candidate_id, model, fold)]
            for candidate_id in CANDIDATES
            for model in MODELS
        ])
        for fold in PREVIEW_FOLDS
    }
    for candidate_id in CANDIDATES:
        preview_embeddings = {(model, fold): embeddings[(candidate_id, model, fold)] for fold in PREVIEW_FOLDS for model in MODELS}
        previews.extend(plot_preview(candidate_id, preview_embeddings, shared_limits))
    (RESULTS / "preview_manifest.json").write_text(json.dumps({"figures": previews, "models": MODELS, "folds": PREVIEW_FOLDS}, indent=2) + "\n")
    reusable = sum(bool(row["reused_historical"]) for row in rows)
    (RESULTS / "scan_manifest.json").write_text(json.dumps({"candidate_count": len(CANDIDATES), "runs": len(rows), "models": MODELS, "folds": PREVIEW_FOLDS, "max_workers": max_workers, "reused_historical_runs": reusable, "fresh_runs": len(rows) - reusable, "parameter_comparison": "results/parameter_comparison.csv", "previews": "results/preview_figures"}, indent=2) + "\n")
    print(f"completed {len(rows)} scan rows, reused {reusable} historical rows")


if __name__ == "__main__":
    main()
