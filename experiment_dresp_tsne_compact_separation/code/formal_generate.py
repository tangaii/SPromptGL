from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import (
    CANDIDATES,
    FOLDS,
    MODELS,
    RESULTS,
    coordinate_sha,
    fit_embedding,
    input_path,
    sha256_file,
)


COORDINATE_DIR = RESULTS / "formal_coordinates"


def run_one(task: tuple[str, int, str]) -> dict:
    model, fold, candidate_id = task
    candidate = CANDIDATES[candidate_id]
    started = time.time()
    embedding, run_info, fit_runtime = fit_embedding(model, fold, candidate)
    output = COORDINATE_DIR / f"{model}_fold{fold:02d}.npy"
    np.save(output, embedding)
    return {
        "model": model,
        "fold": fold,
        "candidate": candidate_id,
        "perplexity": candidate["perplexity"],
        "early_exaggeration": candidate["early_exaggeration"],
        "learning_rate": candidate["learning_rate"],
        "fresh_fit_from_frozen_features": True,
        "input_file_sha256": sha256_file(input_path(model, fold)),
        "coordinate_sha256": coordinate_sha(embedding),
        "coordinate_shape": list(embedding.shape),
        "nonfinite_coordinates": int((~np.isfinite(embedding)).sum()),
        "kl_divergence": run_info["kl_divergence"],
        "fit_runtime_sec": fit_runtime,
        "total_runtime_sec": time.time() - started,
        "coordinate_path": str(output.relative_to(RESULTS.parent)),
    }


def main() -> None:
    selected = json.loads((RESULTS / "selected_parameters.json").read_text())
    candidate_id = selected["id"]
    COORDINATE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(model, fold, candidate_id) for model in MODELS for fold in FOLDS]
    max_workers = min(8, os.cpu_count() or 1)
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            row = future.result()
            rows.append(row)
            print(f"{index}/{len(tasks)} {task[0]} fold{task[1]:02d}", flush=True)
    rows.sort(key=lambda row: (MODELS.index(row["model"]), row["fold"]))
    manifest = {
        "candidate": candidate_id,
        "parameters": selected,
        "models": MODELS,
        "folds": FOLDS,
        "runs": rows,
        "run_count": len(rows),
        "fresh_fit_count": sum(bool(row["fresh_fit_from_frozen_features"]) for row in rows),
        "max_workers": max_workers,
        "coordinate_directory": str(COORDINATE_DIR),
        "labels_passed_to_tsne": False,
    }
    (RESULTS / "formal_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    print(f"completed {len(rows)} fresh formal t-SNE fits using {candidate_id}")


if __name__ == "__main__":
    main()
