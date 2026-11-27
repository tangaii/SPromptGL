#!/usr/bin/env python3
"""Use explicit shuffled-label wording on the N1/N2 formal and paired PNGs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from run_experiment import (
    EXP,
    FORMAL_FIGURES,
    INPUTS,
    MODEL_IDS,
    PAIRED_FIGURES,
    plot_grid,
    plot_pair,
)

RESULTS = EXP / "results"
COORDS = RESULTS / "coordinates"


def main():
    for fold in range(1, 11):
        true_coords = {model: np.load(COORDS / f"C5_{model}_fold{fold:02}.npy", allow_pickle=False) for model in MODEL_IDS}
        a_coords = {model: np.load(COORDS / f"A_N1_{model}_fold{fold:02}_a0.65_b0.50.npy", allow_pickle=False) for model in MODEL_IDS}
        b_coords = {model: np.load(COORDS / f"B_{model}_fold{fold:02}_s0.75_g0.50_shuffled.npy", allow_pickle=False) for model in MODEL_IDS}
        with np.load(INPUTS / "cam_only_features" / f"CAM_only_fold{fold:02}_features.npz", allow_pickle=False) as archive:
            labels = np.asarray(archive["labels"], dtype=np.int64)
        title_a = f"Method A — Post-hoc Label-Guided Transformation — Shuffled-Label Control / EXPLORATORY; panels colored by true labels | N1 | Fold {fold:02}"
        title_b = f"Method B — Supervised t-SNE (Shuffled Labels Used) — SHUFFLED-LABEL CONTROL / EXPLORATORY; panels colored by true labels | N2 | Fold {fold:02}"
        plot_grid(a_coords, labels, title_a, FORMAL_FIGURES / f"A_N1_fold{fold:02}.png", 600)
        plot_grid(b_coords, labels, title_b, FORMAL_FIGURES / f"B_N2_fold{fold:02}.png", 600)
        plot_pair(true_coords, a_coords, labels, f"C5 vs A/N1 | Shuffled-Label Control; true-label colors | Fold {fold:02}", PAIRED_FIGURES / f"A_N1_fold{fold:02}.png")
        plot_pair(true_coords, b_coords, labels, f"C5 vs B/N2 | Shuffled-Label Control; true-label colors | Fold {fold:02}", PAIRED_FIGURES / f"B_N2_fold{fold:02}.png")
    print("SHUFFLED_CONTROL_FIGURES_PASS 40", flush=True)


if __name__ == "__main__":
    main()
