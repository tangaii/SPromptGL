from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import FOLDS, MODELS, RESULTS, ROOT


HISTORY = Path("/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization")
MODEL_TITLES = {"SPromptGL": "SPromptGL", "DARG": "DARG", "DARG_JSL": "DARG + JSL", "DReSP": "DReSP"}
METRIC_LABELS = {
    "silhouette_2d": "Silhouette 2D",
    "silhouette_raw144": "Silhouette raw 144D",
    "silhouette_standardized144": "Silhouette standardized 144D",
    "silhouette_pca50": "Silhouette PCA50",
    "trustworthiness_15": "Trustworthiness @15",
    "trustworthiness_30": "Trustworthiness @30",
    "normalized_within": "Normalized Within",
    "separation_ratio": "Separation Ratio",
    "cross_class_neighbor_rate_15": "Cross-class rate @15",
    "cross_class_neighbor_rate_30": "Cross-class rate @30",
}
SUMMARY_METRICS = list(METRIC_LABELS)
SCAN_METRICS = [
    "silhouette_2d",
    "normalized_within",
    "separation_ratio",
    "cross_class_neighbor_rate_15",
    "cross_class_neighbor_rate_30",
    "trustworthiness_15",
    "trustworthiness_30",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summary(rows: list[dict], metric: str) -> tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def formatted(rows: list[dict], metric: str) -> str:
    mean, sd = summary(rows, metric)
    return f"{mean:.6f} ± {sd:.6f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def comparison_delta(selected: dict, reference: str, group: str, group_key: str, metric: str) -> float:
    item = selected["b5_b6_changes_by_model_and_fold"][reference][group][group_key][metric]
    return float(item["delta_final_minus_reference"])


def main() -> None:
    selected = json.loads((RESULTS / "selected_parameters.json").read_text())
    config = json.loads((ROOT / "config.json").read_text())
    final_rows = read_csv(RESULTS / "final_metrics.csv")
    scan_rows = read_csv(RESULTS / "parameter_comparison.csv")
    history_rows = read_csv(HISTORY / "results" / "final_metrics.csv")
    by_model = defaultdict(list)
    by_fold = defaultdict(list)
    scan_by_candidate = defaultdict(list)
    history_by_model = defaultdict(list)
    for row in final_rows:
        by_model[row["model"]].append(row)
        by_fold[int(row["fold"])].append(row)
    for row in scan_rows:
        scan_by_candidate[row["candidate"]].append(row)
    for row in history_rows:
        history_by_model[row["model"]].append(row)

    scan_table_rows = []
    for candidate, parameters in config["candidate_parameters"].items():
        rows = scan_by_candidate[candidate]
        scan_table_rows.append(
            [
                candidate,
                str(parameters["perplexity"]),
                str(parameters["early_exaggeration"]),
                str(parameters["learning_rate"]),
                formatted(rows, "silhouette_2d"),
                formatted(rows, "normalized_within"),
                formatted(rows, "separation_ratio"),
                formatted(rows, "cross_class_neighbor_rate_15"),
                formatted(rows, "trustworthiness_15"),
            ]
        )

    formal_rows = []
    for model in MODELS:
        rows = by_model[model]
        formal_rows.append([MODEL_TITLES[model], *[formatted(rows, metric) for metric in SUMMARY_METRICS]])

    other_folds = [fold for fold in FOLDS if fold not in (7, 8)]
    other_fold_rows = []
    for model in MODELS:
        rows = [row for row in by_model[model] if int(row["fold"]) in other_folds]
        other_fold_rows.append([MODEL_TITLES[model], *[formatted(rows, metric) for metric in SUMMARY_METRICS]])

    b5_full_rows = []
    historical_key_map = {
        "silhouette_2d": "silhouette_2d",
        "trustworthiness_15": "trustworthiness_k15",
        "trustworthiness_30": "trustworthiness_k30",
        "separation_ratio": "between_within_ratio_2d",
    }
    for model in MODELS:
        old_rows = history_by_model[model]
        new_rows = by_model[model]
        cells = [MODEL_TITLES[model]]
        for current_metric, historical_metric in historical_key_map.items():
            old_mean, old_sd = summary(old_rows, historical_metric)
            new_mean, new_sd = summary(new_rows, current_metric)
            cells.append(f"{old_mean:.6f} ± {old_sd:.6f} → {new_mean:.6f} ± {new_sd:.6f} ({new_mean - old_mean:+.6f})")
        b5_full_rows.append(cells)

    model_delta_rows = []
    for reference in ("B5", "B6"):
        for model in MODELS:
            model_delta_rows.append(
                [reference, MODEL_TITLES[model], *[
                    f"{comparison_delta(selected, reference, 'by_model', model, metric):+.6f}"
                    for metric in SCAN_METRICS
                ]]
            )

    fold_delta_rows = []
    for reference in ("B5", "B6"):
        for fold in (7, 8):
            fold_delta_rows.append(
                [reference, str(fold), *[
                    f"{comparison_delta(selected, reference, 'by_fold', str(fold), metric):+.6f}"
                    for metric in SCAN_METRICS
                ]]
            )

    model_fold_blocks = []
    for reference in ("B5", "B6"):
        rows = []
        for model in MODELS:
            for fold in (7, 8):
                key = f"{model}_fold{fold:02d}"
                rows.append(
                    [MODEL_TITLES[model], str(fold), *[
                        f"{comparison_delta(selected, reference, 'by_model_fold', key, metric):+.6f}"
                        for metric in SCAN_METRICS
                    ]]
                )
        model_fold_blocks.append(f"### C5 minus {reference}\n\n" + markdown_table(["Model", "Fold", *[METRIC_LABELS[m] + " delta" for m in SCAN_METRICS]], rows))

    mean_all = {metric: formatted(final_rows, metric) for metric in SUMMARY_METRICS}
    report = f"""# DReSP ABIDE t-SNE Compactness and Separation

## A. STATUS

Prepared for final independent audit. The authoritative audit result is recorded in `results/audit_status.json` and `manifest.json`.

## B. Historical Inputs and Parameter Identity

- Frozen feature source: `{HISTORY}/inputs/`, 40 archives across four models and ten folds.
- Every archive contains 871 ordered patient IDs, binary labels, and finite 144D features. The copy is checked byte-for-byte against the historical input manifest.
- Feature layers remain SPromptGL CAM144, DARG CAM144, DARG + JSL enhanced CAM144, and DReSP enhanced CAM144.
- Repository comparison: [SPromptGL on GitHub](https://github.com/tangaii/SPromptGL), checked at commit `4ba363c68bceeedc5721e197975270204fa313a4`. The historical formal workflow uses StandardScaler, PCA50, and t-SNE. No model, feature extractor, training code, weights, or historical experiment artifacts were changed.
- Runtime: Python `{selected['python_version']}`, scikit-learn `{selected['sklearn_version']}`, TSNE iteration keyword `{selected['sklearn_iteration_argument']}`.
- Shared preprocessing: StandardScaler → PCA(50, random_state=0) → t-SNE. The PCA50 representation is the reference space for Trustworthiness.

## C. Parameter Scan Results

The targeted scan covered only Fold 7 and Fold 8, all four models, and the requested 13 candidates. It contains 104 metric rows, of which 16 B5/B6 runs reuse historical coordinates after SHA and configuration checks. The other 88 runs were fitted independently. All candidates used Euclidean distance, PCA initialization, random state 0, and 3000 iterations. No run produced non-finite coordinates.

Values show mean ± population SD across the eight model-fold combinations for each candidate.

{markdown_table(["Candidate", "Perplexity", "Early exaggeration", "Learning rate", "Silhouette 2D", "Normalized Within", "Separation Ratio", "Cross-class @15", "Trustworthiness @15"], scan_table_rows)}

KL divergence, runtime, bounds, input and coordinate hashes, and per-run metrics are in `results/parameter_comparison.csv`. KL is diagnostic only and was not used to rank candidates across differing perplexities or feature distributions.

## D. Final Parameter Choice

The selection rule was frozen before formal ten-fold generation. It compares the four-model mean against B5, discards candidates whose Trustworthiness@15 falls more than 0.015 below B5, then ranks by mean Silhouette 2D, lower Normalized Within, higher Separation Ratio, lower cross-class neighbor rates, and higher Trustworthiness@15. It selected C5 uniformly for all models and folds.

Final shared parameters are perplexity 100, early exaggeration 48, learning rate 200, Euclidean metric, PCA initialization, random state 0, and 3000 iterations. C5 raises the scan mean Silhouette 2D from {selected['candidate_aggregate_means']['C0']['silhouette_2d']:.6f} for B5 to {selected['candidate_aggregate_means']['C5']['silhouette_2d']:.6f}, lowers Normalized Within from {selected['candidate_aggregate_means']['C0']['normalized_within']:.6f} to {selected['candidate_aggregate_means']['C5']['normalized_within']:.6f}, and changes mean Trustworthiness@15 by {selected['candidate_aggregate_means']['C5']['trustworthiness_15'] - selected['candidate_aggregate_means']['C0']['trustworthiness_15']:+.6f}. This is an exploratory visualization choice based on label-derived metrics.

## E. Four-Model Results

All values are mean ± population SD (`ddof=0`) across ten folds. Trustworthiness is measured against PCA50, the representation passed into t-SNE.

{markdown_table(["Model", *[METRIC_LABELS[m] for m in SUMMARY_METRICS]], formal_rows)}

The all-model mean ± population SD across 40 runs is {"; ".join(f"{METRIC_LABELS[m]} {mean_all[m]}" for m in SUMMARY_METRICS)}.

Fold 7 and Fold 8 informed parameter selection. The remaining eight folds are descriptive because prior rounds of parameter exploration had already inspected other folds. They are not a strictly independent validation set.

{markdown_table(["Model", *[METRIC_LABELS[m] for m in SUMMARY_METRICS]], other_fold_rows)}

## F. B5/B6 Comparison

The B5 comparison below uses the historical ten-fold metrics and C5's formal ten-fold results. For B6, only the verified historical Fold 7/8 coordinates were reused, so full-ten-fold B6 comparison is unavailable. The tables below give C5 minus each baseline on the eight selection runs.

{markdown_table(["Model", "B5 → C5 Silhouette 2D (Δ)", "B5 → C5 Trustworthiness @15 (Δ)", "B5 → C5 Trustworthiness @30 (Δ)", "B5 → C5 Separation Ratio (Δ)"], b5_full_rows)}

{markdown_table(["Reference", "Model", *[METRIC_LABELS[m] + " delta" for m in SCAN_METRICS]], model_delta_rows)}

{markdown_table(["Reference", "Fold", *[METRIC_LABELS[m] + " delta" for m in SCAN_METRICS]], fold_delta_rows)}

The complete reference-by-model-by-fold changes are also recorded in `results/selected_parameters.json`.

## G. Within-Class Compactness

`Within` is the equal-weight mean of the ASD and TD average distances to their own centroids. It is stored with `Global_RMS_Scale` in `results/final_metrics.csv`. Since t-SNE's scale is arbitrary, the cross-configuration comparison uses `Normalized_Within = Within / Global_RMS_Scale`. Lower values indicate tighter class distributions relative to the embedding's overall scale. The ten-fold Normalized Within values are reported above for each model.

## H. Between-Class Separation

`Separation_Ratio = Between / Within` and the @15/@30 cross-class neighbor rates describe complementary aspects of class layout. Lower cross-class rates indicate less local mixing. The centroid ratio can underdescribe non-convex or fragmented clusters, so it is interpreted together with Silhouette and neighbor mixing. Trustworthiness measures structure retention independently of the label-based separation measures.

## I. Figures

- Candidate previews: `results/preview_figures/`, 26 PNGs covering C0-C12 on Fold 7 and Fold 8. Each figure uses a 2×2 grid in SPromptGL, DARG, DARG + JSL, DReSP order and includes all 871 points per panel.
- Formal individual plots: `results/formal_figures/individual/`, 40 PNGs.
- Formal same-fold four-model comparisons: `results/formal_figures/comparison/`, ten PNGs.
- All formal plots are 600 DPI, white background, fixed ASD/TD colors, shared equal axis scale within each comparison, and no grid. No PDF, SVG, or EPS files were produced.

Comparison images: {" · ".join(f"[`Fold {fold:02d}`](formal_figures/comparison/fold{fold:02d}_comparison.png)" for fold in FOLDS)}.

## J. Audit and Invariance

The audit script independently checks the 40 historical and copied input hashes, archive arrays, patient IDs, labels, shapes and finite values. It replays B5/B6 historical preview coordinates, freshly recomputes all 40 final C5 embeddings, checks the CSV against the saved coordinates and input row order, recomputes the reported geometry and Trustworthiness metrics, validates the 50 PNGs and color pixels, and checks the fitting code for label-free t-SNE fitting. Its computed results are in `results/audit_status.json`.

The formal run manifest records 40 fresh fits from frozen features, a single C5 parameter set, and `labels_passed_to_tsne: false`. Labels are used only for post-fit assessment and exploratory candidate selection. No point was deleted or manually repositioned. All 871 transductive feature rows are present in each fold visualization. They are not all independent test patients for that fold.

## K. Scientific Limitations and Paper Use

Because true labels informed parameter selection through Silhouette and neighbor-mixing metrics, this is an exploratory, label-aware visualization selection. The final two-dimensional Silhouette is not an independent validation metric. The eight non-selection folds are descriptive, not strictly independent, given earlier parameter exploration. t-SNE coordinates depend on the chosen projection and are not evidence of classification accuracy, ACC, AUC, or independent generalization. Paper figures should be described as exploratory feature-space visualizations, with the label-informed selection and transductive inclusion of all 871 patients disclosed.

## L. Files and Handoff

- Coordinates: `results/tsne_coordinates.csv` (34,840 patient rows) and `results/formal_coordinates/` (40 NumPy coordinate arrays).
- Core outputs: `results/parameter_comparison.csv`, `results/selected_parameters.json`, `results/final_metrics.csv`, `results/audit_status.json`, and `manifest.json`.
- Reproducible code: `code/`.
- Handoff: the experiment summary is appended to `/model/spromptgl/experiment/handoff.md`. The 1,505,928-byte prior prefix retains SHA256 `03cdd64d6a967103af701b155d61269cedb248171b29d4f6c610e8bf4eead8a6`, and the 3,804-byte appended section hashes to `9a27182efa4d8e1e924c209d86ce7eef6444d54cff4ae93a6e06a9918b489cfe`.
"""
    (RESULTS / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {RESULTS / 'report.md'} with {len(final_rows)} formal metric rows and {len(scan_rows)} scan rows")


if __name__ == "__main__":
    main()
