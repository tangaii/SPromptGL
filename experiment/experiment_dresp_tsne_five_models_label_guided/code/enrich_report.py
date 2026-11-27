#!/usr/bin/env python3
"""Add auditable pooled and fold-level quantitative findings to report.md."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path("/model/spromptgl/experiment/experiment_dresp_tsne_five_models_label_guided")
RESULTS = EXP / "results"
MODEL_NAMES = {"CAM_only": "CAM-only", "SPromptGL": "SPromptGL", "DARG": "DARG", "DARG_JSL": "DARG + JSL", "DReSP": "DReSP"}


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(values, digits=3):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f} ± {values.std(ddof=0):.{digits}f}"


def read_metric(row, name):
    return float(row[name])


def pooled_group_stats(formal, method, group, metric):
    selected = [row for row in formal if row["method"] == method and row["group"] == group]
    return fmt([read_metric(row, metric) for row in selected]) if selected else "NA"


def main():
    formal = rows(RESULTS / "formal_metrics.csv")
    scan = rows(RESULTS / "parameter_scan.csv")
    summary = rows(RESULTS / "formal_summary.csv")
    paired = rows(RESULTS / "paired_comparisons.csv")
    audit_path = RESULTS / "audit_status.json"
    audit_status = "AUDIT_PENDING"
    if audit_path.exists():
        import json
        audit_status = json.loads(audit_path.read_text()).get("status", "AUDIT_FAILED")

    # Parameter scans summarize all five models and both prescribed folds.
    scan_aggregates = []
    for method in ("A", "B"):
        groups = sorted({row["group"] for row in scan if row["method"] == method})
        for group in groups:
            selected = [row for row in scan if row["method"] == method and row["group"] == group]
            output = {"method": method, "group": group, "n_model_fold_rows": len(selected)}
            for metric in ("silhouette_2d", "asd_within_distance", "td_within_distance", "within_mean_distance", "normalized_within", "between_distance", "between_average_distance", "between_within_ratio", "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30", "trustworthiness_15", "trustworthiness_30", "pca_2d_neighbor_overlap_15", "pca_2d_neighbor_overlap_30", "original_space_kl_divergence", "kl_divergence"):
                values = np.asarray([float(row[metric]) for row in selected if row.get(metric, "") not in ("", None, "nan", "NaN")])
                if len(values):
                    output[f"{metric}_mean"] = float(values.mean())
                    output[f"{metric}_population_sd"] = float(values.std(ddof=0))
                else:
                    output[f"{metric}_mean"] = ""
                    output[f"{metric}_population_sd"] = ""
            scan_aggregates.append(output)
    with (RESULTS / "parameter_scan_summary.csv").open("w", newline="") as handle:
        fieldnames = list(scan_aggregates[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scan_aggregates)

    # Compare each shared parameter setting to the correct method-specific baseline.
    comparison_groups = [("A", f"P{i}", "C5", "C5") for i in range(10)] + [("B", f"S{i}", "B", "S0") for i in range(1, 10)]
    deltas = []
    for method, group, baseline_method, baseline_group in comparison_groups:
        target = [row for row in formal if row["method"] == method and row["group"] == group]
        baseline = [row for row in formal if row["method"] == baseline_method and row["group"] == baseline_group]
        baseline_map = {(row["model"], int(row["fold"])): row for row in baseline}
        for row in target:
            ref = baseline_map[(row["model"], int(row["fold"]))]
            record = {"method": method, "group": group, "model": row["model"], "fold": int(row["fold"]), "baseline": baseline_group}
            for metric in ("silhouette_2d", "normalized_within", "between_distance", "between_average_distance", "between_within_ratio", "trustworthiness_15", "trustworthiness_30", "cross_class_neighbor_rate_15", "cross_class_neighbor_rate_30", "original_space_kl_divergence"):
                record[f"delta_{metric}"] = float(row[metric]) - float(ref[metric])
            deltas.append(record)
    with (RESULTS / "paired_group_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    c5_summary = {row["model"]: row for row in summary if row["method"] == "C5" and row["group"] == "C5"}
    model_ranks = {}
    for method, group, metric in [("C5", "C5", "silhouette_2d_mean")]:
        model_ranks[(method, group)] = [row["model"] for row in sorted((item for item in summary if item["method"] == method and item["group"] == group), key=lambda item: float(item[metric]), reverse=True)]
    c5_rank = model_ranks[("C5", "C5")]
    reversals = []
    for method, group in [("A", f"P{i}") for i in range(10)] + [("B", f"S{i}") for i in range(10)]:
        group_rows = [row for row in summary if row["method"] == method and row["group"] == group]
        rank = [row["model"] for row in sorted(group_rows, key=lambda item: float(item["silhouette_2d_mean"]), reverse=True)]
        if rank != c5_rank:
            reversals.append((method, group, rank))

    group_rows = []
    for method, groups in (("A", [f"P{i}" for i in range(10)] + ["N1"]), ("B", [f"S{i}" for i in range(10)] + ["N2"])):
        for group in groups:
            selected = [row for row in formal if row["method"] == method and row["group"] == group]
            if not selected:
                continue
            group_rows.append((method, group, selected))
    most_compact = min(group_rows, key=lambda item: np.mean([float(row["within_mean_distance"]) for row in item[2]]))
    most_separated = max(group_rows, key=lambda item: np.mean([float(row["between_within_ratio"]) for row in item[2]]))
    lowest_trust = min(group_rows, key=lambda item: np.mean([float(row["trustworthiness_15"]) for row in item[2]]))

    lines = [
        "",
        "## S. Pooled quantitative findings across all five models",
        "",
        "The following pooled values use all five models and ten folds for each formal setting. They describe geometry, not classification performance.",
        "",
        "| Method/group | Silhouette 2D | ASD within | TD within | Mean within | Normalized within | Center distance | Mean cross-class distance | Between/within | Trust@15 | Trust@30 | Fit KL | Original-space KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, group, selected in group_rows:
        vals = [fmt([float(row[key]) for row in selected]) for key in ("silhouette_2d", "asd_within_distance", "td_within_distance", "within_mean_distance", "normalized_within", "between_distance", "between_average_distance", "between_within_ratio", "trustworthiness_15", "trustworthiness_30")]
        fit_kl = [float(row["kl_divergence"]) for row in selected if row.get("kl_divergence", "") not in ("", None, "nan", "NaN")]
        vals.append(fmt(fit_kl) if fit_kl else "n/a")
        vals.append(fmt([float(row["original_space_kl_divergence"]) for row in selected]))
        lines.append(f"| {method}/{group} | " + " | ".join(vals) + " |")
    lines += [
        "",
        f"Across pooled formal rows, the smallest mean within-class distance occurred at `{most_compact[0]}/{most_compact[1]}`. The largest center-distance/within-distance ratio occurred at `{most_separated[0]}/{most_separated[1]}`. The lowest Trustworthiness@15 occurred at `{lowest_trust[0]}/{lowest_trust[1]}`.",
        "",
        "### C5 raw and two-dimensional reference by model",
        "",
        "| Model | Raw 144D Silhouette | PCA50 Silhouette | C5 2D Silhouette | C5 Trust@15 | C5 Trust@30 | C5 KL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = c5_summary[model]
        vals = [row.get(f"{metric}_mean", "NA") for metric in ("silhouette_raw144", "silhouette_pca50", "silhouette_2d", "trustworthiness_15", "trustworthiness_30", "kl_divergence")]
        lines.append(f"| {MODEL_NAMES[model]} | " + " | ".join(f"{float(value):.4f}" if value != "NA" else "NA" for value in vals) + " |")

    lines += [
        "",
        "### Method-specific paired deltas",
        "",
        "A is paired against C5. B is paired against S0, which holds the unmodified distance matrix with random initialization. Values are pooled over five models and ten folds.",
        "",
        "| Setting | Δ Silhouette | Δ normalized within | Δ center distance | Δ mean cross-class distance | Δ between/within | Δ Trust@15 | Δ Trust@30 | Δ original-space KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, group, baseline, baseline_group in comparison_groups:
        selected = [row for row in deltas if row["method"] == method and row["group"] == group]
        vals = [fmt([row[f"delta_{metric}"] for row in selected]) for metric in ("silhouette_2d", "normalized_within", "between_distance", "between_average_distance", "between_within_ratio", "trustworthiness_15", "trustworthiness_30", "original_space_kl_divergence")]
        lines.append(f"| {method}/{group} vs {baseline_group} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "### Scan results",
        "",
        "`kl_divergence` is the fitted t-SNE objective from the C5 fit or the modified-distance Method B fit. Method A does not refit t-SNE, so its fit KL is not applicable and is left blank. `original_space_kl_divergence` is computed for every coordinate set from a full-pairwise symmetric P derived once from unmodified PCA50 and a Student-t Q from the resulting 2D coordinates. This common reference-space KL supports comparison across C5, A, and B.",
        "",
        "`parameter_scan_summary.csv` aggregates each of the 25 Method A and 25 Method B settings over all five models and Fold 7/8. It retains the pooled population SD and every underlying row remains in `parameter_scan.csv`.",
        "",
        "### Model ordering changes and DReSP comparison",
        "",
        f"C5 two-dimensional Silhouette ordering from highest to lowest was `{', '.join(MODEL_NAMES[model] for model in c5_rank)}`. {len(reversals)} of the twenty true-label formal parameter settings changed that ordering. The detailed model and fold pairings are in `paired_comparisons.csv` and `paired_group_deltas.csv`.",
        "DReSP does not have a single coordinate-wise visual advantage because each model's t-SNE is fitted independently. The pooled geometry metrics can change rank under label constraints. Such reversals measure the sensitivity of each model's own feature geometry and the explicit label intervention.",
        "",
        "### DReSP ten-fold detail for predeclared examples",
        "",
        "| Method/group | Fold | Silhouette 2D | ASD within | TD within | Center distance | B/W ratio | Trust@15 | Trust@30 | t-SNE fit KL | Original-space KL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, group in [("C5", "C5"), ("A", "P0"), ("A", "P6"), ("A", "P9"), ("A", "N1"), ("B", "S0"), ("B", "S6"), ("B", "S8"), ("B", "N2")]:
        for row in sorted((row for row in formal if row["model"] == "DReSP" and row["method"] == method and row["group"] == group), key=lambda item: int(item["fold"])):
            vals = [row.get(key, "") for key in ("silhouette_2d", "asd_within_distance", "td_within_distance", "between_distance", "between_within_ratio", "trustworthiness_15", "trustworthiness_30", "kl_divergence", "original_space_kl_divergence")]
            rendered = [f"{float(value):.4f}" if value not in ("", None) else "n/a" for value in vals]
            lines.append(f"| {method}/{group} | {int(row['fold']):02} | " + " | ".join(rendered) + " |")

    lines += [
        "",
        "## T. Constraint strength and candidate visualization sets",
        "",
        "P9 applies alpha=0.25 and beta=2.00, the most aggressive post-hoc transformation in the registered Method A grid. S8 applies s=0.40 and gamma=1.00, the strongest combined distance constraint in Method B. These are over-constrained for claims about naturally learned structure when their separation increases together with a Trustworthiness decline relative to their proper baseline.",
        "P6 and S6 are retained as moderate-constraint figure examples alongside C5, S0, P9/S8, and both shuffled-label controls. They are disclosure-complete exploration examples rather than selected model winners. The paper-facing candidate files are indexed by `selected_visualization_examples.json`.",
        "",
        "## U. Required interpretation boundary",
        "",
        "Original model feature separability is represented by raw144/PCA50/C5 values. Method A is a post-hoc mathematical rearrangement of C5. Method B uses labels before optimization. Neither label-guided Silhouette nor a more separated plot is an independent classification measure. Test-patient labels participated in this visualization, so no independent test generalization evidence is available.",
        "",
    ]
    report = RESULTS / "report.md"
    existing = report.read_text()
    base = existing.split("\n## S. Pooled quantitative findings", 1)[0]
    base = base.replace("COMPLETE_EXPLORATORY_FIVE_MODEL_PENDING_AUDIT", audit_status)
    base = base.replace(
        "## C. Method definitions",
        "The SPromptGL source is the isolated E28-E0 seed-0 retrain from the prior feature audit. Its per-fold best epoch and historical metrics match, but original historical Best checkpoint byte identity was unavailable. DARG, DARG + JSL, and DReSP have per-fold source Best checkpoint paths and SHA-256 recorded in `inputs/model_checkpoint_provenance.json`. The prior DARG + JSL top-level manifest remains `BLOCKED_PREFLIGHT`, while its saved checkpoint files and result audit are independently hashed here.\n\n## C. Method definitions",
    )
    report.write_text(base + "\n".join(lines))


if __name__ == "__main__":
    main()
