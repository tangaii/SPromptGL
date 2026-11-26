from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn

from common import CANDIDATES, FIXED, MODELS, ROOT, RESULTS


METRICS = [
    "silhouette_2d",
    "normalized_within",
    "separation_ratio",
    "cross_class_neighbor_rate_15",
    "cross_class_neighbor_rate_30",
    "trustworthiness_15",
    "trustworthiness_30",
]


def means(rows: list[dict]) -> dict[str, float]:
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def population_sd(rows: list[dict], metric: str) -> float:
    return float(np.std([float(row[metric]) for row in rows], ddof=0))


def compare_groups(reference_rows: list[dict], final_rows: list[dict]) -> dict:
    reference = means(reference_rows)
    final = means(final_rows)
    return {
        metric: {
            "reference": reference[metric],
            "final": final[metric],
            "delta_final_minus_reference": final[metric] - reference[metric],
        }
        for metric in METRICS
    }


def main() -> None:
    rows = list(csv.DictReader((RESULTS / "parameter_comparison.csv").open()))
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate"]].append(row)
    aggregate = {candidate: means(by_candidate[candidate]) for candidate in CANDIDATES}
    b5 = aggregate["C0"]
    trust_floor = b5["trustworthiness_15"] - 0.015
    feasible = [candidate for candidate in CANDIDATES if aggregate[candidate]["trustworthiness_15"] >= trust_floor]

    # The frozen lexicographic rule follows the requested priorities. The
    # first two terms favor class readability and relative compactness. The
    # next terms resolve ties using separation and neighborhood mixing, with
    # trustworthiness retained as the final tie-breaker.
    selected = max(
        feasible,
        key=lambda candidate: (
            aggregate[candidate]["silhouette_2d"],
            -aggregate[candidate]["normalized_within"],
            aggregate[candidate]["separation_ratio"],
            -aggregate[candidate]["cross_class_neighbor_rate_15"],
            -aggregate[candidate]["cross_class_neighbor_rate_30"],
            aggregate[candidate]["trustworthiness_15"],
        ),
    )
    if selected != "C5":
        raise RuntimeError(f"expected the frozen rule to select C5 from the observed scan, got {selected}")
    b6 = aggregate["C1"]
    comparison = {}
    for reference_id, reference in (("B5", b5), ("B6", b6)):
        comparison[reference_id] = {
            metric: {
                "reference": reference[metric],
                "final": aggregate[selected][metric],
                "delta_final_minus_reference": aggregate[selected][metric] - reference[metric],
            }
            for metric in METRICS
        }
    final_rows = by_candidate[selected]
    grouped_comparison = {}
    for reference_id, reference_candidate in (("B5", "C0"), ("B6", "C1")):
        reference_rows = by_candidate[reference_candidate]
        grouped_comparison[reference_id] = {
            "overall": compare_groups(reference_rows, final_rows),
            "by_model": {
                model: compare_groups(
                    [row for row in reference_rows if row["model"] == model],
                    [row for row in final_rows if row["model"] == model],
                )
                for model in MODELS
            },
            "by_fold": {
                str(fold): compare_groups(
                    [row for row in reference_rows if int(row["fold"]) == fold],
                    [row for row in final_rows if int(row["fold"]) == fold],
                )
                for fold in (7, 8)
            },
            "by_model_fold": {
                f"{model}_fold{fold:02d}": compare_groups(
                    [row for row in reference_rows if row["model"] == model and int(row["fold"]) == fold],
                    [row for row in final_rows if row["model"] == model and int(row["fold"]) == fold],
                )
                for model in MODELS
                for fold in (7, 8)
            },
        }

    selected_parameters = {
        "id": selected,
        **CANDIDATES[selected],
        **FIXED,
        "sklearn_version": sklearn.__version__,
        "sklearn_iteration_argument": "max_iter" if "max_iter" in __import__("inspect").signature(__import__("sklearn.manifold", fromlist=["TSNE"]).TSNE).parameters else "n_iter",
        "selection_frozen_before_formal_ten_fold": True,
        "selection_inputs": "Fold 7 and Fold 8, all four models, 104 candidate rows",
        "selection_uses_true_labels": True,
        "selection_is_exploratory": True,
        "selection_rule": {
            "reference": "C0 (historical B5)",
            "trustworthiness_15_floor": trust_floor,
            "trustworthiness_15_rule": "discard candidate if four-model mean falls more than 0.015 below C0",
            "ranking_order": [
                "maximize four-model mean Silhouette_2D",
                "minimize four-model mean Normalized_Within",
                "maximize four-model mean Separation_Ratio",
                "minimize four-model mean Cross_Class_Neighbor_Rate@15",
                "minimize four-model mean Cross_Class_Neighbor_Rate@30",
                "maximize four-model mean Trustworthiness@15",
            ],
            "per_model_or_per_fold_parameters": False,
        },
        "candidate_aggregate_means": aggregate,
        "candidate_aggregate_population_sd": {
            candidate: {metric: population_sd(by_candidate[candidate], metric) for metric in METRICS}
            for candidate in CANDIDATES
        },
        "feasible_candidates": feasible,
        "b5_b6_comparison": comparison,
        "b5_b6_changes_by_model_and_fold": grouped_comparison,
        "historical_reuse": {
            "C0": "B5 Fold 7/8 coordinates from historical stage2_coordinates, verified against frozen inputs and reviewed configuration",
            "C1": "B6 Fold 7/8 coordinates from historical stage2_coordinates, verified against frozen inputs and reviewed configuration",
        },
        "formal_generation_policy": "all 40 final embeddings were freshly fitted from the copied frozen 144-dimensional features",
        "python_version": platform.python_version(),
    }
    (RESULTS / "selected_parameters.json").write_text(json.dumps(selected_parameters, indent=2, ensure_ascii=True) + "\n")
    config = {
        "experiment": "experiment_dresp_tsne_compact_separation",
        "description": "DReSP ABIDE t-SNE compactness and class-separation exploratory visualization",
        "authoritative_directory": str(ROOT),
        "history_directory": "/model/spromptgl/experiment/experiment_dresp_tsne_visual_optimization",
        "models": MODELS,
        "folds": list(range(1, 11)),
        "input_patients_per_fold": 871,
        "feature_dimension": 144,
        "preprocessing": FIXED["preprocessing"],
        "candidate_count": len(CANDIDATES),
        "candidate_parameters": CANDIDATES,
        "selected_parameters_file": "results/selected_parameters.json",
        "selected_candidate": selected,
        "selected_parameters": {**CANDIDATES[selected], **FIXED},
        "scan_folds": [7, 8],
        "scan_runs": len(rows),
        "formal_runs": 40,
        "sklearn_version": sklearn.__version__,
        "sklearn_iteration_argument": selected_parameters["sklearn_iteration_argument"],
        "labels_used_only_after_fit": True,
        "visualization_only": True,
        "dpi_formal": 600,
        "image_format": "PNG only",
        "colors": {"ASD_label_0": "#D98282", "TD_label_1": "#4C9A91"},
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({"selected": selected, "trust_floor": trust_floor, "aggregate": aggregate[selected]}, indent=2))


if __name__ == "__main__":
    main()
