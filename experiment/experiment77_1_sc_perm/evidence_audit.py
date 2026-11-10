#!/usr/bin/env python3
"""Read-only evidence and semantic intervention summaries for saved G2 checkpoints."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch

import run as runner
from audit import _load_checkpoint
from model import forward_sc_perm


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("evidence diagnostic received an empty or non-finite array")
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "p05": float(np.quantile(array, 0.05)), "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)), "min": float(array.min()),
        "max": float(array.max()),
    }


def run() -> dict:
    preflight = json.loads((RESULTS / "preflight.json").read_text(encoding="utf-8"))
    folds = []
    for fold in range(1, 11):
        checkpoint, state, module, tracker, runtime, old_groups, checkpoint_sha, path = _load_checkpoint(
            "E77.1-G2", fold, preflight
        )
        local_epoch = int(checkpoint["best_epoch_1based"]) - 1 - int(
            runner.CFG["base"]["start_epoch_zero_based"]
        )
        memory_snapshot = module.diagnostics()
        capture, swapped_capture = {}, {}
        rng_before = runtime.rng_sha(runtime.pack_rng())
        with torch.no_grad():
            full, _ = forward_sc_perm(
                runner.E74, state, module, "E77.1-G2", training=False,
                local_epoch=local_epoch, capture=capture,
            )
            swapped, _ = forward_sc_perm(
                runner.E74, state, module, "E77.1-G2", training=False,
                local_epoch=local_epoch, semantic_swap=True, capture=swapped_capture,
            )
        rng_after = runtime.rng_sha(runtime.pack_rng())
        if rng_before != rng_after:
            raise RuntimeError(f"evidence-only evaluation changed RNG in fold {fold}")

        test_ids = np.asarray(state["test_np"], dtype=np.int64)
        test = torch.as_tensor(test_ids, dtype=torch.long, device=state["x"].device)
        labels = state["y"][test].detach().cpu().numpy().astype(np.int64)
        base_logp = capture["base_logp"].detach().float().cpu().numpy()[test_ids]
        full_logp = full["logp"].detach().float().cpu().numpy()[test_ids]
        swapped_logp = swapped["logp"].detach().float().cpu().numpy()[test_ids]

        info = capture["evidence"]
        swap_info = swapped_capture["evidence"]
        class_scores = info["class_scores"].detach().float().cpu().numpy()[test_ids]
        hard_scores = info["hard_scores"].detach().float().cpu().numpy()[test_ids]
        total_evidence = info["evidence"].detach().float().cpu().numpy()[test_ids]
        correction = info["correction"].detach().float().cpu().numpy()[test_ids]
        swapped_class_scores = swap_info["class_scores"].detach().float().cpu().numpy()[test_ids]
        swapped_correction = swap_info["correction"].detach().float().cpu().numpy()[test_ids]
        memory_ids = module.hard_subject_ids.detach().cpu().numpy()

        prediction = full_logp.argmax(1)
        swapped_prediction = swapped_logp.argmax(1)
        metrics = runtime.classification_metrics(full["logp"], state["test"], state["y"])
        interventions = {
            "hard_prediction_changes": int((prediction != swapped_prediction).sum()),
            "mean_abs_logp_delta_ASD": float(np.abs(swapped_logp[:, 0] - full_logp[:, 0]).mean()),
            "mean_abs_logp_delta_TD": float(np.abs(swapped_logp[:, 1] - full_logp[:, 1]).mean()),
            "mean_abs_class_score_delta": float(np.abs(swapped_class_scores - class_scores).mean()),
            "mean_abs_correction_delta": float(np.abs(swapped_correction - correction).mean()),
            "mean_abs_base_to_full_logp_delta": float(np.abs(full_logp - base_logp).mean()),
        }
        folds.append({
            "group": "E77.1-G2", "fold": fold, "checkpoint_sha256": checkpoint_sha,
            "best_epoch_1based": int(checkpoint["best_epoch_1based"]),
            "test_n": int(len(test_ids)),
            "test_labels_n": {"ASD": int((labels == 0).sum()), "TD": int((labels == 1).sum())},
            "class_evidence_test_ASD": _stats(class_scores[:, 0]),
            "class_evidence_test_TD": _stats(class_scores[:, 1]),
            "class_evidence_margin_ASD_minus_TD": _stats(class_scores[:, 0] - class_scores[:, 1]),
            "hard_evidence_test_ASD": _stats(hard_scores[:, 0]),
            "hard_evidence_test_TD": _stats(hard_scores[:, 1]),
            "combined_evidence_test_ASD": _stats(total_evidence[:, 0]),
            "combined_evidence_test_TD": _stats(total_evidence[:, 1]),
            "logit_correction_margin_ASD_minus_TD": _stats(correction[:, 0] - correction[:, 1]),
            "semantic_swap_logp_delta_ASD": _stats(swapped_logp[:, 0] - full_logp[:, 0]),
            "semantic_swap_logp_delta_TD": _stats(swapped_logp[:, 1] - full_logp[:, 1]),
            "semantic_swap_abs_logp_delta_ASD": _stats(np.abs(swapped_logp[:, 0] - full_logp[:, 0])),
            "semantic_swap_abs_logp_delta_TD": _stats(np.abs(swapped_logp[:, 1] - full_logp[:, 1])),
            "semantic_swap_intervention": interventions,
            "checkpoint_metrics_percent": {
                key: float(metrics[key]) * 100.0
                for key in ("acc", "hard_auc", "prob_auc", "sen", "spe")
            },
            "memory_diagnostics_at_checkpoint": memory_snapshot,
            "hard_memory_live_patients": {
                "FP": int((memory_ids[0] >= 0).sum()),
                "FN": int((memory_ids[1] >= 0).sum()),
            },
            "rng_unchanged_on_analysis": True,
        })
        del checkpoint, state, module, tracker, runtime, full, swapped, capture, swapped_capture
        torch.cuda.empty_cache()

    numeric_keys = (
        "class_evidence_test_ASD", "class_evidence_test_TD", "class_evidence_margin_ASD_minus_TD",
        "hard_evidence_test_ASD", "hard_evidence_test_TD", "combined_evidence_test_ASD",
        "combined_evidence_test_TD", "logit_correction_margin_ASD_minus_TD",
        "semantic_swap_abs_logp_delta_ASD", "semantic_swap_abs_logp_delta_TD",
    )
    fold_level = {}
    for key in numeric_keys:
        fold_level[key] = {
            statistic: _stats([row[key][statistic] for row in folds])
            for statistic in ("mean", "std", "p05", "p50", "p95", "min", "max")
        }
    result = {
        "group": "E77.1-G2", "folds": 10,
        "analysis": "Best-checkpoint test-fold evidence distribution and semantic-swap inference; no training or optimizer step.",
        "fold_level_aggregate": fold_level,
        "folds_detail": folds,
    }
    runner.atomic_json(RESULTS / "evidence_diagnostics.json", result)
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "folds": report["folds"],
        "evidence_mean": {
            key: value["mean"]["mean"]
            for key, value in report["fold_level_aggregate"].items()
            if key in ("class_evidence_test_ASD", "class_evidence_test_TD", "hard_evidence_test_ASD", "hard_evidence_test_TD", "logit_correction_margin_ASD_minus_TD")
        },
        "semantic_swap_abs_logp_delta": {
            key: value["mean"]["mean"]
            for key, value in report["fold_level_aggregate"].items()
            if key.startswith("semantic_swap_abs_logp_delta")
        },
        "hard_memory_pool_sizes": [row["hard_memory_live_patients"] for row in report["folds_detail"]],
    }, indent=2))
