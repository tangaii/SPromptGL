#!/usr/bin/env python3
"""Independently read and audit all Experiment 53.3 best checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import run_cadr_fallback as S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        parser.print_help()
        return
    _, reference = S.load_reference()
    group = S.GROUP
    training_model_source = S.EXP / ".runtime/training_source/cadr_gfr.py"
    training_model_sha = S.sha_file(training_model_source)
    rows, diagnostics = [], []
    interventions = []
    for fold in range(1, 11):
        row, fold_diagnostics = S.audit_fold(group, fold, reference)
        rows.append(row)
        diagnostics.extend(fold_diagnostics)

        import torch

        S.install_group(group)
        state = S.R.build_state(fold, S.R.prepare_fold(fold))
        checkpoint = S.R.torch_load(S.best_path(group, fold))
        assert checkpoint["source_sha"]["cadr_gfr.py"] == training_model_sha
        state["model"].load_state_dict(checkpoint["model"], strict=True)
        state["cam"].load_state_dict(checkpoint["cam"], strict=True)
        state["model"].eval()
        state["cam"].eval()
        with torch.no_grad():
            full = S.R.forward_state(state, training=False, use_cadr=True)
            positive_off = S.R.forward_state(
                state, training=False, use_cadr=True, intervention="positive_off"
            )
            pheno_off = S.R.forward_state(
                state, training=False, use_cadr=True, intervention="modality_off:PHENO"
            )
        assert full["info"]["propagation_modes"][0] == "ORIGINAL_FALLBACK"
        assert torch.equal(full["info"]["messages"][0], positive_off["info"]["messages"][0])
        expected_pheno_contribution = torch.sigmoid(state["model"].fusion_weight[0]) * full["info"]["messages"][0]
        observed_pheno_contribution = full["fused"] - pheno_off["fused"]
        subtraction_error = (observed_pheno_contribution - expected_pheno_contribution).norm()
        roundoff_bound = (
            2.0
            * torch.finfo(full["fused"].dtype).eps
            * (full["fused"].abs() + pheno_off["fused"].abs()).norm()
        )
        assert subtraction_error <= roundoff_bound
        full_metrics = S.R.classification_metrics(full["logp"], state["test"], state["y"])
        pos_metrics = S.R.classification_metrics(positive_off["logp"], state["test"], state["y"])
        modality_metrics = S.R.classification_metrics(pheno_off["logp"], state["test"], state["y"])
        test = torch.as_tensor(state["test"], dtype=torch.long, device=full["logp"].device)
        full_pred = full["logp"].exp()[test].argmax(1)
        pos_pred = positive_off["logp"].exp()[test].argmax(1)
        modality_pred = pheno_off["logp"].exp()[test].argmax(1)
        interventions.append({
            "group": group,
            "fold": fold,
            "intervention": "positive_off_disease_only",
            "pheno_propagation_mode": full["info"]["propagation_modes"][0],
            "pheno_message_unchanged": True,
            "prediction_flip_count": int((full_pred != pos_pred).sum().item()),
            "acc_delta_pp": (pos_metrics["acc"] - full_metrics["acc"]) * 100.0,
            "hard_auc_delta_pp": (pos_metrics["hard_auc"] - full_metrics["hard_auc"]) * 100.0,
            "prob_auc_delta_pp": (pos_metrics["prob_auc"] - full_metrics["prob_auc"]) * 100.0,
            "spe_delta_pp": (pos_metrics["spe"] - full_metrics["spe"]) * 100.0,
            "sen_delta_pp": (pos_metrics["sen"] - full_metrics["sen"]) * 100.0,
            "representation_delta_norm": float((full["fused"] - positive_off["fused"]).norm().cpu()),
        })
        interventions.append({
            "group": group,
            "fold": fold,
            "intervention": "modality_off_PHENO",
            "pheno_propagation_mode": full["info"]["propagation_modes"][0],
            "pheno_message_unchanged": False,
            "prediction_flip_count": int((full_pred != modality_pred).sum().item()),
            "acc_delta_pp": (modality_metrics["acc"] - full_metrics["acc"]) * 100.0,
            "hard_auc_delta_pp": (modality_metrics["hard_auc"] - full_metrics["hard_auc"]) * 100.0,
            "prob_auc_delta_pp": (modality_metrics["prob_auc"] - full_metrics["prob_auc"]) * 100.0,
            "spe_delta_pp": (modality_metrics["spe"] - full_metrics["spe"]) * 100.0,
            "sen_delta_pp": (modality_metrics["sen"] - full_metrics["sen"]) * 100.0,
            "representation_delta_norm": float((full["fused"] - pheno_off["fused"]).norm().cpu()),
        })
        print(f"independent checkpoint read {fold}/10: PASS", flush=True)
    S.finalize_results(rows, diagnostics, reference, S.load_reference()[0])

    for item in diagnostics:
        scale = float(item["fusion_weight_sigmoid"])
        alpha = 1.0 if item["propagation_mode"] == "ORIGINAL_FALLBACK" else float(item["cadr_reliability_gate"])
        item["final_fused_contribution_norm"] = scale * alpha * float(item["original_gcn_message_norm"])
        item["final_fused_contribution_input_ratio"] = scale * alpha * float(item["original_gcn_message_input_ratio"])
    S.write_rows(S.RESULTS / "diagnostics.csv", diagnostics)
    S.write_rows(S.RESULTS / "interventions.csv", interventions)

    import numpy as np

    lines = [
        "",
        "## Per-modality fused contribution and fixed-checkpoint interventions",
        "",
        "The contribution norm is the modality message multiplied by the fusion coefficient; fallback modalities use only `sigmoid(fusion_weight)`, while disease-reliable modalities also use their learned reliability gate.",
        "",
        "| Modality | Propagation | Mean fused contribution / input norm |",
        "|---|---|---:|",
    ]
    for modality in S.R.MODS:
        subset = [item for item in diagnostics if item["modality"] == modality]
        lines.append(
            f"| {modality} | {subset[0]['propagation_mode']} | "
            f"{np.mean([float(item['final_fused_contribution_input_ratio']) for item in subset]):.3e} |"
        )
    lines.extend([
        "",
        "At the same best checkpoint, `positive_off_disease_only` leaves PHENO's fallback message bitwise unchanged; `modality_off_PHENO` removes its full fallback contribution. These are evaluation-only interventions, not additional training groups.",
        "",
        "| Intervention | Mean ACC delta (pp) | Prediction flips / test patients | Mean representation delta norm |",
        "|---|---:|---:|---:|",
    ])
    for name in ("positive_off_disease_only", "modality_off_PHENO"):
        subset = [item for item in interventions if item["intervention"] == name]
        flip_count = sum(int(item["prediction_flip_count"]) for item in subset)
        test_count = sum(
            int(S.R.prepare_fold(item["fold"])["outer_test_count"])
            for item in subset
        )
        lines.append(
            f"| {name} | {np.mean([float(item['acc_delta_pp']) for item in subset]):+.4f} | "
            f"{flip_count}/{test_count} | {np.mean([float(item['representation_delta_norm']) for item in subset]):.6g} |"
        )
    lines.append("")
    report_path = S.RESULTS / "report.md"
    with report_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

    manifest_path = S.EXP / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training_time_cadr_gfr_sha256"] = training_model_sha
    manifest["audit_time_cadr_gfr_sha256"] = S.sha_file(S.EXP / "cadr_gfr.py")
    manifest["training_source_snapshot_sha256"] = S.sha_file(training_model_source)
    manifest["source_sha256"]["audit.py"] = S.sha_file(Path(__file__))
    manifest["fixed_checkpoint_intervention_audit"] = "PASS"
    manifest["result_sha256"] = {
        path.name: S.sha_file(path) for path in S.RESULTS.iterdir() if path.is_file()
    }
    manifest["handoff"] = {
        "path": "/data/adodas/SPromptGL-main/handoff.md",
        "status": "APPENDED",
        "sha256_after_append": S.sha_file(S.ROOT / "handoff.md"),
    }
    temp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, manifest_path)


if __name__ == "__main__":
    main()
