"""Validate the complete E83.7 artifact set without retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_e837 as runner


GROUPS = ["G1", "G2", "G3", "G4", "G5", "T1", "T2", "T3", "T4", "T5", "T6"]


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(relative):
    return json.loads((runner.ROOT / relative).read_text())


def finite(value):
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def main():
    summary = pd.read_csv(runner.ROOT / "results/group_summary.csv")
    folds = pd.read_csv(runner.ROOT / "results/fold_metrics.csv")
    oof = pd.read_csv(runner.ROOT / "results/oof_predictions.csv")
    paired = pd.read_csv(runner.ROOT / "results/paired_contrasts.csv")
    preflight = read("audits/preflight_start.json")
    resource = read("audits/resource_probe.json")
    seal = read("audits/training_seal.json")
    base_replay = read("audits/base_checkpoint_replay.json")
    new_replay = read("audits/new_checkpoint_replay.json")
    warmup = read("audits/warmup_pairing.json")
    mechanism = read("audits/mechanism_summary.json")

    group_counts = {group: int((folds.group == group).sum()) for group in GROUPS}
    coverage = {}
    for group in GROUPS:
        rows = oof[oof.group == group]
        counts = rows.sample_index.value_counts()
        coverage[group] = {"rows": int(len(rows)), "unique_sample_indices": int(rows.sample_index.nunique()),
                           "min_occurrence": int(counts.min()), "max_occurrence": int(counts.max()),
                           "labels": sorted(rows.true_label.unique().tolist())}
    complete = bool(set(summary.group) == set(GROUPS)
                    and all(group_counts[group] == 10 for group in GROUPS)
                    and all(coverage[group]["rows"] == 598
                            and coverage[group]["unique_sample_indices"] == 598
                            and coverage[group]["min_occurrence"] == 1
                            and coverage[group]["max_occurrence"] == 1 for group in GROUPS))

    sealed_missing, sealed_changed = [], []
    for relative, expected in seal["files"].items():
        path = runner.ROOT / relative
        if not path.is_file():
            sealed_missing.append(relative)
        elif sha_file(path) != expected:
            sealed_changed.append(relative)
    post_training_audit_files = []
    for path in sorted((runner.ROOT / "code").glob("audit_*_e837.py")):
        relative = str(path.relative_to(runner.ROOT))
        if relative not in seal["files"]:
            post_training_audit_files.append({"path": relative, "sha256": sha_file(path)})
    for path in sorted((runner.ROOT / "code").glob("mechanism_e837.py")):
        relative = str(path.relative_to(runner.ROOT))
        if relative not in seal["files"]:
            post_training_audit_files.append({"path": relative, "sha256": sha_file(path)})
    for path in sorted((runner.ROOT / "code").glob("finalize_e837.py")):
        relative = str(path.relative_to(runner.ROOT))
        if relative not in seal["files"]:
            post_training_audit_files.append({"path": relative, "sha256": sha_file(path)})

    replay = {
        "historical": base_replay["status"] == "PASS" and base_replay["best_passes"] == 50
        and base_replay["latest_passes"] == 50 and base_replay["next_step_passes"] == 50,
        "new": new_replay["status"] == "PASS" and new_replay["best_passes"] == 60
        and new_replay["latest_passes"] == 60 and new_replay["next_step_passes"] == 60,
    }
    mechanism_pass = bool(mechanism.get("status") == "PASS" and finite(mechanism)
                          and all(mechanism["groups"][group]["mean"]["residual_off_native_max_abs"] == 0.0
                                  for group in ("G4", "T1", "T2", "T4", "T6"))
                          and all(mechanism["groups"][group]["mean"]["dpsl_residual_max"] <= 0.080001
                                  for group in ("G4", "T1", "T2", "T4", "T6"))
                          and all(mechanism["groups"][group]["mean"]["graph_message_relative_norm_mean_by_modality"]
                                  for group in ("G4", "T1", "T2", "T4", "T6")))
    fidelity_text = (runner.ROOT / "paper_code_fidelity.md").read_text()
    method_fidelity = bool("G2 means physical PEN removal while preserving GFR" in fidelity_text
                           and "G5 is the physical CAM-only model" in fidelity_text)
    statuses = {
        "EXECUTION_AUDIT": complete and resource.get("status") == "PASS",
        "METHOD_FIDELITY": method_fidelity and preflight.get("status") == "PASS",
        "CHECKPOINT_REPLAY": all(replay.values()),
        "WARMUP_PAIRING": warmup.get("status") == "PASS" and warmup.get("paired_passes") == 10,
        "DATA_IDENTITY": preflight.get("data_sha256") == runner.DATA_SHA
        and preflight.get("split_sha256") == runner.SPLIT_SHA and complete,
        "TRAINING_SEAL": seal.get("status") == "SEALED_BEFORE_TRAINING"
        and not sealed_missing and not sealed_changed
        and resource.get("training_seal_sha256") == sha_file(runner.ROOT / "audits/training_seal.json"),
        "MECHANISM_AUDIT": mechanism_pass,
    }
    best_acc = summary.loc[summary.ACC_mean.idxmax()]
    best_auc = summary.loc[summary.AUC_original_mean.idxmax()]
    h4 = summary[summary.group == "G4"].iloc[0]
    new = summary[summary.group.str.startswith("T")]
    audit = {
        "experiment": "Experiment 83.7", "status": "COMPLETE_AUDITED" if all(statuses.values()) else "PARTIAL",
        "actual_directory": str(runner.ROOT),
        "preferred_directory_attempt": "/data/adodas/SPromptGL-main/experiment/experiment83_7_tadpole_g4_targeted_optimization",
        "fallback_reason": "OSSFS errno 5/EIO during concurrent formal output creation",
        "statuses": statuses,
        "execution_audit": {"historical_groups": ["G1", "G2", "G3", "G4", "G5"],
                             "new_groups": [f"T{number}" for number in range(1, 7)],
                             "historical_fold_count": 50, "new_fold_count": 60,
                             "group_fold_counts": group_counts, "oof_coverage": coverage,
                             "paired_contrast_rows": int(len(paired))},
        "method_fidelity": {"paper_code_fidelity": str(runner.ROOT / "paper_code_fidelity.md"),
                             "preflight": preflight, "resource_probe": resource,
                             "post_training_audit_files": post_training_audit_files,
                             "sealed_missing": sealed_missing, "sealed_changed": sealed_changed},
        "checkpoint_replay": {"historical": base_replay, "new": new_replay,
                              "all_pass": all(replay.values())},
        "warmup_pairing": warmup,
        "performance": {
            "best_acc_group": str(best_acc.group), "best_acc_mean": float(best_acc.ACC_mean),
            "best_auc_original_group": str(best_auc.group), "best_auc_original_mean": float(best_auc.AUC_original_mean),
            "historical_g4": {"ACC": float(h4.ACC_mean), "AUC_original": float(h4.AUC_original_mean)},
            "best_new_acc_group": str(new.loc[new.ACC_mean.idxmax(), "group"]),
            "best_new_acc_mean": float(new.ACC_mean.max()),
            "best_new_auc_original_group": str(new.loc[new.AUC_original_mean.idxmax(), "group"]),
            "best_new_auc_original_mean": float(new.AUC_original_mean.max()),
            "paper_reference": {"ACC": 0.9481, "AUC_original": 0.9559},
            "beats_historical_g4_acc": bool(new.ACC_mean.max() > h4.ACC_mean),
            "beats_historical_g4_auc_original": bool(new.AUC_original_mean.max() > h4.AUC_original_mean),
            "paper_acc_reached": bool(summary.ACC_mean.max() >= 0.9481),
            "paper_auc_original_reached": bool(summary.AUC_original_mean.max() >= 0.9559),
        },
        "mechanism": mechanism,
        "known_limitations": ["seed=0", "val_idx equals test_idx", "pre-CV full-data RFE",
                               "historical test-ACC Best selection", "registered variant comparison is exploratory"],
    }
    (runner.ROOT / "audits/final_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": audit["status"], "statuses": statuses,
                      "best_acc_group": str(best_acc.group), "best_auc_group": str(best_auc.group)}, indent=2))
    if audit["status"] != "COMPLETE_AUDITED":
        raise RuntimeError("Final E83.7 audit did not pass")


if __name__ == "__main__":
    main()
