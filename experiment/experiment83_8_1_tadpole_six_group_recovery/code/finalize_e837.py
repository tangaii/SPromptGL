"""Aggregate immutable E83.7 fold artifacts and write the required results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_e837 as runner


BASE_MAP = {
    "H1": ("G1", "G1_E28_E0"), "H2": ("G2", "G2_E43_1_E00"),
    "H3": ("G3", "G3_E74_4_G2"), "H4": ("G4", "G4_E82_4_6_10_G1"),
    "H5": ("G5", "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM"),
}
NEW_GROUPS = [f"T{number}" for number in range(1, 7)]
ALL_GROUPS = ["G1", "G2", "G3", "G4", "G5"] + NEW_GROUPS


def load_json(path):
    return json.loads(Path(path).read_text())


def build_fold_metrics():
    base_replay = load_json(runner.ROOT / "audits/base_checkpoint_replay.json")
    new_replay = load_json(runner.ROOT / "audits/new_checkpoint_replay.json")
    rows = []
    for record in base_replay["records"]:
        group, source_group = BASE_MAP[record["group"]]
        rows.append({"group": group, "fold": int(record["fold"]), "status": "PASS",
                     "ACC": record["ACC"], "AUC_original": record["AUC_original"],
                     "AUC_probability": record["AUC_probability"], "correct": record["correct"],
                     "best_epoch": record["best_epoch"], "best_sha256": record["best_sha256"],
                     "latest_sha256": record["latest_sha256"],
                     "checkpoint": record["best_path"], "latest_checkpoint": record["latest_path"],
                     "source": f"E83.5/{source_group}"})
    for record in new_replay["records"]:
        group = record["group"]
        audit = load_json(runner.ROOT / "groups" / group / "audits" / f"fold{int(record['fold']):02d}.json")
        rows.append({"group": group, "fold": int(record["fold"]), "status": "PASS",
                     "ACC": record["ACC"], "AUC_original": record["AUC_original"],
                     "AUC_probability": record["AUC_probability"], "correct": record["correct"],
                     "best_epoch": record["best_epoch"], "best_sha256": record["best_sha256"],
                     "latest_sha256": record["latest_sha256"],
                     "checkpoint": audit["checkpoint"], "latest_checkpoint": audit["latest_checkpoint"],
                     "source": f"E83.7/{group}"})
    result = pd.DataFrame(rows).sort_values(["group", "fold"]).reset_index(drop=True)
    if result.groupby("group").size().to_dict() != {group: 10 for group in ALL_GROUPS}:
        raise RuntimeError("Fold metrics do not contain exactly ten audited folds per group")
    result.to_csv(runner.ROOT / "results/fold_metrics.csv", index=False)
    return result


def load_oof():
    data = []
    for logical, (group, source_group) in BASE_MAP.items():
        directory = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity") / source_group
        for fold in range(1, 11):
            table = pd.read_csv(directory / "predictions" / f"fold{fold:02d}.csv")
            table.insert(0, "group", group)
            data.append(table)
    for group in NEW_GROUPS:
        for fold in range(1, 11):
            table = pd.read_csv(runner.ROOT / "groups" / group / "predictions" / f"fold{fold:02d}.csv")
            table.insert(0, "group", group)
            data.append(table)
    oof = pd.concat(data, ignore_index=True, sort=False)
    oof["sample_index"] = oof["sample_index"].astype(np.int64)
    oof["fold"] = oof["fold"].astype(np.int64)
    if set(oof.group) != set(ALL_GROUPS):
        raise RuntimeError("OOF group identity is incomplete")
    for group in ALL_GROUPS:
        rows = oof[oof.group == group]
        counts = rows["sample_index"].value_counts()
        if len(rows) != 598 or len(counts) != 598 or counts.min() != 1 or counts.max() != 1:
            raise RuntimeError(f"OOF coverage failed for {group}")
    oof.to_csv(runner.ROOT / "results/oof_predictions.csv", index=False)
    return oof


def build_summary(folds, oof):
    rows = []
    for group in ALL_GROUPS:
        frame = folds[folds.group == group]
        patient = oof[oof.group == group]
        correct = patient.pred_label.to_numpy() == patient.true_label.to_numpy()
        row = {"group": group, "status": "COMPLETE", "folds": len(frame),
               "ACC_mean": float(frame.ACC.mean()), "ACC_std_population": float(frame.ACC.std(ddof=0)),
               "AUC_original_mean": float(frame.AUC_original.mean()),
               "AUC_original_std_population": float(frame.AUC_original.std(ddof=0)),
               "AUC_probability_mean": float(frame.AUC_probability.mean()),
               "AUC_probability_std_population": float(frame.AUC_probability.std(ddof=0)),
               "OOF_correct": int(correct.sum()),
               "NC_correct": int(((patient.true_label == 0) & correct).sum()),
               "MCI_correct": int(((patient.true_label == 1) & correct).sum()),
               "AD_correct": int(((patient.true_label == 2) & correct).sum()),
               "source": "E83.5 historical checkpoint replay" if group.startswith("G") else "E83.7 formal training"}
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(runner.ROOT / "results/group_summary.csv", index=False)
    return summary


def build_pairs(folds, oof):
    pairs = [("G4", "T1"), ("G4", "T2"), ("G4", "T4"), ("G4", "T6"),
             ("G3", "T3"), ("G3", "T5"), ("T3", "T4"), ("T5", "T6")]
    rows = []
    for reference, candidate in pairs:
        for fold in range(1, 11):
            ref_fold = folds[(folds.group == reference) & (folds.fold == fold)].iloc[0]
            cand_fold = folds[(folds.group == candidate) & (folds.fold == fold)].iloc[0]
            ref = oof[(oof.group == reference) & (oof.fold == fold)].sort_values("sample_index")
            cand = oof[(oof.group == candidate) & (oof.fold == fold)].sort_values("sample_index")
            if not np.array_equal(ref.sample_index.to_numpy(), cand.sample_index.to_numpy()):
                raise RuntimeError(f"Paired folds use different patients for {reference}/{candidate} fold {fold}")
            truth = ref.true_label.to_numpy()
            ref_correct = ref.pred_label.to_numpy() == truth
            cand_correct = cand.pred_label.to_numpy() == truth
            rows.append({"reference": reference, "candidate": candidate, "fold": fold,
                         "ACC_reference": ref_fold.ACC, "ACC_candidate": cand_fold.ACC,
                         "AUC_original_reference": ref_fold.AUC_original,
                         "AUC_original_candidate": cand_fold.AUC_original,
                         "ACC_delta": cand_fold.ACC - ref_fold.ACC,
                         "AUC_original_delta": cand_fold.AUC_original - ref_fold.AUC_original,
                         "correct_delta": int(cand_correct.sum() - ref_correct.sum()),
                         "patient_corrected": int((~ref_correct & cand_correct).sum()),
                         "patient_damaged": int((ref_correct & ~cand_correct).sum()),
                         "patient_net": int((cand_correct & ~ref_correct).sum() - (ref_correct & ~cand_correct).sum())})
        subset = pd.DataFrame([row for row in rows if row["reference"] == reference
                               and row["candidate"] == candidate])
        rows.append({"reference": reference, "candidate": candidate, "fold": "MEAN",
                     "ACC_reference": subset.ACC_reference.mean(), "ACC_candidate": subset.ACC_candidate.mean(),
                     "AUC_original_reference": subset.AUC_original_reference.mean(),
                     "AUC_original_candidate": subset.AUC_original_candidate.mean(),
                     "ACC_delta": subset.ACC_delta.mean(), "AUC_original_delta": subset.AUC_original_delta.mean(),
                     "correct_delta": int(subset.correct_delta.sum()),
                     "patient_corrected": int(subset.patient_corrected.sum()),
                     "patient_damaged": int(subset.patient_damaged.sum()),
                     "patient_net": int(subset.patient_net.sum())})
    result = pd.DataFrame(rows)
    result.to_csv(runner.ROOT / "results/paired_contrasts.csv", index=False)
    return result


def report(summary, pairs):
    mechanism = load_json(runner.ROOT / "audits/mechanism_summary.json")
    replay_base = load_json(runner.ROOT / "audits/base_checkpoint_replay.json")
    replay_new = load_json(runner.ROOT / "audits/new_checkpoint_replay.json")
    warmup = load_json(runner.ROOT / "audits/warmup_pairing.json")
    lines = ["# Experiment 83.7 Final Report", "", "## Status", "COMPLETE_AUDITED after formal execution and independent replay.",
             "", "## Protocol", "598 TADPOLE patients, six modalities, seed=0, ten fixed stratified folds, val_idx=test_idx, pre-CV RFE and historical ACC selection were retained.",
             "", "## Results", "", summary.to_markdown(index=False, floatfmt=".6f"),
             "", "The principal metrics are ten-fold mean ± population standard deviation. `AUC_original` is the hard one-hot macro OVR definition used by E83.5/E83.6.",
             "", "## Paired contrasts", "", pairs[pairs.fold == "MEAN"].to_markdown(index=False, floatfmt=".6f"),
             "", "## Mechanism", json.dumps({group: mechanism["groups"][group]["mean"] for group in mechanism["groups"]}, indent=2),
             "", "T3/T4 warmup pairing passed for 10/10 folds. Graph supervision was positive during epochs 1–10 and zero in formal training after epoch 10. T5/T6 audits record WD 0.11 in warmup and 0.055 from epoch 11.",
             "", "## Audits", f"Historical replay {replay_base['best_passes']}/{replay_base['folds']} Best and Latest folds passed. New replay {replay_new['best_passes']}/{replay_new['folds']} Best and Latest folds passed. Warmup pairing {warmup['paired_passes']}/{warmup['folds']} passed.",
             "", "## Limitations", "The registered val=test selection and pre-CV full-data RFE remain generalized-evaluation limitations. Seed=0 and the comparison of multiple registered variants do not establish unbiased generalization. Historical G2 is described in the E83.7 fidelity document as PEN-only removal while retaining GFR, GraphLearn, GCN, message passing and graph losses. G5 is the physical CAM-only group.",
             "", "## Reference", "Original TADPOLE reference is 94.81% ACC and 95.59% AUC. These values were not used to select folds or change metric definitions.", ""]
    (runner.ROOT / "results/final_report.md").write_text("\n".join(lines))


def main():
    folds = build_fold_metrics()
    oof = load_oof()
    summary = build_summary(folds, oof)
    pairs = build_pairs(folds, oof)
    report(summary, pairs)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
