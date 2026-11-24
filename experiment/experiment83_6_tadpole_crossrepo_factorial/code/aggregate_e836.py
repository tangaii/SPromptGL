"""Aggregate E83.6 fold metrics, OOF predictions, and paired contrasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E835 = Path("/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity")
MAIN_GROUPS = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
               "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1"]
HISTORICAL = {
    "H1": "G1_E28_E0", "H2": "G2_E43_1_E00", "H3": "G3_E74_4_G2",
    "H4": "G4_E82_4_6_10_G1", "H5": "G5_E43_4_E0_ANCHORED_PHYSICAL_CAM",
}
BASELINE = {
    "A1": "H1", "A2": "H1", "A3": "H1", "B1": "H2", "B2": "H2", "B3": "H2",
    "C1": "H3", "C2": "H3", "C3": "H3", "C4": "H3",
    "D1": "H4", "D2": "H4", "D3": "H4", "D4": "H4", "D5": "H4", "D6": "H4", "D7": "H4",
    "E1": "H2",
}


def load_new():
    replay = json.loads((ROOT / "audits/new_checkpoint_replay.json").read_text())
    by_group = {group: [] for group in MAIN_GROUPS}
    for row in replay["records"]:
        by_group[row["group"]].append(row)
    return by_group


def load_history():
    result = {}
    for logical, source in HISTORICAL.items():
        frame = pd.read_csv(E835 / source / "fold_metrics.csv")
        frame["group"] = logical
        if "AUC_original" not in frame:
            frame["AUC_original"] = frame["AUC"]
        if "AUC_probability" not in frame:
            values = []
            for fold in frame["fold"].astype(int):
                table = pd.read_csv(E835 / source / "predictions" / f"fold{fold:02d}.csv")
                values.append(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                    np.eye(3)[table["true_label"].to_numpy(dtype=np.int64)],
                    table[["prob_NC", "prob_MCI", "prob_AD"]].to_numpy(dtype=np.float64),
                    average="macro", multi_class="ovr")))
            frame["AUC_probability"] = values
        if "correct_count" in frame:
            frame["correct"] = frame["correct_count"]
        elif "OOF_correct" in frame:
            frame["correct"] = frame["OOF_correct"]
        else:
            frame["correct"] = np.rint(frame["ACC"] * frame["test_count"]).astype(int)
        result[logical] = frame.to_dict("records")
    return result


def load_s0():
    summary = json.loads((ROOT / "groups/S0/sp/runs/mmgpl_class_graph_sup/summary.json").read_text())
    rows = []
    for item in summary["folds"]:
        rows.append({"group": "S0", "fold": int(item["fold"]), "ACC": item["acc"],
                     "AUC_original": item["auc_hard"], "AUC_probability": item["auc_probability"],
                     "correct": int(round(item["acc"] * item["test_size"])),
                     "best_epoch": item["best_epoch_zero_based"]})
    replay = json.loads((ROOT / "audits/s0_checkpoint_replay.json").read_text())
    for row, audit in zip(rows, replay["records"], strict=True):
        row.update({"best_sha256": audit["best_sha256"], "latest_sha256": audit["latest_sha256"],
                    "correct": audit["correct"]})
    return rows


def load_oof(logical, source=None):
    rows = []
    if logical == "S0":
        base = ROOT / "groups/S0/sp/runs/mmgpl_class_graph_sup"
        for fold in range(1, 11):
            frame = pd.read_csv(base / f"fold_{fold:02d}/predictions.csv")
            frame = frame.rename(columns={"row_index": "sample_index", "label": "true_label",
                                          "prediction": "pred_label", "p_NC": "prob_NC",
                                          "p_MCI": "prob_MCI", "p_AD": "prob_AD"})
            frame["fold"] = fold
            frame["group"] = logical
            rows.append(frame[["group", "fold", "sample_index", "true_label", "pred_label",
                               "prob_NC", "prob_MCI", "prob_AD"]])
    else:
        base = (E835 / source / "predictions") if source else (ROOT / "groups" / logical / "predictions")
        for fold in range(1, 11):
            path = base / f"fold{fold:02d}.csv"
            frame = pd.read_csv(path)
            frame["group"] = logical
            rows.append(frame[["group", "fold", "sample_index", "true_label", "pred_label",
                               "prob_NC", "prob_MCI", "prob_AD"]])
    return pd.concat(rows, ignore_index=True)


def main():
    history = load_history()
    new = load_new()
    s0 = load_s0()
    fold_rows = []
    for logical, rows in history.items():
        for row in rows:
            fold_rows.append({"group": logical, "fold": int(row["fold"]), "ACC": row["ACC"],
                              "AUC_original": row["AUC_original"], "AUC_probability": row.get("AUC_probability"),
                              "correct": row.get("correct"), "best_epoch": row.get("best_epoch"),
                              "best_sha256": row.get("best_sha256", ""), "latest_sha256": row.get("latest_sha256", "")})
    for logical, rows in new.items():
        for row in rows:
            fold_rows.append({"group": logical, "fold": int(row["fold"]), "ACC": row["ACC"],
                              "AUC_original": row["AUC_original"], "AUC_probability": row["AUC_probability"],
                              "correct": row["correct"], "best_epoch": row["best_epoch"],
                              "best_sha256": row["best_sha256"], "latest_sha256": row["latest_sha256"]})
    for row in s0:
        fold_rows.append(row)
    fold_frame = pd.DataFrame(fold_rows).sort_values(["group", "fold"])
    fold_frame.to_csv(ROOT / "results/fold_metrics.csv", index=False)

    descriptions = {
        "H1": "E83.5 G1 Original SPromptGL-MMGPL12", "H2": "E83.5 G2 Physical No-PEN",
        "H3": "E83.5 G3 DARG Positive-DCR", "H4": "E83.5 G4 DARG+DPSL",
        "H5": "E83.5 G5 Anchored Physical CAM-only", "S0": "sp mmgpl_class_graph_sup",
    }
    descriptions.update({group: json.loads((ROOT / "configs" / f"{group}.json").read_text())["note"]
                         for group in MAIN_GROUPS})
    summary_rows = []
    for group in [*HISTORICAL, *MAIN_GROUPS, "S0"]:
        frame = fold_frame[fold_frame.group == group]
        summary_rows.append({"group": group, "description": descriptions[group], "folds": len(frame),
                             "correct_total": int(frame.correct.sum()),
                             "ACC_mean": frame.ACC.mean(), "ACC_std_population": frame.ACC.std(ddof=0),
                             "AUC_original_mean": frame.AUC_original.mean(),
                             "AUC_original_std_population": frame.AUC_original.std(ddof=0),
                             "AUC_probability_mean": frame.AUC_probability.mean(),
                             "AUC_probability_std_population": frame.AUC_probability.std(ddof=0),
                             "status": "COMPLETE" if len(frame) == 10 else "INCOMPLETE"})
    pd.DataFrame(summary_rows).to_csv(ROOT / "results/group_summary.csv", index=False)

    oof_parts = []
    for logical, source in HISTORICAL.items():
        oof_parts.append(load_oof(logical, source))
    for logical in MAIN_GROUPS:
        oof_parts.append(load_oof(logical))
    oof_parts.append(load_oof("S0"))
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["group", "sample_index"])
    oof.to_csv(ROOT / "results/oof_predictions.csv", index=False)

    pair_rows = []
    all_groups = list(HISTORICAL) + MAIN_GROUPS + ["S0"]
    metric_map = {(row.group, int(row.fold)): row for row in fold_frame.itertuples()}
    pair_specs = [(group, BASELINE[group], "self_baseline") for group in MAIN_GROUPS]
    pair_specs.extend((group, "H4", "H4") for group in MAIN_GROUPS if group != "H4")
    for group, reference, contrast_type in pair_specs:
        for fold in range(1, 11):
            left, right = metric_map[(group, fold)], metric_map[(reference, fold)]
            pair_rows.append({"group": group, "reference": reference, "contrast_type": contrast_type,
                              "fold": fold, "delta_ACC": left.ACC - right.ACC,
                              "delta_AUC_original": left.AUC_original - right.AUC_original,
                              "group_correct": int(left.correct), "reference_correct": int(right.correct),
                              "group_best_epoch": left.best_epoch})
    paired = pd.DataFrame(pair_rows)
    paired.to_csv(ROOT / "results/paired_contrasts.csv", index=False)
    print(json.dumps({"groups": len(summary_rows), "fold_rows": len(fold_frame),
                      "oof_rows": len(oof), "paired_rows": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
