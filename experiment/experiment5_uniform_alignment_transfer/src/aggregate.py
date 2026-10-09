#!/usr/bin/env python3
"""Aggregate the twelve formal Experiment 5 jobs."""

import argparse
import csv
import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import parse_legacy_reference, sha256_file, source_commit


CONFIGS = [
    "S0_STRICT_BASELINE", "S1_STRICT_UNIFORM_GLOBAL",
    "L0_LEGACY_BASELINE", "L1_LEGACY_UNIFORM_GLOBAL",
]
SEEDS = [0, 1, 2]
EXPECTED_STEMS = ["{}_seed{}".format(c, s) for c in CONFIGS for s in SEEDS]


def write_df(path, frame):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame.to_csv(path, index=False)


def metric_arrays(y_internal, p_asd, pred_internal):
    y_internal = np.asarray(y_internal, dtype=np.int64)
    p_asd = np.asarray(p_asd, dtype=np.float64)
    pred_internal = np.asarray(pred_internal, dtype=np.int64)
    y = (y_internal == 0).astype(np.int64)
    pred = (pred_internal == 0).astype(np.int64)
    tp = np.sum((y == 1) & (pred == 1))
    tn = np.sum((y == 0) & (pred == 0))
    fp = np.sum((y == 0) & (pred == 1))
    fn = np.sum((y == 1) & (pred == 0))
    sen = float(tp / (tp + fn)) if tp + fn else 0.0
    spe = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "ACC": float(np.mean(pred == y)),
        "BA": 0.5 * (sen + spe),
        "probability_ASD_AUC": float(roc_auc_score(y, p_asd)),
        "ASD_F1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "ASD_sensitivity": sen,
        "NC_specificity": spe,
    }


def oof_metrics(frame):
    frame = frame.sort_values("subject_index")
    if len(frame) != 871 or frame.subject_index.astype(int).tolist() != list(range(871)):
        raise ValueError("OOF completeness failure")
    y = frame.true_internal_label.astype(int).to_numpy()
    p = frame.p_asd.astype(float).to_numpy()
    pred = frame.pred_internal.astype(int).to_numpy()
    out = metric_arrays(y, p, pred)
    onehot_y = np.eye(2)[y]
    onehot_pred = np.eye(2)[pred]
    out["legacy_hard_AUC"] = float(roc_auc_score(onehot_y, onehot_pred))
    return out


def bootstrap_pair(base, aligned, seed, protocol, n_boot=10000):
    base = base.sort_values("subject_index").reset_index(drop=True)
    aligned = aligned.sort_values("subject_index").reset_index(drop=True)
    if base.subject_index.tolist() != aligned.subject_index.tolist():
        raise ValueError("Paired bootstrap subject mismatch")
    y = base.true_internal_label.astype(int).to_numpy()
    p0 = base.p_asd.astype(float).to_numpy()
    p1 = aligned.p_asd.astype(float).to_numpy()
    pred0 = base.pred_internal.astype(int).to_numpy()
    pred1 = aligned.pred_internal.astype(int).to_numpy()
    rng = np.random.default_rng(91000 + int(seed) + (0 if protocol == "strict" else 100))
    delta_ba = np.empty(n_boot, dtype=np.float64)
    delta_auc = np.empty(n_boot, dtype=np.float64)
    n = len(y)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m0 = metric_arrays(y[idx], p0[idx], pred0[idx])
        m1 = metric_arrays(y[idx], p1[idx], pred1[idx])
        delta_ba[i] = m1["BA"] - m0["BA"]
        delta_auc[i] = m1["probability_ASD_AUC"] - m0["probability_ASD_AUC"]
    def summarize(values):
        ci = np.quantile(values, [0.025, 0.975])
        p_value = 2.0 * min(float(np.mean(values <= 0.0)), float(np.mean(values >= 0.0)))
        return float(values.mean()), float(ci[0]), float(ci[1]), min(1.0, p_value)
    ba_mean, ba_lo, ba_hi, ba_p = summarize(delta_ba)
    auc_mean, auc_lo, auc_hi, auc_p = summarize(delta_auc)
    return {
        "protocol": protocol,
        "seed": int(seed),
        "n_bootstrap": n_boot,
        "delta_BA": ba_mean,
        "BA_CI_low": ba_lo,
        "BA_CI_high": ba_hi,
        "BA_p_exploratory": ba_p,
        "delta_probability_ASD_AUC": auc_mean,
        "AUC_CI_low": auc_lo,
        "AUC_CI_high": auc_hi,
        "AUC_p_exploratory": auc_p,
    }


def representation_mean(folds, config, seed=None):
    q = folds[folds.config == config]
    if seed is not None:
        q = q[q.seed.astype(int) == int(seed)]
    return {
        key: float(q[key].astype(float).mean())
        for key in ["pre_cos_gap", "pre_R1", "pre_CMMD", "post_cos_gap", "post_R1", "post_CMMD"]
    }


def yes_no_pre(folds, baseline, aligned, prefix):
    b = representation_mean(folds, baseline)
    a = representation_mean(folds, aligned)
    return (
        a[prefix + "_cos_gap"] > b[prefix + "_cos_gap"] and
        a[prefix + "_R1"] > b[prefix + "_R1"] and
        a[prefix + "_CMMD"] < b[prefix + "_CMMD"]
    )


def compute_verdicts(folds, oofs, boot):
    rows = []
    by_seed = {}
    for protocol, bcfg, acfg in [
        ("strict", "S0_STRICT_BASELINE", "S1_STRICT_UNIFORM_GLOBAL"),
        ("legacy", "L0_LEGACY_BASELINE", "L1_LEGACY_UNIFORM_GLOBAL"),
    ]:
        seed_rows = []
        for seed in SEEDS:
            b = oofs[(oofs.config == bcfg) & (oofs.seed.astype(int) == seed)]
            a = oofs[(oofs.config == acfg) & (oofs.seed.astype(int) == seed)]
            mb, ma = oof_metrics(b), oof_metrics(a)
            seed_rows.append({"seed": seed, "delta_BA": ma["BA"] - mb["BA"],
                              "delta_AUC": ma["probability_ASD_AUC"] - mb["probability_ASD_AUC"]})
        mean_ba = float(np.mean([x["delta_BA"] for x in seed_rows]))
        mean_auc = float(np.mean([x["delta_AUC"] for x in seed_rows]))
        wins = int(sum(x["delta_BA"] > 0 for x in seed_rows))
        pre_working = yes_no_pre(folds, bcfg, acfg, "pre")
        post_working = yes_no_pre(folds, bcfg, acfg, "post")
        if mean_ba >= 0.005 and wins >= 2 and mean_auc >= -0.0025 and pre_working:
            supported = "YES"
        elif mean_ba > 0 and mean_auc >= -0.0025:
            supported = "PARTIAL"
        else:
            supported = "NO"
        by_seed[protocol] = seed_rows
        rows.extend([
            {"key": protocol.upper() + "_DELTA_BA", "value": mean_ba},
            {"key": protocol.upper() + "_DELTA_PROBABILITY_AUC", "value": mean_auc},
            {"key": protocol.upper() + "_BA_SEED_WINS", "value": wins},
            {"key": protocol.upper() + "_PRE_ALIGNMENT_WORKING", "value": "YES" if pre_working else "NO"},
            {"key": protocol.upper() + "_ALIGNMENT_PROPAGATES_THROUGH_PROMPT", "value": "YES" if post_working else "NO"},
            {"key": protocol.upper() + "_ALIGNMENT_SUPPORTED", "value": supported},
        ])
    strict = next(r["value"] for r in rows if r["key"] == "STRICT_ALIGNMENT_SUPPORTED")
    legacy = next(r["value"] for r in rows if r["key"] == "LEGACY_ALIGNMENT_SUPPORTED")
    # Cross-model generalization is the strict primary question.  Legacy is
    # used only for the separate protocol-robustness verdict.
    cross = {"YES": "YES", "PARTIAL": "PARTIAL", "NO": "NO"}[strict]
    if strict == "YES" and legacy in ("YES", "PARTIAL"):
        robust = "YES"
    elif strict == "PARTIAL" and legacy in ("YES", "PARTIAL"):
        robust = "PARTIAL"
    else:
        robust = "NO"
    rows.extend([
        {"key": "STRICT_ALIGNMENT_TRANSFER_SUPPORTED", "value": strict},
        {"key": "CROSS_MODEL_GENERALIZATION", "value": cross},
        {"key": "PROTOCOL_ROBUST_SPROMPTGL_ALIGNMENT", "value": robust},
    ])
    return pd.DataFrame(rows), by_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="/data/adodas/SPromptGL-main/experiment/experiment5_uniform_alignment_transfer")
    p.add_argument("--root", default="/data/adodas/SPromptGL-main")
    args = p.parse_args()
    raw = os.path.join(args.exp, "raw")
    fold_files = {os.path.basename(path)[:-len(".folds.csv")]: path
                  for path in glob.glob(os.path.join(raw, "*.folds.csv"))}
    oof_files = {os.path.basename(path)[:-len(".oof.csv")]: path
                 for path in glob.glob(os.path.join(raw, "*.oof.csv"))}
    if not set(EXPECTED_STEMS).issubset(fold_files) or not set(EXPECTED_STEMS).issubset(oof_files):
        raise RuntimeError("Missing one or more of the 12 formal raw fold/OOF files")

    fold_frames, oof_frames = [], []
    for stem in EXPECTED_STEMS:
        f = pd.read_csv(fold_files[stem])
        o = pd.read_csv(oof_files[stem])
        if len(f) != 10 or sorted(f.fold.astype(int).tolist()) != list(range(1, 11)):
            raise RuntimeError("{} does not contain exactly 10 folds".format(stem))
        if len(o) != 871 or o.subject_index.astype(int).nunique() != 871:
            raise RuntimeError("{} does not contain exactly 871 OOF rows".format(stem))
        if int(f.n_test.astype(int).sum()) != 871:
            raise RuntimeError("{} test counts do not cover 871 subjects".format(stem))
        fold_frames.append(f)
        oof_frames.append(o)
    folds = pd.concat(fold_frames, ignore_index=True)
    oofs = pd.concat(oof_frames, ignore_index=True)
    if len(folds) != 120 or len(oofs) != 10452:
        raise RuntimeError("Expected 120 fold rows and 10452 OOF rows")
    write_df(os.path.join(args.exp, "results/fold_results.csv"), folds)
    write_df(os.path.join(args.exp, "results/oof_predictions.csv"), oofs)

    seed_rows = []
    summary_rows = []
    for config in CONFIGS:
        q = folds[folds.config == config]
        for seed in SEEDS:
            oq = oofs[(oofs.config == config) & (oofs.seed.astype(int) == seed)]
            metrics = oof_metrics(oq)
            fr = q[q.seed.astype(int) == seed]
            row = {"config": config, "seed": seed, "n_oof": len(oq)}
            row.update({"oof_" + k: v for k, v in metrics.items() if np.isscalar(v)})
            for key in ["pre_cos_gap", "pre_R1", "pre_CMMD", "post_cos_gap", "post_R1", "post_CMMD", "best_epoch"]:
                row["fold_mean_" + key] = float(fr[key].astype(float).mean())
                row["fold_sd_" + key] = float(fr[key].astype(float).std(ddof=0))
            row["runtime_sec"] = float(fr.runtime_sec.astype(float).sum())
            seed_rows.append(row)
        qseed = pd.DataFrame([r for r in seed_rows if r["config"] == config])
        srow = {"config": config}
        for key in ["oof_ACC", "oof_BA", "oof_probability_ASD_AUC", "oof_legacy_hard_AUC", "oof_ASD_F1", "oof_ASD_sensitivity", "oof_NC_specificity"]:
            vals = qseed[key].astype(float).to_numpy()
            srow[key + "_mean"] = float(vals.mean())
            srow[key + "_sd_sample"] = float(vals.std(ddof=1))
        summary_rows.append(srow)
    write_df(os.path.join(args.exp, "results/seed_summary.csv"), pd.DataFrame(seed_rows))
    write_df(os.path.join(args.exp, "results/summary.csv"), pd.DataFrame(summary_rows))

    boot_rows = []
    for protocol, bcfg, acfg in [
        ("strict", "S0_STRICT_BASELINE", "S1_STRICT_UNIFORM_GLOBAL"),
        ("legacy", "L0_LEGACY_BASELINE", "L1_LEGACY_UNIFORM_GLOBAL"),
    ]:
        for seed in SEEDS:
            base = oofs[(oofs.config == bcfg) & (oofs.seed.astype(int) == seed)]
            aligned = oofs[(oofs.config == acfg) & (oofs.seed.astype(int) == seed)]
            print("BOOTSTRAP protocol={} seed={} n=10000".format(protocol, seed), flush=True)
            boot_rows.append(bootstrap_pair(base, aligned, seed, protocol))
    boot = pd.DataFrame(boot_rows)
    write_df(os.path.join(args.exp, "results/bootstrap.csv"), boot)

    # Audits: strict split/init parity and legacy fixed split/init rates.
    audit_rows = []
    for seed in SEEDS:
        for fold in range(1, 11):
            s0 = folds[(folds.config == "S0_STRICT_BASELINE") & (folds.seed == seed) & (folds.fold == fold)].iloc[0]
            s1 = folds[(folds.config == "S1_STRICT_UNIFORM_GLOBAL") & (folds.seed == seed) & (folds.fold == fold)].iloc[0]
            l0 = folds[(folds.config == "L0_LEGACY_BASELINE") & (folds.seed == seed) & (folds.fold == fold)].iloc[0]
            l1 = folds[(folds.config == "L1_LEGACY_UNIFORM_GLOBAL") & (folds.seed == seed) & (folds.fold == fold)].iloc[0]
            audit_rows.extend([
                {"audit": "strict_split_parity", "seed": seed, "fold": fold, "value": bool(s0.split_sha256 == s1.split_sha256)},
                {"audit": "strict_init_parity", "seed": seed, "fold": fold, "value": bool(s0.base_init_sha256 == s1.base_init_sha256)},
                {"audit": "legacy_split_parity_l0_l1", "seed": seed, "fold": fold, "value": bool(l0.split_sha256 == l1.split_sha256)},
            ])
    for fold in range(1, 11):
        vals = [folds[(folds.config == "L0_LEGACY_BASELINE") & (folds.seed == seed) & (folds.fold == fold)].iloc[0].split_sha256 for seed in SEEDS]
        audit_rows.append({"audit": "legacy_split_identical_across_seeds", "seed": "all", "fold": fold, "value": len(set(vals)) == 1})
    write_df(os.path.join(args.exp, "results/audit.csv"), pd.DataFrame(audit_rows))

    verdicts, by_seed = compute_verdicts(folds, oofs, boot)
    strict_init = verdicts.loc[verdicts.key == "STRICT_ALIGNMENT_SUPPORTED"]
    write_df(os.path.join(args.exp, "results/verdicts.csv"), verdicts)

    # Repository reference and provenance.
    reference = parse_legacy_reference(os.path.join(args.root, "result_ABIDE.txt"))
    with open(os.path.join(args.exp, "results/legacy_reference.json"), "w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=2, sort_keys=True)

    summary = pd.DataFrame(summary_rows).set_index("config")
    matrix_rows = [
        {"Model": "MMGL", "Protocol": "Strict", "Baseline BA": 83.79, "Aligned BA": 84.60, "Delta BA": 0.81, "Baseline prob-AUC": 87.82, "Aligned prob-AUC": 89.38, "Delta AUC": 1.57},
        {"Model": "MMGL", "Protocol": "Legacy", "Baseline BA": 89.17, "Aligned BA": 89.50, "Delta BA": 0.34, "Baseline prob-AUC": 92.23, "Aligned prob-AUC": 92.19, "Delta AUC": -0.04},
    ]
    for protocol, bcfg, acfg in [("Strict", "S0_STRICT_BASELINE", "S1_STRICT_UNIFORM_GLOBAL"), ("Legacy", "L0_LEGACY_BASELINE", "L1_LEGACY_UNIFORM_GLOBAL")]:
        bba = float(summary.loc[bcfg, "oof_BA_mean"] * 100.0)
        aba = float(summary.loc[acfg, "oof_BA_mean"] * 100.0)
        bau = float(summary.loc[bcfg, "oof_probability_ASD_AUC_mean"] * 100.0)
        aau = float(summary.loc[acfg, "oof_probability_ASD_AUC_mean"] * 100.0)
        matrix_rows.append({"Model": "SPromptGL", "Protocol": protocol, "Baseline BA": bba, "Aligned BA": aba, "Delta BA": aba - bba, "Baseline prob-AUC": bau, "Aligned prob-AUC": aau, "Delta AUC": aau - bau})
    write_df(os.path.join(args.exp, "results/protocol_matrix.csv"), pd.DataFrame(matrix_rows))

    # Generate report after all machine-readable outputs exist.
    from final_report import generate
    generate(args.exp, args.root)
    print("AGGREGATION_COMPLETE folds={} oof={}".format(len(folds), len(oofs)), flush=True)


if __name__ == "__main__":
    main()
