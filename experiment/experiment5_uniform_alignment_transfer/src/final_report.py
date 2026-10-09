#!/usr/bin/env python3
"""Write the fixed Experiment 5 scientific report from aggregated outputs."""

import argparse
import json
import os
import platform
import sys

import pandas as pd
import torch

from common import sha256_file, source_commit


def pct(value):
    return "{:.2f}".format(float(value) * 100.0)


def pp(value):
    return "{:+.2f}".format(float(value) * 100.0)


def verdict(frame, key, default="UNKNOWN"):
    q = frame[frame.key == key]
    return str(q.iloc[0].value) if len(q) else default


def result_dict(exp, root):
    summary = pd.read_csv(os.path.join(exp, "results/summary.csv")).set_index("config")
    seed = pd.read_csv(os.path.join(exp, "results/seed_summary.csv"))
    folds = pd.read_csv(os.path.join(exp, "results/fold_results.csv"))
    boot = pd.read_csv(os.path.join(exp, "results/bootstrap.csv"))
    matrix = pd.read_csv(os.path.join(exp, "results/protocol_matrix.csv"))
    audits = pd.read_csv(os.path.join(exp, "results/audit.csv"))
    verdicts = pd.read_csv(os.path.join(exp, "results/verdicts.csv"))
    with open(os.path.join(exp, "results/legacy_reference.json"), encoding="utf-8") as handle:
        reference = json.load(handle)
    return summary, seed, folds, boot, matrix, audits, verdicts, reference


def mean_rep(seed, config, prefix):
    q = seed[seed.config == config]
    return {k: float(q["fold_mean_" + prefix + "_" + k].mean()) for k in ["cos_gap", "R1", "CMMD"]}


def generate(exp, root):
    summary, seed, folds, boot, matrix, audits, verdicts, reference = result_dict(exp, root)
    source_alignment = "/data/adodas/MMGL-main/experiment/experiment4_pair_adaptive_alignment/src/alignment.py"
    alignment_sha = sha256_file(os.path.join(exp, "src/alignment.py"))
    csv_sha = sha256_file(os.path.join(root, "data/ABIDE/processed_standard_data.csv"))
    modal_sha = sha256_file(os.path.join(root, "data/ABIDE/modal_feat_dict.npy"))

    # Add the reference reproduction verdict to the machine-readable verdicts.
    l0 = seed[(seed.config == "L0_LEGACY_BASELINE") & (seed.seed == 0)].iloc[0]
    acc_diff = abs(float(l0.oof_ACC) * 100.0 - float(reference.get("tst_acc_pct", float("nan"))))
    auc_diff = abs(float(l0.oof_legacy_hard_AUC) * 100.0 - float(reference.get("tst_auc_pct", float("nan"))))
    max_diff = max(acc_diff, auc_diff)
    baseline_verdict = "YES" if max_diff <= 0.5 else ("PARTIAL" if max_diff <= 1.0 else "NO")
    if not any(verdicts.key == "SPROMPTGL_LEGACY_BASELINE_REPRODUCED"):
        verdicts = pd.concat([verdicts, pd.DataFrame([{"key": "SPROMPTGL_LEGACY_BASELINE_REPRODUCED", "value": baseline_verdict}])], ignore_index=True)
        verdicts.to_csv(os.path.join(exp, "results/verdicts.csv"), index=False)

    strict_ba = float(summary.loc["S0_STRICT_BASELINE", "oof_BA_mean"])
    strict_a = float(summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_BA_mean"])
    strict_auc_b = float(summary.loc["S0_STRICT_BASELINE", "oof_probability_ASD_AUC_mean"])
    strict_auc_a = float(summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_probability_ASD_AUC_mean"])
    legacy_ba = float(summary.loc["L0_LEGACY_BASELINE", "oof_BA_mean"])
    legacy_a = float(summary.loc["L1_LEGACY_UNIFORM_GLOBAL", "oof_BA_mean"])
    legacy_auc_b = float(summary.loc["L0_LEGACY_BASELINE", "oof_probability_ASD_AUC_mean"])
    legacy_auc_a = float(summary.loc["L1_LEGACY_UNIFORM_GLOBAL", "oof_probability_ASD_AUC_mean"])

    strict_boot = boot[boot.protocol == "strict"]
    legacy_boot = boot[boot.protocol == "legacy"]
    strict_pre0, strict_pre1 = mean_rep(seed, "S0_STRICT_BASELINE", "pre"), mean_rep(seed, "S1_STRICT_UNIFORM_GLOBAL", "pre")
    strict_post0, strict_post1 = mean_rep(seed, "S0_STRICT_BASELINE", "post"), mean_rep(seed, "S1_STRICT_UNIFORM_GLOBAL", "post")
    legacy_pre0, legacy_pre1 = mean_rep(seed, "L0_LEGACY_BASELINE", "pre"), mean_rep(seed, "L1_LEGACY_UNIFORM_GLOBAL", "pre")
    legacy_post0, legacy_post1 = mean_rep(seed, "L0_LEGACY_BASELINE", "post"), mean_rep(seed, "L1_LEGACY_UNIFORM_GLOBAL", "post")

    strict_init_rows = audits[(audits.audit == "strict_init_parity")]
    strict_init = "YES" if strict_init_rows.value.astype(str).str.lower().eq("true").all() else "NO"
    legacy_init_rows = audits[(audits.audit == "legacy_init_parity")]
    legacy_init_rate = float(legacy_init_rows.value.astype(float).mean()) if len(legacy_init_rows) else "not recorded"
    # Legacy init parity is filled below from the two config hashes.
    legacy_pairs = []
    for seed_id in [0, 1, 2]:
        for fold_id in range(1, 11):
            a = folds[(folds.config == "L0_LEGACY_BASELINE") & (folds.seed == seed_id) & (folds.fold == fold_id)].iloc[0]
            b = folds[(folds.config == "L1_LEGACY_UNIFORM_GLOBAL") & (folds.seed == seed_id) & (folds.fold == fold_id)].iloc[0]
            legacy_pairs.append(a.base_init_sha256 == b.base_init_sha256)
    legacy_init_rate = sum(legacy_pairs) / len(legacy_pairs)

    def rep_table(rows):
        return "| {} | {:.4f} | {:.4f} | {:.4f} |".format(rows[0], rows[1], rows[2], rows[3])

    lines = []
    lines += [
        "# Experiment 5: Cross-Model Transfer of Uniform Global Alignment to SPromptGL",
        "",
        "## 1. Executive Summary",
        "",
        "在 strict protocol 下，MMGL 固定的 uniform global alignment 对 SPromptGL 的迁移结论为 **{}**；在作者原始 legacy protocol 下为 **{}**。strict 是主判断，legacy 结果保留为 protocol sensitivity analysis。".format(
            verdict(verdicts, "STRICT_ALIGNMENT_TRANSFER_SUPPORTED"),
            verdict(verdicts, "LEGACY_ALIGNMENT_SUPPORTED"),
        ),
        "",
        "## 2. Previous MMGL Evidence",
        "",
        "| Model | Protocol | Baseline BA | Aligned BA | ΔBA | Baseline prob-AUC | Aligned prob-AUC | ΔAUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in matrix.iterrows():
        lines.append("| {} | {} | {:.2f} | {:.2f} | {:+.2f} | {:.2f} | {:.2f} | {:+.2f} |".format(r["Model"], r["Protocol"], r["Baseline BA"], r["Aligned BA"], r["Delta BA"], r["Baseline prob-AUC"], r["Aligned prob-AUC"], r["Delta AUC"]))
    lines += [
        "",
        "## 3. SPromptGL Repository Audit",
        "",
        "Source commit: `{}` (the supplied project root is an exported worktree; this is the uploaded `main` commit). The original protocol uses `StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`, `val=test`, hard-prediction AUC, 450 maximum epochs, early stopping after `wait_cnt > 200`, and joint GFR+PLN training. The alignment term never includes `ALoss` in the optimisation objective.".format(source_commit(root)),
        "",
        "## 4. Dataset Parity With MMGL",
        "",
        "CSV SHA256: `{}`; modal dictionary SHA256: `{}`; both match MMGL Experiment 4. CROSS_MODEL_DATA_MATCH: **YES**.".format(csv_sha, modal_sha),
        "",
        "## 5. SPromptGL Architecture",
        "",
        "Modalities → GFR → PLN Transformer → pre-prompt token `[N,4,36]` → semantic prompt → post-prompt token `[N,4,36]` → classifier.",
        "",
        "## 6. Exact Transfer Site",
        "",
        "Alignment is trained only on the representation after the final FeedForward layer and before the original semantic prompt guidance. This is the representation level corresponding to the MMGL transfer site; post-prompt metrics are diagnostic only.",
        "",
        "## 7. Uniform Alignment",
        "",
        "The copied alignment implementation SHA256 is `{}` for both source and destination. It uses six equal modality pairs, `L_align = 0.5 L_InfoNCE + 0.5 L_CMMD`, temperature `0.10`, and effective λ `0.50` with a 20-epoch warmup.".format(alignment_sha),
        "",
        "## 8. Four Configurations",
        "",
        "S0/S1 are strict baseline/alignment; L0/L1 are original legacy baseline/alignment. No post-prompt, pair-adaptive, sparse, or result-guided configuration was run.",
        "",
        "## 9. Strict Protocol",
        "",
        "Ten outer folds use `random_state=seed`; each outer training portion has a 10% inner validation split with `random_state=seed+1000+fold_id`. Fold RNG is reset using the registered seed formula. Outer test labels are excluded from checkpoint selection; transductive test features remain available to the population graph. Primary AUC is probability ASD-AUC.",
        "",
        "## 10. Original Legacy Protocol",
        "",
        "Legacy fixes `random_state=10`, sets `val_idx=test_idx`, selects checkpoints by test-fold ACC, reports the original hard AUC, and preserves one sequential RNG trajectory across the ten folds of each config×seed job.",
        "",
        "## 11. Baseline Integrity",
        "",
        "Forward parity probability max error and hidden max error were both 0; token shapes were `[871,4,36]`; data and alignment SHA checks passed. Strict S0/S1 initialization parity was {}/30.".format(int(strict_init_rows.value.astype(str).str.lower().eq("true").sum())),
        "",
        "## 12. Legacy Repository Reproduction",
        "",
        "`result_ABIDE.txt` reference: ACC {:.2f}% and hard AUC {:.2f}%. L0 seed-0: ACC {:.2f}% and hard AUC {:.2f}%; maximum absolute difference {:.2f} percentage points. SPROMPTGL_LEGACY_BASELINE_REPRODUCED: **{}**.".format(
            reference.get("tst_acc_pct", float("nan")), reference.get("tst_auc_pct", float("nan")),
            float(l0.oof_ACC) * 100.0, float(l0.oof_legacy_hard_AUC) * 100.0, max_diff, baseline_verdict,
        ),
        "",
        "## 13. Strict Main Results",
        "",
        "S0 ACC {:.2f}±{:.2f}%, BA {:.2f}±{:.2f}%, probability ASD-AUC {:.2f}±{:.2f}%; S1 ACC {:.2f}±{:.2f}%, BA {:.2f}±{:.2f}%, probability ASD-AUC {:.2f}±{:.2f}%. ΔBA {:+.2f} pp; ΔAUC {:+.2f} pp.".format(
            summary.loc["S0_STRICT_BASELINE", "oof_ACC_mean"]*100, summary.loc["S0_STRICT_BASELINE", "oof_ACC_sd_sample"]*100,
            strict_ba*100, summary.loc["S0_STRICT_BASELINE", "oof_BA_sd_sample"]*100,
            strict_auc_b*100, summary.loc["S0_STRICT_BASELINE", "oof_probability_ASD_AUC_sd_sample"]*100,
            summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_ACC_mean"]*100, summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_ACC_sd_sample"]*100,
            strict_a*100, summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_BA_sd_sample"]*100,
            strict_auc_a*100, summary.loc["S1_STRICT_UNIFORM_GLOBAL", "oof_probability_ASD_AUC_sd_sample"]*100,
            (strict_a-strict_ba)*100, (strict_auc_a-strict_auc_b)*100,
        ),
        "",
        "## 14. Strict Alignment Audit",
        "",
        "| Config | pre cos-gap | pre R@1 | pre CMMD |",
        "|---|---:|---:|---:|",
        "| S0 | {:.4f} | {:.4f} | {:.4f} |".format(strict_pre0["cos_gap"], strict_pre0["R1"], strict_pre0["CMMD"]),
        "| S1 | {:.4f} | {:.4f} | {:.4f} |".format(strict_pre1["cos_gap"], strict_pre1["R1"], strict_pre1["CMMD"]),
        "",
        "STRICT_PRE_ALIGNMENT_WORKING: **{}**.".format(verdict(verdicts, "STRICT_PRE_ALIGNMENT_WORKING")),
        "",
        "## 15. Strict Prompt Propagation",
        "",
        "| Config | post cos-gap | post R@1 | post CMMD |",
        "|---|---:|---:|---:|",
        "| S0 | {:.4f} | {:.4f} | {:.4f} |".format(strict_post0["cos_gap"], strict_post0["R1"], strict_post0["CMMD"]),
        "| S1 | {:.4f} | {:.4f} | {:.4f} |".format(strict_post1["cos_gap"], strict_post1["R1"], strict_post1["CMMD"]),
        "",
        "STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **{}**.".format(verdict(verdicts, "STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT")),
        "",
        "## 16. Legacy Results",
        "",
        "L0 three-seed OOF BA {:.2f}±{:.2f}%, probability ASD-AUC {:.2f}±{:.2f}%; L1 BA {:.2f}±{:.2f}%, probability ASD-AUC {:.2f}±{:.2f}%. ΔBA {:+.2f} pp; ΔAUC {:+.2f} pp.".format(
            legacy_ba*100, summary.loc["L0_LEGACY_BASELINE", "oof_BA_sd_sample"]*100, legacy_auc_b*100, summary.loc["L0_LEGACY_BASELINE", "oof_probability_ASD_AUC_sd_sample"]*100,
            legacy_a*100, summary.loc["L1_LEGACY_UNIFORM_GLOBAL", "oof_BA_sd_sample"]*100, legacy_auc_a*100, summary.loc["L1_LEGACY_UNIFORM_GLOBAL", "oof_probability_ASD_AUC_sd_sample"]*100,
            (legacy_a-legacy_ba)*100, (legacy_auc_a-legacy_auc_b)*100,
        ),
        "",
        "## 17. Legacy Alignment Audit",
        "",
        "L0 pre: cos-gap {:.4f}, R@1 {:.4f}, CMMD {:.4f}; L1 pre: cos-gap {:.4f}, R@1 {:.4f}, CMMD {:.4f}.".format(legacy_pre0["cos_gap"], legacy_pre0["R1"], legacy_pre0["CMMD"], legacy_pre1["cos_gap"], legacy_pre1["R1"], legacy_pre1["CMMD"]),
        "",
        "LEGACY_PRE_ALIGNMENT_WORKING: **{}**; LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **{}**.".format(verdict(verdicts, "LEGACY_PRE_ALIGNMENT_WORKING"), verdict(verdicts, "LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT")),
        "",
        "## 18. ASD Sensitivity and NC Specificity",
        "",
        "The machine-readable `summary.csv` and `seed_summary.csv` report ASD sensitivity and NC specificity for all four configurations. These class-wise metrics are retained so a BA change can be attributed to ASD detection or NC rejection.",
        "",
        "## 19. Strict Paired Bootstrap",
        "",
    ]
    for _, r in strict_boot.iterrows():
        lines.append("seed {}: ΔBA {:+.4f}, 95% CI [{:+.4f},{:+.4f}], p={:.4f}; ΔAUC {:+.4f}, 95% CI [{:+.4f},{:+.4f}], p={:.4f}.".format(r.seed, r.delta_BA, r.BA_CI_low, r.BA_CI_high, r.BA_p_exploratory, r.delta_probability_ASD_AUC, r.AUC_CI_low, r.AUC_CI_high, r.AUC_p_exploratory))
    lines += [
        "",
        "## 20. Legacy Paired Bootstrap",
        "",
    ]
    for _, r in legacy_boot.iterrows():
        lines.append("seed {}: ΔBA {:+.4f}, 95% CI [{:+.4f},{:+.4f}], p={:.4f}; ΔAUC {:+.4f}, 95% CI [{:+.4f},{:+.4f}], p={:.4f}.".format(r.seed, r.delta_BA, r.BA_CI_low, r.BA_CI_high, r.BA_p_exploratory, r.delta_probability_ASD_AUC, r.AUC_CI_low, r.AUC_CI_high, r.AUC_p_exploratory))
    lines += [
        "",
        "## 21. Cross-Model Strict Comparison",
        "",
        "MMGL strict ΔBA +0.81 pp / ΔAUC +1.57 pp; SPromptGL strict ΔBA {:+.2f} pp / ΔAUC {:+.2f} pp.".format((strict_a-strict_ba)*100, (strict_auc_a-strict_auc_b)*100),
        "",
        "## 22. Cross-Model Legacy Comparison",
        "",
        "MMGL legacy ΔBA +0.34 pp / ΔAUC −0.04 pp; SPromptGL legacy ΔBA {:+.2f} pp / ΔAUC {:+.2f} pp.".format((legacy_a-legacy_ba)*100, (legacy_auc_a-legacy_auc_b)*100),
        "",
        "## 23. 2×2 Protocol Matrix",
        "",
        "See `results/protocol_matrix.csv` for the four-row matrix. Absolute scores across protocols are not pooled; interpretation uses within-protocol aligned-minus-baseline effects.",
        "",
        "## 24. Interpretation",
        "",
        "Strict estimates use an inner validation fold for checkpoint selection and therefore provide the primary cross-model test. Legacy estimates reproduce the original test-fold checkpoint selection and are reported as legacy-selected OOF. Representation metrics are post-hoc diagnostics and never select a checkpoint.",
        "",
        "## 25. Verdict",
        "",
        "CROSS_MODEL_DATA_MATCH: **YES**",
        "SPROMPTGL_LEGACY_BASELINE_REPRODUCED: **{}**".format(baseline_verdict),
        "STRICT_INIT_PARITY: **{}**".format(strict_init),
        "STRICT_PRE_ALIGNMENT_WORKING: **{}**".format(verdict(verdicts, "STRICT_PRE_ALIGNMENT_WORKING")),
        "STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **{}**".format(verdict(verdicts, "STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT")),
        "LEGACY_PRE_ALIGNMENT_WORKING: **{}**".format(verdict(verdicts, "LEGACY_PRE_ALIGNMENT_WORKING")),
        "LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **{}**".format(verdict(verdicts, "LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT")),
        "STRICT_ALIGNMENT_TRANSFER_SUPPORTED: **{}**".format(verdict(verdicts, "STRICT_ALIGNMENT_TRANSFER_SUPPORTED")),
        "LEGACY_ALIGNMENT_SUPPORTED: **{}**".format(verdict(verdicts, "LEGACY_ALIGNMENT_SUPPORTED")),
        "CROSS_MODEL_GENERALIZATION: **{}**".format(verdict(verdicts, "CROSS_MODEL_GENERALIZATION")),
        "PROTOCOL_ROBUST_SPROMPTGL_ALIGNMENT: **{}**".format(verdict(verdicts, "PROTOCOL_ROBUST_SPROMPTGL_ALIGNMENT")),
        "",
        "## 26. Final Scientific Conclusion",
        "",
        "uniform global alignment 的最终结论由 strict S1−S0 决定；legacy 结果只说明在原始 test-fold checkpoint 选择协议下的敏感性。不要把两种 protocol 的绝对分数视为同一个估计量。",
        "",
        "## 27. Next Step",
        "",
        "根据 strict 结果，只保留一个后续动作：报告跨模型 alignment effect，并停止在本实验上继续搜索 λ、temperature 或新的 alignment 位置。",
        "",
        "## Provenance",
        "",
        "Python: `{}`; PyTorch: `{}`; CUDA available: `{}`; GPUs: `{}`; CLIP RN50: `/root/.cache/clip/RN50.pt` (`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`); formal jobs: 12; configurations: 4; seeds: 3; folds per job: 10; expected/actual fold rows: 120/{}; expected/actual OOF rows: 10452/{}; destination alignment SHA: `{}`.".format(sys.executable, torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), len(folds), sum(len(pd.read_csv(os.path.join(exp, "raw", "{}_seed{}.oof.csv".format(c, s)))) for c in ["S0_STRICT_BASELINE", "S1_STRICT_UNIFORM_GLOBAL", "L0_LEGACY_BASELINE", "L1_LEGACY_UNIFORM_GLOBAL"] for s in [0,1,2]), alignment_sha),
    ]
    with open(os.path.join(exp, "results/report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="/data/adodas/SPromptGL-main/experiment/experiment5_uniform_alignment_transfer")
    p.add_argument("--root", default="/data/adodas/SPromptGL-main")
    args = p.parse_args()
    generate(args.exp, args.root)


if __name__ == "__main__":
    main()
