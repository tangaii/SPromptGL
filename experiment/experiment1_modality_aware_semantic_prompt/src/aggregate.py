from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from common import (
    EXP_DIR,
    MODALITIES,
    PROMPT_FILES,
    RESULT_DIR,
    RAW_DIR,
    atomic_json,
    load_data,
    prompt_digest,
    sha256_file,
    split_path,
    task_checkpoint_path,
    task_pred_path,
    task_prompt_analysis_path,
    task_raw_path,
)
from artifacts import fold_artifacts_valid
from handoff import update_handoff


CONFIGS = ("P0", "P1", "P2", "P3")
SEEDS = (0, 1, 2)


def metrics(y_true, p_asd, p_nc):
    y_true = np.asarray(y_true, dtype=np.int64)
    probability = np.column_stack([np.asarray(p_asd, dtype=np.float64), np.asarray(p_nc, dtype=np.float64)])
    pred = probability.argmax(axis=1)
    positive = (y_true == 0).astype(np.int64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "probability_auc_asd_positive": float(roc_auc_score(positive, probability[:, 0])),
        "accuracy": float(accuracy_score(y_true, pred)),
        "asd_sensitivity": float(np.mean(pred[y_true == 0] == 0)),
        "nc_specificity": float(np.mean(pred[y_true == 1] == 1)),
        "f1_asd_positive": float(f1_score(positive, (pred == 0).astype(np.int64), zero_division=0)),
    }


def paired_bootstrap(oof_by_config, labels_by_seed, reference, target, repetitions=2000, seed=20261009):
    rng = np.random.default_rng(seed)
    participant_count = len(labels_by_seed[0])
    output = {}
    for metric_name in ("balanced_accuracy", "probability_auc_asd_positive"):
        deltas = np.empty(repetitions, dtype=np.float64)
        for b in range(repetitions):
            sampled = rng.integers(0, participant_count, size=participant_count)
            y_parts = []
            ref_parts = []
            target_parts = []
            for cv_seed in SEEDS:
                y_parts.append(labels_by_seed[cv_seed][sampled])
                ref_parts.append(oof_by_config[reference][cv_seed].iloc[sampled][["prob_asd", "prob_nc"]].to_numpy())
                target_parts.append(oof_by_config[target][cv_seed].iloc[sampled][["prob_asd", "prob_nc"]].to_numpy())
            y_all = np.concatenate(y_parts)
            ref_all = np.concatenate(ref_parts)
            target_all = np.concatenate(target_parts)
            ref_metrics = metrics(y_all, ref_all[:, 0], ref_all[:, 1])
            target_metrics = metrics(y_all, target_all[:, 0], target_all[:, 1])
            deltas[b] = 100 * (target_metrics[metric_name] - ref_metrics[metric_name])
        output[metric_name] = {
            "lower_95": float(np.percentile(deltas, 2.5)),
            "upper_95": float(np.percentile(deltas, 97.5)),
            "repetitions": int(repetitions),
            "unit": "participant cluster; same sampled row indices shared across all seeds/configurations",
        }
    return output


def ensure_complete_inputs():
    _, y, _, _ = load_data()
    all_fold_rows = []
    predictions = {}
    prompt_frames = []
    for config in CONFIGS:
        predictions[config] = {}
        for seed in SEEDS:
            split_doc = json.loads(split_path(seed).read_text(encoding="utf-8"))
            seed_pred = []
            for split in split_doc["folds"]:
                fold = int(split["fold"])
                summary_path = task_raw_path(config, seed, fold)
                pred_path = task_pred_path(config, seed, fold)
                prompt_path = task_prompt_analysis_path(config, seed, fold)
                checkpoint_path = task_checkpoint_path(config, seed, fold)
                if not fold_artifacts_valid(summary_path, pred_path, prompt_path, checkpoint_path, split["test"]):
                    raise RuntimeError(f"incomplete/invalid fold: {config} seed={seed} fold={fold}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                prediction = pd.read_csv(pred_path)
                prompt_frame = pd.read_csv(prompt_path)
                if len(prediction) != len(split["test"]):
                    raise RuntimeError(f"wrong prediction count: {pred_path}")
                seed_pred.append(prediction)
                all_fold_rows.append(summary)
                prompt_frames.append(prompt_frame)
            full_oof = pd.concat(seed_pred, ignore_index=True).sort_values("row_index").reset_index(drop=True)
            if len(full_oof) != len(y) or full_oof["row_index"].duplicated().any():
                raise RuntimeError(f"OOF is incomplete or has duplicate subjects: {config} seed={seed}")
            if set(full_oof.row_index.astype(int)) != set(range(len(y))):
                raise RuntimeError(f"OOF has missing row indices: {config} seed={seed}")
            if not np.array_equal(full_oof.y_true.to_numpy(dtype=np.int64), y):
                raise RuntimeError(f"OOF target labels mismatch for {config} seed={seed}")
            predictions[config][seed] = full_oof
    if len(all_fold_rows) != 120:
        raise RuntimeError(f"expected 120 fold records, got {len(all_fold_rows)}")
    if sum(len(frame) for cfg in predictions.values() for frame in cfg.values()) != 10452:
        raise RuntimeError("expected exactly 10,452 OOF participant predictions")
    return y, predictions, pd.DataFrame(all_fold_rows), pd.concat(prompt_frames, ignore_index=True)


def build_results():
    y, oof_by_config, fold_df, prompt_detail = ensure_complete_inputs()
    fold_df = fold_df.sort_values(["config", "seed", "fold"]).reset_index(drop=True)
    fold_df.to_csv(RESULT_DIR / "fold_results.csv", index=False)

    seed_rows = []
    oof_rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            frame = oof_by_config[config][seed]
            result = metrics(frame.y_true, frame.prob_asd, frame.prob_nc)
            seed_rows.append({"config": config, "seed": seed, "n_subjects": len(frame), **result})
            oof_rows.append(frame)
    seed_df = pd.DataFrame(seed_rows).sort_values(["config", "seed"])
    seed_df.to_csv(RESULT_DIR / "seed_summary.csv", index=False)
    combined_oof = pd.concat(oof_rows, ignore_index=True).sort_values(["config", "seed", "row_index"]).reset_index(drop=True)
    combined_oof.to_csv(RESULT_DIR / "oof_predictions.csv", index=False)

    metric_columns = [
        "balanced_accuracy",
        "probability_auc_asd_positive",
        "accuracy",
        "asd_sensitivity",
        "nc_specificity",
        "f1_asd_positive",
    ]
    summary_rows = []
    for config in CONFIGS:
        seed_rows_cfg = seed_df[seed_df.config == config]
        fold_rows_cfg = fold_df[fold_df.config == config]
        row = {"config": config, "completed_folds": int(len(fold_rows_cfg)), "seed_count": int(seed_rows_cfg.seed.nunique())}
        for metric_name in metric_columns:
            row[f"{metric_name}_mean"] = float(seed_rows_cfg[metric_name].mean())
            row[f"{metric_name}_std"] = float(seed_rows_cfg[metric_name].std(ddof=1))
            row[f"fold_{metric_name}_mean"] = float(fold_rows_cfg[metric_name].mean())
            row[f"fold_{metric_name}_std"] = float(fold_rows_cfg[metric_name].std(ddof=1))
        row["mean_best_epoch"] = float(fold_rows_cfg.best_epoch.mean())
        row["std_best_epoch"] = float(fold_rows_cfg.best_epoch.std(ddof=1))
        row["mean_train_seconds"] = float(fold_rows_cfg.train_seconds.mean())
        row["total_train_seconds"] = float(fold_rows_cfg.train_seconds.sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULT_DIR / "summary.csv", index=False)

    comparison_rows = []
    bootstrap_reports = {}
    labels_by_seed = {seed: oof_by_config["P0"][seed].sort_values("row_index").y_true.to_numpy() for seed in SEEDS}
    for target in ("P1", "P2", "P3"):
        target_vs = "P0" if target == "P1" else "P1"
        seed_deltas = []
        for seed in SEEDS:
            base = oof_by_config[target_vs][seed]
            alt = oof_by_config[target][seed]
            paired = base.merge(alt, on=["row_index", "y_true"], suffixes=("_base", "_target"), validate="one_to_one")
            if len(paired) != len(y):
                raise RuntimeError(f"unpaired OOF subjects in {target}-{target_vs} seed={seed}")
            base_metric = metrics(paired.y_true, paired.prob_asd_base, paired.prob_nc_base)
            alt_metric = metrics(paired.y_true, paired.prob_asd_target, paired.prob_nc_target)
            seed_deltas.append({
                "seed": seed,
                "balanced_accuracy_delta_pp": 100 * (alt_metric["balanced_accuracy"] - base_metric["balanced_accuracy"]),
                "probability_auc_delta_pp": 100 * (alt_metric["probability_auc_asd_positive"] - base_metric["probability_auc_asd_positive"]),
            })
        seed_delta_df = pd.DataFrame(seed_deltas)
        fold_base = fold_df[fold_df.config == target_vs].set_index(["seed", "fold"])
        fold_alt = fold_df[fold_df.config == target].set_index(["seed", "fold"])
        for metric_name, output_name in (("balanced_accuracy", "balanced_accuracy_delta_pp"), ("probability_auc_asd_positive", "probability_auc_delta_pp")):
            fold_delta = 100 * (fold_alt[metric_name] - fold_base[metric_name])
            comparison_rows.append({
                "comparison": f"{target}-{target_vs}",
                "target_config": target,
                "reference_config": target_vs,
                "metric": metric_name,
                "seed_oof_delta_mean_pp": float(seed_delta_df[output_name].mean()),
                "seed_oof_delta_sd_pp": float(seed_delta_df[output_name].std(ddof=1)),
                "seed0_delta_pp": float(seed_delta_df.loc[seed_delta_df.seed == 0, output_name].iloc[0]),
                "seed1_delta_pp": float(seed_delta_df.loc[seed_delta_df.seed == 1, output_name].iloc[0]),
                "seed2_delta_pp": float(seed_delta_df.loc[seed_delta_df.seed == 2, output_name].iloc[0]),
                "paired_fold_delta_mean_pp": float(fold_delta.mean()),
                "paired_fold_delta_sd_pp": float(fold_delta.std(ddof=1)),
                "paired_fold_wins": int((fold_delta > 0).sum()),
                "paired_fold_ties": int((fold_delta == 0).sum()),
                "paired_fold_count": int(len(fold_delta)),
                "bootstrap_ci_lower_95_pp": np.nan,
                "bootstrap_ci_upper_95_pp": np.nan,
            })
        bootstrap_reports[f"{target}-{target_vs}"] = paired_bootstrap(
            oof_by_config, labels_by_seed, target_vs, target
        )

    paired_df = pd.DataFrame(comparison_rows)
    for row_idx, row in paired_df.iterrows():
        metric_key = "balanced_accuracy" if row.metric == "balanced_accuracy" else "probability_auc_asd_positive"
        ci = bootstrap_reports[row.comparison][metric_key]
        paired_df.loc[row_idx, "bootstrap_ci_lower_95_pp"] = ci["lower_95"]
        paired_df.loc[row_idx, "bootstrap_ci_upper_95_pp"] = ci["upper_95"]
    paired_df.to_csv(RESULT_DIR / "paired_comparisons.csv", index=False)

    ablation = summary_df[["config", "completed_folds", "balanced_accuracy_mean", "balanced_accuracy_std", "probability_auc_asd_positive_mean", "probability_auc_asd_positive_std"]].copy()
    ablation["interpretation"] = ablation.config.map({
        "P0": "Original prompt baseline",
        "P1": "Modality-aware prompts",
        "P2": "Fixed wrong-modality prompt mapping",
        "P3": "Count-matched generic prompts",
    })
    ablation.to_csv(RESULT_DIR / "ablation_results.csv", index=False)

    prompt_analysis = prompt_detail.groupby(["config", "modality"], as_index=False).agg(
        folds=("fold", "nunique"),
        seed_fold_observations=("seed", "count"),
        prototype_theme_cosine_mean=("prototype_theme_cosine", "mean"),
        prototype_theme_cosine_std=("prototype_theme_cosine", "std"),
        collapse_fold_count=("prototype_collapse_abs_cosine_gt_0_98", "sum"),
        mean_relative_token_change=("mean_relative_token_change", "mean"),
        std_relative_token_change_between_folds=("mean_relative_token_change", "std"),
        theme1_similarity_mean=("theme1_local_similarity_mean", "mean"),
        theme1_similarity_std=("theme1_local_similarity_std", "mean"),
        theme2_similarity_mean=("theme2_local_similarity_mean", "mean"),
        theme2_similarity_std=("theme2_local_similarity_std", "mean"),
        theme1_weight_mean=("theme1_mean_softmax_weight", "mean"),
        theme2_weight_mean=("theme2_mean_softmax_weight", "mean"),
    )
    prompt_analysis.to_csv(RESULT_DIR / "prompt_analysis.csv", index=False)
    bootstrap_path = RESULT_DIR / "bootstrap.json"
    atomic_json(bootstrap_path, bootstrap_reports)

    manifest = {
        "experiment": "Experiment 1 — Modality-Aware Semantic Prompt Enhancement (MASPE)",
        "status": "COMPLETE",
        "completed_folds": 120,
        "expected_folds": 120,
        "oof_rows": int(len(combined_oof)),
        "expected_oof_rows": 10452,
        "data_sha256": {
            "processed_standard_data.csv": sha256_file(Path("/data/adodas/SPromptGL-main/data/ABIDE/processed_standard_data.csv")),
            "modal_feat_dict.npy": sha256_file(Path("/data/adodas/SPromptGL-main/data/ABIDE/modal_feat_dict.npy")),
        },
        "source_sha256": {
            path.name: sha256_file(path)
            for path in sorted((EXP_DIR / "src").glob("*.py"))
        },
        "prompt_file_sha256": {path.name: sha256_file(path) for path in sorted(PROMPT_FILES.values())},
        "prompt_semantic_sha256": {config: prompt_digest(config) for config in CONFIGS},
        "split_sha256": {f"seed{seed}.json": sha256_file(split_path(seed)) for seed in SEEDS},
        "environment": {
            "python": sys.version,
            "torch": __import__("torch").__version__,
            "cuda_available": __import__("torch").cuda.is_available(),
            "cuda_device_count": __import__("torch").cuda.device_count(),
        },
        "results_files": [
            "fold_results.csv", "seed_summary.csv", "summary.csv", "oof_predictions.csv",
            "paired_comparisons.csv", "ablation_results.csv", "prompt_analysis.csv", "report.md",
        ],
    }
    atomic_json(EXP_DIR / "manifest.json", manifest)
    write_report(summary_df, paired_df, bootstrap_reports, fold_df, seed_df, prompt_analysis)
    update_handoff()
    print(f"complete: folds={len(fold_df)} oof_rows={len(combined_oof)}")
    print(summary_df[["config", "balanced_accuracy_mean", "balanced_accuracy_std", "probability_auc_asd_positive_mean", "probability_auc_asd_positive_std"]].to_string(index=False))
    print(paired_df.to_string(index=False))


def write_report(summary, paired, bootstrap, folds, seeds, prompt_analysis):
    report = [
        "# 实验一：模态感知语义提示增强（MASPE）",
        "",
        "## 执行状态",
        "",
        "正式实验完成 120/120 folds，产生 10,452 条 OOF 预测（4 组配置 × 3 seeds × 871 名受试者）。模型选择仅使用 outer-train 内部验证集 ACC；test 折标签只在 checkpoint 固定后用于最终统计。",
        "",
        "输入保持用户指定的已处理数据不变：871×320 特征，PHENO/ANAT/FUNC/Correlation 维度分别为 48/6/10/256。所有图传播读取全体受试者特征，评估属于 transductive；只有 outer-train 内部 train 标签参与监督图和分类损失。全体标准化与历史 RFE 的预处理范围没有在本实验重算，仍是可能的泄漏来源。",
        "",
        "P0 前向 parity：原始代码和实验副本的可训练参数量相同，prompt 原型、增强特征、输出和隐藏表示最大绝对误差均为 0。P0/P1/P2/P3 的同 fold 初始参数由同一随机种子生成并显式复制，参数 SHA 一致。",
        "",
        "## 主结果",
        "",
        "下表的均值和标准差以每个 seed 完整 871 人 OOF 指标为单位（n=3）；fold 间标准差另列在 `summary.csv`，不与 seed 波动混合。",
        "",
        "| Config | BA mean±SD | prob-AUC mean±SD | ACC | ASD SEN | NC SPE | F1 (ASD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"| {row.config} | {100*row.balanced_accuracy_mean:.2f}±{100*row.balanced_accuracy_std:.2f}% "
            f"| {100*row.probability_auc_asd_positive_mean:.2f}±{100*row.probability_auc_asd_positive_std:.2f}% "
            f"| {100*row.accuracy_mean:.2f}±{100*row.accuracy_std:.2f}% "
            f"| {100*row.asd_sensitivity_mean:.2f}±{100*row.asd_sensitivity_std:.2f}% "
            f"| {100*row.nc_specificity_mean:.2f}±{100*row.nc_specificity_std:.2f}% "
            f"| {100*row.f1_asd_positive_mean:.2f}±{100*row.f1_asd_positive_std:.2f}% |"
        )
    report.extend([
        "",
        "## 配对比较",
        "",
        "Δ 为百分点。bootstrap 按受试者进行 2,000 次成簇重采样，并在所有 seed/config 中共享同一受试者抽样；三个 seed 对同一人的重复预测不会当作独立受试者。区间是描述性 paired bootstrap 95% percentile CI，不替代独立外部验证。",
        "",
        "| 比较 | 指标 | 完整 OOF seed Δ mean±SD | paired fold Δ mean±SD | fold wins / 30 | participant bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in paired.iterrows():
        report.append(
            f"| {row.comparison} | {row.metric} | {row.seed_oof_delta_mean_pp:+.2f}±{row.seed_oof_delta_sd_pp:.2f} | "
            f"{row.paired_fold_delta_mean_pp:+.2f}±{row.paired_fold_delta_sd_pp:.2f} | {int(row.paired_fold_wins)}/30 "
            f"| [{row.bootstrap_ci_lower_95_pp:+.2f}, {row.bootstrap_ci_upper_95_pp:+.2f}] |"
        )
    report.extend([
        "",
        "## 语义机制检查",
        "",
        "`prompt_analysis.csv` 汇总各模态的主题原型余弦、局部相似度、两个主题的平均 softmax 权重和 prompt 加权前后的局部 token 相对变化。它们仅用于描述模型如何使用冻结语义向量，不用于调参。",
        "",
        "| Config | 模态 | 原型余弦均值 | 相对 token 变化 | theme 1 / 2 平均权重 | 塌缩 folds |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in prompt_analysis.iterrows():
        report.append(
            f"| {row.config} | {row.modality} | {row.prototype_theme_cosine_mean:.4f}±{row.prototype_theme_cosine_std:.4f} "
            f"| {row.mean_relative_token_change:.4f}±{row.std_relative_token_change_between_folds:.4f} "
            f"| {row.theme1_weight_mean:.3f} / {row.theme2_weight_mean:.3f} | {int(row.collapse_fold_count)}/30 |"
        )
    report.extend([
        "",
        "## 科学解释",
        "",
        conclusion(paired, summary),
        "",
        "四个 prompt 组共享相同的 GFR、跨模态注意力/Transformer、分类器、可训练 MLP、优化器、训练/验证/测试索引和训练预算。P0 的 8 条原始 prompt 在四个模态之间共享；P1 每模态使用 8 条特征定义驱动的 prompt；P2 使用固定错配四循环；P3 为 32 条组织和语言复杂度相近的通用 prompt。",
        "",
        "## 可复现产物",
        "",
        f"- 折级结果：`{EXP_DIR}/results/fold_results.csv`",
        f"- seed 级 OOF 汇总：`{EXP_DIR}/results/seed_summary.csv`",
        f"- 主表：`{EXP_DIR}/results/summary.csv`",
        f"- 全部预测：`{EXP_DIR}/results/oof_predictions.csv`",
        f"- 配对比较：`{EXP_DIR}/results/paired_comparisons.csv`",
        f"- prompt 机制：`{EXP_DIR}/results/prompt_analysis.csv`",
        f"- 参数/数据/prompt 哈希：`{EXP_DIR}/manifest.json`",
        f"- 项目交接：`{EXP_DIR.parents[1] / 'handoff.md'}`",
        "",
        "结果对应用户给定 shell 的训练参数；论文实现段落公开的 ABIDE 参数不同，差异已记录在 `audit.md`。不据此宣称复现论文报告的 92.30% 数值。",
    ])
    (RESULT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def conclusion(paired, summary):
    # The comparison table is built as P1-P0, P2-P1 and P3-P1.  For the
    # narrative, expose all contrasts in the same direction (P1 versus each
    # control), negating both the point estimate and the CI when needed.
    values = {}
    for metric in ("balanced_accuracy", "probability_auc_asd_positive"):
        values[("P1-P0", metric)] = paired[
            (paired.comparison == "P1-P0") & (paired.metric == metric)
        ].iloc[0]
        for requested, stored in (("P1-P2", "P2-P1"), ("P1-P3", "P3-P1")):
            source = paired[(paired.comparison == stored) & (paired.metric == metric)].iloc[0].copy()
            source.seed_oof_delta_mean_pp = -float(source.seed_oof_delta_mean_pp)
            ci_lower = float(source.bootstrap_ci_lower_95_pp)
            ci_upper = float(source.bootstrap_ci_upper_95_pp)
            source.bootstrap_ci_lower_95_pp = -ci_upper
            source.bootstrap_ci_upper_95_pp = -ci_lower
            values[(requested, metric)] = source
    primary = [values[(comp, metric)] for comp in ("P1-P0", "P1-P2", "P1-P3") for metric in ("balanced_accuracy", "probability_auc_asd_positive")]
    wins_all = all(float(row.seed_oof_delta_mean_pp) > 0 and float(row.bootstrap_ci_lower_95_pp) > 0 for row in primary)
    p1 = summary[summary.config == "P1"].iloc[0]
    p0 = summary[summary.config == "P0"].iloc[0]
    lines = [
        f"P1 相对 P0 的完整 OOF 差异为 BA {values[('P1-P0','balanced_accuracy')].seed_oof_delta_mean_pp:+.2f} pp、prob-AUC {values[('P1-P0','probability_auc_asd_positive')].seed_oof_delta_mean_pp:+.2f} pp；",
        f"相对固定错配 P2 为 BA {values[('P1-P2','balanced_accuracy')].seed_oof_delta_mean_pp:+.2f} pp、prob-AUC {values[('P1-P2','probability_auc_asd_positive')].seed_oof_delta_mean_pp:+.2f} pp；",
        f"相对数量匹配 P3 为 BA {values[('P1-P3','balanced_accuracy')].seed_oof_delta_mean_pp:+.2f} pp、prob-AUC {values[('P1-P3','probability_auc_asd_positive')].seed_oof_delta_mean_pp:+.2f} pp。",
    ]
    if wins_all:
        lines.append("三个 seed 上的均值方向以及参与者配对 bootstrap 区间均支持 P1 在两项主指标上优于 P0/P2/P3；这支持正确模态语义匹配在本冻结数据与 transductive 协议下有独立贡献，但不能消除预处理泄漏和单数据集限制。")
    else:
        lines.append("至少一项主指标/比较未同时满足正向均值与配对 bootstrap 区间下界大于 0，因此证据不足以支持“模态感知语义在各项对照下稳定优越”。需分别解释 BA 与 AUC、seed 稳定性及置信区间，不把单一指标提升扩大为总体机制证明。")
    if p1.balanced_accuracy_mean <= p0.balanced_accuracy_mean:
        lines.append("按完整 OOF 的三 seed 均值，P1 的 BA 未超过 P0。")
    return " ".join(lines)


if __name__ == "__main__":
    build_results()
