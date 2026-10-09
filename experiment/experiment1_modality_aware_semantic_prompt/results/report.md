# 实验一：模态感知语义提示增强（MASPE）

## 执行状态

正式实验完成 120/120 folds，产生 10,452 条 OOF 预测（4 组配置 × 3 seeds × 871 名受试者）。模型选择仅使用 outer-train 内部验证集 ACC；test 折标签只在 checkpoint 固定后用于最终统计。

输入保持用户指定的已处理数据不变：871×320 特征，PHENO/ANAT/FUNC/Correlation 维度分别为 48/6/10/256。所有图传播读取全体受试者特征，评估属于 transductive；只有 outer-train 内部 train 标签参与监督图和分类损失。全体标准化与历史 RFE 的预处理范围没有在本实验重算，仍是可能的泄漏来源。

P0 前向 parity：原始代码和实验副本的可训练参数量相同，prompt 原型、增强特征、输出和隐藏表示最大绝对误差均为 0。P0/P1/P2/P3 的同 fold 初始参数由同一随机种子生成并显式复制，参数 SHA 一致。

## 主结果

下表的均值和标准差以每个 seed 完整 871 人 OOF 指标为单位（n=3）；fold 间标准差另列在 `summary.csv`，不与 seed 波动混合。

| Config | BA mean±SD | prob-AUC mean±SD | ACC | ASD SEN | NC SPE | F1 (ASD) |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 84.50±0.79% | 90.39±0.55% | 84.65±0.88% | 82.46±0.76% | 86.54±2.04% | 83.26±0.74% |
| P1 | 84.34±0.18% | 90.60±0.32% | 84.50±0.23% | 82.13±0.66% | 86.54±0.93% | 83.06±0.14% |
| P2 | 84.77±0.54% | 90.29±1.04% | 84.88±0.57% | 83.21±2.01% | 86.32±2.06% | 83.59±0.61% |
| P3 | 84.66±1.44% | 90.97±0.84% | 84.81±1.44% | 82.63±1.97% | 86.68±1.82% | 83.42±1.57% |

## 配对比较

Δ 为百分点。bootstrap 按受试者进行 2,000 次成簇重采样，并在所有 seed/config 中共享同一受试者抽样；三个 seed 对同一人的重复预测不会当作独立受试者。区间是描述性 paired bootstrap 95% percentile CI，不替代独立外部验证。

| 比较 | 指标 | 完整 OOF seed Δ mean±SD | paired fold Δ mean±SD | fold wins / 30 | participant bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| P1-P0 | balanced_accuracy | -0.17±0.62 | -0.18±3.30 | 12/30 | [-1.03, +0.74] |
| P1-P0 | probability_auc_asd_positive | +0.21±0.29 | +0.80±1.84 | 19/30 | [-0.45, +0.83] |
| P2-P1 | balanced_accuracy | +0.43±0.42 | +0.44±2.50 | 15/30 | [-0.37, +1.25] |
| P2-P1 | probability_auc_asd_positive | -0.31±0.84 | -0.50±1.61 | 12/30 | [-0.88, +0.19] |
| P3-P1 | balanced_accuracy | +0.32±1.29 | +0.33±3.36 | 14/30 | [-0.57, +1.21] |
| P3-P1 | probability_auc_asd_positive | +0.37±1.16 | -0.18±1.54 | 12/30 | [-0.18, +0.91] |

## 语义机制检查

`prompt_analysis.csv` 汇总各模态的主题原型余弦、局部相似度、两个主题的平均 softmax 权重和 prompt 加权前后的局部 token 相对变化。它们仅用于描述模型如何使用冻结语义向量，不用于调参。

| Config | 模态 | 原型余弦均值 | 相对 token 变化 | theme 1 / 2 平均权重 | 塌缩 folds |
|---|---|---:|---:|---:|---:|
| P0 | ANAT | 0.9201±0.2186 | 0.4865±0.0325 | 0.499 / 0.501 | 25/30 |
| P0 | Correlation | 0.9201±0.2186 | 0.4821±0.0426 | 0.501 / 0.499 | 25/30 |
| P0 | FUNC | 0.9201±0.2186 | 0.4867±0.0317 | 0.500 / 0.500 | 25/30 |
| P0 | PHENO | 0.9201±0.2186 | 0.4846±0.0353 | 0.495 / 0.505 | 25/30 |
| P1 | ANAT | 0.9965±0.0090 | 0.4965±0.0065 | 0.500 / 0.500 | 28/30 |
| P1 | Correlation | 0.8153±0.2739 | 0.4683±0.0437 | 0.499 / 0.501 | 16/30 |
| P1 | FUNC | 0.9655±0.0886 | 0.4902±0.0175 | 0.501 / 0.499 | 23/30 |
| P1 | PHENO | 0.9860±0.0509 | 0.4949±0.0118 | 0.500 / 0.500 | 27/30 |
| P2 | ANAT | 0.9556±0.1837 | 0.4946±0.0124 | 0.500 / 0.500 | 26/30 |
| P2 | Correlation | 0.8046±0.3348 | 0.4765±0.0363 | 0.498 / 0.502 | 15/30 |
| P2 | FUNC | 0.9632±0.1822 | 0.4975±0.0050 | 0.500 / 0.500 | 26/30 |
| P2 | PHENO | 0.9596±0.1820 | 0.4953±0.0084 | 0.500 / 0.500 | 27/30 |
| P3 | ANAT | 0.9840±0.0360 | 0.4934±0.0120 | 0.500 / 0.500 | 25/30 |
| P3 | Correlation | 0.7528±0.3521 | 0.4609±0.0570 | 0.500 / 0.500 | 17/30 |
| P3 | FUNC | 0.9931±0.0159 | 0.4946±0.0082 | 0.500 / 0.500 | 26/30 |
| P3 | PHENO | 0.9901±0.0268 | 0.4947±0.0091 | 0.500 / 0.500 | 26/30 |

## 科学解释

P1 相对 P0 的完整 OOF 差异为 BA -0.17 pp、prob-AUC +0.21 pp； 相对固定错配 P2 为 BA -0.43 pp、prob-AUC +0.31 pp； 相对数量匹配 P3 为 BA -0.32 pp、prob-AUC -0.37 pp。 至少一项主指标/比较未同时满足正向均值与配对 bootstrap 区间下界大于 0，因此证据不足以支持“模态感知语义在各项对照下稳定优越”。需分别解释 BA 与 AUC、seed 稳定性及置信区间，不把单一指标提升扩大为总体机制证明。 按完整 OOF 的三 seed 均值，P1 的 BA 未超过 P0。

四个 prompt 组共享相同的 GFR、跨模态注意力/Transformer、分类器、可训练 MLP、优化器、训练/验证/测试索引和训练预算。P0 的 8 条原始 prompt 在四个模态之间共享；P1 每模态使用 8 条特征定义驱动的 prompt；P2 使用固定错配四循环；P3 为 32 条组织和语言复杂度相近的通用 prompt。

## 可复现产物

- 折级结果：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/fold_results.csv`
- seed 级 OOF 汇总：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/seed_summary.csv`
- 主表：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/summary.csv`
- 全部预测：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/oof_predictions.csv`
- 配对比较：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/paired_comparisons.csv`
- prompt 机制：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/results/prompt_analysis.csv`
- 参数/数据/prompt 哈希：`/data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt/manifest.json`
- 项目交接：`/data/adodas/SPromptGL-main/handoff.md`

结果对应用户给定 shell 的训练参数；论文实现段落公开的 ABIDE 参数不同，差异已记录在 `audit.md`。不据此宣称复现论文报告的 92.30% 数值。
