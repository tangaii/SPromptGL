# 实验 83.1 正式复现报告

单卡实际 worker 数为 8。四组各完成 10 折，40 个 Best 均通过独立重放。
ACC 和 AUC 为逐折均值 ± 总体标准差。AUC 按原始源码对 argmax 硬预测做 one-hot 后计算 macro OVR。

| 方法 | ACC | AUC | ΔACC 对论文 | ΔAUC 对论文 |
|---|---:|---:|---:|---:|
| Original SPromptGL | 94.81 ± 2.95% | 95.59 ± 2.57% | +0.00 pp | +0.00 pp |
| E28-E0 | 87.13 ± 5.11% | 90.16 ± 3.35% | -7.68 pp | -5.43 pp |
| E43.1-E00 | 88.78 ± 3.96% | 90.73 ± 2.88% | -6.03 pp | -4.86 pp |
| E74.4-G2 | 92.47 ± 3.19% | 93.98 ± 2.74% | -2.34 pp | -1.61 pp |
| E82.4.6.10-G1 TADPOLE Core-Method Adaptation | 92.31 ± 3.35% | 93.29 ± 2.91% | -2.50 pp | -2.30 pp |

论文对照取自 Table 2。[SPromptGL, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)

## 同折变化

| 对比 | ΔACC | ΔAUC | ACC 改善 / 下降 / 持平 | AUC 改善 / 下降 / 持平 | 净正确人数 |
|---|---:|---:|---:|---:|---:|
| G2-G1 | +1.65 pp | +0.58 pp | 4 / 4 / 2 | 5 / 5 / 0 | +10 |
| G3-G2 | +3.69 pp | +3.25 pp | 9 / 0 / 1 | 9 / 1 / 0 | +22 |
| G4-G3 | -0.16 pp | -0.69 pp | 3 / 3 / 4 | 5 / 5 / 0 | -1 |
| G4-G1 | +5.18 pp | +3.14 pp | 8 / 1 / 1 | 8 / 2 / 0 | +31 |

逐折差异与正确性变化见 [fold_contrasts.csv](results/fold_contrasts.csv)。
G3 相比 G2 同时改变了原始模态消息融合、关系监督、warmup、优化器与权重衰减。G3−G2 是历史方法整体差异。
G4 与 G3 每折共有模型、优化器、关系对和 RNG 的 warmup 档案一致。

## E83 对照

E83 保留原始执行结果，状态为 `COMPLETE_EXECUTION / METHOD_FIDELITY_FAILED`。下表 AUC 取 E83 的硬预测 AUC。

| 组 | E83 ACC | E83 AUC | E83.1−E83 ACC | E83.1−E83 AUC |
|---|---:|---:|---:|---:|
| E28-E0 | 84.44% | 87.51% | +2.68 pp | +2.64 pp |
| E43.1-E00 | 84.10% | 87.69% | +4.68 pp | +3.04 pp |
| E74.4-G2 | 81.59% | 84.89% | +10.89 pp | +9.09 pp |
| E82.4.6.10-G1 TADPOLE Core-Method Adaptation | 82.77% | 86.34% | +9.54 pp | +6.95 pp |

## 方法与限制

论文协议在外层 CV 前完成全数据 RFE、均值填补与标准化，患者图使用全部特征，外层测试折按历史规则参与 Best 选择。这些步骤可能使估计偏乐观，比较限定在该历史协议内。
六个 TADPOLE 模态均使用历史非 CORR 规则。NC、MCI、AD 使用三个高斯类条件分布。SCMR 对原生头前两类别形成同一有向对比，谱证据、Jacobian 和引导共用该方向。
八条 CLIP RN50 文本描述通用脑结构与功能异常。MCI 与 AD 之间的医学语义区分有限。
G4 保留 SCMR 谱与证据、校准、残差和梯度更新机制，病人集合使用 TADPOLE 原始十折。历史 ABIDE Fold8 人工患者加入未迁移。

执行审计 PASS，方法保真审计 PASS，Checkpoint 独立重放 PASS。 [保真矩阵](audits/method_fidelity_matrix.md) · [最终审计](audits/final_audit.json)

## Checkpoint 数值审计

40 份 Latest 与对应 Best 哈希、停止轮数、首次最佳 ACC 和折身份相符。G3/G4 的共用 warmup 状态、优化器状态、RNG 和关系对在 10 个同折配对中一致。G3 的 10 个 Best 均出现正、负边重加权，疾病图中性化和关闭直接残差均改变了输出。G4 的 10 个 Best 均重算出相同的训练谱基及统计银行，关闭 SCMR 残差会精确回到同参数原生输出。实测 CAM 相对残差最大 7.99389%，共享梯度增量与原梯度的最大范数比为 0.000999884，两项均在历史上限内。

首批任务期间，方法矩阵的文档更新触发门禁哈希校验。重新核对后只启动尚未运行的第 8–10 折，已完成的 28 折保留。两批在同一个源码版本上执行，实际 worker 数均为 8。[正式协议](protocol.json) · [独立重放](audits/checkpoint_replay.json)

