# DARG Disease-Aware Patient Relation Quality Analysis

## 1. 实验目标与源码核验

本次工作只优化冻结 DReSP 机制 Figure，不重新训练、不微调、不修改正式模型、Checkpoint、DARG/DPSL 封存源码或历史训练结果。正式任务为 E82.4.6.10-G1，内部模型为 E82.4.6-G2 SCMR-DA-SRPE，DARG 为 E74.4-G2 Positive-DCR。服务器运行文件与 GitHub 提交 85826c2 的对应 raw 文件逐项 SHA-256 一致。

封存实现使用 GraphLearn 原始邻接 S_m，在其真实有效支持上根据 q 和 gate 计算 DeltaA_m，形成 A_m = S_m + DeltaA_m。Positive-DCR 的 direct 增量为 max(DeltaA_m, 0)。传播权重按照正式 normalize_adj(A_m + I) 计算，自环只用于归一化而不进入评价边集合。

## 2. Panel (a) 三组十折关系调整

Panel (a) 只使用 CORR 模态和严格 clean_test_test 范围。每个折先在原始图真实存在的非自环无向边上计算 (A_DARG-A_Original)/A_Original，并分别对 ASD-ASD、ASD-TD、TD-TD 三组求折内均值。最终柱高是十个折内均值的等权平均，误差线是总体标准差 SD = np.std(fold_values, ddof=0)。柱子颜色按实际均值方向决定，红色表示增强，蓝色表示削弱。

| Group | Mean (%) | SD (%) | Total valid edges | Mean enhanced | Mean suppressed |
|---|---:|---:|---:|---:|---:|
| ASD-ASD | 47.6497 | 12.4734 | 937 | 89.10% | 10.90% |
| ASD-TD | -38.4682 | 6.8587 | 2285 | 21.85% | 78.15% |
| TD-TD | 40.3348 | 12.5428 | 1314 | 83.34% | 16.66% |

每个折的三组统计如下，边数为严格 clean_test_test 的原始 GraphLearn 支持中的上三角非自环边。

| Fold | ASD-ASD mean (%) | ASD-TD mean (%) | TD-TD mean (%) | ASD-ASD n | ASD-TD n | TD-TD n |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.8733 | -35.3158 | 44.5701 | 21 | 42 | 26 |
| 2 | 54.1749 | -36.5601 | 36.7175 | 268 | 619 | 322 |
| 3 | 62.9389 | -39.5712 | 24.4413 | 17 | 84 | 55 |
| 4 | 63.5271 | -37.6724 | 15.2753 | 40 | 112 | 59 |
| 5 | 37.8396 | -41.3492 | 47.2085 | 152 | 379 | 208 |
| 6 | 63.4734 | -56.7835 | 54.8336 | 29 | 62 | 31 |
| 7 | 40.9998 | -37.1143 | 53.0216 | 99 | 218 | 151 |
| 8 | 43.9655 | -28.9757 | 31.6738 | 168 | 449 | 311 |
| 9 | 49.4060 | -34.3214 | 53.1243 | 44 | 82 | 30 |
| 10 | 31.2981 | -37.0189 | 42.4824 | 99 | 238 | 121 |

按每折实际 ASD-ASD 与 TD-TD 边数合并后再对十折等权汇总，同病均值为 42.5377%；ASD-TD 结果为 -38.4682%。历史同病参考值为 42.5377%，历史异病参考值为 -38.4682%，两项一致性检查分别为 True 和 True。三组均值按分组张量重算的同病合并值为 42.53769675%，与直接同病边集合的差异为 0.00000069 个百分点，属于 float32 索引求和顺序的数值舍入差异。

## 3. Panel (b) 跨疾病传播权重

Panel (b) 使用 at_least_one_clean_test。Original 与 DARG 是同一正式冻结 DReSP Checkpoint 内部的 GraphLearn 原图和 DARG 重加权图，不是重新训练的 SPromptGL 基线模型。纵轴为跨疾病归一化传播权重占比，使用 normalize_adj(A+I) 并排除自环评价边。

| Fold | Original (%) | DARG (%) | DARG−Original (pp) |
|---:|---:|---:|---:|
| 1 | 49.1968 | 32.5975 | -16.5993 |
| 2 | 49.6284 | 25.5208 | -24.1076 |
| 3 | 48.5272 | 26.4131 | -22.1141 |
| 4 | 46.7431 | 22.6895 | -24.0536 |
| 5 | 48.8393 | 22.5995 | -26.2398 |
| 6 | 45.3349 | 19.9880 | -25.3469 |
| 7 | 47.7125 | 24.5224 | -23.1901 |
| 8 | 48.9748 | 26.5982 | -22.3766 |
| 9 | 49.8409 | 27.6834 | -22.1576 |
| 10 | 47.1466 | 25.5298 | -21.6168 |

Original 十折均值为 48.1945%，DARG 为 25.4142%，配对平均下降 22.7802 pp，十折改善 10/10。严格 clean_test_test 的复算结果为 Original 51.6350%、DARG 32.4521%，平均下降 19.1829 pp。
Score-Permuted DARG 对每个折使用 20 次置乱，保持原始图、候选边、门控和 q 数值多重集合不变，只置乱 q 与患者边的对应关系。对每个置乱编号先收集十折结果，再计算该编号的十折均值，得到 20 个均值。其平均为 48.2344%，最小至最大为 [47.8079%, 48.7299%]。该范围是 20 个置乱均值的 Min–Max，不是置信区间。

## 4. 四模态历史辅助汇总

下表保留原始逐折结果的四模态主范围汇总。FUSED 是历史辅助代理项，不作为真实多模态 GCN 融合传播图，也不进入本 Figure 的主结论。

| Modality | Original mean (%) | DARG mean (%) | Score-Permuted mean (%) | Improved / unchanged / declined |
|---|---:|---:|---:|---:|
| PHENO | 49.8816 | 49.8816 | 49.8816 | 5/0/5 |
| ANAT | 49.4690 | 49.4690 | 49.4690 | 6/2/2 |
| FUNC | 49.2332 | 49.2332 | 49.2332 | 8/0/2 |
| CORR | 48.1945 | 25.4142 | 48.2344 | 10/0/0 |
| FUSED | 49.5443 | 46.3652 | not stored | 10/0/0 |

## 5. 独立审计

独立审计状态为 PASS。关键条件均由实际判断执行，包括十折 Best Checkpoint 身份、归档预测、独立 evaluator、原图/DARG 图共同来源、矩阵有限/对称/非负、DeltaA 精确一致、Positive-DCR 正向截断、候选支持不变、正式归一化、disease-neutral 原图恢复、标签只用于事后评价、三组边计数不重复、逐折 Mean±SD、历史合并统计和置乱 q 多重集合。

分析继承历史 val=test Checkpoint 选择、CV 前 RFE/填补/标准化、transductive 患者图和 19 名训练—测试重叠事实。重叠患者只从机制评价对象中排除，没有修改历史训练集或模型状态。图质量下降表示跨疾病传播系数占比下降，不直接证明分类准确率的因果提高。

## 6. Figure Caption

Disease Relation Reweighting and Cross-Disease Propagation. (a) Ten-fold mean relative edge-weight adjustments for CORR ASD-ASD, ASD-TD, and TD-TD patient connections on the strict clean test-test support; each bar is the equal-weight mean of the ten fold-level means and error bars show population SD (ddof=0). (b) Ten-fold paired cross-disease propagation-weight ratios for the Original GraphLearn and DARG graphs from the same frozen DReSP checkpoints, with 20 score-permuted ten-fold means as a negative control. The ratios use the model's symmetric normalization of A+I and exclude self-loops; a reduction denotes a lower cross-disease propagation-weight fraction, not a classification-accuracy change.

## 7. 文件与校验

最终 Figure：figures/darg_relation_quality.pdf、figures/darg_relation_quality.png。可复核 Figure 数据：results/figure_data.npz。逐折原始图质量数据保留于 results/graph_quality_by_fold.csv。代码和图文件 SHA：{"pdf": "e8121be4dcdf711859a7b5dea6b699d53233535b0542c9e4a42ae917bc923798", "png": "3dcbad7a00437710b9f3bed575261d264679180927ea93d78b0135a68902a752", "data": "aa389bd3e0dc140cab37e044cb564111b626d56632d06ace46493128601532b5"}。
