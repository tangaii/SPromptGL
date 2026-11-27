# DARG Disease-Aware Patient Relation Quality Analysis

## 1. 实验目标与源码核验

本次工作只优化冻结 DReSP 机制 Figure，不重新训练、不微调、不修改正式模型、Checkpoint、DARG/DPSL 封存源码或历史训练结果。正式任务为 E82.4.6.10-G1，内部模型为 E82.4.6-G2 SCMR-DA-SRPE，DARG 为 E74.4-G2 Positive-DCR。服务器关键运行文件与 GitHub 提交 9b0bc6b 的 raw 文件逐字 SHA-256 一致。

封存实现使用 GraphLearn 原始邻接 S_m，在其真实有效支持上根据 q 和 gate 计算 DeltaA_m，形成 A_m = S_m + DeltaA_m。CORR 使用 2q-1，其余模态遵守源码中的 0.45/0.55 资格范围。Positive-DCR 对 direct 增量使用 max(DeltaA_m, 0)。传播权重按照正式 normalize_adj(A_m + I) 计算，自环只用于归一化而不进入评价边集合。

## 2. Panel (a) 最终设计

A1 为严格 clean_test_test 范围的十折统计。每个折先在原始图真实存在的非自环边上计算 (A_DARG-A_Original)/A_Original，分别对同病边和跨病边求折内均值，再汇总十折。红色为 Same-Disease，蓝色为 Cross-Disease，灰线连接同一折，箱线图只提供分布概览，不遮挡十个实际点。

A2 为客观选择的代表折辅助热力图。选择规则是 CORR 原始 clean test-test 有效边密度最接近十折中位数，距离相同取较小折号。热力图中的浅灰色表示原始图无边，白色表示存在边且相对调整为零附近，红色表示增强，蓝色表示抑制。所有缺失边均被掩码，颜色显示范围以零对称，极端值裁剪只影响颜色映射。

## 3. Panel (a) 十折结果

| Fold | Clean test n | Same edges | Cross edges | Same mean (%) | Cross mean (%) | Same enhanced | Cross suppressed | Original density |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 88 | 47 | 42 | 37.5566 | -35.3158 | 97.87% | 83.33% | 2.3250% |
| 2 | 79 | 590 | 619 | 44.6473 | -36.5601 | 85.08% | 77.06% | 39.2405% |
| 3 | 82 | 72 | 84 | 33.5310 | -39.5712 | 76.39% | 79.76% | 4.6974% |
| 4 | 83 | 99 | 112 | 34.7710 | -37.6724 | 76.77% | 74.11% | 6.2004% |
| 5 | 87 | 360 | 379 | 43.2528 | -41.3492 | 80.28% | 76.52% | 19.7541% |
| 6 | 87 | 60 | 62 | 59.0095 | -56.7835 | 96.67% | 91.94% | 3.2612% |
| 7 | 87 | 250 | 218 | 48.2609 | -37.1143 | 88.80% | 77.98% | 12.5100% |
| 8 | 85 | 479 | 449 | 35.9849 | -28.9757 | 76.20% | 69.49% | 25.9944% |
| 9 | 87 | 74 | 82 | 50.9134 | -34.3214 | 89.19% | 74.39% | 4.1700% |
| 10 | 87 | 220 | 238 | 37.4495 | -37.0189 | 82.73% | 76.89% | 12.2427% |

十折同病相对调整均值为 42.5377%，跨病为 -38.4682%。同病增强比例按折均值为 85.00%，跨病抑制比例按折均值为 78.15%。
代表折为 Fold 04。其 clean test-test 原始 CORR 有效边为 211，患者数为 83，边密度为 6.2004%。十折密度中位数为 9.2216%，该折到中位数的绝对距离为 3.0212 个百分点。
Fold 04 热图有限相对调整值范围为 [-0.7924, 0.7639]，颜色显示对称范围为 [-0.7912, 0.7912]，被颜色裁剪的有限值为 4。

## 4. Panel (b) 十折结果

Panel (b) 使用主评价范围 at_least_one_clean_test。Original 与 DARG 是同一正式 DReSP Checkpoint 内部的 GraphLearn 原图和 DARG 重加权图，不是单独训练的基线模型。

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
Score-Permuted DARG 对每个折使用 20 次置乱，保持原始图、候选边、门控和 q 数值多重集合不变，只置乱 q 与患者边的对应关系。对置乱编号 p，先收集十折结果，再计算该 p 的十折均值，得到 20 个十折平均值。图中橙色散点就是这 20 个均值，误差线明确表示这 20 个均值的最小值至最大值，不是置信区间。置乱均值为 48.2344%，实际范围为 [47.8079%, 48.7299%]。

## 5. 四模态和融合辅助汇总

下表保留上一轮正式图质量结果中的四模态十折主范围汇总。FUSED 是历史辅助代理项，不作为真实多模态 GCN 融合传播图，也不进入本 Figure 的主结论。

| Modality | Original mean (%) | DARG mean (%) | Score-Permuted mean (%) | Improved / unchanged / declined |
|---|---:|---:|---:|---:|
| PHENO | 49.8816 | 49.8816 | 49.8816 | 5/0/5 |
| ANAT | 49.4690 | 49.4690 | 49.4690 | 6/2/2 |
| FUNC | 49.2332 | 49.2332 | 49.2332 | 8/0/2 |
| CORR | 48.1945 | 25.4142 | 48.2344 | 10/0/0 |
| FUSED | 49.5443 | 46.3652 | not stored | 10/0/0 |

## 6. 审计与历史限制

独立审计状态为 PASS。逐折检查正式 Best Checkpoint 身份、归档预测和独立 evaluator、原图/DARG 图共同来源、矩阵尺寸/有限性/对称性/非负性、DeltaA=A_DARG-A_Original、Positive-DCR 正向截断、无新边、正式归一化、disease-neutral 原图恢复、标签事后评价隔离、q 多重集合和置乱支持均由实际条件判断。原始主指标与历史结果逐折一致：True。

分析继承历史 val=test Checkpoint 选择、CV 前 RFE/填补/标准化、transductive 患者图和 19 名训练—测试重叠事实。重叠患者只从机制评价对象中排除，没有修改历史训练集或模型状态。该 Figure 只支持传播系数层面的机制结论，不直接证明分类准确率因果提高。

## 7. Figure Caption

Disease-Aware Patient Relation Analysis. (a) Ten-fold relative edge-weight adjustments on the strict clean test-test support, computed within each fold for same-disease and cross-disease patient pairs before averaging across folds. Points show all folds, with paired lines linking the same fold and boxplots summarizing the fold distribution. The auxiliary heatmap shows the signed adjustment on the original CORR GraphLearn support for the objectively selected fold whose clean test-test edge density is closest to the ten-fold median; absent original edges are masked. (b) Ten-fold paired cross-disease propagation-weight ratios from the same frozen DReSP checkpoints before and after DARG reweighting, together with 20 score-permuted negative-control means. Each permutation mean is first computed across the ten folds, and the orange range denotes the minimum to maximum of these 20 permutation means. Ratios use the model's symmetric normalization of A+I with self-loops excluded from the evaluation edges.

## 8. 文件与校验

最终 Figure：figures/darg_relation_quality.pdf、figures/darg_relation_quality.png。可复核 Figure 数据：results/figure_data.npz。逐折原始图质量数据保留于 results/graph_quality_by_fold.csv。代码和图文件 SHA：{"pdf": "4e232b62b9d76007fb2ee804e7cd3e55690180fb1d4f332902746f7a7a6a1e71", "png": "38a7ccb4fbd7916d84d0b5a18befac71960f89e7965adf7a049ce63e7160af89", "data": "806633ee6fe0161bd7262c1c355c522984180a09bb26d2cc02f719e4d79e1fca"}。
