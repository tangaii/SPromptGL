# DARG Disease-Aware Patient Relation Quality Analysis

## 1. 实验目标

本实验冻结正式 `E82.4.6.10-G1` 的十折 Best Checkpoint，诊断 DARG 是否在原始 GraphLearn 患者图的有效连接上学习到疾病区分性的重加权。核心指标是排除自环后的 Cross-Disease Propagation Weight Ratio，使用正式 GCN 的对称归一化 `W = normalize_adj(A + I)`。指标只描述传播系数层面的图质量，不等同于消息幅度或分类性能的因果变化。

## 2. 与 MMGL Fig.4–5 的关系

分析思路借鉴 MMGL 对患者表示相似性和错误图连接的诊断，但没有复用错误连接条数。DARG 保留 GraphLearn 原始图支持，只对候选有效边进行关系重加权，因此本实验比较原始图和更新图在归一化传播权重上的跨疾病比例。

## 3. 原始图与 DARG 更新公式

对每个模态先得到原始邻接 `S_m`。候选边上的关系分数为 `q_m`，门控为 `gate_m`。CORR 使用 `relation = 2q_m - 1`，其余模态对 `q >= 0.55` 或 `q <= 0.45` 的边使用该关系值，不确定分数的增量为零。更新为 `A_m = S_m + DeltaA_m`，其中 `DeltaA_m = S_m * (0.8 * gate_m * relation)`，随后按源码对称写回。Positive-DCR 的直接增量为 `max(DeltaA_m, 0)`。负向关系增量在本报告中称为边权抑制，不称为负权图传播。

四个模态顺序为 PHENO、ANAT、FUNC、CORR，输入维度为 48、6、10、256。原始图、统一图、增量、q、候选边、门控和融合权重均来自封存的 `soft_positive_dcr.py` 实际前向。

## 4. 模型与身份

正式任务为 `E82.4.6.10-G1`，checkpoint 内部模型为 `E82.4.6-G2 SCMR-DA-SRPE`，DARG 为 `E74.4-G2 Positive-DCR`。ABIDE A 共 871 名受试者，seed=0，十折，ASD=0、TD=1。Fold 2、3、4、8 通过封存的历史成员适配器恢复。十折均读取对应 `foldXX.best.pt`，未使用 Latest。

十折正式预测与归档 Best evaluation 逐项相等，独立历史 evaluator 逐项相等。每折 checkpoint SHA、来源、Best epoch 和折身份写入 `results/run_manifest.json`。DARG 源码 SHA-256 为 `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`，封存 CADR 源码 SHA-256 为 `f5032d96c9ac68afda47a253f794ddfe41373fabebbbbe27057f1425f988815d`。

## 5. 评价范围与负控

主范围为至少一端属于当前折未参与 DARG 标签监督的测试患者。严格敏感性范围为两端均属于该集合。Fold 2、3、4、8 的历史训练—测试重叠患者从新增机制评价对象中排除，历史训练集和模型状态没有改变。

Score-Permuted DARG 使用同一冻结模型、同一原始图、同一候选边、同一门控和同一 q 数值集合，只通过局部生成器置乱 q 与患者边的对应关系。每折每模态固定 20 次置乱，不改变正式全局 RNG。`disease_neutral` 验证恢复原始图。

## 6. 十折结果

主范围的均值为：

| 模态 | Original (%) | DARG (%) | Score-Permuted (%) | DARG-Original (pp) | 改善/持平/下降折数 |
|---|---:|---:|---:|---:|---:|
| PHENO | 49.8816 | 49.8816 | 49.8816 | +0.0000 | 5/0/5 |
| ANAT | 49.4690 | 49.4690 | 49.4690 | -0.0000 | 6/2/2 |
| FUNC | 49.2332 | 49.2332 | 49.2332 | -0.0000 | 8/0/2 |
| CORR | 48.1945 | 25.4142 | 48.2344 | -22.7802 | 10/0/0 |

严格 test-test 范围的 DARG-Original 配对变化为 PHENO `+0.0000 pp`、ANAT `+0.0000 pp`、FUNC `-0.0000 pp`、CORR `-19.1829 pp`。有效边数量没有按权重阈值裁剪，主范围平均为 PHENO 15157.5、ANAT 26852.9、FUNC 13371.5、CORR 9409.0。严格范围的有效边数量由各折测试样本量决定，完整逐折数值在 CSV 中保存。

CORR 的 DARG 图在主范围平均同病边调整为 `+0.4630`，异病边调整为 `-0.4498`，增强边比例 `53.04%`，抑制边比例 `46.96%`，邻接相对变化 `66.90%`。q 的候选边均值为同病 `0.8645`、异病 `0.1297`。PHENO、ANAT、FUNC 的邻接相对变化均低于 `0.005%`，同病和异病平均调整量接近零，方向没有形成稳定的跨折证据。

CORR 主范围十折 Original / DARG / Score-Permuted (%) 依次为：

| Fold | Original | DARG | Score-Permuted |
|---:|---:|---:|---:|
| 1 | 49.1968 | 32.5975 | 49.0815 |
| 2 | 49.6284 | 25.5208 | 49.5720 |
| 3 | 48.5272 | 26.4131 | 48.7911 |
| 4 | 46.7431 | 22.6895 | 46.6800 |
| 5 | 48.8393 | 22.5995 | 48.7859 |
| 6 | 45.3349 | 19.9880 | 45.4102 |
| 7 | 47.7125 | 24.5224 | 47.6514 |
| 8 | 48.9748 | 26.5982 | 49.0007 |
| 9 | 49.8409 | 27.6834 | 50.1167 |
| 10 | 47.1466 | 25.5298 | 47.2549 |

CORR 主范围的 Score-Permuted 折均值为 `48.2344%`，折间 SD 为 `1.3763 pp`。20 次置乱重复的折内 SD 均值为 `0.5280 pp`。置乱结果接近 Original，且明显高于 DARG，支持改善依赖 q 与具体患者边的对应关系。

## 7. Figure

Figure X(a) 固定 Fold 1，仅展示测试患者之间的 CORR 有效连接，按真实 ASD、TD 标签分组排序。原始不存在的边为掩码，颜色显示 `DeltaA_CORR / S_CORR`，零为中性，红色为增强，蓝色为抑制。实际有限值范围为 `[-0.7673, 0.5778]`，颜色显示采用对称 99% 分位范围 `[-0.7487, 0.7487]`，178 个有限值中 2 个仅被颜色裁剪，数值统计未裁剪。

Figure X(b) 展示十折 CORR 主范围 Original 与 DARG 的逐折配对线，以及 20 次 Score-Permuted 的汇总区间。没有隐藏下降折。

英文 Figure Caption：

*Disease-Aware Patient Relation Analysis. (a) Fold 1 CORR edges among held-out patients, ordered by ASD and TD, showing the signed relative DARG adjustment on the original GraphLearn support; red and blue indicate enhancement and suppression, respectively, and absent edges are masked. (b) Ten-fold paired cross-disease propagation-weight ratios for the original and DARG graphs, with the score-permuted DARG negative-control distribution. Ratios use the formal symmetric normalization of each updated adjacency after excluding self-loops and historically supervised test-overlap patients.*

## 8. 审计

十折审计状态为 `PASS`。逐折通过 checkpoint SHA、正式预测、独立 evaluator、矩阵尺寸与有限性、对称性、`DeltaA=A-S`、Positive-DCR 正向截断、disease-neutral 原图恢复、候选边支持、归一化有限且非负、置乱 q 多重集合保持和标签事后评价隔离。独立 CSV 复算验证 280 行结果均有 Original、DARG、Score-Permuted 三个条件。

## 9. 科研判断

结论为 `PARTIAL_RELATION_QUALITY_SUPPORT`。CORR 显示稳定且与置乱负控区分明显的疾病关系质量改善，同病边增强、异病边抑制，十折方向一致。PHENO、ANAT、FUNC 的变化极小且跨折不稳定，因此不能声称四个模态均改善。该结果支持 DARG 在 CORR 模态上学习到疾病区分性的患者边权调整，但不证明最终 GCN 消息或分类准确率的因果提高。

## 10. 历史限制与运行记录

结果继承正式历史的 `val=test` checkpoint 选择、CV 前 RFE/填补/标准化、transductive 患者图和 19 名训练—测试重叠患者。重叠患者只从本机制分析对象中排除，没有修改历史训练或重新训练。十折冻结分析耗时约 `25.6 s`。运行代码为 `run_analysis.py`，结果为 `graph_quality_by_fold.csv`、`summary.json`、`audit_status.json`、`run_manifest.json`，图为 `darg_relation_quality.pdf` 和 `darg_relation_quality.png`，Fold 1 热图数据为 `corr_fold1_heatmap.npz`。
