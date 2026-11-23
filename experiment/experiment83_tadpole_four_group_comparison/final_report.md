# Experiment 83 — TADPOLE 四组比较报告

生成时间 UTC：2026-11-23T09:20:33.606660+00:00

执行状态：**完整完成**。四组共 40/40 个正式 fold。

## 执行环境

- GPU：PPU-ZW810E，显存约 98 GiB，单 GPU。
- PyTorch：2.9.0, CUDA 12.9, available=True
 - 实际训练并发：1 个 fold 训练循环，CPU 线程由环境变量限制为 4，总 worker 上限记录为 10。
- E83 目录：`/model/spromptgl/experiment/experiment83_tadpole_four_group_comparison`。
- 由于首选目录创建返回 OSS Input/output error，本次目录切换到 `/model/spromptgl/experiment`。

## 数据确认

- TADPOLE 598 名受试者，NC 209、MCI 315、AD 74。
- 六模态及维度：UCSFFSX 145，UCBERKELEYAV45 148，UPENNBIOMK9 3，RISK_FACTOR 36，COGNITIVE_TEST 24，ROI_AVERAGE 9。
- 处理链来自 MMGL TADPOLE notebook。稳定 DXCHANGE 1/2/3、baseline VISCODE=bl、PMCI_selection=False、全数据 RFE、外层 CV 前 StandardScaler 和均值填补均保留。
- 四组读取同一份 `processed_standard_data.csv` 和同一份十折 `split_indices.json`。

## 源码与方法链

- G1 使用 E28 的 E0_P0 原始基线，保留 PEN、GFR、CAM。
- G2 使用 E43.1 的 E00_B0B_REPLAY，PEN 物理移除。
- G3 使用 E74.4-G2 Positive-DCR，保留正向疾病感知关系残差、图融合和关系损失。
- G4 从 E82.4.6.10 向前追溯至 E82.4.6 PCSR/SCMR 谱提示链，执行 DARG→CAM→DPSL→分类。ABIDE 特有 ASD 重叠患者、二分类 margin 和历史 checkpoint 未移植。
- 完整分支、commit、依赖和文件 SHA 在 `sources/source_manifest.json`。

## 四组十折结果

| 方法 | ACC | AUC_original | AUC_probability | SPE_macro | SEN_macro | F1_macro |
|---|---:|---:|---:|---:|---:|---:|
| E28-E0 | 84.44±2.92% | 87.51±2.37% | 91.88±3.06% | 90.86±1.83% | 84.16±3.30% | 83.86±2.95% |
| E43.1-E00 | 84.10±3.71% | 87.69±2.73% | 92.26±3.17% | 90.76±2.33% | 84.61±3.37% | 83.61±3.33% |
| E74.4-G2 | 81.59±3.91% | 84.89±3.43% | 90.58±3.16% | 89.13±2.57% | 80.65±4.49% | 81.00±4.03% |
| E82.4.6.10-G1 | 82.77±3.11% | 86.34±2.63% | 90.56±2.76% | 89.80±1.85% | 82.89±3.58% | 82.52±3.44% |

## 各类别结果

| 方法 | NC Precision | NC Recall | NC F1 | MCI Precision | MCI Recall | MCI F1 | AD Precision | AD Recall | AD F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E28-E0 | 84.36±4.31% | 83.69±9.19% | 83.72±5.09% | 85.55±4.57% | 85.04±3.63% | 85.19±2.83% | 83.96±11.96% | 83.75±11.91% | 82.66±7.21% |
| E43.1-E00 | 85.16±8.07% | 83.19±11.48% | 83.35±6.61% | 85.87±4.93% | 84.04±7.28% | 84.67±3.96% | 79.88±10.06% | 86.61±8.54% | 82.82±7.80% |
| E74.4-G2 | 79.57±5.22% | 82.69±10.09% | 80.75±5.87% | 83.53±6.40% | 81.94±5.53% | 82.44±3.41% | 83.51±8.58% | 77.32±9.48% | 79.80±6.71% |
| E82.4.6.10-G1 | 82.99±7.42% | 80.79±9.06% | 81.27±4.37% | 83.78±3.61% | 83.79±6.28% | 83.60±3.17% | 81.98±7.90% | 84.11±9.16% | 82.70±6.65% |

## 原论文对照

| 方法 | ACC | AUC | ΔACC pp | ΔAUC pp |
|---|---:|---:|---:|---:|
| Original SPromptGL | 94.81±2.95% | 95.59±2.57% | N/R | N/R |
| E28-E0 | 84.44±2.92% | 87.51±2.37% | -10.37 | -8.08 |
| E43.1-E00 | 84.10±3.71% | 87.69±2.73% | -10.71 | -7.90 |
| E74.4-G2 | 81.59±3.91% | 84.89±3.43% | -13.22 | -10.70 |
| E82.4.6.10-G1 | 82.77±3.11% | 86.34±2.63% | -12.04 | -9.25 |

论文只报告 ACC 和 AUC，因此论文表中的 SEN、SPE、F1 不补造。AUC_original 按原始代码的硬 one-hot 预测计算，AUC_probability 采用三分类 macro one-vs-rest 概率 AUC。论文数字与仓库具体 AUC 统计实现的完全同一性未能从论文表格单独证明，AUC 差值应视为协议对照。

## 方法增量

- E43.1-E00 − E28-E0：ACC -0.34 pp, AUC_original +0.18 pp, AUC_probability +0.38 pp, SPE_macro -0.10 pp, SEN_macro +0.45 pp, F1_macro -0.25 pp
- E74.4-G2 − E43.1-E00：ACC -2.52 pp, AUC_original -2.80 pp, AUC_probability -1.68 pp, SPE_macro -1.63 pp, SEN_macro -3.96 pp, F1_macro -2.61 pp
- E82.4.6.10-G1 − E74.4-G2：ACC +1.19 pp, AUC_original +1.46 pp, AUC_probability -0.02 pp, SPE_macro +0.67 pp, SEN_macro +2.25 pp, F1_macro +1.52 pp
- E82.4.6.10-G1 − E28-E0：ACC -1.67 pp, AUC_original -1.17 pp, AUC_probability -1.32 pp, SPE_macro -1.07 pp, SEN_macro -1.27 pp, F1_macro -1.34 pp

G2 审计中十个 fold 均 `pen_present=false`。G3 十个 fold 的关系更新和关系梯度均非零。G4 十个 fold 的关系更新、DPSL 梯度均非零，DPSL 残差最大比例未超过 8%。

## 训练开销

| 方法 | 每 fold 秒数均值 | 每 fold 秒数标准差 |
|---|---:|---:|
| E28-E0 | 12.69 | 3.68 |
| E43.1-E00 | 12.09 | 3.28 |
| E74.4-G2 | 21.68 | 4.04 |
| E82.4.6.10-G1 | 25.03 | 5.30 |

## 可靠性与限制

- checkpoint 选择保留 `val_idx == test_idx`，best criterion 为 ACC。
- RFE 和标准化保留在外层 CV 之前的历史流程，可能产生较乐观估计。
- TADPOLE 的三分类 same/different 关系使用 NC/MCI/AD 原始类别，不将 MCI 合并。
- E82 历史实现依赖 ABIDE 二分类和人工重叠样本。本次只迁移可验证的 DARG+DPSL 方法链，未复制 ABIDE 患者编号、重叠标签或历史权重。
- 原始 CLIP RN50 权重在服务器上不可用，E83 使用 E82 runtime `prompts.pt` 中冻结的八条对应 prompt embeddings。提示文本逐字保留，没有把 ASD/TD 语义替换成 AD/NC/MCI。
- 适配使用真实六模态维度，CAM 宽度为 6×4×10=240，未复制 ABIDE 的 4 模态或 144 维硬编码。
- 训练采用论文 TADPOLE 基础设置 dropout 0、lr 0.012、hidden 10、2 层、4 heads、alpha 1。DARG 和 DPSL 的独立优化器学习率保留对应历史特有设置，并在各组 config 中记录。

## 结果文件

- `metrics/all_fold_metrics.csv`
- `metrics/all_group_summary.csv`
- `comparison/paper_comparison.csv`
- `comparison/method_comparison.csv`
- `audits/e83_audit.json`
- `audits/mechanism_summary.csv`
- `sources/source_manifest.json`
- `data_audit/data_manifest.json`、`class_distribution.json`、`modality_manifest.json`、`split_indices.json`
- 每组 `fold_metrics.csv`、`summary.json`、`train.log`、`best_checkpoints/`、`predictions/`、`confusion_matrices/`、`audits/`

## 结论

E83 四组 TADPOLE 十折训练、指标计算、机制审计和论文对照均已完成。所有结果均来自本次 TADPOLE 重新训练，未使用 ABIDE 预测、权重、fold 或患者重叠名单。
