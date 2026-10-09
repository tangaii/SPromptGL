# 实验一：模态感知语义提示增强（MASPE）

状态：**全部完成**。任务 12/12 已结束，当前记录 120/120 个测试折。

## 协议与来源

P0 在隔离副本中逐字节运行根目录原始源码及 `ABIDE-simple-2-concat-weighted-cosine.sh` 参数。P1–P3 只在隔离副本的 `network.py` 应用 `prompt_patch.diff`；四组均使用根目录 `main.py`、`model.py` 的训练与评价路径。

保持 `StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`、原始 `val_idx == tst_idx`、ACC 最优 checkpoint、450 epoch 上限、200 patience、原始 hard-label AUC 和根代码提前终止条件。没有额外验证集或调参。图特征传播保持原始 transductive 方式。

数据固定为 `data/ABIDE/processed_standard_data.csv` 和 `modal_feat_dict.npy`。根代码使用的测试折同时参与 checkpoint 选择；这里如实沿用原始协议，不将其称作独立测试评估。输入文件已按要求固定，既有全队列标准化/RFE 预处理不在本实验中重做。

源码和数据哈希见 `source_hashes.json`；P1/P2/P3 提示文本见 `prompts.json`。CLIP 使用 OpenAI RN50，权重 SHA256 应为 `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`。

## P0 论文参考核对

P0 seed=0 当前完成 10/10 折；ACC 91.73% ± 2.95%，Hard-AUC 91.56% ± 2.94%。
论文参考为 ACC 92.30±2.58%，AUC 92.22±2.57%；当前与参考差值：ACC -0.57 个百分点，Hard-AUC -0.66 个百分点。论文值与本地复现的协议/指标实现是否完全同口径无法仅凭数值确认。

## 各配置结果

下表为每个 seed 的 10 折均值，再报告 seed 均值及 seed 间标准差；`fold_sd` 与 `seed_sd` 分开存于 `summary.csv`。SEN/SPE 以 ASD（label 0）为阳性类，另一类结果也保存在逐折 CSV 中。

| 配置 | 完整 seeds | ACC (%; seed mean±SD) | Hard-AUC (%; seed mean±SD) | SEN-ASD (%) | SPE-ASD (%) |
|---|---:|---:|---:|---:|---:|
| P0 | 3/3 | 91.69 ± 0.05 | 91.54 ± 0.08 | 89.66 | 93.42 |
| P1 | 3/3 | 91.80 ± 0.11 | 91.62 ± 0.12 | 89.17 | 94.06 |
| P2 | 3/3 | 91.69 ± 0.05 | 91.52 ± 0.07 | 89.25 | 93.78 |
| P3 | 3/3 | 91.54 ± 0.22 | 91.31 ± 0.22 | 88.27 | 94.35 |

## 对照与稳定性

seed=0: P1-P0 的 ACC -0.00 个百分点，Hard-AUC +0.00 个百分点。
seed=1: P1-P0 的 ACC +0.11 个百分点，Hard-AUC +0.08 个百分点。
seed=2: P1-P0 的 ACC +0.23 个百分点，Hard-AUC +0.16 个百分点。
在 3 个可配对 seed 中，P1 的 ACC 提升 2 个、Hard-AUC 提升 2 个；此处只作描述性稳定性检查，不作显著性推断。
目前没有看到 P1 在 ACC 与 Hard-AUC 上跨 seed 同时稳定超过 P0；请结合 P2/P3 的同协议结果判断收益是否来自模态语义对应。

P2 使用固定错配映射，P3 使用 32 条数量匹配通用提示。它们的逐 seed 差异以上表及 `summary.csv` 为准；当前有效完整 seed 不足时不作科学结论。

## 逐任务状态

| 配置 | Seed | 状态 | 已记录折数 |
|---|---:|---|---:|
| P0 | 0 | COMPLETE | 10/10 |
| P0 | 1 | COMPLETE | 10/10 |
| P0 | 2 | COMPLETE | 10/10 |
| P1 | 0 | COMPLETE | 10/10 |
| P1 | 1 | COMPLETE | 10/10 |
| P1 | 2 | COMPLETE | 10/10 |
| P2 | 0 | COMPLETE | 10/10 |
| P2 | 1 | COMPLETE | 10/10 |
| P2 | 2 | COMPLETE | 10/10 |
| P3 | 0 | COMPLETE | 10/10 |
| P3 | 1 | COMPLETE | 10/10 |
| P3 | 2 | COMPLETE | 10/10 |

## 解释与限制

若 P1 未优于 P0，结果支持的结论仅是：在当前原始模型和冻结 ABIDE 特征下，这组模态描述没有带来稳定分类收益。数据中的 PHENO 含站点/人口学信息，ANAT 与 FUNC 主要包含图像质量和运动/信号质量量，Correlation 为预先 RFE 选出的连接边；提示语义可能与这些压缩后的质量和混合特征并不完全对应。最终解释以完整结果为准。

测试折被用于原始验证 ACC 和 checkpoint 选择，因此不能把本实验表述为独立测试集估计。执行中不修正该协议。
