# 实验二：对比状态驱动的模态语义提示优化（CSMP）

状态：**全部完成**；新任务 6/6，V2/V2S 当前记录 60/60 折。

## 目标与固定设计

V2 使用预先固定的 32 条状态对比 Prompt（4 模态 × 2 主题 × 4 文本）。V2S 完全复用 V2 文本，仅固定错配 PHENO←FUNC、FUNC←PHENO、ANAT←Correlation、Correlation←ANAT。实验只改变 Prompt 内容，不改变模型结构、损失、优化器、训练策略、数据或评价协议。

根目录源码与 ABIDE 输入哈希、RN50 权重哈希和最小 `network.py` patch 记录在 `source_hashes.json`。Prompt 计算沿用实验一 P1：32 条 CLIP RN50 文本特征按每模态两主题拼接为 4096 维，经共享 4096→128→9 MLP 得到 [4,2,9] 原型，再与 [871,4,4,9] 局部 Token 计算 [871,4,4,2] 相似度并进行原始 Softmax、Max、加权和融合。固定 `StratifiedKFold(10, random_state=10)`、`val_idx == tst_idx`、ACC checkpoint、450 epoch、200 patience 和 hard-label AUC 均保持不变。

实验一 P0–P3 逐折结果作为只读参考快照保存于 `results/reference_fold_results.csv`，其中记录了原始 `fold_results.csv` 的 SHA256。测试折参与原始验证和 checkpoint 选择，因此不作独立测试集估计。

## 主结果

| 配置 | 来源 | 完整 seeds | ACC（seed均值±SD） | Hard-AUC（seed均值±SD） | SEN-ASD | SPE-ASD |
|---|---|---:|---:|---:|---:|---:|
| P0 | 实验一参考快照 | 3/3 | 91.69% ± 0.05% | 91.54% ± 0.08% | 89.66% | 93.42% |
| P1 | 实验一参考快照 | 3/3 | 91.80% ± 0.11% | 91.62% ± 0.12% | 89.17% | 94.06% |
| P2 | 实验一参考快照 | 3/3 | 91.69% ± 0.05% | 91.52% ± 0.07% | 89.25% | 93.78% |
| P3 | 实验一参考快照 | 3/3 | 91.54% ± 0.22% | 91.31% ± 0.22% | 88.27% | 94.35% |
| V2 | 本次运行 | 3/3 | 91.77% ± 0.05% | 91.60% ± 0.04% | 89.50% | 93.70% |
| V2S | 本次运行 | 3/3 | 91.54% ± 0.05% | 91.36% ± 0.06% | 89.16% | 93.56% |

## 必须比较的变化

### V2-P0
- acc: +0.08 个百分点
- hard_auc: +0.06 个百分点
- sensitivity_asd: -0.16 个百分点
- specificity_asd: +0.29 个百分点
### V2-P1
- acc: -0.04 个百分点
- hard_auc: -0.02 个百分点
- sensitivity_asd: +0.33 个百分点
- specificity_asd: -0.36 个百分点
### V2-V2S
- acc: +0.23 个百分点
- hard_auc: +0.24 个百分点
- sensitivity_asd: +0.34 个百分点
- specificity_asd: +0.14 个百分点
- acc 逐折胜/平/负：13/7/10（配对 30 折）。
- hard_auc 逐折胜/平/负：15/5/10（配对 30 折）。
- sensitivity_asd 逐折胜/平/负：10/14/6（配对 30 折）。
- specificity_asd 逐折胜/平/负：7/16/7（配对 30 折）。

各 seed 的 ACC 变化：
- seed=0: V2-P0 -0.00 个百分点；V2-P1 +0.00 个百分点；V2-V2S +0.11 个百分点。
- seed=1: V2-P0 +0.11 个百分点；V2-P1 +0.00 个百分点；V2-V2S +0.23 个百分点。
- seed=2: V2-P0 +0.12 个百分点；V2-P1 -0.11 个百分点；V2-V2S +0.34 个百分点。

## 文本嵌入诊断

- V1/V2 各主题四条文本平均 CLIP 向量的余弦诊断（仅描述，不用于筛选）：
  - V1：主题内余弦 PHENO=0.9091, ANAT=0.9293, FUNC=0.8756, Correlation=0.8834；全部主题非对角均值 0.8568，最大值 0.9497。
  - V2：主题内余弦 PHENO=0.8904, ANAT=0.9653, FUNC=0.9465, Correlation=0.9383；全部主题非对角均值 0.8828，最大值 0.9653。

## 新任务状态

| 配置 | Seed | 状态 | 已记录折数 |
|---|---:|---|---:|
| V2 | 0 | COMPLETE | 10/10 |
| V2 | 1 | COMPLETE | 10/10 |
| V2 | 2 | COMPLETE | 10/10 |
| V2S | 0 | COMPLETE | 10/10 |
| V2S | 1 | COMPLETE | 10/10 |
| V2S | 2 | COMPLETE | 10/10 |

## 解释与限制

V2 相对 P0 仅有轻微 ACC/Hard-AUC 提升，SEN-ASD 略低；相对 P1 的 ACC 与 Hard-AUC 略低，但 V2 正确模态映射在三 seed 和逐折汇总上优于 V2S。建议将 V2 保留为候选 Prompt 供后续独立验证，当前不建议用 V2 替换实验一 P1。
V2S 的逐折胜负只作描述性配对比较，不把交叉验证折视为相互独立的外部数据集，不根据结果生成 V3。
