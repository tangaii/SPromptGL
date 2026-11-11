# Experiment 78：DCPSS 正式结果

状态：**COMPLETE_AUDITED**；独立读回 50/50。

## 结论摘要

DCPSS 在 DARG-CAM 的 144 维 CAM 输出上动态聚合疾病原型，并用患者条件化的原型对比选择冻结 CLIP RN50 疾病 Prompt。训练查询按 ID 从其所属簇原型中执行 Leave-One-Out；主分类器输出由同一个原始分类头提供，语义证据以有界有符号 margin 修正加入。未使用 E77 系列 FP/FN 队列或配对检索器。

E78 的技术有效性以实际五组独立训练和同 checkpoint 机制干预判断，不以相似度或注意力可视化单独立论。固定选模仍使用测试折 ACC，历史 CV 前预处理与 full-graph transductive 传播保留，Fold 2 仍有两名历史 train/test 重叠患者；结果不构成无偏泛化估计。

## 文献与源码审计

已阅读本地 SPromptGL MICCAI 2025 论文 11 页全文（含 Fig.1–3、Section 2.1–2.2、Eq.(1)–(7)、实验设置和消融），BrainPrompt MICCAI 2025 论文 11 页全文及官方源文本文件；SPromptGL 官方代码、BrainPrompt 官方代码和本项目 GitHub 只读核验。HP²L 只能读到 ScienceDirect 公开摘要、Highlights 与方法预览（下载全文请求返回 403），未声称精读全文。详细机制和证据边界见 `results/source_review.md`。

实际 E74.4 基座为 DARG + Positive-DCR + CAM + `PLNNoPEN`，原始分类器前 CAM hidden 为 `[871,144]`，OutputLayer 返回 log-softmax；没有活动 PEN。原训练损失为带权 NLL 分类、原图损失、0.05 pair、0.01 adversarial、0.01 relation-preserve，原优化器顺序 `opt_base → opt_cam → opt_added`。DCPSS 只替换训练分类预测项并追加固定 0.05 semantic、0.02 gain 辅助项，DARG 与 CAM 接收新增分类梯度。

BrainPrompt 文本按官方文件 TC→ASD，映射 `[1,0]`；SPromptGL 原始8条按第一组TD、第二组ASD，映射 `[1,1,1,1,0,0,0,0]`。BrainPrompt 组只迁移原疾病级文本并统一使用冻结 RN50，不是完整 BrainPrompt 复现。

## 方法与固定训练

支持集为 `train_indices - explicit_test_indices`。K=3 子簇仅按每折 E61 epoch-10 warmup 的 CAM144、在类内按 PC1 稳定排序后切为3个平衡区间；成员固定，当前 epoch 原型使用当前 CAM 投影的可微均值。K=1 每类一个全局中心。原型数量不引入任何额外可训练分类层。每组新增参数均为 37,377（患者投影 4,608、文本投影 32,768、残差幅度1）。

对每个训练 Query，只从它自己的类别/簇总和中减去该患者当前投影和计数；别的簇不变。测试 Query 不属于 Support。新分支训练只使用合法外层训练患者；Fold 2 的 1-based 行377/407仍保留在原基座历史训练和指标流程中，但排除于所有新增原型和辅助监督。

每条 Prompt 的分数为患者-文本 cosine 加上该 Prompt 类别原型与对立类别原型的 cosine 差；类内 softmax 得到患者条件 Prompt 权重。两个疾病证据形成有符号 `tanh(e_ASD-e_TD)`，残差幅度 `0.5*tanh(alpha)` 初始化为0.1，类别 margin 最大绝对调整0.5，按 `[+delta/2,-delta/2]` 加到 base log-probability 后重新 log-softmax。损失是原 E74.4 非分类项 + 替换后的分类项 + 0.05加权语义CE + 0.02 `softplus((CE_final-stopgrad(CE_base))/0.25)`。

固定训练：seed=0，原始10折，epoch 10 warmup 后训练至450，`wait>200`停止，测试 ACC 严格提升选择最佳 checkpoint。CLIP RN50 在训练前一次缓存，冻结且保留1024维输出。没有 OOF 重训或参数搜索。

## 正式结果

以下 mean±SD 为百分比，SD 是10折总体标准差。Hard-AUC 按历史硬标签 one-hot 实现，二分类下等于 BA；Probability-AUC 使用 P(ASD=0) 的连续分数。

| 组别 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 history | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E78-G1-SPromptGL-K1 | 91.7294±2.8596 | 91.5750±2.8852 | 93.6206±3.2306 | 89.5793±4.4307 | 93.5708±3.8637 | 90.9259±3.1302 |
| E78-G2-SPromptGL-K3 | 92.0742±2.7959 | 91.8460±2.8229 | 93.4304±3.3151 | 88.8354±3.7320 | 94.8566±3.0874 | 91.2057±3.0606 |
| E78-G3-BrainPrompt-K1 | 91.8443±3.1095 | 91.7031±3.1297 | 93.4222±2.7374 | 89.8354±4.0406 | 93.5708±3.3566 | 91.0705±3.3940 |
| E78-G4-BrainPrompt-K3 | 91.8443±2.4968 | 91.6784±2.4959 | 93.2293±2.7541 | 89.5732±3.6777 | 93.7835±3.6749 | 91.0498±2.7187 |
| E78-G5-Anonymous-K3 | 92.0742±3.0229 | 91.8569±3.0584 | 93.6392±2.9729 | 89.0793±4.7593 | 94.6346±4.0197 | 91.2213±3.3284 |

### 相对历史基座的差值

单位为百分点；每一项均是逐折配对差值的均值。

| 组别 | ΔACC | ΔHard-AUC | ΔProb-AUC | ΔSEN | ΔSPE | ΔASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E78-G1-SPromptGL-K1 | -0.4598 | -0.4544 | +0.1260 | -0.4878 | -0.4209 | -0.5331 |
| E78-G2-SPromptGL-K3 | -0.1149 | -0.1834 | -0.0642 | -1.2317 | +0.8649 | -0.2533 |
| E78-G3-BrainPrompt-K1 | -0.3448 | -0.3263 | -0.0724 | -0.2317 | -0.4209 | -0.3885 |
| E78-G4-BrainPrompt-K3 | -0.3448 | -0.3510 | -0.2653 | -0.4939 | -0.2081 | -0.4092 |
| E78-G5-Anonymous-K3 | -0.1149 | -0.1724 | +0.1446 | -0.9878 | +0.6429 | -0.2377 |

### 组间预设比较

| 比较 | 指标 | 均值差 (pp) | 排除 Fold 2 (pp) | W/T/L |
|---|---|---:|---:|---:|
| G2-G1 | ACC | +0.3448 | +0.1277 | 2/8/0 |
| G2-G1 | Hard-AUC/BA | +0.2710 | +0.0448 | 4/2/4 |
| G2-G1 | Probability-AUC | -0.1902 | -0.2231 | 5/0/5 |
| G4-G3 | ACC | +0.0000 | -0.1277 | 2/5/3 |
| G4-G3 | Hard-AUC/BA | -0.0247 | -0.1630 | 2/4/4 |
| G4-G3 | Probability-AUC | -0.1929 | +0.0272 | 4/0/6 |
| G1-G3 | ACC | -0.1149 | -0.1277 | 1/7/2 |
| G1-G3 | Hard-AUC/BA | -0.1280 | -0.1275 | 1/4/5 |
| G1-G3 | Probability-AUC | +0.1984 | +0.2500 | 7/0/3 |
| G2-G4 | ACC | +0.2299 | +0.1277 | 3/6/1 |
| G2-G4 | Hard-AUC/BA | +0.1676 | +0.0802 | 3/5/2 |
| G2-G4 | Probability-AUC | +0.2011 | -0.0004 | 6/0/4 |
| G2-G5 | ACC | +0.0000 | +0.0000 | 1/8/1 |
| G2-G5 | Hard-AUC/BA | -0.0109 | +0.0026 | 3/4/3 |
| G2-G5 | Probability-AUC | -0.2088 | -0.1495 | 6/0/4 |

完整50行逐折六指标见 `fold_metrics.csv`；每个比较的逐折差值见 `paired_comparisons.csv`。

## 机制与审计

G2 最佳 checkpoint 同态基座头到最终头共纠正 5 例、损害 0 例；跨折平均 |delta|=0.127575。该计数按患者-折记录，不等同独立个体数。
G2 测试 Query 类内 Prompt 熵均值=1.000152；SPromptGL 每类4条 Prompt 的最大熵 ln(4)=1.386294。需结合权重跨患者标准差判断是否接近均匀，数值见 `mechanism.csv`。

十折基座关闭 parity 检查前向、原损失、参数和三个原优化器状态、RNG 均逐折精确相等；正式模块初始化使用私有 `fork_rng`，新增前向无 dropout/抽样，逐 epoch 公共 RNG digest 与 E74.4 对照轨迹核对。端到端活动梯度预检记录DARG、CAM、患者投影、Prompt投影及有界残差参数；新增总损失对DARG/CAM非零。训练和恢复均按原三个优化器顺序后更新新增AdamW。恢复重放两步的模型、优化器、随机状态一致。

独立审计重新加载50/50最佳 checkpoint，重算预测标签、连续概率、ACC、Hard-AUC、Prob-AUC、SEN、SPE、ASD-F1，与记录逐值一致。审计同时核对训练/测试成员、原型簇归属、checkpoint SHA、训练 RNG trace、源文件 SHA 与 best/latest 参数及优化器状态。Fold-2 排除的九折描述性比较也列入 `paired_summary.csv`，不是重新划分或无偏估计。

G2 同一 checkpoint 的关闭原型、关闭语义、类别交换、匿名文本替换、关闭残差干预结果在 `interventions.csv`。这些固定模型干预解释即时依赖，不替代五组独立重训比较。

## GO / NO-GO

**NO-GO**：完整医学多子原型组 G2 未同时超过历史 E74.4-G2 的 ACC、Hard-AUC/BA 与 Probability-AUC，不能将当前 DCPSS 作为已成功的第二创新模块。停止自动追加 E78.1、Prompt/温度/原型数搜索。五组仍回答本次预设的粒度和文本来源对照，但任何一项单指标提升不足以改变结论。

局限：协议一致不等于无偏泛化。外层测试参与 checkpoint 选择，预处理/RFE 先于外层CV，训练采用transductive全图；Fold 2 还保留两名历史重叠患者。单 seed 十折不足以支持临床泛化或发表级结论。

输出目录：`/model/spromptgl/experiment/experiment78_dcpss`；本次因首选 OSSFS 路径真实 EIO 而仅使用 `/model` 回退。最终交接已追加到 `/model/spromptgl/experiment/handoff.md`。
