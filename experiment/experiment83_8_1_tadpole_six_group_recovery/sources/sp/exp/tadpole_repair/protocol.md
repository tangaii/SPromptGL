# TADPOLE 实验设置

| 核对项 | 实验处理 |
| --- | --- |
| 论文 §3.1 和式 (7) | dropout 0，学习率 0.012，4 个注意力头、2 个编码层，α、λ、η 均为 1。论文的隐藏层取值 10 按公开代码映射为每头宽度。编码层内部 attention dropout 也设为 0。 |
| 论文之外的参数 | 公开日志每折有 300 轮。学习率之外的 Adam 设置、weight decay 0.0036、余弦阈值 0.9、seed 0、分折种子 10 取自当前公开实现及复现配置。类权重继续按旧代码的 Counter 插入顺序传给 NLL。 |
| 确定性预处理 | 沿用 MMGL 的访视选择、缺失处理、全队列 RFE 和标准化。模态顺序固定，模态内按列名排序，RFE 单线程执行。之前的 CSV 与新 CSV 在列对齐后最大数值差约为 1.78e−15。 |
| CLIP | 官方 RN50 权重，冻结文本编码器并缓存文本特征。Xavier 对照在每折复现公开代码递归初始化对 CLIP 线性层的影响。冻结直算与冻结缓存逐元素相等。 |
| PEN | 公开的 8 条 prompt 分成两组，保留作对照。论文 §2.2 给出 AD 海马萎缩示例，三类实验每类 4 条，经 RN50 编码后拼接为类原型。式 (5)–(6) 组使用线性文本投影、每模态独立的全局和局部投影、跨类别的最大余弦相似度门控，以及跨 4 个局部 token 的 softmax 加权。 |
| MMGPL prompt | 采用本地 MMGPL 论文图 6 的 12 个短概念，NC、LMCI、AD 各 4 个。当前数据只有 MCI，因此将论文 LMCI 概念组映射为 MCI。`mmgpl_concepts` 保留图中文字，`mmgpl_class_prompts` 按 MMGPL 式 (8) 在概念前加类别名。这里只替换 prompt 文本，仍使用本项目 RN50 和 SPromptGL 的 PEN，不能视为完整 MMGPL 复现。 |
| MMGPL + 有效模块 | 在类别名 prompt 上依次加配对均方图监督、PEN 式 (5)–(6)，以及 sigmoid 加权的六关系 GFR。三组使用相同折拆分和种子，构成逐层累加的消融。 |
| MMGPL prompt | 采用本地 MMGPL 论文图 6 的 12 个短概念，NC、LMCI、AD 各 4 个。当前数据只有 MCI，因此将论文 LMCI 概念组映射为 MCI。`mmgpl_concepts` 保留图中文字，`mmgpl_class_prompts` 按 MMGPL 式 (8) 在概念前加类别名。这里只替换 prompt 文本，仍使用本项目 RN50 和 SPromptGL 的 PEN，不能视为完整 MMGPL 复现。 |
| GFR | 保留现有两层 GCN、阈值邻接和残差。加权和对照分别采用独立 sigmoid 权重、跨 6 关系的 softmax 权重。旧实现中第一关系的消息被第二关系覆盖。 |
| 图监督 | 在训练节点配对上用同类为 1、异类为 0 的目标监督阈值前余弦相似度，分别运行式 (1) 对应的 Frobenius 平方和与按配对数归一化的均方误差。传播和结构正则仍使用阈值后的邻接。 |
| 评估 | 按要求保留全队列传导式构图、全队列预处理和外层测试折选轮次。在 300 轮中选外层测试折 ACC 首次达到最大值的权重，沿用 10 折均值及总体标准差。 |
| AUC | 论文对比用硬标签宏平均 OvR AUC，另报告基于三类预测概率的宏平均 OvR ROC AUC。 |

论文和公开仓库未给出 TADPOLE 原始命令、完整 NC/MCI/AD prompt 或受试者级拆分。三类 prompt 是基于论文示例重建的文本，MMGPL 两组文本和 LMCI→MCI 映射见 `config.py` 与 `mmgpl_prompts.json`。
[SPromptGL 论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [SPromptGL 源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [MMGPL 论文](https://doi.org/10.1016/j.media.2024.103225) · [MMGL 预处理](https://github.com/SsGood/MMGL/blob/c13b05c190f85e614ccb3ace60bb6b152a1f4a1b/data/TADPOLE/Data_preprocessing_TADPOLE%20%282%29.ipynb)

## 审计文件

| 文件 | 内容 |
| --- | --- |
| `data/preprocessing.json` 与 `data/determinism.json` | 原始输入哈希、列和组统计、预处理复跑校验 |
| `data/features.json` 与 `data/samples.csv` | 特征顺序、受试者与访视顺序 |
| `mmgpl_prompts.json` | MMGPL 图 6 的概念原文、类别名版本和 LMCI→MCI 映射 |
| `runs/<variant>/config.json` 与 `splits.json` | 完整参数、运行环境、prompt、源码及权重哈希、行级拆分 |
| `runs/<variant>/fold_XX/` | 300 轮损失和外层 ACC、最佳模型、下一折随机状态、外层折预测、混淆矩阵及关系诊断 |
| `runs/<variant>/out_of_fold_predictions.csv` | 598 行折外预测和三类概率 |
| `diagnostics.json` 与 `verification.json` | CLIP/融合/图监督检查、逐折指标复算、各折文件哈希 |
| `comparison.md` 与 `comparison.csv` | 论文、上次复现和所有修复组的指标 |

同一工作目录可执行 `bash exp/tadpole_repair/run_all.sh`，也可在脚本后指定配置名。每折训练结束后提交文件，重启时核对配置并从首个缺失折恢复。运行 `python exp/tadpole_repair/verify.py` 可复算比较表。

续跑短测发现 `cuda` 与 `cuda:0` 两个缓存键会触发额外的 CLIP 初始化。已统一使用特征张量的设备，修复后连续与续跑的模型权重、随机状态和预测逐项一致。正式训练的入口快照保存在 `references/run_as_executed.py`，核验记录在 `smoke/resume_check.json`。
