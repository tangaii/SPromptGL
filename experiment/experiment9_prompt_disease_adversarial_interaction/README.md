# 实验九：PDAI Prompt-Disease-Adversarial Alignment Interaction

本实验检验基础Prompt、ASD/TD疾病类别语义与模态对抗分布对齐是否存在可重复的交互收益。根目录原始源码、实验一至八目录均保持不变；`alignment_patch.diff` 只在每个临时训练副本中应用。

## 前八轮证据

| 轮次 | 真实结果与结论 |
|---|---|
| 1 MASPE | P1 91.80%/91.62%，P0 91.69%/91.54%，P3 91.54%/91.31%；模态Prompt只带来很小差异。 |
| 2 CSMP | 正确映射V2略高于P0且高于错配V2S，但没有超过P1。 |
| 3 PEPCA | P1 91.80%/91.62%；固定缩放与无关文本较差，固定缩放配置语义投影梯度无效。 |
| 4 DAFCP | ASD-FC和匹配通用FC分别91.46%/91.29%、91.50%/91.28%，均低于P1。 |
| 5 SPCR | 去末层ReLU、去MLP衰减及同时修改都未超过E0；观测到MLP参数收缩，但单项修复无分类收益。 |
| 6 SDPG | B1约91.73%/91.59%，B2约91.77%/91.52%，没有稳定超过P1。 |
| 7 DTSF | F1约91.69%/91.49%；错配、pooled、uniform对照也没有稳定显示双主题融合超越P1。 |
| 8 SDPA | S1约91.73%/91.51%，疾病分支/P1预测相关约0.958；疾病类别方向可学到，但总体增益小且存在信息重复迹象。 |

这些实验不支持继续靠改写Prompt、MLP小修或标量门控追分。实验九固定历史文本，直接检验疾病类别监督与跨模态分布对齐是否互补。

## DeepASD依据

DeepASD Fig.1的顺序是模态对抗编码、逐模态患者图学习与融合、GNN分类；原方法使用模态专属特征提取器与模态判别器的竞争训练，论文描述了平方损失和invert-label策略。本实验只借鉴“先在跨模态注意力前约束模态分布，再走SPromptGL原图/分类路径”的位置，不复现DeepASD结构或损失。Table 2中ABIDE A的特征提取器单独为72.78% ACC/77.59% AUC，加入模态判别器后为76.74%/82.99%，完整三模块系统为87.38%/92.76%；原论文正文对82.99到77.59的下降写作6.51%，但表内数值相减是5.40个百分点，因此这里按表值陈述。其队列和协议与本地ABIDE复现不同，不作数值横向比较。[DeepASD原文](https://www.nature.com/articles/s41398-024-02972-2)。

## 配置

P0是原始8句共享文本；P1是实验一冻结的32句模态专属文本；P3是实验一冻结的32句通用文本。D1精确复用实验八S1的ASD/TD四句类别文本、36→1024无偏置投影、归一化点积除以0.2、`gamma=0.2*tanh(alpha)`和训练集加权交叉熵（权重0.1）。A1在原始 `VariLengthInputLayer` 的四个模态V输出 `[N,4,36]` 上接 `LayerNorm(36)->Linear(36,32)->ReLU->Linear(32,4)`，目标依次为PHENO/ANAT/FUNC/Correlation；仅输入特征经 GRL 反向，scale=0.03，判别器交叉熵直接加入总损失。D0/A0分别关闭对应分支。

| 配置 | Prompt | Disease | Adversarial |
|---|---|---|---|
| E0 | P0 | D0 | A0 |
| E1 | P0 | D1 | A0 |
| E2 | P0 | D0 | A1 |
| E3 | P0 | D1 | A1 |
| E4 | P1 | D0 | A0 |
| E5 | P1 | D1 | A0（实验八S1历史折） |
| E6 | P1 | D0 | A1 |
| E7 | P1 | D1 | A1 |
| E8 | P3 | D0 | A0 |
| E9 | P3 | D1 | A0 |
| E10 | P3 | D0 | A1 |
| E11 | P3 | D1 | A1 |

E0/E4/E8冻结实验一30折，E5冻结实验八S1 30折；只训练其余8组共240折。所有配置都沿用 weighted-cosine、concat、GCN、dropout=.35、lr=.0038、reg=.11、450 epoch、early=200、10折 StratifiedKFold(random_state=10)、seed 0/1/2和原始ACC最佳Checkpoint。原始模型优化器保持 weight_decay=.11；疾病和判别器参数均为独立 weight_decay=0 的参数组/优化器。A1判别器通过 `torch.random.fork_rng` 隔离初始化，避免改变原始主干随机轨迹。

P1/P3仍用原始 `4096→128→9` 投影、每模态两个文本原型、Softmax-Max语义标量和原始global/local融合。P0保留根目录原始8句和两原型路径。A1只读取已有w_v的输出 `[871,4,36]`；对抗表示是GFR之后、PLN跨模态注意力之前的特征，不是CSV原始输入。总损失是原分类/图损失 + 可选D1加权CE + 可选四分类模态CE；GRL仅将模态特征梯度乘以−0.03，判别器仍接收正常CE梯度。训练不加入额外损失或层。

## 诊断与判定

`alignment_diagnostics.csv` 每个新配置×seed×fold一行，含判别器 train/test ACC、CE、熵和类别分布，GRL与V梯度有效性，Correlation对其他模态的RBF-MMD、均值距离、方差、有效秩、零率、同/跨模态余弦、配对cosine gap和R@1，以及D1疾病分支的投影梯度、疾病分类、gamma、margin和决策贡献。四个局部Token是潜在模态Token，不解释为ROI。历史诊断不可追溯，因而不伪造为零。

主分析对 ACC、Hard-AUC、BA，并在训练配置可得时报告 probability-AUC，均按相同seed×fold配对：Prompt×Disease、Prompt×Adversarial、Disease×Adversarial 和三阶交互。报告会列出逐seed差值、胜/平/负折、判别器是否仍能识别模态、MMD是否下降、表示是否塌缩，以及E7相对E5是否有额外收益。微小差异不视为独立样本的显著证据。

## 重跑

```bash
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --prepare
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --all
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --resume
python experiment/experiment9_prompt_disease_adversarial_interaction/run.py --refresh
```

训练工作目录在每个任务结束后自动清理；失败attempt不会进入汇总。`manifest.json` 在所有8组任务、240折和240条诊断完整且通过有限值/键校验后标记 `COMPLETE`。
