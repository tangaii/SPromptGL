# 实验十：DSSC Disease Semantic Specificity Control

本实验复用实验九E9的P3、疾病投影、对齐损失、gamma决策融合和ABIDE训练协议，只改变疾病类别原型文本。R0/R1冻结实验九E8/E9；C1使用ADHD/TD；C2使用NEUTRAL_CLASS0/NEUTRAL_CLASS1。真实标签始终为0=ASD、1=TD。

## 研究问题

实验九E9相对E8的改善可能来自ASD语义、额外类别监督或固定文本向量的几何关系。本实验只新增C1和C2，分别检验错误疾病文本和非疾病技术文本是否也能形成类别判别方向。

## 配置与协议

| 配置 | 基础Prompt | 类别0原型 | 类别1原型 | 状态 |
|---|---|---|---|---|
| R0 | P3 | 无 | 无 | 实验九E8历史冻结 |
| R1 | P3 | ASD | TD | 实验九E9历史冻结 |
| C1 | P3 | ADHD | TD | 新训练 |
| C2 | P3 | NEUTRAL_CLASS0 | NEUTRAL_CLASS1 | 新训练 |

C1/C2均使用36→1024无偏置疾病投影、归一化点积、temperature=0.2、lambda_align=0.1、gamma=0.2*tanh(alpha)，疾病参数weight_decay=0，其余参数weight_decay=0.11。PDAI对抗分支完全关闭，没有GRL、MMD、OT、额外分类头或新超参数。训练为3 seeds×10 folds，450 epoch上限、early=200、ACC checkpoint、hard-label AUC。

## 冻结文本

P3的32条Prompt来自实验一；ASD、TD、ADHD和两组中性技术文本逐字来自实验八。完整冻结内容见 `prompts.json`。训练时使用临时映射文件，仅将所选文本放到疾病分支的类别0/1位置，标签不变。

## 诊断与反事实

`disease_diagnostics.csv` 每个C1/C2 seed×fold一行，记录类别原型余弦/距离/范数、疾病分支性能、真实ASD/TD样本上的类别0 margin、gamma、融合贡献、与P3概率相关性及梯度有效性。`counterfactual.csv` 对每个最佳checkpoint记录 TRAINED、REAL、ADHD、NEUTRAL、SWAPPED、OFF 六种推理条件；不重新训练、不改标签、不重新选择checkpoint。

## 运行

```bash
python experiment/experiment10_disease_semantic_specificity/run.py --prepare
python experiment/experiment10_disease_semantic_specificity/run.py --all
python experiment/experiment10_disease_semantic_specificity/run.py --resume
python experiment/experiment10_disease_semantic_specificity/run.py --refresh
```

根目录源码只在临时副本中打补丁；工作目录和失败attempt在任务结束后清理。`manifest.json` 保存版本哈希、参数量、初始化预检、GPU分配和完成状态。
