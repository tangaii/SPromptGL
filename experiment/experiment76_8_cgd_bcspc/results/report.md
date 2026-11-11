# Experiment 76.8 CGD-BCSPC

STATUS: `COMPLETE_AUDITED`。科研判定：`NO_GO_CGD_BCSPC`。

结果沿用E74.4-G2历史协议：10折、测试折参与ACC选模、CV前RFE/标准化及transductive图。该对比是历史协议内的探索性结果，不是严格无偏的独立泛化估计。Fold 2存在历史训练/测试重复行；E76.8新增分支已排除行377/407，但原主干历史协议仍保留该问题。

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2历史基座 | 92.19±2.76 | 92.03±2.73 | 93.49±2.92 | 90.07±3.89 | 93.99±4.73 | 91.46±2.90 |
| E76.7-G1历史参考 | 92.07±2.84 | 91.86±2.86 | 93.76±2.75 | 89.09±3.72 | 94.64±3.24 | 91.23±3.12 |
| E76.7-G2历史参考 | 92.19±2.76 | 92.03±2.73 | 93.67±2.79 | 90.07±3.89 | 93.99±4.73 | 91.46±2.90 |
| E76.8-G1 Conflict-Safe-E2E | 92.07±2.75 | 91.97±2.68 | 93.65±2.90 | 90.57±3.31 | 93.36±4.57 | 91.40±2.86 |
| E76.8-G2 Discriminative-BCSPC | 92.07±2.65 | 91.87±2.64 | 93.33±2.81 | 89.32±3.89 | 94.42±4.14 | 91.26±2.86 |

配对差值单位为百分点；区间为描述性配对t区间，10折不是10个独立外部队列。排除Fold 2结果见 `paired_comparisons.csv`。

| 配对比较 | 指标 | 均值差(pp) | W/T/L | 95%描述区间 | 排除Fold 2差值(pp) |
|---|---|---:|---:|---:|---:|
| G1-Conflict-Safe-E2E − E76.7-G1-CJE-BCSPC-Full-E2E | acc | 0.000 | 1/8/1 | [-0.388, 0.388] | 0.000 |
| G1-Conflict-Safe-E2E − E76.7-G1-CJE-BCSPC-Full-E2E | hard_auc | 0.101 | 3/6/1 | [-0.354, 0.556] | 0.112 |
| G1-Conflict-Safe-E2E − E76.7-G1-CJE-BCSPC-Full-E2E | prob_auc | -0.103 | 4/0/6 | [-0.413, 0.208] | -0.031 |
| G2-Discriminative-BCSPC − G1-Conflict-Safe-E2E | acc | 0.000 | 2/6/2 | [-0.548, 0.548] | 0.128 |
| G2-Discriminative-BCSPC − G1-Conflict-Safe-E2E | hard_auc | -0.093 | 2/6/2 | [-0.637, 0.451] | 0.032 |
| G2-Discriminative-BCSPC − G1-Conflict-Safe-E2E | prob_auc | -0.324 | 5/0/5 | [-1.036, 0.389] | -0.395 |
| G2-Discriminative-BCSPC − E74.4-G2-Positive-DCR | acc | -0.115 | 2/5/3 | [-0.722, 0.492] | 0.000 |
| G2-Discriminative-BCSPC − E74.4-G2-Positive-DCR | hard_auc | -0.157 | 2/5/3 | [-0.857, 0.543] | -0.039 |
| G2-Discriminative-BCSPC − E74.4-G2-Positive-DCR | prob_auc | -0.165 | 5/1/4 | [-1.025, 0.696] | -0.200 |
| G1-Conflict-Safe-E2E − E74.4-G2-Positive-DCR | acc | -0.115 | 0/9/1 | [-0.375, 0.145] | -0.128 |
| G1-Conflict-Safe-E2E − E74.4-G2-Positive-DCR | hard_auc | -0.064 | 1/7/2 | [-0.470, 0.343] | -0.071 |
| G1-Conflict-Safe-E2E − E74.4-G2-Positive-DCR | prob_auc | 0.159 | 5/0/5 | [-0.232, 0.551] | 0.195 |
| G2-Discriminative-BCSPC − E76.7-G1-CJE-BCSPC-Full-E2E | acc | 0.000 | 2/6/2 | [-0.548, 0.548] | 0.128 |
| G2-Discriminative-BCSPC − E76.7-G1-CJE-BCSPC-Full-E2E | hard_auc | 0.008 | 2/5/3 | [-0.616, 0.632] | 0.144 |
| G2-Discriminative-BCSPC − E76.7-G1-CJE-BCSPC-Full-E2E | prob_auc | -0.426 | 4/0/6 | [-1.154, 0.301] | -0.427 |
| G2-Discriminative-BCSPC − E76.7-G2-CJE-BCSPC-StopGrad | acc | -0.115 | 2/5/3 | [-0.722, 0.492] | 0.000 |
| G2-Discriminative-BCSPC − E76.7-G2-CJE-BCSPC-StopGrad | hard_auc | -0.157 | 2/5/3 | [-0.857, 0.543] | -0.039 |
| G2-Discriminative-BCSPC − E76.7-G2-CJE-BCSPC-StopGrad | prob_auc | -0.340 | 4/1/5 | [-1.164, 0.484] | -0.366 |
- E76.7-G1历史复核：相对E74.4-G2的ACC差为-0.115 pp，净纠错-1；最佳epoch新增梯度与基座梯度平均余弦为CAM -0.668、OutputLayer -0.805。这些数字从E76.7可信结果文件读取，用于解释冲突来源，不用于训练或调参。

纠错与机制

- G1-Conflict-Safe-E2E相对历史基座：错误→正确 8，正确→错误 9，净纠错 -1。
- G2-Discriminative-BCSPC相对历史基座：错误→正确 6，正确→错误 7，净纠错 -1。
- E76.8-G1相对E74.4-G2：ACC -0.115 pp；E76.8-G2相对E76.8-G1：ACC 0.000 pp；E76.8-G2相对E74.4-G2：ACC -0.115 pp。
- 独立审计：20/20最佳Checkpoint复算通过；G1/G2共有随机操作逐epoch匹配：True/True。
- 逐epoch主干训练ACC、未校正测试ACC、方法ACC和损失见 `train_curve.csv`；四槽位注意力、可靠/边界证据、独有边界向量、eta、残差及Residual-Off、Boundary-Off、Unique-Evidence-Off、Semantic-Swap、Reliable-only干预详见 `mechanism.csv` 和 `correction_analysis.csv`。纠错表列出被纠正/误伤测试患者的全局行号及G2独有证据。Prompt交换是同Checkpoint干预；本轮没有匿名语义正式训练组，因此不能据此宣称疾病文本有独立预测价值。
- G1/G2均为真实端到端训练；主干新增分类梯度使用 `grad(L_cls_final-L_cls_base_eligible)`，按原optimizer param group先做非对称冲突投影，再乘beta并限幅。G2使用未再次归一化的可靠原型正交补边界证据，并用其门控纠错残差。

