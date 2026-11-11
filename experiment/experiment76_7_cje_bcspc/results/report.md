# Experiment 76.7 CJE-BCSPC

STATUS: `COMPLETE_AUDITED`。科研判定：`NO_GO_CJE_BCSPC`。

结果沿用E74.4-G2历史协议：10折、测试折参与ACC选模、CV前RFE/标准化及transductive图。该对比是历史协议内的探索性结果，不是严格无偏的独立泛化估计。Fold 2存在历史训练/测试重复行；新增分支已排除行377/407，但原主干历史协议仍保留该问题。

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2历史基座 | 92.19±2.76 | 92.03±2.73 | 93.49±2.92 | 90.07±3.89 | 93.99±4.73 | 91.46±2.90 |
| E76.6-G1历史参考 | 92.19±2.76 | 92.03±2.73 | 93.48±2.91 | 90.07±3.89 | 93.99±4.73 | 91.46±2.90 |
| E76.7-G1 Full-E2E | 92.07±2.84 | 91.86±2.86 | 93.76±2.75 | 89.09±3.72 | 94.64±3.24 | 91.23±3.12 |
| E76.7-G2 StopGrad | 92.19±2.76 | 92.03±2.73 | 93.67±2.79 | 90.07±3.89 | 93.99±4.73 | 91.46±2.90 |

配对差值单位为百分点；区间为描述性配对t区间，10折不是10个独立外部队列。排除Fold 2结果见 `paired_comparisons.csv`。

| 配对比较 | 指标 | 均值差(pp) | W/T/L | 95%描述区间 | 排除Fold 2差值(pp) |
|---|---|---:|---:|---:|---:|
| G1-CJE-BCSPC-Full-E2E − E74.4-G2-Positive-DCR | acc | -0.115 | 0/9/1 | [-0.375, 0.145] | -0.128 |
| G1-CJE-BCSPC-Full-E2E − E74.4-G2-Positive-DCR | hard_auc | -0.165 | 1/6/3 | [-0.459, 0.129] | -0.183 |
| G1-CJE-BCSPC-Full-E2E − E74.4-G2-Positive-DCR | prob_auc | 0.262 | 5/2/3 | [-0.204, 0.728] | 0.226 |
| G2-CJE-BCSPC-StopGrad − E74.4-G2-Positive-DCR | acc | 0.000 | 0/10/0 | [0.000, 0.000] | 0.000 |
| G2-CJE-BCSPC-StopGrad − E74.4-G2-Positive-DCR | hard_auc | 0.000 | 0/10/0 | [0.000, 0.000] | 0.000 |
| G2-CJE-BCSPC-StopGrad − E74.4-G2-Positive-DCR | prob_auc | 0.175 | 6/3/1 | [-0.009, 0.360] | 0.165 |
| G1-CJE-BCSPC-Full-E2E − G2-CJE-BCSPC-StopGrad | acc | -0.115 | 0/9/1 | [-0.375, 0.145] | -0.128 |
| G1-CJE-BCSPC-Full-E2E − G2-CJE-BCSPC-StopGrad | hard_auc | -0.165 | 1/6/3 | [-0.459, 0.129] | -0.183 |
| G1-CJE-BCSPC-Full-E2E − G2-CJE-BCSPC-StopGrad | prob_auc | 0.086 | 4/0/6 | [-0.304, 0.477] | 0.061 |
| G1-CJE-BCSPC-Full-E2E − E76.6-G1-BCSPC-Full | acc | -0.115 | 0/9/1 | [-0.375, 0.145] | -0.128 |
| G1-CJE-BCSPC-Full-E2E − E76.6-G1-BCSPC-Full | hard_auc | -0.165 | 1/6/3 | [-0.459, 0.129] | -0.183 |
| G1-CJE-BCSPC-Full-E2E − E76.6-G1-BCSPC-Full | prob_auc | 0.272 | 5/2/3 | [-0.190, 0.735] | 0.238 |

纠错与机制

- G1-CJE-BCSPC-Full-E2E相对历史基座：错误→正确 6，正确→错误 7，净纠错 -1。
- G2-CJE-BCSPC-StopGrad相对历史基座：错误→正确 0，正确→错误 0，净纠错 0。
- 独立审计：20/20最佳Checkpoint复算通过；G2与历史基座RNG逐epoch匹配：True；G1共有随机操作逐epoch匹配：True。
- 四槽位注意力、可靠/边界证据、残差及Residual-Off、Boundary-Off、Semantic-Swap、Reliable-only干预详见 `mechanism.csv` 和 `correction_analysis.csv`。Prompt交换是同Checkpoint干预；本轮没有匿名语义正式训练组，因此不能据此宣称疾病文本有独立预测价值。
- G1为主干保护式预热后受控端到端联合优化；G2为相同结构、始终StopGrad。新增损失的主干梯度使用最终分类损失减原分类损失的梯度差，并按原优化器组限幅。

