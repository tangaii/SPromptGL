# Experiment 78.5 — BSP-AL results

状态：`COMPLETE_AUDITED`；正式目标30折，任务完成30/30，独立最佳Checkpoint审计30/30。
训练源SHA：`7cb7617a024049ca34dc1d3c8451bcc4342215cf3b31b99977af5cfd14fbda23`；正式目录：`/model/spromptgl/experiment/experiment78_5_bspal`；preferred OSSFS errno=5，使用/model回退。

## 六指标（均值±population SD，%）

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.5-G1-BasePreserved-K1-Medical|91.7294 ± 3.0388|91.5569 ± 3.1271|93.7116 ± 3.0825|89.1037 ± 5.5324|94.0102 ± 3.5503|90.8451 ± 3.4593|10|
|E78.5-G2-Selective-K1-Medical|92.0742 ± 2.6503|91.9072 ± 2.6540|93.4286 ± 3.3723|89.5915 ± 3.4414|94.2229 ± 3.1826|91.2818 ± 2.8997|10|
|E78.5-G3-Selective-K1-Anonymous|91.9592 ± 3.0034|91.8118 ± 3.0179|93.6259 ± 2.8957|89.8354 ± 4.0629|93.7882 ± 3.6583|91.1861 ± 3.3017|10|

历史参考：
|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|
|---|---:|---:|---:|---:|---:|---:|---:|
|E74.4-G2|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|10|
|E78.1-G1|92.1891 ± 2.8121|92.0480 ± 2.7367|93.8609 ± 2.6635|90.3171 ± 2.8515|93.7789 ± 4.5987|91.4991 ± 2.8997|10|

## 任务与同Checkpoint分类对照

|组别|TP-ASD|TN-TD|FP-TD|FN-ASD|best epoch均值|耗时均值(min)|峰值显存最大(GB)|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.5-G1-BasePreserved-K1-Medical|359|440|28|44|194.9|20.54|0.96|
|E78.5-G2-Selective-K1-Medical|361|441|27|42|166.6|21.91|1.02|
|E78.5-G3-Selective-K1-Anonymous|362|439|29|41|195.8|21.39|0.97|

同一最佳Checkpoint下，关闭原型残差后回到原MLP路径的 ACC→融合 ACC 平均变化为：G1 +0.1149 pp，G2 +0.2299 pp，G3 +0.2299 pp；对应 Probability-AUC 变化为 +0.0158、+0.0106、+0.0159 pp。G1 相对历史 E78.4-G1 的 ACC/Hard-AUC/Prob-AUC 配对差为 +0.4585/+0.4860/+0.2486 pp。

## 机制与干预

见 `mechanism.csv` 与 `interventions.csv`。G1保持E78.4原聚合，G2/G3使用选择性患者与独立Prompt证据；所有干预均固定最佳Checkpoint。 G1注意力有效支持数约362/420、归一化熵约1.000、最大权重约0.0032；G2约72/97、熵约0.851/0.855、最大权重约0.086/0.111；G3约60/81、熵约0.829/0.825、最大权重约0.120/0.113。G2/G3已明显摆脱均匀聚合，但选择性稀疏本身不等于医学增益。Prompt权重仍接近均匀：G2 ASD约[.244,.260,.244,.252]、TD约[.225,.265,.235,.275]；G3约接近四等分。
- `anonymous`：平均硬翻转 0.000/折；平均概率变化 0.001117。
- `module_off`：平均硬翻转 0.167/折；平均概率变化 0.007376。
- `prompt_class_swap`：平均硬翻转 0.067/折；平均概率变化 0.004638。
- `uniform_attention`：平均硬翻转 0.033/折；平均概率变化 0.000645。

## 不使用LOO的自匹配诊断

训练查询继续参与同类支持库，未使用LOO。对最佳Checkpoint的支持注意力读取显示，G1/G2/G3自身注意力均值分别约0.0026/0.0026/0.0026，最大自身权重约0.0032/0.0864/0.1204；G2/G3选择性注意力提高了单患者峰值，但这没有带来超过E74.4的独立测试增益。完整均值、95百分位和超过1%的比例见 `mechanism.csv`。

## 配对比较

完整逐折差、W/T/L、描述性95% CI和Fold 2排除结果见 `paired_comparisons.csv`。
- E78.5-G1-BasePreserved-K1-Medical - E74.4-G2：ACC差 -0.4598 pp，W/T/L 0/6/4。
- E78.5-G1-BasePreserved-K1-Medical - E78.4-G1：ACC差 +0.4585 pp，Hard-AUC差 +0.4860 pp，Prob-AUC差 +0.2486 pp，ACC W/T/L 5/3/2。
- E78.5-G2-Selective-K1-Medical - E78.5-G1-BasePreserved-K1-Medical：ACC差 +0.3448 pp，W/T/L 3/6/1。
- E78.5-G2-Selective-K1-Medical - E78.5-G3-Selective-K1-Anonymous：ACC差 +0.1149 pp，W/T/L 3/6/1。
- E78.5-G2-Selective-K1-Medical - E74.4-G2：ACC差 -0.1149 pp，W/T/L 2/5/3。
- E78.5-G2-Selective-K1-Medical - E78.1-G1：ACC差 -0.1149 pp，W/T/L 1/7/2。
- E78.5-G3-Selective-K1-Anonymous - E74.4-G2：ACC差 -0.2299 pp，W/T/L 2/5/3。

## GO / NO-GO

- G1保留原始分类监督的结果见G1与历史E78.4对照；G2选择性模块相对G1：`SUPPORTED`。
- G2医学Prompt相对G3匿名Prompt：`MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`。
- G2相对E74.4-G2：`NOT_SUPPORTED`。
- 最终科研判定：`NO-GO`。

## 限制

保留val=test、测试ACC选模、CV前预处理/RFE、transductive图与Fold 2历史重叠；这些结果不是无偏临床泛化估计。
