# Experiment 78.7 — BDP-PL results

状态：`COMPLETE_AUDITED`；正式目标40折，训练完成40/40，独立最佳Checkpoint审计40/40。
正式目录：`/model/spromptgl/experiment/experiment78_7_bdpprompt`；preferred OSSFS errno=5，使用/model回退；训练源SHA：`87af7eafd9da6d5842648dfe7b9f5ee6e98bd6e4d727ef3d25baae0f2f66e66d`。

## 六指标（均值±population SD，%）

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.7-G1-InverseCalibration-Medical|91.8443 ± 3.0233|91.8087 ± 2.9799|93.8382 ± 2.9265|91.3232 ± 3.7191|92.2942 ± 4.5324|91.2307 ± 3.1663|10|
|E78.7-G2-PPALSoftAggregation-Medical|91.8443 ± 3.1095|91.6605 ± 3.0752|93.7535 ± 3.1029|89.3293 ± 4.4636|93.9917 ± 4.8953|91.0467 ± 3.3117|10|
|E78.7-G3-Bidirectional-Medical|91.3845 ± 3.2270|91.2054 ± 3.2175|93.4623 ± 2.9512|88.8354 ± 3.8959|93.5754 ± 4.0816|90.5303 ± 3.4962|10|
|E78.7-G4-Bidirectional-Anonymous|91.6144 ± 3.0928|91.4849 ± 3.1264|93.6515 ± 3.2201|89.8293 ± 5.5086|93.1406 ± 5.0021|90.8292 ± 3.4054|10|

历史参考：
|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|
|---|---:|---:|---:|---:|---:|---:|---:|
|E74.4-G2|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|10|
|E78.1-G1|92.1891 ± 2.8121|92.0480 ± 2.7367|93.8609 ± 2.6635|90.3171 ± 2.8515|93.7789 ± 4.5987|91.4991 ± 2.8997|10|
|E78.5-G2|92.0742 ± 2.6503|91.9072 ± 2.6540|93.4286 ± 3.3723|89.5915 ± 3.4414|94.2229 ± 3.1826|91.2818 ± 2.8997|10|

## 机制与干预

固定最佳Checkpoint的 A/B、uniform_attention、anonymous_text 和 prompt_class_swap 结果见 `mechanism.csv` 与 `interventions.csv`。最终分类始终是DARG，因此语义干预可改变辅助语义概率而不必改变DARG硬预测。
- `a_off`：平均语义概率变化 0.013362；平均DARG硬翻转 0.000/折。
- `anonymous_text`：平均语义概率变化 0.403514；平均DARG硬翻转 0.000/折。
- `b_off`：平均语义概率变化 0.021968；平均DARG硬翻转 0.000/折。
- `both_off`：平均语义概率变化 0.031615；平均DARG硬翻转 0.000/折。
- `full`：平均语义概率变化 0.000000；平均DARG硬翻转 0.000/折。
- `prompt_class_swap`：平均语义概率变化 0.834343；平均DARG硬翻转 0.000/折。
- `uniform_attention`：平均语义概率变化 0.022673；平均DARG硬翻转 0.000/折。

## 配对比较

逐折差、W/T/L、描述性95% CI及排除Fold 2结果见 `paired_comparisons.csv`。
- E78.7-G1-InverseCalibration-Medical - E74.4-G2：ACC差 -0.3448 pp，W/T/L 0/7/3。
- E78.7-G2-PPALSoftAggregation-Medical - E74.4-G2：ACC差 -0.3448 pp，W/T/L 1/5/4。
- E78.7-G3-Bidirectional-Medical - E74.4-G2：ACC差 -0.8046 pp，W/T/L 0/4/6。
- E78.7-G4-Bidirectional-Anonymous - E74.4-G2：ACC差 -0.5747 pp，W/T/L 1/5/4。
- E78.7-G3-Bidirectional-Medical - E78.7-G1-InverseCalibration-Medical：ACC差 -0.4598 pp，W/T/L 1/4/5。
- E78.7-G3-Bidirectional-Medical - E78.7-G2-PPALSoftAggregation-Medical：ACC差 -0.4598 pp，W/T/L 2/4/4。
- E78.7-G3-Bidirectional-Medical - E78.7-G4-Bidirectional-Anonymous：ACC差 -0.2299 pp，W/T/L 1/6/3。
- E78.7-G3-Bidirectional-Medical - E78.5-G2：ACC差 -0.6897 pp，W/T/L 2/2/6。
- E78.7-G3-Bidirectional-Medical - E78.1-G1：ACC差 -0.8046 pp，W/T/L 0/4/6。

## GO / NO-GO

- G3相对E74.4-G2：`NOT_SUPPORTED`。
- G3相对G1/G2双向互补：`NOT_SUPPORTED`。
- G3医学相对G4匿名：`MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`。
- 最终科研判定：`NO-GO_BIDIRECTIONAL_PROMPT`。

## 限制

仍保留val=test、测试ACC选模、CV前RFE/预处理、transductive图和Fold 2历史重叠；本实验是方法探索性证据，不是无偏临床泛化评估。

## 机制补充与 Fold 2 敏感性（report-only）

最终 DARG 输出保持为原始 `out["logp"]`；固定 Checkpoint 的 A/B/文本/均匀权重干预因此没有改变 DARG 硬标签，但改变了辅助语义分支。独立训练后各组最佳状态的 DARG 训练集准确率均约 96.48–96.99%，没有出现 E78.4 直接替换分类器时的原 MLP 退化。

|组别|DARG训练ACC(%)|语义训练ACC(%)|语义测试ACC(%)|有效支持 ASD|有效支持 TD|注意力熵 ASD|注意力熵 TD|Prompt校准位移|Prompt最终位移|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E78.7-G1-InverseCalibration-Medical|96.990|96.887|90.007|783.90|783.90|1.000|1.000|0.00471|0.00471|
|E78.7-G2-PPALSoftAggregation-Medical|96.480|96.339|89.663|325.35|262.75|0.817|0.818|-0.00000|0.14264|
|E78.7-G3-Bidirectional-Medical|96.684|96.836|89.433|531.78|419.87|0.837|0.818|0.00466|0.13892|
|E78.7-G4-Bidirectional-Anonymous|96.773|97.142|89.781|413.72|365.75|0.822|0.828|0.00469|0.15793|

G1 的患者权重为均匀诊断对照（有效支持约 784、熵约 1）；G2/G3/G4 的方向 B 产生了明显非均匀权重，但 G3 的分类结果低于 G1/G2，说明注意力稀疏本身没有形成诊断增益。A/B 关闭分别使语义概率平均变化约 0.0134/0.0220，uniform_attention 约 0.0227；医学文本替换为匿名文本平均改变语义 ASD 概率约 0.4035，但 DARG 硬预测翻转为 0。类别交换的语义概率变化约 0.8343，验证有符号语义分支对类别顺序敏感；这不是最终 DARG 诊断收益。

|组别|ACC（去Fold2）|Hard-AUC（去Fold2）|Prob-AUC（去Fold2）|SEN（去Fold2）|SPE（去Fold2）|ASD-F1（去Fold2）|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.7-G1-InverseCalibration-Medical|91.8321|91.8028|94.0608|91.4431|92.1626|91.2166|
|E78.7-G2-PPALSoftAggregation-Medical|91.9598|91.7442|93.8017|88.9566|94.5318|91.1101|
|E78.7-G3-Bidirectional-Medical|91.3213|91.1325|93.4782|88.6789|93.5862|90.4383|
|E78.7-G4-Bidirectional-Anonymous|91.8321|91.7141|93.8004|90.3252|93.1031|91.0760|

以上补充只更新报告和机制统计，不改变已冻结训练源码、Checkpoint 或 `source_sha256`。
