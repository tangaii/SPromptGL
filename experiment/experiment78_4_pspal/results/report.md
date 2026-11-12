# Experiment 78.4 — PSP-AL results

状态：`COMPLETE_AUDITED`；正式目标60折，任务完成60/60，独立最佳Checkpoint审计60/60。
训练源SHA：`75b6985eb54a73854059b5581481d642aaba68b1eae922da05217b1c30493241`。本实验使用E74.4-G2 Positive-DCR/DARG + CAM144；preferred OSSFS目录errno=5，正式输出统一在`/model/spromptgl/experiment/experiment78_4_pspal`。

## 论文与实现审计

PPAL论文第2.3节及公式8–12、官方ppal.py/attention_layer.py/engine3.py/textprompt.csv已读。公开engine3实际只优化一份CrossEntropy，论文的粗/细联合损失与代码不完全一致；ppal.py的512维MultiheadAttention布局没有显式batch_first，PSP-AL改为显式[batch,query,dim]和[batch,patient,dim]的32维4-head注意力。SPromptGL和服务器E74.4真实run.py/soft_positive_dcr.py也已核验，原始总损失准确替换了`out[cls]`分类贡献。详见`design.md`。

## 六指标（均值±population SD，%）

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|成功任务|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.4-G1-SPromptGL-K1-Restricted|91.2709 ± 2.7369|91.0709 ± 2.7343|93.4630 ± 2.8254|88.3537 ± 3.4563|93.7882 ± 3.1244|90.3663 ± 2.9597|10|
|E78.4-G2-SPromptGL-K1-Direct|89.8929 ± 3.5980|89.8463 ± 3.5204|92.6626 ± 2.9934|89.1098 ± 3.9254|90.5828 ± 5.1949|89.1367 ± 3.7152|10|
|E78.4-G3-SPromptGL-K2-Restricted|91.3845 ± 2.5912|91.1684 ± 2.5678|93.4184 ± 3.1423|88.5671 ± 4.8014|93.7697 ± 5.1753|90.5005 ± 2.7509|10|
|E78.4-G4-SPromptGL-K2-Direct|90.8098 ± 3.8185|90.7050 ± 3.7716|93.0197 ± 3.5364|89.1159 ± 4.5999|92.2942 ± 4.9036|90.0100 ± 4.0748|10|
|E78.4-G5-Anonymous-K1-Restricted|91.6144 ± 3.0062|91.4454 ± 2.9976|93.5682 ± 2.8947|89.3293 ± 4.6145|93.5615 ± 4.7382|90.8067 ± 3.1982|10|
|E78.4-G6-Anonymous-K1-Direct|89.8916 ± 3.6716|89.8246 ± 3.6443|92.4629 ± 3.6793|88.8537 ± 5.4496|90.7956 ± 5.1841|89.0567 ± 4.0602|10|

历史参考：
|参考|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|折数|
|---|---:|---:|---:|---:|---:|---:|---:|
|E74.4-G2|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|10|
|E78.1-G1|92.1891 ± 2.8121|92.0480 ± 2.7367|93.8609 ± 2.6635|90.3171 ± 2.8515|93.7789 ± 4.5987|91.4991 ± 2.8997|10|

## 任务与混淆统计

|组别|TP-ASD|TN-TD|FP-TD|FN-ASD|best epoch均值|训练耗时均值(min)|峰值显存最大(GB)|
|---|---:|---:|---:|---:|---:|---:|---:|
|E78.4-G1-SPromptGL-K1-Restricted|356|439|29|47|171.4|22.26|0.94|
|E78.4-G2-SPromptGL-K1-Direct|359|424|44|44|219.3|23.33|1.02|
|E78.4-G3-SPromptGL-K2-Restricted|357|439|29|46|220.3|31.59|1.00|
|E78.4-G4-SPromptGL-K2-Direct|359|432|36|44|246.5|32.85|1.03|
|E78.4-G5-Anonymous-K1-Restricted|360|438|30|43|217.4|26.04|1.02|
|E78.4-G6-Anonymous-K1-Direct|358|425|43|45|205.3|19.97|1.04|

## 机制与干预

`mechanism.csv`记录原型相似度/坍塌、患者注意力有效支持数、最大注意力、熵、原型路由、训练自匹配偏差及K=1/K=2差异；`interventions.csv`记录DARG-only、均匀患者聚合、匿名Prompt、Prompt类别交换及K=2关闭第二原型。受限残差 G3 关闭模块平均硬翻转0.4/折、替换匿名Prompt约0.3/折、关闭第二原型约0.1/折；直接分类组对Prompt干预更敏感，但其独立测试性能更低。
- E78.4-G1-SPromptGL-K1-Restricted：训练原型分类 95.79%；原型坍塌比例 0.00%；注意力有效支持均值 391.15。
- E78.4-G2-SPromptGL-K1-Direct：训练原型分类 95.94%；原型坍塌比例 0.00%；注意力有效支持均值 391.48。
- E78.4-G3-SPromptGL-K2-Restricted：训练原型分类 95.00%；原型坍塌比例 100.00%；注意力有效支持均值 382.66。
- E78.4-G4-SPromptGL-K2-Direct：训练原型分类 97.03%；原型坍塌比例 100.00%；注意力有效支持均值 391.49。
- E78.4-G5-Anonymous-K1-Restricted：训练原型分类 95.80%；原型坍塌比例 0.00%；注意力有效支持均值 388.75。
- E78.4-G6-Anonymous-K1-Direct：训练原型分类 96.27%；原型坍塌比例 0.00%；注意力有效支持均值 391.53。
- `anonymous`：平均硬翻转 209；平均概率绝对变化 0.049516。
- `module_off`：平均硬翻转 1331；平均概率绝对变化 0.209296。
- `prompt_class_swap`：平均硬翻转 1071；平均概率绝对变化 0.180052。
- `prototype2_off`：平均硬翻转 18；平均概率绝对变化 0.007456。
- `uniform_attention`：平均硬翻转 153；平均概率绝对变化 0.026465。

## 不使用LOO的自匹配偏差诊断

训练查询继续使用支持库本身，未采用LOO；独立诊断对每个支持患者读取其自身CAM行在同类原型注意力中的权重，并与测试折分类差距同时记录在 `mechanism.csv`。六组平均自身注意力约为 0.002566，平均最大自身权重约为 0.002757，超过1%的患者比例为0；训练最终准确率相对测试准确率的平均差距为 G1 4.76、G2 6.05、G3 4.51、G4 6.22、G5 4.42、G6 6.38 个百分点。该结果没有显示原型注意力集中记忆单个训练患者，但训练/测试差距仍受历史选模与协议限制。

## 配对比较

完整六指标逐折差、W/T/L、描述性95% CI及Fold 2排除结果见`paired_comparisons.csv`。
- E78.4-G1-SPromptGL-K1-Restricted - E74.4-G2：ACC差 -0.9182 pp，W/T/L 0/4/6。
- E78.4-G2-SPromptGL-K1-Direct - E74.4-G2：ACC差 -2.2962 pp，W/T/L 1/2/7。
- E78.4-G3-SPromptGL-K2-Restricted - E74.4-G2：ACC差 -0.8046 pp，W/T/L 0/4/6。
- E78.4-G4-SPromptGL-K2-Direct - E74.4-G2：ACC差 -1.3793 pp，W/T/L 1/2/7。
- E78.4-G1-SPromptGL-K1-Restricted - E78.4-G2-SPromptGL-K1-Direct：ACC差 +1.3780 pp，W/T/L 7/1/2。
- E78.4-G3-SPromptGL-K2-Restricted - E78.4-G4-SPromptGL-K2-Direct：ACC差 +0.5747 pp，W/T/L 4/4/2。
- E78.4-G3-SPromptGL-K2-Restricted - E78.4-G1-SPromptGL-K1-Restricted：ACC差 +0.1136 pp，W/T/L 3/5/2。
- E78.4-G4-SPromptGL-K2-Direct - E78.4-G2-SPromptGL-K1-Direct：ACC差 +0.9169 pp，W/T/L 5/3/2。
- E78.4-G1-SPromptGL-K1-Restricted - E78.4-G5-Anonymous-K1-Restricted：ACC差 -0.3435 pp，W/T/L 2/3/5。
- E78.4-G2-SPromptGL-K1-Direct - E78.4-G6-Anonymous-K1-Direct：ACC差 +0.0013 pp，W/T/L 4/1/5。
- E78.4-G1-SPromptGL-K1-Restricted - E78.1-G1：ACC差 -0.9182 pp，W/T/L 1/2/7。
- E78.4-G3-SPromptGL-K2-Restricted - E78.1-G1：ACC差 -0.8046 pp，Hard-AUC差 -0.8796 pp，Prob-AUC差 -0.4425 pp，ACC W/T/L 0/4/6；排除Fold 2后ACC差 -0.6386 pp。

## GO / NO-GO

最高医学组（按十折ACC、同分时Prob-AUC）：`E78.4-G3-SPromptGL-K2-Restricted`。
- 基座分类增益：`NOT_SUPPORTED`。
- 医学Prompt相对匿名：`MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`；G1−G5 的 ACC 为 −0.3435 pp，G2−G6 仅 +0.0013 pp，固定Checkpoint 的匿名替换也未显示稳定的有利硬翻转。
- K=2互异原型机制：`NOT_SUPPORTED`。
- 最终科研判定：`NO-GO`。

## 协议限制

保留历史val=test、测试ACC选模、CV前预处理/RFE、transductive全患者图以及Fold 2重叠；这些限制不构成无偏临床泛化评估。

## 文件

`preflight.json`、`unit_tests.json`、`fold_metrics.csv`、`paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`、`checkpoint_audit.csv`、`audit_status.json`。
