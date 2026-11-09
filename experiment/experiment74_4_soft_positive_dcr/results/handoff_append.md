<!-- EXPERIMENT_74_4_SOFT_POSITIVE_DCR -->
## Experiment 74.4：CORR评分软化与正向局部疾病残差（2026-11-09）

目的：在E74.3-G1基础上固定0.02直接传播尺度，检验CORR构图Logit温度T=2软化和直接残差positive-only两个正交因素。目录`/model/spromptgl/experiment/experiment74_4_soft_positive_dcr`；preferred OSSFS目录真实报OSError(5)，已记录并回退至`/model`。只读E74.3基线锁定`upload/experiment74-3-local-dcr`@`e05275cb4b18dcd01679682b9df02f3cbc13b713`。
组别：E74.4-G1=T2/signed；G2=T1/positive；G3=T2/positive；E74.3-G1=T1/signed只读。不使用Prompt、文本模块、新损失或新参数。T仅改CORR构图概率，BCE仍用原始Logit；positive-only仅作用于直接DCR，统一GCN图仍含负疾病增量。

执行：30/30训练、30/30独立checkpoint审计；PPU-ZW810E，worker峰值9，耗时660.5s，重试0，OOM 0。

10折mean±population SD（%）：
|Group|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E69.3-G2|92.0742±2.7482|91.8995±2.7186|93.3759±2.9908|89.5854±3.4710|94.2137±3.9815|91.2984±2.9222|
|E74.3-G1-Local-DCR|92.0742±2.9789|91.8406±3.0283|94.0087±2.9776|88.8293±4.7826|94.8520±3.8905|91.1921±3.3226|
|E74.4-G1-Soft-DCR|91.3858±2.6904|91.1021±2.7517|92.9784±2.8298|87.3476±4.5089|94.8566±3.0874|90.3484±3.0210|
|E74.4-G2-Positive-DCR|92.1891±2.7648|92.0294±2.7255|93.4946±2.9195|90.0671±3.8862|93.9917±4.7320|91.4590±2.9013|
|E74.4-G3-Soft-Positive-DCR|91.8443±2.6508|91.6225±2.6751|93.6961±3.0908|88.8232±4.6645|94.4218±4.3507|90.9686±2.8953|

因素效应、九折比较、历史对照、CORR q_raw/q_graph及四种固定Checkpoint干预见实验`results/paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`和`report.md`。Fold 2历史重叠row 377/407已在独立审计和干预翻转中核对。

审计：E74.3-G1的T1 signed前向与图逐张量一致；三组均恢复相同E61 epoch-10 warmup/优化器/数据索引，组间共有epoch RNG一致；30/30独立读回，根模型和历史文件保持只读。
结论：**GO**。历史val=test、CV前预处理和transductive图仍限制泛化解释；即使达到目标也需在无重叠、fold-local预处理和独立验证选模下复核。manifest `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr/manifest.json`，报告 `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr/results/report.md`。
<!-- EXPERIMENT_74_4_SOFT_POSITIVE_DCR -->
