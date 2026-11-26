<!-- EXPERIMENT74_3_LOCAL_DCR_AUDIT_CORRECTION_2026-11-06 -->
### Experiment 74.3审计更正

最终文件核验发现首版handoff中G2−G1及G3−G2配对表的左右操作数映射反向。模型训练、checkpoint、逐折预测和独立审计不受影响；已修正`results/paired_comparisons.csv`及`results/report.md`。以下为正确方向，覆盖首版中这两组比较行：
|比较|范围|ACC Δ pp|Hard-AUC Δ pp|Prob-AUC Δ pp|SEN Δ pp|胜/平/负（ACC）|
|---|---|---:|---:|---:|---:|---:|
|G2 vs G1|all10|-0.2286|-0.1995|-0.8181|+0.2439|1/5/4|
|G2 vs G1|nine_without_fold2|-0.1263|-0.1156|-0.8677|+0.0000|1/5/3|
|G3 vs G2|all10|+0.1123|+0.1543|+0.4035|+0.5122|3/4/3|
|G3 vs G2|nine_without_fold2|-0.0029|+0.0654|+0.4954|+0.8401|2/4/3|

排除Fold 2后的各组结果（9折mean ± population SD，%）：
|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E74.3-G1-Local-DCR|92.0875 ± 3.1397|91.8384 ± 3.1921|94.0677 ± 3.1331|88.6721 ± 5.0167|95.0046 ± 4.0724|
|E74.3-G2-Local-DCR-3x|91.9613 ± 2.8215|91.7227 ± 2.9136|93.2000 ± 4.0428|88.6721 ± 4.8626|94.7734 ± 2.9059|
|E74.3-G3-Positive-Local-DCR|91.9584 ± 3.3462|91.7882 ± 3.3683|93.6954 ± 3.4619|89.5122 ± 4.3152|94.0641 ± 3.6197|

Fold 2对10折均值的贡献（10折mean − 排除Fold 2后的9折mean，百分点；仅描述性）：
|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E74.3-G1-Local-DCR|-0.0134|+0.0023|-0.0589|+0.1572|-0.1526|
|E74.3-G2-Local-DCR-3x|-0.1157|-0.0816|-0.0093|+0.4011|-0.5643|
|E74.3-G3-Positive-Local-DCR|-0.0004|+0.0073|-0.1012|+0.0732|-0.0586|

CORR与非CORR关系/传播诊断（每组×模态，30折checkpoint均值）：
|Group|Modality|Train pair AUC|Unexposed test pair AUC|q mean|Modified edges %|DeltaA/S %|Fused DCR/input %|
|---|---|---:|---:|---:|---:|---:|---:|
|E74.3-G1-Local-DCR|PHENO|0.8009|0.5014|0.5002|3.1604|0.0053|0.00000000|
|E74.3-G1-Local-DCR|ANAT|0.6026|0.5006|0.5125|0.9764|0.0014|0.00000001|
|E74.3-G1-Local-DCR|FUNC|0.6557|0.5038|0.5075|2.8122|0.0033|0.00000001|
|E74.3-G1-Local-DCR|CORR|0.9983|0.8406|0.4757|100.0000|69.4045|0.09410744|
|E74.3-G2-Local-DCR-3x|PHENO|0.7847|0.5025|0.4991|3.3947|0.0051|0.00000000|
|E74.3-G2-Local-DCR-3x|ANAT|0.5990|0.5029|0.5126|0.8436|0.0013|0.00000003|
|E74.3-G2-Local-DCR-3x|FUNC|0.6458|0.5042|0.5009|2.4156|0.0029|0.00000001|
|E74.3-G2-Local-DCR-3x|CORR|0.9918|0.8404|0.4208|100.0000|65.4490|0.27633243|
|E74.3-G3-Positive-Local-DCR|PHENO|0.7575|0.5014|0.4976|2.5537|0.0038|0.00000000|
|E74.3-G3-Positive-Local-DCR|ANAT|0.5834|0.5013|0.5129|0.5236|0.0008|0.00000002|
|E74.3-G3-Positive-Local-DCR|FUNC|0.6234|0.5021|0.5024|1.9869|0.0022|0.00000001|
|E74.3-G3-Positive-Local-DCR|CORR|0.9941|0.8255|0.5036|100.0000|60.7414|0.40457693|

各组的Disease-neutral、CORR-neutral、Score-permuted和Direct-off干预均产生少量硬翻转；逐折结果显示这些翻转均为full模型正确、干预后错误，没有纠错样本。平均ACC差异及每组汇总见`results/interventions.csv`。因此可称有小幅可测量的关系对应/残差依赖，但不能称为稳健的大幅诊断增益。
本次更正不改动训练源码或checkpoint。初版审计脚本SHA：`e6c950c3e5f19295ec071db46cbd231d95d8f8f66e9b658321e35bd1f9040006`；组间配对方向修复后SHA：`7f0a3dddd6e9fdd8464e51611a191bebec5f888cb86eddddd8900dd73cf950e4`。当前正式审计仍为30/30通过，交叉组共有epoch RNG匹配无差异。
<!-- EXPERIMENT74_3_LOCAL_DCR_AUDIT_CORRECTION_2026-11-06 -->
