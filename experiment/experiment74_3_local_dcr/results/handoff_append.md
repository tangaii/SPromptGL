<!-- EXPERIMENT74_3_LOCAL_DCR_2026-11-06 -->
## Experiment 74.3：Local-DCR（2026-11-06）

目的：在E74.2疾病边权增量残差上，仅检验模态局部化、固定3倍残差强度和正增量残差三项预注册改动；不扩展关系编码器、损失或分类主干。
代码及运行目录：`/model/spromptgl/experiment/experiment74_3_local_dcr`；首选OSSFS目录因errno 5创建失败，回退至`/model`。E74.2参考为`upload/experiment74-2-dcr-rudag`@`6555308455814af5b6b4c82ceece73eb2333be15`，源文件SHA见实验config与manifest。
G1为模态局部signed DeltaA、lambda=0.02；G2仅将lambda固定为0.06；G3继承G2，但只在直接残差分支使用clamp_min(DeltaA,0)，统一图仍保留负增量抑制。三组均为每模态单一统一图、一次GCN、原CAM/分类器及原优化器与损失。

执行状态：30/30任务、30/30独立checkpoint审计；设备PPU-ZW810E，worker上限10、实际全局并发峰值10；用时859.7s，失败/重试0，OOM 0。

十折结果（mean ± population SD，%）：
|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E69.3-G2|92.0742 ± 2.7482|91.8995 ± 2.7186|93.3759 ± 2.9908|89.5854 ± 3.4710|94.2137 ± 3.9815|91.2984 ± 2.9222|
|E62-E2|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|89.5854 ± 3.4710|93.7789 ± 4.2653|91.0789 ± 3.0892|
|E74-UDAG|91.4995 ± 2.6885|91.2511 ± 2.7308|93.9249 ± 2.6632|87.8537 ± 4.3359|94.6485 ± 3.2152|90.5218 ± 3.0064|
|G1-RUDAG-U|91.9592 ± 2.7747|91.7498 ± 2.7845|93.4439 ± 3.0255|89.0732 ± 3.9177|94.4265 ± 3.7676|91.1148 ± 3.0257|
|G2-RUDAG-Full|91.8443 ± 2.9793|91.6363 ± 3.0151|93.5999 ± 3.0799|88.8415 ± 4.3351|94.4311 ± 3.4953|90.9690 ± 3.3140|
|E74.2-G1-DCR-RUDAG|91.8443 ± 2.7963|91.6442 ± 2.7336|93.6854 ± 2.7687|89.0793 ± 3.0035|94.2091 ± 4.3328|91.0361 ± 2.9228|
|E74.3-G1-Local-DCR|92.0742 ± 2.9789|91.8406 ± 3.0283|94.0087 ± 2.9776|88.8293 ± 4.7826|94.8520 ± 3.8905|91.1921 ± 3.3226|
|E74.3-G2-Local-DCR-3x|91.8456 ± 2.6991|91.6411 ± 2.7749|93.1907 ± 3.8355|89.0732 ± 4.7674|94.2091 ± 3.2351|90.9719 ± 3.0520|
|E74.3-G3-Positive-Local-DCR|91.9579 ± 3.1744|91.7955 ± 3.1955|93.5941 ± 3.2983|89.5854 ± 4.0996|94.0056 ± 3.4385|91.1571 ± 3.4915|

排除Fold 2后的配对ACC变化（E74.3−参考，百分点；其余指标见paired_comparisons.csv）：
|比较|9折ACC变化|胜/平/负|
|---|---:|---:|
|E74.3-G1-Local-DCR−E74.2|+0.1277|2/6/1|
|E74.3-G1-Local-DCR−E74.1-G1|+0.2554|3/5/1|
|E74.3-G1-Local-DCR−E69.3-G2|-0.1277|1/6/2|
|E74.3-G1-Local-DCR−E62-E2|-0.1277|1/6/2|
|E74.3-G1-Local-DCR−E74|+0.3831|4/4/1|
|E74.3-G2-Local-DCR-3x−E74.2|+0.0015|2/5/2|
|E74.3-G2-Local-DCR-3x−E74.1-G1|+0.1292|4/2/3|
|E74.3-G2-Local-DCR-3x−E69.3-G2|-0.2540|2/4/3|
|E74.3-G2-Local-DCR-3x−E62-E2|-0.2540|2/4/3|
|E74.3-G2-Local-DCR-3x−E74|+0.2569|4/3/2|
|E74.3-G3-Positive-Local-DCR−E74.2|-0.0015|2/5/2|
|E74.3-G3-Positive-Local-DCR−E74.1-G1|+0.1263|3/4/2|
|E74.3-G3-Positive-Local-DCR−E69.3-G2|-0.2569|2/3/4|
|E74.3-G3-Positive-Local-DCR−E62-E2|-0.2569|2/3/4|
|E74.3-G3-Positive-Local-DCR−E74|+0.2540|3/5/1|
|G2−G1|-0.1263|1/5/3|
|G3−G2|-0.0029|2/4/3|
|G1−E74.2|+0.1277|2/6/1|

机制诊断：各模态训练患者对AUC、未暴露测试患者对事后AUC、关系评分分布、门控、正负边增量、图变化和GCN/DCR消息比例见mechanism.csv。按组平均的融合加权DCR/320维输入范数（%）如下：
- E74.3-G1-Local-DCR: PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.094107%.
- E74.3-G2-Local-DCR-3x: PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.276332%.
- E74.3-G3-Positive-Local-DCR: PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.404577%.

固定checkpoint干预（每组10折平均full−intervention ACC/Prob-AUC，百分点；硬翻转数）：
- E74.3-G1-Local-DCR / disease_neutral: ACC +0.3448 pp, Prob-AUC +0.0107 pp, 硬翻转 3。
- E74.3-G1-Local-DCR / corr_neutral: ACC +0.3448 pp, Prob-AUC +0.0107 pp, 硬翻转 3。
- E74.3-G1-Local-DCR / score_permuted: ACC +0.4598 pp, Prob-AUC -0.0000 pp, 硬翻转 4。
- E74.3-G1-Local-DCR / direct_off: ACC +0.3448 pp, Prob-AUC +0.0107 pp, 硬翻转 3。
- E74.3-G2-Local-DCR-3x / disease_neutral: ACC +0.2299 pp, Prob-AUC -0.0001 pp, 硬翻转 2。
- E74.3-G2-Local-DCR-3x / corr_neutral: ACC +0.2299 pp, Prob-AUC -0.0001 pp, 硬翻转 2。
- E74.3-G2-Local-DCR-3x / score_permuted: ACC +0.1149 pp, Prob-AUC +0.0054 pp, 硬翻转 1。
- E74.3-G2-Local-DCR-3x / direct_off: ACC +0.2299 pp, Prob-AUC -0.0001 pp, 硬翻转 2。
- E74.3-G3-Positive-Local-DCR / disease_neutral: ACC +0.1149 pp, Prob-AUC -0.0160 pp, 硬翻转 1。
- E74.3-G3-Positive-Local-DCR / corr_neutral: ACC +0.1149 pp, Prob-AUC -0.0160 pp, 硬翻转 1。
- E74.3-G3-Positive-Local-DCR / score_permuted: ACC +0.2299 pp, Prob-AUC +0.0000 pp, 硬翻转 2。
- E74.3-G3-Positive-Local-DCR / direct_off: ACC +0.1149 pp, Prob-AUC -0.0160 pp, 硬翻转 1。

审计：30/30 checkpoint独立读回并复算通过；每折三组warmup、模型/CAM、优化器、划分和患者对初始化身份一致。跨组共享epoch RNG匹配10327/10327；与E74.2共有epoch匹配10338/10338。历史训练曾发生的失败/重试见manifest，正式任务重试数为0。
评价限制：沿用val=test、外层交叉验证前RFE/标准化及transductive图；Fold 2仍有1-based row 377和407同时进入训练与测试。不得解释为无泄漏独立泛化估计；九折剔除不能消除其他协议限制。
结论：PARTIAL。GO requires at least one fixed group to meet the prespecified ACC/Hard-AUC/SEN targets, improve nine-fold ACC over E74.2, and show directionally beneficial disease-score intervention effects. PARTIAL requires a measurable improvement or mechanism signal; otherwise NO-GO.
下一步：不继续扫描本轮固定系数；优先改为fold-local预处理、独立验证集选模和无标签重叠评估，再检验关系边对应对分类决策的可重复作用。
报告：`/model/spromptgl/experiment/experiment74_3_local_dcr/results/report.md`；manifest：`/model/spromptgl/experiment/experiment74_3_local_dcr/manifest.json`；审计结果：`/model/spromptgl/experiment/experiment74_3_local_dcr/results/checkpoint_audit.csv`。
<!-- EXPERIMENT74_3_LOCAL_DCR_2026-11-06 -->
