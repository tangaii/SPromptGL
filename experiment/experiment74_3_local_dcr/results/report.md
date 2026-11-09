# Experiment 74.3：Local-DCR

状态：**COMPLETE_AUDITED**；checkpoint独立审计30/30；设备`PPU-ZW810E`；本实验worker峰值10，全局并发峰值10；训练耗时859.7s；重试0；OOM 0。

## 源码与协议核验

依据SPromptGL MICCAI 2025论文Figure 1及GFR/CAM定义核对实际GraphLearn、两层GCN、CAM与分类器；沿用历史运行损失，不恢复只见于论文公式的ALoss。E74.2固定提交`6555308455814af5b6b4c82ceece73eb2333be15`及本地SHA锁定源码已核验。BrainPrompt疾病级图/语义监督仅作为设计参考，本实验未导入；阅读SABER论文作为门控残差参考，没有声称审计其官方源码。

保留val=test、交叉验证前RFE/标准化及871节点传导图的历史协议。Fold 2继续将1-based row 377（ASD）和407（TD）同时用于训练与测试，这是标签暴露，不是正常交叉验证；剔除该折不能消除其余协议局限。

## 分类结果

10折mean ± population SD，单位%；Hard-AUC按历史硬预测口径。

|组别|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
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

## 逐折配对比较

差异单位为百分点，W/T/L为新组胜/平/负。完整10折及剔除Fold 2后的9折差异均见`paired_comparisons.csv`。

|比较|范围|指标|平均差异|W/T/L|
|---|---|---|---:|---:|
|E74.3-G1-Local-DCR vs E74.2|all10|acc|+0.2299|3/6/1|
|E74.3-G1-Local-DCR vs E74.2|all10|hard_auc|+0.1965|3/4/3|
|E74.3-G1-Local-DCR vs E74.2|all10|prob_auc|+0.3233|8/0/2|
|E74.3-G1-Local-DCR vs E74.2|all10|sen|-0.2500|3/4/3|
|E74.3-G1-Local-DCR vs E74.2|all10|spe|+0.6429|2/7/1|
|E74.3-G1-Local-DCR vs E74.2|nine_without_fold2|acc|+0.1277|2/6/1|
|E74.3-G1-Local-DCR vs E74.2|nine_without_fold2|hard_auc|+0.0828|2/4/3|
|E74.3-G1-Local-DCR vs E74.2|nine_without_fold2|prob_auc|+0.2591|7/0/2|
|E74.3-G1-Local-DCR vs E74.2|nine_without_fold2|sen|-0.5488|2/4/3|
|E74.3-G1-Local-DCR vs E74.2|nine_without_fold2|spe|+0.7144|2/6/1|
|E74.3-G1-Local-DCR vs E74.1-G1|all10|acc|+0.1149|3/5/2|
|E74.3-G1-Local-DCR vs E74.1-G1|all10|hard_auc|+0.0908|3/4/3|
|E74.3-G1-Local-DCR vs E74.1-G1|all10|prob_auc|+0.5648|7/0/3|
|E74.3-G1-Local-DCR vs E74.1-G1|all10|sen|-0.2439|3/4/3|
|E74.3-G1-Local-DCR vs E74.1-G1|all10|spe|+0.4255|1/9/0|
|E74.3-G1-Local-DCR vs E74.1-G1|nine_without_fold2|acc|+0.2554|3/5/1|
|E74.3-G1-Local-DCR vs E74.1-G1|nine_without_fold2|hard_auc|+0.2364|3/4/2|
|E74.3-G1-Local-DCR vs E74.1-G1|nine_without_fold2|prob_auc|+0.5981|6/0/3|
|E74.3-G1-Local-DCR vs E74.1-G1|nine_without_fold2|sen|+0.0000|3/4/2|
|E74.3-G1-Local-DCR vs E74.1-G1|nine_without_fold2|spe|+0.4728|1/8/0|
|E74.3-G1-Local-DCR vs E74.1-G2|all10|acc|+0.2299|2/8/0|
|E74.3-G1-Local-DCR vs E74.1-G2|all10|hard_auc|+0.2044|3/5/2|
|E74.3-G1-Local-DCR vs E74.1-G2|all10|prob_auc|+0.4088|7/0/3|
|E74.3-G1-Local-DCR vs E74.1-G2|all10|sen|-0.0122|2/6/2|
|E74.3-G1-Local-DCR vs E74.1-G2|all10|spe|+0.4209|3/6/1|
|E74.3-G1-Local-DCR vs E74.1-G2|nine_without_fold2|acc|+0.1277|1/8/0|
|E74.3-G1-Local-DCR vs E74.1-G2|nine_without_fold2|hard_auc|+0.0916|2/5/2|
|E74.3-G1-Local-DCR vs E74.1-G2|nine_without_fold2|prob_auc|+0.3364|6/0/3|
|E74.3-G1-Local-DCR vs E74.1-G2|nine_without_fold2|sen|-0.2846|1/6/2|
|E74.3-G1-Local-DCR vs E74.1-G2|nine_without_fold2|spe|+0.4677|3/5/1|
|E74.3-G1-Local-DCR vs E74|all10|acc|+0.5747|5/4/1|
|E74.3-G1-Local-DCR vs E74|all10|hard_auc|+0.5896|6/2/2|
|E74.3-G1-Local-DCR vs E74|all10|prob_auc|+0.0839|6/0/4|
|E74.3-G1-Local-DCR vs E74|all10|sen|+0.9756|4/4/2|
|E74.3-G1-Local-DCR vs E74|all10|spe|+0.2035|3/5/2|
|E74.3-G1-Local-DCR vs E74|nine_without_fold2|acc|+0.3831|4/4/1|
|E74.3-G1-Local-DCR vs E74|nine_without_fold2|hard_auc|+0.3841|5/2/2|
|E74.3-G1-Local-DCR vs E74|nine_without_fold2|prob_auc|+0.0519|5/0/4|
|E74.3-G1-Local-DCR vs E74|nine_without_fold2|sen|+0.5420|3/4/2|
|E74.3-G1-Local-DCR vs E74|nine_without_fold2|spe|+0.2261|3/4/2|
|E74.3-G1-Local-DCR vs E69.3-G2|all10|acc|+0.0000|2/6/2|
|E74.3-G1-Local-DCR vs E69.3-G2|all10|hard_auc|-0.0589|2/6/2|
|E74.3-G1-Local-DCR vs E69.3-G2|all10|prob_auc|+0.6328|7/1/2|
|E74.3-G1-Local-DCR vs E69.3-G2|all10|sen|-0.7561|1/7/2|
|E74.3-G1-Local-DCR vs E69.3-G2|all10|spe|+0.6383|2/8/0|
|E74.3-G1-Local-DCR vs E69.3-G2|nine_without_fold2|acc|-0.1277|1/6/2|
|E74.3-G1-Local-DCR vs E69.3-G2|nine_without_fold2|hard_auc|-0.2009|1/6/2|
|E74.3-G1-Local-DCR vs E69.3-G2|nine_without_fold2|prob_auc|+0.5794|6/1/2|
|E74.3-G1-Local-DCR vs E69.3-G2|nine_without_fold2|sen|-1.1111|0/7/2|
|E74.3-G1-Local-DCR vs E69.3-G2|nine_without_fold2|spe|+0.7092|2/7/0|
|E74.3-G1-Local-DCR vs E62-E2|all10|acc|+0.2299|2/6/2|
|E74.3-G1-Local-DCR vs E62-E2|all10|hard_auc|+0.1585|2/6/2|
|E74.3-G1-Local-DCR vs E62-E2|all10|prob_auc|+0.8078|7/1/2|
|E74.3-G1-Local-DCR vs E62-E2|all10|sen|-0.7561|1/7/2|
|E74.3-G1-Local-DCR vs E62-E2|all10|spe|+1.0731|3/7/0|
|E74.3-G1-Local-DCR vs E62-E2|nine_without_fold2|acc|-0.1277|1/6/2|
|E74.3-G1-Local-DCR vs E62-E2|nine_without_fold2|hard_auc|-0.2009|1/6/2|
|E74.3-G1-Local-DCR vs E62-E2|nine_without_fold2|prob_auc|+0.5794|6/1/2|
|E74.3-G1-Local-DCR vs E62-E2|nine_without_fold2|sen|-1.1111|0/7/2|
|E74.3-G1-Local-DCR vs E62-E2|nine_without_fold2|spe|+0.7092|2/7/0|
|E74.3-G2-Local-DCR-3x vs E74.2|all10|acc|+0.0013|2/6/2|
|E74.3-G2-Local-DCR-3x vs E74.2|all10|hard_auc|-0.0030|4/3/3|
|E74.3-G2-Local-DCR-3x vs E74.2|all10|prob_auc|-0.4947|4/0/6|
|E74.3-G2-Local-DCR-3x vs E74.2|all10|sen|-0.0061|4/4/2|
|E74.3-G2-Local-DCR-3x vs E74.2|all10|spe|+0.0000|3/4/3|
|E74.3-G2-Local-DCR-3x vs E74.2|nine_without_fold2|acc|+0.0015|2/5/2|
|E74.3-G2-Local-DCR-3x vs E74.2|nine_without_fold2|hard_auc|-0.0328|3/3/3|
|E74.3-G2-Local-DCR-3x vs E74.2|nine_without_fold2|prob_auc|-0.6086|3/0/6|
|E74.3-G2-Local-DCR-3x vs E74.2|nine_without_fold2|sen|-0.5488|3/4/2|
|E74.3-G2-Local-DCR-3x vs E74.2|nine_without_fold2|spe|+0.4831|3/4/2|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|all10|acc|-0.1136|4/2/4|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|all10|hard_auc|-0.1087|4/1/5|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|all10|prob_auc|-0.2533|4/0/6|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|all10|sen|-0.0000|4/3/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|all10|spe|-0.2174|3/5/2|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|nine_without_fold2|acc|+0.1292|4/2/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|nine_without_fold2|hard_auc|+0.1208|4/1/4|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|nine_without_fold2|prob_auc|-0.2696|4/0/5|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|nine_without_fold2|sen|+0.0000|4/2/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G1|nine_without_fold2|spe|+0.2415|3/5/1|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|all10|acc|+0.0013|2/5/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|all10|hard_auc|+0.0048|3/3/4|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|all10|prob_auc|-0.4092|5/0/5|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|all10|sen|+0.2317|3/4/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|all10|spe|-0.2220|3/5/2|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|nine_without_fold2|acc|+0.0015|2/4/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|nine_without_fold2|hard_auc|-0.0241|2/3/4|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|nine_without_fold2|prob_auc|-0.5313|4/0/5|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|nine_without_fold2|sen|-0.2846|2/4/3|
|E74.3-G2-Local-DCR-3x vs E74.1-G2|nine_without_fold2|spe|+0.2364|3/5/1|
|E74.3-G2-Local-DCR-3x vs E74|all10|acc|+0.3461|5/3/2|
|E74.3-G2-Local-DCR-3x vs E74|all10|hard_auc|+0.3901|7/1/2|
|E74.3-G2-Local-DCR-3x vs E74|all10|prob_auc|-0.7342|3/1/6|
|E74.3-G2-Local-DCR-3x vs E74|all10|sen|+1.2195|5/4/1|
|E74.3-G2-Local-DCR-3x vs E74|all10|spe|-0.4394|4/2/4|
|E74.3-G2-Local-DCR-3x vs E74|nine_without_fold2|acc|+0.2569|4/3/2|
|E74.3-G2-Local-DCR-3x vs E74|nine_without_fold2|hard_auc|+0.2684|6/1/2|
|E74.3-G2-Local-DCR-3x vs E74|nine_without_fold2|prob_auc|-0.8158|3/0/6|
|E74.3-G2-Local-DCR-3x vs E74|nine_without_fold2|sen|+0.5420|4/4/1|
|E74.3-G2-Local-DCR-3x vs E74|nine_without_fold2|spe|-0.0051|4/2/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|all10|acc|-0.2286|2/5/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|all10|hard_auc|-0.2584|3/3/4|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|all10|prob_auc|-0.1853|7/0/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|all10|sen|-0.5122|2/5/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|all10|spe|-0.0046|4/4/2|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|nine_without_fold2|acc|-0.2540|2/4/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|nine_without_fold2|hard_auc|-0.3166|2/3/4|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|nine_without_fold2|prob_auc|-0.2883|6/0/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|nine_without_fold2|sen|-1.1111|1/5/3|
|E74.3-G2-Local-DCR-3x vs E69.3-G2|nine_without_fold2|spe|+0.4780|4/4/1|
|E74.3-G2-Local-DCR-3x vs E62-E2|all10|acc|+0.0013|3/4/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|all10|hard_auc|-0.0410|3/3/4|
|E74.3-G2-Local-DCR-3x vs E62-E2|all10|prob_auc|-0.0103|7/0/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|all10|sen|-0.5122|2/5/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|all10|spe|+0.4302|4/5/1|
|E74.3-G2-Local-DCR-3x vs E62-E2|nine_without_fold2|acc|-0.2540|2/4/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|nine_without_fold2|hard_auc|-0.3166|2/3/4|
|E74.3-G2-Local-DCR-3x vs E62-E2|nine_without_fold2|prob_auc|-0.2883|6/0/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|nine_without_fold2|sen|-1.1111|1/5/3|
|E74.3-G2-Local-DCR-3x vs E62-E2|nine_without_fold2|spe|+0.4780|4/4/1|
|E74.3-G3-Positive-Local-DCR vs E74.2|all10|acc|+0.1136|3/5/2|
|E74.3-G3-Positive-Local-DCR vs E74.2|all10|hard_auc|+0.1513|4/4/2|
|E74.3-G3-Positive-Local-DCR vs E74.2|all10|prob_auc|-0.0913|5/0/5|
|E74.3-G3-Positive-Local-DCR vs E74.2|all10|sen|+0.5061|4/5/1|
|E74.3-G3-Positive-Local-DCR vs E74.2|all10|spe|-0.2035|1/7/2|
|E74.3-G3-Positive-Local-DCR vs E74.2|nine_without_fold2|acc|-0.0015|2/5/2|
|E74.3-G3-Positive-Local-DCR vs E74.2|nine_without_fold2|hard_auc|+0.0326|3/4/2|
|E74.3-G3-Positive-Local-DCR vs E74.2|nine_without_fold2|prob_auc|-0.1132|4/0/5|
|E74.3-G3-Positive-Local-DCR vs E74.2|nine_without_fold2|sen|+0.2913|3/5/1|
|E74.3-G3-Positive-Local-DCR vs E74.2|nine_without_fold2|spe|-0.2261|1/6/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|all10|acc|-0.0013|3/4/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|all10|hard_auc|+0.0456|5/2/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|all10|prob_auc|+0.1502|4/0/6|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|all10|sen|+0.5122|5/3/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|all10|spe|-0.4209|1/6/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|nine_without_fold2|acc|+0.1263|3/4/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|nine_without_fold2|hard_auc|+0.1862|5/2/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|nine_without_fold2|prob_auc|+0.2258|4/0/5|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|nine_without_fold2|sen|+0.8401|5/3/1|
|E74.3-G3-Positive-Local-DCR vs E74.1-G1|nine_without_fold2|spe|-0.4677|1/5/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|all10|acc|+0.1136|4/3/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|all10|hard_auc|+0.1592|5/2/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|all10|prob_auc|-0.0058|6/1/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|all10|sen|+0.7439|4/4/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|all10|spe|-0.4255|1/7/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|nine_without_fold2|acc|-0.0015|3/3/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|nine_without_fold2|hard_auc|+0.0414|4/2/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|nine_without_fold2|prob_auc|-0.0359|5/1/3|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|nine_without_fold2|sen|+0.5556|3/4/2|
|E74.3-G3-Positive-Local-DCR vs E74.1-G2|nine_without_fold2|spe|-0.4728|1/6/2|
|E74.3-G3-Positive-Local-DCR vs E74|all10|acc|+0.4585|4/5/1|
|E74.3-G3-Positive-Local-DCR vs E74|all10|hard_auc|+0.5444|7/2/1|
|E74.3-G3-Positive-Local-DCR vs E74|all10|prob_auc|-0.3308|5/0/5|
|E74.3-G3-Positive-Local-DCR vs E74|all10|sen|+1.7317|5/5/0|
|E74.3-G3-Positive-Local-DCR vs E74|all10|spe|-0.6429|2/4/4|
|E74.3-G3-Positive-Local-DCR vs E74|nine_without_fold2|acc|+0.2540|3/5/1|
|E74.3-G3-Positive-Local-DCR vs E74|nine_without_fold2|hard_auc|+0.3339|6/2/1|
|E74.3-G3-Positive-Local-DCR vs E74|nine_without_fold2|prob_auc|-0.3204|5/0/4|
|E74.3-G3-Positive-Local-DCR vs E74|nine_without_fold2|sen|+1.3821|4/5/0|
|E74.3-G3-Positive-Local-DCR vs E74|nine_without_fold2|spe|-0.7144|2/3/4|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|all10|acc|-0.1162|3/3/4|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|all10|hard_auc|-0.1041|3/3/4|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|all10|prob_auc|+0.2182|8/0/2|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|all10|sen|+0.0000|2/7/1|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|all10|spe|-0.2081|2/5/3|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|nine_without_fold2|acc|-0.2569|2/3/4|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|nine_without_fold2|hard_auc|-0.2511|2/3/4|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|nine_without_fold2|prob_auc|+0.2071|7/0/2|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|nine_without_fold2|sen|-0.2710|1/7/1|
|E74.3-G3-Positive-Local-DCR vs E69.3-G2|nine_without_fold2|spe|-0.2313|2/4/3|
|E74.3-G3-Positive-Local-DCR vs E62-E2|all10|acc|+0.1136|3/3/4|
|E74.3-G3-Positive-Local-DCR vs E62-E2|all10|hard_auc|+0.1133|3/3/4|
|E74.3-G3-Positive-Local-DCR vs E62-E2|all10|prob_auc|+0.3932|8/0/2|
|E74.3-G3-Positive-Local-DCR vs E62-E2|all10|sen|+0.0000|2/7/1|
|E74.3-G3-Positive-Local-DCR vs E62-E2|all10|spe|+0.2266|3/4/3|
|E74.3-G3-Positive-Local-DCR vs E62-E2|nine_without_fold2|acc|-0.2569|2/3/4|
|E74.3-G3-Positive-Local-DCR vs E62-E2|nine_without_fold2|hard_auc|-0.2511|2/3/4|
|E74.3-G3-Positive-Local-DCR vs E62-E2|nine_without_fold2|prob_auc|+0.2071|7/0/2|
|E74.3-G3-Positive-Local-DCR vs E62-E2|nine_without_fold2|sen|-0.2710|1/7/1|
|E74.3-G3-Positive-Local-DCR vs E62-E2|nine_without_fold2|spe|-0.2313|2/4/3|
|G2 vs G1|all10|acc|-0.2286|1/5/4|
|G2 vs G1|all10|hard_auc|-0.1995|1/4/5|
|G2 vs G1|all10|prob_auc|-0.8181|2/0/8|
|G2 vs G1|all10|sen|+0.2439|3/5/2|
|G2 vs G1|all10|spe|-0.6429|2/5/3|
|G2 vs G1|nine_without_fold2|acc|-0.1263|1/5/3|
|G2 vs G1|nine_without_fold2|hard_auc|-0.1156|1/4/4|
|G2 vs G1|nine_without_fold2|prob_auc|-0.8677|2/0/7|
|G2 vs G1|nine_without_fold2|sen|+0.0000|2/5/2|
|G2 vs G1|nine_without_fold2|spe|-0.2313|2/5/2|
|G3 vs G2|all10|acc|+0.1123|3/4/3|
|G3 vs G2|all10|hard_auc|+0.1543|4/3/3|
|G3 vs G2|all10|prob_auc|+0.4035|7/0/3|
|G3 vs G2|all10|sen|+0.5122|2/6/2|
|G3 vs G2|all10|spe|-0.2035|2/4/4|
|G3 vs G2|nine_without_fold2|acc|-0.0029|2/4/3|
|G3 vs G2|nine_without_fold2|hard_auc|+0.0654|3/3/3|
|G3 vs G2|nine_without_fold2|prob_auc|+0.4954|7/0/2|
|G3 vs G2|nine_without_fold2|sen|+0.8401|2/6/1|
|G3 vs G2|nine_without_fold2|spe|-0.7092|1/4/4|
|G1 vs E74.2|all10|acc|+0.2299|3/6/1|
|G1 vs E74.2|all10|hard_auc|+0.1965|3/4/3|
|G1 vs E74.2|all10|prob_auc|+0.3233|8/0/2|
|G1 vs E74.2|all10|sen|-0.2500|3/4/3|
|G1 vs E74.2|all10|spe|+0.6429|2/7/1|
|G1 vs E74.2|nine_without_fold2|acc|+0.1277|2/6/1|
|G1 vs E74.2|nine_without_fold2|hard_auc|+0.0828|2/4/3|
|G1 vs E74.2|nine_without_fold2|prob_auc|+0.2591|7/0/2|
|G1 vs E74.2|nine_without_fold2|sen|-0.5488|2/4/3|
|G1 vs E74.2|nine_without_fold2|spe|+0.7144|2/6/1|

## 机制与干预

机制记录120行（4模态×30 checkpoint），包括关系训练/未暴露测试AUC、评分分布、门控、有效边及正负增量、DeltaA变化、局部残差/GCN传播比例和关系分类梯度。固定checkpoint干预记录120行，包含四种干预的ACC/AUC/SEN/SPE、硬翻转、纠错/新增错误、ASD概率及融合表示变化。

- E74.3-G1-Local-DCR融合加权DCR消息/输入范数：PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.094107%.
- E74.3-G2-Local-DCR-3x融合加权DCR消息/输入范数：PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.276332%.
- E74.3-G3-Positive-Local-DCR融合加权DCR消息/输入范数：PHENO 0.000000%, ANAT 0.000000%, FUNC 0.000000%, CORR 0.404577%.
- E74.3-G1-Local-DCR/disease_neutral：full−干预ACC +0.3448 pp，Prob-AUC +0.0107 pp，硬翻转3例。
- E74.3-G1-Local-DCR/corr_neutral：full−干预ACC +0.3448 pp，Prob-AUC +0.0107 pp，硬翻转3例。
- E74.3-G1-Local-DCR/score_permuted：full−干预ACC +0.4598 pp，Prob-AUC -0.0000 pp，硬翻转4例。
- E74.3-G1-Local-DCR/direct_off：full−干预ACC +0.3448 pp，Prob-AUC +0.0107 pp，硬翻转3例。
- E74.3-G2-Local-DCR-3x/disease_neutral：full−干预ACC +0.2299 pp，Prob-AUC -0.0001 pp，硬翻转2例。
- E74.3-G2-Local-DCR-3x/corr_neutral：full−干预ACC +0.2299 pp，Prob-AUC -0.0001 pp，硬翻转2例。
- E74.3-G2-Local-DCR-3x/score_permuted：full−干预ACC +0.1149 pp，Prob-AUC +0.0054 pp，硬翻转1例。
- E74.3-G2-Local-DCR-3x/direct_off：full−干预ACC +0.2299 pp，Prob-AUC -0.0001 pp，硬翻转2例。
- E74.3-G3-Positive-Local-DCR/disease_neutral：full−干预ACC +0.1149 pp，Prob-AUC -0.0160 pp，硬翻转1例。
- E74.3-G3-Positive-Local-DCR/corr_neutral：full−干预ACC +0.1149 pp，Prob-AUC -0.0160 pp，硬翻转1例。
- E74.3-G3-Positive-Local-DCR/score_permuted：full−干预ACC +0.2299 pp，Prob-AUC +0.0000 pp，硬翻转2例。
- E74.3-G3-Positive-Local-DCR/direct_off：full−干预ACC +0.1149 pp，Prob-AUC -0.0160 pp，硬翻转1例。

## 审计与结论

训练源码SHA `3122f95ee5e5f27027f20d894b20d1d6071de79b02388cedee7cf711b4e05cc4`；审计脚本SHA `7f0a3dddd6e9fdd8464e51611a191bebec5f888cb86eddddd8900dd73cf950e4`。组间共有epoch RNG核验：10327/10327；相对E74.2共有epoch：10338/10338。30/30 checkpoint独立读回并复算预测、指标、图性质、调用次数及干预；训练前身份和优化器继承均逐折核验。首次预检曾有一次Tensor JSON序列化记录错误，未执行训练更新；改为写入图SHA后完整预检重跑通过。正式任务失败、重试和OOM均为0。

结论：**PARTIAL**。GO requires at least one fixed group to meet the prespecified ACC/Hard-AUC/SEN targets, improve nine-fold ACC over E74.2, and show directionally beneficial disease-score intervention effects. PARTIAL requires a measurable improvement or mechanism signal; otherwise NO-GO.

不根据本轮结果追加组或调整固定参数。建议下一步使用fold-local RFE/标准化、独立验证集选模和无标签重叠协议，检验关系对应是否对决策稳定地产生可复现作用。

实验目录：`/model/spromptgl/experiment/experiment74_3_local_dcr`；首选OSSFS路径因errno 5失败后回退到`/model`。报告、manifest和审计表分别位于`/model/spromptgl/experiment/experiment74_3_local_dcr/results/report.md`、`/model/spromptgl/experiment/experiment74_3_local_dcr/manifest.json`和`/model/spromptgl/experiment/experiment74_3_local_dcr/results/checkpoint_audit.csv`。
