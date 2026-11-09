# Experiment 74.4：CORR评分软化与正向局部疾病残差

状态：**COMPLETE_AUDITED**；正式任务30/30；独立Checkpoint审计30/30；设备`PPU-ZW810E`；worker上限10、实际峰值9；耗时660.5s；重试0，OOM 0。

## 源码与方法核验

E74.3-G1只读基线锁定分支`upload/experiment74-3-local-dcr`@`e05275cb4b18dcd01679682b9df02f3cbc13b713`，并校验run/local-DCR/audit/config/manifest SHA；三组均从逐折E61 epoch-10 warmup恢复，未从任何E74.3最佳Checkpoint微调。SPromptGL论文描述GFR、交互图、GCN与语义提示路径；本次保留实际代码损失，不恢复论文中未进入历史总损失的ALoss。BrainPrompt论文使用人口统计患者信息和疾病级语义监督，本次不引入文本或Prompt模块。SABER仅按论文的语义可靠性/决策交互机制阅读；没有审计或复现未确认的官方代码。

设计：历史E74.3-G1为T=1/signed/0.02；E74.4-G1为T=2/signed/0.02；G2为T=1/positive-only/0.02；G3为T=2/positive-only/0.02。T仅作用于CORR构图概率，患者对BCE仍使用原始Logit；正增量裁剪仅进入直接DCR残差，统一图和GCN仍保留负边抑制。非CORR图、原损失、优化器及每模态一次GCN均沿用E74.3。

历史评价限制：继续使用val=test、CV前RFE/标准化和871节点transductive图；Fold 2的1-based row 377（ASD）及407（TD）同时出现在训练与测试索引。九折描述不能消除其他泄漏与选模偏差，因此不是无偏泛化验证。

## 十折结果

均值±population SD，单位%；Hard-AUC为历史硬预测口径，Prob-AUC为连续概率ROC-AUC。

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
|E74.4-G1-Soft-DCR|91.3858 ± 2.6904|91.1021 ± 2.7517|92.9784 ± 2.8298|87.3476 ± 4.5089|94.8566 ± 3.0874|90.3484 ± 3.0210|
|E74.4-G2-Positive-DCR|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|E74.4-G3-Soft-Positive-DCR|91.8443 ± 2.6508|91.6225 ± 2.6751|93.6961 ± 3.0908|88.8232 ± 4.6645|94.4218 ± 4.3507|90.9686 ± 2.8953|

## 预注册因素比较

差异为逐折配对后再求均值，单位百分点；W/T/L比较每折差值与0，95%区间为描述性t区间。完整数据见`paired_comparisons.csv`。

|对比|范围|ACC Δ|Hard-AUC Δ|Prob-AUC Δ|SEN Δ|SPE Δ|ACC W/T/L|
|---|---|---:|---:|---:|---:|---:|---:|
|T2−T1 signed (G1−E74.3-G1)|all10|-0.6883|-0.7385|-1.0303|-1.4817|+0.0046|1/2/7|
|T2−T1 signed (G1−E74.3-G1)|nine_without_fold2|-0.6371|-0.6851|-0.8561|-1.3753|+0.0051|1/2/6|
|T2−T1 positive (G3−G2)|all10|-0.3448|-0.4069|+0.2015|-1.2439|+0.4302|0/7/3|
|T2−T1 positive (G3−G2)|nine_without_fold2|-0.3831|-0.4521|+0.1944|-1.3821|+0.4780|0/6/3|
|Positive−signed T1 (G2−E74.3-G1)|all10|+0.1149|+0.1887|-0.5141|+1.2378|-0.8603|2/7/1|
|Positive−signed T1 (G2−E74.3-G1)|nine_without_fold2|+0.1277|+0.2097|-0.4534|+1.3753|-0.9559|2/6/1|
|Positive−signed T2 (G3−G1)|all10|+0.4585|+0.5204|+0.7177|+1.4756|-0.4348|4/6/0|
|Positive−signed T2 (G3−G1)|nine_without_fold2|+0.3817|+0.4427|+0.5971|+1.3686|-0.4831|3/6/0|
|Temperature×positive interaction|all10|+0.3435|+0.3317|+1.2318|+0.2378|+0.4255|5/3/2|
|Temperature×positive interaction|nine_without_fold2|+0.2540|+0.2330|+1.0506|-0.0068|+0.4728|4/3/2|

### 新组相对历史参照
- `E74.4-G1-Soft-DCR vs E69.3-G2`：acc -0.6883 pp (0/4/6), hard_auc -0.7974 pp (0/3/7), prob_auc -0.3975 pp (4/0/6), sen -2.2378 pp (0/5/5), spe +0.6429 pp (4/4/2)。
- `E74.4-G1-Soft-DCR vs E74.3-G1`：acc -0.6883 pp (1/2/7), hard_auc -0.7385 pp (1/2/7), prob_auc -1.0303 pp (1/0/9), sen -1.4817 pp (1/4/5), spe +0.0046 pp (2/6/2)。
- `E74.4-G1-Soft-DCR vs E74.2`：acc -0.4585 pp (0/6/4), hard_auc -0.5421 pp (0/5/5), prob_auc -0.7070 pp (2/0/8), sen -1.7317 pp (0/7/3), spe +0.6475 pp (2/6/2)。
- `E74.4-G1-Soft-DCR vs E62-E2`：acc -0.4585 pp (1/3/6), hard_auc -0.5800 pp (1/2/7), prob_auc -0.2226 pp (5/0/5), sen -2.2378 pp (0/5/5), spe +1.0777 pp (5/3/2)。
- `E74.4-G1-Soft-DCR vs E74`：acc -0.1136 pp (1/7/2), hard_auc -0.1490 pp (1/5/4), prob_auc -0.9465 pp (3/1/6), sen -0.5061 pp (1/7/2), spe +0.2081 pp (2/6/2)。
- `E74.4-G1-Soft-DCR vs E74.1-G1`：acc -0.5734 pp (1/4/5), hard_auc -0.6477 pp (1/3/6), prob_auc -0.4655 pp (4/0/6), sen -1.7256 pp (1/5/4), spe +0.4302 pp (3/5/2)。
- `E74.4-G1-Soft-DCR vs E74.1-G2`：acc -0.4585 pp (1/4/5), hard_auc -0.5342 pp (1/2/7), prob_auc -0.6215 pp (3/0/7), sen -1.4939 pp (1/4/5), spe +0.4255 pp (4/4/2)。
- `E74.4-G1-Soft-DCR vs E74.3-G2`：acc -0.4598 pp (1/4/5), hard_auc -0.5390 pp (1/1/8), prob_auc -0.2123 pp (3/0/7), sen -1.7256 pp (0/4/6), spe +0.6475 pp (4/4/2)。
- `E74.4-G1-Soft-DCR vs E74.3-G3`：acc -0.5721 pp (0/6/4), hard_auc -0.6934 pp (0/3/7), prob_auc -0.6157 pp (2/0/8), sen -2.2378 pp (0/4/6), spe +0.8511 pp (3/5/2)。
- `E74.4-G2-Positive-DCR vs E69.3-G2`：acc +0.1149 pp (2/7/1), hard_auc +0.1298 pp (3/6/1), prob_auc +0.1187 pp (6/0/4), sen +0.4817 pp (2/7/1), spe -0.2220 pp (1/8/1)。
- `E74.4-G2-Positive-DCR vs E74.3-G1`：acc +0.1149 pp (2/7/1), hard_auc +0.1887 pp (3/6/1), prob_auc -0.5141 pp (2/0/8), sen +1.2378 pp (3/6/1), spe -0.8603 pp (0/8/2)。
- `E74.4-G2-Positive-DCR vs E74.2`：acc +0.3448 pp (4/5/1), hard_auc +0.3852 pp (5/4/1), prob_auc -0.1908 pp (4/1/5), sen +0.9878 pp (4/5/1), spe -0.2174 pp (1/7/2)。
- `E74.4-G2-Positive-DCR vs E62-E2`：acc +0.3448 pp (2/7/1), hard_auc +0.3472 pp (3/6/1), prob_auc +0.2936 pp (6/0/4), sen +0.4817 pp (2/7/1), spe +0.2128 pp (2/7/1)。
- `E74.4-G2-Positive-DCR vs E74`：acc +0.6897 pp (5/5/0), hard_auc +0.7783 pp (6/4/0), prob_auc -0.4303 pp (2/1/7), sen +2.2134 pp (4/6/0), spe -0.6568 pp (2/6/2)。
- `E74.4-G2-Positive-DCR vs E74.1-G1`：acc +0.2299 pp (3/6/1), hard_auc +0.2796 pp (4/5/1), prob_auc +0.0507 pp (5/0/5), sen +0.9939 pp (4/5/1), spe -0.4348 pp (0/9/1)。
- `E74.4-G2-Positive-DCR vs E74.1-G2`：acc +0.3448 pp (4/5/1), hard_auc +0.3931 pp (5/4/1), prob_auc -0.1053 pp (2/3/5), sen +1.2256 pp (4/5/1), spe -0.4394 pp (2/6/2)。
- `E74.4-G2-Positive-DCR vs E74.3-G2`：acc +0.3435 pp (4/4/2), hard_auc +0.3883 pp (5/3/2), prob_auc +0.3039 pp (4/0/6), sen +0.9939 pp (3/4/3), spe -0.2174 pp (2/5/3)。
- `E74.4-G2-Positive-DCR vs E74.3-G3`：acc +0.2312 pp (3/6/1), hard_auc +0.2339 pp (3/5/2), prob_auc -0.0995 pp (2/0/8), sen +0.4817 pp (1/7/2), spe -0.0139 pp (3/6/1)。
- `E74.4-G3-Soft-Positive-DCR vs E69.3-G2`：acc -0.2299 pp (2/4/4), hard_auc -0.2770 pp (3/3/4), prob_auc +0.3202 pp (6/0/4), sen -0.7622 pp (2/5/3), spe +0.2081 pp (3/5/2)。
- `E74.4-G3-Soft-Positive-DCR vs E74.3-G1`：acc -0.2299 pp (1/6/3), hard_auc -0.2181 pp (3/4/3), prob_auc -0.3127 pp (2/0/8), sen -0.0061 pp (3/5/2), spe -0.4302 pp (1/6/3)。
- `E74.4-G3-Soft-Positive-DCR vs E74.2`：acc -0.0000 pp (2/6/2), hard_auc -0.0217 pp (3/4/3), prob_auc +0.0107 pp (6/1/3), sen -0.2561 pp (3/5/2), spe +0.2128 pp (2/6/2)。
- `E74.4-G3-Soft-Positive-DCR vs E62-E2`：acc -0.0000 pp (2/4/4), hard_auc -0.0596 pp (3/3/4), prob_auc +0.4951 pp (6/0/4), sen -0.7622 pp (2/5/3), spe +0.6429 pp (4/4/2)。
- `E74.4-G3-Soft-Positive-DCR vs E74`：acc +0.3448 pp (2/8/0), hard_auc +0.3714 pp (4/4/2), prob_auc -0.2288 pp (5/1/4), sen +0.9695 pp (3/5/2), spe -0.2266 pp (3/5/2)。
- `E74.4-G3-Soft-Positive-DCR vs E74.1-G1`：acc -0.1149 pp (2/5/3), hard_auc -0.1273 pp (3/2/5), prob_auc +0.2522 pp (6/0/4), sen -0.2500 pp (3/3/4), spe -0.0046 pp (2/6/2)。
- `E74.4-G3-Soft-Positive-DCR vs E74.1-G2`：acc -0.0000 pp (3/4/3), hard_auc -0.0138 pp (4/3/3), prob_auc +0.0962 pp (5/0/5), sen -0.0183 pp (3/5/2), spe -0.0093 pp (2/6/2)。
- `E74.4-G3-Soft-Positive-DCR vs E74.3-G2`：acc -0.0013 pp (3/4/3), hard_auc -0.0186 pp (4/3/3), prob_auc +0.5054 pp (6/0/4), sen -0.2500 pp (2/5/3), spe +0.2128 pp (3/4/3)。
- `E74.4-G3-Soft-Positive-DCR vs E74.3-G3`：acc -0.1136 pp (1/7/2), hard_auc -0.1730 pp (1/4/5), prob_auc +0.1020 pp (6/1/3), sen -0.7622 pp (1/5/4), spe +0.4163 pp (3/5/2)。

## 排除Fold 2

九折均值±population SD（%），仅为描述性配对敏感性结果。

|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E74.4-G1-Soft-DCR|91.4504 ± 2.8286|91.1533 ± 2.8960|93.2115 ± 2.8904|87.2967 ± 4.7501|95.0098 ± 3.2182|
|E74.4-G2-Positive-DCR|92.2153 ± 2.9131|92.0481 ± 2.8723|93.6142 ± 3.0541|90.0474 ± 4.0959|94.0487 ± 4.9847|
|E74.4-G3-Soft-Positive-DCR|91.8321 ± 2.7939|91.5960 ± 2.8186|93.8087 ± 3.2385|88.6653 ± 4.8914|94.5267 ± 4.5741|
|E74.3-G1|92.0875 ± 3.1397|91.8384 ± 3.1921|94.0677 ± 3.1331|88.6721 ± 5.0167|95.0046 ± 4.0724|
|E69.3-G2|92.2153 ± 2.8623|92.0393 ± 2.8314|93.4883 ± 3.1325|89.7832 ± 3.6049|94.2954 ± 4.1889|

九折逐折配对差（新组−历史参照，百分点）：
- `E74.4-G1-Soft-DCR vs E69.3-G2`：acc -0.7648 pp, hard_auc -0.8860 pp, prob_auc -0.2767 pp, sen -2.4864 pp, spe +0.7144 pp；ACC W/T/L 0/3/6。
- `E74.4-G1-Soft-DCR vs E74.3-G1`：acc -0.6371 pp, hard_auc -0.6851 pp, prob_auc -0.8561 pp, sen -1.3753 pp, spe +0.0051 pp；ACC W/T/L 1/2/6。
- `E74.4-G2-Positive-DCR vs E69.3-G2`：acc +0.0000 pp, hard_auc +0.0088 pp, prob_auc +0.1260 pp, sen +0.2642 pp, spe -0.2467 pp；ACC W/T/L 1/7/1。
- `E74.4-G2-Positive-DCR vs E74.3-G1`：acc +0.1277 pp, hard_auc +0.2097 pp, prob_auc -0.4534 pp, sen +1.3753 pp, spe -0.9559 pp；ACC W/T/L 2/6/1。
- `E74.4-G3-Soft-Positive-DCR vs E69.3-G2`：acc -0.3831 pp, hard_auc -0.4433 pp, prob_auc +0.3204 pp, sen -1.1179 pp, spe +0.2313 pp；ACC W/T/L 1/4/4。
- `E74.4-G3-Soft-Positive-DCR vs E74.3-G1`：acc -0.2554 pp, hard_auc -0.2424 pp, prob_auc -0.2590 pp, sen -0.0068 pp, spe -0.4780 pp；ACC W/T/L 1/5/3。

## CORR机制与传播

每个数字为30份checkpoint的CORR模态均值；机制明细按group/fold/modality记录于`mechanism.csv`。q_raw为未缩放疾病关系概率，q_graph为只用于构图的温度后概率；T=2是固定软化，不是验证集概率校准。

|Group|T|Logit均值/SD|q_raw均值/SD|q_graph均值/SD|q_raw 0.45–0.55|q_graph 0.45–0.55|未暴露测试对AUC|ΔA/S %|DCR消息/X %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E74.4-G1-Soft-DCR|2.0|-1.0679/4.2814|0.4794/0.4300|0.4579/0.3405|1.476%|3.018%|0.8440|54.09141%|0.0770998%|
|E74.4-G2-Positive-DCR|1.0|-0.4416/4.0612|0.5103/0.4392|0.5103/0.4392|1.447%|1.447%|0.8326|69.14425%|0.1525772%|
|E74.4-G3-Soft-Positive-DCR|2.0|-0.3047/3.4149|0.5050/0.4128|0.4906/0.3103|1.895%|3.821%|0.8369|48.80570%|0.1022299%|

## 固定Checkpoint干预

干预不重训、不改标签构图。翻转索引以全局1-based行号记录，特别核对Fold 2中是否涉及377/407。

|Group|Intervention|ACC Δ full−intervention|Hard-AUC Δ|Prob-AUC Δ|Flip|Full错→干预对|Full对→干预错|Fold2 flips include 377/407|
|---|---|---:|---:|---:|---:|---:|---:|---|
|E74.4-G1-Soft-DCR|disease_neutral|+0.1149|+0.1087|-0.0212|1|0|1|[]|
|E74.4-G1-Soft-DCR|corr_neutral|+0.1149|+0.1087|-0.0212|1|0|1|[]|
|E74.4-G1-Soft-DCR|score_permuted|+0.1149|+0.1087|-0.0212|1|0|1|[]|
|E74.4-G1-Soft-DCR|direct_off|+0.1149|+0.1087|-0.0212|1|0|1|[]|
|E74.4-G2-Positive-DCR|disease_neutral|+0.1149|+0.1064|+0.0001|1|0|1|[]|
|E74.4-G2-Positive-DCR|corr_neutral|+0.1149|+0.1064|+0.0001|1|0|1|[]|
|E74.4-G2-Positive-DCR|score_permuted|+0.0000|+0.0000|+0.0053|0|0|0|[]|
|E74.4-G2-Positive-DCR|direct_off|+0.1149|+0.1064|+0.0001|1|0|1|[]|
|E74.4-G3-Soft-Positive-DCR|disease_neutral|+0.2299|+0.2337|+0.0000|2|0|2|[]|
|E74.4-G3-Soft-Positive-DCR|corr_neutral|+0.2299|+0.2337|+0.0000|2|0|2|[]|
|E74.4-G3-Soft-Positive-DCR|score_permuted|+0.1149|+0.1250|-0.0001|1|0|1|[]|
|E74.4-G3-Soft-Positive-DCR|direct_off|+0.2299|+0.2337|+0.0000|2|0|2|[]|

组别峰值：ACC `E74.4-G2-Positive-DCR` 92.1891%；Hard-AUC `E74.4-G2-Positive-DCR` 92.0294%；SEN `E74.4-G2-Positive-DCR` 90.0671%。E69.3-G2目标ACC 92.0742%、Hard-AUC 91.8995%、SEN 89.5854%。

## 审计与结论

30/30最佳Checkpoint独立重建并复算通过；硬预测严格一致，概率容差1e-8；每折三组初始模型/CAM、优化器、warmup、患者对与划分一致。单步更新与恢复重放通过，组间共有epoch RNG核对10800/10800，对E74.3-G1相同epoch核对10637/10637。

判定：**GO**。该判断要求因素比较在全十折及排除Fold 2后仍有方向性支持，并由中性化/置换干预显示合理作用；不表示无偏泛化。建议后续先处理fold-local预处理、独立验证选模及Fold 2重叠，再复核关系调整的可复现性；本轮不追加温度/尺度搜索。

训练源码SHA：`d3a60ecdfd0158be1eecb066d451e0246a417f69e17c4975fe022075a37ac9c4`；审计源码SHA：`cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb`。实验目录：`/model/spromptgl/experiment/experiment74_4_soft_positive_dcr`；manifest：`/model/spromptgl/experiment/experiment74_4_soft_positive_dcr/manifest.json`。
