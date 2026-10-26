# Experiment 43.1 — B0-B GFR机制复验

完成 60/60 唯一正式折。正式四指标为十折 mean ± population SD，单位%。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E00_B0B_REPLAY|91.7294 ± 2.8596|91.6046 ± 2.8059|93.1360 ± 4.5354|90.0732 ± 3.5117|
|E01_AGGREGATION_FIX_V2|91.6144 ± 2.9619|91.4694 ± 2.9274|93.3534 ± 4.1526|89.5854 ± 3.4710|
|E02_SAFE_LABEL_ADJ_SUP_V2|91.6144 ± 3.0498|91.4375 ± 3.0446|93.7835 ± 3.4255|89.0915 ± 3.5092|
|E03_SUPPORT_PRESERVING_CAL_V2|91.9592 ± 2.7747|91.7296 ± 2.7559|94.6300 ± 4.4709|88.8293 ± 4.2134|
|E04_MODALITY_LOCAL_PROP_V2|91.4995 ± 2.8321|91.2590 ± 2.8721|94.4265 ± 3.3763|88.0915 ± 4.4156|
|E05_FIX_PLUS_SUPPORT_CAL_V2|91.9592 ± 2.7747|91.7296 ± 2.7559|94.6300 ± 4.4709|88.8293 ± 4.2134|

## 与同折 B0-B 配对

|组别−B0-B|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|判定|
|---|---:|---:|---:|---:|---|---|---|
|E01_AGGREGATION_FIX_V2|-0.1149|-0.1352|+0.2174|-0.4878|1/7/2|2/4/4|NO-GO|
|E02_SAFE_LABEL_ADJ_SUP_V2|-0.1149|-0.1671|+0.6475|-0.9817|2/5/3|3/3/4|NO-GO|
|E03_SUPPORT_PRESERVING_CAL_V2|+0.2299|+0.1250|+1.4940|-1.2439|4/4/2|4/3/3|NO-GO|
|E04_MODALITY_LOCAL_PROP_V2|-0.2299|-0.3456|+1.2905|-1.9817|2/5/3|2/3/5|NO-GO|
|E05_FIX_PLUS_SUPPORT_CAL_V2|+0.2299|+0.1250|+1.4940|-1.2439|4/4/2|4/3/3|NO-GO|

## 基线身份及设计边界

B0-B物理不构造PEN/CLIP；PLN49,544参数，GFR887,836参数。原PLN fusion_weight的6个闲置参数是原有注册，不属于PEN。本轮未使用任何B0-A/E40锚点。B0-B历史为两条seed0连续随机流：GPU0折1–5，GPU1折6–10。保存新基线的首次更新前模型、双Adam与完整四种RNG；其余组逐折恢复这些未训练锚点。
历史未保留预测向量或完整Python/NumPy RNG，不能宣称历史预测SHA或完整四类历史RNG匹配；逐折实际核验所有历史已有的参数SHA、CPU/CUDA入口SHA、混淆、四指标、best/执行epoch。本轮另存完整RNG与预测向量及SHA。
E03与E05按用户定义完全同构；保留两组正式训练作为重复性核验，不能解释成两个独立改进或组合协同。
E02是仓库非平方Frobenius ALoss的训练诱导子图安全实现；论文式(1)为平方范数，故不声称数值复现论文损失。预测与同类目标均先限制train后加I归一化；原对角约1，加I约2；标签同类对角为1，加I为2。lambda只由首个同一次训练前向的四个p.weight/p.bias梯度计算，冻结不调参。原非优化ALoss继续不进入基础损失。
E03/E05校准仅作用传播图，原四GraphConstructLoss继续使用原始GraphLearn输出；每次forward逐元素验证支持、对角不变，没有mutual/topk/新边。E04输入和输出两端均限制对应模态块，参数形状仍320→320。

## 机制诊断

|组别|模态|邻居均值|分类消息梯度|贡献/输入范数|关闭分支log-prob变化|关闭翻转率|
|---|---|---:|---:|---:|---:|---:|
|E00_B0B_REPLAY|PHENO|186.6588|0|0|0|0|
|E00_B0B_REPLAY|ANAT|329.2041|0.01771266|4.43303e-05|0.00132518|0|
|E00_B0B_REPLAY|FUNC|148.6896|0.01516256|1.580155e-05|0.0003896123|0|
|E00_B0B_REPLAY|CORR|112.9917|0.01271634|1.705884e-05|9.154602e-05|0|
|E01_AGGREGATION_FIX_V2|PHENO|175.0762|0.004720359|2.787186e-05|0.0005778442|0|
|E01_AGGREGATION_FIX_V2|ANAT|318.2882|0.002432996|4.567415e-06|9.220062e-05|0|
|E01_AGGREGATION_FIX_V2|FUNC|160.5033|0.002469831|1.216164e-05|0.0001269725|0|
|E01_AGGREGATION_FIX_V2|CORR|132.3947|0.004268034|5.559833e-06|9.540566e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|PHENO|193.2652|0.006353941|4.899822e-05|0.0008988324|0.001149425|
|E02_SAFE_LABEL_ADJ_SUP_V2|ANAT|314.921|0.004722037|9.719336e-06|5.781378e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|FUNC|150.8181|0.002325226|2.65743e-06|1.64456e-05|0|
|E02_SAFE_LABEL_ADJ_SUP_V2|CORR|85.80712|0.002303517|2.115489e-06|1.792822e-05|0|
|E03_SUPPORT_PRESERVING_CAL_V2|PHENO|177.301|0.02134018|7.204938e-05|0.0004739036|0|
|E03_SUPPORT_PRESERVING_CAL_V2|ANAT|332.9706|0.00776689|2.18313e-05|0.0004830677|0.001149425|
|E03_SUPPORT_PRESERVING_CAL_V2|FUNC|149.1242|0.007835815|2.442027e-05|0.0002646228|0|
|E03_SUPPORT_PRESERVING_CAL_V2|CORR|164.4034|0.01224673|3.5402e-05|0.0009848337|0.001149425|
|E04_MODALITY_LOCAL_PROP_V2|PHENO|201.44|2.083407e-05|7.54589e-09|1.166153e-08|0|
|E04_MODALITY_LOCAL_PROP_V2|ANAT|333.7984|3.339858e-06|3.311837e-09|8.389543e-09|0|
|E04_MODALITY_LOCAL_PROP_V2|FUNC|157.6195|5.424794e-06|6.580259e-09|1.303759e-08|0|
|E04_MODALITY_LOCAL_PROP_V2|CORR|164.0225|0.06367716|0.0001099578|0.004562376|0.002298851|
|E05_FIX_PLUS_SUPPORT_CAL_V2|PHENO|177.301|0.02134018|7.204938e-05|0.0004739036|0|
|E05_FIX_PLUS_SUPPORT_CAL_V2|ANAT|332.9706|0.00776689|2.18313e-05|0.0004830677|0.001149425|
|E05_FIX_PLUS_SUPPORT_CAL_V2|FUNC|149.1242|0.007835815|2.442027e-05|0.0002646228|0|
|E05_FIX_PLUS_SUPPORT_CAL_V2|CORR|164.4034|0.01224673|3.5402e-05|0.0009848337|0.001149425|

## 基线逐折审计

|fold|参数SHA|入口CPU/CUDA RNG|混淆/四指标|best epoch|epochs|
|---|---|---|---|---|---|
|1|MATCH|MATCH|MATCH|157|359|
|2|MATCH|MATCH|MATCH|239|441|
|3|MATCH|MATCH|MATCH|180|382|
|4|MATCH|MATCH|MATCH|213|415|
|5|MATCH|MATCH|MATCH|219|421|
|6|MATCH|MATCH|MATCH|155|357|
|7|MATCH|MATCH|MATCH|187|389|
|8|MATCH|MATCH|MATCH|248|450|
|9|MATCH|MATCH|MATCH|206|408|
|10|MATCH|MATCH|MATCH|266|450|

## 统计边界与结论

历史B0-A（仅次级参照，非本轮锚点）：ACC92.0742、Hard-AUC91.8546、SPE94.6300、SEN89.0793%。若仅超过B0-B而未超过B0-A，只能声称相对B0-B的候选改善。
原val=test严格首次最佳、CV前全队列RFE/标准化、全871节点transductive与重叠训练集保留；不构成无偏独立泛化、统计显著性、医学语义或临床因果证据。本轮不调用GPT/CLIP，不下载模型。
所有支持/对角/局部传播及首步梯度等工程证据见.runtime/preflight.json，逐折图统计与分支关闭响应见diagnostics.csv，监督权重及损失见loss_trace.csv。

科研结论：NO-GO。E03/E05十折最佳模型、预测、epoch与四指标逐值相同。GPU0/GPU1正式折数：{'0': 30, '1': 30}；其中创新折分别25/25，基线各5。总墙钟2358.27秒。

<!-- E43_1_INDEPENDENT_FINAL_AUDIT -->
## 独立读回、单因素配对及交付审计

60/60最佳Checkpoint均已重新前向、重算四指标与混淆并核对预测SHA；全部held-out标签置换后的log-prob逐位不变。十个锚点的双Adam均为空的未训练初态，源/数据/split/RNG绑定通过。根源码、输入及受保护历史产物SHA保持不变。

|相对声明前置组|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|ACC W/T/L|AUC W/T/L|
|---|---:|---:|---:|---:|---|---|
|E02_SAFE_LABEL_ADJ_SUP_V2 - E01_AGGREGATION_FIX_V2|-0.0000|-0.0319|+0.4302|-0.4939|2/6/2|3/3/4|
|E03_SUPPORT_PRESERVING_CAL_V2 - E01_AGGREGATION_FIX_V2|+0.3448|+0.2602|+1.2766|-0.7561|3/7/0|3/4/3|
|E04_MODALITY_LOCAL_PROP_V2 - E01_AGGREGATION_FIX_V2|-0.1149|-0.2104|+1.0731|-1.4939|1/7/2|2/2/6|
|E05_FIX_PLUS_SUPPORT_CAL_V2 - E03_SUPPORT_PRESERVING_CAL_V2|+0.0000|+0.0000|+0.0000|+0.0000|0/10/0|0/10/0|

E03/E05定义重复；十折最佳模型、预测和epoch完全一致，不是两个独立改进。校准相对B0-B提高均值，但仅4/10折严格胜，未达7折门槛；相对历史B0-A的ACC仍低约0.1149pp、Hard-AUC低约0.1250pp。其SEN相对B0-B下降约1.2439pp，SPE增加约1.4940pp，须同时披露。

|固定最佳模型同时关闭四支消息|测试类别翻转/871|平均绝对log-prob变化|
|---|---:|---:|
|E00_B0B_REPLAY|0/871|0.0017995839|
|E01_AGGREGATION_FIX_V2|0/871|0.000812296847|
|E02_SAFE_LABEL_ADJ_SUP_V2|1/871|0.000962398892|
|E03_SUPPORT_PRESERVING_CAL_V2|1/871|0.00210632998|
|E04_MODALITY_LOCAL_PROP_V2|2/871|0.00456238045|
|E05_FIX_PLUS_SUPPORT_CAL_V2|1/871|0.00210632998|

以上只验证固定权重下的依赖，不能替代从头训练的无GFR组，也不能据此声称患者图无用。E01恢复四支分类梯度，但没有分类增益。E04辅助消息相对输入约1e−9量级，CORR约1e−4；本轮未支持限制传播到模态块。消息微弱的原因没有被单独隔离，不归因为某个未经对照的超参数。

E02十折lambda范围0.00378618–0.01027879，首步缩放监督/分类投影梯度比均为0.10。第1折PHENO监督梯度仅4.34e−9；非零不能视作实质监督。首步相对E01的参数差异见`.runtime/supervision_update_audit.json`，CORR最明显；Adam首步对梯度幅值的归一化意味着不能仅凭其他模态首步参数差很小就断言整个训练无监督作用。

正式训练区间（首锚点落盘至最后结果提交，含存储恢复）为1703.67秒；本次总执行墙钟见manifest.total_wall_seconds，起点为存储预检。GPU0/GPU1各30折，其中创新各25折、基线各5折。预检峰值显存0.4684GiB，不冒称它是全部正式worker的全程峰值。

存储初检PASS，后续OSSFS原子替换任务JSON时发生瞬时ENOENT，导致汇总器退出；原worker完成基线，无完成折重训。全部文件字节校验后迁移到/tmp继续，同一代码/训练签名不变。最终模型相等性比较复用单卡runtime，首次双卡可见控制器在报告阶段断言失败；以`CUDA_VISIBLE_DEVICES=1`重启控制器后全部已完成折直接跳过，报告成功；子进程仍按各自物理GPU分配。早期预检的计数/诊断键修正发生在任何正式训练之前。失败记录均保留，不计为额外训练折。

科研状态SCIENTIFIC_COMPLETE，结论NO-GO。正式目标`/data/adodas/SPromptGL-main/experiment/experiment43_1_b0b_gfr_mechanism_retest`；本地保护副本`/tmp/SPromptGL-E43_1-stage/experiment/experiment43_1_b0b_gfr_mechanism_retest`。交付以manifest.delivery与delivery_sync为准；/tmp不具持久存储保障。未提交/推送Git。
