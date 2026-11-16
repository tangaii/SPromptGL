# Experiment 81 — PRCL 结果与独立审计

STATUS: COMPLETE_AUDITED

正式目录：`/model/spromptgl/experiment/experiment81_prcl`；训练任务 `30/30`，最佳/最新独立审计 `30/30`。
正式调度总墙钟（含失败重启）12.37 分钟，其中失败尝试 1.69 分钟；完成折任务时间合计 1.41 小时。
独立审计 RNG 三元组不匹配 0；E61 base-off parity 10/10。
均值±population SD，单位%；ASD=0，TD=1。Hard-AUC采用历史硬预测Balanced Accuracy口径，Prob-AUC为原始分类器输出的TD类概率ROC-AUC。

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E81-G1-Visual-PRCL|91.8443 ± 2.7486|91.6760 ± 2.7403|93.5254 ± 2.9076|89.5732 ± 4.1636|93.7789 ± 4.3793|91.0565 ± 2.9186|
|E81-G2-Medical-PRCL|91.7294 ± 2.8596|91.5090 ± 2.8489|93.4620 ± 2.8721|88.5915 ± 3.7117|94.4265 ± 3.7570|90.8523 ± 3.0955|
|E81-G3-Anonymous-PRCL|91.8443 ± 2.9346|91.6069 ± 2.9758|93.5316 ± 3.0353|88.5793 ± 4.6346|94.6346 ± 3.7826|90.9401 ± 3.2784|
|E74.4-G2-Positive-DCR|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|

## 配对比较

差值为百分点；W/T/L为逐折胜/平/负。95%区间为十折配对t区间，仅作描述性比较。逐折六指标差值和去Fold 2九折结果见 `paired_comparisons.csv`。

|比较|ACC Δ [95% CI]|W/T/L|Hard-AUC Δ|Prob-AUC Δ|SEN Δ|SPE Δ|ASD-F1 Δ|去Fold2 ACC Δ|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E81-G1-Visual-PRCL - E74.4-G2-Positive-DCR|-0.3448 [-0.7420,+0.0524]|0/7/3|-0.3533|+0.0308|-0.4939|-0.2128|-0.4025|-0.2554|
|E81-G2-Medical-PRCL - E74.4-G2-Positive-DCR|-0.4598 [-1.1532,+0.2336]|1/5/4|-0.5204|-0.0326|-1.4756|+0.4348|-0.6068|-0.2554|
|E81-G3-Anonymous-PRCL - E74.4-G2-Positive-DCR|-0.3448 [-1.0218,+0.3321]|1/6/3|-0.4224|+0.0370|-1.4878|+0.6429|-0.5189|-0.2554|
|E81-G2-Medical-PRCL - E81-G1-Visual-PRCL|-0.1149 [-0.5817,+0.3518]|1/7/2|-0.1671|-0.0634|-0.9817|+0.6475|-0.2042|+0.0000|
|E81-G2-Medical-PRCL - E81-G3-Anonymous-PRCL|-0.1149 [-0.5817,+0.3518]|1/7/2|-0.0980|-0.0696|+0.0122|-0.2081|-0.0879|+0.0000|

## 排除 Fold 2

九折均值±population SD（%）：

|组别|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E81-G1-Visual-PRCL|91.9598 ± 2.8742|91.7763 ± 2.8711|93.5483 ± 3.0640|89.4986 ± 4.3825|94.0539 ± 4.5335|91.1468 ± 3.0631|
|E81-G2-Medical-PRCL|91.9598 ± 2.9248|91.7261 ± 2.9235|93.6428 ± 2.9730|88.6789 ± 3.9027|94.7734 ± 3.8053|91.0704 ± 3.1892|
|E81-G3-Anonymous-PRCL|91.9598 ± 3.0717|91.6847 ± 3.1271|93.6023 ± 3.1916|88.1233 ± 4.6677|95.2462 ± 3.4867|90.9917 ± 3.4519|
|E74.4-G2-Positive-DCR|92.2153 ± 2.9131|92.0481 ± 2.8723|93.6142 ± 3.0541|90.0474 ± 4.0959|94.0487 ± 4.9847|91.4703 ± 3.0580|

## 机制与科研 Gate

冻结Checkpoint上的 PRCL-off、均匀关系、text-off、Null-only 和 writeback-off 是推理干预，不是重训消融。G2 的医学Prompt替换、gamma=0、组间交换、非语义条件也只用于解释冻结模型。组内Prompt排列按预期置换对称性检查，未作为语义有效性的证据。详细熵、Null拒绝、关系集中度、患者/Prompt差异、槽内/跨槽质量、残差及干预变化见 `mechanism.csv`。

Gate A 局部关系优于DARG：`False`。
Gate B 医学Prompt组优于视觉关系组：`False`。
Gate C 医学文本优于匿名文本：`False`。
Gate D 完整G2优于DARG：`False`。
Gate E 冻结干预显示关系和Prompt实际参与计算：`True`；这不等于诊断收益。
最终标签：`COMPLETE_AUDITED`, `RELATION_VALUE_NOT_SUPPORTED`, `MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`, `NO_GO_PRCL`。

## 审计与限制

E61 epoch-10 warmup逐折独立加载；E74.4三个历史优化器及Positive-DCR目标保留，PRCL使用独立AdamW。30/30 best概率、硬预测、六指标独立复算，30/30 latest状态完整性检查；10/10 base-off parity；E81组间及E74.4共有epoch RNG三元组不匹配为0。训练和报告均保留val=test、测试ACC选模、CV前预处理/RFE和transductive图。Fold 2行377/407仍在历史DARG训练索引内，PRCL无测试标签输入且残差正则排除这两行。十折训练集合相互重叠，区间不表示独立临床显著性。16个CAM局部是潜在坐标，不是ROI。

预检冻结特征诊断仅使用Fold 1外层训练患者，并在训练患者内固定划分；36维局部关系与36维局部均值、随机坐标分组作容量匹配。局部均值ACC=0.9592，关系ACC=0.9592，随机关系ACC=0.9439；真实槽内减跨槽余弦=-0.008235，随机分组P50=+0.004269。该单次内部分割仅为可行性诊断，不用于模型选择。

正式训练失败尝试记录 1次：首轮30个worker均在优化器step前因梯度日志读取错误退出，完成更新0次、Checkpoint写入0个；逐worker日志保留在 `runtime/logs/`。修复后重新通过预检并完成30/30。

report-only审计修复 1项；训练源哈希保持 `8432c183a9a387a0abe4d505425fa5f1944d762692a2fff6ea407ec0d30386e3`。原因及修复前后SHA见 `results/audit_status.json`。

## 复现文件

`prcl.py`、`run.py`、`audit.py`、`make_text_cache.py`；冻结Prompt缓存 `runtime/text_embeddings.pt`；逐折结果 `results/fold_metrics.csv`；逐指标配对差 `results/paired_comparisons.csv`；机制和干预 `results/mechanism.csv`；独立审计 `results/audit_status.json`。
