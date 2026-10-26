

## Experiment 40 — HCER-v2


科研状态：**COMPLETE**；锚点：**ANCHOR_REPLAY_PASS**；科研判定：**NO-GO-CLASSIFICATION / NO-GO-SEMANTICS / NO-GO-EVIDENCE-SPECIFICITY**。
正式折数 100/100；GPU0=50，GPU1=50。交付：PARTIAL_TMP_ONLY。
正式墙钟秒：1696.6874105930328；总执行墙钟秒：2728.39。

## 四项正式指标

十折mean±population SD，单位%；仅完整十折组显示均值。不是全队列micro均值。

|组别|说明|ACC|Hard-AUC|SPE-TD|SEN-ASD|ΔACC vs B0|
|---|---|---:|---:|---:|---:|---:|
|B0|B0A_ANCHOR|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|0.0000|
|E1|Z_TOPK_FULL|91.7294 ± 2.9950|91.5758 ± 2.9531|93.5661 ± 4.1059|89.5854 ± 3.2768|-0.3448|
|E2|ANONYMOUS_FULL|88.0865 ± 8.3045|87.9569 ± 8.1011|89.5236 ± 11.2868|86.3902 ± 5.9814|-3.9877|
|E3|REAL_NO_CF|82.4190 ± 14.7069|81.8434 ± 15.3405|89.2599 ± 11.9230|74.4268 ± 25.6341|-9.6552|
|E4|REAL_KEEP_ONLY|85.2926 ± 12.4563|84.7548 ± 13.2601|91.6559 ± 8.9107|77.8537 ± 24.8420|-6.7816|
|E5|REAL_FULL|82.8788 ± 14.1519|82.1048 ± 14.9529|92.5023 ± 8.3901|71.7073 ± 27.0535|-9.1954|
|E6|ANATOMY_NEUTRAL_FULL|78.0878 ± 13.9343|77.2811 ± 14.3596|88.8853 ± 9.4876|65.6768 ± 22.0198|-13.9864|
|E7|WRONG_PAIR_FULL|81.7385 ± 14.7831|80.8543 ± 15.8096|92.9463 ± 4.7751|68.7622 ± 29.9727|-10.3357|
|E8|SWAPPED_CLASS_FULL|87.1317 ± 10.6609|86.7958 ± 10.9513|91.6281 ± 6.9235|81.9634 ± 16.8416|-4.9425|
|E9|NO_RIVAL_NORMMATCH_FULL|87.7129 ± 8.9954|87.5672 ± 8.5493|89.7502 ± 14.6085|85.3841 ± 5.1964|-4.3613|

## B0-A逐折回放

|fold|本次TP/FN/FP/TN|历史TP/FN/FP/TN|best epoch 本次/历史|执行epoch 本次/历史|差异|
|---|---|---|---|---|---|
|1|37/4/0/47|37/4/0/47|322/322|450/450|EXACT，包括初始参数和入口RNG SHA|
|2|35/6/4/42|35/6/4/42|149/149|351/351|EXACT，包括初始参数和入口RNG SHA|
|3|38/3/7/39|38/3/7/39|428/428|450/450|EXACT，包括初始参数和入口RNG SHA|
|4|36/4/1/46|36/4/1/46|243/243|445/445|EXACT，包括初始参数和入口RNG SHA|
|5|37/3/2/45|37/3/2/45|428/428|450/450|EXACT，包括初始参数和入口RNG SHA|
|6|38/2/2/45|38/2/2/45|124/124|326/326|EXACT，包括初始参数和入口RNG SHA|
|7|33/7/1/46|33/7/1/46|154/154|356/356|EXACT，包括初始参数和入口RNG SHA|
|8|34/6/3/44|34/6/3/44|280/280|450/450|EXACT，包括初始参数和入口RNG SHA|
|9|35/5/3/44|35/5/3/44|169/169|371/371|EXACT，包括初始参数和入口RNG SHA|
|10|36/4/2/45|36/4/2/45|251/251|450/450|EXACT，包括初始参数和入口RNG SHA|

## 逐fold配对

百分点差；W/T/L来自相同fold，训练集重叠，不是10个独立外部数据集。完整逐fold差保存于manifest配对字段。

|比较|ΔACC|ACC W/T/L|ΔHard-AUC|AUC W/T/L|ΔSPE|ΔSEN|净正确数|FN变化|FP变化|
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
|E5-B0|-9.1954|1/4/5|-9.7498|3/2/5|-2.1277|-17.3720|-80|70|10|
|E1-B0|-0.3448|3/3/4|-0.2789|3/3/4|-1.0638|+0.5061|-3|-2|5|
|E5-E1|-8.8506|2/3/5|-9.4709|3/0/7|-1.0638|-17.8780|-77|72|5|
|E2-B0|-3.9877|1/3/6|-3.8977|3/1/6|-5.1064|-2.6890|-35|11|24|
|E5-E2|-5.2077|5/2/3|-5.8521|5/0/5|+2.9787|-14.6829|-45|59|-14|
|E3-B0|-9.6552|1/3/6|-10.0112|1/3/6|-5.3700|-14.6524|-84|59|25|
|E5-E3|+0.4598|3/5/2|+0.2614|4/2/4|+3.2424|-2.7195|4|11|-15|
|E4-B0|-6.7816|1/3/6|-7.0999|2/2/6|-2.9741|-11.2256|-59|45|14|
|E5-E4|-2.4138|2/4/4|-2.6500|3/2/5|+0.8464|-6.1463|-21|25|-4|
|E6-B0|-13.9864|0/3/7|-14.5736|0/2/8|-5.7447|-23.4024|-122|95|27|
|E5-E6|+4.7910|5/4/1|+4.8238|7/1/2|+3.6170|+6.0305|42|-25|-17|
|E7-B0|-10.3357|1/3/6|-11.0003|1/3/6|-1.6836|-20.3171|-90|82|8|
|E5-E7|+1.1403|5/1/4|+1.2505|5/1/4|-0.4440|+2.9451|10|-12|2|
|E8-B0|-4.9425|2/2/6|-5.0589|2/2/6|-3.0019|-7.1159|-43|29|14|
|E5-E8|-4.2529|6/2/2|-4.6910|7/0/3|+0.8742|-10.2561|-37|41|-4|
|E9-B0|-4.3613|1/3/6|-4.2874|1/2/7|-4.8797|-3.6951|-38|15|23|
|E5-E9|-4.8341|5/2/3|-5.4624|5/1/4|+2.7521|-13.6768|-42|55|-13|

## 实现与机制

修复矩阵、参考、文本Query/解剖Key、NULL阈值、同幅残差与CF梯度策略见README/config。所有x/z/mu/sigma/反事实列通过同一canonical_edges；E7只改query membership。B0保留未调用PEN构造以复演E32初始化，PEN无前向。

预检：PASS；详细逐值、梯度、恢复与计时证据保存在`.runtime/preflight.json`。
- E5 residual_mean: fold平均 0.00940590，范围 [0.00754596, 0.01130409]。
- E5 residual_median: fold平均 0.00940590，范围 [0.00754597, 0.01130409]。
- E5 residual_p95: fold平均 0.00940590，范围 [0.00754597, 0.01130409]。
- E5 residual_max: fold平均 0.00940590，范围 [0.00754597, 0.01130409]。
- E5 null_mean: fold平均 0.05776139，范围 [0.03406022, 0.08639466]。
- E5 selection_mean: fold平均 7.99195404，范围 [7.94252872, 8.00000000]。
- E5 empty_fraction: fold平均 0.00000000，范围 [0.00000000, 0.00000000]。
- E5 swap_abs_logp: fold平均 0.00163430，范围 [0.00000229, 0.00398324]。
- E5 swap_class_flip: fold平均 0.00114943，范围 [0.00000000, 0.01149425]。
- E5 disabled_abs_logp: fold平均 0.00511654，范围 [0.00002772, 0.01215074]。
- E5 disabled_class_flip: fold平均 0.00344828，范围 [0.00000000, 0.02298851]。
- E5 selected_delete_logp_drop: fold平均 0.01897011，范围 [-0.00159992, 0.05166243]。
- E5 selected_keep_logp_drop: fold平均 0.08996800，范围 [-0.00389545, 0.19603516]。
- E5 random_delete_logp_drop: fold平均 0.01591534，范围 [-0.00107982, 0.03409335]。
- E5 topz_delete_logp_drop: fold平均 0.06718485，范围 [-0.00200191, 0.14180612]。
- E5 frozen_selected_delete_logp_drop: fold平均 0.01911067，范围 [-0.00020466, 0.05167441]。
- E5 frozen_topz_delete_logp_drop: fold平均 0.06725948，范围 [-0.00129356, 0.14180595]。
- E5 fixed_residual_selected_delete_logp_drop: fold平均 0.01878109，范围 [-0.00136418, 0.05192825]。
- E5 probes: fold平均 153.80000000，范围 [80.00000000, 180.00000000]。
- E5 unique_probe_count: fold平均 153.80000000，范围 [80.00000000, 180.00000000]。
- E5 raw_pair_cos_mean: fold平均 0.91174102，范围 [0.91174102, 0.91174102]。
- E5 projected_pair_cos_mean: fold平均 0.71348624，范围 [0.35530937, 0.87546599]。

E5/E9实际残差比较：{"E5": {"residual_mean": 0.009405895741656423, "residual_median": 0.00940589620731771, "residual_p95": 0.009405897138640285}, "E9": {"residual_mean": 0.010392445651814342, "residual_median": 0.010392445418983699, "residual_p95": 0.010392446434125305}}

## 结论与限制

预注册筛选条件：`{"classification": false, "paired_consistency": false, "semantics": false, "fidelity_vs_random_delete": false, "fidelity_vs_topz_delete": false}`。
正确医学文本只有同时胜过数值Top-|z|、Gram匹配匿名、中性解剖、错误配对及类别交换，且selected删边稳定胜过同预算Top-|z|时，才支持继续研究；内部权重变化或梯度非零本身不证明语义有效。
val=test选checkpoint、全队列预处理/RFE、transductive图、单seed与重叠训练fold均保留；指标只用于历史协议内部比较。FC遮蔽反映模型依赖，可能受分布偏移及图重构影响，不是神经病理因果。

## 来源与交付

[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[ERASER](https://aclanthology.org/2020.acl-main.408/)、[GraphFramEx](https://proceedings.mlr.press/v198/amara22a.html)、[ROAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html)。实际查阅路径与SHA完整记录在manifest/source_audit；未依赖实验39。

真实暂存路径：`/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence`。正式目标：`/data/adodas/SPromptGL-main/experiment/experiment40_hcer_v2_b0a_evidence`。/tmp为临时存储。

故障：{"phase": "post-training report", "resolved": true, "error": "TypeError('Object of type bool_ is not JSON serializable')", "full_trace": ".runtime/failure.json"}。交付细节：{"status": "PARTIAL_TMP_ONLY", "path": "/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence", "target": "/data/adodas/SPromptGL-main/experiment/experiment40_hcer_v2_b0a_evidence", "error": "OSError(5, 'Input/output error')"}；handoff：{"status": "FAILED_PENDING", "error": "OSError(5, 'Input/output error')", "before": {"bytes": 305210, "sha256": "8039c875fbb00794fef975489a4f80a1682c2b9c4b641a65434214368ffce8a6"}, "historical_prefix_preserved": true, "pending": "/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence/handoff_append_pending.md"}。

## 固定最佳模型的补充审计与科学解释

以下内容只读取已完成的最佳Checkpoint；没有优化器更新、没有重新选epoch、没有新增正式分类指标。audit.py及三个10行mechanism_audit CSV保留复算证据。

### 100折完整性与报告修复

10/10组、100/100唯一(group, seed=0, fold)全部完成，GPU0/GPU1各50折；100个best文件SHA、100折共同初始化/RNG、混淆计数重算和保护文件SHA全部通过。B0十折初始哈希、入口哈希、混淆、指标、best epoch及执行epoch与E32逐折EXACT。未重训实验39或改动任何根模型/历史数据。
训练结束后的报告将NumPy bool_交给JSON导致一次序列化异常；训练和Checkpoint早已完成。本次只在独立audit.py报告适配器中把NumPy标量转换为Python标量，冻结run.py/method.py/config完全未变，重新训练次数0。原始错误完整保留于.runtime/failure.json。

### 四层结论

1. 实现层：索引、参考、空mask旁路、梯度、两Adam及恢复预检通过。所有受试者均走统一的OOF训练参考算法，outer-test标签不进入参考拟合。
2. 响应层：医学文本与证据分支有非零log-prob响应，且CF mask梯度存在；这不等于其提供独立医学信息。
3. 分类层：E5明显低于B0、Top-|z|和Gram匹配匿名控制；不能宣称疾病语义收益。
4. 证据层：选中删除未稳定超过随机，也明显未超过Top-|z|；没有证明选边优于简单数值偏离。

### 优化稳定性及对照边界

|组|新增登记参数|疾病Query是否参与选边|实际残差均值%|中位数%|p95%|空mask比例|
|---|---:|---|---:|---:|---:|---:|
|E1|90499|否；数值selector|1.02752|1.02752|1.02752|0.000000|
|E2|90499|是|1.04866|1.04866|1.04866|0.000000|
|E3|90499|是|0.97395|0.97395|0.97395|0.000000|
|E4|90499|是|1.08370|1.08370|1.08370|0.000000|
|E5|90499|是|0.94059|0.94059|0.94059|0.000000|
|E6|90499|是|0.85518|0.83170|0.91028|0.070115|
|E7|90499|是|0.93187|0.93187|0.93187|0.000000|
|E8|90499|是|1.01346|0.90777|1.22484|0.066667|
|E9|90499|是|1.03924|1.03924|1.03924|0.000000|

E5/E9训练后逐折残差均值比范围0.5484–1.3748，没有E38的数量级剂量差；各自学到的gate仍不同，不能当成完全相同干预强度。预检的同gate逐患者校准相同不表示训练后gate相同。E5相对E9的ΔACC=-4.8341pp、ΔAUC=-5.4624pp，不能宣称竞争优于一般异常注意。
E1有相同90499个登记参数，但疾病Query不进入selector，部分Query/温度路径不生效；其数值value仍保留共同解剖信息与冻结软方向先验。因此它是同源数值选择控制，不是完全相同有效容量或完全没有任何医学结构的模型。E7的假说关联及其软方向规则一起错配，raw/z/mu/sigma/物理列从不重排，检验的是假说整体配对。E8仅交换两个Query通道，标签和软方向不交换；E9保留数值软先验但取消类别竞争。匿名控制保持疾病bank全Gram矩阵，但相对真实解剖bank的几何按设计改变。
E3无CF仍有明显折间不稳定：第3/6/9折ACC分别58.6207/66.6667/56.3218%；E5相应为63.2184/65.5172/56.3218%。E5第9折best epoch=1，ASD TP=3/FN=37。由此不能把失败全部归因于删除损失。小前向残差不保证小反向梯度；单位范数、硬选择的STE及共享投影可能带来优化敏感性，这是待另行预注册检验的解释，而非本轮证明的因果。

### 梯度路径分解

在线cf_query_gradient_max在分类backward之后记录，包含累加分类梯度，不能作为纯CF范数。下表用每折固定best模型的同一训练输入独立backward，没有step；CF选择至多2名原预测正确且非空的训练者。比例为各折比值的平均，非两个平均值之比。

|组|分类投影梯度均值|纯CF投影梯度均值|CF/分类投影梯度比均值|纯CF fuse梯度|关闭QC的绝对logp变化|关闭QC类别翻转率|
|---|---:|---:|---:|---:|---:|---:|
|E1|0.00022930313|0|0.00000|0|0.00006539|0.000000|
|E5|0.001808434|0.0039329798|5.36001|0|0.00031755|0.000000|
|E9|0.0011507352|0.014725543|38.36432|0|0.00063112|0.000000|

CF不更新主干或fuse分类残差参数，实际mask路径梯度有限非零；E1的疾病投影CF梯度为0符合其数值selector设计。E5部分折CF投影梯度可大于当时分类投影梯度，但无CF组同样失败，不能据此单独确定失败原因。

### 选边与反事实分解

|机制量|E5十折平均|
|---|---:|
|raw_pair_cos_mean|0.91174102|
|projected_pair_cos_mean|0.71348624|
|query_pair_distance|0.71601067|
|null_mean|0.05776139|
|selection_mean|7.99195404|
|empty_fraction|0.00000000|
|selection_equals_topz_fraction|0.00000000|
|selected_delete_logp_drop|0.01897011|
|random_delete_logp_drop|0.01591534|
|topz_delete_logp_drop|0.06718485|
|selected_keep_logp_drop|0.08996800|
|frozen_selected_delete_logp_drop|0.01911067|
|frozen_topz_delete_logp_drop|0.06725948|
|fixed_residual_selected_delete_logp_drop|0.01878109|
|swap_abs_logp|0.00163430|
|swap_class_flip|0.00114943|
|disabled_abs_logp|0.00511654|
|disabled_class_flip|0.00344828|

Selected-delete减random_delete：均值+0.00305476，逐折胜/平/负=4/0/6。

Selected-delete减topz_delete：均值-0.04821474，逐折胜/平/负=1/0/9。

E5几乎选满8条、空mask比例0，说明可弃权实现可用但训练后没有实际学会广泛弃权；初始化分位数阈值不自动保证训练后NULL校准。选边与Top-|z|并不相同，却没有更好的诊断依赖效应。真实/交换文本改变内部证据和少量logit，类别翻转率很低；不能把非零梯度或文本敏感度替代分类增益。
固定邻接时选中删除仍弱于Top-|z|，因此该劣势不能仅由图重连解释；固定未干预残差时selected-delete结果也相近。固定图/残差是同模型干预，不是额外训练组。keep只在39候选中保留所选边；其余217条FC及其它模态始终保留，不能把它解释为仅8条FC足以诊断。

### CF覆盖、质量与站点

|组|总probes|每折unique范围|非空probes|允许delete的probes|
|---|---:|---|---:|---:|
|E1|1478|134–178|1478|1288|
|E2|1622|106–180|1620|1301|
|E3|0|0–0|0|0|
|E4|1624|80–180|1624|0|
|E5|1538|80–180|1500|1142|
|E6|1500|100–180|1487|1065|
|E7|1456|84–180|1451|1112|
|E8|1592|94–180|1488|1230|
|E9|1612|132–180|1610|1233|

质量层按每fold训练者reliability中位数固定分组；以下是实际人数加权的机制量，不是第五项分类指标，也不用于模型选择：

```json
{
  "lower_reliability": {
    "n": 435,
    "selected_delete_logp_drop": 0.015743572697385973
  },
  "higher_reliability": {
    "n": 436,
    "selected_delete_logp_drop": 0.02218492374405128
  }
}
```

四图非自环边权中站点内比例的fold平均（PHENO/ANAT/FUNC/CORR）：[0.20527519062161445, 0.2348642721772194, 0.3496726393699646, 0.11540865004062653]。各站点样本量及selected-delete效应完整保存在mechanism_audit_E5.csv；小站点和单seed不足以证明无站点混杂。关闭QC没有测试类别翻转，不能宣称可靠性调节已产生疾病判别收益。

### 执行资源与可恢复性

正式100折墙钟1696.69秒（28.28分钟），包括anchor与正式最佳Checkpoint机制诊断；B0单进程连续10折耗时359.95秒。额外审计没有训练更新。预检两患者CF耗时0.453857秒、单进程峰值分配513740288字节；8 worker并发时工具观测每卡约3.7GiB分配，未声称该观测是全训练严格峰值。实际卡名为PPU-ZW810E、每卡98304MiB，环境提供CUDA兼容PyTorch接口。

|组|物理GPU|folds|任务墙钟秒（并行，不可相加当总耗时）|恢复次数|
|---|---:|---:|---:|---:|
|B0|0|10|359.949|0|
|E1|1|10|667.203|0|
|E2|1|10|674.622|0|
|E3|0|10|547.293|0|
|E4|1|10|644.353|0|
|E5|0|10|761.624|0|
|E6|1|10|631.154|0|
|E7|0|10|643.210|0|
|E8|1|10|662.098|0|
|E9|0|10|636.019|0|

所有任务无需故障重启；预检已验证中断后下一步模型/双Adam/RNG逐值复演。最新恢复Checkpoint每10epoch覆盖，完成折精确跳过；必要best、入口anchor、原型/参考缓存及原源码隔离副本保留在.runtime。没有为追分添加seed、修改超参、重写Prompt或挑选成功折。

### 研究判断

最终：**NO-GO-CLASSIFICATION / NO-GO-SEMANTICS / NO-GO-EVIDENCE-SPECIFICITY**。不建议把本方案作为已成立的MICCAI疾病语义创新。后续若研究单位范数校准的梯度放大、弃权阈值漂移或选择器稳定性，需单独预注册，不能把事后修补并入本轮。当前结果也不能证明特定连接没有生物学意义；它只否定这套实现与冻结协议下的分类/证据主张。

## 最终交付状态

{
  "delivery": {
    "status": "PARTIAL_TMP_ONLY",
    "path": "/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence",
    "target": "/data/adodas/SPromptGL-main/experiment/experiment40_hcer_v2_b0a_evidence",
    "error": "OSError(5, 'Input/output error')"
  },
  "handoff": {
    "status": "FAILED_PENDING",
    "error": "OSError(5, 'Input/output error')",
    "before": {
      "bytes": 305210,
      "sha256": "8039c875fbb00794fef975489a4f80a1682c2b9c4b641a65434214368ffce8a6"
    },
    "historical_prefix_preserved": true,
    "pending": "/tmp/SPromptGL-E40-HCER-v2-stage/experiment/experiment40_hcer_v2_b0a_evidence/handoff_append_pending.md"
  }
}

最终验收时间：2026-10-26T06:09:01.144703+00:00；全流程墙钟2728.39秒（45.47分钟）。
