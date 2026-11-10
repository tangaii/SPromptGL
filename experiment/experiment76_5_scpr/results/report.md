# Experiment 76.5: SCPR

审计状态：**COMPLETE_AUDITED**；科学判定：**NO_GO_SCPR_END_TO_END**。形式训练20/20，独立审计30/30（含历史基座10折）。

## 方法与执行

在E74.4-G2唯一基座上，每折从对应E61 epoch-10 warmup恢复GFR/DARG/CAM/OutputLayer及三个历史优化器；没有从E74.4最佳模型微调。新增SCPR以CAM144的4×36来源槽位建立连续可靠、边界和困难混淆记忆，训练期用eval特征以0.9/0.1 EMA更新，训练查询采用固定A/B交叉支持；Fold 2的历史重叠测试患者排除在SCPR memory之外。真实BrainPrompt疾病Prompt与几何匹配匿名向量各训练10折。原分类器结构、DARG/PEN和原损失项保留，SCPR残差零初始化、1.5限幅。

SPromptGL原论文包含GFR多关系图传播、CAM四模态交互、原始8条Prompt及PLN/PEN局部语义加权和RES。E76.5没有重新实现PEN，也没有声称首次组合Prompt与Prototype；BrainPrompt和HP²L已涉及疾病语义与原型。本实验的新增主张限于患者图后CAM来源槽位上的连续混淆证据检索、双疾病竞争和端到端受限决策残差。

原始四个模态路径、DARG图公式、原始8条Prompt、OutputLayer与E74.4-G2损失均沿历史源码只读调用；CAM144由OutputLayer输入hook捕获，每个前向只触发一次。兼容E74.4迁移路径的覆写只存在于E76.5适配器，并校验只读E53 anchor SHA；历史源码未修改。每个新模块参数数为22,106，G1/G2结构和参数量逐折一致。真实RN50文本embedding从已验证冻结缓存读取；G2固定坐标符号/置换对两类向量保持相互内积。E76.5输出目录因`/data` OSSFS新目录写入返回EIO，按指令落在`/model/spromptgl/experiment/experiment76_5_scpr`。
十折均从E61 epoch-10 warmup恢复；每折E74.4准备态的4096关系对SHA与历史任务一致，E61 warmup文件/自有pair SHA也独立校验。共有epoch RNG轨迹逐项匹配7286/7286。G1/G2 Fold 1的latest恢复重放输出与RNG均完全相同。训练高峰期间因初始调度与3个外部worker重叠曾短暂达到13个进程，随后在保存点中断并以7个本实验worker恢复，之后全局上限10；无OOM。

## 十折指标

核心指标单位为百分比，报告mean ± population SD；Hard-AUC沿用历史BA。Prob-AUC以连续ASD概率计算。NLL、Brier、ECE为概率量。

|Method|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|NLL|Brier|ECE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E74.4-G2 historical base|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|0.31068 ± 0.07454|0.08327 ± 0.02244|0.08368 ± 0.02132|
|E76.5-G1 SCPR-Full|91.8443 ± 2.9346|91.6171 ± 2.9010|93.6464 ± 3.3361|88.8171 ± 4.2523|94.4172 ± 4.9607|90.9952 ± 3.1249|0.33577 ± 0.08030|0.09223 ± 0.02356|0.10875 ± 0.05090|
|E76.5-G2 SCPR-Anonymous|91.8443 ± 2.9346|91.5649 ± 2.9867|93.4778 ± 2.8801|87.8476 ± 4.7742|95.2821 ± 3.5809|90.8667 ± 3.2781|0.32106 ± 0.07510|0.08738 ± 0.02315|0.09846 ± 0.02922|

## 配对比较

单位为百分点；W/T/L按逐折差值统计，95%区间为描述性t区间，不把患者数当作折数。Fold 2以外结果仅为敏感性描述。

|Comparison|Scope|Metric|Mean Δ (pp)|W/T/L|Descriptive 95% CI (pp)|
|---|---|---|---:|---:|---:|
|G1-SCPR-Full - E74.4-G2|all10|acc|-0.345|0/8/2|[-0.900, 0.210]|
|G1-SCPR-Full - E74.4-G2|all10|hard_auc|-0.412|0/6/4|[-0.986, 0.161]|
|G1-SCPR-Full - E74.4-G2|all10|prob_auc|+0.152|7/0/3|[-0.447, 0.750]|
|G1-SCPR-Full - E74.4-G2|all10|sen|-1.250|0/6/4|[-2.515, 0.015]|
|G1-SCPR-Full - E74.4-G2|all10|spe|+0.426|3/6/1|[-0.537, 1.388]|
|G1-SCPR-Full - E74.4-G2|all10|asd_f1|-0.464|0/6/4|[-1.098, 0.171]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|acc|-0.383|0/7/2|[-1.008, 0.242]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|hard_auc|-0.458|0/5/4|[-1.101, 0.185]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|prob_auc|+0.063|6/0/3|[-0.580, 0.705]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|sen|-1.389|0/5/4|[-2.785, 0.007]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|spe|+0.473|3/5/1|[-0.617, 1.563]|
|G1-SCPR-Full - E74.4-G2|exclude_fold2|asd_f1|-0.515|0/5/4|[-1.226, 0.195]|
|G2-SCPR-Anonymous - E74.4-G2|all10|acc|-0.345|0/8/2|[-0.900, 0.210]|
|G2-SCPR-Anonymous - E74.4-G2|all10|hard_auc|-0.465|0/5/5|[-1.053, 0.124]|
|G2-SCPR-Anonymous - E74.4-G2|all10|prob_auc|-0.017|5/0/5|[-0.416, 0.383]|
|G2-SCPR-Anonymous - E74.4-G2|all10|sen|-2.220|0/5/5|[-4.150, -0.289]|
|G2-SCPR-Anonymous - E74.4-G2|all10|spe|+1.290|4/6/0|[-0.207, 2.788]|
|G2-SCPR-Anonymous - E74.4-G2|all10|asd_f1|-0.592|0/5/5|[-1.271, 0.086]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|acc|-0.128|0/8/1|[-0.422, 0.167]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|hard_auc|-0.245|0/5/4|[-0.605, 0.114]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|prob_auc|-0.007|5/0/4|[-0.461, 0.448]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|sen|-1.924|0/5/4|[-3.989, 0.141]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|spe|+1.434|4/5/0|[-0.233, 3.101]|
|G2-SCPR-Anonymous - E74.4-G2|exclude_fold2|asd_f1|-0.353|0/5/4|[-0.817, 0.112]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|acc|+0.000|1/7/2|[-0.671, 0.671]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|hard_auc|+0.052|2/5/3|[-0.641, 0.745]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|prob_auc|+0.169|7/0/3|[-0.318, 0.655]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|sen|+0.970|3/5/2|[-1.251, 3.190]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|spe|-0.865|1/7/2|[-2.676, 0.946]|
|G1-SCPR-Full - G2-SCPR-Anonymous|all10|asd_f1|+0.128|2/5/3|[-0.660, 0.917]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|acc|-0.255|0/7/2|[-0.645, 0.134]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|hard_auc|-0.213|1/5/3|[-0.608, 0.183]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|prob_auc|+0.069|6/0/3|[-0.423, 0.562]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|sen|+0.535|2/5/2|[-1.734, 2.804]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|spe|-0.961|1/6/2|[-3.010, 1.088]|
|G1-SCPR-Full - G2-SCPR-Anonymous|exclude_fold2|asd_f1|-0.163|1/5/3|[-0.657, 0.331]|
|G1-Full - G1-Residual-Off|all10|acc|+0.805|4/6/0|[-0.066, 1.676]|
|G1-Full - G1-Residual-Off|all10|hard_auc|+0.821|4/6/0|[-0.072, 1.715]|
|G1-Full - G1-Residual-Off|all10|prob_auc|+0.032|5/3/2|[-0.016, 0.080]|
|G1-Full - G1-Residual-Off|all10|sen|+1.000|3/7/0|[-0.250, 2.250]|
|G1-Full - G1-Residual-Off|all10|spe|+0.643|3/7/0|[-0.098, 1.384]|
|G1-Full - G1-Residual-Off|all10|asd_f1|+0.913|4/6/0|[-0.098, 1.925]|
|G1-Full - G1-Residual-Off|exclude_fold2|acc|+0.766|3/6/0|[-0.222, 1.754]|
|G1-Full - G1-Residual-Off|exclude_fold2|hard_auc|+0.792|3/6/0|[-0.224, 1.808]|
|G1-Full - G1-Residual-Off|exclude_fold2|prob_auc|+0.035|5/2/2|[-0.019, 0.090]|
|G1-Full - G1-Residual-Off|exclude_fold2|sen|+1.111|3/6/0|[-0.285, 2.507]|
|G1-Full - G1-Residual-Off|exclude_fold2|spe|+0.473|2/7/0|[-0.248, 1.194]|
|G1-Full - G1-Residual-Off|exclude_fold2|asd_f1|+0.891|3/6/0|[-0.260, 2.043]|

## 纠错与残差

G1相对历史基座：错误→正确 5，正确→错误 8，净纠错 -3；预测改变 13。按真实类别方向，ASD→TD纠正 4，TD→ASD纠正 1。
固定同一G1最佳模型关闭推理残差后：G1-Full相对Residual-Off错误→正确 7，正确→错误 0，净纠正 7；这是推理残差直接干预，不是未训练SCPR的基座。Checkpoints未被修改。

## 原型和模态机制

可靠/边界/困难候选规模、站点覆盖与Top-K回退、检索支持分布、双疾病证据、门控/残差分布均记录于`mechanism.csv`。测试真实标签只用于下列分组诊断和指标，不进入prototype memory或gate input。模态权重表示CAM来源槽位的模型权重，不能据此宣称具体脑区生物标志物。

|Group|Gate mean (fold mean)|Gate SD|Mean abs delta|Coverage|PHENO|ANAT|FUNC|CORR|Same-site support|Fallback over cap|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|G1-SCPR-Full|0.9132|0.0253|0.1460|1.0000|0.0755|0.0791|0.0658|0.7796|0.1264|0.0000|
|G2-SCPR-Anonymous|0.8548|0.0373|0.1088|1.0000|0.0821|0.0809|0.0699|0.7671|0.1273|0.0000|

原型角色（10折平均训练候选数）：

|Group|Reliable ASD|Reliable TD|Hard ASD|Hard TD|Hard memory enabled|
|---|---:|---:|---:|---:|---:|
|G1-SCPR-Full|352.2|411.1|10.5|10.1|100.0%|
|G2-SCPR-Anonymous|350.0|407.9|12.7|13.3|100.0%|
双候选槽位证据差为e_ASD−e_TD；正值偏向ASD候选，负值偏向TD候选。ρ为冻结文本向量映射后的可学习注入系数。

|Group|Mean e_ASD-e_TD PHENO|ANAT|FUNC|CORR|rho PHENO|ANAT|FUNC|CORR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G1-SCPR-Full|-0.0190|-0.0536|-0.0266|+0.0279|0.0200|0.0206|0.0205|0.0209|
|G2-SCPR-Anonymous|-0.0388|-0.0445|-0.0367|+0.0112|0.0203|0.0205|0.0207|0.0207|

## 结论和限制

判定依据：G1−基座ACC -0.345 pp（W/T/L 0/8/2），Hard-AUC -0.412 pp，Prob-AUC +0.152 pp；排除Fold 2后ACC差 -0.383 pp。G1−匿名组ACC +0.000 pp（W/T/L 1/7/2）。SCPR门控折均值 0.8840，跨折门控标准差均值 0.0313，折均绝对残差均值 0.1274。

审计通过只说明实现和产物按声明可复现，不等于科学方法GO。历史协议仍使用val=test选择checkpoint、CV前RFE/标准化和transductive图；Fold 2中历史基座训练/测试重叠行377与407。九折比较不能消除其他偏差。因此这些是协议匹配的探索性比较，不是严格无偏的独立泛化结果。

训练配置SHA：`15568aa8c459447d8fe4774405cd30a4c102c684ac57c7592eae87a2880cab67`；审计后只将配置status字段更新为`COMPLETE_AUDITED`，最终SHA为`0256460e92918e0fc07fe8563d1fdc998ca3065fb089591cc2dd49fee53e557c`。代码和历史来源SHA锁定未变。本次独立审计源码SHA：`af882907538bd6cb24941fa8c7e766b637cfb55f8dc436f9e6b7e4137f97544b`。逐折结果：`fold_metrics.csv`；配对：`paired_comparisons.csv`；机制：`mechanism.csv`；纠错：`correction_analysis.csv`。
