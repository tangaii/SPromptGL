# Experiment 78.9 — LOO-FAPC final report

STATUS: COMPLETE_AUDITED

单一正式目录：`/model/spromptgl/experiment/experiment78_9_loo_fapc`。preferred OSSFS mkdir errno=5，回退/model。30/30正式训练，10 worker，墙钟98.92分钟；任务时间合计7.869小时；单任务allocated峰值1.088GiB。
冻结训练源SHA `cbec59531ed579278495ef8ef4cc00d93a38014080d14beb098a9398b4514614`；E78.8最终有效修正版 `corrected-neutral-anonymous-and-main-gradient-v2`。固定文本缓存SHA `00fac05f14694d740e9eb1313fd5ceb159d88fac476b8456f8261a04cee5d432`。

## 方法和来源

唯一基座E74.4-G2，全部任务独立从E61 epoch-10参数、三个优化器和全部RNG恢复。复用服务器真实soft_positive_dcr.py完整损失；原始weighted NLL、图损失与pair/adv/preserve权重保持。仅将训练支持查询的辅助原型改成向量化(sum-self)/(count-1)；ASD/TD两个竞争方向共用同一去除自身的支持集。测试查询始终使用完整合法训练支持。患者查询路径及其他支持患者原型路径保留梯度，自身支持路径梯度抵消。
患者投影、疾病方向映射、beta、固定RN50锚点、辅助权重0.1、保持权重0.01、温度0.25、尺度10、20%梯度上限和私有初始化seed788000+fold均与E78.8相同。正式预测仍为DARG MLP，没有语义残差分类。
SPromptGL提供GFR/CAM及八条Prompt依据；BrainPrompt提供两条ABIDE疾病级原文，TC映射TD。冻结参考借鉴KgCoOp，单侧CAM144保护借鉴PCGrad，但不是其多任务优化器复现。四篇论文和官方源码已读取；BrainPrompt文本文件SHA ce6a7bb9755c26cd9643b6867bac5d1496e93bf9a0d7d8d90944dd3b119c431c，与缓存原文核对。
来源：[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[官方代码](https://github.com/wanxixi11/SPromptGL)；[BrainPrompt论文](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、[官方文本](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt)；[KgCoOp](https://arxiv.org/abs/2303.13283)、[代码](https://github.com/htyao89/KgCoOp)；[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)、[代码](https://github.com/tianheyu927/PCGrad)。

## 六指标

均值 ± population SD，%；ASD=0，TD=1。Hard-AUC/BA和ASD概率ROC-AUC分别计算。

|组别|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|92.0742 ± 2.9342|91.8437 ± 2.9518|93.5288 ± 3.5513|88.8354 ± 4.0532|94.8520 ± 3.6503|91.2108 ± 3.1944|
|E78.9-G2-Medical-LOO|91.9592 ± 2.7747|91.7692 ± 2.7701|93.4980 ± 2.8783|89.3293 ± 4.0216|94.2091 ± 4.1186|91.1474 ± 3.0078|
|E78.9-G3-Anonymous-LOO|91.9579 ± 3.1744|91.7876 ± 3.2335|93.8010 ± 2.6528|89.3476 ± 5.2639|94.2276 ± 3.5720|91.1039 ± 3.6259|

历史同协议参考（只读，不重训）：

|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2-Positive-DCR|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|E78.8-G1-VisualOnly|92.0742 ± 2.7959|91.9337 ± 2.8149|93.6579 ± 2.7408|90.0793 ± 3.8474|93.7882 ± 3.4135|91.3220 ± 3.0255|
|E78.8-G2-FrozenAnchor-Medical|91.7294 ± 2.4624|91.5324 ± 2.4361|93.3665 ± 2.7895|89.0732 ± 3.5845|93.9917 ± 4.0988|90.8992 ± 2.6252|
|E78.8-G3-FrozenAnchor-Anonymous|91.9606 ± 2.5761|91.8701 ± 2.5191|93.6327 ± 2.5336|90.8171 ± 2.9689|92.9232 ± 4.2493|91.3054 ± 2.6358|

## 配对比较

差值为百分点，95%为十折配对差的描述性t区间。全部六指标逐折差与W/T/L见paired_comparisons.csv。

|比较|ACC差[95%区间]|W/T/L|Hard-AUC差|Prob-AUC差|去Fold2 ACC差|
|---|---:|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO - E78.8-G1-VisualOnly|-0.0000 [-0.7752,+0.7752]|3/5/2|-0.0900|-0.1291|-0.1277|
|E78.9-G1-Visual-LOO - E74.4-G2-Positive-DCR|-0.1149 [-0.5817,+0.3518]|1/7/2|-0.1857|+0.0342|-0.1277|
|E78.9-G2-Medical-LOO - E78.8-G2-FrozenAnchor-Medical|+0.2299 [-0.2902,+0.7499]|3/6/1|+0.2367|+0.1315|+0.2554|
|E78.9-G2-Medical-LOO - E74.4-G2-Positive-DCR|-0.2299 [-0.8785,+0.4187]|1/7/2|-0.2602|+0.0034|+0.0000|
|E78.9-G3-Anonymous-LOO - E78.8-G3-FrozenAnchor-Anonymous|-0.0026 [-0.9485,+0.9433]|3/4/3|-0.0826|+0.1682|-0.1306|
|E78.9-G3-Anonymous-LOO - E74.4-G2-Positive-DCR|-0.2312 [-0.7493,+0.2869]|1/6/3|-0.2418|+0.3064|-0.2569|
|E78.9-G2-Medical-LOO - E78.9-G1-Visual-LOO|-0.1149 [-0.9326,+0.7027]|1/7/2|-0.0745|-0.0308|+0.1277|
|E78.9-G2-Medical-LOO - E78.9-G3-Anonymous-LOO|+0.0013 [-0.8644,+0.8671]|2/6/2|-0.0184|-0.3030|+0.2569|

去Fold 2九折结果：

|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|92.0875 ± 3.0926|91.8417 ± 3.1114|93.5816 ± 3.7397|88.6789 ± 4.2436|95.0046 ± 3.8174|91.1944 ± 3.3668|
|E78.9-G2-Medical-LOO|92.2153 ± 2.8105|92.0300 ± 2.8011|93.6652 ± 2.9876|89.7696 ± 4.0039|94.2903 ± 4.3338|91.4296 ± 3.0423|
|E78.9-G3-Anonymous-LOO|91.9584 ± 3.3462|91.7794 ± 3.4083|93.8133 ± 2.7960|89.2480 ± 5.5397|94.3108 ± 3.7560|91.0757 ± 3.8210|

## LOO、泛化差距与梯度机制

|组别|LOO方向平均角度°|P95均值°|辅助训练ACC|辅助测试ACC|差距pp|冲突率|缩放率|安全梯度比|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|0.07908|0.12066|96.0328|89.3156|6.7172|4.12%|100.00%|0.20000|
|E78.9-G2-Medical-LOO|0.04632|0.07963|96.8237|89.2072|7.6165|2.21%|52.46%|0.18536|
|E78.9-G3-Anonymous-LOO|0.05229|0.09405|96.0198|90.0078|6.0120|3.28%|30.59%|0.16641|

与E78.8辅助分类差距比较：

|组别|E78.8训练ACC|E78.9训练ACC|E78.8测试ACC|E78.9测试ACC|
|---|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|95.4713|96.0328|87.8292|89.3156|
|E78.9-G2-Medical-LOO|95.7262|96.8237|89.2072|89.2072|
|E78.9-G3-Anonymous-LOO|96.7850|96.0198|89.8955|90.0078|

在同一E78.9最佳Checkpoint上恢复包含自身的训练原型，仅用于量化LOO的即时作用：

|组别|LOO训练辅助ACC|包含自身训练辅助ACC|辅助Logits MAE|辅助硬翻转总数|
|---|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|96.0328|96.0456|0.0022775|1|
|E78.9-G2-Medical-LOO|96.8237|96.8237|0.0001162|0|
|E78.9-G3-Anonymous-LOO|96.0198|96.0198|0.0001107|0|

合法支持共783–784人，单个患者在对应类均值中的直接权重为0.2370%–0.2762%。LOO方向角度在重新归一化后使用float64计算；自身支持特征梯度为零的结论来自分离路径自动微分测试，不能外推为上游传导图中不存在患者间信息。

|组别|梯度余弦折均值|原始辅助/主梯度范数比|安全范数比|冲突epoch|投影epoch|缩放epoch|
|---|---:|---:|---:|---:|---:|---:|
|E78.9-G1-Visual-LOO|0.178050|1.311606|0.200000|158|158|3833|
|E78.9-G2-Medical-LOO|0.390457|0.252897|0.185360|88|88|2087|
|E78.9-G3-Anonymous-LOO|0.407126|0.207898|0.166411|123|123|1147|

固定G2 Checkpoint干预：

|干预|辅助ASD概率平均绝对变化|辅助硬翻转|最终DARG硬翻转|
|---|---:|---:|---:|
|no_calibration|0.119767|19|0|
|anchor_only|0.119767|19|0|
|anonymous_text|0.171311|41|0|
|class_swap|0.438891|512|0|

文本替换影响辅助分类，但最终DARG在冻结Checkpoint推理时不直接读取文本，因此零最终翻转是预期，不能据此否定训练期辅助梯度作用。医学内容价值以G2/G3独立训练为准。

## 审计和科研判断

30/30 best独立前向概率和六指标一致；30/30 latest全部模型/四优化器/RNG读回逐值一致。E74.4共有epoch 11041，不匹配 0；E78.8共有epoch 10699，不匹配 0。10/10 base-off parity，30/30私有初始化参数buffer匹配，LOO显式参考与自梯度抵消、完整Hook全参数合成、零辅助主梯度逐值恢复和两步恢复均通过。所有冻结输入按manifest复核。
判定：`NO_GO_LOO_FAPC`; `LOO_EFFECT_NOT_SUPPORTED`; `MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`; `GRADIENT_PROTECTION_ACTIVE_BUT_NO_GAIN`。
判定区分：LOO实现生效不等于诊断增益；辅助训练ACC降低不等于辅助测试泛化改善；若仅相对E78.8医学组有局部改善而未超过匿名/Visual-only/基座，仍不能宣称完整方案GO。

## 已知限制

单seed，保留val=test、测试ACC选Checkpoint、CV前预处理/RFE及全患者transductive图。Fold2历史重叠仅从新增支持/监督中排除，原历史路径未变。LOO排除的是投影后该患者自身的直接原型贡献；上游传导图仍可传播患者间信息，不能将此等同整体无泄漏。十折训练集重叠，描述性置信区间不是独立临床显著性结论。不自动开展E78.10。

## 实现纠错记录

预检发现CUDA index_add_聚合导致约1e-6的不确定累加误差，正式训练前改为固定支持顺序的分类归约。该修复未改变LOO公式、参数、初始化、损失或超参数。所有正式任务均使用最终冻结SHA。report_only.py是审计/汇总补充，不修改冻结训练源码或Checkpoint。

## 结果解释

医学LOO组相对E78.8医学组的ACC变化+0.2299pp；相对Visual-only -0.1149pp、匿名文本 +0.0013pp、DARG基座 -0.2299pp。相对基座Hard-AUC变化-0.2602pp，Prob-AUC变化+0.0034pp；去Fold2后的ACC差+0.0000pp。
医学内容的独立增量需同时优于匹配匿名对照和Visual-only；相对低于基座的旧医学方案改善，不能单独构成诊断收益证据。梯度保护的触发统计只说明算法工作，本实验没有独立改变保护规则，不能据此证明20%上限最优。
完整LOO-FAPC不满足预定GO方向性条件，建议停止通过当前疾病原型与Prompt辅助监督改善DARG的路线；本次不自动开展E78.10。

## 复现入口

`python run.py --train --workers 10` 使用冻结训练实现并跳过已完成且SHA匹配的任务；独立审计及报告使用 `python report_only.py`。冻结audit.py保留原字节，最终审计/汇总入口明确为report_only.py。身份清单见manifest.json，逐折独立重载见results/checkpoint_audit.csv，完整预检见results/preflight.json。

## Final integrity supplement cbec59531ed5

全部11560个正式训练epoch的梯度保护统计有限，安全辅助/主梯度范数比最大0.200000048（float32容差内不超过0.20）。六指标、CSV类型/空值/数量、混淆矩阵、best SHA、latest结束epoch均通过。E78.8全部30个历史best SHA仍与原task记录一致。

E74.4真实run.py SHA `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`；soft_positive_dcr.py SHA `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`。E78.8最终有效源集合SHA `e2f150dca3c63cd377482a71b7bb126e326a8591cc5e70058641781337345441`。

磁盘保存/加载后的两步恢复已逐值核对完整DARG log-probabilities、语义Logits/log-probabilities、模型、四优化器、RNG、损失和保护统计。独立审计入口SHA `6b11ef0acf636476181a2099a51af86d3bd354180345878d7d8788279329c1d1`。
