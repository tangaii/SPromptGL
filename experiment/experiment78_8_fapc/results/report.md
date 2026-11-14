# Experiment 78.8 — FAPC final report

STATUS: COMPLETE_AUDITED

最终有效任务30/30、best独立重载30/30、latest状态30/30。正式目录：`/model/spromptgl/experiment/experiment78_8_fapc`；单一/model输出回退，原始数据和历史实验只读。
冻结训练源SHA：`e2f150dca3c63cd377482a71b7bb126e326a8591cc5e70058641781337345441`；RN50缓存SHA：`00fac05f14694d740e9eb1313fd5ceb159d88fac476b8456f8261a04cee5d432`。
并发10 worker；从worker启动到全部退出墙钟55.19分钟；各任务训练耗时合计4.666小时；单任务峰值分配显存最大1.131GiB。该显存值不含其他CUDA进程的上下文开销。

## 修复与证据边界

此前两轮产物均已作废并归档：首轮匿名文本不符合定义；第二轮发现CAM144 Hook只返回辅助梯度而漏加主梯度。最终代码返回g_base+g_aux_safe，未修改任何预注册损失权重、温度、优化器或残差上限。全套预检包含Hook开启/辅助为零时原梯度逐值恢复，以及全参数独立梯度合成对照；旧指标不纳入此报告。

## 方法与来源

唯一基座是E74.4-G2 Positive-DCR，从逐折E61 epoch-10完整模型、三个优化器和全部RNG恢复。CAM144在原分类器前捕获一次；Visual-only与医学/匿名组共享144→32患者投影和32→32疾病方向映射。医学与匿名各十条冻结RN50文本，经共同CPU float64全行空间SVD保留20向量Gram矩阵，补零至32维。无可训练文本映射、无Attention库、无多子原型、无LOO。
E78.7实际bdpprompt.py将冻结RN50向量送入可训练text_projection(1024→32)，L_keep比较校准后Prompt与当前可移动的投影输出，因此并未冻结原始参考方向。E78.8删除该可训练文本投影，固定锚点以buffer进入Checkpoint。E78.7-G1的AUC/SEN改善伴随ACC/SPE下降，本轮以六指标与匿名/Visual-only对照共同判断。
服务器真实E74.4 soft_positive_dcr.py的forward_state调用一次DARG与CAM，原分类项是历史ClsLoss(logp, y, train, weights)，其实现为weighted F.nll_loss；total=cls+graph+0.05pair+0.01adv+0.01preserve。本轮直接复用该out[total]，未自行重写或替换分类损失。原始代码路径与SHA由冻结manifest记录。
每类合法训练支持均值产生有符号疾病方向；G2/G3通过q=normalize(anchor+beta*v_c)校准每条Prompt，beta=0.2*sigmoid(raw)。五条Prompt以温度0.25的LogMeanExp形成ASD/TD辅助Logits，尺度10。G1直接匹配视觉疾病方向。所有组优化L_DARG+0.1L_aux+0.01L_keep，G1的keep为0。最终预测始终使用原DARG MLP。
CAM144上仅投影冲突的辅助梯度，并把其范数限制为主梯度的20%；主任务及原分类器直接梯度保留。原三个优化器后执行新增AdamW(lr=5e-4, weight_decay=0.01)。
论文依据：[SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf)、[KgCoOp](https://arxiv.org/abs/2303.13283)、[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。前两者提供原始Prompt与疾病级监督背景；KgCoOp/PCGrad为冻结参考与冲突投影的设计依据，本方法不是其逐行复现。真实历史损失、数据、warmup和SHA链见design.md、manifest.json及preflight.json。

## 六指标

均值 ± population SD，%；ASD=0、TD=1；SEN为ASD敏感度，SPE为TD特异度。Hard-AUC/BA与Probability ROC-AUC分别报告。

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.8-G1-VisualOnly|92.0742 ± 2.7959|91.9337 ± 2.8149|93.6579 ± 2.7408|90.0793 ± 3.8474|93.7882 ± 3.4135|91.3220 ± 3.0255|
|E78.8-G2-FrozenAnchor-Medical|91.7294 ± 2.4624|91.5324 ± 2.4361|93.3665 ± 2.7895|89.0732 ± 3.5845|93.9917 ± 4.0988|90.8992 ± 2.6252|
|E78.8-G3-FrozenAnchor-Anonymous|91.9606 ± 2.5761|91.8701 ± 2.5191|93.6327 ± 2.5336|90.8171 ± 2.9689|92.9232 ± 4.2493|91.3054 ± 2.6358|

历史只读参考，沿用历史评价协议：

|方法|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|E78.7-G1|91.8443 ± 3.0233|91.8087 ± 2.9799|93.8382 ± 2.9265|91.3232 ± 3.7191|92.2942 ± 4.5324|91.2307 ± 3.1663|

## 主要配对差

差值为百分点；逐折差、W/T/L和描述性95%区间全部见paired_comparisons.csv。

|比较|ACC差 [95%区间]|ACC W/T/L|Hard-AUC差|Prob-AUC差|去Fold 2 ACC差|
|---|---:|---:|---:|---:|---:|
|E78.8-G1-VisualOnly − E74.4-G2|-0.1149 [-0.8349, +0.6050]|1/6/3|-0.0957|+0.1633|+0.0000|
|E78.8-G2-FrozenAnchor-Medical − E74.4-G2|-0.4598 [-1.2541, +0.3346]|2/3/5|-0.4970|-0.1281|-0.2554|
|E78.8-G3-FrozenAnchor-Anonymous − E74.4-G2|-0.2286 [-0.7473, +0.2902]|1/6/3|-0.1592|+0.1381|-0.1263|
|E78.8-G2-FrozenAnchor-Medical − E78.8-G1-VisualOnly|-0.3448 [-0.8998, +0.2101]|1/5/4|-0.4013|-0.2915|-0.2554|
|E78.8-G2-FrozenAnchor-Medical − E78.8-G3-FrozenAnchor-Anonymous|-0.2312 [-0.9854, +0.5231]|3/2/5|-0.3377|-0.2663|-0.1292|
|E78.8-G2-FrozenAnchor-Medical − E78.7-G1|-0.1149 [-1.0994, +0.8695]|3/3/4|-0.2762|-0.4717|+0.1277|

去Fold 2后的9折六指标：

|组别|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.8-G1-VisualOnly|92.2153 ± 2.9131|92.0773 ± 2.9323|93.7840 ± 2.8614|90.3320 ± 3.9760|93.8226 ± 3.5965|91.4688 ± 3.1552|
|E78.8-G2-FrozenAnchor-Medical|91.9598 ± 2.4911|91.7522 ± 2.4720|93.4837 ± 2.9169|89.2141 ± 3.7520|94.2903 ± 4.2161|91.1226 ± 2.6755|
|E78.8-G3-FrozenAnchor-Anonymous|92.0890 ± 2.6849|91.9919 ± 2.6273|93.6676 ± 2.6684|90.8808 ± 3.1230|93.1031 ± 4.4429|91.4234 ± 2.7532|

## 机制与独立审计

- 20条文本秩20；float64 Gram最大误差4.663e-15，float32缓存读回误差5.499e-08。医学/匿名每类五条独立文本、无重复和截断；固定锚点逐Checkpoint逐值一致、梯度为None。
- 全部11327个执行epoch中，冲突410（3.62%），投影410（3.62%），范数缩放6875（60.70%）；平均梯度余弦0.32565，原始辅助/主梯度范数比1.09581，安全比0.18386。
- 10/10 base-off parity与两步模型/四优化器/RNG/梯度统计恢复均通过。共有RNG epoch核对10872，不匹配0；无覆盖的历史epoch不宣称已验证。
- best逐个独立前向读回概率、预测和六指标；latest全部存在且状态、最佳epoch、原三个和新增优化器结构与身份均核对。根源码和历史依赖再次按冻结manifest复核。
- 合法支持为unique(train)−unique(test)，Fold 2两条历史重叠行被排除；各类支持人数、原型均值夹角、差异范数、训练/测试辅助准确率、beta与Prompt位移见mechanism.csv。训练查询自身参与支持均值，因此训练辅助表现不代表独立泛化。

|组别|辅助训练ACC|辅助测试ACC|DARG训练ACC|冲突率|缩放率|安全梯度均值比|
|---|---:|---:|---:|---:|---:|---:|
|E78.8-G1-VisualOnly|95.4713|87.8292|95.7773|5.21%|100.00%|0.20000|
|E78.8-G2-FrozenAnchor-Medical|95.7262|89.2072|96.0453|2.56%|50.63%|0.18470|
|E78.8-G3-FrozenAnchor-Anonymous|96.7850|89.8955|96.4151|3.17%|34.51%|0.16841|

医学G2固定Checkpoint干预（最终DARG不直接读取Prompt，故预期最终预测不变）：

|干预|语义ASD概率平均绝对变化|辅助硬翻转总数|DARG概率平均绝对变化|DARG硬翻转|
|---|---:|---:|---:|---:|
|full|0.000000|0|0.000000|0|
|no_calibration|0.126619|25|0.000000|0|
|anchor_only|0.126619|25|0.000000|0|
|anonymous_text|0.168792|60|0.000000|0|
|class_swap|0.401737|430|0.000000|0|

固定Checkpoint文本更换只验证辅助分支的依赖；最终DARG零硬翻转不能用于否定训练中辅助梯度的作用。医学内容的独立证据以G2与G3独立训练为准；视觉类别监督解释以G2与G1比较为准。

## 科研判断

`NO_GO_FAPC`；`MEDICAL_SEMANTIC_VALUE_NOT_ESTABLISHED`；`GRADIENT_PROTECTION_ACTIVE_BUT_NO_GAIN`；`FROZEN_ANCHOR_EFFECT_UNSUPPORTED`。
G2−基座ACC -0.4598 pp、Hard-AUC -0.4970 pp、Prob-AUC -0.1281 pp；G2−匿名ACC -0.2312 pp；G2−Visual-only ACC -0.3448 pp。
冻结锚点与梯度投影是否在数值上成立，与是否产生独立诊断增益分别判断。保护发生在CAM144梯度边界，不能保证经过上游Jacobian、Adam状态和其他图损失后的全部参数更新必然改善主任务。相对E78.7的变化同时包含固定语义空间和梯度保护，现有三组不能单独估计其中每一项的因果增益。统计区间是十折配对差的描述性t区间，折之间训练集重叠，不能解释为无偏独立样本的临床显著性检验。

## 已知限制

单seed=0，保留val=test、测试ACC选Checkpoint、CV前RFE/预处理、全患者transductive图和历史Fold 2主干训练重叠。新支持隔离未消除整体历史协议的泄漏风险。未进行多seed、外部独立测试或新增超参数搜索；不自动开展E78.9。
