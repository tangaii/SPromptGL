# Experiment 79 DEML：Dynamic Error Memory Learning

## 结论

- **STATUS**：`COMPLETE_AUDITED`。
- **科研判定**：`NO_GO_DEML`。
- 20/20正式折完成，20/20最佳Checkpoint独立重放通过，10/10 latest恢复通过。
- G1和G2在同一外层fold中共享完全相同的原DARG主干轨迹、错误事件与记忆写入/淘汰序列。两组相对各自同Checkpoint的Memory-Off都有净纠错+2，但相对E74.4-G2历史最佳Checkpoint，ACC均低0.2299个百分点、Hard-AUC均低0.2895个百分点；G2的修正方向没有带来额外硬预测收益，因此当前不继续把DEML作为MICCAI 2027第二创新模块候选。

## 论文、源码与基座审查

本轮审查了[SAM 2论文](https://arxiv.org/abs/2408.00714)及官方[Memory Attention](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_attention.py)、[sam2_base.py](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py)、[Memory Encoder](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_encoder.py)和[视频预测器](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py)。E79只借鉴历史记忆写入、有限库检索和Cross-Attention的抽象；没有移植视频帧位置传播、视觉编码器或SAM 2的对象跟踪假设。

同时核对了[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[原作者代码](https://github.com/wanxixi11/SPromptGL)和[当前仓库](https://github.com/tangaii/SPromptGL)。E74.4-G2实际是`PLNNoPEN`，CAM144是原OutputLayer输入；原OutputLayer为LayerNorm→Linear(144→36)→GELU→Dropout→Linear(36→2)。E79未引入原始PEN、疾病文本Prompt或E76.9风险门控。

历史E74.4-G2只读基座从审计结果复核为：ACC `92.1891±2.7648`，Hard-AUC/BA `92.0294±2.7255`，Prob-AUC `93.4946±2.9195`，SEN-ASD `90.0671±3.8862`，SPE-TD `93.9917±4.7320`，ASD-F1 `91.4590±2.9013`（%）。

## 实际路径、资源与训练完成度

首选目录`/data/adodas/SPromptGL-main/experiment/experiment79_deml`在目录创建、写入、fsync、原子重命名读回探针阶段触发OSSFS `EIO(errno=5)`，因此全部正式产物使用回退目录`/model/spromptgl/experiment/experiment79_deml`。服务器GPU为`PPU-ZW810E`（98,304 MiB）。正式任务最多10个DEML worker；为资源和恢复控制，后续阶段以5 worker批次完成。训练日志时间窗约`3小时02分14秒`；任务文件记录的10个fold累计elapsed为`35,852.91秒`，该累计值包含恢复段且不等于并行墙钟。日志记录的单worker峰值为`1.60 GiB`，10并发时约`16 GiB`。外部E78.4进程不计入E79结果。

所有fold任务状态为`COMPLETE`：fold执行轮数分别为`314, 362, 374, 407, 440, 440, 339, 440, 377, 424`；每折均保存G1/G2 best，另有10个共享 latest。

## DEML实现与合法性

每个fold只用合法外层训练患者构建`ASD→TD`和`TD→ASD`错误记忆，容量64/类、总容量128、Top-8、站点最多2条、固定优先级`0.5H+0.3F+0.2R`、特征EMA `0.9/0.1`、连续正确10轮删除。当前查询患者在检索前排除；本轮刚写入的事件在下一轮才可读；测试患者从未写入、更新或参与淘汰。Fold 2新增记忆排除了zero-based索引`376,406`。

G1将修正方向通道置零；G2存储合法训练错误患者的`grad_h(z_true-z_wrong)`归一化向量。两组其他参数、记忆、损失和优化器完全相同。CAM144通过一次原始评估前向捕获，新增分支使用冻结参数的eval OutputLayer和detach特征；原DARG、CAM、OutputLayer不接收新增Memory梯度。空记忆时`h_new==h_base`并严格回退原分类。

预检状态为`PASS`：CAM形状`(871,144)`、空记忆精确回退、G1/G2参数量一致、原主干一步更新一致、无Prompt/OOF教师、Fold 2排除均通过。训练后仅对审计/恢复读回逻辑做了源码修订；训练权重、优化器、RNG和预测未改变。当前独立审计记录的20个best概率最大绝对重放误差为0，硬预测全部一致。

## 正式六指标（10折 mean ± population SD，%）

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 Positive-DCR | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E79-G1 Historical Error Memory | 91.9592±3.0471 | 91.7398±3.0202 | 93.4999±2.8429 | 89.0671±4.7944 | 94.4126±5.4242 | 91.1333±3.2138 |
| E79-G2 Corrective Error Memory | 91.9592±3.0471 | 91.7398±3.0202 | 93.4949±2.8338 | 89.0671±4.7944 | 94.4126±5.4242 | 91.1333±3.2138 |

## 配对比较

以下差值单位为百分点；完整逐折差值、描述性95%区间和W/T/L在`results/paired_comparisons.csv`。

- **G1−E74.4-G2**：ACC `-0.2299`（0/8/2，95% CI `[-0.5149, 0.0551]`），Hard-AUC `-0.2895`（0/7/3），Prob-AUC `+0.0053`（4/0/6），SEN `-1.0000`（0/8/2），SPE `+0.4209`（2/7/1），ASD-F1 `-0.3257`（0/7/3）。排除Fold 2后ACC `-0.2554`、Hard-AUC `-0.3217`、Prob-AUC `+0.0174`、SEN `-1.1111`、SPE `+0.4677`、ASD-F1 `-0.3619`。
- **G2−G1**：ACC/Hard-AUC/SEN/SPE/ASD-F1均为`0.0000`（均0/10/0）；Prob-AUC `-0.0051`（1/7/2），排除Fold 2为`+0.0059`。因此局部修正方向没有产生独立的硬分类收益。
- **G2−E74.4-G2**：ACC、Hard-AUC、SEN、SPE、ASD-F1与G1−基座相同；Prob-AUC `+0.0002`（3/1/6），排除Fold 2 `+0.0233`。基座最佳模型仍是三者中ACC/Hard-AUC最高者。

## 记忆实际建立与错误类型

G1/G2共享同一记忆轨迹。两类记忆首次同时达到64条的epoch为`[12,11,11,13,13,11,11,15,11,11]`，均值`11.9`；10个fold都曾达到总容量128。单条共享10-fold轨迹累计：错误写入`201,445`、已有记录更新`198,796`、淘汰`192,629`、连续正确删除`7,862`；累计原DARG错误的patient-epoch观测`400,241`。这说明记忆并非没有建立，但后期仍有大量错误事件导致高频写入和淘汰。

10个best读取记忆的最终占用合计为`ASD→TD 493`条、`TD→ASD 489`条（`50.20%/49.80%`；这是最终占用分布，不是累计独立患者数）。各fold最终总占用为`99,125,100,64,34,128,121,102,89,120`。G1/G2的事件hash、写入、更新和淘汰轨迹一致。

## 实际纠错与残差对照

在各组best相对其同Checkpoint原DARG（Memory-Off）上，G1和G2均为：原测试错误72例，错误→正确3例，正确→错误1例，净纠错`+2`，硬预测变化4例；3例均为ASD误判纠正，TD误判纠正0例；正确患者保留率`99.8684%`。原错误患者上的加权修正方向正确率为G1 `87.50%`、G2 `88.89%`，相应真实类别margin增益为`0.1168`和`0.1200`。但G2的方向通道关闭或打乱后，硬预测仍与完整G2相同；方向提升没有转化为额外测试纠错。

同Checkpoint的当前主干Memory-Off平均ACC为`91.7294%`，完整G1/G2为`91.9592%`，记忆增强贡献`+0.2299pp`；当前主干相对历史E74.4-G2最佳ACC低`0.4598pp`，所以完整方法仍相对历史基座低`0.2299pp`。平均`||delta_h||/||h||`为G1 `9.9588%`、G2 `9.9435%`，平均触及10%相对范数上限的比例约G1 `96.34%`、G2 `95.54%`，增量约束有明显饱和迹象。

## 同Checkpoint机制干预

- **Memory-Off**：ACC `91.7294%`，Prob-AUC `93.5476%`，净纠错0；打开记忆后ACC升至`91.9592%`并产生净纠错+2，但Prob-AUC下降到G1 `93.4999%`、G2 `93.4949%`。
- **G2 Correction-Direction-Off**：ACC/硬纠错与完整G2相同，Prob-AUC `93.4896%`；**Correction-Direction-Permuted**的硬结果也不变，说明G2的修正方向没有可验证的独立硬诊断贡献。
- **Error-Type-Swap、Memory-Feature-Permuted、Recency-Off**均未改变10折硬纠错总量；其概率变化只在小数点后有限范围内。该结果支持“记忆分支产生了可测增量”，但不足以证明错误类型、特征配对、新鲜度或G2方向形成了可靠、可迁移的疾病机制。

## 主干、RNG、Checkpoint与泄漏审计

G1/G2所有fold的best `base_state`逐张比较一致；预检的一步主干更新与历史轨迹一致，新增分支没有给DARG/CAM/OutputLayer注入梯度，OutputLayer仍由原始`L_base`正常更新。每轮日志保存`rng_before`、`rng_after_train`和`rng_after_eval`；记忆分支没有改变共享主干随机流。20/20 best与10/10 latest均可读回；best保存的是当轮实际读取的上一轮记忆快照，latest保存下一轮恢复所需记忆。

独立审计验证了：CAM144维度、原始4096关系对、训练/测试身份、Fold 2排除、错误事件标签、容量/稳定淘汰、查询患者自排除、测试患者不入库、detach、空记忆回退、G1方向屏蔽、G2方向启用、概率和硬预测重放。没有发现新增标签泄漏或Checkpoint不一致。仍需保留历史协议限制：val=test选模、CV前RFE/标准化和transductive患者图会限制泛化结论；训练记忆对外层训练错误的适配也不能据此宣称已排除过拟合。

## 对三个核心问题的回答

1. **历史误判记忆能否帮助其他患者？** 能产生小幅同Checkpoint硬纠错（+2净纠错），但不能超过E74.4-G2，且Prob-AUC没有稳定改善。
2. **局部正确疾病修正方向是否更强？** 没有。G2相对G1的ACC、Hard-AUC和硬纠错完全相同；方向关闭/打乱不改变硬结果。
3. **保持原主干轨迹时能否超过基座？** 不能。工程上保持了原主干和RNG隔离，记忆分支本身有效但不足以抵消方法Checkpoint相对历史基座的差距。

科研结论为`NO_GO_DEML`。不追加G3/G4、不做测试集调参，也不继续扩大记忆容量或修正方向模块。若未来重启，应先预注册独立模型选择、fold-local预处理和严格的跨患者记忆泛化评估。

## 产物

- 实验目录：[experiment79_deml](/model/spromptgl/experiment/experiment79_deml)
- 权威报告：[results/report.md](/model/spromptgl/experiment/experiment79_deml/results/report.md)
- 最终清单：[manifest.json](/model/spromptgl/experiment/experiment79_deml/manifest.json)
- 六指标：[results/summary.csv](/model/spromptgl/experiment/experiment79_deml/results/summary.csv)
- 配对比较：[results/paired_comparisons.csv](/model/spromptgl/experiment/experiment79_deml/results/paired_comparisons.csv)
- 记忆分析：[results/memory_analysis.csv](/model/spromptgl/experiment/experiment79_deml/results/memory_analysis.csv)
- 纠错分析：[results/correction_analysis.csv](/model/spromptgl/experiment/experiment79_deml/results/correction_analysis.csv)
- 机制干预：[results/interventions.csv](/model/spromptgl/experiment/experiment79_deml/results/interventions.csv)
- 审计状态：[results/audit_status.json](/model/spromptgl/experiment/experiment79_deml/results/audit_status.json)
