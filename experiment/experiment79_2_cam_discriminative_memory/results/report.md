# Experiment 79.2：CAM144疾病判别记忆的三路线正式实验

## 状态与结论

- **STATUS**：`COMPLETE_AUDITED`
- **科研判定**：`NO_GO_CAM_MEMORY`
- **子结论**：`PROBABILITY_ONLY_NOT_ESTABLISHED`
- 30/30正式方法折完成（10折共享一条原DARG主干，每折维护G1/G2/G3三个独立记忆分支）。
- 60/60最佳Checkpoint（30 Base-Anchored + 30 Method-Selected）独立重放通过，10/10 shared latest恢复通过。
- 原DARG逐折RNG比对为0个不匹配；历史E74.4-G2最佳Epoch和原主干状态逐折核对通过。

结论是三种CAM记忆都没有产生稳定的硬分类增益。G1、G2与E74.4-G2在六指标的硬结果完全相同；G3的Prob-AUC平均下降0.0106个百分点。记忆分支产生了小幅概率扰动，但Memory-Off、Memory-Content-Permuted以及各组专属干预均没有产生硬预测变化，因此不能宣称三种记忆内容提供了独立、可迁移的疾病判别信息。

## 论文、源码和基座审查

审查了[SAM 2论文](https://arxiv.org/abs/2408.00714)及官方[Memory Attention实现](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_attention.py)、[Memory Encoder](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_encoder.py)、[sam2_base.py](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py)和[视频预测器](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_video_predictor.py)。本实验只借鉴有限记忆库、Cross-Attention和“先读取、后写入”的流式抽象，没有移植视频位置编码、掩码解码、对象跟踪或视频对象传播假设。

审查了[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[原作者代码](https://github.com/wanxixi11/SPromptGL)和当前仓库。E74.4-G2实际使用`PLNNoPEN`；CAM144是原`OutputLayer`输入，原分类头保持只读调用；本实验没有引入原始PEN、疾病文本Prompt、CLIP、OOF教师或分类后Logits残差。

E74.4-G2只读审计复核的历史结果（%，10折mean±population SD）为：ACC `92.1891±2.7648`、Hard-AUC/BA `92.0294±2.7255`、Prob-AUC `93.4946±2.9195`、SEN-ASD `90.0671±3.8862`、SPE-TD `93.9917±4.7320`、ASD-F1 `91.4590±2.9013`。源码、数据、E61 warmup、E74.4-G2权重和历史实验文件SHA已写入[preflight.json](../results/preflight.json)并通过核验。

## 运行路径与资源

首选路径在真实mkdir、写入、fsync、原子rename和读回探针时触发OSSFS `EIO(errno=5)`，因此正式路径为：

`/model/spromptgl/experiment/experiment79_2_cam_discriminative_memory`

设备为`PPU-ZW810E`，显存上限98,304 MiB。按用户要求启动过10个worker，最大并发记录为10；后续折在恢复点继续完成，以避免同一fold重复运行破坏轨迹。可验证的worker日志时间窗约`2小时08分31秒`，10个fold任务`elapsed_seconds`之和为`12,219.56秒`，该累计值包含并行重叠及恢复段，不能当作墙钟时间。训练日志没有独立采样实时显存峰值，因此不虚构峰值数字。

## 三种记忆实际保存的内容

| 组别 | 保存对象 | 读取方式 | CAM144增强来源 |
|---|---|---|---|
| G1 LDCM | 归一化CAM144空间中相近但标签相反的ASD/TD患者对、对照中心、TD→ASD方向、局部双类支持和可靠性 | 当前CAM144检索Top-8对照中心，并以当前患者相对ASD/TD锚点的证据调制方向 | 仅来自对照记忆Value的Cross-Attention读出 |
| G2 DDEM | 正确患者的单维干预margin贡献、最多12维证据掩码、实际证据值、联合遮蔽margin下降和可靠性 | 按证据掩码覆盖的特征匹配ASD/TD证据记忆，各类Top-8 | 仅来自证据掩码Value的Cross-Attention读出 |
| G3 DDM | 每类3个CAM局部分布成分的均值、对角方差、先验和训练支持患者 | 以稳定对角高斯匹配分数对6个分布成分做Distribution Attention | 仅来自分布成分Value的Attention读出 |

三组Query/Key维度均为64，最终无bias特征投影为144维，门控初始bias为−3.0，安全上限为`0.075||CAM144||`。每个新增模块参数量均为36,425；空记忆严格满足`h_new==h_base`。

最终最佳Checkpoint的记忆状态显示：G1固定64对，平均约79.1个不同患者参与这些对照对，局部支持数固定为12；G2固定96条证据记忆，平均保留12个有效证据维度，平均可靠性约0.906；G3固定6个分布成分，平均每成分约130.65名训练患者，使用类别内收缩后的对角方差。Fold 2的zero-based索引376、406没有进入新增记忆。

## 正式六指标

单位为百分比；均值和标准差均为10折population SD。Base-Anchored使用历史E74.4-G2每折最佳Epoch，Method-Selected按同一`val=test`严格提高ACC规则选择。两套Ledger必须分别报告，不能混用。

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G1 Base-Anchored | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G2 Base-Anchored | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G3 Base-Anchored | 92.1891±2.7648 | 92.0294±2.7255 | 93.4840±2.9128 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G1 Method-Selected | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G2 Method-Selected | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| G3 Method-Selected | 92.1891±2.7648 | 92.0294±2.7255 | 93.4840±2.9128 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |

完整逐折指标、每折差值、描述性95%区间和W/T/L见[summary.csv](summary.csv)、[fold_metrics.csv](fold_metrics.csv)和[paired_comparisons.csv](paired_comparisons.csv)。

## 配对结果和Fold 2敏感性

### Method-Selected（Base-Anchored完全相同）

- **G1−E74.4-G2**：ACC、Hard-AUC/BA、SEN-ASD、SPE-TD和ASD-F1均为`0.0000pp`，W/T/L均为`0/10/0`；Prob-AUC为`0.0000pp`，排除Fold 2仍为`0.0000pp`。
- **G2−G1**：ACC、Hard-AUC/BA、SEN-ASD、SPE-TD和ASD-F1均为`0.0000pp`，W/T/L均为`0/10/0`；Prob-AUC为`0.0000pp`，排除Fold 2仍为`0.0000pp`。
- **G3−E74.4-G2**：ACC、Hard-AUC/BA、SEN-ASD、SPE-TD和ASD-F1均为`0.0000pp`；Prob-AUC `−0.0106pp`，排除Fold 2为`−0.0118pp`。
- **G2−G3**：硬指标均为`0.0000pp`；Prob-AUC `+0.0106pp`，排除Fold 2为`+0.0118pp`。

Base-Anchored的结论相同，因为三组在每折历史最佳Epoch均未改变硬预测；所有比较的完整95%描述区间和逐折向量在CSV中保留。

## 真实分类纠错

在同Checkpoint比较（完整方法相对于同一Checkpoint的原DARG输出）中，10折合计每组原始错误均为68例，三组均为：

- 错误→正确：`0`
- 正确→错误：`0`
- 净纠错：`0`
- ASD/TD错误分别纠正：`0/0`
- 硬预测翻转：`0`
- 正确患者保留率：`100%`

这与“方法相对于历史基座”的差异严格分开：由于原DARG在历史最佳Epoch的硬预测已被完全复现，完整E79.2方法没有在该锚定点纠正任何测试患者。三组平均门控/相对增量分别为：G1 `0.0717/0.2343%`、G2 `0.0627/0.6055%`、G3 `0.0630/0.2193%`；平均cap fraction均为`0`，因此本轮没有复现E79式大面积触及安全上限的问题，但增量仍太小而无法跨越分类边界。

## 机制干预

所有干预都是同Checkpoint只读重放，不是新增训练组。

- **Memory-Off**：G1/G2硬指标与完整输出相同；G3也没有硬变化，但Prob-AUC相对完整输出的平均差为`−0.0106pp`。关闭记忆没有产生硬纠错收益。
- **Memory-Content-Permuted**：三组均为0个硬预测翻转。G1/G2 Prob-AUC变化为0；G3为`−0.0106pp`。因此不能证明具体历史Value是硬分类所依赖的信息。
- **G1 Pair-Direction-Permuted**：0个硬翻转、Prob-AUC变化0。
- **G2 Evidence-Mask-Permuted**：0个硬翻转、Prob-AUC变化0。
- **G3 Variance-Off**：0个硬翻转，Prob-AUC相对完整输出为`−0.0053pp`。

干预造成的患者概率平均绝对变化很小：Memory-Off约为G1 `1.57×10^-4`、G2 `2.36×10^-4`、G3 `1.97×10^-4`；这说明增强分支确实产生了有限数值扰动，但扰动没有形成可验证的疾病判别收益。

## 工程与信息泄漏审计

独立[audit_status.json](audit_status.json)确认：

1. 60个最佳Checkpoint均能独立加载，保存的概率、硬预测和指标可重放。
2. 10个latest均保存了下一Epoch的恢复状态、三组模块、三组优化器、三份记忆和完整RNG；恢复指针为最后trace epoch+1。
3. 原DARG/GFR/CAM/OutputLayer只由`L_base`更新，新增Memory损失没有注入原主干；三组共享一条原始主干轨迹。
4. CAM144形状为`(871,144)`，空记忆精确回退，Top-K检索不包含测试患者或当前查询自身。
5. G1对照对、G2证据记忆和G3分布统计只来自合法外层训练患者；Fold 2重复患者376、406被新增记忆排除。
6. Memory-Content-Permuted真正保持Query、Key、Top-K和注意力结构，只打乱Value与Key的对应；各组专属干预也只改变指定内容。
7. 历史根目录源码、原数据和历史实验文件SHA保持不变。

仍需保留历史限制：`val=test`参与Checkpoint选择、CV前RFE/标准化、transductive患者图以及Fold 2历史重叠会限制无偏泛化结论。本实验没有修复这些历史协议，也没有使用测试标签构建新增记忆。

## 科研判定

三个候选均没有超过E74.4-G2，也没有在历史最佳Epoch产生真实硬纠错；关键记忆内容干预不能显示稳定独立贡献。当前证据不足以区分“CAM144中缺乏可迁移的额外记忆信息”和“当前冻结OutputLayer/小幅融合没有有效利用这些信息”这两个根本原因，因此不把失败过度归因于某一方。

最终判定：**`NO_GO_CAM_MEMORY`**。不继续追加E79.3，不扩大记忆类型或容量，不进行测试集超参数搜索。若未来重启，优先改为独立模型选择、fold-local预处理和严格跨患者的外部记忆泛化协议。

## 产物

- [实验清单](../manifest.json)
- [配置](../config.json)
- [预检记录](preflight.json)
- [六指标汇总](summary.csv)
- [逐折指标](fold_metrics.csv)
- [配对比较](paired_comparisons.csv)
- [记忆分析](memory_analysis.csv)
- [纠错分析](correction_analysis.csv)
- [机制汇总](mechanism.csv)
- [机制干预逐折结果](intervention_analysis.csv)
- [独立审计状态](audit_status.json)
