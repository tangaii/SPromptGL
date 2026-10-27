# Experiment 41.1 — SAGE-Align 最终验证

## 执行状态与纠错边界

修正协议已完成100/100个唯一正式折（seed=0，10组各10折）。GPU0执行E0/E2/E4/E6/E8共50折，GPU1执行E1/E3/E5/E7/E9共50折；每卡一个worker、CPU线程固定为1。未修改原始源码、数据或历史模型，未提交或推送Git。

此前实现的100行结果因E5映射不一致、E2/E6重复、QC方向及恢复状态等缺陷被作废，保存在`.runtime/rejected_protocol_audit_v1`；另有早期10折RNG哈希序列化核验失败记录位于`.runtime/e0_rejected`。它们不计入当前100有效折。旧handoff v1的“100有效折”及墙钟声明被本次纠错撤回。纠错前结果已经被观察，当前冻结是工程修正后的冻结，不能声称盲法预注册；未根据这些ACC/AUC调整学习率、8%上限或选模规则。

沿用 **val=test**、历史外层CV之前的RFE/标准化、871人transductive图以及单seed。所有性能只表示历史协议下的内部比较，不是无偏独立泛化证明。

## 论文、源码及历史审计

### 原论文与真实源码逐函数核对

已读[SPromptGL原文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)方法、Fig.1、Eq.(1)–(7)、ABIDE设置与消融，下载PDF与本地PDF字节相同，Fig.1已渲染检查。图中的GFR是患者关系图；CAM在每名患者内部的四个潜在模态位置间交互，PEN包含SPGS和RES。论文(a)消融是CAM基线，当前B0-A含GFR+CAM，二者不能混为同一基线。

| 源码/函数 | 实际行为与本轮处置 |
|---|---|
| `main.RedirectStdStreams.__init__/__enter__/__exit__` | 重定向和恢复日志流；本轮使用实验目录独立日志。 |
| `set_rng_seed`、`set_gpu` | Python/NumPy/CPU/CUDA设种子、cuDNN确定性；E0沿用跨折连续seed0，worker显式固定物理GPU。 |
| `main.sen/spe` | 按混淆矩阵类别计算；本轮明确ASD=0、TD=1，沿用E32精确分数。 |
| `train_and_eval`、`main` | 读既有预处理320维、10折shuffle/random_state10、val=test、ACC严格提升、450/early200；shell覆盖默认超参数。原输出保存会写attn/graph，本轮使用同计算的只读评估辅助函数避免越界写入。 |
| `EvalHelper.__init__` | 四模态DataLayer→GFR→完整PLN构造、两Adam、PLN统一Xavier初始化；保留未执行PEN及RN50构造副作用，匿名临时文件重定向到本实验.runtime/tmp。 |
| `EvalHelper.run_epoch` | 正式使用simple-2：同一分类NLL+四张原始GraphConstructLoss均值，顺序更新GFR和PLN两个Adam。ALoss虽算但不进入反传；其他pre-train/simple/normal/early_stop分支与本轮无关，未借用来改训练。 |
| `print_trn_acc/print_tst_acc/_print_acc` | “train”接口实际使用val索引；hard argmax→one-hot ROC-AUC，最终保存attn/graph有原目录写副作用，故本轮单独计算同口径四指标。 |
| `cal_acc/visualize` | 遗留其他图架构属性、breakpoint与图导出，仅审计、不调用。 |
| `network.get_prompt_text_features/MLP.__init__/forward` | 原八条通用结构文本，拼接两类4096→128→9，含原拼写；构造及缓存不删除，但本轮新分支不使用这些相似度。 |
| `PLN.__init__/forward` | 保留InputLayer/Encoder/FeedForward/OutputLayer。PEN旁路在最后FeedForward之后，直接展平CAM；用hook实测MLP/avg_pool调用次数为0。 |
| `GraphLearn.__init__/forward` | 线性投影后加权余弦，阈值>0.9硬mask；w/t/threshold等遗留参数的注册不删除。 |
| `GCN.__init__/forward`、`adj_normalization` | 两GraphConv、首层ReLU；当前GFR实际使用utils.normalize_adj(A+I)，非新增归一化。 |
| `GFR.__init__/forward` | 四张871×871患者图，各图均传播完整320维；i==1重新赋值会覆盖i==0融合分支，原样保留。训练标签图ALoss不被simple-2优化。 |
| `InitialDataLayer.__init__/forward` | 仅按48+6+10+256切片。 |
| `Attention.__init__/forward` | 实际有除以最小注意力绝对值、F.normalize、softmax及dropout；不能等同论文Eq.(4)的标准缩放点积。 |
| `VariLengthInputLayer.__init__/forward`、`EncodeLayer.__init__/forward` | 按原始模态输入投影Q/K/V及残差LayerNorm；编码后位置已是混合表示，不是解剖区域。 |
| `FeedForwardLayer.__init__/forward` | 两线性、GELU、dropout、残差和LayerNorm，全保留。 |
| `OutputLayer.__init__/forward` | 144维LayerNorm、144→36、GELU、dropout0.35、36→2与log-softmax；传入的attn_embedding实际未使用。只允许适配原参数。 |
| `GraphConv.__init__/forward` | XW后左乘邻接，Xavier初始化，原样保留。 |
| `FusionGate.__init__/forward`、`GraphAttConv.__init__/forward/_prepare_attentional_mechanism_input` | 遗留门控/GAT两两注意实现；当前GCN基座不执行。 |
| `real2col/one_hot` | 图颜色及硬one-hot辅助；one_hot在索引设备创建单位矩阵的兼容实现已在历史版本中存在。 |
| `GraphConstructLoss`、`ClsLoss/ClsLoss_noweight/smooth_CE` | 使用原平滑性、度数、稀疏性图正则和原加权NLL；平滑CE及无权NLL不用于本轮。 |
| `normalize_adj/RBF_kernel/KNN_graph` | 实际仅normalize_adj被基座调用；RBF/KNN遗留路径含未导入distance/NumPy张量API问题，不修复未执行代码。 |
| `my_weight_init/visualize_as_gdf` | 仅线性权重Xavier、bias保持构造值；gdf导出未调用。58个实际函数/方法的行号清单与源SHA记录于.runtime/audit.json。 |

Eq.(1)含监督关系项，实际simple-2丢弃ALoss；Eq.(2)描述图传播，代码有完整320维残差及i==1覆盖；Eq.(3)提出GPT-4文本来源，但本地没有可复核API生成记录；Eq.(4)与实际注意归一化不同；Eq.(5)–(6)的局部token是潜在通道切片和池化，没有真实脑区身份。论文ABIDE为dropout0.75/lr0.0023/编码4/head4等，真实shell为0.35/0.0038/编码1/head2/hidden18/reg0.11；不为“对齐论文”而修改历史基座。Eq.(7)图与分类各权重1，但只保留实际被反传的损失。论文完整PEN 92.30%不能代替E32 B0-A 92.0742%。

### 历史失败与本轮针对性修复

- E11恢复真实HO端点；252条可用、4条未知，语义不借用CAM坐标替代真实FC。
- E17–19的类别原型/Shared/Private、选择性对齐与自由Key投影未稳定超过中性/错配；非零梯度或低对齐距离不能代替疾病增益。
- E28–31细粒度连接文字、方向教师、辅助语义监督和患者条件证据未建立正确语义的必要性；训练后OFF敏感与从头对照增益是两件事。
- E32确定B0-A及PEN构造的RNG边界；E34–35语义关系/消息模式、E36–37表型文字/监督路由没有稳健净收益；E38需用保留PEN构造的B0-A重跑，原物理删除PEN的结果不作为基线。
- E39四独立语义专家/MoE与E40低层选边/反事实损失均NO-GO，存在残差剂量、梯度与稳定性解释边界。本轮只用一个共享小变换，不新增专家或选择器。
- E41的2700维输入、182,016参数Adapter包含CAM直接旁路；医学匹配与校准统计不一致、最终平均95.45%未匹配，冻结分类头E9有9/10最佳epoch0。本轮用必经真实FC证据、同一经验CDF、单向可靠性、8%平滑残差、统一能量正则和head-only/冻结head对照逐项处理。修复不保证科学假设成立。
- E41 correction_audit实查：最终CSV为100个唯一有效折；70个作废折保存在correction审计独立记录，不计入最终CSV，valid=100/excluded=70/invalidated_records=70。其run/ucsma/integration/prompts/ontology/report/diagnostics/counterfactual/correction文件均只读，未混入本轮训练。
- E42四条独立机制链中Q1/Q2/Q3无稳定优势，Q4仅FUNC质量条件有局部价值。因此明确禁止：默认CLIP RN50医学余弦、文献网络结论硬推HO单边方向、患者Top-K/复杂反事实选择器、16项QC统一“偏离中位数越大越差”、静态Prompt冒充真实患者诊断。以上限制在实现和各对照中保持。

### 医学证据、核验范围与创新性

[Ilioska等原文](https://www.nature.com/articles/s44220-026-00656-y.pdf)支持连接异质与区域汇聚的研究动机。其皮质图谱主要为Schaefer而非本数据HO皮质，当前ROI只按名称作保守候选，不复制其功能网络标签或逐边方向；本轮稀疏FC、年龄分箱ridge与其完整连接组/GPR模型不同，不能宣称复现原医学研究。

[Kobbersmed等](https://www.sciencedirect.com/science/article/pii/S0301008226000328)提出全脑规范FC方法。全文请求返回403，实际核查范围为期刊索引摘要/Highlights和[作者FUNCOIN实现](https://github.com/kobbersmed/funcoin)，如实不声称读到被拦截正文。此文仅支持对局部边规范模型的谨慎态度，不作为ASD具体ROI/方向的证据；P4的解剖候选依据来自Ilioska原文。

[BrainPrompt+官方实现](https://github.com/AngusMonroe/BrainPromptPlus)及[期刊摘要](https://pubmed.ncbi.nlm.nih.gov/42127068/)已有多尺度自然语言知识注入。SAGE-Align不使用新文本编码器，只检验冻结结构先验。

[PhenoBrain ICML2026](https://proceedings.mlr.press/v306/meng26f.html)的方法PDF已读，其真实表型/描述条件化长程ROI通信与本轮人口学条件TD参考不同。本数据没有该类患者症状叙事，年龄/性别/站点不能冒充临床症状表型。不要与同名罕见病诊断系统混淆。

[KD-Brain原文](https://arxiv.org/html/2603.19307v1)已有BioMedBERT语义Query和LLM临床交互分布约束。本轮不引入canonical网络身份、自由Text→Key或新增KL病理损失；候选差异仅是实际稀疏FC的训练TD校准、解剖适用性和必经的可拒绝证据残差。规范建模、ROI多尺度汇聚、低秩残差和门控均非独立首创；是否构成有价值的模块取决于真实医学对应是否优于数值、错误对应和普通head微调，不能靠命名宣称创新。

### 修正协议冻结及判定约束

旧实现结果已被看到，因此本轮不是事前盲法研究。纠错只落实用户原始约束，不以旧ACC/AUC调节模型规模、学习率、8%上限、epoch、GO门槛或Prompt数量。为明确“不明显SEN退化”，固定内部容差为0.5pp；这不是临床非劣效界值。训练前冻结全部源码、Prompt、ontology与config哈希；从头执行修正后的100折，作废材料严格隔离。内部GO筛选均不构成统计显著性或跨站点泛化证据。


## 四模态职责与解剖映射

PHENO48为26维年龄分箱、20维站点和2维性别；没有连续年龄。ANAT6为CNR/EFC/FBER/FWHM/QI1/SNR，新增分支不从其推导病理或通用质量方向。FUNC10包括EFC/FBER/FWHM/DVARS/outlier/quality/mean FD/num FD/percent FD/GSR，GSR为ghost-to-signal ratio。新增可靠性只使用全局零基列57/58/60/62的DVARS、outlier、mean FD、percent FD，较高值才受抑制。CORR位于全局64–319列，是新增疾病候选证据的来源。原四模态仍完整进入GFR。

Experiment11核验252条FC的两个端点；195/225/232/236四条未知端点保留于GFR，全部新增对照均排除。HO111标签有110个被已观察FC覆盖；稀疏RFE连接不能代表完整连接组。当前候选只表达解剖范围，不把岛叶分成未经核验的前后网络，也不把CAM的4×36潜变量解释成脑区或原始四模态特征。

## Prompt全文与实际计算角色

没有调用GPT-4/API，没有新增CLIP文本编码器；原PEN构造中的RN50缓存仅为复演B0-A随机副作用且完全旁路。下列文本由研究设计据任务草案撰写并核对可访问原文。验证对象是 **Prompt编译后的结构化医学先验**，不是CLIP嵌入对齐或GPT文本生成的价值。改变ROI适用规则会改变实际FC池化权重；只改措辞而保持结构不变会得到相同计算，不能据此声称验证自然语言含义。正负偏离均保留，文献方向仅为带不确定性的解释，不构成逐边标签。

### P1 Sensorimotor-related convergence

**ASD候选：** At the group level, some autistic individuals exhibit convergent negative normative functional connectivity deviations involving sensorimotor and attention-associated regions. Such convergence is heterogeneous and must not be interpreted as universal hypoconnectivity of every connection attached to these regions.

**TD参考：** In neurotypical participants with comparable age, sex, scanning site and motion conditions, functional connectivity variability should be evaluated against a learned normative reference. Deviation from that reference alone is not a diagnosis of autism.

候选ROI：Right Precentral Gyrus；Left Precentral Gyrus；Right Postcentral Gyrus；Left Postcentral Gyrus；Right Superior Parietal Lobule；Left Superior Parietal Lobule；Right Supramarginal Gyrus, anterior division；Left Supramarginal Gyrus, anterior division；Right Supramarginal Gyrus, posterior division；Left Supramarginal Gyrus, posterior division；Right Angular Gyrus；Left Angular Gyrus；Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)；Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)；Right Central Opercular Cortex；Left Central Opercular Cortex；Right Parietal Operculum Cortex；Left Parietal Operculum Cortex。
适用FC数：68；DOI：10.1038/s44220-026-00656-y；允许进入计算：True；概念SHA256：`6b3578d2f251252522200b4eef1700f975462ad8994812014ceca5db20e9c307`。
不支持的推断：universal ASD hypoconnectivity；patient-specific lesion；canonical attention-network membership；clinical subtype。

### P2 Insular temporal prefrontal associations

**ASD候选：** Some autistic populations show atypical connectivity deviation patterns converging on insular, temporal and prefrontal regions. The direction and anatomical distribution vary between individuals, and these regional associations do not establish a fixed abnormal sign for individual connectivity edges.

**TD参考：** The relevance of connectivity involving insular, temporal and prefrontal regions should be assessed relative to neurotypical conditional distributions and acquisition quality, not from regional identity or a single observed connectivity value alone.

候选ROI：Right Frontal Pole；Left Frontal Pole；Right Insular Cortex；Left Insular Cortex；Right Superior Frontal Gyrus；Left Superior Frontal Gyrus；Right Middle Frontal Gyrus；Left Middle Frontal Gyrus；Right Inferior Frontal Gyrus, pars triangularis；Left Inferior Frontal Gyrus, pars triangularis；Right Inferior Frontal Gyrus, pars opercularis；Left Inferior Frontal Gyrus, pars opercularis；Right Temporal Pole；Left Temporal Pole；Right Superior Temporal Gyrus, anterior division；Left Superior Temporal Gyrus, anterior division；Right Middle Temporal Gyrus, anterior division；Left Middle Temporal Gyrus, anterior division；Right Frontal Medial Cortex；Left Frontal Medial Cortex；Right Frontal Orbital Cortex；Left Frontal Orbital Cortex。
适用FC数：96；DOI：10.1038/s44220-026-00656-y；允许进入计算：True；概念SHA256：`0524674b6c7f0023226726344abf87b0e68ad57b3cb6d5af8b6d976ef29d0944`。
不支持的推断：fixed edge sign；anterior-insula network identity without atlas evidence；individual diagnosis。

### P3 Medial frontal cingulate parietal convergence

**ASD候选：** Regional convergence of positive normative connectivity deviations has been reported in portions of medial prefrontal, cingulate and inferior parietal areas in some autistic populations. The observation is population-level and should not be imposed as a mandatory positive deviation for every patient or individual edge.

**TD参考：** Observed deviations in medial prefrontal, cingulate and parietal regions should be compared with conditional neurotypical expectations while retaining uncertainty about the underlying functional system and individual deviation direction.

候选ROI：Right Supramarginal Gyrus, anterior division；Left Supramarginal Gyrus, anterior division；Right Supramarginal Gyrus, posterior division；Left Supramarginal Gyrus, posterior division；Right Angular Gyrus；Left Angular Gyrus；Right Frontal Medial Cortex；Left Frontal Medial Cortex；Right Subcallosal Cortex；Left Subcallosal Cortex；Right Paracingulate Gyrus；Left Paracingulate Gyrus；Right Cingulate Gyrus, anterior division；Left Cingulate Gyrus, anterior division；Right Cingulate Gyrus, posterior division；Left Cingulate Gyrus, posterior division。
适用FC数：64；DOI：10.1038/s44220-026-00656-y；允许进入计算：True；概念SHA256：`658691e997760e3a61565bc47c037acb6d20aa5fb56041a6633376d524d8130b`。
不支持的推断：DMN identity without independent network mapping；mandatory positive deviation；individual diagnosis。

### P4 Subcortical cortical distributed differences

**ASD候选：** Atypical connectivity involving subcortical and cortical regions may contribute to distributed functional differences in subsets of autistic individuals. Its spatial distribution and direction are heterogeneous, and sparse observed connectivity provides incomplete evidence for the underlying neural system.

**TD参考：** Subcortical-cortical connectivity should be evaluated against a neurotypical reference conditional on measured demographic and acquisition variables, without assuming that atypicality in any single observed connection identifies autism.

候选ROI：Left Thalamus；Left Caudate；Left Putamen；Left Pallidum；Left Hippocampus；Left Amygdala；Left Accumbens；Right Thalamus；Right Caudate；Right Putamen；Right Pallidum；Right Hippocampus；Right Amygdala；Right Accumbens。
适用FC数：55；DOI：10.1038/s44220-026-00656-y; 10.1016/j.pneurobio.2026.102906；允许进入计算：True；概念SHA256：`d16d7650bf3a36f5005629043ed97d864d8153a0aede1146f950949c84b38442`。
不支持的推断：every subcortical edge is abnormal；fixed direction；diagnosis from one connection。

## 正确、错误和数值先验的公平性

E5采用同一个ROI置换，同时决定错误ROI、由两个真实端点诱导的错误FC、错误粗解剖族和全部三尺度池化权重。置换严格限制在相同实际连接度及皮质/皮质下层内；24,000步仅解剖约束搜索不读取任何患者或标签。候选ROI数、ROI重叠、联合ROI及候选ROI度分布严格保持；FC数量、重叠、联合FC、连接类型和端点覆盖的剩余误差如实列在ontology与下表，不声称完全公平。

P4覆盖全部可观察皮质下候选，其皮质下-皮层身份在类型保持置换下不变；因此E5不能单独否定P4宽泛解剖假设，主要检验P1–P3的对应。没有先构造错误ROI再用无关随机FC替代。

| 结构量 | 正确 | 错误 | 差 |
|---|---|---|---|
| roi_counts | [18, 22, 16, 14] | [18, 22, 16, 14] | [0, 0, 0, 0] |
| edge_counts | [68, 96, 64, 55] | [68, 96, 64, 55] | [0, 0, 0, 0] |
| roi_union | 62 | 62 | 0 |
| edge_union | 202 | 202 | 0 |
| covered_endpoint_counts | [60, 78, 59, 56] | [66, 74, 54, 56] | [6, -4, -5, 0] |
| edge_type_counts | [[58, 10, 0], [85, 11, 0], [56, 8, 0], [0, 55, 0]] | [[58, 10, 0], [85, 11, 0], [56, 8, 0], [0, 55, 0]] | [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] |

逐概念改变的FC数量：[90, 86, 90, 0]。完整重叠矩阵、端点连接度分布误差见ontology。

E2固定随机FC索引槽及8/32列索引层次，完全不读取HO身份或训练标签；E6只用outer-train FC协方差KMeans四原型的连续软亲和力，按训练内Kish有效支持量匹配四槽证据规模。不实施患者Top-K选边，不使用outer-test标签。二者共享4×18接口和全部参数，三尺度均有效，数值组解剖覆盖系数固定为1而不读取HO度数；数值索引层次不是解剖层次，因此E9−E2仍含医学解剖组织与数值分组方式的差异，E5是更直接的医学对应对照。

## 条件TD参考、连续校准与拒绝

每外层折仅在训练TD拟合ridge（alpha=10）；全部训练患者使用固定5折交叉拟合，测试使用完整训练TD。条件项为真实26维年龄分箱、站点、性别及mean FD；E3只有全局TD截距。OOF TD残差MAD用于尺度化并在±4处平滑分支之前截断证据输入；这不是经过临床验证的预测置信区间，也不是FUNCOIN完整全脑协方差模型。

每尺度六通道为正z、负z、正向z>1占比、负向z<−1占比、z²/4、|z|；零偏离时全部为零。概念分数为三尺度平均的0.5×平均|z|+0.5×|z|>1占比，训练TD OOF及所有推理都使用同一函数。匹配强度取同一训练TD分数的经验CDF。a为参考支持×单向FUNC支持×解剖覆盖；q=a×CDF，拒绝u=∏k(1−qk)，各概念质量=(1−u)qk/Σq。所有质量严格守恒。参考支持<0.1、条件层无TD或QC负担超过训练TD99.5分位时严格拒绝；这也是冻结规则，不按测试ACC调整。E4仅取消QC，E7仅把可用槽质量归一化并取消拒绝。

训练分层、正负偏离、OOF残差、匹配分位数和正确/错误映射敏感性见`train_calibration.csv`；硬拒绝与连续拒绝质量是不同量。FUNC支持是工程可靠性条件，不是真实扫描质量金标准。

## SAGE-Align与训练协议

有效证据vik=massik×eik；共享无bias 18→8 GELU、CAM均值36→4 sigmoid查询调节、8→36→144线性变换产生raw残差。所有槽为零时共享变换严格为零；CAM没有独立产生残差的通路。最后线性权重全零初始化；r_eff=raw×[c tanh(||raw||/c)/||raw||]，c=0.08||CAM||。在原OutputLayer之前加入残差，未新增独立分类器。

GFR/InputLayer/Encoder/FeedForward权重冻结并处于eval状态，缓存每折B0的CAM等价于其确定性前向。OutputLayer架构不变，E1–E7/E9从相同B0权重适配；E8仅冻结这些权重，其训练Dropout仍与E9相同。原加权NLL加统一1e−3残差相对能量正则，无其他新损失。新阶段Adam、两组lr均3.8e−4，weight_decay=0；B0原双Adam的lr=0.0038、reg=0.11保持。450上限、wait>200停止；只在ACC严格提高时保存best，平局保留首次，epoch0表示第一次更新之后。

所有适配组同折分支初始化种子411000+fold，Dropout训练随机流412000+fold；此协议与B0跨折seed0连续随机流明确区分。best与latest保存Adam及Python/NumPy/CPU/CUDA RNG；B0另存下一折连续RNG，每25epoch覆盖恢复点，完整折跳过。预检已经逐值验证基线双Adam、适配Adam、磁盘恢复和下一折初始化。

| 组别 | 唯一改动 | 分支参数 | 可训练总参数 |
|---|---|---:|---:|
| E0_B0A | B0-A原样重训 | 0 | 1462957 |
| E1_HEAD_ONLY | 仅原OutputLayer | 0 | 5582 |
| E2_NUMERIC_ONLY | 固定数值层次证据 | 5760 | 11342 |
| E3_GLOBAL_REFERENCE | 全局TD参考 | 5760 | 11342 |
| E4_NO_QC | 关闭FUNC支持 | 5760 | 11342 |
| E5_WRONG_SEMANTICS | 共同置换的错误解剖对应 | 5760 | 11342 |
| E6_DATA_PRIOR | 训练折软数值先验 | 5760 | 11342 |
| E7_FORCED_ALIGNMENT | 强制可用概念对齐 | 5760 | 11342 |
| E8_FROZEN_HEAD | 仅分支训练、OutputLayer冻结 | 5760 | 5760 |
| E9_SAGE_FULL | 完整结构化医学先验+联合适配 | 5760 | 11342 |

E2–E9均18维/槽、4槽（共72个输入值）、5,760分支参数；|z|与正负z存在冗余，训练有效证据矩阵的数值秩保存在diagnostics，不将72个值声称为72个独立维度。E2–E9均5,760分支参数，联合适配加OutputLayer5,582参数共11,342；E8仅5,760。零初始化使第一步仅最后残差投影取得非零分支梯度，随后所有分支路径均实测有梯度；不能以第一步部分零梯度伪称参数无效。

## B0-A精确复现

10/10折四指标、best epoch、执行epoch、初始主干SHA和入口RNG SHA逐项与Experiment32匹配。

| 折 | best epoch | 执行epoch | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 322 | 450 | 95.4545 | 95.1220 | 100.0000 | 90.2439 |
| 2 | 149 | 351 | 88.5057 | 88.3351 | 91.3043 | 85.3659 |
| 3 | 428 | 450 | 88.5057 | 88.7328 | 84.7826 | 92.6829 |
| 4 | 243 | 445 | 94.2529 | 93.9362 | 97.8723 | 90.0000 |
| 5 | 428 | 450 | 94.2529 | 94.1223 | 95.7447 | 92.5000 |
| 6 | 124 | 326 | 95.4023 | 95.3723 | 95.7447 | 95.0000 |
| 7 | 154 | 356 | 90.8046 | 90.1862 | 97.8723 | 82.5000 |
| 8 | 280 | 450 | 89.6552 | 89.3085 | 93.6170 | 85.0000 |
| 9 | 169 | 371 | 90.8046 | 90.5585 | 93.6170 | 87.5000 |
| 10 | 251 | 450 | 93.1034 | 92.8723 | 95.7447 | 90.0000 |

## 四项正式结果

单位%；十折均值±population SD。Hard-AUC沿用硬预测one-hot口径，ASD=0、TD=1。

| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E1_HEAD_ONLY | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E2_NUMERIC_ONLY | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E3_GLOBAL_REFERENCE | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E4_NO_QC | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E5_WRONG_SEMANTICS | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E6_DATA_PRIOR | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E7_FORCED_ALIGNMENT | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |
| E8_FROZEN_HEAD | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E9_SAGE_FULL | 91.9592 ± 2.7267 | 91.7482 ± 2.7033 | 94.4172 ± 4.1480 | 89.0793 ± 3.7283 |

## E9核心配对比较

单位百分点；同折W/T/L，不将重叠训练折当作独立外部样本。

| E9减对照 | ΔACC / W-T-L | ΔHard-AUC / W-T-L | ΔSPE | ΔSEN |
|---|---|---|---:|---:|
| E0_B0A | -0.1149 / 0-9-1 | -0.1064 / 0-9-1 | -0.2128 | +0.0000 |
| E1_HEAD_ONLY | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E2_NUMERIC_ONLY | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E3_GLOBAL_REFERENCE | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E4_NO_QC | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E5_WRONG_SEMANTICS | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E6_DATA_PRIOR | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E7_FORCED_ALIGNMENT | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E8_FROZEN_HEAD | -0.1149 / 0-9-1 | -0.1064 / 0-9-1 | -0.2128 | +0.0000 |

逐折全部四指标保存在`fold_metrics.csv`，完整逐折配对差见`summary.json`。

## 优化、残差和固定checkpoint反事实

| 组别 | best epoch=0折数 | best epoch中位数 | 未匹配质量 | 有效残差/CAM | raw残差/8%上限 | 饱和比例 |
|---|---:|---:|---:|---:|---:|---:|
| E1_HEAD_ONLY | 10/10 | 0.0 | 0.344745 | 0 | 0 | 0 |
| E2_NUMERIC_ONLY | 10/10 | 0.0 | 0.324366 | 0.00124925 | 0.0156298 | 0 |
| E3_GLOBAL_REFERENCE | 10/10 | 0.0 | 0.252529 | 0.0013729 | 0.0171785 | 0 |
| E4_NO_QC | 10/10 | 0.0 | 0.263516 | 0.00135645 | 0.0169716 | 0 |
| E5_WRONG_SEMANTICS | 10/10 | 0.0 | 0.341225 | 0.00122561 | 0.0153334 | 0 |
| E6_DATA_PRIOR | 10/10 | 0.0 | 0.317130 | 0.00127343 | 0.0159327 | 0 |
| E7_FORCED_ALIGNMENT | 10/10 | 0.0 | 0.000000 | 0.00179281 | 0.022437 | 0 |
| E8_FROZEN_HEAD | 10/10 | 0.0 | 0.344745 | 0.00121223 | 0.0151662 | 0 |
| E9_SAGE_FULL | 10/10 | 0.0 | 0.344745 | 0.00121223 | 0.0151662 | 0 |

E9有10/10折best停留在首次更新。若该数字很高，只能判断后续学习未在历史选模口径改善，不能宣称这是充分学习后的最优医学模块。每epoch训练NLL、残差能量、验证ACC、预测翻转和raw/effective残差都保存在`.runtime/logs/*_trace.csv`；未事后改选连续指标。

| 固定E9干预 | 翻转数/871 | 平均绝对log-prob变化 | 平均残差变化L2 | 匹配质量 | 拒绝质量 |
|---|---:|---:|---:|---:|---:|
| REAL | 0/871 | 0 | 0 | 0.655255 | 0.344745 |
| WRONG_SEMANTICS | 0/871 | 5.0563285e-05 | 0.0003959186 | 0.661320 | 0.338680 |
| NUMERIC_PRIOR | 0/871 | 5.6957708e-05 | 0.00044048663 | 0.696107 | 0.303893 |
| SEMANTIC_OFF | 0/871 | 0.00041237352 | 0.0033048513 | 0.000000 | 1.000000 |
| QC_OFF | 0/871 | 5.4061065e-05 | 0.00041261181 | 0.736484 | 0.263516 |
| FORCED_ALIGNMENT | 0/871 | 0.00020764782 | 0.0015399008 | 1.000000 | 0.000000 |
| PATIENT_EVIDENCE_MISMATCH | 0/871 | 0.00026503112 | 0.0020619712 | 0.655255 | 0.344745 |
| ALL_RESIDUAL_OFF | 0/871 | 0.00041237352 | 0.0033048513 | 0.000000 | 1.000000 |

REAL逐值重放best预测；其余干预固定E9参数及训练校准CDF，不重新拟合或选模。SEMANTIC_OFF与ALL_RESIDUAL_OFF在单分支结构中应相同；它们保留共同适配后的分类头。非零连续响应说明使用了证据通路，不能替代E5/E6独立重训对照，也不证明临床因果。

站点20类、年龄26分箱和按各折训练QC中位数定义的高低层机制关联见`associations.csv`；患者级机制记录见`patient_diagnostics.csv`。年龄分箱不冒充连续岁数；站点/QC关联只能排查混杂，不能证明无混杂或外部泛化。

## GO/NO-GO与创新性结论

冻结内部筛选门槛：E9−E0 ACC和Hard-AUC均≥0.2pp、二者至少7/10折严格胜，SEN下降不超过0.5pp；医学必要性要求E9同时优于E1/E2/E5/E6并有真实FC驱动的决策响应。可靠性单列要求优于E4/E7。阈值不是统计显著性或跨中心泛化证据。

```json
{
  "GO_CLASSIFICATION": false,
  "GO_MEDICAL_SEMANTICS": false,
  "GO_RELIABILITY": false,
  "final": "FINAL NO-GO",
  "interpretation": "Internal descriptive screening; no statistical significance or independent generalization proof."
}
```

**FINAL NO-GO。** 没有同时满足分类增益与医学语义必要性，不能作为MICCAI2027第一创新点的已成立证据。结束此SAGE-Align/UCSMA医学语义对齐路线，不继续调参，不提出实验41.2。

分类头微调、数值规范偏离、多尺度聚合、医学适用范围和FUNC可靠性是不同贡献；本矩阵只能按相应对照讨论，不能合并称“多模态对齐有效”。P4不变及E5剩余结构误差限制必要性检验强度；当前负结果也不证明相关脑区没有生物学价值。

未来若评价任何其他独立方法，应在外层训练内完成预处理/RFE/规范建模/选模，独立验证或站点留出且测试标签不参与选择，明确inductive/transductive部署条件，并使用独立队列及重复随机种子评估不确定性。此处只说明证据标准，不自动启动新实验。

## 文件、恢复和交付

正式结果、诊断与报告均已写到用户指定目录；所有正式checkpoint含恢复所需Adam和RNG，100折载荷SHA及指标重算通过。原始六文件、数据、Prompt、ROI映射、缓存及checkpoint哈希见manifest与`.sha256`。handoff只幂等追加，明确更正旧v1，保留全部历史内容。

<!-- corrected_v2_gradient_accounting -->

### 实际梯度参数与输入维数补充

`gradient_audit.csv`逐折记录72个证据输入值、18个有效通道、训练有效证据矩阵秩、所选best步骤中非零梯度张量包含的参数量，以及best和终止步骤的Adam一阶矩非零元素数。非零梯度张量的参数量不是逐元素非零梯度计数。5760是完整分支可训练容量；零初始化使首次更新通常仅5184个末层分支参数有非零梯度，不能称全部5760参数已完成学习。后续梯度路径已在训练前验证；终止状态有更新不能替代正式选中的best checkpoint。本补充只读取已选checkpoint，不训练、不更改选模或冻结设计。

E9训练NLL十折均值从首步的0.204806降至终止步骤的0.095635；完整轨迹的最大患者饱和比例为0.585459。终止步骤只用于优化诊断，正式指标始终取首次严格最佳ACC checkpoint。

<!-- corrected_v2_wall_time_scope -->

墙钟统计范围：从首个已记录启动时刻 2026-10-26T14:03:02.293569+00:00 至交付准备 2026-10-26T17:26:32.753226+00:00，共 12210.46 秒；修正阶段从 2026-10-26T16:44:25.560646+00:00 起共 2527.19 秒。更早审计时间没有独立完整记录，因此上述总墙钟仅对应已记录窗口，不伪称精确覆盖全部前置工作；不把并行worker时长相加。

<!-- corrected_v2_full_trajectory_failure -->

### 长期优化失败的明确补充

所选best为首次更新，其raw/有效残差未饱和；这不能推广到完整训练过程。E9后续10/10折出现饱和，每折有120–169个epoch存在raw残差范数大于3倍8%上限的训练患者；峰值患者比例为58.55%，最后100步的十折平均比例为13.29%。tanh上限保持有效残差有界，但未阻止raw残差的长期饱和。训练前短程梯度/恢复预检没有保证长期稳定；该稳定性目标没有通过正式轨迹验证。训练NLL下降不能弥补最佳epoch全部停在0，也不能称充分学习。这些是已冻结实验的科学与优化失败，按原规则保留100个正式折，不为此重新调参或选模。完整90折轨迹汇总见trajectory_audit.csv；最终维持FINAL NO-GO，永久停止本路线。
