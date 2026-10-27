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
