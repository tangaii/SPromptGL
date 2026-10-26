# 实验39：HASE-MoE 完整结果

状态：**训练与分析完成，100/100折**；GPU0/GPU1各50折；单seed=0；worker累计训练耗时（两卡所有配置耗时求和）61.5分钟，双卡训练调度墙钟18.7分钟。判定：**NO-GO**。

交付状态：训练与分析已完成；最终根目录包同步状态为`TEMP_ONLY_OSSFS_EIO`，实验包同步错误为`OSError(5, 'Input/output error')`，handoff追加状态为`PENDING_OSSFS_WRITE_EIO`，错误为`OSError(5, 'Input/output error')`。正式工作区副本仍待写回，完整当前副本与待追加文本位于`/tmp/SPromptGL-E39-HASE-stage/experiment/experiment39_heterogeneity_aware_semantic_experts`。

## 审计与冻结协议

实际读取本地 MICCAI 2025 SPromptGL 原文 Fig.1、Section 2.1–2.2、Eq.(1)–(7)、ABIDE设置与消融；并审计根目录 `main.py`、`model.py`、`network.py`、`layers.py`、`utils.py` 及 `ABIDE-simple-2-concat-weighted-cosine.sh`。上游仓库链接为[SPromptGL GitHub](https://github.com/tangaii/SPromptGL)，审计使用的本地checkout HEAD为 `7747a1521629f95413faca133ba8c82ed7f1faf2`，核心文件哈希见下文。论文 Fig.1 的流程是 GFR 患者图、DGCN、PLN/CAM/PEN；Eq.(1)把监督标签关系图纳入图目标，Eq.(3)–(6)描述CLIP文本提示、局部SPGS与RES，Eq.(7)写图损失与分类损失。论文ABIDE报告92.30% ACC、92.22% AUC；其训练参数与仓库 shell 不同，本实验严格按本地真实shell。

源码中四个GFR视图都是**871名患者之间的关系图**，不是脑区图。`GFR.forward` 的 `i==1` 分支会覆盖 `i==0` 的 `combined_feat`，本实验原样保留。`simple-2` 每轮计算 `ALoss`，但实际反传为分类NLL加四个输入图 `GraphConstructLoss` 的均值；`ALoss` 没有进入目标。GFR/GCN、阈值、传播、四图正则、融合、CAM、OutputLayer均未改。PEN的SPGS与RES前向被整体旁路；保留PLN InputLayer/Encoder/FeedForward及其初始化顺序。

E0按原构造顺序仍构造原始PLN、CLIP文本缓存和优化器，然后把CAM四个36维位置直接展平输入OutputLayer；它与Experiment32 B0-A独立重训逐折核对。B0-A历史为ACC 92.0742%、Hard-AUC 91.8546%、SEN-ASD 89.0793%、SPE-TD 94.6300%。本轮E0重训10折后，初始主干哈希、入口RNG哈希、最佳epoch及四项折指标均10/10逐项匹配。

历史结果边界：原始完整PEN的Experiment32 B2 ACC=91.7294%仅作描述性参照，不能替代本轮B0-A E0。Experiment11恢复的256条连接端点和Experiment19四域映射为本轮冻结映射来源。Experiment3/8/13–22、26、28–31反复显示自由文本投影、静态语义关系或图调节的结果混合，错误/随机先验不总是更差；Experiment20的概念关系严格门槛未通过。Experiment34–35的语义关系/消息增量未稳定优于B0-A或强数值/错配对照；Experiment36的四域文本路由对B0-A平均ACC无增益，Experiment37标签监督混合组未胜过其基础及错配对照。故本轮要求容量、普通MoE、正确/错误解剖、训练折数值集合和患者错配的并行重训对照。实验38在本轮设计冻结时只读本地临时态、未作为结果或模型状态来源。

历史协议保持不变：`seed=0`，`StratifiedKFold(10, shuffle=True, random_state=10)`，val_idx=test_idx，ACC严格提高才替换checkpoint，450 epoch，wait>200早停，双Adam，simple-2，pre-CV RFE/标准化以及871人transductive患者图。由此测试折参与checkpoint选择、预处理在外层CV之前完成、图传播跨越全部受试者；结果只支持该历史协议下的内部配对比较，不构成无偏泛化或外部临床验证。

## 医学先验与结构

四条冻结的机制prompt见 `prompts.json`，其文案是用户提供的候选机制草案，不声称来自某次GPT-4 API调用，也不是逐字文献事实。解剖候选完全复用Experiment19映射与Experiment11数值核验：D1/D2/D3/D4分别72/56/45/32条，联合覆盖165条；映射之间允许重叠，不把这些集合称作完整规范脑网络。四条端点未知连接195/225/232/236仍走原始主干，但不进入真实、错误解剖或数据驱动机制掩码；E2作为无解剖约束的普通MoE仍按注册读取全部256维数值输入。E5错误掩码与E6数据驱动掩码均限于其余252条已核验连接，且维持相同的专家大小和重叠结构；这是训练前为满足端点约束固定的实现细化。每条连接的ROI端点、E5错误集合和索引见 `mechanism_masks.json`。

HASE从CAM输出 `H:[871,4,36]` 形成患者上下文，并从原始CORR256取出按掩码的有符号连接及仅用outer-train中位数/MAD标准化的偏离。该z表示相对训练队列分布，不代表临床病理或ASD专属异常。每个专家有独立无偏FC编码器和无偏Adapter，Softmax路由逐专家接收GFR/CAM上下文、该专家机制响应、年龄分箱one-hot与性别，以及ANAT/FUNC质量和头动相关条件；PHENO站点20维不直接进新增路由。ANAT/FUNC仅是可靠性信息，不是症状或疾病证据。路由连续可导，不用标签预测专家。新增残差初始为严格0，并限制为CAM范数的10%以内；无独立疾病分类头。

四模态作用：PHENO48、ANAT6、FUNC10、CORR256全部保留在原GFR/主干；仅新增HASE直接读取CORR256、PHENO年龄/性别及ANAT/FUNC质量条件，且排除站点one-hot。CAM的四个36维位置是跨模态交互潜表示，不是四个原始单模态，也不是ROI。

## 冻结Prompt原文

以下文案是训练前冻结的候选机制综合文本，由本实验请求提供；不声称来自某次LLM API生成，也不构成逐字临床事实。实际计算只由同一节所列的冻结ROI连接掩码进入模型，单独改写这段文字不会改变训练。

### D1：Social-Affective and Communication Circuit

文献：Gotts et al.，Brain 2012，DOI [10.1093/brain/aws160](https://doi.org/10.1093/brain/aws160)。功能解释：候选社交情感与沟通相关功能系统，不是个体缺陷标签。 候选解剖端点共63个；该域72条连接均来自已核验的252个CORR特征槽，具体边及索引见 `mechanism_masks.json`。

候选ROI端点：Left Amygdala；Left Angular Gyrus；Left Caudate；Left Central Opercular Cortex；Left Cingulate Gyrus, posterior division；Left Cuneal Cortex；Left Frontal Operculum Cortex；Left Frontal Orbital Cortex；Left Frontal Pole；Left Heschl's Gyrus (includes H1 and H2)；Left Hippocampus；Left Inferior Frontal Gyrus, pars triangularis；Left Inferior Temporal Gyrus, posterior division；Left Inferior Temporal Gyrus, temporooccipital part；Left Insular Cortex；Left Lateral Occipital Cortex, superior division；Left Lingual Gyrus；Left Middle Temporal Gyrus, anterior division；Left Middle Temporal Gyrus, posterior division；Left Middle Temporal Gyrus, temporooccipital part；Left Occipital Pole；Left Pallidum；Left Paracingulate Gyrus；Left Parahippocampal Gyrus, anterior division；Left Planum Polare；Left Planum Temporale；Left Superior Frontal Gyrus；Left Superior Parietal Lobule；Left Superior Temporal Gyrus, anterior division；Left Superior Temporal Gyrus, posterior division；Left Supracalcarine Cortex；Left Supramarginal Gyrus, anterior division；Left Supramarginal Gyrus, posterior division；Left Temporal Pole；Left Thalamus；Right Amygdala；Right Angular Gyrus；Right Central Opercular Cortex；Right Cingulate Gyrus, anterior division；Right Cuneal Cortex；Right Frontal Operculum Cortex；Right Frontal Pole；Right Heschl's Gyrus (includes H1 and H2)；Right Inferior Frontal Gyrus, pars opercularis；Right Lateral Occipital Cortex, superior division；Right Middle Frontal Gyrus；Right Middle Temporal Gyrus, anterior division；Right Middle Temporal Gyrus, posterior division；Right Middle Temporal Gyrus, temporooccipital part；Right Pallidum；Right Parahippocampal Gyrus, anterior division；Right Parietal Operculum Cortex；Right Planum Polare；Right Planum Temporale；Right Precentral Gyrus；Right Superior Temporal Gyrus, anterior division；Right Superior Temporal Gyrus, posterior division；Right Supramarginal Gyrus, anterior division；Right Supramarginal Gyrus, posterior division；Right Temporal Fusiform Cortex, anterior division；Right Temporal Occipital Fusiform Cortex；Right Temporal Pole；Right Thalamus。

> Autism spectrum presentations may involve heterogeneous alterations in functional coupling among limbic, temporal, frontal, and parietal regions associated with social-affective processing and communication. These alterations may differ across individuals and should not be assumed to have one universal direction. Examine candidate connectivity patterns only within anatomically verified region pairs available in the current dataset; the presence of a candidate pattern is neither a confirmed lesion nor a definitive ASD diagnosis.

解释限制：不主张ASD连接统一增强或减弱；不代表个体病灶、症状量表、因果通路或完整经典社交网络。

### D2：Frontoparietal Executive-Control Circuit

文献：Just et al.，Cerebral Cortex 2007，DOI [10.1093/cercor/bhl006](https://doi.org/10.1093/cercor/bhl006)。功能解释：候选额-顶叶执行与整合系统；任务态结果不外推为静息态方向。 候选解剖端点共54个；该域56条连接均来自已核验的252个CORR特征槽，具体边及索引见 `mechanism_masks.json`。

候选ROI端点：Left Angular Gyrus；Left Caudate；Left Cingulate Gyrus, posterior division；Left Frontal Operculum Cortex；Left Frontal Orbital Cortex；Left Frontal Pole；Left Heschl's Gyrus (includes H1 and H2)；Left Inferior Frontal Gyrus, pars opercularis；Left Inferior Frontal Gyrus, pars triangularis；Left Inferior Temporal Gyrus, anterior division；Left Inferior Temporal Gyrus, posterior division；Left Inferior Temporal Gyrus, temporooccipital part；Left Insular Cortex；Left Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)；Left Lingual Gyrus；Left Middle Frontal Gyrus；Left Middle Temporal Gyrus, posterior division；Left Middle Temporal Gyrus, temporooccipital part；Left Parahippocampal Gyrus, anterior division；Left Planum Temporale；Left Postcentral Gyrus；Left Precuneous Cortex；Left Superior Frontal Gyrus；Left Superior Parietal Lobule；Left Superior Temporal Gyrus, anterior division；Left Supracalcarine Cortex；Left Supramarginal Gyrus, posterior division；Left Temporal Fusiform Cortex, posterior division；Left Temporal Pole；Right Angular Gyrus；Right Caudate；Right Cingulate Gyrus, anterior division；Right Cuneal Cortex；Right Frontal Medial Cortex；Right Frontal Orbital Cortex；Right Frontal Pole；Right Heschl's Gyrus (includes H1 and H2)；Right Hippocampus；Right Inferior Frontal Gyrus, pars opercularis；Right Inferior Frontal Gyrus, pars triangularis；Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)；Right Middle Frontal Gyrus；Right Middle Temporal Gyrus, anterior division；Right Middle Temporal Gyrus, posterior division；Right Parahippocampal Gyrus, posterior division；Right Planum Polare；Right Subcallosal Cortex；Right Superior Frontal Gyrus；Right Superior Parietal Lobule；Right Superior Temporal Gyrus, anterior division；Right Superior Temporal Gyrus, posterior division；Right Supracalcarine Cortex；Right Supramarginal Gyrus, posterior division；Right Temporal Fusiform Cortex, posterior division。

> Differences in the functional coordination of frontal and parietal systems may contribute to variation in executive and integrative processing among autistic individuals. Prior task-fMRI observations of reduced frontoparietal coupling must not be transferred into a fixed directional label for resting-state functional connectivity. Represent this mechanism using verified frontal and parietal anatomical candidates, allowing patient-specific positive, negative, or near-typical connectivity deviations.

解释限制：不把任务态低连接解释成静息态固定方向；没有个体执行功能测量或临床亚型；掩码不是已验证的完整额顶网络。

### D3：Thalamocortical and Sensory-Integration Circuit

文献：Nair et al.，Brain 2013，DOI [10.1093/brain/awt079](https://doi.org/10.1093/brain/awt079)。功能解释：候选丘脑-皮层信息整合系统，保留个体FC符号。 候选解剖端点共42个；该域45条连接均来自已核验的252个CORR特征槽，具体边及索引见 `mechanism_masks.json`。

候选ROI端点：Left Central Opercular Cortex；Left Cingulate Gyrus, anterior division；Left Frontal Operculum Cortex；Left Heschl's Gyrus (includes H1 and H2)；Left Lateral Occipital Cortex, inferior division；Left Lateral Occipital Cortex, superior division；Left Middle Temporal Gyrus, anterior division；Left Middle Temporal Gyrus, temporooccipital part；Left Paracingulate Gyrus；Left Parahippocampal Gyrus, posterior division；Left Postcentral Gyrus；Left Precentral Gyrus；Left Putamen；Left Superior Frontal Gyrus；Left Superior Temporal Gyrus, anterior division；Left Supracalcarine Cortex；Left Supramarginal Gyrus, posterior division；Left Temporal Fusiform Cortex, posterior division；Left Temporal Occipital Fusiform Cortex；Left Thalamus；Right Caudate；Right Central Opercular Cortex；Right Cuneal Cortex；Right Frontal Medial Cortex；Right Frontal Orbital Cortex；Right Frontal Pole；Right Heschl's Gyrus (includes H1 and H2)；Right Inferior Temporal Gyrus, anterior division；Right Intracalcarine Cortex；Right Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)；Right Middle Temporal Gyrus, anterior division；Right Middle Temporal Gyrus, temporooccipital part；Right Occipital Fusiform Gyrus；Right Parahippocampal Gyrus, anterior division；Right Planum Temporale；Right Postcentral Gyrus；Right Precentral Gyrus；Right Subcallosal Cortex；Right Superior Parietal Lobule；Right Temporal Fusiform Cortex, anterior division；Right Temporal Occipital Fusiform Cortex；Right Thalamus。

> Heterogeneous thalamocortical connectivity patterns may be associated with differences in sensory integration and large-scale information communication in autism. The relevant alterations may involve multiple cortical targets and should not be reduced to a single universally increased or decreased connection. Assess this candidate mechanism through anatomically verified thalamic-cortical connections and preserve the signed variation of each participant's observed resting-state connectivity.

解释限制：没有单条已验证的ASD标志连接、个体感觉表型或因果丘脑皮层通路；掩码不是完整丘脑皮层图谱。

### D4：Insular-Cingulate and Salience-Related Circuit

文献：Uddin et al.，JAMA Psychiatry 2013，DOI [10.1001/jamapsychiatry.2013.104](https://doi.org/10.1001/jamapsychiatry.2013.104)。功能解释：候选岛叶-扣带与注意相关组织，保留年龄和人群异质性。 候选解剖端点共35个；该域32条连接均来自已核验的252个CORR特征槽，具体边及索引见 `mechanism_masks.json`。

候选ROI端点：Left Cingulate Gyrus, anterior division；Left Cingulate Gyrus, posterior division；Left Frontal Medial Cortex；Left Frontal Operculum Cortex；Left Inferior Frontal Gyrus, pars triangularis；Left Inferior Temporal Gyrus, temporooccipital part；Left Insular Cortex；Left Middle Temporal Gyrus, anterior division；Left Middle Temporal Gyrus, posterior division；Left Parahippocampal Gyrus, anterior division；Left Subcallosal Cortex；Left Superior Frontal Gyrus；Left Superior Temporal Gyrus, posterior division；Left Supracalcarine Cortex；Left Supramarginal Gyrus, posterior division；Left Temporal Fusiform Cortex, posterior division；Right Cingulate Gyrus, anterior division；Right Cingulate Gyrus, posterior division；Right Cuneal Cortex；Right Frontal Operculum Cortex；Right Frontal Pole；Right Heschl's Gyrus (includes H1 and H2)；Right Inferior Frontal Gyrus, pars opercularis；Right Inferior Temporal Gyrus, anterior division；Right Inferior Temporal Gyrus, temporooccipital part；Right Insular Cortex；Right Lateral Occipital Cortex, superior division；Right Middle Frontal Gyrus；Right Middle Temporal Gyrus, temporooccipital part；Right Occipital Pole；Right Parahippocampal Gyrus, posterior division；Right Planum Temporale；Right Supramarginal Gyrus, anterior division；Right Temporal Fusiform Cortex, anterior division；Right Thalamus。

> Atypical coordination among insular, cingulate, and associated attentional systems may contribute to heterogeneous salience-related functional organization in autism. Existing findings are population-level and age-dependent; they do not establish an individual lesion or a universal connectivity direction. Use only verified anatomical connectivity candidates involving the relevant regions, and do not equate broad atlas label matching with confirmed canonical salience-network membership.

解释限制：宽泛图谱名称不等于经典显著性网络成员；不推断个体病灶、固定方向、年龄轨迹或临床亚型；掩码不是完整显著性网络。

|组|固定结构|主要对照|
|---|---|---|
|E0_B0A|原始GFR + CAM + OutputLayer，完全移除SPGS/RES|新基线|
|E1_CAPACITY|一个700输入、宽度183的普通无掩码Adapter|参数容量|
|E2_GENERIC_MOE|4个专家读取全256 CORR，动态路由|普通MoE|
|E3_MED_UNIFORM|真实冻结连接掩码，权重固定0.25|动态路由必要性|
|E4_MED_GFR_ROUTE|真实机制FC输入，路由只看GFR/CAM上下文+专家身份|机制响应条件路由|
|E5_WRONG_MASK|仅在252条端点已核验连接上做seed39全局置换，保持大小/重叠|正确解剖对应|
|E6_DATA_DRIVEN|仅在252条已核验连接上做训练折FC相关剖面KMeans，保持重叠结构|文献先验对照|
|E7_PATIENT_MISMATCH|训练/测试池内独立无固定点患者错配|患者对应|
|E8_NO_SPEC|完整结构但关闭L_spec|专门化约束|
|E9_HASE_FULL|真实四机制 + GFR/FC/PHENO/QC路由 + L_spec|完整模型|

E9新增可训练参数154,816；E1宽183的单Adapter为154,452，容量差0.236%。E2–E9使用相同四专家参数结构，改变预注册的掩码、错配、路由或专门化开关；原模型参数保持weight_decay=0.11，新增模块使用独立Adam参数组weight_decay=0。逐折注册参数与实际非零梯度参数量、梯度诊断见 `fold_metrics.csv` / `diagnostics.csv`。

逐组参数审计（注册新增参数、至少一次获得非零梯度的新增参数、有效参数总量及活跃新增张量数）如下；原始主干注册1,462,957参数，实际有效梯度主干约732,238参数、49个活跃张量。固定均匀路由E3的Router没有梯度，因此其有效新增参数为147,456，而注册数仍154,816。E0中PEN路由旁路参数未获梯度。

|组别|注册新增参数|有非零梯度新增参数|有效训练参数总量|新增非零梯度张量/注册张量|
|---|---:|---:|---:|---:|
|E0_B0A|0|0|732,238|0/0|
|E1_CAPACITY|154,452|154,452|886,690|2/2|
|E2_GENERIC_MOE|154,816|154,816|887,054|15/15|
|E3_MED_UNIFORM|154,816|147,456|879,694|12/15|
|E4_MED_GFR_ROUTE|154,816|154,816|887,054|15/15|
|E5_WRONG_MASK|154,816|154,816|887,054|15/15|
|E6_DATA_DRIVEN|154,816|154,816|887,054|15/15|
|E7_PATIENT_MISMATCH|154,816|154,816|887,054|15/15|
|E8_NO_SPEC|154,816|154,816|887,054|15/15|
|E9_HASE_FULL|154,816|154,816|887,054|15/15|

## 正式指标

均值±十折population标准差，单位%。正式分类指标只有ACC、原代码one-hot硬预测ROC-AUC、SPE-TD、SEN-ASD。二分类下硬标签AUC等于BA，没有另算概率AUC、F1或其他正式指标。

|组别|ACC|Hard-AUC|SPE-TD|SEN-ASD|
|---|---:|---:|---:|---:|
|E0_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|94.6300 ± 4.0457|89.0793 ± 3.7283|
|E1_CAPACITY|90.4663 ± 2.5824|90.3561 ± 2.6314|91.6512 ± 3.4232|89.0610 ± 4.2502|
|E2_GENERIC_MOE|90.3514 ± 3.3862|90.3302 ± 3.2951|90.5689 ± 5.6488|90.0915 ± 4.1087|
|E3_MED_UNIFORM|87.3615 ± 3.9591|87.3181 ± 3.9965|88.0204 ± 5.6844|86.6159 ± 6.7266|
|E4_MED_GFR_ROUTE|87.0337 ± 4.4209|87.0440 ± 4.3004|86.7160 ± 7.4640|87.3720 ± 5.6142|
|E5_WRONG_MASK|84.8485 ± 5.8678|84.6515 ± 5.8980|87.1323 ± 6.2018|82.1707 ± 7.5147|
|E6_DATA_DRIVEN|85.9835 ± 5.2223|85.9567 ± 5.2250|86.5171 ± 6.4528|85.3963 ± 6.8588|
|E7_PATIENT_MISMATCH|89.7766 ± 4.3518|89.6959 ± 4.2792|90.5504 ± 6.6164|88.8415 ± 4.8778|
|E8_NO_SPEC|85.9862 ± 3.9889|85.8074 ± 3.9576|88.0111 ± 6.9960|83.6037 ± 7.0005|
|E9_HASE_FULL|85.7654 ± 4.1109|85.6017 ± 3.9648|88.0389 ± 6.7336|83.1646 ± 5.0081|

## 配对比较

Δ为E9减对照，单位百分点；W/T/L为同一fold严格胜/平/负。10折训练集合相互重叠，不作为独立样本或显著性检验。筛选阈值预先冻结为E9相对E0/E2/E5/E6/E7在ACC和Hard-AUC平均均>0.2pp且各至少7/10折严格提高，同时不以ASD灵敏度明显下降换取准确率。

|E9减对照|ACC Δpp (胜/平/负)|Hard-AUC Δpp (胜/平/负)|SPE-TD Δpp|SEN-ASD Δpp|
|---|---:|---:|---:|---:|
|E9 − E0_B0A|-6.3088 (0/0/10)|-6.2529 (0/0/10)|-6.5911|-5.9146|
|E9 − E1_CAPACITY|-4.7009 (1/0/9)|-4.7544 (1/0/9)|-3.6124|-5.8963|
|E9 − E2_GENERIC_MOE|-4.5859 (0/1/9)|-4.7284 (1/0/9)|-2.5301|-6.9268|
|E9 − E3_MED_UNIFORM|-1.5961 (4/2/4)|-1.7164 (4/1/5)|+0.0185|-3.4512|
|E9 − E4_MED_GFR_ROUTE|-1.2683 (4/1/5)|-1.4422 (4/0/6)|+1.3228|-4.2073|
|E9 − E5_WRONG_MASK|+0.9169 (5/2/3)|+0.9502 (5/1/4)|+0.9066|+0.9939|
|E9 − E6_DATA_DRIVEN|-0.2181 (5/0/5)|-0.3550 (5/0/5)|+1.5217|-2.2317|
|E9 − E7_PATIENT_MISMATCH|-4.0112 (1/0/9)|-4.0942 (1/0/9)|-2.5116|-5.6768|
|E9 − E8_NO_SPEC|-0.2207 (3/2/5)|-0.2056 (3/0/7)|+0.0278|-0.4390|

筛选判定逐项：`{"E0_B0A": {"acc_delta_pp": -6.30877742946709, "acc_gain_gt_0_2pp": false, "acc_wins": 0, "acc_wins_at_least_7": false, "auc_delta_pp": -6.252876740145764, "auc_gain_gt_0_2pp": false, "auc_wins": 0, "auc_wins_at_least_7": false}, "E2_GENERIC_MOE": {"acc_delta_pp": -4.585945663531875, "acc_gain_gt_0_2pp": false, "acc_wins": 0, "acc_wins_at_least_7": false, "auc_delta_pp": -4.728447011574658, "auc_gain_gt_0_2pp": false, "auc_wins": 1, "auc_wins_at_least_7": false}, "E5_WRONG_MASK": {"acc_delta_pp": 0.9169278996865193, "acc_gain_gt_0_2pp": true, "acc_wins": 5, "acc_wins_at_least_7": false, "auc_delta_pp": 0.9502352158119143, "auc_gain_gt_0_2pp": true, "auc_wins": 5, "auc_wins_at_least_7": false}, "E6_DATA_DRIVEN": {"acc_delta_pp": -0.21812957157784765, "acc_gain_gt_0_2pp": false, "acc_wins": 5, "acc_wins_at_least_7": false, "auc_delta_pp": -0.3549840933191983, "auc_gain_gt_0_2pp": false, "auc_wins": 5, "auc_wins_at_least_7": false}, "E7_PATIENT_MISMATCH": {"acc_delta_pp": -4.011233019853713, "acc_gain_gt_0_2pp": false, "acc_wins": 1, "acc_wins_at_least_7": false, "auc_delta_pp": -4.094196317772616, "auc_gain_gt_0_2pp": false, "auc_wins": 1, "auc_wins_at_least_7": false}}`。ACC标准差范围与SEN-ASD变化同时报告；本判定是单seed内部探索门槛，不是论文级显著性标准。

## 机制诊断

E9折均值诊断：`{"mean_weight": [0.3739, 0.2371, 0.2006, 0.1884], "between_subject_sd": [0.2353, 0.1774, 0.1726, 0.1708], "top_route_fraction": [0.4717, 0.2078, 0.1689, 0.1516], "entropy": 1.0073795735836029, "expert_cosines": [-0.0783, 0.0029, -0.053, 0.0399, 0.0234, -0.0147], "residual_norms": [12.60954, 11.42682, 10.44676, 9.99863], "mismatch_response_change": 1.2670416474342345, "max_residual_ratio": 0.10000002384185791}`。平均路由使用率只描述模型分配，不等于临床亚型；ASD/TD内部路由均值是训练后按测试标签作描述性汇总，不参与推断。站点、年龄分箱、性别及QC关联为队列描述，不能当病理关联。专家响应错配诊断、掩码覆盖、中心化余弦、残差幅度及逐折梯度见 `diagnostics.csv`。

按专家展开的描述性路由与条件关联：

|专家|平均权重|ASD平均权重|TD平均权重|ASD/TD top-route比例|FC响应-路由Spearman|站点η²|年龄箱η²|性别η²|平均QC绝对ρ|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|D1|0.3739|0.3405|0.4036|0.424/0.514|+0.248|0.252|0.171|0.007|0.104|
|D2|0.2371|0.2558|0.2210|0.248/0.173|+0.165|0.246|0.179|0.011|0.096|
|D3|0.2006|0.2154|0.1876|0.184/0.156|+0.016|0.231|0.136|0.017|0.107|
|D4|0.1884|0.1883|0.1879|0.144/0.158|+0.168|0.208|0.165|0.010|0.110|

E9四专家测试集中心化余弦均值（D1-D2、D1-D3、D1-D4、D2-D3、D2-D4、D3-D4）为`[-0.0783, 0.0029, -0.053, 0.0399, 0.0234, -0.0147]`，各专家平均残差范数为`[12.60954, 11.42682, 10.44676, 9.99863]`，每折平均残差/CAM范数比例为0.0990，最坏值0.1000。新增模块15/15张量均实际获得非零梯度，患者机制FC错配使响应范数平均相对变化1.267；这些检查说明机制输入进入了计算图，但不证明专家代表临床机制。E9的训练L_spec均值为0.01173；E8中该项为0。E8的专家余弦和E9同样接近0，E9相对E8的ACC与Hard-AUC反而分别低-0.2207和-0.2056个百分点，未显示专门化约束必要性。

### 固定checkpoint干预

以下所有干预直接作用于E9每折的同一最佳checkpoint；不重训、不重选epoch。差值为百分点，flip为测试预测翻转比例。

|固定干预|ΔACC pp|ΔHard-AUC pp|ΔSPE pp|ΔSEN pp|翻转率|
|---|---:|---:|---:|---:|---:|
|ALL_OFF|-7.4530|-7.5829|-6.1841|-8.9817|0.1571|
|EXPERT_OFF_1|-5.3801|-5.2599|-7.6966|-2.8232|0.1111|
|EXPERT_OFF_2|-3.2053|-3.4478|-0.2128|-6.6829|0.0642|
|EXPERT_OFF_3|-4.0034|-4.1324|-2.1184|-6.1463|0.0630|
|EXPERT_OFF_4|-3.7866|-3.6921|-4.4635|-2.9207|0.0608|
|PATIENT_SHUFFLE|-12.1460|-12.3292|-9.5791|-15.0793|0.1788|
|UNIFORM_ROUTE|-3.3229|-3.3197|-3.8529|-2.7866|0.0951|
|WRONG_MASK|-7.7887|-7.7935|-7.4468|-8.1402|0.1375|

`ALL_OFF`仅说明E9训练出的主干对新增残差的checkpoint依赖，不能替代从头训练E0/E2/E5/E6/E7。wrong-mask或patient-shuffle同样是模型敏感性诊断，不构成连接因果性或亚型证据。

## 科学解释与决定

判定为 **NO-GO**。完整创新链需要同一数据上的异质机制响应、正确解剖映射优于容量匹配/普通MoE/错误映射/无监督集合/患者错配、且患者条件路由与专家差异带来相互支持的可复现增益。以上预注册筛选门槛任一失败即NO-GO；即使GO也只是单seed内部候选，不代表MICCAI 2027论文成功。路由权重、梯度非零或固定checkpoint干预均不能单独证明医学语义有效。

相关工作边界（文献检索截至2026-10-26）：SPromptGL已包含多模态患者关系图与文本提示局部筛选；BrainPrompt (MICCAI 2025) 已研究ROI/subject/disease多级文本提示，因此多级提示本身不足以构成新增点。PhenoBrain (ICML 2026) 已以表型条件调节长程通信。KD-Brain (MICCAI 2026) 已把语义先验注入子网络交互query并引入pathology-consistent约束。因此HASE可能的增量只能是：将可核验ROI连接集合固定为真实FC专家的唯一FC输入，再由患者自身的真实机制响应参与软路由，并以错误解剖/患者错配和同checkpoint干预检验机制必要性；单说“语义路由/条件MoE”不新颖。NeurIPS 2025专家专门化研究和后续MoE去冗余正则说明响应差异本身需验证，本轮L_spec只是辅助约束，E8用于判断它的必要性。

进一步的边界：2026 BrainMoE已展示模态专属EEG/fMRI专家、共享状态专家和样本级路由，因而条件MoE与专家融合本身不是充分新颖点；KD-Brain已把文本条件放进子网络交互并用先验分布约束注意力；PhenoBrain已将表型用于个体化长程路由。HASE在概念上较窄的区别是“固定、可核验连接掩码限制FC专家输入”，但本轮E9未胜过普通MoE、错误掩码或外层训练折无监督FC集合，不能主张该区别带来诊断收益。已发表的ASD功能亚型工作使用更大样本、规范化建模及独立行为/任务表现评估；本轮ABIDE特征没有症状量表，专家路由不能改称亚型。

既有ASD影像亚型研究报告跨人群功能连接异质性，例如Nature Neuroscience的个体差异/网络机制分析及2023–2025 ABIDE聚类工作。但HASE的D1–D4是候选功能系统专家，不是已验证患者亚型；本实验数据没有症状量表来验证专家分配的临床意义。若核心对照失败，应保持NO-GO。

## 结果解释与可能根因

E9的ACC比E0低6.3088个百分点（逐折0/0/10胜/平/负），比容量匹配E1低4.7009个百分点，也比普通MoE E2低4.5859个百分点。E9相对E0和E2的Hard-AUC分别低6.2529和4.7284个百分点，SEN-ASD分别低5.9146和6.9268个百分点；ASD和TD识别均恶化，并非用牺牲特异度换准确率。E9相对固定均匀路由E3的ACC平均差为-1.5961个百分点，逐折仅4/10胜出；动态路由没有解决整体负增益。

正确掩码的必要性不成立：E9比E5错误解剖掩码仅高+0.9169个百分点ACC，逐折5/2/3胜/平/负；E9比E6训练折无监督FC分组低0.2181个百分点，比训练时患者FC错配E7低4.0112个百分点。因此本轮没有证明正确医学域或患者对应在从头训练中优于反事实控制。E9比仅用GFR/CAM上下文路由的E4低1.2683个百分点，额外FC响应、PHENO年龄/性别及ANAT/FUNC质量条件没有形成可用增益；E9相对E8关闭L_spec也低0.2207个百分点ACC，专门化约束未显示必要性。

固定checkpoint中，ALL_OFF使ACC平均下降7.4530个百分点，PATIENT_SHUFFLE下降12.1460个百分点，WRONG_MASK下降7.7887个百分点。这说明已训练E9的预测依赖新增支路、患者FC和训练时掩码；同一checkpoint对输入扰动敏感不等于重训方法优于E0，也不能证明真实掩码的医学正确性。E9自身的残差关闭结果仍远低于E0，符合新增路径联合训练后改变共享主干解的可能性。

较符合数据的解释是：新增FC编码器把已预处理的高维CORR直接送入CAM后的残差Adapter，在871名受试者上增加了154,816个可训练参数；E9平均残差/CAM范数比达到9.90%，接近每折10%的上限。正则化只约束四个残差的相似度，不保证它们携带互补的疾病信息；E8与E9均呈近零中心化余弦，但E9性能略差。E9路由与站点的平均η²约0.234、年龄箱约0.162、性别约0.011，且D3路由与对应FC响应范数的Spearman相关约0.016。新增路由因此可能部分利用队列或采集分布，或只依赖部分专家响应，尚不足以解释为四种医学机制。E1普通Adapter和E2普通MoE都优于E9，E5错误掩码与E9接近，E6/E7也未输给E9；整体上更支持容量、共享主干优化和一般FC信号影响，而不是正确语义映射收益。以上是与观测相符的候选数学/优化根因，不是单seed实验识别的因果结论；需要严格无泄漏外层预处理、独立多中心验证及重复seed进一步区分。

## 可复现记录

完整原始fold指标、最佳epoch、耗时、入口RNG和初始化哈希位于 `fold_metrics.csv`；路由/专家机制诊断位于 `diagnostics.csv`；固定checkpoint干预位于 `counterfactual.csv`；完整paired差位于 `paired_comparisons.json`；manifest记录prompt、源代码、输入、映射、fold和结果哈希。GPU分配严格按GPU0 E0/E2/E4/E6/E8及GPU1 E1/E3/E5/E7/E9；每卡最多4 worker，每worker2 CPU线程。每配置按seed0播种一次，后续fold连续消费随机流；十组fold1入口RNG和主干初始化一致，但不同早停长度导致之后fold的入口RNG分流，表中逐折记录，不宣称跨组后续初始化完全一致。

E0历史核验：`{"acc": {"matched_folds": 10, "of": 10}, "all_fields_match": true, "auc": {"matched_folds": 10, "of": 10}, "best_epoch": {"matched_folds": 10, "of": 10}, "initial_rng_sha256": {"matched_folds": 10, "of": 10}, "initial_trunk_sha256": {"matched_folds": 10, "of": 10}, "sen_asd": {"matched_folds": 10, "of": 10}, "spe_td": {"matched_folds": 10, "of": 10}}`。

源码SHA256：
- `ABIDE-simple-2-concat-weighted-cosine.sh`: `28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763`
- `layers.py`: `4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a`
- `main.py`: `dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2`
- `model.py`: `254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda`
- `network.py`: `484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec`
- `utils.py`: `95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67`

数据与历史映射/参照SHA256：
- `domain_mapping.csv`: `cd34776253b961f37112d52d588f8e086139446326b93df5b237972821e0c9bf`
- `exp32_fold_metrics.csv`: `0133bb2fd5810beeb777c54b923c8d9b6e9ae2211ed225116ca7db2e3609ffec`
- `modal_feat_dict.npy`: `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`
- `processed_standard_data.csv`: `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`
- `roi_256_named_mapping.csv`: `b383308bb413a94d1646511a9bea5bc16f54c975b938ac4d947d676e357c5eb2`
- `source_paper_pdf`: `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`

训练代码和冻结prompt的SHA256见 `manifest.json`。单卡任务累计秒数：`{"E0_B0A": 367.93696182221174, "E1_CAPACITY": 331.75092401355505, "E2_GENERIC_MOE": 306.8365076817572, "E3_MED_UNIFORM": 363.647519800812, "E4_MED_GFR_ROUTE": 363.5077529028058, "E5_WRONG_MASK": 393.56126579269767, "E6_DATA_DRIVEN": 374.2770184241235, "E7_PATIENT_MISMATCH": 410.7783379778266, "E8_NO_SPEC": 382.7750914916396, "E9_HASE_FULL": 395.6566394530237}`。正式模型checkpoint在训练、反事实和报告验收后已清理；恢复诊断记录、简短worker日志及其余实验源文件保留在独立目录`/tmp/SPromptGL-E39-HASE-stage/experiment/experiment39_heterogeneity_aware_semantic_experts`。根目录包同步受当前OSSFS写入错误阻断，暂不将此临时位置描述为正式工作区目录。

文献：Gotts et al. 2012, DOI 10.1093/brain/aws160; Just et al. 2007, DOI 10.1093/cercor/bhl006; Nair et al. 2013, DOI 10.1093/brain/awt079; Uddin et al. 2013, DOI 10.1001/jamapsychiatry.2013.104. BrainPrompt (MICCAI 2025): https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf . PhenoBrain (ICML 2026): https://proceedings.mlr.press/v306/meng26f.html . KD-Brain (MICCAI 2026): https://papers.miccai.org/miccai-2026/0366-Paper0073.html . BrainMoE (J Vis Exp 2026), DOI 10.3791/73432: https://pubmed.ncbi.nlm.nih.gov/42799620/ . NeurIPS 2025, Advancing Expert Specialization for Better MoE: https://papers.nips.cc/paper_files/paper/2025/hash/4598de7d243d528e38eb0c5d8155fb52-Abstract-Conference.html . ASD subtype studies: https://doi.org/10.1038/s41380-025-03086-x and https://doi.org/10.1038/s41593-026-02287-z .

## 执行恢复与I/O

首轮曾完成100折和90项干预，但因实现错误地在E2–E7也启用了L_spec（冻结协议仅在E9启用），该轮结果全部隔离排除。首轮审计还发现E0初始化RNG构造边界与历史不一致，以及一次诊断钩子恢复事件；首轮材料仅保存在.runtime/first_pass_lspec_audit/，没有复用其中任何指标。随后清空正式任务和结果状态，从seed=0重新完成修正后的100折；E1–E8关闭L_spec、仅E9开启，正式折没有续训恢复或重复计分。E0四项指标、最佳epoch、初始主干哈希和入口RNG均与Experiment32 B0-A逐折10/10匹配。
