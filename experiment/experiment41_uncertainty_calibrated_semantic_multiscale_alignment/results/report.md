# 实验41：UCSMA-v2 完整结果

状态：**COMPLETE，100/100折**；E0历史复现：**PASS（10/10）**。
有效正式折GPU分配：GPU0为E0/E2/E4/E6/E8，共50折；GPU1为E1/E3/E5/E7/E9，共50折。E0独立训练 455.71 秒，初次增量调度 100.11 秒，更正后调度 89.56 秒，含作废运行的训练调度累计墙钟 645.39 秒。静态审计、预检、报告和人工中断不计入该训练耗时。

### 实现更正与排除记录

最终字段/计算图复核发现初次条件参考误取FUNC EFC（ANAT6+FUNC10局部索引6），不是mean FD（局部索引12、全局索引60）；同时概念特征此前没有乘以实际匹配质量，且直接边池化的概念支持没有继承ROI连接度收缩。三项均纠正为原定字段身份、覆盖约束与拒绝约束，未改变Prompt、任何映射、宽度、学习率、阈值、损失、残差上限或GO规则。E3–E9旧70折全部作废，不参与正式统计；更正预检后重算70折。E0/E1/E2保留，E2十折输入逐位一致已核验。总执行170次fold拟合，其中100折为有效正式结果、70折为错误实现的排除记录。GPU0额外作废30折、GPU1额外作废40折。该错误发生在首次正式运行后，是明确的执行偏差，不能写成一次无差错的预注册运行；完整原因、旧结果和代码哈希在`correction_audit.json`。

## 源码与论文审计

本地HEAD `7747a1521629f95413faca133ba8c82ed7f1faf2`；审计时GitHub main HEAD `012ffdb7593981c162c7911649007066dfd59125`。两者不同，以Experiment32冻结的本地源码作为B0-A数学路径；六个核心本地文件SHA256见manifest。

本地原论文 Fig.1、Section 2.1–2.2、Eq.(1)–(7)、ABIDE设置和消融均已阅读；另与公开PDF交叉核对。论文描述的SPPromptGL由GFR、PLN/CAM、PEN构成，但本地实现和shell是实验权威。GFR四支均按患者特征构造患者图；每图的节点特征是完整320维输入；`i==1`覆盖`i==0`写入的`combined_feat`，本实验私有源保持原样。四支GraphConstructLoss均值实际进入`simple-2`反向传播，分类损失使用训练集类别权重；`ALoss`会被计算和打印，但不进入loss。两个原始Adam、lr=0.0038、weight_decay=0.11、dropout=0.35、450 epoch、early=200、`val_idx=test_idx`均按历史入口保留。PEN的SPGS/RES前向被旁路；PLN、CLIP缓存和优化器构造仍保留以维持E0随机初始化副作用。

公式逐项核查：Eq.(1)纸面图目标包含平滑/图约束和标签监督邻接距离，后者对应本地ALoss但实际未优化；Eq.(2)从完整模态拼接进行患者图传播；Eq.(3)为离线疾病概念生成；Eq.(4)为CAM交互；Eq.(5)投影文本与局部潜变量相似后取max；Eq.(6)RES将局部信息回写全局；Eq.(7)联合图/标签损失。本轮只用Eq.(1)/(2)/(4)对应的历史基座路径，旁路SPGS与RES；局部潜变量不是病灶ROI。

原论文ABIDE设置写为dropout=0.75、lr=0.0023、hidden=10、encoding layer=4、head=4，与本地历史shell不同，因此不能把改成论文超参数当作精确复现。原论文Fig.2中ABIDE的GFR、GFR+RES、完整PEN ACC约为91.73%、92.08%、92.30%，Table2完整模型报告AUC为92.22%；这些仅作描述。E32历史B0-A 92.0742%与完整PEN B2 91.7294%才是本地协议参照，正式E0由独立重训产生。Fig.1上部AD示例文本也不能当成本轮ASD知识来源。

CAM输出`[871,4,36]`是跨模态交互后的潜在表征，不是四种原始模态，也不是ROI。PHENO48为年龄箱26维、站点20维、性别2维；ANAT6/FUNC10是已有质量与运动特征，FUNC第7列经字段名核实为mean FD；CORR256为预先RFE并标准化后的rs-fMRI连接特征。其正负值按已有处理输入解释，不解释成神经兴奋或抑制。

实际shell参数为`--GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --datname ABIDE --dropout 0.35 --lr 0.0038 --mode simple-2 --n_head 2 --n_hidden 18 --nlayer 1 --reg 0.11`。

历史审计覆盖Experiment17–19、28–32、34–40：17–18表明强迫CORR/PHENO共享空间可能损伤判别；19–30的CLIP相似度、自由文本投影和静态Query没有稳定独立收益；Experiment38–40已测试规范偏离、疾病文本选边与拒绝/反事实控制，其中E40真实假说ACC 82.8788%、数值Top-|z|对照91.7294%、B0-A 92.0742%。Experiment39正确医学掩码MoE也未胜过容量/随机/数值对照，且强残差风险突出。Experiment34–37未证明语义、条件路由或标签监督的医学语义独立贡献。Experiment41不重复39条选边、自由CLIP文本Query、强制MoE或共享疾病空间。

## 数据与医学映射

数据为本地既有871人、320维、ASD=0 403人、TD=1 468人；没有换样本或重新预处理。Experiment11的256列恢复映射中252条可核验端点连接，4条未知特征索引195、225、232、236保留主干但不进入ROI/语义掩码。252条连接覆盖110/111 ROI；91个ROI度数≥3，19个ROI度数1–2，1个ROI度数0，最高度数12。

未找到可信canonical网络映射，因此仅使用按HO精确名称形成的8种粗粒度解剖区域族代理：感觉运动、额叶关联、顶叶关联、颞叶关联、岛叶、扣带/内侧、视觉、皮质下。它们不等同于Schaefer七网络、DMN、显著性网络或完整脑网络。ROI聚合的正/负偏离分别保留；低度ROI以`degree/(degree+3)`收缩，未观测边不补齐。

## 冻结医学Prompt

未调用GPT-4/API。以下英文候选文本由用户提供，经过来源与ROI身份审查后冻结；自然语言不送入自由文本编码器，实际模型输入是prompts.json/ontology.json中可核验的结构化掩码。

### C01：Sensorimotor and attention-related negative deviations

> At the group level, autistic participants may show increased convergence of negative functional connectivity deviations involving sensorimotor and attention-associated regions. This is a network- or region-level population observation, not a universal reduction in any individual ROI-to-ROI connection.

来源：Ilioska et al., Nature Mental Health (2026)；DOI：10.1038/s44220-026-00656-y。证据尺度：region/network; not a fixed individual edge；方向：negative at population regional-convergence scale only; individual signs remain free；可计算：True；HO ROI数 16，当前252条连接中适用 61 条。

### C02：Insular, temporal and prefrontal regional deviations

> Atypical connectivity deviations may converge on insular, temporal, and prefrontal regions in subsets of autistic individuals. The existence and direction of deviation for a particular subject must be determined from measured connectivity relative to a suitable normative reference, not from the anatomical name alone.

来源：Ilioska et al., Nature Mental Health (2026)；DOI：10.1038/s44220-026-00656-y。证据尺度：region; direction is participant-specific；方向：bidirectional/unspecified；可计算：True；HO ROI数 22，当前252条连接中适用 96 条。

### C03：Medial prefrontal, cingulate and inferior-parietal positive deviations

> Positive normative connectivity deviations have shown increased regional convergence involving medial prefrontal, cingulate and inferior parietal areas in autism research. This population-scale finding should not be translated into a mandatory positive sign for individual observed edges.

来源：Ilioska et al., Nature Mental Health (2026)；DOI：10.1038/s44220-026-00656-y。证据尺度：regional; not a fixed individual edge；方向：positive at population regional-convergence scale only; individual signs remain free；可计算：True；HO ROI数 16，当前252条连接中适用 64 条。

### C04：Default-mode and cognitive-control associations

> Some autistic populations show convergent positive deviations involving default-mode and cognitive-control systems. Apply this concept only if the current ROI-to-system correspondence is independently verified; otherwise retain the concept as literature metadata without asserting precise network membership.

来源：Ilioska et al., Nature Mental Health (2026)；DOI：10.1038/s44220-026-00656-y。证据尺度：canonical-system association；方向：positive at population scale only；可计算：False；HO ROI数 0，当前252条连接中适用 0 条。

### C05：Somatomotor-to-association system negative deviations

> Negative connectivity deviations have been reported across some somatomotor and associative system relationships. These findings are heterogeneous and depend on network definition, age, preprocessing and sampling; anatomical name similarity alone is insufficient to assign an observed connection to a canonical network.

来源：Ilioska et al., Nature Mental Health (2026)；DOI：10.1038/s44220-026-00656-y。证据尺度：canonical-system relationship；方向：negative at population scale only；可计算：False；HO ROI数 0，当前252条连接中适用 0 条。

### C06：Subcortical-cortical deviations

> Atypical subcortical and cortical connectivity may contribute to distributed brain-system differences in autism. The candidate should preserve uncertainty in direction and acknowledge that the current sparse selected-connectivity set may provide incomplete evidence for the underlying system.

来源：Ilioska et al., Nature Mental Health (2026); Kobbersmed et al., Progress in Neurobiology (2026)；DOI：10.1038/s44220-026-00656-y; 10.1016/j.pneurobio.2026.102906。证据尺度：subcortical-cortical edge candidate；方向：bidirectional/unspecified；可计算：True；HO ROI数 14，当前252条连接中适用 55 条。

### C07：Unmatched or insufficient evidence

> No reliable disease-specific semantic correspondence can be asserted for this observed deviation when anatomical coverage, normative reference support, measurement reliability or direct literature support is insufficient. Preserve the observed numeric information without forcing it to match an ASD-related concept.

来源：Experiment 41 pre-registered reject channel；DOI：方法拒绝通道。证据尺度：participant-specific evidence support；方向：not applicable；可计算：True；HO ROI数 0，当前252条连接中适用 0 条。

### 可核验的结构对应

下表均为零基索引；具体左右半球、分区全称可由ontology.json的roi_names逐项核对。C01是感觉运动与顶叶解剖候选代理，并未核验注意网络归属；C02的HO岛叶没有可靠前/后细分，因此仅作宽泛岛叶候选。C03的内侧/扣带/顶叶标签也不构成已验证DMN。

| 概念 | HO ROI索引 | CORR256特征索引 |
|---|---|---|
| C01 | 26, 27, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 64, 65, 97, 98 | 8, 10, 12, 13, 14, 27, 43, 45, 48, 64, 69, 77, 78, 79, 94, 95, 105, 106, 107, 108, 112, 114, 123, 129, 133, 143, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 190, 200, 206, 207, 208, 209, 253, 255 |
| C02 | 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30, 31, 34, 35, 62, 63, 78, 79 | 0, 3, 5, 19, 23, 26, 31, 33, 41, 50, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 138, 157, 197, 198, 199, 200, 201, 202, 203, 204, 205, 231, 233, 234, 235 |
| C03 | 50, 51, 52, 53, 54, 55, 62, 63, 66, 67, 68, 69, 70, 71, 72, 73 | 3, 12, 13, 27, 31, 35, 48, 55, 58, 59, 64, 69, 70, 78, 79, 80, 94, 100, 101, 112, 120, 125, 129, 133, 147, 148, 151, 156, 159, 163, 170, 175, 178, 179, 180, 181, 182, 183, 184, 185, 186, 197, 198, 199, 200, 201, 202, 203, 204, 205, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223 |
| C06 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56 |
| C07 |  |  |
C01–C03和C06为有限、候选的区域或连接集合。C04（DMN/认知控制）与C05（躯体运动—联络系统）因缺少本数据ROI到canonical system的独立对应，只保留文献元数据，不参与建模。C07为未匹配通道。C01负向、C03正向群体描述仅在固定分数15%项中作为弱方向权重（其余方向未知概念对称处理）；个体正、负偏离通道都保留，不把单连接符号转成临床真值。Ilioska et al.报告的是大样本、多尺度规范偏离收敛，不能把其75,855条全连接或高密度分区结果直接外推到本RFE后稀疏子集；Kobbersmed et al.强调系统尺度规范建模，但本数据不具备其全脑覆盖与样本量。

主要文献：[Ilioska et al., Nature Mental Health 2026](https://doi.org/10.1038/s44220-026-00656-y)，[Kobbersmed et al., Progress in Neurobiology 2026](https://doi.org/10.1016/j.pneurobio.2026.102906)。[PhenoBrain](https://proceedings.mlr.press/v306/meng26f.html)已有表型条件化脑网络通信，[KD-Brain](https://papers.miccai.org/miccai-2026/0366-Paper0073.html)已有语义条件子网络交互，[BrainPrompt](https://papers.miccai.org/miccai-2025/0116-Paper2260.html)与[BrainPrompt+](https://doi.org/10.1109/TMI.2026.3692958)已有多层脑提示。因此，本轮不把条件化、层次Prompt或语义注意力本身当作新颖性。[To Align or Not to Align, AAAI 2026](https://doi.org/10.1609/aaai.v40i25.39248)支持按证据避免一概强制对齐。

## 规范偏离与模块

每个外层折的条件参考仅拟合outer-train TD；训练TD与ASD均按同一固定5折内层划分排除自身所在子折，以避免自我参考；测试患者由全部outer-train TD模型预测。条件为年龄箱one-hot、性别、站点one-hot和全局第60列mean FD；年龄箱未被转换成精确年龄。ANAT6/FUNC10共16维质量字段形成稳健距离可靠性代理。Ridge alpha=10；残差尺度为训练TD OOF的1.4826×MAD，尺度下限0.05；连接异常阈值为训练TD OOF经验|z|第95百分位（最低0.5），z截断±4。小站点/年龄条件通过计数支持和ridge leverage降低可信度；计数不足20（年龄/站点）或50（性别）时连续收缩，最低支持0.05，不按测试结果选择站点回退。没有使用测试标签拟合参考、尺度、阈值或Prompt。

连接级输入保持符号、幅度、尾部覆盖、参考可靠性和不确定性；ROI尺度按真实端点取加权平均而非异常总数，再以d/(d+3)降低低度ROI可靠性；区域族尺度只做粗粒度代理。直接概念边池化的支持再乘两端ROI度数收缩的几何均值，未知端点支持为零。概念分数由当前患者正负偏离、尾部覆盖、冻结解剖掩码、规范参考支持和QC支持决定。E9匹配质量为`m_k=raw_k/(1+Σraw)`，未匹配质量为`1/(1+Σraw)`；输入Adapter的概念特征明确为`m_k × concept_features_k`，拒绝后语义贡献为零，原数值证据保留。E6仅将匹配质量强制归一化，使用同一特征门控、剂量和网络。没有自由Text→Query投影。

新分支由固定输入维度2700、隐藏宽度64及输出144的无偏置Adapter构成，共182,016个参数；输出层权重严格零初始化，CAM残差使用`smooth tanh`范数限制为不超过CAM范数5%。原GFR/CAM/OutputLayer参数冻结，且没有新增分类头。E1–E9注册容量相同；实际梯度参数按fold记录。

E0完整保留双Adam与原simple-2。E1–E9的原参数与OutputLayer处于冻结eval状态，只用一个增量Adam更新Adapter（lr=0.0038、weight_decay=0）；目标为原训练索引加权分类损失加缓存图正则常数，不另堆叠对比、蒸馏、正交或语义分类损失。图正则不再对冻结参数产生梯度。分支初始化在RNG fork内使用410000+fold的固定派生流，所有组同折初值相同；E0仍为历史seed0一次播种后连续十折，两种随机协议明确不同。

## 实验组与正式指标

E0独立复现B0-A；E1原始CORR容量控制（固定随机投影/重复填充相同输入宽度）；E2全局TD规范参考；E3表型/站点/meanFD条件参考；E4连接+ROI覆盖汇聚；E5再加数值解剖区域族多尺度；E6真实概念但强制匹配；E7固定seed4107随机已核验连接集合，匹配逐概念边数；E8仅在outer-train拟合|FC相关|轮廓的4簇k-means，seed41041、n_init=20；E9真实候选映射及显式拒绝。E7的ROI度数分层置换仅为元数据，实际边集合不保证相同ROI覆盖或边重叠；E8簇大小和交叠也不匹配医学先验。不能声称二者是完全等覆盖对照。E1–E9具有同一注册Adapter，但关闭证据块会降低输入连接的有效容量。

正式指标只有ACC、原one-hot硬标签ROC-AUC、SPE-TD、SEN-ASD，单位均为百分比；二分类hard-label AUC等于balanced accuracy，不是概率排序AUC。均值±population SD为十折描述统计。

| 组别 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| E0_B0A | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E1_RAW_CAPACITY | 91.9592 ± 2.5255 | 91.7482 ± 2.5067 | 94.4172 ± 4.0374 | 89.0793 ± 3.7283 |
| E2_GLOBAL_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E3_COND_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E4_ROI_NORM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E5_MULTISCALE_NUM | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E6_REAL_NO_REJECT | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E7_WRONG_SEMANTICS | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E8_DATA_PRIOR | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |
| E9_UCSMA_FULL | 92.0742 ± 2.6000 | 91.8546 ± 2.5815 | 94.6300 ± 4.0457 | 89.0793 ± 3.7283 |

## 配对比较

Δ为E9减对照，单位百分点；W/T/L为相同fold严格胜/平/负。十折训练集重叠，折数不是独立队列，配对数仅作描述，不作显著性检验。

| E9 − 对照 | ACC Δ / W-T-L | Hard-AUC Δ / W-T-L | SPE Δ | SEN Δ |
|---|---:|---:|---:|---:|
| E0_B0A | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E1_RAW_CAPACITY | +0.1149 / 1-9-0 | +0.1064 / 1-9-0 | +0.2128 | +0.0000 |
| E2_GLOBAL_NORM | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E3_COND_NORM | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E4_ROI_NORM | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E5_MULTISCALE_NUM | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E6_REAL_NO_REJECT | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E7_WRONG_SEMANTICS | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |
| E8_DATA_PRIOR | +0.0000 / 0-10-0 | +0.0000 / 0-10-0 | +0.0000 | +0.0000 |

各核心比较的逐折差（折1→10，单位pp；四项顺序ACC / Hard-AUC / SPE-TD / SEN-ASD）：

- E9−E0_B0A：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E1_RAW_CAPACITY：[+0.0000, +0.0000, +0.0000, +0.0000, +1.1494, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +1.0638, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +2.1277, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E4_ROI_NORM：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E5_MULTISCALE_NUM：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E6_REAL_NO_REJECT：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E7_WRONG_SEMANTICS：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]
- E9−E8_DATA_PRIOR：[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000] / [+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]

E3−E2（条件参考）、E4−E3（ROI汇聚）、E5−E4（区域族汇聚）的四项逐折差均为零；因此训练TD参考拟合改善并未转化为可观测分类增益。

### 参数与实际梯度

所有新增组均注册182,016参数，但不是所有输入列都激活。下表对每个最佳checkpoint额外进行一次训练索引反向诊断，不更新优化器、不重选checkpoint；有效输入连接参数=训练样本上非零输入列×64+64×144，梯度参数按非零梯度元素计数。初次E1/E2训练日志中的182,016是整个非零梯度张量的参数总数，不能用于证明有效容量相同；本表统一采用元素级诊断。

| 组别 | 注册新增参数 | 非零输入列范围 | 输入连接参数范围 | 最佳checkpoint非零梯度元素范围 |
|---|---:|---:|---:|---:|
| E1_RAW_CAPACITY | 182016 | 2700–2700 | 182016–182016 | 182016–182016 |
| E2_GLOBAL_NORM | 182016 | 1680–1680 | 116736–116736 | 112416–116736 |
| E3_COND_NORM | 182016 | 1680–1680 | 116736–116736 | 113424–116736 |
| E4_ROI_NORM | 182016 | 2340–2340 | 158976–158976 | 155793–158976 |
| E5_MULTISCALE_NUM | 182016 | 2388–2388 | 162048–162048 | 157655–162048 |
| E6_REAL_NO_REJECT | 182016 | 2416–2416 | 163840–163840 | 159259–163840 |
| E7_WRONG_SEMANTICS | 182016 | 2417–2417 | 163904–163904 | 158920–163904 |
| E8_DATA_PRIOR | 182016 | 2417–2417 | 163904–163904 | 159199–163904 |
| E9_UCSMA_FULL | 182016 | 2417–2417 | 163904–163904 | 158758–163904 |

E1使用原始FC的固定投影/重复填充，虽有更多活跃权重，但其特征存在代数冗余；注册或非零梯度计数不等同于独立信息维数。E0没有新增分支参数。E6/E7/E8/E9同架构，C04/C05/C07医学概念块保持关闭；E8四个数值簇使用四个数值槽位，不将它们冒称医学概念。不能把本实验描述为所有组严格等有效容量。

E0原始GFR+PLN构造注册1,462,957参数，历史训练中有梯度的参数张量包含732,238参数（张量级统计，包含保留构造但不参与PEN前向的结构）；完整逐折记录在fold_metrics.csv，不能与上述元素级计数直接等同。E1–E9原主干参数均冻结，实际优化参数仅为本表中的分支。


## 机制诊断

`diagnostics.csv`保存每折参考拟合、OOF尾部、z截断、规范支持、质量代理、ROI度数、概念质量/拒绝质量、E8簇规模、实际梯度参数和残差比例。诊断不能替代四项正式指标。E9固定checkpoint干预在`counterfactual.csv`比较真实映射、错误映射、分区内患者机制错配、均匀语义权重、关闭语义输入及ALL_OFF；它只说明已训练checkpoint依赖，不能替代E0/E5/E7/E8从头训练比较。

- `E3_COND_NORM`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=116404。
- `E4_ROI_NORM`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=158657。
- `E5_MULTISCALE_NUM`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=161608。
- `E6_REAL_NO_REJECT`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=163381。
- `E7_WRONG_SEMANTICS`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=163405。
- `E8_DATA_PRIOR`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=163433。
- `E9_UCSMA_FULL`：outer-test平均参考支持=0.7320，|z|截断比例=0.0004，非零有效梯度参数=163389。

训练TD交叉拟合参考诊断（十折均值；只用于报告机制，未据此选择超参数）：

| 参考 | OOF MSE | OOF MAE | OOF |z| 95%分位 | 全体z截断比例 | 训练参考支持 |
|---|---:|---:|---:|---:|---:|
| global | 0.049654 | 0.175454 | 2.024221 | 0.000359 | 0.948335 |
| conditional | 0.045132 | 0.167321 | 2.031212 | 0.000390 | 0.680896 |

真实与错误语义的逐概念连接数均为[61, 96, 64, 0, 0, 55, 0]；真实先验覆盖并集197条，错误先验覆盖并集182条；同概念真实/错误集合交集数为[15, 41, 17, 0, 0, 9, 0]。完整交叠矩阵在manifest。E7只匹配逐概念边数，覆盖并集及重叠差异是实质性控制局限。

参考支持结合OOF残差尺度、年龄/站点/性别计数、ridge leverage、QC稳健距离和ROI连接度；这些是经验不确定性/可靠性代理，未证明临床概率校准，也未将Softmax称为校准不确定性。条件模型OOF误差下降只说明部分采集/表型条件有预测作用，不能把条件项当成ASD病理证据。


### 固定E9 checkpoint干预

以下干预不重训、不重新选epoch；均在同一折的E9最佳checkpoint上执行。REAL是训练时冻结映射，NO_REJECT关闭拒绝质量，PATIENT_SHUFFLE只在训练/测试分区内部错配FC证据，WRONG_MASK替换预注册错误HO掩码，UNIFORM_ALIGNMENT固定可用概念权重，SEMANTIC_OFF清除概念段，ALL_OFF清除整个残差。

| 干预 | ACC | Hard-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|
| REAL | 92.0742 | 91.8546 | 94.6300 | 89.0793 |
| NO_REJECT | 92.0742 | 91.8546 | 94.6300 | 89.0793 |
| PATIENT_SHUFFLE | 91.8443 | 91.6419 | 94.2044 | 89.0793 |
| WRONG_MASK | 92.0742 | 91.8546 | 94.6300 | 89.0793 |
| UNIFORM_ALIGNMENT | 92.0742 | 91.8546 | 94.6300 | 89.0793 |
| SEMANTIC_OFF | 92.0742 | 91.8546 | 94.6300 | 89.0793 |
| ALL_OFF | 92.0742 | 91.8546 | 94.6300 | 89.0793 |

| 固定checkpoint干预 | 相对REAL预测翻转总数（871次外层测试） | 平均绝对log-prob变化 | 平均残差L2变化 |
|---|---:|---:|---:|
| REAL | 0 | 0.00000000 | 0.00000000 |
| NO_REJECT | 0 | 0.00045528 | 0.00187281 |
| PATIENT_SHUFFLE | 2 | 0.00522425 | 0.03340356 |
| WRONG_MASK | 0 | 0.00000687 | 0.00004663 |
| UNIFORM_ALIGNMENT | 0 | 0.00045566 | 0.00182356 |
| SEMANTIC_OFF | 0 | 0.00010964 | 0.00104271 |
| ALL_OFF | 0 | 0.01853638 | 0.12761670 |

E9的未匹配质量均值为0.9545±0.0125，路由质量熵为0.1086±0.0296；这表示拒绝通道真实被使用，但也说明当前稀疏映射下大多数测试患者证据不足，不能包装成广泛疾病语义覆盖。E9与站点、年龄箱和质量代理的平均η²/秩相关分别为0.1391、0.0955和-0.0805；这些关联是诊断性统计，不代表医学亚型。

E9最佳epoch为0的fold数为9/10；其余fold也没有形成超过B0-A的分类改善。非零梯度和固定checkpoint敏感性只能说明分支可训练/可扰动，不能证明真实医学先验必要。

E5多尺度数值组、E7错误对应和E8训练折数值先验均与E9得到完全相同的四项十折均值；E9相对它们的ACC/Hard-AUC逐折全为平局。结合E9相对E0也完全平局，本轮没有证明医学语义超过数值先验、错误对应或单纯基座。

### 数学与优化原因

E9最佳checkpoint的测试患者中，平滑残差剂量达到上限99%以上的比例为0.1643；原始残差/5%预算之比的逐折中位数均值为18.67。tanh并未消除剂量饱和，说明固定历史学习率与输入尺度对极小残差预算偏大；没有在看到测试表现后调学习率或重缩放。

零初始化输出层使第一步只有输出层获得分类梯度；后续输入层获得有效梯度，精确恢复预检已通过。但5%受限分支、冻结的原分类器和基座本已较高的判别性能，使概率/残差变化不足以跨越多数硬标签边界。epoch=0表示第一步优化后的checkpoint，绝非偷偷恢复未训练的B0-A；严格ACC相同不更新，故9折停留在首步，保留了非零残差。

语义支持在稀疏RFE连接上不足：多数证据质量流入NULL。校准阈值采用训练TD的幅度+最大方向覆盖，而匹配分数使用平均方向覆盖加弱方向项；这两个固定统计并不相同，可能增加保守拒绝。ROI最多12条连接、19个ROI仅1–2条，也限制了区域级收敛证据。不能把高拒绝率自动解释为临床可靠，也不能把非零匹配解释为真实疾病机制。

共享Adapter同时读取CAM与数值证据，仍存在依赖非语义输入的路径。固定医学门控约束了语义特征，但没有保证分类器必须依赖它；这正需要E5/E7/E8和SEMANTIC_OFF对照。本轮这些对照没有可测的医学知识必要性增益。站点/年龄η²受类别数量与小测试折影响，QC相关部分由可靠性公式构造，均不是亚型或病理证据。


## 决定与限制

预注册内部筛选结果：**NO-GO**。判定还需结合真实医学先验相对错误映射和数值先验、拒绝质量及响应诊断；单seed GO不等于论文级成功。逐项条件：`[{"comparison": "E0_B0A", "acc_mean_gain_gt_0_2pp": false, "auc_mean_gain_gt_0_2pp": false, "acc_wins_at_least_7": false, "auc_wins_at_least_7": false}, {"comparison": "E5_MULTISCALE_NUM", "acc_mean_gain_gt_0_2pp": false, "auc_mean_gain_gt_0_2pp": false, "acc_wins_at_least_7": false, "auc_wins_at_least_7": false}, {"comparison": "E7_WRONG_SEMANTICS", "acc_mean_gain_gt_0_2pp": false, "auc_mean_gain_gt_0_2pp": false, "acc_wins_at_least_7": false, "auc_wins_at_least_7": false}, {"comparison": "E8_DATA_PRIOR", "acc_mean_gain_gt_0_2pp": false, "auc_mean_gain_gt_0_2pp": false, "acc_wins_at_least_7": false, "auc_wins_at_least_7": false}]`。E9对E0 SEN-ASD变化为+0.0000pp；灵敏度非劣阈值为−0.5pp。

既有方法中，FUNCOIN/规范模型的完整系统分析基于更密集连接和更大样本；PhenoBrain已研究表型条件化网络路由；KD-Brain已有语义条件子网络交互；BrainPrompt/BrainPrompt+已有多层提示；AAAI 2026对齐研究支持选择性/部分对齐。UCSMA-v2最多新增“稀疏ROI真实覆盖约束下的训练折条件TD偏离、解剖ROI聚合与可拒绝结构语义质量”，不能主张条件化、Prompt或医学注意力本身首创。

| 相关研究 | 已有内容 | 本轮具体区别与边界 |
|---|---|---|
| Ilioska等，Nature Mental Health 2026 | 高密度连接的个体异质性与区域/网络尺度群体收敛 | 本轮是252条稀疏RFE边的分类增量验证，不复现完整网络或个体病灶结论 |
| FUNCOIN，Progress in Neurobiology 2026 | 全脑系统层规范模型；大规模健康参考；神经/精神疾病偏离 | 本轮仅约421名outer-train TD的岭参考与解剖代理，不声称复现其临床分组 |
| PhenoBrain，ICML 2026 | 表型条件化长程通信与注意调节 | 本轮表型只进入TD参考和支持估计；患者图保持原GFR |
| KD-Brain，MICCAI 2026 | 语义条件Query及病理先验分布约束 | 本轮不用自由语义Query或LLM概率真值，采用冻结解剖候选及经验拒绝 |
| BrainPrompt / BrainPrompt+ | ROI/患者/疾病提示；后者还含频谱、空间提示 | 本轮无BOLD频谱或全连接组，不把多层Prompt本身作为首创 |
| AAAI 2026 To Align or Not to Align | 显式对齐强度与模态冗余关系的研究 | 仅提供避免强制共享空间的方法动机，不是本轮医学拒绝规则的临床验证 |

本轮完整创新链在最后两环失败：条件参考和拒绝确实可计算，真实医学结构也能改变内部响应，但未产生超过数值/错误先验的判别互补性与独立收益。因此当前实现不值得作为MICCAI 2027第一创新点继续主推；本轮不联合微调、不追加seed、不调参寻找最好结果，也不启动实验42。

继承协议仍有`val_idx=test_idx`的测试集checkpoint选择、RFE/标准化先于外层CV，以及871人transductive患者图；10折训练样本重叠，且仅一个seed。故结果是历史设置下的内部比较，不是无偏泛化、独立中心或跨seed稳定性证据。

数据/源码SHA256及E0逐折复现记录见`manifest.json`；100折正式结果见`fold_metrics.csv`，机制诊断见`diagnostics.csv`，固定checkpoint干预见`counterfactual.csv`。本实验产物目录为`/tmp/SPromptGL-E41-UCSMA-stage/experiment/experiment41_uncertainty_calibrated_semantic_multiscale_alignment`。
