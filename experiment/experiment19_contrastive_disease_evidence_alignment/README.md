# 实验十九：AG-CDEA 解剖锚定疾病对比证据对齐

冻结14组×3seeds×连续10fold=420折；C4的11项同checkpoint反事实共330行。原P0完整重训30折，不进行超参数搜索。

## 源码、数据及历史审计

**source**：GFR在四模态各建受试者图；i=0赋值而i=1覆盖combined_feat，i=2/3累加。原标签先将ASD转-1、held-out置0来生成A，但A仅用于未反传的ALoss。simple-2反传原加权NLL+四图正则均值。保留这一实际行为，不修补论文/源码差异。

**pln**：原InputLayer/CAM→Encoder→FeedForward输出871×4×36；新增分支安装于原my_weight_init及两个Adam之后，注入最后FeedForward之后、SPGS之前。P0八句冻结RN50文本为8×1024，拼接2×4096→Linear4096/128/9 ReLU→cosine→softmax/max，产生871×4×4标量权重并乘潜在Token。global平均池化、local通道softmax、RES和log-softmax原分类器不改。4×4×9为潜在Token，不能解释为ROI。

**paper**：阅读本地SPromptGL十一页PDF的Fig.1文字、Section2.1/2.2、Eq1–7及ABIDE消融。CAM86.48→GFR91.73→RES92.08→SPGS92.30，主要增量是GFR。论文图加权和与实际覆盖、类别图项与源码ALoss不反传存在差异；本轮沿用实际历史脚本。

**history_9_14**：已阅读9/10/14补丁及报告、11映射表/审计、12/13真实模型。9 E9 ACC91.8826/BA91.6915/ProbAUC93.8227，疾病/P3概率高度相关。10真ASD相对ADHD/中性只微小且seed混合；不证明疾病必要性。12真实ROI文本在不同内层选模协议不如错配/身份，投影趋同。13患者FC状态文字未稳定超过数值/错配。14层级文本仍投影趋同。

**history_15_18**：已阅读15/16模型、Prompt和反事实；17 DS3A实现及18分支/独立runner。15 raw-FC decoder旁路与饱和门控使文本变化不传到预测，28/30折关键文本反事实逐值相同；16修复通路后B5残差32.33%、ACC/BA/ProbAUC均低于P0；17类原型cos约0.9997/0.9986，真实文本反事实零翻转；18保护FC Shared判别性但年龄性别OOF约49.78%BA、FC84.23%，选择代理没有稳定净收益。停止同一年龄性别对齐路线；不将梯度、敏感性或Probe当医学创新。

**data**：PHENO含26年龄分箱/20站点/2性别，ANAT6与FUNC10为QC，func_gsr是ghost-to-signal ratio。FC来自Nilearn0.7.1默认LedoitWolf相关矩阵再Fisher-z，非直接Pearson、非生理兴奋/抑制。全部原320维保留；可信252连接仅用于新辅助分支，4条未知ROI端点连接仍在原主干。

**identifiability**：预训练前数学风险：对满行秩固定Q(4×1024)，自由Wkey(1024×32)令QW可表达任意4×32有效查询；同几何正交旋转、整体符号及域行置换均可被Wkey重新参数化吸收。因此C4/C8/C9/C11容量相同，优化轨迹差异不能单独证明医学先验。C7重复查询为有意rank1对照，C1零查询为有意rank0；不可为满足区分性检查而改写它们。

**novelty**：ROI语义、图文对齐、类别查询、局部cross-attention及对比监督都有先例。候选差异只能是可信连接范围、疾病差异查询、NULL拒绝和破坏性对照联合验证；NULL不是校准的不确定性，静态Prompt没有患者级临床真值。

## 真实解剖范围

实验十一871/871数值唯一匹配，依据原111列索引联合ROI标签恢复名称。未知索引195/225/232/236不进入新分支；不删除原256输入。四域候选数为 [72, 56, 45, 32]，域间允许重叠；至少一个端点满足给定HO名称前缀。Cingulate前缀不包含Paracingulate。所有具体名称与标志见domain_mapping.csv。四域是粗先验，不是验证过的单一功能网络或ASD病灶。

- D1 [Gotts et al. 2012 Fractionation of social brain circuits](https://pubmed.ncbi.nlm.nih.gov/22791801/)：群体社会网络连通性关联；不推出全部颞顶连接同向变化或个体病灶。
- D2 [Just et al. 2007 frontal-parietal executive underconnectivity](https://pmc.ncbi.nlm.nih.gov/articles/PMC4500121/)：任务fMRI研究，仅支持宽泛候选先验；不可直接变成静息FC固定下降标签。
- D3 [Nair et al. 2013 thalamocortical connectivity](https://pmc.ncbi.nlm.nih.gov/articles/PMC3673456/)：丘脑皮层研究支持候选范围，异质方向不设hyper/hypo真值。
- D4 [Uddin et al. 2013 salience network connectivity](https://pubmed.ncbi.nlm.nih.gov/23803651/)：岛叶/前扣带线索不验证整个扣带/岛叶域为单一salience网络；当前HO锚点刻意粗粒度。

## 固定模型与控制

每域ASD/TD冻结RN50文本先各L2归一化，差再L2归一化为1024维q。REAL候选8句逐字保留，NEUTRAL/UNRELATED/CLASS_ONLY共24句一次冻结；P0原8句仍用原函数与原缓存。semantic_cache仅含小型文本向量、mask和置换，不含患者数据。原CLIP构造首次消费的CPU随机流照旧保留；新分支不每epoch编码。
可信z→[z,abs(z),tanh(z)]→Linear3/32→GELU→LayerNorm；加252×32边ID嵌入后再次LayerNorm。边ID仅数值身份，不代表医学知识。Key32/1024与q结合为q_eff=q@Wkey，score=h·q_eff/sqrt32。候选logit=score/0.2，NULL logit=log(候选数)，无效候选为−∞，NULL值为0。用完整softmax的边权重读取Value32/32及域分数u=sum(a*score)。u_global为四域均值。
concat四域Value128、u4、NULL4共136维→Linear64→GELU→Linear36。delta=0.1*tanh(alpha)*detach(norm(base_CORR))*raw/sqrt(sum(raw²)+1e−8)，alpha=0。只增强位置3/CORR；其他三位置严格0。原SPGS、RES、输出头不变。
Ltotal=原加权NLL+原四图正则均值+启用时0.1*BCEWithLogits(u_global[train],(y[train]==0))。只有outer-train标签参与新损失；Attention不读标签。C1/C2关闭新损失。该BCE是弱监督判别目标，不是患者文本—影像真值InfoNCE。

|组|名称|查询|解剖范围|证据监督|NULL|新增参数|
|---|---|---|---|---|---|---:|
|C0|原始P0|REAL|REAL|False|原模型|0|
|C1|数值容量|ZERO|REAL|False|True|53221|
|C2|真实语义无证据监督|REAL|REAL|False|True|53221|
|C3|无解剖锚点|REAL|ALL|True|True|53221|
|C4|完整AG-CDEA|REAL|REAL|True|True|53221|
|C5|中性文本|NEUTRAL|REAL|True|True|53221|
|C6|无关医学文本|UNRELATED|REAL|True|True|53221|
|C7|仅类别文字|CLASS_ONLY|REAL|True|True|53221|
|C8|ASD/TD方向交换|SWAPPED|REAL|True|True|53221|
|C9|功能域文本错配|DOMAIN_PERMUTED|REAL|True|True|53221|
|C10|解剖候选错配|REAL|PERMUTED|True|True|53221|
|C11|同几何随机查询|RANDOM_ROTATED|REAL|True|True|53221|
|C12|可学习查询|LEARNED|REAL|True|True|57317|
|C13|强制匹配|REAL|REAL|True|False|53221|

C1–C13除C12外新增53221参数，C12为57317；C1 Key32768注册但零查询导致结构失活，有效结构上界20453。其他组Key虽全部标量参与矩阵乘，固定查询rank r使其函数自由度仅r×32：真实/同几何随机rank4、类别文字rank1，不能将注册参数误作独立功能容量。C12的额外查询可学习，但自由Key本来已可表达4×32任意方向，其参数更多不等于增加同等函数自由度。
C8交换ASD/TD仅将q取负；C9固定[2,3,0,1]；C10对252边列统一确定性无固定点置换，保留每域候选数与重叠，真实数值不动；C11把真实rank4张成空间等距映射到正交随机子空间，Gram误差<1e−6、方向cos接近0。C7四个相同类别查询为有意消除功能域文字对照。

## 初始化、协议、恢复与运行

原构造、双Adam和my_weight_init之后，fork_rng(seed=190000+task seed)创建新模块，恢复原CPU/CUDA随机流；原参数lr0.0038/WD0.11、新参数独立组lr0.0038/WD0。preflight验证原前向/双Adam两步逐值一致、新组初始前向一致、两步梯度、第三步RNG恢复和只干预辅助通路。
固定871人、320维、ASD0/TD1，原450epoch、wait>200、val=test严格ACC提高选首次checkpoint。一个config×seed仅播种一次，连续原10折，绝不逐折重播种。各组早停不同使后续随机流分叉，逐折输入/初始化RNG哈希如实保存。原全局标准化/RFE、transductive及test选模不能作为独立无泄漏泛化证明。
两卡各21任务/210折，每卡最多4worker，每worker2CPU线程。按(config序号*3+seed)%2固定物理卡；每worker仅暴露一张逻辑cuda0。资源压力只调整后续并发。每25epoch/折边界原子保存带SHA校验的模型、双Adam、RNG、best/wait、唯一签名及已提交结果；恢复先重建原CLIP缓存再恢复随机流。最终删临时源码和全部权重。独立runner没有import旧实验runner。

```bash
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py prepare
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py preflight
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py all
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py resume
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py refresh
python experiment/experiment19_contrastive_disease_evidence_alignment/run.py report
```

COMPLETE后all/resume只核对已完成结果；不自动重训或混入新attempt。根六源码与integration_patch.diff可独立重建临时源码；cdea.py+run.py+冻结JSON/NPZ/mapping以及既有数据哈希足以复核，不需要永久临时源副本。

## 诊断、反事实及判断

每fold一行420诊断，包含损失/梯度/参数、查询几何、四域NULL分布和entropy、患者注意力差别、正负FC权重、train/test类别证据描述、CORR残差及站点聚合。没有患者级预测文件。梯度为训练头两步/每25步采样最大值及最佳checkpoint eval损失反传值，后者不更新权重。结构计数为参与计算的上界，另报告Key函数自由度。
C4同最佳模型测试REAL/NEUTRAL/UNRELATED/CLASS_ONLY/SWAPPED/DOMAIN_PERMUTED/ANATOMY_PERMUTED/RANDOM_ROTATED/NULL_OFF/RESIDUAL_OFF/FC_SHUFFLED，共330行。FC只在test内固定无自身置乱，不读标签；主干输入和注入前CAM逐值不变，参数不改、无训练，结束恢复REAL逐值相同。OFF是同已训练模型消除残差，不是另训练P0。
ACC/BA/HardAUC/ProbAUC/SEN-ASD/SPE-TD；ProbAUC用exp(log-softmax)第0类，HardAUC=BA。先每seed十折均值，再三seed均值±population SD。全部14组与预定14配对完整报告；稳定标准为三个seed在ACC/BA/ProbAUC各自都正向，不做30独立样本显著性推断。真正语义与解剖必要性须同时经重训破坏对照和同模型反事实；敏感不等于正确。工程PASS而语义/分类FAIL允许且完整执行，不追加调参版本。

## 相关工作与可核验阅读范围

- [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)：保留原GFR/CAM/SPGS/RES和实际历史代码。 阅读范围：本地全文上述章节/公式与Fig1图注。
- [MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)：多图、SVD共享/私有表示及私有对比约束已有；本轮不用这些模块。 阅读范围：官方可解析全文方法3.1–3.2及实验表。
- [BrainPrompt](https://arxiv.org/abs/2504.16096)：ROI文本加到消息传递、人口属性与疾病级Prompt已有；不能把ROI语义注入本身称新。 阅读范围：本地11页PDF方法与图1文字。
- [G-LaD](https://proceedings.mlr.press/v308/gupta26a.html)：GIN去噪、Sinkhorn图文分布对齐、LLM推理；NeuroAI Workshop PMLR308，不等同AAAI主会。 阅读范围：官方摘要与论文书目信息，未声称读全文。
- [SABER](https://arxiv.org/abs/2607.01901)：ROI/多尺度超图、Graph-CLS局部cross-attention、门控残差及疾病对比监督已有；作者出版列表https://hanxiangmin.github.io/publications/列为ICME2026。 阅读范围：用户本地6页PDF方法Eq8–13等；作者出版列表核验ICME2026。
- [Class-Aware Semantic Decoupling](https://papers.miccai.org/miccai-2026/1073-Paper1452.html)：类别感知共享/特异语义分解及融合已有，不能将类别查询概念作为首次。 阅读范围：官方摘要，未读可靠可解析全文。
- [LGDEA](https://arxiv.org/html/2602.07540v1)：从真实放射报告构建证据空间、可学习诊断原型与病灶查询；与本轮四域静态弱先验不同。作者出版列表https://zjuchenlong.github.io/Publications/列为ICML2026，阅读方法依据预印本版本。 阅读范围：原始作者预印本方法III-A/B及图注；作者出版列表核验ICML2026。

## 训练前补充：NULL与负向疾病分数的解析限制

令m=1−p_null，连接条件分布w_e=a_e/m，由entropy(w)≤log(n)，可得u/τ=Σa_e log(n a_e/p_null)≥m log(m/(1−m))≥−W(e⁻¹)。因此固定τ=0.2时，u及四域均值均≥-0.055692909，sigmoid证据概率均≥0.486080371。该限制源于指定Softmax和零NULL证据，不是数值bug；C13去除NULL后分数下界不再存在，故其对照同时涉及拒绝与证据监督值域变化。分析在重新正式420折前冻结（manifest有时间戳），不据此改公式或调参。

## 结构参与量的解析审计

注册量不等于结构参与量。真实mask及列置乱mask的候选并集均为165条，故87×32=2784个边ID参数恒不参与输出；C3全252条无此缺项。C1的4个u恒0，C13的4个NULL输入恒0，分别另有4×64=256个融合权重无效。C1还必须扣除零查询的32768个Key权重。由此结构参与量：C0=0；C1=17413；C2/C4–C11=50437；C3=53221；C12=54533；C13=50181。诊断中原component汇总给出上界，最终汇总须保留该上界并另写解析扣除后的实际结构计数，不能把上界当成有效容量。C1/C2已提交的采样非零梯度数与17413/50437吻合。固定q的Key函数自由度仍只有rank(q)×32，进一步不同于这些标量计数。
