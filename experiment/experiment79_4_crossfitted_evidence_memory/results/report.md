# Experiment 79.4：CF-CPDEM正式结果

- 状态：`COMPLETE_AUDITED`；科研判定：`NO_GO_CROSS_FITTED_EVIDENCE_MEMORY`。
- 基座：E74.4-G2 Positive-DCR；seed=0；10折；单折一条共享DARG轨迹。
- 新记忆只读取合法外层训练CAM144。训练Query从A读取B Bank、从B读取A Bank；每个Bank的Source/Support互斥。测试只读取M_full。
- 训练和预测保留历史val=test选模、CV前预处理与transductive图；这些限制不构成独立外部验证。Fold 2原历史重叠仍保留在旧基座，但被新增Memory排除。

## 六指标（%，mean ± population SD）

| 方法 | Ledger | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 | historical | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E79.4-G1 Base-Anchored | base_anchored | 92.0742±2.6000 | 91.9230±2.5676 | 93.4999±2.8973 | 90.0671±3.8862 | 93.7789±4.5987 | 91.3388±2.7215 |
| E79.4-G1 Method-Selected | method | 92.1891±2.7648 | 92.0480±2.7135 | 93.3828±2.8393 | 90.3171±3.4469 | 93.7789±4.5987 | 91.4875±2.8802 |

## 配对和纠错

Base-Anchored ACC差 `-0.1149pp`，排除Fold 2为 `-0.1277pp`；历史最佳Epoch的净纠错为 `-1`。

- base_anchored: 错误→正确 0，正确→错误 1，净纠错 -1，硬翻转 1。
- method: 错误→正确 1，正确→错误 0，净纠错 1，硬翻转 1。

## 机制

Base-Anchored Memory-Off外的内容打乱可观察效应：`False`。可学习匹配梯度、训练/测试Memory证据表现、独立Support可靠性、Gate与增量分布见`mechanism.csv`和`memory_analysis.csv`。

## 审计

best=20/20，latest=10/10，最新状态双重下一步重放=10/10。详细依据见`audit_status.json`。

## 解释边界

本实验中的CAM维度贡献及跨患者支持是模型表示空间内的计算性证据，不等同于医学病理标志物或因果生物标志物。E74.4历史测试集参与选模、CV前RFE/标准化、transductive图与Fold 2重叠限制泛化解释。

## 论文、源码与历史身份

已审查[SPromptGL原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[作者源码](https://github.com/wanxixi11/SPromptGL)、E74.4真实训练恢复接口、E79.3-G2源码与权威结果，以及[SAM 2论文](https://arxiv.org/abs/2408.00714)和官方Memory实现。原CAM进行跨模态交互；本实验新增跨患者Memory，二者不等同。SAM 2仅提供有限Bank、Cross-Attention与先读取/预测、后写入的借鉴；未移植视频掩码、帧位置或跟踪假设。
实际PLNNoPEN的OutputLayer输入是CAM144，内部为LayerNorm(144)→Linear(144,36)→GELU→Dropout→Linear(36,2)。新增分支使用其冻结参数的eval函数调用，梯度仅回到新增模块。原分类头仍由原DARG损失更新。本实验不是完整主干端到端联合优化。
每折从E61 epoch-10恢复模型、三个优化器、4096关系对和全部RNG；opt_base→opt_cam→opt_added次序不变。历史最佳Epoch以1-based锁定为123/171/183/216/295/339/148/263/186/233。10/10历史最佳主干（含优化器）逐张量一致；当前871人整批只读分类头输出与原BaseEval逐元素一致。历史无全部逐轮参数快照，故逐轮参数摘要仅记录本实验，未伪称与不存在的历史摘要比较。

## CF-CPDEM实际方法

A/B按合法训练患者的疾病与站点确定性分层，互不重叠；各Bank以最大余数分配约75% Source、25% Support。A查询只读取B、B只读取A；Source特征干预和Support可靠性/疾病方向完全在该Bank内部。测试只读取Full。此隔离针对新增Memory的显式标签作用域，原DARG仍按历史全训练监督和transductive图运行，不构成整个表示流程的无标签OOF。
正确分类Source患者执行α=0.5和1.0的144维替换，选两种强度下都正贡献的最多12维，并验证联合遮蔽margin下降。背景维度匹配Support，每类最多16人，至少4人，要求同类证据匹配均值高于异类且AUC≥0.60。r_self=1−exp(−joint_drop)，r_within=(same_match_mean+1)/2，r_between=2(AUC−0.5)，r_coverage=min(1,n_support/32)×min(1,n_sites/3)，各分量截断在[0,1]，最终取乘积。
固定Masked Cosine仅预筛每类16项；当前可训练Wq/Wk在同一Mask作用后的Query/Key上计算64维归一化匹配，以cosine/0.2+log(reliability)重排Top-8。真实有效数量归一化logmeanexp形成ASD/TD竞争分数；缺任一类即严格零增强。Wv无bias，将中心化证据与按CAM尺度缩放的Support疾病方向编码为Value；两类读出及其差产生144维修正方向。
Gate使用原logp、Memory类别支持、Attention熵、两类实际可靠性等信息，初始化bias−2.0。有效Gate额外乘sqrt(r_ASD r_TD)/(sqrt(r_ASD r_TD)+0.1)，记录中区分原始Attention可靠性和这一变换后可靠性。平滑tanh范数限制保持7.5%上限。最终投影std=0.01的小非零初始化；无随机Dropout。
新增损失固定为加权NLL+0.20错误margin+0.10正确预测保护+0.10证据CE+0.01相对增量平方；错误样本权重3、正确1，保留历史类权重，以sum(sample_weight×class_weight[label])归一化NLL。AdamW lr=0.0005、WD=0.01。未依据测试结果调整任何阈值或超参数。
每5轮刷新三套Bank，第一次在第11轮预测结束后构建；当轮仅读取旧版本。Key随当前Wk重算，不使用陈旧投影缓存。只保留当前Memory及独立重建所需Bank局部CAM/分类头快照，未保存全Epoch患者特征。

## 资源与完成情况

实际目录：`/model/spromptgl/experiment/experiment79_4_crossfitted_evidence_memory`。首选OSSFS目录在mkdir/写入/fsync/原子rename/读回探测中出现EIO(errno=5)，立即采用/model回退。单张PPU-ZW810E，10个fold worker，1个正式方法。最长worker实测4554.1秒（约75.9分钟；并行训练阶段计时），独立最终审计113.1秒。
设备1秒采样5144条，实测整卡used峰值20439MiB（19.96GiB）；总显存98304MiB。单worker Torch peak allocated最高1055.2MiB；两种内存口径不同。新增可训练参数49,741个。10/10完成，共3917个实际训练Epoch、786次fold级Memory刷新。

## 双Ledger配对结果

以下差值均为方法−E74.4-G2（pp）。95%区间为配对折差mean±1.96×sample-SD/sqrt(10)的描述性区间；十折训练集重叠，不能视为独立外部队列或临床显著性证明。

| Ledger | ACC差与描述性95%区间 | W/T/L | 去Fold 2 ACC差 | Hard-AUC差 | Prob-AUC差 | SEN差 | SPE差 | ASD-F1差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_anchored | -0.1149 [-0.3402,+0.1103] | 0/9/1 | -0.1277 | -0.1064 | +0.0053 | +0.0000 | -0.2128 | -0.1203 |
| method | +0.0000 [+0.0000,+0.0000] | 0/10/0 | +0.0000 | +0.0186 | -0.1118 | +0.2500 | -0.2128 | +0.0284 |

逐折六指标差、六指标95%区间、去Fold 2结果均保存在paired_comparisons.csv。Base-Anchored仅Fold 6降低1/87；Method-Selected各折ACC均与历史持平，但患者身份可变化。Method-Selected最佳Epoch为123/171/183/216/295/377/135/263/186/233，Fold 6与7偏离历史最佳。

## 同Checkpoint与历史基座纠错分开统计

| Ledger/参照 | 错误→正确 | 正确→错误 | 净纠错 | ASD纠正 | TD纠正 |
|---|---:|---:|---:|---:|---:|
| base_anchored/同Epoch Memory-Off | 0 | 1 | -1 | 0 | 0 |
| base_anchored/历史最佳 | 0 | 1 | -1 | 0 | 0 |
| method/同Epoch Memory-Off | 1 | 0 | 1 | 0 | 1 |
| method/历史最佳 | 1 | 1 | 0 | 1 | 0 |

Base-Anchored原正确803人保留802人（99.8755%），损害是1名TD被改判ASD。Method-Selected同Epoch原主干正确802人全部保留，同时纠正1名TD；相对历史最佳则纠正1名ASD、误伤1名TD。两种参照不能相互替代。去Fold 2后上述纠错计数不变。

## 证据能力与融合失效分解

| Ledger | A/B交叉训练证据ACC/Hard-AUC/Prob-AUC | 测试证据ACC/Hard-AUC/Prob-AUC | 原错误中Memory正确/错误 | Memory正确但融合仍错 | Memory错误并造成误伤 |
|---|---|---|---:|---:|---:|
| base_anchored | 95.8666/95.8448/98.8750 | 88.7435/88.8124/92.7362 | 3/65 | 3 | 1 |
| method | 95.5860/95.5683/98.8424 | 88.8584/88.8816/92.6884 | 2/67 | 1 | 0 |

以上证据指标为逐折均值，均有双类别可用，测试总计871个折测试记录。训练指标是A/B交叉Memory查询，不是外部测试或无标签表征OOF。历史最佳的68个原错误中，65个Memory也错，仅3个给出正确类别支持（4.41%）；这3个最终仍全错，另有1个原正确TD被错误Memory支持误伤。故同时存在错误互补性不足和正确Memory支持未转化为硬纠错。现有证据不能唯一归因于记忆内容缺乏独立信息或融合没有有效利用内容。

## 梯度、可靠性和修正幅度

L_evidence单独反传的正式预检确认Wq/Wk梯度均非零有限，并检查参数扰动会改变分数、CAM/主干/原OutputLayer不接收新增梯度。正式各有效更新同样检查Wq/Wk梯度和主干原梯度不变。最佳点的梯度范数是训练总Memory损失的诊断，不能混称为单独证据损失梯度。

| Ledger | Wq/Wk/Wv/Gate平均梯度范数 | 选中支持越出固定Top-8比例 | Mean Gate | 平均相对增量 | 接近7.5%上限比例 |
|---|---|---:|---:|---:|---:|
| base_anchored | 0.056055/0.052938/0.002387/0.001542 | 46.39% | 0.2305 | 6.7634% | 51.10% |
| method | 0.048253/0.046546/0.002419/0.002069 | 46.59% | 0.2297 | 6.6839% | 46.39% |

接近上限预定义为delta_ratio>0.99×0.075（7.425%），不是精确硬裁剪命中率。平滑限制保留修正方向，但Base-Anchored仍有约一半患者接近上限，未充分解决饱和。Gate并未塌缩到0；问题不是E79.2式几乎没有增强。P50/P90/P95是各折分位数的均值，见mechanism.csv；分组结果如下（按患者记录数加权）。

| Ledger | 阶段/人群 | 记录数 | Gate均值 | 平均相对增量 | 接近上限人数 | 全零人数 |
|---|---|---:|---:|---:|---:|---:|
| base_anchored | training/all | 7839 | 0.2264 | 6.7032% | 3888 | 0 |
| base_anchored | training/base_correct | 7596 | 0.2262 | 6.6967% | 3752 | 0 |
| base_anchored | training/base_wrong | 243 | 0.2314 | 6.9070% | 136 | 0 |
| base_anchored | training/memory_conflict | 151 | 0.2568 | 7.1865% | 99 | 0 |
| base_anchored | training/memory_agreement | 7688 | 0.2258 | 6.6937% | 3789 | 0 |
| base_anchored | test/all | 871 | 0.2305 | 6.7629% | 445 | 0 |
| base_anchored | test/base_correct | 803 | 0.2309 | 6.7573% | 409 | 0 |
| base_anchored | test/base_wrong | 68 | 0.2251 | 6.8289% | 36 | 0 |
| base_anchored | test/memory_conflict | 36 | 0.2696 | 7.1991% | 26 | 0 |
| base_anchored | test/memory_agreement | 835 | 0.2288 | 6.7441% | 419 | 0 |
| method | training/all | 7839 | 0.2269 | 6.6543% | 3665 | 0 |
| method | training/base_correct | 7596 | 0.2267 | 6.6478% | 3545 | 0 |
| method | training/base_wrong | 243 | 0.2310 | 6.8585% | 120 | 0 |
| method | training/memory_conflict | 159 | 0.2624 | 7.1688% | 89 | 0 |
| method | training/memory_agreement | 7680 | 0.2261 | 6.6436% | 3576 | 0 |
| method | test/all | 871 | 0.2297 | 6.6835% | 404 | 0 |
| method | test/base_correct | 802 | 0.2304 | 6.6765% | 373 | 0 |
| method | test/base_wrong | 69 | 0.2216 | 6.7658% | 31 | 0 |
| method | test/memory_conflict | 32 | 0.2690 | 7.1184% | 19 | 0 |
| method | test/memory_agreement | 839 | 0.2282 | 6.6669% | 385 | 0 |

训练患者跨折重复计入，不能把这些记录数称为不同的独立患者。

## 实际Memory容量和独立Support

| Ledger/Bank | 容量范围 | 候选合计/接受/拒绝 | 原始可靠性均值 | Support AUC均值 |
|---|---:|---:|---:|---:|
| base_anchored/A | 96–96 | 960/960/0 | 0.7846 | 0.9806 |
| base_anchored/B | 96–96 | 960/960/0 | 0.7640 | 0.9768 |
| base_anchored/full | 95–96 | 960/959/1 | 0.7705 | 0.9734 |
| method/A | 96–96 | 960/960/0 | 0.7885 | 0.9799 |
| method/B | 96–96 | 960/960/0 | 0.7758 | 0.9799 |
| method/full | 96–96 | 960/960/0 | 0.7757 | 0.9732 |

这些是十折最佳点各Bank的候选记录，不是不同患者数或全训练累计数。每个有效候选最多12维。A/B最佳点均48 ASD+48 TD；Full的Base-Anchored仅一折47 TD，其余满容量。独立Support不保证低接受率：强训练表示下，最佳点大部分候选仍全部通过，Full的平均Support AUC为0.9734，而测试Memory证据ACC仅0.8874。不能把高训练Support可靠性等同于外部诊断可靠性。

## 同Checkpoint只读干预

差值为Full−干预。概率MAE按患者ASD概率变化后逐折均值，Prob-AUC差为pp；Memory-Off的delta_ratio严格0。其他干预保持权重，不重训，因而不是训练消融。

| Ledger | 干预 | ACC差pp | Prob-AUC差pp | 相对Full硬翻转 | ASD概率MAE |
|---|---|---:|---:|---:|---:|
| base_anchored | Memory-Off | -0.1149 | +0.0053 | 1 | 0.00224139 |
| base_anchored | Memory-Content-Permuted | +0.0000 | +0.0000 | 0 | 0.00000773 |
| base_anchored | Learnable-Matching-Off | +0.0000 | -0.0105 | 0 | 0.00020086 |
| base_anchored | Cross-Patient-Reliability-Off | +0.0000 | -0.0105 | 0 | 0.00007835 |
| method | Memory-Off | +0.1149 | +0.0106 | 1 | 0.00241364 |
| method | Memory-Content-Permuted | +0.0000 | +0.0000 | 0 | 0.00000832 |
| method | Learnable-Matching-Off | +0.0000 | -0.0105 | 0 | 0.00021664 |
| method | Cross-Patient-Reliability-Off | +0.0000 | -0.0105 | 0 | 0.00008134 |

Content-Permuted固定Query、Key、Top-K和Attention，仅循环置换每个已选类内Value，独立核验Value患者身份非平凡改变。结果是极小概率变化、零硬翻转、零Prob-AUC变化；没有显示具体Key–Value患者对应具有独立分类收益。这一干预不等价于删除类别分区或任意外部噪声。可学习匹配和可靠性Off确实改变分数/幅度及部分概率，但均无硬分类变化，且Full的Prob-AUC反低0.0105pp。不能把可学习检索有梯度和可靠性在公式中生效解释为已证明诊断增益。

## 独立审计、源码版本与复现

10/10正式折、20/20双Ledger最佳Checkpoint独立重放，10/10 Latest恢复并实际执行下一步，比较从未重新加载的live轨迹：主干与新增参数、四个优化器、Memory、预测、损失及Python/NumPy/Torch CPU/CUDA RNG摘要全部一致。3917/3917个共同Epoch的三处RNG记录完全匹配，0不匹配。
Memory的Source/Support/分区、刷新时CAM与冻结head保存于Checkpoint；审计从刷新时材料重建Mask、遮蔽效应、支持ID、对照方向和可靠性逐项一致。不是使用最佳Epoch的新CAM重建旧Memory。Fold 2的376/406和所有外层测试ID在新增Memory及支持/中性统计中均排除。
Memory-Off令增强CAM与原CAM逐元素一致。只读head在原871人批大小下logp逐元素一致；小测试批head与全批原输出存在至多约1e-5的算子归约差异，审计单独记录，不将其当作记忆效应。Memory-Off概率采用原BaseEval概率严格回退。
正式训练后仅audit.py/analyze.py作审计/汇总修正：固定全批分类头等价检查、补充标签分层诊断、JSON布尔序列化及依据原任务规则的判定逻辑。没有改动训练入口、模型、损失、配置、权重或历史文件。训练与最终后处理SHA均列于audit_status.json/source_changes及manifest.json；训练源集合SHA保留不覆盖。复现入口为python run.py，预检/训练参数见其--help；独立重放python audit.py，汇总python analyze.py。

## 与E79.3-G2比较及最终判断

E79.3-G2的Base-Anchored ACC91.9592%，净纠错−2；E79.4为92.0742%，净纠错−1，改善0.1149pp但仍低于DARG。Method-Selected两者ACC均92.1891%；E79.4 Prob-AUC93.3828%低于E79.3-G2的93.5690%。本次实质修复可学习Q/K证据梯度、显式交叉记忆作用域、Source/Support独立性、真实可靠性Gate、无效Top-K和真实Latest续训审计，工程修复不等于科研成功。
最终判定：NO_GO_CROSS_FITTED_EVIDENCE_MEMORY。未超过历史DARG，历史最佳Epoch无错误被纠正且误伤1人；方法选模的单折+1不能抵消历史最佳对比净变化0；具体Value内容干预没有破坏分类表现。Memory证据有类别区分能力，但在DARG原错误上几乎没有互补性，少量正确证据也未被融合成纠错。当前证据同时暴露内容冗余与融合利用不足，无法唯一确定根本原因。没有充分理由将此实现继续作为MICCAI 2027第二创新模块；不自动追加E79.5或测试集调参。
