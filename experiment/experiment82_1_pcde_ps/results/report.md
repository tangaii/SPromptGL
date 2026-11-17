# Experiment 82.1 PCDE-PS

状态：COMPLETE_AUDITED；30/30正式训练，30/30 Best独立审计。

正式目录：`/model/spromptgl/experiment/experiment82_1_pcde_ps`。仅seed=0、871人和原固定10折；ASD=0、TD=1。B0与E82均只读历史结果，所有新任务仅从对应E61 epoch10完整warmup初始化。

## 冻结方法与源码核查

真实历史PLNNoPEN仅执行InputLayer→Encoder/FeedForward→原OutputLayer；原八条Prompt的SPGS/PEN路径已被历史方法移除。E82.1不重新引入原PEN。OutputLayer真实签名为forward(x,attn_embedding)，x=CAM144；新增模块通过try/finally hook增强其输入，每个训练/评估仅一次DARG/CAM/原分类头。

核查[SPromptGL论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)、[原仓库](https://github.com/wanxixi11/SPromptGL)、[项目仓库](https://github.com/tangaii/SPromptGL)、[CoCoOp论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)及[官方Meta-Net源码](https://github.com/KaiyangZhou/CoOp/blob/main/trainers/cocoop.py)、[SSC论文](https://papers.miccai.org/miccai-2026/paper/4444_paper.pdf)与[官方代码](https://github.com/Hanyu1124/ssc)。精确版本、SHA及实际依赖见manifest，原文与源码读取记录合并在runtime/source_review.json。CoCoOp条件化文本context来自CLIP图像特征；本实验不假定CAM与CLIP图像空间对齐。SSC的SKP来自文本谱空间，PCDE来自训练患者CAM谱空间；SSC硬Top-r、预算融合与本文连续双侧覆盖不同。

U逐折与E82原版O逐元素相同，CPU float64；类/总体方差均为n−1。EMA固定0.95旧+0.05当前，只用合法训练h.detach()，训练前向更新一次，评估只读。类方差收缩为0.9类+0.1全训练总体方差；逐分量下限max(1e−8,1e−4总体EMA方差)。e=4tanh(LLR/4)进入路由时detach；辅助BCE仍通过Token/Query/attention产生实际梯度。折2新增统计/损失排除377、407，残差严格0；历史训练重叠仍保留。

0.25患者条件化、0.2有符号偏置、0.5温度、8%平滑限幅、0.01证据辅助损失、0.001残差正则和学习率全程冻结。G2门控同结构但不计算、不入优化器；G1/G3只有文本不同，初始参数与有效参数数一致。

文本秩8；匿名旋转Gram最大误差2.27e-13，运行float32误差3.05e-05；文本相对变化1.440141。冻结共享投影P_text列正交误差1.33e-15，医学/匿名投影Gram误差2.84e-13/1.42e-13。后续编码器可部分吸收几何变换，不能单凭训练对照证明医学语义因果贡献。

## 六项指标

均值±population SD，单位%；Hard-AUC=本历史二分类硬预测BA，Prob-AUC用连续概率。

### all10

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|G1|92.1891 ± 2.8587|92.0192 ± 2.8429|93.7214 ± 3.0621|89.8293 ± 3.4227|94.2091 ± 3.8925|91.4340 ± 3.0243|
|G2|91.7294 ± 2.7174|91.5643 ± 2.6331|93.7271 ± 2.8290|89.5671 ± 3.5241|93.5615 ± 5.0324|90.9709 ± 2.7907|
|G3|91.6144 ± 2.8249|91.4097 ± 2.7970|93.6456 ± 2.9629|88.8232 ± 3.9380|93.9963 ± 4.5098|90.7620 ± 3.0195|
|B0|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|E82-G2|92.1891 ± 2.5145|92.0238 ± 2.5195|93.3136 ± 2.8544|89.8293 ± 3.5923|94.2183 ± 3.3406|91.4166 ± 2.7286|

### without_fold2

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|G1|92.3430 ± 2.9738|92.1722 ± 2.9574|93.8368 ± 3.2071|90.0542 ± 3.5370|94.2903 ± 4.0951|91.5933 ± 3.1479|
|G2|91.8321 ± 2.8459|91.6521 ± 2.7616|93.8667 ± 2.9492|89.4919 ± 3.7071|93.8123 ± 5.2450|91.0517 ± 2.9305|
|G3|91.5767 ± 2.9753|91.3596 ± 2.9440|93.7349 ± 3.1104|88.6653 ± 4.1209|94.0539 ± 4.7502|90.6958 ± 3.1759|
|B0|92.2153 ± 2.9131|92.0481 ± 2.8723|93.6142 ± 3.0541|90.0474 ± 4.0959|94.0487 ± 4.9847|91.4703 ± 3.0580|
|E82-G2|92.3430 ± 2.6055|92.1774 ± 2.6110|93.3955 ± 2.9977|90.0542 ± 3.7192|94.3005 ± 3.5117|91.5740 ± 2.8328|

## 主要配对比较

单位百分点；全部逐折差、W/T/L及描述性95%配对t区间见paired_comparisons.csv，包含排除Fold2后的9折。共享测试折和test选模使其不构成无偏显著性检验。

|比较|ACC Δ|BA Δ|Prob-AUC Δ|SEN Δ|SPE Δ|F1 Δ|ACC W/T/L|
|---|---:|---:|---:|---:|---:|---:|---|
|G3−B0|-0.5747|-0.6196|+0.1510|-1.2439|+0.0046|-0.6970|0/6/4|
|G3−E82-G2|-0.5747|-0.6141|+0.3320|-1.0061|-0.2220|-0.6546|1/3/6|
|G3−G1|-0.5747|-0.6094|-0.0758|-1.0061|-0.2128|-0.6720|1/4/5|
|G3−G2|-0.1149|-0.1546|-0.0815|-0.7439|+0.4348|-0.2089|1/7/2|
|G2−E82-G2|-0.4598|-0.4595|+0.4135|-0.2622|-0.6568|-0.4457|0/6/4|
|G1−B0|+0.0000|-0.0102|+0.2268|-0.2378|+0.2174|-0.0250|1/7/2|

## 机制结果

G1：有效/名义参数10241/10241；warm谱方差覆盖86.106%，CAM相对漂移1.0621，Best投影能量2.9308，EMA均值/方差相对滞后0.2020/0.3155。
双侧cos=0.707682、L1=0.774339；ASD/TD有效方向9.909/10.623；同侧ASD/TD Prompt选择cos=0.995692/0.987737，跨侧=0.645067。患者间Query RMS=0.061213，Query偏离冻结锚点L2=0.479083，LLR符号偏置单独attention与完整attention cos=0.671083。
Null均值0.066811，LLR饱和比例0.007%；gate均值0.523340，近0/近1比例0.000%/0.000%；残差相对范数均值0.9234%，未乘gate的bounded均值1.7621%，近8%比例0.000%；Best epoch均值214.3。

G2：有效/名义参数10144/10241；warm谱方差覆盖86.106%，CAM相对漂移1.0634，Best投影能量2.9374，EMA均值/方差相对滞后0.1960/0.3336。
双侧cos=0.652624、L1=0.798691；ASD/TD有效方向12.377/13.035；同侧ASD/TD Prompt选择cos=0.991556/0.991040，跨侧=0.584498。患者间Query RMS=0.055068，Query偏离冻结锚点L2=0.430693，LLR符号偏置单独attention与完整attention cos=0.828168。
Null均值0.055155，LLR饱和比例0.000%；gate均值1.000000，近0/近1比例0.000%/100.000%；残差相对范数均值1.5477%，未乘gate的bounded均值1.5491%，近8%比例0.000%；Best epoch均值184.0。

G3：有效/名义参数10241/10241；warm谱方差覆盖86.106%，CAM相对漂移1.0671，Best投影能量2.9764，EMA均值/方差相对滞后0.1901/0.3707。
双侧cos=0.701258、L1=0.749945；ASD/TD有效方向11.201/11.732；同侧ASD/TD Prompt选择cos=0.990407/0.992172，跨侧=0.633090。患者间Query RMS=0.056921，Query偏离冻结锚点L2=0.467672，LLR符号偏置单独attention与完整attention cos=0.757414。
Null均值0.057149，LLR饱和比例0.000%；gate均值0.528150，近0/近1比例0.000%/0.000%；残差相对范数均值0.9472%，未乘gate的bounded均值1.7685%，近8%比例0.000%；Best epoch均值189.5。

每折EMA六个状态、各分量能量、完整Prompt相似度矩阵、Query幅度、Margin/门控/残差分位数、错误患者Margin、原三个优化路径及新增路径梯度均见mechanism.csv。初始gate前层因最后层零权重而零梯度；第二次真实更新前已验证连通。

## G3 Best只读干预

|干预|ASD概率平均绝对变化|最大绝对变化|硬翻转|纠错|误伤|净纠错|
|---|---:|---:|---:|---:|---:|---:|
|anonymous|0.000126253|0.002717137|0|0|0|0|
|swap_sides|0.000089072|0.001512229|0|0|0|0|
|conditioning_off|0.000196180|0.003215194|0|0|0|0|
|gate_off|0.000419358|0.004666328|1|0|1|-1|

G3相对B0：纠错2、误伤7、净纠错-5、硬翻转9，按折患者计数。

G3相对G1：纠错9、误伤14、净纠错-5、硬翻转23，按折患者计数。

G3相对G2：纠错3、误伤4、净纠错-1、硬翻转7，按折患者计数。

只读干预属于同一Best模型的即时响应，不等同于重新训练消融；类别交换仅交换锚点，仍保留按位置指定的LLR符号，不能由双侧差异本身推断文本医学含义。概率受到文本影响、文本带来独立收益、医学含义被验证是三个不同命题。

## 独立审计与工程记录

30/30 Best由分离审计代码重放，概率误差≤1e−8、硬预测一致，六项指标与历史工具及sklearn相同。严格ACC首次最大值与每轮日志核对。共有历史epoch RNG三元组共10921条完全一致；三组同折初始参数/共有RNG一致。30份Latest均完整恢复全部参数、4优化器、EMA及RNG；G3/Fold1双重下一训练+评估更新数值完全一致。源模型/数据/warmup/历史Best保护SHA保持不变。

正式调度耗时3131.3s，最大worker 10，失败事件0。峰值按任务PyTorch allocated记录；全部任务最高0.903GiB，不等于整设备驱动总占用。

保留的协议偏倚：val=test选Best、CV前RFE/标准化、transductive患者图、Fold2两人训练测试重叠；九折结果也不能消除前三项偏倚。潜在16方向不是脑区；Gaussian支持是训练数据统计证据，不是病理事实；正常/异常分组不是已验证的ASD/TD知识。

## 预注册科研判断

A 疾病竞争证据：G2/G3相对E82-G2 ACC分别-0.4598/-0.5747pp；完整证据与条件化同时改变，不能归因于单一机制。
B 医学文本：G3−G1 ACC -0.5747pp，ACC逐折W/T/L为1/4/5；Hard-AUC、SEN和ASD-F1也下降。医学文本替换和正常/异常侧交换的平均绝对概率变化分别仅0.000126和0.000089，均无硬预测翻转，未显示独立医学文本收益。
C 可靠性门控：G3−G2 ACC -0.1149pp，方向不支持门控组。G3内关闭门控平均概率变化0.000419，仅翻转1例且误伤1例；门控将平均有界残差由1.7685%缩至0.9472%，但没有带来总体分类收益或可靠性校准证据。
D 候选要求：G3−B0 ACC/BA/Prob-AUC分别-0.5747/-0.6196/+0.1510pp，主要硬分类指标没有一致增益，文本干预也未支持实际语义贡献。最终判断：**NO-GO**，不作为当前MICCAI 2027第二创新模块候选；不追加组别或调参。

疾病竞争路由确实形成了不同的ASD/TD侧权重（G3平均余弦0.7013、L1距离0.7499），患者条件化Query也发生数值变化（患者间RMS 0.0569，偏离冻结锚点L2 0.4677）。但方向选择仍高度相似，同侧Prompt选择余弦约0.990/0.992；查询关闭干预无硬翻转。机制存在不等同于有效分类贡献。G3相对B0按折患者计纠错2、误伤7、净纠错-5；相对匿名G1纠错9、误伤14；相对无门控G2纠错3、误伤4。统计关联不是病理事实，原始正常/异常Prompt分组也未经医学验证。

工程记录：Existing E82 run.py hardcoded old /model path; E82.1 uses read-only E81.3 path adapter at relocated /data location.；GitHub unauthenticated REST rate limit; sources retrieved at git ls-remote HEAD via raw URLs with retry after network TLS interruption.；During E82.1 training a separate concurrent session launched experiment81_6_prcl_res with 10 GPU workers. E82.1 launched no E81.6 process and did not modify/control that session. E82.1 worker limit stayed at 10. Wall time and whole-device sampled memory/utilization are confounded by shared GPU/storage contention; per-task torch.max_memory_allocated remains attributable to E82.1.；First independent audit passed Fold1 then stopped at Fold2 relation-pair SHA: audit held-out-label permutation restored state[labels_np] by rebinding rather than restoring the shared NumPy dataset array in place. This contaminated only that audit process memory. Fixed audit finally to restore the original array in place; original data files and all training/Best/Latest tensor states are unchanged. Independent audit restarted in a fresh process; no formal training rerun.；Independent audit found truncated/missing records in 7 original OSSFS JSONL epoch logs: 19 missing epochs. Raw logs and all formal checkpoints are preserved unchanged. Audit-only deterministic replays start from warmup or an existing Best, reconstruct missing rows, compare every retained row in each replay span exactly, and require exact selected Best/Latest bundle and RNG endpoints. Root cause of log corruption is not established; storage contention was observed. No new research group, hyperparameter change, selected epoch change or formal retraining is introduced.；OSSFS errno5 EIO during fsync of mathematically verified audit recovery proofs. Entire formal directory copied to local /model; all copied file SHA256 verified. No formal task rerun.

Checkpoint工程事件：Best格式检查后补齐显式续训计数wait=0、next_epoch=best_epoch、best_acc和bundle_sha；逐份核对参数/优化器/EMA bundle及RNG哈希未改变。原文件SHA与补齐后SHA保留在manifest和任务记录，训练源码及方法配置不变。

正式训练整设备采样峰值34494MiB，15秒采样可能遗漏瞬时峰值；存在另一会话并发E81.6负载，该数值不能单独归属于E82.1。
