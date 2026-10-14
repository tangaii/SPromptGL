# 实验十八：DS³A-v3 判别性保持与可靠度驱动的选择性对齐

固定10组×3seeds×连续10fold，主模型保留全部48/6/10/256维。无新增疾病Prompt；不做超参数搜索。

## 实际源码、论文与历史审计

GFR四图传播拼接320维；i=0先赋值，i=1覆盖前次combined_feat，i=2/3累加。保留这一实际顺序。输出已混合。PLN InputLayer→Encoder→FeedForward生成871×4×36潜在特征；注入在最后FeedForward后、原SPGS前。P0原8句CLIP RN50→2×4096→4096/128/9 ReLU MLP→cosine/softmax/max标量；随后原global平均池化、局部channel-softmax和RES加法、log-softmax分类器完整保留。

simple-2为加权NLL+四原输入图正则均值；ALoss计算但不反传。两个原Adam，PLN原my_weight_init执行后fork_rng创建辅助模块。450epoch，wait>200停止，严格ACC提升保存首次最佳；val=test，871人transductive，全队列既有标准化/RFE保留。不同组执行epoch不同后连续折随机流自然分叉，不能假称后续折初始化相同。

已实际阅读本地SPromptGL PDF Fig.1、Section2.1–2.2、Eq1–7和消融。Fig.2 ABIDE从CAM86.48到GFR91.73、RES92.08、SPGS92.30；主要增量来自GFR。论文Eq1的类别图项与源码未反传ALoss、论文图加权求和与源码覆盖行为、论文实现超参与本地shell有差异；本实验复现历史源码协议，不修补这些差异。

PHENO48=26年龄分箱+20站点+2性别；辅助分支仅28年龄/性别，主干仍含20站点。ANAT6/FUNC10是QC；func_gsr指Ghost-to-Signal Ratio。CORR256为既有FC/RFE；不重算。当前无ADOS/ADI-R/SRS真实症状；CAM位置和局部Token不能解释为ROI。

实验13患者FC文本未稳定超过自身状态/身份错配；14层级投影趋同且无稳定净收益；15文本内部响应未传递至最终预测、强衰减伴随消息接近零；16修复了文本敏感性但B5较P0 ACC下降0.841pp、残差约32.33%；17D5较P0 ACC/BA下降0.4210/0.3987pp，FC/PHENO类别原型cos0.9997/0.9986，Shared probes约60.14/49.36%BA，文本反事实零翻转。不能把距离降低、低Shared/Private cosine或非零梯度当作临床信息有效。E9真实30折六指标作为历史描述性对照。

Shared/Private、辅助类别监督、蒸馏、加权对齐和残差融合均为已有组件。候选研究点为outer-train跨模态联合正确性代理驱动的可拒绝对齐，且同权重预算、错配权重和同类错配患者分别验证选择与对应。r仅控制训练损失，不进入推理融合；不是推理期患者Gate，也不是真实医学可靠度或实际对齐收益标签。

## 文献核对与创新边界

- [MGDR](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)：模态关系图、SVD共性表示、私有表示对比约束及融合已有先例；本轮不使用SVD或私有对比损失。 阅读范围：官方可解析全文方法及消融；shell下载返回空文本二页包装，未把包装当全文。
- [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)：原GFR/CAM/SPGS/RES作为固定主路径，保留实际源码与论文差异。 阅读范围：根目录11页原论文全文及Fig.1图像。
- [Class-Aware Semantic Decoupling](https://papers.miccai.org/miccai-2026/1073-Paper1452.html)：CSSM/SDL/CSSF在类别语义层实现共享/特异、对齐和融合；本轮无类别文本查询或缺失模态任务。 阅读范围：官方摘要；PDF入口不可可靠解析，不声称已读全文。
- [To Align or Not to Align](https://ojs.aaai.org/index.php/AAAI/article/download/39248/43209)：显式对齐的价值随模态冗余和特异信息变化；本轮冻结系数，用同强度常量控制区分选择与整体强度。 阅读范围：官方AAAI26论文入口与14页arXiv全文。

MICCAI2026 PDF的shell响应为无法抽取正文的二页包装；这里只依据官方摘要比较其CSSM/SDL/CSSF，不借用未读全文的实验数据。

## OOF证据与标签边界

每个outer fold只构建一次：inner StratifiedKFold(5,shuffle=True,random_state=18000+fold)，StandardScaler及LogisticRegression(C=1,lbfgs,max_iter=1000)都只拟合inner train。FC/年龄性别/二者拼接/SITE/FC+SITE/FC+年龄性别+SITE六组仅记录outer-train OOF诊断。

ASD取class0概率。证据=[pFC,pPH,2|pFC−0.5|,2|pPH−0.5|,|pFC−pPH|,预测类别相同]；q为两教师OOF均正确的0/1，仅outer train存在；outer test q=NaN且mask。标准化+相同LR仅在outer train拟合r=P(q=1|证据)。q单类退化为训练先验，manifest记录。outer test教师只用outer train拟合再预测，不读test标签。

r为联合正确性代理，训练期冻结；未校准成医学可靠度，亦非真实对齐收益。可靠度拟合在训练OOF证据上完成，对r与q的相关/AUC是该拟合数据上的描述，不是独立可靠度验证。SITE仅作诊断，辅助模型和r估计器均不读取SITE列；原主干仍保留它。

evidence.npz仅含10份小型OOF概率、6维证据、q/r、mask、inner fold、固定置乱与可靠度参数，不复制原始输入。含训练q的研究产物不得作为测试监督使用。

## 固定数学

四个Linear→GELU→LayerNorm编码器：FC256、PHENO28各有Shared32/Private32。四表示拼接128→Linear64→GELU→Linear72，生成PHENO位置0/CORR位置3的36维方向。delta=0.1*tanh(alpha)*detach(||CAM token||)*direction/sqrt(sum(direction²)+1e−8)，alpha两个零初始化标量；ANAT/FUNC残差恒0，每人每位置相对L2≤10%。注入最后FeedForward之后、原SPGS之前，原GFR/CAM/SPGS/RES/OutputLayer不改。

Ltotal=Loriginal+0.1*enabled_disc*CE(FC Shared→Linear32/2,ytrain)+0.1*enabled_pair*mean_train(stopgrad(w)*||norm(sFC)−norm(sPH_pair)||²)+0.05*enabled_private*BCEWithLogits(FC Private→Linear32/1,FC OOF ASD软概率)。没有正交、对抗、MMD或其他辅助损失。mean分母为训练人数，绝不除权重和。

r不进入融合输入或残差尺度。本轮检验训练期选择性约束；推理期间患者输入仍决定四编码表示和残差方向，可靠度不作为推理Gate。

## 十组冻结矩阵

|组|判别监督|对齐权重|FC教师Private蒸馏|对齐PHENO对象|新增注册参数|
|---|---|---|---|---|---:|
|V0 原始P0|False|none|False|本人|0|
|V1 容量控制|False|none|False|本人|31597|
|V2 FC Shared判别性保护|True|none|False|本人|31597|
|V3 判别保护+统一患者对齐|True|ones|False|本人|31597|
|V4 同强度常量对齐|True|mean|False|本人|31597|
|V5 完整选择性对齐|True|real|False|本人|31597|
|V6 可靠度错配控制|True|permuted|False|本人|31597|
|V7 选择性对齐+FC教师Private蒸馏|True|real|True|本人|31597|
|V8 无判别保护的统一对齐|False|ones|False|本人|31597|
|V9 同类别跨患者配对错配|True|real|False|同类固定无自身置乱|31597|

V4与V5均值、总权重在数学上相同；64位浮点核对容差1e−12且记录实际误差。V6通过train索引无固定点置乱，保留r的数值集合/均值/方差；重复r值可能数值相同，不虚称所有不同患者权重数值必然不同。V9在train每类内无固定点置乱，权重及类别组成不变，仅训练对齐对象变化；分类及推理使用真实输入。

V1–V9注册31597新增参数。disc_head66/private_head33只在各自损失启用时获得梯度；有效结构参数V1=31498，V2–V6/V9=31564，V7=31597，V8=31498。注册相同不等于实际参与相同，diagnostics同时记录结构及实测梯度。

## 初始化、训练及恢复

原参数先完成构造及my_weight_init；fork_rng隔离seed=180000+task seed初始化新参数，恢复原CPU/CUDA随机流。每config×seed仅播种一次，10折原顺序连续，禁止逐折重新播种。P0原8句和RN50缓存保持原函数，正式每worker首次编码一次，其首次CLIP构造所消费的CPU随机流完整保留。

两个原Adam lr0.0038/WD0.11；新参数加入PLN Adam独立组lr0.0038/WD0。固定450epoch、wait>200停止，按val=test首次严格最高ACC checkpoint。Hard-AUC=BA，正式连续AUC用exp(log_softmax)第0类。原全局标准化和RFE、transductive图、test选模都有评价局限，不能称独立无偏泛化。

调度器与worker锁；每25epoch及fold边界原子保存校验和、模型、双Adam、fold/epoch、best、wait、CPU/CUDA/Python/NumPy RNG、已提交结果及源码/证据签名。epoch中断先重建模型与CLIP缓存，再恢复optimizer/RNG；fold边界先加载CLIP再恢复连续随机流。已完成任务校验跳过，唯一结果来自一致checkpoint，不混未完成尝试。

两卡各15任务/150fold，每卡最多4worker、每worker2CPU线程。资源压力仅调后续并发，不改模型、fold或epoch。临时来源副本仅六根文件与最小patch，CLIP只读链接；完成后删除临时源码/checkpoint。GPU总占用日志包含其他项目，不能当作本实验独占显存。

```bash
python experiment/experiment18_discriminability_preserving_selective_alignment/run.py prepare
python experiment/experiment18_discriminability_preserving_selective_alignment/run.py preflight
python experiment/experiment18_discriminability_preserving_selective_alignment/run.py all
python experiment/experiment18_discriminability_preserving_selective_alignment/run.py resume
python experiment/experiment18_discriminability_preserving_selective_alignment/run.py report
```

all自动完成缺失prepare/preflight；成功步骤验证跳过。COMPLETE后all/resume只验收，不自动重跑。完整另行重训需独立备份本目录结果并重置manifest/临时状态，不能把新结果混入这次冻结实验。

## 诊断与反事实

每fold一条诊断：最佳checkpoint eval模式损失及各已启用项梯度范数（不会更新optimizer）、训练期每25epoch和头两步采样component最大梯度、Shared/Private原型/方差/秩/距离/probe、r/q分布、alpha/参数范数及残差。线性probe固定StandardScaler+LR(C1)在outer train拟合、checkpoint确定后在test诊断；实验17用Ridge，跨轮probe数字不直接当同口径差值。原型使用未加r权重的类均值再L2归一化；17原型按Align加权，两者定义不同。V3/V8、V5/V4等本轮同口径比较更重要。

V5同checkpoint REAL/SHARED_OFF/PRIVATE_OFF/RESIDUAL_OFF/PHENO_SHUFFLED，共150行；置乱仅新分支年龄/性别，train/test各自固定循环置乱，不用test标签；主干输入、参数、注入前CAM Token不变，最后恢复REAL逐值验证。记录概率、真实logits、翻转、六指标及残差变化。反事实证明训练后依赖，不替代重训消融；r只在损失，故无推理交换r反事实。

成功判据保守描述：BA和Prob-AUC在三个seed均正向且ACC没有反向退化；逐seed ACC/SEN完整列出。三seed共享数据划分，30折不是30独立数据集，不做独立样本显著性推断。冻结核心比较V2−V1/V3−V2/V3−V8/V5−V4/V5−V6/V5−V9/V7−V5/V5或V7−V0。

实验1–17和根源码哈希保护，不自动Git commit/push/pull/reset/checkout。只有全部300fold、300诊断、150反事实、P0 parity及恢复验收通过，才追加handoff并标记COMPLETE。
