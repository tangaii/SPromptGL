# Experiment 78.1: SH-DCPSS 结果报告

状态：`COMPLETE_AUDITED`。正式目标为3组×10折；已完成任务 30/30，独立审计 30/30。

## 审计与方法

唯一基座是 `E74.4-G2-Positive-DCR`；每折从对应 E61 epoch-10 warmup 的完整参数、三个原优化器和 RNG 恢复。训练源 SHA-256：`b19b074b2f805d1910fa4d33572ec4100a0063d86e118770f2b93c83ab6c51ef`。
CAM144 通过只读 pre-hook 注入原始 OutputLayer。E74.4 分类损失、图/配对/对抗/保持损失和原优化器更新次序保留；新特征残差由第三个类内动态原型与固定文本语义共同生成。
方法审计、论文证据范围和数据身份限制见 `literature/source_review.md`。E74.4历史协议含 val=test、测试ACC选模、外层CV前预处理/RFE、全患者传导图和Fold 2历史重叠；本文结果不代表无偏临床泛化估计。

## 分组结果

|组别|ACC|Hard-AUC/BA|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E78.1-G1-SPromptGL-K1|92.1891 ± 2.8121|92.0480 ± 2.7367|93.8609 ± 2.6635|90.3171 ± 2.8515|93.7789 ± 4.5987|91.4991 ± 2.8997|
|E78.1-G2-SPromptGL-K3|91.6144 ± 2.9619|91.4531 ± 2.9713|93.5111 ± 3.3327|89.3354 ± 4.4613|93.5708 ± 4.3208|90.7977 ± 3.2084|
|E78.1-G3-Anonymous-K3|91.8443 ± 2.8431|91.6526 ± 2.8439|93.4836 ± 2.7962|89.0915 ± 3.8646|94.2137 ± 3.7472|91.0104 ± 3.0602|

平均每折训练耗时（秒）：E78.1-G1-SPromptGL-K1 79.64; E78.1-G2-SPromptGL-K3 84.65; E78.1-G3-Anonymous-K3 82.48

## 配对比较

下表按10折逐折配对；折2剔除敏感性结果同时写入 `paired_comparisons.csv`。W/T/L是每折差值的胜/平/负，95%区间为描述性 t 区间。

|比较|指标|差值(pp)|W/T/L|折2剔除差值(pp)|
|---|---|---:|---:|---:|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|acc|-0.5747|1/3/6|-0.5109|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|hard_auc|-0.5949|2/2/6|-0.5255|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|prob_auc|-0.3498|3/0/7|-0.2414|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|sen|-0.9817|2/4/4|-0.8198|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|spe|-0.2081|3/4/3|-0.2313|
|E78.1-G2-SPromptGL-K3 - E78.1-G1-SPromptGL-K1|asd_f1|-0.7014|2/2/6|-0.6285|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|acc|-0.2299|1/6/3|-0.2554|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|hard_auc|-0.1995|2/4/4|-0.2217|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|prob_auc|0.0275|5/1/4|0.0246|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|sen|0.2439|2/6/2|0.2710|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|spe|-0.6429|3/4/3|-0.7144|
|E78.1-G2-SPromptGL-K3 - E78.1-G3-Anonymous-K3|asd_f1|-0.2127|2/4/4|-0.2364|
|E78.1-G2-SPromptGL-K3 - E78-G2|acc|-0.4598|2/2/6|-0.3831|
|E78.1-G2-SPromptGL-K3 - E78-G2|hard_auc|-0.3929|2/1/7|-0.3011|
|E78.1-G2-SPromptGL-K3 - E78-G2|prob_auc|0.0807|6/0/4|0.1426|
|E78.1-G2-SPromptGL-K3 - E78-G2|sen|0.5000|4/2/4|0.8266|
|E78.1-G2-SPromptGL-K3 - E78-G2|spe|-1.2858|3/3/4|-1.4287|
|E78.1-G2-SPromptGL-K3 - E78-G2|asd_f1|-0.4080|2/1/7|-0.3025|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|acc|-0.5747|1/4/5|-0.5109|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|hard_auc|-0.5763|2/3/5|-0.5048|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|prob_auc|0.0165|5/0/5|0.0537|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|sen|-0.7317|1/5/4|-0.5420|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|spe|-0.4209|3/5/2|-0.4677|
|E78.1-G2-SPromptGL-K3 - E74.4-G2|asd_f1|-0.6613|2/3/5|-0.5839|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|acc|0.0000|1/8/1|0.0000|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|hard_auc|0.0186|1/7/2|0.0207|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|prob_auc|0.3663|6/1/3|0.2951|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|sen|0.2500|1/8/1|0.2778|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|spe|-0.2128|1/7/2|-0.2364|
|E78.1-G1-SPromptGL-K1 - E74.4-G2|asd_f1|0.0401|1/7/2|0.0446|
|E78.1-G3-Anonymous-K3 - E74.4-G2|acc|-0.3448|2/3/5|-0.2554|
|E78.1-G3-Anonymous-K3 - E74.4-G2|hard_auc|-0.3768|2/3/5|-0.2832|
|E78.1-G3-Anonymous-K3 - E74.4-G2|prob_auc|-0.0110|5/0/5|0.0290|
|E78.1-G3-Anonymous-K3 - E74.4-G2|sen|-0.9756|1/6/3|-0.8130|
|E78.1-G3-Anonymous-K3 - E74.4-G2|spe|0.2220|2/6/2|0.2467|
|E78.1-G3-Anonymous-K3 - E74.4-G2|asd_f1|-0.4486|2/3/5|-0.3476|

## 机制证据

G2原型对余弦相似度>0.98的平均比例为 0.0000（E78-G2为0.767）；平均站点 Cramér's V 为 0.4440（E78-G2约0.6703），年龄分组关联为 0.2457。
子原型条件 Prompt 权重TV为 0.06667（E78-G2约0.00749）；患者权重标准差 0.04029，归一化路由熵 0.6721。
CAM特征修正与原特征范数比为 0.03265。每折各原型有效支持数、站点组成、Prompt权重及残差数据见 `mechanism.csv`。

G2同一Checkpoint干预（10折平均，固定训练权重）：

- `module_off`：ACC 91.2696%，平均概率绝对变化 0.008727，硬标签翻转 5。
- `subprototype_off`：ACC 91.2696%，平均概率绝对变化 0.007564，硬标签翻转 5。
- `anonymous`：ACC 91.4995%，平均概率绝对变化 0.002238，硬标签翻转 1。
- `prompt_class_swap`：ACC 91.2696%，平均概率绝对变化 0.004549，硬标签翻转 3。
- `feature_residual_off`：ACC 91.2696%，平均概率绝对变化 0.008727，硬标签翻转 5。

## 假设判断

- H1子原型证据：`SUPPORTED`。
- H2站点关联缓解：`SUPPORTED`。
- H3患者条件Prompt选择：`SUPPORTED`。
- H4真实医学文本独立收益：`NOT_SUPPORTED`（G2要求ACC与Prob-AUC均高于匿名G3）。
- H5相对E74.4基座分类收益：`NOT_SUPPORTED`（ACC、Hard-AUC、Prob-AUC不降且至少一项提升）。

结论：`NO-GO`。

单seed及历史测试集选模下，GO仅用于决定是否值得设计下一阶段研究，不构成无偏临床泛化证据。未自动启动后续实验。

## Checkpoint 与 RNG

本报告由独立读回审计生成；完整逐折状态见 `checkpoint_audit.csv` 与 `audit_status.json`。训练源 SHA-256：`b19b074b2f805d1910fa4d33572ec4100a0063d86e118770f2b93c83ab6c51ef`。

## 文件

- `fold_metrics.csv`：30折指标、混淆计数、最佳epoch和耗时。
- `paired_comparisons.csv`：全折配对差与Fold 2敏感性分析。
- `mechanism.csv`：支持数、坍塌、站点/年龄关联、路由和Prompt差异。
- `interventions.csv`：固定G2 Checkpoint语义/残差干预。
- `checkpoint_audit.csv`：逐折Checkpoint读回、基座、RNG、梯度和标签不变性。
- `preflight.json`：10折base-off parity、全链梯度、文本几何和恢复重放。

审计恢复说明：冻结训练源码与原始audit.py均未修改；独立恢复器仅修正机制汇总中route_prop容器类型，并在已有失败handoff记录后追加本次完成状态。恢复器SHA-256：`c4dd0d2c7cf987e2518b6605b0164751785225f3667c3d1e095a8be64fd2ab8c`。

## 固定设计与基座继承审计

三组新增模块均为 9,425 个可训练参数；G1 的 K=1 与 G2/G3 的 K=3 只改变支持锚点和聚合张量，不增加分类层。`h ∈ R^144` 经独立投影得到 `q=normalize(W_sem h) ∈ R^16` 和 `z=normalize(LN(W_het h)) ∈ R^32`。冻结 RN50 的 16 条文本向量（SPromptGL 原始 8 条与 E78-G5 匿名 8 条）构成同一 rank-16 SVD 子空间；坐标冻结，投影前后两两余弦最大误差为 `7.77e-16`。SPromptGL 原文顺序保持 TD/NC 在前、ASD 在后，映射到模型类别 TD=1、ASD=0。

初始化仅使用各外层训练折 E61 epoch-10 CAM144：先按合法训练标签减类均值，再减去按 `n/(n+10)` 收缩的类内站点偏移；每类用确定性最远点初始化和 20 次 Lloyd 更新产生 K 个软分配锚点。锚点使用完整训练支持集统计，因此初始化本身不是严格嵌套交叉拟合；它们只用于分配参考。正式训练每轮只执行一次 DARG/CAM 主干前向，利用当前 `z` 更新训练支持责任度，再以站点样本量的温和逆平方根权重聚合可微的 `q/z` 子原型。训练 Query 对每个支持子簇的自身加权贡献均被精确扣除后再归一化；空簇/非有限值预检通过。所有推理 Query 同时路由 ASD 与 TD，不读取测试标签。G2 最佳状态的有效软支持数范围为 7.57–352.22，均值 130.65；低端簇说明局部亚型支持并不均匀，原型差异不等于临床亚型已被证实。

Prompt 在每个类别、每个子原型上独立计算 `s=cos(q,t_j)+cos(v_c,k,t_j)-cos(v_opp,t_j)`，以温度 `0.5` 对该类四条 Prompt 做 softmax；患者路由权重 `beta` 与 Prompt 权重 `alpha` 联合形成 ASD/TD 竞争语义向量。单个线性融合层产生有界 CAM144 残差，`lambda` 初值 `0.1`、上限 `0.25`，并在原始 OutputLayer 调用前经 pre-hook 注入；OutputLayer 仍只调用一次，路径为 `144→36→2`，输出为归一化 log-probability。关闭残差时原 CAM144 原样送入 OutputLayer。

动态原型与前向融合的明确计算为：

`rho_i,c,k = softmax_k(cos(z_i,a_c,k)/0.25)`，`p_c,k = normalize(sum_{i∈S_c} w_i rho_i,c,k z_i / sum_{i∈S_c} w_i rho_i,c,k)`，其中 `w_i` 为训练支持集站点样本量平衡权重；语义投影 `q` 同样按 `w_i rho_i,c,k` 聚合为 `v_c,k`。训练支持 Query 的 LOO 通过从对应类别/子原型的加权分子和分母中精确减去该 Query 自身贡献实现。对两类同时路由：`beta_i,c,k=softmax_k(cos(z_i,p_c,k)/0.25)`；每个子原型独立选择 Prompt：`alpha_i,c,k,j=softmax_{j∈P_c}(s_i,c,k,j/0.5)`。最终竞争语义为 `z_sem=normalize(sum_k beta_i,0,k[sum_j alpha_i,0,k,j t_j+0.5(v_i,0,k-v_i,1-k)]-sum_k beta_i,1,k[sum_j alpha_i,1,k,j t_j+0.5(v_i,1,k-v_i,0-k)])`。融合残差为 `d_i=||h_i||_2 r_i/max(||r_i||_2,1e-8)`、`r_i=W_fusion z_sem`，`h_i^fused=h_i+lambda_fuse d_i`；`lambda_fuse` 通过有界参数化限制在 0 与 0.25 之间。K=1 和 K=3 共享相同的患者/文本投影、融合矩阵与主分类器参数数目。

固定损失为 `L_E74.4(h_fused)+0.02 L_sem+0.02 L_het+0.01 L_site`，新增参数使用独立 AdamW（`lr=5e-4, weight_decay=0.01`）。`L_het` 固定组合归一化类内方差保持、余弦相似度 0.95 上界的温和分离 hinge、子原型利用率 KL 与 LOO 数据紧致性；K=1 的多原型项为零。`L_site` 只对训练折内至少 20 人的类别-站点单元计算分配 KL，并配合站点收缩/权重平衡，不使用测试标签或推理时患者标签。所有组继承同折 E61 warmup 参数、三个原优化器完整状态、拆分/4096关系对与初始 RNG；base-off 单步前向、损失、梯度、更新和 RNG parity 为 10/10。新分支全链梯度、融合 off 精确回退和测试标签不变性预检通过。Fold 2 历史重叠为 1-based 行 377/407；它们留在原基座历史损失中，但从新增原型支持及新增监督排除，训练时新增特征残差梯度关闭。此历史重叠仍是研究限制。

逐任务与 E74.4 历史记录的共同 epoch 公共 RNG 轨迹共核对 10,723 项，10,723 项完全相同、0 mismatch（G1 3,605，G2 3,622，G3 3,496）。没有历史记录覆盖的 epoch 不宣称已逐 epoch 证实 RNG 一致。双步恢复重放在预检折通过，模型、优化器、loss、预测和 RNG 逐值一致。30/30 Checkpoint 均核验 warmup/优化器/初始化身份、训练支持集、选模规则、指标与预测读回、测试标签不变、输出 log-probability 语义及新增 DARG/CAM 梯度；原始审计器的机制汇总因列表容器类型错误曾失败，冻结的训练源码和原始审计文件均保持不变，后由独立审计恢复器修正该汇总并完成 30/30。审计器错误及恢复器 SHA 记录于 `results/audit_recovery.json`，未重训或改写任何模型 Checkpoint。

## 补充结果与结论边界

十折测试预测的混淆计数合计（ASD=0 为正类）：G1 TP/TN/FP/FN=`364/439/29/39`，G2=`360/438/30/43`，G3=`359/441/27/44`。对应逐折最佳 epoch、实际训练 epoch、六指标与混淆计数见 `fold_metrics.csv`；平均每折训练 epoch 分别为 G1 `388.8`、G2 `374.2`、G3 `362.8`。折 2 训练/测试重叠与既有 `val=test` 的历史选模协议仍使这些数值不能解释为无偏泛化估计。

G2 三类路由的类内原型平均余弦距离为 `0.4646`，余弦相似度大于 0.98 的原型对比例为 `0%`；测试患者归一化路由熵为 `0.6721`。类内分配与站点的平均 Cramér's V 为 `0.4440`，低于 E78-G2 约 `0.6703`，但仍存在中等站点关联。子原型间 Prompt TV 为 `0.06667`、患者 Prompt 权重标准差为 `0.04029`，高于 E78-G2 的约 `0.00749`；这支持路由和语义权重具有患者差异，但并不证明被选医学概念有独立临床效用。G2 新增分类残差在首个训练 epoch 对 SH-DCPSS / DARG-GFR / CAM 的梯度范数跨折均值分别为 `0.01735 / 0.00290 / 0.00431`（最小值分别 `0.00589 / 0.00145 / 0.00180`），证明新增分支确实进入主干梯度；这不是显著主干更新或分类收益的证据。最佳状态的融合强度均值为 `0.1037`，CAM 特征修正与原始特征范数比为 `0.03265`。

G2 固定 Checkpoint 干预中，匿名文本替换的平均 ASD 概率绝对变化为 `0.00224`，10 折仅 1 个硬标签翻转且为正确变错误；医学 Prompt 类别交换仅 3 个翻转且全部为正确变错误。关闭子原型/整模块各有 5 个翻转，均为 1 个错误变正确、4 个正确变错误。结合独立训练 G2−G3 的 ACC `-0.2299 pp`、Hard-AUC `-0.1995 pp`、Prob-AUC `+0.0275 pp`，不支持医学文本独立诊断收益。G2−E78-G2 的 ACC/Hard-AUC/Prob-AUC 为 `-0.4598/-0.3929/+0.0807 pp`；相对 E74.4-G2 为 `-0.5747/-0.5763/+0.0165 pp`。G2 没有同时保持基座 ACC、Hard-AUC、Probability-AUC，也未优于单原型 G1；因此尽管 H1–H3 的预设结构性证据成立，完整方法 H4/H5 不成立，最终为 `NO-GO`。不据单 seed、测试集选模结果提出自动扩展或临床泛化结论。
