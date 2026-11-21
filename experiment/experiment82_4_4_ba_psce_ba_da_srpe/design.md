# Experiment 82.4.4 — Baseline-Anchored Twin-Path Optimization

## 源码与假设

本实验检验保留 DARG 历史主分类梯度后，有真实分类监督的医学 Prompt 谱残差能否在同一权重下提高 ASD/TD 决策。仅正式训练 BA-PSCE 与 BA-DA-SRPE 各十折。

原论文 §2.1 以各模态相似度形成多关系图，使用关系图卷积传播，优化图结构与标签分类损失。§2.2 的 CAM 融合跨模态特征，PEN 的 SPGS 将文本和局部特征匹配，RES 将局部结果嵌入全局上下文。论文报告 ACC、AUC、SEN、SPE。原仓库根据硬预测计算 AUC，二分类下等于 BA。本实验也报告概率 ROC-AUC。已读原作者提交 `ba3b9446fa451fde6864d6555bdf324de0c4b64a` 和用户提交 `4a86923a28ce9920c0fec28c797c9c6c5933674d`。服务器 Git HEAD 为 `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`，五个运行相关根源码的字节与用户远程一致，SHA 见配置。 [MICCAI 2025 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者仓库源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [用户仓库源码](https://github.com/tangaii/SPromptGL/tree/4a86923a28ce9920c0fec28c797c9c6c5933674d)

历史 E74.4-G2 以 E61 epoch 10 的完整状态开始，实际使用 GFR、Positive-DCR、PLNNoPEN CAM、原 OutputLayer。计算链保持多模态 → GFR → DARG → CAM144 → 校正特征 → 原头。历史原头为 LayerNorm(144) → Linear(144,36) → GELU → Dropout(0.35) → Linear(36,2) → log-softmax。历史选模和输出继续以原始类别 argmax 执行。

E82.2-G2 的 U16 含合法热启动 CAM144 的八个 PCA 方向及八个类别协方差差异方向。语义/统计权重和受限方向遮蔽教师来自该方案。E82.3.2-G2 的 Gaussian LLR 与后评估 EMA 保留，原方案的 Jacobian 用于生成方向训练目标。E82.4.1 与 E82.4.2 将 Jacobian 引入实际残差。五折交叉拟合银行沿用 E82.4.2-G2 合法划分和 E82.4.3 的统计实现。E82.4 到 E82.4.2 出现真实残差或更高谱监督指标，但主干共同适配抵消部分收益。E82.3.2 新分支在开始时分类梯度弱，方向 KL 相对梯度峰值很高。E82.2 的反事实对齐改善与硬预测增益是两个独立终点。

E82.4.3-G1 有效门控是可学习 sigmoid 与可纠错性的乘积，可纠错性随当前绝对 Margin 超过一阶扰动能力下降。该因子同时进入增益监督权重，原正确保护和主分类可进一步降低纠正强度。观测到训练行第一步与 Best 平均门控从 0.21030 降至 0.00535，测试残差比为 0.0247%。E82.4.3-G2 还将冲突与可靠度乘入门控，有符号连续证据也参与共同幅度，两种衰减同时出现。平均训练门控从 0.09829 降至 0.00437，测试残差比为 0.0281%。Gaussian LLR 对固定谱维的边缘分布建模，类内协方差和患者条件分布可偏离该近似。Jacobian 表示增大 ASD 相对 TD 分数的局部方向，是否应当增大该分数由患者真类决定。用单一 E 的符号施加两个 Jacobian 分量时，统计证据出错可使真类 Margin 下降。82.4.3-G2 判错测试行平均真类 Margin 变化为 −0.00083。

82.4.3 历史 CE 使用增强特征，Native CE 以系数 0.10 与逐族 10% 施加上限做辅助。梯度确实连到原各族。DARG 到当前 Residual-off 仍净损失 G1 4 人、G2 2 人。这个对照同时含训练轨迹和各自 Best 轮次，两组的残差改动也不同。由这些对照可观察保护不足以消除退化，单独的 Native CE 因果贡献需要额外设计识别。82.4.4 把历史随机训练链作为无改动的主目标。

## Twin-Path 优化

每折恢复 E61 epoch 10 的 GFR、DARG、CAM、头、三个 Adam/AdamW 状态、合法历史关系对和 Python/NumPy/CPU/GPU RNG。新增初始化在 CPU 隔离 RNG 内完成，组及折唯一种子键由固定 seed 0 派生。历史前向在原 CAM 上产生六次 Dropout，并原样优化 `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve`。同一 CAM 图继续计算模块和当前无 Dropout 头的 Full logits。只有合法外层训练标签参与 Full class-balanced CE 和所有新增有标签目标。无 Dropout 头复用原参数且保持可微。

设 `n` 为一个历史优化器参数族的历史分类梯度，`q = 0.10 ∇L_full`。去冲突后的向量为 `q' = q − min(0, q·n/‖n‖²)n`。每族实际附加梯度的范数至多为 `0.05‖n‖`，方向与 `n` 的实测内积非负。用 float64 计算投影、预算和 float32 梯度实际差值。投影正交边界留出理论舍入余量，记录梯度家族的原始/投影/施加范数、余弦、倍率和回退。历史总梯度保留作为增量基底。新增模块接受系数 1 的 Full 分类梯度，再在其基础上逐项施加新辅助梯度。各优化器按 opt_base → opt_cam → opt_added → 新 AdamW 顺序更新。包括新优化器在内均保持历史全局 RNG。单一正式推理头始终处理增强 CAM。

令 `s_y = 1 − 2y`，`m(C) = logit_ASD(C) − logit_TD(C)`，`Δm_y = s_y[m(C_full) − m(C)]`。以同轮权重的原确定性分类作为固定参考。增益和保护分别用 `sigmoid(−s_y m(C)/0.5)`、`[1 − sigmoid(−s_y m(C)/0.5)] exp(−|s_y m(C)|/0.5)` 加权 `softplus(−Δm_y/0.10)`。另有平方相对位移。各项 ASD 与 TD 等权。增益、保护、残差正则和方向指导系数为 0.002、0.005、0.0005、0.001，其实测梯度预算分别为新增 Full 主分类梯度的 5%、3%、2%、2%。G1 的方向指导仅进入 Prompt 族。历史四族仅接受 Full 分类支路的小幅梯度。预算和数值规则在正式训练前冻结。

## G1 BA-PSCE

用八条固定历史 RN50 Query（含 `Nromal brain volume`）为 16 个谱 token 和一个 Null 评分。使用同组原始拼写、顺序、缓存投影及匹配的匿名 Query。对每个方向归一化八条 Query 的聚合覆盖得到 `w_sem`，谱 MLP 读取白化坐标幅度、方差和疾病协方差描述子生成 `w_sp`。Null 与按 log(16) 归一的熵控制连续 β。方向的重要性 `w = lerp(w_sp,w_sem,β)` 与带符号动作 `a∈[−1,1]^16` 分别计算。

动作 MLP 读取语义 Token、两类谱权重、标准化谱描述子、当前 Margin 与谱内 Jacobian。患者门控 `g = sigmoid(MLP(context))`，context 含可纠错性与一阶 Margin 容量。原位移 `raw = 0.08‖C‖ g U16 Normalize(w⊙a)`。归一化分母下限为 0.001，可使近零动作连续缩小。一阶容量只作特征。沿用 E82.2-G2 的受限遮蔽教师，对合法且教师正确的患者逐方向从 CAM 移除受限正贡献，分配 detach 的 Prompt 目标。教师复用同轮无 Dropout 的原头参数，只是无梯度的 17 个合法视图批次。

## G2 BA-DA-SRPE

五个合法训练交叉拟合银行和全合法推理银行均采用方差收缩、下限、有界 Gaussian LLR 和可靠度校准。用与母方案一致的 float64 计数除法、均值和二阶矩更新，避免上一轮仅计数比值在 float32 中舍入。每轮原生和增强训练、四个优化器更新、以先前 EMA 评估和选 Best，最后用合法 eval CAM 做 0.10 EMA。Best 保存实际生成预测的 EMA。Latest 保存已更新 EMA 以及最近评估使用的快照。

Gaussian LLR 为 `log p(z|ASD) − log p(z|TD)`，限为 `8·tanh(LLR/8)`。Normal 和 Abnormal Query 覆盖在每个方向按连续 LLR 融合，以可靠度加权求和形成患者证据 E。方向 MLP 读取白化谱状态、Prompt 覆盖与语义、可靠度、LLR、Margin、Jacobian 尺度和当前冲突。两维输出加 `0.15·tanh(E/0.30)` 作为连续的弱先验，然后分别 tanh 得 `s_parallel,s_perp`。两个结果允许和疾病证据相反。用单个患者 sigmoid 门控与相同位移尺度作用于 `Normalize(s_parallel Unit(J_parallel)+s_perp Unit(J_perp))`。先将每个 Jacobian 分量以 0.01 为下限归一，再以 0.001 作整体归一化下限。VJP 的 Jacobian 载体 detach，原 CAM、Prompt 和最终原分类头继续接收 Full CE 的梯度。证据一致性辅助项以合法标签与可靠证据兼容的连续权重约束 `J·ΔC` 的方向。关闭证据先验只去掉方向 logits 的加法项。谱内/谱外干预独立把另一个原位移分量置零并重新限幅，归一化和门控保留 Full 的系数。

## 验证与解释

两组共用 `ΔC = raw·tanh(‖raw‖/cap)/(‖raw‖/cap)`，`cap=0.08‖C‖`，近零使用级数极限。由于门控和归一化，raw 已受 cap 限制，审计分别给出 raw 超出和 tanh 造成至少 5% 缩减的比例。活跃残差报告相对范数 > 1e-5，低门控报告 < 0.01。这些阈值只用于诊断。Fold 2 的历史 1-based 行 377 ASD、407 TD 仍进 DARG 及关系对，新增统计与标签排除，两行新增残差严格为零。合法训练索引和固定谱/文本缓存均有 SHA。

最大 epoch 450，wait > 200，严格使用 test-fold ACC 首次最大值保存 Best。保持历史 val=test、CV 前 RFE/预处理、转导患者图及原十折。三历史学习率和权重衰减分别为 Adam(0.0038,0.11)、Adam(0.0038,0.11)、AdamW(0.001,0.01)。新优化器 AdamW(0.0005,0.01)。物理 GPU0，运行两波，每波至多十 workers、每个一个 CPU 数值线程。Latest 在 Best 提升、每十轮、终止时事务提交。审计在新对象加载完整状态，再重放原 Dropout 序列、下一步训练和评估。度量从患者标签、硬预测和概率独立复算，并验证精确的历史选模与固定权重干预。初始 AdamW 步数别名在加载新增优化器时用私有副本隔离。

严格 DARG-PARITY 使用未四舍五入的五项十折指标与净纠错，另列 ACC/BA 的基础门槛和排除 Fold 2 的结果。NONTRIVIAL-RESIDUAL 需真实残差可测、有正净纠错或正 BA 增益且端到端。候选进一步要求 ACC、BA 和净纠错均正，排除 Fold 2 后仍正，同时医学相对匹配匿名在同权重下具有正 ACC、BA 和净纠错及硬翻转证据。单 seed 和历史评价协议限制外部泛化解释。
