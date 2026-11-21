# Experiment 82.4.6 · PCSR-PSCE & SCMR-DA-SRPE

## 来源与训练协议

原 SPromptGL 的 GFR 用多关系图传播，CAM 推进模态交互，PEN 用 SPGS 检索语义并以 RES 聚合局部和全局表征。历史训练使用 vendored PLNNoPEN。实际路径为 PHENO / ANAT / FUNC / CORR → GFR → E74.4-G2 Positive-DCR → PLNNoPEN / CAM → 完整 CAM144 → CAM144 + 新残差 → 原 OutputLayer。该头为 LayerNorm(144)、Linear(144,36)、GELU、历史 Dropout、Linear(36,2)、log_softmax。新分支复用原头权重的确定性计算。服务器 HEAD 为 c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4。用户 main 的五份核心源码与服务器逐字节一致。与原作者仓库的 network.py、utils.py 差异见配置 SHA。E82.4.5 指定上传分支 1685b92b1f42ce463a5b06681596af8e0d3092d2 的指定源码、配置、报告、机制和患者级文件均与历史权威文件逐字节一致。[原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者仓库](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [用户仓库](https://github.com/tangaii/SPromptGL/tree/4a86923a28ce9920c0fec28c797c9c6c5933674d) · [82.4.5 上传分支](https://github.com/tangaii/SPromptGL/tree/1685b92b1f42ce463a5b06681596af8e0d3092d2)

E61 epoch 10 的模型、三优化器、四类 RNG、4096 关系对和外层患者成员逐折恢复。只用 seed 0，从 epoch 11 训练至 450 或 wait > 200。Best 为历史 val=test ACC 首次严格最大。保留转导患者图、CV 前 RFE 与预处理及 Fold 2 既有重叠。1-based 377 ASD 和 407 TD 保留原图与历史损失成员身份，屏蔽新增监督、统计和校准，新增残差严格为零。原损失是 L_cls + L_graph + 0.05L_pair + 0.01L_adv + 0.01L_preserve。历史六个 Dropout 按归档顺序回放。两组新模块用局部 CPU 随机流初始化。历史 Adam(0.0038,0.11)、Adam(0.0038,0.11)、AdamW(0.001,0.01) 顺序执行，然后独立 AdamW(0.0005,0.01)。Best 改进时写入，Latest 每十轮和终止时写入。若 Best 先于下一次 Latest 持久化，恢复用已核验的 Latest 重放持久后缀。仅物理 GPU0，顺序两波，每波十 workers 上限，CPU 数值线程 1。

## 残差差分梯度与共享参数保护

令 C 为同轮历史 forward 的 CAM144，h 为同一原头的无 Dropout 计算。合法训练集合内，F = BalancedCE(h(C+Δ), y)，O = BalancedCE(h(C), y)。新参数直接接受 ∇F。历史 GFR、DARG、CAM 和 OutputLayer 每族收到完整历史总梯度 b 加受限切向差分。差分 q = ∇(F−O)，在同一图的一次历史 VJP 中计算。令 u=b/‖b‖，t=q−u(q·u)。用 float64 投影，预算 0.001‖b‖。写回前留出原 dtype 舍入余量，写回后以 float64 复算附加量的范数和原尺度正交误差。近零 b、非有限 q 或舍入检查失败时保留 b 并记录。验证恒等零残差与结构性关闭下的 L_full、L_off、完整历史更新和 RNG。独立计算原始 Full、Off 梯度和差分，再核对差分切向加量及优化器实际参数位移。Adam 自适应矩会放大或重排参数运动，近正交约束直接作用于瞬时梯度。

历史 82.4.5 的同权重新增残差两组各纠正 1 人。残差关闭时相对 DARG 净 −2、−4，原错患者加残差后平均真类 Margin −0.00510、−0.07203。上一轮给共享族的 q 是 ∇F，故 Δ 为恒零时该共享向量仍可非零。

## G1 PCSR-PSCE

继承 E82.2-G2 合法 warmup 的 Original O、8 个 PCA 加 8 个协方差差异方向、中心与统计量、白化、16 谱 Token、八条历史 RN50 Query、Null、w_sem、w_sp 和连续 beta。保留八条 Query 原始内容与顺序。归一化原 Query q 的公共方向为 c=Unit(mean(q))，保留 q 的原始注意力。差异 Query 为 Unit(q−η(q·c)c)，η=0.50 sigmoid(θ)，初始 0.25。两路八条 Query 各做 16+Null 注意力，注意力均匀融合，上下文按相应路的注意力加权。正常/异常四对四差异用 tanh(contrast/0.10) 定标。医学与匹配匿名统一变换。关闭差异表征时用原 Query 同时进入两路。

在两个固定、互相正交的 PCA–疾病坐标平面 (1,9) 和 (2,10) 分别学习 Givens 角 0.10 tanh(MLP(white, Prompt, contrast, w))。初始精确恒等，矩阵偏离恒等的秩至多四，全谱跨度保持 U16。获得逐轴动作 a=tanh(direction controller)，患者幅度 A=tanh(amplitude controller)。A 的最后一层全零，方向输出以小幅非零初始化。原始位移为 0.08‖C‖ A U16 R [4(w⊙a)]。幅度输入包含证据、Margin、边界距离、对比强度和谱响应与 Jacobian 的一致性。保持绝对动作幅度，允许取零。signed-action-off 取逐方向动作绝对值，随后按干预后的方向重新计算患者幅度所用的 Jacobian 一致性。真实头联合 Margin 增益为主要辅助方向训练。对比指导以合法标签和冻结标签一致的对比权重对各轴真类 Jacobian 位移作软惩罚，预算受限。零幅度初始化使部分上游梯度在最初步为零，后续真实训练步独立验证接通。

## G2 SCMR-DA-SRPE

继承 E82.3.2-G2 的合法五折 Gaussian LLR、方差与可靠度、全合法推理银行和 float64 EMA α=0.1。保持先评估后更新，Best 存评估态，Latest 存后更新态与最近评估态。新增不一致度使用各银行的有界 LLR 标准差。推理比较五个合法子银行。训练样本用其交叉拟合银行 k 获取主证据。比较集合由 k 和其余四个同时排除 k、j 的统计组成，基于现有计数及一二阶矩经 S_k+S_j−S_full 得到。入选比较银行排除本样本所在内折。交叉拟合仅适用于对应类别统计。

校准器接收白化谱状态、Gaussian LLR/可靠度、Prompt 覆盖及正常异常对比、原 Margin、冲突、两空间 Jacobian 响应和银行不一致度。共享 24 维隐藏层输出三项。target=m+0.75 tanh(l_target)，可靠度 r=σ(l_r)，分配 A=σ(l_A)。target 输出行初始化为零，可靠度 bias 初始 −1。desired=r*(target−m) 通过乘显式 shift 计算，避免近似相等 Margin 的浮点相减。两空间候选为 A·desired·J_parallel/(‖J_parallel‖²+λ) 和 (1−A)·desired·J_perp/(‖J_perp‖²+λ)，λ 为历史 Jacobian 安全尺度 0.01 的平方。一阶效应共享符号，联合与单空间均用真实原头复算。calibrated_evidence_off 用 |shift| tanh(E/τ) 替换学习 shift 测试未校准方向，保留可靠度和分配，连续证据绝对值会衰减动作。conflict-off 将冲突特征设为 0.5。单空间干预关闭另一候选，再对保留项平滑限幅。

## 目标、审计与指标

两组使用原 8% CAM 相对范数平滑 L2 限幅。辅助只更新新增参数，系数 0.002 / 0.005 / 0.0005 / 0.001，新增分类梯度逐项预算 5% / 3% / 2% / 2%。真实真类 Margin gain 用当轮困难度和合法 epoch 10 固定困难度。Guard 加权原对且靠近边界患者。G2 Gain 用联合/谱内/谱外权重 1/2、1/4、1/4。G2 校准监督为类别平衡真类 target softplus 和弱可靠度 BCE。可靠度软目标由合法真类符号、冻结原 Margin、证据一致性和校准需求得到。高不确定证据下惩罚无必要位移。测试标签只进入历史选模和独立报告。

逐患者复算 ACC、BA、Prob-AUC、SEN、SPE、ASD-F1、Brier、ECE，另报排除 Fold 2 九折。审计首严格最高 ACC、Latest 下一步、实际梯度投影、历史 RNG、患者配对和零残差不变式。DARG-PARITY 采用未舍入五项均值和非负患者净正确。创新候选还需同权重正净纠错或 BA 增益、医学 Query 相对匿名正向分类、完整 ACC/BA/净正确在两范围均增加，且 ACC/BA 胜折不少于负折。单 seed、val=test、转导图、Fold 2 历史重叠和 CV 前 RFE/预处理限制外部泛化。
