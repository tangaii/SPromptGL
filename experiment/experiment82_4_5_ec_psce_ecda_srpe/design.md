# Experiment 82.4.5 · Evidence-Calibrated Spectral Residuals

## 研究问题与来源

检验保留历史 DARG 训练主路径、保留原始动作幅度并单独监督双空间真实 Margin 后，医学 Prompt 谱残差能否提高患者硬预测。正式组仅为 EC-PSCE 和 ECDA-SRPE，各十折、seed 0。

原论文以 GFR 多关系患者图传播，再经 CAM 交互并以 PEN 内的 SPGS/RES 从局部语义检索到重新嵌入。历史软件采用 vendored PLNNoPEN。本次使用多模态 → GFR → Positive-DCR DARG → PLNNoPEN CAM → 完整 CAM144 → 校正 → 原 OutputLayer。原头仍用两类 argmax。服务器 HEAD 为 `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`。公开原作者与用户提交分别为 `ba3b9446fa451fde6864d6555bdf324de0c4b64a` 与 `4a86923a28ce9920c0fec28c797c9c6c5933674d`。[MICCAI 2025 原论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [用户仓库源码](https://github.com/tangaii/SPromptGL/tree/4a86923a28ce9920c0fec28c797c9c6c5933674d)

已核对 E61、E74.4-G2、E82.2-G2、E82.3.2-G2、E82.4–82.4.4 的模型、配置、历史检查点身份、正式报告和独立审计。E82.2-G2 与 E82.3.2-G2 的 ACC 都是 92.0742%，相对 DARG 各净少判对 1 人。由 82.4.4 归档逐患者复算，G1 ACC 91.9606%，原路径净 −2 与直接残差净 0。G2 ACC 91.5008%，原路径净 −10 与直接残差净 +4。原错测试行的平均真类 Margin 变化为 −0.02240 与 −0.18057。G2 仅谱内、仅谱外的原错行折均变化分别为 −0.03585、−0.15586。82.4.4 的 8 个组别×历史参数族中，每族至少 99.3% 的步达到 4.99% 的历史分类梯度范数。

82.4.4 在动作向量超过下限时把它单位化。同向重缩放几乎被抵消，门控可以给小动作带来近似大动作的位移。其 G2 对两个 Jacobian 动作共享同一个 LLR logit 偏置，而该证据只由谱内坐标估计。同向增加历史分类梯度会改变总损失各项的比例，也会改变 Adam 一阶和二阶矩以及后续参数。小范数切空间注入降低当前梯度扰动，对参数轨迹只提供一期几何约束。逐步实测优化器位移、与历史总梯度及附加量的内积，另在预检、Best、Latest 计算同一状态下基于历史梯度的 Adam/AdamW 更新，用于比较实际状态依赖的差异。

## 历史训练与切空间更新

逐折从 E61 epoch 10 的原模型、三优化器、历史索引、4096 关系对及 Python、NumPy、Torch CPU、GPU RNG 恢复。历史 forward 的原头输入就是原 CAM144，保留同序的六次随机 Dropout。增强头直接复用同轮原参数，依次使用 LayerNorm、Linear、GELU、Linear。历史总损失为 `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve`。历史 Adam、Adam、AdamW 依次更新，随后新 AdamW(0.0005, 0.01)。历史学习率、权重衰减分别为 (0.0038, 0.11)、(0.0038, 0.11)、(0.001, 0.01)。

对 GFR、DARG、CAM、OutputLayer 各族令 `b=∇L_hist`、`q=∇L_full`。若 `‖b‖ ≤ 1e−12`，保留 b 并记安全跳过。其他步用 float64 计算 `u=b/‖b‖`、`t=q−(q·u)u`。归一前的安全分支负责除零保护。clip 上界为 `0.005‖b‖`。写回前在上界预留原 dtype 舍入的 L2 界，写回后以 float64 复算实际差向量和原尺度内积误差界。超过预算或舍入界时保留完整 b，记录原因。在新增族中 Full CE 系数为 1，四个辅助损失只加给所属新增族，实际梯度逐项限制为新增 Full 梯度的 5%、3%、2%、2%。停用新增部分后预检核对历史输出、损失、梯度、优化器状态、RNG 和历史下一步。

## 真实决策目标

ASD=0，TD=1，s=1−2y，m=logit_ASD−logit_TD，真实增强增益 `G=s[m(C+Δ)−m(C)]`。Full CE 的 ASD/TD 权重相等。唯一困难度记忆只保存同折合法训练行的 `w0=sigmoid(−s m_epoch10(C)/0.5)`。与同轮无 Dropout native 计算的 `w_now=sigmoid(−s m(C)/0.5)` 用 `focus=1−(1−w_now)(1−0.25w0)` 融合。保护权重为 `(1−w_now) exp(−|s m(C)|/0.5)`。Gain 和 Guard 使用实际真类 Margin 的 `softplus(−G/0.10)`。G2 的 Gain 用联合、单谱内、单谱外分别 1/2、1/4、1/4 混合，各候选从真实特征和当前原头计算，Guard 约束最终联合位移。正则为合法行相对位移平方。Gain、Guard、正则、指导系数为 0.002、0.005、0.0005、0.001。基准 Margin、难度权重和证据相容权重被固定在同轮参考值。原评估链之外的监督、困难记忆、CF 与 Gaussian 拟合和更新只读合法训练标签。Fold 2 历史重叠 377/407 的旧图成员和关系对保留，新增残差严格为零。

## G1 EC-PSCE

继承 E82.2-G2 合法 warmup CAM 的 Original O U16，前 8 PCA、后 8 类别协方差差异方向、白化、16 Token、八条原医学 RN50 Query 和 Null，保留历史拼写与次序。原语义覆盖 w_sem、统计 w_sp 和连续 beta 混合得到 w。(e_j) 为第 j 方向后四条异常 Prompt 注意力均值减前四条正常 Prompt 均值。显式输入使用 `tanh(e_j/0.10)`，保留绝对差异尺度。动作 MLP 同时接收白化、Token、两种权重、beta、Margin、谱内 Jacobian 和患者 Query 上下文。单 sigmoid 门控接收容量和可纠错性作为输入。

`a=tanh(MLP)`，`raw=0.08‖C‖ g [4(w⊙a)]U16ᵀ`。有符号动作绝对值直接传入位移。CF 用合法训练 CAM 的 16 个受限遮蔽前向生成冻结方向目标，只监督语义方向分配。contrast-off 令显式对比输入为零。signed-action-off 以绝对值替代有符号动作，保留每方向大小。Semantic-off 切断文本身份对残差的影响。

## G2 ECDA-SRPE

按 E82.3.2-G2 方法使用合法类均值/方差、收缩、样本方差校正、8 限幅 Gaussian LLR 与可靠性。保留合法训练五折交叉拟合银行和全合法推理银行，float64 EMA α=0.1。每步先评估，再用合法 eval CAM 更新 EMA。Best 保存其评估态，Latest 保存下一轮状态和最近评估态。

利用当前原头无 Dropout Margin 的 VJP 分解 (J_P=UU^T J)、(J_Q=J-J_P)。载体分量各用下限 0.01 安全单位化。两套独立动作 MLP 共用患者谱/Prompt 与证据上下文，各自预测符号与大小。谱内 `s_P=tanh(a_P+0.15 tanh(E/0.30))`，谱外 `s_Q=tanh(a_Q)`。输出层初始化标准差分别为 0.035 和 0.0035，两条路径均可训练。用一个 sigmoid 门控并以 `raw=0.08‖C‖ g(s_P unit(J_P)+s_Q unit(J_Q))` 生成增强特征。指导只在证据与合法真类方向一致时弱加权谱内候选的真实增益。移除显式先验干预只去掉谱内加法项。单空间干预保留原门控和对应动作，直接移除另一位移并重新平滑限幅。三种位移幅度可以不同。

两组均用 `Bound(raw)=raw*tanh(r)/r`，(r=‖raw‖/(0.08‖C‖))，近零采用级数和安全分母。没有整体动作单位化或残差下限。单独审计缩放动作、零 CAM 和近零载体，记录实际非零比例、残差与门控分位、原错和原对 Margin、跨界距离及硬翻转。

## 正式调度与结果

epoch 11–450，wait >200，严格首次最高 test ACC。Best 改进更新，Latest 在 epoch 10 留恢复锚点，随后每十轮和终止时更新。若 Best 先于下一次 Latest 提交，续训先验其已持久化日志，只从 Latest 重放未提交段并以状态 SHA 核验该 Best。每折最终只保留 Best/Latest。固定同权重干预只用于训练后读出。只用物理 GPU0，两波顺序，最多 10 workers，各一个数值线程。历史 val=test、CV 前 RFE/预处理、转导患者图、Fold 2 重叠与单 seed 限制外部泛化。未四舍五入的五项十折指标都达 DARG 且净正确不低于基座才为严格追平。实测同权重正净纠错或正 BA 与端到端谱及 Prompt 路径共同构成机制通过。候选还要求完整 ACC、BA 和净纠错为正，排除 Fold 2 后维持，以及医学 Query 相对匿名 Query 的真实分类增益。
