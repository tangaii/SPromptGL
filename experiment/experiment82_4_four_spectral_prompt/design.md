# Experiment 82.4 · 四组谱–医学 Prompt CAM144 残差

## 版本与实际调用链

论文的 GFR 为患者构建多关系图，CAM 在模态间交互，PEN 使用文本关联局部表征并聚合到全局。论文的 ABIDE 十折数值与本实验的历史 DARG 对照来源各自独立。论文 PDF SHA-256 为 `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`。[MICCAI 2025 论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)

核验原作者提交 `ba3b9446fa451fde6864d6555bdf324de0c4b64a` 和用户 GitHub 提交 `4a86923a28ce9920c0fec28c797c9c6c5933674d` 下 main、model、network、layers、utils。用户提交五文件与服务器字节相同。作者的 network 和 utils 有差异，本地 Git HEAD `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`。历史适配器加载 E72 vendor 中 E51 的 `dugsa.PLNNoPEN`。保留 `InputLayer → encode/feed_forward → reshape(871,144) → OutputLayer`，原头为 LayerNorm144、Linear144→36、GELU、Dropout 0.35、Linear36→2、log_softmax。CAM 的四个模态行各 36 个连续坐标，EncodeLayer 中每行的头输出各 9 个连续坐标。G2 的连续 16×9 分块按原 flatten 组织，表示潜在槽位。[作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [用户源码](https://github.com/tangaii/SPromptGL/tree/4a86923a28ce9920c0fec28c797c9c6c5933674d)

E74.4 `soft_positive_dcr.rudag_forward/forward_state` 保留四路 GraphLearn、共享患者全集图、历史正边支持/候选边、加自环和行归一化，以及四 GCN。Positive-DCR 仅把允许的正邻接增量做局部消息传递并按原模态坐标回写。直接调用历史 forward，原 `out['total'] = cls + graph + .05*pair + .01*adv + .01*preserve`。在真实 Outputlayer 前仅以 C+ΔC 替换 CAM144 实参。历史更新为 Adam base (0.0038, 0.11)、Adam CAM (0.0038, 0.11)、AdamW added (0.001, 0.01)，之后是组内新增 AdamW (0.0005, 0.01)。新增预算控制在新增参数组上，原损失完整回传至所有已训练的 DARG/CAM 参数。

E74.4 通过 E74.1、E69.3 和 E61 warmup loader 恢复。E61 epoch 10 提供主干/CAM/头、三个优化器矩、配置、数据索引、关系对和 Python/NumPy/CPU Torch/CUDA 的 RNG。E74.4 继承原 adaptive 门控初始化及 Fold 2 原历史重采样，按原审计 SHA 复核每折期初状态与 4096 关系对。Fold 2 的 ID 377/407 保留在 DARG 历史身份里，在新增谱监督、Gaussian、校准和保护目标中排除，并令其残差恒等于零。seed 0 持续消费原六次 Dropout。新增初始化使用隔离 CPU Torch 流，内层划分使用独立 NumPy RandomState。确定性教师只用当前原头的冻结权重及确定性计算。

E82.2 的 Original O 使用合法 E61 CAM 均值，按特征值取八个 PCA 方向，对正交补上的 ASD−TD 协方差差按特征值绝对值取八个方向，重根用正交投影的坐标次序消歧，基最大绝对坐标的符号为正。每折复建，与历史 CPU float64 归档逐元素相等。所有组使用固定的 144×16 基。沿用八条医学文本及原拼写、RN50 原 embedding 和 E82.2 的 32 维缓存投影。八条 Query 按无固定类别分组方式参与 16 Token + Null 的 8×17 Attention，Coverage 为八条 Prompt 响应的连续并集。Matched anonymous Query 只用于只读干预。

## 历史证据与四种固定设计

E82.2 的医学反事实对齐改善而最佳权重下直接硬预测影响很小。E82.2.1 正负动作没有带来稳定净纠错，E82.2.2 的双尺度可靠性监督使残差收缩。E82.3.1 的方向约束、E82.3.2 的分类器 Jacobian KL 和 EMA 类统计均作为谱证据参考。E82.3.2 和 E82.3.3 表明零值或标量门控的绝对大小不能解释分类效果。E82.3.4、.5 的有符号方向在错误患者和整体分类上的响应不一致，.6 的受保护小梯度未换来净收益，.7–.8 的谱空间变化引入主干共同适配，.9–.11 的候选/Keep 没能可靠识别该修正谁。这次四组都固定 Original O，均无动作分支或阈值门控。此前单组同编号方案已被用户撤回，仅保存为历史记录。

- G1 DPR-PSCE。Query 保留逐条原核心并减去 0.4 倍共有均值后归一化。原锚的投影和 z 构成可训练的患者角度。两向量由一个固定锚和一个输入依赖的正交单位向量构成，在该平面作解析秩二 Cayley，角度参数初始严格为零。同步旋转坐标、用于构建方向 Token 的基向量和参考协方差，按谱权重与 1/16 的差调制 z 及轻量 Token 修正，再用同一逆旋转还原。合法训练折的 16 个单向遮蔽反事实用旋转后的实际方向。小权重维持转角受限、保护 Prompt 方向轮廓的非退化差异和残差正则。只读旋转关闭用于量化是否被前后旋转抵消。
- G2 HLG-PSCE。保留原 PSCE 谱 scorer 和全局残差，正交基逐方向的平方载荷映射到原 CAM144 的 16 个潜在槽位。共享小网络使用槽位−全局偏差、全局均值和 Prompt 槽位亲和度，产生局部候选。局部投影回 Original O，有限区间 sigmoid 患者门控融合两支，最后统一限幅。合法类标签只参与弱单向遮蔽反事实。能量保留、局部/全局贡献和关闭局部支路均独立检查。
- G3 RCS-DA-SRPE。冻结 E61 Original O，按两类 CPU float64 协方差加入 0.25 对角收缩与相对 ridge、Cholesky 求解、等先验。五折合法行得到不含自身的模型和校准。每个 held-out 分片的校准也剔除该分片标签及生成校准分数的协方差输入，成对排除只是同一套分片上的小型 Gaussian 拟合。全量合法 OOF 用单调无截距标量校准推理。用拟合集最小 Mahalanobis 距离 90 分位估计高置信度支持。组合校准后后验加权的两类 score、符号似然对比、支持度和当前 Tokens，显式与逐方向 Coverage/竞争权重相乘。带非零方向偏置的控制器和连续正确 Margin 保护配合弱 Jacobian KL。区别于弃用单组的近对消 z 与 e 直接平均形式，并以训练当前原头的连续正 Margin 权重保护正确患者。
- G4 SOT-DA-SRPE。在 Original O 内训练可微秩 4 算子。确定性 DCT 正交锚带有界 12×4 图参数，修正前的秩至少 4，用重复正交化得到 V。归属平方载荷把 16 维 Prompt Coverage 映射到四个患者条件系数，乘入带符号的系数控制；类别条件谱候选通过 V diag(a) Vᵀ 再返回 CAM144。沿用冻结 Gaussian 证据与原 KL，新增零、一半、完整残差在真实非线性原头上的连续训练 Margin。可靠正确使用小容忍保护，困难样本用平滑 Margin 目标在零→半和半→全分别限制，以新增分支局部预算参与优化。只读去掉算子的非对角项评估跨方向作用。

## 保存、指标与判断

统一沿历史 epoch 11 开始、最多 450、wait>200、Test ACC 严格提高选 Best、平局保留最早和每 10 epoch 或 Best/终止保留 Latest。两个 checkpoint 含所有优化器、四 RNG、源码身份、固定引用和预测。残差在共同 8% tanh 平滑限幅后返回原头。四组启动前冻结配置，全部组预检通过后分组顺序执行，每波最多 10 workers。

ASD=0，Prob-AUC 用 p(ASD) 对 ASD 正类，和历史 p(TD) 对 TD 正类等价。六项指标为折均值 ± population SD。正式判定要求 ACC、BA 和 SEN-ASD 相对历史 DARG 均正向、净纠错为正且去 Fold 2 后 ACC 与净纠错为正。机制独立要求固定权重残差净纠错、BA 正向、Prompt 替换出现可测概率与硬预测变化、组专属干预改变残差。对当前权重的干预给出即时作用，训练后和 DARG 的差距也包含主干共同适配。CV 前 RFE/缩放、全患者转导图、val=test 选模、Fold 2 重叠和单 seed 均限制泛化解释。
