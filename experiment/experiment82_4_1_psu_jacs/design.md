# Experiment 82.4.1 · Prompt-guided Signed Utility and Jacobian-Aligned Spectral Correction

## 冻结前的来源核验

原论文的 GFR 使用多关系患者图及 GCN。CAM 交互模态，论文中的 PEN 将原八条文本关联到局部特征并聚合。服务器 DARG 实际用 E72 vendor 的 `PLNNoPEN`。四个连续 36 维 CAM 结果在 `Outputlayer` 前展平为 144 维。本实验在这一入口做特征修正，使用原 LayerNorm144、Linear144→36、GELU、Dropout 0.35、Linear36→2、log_softmax。论文 PDF 与本地文件字节一致，SHA-256 `219ceddfd017e6105490a72864dd24df0bdbbf1c0579cf14af5bbd99a5d60c89`。[MICCAI PDF](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)

五个根文件 `main.py model.py network.py layers.py utils.py` 与用户提交 `4a86923a28ce9920c0fec28c797c9c6c5933674d` 字节一致。原作者提交 `ba3b9446fa451fde6864d6555bdf324de0c4b64a` 的 network、utils 与服务器存在已知差异。服务器 Git HEAD `c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4`。真实训练入口由 E74.4 经 E74.1、E69 和 E61 适配器恢复。[原作者固定版本](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [用户固定版本](https://github.com/tangaii/SPromptGL/tree/4a86923a28ce9920c0fec28c797c9c6c5933674d)

在 E74.4 的 `rudag_forward` 中保留四路 GraphLearn、历史邻接支持、加自环归一化和四路 GCN，Positive-DCR 仅传播允许的正增量。原 `out['total']=cls+graph+.05 pair+.01 adv+.01 preserve` 由历史入口计算。恢复 E61 epoch 10 的模型、四类 RNG、原始 4096 关系对和三个优化器。更新顺序为 base Adam 0.0038/0.11、CAM Adam 0.0038/0.11、added AdamW 0.001/0.01、新分支 AdamW 0.0005/0.01。关闭新增分支逐折复核首步原 forward、loss、梯度、优化器、RNG 及六次训练 Dropout 调用签名。

E82.4 的 80 份 checkpoint、40 份已审计的折记录、发布源码及全部结果文件 SHA 重新比对。362 项底层保护文件的集合仍与其训练后清单一致，前一实验 manifest SHA 为 `2f06559aadf2c2a2fdd1429368027b60007267f3eaa9643bb5889841222d1384`。E82.4 G1、G3 的同权重直接净纠错为 +1、+2，完整模型相对 DARG 的净纠错为 −1、−5。G2、G4 的同权重残差硬预测翻转均为 0。总性能、同权重残差收益与 Query 干预分别评估。

## 共享谱、Prompt 和梯度

E82.2 固定的 Original O 以合法 E61 CAM 训练行均值为中心，八个 PCA 加八个正交补上的类别协方差差异特征方向。每折复建并与缓存逐元素核对。保留原 RN50 八条固定医学 Query 与 16 Token、Null、8×17 Attention，连续覆盖为八条响应的并集。用固定秩 8 正交坐标表示 32 维缓存的真实 Query 子空间。原 32 维 Query 与匿名 Query 均保持原字节，映射后核验重建与 Gram。键与 Null 在有效 8 维求积，避免加入无法被任一固定医学 Query 读取的自由维度。每个方向的 Query 消息由其注意力加权得到，和覆盖及当前 Token 一起输入有符号控制器。两组均使用真实的 CAM144 特征级残差及 8% 平滑范数限制。

原分类头接收 C+ΔC。新增分支只在反传中将 C 的额外梯度乘 0.25，通过显式 identity autograd 实现逐位未改的前向。直接路径的梯度保持原值，新增参数接受主分类全梯度。辅助梯度只加在新增优化器拥有的参数。合法训练行上的确定性原头 margin 具有类别平衡的困难权和正确保护。方向约束用实际应用残差与真实类投影梯度的稳定余弦，增益使用困难权 softplus，保护对正 margin 用连续权和光滑平方惩罚，弱正则为相对残差能量。四项系数/梯度预算分别为 .002/.05、.002/.05、.005/.03、.001/.02。头的副计算禁用 Dropout 且参数冻结。

## 两组正式方法

- G1 PSU-PSCE 从 E82.2 的重要性与单向遮蔽信息转向实际有符号特征效用。共享 24 隐单元逐方向以 Token、八条 Query 的方向消息、标准化坐标和 Coverage 生成 signed action。用 Coverage 与均匀 Coverage 比值做方向尺度，经 tanh 产生正负作用，患者强度为学习的 sigmoid。CAM 范数与固定 0.12 尺度使非零初始化在约 1% 量级。训练时由当前冻结原头的合法行 VJP 和实际增强 margin 直接约束施加到 CAM 的变化。只读 unsigned 保留方向绝对幅度与患者强度，清除学习出的负号。
- G2 JACS-DA-SRPE 继承 E82.3.2 的类别条件统计思想，将原分类头对 d=logit_ASD−logit_TD 的批量 VJP 投影到 U。确定性 VJP 从一批 CAM 求和得到实际逐患者方向，脱离高阶图，以范数加 .01 稳定归一化。医学 Query 与 EMA 证据决定逐方向 tanh 系数。正系数沿 ASD logit 差梯度方向，负系数反向，当前类别预测和标签不输入方向网络。固定尺度和患者 sigmoid 强度乘在系数与投影 Jacobian 的逐元素积上。只读 Jac neutral 用等范数各向单位载体替换投影方向并将其输入通道置零。Reliability-off 保留统计证据并置其可靠性为一。
- G2 在固定 U 上以 E61 合法行估计训练起始类均值、逐方向总体方差。六个状态分别覆盖全训练折及五个排除对应合法内部留出组的估计。分层内折使用独立 RandomState。训练前向读取上一步 EMA，每次历史四优化器更新后用同一步训练 CAM 的合法行摘离计算下一 EMA。方差按两个总体矩混合式保留均值漂移项。推理固定使用全合法 EMA。按 20% E61 方差锚收缩，依类别差异、类内抽样不确定性及患者离群程度连续缩小证据。原类统计和逐方向标准化差通过可靠性因子进入 F 与 signed action，未采用完整 Gaussian LLR。Best/Latest 保存 EMA 当前均值、方差、计数和固定内折身份，日志校验每步前后链。

本实验用 Prompt 有符号输出和实际分类效用取代 E82.4 G1 旋转、G2 局部槽位、G3 冻结全协方差及 G4 秩 4 变换，同时单独控制残差对主干的附加更新。E82.2、E82.3.2 及 E82.4 Best 仅作为冻结几何诊断与既有历史比较来源。

## 冻结的预检、诊断与判断

保留 seed 0、ABIDE 871、PHENO48/ANAT6/FUNC10/CORR256、历史划分、val=test 选 Best、共用转导图、CV 外预处理和 RFE、epoch 11–450、wait>200、严格提高 ACC 才取 Best。Fold 2 ASD 377、TD 407 保留历史成员，新增有标签统计与监督从合法训练索引构建，新增残差恒为零。预检由单 GPU0 worker 实测两组资源，同一物理设备最多十并发。按两波各十个正式折执行。

只读几何诊断预先固定使用 DARG、E82.2-G2、E82.3.2-G2、E82.4 四组真实 Best。用未增强 CAM 的 ASD−TD 标签无关梯度计算投影覆盖、残差余弦、线性和真实非线性 logit 差变化。训练正确性分层使用合法训练标签。测试覆盖在训练前按所有测试行计算，测试正确性分层保留到全部正式训练后的描述性审计。此处的正确性指完整归档模型的硬预测和参考标签一致。几何统计不参与初始化、调参或选模。

历史与正式 ASD=0，Prob-AUC 使用 p(ASD)，硬预测 AUC 等于 BA。性能 GO 要求相对 DARG 的 ACC、BA、SEN、病例净纠错均正向，剔除 Fold 2 后 ACC 与净纠错仍正向。机制 GO 要求同权重残差净纠错及 BA 正向，Gram 匹配 Query 替换产生可测概率和硬预测变化，每项指定的组专属干预改变实际特征。任何固定权重输入干预均为当前模型的即时反事实分析。医学文本独立诊断效用仍需匹配的正式文本对照。
