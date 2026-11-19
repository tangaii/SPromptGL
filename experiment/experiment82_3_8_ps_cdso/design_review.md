# Experiment 82.3.8 · PS-CDSO 设计冻结

## 历史依据

原论文 §2.1–2.2 及式 (7) 将关系患者图、动态图卷积、跨模态 CAM、局部与全局 PEN 和分类联合训练。原作者的 CAM 将 144 维输出交给 LayerNorm、Linear、GELU、Dropout、Linear、log_softmax。E74.4-G2 使用历史 PLNNoPEN。其关系注意力与消息传递、4096 个 Pair、原 CAM 和 0.35 分类 Dropout 按历史调用顺序执行。恢复链为 E74.4 → E74.1 → E69.3 → E61 epoch10，E74.1 恢复时执行历史固定重参数化。两个 Adam 的学习率 0.0038、weight decay 0.11，关系 AdamW 的学习率 0.001、weight decay 0.01，按原顺序恢复与执行。原总损失为分类 NLL + 图损失 + 0.05 Pair BCE + 0.01 条件对抗 + 0.01 关系保留。 [论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者代码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [SABER 工作区版本](https://github.com/tangaii/SPromptGL/tree/c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4)

E82.3.5 的原随机头接收真实增强 CAM，直接纠错为 +1/+3，整体净纠错为 −5/−12。E82.3.6 严格保护 DARG，将残差训练放在确定性旁路，历史错分患者仅 4.41% 的真类 Margin 提高且零硬纠错。E82.3.7 使用同一真实 Dropout 掩码评估原 CAM 与增强 CAM。G1 的 Bhattacharyya 距离读取完整协方差，残差读对角 LLR。G2 的困难样本梯度覆盖提高，历史错分患者的直接平均 Margin 仍下降。两组直接净纠错 +4/+8，主路径分别净损失 8/13 人，完整净纠错为 −4/−5。 [PS-SCAR](../experiment82_3_5_ps_scar/results/report.md) · [PS-PGR](../experiment82_3_6_ps_pgr/results/report.md) · [PS-DSO](../experiment82_3_7_ps_dso/results/report.md)

## 共享模型

每折使用 E82.2 的规范定向 U₀、16 维固定文本缓存及从历史 CAM 初始化的合法训练统计。ASD=0 使用文本行 4–7，TD=1 使用行 0–3（零起点）。保留 E82.3.7 的 FP64 Stiefel 参数化。T=(I−U₀U₀ᵀ)W，E=T/√(1+‖T‖²_F/4)，L=chol(I+EᵀE)，U=(U₀+E)L⁻ᵀ。W 初值精确为零，Gram 条件数≤5。谱锚惩罚为 0.05·max(0,1−‖U₀ᵀU‖²_F/16)。历史统计的 CAM144 原坐标 FP64 类均值和二阶矩以 α=.1 更新，Best 在更新前保存，Latest 在更新后保存。只使用合法训练集的原 CAM。两类当前 16 维分布共享同一全体训练 CAM 中心 μ，收缩协方差为 C_c=Uᵀ[0.9Σ_c+0.1Σ_all]U+max(10⁻⁴,0.01·tr(Σ_all)/144)I。对称化后 FP64 Cholesky 失败即停止。

h=(z−μ)U，x_c=h−(μ_c−μ)U。完整方向贡献 d_k=0.5[x_TD,k(C_TD⁻¹x_TD)_k−x_ASD,k(C_ASD⁻¹x_ASD)_k]+[log|C_TD|−log|C_ASD|]/32。独立计算的完整 log p(ASD)−log p(TD) 使用 Cholesky 白化范数，并检验方向贡献和。交叉项均分给两端坐标，行列式项均分给 16 个方向。方向贡献刻画当前模型坐标中的数值归因。输入医学注意力的方向 token 包含平滑有界 d_k、单位方差方向得分及谱统计。类独立注意力和 null 根据历史八条 Query 计算。语义差和统计差分别用合法 warmup CAM 的 75% 幅度分位标定，通过 [0.35,0.65] 的可学习权重融合。方向可靠性使用两类条件与边际方差比的均值及原谱 MLP。零初始化的逐方向有符号 Gate 乘以 [0,1] 可信度，后者读取确定性当前 ASD–TD Gap、分布总 LLR、两类 Null、方向稳定性以及语义与统计一致性。Gap 和分类头敏感度 J 从当前固定读数得到，J 停止梯度。

r_k=gate_k·difference_k·(JU)_k/√(‖JU‖²+10⁻¹²)，δ=SmoothLimit(rUᵀ,z,0.08)。增强 CAM144 是原随机分类头的真实输入。Fold2 两个历史重叠行 377、407 的新增残差严格为零，并从新统计、新监督和带标签机制诊断剔除。固定 Best 的语义干预将语义和 Null 输入归零，统计路径保留。谱干预将 U 换回同折 U₀，当前完整类统计随之重新投影。

## 两组学习准则

G1 的 Bhattacharyya 距离直接读取上述完整协方差模型的 C_ASD、C_TD 及类均值差 Δμ。取 C_m=(C_ASD+C_TD)/2，距离 B=ΔμᵀC_m⁻¹Δμ/8 + [log|C_m|−(log|C_ASD|+log|C_TD|)/2]/2，谱损失为 −B/(1+B) 加共同谱锚正则。收缩和正定下限同时作用于密度和距离。

G2 从原增强预测和同一真实 Dropout 掩码下的未增强预测取真类 Margin m_on、m_off。参考 Margin 停止梯度。定义 φ(m)=0.75 softplus((1−m)/0.75)，困难权重 w=0.25+0.75 sigmoid(1−m_off)，谱损失为在两类内分别求均值后等权的 w[φ(m_on)−φ(m_off)]，加共同谱锚正则。完整前向的 Margin 每增加一单位均降低该项，越接近困难侧梯度越大。共同保护项按类平衡惩罚 sigmoid(m_off−2)·ReLU(m_off−m_on)²。语义辅助读真实进入残差的方向系数及 J·r，另记归一化残差平方。统一权重依次为 .05、.01、.01、.001，总辅助量逐新增组限至真实分类梯度的 25%。原分类梯度经过整个增强路径，原损失及优化器顺序保持历史定义。

## 共享路径保护

一次真实增强前向获取原 Dropout 的线性掩码。用同一个掩码对未增强 CAM 计算原总损失参考梯度 g_B。原历史损失其余项共用同一计算图。真实增强总梯度为 g_E。新增模块接收 g_E 的分类梯度及受限辅助项。共享历史参数接收 g_B 加有界增量，按实际优化器归属分组，CAM 表征与分类头各自审计。所有梯度计算在 step 前完成，额外计算不调用随机层。

保存合法训练患者的确定性原 CAM 真类 Margin 记忆 r。首轮风险梯度关闭以认证零初始化。每次当轮选模后以更新后的原 CAM 评估训练行，按 r 上升 .03、下降 .003 的 EMA 更新，初次观察直接设置 r。记忆的训练风险为类均衡均值 sigmoid((r−2)/0.5)·ReLU(r−0.5−m_det)²，m_det 来自当轮原 CAM 与相同历史头的无 Dropout 路径。g_P 是该风险对共享参数的梯度。在每个共享组内，把 g_E−g_B+0.05g_P 投影到与 g_B 及可用 g_P 内积非负的半空间交集，再将范数限制为 0.01‖g_B‖。风险与参考近零梯度按配置处理，舍入后重新认证，无法认证时该组回退原参考更新。记录梯度、累计增量、参数漂移、优化器动量及合法训练 Margin。该投影是同一步局部梯度约束。Adam 动量、参数轨迹和长期患者预测单独审计。

## 执行与判定

两组均从同折历史 epoch10、原四类 RNG 和原三套优化器独立恢复。新增初始化隔离，AdamW 学习率 .0005、weight decay .01。最多 450 epoch，wait>200，严格测试 ACC 最早 Best，每十轮和终止保存 Latest。检查全部 20 折共享 Dropout 与历史 RNG，在历史 B0 Best 轮到达时留存 residual-off 概率及共享状态身份。使用原历史未舍入指标作配对。Performance 需要完整审计、ΔACC>0、ΔBA≥0、净纠错>0、ΔProb-AUC≥−0.10 pp、ΔSEN≥−0.50 pp。去 Fold2 后和分别去掉 ACC 与患者净纠错最大贡献折后，ΔACC 与净纠错均须为正。Mechanism 另外要求直接残差硬净纠错为正、谱基有效秩 16 且每折投影变化>10⁻⁴、真实梯度、协方差与证据公式、移动谱基、限幅及共享梯度审计通过，全组患者概率受真实残差影响。val=test 选模、全局 CV 前 RFE/scaling 和转导图沿用历史，适用于历史配对。当前实验仅有医学 Query 两组，医学文本的独立诊断效应需独立对照。
