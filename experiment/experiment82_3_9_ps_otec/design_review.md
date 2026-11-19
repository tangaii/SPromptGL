# Experiment 82.3.9 · PS-OTEC 冻结设计

## 基座与失败归因

原论文的 GFR 学习患者关系图并传播特征，PLN/CAM 与局部及全局 PEN 读取多模态证据。论文采用 ABIDE 871 人、十折及联合分类与图学习。当前基座是 E74.4-G2 Positive-DCR 的 PLNNoPEN。沿用 CAM 的历史局部及全局处理。实际 CAM 输出 871×144，依序进入 LayerNorm、Linear、GELU、Dropout（0.35）、Linear、log_softmax。恢复链 E74.4 → E74.1 → E69.3 → E61 epoch 10，包括 E74.1 的固定重参数化、每折 4096 个历史 Pair 和完整 RNG。原损失为加权分类 NLL + 图损失 + 0.05·Pair + 0.01·条件对抗 + 0.01·关系保留。原优化顺序为 Adam 主干、Adam CAM 和分类头、AdamW 关系参数。前两套学习率 0.0038、weight decay 0.11，第三套为 0.001、0.01。 [MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a) · [SABER 源码](https://github.com/tangaii/SPromptGL/tree/c7e2edbb8c56bdfc416b708da0ec6d743b0ed7d4)

E82.3.6 的严格保护路径由独立确定性旁路训练新增模块，直接硬纠错为零。E82.3.7 的真实增强分类和共享掩码仍发生主路径退化。E82.3.8 以当轮 student 的 residual-off 梯度约束局部梯度差，风险保护由 student 自身不对称 EMA 更新。其 G1/G2 的 A 阶段净损失为 8/7 人，B 阶段净纠错为 5/3 人，C 阶段净损失为 3/4 人。历史错分者直接真类 Margin 平均下降约 0.049/0.053，分别仅有 1/0 人被残差真正新增纠正。在线 DARG、student residual-off、优化器动量差及 Best 轮差本轮分别记录。 [E82.3.6](../experiment82_3_6_ps_pgr/results/report.md) · [E82.3.7](../experiment82_3_7_ps_dso/results/report.md) · [E82.3.8](../experiment82_3_8_ps_cdso/results/report.md)

## 真实在线 DARG 与参数步长保护

同一折的进程内维护独立 Reference、G1、G2。三条流分别继承初始 Python、NumPy、CPU 和 CUDA RNG，保存并取回各自连续状态。一次原生参考步供两组读取。参考使用原生总损失和三优化器，逐轮核对历史 RNG，并在档案 Best 和终止轮验证参数、优化器、最佳概率和可用梯度标记。若 student 超过历史终止轮，Reference 继续原生更新以保持轮次对齐，该区间标为未存档延续。认证更晚的历史终止点时允许 Reference 延续至历史终止，正式 B0 始终使用既有存档。每轮合法训练教师只提供同掩码分类输出及该步更新前的确定性预测。测试预测用于同轮归因，独立保存。

设当前 student 的同掩码未增强更新参数为 x_off，真实增强更新为 x_full。使用原优化器状态的副本和 PyTorch 原生 Adam 函数计算两项。真实优化器按历史顺序执行增强梯度与保留梯度的合成提议步骤，合成的保留梯度逐历史参数组限为增强总梯度的 10%。在此之后仅约束参数位置，真实 Adam/AdamW 的步计数、一阶及二阶动量完整保留。CAM 表征和分类头分开约束。以 x_off 为中心，半径 b=0.15‖x_off−x_t‖。对实际提议及同步参考 x_ref 的相对位置各施加 L2 球截断 P_b。实际位置 x_off+(1−a)P_b(x_proposal−x_off)+aP_b(x_ref−x_off)，其中 a=0.8·d/(d+2(‖Δx_ref‖+‖x_off−x_t‖)+ε)，d=‖x_ref−x_off‖。凸组合保留真实增强成分。距离变大时参考权重上升。FP64 计算、FP32 落位后再次认证约束，舍入预留 0.01% 半径。按组记录 raw full、加保留提议、已施步、参考距离和动量差。预算约束实际参数移动，其数值单位与 E82.3.8 的 1% 梯度预算不同。使用独立的真实优化器副本预检虚拟步。

Reference 的合法训练真类确定性 Margin 为 m_r。保留可靠度 w=1[m_r>0]·sigmoid((m_r−2)/0.5)。loss 为同掩码概率差平方的 0.25 倍加 ReLU(m_ref_mask−0.5−m_student_mask)²。类别内均值后 ASD/TD 等权。对 student full 和 residual-off 分别计算并等权合并。Reference 错分病例的保留权重为零。新分支评估前与真实分类共享一次实际 Dropout 掩码。分类头 J 和路由确定性 Gap 均不采样随机数。参考测试预测、历史 B0 错分病例名单及测试标签均未进入新增损失。

## 共享证据与两种候选

E82.2 历史 U₀ 和八条冻结文本缓存按 SHA 核对，ASD=0 对应文本行 4–7，TD=1 对应 0–3（零起点）。T=(I−U₀U₀ᵀ)W，E=T/√(1+‖T‖²_F/4)，U=(U₀+E)chol(I+EᵀE)⁻ᵀ。谱方向 token 直接取当前 U 的有向基列及当前谱统计，训练和推理共享坐标规则。原 CAM 的合法训练类均值和 FP64 二阶矩以 α=.1 更新。以全体训练 CAM 均值为中心，C_c=Uᵀ(0.9Σ_c+0.1Σ_all)U+max(10⁻⁴,0.01·tr(Σ_all)/144)I。完整高斯方向贡献 0.5[x_TD,k(C_TD⁻¹x_TD)_k−x_ASD,k(C_ASD⁻¹x_ASD)_k]+(log|C_TD|−log|C_ASD|)/32，方向求和与全量 Cholesky LLR 一致。语义与统计差用训练 warmup 数据标定，融合权重区间 [0.35,0.65]。分类头确定性 Jacobian 停止梯度，投影到当前 U。新增 CAM144 残差逐人平滑限幅 8%。Fold2 第 377、407 行在新增残差及两候选中归零，在所有新增有标签损失、教师、EMA 和带标签诊断中剔除。

TR-SPEC 使用可信度调节的逐方向有符号 Gate，零初始化，沿归一化 JU 与 signed difference 生成残差。以同掩码类别均衡真类 Margin 增益和有界 Bhattacharyya 共同学习 U。真实分类损失仍为原生优化主目标，原图和关系损失参与更新。增强及未增强主路径共享可靠预测保留。

CE-SPEC 使用同一 U、按疾病方向分别预测两个非负幅值，沿 ±JU 构造 ASD 与 TD 两个有界候选。候选可靠度联合医学响应、LLR、Null、分类置信度、两类方向稳定性和证据冲突。路由学习保持、ASD、TD 的连续混合，读取两候选在原分类头无随机掩码时产生的可复算 Gap 差。该函数训练和推理一致。监督使用真实主头捕获的同一个 Dropout 掩码求候选 Margin，依照合法训练真实类别监督正向候选并惩罚反向候选改善真类 Margin。软目标 q=1−exp(−ReLU(Δm_true−0.03)/0.25)，保持概率 1−q，真类候选 q，反类候选 0。候选幅度头与路由末层零初始化使候选初始严格反向，ASD 与 TD 初始路由质量完全相同，最终增量为零。

新增参数真实分类梯度加受限辅助梯度。辅助权重为谱 .02、增益 .05、语义 .01、残差稳定性 .001、G2 候选 .03、路由 .02。逐参数组总辅助范数最高为真实分类梯度 25%，第一步独立比对。两组的误差权重依据合法训练参考的正确性和连续真类 Margin，难例权重最低 .25。保留新 AdamW 的 .0005 学习率和 .01 weight decay。所有可选系数在预检与正式训练前固定。冻结判定使用未舍入历史配对值，ΔACC>0、ΔBA≥0、净纠错>0、ΔProb-AUC≥−0.10 pp、ΔSEN≥−0.50 pp，去 Fold2 后 ΔACC 与净纠错均为正。机制另检查预注册审计、真实可训练性及 LLR 和谱基独立一致性。历史 val=test、CV 前 RFE/scaling、转导图及单 seed 保留，支持同协议配对。医学文本独立效果需要相应重训对照，本轮固定 Best 干预只作即时证据。
