# Experiment 82.3.6 · PS-PGR 设计冻结

原论文第 2.1–2.2 节与式 (7) 定义按模态建立患者关系图、图学习与分类联合目标、CAM 以及以冻结文本编码器增强局部特征的 PEN。第 3.1 节的 ABIDE 协议包含 871 人、四模态及十折。历史 E74.4-G2 的 E74.1 → E69.3 → E61 epoch 10 恢复链实际使用 PLNNoPEN。原分类头顺序为 LayerNorm、Linear、GELU、Dropout、Linear、log_softmax。原随机路径继续执行全部运算。新增路径单独用原头的确定性等价式。 [论文](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) · [原作者源码](https://github.com/wanxixi11/SPromptGL/tree/ba3b9446fa451fde6864d6555bdf324de0c4b64a)

E82.3.2 建立了逐折合法 Gaussian 统计及 16 维固定谱空间的使用。E82.3.3 的标量门控和 E82.3.4 的方向门控未产生固定 Best 硬纠错。E82.3.5 的判别方向残差直接净纠正 G1 一人和 G2 三人，端到端模型仍比 DARG 分别净少五人和十二人。其 `auxiliary_losses` 的语义 NLL 读取 `evidence`，而推理残差读取 `difference`。指定 GitHub 提交的 `run.py` 导入 `history.py`，提交内该文件返回 404，服务器副本存在。PS-PGR 保留独立的本地 `history.py`，并锁定历史代码、数据、checkpoint、缓存和数值依赖身份。 [历史提交](https://github.com/tangaii/SPromptGL/commit/7216f3d72db3c16587a2f4b22d1606cd64810584)

## 特征和目标

保留 E82.3.5 的 16 维 U、Prompt 缓存、ASD/TD 注意力、Null、LLR 和合法训练样本拟合的固定尺度。LLR 正号支持 ASD，ASD 标签为 0。采用相同有界 signed fusion、非负方向 gate 和零初始化。J 为原头无 Dropout 的 ASD−TD Gap 对 z 的雅可比，停止梯度。相对 z 的平滑残差上限是 8%，然后通过原分类头评估 z+δ。Fold2 行 377、407 的新增残差为零。新增目标和 EMA 只消费外层合法训练行。

原二分类交叉熵可写为 softplus(−真类 Margin)。以当前头的实际 off/on 为两项，类别等权，权重为 0.25+0.75·sigmoid(1−m_off)。L_inc 为带权的 CE(on)−stopgrad(CE(off))。L_protect 按 sigmoid(m_off−2) 给真类 Margin 的下降平方加权，系数 1。沿用轻量的相对残差平方保护，系数 0.001，分母和连续权重均由 detached z/off 决定。记录有符号 D 的分类一致方向及语义诊断，独立语义 NLL 权重为零。二分类 CE 差在零残差时逐元素为零，on 端仍向门控给出有效梯度。

G1 用不修改输入的 Hook 捕获原随机训练前向的 z。旁路只接收 detached z 和 detached 头参数，求两个梯度后检查实际梯度所有权，再按历史顺序更新。G2 的 CAM 特征输入 δ(z) 保持可微，on 头的直接加法基点使用 stopgrad(z)，前向值就是 z+δ。若直接对 z+δ 的恒等项求导，零残差仍会向上游复制一条分类梯度。以残差路径求共享增量梯度能在零初始化时恢复严格 DARG 首步。原分类头只收原始 DARG 梯度。共享增量按历史 optimizer 参数组在 FP64 去除冲突分量，限制为基准范数的 5%，并检查在实际参数梯度 dtype 中施加的差分。基准为零、历史未参与的张量或平方范数≤1e−24 时无注入，非有限数拒绝。投影的分母安全项残留以正交校正消除。原损失和三优化器时序保持历史行为，模块使用独立 AdamW。

## 轨迹、选择与门槛

G1 以原 E74.4 runner 无新增模块对每折执行只读确定性重放，逐轮比较随机头 logits、损失、全历史梯度、模型及三个优化器哈希、residual-off 概率及共享 RNG。另核对历史 Best 和 Latest 的实际权重及优化器、历史 Best 概率、已有归档日志损失和 RNG。重放只用于认证，不产生新的 B0 指标组或历史文件。超出历史停点的部分标注为同一确定性更新方程的延伸。G1 按本组严格 ACC 提高选取 Best，平局使用早期 epoch。在达到 B0 Best epoch 时，用该轮评估时的新增模块和 EMA 做对齐诊断。若 G1 因原规则先停，在 B0 Best 上以停止时的模块做只读诊断，明确标注两者 epoch。主统计使用各组自己按历史规则选出的 Best。G1 的 A 阶段差异可来自 checkpoint epoch 选择，逐轮的共享参数漂移应为零。

G2 的执行要求 G1 十折审计通过、对 B0 的全模型净纠错为正、固定 Best 直接净纠错为正，且历史错分的真正新增纠错多于残差对 B0 正确患者造成的新误伤。两医学组同时考察绝对性能门槛和机制门槛。为落实不依赖 Fold2 或单折，九折去 Fold2 和从汇总中扣掉各项贡献最大的单折后，要求 ΔACC 仍正、ΔBA 仍非负、总净纠错仍正。只作统计敏感性检验。若有医学组双 GO，先按十折 ACC，再按直接净纠错，最后选 G1 平局优先，运行唯一匿名十折。其余执行规则见配置。

## 独立运行

以历史工作区 /data/adodas/SPromptGL-main 和只读缓存 /model/spromptgl/experiment 为输入，运行本目录 `python -B run.py preflight`、`python -B run.py train --group G1`，再用 `python -B audit.py --stage G1` 复核和发放条件组授权。G2 及 ANON 使用相同入口和各自组名。依赖检查在 Preflight 验证清单中的历史字节，训练和审计拒绝与 Preflight 身份不符的实现。数值包版本固定在 config.json，运行优先使用本目录 runtime/python，也支持安装同一版本的环境。历史工作区包含私有 ABIDE 数据、E61 与 E74.4 权重和逐折谱、统计缓存，这些输入通过 SHA-256 校验并以只读引用使用。

本实验继承 val=test 选模、CV 前 RFE/scaling、转导患者图、Fold2 历史身份和单 seed。结果用于历史配对比较。
