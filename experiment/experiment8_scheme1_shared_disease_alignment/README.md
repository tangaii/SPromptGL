# 实验八（方案一）：SDPA 共享疾病原型对齐

## 假设与代码审计

固定实验一 P1 的 32 条模态文本，在四模态已经融合的 36 维受试者隐藏表示之后引入共享疾病类别原型。核心假设是：显式保留 ASD/TD 的两个语义分数并进行类别对齐，是否能提供原始 P1 尚未利用的判别信息。只运行 S1–S5；无文本、结构或超参数搜索。

审计时 GitHub main 为 0925d5011397b0feb28d04de761b81f163f99914，包含实验七；本地 HEAD 为 0bf4d34。main.py、model.py、network.py、layers.py、utils.py 与原始 Shell 的 SHA256 均与该远端版本相同。完整源文件、ABIDE 输入和 CLIP RN50 权重 SHA256 见 manifest.json。根目录 network.py 仍载入原始八句通用文本；实验八只在临时副本接入已核验的实验一 P1，再追加疾病分支，根源码不改动。

ABIDE 最终表为 871×321：320 特征加 DX_GROUP。模态维度按顺序为 PHENO 48、ANAT 6、FUNC 10、Correlation 256。预处理 notebook 将原始 DX_GROUP 1/2 写入数据；main.py 减一后为 0=ASD（403例）、1=TD/NC（468例）。概率AUC使用 exp(logprob) 的第0列与 labels==0。PHENO包含站点/人口学信息，ANAT/FUNC主要为质量与运动/信号信息，Correlation是压缩连接特征；不能将四个局部Token解释为具体脑区。

P1路径逐计算步骤保留：冻结 RN50 文本 [32,1024] → [4,2,4096] → 原始 4096→128→9 MLP → 模态特异余弦相似度 [871,4,4,2] → Softmax-Max 标量 [871,4,4] → 加权局部特征及原始 global/local 融合 → OutputLayer。分类头返回 log-softmax [871,2] 和 hidden [871,36]。主干 PLN 为575,121参数，GFR为887,836参数；合计1,462,957。所有新配置在此基础上新增36,865参数。

## 前七轮证据

| 实验 | 主要真实结果与结论 |
|---|---|
| 1 模态专属P1 | P1 ACC 91.80%、Hard-AUC 91.62%；P0约91.69%/91.54%，收益很小，P1保留为基线。 |
| 2 对比状态文本 | V2约91.77%/91.60%，低于P1；仅措辞改变没有稳定增益。 |
| 3 有效性诊断 | P1/D1最高均值，但固定缩放及无关文本对照差异很小；原型分散不能直接预测性能。 |
| 4 ASD相关FC文本 | C1约91.46%/91.29%，C2约91.50%/91.28%，均低于P1。 |
| 5 投影退化 | MLP权重显著收缩；去输出ReLU或去MLP衰减的E1/E2/E3均未超过E0/P1。 |
| 6 保留方向标量门控 | B1约91.73%/91.59%，B2约91.77%/91.52%，未显示整体收益。 |
| 7 双主题特征残差 | F1约91.69%/91.49%，F2/F3/F4也未稳定超过P1；反事实敏感性不能单独证明语义价值。 |

这些结果支持停止普通Prompt改写、旧MLP微调、标量方向门控和双主题残差的追分。本实验检验类别级共享语义，不复制实验七模型。历史结果只作描述性证据；当前静态文本与实际压缩特征可能缺乏充分对应。

## 文献核查与借鉴边界

项目中仅发现SPromptGL论文PDF，没有用户提到的三篇上传PDF；以下核查使用公开原始论文，未声称读取不存在的本地附件。

[BrainPrompt MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf) 的3.3节比较受试者表示与疾病类别文本，使用额外疾病交叉熵。本实验借鉴类别语义监督；使用现有CLIP与P1，不复现其Llama、多层Prompt和人群图全套方法。

[SABER ICME 2026](https://arxiv.org/html/2607.01901v1) 的Module III包含图级语义对齐、受控语义注入与疾病监督。本实验借鉴让疾病分数参与决策及限制贡献的思想；其Graph-CLS、患者文本交叉注意力、多尺度超图和对比损失均未加入。

[BrainSCK MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1779_paper.pdf) 将知识注入与疾病知识激活区分为两个阶段，使用HCP人口学和心理测量记录。本项目没有相应个体认知记录，因此没有伪造认知文本。本次不加入ROI知识图谱、个体化文本、Sinkhorn或额外LLM推理；其他候选论文仅作为用户提供的后续方向，未据未核验细节实现组件。

文本审慎性依据还包括[NIMH的ASD核心表型说明](https://www.nimh.nih.gov/health/publications/autism-spectrum-disorder)、[网络特异的混合FC方向研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC5027998/)和[ASD脑网络个体异质性研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9793594/)。这些来源支持群体层面与异质性的措辞，不把静态语句解释为个体生物标志物。所有规定文本保持逐字不变，未根据结果修订。

## 固定模型与对照

| 配置 | 类别0/类别1文本 | 对齐损失权重 | 疾病决策融合 |
|---|---|---:|---|
| S0 | 原始P1，冻结历史30折 | 无 | 无 |
| S1 ASD-Real-Joint | ASD / TD | 0.1 | 有 |
| S2 ADHD-Mismatch-Joint | ADHD / TD | 0.1 | 有 |
| S3 Neutral-Text-Joint | 两组MRI技术文本 | 0.1 | 有 |
| S4 ASD-Align-Only | ASD / TD | 0.1 | 无 |
| S5 ASD-Fusion-Only | ASD / TD | 0 | 有 |

S1–S5仅存在表中的差异。S2保留TD，训练仍是ASD/TD任务；S3仍使用文本及类别监督，不能称为无文本模型。每组四句文本由冻结RN50编码、逐句L2归一化、组内平均并再次归一化。ASD/TD、ADHD/TD、中性对照的初始原型cos分别约0.946976、0.932048、0.968670；单位范数，类别距离约0.325650、0.368651、0.250318。几何差异在机制分析中保留。

计算为：

    z = normalize(disease_projector(subject_hidden))
    disease_logits = z @ disease_prototypes.T / 0.2
    L_align = weighted_CE(disease_logits[train_idx], labels[train_idx])
    gamma = 0.2 * tanh(disease_alpha)
    final_logprob = log_softmax(base_logprob + gamma * disease_logits)
    L_total = original_ClsLoss(final_logprob) + original_graph_loss + lambda * L_align

两个疾病logits完整保留，新增路径不做Max。S4直接返回base_logprob；其alpha注册但不参与决策，零梯度为预期消融。alpha可为负，诊断记录符号，不把负向贡献解释成正确类别语义利用。

原始初始化和ModalFusion.apply(my_weight_init)完成后才注册Linear(36,1024,bias=False)及零alpha；投影采用原有my_weight_init的Xavier规则。新增参数36,864+1=36,865，每个配置完全相同，且优化器无遗漏/重复。原参数Adam lr=.0038、weight_decay=.11，新增参数同lr、weight_decay=0。零alpha初始输出与P1最大绝对差为1.19e-7。最小预检覆盖五配置原有与新增参数初始化、输入/输出形状、参数数量、原型类别对应、归一化、两步真实优化、完整诊断字段和REAL反事实一致性。

## 协议、诊断和运行

沿用原始Shell：weighted-cosine、concat、GCN、dropout .35、lr .0038、simple-2、2 heads、hidden18、1 layer、reg .11；其他默认参数不变。StratifiedKFold(10,random_state=10,shuffle=True)，seed0/1/2，450 epoch上限，patience200。按原程序ACC最优checkpoint选择、类别权重、图损失、transductive传播及Hard-label AUC统计。原程序val_idx=tst_idx，因此数字只作历史一致协议比较，不能表述为独立测试泛化估计。S0缺少逐折概率，Probability-AUC标为N/A。BA直接计算(SEN-ASD+SPE-ASD)/2。

15个进程各自连续十折；GPU0八任务、GPU1七任务同时运行。前一轮因发现两项输出字段遗漏而中断，修正后从各任务第一折完整重跑；旧attempt保留在日志，指标不混入汇总。每任务使用独立文件锁与attempt文件；调度器互斥，完整任务跳过，失败任务完整重跑；临时源码与最佳checkpoint放本机快速磁盘。成功后保留150条折指标、150条机制诊断与150条S1反事实；自动清理临时worker与attempt文件，不保存受试者级预测。

S1反事实共REAL、ADHD、NEUTRAL、SWAPPED、OFF；同一最佳checkpoint和同一输入，仅干预疾病分支。输出ACC、Hard/Probability-AUC、BA、SEN/SPE、预测翻转率、ASD概率变化、融合前logit变化及疾病margin变化。预测敏感性与正确疾病文本收益必须结合S2/S3完整训练对照解释。

    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --prepare
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --all
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --resume
    python experiment/experiment8_scheme1_shared_disease_alignment/run.py --refresh

已完成运行不要再次prepare；resume跳过成功任务，refresh从成功attempt或最终CSV重建报告。所有文本独立存于prompts.json，训练不依赖历史实验run.py。最终产物与主结果见results/report.md，版本、哈希、初始化检查与完成状态见manifest.json。本实验不执行git commit或git push。
