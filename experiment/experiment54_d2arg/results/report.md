# Experiment 54 — D²-ARG（Disease-Discriminative Adversarial Relational Graph Learning）

## 状态与审计结论

**正式科研判定：NO-GO。工程执行与可复现性审计：PASS。** 10/10 组、100/100 group-fold 已完成；100/100 最佳 checkpoint 独立读回一致。B0 的 E1 identity replay 与 Experiment 51 E1 基座均值完全一致，新增模块没有稳定超过 E1。

preflight：PASS；fold 1/5/10 的 E1 前向最大差为 0，首个 optimizer step 最大参数差为 0.0e+00，共同 RNG 相等=True；GRL、标签隔离、有限性、非负邻接和非零传播均通过。checkpoint audit：100/100，最大指标读回差 0.0e+00。

## 研究问题、依据与边界

D²-ARG 只检验一个模块：以 CORR 256 维特征学习对称疾病关系分数，用患者对站点关系条件对抗约束关系表示，直接校准 E1 的四张 GraphLearn 邻接，并在 CORR 子空间加入有界正/负关系消息。没有加入 Prompt、PEN、Shared/Private 解耦、PID 或第二创新模块。

工程基座是 Experiment 51 的 `LegacyFixedGFR + PLNNoPEN`：四路完整 320 维 GCN 消息累加和 PHENO 覆盖修复，`weight_decay=0.11`、lr=0.0038、dropout=0.35、450 epoch、`wait>200`、ACC 严格保存、`val=test`、CV 前预处理和 871 节点 transductive 图全部保留。

方法依据仅限于公开 SPromptGL 的多关系图/GFR 与 GraphConstructLoss/GCN 路径，以及 DeepASD 的“表示学习器与判别器竞争”思想；D²-ARG 的患者对关系、站点条件和关系感知传播是本实验的独立实现。SPromptGL：<https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf>、<https://github.com/wanxixi11/SPromptGL>；DeepASD：<https://doi.org/10.1038/s41398-024-02972-2>。

## E1 初始化、RNG 与标签隔离

每折先按 `seed_all(510000 + fold*100 + 1)` 创建四路 GraphLearn、四路完整 320→320 GCN、fusion_weight 和 PEN-free CAM，再加载匹配的 Experiment 43.1 anchor；新增 DRE/CRA/边校准/传播参数在 `torch.random.fork_rng` 中独立初始化。B0 完整 10 折重放得到与参考 E1 相同的均值（91.9592% ACC、91.7434% Hard-AUC、93.1856% Probability-AUC）。测试标签置换只改变评估口径，前向、损失与概率最大差分别为 2.38e-07、1.19e-06。

关系采样只使用 outer-train 患者对，按同/异诊断与同/跨站点四个 strata 平衡；候选支持由四张 anchor GraphLearn 图的无标签 union 构成。内部 5 折患者级 OOF 只用于关系 AUC、跨站点 AUC 和校准诊断。

## 十组正式设计

| 组 | 定义 |
|---|---|
| B0 E1_IDENTITY_REPLAY | edge=False，propagation=none，adv=False，pair_aux=False； |
| B1 PAIR_AUX_ONLY | edge=False，propagation=none，adv=False，pair_aux=True； |
| B2 SIMILARITY_SIGNED | edge=True，propagation=signed，adv=False，pair_aux=False；普通 CORR 相似度 |
| B3 DISEASE_EDGE_ONLY | edge=True，propagation=none，adv=False，pair_aux=True； |
| B4 DISEASE_EDGE_SIGNED | edge=True，propagation=signed，adv=False，pair_aux=True； |
| B5 FULL_D2ARG | edge=True，propagation=signed，adv=True，pair_aux=True； |
| B6 NO_NEGATIVE_CHANNEL | edge=True，propagation=positive，adv=True，pair_aux=True； |
| B7 WRONG_DISEASE_PAIR | edge=True，propagation=signed，adv=True，pair_aux=True；标签置乱监督 |
| B8 WRONG_PATIENT_GRAPH | edge=True，propagation=signed，adv=True，pair_aux=True；错误患者关系图 |
| B9 NO_CROSS_PATIENT_MESSAGE | edge=True，propagation=none，adv=True，pair_aux=True； |

## 正式分类结果（十折 mean±population SD，百分比）

| 组 | ACC | Hard-AUC/BA | Probability-AUC | SPE-TD | SEN-ASD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 E1_IDENTITY_REPLAY | 91.9592±2.8684 | 91.7434±2.8950 | 93.1856±3.0236 | 94.6392±3.3856 | 88.8476±4.1651 | 91.0928±3.1408 |
| B1 PAIR_AUX_ONLY | 91.9592±2.8684 | 91.7434±2.8950 | 93.1856±3.0236 | 94.6392±3.3856 | 88.8476±4.1651 | 91.0928±3.1408 |
| B2 SIMILARITY_SIGNED | 91.9592±2.9141 | 91.8173±2.8706 | 93.0315±2.9672 | 93.5615±4.6588 | 90.0732±3.7017 | 91.2364±3.0617 |
| B3 DISEASE_EDGE_ONLY | 91.4995±3.0565 | 91.3018±3.0250 | 93.4884±2.9867 | 93.7743±4.6921 | 88.8293±4.0771 | 90.6567±3.2562 |
| B4 DISEASE_EDGE_SIGNED | 91.0410±2.8137 | 90.7939±2.8443 | 93.4291±2.9958 | 93.9963±3.5967 | 87.5915±4.3254 | 90.0403±3.1290 |
| B5 FULL_D2ARG | 91.7294±3.0388 | 91.5306±3.0655 | 93.0581±3.3684 | 94.2137±3.3535 | 88.8476±4.1578 | 90.8609±3.3442 |
| B6 NO_NEGATIVE_CHANNEL | 91.4995±2.9688 | 91.2885±2.9562 | 93.2854±2.7464 | 93.9917±4.0988 | 88.5854±4.0283 | 90.6217±3.1734 |
| B7 WRONG_DISEASE_PAIR | 91.6144±3.0062 | 91.4584±3.0261 | 93.1903±3.1243 | 93.5754±3.5997 | 89.3415±4.2815 | 90.7926±3.2916 |
| B8 WRONG_PATIENT_GRAPH | 91.7294±3.1245 | 91.5408±3.1597 | 93.4359±3.3320 | 93.9963±3.4742 | 89.0854±4.3316 | 90.8800±3.4119 |
| B9 NO_CROSS_PATIENT_MESSAGE | 91.9592±2.7267 | 91.8120±2.6438 | 93.7006±2.8680 | 93.5569±4.9537 | 90.0671±3.3693 | 91.2497±2.7849 |

完整逐折指标在 `results/fold_metrics.csv`，汇总在 `results/summary.csv`。ASD=0，SEN-ASD 使用 TP/(TP+FN)，SPE-TD 使用 TN/(TN+FP)，Hard-AUC 使用硬预测 one-hot，Probability-AUC 使用 TD 概率。

## 关键同折配对比较

| 比较 | 指标 | Δ百分点 | 胜/平/负 |
|---|---|---:|---:|
| B5_minus_B0 | acc | -0.2299 | 0/8/2 |
| B5_minus_B0 | hard_auc | -0.2128 | 2/5/3 |
| B5_minus_B0 | prob_auc | -0.1275 | 6/0/4 |
| B5_minus_B0 | spe | -0.4255 | 1/6/3 |
| B5_minus_B0 | sen | 0.0000 | 2/6/2 |
| B5_minus_B4 | acc | 0.6883 | 6/3/1 |
| B5_minus_B4 | hard_auc | 0.7367 | 6/2/2 |
| B5_minus_B4 | prob_auc | -0.3710 | 5/0/5 |
| B5_minus_B4 | spe | 0.2174 | 3/5/2 |
| B5_minus_B4 | sen | 1.2561 | 5/3/2 |
| B5_minus_B7 | acc | 0.1149 | 1/9/0 |
| B5_minus_B7 | hard_auc | 0.0722 | 4/3/3 |
| B5_minus_B7 | prob_auc | -0.1322 | 5/0/5 |
| B5_minus_B7 | spe | 0.6383 | 4/3/3 |
| B5_minus_B7 | sen | -0.4939 | 3/4/3 |
| B5_minus_B8 | acc | 0.0000 | 1/8/1 |
| B5_minus_B8 | hard_auc | -0.0102 | 2/6/2 |
| B5_minus_B8 | prob_auc | -0.3778 | 4/1/5 |
| B5_minus_B8 | spe | 0.2174 | 2/6/2 |
| B5_minus_B8 | sen | -0.2378 | 1/7/2 |
| B5_minus_B9 | acc | -0.2299 | 1/6/3 |
| B5_minus_B9 | hard_auc | -0.2814 | 2/4/4 |
| B5_minus_B9 | prob_auc | -0.6425 | 4/0/6 |
| B5_minus_B9 | spe | 0.6568 | 2/6/2 |
| B5_minus_B9 | sen | -1.2195 | 1/6/3 |

B5 相对 B0 的 ACC 为 −0.2299 pp、Hard-AUC 为 −0.2128 pp，ACC 胜/平/负为 0/8/2；没有达到预注册的 +0.3 pp 筛选目标。B5 相对 B4 的硬分类指标上升，但 B4 是较弱的无对抗 signed 传播变体，增益没有转化为超过 B0 的独立诊断增量。B5 相对 B7 仅 ACC +0.1149 pp 且 9/10 折平局；相对 B8 ACC 0.0000 pp、Probability-AUC 下降 0.3778 pp；相对 B9 ACC −0.2299 pp、Probability-AUC −0.6425 pp。

## 疾病关系、对抗与图消息诊断

正确关系 10 折 OOF AUC=0.6994±0.0249，跨站点 AUC=0.6890±0.0280，同站点 AUC=0.7098±0.0237，Brier=0.2186±0.0076；10/10 折满足预设可靠性门槛。标签置乱 B7 的关系 OOF AUC=0.5060，跨站点 AUC=0.5059，接近机会水平。

B5 固定 checkpoint 的站点判别器 AUC=0.5193±0.0396，site loss=0.693166；接近机会水平说明站点关系难以由关系表示恢复，但不单独证明疾病泛化或对抗成功。

B5 最佳 checkpoint 的平均邻接绝对变化=0.001282，正消息范数=44.7856，负消息范数=43.1269，CORR 有界校正消息范数=1.7655，相对 E1 表示范数比例=0.7471%。关闭传播后 ACC 从 91.7294% 变为 91.3845%；固定模型关闭边重加权没有产生硬预测变化，说明边变化进入了归一化 GCN 但量级不足以形成分类决策增量。

固定 B5 干预的指标和翻转数见 `results/interventions.csv`、`results/intervention_flips.csv`；关系/对抗 readback 见 `results/adversary_audit.csv`。正负交换产生 6 个测试预测翻转，但平均硬分类仍低于原始 B5；错误图干预没有产生硬翻转。

## 独立 checkpoint、随机性和竞争解释

独立审计重新加载 100 个 best checkpoint，逐折重算 ACC、Hard-AUC、Probability-AUC、SEN、SPE、BA、ASD-F1 及 TP/FN/FP/TN，100/100 通过且保存值与重算值最大差为 0。B1 完全不把关系辅助训练写入 CAM，结果与 B0 相同，排除了单纯关系参数容量收益。B2 的普通 CORR 相似度与 B5 没有形成疾病监督的独立优势。

主要竞争解释是：关系分数虽有 OOF 疾病信号，但在 `weight_decay=0.11` 的原始 GFR 中，边校准对归一化消息的影响很小；有界 signed DRMP 的非零消息进入 CAM，却没有提供超过 E1 的诊断信息。B8 正确/错误关系图 ACC 相同，不能把 pair BCE 或站点 AUC 当成患者关系身份的医学有效性。

## 局限与最终判断

本轮保留历史 `val=test`、CV 前标准化/RFE、transductive 871 节点图和单 seed=0，因此不能宣称无偏外部泛化或跨站点临床性能；十个折也不是十个独立外部数据集。站点关系只由 PHENO 的 `SITE_ID_*` one-hot 精确恢复，D²-ARG 不使用个体临床量表或文本语义。

**最终：工程 GO，科研 NO-GO。** D²-ARG 的疾病关系评分可被交叉拟合验证，但正确关系没有稳定超过 E1、错误患者图 B8，新增边校准和关系消息没有形成独立诊断增量；不建议以本轮结果作为 MICCAI 2027 第一创新模块的正向证据。若继续，应先改用独立验证/外部站点和真实可验证关系证据，避免在同一患者相似图上继续堆叠对抗或消息分支。

## 交付文件

- `config.json`、`run.py`、`d2arg.py`、`audit.py`、`mechanism.py`：冻结配置、实现、调度、独立读回和干预。
- `source/e1_base.py`、`source/drap.py`、`source/layers.py`、`source/network.py`、`source/utils.py`：Experiment 51 E1 必要源码快照。
- `results/fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`checkpoint_audit.csv`、`mechanism.csv`、`interventions.csv`、`intervention_flips.csv`、`adversary_audit.csv`。
- `.runtime/preflight.json`、`.runtime/checkpoint_audit.json`、100 个 task JSON、best checkpoint、关系缓存及其 SHA。

> 交接同步备注：已幂等更新 `/model/spromptgl/experiment/handoff.md`。尝试更新 `/data/adodas/SPromptGL-main/handoff.md` 时底层 OSSFS 返回 `Errno 5 Input/output error`，该文件保持未修改；失败已记录在 `manifest.json`。

## 工程修复记录

正式调度早期发现两处仅影响运行的实现问题：B5 新增有界参数初始设备未统一，B2 普通相似度路径的诊断字典未初始化。两者均在继续正式矩阵前修复；受影响的 B5 fold1 和 B2 folds1–10 全部从冻结配置重新运行，最终 100/100 task COMPLETE，独立读回 100/100 PASS。没有用测试 ACC 改变模型、损失或超参数。
