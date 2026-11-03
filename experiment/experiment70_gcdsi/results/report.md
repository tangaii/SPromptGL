# Experiment 70 — GCDSI

状态：`COMPLETE_AUDITED`，30/30新增折独立读回。科研判定：**NO-GO**。单seed探索性判据，不是统计显著性结论。

## 研究问题与方法

GFR传播后的患者图状态是否能与CAM四个模态表示共同构造不同Query，从固定ASD/TD疾病文本中检索患者、模态相关语义，并在最终分类前受控注入？路径为`320维输入 → CADR-GFR → 871×320 → CAM → 871×4×36 → GCDSI → 871×144 → 原OutputLayer`。GCDSI只读取GFR表示，不回写图；CAM输出进入未修改的分类器。

对患者i、模态m：`q_im=LN(W_h h_im+W_f F_i+q_CLS)`；`A_im=softmax(q_im K^T/sqrt(36))`；`u_im=A_im V`；`g_im=sigmoid(MLP([h_im,q_im,u_im]))`；`h'_im=h_im+0.5*tanh(gamma_raw)*g_im⊙W_Ou_im`。`gamma_raw`从0初始化。分类损失经GCDSI回传至CAM与GFR；不detach图/CAM特征。新增模块无Dropout。损失保持`L_cls+L_graph+0.05L_pair+0.01L_adv+0.01L_preserve`，分类损失使用注入后的logits；没有原PEN、额外语义CE/对比损失或新分类头。

SABER公式(8)-(10)提供Graph-CLS、患者文本Key/Value、cross-attention与门控残差的设计动机；SABER另外把疾病文本用于对比监督，本实验没有复现该监督，不能声称完整复现SABER。SPromptGL中GFR是患者关系图传播、CAM是四模态交互、PEN是Prompt驱动局部增强；实验70在CAM后插入GCDSI，不恢复4×4×9局部切分。`OutputLayer.forward(x, attn_embedding)`虽接收attn参数，其分类实现只消费x。来源：[SABER](https://arxiv.org/abs/2607.01901)、[SPromptGL official](https://github.com/wanxixi11/SPromptGL)、[研究仓库](https://github.com/tangaii/SPromptGL)。

## 基座与审计

唯一基座为`CADR_WD_0p01_GFR_WD_0p11`。逐折从53.3第三组初始构造逻辑重建，未加载最佳权重进行微调；参考manifest、10个checkpoint SHA、task/readback、split、pair/cache、共享/新增参数和初始化RNG均通过校验。CADR AdamW为lr=0.001、WD=0.01；GFR/CAM Adam为lr=0.0038、WD=0.11；GCDSI另用AdamW lr=0.001、WD=0.001。每组先共享同折53.3状态的10轮warmup缓存，再从epoch索引10激活GCDSI；保留450 epoch上限、原early stopping、val=test与严格ACC改善选checkpoint。协议为ABIDE A 871名患者、seed=0、`StratifiedKFold(n_splits=10, shuffle=True, random_state=10)`、PHENO/ANAT/FUNC/CORR维度48/6/10/256、ASD=0/TD=1。

参考基座10折：ACC 92.0742±2.8427%；Hard-AUC/BA 91.8653±2.8348%；Probability-AUC 93.6684±3.2619%；SPE-TD 94.6392±3.5168%；SEN-ASD 89.0915±3.5263%；ASD-F1 91.2470±3.0658%。历史10折仅作只读参考。

初始化/warmup：E70逐折split/pair、原参数SHA、初始化RNG和warmup绑定10/10通过。E59独立预检状态`PASS`，epoch-10共享缓存10/10读回一致；另外2步随机CAM前向、参数、优化器和共享RNG parity通过。E70实际入口在fold 1从epoch 10进行2个active epoch，连续训练、相同起点重复训练及从epoch 11断点恢复的模型/优化器/RNG/logits完全一致。详情在`results/audit_status.json`及manifest provenance。

## 文本来源与三组

编码器均为冻结OpenAI CLIP RN50，输入先按同一归一化规则处理，文本向量宽度1024；K/V使用相同共享投影结构，输出宽度36。训练/推理始终同时读两类Token，不按真实标签挑选。文本与embedding SHA见preflight和manifest。

G1 BrainPrompt原始文本，实际排序[ASD,TD]（原文件排列[TC,ASD]，原文不改）：

```text
A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.
A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.
```

G2 SPromptGL `network.py`原始八条文本，保留`Nromal`拼写；各自归一化后前四条均值为TD、后四条均值为ASD，再归一化并排序[ASD,TD]。这是统一CLIP编码和两Token适配，不是原PEN四向量拼接MLP的实现。这些泛化正常/异常描述不等同于ASD特异生物标志物：

```text
Nromal brain volume
Uniform cortical thickness
Symmetrical brain structure
Normal ventricular morphology
Altered brain volume
Abnormal functional connectivity
White matter integrity issues
Different activation patterns
```

G3为固定匿名2向量，独立私有CPU生成器seed=700070；不使用标签、测试性能或患者特征，匹配G1两向量范数及余弦几何。三组参数数目、模块初始化和训练协议相同，唯一正式差异是语义bank内容。

## 十折结果（% mean ± population SD）

| Group | ACC | Hard-AUC | Prob-AUC | BA | SEN-ASD | SPE-TD | ASD-F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E53.3-BASE | 92.074 ± 2.843 | 91.865 ± 2.835 | 93.668 ± 3.262 | 91.865 ± 2.835 | 89.091 ± 3.526 | 94.639 ± 3.517 | 91.247 ± 3.066 | — | — | — | — |
| E70-G1 | 91.385 ± 2.971 | 91.137 ± 2.945 | 93.117 ± 2.810 | 91.137 ± 2.945 | 87.848 ± 3.199 | 94.426 ± 3.645 | 90.442 ± 3.196 | 354 | 26 | 49 | 442 |
| E70-G2 | 91.499 ± 2.878 | 91.316 ± 2.860 | 93.537 ± 2.943 | 91.316 ± 2.860 | 88.848 ± 3.138 | 93.784 ± 3.425 | 90.654 ± 3.057 | 358 | 29 | 45 | 439 |
| E70-G3 | 90.810 ± 3.005 | 90.653 ± 2.910 | 93.417 ± 2.805 | 90.653 ± 2.910 | 88.823 ± 3.930 | 92.484 ± 5.596 | 89.997 ± 3.067 | 358 | 35 | 45 | 433 |

Probability-AUC沿用历史连续TD类概率；Hard-AUC沿用历史硬预测实现。ASD=0、TD=1，SEN=ASD敏感度、SPE=TD特异度。TP/FP/FN/TN为ASD阳性方向的10折逐折合计；历史基座计数不在该参考CSV中提供。所有SD为population SD。

## 配对比较

`mean delta`定义为A−B百分点；W/T/L按同fold方向计数。

| A | B | 指标 | Mean Δ pp | W/T/L |
|---|---|---|---:|---:|
| E70-G1 | E53.3-BASE | acc | -0.690 | 0/6/4 |
| E70-G1 | E53.3-BASE | hard_auc | -0.728 | 2/4/4 |
| E70-G1 | E53.3-BASE | prob_auc | -0.552 | 3/0/7 |
| E70-G1 | E53.3-BASE | sen | -1.244 | 2/4/4 |
| E70-G1 | E53.3-BASE | spe | -0.213 | 1/7/2 |
| E70-G2 | E53.3-BASE | acc | -0.575 | 0/5/5 |
| E70-G2 | E53.3-BASE | hard_auc | -0.550 | 2/3/5 |
| E70-G2 | E53.3-BASE | prob_auc | -0.132 | 4/0/6 |
| E70-G2 | E53.3-BASE | sen | -0.244 | 2/5/3 |
| E70-G2 | E53.3-BASE | spe | -0.856 | 1/5/4 |
| E70-G3 | E53.3-BASE | acc | -1.264 | 0/3/7 |
| E70-G3 | E53.3-BASE | hard_auc | -1.212 | 1/2/7 |
| E70-G3 | E53.3-BASE | prob_auc | -0.252 | 4/0/6 |
| E70-G3 | E53.3-BASE | sen | -0.268 | 4/2/4 |
| E70-G3 | E53.3-BASE | spe | -2.155 | 1/3/6 |
| E70-G1 | E70-G2 | acc | -0.115 | 2/6/2 |
| E70-G1 | E70-G2 | hard_auc | -0.179 | 2/3/5 |
| E70-G1 | E70-G2 | prob_auc | -0.420 | 3/0/7 |
| E70-G1 | E70-G2 | sen | -1.000 | 1/4/5 |
| E70-G1 | E70-G2 | spe | +0.643 | 4/5/1 |
| E70-G1 | E70-G3 | acc | +0.575 | 5/5/0 |
| E70-G1 | E70-G3 | hard_auc | +0.484 | 6/2/2 |
| E70-G1 | E70-G3 | prob_auc | -0.300 | 4/0/6 |
| E70-G1 | E70-G3 | sen | -0.976 | 2/5/3 |
| E70-G1 | E70-G3 | spe | +1.943 | 6/3/1 |
| E70-G2 | E70-G3 | acc | +0.690 | 5/5/0 |
| E70-G2 | E70-G3 | hard_auc | +0.662 | 6/3/1 |
| E70-G2 | E70-G3 | prob_auc | +0.120 | 6/0/4 |
| E70-G2 | E70-G3 | sen | +0.024 | 3/4/3 |
| E70-G2 | E70-G3 | spe | +1.300 | 4/4/2 |

逐折delta数组在`paired_comparisons.csv`。

## 机制证据

| Group | mean gamma | attention entropy | patient attention SD | gate mean | gate saturation | residual/CAM RMS | W_f grad norm | graph-query attention Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E70-G1 | -0.00227 | 0.68542 | 0.04645 | 0.47404 | 0.16804 | 0.456208 | 0.188725 | 0.0180459 |
| E70-G2 | -0.00918 | 0.55985 | 0.12780 | 0.48722 | 0.16638 | 0.379842 | 0.146003 | 0.1304 |
| E70-G3 | -0.00366 | 0.69092 | 0.02073 | 0.47806 | 0.12831 | 0.384804 | 0.109499 | 0.00894938 |

固定checkpoint干预在全871个transductive节点上、不读取标签：关闭gamma后，平均|ΔP(ASD)|分别为G1/G2/G3 0.0532/0.0411/0.0427，平均logit最大偏移分别为 0.910/0.716/0.901，平均预测翻转为 14.4/11.3/14.0个/871。说明GCDSI确实改变CAM最终分类表示，但固定模型干预不等价于重训消融。平均残差/CAM RMS为G1/G2/G3 0.456/0.380/0.385；单折模态范围分别为 E70-G1 0.025–1.289; E70-G2 0.011–1.156; E70-G3 0.009–1.223，部分折残差超过CAM RMS，数值尺度并非所有折都温和。

清零`W_f`后，平均|ΔP(ASD)|分别为 0.0110/0.0072/0.0102，平均预测翻转为 9.5/6.6/9.7个/871；结合非零`W_f`梯度与注意力变化，图表示确实进入Query/决策交互，但这仍是固定模型敏感性而非独立重训消融。

站点/年龄/头动/质量关联为诊断性相关，不用于选择模型。平均attention站点η²为 0.248/0.259/0.256；最大质量变量|r|平均值为 0.354/0.377/0.376；年龄|r|约0.07–0.09，mean-FD |r|约0.16–0.21。逐折站点η²最高约0.57–0.62、质量相关最高约0.74–0.83，提示存在值得关注的站点与质量混杂，不能称为疾病特异证据。G1/G3的平均注意力熵接近ln(2)，区分性较弱；G2更有选择性。G1/G3仍有跨患者、跨模态变化，但此现象不证明其生物学解释。

站点采用site one-hot的η²，年龄用年龄箱中心的|r|，头动用`func_mean_fd`的|r|，影像质量用16个ANAT/FUNC质量变量中最大|r|；这些关联基于无标签输入特征，不能把ANAT/FUNC质量特征解释为病灶。

- E70-G1机制非塌缩门：PASS；条件={"nonzero_residual": true, "nonconstant_patient_and_modality_attention": true, "two_token_attention_not_saturated": true, "gate_not_collapsed": true, "graph_query_gradient_and_sensitivity": true}。
- E70-G2机制非塌缩门：PASS；条件={"nonzero_residual": true, "nonconstant_patient_and_modality_attention": true, "two_token_attention_not_saturated": true, "gate_not_collapsed": true, "graph_query_gradient_and_sensitivity": true}。
- E70-G3机制非塌缩门：PASS；条件={"nonzero_residual": true, "nonconstant_patient_and_modality_attention": true, "two_token_attention_not_saturated": true, "gate_not_collapsed": true, "graph_query_gradient_and_sensitivity": true}。
- 真实疾病文本相对匿名Token：G1−G3 ΔACC +0.575 pp，W/T/L 5/5/0；独立文本价值判据=FAIL。

## 冻结工程判据与结论

- E70-G1 vs 53.3: FAIL；ΔACC -0.690 pp、ACC胜0/10、ΔSEN -1.244 pp、ΔProb-AUC -0.552 pp。
- E70-G2 vs 53.3: FAIL；ΔACC -0.575 pp、ACC胜0/10、ΔSEN -0.244 pp、ΔProb-AUC -0.132 pp。
- E70-G3 vs 53.3: FAIL；ΔACC -1.264 pp、ACC胜0/10、ΔSEN -0.268 pp、ΔProb-AUC -0.252 pp。
- 文本独立价值：FAIL；机制非塌缩：PASS。
- 结论：**NO-GO**。即使GO，也只支持进入多seed、严格验证与预注册组件消融候选阶段，不证明MICCAI级创新或临床有效性。

## 局限、独立读回与来源指纹

所有30个最佳checkpoint以独立进程/脚本重建53.3原始状态后读回，完整比较分类指标与训练时记录，要求30/30一致；训练规则仍使用历史`val=test`选择checkpoint，且RFE/标准化在外层CV前完成，并在871节点transductive图中传播。结果因此不能视为无偏外部泛化估计。单seed与折间相关也不支持显著性推断。

历史base源码及10个checkpoint SHA与只读参考manifest一致；数据、prompt编码缓存、共享warmup缓存和新实验源码哈希记录在`manifest.json`。实验70只在本目录写结果/日志/权重，handoff仅追加记录。文件路径：`fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`mechanism.csv`、`audit_status.json`。
