# Experiment 70.1 — DCC-GCDSI

状态：`COMPLETE_AUDITED`，30/30正式折完成，30/30最佳checkpoint独立读回通过；科研结论：**NO-GO**。单seed工程判据不是统计显著性。

## 研究动机与修正

Experiment 70 GCDSI因BrainPrompt两类CLIP文本近同向（cos≈0.9848）、G1普通注意力熵接近ln(2)、共享语义Value直接混入CAM、部分模态残差超过CAM RMS而判为NO-GO。70.1把文本共有方向与ASD−TD差异方向分开：差异方向形成相反类别Key和有符号竞争系数；共有方向只调制患者CAM通道；逐患者逐模态将残差RMS严格限制在CAM RMS的15%以内。模块不再注册进CAM，独立管理优化器和checkpoint。

SPromptGL中的GFR表示患者关系传播、CAM表示四模态交互、PEN是Prompt驱动的局部特征增强；本实验沿用无PEN的53.3第三组，并在CAM后、原OutputLayer前插入DCC-GCDSI。SABER公式(8)-(10)提供图条件Query、语义读取和门控残差的思路，疾病级对比监督是额外路径；70.1不添加该损失，也不声称完整复现SABER。

## 方法与数学审计

输入路径：`X[871,320] -> CADR-GFR -> F[871,320] -> CAM -> H[871,4,36] -> DCC-GCDSI -> H'[871,4,36] -> flatten[871,144] -> 原OutputLayer`。CAM每轮只调用一次。

```text
q_im = LayerNorm(W_h h_im + W_f f_i + q_CLS)
c = normalize((t_ASD+t_TD)/2); d=(t_ASD-t_TD)/(||t_ASD-t_TD||+eps)
k_delta = normalize(W_d d); z_im=q_im^T k_delta
A_im=softmax([z_im,-z_im]); s_im=A_ASD-A_TD=tanh(z_im)
v_center=normalize(W_c c); u_im=tanh(W_o(h_im * v_center))
g_im=sigmoid(MLP([h_im,q_im,v_center])); r_im=g_im * u_im
lambda=0.15*tanh(lambda_raw), lambda_raw(0)=0
delta_im=lambda*s_im*RMS(h_im)*r_im/(RMS(r_im)+eps)
h'_im=h_im+delta_im
```

逐患者逐模态审计验证了lambda=0恒等、softmax差等于tanh、两类文本相同则零注入、交换ASD/TD顺序后残差反号且共有方向不变。不同冻结文本Bank改变center和delta方向。

## 基座与训练协议

唯一基座为53.3第三组`CADR_WD_0p01_GFR_WD_0p11`。53.3 manifest SHA为`fae43596a9788d63bf8b0a0e52854fd1f7f014fea36c062bcf8a881806d7fcba`，reference fold metrics SHA为`2849ce9d161d5b5ea32bc1b210ebde4f4bd0daf25e316298934b8029b0f8f9a2`。10个基座checkpoint/task/readback SHA及split、pair、GFR/CAM/CADR、schema、anchor和初始化RNG均在预检逐折匹配。初始化与warmup parity为10/10；warmup Hook关闭两步轨迹和fold1 active重复/恢复均逐位一致。新模块使用每折`701000+fold`隔离初始化，三组同折参数SHA一致。

训练保留seed=0、10折`StratifiedKFold(shuffle=True, random_state=10)`、ABIDE A 871例、ASD=0/TD=1、PHENO/ANAT/FUNC/CORR维度48/6/10/256、原CAM/GFR/CADR优化器、原损失、warmup 10、450 epoch上限、wait>200和严格ACC改善checkpoint选择。DCC独立AdamW lr=0.001、WD=0.001，无额外loss/dropout。

## Prompt来源

G1使用BrainPrompt官方ABIDE两条完整原文，将来源顺序`[TC,ASD]`调整为`[ASD,TD]`。G2逐字使用SPromptGL `network.py`八条原文并保留`Nromal`；每条归一化，前四条聚合成TD、后四条聚合成ASD，再归一化并排序。其泛化正常/异常描述不等同于ASD特异性生物标志物。G3复用Experiment 70固定匿名向量，独立seed=700070，匹配G1范数和余弦几何。三组均使用同一CLIP RN50冻结Embedding。

冻结CLIP RN50 embedding cache SHA：`c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`；BrainPrompt原文SHA：`bd30576e77ceb02886a24686f23a7e4d33828368cc892d0d62a3b5a2759ed48f`；SPromptGL原文SHA：`6095cd1694c0b47fcc35010f0bf6924d307c094db790a8ef11e858ad5dc9016c`。最终Bank SHA：G1 `225e6b1307618a7e019f8105618e65db7d2cb9f9d071c2311995446f80bb867e`，G2 `bcc827ed1c9dcf3e69ddda9eebda87a26b9c4ecf933d2c72ff5ea393c999f1b5`，G3 `a9c2dd39686455e8032b8cbfd44ede1c43f3716f8414446d40f480b5b2e2c797`。

BrainPrompt原文（按正式输入顺序）：

1. `A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.`（ASD）
2. `A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.`（TD）

SPromptGL原文（保持来源顺序与拼写）：

- `Nromal brain volume`
- `Uniform cortical thickness`
- `Symmetrical brain structure`
- `Normal ventricular morphology`
- `Altered brain volume`
- `Abnormal functional connectivity`
- `White matter integrity issues`
- `Different activation patterns`

## 十折结果

全部数值为百分比mean±population SD。

| Group | ACC | Hard-AUC / BA | Probability-AUC | SEN-ASD | SPE-TD | ASD-F1 | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| E53.3-BASE | 92.0742±2.8427 | 91.8653±2.8348 | 93.6684±3.2619 | 89.0915±3.5263 | 94.6392±3.5168 | 91.2470±3.0658 | 359/25/44/443 |
| E70.1-G1 | 91.5008±3.0122 | 91.3872±2.9329 | 93.3947±3.2788 | 90.0732±3.3560 | 92.7012±4.9729 | 90.8018±3.0685 | 363/34/40/434 |
| E70.1-G2 | 91.8443±3.0233 | 91.6821±2.9933 | 93.7431±2.9861 | 89.5854±3.4623 | 93.7789±4.0475 | 91.0745±3.2083 | 361/29/42/439 |
| E70.1-G3 | 91.4995±2.8784 | 91.2964±2.8198 | 93.7270±2.7040 | 88.8232±3.9380 | 93.7697±4.9893 | 90.6655±3.0028 | 358/29/45/439 |
| E70-G1 | 91.3845±2.9712 | 91.1370±2.9451 | 93.1169±2.8097 | 87.8476±3.1989 | 94.4265±3.6455 | 90.4422±3.1956 | 354/26/49/442 |
| E70-G2 | 91.4995±2.8784 | 91.3155±2.8601 | 93.5366±2.9435 | 88.8476±3.1381 | 93.7835±3.4255 | 90.6544±3.0574 | 358/29/45/439 |
| E70-G3 | 90.8098±3.0054 | 90.6535±2.9098 | 93.4165±2.8046 | 88.8232±3.9304 | 92.4838±5.5959 | 89.9974±3.0669 | 358/35/45/433 |

逐折指标、best epoch和混淆计数在`fold_metrics.csv`。逐折配对差数组、mean delta和胜/平/负在`paired_comparisons.csv`；其中比较53.3基座、实验70同Prompt组及三个70.1组之间。

## 配对比较

| A-B | ΔACC pp (W/T/L) | ΔHard-AUC pp | ΔProb-AUC pp | ΔSEN pp | ΔSPE pp |
|---|---:|---:|---:|---:|---:|
| E70.1-G1 - E53.3-BASE | -0.573 (0/6/4) | -0.478 | -0.274 | +0.982 | -1.938 |
| E70.1-G2 - E53.3-BASE | -0.230 (1/6/3) | -0.183 | +0.075 | +0.494 | -0.860 |
| E70.1-G3 - E53.3-BASE | -0.575 (0/5/5) | -0.569 | +0.059 | -0.268 | -0.870 |
| E70.1-G1 - E70-G1 | +0.116 (2/6/2) | +0.250 | +0.278 | +2.226 | -1.725 |
| E70.1-G2 - E70-G2 | +0.345 (4/5/1) | +0.367 | +0.206 | +0.738 | -0.005 |
| E70.1-G3 - E70-G3 | +0.690 (6/3/1) | +0.643 | +0.311 | +0.000 | +1.286 |
| E70.1-G1 - E70.1-G2 | -0.344 (1/5/4) | -0.295 | -0.348 | +0.488 | -1.078 |
| E70.1-G1 - E70.1-G3 | +0.001 (2/6/2) | +0.091 | -0.332 | +1.250 | -1.068 |
| E70.1-G2 - E70.1-G3 | +0.345 (2/8/0) | +0.386 | +0.016 | +0.762 | +0.009 |

## 机制证据

残差上界在每个正式训练前向、读回checkpoint的871×4个模态位置均检查；总违规数为0，全30折上界门：PASS。逐患者逐模态残差比在训练与读回中均未超过15%。

| Group | mean abs(s) | patient SD(s) | cross-modality SD(s) | attention entropy | mean abs(lambda) | residual/CAM RMS mean / max | gate mean | gate saturation | graph-off attention L1 | graph-off dP(ASD) | graph-off flips/fold | residual-off dP(ASD) | residual-off flips/fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E70.1-G1 | 0.985 | 0.081 | 0.019 | 0.019 | 0.0279 | 2.73% / 3.70% | 0.335 | 0.025 | 0.299 | 0.161 pp | 1.30 | 0.315 pp | 1.50 |
| E70.1-G2 | 0.978 | 0.073 | 0.021 | 0.029 | 0.0270 | 2.66% / 3.83% | 0.355 | 0.039 | 0.304 | 0.167 pp | 0.70 | 0.184 pp | 0.80 |
| E70.1-G3 | 0.955 | 0.143 | 0.040 | 0.056 | 0.0250 | 2.37% / 3.55% | 0.391 | 0.017 | 0.350 | 0.127 pp | 0.30 | 0.213 pp | 0.40 |

签名竞争强度具有患者与模态间变异，且其绝对值接近1、注意力熵很低，表示多数位置的二类竞争已接近饱和；低熵本身不证明医学语义有效。实际残差均值约占CAM RMS的2%至3%，低于15%上限较多。固定checkpoint清零GFR条件会明显改变注意力，但平均ASD概率和硬预测改变较小；清零语义残差的预测敏感性也有限，说明条件交互真实存在，却没有形成强分类贡献。

各折W_f、W_d、W_c及门控梯度和更新量，以及站点eta²、年龄、FUNC头动与ANAT/FUNC质量相关见`mechanism.csv`。这些相关只作诊断，不能证明病灶解释；质量变量不是脑部病灶指标。固定checkpoint干预不等于重新训练组件消融，也不能替代外部验证。

## 结果判定

预注册基座性能门：FAIL；真实文本相对匿名Token门：FAIL；严格残差机制门：PASS。科研结论为**NO-GO**。

真实文本与匿名Bank可能经共享可学习投影形成相似潜在方向；性能差异不自动证明CLIP文本包含独立ASD医学知识。较实验70改善但仍未过53.3基座，也不能被解释为第二创新有效。单seed、10折折间相关和按历史测试指标选checkpoint不支持统计显著性或临床泛化主张。

## 审计边界

原始模型源码、数据、历史53.3/59/70目录保持只读；`/data`目标路径创建返回EIO Errno 5，故正式实验放在`/model/spromptgl/experiment/experiment70_1_dcc_gcdsi`。所需运行代码、输入和缓存从服务器实际来源复制到本实验`.runtime/reference_snapshot`并按SHA逐项核验。原始代码、数据及历史SHA列在manifest。

保留历史`val=test` checkpoint选择、外层CV前RFE/标准化和871节点transductive图；因此不是无偏外部泛化估计。下一步建议是多seed、严格嵌套/外部站点验证和预注册组件分析。

交接文档SHA：`129993a002fbc9861a21ef406ee82d6aa74bd0bf0e508f813b7e01e70528706e`。
