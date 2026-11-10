# Experiment 76.4 — LBDP 局部混淆边界原型验证

状态：`PASS`；结论：`NO-GO_LOCAL_PROTOTYPE_CORRECTION`。冻结特征实验，未训练DARG/GFR/CAM/OutputLayer。

## 方法和审计范围

在E74.4-G2 Positive-DCR最佳Checkpoint上，DARG、CAM和原OutputLayer保持冻结。E76.2角色阈值与稳定性（10次CAM144确定性扰动，>=9次预测不变）沿用为共同定义；困难样本直接来自该MLP自身错误预测。M2复用E76.2可靠/困难全局类别原型和全局对比方向。M3对每个真实误判患者建立`hard + same-disease correct boundary/reliable + opposite-disease reliable`局部三元关系，再以CAM144余弦相似度Top-3、温度0.10检索患者条件化方向。每个ASD/TD方向至少3个真实误判且可用三元关系，否则mask。M4在方向内保留角色/类别，仅打乱hard病例与同病/异病正确参照的映射；五个私有置换以校正分数集成，不作为独立折。

残差校正器对M1–M4统一使用训练支持集拟合的StandardScaler、L-BFGS-B二元logistic残差（L2=0.01）：`s_final=s_base+b+w^T z`，ASD分数为`logp_ASD-logp_TD`，正值对应ASD=0；原始分类器未被替换或更新。每个外层训练集内5折，校正训练证据另用3折cross-fit；测试标签只作最终计分。

CoW的困难原型来自支持分割自预测与mask差异，服务像素级分割边界；PGP-SAM使用可学习类内/类间原型和类引导cross-attention生成分割prompt。本实验只借鉴困难/局部关系的分析动机，采用患者分类错误及CAM144的固定特征检索，不是CoW或PGP-SAM的患者级复现。

## 冻结特征内层OOF

mean±population SD（%）；五个内层验证折汇总后以外层fold为统计单位。冻结编码器此前已在外层训练患者上训练，因此这只是条件于冻结表示的读出诊断。

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| M0 原始MLP | 96.9001±0.8541 | 96.7965±0.9266 | 99.1741±0.2209 | 95.3969±2.2061 | 98.1961±1.0291 | 96.5980±0.9757 |
| M1 基础分数校准 | 97.3082±0.5838 | 97.2503±0.5905 | 99.1312±0.2307 | 96.4712±0.9118 | 98.0293±0.7516 | 97.0728±0.6338 |
| M2 全局困难原型 | 97.3465±0.5290 | 97.2915±0.5303 | 99.0378±0.2270 | 96.5537±0.7108 | 98.0294±0.6639 | 97.1160±0.5732 |
| M3 局部边界原型 | 97.3082±0.4963 | 97.2559±0.4916 | 99.0480±0.2264 | 96.5537±0.6889 | 97.9581±0.7453 | 97.0761±0.5342 |
| M4 局部配对打乱 | 97.3593±0.5236 | 97.3033±0.5236 | 99.0453±0.2334 | 96.5536±0.7003 | 98.0531±0.6786 | 97.1296±0.5664 |

## 历史外层测试核验

mean±population SD（%）。M0必须与E74.4-G2原始最佳Checkpoint预测一致。这里的外层测试仍继承历史test折选模、CV前预处理、transductive图和Fold 2训练/测试标签重叠，不能称为无偏独立泛化结果。

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| M0 原始MLP | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| M1 基础分数校准 | 90.9248±3.5846 | 90.8125±3.5251 | 93.4946±2.9195 | 89.3354±3.3306 | 92.2895±4.8463 | 90.1591±3.7250 |
| M2 全局困难原型 | 90.8098±3.1345 | 90.7107±3.0896 | 93.6750±2.6482 | 89.3354±3.3306 | 92.0860±4.2885 | 90.0257±3.3140 |
| M3 局部边界原型 | 90.5799±3.3705 | 90.4956±3.3133 | 93.4898±2.8122 | 89.3354±3.3306 | 91.6559±4.6493 | 89.8121±3.5268 |
| M4 局部配对打乱 | 90.8098±3.1345 | 90.7107±3.0896 | 93.6121±2.6956 | 89.3354±3.3306 | 92.0860±4.2885 | 90.0257±3.3140 |

### 排除Fold 2的描述性结果

排除Fold 2后的9个外层fold均值±总体SD（%）：

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| M0 原始MLP | 92.2153±2.9131 | 92.0481±2.8723 | 93.6142±3.0541 | 90.0474±4.0959 | 94.0487±4.9847 | 91.4703±3.0580 |
| M1 基础分数校准 | 90.8104±3.7611 | 90.6959±3.6974 | 93.6142±3.0541 | 89.2344±3.4962 | 92.1575±5.0913 | 90.0259±3.9039 |
| M2 全局困难原型 | 90.6827±3.2795 | 90.5829±3.2315 | 93.8088±2.7592 | 89.2344±3.4962 | 91.9313±4.4939 | 89.8777±3.4618 |
| M3 局部边界原型 | 90.4273±3.5198 | 90.3439±3.4595 | 93.6148±2.9379 | 89.2344±3.4962 | 91.4534±4.8587 | 89.6403±3.6777 |
| M4 局部配对打乱 | 90.6827±3.2795 | 90.5829±3.2315 | 93.7389±2.8129 | 89.2344±3.4962 | 91.9313±4.4939 | 89.8777±3.4618 |

Fold 2去除1-based行377和407后的单折指标：

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| M0_original_mlp | 91.7647 | 91.6667 | 92.4444 | 90.0000 | 93.3333 | 91.1392 |
| M1_base_score_calibration | 91.7647 | 91.6667 | 92.4444 | 90.0000 | 93.3333 | 91.1392 |
| M2_global_hard_prototype | 91.7647 | 91.6667 | 92.5000 | 90.0000 | 93.3333 | 91.1392 |
| M3_local_boundary_prototype | 91.7647 | 91.6667 | 92.3889 | 90.0000 | 93.3333 | 91.1392 |
| M4_shuffled_local_pair | 91.7647 | 91.6667 | 92.5000 | 90.0000 | 93.3333 | 91.1392 |

## M3配对比较

配对差为M3减对照，单位百分点。下表列外层测试主要指标；完整内层/外层八项指标、全十折与排除Fold 2九折的W/T/L和描述性95%配对区间见`paired_comparisons.csv`。

| 对照 | 指标 | Δ pp | 95% CI | W/T/L | 排除Fold 2 Δ pp |
|---|---|---:|---:|---:|---:|
| M0_original_mlp | ACC | -1.609 | [-2.574, -0.644] | 0/3/7 | -1.788 |
| M0_original_mlp | Hard-AUC/BA | -1.534 | [-2.453, -0.615] | 0/3/7 | -1.704 |
| M0_original_mlp | Prob-AUC | -0.005 | [-0.229, +0.219] | 3/1/6 | +0.001 |
| M0_original_mlp | SEN-ASD | -0.732 | [-2.589, +1.126] | 1/7/2 | -0.813 |
| M0_original_mlp | SPE-TD | -2.336 | [-4.547, -0.125] | 1/4/5 | -2.595 |
| M0_original_mlp | ASD-F1 | -1.647 | [-2.643, -0.651] | 0/3/7 | -1.830 |
| M1_base_score_calibration | ACC | -0.345 | [-1.022, +0.332] | 1/6/3 | -0.383 |
| M1_base_score_calibration | Hard-AUC/BA | -0.317 | [-0.946, +0.313] | 1/6/3 | -0.352 |
| M1_base_score_calibration | Prob-AUC | -0.005 | [-0.229, +0.219] | 3/1/6 | +0.001 |
| M1_base_score_calibration | SEN-ASD | +0.000 | [+0.000, +0.000] | 0/10/0 | +0.000 |
| M1_base_score_calibration | SPE-TD | -0.634 | [-1.893, +0.625] | 1/6/3 | -0.704 |
| M1_base_score_calibration | ASD-F1 | -0.347 | [-1.000, +0.306] | 1/6/3 | -0.386 |
| M2_global_hard_prototype | ACC | -0.230 | [-0.577, +0.117] | 0/8/2 | -0.255 |
| M2_global_hard_prototype | Hard-AUC/BA | -0.215 | [-0.539, +0.109] | 0/8/2 | -0.239 |
| M2_global_hard_prototype | Prob-AUC | -0.185 | [-0.385, +0.014] | 3/1/6 | -0.194 |
| M2_global_hard_prototype | SEN-ASD | +0.000 | [+0.000, +0.000] | 0/10/0 | +0.000 |
| M2_global_hard_prototype | SPE-TD | -0.430 | [-1.079, +0.219] | 0/8/2 | -0.478 |
| M2_global_hard_prototype | ASD-F1 | -0.214 | [-0.536, +0.109] | 0/8/2 | -0.237 |
| M4_shuffled_local_pair | ACC | -0.230 | [-0.577, +0.117] | 0/8/2 | -0.255 |
| M4_shuffled_local_pair | Hard-AUC/BA | -0.215 | [-0.539, +0.109] | 0/8/2 | -0.239 |
| M4_shuffled_local_pair | Prob-AUC | -0.122 | [-0.273, +0.029] | 3/2/5 | -0.124 |
| M4_shuffled_local_pair | SEN-ASD | +0.000 | [+0.000, +0.000] | 0/10/0 | +0.000 |
| M4_shuffled_local_pair | SPE-TD | -0.430 | [-1.079, +0.219] | 0/8/2 | -0.478 |
| M4_shuffled_local_pair | ASD-F1 | -0.214 | [-0.536, +0.109] | 0/8/2 | -0.237 |

## 真实纠错与边界证据

错误翻转按外层fold出现次数汇总；Fold 2两名历史重叠受试者在新增支持/校正拟合中排除，但在完整历史外层测试统计中按预注册保留并单列敏感性分析。稳定误判标签来自原始E74.4-G2的测试预测与冻结特征扰动，不参与推理路由。

| M3相对 | 范围 | Wrong→Correct | Correct→Wrong | 净纠正 | ASD→TD 错误前→后 | TD→ASD 错误前→后 |
|---|---|---:|---:|---:|---:|---:|
| M0_original_mlp | all | 2 | 16 | -14 | 40→43 | 28→39 |
| M0_original_mlp | stable_hard | 1 | 0 | 1 | 39→39 | 28→27 |
| M0_original_mlp | stable_reliable | 0 | 1 | -1 | 0→0 | 0→1 |
| M0_original_mlp | boundary_correct | 0 | 15 | -15 | 0→4 | 0→11 |
| M0_original_mlp | ASD_to_TD_errors | 1 | 0 | 1 | 40→39 | 0→0 |
| M0_original_mlp | TD_to_ASD_errors | 1 | 0 | 1 | 0→0 | 28→27 |
| M1_base_score_calibration | all | 1 | 4 | -3 | 43→43 | 36→39 |
| M1_base_score_calibration | stable_hard | 1 | 0 | 1 | 39→39 | 28→27 |
| M1_base_score_calibration | stable_reliable | 0 | 1 | -1 | 0→0 | 0→1 |
| M1_base_score_calibration | ASD_to_TD_errors | 0 | 0 | 0 | 39→39 | 0→0 |
| M1_base_score_calibration | TD_to_ASD_errors | 1 | 0 | 1 | 0→0 | 28→27 |
| M2_global_hard_prototype | all | 0 | 2 | -2 | 43→43 | 37→39 |
| M2_global_hard_prototype | stable_hard | 0 | 1 | -1 | 39→39 | 26→27 |
| M2_global_hard_prototype | stable_reliable | 0 | 0 | 0 | 0→0 | 1→1 |
| M2_global_hard_prototype | ASD_to_TD_errors | 0 | 0 | 0 | 39→39 | 0→0 |
| M2_global_hard_prototype | TD_to_ASD_errors | 0 | 1 | -1 | 0→0 | 26→27 |
| M4_shuffled_local_pair | all | 0 | 2 | -2 | 43→43 | 37→39 |
| M4_shuffled_local_pair | stable_hard | 0 | 1 | -1 | 39→39 | 26→27 |
| M4_shuffled_local_pair | stable_reliable | 0 | 0 | 0 | 0→0 | 1→1 |
| M4_shuffled_local_pair | ASD_to_TD_errors | 0 | 0 | 0 | 39→39 | 0→0 |
| M4_shuffled_local_pair | TD_to_ASD_errors | 0 | 1 | -1 | 0→0 | 26→27 |

M2/M3按同一原始预测方向区分真实误判患者与正确异类患者的条件AUC：

| 范围 | 方向 | 方法 | 可评估折 | 条件AUC均值 |
|---|---|---|---:|---:|
| inner_oof | ASD_to_TD | M2_global | 10 | 0.4033 |
| inner_oof | ASD_to_TD | M3_local | 10 | 0.3392 |
| inner_oof | TD_to_ASD | M2_global | 10 | 0.3357 |
| inner_oof | TD_to_ASD | M3_local | 10 | 0.3685 |
| outer_test | ASD_to_TD | M2_global | 10 | 0.5007 |
| outer_test | ASD_to_TD | M3_local | 10 | 0.4370 |
| outer_test | TD_to_ASD | M2_global | 9 | 0.5098 |
| outer_test | TD_to_ASD | M3_local | 9 | 0.4281 |

条件AUC只在同一基础预测类别内比较真实误判目标类与正确异类；无双类别折保留NA、不填0.5。局部配对中hard→same-reference同站点比例均值为0.180，hard→other-reference同站点比例均值为0.595。配对类别支持、每折fallback、Top-k、M4改变配对比例及bootstrap局部方向余弦见`prototype_diagnostics.csv`和runtime原型注册表。

## 决策与限制

审计结论：`PASS`。预注册研究判定：`NO-GO_LOCAL_PROTOTYPE_CORRECTION`。

E74.4-G2模型曾用test折选择checkpoint，预处理在外层CV前完成，图为transductive；Fold 2的377/407同时出现在历史训练与测试索引。新增原型库和校正拟合将两人排除，但这不消除表征/图中历史暴露。训练侧困难患者是原始模型对自身训练样本的错误，不能称作独立样本外错误。对M3相对M0的外层测试错误净纠正是Q2的核心证据；不能用校正器内层成绩代替。

源码/文献核验：E74.4-G2 [锁定实现](https://github.com/tangaii/SPromptGL/tree/upload/experiment74-4-soft-positive-dcr/experiment/experiment74_4_soft_positive_dcr)的原OutputLayer输入仍为CAM144。CoW [IJCAI 2025](https://ijcai.org/proceedings/2025/139)及[官方代码](https://github.com/jcjiang99/CoW)研究分割支持集预测误差驱动的hard prototype，不是患者分类。PGP-SAM [arXiv:2501.06692](https://arxiv.org/abs/2501.06692)及[官方代码](https://github.com/PRIS-CV/PGP-SAM)研究类内/类间原型条件化分割prompt，不是此处的患者原型方法。

文件：`fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`correction_analysis.csv`、`boundary_diagnostics.csv`、`prototype_diagnostics.csv`、`calibrator_diagnostics.csv`、`audit_status.json`。

**论文级解释边界：**本实验只验证冻结E74.4表示上的条件化原型检索和轻量校正，不证明独立临床泛化，不证明原型承载CAM144以外的疾病信息。
