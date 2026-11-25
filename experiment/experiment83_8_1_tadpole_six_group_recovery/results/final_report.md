# Experiment 83.8.1 最终报告

## A. STATUS

正式状态为 COMPLETE_AUDITED。60/60 个 fold 完成，方法保真、训练执行、损失公式和 checkpoint replay 均通过。性能结论为 NO_GO，X4 没有同时超过历史 G4 的 ACC 与 AUC_original，新增组也没有达到原论文参考指标。E83.8 原目录和 BLOCKED_PREFLIGHT 记录保持不变。

## B. 实验环境与目录

正式目录为 /model/spromptgl/experiment/experiment83_8_1_tadpole_six_group_recovery。首选 /data/adodas/SPromptGL-main/experiment/experiment83_8_1_tadpole_six_group_recovery 创建时遇到 OSSFS EIO/errno 5，切换到 /model 路径后写入和读回正常。

单张 PPU-ZW810E，正式并发上限和实际 worker 数均为 8。资源探针 17.31 秒完成，探针保留显存峰值 638 MiB，正式训练折的最大峰值 1120 MiB。训练前 source manifest SHA-256 为 44ef58561541d3e57bac124d1835d7991ba026c401376342ac226191fe1e3beb，training seal 文件 SHA-256 为 2e9ac05cab69fd8da6e19bd185e9820ce539b7f90bb7387d0a5361c95a77e079，封存实现 SHA-256 为 397aac9b3be93b38c415d68179a64b27294ea1f0ec15b9d986a7f38b8d0f6b1d。

## C. 论文及源码审查

原论文 Eq.(1) 将图平滑正则和平方 Frobenius 标签图项纳入图目标，Eq.(7) 将图目标与标签分类损失组合。作者仓库锚点为 ba3b9446fa451fde6864d6555bdf324de0c4b64a。作者 network.py 的 ALoss 对四个硬编码图分支计算未平方 torch.linalg.norm 平均。作者 model.py 的 simple-2 反向传播使用 cls_loss + all_graph_loss，ALoss 在反向传播之后计算并打印。作者四模态 GFR 与本实验六个真实 TADPOLE 模态结构不同。

审查的 E83.5、E83.6、E83.7 源码锚点分别为 4649c5d、c2f9919、a08b8e0。E83.6 的 pair_mean 监督连续余弦图上所有训练有序患者对的 MSE，系数 1.0。论文平方 Frobenius、作者源码未平方范数和 E83.6 连续相似度 pair_mean 是三个不同数学对象。

## D. E83.8 阻断原因

E83.8 的 ALoss 公式身份检查通过，ALoss 确实进入 historical_total 和 backward，X1-X4 fold 1 有五个模态接收有效梯度。其全局门禁额外要求六个模态都具有非零梯度。固定 threshold=0.9 下，M1 阈值图没有非对角边，加入自环并归一化后为恒等图，因此 M1 ALoss 梯度约 7.5e-14。该结果是阈值图导致的结构性零梯度。

E83.8 的 X5/X6 PairMean 独立预检通过。全局 ALoss 梯度门禁却阻止全部组，所以 0/60 是门禁策略造成的 BLOCKED_PREFLIGHT，不表示六组训练实现都失败。公式错误、数值不稳定、意外 detach、方法保真失败和结构性零梯度已分别审计。E83.8 原审计仍标记 BLOCKED_PREFLIGHT，未改写。

## E. E83.8.1 门禁修正及源码差异

X1-X4 按有效图边判断梯度门禁。六个真实模态都必须参与公式并提供 finite loss，存在非对角边的可微模态必须提供有效梯度，无有效边模态可标记 STRUCTURAL_ZERO_GRADIENT。所有组还必须通过训练标签隔离、测试标签置换不变性和整体 GraphLearn 有效梯度检查。X5 与 X6 各自运行 PairMean 门禁。损失特有检查只影响依赖该损失的组，共享数据和源码身份检查影响全部组。

源码 AST 对比确认 paper_aloss、PairMean、historical_loss、warmup、train_fold、DPSL optimizer step、模型构造和优化器配置均与 E83.8 一致。六组配置的 method 字段一致，只有 experiment identity 改为 E83.8.1。训练执行器的允许差异是 per-group gate 调用。fidelity_model.py 字节级一致。源文件和配置对比见 audits/source_comparison.json。

训练结束后的 finalizer 有一个报告阶段作用域错误，引用了未设置的 folds 全局变量。训练、checkpoint、训练 seal 和预测均已完成。使用 audits/finalize_recovery_wrapper.py 调用原 finalizer 完成结果聚合，未修改封存训练源码。封印中的 232 个文件与当前哈希全部一致。

## F. ALoss 数学保真

X1-X4 使用冻结的 E83.8 定义，ALoss = (1/6) Σ[m=1..6] || Abar_m[train,train] - Abar_label[train,train] ||_F²，系数为 1。六个 Abar 来自 GFR 实际构造的六个模态图。每个图均按对称 D^-1/2(A+I)D^-1/2 在 598 个转导节点上归一化后，再抽取训练患者子矩阵。标签图由训练患者真实标签生成，同病为 1、异病为 0，归一化后只使用训练乘训练子矩阵。训练对角线参与监督，held-out 标签关系不构造，测试标签置换不改变目标和梯度。

原有图平滑、度数和稀疏正则保持不变。ALoss 作为唯一新图监督项加入 historical_total，并参与实际反向传播和 GFR 优化器更新。此定义结合论文平方 Frobenius 约束、作者图归一化操作与训练标签限定，是登记的 TADPOLE 实现版本，不声称等同于未公开的论文 TADPOLE 代码。

## G. 六模态图结构及梯度分析

预检 fold 1 的 X4 非对角图边数 M1-M6 为 0、204、91690、1162、3334、4276。M1 归一化恒等图的单模态 ALoss 标量为 529.117，除以六后仍贡献约 88.186 的常数损失，但没有有效图参数梯度。M3 图密度约 0.2568，明显高于 M1 的 0。

X4 fold 1 的六个单模态 GraphLearn p.weight 梯度范数为 7.5e-14、0.3715、0.0344、0.2951、0.6605、0.7272。M2-M6 梯度有效，整体 ALoss GFR 梯度范数 1.1547。训练后 Best 的 X4 六模态平均梯度范数分别为 0.0111、0.3361、0.1059、0.3171、0.2116、0.1221，M2 在 6 折最大，M4 在 4 折最大。

X4 Best 相对 H4 的六模态邻接矩阵改动项数为 106510、306870、1876275、306144、585844、1051977。跨十折汇总的 M1 非零项由基线 6078 增至 106786，M3 保持高密度，非零项由 1576792 变为 1565426。ALoss 训练改变了图学习结果，M1 的 fold 1 初始结构零梯度没有通过调整阈值或伪造梯度规避。

## H. 六组方法保真

X1 为 G1 + Paper-ALoss，保留六模态 GFR、CAM、MMGPL-12 三分类 PEN、原始融合、分类头、损失和优化器。X2 为物理 No-PEN G2 + 同一 ALoss，没有恢复 PEN、SPGS、CLIP 或额外 Prompt 参数。X3 保留 DARG 正负 signed 关系、Positive-DCR、关系融合、训练损失、优化器和十轮 warmup，再加入同一 ALoss。X4 保留 G4 DARG+DPSL、医学 Prompt、三组诊断证据、残差上限 0.08、共享梯度预算 0.001 和原优化器，ALoss 只加入 GFR historical_total。

X5 是 G4 + Class-Balanced PairMean，系数 0.1，前十轮关闭，epoch 11 起启用。X6 再使用原生分类头训练预测的归一化熵权重，系数、时序和结构与 X5 相同。DARG、DPSL、GFR 阈值、Prompt、图构造和历史数据协议均未改变。

## I. 初始化和 warmup 配对

X3/X4 逐折共享初始参数、optimizer、RNG、DARG 关系对及十轮 ALoss warmup，10/10 配对通过。X5/X6 逐折共享初始状态、optimizer、RNG、关系对和前十轮无新增损失 warmup，10/10 完全一致，epoch 1-10 的 PairMean 为零。

历史 H4 缺少初始模型、optimizer 和 RNG 哈希，因此 X5/H4、X6/H4 为 PARTIAL_MATCH。X1/H1 与 X2/H2 的历史初始哈希也无法证明，按 PARTIAL_MATCH 记录。没有将缺失历史哈希包装成完全配对。

## J. 六组正式训练完成情况

| 组 | 正式 folds | 结果 |
|---|---:|---|
| X1 | 10/10 | COMPLETE |
| X2 | 10/10 | COMPLETE |
| X3 | 10/10 | COMPLETE |
| X4 | 10/10 | COMPLETE |
| X5 | 10/10 | COMPLETE |
| X6 | 10/10 | COMPLETE |

共 60/60 新增正式 fold。每折均从注册的初始状态完整训练。没有训练失败或覆盖已完成 fold。每组保留 Best、Latest、fold audit、预测和指标。Best 由首次严格提高的测试折 ACC 保存，训练上限 450 epoch，wait > 200。

## K. ACC/AUC 完整结果

十折结果按 mean ± population std 报告。ACC、AUC_original 和 AUC_probability 为百分比。AUC_original 使用历史硬预测 one-hot macro OVR，AUC_probability 只作辅助。

| 组 | ACC | AUC_original | AUC_probability | OOF正确 | NC/MCI/AD正确 |
|---|---:|---:|---:|---:|---:|
| H1 | 87.1130 ± 3.9345 | 90.6485 ± 3.0545 | 93.8880 ± 2.0794 | 521/598 | 184/269/68 |
| H2 | 88.7797 ± 3.9571 | 90.7319 ± 2.8838 | 95.3951 ± 2.2256 | 531/598 | 185/283/63 |
| H3 | 92.4746 ± 3.1913 | 93.9800 ± 2.7433 | 95.5545 ± 3.0104 | 553/598 | 205/283/65 |
| H4 | 92.9746 ± 2.7816 | 94.0967 ± 2.6088 | 96.2710 ± 2.2532 | 556/598 | 205/287/64 |
| H5 | 85.4407 ± 6.4461 | 88.3072 ± 4.7604 | 90.7601 ± 4.3304 | 511/598 | 188/263/60 |
| X1 | 79.5932 ± 2.7106 | 85.7120 ± 2.9594 | 90.0293 ± 2.4299 | 476/598 | 181/230/65 |
| X2 | 75.3927 ± 5.5729 | 83.5551 ± 4.3229 | 88.1499 ± 3.6390 | 451/598 | 177/207/67 |
| X3 | 92.1412 ± 3.7292 | 93.2876 ± 2.8881 | 96.4810 ± 1.9123 | 551/598 | 197/290/64 |
| X4 | 91.8023 ± 2.5436 | 92.7822 ± 2.4730 | 95.6699 ± 1.6754 | 549/598 | 200/287/62 |
| X5 | 91.7966 ± 3.8748 | 93.3526 ± 3.7211 | 95.4142 ± 2.2385 | 549/598 | 199/285/65 |
| X6 | 92.6384 ± 3.2867 | 93.3559 ± 2.9560 | 95.2975 ± 3.3128 | 554/598 | 201/291/62 |

逐折 ACC/AUC、患者级 OOF 预测和类别统计分别保存在 results/fold_metrics.csv 与 results/oof_predictions.csv。

## L. 与历史 G1-G4 对照

| 对比 | ΔACC | ΔAUC_original | 净正确变化 | 纠正 | 误伤 |
|---|---:|---:|---:|---:|---:|
| X1 − H1 | -7.5198 pp | -4.9364 pp | -45 | 41 | 86 |
| X2 − H2 | -13.3870 pp | -7.1769 pp | -80 | 23 | 103 |
| X3 − H3 | -0.3333 pp | -0.6924 pp | -2 | 20 | 22 |
| X4 − H4 | -1.1723 pp | -1.3145 pp | -7 | 26 | 33 |
| X5 − H4 | -1.1780 pp | -0.7441 pp | -7 | 23 | 30 |
| X6 − H4 | -0.3362 pp | -0.7407 pp | -2 | 24 | 26 |
| X6 − X5 | +0.8418 pp | +0.0034 pp | +5 | 24 | 19 |

所有新组均未同时超过各自历史参考组的 ACC 与 AUC_original。H1-H5 是 E83.5 的真实历史 checkpoint 和预测，未重新训练。H2 是保留 GFR 的物理 No-PEN 组，H5 才是 CAM-only。

## M. 与 E83.6、E83.7 图监督实验对照

| 历史组 | ACC mean ± std | AUC_original mean ± std | 正确人数 |
|---|---:|---:|---:|
| E83.6-D2 | 92.4746 ± 2.5090% | 93.5976 ± 2.6074% | 553/598 |
| E83.7-T4 | 91.9689 ± 3.3345% | 93.4678 ± 3.1796% | 550/598 |

E83.6-D2 使用全训练有序患者对的连续相似度 MSE，系数 1.0。E83.7-T4 只在前十轮加入图监督。它们的目标函数或启用时点与 E83.8.1 不同，以上为描述性比较。

## N. ALoss 训练效应与梯度尺度

fold 1 初始 ALoss 为 X1/X2 288.405，X3/X4 297.405。分类损失分别约 1.202-1.395，ALoss 标量约为分类损失的 213-240 倍。标量大小没有被用作自动阻断或动态调系数的理由。历史图构造项为负值，直接以该标量计算损失比没有解释意义。

X1 fold 1 的整体 GFR ALoss 梯度范数为 1.3733，历史 GFR 梯度范数为 2.6899，范数比 0.5105，余弦相似度 0.00185。X4 对应为 1.1547、2.5583、0.4514 和 0.00214。只看 GraphLearn p.weight 参数时，X1 的 ALoss 与历史梯度范数为 1.3177 与 0.01670，范数比 78.89，余弦 0.2705；X4 为 1.0915 与 0.01303，范数比 83.76，余弦 0.3650。不同参数子集呈现不同尺度。新旧梯度均 finite，不能仅用单个损失标量比例描述实际更新。

X4 fold 1 初始 Adam 反事实步的 GFR 更新范数，历史损失单独更新为 15.3858976，合入 ALoss 后为 15.3858995，范数比 1.00000012。此值只描述初始化的一步 Adam 预检，不能替代完整训练的因果结论。ALoss 训练轨迹十折均下降，平均起始到最后活动值为 X1 287.929 至 12.762，X2 287.929 至 13.322，X3 121.080 至 12.687，X4 121.080 至 13.924。训练 Best 的平均 ALoss 分别为 56.901、76.902、58.317、82.419。

ALoss 实际改变六模态图和 GraphLearn 参数，但未改善 G4 原生分类结果。X4 的 DARG 正负 signed 关系仍存在，十折正边数按模态为 3504、13690、2058、15742、24670、17904，负边数为 3020、6846、962、6652、1286、730。DARG 关系定义未被 ALoss 替代，但关系边分布相较 H4 改变。X4 DARG pair loss 均值 0.5762，高于 H4 的 0.4531，保留损失仍在训练路径中。

## O. X5/X6 优化版 PairMean 分析

X5 将 NC-NC、MCI-MCI、AD-AD、NC-MCI、NC-AD、MCI-AD 六组 unique 训练患者对各自计算连续余弦相似度与 0/1 同病标签 MSE，再对六组等权平均，系数 0.1，epoch 11 起启用。同类自连接排除，跨类对只计一次。fold 1 各组对数为 17578、39903、2211、53204、12596、18961，共 144453。类别对组各贡献总监督的 1/6，MCI 患者对数量不会因数量优势占据大部分组权重。

X6 在 X5 上应用 w_ij = 0.5 + 0.5*(u_i+u_j)/2，其中 u_i = H(p_i)/log(3)。权重取自训练患者当前原生预测、范围 [0.5,1]、detach，并在类别对组内归一化加权 MSE。测试标签置换保持损失及梯度不变。

初始相同状态的 X5/X6 PairMean 分别为 0.502272 和 0.502806，绝对差 0.000534，差异约 0.106%。图监督梯度范数分别为 0.0026125 和 0.0026140，范数比 1.000566，方向余弦 0.99822。X6 权重已 detach，权重路径没有原生 logits 梯度。

训练后 X6 六组平均权重 NC-NC 0.6532，NC-MCI 0.6531，MCI-MCI 0.6530，NC-AD 0.6406，MCI-AD 0.6405，AD-AD 0.6280。组内权重标准差约 0.044-0.059，观测最小值约 0.560，最大值约 0.918。权重主要略偏向 NC/MCI 相关对，组间差异有限，没有集中到少数类别。六组训练后平均损失 X5 为 0.4695、0.6052、0.2942、0.3552、0.2619、0.3185，X6 为 0.4255、0.5408、0.2717、0.3860、0.2637、0.3351。

X6 相比 X5 净增加 5 名 OOF 正确患者，ACC 增加 0.8418 pp，AUC_original 仅增加 0.0034 pp。X6 相比 H4 仍少 2 名正确患者且 AUC_original 低 0.7407 pp。它提供相对 X5 的 ACC 增量，没有保护到超过 H4 的完整模型性能。

## P. G4 Native/Full 患者级纠错

Native 与 Full 使用同一个 Best checkpoint，区别仅是是否启用 DPSL 推理纠错。下表的 H4 对比按患者预测单独计算，不能与同一 checkpoint 的 Full-Native 直接纠正人数混用。

| 组 | Native正确 | Full正确 | Full-Native纠正/误伤 | 相对H4 Full纠正/误伤/净变化 | NC/MCI/AD正确人数净变化 |
|---|---:|---:|---:|---:|---:|
| X4 | 549 | 549 | 0/0 | 26/33/-7 | -5/0/-2 |
| X5 | 547 | 549 | 2/0 | 23/30/-7 | -6/-2/+1 |
| X6 | 554 | 554 | 0/0 | 24/26/-2 | -4/+4/-2 |

X5 的两名 Full-Native 纠正患者在历史 H4 Full 中本来已正确。X4、X6 的 DPSL 没有产生 Native 到 Full 的翻转。相对 H4 的患者级变动主要体现新训练后的分类改变，不能归因于 DPSL 推理纠错。X6 对比 X5 增加六名 MCI 正确患者，同时少三名 AD 正确患者，类别变化不足以构成超过 H4 的整体提升。

X4 的 DPSL 最大残差 0.079999，低于 0.08 上限。共享主干梯度比例最大为 0.000999894，低于 0.001 预算。DPSL 参数梯度非零不等价于共享主干梯度非零，本报告依据逐折实测比例确认预算。

## Q. 与原论文结果对照

原论文参考 ACC 为 94.81 ± 2.95%，AUC 为 95.59 ± 2.57%。新组最高 ACC 和 AUC_original 均为 X6，92.6384% 和 93.3559%，两项均未达到参考均值。H4 历史最好结果为 92.9746% 和 94.0967%，也未达到论文参考。

## R. Best/Latest 独立审计

EXECUTION_AUDIT 为 PASS，60/60 fold 完整，598 名患者在每组 OOF 中恰好出现一次，训练和测试索引无交集，数据 SHA、split SHA 与 seal 一致。METHOD_FIDELITY 与 LOSS_FORMULA_AUDIT 为 PASS。X1-X4 的梯度门禁各为 PASS_WITH_STRUCTURAL_ZERO_GRADIENT，X5/X6 特有门禁通过。X3/X4 与 X5/X6 warmup 配对均为 10/10。历史 H1-H5 checkpoint 哈希与独立快照一致。

CHECKPOINT_REPLAY 中 Best 60/60、Latest 60/60、恢复后下一步确定性 60/60 通过。DPSL 残差及共享梯度上限、六模态图监督、标签隔离和公式身份均通过。独立记录 STRUCTURAL_ZERO_GRADIENT_WARNING，M1 现象不被作为公式错误或训练失败。

PERFORMANCE_GO_NO_GO 为 NO_GO。审计通过不代表性能成功。训练仍保留 val=test 的测试 ACC 选 Best、全数据 RFE、CV 前 StandardScaler 和转导患者图，结果可能有乐观偏差。

## S. 必要结果文件

主要指标和患者结果为 results/group_summary.csv、results/fold_metrics.csv、results/oof_predictions.csv、results/paired_contrasts.csv、results/dpsl_patient_mechanism.csv。X6 权重样例见 results/x6_uncertainty_pair_examples.csv。

源码与身份文件为 protocol.json、source_manifest.json、training_seal.json、audits/source_comparison.json、audits/method_fidelity_matrix.md。审计文件为 audits/preflight_start.json、audits/preflight_X1.json 至 preflight_X6.json、audits/loss_fidelity.json、audits/checkpoint_replay.json、audits/warmup_pairing.json、audits/mechanism_summary.json、audits/final_audit.json 和 audits/historical_checkpoint_snapshot.json。每组 Best、Latest 和 fold audit 位于 groups/X1 至 groups/X6。

## T. handoff 更新

已向 /model/spromptgl/experiment/handoff.md 追加唯一 Experiment 83.8.1 章节，并在写入后读回确认章节只出现一次。章节包括阻断原因、门禁修正、源码和数据 SHA、六组正式 fold、三类 AUC/ACC、M1 结构性零梯度、DPSL机制、独立审计、科研 GO/NO-GO 和恢复命令。

## U. 最终科研结论

论文式平方 ALoss 确实进入训练、产生整体 GFR 梯度并改变六模态图，但没有弥补 G4 的分类性能损失。G4 + ALoss 的 X4 比 H4 少 7 名正确患者，ACC 与 AUC_original 均下降。类别均衡 PairMean 及不确定性加权版本保持了相对 X4 更高的 Native 性能，其中 X6 比 X5 多 5 名正确患者，主要来自 MCI；但 X5、X6 仍未超过历史 G4 的完整性能。没有证据支持 ALoss、CB-PairMean 或 CB-UPairMean 替代 G4 · E82.4.6.10-G1。当前最终完整模型仍为 DARG + DPSL 的 G4，实验方法审计为 GO，性能改进判定为 NO-GO。

