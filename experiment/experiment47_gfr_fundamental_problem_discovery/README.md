# Experiment 47 — GFR Fundamental Problem Discovery

本目录是问题发现与机制归因实验，不包含新的 GFR、Prompt、反事实教师或训练模块。Stage A/B 只读取经核验的 B0-B 最佳 checkpoint，执行原图统计、固定模型干预和训练折内条件信息探针；Stage C 未启动。

## 冻结协议

seed=0、10 折 `StratifiedKFold(random_state=10)`、871 名受试者、四模态 48/6/10/256、阈值 0.9、原两层 GCN、原始 GFR 的 `i==1` 覆盖行为、原始 `GraphConstructLoss` 和 `val=test` 选模均保持不变。所有新的候选零假设只使用输入特征和 outer-train 协变量；测试标签只在固定模型诊断和最终统计时读取。

原始代码和 checkpoint 通过 Experiment43.1 的隔离 PEN-free 源码快照加载。CAM-only 复核通过 Experiment43.4 的物理 CAM checkpoint 加载。由于 `/data` 的 OSSFS 在本轮开始时返回 `Errno 5`，运行暂在 `/tmp/SPromptGL-E47/experiment47_gfr_fundamental_problem_discovery`，最终同步状态记录在 handoff 中。

## 历史事实—原始证据—尚未回答的问题

| 历史事实 | 原始证据 | 本轮仍需回答 |
|---|---|---|
| B0-B 物理移除 PEN 后 ACC=91.7294%；GFR/CAM 仍保留 | `experiment43_1.../results/report.md`、fold checkpoints | 原图是否包含可利用的条件疾病关系 |
| 原 GFR 的 PHENO 消息在 `i==1` 被 `combined_feat=f_w*feat1+raw` 覆盖 | 隔离 `network.py:GFR.forward` | 该错误是否对最终分类有实质影响，还是被其他分支抵消 |
| Experiment43.2 报告 GFR 输出相对输入约 6.9e-5、关闭消息无类别翻转 | `experiment43_2.../results/report.md` | 弱消息来自优化捷径、图无信息，还是传播目标冲突 |
| Experiment44 增强了消息但未建立有益关系证据 | `experiment44.../results/report.md` | 固定原图的疾病增量是否存在 |
| Experiment44.1 的效用教师未通过未见接收者可靠性预检 | `experiment44_1.../results/report.md` | 不能把该停机当成患者图信息不存在的证明 |
| Experiment45 的 PID 只得到 `RELIABLE_BUT_NONDISCRIMINATIVE` | `experiment45.../results/report.md` | 不能据此否定所有模态互补 |
| Experiment46 只测试固定 Top-16 均值；RAW320 探针存在整体 LayerNorm 耦合 | `experiment46.../results/report.md` | 用分离归一化和真实原 GFR 四图重测条件增量 |
| Experiment39 当前工作树没有可读取目录，仅在历史 handoff 留有 `/tmp` 引用 | `handoff.md` | 相关结论只能作为未复核的历史线索，不能作为本轮直接证据 |

## 运行与审计

`run.py` 为唯一执行入口，`integration.py` 统一数据顺序、checkpoint、模态切片和原始 GFR 前向。`graph_audit.py`、`confound_audit.py`、`incremental_probe.py`、`propagation_audit.py`、`optimization_audit.py` 是审计入口。GPU0 运行 fold 1–5，GPU1 运行 fold 6–10。每折只写一个 compact JSON，再合并到 `results/`。

图零假设在 outer-train 子图上按每行非自环度保持，并同时构造站点匹配随机关系。站点、年龄和性别只来自实际 PHENO 字段；没有把不存在的 FD、DVARS 等字段当作可用变量。条件探针先在训练折拟合 `neighbor_summary ~ self` 的固定 ridge 残差化，再分别标准化自身和邻居残差，避免整体 LayerNorm 把二者耦合。RAW320 是主检验，CAM36 是辅助检验；合成正控制验证探针能检测已知邻居信号。

所有干预均固定 B0-B 参数。`ALL_TEST_ROWS_PERMUTED` 是包含目标自身的压力测试，不能解释为 cohort dependence；真正的 P6 检验是 `OTHER_TEST_ONLY_TARGET_HELD_FIXED_GRAPH` 与 `...RECOMPUTE_GRAPH`，目标特征保持不变，只置换其余测试患者。

## 解释边界

图同质性是统计关系，不能单独等同于疾病机制；固定图干预是模型依赖分析，不能单独证明重新训练后的性能因果；`val=test`、预 CV RFE/标准化、transductive 871 节点和单 seed 都限制泛化解释。最终报告必须将已证实的问题、仅观察到的关联、竞争解释和未能判断的假设分开。
