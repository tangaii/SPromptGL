# Experiment 48.8 — Disease-Relation Consistent APCR

本实验固定 Experiment 48.7 的 G-F-05：Experiment51 E1 `LegacyFixedGFR + CAM`、四路完整 320 维消息累加、APCR `alpha=0.0015` fixed，物理移除 PEN。仅改变 SRD、SCA、CRA 内部逻辑。

## 研究组

- E0：G-F-05 原版重训，先作为历史门控。
- E1：候选图感知 SRD。
- E2：可靠教师定向 SCA。
- E3：边界患者局部最坏 CRA。
- E4：E1+E2；E5：E1+E3；E6：E2+E3；E7：三模块完整方案。
- E8：患者级疾病标签置乱；E9：同候选预算随机攻击；E10：同权重但不改边的伪攻击。

所有组为 seed=0、10-fold StratifiedKFold（shuffle=True, random_state=10），每组 10 折。候选 SRD 使用 1024 个全局关系对和 1024 个当前 Top-24 候选关系对；SCA 只在同一患者 ID 的关系对上选择校准可靠教师；CRA 每轮选择 16 个边界患者和 16 个分层随机患者，在合法候选集合内做两步有限 CE 最大化。

## 审计和复现

论文依据包括 SPromptGL（MICCAI 2025）、AAAI 2026 alignment paper 和 TRADES/ICML 2019。实际基座源码来自 Experiment51 E1 和 48.7 G-F-05 冻结快照；训练代码签名记录在 `manifest.json`。E0 逐折指标、best epoch、预测 SHA、初始化 SHA 和 split SHA 与历史 G-F-05 完全一致。E1–E10 的 100 个最佳 checkpoint 已由独立进程读回，五项指标和预测 SHA 全部一致；旧签名 partial tasks 已归档到 `.runtime/formal_v2_pre_fix/`，不进入统计。

## 结果和限制

结果位于 `results/report.md`、`summary.csv`、`fold_metrics.csv`、`paired_comparisons.csv`、`mechanism_diagnostics.csv` 和 `interventions.csv`。E7 没有超过 E0，且没有优于疾病标签置乱 E8；低 alpha 下关闭 APCR 或错配关系不改变硬预测。工程审计通过，科研机制证据不足，报告判定 `MICCAI_CANDIDATE_STATUS=NO-GO`。

沿用历史全局预处理/CV 前 RFE、871 节点 transductive 图、`val=test` checkpoint 选择和单 seed；因此结果不是严格无偏外部泛化估计，也不应作为多重比较后的确认性结论。

由于 `/data/adodas/SPromptGL-main` 当前写入探针返回 OSSFS `Errno 5 Input/output error`，本次产物暂存于 `/tmp/SPromptGL-E48.8`，handoff 追加文本在 `.runtime/handoff_pending.md`。
