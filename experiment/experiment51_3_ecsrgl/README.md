# Experiment 51.3 — EC-SRGL

EC-SRGL 在 Experiment 51 E1（四路 GFR 消息聚合修复、无 PEN）上，仅在原始 GraphLearn 支持边内学习患者对选择；同一选择权重同时控制新增邻域消息和 CORR 锚定关系几何对齐，并将图残差与私有残差分开写回原 CAM 输入。

## 运行与恢复

工作目录应设置为本目录，数据只读引用 `/data/adodas/SPromptGL-main/data/ABIDE`。完整训练需要本实验的 `.runtime/`，交付包按约束不包含该目录；scratch 运行副本保留 checkpoint、fold 数据、固定 pair/align 索引和 RNG 状态。

```bash
export ECSR_EXP=/path/to/experiment51_3_ecsrgl
python -B tests.py
python -B run.py prepare
ECSR_WORKERS=5 python -B run.py controller
python -B run.py aggregate
python -B audit.py
```

`controller` 可重复执行，已完成的 `group/fold` 状态不会重训；中断任务从 `.runtime/*latest.pt` 恢复，最佳 checkpoint 只在严格更高 ACC 时更新。`audit.py` 从最佳 checkpoint 独立重算指标，并生成 `results/mechanisms.csv`、`results/interventions.csv` 和 `.runtime/independent_audit.json`。

## 固定协议

seed=0；10-fold StratifiedKFold，`shuffle=True, random_state=10`；历史 `val=test` 选模；450 epoch、`wait>200`；E1 主干 lr/weight_decay 为 0.0038/0.11；新增分支为 0.001/0.001；ASD=0、TD=1。E8 使用与 E7 相同的连续原图支持和边权预算统一缩放，E9 只置换几何距离对应，E10 只置换新增分支关系身份。

## 结果审计

查看 `results/report.md`、`summary.csv`、`fold_metrics.csv`、`paired_comparisons.csv`、`mechanisms.csv`、`interventions.csv` 和 `manifest.json`。`.runtime` 不属于正式交付文件，但独立审计所需的 checkpoint 和 RNG 必须保留在受控 scratch 归档中；删除它会使逐折重算无法进行。

本实验保留历史 CV 前 RFE/标准化、transductive 871 节点图以及测试折参与 checkpoint 选择，因此结果用于历史协议内比较，不能称为无偏外部泛化估计。
