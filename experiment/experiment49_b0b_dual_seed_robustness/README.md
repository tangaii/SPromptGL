# Experiment 49 — B0-B Dual-Seed Robustness Study

本实验只改变训练随机种子和 StratifiedKFold 的 random_state，复用物理移除 PEN 的 B0-B GFR+CAM。训练 seed 在每个组合开始时设置一次，随后同一组合按 fold 1–10 顺序连续消费 RNG；CV seed 独立控制十折划分。

模型、损失、weight_decay=0.11、原始 GraphLearn/两层 GCN、PHENO 覆盖行为、val=test checkpoint 选择、外层前 RFE/标准化和 871 人 transductive 输入均保持历史协议。

正式配置位于 `config.json`。执行脚本为 `run.py`，独立模型和损失实现快照位于 `.runtime/sources/`。`.runtime` 中的 checkpoint 和任务记录用于恢复及独立复核；结果表位于 `results/`。

本实验的 20 组结果是描述性的 seed×split 稳定性分析，不把最高组合当成新方法，不把 200 个测试折当成相互独立的数据集。
