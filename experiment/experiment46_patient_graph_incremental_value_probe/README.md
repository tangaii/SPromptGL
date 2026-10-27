# Experiment 46 — Patient Graph Incremental Value Probe

本实验验证固定患者关系能否在冻结 CAM-only 患者表示之上提供增量判别信息。它不设计新的 GFR，也不训练 GraphLearn、GCN、GAT、Transformer、Prompt、CLIP、GPT、PID 或 CU-GL 模块。

## 预注册问题

对每个 outer-fold，冻结 Experiment 43.4 的 `E0_ANCHORED_PHYSICAL_CAM` best checkpoint，提取实际 `OutputLayer.mlp_head` 后的 36 维 `combined_x`。同时保留原始 320 维四模态输入。每种模态从 outer-train 发送者中按正确切片和训练折归一化的余弦 Top-16 固定候选。

每个 Arm 使用同一个探针：`LayerNorm → Linear(5d,36) → GELU → Dropout(0.35) → Linear(36,2) → LogSoftmax`。SELF 将自身表示重复到四个邻居槽位；TRUE 使用真实模态邻居均值；RANDOM 使用固定种子随机训练患者；PERMUTED 保留真实邻居支持但用固定双射置换发送者。四组共享同一 Arm、fold 的逐张量初始化、优化器和训练预算。

H Arm 的输入为 180 维，X Arm 的输入为 1600 维。两者参数量不同，只在 Arm 内部进行 TRUE 相对 SELF、RANDOM、PERMUTED 的配对比较。

历史 B0-B 和 CAM-only 结果只作参考。所有新探针仍用历史 `val=test` 的 ACC 最佳 epoch 选择，同时记录固定最后 epoch；这会重复使用测试标签，结果属于探索性内部比较，不是无偏泛化评估。

## 代码审计边界

`integration.py` 只读取历史输入和 checkpoint，`PhysicalCAMOnly` 私有封装保持 49,544 个 CAM 参数和原始 forward 数值。`probe.py` 只实现轻量分类探针。`run.py` 负责冻结表示、候选图、四类邻居、训练、恢复、固定模型干预和报告。`audit.py` 独立重载 checkpoint 检查指标、冻结权重、输入哈希和候选支持。所有 mutable artifacts 位于本目录；根源码、数据和历史实验只读。

若 TRUE 没有稳定超过 SELF、RANDOM 和 PERMUTED，则结论为 `NO_GO_FIXED_NEIGHBOR_INCREMENTAL_VALUE`，暂停固定相似患者图路线，不启动 Experiment 47。
