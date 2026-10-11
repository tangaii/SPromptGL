# 实验五：SPCR 语义原型投影退化与最小修复

本实验固定实验一 P1 的 32 条 Prompt，只做 2×2 机制消融：是否保留 MLP 最后一层 ReLU，以及 MLP 参数是否使用 Adam 权重衰减。E0 是原始 P1，E1 去掉输出 ReLU，E2 只将 MLP 的 weight_decay 设为 0，E3 同时采用两项改动。

源码审计：原始 MLP 为 4096 -> 128 -> 9，两层原始均使用 ReLU；MLP 共 525577 个参数。原始 ModalFusion 使用 Adam，学习率 0.0038、weight_decay 0.11，E2/E3 将 t1/t2 的 weight 与 bias 放入唯一的 weight_decay=0 参数组，其余 ModalFusion 参数保持 0.11，MessagePassing0 不变。my_weight_init 只对 Linear weight 使用 Xavier uniform，不重置 bias。

每个 fold 在第一次反向传播前写入 INIT 快照，在恢复 ACC 最佳 checkpoint 后的测试前向写入 BEST 快照。快照记录 CLIP 输入、Linear1 前、ReLU1 后、Linear2 前和最终输出五个阶段的余弦、范数、零元素、非有限值、同向/反向高相似比例，并记录 MLP 参数范数和实际 Softmax-Max 权重。

运行与恢复：

    python experiment/experiment5_semantic_projection_collapse/run.py --prepare
    python experiment/experiment5_semantic_projection_collapse/run.py --all
    python experiment/experiment5_semantic_projection_collapse/run.py --refresh

运行器在临时隔离副本中应用 projection_patch.diff，根目录源码不改动；任务按 prompt 要求分配到两张 GPU，每个任务顺序完成十折。运行器只解析最新 attempt，已完成任务可自动跳过。
