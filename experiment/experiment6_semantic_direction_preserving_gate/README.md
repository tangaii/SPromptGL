# 实验六：SDPG 语义方向保持门控

固定实验一 P1 的 32 条 Prompt 和原始 Softmax-Max 门控，仅增加 B1 signed direction (`d=p0-p1`) 与 B2 magnitude-only (`abs(d)`)。B0 使用实验五 E0 的冻结 30 折，并逐折与实验一 P1 核对。B1/B2 各运行 seed 0/1/2 的原始十折协议。每个模型只新增 4 个 theta 参数，theta 初值为 0，rho=.5*tanh(theta)，theta 不使用 weight decay。

每个 BEST 测试前向写入四模态门控诊断，共 240 行。六个任务在两张 GPU 上各并行三个任务。

    python experiment/experiment6_semantic_direction_preserving_gate/run.py --prepare
    python experiment/experiment6_semantic_direction_preserving_gate/run.py --all
    python experiment/experiment6_semantic_direction_preserving_gate/run.py --refresh

根源码只在临时副本中打补丁；此实验不执行 git commit 或 git push。
