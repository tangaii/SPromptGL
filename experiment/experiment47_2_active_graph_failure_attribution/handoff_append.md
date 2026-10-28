## Experiment 47.2 — Active Patient Graph Failure Attribution

- 执行位置：`/tmp/SPromptGL-E47/experiment47_2_active_graph_failure_attribution`；Stage A 十折只读审计、D0–D3 共40/40新增训练折完成。
- GPU分配：GPU0运行fold1–5，GPU1运行fold6–10；pilot fold1/6均通过Gate A/B，按预注册规则扩展全部四组。
- B0-B与CAM-only历史checkpoint身份、T1十折结果通过核验；未修改根目录模型、数据和历史checkpoint。
- rho=0.25，仅outer-train估计中心化尺度；D2十折所有模态均保持对称、节点度、边数、自环、权重多重集和站点对边数约束。
- 正式新增结果：D0 ACC 90.4676±2.7781、Hard-AUC 90.3614±2.6960、SPE 91.6374±4.5046、SEN 89.0854±2.7628%；D1 90.6962±2.9809、90.6190±3.0133、91.6466±3.4323、89.5915±4.1041%；D2 90.4676±2.8717、90.2928±2.8866、92.7197±3.6359、87.8659±4.2934%；D3 88.3986±3.4762、88.3317±3.4310、89.0842±4.9485、87.5793±4.1998%。
- D1−D2：ACC +0.2286pp（4/4/2胜平负），Hard-AUC +0.3263pp（7/1/2），未同时达到七折门槛；D1−D3：ACC +2.2975pp（8/2/0），Hard-AUC +2.2873pp（9/0/1）。
- 机制：T1训练端本身失效；rho控制后D0/D1训练NLL显著恢复，支持尺度/优化失衡。真实固定图未稳定超过严格结构匹配重连图，当前协议下未证实独立疾病增量。
- 判定：`SCALE_OPTIMIZATION_FAILURE_VERIFIED`；`ACTIVE_GRAPH_NONDISCRIMINATIVE`；不支持继续以放大消息、取消weight decay或简单重连作为MICCAI2027第一创新点。
- 审计：`.runtime/final_audit.json` PASS；结果、源码、anchor、checkpoint、固定图和输入SHA见`manifest.json`。仍沿用历史`val=test`、全队列RFE/标准化和transductive协议。
- handoff追加状态以实际OSSFS写入为准；本实验目录保留完整待追加正文。
