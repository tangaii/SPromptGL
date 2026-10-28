# Experiment 47.1 — GCN Message Amplitude Collapse Attribution

本目录是一个只读复用 SPromptGL B0-B 的归因实验工作区。它不引入新的
GFR、图结构或损失，只按预注册顺序比较原始 B0-B、仅去掉有效 GCN
`W1/W2` 的 Adam weight decay（T1）以及仅去掉 `GraphConstructLoss`
（T2）。工作目录位于 `/tmp/SPromptGL-E47/...`，根目录源码、数据和历史
checkpoint 未被修改。

## 审计范围

论文与源码审计的关键事实来自 `network.py`、`layers.py`、`utils.py` 的
Experiment43.1 immutable source snapshot：`GraphConv` 先计算 `X @ W` 再
计算 `A @ support`；每路 GCN 为 `ReLU(gcn1)` 后接 `gcn2`；GFR 在
`i==1` 覆盖此前 PHENO 融合；四路 GCN 都读取完整 320 维输入；
`GraphConstructLoss` 只依赖 `parts` 与 GraphLearn 的邻接，不直接依赖
GCN 的 `W1/W2`；simple-2 的 `ALoss` 返回但不进入反向传播。B0-B 与
CAM-only 的历史身份来自 Experiment43.1/43.4，未使用 B0-A。

|历史事实|本轮原始证据|尚未由此直接推出的结论|
|---|---|---|
|ANAT/FUNC/CORR 最佳输出很小|十折逐层矩阵分解、初始/最佳 checkpoint 对照|仅凭最终输出不能区分初始化、W 收缩和图算子|
|PHENO 路存在但会被覆盖|源码 `GFR.forward` 的 `i==1` 分支|修复覆盖本身能否改善疾病分类|
|图正则很大|逐步 loss 与 GraphLearn 梯度|图正则直接压缩 GCN W（源码依赖关系不支持）|

## 运行

```bash
python -u run.py stage_a
python -u run.py merge_stage_a
python -u run.py replay
python -u run.py pilot
python -u run.py expand       # 仅在预注册机制门槛已通过时
python -u run.py audits
```

`run.py` 的命令是显式的，不会自动启动后续实验。结果表只包含实际执行
的折；T0/T2 的两折 pilot 不伪造十折均值。

## 预注册判定

T1 只有在 fold1 与 fold6 的 ANAT/FUNC/CORR 聚合有效消息相对 T0 均至少
提高 10 倍时才扩展到其余八折；T1 优先于 T2。幅值恢复、模型决策依赖和
真实跨患者关系的判别增益分别审计，不能相互替代。
