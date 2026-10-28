# Experiment 47.3 — Patient Graph Relation-Specific Value Verification

本目录是问题发现实验，不是新的 GFR 模块。所有源码通过 `.runtime/sources`
隔离加载，根目录模型、数据和历史 checkpoint 只读。

## 冻结协议

- seed=0，原始 10 折，871 名受试者，输入 `[PHENO48, ANAT6, FUNC10, CORR256]`；
- B0-B 物理移除 PEN，保留原 GFR、两层 GCN、CAM 和 PHENO 覆盖行为；
- 有效 ANAT/FUNC/CORR GCN 的六个 W 参数 weight decay=0，其余参数按历史 0.11；
- 固定消息中心化及 `rho=0.25`，只用 outer-train 估计尺度；
- checkpoint 仍按历史 `val=test` 的 ACC 选择；
- D4/D5 只改变预注册的关系身份/消息接收对应，不搜索图、rho 或模型结构。

## 历史事实—证据—本轮问题

|历史事实|原始证据|本轮回答|
|---|---|---|
|原 GFR 的 PHENO 消息在 `i==1` 被覆盖|`.runtime/sources/network.py` 的 GFR 分支|D4/D5 保持该行为，PHENO 不作为有效跨患者分支|
|B0-B 消息几乎不进入决策|Experiment43.2、47 固定消息干预|D1–D5 统一 rho 后检验关系身份是否仍有价值|
|T1 放大消息后训练失败|Experiment47.1 T1 checkpoint 与尺度审计|本轮不把消息变大等同于关系价值|
|47.2 D2 可能扰动太弱|47.2 `graph_control_audit.csv`|本轮重新测算实际算子和消息距离|
|D4/D5 是否是有效强负控制|本目录 `strong_null_checks.csv`|10/10 折通过 25% 消息扰动门槛|

## 执行入口

```bash
PYTHONPATH=. python repair_audit.py
PYTHONPATH=. python audit.py
PYTHONPATH=. python parity_check.py
PYTHONPATH=. python train.py --group D4 --fold 1 --device cuda:0
PYTHONPATH=. python train.py --group D5 --fold 1 --device cuda:0
PYTHONPATH=. python finalize.py
```

GPU1 任务使用 `CUDA_VISIBLE_DEVICES=1` 后在进程内使用 `cuda:0`，避免原始
layers 代码中的固定 CUDA 设备假设。正式结果位于 `results/`，必要模型状态位于
`.runtime/`；`.runtime` 不作为代码交付内容。

## 结果边界

D1、D2、D3 是 47.2 历史固定图训练结果；D4/D5 是本轮新增的 20 个完整 fold。
强负控制通过后，D4 的平均 ACC 高于 D1，D5 与 D1 相近，二者均没有支持真实
患者—消息对应关系的稳定增量。因此报告将一般跨患者传播作用与关系特异性价值
分开，不把 D1 优于 D3 自动解释成疾病相关邻居有效。
