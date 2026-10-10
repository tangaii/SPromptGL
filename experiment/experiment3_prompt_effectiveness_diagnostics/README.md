# 实验三：语义提示有效性与原型塌缩诊断（PEPCA）

本实验诊断原始 SPromptGL Prompt 分支在 ABIDE 上的实际作用，比较原始医学 Prompt、实验一模态 Prompt、固定 0.5/1.0 缩放和固定无关文本负对照。运行目录独立于实验一、实验二；根目录源码不修改，任务在临时隔离副本中执行。

## 源码审计结论

GitHub `main` 提交为 `79b3a62`，与本地 `main.py`、`model.py`、`network.py`、`layers.py`、`utils.py` 的 SHA-256 一致。原始 `network.py` 用 CLIP RN50 编码 8 条文本（8×1024），前四条和后四条分别拼接成 4096 维，经共享 4096→128→9 MLP 得到两个 9 维原型。PLN 的局部 Token 形状为 `[871,4,4,9]`；对 Token 与两个原型计算余弦相似度，沿原型维做 Softmax 后取 Max，再将标量权重乘回局部 Token，随后与 AvgPool 得到的 global 表示进行局部融合。若相似度为 `s1,s2`，原始权重满足 `max(softmax([s1,s2])) = sigmoid(abs(s1-s2))`，理论范围为 0.5 到 `sigmoid(2)=0.8808`；实际分布由本实验记录。

原始训练协议全部保留：`StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`、`val_idx == tst_idx`、ACC 选择 checkpoint、450 epoch、patience 200、原始损失/Adam/初始化、transductive 图传播和 hard-label AUC。测试折参与 checkpoint 选择，因此报告使用 Legacy 评价口径，不称为独立测试集性能。

## 配置

- **D0**：根目录原始 8 条医学 Prompt；共享两个原型，复现实验一 P0。
- **D1**：逐字冻结实验一 P1 的 32 条模态专属 Prompt，形状为 4 模态×2 主题×4 文本，复现实验一 P1。
- **D2**：原始骨干，最终局部语义缩放固定为 0.5；MLP 保留注册和初始化，但该分支不获得有效梯度。
- **D3**：原始骨干，最终局部语义缩放固定为 1.0；后续融合不变。
- **D4**：原始 8 条文本的组织和算法不变，仅使用 `controls.json` 中预先固定的 8 条无关英文文本。

D0、D1、D4 的诊断在最佳 checkpoint 加载后、最终 `print_tst_acc()` 前向中计算，并按当前测试折索引统计。每个 seed/fold/modality 保存原型余弦、原始 Token 相似度、Softmax-Max 权重分位数/阈值比例、范数、零向量/非有限值和模态差异。D2/D3 的语义字段写为 N/A，不伪造统计。

## 执行与恢复

准备并校验冻结输入：

```bash
python experiment/experiment3_prompt_effectiveness_diagnostics/run.py --prepare
```

完整执行会先完成 D0 seed0、D1 seed0 parity，再启动所有 config×seed 任务。需要在已有任务上动态补位时使用：

```bash
python experiment/experiment3_prompt_effectiveness_diagnostics/run.py --pending
```

单任务恢复命令示例：

```bash
python experiment/experiment3_prompt_effectiveness_diagnostics/run.py --config D2 --seed 0 --gpu 0
```

只刷新汇总、不启动任务：

```bash
python experiment/experiment3_prompt_effectiveness_diagnostics/run.py --refresh
```

任务日志在 `logs/`；临时诊断 JSONL 和梯度摘要在隐藏的 `logs/.working/`，用于汇总，不保存 epoch 级 checkpoint。最终产物为 `controls.json`、`prompt_patch.diff`、`provenance.json`、`results/reference.csv`、`results/fold_metrics.csv`、`results/prompt_diagnostics.csv` 和 `results/report.md`。运行器会原子更新根目录 `handoff.md`，不自动提交或推送 GitHub。

