# 实验二：对比状态驱动的模态语义提示优化（CSMP）

本实验只研究固定 Prompt 文本是否改善 SPromptGL 的模态语义增强效果。V2 使用预先固定的 32 条状态对比 Prompt；V2S 完全复用这 32 条文本，只固定错配模态：PHENO←FUNC、FUNC←PHENO、ANAT←Correlation、Correlation←ANAT。不得根据结果生成 V3 或修改训练协议。

运行基座是根目录原始 SPromptGL。审计确认：GFR 在 `model.py` 中执行逐模态图特征传播，PLN/CAM 在 `network.py` 和 `layers.py` 中完成模态输入、注意力及分类融合；原始局部 Token 为 `[871,4,4,9]`。实验 patch 只向隔离副本的 `network.py` 增加 JSON Prompt 加载、V2/V2S 配置选择、32 条 RN50 文本编码、`[4,2,9]` 共享 `4096→128→9` 原型和对应的模态语义相似度计算，保留余弦、Softmax、Max、局部加权和后续融合公式。根目录源码、数据和 RN50 哈希见 `source_hashes.json`。

原始训练协议保持不变：`StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`，`val_idx == tst_idx`，ACC 选择 checkpoint，450 epoch 上限，200 patience，原始优化器/损失/初始化/transductive 传播和 hard-label AUC。参数来自 `ABIDE-simple-2-concat-weighted-cosine.sh`。实验一 P0–P3 的 120 个逐折结果只读复制到 `results/reference_fold_results.csv`，并记录源文件哈希。

准备并执行：

```bash
python experiment/experiment2_contrastive_state_prompts/run.py --prepare
python experiment/experiment2_contrastive_state_prompts/run.py --all
```

`--prepare` 会校验 32 条文本和 CLIP token 长度，生成并精确应用 `prompt_patch.diff`，完成一次不更新参数的 V2/V2S 前向检查，并保存 V1/V2 主题嵌入诊断。`--all` 启动 6 个独立任务：GPU0 运行 V2 的三个 seed，GPU1 运行 V2S 的三个 seed；每个任务内部按原始顺序完成 10 折。任务日志在 `logs/`，新结果在 `results/fold_results.csv`、`results/summary.csv` 和 `results/report.md`。主调度器独占更新根目录 `handoff.md`，任务可安全恢复。
