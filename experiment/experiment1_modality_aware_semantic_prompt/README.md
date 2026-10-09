# 实验一：模态感知语义提示增强（MASPE）

本目录从根目录 SPromptGL 源码重新运行 P0–P3。上一版严格协议实验完整保存在 `experiment/_archive/experiment1_maspe_strict_20261009`，不得与本轮结果混用。

`run.py --prepare` 保存并核验根目录源码和 ABIDE 输入哈希，审查精确生成的 `network.py` patch，并检查提示 token 长度。P0 使用未修改的根目录源码副本；P1–P3 只对隔离副本中的 `network.py` 应用 `prompt_patch.diff`。原始 shell 参数、训练循环、10 折切分、checkpoint 选择和 hard-AUC 均来自根目录实现。

完整执行全部 12 个 config×seed 任务。运行器会把前 6 个 worker 绑定物理卡 0，把后 6 个 worker 绑定物理卡 1；每个 worker 内部仍按原始 `main.py` 顺序完成 10 折，不拆分 fold：

```bash
python experiment/experiment1_modality_aware_semantic_prompt/run.py --prepare
python experiment/experiment1_modality_aware_semantic_prompt/run.py --all
```

调度日志写入 `logs/scheduler.log`，并发 worker 不直接写汇总文件；主调度器在每个 worker 结束后更新 `results/` 与根目录 `handoff.md`。

也可以单独运行或恢复一个任务：

```bash
python experiment/experiment1_modality_aware_semantic_prompt/run.py --config P0 --seed 0
```

每个 worker 内部让原始 `main.py` 按顺序完成 10 折。只跳过有完整结束记录且日志折数校验一致的任务。运行源码副本和输出目录位于系统临时目录，任务结束后自动清理。每个任务的完整控制台输出保存在 `logs/`。

结果写入 `results/fold_results.csv`、`results/summary.csv` 和 `results/report.md`。汇总中分别记录折间标准差与 seed 间标准差。根目录 `handoff.md` 在每个 config×seed 任务结束后同步状态。
