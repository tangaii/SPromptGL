# Experiment 83.1

四组在同一套 TADPOLE 特征、RID 顺序和十折上执行。工作目录因首选 OSS 路径返回 errno 5 而使用 `/model/spromptgl/experiment/experiment83_1_tadpole_fidelity_repair`。E83 原目录保持只读。`unaccepted_pre_repair/` 保存门禁生效前的探索产物，正式结果生成器只读带正式 release 身份的四组目录。

保真矩阵记录归档源码、E83 差异、数值验证和三分类适配。执行门禁检查 `audits/preflight.json`、冻结的源码与 `protocol.json` 一致。正式训练使用一张 98304 MiB GPU 和 8 个 fold worker，每 worker 各用一个 intra-op 与 inter-op CPU 线程。ACC 与硬预测 one-hot macro OVR AUC 是正式指标。

在此目录执行或恢复正式实验。已完成的折通过独立重放后会跳过。可以用 `E831_FOLDS=单个或逗号分隔折号` 配合指定组重试失败的折。

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python common/run_e831.py all --workers 8
python common/replay_check.py
python common/report.py
```

保真预检脚本是 `common/preflight.py`。冻结版本的预检产物含执行时间和设备测量值，重新运行会更新产物哈希。复验时保留冻结产物与协议的匹配，先确认正式执行已结束。审计、逐折产物、结果和报告分别位于 `audits/`、四组子目录、`results/` 和 `final_report.md`。
