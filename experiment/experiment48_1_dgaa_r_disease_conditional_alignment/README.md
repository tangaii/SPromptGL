# Experiment 48.1 — DGAA-R

本目录是隔离的 Experiment 48.1 正式运行目录。唯一正式基座是物理移除 PEN 的 B0-B；E0 读取并核验 Experiment43.1 的十折最佳检查点，E1–E7 从同折 B0-B 初始化锚点重新训练。

## 固定协议

- ABIDE A，871例（ASD=403、TD=468），PHENO/ANAT/FUNC/CORR=48/6/10/256，总输入320维。
- seed=0；10-fold `StratifiedKFold(shuffle=True, random_state=10)`；沿用历史 `val_idx=test_idx` 和首次严格提升 ACC 的 checkpoint 规则。
- 原始 GFR、CAM、两个 Adam、`lr=0.0038`、`weight_decay=0.11`、450轮上限、`wait>200`、原图正则和历史 PHENO 覆盖行为保持不变；原始 ALoss 不加入反向传播。
- 新增共享 `320→128→64` 投影、共享疾病分类头和三组关系来源判别器；新增模块使用 `lr=0.00038, weight_decay=0`。
- 条件对抗只在outer-train标签分组内运行；E7只置乱条件分组标签。E6只按站点及train/test分层错配传播邻接，GraphConstructLoss仍使用原图。

## 组别

`E0 B0B_REFERENCE`、`E1 AUX_ONLY`、`E2 UNCONDITIONAL_DGAA`、`E3 DISEASE_CONDITIONAL_DGAA`、`E4 SELECTIVE_CONDITIONAL_DGAA`、`E5 CORR_PRIORITY_DGAA_R`、`E6 SITE_STRATIFIED_RELATION_NULL`、`E7 PERMUTED_DISEASE_CONDITION`。所有组均为10折，完成后由 `audit.py` 独立加载最佳模型重算四项指标。

## 复现

```bash
python -B run.py init
python -B run.py preflight
CUDA_VISIBLE_DEVICES=0 python -B run.py worker --gpu 0 --folds 1,3,5
CUDA_VISIBLE_DEVICES=0 python -B run.py worker --gpu 0 --folds 2,4
CUDA_VISIBLE_DEVICES=1 python -B run.py worker --gpu 1 --folds 6,8,10
CUDA_VISIBLE_DEVICES=1 python -B run.py worker --gpu 1 --folds 7,9
python -B audit.py
```

`.runtime/` 只保存锚点、恢复checkpoint、任务状态、源快照和日志。正式报告、逐折指标、汇总、配对差和机制诊断位于 `results/`。当前历史评估协议包含全局预处理/CV前RFE、871节点transductive图、`val=test` 和单seed，结论应限定在该协议内。
