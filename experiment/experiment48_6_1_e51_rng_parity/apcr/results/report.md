# Experiment 48.6.1 — 修正随机轨迹并复现实验51 E1

## 执行状态

- R0、E1、E2共30折全部完成；R0 10折原始实验51 E1重训，E1/E2各10折APCR重训。
- 单GPU、5个逻辑worker；20个APCR checkpoint独立读回通过，R0独立读回通过。
- `/data`仍返回`Errno 5 Input/output error`，完整结果位于`/tmp/SPromptGL-E48.6.1`；正式目录和handoff暂为待同步。

## 源码差异与随机流修正

- R0直接使用实验51 E1冻结runner，未加入APCR、未改模型、优化器、checkpoint规则或随机数逻辑。
- 48.6.1删除48.6原有的每epoch `torch.manual_seed(481400000 + fold*10000 + epoch)`和CUDA重置。E1/E2从实验51 E1的`seed_all(510000 + fold*100 + 1)`初始化后连续推进主RNG。
- APCR初始化使用独立fold种子并在初始化后恢复基座Python/NumPy/Torch CPU/CUDA RNG；APCR额外前向通过独立连续RNG上下文执行，checkpoint同时保存主RNG和APCR RNG。
- SRD/SCA/CRA算法、group=6路径、损失权重、学习率、weight decay、候选数量、攻击预算均未改变。

## R0：实验51 E1严格复现

| 指标 | 历史实验51 E1 | R0重训 | 逐折结果 |
|---|---:|---:|---|
| ACC | 91.9592 | 91.9592 ± 2.8684 | 完全一致 |
| Hard-AUC | 91.7434 | 91.7434 ± 2.8950 | 完全一致 |
| SPE-TD | 94.6392 | 94.6392 ± 3.3856 | 完全一致 |
| SEN-ASD | 88.8476 | 88.8476 ± 4.1651 | 完全一致 |
| Probability-AUC | 93.1856 | 93.1856 ± 3.0236 | 完全一致 |

- 10/10折ACC、Hard-AUC、SPE、SEN、Probability-AUC逐值一致。
- 10/10折best epoch与执行epoch一致。
- 初始化参数SHA和初始化RNG SHA一致。
- 历史E51 task记录没有保存预测SHA，因此预测SHA历史对照为`NOT_VERIFIABLE`；本次R0独立预测SHA已保存。

## 正式结果

| 模型 | α最佳均值 | ACC | Hard-AUC | SPE-TD | SEN-ASD | Probability-AUC |
|---|---:|---:|---:|---:|---:|---:|
| R0 实验51 E1重训 | — | 91.9592 ± 2.8684 | 91.7434 ± 2.8950 | 94.6392 ± 3.3856 | 88.8476 ± 4.1651 | 93.1856 ± 3.0236 |
| E1 固定α | 0.0022000 | 91.8443 ± 2.8892 | 91.6370 ± 2.8941 | 94.4265 ± 3.5078 | 88.8476 ± 3.8533 | 93.7062 ± 2.5966 |
| E2 可学习α | 0.0024182 ± 0.0000896 | 91.8443 ± 2.9346 | 91.6370 ± 2.9135 | 94.4265 ± 3.2394 | 88.8476 ± 2.9322 | 93.1482 ± 3.2693 |

## 配对比较

| 比较 | ΔACC | ΔHard-AUC | ΔSPE | ΔSEN | ΔProb-AUC | ACC胜/平/负 | Hard-AUC胜/平/负 |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1_vs_R0 | -0.1149 | -0.1064 | -0.2128 | +0.0000 | +0.5205 | 1/7/2 | 1/7/2 |
| E2_vs_R0 | -0.1149 | -0.1064 | -0.2128 | +0.0000 | -0.0375 | 1/7/2 | 1/6/3 |
| E2_vs_E1 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | -0.5580 | 1/8/1 | 2/6/2 |

相对旧48.6：E1 ACC +0.2299pp、Hard-AUC +0.2235pp；E2 ACC +0.6897pp、Hard-AUC +0.7523pp。两组均超过旧48.6对应结果，但这是随机轨迹修正后的重训差异，不能归因于新增算法。

## α与APCR关闭干预

- E1最佳α均为0.0022000005，固定不变；E2最佳α均值为0.0024182，范围0.0023174–0.0026588。
- E2相对E1的硬预测指标完全相同或仅浮点等价：ACC、Hard-AUC、SPE、SEN均无实质差异；E2 Probability-AUC低0.5580pp。
- α=0干预：E1 ACC平均下降0.2299pp，E2下降0.1149pp；有效APCR残差/GFR比约E1 0.0003135、E2 0.0003923。

RNG隔离单元测试：
- `base_rng_unchanged_by_apcr_init = True`
- `apcr_scope_preserves_main_rng = True`
- `apcr_scope_advances_independent_rng = True`

## 结论

1. **实验51 E1可严格复现。** R0逐折主要指标和best epoch均一致；历史预测SHA缺失，因此该项只能标记不可核验。
2. **48.6的epoch级随机重置已修复。** 主RNG连续推进，APCR初始化和额外前向不污染主RNG，且状态可恢复。
3. **E1/E2均未超过本次R0。** 两组ACC均91.8443%，低于R0 91.9592%约0.1149pp。
4. **固定与可学习α的硬分类结果相同。** E2没有带来可检测的ACC/Hard-AUC/SPE/SEN收益，且Probability-AUC较低。
5. **APCR没有提高修复版GFR的诊断性能。** 与R0相比两组主指标均未提升；相对旧48.6的提升来自随机轨迹修正，不能解释为算法增益。

## 判定

- `R0_REPRODUCTION_STATUS = PASS_METRIC_AND_BEST_EPOCH; PREDICTION_SHA_NOT_VERIFIABLE`
- `RNG_ALIGNMENT_STATUS = PASS`
- `TRAINING_STATUS = 30/30_COMPLETE`
- `CHECKPOINT_AUDIT_STATUS = PASS`
- `E1_OVER_E51 = NO`
- `E2_OVER_E51 = NO`
- `SCIENTIFIC_STATUS = NO-GO_FOR_APCR_GAIN`
- `DELIVERY_STATUS = DELIVERY_PENDING_DATA_EIO`

## 限制与产物

- 保留历史CV前全局RFE/预处理、transductive图、`val=test` checkpoint选择和单seed。
- 结果文件：`results/fold_metrics.csv`、`results/summary.csv`、`results/paired_comparisons.csv`、`results/apcr_off_interventions.csv`。
