# Experiment 50.1 — C²R-Align Corrected & Validated

本实验修复Experiment 50的关系语义错误，并在同一B0-B基座上重新验证FC锚定的选择性跨模态患者关系。

最终版本中E8（去站点关系偏置校正）在审计后从同一未训练锚点重跑十折；首次错误的无显式对齐状态不进入正式结果。

## 固定协议

- B0-B：物理移除PEN，保留原始GFR和CAM；PHENO覆盖、`ALoss`不反传、`weight_decay=0.11`均保留。
- ABIDE 871人，输入顺序为PHENO/ANAT/FUNC/CORR-FC，维度48/6/10/256。
- seed=0，10折StratifiedKFold，`shuffle=True, random_state=10`。
- `simple-2`、`val_idx=test_idx`、450 epoch、`wait>200`、原始RFE/标准化协议均保留。
- FC索引固定为3，辅助关系索引为0/1/2；候选发送者来自outer-train，K=8。
- 新分支使用embed=32、hidden=64、eta=0.25、rho=0.05、temperature=0.25、lambda_relation=0.1、lambda_align=0.03、site_lambda=1.0、lr=0.00038、weight_decay=0.11。

## 修复内容

1. FC锚点和三个辅助模态使用显式常量，所有门控差分以FC为基准。
2. 对齐损失只把辅助关系拉向停止梯度的FC关系。
3. NULL只取消辅助关系贡献；NULL=1时关系和残差路径与同参数FC-only一致。
4. E6只置乱PHENO/ANAT/FUNC，FC关系保持不变。
5. E7在重连后的邻居上重新计算四模态关系、gate、softmax和value。
6. 站点关系偏置从训练折平衡患者对估计，统计项停止梯度；E8关闭该校正。

## 执行

```bash
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B tests.py --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B tests.py --gpu 1
python -B run.py controller
python -B run.py aggregate
CUDA_VISIBLE_DEVICES=0 python -B run.py audit
python -B run.py aggregate
python -B run.py report
python -B run.py handoff
```

预检覆盖模态身份、FC-only、对齐梯度、NULL恒等、E6/E7关系路径、标签隔离、站点校正、优化器、初始化、train/eval切换和checkpoint/RNG恢复。控制器按fold分配GPU0的fold1–5和GPU1的fold6–10，每卡最多3个worker。

## 结果解释

Hard-AUC使用硬预测one-hot ROC-AUC；Probability-AUC使用TD概率与标签1计算。固定checkpoint干预只说明模型依赖路径，独立重训组才用于关系增量判断。历史`val=test`、CV前RFE/标准化、transductive输入和单seed限制仍然适用。

正式交付目录中的源码快照位于`source/`（运行scratch副本同时保留`.runtime/sources/`），B0-B锚点位于运行scratch的`.runtime/anchors/`，正式折级结果位于`results/`。80个大体积最佳checkpoint不复制到正式目录；其校验与任务状态保留在scratch manifest中，避免结果目录膨胀。
