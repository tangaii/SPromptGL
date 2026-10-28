# Experiment 50 — C²R-Align

本实验验证 **Confounder-Controlled and Class-Discriminative Relational Alignment** 是否能在B0-B（物理移除PEN、保留原始GFR+CAM）上形成真实的跨患者增量。

## 固定协议

- ABIDE 871人，四模态 `[48, 6, 10, 256]`，ASD=0、TD=1。
- seed=0，StratifiedKFold 10折，`shuffle=True, random_state=10`。
- 原始`simple-2`、`val_idx=test_idx`、450 epoch、`wait>200`、`lr=0.0038`、`weight_decay=0.11`、dropout=0.35、GraphLearn阈值0.9和四图GraphConstructLoss均保留。
- B0-B的`ALoss`仍只计算、不加入反传；新关系损失只属于C²R-Align。
- 新模块默认`embed=32, K=8, rho=0.05, eta=0.25, lambda_rel=0.1, lambda_align=0.03, lr=0.00038, weight_decay=0.11`，在正式训练前冻结。

## 运行

```bash
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B run.py preflight --gpu 1
python -B run.py controller
python -B run.py aggregate
python -B run.py audit
python -B run.py aggregate   # preserve audited E0 probability-AUC
python -B run.py report
python -B run.py handoff
```

`controller`按fold把所有新增组的fold1–5交给GPU0、fold6–10交给GPU1，并支持已完成任务跳过；每卡并发上限固定为3。训练在`.runtime`写入唯一best checkpoint和任务状态；结果在`results/`。如果OSSFS写入缓慢，可将整个实验目录复制到本地scratch执行，再同步非`.runtime`交付文件并在manifest记录。

## 组别

E0复用经独立读回的Experiment43.1 B0-B结果。E1只用自身受限残差；E2只用CORR/FC关系；E3使用四模态均匀辅助关系且无NULL；E4使用条件门控、疾病关系监督和混杂特征但关闭显式对齐损失；E5为完整C²R-Align；E6固定站点内辅助关系错配；E7固定站点内患者消息关系重连。E6/E7不使用测试标签构建置乱。

## 独立审计

`audit.py`重新加载最佳checkpoint，重算七项指标、混淆计数、C²R残差及E5固定模型干预；检查候选发送者均来自outer-train、监督pair均来自outer-train、E1无跨患者消息、NULL/门控权重和为1、根源码保护SHA不变。Hard-AUC是硬预测one-hot ROC-AUC，Probability-AUC使用ASD连续概率，二者分开报告。

历史`val=test`选模、CV前RFE/标准化和transductive全队列输入保留，因此结果是协议内比较，不是无偏独立泛化或医学因果证据。
