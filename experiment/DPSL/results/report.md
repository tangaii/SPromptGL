# DPSL 独立实验基座验收

**COMPLETE_AUDITED_STANDALONE**

权威目录为 `/model/spromptgl/experiment/DPSL`。E82.4.6.10-G1 的外部实验身份和 G2 SCMR-DA-SRPE 的内部模型分派均已保留。Fold 8 正式来源使用患者 383、430。

## 正式十折来源及真实前向

| Fold | 正式来源 | Best epoch | Latest epoch | 正确 / 测试 |
|---:|---|---:|---:|---:|
| 1 | E82.4.6-G2 | 102 | 303 | 84 / 88 |
| 2 | E82.4.6.4-G1 | 166 | 367 | 84 / 87 |
| 3 | E82.4.6.3-G2 | 244 | 445 | 79 / 87 |
| 4 | E82.4.6.6-G1 | 182 | 383 | 83 / 87 |
| 5 | E82.4.6-G2 | 428 | 450 | 81 / 87 |
| 6 | E82.4.6-G2 | 157 | 358 | 84 / 87 |
| 7 | E82.4.6-G2 | 166 | 367 | 79 / 87 |
| 8 | E82.4.6.10-G1 | 201 | 402 | 78 / 87 |
| 9 | E82.4.6-G2 | 231 | 432 | 78 / 87 |
| 10 | E82.4.6-G2 | 152 | 353 | 81 / 87 |

各折新路径为 `checkpoints/foldXX.best.pt` 与 `checkpoints/foldXX.latest.pt`。原始绝对路径、SHA-256 和 epoch 见 [manifest.json](../manifest.json)。

实际前向 10/10 折通过。871 名患者的顺序、标签、硬预测、两类 logits、ASD/TD 概率及真类 Margin 与历史证据一致，最大绝对误差为 0。完整 CAM144、增强特征、谱内外分量及总残差与原历史代码前向的张量摘要逐折相同。原 GPU UUID 已映射至当前物理 GPU0。Fold 4 保留原核心和原评价实现，历史目录中后续修订的审计包装器未参与前向。

| 指标 | 十折均值 ± population SD，% | Pooled，% |
|---|---:|---:|
| ACC | 93.1087 ± 2.6257 | 93.1114 |
| Hard-AUC / BA | 93.0348 ± 2.6450 | 93.0383 |
| Prob-AUC | 94.8628 ± 2.8903 | 94.2053 |
| SEN-ASD | 92.0549 ± 3.6745 | 92.0596 |
| SPE-TD | 94.0148 ± 3.4314 | 94.0171 |
| ASD-F1 | 92.5096 ± 2.8797 | 92.5187 |

Pooled 混淆矩阵按 ASD 为阳性计算，TP=371、FP=28、TN=440、FN=32，共 811/871 正确。Hard-AUC 取硬预测的 balanced accuracy，Prob-AUC 根据连续 ASD 概率计算。

## 恢复与训练入口

正式 20 份 Best/Latest 按来源 Manifest 逐份核验字节 SHA。10 折 Latest 实际恢复模型、四个优化器、Python/NumPy/CPU/设备 RNG、Gaussian/EMA 和早停计数。同轮评估以及下一步训练加评估的归档见证逐折精确匹配。历史正式 Latest 已达到停止条件。

10 折从各自 E61 epoch 10 恢复，实际执行 epoch 11、保存 Best 和 Latest、从存盘 Latest 恢复并执行 epoch 12。两轮训练及评估日志与历史记录逐折一致。历史顺序有 6 次 Dropout，训练轮次未重设种子。审计过程的临时训练文件已清理。完整十折训练未重复执行。

正式训练入口提供单折、十折和运行目录续训。共用差分梯度预算为 0.1%，新增 CAM144 残差限幅为 8%。19 名重叠患者保留 DARG 分类和关系资格，新增谱及 Prompt 标签监督、Gaussian/EMA 更新均使用历史合法成员，重叠患者的新增残差各分量严格为零。

## 完整性和边界

在 `/tmp` 启动 CLI，Python 导入和文件审计与 strace 的 286,943 次文件系统调用均未发现外部项目访问或正式权重和输入的写入。迁移前后 1,423 份历史来源 SHA-256 相同。缺少字段、优化器损坏、SHA 错误、患者顺序改变、非法监督、非零重叠残差、外部路径与缺少 RNG 的负向审计全部正确判为失败。

19 名有意保留的训练—测试标签重叠、历史 `val=test`、CV 前 RFE/预处理及转导患者图限制了独立泛化推断。跨其他设备或数值库的逐位复现尚未验证。公开 GitHub main 为旧快照 `bb6b010`，受控制品恢复指令见 [README.md](../README.md)。

[完整审计](audit_status.json) · [逐折指标](fold_metrics_real.csv) · [逐患者前向](evaluation_real.csv) · [Latest 恢复](resume_check.json) · [Warmup 训练](warmup_step_check.json) · [完整文件 SHA](../SHA256SUMS)
