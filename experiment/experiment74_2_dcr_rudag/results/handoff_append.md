
## Experiment 74.2 — DCR-RUDAG

- 状态：`COMPLETE_AUDITED`；10/10正式折完成，10/10独立checkpoint审计通过。科学结论：`PARTIAL`。
- 目的：将E74.1普通邻居差分替换成只由疾病边权增量驱动的有界残差，检验疾病图变化是否比原始相似性邻居聚合更直接地影响患者表示与ASD分类。
- 实际目录：`/model/spromptgl/experiment/experiment74_2_dcr_rudag`。偏好的`/data/adodas/SPromptGL-main/experiment/experiment74_2_dcr_rudag`创建失败，OSSFS返回`Input/output error`（errno 5），因此使用`/model`回退目录。
- 源码版本：SPromptGL根目录核心代码在本地HEAD `fa1570f45dea5c3027a89ccc93aef3060657bacd`，`main.py/model.py/network.py/layers.py/utils.py`的SHA与GitHub `main` commit `47b992ad6e11c58b15853cecb03384c0bff372a4`逐文件一致。E74.1分支`upload/experiment74-1-rudag`固定commit `d1522a23682b62e41d899d95c9721a68d3828456`，实验代码和参考结果SHA一致。BrainPrompt代码固定commit `c85304a023636a2701d465e4cb0acb13b637127e`；SABER仅审阅arXiv v1论文，没有确认官方源码。
- 仅有一个正式组`E74.2-G1-DCR-RUDAG`，10折从E61 epoch-10 warmup独立恢复；没有从E74.1最佳checkpoint继续训练。唯一模型方法差异为`DeltaA=A_unified-S`上的有界差分残差`R=(DeltaA_off@X-row_sum(DeltaA_off)*X)/clamp_min(row_sum(S_off),1)`，继承E53.3 CADR的48/6/10/256 RMS-tanh限制，并使用固定`0.02*g`进入原320维GFR输出。一个统一图、每模态一次GCN、原CAM和144维分类器保持不变；没有新增参数、损失、Prompt或语义分支。
- 训练资源：单卡PPU-ZW810E，最多10个并行worker，正式训练墙钟约697秒，最大观测显存约11.7 GiB/98 GiB，最终训练批次重试0、OOM 0。最初代码曾启动20次worker调用并在optimizer step前因诊断重复反传而失败；没有完成epoch更新或写入checkpoint。修复后从同一10份E61 warmup重新运行，10/10折成功。审计脚本在训练后为保留DCR-only梯度图而修复；训练脚本、模型代码、配置、数据、warmup和历史参考SHA均与通过预检时一致。训练源SHA为`50dcd49f02c441a1e91ca2651873cbc07af270c16bca06044702b599155d3999`，最终独立审计脚本SHA为`6ae53ad1174b9cf7e7f3a4fa19b9f4bc47d4a311988161c9df77a0c333f019fc`。

十折分类结果为百分比`mean ± population SD`：

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|---:|
| E69.3-G2历史 | 92.0742 ± 2.7482 | 91.8995 ± 2.7186 | 93.3759 ± 2.9908 | 89.5854 ± 3.4710 | 94.2137 ± 3.9815 |
| E62-E2历史 | 91.8443 ± 2.9346 | 91.6821 ± 2.8919 | 93.2010 ± 3.0942 | 89.5854 ± 3.4710 | 93.7789 ± 4.2653 |
| E74-UDAG历史 | 91.4995 ± 2.6885 | 91.2511 ± 2.7308 | 93.9249 ± 2.6632 | 87.8537 ± 4.3359 | 94.6485 ± 3.2152 |
| E74.1-G1 RUDAG-U历史 | 91.9592 ± 2.7747 | 91.7498 ± 2.7845 | 93.4439 ± 3.0255 | 89.0732 ± 3.9177 | 94.4265 ± 3.7676 |
| E74.1-G2 RUDAG-Full历史 | 91.8443 ± 2.9793 | 91.6363 ± 3.0151 | 93.5999 ± 3.0799 | 88.8415 ± 4.3351 | 94.4311 ± 3.4953 |
| E74.2-G1 DCR-RUDAG | 91.8443 ± 2.7963 | 91.6442 ± 2.7336 | 93.6854 ± 2.7687 | 89.0793 ± 3.0035 | 94.2091 ± 4.3328 |

- 逐折配对差值按`ACC/Hard-AUC/Prob-AUC/SEN/SPE`，单位pp；历史对照均为描述性比较：
  - E74.2−E74.1-G1：全10折`−0.1149/−0.1056/+0.2415/+0.0061/−0.2174`；去Fold 2后`+0.1277/+0.1536/+0.3390/+0.5488/−0.2415`。
  - E74.2−E74.1-G2（普通邻居差分）：全10折`0.0000/+0.0079/+0.0855/+0.2378/−0.2220`；去Fold 2后`0.0000/+0.0088/+0.0773/+0.2642/−0.2467`。ACC无净差，固定Direct-off也没有硬预测变化，因此DCR残差未证明优于普通邻居差分。
  - E74.2−E74：全10折`+0.3448/+0.3931/−0.2395/+1.2256/−0.4394`；去Fold 2后`+0.2554/+0.3013/−0.2072/+1.0908/−0.4882`。
  - E74.2−E69.3-G2：全10折`−0.2299/−0.2554/+0.3095/−0.5061/−0.0046`；去Fold 2后`−0.2554/−0.2837/+0.3203/−0.5623/−0.0051`。
  - E74.2−E62-E2：全10折`0.0000/−0.0380/+0.4844/−0.5061/+0.4302`；去Fold 2后`−0.2554/−0.2837/+0.3203/−0.5623/−0.0051`。
- Fold 2本折ACC/Hard-AUC/Prob-AUC/SEN/SPE为`90.8046/90.6416/92.5769/87.8049/93.4783%`；1-based row 377（ASD）和407（TD）同时在训练和测试索引中。去掉Fold 2后E74.2为ACC`91.9598±2.9248%`、Hard-AUC`91.7556±2.8598%`、Prob-AUC`93.8086±2.8923%`、SEN`89.2209±3.1341%`、SPE`94.2903±4.5600%`。九折仍使用`val=test`、预交叉验证RFE/标准化和871节点transductive图，不能作为无偏独立泛化估计。
- CORR平均47,625.9/47,625.9条原有效边均被评分并修改，`||DeltaA||/||S||=0.6515`，门控从0.9900到0.9920；关系训练/未暴露测试AUC为0.9948/0.8326。PHENO/ANAT/FUNC的非CORR弱门控终值约0.001004，平均评分约3.6k个有效Top-12候选；三者未暴露测试关系AUC为0.501–0.503，`||DeltaA||/||S||`约`1.18e-5–5.14e-5`。
- CORR有界残差范数为输入范数的13.2682%；乘固定系数、门控与原融合权重后为0.156809%。同模态融合GCN消息约0.000701%，DCR消息约为其229倍；完整融合特征与输入的净差约0.160427%。这改变了患者表示，但不能单独证明分类贡献。DCR-only分类梯度在CORR评分器为`3.83e-4`，其余模态约`2.9e-11–3.8e-11`。
- 固定checkpoint干预均为10折同一模型、无重训：Disease-neutral、CORR-neutral、Score-permuted、Direct-off的硬翻转数均为0，纠正/新错均为0。full−干预ACC、Hard-AUC、SEN、SPE均为0；full−干预Prob-AUC分别为`−0.0267/−0.0267/−0.0266/−0.0267 pp`。平均ASD概率绝对变化分别为`0.001366/0.001366/0.001393/0.001366`，融合表示相对变化为`0.001568/0.001568/0.001962/0.001568`。Disease-neutral和置换没有硬分类影响；Direct-off下Prob-AUC略高0.0267 pp，未显示DCR残差的净分类收益。
- 审计：10/10 best checkpoint独立重建，通过标签/索引/Fold 2暴露/训练pair作用域/模型来源与SHA/best epoch检查；10/10预测硬标签完全一致，概率在`1e-8`误差内；每折每模态一次GCN。E74.1-G1初始化身份10/10完全匹配；共有epoch RNG三元组`3530/3530`完全一致。预检验证`q=0.5`时DeltaA及DCR残差严格为零、Direct-off与E74.1-G1前向逐值一致、残差梯度/上下界、单步checkpoint恢复与下一步更新复现。
- 结论：`PARTIAL`。相对E74部分恢复ACC/Hard-AUC/SEN，且排除Fold 2后略高于E74.1-G1；但全10折ACC低于E74.1-G1，未达到E69.3-G2 ACC/Hard-AUC/SEN目标，疾病中性化/评分置换无硬预测翻转，DCR残差对Direct-off无硬预测净收益。因此不能声称疾病关系对应或新残差带来可靠诊断增益。
- 下一步：不调0.8边权强度、0.02残差系数、门控、WD、学习率或Top-12。优先在fold-local特征选择/标准化、独立验证集选checkpoint的无泄漏协议下评估CORR关系得分是否对连续决策概率和硬分类都有稳定贡献。
- 主要文件：`/model/spromptgl/experiment/experiment74_2_dcr_rudag/results/report.md`、`manifest.json`、`results/fold_metrics.csv`、`summary.csv`、`paired_comparisons.csv`、`mechanism.csv`、`interventions.csv`、`checkpoint_audit.csv`、`audit_status.json`。已移除10个latest恢复checkpoint和生成字节码缓存；保留10个best checkpoint、任务记录及审计日志。

<!-- EXPERIMENT74_2_DCR_RUDAG_2026-11-06 -->
