# Experiment 76.9 RBCD 独立审计报告

- STATUS：`COMPLETE_AUDITED`；科研判定：`RISK_DETECTION_WITHOUT_CORRECTION`。
- 目录：`/model/spromptgl/experiment/experiment76_9_rbcd`；首选目录写入/原子重命名探针因OSSFS errno=5失败，使用`/model`回退。
- 正式训练：20/20折，5个worker，wall time 1286.17s；任务记录最大单worker显存 1521.5 MiB。
- 独立复算：best 20/20，latest恢复 20/20；OOF 10/10。

## 六项指标（mean ± population SD，%）

| 方法 | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 | 92.1891±2.7648 | 92.0294±2.7255 | 93.4946±2.9195 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E76.8-G1 | 92.0742±2.7482 | 91.9656±2.6826 | 93.6538±2.8990 | 90.5732±3.3075 | 93.3580±4.5681 | 91.3989±2.8583 |
| E76.8-G2 | 92.0742±2.6503 | 91.8725±2.6410 | 93.3301±2.8091 | 89.3232±3.8890 | 94.4218±4.1374 | 91.2634±2.8635 |
| E76.9-G1 RBCD-StopGrad | 92.1891±2.7648 | 92.0294±2.7255 | 93.4627±2.9423 | 90.0671±3.8862 | 93.9917±4.7320 | 91.4590±2.9013 |
| E76.9-G2 Protected-E2E | 91.4995±2.5368 | 91.2559±2.5065 | 93.6905±2.6673 | 88.0854±3.5077 | 94.4265±4.0007 | 90.5752±2.7188 |

## 配对差（pp；均值、W/T/L；描述性95%区间见CSV）

**A_G1_minus_E74.4-G2**
- all10: acc +0.0000 (0/10/0); hard_auc +0.0000 (0/10/0); prob_auc -0.0319 (1/4/5); sen +0.0000 (0/10/0); spe +0.0000 (0/10/0); asd_f1 +0.0000 (0/10/0)
- exclude_fold2: acc +0.0000 (0/9/0); hard_auc +0.0000 (0/9/0); prob_auc -0.0236 (1/4/4); sen +0.0000 (0/9/0); spe +0.0000 (0/9/0); asd_f1 +0.0000 (0/9/0)
**B_G2_minus_G1**
- all10: acc -0.6897 (0/4/6); hard_auc -0.7735 (0/3/7); prob_auc +0.2278 (7/0/3); sen -1.9817 (0/4/6); spe +0.4348 (2/7/1); asd_f1 -0.8839 (0/3/7)
- exclude_fold2: acc -0.6386 (0/4/5); hard_auc -0.7239 (0/3/6); prob_auc +0.1942 (6/0/3); sen -1.9309 (0/4/5); spe +0.4831 (2/6/1); asd_f1 -0.8312 (0/3/6)
**C_G2_minus_E74.4-G2**
- all10: acc -0.6897 (0/4/6); hard_auc -0.7735 (0/3/7); prob_auc +0.1959 (7/0/3); sen -1.9817 (0/4/6); spe +0.4348 (2/7/1); asd_f1 -0.8839 (0/3/7)
- exclude_fold2: acc -0.6386 (0/4/5); hard_auc -0.7239 (0/3/6); prob_auc +0.1706 (6/0/3); sen -1.9309 (0/4/5); spe +0.4831 (2/6/1); asd_f1 -0.8312 (0/3/6)

## OOF与风险/纠错机制

- 每个外层fold对eligible_train_ids做固定5折StratifiedKFold（random_state=1000+fold），每名患者恰好一次留出预测；教师从原始shared anchor/DARG构造开始，固定250轮，不读E61 epoch-10 warmup，不做内层选模。Fold 2的376、406 zero-based重复行未进入新增监督。
- 风险AUROC/AUPRC均值：G1 `0.5915 / 0.1416`，G2 `0.6402 / 0.1701`。高风险阈值r≥0.5在测试折中没有样本，风险分布整体偏低，需按风险识别信号不足解读。
- G1相对历史基座：错误→正确0、正确→错误0、净纠错0；G2相对历史基座：错误→正确1、正确→错误0、净纠错+1（跨折计数，非独立患者队列）。G2相对G1：新纠正4、新误伤10、双方原本错误但翻转0，净变化−6。
- 原模型错误患者可靠疾病证据方向正确率：G1 0.1383，G2 0.2327；残差方向正确率：G1 0.1383，G2 0.2327。原模型正确患者可靠方向正确率：G1 0.9165，G2 0.8998。

## Residual-Off、Prompt和来源槽位

- Residual-Off的当前主干均值：G1 ACC/Hard-AUC/Prob-AUC `92.1891/92.0294/93.4946`；G2 `91.3845/91.1309/93.6801`。G2完整残差仅把ACC从91.3845提高到91.4995，仍比历史基座低0.6897pp；端到端主干退化是主要损失来源。
- Semantic-Swap与No-Semantic-Key均无测试硬预测变化；Semantic-Swap平均概率绝对变化约G1 `5.1e-9`、G2 `5.6e-9`。疾病Prompt没有得到独立硬诊断增量证据。
- 四槽位平均权重 PHENO/ANAT/FUNC/CORR：G1 `0.2147/0.2450/0.2113/0.3290`，G2 `0.2191/0.2485/0.2220/0.3105`；这是跨模态来源槽位权重，不能解释为医学脑区因果。站点配额违规和无效候选均为0。

## 审计结论

- G1保持原主干轨迹但没有净硬纠错；G2的保护式端到端路径确实到达GFR、DARG、CAM且OutputLayer新增梯度恒为0，但主干ACC/Hard-AUC/SEN/ASD-F1下降。
- 因此本轮是`RISK_DETECTION_WITHOUT_CORRECTION`：风险估计有中等区分信号，却没有可靠的反向疾病证据和实际纠错。不能将其作为MICCAI 2027第二创新模块继续扩展，也不应事后增加Prompt、原型或残差权重。
- 历史val=test选模、CV前RFE/标准化、transductive图和Fold 2历史重叠仍限制泛化结论；OOF仅隔离了本次新增风险监督。

产物：
- [配置](/model/spromptgl/experiment/experiment76_9_rbcd/config.json)
- [OOF教师脚本](/model/spromptgl/experiment/experiment76_9_rbcd/oof_teacher.py)
- [正式运行器](/model/spromptgl/experiment/experiment76_9_rbcd/run.py)
- [审计状态](/model/spromptgl/experiment/experiment76_9_rbcd/results/audit_status.json)
- [逐折六指标](/model/spromptgl/experiment/experiment76_9_rbcd/results/fold_metrics.csv)
- [配对比较](/model/spromptgl/experiment/experiment76_9_rbcd/results/paired_comparisons.csv)
- [机制](/model/spromptgl/experiment/experiment76_9_rbcd/results/mechanism.csv)
- [纠错与干预](/model/spromptgl/experiment/experiment76_9_rbcd/results/correction_analysis.csv)
- [患者翻转](/model/spromptgl/experiment/experiment76_9_rbcd/results/patient_transitions.csv)
- [G2相对G1翻转](/model/spromptgl/experiment/experiment76_9_rbcd/results/g2_vs_g1_transitions.csv)
