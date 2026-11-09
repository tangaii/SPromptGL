<!-- EXPERIMENT_74_4_AUDIT_DIRECTION_CORRECTION -->
### Experiment 74.4审计判定方向更正

追加更正：首版`manifest.json`把固定Checkpoint干预中的翻转方向解释反了。`full_correct_intervention_wrong`表示完整模型判对、干预后判错，是完整疾病关系分支的正向贡献；`full_wrong_intervention_correct`才是移除关系后纠正。训练、Checkpoint、预测和指标未改动；仅更正审计解释及GO/PARTIAL判定，并再次完整独立读回30个Checkpoint。

E74.4-G2-Positive-DCR相对E74.3-G1的配对差（ACC/Hard-AUC/SEN；W/T/L）：
|范围|ACC|Hard-AUC|SEN|
|---|---:|---:|---:|
|all10|+0.1149 pp (2/7/1)|+0.1887 pp (3/6/1)|+1.2378 pp (3/6/1)|
|nine_without_fold2|+0.1277 pp (2/6/1)|+0.2097 pp (3/5/1)|+1.3753 pp (3/5/1)|

G2固定Checkpoint的Disease-neutral与Score-permuted合计：完整模型正确、干预后错误 1 例；完整模型错误、干预后纠正 0 例。Disease-neutral的ACC和Hard-AUC均有正向 full−intervention 差；因此在预注册规则下判为GO候选，但效果幅度小、ACC逐折多数持平，需保守解释。

更正前结论：PARTIAL；更正后结论：**GO**。审计源码SHA：预检/首版`615427238eaa9f1c1b15da29a442b5f3a0299360f42b0c2bcc43f2a054905788`，更正前`406001c058b799b354f6583ce622d92be5fdfe854ae5c2a5262d47cc2cfc2400`，更正后`cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb`。30/30 checkpoint重新审计通过，训练源码SHA与训练产物不变。详情见`/model/spromptgl/experiment/experiment74_4_soft_positive_dcr/results/report.md`及`audit_status.json`。
<!-- EXPERIMENT_74_4_AUDIT_DIRECTION_CORRECTION -->
