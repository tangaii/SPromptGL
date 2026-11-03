<!-- EXPERIMENT60_DSS_PEN_2026-11-02 -->
## Experiment 60 — DSS-PEN

- **状态**：18/18正式组、180/180折训练完成；180/180 checkpoint 独立读回和组别实际生效核验均为`PASS`。R01 基座 parity：`PASS`，10/10 折。源文件、逐折SHA及审计记录位于`/model/spromptgl/experiment/experiment60_dss_pen/`。
- **唯一基座**：Experiment 53.3第三组`CADR_WD_0p01_GFR_WD_0p11`；CADR AdamW 两个参数组 WD=0.01，GFR/CAM Adam WD=0.11。所有新增折均从同一逐折未训练锚点独立开始。固定 seed=0、10折、传导式图、全局预处理与历史`val=test`选模协议保留。
- **方法和对照**：R02复原SPromptGL原SPGS+RES代码路径；R03/R04为统一CLIP编码器的BrainPrompt式类别文本/医学中心适配；R05按SABER公式检索PHENO患者文本并另加疾病监督，R06为疾病文本查询改编。R07使用每类6个医学概念、共享64维适配器、128步可微部分集合投影（系数非负且总和≤1、含空匹配）、集合距离损失和有界CAM残差。R08–R18分别对比类别名、错配/随机/可学习语义、单原型、Softmax、强制/自由匹配、无集合损失、无残差及CORR单槽干预式训练消融。全部细节见`design.md`和`config.json`。

指标为十折mean±population SD（%）；ΔACC为R07减相应组，胜/平/负按折统计：

| Group | ACC | Hard-AUC | Probability-AUC |
|---|---:|---:|---:|
| E60_REF_533_G3 | 92.074 ± 2.843 | 91.865 ± 2.835 | 93.668 ± 3.262 |
| E60_BASE_REPLAY | 92.074 ± 2.843 | 91.865 ± 2.835 | 93.668 ± 3.262 |
| E60_ORIGINAL_PEN | 91.270 ± 3.225 | 91.116 ± 3.207 | 93.181 ± 3.268 |
| E60_BRAINPROMPT_LABEL | 91.499 ± 2.924 | 91.320 ± 2.913 | 93.356 ± 2.756 |
| E60_BRAINPROMPT_MEDICAL | 91.155 ± 3.136 | 90.922 ± 3.157 | 93.329 ± 3.275 |
| E60_SABER_DECISION | 91.614 ± 2.825 | 91.501 ± 2.832 | 92.936 ± 2.933 |
| E60_DISEASE_QUERY | 91.385 ± 3.464 | 91.194 ± 3.481 | 92.875 ± 3.439 |
| E60_DSS_FULL | 91.499 ± 3.225 | 91.326 ± 3.223 | 93.769 ± 2.829 |
| E60_DSS_CLASS_ONLY | 91.846 ± 2.842 | 91.671 ± 2.819 | 93.539 ± 2.613 |
| E60_DSS_WRONG_SEMANTICS | 91.728 ± 3.329 | 91.589 ± 3.288 | 93.436 ± 2.914 |
| E60_DSS_RANDOM_ANCHORS | 91.844 ± 2.889 | 91.634 ± 2.822 | 93.657 ± 2.856 |
| E60_DSS_LEARNED_PROTOTYPES | 91.614 ± 3.135 | 91.475 ± 3.035 | 93.394 ± 3.194 |
| E60_DSS_SINGLE_PROTOTYPE | 91.614 ± 3.050 | 91.459 ± 2.963 | 93.565 ± 2.828 |
| E60_DSS_SOFTMAX_ATTENTION | 91.386 ± 2.641 | 91.153 ± 2.644 | 93.245 ± 2.631 |
| E60_DSS_FORCE_MATCH | 91.155 ± 3.300 | 91.045 ± 3.246 | 93.423 ± 3.182 |
| E60_DSS_FREE_SET | 91.499 ± 3.306 | 91.276 ± 3.312 | 93.235 ± 3.019 |
| E60_DSS_NO_SET_LOSS | 91.385 ± 3.059 | 91.206 ± 3.023 | 93.548 ± 3.072 |
| E60_DSS_NO_RESIDUAL | 91.385 ± 3.059 | 91.244 ± 3.013 | 93.421 ± 3.105 |
| E60_DSS_CORR_ONLY | 91.156 ± 3.177 | 91.026 ± 3.089 | 93.662 ± 2.883 |

| R07 comparison | ΔACC (pp) | Wins/Ties/Losses |
|---|---:|---:|
| E60_REF_533_G3 | -0.575 | 2/3/5 |
| E60_ORIGINAL_PEN | +0.230 | 3/4/3 |
| E60_BRAINPROMPT_LABEL | -0.000 | 2/6/2 |
| E60_BRAINPROMPT_MEDICAL | +0.345 | 3/5/2 |
| E60_SABER_DECISION | -0.115 | 3/4/3 |
| E60_DISEASE_QUERY | +0.115 | 3/3/4 |
| E60_DSS_CLASS_ONLY | -0.346 | 3/3/4 |
| E60_DSS_WRONG_SEMANTICS | -0.229 | 1/6/3 |
| E60_DSS_RANDOM_ANCHORS | -0.345 | 2/5/3 |
| E60_DSS_LEARNED_PROTOTYPES | -0.115 | 2/5/3 |
| E60_DSS_SINGLE_PROTOTYPE | -0.115 | 4/3/3 |
| E60_DSS_SOFTMAX_ATTENTION | +0.114 | 4/4/2 |
| E60_DSS_FORCE_MATCH | +0.345 | 3/5/2 |
| E60_DSS_FREE_SET | +0.000 | 2/6/2 |
| E60_DSS_NO_SET_LOSS | +0.115 | 2/6/2 |
| E60_DSS_NO_RESIDUAL | +0.115 | 3/6/1 |
| E60_DSS_CORR_ONLY | +0.344 | 4/4/2 |

- R07 vs历史R00：ΔACC -0.575 pp，2/3/5 折胜/平/负。完整配对结果、exact sign-flip、bootstrap区间和Holm校正见`results/paired_comparisons.csv`。
- R07 ASD患者异质性冻结规则通过4/10折；ASD/TD空匹配率分别为12.63%/1.30%。语义分配、几何、分类梯度、拒绝分层与固定checkpoint干预见`results/diagnostics.csv`、`semantic_patient_assignments.csv`、`interventions.csv`。

### 预注册判定

- GO_BASE_PARITY: **PASS** — R01 10/10 checkpoint state and prediction parity
- GO_DIAGNOSTIC: **FAIL** — ΔACC -0.575 pp; W/T/L 2/3/5; 95% CI [-1.264, +0.115]; ΔSEN +0.000 pp
- GO_SET_PROJECTION: **FAIL** — R12: ΔACC -0.115 pp; W/T/L 4/3/3; 95% CI [-0.920, +0.690]; ΔSEN -0.482 pp; R13: ΔACC +0.114 pp; W/T/L 4/4/2; 95% CI [-0.809, +0.920]; ΔSEN +1.000 pp; heterogeneity 4/10 folds
- GO_PATIENT_HETEROGENEITY: **FAIL** — frozen criteria met in 4/10 folds
- GO_SEMANTIC_CONTENT: **FAIL** — CLASS_ONLY: ΔACC -0.346 pp; W/T/L 3/3/4; 95% CI [-1.149, +0.456]; ΔSEN -0.250 pp; WRONG_SEMANTICS: ΔACC -0.229 pp; W/T/L 1/6/3; 95% CI [-0.686, +0.230]; ΔSEN -0.738 pp; RANDOM_ANCHORS: ΔACC -0.345 pp; W/T/L 2/5/3; 95% CI [-1.034, +0.345]; ΔSEN +0.018 pp; LEARNED_PROTOTYPES: ΔACC -0.115 pp; W/T/L 2/5/3; 95% CI [-0.575, +0.345]; ΔSEN -0.732 pp
- GO_DECISION_INJECTION: **PASS** — residual_off: max |Δlogit|=0.0342, mean |ΔP_ASD|=0.001034, flips=1; wrong_semantics: max |Δlogit|=4.77e-07, mean |ΔP_ASD|=6.423e-10, flips=0
- GO_VS_BRAINPROMPT: **FAIL** — R03: ΔACC -0.000 pp; W/T/L 2/6/2; 95% CI [-0.690, +0.690]; ΔSEN +0.012 pp; R04: ΔACC +0.345 pp; W/T/L 3/5/2; 95% CI [-0.345, +1.264]; ΔSEN +1.244 pp
- GO_VS_SABER: **FAIL** — R05: ΔACC -0.115 pp; W/T/L 3/4/3; 95% CI [-1.379, +1.034]; ΔSEN -0.982 pp; R06: ΔACC +0.115 pp; W/T/L 3/3/4; 95% CI [-0.690, +0.920]; ΔSEN +0.488 pp
- GO_MICCAI_CANDIDATE: **FAIL** — all frozen scientific gates, 180/180 readback audits, and 180/180 method audits

报告：`/model/spromptgl/experiment/experiment60_dss_pen/results/report.md`。保留的全局预处理、transductive图和`val=test`协议限制了外部泛化解释；单seed配对统计仅作探索性证据。模型诊断不构成因果解释或经过验证的生物标志物。
<!-- /EXPERIMENT60_DSS_PEN_2026-11-02 -->
