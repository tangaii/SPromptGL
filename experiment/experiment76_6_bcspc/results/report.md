# Experiment 76.6: BCSPC

- Status: `COMPLETE_AUDITED`; scientific decision: `NO_GO_BCSPC`.
- Runtime: approximately 1.03 active training hours (fold task wall times plus Fold 1's logged training span); independent audit: 37.1 seconds; actual training workers: 1; shared backbone trajectories: 10.
- Historical evaluation protocol retained: test-fold ACC selects checkpoints, preprocessing precedes CV, graph propagation is transductive, and Fold 2 retains the historical overlap rows 377/407. These results are matched exploratory comparisons, not unbiased independent generalization estimates.
- E76.6 is backbone-protected online plugin training: the E74.4-G2 backbone resumes from E61 epoch-10 and follows its original loss/update trajectory; new branch inputs CAM144 and the base score are detached, so branch gradients do not update the backbone.
- The restored E74.4-G2 CAM class is `PLNNoPEN` and has no original prompt buffer. To preserve the audited checkpoint, E76.6 supplies the eight SHA-verified `network.py` prompt embeddings only to the new semantic branch; it does not claim that the historical backbone ran the original PEN.

## Core Metrics

| Method | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 history | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4946 +/- 2.9195 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 | 91.4590 +/- 2.9013 |
| E76.6-G1 BCSPC-Full | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4840 +/- 2.9127 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 | 91.4590 +/- 2.9013 |
| E76.6-G2 BCSPC-NoBoundary | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4946 +/- 2.9195 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 | 91.4590 +/- 2.9013 |

Values are 10-fold mean +/- population SD in percent. Full per-fold NLL, Brier, ECE, and core metrics are in `fold_metrics.csv`.

## Paired Comparisons

- G1-E74.4-G2 ACC: +0.0000 pp (95% descriptive paired CI +0.0000, +0.0000); W/T/L=0/10/0; excluding Fold 2: +0.0000 pp.
- G2-E74.4-G2 ACC: +0.0000 pp (95% descriptive paired CI +0.0000, +0.0000); W/T/L=0/10/0; excluding Fold 2: +0.0000 pp.
- G1-G2 ACC: +0.0000 pp (95% descriptive paired CI +0.0000, +0.0000); W/T/L=0/10/0; excluding Fold 2: +0.0000 pp.

| Paired comparison | Metric | Mean difference (pp) | Descriptive 95% CI (pp) | W/T/L | Excluding Fold 2 (pp) |
|---|---|---:|---:|---:|---:|
| G1-E74.4-G2 | acc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-E74.4-G2 | hard_auc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-E74.4-G2 | prob_auc | -0.0106 | [-0.0267, +0.0054] | 0/8/2 | -0.0118 |
| G1-E74.4-G2 | sen | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-E74.4-G2 | spe | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-E74.4-G2 | asd_f1 | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | acc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | hard_auc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | prob_auc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | sen | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | spe | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G2-E74.4-G2 | asd_f1 | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-G2 | acc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-G2 | hard_auc | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-G2 | prob_auc | -0.0106 | [-0.0267, +0.0054] | 0/8/2 | -0.0118 |
| G1-G2 | sen | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-G2 | spe | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |
| G1-G2 | asd_f1 | +0.0000 | [+0.0000, +0.0000] | 0/10/0 | +0.0000 |

## Mechanism

- Historical E74.4-G2 to G1 prediction changes: error-to-correct 0, correct-to-error 0, net 0; original-correct retention 100.00%.
- Same G1 checkpoint, Full versus Residual-Off: error-to-correct 0, correct-to-error 0, net 0; ACC change +0.0000 pp.
- Same G1 checkpoint, Full versus Boundary-Off: net correction 0; ACC change +0.0000 pp.
- Same G1 checkpoint, Full versus Semantic-Swap: net correction 0; ACC change +0.0000 pp.
- Mean G1 gate=0.1216, mean |delta|=0.00653, attention reliable/boundary=0.4998/0.5002.
- Mean G1 modality weights PHENO/ANAT/FUNC/CORR=0.0654, 0.0540, 0.0484, 0.8323. These are CAM source-slot weights, not biological localization evidence.
- Boundary value is tested by the independently trained NoBoundary group and same-checkpoint Boundary-Off intervention. Text semantics are tested by Semantic-Swap and attention/evidence diagnostics; neither alone establishes clinically validated semantic knowledge.

## Audit

- Independent audit status: `COMPLETE_AUDITED`; checkpoint recalc: 20; RNG shared epoch triplets: 3917/3917 exact; base parameter/optimizer best state: True.
- The independent audit verifies checkpoint identity and SHA, exact base trajectory RNG triplets, original best model/optimizer state, held-out label exclusion from the new memory, A/B non-self retrieval support, parameter-count parity, and recalculated best-checkpoint predictions and metrics.

The scientific decision is deliberately separate from audit completion. See `audit_status.json`, `paired_comparisons.csv`, `mechanism.csv`, and `correction_analysis.csv` for machine-readable details.
