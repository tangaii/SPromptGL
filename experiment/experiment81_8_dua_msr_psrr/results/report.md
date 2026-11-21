# Experiment 81.8 — DUA-PSRR / MSR-PSRR

Status: `COMPLETE_AUDITED`. Independent Best and Latest audits: 20/20 and 20/20. Latest next-step replay: 20/20.

All values are ten-fold mean ± population SD in percent. Hard-AUC is the historical hard-prediction Balanced Accuracy. Prob-AUC is ROC-AUC from continuous ASD probability.

| Model | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E74.4-G2 DARG | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |
| E81.7.2-G1 | 92.0742 ± 2.7959 | 91.8832 ± 2.7902 | 93.4151 ± 3.0773 | 89.3354 ± 3.3306 | 94.4311 ± 3.3632 | 91.2647 ± 3.0317 |
| G1 DUA-PSRR | 91.8443 ± 2.3326 | 91.7342 ± 2.2903 | 93.7323 ± 3.2550 | 90.3232 ± 3.2370 | 93.1452 ± 3.9664 | 91.1335 ± 2.4189 |
| G2 MSR-PSRR | 91.8443 ± 2.7486 | 91.6753 ± 2.7503 | 93.4517 ± 2.7294 | 89.5671 ± 3.8625 | 93.7835 ± 3.9338 | 91.0496 ± 2.9557 |

## Excluding Fold 2

Nine-fold descriptive mean ± population SD in percent.

| Model | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E74.4-G2 DARG | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 | 91.4703 ± 3.0580 |
| E81.7.2-G1 | 92.0875 ± 2.9468 | 91.8857 ± 2.9411 | 93.4964 ± 3.2336 | 89.2344 ± 3.4962 | 94.5370 ± 3.5293 | 91.2543 ± 3.1956 |
| G1 DUA-PSRR | 91.9598 ± 2.4315 | 91.8556 ± 2.3835 | 93.8018 ± 3.4240 | 90.6030 ± 3.2954 | 93.1082 ± 4.1793 | 91.2595 ± 2.5184 |
| G2 MSR-PSRR | 91.8321 ± 2.8971 | 91.6547 ± 2.8983 | 93.5312 ± 2.8660 | 89.4919 ± 4.0645 | 93.8175 ± 4.1452 | 91.0153 ± 3.1137 |

## Paired Differences

Difference is left minus reference in percentage points. W/T/L is computed foldwise.

| Comparison | Subset | ACC Δ (W/T/L) | BA Δ (W/T/L) | Prob-AUC Δ | SEN Δ | SPE Δ | ASD-F1 Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1-DARG | all10 | -0.3448 (2/3/5) | -0.2952 (3/1/6) | +0.2377 | +0.2561 | -0.8464 | -0.3255 |
| G1-DARG | exclude_fold2 | -0.2554 (2/3/4) | -0.1925 (3/1/5) | +0.1876 | +0.5556 | -0.9405 | -0.2108 |
| G2-DARG | all10 | -0.3448 (0/7/3) | -0.3541 (2/4/4) | -0.0429 | -0.5000 | -0.2081 | -0.4094 |
| G2-DARG | exclude_fold2 | -0.3831 (0/6/3) | -0.3934 (2/3/4) | -0.0830 | -0.5556 | -0.2313 | -0.4549 |
| G1-E81.7.2-G1 | all10 | -0.2299 (2/4/4) | -0.1490 (5/1/4) | +0.3173 | +0.9878 | -1.2858 | -0.1312 |
| G1-E81.7.2-G1 | exclude_fold2 | -0.1277 (2/4/3) | -0.0301 (5/1/3) | +0.3054 | +1.3686 | -1.4287 | +0.0052 |
| G2-E81.7.2-G1 | all10 | -0.2299 (0/8/2) | -0.2079 (2/5/3) | +0.0366 | +0.2317 | -0.6475 | -0.2151 |
| G2-E81.7.2-G1 | exclude_fold2 | -0.2554 (0/7/2) | -0.2310 (2/4/3) | +0.0348 | +0.2575 | -0.7195 | -0.2390 |
| G2-G1 | all10 | +0.0000 (3/4/3) | -0.0589 (3/3/4) | -0.2807 | -0.7561 | +0.6383 | -0.0839 |
| G2-G1 | exclude_fold2 | -0.1277 (2/4/3) | -0.2009 (2/3/4) | -0.2706 | -1.1111 | +0.7092 | -0.2441 |

## Patient Corrections

A is historical DARG Best, B is the current Best checkpoint with the local residual disabled, and C is the full current checkpoint. A→B includes the changed training trajectory and Best epoch.

| Group | Transition | Corrected | Harmed | Net | ASD corrected/harmed | TD corrected/harmed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G1 | A_to_B | 9 | 12 | -3 | 6/5 | 3/7 |
| G1 | B_to_C | 0 | 0 | +0 | 0/0 | 0/0 |
| G1 | A_to_C | 9 | 12 | -3 | 6/5 | 3/7 |
| G2 | A_to_B | 8 | 11 | -3 | 5/7 | 3/4 |
| G2 | B_to_C | 0 | 0 | +0 | 0/0 | 0/0 |
| G2 | A_to_C | 8 | 11 | -3 | 5/7 | 3/4 |

## Mechanism

### G1

```json
{
  "dua_score_changed_relation": true,
  "mean_dua_score_delta_abs": 0.057305192574858665,
  "mean_utility_direction_consistency": null,
  "mean_u_actual_margin_correlation": null,
  "positive_actual_margin_correlation_observed": false,
  "B_to_C_net_corrections": 0,
  "utility_alignment_supported": false
}
```

### G2

```json
{
  "second_order_adds_distinct_information": true,
  "second_minus_first_relation_norm_mean": 0.9523275792598724,
  "train_relation_stability_js": 4.518085261224769e-05,
  "asd_td_relation_l1_mean": 0.34940032511949537,
  "second_order_off_probability_mae": 0.005777307454137452,
  "second_order_dependent_corrections": 0,
  "B_to_C_net_corrections": 0,
  "multiscale_mechanism_supported": false
}
```

## Scientific Decision

- G1: **NO-GO**. Performance gate `False`. Mechanism gate `False`.
- G2: **NO-GO**. Performance gate `False`. Mechanism gate `False`.

## Protocol Limits

- historical val=test Best selection.
- cross-validation was preceded by historical RFE and scaling.
- transductive patient graph.
- Fold2 retains two historical train/test-overlap rows, excluded only from new auxiliary labels and training residual.
- single seed and fixed ten-fold history.

## Outputs

- `fold_metrics.csv` contains independently recomputed Best metrics and confusion matrices.
- `patient_predictions.csv` contains fold-level Best probabilities and hard predictions.
- `paired_comparisons.csv` contains paired fold differences and W/T/L.
- `mechanism.csv` contains fold-level A/B/C and group mechanism measurements.
- `checkpoint_audit.csv` records Best/Latest load, metric, and Latest replay checks.
- `audit_status.json` contains full per-fold and gate records.

## Audit Source Provenance

The preflight recorded audit source SHA256 `f7580fa64c429e2b5f305e7612421e42e2f6364c1930d36781d7ff769f658a38`. During independent audit, audit-only compatibility and aggregation fixes were applied. The original preflight file was overwritten before a byte-identical copy was retained. A reconstructed candidate is preserved as `audit_preflight_reconstructed.py` with SHA256 `f899376402303906cf90d7804606a2f0128c460e1c514cf5f56ee81c201802f0`, which does not match the preflight SHA. Therefore `audit_source_reconstruction.diff` compares the reconstruction with the final auditor and is not an exact diff from the preflight source. The training source digest stayed at `937280975022508e5eb24c12d40b9c49db7fa0d6bf4cb2d9f4f7e2a7f2cfb6d8`; all 40 checkpoints independently passed the repaired audit. This is a source-retention gap in the audit tooling record, not a detected training-source or checkpoint discrepancy.
