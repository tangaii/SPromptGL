# Experiment 55 — DCRAA-GFR

**Status:** 110/110 independently reloaded and audited group-fold checkpoints.

## Results

All metrics are percent mean ± population SD over the fixed ten folds. Hard-AUC uses hard class predictions (equal to BA for this binary setup). Probability-AUC uses continuous ASD probability `P(ASD)` with ASD label as positive.

|Group|Definition|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E0|E51_E1_EXACT_REPLAY|91.9592 ± 2.8684|91.7434 ± 2.8950|93.1856 ± 3.0236|94.6392 ± 3.3856|88.8476 ± 4.1651|91.7434 ± 2.8950|91.0928 ± 3.1408|
|E1|RELATION_CAPACITY_CONTROL|91.2709 ± 2.9237|91.0926 ± 2.9185|93.4094 ± 2.7886|93.5754 ± 3.7234|88.6098 ± 3.9623|91.0926 ± 2.9185|90.3935 ± 3.1610|
|E2|DISEASE_RELATION_ONLY|91.3845 ± 3.1017|91.2433 ± 3.1045|93.3109 ± 2.8077|93.1452 ± 3.3236|89.3415 ± 3.6429|91.2433 ± 3.1045|90.5758 ± 3.3525|
|E3|UNCONDITIONAL_RELATION_ADV|91.8443 ± 2.9346|91.6424 ± 2.9468|93.4987 ± 2.7822|94.4311 ± 3.4896|88.8537 ± 4.1343|91.6424 ± 2.9468|90.9813 ± 3.2201|
|E4|DISEASE_CONDITIONAL_ADV|91.5008 ± 3.0557|91.3793 ± 3.0274|93.4045 ± 2.9583|92.9232 ± 4.0158|89.8354 ± 3.3922|91.3793 ± 3.0274|90.7572 ± 3.2154|
|E5|FULL_DCRAA_GFR|91.6144 ± 3.0928|91.4484 ± 3.0923|93.4419 ± 2.9490|93.5615 ± 4.5605|89.3354 ± 4.5730|91.4484 ± 3.0923|90.8057 ± 3.2920|
|E6|WRONG_DISEASE_CONDITION|91.7294 ± 2.7174|91.5115 ± 2.6844|93.1457 ± 3.0774|94.1998 ± 4.9534|88.8232 ± 4.3815|91.5115 ± 2.6844|90.8800 ± 2.8706|
|E7|WRONG_SITE_ADVERSARY|91.9592 ± 3.0901|91.7885 ± 3.0261|93.6547 ± 3.0105|93.9917 ± 4.3141|89.5854 ± 2.8701|91.7885 ± 3.0261|91.2072 ± 3.2336|
|E8|WRONG_PATIENT_RELATION|91.3845 ± 2.7399|91.1582 ± 2.6798|93.5219 ± 2.9039|93.9870 ± 5.1964|88.3293 ± 4.3396|91.1582 ± 2.6798|90.4959 ± 2.8723|
|E9|NO_CROSS_PATIENT_MESSAGE|91.9592 ± 2.8684|91.7434 ± 2.8950|93.1856 ± 3.0236|94.6392 ± 3.3856|88.8476 ± 4.1651|91.7434 ± 2.8950|91.0928 ± 3.1408|
|E10|NO_DISEASE_PROTECTION|91.7294 ± 3.0819|91.5462 ± 3.0715|93.3120 ± 3.6554|94.0009 ± 4.0539|89.0915 ± 4.0230|91.5462 ± 3.0715|90.8995 ± 3.3355|

## E5 Paired Controls

Same-fold mean differences in percentage points; W/T/L counts describe ten paired folds and are not independent-sample significance tests.

|Comparison|Metric|Mean Δ pp|Fold deltas|W/T/L|
|---|---|---:|---|---:|
|E5_vs_E0|acc|-0.3448|+0.000, -1.149, +0.000, -1.149, +0.000, +0.000, +1.149, -1.149, -1.149, +0.000|1/5/4|
|E5_vs_E0|hard_auc|-0.2950|+0.000, -1.220, +0.398, -1.064, +0.000, +0.186, +1.250, -1.064, -1.250, -0.186|3/2/5|
|E5_vs_E0|prob_auc|+0.2563|-0.259, +0.053, -1.060, -0.532, -1.277, +0.798, +0.053, +1.277, +1.596, +1.915|6/0/4|
|E5_vs_E0|spe|-1.0777|+0.000, +0.000, -6.522, -2.128, +0.000, -2.128, +0.000, -2.128, +0.000, +2.128|1/5/4|
|E5_vs_E0|sen|+0.4878|+0.000, -2.439, +7.317, +0.000, +0.000, +2.500, +2.500, +0.000, -2.500, -2.500|3/4/3|
|E5_vs_E3|acc|-0.2299|+0.000, -1.149, +0.000, -1.149, +1.149, +0.000, +0.000, +0.000, -1.149, +0.000|1/6/3|
|E5_vs_E3|hard_auc|-0.1939|+0.000, -1.220, +0.530, -1.064, +1.064, +0.186, +0.000, +0.000, -1.250, -0.186|3/3/4|
|E5_vs_E3|prob_auc|-0.0568|-0.571, +0.371, -1.220, +0.053, -2.872, +0.213, +1.543, -0.426, +0.904, +1.436|6/0/4|
|E5_vs_E3|spe|-0.8696|+0.000, +0.000, -8.696, -2.128, +2.128, -2.128, +0.000, +0.000, +0.000, +2.128|2/5/3|
|E5_vs_E3|sen|+0.4817|+0.000, -2.439, +9.756, +0.000, +0.000, +2.500, +0.000, +0.000, -2.500, -2.500|2/5/3|
|E5_vs_E4|acc|+0.1136|+1.136, +0.000, +0.000, -1.149, +0.000, +0.000, +1.149, +0.000, +0.000, +0.000|2/7/1|
|E5_vs_E4|hard_auc|+0.0691|+1.064, -0.398, +0.398, -1.064, +0.000, +0.186, +0.878, +0.000, -0.186, -0.186|4/2/4|
|E5_vs_E4|prob_auc|+0.0374|-0.052, +0.689, -0.795, +0.000, -2.660, +0.213, +1.543, +0.053, +1.649, -0.266|5/1/4|
|E5_vs_E4|spe|+0.6383|+2.128, +6.522, -6.522, -2.128, +0.000, -2.128, +4.255, +0.000, +2.128, +2.128|5/2/3|
|E5_vs_E4|sen|-0.5000|+0.000, -7.317, +7.317, +0.000, +0.000, +2.500, -2.500, +0.000, -2.500, -2.500|2/4/4|
|E5_vs_E6|acc|-0.1149|+0.000, -2.299, +0.000, +0.000, +1.149, +0.000, +1.149, -1.149, +0.000, +0.000|2/6/2|
|E5_vs_E6|hard_auc|-0.0631|+0.000, -2.439, +0.000, +0.186, +1.064, +0.186, +1.250, -1.064, +0.186, +0.000|5/3/2|
|E5_vs_E6|prob_auc|+0.2962|+0.415, +2.174, -0.265, +1.011, -2.979, +0.585, -0.053, -0.106, +1.436, +0.745|6/0/4|
|E5_vs_E6|spe|-0.6383|+0.000, +0.000, +0.000, -2.128, +2.128, -2.128, +0.000, -2.128, -2.128, +0.000|1/5/4|
|E5_vs_E6|sen|+0.5122|+0.000, -4.878, +0.000, +2.500, +0.000, +2.500, +2.500, +0.000, +2.500, +0.000|4/5/1|
|E5_vs_E7|acc|-0.3448|+0.000, -1.149, +0.000, -1.149, -1.149, +0.000, +0.000, +0.000, +0.000, +0.000|0/7/3|
|E5_vs_E7|hard_auc|-0.3401|+0.000, -1.220, +0.133, -1.064, -0.691, +0.186, -0.372, +0.000, -0.186, -0.186|2/2/6|
|E5_vs_E7|prob_auc|-0.2128|+0.104, +0.159, -0.901, -0.745, -3.830, +0.266, +0.213, -0.638, +2.394, +0.851|6/0/4|
|E5_vs_E7|spe|-0.4302|+0.000, +0.000, -2.174, -2.128, -6.383, -2.128, +4.255, +0.000, +2.128, +2.128|3/3/4|
|E5_vs_E7|sen|-0.2500|+0.000, -2.439, +2.439, +0.000, +5.000, +2.500, -5.000, +0.000, -2.500, -2.500|3/3/4|
|E5_vs_E8|acc|+0.2299|+0.000, -1.149, +0.000, +0.000, +1.149, +1.149, +1.149, +0.000, +0.000, +0.000|3/6/1|
|E5_vs_E8|hard_auc|+0.2903|+0.000, -1.220, +0.000, +0.186, +1.064, +1.436, +1.436, +0.000, +0.000, +0.000|4/5/1|
|E5_vs_E8|prob_auc|-0.0800|+0.104, +0.371, -0.530, -0.745, -2.287, +0.319, +0.638, +0.000, +0.106, +1.223|6/1/3|
|E5_vs_E8|spe|-0.4255|+0.000, +0.000, +0.000, -2.128, +2.128, -2.128, -2.128, +0.000, +0.000, +0.000|1/6/3|
|E5_vs_E8|sen|+1.0061|+0.000, -2.439, +0.000, +2.500, +0.000, +5.000, +5.000, +0.000, +0.000, +0.000|3/6/1|
|E5_vs_E9|acc|-0.3448|+0.000, -1.149, +0.000, -1.149, +0.000, +0.000, +1.149, -1.149, -1.149, +0.000|1/5/4|
|E5_vs_E9|hard_auc|-0.2950|+0.000, -1.220, +0.398, -1.064, +0.000, +0.186, +1.250, -1.064, -1.250, -0.186|3/2/5|
|E5_vs_E9|prob_auc|+0.2563|-0.259, +0.053, -1.060, -0.532, -1.277, +0.798, +0.053, +1.277, +1.596, +1.915|6/0/4|
|E5_vs_E9|spe|-1.0777|+0.000, +0.000, -6.522, -2.128, +0.000, -2.128, +0.000, -2.128, +0.000, +2.128|1/5/4|
|E5_vs_E9|sen|+0.4878|+0.000, -2.439, +7.317, +0.000, +0.000, +2.500, +2.500, +0.000, -2.500, -2.500|3/4/3|
|E5_vs_E10|acc|-0.1149|+0.000, -2.299, +0.000, -1.149, +1.149, +0.000, +1.149, +0.000, +1.149, -1.149|3/4/3|
|E5_vs_E10|hard_auc|-0.0978|+0.000, -2.439, +0.398, -1.064, +1.064, +0.186, +1.250, +0.000, +0.878, -1.250|5/2/3|
|E5_vs_E10|prob_auc|+0.1299|-1.142, +2.492, -0.583, -0.957, -3.085, +0.160, -0.266, +1.277, +1.755, +1.649|5/0/5|
|E5_vs_E10|spe|-0.4394|+0.000, +0.000, -6.522, -2.128, +2.128, -2.128, +0.000, +0.000, +4.255, +0.000|2/5/3|
|E5_vs_E10|sen|+0.2439|+0.000, -4.878, +7.317, +0.000, +0.000, +2.500, +2.500, +0.000, -2.500, -2.500|3/4/3|

## Mechanism

CORR relation OOF macro AUC was 0.6667 ± 0.0085; independent outer-test relation-pair macro AUC was 0.8072 ± 0.0399. The test relation classifier is diagnostic only and its labels never enter conditions, graph construction or training.
The same-site/cross-site adversary was active in 10/10 folds. A held-out raw-CORR site-source probe averaged AUC 0.5667; the trained conditional discriminator averaged 0.5207. Near-chance post-adversary AUC alone is not alignment evidence.
No PHENO, ANAT or FUNC modality passed the pre-registered auxiliary OOF admission rule in any fold. This validates only CORR-internal site-pair alignment, not cross-modal disease-relation alignment. E5 mean added-message residual ratio was 0.001919; true-edge vs degree/site-matched wrong-edge independent retraining delta ACC was +0.2299 pp; E5 vs no-message E9 delta ACC was -0.3448 pp.

Fixed E5 best-checkpoint inference interventions (mean over folds; not independent retrains):

|Intervention|ΔACC pp|Flip rate|MeanAbsDeltaP_ASD|MeanAbsDeltaLogit|Edge-weight L1|
|---|---:|---:|---:|---:|---:|
|no_cross_patient_message|+0.0000|0.0000|0.000339|0.001385|0.153004|
|normal_true_relation|+0.0000|0.0000|0.000000|0.000000|0.000000|
|only_original_e1|+0.0000|0.0000|0.000339|0.001385|0.153004|
|relation_reweighting_disabled|+0.0000|0.0000|0.000073|0.000284|0.026302|
|wrong_disease_condition|+0.0000|0.0000|0.000036|0.000148|0.014878|
|wrong_patient_neighbors|+0.0000|0.0000|0.000035|0.000100|0.058048|
|wrong_site_source_label|+0.0000|0.0000|0.000000|0.000000|0.000000|

## Decision

- `GO_RELATION_DISCRIMINATION`: **GO**
- `GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT`: **NO-GO**
- `GO_DISEASE_PRESERVATION`: **GO**
- `GO_TRUE_PATIENT_RELATION`: **NOT_SUPPORTED**
- `GO_CROSS_PATIENT_PROPAGATION`: **NOT_SUPPORTED**
- `GO_CLASSIFICATION_GAIN`: **NO-GO**
- `GO_MICCAI2027_MODULE`: **NO-GO**

The result is exploratory: seed=0, one ABIDE A cohort, historical `val_idx=test_idx` checkpoint selection, preprocessing/RFE before outer CV, and an 871-node transductive graph. The ten folds overlap in training patients; no multi-seed stability, statistical significance or unbiased external-site generalization is claimed.

## Audit

Independent checkpoint audit: 110/110 passed. RNG base parity: PASS on all ten E1 folds. Resume equivalence and structural/cache preflight results are in `results/preflight_structural.json` and `results/preflight_cache.json`.
