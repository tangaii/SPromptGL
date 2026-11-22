# Experiment 81.8.1 - CM-PSRR

Status: `COMPLETE_AUDITED`. Independent Best 10/10, Latest 10/10, Latest replay 10/10.

## Objective and Method

CM-PSRR keeps the E81.7.2-G1 node-level disease-Prompt relation path, Entmax1.5 selection and Null candidate, then trains a same-state residual-off B and full-residual C pair. The auxiliary objective uses the legal training patients' change in true-class OutputLayer margin, with difficult-case gain and reliable-case harm protection. Auxiliary gradients are isolated to the relation module, conflict-projected against classification gradients, and capped at 20% of the relation classification-gradient norm. The original E74.4-G2 losses, residual penalty, optimizer order and shared-gradient protection remain in the main training path.

The tested 16 x 9 slots are latent CAM coordinates, not anatomical ROIs. The base is E74.4-G2 PLNNoPEN. No original PEN or extra diagnostic classifier was enabled.

## Training and Provenance

Training completed 10/10 tasks on physical GPU 0 `GPU-019e120d-89d0-0224-0000-000060af1149` with a maximum of 10 workers. Training wall time was 4549.2 s. Preflight status is `PASS`. The frozen training-source digest is `b59bd54735744001c012421e5299810861f289eb3d4950f9f56d366f78db237e`. The audit source was revised after preflight for independent evaluation and intervention correctness, and its final SHA256 is `0858e9a9270e81a74beb90f83269265e48f6706db824f2365340725015b83f6e`. The preflight recorded audit SHA256 `0f1bc591b88412c47ee14a54ea6ea6b3d3f7e87f8b7a462b735184c771687709`. The archived `audit_preflight_original.py` SHA256 is `edf07a61dd5f7ad2b26fce136b310c9e58b1ca676c2ffb587461ebe3978e2f93`, which does not match the recorded preflight digest. The saved `audit_fixes.diff` SHA256 is `bf60de9cf77a015280b21712877203f0dd1443753631e1c0cead9f0a911ad5e1`. This audit-source archive mismatch is a provenance limitation. Training source and checkpoints were not changed by the audit update.

The E74.4-G2 `prepare(fold)` identities, warmup snapshots, three optimizer states, RNG, fold membership and 4096 relation pairs passed 10-fold preflight. Best checkpoint epochs are F1=103, F2=157, F3=177, F4=184, F5=338, F6=301, F7=102, F8=202, F9=360, F10=344.

## Ten-Fold Results

All values are ten-fold mean +/- population SD in percent. Hard-AUC/BA is historical hard-prediction Balanced Accuracy. Prob-AUC is continuous ASD-probability ROC-AUC.

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E74.4-G2 DARG | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4946 +/- 2.9195 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 | 91.4590 +/- 2.9013 |
| E81.7.2-G1 | 92.0742 +/- 2.7959 | 91.8832 +/- 2.7902 | 93.4151 +/- 3.0773 | 89.3354 +/- 3.3306 | 94.4311 +/- 3.3632 | 91.2647 +/- 3.0317 |
| E81.8 G1 DUA-PSRR | 91.8443 +/- 2.3326 | 91.7342 +/- 2.2903 | 93.7323 +/- 3.2550 | 90.3232 +/- 3.2370 | 93.1452 +/- 3.9664 | 91.1335 +/- 2.4189 |
| E81.8 G2 MSR-PSRR | 91.8443 +/- 2.7486 | 91.6753 +/- 2.7503 | 93.4517 +/- 2.7294 | 89.5671 +/- 3.8625 | 93.7835 +/- 3.9338 | 91.0496 +/- 2.9557 |
| E81.8.1 G1 CM-PSRR | 91.9606 +/- 2.6767 | 91.7403 +/- 2.6908 | 92.9635 +/- 2.6766 | 88.8415 +/- 4.3420 | 94.6392 +/- 4.0086 | 91.0957 +/- 2.9042 |

## Source Verification

The method was checked against the [MICCAI 2025 SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), the [original authors' repository](https://github.com/wanxixi11/SPromptGL), and the [current project repository](https://github.com/tangaii/SPromptGL). The server-side E74.4-G2 and E81.7.2-G1 files listed below were used as the implementation authority. The complete input/checkpoint inventory is recorded in `results/preflight.json`.

| Historical source | SHA256 |
| --- | --- |
| `/data/adodas/SPromptGL-main/network.py` | `484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec` |
| `/data/adodas/SPromptGL-main/layers.py` | `4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a` |
| `/data/adodas/SPromptGL-main/model.py` | `254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda` |
| `/data/adodas/SPromptGL-main/main.py` | `dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2` |
| `/data/adodas/SPromptGL-main/utils.py` | `95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67` |
| `/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/run.py` | `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd` |
| `/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/soft_positive_dcr.py` | `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718` |
| `/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr/config.json` | `a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a` |
| `/model/spromptgl/experiment/experiment81_7_2_tp_psrr/run.py` | `703ec224462656b0d6864dd76ab6f6b9c99d5f6eac63dd2e5a2fc95888b2ff99` |
| `/model/spromptgl/experiment/experiment81_7_2_tp_psrr/tp_psrr.py` | `31ffb4cb8694232de6d238cbdea20151fe1461a2db25ba8ed47af2322d435e15` |
| `/model/spromptgl/experiment/experiment81_7_2_tp_psrr/config.json` | `973795c569c13f034b53d4dcdfbb634b8d56f4010df12d76a4ce11afc2085bd5` |
| `/model/spromptgl/experiment/experiment81_7_2_tp_psrr/results/fold_metrics.csv` | `5b4d163d904413776b0bec1a3781617be4cc9878d28930a93b172b8697b4c05f` |
| `/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/fold_metrics.csv` | `f25688b0b06ad9baab2a2510b1b6340485461c16a36d899e1d2c33dcfe436dcd` |
| `/model/spromptgl/experiment/experiment81_8_dua_msr_psrr/results/audit_status.json` | `4a8d8a0301b24e8654fa036b5b45da98b56c2511dda8c944fbf3b2d9f5c87910` |

## Excluding Fold 2

| Model | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E74.4-G2 DARG | 92.2153 +/- 2.9131 | 92.0481 +/- 2.8723 | 93.6142 +/- 3.0541 | 90.0474 +/- 4.0959 | 94.0487 +/- 4.9847 | 91.4703 +/- 3.0580 |
| E81.7.2-G1 | 92.0875 +/- 2.9468 | 91.8857 +/- 2.9411 | 93.4964 +/- 3.2336 | 89.2344 +/- 3.4962 | 94.5370 +/- 3.5293 | 91.2543 +/- 3.1956 |
| E81.8 G1 DUA-PSRR | 91.9598 +/- 2.4315 | 91.8556 +/- 2.3835 | 93.8018 +/- 3.4240 | 90.6030 +/- 3.2954 | 93.1082 +/- 4.1793 | 91.2595 +/- 2.5184 |
| E81.8 G2 MSR-PSRR | 91.8321 +/- 2.8971 | 91.6547 +/- 2.8983 | 93.5312 +/- 2.8660 | 89.4919 +/- 4.0645 | 93.8175 +/- 4.1452 | 91.0153 +/- 3.1137 |
| E81.8.1 G1 CM-PSRR | 92.0890 +/- 2.7921 | 91.8624 +/- 2.8100 | 92.9475 +/- 2.8209 | 88.9566 +/- 4.5624 | 94.7682 +/- 4.2057 | 91.2175 +/- 3.0370 |

## Paired Differences

| Comparison | Subset | Metric | Mean delta pp | Population SD pp | W/T/L |
| --- | --- | --- | ---: | ---: | --- |
| G1-DARG | all10 | ACC | -0.2286 | 0.4572 | 0/8/2 |
| G1-DARG | all10 | Hard-AUC/BA | -0.2890 | 0.4737 | 0/6/4 |
| G1-DARG | all10 | Prob-AUC | -0.5311 | 1.2611 | 4/0/6 |
| G1-DARG | all10 | SEN-ASD | -1.2256 | 1.6408 | 0/6/4 |
| G1-DARG | all10 | SPE-TD | +0.6475 | 1.3870 | 2/8/0 |
| G1-DARG | all10 | ASD-F1 | -0.3633 | 0.5338 | 0/6/4 |
| G1-DARG | exclude_fold2 | ACC | -0.1263 | 0.3571 | 0/8/1 |
| G1-DARG | exclude_fold2 | Hard-AUC/BA | -0.1856 | 0.3774 | 0/6/3 |
| G1-DARG | exclude_fold2 | Prob-AUC | -0.6668 | 1.2582 | 3/0/6 |
| G1-DARG | exclude_fold2 | SEN-ASD | -1.0908 | 1.6762 | 0/6/3 |
| G1-DARG | exclude_fold2 | SPE-TD | +0.7195 | 1.4442 | 2/7/0 |
| G1-DARG | exclude_fold2 | ASD-F1 | -0.2528 | 0.4410 | 0/6/3 |
| G1-E81.7.2-G1 | all10 | ACC | -0.1136 | 0.8029 | 2/5/3 |
| G1-E81.7.2-G1 | all10 | Hard-AUC/BA | -0.1429 | 0.9281 | 3/4/3 |
| G1-E81.7.2-G1 | all10 | Prob-AUC | -0.4516 | 1.3774 | 4/0/6 |
| G1-E81.7.2-G1 | all10 | SEN-ASD | -0.4939 | 3.1090 | 2/5/3 |
| G1-E81.7.2-G1 | all10 | SPE-TD | +0.2081 | 1.7735 | 2/6/2 |
| G1-E81.7.2-G1 | all10 | ASD-F1 | -0.1689 | 1.0692 | 3/4/3 |
| G1-E81.7.2-G1 | exclude_fold2 | ACC | +0.0015 | 0.7641 | 2/5/2 |
| G1-E81.7.2-G1 | exclude_fold2 | Hard-AUC/BA | -0.0233 | 0.9022 | 3/4/2 |
| G1-E81.7.2-G1 | exclude_fold2 | Prob-AUC | -0.5489 | 1.4189 | 3/0/6 |
| G1-E81.7.2-G1 | exclude_fold2 | SEN-ASD | -0.2778 | 3.2051 | 2/5/2 |
| G1-E81.7.2-G1 | exclude_fold2 | SPE-TD | +0.2313 | 1.8680 | 2/5/2 |
| G1-E81.7.2-G1 | exclude_fold2 | ASD-F1 | -0.0368 | 1.0468 | 3/4/2 |
| G1-E81.8-G1 | all10 | ACC | +0.1162 | 0.9531 | 4/3/3 |
| G1-E81.8-G1 | all10 | Hard-AUC/BA | +0.0061 | 1.1698 | 4/2/4 |
| G1-E81.8-G1 | all10 | Prob-AUC | -0.7689 | 1.4756 | 1/1/8 |
| G1-E81.8-G1 | all10 | SEN-ASD | -1.4817 | 5.1385 | 2/5/3 |
| G1-E81.8-G1 | all10 | SPE-TD | +1.4940 | 3.4382 | 5/3/2 |
| G1-E81.8-G1 | all10 | ASD-F1 | -0.0378 | 1.3323 | 4/2/4 |
| G1-E81.8-G1 | exclude_fold2 | ACC | +0.1292 | 1.0038 | 4/2/3 |
| G1-E81.8-G1 | exclude_fold2 | Hard-AUC/BA | +0.0068 | 1.2331 | 4/1/4 |
| G1-E81.8-G1 | exclude_fold2 | Prob-AUC | -0.8543 | 1.5318 | 1/0/8 |
| G1-E81.8-G1 | exclude_fold2 | SEN-ASD | -1.6463 | 5.3914 | 2/4/3 |
| G1-E81.8-G1 | exclude_fold2 | SPE-TD | +1.6600 | 3.5860 | 5/2/2 |
| G1-E81.8-G1 | exclude_fold2 | ASD-F1 | -0.0420 | 1.4043 | 4/1/4 |
| G1-E81.8-G2 | all10 | ACC | +0.1162 | 0.8026 | 3/5/2 |
| G1-E81.8-G2 | all10 | Hard-AUC/BA | +0.0650 | 0.9796 | 3/3/4 |
| G1-E81.8-G2 | all10 | Prob-AUC | -0.4882 | 1.1840 | 4/1/5 |
| G1-E81.8-G2 | all10 | SEN-ASD | -0.7256 | 4.3206 | 3/3/4 |
| G1-E81.8-G2 | all10 | SPE-TD | +0.8557 | 2.8882 | 3/5/2 |
| G1-E81.8-G2 | all10 | ASD-F1 | +0.0461 | 1.1367 | 3/3/4 |
| G1-E81.8-G2 | exclude_fold2 | ACC | +0.2569 | 0.7196 | 3/5/1 |
| G1-E81.8-G2 | exclude_fold2 | Hard-AUC/BA | +0.2078 | 0.9287 | 3/3/3 |
| G1-E81.8-G2 | exclude_fold2 | Prob-AUC | -0.5837 | 1.2109 | 3/1/5 |
| G1-E81.8-G2 | exclude_fold2 | SEN-ASD | -0.5352 | 4.5144 | 3/3/3 |
| G1-E81.8-G2 | exclude_fold2 | SPE-TD | +0.9508 | 3.0295 | 3/4/2 |
| G1-E81.8-G2 | exclude_fold2 | ASD-F1 | +0.2021 | 1.0918 | 3/3/3 |

## Fold-Level Metrics

| Fold | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 | Confusion ASD→ASD / ASD→TD / TD→ASD / TD→TD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 94.3182 | 93.9024 | 95.5890 | 87.8049 | 100.0000 | 93.5065 | 36/5/0/47 |
| 2 | 90.8046 | 90.6416 | 93.1071 | 87.8049 | 93.4783 | 90.0000 | 36/5/3/43 |
| 3 | 87.3563 | 87.3807 | 89.2895 | 87.8049 | 86.9565 | 86.7470 | 36/5/6/40 |
| 4 | 94.2529 | 93.9362 | 95.6383 | 90.0000 | 97.8723 | 93.5065 | 36/4/1/46 |
| 5 | 93.1034 | 93.2447 | 91.7553 | 95.0000 | 91.4894 | 92.6829 | 38/2/4/43 |
| 6 | 96.5517 | 96.4362 | 96.5426 | 95.0000 | 97.8723 | 96.2025 | 38/2/1/46 |
| 7 | 90.8046 | 90.0000 | 95.5319 | 80.0000 | 100.0000 | 88.8889 | 32/8/0/47 |
| 8 | 88.5057 | 88.2447 | 88.7766 | 85.0000 | 91.4894 | 87.1795 | 34/6/4/43 |
| 9 | 90.8046 | 90.5585 | 90.5851 | 87.5000 | 93.6170 | 89.7436 | 35/5/3/44 |
| 10 | 93.1034 | 93.0585 | 92.8191 | 92.5000 | 93.6170 | 92.5000 | 37/3/3/44 |

## Patient Corrections

A is historical DARG Best, B is this checkpoint with relation residual disabled, and C is the full CM-PSRR checkpoint. A→B includes both training trajectory and Best-epoch changes. B→C is the same-checkpoint residual intervention.

```json
{
  "G1": {
    "A_to_B": {
      "A_to_B_corrected": 3,
      "A_to_B_harmed": 8,
      "A_to_B_net": -5,
      "A_to_B_asd_corrected": 0,
      "A_to_B_asd_harmed": 6,
      "A_to_B_td_corrected": 3,
      "A_to_B_td_harmed": 2
    },
    "B_to_C": {
      "B_to_C_corrected": 3,
      "B_to_C_harmed": 0,
      "B_to_C_net": 3,
      "B_to_C_asd_corrected": 1,
      "B_to_C_asd_harmed": 0,
      "B_to_C_td_corrected": 2,
      "B_to_C_td_harmed": 0
    },
    "A_to_C": {
      "A_to_C_corrected": 3,
      "A_to_C_harmed": 5,
      "A_to_C_net": -2,
      "A_to_C_asd_corrected": 0,
      "A_to_C_asd_harmed": 5,
      "A_to_C_td_corrected": 3,
      "A_to_C_td_harmed": 0
    }
  }
}
```

The 871 patient-fold rows in `patient_predictions.csv` contain A/B/C probabilities, predictions, true-class margins, margin deltas and correction/harm flags. Exact fold-level correction counts and training/test margin distributions are in `mechanism.csv`.

## Mechanism and Audit

```json
{
  "residual_ratio_mean_by_fold": {
    "mean": 0.07426601685583592,
    "population_sd": 0.017432640012603966
  },
  "residual_ratio_p95_mean_by_fold": {
    "mean": 0.08552703503519296,
    "population_sd": 0.015492022626873995
  },
  "residual_ratio_max_across_folds": 0.09831497818231583,
  "beta": {
    "asd_entropy_mean": 1.3601037621498109,
    "td_entropy_mean": 1.3678298950195313,
    "asd_variance_mean": 0.000721615066868253,
    "td_variance_mean": 0.0007421808229992166
  },
  "relations": {
    "asd_effective_neighbors_mean": 11.874777221679688,
    "td_effective_neighbors_mean": 11.72859468460083,
    "asd_td_support_disagreement_mean": 0.19752200469374656
  },
  "null": {
    "asd_mass_mean": 0.016703389282338322,
    "td_mass_mean": 0.018577876326162368,
    "asd_selected_rate_mean": 0.01490384575445205,
    "td_selected_rate_mean": 0.009026980266207829
  },
  "margins": {
    "test_delta_mean": {
      "mean": 0.04466013684868812,
      "population_sd": 0.028568197424922406
    },
    "test_positive_delta_fraction_mean": 0.78978578892372,
    "train_delta_mean": {
      "mean": 0.05366376843303442,
      "population_sd": 0.033978676472681796
    },
    "train_positive_delta_fraction_mean": 0.8471845922276957,
    "train_crossings_to_correct": 2,
    "train_crossings_to_incorrect": 5,
    "test_crossings_to_correct": 3,
    "test_crossings_to_incorrect": 0
  },
  "interventions": {
    "relation_off_probability_mae_mean": {
      "mean": 0.004994808327409307,
      "population_sd": 0.0022360563534681087
    },
    "prompt_off_probability_mae_mean": {
      "mean": 0.004994808327409307,
      "population_sd": 0.0022360563534681087
    },
    "prompt_off_hard_flips": 3,
    "prompt_off_classification_effect": {
      "corrected": 0,
      "harmed": 3,
      "net": -3
    },
    "uniform_beta_probability_mae_mean": {
      "mean": 0.00024673627678034214,
      "population_sd": 0.00024669812642127064
    },
    "uniform_beta_hard_flips": 0,
    "uniform_beta_classification_effect": {
      "corrected": 0,
      "harmed": 0,
      "net": 0
    },
    "uniform_beta_attention_support_change_rate_mean": 0.10872763192267502,
    "scrambled_group_probability_mae_mean": {
      "mean": 0.005708827233592274,
      "population_sd": 0.0037753334628931523
    },
    "scrambled_group_hard_flips": 5,
    "scrambled_group_classification_effect": {
      "corrected": 0,
      "harmed": 5,
      "net": -5
    },
    "scrambled_asd_support_change_rate_mean": 0.6162723680773249,
    "scrambled_td_support_change_rate_mean": 0.5998571381922675
  },
  "auxiliary_gradients": {
    "mean_classification_norm": 0.029661384871353967,
    "mean_raw_auxiliary_norm": 0.095039759746342,
    "mean_effective_auxiliary_norm": 0.005897999181029834,
    "max_auxiliary_to_classification_ratio": 0.20000003624544985,
    "conflict_projection_records": 2,
    "conflict_projection_fraction_mean": 0.006274509803921568
  }
}
```

`prompt_off` zeros the disease relation contrast and therefore disables the relation message contribution. Its output change cannot identify the independent value of medical text semantics. Uniform beta changed ASD probability by mean MAE 0.000247 with no hard flips. The fixed scrambled Prompt grouping changed probability by mean MAE 0.005709 and caused five hard flips, all harmful relative to the medical grouping. This shows output sensitivity to the tested grouping, while one fixed permutation does not establish general medical-semantic validity. The edge intervention verified that a selected edge becomes zero and its mass moves to Null in the actual forward pass, then clears its override.

Gradient reach, 20% auxiliary budget, Fold2 mask, residual bound, finite tensors, shared RNG, exact OutputLayer call count and zero-residual path were checked in preflight. Best/Latest reload and independent six-metric/prediction recomputation passed 10/10 each. Latest next-step state, optimizer, RNG and loss replay passed 10/10. Checkpoint SHA256 values are in `checkpoint_sha256.txt`.

## Scientific Decision

`NO_GO_PERFORMANCE`. The direct relation residual made 3 net B→C corrections, but A→C remained net -2 and ACC/BA stayed below DARG. Engineering/checkpoint audit passed. The registered performance criteria did not pass, so CM-PSRR is not a GO candidate from this run.

## Protocol Limits

- historical val=test Best selection.
- RFE and standardization precede outer folds.
- transductive patient graph.
- Fold2 retains the historical 1-based rows 377 and 407 overlap, excluded from new auxiliary supervision.
- single seed.

## Outputs

- `fold_metrics.csv` contains independently recomputed metrics and confusion matrices.
- `paired_comparisons.csv` contains paired fold deltas and W/T/L.
- `patient_predictions.csv` contains Best patient-level A/B/C predictions and margins.
- `mechanism.csv` contains fold-level A/B/C, Prompt interventions, relation, residual, gradient and edge-override audits.
- `audit_status.json` contains checkpoint and gate status.
- `checkpoint_sha256.txt` contains all twenty checkpoint hashes.
