# Experiment 83.7 Final Report

## Status
COMPLETE_AUDITED after formal execution and independent replay.

## Protocol
598 TADPOLE patients, six modalities, seed=0, ten fixed stratified folds, val_idx=test_idx, pre-CV RFE and historical ACC selection were retained.

## Results

| group   | status   |   folds |   ACC_mean |   ACC_std_population |   AUC_original_mean |   AUC_original_std_population |   AUC_probability_mean |   AUC_probability_std_population |   OOF_correct |   NC_correct |   MCI_correct |   AD_correct | source                             |
|:--------|:---------|--------:|-----------:|---------------------:|--------------------:|------------------------------:|-----------------------:|---------------------------------:|--------------:|-------------:|--------------:|-------------:|:-----------------------------------|
| G1      | COMPLETE |      10 |   0.871130 |             0.039345 |            0.906485 |                      0.030545 |               0.938880 |                         0.020794 |           521 |          184 |           269 |           68 | E83.5 historical checkpoint replay |
| G2      | COMPLETE |      10 |   0.887797 |             0.039571 |            0.907319 |                      0.028838 |               0.953951 |                         0.022256 |           531 |          185 |           283 |           63 | E83.5 historical checkpoint replay |
| G3      | COMPLETE |      10 |   0.924746 |             0.031913 |            0.939800 |                      0.027433 |               0.955545 |                         0.030104 |           553 |          205 |           283 |           65 | E83.5 historical checkpoint replay |
| G4      | COMPLETE |      10 |   0.929746 |             0.027816 |            0.940967 |                      0.026088 |               0.962710 |                         0.022532 |           556 |          205 |           287 |           64 | E83.5 historical checkpoint replay |
| G5      | COMPLETE |      10 |   0.854407 |             0.064461 |            0.883072 |                      0.047604 |               0.907601 |                         0.043304 |           511 |          188 |           263 |           60 | E83.5 historical checkpoint replay |
| T1      | COMPLETE |      10 |   0.928051 |             0.032667 |            0.940898 |                      0.022180 |               0.963915 |                         0.024273 |           555 |          203 |           287 |           65 | E83.7 formal training              |
| T2      | COMPLETE |      10 |   0.918079 |             0.035278 |            0.934221 |                      0.029728 |               0.952155 |                         0.034104 |           549 |          202 |           282 |           65 | E83.7 formal training              |
| T3      | COMPLETE |      10 |   0.926384 |             0.034433 |            0.937840 |                      0.032420 |               0.952884 |                         0.026078 |           554 |          204 |           286 |           64 | E83.7 formal training              |
| T4      | COMPLETE |      10 |   0.919689 |             0.033345 |            0.934678 |                      0.031796 |               0.957000 |                         0.019312 |           550 |          203 |           283 |           64 | E83.7 formal training              |
| T5      | COMPLETE |      10 |   0.919718 |             0.037989 |            0.934540 |                      0.031660 |               0.961596 |                         0.024350 |           550 |          200 |           285 |           65 | E83.7 formal training              |
| T6      | COMPLETE |      10 |   0.923079 |             0.029123 |            0.932263 |                      0.025615 |               0.948783 |                         0.030557 |           552 |          204 |           286 |           62 | E83.7 formal training              |

The principal metrics are ten-fold mean ± population standard deviation. `AUC_original` is the hard one-hot macro OVR definition used by E83.5/E83.6.

## Paired contrasts

| reference   | candidate   | fold   |   ACC_reference |   ACC_candidate |   AUC_original_reference |   AUC_original_candidate |   ACC_delta |   AUC_original_delta |   correct_delta |   patient_corrected |   patient_damaged |   patient_net |
|:------------|:------------|:-------|----------------:|----------------:|-------------------------:|-------------------------:|------------:|---------------------:|----------------:|--------------------:|------------------:|--------------:|
| G4          | T1          | MEAN   |        0.929746 |        0.928051 |                 0.940967 |                 0.940898 |   -0.001695 |            -0.000069 |              -1 |                  23 |                24 |            -1 |
| G4          | T2          | MEAN   |        0.929746 |        0.918079 |                 0.940967 |                 0.934221 |   -0.011667 |            -0.006745 |              -7 |                  24 |                31 |            -7 |
| G4          | T4          | MEAN   |        0.929746 |        0.919689 |                 0.940967 |                 0.934678 |   -0.010056 |            -0.006288 |              -6 |                  24 |                30 |            -6 |
| G4          | T6          | MEAN   |        0.929746 |        0.923079 |                 0.940967 |                 0.932263 |   -0.006667 |            -0.008703 |              -4 |                  22 |                26 |            -4 |
| G3          | T3          | MEAN   |        0.924746 |        0.926384 |                 0.939800 |                 0.937840 |    0.001638 |            -0.001960 |               1 |                  20 |                19 |             1 |
| G3          | T5          | MEAN   |        0.924746 |        0.919718 |                 0.939800 |                 0.934540 |   -0.005028 |            -0.005260 |              -3 |                  18 |                21 |            -3 |
| T3          | T4          | MEAN   |        0.926384 |        0.919689 |                 0.937840 |                 0.934678 |   -0.006695 |            -0.003161 |              -4 |                  19 |                23 |            -4 |
| T5          | T6          | MEAN   |        0.919718 |        0.923079 |                 0.934540 |                 0.932263 |    0.003362 |            -0.002277 |               2 |                  21 |                19 |             2 |

## Mechanism
{
  "G4": {
    "native_full_logit_max_abs": 0.2320035696029663,
    "native_full_flip_count": 0.0,
    "native_full_corrected": 0.0,
    "native_full_damaged": 0.0,
    "native_top1_top2_margin_mean": 2.607584607601166,
    "full_top1_top2_margin_mean": 2.7455440759658813,
    "residual_off_native_max_abs": 0.0,
    "dpsl_residual_max": 0.05867824070155621,
    "direct_off": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "disease_neutral": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "native_class_correct": [
      205,
      287,
      64
    ],
    "full_class_correct": [
      205,
      287,
      64
    ],
    "direct_off_class_correct": [
      205,
      287,
      64
    ],
    "disease_neutral_class_correct": [
      205,
      287,
      64
    ],
    "dpsl_shared_cap_configured": null,
    "dpsl_shared_cap_observed_max": null,
    "dpsl_shared_gradient_ratio_max": null,
    "graph_message_relative_norm_mean_by_modality": [
      0.022083045536419377,
      0.006951930167269893,
      0.0008387659538367842,
      0.0001762788144573424,
      0.00028950172664738983,
      0.0011764995124572408
    ]
  },
  "T1": {
    "native_full_logit_max_abs": 0.21842548847198487,
    "native_full_flip_count": 0.2,
    "native_full_corrected": 0.1,
    "native_full_damaged": 0.1,
    "native_top1_top2_margin_mean": 2.723576307296753,
    "full_top1_top2_margin_mean": 2.875433695316315,
    "residual_off_native_max_abs": 0.0,
    "dpsl_residual_max": 0.05677795452065766,
    "direct_off": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "disease_neutral": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "native_class_correct": [
      204,
      286,
      65
    ],
    "full_class_correct": [
      203,
      287,
      65
    ],
    "direct_off_class_correct": [
      203,
      287,
      65
    ],
    "disease_neutral_class_correct": [
      203,
      287,
      65
    ],
    "dpsl_shared_cap_configured": 0.003,
    "dpsl_shared_cap_observed_max": 0.003,
    "dpsl_shared_gradient_ratio_max": 0.0029998909304454046,
    "graph_message_relative_norm_mean_by_modality": [
      0.015221555763855577,
      0.007469205249799415,
      0.0006483806025698868,
      0.0009765289870728111,
      0.0008141615133752111,
      0.002109646981989499
    ]
  },
  "T2": {
    "native_full_logit_max_abs": 0.1822864294052124,
    "native_full_flip_count": 0.0,
    "native_full_corrected": 0.0,
    "native_full_damaged": 0.0,
    "native_top1_top2_margin_mean": 2.458974289894104,
    "full_top1_top2_margin_mean": 2.5802221417427065,
    "residual_off_native_max_abs": 0.0,
    "dpsl_residual_max": 0.0394688248867169,
    "direct_off": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "disease_neutral": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "native_class_correct": [
      202,
      282,
      65
    ],
    "full_class_correct": [
      202,
      282,
      65
    ],
    "direct_off_class_correct": [
      202,
      282,
      65
    ],
    "disease_neutral_class_correct": [
      202,
      282,
      65
    ],
    "dpsl_shared_cap_configured": 0.009999999999999998,
    "dpsl_shared_cap_observed_max": 0.009999999999999998,
    "dpsl_shared_gradient_ratio_max": 0.009999888609999365,
    "graph_message_relative_norm_mean_by_modality": [
      0.015835825819522142,
      0.007320939068449661,
      0.0016427562119588402,
      0.0019198674142813844,
      0.0031456396291559942,
      0.001828665587163414
    ]
  },
  "T4": {
    "native_full_logit_max_abs": 0.144073486328125,
    "native_full_flip_count": 0.3,
    "native_full_corrected": 0.3,
    "native_full_damaged": 0.0,
    "native_top1_top2_margin_mean": 2.31642507314682,
    "full_top1_top2_margin_mean": 2.414489817619324,
    "residual_off_native_max_abs": 0.0,
    "dpsl_residual_max": 0.03631666873698123,
    "direct_off": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "disease_neutral": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "native_class_correct": [
      203,
      280,
      64
    ],
    "full_class_correct": [
      203,
      283,
      64
    ],
    "direct_off_class_correct": [
      203,
      283,
      64
    ],
    "disease_neutral_class_correct": [
      203,
      283,
      64
    ],
    "dpsl_shared_cap_configured": 0.0010000000000000002,
    "dpsl_shared_cap_observed_max": 0.0010000000000000002,
    "dpsl_shared_gradient_ratio_max": 0.0009998908080380185,
    "graph_message_relative_norm_mean_by_modality": [
      0.01321071465499699,
      0.0056369306112173945,
      0.0014526729664567028,
      0.002345239710169267,
      0.002523548580919055,
      0.002025912982958289
    ]
  },
  "T6": {
    "native_full_logit_max_abs": 0.18998425006866454,
    "native_full_flip_count": 0.0,
    "native_full_corrected": 0.0,
    "native_full_damaged": 0.0,
    "native_top1_top2_margin_mean": 2.9393490076065065,
    "full_top1_top2_margin_mean": 3.041047763824463,
    "residual_off_native_max_abs": 0.0,
    "dpsl_residual_max": 0.04854614990763366,
    "direct_off": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "disease_neutral": {
      "changed": 0,
      "corrected": 0,
      "damaged": 0
    },
    "native_class_correct": [
      204,
      286,
      62
    ],
    "full_class_correct": [
      204,
      286,
      62
    ],
    "direct_off_class_correct": [
      204,
      286,
      62
    ],
    "disease_neutral_class_correct": [
      204,
      286,
      62
    ],
    "dpsl_shared_cap_configured": 0.0010000000000000002,
    "dpsl_shared_cap_observed_max": 0.0010000000000000002,
    "dpsl_shared_gradient_ratio_max": 0.0009998881077362637,
    "graph_message_relative_norm_mean_by_modality": [
      0.03503416597377509,
      0.020626165530120488,
      0.003404577649348539,
      0.0022893134649751624,
      0.004351603460430909,
      0.004594371422274435
    ]
  }
}

T3/T4 warmup pairing passed for 10/10 folds. Graph supervision was positive during epochs 1–10 and zero in formal training after epoch 10. T5/T6 audits record WD 0.11 in warmup and 0.055 from epoch 11.

## Audits
Historical replay 50/50 Best and Latest folds passed. New replay 60/60 Best and Latest folds passed. Warmup pairing 10/10 passed.

## Limitations
The registered val=test selection and pre-CV full-data RFE remain generalized-evaluation limitations. Seed=0 and the comparison of multiple registered variants do not establish unbiased generalization. Historical G2 is described in the E83.7 fidelity document as PEN-only removal while retaining GFR, GraphLearn, GCN, message passing and graph losses. G5 is the physical CAM-only group.

## Reference
Original TADPOLE reference is 94.81% ACC and 95.59% AUC. These values were not used to select folds or change metric definitions.
