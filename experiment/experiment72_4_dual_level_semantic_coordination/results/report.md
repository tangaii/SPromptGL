# Experiment 72.4 - Dual-Level Patient-Disease Semantic Prompt Coordination

Status: **COMPLETE_AUDITED**; independent checkpoint audit: **80/80**; decision: **NO_GO**.

## Source and implementation audit

The locked base is E69.3-G2 at commit `41dbcb45a0273673857051d8e821b8c3315eef00`, using the E61-E2 end-to-end adaptive DARG forward. SPromptGL reports the 320-dimensional GFR, four 36-dimensional CAM positions and 144-dimensional classifier input. BrainPrompt's official code uses participant metadata for patient text and disease class prompts for class supervision. SABER Module III was checked against paper equations (8)-(13); no verifiable public official code was found, so only the paper mechanism was checked.
References: [SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [BrainPrompt MICCAI 2025](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [SABER arXiv](https://arxiv.org/abs/2607.01901), [BrainPrompt code](https://github.com/AngusMonroe/BrainPrompt), [SPromptGL code](https://github.com/wanxixi11/SPromptGL).

## Results

Values are mean +/- population SD across 10 folds (%). Hard-AUC uses hard predictions; Prob-AUC uses continuous TD probabilities.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | 92.0742 +/- 2.7482 | 91.8995 +/- 2.7186 | 93.3759 +/- 2.9908 | 89.5854 +/- 3.4710 | 94.2137 +/- 3.9815 | 91.2984 +/- 2.9222 |
| G1 | 84.3757 +/- 5.3351 | 84.2081 +/- 5.4999 | 87.3364 +/- 4.8221 | 81.6585 +/- 10.5057 | 86.7576 +/- 6.4478 | 82.6037 +/- 6.7844 |
| G2 | 84.2646 +/- 4.9180 | 84.1811 +/- 4.7189 | 87.7632 +/- 5.2873 | 83.1402 +/- 4.0827 | 85.2220 +/- 8.6407 | 83.1744 +/- 4.4034 |
| G3 | 84.1523 +/- 5.0144 | 83.9528 +/- 4.9929 | 87.3200 +/- 4.1821 | 81.4024 +/- 7.4757 | 86.5032 +/- 7.3545 | 82.6105 +/- 5.2853 |
| G4 | 84.0347 +/- 5.9049 | 83.6703 +/- 5.9626 | 86.9823 +/- 5.0375 | 78.9085 +/- 8.2072 | 88.4320 +/- 7.2630 | 82.0036 +/- 6.5547 |
| G5 | 84.6082 +/- 5.4569 | 84.3698 +/- 5.4651 | 86.0542 +/- 7.3586 | 81.1402 +/- 6.6848 | 87.5994 +/- 6.5857 | 82.9933 +/- 5.9877 |
| G6 | 84.2672 +/- 4.5222 | 83.9369 +/- 4.6643 | 86.6474 +/- 3.7601 | 79.2012 +/- 8.2068 | 88.6725 +/- 4.0504 | 82.1647 +/- 5.7910 |
| G7 | 83.8049 +/- 7.1037 | 83.5735 +/- 7.2624 | 87.0951 +/- 5.9464 | 80.1951 +/- 11.2654 | 86.9519 +/- 7.1966 | 81.8682 +/- 8.5240 |
| G8 | 83.6860 +/- 7.1848 | 83.3411 +/- 7.1995 | 87.2739 +/- 5.9288 | 78.9024 +/- 8.2327 | 87.7798 +/- 7.9642 | 81.7497 +/- 7.7148 |

### Paired ACC changes

Positive values favor the left-hand group. W/T/L covers 10 folds; the 9-fold result omits fold 2 and remains descriptive.

| Comparison | Mean change (pp) | 95% paired t CI | W/T/L | 9-fold mean | 9-fold W/T/L |
|---|---:|---:|---:|---:|---:|
| G1_vs_E69.3-G2 | -7.6985 | [-10.5772, -4.8198] | 0/0/10 | -8.0431 | 0/0/9 |
| G8_vs_E69.3-G2 | -8.3882 | [-12.3443, -4.4321] | 0/0/10 | -9.0648 | 0/0/9 |
| G8_minus_G7 | -0.1189 | [-2.0702, +1.8325] | 3/2/5 | -0.3875 | 2/2/5 |
| G3_minus_G2 | -0.1123 | [-1.4215, +1.1968] | 3/1/6 | -0.1248 | 3/0/6 |
| G5_minus_G4 | +0.5734 | [-0.2251, +1.3720] | 6/2/2 | +0.5094 | 5/2/2 |
| G5_minus_G3 | +0.4559 | [-2.1185, +3.0302] | 7/2/1 | +0.1234 | 6/2/1 |
| G7_minus_G6 | -0.4624 | [-3.6621, +2.7374] | 4/3/3 | -0.7692 | 3/3/3 |
| G6_minus_G3 | +0.1149 | [-1.8017, +2.0316] | 3/2/5 | +0.3831 | 3/2/4 |
| G7_minus_G5 | -0.8033 | [-2.6641, +1.0575] | 4/0/6 | -0.5094 | 4/0/5 |
| G8_minus_G1 | -0.6897 | [-3.0538, +1.6745] | 3/4/3 | -1.0217 | 2/4/3 |

### Read-only H1 E72.1 reference

H1 was COMPLETE_AUDITED with 60 tasks and 60 checkpoint audits. Its summary SHA-256 is `52872f8d584b84294d84c4af8c8a3ab78a323c6a8e1da12c44dd7a752151b9b8`. These are historical results on the inherited E69.3 protocol, without the E72.4 patient residual; they were not retrained.

| H1 group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| H1-G1 | 91.6144 +/- 3.0928 | 91.3449 +/- 3.1326 | 93.5796 +/- 3.1493 | 87.6159 +/- 4.4998 | 95.0740 +/- 3.0461 | 90.6150 +/- 3.4682 |
| H1-G2 | 91.8456 +/- 2.8422 | 91.6549 +/- 2.8624 | 93.7021 +/- 2.7258 | 89.0915 +/- 4.0005 | 94.2183 +/- 3.2023 | 91.0000 +/- 3.1090 |
| H1-G3 | 91.9592 +/- 3.0034 | 91.8252 +/- 2.9378 | 93.5612 +/- 2.9438 | 90.3110 +/- 3.9653 | 93.3395 +/- 5.2721 | 91.2679 +/- 3.1143 |
| H1-G4 | 91.9592 +/- 2.8219 | 91.6807 +/- 2.8693 | 93.9136 +/- 2.8305 | 88.0793 +/- 4.0074 | 95.2821 +/- 2.8803 | 91.0072 +/- 3.1472 |
| H1-G5 | 91.8443 +/- 2.5491 | 91.6574 +/- 2.5272 | 93.8600 +/- 2.5634 | 89.3232 +/- 3.7249 | 93.9917 +/- 4.0891 | 91.0367 +/- 2.7219 |
| H1-G6 | 92.0755 +/- 2.5988 | 91.9633 +/- 2.5482 | 93.4338 +/- 3.3954 | 90.5732 +/- 3.1031 | 93.3534 +/- 4.1622 | 91.3944 +/- 2.7034 |

Descriptive ACC differences from the corresponding H1 configuration are:
- E72.4 G3 minus H1 G1: -7.4621 pp.
- E72.4 G6 minus H1 G2: -7.5784 pp.
- E72.4 G5 minus H1 G3: -7.3511 pp.
- E72.4 G7 minus H1 G4: -8.1544 pp.
- E72.4 G2 minus H1 G5: -7.5797 pp.

H1-G6 (BrainPrompt-2, tau=0.1, global-only) has no exact E72.4 group match; E72.4-G8 also has no H1 counterpart because it adds the disease-conditioned patient query. These cross-experiment differences include the E72.4 patient prompt and training-path changes and are not single-factor causal estimates.


## Mechanism findings

Patient prompts used PHENO-recoverable age intervals, recorded sex category codes, and site IDs. All three fields were present for 871/871 rows. Frozen CLIP RN50 encoded three 1024-dimensional tokens per patient once, aligned by source row. The loader read PHENO columns only and did not read diagnosis labels. It generated no exact ages, VIQ/PIQ, symptoms, or unmeasured clinical facts.

Across selected checkpoints, disabling the patient residual caused 1948 held-out prediction flips. The mean active-minus-disabled ACC difference across group-folds was +13.6031 pp. Activating the residual corrected 1448 cases and introduced 500 errors. These fixed-checkpoint interventions establish model sensitivity, not clinical meaning.

G8 vs G7 ACC changed -0.1189 pp (3/2/5); G8 vs E69.3-G2 changed -8.3882 pp. Disabling the G8 disease-conditioned query corrected 7 cases and introduced 2; swapping prototype order changed 0 predictions. Both ASD and TD prototypes enter as soft probabilities; no true test label is used for routing.

G8's test-time condition removal changed 9 hard predictions in 10 folds (7 corrections, 2 new errors). Swapping both prototype rows is permutation-invariant for the softmax-weighted prototype sum by construction; its zero hard flips are an implementation invariance check, not evidence that disease conditions are unused. The active condition branch has nonzero classification gradient in all folds.

Across OOF patients G8 made 39 corrections and 40 new errors versus G7. Attention-change associations are descriptive only: `{"age": {"pearson_r_vs_G8_corrections": -0.007152954686067804, "p_descriptive": 0.8330413043660875}, "sex": {"pearson_r_vs_G8_corrections": 0.04343918676740127, "p_descriptive": 0.20027390219306082}, "site": {"pearson_r_vs_G8_corrections": -0.03516135811988481, "p_descriptive": 0.29995158299315805}}`.

Patient tokens received mean attention `0.180` age, `0.355` sex, and `0.465` site across group-fold checkpoints; site attention was highest. The mean residual ratio was `0.0997` with mean p95 `0.1000`, so most subjects reached the fixed 10% cap. Site-token masking (masked minus active) changed G8 ACC by `-2.75` pp on average; the negative change means masking lowered ACC. For sites with at least 20 G8 OOF subjects, the largest site was NYU (n=172); accuracy ranged from MAX_MUN 71.74% (n=46) to LEUVEN_2 96.43% (n=28). These figures indicate site information materially affects predictions and leave a site-confounding risk.

G8 OOF sex-code accuracies were 1=83.91% (n=727) and 2=82.64% (n=144). For age bins with n>=20, accuracy ranged from AGE_AT_SCAN_[28.47, 30.47) 70.83% (n=24) to AGE_AT_SCAN_[26.47, 28.47) 100.00% (n=25); smaller bins are listed in `subgroup_diagnostics.csv` and are noisy.

Held-out disease-head accuracy at G8 was 80.24% global and `{"PHENO": 72.20872521400452, "ANAT": 71.51515245437622, "FUNC": 64.7452974319458, "CORR": 80.93129515647888}` across the four CAM positions. At the best checkpoints, G8 mean gradient norms were CLS/semantic `0.00443/0.00939` at CAM Z and `0.08721/0.10392` at DARG F; patient CLS/semantic parameter gradient norms were `0.5411/0.6967` and condition-query CLS gradient norm was `0.00738`. Alpha was calibrated at the predeclared 0.1 CAM-gradient ratio, ranged from `0.0282` to `1.8386`, and remained frozen. At the selected G8 checkpoints the weighted CAM-gradient ratio averaged `0.083` and the weighted F-gradient ratio was approximately `0.050`; this is not a constant 10% through training. Gradient cosine means were Z `0.376`, F `0.889`, and patient module `0.935`.

The fold-averaged G8 adaptive gates were `[0.000998, 0.00099, 0.000979, 0.990848]`; changes versus the matched E69.3 final gates were `[3.7e-05, 1.3e-05, 3.9e-05, -0.000919]` in PHENO/ANAT/FUNC/CORR order. These near-initial values do not show a material DARG relation-learning shift. The four CAM positions are interaction-processed modality-related features, not four independent raw modalities or localized brain lesions.

At tau=1, adding fine-grained supervision changed ACC by G3 minus G2 `-0.112` pp for BrainPrompt-2 and G5 minus G4 `+0.573` pp for SPromptGL-8. At tau=1 with fine+global supervision, SPromptGL-8 minus BrainPrompt-2 was `+0.456` pp; at tau=0.1 it was G7 minus G6 `-0.462` pp. Temperature changes were G6 minus G3 `+0.115` pp for BrainPrompt and G7 minus G5 `-0.803` pp for SPromptGL. Their direction varies by metric and text source. The two CE forms passed numerical/gradient equivalence; tau=0.1 is this experiment's adaptation, not a fixed SABER paper value. No parameter-matched label-only CE or numeric-PHENO control was trained, so semantic-text gains cannot be separated from added supervision or capacity.

Adaptive/edge gate measurements are diagnostic; they do not establish improved relational learning. No additional relation architecture or loss was introduced.

### Fold 2 sensitivity

Row 377 (ASD) and row 407 (TD) remain in the fold-2 test set and are both added to effective training. Their labels enter classification in every group and disease supervision/alpha calibration in G2-G8.

| Model | N full | N without | ACC full | ACC without | Hard-AUC full | Hard-AUC without | Prob-AUC full | Prob-AUC without |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | 87 | 85 | 90.8046 | 90.5882 | 90.6416 | 90.4167 | 92.3648 | 92.3333 |
| G1 | 87 | 85 | 86.2069 | 85.8824 | 86.4263 | 86.1111 | 88.2291 | 88.0000 |
| G2 | 87 | 85 | 86.2069 | 85.8824 | 86.0286 | 85.6944 | 89.6607 | 89.5556 |
| G3 | 87 | 85 | 86.2069 | 87.0588 | 86.5589 | 87.3611 | 88.6002 | 88.6667 |
| G4 | 87 | 85 | 88.5057 | 88.2353 | 88.3351 | 88.0556 | 90.7211 | 90.5556 |
| G5 | 87 | 85 | 89.6552 | 89.4118 | 89.2895 | 89.0278 | 91.9406 | 91.8333 |
| G6 | 87 | 85 | 83.9080 | 83.5294 | 83.7222 | 83.3333 | 85.4719 | 85.5556 |
| G7 | 87 | 85 | 86.2069 | 85.8824 | 86.0286 | 85.6944 | 89.5546 | 89.3889 |
| G8 | 87 | 85 | 88.5057 | 88.2353 | 88.4677 | 88.1944 | 90.8802 | 90.7222 |

| Model | SEN-ASD full | SEN-ASD without | SPE-TD full | SPE-TD without | ASD-F1 full | ASD-F1 without |
|---|---:|---:|---:|---:|---:|---:|
| E69.3-G2 | 87.8049 | 87.5000 | 93.4783 | 93.3333 | 90.0000 | 89.7436 |
| G1 | 90.2439 | 90.0000 | 82.6087 | 82.2222 | 86.0465 | 85.7143 |
| G2 | 82.9268 | 82.5000 | 89.1304 | 88.8889 | 85.0000 | 84.6154 |
| G3 | 92.6829 | 92.5000 | 80.4348 | 82.2222 | 86.3636 | 87.0588 |
| G4 | 85.3659 | 85.0000 | 91.3043 | 91.1111 | 87.5000 | 87.1795 |
| G5 | 82.9268 | 82.5000 | 95.6522 | 95.5556 | 88.3117 | 88.0000 |
| G6 | 80.4878 | 80.0000 | 86.9565 | 86.6667 | 82.5000 | 82.0513 |
| G7 | 82.9268 | 82.5000 | 89.1304 | 88.8889 | 85.0000 | 84.6154 |
| G8 | 87.8049 | 87.5000 | 89.1304 | 88.8889 | 87.8049 | 87.5000 |

Removing those two rows does not remove other limitations: preprocessing precedes CV, validation equals test for checkpoint choice, and the graph is transductive over 871 patients.

Independent reconstruction/readback: 80/80. Common-operation RNG checks and protected source/data/history hashes are in `audit_status.json`; protected snapshot status is PASS. Optimizer parameter groups and state tensors were compared directly after reloading; task optimizer hashes pickle device-specific tensors and are not used as cross-device equality checks. No historical model, root model file, or input data was altered by this experiment.

Audit-reader provenance: `After training, independent readback corrections aligned the reader with the flat prefixed module-state schema, compared optimizer state trees directly because task optimizer digests pickle device-specific tensors, made paired contrast directions explicit, and added the historical fold-2 sensitivity checkpoint. All corrections affect only independent audit/reporting code; training code, configuration, caches, and checkpoints remain byte-identical.` Frozen training/config hashes remain the checkpoint provenance.

## Decision

**NO_GO** under the predeclared candidate rule. G8 minus G7 ACC is -0.1189 pp; G8 minus E69.3-G2 is -8.3882 pp. G8 does not beat either model on mean ACC, so the cross-layer mechanism is NO-GO as the next MICCAI module candidate, despite mixed secondary metric/component effects. The eight groups are paired comparisons within a patient-prompt background, not a complete factorial design. Any future candidate still requires no overlap, fold-local preprocessing, and validation-only checkpoint selection.

Artifacts: `summary.csv`, `fold_metrics.csv`, `fold2_sensitivity.csv`, `paired_comparisons.csv`, `mechanism.csv`, `subgroup_diagnostics.csv`, `checkpoint_audit.csv`, and `audit_status.json`.
