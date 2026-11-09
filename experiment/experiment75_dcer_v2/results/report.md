# Experiment 75: DCER-v2

## Execution
- Tasks 40/40; independent audits 40/40.
- Runtime /model/spromptgl/experiment/experiment75_dcer_v2; preferred OSSFS path failed with Input/output error, so fallback was used.
- Device cuda:0; worker cap 10; recorded task failures 0.
- Every fold starts from its E61 epoch-10 audited warmup; E69.3-G2 is read-only.

## Source and method audit
- E69.3-G2 uses the audited E53 PLNNoPEN. DARG 320-D features enter the CAM; its final 144-D latent representation is the input to OutputLayer. DCER hooks that boundary, retaining one CAM forward and one classifier path; it adds no logit head.
- The root SPromptGL PLN has PEN-like text selection/RES operations. E69 uses PLNNoPEN. view(N,4,4,9) is a latent partition, not anatomical ROI localization.
- SPromptGL: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf . BrainPrompt disease-level supervision uses normalized patient graph/text prototype similarity cross-entropy; its paper uses Llama/LLM2Vec representations, while E75 uses cached CLIP RN50 SPromptGL prompts: https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf .
- SABER Module III is read as patient-conditioned semantic queries, cross-attention, gated residual and disease supervision; no verified official source code was found: https://arxiv.org/abs/2607.01901 .
- Differential Transformer subtracts two Softmax attention maps to suppress common attention noise. DCER borrows only this idea and does not reproduce the architecture: https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html .
- Frozen RN50 cache, all eight original prompt strings/order, and the anonymous prototype geometry were SHA/geometry checked.

## Classification results
Mean ± population SD, percent. Hard-AUC is historical hard-prediction AUC (binary BA); Probability-AUC uses continuous ASD probability.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E69.3-G2 historical | 92.0742 ± 2.7482 | 91.8995 ± 2.7186 | 93.3759 ± 2.9908 | 89.5854 ± 3.4710 | 94.2137 ± 3.9815 | 91.2984 ± 2.9222 |
| G1 | 91.6144 ± 3.0062 | 91.3972 ± 3.0258 | 93.8611 ± 2.9450 | 88.5854 ± 4.8896 | 94.2091 ± 4.4226 | 90.7183 ± 3.2919 |
| G2 | 91.6144 ± 2.6312 | 91.4205 ± 2.6202 | 93.5004 ± 2.8072 | 88.8354 ± 3.7240 | 94.0056 ± 3.6924 | 90.7531 ± 2.8514 |
| G3 | 91.8443 ± 2.7963 | 91.6085 ± 2.8653 | 93.3755 ± 3.3199 | 88.5732 ± 4.4990 | 94.6438 ± 2.9422 | 90.9245 ± 3.1369 |
| G4 | 91.9592 ± 2.7747 | 91.7108 ± 2.8262 | 93.3121 ± 2.8603 | 88.3476 ± 4.7048 | 95.0740 ± 3.4633 | 91.0281 ± 3.1193 |

### Excluding Fold 2 (descriptive only)

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|---:|
| E69.3-G2 historical | 92.2153 ± 2.8623 | 92.0393 ± 2.8314 | 93.4883 ± 3.1325 | 89.7832 ± 3.6049 | 94.2954 ± 4.1889 |
| G1 | 91.7044 ± 3.1560 | 91.4664 ± 3.1819 | 93.9213 ± 3.0984 | 88.4011 ± 5.1210 | 94.5318 ± 4.5487 |
| G2 | 91.7044 ± 2.7589 | 91.5070 ± 2.7484 | 93.6561 ± 2.9178 | 88.9499 ± 3.9087 | 94.0641 ± 3.8877 |
| G3 | 91.8321 ± 2.9473 | 91.5804 ± 3.0190 | 93.4819 ± 3.4833 | 88.3875 ± 4.7059 | 94.7734 ± 3.0741 |
| G4 | 92.0875 ± 2.8965 | 91.8296 ± 2.9553 | 93.3996 ± 3.0023 | 88.4079 ± 4.9556 | 95.2513 ± 3.6074 |

## Paired comparisons

Delta = group A minus group B, percentage points; W/T/L are fold counts. Historical comparisons are descriptive.

| Comparison | Scope | Metric | Mean delta pp | W/T/L |
|---|---|---|---:|---:|
| G1 - E69.3-G2 historical | all_10 | acc | -0.4598 | 0/6/4 |
| G1 - E69.3-G2 historical | all_10 | hard_auc | -0.5023 | 1/5/4 |
| G1 - E69.3-G2 historical | all_10 | prob_auc | +0.4852 | 5/1/4 |
| G1 - E69.3-G2 historical | all_10 | sen | -1.0000 | 1/6/3 |
| G1 - E69.3-G2 historical | exclude_fold2_descriptive_9 | acc | -0.5109 | 0/5/4 |
| G1 - E69.3-G2 historical | exclude_fold2_descriptive_9 | hard_auc | -0.5729 | 0/5/4 |
| G1 - E69.3-G2 historical | exclude_fold2_descriptive_9 | prob_auc | +0.4331 | 4/1/4 |
| G1 - E69.3-G2 historical | exclude_fold2_descriptive_9 | sen | -1.3821 | 0/6/3 |
| G2 - E69.3-G2 historical | all_10 | acc | -0.4598 | 0/6/4 |
| G2 - E69.3-G2 historical | all_10 | hard_auc | -0.4791 | 1/4/5 |
| G2 - E69.3-G2 historical | all_10 | prob_auc | +0.1245 | 4/1/5 |
| G2 - E69.3-G2 historical | all_10 | sen | -0.7500 | 2/4/4 |
| G2 - E69.3-G2 historical | exclude_fold2_descriptive_9 | acc | -0.5109 | 0/5/4 |
| G2 - E69.3-G2 historical | exclude_fold2_descriptive_9 | hard_auc | -0.5323 | 1/3/5 |
| G2 - E69.3-G2 historical | exclude_fold2_descriptive_9 | prob_auc | +0.1678 | 4/1/4 |
| G2 - E69.3-G2 historical | exclude_fold2_descriptive_9 | sen | -0.8333 | 2/3/4 |
| G3 - E69.3-G2 historical | all_10 | acc | -0.2299 | 1/6/3 |
| G3 - E69.3-G2 historical | all_10 | hard_auc | -0.2910 | 2/3/5 |
| G3 - E69.3-G2 historical | all_10 | prob_auc | -0.0005 | 7/0/3 |
| G3 - E69.3-G2 historical | all_10 | sen | -1.0122 | 2/3/5 |
| G3 - E69.3-G2 historical | exclude_fold2_descriptive_9 | acc | -0.3831 | 0/6/3 |
| G3 - E69.3-G2 historical | exclude_fold2_descriptive_9 | hard_auc | -0.4589 | 1/3/5 |
| G3 - E69.3-G2 historical | exclude_fold2_descriptive_9 | prob_auc | -0.0064 | 6/0/3 |
| G3 - E69.3-G2 historical | exclude_fold2_descriptive_9 | sen | -1.3957 | 1/3/5 |
| G4 - E69.3-G2 historical | all_10 | acc | -0.1149 | 0/9/1 |
| G4 - E69.3-G2 historical | all_10 | hard_auc | -0.1887 | 0/7/3 |
| G4 - E69.3-G2 historical | all_10 | prob_auc | -0.0639 | 7/0/3 |
| G4 - E69.3-G2 historical | all_10 | sen | -1.2378 | 0/7/3 |
| G4 - E69.3-G2 historical | exclude_fold2_descriptive_9 | acc | -0.1277 | 0/8/1 |
| G4 - E69.3-G2 historical | exclude_fold2_descriptive_9 | hard_auc | -0.2097 | 0/6/3 |
| G4 - E69.3-G2 historical | exclude_fold2_descriptive_9 | prob_auc | -0.0886 | 6/0/3 |
| G4 - E69.3-G2 historical | exclude_fold2_descriptive_9 | sen | -1.3753 | 0/6/3 |
| G2 - G1 | all_10 | acc | +0.0000 | 1/8/1 |
| G2 - G1 | all_10 | hard_auc | +0.0232 | 4/3/3 |
| G2 - G1 | all_10 | prob_auc | -0.3607 | 3/1/6 |
| G2 - G1 | all_10 | sen | +0.2500 | 3/4/3 |
| G2 - G1 | exclude_fold2_descriptive_9 | acc | +0.0000 | 1/7/1 |
| G2 - G1 | exclude_fold2_descriptive_9 | hard_auc | +0.0406 | 4/3/2 |
| G2 - G1 | exclude_fold2_descriptive_9 | prob_auc | -0.2653 | 3/1/5 |
| G2 - G1 | exclude_fold2_descriptive_9 | sen | +0.5488 | 3/4/2 |
| G3 - G2 | all_10 | acc | +0.2299 | 2/8/0 |
| G3 - G2 | all_10 | hard_auc | +0.1881 | 3/5/2 |
| G3 - G2 | all_10 | prob_auc | -0.1250 | 6/0/4 |
| G3 - G2 | all_10 | sen | -0.2622 | 3/5/2 |
| G3 - G2 | exclude_fold2_descriptive_9 | acc | +0.1277 | 1/8/0 |
| G3 - G2 | exclude_fold2_descriptive_9 | hard_auc | +0.0734 | 2/5/2 |
| G3 - G2 | exclude_fold2_descriptive_9 | prob_auc | -0.1742 | 5/0/4 |
| G3 - G2 | exclude_fold2_descriptive_9 | sen | -0.5623 | 2/5/2 |
| G3 - G4 | all_10 | acc | -0.1149 | 1/7/2 |
| G3 - G4 | all_10 | hard_auc | -0.1023 | 3/4/3 |
| G3 - G4 | all_10 | prob_auc | +0.0634 | 6/0/4 |
| G3 - G4 | all_10 | sen | +0.2256 | 3/5/2 |
| G3 - G4 | exclude_fold2_descriptive_9 | acc | -0.2554 | 0/7/2 |
| G3 - G4 | exclude_fold2_descriptive_9 | hard_auc | -0.2491 | 2/4/3 |
| G3 - G4 | exclude_fold2_descriptive_9 | prob_auc | +0.0822 | 6/0/3 |
| G3 - G4 | exclude_fold2_descriptive_9 | sen | -0.0203 | 2/5/2 |

## Mechanism results
- G1: gamma -0.00059 ± 0.01794; residual/CAM norm 0.67692% ± 0.26374; token-attention absolute difference 0.10920; common-high fraction 0.00000.
- G2: gamma -0.00127 ± 0.02051; residual/CAM norm 0.77036% ± 0.28227; token-attention absolute difference 0.10890; common-high fraction 0.00626.
- G3: gamma -0.00043 ± 0.01718; residual/CAM norm 0.15850% ± 0.18147; token-attention absolute difference 0.10197; common-high fraction 0.01147.
- G4: gamma -0.00216 ± 0.01256; residual/CAM norm 0.10168% ± 0.04770; token-attention absolute difference 0.10296; common-high fraction 0.01940.
- G3 validity: training-only teacher correlation 0.12412 ± 0.15344; high-vs-low validity quartile teacher contribution +0.06431.
- G4 validity: training-only teacher correlation 0.10817 ± 0.08436; high-vs-low validity quartile teacher contribution +0.04572.
- direct_off: 0 hard-label flips; wrong-to-correct 0, correct-to-wrong 0 across 40 checkpoints.
- semantic_neutral: 0 hard-label flips; wrong-to-correct 0, correct-to-wrong 0 across 40 checkpoints.
- common_off: 0 hard-label flips; wrong-to-correct 0, correct-to-wrong 0 across 30 checkpoints.
- validity_off: 1 hard-label flips; wrong-to-correct 0, correct-to-wrong 1 across 20 checkpoints.

## Decision and limits
Decision: NO-GO. This is the prespecified evidence category, not a significance claim.

Historical test-fold ACC checkpoint selection (val=test), pre-CV RFE/standardization, and the 871-node transductive graph were retained. Fold 2 trains on test rows 377 (ASD) and 407 (TD). Ten-fold results are not unbiased independent generalization estimates; excluding Fold 2 is descriptive only.

The 9-D tokens are latent feature partitions, not anatomical regions. Attention and validity are model-internal predictive quantities, not clinical or causal biomarkers. The masking teacher measures sensitivity of the current model. G3 versus G4 is the test of whether the supplied medical text contributes beyond matched prototype geometry.

## Audit
- Independent checkpoint audits: 40/40 PASS.
- Exact RNG checks against E69 shared epochs: 14314 epoch-fold matches.
- Fold 2 label exposure: rows 377 and 407 only; all other folds have no train/test overlap.
- Report: /model/spromptgl/experiment/experiment75_dcer_v2/results/report.md; manifest: /model/spromptgl/experiment/experiment75_dcer_v2/manifest.json.

## Historical experiment review

- E71 GCDER tested dual ASD/TD disease queries and decision-level logit adjustment, with best ACC 89.0857%; DCER-v2 instead modifies the 144-D CAM feature input and does not add disease evidence to logits.
- E59 SHAPE-PEN and E70/E70.1 semantic/conditioned residual studies did not establish a stable improvement over their matched base. E70.1's bounded residual mechanism passed its numerical mechanism checks, while the registered base-performance and independent-text criteria failed.
- E72.1/72.2 disease-level supervision was diagnostically negative for a general gain; E72.4 dual-level patient/disease prompting was NO-GO and showed substantial site dependence; E72.5 reduced residual magnitude but remained a partial residual-safety repair, without proving an overall medical-text benefit.
- These earlier results motivated testing local feature competition with a matched anonymous prototype. In E75, G4 exceeded text-based G3 by 0.1149 pp ACC (W/T/L 1/7/2), so an independent medical-text benefit was not demonstrated.

## Audit correction

The append-only handoff initially copied the pre-audit manifest value 0/40. Independent reconstruction completed successfully for 40/40 checkpoints; the corrected count is 40/40. This reporting correction does not alter training, predictions, or checkpoint files.
