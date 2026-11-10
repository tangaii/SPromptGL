# Experiment 76 — DCMP

Status: independently audited after reconstructing all best checkpoints from disk.

## Literature and source audit

SPromptGL (MICCAI 2025) was checked against its paper and locked E74.4-G2 implementation. The experiment retains four-modality GFR/DARG, a single CAM pass, the 144-dimensional OutputLayer input, and the original classifier. BrainPrompt's official ABIDE labels were verified in source order TC then ASD, then mapped into the audited cache order ASD=0, TD=1; the frozen E72.4 CLIP RN50 cache is reused, so this is not a full BrainPrompt encoder reproduction. PGP-SAM motivates class-conditioned inter-/intra-class prototypes and prompt generation; its SAM segmentation decoder is not used. CoW motivates training-support self-prediction and hard prototypes; its image-mask boundary/segmentation machinery is not used. HP²L's 2026 Medical Image Analysis article describes hierarchical prompt/prototype learning for brain-disorder classification. DCMP differs by using cross-fitted bidirectional patient-confusion memories and selective correction rather than disease-taxonomy levels; this experiment does not reproduce HP²L's hierarchy.

Sources: [SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf), [BrainPrompt official ABIDE prompts](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt), [PGP-SAM paper](https://arxiv.org/abs/2501.06692), [PGP-SAM code](https://github.com/PRIS-CV/PGP-SAM), [CoW paper](https://ijcai.org/proceedings/2025/139), [CoW code](https://github.com/jcjiang99/CoW), [HP²L DOI](https://doi.org/10.1016/j.media.2026.104063).

## Method and protocol

The four groups share one CAM144-boundary DCMP class and common parameter initialization. G1 uses typical ASD/TD memories; G2 adds two-direction cross-fitted-probe hard memories; G3 adds class-balanced site-offset background candidates; G4 replaces the two frozen BrainPrompt vectors with norm- and cosine-matched anonymous vectors. The risk auxiliary target is the outer-training-only OOF LogisticRegression low-margin top 20% in each class; this is not strict OOF for the shared warmup encoder. Prototype membership is fixed from warmup features and vectors receive deterministic EMA updates (0.9) from detached same-step CAM144 representations. Query routing never consumes query labels, and the current training query is subtracted from its assigned typical/hard and site/global class means. Outer-test IDs and labels/features are masked from the persisted memory; Fold 2 rows 377/407 remain in the historical base training/test indices but are excluded from the added memory/risk labels.

Checkpoint selection retains historical val=test strict-ACC selection, pre-CV RFE/standardization, and the transductive 871-node graph. Metrics are therefore paired mechanism results under a known biased protocol, not unbiased generalization estimates.

## Execution and audit provenance

The preferred OSS path `/data/adodas/SPromptGL-main/experiment/experiment76_dcmp` raised `OSError(5, Input/output error)` while the controller wrote its manifest. The run switched to `/model/spromptgl/experiment/experiment76_dcmp`; copied source inventory hashes matched preflight, and the two already-completed folds' checkpoint hashes matched. Nine latest snapshots were available for resume. A previous 2-worker run had been stopped at saved states (G1 fold 1 next epoch 200, G2 fold 1 next epoch 190) and resumed with 8 workers. No OOM occurred. One earlier worker dispatch failed on a text-bank container type before model restoration or optimizer updates; it was repaired before the successful resumed run. Final task records contain 40/40 successful tasks and no training failures.

Training took 4187.5 seconds on one PPU-ZW810E, with 8 concurrent workers. Independent audit took 117.4 seconds. Disk reconstruction passed 40/40: hard predictions and metrics were exact, probability error was 0, all optimizer states restored, and prototype test-ID isolation passed. Base RNG triplets matched on 14,915 common epochs; group-to-group triplets matched on 22,022 common epochs, with zero mismatches. The root model/data source inventory matched the preflight SHA. The Fold 2 base overlap remains disclosed; the overlapping rows were excluded from the added labeled memories and risk targets.

## Results

All values are mean ± population SD in percent. Hard-AUC is the historical hard-prediction AUC/BA; Prob-AUC is continuous-score ROC-AUC.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 92.1891 ± 2.8121 | 92.0029 ± 2.8083 | 93.7435 ± 2.8209 | 89.5793 ± 3.6663 | 94.4265 ± 3.7676 | 91.4027 ± 3.0323 |
| G2 | 91.5008 ± 3.0122 | 91.3413 ± 2.9476 | 93.7317 ± 2.6303 | 89.3293 ± 2.9654 | 93.3534 ± 4.4944 | 90.7219 ± 3.1095 |
| G3 | 91.7307 ± 2.6674 | 91.5192 ± 2.6532 | 93.5245 ± 3.0665 | 88.8293 ± 3.4093 | 94.2091 ± 3.7745 | 90.8768 ± 2.8450 |
| G4 | 91.9592 ± 2.8684 | 91.7692 ± 2.8773 | 93.9753 ± 3.1117 | 89.3293 ± 4.0366 | 94.2091 ± 3.8772 | 91.1435 ± 3.1141 |
| E74.4-G2-Positive-DCR | 92.1891 ± 2.7648 | 92.0294 ± 2.7255 | 93.4946 ± 2.9195 | 90.0671 ± 3.8862 | 93.9917 ± 4.7320 | 91.4590 ± 2.9013 |

Excluding Fold 2, descriptive nine-fold results are:

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|---:|
| G1 | 92.2153 ± 2.9631 | 92.0186 ± 2.9598 | 93.8260 ± 2.9621 | 89.5054 ± 3.8576 | 94.5318 ± 3.9574 |
| G2 | 91.5781 ± 3.1657 | 91.4191 ± 3.0973 | 93.9071 ± 2.7165 | 89.4986 ± 3.0796 | 93.3395 ± 4.7373 |
| G3 | 91.8336 ± 2.7928 | 91.6167 ± 2.7796 | 93.7005 ± 3.1841 | 88.9431 ± 3.5756 | 94.2903 ± 3.9703 |
| G4 | 92.0875 ± 2.9962 | 91.8797 ± 3.0127 | 94.0365 ± 3.2744 | 89.2276 ± 4.2428 | 94.5318 ± 3.9574 |
| E74.4-G2-Positive-DCR | 92.2153 ± 2.9131 | 92.0481 ± 2.8723 | 93.6142 ± 3.0541 | 90.0474 ± 4.0959 | 94.0487 ± 4.9847 |

Fold 2 complete-test metrics (the historical overlapping rows remain in both train and test for the base protocol) are:

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|---:|
| G1 | 91.9540 | 91.8611 | 93.0011 | 90.2439 | 93.4783 |
| G2 | 90.8046 | 90.6416 | 92.1527 | 87.8049 | 93.4783 |
| G3 | 90.8046 | 90.6416 | 91.9406 | 87.8049 | 93.4783 |
| G4 | 90.8046 | 90.7741 | 93.4252 | 90.2439 | 91.3043 |
| E74.4-G2-Positive-DCR | 91.9540 | 91.8611 | 92.4178 | 90.2439 | 93.4783 |

## Paired comparisons

Comparisons use within-fold differences; confidence intervals are descriptive paired t intervals.

| Comparison | Metric | Δ pp | 95% CI pp | W/T/L |
|---|---|---:|---:|---:|
| G1 vs E74.4-G2 | acc | 0.0000 | [-0.3876, 0.3876] | 1/8/1 |
| G1 vs E74.4-G2 | hard_auc | -0.0265 | [-0.4523, 0.3993] | 1/7/2 |
| G1 vs E74.4-G2 | prob_auc | 0.2489 | [-0.0620, 0.5598] | 5/1/4 |
| G1 vs E74.4-G2 | sen | -0.4878 | [-1.8765, 0.9009] | 1/7/2 |
| G1 vs E74.4-G2 | spe | 0.4348 | [-0.5488, 1.4183] | 1/9/0 |
| G2 vs E74.4-G2 | acc | -0.6883 | [-1.2627, -0.1140] | 0/5/5 |
| G2 vs E74.4-G2 | hard_auc | -0.6881 | [-1.2720, -0.1042] | 1/4/5 |
| G2 vs E74.4-G2 | prob_auc | 0.2371 | [-0.3227, 0.7968] | 5/0/5 |
| G2 vs E74.4-G2 | sen | -0.7378 | [-2.2020, 0.7264] | 1/6/3 |
| G2 vs E74.4-G2 | spe | -0.6383 | [-1.8913, 0.6148] | 1/6/3 |
| G3 vs E74.4-G2 | acc | -0.4585 | [-0.8819, -0.0351] | 0/6/4 |
| G3 vs E74.4-G2 | hard_auc | -0.5102 | [-0.9833, -0.0371] | 1/3/6 |
| G3 vs E74.4-G2 | prob_auc | 0.0299 | [-0.4885, 0.5483] | 5/0/5 |
| G3 vs E74.4-G2 | sen | -1.2378 | [-3.3414, 0.8658] | 1/4/5 |
| G3 vs E74.4-G2 | spe | 0.2174 | [-1.4607, 1.8954] | 3/5/2 |
| G4 vs E74.4-G2 | acc | -0.2299 | [-0.5766, 0.1168] | 0/8/2 |
| G4 vs E74.4-G2 | hard_auc | -0.2602 | [-0.6088, 0.0884] | 0/7/3 |
| G4 vs E74.4-G2 | prob_auc | 0.4807 | [0.0585, 0.9030] | 6/1/3 |
| G4 vs E74.4-G2 | sen | -0.7378 | [-1.9205, 0.4449] | 0/8/2 |
| G4 vs E74.4-G2 | spe | 0.2174 | [-0.9301, 1.3649] | 1/8/1 |
| G2 vs G1 | acc | -0.6883 | [-1.1122, -0.2645] | 0/4/6 |
| G2 vs G1 | hard_auc | -0.6615 | [-1.0826, -0.2405] | 0/4/6 |
| G2 vs G1 | prob_auc | -0.0119 | [-0.4855, 0.4617] | 6/0/4 |
| G2 vs G1 | sen | -0.2500 | [-1.8049, 1.3049] | 2/6/2 |
| G2 vs G1 | spe | -1.0731 | [-2.5637, 0.4175] | 1/5/4 |
| G3 vs G2 | acc | 0.2299 | [-0.2902, 0.7499] | 3/6/1 |
| G3 vs G2 | hard_auc | 0.1778 | [-0.3033, 0.6590] | 4/5/1 |
| G3 vs G2 | prob_auc | -0.2072 | [-0.6878, 0.2735] | 4/0/6 |
| G3 vs G2 | sen | -0.5000 | [-1.9107, 0.9107] | 1/7/2 |
| G3 vs G2 | spe | 0.8557 | [-0.7825, 2.4939] | 3/6/1 |
| G3 vs G4 | acc | -0.2286 | [-0.5733, 0.1161] | 0/8/2 |
| G3 vs G4 | hard_auc | -0.2500 | [-0.6207, 0.1207] | 2/3/5 |
| G3 vs G4 | prob_auc | -0.4508 | [-1.0745, 0.1728] | 3/0/7 |
| G3 vs G4 | sen | -0.5000 | [-2.5219, 1.5219] | 2/4/4 |
| G3 vs G4 | spe | 0.0000 | [-1.7639, 1.7639] | 3/4/3 |

## Mechanism findings

Independent checkpoint audits: 40/40. Maximum test residual fraction: 0.028419; mean test residual fraction: 0.012590. These fractions are direct measured ‖delta‖/‖h‖, not a nominal cap.
Correction-off hard flips across all folds/groups: 0. Fold 2 exposed rows were checked in every intervention; affected counts are reported in `interventions.csv`.
Shared-epoch base RNG comparisons: 14915; exact triplets: 14915; mismatches: 0.

G2−G1 tests the added hard-memory contribution; G3−G2 tests the candidate site-background memory; G3−G4 tests whether the audited natural-language vectors outperform the geometry-matched anonymous control. A positive model-level difference alone does not establish clinical interpretability.

G2−G1 ACC is −0.6883 pp (W/T/L 0/4/6); G3−G2 is +0.2299 pp (3/6/1, descriptive CI crosses zero); G3−G4 is −0.2286 pp (0/8/2). G1 ties the historical base in mean ACC and also has the same nine-fold mean ACC, while G2/G3/G4 remain below it. Across all 40 fixed-checkpoint correction-off interventions there were zero hard-label flips. Mean absolute ASD-probability shifts were small (0.00196–0.00325 by group); the module changed confidence slightly but did not correct hard decisions. Mean test residual fractions were G1 1.502%, G2 1.072%, G3 1.274%, and G4 1.340%, with global max 2.842%, below the 5% cap. G3/G4 retrieval assigned about 71.6% of role attention to candidate site-background evidence; this is not evidence that the retrieved component is pure nuisance.

## Decision and limitations

The experiment conclusion is assigned in `audit_status.json` from the preregistered cross-fold performance and mechanism checks. This single-seed historical protocol does not justify an unbiased generalization claim. The site-offset bank is a candidate background representation, not validated pure noise. OOF difficulty is cross-fitted only at the logistic-probe stage because the shared warmup encoder has seen the outer training cohort. Fold 2 retains known train/test label overlap in the base experiment. Follow-up should first address independent, leakage-controlled validation rather than add more prototype groups.
