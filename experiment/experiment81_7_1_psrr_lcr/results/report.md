# Experiment 81.7.1 PSRR-LCR

## Purpose and evidence boundaries

The experiment tests whether frozen disease Prompts select patient-local relations, construct competing ASD/TD sparse graphs, and change pairwise messages that reach the inherited classifier through a bounded CAM residual. The 16 positions are latent CAM feature slots without validated anatomical labels.

Engineering facts below come from source inspection, frozen identities, and replay audits. Fold summaries are observed results. Research judgments are interpretations under one seed and the inherited selection protocol.

## Historical source review

SPromptGL's GFR learns a transductive patient graph and performs graph propagation. CAM is the modality-aware feature encoder that maps representations to CAM144. SPGS forms label-independent class cues from eight frozen CLIP RN50 text descriptions, scores local and pooled modality features against those cues, and weights local slots. RES combines local and global CAM information before classification. In this repository CAM144 reaches the original OutputLayer as a 144-wide vector. PSRR-LCR intercepts that call with a forward pre-hook and invokes the original classifier once.

E74.4-G2 is the sole inherited DARG base. Each fold and group restores its corresponding E61 epoch-10 warmup, all three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG state, fold membership, and original optimizer order. Historical test-equals-validation Best selection, pre-CV feature selection and preprocessing, the transductive patient graph, and the Fold2 train/test overlap remain part of this inherited protocol.

E81.7 produced near-uniform patient-level Prompt weights after averaging evidence across all 16 slots. Its ASD/TD supports differed weakly. Its direct residual contribution added four net corrections at the selected epoch, while the DARG-Best to complete E81.7 comparison lost three net predictions. E81.7 did not run the scrambled semantic control. The historical audit also recorded algebraic cancellation of the visual-baseline message in the final disease-message difference.

## Frozen PSRR-LCR design

The original eight CLIP RN50 Prompt encodings and their order are unchanged, including the original spelling. The frozen 8x8 SVD basis E reproduces the normalized Prompt Gram. E_center is the fixed row-normalized result of subtracting mean(E) from each row. Centering changes the Gram geometry and does not claim to preserve the original Prompt Gram. E and E_center are non-trainable.

CAM144 is reshaped to [N,4,4,9], yielding 16 latent slots of width 9. The original four contiguous 36-value blocks are AvgPool1d(4,4)-pooled to [N,4,9]. The shared local encoder remains Linear(9,32), learned position[16,32], and LayerNorm. Each receiver has 15 non-self candidates and one Null candidate. Shared pair keys are [u_i*u_j, abs(u_j-u_i)] and values are [u_j, u_i*u_j, u_j-u_i]. The shared semantic mapping remains bias-free 64->4->8, and the model retains 7,808 added trainable parameters.

For each patient n, receiver i, and Prompt k, evidence is logsumexp over the 15 visual-plus-semantic candidate scores minus logsumexp over visual candidate scores. There is no reduction over i. TD and ASD each apply a temperature-1.0 softmax over their four fixed Prompts. Their weighted semantic scores form D = S_ASD_sem - S_TD_sem. The two probability graphs are Entmax1.5(S_visual + 0.5D) and Entmax1.5(S_visual - 0.5D), with lambda_rel fixed at 1.0. The contrast message is sum_j((A_ASD-A_TD)*V_ij). A zero-initialized 32->9 projection produces the new residual, which is bounded by rho=0.1 of CAM norm and combined with the original local/global interaction.

G1 uses TD Prompts [0,1,2,3] and ASD Prompts [4,5,6,7]. G2 keeps the same text, cache, architecture, initialization, training schedule, and optimizers, using TD [0,1,4,5] and ASD [2,3,6,7]. It is independently initialized from the same E61 warmup for each fold. No third formal group was trained.

## Training protocol

Both groups use seed 0 over the historical ten folds, begin from E61 epoch 10, train for at most 450 epochs, and stop when the historical wait counter exceeds 200. Best is the strictly improved test-fold ACC, retaining the earlier epoch on ties. DARG optimizers remain Adam at lr=0.0038 and weight decay 0.11 for GFR and CAM, followed by AdamW at lr=0.001 and weight decay 0.01 for DARG parameters. DARG loss and optimizer order remain unchanged. PSRR-LCR adds 0.001 times its raw residual penalty and an independent AdamW optimizer at 0.0005 with weight decay 0.01.

Shared gradients are split using autograd VJPs on the same enhanced forward graph. The branch component includes the historical loss through the new residual and the weighted raw-residual regularizer. Negative branch/direct cosine triggers group-level orthogonal projection, then a single group scale enforces ||g_branch|| <= 0.1||g_direct|| separately for GFR, CAM, and DARG parameters. The epoch-12 probe records classification-to-residual-to-CAM/DARG parameter paths. This is a training stability control, not a medical contribution.

## Six metrics

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.1891 +/- 2.7648|92.0294 +/- 2.7255|93.4946 +/- 2.9195|90.0671 +/- 3.8862|93.9917 +/- 4.7320|91.4590 +/- 2.9013|
|E81.7-G1|91.8443 +/- 2.9346|91.6628 +/- 2.9176|93.7843 +/- 2.8808|89.3293 +/- 4.0366|93.9963 +/- 4.4083|91.0394 +/- 3.1531|
|G1|91.9592 +/- 2.9591|91.7505 +/- 2.9241|93.3559 +/- 2.9235|89.0793 +/- 3.7445|94.4218 +/- 4.4536|91.1411 +/- 3.1304|
|G2|91.8443 +/- 3.0233|91.6074 +/- 3.0756|93.0737 +/- 3.4746|88.3537 +/- 4.8037|94.8612 +/- 3.3483|90.9109 +/- 3.4103|

## Per-fold metrics

|Group|Fold|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|DARG|1|95.4545|95.1220|96.0042|90.2439|100.0000|94.8718|
|DARG|2|91.9540|91.8611|92.4178|90.2439|93.4783|91.3580|
|DARG|3|87.3563|87.6458|89.1304|92.6829|82.6087|87.3563|
|DARG|4|94.2529|93.9362|95.7979|90.0000|97.8723|93.5065|
|DARG|5|93.1034|93.2447|95.3723|95.0000|91.4894|92.6829|
|DARG|6|96.5517|96.4362|97.0745|95.0000|97.8723|96.2025|
|DARG|7|90.8046|90.1862|95.3723|82.5000|97.8723|89.1892|
|DARG|8|88.5057|88.2447|87.8191|85.0000|91.4894|87.1795|
|DARG|9|90.8046|90.5585|92.3404|87.5000|93.6170|89.7436|
|DARG|10|93.1034|93.0585|93.6170|92.5000|93.6170|92.5000|
|E81.7-G1|1|95.4545|95.1220|96.9901|90.2439|100.0000|94.8718|
|E81.7-G1|2|90.8046|90.6416|92.7359|87.8049|93.4783|90.0000|
|E81.7-G1|3|87.3563|87.5133|90.1379|90.2439|84.7826|87.0588|
|E81.7-G1|4|94.2529|93.9362|96.3830|90.0000|97.8723|93.5065|
|E81.7-G1|5|91.9540|92.1809|95.9043|95.0000|89.3617|91.5663|
|E81.7-G1|6|96.5517|96.4362|96.5426|95.0000|97.8723|96.2025|
|E81.7-G1|7|91.9540|91.4362|95.6383|85.0000|97.8723|90.6667|
|E81.7-G1|8|88.5057|88.2447|88.1383|85.0000|91.4894|87.1795|
|E81.7-G1|9|88.5057|88.0585|91.7553|82.5000|93.6170|86.8421|
|E81.7-G1|10|93.1034|93.0585|93.6170|92.5000|93.6170|92.5000|
|G1|1|95.4545|95.1220|96.2117|90.2439|100.0000|94.8718|
|G1|2|91.9540|91.8611|92.5769|90.2439|93.4783|91.3580|
|G1|3|86.2069|86.2937|88.7593|87.8049|84.7826|85.7143|
|G1|4|93.1034|92.6862|94.2553|87.5000|97.8723|92.1053|
|G1|5|91.9540|92.1809|95.9043|95.0000|89.3617|91.5663|
|G1|6|96.5517|96.4362|96.7553|95.0000|97.8723|96.2025|
|G1|7|93.1034|92.6862|95.5319|87.5000|97.8723|92.1053|
|G1|8|88.5057|88.2447|87.7660|85.0000|91.4894|87.1795|
|G1|9|89.6552|89.1223|92.5000|82.5000|95.7447|88.0000|
|G1|10|93.1034|92.8723|93.2979|90.0000|95.7447|92.3077|
|G2|1|95.4545|95.1220|96.4193|90.2439|100.0000|94.8718|
|G2|2|90.8046|90.6416|92.2057|87.8049|93.4783|90.0000|
|G2|3|86.2069|85.8961|88.2821|80.4878|91.3043|84.6154|
|G2|4|94.2529|93.9362|95.8511|90.0000|97.8723|93.5065|
|G2|5|91.9540|92.1809|95.1064|95.0000|89.3617|91.5663|
|G2|6|96.5517|96.4362|96.4362|95.0000|97.8723|96.2025|
|G2|7|91.9540|91.4362|96.0638|85.0000|97.8723|90.6667|
|G2|8|88.5057|88.2447|85.8511|85.0000|91.4894|87.1795|
|G2|9|89.6552|89.1223|91.9149|82.5000|95.7447|88.0000|
|G2|10|93.1034|93.0585|92.6064|92.5000|93.6170|92.5000|

## Excluding Fold 2

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.2153 +/- 2.9131|92.0481 +/- 2.8723|93.6142 +/- 3.0541|90.0474 +/- 4.0959|94.0487 +/- 4.9847|91.4703 +/- 3.0580|
|E81.7-G1|91.9598 +/- 3.0717|91.7763 +/- 3.0544|93.9007 +/- 3.0142|89.4986 +/- 4.2211|94.0539 +/- 4.6432|91.1549 +/- 3.3036|
|G1|91.9598 +/- 3.1191|91.7383 +/- 3.0820|93.4424 +/- 3.0695|88.9499 +/- 3.9257|94.5267 +/- 4.6828|91.1170 +/- 3.2989|
|G2|91.9598 +/- 3.1658|91.7148 +/- 3.2241|93.1701 +/- 3.6498|88.4146 +/- 5.0599|95.0149 +/- 3.4959|91.0121 +/- 3.5805|

## Paired comparisons

Differences are in percentage points. W/T/L counts are fold-paired. The reported population-SD interval is descriptive and does not establish independent-patient significance.

|Comparison|Subset|Metric|Mean delta pp|W/T/L|
|---|---|---|---:|---:|
|G1_vs_DARG|all10|ACC|-0.2299|1/5/4|
|G1_vs_DARG|all10|Hard-AUC/BA|-0.2788|1/4/5|
|G1_vs_DARG|all10|Prob-AUC|-0.1387|5/0/5|
|G1_vs_DARG|all10|SEN-ASD|-0.9878|1/5/4|
|G1_vs_DARG|all10|SPE-TD|+0.4302|3/6/1|
|G1_vs_DARG|all10|ASD-F1|-0.3180|1/4/5|
|G1_vs_DARG|exclude_fold2|ACC|-0.2554|1/4/4|
|G1_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.3098|1/3/5|
|G1_vs_DARG|exclude_fold2|Prob-AUC|-0.1718|4/0/5|
|G1_vs_DARG|exclude_fold2|SEN-ASD|-1.0976|1/4/4|
|G1_vs_DARG|exclude_fold2|SPE-TD|+0.4780|3/5/1|
|G1_vs_DARG|exclude_fold2|ASD-F1|-0.3533|1/3/5|
|G2_vs_DARG|all10|ACC|-0.3448|1/5/4|
|G2_vs_DARG|all10|Hard-AUC/BA|-0.4219|1/5/4|
|G2_vs_DARG|all10|Prob-AUC|-0.4209|3/0/7|
|G2_vs_DARG|all10|SEN-ASD|-1.7134|1/6/3|
|G2_vs_DARG|all10|SPE-TD|+0.8696|2/7/1|
|G2_vs_DARG|all10|ASD-F1|-0.5482|1/5/4|
|G2_vs_DARG|exclude_fold2|ACC|-0.2554|1/5/3|
|G2_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.3333|1/5/3|
|G2_vs_DARG|exclude_fold2|Prob-AUC|-0.4441|3/0/6|
|G2_vs_DARG|exclude_fold2|SEN-ASD|-1.6328|1/6/2|
|G2_vs_DARG|exclude_fold2|SPE-TD|+0.9662|2/6/1|
|G2_vs_DARG|exclude_fold2|ASD-F1|-0.4582|1/5/3|
|G1_vs_G2|all10|ACC|+0.1149|2/7/1|
|G1_vs_G2|all10|Hard-AUC/BA|+0.1431|3/5/2|
|G1_vs_G2|all10|Prob-AUC|+0.2822|7/0/3|
|G1_vs_G2|all10|SEN-ASD|+0.7256|3/5/2|
|G1_vs_G2|all10|SPE-TD|-0.4394|1/8/1|
|G1_vs_G2|all10|ASD-F1|+0.2302|3/5/2|
|G1_vs_G2|exclude_fold2|ACC|+0.0000|1/7/1|
|G1_vs_G2|exclude_fold2|Hard-AUC/BA|+0.0235|2/5/2|
|G1_vs_G2|exclude_fold2|Prob-AUC|+0.2723|6/0/3|
|G1_vs_G2|exclude_fold2|SEN-ASD|+0.5352|2/5/2|
|G1_vs_G2|exclude_fold2|SPE-TD|-0.4882|1/7/1|
|G1_vs_G2|exclude_fold2|ASD-F1|+0.1049|2/5/2|
|G1_vs_E81.7|all10|ACC|+0.1149|3/5/2|
|G1_vs_E81.7|all10|Hard-AUC/BA|+0.0878|3/4/3|
|G1_vs_E81.7|all10|Prob-AUC|-0.4284|2/1/7|
|G1_vs_E81.7|all10|SEN-ASD|-0.2500|2/5/3|
|G1_vs_E81.7|all10|SPE-TD|+0.4255|2/8/0|
|G1_vs_E81.7|all10|ASD-F1|+0.1016|3/4/3|
|G1_vs_E81.7|exclude_fold2|ACC|-0.0000|2/5/2|
|G1_vs_E81.7|exclude_fold2|Hard-AUC/BA|-0.0380|2/4/3|
|G1_vs_E81.7|exclude_fold2|Prob-AUC|-0.4583|2/1/6|
|G1_vs_E81.7|exclude_fold2|SEN-ASD|-0.5488|1/5/3|
|G1_vs_E81.7|exclude_fold2|SPE-TD|+0.4728|2/7/0|
|G1_vs_E81.7|exclude_fold2|ASD-F1|-0.0380|2/4/3|

## Local Prompt and relation mechanisms

|Group|TD/ASD local beta SD|TD/ASD patient beta SD|Raw/centered Prompt score spread|Mean ASD/TD support disagreement|Mean relation-difference norm|Mean contrast-message norm|Mean residual ratio|Prompt-off probability MAE|Uniform-beta probability MAE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|G1|0.0274221/0.0252234|0.0248289/0.0233131|0.186003/0.403838 (2.17345x)|0.247625|8.15666|3.86869|0.0771523|0.00837328|0.000219778|
|G2|0.0277467/0.0277339|0.0224208/0.0230111|0.176344/0.391761 (2.22471x)|0.188614|6.59675|3.62858|0.0826585|0.00952578|0.000424765|

Full per-patient predictions and probabilities for DARG Best, PSRR-off, complete PSRR-LCR, and all four fixed interventions are in `results/interventions.csv`. `results/mechanism.csv` records local beta entropy and variance, sparse edge counts, Null use, support overlap, relation and message norms, residual bounds, and gradient-control traces for both groups.

## Corrections and main-trunk trajectory

A is historical DARG Best, B is the current checkpoint with the added residual disabled, and C is the full model. A-to-B combines changed main-trunk weights and checkpoint selection, so it is not attributed solely to gradient conflict. B-to-C isolates the selected checkpoint's residual effect.

|Group|Stage|Corrected|Harmed|Net|
|---|---|---:|---:|---:|
|G1|A_to_B|4|10|-6|
|G1|B_to_C|4|0|4|
|G1|A_to_C|5|7|-2|
|G2|A_to_B|6|15|-9|
|G2|B_to_C|6|0|6|
|G2|A_to_C|6|9|-3|

|Group|Stage|ASD corrected|ASD harmed|TD corrected|TD harmed|
|---|---|---:|---:|---:|---:|
|G1|A_to_B|2|7|2|3|
|G1|B_to_C|1|0|3|0|
|G1|A_to_C|2|6|3|1|
|G2|A_to_B|1|9|5|6|
|G2|B_to_C|1|0|5|0|
|G2|A_to_C|1|8|5|1|

## Gradient and checkpoint audit

G1/G2 Best readback: `PASS` / `PASS`. G1/G2 Latest next-step replay: `PASS` / `PASS`. Public RNG comparison against DARG: `PASS` / `PASS`. End-to-end gradient audit: `PASS` / `PASS`.

`results/audit_status.json` stores per-fold Latest replay hashes, gradient reconstruction and group-kappa checks, historical identities, and source hashes. The audit reads Best checkpoints independently, recomputes continuous ASD probabilities and all six metrics, and replays one next training step twice from each Latest state.

## Decision and limitations

Research decision: `NO_GO_PERFORMANCE`.

## Post-training audit repair

The frozen auditor SHA256 was `e21e0e49f010efecb8cb3a7373e90237a08840c769e84856959dabc2b9ebc7ca` and the repaired auditor SHA256 is `0284a6d5d77d5082eb3990ba89e99f4adb8924bbfc94d90827bfeecf32c1ba56`. Historical DARG pickle metadata uses the NumPy 2 module path, so a separate NumPy 2 extractor carries only saved predictions and metrics into the primary runtime. RNG hashes also encode the NumPy ndarray pickle module path. The auditor therefore replays every public train/evaluation forward and compares both current-runtime hashes and NumPy-2-compatible hashes with the original DARG trace. It verifies original checkpoint SHA and independently recomputes metrics. The complete source diff is recorded in `results/audit_repair.diff`. Training source and checkpoints were not changed.
G1 over DARG main performance gate: `False`.
G1 over G2 medical-semantic gate: `False`; G1/G2 full predictions changed for 11 test-fold samples.
Mechanism gate by group: `{"G1": {"asd_beta_patient_sd_mean": 0.02331312894821167, "asd_td_support_disagreement_rate": 0.2476254478096962, "folds": 10, "message_contrast_norm_mean": 3.868692100048065, "passed": true, "prompt_off_probability_mae": 0.008373275185447372, "relation_difference_norm_mean": 8.15666241645813, "td_beta_patient_sd_mean": 0.024828879907727242, "uniform_beta_probability_mae": 0.0002197782236748034, "within_patient_local_asd_sd_mean": 0.02522343462333083, "within_patient_local_td_sd_mean": 0.027422088757157327}, "G2": {"asd_beta_patient_sd_mean": 0.02301111342385411, "asd_td_support_disagreement_rate": 0.18861437886953353, "folds": 10, "message_contrast_norm_mean": 3.628580963611603, "passed": true, "prompt_off_probability_mae": 0.00952578021036794, "relation_difference_norm_mean": 6.596747827529907, "td_beta_patient_sd_mean": 0.022420791164040565, "uniform_beta_probability_mae": 0.00042476534183367504, "within_patient_local_asd_sd_mean": 0.027733893319964408, "within_patient_local_td_sd_mean": 0.02774670012295246}}`.

The result inherits test-fold checkpoint selection, pre-CV preprocessing/RFE, transductive patient-graph propagation, and the Fold2 overlap. Fold2 rows 376 and 406 (zero-based) are excluded only from added-branch training contribution. A single seed and these inherited protocol choices limit scientific generalization. A candidate decision means continued research, not final MICCAI 2027 evidence.

Experiment directory: `/model/spromptgl/experiment/experiment81_7_1_psrr_lcr`.
Frozen source inventory SHA256: `5549e0cbde2edd402b8d0742045f5f92a603138fdc3c996826b79ec8830a2ded`.
Audit SHA at freeze/current: `e21e0e49f010efecb8cb3a7373e90237a08840c769e84856959dabc2b9ebc7ca` / `0284a6d5d77d5082eb3990ba89e99f4adb8924bbfc94d90827bfeecf32c1ba56`.
Audit-only code changed after training: `True`.

## Medical grouping and local-choice interpretation

The full G1 and G2 predictions differ for 11 held-out samples. G1 is correct and G2 wrong on 6, while G2 is correct and G1 wrong on 5. Excluding Fold 2, ten samples differ and the correct-only split is 5 versus 5. The fold-paired G1-G2 ACC delta is +0.1149 pp across all folds and 0.0000 pp without Fold 2. This does not establish an independent medical-prompt gain.

Prompt-off and Relation-off each change four G1 and six G2 hard predictions relative to the full checkpoint. Uniform-beta changes no hard predictions in either group, with probability MAE 0.000220 for G1 and 0.000425 for G2. Mean local beta entropies remain near the four-prompt maximum log(4): TD/ASD 1.368/1.374 for G1 and 1.362/1.359 for G2. Thus patient-local beta varies across patients and slots, but its learned distributions remain close to uniform and the uniform-beta intervention has little effect.

The mean ASD/TD support disagreement is 0.248 for G1 and 0.189 for G2, with mean contrast-message norms 3.869 and 3.629. Centered prompt score spread is 2.17x and 2.22x the corresponding raw-basis spread in these trained checkpoints. These are observed mechanism statistics, not a controlled causal comparison proving that centering improves diagnostic discrimination.
