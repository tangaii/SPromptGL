# Experiment 81.7.2 TP-PSRR

## Purpose and evidence boundary

This experiment tests fixed disease Prompt guidance of local relation construction and message propagation while preserving the inherited DARG trajectory. CAM144 is reshaped to 16 latent feature slots and is not interpreted as 16 validated anatomical regions.

Source facts come from read-only source review, frozen identities, and replay audits. Fold summaries are observed results under one seed and the inherited test-as-validation protocol. Research conclusions are stated separately from those facts.

## Historical review and frozen changes

The SPromptGL paper and official/local sources were checked. GFR builds and propagates the transductive patient graph, CAM produces CAM144, SPGS uses frozen CLIP RN50 text features, and RES performs the original local/global interaction. CAM144 reaches the original Outputlayer. The experiment uses a local forward pre-hook and invokes the original classifier once.

E74.4-G2 remains the only base. Every fold restores its E61 epoch-10 warmup, three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG, 4096 relation pairs, fold membership, and optimizer order. Pre-CV preprocessing/RFE, the transductive graph, test-as-validation Best selection, and Fold2 overlap remain inherited. Added-branch residual and raw penalty contribution are masked only for zero-based rows 376 and 406 in Fold2.

E81.7.1 used node-local Prompt beta and a same-enhanced-graph direct VJP split. Its beta distributions remained close to uniform and its full model stayed below DARG. TP-PSRR replaces that protection with a same-state original DARG reference gradient and compares node-conditioned G1 with edge-conditioned G2.

## Frozen TP-PSRR design

The original eight Prompt strings, order, CLIP cache, raw normalized Gram, and SVD source are unchanged. E_center is the fixed row-centered normalized semantic basis and is not trainable. The shared local encoder, pair key/value, 64->4->8 semantic map, 15 real neighbors plus Null, Entmax1.5, symmetric ASD/TD competition, pairwise message, zero-initialized 32->9 projection, original RES, and rho=0.1 residual bound are retained. Added parameters remain 7,808.

G1 computes beta per patient and receiver from visual-plus-semantic candidate evidence with a four-Prompt disease-group softmax. G2 computes beta independently for every patient, receiver, real candidate edge, and Prompt from that edge's semantic scores. Both groups use medical Prompt indices TD [0,1,2,3] and ASD [4,5,6,7]. G2 changes only the beta application dimension.

## DARG-reference gradient protection

Each training step first computes the original DARG loss L_B with the residual disabled and records g_B for GFR, CAM, and DARG. All RNG components and model/CAM buffers are restored. The enhanced loss L_F = L_DARG(enhanced) + 0.001 L_raw then uses the same stochastic masks and yields g_F. delta_g = g_F - g_B. Negative group dot products trigger vector orthogonal projection, followed by one group-level scale enforcing ||delta_g_controlled|| <= 0.05||g_B||. The PSRR parameters receive the unmodified enhanced gradient. This protection is a training control and is not counted as a medical contribution.

## Metrics

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.1891 +/- 2.7648|92.0294 +/- 2.7255|93.4946 +/- 2.9195|90.0671 +/- 3.8862|93.9917 +/- 4.7320|91.4590 +/- 2.9013|
|E81.7-G1|91.8443 +/- 2.9346|91.6628 +/- 2.9176|93.7843 +/- 2.8808|89.3293 +/- 4.0366|93.9963 +/- 4.4083|91.0394 +/- 3.1531|
|E81.7.1-G1|91.9592 +/- 2.9591|91.7505 +/- 2.9241|93.3559 +/- 2.9235|89.0793 +/- 3.7445|94.4218 +/- 4.4536|91.1411 +/- 3.1304|
|G1|92.0742 +/- 2.7959|91.8832 +/- 2.7902|93.4151 +/- 3.0773|89.3354 +/- 3.3306|94.4311 +/- 3.3632|91.2647 +/- 3.0317|
|G2|91.7294 +/- 2.9054|91.5222 +/- 2.8872|93.5544 +/- 2.5468|88.8354 +/- 3.7320|94.2091 +/- 4.1186|90.8804 +/- 3.1056|

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
|G1|1|95.4545|95.1220|96.0042|90.2439|100.0000|94.8718|
|G1|2|91.9540|91.8611|92.6829|90.2439|93.4783|91.3580|
|G1|3|87.3563|87.2481|88.0700|85.3659|89.1304|86.4198|
|G1|4|94.2529|93.9362|96.5426|90.0000|97.8723|93.5065|
|G1|5|93.1034|93.2447|95.4787|95.0000|91.4894|92.6829|
|G1|6|96.5517|96.4362|96.8617|95.0000|97.8723|96.2025|
|G1|7|91.9540|91.6223|94.6277|87.5000|95.7447|90.9091|
|G1|8|88.5057|88.2447|88.0319|85.0000|91.4894|87.1795|
|G1|9|89.6552|89.4947|91.9149|87.5000|91.4894|88.6076|
|G1|10|91.9540|91.6223|93.9362|87.5000|95.7447|90.9091|
|G2|1|95.4545|95.1220|95.4852|90.2439|100.0000|94.8718|
|G2|2|89.6552|89.4221|92.2587|85.3659|93.4783|88.6076|
|G2|3|87.3563|87.5133|89.2895|90.2439|84.7826|87.0588|
|G2|4|94.2529|93.9362|95.1064|90.0000|97.8723|93.5065|
|G2|5|93.1034|93.2447|95.7447|95.0000|91.4894|92.6829|
|G2|6|96.5517|96.4362|97.0213|95.0000|97.8723|96.2025|
|G2|7|90.8046|90.3723|95.3191|85.0000|95.7447|89.4737|
|G2|8|88.5057|88.2447|89.3617|85.0000|91.4894|87.1795|
|G2|9|89.6552|89.3085|92.3936|85.0000|93.6170|88.3117|
|G2|10|91.9540|91.6223|93.5638|87.5000|95.7447|90.9091|

## Excluding Fold2

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.2153 +/- 2.9131|92.0481 +/- 2.8723|93.6142 +/- 3.0541|90.0474 +/- 4.0959|94.0487 +/- 4.9847|91.4703 +/- 3.0580|
|E81.7-G1|91.9598 +/- 3.0717|91.7763 +/- 3.0544|93.9007 +/- 3.0142|89.4986 +/- 4.2211|94.0539 +/- 4.6432|91.1549 +/- 3.3036|
|E81.7.1-G1|91.9598 +/- 3.1191|91.7383 +/- 3.0820|93.4424 +/- 3.0695|88.9499 +/- 3.9257|94.5267 +/- 4.6828|91.1170 +/- 3.2989|
|G1|92.0875 +/- 2.9468|91.8857 +/- 2.9411|93.4964 +/- 3.2336|89.2344 +/- 3.4962|94.5370 +/- 3.5293|91.2543 +/- 3.1956|
|G2|91.9598 +/- 2.9746|91.7556 +/- 2.9525|93.6984 +/- 2.6457|89.2209 +/- 3.7403|94.2903 +/- 4.3338|91.1329 +/- 3.1747|

## Paired comparisons

Differences are percentage points. W/T/L is fold-paired. The descriptive interval is not independent-patient significance.

|Comparison|Subset|Metric|Mean delta pp|W/T/L|
|---|---|---|---:|---:|
|G1_vs_DARG|all10|ACC|-0.1149|1/7/2|
|G1_vs_DARG|all10|Hard-AUC/BA|-0.1461|1/6/3|
|G1_vs_DARG|all10|Prob-AUC|-0.0795|5/1/4|
|G1_vs_DARG|all10|SEN-ASD|-0.7317|1/7/2|
|G1_vs_DARG|all10|SPE-TD|+0.4394|2/6/2|
|G1_vs_DARG|all10|ASD-F1|-0.1944|1/6/3|
|G1_vs_DARG|exclude_fold2|ACC|-0.1277|1/6/2|
|G1_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.1624|1/5/3|
|G1_vs_DARG|exclude_fold2|Prob-AUC|-0.1178|4/1/4|
|G1_vs_DARG|exclude_fold2|SEN-ASD|-0.8130|1/6/2|
|G1_vs_DARG|exclude_fold2|SPE-TD|+0.4882|2/5/2|
|G1_vs_DARG|exclude_fold2|ASD-F1|-0.2160|1/5/3|
|G2_vs_DARG|all10|ACC|-0.4598|0/7/3|
|G2_vs_DARG|all10|Hard-AUC/BA|-0.5072|1/5/4|
|G2_vs_DARG|all10|Prob-AUC|+0.0598|4/0/6|
|G2_vs_DARG|all10|SEN-ASD|-1.2317|1/5/4|
|G2_vs_DARG|all10|SPE-TD|+0.2174|2/7/1|
|G2_vs_DARG|all10|ASD-F1|-0.5786|1/5/4|
|G2_vs_DARG|exclude_fold2|ACC|-0.2554|0/7/2|
|G2_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.2925|1/5/3|
|G2_vs_DARG|exclude_fold2|Prob-AUC|+0.0841|4/0/5|
|G2_vs_DARG|exclude_fold2|SEN-ASD|-0.8266|1/5/3|
|G2_vs_DARG|exclude_fold2|SPE-TD|+0.2415|2/6/1|
|G2_vs_DARG|exclude_fold2|ASD-F1|-0.3373|1/5/3|
|G1_vs_G2|all10|ACC|+0.3448|2/8/0|
|G1_vs_G2|all10|Hard-AUC/BA|+0.3610|3/6/1|
|G1_vs_G2|all10|Prob-AUC|-0.1393|4/0/6|
|G1_vs_G2|all10|SEN-ASD|+0.5000|3/6/1|
|G1_vs_G2|all10|SPE-TD|+0.2220|1/8/1|
|G1_vs_G2|all10|ASD-F1|+0.3843|3/6/1|
|G1_vs_G2|exclude_fold2|ACC|+0.1277|1/8/0|
|G1_vs_G2|exclude_fold2|Hard-AUC/BA|+0.1301|2/6/1|
|G1_vs_G2|exclude_fold2|Prob-AUC|-0.2020|3/0/6|
|G1_vs_G2|exclude_fold2|SEN-ASD|+0.0136|2/6/1|
|G1_vs_G2|exclude_fold2|SPE-TD|+0.2467|1/7/1|
|G1_vs_G2|exclude_fold2|ASD-F1|+0.1214|2/6/1|
|G1_vs_E81.7|all10|ACC|+0.2299|3/6/1|
|G1_vs_E81.7|all10|Hard-AUC/BA|+0.2204|4/4/2|
|G1_vs_E81.7|all10|Prob-AUC|-0.3692|4/0/6|
|G1_vs_E81.7|all10|SEN-ASD|+0.0061|3/5/2|
|G1_vs_E81.7|all10|SPE-TD|+0.4348|3/5/2|
|G1_vs_E81.7|all10|ASD-F1|+0.2253|4/4/2|
|G1_vs_E81.7|exclude_fold2|ACC|+0.1277|2/6/1|
|G1_vs_E81.7|exclude_fold2|Hard-AUC/BA|+0.1094|3/4/2|
|G1_vs_E81.7|exclude_fold2|Prob-AUC|-0.4043|4/0/5|
|G1_vs_E81.7|exclude_fold2|SEN-ASD|-0.2642|2/5/2|
|G1_vs_E81.7|exclude_fold2|SPE-TD|+0.4831|3/4/2|
|G1_vs_E81.7|exclude_fold2|ASD-F1|+0.0994|3/4/2|
|G1_vs_E81.7.1|all10|ACC|+0.1149|3/5/2|
|G1_vs_E81.7.1|all10|Hard-AUC/BA|+0.1327|4/4/2|
|G1_vs_E81.7.1|all10|Prob-AUC|+0.0592|5/0/5|
|G1_vs_E81.7.1|all10|SEN-ASD|+0.2561|2/6/2|
|G1_vs_E81.7.1|all10|SPE-TD|+0.0093|2/6/2|
|G1_vs_E81.7.1|all10|ASD-F1|+0.1236|4/4/2|
|G1_vs_E81.7.1|exclude_fold2|ACC|+0.1277|3/4/2|
|G1_vs_E81.7.1|exclude_fold2|Hard-AUC/BA|+0.1474|4/3/2|
|G1_vs_E81.7.1|exclude_fold2|Prob-AUC|+0.0540|4/0/5|
|G1_vs_E81.7.1|exclude_fold2|SEN-ASD|+0.2846|2/5/2|
|G1_vs_E81.7.1|exclude_fold2|SPE-TD|+0.0103|2/5/2|
|G1_vs_E81.7.1|exclude_fold2|ASD-F1|+0.1374|4/3/2|
|G2_vs_E81.7.1|all10|ACC|-0.2299|3/4/3|
|G2_vs_E81.7.1|all10|Hard-AUC/BA|-0.2283|4/3/3|
|G2_vs_E81.7.1|all10|Prob-AUC|+0.1986|5/0/5|
|G2_vs_E81.7.1|all10|SEN-ASD|-0.2439|3/4/3|
|G2_vs_E81.7.1|all10|SPE-TD|-0.2128|1/7/2|
|G2_vs_E81.7.1|all10|ASD-F1|-0.2606|4/3/3|
|G2_vs_E81.7.1|exclude_fold2|ACC|+0.0000|3/4/2|
|G2_vs_E81.7.1|exclude_fold2|Hard-AUC/BA|+0.0173|4/3/2|
|G2_vs_E81.7.1|exclude_fold2|Prob-AUC|+0.2560|5/0/4|
|G2_vs_E81.7.1|exclude_fold2|SEN-ASD|+0.2710|3/4/2|
|G2_vs_E81.7.1|exclude_fold2|SPE-TD|-0.2364|1/6/2|
|G2_vs_E81.7.1|exclude_fold2|ASD-F1|+0.0160|4/3/2|

## Mechanism audit

|Group|Mode|Prompt entropy TD/ASD|Patient beta SD TD/ASD|Within-patient beta SD TD/ASD|ASD/TD support disagreement|Relation difference norm|Contrast message norm|Residual ratio|Prompt-off MAE|Uniform-beta MAE|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|G1|node|1.36602/1.3661|0.0240965/0.0238149|0.0288437/0.0276476|0.223978|7.19282|3.70324|0.0807124|0.00719114|0.00027162|
|G2|edge|1.3453/1.33985|0.0489975/0.0499357|0.0596212/0.0598451|0.228461|6.88255|3.62748|0.0722461|0.00800087|0.000465553|

Across the ten selected checkpoints, intervention effects were as follows. Probability MAE is averaged over folds and hard flips are summed across folds.

|Group|Prompt-off MAE / flips|Relation-off MAE / flips|Uniform-beta MAE / flips|Global-interaction-off MAE / flips|
|---|---:|---:|---:|---:|
|G1|0.007191 / 3|0.007191 / 3|0.000272 / 0|0.000860 / 0|
|G2|0.008001 / 4|0.008001 / 4|0.000466 / 0|0.000669 / 2|

Prompt-off and Relation-off both remove the added residual at the final output in this implementation, so their equality is a structural property and is not evidence of independent Prompt diagnosis.

## Corrections and trajectory

A is historical DARG Best, B is the current checkpoint with the added residual disabled, and C is the complete checkpoint. A-to-B combines changed main-trunk weights and Best epoch. B-to-C isolates the direct residual effect at the selected checkpoint.

|Group|Stage|Corrected|Harmed|Net|
|---|---|---:|---:|---:|
|G1|A_to_B|6|10|-4|
|G1|B_to_C|3|0|3|
|G1|A_to_C|6|7|-1|
|G2|A_to_B|3|11|-8|
|G2|B_to_C|4|0|4|
|G2|A_to_C|3|7|-4|

## Independent audit

G1/G2 Best: `PASS` / `PASS`. G1/G2 Latest replay: `PASS` / `PASS`. Public RNG vs DARG: `PASS` / `PASS`. End-to-end gradient: `PASS` / `PASS`.

The auditor independently reads Best checkpoints, recomputes all six metrics and continuous ASD probabilities, audits interventions, replays Latest, verifies source/checkpoint identities, and compares current and NumPy-2-compatible RNG hashes. Any audit-only repair is disclosed in audit_status.json and does not change training code or checkpoints.

## Research decision

`NO_GO_PERFORMANCE`

The result remains limited by one seed, test-as-validation selection, pre-CV preprocessing/RFE, transductive propagation, and the Fold2 overlap. A mechanism pass does not replace the ACC and Hard-AUC performance gate.

Experiment directory: `/model/spromptgl/experiment/experiment81_7_2_tp_psrr`
Frozen source inventory SHA256: `277ea68833bc4f16686187772c6d1424200e2f01f0c2c0edb3d825893e110f59`
Report and tables: `/model/spromptgl/experiment/experiment81_7_2_tp_psrr/results`
