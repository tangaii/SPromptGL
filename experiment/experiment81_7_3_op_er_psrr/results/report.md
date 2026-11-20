# Experiment 81.7.3 OP-PSRR / ER-PSRR

## Purpose and evidence boundary

This experiment tests optimizer-aware DARG trajectory protection and two end-to-end Prompt-guided local relation candidates. G1 retains node-conditioned PSRR. G2 adds a reliability-gated relation path before a local SPGS/RES compatibility branch and uses a training-only evidence margin. The 16 CAM slots are latent feature positions, not anatomically validated regions.

Source facts come from read-only source review, frozen identities, and replay audits. Fold summaries are observed results under one seed and the inherited test-as-validation protocol. Research conclusions are stated separately from those facts.

## Historical review and frozen changes

The original SPromptGL PLN executes transformer encoding, frozen-CLIP SPGS local weighting, contiguous global pooling, local/global RES, then CAM144 into Outputlayer. GFR builds and propagates the transductive patient graph. The actual E74.4 CAM is vendored PLNNoPEN and contains no SPGS or RES. E81.7.2's audited node-level PSRR implementation is reused, with no initialization from its trained checkpoints.

Each fold restores the matching E61 epoch-10 E74.4-G2 state, three optimizer states, RNG streams, historical split, 4096 training relation pairs, and optimizer step order. The val=test Best rule, pre-CV preprocessing/RFE, transductive graph, and Fold2 overlap remain historical. Rows 376/406 are excluded only from added residual and evidence supervision in Fold2.

G2 compatibility correction: because PLNNoPEN lacks the requested internal SPGS-to-RES insertion point, ER-PSRR computes a local SPGS-to-PSRR-to-RES difference path and adds RES(P+deltaP)-RES(P) to the inherited CAM144. Zero residual preserves the DARG output, loss, gradient, and RNG trajectory. The local SPGS MLP adds 525,577 parameters and the reliability gate adds 49, for 533,434 trainable parameters total. E74.4 has no matching SPGS or RES weights. This is a combined G2 candidate, not an isolated test of its gate, evidence loss, or insertion point.

## Frozen OP-PSRR and ER-PSRR design

Both groups use the fixed eight Prompt cache and centered frozen semantic directions, node-level Prompt selection, shared pair key/value and low-rank semantic map, 15 real neighbors plus Null, Entmax1.5 ASD/TD competition, pairwise contrast messages, a zero-initialized output projection, and rho=0.1 residual limit.

Both groups compute beta for each patient and receiver over the four Prompt vectors in each disease group. OP-PSRR uses shared node-level relations. ER-PSRR multiplies relation contrast by a [0,1] sigmoid gate derived from Prompt evidence, ASD/TD evidence difference, visual relation score, and Prompt uncertainty. Its auxiliary evidence score comes from Prompt-weighted sparse relation support and is supervised only on legal training rows with ASD target +1, TD target -1, margin 0.1, and coefficient 0.005. Legal-training evidence alignment and signed margins are reported as observations, not additional performance endpoints.

## DARG-reference gradient protection

Each step computes DARG reference gradient g_B, then restores RNG and model buffers before the enhanced g_F pass. A conflicting group delta is orthogonally projected. The runner simulates the current native Adam/AdamW parameter update on isolated moment copies and selects the largest alpha satisfying an actual update-distance ratio of 0.02. A cloned native optimizer check includes grad=None semantics. Each real shared optimizer steps once. PSRR parameters receive the full task gradient. This is a training stability method, not a medical contribution.

## Metrics

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.1891 +/- 2.7648|92.0294 +/- 2.7255|93.4946 +/- 2.9195|90.0671 +/- 3.8862|93.9917 +/- 4.7320|91.4590 +/- 2.9013|
|E81.7-G1|91.8443 +/- 2.9346|91.6628 +/- 2.9176|93.7843 +/- 2.8808|89.3293 +/- 4.0366|93.9963 +/- 4.4083|91.0394 +/- 3.1531|
|E81.7.1-G1|91.9592 +/- 2.9591|91.7505 +/- 2.9241|93.3559 +/- 2.9235|89.0793 +/- 3.7445|94.4218 +/- 4.4536|91.1411 +/- 3.1304|
|E81.7.2-G1|92.0742 +/- 2.7959|91.8832 +/- 2.7902|93.4151 +/- 3.0773|89.3354 +/- 3.3306|94.4311 +/- 3.3632|91.2647 +/- 3.0317|
|G1|91.9592 +/- 2.6778|91.7077 +/- 2.7189|93.7070 +/- 2.7261|88.3415 +/- 4.0168|95.0740 +/- 3.0461|91.0353 +/- 2.9920|
|G2|91.8443 +/- 2.7486|91.6921 +/- 2.7739|93.7691 +/- 2.9946|89.5915 +/- 4.3981|93.7928 +/- 3.7692|91.0411 +/- 3.0287|

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
|E81.7.1-G1|1|95.4545|95.1220|96.2117|90.2439|100.0000|94.8718|
|E81.7.1-G1|2|91.9540|91.8611|92.5769|90.2439|93.4783|91.3580|
|E81.7.1-G1|3|86.2069|86.2937|88.7593|87.8049|84.7826|85.7143|
|E81.7.1-G1|4|93.1034|92.6862|94.2553|87.5000|97.8723|92.1053|
|E81.7.1-G1|5|91.9540|92.1809|95.9043|95.0000|89.3617|91.5663|
|E81.7.1-G1|6|96.5517|96.4362|96.7553|95.0000|97.8723|96.2025|
|E81.7.1-G1|7|93.1034|92.6862|95.5319|87.5000|97.8723|92.1053|
|E81.7.1-G1|8|88.5057|88.2447|87.7660|85.0000|91.4894|87.1795|
|E81.7.1-G1|9|89.6552|89.1223|92.5000|82.5000|95.7447|88.0000|
|E81.7.1-G1|10|93.1034|92.8723|93.2979|90.0000|95.7447|92.3077|
|E81.7.2-G1|1|95.4545|95.1220|96.0042|90.2439|100.0000|94.8718|
|E81.7.2-G1|2|91.9540|91.8611|92.6829|90.2439|93.4783|91.3580|
|E81.7.2-G1|3|87.3563|87.2481|88.0700|85.3659|89.1304|86.4198|
|E81.7.2-G1|4|94.2529|93.9362|96.5426|90.0000|97.8723|93.5065|
|E81.7.2-G1|5|93.1034|93.2447|95.4787|95.0000|91.4894|92.6829|
|E81.7.2-G1|6|96.5517|96.4362|96.8617|95.0000|97.8723|96.2025|
|E81.7.2-G1|7|91.9540|91.6223|94.6277|87.5000|95.7447|90.9091|
|E81.7.2-G1|8|88.5057|88.2447|88.0319|85.0000|91.4894|87.1795|
|E81.7.2-G1|9|89.6552|89.4947|91.9149|87.5000|91.4894|88.6076|
|E81.7.2-G1|10|91.9540|91.6223|93.9362|87.5000|95.7447|90.9091|
|G1|1|95.4545|95.1220|96.0042|90.2439|100.0000|94.8718|
|G1|2|90.8046|90.6416|92.7359|87.8049|93.4783|90.0000|
|G1|3|88.5057|88.3351|89.8727|85.3659|91.3043|87.5000|
|G1|4|94.2529|93.9362|95.8511|90.0000|97.8723|93.5065|
|G1|5|93.1034|93.2447|95.9043|95.0000|91.4894|92.6829|
|G1|6|96.5517|96.4362|97.1277|95.0000|97.8723|96.2025|
|G1|7|90.8046|90.1862|95.8511|82.5000|97.8723|89.1892|
|G1|8|88.5057|88.2447|88.7766|85.0000|91.4894|87.1795|
|G1|9|89.6552|89.3085|92.0745|85.0000|93.6170|88.3117|
|G1|10|91.9540|91.6223|92.8723|87.5000|95.7447|90.9091|
|G2|1|95.4545|95.1220|96.8345|90.2439|100.0000|94.8718|
|G2|2|90.8046|90.6416|92.6829|87.8049|93.4783|90.0000|
|G2|3|87.3563|87.2481|88.8123|85.3659|89.1304|86.4198|
|G2|4|93.1034|92.6862|95.0532|87.5000|97.8723|92.1053|
|G2|5|91.9540|92.1809|96.1702|95.0000|89.3617|91.5663|
|G2|6|96.5517|96.4362|97.7660|95.0000|97.8723|96.2025|
|G2|7|91.9540|92.1809|95.5851|95.0000|89.3617|91.5663|
|G2|8|88.5057|88.2447|88.6170|85.0000|91.4894|87.1795|
|G2|9|89.6552|89.1223|92.6596|82.5000|95.7447|88.0000|
|G2|10|93.1034|93.0585|93.5106|92.5000|93.6170|92.5000|

## Excluding Fold2

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.2153 +/- 2.9131|92.0481 +/- 2.8723|93.6142 +/- 3.0541|90.0474 +/- 4.0959|94.0487 +/- 4.9847|91.4703 +/- 3.0580|
|E81.7-G1|91.9598 +/- 3.0717|91.7763 +/- 3.0544|93.9007 +/- 3.0142|89.4986 +/- 4.2211|94.0539 +/- 4.6432|91.1549 +/- 3.3036|
|E81.7.1-G1|91.9598 +/- 3.1191|91.7383 +/- 3.0820|93.4424 +/- 3.0695|88.9499 +/- 3.9257|94.5267 +/- 4.6828|91.1170 +/- 3.2989|
|E81.7.2-G1|92.0875 +/- 2.9468|91.8857 +/- 2.9411|93.4964 +/- 3.2336|89.2344 +/- 3.4962|94.5370 +/- 3.5293|91.2543 +/- 3.1956|
|G1|92.0875 +/- 2.7933|91.8262 +/- 2.8414|93.8149 +/- 2.8532|88.4011 +/- 4.2299|95.2513 +/- 3.1615|91.1504 +/- 3.1328|
|G2|91.9598 +/- 2.8742|91.8089 +/- 2.9005|93.8898 +/- 3.1334|89.7900 +/- 4.5933|93.8277 +/- 3.9715|91.1568 +/- 3.1715|

## Paired comparisons

Differences are percentage points. W/T/L is fold-paired. The descriptive interval is not independent-patient significance.

|Comparison|Subset|Metric|Mean delta pp|W/T/L|
|---|---|---|---:|---:|
|G1_vs_DARG|all10|ACC|-0.2299|1/6/3|
|G1_vs_DARG|all10|Hard-AUC/BA|-0.3216|1/6/3|
|G1_vs_DARG|all10|Prob-AUC|+0.2124|7/1/2|
|G1_vs_DARG|all10|SEN-ASD|-1.7256|0/6/4|
|G1_vs_DARG|all10|SPE-TD|+1.0823|2/8/0|
|G1_vs_DARG|all10|ASD-F1|-0.4237|1/6/3|
|G1_vs_DARG|exclude_fold2|ACC|-0.1277|1/6/2|
|G1_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.2219|1/6/2|
|G1_vs_DARG|exclude_fold2|Prob-AUC|+0.2007|6/1/2|
|G1_vs_DARG|exclude_fold2|SEN-ASD|-1.6463|0/6/3|
|G1_vs_DARG|exclude_fold2|SPE-TD|+1.2026|2/7/0|
|G1_vs_DARG|exclude_fold2|ASD-F1|-0.3199|1/6/2|
|G2_vs_DARG|all10|ACC|-0.3448|1/5/4|
|G2_vs_DARG|all10|Hard-AUC/BA|-0.3372|1/4/5|
|G2_vs_DARG|all10|Prob-AUC|+0.2745|7/0/3|
|G2_vs_DARG|all10|SEN-ASD|-0.4756|1/5/4|
|G2_vs_DARG|all10|SPE-TD|-0.1989|2/6/2|
|G2_vs_DARG|all10|ASD-F1|-0.4179|1/4/5|
|G2_vs_DARG|exclude_fold2|ACC|-0.2554|1/5/3|
|G2_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.2392|1/4/4|
|G2_vs_DARG|exclude_fold2|Prob-AUC|+0.2756|6/0/3|
|G2_vs_DARG|exclude_fold2|SEN-ASD|-0.2575|1/5/3|
|G2_vs_DARG|exclude_fold2|SPE-TD|-0.2210|2/5/2|
|G2_vs_DARG|exclude_fold2|ASD-F1|-0.3134|1/4/4|
|G1_vs_G2|all10|ACC|+0.1149|3/5/2|
|G1_vs_G2|all10|Hard-AUC/BA|+0.0156|4/4/2|
|G1_vs_G2|all10|Prob-AUC|-0.0621|5/0/5|
|G1_vs_G2|all10|SEN-ASD|-1.2500|2/6/2|
|G1_vs_G2|all10|SPE-TD|+1.2812|4/5/1|
|G1_vs_G2|all10|ASD-F1|-0.0058|4/4/2|
|G1_vs_G2|exclude_fold2|ACC|+0.1277|3/4/2|
|G1_vs_G2|exclude_fold2|Hard-AUC/BA|+0.0173|4/3/2|
|G1_vs_G2|exclude_fold2|Prob-AUC|-0.0749|4/0/5|
|G1_vs_G2|exclude_fold2|SEN-ASD|-1.3889|2/5/2|
|G1_vs_G2|exclude_fold2|SPE-TD|+1.4236|4/4/1|
|G1_vs_G2|exclude_fold2|ASD-F1|-0.0065|4/3/2|
|G1_vs_E81.7|all10|ACC|+0.1149|3/5/2|
|G1_vs_E81.7|all10|Hard-AUC/BA|+0.0450|3/5/2|
|G1_vs_E81.7|all10|Prob-AUC|-0.0772|4/2/4|
|G1_vs_E81.7|all10|SEN-ASD|-0.9878|1/6/3|
|G1_vs_E81.7|all10|SPE-TD|+1.0777|3/7/0|
|G1_vs_E81.7|all10|ASD-F1|-0.0041|3/5/2|
|G1_vs_E81.7|exclude_fold2|ACC|+0.1277|3/4/2|
|G1_vs_E81.7|exclude_fold2|Hard-AUC/BA|+0.0499|3/4/2|
|G1_vs_E81.7|exclude_fold2|Prob-AUC|-0.0858|4/1/4|
|G1_vs_E81.7|exclude_fold2|SEN-ASD|-1.0976|1/5/3|
|G1_vs_E81.7|exclude_fold2|SPE-TD|+1.1975|3/6/0|
|G1_vs_E81.7|exclude_fold2|ASD-F1|-0.0046|3/4/2|
|G1_vs_E81.7.1|all10|ACC|-0.0000|3/4/3|
|G1_vs_E81.7.1|all10|Hard-AUC/BA|-0.0428|4/3/3|
|G1_vs_E81.7.1|all10|Prob-AUC|+0.3512|6/1/3|
|G1_vs_E81.7.1|all10|SEN-ASD|-0.7378|2/4/4|
|G1_vs_E81.7.1|all10|SPE-TD|+0.6522|2/7/1|
|G1_vs_E81.7.1|all10|ASD-F1|-0.1057|4/3/3|
|G1_vs_E81.7.1|exclude_fold2|ACC|+0.1277|3/4/2|
|G1_vs_E81.7.1|exclude_fold2|Hard-AUC/BA|+0.0879|4/3/2|
|G1_vs_E81.7.1|exclude_fold2|Prob-AUC|+0.3725|5/1/3|
|G1_vs_E81.7.1|exclude_fold2|SEN-ASD|-0.5488|2/4/3|
|G1_vs_E81.7.1|exclude_fold2|SPE-TD|+0.7246|2/6/1|
|G1_vs_E81.7.1|exclude_fold2|ASD-F1|+0.0334|4/3/2|
|G1_vs_E81.7.2|all10|ACC|-0.1149|1/7/2|
|G1_vs_E81.7.2|all10|Hard-AUC/BA|-0.1755|1/6/3|
|G1_vs_E81.7.2|all10|Prob-AUC|+0.2920|7/1/2|
|G1_vs_E81.7.2|all10|SEN-ASD|-0.9939|0/7/3|
|G1_vs_E81.7.2|all10|SPE-TD|+0.6429|3/7/0|
|G1_vs_E81.7.2|all10|ASD-F1|-0.2294|1/6/3|
|G1_vs_E81.7.2|exclude_fold2|ACC|-0.0000|1/7/1|
|G1_vs_E81.7.2|exclude_fold2|Hard-AUC/BA|-0.0595|1/6/2|
|G1_vs_E81.7.2|exclude_fold2|Prob-AUC|+0.3185|6/1/2|
|G1_vs_E81.7.2|exclude_fold2|SEN-ASD|-0.8333|0/7/2|
|G1_vs_E81.7.2|exclude_fold2|SPE-TD|+0.7144|3/6/0|
|G1_vs_E81.7.2|exclude_fold2|ASD-F1|-0.1040|1/6/2|
|G2_vs_E81.7|all10|ACC|+0.0000|1/8/1|
|G2_vs_E81.7|all10|Hard-AUC/BA|+0.0293|2/6/2|
|G2_vs_E81.7|all10|Prob-AUC|-0.0151|4/0/6|
|G2_vs_E81.7|all10|SEN-ASD|+0.2622|1/7/2|
|G2_vs_E81.7|all10|SPE-TD|-0.2035|2/7/1|
|G2_vs_E81.7|all10|ASD-F1|+0.0017|2/6/2|
|G2_vs_E81.7|exclude_fold2|ACC|+0.0000|1/7/1|
|G2_vs_E81.7|exclude_fold2|Hard-AUC/BA|+0.0326|2/5/2|
|G2_vs_E81.7|exclude_fold2|Prob-AUC|-0.0109|4/0/5|
|G2_vs_E81.7|exclude_fold2|SEN-ASD|+0.2913|1/6/2|
|G2_vs_E81.7|exclude_fold2|SPE-TD|-0.2261|2/6/1|
|G2_vs_E81.7|exclude_fold2|ASD-F1|+0.0019|2/5/2|
|G2_vs_E81.7.1|all10|ACC|-0.1149|1/7/2|
|G2_vs_E81.7.1|all10|Hard-AUC/BA|-0.0584|2/6/2|
|G2_vs_E81.7.1|all10|Prob-AUC|+0.4133|10/0/0|
|G2_vs_E81.7.1|all10|SEN-ASD|+0.5122|2/6/2|
|G2_vs_E81.7.1|all10|SPE-TD|-0.6290|1/7/2|
|G2_vs_E81.7.1|all10|ASD-F1|-0.0999|2/6/2|
|G2_vs_E81.7.1|exclude_fold2|ACC|+0.0000|1/7/1|
|G2_vs_E81.7.1|exclude_fold2|Hard-AUC/BA|+0.0706|2/6/1|
|G2_vs_E81.7.1|exclude_fold2|Prob-AUC|+0.4474|9/0/0|
|G2_vs_E81.7.1|exclude_fold2|SEN-ASD|+0.8401|2/6/1|
|G2_vs_E81.7.1|exclude_fold2|SPE-TD|-0.6989|1/6/2|
|G2_vs_E81.7.1|exclude_fold2|ASD-F1|+0.0399|2/6/1|
|G2_vs_E81.7.2|all10|ACC|-0.2299|1/6/3|
|G2_vs_E81.7.2|all10|Hard-AUC/BA|-0.1911|2/4/4|
|G2_vs_E81.7.2|all10|Prob-AUC|+0.3541|7/1/2|
|G2_vs_E81.7.2|all10|SEN-ASD|+0.2561|2/5/3|
|G2_vs_E81.7.2|all10|SPE-TD|-0.6383|1/6/3|
|G2_vs_E81.7.2|all10|ASD-F1|-0.2235|2/4/4|
|G2_vs_E81.7.2|exclude_fold2|ACC|-0.1277|1/6/2|
|G2_vs_E81.7.2|exclude_fold2|Hard-AUC/BA|-0.0768|2/4/3|
|G2_vs_E81.7.2|exclude_fold2|Prob-AUC|+0.3934|7/0/2|
|G2_vs_E81.7.2|exclude_fold2|SEN-ASD|+0.5556|2/5/2|
|G2_vs_E81.7.2|exclude_fold2|SPE-TD|-0.7092|1/5/3|
|G2_vs_E81.7.2|exclude_fold2|ASD-F1|-0.0975|2/4/3|

## Mechanism audit

|Group|Mode|Prompt entropy TD/ASD|Patient beta SD TD/ASD|Within-patient beta SD TD/ASD|ASD/TD support disagreement|Relation difference norm|Contrast message norm|Residual ratio|Prompt-off MAE|Uniform-beta MAE|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|G1|node|1.36825/1.36721|0.0213102/0.018711|0.0241945/0.0219983|0.231171|8.15972|3.86743|0.0827105|0.00767629|0.000276148|
|G2|node|1.35223/1.36907|0.027943/0.0253578|0.0200888/0.0188603|0.0920862|5.12928|0.549762|0.00524381|0.000496061|2.44442e-05|

## Corrections and trajectory

A is historical DARG Best, B is the current checkpoint with the added residual disabled, and C is the complete checkpoint. A-to-B combines changed main-trunk weights and Best epoch. B-to-C isolates the direct residual effect at the selected checkpoint.

|Group|Stage|Corrected|Harmed|Net|
|---|---|---:|---:|---:|
|G1|A_to_B|6|13|-7|
|G1|B_to_C|6|1|5|
|G1|A_to_C|5|7|-2|
|G2|A_to_B|9|12|-3|
|G2|B_to_C|0|0|0|
|G2|A_to_C|9|12|-3|

## Independent audit

G1/G2 Best: `PASS` / `PASS`. G1/G2 Latest replay: `PASS` / `PASS`. Public RNG vs DARG: `PASS` / `PASS`. End-to-end gradient: `PASS` / `PASS`.

The auditor independently reads Best checkpoints, recomputes all six metrics and continuous ASD probabilities, audits interventions, replays Latest, verifies source/checkpoint identities, and compares current and NumPy-2-compatible RNG hashes. Any audit-only repair is disclosed in audit_status.json and does not change training code or checkpoints.

## Research decision

`NO_GO_PERFORMANCE`

The result remains limited by one seed, test-as-validation selection, pre-CV preprocessing/RFE, transductive propagation, and the Fold2 overlap. A mechanism pass does not replace the ACC and Hard-AUC performance gate.

Experiment directory: `/model/spromptgl/experiment/experiment81_7_3_op_er_psrr`
Frozen source inventory SHA256: `c1e69afe0e6eff8982c57254f60275fc3745e9625d678c5c07bc4eaea076e179`
Report and tables: `/model/spromptgl/experiment/experiment81_7_3_op_er_psrr/results`
