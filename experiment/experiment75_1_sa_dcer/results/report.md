# Experiment 75.1 — SA-DCER

## Execution and audit
- Status: COMPLETE_AUDITED; tasks 30/30; independent checkpoints 30/30.
- Directory: `/model/spromptgl/experiment/experiment75_1_sa_dcer`. Preferred OSSFS path failed with `Attempted preferred directory creation; OSSFS raised OSError(5, Input/output error). The /model experiment mount is used.`.
- Device: PPU-ZW810E (`cuda:0` runtime); worker cap 10; peak concurrent workers 10.
- Wall time 843.3s; failed tasks 0; retries 0; OOM 0.
- E74.4-G2 source manifest `d3a60ecdfd0158be1eecb066d451e0246a417f69e17c4975fe022075a37ac9c4`; E74.4 source/run/model SHA values are recorded in `manifest.json`.
- SA-DCER training-source SHA `875018eb83510db27d2dfeeb0266734a79244c5eb3d39dd1d1824f96ec3e2872`; current independent-audit SHA `caff898bd4e6e0eadcfd31b586de83bc0314951b80b937687850b5326aea483b`. Audit-only repair: `{'file': 'audit.py', 'preflight_sha256': 'b1c1e54e334c24a310202861d5904bbaf813921c7330150848e2ecc8fa14bd69', 'current_sha256': 'caff898bd4e6e0eadcfd31b586de83bc0314951b80b937687850b5326aea483b', 'reason': 'Corrected fixed-checkpoint CAM feature-change reference from raw H to full H-prime, and converted test indices to a Torch tensor after the first audit attempt; training code, weights and predictions are unchanged.'}`.
- Prior audit attempts: two index-type failures before checkpoint replay, then a result-aggregation `NameError` after checkpoint audits passed 30/30; all three were repaired without changing training code, checkpoints, predictions, or metrics. Full details and the 30/30 pass CSV SHA are in `manifest.json`.
- Fold-2 overlap: 1-based rows 377 (ASD=0) and 407 (TD=1). This is historical label exposure; `val=test`, pre-CV preprocessing and the 871-node transductive graph remain. The nine-fold rows are descriptive only.
- Independent audit: 30/30 pass; exact hard predictions and probability error <=1e-8. Same-fold initialization/warmup 10/10. Common-epoch RNG comparisons: `{'group_vs_group': 31815, 'vs_e74_4_g2': 31815}`.

## Method audit
- The MICCAI SPromptGL paper presents graph-based GFR and a PLN composed of CAM, prompt-guided local evidence and relation-aware refinement; repository code has a latent `view(N,4,4,9)` partition and the original pool on each 36-D position. These are learned feature coordinates, not anatomical ROIs. E74.4-G2 runs PLNNoPEN and E75.1 inserts at its CAM 144-D OutputLayer input; this does not restore the original PEN.
- BrainPrompt combines ROI-, subject- and disease-level prompts; its `compute_label_loss` compares normalized fused graph representation with normalized label embeddings by cross-entropy. Its paper uses Llama/LLM2Vec, unlike this experiment's cached CLIP RN50 prompts. SA-DCER borrows semantic supervision at a local feature boundary, without reproducing BrainPrompt's multi-level network.
- SABER Module III conditions semantic retrieval on graph context and injects patient-specific text at decision level. Its official code was not verified. ERASER and Right for the Right Reasons motivate prediction-sensitivity checks; the token-halving teacher here is model-internal sensitivity, not clinical or causal evidence.
- E75 previously collapsed eight prompts into two, had <1% common-high responses, normalized token teachers to sum to one, reported G3 teacher correlation 0.1241 and residual/CAM ratio 0.1585%, no hard flip under direct-off, and anonymous G4 exceeded real-text G3. This experiment retains all eight embeddings, uses signed non-normalized margin teachers, and confidence factor 0.75+0.5r.

## Classification results
Mean ± population SD (%); Hard-AUC is historical hard prediction/BA, Prob-AUC uses continuous ASD probability.

|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2 historical|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|G1-Anchor|91.8430 ± 3.1094|91.6784 ± 3.1125|93.6047 ± 3.0478|89.5732 ± 4.2901|93.7835 ± 4.1432|91.0485 ± 3.3699|
|G2-Faithful|91.8443 ± 3.0233|91.6551 ± 2.9484|93.6462 ± 2.7374|89.3232 ± 4.0465|93.9870 ± 5.4697|91.0665 ± 3.1069|
|G3-Anonymous|92.0742 ± 2.4963|91.8755 ± 2.4648|93.1287 ± 2.7077|89.3293 ± 3.5417|94.4218 ± 4.1374|91.2736 ± 2.6309|

### Paired ACC differences
- G2−E74.4-G2: all folds -0.3448 pp (1/5/4); excluding Fold 2 -0.2554 pp (1/5/3).
- G2−G1: all folds +0.0013 pp (2/6/2).
- G2−G3 on the descriptive nine folds: -0.2554 pp (2/3/4).
- Complete metric-by-metric and fold-level deltas, including W/T/L and nine-fold comparisons, are in `paired_comparisons.csv`.

### Excluding Fold 2
Descriptive 9-fold mean ± population SD (%), without retraining:

|Group|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2 historical|92.2153 ± 2.9131|92.0481 ± 2.8723|93.6142 ± 3.0541|90.0474 ± 4.0959|94.0487 ± 4.9847|91.4703 ± 3.0580|
|G1-Anchor|91.8307 ± 3.2774|91.6433 ± 3.2790|93.6011 ± 3.2127|89.2276 ± 4.3882|94.0590 ± 4.2796|90.9910 ± 3.5476|
|G2-Faithful|91.9598 ± 3.1658|91.7677 ± 3.0874|93.7120 ± 2.8780|89.4919 ± 4.2319|94.0436 ± 5.7628|91.1850 ± 3.2535|
|G3-Anonymous|92.2153 ± 2.5932|92.0127 ± 2.5616|93.1900 ± 2.8476|89.4986 ± 3.6947|94.5267 ± 4.3486|91.4151 ± 2.7368|

### Paired comparison details
Mean paired change in percentage points, W/T/L; negative values indicate the left-hand group is lower.

|Comparison|Scope|ACC|Hard-AUC|Prob-AUC|SEN|SPE|ASD-F1|
|---|---|---:|---:|---:|---:|---:|---:|
|G1 - base|10 folds|-0.3461 (1/5/4)|-0.3510 (2/4/4)|+0.1101 (7/0/3)|-0.4939 (2/6/2)|-0.2081 (1/6/3)|-0.4105 (2/4/4)|
|G1 - base|9 folds, descriptive|-0.3846 (1/4/4)|-0.4048 (1/4/4)|-0.0132 (6/0/3)|-0.8198 (1/6/2)|+0.0103 (1/6/2)|-0.4793 (1/4/4)|
|G2 - base|10 folds|-0.3448 (1/5/4)|-0.3743 (1/4/5)|+0.1516 (6/0/4)|-0.7439 (1/6/3)|-0.0046 (2/6/2)|-0.3926 (1/4/5)|
|G2 - base|9 folds, descriptive|-0.2554 (1/5/3)|-0.2803 (1/4/4)|+0.0978 (5/0/4)|-0.5556 (1/6/2)|-0.0051 (2/5/2)|-0.2853 (1/4/4)|
|G3 - base|10 folds|-0.1149 (1/7/2)|-0.1538 (1/6/3)|-0.3659 (3/0/7)|-0.7378 (0/7/3)|+0.4302 (2/8/0)|-0.1854 (1/6/3)|
|G3 - base|9 folds, descriptive|-0.0000 (1/7/1)|-0.0354 (1/6/2)|-0.4242 (2/0/7)|-0.5488 (0/7/2)|+0.4780 (2/7/0)|-0.0551 (1/6/2)|
|G2 - G1|10 folds|+0.0013 (2/6/2)|-0.0232 (3/3/4)|+0.0415 (6/0/4)|-0.2500 (3/3/4)|+0.2035 (3/6/1)|+0.0180 (3/3/4)|
|G2 - G1|9 folds, descriptive|+0.1292 (2/6/1)|+0.1244 (3/3/3)|+0.1109 (6/0/3)|+0.2642 (3/3/3)|-0.0154 (2/6/1)|+0.1940 (3/3/3)|
|G2 - G3|10 folds|-0.2299 (2/4/4)|-0.2204 (2/3/5)|+0.5175 (6/0/4)|-0.0061 (3/5/2)|-0.4348 (2/5/3)|-0.2072 (2/3/5)|
|G2 - G3|9 folds, descriptive|-0.2554 (2/3/4)|-0.2449 (2/2/5)|+0.5220 (5/0/4)|-0.0068 (3/4/2)|-0.4831 (2/4/3)|-0.2302 (2/2/5)|

## Mechanism findings
- G2 mean gamma 0.05308; mean residual/CAM norm 0.8789%; mean training signed-direction correlation 0.4424, unlabeled test-node correlation 0.4670; confidence-vs-|teacher| correlation train 0.2644, unlabeled test 0.2669.
- Turning off SA-DCER caused 2 hard flips across 10 G2 checkpoints (full correct→off wrong 2; full wrong→off correct 0). Confidence-neutral caused 0 flips in G2/G3; this measures fixed-checkpoint sensitivity, not a retrained ablation.
- G1/G2/G3 signed evidence, prompt weights, token attention concentration, per-modality residual norms, gamma samples, teacher agreement, and confidence distributions are recorded per fold in `mechanism.csv`.
- Real text vs anonymous text comparison is a single geometry-preserving transform per fold. Similar performance or one transform's advantage cannot establish general medical-language value.
- Latent 9-D token blocks have no anatomical ROI mapping. PHENO includes site/demographic features; ANAT/FUNC quality variables and CORR are not lesion labels.

## Decision
- Scientific decision: **PARTIAL** (predeclared thresholds; no post-hoc tuning). G2 ACC change vs base is -0.3448 pp, G2 wins 1/10, real/anonymous ACC means are 91.8443% / 92.0742%.
- This decision does not support an unbiased independent-generalization claim because of the inherited Fold-2 overlap, `val=test` selection, preprocessing before CV and transductive graph.
- Report: `/model/spromptgl/experiment/experiment75_1_sa_dcer/results/report.md`; manifest: `/model/spromptgl/experiment/experiment75_1_sa_dcer/manifest.json`; audit: `/model/spromptgl/experiment/experiment75_1_sa_dcer/results/audit_status.json`.
