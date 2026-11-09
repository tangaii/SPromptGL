# Experiment 72.5 - Safe Adaptive Dual Prompt and Disease-Difference Conditioning

Status: **COMPLETE_AUDITED**; independent checkpoints **60/60**; decision **PARTIAL_SAFE_RESIDUAL_REPAIR_ONLY**.

## Protocol and source review

The only training base is E69.3-G2 at `41dbcb45a0273673857051d8e821b8c3315eef00`, continuing the E61-E2 end-to-end adaptive DARG forward. No E72.4 checkpoint was used for initialization. E72.4 was read-only and contributed only its previously audited patient text cache.
SPromptGL's MICCAI 2025 paper/code place a 320-dimensional GFR output before CAM and preserve four 36-dimensional positions that flatten to the original 144-dimensional classifier input. BrainPrompt's official ABIDE label text order is TC then ASD; this experiment maps the resulting disease class bank to ASD=0, TD=1. Its patient metadata source includes real participant demographics. SABER Module III's Graph-CLS-conditioned query, patient-text cross-attention and gated residual were verified from the paper equations (8)-(13); no public official SABER implementation was verified. The frozen CLIP RN50 in this experiment is a comparability choice, not a full reproduction of either paper's encoder.
References: [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [SPromptGL official code](https://github.com/wanxixi11/SPromptGL), [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [BrainPrompt official code](https://github.com/AngusMonroe/BrainPrompt), [SABER paper](https://arxiv.org/abs/2607.01901).

Patient text uses only PHENO-recoverable age intervals, recorded sex category codes and site IDs from the existing E72.4 cache (871 rows, three tokens, RN50 1024-d; no truncation). G6 maps the same category identities to fixed, nonlinguistic 1024-d random vectors using a private CPU generator. Neither construction uses diagnosis labels. These demographic/site fields partly duplicate PHENO48; G5 versus G6 tests this specific encoding control and does not establish independent clinical knowledge.

The safety residual is `rho = 0.02 * sigmoid(gate) * (1 - null_attention)` times the DARG feature norm along a normalized learned direction. Its maximum is 2%; the scalar gate starts at 0.1, so the initial preflight maximum is bounded by 0.2% before null attenuation. Disease conditioning uses the signed `tanh(cos(W_condition(F), normalize(T_ASD-T_TD)))` strength, no hard/true label, and a bounded lambda initialized at 0.05. Disease supervision remains CAM-after global plus four-position CE with the E72.1 10-d frozen RN50 text space and fixed fine weights.

Historical limitations remain: fold 2 trains on rows 377 (ASD) and 407 (TD), which remain in its test fold; checkpoint selection uses `val=test`; RFE/standardization precede CV; and the patient graph is transductive. Metrics below are mechanism-paired historical-protocol results, not unbiased generalization estimates.

## Results

Mean +/- population SD across ten folds, in percent. Hard-AUC is computed from hard predictions; Prob-AUC uses continuous TD probability.

|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E69.3-G2|92.0742 +/- 2.7482|91.8995 +/- 2.7186|93.3759 +/- 2.9908|89.5854 +/- 3.4710|94.2137 +/- 3.9815|91.2984 +/- 2.9222|
|G1|88.9786 +/- 3.6767|88.8808 +/- 3.6713|92.0717 +/- 2.9605|87.8171 +/- 4.5658|89.9445 +/- 4.5081|88.0632 +/- 3.9946|
|G2|89.2072 +/- 2.5828|89.1501 +/- 2.5533|92.6149 +/- 2.6792|88.5732 +/- 3.4164|89.7271 +/- 3.8425|88.3768 +/- 2.7635|
|G3|89.4344 +/- 3.5232|89.3309 +/- 3.4131|91.7141 +/- 3.0658|88.0976 +/- 4.0633|90.5643 +/- 5.9817|88.5959 +/- 3.5891|
|G4|88.8610 +/- 2.5292|88.6233 +/- 2.6072|91.4067 +/- 2.7479|85.3780 +/- 5.6226|91.8686 +/- 4.0645|87.5967 +/- 2.9965|
|G5|88.8610 +/- 3.4943|88.6631 +/- 3.5397|91.9961 +/- 3.5017|86.1098 +/- 4.7050|91.2165 +/- 3.3174|87.7242 +/- 3.8503|
|G6|84.1536 +/- 3.0434|83.8176 +/- 3.2315|87.8345 +/- 3.8857|79.1707 +/- 7.4303|88.4644 +/- 4.1657|82.0770 +/- 3.9630|

Paired changes are percentage points. W/T/L counts are paired folds; confidence intervals are descriptive t intervals over correlated folds.

|Comparison|Metric|Folds|Mean delta|95% CI|W/T/L|
|---|---|---:|---:|---:|---:|
|G1_vs_E69.3-G2|acc (ALL)|10|-3.0956|[-5.2823, -0.9089]|1/0/9|
|G1_vs_E69.3-G2|hard_auc (ALL)|10|-3.0187|[-5.1408, -0.8967]|1/0/9|
|G1_vs_E69.3-G2|prob_auc (ALL)|10|-1.3042|[-3.1163, +0.5078]|2/0/8|
|G1_vs_E69.3-G2|sen (ALL)|10|-1.7683|[-4.8980, +1.3614]|1/4/5|
|G1_vs_E69.3-G2|spe (ALL)|10|-4.2692|[-8.2150, -0.3234]|2/2/6|
|G1_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-3.3119|[-5.7409, -0.8828]|1/0/8|
|G1_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-3.2776|[-5.6022, -0.9530]|1/0/8|
|G1_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-1.1134|[-3.1191, +0.8924]|2/0/7|
|G1_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-2.7778|[-5.2169, -0.3386]|0/4/5|
|G1_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-3.7774|[-8.0919, +0.5372]|2/2/5|
|G1_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-3.5563|[-6.0203, -1.0924]|1/0/8|
|G2_vs_E69.3-G2|acc (ALL)|10|-2.8670|[-4.0369, -1.6971]|0/1/9|
|G2_vs_E69.3-G2|hard_auc (ALL)|10|-2.7494|[-3.8118, -1.6870]|0/1/9|
|G2_vs_E69.3-G2|prob_auc (ALL)|10|-0.7610|[-2.1229, +0.6009]|4/0/6|
|G2_vs_E69.3-G2|sen (ALL)|10|-1.0122|[-3.8190, +1.7946]|3/3/4|
|G2_vs_E69.3-G2|spe (ALL)|10|-4.4866|[-8.0466, -0.9266]|1/1/8|
|G2_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-2.9302|[-4.2535, -1.6068]|0/1/8|
|G2_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-2.8428|[-4.0294, -1.6562]|0/1/8|
|G2_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-0.8338|[-2.3745, +0.7070]|4/0/5|
|G2_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-1.6667|[-4.3843, +1.0510]|2/3/4|
|G2_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-4.0189|[-7.8930, -0.1449]|1/1/7|
|G2_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-3.0653|[-4.3211, -1.8095]|0/1/8|
|G3_vs_E69.3-G2|acc (ALL)|10|-2.6398|[-3.7379, -1.5416]|0/2/8|
|G3_vs_E69.3-G2|hard_auc (ALL)|10|-2.5686|[-3.5846, -1.5526]|0/1/9|
|G3_vs_E69.3-G2|prob_auc (ALL)|10|-1.6618|[-3.2410, -0.0826]|2/0/8|
|G3_vs_E69.3-G2|sen (ALL)|10|-1.4878|[-4.3714, +1.3958]|1/4/5|
|G3_vs_E69.3-G2|spe (ALL)|10|-3.6494|[-7.0606, -0.2382]|1/3/6|
|G3_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-2.5499|[-3.7799, -1.3200]|0/2/7|
|G3_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-2.5359|[-3.6908, -1.3810]|0/1/8|
|G3_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-1.6226|[-3.4196, +0.1745]|2/0/7|
|G3_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-2.4661|[-4.5729, -0.3593]|0/4/5|
|G3_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-2.6056|[-5.4114, +0.2002]|1/3/5|
|G3_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-2.7406|[-3.9942, -1.4869]|0/1/8|
|G4_vs_E69.3-G2|acc (ALL)|10|-3.2132|[-3.9634, -2.4629]|0/0/10|
|G4_vs_E69.3-G2|hard_auc (ALL)|10|-3.2762|[-4.1034, -2.4490]|0/0/10|
|G4_vs_E69.3-G2|prob_auc (ALL)|10|-1.9692|[-3.1050, -0.8335]|1/0/9|
|G4_vs_E69.3-G2|sen (ALL)|10|-4.2073|[-8.1844, -0.2302]|2/1/7|
|G4_vs_E69.3-G2|spe (ALL)|10|-2.3451|[-5.6311, +0.9410]|1/4/5|
|G4_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-3.3148|[-4.1287, -2.5008]|0/0/9|
|G4_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-3.4281|[-4.2857, -2.5706]|0/0/9|
|G4_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-2.0702|[-3.3382, -0.8022]|1/0/8|
|G4_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-5.2168|[-8.9278, -1.5058]|1/1/7|
|G4_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-1.6394|[-4.9130, +1.6342]|1/4/4|
|G4_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-3.9321|[-4.9690, -2.8952]|0/0/9|
|G5_vs_E69.3-G2|acc (ALL)|10|-3.2132|[-4.3595, -2.0668]|0/0/10|
|G5_vs_E69.3-G2|hard_auc (ALL)|10|-3.2364|[-4.3627, -2.1102]|0/0/10|
|G5_vs_E69.3-G2|prob_auc (ALL)|10|-1.3798|[-2.7247, -0.0350]|3/0/7|
|G5_vs_E69.3-G2|sen (ALL)|10|-3.4756|[-6.1350, -0.8162]|0/4/6|
|G5_vs_E69.3-G2|spe (ALL)|10|-2.9972|[-5.7227, -0.2717]|1/3/6|
|G5_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-3.0593|[-4.3042, -1.8145]|0/0/9|
|G5_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-3.1129|[-4.3564, -1.8694]|0/0/9|
|G5_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-1.0501|[-2.3253, +0.2252]|3/0/6|
|G5_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-3.8618|[-6.7245, -0.9991]|0/3/6|
|G5_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-2.3641|[-5.0068, +0.2787]|1/3/5|
|G5_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-3.4951|[-4.8943, -2.0960]|0/0/9|
|G6_vs_E69.3-G2|acc (ALL)|10|-7.9206|[-9.5812, -6.2600]|0/0/10|
|G6_vs_E69.3-G2|hard_auc (ALL)|10|-8.0820|[-9.8688, -6.2952]|0/0/10|
|G6_vs_E69.3-G2|prob_auc (ALL)|10|-5.5414|[-8.1811, -2.9018]|1/0/9|
|G6_vs_E69.3-G2|sen (ALL)|10|-10.4146|[-15.6354, -5.1939]|0/0/10|
|G6_vs_E69.3-G2|spe (ALL)|10|-5.7493|[-9.6315, -1.8671]|1/1/8|
|G6_vs_E69.3-G2|acc (FOLD2_EXCLUDED)|9|-7.7789|[-9.6359, -5.9219]|0/0/9|
|G6_vs_E69.3-G2|hard_auc (FOLD2_EXCLUDED)|9|-7.9401|[-9.9435, -5.9368]|0/0/9|
|G6_vs_E69.3-G2|prob_auc (FOLD2_EXCLUDED)|9|-5.3383|[-8.3007, -2.3758]|1/0/8|
|G6_vs_E69.3-G2|sen (FOLD2_EXCLUDED)|9|-10.2168|[-16.1450, -4.2886]|0/0/9|
|G6_vs_E69.3-G2|spe (FOLD2_EXCLUDED)|9|-5.6635|[-10.0825, -1.2444]|1/1/7|
|G6_vs_E69.3-G2|asd_f1 (FOLD2_EXCLUDED)|9|-9.0779|[-11.7082, -6.4475]|0/0/9|
|G1_vs_E72.4-G1|acc (ALL)|10|+4.6029|[+0.4710, +8.7349]|7/1/2|
|G1_vs_E72.4-G1|hard_auc (ALL)|10|+4.6727|[+0.4394, +8.9060]|7/1/2|
|G1_vs_E72.4-G1|prob_auc (ALL)|10|+4.7353|[+0.8837, +8.5868]|9/0/1|
|G1_vs_E72.4-G1|sen (ALL)|10|+6.1585|[-1.6592, +13.9763]|7/2/1|
|G1_vs_E72.4-G1|spe (ALL)|10|+3.1869|[-2.1047, +8.4785]|6/1/3|
|G3_vs_G2|acc (ALL)|10|+0.2273|[-1.0441, +1.4987]|5/2/3|
|G3_vs_G2|hard_auc (ALL)|10|+0.1808|[-1.0343, +1.3959]|6/1/3|
|G3_vs_G2|prob_auc (ALL)|10|-0.9008|[-1.9902, +0.1886]|3/0/7|
|G3_vs_G2|sen (ALL)|10|-0.4756|[-3.3323, +2.3811]|3/3/4|
|G3_vs_G2|spe (ALL)|10|+0.8372|[-2.4770, +4.1513]|4/2/4|
|G3_vs_G4|acc (ALL)|10|+0.5734|[-0.4699, +1.6167]|5/2/3|
|G3_vs_G4|hard_auc (ALL)|10|+0.7076|[-0.3032, +1.7184]|6/1/3|
|G3_vs_G4|prob_auc (ALL)|10|+0.3074|[-0.9411, +1.5560]|6/0/4|
|G3_vs_G4|sen (ALL)|10|+2.7195|[+0.1460, +5.2930]|7/2/1|
|G3_vs_G4|spe (ALL)|10|-1.3043|[-4.2242, +1.6155]|3/3/4|
|G5_vs_G3|acc (ALL)|10|-0.5734|[-1.5420, +0.3952]|2/1/7|
|G5_vs_G3|hard_auc (ALL)|10|-0.6678|[-1.6306, +0.2950]|2/0/8|
|G5_vs_G3|prob_auc (ALL)|10|+0.2819|[-0.7421, +1.3060]|6/0/4|
|G5_vs_G3|sen (ALL)|10|-1.9878|[-5.0634, +1.0878]|2/1/7|
|G5_vs_G3|spe (ALL)|10|+0.6522|[-2.2303, +3.5347]|5/3/2|
|G5_vs_G6|acc (ALL)|10|+4.7074|[+2.6037, +6.8111]|10/0/0|
|G5_vs_G6|hard_auc (ALL)|10|+4.8456|[+2.5671, +7.1240]|10/0/0|
|G5_vs_G6|prob_auc (ALL)|10|+4.1616|[+1.0121, +7.3110]|9/0/1|
|G5_vs_G6|sen (ALL)|10|+6.9390|[+1.3111, +12.5670]|8/0/2|
|G5_vs_G6|spe (ALL)|10|+2.7521|[-0.6319, +6.1361]|7/1/2|

## Mechanism

The safe-residual unit test verified the initial 0.2% upper bound, explicit Null-token attenuation, lower-gate/lower-residual behavior, finite nonzero gate and direction gradients, and unchanged main RNG. Per-checkpoint diagnostics are in `mechanism.csv`; attention is reported as model behavior, not clinical explanation.

- G1: residual mean/p95/max 0.01432/0.01952/0.01988; fraction at >=1.9% cap 0.393; null attention 0.277; age/sex/site attention 0.131/0.264/0.329; residual-off hard flips 11.40 per fold.
- G2: residual mean/p95/max 0.01429/0.01948/0.01986; fraction at >=1.9% cap 0.389; null attention 0.275; age/sex/site attention 0.126/0.258/0.341; residual-off hard flips 12.10 per fold.
- G3: residual mean/p95/max 0.01449/0.01968/0.01992; fraction at >=1.9% cap 0.442; null attention 0.274; age/sex/site attention 0.130/0.263/0.333; residual-off hard flips 11.80 per fold.
- G4: residual mean/p95/max 0.01522/0.01960/0.01989; fraction at >=1.9% cap 0.494; null attention 0.230; age/sex/site attention 0.131/0.293/0.346; residual-off hard flips 13.50 per fold.
- G5: residual mean/p95/max 0.01437/0.01963/0.01990; fraction at >=1.9% cap 0.444; null attention 0.278; age/sex/site attention 0.137/0.262/0.323; residual-off hard flips 13.80 per fold.
- G6: residual mean/p95/max 0.01803/0.01978/0.01993; fraction at >=1.9% cap 0.613; null attention 0.069; age/sex/site attention 0.308/0.293/0.330; residual-off hard flips 14.30 per fold.
- G5 disease-difference query: condition lambda mean 0.0514; signed strength SD 0.2684; lambda-off flips 0.30 per fold; reverse flips 0.80; strength-permutation flips 0.00; site-token mask ACC delta -7.226 pp.

Fold-2 full and row-removal sensitivities are in `fold2_sensitivity.csv`. Removing rows 377/407 only removes this known overlap; all other historical evaluation limitations remain.

Subgroup summaries are in `subgroup_diagnostics.csv`. Site-token reliance or site-specific performance is a dataset-bias risk, not an ASD mechanism. DARG gate changes and relation-edge diagnostics are recorded per checkpoint; they do not alone establish useful relation learning.

## Decision

Pre-registered decision: **PARTIAL_SAFE_RESIDUAL_REPAIR_ONLY**. The first-stage performance repair compares G1 against E69.3-G2 and the historical E72.4-G1 loss of -7.6985 pp ACC. The innovation stage requires G5 to match/exceed E69.3-G2, consistently beat G3 and G6, and show a net positive fixed-checkpoint condition intervention. No conclusion is upgraded from a single fold or hard-flip count.

Even a positive result would require a clean follow-up with no train/test overlap, fold-local preprocessing, validation-only checkpoint selection, and external or nested validation before a generalization claim.

## Audit

- Independent best-checkpoint reads: 60/60; failures: 0.
- Exact common epoch RNG triplets: 19069/19069.
- Protected sources/data/history: PASS (1904 files).
- Frozen training source/config SHA is in `manifest.json`; report SHA is recorded after audit. Experiment root: `/model/spromptgl/experiment/experiment72_5_safe_dual_prompt`.
- Three audit-only corrections were recorded: E72.4 percentage-column mapping and passing the frozen prototype/checkpoint into mechanism diagnostics. The final `audit.py` SHA is `a4cb2af83b42e2e70987f499f68543263084e611b21af77b24cd1ee285fcf8ab`; training code, predictions, and checkpoints did not change.

## Fold 2 and Mechanism Detail

The original 10-fold comparison remains primary. Excluding fold 2 is descriptive only: G1 is `-3.3119 pp` ACC versus E69.3-G2 over the other nine folds (W/T/L `1/0/8`), and G5 is `-3.0593 pp` (W/T/L `0/0/9`). The two-row removal does not remove their effect on training; rows 377 and 407 were still used for fold-2 classification and disease supervision under the inherited protocol.

|Fold 2 model|ACC / Prob-AUC, all 87 test rows (%)|ACC / Prob-AUC after removing rows 377 and 407 (%)|
|---|---:|---:|
|E69.3-G2|90.8046 / 92.3648|90.5882 / 92.3333|
|G1|89.6552 / 89.3425|89.4118 / 89.2222|
|G2|88.5057 / 92.2587|88.2353 / 92.1667|
|G3|87.3563 / 90.3499|87.0588 / 90.3333|
|G4|88.5057 / 91.3043|88.2353 / 91.2778|
|G5|86.2069 / 88.0170|85.8824 / 87.9444|
|G6|81.6092 / 84.9947|81.1765 / 84.6111|

The new residual removes E72.4's approximately `9.97%` mean residual and `10%` forced cap. G1's trained mean is `1.432%`, p95 `1.952%`, max `1.988%`; G5's mean is `1.437%`, p95 `1.963%`, max `1.990%`. The explicit gate and Null route work numerically, but the learned G1/G5 scalar gate means are `0.987/0.994`, and `39.3%/44.4%` of test rows remain at or above `1.9%`. G6 reaches `1.803%` mean with `61.3%` at or above `1.9%`, while its Null attention is only `0.069` versus about `0.278` for G5. The cap is no longer mechanically forcing every sample to the boundary, yet the trained gate often learns a near-ceiling injection; this is a partial repair, not evidence of robust suppression when text is unhelpful. In fixed G1 checkpoint interventions, disabling residual injection changes `11.4` hard predictions per fold on average, with `8.4` corrected and `3.0` made incorrect by the active residual.

At tau `1.0`, SPromptGL-8 G3 exceeds BrainPrompt-2 G2 by `0.2273 pp` ACC (W/T/L `5/2/3`), while Prob-AUC is `0.9008 pp` lower. G3 at tau `1.0` exceeds G4 at tau `0.1` by `0.5734 pp` ACC (W/T/L `5/2/3`); SEN is `2.7195 pp` higher (W/T/L `7/2/1`), with no overall stable temperature winner. Every disease-supervised group uses the fixed Fine+Global objective, so this six-group design cannot isolate Fine+Global against Global-only supervision.

G5's signed condition strength has mean fold-level SD `0.2684`; learned lambda averages `0.0514` from a `0.05` initialization. On fixed checkpoints, turning lambda off flips 3 test predictions total, all corrected by the active condition and none newly wrong. Reversing the signed strength flips 8 total (6 corrected, 2 newly wrong); permuting patient strengths changes logits/attention but causes 0 hard flips. Despite this small positive lambda-off intervention, G5 is `0.5734 pp` below G3 in ACC, loses on 7 folds, and does not meet the registered cross-layer criterion. G5 exceeds the specific nonlinguistic G6 control by `4.7074 pp` ACC (W/T/L `10/0/0`), but this only supports the natural-language encoding over that fixed random category-vector control; it does not establish independent medical knowledge.

G5 attention averages age `0.137`, sex `0.262`, site `0.323`, and Null `0.278`. Masking the site token lowers G5 ACC by `7.226 pp` on average; the site subgroups with at least 10 held-out observations range from `71.4%` to `96.4%` ACC. This is a material site-dependence risk. Adaptive DARG gate distance from the E69.3-G2 reference is only `0.00017` to `0.00028` on average across groups, so these runs do not show a meaningful change in DARG relation-gate learning.

For G5, held-out age-bin ACC ranges from `79.2%` to `100%` (bin sizes 12-160); sex-category ACC is `89.27%` for SEX_1 (`n=727`) and `86.81%` for SEX_2 (`n=144`). These subgroup results are descriptive under the inherited test-based checkpoint-selection and Fold-2 exposure protocol. At the best checkpoints, disease-head global semantic accuracy averages `85.3%` for G5 and `88.1%` for G3; the CORR-position semantic accuracy averages `85.1%` and `87.3%`, respectively. The four CAM positions have already interacted, so they are not independent raw modality or ROI measurements. The weighted auxiliary-to-classification CAM-gradient ratio averages `0.075` to `0.219` across supervised groups, rather than remaining at the initial 0.1 target throughout training. Semantic supervision is learned, but its metrics do not establish clinical interpretability or classification benefit.
