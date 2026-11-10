# Experiment 77.1: SC-PERM

**Decision: NO-GO.** Metrics are percent; means and population standard deviations are across the inherited ten outer folds.

## Audit basis

- Formal base: `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr`; runner SHA `[cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd]` and E74.4 preflight SHA `83f6973ab5e99d2c0863c15982b8595b6d165baeddf210200d10cabeac1770b9`. Each task restores the matching E61 epoch-10 warmup model, optimizer, split and Python/NumPy/Torch CPU/CUDA RNG snapshot.
- E74.4 reference: ACC 92.1891+/-2.7648; Hard-AUC/BA 92.0294+/-2.7255; Prob-AUC 93.4946+/-2.9195; SEN 90.0671+/-3.8862; SPE 93.9917+/-4.7320. The fold checkpoint source is `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr/results/fold_metrics.csv`.
- Local source manifest SHA: `9099d86732c410054234ab15e859d032dd8310af246fc31956b144c9b292a4aa`. All protected source hashes were rechecked after training; see `manifest.json`.
- The true E74.4-G2 model is `dugsa.PLNNoPEN`, so there is no active PEN in this base. SC-PERM intercepts the audited CAM-to-OutputLayer boundary: CAM input `[871,144]`, four modality blocks each `[871,36]`, OutputLayer hidden `[871,36]`, two output log-probabilities. DARG `[871,320]`, CAM internals, attention outputs and the original head are retained.
- The 36-D modality representations are learned latent blocks. They have no verified anatomical ROI identities; this method assigns none.
- Historical training settings are unchanged: Adam on GFR/CAM (`lr=.0038`, `wd=.11`), AdamW on DARG and the appended SC-PERM parameters (`lr=.001`, `wd=.01`), fixed learning rates, full-graph one-step epochs, 450 max epochs, stop when `wait > 200`, and strict test-fold ACC maxima for checkpoint selection. Original loss terms remain `L_cls + L_graph + .05 L_pair + .01 L_adv + .01 L_preserve`; SC-PERM adds only the configured prototype CE and hard-evidence ranking losses.
- The historical Fold-2 train/test overlap is retained at one-based IDs 377 and 407. Both are excluded from prototype, temporal error, and auxiliary-loss indexes. The original base classification loss still follows E74.4's train indices, which include this inherited overlap.
- `val=test` checkpoint selection, pre-outer-CV RFE and transductive full-graph message propagation are preserved. Train memories only store allowed training rows, but their DARG-CAM features can have indirect transductive dependence on test-node features. These results are one seed under a biased historical evaluation protocol, not independent clinical generalization evidence.

## Literature and text audit

SPromptGL's published PEN uses frozen CLIP RN50 disease descriptions to select learned modality features, with modality-specific input projections and transformer/fusion processing in the original network. BrainPrompt uses multilevel brain and disease prompts with representation alignment. SABER proposes semantic-aligned brain-network representations. SC-PERM instead combines patient-to-prototype feature similarity with a shared text-to-prototype semantic relation inside top-2 case retrieval, then uses the resulting class evidence to correct the inherited classifier scores. PGP-SAM, Concentrate on Weakness and HP²L motivate only general prototype or hard-case mechanisms; their segmentation or broader disease settings do not establish transfer validity here.

- SPromptGL: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) and [official code](https://github.com/wanxixi11/SPromptGL).
- BrainPrompt: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf).
- SABER: [arXiv 2607.01901](https://arxiv.org/abs/2607.01901); no claim of a locally audited official implementation.
- PGP-SAM: [arXiv 2501.06692](https://arxiv.org/abs/2501.06692); Concentrate on Weakness: [IJCAI 2025](https://ijcai.org/proceedings/2025/139); HP²L: [Medical Image Analysis DOI](https://doi.org/10.1016/j.media.2026.104063).
- Prompt wording follows the ASD functional-connectivity heterogeneity study [Nature Mental Health DOI](https://doi.org/10.1038/s44220-026-00656-y), which reports individual heterogeneity with network-level convergence, not a universal edge-level ASD signature. Motion-related connectivity confounding is supported by [Power et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254728/). The eight texts are hypotheses for model conditioning, not expert-approved clinical descriptions.

The fixed medical texts are the four TD concepts followed by four ASD concepts in `config.json`. They discuss population-level variability and cautious interpretation; they do not assert that every ASD participant has a specific connection anomaly. Motion, acquisition quality and site are treated as shared measurement context. PHENO, ANAT and FUNC are not assumed to be ASD-specific biomarkers; CORR is the primary functional-connectivity context. No latent CAM block is mapped to an anatomical ROI.

Frozen text cache SHA: `4ee8ad50d69d7d3a40b21933cec69a37be50f2d580cd5a17a4b8d07dd42725d7`. The cache keeps both the exact original eight SPromptGL texts and the fixed medical eight, with RN50 weights SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`. CLIP class-anchor cosine values, token lengths and full pairwise text cosine matrices are recorded in `preflight.json`.

## Method

For CAM input `H in R^(N x 144)`, reshape `h[i,m] in R^36`, with four modalities in `[PHENO, ANAT, FUNC, CORR]` order. Each class `c in {ASD=0, TD=1}` and modality `m` has up to four normalized train-fold centers `p[c,m,k] in R^36`. A training query subtracts its own detached subject contribution from its assigned center before retrieval.

The shared text and prototype projections are `a[c] = normalize(W_t mean_j CLIP(t[c,j])) in R^32` and `v[c,m,k] = normalize(W_p p[c,m,k]) in R^32`. The retrieval score and class evidence are:

```text
q[i,c,m,k] = cos(h[i,m], p[c,m,k]) + beta * eta[m] * cos(v[c,m,k], a[c])
w[i,c,m] = Top2Softmax(q / 0.2)
r[i,c,m] = sum_k w[i,c,m,k] * q[i,c,m,k]
S[i,c] = sum_m eta[m] * r[i,c,m],  eta = [0.15, 0.10, 0.10, 1.0]
FP: H[i,TD] += gated_relu(sim_FP - S[i,ASD]/1.35 - 0.05)
FN: H[i,ASD] += gated_relu(sim_FN - S[i,TD]/1.35 - 0.05)
E[i,c] = S[i,c] + clip(H[i,c], 0, 0.5)
d[i] = lambda * clip(E[i,ASD] - E[i,TD], -2, 2),  0 <= lambda <= 0.75
logp[i] = log_softmax(logp_base[i] + [d[i]/2, -d[i]/2])
L = L_E74.4 + 0.10 * CE(S[allowed_train], y) + 0.05 * hard_ranking_loss(stable_train_FP_FN)
```

Only the log-probabilities returned by the original CAM OutputLayer are modified. FP evidence is TD-directed when FP-pattern similarity exceeds competing ASD typical support by the fixed margin; FN evidence is ASD-directed when FN-pattern similarity exceeds competing TD support. Ambiguous overlaps yield no hard evidence. Error cases are unique training-patient records with an EMA feature, first/last error epoch, capped vote support, frequency and deterministic novelty/recency eviction. The 3-epoch history requires at least two wrong predictions; near-boundary cases remain separate diagnostics. Each memory bank is updated at epoch boundaries and never stores test rows.

## Groups and results

| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) |
|---|---:|---:|---:|---:|---:|
| E74.4-G2 reference | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4946 +/- 2.9195 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 |
| E77.1-G1 | 91.9579 +/- 3.0034 | 91.8431 +/- 2.9327 | 93.7538 +/- 2.8006 | 90.5549 +/- 3.5322 | 93.1314 +/- 5.0212 |
| E77.1-G2 | 91.9592 +/- 2.7267 | 91.7957 +/- 2.6486 | 93.8660 +/- 2.6681 | 89.8171 +/- 3.0738 | 93.7743 +/- 4.7048 |
| E77.1-G3 | 91.6144 +/- 2.8249 | 91.3304 +/- 2.8753 | 93.5651 +/- 2.6252 | 87.5915 +/- 4.3254 | 95.0694 +/- 3.2059 |

G1 changes only the text bank from the audited original SPromptGL prompts; G2 uses the predeclared medical candidate texts; G3 keeps the medical text and full typical-prototype structure while disabling hard-memory reads and training contribution. Each group has ten independently initialized fold runs.

Paired fold deltas (mean +/- population SD, percentage points):

| Comparison | Metric | Mean delta (pp) | SD (pp) |
|---|---|---:|---:|
| G1_vs_E74.4_G2 | acc | -0.2312 | 0.6870 |
| G1_vs_E74.4_G2 | hard_auc | -0.1863 | 0.6621 |
| G1_vs_E74.4_G2 | prob_auc | +0.2591 | 0.5918 |
| G1_vs_E74.4_G2 | sen | +0.4878 | 1.4839 |
| G1_vs_E74.4_G2 | spe | -0.8603 | 1.7208 |
| G2_vs_E74.4_G2 | acc | -0.2299 | 0.6897 |
| G2_vs_E74.4_G2 | hard_auc | -0.2337 | 0.6731 |
| G2_vs_E74.4_G2 | prob_auc | +0.3714 | 0.3699 |
| G2_vs_E74.4_G2 | sen | -0.2500 | 1.3463 |
| G2_vs_E74.4_G2 | spe | -0.2174 | 1.4954 |
| G3_vs_E74.4_G2 | acc | -0.5747 | 1.0597 |
| G3_vs_E74.4_G2 | hard_auc | -0.6990 | 1.0840 |
| G3_vs_E74.4_G2 | prob_auc | +0.0705 | 0.6685 |
| G3_vs_E74.4_G2 | sen | -2.4756 | 2.9275 |
| G3_vs_E74.4_G2 | spe | +1.0777 | 2.7889 |
| G2_minus_G1 | acc | +0.0013 | 0.5111 |
| G2_minus_G1 | hard_auc | -0.0474 | 0.5147 |
| G2_minus_G1 | prob_auc | +0.1122 | 0.6471 |
| G2_minus_G1 | sen | -0.7378 | 1.1271 |
| G2_minus_G1 | spe | +0.6429 | 0.9822 |
| G2_minus_G3 | acc | +0.3448 | 0.7360 |
| G2_minus_G3 | hard_auc | +0.4653 | 0.7182 |
| G2_minus_G3 | prob_auc | +0.3009 | 0.5278 |
| G2_minus_G3 | sen | +2.2256 | 2.8054 |
| G2_minus_G3 | spe | -1.2951 | 2.9326 |

Excluding Fold 2 is a descriptive nine-fold summary only; no split or checkpoint rule is changed. Full per-fold values and all base/E77 paired differences are in `fold_metrics.csv` and `paired_comparisons.csv`.

For reference, the paired mean differences against E74.4-G2 and the read-only E77-G2 result are (ACC / Hard-AUC / Prob-AUC / SEN / SPE, percentage points):

| Group | vs E74.4-G2 | vs E77-G2 |
|---|---|---|
| E77.1-G1 | -0.2312 / -0.1863 / +0.2591 / +0.4878 / -0.8603 | +0.1136 / +0.1882 / +0.3240 / +1.4634 / -1.0870 |
| E77.1-G2 | -0.2299 / -0.2337 / +0.3714 / -0.2500 / -0.2174 | +0.1149 / +0.1408 / +0.4363 / +0.7256 / -0.4440 |
| E77.1-G3 | -0.5747 / -0.6990 / +0.0705 / -2.4756 / +1.0777 | -0.2299 / -0.3245 / +0.1353 / -1.5000 / +0.8511 |

The descriptive Fold-2-excluded means (ACC / Hard-AUC / Prob-AUC / SEN / SPE, %) are: E74.4-G2 `92.2153 / 92.0481 / 93.6142 / 90.0474 / 94.0487`; E77.1-G1 `92.0861 / 91.9472 / 93.8727 / 90.3184 / 93.5759`; E77.1-G2 `92.0875 / 91.9092 / 93.9503 / 89.7696 / 94.0487`; E77.1-G3 `91.8321 / 91.5277 / 93.6454 / 87.5678 / 95.4877`; and historical E77-G2 `91.9598 / 91.7675 / 93.4067 / 89.2344 / 94.3005`. These are descriptive summaries of nine inherited folds, not a new cross-validation estimate.

## Mechanism diagnostics

- Independent checkpoint readback passed 30/30 folds. Each saved model, optimizer, RNG, typical memory, error history, selected epoch and test probability vector was reloaded; predictions and all five primary metrics were recalculated.
- G2 fixed-checkpoint interventions: 40/40 conditions recorded across prototype-off, hard-memory-off, ASD/TD semantic swap and original-text replacement. Counts, probability shifts, true-class log-probability margin changes and affected row indices are in `interventions.csv`.
- On the ten G2 best checkpoints, mean-of-fold test-patient class evidence was ASD `0.8790 +/- 0.1617` and TD `0.9390 +/- 0.1266`; direction-specific hard evidence was ASD `0.0501 +/- 0.0472` and TD `0.0700 +/- 0.0500`. The mean ASD-minus-TD correction margin was `-0.0416`. These are distributions of evidence values, not calibrated probabilities.
- Fixed-weight G2 interventions produced 1 aggregate fold-level hard-prediction flip with hard memory off (mean absolute P(ASD) change `0.00494`, mean absolute true-class log-probability-margin change `0.05395`); 4 flips with typical prototypes off (`0.02850`, `0.33455`); 33 flips after ASD/TD semantic-anchor swap (`0.06869`, `0.67921`); and 7 flips when medical texts were replaced by the original prompts (`0.03108`, `0.34326`). The mean fold ACC after semantic swap was `88.8597%`. Counts are fold-level test-row changes, not unique participants across folds.
- The text intervention confirms that semantic anchors affect retrieval and decision scores, but training G2 versus G1 changed mean ACC by only `+0.0013 pp`, Hard-AUC by `-0.0474 pp`, and Prob-AUC by `+0.1122 pp`; the medical wording therefore has no demonstrated training benefit. G2 versus G3 was `+0.3448 / +0.4653 / +0.3009 pp` in ACC / Hard-AUC / Prob-AUC, with SEN `+2.2256 pp` and SPE `-1.2951 pp`. This is consistent with an incremental hard-memory effect, but G2 still misses the base ACC and Hard-AUC targets, and its hard-off inference effect is small.
- Prototype coverage and collapse were audited per selected checkpoint. All 32/32 typical class-modality slots were active and had nonzero retrievals in every fold; per-slot retrieval-use rates ranged 0.094-0.445 (G1), 0.098-0.409 (G2), and 0.082-0.416 (G3). The fraction of within-class prototype pairs with cosine >0.98 was 0 in G1/G3 and at most 2.08% in G2 (maximum pair cosine 0.9831). G1/G2 hard banks had 32/32 active slots per fold, nonzero use rates of 0.059-0.429 and 0.066-0.419 respectively, and no pair above cosine 0.98; G3's hard bank was 0/32 by design. In G2, selected-epoch fold-specific stable-patient pools had 333-420 FP and 256-363 FN records per fold (mean 362.0/310.3); current-epoch real errors summed across the ten fold contexts were 133 FP and 97 FN, while separately marked near-FP/near-FN counts were 20/64. The selected hard-pool occupancies summed to 567 FP and 620 FN slots across folds. Cross-fold sums are not globally unique participants. Full history, ages, occupancy and exclusions are in `memory_audit.json`; Fold-2 IDs 377 and 407 remain outside new memories and auxiliary loss.
- Preflight on a real training fold measured original-versus-medical class-evidence delta `0.043994881`, semantic-swap retrieval-weight delta `0.00052804692`, and decision-correction delta `0.0009461873`. The fixed-weight evidence audit is in `evidence_diagnostics.json`, generated by `evidence_audit.py` without training or optimizer updates.
- Final audit required two audit-only repairs: account for the selected pre-update forward snapshot when checking hard-memory expiry, and serialize NumPy integer overlap IDs in the CSV audit. The repaired audit then passed all 30 checkpoint reads. No training code, model, configuration, data, optimizer state, or checkpoint changed, and no retraining was performed. The preflight and final audit script SHAs and repair record are in `audit_repair.json` and `manifest.json`.
- The measured semantic and hard-memory effects establish that both branches can affect fixed-weight inference; they do not establish that medical text is superior to original prompts or that hard memory yields a reliable improvement over the base. The primary performance gate fails.

## Decision and limits

Final decision: **NO-GO**. G1/G2/G3 all remain below the fixed base in ACC and Hard-AUC; G2's Prob-AUC gain does not compensate for the lower hard-decision metrics. G2 versus G1 does not show a useful medical-prompt gain. The hard-memory comparison is positive against G3 but small and still below the base. This is a single-seed result under the inherited historical protocol; it does not establish seed robustness or unbiased independent clinical generalization. Do not automatically start E77.2 or a broad hyperparameter search.
