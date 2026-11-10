# Experiment 77: EC-PSM

**Decision: NO-GO.** All values are percentages, mean +/- population SD across ten folds unless stated.

## Audit and implementation

- Base: E74.4-G2 at `/model/spromptgl/experiment/experiment74_4_soft_positive_dcr`; runner SHA `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`, model SHA `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`, config SHA `a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a`, manifest SHA `4bdfb4fbf191d830c9484c9fdc5d584079895ba9157add248f3f354ea2b48840`.
- Inheritance gate: preflight `PASS`; 10/10 fold identities and E61 epoch-10 RNG snapshots matched; disabled-module one-step parity `True`; G1/G2 first ten epochs matched in 10/10 folds; checkpoint continuation replay `PASS`.
- E77 source SHA: `8c2417fb720a8aab55f1374317dc500ef12f02935cdea8184d78627cfc5fee50`. Protected dependency hashes were unchanged through final audit: `True`.
- Actual data path: DARG ordered `[PHENO, ANAT, FUNC, CORR]`, features `[871,320]` from modality dimensions `[48,6,10,256]`; inherited CAM emits four 36-D blocks at `[871,144]`; the pre-hook replaces the 144-D input to `cam.Outputlayer`, whose output is 36-D before the unchanged two-class classifier.
- E74.4-G2 uses `dugsa.PLNNoPEN`: there is no active PEN or original semantic prompt computation in this audited base. EC-PSM replaces that no-PEN classifier-input boundary; therefore this comparison tests E77 against the audited DARG-CAM G2 base and does not isolate gains over an active E74.4 PEN.
- E74.4 uses Adam (base and CAM: lr 0.0038, weight decay 0.11) and AdamW (added group: lr 0.001, weight decay 0.01), fixed learning rates with no scheduler, 450 maximum epochs, stopping when wait > 200, full-graph one-step-per-epoch training, and strict test-fold ACC maxima for checkpoint selection. E77 appends its parameters to the existing AdamW added group with the inherited lr/weight decay. Loss remains `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve`.
- Fixed E77 settings: four class prototypes per class and modality, at most two FP and two FN prototypes per modality, normalized cosine retrieval with top-2 and temperature 0.2, per-subject EMA 0.9, deterministic farthest-first/Lloyd clustering (25 iterations), stable FP/FN after at least two of the last three eval-mode train predictions, common 10-epoch class-only warmup, bounded residual amplitude 0.05. No additional loss or text encoder run was added.
- Frozen text source: audited E60 CLIP RN50 cache, 8 exact prompts, SHA `c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`; first four prompts define the TD anchor, last four ASD. Four 9-D latent blocks are treated only as learned latent blocks, never anatomical ROIs.
- Class memory updates use detached train-fold CAM features and deterministic epoch-boundary clustering; training queries subtract their own subject feature and count from their assigned cluster before retrieval. Test indices are excluded from stored class and hard memory. DARG is transductive over the full node set, so train representations may still indirectly depend on test-node features.
- The archived base reports Fold-2 train/test overlap at one-based IDs 377 and 407. E77 retained it, excluded those IDs from memory, and did not change the base train split. `val=test`, test-fold ACC selection, pre-CV RFE, and full-graph transductive propagation remain inherited limitations.
- Checkpoint readback: 20/20 independent audits reproduced all metrics, hard predictions, and probabilities. Fold metrics are recomputed from each saved checkpoint.

## Results

| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) | Delta ACC / Hard-AUC / Prob-AUC / SEN / SPE vs base (pp) |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 reference | 92.1891 +/- 2.7648 | 92.0294 +/- 2.7255 | 93.4946 +/- 2.9195 | 90.0671 +/- 3.8862 | 93.9917 +/- 4.7320 | -- |
| E77-G1 | 91.6144 +/- 2.7778 | 91.4579 +/- 2.7242 | 93.3762 +/- 3.0092 | 89.5671 +/- 3.8625 | 93.3488 +/- 4.8886 | -0.5747 / -0.5715 / -0.1184 / -0.5000 / -0.6429 |
| E77-G2 | 91.8443 +/- 2.8431 | 91.6549 +/- 2.8558 | 93.4297 +/- 3.3335 | 89.0915 +/- 3.8646 | 94.2183 +/- 3.4735 | -0.3448 / -0.3745 / -0.0649 / -0.9756 / +0.2266 |

Paired E77-G2 minus E77-G1 differences across the ten folds (mean +/- population SD, percentage points):

| Metric | Mean delta +/- SD (pp) |
|---|---:|
| ACC | +0.2299 +/- 0.6897 |
| HARD_AUC | +0.1970 +/- 0.6964 |
| PROB_AUC | +0.0535 +/- 0.9868 |
| SEN | -0.4756 +/- 2.8675 |
| SPE | +0.8696 +/- 2.5857 |

G1 and G2 have ten paired folds in `fold_metrics.csv`; `paired_comparisons.csv` gives foldwise mean +/- population SD differences versus E74.4-G2 and G2 minus G1. It also reports descriptive nine-fold comparisons excluding Fold 2. These remain under the historical test-selected checkpoint protocol and are not unbiased generalization estimates.

Nine-fold descriptive means +/- population SD after omitting Fold 2 (no retraining):

| Group | ACC | Hard-AUC / BA | Prob-AUC | SEN (ASD) | SPE (TD) |
|---|---:|---:|---:|---:|---:|
| E74.4-G2 reference | 92.2153 +/- 2.9131 | 92.0481 +/- 2.8723 | 93.6142 +/- 3.0541 | 90.0474 +/- 4.0959 | 94.0487 +/- 4.9847 |
| E77-G1 | 91.7044 +/- 2.9142 | 91.5339 +/- 2.8615 | 93.4415 +/- 3.1652 | 89.4919 +/- 4.0645 | 93.5759 +/- 5.1027 |
| E77-G2 | 91.9598 +/- 2.9746 | 91.7675 +/- 2.9892 | 93.4067 +/- 3.5131 | 89.2344 +/- 4.0485 | 94.3005 +/- 3.6522 |

## Mechanism diagnostics

- G1 class banks have 320/320 active prototype slots; 320/320 slots were retrieved, mean bank query coverage is 1.0000, and the maximum within-bank collapse fraction (`cos > 0.98`) is 0.0000. Occupancy and per-modality usage are in `memory_audit.json`.
- G2 selected memory represents 88 fold-summed unique training FP patients (TD predicted ASD) and 179 FN patients (ASD predicted TD); counts are training subjects only, never test labels. FP has 78/80 and FN 80/80 active prototypes, with nonzero retrieval in 40/40 FP and 40/40 FN modality-fold banks. Their cumulative selected-checkpoint usage counters total 51117008 prototype assignment events across training/evaluation forwards.
- Stable-error epoch observations are FP 115510 and FN 99585; separately marked near-boundary observations are near-FP 107811 and near-FN 87974. These are repeated epoch observations, not unique patients; confidence, age, occupancy, and direction-specific counts are in `memory_audit.json`.
- Mean test prediction change rates under fixed-checkpoint interventions: hard-memory off 0.0000, semantic ASD/TD swap 0.0000, and prototype retrieval off 0.0011. These are inference diagnostics only, not retrained causal ablations.
- FP contributes negative ASD-direction evidence and FN contributes positive ASD-direction evidence. If either bank has no selected prototypes or retrievals, the diagnostic is reported as inactive and no gain is attributed to that branch.

## Interpretation and limits

The result is a single-seed comparison under the inherited historical protocol. It does not establish cross-seed stability or independent clinical generalization. Validation equals test for checkpoint selection; RFE preceded outer CV; Fold 2 retains the documented overlap; and transductive full-graph propagation can indirectly expose test-node features to training representations. Excluding Fold 2 is descriptive only and does not redefine the original CV.

## Core computation

For patient i and modality m, the inherited classifier context is h[i,m] in R^36 and its four learned sub-blocks are z[i,m,b] in R^9 (b=1..4). DARG's 320-D output and CAM's 144-D classifier input are untouched outside the hook boundary.

```text
r[i,c,m] = Top2CosineRetrieval(h[i,m], leave-one-subject-out prototypes P[c,m])
a[c] = normalize(W_text * frozen_text_anchor[c]) in R^9
u[i,c,m] = normalize(a[c] + 0.1*tanh(W_prompt * [r[i,c,m], r[i,c,m]-r[i,1-c,m], a[c]])) in R^9
alpha[i,c,m,b] = softmax_b(cos(normalize(z[i,m,b]), u[i,c,m]) / 0.2)
s[i,c,m] = sum_b alpha[i,c,m,b] * z[i,m,b] in R^9
e[i,m] = evidence[i,ASD,m] - evidence[i,TD,m] + 0.1*cos(h[i,m], FN[m]) - 0.1*cos(h[i,m], FP[m])
Delta[i,m] = 0.05*tanh(W_e * [h[i,m], r[i,ASD,m]-r[i,TD,m], s[i,ASD,m], s[i,TD,m], e[i,m]]) in R^36
h_E77[i] = h[i] + concat_m(Delta[i,m]) in R^144
```

Here evidence[i,c,m] is the prompt-conditioned average of four block-to-prompt cosine scores. Empty class or hard banks produce zero retrieval evidence; test IDs cannot become bank members. ASD is positive (label 0), so FN evidence raises and FP evidence lowers ASD-direction evidence. The prompt uses the frozen class text anchor, a retrieved prototype, and an explicit ASD/TD prototype difference; it is a learned feature selector, not generated natural language.

SPromptGL supplies the four learned 9-D blocks per modality; they are not anatomical ROI labels. PGP-SAM uses class/instance prototype guidance for segmentation, Concentrate on Weakness mines weak segmentation regions through self-prediction, and HP²L uses hierarchical disease prompts/prototypes in a large multi-disease setting. E77 borrows only the general prototype-conditioning and error-history ideas; those works do not validate this ASD/TD transfer or E77's clinical interpretation.

No follow-on search or extra training was launched. Final decision: **NO-GO**.
