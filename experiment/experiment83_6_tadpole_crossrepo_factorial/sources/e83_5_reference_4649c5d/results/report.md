# Experiment 83.5

## TADPOLE Five-Group Semantic and CAM-Only Fidelity Experiment

**Status**: COMPLETE. Five formal groups, 50/50 folds, with independent replay and audit PASS.

### 1. Scope and locked protocol

The experiment uses the E83.4 TADPOLE release with 598 patients, NC/MCI/AD counts 209/315/74, six modalities with dimensions 145/148/3/36/24/9, 365 input features and CAM dimension 240. Labels are 0/1/2. The data SHA is `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc` and the split SHA is `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`.

The ten-fold protocol is `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)` with training seed 0. Historical full-data RFE, outer-CV StandardScaler, mean imputation, transductive patient graph for G1-G4, `val_idx == test_idx`, strict first test-ACC improvement, 450-epoch limit, patience 200 and no per-epoch seed reset were retained. G5 has no patient graph or graph loss.

Source anchors are the [project repository at E83.4 commit `5ebc4f5`](https://github.com/tangaii/SPromptGL/tree/5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11) and the [author repository](https://github.com/wanxixi11/SPromptGL) at `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The frozen source manifest SHA is `0672b355f6cbf7e00eb34cb805f2bd44a89de7ff99f91313ffbcbef5f3ac6865`, Prompt manifest SHA is `640c0e76c06af284fd14ba4d6ea17d9fa2de9e56710e995f4ad8b34af9e16004`, protocol SHA is `4a12135e8775d275b89bde647ae43589051d8069d415313cfbfde242d1b98001` and training seal SHA is `9c9b2690487977762a6a8d2e997d2c264a5ecef34ad54b7a68570746a95a2ac1`.

The preferred OSSFS directory failed its write probe with EIO/errno 5. The complete self-contained experiment is therefore at `/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity`.

### 2. Formal results

ACC and both AUC values are ten-fold means with population standard deviation. `AUC_original` is the historical hard-prediction one-hot macro OVR calculation. `AUC_probability` is the additional probability macro OVR calculation.

| Group | Method | ACC | AUC_original | AUC_probability | Correct | NC/MCI/AD correct |
|---|---|---:|---:|---:|---:|---:|
| G1 | SPromptGL-MMGPL12 | 87.11 ± 3.93% | 90.65 ± 3.05% | 93.89 ± 2.08% | 521/598 | 184/269/68 |
| G2 | No-PEN | 88.78 ± 3.96% | 90.73 ± 2.88% | 95.40 ± 2.23% | 531/598 | 185/283/63 |
| G3 | DARG | 92.47 ± 3.19% | 93.98 ± 2.74% | 95.55 ± 3.01% | 553/598 | 205/283/65 |
| G4 | DARG+DPSL | **92.97 ± 2.78%** | **94.10 ± 2.61%** | **96.27 ± 2.25%** | **556/598** | 205/287/64 |
| G5 | E43.4-E0 Anchored Physical CAM-only | 85.44 ± 6.45% | 88.31 ± 4.76% | 90.76 ± 4.33% | 511/598 | 188/263/60 |

The machine-readable sources are `results/group_summary.csv`, `results/group_metrics.json` and `results/fold_metrics.csv`.

### 3. G1 semantic repair

The [author implementation](https://github.com/wanxixi11/SPromptGL) and E83.4 legacy branch use eight prompt rows and two-class competition. G1 retains the original PEN MLP and local/global fusion, replaces only the semantic branch with twelve frozen RN50 text rows, four per NC/MCI/AD, concatenates each class group to 4096 dimensions and performs a three-class competition. No new attention layer or CAM/RES/GFR arithmetic was introduced.

The preflight passed the exact eight-query two-class regression, twelve-query three-class forward and gradient checks, participation checks for all three class groups and the held-out-label permutation test. The new G1 result is ACC 87.11%, hard AUC 90.65% and probability AUC 93.89%. Relative to the E83.4 eight-query historical G1 reference, the descriptive differences are ACC -0.01 percentage points, hard AUC +0.49 points and probability AUC -0.65 points. This comparison is not a causal estimate because the runs use different semantic branches and formal release code.

### 4. G2 and G3 inheritance

G2 and G3 were not retrained. Their E83.4 Best and Latest artifacts were copied by hard link, checked against the E83.4 source paths and SHA manifests, independently replayed and evaluated on all ten folds. G2 remains physically No-PEN. G3 retains Positive-DCR, disease-relation learning, original GFR, warmup, loss and optimizer. No GitHub checkpoint was assumed or substituted.

### 5. G4 fixes and mechanism audit

G4 was retrained from the corresponding G3 ten-fold warmup anchors. The final G4 training weights were not used as their own initialization.

The registered E83.4 DPSL parameter set, three-pair weights, training loss, optimizer and update behavior were retained. The fixes affect query normalization and the fixed medical-query adapter only.

The pair context now receives the same L2-normalized query used by the spectral path. On the old E83.4 checkpoint, query norms ranged from 0.8474 to 1.0000. The three normalized context outputs changed by maximum absolute values 0.06122, 0.05409 and 0.09569 for the three class pairs. Other pair arithmetic matched exactly, all three contexts were recomputed, and the differential-query branch was verified inactive. The twelve RN50 queries use a deterministic, label-independent float64 Gram-preserving Cholesky factor, zero padded from 12x12 to 12x32. The Gram reconstruction and float32 cosine checks passed. The condition number was 722.35 and the minimum Gram eigenvalue was 0.01314, so no fallback projection was used. The historical `P_text` remains in the source payload for lineage only. The new adapter preserves the raw RN50 class-centroid cosine values NC-MCI 0.9118, NC-AD 0.9337 and MCI-AD 0.9688, rather than the higher historical transformed values 0.9700, 0.9735 and 0.9983.

All three pairs were active with weights `(p_i+p_j)/2` and no Top-2 exclusion. Across ten folds, the mean test pair weights were 0.4244 for NC-MCI, 0.2635 for NC-AD and 0.3121 for MCI-AD. Mean test residual norms for the weighted NC-MCI, NC-AD and MCI-AD pair corrections were 0.03320, 0.04115 and 0.03903. The summed raw correction mean norm was 0.09458 and the final combined correction mean norm was 0.07171. Mean pair correction cosines were 0.1209 for NC-MCI versus NC-AD, -0.1428 for NC-MCI versus MCI-AD and -0.1192 for NC-AD versus MCI-AD. Mean pairwise cancellation was 19.24%, with a maximum-fold mean of 25.07% and a maximum patient value recorded in the audit of 86.42%.

The configured shared gradient cap was 0.001. The largest observed family fraction was 0.00099989. The residual cap was 0.08. Raw residuals exceeded that limit for 22.47% of all transductive graph nodes aggregated across folds before final cap enforcement, while every final residual satisfied the cap. The ten-fold aggregate DPSL Native/Full counts were identical at 556 correct, with zero direct corrections and zero injuries. Native and Full ACC were both 92.97 ± 2.78%, and hard AUC was 94.10 ± 2.61%. Native probability AUC was 96.30 ± 2.21% and Full was 96.27 ± 2.25%. Relative to G3 full predictions, G4 joint training corrected 26 patients and damaged 23, for a net gain of three patients across the ten folds. The per-fold transition matrix and all pair vectors are in `audits/mechanism_summary.json`.

G4 Full versus Native probability-AUC differences by fold, in percentage points, were `0.0000`, `+0.0779`, `-0.1270`, `+0.0372`, `0.0000`, `0.0000`, `+0.1185`, `0.0000`, `0.0000` and `-0.4049`. The mean paired difference was -0.0298 points. Full and Native hard predictions were identical in every fold, so `AUC_original` was unchanged by inference-time DPSL.

DARG Direct-Off changed zero predicted patients in both G3 and G4. Its maximum log-probability difference was about 1.7e-6 in G3 and 1.9e-6 in G4. Disease-Neutral changed zero predicted patients. Its maximum message-norm intervention was 0.000552 in G3 and 0.001383 in G4. The aggregate G3 confusion matrix, with true NC/MCI/AD rows and predicted NC/MCI/AD columns, was `[[205,4,0],[22,283,10],[0,9,65]]`. The corresponding G4 matrix was `[[205,4,0],[16,287,12],[0,10,64]]`. G4 Native and Full used the same argmax matrix. The G3-to-G2 improvement therefore cannot be attributed to Direct-Off prediction changes, and the complete G3 gain should not be assigned to a single DARG mechanism without a matched intervention.

### 6. G5 physical CAM-only audit

E43.4 E0's verified source flow restored the untrained CAM, CAM Adam and post-construction RNG from E43.2 E01, then trained a physical CAM-only model. E83.5 preserves that initialization procedure while rebuilding each TADPOLE CAM from the E83.4 G2 constructor. For each TADPOLE fold, an isolated reference `FidelityModel` constructor rebuilt the untrained CAM state, initial CAM Adam state and post-construction RNG. No trained G1/G2/G3/G4 weights were used. The formal model contains only `HistoricalCAM`, with `VariLengthInputLayer`, two `EncodeLayer`s, two `FeedForwardLayer`s and `OutputLayer`. It has 75,449 CAM parameters, one CAM-only Adam parameter group with learning rate 0.012 and weight decay 0.0036, and all registered parameter and state names begin with `cam.`. It uses the original weighted three-class NLL. No GFR, GraphLearn, GCN, MessagePassing, PEN, SPGS, CLIP, prompt or graph-loss module is registered. All nine dropout modules have probability 0.

The ten fresh anchors matched their isolated reconstructions. The two-step functional reference path and physical model matched logits, loss, every gradient, Adam state, updated parameters and RNG on all ten folds. G5 was then trained independently for ten folds. Its confusion matrix was `[[188,21,0],[39,263,13],[0,14,60]]`, with true NC/MCI/AD rows and predicted NC/MCI/AD columns. Compared with G1/G2/G3/G4, G5 mean ACC was lower by 1.67/3.34/7.03/7.53 percentage points, hard AUC by 2.34/2.42/5.67/5.79 points, and probability AUC by 3.13/4.64/4.79/5.51 points. Its lower final ACC and AUC are therefore a CAM-only TADPOLE result under this anchored initialization and should not be replaced by the E43.4 ABIDE E0 result of 91.73%.

### 7. Historical and paper comparisons

The corresponding E83.1/E83.3/E83.4 summaries are retained in their original experiment directories. E83.1 and E83.3 G1-G3 are the same inherited results used here: G1 ACC/AUC_original `87.13/90.16%`, G2 `88.78/90.73%` and G3 `92.47/93.98%`; E83.3 and E83.4 probability AUCs for these groups were `94.53/95.40/95.55%`. Historical G4 comparison is:

| Experiment | ACC | AUC_original | AUC_probability |
|---|---:|---:|---:|
| E83.1 | 92.31 ± 3.35% | 93.29 ± 2.91% | not recorded |
| E83.3 | 92.48 ± 3.00% | 93.42 ± 2.64% | 94.56 ± 2.89% |
| E83.4 | 91.64 ± 3.68% | 93.18 ± 3.34% | 96.15 ± 2.03% |
| E83.5 | 92.97 ± 2.78% | 94.10 ± 2.61% | 96.27 ± 2.25% |

E83.1 G4's stored `AUC` uses the historical hard-prediction calculation and has no probability AUC result. E83.5 G4 changes from E83.4 by +1.34 ACC points, +0.92 hard-AUC points and +0.12 probability-AUC points. These changes include the normalized-query and Gram-preserving fixes and a new training run.

The [MICCAI paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf) reports TADPOLE complete SPromptGL at 94.81 ± 2.95% ACC and 95.59 ± 2.57% AUC. G4 is below those paper values by 1.84 ACC points and 1.49 hard-AUC points. G4 probability AUC exceeds 95.59%, but it is a different probability-based metric and is not used to claim the paper target. The paper's CAM-only Fig. 2 reference is 89.12% ACC and 90.84% AUC. G5 is 3.68 ACC points and 2.53 hard-AUC points below those reference values. The paper describes 10-fold evaluation and reports TADPOLE dropout 0, learning rate 0.012, hidden layers 10, encoding layers 2 and four attention heads, while this experiment retains the repository's stated historical data, initialization and checkpoint protocol.

### 8. Audit decisions

| Decision | Result | Evidence |
|---|---|---|
| EXECUTION_AUDIT | PASS | 50/50 formal folds; all five groups have ten predictions covering each patient once |
| METHOD_FIDELITY | PASS | Immutable preflight and source/config/prompt/data hashes |
| CHECKPOINT_REPLAY | PASS | Best 50/50; Latest restore and deterministic next step 50/50 |
| G5_PHYSICAL_REMOVAL | PASS | 10/10 physical structure audits |
| G5_ANCHOR_PARITY | PASS | 10/10 fresh anchor and two-step CAM/Adam/RNG parity |
| G4_BEATS_G3 | PASS on fold means | ACC +0.50 points; hard AUC +0.12 points; probability AUC +0.72 points; positive paired deltas in 3/10, 5/10 and 6/10 folds, respectively |
| G4_PAPER_TARGET_MET | FAIL for ACC and hard AUC | G4 is below 94.81% ACC and 95.59% hard AUC |
| PROMPT_CAUSALITY | DESCRIPTIVE_ONLY | G1 twelve-query versus E83.4 eight-query comparison is not a controlled causal intervention |

The independent replay summary is `audits/independent_replay.json`. The accepted immutable start files are `audits/preflight_start.json` and `audits/training_seal.json`. The first preflight attempts and the corrected audit-entry compatibility run are retained as audit history. No formal training checkpoint was overwritten by the audit-entry correction.

### 9. Reproduction and recovery

The formal gate, source and configuration hashes are bound in `audits/training_seal.json`. G1/G4/G5 were trained with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and eight workers, selected by the resource probe. G5 recovery is:

```bash
cd /model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B common/run_g5.py --workers 8
```

The independent replay is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B audits/run_independent_audit.py
```

The audit-entry shim only repairs the path-key lookup for inherited E83.4 SHA manifests and deep-copies the G5 optimizer state for repeated recovery signatures. It does not change model code, formal checkpoints or the frozen training seal.

