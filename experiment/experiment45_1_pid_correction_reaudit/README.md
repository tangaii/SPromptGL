# Experiment 45.1: PID Correction and Independent Re-audit

## Scope

This directory independently re-estimates Experiment 45's training-fold discrete PID, repairs deterministic sampling and adversarial-loss errors in local 45.1 code, and audits one-step model behavior. The Experiment 45 directory, root model files, ABIDE data, and Experiment 43.2 untrained anchors are treated as read-only. E1-E6 formal training is disabled in this run; even a passed PID gate only permits a later training stage.

Run in order:

```bash
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B run.py preflight --gpu 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B run.py pid
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B run.py power
python -B run.py report
python -B run.py audit
python -B run.py handoff
```

`run.py train` deliberately exits with `FORMAL_TRAINING_DISABLED_BY_DESIGN`. No result in this directory is a new E1-E6 performance result.

## Paper and Estimator Audit

The old name `I_min_two_source_discrete_pid_v1` did not match the implemented optimization. `pid_estimator.py` minimizes `I_Q((X1,X2);Y)` over joint distributions Q that preserve `P(X1,Y)` and `P(X2,Y)`, and derives R, U1, U2, and S from the optimizer. This is the BROJA two-source optimization framework, not the Williams-Beer `I_min` redundancy functional. The correction changes the 45.1 estimator metadata and report terminology. The 45.1 mathematical objective itself is retained.

The BROJA-style discrete optimization is convex for fixed empirical pairwise marginals and target marginal. The implementation checks the analytical gradient, the equality constraints, nonnegative probability support, the target marginal, SLSQP's returned objective against an independent recomputation, both source-target mutual informations, the PID sum identity, and component nonnegativity within numerical tolerance. A separate `trust-constr` implementation with an explicit Hessian validates five small distributions; it does not call the primary SLSQP routine. CVXPY was unavailable, so the SciPy reference solver was used.

PID is computed after KMeans discretization into K=4 states (K=3/5 sensitivity), with a binary outcome and a finite training-fold sample. Its numerical identity is not evidence of reliable population estimation. R-to-lambda is a registered statistical heuristic, not a theorem that an optimal adversarial strength equals `0.03 * redundancy_fraction * q`.

Paper basis:

- SPromptGL, MICCAI 2025: [official MICCAI paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html). It motivates semantic prompt-guided multimodal graph learning; it does not establish a PID-optimal adversarial weight.
- DeepASD, Translational Psychiatry 2024: [publisher paper](https://www.nature.com/articles/s41398-024-02972-2). It studies graph adversarial regularization for ASD classification, without the PID-to-strength mapping used here.
- To Align or Not to Align, AAAI 2026: [official proceedings paper](https://ojs.aaai.org/index.php/AAAI/article/download/39248/43209). It supports the hypothesis that modality alignment should depend on modality interaction, not this particular q rule or its operating threshold.
- Quantifying & Modeling Multimodal Interactions, NeurIPS 2023: [official proceedings paper](https://papers.neurips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html) and [official project code](https://github.com/pliang279/PID). Its estimators and experiments motivate multimodal interaction analysis; clustering each modality to four states is a different, lower-capacity estimation regime.
- Williams and Beer, 2010: [Nonnegative Decomposition of Multivariate Information](https://arxiv.org/abs/1004.2515) defines a redundancy lattice and an `I_min` measure. That name must not be used for the optimizer implemented here.
- Bertschinger et al., 2014: [Quantifying unique information](https://arxiv.org/abs/1311.2852) gives the pairwise-marginal-preserving optimization underpinning BROJA unique information.

Across the sources, disease-related redundancy is a possible reason to align representations, but an observed PID estimate does not show that two imaging modalities share a stable disease signal. PHENO mixes age, site, and sex; ANAT/FUNC are quality and motion summaries; CORR contains selected functional connections. The site-conditional shuffle is therefore a confounding sensitivity analysis, while the frozen q formula continues to use the global label permutation null.

## Deterministic Errors and Their Effects

| Historical location | Deterministic mechanism | 45.1 correction and mathematical meaning | Effect |
|---|---|---|---|
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:121-124, 133-137` | `_fit_labels` returns arrays in train-fold local order. The bootstrap draws local `ix`, but `_pid_on(labels, y, tr[ix], ...)` leaves both modality arrays in their original order while indexing only global Y. This breaks patient pairing. | `run.py:assert_paired` and `pid_phase` use one local `ix` for `labels[i][ix]`, `labels[j][ix]`, and `y_train[ix]`. Global patient indices are never used to index local cluster arrays. | Main point estimates are unaffected; the bootstrap R distribution, q, and lambda are affected. The corrected bootstrap still conditions on frozen fold-fitted scaler/KMeans labels; it does not estimate refitting uncertainty. |
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:134` | `argmax(x[tr, :46])` combines age indicators and site indicators, so its categories do not identify acquisition sites. | `integration.py:site_mapping` resolves all `SITE_ID_*` columns by names in `modal_feat_dict.npy` and exact CSV headers. It validates 20 columns and exactly one active site per patient (871/871). | Historical site-conditioned q95 is not interpretable as site control. New site-conditioned permutation is valid as a sensitivity diagnostic; it does not enter the frozen q formula. |
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:138-142` | It first globally permutes labels, then permutes those already-shuffled labels within guessed groups. This does not preserve each real site's original ASD/TD counts. | `condition_site_permutation_trace` starts from `y_train.copy()` and assigns `y_train[rng.permutation(where)]` only within each true site. It records source positions and tests total and site-level counts. | Corrected site-null values are new diagnostics. They do not replace the global permutation distribution used by q. |
| Historical `config.json:pid.estimator`; `pid_estimator.py:1` | Metadata says `I_min`, while the code minimizes joint-source mutual information subject to both source-target marginals. That is BROJA-style, not Williams-Beer `I_min`. | `config.json` and this report call it `BROJA_two_source_discrete_pid_v1`. The optimizer, pairwise marginal constraints, and decomposition are retained. | Naming/documentation changes. Correct K=4 PID values should reproduce historical values up to recorded numerical/environment differences; q changes due to corrected bootstrap pairing. |
| `experiment45_pid_guided_graph_adversarial_alignment/method.py:51, 132-139` | The code concatenates `z[i]` and `z[j]` along features into `[N,128]`, then labels the first and second half of patients as sources. These labels do not denote the two graph views; the discriminator is declared for 130 inputs. | `method.py:pair_batch` concatenates `[z_i; z_j]` along samples, source 0/1 identifies the first/second graph, duplicates the same training Y for both views, and builds `[2*N_train,66]` from 64 representation plus two disease-condition values. | Corrects adversarial source classification, gradients, and the discriminator input shape. Previous adversarial results cannot be treated as valid view alignment. |
| `experiment45_pid_guided_graph_adversarial_alignment/method.py:127-140` | E6 is not in the historical adversarial mode map, despite the protocol requiring E4's auxiliary and adversarial path. | `method.py:group_lambdas` and `forward_train` give E6 E4's per-pair lambdas, condition labels, GRL, shared projection, auxiliary loss, GFR and graph regularization. Only the CAM input changes to raw X. | Affects future E6 training behavior. No E6 metric is generated here. |
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:147-153` | Historical `lambda_bar` multiplies mean(q) by mean(r), which is not `mean(0.03*q*r)` when q and r covary across AF/AC/FC. | Per fold, 45.1 computes each registered `lambda_mn=0.03*r_mn*q_mn`, then `lambda_bar=mean(lambda_AF,lambda_AC,lambda_FC)`. | Changes fixed-strength E2/E3 when weights are nonzero; E4 already uses per-pair lambda values. |
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:185, 190-203` | The controller writes `TRAINING`, while worker asserts `PID_PASS`; the worker can reject a valid dispatched task. | `dry_run_state_machine` tests a signed PID result and a `TRAINING` status, rejects stale signatures, models no-go reporting as ten reused E0 rows, and records atomic status transitions. The formal worker remains disabled. | Prevents the historical state mismatch from blocking a later approved run. It does not authorize or start this run's training. |
| `experiment45_pid_guided_graph_adversarial_alignment/run.py:225-228` | Report row expectations depend on a mutable completion status; rerunning after a no-go/scientific-complete transition can expect 70 rows even with zero new training folds. | 45.1 reports the explicit `NOT_TRAINED_BY_DESIGN` state and checks the 10 historical E0 rows independently from the gate. | Avoids presenting absent E1-E6 results as zero or missing measurements. |

Historical GFR behavior was reviewed and preserved: four patient relation graphs use the concatenated 320-dimensional multimodal input; ANAT/FUNC/CORR graph propagation contributes to the final fusion; the PHENO propagation output is overwritten by the next branch; GFR computes `ALoss` but the historical simple-2 objective does not optimize it. Experiment 43.2's untrained fold anchor, GFR/CAM structure, Adam states, and RNG are read-only references. The B0-B zero-new-loss path is compared for two updates on both GPUs.

## Statistical Protocol

The fold split remains `StratifiedKFold(10, shuffle=True, random_state=10)`. KMeans and PID use only outer-training rows; the scaler and KMeans fit only on the outer-training fold. Bootstrap uses 12 paired resamples, global label permutation uses 19 independent streams, and true-site conditional permutation uses 19 independent streams when site mapping validates. Each fold/pair/statistic/replicate has its own `SeedSequence` stream so adding a corrected bootstrap does not shift KMeans or permutation streams.

Frozen q and weight formulas are unchanged:

```text
q = clip((Q10(R_boot) - Q95(R_global_perm)) / (median(R_boot) + 1e-12), 0, 1)
r = R / (R + U1 + U2 + S + 1e-12)
lambda_pair = 0.03 * r * q
lambda_bar = mean(lambda_AF, lambda_AC, lambda_FC) within each fold
```

The calibration simulation uses the first training-fold sample size and ASD/TD ratio, K=4, 12 bootstraps and 19 global permutations. It constructs strong/weak redundant, unique-dominant, synergy-dominant, and label-independent pairs and runs the unchanged q/lambda gate. Calibration diagnoses power and false positives; it cannot tune the registered threshold.

## Adversarial and Group Checks

On each GPU, preflight validates a source classifier on deliberately separated view distributions; verifies actual GRL feature gradients equal `-lambda` times ordinary feature gradients; runs one backward/optimizer update for each E1-E6 configuration; checks E2/E3 equal fixed strengths, E4 relation-specific strengths, E5 cyclic weight ownership, and E6's raw CAM input; and verifies shared projection/GFR gradients. It also compares B0-B and the disabled-new-loss path for two gradient/update steps from the same untrained anchor, Adam state, and RNG.

Group map:

| Group | Training-side change |
|---|---|
| E1 | Auxiliary disease classifier only |
| E2 | Unconditional source adversary at the fold lambda_bar |
| E3 | Disease-conditional source adversary at the same lambda_bar as E2 |
| E4 | Disease-conditional per-relation PID lambdas |
| E5 | E4 lambda values cyclically reassigned to AF/AC/FC |
| E6 | E4 losses and graph path retained; main CAM input bypasses GFR to raw X |

Only temporary gradients and parameter updates from one-step checks are made; no model checkpoints are saved or promoted as results.

## Limitations and Interpretation

`q` is based on 12 bootstrap and 19 permutation values per fold/pair. The bootstrap holds the KMeans labels fixed. The 871-row feature matrix has globally preprocessed/RFE-derived inputs, patient graphs are transductive, and the historical validation protocol uses the test fold for selection. These limitations mean a passed gate only supports entering the registered internal model comparison. It cannot support a clinical generalization claim or prove that PID weights are optimal.

The ABIDE modality groups also differ in meaning and dimension. PHENO contains 20 site indicators alongside age/sex columns, ANAT and FUNC are image quality/motion summaries, and CORR contains selected connectivity features. A redundancy estimate after KMeans discretization can be small because of weak shared signal, finite-sample variance, cluster information loss, or q's conservative threshold; zero q is not evidence that the continuous modalities have no shared disease information.
