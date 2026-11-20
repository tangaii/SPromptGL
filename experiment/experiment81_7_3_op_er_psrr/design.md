# Experiment 81.7.3 frozen design

## Research question

OP-PSRR tests whether preserving the actual per-step Adam/AdamW update of E74.4-G2 while admitting a bounded, disease-Prompt relation increment can improve the previously validated node-conditioned PSRR branch. ER-PSRR tests a second end-to-end candidate that places a gated Prompt relation change before the original SPGS/RES interaction and supervises the relation-derived disease evidence on legal outer-training rows.

The original SPromptGL MICCAI 2025 paper and official repository were reviewed together with the local `main.py`, `network.py`, `model.py`, `layers.py`, and `utils.py`. Original `PLN` performs transformer encoding, frozen CLIP Prompt grouping through its MLP, local disease-similarity weighting, four contiguous 36-value global pools, local/global RES interaction, then flattens CAM144 into `Outputlayer`. GFR constructs transductive patient graphs and applies graph convolution before CAM. The 16 local slots are latent feature slots and have no validated anatomical mapping.

E74.4-G2 source and its E61/E69.3/E53.3 initialization chain were checked through the executable `prepare(fold)` path. It restores each fold's E61 epoch-10 GFR/CAM/DARG state, three optimizer states, RNG, split and 4096 training relation pairs. E74.4 retains `val=test`, preprocessing/RFE before outer CV, the transductive graph and Fold2's 1-based subjects 377/407 overlap. The copied E81.7.2 read-only adapter reproduces the local NumPy pickle compatibility path without writing to historical experiment directories.

E81.7.2 uses node-conditioned Prompt weights, 15 real non-self neighbors plus Null, shared low-rank semantic mapping, Entmax1.5 competition, a zero-initialized 32-to-9 message projection and bounded CAM144 residual. Its gradient protection limits a gradient-space delta. This experiment instead limits the change in the optimizer's actual parameter update after accounting for optimizer moments and weight decay.

## Architecture compatibility correction

The DARG/E61 CAM class is the vendored `PLNNoPEN`, whose forward is `InputLayer -> Encoder/FeedForward -> Outputlayer`. It contains no SPGS Prompt MLP, local disease weighting, AvgPool1d global representation or RES. Therefore the requested direct insertion between DARG's SPGS and RES cannot occur inside the inherited model without changing the base architecture and its zero-residual output.

G2 adds a local compatibility branch at the pre-hooked `Outputlayer` input. It applies the original SPGS formula to the DARG CAM144 representation, projects the node-conditioned pair-relation message directly to a bounded local `deltaP`, then applies the original RES formula to `P` and `P + deltaP`. Only `RES(P + deltaP) - RES(P)` is added to the inherited DARG CAM144. It does not run the G1 post-RES local/global matching a second time. With `deltaP == 0`, output and loss are exact, and the shared reference-gradient comparison uses `atol=1e-8, rtol=1e-6` to accommodate floating-point cancellation in the paired RES difference. The original DARG `Outputlayer` still runs once. This correction preserves the required baseline identity and gives the relation branch the intended SPGS-to-relation-to-RES ordering. The SPGS MLP is necessarily newly initialized because E74.4 contains no corresponding SPGS weights. Its 525,577 parameters plus 49 gate parameters bring ER-PSRR to 533,434 trainable parameters. G2 is interpreted as a combined candidate, not a causal isolation of its gate, evidence loss, or insertion point.

## OP-PSRR update protection

For each shared optimizer group, compute the original same-state DARG gradient `g_B`, then compute the full objective gradient `g_F` after restoring Python, NumPy, PyTorch CPU/CUDA RNG and mutable model buffers. The enhanced pass consumes the public RNG stream once. Set `delta_g = g_F - g_B`. If the group dot product is negative, remove the component of `delta_g` parallel to `g_B` using one vector projection over the complete GFR, CAM, or DARG optimizer group.

For each candidate `alpha`, construct `g_alpha = g_B + alpha * delta_g_projected`. Use PyTorch's functional Adam implementation on isolated parameter and moment copies to compute the exact virtual update with the current optimizer settings, including Adam L2 decay, AdamW decoupled decay, step count, bias correction, AMSGrad, and `grad=None` skip behavior. Choose the largest feasible alpha found by the frozen bounded search such that `||U_alpha - U_B|| / max(||U_B||, epsilon_U) <= 0.02`. The search uses deterministic monotonicity probes followed by bisection. If probes are non-monotone, it uses a fixed 65-point alpha grid and chooses its largest feasible point. `epsilon_U=1e-12`; if the reference update norm is at or below that threshold, alpha is zero unless the candidate and reference updates are identical. This case is counted explicitly and is not treated as evidence of successful branch training.

Only one real update is performed for each historical optimizer. The virtual optimizer never mutates live optimizer state. A cloned native optimizer step must match the virtual update in preflight, including a parameter with `grad=None`. The independent PSRR AdamW receives the unmodified full-model PSRR gradient. The raw-residual term and G2 evidence loss both contribute through their actual autograd paths.

## ER-PSRR local relation branch

Both groups use the same fixed eight Prompt cache and semantic basis, node-level Prompt selection, pairwise relation scorer, Entmax1.5 competition, message values and bounded residual. ER-PSRR multiplies the ASD/TD relation contrast by a sigmoid gate built from Prompt match strength, ASD/TD evidence difference, visual relation score, and node Prompt-selection uncertainty. The gate remains in [0,1] and is initialized independently so it is neither a constant pass-through nor a constant zero gate. The shared zero-initialized output projection still makes the complete initial feature residual exactly zero. The gate is not conditioned on test labels.

ER-PSRR computes patient-level ASD and TD evidence from the sparse relation scores and normalized relation support. The margin objective is `mean(softplus(0.1 - y_star * (E_ASD - E_TD)))`, with `y_star=+1` for ASD and `-1` for TD and coefficient `0.005`. It uses only legal training rows. Fold2 rows 376 and 406 are excluded from the new branch residual and evidence loss; DARG membership, supervision, graph and propagation remain historical.

The local SPGS branch uses the original eight frozen CLIP RN50 text vectors and the original `4096 -> 128 -> 9` Prompt MLP layout. The four contiguous 36-value groups use the original `AvgPool1d(4,4)` operation and local/global RES equation. The branch does not add modalities or label anatomical regions.

## Training and decision

Only G1 OP-PSRR and G2 ER-PSRR are formal groups. Both independently restore each matching E61 epoch-10 state and retain the historical three optimizer settings, loss, data split, pair construction, update order, 450-epoch limit, `wait > 200` early stopping and strict test-fold ACC Best selection. New module initialization uses isolated fold-specific RNG. No per-epoch seed reset or test-driven hyperparameter changes are allowed.

The physical GPU is GPU1, identified by UUID `GPU-019e1211-c060-0210-0000-00006059cf27`. The only experiment directory is the `/model` fallback because the single preferred `/data` directory creation attempt returned EIO errno 5. At most ten workers are permitted. A real one-worker run will establish peak allocation before admitting additional workers.

The performance gate requires at least one candidate to exceed DARG in ACC and Hard-AUC over all ten folds and after excluding Fold2. Passing remains single-seed candidate evidence only. Mechanism statistics, Prompt variation, nonzero gradients, or improved Prob-AUC do not replace the performance gate.
