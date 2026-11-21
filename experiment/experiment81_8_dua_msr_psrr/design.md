# Experiment 81.8 — Disease Prompt-guided Local Relational Reasoning

## Frozen Sources and Provenance

The method is implemented on the server's audited E74.4-G2 Positive-DCR/DARG path. The E74.4 `prepare(fold)` entry restores the corresponding E61 epoch-10 state, including GFR/CAM/DARG parameters, buffers, all three optimizer states, Python/NumPy/Torch CPU/CUDA RNG, the historical split, and the 4096 training relation pairs. Each group/fold calls `prepare(fold)` independently. G1 and G2 never share trained weights.

The relevant source implementation confirms the active CAM is `PLNNoPEN`, which has no active PEN. E74.4 emits CAM144 and its original OutputLayer is intercepted only at its pre-hook. The E81.7.2 relation implementation reshapes CAM144 into `[N,16,9]` latent slots. These slots are not anatomical ROIs. For each receiver it enumerates the 15 non-self senders plus a Null candidate. The frozen RN50 cache contains the same eight Prompt strings and semantic basis as E81.7.2-G1, with the first four assigned to TD and the last four to ASD.

The MICCAI 2025 paper describes GFR, frozen CLIP semantic prompts, prompt-guided local representations, and relation-aware embedding. Its open-access record links the authors' repository at `wanxixi11/SPromptGL`. The user-provided `tangaii/SPromptGL` URL was also checked. Server-side E74.4 and E81.7.2 sources remain the operative implementation authority for this experiment.

## Shared Forward Path

`four modalities -> E74.4-G2 GFR/DARG -> PLNNoPEN CAM -> CAM144 -> E81.7.2 node-level Prompt-conditioned sparse local relations -> group-specific reasoning -> E81.7.2 RES-style local/global integration -> bounded CAM144 residual (rho=0.1) -> original OutputLayer`.

Both groups preserve the original OutputLayer weights, LayerNorm, Dropout, class mapping, and call semantics. There is no additional classifier, modality split, cross-patient prototype bank, or LOO design. The relation module is zero-residual at initialization and uses private initialization contexts that restore public RNG state.

## G1 DUA-PSRR

A shared `18 -> 32 -> 8` relation map consumes `(V_a, V_b-V_a)`. It interacts with each receiver's node-level disease Prompt representation in the eight-dimensional centered semantic basis. A zero-initialized final projection makes the initial score correction exactly zero. The correction is bounded by `0.5*tanh(dot(h,p))` and is applied to real-neighbor scores before the existing Entmax1.5 competition. Null remains a distinct zero-message candidate.

On legal training patients only, the correct-class log-probability margin from the single original OutputLayer forward is differentiated with respect to enhanced CAM144. Its detached gradient yields a cosine first-order local utility proxy for each true neighbor direction. Weighted edge BCE uses temperature 0.20 and confidence `clamp(abs(u)/0.20,0,1)`. A soft Null BCE compares the correct-class Null score to the logsumexp of its real-neighbor scores. Hard training examples receive weight 1.5. The auxiliary coefficient ramps over the first 20 post-warmup training epochs to 0.02. The auxiliary gradient is requested only for relation-module parameters. It does not claim exact causal edge effect.

## G2 MSR-PSRR

The unmodified E81.7.2 node-level disease scores produce separate ASD and TD Entmax distributions. Their real-neighbor portions form 16x16 operators without renormalizing away Null probability. For each disease, the module computes first order `A V`, exactly two-hop `A(A V)`, and high-frequency `V-A V`. A shared 27-to-32 projection processes each disease's scales. A shared 96-to-32 projection combines the two results and their delayed difference, then feeds the original RES-style integration and bounded CAM residual.

Training-only stability uses a deterministic 1% feature-scale perturbation generated from fixed sinusoidal coordinates and compares full neighbor-plus-Null distributions. Training-only supervised contrastive loss uses legal training rows, preferring cross-site same-class positives and same-site opposite-class negatives. When no cross-site positive exists, same-class train-only positives are used. Anchors with no valid positive or negative are skipped. The contrastive vector is pooled relation evidence, not a class prototype or inference head. Both auxiliary gradients are requested only for relation-module parameters.

## Gradient and RNG Protocol

The shared GFR/CAM/DARG protection reuses E81.7.2-G1's exact implementation. It computes the original DARG reference gradient `g_B`, restores RNG and model/CAM buffers, computes the enhanced main-loss gradient `g_F`, projects a negative conflict component, and caps each optimizer group's difference to `kappa=0.05`. The original three optimizer updates occur in their historical order. One independent relation AdamW step follows. Auxiliary losses are excluded from `g_F` and are added only to relation-module gradients.

Every training step uses one reference and one enhanced historical forward, each invoking the original OutputLayer once. The utility gradient is derived from the enhanced forward's existing graph. G2's stability branch calls only the relation module. Neither group reseeds per epoch. The E81.7.2 historical optimizer includes a raw-residual penalty with coefficient 0.001. The registered E81.8 total-loss equations specify only original DARG plus the new group-specific auxiliary terms, so this E81.7.2-only penalty is omitted and recorded as a deliberate objective difference.

Fold 2 retains the historical train/test overlap for 1-based rows 377 ASD and 407 TD. In zero-based indexing, 376 and 406 receive zero new residual during training and are excluded from all new labeled auxiliary supervision. Original DARG training membership and evaluation remain historical.

## Locked Training and Evaluation

Seed 0, ten folds per group, 450-epoch maximum, historical `wait > 200` early stopping, `val=test`, strict ACC improvement Best selection, and earlier-epoch tie retention. Original CV split, pre-CV RFE/scaling, transductive patient graph, label mapping, and metric definitions are unchanged. Hard-AUC is the historical hard-prediction balanced accuracy. Prob-AUC uses continuous ASD probability.

The registered experiment contains only G1 DUA-PSRR and G2 MSR-PSRR. Both are complete independent methods, each trained end-to-end from its own matching E61 warmup restoration. No extra groups, searches, or simplified training arms are included.
