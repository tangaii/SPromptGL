# Experiment 72 Design Freeze

## Objective

Test disease-prompt auxiliary supervision on the frozen Experiment 53.3 third-group CADR-GFR + CAM model. The original prediction path remains `320-D GFR -> CAM [N,4,36] -> flatten 144 -> original OutputLayer`. The auxiliary heads read the same final CAM tensor and never alter prediction logits. There is one CAM forward per model pass.

This is an adaptation of disease-level supervision, not a reimplementation of BrainPrompt or SABER. Experiment 59 already tested global BrainPrompt-style and SABER-style supervision: both reached 91.8443% ACC, below the 53.3 reference at 92.0742%. The current G5/G6 revisit the global losses with frozen RN50 text and a common low-rank projection; they are not a first test of the general supervision idea.

## Source Audit

- SPromptGL MICCAI 2025 paper ([paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [official repository](https://github.com/wanxixi11/SPromptGL)) and official `network.py` were checked. Its GFR operates on four modality feature blocks. Its PLN contains the PEN; PEN uses prompt-guided local feature weighting and RES integration before the original OutputLayer. The eight prompts and their source typo `Nromal` are preserved. Experiment 72 does not reuse PEN or change GFR/CAM/OutputLayer.
- Experiment 53.3 source and reference artifacts identify the selected base `CADR_WD_0p01_GFR_WD_0p11`. The source and caches are byte-copied into `.runtime/vendor`; each fold rebuilds from the 43.1 untrained anchor, then loads the shared ten-epoch E59 warmup. Best 53.3 weights are read only for provenance and comparison.
- BrainPrompt MICCAI 2025 ([paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [official repository](https://github.com/AngusMonroe/BrainPrompt)) official `brainpromptg_net.py` and `brainpromptc_net.py` define normalized graph and transformed-label vectors, cosine logits, and `F.cross_entropy` in `compute_label_loss()`. The official ABIDE label text order is `[TC, ASD]`; Experiment 72 reorders the exact two texts to `[ASD, TD]`. Its released encoder is Llama-encoder-1.0B; this experiment instead fixes CLIP RN50 to control the text-encoder variable, so groups are described as BrainPrompt text source plus BrainPrompt-style supervision.
- SABER ([ICME 2026 paper](https://arxiv.org/abs/2607.01901)) equations (8)-(10) define Graph-CLS conditioning, patient-text cross-attention, and gated residual injection; (11) is the ordinary classifier loss; (12) is disease-class contrastive supervision; (13) combines them. The experiment implements only the equation (12) class loss as an auxiliary adaptation. It does not claim a complete SABER reproduction or adopt its other architecture.
- Experiment 60's SHA-verified frozen CLIP RN50 cache supplies both prompt sources. Experiment 70, 70.1, and 71 manifests, source, and reports were read and preserved in `.runtime/history`; this experiment does not use their best model parameters.

## Frozen Text Geometry

BrainPrompt's two official descriptions and SPromptGL's eight exact source descriptions are encoded by frozen CLIP RN50. For SPromptGL, each vector is L2-normalized, the first four are averaged for TD, the last four for ASD, each centroid is normalized, and the order is `[ASD, TD]`.

The ten raw source embeddings are arranged as rows and decomposed with CPU float64 SVD under one CPU thread. The first `r=10` right singular vectors form one fixed orthonormal basis `B`; each axis sign is anchored by its largest-magnitude entry to remove SVD sign ambiguity. Each two-class bank is mapped to `T @ B`, then normalized. The frozen basis SHA-256 is `3e79ee238447ff47ad1029408b99db94c3b4c4fbc657aa5a5251754563dc3868`. Projection preserves the BrainPrompt class cosine `0.9847825654` and SPromptGL class cosine `0.9070363700` to below `1.2e-16` absolute error.

## Auxiliary Representation and Losses

For one captured CAM tensor `H in R^(N x 4 x 36)`, fine-grained patient vectors use a shared `P_f: Linear(36,r)` at each of the four positions. The global vector uses `P_g: Linear(144,r)` on `flatten(H)`. Both are L2-normalized before dot products with frozen class prototypes.

For `s(i,c)=u_i^T t_c`, BrainPrompt-style loss is `CE(s/tau=1, y)`. SABER-style adaptation is `-log(exp(s_y/0.1) / sum_c exp(s_c/0.1))`. The two are the same cross-entropy/log-sum-exp expression at `tau=1`; this experiment compares temperature, not distinct loss theory. A seeded value-and-gradient equivalence test passes with zero loss difference and maximum gradient error below `2e-18`.

Fine loss is `0.10 L_PHENO + 0.10 L_ANAT + 0.10 L_FUNC + 0.70 L_CORR`. Joint groups use `0.5 L_global + 0.5 L_fine`; global-only groups use `L_global`. This is a fixed auxiliary budget allocation comparison, not an unchanged global loss plus an added fine loss. Auxiliary losses receive only outer-training CAM rows and labels.

The two patient projections are initialized by Xavier uniform with zero bias, under `fork_rng` seed `720000 + fold`; all six groups share identical same-fold initialization. A separate AdamW (`lr=0.001`, `weight_decay=0.001`) updates both projections. In global-only groups `P_f` is constructed but does not enter the loss or update.

## Gradient Budget and Training

For each fold and group, after restoring the epoch-10 warmup, deterministic eval mode computes the original weighted classification loss and unscaled auxiliary loss on training patients only. `alpha=0.10*||grad_H(L_cls)||/(||grad_H(L_sem)||+1e-12)` is frozen before training. Nonfinite/degenerate values or `alpha>10` fail the fold; there is no clipping or performance-based recalibration. Calibration must leave parameters, all optimizer states, and every main RNG state unchanged.

The total loss is exactly `L_base + alpha*L_sem`, where `L_base=L_cls+L_graph+0.05 L_pair+0.01 L_adv+0.01 L_preserve`. The inherited ten-epoch warmup precedes semantic training; there are at most 450 total epochs and stopping is `wait>200`. Checkpoint selection retains strict ACC improvement and historical `val=test` behavior. Inference uses only the original classifier.

## Six Frozen Groups

| ID | Text source | Temperature | Grain |
|---|---|---:|---|
| G1 | BrainPrompt-2 | 1.0 | global + fine |
| G2 | BrainPrompt-2 | 0.1 | global + fine |
| G3 | SPromptGL-8 | 1.0 | global + fine |
| G4 | SPromptGL-8 | 0.1 | global + fine |
| G5 | BrainPrompt-2 | 1.0 | global only |
| G6 | BrainPrompt-2 | 0.1 | global only |

Only G1/G5 and G2/G6 identify the grain contrast. G3/G4 have no SPromptGL global-only counterpart, so their results cannot isolate a text-source-specific grain effect.

## Frozen Exploratory Decision

A candidate must improve mean ACC over 53.3 by at least 0.5 percentage points, strictly win at least six folds, keep Probability-AUC within -0.5 points and SEN within -1.0 point, and pass all state/RNG/checkpoint audits. A multi-grain claim additionally needs a positive G1-G5 or G2-G6 paired ACC difference and at least six strict fold wins. Even a GO result remains a one-seed exploratory signal, not a validated clinical model or established MICCAI contribution.

Historical `val=test`, preprocessing before outer CV, and the 871-node transductive graph remain known limits.
