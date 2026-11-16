# Experiment 81.2: End-to-End Reliability-Gated Relational Correction

## Research question

This experiment tests whether directed interactions among the 16 latent 9-D coordinates obtained from the original CAM144 representation add diagnostic value beyond an equally sized self-local correction branch. These coordinates are learned CAM partitions, not ROIs, brain regions, or independently validated pathological concepts. The formal design has two groups and ten historical folds: G1 repeats the receiving local token as each candidate source; G2 uses each of the other 15 local tokens. The repeated G1 candidates preserve tensor shape and trainable capacity, but change the softmax competition with Null and therefore introduce a known inductive-bias difference.

## Source review

The MICCAI 2025 SPromptGL paper describes GFR as modality-specific patient relations followed by graph convolutional propagation (Section 2.1). Section 2.2 describes CAM cross-modal aggregation and PEN: SPGS scores local features against frozen CLIP text embeddings and selects the largest semantic similarity (Eq. 5); RES then combines the selected local representation with the global representation (Eq. 6). These are local-to-global operations in the original model. The present experiment uses the audited PEN-free E61/E74.4 CAM implementation and does not equate its 16 fixed CAM coordinate slices with the paper's learned local concepts.

The official SPromptGL repository and the project branch were checked alongside the local E74.4-G2 code. The actual base path is `prepare(fold) -> soft_positive_dcr.forward_state() -> rudag_forward() -> PEN-free PLNNoPEN -> OutputLayer`. The base loss is the historical weighted `ClsLoss` plus graph, pair, conditional adversarial, and preservation terms. E74.4's validated helper chain restores the E61 epoch-10 full state, including GFR/CAM/DARG parameters, three optimizer states, fold membership, 4096 training pairs, and RNG. Existing checkpoint selection uses strict test-fold ACC improvement and validation equals test.

E81 and E81.1 provide read-only historical evidence. E81's hard residual cap left many samples in a clipped-gradient region; E81.1's smooth cap and strict zero Null writeback repaired those implementation issues, but neither run established incremental classification value. Its nonlinear normal/abnormal competition and prompt effects were small, and all E81.1 groups remained below DARG in ACC. E81.2 therefore removes prompts and CAM writeback from the hypothesis and contrasts cross-local information against self-local correction.

## Model and training

The original OutputLayer runs once. A pre-hook records its CAM144 input; a forward hook receives its original `(logp_base, hidden)`, computes a bounded binary logit-margin correction, and returns `(logp_final, hidden)`. The original classifier, hidden representation, attention argument, and dropout call remain intact. The correction is `0.75*tanh(raw_margin)*g_unc*g_rel`; `g_unc` uses the detached original log-probability margin and fixed constants 0.5 and 0.3. A patient relation summary and non-Null/entropy statistics determine `g_rel`. The correction output is zero-initialized and bias-free, giving exact initial and Null-only identity behavior while retaining a gradient to the correction head. G1/G2 share parameter names, shapes, initialization, AdamW settings, loss, and forward path.

The classification term is `0.75*L_final + 0.25*L_base` using the historical weighted NLL helper. The unchanged DARG graph, pair, adversarial, and preservation losses retain their E74.4-G2 coefficients. A 0.001 squared correction penalty is added. The original three optimizers step in historical order, followed by the fourth AdamW correction optimizer. Fold 2 rows 377 and 407 retain the original DARG training path but have zero direct correction during training.

New module initialization uses a private CPU RNG scope. No PRCL dropout or sampling is introduced. Training and evaluation each run the DARG/CAM backbone once. Public RNG states are recorded before training, after training, and after evaluation each epoch.

## Pre-registered interpretation

The primary contrasts are G2 versus DARG and G2 versus G1. G1 versus DARG measures generic self-local correction. G2 versus G1 is the test of cross-local value, with the repeated-candidate/Null competition caveat stated above. The final analysis separates the current trained base logp from the corrected logp and from the historical E74.4-G2 best predictions. Checkpoint selection still uses the historical test-ACC rule, CV preprocessing remains global, and the patient graph remains transductive; results are protocol-matched exploratory estimates, not unbiased external validation. Fold 2 contains historical train/test overlap at global rows 377 and 407. Ten-fold intervals are descriptive because training partitions overlap.

## References

- SPromptGL paper: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf
- Official implementation: https://github.com/wanxixi11/SPromptGL
- Project E74.4-G2 branch: https://github.com/tangaii/SPromptGL/tree/upload/experiment74-4-soft-positive-dcr
- Historical E81 branch: https://github.com/tangaii/SPromptGL/tree/upload/experiment81-prcl
- Historical E81.1 branch: https://github.com/tangaii/SPromptGL/tree/upload/experiment81_1-prcl-v2
