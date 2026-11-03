# Experiment 70 — GCDSI Design Freeze

## Research question

Experiment 70 tests whether the 320-dimensional patient graph state produced by the verified Experiment 53.3 third group can condition four different queries from the unchanged CAM's four 36-dimensional modality representations. Each query attends to both frozen disease text tokens, `[ASD, TD]`, then injects a gated, bounded residual before the existing 144-dimensional `OutputLayer` input. There is no new classifier, auxiliary semantic objective, graph writeback, or PEN local partition.

For patient i and CAM modality m:

`q(i,m) = LayerNorm(W_h h(i,m) + W_f F(i) + q_CLS)`

`A(i,m) = softmax(q(i,m) K^T / sqrt(36))`, `u(i,m) = A(i,m) V`

`g(i,m) = sigmoid(MLP([h(i,m), q(i,m), u(i,m)]))`

`h'(i,m) = h(i,m) + 0.5*tanh(gamma_raw) * g(i,m) * W_O u(i,m)`

The existing OutputLayer receives `flatten(h')`. `gamma_raw` starts at zero, so the initial model is exactly the base model. `W_O` has a small nonzero initialization so classification can update `gamma_raw` at the first active step; after gamma moves, query, text, gate, CAM and graph parameters receive gradients. No GCDSI dropout is used.

## Paper and source audit

SPromptGL's paper and official code describe GFR patient graph propagation, CAM cross-modal patient interaction, and PEN prompt-weighted local enhancement. The source `network.py` has eight text strings, grouped as four broad normal descriptions and four altered descriptions. The source CAM's `OutputLayer.forward(x, attn_embedding)` accepts the attention argument but its body classifies `x` only. The local Experiment 51/53.3 `PLNNoPEN` therefore sends 4x36 CAM features flattened to 144 directly to that existing layer. GCDSI inserts before it and does not restore SPGS/RES or the original 4x4x9 PEN operations.

SABER equations (8)-(10) motivate graph-conditioned Graph-CLS queries, patient-text keys/values, cross-attention and gated residual injection. SABER separately uses disease-level text in its semantic alignment objective; Experiment 70 uses fixed ASD/TD text as inference-time keys/values and adds no separate semantic loss. This is a related design adaptation, not a claim of full SABER reproduction. The paper is [arXiv:2607.01901](https://arxiv.org/abs/2607.01901); the public paper describes decision-level injection of patient-specific textual embeddings.

BrainPrompt's ABIDE text file contains two exact strings in `[TC, ASD]` order. Experiment 70 uses them unmodified and orders cached embeddings `[ASD, TD]`. The official implementation's LLM2Vec cache is not used for the main groups: all banks use the same frozen OpenAI CLIP RN50 cache so encoder differences do not confound text source. The original text is available at [BrainPrompt abide_label.txt](https://github.com/AngusMonroe/BrainPrompt/blob/main/data/prompts/label_prompts/abide_label.txt).

G1 is those two BrainPrompt texts. G2 uses the exact eight strings in local `network.py`: `Nromal brain volume`, `Uniform cortical thickness`, `Symmetrical brain structure`, `Normal ventricular morphology`, `Altered brain volume`, `Abnormal functional connectivity`, `White matter integrity issues`, and `Different activation patterns`. Each text vector is normalized, the first four are averaged and normalized as TD, and the last four are averaged and normalized as ASD; the bank is then reordered `[ASD, TD]`. This controls the number of attention tokens and is not a reproduction of original PEN's grouped MLP and local feature weighting. These broad normal/altered descriptions are not asserted to be ASD-specific biomarkers.

G3 uses two fixed anonymous vectors from a private CPU generator seeded 700070. Their norms and pairwise cosine match G1's frozen text vectors. It uses no patient features, labels, or evaluation outcomes.

## Base and training protocol

The base is Experiment 53.3 `CADR_WD_0p01_GFR_WD_0p11`, verified from the `/model` sweep manifest, per-fold task records, checkpoint SHA and fold-metric rows. The reference has 10/10 checkpoint readbacks and reproduces the supplied metrics: ACC 92.0742±2.8427%, Hard-AUC/BA 91.8653±2.8348%, Probability-AUC 93.6684±3.2619%, SPE 94.6392±3.5168%, SEN 89.0915±3.5263%, and ASD-F1 91.2470±3.0658%. Historical checkpoints remain read-only.

Each fold rebuilds from the Experiment 43.1 untrained anchor through the 53.3 third-group builder. The matching Experiment 59 common ten-epoch warmup cache is reused only after checking split, pair/cache, anchor-bound initialization, shared/added parameter and initialization RNG fingerprints. Three groups share the same fold cache. GCDSI is isolated from the main initialization RNG using `torch.random.fork_rng` and seed `700000+fold`; corresponding groups therefore begin with identical GCDSI weights. Active training begins at epoch index 10. The original CADR/GFR/CAM optimizers, loss, fold sampler, scheduler-free 450 epoch budget, early stopping, and strict ACC-best rule are retained. GCDSI uses separate AdamW at LR 0.001 and WD 0.001.

The data are ABIDE A, 871 patients, ASD=0/TD=1, PHENO/ANAT/FUNC/CORR dimensions 48/6/10/256. Historical global pre-CV RFE/standardization, transductive graph and `val=test` checkpoint selection are retained. Metrics are single-seed exploratory evidence and do not estimate unbiased external generalization.

## Frozen interpretation gates

A group clears the engineering base gate only with at least +0.5 pp mean ACC, at least 6/10 ACC wins, Probability-AUC decline no larger than 0.5 pp, and SEN decline no larger than 1 pp. Independent text value requires G1 to have positive mean ACC and at least 6/10 wins over G3. Attention must remain nonconstant, gamma/residual and graph-query parameters must update, and gates must not saturate. These are descriptive gates, not significance or clinical-validity claims. No extra groups or post-result parameter search are planned.
