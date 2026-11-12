# Experiment 78.4 — PSP-AL design freeze

## Audit record

PPAL (MICCAI 2025) was read from the supplied proceedings PDF and the official `diaoyq121/PPAL` repository. Section 2.3 and Eqs. (8–12) define text-derived disease priors, prototype aggregation attention, text aggregation attention, three progressive updates, cosine-similarity diagnosis, and `L_overall=L_f+L_c`. The public `engine3.py` actually optimizes only one `CrossEntropyLoss(y_pred,y)` in its training step; the coarse/fine loss described by Eq. (12) is not both applied there. The public `ppal.py` uses 512-dimensional learnable AD/CN embeddings and `nn.MultiheadAttention` without `batch_first=True`; its forward passes tensors shaped as if batch-first, so PSP-AL does not copy that layout. `attention_layer.py` explicitly sets `batch_first=True` for its other attention path, illustrating the inconsistency. `textprompt.csv` contains one long AD and one long CN description.

SPromptGL was checked from the local paper and official `wanxixi11/SPromptGL` repository. Its GFR, DGCN, CAM/PLN and eight prompt ordering were compared with the server-side E74.4 implementation. E78.4 uses the SHA-checked E74.4-G2 Positive-DCR/DARG plus CAM144 and the historical `OutputLayer` once per forward. It does not reintroduce PEN, split CAM144 into four 36-dimensional modality tokens, or replace DARG.

The exact server-side E74.4 `soft_positive_dcr.py` computes:

`out["total"] = out["cls"] + out["graph"] + 0.05*out["pair"] + 0.01*out["adv"] + 0.01*out["preserve"]`.

`out["cls"]` is the historical `utils.ClsLoss(logp, y, train, weights)` (weighted NLL on log-probabilities); `out["logp"]` is the normalized OutputLayer log-probability. E78.4 replaces exactly this classification contribution with the legal-support final classification loss and leaves graph, pair, adversarial and preservation terms byte-for-byte in the base path.

## Frozen groups

Six groups × ten folds, seed 0:

- G1: SPromptGL prompts, K=1, restricted logit residual.
- G2: SPromptGL prompts, K=1, PPAL-style direct prototype logits.
- G3: SPromptGL prompts, K=2, restricted logit residual.
- G4: SPromptGL prompts, K=2, PPAL-style direct prototype logits.
- G5: E78.1 anonymous prompts, K=1, restricted logit residual.
- G6: E78.1 anonymous prompts, K=1, PPAL-style direct prototype logits.

All groups instantiate the same two-candidate 32-dimensional module; K=1 is a strict mask of the second candidate, so parameter count and initialization procedure are shared. Only prompt bank, active K, and the pre-registered final classification rule differ.

## PSP-AL implementation

The current CAM144 patient matrix `H∈R^(871×144)` is captured with an OutputLayer pre-hook and is never recomputed or detached. Legal support is `train_indices \ test_indices`; Fold 2 historical overlap rows 377 and 407 (1-based) are excluded from new support and new classification supervision while the original base loss path remains unchanged. No LOO is used. No static category mean substitutes for the patient feature library.

Frozen RN50 text coordinates are the SHA-checked E78.1 shared 16-dimensional SVD geometry. A trainable shared `16→32` text map and `144→32` patient map produce semantic and patient tokens. Each disease has four text tokens and up to two query tokens. Query initialization is projected class-text mean plus a small private-RNG offset; K=1 masks query 2. Three iterations share one explicit `batch_first=True` 4-head patient Cross-Attention, one text Cross-Attention, one `32→64→32` FFN and LayerNorm. Prototype attention is `[1,K,32]` querying `[1,N_class,32]`; text attention is `[1,4,32]` querying the same class-specific patient sequence. Softmax therefore runs along the legal class-patient axis. Attention dropout is zero.

Final class prototypes are normalized 32-dimensional tokens. Patient-to-prototype cosine scores are pooled by `0.25*logmeanexp(cos/0.25)`, including `-log(K)` so K=2 has no count advantage. Restricted groups use `logp_DARG + [delta/2,-delta/2]`, where `delta=0.5*sigmoid(lambda_raw)*tanh((r_ASD-r_TD)/0.5)` and lambda is initialized to 0.1. Direct groups use fixed `10*[r_ASD,r_TD]` logits. Both paths use one final weighted NLL on legal support labels.

The frozen loss is:

`L_new = L_E74.4 - L_cls_E74.4 + CE(L_final, legal_train_labels)`.

The original graph, pair, adversarial and preservation losses and their coefficients remain unchanged. The new AdamW (`lr=5e-4`, `weight_decay=0.01`) steps after the three historical optimizers. New initialization uses a private `fork_rng`; the attention path contains no random operation, so the public DARG/CAM RNG sequence is unchanged.

## Mandatory controls

Preflight covers 10/10 base-off parity, explicit attention shapes, patient-axis softmax, K=1/K=2 and asymmetric support masks, support permutation, K=2 query permutation, LogMeanExp count normalization, ASD/TD prompt mapping, test-label invariance, finite end-to-end gradients, two-step recovery, and text/attention interventions. Formal Checkpoints store original and new optimizer states, module buffers, RNG, source/config SHA, final probabilities/predictions, loss history, attention diagnostics, and legal support identities.

Historical limitations are retained and reported: `val=test`, test-ACC selection, preprocessing/RFE before outer CV, transductive graph, and Fold 2 overlap. These are not unbiased clinical generalization estimates.
