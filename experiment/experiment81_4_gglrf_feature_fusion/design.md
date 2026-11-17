# E81.4 — Global-Guided Local Relational Fusion (GGLRF)

## Research question and locked protocol

Can patient-global guidance select useful within-patient latent local relations,
retain all 16 receiving locals, and improve DARG through a bounded CAM144 feature
residual? G1 is global-guided unary attention; G2 changes only Key/Value inputs
to explicit products and signed differences. Exactly two groups × ten folds,
seed 0; no hyperparameter search, new classifier, logit correction or text prompt.

The official directory is `/model/spromptgl/experiment/experiment81_4_gglrf_feature_fusion`.
The preferred OSSFS mkdir returned errno=5 (EIO). Historical code/data/checkpoints
are read-only; the E81.3 read-only path adapter and checkpoint/RNG utilities are
reused. Only the final requested handoff is appended outside this directory.

## Source review

[SPromptGL paper, sections 2.1, 2.2 and 3](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf):
GFR builds modality-specific patient similarity graphs and propagates contextual
features. CAM exchanges information across modalities. PEN includes semantic
prompt selection (SPGS) and a RES local/global matching readout. Equation (6)
adds weighted local evidence to its global feature, then concatenates the modal
representations before classification. Its training combines graph and label
objectives. The paper reports ten-fold means and SDs of ACC/AUC/SEN/SPE, without
fully specifying all implementation-level AUC conventions.

[Official network.py](https://github.com/wanxixi11/SPromptGL/blob/main/network.py),
[layers.py](https://github.com/wanxixi11/SPromptGL/blob/main/layers.py),
[main.py](https://github.com/wanxixi11/SPromptGL/blob/main/main.py), and
[project implementation](https://github.com/tangaii/SPromptGL): the original
network has CLIP/text-guided local enhancement and an additive RES-like operation
before flattening to OutputLayer. Its released tensor implementation uses a
coordinate-wise local/global product and weighting; it is not identical to a
standalone abstract weighted-sum module. This experiment borrows global-guided
selection, and does not reinstall PEN or claim prompt evidence. Online source
SHA records and PDF SHA are stored in preflight provenance.

The actual inherited chain is E74.4 prepare → E74.1/E69 → E61 epoch-10 full
warmup → historical GFR/DARG → E51 PEN-free CAM → original OutputLayer.
The active OutputLayer is LayerNorm(144), Linear(144,36), GELU, Dropout(0.35),
Linear(36,2), log_softmax. Attention embedding is passed unchanged; the historical
OutputLayer does not use it in classification. The historical hard-prediction
AUC is balanced accuracy, distinct from probability ROC-AUC. ASD=0, TD=1.

E81.1 supplies the audited feature pre-hook and smooth residual-bound pattern;
E81.3 supplies fair 15-neighbor unary/pairwise construction, private initialization,
eta diagnostics, checkpoint restoration and public RNG comparison. E81.3 G2 ACC
91.7294% tied G1, below DARG 92.1891%; its current original branch lost 11 net
predictions and its correction recovered seven. E81.4 changes the integration
point and preserves 16 local messages rather than a single 32-D summary.

## Architecture (fixed before formal training)

All new differentiable features use `z_rel=z.detach()+0.1*(z-z.detach())`.
Global: Linear(144,32) → LayerNorm → GELU. Local: reshape 16×9,
Linear(9,32)+learned position, LayerNorm. Query: LayerNorm(Linear(64,32)
concatenated local/global); eight deterministic one-hot head offsets, as in
E81.3, give eight 32-D visual queries without extra random operations.

G1 Key=(sender,tanh(sender)); Value=(sender,tanh(sender),sender²).
G2 Key=(receiver*sender,sender−receiver); Value=(sender,product,difference).
Both have Linear(64,32) keys and Linear(96,32) values, no added key normalization.
Each receiver has all 15 distinct other tokens and one trainable null key; null
value is an exact zero buffer. Dot products /sqrt(32), softmax, weighted values;
mean only over eight heads, preserving N×16×32.

Read key: Linear(64,32) of local/message. Global read query: Linear(32,32).
Their dot-product softmax produces beta over 16 locals. Each selected local is
`16*beta*m_local`, shared zero-initialized bias-free Linear(32,9), then reshape
in the original coordinate order. No latent coordinate slice is identified as
an anatomical ROI or original data modality.

`bound=0.10*norm(z.detach())`;
`delta=bound*raw/sqrt(bound²+sum(raw²)+1e-12)`;
`z_enhanced=z+delta`. This preserves exact zero initialization and null-only
fallback, while smoothly limiting the residual to 10% of CAM norm. Detached bound
prevents a gradient route around eta. There is no margin gate.

The forward pre-hook replaces only OutputLayer's first input. The original CAM,
OutputLayer, dropout calls and historical total run exactly once. Loss is
`historical_out.total + 0.001*mean_legal_train(delta_ratio²)`; original graph,
0.05 pair, 0.01 adversarial and 0.01 preservation losses remain unchanged.
Original classifier/GFR/CAM/DARG and GGLRF train end to end. Eta scales only the
new input path. Once residuals become nonzero the original classifier's incoming
features and its backbone gradient change; equality to historical optimization
is guaranteed only at zero residual, not throughout training.

## Initialization, training and recovery

Each formal task independently calls historical prepare(fold); no historical
best checkpoint is a training initializer. Restore all three optimizers, model
states, split/pairs, Python/NumPy/CPU/device RNG. New module initializes on a
forked private CPU stream (salt=814000+fold), then transfers to device; G1/G2 have
identical state dictionaries and 16,384 parameters. No stochastic new layer.
The salt is not an additional public training seed.

Original steps: GFR Adam .0038/wd .11, CAM Adam .0038/wd .11, DARG AdamW
.001/wd .01; new AdamW .0005/wd .01 steps last. Start after epoch 10, maximum
450, stop when wait>200. Best uses strict test-fold ACC increase, earlier tie
retention. Best and latest serialize all four optimizers; latest also includes
all RNG, next epoch, wait, best and trace. Completed tasks are skipped.

Fold 2 historical overlap IDs 377/407 is preserved. During training only, their
new delta is zero and they are excluded from residual penalty. Evaluation uses
the same full model for every patient. This does not repair historical leakage
or transductive indirect influence.

## Preflight, audit and analysis

Ten-fold exact base-off checks include CAM/OutputLayer inputs, hidden/logp,
losses, backbone gradients, three updates and public RNG; one classifier call.
Both group initializations match. Deterministic unit checks test genuine global
conditioning, no cross-patient mixing, no duplicated/self candidates, tiny inputs,
smooth bounds and null identity. Real supervised first/second updates test new
head and internal gradients. Eta=0/.1/1 are diagnostics only. A serialized two-step
fresh-model replay tests full restoration. Code/config/history SHA is frozen
only after preflight, and verified by every worker and independent audit.

Formal scheduling uses actual free memory, observed peak, a 2× worker memory
factor and at least 30% scheduling headroom; at most ten independent one-thread
workers. Logs store scalars and RNG hashes, not per-epoch patient tensors.

Independent best reload recomputes six metrics with a separate sklearn routine,
checks latest full states, strict selection from logs, SHA and all common-epoch
RNG triples against DARG and between groups. Frozen inference interventions are
residual off, null-only, global guidance off (zero global inputs to both relation
query and read query, keeping local inputs/parameters), and G2 unary feature
substitution with G2-trained parameters. They do not select checkpoints.

Compare historical best, current trained residual-off branch, and full model;
report corrected/harmed/net for each transition. Record attention/beta diversity,
residual sizes/cancellation, hidden changes, global gradients and routed input
gradients over training. Report ten-fold and no-Fold-2 six metrics, fold differences,
W/T/L and descriptive paired intervals. Overlapping training folds preclude
strict independent-sampling inference. No ensemble, label-conditioned fallback,
new formal group, post-test tuning or automatic E81.5.
