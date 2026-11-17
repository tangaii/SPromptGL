# E81.6 — PRCL-RES

Three fixed groups, seed 0, original ten folds, no search: G1 Medical-Pairwise,
G2 Medical-Unary, G3 Anonymous-Pairwise. DARG E74.4-G2 is the sole training base;
all tasks independently restore the complete E61 epoch-10 warmup through the
audited historical prepare chain. Historical files remain read-only.

## Source review and scope

The SPromptGL paper (MICCAI 2025 paper 5129), Sections 2.1/2.2, describes
patient similarity graphs, multimodal CAM, disease-guided SPGS and RES.
Official network.py attr_words retains the original misspelling `Nromal`.
The actual RN50 implementation concatenates the first/last four text vectors
before its category projections. E81.6 instead preserves eight independent
conditions until message aggregation. It does not reinstall original PEN.
Official RES pools each contiguous 36-coordinate CAM slot with AvgPool1d(4,4),
then weights each 9-coordinate local vector by softmax(local*global). This is
coordinate matching rather than a learned 144-to-32 global encoder. The original
paper contains no fixed 10% residual cap. Official layers.py OutputLayer takes
the 144-vector and retains its original attention-embedding argument; its
forward does not otherwise use that argument.

KD-Brain KDBrain/models/model.py Attention_AggregateLayer injects a projected
text prior into Query. Its code uses tanh projection and an unconstrained scalar
text_gate, initially 0.5. We borrow that conditioning idea, with the specified
shared 1024-to-32 projection and sigmoid gamma. We do not copy its atlas,
BioMedBERT, multistep aggregation, or prior-KL objectives.

ItemizedCLIP Radiology_Tasks/src/open_mri/my_model.py PureAttentionPoolingBlock
and itemizedclip_loss.py _loss_with_attn_pool retain separate text-conditioned
visual representations. We borrow independent conditions only. ABIDE provides
no patient-level labels for these eight statements, so ILA/IIS/KTA/MPS losses
are not transferred. The requested CVF URL returned 403; its official repository
links the paper at https://arxiv.org/html/2512.11141v2, which was read instead.
Remote source content, Git blob identities and SHA256 are retained in
runtime/reviewed_sources.json and runtime/source_review.json. The frozen source
map also covers actual local training dependencies and prior results.

E81/E81.1 did not establish medical-text value. E81.3 pairwise did not outperform
its unary control. E81.4 motivates bounded feature fusion. E81.5 unbounded
fusion produced 161.84% mean residual, strong branch dependence and NO_GO;
it does not justify unbounded writeback here. E81.4 source identities distinguish
original frozen training code from subsequent audit-only corrections.

## Fixed model

CAM144 is reshaped to four slots, four 9-coordinate locals each. A shared
9-to-32 Linear plus learned positions and LayerNorm gives 16 latent tokens.
Eight frozen RN50 1024-vectors are independently L2-normalized and projected by
one trainable Linear. Query is LayerNorm(Wq*u+sigmoid(gamma_logit)*Wt*t), gamma
initially 0.5. The eight Query conditions are prompts, without additional heads.

G1/G3 Key uses product and absolute difference; Value uses sender, product,
signed difference. G2 Key uses sender and tanh(sender); Value adds sender squared.
All use 15 distinct other tokens plus one learnable Null Key and zero Null Value.
Normal and abnormal messages are averaged only after per-prompt attention,
compete via a zero-initialized Linear(64,2), and pass through a shared
Linear(64,32,bias=False). The bias-free choice guarantees exact zero output
from zero messages throughout training; it is identical in every group.

A zero-initialized shared Linear(32,9,bias=False) writes each corresponding
local position. Slot-wise global pooling and softmax(d*g, feature dimension)
produce the RES-weighted local residual. Its smooth norm bound is exactly
rho=0.1, eps=1e-12, with detached CAM reference norm. Every differentiable new
CAM input uses detach(z)+0.1*(z-detach(z)); the identity path uses unscaled z.
Only one historical OutputLayer call occurs via an input pre-hook. The original
LN144 -> Linear144,36 -> GELU -> Dropout0.35 -> Linear36,2 stays trainable.

## Conditions, initialization and losses

Original eight texts and their order are fixed in prcl_res.py and the RN50 cache.
Anonymous vectors use one private CPU Gaussian generator with salt 8160000,
and match each medical vector's raw L2 norm. No seed selection is permitted.
Their Gram geometry differs; a medical advantage cannot alone prove clinical
semantic understanding. Full input Gram matrices and similarity distributions
are reported. Raw frozen CLIP weights and tokenizer identities are recorded.

Every fold uses private CPU module salt 816000+fold. All three groups have
identical trainable parameters at initialization; fixed buffers contain both
cached condition sets, and only the selected set participates in computation.
No new stochastic training operation is introduced.

Loss = historical_out.total + .001*L_res + .001*L_geo. Historical total remains
weighted ClsLoss + graph + .05*pair + .01*adv + .01*preserve. L_res is relative
residual norm squared over legal training rows. L_geo compares normalized
projected-text Gram to the chosen condition set's own raw normalized Gram.
G3 never uses the medical Gram target. L_geo updates only the text projection;
it may be nonzero at zero output, so Base-off compares the historical loss and
backbone gradients rather than requiring complete total-loss equality.

Four optimizers retain the specified order, learning rates and decay. Fold 2
training zeros new residual rows 376/406 and excludes their residual penalty;
evaluation applies the full model to all rows. The historical overlap remains.
Strictly increasing test ACC selects best; ties preserve the earlier epoch.
Maximum epoch 450, start after epoch 10, wait>200, no per-epoch RNG reset.

## Preflight, execution and audit

Ten-fold exact zero-residual forward, hidden, historical losses, original
gradients, three optimizer steps and RNG parity are mandatory. Other checks
cover all-group parameter identity, private RNG, prompt effects before/following
a deterministic nonzero test head, official pooling/RES equality, 120 unordered
pairs, no row mixing, Null identity after learning, first/second-step gradients,
eta routing and a serialized fresh-model two-step replay.

The scalar journal is replaced atomically and read back every epoch. Latest
also contains its complete rows, RNG trace, best state, all four optimizers,
early stopping and next epoch. This prevents silent append loss seen in E81.5.
GPU admission uses measured peak and current free memory, with at most ten
one-thread workers. No completed task is retrained.

Audit independently reloads 30 best/latest states and recomputes all six
metrics, log coverage, strict best selection, RNG traces, source identities and
mechanisms. Frozen interventions never select checkpoints: residual-off,
Null-only, prompt contribution off, anonymous replacement, group swap,
individual cross-group prompt swaps/deletions and global-off. Global-off uses
uniform 1/9 coordinate matching, preserving scale while removing global input.

G1-vs-DARG, G1-vs-G2, G1-vs-G3 use all ten folds and exclude Fold 2. Descriptive
paired t intervals do not assume independent CV training sets. Backbone-off
metrics are counterfactual at the complete-model-selected checkpoint, not an
independently selected DARG. Historical val=test, pre-CV preprocessing/RFE,
transductive graphs and Fold 2 overlap remain limitations. Tokens are latent
coordinates, never validated ROIs or pathology labels.

Three conclusions are judged separately: complete baseline gain, pairwise
increment, and actual medical-input increment. PRELIMINARY_GO requires both
primary metrics above DARG, supportive nine-fold and control results, positive
net corrections, noncollapsed mechanisms and complete independent audit. No
post-result hyperparameter changes or successor experiment are authorized.
