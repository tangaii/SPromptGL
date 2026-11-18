# Experiment 81.7 PSRR

Design status is frozen before implementation and training. The preferred `/data/adodas/SPromptGL-main/experiment/experiment81_7_psrr` directory returned `Input/output error` at creation. The isolated experiment therefore resides at `/model/spromptgl/experiment/experiment81_7_psrr`. No historical source, checkpoint, dataset, preprocessing artifact, or result is modified.

## Source Review

Sources reviewed: [MICCAI 2025 SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [official author repository](https://github.com/wanxixi11/SPromptGL), [KD-Brain MICCAI 2026 paper](https://papers.miccai.org/miccai-2026/paper/0073_paper.pdf), [Entmax-1.5 paper](https://arxiv.org/abs/1905.05702), and [official Entmax implementation](https://github.com/deep-spin/entmax/blob/master/entmax/activations.py). The MICCAI 2025 paper and author source were reviewed. Section 2.1 constructs a cross-patient graph in GFR. Section 2.2 first pools prompts into two PEN category vectors, uses SPGS to score each local feature slot, then applies RES. Equation 6 and the author's `network.py` organize CAM144 as `[N,4,4,9]`, pool each contiguous 36-coordinate slot with `AvgPool1d(4,4)`, and compute `softmax(local*global)*local`; the original RES adds the resulting local residual to its input. The original `OutputLayer` is LayerNorm144, Linear144-to-hidden, GELU, Dropout, and a classifier. Its `attn_embedding` argument is unused. The author's `main.py` optimizes classification and graph losses end to end and selects the checkpoint by test-fold performance. The paper evaluates ABIDE with ten folds and ACC/AUC/SEN/SPE.

PSRR uses the historical CAM144 ordering and borrows the local/global coordinate interaction from RES. Its substantive change is that each fixed Prompt directly affects scores for patient-internal candidate pairs before relation selection. It creates separate ASD and TD Entmax sparse supports, propagates values on both supports, and projects their difference through the RES-style feature interaction. These 16 slots are latent feature locations, not anatomically verified ROIs. Unlike the paper's PEN/SPGS, PSRR does not merge eight prompts into two learned category vectors or use a post-hoc per-slot maximum Prompt match. Unlike original RES, the learned residual is smoothly bounded to ten percent of CAM144 and starts at zero.

KD-Brain was reviewed at commit `180873a5dc2e6b5caf862528663f25e986b8f4f4`. `KDBrain/models/model.py`, SHA256 `2e7f32693847e0794a9d0b2deb172dbcd285c686b74827eea31d6e30335a887d`, uses a text-conditioned query to influence relation attention. PSRR only adopts the principle that disease semantics can enter relation scoring. It does not use its brain-network modality, BioMedBERT, pathology consistency loss, anatomical labels, or additional graph.

Entmax-1.5 follows the sparse probability mapping described in Peters et al. and the official `deep-spin/entmax` implementation. The package is absent in this environment, so a local deterministic autograd implementation will be tested against the public threshold/squared-ReLU definition, exact-zero support, normalization, finite differences, and `gradcheck`. It is applied over 15 real within-patient candidate neighbors plus one fixed-zero-value Null candidate. It is not replaced by Softmax or sampled Top-K.

## Historical Base and Audit Findings

The only training base is E74.4-G2 Positive-DCR. The locked entrypoint is the existing E74.4 `prepare(fold)`, whose call chain invokes E74.1 and restores the matching E61 epoch-10 full warmup, all model/CAM parameters, three optimizer states, and public RNG. It verifies E74.1 initialization identity, original fold membership, the intentional Fold2 overlap at 1-based rows 377 and 407, and exactly 4096 permitted relation pairs. Every formal fold starts independently from that state. No E74.4 Best or E81 checkpoint is an initialization source.

The E74.4 historical objective remains `L_cls + L_graph + 0.05*L_pair + 0.01*L_adv + 0.01*L_preserve`, with its original sample scopes, formulas, training call order, three optimizers and Dropout. The only added term is `0.001*L_raw`. The original OutputLayer runs once. A forward pre-hook adds the PSRR residual to its CAM144 input. There is no second classifier and no extra stochastic operation. New module parameters are initialized inside an isolated CPU RNG scope with salt `817000+fold`.

E81.6 used learned semantic projections with pairwise messages. E81.6.1 added a free 1024-to-32 prompt mapping, which accounted for 32,800 of 42,336 trainable parameters and led to highly collinear prompt queries. E81.6.2 fixed an 8x8 SVD text geometry but retained a trainable 64-to-8 relation map. Its G1 and G2 medical/scrambled groups both scored 92.0742% ACC and 91.9230% Hard-AUC/BA, and their hard predictions were identical on every fold, so medical grouping increment was not established. E81.6.2-G3 scored 91.8443% ACC and 91.6411% Hard-AUC/BA. At its own best epoch, G1 residual-off to full gave net +2 corrections, while historical DARG Best to current full was net -1. E81.6.3 demonstrated controlled end-to-end gradient flow with eta=0.1 on the relation-input path, but scored 91.8443% ACC and 91.6202% Hard-AUC/BA against DARG 92.1891% and 92.0294%; its controlled added-gradient ratio averaged about 0.00324 and negative-cosine frequency was substantial. PSRR changes the actual relational construction and selection mechanism, not residual size or gradient clipping.

Fixed CLIP geometry is a constraint on parameterization, not validation that the prompts contain clinically valid knowledge. A fixed Gram matrix or a nonzero semantic score is insufficient evidence of medical validity. The full G1/G2/G3 comparisons and sparse support diagnostics address this limitation.

## Frozen PSRR Mechanism

The exact official eight text strings and order, including `Nromal`, use the existing frozen CLIP RN50 feature cache. The cached normalized `[8,1024]` vectors yield a deterministic CPU float64 thin SVD basis `E[8,8]`; the basis is frozen and preserves the original Gram matrix. The cache is copied from audited E81.6.2 and SHA-checked before use.

CAM144 reshapes in original coordinate order to sixteen 9-dimensional latent slots. A shared `Linear(9,32)`, learned positional tensor, and LayerNorm encode the slots. For each ordered receive/send pair without self-edges, the shared 64-dimensional key input is `[u_i*u_j, abs(u_j-u_i)]`; the shared 96-dimensional value input is `[u_j, u_i*u_j, u_j-u_i]`. A shared visual score and a shared bias-free `64->4->8` semantic map score each pair against all eight fixed prompt vectors. Per-patient prompt evidence is aggregated from relation score changes over real neighbors only. TD and ASD beta distributions are softmax-normalized separately within their fixed four-prompt groups, without labels.

The visual, ASD-conditioned, and TD-conditioned adjacency rows each apply Entmax-1.5 over 15 real neighbors and Null. Null has a learned key and fixed zero value. ASD and TD messages subtract the visual-baseline message; their difference is mapped by a shared zero-initialized bias-free `32->9` layer. The output is arranged as `[N,4,4,9]`, matched coordinatewise with the original contiguous 36-value global pooling, and converted to a smooth residual with `||delta|| <= 0.1||CAM144||`. Initial residual is exactly zero. Prompt-off and relation-off force the two conditioned supports to the visual support. Null-only makes all three supports Null-only so its zero value yields zero differential message. RES global-off uses uniform coordinate weights.

The fixed shared modules contain 7808 trainable scalars. Position vectors use the private-RNG `Normal(0,0.02)` initialization, the learned Null Key uses `Normal(0,1/sqrt(32))`, all Linear layers use PyTorch's recorded default initialization inside the same private CPU RNG scope, and the output projection is exactly zero. G1 uses the pairwise construction. G2 is independently trained and uses per-slot Prompt matching with no cross-slot candidate or message. It retains the same module parameter shapes and RES output channel. G3 has the same structure and same-fold initialization as G1, with groups `[0,1,4,5]` and `[2,3,6,7]`. It retains the original medical text, so it tests grouping rather than text presence.

## Training and Conditional Groups

All groups inherit seed 0, original ten folds, E61 epoch-10 start, E74.4 loss and optimizer states, public Python/NumPy/Torch CPU/CUDA RNG, 4096 relation pairs, original preprocessing/RFE, transductive graph, val=test checkpoint selection, maximum 450 epochs, and historical `wait>200` stopping. There is no per-epoch reseeding, additional Dropout, edge sampling, or Gumbel noise. Historical optimizers step in order `opt_base`, `opt_cam`, `opt_added`, followed by PSRR AdamW at `0.0005/0.01`.

G1 PSRR-Full-Medical is the only first-stage formal group and must complete ten folds. G2 PSRR-Unary-Medical runs only if G1's all-ten and fold-2-excluded ACC and Hard-AUC/BA each strictly exceed DARG and all implementation, RNG, checkpoint and end-to-end audits pass. G3 PSRR-Full-Scrambled runs only if G1 exceeds G2 on those four comparisons and the mechanism audit does not show relation collapse. Unrun conditional groups are recorded as `NOT_EXECUTED_CONDITIONAL`, never as negative evidence.

For the G1-to-G2 execution gate, the two primary metrics must also have positive mean paired difference in every leave-one-fold-out subset and more positive than negative per-fold differences. This was frozen before formal training to guard against a result driven by one fold. This is an execution criterion, not a significance test.

Fold2's original DARG supervision remains unchanged. During training only, the PSRR residual is exactly zero for zero-based rows 376 and 406 and these rows are excluded from the new regularizer. At evaluation every test patient uses the full model. The historical overlap is retained and remains a protocol limitation.

## Source SHA256

SHA values below identify the inspected local source at design freeze. Exact fold warmup and DARG Best checkpoint SHA values are recorded in `manifest.json`.

| Source | SHA256 |
|---|---|
|Original `main.py`|`dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2`|
|Original `network.py`|`484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec`|
|Original `layers.py`|`4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a`|
|Original `utils.py`|`95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67`|
|E74.4 `run.py`|`cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`|
|E74.4 `soft_positive_dcr.py`|`3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`|
|E74.4 `config.json`|`a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a`|
|E74.4 `audit.py`|`cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb`|
|E74.1 `run.py`|`a487c87d52a0faeded5ff6406ff9908290fc2fdfd9d5b66e6378a5b2aad6ab5c`|
|E74.1 `rudag.py`|`7fb5920b21f43a117a47137fcce58b5411b91c13dfaaf7df85fc502124723a8e`|
|E69.3 `run.py`|`be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44`|
|E61 `run.py`|`f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334`|
|E61 `model_e61.py`|`28266568bcebcfdbab961dbedb7257c15ff4e862cd02ae744feacad54b7bc81b`|
|E81.6 `prcl_res.py`|`4dc485491341f9293ad1c7de36d00db55900659901e04f1444f6e4c9645d8470`|
|E81.6.1 `cprl_res.py`|`bcda661ad0c74434fcdb46db2c1dd967c45ec28e1586ffbc5e5aa8837412c3be`|
|E81.6.2 `sa_prcl.py`|`c05fe067bca1e8c4be2ef587b7c7fcb9fcca29417f2861a85068e8c14f414beb`|
|E81.6.2 `design.md`|`fe988abf8887969c505592f5c63f4d724bb5a2652aa124059c632546604e1b11`|
|E81.6.3 `run.py`|`18637f96d089bf2645a1d74282860619078f0583e8e49ca999d631f54c78c36a`|
|E81.6.3 `audit.py`|`7f9b7c284e76c5c17657419bc99f29d2e22a56bc5f54f62d6a3331bbe4fd7d4d`|
|E81.6.2 fixed prompt cache|`7bfd1d6e11bb9983e9896e682b5a8c40bf606c919dfd38bae368436955f83aff`|

The approved loader maps historical `/model/spromptgl/experiment/...` paths to byte-identical local historical directories when those sources are absent from the mount. It verifies the locked source SHA and task identities before returning a fold. This mapping is local to the new experiment loader and does not change historical configuration or files.
