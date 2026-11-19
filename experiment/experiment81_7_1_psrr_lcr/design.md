# Experiment 81.7.1

PSRR-LCR investigates prompt-guided sparse relational reasoning with locally conditioned disease-contrastive relations. It tests whether fixed disease prompts can guide local relation construction, candidate selection, and message interaction, exposing disease-discriminative evidence that single local semantic matching cannot represent.

## Source Review

The MICCAI 2025 SPromptGL paper and the official author repository were reviewed alongside local `main.py`, `network.py`, `model.py`, `layers.py`, and `utils.py`. SPromptGL's GFR learns a transductive patient graph and performs graph propagation. CAM is the modality-aware feature encoder. The paper's SPGS forms label-independent class cues from eight frozen CLIP RN50 text descriptions, scores local and pooled modality features against those cues, and weights local slots. RES forms local/global feature interactions before the original output classifier. The local CAM tensor used by this experiment is 144-dimensional and is arranged as `[N,4,4,9]`, with 16 latent feature slots. The slots are not verified anatomical ROIs.

The original SPromptGL classifier consumes the flattened CAM144 representation at `Outputlayer`. E81.7 used an `Outputlayer.register_forward_pre_hook` to insert a bounded residual while preserving the original classifier call. This experiment retains that entry point and the original classifier.

E74.4-G2 is the only DARG base. Its source uses the per-fold E61 epoch-10 complete warmup, all three inherited optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG, the original data indices and 4096 training relation pairs. The original total loss is `L_cls + L_graph + 0.05*L_pair + 0.01*L_adv + 0.01*L_preserve`. The three original optimizers update in their historical order. The protocol starts at epoch 11, stops after wait exceeds 200 or at epoch 450, and selects Best on strict test-fold ACC improvement. The historical test-equals-validation rule, preprocessing/RFE before outer CV, transductive patient graph, and Fold2 overlap rows 377 and 407 are inherited. Only the new PSRR-LCR contribution is blocked on the two Fold2 overlap rows.

E81.7 used the same eight cached prompt strings and fixed SVD semantic basis, but averaged local prompt evidence across all 16 positions before computing a patient-level prompt distribution. It obtained near-uniform patient-level weights and weak ASD/TD support differences. Its same-epoch PSRR residual added four correct test predictions, while the full selected checkpoint relative to DARG lost three net predictions because the historical trunk and Best epoch had changed. Its audit also established that no scrambled-prompt group was trained. The E81.7 source audit documented a post-training audit-only correction, leaving its training bundle SHA unchanged.

## Frozen Method

The original eight prompt strings, order, CLIP RN50 features, cache identity, and SVD basis `E` remain fixed. The preflight verifies `E @ E.T` against the original normalized prompt Gram. A second non-trainable basis is computed once as `E_center[k] = normalize(E[k] - mean(E))`. This centered basis is used for relation scores. Its Gram is not asserted to preserve the original prompt geometry.

CAM144 remains `[N,4,4,9]`. The unchanged 4-by-36 AvgPool1d operation yields global `[N,4,9]` information. A shared `Linear(9,32)`, position embedding, and LayerNorm encode the 16 latent slots. Each receiver compares its 15 other slots and one Null candidate. The shared relation key is `[u_i*u_j, abs(u_j-u_i)]`; the shared value is `[u_j, u_i*u_j, u_j-u_i]`. A shared bias-free 64-to-4-to-8 projection scores the fixed centered prompts.

For each patient, receiver slot, and prompt, evidence is `logsumexp_j(S_visual + S_semantic) - logsumexp_j(S_visual)`. The four TD prompts and four ASD prompts are independently softmaxed over the prompt axis at fixed temperature 1.0. No receiver-axis averaging, label input, random selection, or entropy penalty is used.

The locally weighted semantic scores define `D = S_ASD_sem - S_TD_sem`. Both disease relations use the same visual score and Null score. `A_ASD = Entmax1.5(S_visual + 0.5*D)` and `A_TD = Entmax1.5(S_visual - 0.5*D)` use the fixed relation scale 1.0. The contrast message is directly `sum_j((A_ASD-A_TD)*V_ij)`. The original shared value map, RES local/global interaction, zero-initialized bias-free 32-to-9 output projection, rho 0.1 smooth residual cap, 144-dimensional CAM residual, original classifier, and 0.001 raw residual penalty are retained. The added module has 7808 trainable parameters.

## Direction-Aware Shared Gradients

The training step executes one enhanced forward and obtains the full shared-parameter gradient from the total loss. The branch gradient is computed by autograd VJP through the actual residual from every DARG loss component, plus the shared-parameter VJP of the raw residual regularizer. The direct component is the full gradient minus that branch VJP on the same forward graph. Preflight checks their reconstruction and verifies that the residual path reaches CAM and DARG parameters.

For each GFR, CAM, and DARG optimizer group, a negative direct/branch dot product triggers orthogonal projection of the branch component away from the direct gradient. A single coefficient per whole group then enforces `||g_branch_projected|| <= 0.1*||g_direct||`. No artificial gradient is injected. All original and PSRR optimizer updates occur after this control is applied.

## Formal Groups

G1 uses TD prompts `[0,1,2,3]` and ASD prompts `[4,5,6,7]`. G2 has the identical model, initialization, data, optimizers, loss, training, RNG handling, and gradient control, but groups TD `[0,1,4,5]` and ASD `[2,3,6,7]`. Both are independently trained for all ten folds from the fold-specific E61 warmup. The group-specific initialization salt is `817100 + fold`, inside an isolated CPU RNG scope. G2 runs regardless of G1's metric result.

Prompt-off, relation-off, uniform-beta, and global-interaction-off are deterministic interventions on each Best checkpoint and do not retrain the model. Their interpretation is limited to the pathways each intervention changes. Medical prompt value is assessed by independent G1/G2 training comparisons.

## Execution and Decision

Only physical GPU1 is used via `CUDA_VISIBLE_DEVICES=1`; the runner verifies the device UUID and obtains free memory for physical index 1. One worker is measured first, then concurrency is limited by measured peak memory and the GPU1 safety reserve, up to ten workers.

Engineering validity requires historical initialization, data membership, RNG, end-to-end gradient, Best and Latest resume, metrics, Fold2, and device audits. G1 must exceed DARG on all-fold and Fold2-excluded ACC and Hard-AUC/BA. Medical semantics additionally require interpretable positive paired gains over independently trained G2. The relation mechanism must connect prompt scores to sparse support, messages, and classifier output. A single seed and this inherited validation protocol do not establish final clinical generalization.
