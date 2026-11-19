# Experiment 81.7.2 TP-PSRR frozen design

## Research question

TP-PSRR tests whether fixed disease Prompts can select local relations and drive competing ASD/TD sparse messages while preserving the inherited DARG training trajectory. CAM144 remains the inherited 144-wide representation reshaped as `[N,4,4,9]`. The 16 slots are latent feature slots and are not assigned anatomical brain-region labels.

## Source review and inherited protocol

The MICCAI 2025 SPromptGL paper, the official repository, the local `main.py`, `network.py`, `model.py`, `layers.py`, and `utils.py` were reviewed. GFR constructs and propagates the transductive patient graph. CAM produces CAM144. SPGS uses the frozen CLIP RN50 text features. RES performs the original local/global interaction. CAM144 enters the inherited `Outputlayer`, which the experiment intercepts through a local forward pre-hook and then invokes once.

E74.4-G2 is the only base. Each fold is restored from the matching E61 epoch-10 warmup, including GFR/CAM/DARG parameters, all three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG, data membership, 4096 training relation pairs, and optimizer order. The test-as-validation ACC checkpoint rule, pre-CV preprocessing/RFE, transductive graph, and Fold2 overlap are retained. Zero-based rows 376 and 406 are excluded only from the new PSRR residual and raw-residual regularizer on Fold2.

E81.7.1 demonstrated that local receiver beta varied, but its distributions stayed close to uniform and uniform-beta did not change hard predictions. Its direct VJP decomposition used the enhanced graph as the direct reference. TP-PSRR changes the training protection to a same-state original DARG reference gradient and adds an edge-conditioned Prompt group.

## Frozen architecture

The original eight Prompt strings, order, CLIP RN50 cache, normalized Gram, and SVD source are immutable. `E_center[k] = normalize(E[k] - mean(E, dim=0))` is fixed and non-trainable. It is a centered semantic direction and does not preserve the original Prompt Gram.

The shared local encoder is `Linear(9,32) + position[16,32] + LayerNorm`. Pair keys are `[u_i*u_j, abs(u_j-u_i)]`, pair values are `[u_j, u_i*u_j, u_j-u_i]`, and the shared semantic map is bias-free `64 -> 4 -> 8`. Every receiver has 15 real non-self neighbors and one Null candidate. Entmax-1.5 is applied over the 16 candidates. The ASD/TD relation difference uses `lambda_rel=1.0`, and the message is `sum_j((A_ASD-A_TD)*V_ij)`. A zero-initialized `32 -> 9` projection feeds the original local/global interaction and a bounded residual with `rho=0.1`. The added parameter set remains 7,808 parameters.

G1 is node-conditioned. For each patient and receiver, visual-plus-semantic edge evidence is aggregated over the 15 real candidates, then TD and ASD each apply a temperature-1.0 softmax over their four Prompts. No average over the 16 receiver slots is performed.

G2 is edge-conditioned. For each patient, receiver, real candidate edge, and Prompt, beta is the temperature-1.0 softmax over that edge's four Prompt semantic scores. The resulting TD/ASD semantic edge scores form `D` before the same symmetric Entmax competition. The Null candidate receives no semantic score and remains visual-only. G1 and G2 use the same Prompt grouping, parameters, initialization salt rule, optimizer, loss, and data protocol.

## DARG-reference gradient protection

For each training step, the runner first evaluates the original DARG loss `L_B` with the PSRR residual disabled. It records `g_B` for the GFR, CAM, and DARG optimizer parameter groups. It restores all RNG components and model/CAM buffers, then evaluates the enhanced loss `L_F = L_DARG(enhanced) + 0.001 L_raw` with the same stochastic masks. It records `g_F` and all PSRR gradients. The shared delta is `delta_g = g_F - g_B`.

For each complete parameter group, a negative dot product between `g_B` and `delta_g` removes the parallel conflict component. A single group scalar then enforces `||delta_g_controlled|| <= 0.05 ||g_B||`. If the reference norm is zero, the controlled delta is zero and no gradient is injected. The final shared gradient is `g_B + delta_g_controlled`. The PSRR parameters receive their unmodified enhanced-loss gradient. Assignment happens before the historical optimizer steps and the independent PSRR AdamW step.

The reference forward is a diagnostic computation and does not consume public RNG. It is followed by the enhanced forward from the exact pre-reference RNG state, so the enhanced forward consumes one historical public train stream. Model and CAM buffers are restored between forwards. The preflight checks zero-residual output/loss/gradient parity, RNG restoration, reference-gradient replay, projection reconstruction, group kappa, and nonzero end-to-end PSRR-to-CAM/DARG VJP paths.

## Formal groups and decision

Only G1 TP-PSRR-Node and G2 TP-PSRR-Edge are trained, each for all ten folds and each independently from E61 epoch-10. No anonymous, scrambled, unary, frozen-backbone, or extra group is authorized. The performance gate requires a group to exceed DARG in ACC and Hard-AUC both across all ten folds and after excluding Fold2. Mechanism evidence cannot replace that gate. A single seed, test-as-validation selection, transductive graph, pre-CV processing, and Fold2 overlap limit generalization.
