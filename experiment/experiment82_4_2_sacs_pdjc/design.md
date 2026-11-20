# Experiment 82.4.2

## Research Question

This experiment tests whether medical prompts can control a real, signed CAM144 feature correction that combines the fixed Original O disease spectrum with classifier-sensitive directions. It has two independently trained groups, SACS-PSCE and PDJC-DA-SRPE. Each group has ten folds, seed zero, and one independently restored E61 epoch-10 E74.4-G2 initialization per fold.

## Source and History Review

The MICCAI 2025 paper describes a multi-relation GFR, graph convolution, CAM interaction and a semantic prompt-guided local feature path. In the server training path, E74.4-G2 calls E74.4 Positive-DCR through the audited history adapter, then the vendored `PLNNoPEN` CAM. The original head receives a flattened 144-dimensional CAM and applies LayerNorm(144), Linear(144,36), GELU, Dropout(0.35), Linear(36,2), then log-softmax. This experiment injects its feature residual at the existing OutputLayer input hook. It does not replace the classifier or use a test label to construct a patient correction.

The official author repository, user repository, paper PDF, local `main.py`, `model.py`, `network.py`, `layers.py`, and `utils.py` are read-only references. The local root sources are not part of the implementation. The implementation uses the same E74.4-G2 historical GFR/DARG path, original three optimizers, loss, update order, graph, patient membership and six training Dropout calls.

E82.2 established Original O as eight PCA axes plus eight disease-covariance axes, fit from legal E61 epoch-10 training CAM rows. E82.2-G2 ended below DARG in ACC and BA. Its counterfactual target supervised prompt allocation rather than utility of the actual classifier feature correction. E82.3.2-G2's fixed-weight residual changed one hard prediction, while end-to-end ACC remained below DARG. Its prompt replacement caused no hard flip. E82.4 tested several fixed-spectral prompt transformations. G1 and G3 had small direct net correction but the jointly adapted trunks lost more performance. G2 and G4 had no residual-induced hard flips. E82.4.1-G1 had zero same-weight hard correction, while G2 had one ASD correction and no direct harm. Both final groups remained below DARG in ACC and BA. The E82.4.1 Query replacement changed probabilities but G2 had no hard flip. Attention entropy remained about 0.99. These observations motivate a complementary classifier-sensitive space, stronger but bounded cosine-attention scale, and direct Margin supervision of the actual CAM144 delta.

## Frozen Shared Design

Original O remains fixed and byte-identified per fold. For CAM row `C_i`, its centered 16 coordinates are `zU_i = (C_i - mu) U`, with `U` shaped 144 by 16. Its 16 tokens, eight archived medical RN50 Query vectors, a learned Null token and Query attention remain active. Query/key cosine similarity is divided by a trainable temperature constrained to `[0.20, 0.60]`, initialized to `0.35`.

The original OutputLayer remains the only classifier. Its full classification loss still backpropagates through GFR, DARG, CAM, OutputLayer and the new prompt correction. The new branch receives the audited 0.25 gradient multiplier only on the CAM input path. The direct classifier gradient is unscaled. Auxiliary Margin and alignment objectives update only new branch parameters and obey fixed gradient caps. Each residual is a continuous signed CAM144 vector with a smooth per-patient norm bound strictly below 8% of the uncorrected CAM norm. The three historical optimizers update in their historical order. The new AdamW optimizer updates last.

The unchanged historical objective is `cls + graph + 0.05 pair + 0.01 adv + 0.01 preserve`. No additional Dropout, random decomposition, per-epoch reseeding, test-label feature, or auxiliary training group is introduced. New parameter initialization runs in a forked RNG stream. CPU float64 eigensolvers and Jacobian calculations do not consume the shared training RNG.

## G1 SACS-PSCE

G1 keeps the E82.2 `U16` fixed. On legal training rows only, it computes the deterministic no-Dropout original-head Jacobian of `logit_ASD - logit_TD` at CAM144. It projects each row outside U and forms a 144 by 144 float64 second-moment matrix. CPU float64 eigendecomposition selects eight classifier-sensitive directions. Near-degenerate eigenspaces use their spectral projector and ordered coordinate Gram-Schmidt to obtain a deterministic basis. Signs are canonicalized by the largest-magnitude coordinate. The final V8 is reprojected against U and reorthogonalized.

V8 is initialized from E61 epoch-10 legal training CAM rows. After every tenth completed training epoch, a candidate is recomputed from the current legal training CAM and current no-Dropout classifier Jacobian. Orthogonal Procrustes alignment, rate 0.1 interpolation, projection against U and deterministic reorthogonalization update V. This operation uses no random numbers. The current V, eigenvalues, update count and next scheduled epoch are checkpoint state.

Eight additional tokens remain distinct from the 16 Original O tokens. Both token sets interact with the same eight medical Query vectors and Null token. Separate signed controllers produce `delta_zU` and `delta_zV`; a patient-dependent sigmoid beta fuses `U delta_zU` and `V delta_zV`. Both component residuals and their full combination use the shared smooth 8% bound. Fold 2 overlap rows 377 and 407 retain the historical DARG path and receive exactly zero new residual. Neither row enters V construction or auxiliary supervision.

## G2 PDJC-DA-SRPE

G2 keeps U16 fixed and computes one batched VJP through a deterministic no-Dropout copy of the current original head. For each CAM row it obtains `g = d(logit_ASD - logit_TD)/dC`. It decomposes `g_parallel = (g U) U^T` and `g_perp = g - g_parallel`. The components are independently normalized with a fixed numerical floor. Reconstruction and orthogonality are checked at preflight and audit.

Five-fold stratified inner groups use only legal outer-training rows. Fold-local ASD and TD means, per-coordinate variances, continuous class effects and reliability follow the audited E82.3.2/E82.4.1 cross-fitted EMA method. Fold 2 overlap rows are excluded from the EMA reference, update and labeled auxiliary objective. The state, counts, fixed inner-fold membership, EMA update counter and next state are serialized.

The eight medical Query responses, U16 features, cross-fitted class evidence, continuous reliability, and label-free classifier uncertainty feed two signed scalar actions and two patient-dependent sigmoid strengths. The resulting CAM144 residual is the weighted sum of the normalized spectral-parallel and spectral-perpendicular Jacobians. Both components remain in the graph and are separately audited. No hard predicted-class rule determines the action sign.

## Training and Selection

Every formal task restores its own E61 epoch-10 state, including original model and CAM parameters, all three historical optimizer states, relationship pairs, patient membership and Python, NumPy, Torch CPU and CUDA RNG. Training resumes at epoch 11, runs at most 450 epochs and stops only after wait exceeds 200. The first strict test-fold ACC maximum is retained. Latest checkpoints commit every ten epochs and at Best updates or termination. The G1 wave completes before G2 begins. At most ten workers use physical GPU0.

G1 auxiliary objectives use legal training patients' true-class Margin gain from `C` to `C + delta`, a continuous protection penalty for already-correct high-Margin patients, and direction alignment with the current classifier gradient. G2 uses the same class-balanced actual-residual Margin and protection objectives. Auxiliary gradients are added only to new parameters under the frozen E82.4.1 coefficient and per-family cap schedule. Test labels are absent from all train-time correction, V construction, EMA, and auxiliary updates.

## Audit and Decision Rules

Audits distinguish DARG to current Residual-off, which measures shared trunk adaptation, from current Residual-off to Full, which measures direct fixed-weight residual utility. All fixed-weight interventions use the same Best parameters. G1 checks Full, Residual-off, Query-replaced, Semantic-off and Complement-off. G2 checks Full, Residual-off, Query-replaced, Semantic-off, Spectral-only, Complement-only and Reliability-off. Each intervention recomputes CAM144, logits, probabilities, predictions and class-specific corrections and harms.

The report includes ACC, hard-prediction AUC/BA, probability AUC, ASD sensitivity, TD specificity and ASD-F1, along with per-fold paired deltas, correction and harm counts, and the same comparisons excluding Fold 2. Performance GO and mechanism GO follow the user-registered criteria exactly and are judged separately. Medical Query probability and hard-decision effects are described as model response, not evidence of independent diagnostic value.
