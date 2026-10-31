# Experiment 56.2 — DCRAG-GFR-R2 design freeze

## Scope and audit

The sole formal group is `E56_2_FULL_DCRAG_GFR_R2`, seed 0, the E51 `E1_GFR_AGGREGATION_FIX` fold anchors, and the historical ten-fold transductive ABIDE-A protocol. The four inputs remain PHENO 48, ANAT 6, FUNC 10 and CORR 256 (320 dimensions). E51's PEN-free `PLNNoPEN` CAM, four GraphLearn modules, four complete 320-dimensional GCN branches, split files, class weights, 450 epoch limit, strict ACC checkpoint selection, `wait>200`, `val=test`, global preprocessing/RFE and 871-node graph are retained. The last three are historical comparability choices, not unbiased independent evaluation.

The SPromptGL paper's GFR is mapped to the actual local implementation in `source/network.py` and `source/dugsa.py`: each GraphLearn is a modality projection/cosine graph and each GCN receives the concatenated 320-D patient feature. The audited E51 E1 path accumulates all four messages. The paper's graph-loss wording is not substituted for the source `GraphConstructLoss`; the source loss and its actual call remain in `run.py`. The original ALoss is not silently treated as an optimization term. The CAM receives one final 320-D GFR representation.

The original GitHub implementation and local E51 source snapshots are recorded in `manifest.json`; root model/data files are read-only. The historical E56 failure was message scale amplification and a disconnected classification-to-GraphLearn path. E56.1 repaired scale and dynamic edges but had zero-initialized Gate (patient standard deviation exactly zero), an entropy sign error (confidence saturated at one), eta initialized at 0.03 rather than 0.10, forced equal message norms, an equal-weight prototype, and wrong-neighbor changes applied to the candidate pool rather than selected propagation edges. Its source/report are read-only historical evidence, not initialization.

## R2 fixes frozen before training

1. **Patient Gate.** The first Gate linear uses Xavier initialization and the second uses independent `N(0,1e-3)` initialization; both biases are zero. Initialization is inside `torch.random.fork_rng`. Its input contains bounded raw 320-D features, normalized four message directions/intensities, receiver-specific mean confidence computed with `index_add`, and fixed train-only reliability. Initial softmax weights are near uniform but not forced identical. A test requires finite nonnegative weights, sum one, patient variance, and nonzero gradients/updates in both linear layers.
2. **Confidence.** `entropy_norm=-sum(p log p)/log(3)` and `confidence=clamp(1-entropy_norm,.05,1)`. The value is computed per selected receiver/sender pair and enters both edge weights and the Gate summary; it is not a global mean. Uniform, one-hot, monotonicity, range and gradient tests are fixed.
3. **AT eta.** `eta=0.30*sigmoid(eta_raw)` and `eta_raw=logit(0.10/0.30)`, so every modality starts at exactly 0.10 and remains in [0, .30].
4. **SoftCap.** For radius `R`, `SoftCap(v,R)=v/(1+||v||/(R+eps))`. Same messages use `R=.5||X_i||`, AT difference messages `.15||X_i||`, and the final weighted message `.25||X_i||`. Small messages are not amplified, large messages are continuously compressed, directions and gradients are retained, and `H=X+sigmoid(beta) M_bound` is the integrated GFR output. No old GFR output is used as an external residual.
5. **Reliability-weighted prototype.** For AA/TT/AT, normalized shared prototypes are combined with detached, train-only OOF relation reliabilities. The target is `stopgrad(sum(r_m p_m)/sum(r_m))`; each modality is compared to that target with class balancing. Weak modalities therefore do not have the same authority as CORR, while all four still have independent relation, edge and propagation paths.
6. **Selected-edge wrong graph.** The formal forward stores actual Top-16 receiver/sender edges. The diagnostic uses deterministic directed two-edge swaps on those selected edges only, recomputes priors, current GraphLearn scores, relation probabilities and weights for the new patient pairs, and checks no self-loops, no duplicates, receiver outdegree, sender indegree and sender site/outer-split composition. Coverage below 30% total or 20% in any modality is reported `INSUFFICIENT_MATCHED_CONTROL`; constraints are never relaxed. `compute_edge_prior` is a pure pair-dependent function.

## Fixed losses and optimization

`L=L_cls+L_graph+.02 L_relation+.01 L_adversarial+.01 L_prototype+.002 L_consistency+.001 L_edge_stability`; GRL maximum .10 with 20-epoch warm-up. GraphLearn Adam is lr .0038/WD .11, inherited GCN Adam is .0038/WD 1e-4, CAM Adam is .0038/WD .11, and new modules AdamW are .001/WD 1e-4. No test-driven changes are permitted.

## Required preflight evidence

The preflight records E51 bitwise bypass/CAM/RNG checks; Gate two-layer initialization, patient variance, gradient and update tests; confidence math; eta; SoftCap bounds/gradients; reliability prototype contribution; shared/private GRL sign and gradient isolation; four GraphLearn/GCN/edge classification gradients; no outer-test-label leakage; graph orientation and one-pass row normalization; self-only `GCN(X,I)`; selected-edge swap invariants; strict prior recomputation; four-modality participation; and checkpoint resume equivalence. Formal training is refused unless these checks pass.

## Scientific boundaries

The experiment can be engineering-complete while the method is scientifically `NO-GO` or `PARTIAL`. A nonzero gradient, a low source-probe AUC, or a self-only drop alone does not establish alignment or a patient-relation gain. Results are ten paired folds from one seed and one ABIDE-A cohort under the historical validation/test and transductive protocol.

## External paper/repository audit record

The MICCAI open-access record (https://papers.miccai.org/miccai-2025/0861-Paper5129.html) describes an interactively multi-relation graph dynamically learned with graph-learning losses followed by multi-layer graph convolution and a semantic prompt/CAM pathway. Its reviewer discussion explicitly notes that the paper concatenates modality features before DGCN propagation and uses label-guided graph structure regularization. The original repository (https://github.com/wanxixi11/SPromptGL) and the local read-only source snapshots were checked against this description. R2 retains the audited E1 code behavior and changes only the registered DCRAG-GFR defect paths; it does not reintroduce prompt/PEN components.
