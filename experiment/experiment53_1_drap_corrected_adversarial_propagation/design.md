# Experiment 53.1 — Design Freeze and Audit

## Paths and protection

- Requested root `/data/adodas/SPromptGL-main/experiment` returned OSSFS `Input/output error` on an actual directory-creation probe. The verified writable overlay path is `/model/spromptgl/experiment/experiment53_1_drap_corrected_adversarial_propagation`; the directory is created there and this fallback is recorded in the manifest.
- `/data/adodas/SPromptGL-main` and `/model/spromptgl` resolve to different mount roots (`fuseblk` and `overlayfs`). ABIDE data and Experiment 43.1 anchors are read-only inputs from the data mount. Experiment 52/53 implementations, relation-cache inputs, and Experiment 53 historical checkpoints are read-only inputs from `/model/spromptgl`.
- Only this directory receives Experiment 53.1 code, state, checkpoints, and results. Root model code, data, Experiments 51–53, and their checkpoints/results remain protected.

## Paper and source review

- SPromptGL MICCAI 2025, Section 2.1, Eqs. (1)–(2): four modality-specific GraphLearn relations; cosine similarities; parameterized graph convolution; GFR fusion. The public code uses modality projection and thresholded weighted cosine graphs, `GraphConv` computes `A_norm @ (X @ W)`, and the original GCN has two layers with ReLU after layer one. `normalize_adj(A + I)` is symmetric degree normalization. The public `simple-2` path trains classification plus `GraphConstructLoss`; paper Eq. (1)'s label-similarity term is not in that path's optimized loss. The original GFR's `i == 1` assignment overwrites the previous PHENO message; the Experiment 51 E1 `LegacyFixedGFR` fixes this by summing all four full-320D messages. PEN is physically absent and CAM is retained.
- DeepASD (Translational Psychiatry, 2024) first learns modality-specific common-space features with a feature extractor and adversarial modality discriminator, then learns modality patient-similarity graphs, fuses them, and classifies with S²GC. Experiment 53.1 keeps the SPromptGL GFR/CAM base and places the discriminator on symmetric patient-pair relation representations. A disease-reliable modality is distinct from an adversarial source modality.
- Official references: [SPromptGL paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [official SPromptGL repository](https://github.com/wanxixi11/SPromptGL), [DeepASD paper](https://www.nature.com/articles/s41398-024-02972-2).

## Historical evidence reviewed

- Experiment 47.1: with `weight_decay=0.11`, learned GCN W1/W2 collapsed to roughly 0.15–0.40% of their initial norms; adjacency normalization stayed near 0.93–0.99. Restoring initial W recovered message magnitude, while restoring only the initial graph did not. Removing decay restored amplitude but damaged classification; amplitude alone is not disease-relation value.
- Experiment 47.2 and 47.3: active graph propagation can affect predictions, but a true patient graph did not reliably beat structurally matched rewired or message-identity controls. Experiment 47.3 reports D4 rewired ACC above D1 true-graph ACC (91.3858% vs 90.6962%).
- Experiment 48.1: conditional adversarial alignment was attempted, but condition/source diagnostics often degenerated and the full candidate did not establish a disease increment.
- Experiments 51.2/51.3: relation selection/propagation had no stable relation-specific benefit over E1 or controls. Experiment 52 showed relation scores near 0.5 and near-zero original GCN messages at high decay; its follow-up confirmed further GFR decay further shrank messages.
- Experiment 53: OOF relation evidence was present only for CORR (mean OOF AUC 0.7257; PHENO 0.4984, ANAT 0.5031, FUNC 0.4994). Its E1–E5 adversarial path nevertheless gated on the reliable set and therefore disabled the discriminator. Its `_propagate` did not disable negative messages for E1/E3; its training loop computed the warmup flag but did not pass it into forward; E7 rewired endpoints; E8 had about five times E4's edge budget. Experiment 53's fixed-intervention E4–E7 and E4–E8 ACC differences were both zero. Historical E0 and E1–E8 results remain read-only references.
- Experiment 47.1/47.2 source directories are not present in the data-mount experiment tree; their summarized audited reports were reviewed in `/model/spromptgl/experiment/handoff.md`. No missing local source is represented as directly re-read code.

## Frozen corrections

F1. Store explicit per-group positive/negative/uncertainty/identity/adversarial modes and use them in the propagation call. E1/E3 negative message must be identically zero; E2/E4/E5/E7/E8 must calculate it; E6 uses continuous mutually exclusive positive/negative weights.

F2. Pass `propagation_enabled=(epoch >= 10)` through the complete forward path. Epochs 0–9 optimize pair and configured adversarial losses with exactly zero DRAP message. E1–E8 can enter the formal best-checkpoint comparison only from epoch 10. A separate warmup best metric is diagnostic only.

F3/F4. Preserve the OOF reliable set and omega for disease-edge fusion (currently CORR only). For adversarial training, feed all four modality relation representations for the same balanced outer-train pairs. Use fixed stopped-gradient CORR confidence weights, normalized within same/different strata. E3/E4 use the true same-diagnosis condition; E5 uses an all-zero condition with the same 2-wide input and same discriminator. Test labels never enter relation, adversarial, confidence, or propagation inputs.

F5. Replace CUDA `index_add_` aggregation in the new DRAP path with a differentiable dense weighted adjacency and matrix aggregation; enable deterministic Torch algorithms for new runs and verify two independently built equivalent E4 instances through two optimizer updates, outputs, parameters, optimizer state, and epoch-end RNG. The first 80-fold pass exposed that the deterministic setting was declared but not called by the worker entry point; independent reads showed six metric discrepancies, including rank-sensitive Probability-AUC changes and one hard-threshold prediction change. That pass and its E7 predecessor are archived under `.runtime/superseded/`; the final worker, preflight, and audit entry points now call the same deterministic setup, and all eight groups are rerun. Historical E0 is read only; deterministic settings apply equally to E1–E8, without claiming its old training trajectory was repeated.

F6. E7 permutes calibrated q values among fixed CORR candidate-edge slots on every forward, keeps the original endpoints, and reselects edges under E4's fold-local per-node sign budgets, capped at the initial matched sign edge count. E8 selects within the same GraphLearn candidate support using original projected cosine similarity and fold-fixed per-node sign budgets derived from the pretraining CORR score graph; it never reads q in its forward propagation. Both controls report edge count, degree and message-scale matching. Any residual mismatch is labelled `CONTROL_MATCHING_LIMITED`.

An independent post-run code audit found that the first E7 implementation preselected shuffled edges from the cached initial relation scores and reused those edge identities during training. Although this preserved a shuffled initialization, it did not implement per-forward score permutation/reselection. Those first E7 checkpoints are archived under `.runtime/superseded/` and are not used for the final comparison. The corrected E7 branch is preflighted and all ten E7 folds are retrained with the other seven groups' checkpoints preserved.

Propagation keeps the Experiment 53 operator and coefficients: positive/negative separately row-normalized weighted differences over raw 320D features; `Delta = M_pos - 0.25*M_neg`; per-block RMS-tanh bound; `lambda_prop=0.02`. Fixed checks quantify single-neighbor weight cancellation, finite-difference q sensitivity, classifier-to-q gradients and calibration migration. No coefficient or threshold search is allowed.

## Training protocol

E1–E8: seed 0; 10-fold `StratifiedKFold(shuffle=True, random_state=10)`; 871 nodes; dimensions 48/6/10/256; initialization seed `510000 + fold*100 + 1`; same Experiment 43.1 untrained anchors; original GFR/CAM Adam at lr 0.0038, weight decay 0.11; DRAP AdamW lr 0.001, matrix decay 1e-4, bias/LayerNorm decay 0; pair/adv weights 0.05/0.01; GRL 0.03; max 450 epochs; patience `wait > 200`; strict ACC best; historical `val=test`; max five independent GPU0 workers. There is no E0 training in this experiment. E0 is independently read from Experiment 53 historical checkpoints and checked against its saved metrics.

## Preflight gates

Before formal training: historical E0 readback 10/10; common GFR/CAM, post-construction RNG, relation initialization, fold and pair hashes equal across groups; explicit sign branch checks; ten-epoch warmup zero-message checks; four-modality conditional/unconditional adversarial input checks and GRL/discriminator gradients; held-out label permutation invariance; CORR-only reliability isolation; E7 identity permutation and E8 q-free structural selector; E6 positive/negative non-overlap; q finite-difference/gradient checks; deterministic equivalent E4 2-update parity; checkpoint forward mode; finite losses/gradients. A failed hard gate blocks the 80-fold run.

## Evaluation boundary

All historical folds retain `val=test`, CV-before preprocessing/RFE and the 871-node transductive graph. These results can be optimistic and are not unbiased external generalization estimates. Ten folds share the same cohort and are not ten independent clinical cohorts. Only one seed is used.
