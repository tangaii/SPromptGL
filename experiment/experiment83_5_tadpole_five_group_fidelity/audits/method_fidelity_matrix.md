# Experiment 83.5 Method-Fidelity Criteria

## Source checks

- The MICCAI 2025 SPromptGL paper, the author repository and project commit `5ebc4f5` were checked against the frozen E83.4 source copies.
- The author implementation uses eight prompt rows and a two-class semantic competition. G1 preserves that eight-row branch and adds only a 12-row three-class PEN route for NC, MCI and AD.
- E43.4 E0 is an anchored, physically instantiated CAM-only experiment on ABIDE. Its checkpoint and reported score are not treated as TADPOLE weights or results.
- E43.2 is an initialization-method source only. It supplies no assumption that excluded checkpoints exist on GitHub.

## G1

- Historical eight-query forward parity and its gradient path must pass.
- The 12-query PEN must use the frozen RN50 features, the original 4096-128-hidden MLP, three-way semantic competition and original local/global fusion.
- Every four-prompt class group must have a nonzero gradient in the probe. The PEN forward signature accepts only token features, so test labels cannot route semantic attention.

## G2 and G3

- E83.4 Best and Latest checkpoints are read-only inputs. Each fold must independently restore Best and Latest, verify output and optimizer/RNG state, and deterministically replay the next step.
- G2 must have no PEN state or optimizer parameter. G3 must preserve the registered Positive-DCR source behavior and pass Direct-Off and Disease-Neutral interventions.

## G4

- The query is L2-normalized before both spectral attention and pair context. A frozen E83.4 G4 checkpoint is used only for a numerical before/after diagnostic.
- The 12 frozen RN50 features are normalized in float64. Their Gram matrix must be symmetric, positive definite and rank 12. Deterministic Cholesky rows are padded to width 32 and reproduce the original RN50 Gram matrix within the stated tolerance.
- All three class pairs remain active with the E83.4 weights, losses, optimizer, residual cap and shared gradient budget. The differential-query branch is disabled in the registered source configuration.
- Each fold begins from the matched G3 ten-epoch warmup state. E83.4 G4 final weights are not an initialization source.

## G5

- Each fold is anchored to a fresh, untrained TADPOLE CAM constructor state, initial CAM-only Adam state and post-construction RNG. No trained checkpoint initializes G5.
- The formal model contains only HistoricalCAM. GFR, graph learning, GCN, message passing, PEN, SPGS, CLIP, prompt embedding and graph loss are absent from its module tree, parameters, state and optimizer.
- A functional direct-CAM reference and the registered physical model must match logits, loss, every CAM gradient, updated parameter, Adam state and RNG for two steps from the same anchor.

## Shared protocol and release gate

- The frozen 598-patient TADPOLE matrix, 365 input features, six modality widths, 240-dimensional CAM and ten provided stratified folds must match their registered SHA-256 values.
- Historical whole-dataset RFE, pre-outer-CV scaling, mean imputation, transductive patient graphs for G1-G4, `val_idx == test_idx`, first strict test-fold ACC improvement, original losses, original optimizer behavior and early stopping remain fixed.
- G5 does not build a patient graph. No per-epoch seed reset, test-driven selection or threshold search is permitted.
- The formal release starts only when all preflight checks, the throughput probe and all five group configuration hashes pass. The start preflight and seal are immutable during training.
