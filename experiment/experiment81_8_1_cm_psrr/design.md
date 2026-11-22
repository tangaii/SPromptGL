# Experiment 81.8.1 - CM-PSRR

This directory is the authoritative implementation of the single formal
`G1` group, Classification-Margin Guided Prompt Sparse Relational Reasoning,
with seed 0 and the historical ten folds.

## Frozen history

The training base is E74.4-G2 Positive-DCR/DARG. Each fold is reconstructed by
the read-only historical `prepare(fold)` entry point from its E61 epoch-10
warmup state. The base contains the active `PLNNoPEN` CAM and the original
three optimizers, buffers, relation pairs, split membership, RNG states and
checkpoint selection protocol. E81.7.2-G1 supplies the node-conditioned
TP-PSRR relation path and RES-style bounded CAM144 residual. The eight cached
OpenAI CLIP RN50 text features remain frozen and centered. The four TD prompts
are followed by the four ASD prompts. CAM144 is reshaped into 16 by 9 latent
slots. These slots are latent coordinates, not anatomical ROIs.

The original classification, graph and auxiliary DARG losses remain intact,
including the E81.7.2-G1 raw residual regularizer with coefficient `0.001`.
The original shared gradient reference and `kappa=0.05` protection are used
for GFR, CAM and DARG. The historical optimizer order is unchanged.

## CM-PSRR change

CM-PSRR uses the original node-level Prompt-conditioned ASD/TD Entmax1.5
relations, 15 real non-self neighbors and one Null candidate. The Null message
is exactly zero and its probability is retained as stopped propagation mass.
The relation output is zero initialized and the CAM144 residual remains bounded
by the original 10 percent rule.

For each training step, the current state is evaluated twice with the same
public stochastic state. B uses `relation_off` and C uses the full relation
path. For legal training rows only,

`m = logp[true class] - logp[other class]` and
`delta_margin = m_C - stop_gradient(m_B)`.

The auxiliary loss is

`w = sigmoid(-stop_gradient(m_B) / 0.5)`

`L_gain = weighted_mean(w * softplus(0.05 - delta_margin))`

`L_safe = weighted_mean((1-w) * relu(-delta_margin - 0.02)^2)`

`L_margin = L_gain + L_safe`.

The auxiliary coefficient is zero through epoch 30, ramps linearly from epoch
31 through epoch 50, and is one from epoch 51 onward. Its gradient is extracted
only for the new relation parameters. Negative alignment with the main
relation gradient is projected out, then the effective auxiliary norm is capped
at 20 percent of the main relation norm. A zero main relation norm produces a
zero auxiliary update. The main classification loss remains the only source of
new gradients into the shared GFR/CAM/DARG parameters.

Fold 2 retains its historical overlap rows 1-based 377 and 407 in the original
data membership. Their zero-based rows 376 and 406 are excluded from the new
auxiliary supervision and from the added training residual contribution.

## Execution and audit

Only physical GPU 0 is allowed, through `CUDA_VISIBLE_DEVICES=0`. The runner
verifies the physical UUID before preflight and each worker. A private RNG
context initializes the new relation module and restores the public CPU RNG.
No epoch reseeding is performed. Best and Latest checkpoints include the full
base and relation state, all optimizer state, RNG state, warmup identity,
training trace and frozen source digest.

The independent audit reloads every Best and Latest checkpoint, recomputes all
six metrics, replays one Latest step, and evaluates A/B/C patient corrections.
Its edge deletion test passes explicit overrides into evaluation and verifies
that deleted real-edge mass moves to Null. A is historical DARG Best, B is the
current checkpoint with the new local residual disabled, and C is the full
checkpoint. A-to-B includes training-trajectory and Best-epoch differences.

Historical limitations remain part of the protocol: val equals test for Best
selection, RFE and standardization precede the outer folds, the patient graph
is transductive, and Fold 2 retains its historical train/test overlap.
