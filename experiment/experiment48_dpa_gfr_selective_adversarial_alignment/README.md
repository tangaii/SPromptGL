# Experiment 48 — DPA-GFR

This directory is an isolated implementation of **Discriminability-Preserving
Adversarial Graph Learning** for the ABIDE A four-modal protocol. It preserves
the verified Experiment 43.1 B0-B base (original GFR and CAM/PLN without PEN)
and adds a sender-restricted, non-self patient evidence branch, selective
CORR-anchored domain adversaries, and an explicit PCGrad projection.

The eight pre-registered groups are E0 B0-B reference, E1 self-capacity,
E2 true graph, E3 all three adversaries, E4 CORR-FUNC selective adversary, E5
DPA-GFR with PCGrad, E6 random graph control, and E7 CORR-ANAT wrong-pair
control. Every group uses the same outer folds and checkpoint rule. E0 uses
the independently read-back, immutable 43.1 B0-B best checkpoints as explicitly
allowed by the protocol; E1-E7 are trained from the same per-fold base anchors.

The new branch uses 32-dimensional modality projections, K=16 sender-restricted
non-self neighbors, 64-dimensional evidence, a graph-only auxiliary classifier,
and a bounded fusion coefficient. Test labels are not used to build graphs,
train encoders, train domain discriminators, or choose hyperparameters. The
historical `val_idx=test_idx` checkpoint convention is retained and reported as
a protocol bias. The original GFR graph regularizer and two B0-B Adam optimizers
remain unchanged; the original GFR ALoss remains computed but excluded from
the simple-2 loss as in the audited base.

Run from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 python -B run.py init
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight
CUDA_VISIBLE_DEVICES=0 python -B run.py worker --gpu 0 --folds 1,2,3,4,5
CUDA_VISIBLE_DEVICES=1 python -B run.py worker --gpu 1 --folds 6,7,8,9,10
python -B audit.py all
```

Training is resumable from `.runtime/checkpoints/*.latest.pt`; completed tasks
are skipped only when their code signature and fold signature match. The
results are diagnostic evidence on this preprocessed ABIDE cohort, not an
unbiased external-generalization claim. No PID threshold, extra seed, PEN,
GFR replacement, or adversarial lambda search is used.
