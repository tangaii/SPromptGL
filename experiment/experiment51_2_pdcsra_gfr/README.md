# Experiment 51.2 — PDC-SRA-GFR

PDC-SRA-GFR implements **Pairwise Disease-Conditioned Selective Relational Alignment for Graph-based Feature Representation**. It keeps the audited Experiment 51 E1 four-branch GFR+CAM path and adds a patient-pair selector. The selector is symmetric in `(i,j)`, is masked by the original GraphLearn support, and the same selected relation matrix drives the added graph message and the CORR-anchored geometry loss. The private `C` branch is never included in the geometry loss.

## Fixed protocol

- ABIDE 871 participants, input order `PHENO48 / ANAT6 / FUNC10 / CORR256`, labels ASD=0 and TD=1.
- seed=0; `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)`.
- Historical transductive preprocessing, `val_idx=test_idx` checkpoint selection, strict ACC improvement, 450 epochs and `wait>200` are retained for internal comparability.
- E1 initial parameter/RNG anchor is taken from `experiment43_1_b0b_gfr_mechanism_retest/.runtime/anchors/foldXX.pt`; the first epoch state is restored from `anchor['rng']`. New parameters are initialized in a fold-specific local Torch RNG fork and never alter the E1 global RNG.
- Original GFR/CAM optimizers retain `lr=0.0038, weight_decay=0.11`; new branches use `lr=0.001, weight_decay=0.001`.

## Groups

E0 is the readback-only B0-B reference. E1 is re-trained from the exact E1 anchor. E2 is self-only capacity, E3 ordinary graph propagation, E4 learned pair selection without pair supervision, E5 adds training-side pair supervision, E6 is the complete method, E7 replaces selected weights by uniform support weights, E8 permutes only non-CORR alignment correspondence, E9 permutes only the added graph relation, and E10 removes private `C` from the final residual while keeping its computation.

The pair task uses only outer-train labels and fixed balanced samples (at most 4096 pairs per fold). Geometry alignment uses a fixed balanced 128-patient outer-train subset and only valid shared GraphLearn support. No outer-test labels enter selection, pair loss, alignment, or routing.

## Reproduction

```bash
export PDC_EXP=/tmp/SPromptGL-E51_2/experiment51_2_pdcsra_gfr
python -B run.py prepare
python -B tests.py
PDC_WORKERS=3 CUDA_VISIBLE_DEVICES=0 python -B run.py controller
python -B run.py aggregate
python -B audit.py
```

The controller is resumable: completed task JSON files are skipped, and a `latest` checkpoint contains model, optimizer and Python/NumPy/Torch CPU/CUDA RNG states. Runtime checkpoints are deliberately kept outside the deliverable source/results tree.

## Audit scope and literature boundary

The implementation is grounded in the local SPromptGL source audit and the supplied SPromptGL, MGDR, MM-GTUNets and SMGD papers. MGDR motivates retaining private information; MM-GTUNets motivates unequal modality roles; SMGD motivates separating diagnosis relevance from shared geometry. SMGD's ROI-internal graphs are not copied: this experiment operates on SPromptGL's patient-to-patient graphs. The supplied online paper endpoints were unavailable during this run, so claims are limited to the local source snapshots and supplied bibliographic links. The historical `val=test`, CV-before-preprocessing and transductive protocol can be optimistic and is not an unbiased external-test estimate.
