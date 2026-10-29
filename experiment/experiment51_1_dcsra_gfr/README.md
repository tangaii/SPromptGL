# Experiment 51.1 — DC-SRA-GFR

DC-SRA-GFR keeps the audited Experiment 51 E1 four-branch GFR as its main path. It adds a small modality-specific graph/self view, a CORR anchored ASD/TD prototype, patient-level gates, CORR-to-auxiliary relation geometry, and a bounded residual written back to the original 48/6/10/256 positions before the historical PEN-free CAM.

The fixed protocol is seed 0, 10-fold StratifiedKFold (`shuffle=True`, `random_state=10`), historical `val=test` checkpoint selection, 450 epochs with `wait>200`, base learning rate 0.0038, base weight decay 0.11, and one GPU0 with at most five workers. E0 is independently read from Experiment 43.1; E1 is read from the complete Experiment 51 scratch checkpoints when available; E2–E11 are trained from the same fold-local E1 anchor and the same group-independent new-module initialization.

## Reproduction

```bash
export DCSRA_EXP=/tmp/SPromptGL-E51_1/experiment51_1_dcsra_gfr
python -B run.py prepare
python -B tests.py
CUDA_VISIBLE_DEVICES=0 python -B run.py controller
python -B run.py aggregate
python -B audit.py
python -B finalize_report.py
```

`DCSRA_EXP` may point to another writable copy. The runner reads the original data and historical anchors, never edits root model code, and writes fold state atomically. Completed tasks are skipped on recovery. `.runtime` contains only recovery state and best checkpoints.

The results retain the historical global preprocessing, transductive graph, and `val=test` selection limitations. They are internal protocol comparisons, not unbiased external-test estimates.

## Group controls

E0 is historical B0-B. E1 is the corrected four-message aggregation. E2 uses only self views for the added branch. E3 adds ordinary graph increments. E4 uses a fixed 0.5 gate. E5 learns patient gates without prototype or alignment. E6 adds CORR ASD/TD prototypes. E7 is the complete method. E8 uses the mean alignment weight. E9 permutes only auxiliary correspondence in the alignment loss. E10 permutes only the added graph branch adjacency. E11 computes the private branch but removes its direct residual contribution.
