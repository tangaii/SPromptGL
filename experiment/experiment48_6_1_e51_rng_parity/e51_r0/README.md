# Experiment 51 — DUGSA

DUGSA (Disease-Utility-Guided Selective Graph Alignment) is an isolated experiment for the SPromptGL B0-B setting. It keeps the original four GraphLearn graphs, adds modality-specific 32-dimensional encoders and graph/self GCN paths, separates shared/complementary/nuisance candidate channels, and uses outer-train-only inner-five-fold relation utility to gate shared graph messages and geometric alignment.

The experiment uses seed 0, 10-fold StratifiedKFold (`shuffle=True, random_state=10`), historical `val_idx=test_idx` checkpoint selection, 450 epochs, `wait>200`, original B0-B GFR/CAM weight decay 0.11, and one GPU0 with five workers. E0 is read back from verified Experiment43.1 B0-B checkpoints. E1-E11 are 110 new folds.

## Reproduction

```bash
export DUGSA_EXP=/path/to/experiment51_dugsa
python -B run.py prepare
python -B tests.py
CUDA_VISIBLE_DEVICES=0 python -B run.py controller
python -B run.py aggregate
python -B audit.py
python -B finalize_report.py
```

`DUGSA_EXP` can point to a local scratch copy. The runner uses the original read-only data and historical anchors; it never edits root model files. Tasks are atomic JSON records and each fold has a best checkpoint, so completed folds are skipped on recovery. The final report is generated only from independent checkpoint readback and the saved CSV files.

## Groups

E0 is B0-B; E1 repairs the historical four-message overwrite; E2 uses modality-specific graph encoders; E3 adds shared/private candidates; E4 adds S/C/N constraints without utility alignment; E5 is full DUGSA; E6 removes geometric alignment; E7 uses a uniform utility; E8 permutes utility across modality pairs; E9 uses a site/split-stratified patient relation permutation; E10 uses self-loops only; E11 removes the complementary classifier path.

The results inherit the historical global preprocessing, transductive graph, and `val=test` checkpoint-selection limitations. They are internal protocol comparisons rather than unbiased external-test estimates.
