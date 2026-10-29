# Experiment 50.2 — PCE-Router

PCE-Router tests patient-specific, signed ASD/TD FC evidence routing after the verified B0-B GFR output. It changes only the current patient's FC slice (indices 64:320), keeps the first 64 GFR dimensions unchanged, and sends the result through the original PEN-free CAM. It does not construct patient graphs, add cross-patient messages, use prompts, CLIP, GPT, or counterfactual teacher supervision.

## Fixed protocol

- B0-B: physical PEN removal, original GFR/CAM, original PHENO overwrite and `ALoss` non-backprop behavior.
- ABIDE 871, PHENO/ANAT/FUNC/CORR dimensions 48/6/10/256, ASD=0 and TD=1.
- seed 0; 10-fold StratifiedKFold(shuffle=True, random_state=10).
- `simple-2`, `val_idx=test_idx`, strict higher test ACC checkpoint selection, max epoch 450, `wait>200`.
- Original GFR/CAM Adam weight decay 0.11; PCE Adam learning rate 0.00038 and weight decay 0.11.
- Only CUDA device 0 is used; controller permits at most five workers.

The formal outputs are under `results/`. Runtime checkpoints and recovery state are under `.runtime/` in the execution scratch and are excluded from compact delivery copies. The packaged `source/` directory contains the immutable B0-B source snapshot used by the runner.

## Groups

E0 is the independently read Experiment43.1 B0-B reference. E1 is a category-independent self residual, E2 is class-agnostic FC selection, E3 is competitive ASD/TD evidence without conditions, E4 is condition routing without class competition, E5 is the full PCE-Router, E6 permutes individual conditions within train/test site strata, E7 permutes the learned FC evidence positions per patient, and E8 removes only the evidence auxiliary loss.

## Commands

```bash
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight
python -B run.py controller
python -B run.py aggregate
CUDA_VISIBLE_DEVICES=0 python -B run.py audit
python -B run.py aggregate
python -B run.py report
python -B run.py handoff
```

Hard-AUC is the original hard-label one-hot ROC-AUC; Probability-AUC uses TD probability and label 1. The historical `val=test`, pre-CV RFE/standardization, transductive GFR and single-seed limits remain part of the protocol.
