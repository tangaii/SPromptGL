# Experiment 48.2 — CARE-GFR

CARE-GFR (Conflict-Aware Reliable Edge Adversarial Alignment for Multi-Modal ASD Diagnosis) evaluates disease-relation supervision, conflict-aware consensus weighting, and constrained counterfactual edge perturbation on the physical-PEN-removed B0-B setting.

This experiment is isolated from the project model files. It preserves the ABIDE A 871-subject protocol (ASD=0, TD=1), the historical 320-dimensional input and 10-fold StratifiedKFold (`seed=0`, `shuffle=True`, `random_state=10`). E0 is a read-back of the verified B0-B checkpoint set. E1–E9 are independently initialized, fully trained ten-fold tasks under the frozen configuration.

## Groups

- E0: historical B0-B reference.
- E1: repaired GFR with PHENO participation, candidate/propagation edges, and low-decay residual message blocks.
- E2: E1 plus outer-train disease-relation scoring (DREL).
- E3: E2 plus confidence-weighted same-pair relation consensus (CATA).
- E4: E3 plus constrained counterfactual relation attack and stability loss (ACRL); primary CARE-GFR group.
- E5: E4 with consensus alignment removed.
- E6: E3 with a source-style FUNC/CORR feature control in place of ACRL.
- E7: E4 with deterministic site/split-stratified relation identity permutation.
- E8: E4 with shuffled outer-train pair labels only for DREL supervision.
- E9: E3 with budget-matched random legal edge replacements instead of inner-maximized attacks.

The method implementation is in `method.py`; training and resumption are in `run.py`; independent readback is in `audit.py`; the final Chinese report is reproducibly assembled from audited CSV files by `finalize_report.py`. `config.json` is frozen before formal execution. `.runtime/sources/` stores the source snapshots and `.runtime/tasks/` stores one atomic record for every group/fold task.

## Reproduction

```bash
python -B run.py init
python -B run.py preflight
CUDA_VISIBLE_DEVICES=0 python -B run.py worker --gpu 0 --folds 1
CUDA_VISIBLE_DEVICES=1 python -B run.py worker --gpu 1 --folds 6
CUDA_VISIBLE_DEVICES=0 python -B audit.py
```

The workers actually used one process per assigned fold to avoid task-file collisions. During execution a deterministic vectorized implementation replaced the original per-candidate scalar synchronization in the inner edge choice; candidate pool, edge budget, objective and outputs are otherwise unchanged, and all E1–E9 tasks were rerun under the resulting uniform signature. The formal protocol does not use test labels for DREL, CATA, or ACRL. Historical `val=test` checkpoint selection is retained only for comparability with B0-B and is reported as a protocol limitation.

## Interpretation limits

The repaired GFR is an engineering control and is not itself claimed as the CARE innovation. The current implementation reports the actual effects of its frozen low-capacity relation scorer, selective consensus, and constrained edge perturbation; no parameter was selected from outer-test accuracy. Results are one seed, transductive, and inherit the historical global preprocessing/CV-before-RFE and checkpoint-selection biases.
