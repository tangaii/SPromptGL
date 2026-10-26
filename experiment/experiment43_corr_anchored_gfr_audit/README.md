# Experiment 43 — CORR-anchored GFR audit

This seed-0 study separates aggregation overwrite, graph supervision, density,
propagation scope, and train-estimated auxiliary refinement. It does not study
Prompt semantics or introduce a text model. The immutable request snapshot and
its SHA are under `.runtime/request.txt` and `manifest.json`. Configuration is
frozen before formal tasks; no outer-test results tune parameters.

## Source audit and historical evidence

| Source/function | Executable fact | Test |
|---|---|---|
| root network.py:GraphLearn.forward, lines 153–172 | only `p` projects cosine; `>th` is detached; `w/t/threshold` are unused | per-parameter classification/graph/supervision gradients |
| root network.py:GFR.forward, lines 227–290 | i==1 overwrites i==0; `combined_weight` unused; every branch receives full 320D | E00 classification branch0 zero; E01 all4 active |
| root layers.py:GraphConv.forward, lines 192–205 | `A @ (X @ W)`, two GCN layers | shared original parameter shapes, exact initial state |
| root utils.py:GraphConstructLoss, lines 31–44; normalize_adj:101–108 | smoothness minus degree plus sparsity; symmetric D^-1/2 A D^-1/2 | four losses averaged and self loops unchanged in propagation |
| root model.py:EvalHelper.__init__/run_epoch | two original Adams, PLN apply(my_weight_init); simple-2 excludes computed ALoss | original loss/grad/two Adam steps parity |
| root main.py:train_and_eval | val=test, strict-first ACC, wait>200, seed0 split10 | anchor metrics/epochs/confusion/predictions |

The MICCAI2025 paper Sec2.1 Eq1 already describes supervised patient adjacency;
Eq2 propagates the full multimodal concatenation. Full propagation is an
architectural choice, not a bug. The local repository caches eight unused PEN
embeddings, whereas the author's current network retains RN50 and encodes in
forward. Official model.py matches the local SHA; utils differs in device-safe
one_hot. URLs, source hashes, downloaded paper text and exact diffs are audited.

Read-only history includes E32 integration/run/manifest/metrics and E35/36/37/40/42
reports and source records. E35 corrected message dose but not semantic gains;
E36/37 conditional edges and auxiliary label gradients did not establish useful
clinical text dependence. E40/42 showed failure against simple numerical evidence
and the need to audit actual branch effect. E42's validated historical degree
means (PHENO191.100/ANAT330.582/FUNC155.409/CORR93.219) are references only; E43
recomputes every group's best-checkpoint graph statistics.

## Data and paired anchor reconstruction

The existing 871×320 input is read in modal_feat_dict order, not guessed slices:
PHENO48 = 26 age bins, 20 site indicators and 2 sex indicators; ANAT6 and FUNC10
are imaging QC/motion summaries; CORR256 are selected FC features. ASD0=403,
TD1=468. Age bin index difference is an ordinal two-year-bin difference, not an
exact continuous age. No new clinical fields, preprocessing or RFE are added.

Each E40 immutable initial snapshot is checked against E32's per-fold SHA/RNG,
fold membership, protocol and four metrics. E43 retains the complete original
PLN/PEN parameter registry and eight historical buffers, but bypasses SPGS/RES.
The eight buffers are loaded from each snapshot without importing or running
CLIP. We reconstruct the original initial model/two empty Adam states and
explicitly restore its saved entry and post-construction Python/NumPy/CPU/CUDA
RNG. We do not claim to independently reproduce CLIP's constructor side effects;
the exact frozen initial state is the pairing definition. E00 is freshly trained
for all ten folds, five per GPU as requested. No historical best is used to train;
historical best predictions are checked after replay only. E01–E09 start only
after E00 exact acceptance, and every fold restores the same initial anchor.

## Frozen groups

| Group | Only declared change |
|---|---|
| E00 | Original GFR and B0-A forward |
| E01 | Restore X once plus all four gated messages |
| E02 | Original aggregation/threshold, add train-only unordered raw-cosine MSE, lambda1 |
| E03 | Original aggregation, mutual96 + detached train-pair median/IQR sigmoid |
| E04 | E01, zero other input modalities before each same-shaped GCN |
| E05 | E01 + E03 |
| E06 | E05 aggregation, all supports CORR mutual; auxiliary exp(.25*q*(2*C-1)) |
| E07 | E06, q=1/3 each |
| E08 | E06, independently permute each auxiliary C's rows/columns within outer train/test pools |
| E09 | E06, q=0; auxiliary graph builders may be unused |

For calibrated graphs the raw diagonal is zero; propagation adds I once. The
original threshold graphs retain their original self-similarity diagonal and
also add I, exactly as history. This diagonal difference is part of the frozen
density construction and is disclosed rather than silently hidden.

Reliability uses five subject-disjoint inner folds, random_state=43000+outer_fold.
All unordered inner-fit subject pairs fit StandardScaler and logistic regression
(C1, balanced classes, lbfgs, max_iter500, tol1e-6); only pairs entirely inside
inner-held subjects evaluate continuous same-diagnosis pair AUC. Baseline is raw
CORR cosine plus age-bin difference/sex/site match; add each raw auxiliary cosine.
Mean positive AUC increments normalize to q, or all zero if none positive. This
is a mechanism endpoint, not an extra classification metric. Learned GraphLearn
scores, not those raw input cosines, drive formal graph calibration/refinement.
Held-out outer labels never enter these functions. Coefficients are fixed, not
tuned on the outer test set.

## Reproduction and recovery

`python run.py all` runs prepare → preflight → 10-fold E00 gate → 90 formal folds →
report and safe handoff append. `python run.py resume` validates and skips only
atomic completed tasks with matching best-checkpoint SHA. A task is
(group,seed0,fold), max4 workers per physical GPU, two CPU threads per worker.
The requested dynamic schedule gives each GPU five E00 and 45 innovation folds.
Each ten epochs and at stopping, checkpoints save full model, both Adams, best,
wait, progress, four RNG states and source/input/anchor signatures. A simulated
disk interruption must reproduce the third update exactly. `.runtime` contains
the minimal private sources, anchors, caches, task locks, checkpoints and logs.
One best per completed fold is retained; per-epoch large tensor dumps are absent.

Formal metrics only ACC / original hard-one-hot AUC / SPE-TD / SEN-ASD, mean±ten
fold population SD. Per-epoch E02 losses and component projection gradients are
in loss_trace.csv. Best-checkpoint classification-only gradients, graph geometry,
branch-off effects and q are separate diagnostics. The predefined E06 GO requires
>.1pp ACC and Hard-AUC plus ≥7/10 wins against E00/E05/E07/E08/E09 and actual
bounded auxiliary use. No added groups or seeds are permitted after inspection.

Results inherit test-selected checkpoints, pre-CV feature selection/global
preprocessing and transductive patient graphs. Overlapping folds are not ten
independent datasets; no independent clinical generalization, significance or
causality is claimed. Only handoff.md is appended outside this directory; root
models/data/history stay read-only, no Git commit or push.

## Independent final verification

After the 100 folds, run `CUDA_VISIBLE_DEVICES=1 python audit.py`. This small,
separate audit is necessary because disabling a whole GCN branch does not isolate
the auxiliary edge refinement. It changes only q or auxiliary correspondence at
the fixed E06 checkpoint, restores and checks the original output bitwise, and
recomputes all four formal metrics from the saved predictions. It neither trains
another group nor changes the frozen method. Evidence and environment hashes are
saved in `.runtime/final_check.json`; additional fields join diagnostics.csv.
If `python run.py report` regenerates the basic report, rerunning this audit
restores the appendix. Both handoff additions are append-only and idempotent.
