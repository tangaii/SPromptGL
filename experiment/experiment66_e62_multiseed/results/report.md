# Experiment 66 — E62-E2 Multi-seed Robustness

Five requested seeds (1, 2, 3, 2020, 2026) were run on the fixed E62-E2 architecture and protocol. For each fold, all seeds load identical historical anchor parameters, split and balanced patient-pair cache. The requested seed is added to the E61 per-fold construction seed `510000 + fold*100 + 1`; it creates a distinct deterministic Python/NumPy/PyTorch CPU/CUDA stream from model construction onward. Relation model initialization remains isolated at the fixed per-fold CADR init seed. This is training-RNG robustness with fixed initialized weights, not a random-weight initialization study.

Each seed/fold independently replays ten warmup epochs from epoch 1 and continues through the E2 branch from epoch 11. No best checkpoint is used as the training start, no per-epoch seed reset occurs, and no OOF or reliability multiplier is loaded. All E62-E2 settings, including gates, optimizer/loss configuration, folds, stopping, and checkpoint selection, remain fixed.

Base lineage: E62-E2 and E61/E53.3, commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. Source, configuration, historical references, anchors, splits, pair caches and checkpoints were SHA-verified. Historical `val=test` selection, pre-CV preprocessing and the transductive 871-node graph remain, so these are exploratory protocol-matched results rather than unbiased external validation.

Status: **COMPLETE_AUDITED**; 50/50 tasks completed, 50/50 independent checkpoint audits passed. Device `PPU-ZW810E`, worker limit 10, observed peak 10.

## Results

Ten-fold mean ± population SD, percent. Historical rows are read-only references.

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|53.3 third-group read-only|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|
|E61-E2 historical|91.4995 ± 2.4842|91.2480 ± 2.5179|93.8126 ± 2.6938|94.6485 ± 2.4105|87.8476 ± 3.5514|91.2480 ± 2.5179|90.5253 ± 2.7692|
|E62-E2 historical (read-only)|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|93.7789 ± 4.2653|89.5854 ± 3.4710|91.6821 ± 2.8919|91.0789 ± 3.0892|
|E66-S1|91.8443 ± 2.6508|91.5625 ± 2.6336|93.4891 ± 2.7033|95.2775 ± 3.4660|87.8476 ± 3.3975|91.5625 ± 2.6336|90.9003 ± 2.8609|
|E66-S2|91.7294 ± 2.8596|91.5643 ± 2.8384|93.2326 ± 3.3309|93.5615 ± 4.6416|89.5671 ± 4.3208|91.5643 ± 2.8384|90.9449 ± 3.0496|
|E66-S3|91.7294 ± 2.9950|91.5439 ± 2.9605|93.0725 ± 3.0079|93.9963 ± 4.0691|89.0915 ± 3.5263|91.5439 ± 2.9605|90.9133 ± 3.1924|
|E66-S2020|91.3845 ± 3.0154|91.1347 ± 3.0081|93.3587 ± 3.3137|94.4218 ± 3.5259|87.8476 ± 3.5683|91.1347 ± 3.0081|90.4348 ± 3.2858|
|E66-S2026|91.7294 ± 2.7174|91.5518 ± 2.6926|93.1173 ± 2.9231|93.7743 ± 4.5082|89.3293 ± 4.0216|91.5518 ± 2.6926|90.9278 ± 2.8607|

## Paired Changes

Changes are seed-group minus reference in percentage points, paired by the unchanged outer fold. Descriptive t-based 95% intervals are not confirmatory because folds share training data.

|Seed group − E62-E2|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E66-S1|-0.000 (1/8/1; -0.388,+0.388)|-0.120 (1/3/6; -0.518,+0.279)|+0.288 (4/0/6; -0.484,+1.060)|-1.738 (0/4/6; -2.941,-0.535)|+1.499 (6/4/0; +0.468,+2.529)|
|E66-S2|-0.115 (1/7/2; -0.582,+0.352)|-0.118 (3/4/3; -0.630,+0.395)|+0.032 (5/1/4; -0.289,+0.352)|-0.018 (3/5/2; -1.689,+1.652)|-0.217 (2/5/3; -1.345,+0.910)|
|E66-S3|-0.115 (0/9/1; -0.375,+0.145)|-0.138 (0/7/3; -0.376,+0.099)|-0.128 (3/1/6; -0.780,+0.523)|-0.494 (0/8/2; -1.239,+0.251)|+0.217 (2/7/1; -0.652,+1.087)|
|E66-S2020|-0.460 (1/5/4; -1.153,+0.234)|-0.547 (1/3/6; -1.262,+0.167)|+0.158 (5/0/5; -0.797,+1.112)|-1.738 (0/5/5; -3.195,-0.281)|+0.643 (4/5/1; -0.388,+1.674)|
|E66-S2026|-0.115 (2/5/3; -0.722,+0.492)|-0.130 (4/2/4; -0.787,+0.526)|-0.084 (4/0/6; -0.726,+0.559)|-0.256 (2/5/3; -2.689,+2.177)|-0.005 (5/2/3; -1.918,+1.909)|

## Initialization and RNG Audit

Within every fold, preflight confirmed identical shared/GFR/CAM/CADR parameter digest, split SHA, pair SHA and anchor SHA across all five seeds; each requested seed had a distinct initial RNG SHA. Rebuilding each fold-1 seed twice reproduced its parameter and RNG SHA. Epoch RNG hashes intentionally diverge across seed groups; the loop call sites are shared and no epoch reseeding is used. This is not identical random-number parity across seeds.

|Group|Seed|Completed folds|Independent audits|Mean best epoch|Mean executed epochs|
|---|---:|---:|---:|---:|---:|
|E66-S1|1|10/10|10/10|208.8|387.1|
|E66-S2|2|10/10|10/10|232.6|406.7|
|E66-S3|3|10/10|10/10|231.9|414.7|
|E66-S2020|2020|10/10|10/10|214.4|395.4|
|E66-S2026|2026|10/10|10/10|222.9|396.8|

## Learned Gates

Ten-fold means from independently read-back best checkpoints.

|Group|Modality|Edge gate|Fusion gate|Weighted CADR/input norm|
|---|---|---:|---:|---:|
|E66-S1|PHENO|0.48242|0.000945|0.0000026|
|E66-S1|ANAT|0.48261|0.000966|0.0000012|
|E66-S1|FUNC|0.46742|0.000946|0.0000018|
|E66-S1|CORR|0.54742|0.991795|0.0040544|
|E66-S2|PHENO|0.47571|0.000933|0.0000030|
|E66-S2|ANAT|0.47486|0.000963|0.0000012|
|E66-S2|FUNC|0.45878|0.000918|0.0000019|
|E66-S2|CORR|0.55291|0.992032|0.0039791|
|E66-S3|PHENO|0.47634|0.000937|0.0000027|
|E66-S3|ANAT|0.47795|0.000973|0.0000014|
|E66-S3|FUNC|0.45972|0.000917|0.0000020|
|E66-S3|CORR|0.55271|0.992036|0.0039775|
|E66-S2020|PHENO|0.48135|0.000942|0.0000026|
|E66-S2020|ANAT|0.47900|0.000965|0.0000012|
|E66-S2020|FUNC|0.46659|0.000935|0.0000019|
|E66-S2020|CORR|0.54914|0.991841|0.0039796|
|E66-S2026|PHENO|0.47922|0.000959|0.0000029|
|E66-S2026|ANAT|0.47925|0.000972|0.0000014|
|E66-S2026|FUNC|0.46289|0.000929|0.0000020|
|E66-S2026|CORR|0.55106|0.991928|0.0040195|

## Interpretation

This five-seed extension estimates sensitivity to training randomness under fixed fold anchors and fixed data/pair construction. Seed-level rankings are descriptive; they do not establish a generally superior seed. Any comparison with E62-E2 is additionally limited by the historical checkpoint-selection and preprocessing protocol.

Files: `/model/spromptgl/experiment/experiment66_e62_multiseed/results/fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `gate_diagnostics.csv`, `checkpoint_audit.csv`; manifest `/model/spromptgl/experiment/experiment66_e62_multiseed/manifest.json`.
