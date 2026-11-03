# Experiment 65 — E62-E2 Weight-Decay Sensitivity

## Protocol

Three formal groups change one optimizer weight decay relative to E62-E2: G1 changes all CADR AdamW parameter groups from 0.01 to 0.1; G2 changes them to 0.001; G3 changes GFR Adam from 0.11 to 0.2. All other settings, including CAM WD=0.11, learning rates, edge gates, adaptive gates, losses, data, folds, seed, and E2 forward, remain fixed.

Because weight decay changes parameter updates from the first optimizer step, each group/fold replays epochs 1–10 from the same SHA-verified untrained Experiment 43.1 fold anchor and initial RNG. The already-trained E61/E62 warmup is used only as a read-only source for verifying anchor, split, pair, and provenance; no best checkpoint is used as a fine-tuning start. Group-specific warmup model/optimizer/RNG states are saved and then continued from epoch 11.

E61/E62 base source is pinned to commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. E61 runner/model/config and Experiment 53.3 dependencies, E61 epoch-10 provenance files, E62-E2 config/manifest/checkpoints, and all fold anchor identities were SHA-checked. E62-E2 initial sigmoid gates are edge `[0.5]*4` and fusion `[0.001,0.001,0.001,0.99]`.

Training runner SHA is `20fe73063d650b6b5109a48edbd52b5f3e81021ffd6a6ba968c69c6b4db4751d`. After formal training, audit/finalization-only code was added; the current report/manifest runner SHA is `30f0e2dbcc1fda2bb07e192ba33a51ad8b62a1a40797edca6fdea17602e8a666`. Per-task metadata retains the training runner SHA, and the saved training logic/model states were not changed by that reporting update.

Historical `val=test` checkpoint selection, pre-CV RFE/standardization and the transductive 871-node graph remain. The results are protocol-matched exploratory comparisons, not unbiased external generalization estimates. No OOF, reliability multiplier, extra model, loss, or parameter sweep was added.

Formal status: **COMPLETE_AUDITED**, 30/30 tasks and 30/30 independent readbacks; device PPU-ZW810E, worker limit 10. No worker failure or OOM was recorded.

## Classification results

Values are ten-fold mean ± population SD, percent. Historical groups are read-only.

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|53.3 third-group read-only|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|
|E61-E2 historical|91.4995 ± 2.4842|91.2480 ± 2.5179|93.8126 ± 2.6938|94.6485 ± 2.4105|87.8476 ± 3.5514|91.2480 ± 2.5179|90.5253 ± 2.7692|
|E62-E2 historical (read-only)|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|93.7789 ± 4.2653|89.5854 ± 3.4710|91.6821 ± 2.8919|91.0789 ± 3.0892|
|E65-G1|91.5008 ± 3.0987|91.2651 ± 3.1456|93.5200 ± 2.8186|94.4265 ± 3.2394|88.1037 ± 4.6438|91.2651 ± 3.1456|90.5471 ± 3.4503|
|E65-G2|91.7294 ± 2.9950|91.4778 ± 2.9886|93.5832 ± 2.8210|94.8520 ± 3.3815|88.1037 ± 3.5966|91.4778 ± 2.9886|90.8070 ± 3.2445|
|E65-G3|91.4995 ± 3.0565|91.2651 ± 3.0828|93.6804 ± 2.7211|94.4265 ± 3.3763|88.1037 ± 4.2350|91.2651 ± 3.0828|90.5586 ± 3.3521|

## Paired comparisons

Differences are percentage points with descriptive paired-t 95% intervals over ten overlapping CV folds; they are not confirmatory confidence claims.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E65-G1 − 53.3-HISTORICAL|-0.573 (0/5/5; -1.006,-0.141)|-0.600 (0/5/5; -1.057,-0.144)|-0.148 (3/0/7; -0.953,+0.656)|-0.988 (0/7/3; -2.216,+0.241)|-0.213 (1/7/2; -1.090,+0.664)|
|E65-G1 − E61-E2-HISTORICAL|+0.001 (4/3/3; -0.864,+0.867)|+0.017 (5/1/4; -0.869,+0.903)|-0.293 (6/0/4; -1.265,+0.679)|+0.256 (4/3/3; -1.293,+1.805)|-0.222 (2/5/3; -1.354,+0.910)|
|E65-G1 − E62-E2-HISTORICAL|-0.344 (1/5/4; -0.897,+0.210)|-0.417 (1/4/5; -0.996,+0.162)|+0.319 (6/0/4; -0.525,+1.163)|-1.482 (0/6/4; -3.173,+0.210)|+0.648 (3/6/1; -0.621,+1.916)|
|E65-G2 − 53.3-HISTORICAL|-0.345 (1/5/4; -0.900,+0.210)|-0.388 (1/5/4; -0.958,+0.183)|-0.085 (3/0/7; -1.158,+0.988)|-0.988 (0/6/4; -1.900,-0.075)|+0.213 (1/9/0; -0.269,+0.694)|
|E65-G2 − E61-E2-HISTORICAL|+0.230 (4/5/1; -0.526,+0.985)|+0.230 (5/3/2; -0.494,+0.954)|-0.229 (4/0/6; -1.197,+0.738)|+0.256 (2/7/1; -0.750,+1.262)|+0.204 (4/4/2; -1.326,+1.733)|
|E65-G2 − E62-E2-HISTORICAL|-0.115 (1/7/2; -0.582,+0.352)|-0.204 (1/5/4; -0.719,+0.310)|+0.382 (6/0/4; -0.555,+1.319)|-1.482 (0/6/4; -2.972,+0.008)|+1.073 (5/5/0; +0.264,+1.882)|
|E65-G3 − 53.3-HISTORICAL|-0.575 (0/5/5; -1.008,-0.141)|-0.600 (0/4/6; -1.012,-0.189)|+0.012 (4/0/6; -0.690,+0.714)|-0.988 (1/5/4; -2.489,+0.513)|-0.213 (1/7/2; -1.726,+1.301)|
|E65-G3 − E61-E2-HISTORICAL|+0.000 (2/7/1; -0.671,+0.671)|+0.017 (3/4/3; -0.655,+0.689)|-0.132 (5/0/5; -0.698,+0.434)|+0.256 (3/5/2; -1.056,+1.569)|-0.222 (2/6/2; -1.571,+1.127)|
|E65-G3 − E62-E2-HISTORICAL|-0.345 (1/5/4; -0.900,+0.210)|-0.417 (1/3/6; -0.968,+0.134)|+0.479 (7/0/3; -0.164,+1.123)|-1.482 (1/5/4; -3.390,+0.427)|+0.648 (5/3/2; -1.122,+2.417)|
|E65-G1 − E65-G2|-0.229 (2/4/4; -0.876,+0.419)|-0.213 (3/2/5; -0.891,+0.466)|-0.063 (3/2/5; -0.794,+0.667)|+0.000 (3/4/3; -1.448,+1.448)|-0.426 (1/6/3; -1.400,+0.549)|
|E65-G1 − E65-G3|+0.001 (2/6/2; -0.545,+0.548)|+0.000 (3/3/4; -0.521,+0.521)|-0.160 (5/0/5; -0.905,+0.584)|+0.000 (3/5/2; -1.676,+1.676)|-0.000 (3/4/3; -1.764,+1.764)|
|E65-G2 − E65-G3|+0.230 (3/6/1; -0.290,+0.750)|+0.213 (4/5/1; -0.281,+0.706)|-0.097 (4/0/6; -0.986,+0.791)|+0.000 (1/7/2; -1.460,+1.460)|+0.426 (3/6/1; -1.146,+1.997)|

## Optimizer and RNG audit

Each group's ten-epoch warmup trace is stored with its checkpoint provenance. Weight decay does not consume RNG; group warmup and continuation traces are compared only at common epochs. Any divergence is reported in `manifest.json` and is not described as exact parameter parity after optimizer updates.

|Group|GFR WD|CAM WD|CADR WD|Folds|Audits|
|---|---:|---:|---:|---:|---:|
|E65-G1|0.11|0.11|0.1|10|10/10|
|E65-G2|0.11|0.11|0.001|10|10/10|
|E65-G3|0.2|0.11|0.01|10|10/10|

## Mechanism checks

Mean final gates and fixed-checkpoint interventions are descriptive. Intervention ΔACC is full model minus propagation ablation across the ten folds; the historical checkpoint selection protocol remains in force.

|Group|Modality|Mean final edge gate|Mean final fusion gate|Mean weighted CADR/input norm|Total active scored edges|
|---|---|---:|---:|---:|---:|
|E65-G1|PHENO|0.48139|0.001072|0.0000042|23622|
|E65-G1|ANAT|0.48130|0.001089|0.0000019|11243|
|E65-G1|FUNC|0.46825|0.001050|0.0000022|17649|
|E65-G1|CORR|0.54698|0.991145|0.0039274|30416|
|E65-G2|PHENO|0.48740|0.000961|0.0000027|24380|
|E65-G2|ANAT|0.48785|0.000969|0.0000010|8438|
|E65-G2|FUNC|0.47446|0.000942|0.0000017|16364|
|E65-G2|CORR|0.54204|0.991615|0.0040407|32188|
|E65-G3|PHENO|0.48478|0.000966|0.0000026|25213|
|E65-G3|ANAT|0.48686|0.000993|0.0000011|9033|
|E65-G3|FUNC|0.47214|0.000959|0.0000017|15962|
|E65-G3|CORR|0.54346|0.991581|0.0040282|32526|

|Group|Intervention|Mean ΔACC (pp)|Total hard prediction flips|Mean absolute probability change|
|---|---|---:|---:|---:|
|E65-G1|all_cadr_off|+0.0000|0|0.0015300|
|E65-G1|corr_cadr_off|+0.0000|0|0.0015302|
|E65-G1|direct_off|+0.0000|0|0.0015242|
|E65-G1|score_permutation|+0.0000|0|0.0022349|
|E65-G2|all_cadr_off|+0.3448|3|0.0016089|
|E65-G2|corr_cadr_off|+0.3448|3|0.0016091|
|E65-G2|direct_off|+0.3448|3|0.0014784|
|E65-G2|score_permutation|+0.2299|2|0.0021995|
|E65-G3|all_cadr_off|+0.2299|4|0.0015905|
|E65-G3|corr_cadr_off|+0.2299|4|0.0015908|
|E65-G3|direct_off|+0.2299|4|0.0015904|
|E65-G3|score_permutation|+0.1149|3|0.0022376|

Removing all CADR propagation changed 0, 3, and 4 hard predictions across the ten folds for G1, G2, and G3, respectively; mean probability changes were only about 0.0015–0.0016. This is limited evidence of diagnostic reliance on CADR propagation.

## Scientific reading

E65-G2 (CADR WD=0.001) is the strongest new group on ACC, Hard-AUC, and SPE, but remains 0.115 pp below E62-E2 in ACC and 1.482 pp lower in SEN-ASD. No tested setting improves ACC, Hard-AUC, and SEN together over E62-E2 or the 53.3 reference. The paired differences among new groups are small, with descriptive intervals overlapping zero. These are optimizer sensitivity results, not architecture innovations or evidence that a new CADR method is established. The single seed, overlapping CV training sets, historical `val=test` checkpoint selection, and preprocessing before CV limit inference; intervals are descriptive only.

## Files

- Experiment root: `/model/spromptgl/experiment/experiment65_wd_sensitivity`
- Task manifest: `manifest.json`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Gate diagnostics: `results/gate_diagnostics.csv`
- Independent checkpoint audit: `results/checkpoint_audit.csv`
