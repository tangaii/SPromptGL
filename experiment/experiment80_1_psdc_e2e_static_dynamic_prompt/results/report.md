# Experiment 80.1 — PSDC End-to-End Dynamic Complementary Learning

Status: `COMPLETE_AUDITED`. Scientific decision: `NO_GO_PSDC_END_TO_END`.

The only trained groups are G1, G2, and G3, each initialized from the matching E61 epoch-10 warmup. B0 is the read-only E74.4-G2 historical DARG. The 90-second sequence cache covers all 871 ABIDE A participants and contains 252 fixed E11 connections after excluding the four edges incident to unknown ROI index 82.

The historical protocol is retained: validation equals test for historical checkpoint selection, pre-CV RFE and standardization, a transductive 871-patient graph, and the Fold 2 overlap at zero-based rows 376 and 406 (one-based 377 and 407). These scores are protocol-matched exploratory comparisons, not unbiased clinical generalization estimates. The BOLD files have no framewise censoring timestamps, so irregular deletion or concatenation cannot be audited.

## Fold Mean ± Population SD (%)

| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 historical DARG | 92.19 ± 2.76 | 92.03 ± 2.73 | 93.49 ± 2.92 | 90.07 ± 3.89 | 93.99 ± 4.73 | 91.46 ± 2.90 |
| Static-Surrogate + Anonymous Prompt | 87.48 ± 3.59 | 87.34 ± 3.67 | 89.65 ± 3.32 | 85.37 ± 5.10 | 89.31 ± 2.73 | 86.29 ± 4.03 |
| Real-Dynamic + Anonymous Prompt | 87.71 ± 5.20 | 87.47 ± 5.13 | 90.48 ± 5.38 | 84.38 ± 4.75 | 90.56 ± 6.21 | 86.49 ± 5.40 |
| PSDC-Medical | 88.29 ± 4.57 | 88.10 ± 4.57 | 90.79 ± 4.94 | 85.63 ± 5.21 | 90.57 ± 4.97 | 87.15 ± 4.90 |

## Pooled OOF Metrics (%)

| Method | ACC | Hard-AUC / BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E74.4-G2 historical DARG | 92.19 | 92.05 | 92.90 | 90.07 | 94.02 | 91.44 |
| Static-Surrogate + Anonymous Prompt | 87.49 | 87.34 | 89.14 | 85.36 | 89.32 | 86.32 |
| Real-Dynamic + Anonymous Prompt | 87.72 | 87.48 | 90.42 | 84.37 | 90.60 | 86.40 |
| PSDC-Medical | 88.29 | 88.10 | 89.09 | 85.61 | 90.60 | 87.12 |

## Paired Comparisons

Each interval is a descriptive paired t interval over ten folds; it is not a confirmatory significance test. W/T/L counts favor the first method.

| Contrast | Metric | Mean Δ pp | 95% interval pp | W/T/L | Pooled OOF Δ pp | Excluding Fold 2 Δ pp |
|---|---|---:|---:|---:|---:|---:|
| G1_minus_B0 | ACC | -4.71 | [-6.10, -3.31] | 0/1/9 | -4.71 | -4.72 |
| G1_minus_B0 | Hard-AUC / BA | -4.69 | [-6.22, -3.16] | 1/0/9 | -4.71 | -4.68 |
| G1_minus_B0 | Prob-AUC | -3.85 | [-5.78, -1.92] | 1/0/9 | -3.76 | -4.29 |
| G1_minus_B0 | SEN-ASD | -4.70 | [-9.13, -0.26] | 1/1/8 | -4.71 | -4.40 |
| G1_minus_B0 | SPE-TD | -4.68 | [-7.47, -1.89] | 1/1/8 | -4.70 | -4.96 |
| G1_minus_B0 | ASD-F1 | -5.17 | [-6.93, -3.42] | 1/0/9 | -5.11 | -5.16 |
| G2_minus_G1 | ACC | +0.23 | [-2.34, +2.79] | 4/2/4 | +0.23 | +0.63 |
| G2_minus_G1 | Hard-AUC / BA | +0.13 | [-2.39, +2.66] | 5/1/4 | +0.14 | +0.52 |
| G2_minus_G1 | Prob-AUC | +0.83 | [-2.57, +4.24] | 7/0/3 | +1.27 | +1.44 |
| G2_minus_G1 | SEN-ASD | -0.99 | [-3.78, +1.80] | 3/3/4 | -0.99 | -0.83 |
| G2_minus_G1 | SPE-TD | +1.25 | [-2.38, +4.89] | 5/2/3 | +1.28 | +1.88 |
| G2_minus_G1 | ASD-F1 | +0.20 | [-2.53, +2.93] | 5/1/4 | +0.08 | +0.62 |
| G3_minus_G2 | ACC | +0.58 | [-0.27, +1.42] | 5/3/2 | +0.57 | +0.38 |
| G3_minus_G2 | Hard-AUC / BA | +0.63 | [-0.20, +1.45] | 7/1/2 | +0.62 | +0.43 |
| G3_minus_G2 | Prob-AUC | +0.31 | [-0.85, +1.47] | 7/0/3 | -1.32 | +0.21 |
| G3_minus_G2 | SEN-ASD | +1.25 | [-0.87, +3.37] | 6/2/2 | +1.24 | +0.85 |
| G3_minus_G2 | SPE-TD | +0.00 | [-2.04, +2.05] | 3/3/4 | +0.00 | +0.01 |
| G3_minus_G2 | ASD-F1 | +0.66 | [-0.23, +1.55] | 7/1/2 | +0.72 | +0.42 |
| G3_minus_B0 | ACC | -3.90 | [-5.82, -1.99] | 0/1/9 | -3.90 | -3.70 |
| G3_minus_B0 | Hard-AUC / BA | -3.93 | [-5.98, -1.87] | 0/1/9 | -3.94 | -3.73 |
| G3_minus_B0 | Prob-AUC | -2.71 | [-4.83, -0.58] | 0/0/10 | -3.81 | -2.64 |
| G3_minus_B0 | SEN-ASD | -4.43 | [-9.09, +0.22] | 1/3/6 | -4.47 | -4.38 |
| G3_minus_B0 | SPE-TD | -3.42 | [-5.61, -1.24] | 1/1/8 | -3.42 | -3.08 |
| G3_minus_B0 | ASD-F1 | -4.31 | [-6.62, -2.00] | 0/1/9 | -4.31 | -4.12 |

## Complementarity and Interventions

The correction counts compare independent predictors after the fact; they are potential complementary evidence, not an actual fused correction. Dynamic-Off, temporal shuffle, and prompt permutation are read-only interventions on each selected checkpoint.

### Error Changes

| Contrast | Class | Wrong→Right | Right→Wrong | Both Wrong | Both Right | Net | Excluding Fold 2 W→R / R→W |
|---|---|---:|---:|---:|---:|---:|---:|
| B0_to_G2 | ASD | 11 | 34 | 29 | 329 | -23 | 11 / 30 |
| B0_to_G2 | TD | 7 | 23 | 21 | 417 | -16 | 7 / 20 |
| B0_to_G2 | ALL | 18 | 57 | 50 | 746 | -39 | 18 / 50 |
| B0_to_G3 | ASD | 11 | 29 | 29 | 334 | -18 | 10 / 26 |
| B0_to_G3 | TD | 9 | 25 | 19 | 415 | -16 | 9 / 22 |
| B0_to_G3 | ALL | 20 | 54 | 48 | 749 | -34 | 19 / 48 |
| G2_to_G3 | ASD | 15 | 10 | 48 | 330 | 5 | 13 / 10 |
| G2_to_G3 | TD | 12 | 12 | 32 | 412 | 0 | 11 / 11 |
| G2_to_G3 | ALL | 27 | 22 | 80 | 742 | 5 | 24 / 21 |

### Intervention Summary

| Intervention | Checkpoints | Mean abs ASD-probability change | Hard flips | Mean ACC Δ pp | Mean Prob-AUC Δ pp |
|---|---:|---:|---:|---:|---:|
| dynamic_off | 30 | 0.18254 | 703 | -18.10 | -9.40 |
| medical_prompt_permutation | 10 | 0.00631 | 12 | -0.69 | -0.25 |
| temporal_shuffle | 20 | 0.01430 | 42 | -0.81 | -0.43 |

## Mechanism and Confounding

The independent audit restored and recomputed all 30 Best Checkpoints and replayed one update twice from each Latest Checkpoint. The saved Latest parameters, optimizer states, private/shared RNG and repeated next-step results match bitwise for all 30 tasks (maximum parameter and optimizer drift: 0). The first replay helper was rejected after it was found to reuse a mutable AdamW state snapshot; `audit_backend.py` deep-copies each saved state before both replays. The source-level dynamic branch receives only the 6th prompt slot (weight 1.0) and 8th prompt slot (weight 0.25); the other six slots are masked from dynamic evidence. The exact medical texts, anonymous texts, CLIP cache, and embedding hashes are stored in `config.json` and `.runtime/prompt_cache.pt`.

Analysis used `analyze_backend.py` to apply two in-memory compatibility fixes to the frozen `analyze.py` (include the G1 representation buffer and cast Boolean site-accuracy indicators before subtraction). The original analysis/training sources and their recorded SHA remain unchanged; the wrapper SHA is included in `manifest.json`.

Fold-level gate, residual, attention entropy, class-gradient and RNG trace diagnostics are in `mechanism.csv`; per-fold Dynamic-Off and temporal/prompt interventions are in `interventions.csv`.

| Diagnostic | Value |
|---|---:|
| G2 dynamic evidence site OOF accuracy / balanced accuracy / macro-F1 | 0.083 / 0.084 / 0.076 |
| G3 dynamic evidence site OOF accuracy / balanced accuracy / macro-F1 | 0.065 / 0.067 / 0.060 |
| Spearman(dynamic residual RMS, timepoints) (n=871) | 0.661 (descriptive p=1.18e-110) |
| Spearman(dynamic residual RMS, TR_seconds) (n=871) | -0.405 (descriptive p=1.06e-35) |
| Spearman(dynamic residual RMS, scan_duration_seconds) (n=871) | 0.532 (descriptive p=8.57e-65) |
| Spearman(dynamic residual RMS, window_count) (n=871) | 0.499 (descriptive p=3.92e-56) |
| Spearman(dynamic residual RMS, within_subject_residual_rms) (n=871) | 1.000 (descriptive p=0) |
| Spearman(dynamic residual RMS, adjacent_window_change) (n=871) | 0.802 (descriptive p=5.88e-197) |
| Spearman(dynamic residual RMS, func_mean_fd) (n=871) | 0.271 (descriptive p=3.52e-16) |
| Spearman(dynamic residual RMS, func_dvars) (n=871) | -0.034 (descriptive p=0.31) |
| Spearman(dynamic residual RMS, func_perc_fd) (n=871) | 0.272 (descriptive p=3.22e-16) |
| Spearman(dynamic residual RMS, func_outlier) (n=871) | 0.041 (descriptive p=0.227) |
| Spearman(dynamic residual RMS, func_num_fd) (n=871) | 0.359 (descriptive p=7.34e-28) |

Site-wise counts and OOF accuracy changes are in `site_analysis.csv`; motion metadata coverage and phenotype/site consistency are recorded there as well. Correlations are descriptive and unadjusted for multiple comparisons.

## Training and Identity

The actual device was `cuda:0` / `PPU-ZW810E`; eight E80.1 workers ran concurrently. The formal dispatch took 1,487 seconds (24m 47s) from first worker start through final task completion; preflight through completed analysis took about 54m 30s. Each group has 958,900 trainable DARG/CAM parameters plus 256,144 PSDC parameters (1,215,044 total), with identical group initialization and optimizer schemas within every fold. The 30 tasks executed 10,197 epochs in total. Their records store the exact E61 epoch-10 warmup SHA, split/pair identities, initialization hashes, optimizer identity, private RNG state, common RNG trace, and Best/Latest Checkpoint SHA. All 9,686 overlapping E74.4-G2 RNG triplets match, with zero mismatches. E80.1 reached the historical E74.4 Best Epoch in 29/30 tasks; epoch alignment is auxiliary and does not replace the configured test-ACC checkpoint selection.

## Limits and Decision

No dynamic-only score is used as a gate. The E80 result did not establish an increment for the V+D/PCA32 probe; E80.1 tests only this fixed 90-second, 252-edge, two-layer TCN design. E80.1 reached 100% dynamic coverage (871/871), 3–13 windows per participant (mean 7.26), and 90.00–91.08-second physical windows. Static FC reconstruction matched all 222,976 values with maximum absolute error 3.997e-15. Window shrinkage ranged 0.036–0.430 (median 0.156; none >=0.99). The relative residual reached the configured 10% CAM144 cap for about 98% of test subjects, and Dynamic-Off changed 703 hard predictions over 30 checkpoints; this shows the learned residual path affects outputs, but does not isolate real temporal information because G1 also has a trainable static-mean path. Temporal shuffling changed 42 predictions across 20 G2/G3 checkpoints, with mean ACC changes of -0.69 pp for G2 and -0.92 pp for G3. Medical-prompt slot permutation changed 12 predictions across 10 G3 checkpoints (mean ACC -0.69 pp; mean Prob-AUC -0.25 pp). These effects are small relative to the G3 deficit versus B0.

Dynamic residual RMS correlated with scan timepoints (Spearman rho=0.661), duration (0.532), window count (0.499), mean FD (0.271), and percent FD (0.272); DVARS correlation was -0.034. A 20-class OOF site probe achieved 8.3% accuracy / 8.4% balanced accuracy for G2 dynamic evidence and 6.5% / 6.7% for G3; site-level changes were mixed. All 871 participants had finite listed FD/DVARS phenotype measures and site identity matched the locked BOLD map. The `.1D` files do not contain framewise censor/delete timestamps, so physical temporal gaps remain unaudited.

G2−G1 has a small pooled OOF Prob-AUC gain (+1.27 pp), but pooled ACC/BA gains are only +0.23/+0.14 pp and fold-level ACC/BA intervals cross zero. G3−G2 has pooled ACC/BA gains of +0.57/+0.62 pp, but pooled Prob-AUC falls by 1.32 pp and the fold-level intervals cross zero. G3 remains below B0 by 3.90 pp ACC, 3.94 pp BA and 3.81 pp pooled Prob-AUC; excluding Fold 2, its mean ACC/BA/Prob-AUC deficits are 3.70/3.73/2.64 pp. B0-to-G3 has 20 wrong-to-right cases and 54 right-to-wrong cases (net -34), including 11/29 ASD and 9/25 TD corrections/regressions. Therefore the predeclared full-model gate fails: `NO_GO_PSDC_END_TO_END`, and Stage B is not recommended. This result is specific to this fixed window/TCN/prompt design; it neither proves all BOLD dynamic methods ineffective nor identifies an ASD biomarker.
