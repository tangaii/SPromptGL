# Experiments 62–66: ChatGPT Brief

## Scope and protocol

Experiments 62–66 study initialization and optimization sensitivity of the E61-E2 end-to-end adaptive CADR graph propagation model. The read-only reference is E62-E2: edge gates initialized at 0.5 for all four modalities; fusion gates initialized to `[0.001, 0.001, 0.001, 0.99]` in `[PHENO, ANAT, FUNC, CORR]` order. E62–E65 use seed 0 and the fixed ten-fold split. E66 tests requested RNG seeds 1, 2, 3, 2020, and 2026.

The experiments retain the historical `val=test` best-checkpoint selection, preprocessing before cross-validation, and transductive 871-subject graph. Treat all comparisons as exploratory and protocol-matched, not as unbiased external generalization estimates. Fold-level differences are paired by the same split; folds have overlapping training data.

## Formal groups

- **Experiment 62, gate initialization:** E62-E1 uses edge gates 0.5 and fusion gates `[0.01,0.01,0.01,0.99]`; E62-E2 is the reference; E62-E3 changes all edge gates to 0.4; E62-E4 changes them to 0.6.
- **Experiment 63, CORR gate refinement:** E63-G1 uses CORR fusion 0.999 with E62-E2's other settings; G2 and G3 also set all edge gates to 0.45 and 0.55, respectively.
- **Experiment 64, CORR fusion refinement:** E64-G1/G2/G3 set CORR fusion to 0.995/0.992/0.993; other E62-E2 settings remain fixed.
- **Experiment 65, weight decay:** E65-G1 sets CADR AdamW WD to 0.1; G2 sets it to 0.001; G3 sets GFR Adam WD to 0.2.
- **Experiment 66, RNG robustness:** five E62-E2 runs use requested seeds 1, 2, 3, 2020, and 2026. Each fold retains identical initial parameter, split, pair-cache, and anchor hashes. The requested seed offsets E61's per-fold construction seed `510000 + fold*100 + 1`; it changes the Python/NumPy/PyTorch RNG stream. CADR relation-module initialization remains fixed. Each seed replays its own ten-epoch warmup; no per-epoch reseeding or best-checkpoint fine-tuning is used. This tests training-RNG sensitivity under fixed initial weights, not random weight initialization variance.

## Results

Ten-fold means in percent. Columns: ACC / Hard-AUC / Probability-AUC / SPE-TD / SEN-ASD.

| Group | ACC | Hard-AUC | Probability-AUC | SPE-TD | SEN-ASD |
|---|---:|---:|---:|---:|---:|
| E62-E1 | 91.7294 | 91.4804 | 93.2417 | 94.6253 | 88.3354 |
| E62-E2 reference | 91.8443 | 91.6821 | 93.2010 | 93.7789 | 89.5854 |
| E62-E3 | 91.3845 | 91.1929 | 93.5756 | 93.7882 | 88.5976 |
| E62-E4 | 91.7294 | 91.5487 | 93.1600 | 93.7743 | 89.3232 |
| E63-G1 | 91.6144 | 91.4214 | 93.3886 | 93.7697 | 89.0732 |
| E63-G2 | 91.2696 | 91.0834 | 93.6322 | 93.5754 | 88.5915 |
| E63-G3 | 91.6144 | 91.4219 | 93.5200 | 93.9963 | 88.8476 |
| E64-G1 | 91.4995 | 91.2645 | 93.5413 | 94.1998 | 88.3293 |
| E64-G2 | 91.2696 | 90.9995 | 93.5018 | 94.6392 | 87.3598 |
| E64-G3 | 91.4995 | 91.2543 | 93.5647 | 94.4172 | 88.0915 |
| E65-G1 | 91.5008 | 91.2651 | 93.5200 | 94.4265 | 88.1037 |
| E65-G2 | 91.7294 | 91.4778 | 93.5832 | 94.8520 | 88.1037 |
| E65-G3 | 91.4995 | 91.2651 | 93.6804 | 94.4265 | 88.1037 |
| E66-S1 | 91.8443 | 91.5625 | 93.4891 | 95.2775 | 87.8476 |
| E66-S2 | 91.7294 | 91.5643 | 93.2326 | 93.5615 | 89.5671 |
| E66-S3 | 91.7294 | 91.5439 | 93.0725 | 93.9963 | 89.0915 |
| E66-S2020 | 91.3845 | 91.1347 | 93.3587 | 94.4218 | 87.8476 |
| E66-S2026 | 91.7294 | 91.5518 | 93.1173 | 93.7743 | 89.3293 |

Reference E62-E2 standard deviations are ACC 2.9346, Hard-AUC 2.8919, Probability-AUC 3.0942, SPE 4.2653, SEN 3.4710. Each experiment's `results/summary.csv` and `results/report.md` contain the full means, population SDs, paired comparisons, and mechanism diagnostics.

## Interpretation

No tested initialization or weight-decay setting gives a consistent improvement over E62-E2 across ACC, AUC, and ASD sensitivity. E65-G2 has the highest SPE among E62–E65 new groups but lower ACC and SEN than E62-E2. In E66, S1 matches E62-E2 ACC to reported precision and improves mean Probability-AUC/SPE while losing 1.7378 percentage points in SEN. S2 has the closest SEN to the reference but slightly lower ACC and Hard-AUC. Seed rankings are descriptive; they do not establish a generally superior seed. None of the E62–E66 groups exceeds the 53.3 third-group historical ACC mean of 92.0742%.

## Completion and files

All five experiments report `COMPLETE_AUDITED`: E62 40/40, E63 30/30, E64 30/30, E65 30/30, and E66 50/50 tasks independently audited. The upload package contains source runners, frozen configs, manifests, CSV results, preflight records, and reports. Runtime folders, checkpoints, Python caches, and logs are excluded.

Experiment directories: `experiment/experiment62_e2e_gate_init_sensitivity`, `experiment/experiment63_corr_gate_init_refinement`, `experiment/experiment64_corr_fusion_gate_sensitivity`, `experiment/experiment65_wd_sensitivity`, and `experiment/experiment66_e62_multiseed`.
