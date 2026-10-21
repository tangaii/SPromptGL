# Experiment 33: QCAR Conditional Cross-Modal Alignment

Status: **COMPLETE**, 80/80 folds. GPU 0 completed 40 folds and GPU 1 completed 40 folds. Wall time by worker and exact fold assignment are in `manifest.json`.

## Audit and Protocol

The MICCAI 2025 SPromptGL paper describes multi-relation GFR and CAM/PEN (SPGS+RES). Its Figure 2 reports a large contribution from GFR; the local source is the authority for this reproduction because paper equations and current `simple-2` code differ in the label-graph loss and graph-fusion details. Current Experiment 32 B0-A is reproduced with the existing PLN constructor, including the inactive MLP, CLIP buffer and avg_pool, while the forward path bypasses PEN and passes CAM flatten directly to the unchanged OutputLayer.

Experiment 18/23/24/25/27/28/29/30/31 repeatedly found that semantic/QC auxiliary alignment or graph calibration did not reliably improve the frozen baseline or its controls. This experiment therefore tests a label-free, outer-train-only conditional QC-FC estimate before GFR and includes site-only, unconditional, reliability-off, wrong-pair, FC-only and reverse-sign controls. CORR256 is the only modified input. ANAT6/FUNC10 are QC proxies rather than anatomy/activation truth; the estimated FC component is a conditional association, not a verified artifact.

Data are the existing 871×320 standardized/RFE-selected ABIDE table, without reprocessing. PHENO columns were verified in `modal_feat_dict.npy`: age-bin [0:26], site [26:46], sex [46:48]; Q=ANAT6+FUNC10 [48:64]; F=CORR256 [64:320]. Splits are `StratifiedKFold(10, shuffle=True, random_state=10)`, labels ASD=0/TD=1, and full 871-subject transductive GFR remains unchanged.

The original seed-0 simple-2 protocol is retained: `ClsLoss + mean(GraphConstructLoss over four input modalities)`, both original Adam optimizers, 450 epoch cap, patience 200, `val_idx=test_idx`, and a checkpoint is replaced only on strict ACC improvement. `ALoss` is still calculated but excluded from the loss. The four reported classification metrics are ACC, one-hot hard-label ROC-AUC, SEN-ASD and SPE-TD. No historical checkpoint was loaded.

Each of the eight independent workers initializes seed=0 once, then runs folds 1–10 in order with one continuous Python/NumPy/PyTorch CPU/CUDA RNG stream. Four workers run per physical GPU. QCAR statistics use local deterministic NumPy generators and do not consume the model RNG.

For each fold, standardized float64 ridge fits use only outer-train rows. C→Q and C→F residualization precede QC-residual→FC-residual prediction. Reliability compares three-fold outer-train OOF C→F and C+Q→F errors, with `gain=(MSE_C−MSE_C+Q)/(MSE_C+1e-12)` and `w=clip(max(gain,0),0,1)`. Site-wise wrong pairing uses a deterministic within-site permutation. A single gamma starts at zero, uses straight-through gradients with forward projection into [0,.25], and each patient's correction is capped at 15% of its CORR norm. All four non-FC modalities are copied unchanged.

## Four-Metric Results

Mean ± population SD over ten folds, in percent. The historical Experiment 32 B0-A row is read-only and is not combined with the new folds.

| Group | ACC | Hard-AUC | SEN-ASD | SPE-TD |
|---|---:|---:|---:|---:|
| G0_B0A | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 89.0793 ± 3.7283% | 94.6300 ± 4.0457% |
| G1_SITE_ONLY | 91.2709 ± 3.0992% | 91.0740 ± 3.1637% | 88.3598 ± 4.6579% | 93.7882 ± 2.8197% |
| G2_UNCONDITIONAL | 91.3845 ± 3.2677% | 91.2436 ± 3.2079% | 89.5732 ± 4.0032% | 92.9140 ± 5.3294% |
| G3_NO_RELIABILITY | 91.3845 ± 3.1017% | 91.2038 ± 3.0990% | 88.8415 ± 4.2932% | 93.5661 ± 4.1156% |
| G4_QCAR_FULL | 91.4995 ± 3.0130% | 91.3365 ± 2.9863% | 89.0976 ± 3.4981% | 93.5754 ± 3.8430% |
| G5_WRONG_PAIR | 91.6144 ± 3.0498% | 91.4087 ± 3.0473% | 88.6037 ± 3.4759% | 94.2137 ± 3.2156% |
| G6_FC_ONLY | 92.1904 ± 2.5132% | 92.0936 ± 2.5156% | 90.8293 ± 3.6819% | 93.3580 ± 3.5440% |
| G7_REVERSE_SIGN | 91.3845 ± 3.2677% | 91.1691 ± 3.2314% | 88.5732 ± 4.5190% | 93.7650 ± 5.2773% |
| Historical Experiment 32 B0-A (read-only) | 92.0742 ± 2.6000% | 91.8546 ± 2.5815% | 89.0793 ± 3.7283% | 94.6300 ± 4.0457% |

## Paired Comparisons

Each difference is G4 minus the control in percentage points, paired by fold. Win/tie/loss counts are descriptive; these overlapping folds are not independent cohorts.

| Comparison | Metric | Mean Δ (pp) | Fold W/T/L |
|---|---|---:|---:|
| G4 − G0_B0A | ACC | -0.5747 | 2/3/5 |
| G4 − G0_B0A | Hard-AUC | -0.5181 | 2/3/5 |
| G4 − G0_B0A | SEN-ASD | +0.0183 | 2/6/2 |
| G4 − G0_B0A | SPE-TD | -1.0546 | 2/4/4 |
| G4 − G1_SITE_ONLY | ACC | +0.2286 | 4/4/2 |
| G4 − G1_SITE_ONLY | Hard-AUC | +0.2625 | 6/1/3 |
| G4 − G1_SITE_ONLY | SEN-ASD | +0.7378 | 4/5/1 |
| G4 − G1_SITE_ONLY | SPE-TD | -0.2128 | 3/3/4 |
| G4 − G2_UNCONDITIONAL | ACC | +0.1149 | 3/5/2 |
| G4 − G2_UNCONDITIONAL | Hard-AUC | +0.0929 | 3/3/4 |
| G4 − G2_UNCONDITIONAL | SEN-ASD | -0.4756 | 2/5/3 |
| G4 − G2_UNCONDITIONAL | SPE-TD | +0.6614 | 3/4/3 |
| G4 − G3_NO_RELIABILITY | ACC | +0.1149 | 2/6/2 |
| G4 − G3_NO_RELIABILITY | Hard-AUC | +0.1327 | 3/4/3 |
| G4 − G3_NO_RELIABILITY | SEN-ASD | +0.2561 | 4/4/2 |
| G4 − G3_NO_RELIABILITY | SPE-TD | +0.0093 | 2/6/2 |
| G4 − G5_WRONG_PAIR | ACC | -0.1149 | 1/8/1 |
| G4 − G5_WRONG_PAIR | Hard-AUC | -0.0722 | 3/5/2 |
| G4 − G5_WRONG_PAIR | SEN-ASD | +0.4939 | 3/6/1 |
| G4 − G5_WRONG_PAIR | SPE-TD | -0.6383 | 1/6/3 |
| G4 − G6_FC_ONLY | ACC | -0.6910 | 1/2/7 |
| G4 − G6_FC_ONLY | Hard-AUC | -0.7572 | 1/2/7 |
| G4 − G6_FC_ONLY | SEN-ASD | -1.7317 | 0/6/4 |
| G4 − G6_FC_ONLY | SPE-TD | +0.2174 | 3/4/3 |
| G4 − G7_REVERSE_SIGN | ACC | +0.1149 | 3/5/2 |
| G4 − G7_REVERSE_SIGN | Hard-AUC | +0.1674 | 4/3/3 |
| G4 − G7_REVERSE_SIGN | SEN-ASD | +0.5244 | 4/3/3 |
| G4 − G7_REVERSE_SIGN | SPE-TD | -0.1896 | 2/5/3 |

Current G0 ACC is 92.0742 ± 2.6000%; historical B0-A ACC was 92.0742 ± 2.6000%. G0 matched historical initial trunk hashes on 10/10 folds, initial RNG hashes on 10/10, best epochs on 10/10, and all four metrics on 10/10. Fold-level values are in `fold_metrics.csv`; no historical checkpoint was loaded.

### B0-A Fold-by-Fold Comparison

Percent values are shown; deltas are new G0 minus the historical Experiment 32 B0-A row for the same fold.

| Fold | E32 ACC | New G0 ACC | Δ ACC (pp) | E32 Hard-AUC | New G0 Hard-AUC | Δ AUC (pp) | E32 SEN | New G0 SEN | Δ SEN (pp) | E32 SPE | New G0 SPE | Δ SPE (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.4545 | 95.4545 | +0.0000 | 95.1220 | 95.1220 | +0.0000 | 90.2439 | 90.2439 | +0.0000 | 100.0000 | 100.0000 | +0.0000 |
| 2 | 88.5057 | 88.5057 | +0.0000 | 88.3351 | 88.3351 | +0.0000 | 85.3659 | 85.3659 | +0.0000 | 91.3043 | 91.3043 | +0.0000 |
| 3 | 88.5057 | 88.5057 | +0.0000 | 88.7328 | 88.7328 | +0.0000 | 92.6829 | 92.6829 | +0.0000 | 84.7826 | 84.7826 | +0.0000 |
| 4 | 94.2529 | 94.2529 | +0.0000 | 93.9362 | 93.9362 | +0.0000 | 90.0000 | 90.0000 | +0.0000 | 97.8723 | 97.8723 | +0.0000 |
| 5 | 94.2529 | 94.2529 | +0.0000 | 94.1223 | 94.1223 | +0.0000 | 92.5000 | 92.5000 | +0.0000 | 95.7447 | 95.7447 | +0.0000 |
| 6 | 95.4023 | 95.4023 | +0.0000 | 95.3723 | 95.3723 | +0.0000 | 95.0000 | 95.0000 | +0.0000 | 95.7447 | 95.7447 | +0.0000 |
| 7 | 90.8046 | 90.8046 | +0.0000 | 90.1862 | 90.1862 | +0.0000 | 82.5000 | 82.5000 | +0.0000 | 97.8723 | 97.8723 | +0.0000 |
| 8 | 89.6552 | 89.6552 | +0.0000 | 89.3085 | 89.3085 | +0.0000 | 85.0000 | 85.0000 | +0.0000 | 93.6170 | 93.6170 | +0.0000 |
| 9 | 90.8046 | 90.8046 | +0.0000 | 90.5585 | 90.5585 | +0.0000 | 87.5000 | 87.5000 | +0.0000 | 93.6170 | 93.6170 | +0.0000 |
| 10 | 93.1034 | 93.1034 | +0.0000 | 92.8723 | 92.8723 | +0.0000 | 90.0000 | 90.0000 | +0.0000 | 95.7447 | 95.7447 | +0.0000 |

### G4 Versus Historical Experiment 32 B0-A

Paired fold differences compare the new G4 run against Experiment 32 B0-A from the archived historical CSV; the historical weights/checkpoints were not loaded.

| Metric | Mean Δ (pp) | Fold W/T/L |
|---|---:|---:|
| ACC | -0.5747 | 2/3/5 |
| Hard-AUC | -0.5181 | 2/3/5 |
| SEN-ASD | +0.0183 | 2/6/2 |
| SPE-TD | -1.0546 | 2/4/4 |

## Mechanism and Decision

Fold-level OOF gains, reliability distributions, fitted-statistics hashes, wrong-pair rate, gamma, relative correction size and high/low QC-residual correction summaries are in `diagnostics.csv`. G4 starts each fold at gamma=0; final gamma and nonzero correction counts are reported per fold. A small or zero gamma is not evidence that QCAR worked. OOF reliability estimates prediction improvement only and does not identify causal acquisition artifacts.

In G4, the outer-train OOF mean gain was positive on only 3/10 folds; the mean across all fold/connection gain values was −0.0607. The mean number of nonzero reliability weights was 96.2/256 per fold (range 54–150), with mean reliability 0.0145. Gamma reached 0.25 in all ten folds, but the mean correction was only 0.127% of FC norm (range 0.088–0.185%); it was 0.233% in the train-defined high-QC-residual group versus 0.100% in the remaining group. Within-site wrong pairing affected 94.4–96.3% of subjects. Despite this intervention, G4 trailed G5 by 0.1149 pp ACC and 0.0722 pp Hard-AUC, and trailed the FC-only G6 by 0.6910 pp ACC and 0.7572 pp Hard-AUC. G4 exceeded the no-reliability G3 by only 0.1149 pp ACC, below the frozen +0.23 pp target. SEN-ASD was effectively unchanged from G0 (+0.0183 pp), while SPE-TD fell by 1.0546 pp. These results do not support a reliable QC-conditioned mechanism or an advantage over generic FC correction.

The eight worker processes consumed 4,066.8 worker-seconds in aggregate: GPU 0 handled 40 folds and 2,102.1 worker-seconds; GPU 1 handled 40 folds and 1,964.7 worker-seconds. End-to-end elapsed time from the first worker start through completion was 921.3 seconds, including checkpoint-safe recovery after startup/diagnostic-code failures; the final resumed batch ran from 15:07:03Z to 15:15:36Z (about 513 seconds).

Frozen GO conditions: G4 must exceed current G0 on ACC and hard-AUC with at least +0.23 pp ACC; exceed G5 on ACC and hard-AUC; not be weaker than G6 on ACC and hard-AUC; retain SEN-ASD within 0.50 pp of G0; and show nonzero correction plus positive OOF mean gain. These are internal seed-0 candidate criteria, not external clinical validation.

**Decision: NO-GO.** Failed conditions: ACC_and_hard_AUC_above_G0, ACC_gain_at_least_0.23pp, G4_above_G5_on_ACC_and_hard_AUC, G4_not_weaker_than_G6_on_ACC_and_hard_AUC, positive_mean_OOF_gain. No control is renamed as the primary method and no post-hoc parameter search was performed.

## Reproducibility and Limitations

`manifest.json` records source, code, data and historical-reference hashes; configuration, GPU assignment, preflight evidence, RNG/worker state and wall times. Runtime copies of the original sources and the minimal bypass patch are isolated under `.runtime/source/` and `integration.patch`. Completed epoch checkpoints were removed after final validation; per-fold metrics and statistical fit summaries remain.

Historical Experiment 32 B0-A per-fold trunk/RNG hashes, best epochs and metrics were available and matched exactly. Its archived `B0_seed0.pt` was a `BOUNDARY` task record containing fold rows and RNG state, not final model weights; this run therefore did not claim a tensor-by-tensor comparison with an archived best-weight checkpoint. During execution, a duplicate diagnostic key caused seven workers to exit after saving their first-fold epoch state while G0 continued. The run was stopped at an epoch checkpoint; the resume repair changed diagnostic-row merging and avoided one extra epoch when an early-stop checkpoint was already final. Checkpoint signature fields were migrated while model, optimizer, RNG and fold-row payloads were preserved. Final-code preflight was repeated, then all 80 folds completed with no successful fold or epoch retrained.

ANAT/FUNC are imperfect acquisition-quality proxies; PHENO/site adjustment does not establish causal artifact removal. Validation equals test for checkpoint selection, the existing RFE and standardization predate outer folds, and GFR propagates over all 871 participants. These inherited conditions make the results internal historical-protocol comparisons rather than unbiased generalization evidence.

## Fold Metrics

All 80 rows with the four metrics, best epoch, hashes and same-fold G0/Experiment 32 B0-A references are in `fold_metrics.csv`.
