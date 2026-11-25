# Experiment 83.6 Final Report

## A. STATUS

`COMPLETE_AUDITED` for the registered execution. The selected formal directory is `/model/spromptgl/experiment/experiment83_6_tadpole_crossrepo_factorial`.

The run contains 24 groups, 50 historical folds, and 190 new folds. All groups have ten folds. The final audit is in `audits/final_audit.json`.

## B. GPU and Environment

The formal run used one `PPU-ZW810E` GPU and seed 0. The eight-worker resource probe passed with eight simultaneous workers. Main groups used the 450-epoch limit, `wait > 200`, `val_idx == test_idx`, strict first test-ACC Best promotion, and the locked ten-fold split. S0 used its independent 300-epoch protocol.

## C. Source Checks

The author source was audited from commit `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. The checked files include `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, and `result_Tadpole.txt`. The cross-repository snapshot is `tangaii/sp` commit `f4534f8`, with `exp/tadpole_repair`, `experiment_t6m_graph_ablation`, and `experiment_t6m_cbsr` source copies sealed locally.

The E83.5 reference is recorded as project commit `4649c5dddf166a4aa957c16c2f667a5cea485cdc` and sealed runtime commit `5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11`. Checkpoints are inherited from the server E83.5 directory. No GitHub checkpoint is claimed. The paper reference is [SPromptGL MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf).

The fidelity matrix labels original behavior, TADPOLE adaptation, and E83.6 experimental changes. It preserves the historical second-modality overwrite, omitted historical ALoss, RES inside PEN, physical No-PEN removal of RES and SPGS, G3/G4 weight decay, internal attention dropout, and optimizer identities.

## D. Data and Feature Order

The main data SHA is `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`. The split SHA is `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`. The cohort has 598 patients with NC/MCI/AD counts `209/315/74`, six modalities of `145/148/3/36/24/9`, and 365 input features.

S0 retained its own prepared data order and preprocessing. Its feature order differs from the main order in 359 of 365 positions, while the locked split indices match. Main RFE, mean imputation, and pre-CV scaling were retained.

## E. Historical H1-H5

H1-H5 used the server's formal E83.5 Best and Latest checkpoints. Each of the 50 Best checkpoints and 50 Latest checkpoints passed independent replay, including the next-step determinism check. H5 was verified as physical CAM-only with no PEN, GFR, DARG, DPSL, graph learning, message passing, prompt, or graph-loss module.

## F. MMGPL12 Prompt Adaptation

The twelve Figure 6 concepts were preserved in order, with `NC:`, `MCI:`, and `AD:` prefixes. MCI uses the four LMCI concepts as the registered semantic proxy. Official RN50 weights SHA `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`. The prefixed RN50 tensor SHA is `0aca103a031ae8971a07109603ac52e2a11633cadc675d1380d46f2d907c9d0b`.

The old-to-prefixed prompt cosine mean is `0.929146` and the minimum is `0.802011`. Class centroid cosines are NC `0.961049`, MCI `0.865258`, and AD `0.978521`. The lower MCI similarity is consistent with the LMCI proxy limitation. G1 PEN weights and G4 DPSL query semantics were measured from saved Best checkpoints.

## G. Graph Supervision

The registered `pair_mean` loss uses only outer-training patient pairs, six continuous pre-threshold cosine score matrices, and coefficient 1.0. It does not use test labels or historical masked target adjacency. The measured six-modality pair MSE and minimum GraphLearn gradient norm are

| Group | Pair MSE | Minimum GraphLearn gradient norm |
|---|---:|---:|
| A2 | 0.267126 | 0.017553 |
| B1 | 0.269580 | 0.014013 |
| C1 | 0.312430 | 0.077295 |

The audit confirms finite values, differentiable scores, train-only targets, unchanged threshold propagation, and preserved second-modality overwrite. Message norms and per-modality edge densities are recorded in `audits/mechanism_summary.json`.

## H. RES-only

RES-only is the original parameter-free local/global relation computation with the original token organization, average pooling, local/global interaction, residual addition, and feature width. SPGS, CLIP, disease prompts, and text MLP are absent. Parameter count is zero. RES input-output relative norm changes are B2 `0.173257`, C2 `0.110801`, and D4 `0.109243`. Turning RES off changes the saved model output in all three checks. Forward and backward checks passed.

## I. DARG and DPSL Warmup Pairing

D2, D3, D4, D5, D6, and D7 passed all 60 fold-level warmup comparisons for shared state, optimizer state, RNG, pair indices, labels, first ten losses, and post-warmup state. D1 is `PARTIAL_MATCH` on all ten folds because the archived E83.5 H3 warmup lacks initial model, optimizer, and RNG hashes. Its pair, labels, history, and post-warmup state matched.

D5 and D6 had dependency-start refusals when first launched before C3 warmup artifacts were available. The affected folds were safely rerun after C3 completed. Final fold artifacts are unique and committed. No G4 final model was used to initialize a new variant.

## J. Weight Decay and Attention Dropout

H3 and H4 retain GFR/CAM weight decay `0.11`. C4 and D7 change only GFR/CAM weight decay to `0.0036`; DARG and DPSL optimizer contracts remain unchanged. E1 changes only CAM internal EncodeLayer attention dropout from `0.1` to `0`.

The H3/C4 and H4/D7 mechanism checks show much larger GCN parameter and message norms in the low-WD variants, while their trained ACC/AUC fall sharply. Under this registered protocol, `0.11` is consistent with suppressing graph messages, but reducing it destabilized the trained result rather than improving it.

## K. Formal Execution

All 18 main new groups A1-E1 completed 180 folds. S0 completed 10 folds under its own cross-repository protocol. H1-H5 contributed 50 historical replay folds. There are 24 complete group summaries, 240 fold metric rows, 14,352 patient-level OOF rows, and 360 paired contrast rows.

## L. Complete ACC/AUC Results

Values are ten-fold mean plus population standard deviation. `AUC_original` is the hard one-hot macro OVR paper metric. `AUC_probability` is supplementary and cannot replace it.

| Group | OOF correct | ACC | AUC_original | AUC_probability |
|---|---:|---:|---:|---:|
| H1 | 521 | 0.8711 +/- 0.0393 | 0.9065 +/- 0.0305 | 0.9389 +/- 0.0208 |
| H2 | 531 | 0.8878 +/- 0.0396 | 0.9073 +/- 0.0288 | 0.9540 +/- 0.0223 |
| H3 | 553 | 0.9247 +/- 0.0319 | 0.9398 +/- 0.0274 | 0.9555 +/- 0.0301 |
| H4 | 556 | 0.9297 +/- 0.0278 | 0.9410 +/- 0.0261 | 0.9627 +/- 0.0225 |
| H5 | 511 | 0.8544 +/- 0.0645 | 0.8831 +/- 0.0476 | 0.9076 +/- 0.0433 |
| A1 | 525 | 0.8779 +/- 0.0330 | 0.9107 +/- 0.0221 | 0.9423 +/- 0.0200 |
| A2 | 547 | 0.9147 +/- 0.0254 | 0.9327 +/- 0.0204 | 0.9592 +/- 0.0189 |
| A3 | 550 | 0.9197 +/- 0.0279 | 0.9331 +/- 0.0256 | 0.9617 +/- 0.0184 |
| B1 | 549 | 0.9180 +/- 0.0264 | 0.9302 +/- 0.0299 | 0.9655 +/- 0.0168 |
| B2 | 538 | 0.8996 +/- 0.0291 | 0.9239 +/- 0.0241 | 0.9569 +/- 0.0173 |
| B3 | 554 | 0.9264 +/- 0.0249 | 0.9344 +/- 0.0229 | 0.9559 +/- 0.0260 |
| C1 | 550 | 0.9197 +/- 0.0281 | 0.9260 +/- 0.0279 | 0.9564 +/- 0.0237 |
| C2 | 550 | 0.9197 +/- 0.0379 | 0.9349 +/- 0.0295 | 0.9603 +/- 0.0202 |
| C3 | 557 | 0.9314 +/- 0.0264 | 0.9345 +/- 0.0275 | 0.9545 +/- 0.0250 |
| C4 | 518 | 0.8662 +/- 0.0301 | 0.8984 +/- 0.0257 | 0.9253 +/- 0.0303 |
| D1 | 552 | 0.9231 +/- 0.0397 | 0.9385 +/- 0.0314 | 0.9607 +/- 0.0201 |
| D2 | 553 | 0.9247 +/- 0.0251 | 0.9360 +/- 0.0261 | 0.9551 +/- 0.0239 |
| D3 | 552 | 0.9231 +/- 0.0201 | 0.9349 +/- 0.0219 | 0.9545 +/- 0.0256 |
| D4 | 547 | 0.9147 +/- 0.0285 | 0.9236 +/- 0.0286 | 0.9539 +/- 0.0261 |
| D5 | 550 | 0.9197 +/- 0.0298 | 0.9316 +/- 0.0269 | 0.9593 +/- 0.0193 |
| D6 | 552 | 0.9231 +/- 0.0382 | 0.9345 +/- 0.0331 | 0.9576 +/- 0.0251 |
| D7 | 515 | 0.8611 +/- 0.0480 | 0.8914 +/- 0.0297 | 0.9307 +/- 0.0315 |
| E1 | 533 | 0.8911 +/- 0.0450 | 0.9109 +/- 0.0404 | 0.9554 +/- 0.0204 |
| S0 | 549 | 0.9181 +/- 0.0188 | 0.9356 +/- 0.0245 | 0.9659 +/- 0.0189 |

## M. E83.5 Contrasts

Relative to each registered baseline, the largest G1 improvement is A3 with delta ACC `+0.048559` and delta AUC_original `+0.026567`. A2 alone gives `+0.043559` ACC and `+0.026191` AUC_original. B3 improves H2 by `+0.038644` ACC and `+0.027079` AUC_original. C3 improves H3 ACC by `+0.006667`, while its AUC_original change is `-0.005291`.

No D group exceeds H4. The best D-group ACC is D2 at `0.924746`, and the best D-group AUC_original is D1 at `0.938479`. E1 improves H2 by `+0.003333` ACC and `+0.003562` AUC_original. C4 and D7 are substantially below their high-WD baselines.

## N. Paper Contrast

The paper reference is ACC `0.9481` and AUC `0.9559`. No group reaches either paper target on this locked ten-fold evaluation. The best overall ACC is C3 at `0.931412`. The best AUC_original is H4 at `0.940967`. S0's hard AUC is `0.935587`, so its supplementary probability AUC `0.965851` is not used to claim paper-level reproduction.

## O. NC/MCI/AD Patient-level Changes

The following are OOF correction/damage counts against each group's own historical baseline.

| Group | NC corrected/damaged | MCI corrected/damaged | AD corrected/damaged |
|---|---:|---:|---:|
| A3 vs H1 | 19/8 | 29/8 | 1/4 |
| B3 vs H2 | 19/6 | 23/13 | 3/3 |
| C3 vs H3 | 1/4 | 18/7 | 3/7 |
| D2 vs H4 | 4/9 | 18/16 | 4/4 |
| D7 vs H4 | 2/25 | 15/33 | 6/6 |

MCI supplies most of the positive OOF changes in A3, B3, C3, and D2. The low-WD groups introduce many more NC and MCI damages. Complete class-transition matrices for all 19 new comparisons are in `audits/mechanism_summary.json`.

## P. DARG and DPSL Mechanisms

DARG Best checkpoints contain both positive and negative signed edges. Direct-off and disease-neutral interventions changed no test argmax on the audited saved folds, so the measured DARG contribution is a training-time method effect rather than a large direct inference flip. Message norms and signed-edge counts are recorded for H3, H4, C1-C4, D1, D5-D7.

DPSL preserves all three pairs NC-MCI, NC-AD, and MCI-AD, with detached native-probability weights. The shared gradient cap is `0.001` and the residual cap is `0.08`. Direct Native-to-Full corrections average zero for H4 and at most `0.1` patient per fold for the changed D groups, with no measured direct damages. Joint training changes remain distinct from direct inference corrections. DPSL probability-AUC changes are reported in the results table and do not substitute for AUC_original.

## Q. Best/Latest Independent Replay

Historical replay passed 50 Best, 50 Latest, and 50 next-step checks. Main E83.6 replay passed 180 Best, 180 Latest, and 180 next-step checks. S0 replay passed 10 Best, 10 Latest, and 10 next-step checks. RNG restoration, optimizer restoration, prediction recomputation, and patient-level OOF coverage passed.

## R. Source Identity and Fidelity Audit

The pre-training seal is `d8a7c9c48b1fa32413c5dd85d10467a340243102b0d4a3eb375f604e62b51781`. Every sealed training file remains byte-identical. `mechanism_e836.py` and `finalize_e836.py` were added as post-training audit-only scripts and are listed separately in the final audit. No training source was modified after formal training, and no historical directory was modified.

## S. Required Result Files

The directory contains `README.md`, `protocol.json`, `source_manifest.json`, `data_manifest.json`, `prompt_manifest.json`, `code/`, `configs/`, `inputs/`, `groups/`, `results/`, and `audits/`. Required result files are `group_summary.csv`, `fold_metrics.csv`, `oof_predictions.csv`, `paired_contrasts.csv`, and this report. Required audit files include `preflight_start.json`, `method_fidelity_matrix.md`, `checkpoint_replay.json`, `new_checkpoint_replay.json`, `s0_checkpoint_replay.json`, `mechanism_summary.json`, and `final_audit.json`.

## T. Handoff

The E83.6 chapter was appended to `/model/spromptgl/experiment/handoff.md` and read back after writing. The handoff records the selected directory, source and data identities, 24 group status, fold counts, metrics, replay, warmup limitation, mechanism evidence, paper comparison, recovery event, and conclusions.

## U. Final Research Conclusions

1. S0 independently reproduces the sp reference result. It obtains ACC `91.81%` and AUC_original `93.56%` under the sp protocol.
2. A3 gives the largest G1 ACC and AUC_original improvement. Graph supervision alone A2 supplies most of the gain, and the prompt plus graph combination is best.
3. No-PEN base GFR improves with graph supervision. B1 reaches `91.80%` ACC, and B3 reaches `92.64%`.
4. RES-only is effective against H2 and is parameter-free, although RES plus graph supervision is the strongest B variant.
5. Graph supervision is compatible with DARG. C3 is the best C ACC result, while its AUC_original tradeoff is negative versus H3.
6. The `0.11` base weight decay is consistent with weaker graph messages in H3/H4. Lowering it to `0.0036` did not improve either G3 or G4 and produced much worse trained metrics.
7. No G4 change improves H4 in this run. D1 has the best D AUC_original and D2 the best D ACC, both below H4.
8. DARG and DPSL produce real training-path changes and satisfy their structural audits. Direct saved-checkpoint argmax interventions are small, so they must not be conflated with joint-training gains.
9. C3 exceeds E83.5 H4 ACC by `0.1667` percentage points. No new group exceeds H4 AUC_original.
10. The experiment does not reach paper ACC `94.81%` or paper AUC_original `95.59%`.
11. A3, B3, and C3 are the strongest candidates for later controlled work. E1 is a small positive dropout control. C4 and D7 should not be promoted based on these results. D-group changes do not justify replacing H4 without a new registered hypothesis.

These conclusions apply to the fixed historical protocol, which includes `val=test` selection, pre-CV full-data RFE, and comparison of multiple registered variants. They are method-exploration results on the common TADPOLE ten-fold split.
