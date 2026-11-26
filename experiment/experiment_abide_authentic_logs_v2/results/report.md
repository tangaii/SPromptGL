# ABIDE Authentic Training Log Reconstruction v2

The three training stages completed on physical GPU 0. Independent metric recalculation evidence is complete for all required Warmup, DARG, and DPSL audit points. E61 source-byte provenance remains unresolved.
Recovered source snapshot: `https://github.com/tangaii/SPromptGL.git` branch `main` commit `7754df1978286a53a8b86a88af748fc444ace933` (Add experiment_abide training logs).

- Warmup: WARMUP_EXACT_REPLAY_PASS, 10/10 folds, epoch-10 endpoint correct 475/871.
- DARG: TRAINED_AUDITED, 10/10 folds, 803/871 correct.
- DPSL: TRAINED_AUDITED_EXACT_FROM_WARMUP, 10/10 folds, 811/871 correct.
- All three author-format logs pass exact row formatting, field ordering, summary formatting, timing, and attention-map checks.
- Ten randomly selected DPSL training epochs, one per fold, were independently recomputed from stored train logits and labels for both train and validation metrics.
- Warmup and DARG per-epoch raw train logits were captured in this closure replay and independently recalculated. DPSL's existing ten raw-logit samples were revalidated.
- `METRIC_RECALCULATION_COMPLETE_AUDITED` is reached. `COMPLETE_AUDITED_THREE_STAGE_LOG_RECONSTRUCTION` remains withheld because the historical E61 run.py source-byte provenance discrepancy is unresolved.

## Logs

- Warmup: `/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/Warmup/result_ABIDE.txt`, 165 lines, 100 epoch rows, SHA-256 `3ad8449480bbc2d552aadaee86ba74a03be7c826cc60aff57ec6ec91d41ccef4`.
- DARG: `/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/DARG/result_ABIDE.txt`, 4082 lines, 4017 epoch rows, SHA-256 `a72c16b9735700891623648b08110d4db5ff5bc0d3b2c6e9eec1b06aa6aac63b`.
- DPSL: `/model/spromptgl/experiment/experiment_abide_authentic_logs_v2/DPSL/result_ABIDE.txt`, 3925 lines, 3860 epoch rows, SHA-256 `e789020e497344c58d3830bd54f7888d4d1bb83e842032b06b3469fe8373573e`.

## Performance

| Stage | Correct | ACC | Hard-AUC/BA | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DARG | 803/871 | 92.1891% +/- 2.7648% | 92.0294% +/- 2.7255% | 93.4946% +/- 2.9195% | 90.0671% +/- 3.8862% | 93.9917% +/- 4.7320% | 91.4590% +/- 2.9013% |
| DPSL | 811/871 | 93.1087% +/- 2.6257% | 93.0348% +/- 2.6450% | 94.8628% +/- 2.8903% | 92.0549% +/- 3.6745% | 94.0148% +/- 3.4314% | 92.5096% +/- 2.8797% |

## Fold Epochs And Time

| Fold | DARG Best/Latest | DPSL Best/Latest | Warmup sec | DARG sec | DPSL sec |
|---:|---:|---:|---:|---:|---:|
| 1 | 123/324 | 102/303 | 5.6658 | 62.8432 | 112.4420 |
| 2 | 171/372 | 166/367 | 5.6619 | 73.8927 | 573.9736 |
| 3 | 183/384 | 244/445 | 1.5626 | 70.9228 | 169.0636 |
| 4 | 216/417 | 182/383 | 1.6628 | 76.2498 | 138.5969 |
| 5 | 295/450 | 428/450 | 1.5435 | 82.6950 | 163.0134 |
| 6 | 339/450 | 157/358 | 1.5482 | 82.9836 | 120.5841 |
| 7 | 148/349 | 166/367 | 1.7185 | 64.0211 | 125.4232 |
| 8 | 263/450 | 201/402 | 1.5477 | 84.1696 | 140.3283 |
| 9 | 186/387 | 231/432 | 1.6266 | 73.0725 | 148.2514 |
| 10 | 233/434 | 152/353 | 1.5368 | 81.4397 | 117.4176 |
- Warmup measured stage time total: 24.0744 sec.
- DARG measured stage time total: 752.2899 sec.
- DPSL measured stage time total: 1809.0941 sec.

## Audit

- E61 endpoint model, CAM, all three optimizer states, RNG, split, pairs, anchor identity, Best epoch/ACC, and structured state SHA match in 10/10 folds. Epoch-by-epoch RNG and six-call Dropout order match. Historical per-epoch model parameter snapshots do not exist, so only endpoint parameters and RNG trajectory are independently comparable.
- The E61 checkpoint `source_sha` metadata records a different `run.py` hash than the recovered audited bytes in all ten folds. No override was used. Current source bytes match the audited source snapshot, and the reproduced initialization and endpoint are exact.
- DARG and DPSL Best/Latest tensors, per-epoch RNG, patient predictions/probabilities, and post-Latest restoration witnesses match their frozen reference archives in 10/10 folds.
- DPSL adapter preflights and complete per-update journals preserve the 19 overlap patients in the DARG path while excluding them from new-module legal supervision and EMA/Gaussian statistics. The original transition enforces zero excluded-row residual and the configured residual bound.
- The frozen historical file inventory passed SHA-256 verification for 1423 files. The original author log SHA-256 is `9eaed7f09d031aa751d9473201bfa25eba4b50992a4e6f723c37064e620f851f`.
- The project worktree contains pre-existing local changes. This run did not clean or overwrite them. Current status contains 15669 porcelain lines with SHA-256 `27252f5419a359e316b04660d078e010fd2cfabda8970bfa4980887719ecafcc`; the protected historical SHA inventory passes.
- One preserved fold-2 DPSL partial attempt lacked author capture and was excluded from results. The formal fold was rerun from epoch 10 and matches the archive. A fold-5 capture attempt stopped before durable commit and was also restarted from epoch 10.
- The fold-2 unlogged checkpoints and journal are preserved under `DPSL/failed_attempts/fold02_unlogged/`.
- The remaining audit limitation is E61 source provenance. The complete metric-recalculation evidence is in `results/metric_recalculation/independent_metric_audit.json`.


## Independent metric recalculation closure

- Status: `METRIC_RECALCULATION_COMPLETE_AUDITED`. Warmup was independently recomputed from 100 captured train/evaluation logits and labels across 10 folds. DARG was deterministically replayed from E61 epoch 10 for 30 frozen audit points, with internal epochs 11, one precommitted intermediate epoch, and historical Best per fold. All 130 points matched independently calculated ACC, Hard-AUC/BA, Prob-AUC, SEN, SPE, and ASD-F1, with the four-decimal author-log fields matching exactly. Prob-AUC comparison allows the documented 1e-5 CPU/GPU exponential rounding tolerance.
- DPSL's existing 10/10 raw-logit evidence was read back and remains `PASS`. No DPSL training or formal model output was changed.
- The independent calculator uses NumPy argmax and scikit-learn `balanced_accuracy_score`, `roc_auc_score`, `recall_score`, and `f1_score`. It does not call the training `classification_metrics()` implementation.
- Warmup and DARG capture archives are under `results/metric_recalculation/`. The DARG sampling plan SHA-256 is `cdc90180708714150da85fce0c94e760478ec5f51b76a820eb5f57eccfb3598b`.
- Reference commit `233ddcd16a9bca24c22ef2e08ebbbc29273f172d` contains the packaged vendor E61 source at `experiment/DPSL/code/vendor/repo/experiment/experiment61_cadr_postalign_vs_e2e/run.py`, SHA-256 `f67b8b11fbe0619fa4a942fa84ce04dca2ae8dac97168e43e7c9087b95b9b0c5`. Its differences are path/runtime adapter constants and imports, so it does not explain the historical checkpoint hash.
- E61 source provenance remains `SOURCE_PROVENANCE_DISCREPANCY_UNRESOLVED`. `run.py` source SHA recorded in historical checkpoints is `77b0ba096a46df53f69c6b570f1dbbd6f6907f7ada3b2a67b4092f3e0750412b`. The recovered `run.py` is `f5bd8236b6a5f66903f56a01832caab6c21177238f24676562f536df3af2b334` and is byte-identical to GitHub commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. No source override or checkpoint modification was used.
- The project worktree contains pre-existing local changes. This run did not clean or overwrite them. Current `git status --porcelain=v1 --untracked-files=all` has 15669 lines with SHA-256 `27252f5419a359e316b04660d078e010fd2cfabda8970bfa4980887719ecafcc`. The protected historical SHA inventory remains PASS for 1423 files.
- Overall status: `INCOMPLETE_SOURCE_PROVENANCE`. The metric-recalculation evidence is complete, but the requested full three-stage status remains withheld because historical E61 `run.py` byte provenance is unresolved.
