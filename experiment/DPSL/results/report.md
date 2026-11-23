# E82.4.6.10-G1 DPSL formal migration report

## Status

`PARTIALLY_STANDALONE`. The preferred directory
`/data/adodas/SPromptGL-main/experiment/DPSL` returned errno 5 while being
created. The single authoritative directory is
`/model/spromptgl/experiment/DPSL`. No historical experiment file was
modified and no Fold was trained.

The external task identity is `E82.4.6.10-G1`. The serialized model identity
in every selected checkpoint is `E82.4.6-G2 SCMR-DA-SRPE`, based on
`E74.4-G2 DARG`. GFR, DARG, CAM144, the output classifier, disease Prompt,
spectral and Jacobian in/out paths, Gaussian, EMA, classification evidence,
reliability, end-to-end CAM144 enhancement, residual and shared-gradient
limits remain represented by the archived model state and source snapshots.

## Formal ten-fold source mapping

| Fold | Historical source | Target Best | Target Latest |
| --- | --- | --- | --- |
| 1 | E82.4.6-G2 | `checkpoints/fold01.best.pt` | `checkpoints/fold01.latest.pt` |
| 2 | E82.4.6.4-G1 | `checkpoints/fold02.best.pt` | `checkpoints/fold02.latest.pt` |
| 3 | E82.4.6.3-G2 | `checkpoints/fold03.best.pt` | `checkpoints/fold03.latest.pt` |
| 4 | E82.4.6.6-G1 | `checkpoints/fold04.best.pt` | `checkpoints/fold04.latest.pt` |
| 5 | E82.4.6-G2 | `checkpoints/fold05.best.pt` | `checkpoints/fold05.latest.pt` |
| 6 | E82.4.6-G2 | `checkpoints/fold06.best.pt` | `checkpoints/fold06.latest.pt` |
| 7 | E82.4.6-G2 | `checkpoints/fold07.best.pt` | `checkpoints/fold07.latest.pt` |
| 8 | E82.4.6.10-G1 | `checkpoints/fold08.best.pt` | `checkpoints/fold08.latest.pt` |
| 9 | E82.4.6-G2 | `checkpoints/fold09.best.pt` | `checkpoints/fold09.latest.pt` |
| 10 | E82.4.6-G2 | `checkpoints/fold10.best.pt` | `checkpoints/fold10.latest.pt` |

The complete source and target SHA mapping is in `manifest.json` and
`runtime/copy_records.json`. Fold 8 is the formal G1 checkpoint from
`E82.4.6.10-G1`, never the E82.4.6.10-G2 or E82.4.6.9-G2 Fold 8 weight.

## Frozen inputs and restoration state

The local snapshot contains the processed feature table and original order,
labels and encodings, ten train/test and legal-supervision memberships, the
19-patient overlap registration, relation pairs and balanced relation labels,
E61 epoch-10 warmup state and RNG, ten spectrum files, the eight prompt and
text-embedding resources, and the selected runtime closure. Each Best and
Latest payload retains model, native optimizer, EMA/Gaussian, RNG, epoch,
early-stop, evaluation, membership, relation and journal state. The ten
warmup files are byte-identical to their recorded historical sources.

`cli.py` is independent of the working directory and resolves all runtime
paths from its own file. `inspect`, `evaluate`, `audit`, `resume-check`, and
`train-resume` are available. The public CLI imports only local
`code/dpsl_runtime.py`; an external working-directory audit recorded no read
of an adjacent historical experiment directory.

## Exact replay and audit outcome

Best checkpoint SHA verification passed 10/10. Best payload loading and the
serialized evaluation replay passed 10/10. The replay contains all 871
historical test rows in their original fold/test order. ASD probabilities,
classification logits, and hard predictions stored in the Best evaluation
payload match the frozen patient table exactly, with maximum probability error
zero. `results/evaluation_replay.csv` contains the patient-level logits,
probabilities, labels and predictions.

Latest checkpoint loading, optimizer/RNG/EMA presence, epoch and journal
identity passed 10/10. The `resume-check` command intentionally performs only
an in-memory serialized recovery validation. It does not advance an optimizer,
consume a training RNG stream, or write a checkpoint.

The 19 historical train-test overlaps remain explicit. Fold 2 has
377, 407, 681, 779, 784, 260, 288 and 375. Fold 3 has 46, 66, 172, 266 and
388. Fold 4 has 712, 713, 727 and 773. Fold 8 has 383 and 430. They retain
DARG classification and relation eligibility, stay outside the added spectral,
Prompt, Gaussian and EMA label supervision, and have zero added residuals in
the frozen patient audit. These historical results are not an independent
generalization estimate.

The two remaining audit blockers are deliberate. The public standalone entry
has not executed a new full historical model forward for all ten folds, and it
has not executed a Latest next-step optimizer/training/evaluation replay.
Consequently the package is not marked `COMPLETE_AUDITED_STANDALONE`.

## Formal ten-fold metrics

All values are percentages and use population SD across the ten folds.

| Metric | Mean ± population SD |
| --- | ---: |
| ACC | 93.1087 ± 2.6257 |
| Hard-AUC / BA | 93.0348 ± 2.6450 |
| Prob-AUC | 94.8628 ± 2.8903 |
| SEN-ASD | 92.0549 ± 3.6745 |
| SPE-TD | 94.0148 ± 3.4314 |
| ASD-F1 | 92.5096 ± 2.8797 |

The complete per-fold values are in `results/fold_metrics.csv`. The formal
871-row historical table is in `results/patient_level.csv`, and the exact
serialized prediction replay is in `results/evaluation_replay.csv`.

## Minimum formal files

`runtime/required_sha256.json` records the final SHA256 of the formal code,
configuration, manifest, input snapshots, warmups, ten Best and ten Latest
checkpoints, reports and audit outputs. `code/provenance` contains read-only
historical manifests, source result snapshots, source implementation snapshots
and the paper provenance copy. The provenance paths in those records are
metadata only and are not runtime imports or data sources.
