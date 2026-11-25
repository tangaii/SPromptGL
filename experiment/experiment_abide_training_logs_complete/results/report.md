# ABIDE training log reconstruction audit

Status: `INCOMPLETE_AUDITED_STAGE_LINKAGE`

E61 Warmup replay stopped before the first optimizer update because isolated `build_state()` produced an `initial_rng_sha` different from the historical fold identity. The current workspace lacks the original E53/E43.1 source tree required by that initialization chain. Epoch-10 checkpoints were read for identity and were not used as a replay substitute.

The formal E74.4-G2 archive is complete and audited with 803/871 and the requested six metrics. Its retained logs do not contain per-epoch author-log fields, so a truthful `result_ABIDE.txt` cannot be emitted from that archive. The available `experiment_darg_jsl` result is a separate JSL variant with 808/871 and is not copied into the requested DARG log path.

DPSL full retrain is independently audited exact from E61 epoch 10 with ten journals, Best/Latest state, next-step recovery checks, exact RNG journals, unchanged protected files, and 811/871 correct patients. Archived training time is 1195.6754987239838 seconds.

No result file is emitted for a blocked stage. Synthetic or empty `result_ABIDE.txt` files would falsely claim real epoch records and violate the loss and timing requirements.

Authority directory: `/model/spromptgl/experiment/experiment_abide_training_logs_complete`. Evidence is in `results/stage_linkage.json` and `results/audit_status.json`.
