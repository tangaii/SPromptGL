# Experiment 83.3

TADPOLE Four-Group Fidelity Restoration and MMGPL-12 DPSL Adaptation.

Status is `COMPLETE_AUDITED`. The locked 598-patient TADPOLE six-modality input uses ten fixed stratified folds with seed 0. G1, G2, and G3 are inherited E83.1 formal checkpoints marked `REUSED_VERIFIED_FORMAL` after independent replay. G4 contains ten new formal MMGPL-12 SCMR training folds.

All source changes are local to this directory. The `/model` path is active because the preferred `/data` OSSFS path returned an I/O error. Formal evidence is in `audits/preflight.json`, `audits/checkpoint_replay.json`, `audits/latest_replay.json`, and `audits/final_audit.json`. `common/run_e833.py` is the G4 training entry point. `common/replay_check.py` and `common/latest_replay.py` are the checkpoint audit entry points. Metrics are recomputed by `common/report_e833.py` from fold-out predictions.
