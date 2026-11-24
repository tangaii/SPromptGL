# Experiment 83.4

This directory is the self-contained TADPOLE four-group release. The preferred OSSFS experiment path returned EIO, so the active root is `/model/spromptgl/experiment/experiment83_4_tadpole_four_group_complete_evidence`.

The locked E83.3 data and folds are reused. G1-G3 are imported only after independent replay. G4 is freshly trained with the preregistered MMGPL-12 and three-pair evidence adaptation. Formal training is gated by `audits/preflight.json` and `audits/training_seal.json`.

Measure one- and two-process GPU throughput, then run the method preflight and create the source-bound seal:

```bash
cd /model/spromptgl/experiment/experiment83_4_tadpole_four_group_complete_evidence
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B common/resource_probe.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B common/preflight.py
```

Read `audits/resource_probe.json` and use its `selected_gpu_workers` value to start or resume the registered G4 folds:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B common/run_e833.py G4_E82_4_6_10_G1 --workers 2
```

Set `E831_FOLDS=1,2` to restrict a recovery invocation to listed folds. Existing Latest checkpoints are restored under the fold lock. Best and Latest are committed atomically.

`common/report_e833.py`, `common/replay_check.py`, and `common/latest_replay.py` provide the result and checkpoint audits. No source outside this experiment directory is used for formal training.
