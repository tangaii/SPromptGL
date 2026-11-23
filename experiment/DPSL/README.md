# DPSL standalone base

This directory freezes the formal `E82.4.6.10-G1` ten-fold composition. The external task name is G1. The serialized SCMR model identity remains `E82.4.6-G2 SCMR-DA-SRPE` in every selected checkpoint.

Run from any working directory with the local Python and PyTorch environment:

```bash
python /model/spromptgl/experiment/DPSL/cli.py inspect
python /model/spromptgl/experiment/DPSL/cli.py evaluate
python /model/spromptgl/experiment/DPSL/cli.py audit
python /model/spromptgl/experiment/DPSL/cli.py resume-check
python /model/spromptgl/experiment/DPSL/cli.py train-resume
```

`evaluate` replays the exact evaluation payload stored in each frozen Best checkpoint and joins it to the local 871-patient table. `train-resume` is intentionally a no-op in this migration.

The 19 listed patients are intentional historical train-test label overlaps. They retain DARG classification and relation eligibility while remaining outside the added spectral, Prompt, Gaussian, and EMA label supervision. Their added CAM residual is zero.

`code/provenance` contains read-only source identity and historical result snapshots. The public CLI imports only `code/dpsl_runtime.py` and local files below this directory.
