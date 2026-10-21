# Experiment 33: QCAR Conditional Cross-Modal Alignment

This experiment tests whether fold-local, label-free ANAT/FUNC acquisition covariates explain reliable CORR changes beyond age, sex, and site. It uses the current Experiment 32 B0-A implementation: original GFR, CAM and OutputLayer; PEN modules remain constructed but are bypassed in the forward path.

The eight frozen configurations are G0_B0A, G1_SITE_ONLY, G2_UNCONDITIONAL, G3_NO_RELIABILITY, G4_QCAR_FULL, G5_WRONG_PAIR, G6_FC_ONLY and G7_REVERSE_SIGN. Every configuration receives one seed-0 worker that runs folds 1 through 10 continuously. GPU 0 runs G0/G2/G4/G6; GPU 1 runs G1/G3/G5/G7.

Run from the repository root:

```bash
python -B experiment/experiment33_qcar_conditional_alignment/run.py prepare
python -B experiment/experiment33_qcar_conditional_alignment/run.py preflight
python -B experiment/experiment33_qcar_conditional_alignment/run.py run
```

Workers write atomic epoch and fold-boundary checkpoints under `.runtime/`. Re-running `run` resumes each configuration from its latest valid checkpoint and skips completed folds. The fold statistics and code/input hashes are recorded in `manifest.json`; final four-metric results are in `results/fold_metrics.csv` and `results/report.md`.
