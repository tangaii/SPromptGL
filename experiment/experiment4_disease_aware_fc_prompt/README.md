# Experiment 4: Disease-Aware Functional-Connectivity Prompting

This directory contains the frozen DAFCP experiment for SPromptGL.

## Configurations

- `C0`: frozen Experiment 1 P1 baseline.
- `C1`: P1 with four ASD-aware Correlation prompts.
- `C2`: P1 with matched population-level Correlation prompts.

Only the eight Correlation prompts differ between C1 and C2. PHENO, ANAT,
and FUNC remain byte-for-byte equal to P1. Prompt text is validated against
the CLIP context limit before training.

## Reproduction

Run from the repository root:

```bash
python experiment/experiment4_disease_aware_fc_prompt/run.py --prepare
python experiment/experiment4_disease_aware_fc_prompt/run.py --all
python experiment/experiment4_disease_aware_fc_prompt/run.py --refresh
```

The scheduler launches six independent workers, with three workers on each
GPU, and each worker runs the original ten-fold SPromptGL protocol. The
worker patch records one diagnostic row for each modality in each test fold.

## Audit files

- `prompts.json`: exact prompt definitions and provenance.
- `prompt_patch.diff`: generated source patch applied to isolated workers.
- `manifest.json`: source, data, CLIP, protocol, assignment, and reference hashes.
- `results/reference.csv`: frozen C0 fold results from Experiment 1 P1.
- `results/fold_results.csv`: C1/C2 fold-level metrics after completion.
- `results/diagnostics.csv`: prototype diagnostics for C0, C1, and C2.
- `results/report.md`: generated comparison and interpretation report.
- `logs/`: scheduler and per-task execution logs.

The Experiment 3 diagnostic fold-number issue is handled here by freezing the
corrected C0 diagnostic reference and writing the active `clk` value into each
new worker diagnostic row.
