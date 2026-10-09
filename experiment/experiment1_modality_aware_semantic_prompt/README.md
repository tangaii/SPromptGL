# Experiment 1: Modality-Aware Semantic Prompt Enhancement

This directory contains an isolated implementation of the preregistered MASPE comparison: P0 original prompts, P1 modality-aware prompts, P2 fixed modality-mismatch control, and P3 count-matched generic prompts. Each configuration uses the copied SPromptGL GFR, CAM, PLN, classifier, and training objective; the experiment changes only the semantic text grouping and matching inside PLN.

The source snapshot is under `src/reference/`; executable code is under `src/`. Runtime data are read-only from `/data/adodas/SPromptGL-main/data/ABIDE`. The experiment does not import another experiment directory and does not modify the root model or data files.

The protocol uses three seeds and ten stratified outer folds. Each outer training partition is split again to obtain a validation set for early stopping. All four configurations share the stored split files. Features from all subjects remain visible to graph propagation, matching the original transductive setting; only training labels enter the supervised target graph and classification loss. The standardized/selected input files were created before this fold-wise experiment, so their full-cohort preprocessing leakage is not removed here.

The shell command's ABIDE parameters are recorded in `configs/experiment.json`. The local shell settings differ from the implementation values described in the paper; this experiment follows the requested shell settings consistently and records the difference in `audit.md`.

Run prompt preparation and checks, then the full scheduler:

```bash
cd /data/adodas/SPromptGL-main/experiment/experiment1_modality_aware_semantic_prompt
python src/prepare.py
python -m unittest discover -s tests -v
bash run_all.sh
```

Resume safely with:

```bash
bash resume.sh
```

The scheduler runs up to two configuration/seed workers per accelerator, validates each saved fold before skipping it, and updates the root `handoff.md` after each completed task. Final tables and the Chinese report are written to `results/`. No Git push is performed by these scripts.
