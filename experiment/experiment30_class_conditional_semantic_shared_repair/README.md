# Experiment 30: Class-Conditional Semantic Shared Repair

Experiment 30 replaces SPGS for all new Z/C/F configurations with class-conditional connectivity semantic alignment (CCSA) and reliability-aware shared auxiliary repair (RSAR). Historical P0/E1 metrics are read-only references; no new P0 was trained.

The frozen design trains 18 configurations at seed 0 for 10 folds each. The baseline data protocol, 871-subject transductive graph, original GFR/CAM/RES/classifier and graph losses, and test-equals-validation checkpoint selection are retained. Hard-AUC uses the original one-hot hard-prediction ROC definition.

`prompt_banks.npz` contains the frozen 764-text RN50 bank and registered permutations. Training-fold statistics select up to 16 of 252 verified CORR connections. New residuals modify only CAM-CORR and are bounded relative to its norm. Auxiliary losses and training-fold class supervision are documented in `method.py` and `manifest.json`.

## Results

- Status: COMPLETE, 18/18 tasks and 180/180 folds.
- GPU allocation: 90 folds each on GPU0 and GPU1.
- Metrics, paired contrasts, mechanism diagnostics, limitations, and the pre-registered NO-GO decision: `results/report.md`.
- Fold metrics: `results/fold_metrics.csv`.
- Fold diagnostics: `results/diagnostics.csv`.
- Source, data, prompt, result hashes, preflight evidence, and handoff status: `manifest.json`.

The retained protocol uses test-equals-validation checkpoint selection, preprocessing completed before outer cross-validation, and transductive graph propagation over all 871 subjects. Results are internally comparable under that protocol, not an unbiased independent-generalization estimate. No clinical causal or lesion-localization claim is supported.

No root core source or historical experiment was modified. No Git commit or push was performed. Temporary runtime checkpoints and logs were removed after completion.
