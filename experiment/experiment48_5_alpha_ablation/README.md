# Experiment 48.5 — APCR Alpha Ablation

This experiment compares only whether the APCR residual coefficient is trainable. Both arms use alpha=0.0022 initially, the original sigmoid parameterization and alpha_max=0.2. Both invoke the unchanged Experiment 48.4 `group=6` training path for all SRD, SCA, CRA, graph, classification and KL losses. E1/E2 are task identifiers, not values passed as method group IDs.

The requested `/data/adodas/SPromptGL-main/experiment/experiment48_5_alpha_ablation` path is on an OSSFS mount that returned EIO on directory creation. The experiment is staged at `/tmp/experiment48_5_alpha_ablation` until synchronization succeeds. The parent Experiment 48.4 source, exact fold anchors, and static historical summaries are retained under `.runtime`; the parent experiment and all root model/data files remain read-only.

The historical evaluation protocol is preserved: seed 0, 10-fold StratifiedKFold with shuffle and random_state 10, transductive patient graph, globally preprocessed/CV-before-RFE features, and `val=test` checkpoint selection. These results are not unbiased external generalization estimates.

Run `python run.py init`, `python run.py preflight`, then launch five logical workers on the single GPU with fixed fold assignments from config. Run `python audit.py` as a separate process after all 20 folds complete.
