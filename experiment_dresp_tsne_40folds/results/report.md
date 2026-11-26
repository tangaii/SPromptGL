# ABIDE DReSP t-SNE feature audit

## Status

`COMPLETE_AUDITED_TSNE_40_WITH_E28_METRIC_PARITY`

E28-E0 seed=0 was retrained in the isolated audit directory because no independently loadable historical Best files could be recovered. Its ten-fold Best epoch, ACC, BA and probability-AUC values match the archived E28-E0 seed=0 task records exactly. The lost historical Best byte identity is not claimed.

## Model and source audit

SPromptGL uses the isolated E28-E0 seed=0 retrain Best states. DARG uses the formal E74.4-G2 Best checkpoints from the authenticated DARG runtime. DARG + JSL uses the ten formal `foldNN.best.pt` files under `experiment_darg_jsl`. DReSP uses the ten formal DPSL Best checkpoints for E82.4.6.10-G1. Each model was evaluated in its own Python process. JSL and DPSL use their historical Jacobian-compatible evaluation paths. The captured representations are classifier-head input CAM144 for SPromptGL and DARG, JSL `enhanced`, and DPSL `capture["enhanced"]`.

## E28 retraining and log

The E28 controller preserved the continuous seed=0 RNG stream across folds, original validation-equals-test selection, original fold order, original stopping rule, and original training update order. Additional serialization occurs after the existing update and evaluation steps. The generated `E28_E0_seed0/result_ABIDE.txt` contains 4,197 real epoch rows with the original repeated `trn-loss-MF` field names and formatting. Best and Latest files are present for all ten folds. Checkpoint SHA-256 values are recorded in `audit_status.json`.

## Features and t-SNE

There are 40 arrays with shape `(871, 144)`, one per model and fold. Patient IDs are `0..870` in identical order and labels are ASD=0, TD=1 for every model within each fold. The saved feature archive is `features_all_folds.npz`. The coordinate file contains 34,840 rows. Each model-fold was independently standardized, reduced from 144 to 50 dimensions with PCA, and passed through t-SNE with `random_state=0`, `perplexity=40`, `early_exaggeration=12`, `learning_rate="auto"`, `init="pca"`, `max_iter=2000`, and Euclidean distance. Labels were used only for color and geometry summaries.

## Figures

Forty individual PNGs and ten same-fold four-model comparison PNGs were generated at 600 DPI. `plot_tsne.py` reads the saved coordinates and reproduces the figures without model execution or t-SNE refitting. The comparison panels are ordered SPromptGL, DARG, DARG + JSL, DReSP for each Fold.

## Geometry and enhancement analysis

`feature_geometry.csv` contains raw-144-dimensional within-class distance, between-class distance, between/within ratio, Silhouette score, feature norm, range, non-finite count, and rounded duplicate rate for every model-fold. `audit_status.json` contains the JSL and DReSP same-forward CAM-to-enhanced L2 and relative changes. These summaries are descriptive and do not establish classification performance or causal module benefit.

## Protocol limitations

The historical experiments include transductive patient-graph processing, validation-equals-test checkpoint selection, pre-CV preprocessing, and differing training-member sets and overlap conventions. Cross-model visual differences therefore require those protocol differences to be reported. No paper Fold was selected in this audit.
