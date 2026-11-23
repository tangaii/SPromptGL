# Experiment 83 — TADPOLE Four-Group Comparison

This directory is the isolated execution area for E83. The raw TADPOLE files are read from `/model/spromptgl/Tadpole/tadpole_challenge` because the preferred repository-backed experiment path returned an OSS `Input/output error` during directory creation.

The preprocessing script is `preprocessing/preprocess_tadpole_mmgl.py`. It follows the public MMGL TADPOLE notebook, including stable NC/sMCI/AD selection, baseline visit selection, full-data RFE before outer CV, StandardScaler before outer CV, and mean imputation. The generated data are `preprocessing/processed_standard_data.csv`, `preprocessing/modal_feat_dict.npy`, and `preprocessing/sample_ids.csv`.

`common/run_e83.py` is the only training entry point. It uses seed 0, shared `StratifiedKFold(n_splits=10, shuffle=True, random_state=10)` splits, `val_idx == test_idx` checkpoint selection by ACC, and dynamic six-modality dimensions. Completed folds are resumed only when their checkpoint and finite metric row both exist.

The isolated model in `common/tadpole_model.py` keeps the graph relation learner, CAM, PEN, E74 positive disease-aware residual, and E82 spectral prompt residual. Historical ABIDE samples, checkpoints, predictions, overlap IDs, and fold outputs are not used.

Formal commands are:

```bash
python -u common/run_e83.py G1_E28_E0
python -u common/run_e83.py G2_E43_1_E00
python -u common/run_e83.py G3_E74_4_G2
python -u common/run_e83.py G4_E82_4_6_10_G1
```

Use `python common/run_e83.py all` to resume every group. Result aggregation and audits are generated after all groups finish.
