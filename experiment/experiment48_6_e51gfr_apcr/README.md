# Experiment 48.6 — E51 LegacyFixedGFR + APCR alpha ablation

This experiment changes only the GFR base from the B0-B GFR used by Experiment 48.5 to Experiment 51 E1's `LegacyFixedGFR`. Its four existing `GraphLearn` modules are also the sole source of APCR candidate scores through `graphs[m].score431`; no duplicate graph learners are created. The E51 GFR returns `(fused, adjs, details)`, while the 48.5 APCR runner expected the old four-argument GFR interface, so the local adapter calls the E51 interface and reads scores from those same graph modules. E51's CAM output is consumed at index 0, preserving the original CAM computation.

Both arms execute the unchanged Experiment 48.5 full APCR path (`training_group=6`): SRD, same-patient-pair SCA, finite-candidate CRA, four relation messages, gate, all loss weights, and APCR Adam settings. E1 freezes `alpha_raw`; E2 permits its update. Both start with alpha 0.0022 and the same fold initialization bundle. E51 base/CAM initialization is rebuilt from the corresponding Experiment 43.1 fold initialization anchor using the frozen Experiment 51 E1 runner; no trained best checkpoint is used.

The server's `/data` filesystem returned `EIO` on a minimal directory creation. Until it recovers, all new artifacts are staged under `/tmp/SPromptGL-E48.6`; delivery to the requested `/data/.../experiment48_6_e51gfr_apcr` path is not claimed unless verified by a complete hash-checked sync.

The historical protocol retains global preprocessing/CV-before RFE, transductive 871-node graph construction, and `val=test` checkpoint selection. The results therefore are not unbiased outer-test estimates. E51 E1 and Experiment 48.5 results are historical references and are not retrained.

Run `python run.py init`, `python run.py preflight`, then the five workers described in `config.json`. After all 20 tasks finish, run `python audit.py` in a fresh process.
