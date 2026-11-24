# E83.3 method-fidelity matrix

| Area | Historical source | E83.3 treatment | Evidence |
|---|---|---|---|
| TADPOLE input | E83.1 locked MMGL preprocessing | Reused local frozen CSV, modality dictionary, sample IDs, and ten splits | `audits/preflight.json`, TEST_20 |
| G1 | E28-E0 Original SPromptGL | Historical GFR overwrite, CAM, PEN, SPGS/RES, graph loss, and 8-prompt RN50 path | TEST_01, TEST_05-07 |
| G2 | E43.1-E00 | PEN physically absent, historical GFR/CAM/head and optimizer retained | TEST_02 |
| G3 | E74.4-G2 | Six-modality DARG adaptation with train-only 4096 relation pairs, signed unified graph, Positive-DCR and original warmup | TEST_08-11, inherited replay |
| G4 base | E74.4-G2 plus E82.4.6-G2 SCMR-DA-SRPE | Same DARG warmup and optimizer state per fold, then DPSL attached at the historical post-warmup point | TEST_15-18, G4 warmup records |
| Historical DPSL regression | E82.4.6-G2 | Eight-query binary branch is retained for exact tensor, objective, EMA and intervention regression | TEST_13 |
| MMGPL-12 query bridge | E82.4.6-G2 prompt interface | Frozen RN50 vectors projected through historical `P_text`. Anonymous queries use deterministic 8-row orthogonal Procrustes alignment | `prompt_manifest.json`, `inputs/prompts/mmGPL12_scmr.pt` |
| Three-class routing | Historical two-class normal/abnormal path | Fixed NC/MCI/AD groups of four. Native Top-2 determines low/high disease-stage direction for semantic coverage, Gaussian evidence and Jacobian margin | TEST_13B, G4 replay records |
| Residual and gradient controls | E82.4.6-G2 | Residual ratio capped at 0.08 and shared differential gradient budget capped at 0.001 | TEST_15-16, G4 replay |
| Checkpoint selection | Historical SPromptGL protocol | `val_idx == test_idx`, first strict ACC improvement only, equal ACC does not replace Best | G4 train log and checkpoint payloads |

E83.3 local modifications are limited to the copied experiment directory. The source root and historical experiment directories were not modified. The MCI prompts correspond to MMGPL's LMCI wording and are recorded as TADPOLE MCI semantic proxies.
