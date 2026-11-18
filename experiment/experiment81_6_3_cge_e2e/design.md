# Experiment 81.6.3 SA-PRCL-CGE design freeze

## Source review

The original SPromptGL paper was reviewed at https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf. The local author-code files were reviewed with these SHA256 values: `main.py` `dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2`, `network.py` `484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec`, `layers.py` `4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a`, and `utils.py` `95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67`. The review confirmed GFR, CAM/PEN/SPGS/RES, the original OutputLayer LayerNorm144 -> Linear144,36 -> GELU -> Dropout0.35 -> Linear36,2, and the ABIDE classification flow. SA-PRCL uses the local relation branch and does not reactivate PEN.

The locked E74.4-G2 source files are `experiment/experiment74_4_soft_positive_dcr/run.py` `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd`, `soft_positive_dcr.py` `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718`, `config.json` `a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a`, and `audit.py` `cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb`. The actual historical forward defines `total = cls + graph + 0.05*pair + 0.01*adv + 0.01*preserve`. Its `prepare(fold)` follows the audited E74.4 -> E74.1 -> E69.3 -> E61 chain and restores the E61 epoch-10 warmup state, including the three optimizer states and RNG. Fold 2 retains 1-based overlap patients 377 and 407 and 4096 training relation pairs.

E81.6.2 was read from `/model/spromptgl/experiment/experiment81_6_2_semantic_anchored_prcl`. Actual file hashes are `run.py` `8c0d9a2fea12b24ed6da7f30af239f0d36136723acfd6f03efdc2abb825cf7a6`, `sa_prcl.py` `c05fe067bca1e8c4be2ef587b7c7fcb9fcca29417f2861a85068e8c14f414beb`, `audit.py` `1f853e481b47cb27fcd769ce3ede2d526b8905c229a6cd62968407dc51e09966`, `config.json` `487221b64e2a0a35ff40f9e8320d32456c269fa25bc4d8e73da8d04b775a7ff8`, `design.md` `fe988abf8887969c505592f5c63f4d724bb5a2652aa124059c632546604e1b11`, and `results/report.md` `e76f572ba55c3bdb292f72e8c777801b53e27e846e3b14c822fd87f60bec00b2`. The user-declared historical commit is `0ef7b957bde68bb840d844ebffc5c83cfebad2ed`, recorded separately from these verified filesystem hashes.

E81.6.2 G1 used detached CAM and a frozen functional classifier branch. G3 injected the enhanced CAM into the original OutputLayer. Its `eta=0.1` scaled the relation branch input gradient only, while the `z_cam + delta` identity path still carried the full classifier gradient. The audited E81.6.2 results were G1/G2 ACC 92.0742% and G3 ACC 91.8443%, versus DARG ACC 92.1891%. G1/G2 hard predictions were fold-wise identical. G1 own-best residual-on versus residual-off gave net correction +2, while residual-on at the selected E81.6.2 checkpoint versus historical DARG Best gave net correction -1. These are distinct branch and checkpoint-selection comparisons.

## Frozen 81.6.3 method

The eight original prompts, frozen RN50 text features, 8-dimensional thin-SVD basis, pairwise key/value relations, 15 real neighbours plus Null, patient-conditioned beta weights, medical/scrambled grouping, RES AvgPool1d(4,4) matching, zero-initialized Linear32,9, 10% residual bound, and `lambda_raw=0.001` are inherited unchanged from E81.6.2. The new module has 8032 trainable parameters and no trainable text projection or Prompt embedding.

G1 uses normal `[0,1,2,3]` and abnormal `[4,5,6,7]`. G2 uses normal `[0,1,4,5]` and abnormal `[2,3,6,7]`. G2 is conditional and is run only after G1 passes the preregistered all-10 and excluding-Fold-2 ACC and Hard-AUC gate with mechanism and audit checks.

For each step, the original DARG forward produces `p0` and the exact historical loss. The same OutputLayer parameters then produce `p1` from `z+delta` through a functional implementation with the captured original Dropout RNG state. The replay restores the public RNG after evaluation. `L_delta = ClsLoss(p1)-ClsLoss(p0)` uses the original training rows. Historical parameters receive `g_base + alpha*project(g_delta)` with `alpha=0.2`, conflict projection, and per-optimizer `kappa=0.1` norm clipping. SA-PRCL parameters receive the full `ClsLoss(p1)+0.001*L_raw` gradient. `eta=0.1` remains limited to the relation input path. Historical optimizer order is GFR, CAM, DARG, then SA-PRCL.

Every fold restores E61 epoch-10, seed 0, the complete DARG parameter and optimizer states, and the public CPU/CUDA RNG. SA initialization uses the isolated private salt `816200+fold`. Fold 2 rows 376 and 406 are excluded from SA supervision and residual regularization while the historical DARG loss is unchanged.

The preferred directory creation returned OSSFS `EIO/errno=5`, so all files are isolated under `/model/spromptgl/experiment/experiment81_6_3_cge_e2e`. No historical checkpoints or root model files are modified.

## Preflight and audit commitments

Preflight checks cover all ten historical starts, SVD Gram identity, shapes and 8032 parameters, zero residual and Prompt-off/Null invariants, Dropout mask and public RNG replay, alpha-zero DARG update identity, nonzero controlled gradient feedback, conflict projection, Fold 2 blocking, and exact Latest recovery. Best and Latest checkpoints contain the four optimizer states, all model parameters, RNG, source hash, current and Best epoch, and gradient configuration. Independent audit reloads checkpoints before computing the six metrics, paired comparisons, mechanism rows, and the G1/G2 continuation decision.

## References borrowed and differences

KD-Brain's semantic prior entering relation attention was used only as conceptual motivation. ItemizedCLIP's independent text-condition idea motivated fixed Prompt identities. Their brain-specific networks, BioMedBERT, item labels, graph modeling, and supervision are not used. The reference source hashes and read-only paths are recorded in `results/preflight.json` and `manifest.json`.
