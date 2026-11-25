# Experiment 83.7 Paper and Code Fidelity

## Original paper and author code

The MICCAI 2025 SPromptGL paper defines Graph-based Feature Representation (GFR), multi-relation graph learning and graph convolution, a Cross-Attention Module (CAM), Prompt-guided Embedding Network (PEN), and Relation-aware Embedding Strategy (RES). Its loss combines graph learning loss and label cross-entropy. The paper reports the TADPOLE three-class reference of 94.81 +/- 2.95% ACC and 95.59 +/- 2.57% AUC.

The author snapshot is commit `ba3b9446fa451fde6864d6555bdf324de0c4b64a` from `wanxixi11/SPromptGL`. The checked files are `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, and `result_Tadpole.txt`. The author repository is primarily ABIDE-oriented, so its fixed subject count, four modalities, and binary prompt assumptions are not used for this TADPOLE run.

## Project and sp lineage

The E83.5 project reference is commit `4649c5dddf166a4aa957c16c2f667a5cea485cdc`, with sealed runtime commit `5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11`. The E83.6 sealed cross-repository implementation is retained as the local source basis. The `tangaii/sp` snapshot used in E83.6 is recorded as commit `f4534f8` in its prior manifest. E83.7 does not transplant its failed complete or half-balanced graph-supervision variants.

## TADPOLE adaptations

The locked TADPOLE data has 598 patients, NC/MCI/AD labels 0/1/2, six modalities with dimensions 145/148/3/36/24/9, and CAM width 240. MMGPL12 uses 12 prompts arranged as four NC, four MCI semantic proxies, and four AD queries. DPSL routes all three pairs NC-MCI, NC-AD, and MCI-AD. DARG keeps Positive-DCR signed positive and negative relation updates. The historical second-modality GFR message overwrite and omitted ALoss optimization behavior are retained.

G2 means physical PEN removal while preserving GFR, GraphLearn, GCN, message passing, and graph-related losses. G5 is the physical CAM-only model. This wording corrects the earlier E83.6 description in this new document only.

## E83.7 controlled variables

T1 and T2 change only the DPSL shared gradient budget from 0.001 to 0.003 and 0.010. T3 and T4 add the continuous six-modality `pair_mean` supervision only during epochs 1-10, then set its coefficient and contribution to zero. T5 and T6 preserve the original ten-round warmup at GFR/CAM weight decay 0.11 and update only those two optimizer groups to 0.055 at epoch 11. DARG and DPSL optimizer settings remain unchanged.

## Historical protocol and limitations

E83.7 preserves full-data RFE, pre-outer-CV standardization and mean imputation, the transductive patient graph, `val_idx == test_idx`, strict first test-ACC Best selection, the 450 epoch limit, `wait > 200`, seed 0, and the ten locked folds. These choices are retained for fidelity and comparability and limit unbiased clinical generalization interpretation. Historical checkpoint files remain outside this directory and are referenced by absolute path and SHA-256.
