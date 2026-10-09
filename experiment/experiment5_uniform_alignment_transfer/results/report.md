# Experiment 5: Cross-Model Transfer of Uniform Global Alignment to SPromptGL

## 1. Executive Summary

在 strict protocol 下，MMGL 固定的 uniform global alignment 对 SPromptGL 的迁移结论为 **YES**；在作者原始 legacy protocol 下为 **NO**。strict 是主判断，legacy 结果保留为 protocol sensitivity analysis。

## 2. Previous MMGL Evidence

| Model | Protocol | Baseline BA | Aligned BA | ΔBA | Baseline prob-AUC | Aligned prob-AUC | ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMGL | Strict | 83.79 | 84.60 | +0.81 | 87.82 | 89.38 | +1.57 |
| MMGL | Legacy | 89.17 | 89.50 | +0.34 | 92.23 | 92.19 | -0.04 |
| SPromptGL | Strict | 84.04 | 84.80 | +0.76 | 89.71 | 91.05 | +1.34 |
| SPromptGL | Legacy | 91.55 | 91.30 | -0.25 | 93.02 | 92.90 | -0.12 |

## 3. SPromptGL Repository Audit

Source commit: `ce983b85a36d19b5d22340d1cee101add0a0a060` (the supplied project root is an exported worktree; this is the uploaded `main` commit). The original protocol uses `StratifiedKFold(n_splits=10, random_state=10, shuffle=True)`, `val=test`, hard-prediction AUC, 450 maximum epochs, early stopping after `wait_cnt > 200`, and joint GFR+PLN training. The alignment term never includes `ALoss` in the optimisation objective.

## 4. Dataset Parity With MMGL

CSV SHA256: `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`; modal dictionary SHA256: `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`; both match MMGL Experiment 4. CROSS_MODEL_DATA_MATCH: **YES**.

## 5. SPromptGL Architecture

Modalities → GFR → PLN Transformer → pre-prompt token `[N,4,36]` → semantic prompt → post-prompt token `[N,4,36]` → classifier.

## 6. Exact Transfer Site

Alignment is trained only on the representation after the final FeedForward layer and before the original semantic prompt guidance. This is the representation level corresponding to the MMGL transfer site; post-prompt metrics are diagnostic only.

## 7. Uniform Alignment

The copied alignment implementation SHA256 is `7c39a2ebdc136f66c00ac698975803119c57df31110a96bceece9cb7cec800cc` for both source and destination. It uses six equal modality pairs, `L_align = 0.5 L_InfoNCE + 0.5 L_CMMD`, temperature `0.10`, and effective λ `0.50` with a 20-epoch warmup.

## 8. Four Configurations

S0/S1 are strict baseline/alignment; L0/L1 are original legacy baseline/alignment. No post-prompt, pair-adaptive, sparse, or result-guided configuration was run.

## 9. Strict Protocol

Ten outer folds use `random_state=seed`; each outer training portion has a 10% inner validation split with `random_state=seed+1000+fold_id`. Fold RNG is reset using the registered seed formula. Outer test labels are excluded from checkpoint selection; transductive test features remain available to the population graph. Primary AUC is probability ASD-AUC.

## 10. Original Legacy Protocol

Legacy fixes `random_state=10`, sets `val_idx=test_idx`, selects checkpoints by test-fold ACC, reports the original hard AUC, and preserves one sequential RNG trajectory across the ten folds of each config×seed job.

## 11. Baseline Integrity

Forward parity probability max error and hidden max error were both 0; token shapes were `[871,4,36]`; data and alignment SHA checks passed. Strict S0/S1 initialization parity was 30/30.

## 12. Legacy Repository Reproduction

`result_ABIDE.txt` reference: ACC 92.30% and hard AUC 92.22%. L0 seed-0: ACC 91.73% and hard AUC 91.57%; maximum absolute difference 0.65 percentage points. SPROMPTGL_LEGACY_BASELINE_REPRODUCED: **PARTIAL**.

## 13. Strict Main Results

S0 ACC 84.35±1.45%, BA 84.04±1.29%, probability ASD-AUC 89.71±0.61%; S1 ACC 84.96±1.13%, BA 84.80±1.02%, probability ASD-AUC 91.05±1.06%. ΔBA +0.76 pp; ΔAUC +1.34 pp.

## 14. Strict Alignment Audit

| Config | pre cos-gap | pre R@1 | pre CMMD |
|---|---:|---:|---:|
| S0 | 0.1549 | 0.0216 | 0.3377 |
| S1 | 0.3771 | 0.2095 | 0.1384 |

STRICT_PRE_ALIGNMENT_WORKING: **YES**.

## 15. Strict Prompt Propagation

| Config | post cos-gap | post R@1 | post CMMD |
|---|---:|---:|---:|
| S0 | 0.1550 | 0.0216 | 0.3377 |
| S1 | 0.3772 | 0.2092 | 0.1384 |

STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **YES**.

## 16. Legacy Results

L0 three-seed OOF BA 91.55±0.11%, probability ASD-AUC 93.02±0.66%; L1 BA 91.30±0.41%, probability ASD-AUC 92.90±0.26%. ΔBA -0.25 pp; ΔAUC -0.12 pp.

## 17. Legacy Alignment Audit

L0 pre: cos-gap 0.2110, R@1 0.0223, CMMD 0.3224; L1 pre: cos-gap 0.4300, R@1 0.2617, CMMD 0.1301.

LEGACY_PRE_ALIGNMENT_WORKING: **YES**; LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **YES**.

## 18. ASD Sensitivity and NC Specificity

The machine-readable `summary.csv` and `seed_summary.csv` report ASD sensitivity and NC specificity for all four configurations. These class-wise metrics are retained so a BA change can be attributed to ASD detection or NC rejection.

## 19. Strict Paired Bootstrap

seed 0: ΔBA +0.0174, 95% CI [+0.0029,+0.0322], p=0.0194; ΔAUC +0.0297, 95% CI [+0.0201,+0.0397], p=0.0000.
seed 1: ΔBA -0.0091, 95% CI [-0.0257,+0.0071], p=0.2886; ΔAUC +0.0083, 95% CI [-0.0033,+0.0202], p=0.1658.
seed 2: ΔBA +0.0142, 95% CI [-0.0026,+0.0315], p=0.1048; ΔAUC +0.0019, 95% CI [-0.0093,+0.0133], p=0.7376.

## 20. Legacy Paired Bootstrap

seed 0: ΔBA -0.0005, 95% CI [-0.0101,+0.0094], p=0.9366; ΔAUC -0.0085, 95% CI [-0.0191,+0.0014], p=0.0968.
seed 1: ΔBA +0.0012, 95% CI [-0.0063,+0.0090], p=0.7792; ΔAUC +0.0057, 95% CI [-0.0051,+0.0167], p=0.2860.
seed 2: ΔBA -0.0084, 95% CI [-0.0189,+0.0019], p=0.1118; ΔAUC -0.0008, 95% CI [-0.0093,+0.0080], p=0.8370.

## 21. Cross-Model Strict Comparison

MMGL strict ΔBA +0.81 pp / ΔAUC +1.57 pp; SPromptGL strict ΔBA +0.76 pp / ΔAUC +1.34 pp.

## 22. Cross-Model Legacy Comparison

MMGL legacy ΔBA +0.34 pp / ΔAUC −0.04 pp; SPromptGL legacy ΔBA -0.25 pp / ΔAUC -0.12 pp.

## 23. 2×2 Protocol Matrix

See `results/protocol_matrix.csv` for the four-row matrix. Absolute scores across protocols are not pooled; interpretation uses within-protocol aligned-minus-baseline effects.

## 24. Interpretation

Strict estimates use an inner validation fold for checkpoint selection and therefore provide the primary cross-model test. Legacy estimates reproduce the original test-fold checkpoint selection and are reported as legacy-selected OOF. Representation metrics are post-hoc diagnostics and never select a checkpoint.

## 25. Verdict

CROSS_MODEL_DATA_MATCH: **YES**
SPROMPTGL_LEGACY_BASELINE_REPRODUCED: **PARTIAL**
STRICT_INIT_PARITY: **YES**
STRICT_PRE_ALIGNMENT_WORKING: **YES**
STRICT_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **YES**
LEGACY_PRE_ALIGNMENT_WORKING: **YES**
LEGACY_ALIGNMENT_PROPAGATES_THROUGH_PROMPT: **YES**
STRICT_ALIGNMENT_TRANSFER_SUPPORTED: **YES**
LEGACY_ALIGNMENT_SUPPORTED: **NO**
CROSS_MODEL_GENERALIZATION: **YES**
PROTOCOL_ROBUST_SPROMPTGL_ALIGNMENT: **NO**

## 26. Final Scientific Conclusion

uniform global alignment 的最终结论由 strict S1−S0 决定；legacy 结果只说明在原始 test-fold checkpoint 选择协议下的敏感性。不要把两种 protocol 的绝对分数视为同一个估计量。

## 27. Next Step

根据 strict 结果，只保留一个后续动作：报告跨模型 alignment effect，并停止在本实验上继续搜索 λ、temperature 或新的 alignment 位置。

## Provenance

Python: `/root/venvs/111/bin/python`; PyTorch: `2.9.0`; CUDA available: `True`; GPUs: `2`; CLIP RN50: `/root/.cache/clip/RN50.pt` (`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`); formal jobs: 12; configurations: 4; seeds: 3; folds per job: 10; expected/actual fold rows: 120/120; expected/actual OOF rows: 10452/10452; destination alignment SHA: `7c39a2ebdc136f66c00ac698975803119c57df31110a96bceece9cb7cec800cc`.
