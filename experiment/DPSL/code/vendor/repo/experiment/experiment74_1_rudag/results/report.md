# Experiment 74.1 — RUDAG

Status: **COMPLETE_AUDITED**; independent checkpoint audit 20/20; device `PPU-ZW810E`; maximum concurrent workers 10/10; retried task attempts 0; OOM events 0.

## Source and protocol audit

SPromptGL's Figure 1 and Sections 2.1–2.2 / Eqs. (1)–(7) describe modality graphs, graph propagation, prompt-guided embedding, CAM and RES. The root implementation used here has a learned linear projection followed by cosine similarity, a detached `score > 0.9` support mask, two GraphConv layers with a ReLU between them, one `normalize_adj(A+I)`, and the original four GFR fusion weights. The original CAM consumes four 36-D modality slots and produces the same 144-D classifier input. The paper writes a label-supervision graph term and a more general DGCN formulation; neither changes the actual E61/E74 objective used here. Historical `GraphConstructLoss(S_m)` and the total loss `L_cls + L_graph + .05 L_pair + .01 L_adv + .01 L_preserve` were retained.

The E53.3 DARG relation path is retained: four independent 32-D relation encoders; symmetric 65-D pair features (`abs difference`, elementwise product, cosine); 65→32→1 scorers; balanced 4,096 training pairs; CORR-detached conditional modality adversary with GRL on the other three modes; and relation-preservation loss. The original uncalibrated `sigmoid(logit)` score remains. No calibration, OOF fitting, text prompt, extra label-semantic head, or new loss was introduced.

E74 scored only mutual Top-12 candidates, used the same strength for all modes and allowed fallback additions. RUDAG instead reweights every original-positive CORR edge continuously, while non-CORR scores only the historical candidate set intersected with original-positive GraphLearn support. Each mode has one unified adjacency, one historical GCN call and the original CAM. G2 adds the CADR-bounded direct neighbor difference from that same adjacency; G1 leaves it disabled. Both groups share modules, parameter count, optimizer groups, gate initialization and RNG starting state.

BrainPrompt's paper and official `brainpromptg_net.py`, `brainpromptc_net.py`, and ABIDE label prompt were reviewed. Its disease-level objective compares graph representations with encoded class text prototypes through cross-entropy. SABER's available paper describes decision-level semantic injection, gated residual fusion and disease contrastive supervision. No SABER official source repository was confirmed, so no source-code audit is claimed. Both are design references only and are absent from this implementation.

## Classification

Values are 10-fold mean ± population SD in percent. Hard-AUC uses historical hard predictions and equals balanced accuracy for binary labels; Probability-AUC uses continuous ASD probability. Historical rows are read-only references.

|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E69.3-G2 historical|92.0742 ± 2.7482|91.8995 ± 2.7186|93.3759 ± 2.9908|89.5854 ± 3.4710|94.2137 ± 3.9815|91.2984 ± 2.9222|
|E62-E2 historical|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|89.5854 ± 3.4710|93.7789 ± 4.2653|91.0789 ± 3.0892|
|E74-UDAG historical|91.4995 ± 2.6885|91.2511 ± 2.7308|93.9249 ± 2.6632|87.8537 ± 4.3359|94.6485 ± 3.2152|90.5218 ± 3.0064|
|G1 RUDAG-U|91.9592 ± 2.7747|91.7498 ± 2.7845|93.4439 ± 3.0255|89.0732 ± 3.9177|94.4265 ± 3.7676|91.1148 ± 3.0257|
|G2 RUDAG-Full|91.8443 ± 2.9793|91.6363 ± 3.0151|93.5999 ± 3.0799|88.8415 ± 4.3351|94.4311 ± 3.4953|90.9690 ± 3.3140|

## Paired comparisons

Differences are first model minus second model, in percentage points. Confidence intervals are descriptive only because folds share training samples and the historical protocol selects checkpoints on test ACC.

|Comparison|Scope|ΔACC|ΔHard-AUC|ΔProb-AUC|ΔSEN|ΔSPE|
|---|---|---:|---:|---:|---:|---:|
|G1−E74|all10|+0.4598 (3/6/1)|+0.4987 (4/4/2)|-0.4809 (4/0/6)|+1.2195 (3/5/2)|-0.2220 (2/6/2)|
|G1−E74|nine_without_fold2|+0.1277 (2/6/1)|+0.1477 (3/4/2)|-0.5462 (3/0/6)|+0.5420 (2/5/2)|-0.2467 (2/5/2)|
|G2−E74|all10|+0.3448 (4/5/1)|+0.3852 (5/3/2)|-0.3250 (2/0/8)|+0.9878 (4/4/2)|-0.2174 (2/5/3)|
|G2−E74|nine_without_fold2|+0.2554 (3/5/1)|+0.2925 (4/3/2)|-0.2845 (2/0/7)|+0.8266 (3/4/2)|-0.2415 (2/4/3)|
|G2−G1|all10|-0.1149 (2/6/2)|-0.1135 (3/3/4)|+0.1560 (6/0/4)|-0.2317 (3/3/4)|+0.0046 (2/6/2)|
|G2−G1|nine_without_fold2|+0.1277 (2/6/1)|+0.1448 (3/3/3)|+0.2617 (6/0/3)|+0.2846 (3/3/3)|+0.0051 (2/5/2)|
|G2−E69.3-G2 historical|all10|-0.2299 (0/8/2)|-0.2633 (1/6/3)|+0.2240 (6/0/4)|-0.7439 (1/6/3)|+0.2174 (2/7/1)|
|G2−E69.3-G2 historical|nine_without_fold2|-0.2554 (0/7/2)|-0.2925 (1/5/3)|+0.2430 (5/0/4)|-0.8266 (1/5/3)|+0.2415 (2/6/1)|
|G2−E62-E2 historical|all10|+0.0000 (1/7/2)|-0.0459 (2/5/3)|+0.3989 (6/0/4)|-0.7439 (1/6/3)|+0.6522 (3/6/1)|
|G2−E62-E2 historical|nine_without_fold2|-0.2554 (0/7/2)|-0.2925 (1/5/3)|+0.2430 (5/0/4)|-0.8266 (1/5/3)|+0.2415 (2/6/1)|

Per-fold ACC differences (pp) for the predeclared core comparisons:

|Fold|G1−E74|G2−E74|G2−G1|
|---:|---:|---:|---:|
|1|+0.0000|+0.0000|+0.0000|
|2|+3.4483|+1.1494|-2.2989|
|3|+0.0000|+0.0000|+0.0000|
|4|-1.1494|+0.0000|+1.1494|
|5|+0.0000|+0.0000|+0.0000|
|6|+1.1494|+1.1494|+0.0000|
|7|+0.0000|+1.1494|+1.1494|
|8|+0.0000|-1.1494|-1.1494|
|9|+0.0000|+0.0000|+0.0000|
|10|+1.1494|+1.1494|+0.0000|

## Graph and propagation mechanism

The table reports best-checkpoint means across ten folds. Fusion weights are included in the message columns. Per-mode component norms need not sum because message vectors can cancel; `Net fused Δ/input` is the exact norm of the complete fused change relative to the original 320-D input. Exact fold×modality values are in `mechanism.csv`; full-minus-intervention prediction effects are in `interventions.csv`.

|Group|Mode|Effective edges|Scored edges|Changed edges|ΔA/A|Fusion×GCN/input %|Fusion×direct/input %|Fusion×mode-total/input %|Net fused Δ/input %|Final gate|Train relation AUC|Unexposed test relation AUC|Scorer cls grad|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|G1-RUDAG-U|PHENO|81382.0|3497.0|2473.3|5.49e-05|0.016274|0|0.016274|0.037634|0.00100381|0.801539|0.500501|1.402e-09|
|G1-RUDAG-U|ANAT|134833.8|3694.1|1320.7|1.4605e-05|0.010747|0|0.010747|0.037634|0.00100432|0.608975|0.503233|1.191e-09|
|G1-RUDAG-U|FUNC|67921.0|3544.4|2026.9|3.7346e-05|0.007119|0|0.007119|0.037634|0.00100506|0.661480|0.502220|1.327e-09|
|G1-RUDAG-U|CORR|74559.8|74559.8|74559.8|0.65708|0.020050|0|0.020050|0.037634|0.990125|0.995883|0.826201|1.060e-04|
|G2-RUDAG-Full|PHENO|70237.6|3587.1|2342.6|4.0825e-05|0.002993|0.000498|0.003201|0.507266|0.000865824|0.773102|0.502627|9.196e-11|
|G2-RUDAG-Full|ANAT|140404.0|3684.1|920.4|8.5558e-06|0.002107|0.000483|0.002316|0.507266|0.000869758|0.590999|0.501807|1.216e-10|
|G2-RUDAG-Full|FUNC|66321.0|3570.2|1641.4|2.2687e-05|0.000218|0.000448|0.000570|0.507266|0.000873536|0.634375|0.502611|5.798e-11|
|G2-RUDAG-Full|CORR|57949.2|57949.2|57949.2|0.6545|0.000881|0.506115|0.506118|0.507266|0.990172|0.996508|0.836695|3.159e-04|

### Fixed-checkpoint interventions

All intervention metrics use the same checkpoint, test fold and model weights. Positive ACC deltas mean the trained full path exceeds the named intervention.

|Group|Intervention|Mean full−intervention ACC pp|Hard flips|Corrected / newly wrong|Mean abs ΔP(ASD)|Mean fused relative Δ|
|---|---|---:|---:|---:|---:|---:|
|G1-RUDAG-U|disease_neutral|+0.0000|0|0 / 0|0.000108114|1.31565e-05|
|G1-RUDAG-U|corr_neutral|+0.0000|0|0 / 0|0.000108112|1.31564e-05|
|G1-RUDAG-U|score_permuted|+0.0000|0|0 / 0|0.000117033|1.38355e-05|
|G2-RUDAG-Full|disease_neutral|+0.1149|1|0 / 1|0.00119272|0.00084511|
|G2-RUDAG-Full|corr_neutral|+0.1149|1|0 / 1|0.00119272|0.00084511|
|G2-RUDAG-Full|score_permuted|+0.1149|1|0 / 1|0.00121793|0.00114041|
|G2-RUDAG-Full|direct_off|+0.0000|0|0 / 0|0.000978417|0.00508678|

## Audit and interpretation

Independent readback passed 20/20 best checkpoints. Each loaded checkpoint reproduced stored metrics and hard predictions; probability vectors matched within 1e-8. G1/G2 had identical initialization and optimizer-state hashes in every fold. Shared-epoch RNG matches: G1/G2 3532/3532; versus E74 6851/6851 across both groups.
G2−E74 nine-fold ACC is +0.2554 pp; G1−E74 nine-fold ACC is +0.1277 pp. G2−E62-E2 historical ten-fold ACC is +0.0000 pp. The E62/E69/E74 rows differ in training lineage or include the deliberate fold-2 label exposure, so only the same-checkpoint interventions and G1/G2 retraining are direct controls.
Fold 2 intentionally trains on rows 377 (ASD) and 407 (TD) while retaining them in test. Their relation pairs and classification labels enter training; the full-fold result is contaminated. The nine-fold table removes fold 2 only and still retains test-ACC checkpoint selection, pre-CV feature selection/standardization and transductive use of the 871-node graph.
RUDAG final decision: **PARTIAL**. G2 reaches the E62-E2 ten-fold ACC reference (91.8443%) and exceeds E74 by +0.2554 pp on the nine folds excluding Fold 2. G2 versus G1 is -0.1149 pp in all ten folds and +0.1277 pp after excluding Fold 2. The disease-neutral and score-permuted interventions each change one hard prediction, both in Fold 8; that one changed prediction remains in the nine-fold subset. This is measurable but sparse evidence for score correspondence. Direct-off changes no hard predictions and lowers probability-AUC by 0.0477 pp on average (full minus direct-off), so the direct path has no net hard-classification gain and slightly worsens mean probability-AUC. Its CORR fused message norm is 0.5061% of the raw input, compared with 0.0009% for the fusion-weighted CORR GCN message, but that larger representation change did not improve hard classification. Hard-AUC and sensitivity recover part of E74's decline, while probability-AUC is -0.3250 pp versus E74 and G2 sensitivity remains -0.7439 pp versus E62-E2. The runner's mechanical decision flag counted any nonzero intervention flip as positive mechanism evidence; the scientific conclusion here requires meaningful, consistent support. Since the observed score-correspondence effect occurs on only one of ten folds and Direct-off produces no hard-prediction changes, the evidence supports PARTIAL rather than GO. These results do not imply unbiased external generalization.

## Next question

The next scientifically valuable test is whether the observed CORR score correspondence remains useful under an evaluation protocol with fold-local preprocessing and a separate validation set for checkpoint selection. The present result should not trigger within-experiment tuning of the fixed 0.8 edge strength, 0.02 direct scale, WD, learning rates or Top-k.

## Files and sources

- Experiment directory: `/model/spromptgl/experiment/experiment74_1_rudag`
- Formal checkpoints: `.runtime/checkpoints/*.best.pt`; recovery states are removed after a complete audit.
- Core outputs: `results/fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `mechanism.csv`, `interventions.csv`, `checkpoint_audit.csv`, `manifest.json`.
- SPromptGL paper: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf
- BrainPrompt paper: https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf
- BrainPrompt official source: https://github.com/AngusMonroe/BrainPrompt
- SABER paper: https://arxiv.org/pdf/2607.01901
- Locked E69.3-G2 source commit: https://github.com/tangaii/SPromptGL/tree/41dbcb45a0273673857051d8e821b8c3315eef00/experiment/experiment69_3_g2_two_rows
