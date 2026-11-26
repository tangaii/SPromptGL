# Experiment 74.2 — DCR-RUDAG

Status: **COMPLETE_AUDITED**; independent checkpoint audit 10/10; device `PPU-ZW810E`; maximum workers 10; formal training wall time 697.2s; current-run retries 0; prior failed worker starts 20; OOM events 0.

## Source and protocol audit

The MICCAI 2025 SPromptGL paper's Figure 1 / Sections 2.1–2.2 / Eqs. (1)–(7) describe GFR graph learning and fusion, CAM, and semantic PEN. The implementation used here retains the verified root behavior: learned linear projection, cosine similarity with a detached `score > 0.9` support mask, two GraphConv layers with an intervening ReLU, one symmetric `normalize_adj(A+I)`, original four GFR weights, CAM, and the 144-D classifier input. The local core files (`main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`) match GitHub `main` at `47b992ad6e11c58b15853cecb03384c0bff372a4` byte-for-byte; workspace HEAD is `fa1570f45dea5c3027a89ccc93aef3060657bacd`. Eq. (1) writes a label-derived supervision-graph term and Eq. (7) includes the paper's full `L_G`; active E61/E74 training instead uses `GraphConstructLoss(S_m)`, with E74.1 adding the verified `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve` objective. ALoss is absent from that actual training total and was not restored. E74.1 source files match pinned GitHub commit `d1522a23682b62e41d899d95c9721a68d3828456` byte-for-byte.

E53.3 DARG remains unchanged: four 32-D relation encoders; 65-D symmetric pair features; 65→32→1 relation scorers; 4,096 balanced training pairs; CORR-anchored conditional adversarial alignment; and relation preservation. The original uncalibrated sigmoid relation score is used. Total loss remains `L_cls + L_graph + 0.05 L_pair + 0.01 L_adv + 0.01 L_preserve`.

The sole method change from E74.1 is the bounded direct path. E74.1 G1 propagates the ordinary unified-graph neighbor difference; DCR-RUDAG forms `DeltaA=A_unified-S`, excludes its diagonal, and computes `(DeltaA_off@X - row_sum(DeltaA_off)*X)/clamp_min(row_sum(S_off),1)` before the existing per-block CADR RMS-tanh bound and fixed `0.02*g` scaling. This residual is signed and differentiable; the GCN still uses only `A_unified` and is called once per modality. With `q=0.5`, `DeltaA`, bounded residual and direct message are exactly zero.

BrainPrompt's MICCAI 2025 paper and official `brainpromptg_net.py`, `brainpromptc_net.py`, and ABIDE label prompts were checked at commit `c85304a023636a2701d465e4cb0acb13b637127e` and their three SHA-256 values are pinned in `config.json`. Disease-prompt loss compares normalized graph representations with linearly transformed class-text prototypes using cross-entropy; the current `brainpromptc_net.py` also references `hg` without defining it in `forward`, so it is treated only as a mechanism reference and is not imported. SABER v1 describes decision-level semantic alignment, `g'=g+sigmoid(W_g h_fused)⊙h_fused`, and disease-embedding contrastive loss; no official SABER code was confirmed. Neither method's text/semantic module is included in DCR-RUDAG.

Historical evaluation constraints are retained: `val_idx == test_idx`, best checkpoint by strict test-fold ACC, pre-CV RFE/standardization, and transductive use of all 871 graph nodes. Fold 2 includes 1-based rows 377 (ASD) and 407 (TD) in both train and test; this is an inherited label-exposure exception, not ordinary cross-validation. Nine-fold analysis excludes Fold 2 only and does not remove the other protocol limitations.

## Classification

All values are percent, 10-fold mean ± population SD. Hard-AUC uses hard predictions (equal to balanced accuracy for two classes); Probability-AUC uses continuous ASD probability. SEN is ASD recall and SPE is TD specificity. Historical groups are read-only descriptive references.

|Group|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|E69.3-G2 historical|92.0742 ± 2.7482|91.8995 ± 2.7186|93.3759 ± 2.9908|89.5854 ± 3.4710|94.2137 ± 3.9815|91.2984 ± 2.9222|
|E62-E2 historical|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|89.5854 ± 3.4710|93.7789 ± 4.2653|91.0789 ± 3.0892|
|E74-UDAG historical|91.4995 ± 2.6885|91.2511 ± 2.7308|93.9249 ± 2.6632|87.8537 ± 4.3359|94.6485 ± 3.2152|90.5218 ± 3.0064|
|E74.1-G1 RUDAG-U historical|91.9592 ± 2.7747|91.7498 ± 2.7845|93.4439 ± 3.0255|89.0732 ± 3.9177|94.4265 ± 3.7676|91.1148 ± 3.0257|
|E74.1-G2 RUDAG-Full historical|91.8443 ± 2.9793|91.6363 ± 3.0151|93.5999 ± 3.0799|88.8415 ± 4.3351|94.4311 ± 3.4953|90.9690 ± 3.3140|
|E74.2-G1 DCR-RUDAG|91.8443 ± 2.7963|91.6442 ± 2.7336|93.6854 ± 2.7687|89.0793 ± 3.0035|94.2091 ± 4.3328|91.0361 ± 2.9228|

Fold 2 alone: ACC/Hard-AUC/Prob-AUC/SEN/SPE = 90.8046/90.6416/92.5769/87.8049/93.4783%; its 1-based rows 377 (ASD) and 407 (TD) are in both train and test.
Excluding Fold 2, DCR-RUDAG mean ± population SD is ACC 91.9598±2.9248% / Hard-AUC 91.7556±2.8598% / Prob-AUC 93.8086±2.8923% / SEN 89.2209±3.1341% / SPE 94.2903±4.5600%. This removes only the known two-row exposure and retains test-ACC checkpoint selection, pre-CV preprocessing, and transductive graph use.

Execution audit: the first launch ended in 20 worker invocations across ten folds because a diagnostics-only `autograd.grad` call reused the graph after `total.backward()`. All failures occurred before optimizer steps; no epoch update, checkpoint, or task result was written. The diagnostic now reads the existing gradient buffers, and all ten folds were relaunched independently from the E61 epoch-10 warmups. The failed logs remain under `.runtime/logs/` and are excluded from the result tables.

## Paired comparisons

Differences are DCR-RUDAG minus the named history in percentage points. `all10` and `nine_without_fold2` are descriptive paired comparisons; they are not unbiased generalization estimates.

|Comparison|Scope|ΔACC|ΔHard-AUC|ΔProb-AUC|ΔSEN|ΔSPE|
|---|---|---:|---:|---:|---:|---:|
|E74.2−E74.1-G1|all10|-0.1149 (2/6/2)|-0.1056 (4/4/2)|+0.2415 (5/0/5)|+0.0061 (4/4/2)|-0.2174 (1/7/2)|
|E74.2−E74.1-G1|nine_without_fold2|+0.1277 (2/6/1)|+0.1536 (4/4/1)|+0.3390 (5/0/4)|+0.5488 (4/4/1)|-0.2415 (1/6/2)|
|E74.2−E74.1-G2|all10|+0.0000 (1/8/1)|+0.0079 (3/4/3)|+0.0855 (5/0/5)|+0.2378 (3/4/3)|-0.2220 (3/5/2)|
|E74.2−E74.1-G2|nine_without_fold2|+0.0000 (1/7/1)|+0.0088 (3/3/3)|+0.0773 (4/0/5)|+0.2642 (3/3/3)|-0.2467 (3/4/2)|
|E74.2−E74|all10|+0.3448 (3/7/0)|+0.3931 (5/4/1)|-0.2395 (4/0/6)|+1.2256 (4/5/1)|-0.4394 (2/6/2)|
|E74.2−E74|nine_without_fold2|+0.2554 (2/7/0)|+0.3013 (4/4/1)|-0.2072 (4/0/5)|+1.0908 (3/5/1)|-0.4882 (2/5/2)|
|E74.2−E69.3-G2|all10|-0.2299 (0/8/2)|-0.2554 (1/6/3)|+0.3095 (7/0/3)|-0.5061 (1/7/2)|-0.0046 (2/6/2)|
|E74.2−E69.3-G2|nine_without_fold2|-0.2554 (0/7/2)|-0.2837 (1/5/3)|+0.3203 (6/0/3)|-0.5623 (1/6/2)|-0.0051 (2/5/2)|
|E74.2−E62-E2|all10|+0.0000 (1/7/2)|-0.0380 (2/5/3)|+0.4844 (7/0/3)|-0.5061 (1/7/2)|+0.4302 (3/5/2)|
|E74.2−E62-E2|nine_without_fold2|-0.2554 (0/7/2)|-0.2837 (1/5/3)|+0.3203 (6/0/3)|-0.5623 (1/6/2)|-0.0051 (2/5/2)|

## Mechanism

Mechanism values below are best-checkpoint averages across ten folds; exact fold×modality diagnostics are in `mechanism.csv`.

|Mode|Edges scored/valid|Changed edges|ΔA/S|Fusion GCN/input %|Bounded residual/input %|Fusion DCR/input %|GCN/DCR norm ratio|Final gate|Train pair AUC|Unexposed test pair AUC|DCR-only scorer grad|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PHENO|3575.5/82505.1|2609.5|5.14431e-05|0.0115475|0.000778595|9.24603e-09|1.13195e+06|0.00100363|0.782146|0.502712|3.225e-11|
|ANAT|3680.0/137610.4|1052.8|1.17832e-05|0.0100948|0.000733725|8.40604e-09|1.5593e+06|0.00100373|0.598817|0.501079|2.907e-11|
|FUNC|3603.0/69223.7|1861.8|3.27213e-05|0.000598975|0.000940221|1.10287e-08|64866.4|0.00100367|0.649473|0.502748|3.820e-11|
|CORR|47625.9/47625.9|47625.9|0.651513|0.000700645|13.2682|0.156809|0.00435608|0.991963|0.994828|0.832641|3.833e-04|

The fusion-weighted DCR residual/input ratios are PHENO `0.000000%`, ANAT `0.000000%`, FUNC `0.000000%`, CORR `0.156809%`. These norm changes show message magnitude only; classification effects are evaluated separately below.

|Intervention|ΔACC pp|ΔHard-AUC pp|ΔProb-AUC pp|ΔSEN pp|ΔSPE pp|Flips|Corrected / newly wrong|Mean abs ΔP(ASD)|Mean relative ΔF|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|disease_neutral|+0.0000|+0.0000|-0.0267|+0.0000|+0.0000|0|0 / 0|0.00136565|0.00156797|
|corr_neutral|+0.0000|+0.0000|-0.0267|+0.0000|+0.0000|0|0 / 0|0.00136565|0.00156797|
|score_permuted|+0.0000|+0.0000|-0.0266|+0.0000|+0.0000|0|0 / 0|0.00139347|0.00196156|
|direct_off|+0.0000|+0.0000|-0.0267|+0.0000|+0.0000|0|0 / 0|0.00136578|0.00156796|

DCR-RUDAG final decision: **PARTIAL**. Disease-score intervention support by the predefined evidence check: `False`. The E74.2−E74.1-G1 nine-fold ACC difference is `+0.1277` pp. Against E74.1-G2's ordinary neighbor difference, ACC is `+0.0000` pp on all folds and nine folds; direct-off has zero hard flips and improves Prob-AUC by only `0.0267` pp over the full path. DCR therefore does not demonstrate a net hard-classification gain over ordinary neighbor aggregation. The same-checkpoint result must be read alongside the independently trained E74.1 comparisons.
Checkpoint/RNG audit: `10/10` best checkpoints independently reloaded and reproduced; DCR versus E74.1-G1 shared-epoch RNG snapshots match `3530/3530`. Both interventions and 9-fold metrics preserve the Fold 2 and historical protocol caveats.

## Next step

If DCR residual or score correspondence has limited effect, do not change the fixed 0.8 reweighting, 0.02 coefficient, gate initialization, WD, learning rates, Top-12 set, or fold protocol from these results. The next test should use fold-local feature selection/standardization and a separate validation set, then examine whether disease-edge increments generalize beyond this transductive endpoint.

## Artifacts

- Experiment directory: `/model/spromptgl/experiment/experiment74_2_dcr_rudag` (preferred OSSFS path failed with errno 5).
- E74.1 reference: GitHub branch `upload/experiment74-1-rudag`, commit `d1522a23682b62e41d899d95c9721a68d3828456`; local code hashes are verified in `manifest.json`.
- Training sources remained byte-identical to the passing preflight. The independent audit utility was repaired after training to retain the graph across DCR-only and full-classification gradient checks; training source SHA remains `50dcd49f02c441a1e91ca2651873cbc07af270c16bca06044702b599155d3999`, and the audit utility SHA used is `6ae53ad1174b9cf7e7f3a4fa19b9f4bc47d4a311988161c9df77a0c333f019fc`.
- Outputs: `results/fold_metrics.csv`, `summary.csv`, `paired_comparisons.csv`, `mechanism.csv`, `interventions.csv`, `checkpoint_audit.csv`, `audit_status.json`, and `manifest.json`; ten best checkpoints, task records, and audit logs are retained, recovery checkpoints and bytecode caches were removed.
- Source basis: SPromptGL MICCAI 2025 paper; BrainPrompt MICCAI 2025 paper and official source at the pinned repository snapshot; SABER arXiv paper only, with no official-source audit claimed.
