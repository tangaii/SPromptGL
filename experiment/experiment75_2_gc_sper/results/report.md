# Experiment 75.2 — GC-SPER

Status: **COMPLETE_AUDITED**; tasks 30/30; independent audits 30/30.
Directory: `/model/spromptgl/experiment/experiment75_2_gc_sper`. Device `None`; worker cap 2, observed peak 10; wall time 0.0s; retries 0; OOM 0.
E74.4-G2 source manifest SHA: `d3a60ecdfd0158be1eecb066d451e0246a417f69e17c4975fe022075a37ac9c4`; current source digest `a5f4b57e5ba16bf4fc8509c04d092f0670488ee0b872a795cec129d44246264a`.

## Method and audit boundaries
- E74.4-G2 is the only training base. Every task restored the corresponding E61 epoch-10 warmup through E74.4 `prepare(fold)` and called E74.4 `forward_state` through its own runner. The E74.4 best checkpoint was audited as a reference and was not used for fine-tuning.
- The original DARG produces 320-D F and the original CAM runs once. Hooks capture F at the CAM input and replace only its 144-D Outputlayer input; the Outputlayer also runs once. No second graph, GCN, CAM, or classifier was introduced.
- Prototypes use only eligible outer-fold training nodes. Five deterministic class-stratified blocks provide leave-block-out prototypes for training patients; test nodes use the full training-only bank. Fold-2 rows 377/407 were excluded from new prototype labels and auxiliary losses, while the inherited base protocol still exposes them to its original train/test overlap.
- G1 uses prototypical evidence only; G2 adds eight original frozen SPromptGL prompts with F320-conditioned class-specific FiLM, evidence mixture and alignment loss; G3 replaces projected semantic vectors by a post-projection orthogonal rotation. Its fixed 8x8 Gram is checked below. This is a single geometry-matched anonymous transform, not a general proof of medical semantics.
- SPromptGL's learned 4×4×9 feature partition does not identify 16 anatomical ROIs. Prompt/prototype features are not lesion maps. The paper distinguishes GFR and PLN components; the E74.4 baseline uses the audited `PLNNoPEN` path, and this module is inserted before its Outputlayer rather than restoring original PEN.
- Related work already combines prompts and prototypes, including HP²L; the candidate distinction here is the joint use of training-fold token prototypes, DARG-conditioned disease prompts, cross-fitted evidence reliability, and CAM feature correction. Prompt/prototype combination alone is not claimed as novel.

## Classification results
Mean ± population SD; percentages. Hard-AUC is the historical hard-prediction/BA convention; Prob-AUC uses continuous ASD probabilities.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2 historical|92.1891 ± 2.7648|92.0294 ± 2.7255|93.4946 ± 2.9195|90.0671 ± 3.8862|93.9917 ± 4.7320|91.4590 ± 2.9013|
|G1-Proto|91.7294 ± 3.2080|91.5727 ± 3.1557|93.5309 ± 2.9760|89.5793 ± 3.4917|93.5661 ± 4.7465|90.9705 ± 3.3699|
|G2-GC-SPER|91.4995 ± 2.9239|91.2722 ± 2.9383|93.6893 ± 2.9162|88.3354 ± 4.3317|94.2091 ± 3.9922|90.5830 ± 3.1956|
|G3-Anonymous|91.7294 ± 3.0388|91.5166 ± 3.1011|93.1560 ± 3.1692|88.5976 ± 4.7003|94.4357 ± 3.0615|90.8125 ± 3.4449|

## Paired comparisons
- G2−E74.4-G2 ACC: all folds -0.6897 pp (0/4/6); excluding Fold 2 -0.6386 pp (0/4/5).
- G2−G1 ACC: all folds -0.2299 pp (1/6/3); excluding Fold 2 -0.2554 pp (1/5/3).
- G2−G3 ACC: all folds -0.2299 pp (1/6/3); excluding Fold 2 -0.1277 pp (1/6/2).
- Full metric-by-metric deltas and per-fold W/T/L are in `paired_comparisons.csv`.

## Mechanisms and interventions
- G2 mean alpha `0.17885`, mean gamma `0.09426`, mean residual/CAM ratio `1.0240%`.
- `module_off` caused 0 hard prediction flips across the 10 G2 checkpoints. This is fixed-checkpoint reliance, not a retrained ablation or historical-checkpoint recovery.
- `semantic_off`, `graph_context_off`, `agreement_off`, and `prompt_class_swap` are separately reported per fold and group in `interventions.csv`; no outcomes are pooled across groups.
- Per-fold prototype separation, semantic/prototype probability difference and agreement, reliability by 4-token modality block, query/context variation, losses and residual magnitudes are in `mechanism.csv`.

## Evaluation limits and decision
- The historical protocol retains `val=test` checkpoint selection, RFE/standardization before CV, a transductive 871-node graph and Fold-2 exposure of 1-based rows 377 (ASD) and 407 (TD). Excluding Fold 2 is descriptive and does not remove the other limitations.
- Historical E74.4-G2 is a read-only paired reference, not a randomized external test. A result here cannot support an unbiased independent-generalization claim.
- Independent checkpoint readback: 30/30; hard predictions exact and maximum probability replay error `0`.
- Scientific decision: **NO-GO** under the preregistered criteria. G2 ACC change vs base is -0.6897 pp (0/4/6); G2 module-off hard flips: 0.
- Report: `/model/spromptgl/experiment/experiment75_2_gc_sper/results/report.md`; manifest: `/model/spromptgl/experiment/experiment75_2_gc_sper/manifest.json`; audit: `/model/spromptgl/experiment/experiment75_2_gc_sper/results/audit_status.json`.

## Excluding Fold 2 (descriptive nine-fold subset)

| Group | ACC | Hard-AUC | Prob-AUC | SEN | SPE | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|
|E74.4-G2 historical|92.2153 ± 2.9131|92.0481 ± 2.8723|93.6142 ± 3.0541|90.0474 ± 4.0959|94.0487 ± 4.9847|91.4703 ± 3.0580|
|G1-Proto|91.8321 ± 3.3658|91.6762 ± 3.3102|93.6664 ± 3.1076|89.7764 ± 3.6274|93.5759 ± 5.0032|91.0783 ± 3.5358|
|G2-GC-SPER|91.5767 ± 3.0724|91.3276 ± 3.0922|93.8070 ± 3.0514|88.1233 ± 4.5165|94.5318 ± 4.0825|90.6207 ± 3.3663|
|G3-Anonymous|91.7044 ± 3.2022|91.4784 ± 3.2666|93.1615 ± 3.3406|88.4146 ± 4.9207|94.5421 ± 3.2096|90.7518 ± 3.6262|

## Complete paired metrics

| Comparison | Scope | Metric | Δ pp | W/T/L |
|---|---|---|---:|---:|
|G2 vs historical base|all_10|acc|-0.6897|0/4/6|
|G2 vs historical base|all_10|hard_auc|-0.7572|1/3/6|
|G2 vs historical base|all_10|prob_auc|+0.1947|7/0/3|
|G2 vs historical base|all_10|sen|-1.7317|1/5/4|
|G2 vs historical base|all_10|spe|+0.2174|2/5/3|
|G2 vs historical base|all_10|asd_f1|-0.8761|1/3/6|
|G2 vs historical base|exclude_fold2_descriptive_9|acc|-0.6386|0/4/5|
|G2 vs historical base|exclude_fold2_descriptive_9|hard_auc|-0.7205|1/3/5|
|G2 vs historical base|exclude_fold2_descriptive_9|prob_auc|+0.1928|6/0/3|
|G2 vs historical base|exclude_fold2_descriptive_9|sen|-1.9241|1/4/4|
|G2 vs historical base|exclude_fold2_descriptive_9|spe|+0.4831|2/5/2|
|G2 vs historical base|exclude_fold2_descriptive_9|asd_f1|-0.8496|1/3/5|
|G2 vs G1 prototype-only|all_10|acc|-0.2299|1/6/3|
|G2 vs G1 prototype-only|all_10|hard_auc|-0.3005|2/2/6|
|G2 vs G1 prototype-only|all_10|prob_auc|+0.1584|7/0/3|
|G2 vs G1 prototype-only|all_10|sen|-1.2439|3/3/4|
|G2 vs G1 prototype-only|all_10|spe|+0.6429|3/4/3|
|G2 vs G1 prototype-only|all_10|asd_f1|-0.3875|2/2/6|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|acc|-0.2554|1/5/3|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|hard_auc|-0.3486|1/2/6|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|prob_auc|+0.1406|6/0/3|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|sen|-1.6531|2/3/4|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|spe|+0.9559|3/4/2|
|G2 vs G1 prototype-only|exclude_fold2_descriptive_9|asd_f1|-0.4577|1/2/6|
|G2 vs G3 anonymous semantics|all_10|acc|-0.2299|1/6/3|
|G2 vs G3 anonymous semantics|all_10|hard_auc|-0.2444|3/4/3|
|G2 vs G3 anonymous semantics|all_10|prob_auc|+0.5333|6/1/3|
|G2 vs G3 anonymous semantics|all_10|sen|-0.2622|2/5/3|
|G2 vs G3 anonymous semantics|all_10|spe|-0.2266|2/5/3|
|G2 vs G3 anonymous semantics|all_10|asd_f1|-0.2295|3/4/3|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|acc|-0.1277|1/6/2|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|hard_auc|-0.1508|3/4/2|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|prob_auc|+0.6455|6/1/2|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|sen|-0.2913|2/4/3|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|spe|-0.0103|2/5/2|
|G2 vs G3 anonymous semantics|exclude_fold2_descriptive_9|asd_f1|-0.1312|3/4/2|

## RNG and attempt audit

- Corrected formal run: exact shared RNG hashes on 3470 common fold-epoch observations; E74.4 comparisons 31230/31230; three-group comparisons 10410/10410; warmup-to-first-epoch parity 10/10.
- The first 30-task attempt was disqualified after discovering CUDA RNG contamination; it is preserved in `.runtime/attempt_1_rng_contamination/`. The corrected attempt had a BrokenProcessPool after 28 tasks; two tasks resumed from latest checkpoints. Root cause is unconfirmed. No OOM was observed. Accepted-run elapsed training time including resume: 2116.93s; discarded attempt: 1189.08s.
- The module initialization fix preserves the active CUDA RNG by including the current accelerator in `torch.random.fork_rng`; the corrected preflight confirmed the full RNG SHA is unchanged by module construction.

## Mechanism interpretation

| Group | alpha | gamma | Residual/CAM | Reliability | Sem/Proto agreement | p_sem/p_proto MAE | Graph-context SD |
|---|---:|---:|---:|---:|---:|---:|---:|
|G1-Proto|0.2000|0.0945|1.222%|0.546|0.462|0.263|0.657|
|G2-GC-SPER|0.1789|0.0943|1.024%|0.448|0.794|0.134|0.204|
|G3-Anonymous|0.1784|0.0944|1.050%|0.440|0.800|0.127|0.208|

Saved-bank class prototype separation (mean `1-cos(P_ASD,P_TD)` by token block):

| Group | PHENO | ANAT | FUNC | CORR |
|---|---:|---:|---:|---:|
|G1-Proto|0.3651|0.6954|0.6547|1.1302|
|G2-GC-SPER|0.3756|0.6749|0.6691|0.9297|
|G3-Anonymous|0.3400|0.6620|0.6576|1.0789|

| Group | Intervention | Hard flips | Correct→wrong | Wrong→correct | Mean probability Δ | Mean CAM Δ |
|---|---|---:|---:|---:|---:|---:|
|G1-Proto|module_off|2|2|0|0.000977|1.226%|
|G1-Proto|semantic_off|0|0|0|0.000000|0.000%|
|G1-Proto|graph_context_off|0|0|0|0.000000|0.000%|
|G1-Proto|agreement_off|0|0|0|0.000000|0.000%|
|G1-Proto|prompt_class_swap|0|0|0|0.000000|0.000%|
|G2-GC-SPER|module_off|0|0|0|0.000664|1.028%|
|G2-GC-SPER|semantic_off|0|0|0|0.000186|0.204%|
|G2-GC-SPER|graph_context_off|0|0|0|0.000164|0.222%|
|G2-GC-SPER|agreement_off|0|0|0|0.000129|0.223%|
|G2-GC-SPER|prompt_class_swap|0|0|0|0.000287|0.301%|
|G3-Anonymous|module_off|0|0|0|0.000752|1.053%|
|G3-Anonymous|semantic_off|0|0|0|0.000160|0.195%|
|G3-Anonymous|graph_context_off|0|0|0|0.000156|0.242%|
|G3-Anonymous|agreement_off|0|0|0|0.000139|0.209%|
|G3-Anonymous|prompt_class_swap|0|0|0|0.000264|0.330%|

- G2 residual changes CAM features in magnitude but causes no hard-label flips in module-off, semantic-off, graph-context-off, agreement-off, or prompt-class-swap tests. G1 module-off flips two predictions, both correct-to-wrong; G3 has no hard flips. This indicates weak fixed-decision dependence, not evidence that the semantic branch improves generalization.
- G2 is below the base in ACC, Hard-AUC, SEN, and F1; the modest Prob-AUC increase does not offset the classification tradeoff. G3's ACC is higher than G2 but remains below base; real text does not show an independent ACC advantage over the geometry-matched anonymous embeddings. The registered decision is **NO-GO**.
- Per-modality prototype class separation was independently computed from each saved checkpoint bank and added to `mechanism.csv`; the 16 slots remain learned latent features, not anatomical ROIs.

## Sources

- [SPromptGL MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf); [original SPromptGL repository](https://github.com/wanxixi11/SPromptGL).
- [BrainPrompt repository](https://github.com/AngusMonroe/BrainPrompt); [HP²L, Medical Image Analysis](https://doi.org/10.1016/j.media.2026.104063).
