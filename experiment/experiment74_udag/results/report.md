# Experiment 74 — UDAG

Status: **COMPLETE_AUDITED**; 10/10 trained and independently rebuilt/read back. Scientific judgment: **NO-GO**.

## Source and protocol audit

Read-only E69.3-G2 commit `41dbcb45a0273673857051d8e821b8c3315eef00` on `upload/experiment69-3-g2-two-rows`; local runner SHA `be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44` exactly matches the raw GitHub file at that commit. Its source loader validates E61, E62 and all five E53.3 dependency sources against historical manifests. The OSS experiment-directory creation and Git object fetch both encountered errno-5 I/O errors; all new work uses the configured local root. No historical experiment, root model or data files were modified.

SPromptGL Section 2.1 defines modality-specific patient edges, propagation on the concatenated input, then fusion. The actual historical implementation is retained: GraphLearn applies a learned linear projection, normalized cosine, threshold >0.9; GCN is two GraphConv layers with one intermediate ReLU, no dropout. Original GraphLearn includes its native diagonal; the historical `normalize_adj(A+I)` adds identity once. UDAG preserves that exact convention. Paper Eq.1 contains label-graph supervision not restored by this experiment: the current historical graph loss stays unchanged. CAM/OutputLayer are untouched; 320-D GFR output becomes four 36-D CAM positions (144-D classifier input); OutputLayer reduces to 36-D then two-class log-probabilities and does not use its `attn_embedding` argument.

DARG uses the same modality GCN four times: original graph, positive graph, negative graph, identity subtraction. Its signed message combines positive and negative messages plus direct difference; adaptive mixing retains original propagation. UDAG removes those separate message branches, direct differences, and adaptive message fusion from the formal forward. Compatibility `adaptive_gate_logits` remains in state_dict and optimizer groups but is frozen, unused, and never acquires optimizer state; four existing edge gates remain trainable. The original fusion_weight is retained.

Candidate support is the unchanged `_candidate_support`: original positive off-diagonal edges; fallback similarity candidates only for historically enabled modalities; mutual Top-12; upper-triangle scoring of symmetric 65-D pair features. Mutual Top-12 selects reweighting locations, not deletion of all other original edges. Selected original positive edges use A[1+g(2q−1)], negative edges use A[1−g(1−2q)], uncertain edges and unselected original edges stay unchanged. A selected fallback edge with A=0 is added only for q≥0.55 with similarity×2(q−0.5)×g. Scatter-add uses symmetric deltas and intact autograd, no `.data` update. q=sigmoid(logit), no Platt or OOF. Original adjacency A, not modified adjacency, is supplied to GraphConstructLoss.

The four relation encoders, independent 65→32→1 scorers, CORR-detached conditional modality adversary, GRL=0.03, fixed 4096 balanced training pairs and original total loss remain. CORR is detached only in the adversarial branch, not relation supervision or classification propagation. This is conditional distribution alignment, not an explicit pairwise teacher-consistency objective.

All ten E61 epoch-10 warmups are read-only reused. All shared model parameters, three optimizer states and Python/NumPy/CPU/CUDA RNG states are inherited. E69.3-G2 gate initialization is copied deterministically before freezing the unused adaptive gate. No per-epoch reseeding and no extra stochastic operation are introduced. A controlled one-update save/reload test matches parameters, optimizer states, outputs and RNG exactly.

**Evaluation limits:** fold 2 deliberately adds 1-based rows 377 (ASD) and 407 (TD) to training while keeping them in test, and rebuilds pairs and class weights exactly as E69.3-G2. These are label-exposure results. val=test checkpoint selection, pre-CV RFE/standardization and the 871-node transductive graph remain. Results are paired exploratory mechanism diagnostics, not leakage-free or unbiased generalization. The separate nine-fold summary removes index-overlap fold 2 only; other historical evaluation limits remain. No post-hoc removal of the two exposed test rows is presented.

## Classification results

All values: 10-fold mean ± population SD, percent.

|Model|ACC|Hard-AUC|Prob-AUC|SEN-ASD|SPE-TD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|E69.3-G2|92.0742 ± 2.7482|91.8995 ± 2.7186|93.3759 ± 2.9908|89.5854 ± 3.4710|94.2137 ± 3.9815|91.8995 ± 2.7186|91.2984 ± 2.9222|
|E74-UDAG|91.4995 ± 2.6885|91.2511 ± 2.7308|93.9249 ± 2.6632|87.8537 ± 4.3359|94.6485 ± 3.2152|91.2511 ± 2.7308|90.5218 ± 3.0064|

## Paired differences

Differences are E74 minus E69.3-G2, percentage points; 95% paired-t intervals are descriptive because CV training sets overlap and model selection uses test ACC.

|Scope|Metric|Mean Δ pp|W/T/L|Descriptive 95% interval|
|---|---|---:|---|---|
|all10|acc|-0.5747|0/5/5|[-1.0081, -0.1413]|
|all10|hard_auc|-0.6485|0/4/6|[-1.1054, -0.1915]|
|all10|prob_auc|+0.5489|7/0/3|[-0.0803, +1.1782]|
|all10|sen|-1.7317|0/5/5|[-3.1872, -0.2762]|
|all10|spe|+0.4348|2/7/1|[-0.7827, +1.6522]|
|all10|ba|-0.6485|0/4/6|[-1.1054, -0.1915]|
|all10|asd_f1|-0.7766|0/4/6|[-1.2863, -0.2670]|
|nine_no_index_overlap|acc|-0.5109|0/5/4|[-0.9765, -0.0452]|
|nine_no_index_overlap|hard_auc|-0.5850|0/4/5|[-1.0794, -0.0906]|
|nine_no_index_overlap|prob_auc|+0.5275|6/0/3|[-0.1876, +1.2425]|
|nine_no_index_overlap|sen|-1.6531|0/5/4|[-3.2995, -0.0067]|
|nine_no_index_overlap|spe|+0.4831|2/6/1|[-0.8988, +1.8650]|
|nine_no_index_overlap|ba|-0.5850|0/4/5|[-1.0794, -0.0906]|
|nine_no_index_overlap|asd_f1|-0.7082|0/4/5|[-1.2616, -0.1548]|

## Per-fold results

|Fold|ACC|Hard-AUC|Prob-AUC|SEN|SPE|Best epoch|Executed epochs|ΔACC pp|RNG matches/common|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|95.4545|95.1220|96.0042|90.2439|100.0000|123|324|+0.0000|314/314|
|2|89.6552|89.4221|93.1071|85.3659|93.4783|165|366|-1.1494|356/356|
|3|87.3563|87.1156|90.2439|82.9268|91.3043|179|380|+0.0000|368/368|
|4|94.2529|93.9362|95.6915|90.0000|97.8723|207|408|+0.0000|368/368|
|5|91.9540|92.1809|96.1170|95.0000|89.3617|278|450|+0.0000|408/408|
|6|95.4023|95.3723|97.1277|95.0000|95.7447|109|310|-1.1494|297/297|
|7|90.8046|90.1862|95.3191|82.5000|97.8723|100|301|-1.1494|291/291|
|8|88.5057|88.2447|88.3511|85.0000|91.4894|179|380|+0.0000|370/370|
|9|89.6552|89.3085|92.9255|85.0000|93.6170|103|304|-1.1494|294/294|
|10|91.9540|91.6223|94.3617|87.5000|95.7447|161|362|-1.1494|352/352|

## Graph and message mechanism

Quantities below average ten best checkpoints; exact per-fold/per-modality counts and gradients are in mechanism.csv. Relative message differences can be large when the denominator is nearly zero; raw-input-normalized absolute effects are also reported.

|Modality|Original edges|Candidates|Positive|Negative|Uncertain|Fallback added|Edge gate|Adjacency relative Δ|Original GCN/input %|UDAG GCN/input %|Weighted message Δ/input %|Scorer cls grad|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PHENO|78979.9|3520.7|1255.5|1090.9|1174.3|0.0|0.50117|0.026031|0.00413343|0.0041332|5.80032e-06|7.582e-08|
|ANAT|127534.6|3700.4|395.4|388.8|2916.2|0.0|0.50027|0.0057221|0.000886649|0.000886653|1.00945e-06|7.273e-09|
|FUNC|67326.1|3567.7|964.8|501.5|2101.4|0.0|0.49919|0.014718|0.000989194|0.000989263|1.31207e-06|1.821e-08|
|CORR|57564.2|3369.5|1796.4|1502.4|70.7|0.0|0.50079|0.12974|0.00221416|0.00221359|3.29626e-05|1.977e-07|

Training and unexposed-test relation AUC/PR-AUC are descriptive checkpoint diagnostics in mechanism.csv. The latter excludes any training-index overlap from relation evaluation, but does not remove historical test-based checkpoint selection. There is no independently retrained pre-alignment relation baseline, so these results do not isolate an adversarial-alignment improvement.

## Fixed-checkpoint interventions

Full minus intervention ACC: positive values favor UDAG. Hard-prediction flips, corrections and new errors are counted on each fold's complete historical test set.

|Fold|Intervention|Full−off ACC pp|Hard flips|Full correct / off wrong|Full wrong / off correct|Mean abs Δ probability|
|---:|---|---:|---:|---:|---:|---:|
|1|disease_off|+0.0000|0|0|0|3.8723968e-08|
|1|positive_off|+0.0000|0|0|0|3.3845108e-08|
|1|negative_off|+0.0000|0|0|0|3.0521981e-08|
|2|disease_off|+0.0000|0|0|0|2.4748988e-06|
|2|positive_off|+0.0000|0|0|0|2.3336268e-06|
|2|negative_off|+0.0000|0|0|0|1.7914418e-06|
|3|disease_off|+0.0000|0|0|0|1.2518689e-06|
|3|positive_off|+0.0000|0|0|0|7.5979864e-07|
|3|negative_off|+0.0000|0|0|0|8.6562687e-07|
|4|disease_off|+0.0000|0|0|0|3.3401295e-07|
|4|positive_off|+0.0000|0|0|0|2.0550114e-07|
|4|negative_off|+0.0000|0|0|0|2.5621006e-07|
|5|disease_off|+0.0000|0|0|0|2.5820116e-08|
|5|positive_off|+0.0000|0|0|0|2.1259847e-08|
|5|negative_off|+0.0000|0|0|0|2.1516763e-08|
|6|disease_off|+0.0000|0|0|0|1.8144734e-08|
|6|positive_off|+0.0000|0|0|0|1.8465879e-08|
|6|negative_off|+0.0000|0|0|0|1.6132219e-08|
|7|disease_off|+0.0000|0|0|0|3.3094064e-08|
|7|positive_off|+0.0000|0|0|0|2.5991394e-08|
|7|negative_off|+0.0000|0|0|0|2.525811e-08|
|8|disease_off|+0.0000|0|0|0|4.9511033e-07|
|8|positive_off|+0.0000|0|0|0|1.1354642e-07|
|8|negative_off|+0.0000|0|0|0|4.7086385e-07|
|9|disease_off|+0.0000|0|0|0|1.0906108e-07|
|9|positive_off|+0.0000|0|0|0|7.3531666e-08|
|9|negative_off|+0.0000|0|0|0|1.2994626e-07|
|10|disease_off|+0.0000|0|0|0|4.7224048e-06|
|10|positive_off|+0.0000|0|0|0|2.3977382e-06|
|10|negative_off|+0.0000|0|0|0|3.1748145e-06|

## Efficiency, audit and decision

Formal GCN calls decrease from 16 to 4 per full graph forward (75% fewer GCN invocations), with no additional GCN, no positive/negative message branches or direct-difference computation. This does not mean 75% total runtime reduction: GraphLearn, relation scoring, CAM and auxiliary losses remain. Four adaptive parameters are frozen; compatible stored parameter count is unchanged. Runtime: `{"device": "PPU-ZW810E", "single_device": true, "max_workers_observed": 10, "worker_limit": 10, "formal_training_failures": 0, "formal_training_retries": 0, "oom_events": 0, "warmup_retrained": false, "audit_restarted_after_explicit_module_path_fix": true, "provenance": "Training controller manifest reported maximum 10 live workers; controller.log records RAMP workers=10. All ten task JSONs are COMPLETE.", "audit_only": true}`. Ten independent workers had separate models, optimizers and RNG. Peak allocated bytes and task wall times are in manifest.json.

All 10 independently rebuilt checkpoints reproduce hard predictions exactly and probabilities within 1e-8. All ten inheritance audits pass. 3418/3418 common epoch RNG triplets match E69.3-G2. Each formal forward has one GCN per modality and one CAM call. 97 protected source/data/history/checkpoint files retain their SHA. Synthetic graph cases, natural and controlled classification gradients, frozen adaptive gate, and save/reload one-step parity pass.

**NO-GO**. Paired ACC Δ=-0.5747 pp; nine-fold Δ=-0.5109 pp. Disease-reweighting off changes 0 hard predictions across 10 folds; mean full−off ACC=+0.0000 pp. The maximum weighted message difference/input ratio is 2.5992e-06. Interpretation must combine this absolute contribution with edge-weight changes; nonzero edges or gradients alone do not establish diagnostic benefit. GFR WD=0.11 remains fixed and can suppress GCN message amplitudes.

UDAG should replace DARG for diagnostic efficacy only if its paired performance and fixed-checkpoint interventions support it. A simpler graph pathway may still be an engineering simplification, but efficiency alone does not justify a new disease-aware method or disease-semantic module. Further semantic supervision is not established by this experiment; a future study would need a clean evaluation protocol and demonstrable disease-graph contribution.

## Sources and files

- Official paper: https://papers.miccai.org/miccai-2025/0861-Paper5129.html
- Author code: https://github.com/wanxixi11/SPromptGL
- Locked base: https://github.com/tangaii/SPromptGL/tree/41dbcb45a0273673857051d8e821b8c3315eef00/experiment/experiment69_3_g2_two_rows
- Experiment root: `/model/spromptgl/experiment/experiment74_udag`
- config.json; run.py; udag.py; audit.py; manifest.json
- results/fold_metrics.csv; summary.csv; paired_comparisons.csv; mechanism.csv; audit_status.json
- .runtime/checkpoints: best and resumable states; historical warmups are read-only referenced.

## Concrete interpretation and audit recovery

The unified graphs are valid and the intended structural simplification succeeded. Disease reweighting, positive enhancement and negative suppression each cause zero hard-prediction flips over all ten best checkpoints. Improved probability AUC accompanies lower ACC, Hard-AUC and ASD sensitivity. The nine folds without index overlap also lose ACC; the deficit is not confined to fold 2.

Retain DARG as the historical comparison base for now; do not adopt UDAG as a demonstrated efficacy improvement. This does not establish historical DARG disease-message efficacy either. A disease-semantic extension of UDAG is not currently supported by these results.

A plausible bottleneck is the small GCN signal under inherited WD=0.11: graph changes act only through those messages after removing the direct-difference pathway. Candidate selection alters a subset of denser original graphs. These are possible explanations, not a causal ablation of weight decay or direct differences; neither was changed here. CORR relation quality does not by itself establish propagated diagnostic benefit.

Hard thresholds and mutual Top-12 define non-differentiable support. Classification gradients flow through active continuous edge weights, the existing edge gates and relation networks, not through topology selection. The natural initial warmup has inactive disease edges; isolated controlled q=0.9 tests prove the gradient path without changing formal thresholds.

Preflight repaired a boolean-degree statistic and optimizer-snapshot aliasing before formal training. After 10/10 training tasks completed, same-name audit module shadowing was fixed using absolute-path loading. Original training sources were reconstructed byte-for-byte and SHA-verified. Trained runner functions remain AST-identical; the model forward, losses, optimizer steps and RNG logic were unchanged. Training and audit source hashes are recorded separately. No formal training task was repeated for these audit-harness repairs.

Single-seed paired intervals here are descriptive; overlapping CV training sets and historical test-based checkpoint selection do not support independent clinical significance claims.
