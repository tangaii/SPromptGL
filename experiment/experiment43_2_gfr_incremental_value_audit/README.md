# Experiment 43.2 — GFR Incremental Value Audit

## Frozen question and comparison

Does the original GFR improve classification over raw input when the physically PEN-free PLN, 320 input features, initialization, split and historical training protocol are held constant? Only seed 0 and two ten-fold groups are registered.

* E00_FULL_GFR_B0B_REPLAY: original four GraphLearn relations, four two-layer GCNs, original branch overwrite, weighted NLL plus mean four-graph regularization. ALoss is observed but excluded.
* E01_RAW_INPUT_NO_GFR: the identical PLN receives `agent.feat` directly; only weighted NLL is optimized. The initialized GFR and its Adam are retained as initialization scaffolding, frozen, never called and never stepped.

E00 is retrained and must pass all available E43.1/historical B0-B fields before E01 starts. Historical best checkpoints are used only for read-only final equality verification, never as training initializations. Ten **untrained** E43.1 anchors are verified against their historical signature, index, data/split, initial model and RNG hashes, then copied. Their old signatures are not expected to equal this experiment's signature. Both groups reconstruct the corresponding anchor and restore its full post-construction RNG. Folds 1–5 stay on physical GPU0 and 6–10 on GPU1. No new per-fold seed is used. At every shared training epoch both groups must consume identical four-family RNG streams; different stopping epochs are allowed.

## Source and paper audit

The audit reads local executable sources, E43.1 run/integration/method/config/manifest/README/results, the original B0-B runner/results, and E43 graph-audit materials. File SHA256 values, Git HEAD and the original handoff prefix are in `manifest.json`. The cached original paper is `experiment/experiment43_corr_anchored_gfr_audit/.runtime/paper.pdf` and its extracted `paper.txt`; no model download or text encoder is used.

|Source / location|Actual evidence|Action in E43.2|
|---|---|---|
|`main.py:set_rng_seed`, `cross_validation`|Historical seeding, stratified folds, strict first improved test/validation ACC, `wait > early`|Preserve E43.1 split and selection logic|
|`model.py:EvalHelper.__init__`, `run_epoch(simple-2)`|GFR then PLN constructed; two Adam optimizers; PLN Xavier initialization after optimizer construction; weighted NLL + four graph losses|Reproduce constructor order and E00 update|
|`network.py:GraphLearn.forward`|Learned projection, cosine, detached hard support `score > 0.9`; raw adjacency returned|Unchanged, no cached dynamic graph|
|`network.py:GCN.forward`|`A ReLU(A X W1) W2`; complete concatenated 320-dimensional input in every branch|Unchanged in E00, not called in E01|
|`network.py:GFR.forward`|`i==1` overwrites `i==0`; FUNC and CORR accumulate; propagation uses `normalize_adj(A+I)`|Retain overwrite; PHENO intermediate norm is not a final contribution|
|`layers.py:InitialDataLayer.forward`|Contiguous split into 48/6/10/256; concatenation equals raw input bitwise|Explicit assertion|
|`layers.py:VariLengthInputLayer`, `EncodeLayer`, `FeedForwardLayer`, `OutputLayer`|Input/CAM transformations and dropout; OutputLayer ignores passed attention embedding|Use identical original layers in both groups|
|`utils.py:ClsLoss`, `GraphConstructLoss`, `normalize_adj`, `my_weight_init`|Weighted NLL; smoothness minus degree reward plus sparsity; symmetric normalization; Xavier only for Linear weights|Unchanged formulas|
|E43.1 `run.py:make_pln_without_pen`, `make_agent`|Physical absence of MLP/avg_pool/prompt cache; PLN 49,544 and GFR 887,836 parameters; registered idle layers retained|Copy minimal self-contained implementation, no runtime dependency on old Python|
|E43.1 best-model audits|Weak message dependence under fixed weights does not answer retraining comparison|Run E01 from untrained paired states|
|Paper Sec.2.1 Eq.(1)–(2), Fig.1|Relations connect patients; paper includes supervised adjacency and describes summation across branches|Distinguish from executable ALoss omission/overwrite; do not repair historical behavior|
|Paper Sec.2.2 and Fig.2|CAM followed by PEN in full paper; paper claims GFR improvement over CAM baseline|Here isolate the physical PEN-free B0-B backbone, not full paper performance|
|Paper implementation details versus shell|Paper ABIDE hyperparameters differ from this repository's shell|Use frozen historical shell/E43.1 parameters, not substitute paper values|

Graph regularization for each modal feature matrix is `theta_smooth * tr(X^T L X)/N² - theta_degree * mean(log(A1 + 1e-5)) + theta_sparsity * sum(A²)/N²`, averaged across four relations. Here theta is 1 / 0.5 / 0. The paper's label-adjacency term is absent from the executable simple-2 optimization. Original ALoss still computes a label-masked adjacency for observation and never enters either group's classification output.

## Data, protocol and interpretation

871 rows × 320 inputs; ASD=0 (403), TD=1 (468). PHENO48 comprises age bins/site/sex encoding; ANAT6 and FUNC10 are image quality and motion summaries, not disease lesions; CORR256 contains selected functional connections. GFR graphs connect patients; its output is mixed, and CAM's four 36-dimensional positions are latent features, not ROIs.

The fixed configuration is `config.json`: 450 epochs, patience 200 with strict `wait>200`, lr .0038, weight decay .11, dropout .35, hidden 18, heads 2, one encoder, threshold .9. Historical class-weight calculation order is retained. No CLIP/GPT/PEN execution or new pretrained model is permitted. No third group or parameter search.

The only formal metrics are ACC, original one-hot hard-label AUC, SPE-TD, SEN-ASD, in that order, with ten-fold population SD. Hard-AUC equals balanced accuracy for binary hard predictions and is not independent evidence from both recalls. Confusion uses `labels=[0,1]`; ASD TP/FN and TD FP/TN are stored explicitly. Tests serve as validation for checkpoint selection; global preprocessing/RFE and transductive graphs are preserved. This is an internal historical-protocol comparison, not an unbiased generalization estimate or causal claim. The no-GFR contrast isolates the complete GFR package (propagation plus its necessary graph regularization), not separate effects of graph learning versus message passing.

Pre-registered GO requires E00−E01 mean ACC and Hard-AUC ≥ .1 pp, and at least 7/10 **strict** wins for each. Otherwise report insufficient incremental evidence; no tuning or further groups.

## Engineering and reproduction

`python -B run.py prepare`; `CUDA_VISIBLE_DEVICES=0 python -B run.py preflight --gpu 0`; `CUDA_VISIBLE_DEVICES=1 python -B run.py preflight --gpu 1`; `python -B run.py all` (or `resume`); `python -B run.py audit`; `python -B run.py report`.

Preflight independently compares E00 with the historical original-step formula, model gradients, both Adams and RNG; proves E01 never calls GFR/GraphConstructLoss, changes only PLN and consumes the same dropout RNG; verifies held-out-label permutation invariance; and simulates an on-disk interruption followed by exact third-update replay for each group. All ten anchors are reconstructed on their assigned GPUs. No registered idle parameter is removed.

Each group-fold has a nonblocking file lock, atomic fsync/replace checkpoints every ten epochs and after early stopping, best weights/predictions and provenance, two Adam states and full RNG in resume checkpoints. Results are committed only after best-checkpoint readback verification. Best checkpoints retain best-epoch optimizer/RNG as well, enabling exact continuation of that model if needed. Completed keys are skipped only after source/split/anchor/checkpoint validation. Final independent audit reloads all 20 best states and recomputes metrics, confusion, predictions and hashes; validates GFR immutability in E01 and pairwise RNG traces; and rehashes protected sources and historical artifacts. There are no epoch-by-epoch model copies. Runtime files are necessary recovery/equivalence evidence and are not intended for Git upload.

Per-fold E00 diagnostics record input/GFR/difference norms, branch classification and graph gradients, effective contributions, and fixed-checkpoint individual/joint message disabling effects. E01 records zero calls/gradients/updates and unchanged GFR model/optimizer. Fixed-model disabling is a mechanism diagnostic, not a substitute for retraining.

The handoff is appended only after successful independent acceptance, with original prefix SHA protection. Storage or scientific failure is recorded truthfully. No automatic Git commit/push.

## Audited code line references

- `main.py:44` `set_rng_seed`.
- `model.py:24` `EvalHelper`.
- `network.py:53` `PLN`.
- `network.py:153` `GraphLearn`.
- `network.py:175` `GCN`.
- `network.py:197` `GFR`.
- `layers.py:21` `InitialDataLayer`.
- `layers.py:60` `FeedForwardLayer`.
- `layers.py:80` `VariLengthInputLayer`.
- `layers.py:144` `EncodeLayer`.
- `layers.py:175` `OutputLayer`.
- `utils.py:31` `GraphConstructLoss`.
- `utils.py:92` `ClsLoss`.
- `utils.py:101` `normalize_adj`.
- `utils.py:127` `my_weight_init`.
