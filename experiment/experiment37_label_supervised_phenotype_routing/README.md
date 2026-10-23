# Experiment 37 — LSPCR

LSPCR tests whether labels available only for outer-training patients can supervise the actual phenotype-conditioned PCPR query/key scores, improving patient-to-patient message routing while inference remains label-free. The experiment is isolated in this directory; the repository model, data, and earlier experiments are read-only inputs.

## Source audit

The SPromptGL MICCAI 2025 paper already defines a same-class/different-class supervised adjacency `Ahat` in its graph regularization, so label-supervised graph relations are not claimed as new here. In the executable repository's `simple-2` path, `ALoss` is calculated but the optimized objective is classification loss plus the mean four `GraphConstructLoss` terms; this experiment does not silently add that historical `ALoss`. Repository `GraphLearn` projects each modality, computes cosine similarity, and applies a detached hard `>0.9` mask. Consequently an absent edge receives no direct gradient through its thresholded adjacency. The original GFR's PHENO branch output at `i==0` is overwritten when `i==1`; the original B0 implementation is not edited or repaired.

Experiment 36's four-graph trunk is a separate fair propagation implementation, not the original B0: it shares a two-layer GCN across four graphs, propagates the full 320D concatenated patient features, and combines graph messages using learned softmax weights with row-stochastic propagation. Its E36 T0 changed some existing edge weights, but did not exceed B0 in ACC; disabling PCPR at a fixed checkpoint did not change predicted classes, and four graph weights stayed near 0.25. E36's single-graph fusion was much weaker and dense, so E37 does not search one-graph or directed alternatives.

PHENO48 primarily encodes age bins, sex, and site. The E36 verified SUB_ID alignment additionally provides real age, sex, FIQ/VIQ/PIQ, and handedness for text. ANAT and FUNC describe image quality/motion, not ASD symptoms. CORR text describes blocks of selected FC feature slots, not verified anatomical connections. E36's BioLORD-versus-numeric conditions also differed in representation density (768D dense embeddings versus zero-padded numeric vectors), so that comparison cannot establish a general language advantage.

PhenoBrain's phenotype-guided kernel modulates communication within a participant's ROI graph. LSPCR supervises relation scores between different participants. The distinction is scope and target: labels supervise the same pre-tanh, centered Q/K score that gates existing patient edges; labels never construct the inference adjacency. The historical source's full label-derived `Ahat` is not reused. In G12, the separate control applies the soft target only to existing, thresholded GraphLearn train-train edges, so the hard threshold's missing-edge gradient limitation is measured rather than hidden.

## Frozen data and text conditions

The 871 rows are aligned to the phenotype CSV by Experiment 11's verified unique SUB_ID map, never CSV row order or `SUB_IN_SMP`. The identity chain, reconstructed 48 PHENO and 16 QC values, and source hashes are checked by `phenotype.py`. Diagnostic columns are used only for an after-join audit and never enter text, conditions, or auxiliary pairs. ANAT/FUNC remain QC and motion measurements. No labels, test-fold labels, site names, or diagnostic terms appear in text.

The encoder is the frozen BioLORD-2023 revision pinned and validated by Experiment 36, with attention-mask mean pooling and L2 normalization. Cached conditions are accepted only when their NPZ hash matches the copied E36 metadata. HYBRID is constructed and asserted as `modal_full[:,0]` plus `modal_only[:,1:4]`; no embeddings are regenerated or fine-tuned. The prior user-confirmed licensing attestation and the E36 encoder manifest are preserved in `.runtime/`.

## PCPR and label supervision

For receiver `i` and sender `j`, `e(i<-j)=q(z_i)^T k(z_j)/sqrt(64)`. The router centers scores over each receiver's existing non-self neighbors, symmetrizes that score for the four-graph symmetric path, and applies `A_ij * exp(beta*tanh(e_ij-centered_i))`, with `beta<=0.25` initialized at 0.10. Absent raw edges stay absent, self-loop values are unchanged, and row-stochastic propagation carries the sender's full 320D feature. The symmetric router exposes its actual pre-tanh centered score matrix for supervision.

For every auxiliary loss, the only label tensor passed to the relation-loss method is the outer-train label vector paired with the train indices. Same/different targets are soft 0.9/0.1. BCE averages within class then gives each class equal mass. SUPPORT-BCE restricts to existing train-train edges; ALL-PAIR-BCE supervises every train-train pair but never creates a missing graph edge. Ranking uses up to four lowest-scoring same-class and four highest-scoring different-class training peers per anchor, margin 0.5, softplus. G13 uses only same-site training pairs and averages balanced losses over sites with both pair classes. G10 shuffles only the auxiliary train labels within site; classification always retains the true train labels. G12 applies the separate balanced BCE to existing raw GraphLearn train-train edges. Held-out labels never enter any of these mechanisms.

The optimized loss is `Lcls + Lgraph + lambda_relation*Lrelation`, with the historical two Adam optimizers and unchanged graph-loss weights. Router Q/K/beta parameters are a separate group in the GFR Adam with weight decay zero; existing parameters retain 0.11. The test labels remain available only to the inherited `val_idx=test_idx` checkpoint selection and final metric computation. This inherited protocol is optimistic and is not an independent, leakage-free generalization estimate.

SUPPORT-BCE uses the current detached-threshold graph support. If one modality/fold has no supported pairs of a class at an epoch, that modality contributes a zero auxiliary term for that absent class; no edge is synthesized and the other modalities continue normally. This is reported as a support limitation, not treated as evidence that an absent edge was learned.

## Registered groups

All groups use seed 0, the same ten `StratifiedKFold(shuffle=True, random_state=10)` folds, the same four-graph backbone, original CAM, and full 320D input. G00 is an E36 T0 replay. G01 changes only to HYBRID text. G02/G03 use support-edge BCE for FULL/HYBRID. G04/G05 use all train pairs for FULL/HYBRID, with G05 the registered main method. G06 uses hard-pair ranking. G07/G08 lower/raise the registered relation-loss weight. G09 permutes text patient mappings within train/test pools and site/missingness strata. G10 site-shuffles only auxiliary train labels. G11 uses same-source numeric conditions through the 768D interface, a limited representation-capacity control. G12 supervises raw thresholded GraphLearn train edges without text or Q/K. G13 uses site-balanced train pairs.

GPU0 runs G00, G02, G04, G06, G08, G10, and G12; GPU1 runs G01, G03, G05, G07, G09, G11, and G13. Each worker owns a unique group/checkpoint/task key and processes its full ten-fold stream. It restores E36's validated T0 per-fold entry RNG so shared initial trunks are paired, without claiming all later dropout or optimization trajectories match. Atomic current checkpoints contain both optimizers, current/best state, epoch/early-stop state, RNG, and provenance hashes.

## Run and outputs

From this folder, run:

```bash
python run.py audit --root /data/adodas/SPromptGL-main
python run.py preflight --root /data/adodas/SPromptGL-main
python run.py dispatch --root /data/adodas/SPromptGL-main
```

The ten fold caches, split hashes, E36 RNG entries, task state, logs, and resumable checkpoints are under `.runtime/`. Formal outputs are `results/fold_metrics.csv`, `results/diagnostics.csv`, `results/summary.json`, and `results/report.md`. Only the original four metrics are reported: ACC, original hard-label AUC, SPE-TD, SEN-ASD. The hard-label AUC equals balanced accuracy in this binary setting. No `probability-AUC` or F1 is reported.

The final `initial_aux_gradient_audit.csv` is generated by a disposable one-step replay per supervised group/fold, restoring the saved fold-entry RNG and checking common trunk hashes. It does not modify formal checkpoints or metrics. To recreate it after dispatch, run:

```bash
python audit_initial_gradients.py --root /data/adodas/SPromptGL-main --gpu 0 --groups G02_FULL_SUPPORT_BCE G04_FULL_ALL_BCE G06_HYBRID_RANK G08_HYBRID_ALL_BCE_HIGH G10_HYBRID_AUX_LABEL_SHUFFLE G12_RAW_GRAPH_LABEL_SUP --output results/initial_aux_gradient_audit_gpu0.csv
python audit_initial_gradients.py --root /data/adodas/SPromptGL-main --gpu 1 --groups G03_HYBRID_SUPPORT_BCE G05_HYBRID_ALL_BCE G07_HYBRID_ALL_BCE_LOW G09_HYBRID_TEXT_SHUFFLE G11_HYBRID_NUMERIC_BCE G13_HYBRID_SITE_BALANCED --output results/initial_aux_gradient_audit_gpu1.csv
python run.py summarize --root /data/adodas/SPromptGL-main
```

`text_cosine_audit.csv` records BioLORD cosine distributions over outer-train patient pairs for text groups. `diagnostics.csv` joins these with actual supported-edge Q/K scores and the first-step gradient audit. Split GPU audit files are retained beside the combined audit for traceability.
