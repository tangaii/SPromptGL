# Experiment 78.9 — LOO-FAPC design record

## Evidence and lineage

The model lineage is E74.4-G2 Positive-DCR restored independently per fold from E61 epoch-10 warmup. E78.8 supplies only the validated FAPC implementation pattern, exact private initialization (`788000 + fold`), frozen RN50 text cache, one-forward CAM144 hook, optimizer order, and corrected CAM144 gradient hook `g_base + g_aux_safe`. Its two superseded runs are not used. E78.9 does not initialize from an E78.8 checkpoint.

SPromptGL describes graph-based patient context and semantic prompt-guided local representation, with CAM/PEN downstream; this experiment retains DARG+CAM and signed auxiliary losses, and does not restore the original `max(softmax)` PEN operator. BrainPrompt contributes the two published ABIDE disease-level strings, mapping ASD=0 and TC=TD=1; its original Llama encoder is not used. KgCoOp motivates keeping fixed hand-crafted text references while learning discriminative adjustments. PCGrad motivates projection of conflicting task gradients; E78.9 retains the E78.8 one-sided CAM144 projection and 20% cap rather than reproducing all PCGrad optimizer behavior. Sources: [SPromptGL](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [BrainPrompt](https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf), [KgCoOp](https://arxiv.org/abs/2303.13283), [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

The E78.8 final corrected source revision is `corrected-neutral-anonymous-and-main-gradient-v2` with source SHA recorded in E78.8 `results/audit_status.json`. Its earlier anonymous-text and main-gradient-hook runs were explicitly superseded. E78.9 reuses the same ten medical and ten neutral anonymous RN50 anchors in the shared full row-space SVD coordinate system; cache SHA is checked against E78.8.

## Single algorithm change

For live CAM144 `h`, shared `u=W_patient(h)`, `z=normalize(u)`, and `a=LayerNorm(u)`, legal support is `S=unique(train)-unique(test)`. Class sums and counts are computed once. For query `i`, a support-membership indicator subtracts `a_i` only from its true training class sum, with the same leave-one-out support set used for both class means. Testing queries are absent from support and therefore use full class means. The two signed directions are projected by the shared bias-free `W_proto` and normalized.

The vectorized identity `(sum_c - 1[y_i=c] a_i)/(N_c - 1[y_i=c])` gives zero derivative from a support query's own feature into its LOO prototype while preserving gradients from all other support features. The query's independent `z_i` path remains differentiable. No query-specific encoder pass, random sampling, or public RNG operation is added.

G1 uses `10*cos(z_i,v_i,c)` visual logits. G2/G3 calibrate each frozen prompt per query with `normalize(anchor_c,k + beta_c,k v_i,c)`, then use signed classwise normalized log-mean-exp (temperature 0.25, logit scale 10). `L_keep` averages over query and prompt dimensions. The full objective remains `L_DARG + 0.1 L_aux + 0.01 L_keep`; formal prediction remains the original DARG `logp`.

The single CAM144 hook computes separate base and auxiliary gradients, projects only a conflicting auxiliary gradient, caps it at 0.20 of the base norm, and returns `g_base + g_aux_safe`. DARG classifier gradients, the original three optimizers, and the new AdamW optimizer (`5e-4`, WD `0.01`) follow E78.8 order and settings. All trainable module shapes, registration order, and private initialization are kept identical to corresponding E78.8 groups.

## Protocol limits

The experiment preserves historical `val=test`, strict test-fold ACC checkpoint selection, pre-CV preprocessing/RFE, transductive full-patient graph, single seed, and no changes to the historical base training path. Fold 2's known overlap rows are excluded from new support and auxiliary supervision only. These choices preserve comparison lineage but prevent interpreting results as unbiased independent clinical generalization.
