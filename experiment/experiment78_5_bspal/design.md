# Experiment 78.5 — BSP-AL design freeze

## Audit basis

PPAL MICCAI 2025 Section 2.3, Figure 3 and Eqs. 8–12 were audited from the supplied proceedings PDF and official `diaoyq121/PPAL` source. The paper describes prototype aggregate attention, text aggregate attention, three progressive updates and coarse/fine supervision; public `engine3.py` actually backpropagates one CrossEntropy loss, and `ppal.py` mixes batch-first-looking tensors with MultiheadAttention paths that do not always declare `batch_first`. BSP-AL therefore uses explicit patient/query axes and does not copy PPAL's 512-dimensional classifier.

SPromptGL's GFR/CAM, eight prompt order and OutputLayer were checked against the server-side E74.4 implementation. E78.5 uses the SHA-checked E74.4-G2 Positive-DCR/DARG and full CAM144, keeps the original graph/pair/adversarial/preservation path, does not restore PEN, and does not split CAM144 into four modality tokens. E74.4's actual loss is `out[total] = cls + graph + 0.05*pair + 0.01*adv + 0.01*preserve`, with weighted NLL in `out[cls]`.

E78.4 was independently read: replacing `out[cls]` by the fused loss weakened the original MLP path; its patient attention was nearly uniform and its K=2 prototypes collapsed. E78.5 therefore registers only three groups and keeps K=1/restricted residual.

## Frozen groups

- G1: E78.4 K=1 medical module, unchanged aggregation, new baseline-preserving loss.
- G2: G1 loss plus selective patient attention, separate LayerNorm attention features and L2 classification features, and four prompt evidences weighted after patient-conditioned text residuals.
- G3: byte-identical G2 structure/configuration with the E78.1 anonymous prompt bank.

All groups use one ASD and one TD prototype, 32 dimensions, four heads, three shared iterations, no LOO, no dropout and the same private-RNG initialization. Only prompt source and selective flag differ.

## Frozen objective

For legal support `S = sorted(train_indices \ test_indices)`,
`L = out[total] + 0.2 * (weighted_NLL(final_logp[S], y[S]) - weighted_NLL(base_logp[S], y[S]))`.
Both terms use exactly the same support and class weights; neither is detached. This preserves the original DARG loss while adding a bounded fused correction. Fold 2 keeps the historical base path, while the differential terms use only legal support rows.

G1 retains the E78.4 legacy MultiheadAttention path. G2/G3 use explicit cosine Q/K logits with bounded `0.05–0.5` temperature, unnormalized V features, LayerNorm attention features and L2-normalized prototype matching. Each of four prompts independently queries same-class patients; `normalize(text + beta*e)` is weighted by current prototype similarity at fixed temperature 0.5. The final residual is `base_logp + [delta/2,-delta/2]`, with `delta = lambda*tanh((r_ASD-r_TD)/0.5)` and `lambda <= 0.5`.

## Protocol and limitations

Each fold restores E61 epoch-10 warmup, the E74.4-G2 model, three optimizer states and RNG. Base-off eta=0 parity, differential-loss support identity, attention axes, test-label invariance, gradients and two-step recovery are required before freezing source SHA. Historical `val=test`, test-ACC checkpoint selection, CV-before preprocessing/RFE, transductive graph and Fold 2 overlap remain limitations and are disclosed in the final report.
