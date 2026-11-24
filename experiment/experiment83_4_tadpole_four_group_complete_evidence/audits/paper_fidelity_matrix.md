# SPromptGL TADPOLE Paper-to-Code Fidelity

Sources reviewed: [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [official author code](https://github.com/wanxixi11/SPromptGL), and the frozen E83.1-E83.3 source snapshots in this directory.

| Area | Paper / official implementation | E83.4 finding and handling |
|---|---|---|
| TADPOLE input | 598 patients, six modalities, 365 total features | Exact inherited order 145/148/3/36/24/9 and locked file SHA. Full-data RFE, mean imputation and pre-CV scaling are historical leakage risks and remain fixed by protocol. |
| GraphLearn | Per-modality learned graph, thresholded weighted cosine and normalized adjacency | E83.1 archive parity covers all six modalities. No E83.3 TADPOLE graph-construction error found. |
| GCN and GFR | Graph propagation plus historical modality fusion | Exact baseline source parity. Historical second-branch overwrite of the first fused feature is preserved. It looks unusual but is source behavior, not an E83.3 regression. |
| CAM | TADPOLE hidden dimension 10, four heads, two encoding layers, dropout 0, alpha 1 | CAM width is 6 x 4 x 10 = 240. E83.1 initialization, forward, RNG and gradients match the archived implementation. |
| RES and PEN | Local/global token interaction and eight original prompts for full SPromptGL | G1 retains the original eight-prompt PEN path and historical RES computation. No MMGPL prompt substitution is applied to G1. |
| No-PEN | Historical ablation removes PEN | G2 physically omits PEN and all PEN parameters. |
| Optimizer | TADPOLE learning rate 0.012; historical configured regularization | G1/G2 retain Adam with weight decay 0.0036. G3/G4 retain E61 DARG optimizer and warmup settings. |
| Losses | Classification and graph construction losses from source | Historical ALoss is computed for diagnostics but is not added to the optimized objective. E83.4 preserves this behavior. |
| Best checkpoint | Historical training evaluates on the held-out fold during selection | `val_idx == test_idx`; Best is the first strict test-fold ACC improvement. This is retained and reported as evaluation leakage. |
| Three-class DPSL | Historical E82 core is binary | E83.3's hard Top-2 adaptation excluded one pair for most patients. E83.4 fixes only this TADPOLE adaptation with all three pairs and shared historical DPSL parameters. Binary regression remains exact. |

No E83.3 G1/G2/G3 base-computation defect was established by the archived parity evidence. The paper-to-experiment gap is not attributed to a guessed code bug. The locked protocol differs from an unbiased independent-validation evaluation, and the paper ablation rows are not direct equivalents of the four E83.4 groups.
