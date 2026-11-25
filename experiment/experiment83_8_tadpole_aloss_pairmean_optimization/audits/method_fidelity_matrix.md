# Experiment 83.8 Method Fidelity Matrix

| Component | Frozen historical behavior | E83.8 treatment | Verification |
|---|---|---|---|
| Inputs and CV | 598 patients, six modalities, data/split SHA frozen, seed 0 | Same E83.5-E83.7 data and ten folds | SHA, row order, dimensions, labels, and split audit |
| Preprocessing | Full-data RFE, historical mean imputation, scaling before outer CV | Preserved from the locked processed input | Input manifest and SHA audit |
| Validation | Transductive patient graph, `val_idx == test_idx`, first strict test ACC improvement | Preserved exactly | Fold identity and checkpoint replay audit |
| G1 | Six-modality GFR, CAM, MMGPL-12 PEN, RES, three-class head | X1 adds only train-only Paper-ALoss to the native historical total | Architecture and optimizer match; historical initial-state hashes are unavailable, so initialization is PARTIAL_MATCH |
| G2 | Physical No-PEN; GFR, CAM, graph learning and graph losses remain | X2 adds only train-only Paper-ALoss | No-PEN module and parameter inventory; historical initial-state hashes are unavailable, so initialization is PARTIAL_MATCH |
| G3 | DARG Positive-DCR with signed positive/negative relations and ten warmup epochs | X3 retains DARG and adds Paper-ALoss in warmup and formal training | Relation, optimizer and paired X3/X4 warmup audit; ALoss intentionally changes the shared warmup trajectory |
| G4 | G3 plus DPSL SCMR-DA-SRPE, 12 prompts and all three class pairs | X4 adds Paper-ALoss to `historical_total` before DPSL gradient budgeting | DPSL residual/shared-gradient and loss-in-backward audit; ALoss intentionally changes the shared warmup trajectory |
| Paper ALoss | Eq.(1) uses a squared Frobenius norm for each modality | Six original GFR adjacency matrices, normalized over the 598-node transductive graph before train-by-train restriction, fixed coefficient 1 | Formula and label isolation pass; fold-1 M1 has no off-diagonal graph edges and no effective ALoss gradient, so the required six-branch preflight fails |
| Author ALoss | Four-branch author GFR returns the mean of unsquared `torch.linalg.norm` values | Kept as an independent diagnostic only | Frozen author source and source-line audit |
| E83.6 pair_mean | Continuous cosine similarities, one unbalanced all-training-pair mean, coefficient 1 | Not reused under the same name as X5/X6 | E83.6 source and report reviewed |
| X5 pair supervision | Not historical | Six equally weighted class-pair group MSEs, unique pairs, no self-pairs, coefficient 0.1 from epoch 11 | Exact pair counts, target mask, warmup and loss audit |
| X6 pair supervision | Not historical | X5 plus detached native-head normalized entropy pair weights in [0.5,1] | Detach, range, weighted-mean and warmup audit |
| DARG optimizers | Historical GFR/CAM Adam and DARG AdamW groups | Preserved | Optimizer config and update audit |
| DPSL | Shared gradient cap 0.001, residual fraction 0.08, all three class pairs | Preserved for X4-X6 | Direct optimizer, residual, pair-route and replay audit |
| Historical H1-H5 | E83.5 immutable Best/Latest checkpoint outputs | Read-only references, not retrained | Existing E83.5 independent replay and current SHA checks; any missing initial-state digest is reported PARTIAL_MATCH |

The three graph-related objectives are separate mathematical objects. The paper's squared Frobenius term, the author's unsquared norm expression, and E83.6's continuous-similarity `pair_mean` are not interchangeable.

The historical `val=test` checkpoint rule is retained, while every new label-graph and pair target is built only from outer-training labels.

Under the frozen GraphLearn threshold 0.9, fold-1 M1 has only diagonal entries. After adding identity and symmetric normalization its graph is constant, so the squared-Frobenius label target cannot produce an effective M1 GraphLearn gradient. The threshold and loss definition are frozen by the protocol, so formal training remains blocked until a separately approved method resolves this incompatibility.
