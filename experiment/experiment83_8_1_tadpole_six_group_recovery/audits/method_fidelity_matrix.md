# Experiment 83.8.1 Method Fidelity Matrix

| Component | E83.8 frozen definition | E83.8.1 treatment | Gate scope |
|---|---|---|---|
| Paper ALoss | Six actual thresholded GFR graphs, squared Frobenius norm, coefficient 1, train-label-only target | Same function and arithmetic | X1-X4 only |
| Structural zero gradient | A modality with no off-diagonal graph edges can have no graph-learning gradient | Record `STRUCTURAL_ZERO_GRADIENT`, retain edge count, scalar term, and gradient norm | Warning, not an automatic block |
| Effective graph gradient | Existing thresholded graph and GraphLearn path | Require finite effective gradient for every modality with nonzero off-diagonal edges and a finite nonzero aggregate ALoss gradient | X1-X4 only |
| Class-Balanced PairMean | Six unique class-pair groups, equal group weighting, coefficient 0.1, epoch 11 onward | Unchanged | X5 only |
| Uncertainty PairMean | X5 plus detached normalized-entropy pair weights, coefficient 0.1, epoch 11 onward | Unchanged, with independent weight and gradient audit | X6 only |
| Model and optimizers | G1, physical No-PEN G2, DARG G3, DARG+DPSL G4 | Unchanged from E83.8 | Per registered group |
| Data and CV | Frozen TADPOLE SHA, ten folds, seed 0, historic preprocessing and checkpoint rule | Unchanged | Shared gate for all groups |
| DPSL | Residual cap 0.08, shared gradient budget 0.001 | Unchanged | X4-X6 |
| Initialization pairing | E83.8 pairing rules, with missing historical initial hashes marked partial | Unchanged | Per paired group |

The E83.8.1 change addresses the gate interpretation. It does not change the ALoss formula, thresholded adjacency, gradient, model, optimizer, or schedule.

The paper's squared Frobenius term, the author source's unsquared `torch.linalg.norm`, and E83.6's continuous-similarity `pair_mean` remain distinct mathematical objects.
