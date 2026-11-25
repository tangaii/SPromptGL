# E83.7 Method Fidelity Matrix

| Component | Historical or adapted behavior | E83.7 status |
|---|---|---|
| Data | 598 TADPOLE patients, six modalities, locked SHA and ten folds | preserved |
| GFR | six GraphLearn graphs, thresholded propagation, second-modality overwrite | preserved |
| CAM | hidden 10, four heads, two layers, width 240, dropout 0 | preserved |
| G2 | PEN physically removed, GFR/GraphLearn/GCN/message passing retained | corrected description |
| G5 | physical CAM-only model with no graph or prompt modules | preserved |
| DARG | Positive-DCR signed positive and negative relation updates | preserved |
| DPSL | three TADPOLE class pairs, residual cap 0.08, native/full split | preserved |
| Native classification | historical native classification loss drives shared parameters | preserved |
| Shared gradient | historical tangent direction protection, cap 0.001 for G4, T1/T2 controlled caps | controlled |
| Graph supervision | six continuous pre-threshold cosine scores, train patients only, pair_mean coefficient 1.0 | T3/T4 warmup-only |
| Weight decay | G3/G4 historical GFR/CAM WD 0.11 | T5/T6 0.055 from epoch 11 |
| ALoss | computed for historical diagnostics but omitted from final optimization | preserved |
| Checkpoint | Best first strict test ACC, Latest model/optimizer/RNG state | preserved |

The E83.7 implementation hash and pre-training seal are recorded in `audits/training_seal.json`. Post-training audit code is kept separate from the sealed training modules.
