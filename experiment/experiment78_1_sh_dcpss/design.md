# Experiment 78.1 design freeze

## Objective and scope

Test whether train-fold-supported ASD/TD within-class modes can yield patient-specific medical Prompt selection and improve the original classifier when the semantic evidence modifies CAM144 before the original OutputLayer. This is a single-seed, historical-protocol comparison, not an unbiased external-generalization estimate.

## Fixed groups

| Group | Text | K per class | Trainable parameter count |
|---|---|---:|---:|
| E78.1-G1 | Original eight SPromptGL prompts | 1 | 9,425 |
| E78.1-G2 | Original eight SPromptGL prompts | 3 | 9,425 |
| E78.1-G3 | E78-G5 anonymous eight prompts | 3 | 9,425 |

All groups share the same trainable layers and private-RNG initialization. K changes only data-derived, non-trainable support anchors and the shapes of dynamic aggregates. Text coordinates for both prompt banks use one frozen rank-16 SVD basis spanning the 16 unit-normalized CLIP RN50 vectors.

## Model

For CAM feature `h_i in R^144`, `q_i = normalize(W_sem h_i) in R^16` and `z_i = normalize(LN(W_het h_i)) in R^32`. Initial anchors are derived from E61 epoch-10 CAM144 using only legal outer-train support patients. Within each class, subtract the class mean, estimate a within-class site residual mean, and shrink that site offset by `n_site/(n_site+10)`. Deterministic farthest-first initialization and 20 fixed Lloyd updates form K anchors. These anchors initialize assignment only; each epoch's prototypes are recomputed from current train-support CAM features.

For each support patient, `rho_i,c,k = softmax_k(cos(z_i,a_c,k)/0.25)` is evaluated only at its known support class. A modest inverse-square-root site-size factor, normalized to mean one within class, weights prototype contributions. For each query and each candidate class, weighted `z` and `q` prototype sums are aggregated from support. If the query identity is in support, its exact weighted soft contribution is subtracted from every prototype of its support class before division. Test query labels are never read. `beta_i,c,k = softmax_k(cos(z_i,p_c,k^het)/0.25)` routes every query against both disease classes; training support queries route against exact LOO heterogeneity prototypes.

For class c and subtype k, every prompt j assigned to c receives `s_i,c,k,j = cos(q_i,t_j) + cos(v_c,k,t_j) - cos(v_1-c,t_j)`. The class-specific four-prompt softmax at temperature 0.5 yields `alpha_i,c,k,j`. The semantic classifier auxiliary uses `e_i,c = sum_k beta_i,c,k * sum_j alpha_i,c,k,j*s_i,c,k,j` and weighted two-class cross entropy. The feature vector uses both classes simultaneously: `z_sem = normalize(sum_k beta_i,0,k*(sum_j alpha_i,0,k,j*t_j + 0.5*(v_i,0,k-v_i,1-k)) - sum_k beta_i,1,k*(sum_j alpha_i,1,k,j*t_j + 0.5*(v_i,1,k-v_i,0-k)))`.

One linear map produces `r_i = W_fusion z_sem`. The bounded residual is `d_i = ||h_i||_2 * r_i/ max(||r_i||_2,1e-8)`. A sigmoid-parameterized scalar is initialized to 0.1 and capped at 0.25; `h_i^fused = h_i + lambda*d_i`. When the residual intervention is off, the original `h_i` tensor is passed unchanged to the historical OutputLayer. The only active classifier remains the original `OutputLayer`; DARG/CAM executes once and the OutputLayer executes once per forward.

## Losses

`L = L_E74.4(h_fused) + 0.02 L_sem + 0.02 L_het + 0.01 L_site`. `L_sem` is weighted two-class cross entropy over the legal support queries. `L_het` is a fixed weighted combination of normalized within-class variance-floor, cosine-separation hinge at cosine 0.95, assignment-utilization KL, and data-supported leave-one-out compactness. K=1 naturally has zero K-specific separation/utilization terms. `L_site` is the mean normalized KL between sufficiently represented (n>=20) within-class-site responsibilities and the class-average responsibilities; K=1 has zero site loss. Auxiliaries use support rows only. New parameters use an independent AdamW at 5e-4, weight decay 0.01; the three historical optimizers and their update order remain unchanged and the new optimizer steps after them.

## Data and protocol boundaries

All three groups restore the matching E61 epoch-10 warmup model, all three E74.4 optimizer states, split/pair identities and Python/NumPy/CPU/CUDA RNG. Outer test rows are removed from prototype support and auxiliary supervision. Fold-2 historical overlap rows 377 and 407 stay in the original historical training loss but receive zero new feature residual during training. At evaluation, all test patients are processed by the same dual-class inference rule without their labels. Selection remains strict test-fold ACC with val=test, preprocessing/RFE before outer CV, transductive full-patient graphs, seed 0, 450-epoch cap and `wait>200` stopping.

## Frozen environment and output

The preferred OSSFS path failed with errno 5. The only active output is `/model/spromptgl/experiment/experiment78_1_sh_dcpss`. No code, configuration, checkpoint, or runtime artifact under E74.4, E78, or the repository-root model is changed. Formal training begins only after all-fold base-off parity, one-fold full-chain gradient/LOO/text-geometry/fusion checks, numeric-loss validation, and a deterministic two-step recovery replay pass.
