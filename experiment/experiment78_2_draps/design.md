# Experiment 78.2 design freeze — DRAPS

## Objective

Test whether explicit ASD/TD disease prototype supervision, reliability-aware adaptive prototype capacity, index-free cross-disease competition, and a low-capacity patient gate can add diagnostic value to the unique E74.4-G2-Positive-DCR base. The four formal groups differ only in prompt bank and maximum candidate capacity.

## Frozen groups

| Group | Prompt bank | Maximum candidate K per class |
|---|---|---:|
| E78.2-G1 | SPromptGL original eight prompts | 1 |
| E78.2-G2 | SPromptGL original eight prompts | 3 |
| E78.2-G3 | SPromptGL original eight prompts | 5 |
| E78.2-G4 | anonymous eight prompts | 5 |

K is candidate capacity. Effective K is determined from train-support reliability rules and is never selected from test metrics.

## Representation and text geometry

The E74.4-G2 `PLNNoPEN` CAM input is `h∈R^144`. DRAPS computes `q=normalize(W_sem h)∈R^16` and `z=normalize(LN(W_het h))∈R^32`. Frozen RN50 text features for the eight SPromptGL prompts and eight anonymous prompts are represented in one rank-16 SVD coordinate system; coordinates are copied from the SHA-checked E78.1 cache and the original pairwise cosine geometry is audited before training.

## Prototype and reliability equations

For class `c`, support responsibility is `rho_i,c,k=softmax_k(cos(z_i,a_c,k)/0.25)`. Every group, including K=1, uses within-class inverse-square-root site-size weights normalized to mean one. The weighted support mass is `m_i,c,k=w_i rho_i,c,k`. Dynamic prototypes use the same mass in numerator and denominator. Training queries are exact leave-one-out: their own mass and projected feature are removed only when the query is an explicit member of legal support.

`N_eff,c,k=(Σ_i m_i,c,k)^2/(Σ_i m_i,c,k^2+eps)` and `R_support=clip((N_eff-20)/10,0,1)`. Site reliability is the mean of train-support site coverage and normalized mass entropy. The product is normalized within each disease class to `a_c,k`; if all products are negligible, every candidate falls back to the LOO global class center and uniform candidate weights.

Each epoch updates the non-gradient anchor buffer once, after the four optimizer steps, using `a←normalize(0.9a+0.1 stopgrad(p))`. The update uses the already computed dynamic support prototypes and consumes no public RNG.

Disease prototype evidence is `S_i,c=0.25 logsumexp_k(log(a_c,k)+cos(z_i,p_c,k)/0.25)`, with `L_proto=CE([S_ASD,S_TD]/0.25,y)`. Prompt evidence uses patient-conditioned opposing references, never equal-index ASD/TD subtraction. For each class and candidate, `score=cos(q,t_j)+cos(p_q,c,k,t_j)-cos(vbar_opp,t_j)` and class evidence is the reliability-and-routing weighted sum of the within-class prompt softmax scores. `L_sem=CE([e_ASD,e_TD],y)`.

The patient gate receives six fixed statistics (two class reliability values, prototype and semantic margins, their mean reliability, and a conflict magnitude). `g=0.25 sigmoid(w^T statistics+b) reliability_i`, where reliability is reduced when support is weak or the two evidence margins disagree. A shared 16→144 map produces an L2-normalized residual, and the original OutputLayer is called once on `h+g residual`.

## Frozen loss and protocol

`L=L_E74.4(h_fused)+0.05 L_proto+0.02 L_sem+0.005 L_site`. No E78 gain loss, separation hinge, equal-utilization KL, hard-example memory, or post-hoc classifier is used. The three E74.4 optimizers and one independent AdamW (`lr=5e-4`, `weight_decay=0.01`) update in the fixed historical order. Each fold starts from the SHA-checked E61 epoch-10 warmup state, original optimizer states, original RNG, data order, and split. Selection remains strict test-fold ACC with `val=test`, preserving the historical protocol and its limitations.

## Mandatory controls

Base-off parity, exact LOO, manual effective-sample-size arithmetic, candidate-index permutation invariance, K=1/3/5 forwarding, all-low-reliability global-center fallback, finite gradients, two-step recovery, test-label invariance, and fixed-checkpoint interventions are run before/after formal training.
