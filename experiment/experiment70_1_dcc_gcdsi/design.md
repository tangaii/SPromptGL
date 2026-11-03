# Experiment 70.1 — DCC-GCDSI

## Motivation and relation to prior work

Experiment 70 GCDSI was audited as NO-GO. Its two BrainPrompt CLIP embeddings had cosine similarity 0.984782; G1 attention entropy was close to `ln(2)`, so two full-text keys provided weak class competition. The direct value average retained shared text content, and the learned residual reached 0.456 of CAM RMS on average with per-fold modality ratios above 1. The implementation also assigned the module to `cam.gcdsi`, registering it inside the CAM state tree and duplicating checkpoint state. The configured output projection initialization did not match its Xavier implementation.

SPromptGL describes patient-relation graph learning (GFR), cross-modal CAM interaction and prompt-guided local PEN enhancement. The experiment-53.3 server implementation is the sole training base. Experiment 70.1 inserts the new module after CAM and before its original OutputLayer, without reintroducing PEN's fixed local feature partition.

SABER equations (8)-(10) motivate a graph-conditioned query, text cross-attention and gated residual injection. SABER separately uses disease-level contrastive supervision; DCC-GCDSI does not add such a loss and is a research adaptation, not a reproduction of SABER.

## Frozen architecture

For patient `i`, modality `m`, CAM feature `h_im in R^36` and GFR feature `f_i in R^320`:

```text
q_im = LayerNorm(W_h h_im + W_f f_i + q_CLS)
t_ASD,t_TD = L2-normalized frozen CLIP text embeddings
c = normalize((t_ASD + t_TD)/2)
d = (t_ASD - t_TD)/(||t_ASD - t_TD|| + eps)
k_delta = normalize(W_d d)
z_im = q_im^T k_delta
A_im = softmax([z_im,-z_im])
s_im = A_im,ASD - A_im,TD = tanh(z_im)
v_center = normalize(W_c c)
u_im = tanh(W_o(h_im elementwise v_center))
g_im = sigmoid(MLP([h_im,q_im,v_center]))
r_im = g_im elementwise u_im
lambda = 0.15 tanh(lambda_raw), lambda_raw starts at zero
delta_im = lambda s_im RMS(h_im) r_im/(RMS(r_im)+eps)
h'_im = h_im + delta_im
```

Tensor shapes are `F:[871,320]`, `H:[871,4,36]`, `Q:[871,4,36]`, attention `[871,4,2]`, and final classifier input `[871,144]`. The signed class coefficient changes sign when the disease text order is exchanged, while the shared center and modulation direction remain unchanged. Identical class text produces zero difference and zero residual. Per patient and modality, residual RMS is bounded by 15% of CAM RMS.

The main classification path is `X320 -> CADR-GFR -> F320 -> CAM -> H[4,36] -> DCC-GCDSI -> H'[4,36] -> flatten144 -> original OutputLayer`. The CAM module is called once. A scoped OutputLayer wrapper owns no DCC parameters; DCC-GCDSI has a separate optimizer and checkpoint state.

## Frozen protocol

The base is Experiment 53.3 `CADR_WD_0p01_GFR_WD_0p11`, ABIDE A, 871 subjects, seed 0, `StratifiedKFold(10, shuffle=True, random_state=10)`, PHENO/ANAT/FUNC/CORR dimensions 48/6/10/256, labels ASD=0 and TD=1. It reuses the audited 10-epoch 53.3 warmup state. GCDSI activates at epoch index 10. Existing CADR/GFR/CAM optimizers, graph logic, losses, 450-epoch cap, wait>200 early stop and strict ACC checkpoint rule are unchanged. DCC-GCDSI uses AdamW lr=0.001 and weight decay 0.001. There is no new auxiliary loss or dropout.

G1 uses BrainPrompt ABIDE's original `[TC,ASD]` lines reordered to `[ASD,TD]`. G2 uses SPromptGL's eight exact original strings, including `Nromal`, normalized individually and averaged as `[ASD,TD]`. G3 reuses Experiment 70's deterministic anonymous bank, matched to G1 row norms and cosine geometry. All banks use the same frozen CLIP RN50 cache and their Experiment 70 SHA values are checked before training.

## Evaluation boundaries

Metrics retain the historical hard-prediction AUC, continuous TD-probability AUC, ASD sensitivity and TD specificity. The historical `val=test` checkpoint selection, preprocessing before outer cross-validation, transductive 871-node graph and single seed remain. These results are exploratory and do not estimate unbiased external generalization.
