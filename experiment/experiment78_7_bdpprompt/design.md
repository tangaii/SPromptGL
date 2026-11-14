# Experiment 78.7 — BDP-PL design and audit record

## Scope and sources

BDP-PL is a fixed four-group, ten-fold seed-0 experiment built from the SHA-checked E74.4-G2 Positive-DCR/DARG warmup chain. The server implementation is loaded read-only through `path_adapter.py`; no root model file is copied or modified. The executable E74.4 loss is `out["total"] = cls + graph + 0.05*pair + 0.01*adv + 0.01*preserve` in `soft_positive_dcr.py`. E78.7 preserves that tensor exactly and adds only `0.1*L_sem + 0.01*L_keep`.

The SPromptGL paper and official repository motivate GFR, CAM and semantic prompt interaction. The local `network.py`, `model.py` and `layers.py` are the authority for tensor order and the original eight strings, including `Nromal`. BrainPrompt's ABIDE disease-level file contributes the two exact TC/ASD sentences; TC is mapped to this project's TD=1 and ASD remains 0. They are encoded with the existing frozen OpenAI CLIP RN50 bank, so this is an adaptation rather than a reproduction of BrainPrompt's Llama encoder. PPAL motivates prior-guided prototype aggregation and progressive text/visual updates; the implementation below uses only the small 32-D patient/text branch and explicit four-head patient attention, not PPAL's MRI encoder or high-capacity classifier. HP²L was used only for motivation; no unverifiable formula or code is imported.

E78.6 established that original SPPromptGL SPGS is computationally active but its `max(softmax([s_ASD,s_TD]))` is invariant to swapping the two class vectors and therefore does not preserve signed disease direction. BDP-PL therefore uses signed ASD-vs-TD semantic logits and never uses that max-softmax operator.

## Fixed prompt banks

Medical order is `[ASD four SPromptGL, ASD BrainPrompt, TD four SPromptGL, TD BrainPrompt]`. The exact BrainPrompt sentences and original spelling are kept. The anonymous bank reuses the SHA-checked E78 bank's eight label-only strings and repeats the first ASD and first TD string for the two additional slots; this avoids inventing or encoding an unverified text while giving both classes five fixed slots. `runtime/text_embeddings.pt` records the source-cache SHA, raw strings, class mapping and the reuse explicitly.

## BDP-PL equations

For CAM144 `h_i`, `u_i=W_patient(h_i)`, `z_i=normalize(u_i)` and `a_i=LayerNorm(u_i)`. Legal support is `unique(train_indices)-unique(test_indices)`; no LOO and no test labels are used by the new branch.

Direction A uses hard training labels only to compute `mu_c=mean(a_j:y_j=c)`, `d_c=normalize(mu_c-mu_{1-c})`, `v_c=normalize(W_proto(d_c))`, and each prompt independently becomes `q_A,c,k=normalize(b_c,k+beta_c,k v_c)`, where `beta=0.2*sigmoid(raw)` starts at 0.1. `L_keep` is the mean cosine displacement from the uncalibrated projected prompt.

Direction B detaches only `teacher_p=softmax(DARG_logp)` and uses all legal support patients (no hard class partition): for each class prompt, explicit four-head attention computes
`score_j = cosine(Q,K_j)/0.1 + log(teacher_p[j,c])`, softmaxed over the support-patient axis. Values remain unnormalized projected `a_j`. Three shared updates use `q <- normalize(LN(q+gamma W_o(E))+0.1 FFN(q))`, with `gamma=0.2*sigmoid(raw)` initialized to 0.1. G1 enables A only, G2 B only, G3 A+B, and G4 A+B with anonymous text.

Five prompt scores remain separate and are pooled with normalized log-mean-exp: `r_c=0.25*logmeanexp_k(cos(z,q_c,k)/0.25)`. The signed semantic auxiliary logits are `10*[r_ASD,r_TD]`, with ASD=0, TD=1. The final prediction is always the original DARG log-probability; semantic logits are auxiliary supervision and mechanism diagnostics only.

`L_total=L_DARG+0.1*CE(L_sem,y_support)+0.01*L_keep`. The three historical optimizers retain their exact order; the new AdamW is stepped last. New initialization is under private `fork_rng`; the public RNG stream is never reseeded.

## Preflight and audit decisions

The runner first checks ten-fold E74.4 base-off parity, prompt cache identity, support isolation, signed semantic shapes, patient-axis normalization, active gradients into DARG/CAM and enabled A/B parameters, auxiliary-off exactness, query permutation and test-label invariance, and a two-step checkpoint/RNG replay. Formal checkpoints store all historical and new optimizer states, RNG, source map, support identity, semantic diagnostics and best test-fold ACC selection. `audit.py` independently reloads every best state, recomputes DARG probabilities and diagnostics, checks task/checkpoint/source identities and produces paired comparisons, mechanism rows and fixed-checkpoint A/B/uniform/Prompt interventions.

The retained protocol still has historical `val=test`, test-fold ACC selection, pre-CV preprocessing/RFE, a transductive full-patient graph and the known Fold-2 overlap. The new branch excludes the overlap from support and semantic supervision, but these protocol limitations prevent an unbiased clinical generalization claim.
