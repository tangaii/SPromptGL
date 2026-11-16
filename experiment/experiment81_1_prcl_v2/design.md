# Experiment 81.1: Mechanism-Refined PRCL-v2

## Motivation and scope

E81 tested frozen disease prompts in patient-internal CAM144 local-local relation selection, beyond SPromptGL's existing cross-patient GFR and CAM's prompt-guided local-global refinement. Its audited code exposed four limits: hard residual clipping produced a flat gate-gradient region; biased fusion/output layers could write a residual under Null-only attention; the linear map over [mN,mA,mA-mN] made the difference term redundant; and the medical prompt changed prediction probabilities very little relative to disabling PRCL.

E81.1 makes one common mechanism repair and tests a different prompt-query interaction. The common repair uses a smooth relative residual map, bias-free zero-preserving writeback, and an explicit nonlinear normal/abnormal interaction. The semantic comparison isolates the E81 additive Query from a multiplicative Query, then compares the multiplicative medical anchors with a fixed orthogonal rotation preserving their full Gram geometry. This is a controlled experiment, not a claim that multiplication, FiLM, attention, or prompt conditioning is itself novel.

The 16 CAM coordinates are learned latent subspaces. They are not ROIs, anatomical regions, or independently validated disease concepts. No new input graph, hierarchy, prototype classifier, memory, prompt labels, or second classifier is added.

## Source audit

- SPromptGL MICCAI 2025, sections 2.1-2.2: GFR builds modality-specific patient graphs and propagates through GCN; CAM uses InputLayer, Encoder, FeedForward and OutputLayer; PEN's SPGS computes prompt-conditioned local/global semantic selection (Eq. 5), and RES injects selected local features into the global feature (Eq. 6). The original code has eight fixed English prompts and frozen CLIP RN50 text features. E81.1 retains the E74.4 PEN-free CAM and does not re-enable PEN.
- Root network.py and layers.py were checked against the paper's CAM and PEN definitions: the root PLN contains the original InputLayer, Encoder, FeedForward, and prompt-guided PEN path, while layers.py defines the attention blocks and OutputLayer. The actual E74.4 base uses the locked PEN-free PLNNoPEN in the read-only E51 vendor snapshot at experiment/experiment72_mgdss/.runtime/vendor/root/experiment/experiment51_dugsa/dugsa.py; its InputLayer/Encoder/FeedForward produce [871,4,36], then Outputlayer receives the flattened [871,144] vector. E81.1 hooks this real Outputlayer input and keeps the original Outputlayer. E74.4 forward_state computes classification with the inherited weighted NLL, graph, pair, adversarial, and preservation terms.
- E74.4-G2: run.py -> prepare(fold) -> soft_positive_dcr.forward_state() -> rudag_forward() -> PLNNoPEN -> OutputLayer; the actual total is classification + graph + 0.05*pair + 0.01*adv + 0.01*preserve. Each task restores E61 epoch-10 model, optimizers, RNG, split, pairs, and Positive-DCR state.
- E81: its completed 30-fold code and independent audit were read-only. Its OutputLayer forward-pre-hook already gives one end-to-end DARG/CAM pass; this integration is retained.
- KD-Brain's `Attention_AggregateLayer` projects text through `text_projector`, applies `tanh`, adds it through `text_gate` to `self_ft`, and then computes Query-Key attention. E81.1 tests additive versus multiplicative conditioning in an unrelated CAM latent coordinate space; it does not reproduce KD-Brain's subnetwork identity model or clinical-prior constraint.
- BrainHO's `HierarchicalAttention` uses hierarchical local/group/global attention and a larger Transformer block; its accompanying losses and sparsemax support high-order structure. E81.1 only uses a single local relation read and does not claim hierarchical organization or ROI structure.
- ItemizedCLIP uses item-level text-conditioned visual representations with completeness/independence objectives. ABIDE has no patient-by-prompt finding labels, so E81.1 does not copy itemwise supervision or impose prompt orthogonality.

## Model and groups

The base stream remains DARG320 -> CAM144 -> PRCL-v2 -> original OutputLayer. A local token is LayerNorm(W_h h + position), with 16 tokens of width 32 derived from the 16 reshaped 9D CAM coordinates. Candidate pair features produce a 32D relation Key; a directed Value carries the receiving local, elementwise product, and signed difference. Each patient, head, and receiving local softmaxes over 15 other locals plus one zero-Value Null.

G1 uses eight fixed nonsemantic head identities and no text term. G2 uses the E81 medical anchors additively. G3 uses the same anchors multiplicatively on the patient Query. G4 uses the G3 formula with T_med @ R; R is a private-seed orthogonal transform in the frozen 16D row space. G4 is called a same-Gram rotated pseudo-semantic control, not a natural-language anonymous control. It controls the anchors' internal Gram geometry but cannot establish semantic causality by itself.

Shared v2 repairs:

- u = alpha * tanh(output(relation_message)), b = 0.1*||z_cam||, and delta = b*u/sqrt(b*b + ||u||^2 + eps), per patient. alpha = 0.1*sigmoid(alpha_gate(z_cam)), initialized to 0.02. This gives a smooth relative bound without hard saturation at the cap.
- fusion (96->32) and output (32->9) have no bias. Their input nonlinearity is GELU; Null-only therefore produces exactly zero residual.
- Group interaction is c=(mA-mN)*tanh(mA+mN) followed by a bias-free linear map and GELU.
- L_total = L_DARG + 0.001*L_res; the residual term includes only legal training patients and excludes the two historical Fold-2 train/test-overlap rows.

The unchanged original optimizers step in historical order before the fourth PRCL AdamW (lr=0.0005, weight_decay=0.01). All groups use 450 epochs maximum, original early stopping, and strict test ACC checkpoint improvement. G2/G3/G4 share all trainable parameters and same-fold initialization; G3/G4 differ only in the frozen anchor values.

## Evaluation and limits

Primary comparisons are G1-DARG, G2-E81-G2, G3-G2, G3-G4, G3-G1, and G3-DARG. Report ACC, hard-prediction balanced accuracy (historical Hard-AUC), probability ROC-AUC, ASD sensitivity, TD specificity, and ASD-F1 as ten-fold mean and population SD, with fold-paired descriptive intervals and W/T/L. Also report nine folds excluding Fold 2.

The historical evaluation protocol is retained: validation equals test, test ACC selects the checkpoint, preprocessing/RFE occurs before CV, and the 871-subject graph is transductive. Fold 2 has two training/test-overlap rows (377 and 407); original DARG uses its historical path while the new residual penalty excludes these rows. Ten folds share training subjects, so intervals are descriptive rather than independent clinical inference.
