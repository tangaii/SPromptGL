# Experiment 78.6 — PEN-SVA design and audit record

## Scope and output identity

PEN-SVA is a frozen, zero-training audit of the original SPromptGL disease-Prompt mechanism. No E78.x model was substituted and no historical source, data, or Checkpoint was modified. The preferred OSSFS directory `/data/adodas/SPromptGL-main/experiment/experiment78_6_pen_semantic_audit` failed with errno 5; the single authoritative output directory is `/model/spromptgl/experiment/experiment78_6_pen_semantic_audit`.

## Source and paper audit

The SPromptGL paper and official repository were used as read-only references. The protected local `network.py`, `model.py`, `layers.py`, and `utils.py` were read directly; their SHA-256 values are in `manifest.json`. The verified source path is GFR → CAM → PEN/SPGS → RES → OutputLayer. The original prompt strings and order are preserved, including the spelling `Nromal brain volume`. The local source evidence is line-recorded in `manifest.json`: frozen RN50 text buffer (line 91), 4096→128→9 MLP (line 86), CAM reshape 871×4×4×9 (line 110), cosine normalization and similarity (111–119), class softmax (126), class-axis max (129), SPGS scaling (132), AvgPool (84), RES `Vsim` (144), residual addition (147), and OutputLayer (150).

E32 B0/B1/B2, E28, E43 and E60 were treated as separate historical protocols. E60 `E60_ORIGINAL_PEN` is the only identity-consistent original-PEN best set available for this audit and uses the E53.3 CADR/GFR protocol; it is not relabeled as E32 B2.

## Mathematical audit

For class similarities `s_A` and `s_T`, the original operation is

`softmax([s_A,s_T])_j = exp(s_j)/(exp(s_A)+exp(s_T))`, followed by `w=max_j softmax_j`.

Therefore `w = exp(max(s_A,s_T))/(exp(s_A)+exp(s_T)) = 1/(1+exp(-|s_A-s_T|)) = sigmoid(|s_A-s_T|)`. With cosine similarities in [-1,1], the theoretical interval is `[0.5, sigmoid(2)] = [0.5, 0.8807970779]`. The independent numerical unit test has maximum absolute error 0. The sign of `s_A-s_T` is discarded. Swapping the two already-formed class vectors leaves `w`, SPGS-scaled features, RES output, logits, and predictions unchanged. This conclusion concerns the SPGS gate; other parts of the model may still use class-dependent parameters.

## Frozen Checkpoint replay

E60 `checkpoint_audit.csv` records 10/10 PASS identities. Ten current best files were SHA-checked against that history, copied to `/tmp/e78_6_e60_cp_run1` solely as stable read-only audit inputs, and independently loaded. Best epochs are 123, 159, 153, 207, 213, 154, 134, 190, 122, and 215 for folds 1–10. The E53.3-shaped constructor anchors in `/tmp/e78_6_pen_audit_constructor_anchors` were used only to instantiate the loader; every trained `model`, `cam`, and `semantic_module` state was then strictly replaced by the E60 Checkpoint state. No anchor was trained or used as a trained weight.

The replay reproduced A0 probabilities and all seven stored metrics in 10/10 folds. It then independently evaluated A0 FULL_PEN, A1 RES_ONLY, A2 PEN_OFF, A3 CLASS_VECTOR_SWAP, A4 ANONYMOUS_TEXT, A5 WRONG_SEMANTIC_TEXT, A6 NUMERIC_TEXT_CONTROL, and A7 NEUTRAL_SPGS from a fresh frozen state for each intervention.

## Gradient and positive-control audit

The original text MLP parameters are nonzero in all ten folds; train-label NLL gradient norms are nonzero in 9/10 folds (fold 2 is a local zero-gradient case), while the complete CAM parameter gradient norm is nonzero in 10/10 folds after correcting the probe to aggregate all CAM parameters rather than the unused historical `fusion_weight` first parameter. Frozen CLIP has no gradient. Deterministic OutputLayer and PEN-feature perturbations both produced nonzero logit changes. No parameter update was performed during this diagnostic.

## Evidence boundary

A fixed Checkpoint intervention measures dependence of a trained model on an input path; it is not equivalent to retraining with different text. A4–A6 are therefore text-dependence controls, not training-level evidence that medical text is causally superior. Historical val=test selection, test-ACC model selection, CV-before-preprocessing/RFE, and the transductive patient graph limit generalization claims.
