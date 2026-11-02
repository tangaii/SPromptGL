# Experiment 58 Design Freeze

## Baseline audit

The sole training base is the Experiment 53.3 fallback-WD sweep group `CADR_WD_0p01_GFR_WD_0p11`: CADR AdamW decay 0.01 in both stored parameter groups, GFR Adam 0.11, and CAM Adam 0.11. Its 10 independently read-back audited checkpoints and exact fold metrics are read-only references; Experiment 58 rebuilds every fold from the Experiment 43.1 untrained anchor. The sweep runner wraps the Experiment 53.3 fallback branch; the older `config.json` value 0.001 is not the selected setting.

Dataset and fold inputs under `/data/adodas/SPromptGL-main` were byte-compared with the audited `/model/spromptgl` copy before use. The two mount roots are distinct filesystems. The Experiment 51 private source, Experiment 43.1 anchors, and sweep reference checkpoints remain read-only under `/model/spromptgl`.

The base uses 871 transductive nodes; labels are ASD=0 and TD=1; inputs are 48/6/10/256. Four GraphLearn graphs and four anchor-loaded 320x320 two-layer GCNs feed the repaired four-way GFR sum. The unchanged `PLNNoPEN` CAM has four 36-dimensional positions and no PEN. Original graph loss and CADR losses remain active. Hard-AUC remains the historical hard-prediction AUC. The 10-epoch warmup, checkpoint selection start, 450 epoch maximum, `wait>200`, and strict ACC-improvement checkpoint rule are inherited from the actual sweep runner.

## Source and method review

The local MICCAI 2025 SPromptGL paper was read through GFR equations (1)-(2), PLN equations (3)-(7), and the ABIDE protocol. Its paper-level label-supervised graph term is not silently added: this experiment keeps the audited implementation's actual graph loss. The repaired Experiment 51 implementation is used; PEN is not restored. BrainPrompt (MICCAI 2025) uses actual AAL ROI descriptions, while BrainPrompt+ (IEEE TMI 2026, DOI 10.1109/TMI.2026.3692958) motivates multilevel knowledge prompts. Neither paper's atlas-specific claims are transferred to this dataset. The 2026 Ilioska study used Schaefer-400 plus Harvard-Oxford subcortical parcels, and found individual edge deviations highly heterogeneous; its network-scale findings are not treated as edge-level evidence here. MICCAI 2026 testable-concept work motivates direct interventions and explicit faithfulness checks, not any ASD claim.

## Frozen evidence scope

Experiment 11 verified the numeric order and endpoint labels for 252/256 selected CORR features. Features 195, 225, 232, and 236 touch unknown atlas code 3455 and are excluded from prompt matching while remaining in all 256 base CORR features. Only one directional ASD hypothesis has a direct, traceable match in the selected feature set: greater ipsilateral thalamus-to-superior-temporal functional connectivity (Nair et al., 2015, DOI 10.1002/hbm.22938). The study explicitly used Harvard-Oxford cortical/subcortical regions and reports greater connectivity for bilateral superior temporal gyri; it is a small pediatric cohort and does not establish an individual ASD biomarker. Its mapping to the present left-thalamus/left-anterior-STG connection is anatomically compatible, not an exact equivalence of estimands. Thus the actual candidate support is one of 252 named connections; other candidates reject. The decrease-direction text is a counterfactual, not a second literature claim.

The processed PHENO fields are categorical 2-year age bins, site indicators, and sex indicators; no continuous age is inferred. A ridge conditional mean model uses only outer-train TD rows. Five-fold cross-fitted TD residuals estimate scale for training TD subjects; a variance floor and shrinkage prevent unstable divisions. Outer-train ASD and all outer-test subjects use the final model fit on outer-train TD. No outer-test labels enter fitting, matching, gating, or losses.

## FAVE-PEN definition

The fixed direction threshold is `|z| >= 2.3`, used as a conservative extreme-deviation gate inspired by the 2026 normative FC study, not as a p-value in this dataset. A sparse anatomy mask permits only the verified feature-0 edge and its increase/decrease language variants. A per-patient softmax over the allowed variant(s) plus a reject slot is a row-mass-bounded partial assignment; disallowed cells are exactly zero. Evidence state is support, contradiction, or unknown according to the pre-registered direction and deviation threshold. Unknown cases train the reject option. The assignment and learned projections are differentiable.

The evidence encoder maps the fold-local deviation and auxiliary quality into 32 dimensions; a frozen CLIP RN50 text embedding is projected to the same space. `L_match` is balanced across observed evidence and unknown/rejected rows. `L_verify` computes the signed interaction `(S(z,p+)-S(0,p+))-(S(z,p-)-S(0,p-))` only on extreme, anatomically eligible rows. Its target sign follows the observed deviation, testing computational direction fidelity rather than biological causality. ANAT/FUNC produce a sigmoid quality gate that only scales the new evidence residual.

After the final CAM encoder, the evidence vector projects to 36 dimensions and modifies only CAM position 4 (CORR): `h' = h + 0.1*RMS(h)*tanh(alpha*q*delta/RMS(h))`; `alpha` initializes to zero. The original OutputLayer is reused. E58-NUM has the same parameters, initialization, masks, and objectives, but substitutes a fixed, seeded, normalized anonymous text bank; it still retains the same anatomy/direction prior and is not a no-medical-knowledge control.

Only two groups train: E58-FULL and E58-NUM. Both start from identical per-fold GFR/CAM/CADR and FAVE parameter initialization. FAVE AdamW is separate (`lr=0.001`, matrix decay `1e-4`, bias/norm decay 0); `lambda_match=0.05`, `lambda_verify=0.01`, and residual cap 0.1 are frozen before training. FAVE is inactive throughout the inherited 10-epoch warmup. Its operations use no dropout or sampling and do not consume the shared RNG. The original CADR, GFR, and CAM optimizers and shared random-operation order are preserved.

## Limits

The prior feature selection/standardization is CV-before-fold and the graph is transductive; `val=test` still selects checkpoints. These are historical protocol-matched exploratory comparisons, not unbiased generalization estimates. The single-seed, one-edge semantic coverage is deliberately narrow. No edge-direction hypothesis is interpreted as a cause, diagnosis, or validated patient-level biomarker.
