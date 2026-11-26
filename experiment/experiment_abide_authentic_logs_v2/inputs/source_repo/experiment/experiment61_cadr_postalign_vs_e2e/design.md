# Experiment 61 Design Freeze

## Question and locked reference

Experiment 61 compares two ways to route supervised disease relations into the existing four-modality CADR-GFR patient graph. E1 waits for an aligned, patient-disjoint OOF estimate and then uses fixed fold-level reliability. E2 skips OOF and lets the outer-fold ASD/TD classification objective learn one scalar blend per modality. There are exactly two formal groups and twenty outer-fold classification tasks.

The sole reference is Experiment 53.3 group `CADR_WD_0p01_GFR_WD_0p11`, from locked source commit `b7e477c9657422a119adc7872ccf8c8188871e9e`. The user-supplied 39-character prefix resolved to this full 40-character Git object ID. The five requested 53.3 source files were read from `/model/spromptgl/experiment/experiment53_3_cadr_graph_fallback_fix` and their SHA256 values match the locked Git objects. Ten base checkpoints, task records, 43.1 fold anchors, reference metrics, data files and the 51 source link are read-only inputs.

## Source audit

SPromptGL Section 2.1 describes modality-specific patient relations, graph propagation over concatenated multimodal node features, and learned fusion. Experiment 61 retains the actual 53.3 implementation: four threshold-0.9 `GraphLearn` matrices, the original graph construction loss, the four 320-to-320 GCNs, sigmoid fusion weights and the unchanged `PLNNoPEN` CAM/classifier. The raw modality order is PHENO48, ANAT6, FUNC10, CORR256; the 871-node label vector is ASD=0, TD=1.

The 53.3 CADR source has four `Linear(d,32)-GELU-LayerNorm-L2Norm` relation encoders, four `65->32->1` scorers and a shared four-way modality discriminator conditioned on a two-value same/different code. Pair features concatenate absolute difference, elementwise product and cosine. The discriminator receives CORR features detached and reverses gradients for PHENO/ANAT/FUNC at scale 0.03. This is relation-conditioned modality adversarial learning; there is no explicit per-patient-pair CORR teacher consistency loss.

The disease graph starts from mutual top-12 support within each modality's original GraphLearn candidates, with the source-defined feature-cosine fallback when candidate coverage is sparse. The scorer output becomes `q=sigmoid(logit)` in E61, without old Platt parameters. Edges with q>=0.55 or q<=0.45 receive the source confidence factors; uncertain edges are skipped. Similarity, confidence, reliability and sigmoid edge gate determine weights. Positive and negative matrices are normalized separately; the negative GCN message subtracts the identity message and is scaled by 0.25. Direct weighted neighbor difference is bounded by block RMS and scaled by 0.02. When E1 reliability is zero, the exact same original GraphLearn adjacency and GCN path is used, preserving the 53.3 fallback behavior.

The old `_patient_disjoint_oof` trains four independent single-modality relation models, has no conditional adversarial alignment, and selects epochs using validation AUC. E61 does not call it or read its cache. E1 trains one joint four-modality encoder/scorer/discriminator per inner split for exactly ten epochs, then evaluates once at that fixed endpoint. Validation labels are used only for OOF metrics. The OOF models produce no main-model edge scores.

All CADR-added parameters, including bias and LayerNorm parameters, use AdamW lr=0.001 and WD=0.01; this matches the actual WD sweep behavior, which overrides both historical decay/no-decay groups. GFR and CAM remain Adam lr=0.0038, WD=0.11. The original first ten epochs run once per outer fold; classification and relation paths, dropout call order and loss/optimizer steps match the base forward. OOF-only reliability-gate state is fixed to a documented neutral value until E1's post-warmup OOF is available. The E2 adaptive gate is a non-random scalar initialized to sigmoid 0.1 and is inactive in warmup.

## E1 and E2

E1 runs the patient-disjoint three-fold OOF after shared warmup. Each inner training pair endpoint belongs only to its inner training partition; each validation pair endpoint belongs only to its validation partition. All four encoders, scorers and the same conditioned discriminator train together with BCE, GRL adversarial and relation-preservation terms. AUC>=0.55 and positive same-minus-different mean logits are both required for nonzero reliability. Reliability is fixed for all outer epochs after OOF. The main scorer uses uncalibrated sigmoid logits and is trained jointly as before.

E2 reads the identical shared warmup state and performs no OOF. It computes the same candidate support, logits, q, signed graphs and CADR messages as E1, with reliability multiplier one. The unchanged original graph message and CADR disease message are blended by four learned scalars initialized to 0.1. Classification gradients reach active edge weights, scorers, relation encoders and scalar gates. The hard threshold that chooses graph topology remains nondifferentiable.

## Initialization, RNG and evaluation

Each fold is reconstructed from the Experiment 43.1 untrained anchor. The 53.3 constructor order supplies identical GFR/CAM parameters and initial RNG. CADR module initialization runs inside a forked torch RNG scope with the 53.3 seed; relation encoders, scorers, discriminator and edge gates therefore retain the seed-derived values. Only the old OOF-dependent reliability buffer/calibration/gate initialization is replaced. The E2 scalar gate is deterministic and consumes no RNG. OOF pair sampling uses local NumPy generators and OOF torch initialization/training uses an isolated torch RNG scope. CPU/CUDA main RNG hashes are saved after warmup and at each outer epoch to verify matched stochastic call traces after the E1/E2 fork.

The historical 10-fold split, pre-CV RFE/standardization, transductive 871-patient graph and `val=test` checkpoint rule remain unchanged. E61 intentionally does not use E53.3 Platt calibration because E2 has no OOF; both E61 groups use the same uncalibrated sigmoid score. Historical comparison is consequently exploratory and not an unbiased external test.

## Paper citation

Wan Xixi et al., “SPromptGL: Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease,” MICCAI 2025, Section 2.1: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf. The paper's GFR description is used to frame the source audit; Experiment 61 retains the checked 53.3 code behavior rather than importing unimplemented paper equations.
