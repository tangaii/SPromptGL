# Experiment 60 Design Freeze

Status: frozen before implementation and formal training. This document and `config.json` define the only planned method variants. `knowledge_base.json` is frozen separately and hashed before encoding. No outer-test result may change these choices.

## Question and comparison base

Experiment 60 asks whether patient-specific projection onto a small set of medically described ASD and typical-control functional-connectivity concepts improves a strong CADR-GFR/CAM classifier, and whether any increment comes from the medical text, the partial-set constraint, or simply additional capacity.

The sole base is Experiment 53.3's third fallback-WD sweep group, `CADR_WD_0p01_GFR_WD_0p11`. Its optimizer settings are CADR AdamW weight decay 0.01 in both stored parameter groups, GFR Adam weight decay 0.11, and CAM Adam weight decay 0.11. The historical source/configuration signatures match the signed sweep manifest, all ten selected-group checkpoint hashes match, and all ten reference anchor hashes and the two ABIDE input hashes match the historical manifest. The 53.3 sweep's 40 checkpoint audits passed. The older top-level 53.3 config's CADR WD=0.001 is not this selected group.

Every new fold reconstructs the untrained 43.1 fold anchor and the same CADR initialization, split, fixed patient-pair cache, relation OOF cache, Platt calibration, reliability and construction RNG. Historical trained checkpoints are read-only references; they are never used to initialize a new group. The separate base source snapshot is byte-for-byte copied and its original SHA values are recorded in `manifest.json`.

## Paper and history audit

| Source | Audited method and boundary | Experiment 60 use |
|---|---|---|
| SPromptGL, MICCAI 2025, paper and repository `network.py` | The paper's GFR/CAM/PEN description is not identical in every detail to this repository's executable code. Original PEN encodes eight generic brain-condition phrases with frozen CLIP RN50; a trainable 4096→128→9 MLP forms two text vectors; CAM's 4×4×9 positions are reweighted by maximum two-class softmax similarity and receive the original pooled residual. It is not a disease-prototype contrastive loss. | R02 reproduces that source-code path and text bank on top of the PEN-free CAM; it is labelled an original-code adaptation because the classifier/base are 53.3. |
| BrainPrompt, MICCAI 2025, paper, repository and ABIDE label file | The ABIDE label file contains two sentences, one for TC and one for ASD, whose distinction is the class name. The disease-level branch uses frozen text vectors, a shared projection, normalized cosine logits and cross-entropy. Its original encoder is LLM2Vec; this run uses CLIP RN50 for one fixed encoder across groups and is therefore a BrainPrompt-style adaptation, not an official reproduction. | R03 uses the exact two ABIDE sentences with the official code's `lambda1=0.001` default. R04 uses the same cosine-CE structure and coefficient with the frozen medical class centroids. |
| SABER, ICME 2026 paper-equation reimplementation | Module III uses a graph/subject representation as query over patient-level text and adds a gated residual; the separate disease-text objective is contrastive supervision. The original patient-text retrieval target is not the disease bank. No verifiable official implementation was identified. | R05 queries only permitted PHENO-derived age-bin, site and sex text tokens, and separately uses disease-text contrastive supervision. Its coefficient is calibrated to a fixed 0.1 train-only gradient-norm ratio against base classification on fold 1. R06 changes the retrieval target to disease text and is explicitly a modified adaptation, not SABER. |
| BrainPrompt+, IEEE TMI 2026; HP²L, Medical Image Analysis 2026 | Multi-level language prompts, hierarchical learnable prompts and disease prototypes already exist in brain/medical imaging. Prototype mixtures or constrained sets are not claimed as generally new. | DSS-PEN's testable distinction is patient-specific partial projection onto fixed medical concept sets, with a reject state, a set-distance margin and a bounded pre-classifier correction, compared with matched class-only, wrong, random, learned, single-prototype and ordinary-attention controls. |
| Ilioska et al., Nature Mental Health 2026 | In a separate multisite cohort with a different parcellation, individual FC deviations were heterogeneous; common regional/network convergence coexisted with that heterogeneity. Network-level results do not certify any edge or ROI in the present data. | Supports testing a set rather than asserting one ASD edge direction. It is motivation only, not an ABIDE feature label or biomarker claim. |
| Experiment 1–18, 41/41.1, 50.2, 58 and 59 | Prior work reports inactive or near-identity semantic paths, weak or absent label effects, text not beating matched numerical controls, a sparsely supported anatomical hypothesis, and residual saturation/near-zero interventions. Experiment 41.1 and 58 reached NO-GO under their own gates. | These are reasons to test gradient flow, coefficient geometry, rejection rate, residual magnitude and checkpoint interventions. They do not establish that every prompt method is ineffective. |

Primary source links and source-file hashes are recorded in `manifest.json`. The BrainPrompt label order is normalized to ASD=0, TD=1 in code; its published TC/ASD text is not silently rewritten.

## Data and semantic scope

The input remains 871 transductive patients, ASD=0 (403) and TD=1 (468), in the historical PHENO48 / ANAT6 / FUNC10 / CORR256 layout. PHENO contains categorical two-year age bins, site indicators and sex indicators. It contains no clinical symptom scale in the model input. R05 creates three independent patient-text tokens from those existing fields only; it never uses diagnosis, split identity, test-set statistics or label-derived prose.

The current CORR representation has 256 selected correlations. The ROI audit names 252 features and leaves four unknown; a previous experiment found only one directionally supported, anatomically compatible selected connection. DSS-PEN therefore uses broad, direction-agnostic connectivity findings and heterogeneity evidence, not fabricated ROI/network labels. Every ontology item records whether the present feature set can observe it and the cross-atlas/cohort limitation. No clinical symptom statement is presented as observed in this matrix.

`knowledge_base.json` holds six distinct ASD concepts and six typical-reference concepts with sources and evidence limits. Six is the largest frozen count for which the reviewed evidence supports distinct descriptions without manufacturing ROI claims or filling slots with paraphrases. The separate R08 bank deliberately uses only neutral class-name templates.

## Frozen DSS-PEN definition

CADR-GFR emits `X ∈ R^(871×320)`. The unmodified PEN-free CAM produces `H ∈ R^(871×4×36)` after its existing input projection, encoder and feed-forward block. DSS-PEN reads that tensor and emits `H'` with the same shape. The existing OutputLayer receives `flatten(H') ∈ R^144`; no post-classifier head is added. Its `attn_embedding` argument remains unused as in source.

Patient and text mappings are shared and small:

\[
z_i=\operatorname{normalize}(f_\theta(\operatorname{vec}(H_i))),\qquad
P_c=\operatorname{normalize}(W_t E_c),
\]

where `fθ` is Linear(144,128)→GELU→Linear(128,64), `E_c` is six frozen OpenAI CLIP RN50 text vectors, and one shared bias-free Linear(1024,64) is `W_t`. No class-specific text MLP is used. `L_geo` is the mean squared difference between the 12×12 cosine Gram matrix before and after `W_t`, using the frozen bank for that group (including its corresponding control bank). Prototype distances, Gram geometry, numerical rank and cosine distributions are logged.

For each class, the default feasible set is `a ≥ 0, sum(a) ≤ 1`. The deterministic 128-step FISTA solver minimizes `||z-Pa||² + 0.05||a||²`; its step size uses the Frobenius upper bound on the Hessian Lipschitz constant. It is unrolled with autograd and projection onto the stated feasible set. `d_ic = ||z_i-P_c a*_ic||²` excludes the coefficient regularizer. A zero vector is feasible and is the explicit empty-match state. There is no `R_c` incompatibility constraint because the reviewed evidence does not establish mutually exclusive concepts.

The training-only set loss is the class-balanced mean of `relu(0.1 + d_i,y - d_i,1-y)`. `L_geo` is the shared geometry regularizer above. Their coefficients are fixed by the train-only gradient calibration rule in `config.json`, once on fold 1 before any formal fold; no validation/test metric enters that calibration. The total objective is the exact 53.3 objective plus the frozen new terms.

At inference, `π_ic = softmax(-d_ic / 0.2)`. The absolute match gate is `g_i = clamp(1 - min_c d_ic, 0, 1)`; when both classes have no better match than the empty solution (`d ≥ 1`), it is exactly zero. The shared Linear(64,144) correction is zero-initialized and reshaped to four 36-D positions. A per-position norm limiter caps the unscaled correction at 0.1 times the corresponding CAM norm. After the inherited 10-epoch warmup, the residual scale ramps linearly from 0 at epoch 10 to 0.1 at epoch 20, then remains 0.1. Thus no initial semantic residual changes the base forward, while `L_set` trains the projection before the correction layer opens. All classes are evaluated without labels at inference.

No auxiliary loss is allowed to update shared GFR/CAM during the inherited 10-epoch warmup; semantic losses use detached CAM features in that interval. The original base losses, optimizer steps and stochastic forward order remain in place. New deterministic encoders/solvers do not consume the shared RNG. In active epochs, semantic loss gradients may update shared parameters as intended.

## Formal groups

R00 is the read-only selected 53.3 reference. R01 is an unchanged base replay. R02–R06 are direct-method or modified-method controls. R07 is the only complete DSS-PEN. R08–R18 each retrain from the same fold initialization and change only the named semantic source, projection/constraint, loss or correction scope. The exact formulas and enabled/disabled components are in `config.json`.

| Group | Frozen contrast |
|---|---|
| R01 `E60_BASE_REPLAY` | Exact 53.3 third-group replay; no new module or loss. |
| R02 `E60_ORIGINAL_PEN` | Original SPPromptGL SPGS+RES source-code path and eight original generic CLIP phrases. |
| R03 `E60_BRAINPROMPT_LABEL` | Original ABIDE TC/ASD text and cosine CE; no residual. |
| R04 `E60_BRAINPROMPT_MEDICAL` | R03 loss with the two medical class-centroid vectors; no residual. |
| R05 `E60_SABER_DECISION` | PHENO patient-text cross-attention, gated decision residual and disease-text contrastive CE. |
| R06 `E60_DISEASE_QUERY` | SABER-style equation adapted to retrieve disease text rather than patient text; gated residual plus disease CE. |
| R07 `E60_DSS_FULL` | Medical six-concept sets, partial constrained projection, empty match, set margin, geometry loss and bounded correction. |
| R08 `E60_DSS_CLASS_ONLY` | Six class-name-only neutral prompts per class. |
| R09 `E60_DSS_WRONG_SEMANTICS` | ASD and TD medical banks swapped. |
| R10 `E60_DSS_RANDOM_ANCHORS` | Fixed random vectors with the real bank's exact normalized Gram matrix and randomized orientation. |
| R11 `E60_DSS_LEARNED_PROTOTYPES` | No text content; fixed orthogonal slot codes passed through the same trainable shared adapter to yield learned prototypes. |
| R12 `E60_DSS_SINGLE_PROTOTYPE` | Each class centroid repeated across six slots, preserving adapter and solver shapes. |
| R13 `E60_DSS_SOFTMAX_ATTENTION` | Per-class ordinary softmax text attention replaces the constrained projection. |
| R14 `E60_DSS_FORCE_MATCH` | Projection mass is constrained to equal one; no empty match. |
| R15 `E60_DSS_FREE_SET` | Nonnegative per-concept box `[0,1]`, removing the row-mass budget while retaining the 0.05 stability term. |
| R16 `E60_DSS_NO_SET_LOSS` | Projection, geometry loss and residual retained; set-distance margin removed. |
| R17 `E60_DSS_NO_RESIDUAL` | Projection and set/geometry losses retained; correction is disabled. |
| R18 `E60_DSS_CORR_ONLY` | Query input is masked to CAM position 4 (CORR-labelled slot); only that slot can receive correction. CAM slots are post-attention latent positions, not independent ROIs or causal modality pathways. |

R08–R18 use the same `fθ`, `W_t`, correction parameter shapes, coefficient initialization seed, fixed six-slot bank shape and training budget wherever the component exists. R11 uses a frozen orthogonal code bank rather than medical text; R12 has repeated equal anchors; these intentional structural differences are recorded, not described as identical effective capacity. R02's original PEN parameters join the existing CAM Adam group at WD 0.11, matching the source's single CAM optimizer. Other new modules use a separate AdamW at LR 0.001 and WD 0.001. The 53.3 optimizers remain GFR Adam (0.0038/0.11), CAM Adam (0.0038/0.11), CADR AdamW (0.001/0.01 in both groups).

## Reproducibility and parity gates

The inherited protocol is fixed: seed 0, `StratifiedKFold(10, shuffle=True, random_state=10)`, global historical RFE/standardization, transductive all-node graph, validation equal to test, 10-epoch warmup, at most 450 epochs, checkpoint on strict ACC improvement, and stop when wait exceeds 200. These choices preserve historical comparability and make the results optimistic for external generalization; they will be stated plainly.

Before R01, preflight must verify 10/10 base initialization, split, pair/cache and main-RNG hashes; actual optimizer WD; base loss/forward/first-step parity; zero-residual parity; no shared RNG change from semantic module construction/forward; solver constraints, finiteness and convergence; and atomic save/rename/fsync/load. R01 then must reproduce all ten historical selected-group trajectories/checkpoint states within the frozen numerical tolerance before any R02–R18 training is launched. A failed R01 gate halts the formal comparison and is reported without relabelling partial results.

Each task has its own process, model, optimizer, RNG seed derived from experiment/group/fold, logs, task record and checkpoint. Construction of additional modules is inside a forked RNG scope and restores the base RNG exactly. Existing 53.3 data, sources, caches, checkpoints and handoff history are read-only. Only Experiment 60 owns new files until the final append to handoff.

Every fold records ACC, historical hard-prediction AUC, continuous ASD-probability AUC, TD specificity, ASD sensitivity, balanced accuracy, ASD F1, TP/TN/FP/FN, best/executed epoch and checkpoint SHA. Summary is mean ± population SD in percentage units. Paired differences use all ten folds; two-sided exact sign-flip tests (`2^10` assignments), fold-bootstrap 95% intervals (10,000 resamples, fixed seed) and Holm correction per family of group comparisons are reported as exploratory, not definitive inference.

## Frozen scientific decision rules

- `GO_BASE_PARITY`: 10/10 initial-state/RNG/cache/split parity and 10/10 R01 historical checkpoint-state and prediction replay within `rtol=1e-6, atol=1e-7`; otherwise halt.
- `GO_DIAGNOSTIC`: R07 mean ACC improves R00 by at least 0.3 percentage points, wins at least 7/10 folds, paired bootstrap lower 95% bound is above zero, and SEN-ASD does not fall by more than 1.0 point.
- `GO_SET_PROJECTION`: R07 meets the diagnostic rule against both R12 and R13 and exhibits non-collapsed, patient-varying coefficients in at least 8/10 folds.
- `GO_PATIENT_HETEROGENEITY`: in at least 8/10 folds, at least two concepts each carry ≥10% of accepted ASD mass, no concept carries >80% of accepted ASD mass, and at least 20% of ASD patients select different nonempty support patterns; all-zero/empty matches are reported separately.
- `GO_SEMANTIC_CONTENT`: R07 meets the diagnostic rule against each of R08, R09, R10 and R11. Otherwise any structural gain is not attributed to medical content.
- `GO_DECISION_INJECTION`: R07 correction-off and true-vs-wrong-text interventions produce finite, non-identical logits; report probability shifts and flips. Intervention evidence cannot substitute for the trained R16/R17 contrasts.
- `GO_VS_BRAINPROMPT`: R07 meets the diagnostic rule against R03 and R04. `GO_VS_SABER`: R07 meets it against R05 and R06.
- `GO_MICCAI_CANDIDATE`: all above scientific gates pass and 180/180 checkpoints pass independent audit. Any failed gate yields NO-GO for a medical-semantic contribution; uncertainty is labelled inconclusive.

No hyperparameter, text, dimension, ontology or gate may be changed after outcome inspection. One higher ACC cannot turn all gates into GO.

## Reporting limits

This is a single-seed internal comparison under `val=test`, global pre-CV preprocessing and a transductive 871-node graph. CLIP RN50 is not a clinical language model. The ASD findings were reported in cohorts/atlases different from this selected 256-feature ABIDE representation. PHENO patient text exposes site, which can carry acquisition/dataset confounding; R05 is a method-equation adaptation and not evidence of clinical text alignment. Attention/projection weights, gradients and fixed-checkpoint interventions are model diagnostics, not causal explanations or validated biomarkers.
