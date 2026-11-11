# Literature and source review for Experiment 78.1

## SPromptGL

The primary source is the MICCAI 2025 paper, [Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), with project code at [tangaii/SPromptGL](https://github.com/tangaii/SPromptGL) and [wanxixi11/SPromptGL](https://github.com/wanxixi11/SPromptGL). The paper couples a learned disease relation graph and multi-modal GCN with text concepts encoded by CLIP. Its semantic prompt graph selection uses image/text similarity; CAM and the prompt-ensemble/semantic-guided fusion then provide class evidence. This is not a bank of patient-supported disease subtypes. The current repository's `network.py` stores eight fixed `attr_words` in TD/NC-first order. The E74.4 base instead instantiates the copied `PLNNoPEN` CAM and does not activate the original root model's PEN/SPGS forward. E78.1 therefore reuses the exact eight strings as frozen semantic coordinates; it does not claim to reproduce the complete SPromptGL architecture.

## BrainPrompt

The full MICCAI 2025 paper, [BrainPrompt: Multi-Level Brain Prompt Enhancement for Neurological Condition Identification](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), and the official [AngusMonroe/BrainPrompt implementation](https://github.com/AngusMonroe/BrainPrompt) were inspected, including the method figure, equations, and `nets/brainpromptg_net.py`. The implementation injects ROI text prompts into message passing, builds subject-level prompt relations from demographic/site features, and applies disease-prompt classification supervision. Its code uses a frozen LLM2Vec text encoder and does not implement patient-supported disease prototypes or subtype-conditioned prompt selection. It is a relevant prompt-integration reference, not a direct implementation template for E78.1.

## HP2L

The publisher record for [A Hierarchical Prompt and Prototype Learning Framework for Brain Disorder Classification](https://doi.org/10.1016/j.media.2026.104063) identifies a hierarchical diagnosis setting, prompt tokens, and disease prototypes. The accessible ScienceDirect page exposed the bibliographic record, abstract, and highlights; a full-text/PDF request returned HTTP 403 in this environment. Thus the original full method figure, equations, and source implementation could not be independently inspected here, and no detailed HP2L mechanism is inferred beyond the publisher-visible material. No accessible official code repository was identified during this audit. This is an explicit literature-audit limitation.

## E74.4 and E78 source mechanics

The E74.4-G2 code path was inspected from its local run/model files and fold artifacts. The model is DARG/GFR + Positive-DCR + CAM `PLNNoPEN`; its three original optimizers are restored with their full states. The CAM output pre-hook receives a `[871,144]` tensor immediately before `OutputLayer`; the layer then maps 144 to 36 and 36 to two log-softmax outputs. The original `ClsLoss` is weighted NLL over the training indices. E78 captured that pre-hook input but added an end-stage bounded log-score correction, learned free text projections and static, PCA-ordered class subdivisions. The E78 diagnostics document near-identical within-class prototypes, high site association, near-uniform routing, negligible subtype-specific prompt TV, and no gain over anonymous text.

The E78.1 design maps these failure modes to explicit changes: train-support site-shrunk residual anchors with deterministic clustering; separate semantic and heterogeneity projections; dynamic weighted prototypes with exact support-query LOO; per-subprototype prompt softmax in one frozen joint text geometry; and CAM144 feature residual fusion before the original OutputLayer. The text geometry is checked against all 16 fixed CLIP vectors before training. The full random stream and historical optimizer update order are tested by exact replay; initialization and clustering use deterministic or private RNG operations.

## Identity and evidence boundaries

The E78 branch/commit requested by the study was verified remotely with `git ls-remote`: `experiment76-5-scpr-upload` points to `563234e125b7c36c630b5b36c8bb04810d541a54`. The local Git object database could not fetch/index that tree because the preferred OSSFS mount returned EIO; the server-mounted E78 source and results were read directly and remain read-only.

The processed ABIDE table has 871 rows and contains feature columns plus labels, but no explicit subject-ID column. E78.1 therefore verifies exact row/value equality against the historical loader's 871x320 feature and 871-label arrays, and verifies one SITE_ID and one AGE_AT_SCAN one-hot value per row. This proves the alignment used by the model and metadata table, but it does not create a subject-ID crosswalk that the processed source does not contain.

The historical protocol includes `val=test`, preprocessing/RFE before outer CV, a transductive full-patient graph, test-fold ACC checkpoint selection, and the known fold-2 overlap. These are retained for direct comparison. Result tables are not unbiased external-generalization estimates.
