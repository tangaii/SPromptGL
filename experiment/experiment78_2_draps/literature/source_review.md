# Literature and source audit for E78.2

Audit date: 2026-11-12 UTC.

## SPromptGL (MICCAI 2025)

The supplied local PDF was parsed page-by-page with PyMuPDF. The paper describes Graph-based Feature Representation (GFR), a learned multi-relation graph with graph regularization and a label supervision graph (Eq. 1), followed by DGCN message propagation (Eq. 2). Its Prompt-guided Learning Network uses GPT-4-derived disease concepts, frozen CLIP text features, a Cross-Attention Module (CAM), and a Prompt-guided Embedding Network (PEN). The Semantic Prompt Guidance Scheme (SPGS) scores local representations against prompt semantics and PEN combines local/global features. The overall objective is `L=λ L_G+η L_Label` with cross-entropy label loss (Eq. 7), with λ and η set to 1 in the reported experiments. The ABIDE protocol reports 871 subjects and 10-fold cross-validation with ACC/AUC/SEN/SPE.

The accessible proceedings PDF is https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf. The paper itself and the official repository identify the code as `wanxixi11/SPromptGL`; the alternate `tangaii/SPromptGL` repository was also checked.

## BrainPrompt (MICCAI 2025)

The supplied local PDF was parsed page-by-page. BrainPrompt uses ROI-level, subject-level, and disease-level prompts. Its disease-level prompts are class descriptions such as “A brain network of a(n) [CLASS] subject ...”; a disease-prompt cross-entropy term compares subject representations with class text embeddings. The paper defines `L=L_cls+λ L_disease`. BrainPrompt's reported protocol uses 8:1:1 train/validation/test splits and 10-fold CV, with validation-based early stopping and hyperparameter tuning. E78.2 uses only the disease-level supervision motivation and does not import BrainPrompt's ROI/subject prompts or its GNN.

The accessible proceedings PDF is https://papers.miccai.org/miccai-2025/paper/2260_paper.pdf. The official repository checked was `AngusMonroe/BrainPrompt`.

## HP²L (Medical Image Analysis 2026, DOI 10.1016/j.media.2026.104063)

The DOI landing page and ScienceDirect article preview were accessible, but the full article was not available through the audit environment. The accessible abstract/preview describes a hierarchical prompt and prototype learning framework for 23 brain disorders, with class-specific prototype tokens injected into prompt tokens, prompt refinement across hierarchy levels, and EMA-style stable prototype anchors. It is a motivation and distinction reference for prototype–prompt coupling. It is not used as a source of unretrieved formulas or implementation details, and E78.2 does not copy its hierarchical ViT or top-down classifier.

Accessible source: https://www.sciencedirect.com/science/article/pii/S1361841526001313 and DOI https://doi.org/10.1016/j.media.2026.104063.

## GitHub audit

The requested repositories were inspected as read-only web sources:

- `https://github.com/tangaii/SPromptGL`
- `https://github.com/wanxixi11/SPromptGL`
- `https://github.com/AngusMonroe/BrainPrompt`

The requested reference commit `d3e78fb1337d67d0be6ec899d914f5270b7740eb` is visible in `tangaii/SPromptGL` and is the “Add experiment78_1 SH DCPSS” commit. Its E78.1 files are treated as historical evidence only. The actual E74.4-G2 implementation is loaded through the server-side SHA-checked chain, not substituted by repository-root code.

## Historical E78.1 audit finding used by this experiment

E78.1's preserved implementation was read directly. It used K-dependent site weighting (all-one weights for K=1), fixed assignment anchors, strong heterogeneity separation/utilization losses, an index-wise `proto_q[:,cls]-proto_q[:,opposite]` comparison, and one learned scalar fusion strength. E78.2 removes these mechanisms and implements the fixed DRAPS design in `draps.py`.
