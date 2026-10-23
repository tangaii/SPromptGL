# Experiment 36 — PCPR

This directory contains the isolated implementation and registered 22-group experiment for phenotype-conditioned patient relational routing. The execution stage is `/tmp/SPromptGL-E36-PCPR-stage/experiment/experiment36_phenotype_conditioned_patient_routing`; the original repository is read-only input.

The experiment keeps the original B0-A GFR/PEN-bypass/PLN path for `B0_ORIGINAL_B0A`. B1-B3 use a capacity-matched shared two-layer GCN implementation: four graphs are each propagated and then fused for B1, four raw graphs are fused before propagation for B2, and only CORR propagates the complete 320-dimensional input for B3. The four raw GraphLearn outputs and their graph losses remain present in every new structure. PCPR modulates only non-self-loop edges on the existing support; `P[i,j]` means receiver i aggregates sender j.

## Source and Prior-Work Audit

The MICCAI 2025 SPromptGL paper describes modality-specific subject relations, graph propagation, then semantic prompt guidance over local representations and relation-aware local/global fusion. The executable repository is the implementation authority here: each GFR relation is a graph over participants; all four graph branches receive the concatenated 320D feature; the `i==1` assignment replaces the `i==0` combined state; and the historical B0-A forward path bypasses the PEN while retaining its construction. Repository `GraphLearn` thresholds learned cosine similarities at 0.9, and original propagation uses symmetric `normalize_adj(A+I)`. The original `simple-2` training objective optimizes classification plus the mean four-graph construction loss; `ALoss` is computed but not included.

Phenotype-conditioned graph routing is not claimed as a new primitive. PhenoBrain conditions multi-hop communication inside each participant's brain network; PCPR instead conditions existing edges between participants. DeepASD learns modality-specific patient-similarity graphs and fuses them into a global population graph. B2P-GL uses GPT-4 atlas knowledge in brain-graph construction and then phenotypic data for population-graph diagnosis. GPGATNet uses a phenotypic skeleton with functional-connectivity edge updates. MHPG combines imaging and phenotype similarity in a hybrid participant graph. The experiment's narrow question is whether actual participant-specific CSV text, encoded without fine-tuning, improves propagation over same-source numeric conditions, patient-mismatched text, and shared constants under the frozen SPromptGL protocol. Any positive result remains specific to this setup; graph attention, graph fusion, and phenotype conditioning are not claimed as inventions.

The pinned BioLORD model is a 768-dimensional biomedical sentence encoder; its model card documents attention-mask mean pooling followed by normalization. The same model card says users must ensure valid UMLS and SNOMED CT licensing. No model download or use is allowed until both are confirmed for this project. No acceptable cached clinical encoder was found during the initial environment audit.

Primary references:

- [SPromptGL, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf)
- [PhenoBrain, ICML 2026](https://proceedings.mlr.press/v306/meng26f.html)
- [DeepASD, Translational Psychiatry 2024](https://www.nature.com/articles/s41398-024-02972-2)
- [B2P-GL, arXiv preprint](https://arxiv.org/abs/2506.16096)
- [GPGATNet, PubMed record](https://pubmed.ncbi.nlm.nih.gov/41945846/)
- [MHPG, Expert Systems with Applications 2026](https://www.sciencedirect.com/science/article/pii/S0957417426008651)
- [BioLORD-2023 model card and license notice](https://huggingface.co/FremyCompany/BioLORD-2023)

The CSV-to-processed-row identity chain is taken from Experiment 11's verified `subject_row_mapping.csv`. It is checked against unique SUB_IDs and the existing 871x320 processed matrix. Diagnosis fields are used only in an after-the-fact alignment audit, never in text construction or relation conditioning. No site name is put into text. ANAT/FUNC values are described as imaging quality measurements; CORR summaries refer to selected feature slots, not verified anatomical regions.

`config.json` freezes the protocol and group allocation. `build_phenotype_text.py` performs the ID/field audit and fold-local text/numeric feature construction. `pcpr.py` implements the bounded existing-edge router. `integration.py` patches only the experiment process's in-memory PLN forward to the B0-A PEN-bypass path and implements the fair propagation backbone. `run.py` runs preflight, text encoding and caching, resumable fold workers, and aggregation. The new router's Q/K/beta optimizer group uses `weight_decay=0.0`; original GFR and PLN groups retain 0.11. This prevents the high-dimensional Q/K projection from being driven to zero by the historical L2 term before it can learn a condition score. A real fold-1 BioLORD cache is used in preflight, which asserts nonconstant edge scores and nonzero supported-edge modulation both at initialization and after an optimizer update.

BioLORD-2023 is pinned to revision `167aab527b238a50ca65224e6319215d2ff4fc9f`. Its model card requires users to ensure UMLS and SNOMED CT licensing. The encoder must not be downloaded or used until the project's licensing status is confirmed. `transformers` is used directly with the model-card mean-pooling recipe; no package is installed and no encoder fine-tuning occurs.

Expected commands after the licensing gate is satisfied:

```bash
python run.py audit --root /data/adodas/SPromptGL-main
python run.py encode --root /data/adodas/SPromptGL-main
python run.py preflight --root /data/adodas/SPromptGL-main
python run.py worker --root /data/adodas/SPromptGL-main --config B0_ORIGINAL_B0A --gpu 0
python run.py dispatch --root /data/adodas/SPromptGL-main
python run.py summarize --root /data/adodas/SPromptGL-main
```

All resumable checkpoints, RNG states, text embeddings, and worker status files stay below `.runtime/`. The aggregate tables and final report belong in `results/`. The inherited protocol intentionally retains `val_idx=test_idx`, pre-CV RFE/standardization, and transductive propagation over all 871 participants; it is not an unbiased external generalization estimate.
