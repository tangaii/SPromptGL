# Experiment 78 source and literature review

## Sources inspected

- Local SPromptGL MICCAI 2025 paper, 11 pages, including Fig. 1, Sections 2.1–2.2, Eqs. (1)–(7), experimental setup and ablations. Public source: [paper page](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [PDF](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf).
- Project repository [tangaii/SPromptGL](https://github.com/tangaii/SPromptGL), read-only, and upstream [wanxixi11/SPromptGL](https://github.com/wanxixi11/SPromptGL). The project checkout is the implementation source; upstream was used only to cross-check the publication's official code.
- Local BrainPrompt MICCAI 2025 paper, 11 pages; [official repository](https://github.com/AngusMonroe/BrainPrompt) and its official [ABIDE disease-label prompt file](https://raw.githubusercontent.com/AngusMonroe/BrainPrompt/main/data/prompts/label_prompts/abide_label.txt).
- HP²L public [ScienceDirect record](https://www.sciencedirect.com/science/article/pii/S1361841526001313), including the accessible abstract, highlights and method preview. The full-text request returned HTTP 403. This is not a full-text audit.
- Read-only server sources: repository `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, `ABIDE-simple-2-concat-weighted-cosine.sh`; E61/E69 restoration path; E74.4 `run.py`, `soft_positive_dcr.py`, config, manifest, audit, task/warmup/checkpoint/metric artifacts; E76.5 SCPR, E76.6 BCSPC, E77.3 DS-CBPM and E77.4 QCRL-PM implementations and reports; `/model/spromptgl/experiment/handoff.md`.

## Source-grounded method notes

The SPromptGL paper first constructs modality-relation graphs and performs disease-aware graph propagation, then uses CAM for cross-modal feature interaction. Its original semantic path compares local patient features with disease-attribute text embeddings and uses the resulting prompt-conditioned weights to select/reweight local semantic evidence before forming the classification representation. The paper's Eq. (1)–(7) describe this graph, propagation, prompt, CAM and joint classification/graph-loss path. The public root `PLN` contains a PEN path, while the E74.4 implementation selects `PLNNoPEN`; E78 inherits the latter and does not silently restore PEN.

The project root `network.py` defines eight independent SPromptGL attribute strings in their original order: four normal/typical-control descriptions followed by four ASD descriptions. These are retained as eight candidates, mapped to model labels `[1,1,1,1,0,0,0,0]`. The official BrainPrompt ABIDE text order is TC then ASD, so its two prompts map explicitly to `[1,0]`. BrainPrompt's hierarchy includes ROI-, subject- and disease-level prompts, but E78 transfers only the two disease-level strings. The BrainPrompt comparison uses the same frozen RN50 text encoder as other E78 groups and is not a reproduction of the full BrainPrompt model or its original text encoder.

The public SPromptGL training entry uses stratified ten-fold evaluation, an early-stop counter, and a fusion/classification path; the E74.4 protocol is the governing protocol here because it adds the actual DARG/Positive-DCR implementation and saved initialization state. In `PLNNoPEN`, the four 36-dimensional CAM slots are flattened to `[871,144]` and passed as the first input to `OutputLayer`; the historical runner's returned `hidden` is the 36-dimensional output after `OutputLayer.mlp_head`. E78 therefore attaches a temporary pre-hook to `OutputLayer` and captures its input as the differentiable CAM144 representation without changing historical files or operations. E74.4 returns normalized log probabilities. The loss is weighted NLL plus graph, pair, adversarial and preserve terms, and the original optimizer step order is `opt_base`, `opt_cam`, `opt_added`. Its restored CAM class is `PLNNoPEN`.

## E74.4 identity and protected boundary

The only base is `/data/adodas/SPromptGL-main/experiment/experiment74_4_soft_positive_dcr`. Core source hashes recorded before E78 implementation:

| File | SHA256 |
|---|---|
| `run.py` | `cbb226657480b5cd28f9c8e13391da70d3f8cbed23a27f8befbcfb0cd47595cd` |
| `soft_positive_dcr.py` | `3374598e5ab5547a853c4298000710bb7e8f11f93ec3bd63ddb8ff330aa36718` |
| `config.json` | `a369c644e0dabc55f8dd40056fedde5726f749e33eba931ebf59da7d55a2761a` |
| `manifest.json` | `4bdfb4fbf191d830c9484c9fdc5d584079895ba9157add248f3f354ea2b48840` |
| `audit.py` | `cba6be99dd4eb418fed230d1b545199ad95ae674b520883680fe7b9d1ab8cafb` |

The E78 output mount under `/data/.../experiment` failed a real write with `EIO`/errno 5 before formal output was created. The single active output is `/model/spromptgl/experiment/experiment78_dcpss`; E74.4 remains read-only at its migrated `/data` path. Its source and saved per-fold initialization artifacts are checked through the E77.4 read-only relocation adapter; no historical path symlink or source edit is used.

## E76–E77 findings relevant to E78

- E76.5 SCPR trained a separate semantic/prototype correction branch. Its full-text condition had ACC 91.8443%, below E74.4's 92.1891%; the anonymous comparator had the same ACC. The report found a negative net correction versus base and no demonstrated semantic advantage.
- E76.6 BCSPC preserved the base head and used a protected correction branch. Full and no-boundary groups reproduced the same hard predictions and metrics as E74.4; mean correction was small and fixed-checkpoint interventions changed no hard prediction. This supports checking whether a semantic branch actually changes predictions and adds value, rather than relying on nonzero gates.
- E77.3 DS-CBPM used CAM/raw-CORR prototypes and FP/FN pairs. Its E77.3-G2 ACC was 91.8443%, below E74.4. Its audit identified an endpoint self-comparison shortcut in pair ranking and strong shared-gradient co-adaptation.
- E77.4 QCRL-PM explicitly supervised an independent third query and protected the base training path. Its full medical-text G2 reached ACC 92.0742%, still below E74.4; its anonymous-text group was higher. The report found the eight-concept attention close to uniform. Therefore E78 tests current differentiable CAM prototypes and patient-conditioned prompt selection jointly, with K and text-source controls, rather than adding another FP/FN memory or post-hoc retrieval branch.

These historical results motivate E78 but do not establish that DCPSS will improve diagnosis. The principal E78 evidence must come from the prespecified five independently trained groups, matched fold comparisons, fixed-checkpoint interventions and gradient/checkpoint audits.

## Evidence boundary

The historical protocol uses test-fold ACC for checkpoint selection, preprocessing/RFE before outer cross-validation, and transductive full-graph propagation. Fold 2 has historical train/test overlap at 1-based rows 377 and 407. E78 preserves those base-protocol behaviors for comparability, while excluding explicit test IDs from all new prototype support and auxiliary-label supervision. Its result is not an unbiased external-generalization estimate. HP²L is cited only as a limited public-preview review; no claims here depend on inaccessible full-text details.
