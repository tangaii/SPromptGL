# Experiment 83.6 Method-Fidelity Matrix

| Topic | Registered fact | Classification | E83.6 handling |
|---|---|---|---|
| Original G1/G2 base decay | GFR/CAM Adam uses `weight_decay=0.0036` | `ORIGINAL_BEHAVIOR` | Held fixed except registered C4/D7 contrast |
| Original G3/G4 base decay | GFR/CAM Adam uses `weight_decay=0.11`; DARG AdamW remains `lr=0.001, weight_decay=0.01`; DPSL AdamW remains `lr=0.0005, weight_decay=0.01` | `ORIGINAL_BEHAVIOR` | C4/D7 change GFR/CAM only to 0.0036 |
| GFR modality fusion | Historical loop's second modality overwrites the first modality's message contribution | `ORIGINAL_BEHAVIOR` | Kept in H1-H5 and every G1-G4-derived group; supervised modalities and actual fused messages are audited separately |
| Historical ALoss | GFR computes ALoss against its masked target adjacency, but historical final loss does not include ALoss | `ORIGINAL_BEHAVIOR` | Not added to the objective |
| RES location | Paper Eq. 6 RES is the local/global operation inside PEN, coupled to SPGS text weighting in the full PEN forward | `ORIGINAL_BEHAVIOR` | Full PEN preserved for H1; physical No-PEN removes both SPGS and RES |
| RES-only | Use the paper's exact local tokens, global average pooling, elementwise local/global softmax interaction and residual add, with no parameters | `E83_6_EXPERIMENTAL_CHANGE` | B2/B3/C2/C3/D4/D5/D6, with numerical forward/backward and parameter-count audit |
| Figure 6 medical prompts | Four NC, four LMCI/MCI-proxy and four AD concepts, in source order | `TADPOLE_ADAPTATION` | Preserve concept wording/order and add the matching `NC:`, `MCI:`, or `AD:` prefix only in registered prefix variants |
| MCI semantics | MMGPL Figure 6 names LMCI, not TADPOLE MCI | `TADPOLE_ADAPTATION` | Record the four concepts as MCI semantic proxies, not equivalent diagnostic criteria |
| Prompt encoder | Frozen official CLIP RN50 text encoder with 1024-dimensional vectors | `TADPOLE_ADAPTATION` | Store official weight hash, text vectors, class map, query map and their hashes |
| Pair-mean graph supervision | `T_ij=1[y_i=y_j]`; use only outer-train labels and the six continuous cosine matrices before thresholding; MSE averaged over train-pair cells and six modalities, coefficient 1.0 | `TADPOLE_ADAPTATION` | Add only to A2/A3/B1/B3/C1/C3/D2/D3/D5/D6/S0; thresholded adjacency and propagation remain unchanged |
| Graph supervision versus ALoss | The new pair-mean term is distinct from the historical ALoss and cannot reuse its test-node sentinel target | `TADPOLE_ADAPTATION` | Construct targets directly from training-label equality and index only train rows/columns |
| CAM external dropout | MICCAI paper Section 3.1 reports external dropout 0 for TADPOLE | `ORIGINAL_BEHAVIOR` | Existing external CAM dropout stays 0 |
| CAM Encoder attention dropout | E83.5 internal EncodeLayer attention defaults to 0.1 despite external dropout 0 | `ORIGINAL_BEHAVIOR` | Kept except E1, which changes only internal EncodeLayer attention dropout to 0 |
| H2 physical No-PEN | No PEN, GFR, DARG, DPSL, GraphLearn, GCN, message passing, prompt, or graph loss | `ORIGINAL_BEHAVIOR` | H2 replay is checked against E83.5. H5 remains a separate strict physical CAM-only baseline |
| Positive-DCR and DPSL | Keep signed positive/negative relations, Direct-Off semantics, NC/MCI/AD classification, all three class pairs, native detached probability weights, source spectrum and losses, 0.08 residual cap and 0.001 shared-gradient budget | `ORIGINAL_BEHAVIOR` | No binary Top-2 routing or test-label routing; only registered input/RES/WD variations |
| Warmup alignment | Identical seeds alone do not imply identical initialization, optimizer, pair sampling, RNG or warmup trajectory | `ORIGINAL_BEHAVIOR` | D1 maps to H3; D2/D3 to C1; D4 to C2; D5/D6 to C3; D7 to C4. Compare each fold's full warmup payload and hashes |
| Fold RNG | E83.5 and `sp` have different RNG trajectories and fold handoff rules | `TADPOLE_ADAPTATION` | Main groups retain E83.5 seed/fold behavior; S0 uses `sp`'s own continuous ten-fold RNG handoff |
| Feature order | Identical feature names do not imply identical column order | `TADPOLE_ADAPTATION` | Main groups use the locked E83.5 order; S0 prepares and verifies `sp`'s deterministic lexical order |
| Evaluation | Full-cohort RFE, imputation, StandardScaler before outer CV, transductive graph, `val_idx == test_idx`, first strict test-fold ACC improvement, 450 epochs and `wait > 200` are historical protocol | `ORIGINAL_BEHAVIOR` | Retained for H and all non-S0 new groups; S0 retains its 300-epoch `sp` protocol |
| Multiple comparisons | All variants use the same ten-fold CV and are compared after seeing those folds | `E83_6_EXPERIMENTAL_CHANGE` | Treat as protocol-bound method exploration; describe potential optimism from test-fold selection, preprocessing and multi-candidate comparisons |

## Source Scope

The paper comparison is SPromptGL, MICCAI 2025 Section 3.1, GFR, CAM, RES, SPGS/PEN, Eq. 5-7, Fig. 2 and Table 2. The author implementation snapshot is `ba3b9446fa451fde6864d6555bdf324de0c4b64a`. E83.5's sealed local project snapshot is `5ebc4f5a6602d3c2e5ea1ea0ea4c3d9009aa8a11`. The requested project reference `4649c5d` is recorded separately from the sealed E83.5 snapshot. The `sp` source snapshot is `f4534f8`.

The E83.5 repository upload excluded `.pt` files. H1-H5 replay therefore uses the server's existing E83.5 checkpoint files and records their hashes and absolute source paths. No GitHub checkpoint availability is claimed.
