# Experiment 81: Prompt-guided Relational Composition Learning

## Registered question

Does a frozen disease-description Prompt improve the selection and composition of patient-specific relations among the sixteen latent local coordinates available in CAM144, beyond ordinary visual local relations and an anonymous text condition? The final diagnosis always comes from the historical E74.4 DARG CAM OutputLayer.

The contribution under test is disease-Prompt-conditioned patient-internal local-to-local interaction. SPromptGL already learns patient-level multi-relational graphs in GFR, propagates them with GCNs, and models CAM local-to-global interactions in PEN. E81 retains DARG's patient graph and the PEN-free CAM and adds a separate relation computation among CAM144 latent coordinates before the original classifier. The sixteen coordinates are a reshape of the learned CAM representation. They are not brain regions, ROIs, or independently identified pathophysiological concepts.

## Literature and source audit

- SPromptGL, MICCAI 2025, Sections 2.1 and 2.2: [paper](https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf), [official source](https://github.com/wanxixi11/SPromptGL), [SABER project source](https://github.com/tangaii/SPromptGL). GFR creates modality-specific patient relations and applies graph propagation. CAM uses InputLayer, Encoder, FeedForward and OutputLayer. PEN's SPGS and RES are equations (5) and (6), with frozen CLIP RN50 encodings for the original eight medical descriptions. E81 does not reactivate PEN.
- KD-Brain, MICCAI 2026: [paper](https://papers.miccai.org/miccai-2026/0366-Paper0073.html), [official model](https://github.com/LiuSy0731/KDBrain/blob/main/KDBrain/models/model.py). `Attention_AggregateLayer` projects text and uses a text gate and guided query before attention scoring. E81 takes the semantic-conditioning motivation but implements its own local relation operation; a CAM token is not a brain parcel.
- BrainHO, MICCAI 2026: [paper](https://papers.miccai.org/miccai-2026/0569-Paper2916.html), [model](https://github.com/Jingfeng-Tang/BrainHO/blob/main/model.py), [losses](https://github.com/Jingfeng-Tang/BrainHO/blob/main/losses.py), [sparsemax](https://github.com/Jingfeng-Tang/BrainHO/blob/main/sparsemax.py). E81 tests one local relation layer and a Null option without hierarchical Transformer stacks, Sparsemax, or orthogonality losses.
- ItemizedCLIP, CVPR 2026: [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Learning_complete_and_explainable_visual_representations_from_itemized_text_supervision_ItemizedCLIP_CVPR_2026_paper.html), [official source](https://github.com/MLNeurosurg/ItemizedCLIP). Its item-conditioned representations use item-level supervision and independence objectives. ABIDE supplies no per-patient, per-Prompt finding labels, so neither objective is transferred.
- KD-Brain and BrainHO are method references, not reused network definitions. The three-way E81 comparison isolates visual relations, frozen medical text, and frozen anonymous text. E78.9's audited LOO-FAPC comparison found medical and anonymous text classification nearly identical (ACC 91.9592% and 91.9579%); this motivates testing semantic necessity at relation-query time rather than treating a changed attention map as evidence.
- E20 Grounded Concept Graph Alignment and E32 Disease Contrastive Evidence Routing were checked as historical semantic-routing approaches. E75-E79 and E78.9 were checked for prior local semantic, prototype, Prompt, and correction routes. E81 adds no disease prototypes, retrieval memory, semantic logits, or auxiliary semantic classifier.

## Historical base and exact integration

Every group/fold calls the read-only E74.4 `prepare(fold)`, which follows `prepare -> E74.1 -> E69.3 -> E61.load_warmup_state`. It restores the matching E61 epoch-10 model, CAM, three optimizer states, Python/NumPy/PyTorch CPU/CUDA RNG, original patient split and 4096 relation pairs. Fold 2 preserves the historical train/test-overlap rows 377 and 407 in the original DARG classification path. E81 supplies no test labels to PRCL and excludes overlap rows from its residual regularizer.

One OutputLayer forward-pre-hook replaces only the first input `[871,144]` with `[871,144]` after PRCL and preserves the original attention argument. The historical DARG forward executes once. Its original weighted NLL, graph loss, pair BCE, adversarial loss, and preservation loss are unchanged. The added term is `0.001 * mean_train(||delta||^2/(||z_cam||^2+eps))`. The three original optimizers step in historical order, followed by the PRCL AdamW optimizer.

## Registered groups

- `E81-G1-Visual-PRCL`: same eight nonsemantic head identities, local relation keys and values, Null, group competition, and residual path; text injection is identically zero.
- `E81-G2-Medical-PRCL`: the eight original root `network.py::attr_words`, in original order and spelling, encoded once by frozen OpenAI CLIP RN50.
- `E81-G3-Anonymous-PRCL`: eight fixed non-neurobiological descriptions, in two four-text groups. Its only difference from G2 is frozen text values.

G2 and G3 share one joint SVD row space computed over the sixteen L2-normalized RN50 vectors, zero-padded to 32 coordinates. The common basis preserves their joint Gram matrix; their individual group geometry is not claimed to be identical. Each of the eight heads receives a distinct fixed orthogonal nonsemantic identity vector; G1 retains all eight identities while its text contribution is zero. Each fold uses a CPU-private forked initialization seed `810000 + fold`, the same for all groups. The public restored RNG is not consumed by PRCL construction. A within-group permutation moves the text and its head identity together, so it only relabels the four terms entering that group's mean.

## Registered PRCL computation

`z_cam [N,144] -> [N,4,4,9] -> [N,16,9] -> shared 9-to-32 projection + position -> LayerNorm`.
For the 120 unordered local pairs, the module computes one symmetric key from `[u_p*u_q, abs(u_p-u_q)]` and two direction-specific values from `[u_q, u_p*u_q, u_q-u_p]`. These are stored as 240 directed message positions. Eight patient-conditioned queries combine a 32-D visual projection, fixed group-shared nonsemantic head identity, and `gamma*sqrt(32)*t_k`; `gamma = 0.5*sigmoid(gamma_raw)` starts at 0.2.

Each patient, query head, and target coordinate has a separate softmax over its fifteen other coordinates and one shared learned Null key. Null's value is exactly zero. No self edge, cross-patient relation, random Top-K, stacked graph layer, or per-patient backbone pass is used. The first and last four query messages are averaged as the two original text groups and combined as `[m_normal,m_abnormal,m_abnormal-m_normal]`. This grouping does not assign either text group to TD or ASD.

The output maps to nine dimensions per coordinate, is `tanh` bounded, and is multiplied by a patient gate initialized to 0.05. A per-patient scale then enforces `||delta_i||/(||z_i||+eps) <= 0.10`; the classifier sees `z_cam + delta`. No second classifier exists.

## Training and audit protocol

Seed 0, ten historical folds, 450 maximum epochs, stop when wait exceeds 200, and save on strict test-fold ACC improvement with ties retaining the earlier checkpoint. Optimizers: inherited GFR Adam 0.0038/0.11, CAM Adam 0.0038/0.11, DARG AdamW 0.001/0.01, and new PRCL AdamW 0.0005/0.01. Thirty isolated jobs use at most ten workers on one 98-GB GPU.

Preflight includes 10/10 exact base-off forward/loss/gradient/update/RNG comparisons; local indexing, Null and patient-isolation checks; frozen Prompt and cross-text Query/attention checks; component-level classification-gradient checks; identical initial PRCL parameter checks; a train-only frozen CAM local-versus-pair diagnostic with dimension-matched random coordinate groups; and a two-step disk-recovery replay of the CAM, DARG, all four optimizers, RNG, loss, output, and attention. It also checks E61 warmup, optimizer, patient-pair and RNG identities, and Fold 2 membership. Formal source and configuration hashes freeze before any training task starts.

The independent audit reloads all 30 best and latest states, reproduces probabilities and six metrics, validates all four optimizer states, compares common-epoch RNG triples against E74.4-G2, and runs frozen interventions: PRCL off, uniform relation weights, text off, Null-only, writeback off, medical-to-anonymous replacement, zero gamma, group swap, nonsemantic query, and within-group prompt permutation. Interventions are checkpoint probes, not retrained ablations.

The pre-registered decision rule uses paired folds. Gate A requires positive mean ACC difference and more wins than losses. Gates B and C additionally require a positive W>L result for at least one of Hard-AUC or Prob-AUC, and SEN/SPE may not both regress in mean. Gate D applies the same rule against E74.4 and must retain positive W>L ACC and AUC evidence after excluding Fold 2. Gate E requires measurable PRCL relation/residual effects and a medical-to-anonymous intervention that changes Query, attention, and at least one prediction probability or hard prediction. Medical semantic value requires B, C, D, and E; GO_CANDIDATE additionally requires A. These paired rules are descriptive evidence, not independent-fold significance tests.

## Interpretation limits

The inherited protocol uses validation equal to test, test ACC for checkpoint choice, pre-CV preprocessing/RFE, and a transductive 871-node graph. Fold 2 additionally contains the two historical overlapping rows. Ten folds have overlapping training sets; paired intervals are descriptive and do not show independent clinical significance. Only independently trained G2-vs-G3 performance can support medical-content value; changed Queries or attention weights alone cannot.

