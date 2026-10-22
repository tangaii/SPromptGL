# Experiment 34: SeRCA

SeRCA tests whether four pre-registered, anatomically anchored FC concepts can condition the existing GFR subject graphs and improve classification through a bounded difference-message residual. It runs the Experiment 32 B0-A protocol from scratch for S0, then independently trains the nine registered controls and S9, restoring the corresponding S0 pre-construction RNG state at each fold entry.

The only new operator is between the existing PLN `InputLayer` and `Encoder`. For each existing GFR relation `A_m`, it forms `B_mk = normalize_adj(diag(A_m) + offdiag(A_m) * exp(S_k - center_k) + I)` using the repository's `normalize_adj`; no new edge is added. It pools `B_mk - B_m` messages with a low-capacity relation/concept softmax and adds `0.10*tanh(alpha_t)` bounded by 10% of each input-token norm. S0 has no SeRCA forward intervention. Original GFR propagation, four graph losses, CAM, classifier, two Adam optimizers, early stopping and checkpoint selection are retained.

The four domains and their cautious source wording are frozen from Experiment 19. Only its 252 mapped FC features enter semantic concepts; FC indices 195, 225, 232 and 236 remain in the B0-A trunk but are excluded from concept mappings. `concepts.json` is generated from the immutable Experiment 19 prompts/mapping and the Experiment 11 verified identity table. It records the random feature permutation and source hashes. No CLIP text encoding is added.

The registered groups are S0_B0A, S1_NUMERIC_GLOBAL, S2_RANDOM_CONCEPTS, S3_SINGLE_CONCEPT, S4_CORR_REL_ONLY, S5_AUX_REL_ONLY, S6_UNIFORM_RELATION, S7_DIRECT_MESSAGE, S8_PATIENT_PERMUTED and S9_SERCA_FULL. The exact assignments and internal screening conditions are in `manifest.json` after `prepare`.

Run commands from the repository root:

```bash
python -B experiment/experiment34_serca_semantic_relational_alignment/run.py prepare
python -B experiment/experiment34_serca_semantic_relational_alignment/run.py preflight
python -B experiment/experiment34_serca_semantic_relational_alignment/run.py run
```

`run` first completes S0 and refuses to start the other groups unless its fold-entry RNG, initial trunk hashes, best epochs and four metrics reproduce Experiments 32/33 exactly. Workers checkpoint model, both Adam states, best model, early-stop counter and Python/NumPy/CPU/CUDA RNG under this directory's `.runtime`. Re-running `run` resumes incomplete work and skips completed groups. Final fold rows and mechanism diagnostics are written under `results/`; no Git commit or push is part of this experiment.

## Literature boundary

- SPromptGL, MICCAI 2025: https://papers.miccai.org/miccai-2025/paper/5129_paper.pdf
- Author implementation: https://github.com/wanxixi11/SPromptGL
- Current project implementation: https://github.com/tangaii/SPromptGL
- MMGPL, Medical Image Analysis 2024: https://doi.org/10.1016/j.media.2024.103225
- MGDR, MICCAI 2024: https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf
- KD-Brain, MICCAI 2026: https://papers.miccai.org/miccai-2026/paper/0073_paper.pdf
- InfoBridge, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/papers/Li_InfoBridge_Balanced_Multimodal_Integration_through_Conditional_Dependency_Modeling_ICCV_2025_paper.pdf
- To Align or Not to Align, AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/39248

SeRCA does not claim that semantic guidance, concept graphs, pathology priors, multimodal complementarity or representation alignment are new by themselves. Its testable distinction is conditioning already learned between-subject GFR relations on anatomically supported FC concepts and injecting only the resulting difference-message through the existing CAM input stream.
