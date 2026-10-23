# Experiment 36 — PCPR

Status: **COMPLETE**; 220/220 folds. GPU0=110, GPU1=110. End-to-end dispatch wall time=1114.8s (18m34.8s).

## Audit and Method

The 871 processed rows are joined to the 1,112-row ABIDE phenotype CSV through Experiment 11's unique SUB_ID mapping, not by source row order or SUB_IN_SMP. The mapping has 871 unique verified IDs. PHENO/QC values are independently reconstructed from age bins, site/sex one-hot encodings and full-cohort QC standardization; diagnosis agrees only in a post-join audit. No identifier, label, site name, inclusion flag or diagnosis scale enters any generated text.

The stable clinical template includes age, recorded sex, available FIQ/VIQ/PIQ scores and handedness without diagnosis language or inferred symptoms. ANAT/FUNC are represented only as outer-training-standardized image-quality/motion values. CORR text reports signed summaries over sixteen consecutive 16-feature slots; those slots are not verified anatomical regions. All fold statistics use outer-train only. Missingness is audited by label and site, but no missingness indicator is supplied to the PCPR numeric condition.

Frozen encoder: `FremyCompany/BioLORD-2023@167aab527b238a50ca65224e6319215d2ff4fc9f`, attention-mask mean pooling, L2-normalized 768D vectors, no fine-tuning. Encoder manifest: `.runtime/text_encoder_manifest.json`. The model card requires appropriate UMLS and SNOMED CT licenses; the run manifest records the user-confirmed gate.

For receiver i and sender j, the edge score is `e(i<-j)=q(z_i)^T k(z_j)/sqrt(64)`. Row-centered scores are tanh bounded; a learned `beta=0.25*sigmoid(beta_raw)` starts at 0.10. Existing off-diagonal raw GraphLearn support is multiplied by `exp(beta*tanh(centered_score))`; absent edges remain zero and self-loop values are untouched. The added branch uses row-stochastic propagation. In symmetric arms the pre-normalized conditioned edge matrix is symmetrized; in directed arms it remains receiver-row oriented. `P[i,j]` multiplies sender j's full 320D feature in receiver i's row.

B0 uses original source GFR, `normalize_adj(A+I)`, the original CAM and OutputLayer, with PEN forward bypassed exactly as Experiment 32 B0-A. B1 propagates four graphs through the same two-layer GCN and then fuses messages. B2 fuses four raw graphs before one propagation path. B3 propagates only CORR while retaining all 320 input features. All new structures preserve four raw GraphLearn matrices and the mean of four original GraphConstructLoss terms; ALoss is not added. Two Adam optimizers and historical `simple-2` loss are retained.

PCPR Q/K/beta parameters use a separate group in the original GFR Adam with weight_decay=0.0; all original GFR/PLN parameter groups keep weight_decay=0.11. An initial implementation pass applied 0.11 to the new high-dimensional Q/K projection and collapsed its weights, leaving edge modulation effectively zero. That pass was archived under `.runtime/superseded_l2_conditioner_pass/` and is excluded from all final metrics. The corrected implementation passed real-cache score-variation and post-update edge-response checks before the official rerun.

## Formal Metrics

Mean ± population SD over ten folds, percent. AUC is original hard-label AUC (binary equivalent to balanced accuracy), not probability ROC-AUC.

|Group|ACC|Hard-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|
|B0_ORIGINAL_B0A|92.0742 ± 2.6000|91.8546 ± 2.5815|89.0793 ± 3.7283|94.6300 ± 4.0457|
|B1_FOUR_GRAPH_BASE|91.2696 ± 3.2251|91.0819 ± 3.2037|88.5976 ± 4.1562|93.5661 ± 4.4333|
|B2_ONE_GRAPH_BASE|91.7294 ± 2.7656|91.5564 ± 2.7835|89.3293 ± 4.1741|93.7835 ± 3.6749|
|B3_CORR_ONLY_BASE|91.2696 ± 3.1839|91.0867 ± 3.1174|88.8293 ± 3.9209|93.3441 ± 5.1626|
|N0_FOUR_SYM_NUMERIC|91.8430 ± 3.2344|91.7242 ± 3.1579|90.3171 ± 3.6154|93.1314 ± 5.1984|
|N1_FOUR_DIR_NUMERIC|91.6144 ± 2.8249|91.3763 ± 2.7914|88.3354 ± 4.1849|94.4172 ± 4.6618|
|N2_ONE_SYM_NUMERIC|91.0397 ± 2.9517|90.8584 ± 2.8945|88.5854 ± 3.8934|93.1314 ± 4.9302|
|N3_ONE_DIR_NUMERIC|91.2696 ± 3.0134|91.0416 ± 3.0050|88.0915 ± 4.1374|93.9917 ± 4.1984|
|T0_FOUR_SYM_TEXT|92.0742 ± 2.4963|91.9423 ± 2.4248|90.3232 ± 2.8246|93.5615 ± 4.2383|
|T1_FOUR_DIR_TEXT|91.3845 ± 3.3079|91.2627 ± 3.2920|89.5976 ± 4.4887|92.9278 ± 4.5321|
|T2_ONE_SYM_TEXT|91.1560 ± 3.0064|90.9864 ± 2.9491|88.8415 ± 3.8785|93.1314 ± 4.9302|
|T3_ONE_DIR_TEXT|91.1546 ± 2.9178|90.8816 ± 2.9273|87.3415 ± 3.7838|94.4218 ± 3.5259|
|S0_FOUR_SYM_SHUFFLED|91.7294 ± 2.9950|91.6579 ± 3.0375|90.5915 ± 4.3635|92.7243 ± 3.2175|
|S1_FOUR_DIR_SHUFFLED|91.7294 ± 2.7656|91.5036 ± 2.7805|88.5854 ± 4.4898|94.4218 ± 4.1230|
|S2_ONE_SYM_SHUFFLED|91.3845 ± 3.0154|91.1508 ± 3.0410|87.8659 ± 4.5555|94.4357 ± 3.4770|
|S3_ONE_DIR_SHUFFLED|91.8443 ± 3.1095|91.6214 ± 3.1100|88.6037 ± 4.2816|94.6392 ± 3.8838|
|C0_FOUR_SYM_CONSTANT|91.4995 ± 3.3057|91.2814 ± 3.3047|88.3537 ± 4.1166|94.2091 ± 3.8772|
|C1_ONE_SYM_CONSTANT|91.8443 ± 2.9793|91.6898 ± 3.0017|89.5915 ± 4.2536|93.7882 ± 3.6583|
|H0_FOUR_SYM_SHARED_CLINICAL|91.4995 ± 2.9239|91.3444 ± 2.8707|89.3354 ± 3.3306|93.3534 ± 4.3743|
|H1_ONE_SYM_CLINICAL_ONLY|91.3845 ± 3.2677|91.1587 ± 3.2904|88.1037 ± 4.3801|94.2137 ± 3.6134|
|M0_FOUR_SYM_MODAL_ONLY|91.6144 ± 3.1353|91.3528 ± 3.1235|87.8537 ± 3.7148|94.8520 ± 3.6394|
|M1_ONE_SYM_MODAL_ONLY|91.4995 ± 3.2248|91.3367 ± 3.1306|89.3293 ± 3.6978|93.3441 ± 5.5239|

## Paired Comparisons

Δ is left minus right in percentage points. W/T/L compare same-fold ACC and Hard-AUC. Folds overlap in their training data and are descriptive, not independent cohorts.

|Left − right|ΔACC|ACC W/T/L|ΔHard-AUC|Hard-AUC W/T/L|ΔSEN|ΔSPE|
|---|---:|---:|---:|---:|---:|---:|
|T0_FOUR_SYM_TEXT − B0_ORIGINAL_B0A|+0.0000|1/8/1|+0.0877|3/5/2|+1.2439|-1.0685|
|T0_FOUR_SYM_TEXT − B1_FOUR_GRAPH_BASE|+0.8046|7/2/1|+0.8605|8/1/1|+1.7256|-0.0046|
|T0_FOUR_SYM_TEXT − N0_FOUR_SYM_NUMERIC|+0.2312|3/5/2|+0.2181|4/4/2|+0.0061|+0.4302|
|T0_FOUR_SYM_TEXT − S0_FOUR_SYM_SHUFFLED|+0.3448|4/5/1|+0.2844|5/2/3|-0.2683|+0.8372|
|T0_FOUR_SYM_TEXT − C0_FOUR_SYM_CONSTANT|+0.5747|5/4/1|+0.6610|6/3/1|+1.9695|-0.6475|
|T0_FOUR_SYM_TEXT − H0_FOUR_SYM_SHARED_CLINICAL|+0.5747|6/3/1|+0.5980|8/1/1|+0.9878|+0.2081|
|T0_FOUR_SYM_TEXT − M0_FOUR_SYM_MODAL_ONLY|+0.4598|5/4/1|+0.5895|7/2/1|+2.4695|-1.2905|
|T1_FOUR_DIR_TEXT − B0_ORIGINAL_B0A|-0.6897|2/2/6|-0.5919|3/1/6|+0.5183|-1.7021|
|T1_FOUR_DIR_TEXT − B1_FOUR_GRAPH_BASE|+0.1149|2/7/1|+0.1809|5/3/2|+1.0000|-0.6383|
|T1_FOUR_DIR_TEXT − N1_FOUR_DIR_NUMERIC|-0.2299|3/3/4|-0.1136|4/2/4|+1.2622|-1.4894|
|T1_FOUR_DIR_TEXT − S1_FOUR_DIR_SHUFFLED|-0.3448|2/4/4|-0.2409|3/3/4|+1.0122|-1.4940|
|T2_ONE_SYM_TEXT − B0_ORIGINAL_B0A|-0.9182|1/2/7|-0.8682|1/2/7|-0.2378|-1.4986|
|T2_ONE_SYM_TEXT − B2_ONE_GRAPH_BASE|-0.5734|0/6/4|-0.5700|1/5/4|-0.4878|-0.6522|
|T2_ONE_SYM_TEXT − B3_CORR_ONLY_BASE|-0.1136|2/4/4|-0.1003|3/2/5|+0.0122|-0.2128|
|T2_ONE_SYM_TEXT − N2_ONE_SYM_NUMERIC|+0.1162|3/5/2|+0.1280|3/5/2|+0.2561|+0.0000|
|T2_ONE_SYM_TEXT − S2_ONE_SYM_SHUFFLED|-0.2286|1/6/3|-0.1644|3/4/3|+0.9756|-1.3043|
|T2_ONE_SYM_TEXT − C1_ONE_SYM_CONSTANT|-0.6883|1/2/7|-0.7034|1/2/7|-0.7500|-0.6568|
|T2_ONE_SYM_TEXT − H1_ONE_SYM_CLINICAL_ONLY|-0.2286|2/4/4|-0.1723|3/3/4|+0.7378|-1.0823|
|T2_ONE_SYM_TEXT − M1_ONE_SYM_MODAL_ONLY|-0.3435|1/4/5|-0.3503|2/2/6|-0.4878|-0.2128|
|T3_ONE_DIR_TEXT − B0_ORIGINAL_B0A|-0.9195|1/4/5|-0.9730|3/2/5|-1.7378|-0.2081|
|T3_ONE_DIR_TEXT − B2_ONE_GRAPH_BASE|-0.5747|1/5/4|-0.6748|2/3/5|-1.9878|+0.6383|
|T3_ONE_DIR_TEXT − N3_ONE_DIR_NUMERIC|-0.1149|2/6/2|-0.1599|3/4/3|-0.7500|+0.4302|
|T3_ONE_DIR_TEXT − S3_ONE_DIR_SHUFFLED|-0.6897|2/3/5|-0.7398|3/2/5|-1.2622|-0.2174|
|T0_FOUR_SYM_TEXT − T2_ONE_SYM_TEXT|+0.9182|8/1/1|+0.9559|8/1/1|+1.4817|+0.4302|
|T1_FOUR_DIR_TEXT − T3_ONE_DIR_TEXT|+0.2299|3/3/4|+0.3811|4/2/4|+2.2561|-1.4940|
|T0_FOUR_SYM_TEXT − T1_FOUR_DIR_TEXT|+0.6897|6/2/2|+0.6796|6/2/2|+0.7256|+0.6337|
|T2_ONE_SYM_TEXT − T3_ONE_DIR_TEXT|+0.0013|2/4/4|+0.1048|3/2/5|+1.5000|-1.2905|
|B1_FOUR_GRAPH_BASE − B2_ONE_GRAPH_BASE|-0.4598|0/7/3|-0.4745|1/4/5|-0.7317|-0.2174|

## Mechanism and Decision

E0 history check: `MATCH`, metric checks `{'acc': True, 'hard_auc': True, 'sen_asd': True, 'spe_td': True}`, best epochs `True`.

For every fold, `diagnostics.csv` records raw edge counts/densities, conditioned edge-weight changes, symmetry/direction, row sums, self-loop propagation weight, within-site propagation share, edge-change association with age and FIQ distance, graph fusion weights, message norms, query/key/beta gradient maxima, text-off fixed-checkpoint flip fraction, and shuffle fixed points. This fixed-checkpoint text-off test is only mechanism dependence, not a trained ablation.

Across folds, T0's four modality graphs had mean absolute supported-edge changes of 0.0594, 0.0585, 0.0546 and 0.0507; its relation-score standard deviations were 2.01, 1.89, 1.81 and 1.35. T0's mean learned beta was 0.1227, and query/key/beta gradients were nonzero. For comparison, N0 edge changes were 0.0082, 0.0060, 0.0049 and 0.0111; shuffled S0 changes were 0.0296, 0.0289, 0.0233 and 0.0226. T0's source-fusion weights remained near 0.25 per modality, so the run does not show selective modality routing. Disabling the condition branch at the selected T0 checkpoints changed no fold's predicted class (0/10), despite measurable edge changes. Thus the condition path changed propagation weights, but a fixed-checkpoint prediction change and robust four-modality specialization were not demonstrated.

Decision: **NO-GO**. The pre-registered internal screen requires a text main arm to improve ACC and Hard-AUC over B0, matched structure, numeric conditioning, and shuffled patient text, while the condition path is used. A single seed and inherited `val=test`, pre-CV RFE/preprocessing, and all-node transduction limit the result to internal historical-protocol evidence.

## Reproducibility

`manifest.json` records source/data hashes, frozen configuration, task and GPU counts, RNG strategy, encoder revision/license gate, and delivery state. `fold_metrics.csv` contains 220 unique folds; `.runtime/` contains atomic task/epoch checkpoints, RNG snapshots and fold text embeddings. No root model, data or historical experiment was edited. No Git commit or push was performed.

The requested root `handoff.md` append returned a filesystem write error. A post-failure check confirmed its original 305210-byte size and SHA-256 were unchanged. The complete append is staged as `handoff_append.md`; the manifest records the error and staged file hash.
