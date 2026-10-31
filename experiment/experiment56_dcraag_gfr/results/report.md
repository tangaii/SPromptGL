# Experiment 56 — DCRAG-GFR

**Status:** independently reloaded and audited 10/10 E56 checkpoints. This is a single-seed historical-protocol exploration.

## Fold results

Metrics are percent mean ± population SD over ten folds. Hard-AUC is calculated from hard predictions; probability-AUC uses continuous P(ASD), with ASD (label 0) positive. BA and hard-AUC coincide in this binary setup.

|Fold|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|Best epoch|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|84.0909|84.3280|85.3140|80.8511|87.8049|84.3280|83.7209|287|
|2|85.0575|84.8091|91.1453|89.1304|80.4878|84.8091|83.5443|303|
|3|82.7586|82.2375|85.2068|91.3043|73.1707|82.2375|80.0000|296|
|4|86.2069|85.1862|92.5532|97.8723|72.5000|85.1862|82.8571|297|
|5|75.8621|76.3564|80.1596|70.2128|82.5000|76.3564|75.8621|380|
|6|93.1034|92.8723|96.1702|95.7447|90.0000|92.8723|92.3077|389|
|7|82.7586|82.1809|81.5957|89.3617|75.0000|82.1809|80.0000|307|
|8|85.0575|84.8670|88.7766|87.2340|82.5000|84.8670|83.5443|119|
|9|77.0115|76.8617|79.5213|78.7234|75.0000|76.8617|75.0000|194|
|10|87.3563|87.3670|90.3191|87.2340|87.5000|87.3670|86.4198|96|

## E56 vs historical E51 E1

Values below are mean ± population SD (%) and paired same-fold mean difference in percentage points. Win/tie/loss counts describe the ten overlapping CV folds and are not significance tests.

|Metric|E56|E51 E1|Paired Δ pp|W/T/L|
|---|---:|---:|---:|---:|
|acc|83.9263 ± 4.6834|91.9592 ± 2.8684|-8.0329|0/0/10|
|hard_auc|83.7066 ± 4.5597|91.7434 ± 2.8950|-8.0368|0/0/10|
|prob_auc|87.0762 ± 5.3312|93.1856 ± 3.0236|-6.1095|1/0/9|
|spe|86.7669 ± 7.8167|94.6392 ± 3.3856|-7.8723|1/1/8|
|sen|80.6463 ± 6.1532|88.8476 ± 4.1651|-8.2012|0/0/10|
|ba|83.7066 ± 4.5597|91.7434 ± 2.8950|-8.0368|0/0/10|
|asd_f1|82.3256 ± 4.7735|91.0928 ± 3.1408|-8.7672|0/0/10|

## Pooled OOF

Across all 871 held-out predictions, pooled ACC=83.9265%, hard-AUC=83.6986%, Probability-AUC=85.4123%, SPE=86.7521%, SEN=80.6452%, BA=83.6986%. Test-fold labels participated in historical checkpoint selection (`val=test`), so these pooled values are not an unbiased independent OOF estimate.

## Mechanism findings

|Modality|Train patient-disjoint relation OOF AUC|Posthoc held-out relation AUC|Edge reweight L1|Same-message/input|AT-message/input|
|---|---:|---:|---:|---:|---:|
|PHENO|0.4972|0.5023|0.08383|4.868029|0.070865|
|ANAT|0.5266|0.5049|0.09834|4.385236|0.090924|
|FUNC|0.5268|0.5209|0.03775|4.925739|0.051748|
|CORR|0.7268|0.7244|0.20859|5.293269|0.048181|

Q1: Patient-disjoint relation OOF is training-only evidence. Held-out relation heads are posthoc diagnostics and never feed back into training.

Q2: Conditional adversary gradients were nonzero in 10/10 folds, but the held-out source probe changed only from macro-AUC 0.9997 before training to 0.9989 after training; the trained conditional discriminator's held-out macro-AUC averaged 0.8598. The source remains readily identifiable, so conditional cross-modal alignment is not demonstrated.

Q3: Each graph's edge counts, same/AT weights, reweighting distance, effective neighbors, isolated nodes, connected components and parameter changes are in `mechanism.json`.

Q4: The new path is numerically active, but the mean same-message/input ratios range from 4.39 to 5.29; mean total residual/input ratio is 13.39. This is excessive amplification for the frozen bounded propagation design and is a mechanism failure, despite nonzero gradients and prediction effects.

Q5: The fixed checkpoint loses 13.09 pp ACC when candidate neighbors are replaced and 34.21 pp when messages are disabled. These interventions show dependence of the selected checkpoint on its trained graph path, not an independent retraining gain; the graph identity gate remains partial.
The wrong-neighbor control changed 100.00% of directed edges, preserved receiver outdegree for 100.00% of nodes, and preserved train/test partition for 100.00% of edges. Only 88.96% of replacement edges retained same-site status; sender indegree exact-match averaged 3.36% (mean indegree total variation 0.3405). This is not a fully site- and degree-matched graph control.

|Fixed-checkpoint intervention|Mean ΔACC pp|Mean prediction flip rate|Mean absolute ΔP(ASD)|
|---|---:|---:|---:|
|disable_ANAT_message|-12.5287|0.1782|0.159278|
|disable_CORR_message|-8.6207|0.1230|0.121172|
|disable_FUNC_message|-2.9885|0.0460|0.040887|
|disable_PHENO_message|-7.3211|0.0846|0.084346|
|disable_cross_patient_messages|-34.2137|0.4892|0.413139|
|neutral_edge_weights|-4.3652|0.1125|0.111857|
|original_e1_gfr_bypass|-32.1486|0.4523|0.411094|
|wrong_disease_condition|-0.4598|0.0046|0.003665|
|wrong_patient_neighbors|-13.0943|0.2412|0.212869|

## Research gates

|Gate|Result|
|---|---|
|GO_RELATION_DISCRIMINATION|GO|
|GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT|NO-GO|
|GO_DISEASE_PRESERVATION|GO|
|GO_TRUE_PATIENT_RELATION|PARTIAL_FIXED_CHECKPOINT_ONLY|
|GO_CROSS_PATIENT_PROPAGATION|NO-GO_SCALE_DOMINATED|
|GO_CLASSIFICATION_GAIN|NO-GO|
|GO_MICCAI2027_MODULE|NO-GO|

## Decision

**DCRAG-GFR continuation decision: NO-GO**. The predeclared exploratory screen against E51 E1 is not met on mean ACC (+0.30 pp threshold) with nondecreasing BA and Probability-AUC. Mechanism-level fixed-checkpoint responses do not replace independent retraining controls.

History limits: the fold's test indices are also the validation indices for strict ACC checkpoint selection; preprocessing and RFE preceded outer CV; the graph is transductive over all 871 subjects; only seed 0 and one ABIDE A cohort were used. No multi-seed stability, statistical significance, independent-site generalization, or unbiased external test claim is made.

The primary data, split definitions, E51 source and historical checkpoints were read-only. Root protected SHA values are recorded in `manifest.json`.
