# Experiment 56.1 — DCRAG-GFR-R

**Status:** independently reloaded and audited 10/10 E56.1 checkpoints. This is the single formal group `E56_1_FULL_DCRAG_GFR_REPAIRED`, a single-seed historical-protocol exploration.

## Fold results

Metrics are percent mean ± population SD over ten folds. Hard-AUC is calculated from hard predictions; probability-AUC uses continuous P(ASD), with ASD (label 0) positive. BA and hard-AUC coincide in this binary setup.

|Fold|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|Best epoch|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|94.3182|94.0581|95.3814|97.8723|90.2439|94.0581|93.6709|99|
|2|87.3563|87.2481|88.8653|89.1304|85.3659|87.2481|86.4198|89|
|3|85.0575|84.6766|88.1760|91.3043|78.0488|84.6766|83.1169|95|
|4|93.1034|93.0585|95.4255|93.6170|92.5000|93.0585|92.5000|151|
|5|89.6552|89.4947|94.7340|91.4894|87.5000|89.4947|88.6076|126|
|6|94.2529|94.1223|94.8936|95.7447|92.5000|94.1223|93.6709|92|
|7|88.5057|87.6862|92.4468|97.8723|77.5000|87.6862|86.1111|86|
|8|83.9080|83.4309|87.5532|89.3617|77.5000|83.4309|81.5789|112|
|9|89.6552|89.6809|93.0319|89.3617|90.0000|89.6809|88.8889|95|
|10|90.8046|90.7447|91.2766|91.4894|90.0000|90.7447|90.0000|104|

## E56.1 vs historical E51 E1

Values below are mean ± population SD (%) and paired same-fold mean difference in percentage points. Win/tie/loss counts describe the ten overlapping CV folds and are not significance tests.

|Metric|E56|E51 E1|Paired Δ pp|W/T/L|
|---|---:|---:|---:|---:|
|acc|89.6617 ± 3.4186|91.9592 ± 2.8684|-2.2975|0/0/10|
|hard_auc|89.4201 ± 3.5325|91.7434 ± 2.8950|-2.3233|0/0/10|
|prob_auc|92.1784 ± 2.9105|93.1856 ± 3.0236|-1.0072|3/0/7|
|spe|92.7243 ± 3.2175|94.6392 ± 3.3856|-1.9149|1/2/7|
|sen|86.1159 ± 5.8715|88.8476 ± 4.1651|-2.7317|2/2/6|
|ba|89.4201 ± 3.5325|91.7434 ± 2.8950|-2.3233|0/0/10|
|asd_f1|88.4565 ± 3.9848|91.0928 ± 3.1408|-2.6363|0/0/10|

## Pooled OOF

Across all 871 held-out predictions, pooled ACC=89.6670%, hard-AUC=89.4196%, Probability-AUC=91.7271%, SPE=92.7350%, SEN=86.1042%, BA=89.4196%. Test-fold labels participated in historical checkpoint selection (`val=test`), so these pooled values are not an unbiased independent OOF estimate.

## Mechanism findings

|Modality|Train patient-disjoint relation OOF AUC|Posthoc held-out relation AUC|Edge reweight L1|Same-message/input|AT-message/input|
|---|---:|---:|---:|---:|---:|
|PHENO|0.4972|0.5023|0.74486|1.000000|0.034134|
|ANAT|0.5266|0.5269|0.20930|1.000000|0.034125|
|FUNC|0.5268|0.5283|0.48298|1.000000|0.033966|
|CORR|0.7268|0.7435|0.50622|1.000000|0.033069|

Q1: Patient-disjoint relation OOF is training-only evidence. Held-out relation heads are posthoc diagnostics and never feed back into training.

Q2: Conditional adversary gradients were nonzero in 10/10 folds, but the held-out source probe changed only from macro-AUC 0.9998 before training to 1.0000 after training; the trained conditional discriminator's held-out macro-AUC averaged 0.6604. The source remains readily identifiable, so conditional cross-modal alignment is not demonstrated.

Q3: Each graph's edge counts, same/AT weights, reweighting distance, effective neighbors, isolated nodes, connected components and parameter changes are in `mechanism.json`.

Q4: The repaired path is numerically active. Same-message/input ratios are intentionally normalized to approximately 1.00; the final residual/input ratio is 0.234, below the fixed rho=0.25 bound, with bound activation and nonzero GraphLearn/GCN/edge gradients recorded in `mechanism.json`. The scale repair therefore passes its engineering criterion. The trained gate standard deviation is 0.000000, so patient-level adaptive fusion is not demonstrated.

Q5: The fixed checkpoint loses 0.80 pp ACC when candidate neighbors are replaced and 7.24 pp when messages are disabled. These interventions show dependence of the selected checkpoint on its trained graph path, not an independent retraining gain; the graph identity gate remains partial.
The strict two-edge-swap control changed 14.12% of directed edges, preserved receiver outdegree for 100.00% of nodes, sender indegree for 100.00% of nodes, destination site attributes for 100.00% of edges, and destination train/test partition for 100.00% of edges. Mean indegree total variation is 0.0000; only 14.12% of edges were exchangeable under all constraints, so the full control is `INSUFFICIENT_MATCHED_CONTROL`.

|Fixed-checkpoint intervention|Mean ΔACC pp|Mean prediction flip rate|Mean absolute ΔP(ASD)|
|---|---:|---:|---:|
|disable_ANAT_message|-1.2657|0.0172|0.014322|
|disable_CORR_message|-3.9080|0.0529|0.052424|
|disable_FUNC_message|-0.6910|0.0138|0.014380|
|disable_PHENO_message|-1.0345|0.0126|0.015048|
|disable_cross_patient_messages|-7.2401|0.0954|0.090451|
|neutral_edge_weights|-2.6398|0.0563|0.054631|
|original_e1_gfr_bypass|-37.2087|0.4732|0.432973|
|wrong_disease_condition|-0.6897|0.0069|0.006250|
|wrong_patient_neighbors|-0.8033|0.0080|0.011261|

## Research gates

|Gate|Result|
|---|---|
|GO_RELATION_DISCRIMINATION|GO|
|GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT|NO-GO|
|GO_DISEASE_PRESERVATION|GO|
|GO_TRUE_PATIENT_RELATION|INSUFFICIENT_MATCHED_CONTROL|
|GO_CROSS_PATIENT_PROPAGATION|PARTIAL_CHECKPOINT_DEPENDENCE|
|GO_CLASSIFICATION_GAIN|NO-GO|
|GO_MICCAI2027_MODULE|NO-GO|

## Decision

**DCRAG-GFR-R continuation decision: NO-GO**. The predeclared exploratory screen against E51 E1 is not met on mean ACC (+0.30 pp threshold) with nondecreasing BA and Probability-AUC. Mechanism-level fixed-checkpoint responses do not replace independent retraining controls.

History limits: the fold's test indices are also the validation indices for strict ACC checkpoint selection; preprocessing and RFE preceded outer CV; the graph is transductive over all 871 subjects; only seed 0 and one ABIDE A cohort were used. No multi-seed stability, statistical significance, independent-site generalization, or unbiased external test claim is made.

The primary data, split definitions, E51 source and historical checkpoints were read-only. Root protected SHA values are recorded in `manifest.json`.
