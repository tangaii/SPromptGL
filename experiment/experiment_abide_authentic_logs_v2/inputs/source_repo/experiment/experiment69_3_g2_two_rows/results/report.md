# Experiment 69.3-G2 - Two selected rows added to fold 2 training

## Protocol

This diagnostic reruns E62-E2 from the exact, SHA-verified E61 epoch-10 shared warmup for each fold. It uses the unchanged E62-E2 gates (edge gates 0.5; adaptive gates PHENO/ANAT/FUNC=0.001 and CORR=0.99), model, optimizer settings, loss, split, seed, and training schedule. Rows are appended to the classification training indices and CADR relation-pair training population only when their own fold is held out: fold 2 gets rows 377, 407. They remain in their corresponding test sets. Folds 1, 3, 4, 5, 6, 7, 8, 9, 10 are unchanged.

The 1 affected fold(s) deliberately overlap training and test labels across 2 row(s). Their metrics include label-exposure/memorization effects and cannot support generalization claims. The ten-fold mean is consequently not a fair performance comparison with E62-E2 or 53.3. Historical `val=test`, pre-CV preprocessing, and the 871-node transductive graph are also retained.

Formal status: **COMPLETE_AUDITED**, 10/10 tasks and 10/10 independent checkpoint audits; device PPU-ZW810E; maximum workers 10. E61 commit `31ced6bdd279654154b90dc2ece056d8e435df2b` and E62-E2 source/config/results were SHA-checked. The fold-specific index/pair update used a local NumPy generator and left each warmup's main RNG state unchanged.

Training source SHA is recorded per task as `be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44`. The audit/report runner SHA is `be2d534ea2df90d3b462034ee4c9fc859b3f79507dfba7c62a2faa48aee49c44`; after training, only the gate-metadata audit check, audit-only entry point, and report/handoff generation changed. The training forward, optimizer updates, and RNG trace logic were not changed.

## Classification results

Values are 10-fold mean ± population SD, percent. Historical groups are read-only. E69.3-G2 aggregate values are contaminated by the listed overlap and are shown only as a diagnostic summary.

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|53.3 third-group read-only|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|
|E62-E2 historical|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|93.7789 ± 4.2653|89.5854 ± 3.4710|91.6821 ± 2.8919|91.0789 ± 3.0892|
|E69.3-G2|92.0742 ± 2.7482|91.8995 ± 2.7186|93.3759 ± 2.9908|94.2137 ± 3.9815|89.5854 ± 3.4710|91.8995 ± 2.7186|91.2984 ± 2.9222|

## Fold Membership and RNG

|Fold|Added train rows|Train/test overlap|Effective train n|E62 RNG common epochs matched|
|---:|---|---|---:|---:|
|1|none|none|783|414/414|
|2|[377, 407]|[377, 407]|786|352/352|
|3|none|none|784|368/368|
|4|none|none|784|368/368|
|5|none|none|784|408/408|
|6|none|none|784|297/297|
|7|none|none|784|338/338|
|8|none|none|784|410/410|
|9|none|none|784|355/355|
|10|none|none|784|440/440|

RNG SHA comparisons are against read-only E62-E2 task traces on common epochs. Matching RNG hashes demonstrate matching random-number consumption, not matching model updates; the 1 modified fold(s) have changed labels and relation pairs.

## Paired Comparisons

Differences are percentage points with descriptive paired-t 95% intervals across the original folds; they are exploratory, and E69.3-G2 comparisons are invalid as held-out performance estimates because the listed fold(s) contain train/test overlap.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E69.3-G2 − E62-E2-HISTORICAL|+0.230 (1/9/0; -0.290,+0.750)|+0.217 (1/9/0; -0.274,+0.709)|+0.175 (1/9/0; -0.221,+0.571)|+0.000 (0/10/0; +0.000,+0.000)|+0.435 (1/9/0; -0.549,+1.418)|
|E69.3-G2 − 53.3-HISTORICAL|+0.000 (1/8/1; -0.388,+0.388)|+0.034 (3/6/1; -0.332,+0.400)|-0.293 (4/0/6; -0.856,+0.271)|+0.494 (2/8/0; -0.251,+1.239)|-0.426 (1/6/3; -1.400,+0.549)|

## Label-Exposure Diagnostic

|Fold|Added rows|ΔACC pp|ΔHard-AUC pp|ΔProbability-AUC pp|ΔSEN pp|ΔSPE pp|
|---:|---:|---:|---:|---:|---:|---:|
|1|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|2|2|+2.299|+2.174|+1.750|+0.000|+4.348|
|3|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|4|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|5|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|6|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|7|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|8|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|9|0|+0.000|+0.000|+0.000|+0.000|+0.000|
|10|0|+0.000|+0.000|+0.000|+0.000|+0.000|

Across the 9 untouched folds (1, 3, 4, 5, 6, 7, 8, 9, 10), the maximum absolute difference from E62-E2 over all seven metrics is 0; every metric matches exactly. This confirms the unchanged-fold continuation and RNG path. Changes in folds 2 occur where selected test rows were also used for training and must be read as label-exposure/memorization effects.


## Fold Results

|Fold|ACC|Hard-AUC|Prob-AUC|SEN|SPE|Best epoch|Overlap n|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|95.455|95.122|96.731|90.244|100.000|223|0|
|2|90.805|90.642|92.365|87.805|93.478|179|2|
|3|87.356|87.381|88.653|87.805|86.957|177|0|
|4|94.253|93.936|94.734|90.000|97.872|177|0|
|5|91.954|92.181|95.532|95.000|89.362|217|0|
|6|96.552|96.436|97.021|95.000|97.872|106|0|
|7|91.954|91.436|95.957|85.000|97.872|147|0|
|8|88.506|88.245|88.511|85.000|91.489|219|0|
|9|90.805|90.559|91.223|87.500|93.617|164|0|
|10|93.103|93.059|93.032|92.500|93.617|334|0|

## Gate and Propagation

Gate values, edge counts, gradient magnitudes, original graph and CADR/direct message contributions are available in `gate_diagnostics.csv`. A nonzero message does not establish diagnostic benefit.

|Fold|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|PHENO|0.500|0.5091|0.001|0.0010|565|1865|29.52|0.00024|
|1|ANAT|0.500|0.4674|0.001|0.0009|957|906|49.73|0.00018|
|1|FUNC|0.500|0.4497|0.001|0.0009|1583|799|32.67|0.00023|
|1|CORR|0.500|0.5553|0.990|0.9920|1738|1613|1.82|0.40069|
|2|PHENO|0.500|0.4863|0.001|0.0010|1022|869|45.24|0.00025|
|2|ANAT|0.500|0.4752|0.001|0.0010|552|418|73.66|0.00012|
|2|FUNC|0.500|0.4710|0.001|0.0010|1250|350|55.24|0.00020|
|2|CORR|0.500|0.5432|0.990|0.9916|1696|1620|3.04|0.41020|
|3|PHENO|0.500|0.4683|0.001|0.0009|1571|626|36.23|0.00029|
|3|ANAT|0.500|0.4851|0.001|0.0010|388|309|81.00|0.00012|
|3|FUNC|0.500|0.4704|0.001|0.0009|1368|646|42.98|0.00022|
|3|CORR|0.500|0.5444|0.990|0.9916|1812|1640|2.71|0.41535|
|4|PHENO|0.500|0.4788|0.001|0.0010|1185|851|43.16|0.00026|
|4|ANAT|0.500|0.4912|0.001|0.0010|455|483|74.33|0.00011|
|4|FUNC|0.500|0.4633|0.001|0.0009|1342|339|52.58|0.00019|
|4|CORR|0.500|0.5441|0.990|0.9917|1745|1704|1.77|0.41168|
|5|PHENO|0.500|0.4911|0.001|0.0010|844|1751|24.01|0.00023|
|5|ANAT|0.500|0.4791|0.001|0.0010|564|335|75.57|0.00013|
|5|FUNC|0.500|0.4742|0.001|0.0010|802|1399|38.71|0.00017|
|5|CORR|0.500|0.5567|0.990|0.9923|1809|1685|1.41|0.40867|
|6|PHENO|0.500|0.5089|0.001|0.0010|240|1195|58.11|0.00015|
|6|ANAT|0.500|0.4986|0.001|0.0010|143|204|90.76|0.00005|
|6|FUNC|0.500|0.4857|0.001|0.0010|896|209|68.33|0.00017|
|6|CORR|0.500|0.5255|0.990|0.9909|1675|1631|6.90|0.39130|
|7|PHENO|0.500|0.4722|0.001|0.0009|2084|305|33.68|0.00033|
|7|ANAT|0.500|0.4988|0.001|0.0010|14|707|80.26|0.00004|
|7|FUNC|0.500|0.4795|0.001|0.0010|527|207|79.23|0.00011|
|7|CORR|0.500|0.5349|0.990|0.9912|1750|1590|2.88|0.42191|
|8|PHENO|0.500|0.5111|0.001|0.0011|737|1658|32.82|0.00026|
|8|ANAT|0.500|0.4945|0.001|0.0010|728|634|62.82|0.00016|
|8|FUNC|0.500|0.4676|0.001|0.0009|1042|1248|37.26|0.00020|
|8|CORR|0.500|0.5525|0.990|0.9920|1706|1571|1.80|0.40269|
|9|PHENO|0.500|0.4670|0.001|0.0009|2908|326|9.16|0.00035|
|9|ANAT|0.500|0.4882|0.001|0.0010|305|137|87.93|0.00009|
|9|FUNC|0.500|0.4835|0.001|0.0010|753|816|56.63|0.00016|
|9|CORR|0.500|0.5390|0.990|0.9914|1674|1457|1.63|0.41276|
|10|PHENO|0.500|0.4490|0.001|0.0008|1715|1040|22.26|0.00025|
|10|ANAT|0.500|0.4480|0.001|0.0009|1045|945|46.45|0.00017|
|10|FUNC|0.500|0.4340|0.001|0.0009|1269|1157|28.27|0.00019|
|10|CORR|0.500|0.5738|0.990|0.9929|1663|1752|1.59|0.37504|

### Fixed-checkpoint interventions

Each effect is full model minus intervention. Score permutation was fixed independently of test labels.

|Fold|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|
|---:|---|---:|---:|---:|
|1|all_cadr_off|+0.000|0|0.001767|
|1|corr_cadr_off|+0.000|0|0.001767|
|1|direct_off|+0.000|0|0.001767|
|1|score_permutation|+0.000|0|0.002264|
|2|all_cadr_off|+0.000|0|0.001561|
|2|corr_cadr_off|+0.000|0|0.001561|
|2|direct_off|+0.000|0|0.001561|
|2|score_permutation|+0.000|0|0.001856|
|3|all_cadr_off|+0.000|0|0.002868|
|3|corr_cadr_off|+0.000|0|0.002868|
|3|direct_off|+0.000|0|0.002868|
|3|score_permutation|+0.000|0|0.003208|
|4|all_cadr_off|+0.000|0|0.001554|
|4|corr_cadr_off|+0.000|0|0.001554|
|4|direct_off|+0.000|0|0.001554|
|4|score_permutation|+0.000|0|0.001778|
|5|all_cadr_off|+0.000|0|0.001816|
|5|corr_cadr_off|+0.000|0|0.001816|
|5|direct_off|+0.000|0|0.001816|
|5|score_permutation|+1.149|1|0.002182|
|6|all_cadr_off|+0.000|0|0.001141|
|6|corr_cadr_off|+0.000|0|0.001141|
|6|direct_off|+0.000|0|0.001140|
|6|score_permutation|+0.000|0|0.001499|
|7|all_cadr_off|+0.000|0|0.001111|
|7|corr_cadr_off|+0.000|0|0.001112|
|7|direct_off|+0.000|0|0.001113|
|7|score_permutation|+0.000|0|0.001732|
|8|all_cadr_off|+0.000|0|0.001647|
|8|corr_cadr_off|+0.000|0|0.001647|
|8|direct_off|+0.000|0|0.001647|
|8|score_permutation|+0.000|0|0.002629|
|9|all_cadr_off|+0.000|0|0.001184|
|9|corr_cadr_off|+0.000|0|0.001184|
|9|direct_off|+0.000|0|0.001184|
|9|score_permutation|+0.000|0|0.001760|
|10|all_cadr_off|+0.000|0|0.000872|
|10|corr_cadr_off|+0.000|0|0.000872|
|10|direct_off|+0.000|0|0.000868|
|10|score_permutation|+0.000|0|0.001845|

## Interpretation

E69.3-G2 minus E62-E2 aggregate ACC is +0.230 pp (1/9/0 folds won/tied/lost), but this is not a valid generalization comparison because 2 test rows in folds 2 were also supervised during training. The 9 unaffected folds reproduce E62-E2 exactly. This experiment tests the effect of explicit label exposure, not a deployable training protocol.

## Files

- Experiment root: `/model/spromptgl/experiment/experiment69_3_g2_two_rows`
- Task manifest: `manifest.json`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Gate diagnostics: `results/gate_diagnostics.csv`
- Independent checkpoint audit: `results/checkpoint_audit.csv`
