# Experiment 62 — E61-E2 Gate Initialization Sensitivity

## Protocol

Four groups vary only the sigmoid-space initialization of CADR edge gates and adaptive fusion gates. All groups start from each exact E61 shared epoch-10 checkpoint, including GFR/CAM/CADR weights, all three optimizer states, and Python/NumPy/Torch CPU/CUDA RNG. The E61-E2 forward, loss, data, pair cache, fold split, epoch budget, early stopping, and checkpoint selection are reused. No OOF or reliability multiplier is used.

E61 is pinned to commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. Its published source files and historical manifest were SHA-checked. Warmup files were individually checked against E61 manifest/task SHA, split and pair identity. The E61 training runner SHA recorded inside warmup is preserved separately from the published post-training audit runner SHA; model and config SHA match. Both gate parameters had empty optimizer-state entries after warmup, were already in the CADR AdamW parameter groups at WD=0.01, and their state was not reset.

History `val=test` checkpoint selection, pre-CV RFE/standardization and the transductive 871-node graph remain. Results are exploratory, not unbiased external generalization estimates.

The actual E61 epoch-10 relation scores were all between the 0.45/0.55 thresholds in fold 1, so the initial disease graphs were empty and edge-gate classification gradients were zero at that exact checkpoint. A separate deterministic controlled-edge test verified nonzero classification gradients through activated edge gates and relation scorers. Formal training determines whether the learned relation scores create active edges.

Formal status: **COMPLETE_AUDITED**, 40/40 tasks and 40/40 independent readbacks; device PPU-ZW810E, maximum workers 10.

## Classification results

Values are ten-fold mean ± population SD, percent. Historical groups are read-only.

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|53.3 third-group read-only|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|
|E61-E2 historical|91.4995 ± 2.4842|91.2480 ± 2.5179|93.8126 ± 2.6938|94.6485 ± 2.4105|87.8476 ± 3.5514|91.2480 ± 2.5179|90.5253 ± 2.7692|
|E62-E1|91.7294 ± 3.3292|91.4804 ± 3.3144|93.2417 ± 3.0625|94.6253 ± 4.8724|88.3354 ± 4.6178|91.4804 ± 3.3144|90.8361 ± 3.5693|
|E62-E2|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|93.7789 ± 4.2653|89.5854 ± 3.4710|91.6821 ± 2.8919|91.0789 ± 3.0892|
|E62-E3|91.3845 ± 2.8809|91.1929 ± 2.8559|93.5756 ± 2.6162|93.7882 ± 3.6583|88.5976 ± 3.3293|91.1929 ± 2.8559|90.5143 ± 3.1093|
|E62-E4|91.7294 ± 2.9054|91.5487 ± 2.8374|93.1600 ± 2.9441|93.7743 ± 4.8934|89.3232 ± 3.5531|91.5487 ± 2.8374|90.9474 ± 3.0379|

## Paired comparisons

Differences are percentage points with descriptive paired-t 95% intervals over ten overlapping CV folds; they are not confirmatory confidence claims.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E62-E1 − E61-E2-HISTORICAL|+0.230 (4/4/2; -0.704,+1.163)|+0.232 (5/1/4; -0.674,+1.139)|-0.571 (5/0/5; -1.746,+0.604)|+0.488 (5/2/3; -1.689,+2.664)|-0.023 (4/4/2; -2.547,+2.501)|
|E62-E2 − E61-E2-HISTORICAL|+0.345 (4/5/1; -0.210,+0.900)|+0.434 (7/1/2; -0.088,+0.956)|-0.612 (3/0/7; -1.303,+0.079)|+1.738 (6/3/1; +0.064,+3.411)|-0.870 (3/2/5; -2.819,+1.080)|
|E62-E3 − E61-E2-HISTORICAL|-0.115 (2/6/2; -0.835,+0.605)|-0.055 (3/4/3; -0.757,+0.647)|-0.237 (6/0/4; -0.938,+0.464)|+0.750 (3/6/1; -0.711,+2.211)|-0.860 (2/4/4; -2.507,+0.787)|
|E62-E4 − E61-E2-HISTORICAL|+0.230 (5/2/3; -0.526,+0.985)|+0.301 (5/1/4; -0.400,+1.002)|-0.653 (2/0/8; -1.340,+0.035)|+1.476 (5/3/2; -1.029,+3.980)|-0.874 (3/5/2; -3.718,+1.969)|
|E62-E1 − E62-E2|-0.115 (2/5/3; -0.722,+0.492)|-0.202 (2/4/4; -0.845,+0.441)|+0.041 (6/0/4; -1.131,+1.213)|-1.250 (0/7/3; -2.988,+0.488)|+0.846 (4/5/1; -0.443,+2.136)|
|E62-E1 − E62-E3|+0.345 (5/3/2; -0.332,+1.022)|+0.287 (6/2/2; -0.412,+0.987)|-0.334 (5/0/5; -1.513,+0.845)|-0.262 (4/4/2; -2.705,+2.181)|+0.837 (4/4/2; -1.355,+3.029)|
|E62-E1 − E62-E4|+0.000 (3/4/3; -0.671,+0.671)|-0.068 (3/3/4; -0.787,+0.651)|+0.082 (6/1/3; -0.834,+0.997)|-0.988 (2/5/3; -3.238,+1.262)|+0.851 (3/6/1; -0.792,+2.494)|
|E62-E2 − E62-E3|+0.460 (3/7/0; -0.115,+1.035)|+0.489 (5/4/1; -0.131,+1.109)|-0.375 (3/1/6; -1.137,+0.387)|+0.988 (5/4/1; -0.929,+2.905)|-0.009 (2/5/3; -1.452,+1.434)|
|E62-E2 − E62-E4|+0.115 (3/5/2; -0.492,+0.722)|+0.133 (4/3/3; -0.474,+0.741)|+0.041 (5/0/5; -0.733,+0.815)|+0.262 (3/6/1; -1.281,+1.805)|+0.005 (2/5/3; -1.450,+1.459)|
|E62-E3 − E62-E4|-0.345 (1/5/4; -0.900,+0.210)|-0.356 (2/3/5; -0.956,+0.244)|+0.416 (4/1/5; -0.328,+1.159)|-0.726 (2/4/4; -3.086,+1.635)|+0.014 (1/7/2; -1.911,+1.939)|

## Gate and propagation mechanisms

Initial and best-checkpoint gate values, edge counts, gradient magnitudes, individual GCN/direct messages, and normalized contribution are in `gate_diagnostics.csv`. A nonzero or larger message norm does not establish diagnostic benefit.

|Group|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E62-E1|PHENO|0.500|0.5240|0.010|0.0104|235|2600|22.54|0.00173|
|E62-E1|ANAT|0.500|0.4789|0.010|0.0099|549|247|78.07|0.00138|
|E62-E1|FUNC|0.500|0.4742|0.010|0.0098|1355|171|56.33|0.00194|
|E62-E1|CORR|0.500|0.5337|0.990|0.9912|1743|1594|3.16|0.42289|
|E62-E1|PHENO|0.500|0.4958|0.010|0.0100|521|1236|50.77|0.00213|
|E62-E1|ANAT|0.500|0.5029|0.010|0.0101|464|287|79.56|0.00110|
|E62-E1|FUNC|0.500|0.5097|0.010|0.0099|456|880|62.88|0.00142|
|E62-E1|CORR|0.500|0.5331|0.990|0.9912|1627|1434|3.80|0.41266|
|E62-E1|PHENO|0.500|0.4638|0.010|0.0095|1438|768|37.13|0.00271|
|E62-E1|ANAT|0.500|0.4749|0.010|0.0098|349|212|84.72|0.00108|
|E62-E1|FUNC|0.500|0.4593|0.010|0.0095|1296|639|45.39|0.00205|
|E62-E1|CORR|0.500|0.5417|0.990|0.9915|868|788|1.90|0.32054|
|E62-E1|PHENO|0.500|0.4044|0.010|0.0087|1430|1539|17.48|0.00247|
|E62-E1|ANAT|0.500|0.4361|0.010|0.0095|825|1068|48.26|0.00152|
|E62-E1|FUNC|0.500|0.4211|0.010|0.0087|1517|959|31.34|0.00268|
|E62-E1|CORR|0.500|0.5832|0.990|0.9931|1645|1626|1.33|0.37156|
|E62-E1|PHENO|0.500|0.4482|0.010|0.0092|1140|2068|8.58|0.00242|
|E62-E1|ANAT|0.500|0.4024|0.010|0.0093|832|1023|49.30|0.00126|
|E62-E1|FUNC|0.500|0.4051|0.010|0.0080|1122|1544|24.07|0.00164|
|E62-E1|CORR|0.500|0.5805|0.990|0.9934|1517|1618|1.20|0.35414|
|E62-E1|PHENO|0.500|0.4867|0.010|0.0097|1504|1415|18.62|0.00340|
|E62-E1|ANAT|0.500|0.4652|0.010|0.0099|803|968|51.72|0.00211|
|E62-E1|FUNC|0.500|0.4241|0.010|0.0087|1438|936|30.30|0.00217|
|E62-E1|CORR|0.500|0.5660|0.990|0.9927|1571|1649|1.98|0.37880|
|E62-E1|PHENO|0.500|0.4736|0.010|0.0093|1647|350|44.23|0.00297|
|E62-E1|ANAT|0.500|0.4930|0.010|0.0099|10|623|82.73|0.00040|
|E62-E1|FUNC|0.500|0.4724|0.010|0.0097|561|84|82.13|0.00113|
|E62-E1|CORR|0.500|0.5323|0.990|0.9911|1710|1598|3.92|0.41468|
|E62-E1|PHENO|0.500|0.5260|0.010|0.0110|237|1885|40.02|0.00180|
|E62-E1|ANAT|0.500|0.5139|0.010|0.0098|101|555|82.27|0.00047|
|E62-E1|FUNC|0.500|0.4706|0.010|0.0095|767|851|54.46|0.00165|
|E62-E1|CORR|0.500|0.5387|0.990|0.9914|1782|1625|1.87|0.41688|
|E62-E1|PHENO|0.500|0.4646|0.010|0.0088|2710|346|13.72|0.00337|
|E62-E1|ANAT|0.500|0.4741|0.010|0.0099|328|129|87.58|0.00090|
|E62-E1|FUNC|0.500|0.4819|0.010|0.0097|759|712|58.75|0.00153|
|E62-E1|CORR|0.500|0.5387|0.990|0.9914|1754|1641|2.10|0.41342|
|E62-E1|PHENO|0.500|0.4704|0.010|0.0092|2045|233|37.86|0.00316|
|E62-E1|ANAT|0.500|0.4899|0.010|0.0099|589|431|71.97|0.00120|
|E62-E1|FUNC|0.500|0.4650|0.010|0.0097|846|336|66.76|0.00141|
|E62-E1|CORR|0.500|0.5376|0.990|0.9913|1607|1601|3.26|0.41049|
|E62-E2|PHENO|0.500|0.5091|0.001|0.0010|565|1865|29.52|0.00024|
|E62-E2|ANAT|0.500|0.4674|0.001|0.0009|957|906|49.73|0.00018|
|E62-E2|FUNC|0.500|0.4497|0.001|0.0009|1583|799|32.67|0.00023|
|E62-E2|CORR|0.500|0.5553|0.990|0.9920|1738|1613|1.82|0.40069|
|E62-E2|PHENO|0.500|0.4899|0.001|0.0010|712|1167|47.69|0.00026|
|E62-E2|ANAT|0.500|0.4945|0.001|0.0010|695|371|70.93|0.00014|
|E62-E2|FUNC|0.500|0.4986|0.001|0.0010|691|961|54.43|0.00017|
|E62-E2|CORR|0.500|0.5377|0.990|0.9914|1663|1462|3.07|0.40913|
|E62-E2|PHENO|0.500|0.4683|0.001|0.0009|1571|626|36.23|0.00029|
|E62-E2|ANAT|0.500|0.4851|0.001|0.0010|388|309|81.00|0.00012|
|E62-E2|FUNC|0.500|0.4704|0.001|0.0009|1368|646|42.98|0.00022|
|E62-E2|CORR|0.500|0.5444|0.990|0.9916|1812|1640|2.71|0.41535|
|E62-E2|PHENO|0.500|0.4788|0.001|0.0010|1185|851|43.16|0.00026|
|E62-E2|ANAT|0.500|0.4912|0.001|0.0010|455|483|74.33|0.00011|
|E62-E2|FUNC|0.500|0.4633|0.001|0.0009|1342|339|52.58|0.00019|
|E62-E2|CORR|0.500|0.5441|0.990|0.9917|1745|1704|1.77|0.41168|
|E62-E2|PHENO|0.500|0.4911|0.001|0.0010|844|1751|24.01|0.00023|
|E62-E2|ANAT|0.500|0.4791|0.001|0.0010|564|335|75.57|0.00013|
|E62-E2|FUNC|0.500|0.4742|0.001|0.0010|802|1399|38.71|0.00017|
|E62-E2|CORR|0.500|0.5567|0.990|0.9923|1809|1685|1.41|0.40867|
|E62-E2|PHENO|0.500|0.5089|0.001|0.0010|240|1195|58.11|0.00015|
|E62-E2|ANAT|0.500|0.4986|0.001|0.0010|143|204|90.76|0.00005|
|E62-E2|FUNC|0.500|0.4857|0.001|0.0010|896|209|68.33|0.00017|
|E62-E2|CORR|0.500|0.5255|0.990|0.9909|1675|1631|6.90|0.39130|
|E62-E2|PHENO|0.500|0.4722|0.001|0.0009|2084|305|33.68|0.00033|
|E62-E2|ANAT|0.500|0.4988|0.001|0.0010|14|707|80.26|0.00004|
|E62-E2|FUNC|0.500|0.4795|0.001|0.0010|527|207|79.23|0.00011|
|E62-E2|CORR|0.500|0.5349|0.990|0.9912|1750|1590|2.88|0.42191|
|E62-E2|PHENO|0.500|0.5111|0.001|0.0011|737|1658|32.82|0.00026|
|E62-E2|ANAT|0.500|0.4945|0.001|0.0010|728|634|62.82|0.00016|
|E62-E2|FUNC|0.500|0.4676|0.001|0.0009|1042|1248|37.26|0.00020|
|E62-E2|CORR|0.500|0.5525|0.990|0.9920|1706|1571|1.80|0.40269|
|E62-E2|PHENO|0.500|0.4670|0.001|0.0009|2908|326|9.16|0.00035|
|E62-E2|ANAT|0.500|0.4882|0.001|0.0010|305|137|87.93|0.00009|
|E62-E2|FUNC|0.500|0.4835|0.001|0.0010|753|816|56.63|0.00016|
|E62-E2|CORR|0.500|0.5390|0.990|0.9914|1674|1457|1.63|0.41276|
|E62-E2|PHENO|0.500|0.4490|0.001|0.0008|1715|1040|22.26|0.00025|
|E62-E2|ANAT|0.500|0.4480|0.001|0.0009|1045|945|46.45|0.00017|
|E62-E2|FUNC|0.500|0.4340|0.001|0.0009|1269|1157|28.27|0.00019|
|E62-E2|CORR|0.500|0.5738|0.990|0.9929|1663|1752|1.59|0.37504|
|E62-E3|PHENO|0.400|0.4152|0.001|0.0010|784|1894|25.53|0.00027|
|E62-E3|ANAT|0.400|0.3570|0.001|0.0009|959|974|47.54|0.00015|
|E62-E3|FUNC|0.400|0.3446|0.001|0.0009|1501|1179|26.78|0.00021|
|E62-E3|CORR|0.400|0.4607|0.990|0.9922|1697|1661|1.64|0.38494|
|E62-E3|PHENO|0.400|0.3909|0.001|0.0010|798|1174|44.01|0.00020|
|E62-E3|ANAT|0.400|0.3881|0.001|0.0010|795|374|68.21|0.00013|
|E62-E3|FUNC|0.400|0.3918|0.001|0.0010|846|1050|48.60|0.00016|
|E62-E3|CORR|0.400|0.4438|0.990|0.9916|1788|1552|3.10|0.39690|
|E62-E3|PHENO|0.400|0.3562|0.001|0.0009|1297|1238|28.00|0.00022|
|E62-E3|ANAT|0.400|0.3747|0.001|0.0010|407|889|64.60|0.00010|
|E62-E3|FUNC|0.400|0.3548|0.001|0.0009|1602|1047|27.06|0.00022|
|E62-E3|CORR|0.400|0.4557|0.990|0.9921|1656|1602|1.75|0.38721|
|E62-E3|PHENO|0.400|0.3475|0.001|0.0009|1467|1166|25.45|0.00024|
|E62-E3|ANAT|0.400|0.3671|0.001|0.0009|873|827|53.60|0.00014|
|E62-E3|FUNC|0.400|0.3450|0.001|0.0009|1387|1029|31.54|0.00019|
|E62-E3|CORR|0.400|0.4656|0.990|0.9926|1735|1672|1.56|0.37212|
|E62-E3|PHENO|0.400|0.3843|0.001|0.0010|1050|1890|18.01|0.00024|
|E62-E3|ANAT|0.400|0.3732|0.001|0.0010|681|503|67.68|0.00012|
|E62-E3|FUNC|0.400|0.3647|0.001|0.0009|875|1353|33.87|0.00015|
|E62-E3|CORR|0.400|0.4612|0.990|0.9925|1738|1528|0.94|0.38309|
|E62-E3|PHENO|0.400|0.4092|0.001|0.0010|188|1188|60.25|0.00011|
|E62-E3|ANAT|0.400|0.3977|0.001|0.0010|196|318|86.14|0.00006|
|E62-E3|FUNC|0.400|0.3837|0.001|0.0010|840|189|70.82|0.00013|
|E62-E3|CORR|0.400|0.4277|0.990|0.9910|1604|1528|5.83|0.37494|
|E62-E3|PHENO|0.400|0.3676|0.001|0.0009|2020|505|28.49|0.00029|
|E62-E3|ANAT|0.400|0.4053|0.001|0.0010|95|996|70.46|0.00005|
|E62-E3|FUNC|0.400|0.3713|0.001|0.0010|864|621|57.93|0.00014|
|E62-E3|CORR|0.400|0.4454|0.990|0.9916|1817|1709|2.43|0.40558|
|E62-E3|PHENO|0.400|0.4041|0.001|0.0011|745|1570|31.33|0.00022|
|E62-E3|ANAT|0.400|0.3834|0.001|0.0009|988|576|57.67|0.00017|
|E62-E3|FUNC|0.400|0.3619|0.001|0.0009|1094|1310|32.66|0.00018|
|E62-E3|CORR|0.400|0.4638|0.990|0.9923|1820|1698|1.40|0.39462|
|E62-E3|PHENO|0.400|0.3669|0.001|0.0009|2880|238|8.64|0.00031|
|E62-E3|ANAT|0.400|0.3884|0.001|0.0010|319|132|87.77|0.00007|
|E62-E3|FUNC|0.400|0.3835|0.001|0.0010|736|805|57.03|0.00012|
|E62-E3|CORR|0.400|0.4402|0.990|0.9914|1759|1633|1.62|0.41105|
|E62-E3|PHENO|0.400|0.3779|0.001|0.0009|2370|254|26.29|0.00031|
|E62-E3|ANAT|0.400|0.3977|0.001|0.0010|348|167|85.92|0.00007|
|E62-E3|FUNC|0.400|0.3824|0.001|0.0010|776|244|71.15|0.00011|
|E62-E3|CORR|0.400|0.4330|0.990|0.9910|1703|1511|4.37|0.39126|
|E62-E4|PHENO|0.600|0.6025|0.001|0.0010|911|1930|17.05|0.00027|
|E62-E4|ANAT|0.600|0.5504|0.001|0.0009|992|1097|42.91|0.00018|
|E62-E4|FUNC|0.600|0.5384|0.001|0.0008|1623|1145|23.05|0.00026|
|E62-E4|CORR|0.600|0.6633|0.990|0.9926|1658|1714|1.69|0.37965|
|E62-E4|PHENO|0.600|0.5912|0.001|0.0010|426|1313|50.54|0.00020|
|E62-E4|ANAT|0.600|0.5994|0.001|0.0010|578|273|76.72|0.00014|
|E62-E4|FUNC|0.600|0.6029|0.001|0.0010|542|914|59.53|0.00017|
|E62-E4|CORR|0.600|0.6300|0.990|0.9912|1739|1582|3.91|0.42643|
|E62-E4|PHENO|0.600|0.5728|0.001|0.0009|1551|481|39.85|0.00030|
|E62-E4|ANAT|0.600|0.5884|0.001|0.0010|326|93|88.51|0.00012|
|E62-E4|FUNC|0.600|0.5758|0.001|0.0010|1198|607|48.90|0.00022|
|E62-E4|CORR|0.600|0.6358|0.990|0.9914|1674|1656|3.08|0.42259|
|E62-E4|PHENO|0.600|0.5878|0.001|0.0010|723|984|51.98|0.00023|
|E62-E4|ANAT|0.600|0.5959|0.001|0.0010|266|354|83.02|0.00010|
|E62-E4|FUNC|0.600|0.5695|0.001|0.0009|1325|237|56.34|0.00021|
|E62-E4|CORR|0.600|0.6361|0.990|0.9915|1630|1541|2.01|0.42033|
|E62-E4|PHENO|0.600|0.5717|0.001|0.0009|1234|1878|10.88|0.00029|
|E62-E4|ANAT|0.600|0.5570|0.001|0.0009|804|694|59.07|0.00015|
|E62-E4|FUNC|0.600|0.5491|0.001|0.0009|1116|1396|29.54|0.00022|
|E62-E4|CORR|0.600|0.6676|0.990|0.9930|1736|1676|1.47|0.38338|
|E62-E4|PHENO|0.600|0.6046|0.001|0.0010|296|1161|59.55|0.00019|
|E62-E4|ANAT|0.600|0.5984|0.001|0.0010|143|203|90.79|0.00006|
|E62-E4|FUNC|0.600|0.5864|0.001|0.0010|896|211|68.23|0.00020|
|E62-E4|CORR|0.600|0.6240|0.990|0.9909|1690|1606|7.05|0.40758|
|E62-E4|PHENO|0.600|0.5667|0.001|0.0009|1966|672|26.09|0.00033|
|E62-E4|ANAT|0.600|0.6039|0.001|0.0010|160|1105|65.67|0.00008|
|E62-E4|FUNC|0.600|0.5668|0.001|0.0009|1049|798|48.85|0.00022|
|E62-E4|CORR|0.600|0.6397|0.990|0.9918|1818|1688|1.82|0.41437|
|E62-E4|PHENO|0.600|0.6052|0.001|0.0011|524|1815|33.74|0.00023|
|E62-E4|ANAT|0.600|0.6013|0.001|0.0010|485|647|69.28|0.00014|
|E62-E4|FUNC|0.600|0.5702|0.001|0.0009|927|1124|42.65|0.00020|
|E62-E4|CORR|0.600|0.6438|0.990|0.9918|1851|1701|1.44|0.42382|
|E62-E4|PHENO|0.600|0.5624|0.001|0.0008|2738|202|12.11|0.00034|
|E62-E4|ANAT|0.600|0.5737|0.001|0.0010|460|389|76.83|0.00013|
|E62-E4|FUNC|0.600|0.5660|0.001|0.0009|997|1205|39.00|0.00022|
|E62-E4|CORR|0.600|0.6447|0.990|0.9919|1791|1692|1.50|0.40614|
|E62-E4|PHENO|0.600|0.5498|0.001|0.0008|1706|882|24.22|0.00025|
|E62-E4|ANAT|0.600|0.5569|0.001|0.0009|928|981|47.93|0.00017|
|E62-E4|FUNC|0.600|0.5353|0.001|0.0008|1349|1353|24.78|0.00021|
|E62-E4|CORR|0.600|0.6695|0.990|0.9931|1502|1636|0.95|0.36923|

### Fixed-checkpoint interventions

Each effect is full model minus intervention. Permutation is fixed independently of test labels.

|Group|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|
|---|---|---:|---:|---:|
|E62-E1|all_cadr_off|+0.000|0|0.001869|
|E62-E1|corr_cadr_off|+0.000|0|0.001868|
|E62-E1|direct_off|+0.000|0|0.001869|
|E62-E1|score_permutation|+0.000|0|0.002660|
|E62-E1|all_cadr_off|+0.000|0|0.001415|
|E62-E1|corr_cadr_off|+0.000|0|0.001416|
|E62-E1|direct_off|+0.000|0|0.001415|
|E62-E1|score_permutation|+0.000|0|0.001983|
|E62-E1|all_cadr_off|+0.000|0|0.001996|
|E62-E1|corr_cadr_off|+0.000|0|0.001992|
|E62-E1|direct_off|+0.000|0|0.001996|
|E62-E1|score_permutation|+0.000|0|0.001814|
|E62-E1|all_cadr_off|+0.000|0|0.001879|
|E62-E1|corr_cadr_off|+0.000|0|0.001924|
|E62-E1|direct_off|+0.000|0|0.001941|
|E62-E1|score_permutation|+0.000|0|0.002741|
|E62-E1|all_cadr_off|+0.000|0|0.001221|
|E62-E1|corr_cadr_off|+0.000|0|0.001223|
|E62-E1|direct_off|+0.000|0|0.001224|
|E62-E1|score_permutation|+0.000|0|0.001835|
|E62-E1|all_cadr_off|+0.000|0|0.000876|
|E62-E1|corr_cadr_off|+0.000|0|0.000891|
|E62-E1|direct_off|+0.000|0|0.000888|
|E62-E1|score_permutation|+0.000|0|0.001449|
|E62-E1|all_cadr_off|+0.000|0|0.001207|
|E62-E1|corr_cadr_off|+0.000|0|0.001209|
|E62-E1|direct_off|+0.000|0|0.001208|
|E62-E1|score_permutation|+0.000|0|0.001739|
|E62-E1|all_cadr_off|+0.000|0|0.001767|
|E62-E1|corr_cadr_off|+0.000|0|0.001767|
|E62-E1|direct_off|+0.000|0|0.001766|
|E62-E1|score_permutation|+0.000|0|0.002007|
|E62-E1|all_cadr_off|+0.000|0|0.001278|
|E62-E1|corr_cadr_off|+0.000|0|0.001282|
|E62-E1|direct_off|+0.000|0|0.001279|
|E62-E1|score_permutation|+0.000|0|0.002648|
|E62-E1|all_cadr_off|+1.149|1|0.001124|
|E62-E1|corr_cadr_off|+1.149|1|0.001131|
|E62-E1|direct_off|+1.149|1|0.001124|
|E62-E1|score_permutation|+1.149|1|0.002120|
|E62-E2|all_cadr_off|+0.000|0|0.001767|
|E62-E2|corr_cadr_off|+0.000|0|0.001767|
|E62-E2|direct_off|+0.000|0|0.001767|
|E62-E2|score_permutation|+0.000|0|0.002264|
|E62-E2|all_cadr_off|+1.149|1|0.001598|
|E62-E2|corr_cadr_off|+1.149|1|0.001598|
|E62-E2|direct_off|+1.149|1|0.001598|
|E62-E2|score_permutation|+0.000|0|0.002184|
|E62-E2|all_cadr_off|+0.000|0|0.002868|
|E62-E2|corr_cadr_off|+0.000|0|0.002868|
|E62-E2|direct_off|+0.000|0|0.002868|
|E62-E2|score_permutation|+0.000|0|0.003208|
|E62-E2|all_cadr_off|+0.000|0|0.001554|
|E62-E2|corr_cadr_off|+0.000|0|0.001554|
|E62-E2|direct_off|+0.000|0|0.001554|
|E62-E2|score_permutation|+0.000|0|0.001778|
|E62-E2|all_cadr_off|+0.000|0|0.001816|
|E62-E2|corr_cadr_off|+0.000|0|0.001816|
|E62-E2|direct_off|+0.000|0|0.001816|
|E62-E2|score_permutation|+1.149|1|0.002182|
|E62-E2|all_cadr_off|+0.000|0|0.001141|
|E62-E2|corr_cadr_off|+0.000|0|0.001141|
|E62-E2|direct_off|+0.000|0|0.001140|
|E62-E2|score_permutation|+0.000|0|0.001499|
|E62-E2|all_cadr_off|+0.000|0|0.001111|
|E62-E2|corr_cadr_off|+0.000|0|0.001112|
|E62-E2|direct_off|+0.000|0|0.001113|
|E62-E2|score_permutation|+0.000|0|0.001732|
|E62-E2|all_cadr_off|+0.000|0|0.001647|
|E62-E2|corr_cadr_off|+0.000|0|0.001647|
|E62-E2|direct_off|+0.000|0|0.001647|
|E62-E2|score_permutation|+0.000|0|0.002629|
|E62-E2|all_cadr_off|+0.000|0|0.001184|
|E62-E2|corr_cadr_off|+0.000|0|0.001184|
|E62-E2|direct_off|+0.000|0|0.001184|
|E62-E2|score_permutation|+0.000|0|0.001760|
|E62-E2|all_cadr_off|+0.000|0|0.000872|
|E62-E2|corr_cadr_off|+0.000|0|0.000872|
|E62-E2|direct_off|+0.000|0|0.000868|
|E62-E2|score_permutation|+0.000|0|0.001845|
|E62-E3|all_cadr_off|+0.000|0|0.001433|
|E62-E3|corr_cadr_off|+0.000|0|0.001433|
|E62-E3|direct_off|+0.000|0|0.001433|
|E62-E3|score_permutation|+0.000|0|0.001903|
|E62-E3|all_cadr_off|+0.000|0|0.001746|
|E62-E3|corr_cadr_off|+0.000|0|0.001747|
|E62-E3|direct_off|+0.000|0|0.001754|
|E62-E3|score_permutation|+0.000|0|0.002143|
|E62-E3|all_cadr_off|+1.149|1|0.002744|
|E62-E3|corr_cadr_off|+1.149|1|0.002744|
|E62-E3|direct_off|+1.149|1|0.002744|
|E62-E3|score_permutation|+1.149|1|0.002874|
|E62-E3|all_cadr_off|+0.000|0|0.001631|
|E62-E3|corr_cadr_off|+0.000|0|0.001631|
|E62-E3|direct_off|+0.000|0|0.001639|
|E62-E3|score_permutation|+0.000|0|0.003038|
|E62-E3|all_cadr_off|+0.000|0|0.002022|
|E62-E3|corr_cadr_off|+0.000|0|0.002022|
|E62-E3|direct_off|+0.000|0|0.002022|
|E62-E3|score_permutation|+0.000|0|0.002745|
|E62-E3|all_cadr_off|+0.000|0|0.001381|
|E62-E3|corr_cadr_off|+0.000|0|0.001380|
|E62-E3|direct_off|+0.000|0|0.001381|
|E62-E3|score_permutation|+0.000|0|0.001919|
|E62-E3|all_cadr_off|+0.000|0|0.002106|
|E62-E3|corr_cadr_off|+0.000|0|0.002106|
|E62-E3|direct_off|+0.000|0|0.002107|
|E62-E3|score_permutation|+0.000|0|0.003368|
|E62-E3|all_cadr_off|+1.149|1|0.001382|
|E62-E3|corr_cadr_off|+1.149|1|0.001382|
|E62-E3|direct_off|+1.149|1|0.001381|
|E62-E3|score_permutation|+0.000|0|0.001949|
|E62-E3|all_cadr_off|+0.000|0|0.001200|
|E62-E3|corr_cadr_off|+0.000|0|0.001200|
|E62-E3|direct_off|+0.000|0|0.001199|
|E62-E3|score_permutation|+0.000|0|0.002417|
|E62-E3|all_cadr_off|+0.000|0|0.001180|
|E62-E3|corr_cadr_off|+0.000|0|0.001181|
|E62-E3|direct_off|+0.000|0|0.001180|
|E62-E3|score_permutation|+0.000|0|0.001676|
|E62-E4|all_cadr_off|+1.136|1|0.002264|
|E62-E4|corr_cadr_off|+1.136|1|0.002264|
|E62-E4|direct_off|+1.136|1|0.001995|
|E62-E4|score_permutation|+1.136|1|0.003233|
|E62-E4|all_cadr_off|+0.000|0|0.001716|
|E62-E4|corr_cadr_off|+0.000|0|0.001716|
|E62-E4|direct_off|+0.000|0|0.001718|
|E62-E4|score_permutation|+1.149|1|0.002652|
|E62-E4|all_cadr_off|+0.000|0|0.003023|
|E62-E4|corr_cadr_off|+0.000|0|0.003024|
|E62-E4|direct_off|+0.000|0|0.003023|
|E62-E4|score_permutation|+0.000|0|0.003084|
|E62-E4|all_cadr_off|+0.000|0|0.002001|
|E62-E4|corr_cadr_off|+0.000|0|0.002002|
|E62-E4|direct_off|+0.000|0|0.002001|
|E62-E4|score_permutation|+0.000|0|0.002588|
|E62-E4|all_cadr_off|+0.000|0|0.000774|
|E62-E4|corr_cadr_off|+0.000|0|0.000775|
|E62-E4|direct_off|+0.000|0|0.000775|
|E62-E4|score_permutation|+0.000|0|0.000995|
|E62-E4|all_cadr_off|+0.000|0|0.001339|
|E62-E4|corr_cadr_off|+0.000|0|0.001339|
|E62-E4|direct_off|+0.000|0|0.001336|
|E62-E4|score_permutation|+0.000|0|0.001876|
|E62-E4|all_cadr_off|+0.000|0|0.001341|
|E62-E4|corr_cadr_off|+0.000|0|0.001342|
|E62-E4|direct_off|+0.000|0|0.001342|
|E62-E4|score_permutation|+0.000|0|0.001796|
|E62-E4|all_cadr_off|+1.149|1|0.001730|
|E62-E4|corr_cadr_off|+1.149|1|0.001730|
|E62-E4|direct_off|+1.149|1|0.001730|
|E62-E4|score_permutation|+2.299|2|0.002329|
|E62-E4|all_cadr_off|+0.000|0|0.001504|
|E62-E4|corr_cadr_off|+0.000|0|0.001503|
|E62-E4|direct_off|+0.000|0|0.001505|
|E62-E4|score_permutation|+1.149|1|0.001592|
|E62-E4|all_cadr_off|+0.000|0|0.000799|
|E62-E4|corr_cadr_off|+0.000|0|0.000799|
|E62-E4|direct_off|+0.000|0|0.000803|
|E62-E4|score_permutation|+0.000|0|0.001246|

## Scientific judgment

**Decision: NO-GO as evidence for a reliable diagnostic gain or a new method.** E62-E2 is the best new group on ACC (91.8443%) and Hard-AUC (91.6821%). Versus historical E61-E2 its paired changes are +0.345 pp ACC (95% descriptive CI -0.210 to +0.900; 4/5/1 wins/ties/losses), +0.434 pp Hard-AUC (CI -0.088 to +0.956; 7/1/2), and +1.738 pp SEN (CI +0.064 to +3.411; 6/3/1); Probability-AUC falls 0.612 pp and SPE falls 0.870 pp. Versus the 53.3 reference, E62-E2 remains lower by 0.230 pp ACC, 0.183 pp Hard-AUC, 0.467 pp Probability-AUC and 0.860 pp SPE, while SEN is 0.494 pp higher. The new settings therefore do not improve the full diagnostic profile or surpass the sole historical base.

At fixed edge initialization 0.5, reducing non-CORR fusion starts from 0.01 (E62-E1) to 0.001 (E62-E2) corresponds to +0.115 pp ACC, +0.202 pp Hard-AUC and +1.250 pp SEN, with -0.846 pp SPE and -0.041 pp Probability-AUC for E2 relative to E1; the paired intervals include zero. At fixed fusion initialization, edge starts 0.4/0.5/0.6 show no monotonic outcome trend. The 0.5 group has the best ACC and Hard-AUC, while 0.4 has the highest Probability-AUC among the four new groups. Final edge gates retain their starts rather than converging: fold means are about 0.382/0.383/0.368/0.450 for E62-E3 and 0.581/0.583/0.566/0.645 for E62-E4, in PHENO/ANAT/FUNC/CORR order.

Fusion gates also remain near their starts: group-mean CORR gates finish at 0.9917–0.9919, while non-CORR gates finish around 0.0093–0.0098 in E62-E1 and 0.0009–0.0010 in E62-E2–E4. The model did not learn to substantially increase the non-CORR disease-message share. Their mean best-epoch adaptive-logit gradient magnitudes were only about 0.9–1.4e-7 in E62-E2, compared with 2.7–3.6e-6 in E62-E1, consistent with the smaller sigmoid derivatives near 0.001; no gradient scaling was added. CORR carries the largest weighted CADR message, about 0.39–0.41% of the 320-D input norm, versus far smaller non-CORR contributions. Within the CORR CADR message, the scaled direct-neighbor-difference norm exceeds the positive-plus-negative GCN norm by roughly 218–1814 times across the groups. Yet disabling all CADR, disabling CORR CADR, or disabling direct differences changes only 1, 1, 2 and 2 hard predictions in E62-E1 through E62-E4 across the ten folds, respectively; mean absolute probability changes are about 0.0015–0.0017. E62-E4 score permutation changes five predictions, still a small effect. Thus message magnitude does not translate into a material classification gain, and the observed pathway is dominated numerically by the direct-difference term rather than disease-graph GCN messages.

All 60 pairwise group/fold RNG comparisons match exactly over 21,416 shared epochs. Thirty-nine tasks used runner SHA `c2036b6ac3db832e828a839b99c59b5c4c48fc7ff39910f0cdce4bc80da57acc`; E62-E1 fold 1 used `b3c23c4d5f6434c1db718d5a1958e418c388a82516655cd13f2d533016f0aeff`. After that first task, only the independent-audit source check and floating-point gate comparison were corrected; the training loop, model, configuration and locked dependencies did not change. Both runner hashes are retained per task in the manifest.

The intervals are descriptive only: this is one seed with overlapping CV folds, historical `val=test` checkpoint selection, pre-CV RFE/standardization and an 871-node transductive graph. These results do not establish unbiased external generalization. The initializations altered optimization trajectories, but fixed-checkpoint interventions show little hard-decision dependence on CADR propagation. Keep E62 as a gate-initialization sensitivity study, not as a new architecture or evidence of a dependable ASD diagnostic improvement.

## Files

- Experiment root: `/model/spromptgl/experiment/experiment62_e2e_gate_init_sensitivity`
- Task manifest: `manifest.json`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Gate diagnostics: `results/gate_diagnostics.csv`
- Independent checkpoint audit: `results/checkpoint_audit.csv`
