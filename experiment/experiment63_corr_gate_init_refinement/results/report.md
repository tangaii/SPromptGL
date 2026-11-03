# Experiment 63 — CORR Fusion and Edge Gate Initialization Refinement

## Protocol

Three groups vary only the sigmoid-space initialization of CADR edge gates and the CORR adaptive fusion gate from E62-E2. All groups start from each exact E61 shared epoch-10 checkpoint used by E62-E2, including GFR/CAM/CADR weights, all three optimizer states, and Python/NumPy/Torch CPU/CUDA RNG; configured gate logits are set immediately before epoch 11. E62-E2 best checkpoints are read-only references, not fine-tuning starts. The E61-E2 forward, loss, data, pair cache, fold split, epoch budget, early stopping, and checkpoint selection are reused. No OOF or reliability multiplier is used.

E61 is pinned to commit `31ced6bdd279654154b90dc2ece056d8e435df2b`. Its published source files and historical manifest were SHA-checked. Warmup files were individually checked against E61 manifest/task SHA, split and pair identity. The E61 training runner SHA recorded inside warmup is preserved separately from the published post-training audit runner SHA; model and config SHA match. Both gate parameters had empty optimizer-state entries after warmup, were already in the CADR AdamW parameter groups at WD=0.01, and their state was not reset.

History `val=test` checkpoint selection, pre-CV RFE/standardization and the transductive 871-node graph remain. Results are exploratory, not unbiased external generalization estimates.

The actual E61 epoch-10 relation scores were all between the 0.45/0.55 thresholds in fold 1, so the initial disease graphs were empty and edge-gate classification gradients were zero at that exact checkpoint. A separate deterministic controlled-edge test verified nonzero classification gradients through activated edge gates and relation scorers. Formal training determines whether the learned relation scores create active edges.

Formal status: **COMPLETE_AUDITED**, 30/30 tasks and 30/30 independent readbacks; device PPU-ZW810E, maximum workers 10.

## Classification results

Values are ten-fold mean ± population SD, percent. Historical groups are read-only.

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|53.3 third-group read-only|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|
|E61-E2 historical|91.4995 ± 2.4842|91.2480 ± 2.5179|93.8126 ± 2.6938|94.6485 ± 2.4105|87.8476 ± 3.5514|91.2480 ± 2.5179|90.5253 ± 2.7692|
|E62-E2 historical (read-only)|91.8443 ± 2.9346|91.6821 ± 2.8919|93.2010 ± 3.0942|93.7789 ± 4.2653|89.5854 ± 3.4710|91.6821 ± 2.8919|91.0789 ± 3.0892|
|E63-G1|91.6144 ± 3.0062|91.4214 ± 2.9596|93.3886 ± 3.0773|93.7697 ± 4.9973|89.0732 ± 4.0741|91.4214 ± 2.9596|90.8014 ± 3.1524|
|E63-G2|91.2696 ± 2.6890|91.0834 ± 2.7421|93.6322 ± 2.9261|93.5754 ± 3.0556|88.5915 ± 4.3192|91.0834 ± 2.7421|90.3613 ± 2.9716|
|E63-G3|91.6144 ± 3.2184|91.4219 ± 3.2192|93.5200 ± 3.0031|93.9963 ± 4.0740|88.8476 ± 4.3055|91.4219 ± 3.2192|90.7595 ± 3.4971|

## Paired comparisons

Differences are percentage points with descriptive paired-t 95% intervals over ten overlapping CV folds; they are not confirmatory confidence claims.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E63-G1 − 53.3-HISTORICAL|-0.460 (0/6/4; -0.884,-0.035)|-0.444 (1/5/4; -0.924,+0.036)|-0.280 (4/0/6; -0.869,+0.310)|-0.018 (1/6/3; -2.046,+2.009)|-0.870 (0/8/2; -2.372,+0.633)|
|E63-G1 − E61-E2-HISTORICAL|+0.115 (3/5/2; -0.492,+0.722)|+0.173 (4/2/4; -0.385,+0.731)|-0.424 (3/0/7; -1.062,+0.214)|+1.226 (4/4/2; -1.284,+3.736)|-0.879 (3/4/3; -3.631,+1.874)|
|E63-G1 − E62-E2-HISTORICAL|-0.230 (1/6/3; -0.750,+0.290)|-0.261 (2/5/3; -0.831,+0.309)|+0.188 (5/0/5; -0.415,+0.790)|-0.512 (1/6/3; -2.339,+1.314)|-0.009 (2/7/1; -1.270,+1.252)|
|E63-G2 − 53.3-HISTORICAL|-0.805 (0/3/7; -1.202,-0.407)|-0.782 (1/2/7; -1.198,-0.366)|-0.036 (4/0/6; -0.494,+0.422)|-0.500 (1/6/3; -1.616,+0.616)|-1.064 (0/5/5; -1.866,-0.262)|
|E63-G2 − E61-E2-HISTORICAL|-0.230 (0/8/2; -0.577,+0.117)|-0.165 (3/4/3; -0.494,+0.165)|-0.180 (4/0/6; -0.857,+0.496)|+0.744 (4/5/1; -0.458,+1.946)|-1.073 (1/4/5; -2.380,+0.233)|
|E63-G2 − E62-E2-HISTORICAL|-0.575 (0/5/5; -1.008,-0.141)|-0.599 (1/2/7; -1.040,-0.157)|+0.431 (6/0/4; -0.265,+1.128)|-0.994 (1/4/5; -2.226,+0.239)|-0.204 (2/5/3; -1.338,+0.931)|
|E63-G3 − 53.3-HISTORICAL|-0.460 (0/7/3; -1.035,+0.115)|-0.443 (2/5/3; -1.062,+0.175)|-0.148 (4/0/6; -0.693,+0.396)|-0.244 (2/6/2; -1.783,+1.295)|-0.643 (0/7/3; -1.384,+0.098)|
|E63-G3 − E61-E2-HISTORICAL|+0.115 (4/3/3; -0.605,+0.835)|+0.174 (6/0/4; -0.557,+0.905)|-0.293 (5/0/5; -1.004,+0.419)|+1.000 (5/2/3; -1.075,+3.075)|-0.652 (3/3/4; -2.581,+1.277)|
|E63-G3 − E62-E2-HISTORICAL|-0.230 (0/8/2; -0.577,+0.117)|-0.260 (0/8/2; -0.653,+0.133)|+0.319 (7/1/2; +0.048,+0.590)|-0.738 (0/8/2; -1.921,+0.445)|+0.217 (1/9/0; -0.274,+0.709)|
|E63-G1 − E63-G2|+0.345 (3/7/0; -0.052,+0.742)|+0.338 (5/3/2; -0.038,+0.714)|-0.244 (2/0/8; -0.738,+0.251)|+0.482 (2/6/2; -1.504,+2.468)|+0.194 (5/3/2; -1.794,+2.182)|
|E63-G1 − E63-G3|+0.000 (3/4/3; -0.671,+0.671)|-0.001 (4/3/3; -0.746,+0.745)|-0.131 (3/0/7; -0.617,+0.354)|+0.226 (3/4/3; -2.047,+2.498)|-0.227 (2/6/2; -1.579,+1.126)|
|E63-G2 − E63-G3|-0.345 (2/3/5; -1.022,+0.332)|-0.339 (3/1/6; -1.050,+0.373)|+0.112 (5/1/4; -0.584,+0.809)|-0.256 (3/3/4; -1.805,+1.293)|-0.421 (1/6/3; -1.391,+0.549)|

## Gate and propagation mechanisms

Initial and best-checkpoint gate values, edge counts, gradient magnitudes, individual GCN/direct messages, and normalized contribution are in `gate_diagnostics.csv`. A nonzero or larger message norm does not establish diagnostic benefit.

|Group|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E63-G1|PHENO|0.500|0.5175|0.001|0.0010|206|2298|27.86|0.00017|
|E63-G1|ANAT|0.500|0.4878|0.001|0.0010|654|320|73.65|0.00015|
|E63-G1|FUNC|0.500|0.4721|0.001|0.0010|1461|238|52.17|0.00021|
|E63-G1|CORR|0.500|0.5371|0.999|0.9991|1741|1597|2.85|0.42816|
|E63-G1|PHENO|0.500|0.4695|0.001|0.0009|1355|996|35.18|0.00027|
|E63-G1|ANAT|0.500|0.4705|0.001|0.0010|1016|470|59.54|0.00018|
|E63-G1|FUNC|0.500|0.4758|0.001|0.0010|987|1196|38.12|0.00021|
|E63-G1|CORR|0.500|0.5488|0.999|0.9992|1712|1686|1.65|0.39781|
|E63-G1|PHENO|0.500|0.4508|0.001|0.0009|1265|1154|31.90|0.00024|
|E63-G1|ANAT|0.500|0.4700|0.001|0.0010|432|1014|60.58|0.00012|
|E63-G1|FUNC|0.500|0.4442|0.001|0.0009|1633|965|26.09|0.00024|
|E63-G1|CORR|0.500|0.5651|0.999|0.9992|1797|1676|1.84|0.39409|
|E63-G1|PHENO|0.500|0.4885|0.001|0.0010|614|895|57.97|0.00021|
|E63-G1|ANAT|0.500|0.4982|0.001|0.0010|218|425|82.41|0.00008|
|E63-G1|FUNC|0.500|0.4729|0.001|0.0009|1311|180|58.24|0.00018|
|E63-G1|CORR|0.500|0.5368|0.999|0.9991|1745|1624|2.23|0.41950|
|E63-G1|PHENO|0.500|0.4766|0.001|0.0009|1122|1797|14.60|0.00023|
|E63-G1|ANAT|0.500|0.4559|0.001|0.0009|799|679|59.64|0.00014|
|E63-G1|FUNC|0.500|0.4470|0.001|0.0009|991|1581|27.37|0.00019|
|E63-G1|CORR|0.500|0.5727|0.999|0.9993|1435|1300|0.69|0.36806|
|E63-G1|PHENO|0.500|0.4717|0.001|0.0009|1170|1390|26.35|0.00026|
|E63-G1|ANAT|0.500|0.4868|0.001|0.0010|761|948|53.18|0.00017|
|E63-G1|FUNC|0.500|0.4406|0.001|0.0009|1423|880|34.67|0.00022|
|E63-G1|CORR|0.500|0.5653|0.999|0.9992|1740|1536|3.73|0.37472|
|E63-G1|PHENO|0.500|0.4711|0.001|0.0009|2109|394|30.01|0.00031|
|E63-G1|ANAT|0.500|0.4999|0.001|0.0010|22|715|79.89|0.00004|
|E63-G1|FUNC|0.500|0.4782|0.001|0.0010|704|216|74.25|0.00014|
|E63-G1|CORR|0.500|0.5354|0.999|0.9991|1511|1441|2.96|0.40698|
|E63-G1|PHENO|0.500|0.5178|0.001|0.0011|299|1903|38.88|0.00019|
|E63-G1|ANAT|0.500|0.5064|0.001|0.0010|151|586|79.93|0.00006|
|E63-G1|FUNC|0.500|0.4767|0.001|0.0009|755|866|53.63|0.00016|
|E63-G1|CORR|0.500|0.5394|0.999|0.9991|1563|1432|2.38|0.41667|
|E63-G1|PHENO|0.500|0.4723|0.001|0.0009|3082|173|13.29|0.00038|
|E63-G1|ANAT|0.500|0.4929|0.001|0.0010|190|108|92.04|0.00006|
|E63-G1|FUNC|0.500|0.4904|0.001|0.0010|581|570|68.38|0.00014|
|E63-G1|CORR|0.500|0.5341|0.999|0.9991|1833|1639|2.06|0.42841|
|E63-G1|PHENO|0.500|0.4744|0.001|0.0009|2093|329|32.00|0.00032|
|E63-G1|ANAT|0.500|0.4904|0.001|0.0010|584|425|72.29|0.00012|
|E63-G1|FUNC|0.500|0.4762|0.001|0.0010|863|367|65.36|0.00015|
|E63-G1|CORR|0.500|0.5361|0.999|0.9991|1741|1661|2.94|0.42217|
|E63-G2|PHENO|0.450|0.4671|0.001|0.0010|512|1994|31.81|0.00025|
|E63-G2|ANAT|0.450|0.4193|0.001|0.0010|987|810|51.09|0.00018|
|E63-G2|FUNC|0.450|0.4006|0.001|0.0009|1643|786|31.29|0.00022|
|E63-G2|CORR|0.450|0.5087|0.999|0.9992|1704|1546|1.90|0.39915|
|E63-G2|PHENO|0.450|0.4157|0.001|0.0009|1211|1094|31.70|0.00023|
|E63-G2|ANAT|0.450|0.4131|0.001|0.0010|1062|505|57.28|0.00016|
|E63-G2|FUNC|0.450|0.4209|0.001|0.0009|1108|1154|35.52|0.00020|
|E63-G2|CORR|0.450|0.5119|0.999|0.9992|1783|1767|1.72|0.38991|
|E63-G2|PHENO|0.450|0.4145|0.001|0.0009|1406|1133|28.62|0.00025|
|E63-G2|ANAT|0.450|0.4292|0.001|0.0010|413|561|73.41|0.00011|
|E63-G2|FUNC|0.450|0.4137|0.001|0.0009|1501|814|34.38|0.00022|
|E63-G2|CORR|0.450|0.5008|0.999|0.9992|1599|1470|2.26|0.39918|
|E63-G2|PHENO|0.450|0.4130|0.001|0.0009|1053|1279|34.59|0.00022|
|E63-G2|ANAT|0.450|0.4274|0.001|0.0010|700|717|61.28|0.00013|
|E63-G2|FUNC|0.450|0.4013|0.001|0.0009|1374|712|40.60|0.00019|
|E63-G2|CORR|0.450|0.5063|0.999|0.9992|1811|1712|1.67|0.39444|
|E63-G2|PHENO|0.450|0.4274|0.001|0.0009|1151|1886|15.78|0.00025|
|E63-G2|ANAT|0.450|0.4154|0.001|0.0010|687|538|66.62|0.00012|
|E63-G2|FUNC|0.450|0.4095|0.001|0.0009|1010|1505|31.60|0.00018|
|E63-G2|CORR|0.450|0.5214|0.999|0.9993|1582|1508|1.25|0.38734|
|E63-G2|PHENO|0.450|0.4549|0.001|0.0010|388|1267|51.14|0.00014|
|E63-G2|ANAT|0.450|0.4469|0.001|0.0010|265|589|76.44|0.00008|
|E63-G2|FUNC|0.450|0.4281|0.001|0.0010|1019|284|63.51|0.00015|
|E63-G2|CORR|0.450|0.4805|0.999|0.9991|1762|1567|3.90|0.41796|
|E63-G2|PHENO|0.450|0.4298|0.001|0.0009|2165|109|34.30|0.00031|
|E63-G2|ANAT|0.450|0.4451|0.001|0.0010|13|434|88.00|0.00003|
|E63-G2|FUNC|0.450|0.4405|0.001|0.0010|566|32|83.91|0.00011|
|E63-G2|CORR|0.450|0.4749|0.999|0.9991|1517|1392|4.97|0.38353|
|E63-G2|PHENO|0.450|0.4671|0.001|0.0011|694|1774|30.10|0.00024|
|E63-G2|ANAT|0.450|0.4474|0.001|0.0010|662|645|64.42|0.00015|
|E63-G2|FUNC|0.450|0.4188|0.001|0.0009|1100|1279|34.61|0.00021|
|E63-G2|CORR|0.450|0.5024|0.999|0.9992|1288|1196|2.05|0.36116|
|E63-G2|PHENO|0.450|0.4210|0.001|0.0009|2972|186|13.38|0.00035|
|E63-G2|ANAT|0.450|0.4426|0.001|0.0010|173|107|92.58|0.00005|
|E63-G2|FUNC|0.450|0.4403|0.001|0.0010|437|603|70.35|0.00009|
|E63-G2|CORR|0.450|0.4845|0.999|0.9991|1838|1637|2.22|0.42419|
|E63-G2|PHENO|0.450|0.4126|0.001|0.0009|1752|744|31.11|0.00026|
|E63-G2|ANAT|0.450|0.4188|0.001|0.0010|910|527|60.82|0.00015|
|E63-G2|FUNC|0.450|0.4070|0.001|0.0009|1141|906|42.11|0.00015|
|E63-G2|CORR|0.450|0.5007|0.999|0.9992|1788|1704|2.59|0.39724|
|E63-G3|PHENO|0.550|0.5648|0.001|0.0010|441|1733|39.26|0.00024|
|E63-G3|ANAT|0.550|0.5296|0.001|0.0010|757|648|61.54|0.00019|
|E63-G3|FUNC|0.550|0.5130|0.001|0.0009|1540|477|42.83|0.00024|
|E63-G3|CORR|0.550|0.5929|0.999|0.9992|1756|1567|1.86|0.42116|
|E63-G3|PHENO|0.550|0.5458|0.001|0.0010|432|1113|55.83|0.00018|
|E63-G3|ANAT|0.550|0.5535|0.001|0.0010|366|184|85.05|0.00010|
|E63-G3|FUNC|0.550|0.5551|0.001|0.0010|272|661|73.28|0.00010|
|E63-G3|CORR|0.550|0.5799|0.999|0.9991|1634|1555|3.39|0.41977|
|E63-G3|PHENO|0.550|0.5192|0.001|0.0009|1409|715|38.74|0.00026|
|E63-G3|ANAT|0.550|0.5343|0.001|0.0010|350|187|85.34|0.00012|
|E63-G3|FUNC|0.550|0.5208|0.001|0.0009|1266|625|46.32|0.00022|
|E63-G3|CORR|0.550|0.5889|0.999|0.9991|1691|1613|2.62|0.41816|
|E63-G3|PHENO|0.550|0.4916|0.001|0.0009|1410|1235|23.69|0.00026|
|E63-G3|ANAT|0.550|0.5048|0.001|0.0009|824|1018|50.54|0.00017|
|E63-G3|FUNC|0.550|0.4884|0.001|0.0008|1645|937|29.38|0.00023|
|E63-G3|CORR|0.550|0.6175|0.999|0.9993|1685|1623|1.28|0.37657|
|E63-G3|PHENO|0.550|0.5258|0.001|0.0009|1065|1825|15.67|0.00026|
|E63-G3|ANAT|0.550|0.5203|0.001|0.0010|701|506|67.05|0.00015|
|E63-G3|FUNC|0.550|0.5085|0.001|0.0009|964|1438|32.85|0.00019|
|E63-G3|CORR|0.550|0.6121|0.999|0.9993|1812|1633|1.03|0.40101|
|E63-G3|PHENO|0.550|0.5194|0.001|0.0009|1273|1484|20.91|0.00026|
|E63-G3|ANAT|0.550|0.5350|0.001|0.0010|733|992|52.91|0.00018|
|E63-G3|FUNC|0.550|0.4911|0.001|0.0009|1511|896|32.37|0.00024|
|E63-G3|CORR|0.550|0.6057|0.999|0.9992|1730|1682|1.76|0.38977|
|E63-G3|PHENO|0.550|0.5236|0.001|0.0009|1996|209|34.97|0.00032|
|E63-G3|ANAT|0.550|0.5482|0.001|0.0010|16|627|82.57|0.00005|
|E63-G3|FUNC|0.550|0.5327|0.001|0.0010|452|96|84.49|0.00010|
|E63-G3|CORR|0.550|0.5811|0.999|0.9991|1695|1588|2.84|0.42329|
|E63-G3|PHENO|0.550|0.5665|0.001|0.0011|278|1967|36.13|0.00022|
|E63-G3|ANAT|0.550|0.5547|0.001|0.0010|38|499|85.42|0.00003|
|E63-G3|FUNC|0.550|0.5287|0.001|0.0009|909|766|54.83|0.00020|
|E63-G3|CORR|0.550|0.5865|0.999|0.9991|1795|1588|1.97|0.43066|
|E63-G3|PHENO|0.550|0.5113|0.001|0.0008|2831|269|11.15|0.00035|
|E63-G3|ANAT|0.550|0.5272|0.001|0.0010|439|332|79.02|0.00012|
|E63-G3|FUNC|0.550|0.5192|0.001|0.0009|1087|1133|38.49|0.00021|
|E63-G3|CORR|0.550|0.5970|0.999|0.9992|1781|1655|1.43|0.40464|
|E63-G3|PHENO|0.550|0.5086|0.001|0.0009|2041|712|21.88|0.00031|
|E63-G3|ANAT|0.550|0.5156|0.001|0.0009|1048|727|51.70|0.00019|
|E63-G3|FUNC|0.550|0.4991|0.001|0.0009|1237|1042|36.82|0.00020|
|E63-G3|CORR|0.550|0.6036|0.999|0.9992|1785|1702|2.46|0.39735|

### Fixed-checkpoint interventions

Each effect is full model minus intervention. Permutation is fixed independently of test labels.

|Group|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|
|---|---|---:|---:|---:|
|E63-G1|all_cadr_off|+0.000|0|0.002415|
|E63-G1|corr_cadr_off|+0.000|0|0.002415|
|E63-G1|direct_off|+0.000|0|0.002415|
|E63-G1|score_permutation|+0.000|0|0.003413|
|E63-G1|all_cadr_off|+0.000|0|0.001737|
|E63-G1|corr_cadr_off|+0.000|0|0.001737|
|E63-G1|direct_off|+0.000|0|0.001737|
|E63-G1|score_permutation|+0.000|0|0.002501|
|E63-G1|all_cadr_off|+0.000|0|0.002456|
|E63-G1|corr_cadr_off|+0.000|0|0.002458|
|E63-G1|direct_off|+0.000|0|0.002457|
|E63-G1|score_permutation|+1.149|1|0.002792|
|E63-G1|all_cadr_off|+1.149|1|0.001555|
|E63-G1|corr_cadr_off|+1.149|1|0.001555|
|E63-G1|direct_off|+1.149|1|0.001555|
|E63-G1|score_permutation|+0.000|0|0.001955|
|E63-G1|all_cadr_off|+0.000|0|0.001055|
|E63-G1|corr_cadr_off|+0.000|0|0.001056|
|E63-G1|direct_off|+0.000|0|0.001056|
|E63-G1|score_permutation|+0.000|0|0.001463|
|E63-G1|all_cadr_off|+0.000|0|0.000858|
|E63-G1|corr_cadr_off|+0.000|0|0.000858|
|E63-G1|direct_off|+0.000|0|0.000858|
|E63-G1|score_permutation|+0.000|0|0.001191|
|E63-G1|all_cadr_off|+0.000|0|0.001300|
|E63-G1|corr_cadr_off|+0.000|0|0.001300|
|E63-G1|direct_off|+0.000|0|0.001300|
|E63-G1|score_permutation|+0.000|0|0.001970|
|E63-G1|all_cadr_off|+0.000|0|0.001607|
|E63-G1|corr_cadr_off|+0.000|0|0.001607|
|E63-G1|direct_off|+0.000|0|0.001608|
|E63-G1|score_permutation|+0.000|0|0.002545|
|E63-G1|all_cadr_off|+0.000|0|0.001377|
|E63-G1|corr_cadr_off|+0.000|0|0.001377|
|E63-G1|direct_off|+0.000|0|0.001377|
|E63-G1|score_permutation|+0.000|0|0.002422|
|E63-G1|all_cadr_off|+0.000|0|0.001412|
|E63-G1|corr_cadr_off|+0.000|0|0.001413|
|E63-G1|direct_off|+0.000|0|0.001412|
|E63-G1|score_permutation|+0.000|0|0.002264|
|E63-G2|all_cadr_off|+0.000|0|0.001951|
|E63-G2|corr_cadr_off|+0.000|0|0.001951|
|E63-G2|direct_off|+0.000|0|0.001948|
|E63-G2|score_permutation|+0.000|0|0.002887|
|E63-G2|all_cadr_off|+0.000|0|0.001768|
|E63-G2|corr_cadr_off|+0.000|0|0.001769|
|E63-G2|direct_off|+0.000|0|0.001768|
|E63-G2|score_permutation|+0.000|0|0.002077|
|E63-G2|all_cadr_off|+0.000|0|0.002048|
|E63-G2|corr_cadr_off|+0.000|0|0.002048|
|E63-G2|direct_off|+0.000|0|0.002048|
|E63-G2|score_permutation|+0.000|0|0.002448|
|E63-G2|all_cadr_off|+1.149|1|0.001614|
|E63-G2|corr_cadr_off|+1.149|1|0.001614|
|E63-G2|direct_off|+1.149|1|0.001611|
|E63-G2|score_permutation|+1.149|1|0.002462|
|E63-G2|all_cadr_off|+0.000|0|0.001635|
|E63-G2|corr_cadr_off|+0.000|0|0.001635|
|E63-G2|direct_off|+0.000|0|0.001635|
|E63-G2|score_permutation|+0.000|0|0.002163|
|E63-G2|all_cadr_off|+1.149|1|0.001576|
|E63-G2|corr_cadr_off|+1.149|1|0.001576|
|E63-G2|direct_off|+1.149|1|0.001576|
|E63-G2|score_permutation|+0.000|0|0.001981|
|E63-G2|all_cadr_off|+0.000|0|0.001150|
|E63-G2|corr_cadr_off|+0.000|0|0.001151|
|E63-G2|direct_off|+0.000|0|0.001150|
|E63-G2|score_permutation|+1.149|1|0.002178|
|E63-G2|all_cadr_off|+0.000|0|0.001363|
|E63-G2|corr_cadr_off|+0.000|0|0.001364|
|E63-G2|direct_off|+0.000|0|0.001363|
|E63-G2|score_permutation|+1.149|1|0.002024|
|E63-G2|all_cadr_off|+0.000|0|0.001659|
|E63-G2|corr_cadr_off|+0.000|0|0.001658|
|E63-G2|direct_off|+0.000|0|0.001659|
|E63-G2|score_permutation|+0.000|0|0.002457|
|E63-G2|all_cadr_off|+0.000|0|0.000955|
|E63-G2|corr_cadr_off|+0.000|0|0.000956|
|E63-G2|direct_off|+0.000|0|0.000955|
|E63-G2|score_permutation|+0.000|0|0.001122|
|E63-G3|all_cadr_off|+1.136|1|0.002199|
|E63-G3|corr_cadr_off|+1.136|1|0.002199|
|E63-G3|direct_off|+1.136|1|0.002199|
|E63-G3|score_permutation|+1.136|1|0.003177|
|E63-G3|all_cadr_off|+0.000|0|0.001573|
|E63-G3|corr_cadr_off|+0.000|0|0.001573|
|E63-G3|direct_off|+0.000|0|0.001572|
|E63-G3|score_permutation|+0.000|0|0.002675|
|E63-G3|all_cadr_off|+0.000|0|0.002757|
|E63-G3|corr_cadr_off|+0.000|0|0.002758|
|E63-G3|direct_off|+0.000|0|0.002765|
|E63-G3|score_permutation|+0.000|0|0.002916|
|E63-G3|all_cadr_off|+0.000|0|0.001474|
|E63-G3|corr_cadr_off|+0.000|0|0.001475|
|E63-G3|direct_off|+0.000|0|0.001475|
|E63-G3|score_permutation|+0.000|0|0.002671|
|E63-G3|all_cadr_off|+0.000|0|0.001873|
|E63-G3|corr_cadr_off|+0.000|0|0.001873|
|E63-G3|direct_off|+0.000|0|0.001873|
|E63-G3|score_permutation|+0.000|0|0.002598|
|E63-G3|all_cadr_off|+0.000|0|0.000992|
|E63-G3|corr_cadr_off|+0.000|0|0.000992|
|E63-G3|direct_off|+0.000|0|0.000992|
|E63-G3|score_permutation|+0.000|0|0.001600|
|E63-G3|all_cadr_off|+1.149|1|0.001345|
|E63-G3|corr_cadr_off|+1.149|1|0.001345|
|E63-G3|direct_off|+1.149|1|0.001346|
|E63-G3|score_permutation|+1.149|1|0.002055|
|E63-G3|all_cadr_off|+1.149|1|0.001892|
|E63-G3|corr_cadr_off|+1.149|1|0.001893|
|E63-G3|direct_off|+1.149|1|0.001892|
|E63-G3|score_permutation|+1.149|1|0.002916|
|E63-G3|all_cadr_off|+0.000|0|0.001919|
|E63-G3|corr_cadr_off|+0.000|0|0.001919|
|E63-G3|direct_off|+0.000|0|0.001943|
|E63-G3|score_permutation|+0.000|0|0.002363|
|E63-G3|all_cadr_off|+0.000|0|0.000842|
|E63-G3|corr_cadr_off|+0.000|0|0.000842|
|E63-G3|direct_off|+0.000|0|0.000842|
|E63-G3|score_permutation|+0.000|0|0.001631|

## Scientific judgment

These groups test initialization sensitivity, not a new architecture. E63-G1 changes only the CORR fusion start from E62-E2's 0.99 to 0.999. Its paired change versus E62-E2 is -0.230 pp ACC (95% descriptive CI -0.750 to +0.290; W/T/L 1/6/3), -0.261 pp Hard-AUC (-0.831 to +0.309; 2/5/3), +0.188 pp Probability-AUC (-0.415 to +0.790; 5/0/5), -0.512 pp SEN (-2.339 to +1.314), and -0.009 pp SPE. This does not show a reliable gain from starting CORR at 0.999.

E63-G2 (edge start 0.45) is lower than E62-E2 by 0.575 pp ACC and 0.599 pp Hard-AUC; both descriptive paired intervals exclude zero, but this is a single-seed, overlapping-fold comparison. E63-G3 (edge start 0.55) is lower by 0.230 pp ACC and 0.260 pp Hard-AUC, while Probability-AUC is higher by 0.319 pp (95% descriptive CI +0.048 to +0.590; W/T/L 7/1/2). Its SEN is lower by 0.738 pp. No new setting improves the full diagnostic profile. Across E63 groups, ACC ties between G1/G3 at 91.6144%; G3 has the highest Hard-AUC by a negligible 0.0005 pp, G2 the highest Probability-AUC, and G1 the highest SEN.

Mean best-checkpoint gates retain their initialized separation. In `[PHENO, ANAT, FUNC, CORR]` order, edge means are G1 `[0.4810, 0.4859, 0.4674, 0.5471]`, G2 `[0.4323, 0.4305, 0.4181, 0.4992]`, and G3 `[0.5276, 0.5323, 0.5156, 0.5965]`. CORR fusion remains about 0.99917–0.99918 and non-CORR fusion remains about 0.00093–0.00098. Average positive/negative edge counts and uncertain-edge fractions are similar across groups. CORR's weighted CADR message is about 0.4% of input norm; its positive/negative GCN components remain small relative to the direct-difference component.

At fixed checkpoints, turning all CADR off, CORR CADR off, or direct differences off changes 1, 2, and 3 hard predictions across ten folds for G1, G2, and G3, respectively; mean absolute probability shifts are about 0.0016–0.0017. Fixed score permutation changes 1/3/3 predictions. Thus the raised CORR fusion initialization does not produce a substantial measured dependence of hard diagnoses on CADR propagation. All 30 pairwise group/fold RNG comparisons matched exactly over 10,757 shared epochs.

**Decision: NO-GO as evidence of diagnostic improvement.** The most defensible finding is limited initialization sensitivity: G3 raises Probability-AUC against E62-E2 while lowering ACC, Hard-AUC, and SEN; G1 does not benefit overall from CORR=0.999. These results do not surpass the 53.3 reference on the full metric profile and do not establish unbiased external generalization. The folds overlap in training patients, use one seed and retain historical `val=test`, pre-CV preprocessing, and the transductive graph protocol.

## Files

- Experiment root: `/model/spromptgl/experiment/experiment63_corr_gate_init_refinement`
- Task manifest: `manifest.json`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Gate diagnostics: `results/gate_diagnostics.csv`
- Independent checkpoint audit: `results/checkpoint_audit.csv`
