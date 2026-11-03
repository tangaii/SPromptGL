# Experiment 64 — CORR Adaptive Fusion Gate Initialization Sensitivity

## Protocol

Three groups vary only the sigmoid-space initial value of the CORR adaptive fusion gate from E62-E2: 0.995, 0.992, and 0.993. Edge gates stay at 0.5 and PHENO/ANAT/FUNC fusion gates stay at 0.001. All groups start from each exact E61 shared epoch-10 checkpoint used by E62-E2, including GFR/CAM/CADR weights, all three optimizer states, and Python/NumPy/Torch CPU/CUDA RNG; configured gate logits are set immediately before epoch 11. E62-E2 and E63-G1 best checkpoints are read-only references, not fine-tuning starts. The E61-E2 forward, loss, data, pair cache, fold split, epoch budget, early stopping, and checkpoint selection are reused. No OOF or reliability multiplier is used.

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
|E63-G1 (CORR 0.999, read-only)|91.6144 ± 3.0062|91.4214 ± 2.9596|93.3886 ± 3.0773|93.7697 ± 4.9973|89.0732 ± 4.0741|91.4214 ± 2.9596|90.8014 ± 3.1524|
|E64-G1|91.4995 ± 2.7851|91.2645 ± 2.7442|93.5413 ± 2.4086|94.1998 ± 4.3526|88.3293 ± 3.5640|91.2645 ± 2.7442|90.6101 ± 2.9352|
|E64-G2|91.2696 ± 3.3458|90.9995 ± 3.3631|93.5018 ± 3.2356|94.6392 ± 3.5168|87.3598 ± 4.3134|90.9995 ± 3.3631|90.2571 ± 3.6972|
|E64-G3|91.4995 ± 3.0130|91.2543 ± 2.9998|93.5647 ± 2.9441|94.4172 ± 4.2557|88.0915 ± 4.1228|91.2543 ± 2.9998|90.5754 ± 3.2499|

## Paired comparisons

Differences are percentage points with descriptive paired-t 95% intervals over ten overlapping CV folds; they are not confirmatory confidence claims.

|Comparison|ACC|Hard-AUC|Probability-AUC|SEN-ASD|SPE-TD|
|---|---:|---:|---:|---:|---:|
|E64-G1 − 53.3-HISTORICAL|-0.575 (1/3/6; -1.156,+0.007)|-0.601 (2/2/6; -1.198,-0.004)|-0.127 (5/0/5; -1.210,+0.956)|-0.762 (2/3/5; -2.219,+0.695)|-0.439 (1/7/2; -1.661,+0.782)|
|E64-G1 − E61-E2-HISTORICAL|+0.000 (2/6/2; -0.548,+0.548)|+0.017 (3/2/5; -0.498,+0.531)|-0.271 (3/1/6; -1.140,+0.598)|+0.482 (3/4/3; -1.513,+2.476)|-0.449 (4/3/3; -2.609,+1.711)|
|E64-G1 − E62-E2-HISTORICAL|-0.345 (1/5/4; -0.900,+0.210)|-0.418 (2/3/5; -1.011,+0.176)|+0.340 (6/0/4; -0.527,+1.207)|-1.256 (1/4/5; -2.767,+0.255)|+0.421 (3/6/1; -0.549,+1.391)|
|E64-G1 − E63-G1-HISTORICAL|-0.115 (1/7/2; -0.582,+0.352)|-0.157 (2/5/3; -0.640,+0.326)|+0.153 (4/0/6; -0.719,+1.024)|-0.744 (1/6/3; -2.189,+0.701)|+0.430 (2/7/1; -0.793,+1.653)|
|E64-G2 − 53.3-HISTORICAL|-0.805 (0/5/5; -1.482,-0.128)|-0.866 (0/5/5; -1.582,-0.149)|-0.167 (5/0/5; -0.998,+0.664)|-1.732 (0/5/5; -3.187,-0.276)|+0.000 (1/8/1; -0.733,+0.733)|
|E64-G2 − E61-E2-HISTORICAL|-0.230 (3/4/3; -1.163,+0.704)|-0.249 (3/3/4; -1.175,+0.678)|-0.311 (5/0/5; -1.527,+0.906)|-0.488 (1/6/3; -1.607,+0.631)|-0.009 (3/4/3; -1.261,+1.243)|
|E64-G2 − E62-E2-HISTORICAL|-0.575 (0/6/4; -1.156,+0.007)|-0.683 (0/5/5; -1.330,-0.035)|+0.301 (6/0/4; -0.579,+1.180)|-2.226 (0/5/5; -4.166,-0.285)|+0.860 (3/7/0; -0.221,+1.941)|
|E64-G2 − E63-G1-HISTORICAL|-0.345 (1/5/4; -0.900,+0.210)|-0.422 (1/5/4; -1.073,+0.229)|+0.113 (6/0/4; -0.946,+1.172)|-1.713 (1/5/4; -4.575,+1.149)|+0.870 (1/9/0; -1.098,+2.837)|
|E64-G3 − 53.3-HISTORICAL|-0.575 (0/6/4; -1.156,+0.007)|-0.611 (2/3/5; -1.266,+0.044)|-0.104 (5/0/5; -0.604,+0.397)|-1.000 (2/3/5; -3.232,+1.232)|-0.222 (2/6/2; -1.571,+1.127)|
|E64-G3 − E61-E2-HISTORICAL|+0.000 (3/4/3; -0.671,+0.671)|+0.006 (3/2/5; -0.664,+0.677)|-0.248 (3/0/7; -1.029,+0.533)|+0.244 (2/4/4; -2.168,+2.656)|-0.231 (4/4/2; -2.588,+2.126)|
|E64-G3 − E62-E2-HISTORICAL|-0.345 (1/5/4; -0.900,+0.210)|-0.428 (2/4/4; -1.011,+0.155)|+0.364 (8/0/2; -0.245,+0.973)|-1.494 (1/4/5; -3.200,+0.212)|+0.638 (3/6/1; -0.624,+1.900)|
|E64-G3 − E63-G1-HISTORICAL|-0.115 (1/7/2; -0.582,+0.352)|-0.167 (1/4/5; -0.714,+0.380)|+0.176 (5/0/5; -0.531,+0.883)|-0.982 (1/4/5; -2.887,+0.924)|+0.648 (4/5/1; -0.387,+1.683)|
|E64-G1 − E64-G2|+0.230 (4/4/2; -0.419,+0.878)|+0.265 (4/4/2; -0.456,+0.986)|+0.040 (6/0/4; -0.975,+1.054)|+0.970 (3/5/2; -1.251,+3.190)|-0.439 (1/7/2; -1.661,+0.782)|
|E64-G1 − E64-G3|+0.000 (3/4/3; -0.671,+0.671)|+0.010 (4/2/4; -0.733,+0.753)|-0.023 (7/0/3; -1.066,+1.019)|+0.238 (3/4/3; -2.178,+2.654)|-0.217 (3/4/3; -1.749,+1.314)|
|E64-G2 − E64-G3|-0.230 (2/4/4; -0.878,+0.419)|-0.255 (3/3/4; -0.987,+0.478)|-0.063 (5/1/4; -1.059,+0.933)|-0.732 (3/5/2; -3.485,+2.021)|+0.222 (2/5/3; -1.630,+2.074)|

## Gate and propagation mechanisms

Initial and best-checkpoint gate values, edge counts, gradient magnitudes, individual GCN/direct messages, and normalized contribution are in `gate_diagnostics.csv`. A nonzero or larger message norm does not establish diagnostic benefit.

|Group|Modality|Initial edge|Final edge|Initial fusion|Final fusion|Positive edges|Negative edges|Uncertain %|CADR / input %|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E64-G1|PHENO|0.500|0.5163|0.001|0.0010|393|1733|39.26|0.00022|
|E64-G1|ANAT|0.500|0.4746|0.001|0.0010|856|744|56.46|0.00018|
|E64-G1|FUNC|0.500|0.4588|0.001|0.0009|1557|620|38.62|0.00022|
|E64-G1|CORR|0.500|0.5457|0.995|0.9959|1727|1639|1.61|0.41042|
|E64-G1|PHENO|0.500|0.4926|0.001|0.0010|560|1196|47.89|0.00020|
|E64-G1|ANAT|0.500|0.4927|0.001|0.0010|698|343|71.57|0.00014|
|E64-G1|FUNC|0.500|0.4995|0.001|0.0010|666|927|54.95|0.00016|
|E64-G1|CORR|0.500|0.5402|0.995|0.9957|1662|1585|2.90|0.40846|
|E64-G1|PHENO|0.500|0.4589|0.001|0.0009|1307|1116|29.79|0.00025|
|E64-G1|ANAT|0.500|0.4771|0.001|0.0010|395|755|68.66|0.00011|
|E64-G1|FUNC|0.500|0.4568|0.001|0.0009|1483|970|31.71|0.00023|
|E64-G1|CORR|0.500|0.5546|0.995|0.9960|1797|1616|2.15|0.40550|
|E64-G1|PHENO|0.500|0.4878|0.001|0.0010|531|913|59.09|0.00020|
|E64-G1|ANAT|0.500|0.4982|0.001|0.0010|179|372|85.18|0.00007|
|E64-G1|FUNC|0.500|0.4729|0.001|0.0009|1275|137|59.89|0.00017|
|E64-G1|CORR|0.500|0.5353|0.995|0.9957|1715|1624|2.20|0.41718|
|E64-G1|PHENO|0.500|0.4633|0.001|0.0009|1142|1921|15.57|0.00024|
|E64-G1|ANAT|0.500|0.4547|0.001|0.0009|815|814|55.93|0.00014|
|E64-G1|FUNC|0.500|0.4406|0.001|0.0009|1274|1563|25.11|0.00022|
|E64-G1|CORR|0.500|0.5736|0.995|0.9965|1750|1712|1.37|0.37729|
|E64-G1|PHENO|0.500|0.4883|0.001|0.0010|692|1454|38.95|0.00024|
|E64-G1|ANAT|0.500|0.5010|0.001|0.0010|488|1142|55.51|0.00014|
|E64-G1|FUNC|0.500|0.4611|0.001|0.0010|1257|748|43.76|0.00021|
|E64-G1|CORR|0.500|0.5400|0.995|0.9959|1398|1253|1.63|0.37969|
|E64-G1|PHENO|0.500|0.4760|0.001|0.0009|2021|272|32.78|0.00031|
|E64-G1|ANAT|0.500|0.4962|0.001|0.0010|5|570|84.31|0.00004|
|E64-G1|FUNC|0.500|0.4841|0.001|0.0010|418|49|86.93|0.00009|
|E64-G1|CORR|0.500|0.5310|0.995|0.9955|1733|1592|3.87|0.41910|
|E64-G1|PHENO|0.500|0.4957|0.001|0.0010|1106|1716|21.87|0.00030|
|E64-G1|ANAT|0.500|0.4661|0.001|0.0009|1202|686|48.85|0.00019|
|E64-G1|FUNC|0.500|0.4480|0.001|0.0009|1224|1250|28.62|0.00020|
|E64-G1|CORR|0.500|0.5642|0.995|0.9963|1780|1710|1.80|0.38418|
|E64-G1|PHENO|0.500|0.4610|0.001|0.0009|2821|266|10.42|0.00034|
|E64-G1|ANAT|0.500|0.4783|0.001|0.0010|380|247|83.13|0.00010|
|E64-G1|FUNC|0.500|0.4710|0.001|0.0010|996|1031|43.19|0.00020|
|E64-G1|CORR|0.500|0.5458|0.995|0.9958|1654|1532|1.88|0.40947|
|E64-G1|PHENO|0.500|0.4727|0.001|0.0009|2029|290|36.50|0.00031|
|E64-G1|ANAT|0.500|0.4907|0.001|0.0010|614|423|71.47|0.00012|
|E64-G1|FUNC|0.500|0.4749|0.001|0.0010|811|427|63.70|0.00015|
|E64-G1|CORR|0.500|0.5395|0.995|0.9957|1743|1680|3.09|0.41470|
|E64-G2|PHENO|0.500|0.5093|0.001|0.0010|790|1995|21.48|0.00024|
|E64-G2|ANAT|0.500|0.4555|0.001|0.0009|979|1040|44.84|0.00046|
|E64-G2|FUNC|0.500|0.4444|0.001|0.0009|1504|1104|27.19|0.00023|
|E64-G2|CORR|0.500|0.5629|0.992|0.9939|1710|1683|2.11|0.39148|
|E64-G2|PHENO|0.500|0.4971|0.001|0.0010|494|1269|50.00|0.00022|
|E64-G2|ANAT|0.500|0.4996|0.001|0.0010|549|289|77.25|0.00013|
|E64-G2|FUNC|0.500|0.5043|0.001|0.0010|475|830|63.15|0.00013|
|E64-G2|CORR|0.500|0.5346|0.992|0.9930|1333|1283|3.36|0.38604|
|E64-G2|PHENO|0.500|0.4681|0.001|0.0009|1601|805|34.55|0.00028|
|E64-G2|ANAT|0.500|0.4830|0.001|0.0010|367|308|81.62|0.00011|
|E64-G2|FUNC|0.500|0.4701|0.001|0.0009|1377|658|42.47|0.00022|
|E64-G2|CORR|0.500|0.5436|0.992|0.9933|1783|1567|2.25|0.40992|
|E64-G2|PHENO|0.500|0.4377|0.001|0.0008|1606|1242|17.78|0.00025|
|E64-G2|ANAT|0.500|0.4579|0.001|0.0009|878|891|51.80|0.00015|
|E64-G2|FUNC|0.500|0.4356|0.001|0.0009|1562|924|30.56|0.00021|
|E64-G2|CORR|0.500|0.5685|0.992|0.9942|1732|1660|1.42|0.37779|
|E64-G2|PHENO|0.500|0.4643|0.001|0.0009|945|1837|18.58|0.00021|
|E64-G2|ANAT|0.500|0.4490|0.001|0.0009|886|694|57.15|0.00014|
|E64-G2|FUNC|0.500|0.4425|0.001|0.0009|1092|1461|27.47|0.00020|
|E64-G2|CORR|0.500|0.5723|0.992|0.9943|1730|1676|1.33|0.37880|
|E64-G2|PHENO|0.500|0.4959|0.001|0.0010|948|1245|37.82|0.00028|
|E64-G2|ANAT|0.500|0.4968|0.001|0.0010|478|1126|56.16|0.00014|
|E64-G2|FUNC|0.500|0.4599|0.001|0.0009|1242|717|42.45|0.00022|
|E64-G2|CORR|0.500|0.5506|0.992|0.9936|1677|1568|2.14|0.40594|
|E64-G2|PHENO|0.500|0.4643|0.001|0.0009|2261|544|21.78|0.00032|
|E64-G2|ANAT|0.500|0.5064|0.001|0.0010|137|1074|67.08|0.00007|
|E64-G2|FUNC|0.500|0.4668|0.001|0.0010|955|809|50.51|0.00018|
|E64-G2|CORR|0.500|0.5469|0.992|0.9934|1714|1678|2.14|0.40663|
|E64-G2|PHENO|0.500|0.5162|0.001|0.0011|653|1904|28.30|0.00024|
|E64-G2|ANAT|0.500|0.4994|0.001|0.0010|588|661|65.99|0.00015|
|E64-G2|FUNC|0.500|0.4698|0.001|0.0009|988|1076|40.52|0.00019|
|E64-G2|CORR|0.500|0.5489|0.992|0.9935|1740|1667|1.99|0.41134|
|E64-G2|PHENO|0.500|0.4558|0.001|0.0008|2790|491|9.39|0.00032|
|E64-G2|ANAT|0.500|0.4691|0.001|0.0010|599|413|72.74|0.00013|
|E64-G2|FUNC|0.500|0.4645|0.001|0.0009|1358|1099|33.74|0.00025|
|E64-G2|CORR|0.500|0.5514|0.992|0.9936|1588|1430|1.66|0.40059|
|E64-G2|PHENO|0.500|0.4799|0.001|0.0009|2528|278|20.19|0.00036|
|E64-G2|ANAT|0.500|0.4997|0.001|0.0010|241|154|89.44|0.00007|
|E64-G2|FUNC|0.500|0.4846|0.001|0.0010|731|211|73.34|0.00013|
|E64-G2|CORR|0.500|0.5302|0.992|0.9928|1592|1371|4.85|0.40038|
|E64-G3|PHENO|0.500|0.5020|0.001|0.0010|912|2062|16.69|0.00026|
|E64-G3|ANAT|0.500|0.4461|0.001|0.0009|1120|1038|41.49|0.00018|
|E64-G3|FUNC|0.500|0.4346|0.001|0.0008|1581|1132|24.70|0.00023|
|E64-G3|CORR|0.500|0.5671|0.993|0.9948|1529|1523|1.42|0.37686|
|E64-G3|PHENO|0.500|0.4841|0.001|0.0010|742|1161|44.39|0.00023|
|E64-G3|ANAT|0.500|0.4874|0.001|0.0010|775|423|67.43|0.00016|
|E64-G3|FUNC|0.500|0.4909|0.001|0.0010|781|1071|48.09|0.00018|
|E64-G3|CORR|0.500|0.5445|0.993|0.9941|1667|1556|2.98|0.40406|
|E64-G3|PHENO|0.500|0.4598|0.001|0.0009|1405|1104|28.31|0.00025|
|E64-G3|ANAT|0.500|0.4757|0.001|0.0010|394|770|68.25|0.00011|
|E64-G3|FUNC|0.500|0.4605|0.001|0.0009|1541|899|31.31|0.00023|
|E64-G3|CORR|0.500|0.5523|0.993|0.9944|1634|1507|2.09|0.39961|
|E64-G3|PHENO|0.500|0.4814|0.001|0.0010|839|968|48.25|0.00022|
|E64-G3|ANAT|0.500|0.4927|0.001|0.0010|412|460|76.36|0.00011|
|E64-G3|FUNC|0.500|0.4654|0.001|0.0009|1378|315|52.58|0.00019|
|E64-G3|CORR|0.500|0.5436|0.993|0.9942|1690|1517|1.78|0.40789|
|E64-G3|PHENO|0.500|0.4758|0.001|0.0009|1270|1757|13.86|0.00025|
|E64-G3|ANAT|0.500|0.4569|0.001|0.0009|780|680|60.08|0.00014|
|E64-G3|FUNC|0.500|0.4478|0.001|0.0009|1025|1451|29.98|0.00019|
|E64-G3|CORR|0.500|0.5699|0.993|0.9949|1715|1554|1.71|0.38118|
|E64-G3|PHENO|0.500|0.4901|0.001|0.0010|794|1369|36.42|0.00022|
|E64-G3|ANAT|0.500|0.4949|0.001|0.0010|491|1135|55.85|0.00014|
|E64-G3|FUNC|0.500|0.4596|0.001|0.0010|1258|814|41.12|0.00023|
|E64-G3|CORR|0.500|0.5427|0.993|0.9942|1655|1394|2.15|0.40417|
|E64-G3|PHENO|0.500|0.4741|0.001|0.0009|2069|281|31.67|0.00031|
|E64-G3|ANAT|0.500|0.4977|0.001|0.0010|11|599|83.46|0.00004|
|E64-G3|FUNC|0.500|0.4833|0.001|0.0010|609|74|81.15|0.00012|
|E64-G3|CORR|0.500|0.5324|0.993|0.9938|1690|1592|3.27|0.41540|
|E64-G3|PHENO|0.500|0.5168|0.001|0.0011|489|1617|38.04|0.00021|
|E64-G3|ANAT|0.500|0.5038|0.001|0.0010|418|636|71.20|0.00012|
|E64-G3|FUNC|0.500|0.4716|0.001|0.0009|805|1096|45.42|0.00017|
|E64-G3|CORR|0.500|0.5450|0.993|0.9942|1585|1499|1.34|0.40675|
|E64-G3|PHENO|0.500|0.4645|0.001|0.0009|2940|238|9.48|0.00036|
|E64-G3|ANAT|0.500|0.4860|0.001|0.0010|324|153|87.05|0.00009|
|E64-G3|FUNC|0.500|0.4805|0.001|0.0010|781|847|54.27|0.00015|
|E64-G3|CORR|0.500|0.5406|0.993|0.9940|1747|1592|2.00|0.41849|
|E64-G3|PHENO|0.500|0.4627|0.001|0.0009|1967|580|29.01|0.00030|
|E64-G3|ANAT|0.500|0.4709|0.001|0.0010|1028|538|57.28|0.00018|
|E64-G3|FUNC|0.500|0.4574|0.001|0.0009|1141|852|43.65|0.00017|
|E64-G3|CORR|0.500|0.5493|0.993|0.9943|1771|1742|2.79|0.39765|

### Fixed-checkpoint interventions

Each effect is full model minus intervention. Permutation is fixed independently of test labels.

|Group|Intervention|ΔACC pp|Hard prediction flips|Mean absolute Δprobability|
|---|---|---:|---:|---:|
|E64-G1|all_cadr_off|+0.000|0|0.001700|
|E64-G1|corr_cadr_off|+0.000|0|0.001700|
|E64-G1|direct_off|+0.000|0|0.001690|
|E64-G1|score_permutation|+0.000|0|0.002866|
|E64-G1|all_cadr_off|+0.000|0|0.001905|
|E64-G1|corr_cadr_off|+0.000|0|0.001905|
|E64-G1|direct_off|+0.000|0|0.001905|
|E64-G1|score_permutation|+0.000|0|0.003198|
|E64-G1|all_cadr_off|+0.000|0|0.002800|
|E64-G1|corr_cadr_off|+0.000|0|0.002801|
|E64-G1|direct_off|+0.000|0|0.002799|
|E64-G1|score_permutation|+0.000|0|0.002702|
|E64-G1|all_cadr_off|+0.000|0|0.001578|
|E64-G1|corr_cadr_off|+0.000|0|0.001578|
|E64-G1|direct_off|+0.000|0|0.001578|
|E64-G1|score_permutation|+0.000|0|0.002405|
|E64-G1|all_cadr_off|+0.000|0|0.002090|
|E64-G1|corr_cadr_off|+0.000|0|0.002090|
|E64-G1|direct_off|+0.000|0|0.002091|
|E64-G1|score_permutation|+0.000|0|0.002642|
|E64-G1|all_cadr_off|+0.000|0|0.001027|
|E64-G1|corr_cadr_off|+0.000|0|0.001027|
|E64-G1|direct_off|+0.000|0|0.001027|
|E64-G1|score_permutation|+0.000|0|0.001537|
|E64-G1|all_cadr_off|+0.000|0|0.001251|
|E64-G1|corr_cadr_off|+0.000|0|0.001251|
|E64-G1|direct_off|+0.000|0|0.001250|
|E64-G1|score_permutation|+0.000|0|0.002170|
|E64-G1|all_cadr_off|+0.000|0|0.001493|
|E64-G1|corr_cadr_off|+0.000|0|0.001495|
|E64-G1|direct_off|+0.000|0|0.001494|
|E64-G1|score_permutation|+0.000|0|0.002149|
|E64-G1|all_cadr_off|+0.000|0|0.001361|
|E64-G1|corr_cadr_off|+0.000|0|0.001362|
|E64-G1|direct_off|+0.000|0|0.001361|
|E64-G1|score_permutation|+0.000|0|0.002171|
|E64-G1|all_cadr_off|+0.000|0|0.001307|
|E64-G1|corr_cadr_off|+0.000|0|0.001308|
|E64-G1|direct_off|+0.000|0|0.001308|
|E64-G1|score_permutation|+0.000|0|0.001871|
|E64-G2|all_cadr_off|+0.000|0|0.002117|
|E64-G2|corr_cadr_off|+0.000|0|0.002113|
|E64-G2|direct_off|+0.000|0|0.002113|
|E64-G2|score_permutation|+0.000|0|0.002648|
|E64-G2|all_cadr_off|+0.000|0|0.001129|
|E64-G2|corr_cadr_off|+0.000|0|0.001129|
|E64-G2|direct_off|+0.000|0|0.001129|
|E64-G2|score_permutation|+0.000|0|0.002204|
|E64-G2|all_cadr_off|+0.000|0|0.002695|
|E64-G2|corr_cadr_off|+0.000|0|0.002696|
|E64-G2|direct_off|+0.000|0|0.002696|
|E64-G2|score_permutation|+0.000|0|0.003009|
|E64-G2|all_cadr_off|+1.149|1|0.001782|
|E64-G2|corr_cadr_off|+1.149|1|0.001784|
|E64-G2|direct_off|+1.149|1|0.001783|
|E64-G2|score_permutation|+2.299|2|0.002848|
|E64-G2|all_cadr_off|+0.000|0|0.001246|
|E64-G2|corr_cadr_off|+0.000|0|0.001247|
|E64-G2|direct_off|+0.000|0|0.001253|
|E64-G2|score_permutation|+0.000|0|0.001973|
|E64-G2|all_cadr_off|+0.000|0|0.001241|
|E64-G2|corr_cadr_off|+0.000|0|0.001241|
|E64-G2|direct_off|+0.000|0|0.001245|
|E64-G2|score_permutation|+0.000|0|0.002067|
|E64-G2|all_cadr_off|+0.000|0|0.001889|
|E64-G2|corr_cadr_off|+0.000|0|0.001889|
|E64-G2|direct_off|+0.000|0|0.001889|
|E64-G2|score_permutation|+0.000|0|0.002271|
|E64-G2|all_cadr_off|+0.000|0|0.001488|
|E64-G2|corr_cadr_off|+0.000|0|0.001488|
|E64-G2|direct_off|+0.000|0|0.001488|
|E64-G2|score_permutation|+0.000|0|0.002129|
|E64-G2|all_cadr_off|+0.000|0|0.001278|
|E64-G2|corr_cadr_off|+0.000|0|0.001279|
|E64-G2|direct_off|+0.000|0|0.001278|
|E64-G2|score_permutation|+0.000|0|0.001933|
|E64-G2|all_cadr_off|+0.000|0|0.001269|
|E64-G2|corr_cadr_off|+0.000|0|0.001270|
|E64-G2|direct_off|+0.000|0|0.001269|
|E64-G2|score_permutation|+0.000|0|0.002212|
|E64-G3|all_cadr_off|+0.000|0|0.000507|
|E64-G3|corr_cadr_off|+0.000|0|0.000507|
|E64-G3|direct_off|+0.000|0|0.000506|
|E64-G3|score_permutation|+0.000|0|0.000781|
|E64-G3|all_cadr_off|+0.000|0|0.001619|
|E64-G3|corr_cadr_off|+0.000|0|0.001619|
|E64-G3|direct_off|+0.000|0|0.001619|
|E64-G3|score_permutation|+0.000|0|0.002275|
|E64-G3|all_cadr_off|+1.149|1|0.002436|
|E64-G3|corr_cadr_off|+1.149|1|0.002436|
|E64-G3|direct_off|+1.149|1|0.002436|
|E64-G3|score_permutation|+0.000|0|0.002900|
|E64-G3|all_cadr_off|+0.000|0|0.001787|
|E64-G3|corr_cadr_off|+0.000|0|0.001787|
|E64-G3|direct_off|+0.000|0|0.001787|
|E64-G3|score_permutation|+0.000|0|0.002352|
|E64-G3|all_cadr_off|+1.149|1|0.001843|
|E64-G3|corr_cadr_off|+1.149|1|0.001843|
|E64-G3|direct_off|+1.149|1|0.001843|
|E64-G3|score_permutation|+1.149|1|0.002642|
|E64-G3|all_cadr_off|+0.000|0|0.001064|
|E64-G3|corr_cadr_off|+0.000|0|0.001064|
|E64-G3|direct_off|+0.000|0|0.001064|
|E64-G3|score_permutation|+0.000|0|0.001270|
|E64-G3|all_cadr_off|+0.000|0|0.001278|
|E64-G3|corr_cadr_off|+0.000|0|0.001278|
|E64-G3|direct_off|+0.000|0|0.001278|
|E64-G3|score_permutation|+0.000|0|0.001869|
|E64-G3|all_cadr_off|+0.000|0|0.001549|
|E64-G3|corr_cadr_off|+0.000|0|0.001550|
|E64-G3|direct_off|+0.000|0|0.001550|
|E64-G3|score_permutation|+1.149|1|0.002538|
|E64-G3|all_cadr_off|+1.149|1|0.001641|
|E64-G3|corr_cadr_off|+1.149|1|0.001642|
|E64-G3|direct_off|+1.149|1|0.001641|
|E64-G3|score_permutation|+0.000|0|0.002330|
|E64-G3|all_cadr_off|+0.000|0|0.001000|
|E64-G3|corr_cadr_off|+0.000|0|0.001001|
|E64-G3|direct_off|+0.000|0|0.001000|
|E64-G3|score_permutation|+1.149|1|0.001779|

## Scientific judgment

These groups test CORR fusion-gate initialization sensitivity, not a new architecture. They hold edge gates at 0.5 and non-CORR fusion gates at 0.001 while varying only the CORR start at 0.995, 0.992, and 0.993. E62-E2 (0.99) and E63-G1 (0.999) are read-only reference points.

E64-G1 and E64-G3 tie for highest ACC at 91.4995%; E64-G1 has the highest Hard-AUC at 91.2645% by 0.0102 pp over G3, while G3 has the highest Probability-AUC at 93.5647%. All three Probability-AUC means exceed E62-E2's 93.2010%, by 0.301–0.364 pp, but all paired descriptive 95% intervals include zero. Relative to E62-E2, G1 is -0.345 pp ACC, -0.418 pp Hard-AUC, +0.340 pp Probability-AUC, -1.256 pp SEN, and +0.421 pp SPE. G2 is -0.575/-0.683/+0.301 pp on ACC/Hard-AUC/Probability-AUC and -2.226 pp SEN. G3 is -0.345/-0.428/+0.364 pp on ACC/Hard-AUC/Probability-AUC and -1.494 pp SEN. None improves the full diagnostic profile; all remain below the historical 53.3 group on ACC, Hard-AUC, and Probability-AUC.

The result does not show a monotonic response to CORR initialization: G2 at 0.992 has the lowest ACC and Hard-AUC, while G1 at 0.995 and G3 at 0.993 tie on ACC. Compared with E63-G1's 0.999 initialization, the lower starts yield slightly higher mean Probability-AUC and SPE but lower SEN; paired intervals include zero. Mean final CORR gates are 0.99589, 0.99355, and 0.99430 for G1/G2/G3, respectively; non-CORR fusion gates remain near 0.00093–0.00097. Thus the learned CORR gate moves slightly upward from each start, but the initialization ordering remains.

The CORR weighted CADR message is about 0.40% of the input norm. Within it, the direct-difference term is 216–22,463 times the positive-plus-negative GCN message norm across these runs. Fixed-checkpoint removal of all CADR, CORR CADR, or direct differences changes 0/1/3 hard predictions across ten folds in G1/G2/G3; mean absolute probability changes are about 0.0015–0.0017. Score permutation changes 0/2/3 predictions. The larger initial CORR fusion values therefore do not produce material measured dependence of hard predictions on CADR propagation. All 30 pairwise group/fold RNG comparisons matched exactly over 11,055 shared epochs.

**Decision: no evidence of a reliable diagnostic gain.** The initialization sweep is useful as a sensitivity check, but it does not establish that a CORR fusion start between 0.992 and 0.995 improves diagnosis over E62-E2 or the historical 53.3 base. Results remain exploratory because this is one seed with overlapping folds, historical `val=test` checkpoint selection, pre-CV preprocessing, and a transductive graph.

## Files

- Experiment root: `/model/spromptgl/experiment/experiment64_corr_fusion_gate_sensitivity`
- Task manifest: `manifest.json`
- Fold metrics: `results/fold_metrics.csv`
- Paired comparisons: `results/paired_comparisons.csv`
- Gate diagnostics: `results/gate_diagnostics.csv`
- Independent checkpoint audit: `results/checkpoint_audit.csv`
