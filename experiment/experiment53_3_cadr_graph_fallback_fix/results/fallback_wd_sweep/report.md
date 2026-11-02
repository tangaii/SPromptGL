# Experiment 53.3 CADR/GFR weight-decay sweep

Four groups were trained from the fallback-fixed Experiment 53.3 implementation. The original graph fallback, relation cache, initialization, RNG handling, CAM WD, and training protocol were held fixed. Only CADR and GFR optimizer weight decay varied. All 40 checkpoints were independently loaded and audited.

|组|CADR WD|GFR WD|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|来源|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|E1_CADR_WD0001_GRAPH_FALLBACK_FIX|0.001|0.11|91.7294 ± 2.9950|91.6046 ± 2.9574|93.5681 ± 3.0886|93.1360 ± 4.3174|90.0732 ± 3.5288|91.6046 ± 2.9574|91.0065 ± 3.1369|53.3 historical read-only reference|
|CADR_WD_0p0001_GFR_WD_0p11|0.0001|0.11|91.6144 ± 3.0062|91.4242 ± 3.0003|93.3971 ± 2.9957|94.0009 ± 3.4506|88.8476 ± 3.5140|91.4242 ± 3.0003|90.7623 ± 3.2518|new_training|
|CADR_WD_0p001_GFR_WD_0p20|0.001|0.2|91.2709 ± 2.9237|91.0577 ± 3.0159|93.0486 ± 3.0915|94.0056 ± 2.5201|88.1098 ± 4.8927|91.0577 ± 3.0159|90.2957 ± 3.3285|new_training|
|CADR_WD_0p01_GFR_WD_0p11|0.01|0.11|92.0742 ± 2.8427|91.8653 ± 2.8348|93.6684 ± 3.2619|94.6392 ± 3.5168|89.0915 ± 3.5263|91.8653 ± 2.8348|91.2470 ± 3.0658|new_training|
|CADR_WD_0p01_GFR_WD_0p20|0.01|0.2|91.6144 ± 2.8249|91.4554 ± 2.8484|93.7277 ± 2.7780|93.5754 ± 3.3387|89.3354 ± 4.0041|91.4554 ± 2.8484|90.7898 ± 3.0956|new_training|

## Paired differences versus Experiment 53.3 baseline

|组|指标|均值差(pp)|胜/平/负|
|---|---|---:|---:|
|CADR_WD_0p0001_GFR_WD_0p11|ACC|-0.1149|2/6/2|
|CADR_WD_0p0001_GFR_WD_0p11|Hard-AUC|-0.1803|2/3/5|
|CADR_WD_0p0001_GFR_WD_0p11|Probability-AUC|-0.1710|5/0/5|
|CADR_WD_0p0001_GFR_WD_0p11|SPE-TD|+0.8649|5/4/1|
|CADR_WD_0p0001_GFR_WD_0p11|SEN-ASD|-1.2256|1/5/4|
|CADR_WD_0p001_GFR_WD_0p20|ACC|-0.4585|2/3/5|
|CADR_WD_0p001_GFR_WD_0p20|Hard-AUC|-0.5469|3/1/6|
|CADR_WD_0p001_GFR_WD_0p20|Probability-AUC|-0.5195|3/0/7|
|CADR_WD_0p001_GFR_WD_0p20|SPE-TD|+0.8696|5/2/3|
|CADR_WD_0p001_GFR_WD_0p20|SEN-ASD|-1.9634|1/4/5|
|CADR_WD_0p01_GFR_WD_0p11|ACC|+0.3448|4/5/1|
|CADR_WD_0p01_GFR_WD_0p11|Hard-AUC|+0.2608|4/3/3|
|CADR_WD_0p01_GFR_WD_0p11|Probability-AUC|+0.1003|6/0/4|
|CADR_WD_0p01_GFR_WD_0p11|SPE-TD|+1.5032|6/4/0|
|CADR_WD_0p01_GFR_WD_0p11|SEN-ASD|-0.9817|0/7/3|
|CADR_WD_0p01_GFR_WD_0p20|ACC|-0.1149|1/7/2|
|CADR_WD_0p01_GFR_WD_0p20|Hard-AUC|-0.1492|1/3/6|
|CADR_WD_0p01_GFR_WD_0p20|Probability-AUC|+0.1595|4/0/6|
|CADR_WD_0p01_GFR_WD_0p20|SPE-TD|+0.4394|5/3/2|
|CADR_WD_0p01_GFR_WD_0p20|SEN-ASD|-0.7378|1/5/4|

The historical `val=test` checkpoint selection, pre-CV feature preprocessing, and 871-node transductive graph remain in effect; these are protocol-matched exploratory results, not unbiased external generalization estimates.
