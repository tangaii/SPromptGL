# Experiment 81.6.3 SA-PRCL-CGE independent audit

Status: 10/10 requested tasks independently audited. Audit status is `PASS`.

All metrics are percentages with 10-fold mean plus population SD. Paired intervals are descriptive intervals and are not significance tests.

|Model|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|SA-PRCL-CGE-G1-Medical|91.8443±2.6508|91.6202±2.5587|93.7655±2.9619|88.8232±2.8192|94.4172±4.6789|91.0211±2.7240|
|DARG|92.1891±2.7648|92.0294±2.7255|93.4946±2.9195|90.0671±3.8862|93.9917±4.7320|91.4590±2.9013|
|E81.6.1-G1|91.8456±2.5480|91.5990±2.5788|93.8184±2.8718|88.3415±4.0168|94.8566±3.2307|90.9253±2.8120|
|E81.6.2-G1|92.0742±2.6000|91.9230±2.5481|93.2339±2.7999|90.0671±3.3693|93.7789±4.4992|91.3480±2.6928|
|E81.6.2-G3|91.8443±2.6004|91.6411±2.5939|93.3385±2.8969|89.0732±3.9177|94.2091±4.1186|91.0063±2.8194|

## Excluding Fold 2

|Model|Metric|9-fold mean±population SD|
|---|---|---:|
|SA-PRCL-CGE-G1-Medical|ACC|91.9598±2.7702|
|SA-PRCL-CGE-G1-Medical|Hard-AUC/BA|91.7289±2.6751|
|SA-PRCL-CGE-G1-Medical|Prob-AUC|93.8504±3.1106|
|SA-PRCL-CGE-G1-Medical|SEN-ASD|88.9363±2.9500|
|SA-PRCL-CGE-G1-Medical|SPE-TD|94.5215±4.9209|
|SA-PRCL-CGE-G1-Medical|ASD-F1|91.1346±2.8488|
|DARG|ACC|92.2153±2.9131|
|DARG|Hard-AUC/BA|92.0481±2.8723|
|DARG|Prob-AUC|93.6142±3.0541|
|DARG|SEN-ASD|90.0474±4.0959|
|DARG|SPE-TD|94.0487±4.9847|
|DARG|ASD-F1|91.4703±3.0580|
|E81.6.1-G1|ACC|91.9613±2.6608|
|E81.6.1-G1|Hard-AUC/BA|91.7054±2.6974|
|E81.6.1-G1|Prob-AUC|93.9858±2.9805|
|E81.6.1-G1|SEN-ASD|88.4011±4.2299|
|E81.6.1-G1|SPE-TD|95.0098±3.3709|
|E81.6.1-G1|ASD-F1|91.0282±2.9462|
|E81.6.2-G1|ACC|92.0875±2.7403|
|E81.6.2-G1|Hard-AUC/BA|91.9299±2.6858|
|E81.6.2-G1|Prob-AUC|93.3246±2.9374|
|E81.6.2-G1|SEN-ASD|90.0474±3.5510|
|E81.6.2-G1|SPE-TD|93.8123±4.7414|
|E81.6.2-G1|ASD-F1|91.3469±2.8385|
|E81.6.2-G3|ACC|91.9598±2.7166|
|E81.6.2-G3|Hard-AUC/BA|91.7522±2.7115|
|E81.6.2-G3|Prob-AUC|93.4467±3.0344|
|E81.6.2-G3|SEN-ASD|89.2141±4.1055|
|E81.6.2-G3|SPE-TD|94.2903±4.3338|
|E81.6.2-G3|ASD-F1|91.1181±2.9508|

## Paired comparisons

|Comparison|Subset|Metric|Mean diff|Descriptive interval|W/T/L|
|---|---|---|---:|---:|---:|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|ACC|-0.3448|[-0.8010, +0.1113]|1/5/4|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|Hard-AUC/BA|-0.4092|[-0.9126, +0.0943]|1/4/5|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|Prob-AUC|+0.2709|[-0.0764, +0.6182]|8/0/2|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|SEN-ASD|-1.2439|[-2.4909, +0.0031]|1/4/5|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|SPE-TD|+0.4255|[-0.1020, +0.9530]|2/8/0|
|SA-PRCL-CGE-G1-Medical_vs_DARG|all10|ASD-F1|-0.4379|[-1.0043, +0.1285]|1/4/5|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|ACC|-0.2554|[-0.7274, +0.2166]|1/5/3|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.3191|[-0.8467, +0.2084]|1/4/4|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|Prob-AUC|+0.2362|[-0.1430, +0.6153]|7/0/2|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|SEN-ASD|-1.1111|[-2.4692, +0.2470]|1/4/4|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|SPE-TD|+0.4728|[-0.1051, +1.0507]|2/7/0|
|SA-PRCL-CGE-G1-Medical_vs_DARG|exclude_fold2|ASD-F1|-0.3357|[-0.9285, +0.2571]|1/4/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|ACC|-0.0013|[-0.4506, +0.4480]|2/6/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|Hard-AUC/BA|+0.0211|[-0.4474, +0.4897]|4/3/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|Prob-AUC|-0.0529|[-0.5159, +0.4100]|6/0/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|SEN-ASD|+0.4817|[-1.4284, +2.3918]|3/5/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|SPE-TD|-0.4394|[-2.1059, +1.2271]|2/5/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|all10|ASD-F1|+0.0958|[-0.4416, +0.6331]|4/3/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|ACC|-0.0015|[-0.5007, +0.4978]|2/5/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|Hard-AUC/BA|+0.0235|[-0.4971, +0.5441]|4/2/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|Prob-AUC|-0.1354|[-0.6207, +0.3500]|5/0/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|SEN-ASD|+0.5352|[-1.5842, +2.6546]|3/4/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|SPE-TD|-0.4882|[-2.3371, +1.3607]|2/4/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.1-G1|exclude_fold2|ASD-F1|+0.1064|[-0.4902, +0.7031]|4/2/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|ACC|-0.2299|[-0.6573, +0.1976]|1/6/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|Hard-AUC/BA|-0.3028|[-0.7567, +0.1510]|1/5/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|Prob-AUC|+0.5315|[+0.1396, +0.9235]|9/0/1|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|SEN-ASD|-1.2439|[-2.2806, -0.2072]|0/6/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|SPE-TD|+0.6383|[+0.0340, +1.2426]|3/7/0|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|all10|ASD-F1|-0.3269|[-0.8323, +0.1785]|1/5/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|ACC|-0.1277|[-0.5532, +0.2977]|1/6/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|Hard-AUC/BA|-0.2009|[-0.6592, +0.2573]|1/5/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|Prob-AUC|+0.5258|[+0.0905, +0.9611]|8/0/1|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|SEN-ASD|-1.1111|[-2.2298, +0.0076]|0/6/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|SPE-TD|+0.7092|[+0.0539, +1.3645]|3/6/0|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G1|exclude_fold2|ASD-F1|-0.2123|[-0.7215, +0.2969]|1/5/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|ACC|+0.0000|[-0.4506, +0.4506]|2/6/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|Hard-AUC/BA|-0.0209|[-0.4822, +0.4403]|3/3/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|Prob-AUC|+0.4270|[+0.0974, +0.7566]|8/0/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|SEN-ASD|-0.2500|[-1.3347, +0.8347]|2/5/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|SPE-TD|+0.2081|[-0.7195, +1.1358]|3/5/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|all10|ASD-F1|+0.0148|[-0.4798, +0.5095]|3/3/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|ACC|+0.0000|[-0.5006, +0.5006]|2/5/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|Hard-AUC/BA|-0.0233|[-0.5357, +0.4892]|3/2/4|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|Prob-AUC|+0.4038|[+0.0407, +0.7668]|7/0/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|SEN-ASD|-0.2778|[-1.4816, +0.9260]|2/4/3|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|SPE-TD|+0.2313|[-0.7983, +1.2609]|3/4/2|
|SA-PRCL-CGE-G1-Medical_vs_E81.6.2-G3|exclude_fold2|ASD-F1|+0.0165|[-0.5331, +0.5661]|3/2/4|

## Same-epoch correction

A is historical DARG Best, B is the current selected epoch with the residual disabled, and C is the complete CGE output.

|Group|Subset|Stage|Corrected|Harmed|Net|
|---|---|---|---:|---:|---:|
|SA-PRCL-CGE-G1-Medical|all10|A_to_B|4|9|-5|
|SA-PRCL-CGE-G1-Medical|all10|B_to_C|2|0|2|
|SA-PRCL-CGE-G1-Medical|all10|A_to_C|4|7|-3|
|SA-PRCL-CGE-G1-Medical|exclude_fold2|A_to_B|3|7|-4|
|SA-PRCL-CGE-G1-Medical|exclude_fold2|B_to_C|2|0|2|
|SA-PRCL-CGE-G1-Medical|exclude_fold2|A_to_C|3|5|-2|

## Audit and research decisions

- Best independent metric recomputations: 10/10.
- Latest reloads: 10/10.
- Historical starts and same-fold SA initialization: 10/10 and 10/10.
- Controlled end-to-end gradient mechanism: `PASS`.
- Public RNG and Dropout replay: `PASS`.
- Fold 2 protection: `PASS`.
- G2 continuation: `NOT_EXECUTED_CONDITIONAL`.
- Research decision: `E2E_MECHANISM_PASS_PERFORMANCE_NO_GO`.

Prompt-off and Null-only disable the whole residual branch and cannot alone establish medical semantic validity. G2 keeps the same original medical text and changes only the fixed grouping indices. Fold 2 overlap, test=validation, global pre-CV processing, RFE and transductive graph construction remain historical protocol limitations.

Source SHA256: `dba2b1cc1f23f7f6fd2844bd6f337f4e1570a61f05e052bb7e3989ca2b0b7e0f`. Prompt cache SHA256: `7bfd1d6e11bb9983e9896e682b5a8c40bf606c919dfd38bae368436955f83aff`.
