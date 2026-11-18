# Experiment 81.7 PSRR report

Engineering audit for the completed groups is recorded in `results/audit_status.json`. Current group status is `NO_GO_PSRR_BASELINE`.

Metrics are percentages. Hard-AUC is the historical hard-prediction balanced accuracy. Prob-AUC uses `P(ASD=0)` against ASD label 0. Fold summaries use population SD. Paired intervals are descriptive and are not significance intervals for independent patients.

## Six Metrics

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.1891±2.7648|92.0294±2.7255|93.4946±2.9195|90.0671±3.8862|93.9917±4.7320|91.4590±2.9013|
|G1|91.8443±2.9346|91.6628±2.9176|93.7843±2.8808|89.3293±4.0366|93.9963±4.4083|91.0394±3.1531|

## Excluding Fold 2

|Group|ACC|Hard-AUC/BA|Prob-AUC|SEN-ASD|SPE-TD|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|
|DARG|92.2153±2.9131|92.0481±2.8723|93.6142±3.0541|90.0474±4.0959|94.0487±4.9847|91.4703±3.0580|
|G1|91.9598±3.0717|91.7763±3.0544|93.9007±3.0142|89.4986±4.2211|94.0539±4.6432|91.1549±3.3036|

## Paired Comparison Summary

|Comparison|Subset|Metric|Mean Δ pp|Descriptive interval|W/T/L|
|---|---|---|---:|---:|---:|
|G1_vs_DARG|all10|ACC|-0.3448|[-0.9012, +0.2116]|1/6/3|
|G1_vs_DARG|all10|Hard-AUC/BA|-0.3666|[-0.9580, +0.2248]|1/5/4|
|G1_vs_DARG|exclude_fold2|ACC|-0.2554|[-0.8454, +0.3346]|1/6/2|
|G1_vs_DARG|exclude_fold2|Hard-AUC/BA|-0.2718|[-0.8991, +0.3555]|1/5/3|
|G1_vs_E81.6.2-G1|all10|ACC|-0.2299|[-0.8510, +0.3912]|2/5/3|
|G1_vs_E81.6.2-G1|all10|Hard-AUC/BA|-0.2602|[-0.8955, +0.3751]|3/3/4|
|G1_vs_E81.6.2-G1|exclude_fold2|ACC|-0.1277|[-0.7847, +0.5293]|2/5/2|
|G1_vs_E81.6.2-G1|exclude_fold2|Hard-AUC/BA|-0.1536|[-0.8243, +0.5171]|3/3/3|
|G1_vs_E81.6.2-G3|all10|ACC|+0.0000|[-0.4506, +0.4506]|2/6/2|
|G1_vs_E81.6.2-G3|all10|Hard-AUC/BA|+0.0217|[-0.4314, +0.4747]|3/5/2|
|G1_vs_E81.6.2-G3|exclude_fold2|ACC|+0.0000|[-0.5006, +0.5006]|2/5/2|
|G1_vs_E81.6.2-G3|exclude_fold2|Hard-AUC/BA|+0.0241|[-0.4793, +0.5274]|3/4/2|
|G1_vs_E81.6.3-G1|all10|ACC|+0.0000|[-0.3186, +0.3186]|1/8/1|
|G1_vs_E81.6.3-G1|all10|Hard-AUC/BA|+0.0426|[-0.3133, +0.3985]|3/5/2|
|G1_vs_E81.6.3-G1|exclude_fold2|ACC|+0.0000|[-0.3540, +0.3540]|1/7/1|
|G1_vs_E81.6.3-G1|exclude_fold2|Hard-AUC/BA|+0.0473|[-0.3480, +0.4427]|3/4/2|
|G1_vs_E81.6.1-G1|all10|ACC|-0.0013|[-0.4506, +0.4480]|2/6/2|
|G1_vs_E81.6.1-G1|all10|Hard-AUC/BA|+0.0637|[-0.3954, +0.5229]|3/5/2|
|G1_vs_E81.6.1-G1|exclude_fold2|ACC|-0.0015|[-0.5007, +0.4978]|2/5/2|
|G1_vs_E81.6.1-G1|exclude_fold2|Hard-AUC/BA|+0.0708|[-0.4391, +0.5808]|3/4/2|

## Mechanism and Corrections

All corrections count test-fold rows under the historical Fold2 overlap protocol. A is the E74.4-G2 DARG Best, B is the current PSRR Best epoch with its relational residual disabled, and C is the full PSRR output.

|Stage|Corrected|Harmed|Net|
|---|---:|---:|---:|
|A_to_B|2|9|-7|
|B_to_C|4|0|4|
|A_to_C|2|5|-3|

|Group|Fold|Real edges/row|Null positive|ASD-TD support disagreement|Consensus Jaccard|TD beta entropy|ASD beta entropy|Residual ratio mean/max|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|G1|1|5.751065254211426|0.26100853085517883|0.06112689524888992|0.6331937313079834|1.3826711177825928|1.378185749053955|0.06276391446590424/0.08340027183294296|
|G1|2|11.10380744934082|0.2399425357580185|0.09094827622175217|0.7495958209037781|1.3806030750274658|1.3856581449508667|0.07062206417322159/0.09210407733917236|
|G1|3|9.883261680603027|0.06896551698446274|0.05771072953939438|0.6947253346443176|1.3838335275650024|1.384535789489746|0.0782458707690239/0.08957278728485107|
|G1|4|13.09375|0.42133620381355286|0.12964560091495514|0.8772216439247131|1.3835723400115967|1.3858628273010254|0.08915655314922333/0.09785858541727066|
|G1|5|9.736709594726562|0.3541666567325592|0.08740421384572983|0.7146438956260681|1.3765751123428345|1.3856265544891357|0.09073501080274582/0.09832526743412018|
|G1|6|11.984554290771484|0.09446839243173599|0.10138888657093048|0.8146267533302307|1.3774269819259644|1.3841444253921509|0.08688610792160034/0.09474720805883408|
|G1|7|6.076508522033691|0.445761501789093|0.07121647149324417|0.5003871917724609|1.386085867881775|1.3853636980056763|0.08355031907558441/0.09623405337333679|
|G1|8|12.663793563842773|0.6591235399246216|0.10411877185106277|0.8452723622322083|1.3847264051437378|1.385935664176941|0.08849146217107773/0.09424196928739548|
|G1|9|11.08872127532959|0.642241358757019|0.11748084425926208|0.7447329759597778|1.3847932815551758|1.3858380317687988|0.06423760205507278/0.08590009808540344|
|G1|10|10.279812812805176|0.28663793206214905|0.09535440802574158|0.7390593886375427|1.3830279111862183|1.381600260734558|0.08404725790023804/0.09462160617113113|

## End-to-End Gradient Path

Norms summarize the epoch-12 classification-loss VJP through the learned residual, measured separately at the CAM and DARG parameters.

|Group|Folds|PSRR residual VJP|CAM VJP|DARG VJP|
|---|---:|---:|---:|---:|
|G1|10|0.010033|1.09375e-05|6.84965e-06|

## Audit Decisions

- Best independent inference and six-metric recomputation: PASS for G1 and the corresponding status for conditional groups.
- Latest full restore and next-step replay: PASS for G1 and the corresponding status for conditional groups.
- G1 vs DARG pre-registered baseline gate: False.
- End-to-end gradient path through PSRR, CAM, and DARG: PASS.
- Pairwise increment G1 vs independently trained unary G2: CONDITIONAL_NOT_RUN.
- Medical grouping increment G1 vs scrambled G3: CONDITIONAL_NOT_RUN.
- Conditional group execution: G2 `NOT_EXECUTED_CONDITIONAL`, G3 `NOT_EXECUTED_CONDITIONAL`.
- Current research decision: `NO_GO_PSRR_BASELINE`.
- Frozen training source SHA256: `fdee19d65ff982e6962f2ded91928130131a9c3d3af8ffef51f8327d3517698d`.
- Audit script SHA256 at training freeze/current audit: `b7e8a884c3460e1813e4e674e5866291cd490721990e986839f9419374aa496e` / `75aee0026edc73f39926c367c6c08218e2f507bcbc3e0f57a2df1790e066d0f4`.
- Post-training audit-only corrections cover group-swap routing, DARG metric-field normalization, RNG epoch reconstruction, optimizer-state deep copies during replay, and percentage-point reporting. The frozen training implementation and checkpoints were not modified.

Fixed Prompt geometry constrains the parameterization but does not validate clinical meaning. The 16 local slots are latent feature locations, not anatomical ROIs. Fold2 training/test overlap, test=validation model selection, pre-CV processing/RFE and the transductive patient graph remain inherited limitations. A single seed and overlapping CV do not establish independent clinical generalization.

Files: `results/fold_metrics.csv`, `results/paired_comparisons.csv`, `results/mechanism.csv`, `results/audit_status.json`, `results/preflight.json`, and `manifest.json`.
