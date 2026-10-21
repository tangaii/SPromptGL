# Experiment 32 Results: DCER

DCER replaces the full original PEN at the original PLN location. The original GFR, CAM, OutputLayer, graph loss, two Adam optimizers, train/test split protocol, and validation-equals-test checkpoint rule remain intact. B2 is the newly trained internal full-SPromptGL baseline; paper-reported values 91.73% after GFR, 92.08% after RES, and 92.30% for full PEN are historical paper values only.

All formal values use seed 0, ten contiguous StratifiedKFold folds per configuration, one-hot hard-label ROC-AUC, and population SD. Only ACC, hard-AUC, ASD sensitivity, and TD specificity are formal metrics. Mechanism measurements are kept separately in `diagnostics.csv`.

Status: **COMPLETE**; folds 120/120; GPU0=60, GPU1=60; formal wall time 1439.78 seconds.

|Config|ACC %|Hard-AUC %|SEN-ASD %|SPE-TD %|ASD FN|TD FP|
|---|---:|---:|---:|---:|---:|---:|
|B0|92.0742 +/- 2.6000|91.8546 +/- 2.5815|89.0793 +/- 3.7283|94.6300 +/- 4.0457|44|25|
|B1|91.2696 +/- 3.1421|91.0897 +/- 3.0778|88.8354 +/- 3.7320|93.3441 +/- 4.9921|45|31|
|B2|91.7294 +/- 2.9505|91.5625 +/- 2.9444|89.3415 +/- 3.6677|93.7835 +/- 3.6749|43|29|
|D0|91.6144 +/- 3.1771|91.4803 +/- 3.0794|89.8293 +/- 3.4050|93.1314 +/- 5.3846|41|32|
|D1|91.8430 +/- 3.0666|91.6707 +/- 3.0592|89.5671 +/- 4.0211|93.7743 +/- 4.2842|42|29|
|D2|91.8430 +/- 2.6507|91.6544 +/- 2.6009|89.3171 +/- 3.1945|93.9917 +/- 4.2078|43|28|
|D3|91.6157 +/- 2.6800|91.5191 +/- 2.6585|90.3232 +/- 2.8246|92.7151 +/- 3.4059|39|34|
|D4|91.1546 +/- 3.0506|91.0362 +/- 2.9748|89.5793 +/- 3.8330|92.4931 +/- 5.2301|42|35|
|D5|91.9579 +/- 3.0901|91.7590 +/- 3.1302|89.0915 +/- 4.3086|94.4265 +/- 3.2394|44|26|
|D6|91.6157 +/- 2.6800|91.4352 +/- 2.6307|89.0915 +/- 3.8411|93.7789 +/- 4.5987|44|29|
|D7|91.6144 +/- 3.0498|91.4671 +/- 3.0174|89.5854 +/- 3.6383|93.3488 +/- 4.2787|42|31|
|D8|91.7294 +/- 2.8130|91.5776 +/- 2.7545|89.8110 +/- 3.8125|93.3441 +/- 4.9006|41|31|

## Paired Comparisons

D4 minus each comparator; percentage-point mean difference and paired fold wins/losses.

|Comparison|ACC (pp; W/L)|Hard-AUC (pp; W/L)|SEN-ASD (pp)|SPE-TD (pp)|
|---|---:|---:|---:|---:|
|D4-B0|-0.9195; 0/5|-0.8185; 2/6|+0.5000|-2.1369|
|D4-B1|-0.1149; 2/3|-0.0536; 3/3|+0.7439|-0.8511|
|D4-B2|-0.5747; 1/6|-0.5263; 1/6|+0.2378|-1.2905|
|D4-D0|-0.4598; 0/4|-0.4441; 1/5|-0.2500|-0.6383|
|D4-D1|-0.6883; 1/6|-0.6345; 1/7|+0.0122|-1.2812|
|D4-D2|-0.6883; 1/6|-0.6182; 1/8|+0.2622|-1.4986|
|D4-D3|-0.4611; 2/6|-0.4830; 3/6|-0.7439|-0.2220|
|D4-D5|-0.8033; 0/7|-0.7228; 2/7|+0.4878|-1.9334|
|D4-D6|-0.4611; 1/4|-0.3990; 3/4|+0.4878|-1.2858|
|D4-D7|-0.4598; 0/4|-0.4309; 1/6|-0.0061|-0.8557|
|D4-D8|-0.5747; 1/6|-0.5414; 1/7|-0.2317|-0.8511|

## Trainable Parameters

Counts are for the complete PLN, including DCER32 when present. They make the D0 capacity caveat explicit; D0's frozen reference implementation retains the unused text Query/Key projections.

|Config|PLN trainable parameters|
|---|---:|
|B0|575121|
|B1|575121|
|B2|575121|
|D0|610314|
|D1|610314|
|D2|610314|
|D3|610314|
|D4|610314|
|D5|610314|
|D6|610314|
|D7|610314|
|D8|610314|

## Evidence Diagnostics

D4 selected-token deletion reduced held-out true-class log probability by 0.066916 on average; equal-count random-token deletion reduced it by 0.023244. Their difference was +0.043672. D4 wrong-class-prompt-block counterfactual changed mean output log probability by 0.016554; this is a model-dependence diagnostic, not proof of medical causality.

Selected latent tokens are four-way subspaces of the CAM representation, not anatomical ROIs or lesions. NULL probabilities are routing scores, not calibrated clinical likelihoods. The eight inherited GPT-4 sentences mostly describe general brain structure or function and are not validated ASD pathology priors.

## Scientific Decision

**NO-GO**. GO requires D4 to gain >0.1 pp over B2, D1, D3, D5, D6, D7 and D8 in both ACC and hard-AUC with at least 7/10 paired-fold wins, plus greater selected-token deletion impact than an equal-budget random deletion. This is a frozen internal protocol, not independent clinical validation.

Prior-art review: MMGPL (MedIA 2024), BrainPrompt (MICCAI 2025), and ERASER (ACL 2020) already cover prompt weighting or explanation-completeness ideas. DCER is not supported by sparse attention alone; it must show predictive gains and faithful keep/delete behavior against same-budget controls.

Historical protocol limitations remain: validation equals test for checkpoint selection; preprocessing/RFE was run before outer splits; GFR propagates over all 871 participants. Results are internal comparability evidence, not unbiased generalization or clinical evidence.

## Audit

Runtime sources are isolated in `.runtime/sources`; protected root SHA-256: `{"ABIDE-simple-2-concat-weighted-cosine.sh": "28c219e7c2deef4078cf8c28dd1b72f06f7494e997b9e38f0d933271521d5763", "layers.py": "4f7c07861bdeeb35fe156448aac4b32bae1b2f813c957d679f50dbd0225e819a", "main.py": "dbcec1c06dbe42de3b8750d25d49e206aefb13108e0c7032a6b063459ddfb0e2", "model.py": "254a9d757d5a891e5191b284b9ee4756926804688b1923358e607c1552580cda", "network.py": "484b61d95b7202140400d361de35fe6ecbce95c8d9823b476027de0059929fec", "utils.py": "95a8cd0326bc95e90ff4e0a5fc85ebc2b4fd2150955bc975a604374d81267b67"}`.
Code SHA-256: `{"integration.patch": "f780479adc25b9b07bfbd0e89f29b133c070f9476f29b990e12a5c7074f85c68", "integration.py": "51b13af79c967749386ac0ded079ba02d0549d445c177a8cb9b08941e3612d31", "method.py": "e148adb283d15f47ae58b9337f6564543332fe36dd22fa1d9967461966498036", "run.py": "0979a8435cd796116f083d86999bb6743185290e2d4037d40691e30cd14b2139"}`.
Frozen RN50 prompt SHA-256: `7b6571d5e55a36fa5f11fa88827fb52c2cca1e7b651af90afce4f0c913f0d939`.
Control matrix SHA-256: `{"D4": "7b6571d5e55a36fa5f11fa88827fb52c2cca1e7b651af90afce4f0c913f0d939", "D5": "859fa673b6df5a81e926ef1f73d598f95902602068f72f8204fa81acaed5d86e", "D6": "6656960ff7142c14c7eef44c8a3d77342ef3333ed264487510c5848fbfe5e294", "D7": "2ad57969168b7668a5967939986d79c0c6a335faaf8c2ddcf4d6844eeebe0c18", "D8": "46c98fcdd408d2f950eca20bd0d972af00f2e658c7d490d87718d97e53b3a22b"}`.
Worker count: 4/GPU; GPU fold counts: `{"0": 60, "1": 60}`.

Full fold metrics are in `fold_metrics.csv`; route, null, residual, counterfactual, and deletion diagnostics are in `diagnostics.csv`.
