# Experiment 71: GCDER

## Research question and hypothesis

GCDER tests whether two disease-conditioned queries, each combining a frozen disease text embedding with the patient-level CADR-GFR representation, retrieve different diagnostic evidence from the four CAM modality positions and improve the final ASD/TD decision through bounded score fusion. The retrieved vectors are learned latent evidence, not clinically validated lesion evidence.

## Related methods and distinction

SPromptGL combines patient-level multimodal representations with CAM and uses disease prompts in PEN to weight predefined local features; the present module does not re-create PEN, RES, SPGS, or the 4x4x9 partition. The audited local OutputLayer consumes the 144-dimensional CAM representation and returns log-softmax probabilities; its optional second argument is not a semantic injection path. [SPromptGL paper](https://papers.miccai.org/miccai-2025/0861-Paper5129.html), [official repository](https://github.com/wanxixi11/SPromptGL), [current research repository](https://github.com/tangaii/SPromptGL).

SABER uses Graph-CLS and graph representation to query patient text through cross-attention, then applies a gated semantic residual to classification; disease-level text is also used for contrastive supervision. GCDER instead forms one shared-parameter query per disease from frozen text plus the patient's GFR vector, reads the four CAM modality positions, and adds the two shared-score values to the original class log probabilities. [SABER paper](https://arxiv.org/html/2607.01901).

BrainPrompt uses multi-level ROI/subject/disease prompting. G1 uses only its official two ABIDE descriptions as fixed CLIP text embeddings, reordered to [ASD, TD]; it does not import BrainPrompt's full prompting architecture. [BrainPrompt paper](https://papers.miccai.org/miccai-2025/0116-Paper2260.html), [official repository](https://github.com/AngusMonroe/BrainPrompt).

## Frozen base and audit

The only base was `CADR_WD_0p01_GFR_WD_0p11`: 53.3 third group. Frozen optimizers are GFR/CAM Adam LR=0.0038, WD=0.11; CADR AdamW LR=0.001, WD=0.01 for every original parameter group; GCDER AdamW LR=0.001, WD=0.001. The read-only reference CSV, manifest, 10 tasks, checkpoint hashes, and historical checkpoint audit were cross-checked; the ten base fold means/population SDs were recomputed against the frozen reference table. No historical base checkpoint was used as a training start.
The 871-subject ABIDE input has 403 ASD (label 0) and 468 TD (label 1), with PHENO/ANAT/FUNC/CORR dimensions 48/6/10/256. The fallback execution root is `/model/spromptgl/experiment/experiment71_gcder` after the primary `/data` experiment-directory creation returned an I/O error. E59's ten warmup caches and CAM position hook were read-only reused after SHA and parity verification. E70 was inspected read-only for its base-bound warmup and prompt-cache provenance; no E70 best model was loaded.
The active `/data` CADR relation cache and the `/model` reference cache each contain ten JSON and ten NPZ files; all 20 SHA values match byte-for-byte, and each fold's pair SHA matches the audited 53.3 task. The caches were reused without recomputation. The `/data` and `/model` frozen 53.3 `run.py`, `cadr_gfr.py`, and `config.json` also had matching SHA values (574fb237..., f5032d96..., 4a2aeab5...).
The historical Experiment 51 CAM-only path physically removed GFR/PEN at that stage; Experiment 53.3 then added CADR-GFR and retained the CAM four-position output `[PHENO, ANAT, FUNC, CORR]` at 4x36. Its original classifier receives 144 features and has the 144->36->2 shape. The audited local `OutputLayer.forward(x, attn_embedding)` uses `x`; its second argument is not a semantic injection interface. Experiment 71 leaves both CAM and this head unchanged.
SHA anchors: 53.3 manifest `fae43596a9788d63bf8b0a0e52854fd1f7f014fea36c062bcf8a881806d7fcba`, fold CSV `2849ce9d161d5b5ea32bc1b210ebde4f4bd0daf25e316298934b8029b0f8f9a2`, checkpoint audit `a38f541f3afea51c28d64ca586713431b8b067ba1ad676c37a708a0a91718007`, CLIP cache `c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`. The complete protected inventory (1678 files) and all source/data SHA values are retained in `.runtime/preflight.json`.

## Model and equations

For patient i, GFR gives f_i in R^320, and one CAM forward gives H_i in R^(4x36), ordered [PHENO, ANAT, FUNC, CORR]. For every patient both disease queries are always evaluated; the true label never selects a query or score. For c in {ASD, TD}: q_i,c = LayerNorm(W_t t_c + W_f f_i), with shared W_t: D->36 and W_f: 320->36. Shared projections produce K_i=W_K H_i and V_i=W_V H_i. Attention is A_i=softmax(Q_i K_i^T/sqrt(36), over the four modality positions), and retrieved evidence is R_i=A_i V_i. A single shared score function is Score(r)=Linear(36,1)(GELU(Linear(36,36)(LayerNorm(r)))); it yields s_i=[s_i,ASD,s_i,TD].

The original OutputLayer emits logp_base. GCDER uses logp_final=log_softmax(logp_base+beta*s), beta=0.5*tanh(beta_raw), beta_raw initialized to zero. This is equivalent to applying the same correction to raw logits and re-normalizing. At beta=0 the original classifier is exactly preserved; the binary ASD-minus-TD margin changes by beta*(s_ASD-s_TD). The loss remains L_cls(final)+L_graph+0.05 L_pair+0.01 L_adv+0.01 L_preserve. GCDER adds 52,490 parameters at D=1024; the full model has 1,011,386 parameters in each fold.

Preflight tensor sizes: `{"gfr": [871, 320], "cam": [871, 4, 36], "query": [871, 2, 36], "key": [871, 4, 36], "value": [871, 4, 36], "attention": [871, 2, 4], "evidence": [871, 2, 36], "scores": [871, 2], "final_logp": [871, 2]}`. Identity max error was 1.19e-07; score common-shift, raw-logit equivalence, and log-probability normalization checks passed. The CAM feature hook captures `FeedForward[-1]` from the same forward and preserves gradients.

## Prompt groups

All groups use the same frozen OpenAI CLIP RN50 cache and the same two-token GCDER structure. G1 uses the official BrainPrompt descriptions: `A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.` and `A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD.` The published TC/ASD order is adapted to [ASD, TD].

G2 uses these original SPromptGL texts, preserving the spelling `Nromal`: `Nromal brain volume`; `Uniform cortical thickness`; `Symmetrical brain structure`; `Normal ventricular morphology`; `Altered brain volume`; `Abnormal functional connectivity`; `White matter integrity issues`; `Different activation patterns`. Each embedding is L2-normalized; the first four are averaged as TD, the last four as ASD, each centroid is normalized, then the order is [ASD, TD]. This fixed aggregation is not SPromptGL's PEN projection.

G3 is a fixed anonymous pair generated from a private CPU torch.Generator with seed 710071; it matches G1 vector norms and inter-class cosine without using labels or predictions. The recorded maximum norm error was 0; cosine error was 5.96e-08. Text cache SHA: `c4b44b30acea3a59183c8a7df11139295782c1f5d5adecd670022f24108b383a`.

## Results

All values are percent, mean +/- population SD across ten folds. ACC, hard AUC (historical one-hot hard-prediction definition), probability AUC (class-1 probability, TD positive), SEN-ASD, SPE-TD, BA, and ASD-F1 were independently recomputed from the loaded best checkpoints.

| Group | ACC | Hard-AUC | Prob-AUC | SEN-ASD | SPE-TD | BA | ASD-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCDER-BRAINPROMPT-2 | 89.0857 +/- 4.4665 | 88.8736 +/- 4.5315 | 92.3851 +/- 3.7013 | 86.1098 +/- 6.2718 | 91.6374 +/- 4.6040 | 88.8736 +/- 4.5315 | 87.9228 +/- 4.9882 |
| GCDER-SPROMPTGL-8 | 88.9760 +/- 6.1759 | 88.8934 +/- 6.0034 | 91.9330 +/- 4.6396 | 87.6341 +/- 5.9510 | 90.1526 +/- 9.0195 | 88.8934 +/- 6.0034 | 88.1911 +/- 6.1285 |
| GCDER-ANONYMOUS-2 | 88.6272 +/- 3.2401 | 88.4731 +/- 3.2260 | 91.7517 +/- 2.9519 | 86.5854 +/- 4.2213 | 90.3608 +/- 4.4704 | 88.4731 +/- 3.2260 | 87.5811 +/- 3.5255 |
| 53.3 third-group base | 92.0742 +/- 2.8427 | 91.8653 +/- 2.8348 | 93.6684 +/- 3.2619 | 89.0915 +/- 3.5263 | 94.6392 +/- 3.5168 | 91.8653 +/- 2.8348 | 91.2470 +/- 3.0658 |

| Group | Best epoch mean +/- SD (range) | Actual epochs mean +/- SD (range) | Parameters | Confusion matrix totals (actual ASD row; actual TD row) |
|---|---:|---:|---:|---|
| G1 | 153.3 +/- 76.3 (93-363) | 343.8 +/- 46.7 (295-450) | 1,011,386 | [[347, 56], [39, 429]] |
| G2 | 147.9 +/- 97.9 (72-434) | 331.3 +/- 45.2 (274-450) | 1,011,386 | [[353, 50], [46, 422]] |
| G3 | 136.4 +/- 35.3 (77-183) | 338.4 +/- 35.3 (279-385) | 1,011,386 | [[349, 54], [45, 423]] |

### Fold-paired comparisons

Each paired row reports the first system minus the second in percentage points. The CSV carries all ten fold differences, their mean, and win/tie/loss counts for every metric.

| Comparison | Metric | Mean delta (pp) | W/T/L | Fold deltas (pp) |
|---|---|---:|---:|---|
| G1_minus_BASE | acc | -2.9885 | 0/2/8 | [0.0,-2.2988505747126453,-6.896551724137936,-3.448275862068968,-6.896551724137936,-2.2988505747126453,0.0,-4.597701149425291,-2.298850574712631,-1.1494252873563227] |
| G1_minus_BASE | hard_auc | -2.9918 | 1/1/8 | [0.0,-2.04135737009544,-7.051961823966053,-3.3776595744680833,-7.499999999999986,-2.127659574468069,0.5585106382978609,-4.627659574468069,-2.686170212765944,-1.0638297872340416] |
| G1_minus_BASE | prob_auc | -1.2833 | 2/0/8 | [-0.7784120394395444,-2.704135737009551,-3.5524920466595944,-0.5851063829787222,-5.212765957446791,-0.26595744680851396,-0.4255319148936252,1.1170212765957501,0.3723404255319167,-0.7978723404255419] |
| G1_minus_BASE | sen | -2.9817 | 2/3/5 | [0.0,2.439024390243901,-9.756097560975604,-2.5,-15.0,0.0,7.5,-5.0,-7.5,0.0] |
| G1_minus_BASE | spe | -3.0019 | 1/2/7 | [0.0,-6.521739130434781,-4.347826086956516,-4.255319148936167,0.0,-4.255319148936167,-6.382978723404264,-4.255319148936152,2.1276595744680833,-2.1276595744680833] |
| G1_minus_BASE | ba | -2.9918 | 1/1/8 | [0.0,-2.041357370095426,-7.051961823966053,-3.3776595744680833,-7.5,-2.1276595744680833,0.5585106382978751,-4.627659574468083,-2.6861702127659584,-1.0638297872340416] |
| G1_minus_BASE | asd_f1 | -3.3242 | 1/1/8 | [0.0,-1.8300653594771177,-7.938740428191906,-3.762903762903747,-8.472400513478831,-2.3753711517424705,0.6913580246913682,-5.128205128205138,-3.2571032571032674,-1.1684518013631902] |
| G2_minus_BASE | acc | -3.0982 | 0/5/5 | [-4.545454545454547,0.0,-8.045977011494259,0.0,-3.4482758620689538,0.0,0.0,-12.64367816091955,-2.298850574712631,0.0] |
| G2_minus_BASE | hard_auc | -2.9720 | 2/3/5 | [-4.878048780487802,0.13255567338282503,-8.138918345705193,0.3723404255319167,-3.377659574468069,0.0,0.0,-11.702127659574458,-2.1276595744680833,0.0] |
| G2_minus_BASE | prob_auc | -1.7354 | 2/0/8 | [-2.9579657498702687,-1.3785790031813434,-4.506892895015909,-0.3723404255319025,-3.0319148936170137,0.6382978723404307,-0.5851063829787222,-5.319148936170222,1.4361702127659584,-1.2765957446808613] |
| G2_minus_BASE | sen | -1.4573 | 2/5/3 | [-9.756097560975604,2.439024390243901,-9.756097560975604,5.0,-2.5,0.0,0.0,0.0,0.0,0.0] |
| G2_minus_BASE | spe | -4.4866 | 0/4/6 | [0.0,-2.173913043478265,-6.521739130434781,-4.255319148936167,-4.255319148936152,0.0,0.0,-23.40425531914893,-4.255319148936181,0.0] |
| G2_minus_BASE | ba | -2.9720 | 2/3/5 | [-4.878048780487802,0.13255567338282503,-8.138918345705193,0.3723404255319167,-3.3776595744680833,0.0,0.0,-11.702127659574472,-2.1276595744680833,0.0] |
| G2_minus_BASE | asd_f1 | -3.0559 | 2/3/5 | [-5.682605682605669,0.2677376171352108,-8.919753086419746,0.3206669873336523,-3.5263003232442003,0.0,0.0,-10.77499279746472,-2.2435897435897516,0.0] |
| G3_minus_BASE | acc | -3.4470 | 0/1/9 | [-1.136363636363626,-2.2988505747126453,-2.2988505747126595,-8.045977011494259,-8.045977011494259,-4.597701149425291,0.0,-3.448275862068982,-2.298850574712631,-2.2988505747126453] |
| G3_minus_BASE | hard_auc | -3.3923 | 1/0/9 | [-1.0638297872340416,-1.9088016967126151,-2.306468716861076,-8.00531914893618,-8.563829787234027,-4.627659574468069,0.3723404255319167,-3.3776595744680975,-1.9414893617021391,-2.499999999999986] |
| G3_minus_BASE | prob_auc | -1.9168 | 0/0/10 | [-1.1416709911780174,-1.0074231177094362,-0.7953340402969218,-5.478723404255305,-3.1382978723404165,-1.2234042553191387,-0.5319148936170279,-0.7978723404255277,-1.2234042553191529,-3.8297872340425556] |
| G3_minus_BASE | sen | -2.5061 | 3/1/6 | [0.0,4.878048780487802,-2.439024390243901,-7.5,-15.0,-5.0,5.0,-2.5,2.5,-5.0] |
| G3_minus_BASE | spe | -4.2784 | 0/1/9 | [-2.1276595744680833,-8.695652173913047,-2.173913043478251,-8.510638297872347,-2.1276595744680833,-4.255319148936167,-4.255319148936167,-4.255319148936152,-6.38297872340425,0.0] |
| G3_minus_BASE | ba | -3.3923 | 1/0/9 | [-1.0638297872340416,-1.9088016967126151,-2.306468716861076,-8.00531914893618,-8.563829787234042,-4.627659574468083,0.3723404255319167,-3.3776595744680833,-1.9414893617021391,-2.5] |
| G3_minus_BASE | asd_f1 | -3.6658 | 1/0/9 | [-1.2009087958454927,-1.5325670498084207,-2.4691358024691397,-8.891108891108885,-9.56604371238518,-5.063291139240505,0.47257383966245925,-3.6351833820188233,-1.9387116948092569,-2.8340080971659916] |
| G1_minus_G2 | acc | +0.1097 | 3/2/5 | [4.545454545454547,-2.2988505747126453,1.1494252873563227,-3.448275862068968,-3.448275862068982,-2.2988505747126453,0.0,8.045977011494259,0.0,-1.1494252873563227] |
| G1_minus_G2 | hard_auc | -0.0198 | 4/0/6 | [4.878048780487802,-2.173913043478265,1.0869565217391397,-3.75,-4.122340425531917,-2.127659574468069,0.5585106382978609,7.074468085106389,-0.5585106382978609,-1.0638297872340416] |
| G1_minus_G2 | prob_auc | +0.4521 | 5/0/5 | [2.1795537104307243,-1.3255567338282077,0.9544008483563147,-0.2127659574468197,-2.1808510638297776,-0.9042553191489446,0.159574468085097,6.436170212765973,-1.0638297872340416,0.47872340425531945] |
| G1_minus_G2 | sen | -1.5244 | 2/4/4 | [9.756097560975604,0.0,0.0,-7.5,-12.5,0.0,7.5,-5.0,-7.5,0.0] |
| G1_minus_G2 | spe | +1.4847 | 4/2/4 | [0.0,-4.347826086956516,2.173913043478265,0.0,4.255319148936152,-4.255319148936167,-6.382978723404264,19.148936170212778,6.382978723404264,-2.1276595744680833] |
| G1_minus_G2 | ba | -0.0198 | 4/0/6 | [4.878048780487802,-2.173913043478251,1.0869565217391397,-3.75,-4.122340425531917,-2.1276595744680833,0.5585106382978751,7.074468085106389,-0.5585106382978751,-1.0638297872340416] |
| G1_minus_G2 | asd_f1 | -0.2683 | 4/0/6 | [5.682605682605669,-2.0978029766123285,0.98101265822784,-4.083570750237399,-4.946100190234631,-2.3753711517424705,0.6913580246913682,5.646787669259581,-1.0135135135135158,-1.1684518013631902] |
| G1_minus_G3 | acc | +0.4585 | 5/3/2 | [1.136363636363626,0.0,-4.597701149425276,4.597701149425291,1.1494252873563227,2.2988505747126453,0.0,-1.1494252873563084,0.0,1.1494252873563227] |
| G1_minus_G3 | hard_auc | +0.4005 | 6/0/4 | [1.0638297872340416,-0.13255567338282503,-4.745493107104977,4.6276595744680975,1.0638297872340416,2.5,0.18617021276594414,-1.2499999999999716,-0.744680851063805,1.4361702127659441] |
| G1_minus_G3 | prob_auc | +0.6335 | 7/0/3 | [0.363258951738473,-1.696712619300115,-2.7571580063626726,4.893617021276583,-2.074468085106375,0.9574468085106247,0.10638297872340274,1.9148936170212778,1.5957446808510696,3.0319148936170137] |
| G1_minus_G3 | sen | -0.4756 | 4/2/4 | [0.0,-2.439024390243901,-7.317073170731703,5.0,0.0,5.0,2.5,-2.5,-10.0,5.0] |
| G1_minus_G3 | spe | +1.2766 | 5/2/3 | [2.1276595744680833,2.173913043478265,-2.173913043478265,4.255319148936181,2.1276595744680833,0.0,-2.1276595744680975,0.0,8.510638297872333,-2.1276595744680833] |
| G1_minus_G3 | ba | +0.4005 | 6/0/4 | [1.0638297872340416,-0.13255567338281082,-4.745493107104977,4.6276595744680975,1.0638297872340416,2.5,0.18617021276595835,-1.25,-0.7446808510638192,1.4361702127659584] |
| G1_minus_G3 | asd_f1 | +0.3417 | 6/0/4 | [1.2009087958454927,-0.29749830966869695,-5.469604625722766,5.128205128205138,1.0936431989063493,2.6879199874980344,0.21878418502890895,-1.4930217461863151,-1.3183915622940106,1.6655562958028014] |
| G2_minus_G3 | acc | +0.3487 | 5/2/3 | [-3.4090909090909207,2.2988505747126453,-5.747126436781599,8.045977011494259,4.597701149425305,4.597701149425291,0.0,-9.195402298850567,0.0,2.2988505747126453] |
| G2_minus_G3 | hard_auc | +0.4203 | 5/0/5 | [-3.8142189932537605,2.04135737009544,-5.832449628844117,8.377659574468098,5.186170212765958,4.627659574468069,-0.3723404255319167,-8.32446808510636,-0.18617021276594414,2.499999999999986] |
| G2_minus_G3 | prob_auc | +0.1814 | 5/0/5 | [-1.8162947586922513,-0.37115588547190725,-3.7115588547189873,5.106382978723403,0.10638297872340274,1.8617021276595693,-0.053191489361694266,-4.521276595744695,2.659574468085111,2.5531914893616943] |
| G2_minus_G3 | sen | +1.0488 | 5/0/5 | [-9.756097560975604,-2.439024390243901,-7.317073170731703,12.5,12.5,5.0,-5.0,2.5,-2.5,5.0] |
| G2_minus_G3 | spe | -0.2081 | 6/1/3 | [2.1276595744680833,6.521739130434781,-4.34782608695653,4.255319148936181,-2.127659574468069,4.255319148936167,4.255319148936167,-19.148936170212778,2.127659574468069,0.0] |
| G2_minus_G3 | ba | +0.4203 | 5/0/5 | [-3.8142189932537605,2.04135737009544,-5.832449628844117,8.377659574468098,5.186170212765958,4.627659574468083,-0.3723404255319167,-8.324468085106389,-0.18617021276594414,2.5] |
| G2_minus_G3 | asd_f1 | +0.6100 | 5/0/5 | [-4.481696886760176,1.8003046669436316,-6.450617283950606,9.211775878442538,6.03974338914098,5.063291139240505,-0.47257383966245925,-7.139809415445896,-0.30487804878049474,2.8340080971659916] |

## Mechanism analysis

These are diagnostics of model-produced latent evidence. They do not establish clinical lesion localization.

| Group | beta mean (mean abs; SD; range) | s_ASD / s_TD means | Delta-s mean (SD; mean abs) | beta*Delta-s mean (mean abs) | attention entropy ASD / TD | attention | query cosine | GCDER test prediction changes | no-GFR attention change | no-GFR margin change | no-GFR prediction flips |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| G1 | 0.01788 (0.05942; 0.05823; -0.07242..0.07123) | 2.9004 / -1.4839 | 4.3844 (15.243; 9.8738) | 0.032848 (0.6409) | 1.0626 / 1.0630 | PHENO: 0.227/0.278 (|Delta|=0.1819), ANAT: 0.238/0.252 (|Delta|=0.1762), FUNC: 0.322/0.271 (|Delta|=0.2214), CORR: 0.213/0.199 (|Delta|=0.1698) | 0.78743 | 0.0677 | 0.133421 | 0.459631 | 0.0655 |
| G2 | -0.01785 (0.06662; 0.06824; -0.07761..0.12457) | -1.4356 / -5.512 | 4.0765 (20.033; 13.15) | 0.29958 (0.98553) | 0.9932 / 1.0154 | PHENO: 0.294/0.272 (|Delta|=0.2792), ANAT: 0.226/0.230 (|Delta|=0.2276), FUNC: 0.221/0.255 (|Delta|=0.2105), CORR: 0.259/0.243 (|Delta|=0.2817) | 0.52922 | 0.0999 | 0.128253 | 1.54566 | 0.0965 |
| G3 | 0.00838 (0.06598; 0.06681; -0.07584..0.08401) | -1.5429 / -1.2228 | -0.32016 (19.737; 12.81) | -0.12508 (0.92732) | 1.0526 / 1.0555 | PHENO: 0.320/0.259 (|Delta|=0.2403), ANAT: 0.236/0.240 (|Delta|=0.1983), FUNC: 0.228/0.248 (|Delta|=0.1945), CORR: 0.216/0.254 (|Delta|=0.2031) | 0.74838 | 0.0850 | 0.145155 | 0.569519 | 0.0563 |

#### Site and quality-block screening

| Group | max abs evidence/site correlation | site | Evidence vs PHENO RMS | ANAT RMS | FUNC RMS | CORR RMS |
|---|---:|---|---:|---:|---:|---:|
| G1 | 0.0328 | SITE_ID_NYU | +0.0000 | -0.0045 | -0.0109 | -0.0165 |
| G2 | 0.0371 | SITE_ID_NYU | +0.0000 | -0.0064 | -0.0177 | -0.0477 |
| G3 | 0.0387 | SITE_ID_USM | +0.0000 | -0.0009 | -0.0048 | -0.0068 |

#### Active gradient path

Post-identity-step mean gradient norms over the saved snapshots (all finite):

| Group | beta | text projection | GFR condition projection | Key | Value | shared Score | correction to CAM | correction to GFR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 | 0.16348 | 0.10049 | 0.61113 | 0.051476 | 0.037713 | 0.025346 | 14.379 | 60.253 |
| G2 | 0.29897 | 0.13125 | 0.80406 | 0.087777 | 0.059895 | 0.044264 | 20.324 | 73.764 |
| G3 | 0.17544 | 0.095254 | 0.59951 | 0.068263 | 0.059072 | 0.032253 | 13.516 | 52.573 |

Attention deltas are A_ASD,m - A_TD,m; their four components necessarily sum to zero per patient. Interpret the component-wise differences and entropy above, not that sum. `mechanism.csv` records all 871 graph nodes for each of the 30 checkpoints (26,130 patient-fold rows), marking each outer test fold; classification metrics and confusion matrices use only each fold's held-out test nodes. It includes patient attention, score, margin, prediction-change, graph-condition removal, input-block magnitude summaries, and PHENO site category. OOF correlations between evidence score difference and block RMS/site indicators are descriptive checks only; they are not causal or site-removal tests. The graph removal is a fixed-checkpoint sensitivity diagnostic, not a retrained ablation.

### Gradient, initialization, and recovery

GCDER initialization used `init_seed=710000+fold`, isolated by `torch.random.fork_rng`; the original Python/NumPy/CPU/CUDA RNG remained unchanged. Preflight passed PASS_10_OF_10 and warmup PASS_10_OF_10_REUSED_FROM_AUDITED_CACHE. The active two-epoch continuous-versus-resume replay was `PASS` with bitwise-equal predictions, model/optimizer states, and RNG. Independent audit status: 30/30 PASS.

Training gradient snapshots after the identity step are included per checkpoint. They cover beta, text projection, GFR condition projection, shared K/V and score, plus gradients from the GCDER correction back to CAM and GFR. Fold-level graph projection update norms and graph-zero attention/margin/prediction changes are in `fold_metrics.csv`; patient diagnostics are in `mechanism.csv`.

## Protocol and limitations

The frozen protocol is seed 0, StratifiedKFold(10, shuffle=True, random_state=10), 10 warmup epochs, up to 450 total epochs, strict ACC improvement, `wait>200`, and historical `val=test` checkpoint choice. Existing pre-outer-CV RFE and standardization remain. The patient graph contains all 871 subjects and is transductive. These results are exploratory comparisons under the same historical protocol, not an unbiased external test. PHENO contains demographic/site information; ANAT and FUNC are quality-related feature blocks and are not interpreted as lesions or neural time series. No site/quality causal attribution is claimed from descriptive block/site correlations alone.

## Decision: NO-GO

Pre-registered checks are recorded below; this single-seed decision is an engineering screen, not a statistical significance claim.

```json
{
  "candidate_conditions": {
    "G1": {
      "mean_acc_gain_at_least_0_5pp": false,
      "at_least_6_acc_winning_folds": false,
      "prob_auc_drop_no_more_than_0_5pp": false,
      "sen_drop_no_more_than_1pp": false,
      "positive_semantic_advantage_over_anonymous": true,
      "noncollapsed_mechanism_and_gradients": true,
      "performance_pass": false,
      "all_preregistered_conditions": false
    },
    "G2": {
      "mean_acc_gain_at_least_0_5pp": false,
      "at_least_6_acc_winning_folds": false,
      "prob_auc_drop_no_more_than_0_5pp": false,
      "sen_drop_no_more_than_1pp": false,
      "positive_semantic_advantage_over_anonymous": true,
      "noncollapsed_mechanism_and_gradients": true,
      "performance_pass": false,
      "all_preregistered_conditions": false
    }
  },
  "all_30_checkpoint_audits_pass": true,
  "all_patients_get_both_disease_queries": true,
  "no_label_conditioned_prompt_routing_or_score_selection": true,
  "historical_val_equals_test_retained": true,
  "decision_rule": "GO only if all preregistered conditions pass; CONDITIONAL only when a real-text group has positive mean ACC delta versus base and a noncollapsed mechanism; otherwise NO-GO."
}
```

A GO result would still require independent multi-seed validation, strict generalization testing, and component ablations before any MICCAI 2027 novelty claim. A failed or conditional screen ends this registered run without adding groups or tuning against these folds.
