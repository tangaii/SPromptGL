# Experiment 53.1: Corrected DRAP Propagation

## Execution and Audit

Training: E1-E8, 80/80 folds. Independent best-checkpoint readback: 80/80 (80/80 exact stored-metric matches). Historical E0: read-only 10/10 checkpoint readback. Preflight: PASS.
The requested `/data/adodas/SPromptGL-main` experiment mount returned `OSError(5)` on directory creation. This run is stored on the separately verified writable `/model/spromptgl` overlay; source data, anchors and historical results remain on `/data` and are read-only. The two roots are different filesystems.

## Method Audit

SPromptGL uses four GraphLearn weighted-cosine patient graphs (threshold 0.9), two parameterized GCN layers per graph, normalized `A+I`, four correctly accumulated 320-dimensional messages and the original CAM. The paper's label-supervised adjacency term is distinct from the public simple-2 code's optimized classification plus GraphConstructLoss objective. The original GFR PHENO assignment overwrite is repaired in the Experiment 51 E1 LegacyFixedGFR base.
DeepASD aligns modality feature representations before patient-graph learning. Here the adversary receives paired-patient relation features, for the same balanced outer-train pairs across all four modalities. Only OOF-reliable modalities determine disease edges; the adversarial source set remains four modalities. Conditional E3/E4/E6/E7/E8 inputs carry train-pair same/different one-hot labels; E5 uses the same 2D condition slot set to zero.
Historical audits found GCN message collapse under strong decay (47.1/52), no robust superiority of true patient relations over strong controls (47.3/51.3), and Experiment 53 branch errors: negative propagation leaked into E1/E3, warmup was not passed, adversarial training gated off with CORR as the only reliable modality, and E7/E8 controls were poorly budget-matched. This run fixes these in its private module and does not modify historical code.

## Frozen Group Differences

E1: disease-reliable positive propagation only. E2: positive and negative, no adversary. E3: positive plus conditional four-modality adversary. E4: full signed conditional DRAP. E5: unconditional four-modality adversary. E6: continuous signed confidence without uncertainty skipping. E7: relation scores permuted over fixed CORR candidate edges with E4 fold-local degree budgets. E8: ordinary initial GraphLearn similarity neighbors with E4 per-node sign budgets, selected without disease q. All groups retain the same initial GFR/CAM and original `weight_decay=0.11`; pair and adversarial losses are 0.05 and 0.01; GRL scale 0.03; propagation coefficient 0.02 and negative factor 0.25. Epochs 0-9 have exactly zero added message; formal best selection starts at epoch 10.

## Ten-Fold Classification (mean ± population SD, percent)

|Group|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|
|---|---:|---:|---:|---:|---:|---:|---:|
|E0 E51_E1_EXACT_REPLAY_READ_ONLY|91.9592 ± 2.8684|91.7434 ± 2.8950|93.1856 ± 3.0236|94.6392 ± 3.3856|88.8476 ± 4.1651|91.7434 ± 2.8950|91.0928 ± 3.1408|
|E1 DRAP_RELATION_POSITIVE|91.6157 ± 2.8704|91.4538 ± 2.8414|93.3392 ± 3.4768|93.5661 ± 4.3300|89.3415 ± 3.8343|91.4538 ± 2.8414|90.8199 ± 3.0443|
|E2 DRAP_SIGNED_NO_ADV|91.7281 ± 2.9949|91.5648 ± 2.9888|93.2770 ± 3.3538|93.7882 ± 3.6583|89.3415 ± 3.8106|91.5648 ± 2.9888|90.9184 ± 3.2515|
|E3 DRAP_POSITIVE_CONDITIONAL_ADV|91.7294 ± 2.9505|91.5385 ± 2.9596|93.3394 ± 3.1986|93.9917 ± 4.4177|89.0854 ± 4.6177|91.5385 ± 2.9596|90.8903 ± 3.1622|
|E4 DRAP_FULL|91.7294 ± 2.8596|91.4587 ± 2.8090|92.8079 ± 4.1975|94.8381 ± 4.8824|88.0793 ± 4.0074|91.4587 ± 2.8090|90.8208 ± 3.0024|
|E5 DRAP_UNCONDITIONAL_ADV|91.6144 ± 2.9170|91.4375 ± 2.9081|93.5262 ± 2.7513|93.7835 ± 4.0275|89.0915 ± 4.0230|91.4375 ± 2.9081|90.7842 ± 3.1459|
|E6 DRAP_NO_UNCERTAINTY_SKIP|91.9592 ± 2.8684|91.6814 ± 2.9070|93.2877 ± 3.1258|95.2775 ± 3.5943|88.0854 ± 4.4434|91.6814 ± 2.9070|91.0121 ± 3.1922|
|E7 DRAP_SHUFFLED_RELATION_CONTROL|91.4982 ± 3.2653|91.3041 ± 3.2454|93.6582 ± 3.1313|93.7789 ± 4.5813|88.8293 ± 4.2134|91.3041 ± 3.2454|90.6495 ± 3.5053|
|E8 DRAP_UNIFORM_RELATION_CONTROL|91.4995 ± 3.1418|91.3102 ± 3.1325|93.2227 ± 3.2960|93.7789 ± 4.2653|88.8415 ± 4.1885|91.3102 ± 3.1325|90.6494 ± 3.3742|

E0 is historical read-only reference; it was not retrained. Hard-AUC derives from hard labels and is not interchangeable with Probability-AUC.

## Paired Differences

|Comparison|Metric|Mean pp|Wins/Ties/Losses|
|---|---|---:|---:|
|E1_minus_E0|acc|-0.3435|1/5/4|
|E1_minus_E0|hard_auc|-0.2896|2/4/4|
|E1_minus_E0|prob_auc|0.1535|6/0/4|
|E1_minus_E0|spe|-1.0731|0/6/4|
|E1_minus_E0|sen|0.4939|2/6/2|
|E2_minus_E1|acc|0.1123|2/6/2|
|E2_minus_E1|hard_auc|0.1110|3/3/4|
|E2_minus_E1|prob_auc|-0.0621|5/1/4|
|E2_minus_E1|spe|0.2220|2/6/2|
|E2_minus_E1|sen|0.0000|4/3/3|
|E3_minus_E1|acc|0.1136|2/7/1|
|E3_minus_E1|hard_auc|0.0847|4/4/2|
|E3_minus_E1|prob_auc|0.0003|4/0/6|
|E3_minus_E1|spe|0.4255|3/5/2|
|E3_minus_E1|sen|-0.2561|3/5/2|
|E4_minus_E2|acc|0.0013|2/6/2|
|E4_minus_E2|hard_auc|-0.1061|3/2/5|
|E4_minus_E2|prob_auc|-0.4691|3/1/6|
|E4_minus_E2|spe|1.0500|4/5/1|
|E4_minus_E2|sen|-1.2622|2/2/6|
|E4_minus_E3|acc|0.0000|2/6/2|
|E4_minus_E3|hard_auc|-0.0798|3/3/4|
|E4_minus_E3|prob_auc|-0.5315|3/0/7|
|E4_minus_E3|spe|0.8464|5/4/1|
|E4_minus_E3|sen|-1.0061|2/4/4|
|E4_minus_E5|acc|0.1149|2/7/1|
|E4_minus_E5|hard_auc|0.0212|3/3/4|
|E4_minus_E5|prob_auc|-0.7182|5/0/5|
|E4_minus_E5|spe|1.0546|6/3/1|
|E4_minus_E5|sen|-1.0122|1/5/4|
|E4_minus_E6|acc|-0.2299|2/5/3|
|E4_minus_E6|hard_auc|-0.2228|3/4/3|
|E4_minus_E6|prob_auc|-0.4798|5/0/5|
|E4_minus_E6|spe|-0.4394|2/6/2|
|E4_minus_E6|sen|-0.0061|2/6/2|
|E4_minus_E7|acc|0.2312|2/7/1|
|E4_minus_E7|hard_auc|0.1546|3/5/2|
|E4_minus_E7|prob_auc|-0.8503|3/0/7|
|E4_minus_E7|spe|1.0592|3/6/1|
|E4_minus_E7|sen|-0.7500|1/6/3|
|E4_minus_E8|acc|0.2299|3/6/1|
|E4_minus_E8|hard_auc|0.1485|4/3/3|
|E4_minus_E8|prob_auc|-0.4148|5/0/5|
|E4_minus_E8|spe|1.0592|6/3/1|
|E4_minus_E8|sen|-0.7622|1/5/4|
|E4_minus_E0|acc|-0.2299|1/6/3|
|E4_minus_E0|hard_auc|-0.2847|2/5/3|
|E4_minus_E0|prob_auc|-0.3777|3/0/7|
|E4_minus_E0|spe|0.1989|4/5/1|
|E4_minus_E0|sen|-0.7683|1/6/3|

## OOF Relation Evidence and Propagation

OOF evidence remains derived from inner patient-disjoint folds; it is not an independent clinical cohort. Only CORR is OOF-reliable when it meets AUC ≥0.55 and positive same/different separation. The adversarial branch nevertheless trains on four modality sources; this is CORR-anchored forced alignment, not evidence that all modalities have reliable disease relations.

|Group|OOF CORR AUC|Positive edges|Negative edges|Positive msg norm|Negative msg norm|Delta/H_E1|H_DRAP-X|q-to-message grad|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E1|0.7257|580.8|0.0|57.3311|0|0.00485241|0.0050492|0.0223221|
|E2|0.7257|581.4|416.5|57.4504|52.8643|0.0043648|0.00437198|0.0199075|
|E3|0.7257|580.7|0.0|57.8207|0|0.00489363|0.00489899|0.0233645|
|E4|0.7257|581.1|417.4|57.545|52.7176|0.00436653|0.00452224|0.0208461|
|E5|0.7257|580.3|417.2|57.5174|53.1993|0.00435477|0.00435925|0.0208624|
|E6|0.7257|3515.4|3887.6|64.2196|64.723|0.00441872|0.00447769|0.0241504|
|E7|0.7257|332.8|173.5|56.2476|50.591|0.00431425|0.00431488|0.0213869|
|E8|0.7257|571.6|408.3|57.5813|58.7943|0.00432373|0.00432446|0|

E1/E3 negative message norms are required to be exactly zero; E2/E4/E5/E7/E8 require active negative messages. E6 positive/negative edge masks are disjoint. See diagnostics.csv for each modality's OOF AUC/PR-AUC/Brier/ECE, q distribution, GCN norms, graph support, degree-one node fractions, and gradient measurements.

## Fixed-Checkpoint Interventions

These are eval-only interventions on each group's own best checkpoint, not additional training groups.

|Group|Intervention|ACC|Probability-AUC|Test flip fraction|Representation relative diff|Log-prob relative diff|
|---|---|---:|---:|---:|---:|---:|
|E1|propagation_off|91.6157|93.3285|0|0.00485241|0.0036328|
|E1|positive_off|91.6157|93.3285|0|0.00485241|0.0036328|
|E1|negative_off|91.6157|93.3392|0|0|0|
|E1|shuffled_identity|91.6157|93.3285|0|0.00411493|0.00622953|
|E1|ordinary_neighbors|91.6157|93.3072|0|0.00426938|0.00652936|
|E2|propagation_off|91.6131|93.2877|0.00114943|0.0043648|0.00401999|
|E2|positive_off|91.6131|93.2877|0.00114943|0.00486099|0.00296403|
|E2|negative_off|91.7281|93.2717|0|0.00111974|0.00184495|
|E2|shuffled_identity|91.4982|93.2877|0.00229885|0.00429539|0.00631174|
|E2|ordinary_neighbors|91.3832|93.2823|0.00344828|0.0044544|0.00699283|
|E3|propagation_off|91.6144|93.3341|0.00114943|0.00489363|0.0029132|
|E3|positive_off|91.6144|93.3341|0.00114943|0.00489363|0.0029132|
|E3|negative_off|91.7294|93.3394|0|0|0|
|E3|shuffled_identity|91.6144|93.3340|0.00114943|0.00411073|0.00571781|
|E3|ordinary_neighbors|91.8443|93.3288|0.00114943|0.0042931|0.00612762|
|E4|propagation_off|91.6144|92.8026|0.00114943|0.00436653|0.00428964|
|E4|positive_off|91.7294|92.8079|0|0.00486925|0.00320161|
|E4|negative_off|91.7294|92.8079|0|0.00111671|0.00198191|
|E4|shuffled_identity|91.4995|92.7920|0.00229885|0.004229|0.00694632|
|E4|ordinary_neighbors|91.3845|92.8079|0.00344828|0.00439587|0.00752177|
|E5|propagation_off|91.4995|93.5315|0.00114943|0.00435477|0.00398218|
|E5|positive_off|91.4995|93.5315|0.00114943|0.00486662|0.00292892|
|E5|negative_off|91.6144|93.5315|0|0.00112685|0.00188545|
|E5|shuffled_identity|91.6144|93.5102|0|0.00420555|0.00636387|
|E5|ordinary_neighbors|91.4995|93.5102|0.00114943|0.00440403|0.00696681|
|E6|propagation_off|91.7294|93.3249|0.00229885|0.00441872|0.00523101|
|E6|positive_off|91.9592|93.3249|0|0.00543421|0.00414304|
|E6|negative_off|91.9592|93.2984|0|0.00137102|0.00228611|
|E6|shuffled_identity|91.7294|93.3196|0.00229885|0.00416646|0.00760965|
|E6|ordinary_neighbors|91.7307|93.2877|0.00458464|0.00433654|0.00778085|
|E7|propagation_off|91.4982|93.6320|0|0.00431425|0.00432581|
|E7|positive_off|91.4982|93.6320|0|0.00475959|0.00445854|
|E7|negative_off|91.4982|93.6582|0|0.00107145|0.00104912|
|E7|shuffled_identity|91.4982|93.6582|0|0|0|
|E7|ordinary_neighbors|91.4982|93.6106|0|0.00434723|0.005348|
|E8|propagation_off|91.3845|93.2014|0.00114943|0.00432373|0.0046111|
|E8|positive_off|91.3845|93.2227|0.00114943|0.004872|0.00488118|
|E8|negative_off|91.4995|93.2068|0|0.00124531|0.00122332|
|E8|shuffled_identity|91.4995|93.2227|0|0|0|
|E8|ordinary_neighbors|91.4995|93.2227|0|0|0|

## Mechanism Conclusions

- Branch corrections are enforced from the frozen group table and checkpoint branch signatures, rather than inferred from group-name substrings.
- Warmup and active propagation status are stored per checkpoint; all formal best checkpoints must be at epoch ≥10.
- The adversarial source classifier receives all four modality labels even though only CORR is permitted to define disease edges. Conditional and unconditional modes share architecture, pair set, balance and fixed CORR confidence weights.
- E7 preserves candidate endpoints and reselects under the initial E4 degree budgets after per-forward CORR score permutation. The realized undirected edge budget can still be underfilled; E8 uses GraphLearn cosine and is q-independent. See diagnostics for undirected and directed edge counts and degree fractions.
- Across E1–E8, the original four-GCN message/input norm ratio ranges 2.61e-05–0.000306, while `||H_GFR-X||/||X||` ranges 4.57e-05–0.000541; total DRAP output displacement ranges 0.00431–0.00505. The relation message therefore changes the representation beyond the near-identity original GFR, while the original `weight_decay=0.11` GCN messages remain strongly contracted.
- The row-normalized difference operator can cancel edge-weight scale for degree-one nodes. Degree-0/1/multi fractions and q/message gradient are reported; relation BCE/AUC alone does not establish classification value.
- OOF calibration is applied to the final all-outer-train relation model without test-driven refitting. Diagnostics show OOF metrics beside post-training outer-train relation AUC; the latter is a fit diagnostic, not an independent generalization estimate.

## Decision

- ENGINEERING_GO: 80/80 checkpoints were independently reloaded; 80/80 saved metrics matched; historical E0 readback=10/10; preflight=PASS.
- RELATION_GO: CORR patient-disjoint OOF AUC=0.7257, reliable in 10/10 folds; PHENO/ANAT/FUNC mean OOF AUCs=0.4984/0.5031/0.4994. Site confounding cannot be tested because site metadata is absent from the processed 321-column feature table.
- ADVERSARIAL_NO-GO: four-source and GRL-gradient execution checks=passed; E4 modality accuracy=0.5901 (four-way chance=0.25), so modality origin remains distinguishable. No patient-disjoint post-adversarial relation OOF was available; E4−E5 Probability-AUC=-0.7182 pp and does not support an adversarial diagnostic gain.
- PROPAGATION_GO: E4 mean bounded delta norm=1.0318, q-to-message gradient=0.0208461, and propagation-off logit relative difference=0.00428964, with mean test flip fraction=0.00114943. This confirms a real but small fixed-checkpoint effect.
- CONTROL_MATCHING_LIMITED: E7 is short by 248.3 positive and 243.9 negative undirected edges on average relative to E4; edge identity is randomized correctly but realized graph budgets are not sufficiently matched.
- DIAGNOSTIC_NO-GO: E4−E0 ACC=-0.2299 pp (1/6/3 win/tie/loss), Probability-AUC=-0.3777 pp (3/0/7); E4 also lacks a consistent Probability-AUC advantage over E7/E8.
- MICCAI_CANDIDATE_NO-GO: the full candidate does not meet the preregistered diagnostic-increment criterion; real relation propagation is implemented, but disease-specific diagnostic value and fair E7 control separation are not established.

All folds use historical `val=test`, CV-before preprocessing/RFE and an 871-node transductive graph. They can overstate generalization. The ten folds overlap on one cohort and one seed; they are not ten independent clinical cohorts or external validation. The processed data have no site metadata, so site-confounding remains untested.

Report generated from 80 independently read model checkpoints. Source hashes and task status are in `manifest.json`.
