# Experiment 56.2 — DCRAG-GFR-R2

**Engineering status:** `COMPLETE_AUDITED`; one formal group `E56_2_FULL_DCRAG_GFR_R2`, seed 0, 10/10 independently reloaded checkpoints.

Hard-AUC uses hard predictions in the historical one-hot definition. Probability-AUC uses continuous P(ASD), with label 0 as positive. Values are fold mean ± population SD unless marked pooled. Historical `val=test`, pre-CV preprocessing/RFE and transductive 871-node graph are retained and limit unbiased generalization claims.

## Fold metrics (%)

|Fold|ACC|Hard-AUC|Probability-AUC|SPE-TD|SEN-ASD|BA|ASD-F1|Best epoch|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|86.3636|86.2999|91.0223|87.2340|85.3659|86.2999|85.3659|99|
|2|82.7586|83.2980|83.2450|73.9130|92.6829|83.2980|83.5165|97|
|3|75.8621|75.7158|78.1018|78.2609|73.1707|75.7158|74.0741|76|
|4|87.3563|86.9947|92.7128|91.4894|82.5000|86.9947|85.7143|89|
|5|88.5057|88.4309|93.8830|89.3617|87.5000|88.4309|87.5000|98|
|6|94.2529|93.7500|97.3936|100.0000|87.5000|93.7500|93.3333|104|
|7|87.3563|86.9947|91.2766|91.4894|82.5000|86.9947|85.7143|114|
|8|87.3563|87.1809|88.5638|89.3617|85.0000|87.1809|86.0759|116|
|9|81.6092|82.2340|84.6277|74.4681|90.0000|82.2340|81.8182|90|
|10|88.5057|88.4309|90.7979|89.3617|87.5000|88.4309|87.5000|91|

## Comparison with historical E51 E1

|Metric|E51 E1|E56|E56.1|E56.2|Δ E56.2−E51 E1 (pp)|
|---|---:|---:|---:|---:|---:|
|acc|91.9592|83.9263|89.6617|85.9927 ± 4.6785|-5.9666|
|hard_auc|91.7434|83.7066|89.4201|85.9330 ± 4.5005|-5.8104|
|prob_auc|93.1856|87.0762|92.1784|89.1624 ± 5.3993|-4.0232|
|spe|94.6392|86.7669|92.7243|86.4940 ± 7.9261|-8.1452|
|sen|88.8476|80.6463|86.1159|85.3720 ± 5.0394|-3.4756|
|ba|91.7434|83.7066|89.4201|85.9330 ± 4.5005|-5.8104|

## Mechanism audit

Train-only patient-disjoint relation OOF AUC: **['0.4972', '0.5266', '0.5268', '0.7268']** for PHENO/ANAT/FUNC/CORR. CORR retains the only strong disease relation evidence; weak modalities are reliability-weighted and remain in independent graph paths.

Conditional adversarial source-probe macro-AUC changed from 0.9996 to 1.0000; the trained discriminator macro-AUC was 0.7162. A source probe remaining near one means cross-modal alignment is not supported, even though GRL gradients are nonzero.

|Modality|Relation OOF AUC|Posthoc relation AUC|Gate mean|Gate patient SD|Confidence mean|Confidence SD|Same msg/input|AT msg/input|eta|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PHENO|0.4972|0.5004|0.3971|0.026958|0.2588|0.1792|0.2713|0.0120|0.1089|
|ANAT|0.5266|0.5186|0.1415|0.026122|0.1812|0.1109|0.3024|0.0115|0.1084|
|FUNC|0.5268|0.5315|0.1631|0.018825|0.2229|0.1477|0.3100|0.0116|0.1088|
|CORR|0.7268|0.7521|0.2983|0.023996|0.6568|0.2159|0.3504|0.0098|0.1052|

SoftCap final residual/input mean is 0.0889; preflight and fold diagnostics show no 56.1-style unbounded amplification. The Gate is no longer mathematically dead: mean patient SD is 0.023975, with nonzero first/second-layer preflight gradients and updates.

Strict selected-edge double swaps changed 98.22% of actual propagation edge identities on average and preserved all registered invariants. The fixed-checkpoint strict wrong-graph ΔACC is -1.9540 pp; this is an intervention dependency result, not an independent retraining gain.

## Fixed-checkpoint interventions

|Intervention|ΔACC pp|Flip rate|ΔP(ASD)|
|---|---:|---:|---:|
|disable_ANAT_message|-0.4598|0.0046|0.002502|
|disable_CORR_message|-6.8939|0.0942|0.079910|
|disable_FUNC_message|-0.4585|0.0069|0.002807|
|disable_PHENO_message|-1.3780|0.0138|0.007591|
|disable_cross_patient_messages_self_only|-7.9271|0.1000|0.086981|
|disable_cross_patient_messages_self_only_frozen_gate|-7.6972|0.0977|0.085764|
|neutral_disease_edge_weights|-3.4470|0.0528|0.054215|
|reference_e51_bypass|-34.6774|0.4731|0.437883|
|strict_wrong_patient_neighbors|-1.9540|0.0241|0.021285|
|strict_wrong_patient_neighbors_frozen_gate|-1.9540|0.0241|0.020923|
|wrong_disease_condition|-1.1468|0.0115|0.003864|

## Research gates

|Gate|Result|
|---|---|
|GO_RELATION_DISCRIMINATION|GO|
|GO_CONDITIONAL_ADVERSARIAL_ALIGNMENT|NO-GO|
|GO_DISEASE_PRESERVATION|GO|
|GO_TRUE_PATIENT_RELATION|GO|
|GO_CROSS_PATIENT_PROPAGATION|PARTIAL_CHECKPOINT_DEPENDENCE|
|GO_CLASSIFICATION_GAIN|NO-GO|
|GO_MICCAI2027_MODULE|NO-GO|

**Scientific decision: `PARTIAL`.** The relation path and fixed-checkpoint propagation are active, but the full preregistered classification/alignment evidence is incomplete.

The method is not promoted solely because code tests pass or because a fixed checkpoint responds to interventions. One seed, one ABIDE-A cohort and historical checkpoint selection remain limitations.
