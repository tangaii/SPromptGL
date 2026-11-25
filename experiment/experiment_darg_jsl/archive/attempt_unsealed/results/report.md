# DARG + JSL Formal Audit

Status: `PASS`.

## PERFORMANCE

The ten-fold Best-model macro results are reported as mean +/- population SD in percent.

ACC 92.7638 +/- 2.6289; Hard-AUC/BA 92.5968 +/- 2.6616; Prob-AUC 94.7514 +/- 2.9449.
SEN-ASD 90.3232 +/- 3.9342; SPE-TD 94.8705 +/- 3.3387; ASD-F1 92.0213 +/- 2.9474.

## MECHANISM

Residual-off to Full transitions across all test patients are recorded in mechanism.csv. Full JSL corrected 1 patients and harmed 0 relative to Residual-off.
The maximum observed CAM residual fraction was 0.07485498. Spectral-only and complement-only results are reported per fold.

## ABLATION_VALIDITY

JSL was trained from E61 epoch-10 full warmup states with the Original O 144x16 basis. DSE text, Gaussian/EMA evidence, and Prompt state were absent from the active model. The 19 historical training-test overlap patients were excluded from JSL supervision and had zero JSL residual.

## AUDIT_STATUS

Best forward 10/10; Latest reload and next-step replay 10/10; archive RNG mismatches 0; author log format PASS.

The protocol retains validation-equals-test checkpoint selection, pre-CV feature processing, a transductive patient graph, and intentional 19-patient training-test overlap. The results are an ablation audit and are not independent generalization evidence.

Reference comparison rows are in paired_comparisons.csv. Patient-level predictions and intervention margins are in patient_level.csv.
