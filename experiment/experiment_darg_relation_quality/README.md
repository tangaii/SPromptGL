# DARG Disease-Aware Patient Relation Quality Analysis

This frozen mechanism analysis uses the audited `E82.4.6.10-G1` ten-fold Best checkpoints. The checkpoint-internal model is `E82.4.6-G2 SCMR-DA-SRPE`, with `E74.4-G2 Positive-DCR` as the DARG graph path. No model parameter, checkpoint, historical input, or historical result is modified.

The authoritative directory is `/model/spromptgl/experiment/experiment_darg_relation_quality`. The preferred OSSFS directory failed with `Input/output error`, so no duplicate was created there.

Run with `python3.12 -I -B /model/spromptgl/experiment/experiment_darg_relation_quality/run_analysis.py`. The script restores each formal Best checkpoint through `real_model_runtime.py`, calls the verified DARG `rudag_forward` interface, calculates `normalize_adj(A + I)` with the formal utility, performs 20 local score permutations, and writes the report inputs and one two-panel figure.

The primary evaluation set contains edges with at least one endpoint in the current fold's test set after excluding the fold's historical DARG-supervised overlap. The strict sensitivity set contains only clean test-test edges. Patient identifiers are not stored in the heatmap artifact.
