# Core Model Diff

Baseline: Experiment 53.2 `cadr_gfr.py`, SHA256 `8eb060ca810992af2d03d6e411746553f7dcf9a91746fb7048e05c5e0b43220f`.

Experiment 53.3 changes only the active CADR forward path after the modality's fold-local reliability and disease adjacency weights are computed:

```diff
 if reliability_m == 0:
-    M_m = GCN_m(X, normalize_adj(0 + I))
-    H += sigmoid(fusion_weight_m) * alpha_m * M_m
+    M_m = GCN_m(X, normalize_adj(A_original_m + I))
+    H += sigmoid(fusion_weight_m) * M_m
+    positive_adjacency_m = negative_adjacency_m = direct_message_m = 0
 else:
     # Existing Experiment 53.2 CADR branch, unchanged.
```

For fixed-checkpoint audits only, the existing `modality_off:<name>` intervention also omits the fallback message from the fused representation. This preserves the established meaning of disabling a whole modality; disease-edge interventions such as `positive_off` continue to leave the fallback message active. Neither intervention branch is used during formal training (`intervention="full"`).

The implementation adds `propagation_modes` and `original_fallback_messages` to forward diagnostics. The `use_cadr=False` branch and the reliable-modality calculations are unchanged. The checkpoint records the training-time model SHA; the manifest records both that SHA and the post-training audit-only model SHA.
