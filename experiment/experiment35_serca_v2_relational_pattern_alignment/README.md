# Experiment 35 — SeRCA-v2

This experiment tests whether a frozen medical relation rule over signed, TD-referenced FC co-deviations improves participant-relation messages beyond numeric FC similarity, unstructured pair interactions, incorrect rules, or patient mismatch. The only registered full model is `E11_SERCA_V2_FULL`.

The pair rules are `D2-D4` and `D3-D4`, frozen in [medical_prior.json](medical_prior.json). They are conservative concept-level hypotheses synthesized from the existing Experiment 19 prompt text and cited literature. They do not claim direct between-network FC edges, complete canonical networks, a common ASD direction, or patient-level lesions. The historical Experiment 19 package does not retain a raw GPT-4 response transcript or exact model version; no new LLM response is represented as a source.

The method fits a Ridge FC reference on outer-train TD participants using PHENO age/site/sex only. Training TD references are 5-fold OOF; training ASD and held-out participants use the full outer-train TD fit. Signed deviations are clipped to `[-3,3]`, converted to positive/negative domain scores above `|z|=1`, and combined into four signed pair products. Pair products are residualized against first-order scores with outer-train rows only. The full prior keeps two of the six possible domain pairs.

The original GFR graphs and main propagation stay unchanged. SeRCA-v2 adds directed Top-8 semantic recipients whose message sources are outer-train participants. Each message value comes from one explicit input group (PHENO, ANAT, FUNC or CORR) projected to 36 dimensions. A low-capacity router can assign all mass to a null source. The full branch adds a 5%-capped, zero-initialized residual from `T_sem @ V - T_base @ V`, after a detached outer-train RMS calibration. Direct messages use the same calibration and cap.

All 12 groups run 10 seed-0 folds. `E0_B0A` runs first and records each fold-entry RNG state. The other workers restore the corresponding E0 entry state before construction. The historical protocol remains unchanged, including `val=test` checkpoint selection, preprocessing/RFE before CV, transductive GFR over all 871 subjects, two original Adam optimizers, `simple-2`, and hard-label AUC. The report must state that this does not provide unbiased independent-test generalization evidence.

The runner and all outputs live in this experiment directory. Run on the current server after verifying the cached CLIP RN50 checkpoint:

```bash
python run.py prepare
python run.py preflight
python run.py run
```

The current execution is staged under `/tmp/SPromptGL-E35-SeRCA-v2-stage/experiment/experiment35_serca_v2_relational_pattern_alignment` because the `/data` OSSFS experiment directory returned `Input/output error`. No project-root model, dataset, or historical experiment file is written by the runner. It does not commit or push Git changes.
