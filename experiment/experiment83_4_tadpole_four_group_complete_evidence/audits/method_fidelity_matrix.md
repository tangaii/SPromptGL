# Experiment 83.4 Method-Fidelity Matrix

| Group or mechanism | Locked historical definition | E83.4 implementation and evidence |
|---|---|---|
| Data and folds | E83.3 TADPOLE MMGL preprocessing, whole-data RFE/scaling, ten stratified folds | Byte-identical processed data and splits, label/RID order and all six modality boundaries checked in preflight. |
| G1 E28-E0 | Original GFR, CAM, RES, PEN, eight original prompts, losses and optimizer | Inherited only after E83.3 Best/Latest replay, exact predictions, source/config/input hashes and independent replay pass. |
| G2 E43.1-E00 | No-PEN historical baseline | Same inheritance conditions. PEN is physically absent from the model and optimizer. |
| G3 E74.4-G2 | DARG, signed positive/negative edges, Positive-DCR, original GFR fusion and ten-epoch warmup | Same inheritance conditions. E83.4 does not redesign DARG or reselect a checkpoint. |
| G4 shared base | E74.4-G2 plus E82.4.6.10-G1 DPSL core | Fresh ten-fold training. For each fold, E83.4 deterministically replays G3's ten-epoch warmup, then compares the model, optimizer, RNG, history, and all relation-pair tensors against G3 Best and Latest before formal G4 optimization. |
| Binary DPSL core | SCMR-DA-SRPE, train-fold spectrum and evidence statistics, class evidence, dual-space residual, historical classifier, EMA and optimization budget | Archived 8-query two-class branch remains exact in forward, logits, losses, EMA and interventions. No new trainable pair-specific modules. |
| Three-class evidence | NC-MCI, NC-AD, MCI-AD all use the same SCMR parameters | Detached native probabilities set weights `(p_i+p_j)/2`; pair order is NC-MCI, NC-AD, MCI-AD. Direction is higher disease stage minus lower stage. No test labels enter routing. |
| Pair supervision | Full three-class CE; pair auxiliaries use only member classes | NC/MCI/AD all contribute to macro CE. Each pair loss selects only its two training classes, is class-balanced, and the three pair losses are averaged. |
| Prompt path | MMGPL Figure 6 12 prompts, historical P_text projection | RN50 vector and query file SHA fixed. `q_medical` is used by formal Full inference. The 8-row Procrustes fit has rank at most 8 in 32 dimensions and is diagnostic only. |
| Limits and training | Historical residual cap, gradient budget, optimizer, Best/Latest protocol | CAM residual <= 0.08; shared differential-gradient increment <= 0.001; test-fold Best selection and `val_idx == test_idx` retained. |
| GPU execution | One assigned GPU and worker ceiling 10 | A sealed one-versus-two process throughput and memory probe selects the formal GPU worker count. CPU input work and complete GPU model processes are recorded separately. |

Three-pair aggregation changes G4 computation order and consumes additional kernels relative to the E83.3 Top-2 route. It uses no additional random draws and does not reseed epochs. E83.4 records the first actual common-state difference in fold audits rather than claiming the full post-warmup trajectories are identical.
