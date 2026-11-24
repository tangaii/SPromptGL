# Experiment 83.6

Formal run for TADPOLE cross-repository reproduction, fidelity repair, and DARG-DPSL improvement. The selected directory is `/model/spromptgl/experiment/experiment83_6_tadpole_crossrepo_factorial` because the preferred `/data/adodas/SPromptGL-main/experiment` OSSFS mount returned `EIO` during setup. No project-root model code or historic experiment was edited.

## Frozen identities

- Main E83.5-order data: 598 patients, 365 features, six modalities, 10 fixed folds.
- Main data SHA-256: `0b8a14a9c066752c2d937c93d43c2824c41beaaa7d0a451b083db8dcba07c5bc`.
- Split SHA-256: `7c5a4dc9bd8ee7ffc6e5aff8bf841513f26a5ff6cf1b6c110a8153aa6a106ec5`.
- Author source: `ba3b9446fa451fde6864d6555bdf324de0c4b64a`.
- E83.5 reference source: `4649c5dddf166a4aa957c16c2f667a5cea485cdc`. The sealed server runtime is also recorded separately.
- `sp` source: `f4534f8d6cc9288dd1d08f85470c4eb78e64789c`.
- Prompt manifests record the RN50 weights, ordered 12 texts, class mapping, encoded tensors, and SCMR query mapping. MCI concepts proxy MMGPL's LMCI concepts.

## Execution

1. `code/finalize_registration.py` refreshes local source and cross-repository data inventories.
2. `code/seal_e836.py` freezes code, sources, inputs, and group configurations before any formal fold.
3. `code/preflight_e836.py` checks the fold identities, model paths, pair-mean gradients, RES equation, dropout/WD identities, and server E83.5 checkpoint inventory.
4. `code/resource_probe_e836.py` runs eight simultaneous GPU training steps. Both preflight and resource probe must pass before a training entry point will run.
5. `code/audit_historical_e836.py` independently replays H1-H5 Best and Latest checkpoints from the server E83.5 directory.
6. `code/run_s0.py` runs the `sp` positive control. `code/run_e836.py` runs registered main-experiment groups.

The main runner resumes verified Latest checkpoints and schedules at most eight folds globally. Requested batches are:

```bash
python code/run_e836.py --groups A1 A2 A3 B1 B2 --workers 8
python code/run_e836.py --groups C1 C2 C4 D1 D2 D3 D4 --workers 8
python code/run_e836.py --groups B3 C3 D5 D6 D7 E1 --workers 8
```

Run S0 in a separate process only when one of the eight worker slots is reserved for it. Do not exceed eight concurrent GPU fold processes across both runners.

## Outputs

Each group stores one Best and one resumable Latest checkpoint per fold, patient-level predictions, fold audit, fold metrics, and a group summary. Aggregated metrics, paired contrasts, mechanism diagnostics, Best/Latest independent replay, and the final report are written under `results/` and `audits/`. A run is complete only after all 190 new folds, 50 historical Best/Latest replays, next-step restoration checks, OOF coverage, and audit gates pass.

The registered main protocol retains full-cohort RFE and scaling before outer CV, transductive patient graphs, `val_idx == test_idx`, and first strict test-ACC checkpoint selection. Results are exploratory under that fixed historical protocol and can be optimistic under repeated candidate comparisons.
