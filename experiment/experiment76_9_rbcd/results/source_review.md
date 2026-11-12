# E76.9 source and paper review

- Read the local copy of the MICCAI 2025 SPromptGL paper PDF (converted with `pdftotext -layout`), including Section 2.1 (GFR, multi-relation patient graph and GCN), Section 2.2 (CAM/PEN), original classification/loss/training protocol, ABIDE split protocol, and ACC/AUC/SEN/SPE definitions.
- Inspected the current repository `main.py`, `model.py`, `network.py`, `layers.py`, `utils.py`, and the ABIDE training shell. E74.4-G2 is the actual DARG/Positive-DCR + CAM `PLNNoPEN` path; the original PEN is not reintroduced in E76.9.
- Read-only checked E74.4-G2 `run.py`, `soft_positive_dcr.py`, configuration, manifest, audit, results and warmup/best checkpoints. Historical E74.4-G2 identity and 30/30 audit were required by preflight.
- Read-only checked E76.7 and E76.8 source, configuration, manifests, mechanism/correction results and audit status. E76.8 was the prototype retrieval reference; E76.9 changes the boundary role to risk evidence and does not add boundary evidence directly to the disease score.
- E76.9 code is isolated in `/model/spromptgl/experiment/experiment76_9_rbcd`; no root model/data/history files were modified. The preferred `/data/adodas/SPromptGL-main/experiment/experiment76_9_rbcd` path failed the real write/fsync/atomic-readback probe with OSSFS `EIO errno=5`.
- Historical protocol limits retained: `val=test` checkpoint selection, pre-CV RFE/standardization, transductive patient graph, and Fold-2 historical overlap. The inner OOF teacher isolates only the new risk-supervision path.
