# Experiment 41.1: SAGE-Align Final

This directory contains the seed-0, ten-fold final validation of
Semantic-Anchored Gated Evidence Alignment (SAGE-Align) on the historical
SPromptGL ABIDE protocol.

- `run.py prepare` builds frozen fold evidence caches.
- `run.py preflight` runs structural and gradient checks.
- `run.py e0 0` reproduces the B0-A anchor on GPU 0.
- `run.py dispatch` runs the two GPU adaptation workers.
- `run.py report` writes the Chinese report and diagnostics.
- `run.py finalize` appends the idempotent handoff section.

The completed result is `FINAL NO-GO`. The protocol retains `val=test`
checkpoint selection, pre-CV RFE/standardization, and the 871-subject
transductive patient graph. No root source, data, historical experiment, or
model parameter file is modified by this experiment.
