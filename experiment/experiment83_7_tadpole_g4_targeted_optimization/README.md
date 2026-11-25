# Experiment 83.7

This directory contains the isolated Experiment 83.7 TADPOLE G4-targeted optimization run. Training source, locked inputs, configs, checkpoints, predictions, and audits are local to this directory. Historical E83.5 checkpoints are referenced by absolute path and SHA-256 in `audits/base_checkpoint_identity.json` and are not copied here.

The formal training entry point is `code/run_e837.py`. The six registered groups are T1 through T6. Training uses one PPU-ZW810E GPU with at most eight worker processes, seed 0, the locked E83.5/E83.6 data and split, and the historical Best/Latest protocol.

Recovery uses the same command with the affected group and fold IDs. The runner verifies checkpoint identity, restores model, optimizer, and RNG state, and never restarts a completed audited fold.
