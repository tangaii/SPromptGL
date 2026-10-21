# Experiment 32: DCER

Disease-Contrastive Evidence Routing replaces the complete original PEN (SPGS + RES) only inside this experiment's runtime copies. The original GFR, CAM generation, and OutputLayer remain unchanged. B0/B1/B2 isolate the PEN and provide the new internal baseline; D0-D8 are frozen ablations.

The formal design runs twelve configurations, one seed and ten contiguous folds per configuration, for 120 folds total. It follows the original ABIDE split, transductive graph, validation-equals-test checkpoint rule, and simple-2 optimizer/loss protocol. See `manifest.json` for hashes and `results/report.md` for the completed results.

Six configurations are assigned to each physical GPU, with four independent workers per card when resources permit. Checkpoints, runtime source copies, locks, and task logs are kept in `.runtime/`. No Git commit or push is part of this experiment.
