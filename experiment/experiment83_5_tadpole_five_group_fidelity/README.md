# Experiment 83.5

Formal five-group TADPOLE experiment. The preferred OSSFS directory returned EIO errno 5 during create/write verification, so this release lives at `/model/spromptgl/experiment/experiment83_5_tadpole_five_group_fidelity`.

The protocol preserves the E83.4 TADPOLE data and fold hashes. G1, G4 and G5 are new ten-fold training groups. G2 and G3 are imported only after independent Best/Latest replay. E43.4 ABIDE checkpoints are not loaded into TADPOLE.

The immutable startup audit is `audits/preflight_start.json`; the startup source/config/data/prompt binding is `audits/training_seal.json`. Training entry points are `python -B common/run_e833.py G1_E28_E0 --workers N`, `python -B common/run_e833.py G4_E82_4_6_10_G1 --workers N`, and `python -B common/run_g5.py --workers N`. Inherited checkpoint replay and final audits are performed by `common/audit_e835.py`.

G5 reconstructs one untrained TADPOLE CAM anchor per fold from the E83.4 G2 construction order, restores the CAM Adam initial state and post-construction RNG, verifies the functional CAM-only reference for two updates, and trains the registered `PhysicalCAMOnly` model. The model contains only CAM state and one CAM Adam.
