# Experiment 43.4 — Physical CAM-only validation

This focused audit asks whether a model that physically constructs only the original CAM/PLN and OutputLayer can reproduce the functional no-GFR result from Experiment 43.2. It has two registered groups and 20 folds total:

* **E0_ANCHORED_PHYSICAL_CAM** loads only the CAM state and CAM Adam state from each verified Experiment 43.2 untrained B0-B anchor, restores that anchor's post-construction four-family RNG, and trains the physical CAM-only model. It never constructs a GFR, GraphLearn, GCN, PEN, CLIP, MLP, AvgPool, prompt buffer, graph adjacency, graph loss, or GFR Adam. Its first two steps are compared against the read-only historical E43.2 E01 implementation and then its full fold is compared to E01.
* **E1_NATURAL_PHYSICAL_CAM** constructs the same `PhysicalCAMOnly` from its own constructor and applies the original Xavier Linear initialization flow. Each GPU seeds once at 0 and processes its registered five folds sequentially. It does not read a historical model or anchor for parameter initialization.

The model is exactly `X[871,320] → VariLengthInputLayer → EncodeLayer(s) → FeedForwardLayer(s) → OutputLayer`. The four input slices remain PHENO48, ANAT6, FUNC10 and CORR256. The CAM's `fusion_weight` parameter is retained because it belongs to the original CAM registration but is unused by the original forward; no unrelated parameter cleanup is performed. The only optimizer is one Adam over CAM parameters. The verified parameter count is 49,544, with no forbidden GFR/PEN/CLIP state keys.

The historical shell protocol is preserved: seed 0, StratifiedKFold(10, shuffle=True, random_state=10), 450 epochs, wait `>200`, lr .0038, weight decay .11, dropout .35, n_hidden18, n_head2, nlayer1, `val_idx=test_idx`, strict first improved ACC checkpoint, weighted NLL (`ASD=0`, `TD=1`). No graph loss is present because no graph exists. Hard-AUC is the original one-hot hard-prediction ROC-AUC. Results are ten-fold mean ± population SD and retain confusion counts.

E0 is a controlled equivalence test rather than a new natural trajectory. It restores the same CAM/Adam/RNG state as E43.2 E01. E1 is independent in initialization and therefore is not expected to match E0 fold by fold. Neither result is an unbiased clinical generalization estimate because the historical `val=test`, global preprocessing/RFE and transductive protocol are retained.

## Audits and source boundaries

The read-only source and anchor provenance is in `manifest.json`. E43.2's verified untrained anchors are used only by E0; E43.2 best checkpoints are not used for initialization. The runtime `layers.py` is a byte copy of the root CAM layer implementation. `integration.py` contains the only model constructor and has no import or construction path for network/GFR/PEN/CLIP. Local loss and Xavier initialization wrappers are mathematically the same as `utils.ClsLoss` and `utils.my_weight_init`, without importing `utils.py` (which imports the graph network module).

The preflight checks forbidden structure keys, parameter count, data and fold hashes, E0 CAM state/Adam/RNG restoration, two-step historical E01 equality, input identity and no graph call path. Formal fold checkpoints include model, CAM Adam, best state, RNG, provenance and prediction SHA. A separate readback audits every completed best checkpoint and recomputes the four metrics.

## Reproduction

```bash
python -B run.py prepare
CUDA_VISIBLE_DEVICES=0 python -B run.py preflight --gpu 0
CUDA_VISIBLE_DEVICES=1 python -B run.py preflight --gpu 1
python -B run.py all
python -B run.py audit
python -B run.py report
```

The root handoff is appended only after all 20 folds and independent readback pass. No root model/data/history files are modified and no Git commit or push is performed by the experiment runner.
