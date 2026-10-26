# Experiment 39: HASE-MoE

Heterogeneity-Aware Semantic Expert Mixture tests whether four literature-constrained, numerically verified CORR connection sets can support useful patient-specific expert routing after the original SPromptGL CAM representation.

## Frozen question

Does routing among four independently parameterized, masked signed-FC experts provide a stable internal gain over the Experiment32 B0-A backbone, an equal-capacity ordinary adapter, a dense generic MoE, uniform fusion, GFR-only routing, wrong anatomical masks, fold-local unsupervised feature sets, patient-mismatched FC, and a model without the specialization penalty?

All formal arms use ABIDE 871, four inputs PHENO48/ANAT6/FUNC10/CORR256, ASD=0 and TD=1. Inputs are read from the exact historical processed CSV and modal dictionary. RFE and standardization are not rerun. Each config seeds once at 0 and then runs the original ten continuous `StratifiedKFold(shuffle=True, random_state=10)` folds.

## Audited SPromptGL path

The upstream repository is [tangaii/SPromptGL](https://github.com/tangaii/SPromptGL); the audited local checkout commit and SHA-256 values are frozen in `manifest.json`. The original MICCAI 2025 paper's Figure 1, Sections 2.1–2.2, Equations (1)–(7), ABIDE results and ablations were read alongside the actual Python sources and shell command. Root sources are copied into `.runtime/source` and hash-checked. The private `network.py` copy changes only `PLN.forward` to call an instance-bound method. GFR, its four patient graphs, GraphLearn thresholding, GCN propagation, the `i==1` overwrite, four GraphConstruct losses, input encoders, CAM, and OutputLayer remain from the source. The original PLN and its frozen CLIP cache are still constructed in the historical order, so initialization side effects are retained. The method binding then bypasses both SPGS and RES.

E0 flattens `[871,4,36]` CAM output directly to the original OutputLayer. Its full ten-fold training is checked against Experiment32 B0-A using fold metrics, initial trunk/RNG hashes, and selected epochs. The original simple-2 loss is reproduced as weighted NLL plus the arithmetic mean of the four GraphConstructLoss terms; ALoss is computed by untouched GFR but excluded from backpropagation.

## HASE inputs

The 36-wide CAM slots are mixed latent features. They are never treated as raw modalities or named brain regions. HASE also reads the original raw input's CORR block at columns 64:320. For expert k, both raw signed FC and a robust deviation `(FC - train_median) / (1.4826 * train_MAD + 1e-8)` are multiplied by its frozen binary mask before a bias-free encoder. Median/MAD are fit only on the outer training rows. This describes relative position in the training cohort and is not a clinical abnormality score.

The router evaluates each expert separately using CAM context, that expert's encoded FC response, 26 age-bin indicators, 2 sex indicators, and 16 ANAT/FUNC QC values. It does not directly receive the 20 PHENO site indicators. ANAT/FUNC are reliability metadata, not symptoms. The main GFR trunk retains all four original modalities as before.

Each expert adapter consumes the 144-dimensional CAM context and its own 32-dimensional mechanism response. The final projection is exactly zero initialized; the fused residual is bounded to at most 10% of the patient's CAM norm. There is no auxiliary diagnosis head. New parameters are an independent Adam group with zero weight decay; original parameter groups keep 0.11 weight decay.

`L_spec` is the mean squared cosine among centered expert residuals on outer-training rows, with weight 0.01. It is zero-safe and is removed only in E8. It is a representation-diversity bias, not supervision for real ASD subtypes.

## Frozen controls

The specialization penalty is enabled in E9 only. E8 disables it with the otherwise identical full anatomical model. E0–E7 do not use this penalty.

E1 is a single unmasked adapter with 183 hidden units. Its 154,452 parameters are within 0.236% of HASE's 154,816. E2 uses all 256 connections for each of four unconstrained generic experts. E5 applies one global seed-39 permutation over the 252 anatomically verified CORR slots, preserving each mask size and all overlap counts while scrambling membership. E6 clusters training-only FC correlation profiles over those same 252 verified slots and reassigns features while preserving the four masks' membership-pattern counts. The four unknown endpoint slots (195, 225, 232, 236) remain in the trunk and are excluded from true, wrong-anatomy and data-driven mechanism masks; E2 reads them as ordinary unlabelled numeric inputs. This restriction was frozen before formal training. E7 independently deranges FC rows within train and test, never across the split. E3 uses equal weights; E4 routes on CAM context and learned expert identity only while its experts still read their own real FC masks. E8 and E9 differ only by the specialization loss.

The four real masks are copied from the audited Experiment19 D1-D4 mapping and checked against Experiment11's numerical ROI mapping. The masks contain 72/56/45/32 edges, overlap, and jointly cover 165 mapped features. Indices 195, 225, 232 and 236 have unverified endpoints and are excluded from all anatomy-specific masks while remaining in the trunk's original 320-dimensional input; the unrestricted E2 control reads them as unlabeled numeric features.

## Evaluation protocol and limitations

Only ACC, hard one-hot ROC-AUC, SPE-TD and SEN-ASD are formal classification metrics. The hard AUC is the historical implementation; with two classes it equals balanced accuracy. We do not add probability AUC, F1 or a new classifier.

The historical checkpoint rule uses `val_idx=test_idx`, strict ACC improvement, 450 epochs and patience 200. The outer-test fold therefore selects the reported checkpoint. RFE/standardization was performed before the outer CV, and the patient graph is transductive over all 871 rows. Results are internal comparisons under that inherited protocol, not unbiased generalization estimates. Ten overlapping folds are not ten independent cohorts; one seed cannot establish cross-seed stability.

## Run and recovery

All execution is staged under `/tmp/SPromptGL-E39-HASE-stage/experiment/experiment39_heterogeneity_aware_semantic_experts`.

```bash
python run.py prepare
CUDA_VISIBLE_DEVICES=0 python run.py preflight
CUDA_VISIBLE_DEVICES=0 python run.py dispatch
```

Preflight and the dispatch parent expose only GPU0 to match Experiment32's single-visible-GPU RNG scope. `dispatch` trains all ten E0 folds continuously and blocks the other groups unless every E0 metric, best epoch, epoch count, trunk initialization hash and entry RNG hash matches Experiment32 B0-A fold by fold. It then starts the remaining five assigned config processes per GPU with no more than four active workers per card and two CPU threads per worker; each child sets its own physical `CUDA_VISIBLE_DEVICES`. Every 20 epochs, an atomic checkpoint stores current and best model states, both Adam states, wait/best-ACC state, and Python/NumPy/PyTorch CPU/CUDA RNG. A completed fold is never repeated on resume.

E9 fixed-checkpoint interventions are run after all ten training jobs finish. The report, formal CSVs, diagnostic CSV, counterfactual CSV and manifest are generated only after all 100 folds are uniquely present. Do not tune from the test folds or alter the frozen prompts/masks after seeing results.
