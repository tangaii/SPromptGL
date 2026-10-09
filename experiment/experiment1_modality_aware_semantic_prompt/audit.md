# Code and Data Audit

## Scope and provenance

The audited baseline is the project-root snapshot copied to `src/reference/` before experiment implementation. The ABIDE command is `ABIDE-simple-2-concat-weighted-cosine.sh`; the local paper copy is `2025 MICCAI SPromptGL、Semantic Prompt Guided Graph Learning for Multi-modal Brain Disease.pdf`. The paper's official MICCAI open-access copy was used to verify the published method and implementation section when this environment lacked a local PDF text extractor.

## ABIDE inputs

`modal_feat_dict.npy` defines four ordered blocks, and all block names and positions exactly match `processed_standard_data.csv`:

| Block | Dimensions | Actual fields |
|---|---:|---|
| PHENO | 48 | 26 one-hot age bins, 20 one-hot sites, 2 one-hot sex categories |
| ANAT | 6 | `anat_cnr`, `anat_efc`, `anat_fber`, `anat_fwhm`, `anat_qi1`, `anat_snr` |
| FUNC | 10 | `func_efc`, `func_fber`, `func_fwhm`, `func_dvars`, `func_outlier`, `func_quality`, `func_mean_fd`, `func_num_fd`, `func_perc_fd`, `func_gsr` |
| Correlation | 256 | RFE-selected values from Fisher-transformed pairwise rs-fMRI correlations; original selected edge IDs were not retained |

The CSV has 871 rows, 320 predictors, and a final `label` column. Raw `DX_GROUP=1` (403 people) is ASD and `DX_GROUP=2` (468 people) is NC; subtracting one yields ASD class 0 and NC class 1. The model's ASD probability column is therefore column 0.

ANAT/FUNC are primarily image-quality, signal, motion, and preprocessing measures, not regional structural or functional pathology maps. PHENO includes acquisition site, a collection-context variable. Correlation features are selected edges, not 256 identified ROIs.

## Preprocessing provenance and limitations

The phenotype notebook one-hot encodes age, site, and sex. It applies `StandardScaler.fit_transform` to ANAT and FUNC over the full cohort before exporting the final table. The connectivity notebook starts from 111 Harvard–Oxford time series (6,105 upper-triangle edges), runs a five-fold stratified splitter, then uses the final loop's `train_index` for one RFE selection applied to all participants. That selected matrix is then concatenated with the phenotype/quality table. The selected original edge indices are not preserved in the final feature dictionary. Both operations precede this experiment's ten-fold CV. They remain potential preprocessing leakage by explicit frozen-data requirement and are not described as fold-wise clean.

Input SHA256 values at audit time:

- `processed_standard_data.csv`: `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`
- `modal_feat_dict.npy`: `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`

## Baseline prompt path

Root `network.py` defines eight `attr_words` strings, with the first four grouped into one semantic vector and the next four into a second. RN50 CLIP text features have shape `[8, 1024]`; each group is flattened to 4096 dimensions, passed through a shared `4096 -> 128 -> 9` ReLU MLP, and compared against each local 9-D token. On ABIDE, PLN reshapes the feature stream into `[871, 4, 4, 9]`, normalizes tokens and semantic vectors, calculates cosine similarity, applies a two-entry softmax, selects the maximum weight, scales each local token, and uses the existing pooled global/local fusion. The first-four/last-four prompts are not actual ASD/NC labels.

The experiment uses the same CLIP RN50 implementation and validated cached weights. The cached RN50 SHA256 equals the checksum embedded in the official model URL. CLIP tokens have a 77-token context; all audited prompts are checked before encoding.

## Training and evaluation findings in root code

- `EvalHelper` sets validation indices equal to test indices; each epoch's best checkpoint is therefore selected on the outer test fold.
- AUC is calculated from one-hot hard predictions rather than positive-class probabilities.
- `main.py` can stop after five folds when cumulative test accuracy is below 0.6.
- `GFR` receives all labels but masks non-training labels before making its label target. Its unknown-label block is shared. `ALoss` is returned and printed, but `simple-2` does not include it in the optimized loss; the actual optimized objective is classification NLL plus the average four graph regularizers.
- Input features for all 871 subjects flow through the graphs and PLN on each step. This is transductive feature propagation, not inductive evaluation.
- The root loop seeds the run once, not each fold. It applies `my_weight_init` to linear layers after PLN construction. Experiment workers reset each split/seed and copy one canonical initial trainable state across P0-P3.
- Root checkpoints and diagnostic outputs are not a suitable per-fold artifact format; this experiment writes isolated fold checkpoints and predictions.

## Parameter source

The required local shell sets `GC_mode=weighted-cosine`, `MF_mode=concat`, `MP_mode=GCN`, `mode=simple-2`, dropout `0.35`, learning rate `0.0038`, `n_head=2`, `n_hidden=18`, `nlayer=1`, and weight decay `0.11`. Root `main.py` defaults the epoch cap to 450 and patience to 200. The paper's implementation paragraph reports different ABIDE settings (dropout 0.75, learning rate 0.0023, 4 attention heads, 10 hidden layers, and 4 encoding layers); this run follows the explicitly requested shell, not those differing paragraph values.

## Runtime environment

Python 3.12.3, PyTorch 2.9.0, two PPU-ZW810E devices exposed through the CUDA API, 96 GiB reported per device. The cached CLIP RN50 text encoder loaded successfully and produced finite `[8, 1024]` outputs. `/root/venvs/111/bin/python` resolves to the same Python 3.12.3 installation. No root model/data files are edited by the experiment.
