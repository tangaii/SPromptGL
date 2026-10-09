# Experiment 5: Uniform Global Alignment Transfer to SPromptGL

This directory contains the isolated Experiment 5 implementation described in
the repository-level `prompt` file. The original SPromptGL files under the
repository root are not modified.

The registered configurations are:

- `S0_STRICT_BASELINE` and `S1_STRICT_UNIFORM_GLOBAL`
- `L0_LEGACY_BASELINE` and `L1_LEGACY_UNIFORM_GLOBAL`

Each configuration runs seeds 0, 1 and 2 with ten folds. Strict evaluation
uses an outer stratified fold plus a 10% inner validation split and per-fold
RNG reset. Legacy evaluation restores the original fixed `random_state=10`,
`val=test`, hard-AUC checkpoint-selection protocol and sequential fold RNG.

Alignment is copied byte-for-byte from MMGL Experiment 4:

`SOURCE_ALIGNMENT_SHA256 = 7c39a2ebdc136f66c00ac698975803119c57df31110a96bceece9cb7cec800cc`

The alignment is trained only on pre-prompt `[N,4,36]` tokens with six equal
modality pairs, temperature `0.10`, λ `0.50`, and a 20-epoch warmup. The
original `ALoss` is recorded but is not added to the optimisation objective.

Data SHA256 matches the MMGL Experiment 4 ABIDE data:

- `processed_standard_data.csv`: `b58317f1c37fc60a957f946a32dc68dcaa79d46b72d77670b1f5b53d8db57fe1`
- `modal_feat_dict.npy`: `8fbc23f7933b831ad77693fd75e3bca3f295f7efa9fbe5e19b8a1b7abf6f041e`

Formal execution uses `/root/venvs/111/bin/python`, PyTorch 2.9.0, two GPUs,
`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and at most four concurrent jobs.
SPromptGL source commit: `ce983b85a36d19b5d22340d1cee101add0a0a060`.
The RN50 CLIP checkpoint is loaded through the original local CLIP code at
`/root/.cache/clip/RN50.pt` with SHA256
`afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`.
The only added dependency was `ftfy==6.1.1`.

Run the complete experiment with:

```bash
./run_all.sh
```

Machine-readable outputs are written to `results/`, with one job log per
configuration and seed under `logs/`.
