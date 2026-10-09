#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/adodas/SPromptGL-main"
EXP="$ROOT/experiment/experiment5_uniform_alignment_transfer"
SRC="$EXP/src"
PYTHON="${PYTHON:-/root/venvs/111/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$SRC:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$EXP/logs" "$EXP/raw" "$EXP/results"

jobs=(
  "strict S0_STRICT_BASELINE 0 none 0.00"
  "strict S1_STRICT_UNIFORM_GLOBAL 0 hybrid 0.50"
  "strict S0_STRICT_BASELINE 1 none 0.00"
  "strict S1_STRICT_UNIFORM_GLOBAL 1 hybrid 0.50"
  "strict S0_STRICT_BASELINE 2 none 0.00"
  "strict S1_STRICT_UNIFORM_GLOBAL 2 hybrid 0.50"
  "legacy L0_LEGACY_BASELINE 0 none 0.00"
  "legacy L1_LEGACY_UNIFORM_GLOBAL 0 hybrid 0.50"
  "legacy L0_LEGACY_BASELINE 1 none 0.00"
  "legacy L1_LEGACY_UNIFORM_GLOBAL 1 hybrid 0.50"
  "legacy L0_LEGACY_BASELINE 2 none 0.00"
  "legacy L1_LEGACY_UNIFORM_GLOBAL 2 hybrid 0.50"
)

run_job() {
  local protocol="$1" config="$2" seed="$3" mode="$4" lambda="$5"
  local prefix="${protocol}_${config}_seed${seed}"
  local log="$EXP/logs/${prefix}.log"
  local folds="$EXP/raw/${config}_seed${seed}.folds.csv"
  local oof="$EXP/raw/${config}_seed${seed}.oof.csv"
  local gpu="$6"
  local script="$SRC/main_${protocol}.py"
  echo "START $prefix GPU=$gpu" >&2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$script" \
    --config "$config" --seed "$seed" --nepoch 450 --early 200 \
    --align_mode "$mode" --align_lambda "$lambda" \
    --align_temperature 0.10 --align_warmup 20 \
    --fold_results "$folds" --oof_results "$oof" \
    >"$log" 2>&1 &
  LAST_PID=$!
}

for ((start=0; start<${#jobs[@]}; start+=4)); do
  pids=()
  for ((offset=0; offset<4 && start+offset<${#jobs[@]}; offset++)); do
    read -r protocol config seed mode lambda <<<"${jobs[$((start+offset))]}"
    gpu=$(( (start + offset) % 2 ))
    run_job "$protocol" "$config" "$seed" "$mode" "$lambda" "$gpu"
    pids+=("$LAST_PID")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

"$PYTHON" "$SRC/aggregate.py" --exp "$EXP" --root "$ROOT"
