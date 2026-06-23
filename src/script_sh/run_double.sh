#!/usr/bin/env bash

set -euo pipefail

# Run from src/: bash script_sh/run_double.sh

MODE="ga"

# Format: "baseline fitness_type"
EXPERIMENTS=(
  "GA normal"
  "GA adaptive"
  "NSGAII normal"
)

NUM_SAMPLES=100
N_ITER=1000
PATCH_SIZE=16
POP_SIZE=100
PROB_MUTATE_PATCH=0.9
PROB_MUTATE_LOCATION=0.2
TOUR_SIZE=4

MODELS=(
  "restnet_vggface"
  "arcface_ms1mv3"
  "cosface_glint360k"
)

LABELS=(0 1)

# Format: "attack_w recons_w"
WEIGHT_PAIRS=(
  "1.0 0.0"
  "0.5 0.5"
)

LOG_DIR="../output/run_logs"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  for label in "${LABELS[@]}"; do
    for wr in "${WEIGHT_PAIRS[@]}"; do
      read -r attack_w recons_w <<< "$wr"
      for exp in "${EXPERIMENTS[@]}"; do
        read -r baseline fitness_type <<< "$exp"

        run_tag="double_${baseline}_${fitness_type}_${model}_lb${label}_aw${attack_w}_rw${recons_w}_$(date +%Y%m%d_%H%M%S)"
        log_file="${LOG_DIR}/${run_tag}.log"

        echo "============================================================"
        echo "Running: baseline=${baseline}, fitness_type=${fitness_type}, model=${model}, label=${label}, attack_w=${attack_w}, recons_w=${recons_w}"
        echo "Log: ${log_file}"

        python main_double_img.py \
          --mode "$MODE" \
          --baseline "$baseline" \
          --fitness_type "$fitness_type" \
          --model_name "$model" \
          --label "$label" \
          --num_samples "$NUM_SAMPLES" \
          --n_iter "$N_ITER" \
          --patch_size "$PATCH_SIZE" \
          --pop_size "$POP_SIZE" \
          --prob_mutate_patch "$PROB_MUTATE_PATCH" \
          --prob_mutate_location "$PROB_MUTATE_LOCATION" \
          --tourament_size "$TOUR_SIZE" \
          --attack_w "$attack_w" \
          --recons_w "$recons_w" \
          2>&1 | tee "$log_file"
      done
    done
  done
done

echo "All double-image runs completed."

