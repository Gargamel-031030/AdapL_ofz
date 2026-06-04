#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/data}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/results/min_grid}"
GPU="${GPU:-0}"

GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-50}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
CLIENT_FRACTION="${CLIENT_FRACTION:-0.8}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SEED="${SEED:-41}"
DELTA="${DELTA:-1e-5}"
EPSILONS=(${EPSILONS:-16 8 4})
LRS=(${LRS:-0.01 0.005})
LOCAL_STEPS_GRID=(${LOCAL_STEPS_GRID:-5 10})
CLIPPING_NORMS=(${CLIPPING_NORMS:-0.1 0.5 1.0})

mkdir -p "$RESULT_ROOT"

sanitize_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  echo "$value"
}

echo "Min grid started at $(date)"
echo "DATA_DIR=$DATA_DIR"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "GPU=$GPU"
echo "GLOBAL_ROUNDS=$GLOBAL_ROUNDS"
echo "EPSILONS=${EPSILONS[*]}"
echo "LRS=${LRS[*]}"
echo "LOCAL_STEPS_GRID=${LOCAL_STEPS_GRID[*]}"
echo "CLIPPING_NORMS=${CLIPPING_NORMS[*]}"

for epsilon in "${EPSILONS[@]}"; do
  for lr in "${LRS[@]}"; do
    for local_steps in "${LOCAL_STEPS_GRID[@]}"; do
      for clipping_norm in "${CLIPPING_NORMS[@]}"; do
        eps_tag="$(sanitize_tag "$epsilon")"
        lr_tag="$(sanitize_tag "$lr")"
        clip_tag="$(sanitize_tag "$clipping_norm")"
        tag="min_cifar100_alpha${DIRICHLET_ALPHA}_k${NUM_CLIENTS}_sr${CLIENT_FRACTION}_steps${local_steps}_b${BATCH_SIZE}_lr${lr_tag}_eps${eps_tag}_clip${clip_tag}_r${GLOBAL_ROUNDS}"
        run_dir="$RESULT_ROOT/$tag"
        mkdir -p "$run_dir"

        echo
        echo "===== Running $tag at $(date) ====="

        CUDA_VISIBLE_DEVICES="$GPU" python -u main.py \
          --method Min \
          --epsilon_min "$epsilon" \
          --delta "$DELTA" \
          --clipping_norm "$clipping_norm" \
          --data_dir "$DATA_DIR" \
          --output_csv "$run_dir/${tag}.csv" \
          --run_config_json "$run_dir/${tag}_config.json" \
          --client_distribution_csv "$run_dir/${tag}_client_dist.csv" \
          --client_distribution_json "$run_dir/${tag}_client_dist.json" \
          --partition dirichlet \
          --dirichlet_alpha "$DIRICHLET_ALPHA" \
          --global_rounds "$GLOBAL_ROUNDS" \
          --local_update_mode random-batch \
          --local_steps "$local_steps" \
          --num_clients "$NUM_CLIENTS" \
          --client_fraction "$CLIENT_FRACTION" \
          --batch_size "$BATCH_SIZE" \
          --test_batch_size "$TEST_BATCH_SIZE" \
          --num_workers "$NUM_WORKERS" \
          --lr "$lr" \
          --momentum "$MOMENTUM" \
          --weight_decay "$WEIGHT_DECAY" \
          --seed "$SEED" \
          2>&1 | tee "$run_dir/${tag}.log"

        python scripts/summarize_min_grid.py "$RESULT_ROOT" \
          > "$RESULT_ROOT/summary.csv"
        echo "Updated summary: $RESULT_ROOT/summary.csv"
      done
    done
  done
done

python scripts/summarize_min_grid.py "$RESULT_ROOT" \
  > "$RESULT_ROOT/summary.csv"

echo
echo "Min grid finished at $(date)"
echo "Summary: $RESULT_ROOT/summary.csv"
