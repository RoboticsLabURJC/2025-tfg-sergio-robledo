#!/bin/bash
set -euo pipefail

DATASET_BASE="../datasets"
TEST_BASE="../datasets/test"
VALID_BASE="../datasets/validation"

TRAIN_SCRIPT="train_weights_turns.py"
EXPERIMENT_NAME="exp_weights_estados_$(date +%s)"

DATA_DIRS=()
echo "Buscando datasets de TRAIN..."
for dir in "$DATASET_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Train: $dir"
    DATA_DIRS+=(--data_dir "$dir")
  fi
done

TEST_DIRS=()
echo "Buscando datasets de TEST..."
for dir in "$TEST_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Test: $dir"
    TEST_DIRS+=(--test_dir "$dir")
  fi
done

VALID_DIRS=()
echo "Buscando datasets de VALIDACIÓN..."
for dir in "$VALID_BASE"/Deepracer_BaseMap_*; do
  if [ -d "$dir" ]; then
    echo "Val: $dir"
    VALID_DIRS+=(--val_dir "$dir")
  fi
done

if [ ${#DATA_DIRS[@]} -eq 0 ]; then
  echo "No se encontraron carpetas de TRAIN en $DATASET_BASE/Deepracer_BaseMap_*"
  exit 1
fi

if [ ${#VALID_DIRS[@]} -eq 0 ]; then
  echo "No se encontraron carpetas de VALIDACIÓN en $VALID_BASE/Deepracer_BaseMap_* (requerido)."
  exit 1
fi

echo ""
echo "Ejecutando:"
echo "  Script      : $TRAIN_SCRIPT"
echo "  Experimento : $EXPERIMENT_NAME"
echo ""

python3 "$TRAIN_SCRIPT" \
  "${DATA_DIRS[@]}" \
  "${VALID_DIRS[@]}" \
  "${TEST_DIRS[@]}" \
  --num_epochs 120 \
  --batch_size 128 \
  --lr 3e-4 \
  --base_dir "$EXPERIMENT_NAME" \
  --drop_unlabeled \
  --print_terminal
