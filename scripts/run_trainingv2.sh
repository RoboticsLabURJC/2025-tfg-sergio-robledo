#!/bin/bash

# Carpeta base donde están los datos
DATASET_BASE="./datasets"
TEST_BASE="./datasets/test"
VALID_BASE="./datasets/validation"
# Script de entrenamiento
TRAIN_SCRIPT="train_v3.py"

# Otros parámetros
EXPERIMENT_NAME="exp_debug_$(date +%s)"

# Construir lista de --data_dir para cada subcarpeta Deepracer_BaseMap_*
DATA_DIRS=""
echo "🔎 Buscando datasets..."
for dir in "$DATASET_BASE"/Deepracer_BaseMap_*; do
    if [ -d "$dir" ]; then
        echo "📂 Añadiendo: $dir"
        DATA_DIRS+="--data_dir $dir "
    fi
done

TEST_DIRS=""
for dir in "$TEST_BASE"/Deepracer_BaseMap_*; do
    if [ -d "$dir" ]; then
        echo "📂 Añadiendo test: $dir"
        TEST_DIRS+="--test_dir $dir "
    fi
done


VALID_DIRS=""
for dir in "$VALID_BASE"/Deepracer_BaseMap_*; do
    if [ -d "$dir" ]; then
      echo "✅ Val: $dir"
      VALID_DIRS+=" --val_dir $dir " 
    fi
done


echo ""
echo "🚀 Iniciando entrenamiento con:"
echo "   Epochs     : 50"
echo "   Batch size : 128"
echo "   Experimento: $EXPERIMENT_NAME"
echo ""


# Ejecutar entrenamiento
python $TRAIN_SCRIPT \
  $DATA_DIRS \
  $TEST_DIRS \
  $VALID_DIRS \
  --num_epochs 50 \
  --batch_size 128 \
  --base_dir "$EXPERIMENT_NAME" \
  --comment "Depuración con impresión y TensorBoard" \
  --print_terminal True \
  --mirrored_imgs
  #--resume_epoch 35