#!/bin/bash

set -e

cd "$(dirname "$0")" || exit 1
cd .. || exit 1

DATASET_NAME="${DATASET_NAME:-merged}" # merged, daic_woz, eatd
MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
TASK_VARIANT="${TASK_VARIANT:-default}" # default or filtered
ENABLE_PRUNING="${ENABLE_PRUNING:-1}"
PRUNER_STARTUP_TRIALS="${PRUNER_STARTUP_TRIALS:-5}"
PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-2}"
PRUNER_INTERVAL_STEPS="${PRUNER_INTERVAL_STEPS:-1}"

case "${MODEL_FAMILY}:${PROMPT_MODE}" in
    audio:full|audio:audiotext|text:textonly)
        ;;
    *)
        echo "Invalid MODEL_FAMILY / PROMPT_MODE combination: ${MODEL_FAMILY} + ${PROMPT_MODE}"
        echo "Allowed combinations are: audio+full, audio+audiotext, text+textonly"
        exit 1
        ;;
esac

case "$DATASET_NAME" in
    merged|daic_woz|eatd)
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME"
        echo "Use DATASET_NAME=merged, DATASET_NAME=daic_woz, or DATASET_NAME=eatd"
        exit 1
        ;;
esac

N_TRIALS=${1:-20}
STUDY_NAME_DEFAULT="${DATASET_NAME}_${MODEL_FAMILY}_${PROMPT_MODE}_hpo_$(date +%Y%m%d_%H%M%S)"
STUDY_NAME=${2:-$STUDY_NAME_DEFAULT}
STORAGE_PATH="${STORAGE_PATH:-optuna_studies/optuna_${DATASET_NAME}}"
SAVE_PATH="${SAVE_PATH:-output_model/optuna_${DATASET_NAME}_hpo/${PROMPT_MODE}}"
LOG_DIR="${LOG_DIR:-logs/optuna_${DATASET_NAME}}"
PRUNING_FLAG="--disable-pruning"
case "${ENABLE_PRUNING,,}" in
    1|true|yes|on)
        PRUNING_FLAG="--enable-pruning"
        ;;
esac

echo "======================================"
echo "Single-Dataset Optuna Hyperparameter Search"
echo "======================================"
echo "Dataset Name: $DATASET_NAME"
echo "Model Family: $MODEL_FAMILY"
echo "Prompt Mode: $PROMPT_MODE"
echo "Task Variant: $TASK_VARIANT"
echo "Number of Trials: $N_TRIALS"
echo "Enable Pruning: $ENABLE_PRUNING"
echo "Pruner Startup Trials: $PRUNER_STARTUP_TRIALS"
echo "Pruner Warmup Steps: $PRUNER_WARMUP_STEPS"
echo "Pruner Interval Steps: $PRUNER_INTERVAL_STEPS"
echo "Study Name: $STUDY_NAME"
echo "Storage Path: $STORAGE_PATH"
echo "Save Path: $SAVE_PATH"
echo "Log Dir: $LOG_DIR"
echo ""

if [ -d "/gpfs/projects/etur92" ]; then
    echo "Detected MN5 cluster. Submitting SLURM job..."
    echo ""

    DATASET_NAME="$DATASET_NAME" \
    MODEL_FAMILY="$MODEL_FAMILY" \
    PROMPT_MODE="$PROMPT_MODE" \
    TASK_VARIANT="$TASK_VARIANT" \
    N_TRIALS="$N_TRIALS" \
    ENABLE_PRUNING="$ENABLE_PRUNING" \
    PRUNER_STARTUP_TRIALS="$PRUNER_STARTUP_TRIALS" \
    PRUNER_WARMUP_STEPS="$PRUNER_WARMUP_STEPS" \
    PRUNER_INTERVAL_STEPS="$PRUNER_INTERVAL_STEPS" \
    STUDY_NAME="$STUDY_NAME" \
    STORAGE_PATH="$STORAGE_PATH" \
    SAVE_PATH="$SAVE_PATH" \
    LOG_DIR="$LOG_DIR" \
    sbatch optuna_hpo/train_hpo.slurm

    echo "SLURM job submitted! Check ${LOG_DIR} for progress."
else
    echo "Not on MN5 cluster. Running locally (requires 4 GPUs)..."
    echo ""

    mkdir -p "$LOG_DIR" "$SAVE_PATH" "$STORAGE_PATH"

    python optuna_hpo/hpo.py \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage-path "$STORAGE_PATH" \
        --save-root "$SAVE_PATH" \
        --dataset-name "$DATASET_NAME" \
        --model-family "$MODEL_FAMILY" \
        --prompt-mode "$PROMPT_MODE" \
        --task-variant "$TASK_VARIANT" \
        --pruner-startup-trials "$PRUNER_STARTUP_TRIALS" \
        --pruner-warmup-steps "$PRUNER_WARMUP_STEPS" \
        --pruner-interval-steps "$PRUNER_INTERVAL_STEPS" \
        "$PRUNING_FLAG"
fi

echo ""
echo "Results will be saved in: ${STORAGE_PATH}/${STUDY_NAME}_results.json"
