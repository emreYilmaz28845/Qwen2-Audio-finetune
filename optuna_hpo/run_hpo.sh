#!/bin/bash

set -e

cd "$(dirname "$0")" || exit 1
cd .. || exit 1

DATASET_NAME="${DATASET_NAME:-merged}" # merged, daic_woz, eatd
MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
TASK_VARIANT="${TASK_VARIANT:-default}" # default or filtered
DAIC_WOZ_EVAL_LEVEL="${DAIC_WOZ_EVAL_LEVEL:-${DAIC_EVAL_LEVEL:-person}}"
DAIC_WOZ_EVAL_MODE="${DAIC_WOZ_EVAL_MODE:-${DAIC_EVAL_MODE:-majority_vote}}"
DAIC_WOZ_PERSON_THRESHOLD="${DAIC_WOZ_PERSON_THRESHOLD:-${DAIC_PERSON_THRESHOLD:-0.5}}"
EATD_EVAL_LEVEL="${EATD_EVAL_LEVEL:-person}"
EATD_EVAL_MODE="${EATD_EVAL_MODE:-majority_vote}"
EATD_PERSON_THRESHOLD="${EATD_PERSON_THRESHOLD:-0.5}"
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

DATASET_LEVEL_SUFFIX=""
if [ "$DATASET_NAME" = "daic_woz" ]; then
    DATASET_LEVEL_SUFFIX="_${DAIC_WOZ_EVAL_LEVEL}"
elif [ "$DATASET_NAME" = "eatd" ]; then
    DATASET_LEVEL_SUFFIX="_${EATD_EVAL_LEVEL}"
fi

N_TRIALS=${1:-20}
STUDY_TIMESTAMP="${STUDY_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
STUDY_NAME_DEFAULT="Hpo_Study_${PROMPT_MODE}_${STUDY_TIMESTAMP}"
STUDY_NAME=${2:-$STUDY_NAME_DEFAULT}
STORAGE_PATH="${STORAGE_PATH:-optuna_studies/optuna_${DATASET_NAME}${DATASET_LEVEL_SUFFIX}}"
SAVE_PATH="${SAVE_PATH:-output_model/optuna_${DATASET_NAME}_hpo${DATASET_LEVEL_SUFFIX}}"
LOG_DIR="${LOG_DIR:-logs/optuna_${DATASET_NAME}${DATASET_LEVEL_SUFFIX}}"
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
echo "DAIC Eval Level: $DAIC_WOZ_EVAL_LEVEL"
echo "DAIC Eval Mode: $DAIC_WOZ_EVAL_MODE"
echo "DAIC Person Threshold: $DAIC_WOZ_PERSON_THRESHOLD"
echo "EATD Eval Level: $EATD_EVAL_LEVEL"
echo "EATD Eval Mode: $EATD_EVAL_MODE"
echo "EATD Person Threshold: $EATD_PERSON_THRESHOLD"
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
    DAIC_WOZ_EVAL_LEVEL="$DAIC_WOZ_EVAL_LEVEL" \
    DAIC_WOZ_EVAL_MODE="$DAIC_WOZ_EVAL_MODE" \
    DAIC_WOZ_PERSON_THRESHOLD="$DAIC_WOZ_PERSON_THRESHOLD" \
    DAIC_EVAL_LEVEL="$DAIC_WOZ_EVAL_LEVEL" \
    DAIC_EVAL_MODE="$DAIC_WOZ_EVAL_MODE" \
    DAIC_PERSON_THRESHOLD="$DAIC_WOZ_PERSON_THRESHOLD" \
    EATD_EVAL_LEVEL="$EATD_EVAL_LEVEL" \
    EATD_EVAL_MODE="$EATD_EVAL_MODE" \
    EATD_PERSON_THRESHOLD="$EATD_PERSON_THRESHOLD" \
    N_TRIALS="$N_TRIALS" \
    ENABLE_PRUNING="$ENABLE_PRUNING" \
    PRUNER_STARTUP_TRIALS="$PRUNER_STARTUP_TRIALS" \
    PRUNER_WARMUP_STEPS="$PRUNER_WARMUP_STEPS" \
    PRUNER_INTERVAL_STEPS="$PRUNER_INTERVAL_STEPS" \
    STUDY_NAME="$STUDY_NAME" \
    STORAGE_PATH="$STORAGE_PATH" \
    SAVE_PATH="$SAVE_PATH" \
    LOG_DIR="$LOG_DIR" \
    STUDY_TIMESTAMP="$STUDY_TIMESTAMP" \
    PRINT_PATHS_ONLY="${PRINT_PATHS_ONLY:-0}" \
    TRIAL_NUMBER="${TRIAL_NUMBER:-1}" \
    LR="${LR:-4e-05}" \
    BATCH_SIZE="${BATCH_SIZE:-1}" \
    LORA_R="${LORA_R:-8}" \
    LORA_ALPHA="${LORA_ALPHA:-16}" \
    sbatch optuna_hpo/train_hpo.slurm

    echo "SLURM job submitted! Check ${LOG_DIR} for progress."
else
    echo "Not on MN5 cluster. Running locally (requires 4 GPUs)..."
    echo ""

    mkdir -p "$LOG_DIR" "$SAVE_PATH" "$STORAGE_PATH"

    EXTRA_ARGS=()
    if [ "${PRINT_PATHS_ONLY:-0}" = "1" ]; then
        EXTRA_ARGS+=(--print-paths-only)
    fi

    python optuna_hpo/hpo.py \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage-path "$STORAGE_PATH" \
        --save-root "$SAVE_PATH" \
        --dataset-name "$DATASET_NAME" \
        --model-family "$MODEL_FAMILY" \
        --prompt-mode "$PROMPT_MODE" \
        --task-variant "$TASK_VARIANT" \
        --daic-eval-level "$DAIC_WOZ_EVAL_LEVEL" \
        --daic-eval-mode "$DAIC_WOZ_EVAL_MODE" \
        --daic-person-threshold "$DAIC_WOZ_PERSON_THRESHOLD" \
        --eatd-eval-level "$EATD_EVAL_LEVEL" \
        --eatd-eval-mode "$EATD_EVAL_MODE" \
        --eatd-person-threshold "$EATD_PERSON_THRESHOLD" \
        --study-timestamp "$STUDY_TIMESTAMP" \
        --trial-number "${TRIAL_NUMBER:-1}" \
        --lr "${LR:-4e-05}" \
        --batch-size "${BATCH_SIZE:-1}" \
        --lora-r "${LORA_R:-8}" \
        --lora-alpha "${LORA_ALPHA:-16}" \
        --pruner-startup-trials "$PRUNER_STARTUP_TRIALS" \
        --pruner-warmup-steps "$PRUNER_WARMUP_STEPS" \
        --pruner-interval-steps "$PRUNER_INTERVAL_STEPS" \
        "$PRUNING_FLAG" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "Results will be saved in: ${STORAGE_PATH}/${STUDY_NAME}_results.json"
