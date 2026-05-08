#!/bin/bash

set -e

cd "$(dirname "$0")" || exit 1
cd .. || exit 1

MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
FOLDS="${FOLDS:-fold1 fold2 fold3 fold4 fold5}"
STUDY_MODE="${STUDY_MODE:-cv_mean}" # cv_mean or per_fold
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

N_TRIALS=${1:-20}
STUDY_NAME_DEFAULT="cmdc_${MODEL_FAMILY}_${PROMPT_MODE}_${STUDY_MODE}_hpo_$(date +%Y%m%d_%H%M%S)"
STUDY_NAME=${2:-$STUDY_NAME_DEFAULT}
CMDC_ROOT="${CMDC_ROOT:-$(pwd)/data/cmdc}"
SAVE_ROOT="${SAVE_ROOT:-output_model/optuna_cmdc_5fold_hpo/${PROMPT_MODE}/optuna_cmdc_cv_5fold_${STUDY_MODE}}"
STORAGE_PATH="${STORAGE_PATH:-optuna_studies/optuna_cmdc_5fold_${STUDY_MODE}}"
LOG_DIR="${LOG_DIR:-logs/optuna_cmdc_cv_5fold_${STUDY_MODE}}"
NUM_GPUS="${NUM_GPUS:-4}"
PRUNING_FLAG="--disable-pruning"
case "${ENABLE_PRUNING,,}" in
    1|true|yes|on)
        PRUNING_FLAG="--enable-pruning"
        ;;
esac

echo "======================================"
echo "CMDC 5-Fold Cross-Validated HPO"
echo "======================================"
echo "Model Family: $MODEL_FAMILY"
echo "Prompt Mode: $PROMPT_MODE"
echo "Study Mode: $STUDY_MODE"
echo "Number of Trials: $N_TRIALS"
echo "Enable Pruning: $ENABLE_PRUNING"
echo "Pruner Startup Trials: $PRUNER_STARTUP_TRIALS"
echo "Pruner Warmup Steps: $PRUNER_WARMUP_STEPS"
echo "Pruner Interval Steps: $PRUNER_INTERVAL_STEPS"
echo "Study Name: $STUDY_NAME"
echo "Folds: $FOLDS"
echo "CMDC Root: $CMDC_ROOT"
echo "Storage Path: $STORAGE_PATH"
echo "Save Root: $SAVE_ROOT"
echo "Log Dir: $LOG_DIR"
echo ""

if [ -d "/gpfs/projects/etur92" ]; then
    echo "Detected MN5 cluster. Submitting SLURM job..."
    echo ""

    MODEL_FAMILY="$MODEL_FAMILY" \
    PROMPT_MODE="$PROMPT_MODE" \
    N_TRIALS="$N_TRIALS" \
    ENABLE_PRUNING="$ENABLE_PRUNING" \
    PRUNER_STARTUP_TRIALS="$PRUNER_STARTUP_TRIALS" \
    PRUNER_WARMUP_STEPS="$PRUNER_WARMUP_STEPS" \
    PRUNER_INTERVAL_STEPS="$PRUNER_INTERVAL_STEPS" \
    STUDY_NAME="$STUDY_NAME" \
    FOLDS="$FOLDS" \
    STUDY_MODE="$STUDY_MODE" \
    CMDC_ROOT="$CMDC_ROOT" \
    SAVE_ROOT="$SAVE_ROOT" \
    STORAGE_PATH="$STORAGE_PATH" \
    LOG_DIR="$LOG_DIR" \
    NUM_GPUS="$NUM_GPUS" \
    sbatch optuna_hpo/train_hpo_cmdc_cv_5fold.slurm

    echo "SLURM job submitted! Check ${LOG_DIR} for progress."
else
    echo "Not on MN5 cluster. Running locally (requires 4 GPUs)..."
    echo ""

    mkdir -p "$LOG_DIR" "$SAVE_ROOT" "$STORAGE_PATH"

    if [ "$MODEL_FAMILY" = "audio" ]; then
        MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
    else
        MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
    fi

    export MODEL_PATH
    export CMDC_ROOT
    export SAVE_ROOT
    export STORAGE_PATH
    export LOG_DIR
    export FOLDS
    export NUM_GPUS
    export STUDY_MODE
    export MODEL_FAMILY
    export PROMPT_MODE

    python optuna_hpo/hpo_cv_5fold.py \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage-path "$STORAGE_PATH" \
        --cmdc-root "$CMDC_ROOT" \
        --folds "$FOLDS" \
        --save-root "$SAVE_ROOT" \
        --num-gpus "$NUM_GPUS" \
        --model-family "$MODEL_FAMILY" \
        --prompt-mode "$PROMPT_MODE" \
        --study-mode "$STUDY_MODE" \
        --pruner-startup-trials "$PRUNER_STARTUP_TRIALS" \
        --pruner-warmup-steps "$PRUNER_WARMUP_STEPS" \
        --pruner-interval-steps "$PRUNER_INTERVAL_STEPS" \
        "$PRUNING_FLAG"
fi

echo ""
echo "Results will be saved in: ${STORAGE_PATH}/${STUDY_NAME}_results.json"
