#!/bin/bash
# ==============================
# CMDC 5-fold Qwen2-Audio training launch script
# Runs one fold at a time with fold-specific train/test manifests
# ==============================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
CMDC_ROOT="${CMDC_ROOT:-$LOCAL_DIR/data/cmdc}"
FOLDS="${FOLDS:-1 2 3 4 5}"

TRAIN_STRATEGY="${TRAIN_STRATEGY:-ddp}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_STEP="${EVAL_STEP:-50}"
GRAD_ACCUMULATE_STEP="${GRAD_ACCUMULATE_STEP:-5}"
TRAIN_EPOCH="${TRAIN_EPOCH:-20}"
USE_BFLOAT16="${USE_BFLOAT16:-True}"
NUM_GPUS="${NUM_GPUS:-4}"
SAVE_PATH="${SAVE_PATH:-$LOCAL_DIR/output_model/cmdc_5fold}"

WANDB_ENABLED="${WANDB_ENABLED:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen2-audio-cmdc-5fold}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-cmdc}"
WANDB_GROUP="${WANDB_GROUP:-cmdc_5fold}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_STEP="${WANDB_LOG_STEP:-10}"

NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SUMMARY_JSON="${SUMMARY_JSON:-$SAVE_PATH/results_cmdc_5fold.json}"
SUMMARY_CSV="${SUMMARY_CSV:-$SAVE_PATH/results_cmdc_5fold.csv}"

mkdir -p "$SAVE_PATH"

echo "============================================"
echo "  CMDC 5-Fold Training"
echo "============================================"
echo "  MODEL_PATH : $MODEL_PATH"
echo "  CMDC_ROOT  : $CMDC_ROOT"
echo "  FOLDS      : $FOLDS"
echo "  LR         : $LR"
echo "  NUM_GPUS   : $NUM_GPUS"
echo "============================================"

for FOLD in $FOLDS; do
    FOLD_NAME="fold${FOLD}"
    TRAIN_DATA_PATH="$CMDC_ROOT/$FOLD_NAME/train"
    EVAL_DATA_PATH="$CMDC_ROOT/$FOLD_NAME/test"
    TRAIN_PROMPT_FILE="${FOLD_NAME}_multiprompt.jsonl"
    EVAL_PROMPT_FILE="${FOLD_NAME}_multiprompt.jsonl"
    TRAIN_SCP_FILE="${FOLD_NAME}.scp"
    EVAL_SCP_FILE="${FOLD_NAME}.scp"
    TRAIN_TASK_FILE="${FOLD_NAME}_multitask.jsonl"
    EVAL_TASK_FILE="${FOLD_NAME}_multitask.jsonl"
    FOLD_SAVE_PATH="$SAVE_PATH/$FOLD_NAME"
    FOLD_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${FOLD_NAME}"

    if [[ ! -d "$TRAIN_DATA_PATH" ]]; then
        echo "Missing train directory: $TRAIN_DATA_PATH"
        exit 1
    fi
    if [[ ! -d "$EVAL_DATA_PATH" ]]; then
        echo "Missing eval directory: $EVAL_DATA_PATH"
        exit 1
    fi

    mkdir -p "$FOLD_SAVE_PATH"

    WANDB_ARGS=(
        "++wandb.enabled=$WANDB_ENABLED"
        "++wandb.project=$WANDB_PROJECT"
        "++wandb.mode=$WANDB_MODE"
        "++wandb.log_step=$WANDB_LOG_STEP"
        "++wandb.group=$WANDB_GROUP"
        "++wandb.run_name=$FOLD_RUN_NAME"
    )

    if [[ -n $WANDB_ENTITY ]]; then
        WANDB_ARGS+=("++wandb.entity=$WANDB_ENTITY")
    fi

    echo
    echo "============================================"
    echo "  Running $FOLD_NAME"
    echo "============================================"
    echo "  TRAIN_DATA_PATH : $TRAIN_DATA_PATH"
    echo "  EVAL_DATA_PATH  : $EVAL_DATA_PATH"
    echo "  SAVE_PATH       : $FOLD_SAVE_PATH"
    echo "  WANDB_RUN_NAME  : $FOLD_RUN_NAME"
    echo "============================================"

    torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
        --standalone \
        main.py \
        ++train.train_strategy=$TRAIN_STRATEGY \
        ++env.device_type=$DEVICE_TYPE \
        ++env.model_path=$MODEL_PATH \
        ++env.save_path=$FOLD_SAVE_PATH \
        ++data.train_data_path=$TRAIN_DATA_PATH \
        ++data.eval_data_path=$EVAL_DATA_PATH \
        ++data.num_workers=$NUM_WORKERS \
        ++data.prefetch_factor=$PREFETCH_FACTOR \
        ++data.train_prompt_path=$TRAIN_DATA_PATH/$TRAIN_PROMPT_FILE \
        ++data.val_prompt_path=$EVAL_DATA_PATH/$EVAL_PROMPT_FILE \
        ++data.train_scp_filename=$TRAIN_SCP_FILE \
        ++data.eval_scp_filename=$EVAL_SCP_FILE \
        ++data.train_task_filename=$TRAIN_TASK_FILE \
        ++data.eval_task_filename=$EVAL_TASK_FILE \
        ++train.eval_step=$EVAL_STEP \
        ++train.lr=$LR \
        ++train.batch_size=$BATCH_SIZE \
        ++train.grad_accumulate_step=$GRAD_ACCUMULATE_STEP \
        ++train.train_epoch=$TRAIN_EPOCH \
        ++train.use_bfloat16=$USE_BFLOAT16 \
        "${WANDB_ARGS[@]}"

    STATUS=$?
    if [[ $STATUS -ne 0 ]]; then
        echo "$FOLD_NAME failed with exit code $STATUS"
        exit $STATUS
    fi

    python "$LOCAL_DIR/tools/summarize_cmdc_5fold.py" \
        --results-root "$SAVE_PATH" \
        --folds fold1 fold2 fold3 fold4 fold5 \
        --json-out "$SUMMARY_JSON" \
        --csv-out "$SUMMARY_CSV"
done

python "$LOCAL_DIR/tools/summarize_cmdc_5fold.py" \
    --results-root "$SAVE_PATH" \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --json-out "$SUMMARY_JSON" \
    --csv-out "$SUMMARY_CSV"

echo
echo "============================================"
echo "  Cross-validation summary saved"
echo "============================================"
echo "  JSON : $SUMMARY_JSON"
echo "  CSV  : $SUMMARY_CSV"
echo "============================================"
