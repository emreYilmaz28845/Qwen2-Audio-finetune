#!/bin/bash
# ==============================
# Text-only Qwen2-7B training launch script
# Replicates "Text (Qwen2-7B)" ablation experiment
# ==============================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

# ---- CHANGE THIS to your Qwen2-7B model path on MN5 ----
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$LOCAL_DIR/data/merged/train}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$LOCAL_DIR/data/merged/val}"

# Text-only prompt files (no audio markers, no emotion)
TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-merged_multiprompt_textonly.jsonl}"
EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-merged_multiprompt_textonly.jsonl}"

TRAIN_STRATEGY="textonly"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
LR="${LR:-1e-4}"
EVAL_STEP="${EVAL_STEP:-10}"
GRAD_ACCUMULATE_STEP="${GRAD_ACCUMULATE_STEP:-5}"
TRAIN_EPOCH="${TRAIN_EPOCH:-20}"
USE_BFLOAT16="${USE_BFLOAT16:-True}"
SAVE_PATH="${SAVE_PATH:-$LOCAL_DIR/output_model}"

WANDB_ENABLED="${WANDB_ENABLED:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen2-textonly-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_STEP="${WANDB_LOG_STEP:-10}"

NUM_WORKERS="${NUM_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

WANDB_ARGS=(
    "++wandb.enabled=$WANDB_ENABLED"
    "++wandb.project=$WANDB_PROJECT"
    "++wandb.mode=$WANDB_MODE"
    "++wandb.log_step=$WANDB_LOG_STEP"
)

if [[ -n $WANDB_ENTITY ]]; then
    WANDB_ARGS+=("++wandb.entity=$WANDB_ENTITY")
fi
if [[ -n $WANDB_RUN_NAME ]]; then
    WANDB_ARGS+=("++wandb.run_name=$WANDB_RUN_NAME")
fi
if [[ -n $WANDB_GROUP ]]; then
    WANDB_ARGS+=("++wandb.group=$WANDB_GROUP")
fi

echo "============================================"
echo "  Text-Only Training (Qwen2-7B)"
echo "============================================"
echo "  MODEL_PATH : $MODEL_PATH"
echo "  PROMPTS    : $TRAIN_PROMPT_FILE"
echo "  LR         : $LR"
echo "  STRATEGY   : $TRAIN_STRATEGY"
echo "============================================"

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --standalone \
    main.py \
    ++train.train_strategy=$TRAIN_STRATEGY \
    ++env.device_type=$DEVICE_TYPE \
    ++env.model_path=$MODEL_PATH \
    ++data.train_data_path=$TRAIN_DATA_PATH \
    ++data.eval_data_path=$EVAL_DATA_PATH \
    ++data.num_workers=$NUM_WORKERS \
    ++data.prefetch_factor=$PREFETCH_FACTOR \
    ++data.train_prompt_path=$TRAIN_DATA_PATH/$TRAIN_PROMPT_FILE \
    ++data.val_prompt_path=$EVAL_DATA_PATH/$EVAL_PROMPT_FILE \
    ++data.train_scp_filename=merged.scp \
    ++data.eval_scp_filename=merged.scp \
    ++data.train_task_filename=merged_multitask.jsonl \
    ++data.eval_task_filename=merged_multitask.jsonl \
    ++train.eval_step=$EVAL_STEP \
    ++train.lr=$LR \
    ++train.grad_accumulate_step=$GRAD_ACCUMULATE_STEP \
    ++train.train_epoch=$TRAIN_EPOCH \
    ++train.use_bfloat16=$USE_BFLOAT16 \
    ++env.save_path=$SAVE_PATH \
    "${WANDB_ARGS[@]}"
