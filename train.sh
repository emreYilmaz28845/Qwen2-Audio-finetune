#!/bin/bash
# ==============================
# 2-GPU 分布式训练启动脚本
# ==============================

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LOCAL_DIR" || exit 1

MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$LOCAL_DIR/data/merged/train}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$LOCAL_DIR/data/merged/val}"

TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-merged_multiprompt.jsonl}"
EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-merged_multiprompt.jsonl}"


TRAIN_STRATEGY="${TRAIN_STRATEGY:-ddp}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_STEP="${EVAL_STEP:-200}"
GRAD_ACCUMULATE_STEP="${GRAD_ACCUMULATE_STEP:-5}"
TRAIN_EPOCH="${TRAIN_EPOCH:-20}"
USE_BFLOAT16="${USE_BFLOAT16:-True}"
NUM_GPUS="${NUM_GPUS:-1}"

WANDB_ENABLED="${WANDB_ENABLED:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen2-audio-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_LOG_STEP="${WANDB_LOG_STEP:-10}"


# Data loader knobs
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


if [[ $TRAIN_STRATEGY == "ddp" ]]; then
    # export CUDA_VISIBLE_DEVICES=0,1

    torchrun \
        --nnodes=1 \
        --nproc_per_node="$NUM_GPUS" \
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
        ++train.batch_size=$BATCH_SIZE \
        ++train.grad_accumulate_step=$GRAD_ACCUMULATE_STEP \
        ++train.train_epoch=$TRAIN_EPOCH \
        ++train.use_bfloat16=$USE_BFLOAT16 \
        "${WANDB_ARGS[@]}"

else
    export DEEPSPEED_CONFIG=./config/deepspeed.json
    deepspeed \
        --num_nodes=1 \
        --num_gpus=$NUM_GPUS \
        main.py \
        ++train.train_strategy=$TRAIN_STRATEGY \
        ++train.deepspeed_config=$DEEPSPEED_CONFIG \
        ++env.device_type=$DEVICE_TYPE \
        ++env.model_path=$MODEL_PATH \
        ++data.train_data_path=$TRAIN_DATA_PATH \
        ++data.eval_data_path=$EVAL_DATA_PATH \
        ++data.train_prompt_path=$TRAIN_DATA_PATH/$TRAIN_PROMPT_FILE \
        ++data.val_prompt_path=$EVAL_DATA_PATH/$EVAL_PROMPT_FILE \
        ++data.train_scp_filename=merged.scp \
        ++data.eval_scp_filename=merged.scp \
        ++data.train_task_filename=merged_multitask.jsonl \
        ++data.eval_task_filename=merged_multitask.jsonl \
        ++train.eval_step=$EVAL_STEP \
        ++train.lr=$LR \
        ++train.batch_size=$BATCH_SIZE \
        ++train.grad_accumulate_step=$GRAD_ACCUMULATE_STEP \
        ++train.train_epoch=$TRAIN_EPOCH \
        ++train.use_bfloat16=$USE_BFLOAT16 \
        "${WANDB_ARGS[@]}"
fi
