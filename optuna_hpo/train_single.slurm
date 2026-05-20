#!/bin/bash
#SBATCH -J qwen-single-trial
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model

export DATASET_NAME="${DATASET_NAME:-merged}" # merged, daic_woz, eatd
export MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
export PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
export TASK_VARIANT="${TASK_VARIANT:-default}" # default or filtered
export NUM_GPUS="${NUM_GPUS:-4}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export AUDIOLLM_ENABLE_GPU_LOG="${AUDIOLLM_ENABLE_GPU_LOG:-1}"
export AUDIOLLM_GPU_LOG_INTERVAL_SEC="${AUDIOLLM_GPU_LOG_INTERVAL_SEC:-60}"
export AUDIOLLM_ENABLE_MODEL_IO_DEBUG="${AUDIOLLM_ENABLE_MODEL_IO_DEBUG:-0}"
export AUDIOLLM_MODEL_IO_DEBUG_LIMIT="${AUDIOLLM_MODEL_IO_DEBUG_LIMIT:-2}"
export AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS="${AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS:-1}"
export AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS="${AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS:-1}"
export AUDIOLLM_ENABLE_TEACHER_FORCED_EVAL_DIAGNOSTIC="${AUDIOLLM_ENABLE_TEACHER_FORCED_EVAL_DIAGNOSTIC:-0}"
export AUDIOLLM_EVAL_MAX_NEW_TOKENS="${AUDIOLLM_EVAL_MAX_NEW_TOKENS:-16}"
export AUDIOLLM_EVAL_PRINT_PREDICTIONS_LIMIT="${AUDIOLLM_EVAL_PRINT_PREDICTIONS_LIMIT:-5}"

case "$MODEL_FAMILY" in
    audio)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
        ;;
    text|textonly)
        export MODEL_FAMILY="text"
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
        ;;
    *)
        echo "Unsupported MODEL_FAMILY: $MODEL_FAMILY"
        echo "Use MODEL_FAMILY=audio or MODEL_FAMILY=text"
        exit 1
        ;;
esac

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
    merged)
        LOG_DIR_NAME="single_trial_merged"
        OUTPUT_DIR_NAME="single_trial/merged"
        ;;
    daic_woz)
        LOG_DIR_NAME="single_trial_daic_woz"
        OUTPUT_DIR_NAME="single_trial/daic_woz"
        ;;
    eatd)
        LOG_DIR_NAME="single_trial_eatd"
        OUTPUT_DIR_NAME="single_trial/eatd"
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME"
        echo "Use DATASET_NAME=merged, DATASET_NAME=daic_woz, or DATASET_NAME=eatd"
        exit 1
        ;;
esac

export LOG_DIR="${LOG_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/logs/${LOG_DIR_NAME}}"
export SAVE_PATH="${SAVE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/${OUTPUT_DIR_NAME}/${PROMPT_MODE}}"
mkdir -p "$LOG_DIR" "$SAVE_PATH"

SLURM_STDOUT_FILE="$LOG_DIR/single-trial-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_DIR/single-trial-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

TRIAL_NUMBER="${TRIAL_NUMBER:-999}"
LR="${LR:-0.0009308255549801389}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="$LOG_DIR/single_trial_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Single Training Run" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Dataset Name: $DATASET_NAME" | tee -a "$LOG_FILE"
echo "Model Family: $MODEL_FAMILY" | tee -a "$LOG_FILE"
echo "Prompt Mode: $PROMPT_MODE" | tee -a "$LOG_FILE"
echo "Task Variant: $TASK_VARIANT" | tee -a "$LOG_FILE"
echo "Trial Number: $TRIAL_NUMBER" | tee -a "$LOG_FILE"
echo "LR: $LR" | tee -a "$LOG_FILE"
echo "Batch Size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "LoRA R: $LORA_R" | tee -a "$LOG_FILE"
echo "LoRA Alpha: $LORA_ALPHA" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Save Path: $SAVE_PATH" | tee -a "$LOG_FILE"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM" | tee -a "$LOG_FILE"
echo "GPU Log: $AUDIOLLM_ENABLE_GPU_LOG (interval=${AUDIOLLM_GPU_LOG_INTERVAL_SEC}s)" | tee -a "$LOG_FILE"
echo "Model IO Debug: $AUDIOLLM_ENABLE_MODEL_IO_DEBUG (limit per phase=${AUDIOLLM_MODEL_IO_DEBUG_LIMIT}, train_every_steps=${AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS}, eval_every_calls=${AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS})" | tee -a "$LOG_FILE"
echo "Teacher-Forced Eval Diagnostic: $AUDIOLLM_ENABLE_TEACHER_FORCED_EVAL_DIAGNOSTIC" | tee -a "$LOG_FILE"
echo "Eval Generation: max_new_tokens=${AUDIOLLM_EVAL_MAX_NEW_TOKENS}, print_predictions_limit=${AUDIOLLM_EVAL_PRINT_PREDICTIONS_LIMIT}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

GPU_LOG_PID=""

start_gpu_logger() {
    if [ "$AUDIOLLM_ENABLE_GPU_LOG" != "1" ]; then
        return
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[GPU_LOG] nvidia-smi not found, skipping GPU memory logging" | tee -a "$LOG_FILE"
        return
    fi

    (
        while true; do
            while IFS=, read -r gpu_index mem_used mem_total util_gpu; do
                gpu_index=$(echo "$gpu_index" | xargs)
                mem_used=$(echo "$mem_used" | xargs)
                mem_total=$(echo "$mem_total" | xargs)
                util_gpu=$(echo "$util_gpu" | xargs)

                if [ -n "$gpu_index" ] && [ -n "$mem_used" ] && [ -n "$mem_total" ]; then
                    mem_pct=$(awk -v used="$mem_used" -v total="$mem_total" 'BEGIN { if (total > 0) printf "%.1f", (used / total) * 100; else printf "0.0" }')
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [GPU_LOG] gpu=${gpu_index} vram_used_mb=${mem_used} vram_total_mb=${mem_total} vram_used_pct=${mem_pct}% gpu_util_pct=${util_gpu}%" | tee -a "$LOG_FILE"
                fi
            done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

            sleep "$AUDIOLLM_GPU_LOG_INTERVAL_SEC"
        done
    ) &

    GPU_LOG_PID=$!
}

stop_gpu_logger() {
    if [ -n "$GPU_LOG_PID" ] && kill -0 "$GPU_LOG_PID" >/dev/null 2>&1; then
        kill "$GPU_LOG_PID" >/dev/null 2>&1 || true
        wait "$GPU_LOG_PID" 2>/dev/null || true
    fi
}

trap stop_gpu_logger EXIT
start_gpu_logger

python optuna_hpo/run_one_trial.py \
    --trial-number "$TRIAL_NUMBER" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --num-gpus "$NUM_GPUS" \
    --dataset-name "$DATASET_NAME" \
    --model-family "$MODEL_FAMILY" \
    --prompt-mode "$PROMPT_MODE" \
    --task-variant "$TASK_VARIANT" \
    --save-path "$SAVE_PATH" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Single Trial Complete!" | tee -a "$LOG_FILE"
echo "Model saved to: $SAVE_PATH" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
