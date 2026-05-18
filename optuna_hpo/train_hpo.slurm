#!/bin/bash
#SBATCH -J qwen-optuna-hpo
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
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
mkdir -p logs output_model optuna_studies

export DATASET_NAME="${DATASET_NAME:-merged}" # merged, daic_woz, eatd, cmdc
export MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
export PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
export TASK_VARIANT="${TASK_VARIANT:-default}" # default or filtered
export DAIC_WOZ_EVAL_LEVEL="${DAIC_WOZ_EVAL_LEVEL:-${DAIC_EVAL_LEVEL:-person}}"
export DAIC_WOZ_EVAL_MODE="${DAIC_WOZ_EVAL_MODE:-${DAIC_EVAL_MODE:-majority_vote}}"
export DAIC_WOZ_PERSON_THRESHOLD="${DAIC_WOZ_PERSON_THRESHOLD:-${DAIC_PERSON_THRESHOLD:-0.5}}"
export DAIC_EVAL_LEVEL="$DAIC_WOZ_EVAL_LEVEL"
export DAIC_EVAL_MODE="$DAIC_WOZ_EVAL_MODE"
export DAIC_PERSON_THRESHOLD="$DAIC_WOZ_PERSON_THRESHOLD"
export EATD_EVAL_LEVEL="${EATD_EVAL_LEVEL:-person}"
export EATD_EVAL_MODE="${EATD_EVAL_MODE:-majority_vote}"
export EATD_PERSON_THRESHOLD="${EATD_PERSON_THRESHOLD:-0.5}"
export CMDC_EVAL_LEVEL="${CMDC_EVAL_LEVEL:-person}"
export CMDC_EVAL_MODE="${CMDC_EVAL_MODE:-majority_vote}"
export CMDC_PERSON_THRESHOLD="${CMDC_PERSON_THRESHOLD:-0.5}"
export ENABLE_PRUNING="${ENABLE_PRUNING:-1}"
export PRUNER_STARTUP_TRIALS="${PRUNER_STARTUP_TRIALS:-5}"
export PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-2}"
export PRUNER_INTERVAL_STEPS="${PRUNER_INTERVAL_STEPS:-1}"

case "$MODEL_FAMILY" in
    audio)
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
        LAUNCH_INPUT_MODE="audiotext"
        ;;
    text|textonly)
        export MODEL_FAMILY="text"
        export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
        LAUNCH_INPUT_MODE="textonly"
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
        DATASET_ROOT_DEFAULT="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/merged"
        EVAL_SPLIT="val"
        LOG_DIR_NAME="optuna_merged"
        STORAGE_DIR_NAME="optuna_merged"
        OUTPUT_DIR_NAME="optuna_merged_hpo"
        LOG_FILE_PREFIX="optuna_merged"
        JOB_FILE_PREFIX="optuna-qwen-optuna-merged"
        ;;
    daic_woz)
        DATASET_ROOT_DEFAULT="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/daic_woz"
        EVAL_SPLIT="val"
        LOG_DIR_NAME="optuna_daic_woz"
        STORAGE_DIR_NAME="optuna_daic_woz"
        OUTPUT_DIR_NAME="optuna_daic_woz_hpo"
        LOG_FILE_PREFIX="optuna_daic_woz"
        JOB_FILE_PREFIX="optuna-qwen-optuna-daic-woz"
        ;;
    eatd)
        DATASET_ROOT_DEFAULT="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/eatd"
        EVAL_SPLIT="val"
        LOG_DIR_NAME="optuna_eatd"
        STORAGE_DIR_NAME="optuna_eatd"
        OUTPUT_DIR_NAME="optuna_eatd_hpo"
        LOG_FILE_PREFIX="optuna_eatd"
        JOB_FILE_PREFIX="optuna-qwen-optuna-eatd"
        ;;
    cmdc)
        DATASET_ROOT_DEFAULT="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/cmdc"
        EVAL_SPLIT="val"
        LOG_DIR_NAME="optuna_cmdc"
        STORAGE_DIR_NAME="optuna_cmdc"
        OUTPUT_DIR_NAME="optuna_cmdc_hpo"
        LOG_FILE_PREFIX="optuna_cmdc"
        JOB_FILE_PREFIX="optuna-qwen-optuna-cmdc"
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME"
        echo "Use DATASET_NAME=merged, DATASET_NAME=daic_woz, DATASET_NAME=eatd, or DATASET_NAME=cmdc"
        exit 1
        ;;
esac

DATASET_LEVEL_SUFFIX=""
if [ "$DATASET_NAME" = "daic_woz" ]; then
    DATASET_LEVEL_SUFFIX="_${DAIC_WOZ_EVAL_LEVEL}"
elif [ "$DATASET_NAME" = "eatd" ]; then
    DATASET_LEVEL_SUFFIX="_${EATD_EVAL_LEVEL}"
elif [ "$DATASET_NAME" = "cmdc" ]; then
    DATASET_LEVEL_SUFFIX="_${CMDC_EVAL_LEVEL}"
fi

case "$PROMPT_MODE" in
    full)
        PROMPT_FILE="${DATASET_NAME}_multiprompt.jsonl"
        ;;
    audiotext)
        PROMPT_FILE="${DATASET_NAME}_multiprompt_audiotext.jsonl"
        ;;
    textonly)
        PROMPT_FILE="${DATASET_NAME}_multiprompt_textonly.jsonl"
        ;;
    *)
        echo "Unsupported PROMPT_MODE: $PROMPT_MODE"
        exit 1
        ;;
esac

case "$TASK_VARIANT" in
    default)
        TASK_FILE="${DATASET_NAME}_multitask.jsonl"
        ;;
    filtered)
        TASK_FILE="${DATASET_NAME}_multitask_filtered.jsonl"
        ;;
    *)
        echo "Unsupported TASK_VARIANT: $TASK_VARIANT"
        echo "Use TASK_VARIANT=default or TASK_VARIANT=filtered"
        exit 1
        ;;
esac

export DATASET_ROOT="${DATASET_ROOT:-$DATASET_ROOT_DEFAULT}"
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATASET_ROOT/train}"
export VAL_DATA_PATH="${VAL_DATA_PATH:-$DATASET_ROOT/$EVAL_SPLIT}"
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-$VAL_DATA_PATH}"
export TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-$PROMPT_FILE}"
export VAL_PROMPT_FILE="${VAL_PROMPT_FILE:-$PROMPT_FILE}"
export EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-$VAL_PROMPT_FILE}"
export TRAIN_TASK_FILE="${TRAIN_TASK_FILE:-$TASK_FILE}"
export VAL_TASK_FILE="${VAL_TASK_FILE:-$TASK_FILE}"
export EVAL_TASK_FILE="${EVAL_TASK_FILE:-$VAL_TASK_FILE}"
export TRAIN_SCP_FILE="${TRAIN_SCP_FILE:-${DATASET_NAME}.scp}"
export VAL_SCP_FILE="${VAL_SCP_FILE:-${DATASET_NAME}.scp}"
export EVAL_SCP_FILE="${EVAL_SCP_FILE:-$VAL_SCP_FILE}"
export WAV_TYPE="${WAV_TYPE:-wav}"
export NUM_GPUS="${NUM_GPUS:-4}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AUDIOLLM_ENABLE_GPU_LOG="${AUDIOLLM_ENABLE_GPU_LOG:-1}"
export AUDIOLLM_GPU_LOG_INTERVAL_SEC="${AUDIOLLM_GPU_LOG_INTERVAL_SEC:-60}"

export LOG_DIR="${LOG_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/logs/${LOG_DIR_NAME}${DATASET_LEVEL_SUFFIX}}"
export STORAGE_PATH="${STORAGE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/optuna_studies/${STORAGE_DIR_NAME}${DATASET_LEVEL_SUFFIX}}"
export SAVE_PATH="${SAVE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/${OUTPUT_DIR_NAME}${DATASET_LEVEL_SUFFIX}/${PROMPT_MODE}}"

mkdir -p "$LOG_DIR" "$STORAGE_PATH" "$SAVE_PATH"

SLURM_STDOUT_FILE="$LOG_DIR/${JOB_FILE_PREFIX}-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_DIR/${JOB_FILE_PREFIX}-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-${DATASET_NAME}${DATASET_LEVEL_SUFFIX}_${MODEL_FAMILY}_${PROMPT_MODE}_hpo_$(date +%Y%m%d_%H%M%S)}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="$LOG_DIR/${LOG_FILE_PREFIX}_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Single-Dataset Optuna Hyperparameter Tuning" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Enable Pruning: $ENABLE_PRUNING" | tee -a "$LOG_FILE"
echo "Pruner Startup Trials: $PRUNER_STARTUP_TRIALS" | tee -a "$LOG_FILE"
echo "Pruner Warmup Steps: $PRUNER_WARMUP_STEPS" | tee -a "$LOG_FILE"
echo "Pruner Interval Steps: $PRUNER_INTERVAL_STEPS" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Dataset Name: $DATASET_NAME" | tee -a "$LOG_FILE"
echo "Model Family: $MODEL_FAMILY" | tee -a "$LOG_FILE"
echo "Prompt Mode: $PROMPT_MODE" | tee -a "$LOG_FILE"
echo "Task Variant: $TASK_VARIANT" | tee -a "$LOG_FILE"
echo "DAIC Eval Level: $DAIC_WOZ_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "DAIC Eval Mode: $DAIC_WOZ_EVAL_MODE" | tee -a "$LOG_FILE"
echo "DAIC Person Threshold: $DAIC_WOZ_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "EATD Eval Level: $EATD_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "EATD Eval Mode: $EATD_EVAL_MODE" | tee -a "$LOG_FILE"
echo "EATD Person Threshold: $EATD_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "CMDC Eval Level: $CMDC_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "CMDC Eval Mode: $CMDC_EVAL_MODE" | tee -a "$LOG_FILE"
echo "CMDC Person Threshold: $CMDC_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "Launch Input Mode (derived): $LAUNCH_INPUT_MODE" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Train Data Path: $TRAIN_DATA_PATH" | tee -a "$LOG_FILE"
echo "Validation Data Path: $VAL_DATA_PATH" | tee -a "$LOG_FILE"
echo "Train Prompt File: $TRAIN_PROMPT_FILE" | tee -a "$LOG_FILE"
echo "Validation Prompt File: $VAL_PROMPT_FILE" | tee -a "$LOG_FILE"
echo "Log Dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Storage Path: $STORAGE_PATH" | tee -a "$LOG_FILE"
echo "Save Path: $SAVE_PATH" | tee -a "$LOG_FILE"
echo "GPU Log: $AUDIOLLM_ENABLE_GPU_LOG (interval=${AUDIOLLM_GPU_LOG_INTERVAL_SEC}s)" | tee -a "$LOG_FILE"
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

if [[ "${SKIP_SPLIT_VALIDATION:-0}" != "1" ]]; then
    python tools/validate_splits.py --dataset "$DATASET_NAME" --strict 2>&1 | tee -a "$LOG_FILE"
fi

PRUNING_FLAG="--disable-pruning"
case "${ENABLE_PRUNING,,}" in
    1|true|yes|on)
        PRUNING_FLAG="--enable-pruning"
        ;;
esac

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
    --cmdc-eval-level "$CMDC_EVAL_LEVEL" \
    --cmdc-eval-mode "$CMDC_EVAL_MODE" \
    --cmdc-person-threshold "$CMDC_PERSON_THRESHOLD" \
    --pruner-startup-trials "$PRUNER_STARTUP_TRIALS" \
    --pruner-warmup-steps "$PRUNER_WARMUP_STEPS" \
    --pruner-interval-steps "$PRUNER_INTERVAL_STEPS" \
    "$PRUNING_FLAG" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
