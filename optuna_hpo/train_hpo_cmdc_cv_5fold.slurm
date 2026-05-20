#!/bin/bash
#SBATCH -J qwen-optuna-cmdc-cv
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune

# =================================
# CMDC 5-fold cross-validated Optuna Hyperparameter Tuning on MN5
# =================================

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model optuna_studies

export MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
export PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly

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
export CMDC_ROOT="${CMDC_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/cmdc}"
export NUM_GPUS="${NUM_GPUS:-4}"
export FOLDS="${FOLDS:-fold1 fold2 fold3 fold4 fold5}"
export STUDY_MODE="${STUDY_MODE:-cv_mean}" # per_fold or cv_mean
export CMDC_EVAL_LEVEL="${CMDC_EVAL_LEVEL:-person}"
export CMDC_EVAL_MODE="${CMDC_EVAL_MODE:-majority_vote}"
export CMDC_PERSON_THRESHOLD="${CMDC_PERSON_THRESHOLD:-0.5}"
export ENABLE_PRUNING="${ENABLE_PRUNING:-1}"
export PRUNER_STARTUP_TRIALS="${PRUNER_STARTUP_TRIALS:-5}"
export PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-2}"
export PRUNER_INTERVAL_STEPS="${PRUNER_INTERVAL_STEPS:-1}"
export LOG_DIR="${LOG_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/logs/optuna_cmdc_cv_5fold_${STUDY_MODE}_${CMDC_EVAL_LEVEL}}"
export STORAGE_PATH="${STORAGE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/optuna_studies/optuna_cmdc_cv_5fold_${STUDY_MODE}_${CMDC_EVAL_LEVEL}}"
export SAVE_ROOT="${SAVE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_cmdc_cv_5fold_${STUDY_MODE}_${CMDC_EVAL_LEVEL}}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AUDIOLLM_ENABLE_COLLATE_DEBUG="${AUDIOLLM_ENABLE_COLLATE_DEBUG:-0}"
export AUDIOLLM_COLLATE_DEBUG_LIMIT="${AUDIOLLM_COLLATE_DEBUG_LIMIT:-6}"
export AUDIOLLM_ENABLE_METRIC_DEBUG="${AUDIOLLM_ENABLE_METRIC_DEBUG:-0}"
export AUDIOLLM_METRIC_DEBUG_LIMIT="${AUDIOLLM_METRIC_DEBUG_LIMIT:-12}"
export AUDIOLLM_ENABLE_GPU_LOG="${AUDIOLLM_ENABLE_GPU_LOG:-1}"
export AUDIOLLM_GPU_LOG_INTERVAL_SEC="${AUDIOLLM_GPU_LOG_INTERVAL_SEC:-60}"
export AUDIOLLM_ENABLE_MODEL_IO_DEBUG="${AUDIOLLM_ENABLE_MODEL_IO_DEBUG:-0}"
export AUDIOLLM_MODEL_IO_DEBUG_LIMIT="${AUDIOLLM_MODEL_IO_DEBUG_LIMIT:-2}"
if [ "$MODEL_FAMILY" = "audio" ]; then
    LAUNCH_INPUT_MODE="audiotext"
else
    LAUNCH_INPUT_MODE="textonly"
fi

mkdir -p "$LOG_DIR" "$STORAGE_PATH" "$SAVE_ROOT"

SLURM_STDOUT_FILE="$LOG_DIR/optuna-qwen-optuna-cmdc-cv-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_DIR/optuna-qwen-optuna-cmdc-cv-${SLURM_JOB_ID}.err"

exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

N_TRIALS="${N_TRIALS:-20}"
TARGET_TOTAL_TRIALS="${TARGET_TOTAL_TRIALS:-}"
STUDY_NAME="${STUDY_NAME:-cmdc_${CMDC_EVAL_LEVEL}_${MODEL_FAMILY}_${PROMPT_MODE}_${STUDY_MODE}_cv_hpo_$(date +%Y%m%d_%H%M%S)}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="$LOG_DIR/optuna_cmdc_cv_5fold_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC 5-Fold Cross-Validated Hyperparameter Tuning" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of CV Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Enable Pruning: $ENABLE_PRUNING" | tee -a "$LOG_FILE"
echo "CMDC Eval Level: $CMDC_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "CMDC Eval Mode: $CMDC_EVAL_MODE" | tee -a "$LOG_FILE"
echo "CMDC Person Threshold: $CMDC_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "Pruner Startup Trials: $PRUNER_STARTUP_TRIALS" | tee -a "$LOG_FILE"
echo "Pruner Warmup Steps: $PRUNER_WARMUP_STEPS" | tee -a "$LOG_FILE"
echo "Pruner Interval Steps: $PRUNER_INTERVAL_STEPS" | tee -a "$LOG_FILE"
echo "Target Total Completed Trials: ${TARGET_TOTAL_TRIALS:-disabled}" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Study Mode: $STUDY_MODE" | tee -a "$LOG_FILE"
echo "Model Family: $MODEL_FAMILY" | tee -a "$LOG_FILE"
echo "Prompt Mode: $PROMPT_MODE" | tee -a "$LOG_FILE"
echo "Launch Input Mode (derived): $LAUNCH_INPUT_MODE" | tee -a "$LOG_FILE"
echo "Folds: $FOLDS" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Log Dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Storage Path: $STORAGE_PATH" | tee -a "$LOG_FILE"
echo "Save Root: $SAVE_ROOT" | tee -a "$LOG_FILE"
echo "Collate Debug: $AUDIOLLM_ENABLE_COLLATE_DEBUG (limit=$AUDIOLLM_COLLATE_DEBUG_LIMIT)" | tee -a "$LOG_FILE"
echo "Metric Debug: $AUDIOLLM_ENABLE_METRIC_DEBUG (limit=$AUDIOLLM_METRIC_DEBUG_LIMIT)" | tee -a "$LOG_FILE"
echo "GPU Log: $AUDIOLLM_ENABLE_GPU_LOG (interval=${AUDIOLLM_GPU_LOG_INTERVAL_SEC}s)" | tee -a "$LOG_FILE"
echo "Model IO Debug: $AUDIOLLM_ENABLE_MODEL_IO_DEBUG (limit per phase=${AUDIOLLM_MODEL_IO_DEBUG_LIMIT})" | tee -a "$LOG_FILE"
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

TARGET_TOTAL_ARGS=()
if [ -n "$TARGET_TOTAL_TRIALS" ]; then
    TARGET_TOTAL_ARGS=(--target-total-trials "$TARGET_TOTAL_TRIALS")
fi

PRUNING_FLAG="--disable-pruning"
case "${ENABLE_PRUNING,,}" in
    1|true|yes|on)
        PRUNING_FLAG="--enable-pruning"
        ;;
esac

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
    --cmdc-eval-level "$CMDC_EVAL_LEVEL" \
    --cmdc-eval-mode "$CMDC_EVAL_MODE" \
    --cmdc-person-threshold "$CMDC_PERSON_THRESHOLD" \
    --pruner-startup-trials "$PRUNER_STARTUP_TRIALS" \
    --pruner-warmup-steps "$PRUNER_WARMUP_STEPS" \
    --pruner-interval-steps "$PRUNER_INTERVAL_STEPS" \
    "$PRUNING_FLAG" \
    "${TARGET_TOTAL_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC CV-HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
