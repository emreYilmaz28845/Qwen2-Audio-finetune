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

export DATASET_NAME="${DATASET_NAME:-merged}" # merged, daic_woz, eatd
export MODEL_FAMILY="${MODEL_FAMILY:-audio}" # audio or text
export PROMPT_MODE="${PROMPT_MODE:-audiotext}" # full, audiotext, or textonly
export TASK_VARIANT="${TASK_VARIANT:-default}" # default or filtered

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
        EVAL_SPLIT="test"
        LOG_DIR_NAME="optuna_eatd"
        STORAGE_DIR_NAME="optuna_eatd"
        OUTPUT_DIR_NAME="optuna_eatd_hpo"
        LOG_FILE_PREFIX="optuna_eatd"
        JOB_FILE_PREFIX="optuna-qwen-optuna-eatd"
        ;;
    *)
        echo "Unsupported DATASET_NAME: $DATASET_NAME"
        echo "Use DATASET_NAME=merged, DATASET_NAME=daic_woz, or DATASET_NAME=eatd"
        exit 1
        ;;
esac

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
export EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATASET_ROOT/$EVAL_SPLIT}"
export TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-$PROMPT_FILE}"
export EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-$PROMPT_FILE}"
export TRAIN_TASK_FILE="${TRAIN_TASK_FILE:-$TASK_FILE}"
export EVAL_TASK_FILE="${EVAL_TASK_FILE:-$TASK_FILE}"
export TRAIN_SCP_FILE="${TRAIN_SCP_FILE:-${DATASET_NAME}.scp}"
export EVAL_SCP_FILE="${EVAL_SCP_FILE:-${DATASET_NAME}.scp}"
export WAV_TYPE="${WAV_TYPE:-wav}"
export NUM_GPUS="${NUM_GPUS:-4}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

export LOG_DIR="${LOG_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/logs/${LOG_DIR_NAME}}"
export STORAGE_PATH="${STORAGE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/optuna_studies/${STORAGE_DIR_NAME}}"
export SAVE_PATH="${SAVE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/${OUTPUT_DIR_NAME}/${PROMPT_MODE}}"

mkdir -p "$LOG_DIR" "$STORAGE_PATH" "$SAVE_PATH"

SLURM_STDOUT_FILE="$LOG_DIR/${JOB_FILE_PREFIX}-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_DIR/${JOB_FILE_PREFIX}-${SLURM_JOB_ID}.err"
exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-${DATASET_NAME}_${MODEL_FAMILY}_${PROMPT_MODE}_hpo_$(date +%Y%m%d_%H%M%S)}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="$LOG_DIR/${LOG_FILE_PREFIX}_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Single-Dataset Optuna Hyperparameter Tuning" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Dataset Name: $DATASET_NAME" | tee -a "$LOG_FILE"
echo "Model Family: $MODEL_FAMILY" | tee -a "$LOG_FILE"
echo "Prompt Mode: $PROMPT_MODE" | tee -a "$LOG_FILE"
echo "Task Variant: $TASK_VARIANT" | tee -a "$LOG_FILE"
echo "Launch Input Mode (derived): $LAUNCH_INPUT_MODE" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Train Data Path: $TRAIN_DATA_PATH" | tee -a "$LOG_FILE"
echo "Eval Data Path: $EVAL_DATA_PATH" | tee -a "$LOG_FILE"
echo "Train Prompt File: $TRAIN_PROMPT_FILE" | tee -a "$LOG_FILE"
echo "Eval Prompt File: $EVAL_PROMPT_FILE" | tee -a "$LOG_FILE"
echo "Log Dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Storage Path: $STORAGE_PATH" | tee -a "$LOG_FILE"
echo "Save Path: $SAVE_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python optuna_hpo/hpo.py \
    --n-trials "$N_TRIALS" \
    --study-name "$STUDY_NAME" \
    --storage-path "$STORAGE_PATH" \
    --save-root "$SAVE_PATH" \
    --dataset-name "$DATASET_NAME" \
    --model-family "$MODEL_FAMILY" \
    --prompt-mode "$PROMPT_MODE" \
    --task-variant "$TASK_VARIANT" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
