#!/bin/bash
#SBATCH -J qwen-eval
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
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
mkdir -p logs eval_results

export AUDIOLLM_DATASETS_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Override via:
# sbatch --export=PEFT_PATH=/path/to/checkpoint,DATASET_NAME=daic_woz,MODEL_FAMILY=audio,PROMPT_MODE=audiotext eval.slurm
export PEFT_PATH="${PEFT_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_daic_woz_hpo_person/audiotext_trial_trial_001_lr4e-05_bs1_r8_a16/best_model}"
export MODEL_FAMILY="${MODEL_FAMILY:-audio}"
export PROMPT_MODE="${PROMPT_MODE:-audiotext}"
export DATASET_NAME="${DATASET_NAME:-merged}"
export CMDC_FOLD="${CMDC_FOLD:-fold1}"
export DAIC_WOZ_EVAL_LEVEL="${DAIC_WOZ_EVAL_LEVEL:-${DAIC_EVAL_LEVEL:-person}}"
export DAIC_WOZ_EVAL_MODE="${DAIC_WOZ_EVAL_MODE:-${DAIC_EVAL_MODE:-majority_vote}}"
export DAIC_WOZ_PERSON_THRESHOLD="${DAIC_WOZ_PERSON_THRESHOLD:-${DAIC_PERSON_THRESHOLD:-0.5}}"
export EATD_EVAL_LEVEL="${EATD_EVAL_LEVEL:-person}"
export EATD_EVAL_MODE="${EATD_EVAL_MODE:-majority_vote}"
export EATD_PERSON_THRESHOLD="${EATD_PERSON_THRESHOLD:-0.5}"
export CMDC_EVAL_LEVEL="${CMDC_EVAL_LEVEL:-person}"
export CMDC_EVAL_MODE="${CMDC_EVAL_MODE:-majority_vote}"
export CMDC_PERSON_THRESHOLD="${CMDC_PERSON_THRESHOLD:-0.5}"
export CHECKPOINT_MODE="${CHECKPOINT_MODE:-auto}"
export ADAPTER_PATH="${ADAPTER_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_daic_woz_hpo_person/audiotext_trial_trial_001_lr4e-05_bs1_r8_a16/best_model/audio_adapter_state.pt}"

DAIC_LEVEL_SUFFIX=""
if [ "$DATASET_NAME" = "daic_woz" ]; then
    DAIC_LEVEL_SUFFIX="_${DAIC_WOZ_EVAL_LEVEL}"
fi

export LOG_DIR="${LOG_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/logs/eval_${DATASET_NAME}${DAIC_LEVEL_SUFFIX}}"
export RESULTS_DIR="${RESULTS_DIR:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/eval_results/eval_${DATASET_NAME}${DAIC_LEVEL_SUFFIX}}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

SLURM_STDOUT_FILE="$LOG_DIR/qwen-eval-${DATASET_NAME}-${SLURM_JOB_ID}.out"
SLURM_STDERR_FILE="$LOG_DIR/qwen-eval-${DATASET_NAME}-${SLURM_JOB_ID}.err"

exec > >(tee -a "$SLURM_STDOUT_FILE")
exec 2> >(tee -a "$SLURM_STDERR_FILE" >&2)

TIMESTAMP="$(date +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="$LOG_DIR/eval_${DATASET_NAME}_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Per-Dataset Evaluation" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Dataset Name: $DATASET_NAME" | tee -a "$LOG_FILE"
echo "CMDC Fold: $CMDC_FOLD" | tee -a "$LOG_FILE"
echo "Model Family: $MODEL_FAMILY" | tee -a "$LOG_FILE"
echo "Prompt Mode: $PROMPT_MODE" | tee -a "$LOG_FILE"
echo "DAIC Eval Level: $DAIC_WOZ_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "DAIC Eval Mode: $DAIC_WOZ_EVAL_MODE" | tee -a "$LOG_FILE"
echo "DAIC Person Threshold: $DAIC_WOZ_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "EATD Eval Level: $EATD_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "EATD Eval Mode: $EATD_EVAL_MODE" | tee -a "$LOG_FILE"
echo "EATD Person Threshold: $EATD_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "CMDC Eval Level: $CMDC_EVAL_LEVEL" | tee -a "$LOG_FILE"
echo "CMDC Eval Mode: $CMDC_EVAL_MODE" | tee -a "$LOG_FILE"
echo "CMDC Person Threshold: $CMDC_PERSON_THRESHOLD" | tee -a "$LOG_FILE"
echo "Checkpoint Mode: $CHECKPOINT_MODE" | tee -a "$LOG_FILE"
echo "PEFT Path: $PEFT_PATH" | tee -a "$LOG_FILE"
echo "Adapter Path: ${ADAPTER_PATH:-auto}" | tee -a "$LOG_FILE"
echo "Log Dir: $LOG_DIR" | tee -a "$LOG_FILE"
echo "Results Dir: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

bash eval.sh "$PEFT_PATH" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Evaluation Complete!" | tee -a "$LOG_FILE"
echo "Stdout: $SLURM_STDOUT_FILE" | tee -a "$LOG_FILE"
echo "Stderr: $SLURM_STDERR_FILE" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
