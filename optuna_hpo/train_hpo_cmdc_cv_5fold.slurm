#!/bin/bash
#SBATCH -J qwen-optuna-cmdc-cv
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 24:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH -o ./logs/optuna-%x-%j.out
#SBATCH -e ./logs/optuna-%x-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune

# =================================
# CMDC 5-fold cross-validated Optuna Hyperparameter Tuning on MN5
# =================================

set -e
set -o pipefail

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model optuna_studies

export INPUT_MODE="${INPUT_MODE:-audiotext}" # textonly or audiotext
if [ "$INPUT_MODE" = "audiotext" ]; then
    export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
else
    export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
fi
export CMDC_ROOT="${CMDC_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/cmdc}"
export SAVE_ROOT="${SAVE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_cmdc_cv_5fold}"
export STORAGE_PATH="${STORAGE_PATH:-optuna_studies}"
export NUM_GPUS="${NUM_GPUS:-4}"
export FOLDS="${FOLDS:-fold1 fold2 fold3 fold4 fold5}"
export STUDY_MODE="${STUDY_MODE:-cv_mean}" # per_fold or cv_mean
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AUDIOLLM_ENABLE_COLLATE_DEBUG="${AUDIOLLM_ENABLE_COLLATE_DEBUG:-0}"
export AUDIOLLM_COLLATE_DEBUG_LIMIT="${AUDIOLLM_COLLATE_DEBUG_LIMIT:-6}"
export AUDIOLLM_ENABLE_METRIC_DEBUG="${AUDIOLLM_ENABLE_METRIC_DEBUG:-0}"
export AUDIOLLM_METRIC_DEBUG_LIMIT="${AUDIOLLM_METRIC_DEBUG_LIMIT:-12}"

N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-cmdc_${INPUT_MODE}_${STUDY_MODE}_cv_hpo_$(date +%Y%m%d_%H%M%S)}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="logs/optuna_cmdc_cv_5fold_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC 5-Fold Cross-Validated Hyperparameter Tuning" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of CV Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Study Mode: $STUDY_MODE" | tee -a "$LOG_FILE"
echo "Input Mode: $INPUT_MODE" | tee -a "$LOG_FILE"
echo "Folds: $FOLDS" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Save Root: $SAVE_ROOT" | tee -a "$LOG_FILE"
echo "Collate Debug: $AUDIOLLM_ENABLE_COLLATE_DEBUG (limit=$AUDIOLLM_COLLATE_DEBUG_LIMIT)" | tee -a "$LOG_FILE"
echo "Metric Debug: $AUDIOLLM_ENABLE_METRIC_DEBUG (limit=$AUDIOLLM_METRIC_DEBUG_LIMIT)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python optuna_hpo/hpo_cv_5fold.py \
    --n-trials "$N_TRIALS" \
    --study-name "$STUDY_NAME" \
    --storage-path "$STORAGE_PATH" \
    --cmdc-root "$CMDC_ROOT" \
    --folds "$FOLDS" \
    --save-root "$SAVE_ROOT" \
    --num-gpus "$NUM_GPUS" \
    --input-mode "$INPUT_MODE" \
    --study-mode "$STUDY_MODE" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC CV-HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
