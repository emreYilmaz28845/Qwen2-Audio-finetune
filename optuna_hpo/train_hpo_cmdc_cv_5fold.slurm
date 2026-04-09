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

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model optuna_studies

export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
export CMDC_ROOT="${CMDC_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/cmdc}"
export SAVE_ROOT="${SAVE_ROOT:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_cmdc_cv_5fold}"
export STORAGE_PATH="${STORAGE_PATH:-optuna_studies}"
export NUM_GPUS="${NUM_GPUS:-4}"
export FOLDS="${FOLDS:-fold1 fold2 fold3 fold4 fold5}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

N_TRIALS="${N_TRIALS:-20}"
STUDY_NAME="${STUDY_NAME:-cmdc_textonly_cv_hpo_$(date +%Y%m%d)}"

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="logs/optuna_cmdc_cv_5fold_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC 5-Fold Cross-Validated Hyperparameter Tuning" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of CV Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Folds: $FOLDS" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "Save Root: $SAVE_ROOT" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python optuna_hpo/hpo_cv_5fold.py \
    --n-trials "$N_TRIALS" \
    --study-name "$STUDY_NAME" \
    --storage-path "$STORAGE_PATH" \
    --cmdc-root "$CMDC_ROOT" \
    --folds "$FOLDS" \
    --save-root "$SAVE_ROOT" \
    --num-gpus "$NUM_GPUS" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "CMDC CV-HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
