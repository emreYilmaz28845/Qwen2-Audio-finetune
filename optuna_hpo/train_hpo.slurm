#!/bin/bash
#SBATCH -J qwen-optuna-hpo
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
# Optuna Hyperparameter Tuning on MN5 (4 GPU DDP)
# =================================

module purge
module load bsc/1.0
module load miniforge/24.3.0-0

# Activate your Optuna-enabled venv
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model optuna_studies

# =================================
# Environment Variables
# =================================

export INPUT_MODE="${INPUT_MODE:-textonly}" # textonly or audiotext
if [ "$INPUT_MODE" = "audiotext" ]; then
    export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
    export TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-merged_multiprompt.jsonl}"
    export EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-merged_multiprompt.jsonl}"
else
    export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
    export TRAIN_PROMPT_FILE="${TRAIN_PROMPT_FILE:-merged_multiprompt_textonly.jsonl}"
    export EVAL_PROMPT_FILE="${EVAL_PROMPT_FILE:-merged_multiprompt_textonly.jsonl}"
fi
export TRAIN_DATA_PATH="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/merged/train"
export EVAL_DATA_PATH="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/merged/val"
export SAVE_PATH="/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna"
export TRAIN_SCP_FILE="${TRAIN_SCP_FILE:-merged.scp}"
export EVAL_SCP_FILE="${EVAL_SCP_FILE:-merged.scp}"
export WAV_TYPE="${WAV_TYPE:-wav}"
export NUM_GPUS="${NUM_GPUS:-4}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# File names for prompts and tasks
export TRAIN_TASK_FILE="merged_multitask.jsonl"
export EVAL_TASK_FILE="merged_multitask.jsonl"

# =================================
# Optuna Configuration
# =================================

# Number of trials to run
N_TRIALS="${N_TRIALS:-20}"

# Study name (for resuming if needed)
STUDY_NAME="${STUDY_NAME:-qwen2_textonly_hpo_$(date +%Y%m%d)}"

# Storage path for Optuna database
STORAGE_PATH="optuna_studies"

# =================================
# Logging
# =================================

TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)
LOG_FILE="logs/optuna_run_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Qwen2-7B Text-Only Hyperparameter Tuning (4 GPU DDP)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"
echo "Number of Trials: $N_TRIALS" | tee -a "$LOG_FILE"
echo "Study Name: $STUDY_NAME" | tee -a "$LOG_FILE"
echo "Input Mode: $INPUT_MODE" | tee -a "$LOG_FILE"
echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# =================================
# Run Optuna Optimization
# =================================

python optuna_hpo/hpo.py \
    --n-trials "$N_TRIALS" \
    --study-name "$STUDY_NAME" \
    --storage-path "$STORAGE_PATH" \
    --input-mode "$INPUT_MODE" 2>&1 | tee -a "$LOG_FILE"

# =================================
# Post-run summary
# =================================

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "HPO Run Complete!" | tee -a "$LOG_FILE"
echo "Study database: $STORAGE_PATH/${STUDY_NAME}.db" | tee -a "$LOG_FILE"
echo "Results: $STORAGE_PATH/${STUDY_NAME}_results.json" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
