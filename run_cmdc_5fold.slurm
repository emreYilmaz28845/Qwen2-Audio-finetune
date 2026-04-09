#!/bin/bash
#SBATCH -J qwen-cmdc-5fold
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH -o ./logs/slurm-%x-%j.out
#SBATCH -e ./logs/slurm-%x-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model

export CMDC_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/cmdc
export AUDIOLLM_DATASETS_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NUM_GPUS="${NUM_GPUS:-4}"

export FOLDS="${FOLDS:-1 2 3 4 5}"
export EVAL_STEP="${EVAL_STEP:-50}"
export LR="${LR:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUMULATE_STEP="${GRAD_ACCUMULATE_STEP:-5}"
export TRAIN_EPOCH="${TRAIN_EPOCH:-20}"
export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"
export SAVE_PATH="${SAVE_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/cmdc_5fold}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen2-audio-cmdc-5fold}"
export WANDB_GROUP="${WANDB_GROUP:-cmdc_5fold}"
export WANDB_RUN_NAME_PREFIX="${WANDB_RUN_NAME_PREFIX:-cmdc}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SAFE_LR="${LR//\//-}"
LOG_RUN_DIR="${LOG_RUN_DIR:-logs/cmdc_5fold_${SAFE_LR}_${RUN_TIMESTAMP}}"
mkdir -p "$LOG_RUN_DIR"

# Slurm resolves #SBATCH output paths before the script starts, so redirect
# the job output here to get per-run log folders based on LR and timestamp.
exec > >(tee -a "$LOG_RUN_DIR/train-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_RUN_DIR/train-${SLURM_JOB_ID}.err" >&2)

echo "Logging this run under: $LOG_RUN_DIR"
echo "Learning rate: $LR"
echo "Model: $MODEL_PATH"
echo "Folds: $FOLDS"
echo "Started at: $RUN_TIMESTAMP"

bash run_cmdc_5fold.sh
