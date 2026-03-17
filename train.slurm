#!/bin/bash
#SBATCH -J qwen-train
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
#SBATCH -J qwen-cv


module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs output_model

export TRAIN_DATA_PATH=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/daic_woz/train
export EVAL_DATA_PATH=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/data/daic_woz/val
export TRAIN_PROMPT_FILE=daic_woz_multiprompt.jsonl
export EVAL_PROMPT_FILE=daic_woz_multiprompt.jsonl
export AUDIOLLM_DATASETS_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True


export EVAL_STEP="${EVAL_STEP:-50}"
export GRAD_ACCUMULATE_STEP="${GRAD_ACCUMULATE_STEP:-5}"
export TRAIN_EPOCH="${TRAIN_EPOCH:-20}"
export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct}"

bash train.sh
