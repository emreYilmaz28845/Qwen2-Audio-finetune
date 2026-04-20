#!/bin/bash
#SBATCH -J qwen-eval-2
#SBATCH -A etur92
#SBATCH -q acc_ehpc
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -o ./logs/%x-%j.out
#SBATCH -e ./logs/%x-%j.err
#SBATCH --chdir=/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune

module purge
module load bsc/1.0
module load miniforge/24.3.0-0
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs

export AUDIOLLM_DATASETS_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Set the checkpoint to evaluate ---
# Override via: sbatch --export=PEFT_PATH=/path/to/checkpoint eval.slurm
# Base model baseline: sbatch --export=PEFT_PATH=none eval.slurm
export PEFT_PATH="${PEFT_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/1e-05_20260330_023140/07-15}"

echo "Evaluating checkpoint: $PEFT_PATH"

bash eval.sh "$PEFT_PATH"
