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
source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate

cd /gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune
mkdir -p logs

export AUDIOLLM_DATASETS_ROOT=/gpfs/projects/etur92/ozu647717/AudioLLM/Datasets
export PYTORCH_ALLOC_CONF=expandable_segments:True

# --- Set the checkpoint to evaluate ---
# Override via: sbatch --export=PEFT_PATH=/path/to/checkpoint eval.slurm
export PEFT_PATH="${PEFT_PATH:-/gpfs/projects/etur92/ozu647717/AudioLLM/Qwen2-Audio-finetune/output_model/optuna_cmdc_cv_5fold/trial_013/fold3_trial_trial_013_lr9e-04_bs2_r12_a32/best_model}"
export MODEL_FAMILY="${MODEL_FAMILY:-audio}"
export PROMPT_MODE="${PROMPT_MODE:-audiotext}"
export CHECKPOINT_MODE="${CHECKPOINT_MODE:-auto}"
export ADAPTER_PATH="${ADAPTER_PATH:-}"

echo "Evaluating checkpoint: $PEFT_PATH"
echo "MODEL_FAMILY=$MODEL_FAMILY"
echo "PROMPT_MODE=$PROMPT_MODE"
echo "CHECKPOINT_MODE=$CHECKPOINT_MODE"
if [[ -n "$ADAPTER_PATH" ]]; then
    echo "ADAPTER_PATH=$ADAPTER_PATH"
else
    echo "ADAPTER_PATH=(auto)"
fi

bash eval.sh "$PEFT_PATH"
