#!/bin/bash

# =================================
# Quick-start script for Optuna HPO
# Run from optuna_hpo folder like: ./run_hpo.sh
# =================================

set -e

cd "$(dirname "$0")" || exit 1
cd .. # Go to parent directory

echo "======================================"
echo "Qwen2-7B Text-Only Hyperparameter Search (4 GPU DDP)"
echo "======================================"
echo ""

# Parse arguments
N_TRIALS=${1:-20}
STUDY_NAME=${2:-qwen2_textonly_hpo_$(date +%Y%m%d_%H%M%S)}

echo "Number of trials: $N_TRIALS"
echo "Study name: $STUDY_NAME"
echo ""

# Check if on MN5
if [ -d "/gpfs/projects/etur92" ]; then
    echo "Detected MN5 cluster. Submitting SLURM job..."
    echo ""
    
    N_TRIALS="$N_TRIALS" STUDY_NAME="$STUDY_NAME" sbatch optuna_hpo/train_hpo.slurm
    echo "SLURM job submitted! Check logs/optuna*.out for progress."
else
    echo "Not on MN5 cluster. Running locally (requires 4 GPUs)..."
    echo ""
    
    mkdir -p logs output_model optuna_studies
    
    python optuna_hpo/hpo.py \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage-path "optuna_studies"
fi

echo ""
echo "Results will be saved in: optuna_studies/${STUDY_NAME}_results.json"
