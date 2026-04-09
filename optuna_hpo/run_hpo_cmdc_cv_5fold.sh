#!/bin/bash

# =================================
# Quick-start script for CMDC 5-fold cross-validated Optuna HPO
# Run from optuna_hpo folder like: ./run_hpo_cmdc_cv_5fold.sh
# =================================

set -e

cd "$(dirname "$0")" || exit 1
cd .. # Go to parent directory

echo "======================================"
echo "CMDC 5-Fold Cross-Validated HPO"
echo "======================================"
echo ""

N_TRIALS=${1:-20}
STUDY_NAME=${2:-cmdc_textonly_cv_hpo_$(date +%Y%m%d_%H%M%S)}
FOLDS="${FOLDS:-fold1 fold2 fold3 fold4 fold5}"

echo "Number of CV trials: $N_TRIALS"
echo "Study name: $STUDY_NAME"
echo "Folds: $FOLDS"
echo ""

if [ -d "/gpfs/projects/etur92" ]; then
    echo "Detected MN5 cluster. Submitting SLURM job..."
    echo ""

    N_TRIALS="$N_TRIALS" STUDY_NAME="$STUDY_NAME" FOLDS="$FOLDS" \
        sbatch optuna_hpo/train_hpo_cmdc_cv_5fold.slurm
    echo "SLURM job submitted! Check logs/optuna*.out for progress."
else
    echo "Not on MN5 cluster. Running locally (requires 4 GPUs)..."
    echo ""

    mkdir -p logs output_model optuna_studies

    CMDC_ROOT="${CMDC_ROOT:-$(pwd)/data/cmdc}"
    MODEL_PATH="${MODEL_PATH:-/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct}"
    SAVE_ROOT="${SAVE_ROOT:-$(pwd)/output_model/optuna_cmdc_cv_5fold}"
    STORAGE_PATH="${STORAGE_PATH:-optuna_studies}"
    NUM_GPUS="${NUM_GPUS:-4}"

    export MODEL_PATH
    export CMDC_ROOT
    export SAVE_ROOT
    export STORAGE_PATH
    export FOLDS
    export NUM_GPUS

    python optuna_hpo/hpo_cv_5fold.py \
        --n-trials "$N_TRIALS" \
        --study-name "$STUDY_NAME" \
        --storage-path "$STORAGE_PATH" \
        --cmdc-root "$CMDC_ROOT" \
        --folds "$FOLDS" \
        --save-root "$SAVE_ROOT" \
        --num-gpus "$NUM_GPUS"
fi

echo ""
echo "Results will be saved in: optuna_studies/${STUDY_NAME}_results.json"
