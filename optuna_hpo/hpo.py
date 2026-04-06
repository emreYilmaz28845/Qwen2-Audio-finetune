"""
Optuna Hyperparameter Optimization for Text-Only Qwen2-7B Training

This package uses Optuna to search for the best hyperparameters:
- Learning Rate (LR)
- Batch Size
- LoRA R
- LoRA Alpha

Each trial runs a DDP training on 4 GPUs and reports the best validation F1 score.
"""

import optuna
from optuna.trial import TrialState
from omegaconf import OmegaConf
import torch
import os
import json
import time
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from optuna_hpo.train_launcher import launch_ddp_training


# ===============================
# Configure logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


# ===============================
# Objective function for Optuna
# ===============================
def objective(trial: optuna.Trial):
    """
    Objective function for Optuna optimization.
    Suggests hyperparameters and runs DDP training to get best F1 score.
    """
    
    # ===============================
    # Suggest hyperparameters
    # ===============================
    
    # Learning Rate: logarithmic scale from 1e-6 to 1e-3
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    
    # Batch Size: 1, 2, 4, 8, 16
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    
    # LoRA R: rank dimension
    lora_r = trial.suggest_int("lora_r", 8, 16, step=4)
    
    # LoRA Alpha: scaling factor
    lora_alpha = trial.suggest_int("lora_alpha", 8, 32, step=8)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Trial {trial.number}: Starting Optuna trial")
    logger.info(f"  LR: {lr:.2e}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  LoRA R: {lora_r}")
    logger.info(f"  LoRA Alpha: {lora_alpha}")
    logger.info(f"{'='*70}\n")
    
    try:
        # ===============================
        # Set environment variables
        # ===============================
        model_path = os.environ.get(
            "MODEL_PATH",
            "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"
        )
        train_data_path = os.environ.get(
            "TRAIN_DATA_PATH",
            "Qwen2-Audio-finetune/data/merged/train"
        )
        eval_data_path = os.environ.get(
            "EVAL_DATA_PATH",
            "Qwen2-Audio-finetune/data/merged/val"
        )
        save_path = os.environ.get("SAVE_PATH", "output_model/optuna")
        
        # ===============================
        # Launch DDP training
        # ===============================
        trial_params = {
            "lr": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        }
        
        best_f1 = launch_ddp_training(
            trial_params=trial_params,
            trial_number=trial.number,
            model_path=model_path,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            save_path=save_path,
            num_gpus=4
        )
        
        # ===============================
        # Log results
        # ===============================
        logger.info(f"\nTrial {trial.number} completed with Best F1: {best_f1:.4f}\n")
        
        return best_f1
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}")
        logger.exception(e)
        return -1.0  # Return negative score for failed trials


# ===============================
# Main optimization function
# ===============================
def run_optimization(
    n_trials=20,
    study_name="qwen2_textonly_hpo",
    storage_path="optuna_studies"
):
    """
    Run Optuna hyperparameter optimization.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the Optuna study
        storage_path: Directory to store Optuna study database
    """
    
    # ===============================
    # Create storage directory
    # ===============================
    os.makedirs(storage_path, exist_ok=True)
    
    # ===============================
    # Create Optuna study
    # ===============================
    storage = f"sqlite:///{os.path.abspath(storage_path)}/{study_name}.db"
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Optuna Hyperparameter Optimization")
    logger.info(f"  Study Name: {study_name}")
    logger.info(f"  Number of Trials: {n_trials}")
    logger.info(f"  Storage: {storage}")
    logger.info(f"{'='*70}\n")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize F1 score
        load_if_exists=True,  # Resume if study exists
    )
    
    # ===============================
    # Run optimization
    # ===============================
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)  # Sequential execution
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    
    # ===============================
    # Print results
    # ===============================
    logger.info(f"\n{'='*70}")
    logger.info(f"Optimization Results")
    logger.info(f"{'='*70}\n")
    
    # Get best trial
    best_trial = study.best_trial
    logger.info(f"Best Trial: #{best_trial.number}")
    logger.info(f"Best F1 Score: {best_trial.value:.4f}\n")
    logger.info("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Print all completed trials
    logger.info(f"\n{'='*70}")
    logger.info("All Completed Trials (sorted by F1 score)")
    logger.info(f"{'='*70}\n")
    
    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].sort_values(
        'value', ascending=False
    )
    
    logger.info(completed_trials[['number', 'value', 'params_lr', 'params_batch_size', 
                                   'params_lora_r', 'params_lora_alpha']].to_string())
    
    # Save results to JSON
    results_file = f"{storage_path}/{study_name}_results.json"
    results = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_trial_number": best_trial.number,
        "best_f1": float(best_trial.value),
        "best_params": best_trial.params,
        "all_trials": []
    }
    
    for _, row in completed_trials.iterrows():
        results["all_trials"].append({
            "trial_number": int(row['number']),
            "f1": float(row['value']),
            "params": {
                "lr": row['params_lr'],
                "batch_size": int(row['params_batch_size']),
                "lora_r": int(row['params_lora_r']),
                "lora_alpha": int(row['params_lora_alpha']),
            }
        })
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    return study, best_trial


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for Qwen2-7B Text-Only Training"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="qwen2_textonly_hpo",
        help="Name of the Optuna study (default: qwen2_textonly_hpo)"
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="optuna_studies",
        help="Directory to store Optuna study database (default: optuna_studies)"
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    logger.info(f"GPU Available: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run optimization
    study, best_trial = run_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage_path
    )
