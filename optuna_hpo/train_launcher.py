"""
Launcher wrapper for DDP-based Optuna trial training.

This script is called by Optuna to launch training on 4 GPUs using torchrun.
"""

import subprocess
import sys
import os
import json
import pickle
import tempfile
import argparse


def launch_ddp_training(
    trial_params: dict,
    trial_number: int,
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    save_path: str,
    num_gpus: int = 4
) -> float:
    """
    Launch DDP training for a single Optuna trial.
    
    Args:
        trial_params: Dict with keys 'lr', 'batch_size', 'lora_r', 'lora_alpha'
        trial_number: Trial number from Optuna
        model_path: Path to Qwen2-7B model
        train_data_path: Path to training data
        eval_data_path: Path to evaluation data
        save_path: Base path to save outputs
        num_gpus: Number of GPUs to use (default: 4)
        
    Returns:
        best_f1: Best F1 score from the trial
    """
    
    # Create temporary JSON file with trial params
    temp_dir = tempfile.gettempdir()
    trial_config_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}.json")
    
    config_data = {
        "trial_number": trial_number,
        "trial_params": trial_params,
        "model_path": model_path,
        "train_data_path": train_data_path,
        "eval_data_path": eval_data_path,
        "save_path": save_path,
    }
    
    with open(trial_config_file, 'w') as f:
        json.dump(config_data, f)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build torchrun command
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        os.path.join(script_dir, "train_ddp_launcher.py"),
        f"--config-file={trial_config_file}"
    ]
    
    print(f"\n{'='*70}")
    print(f"Launching Trial {trial_number} with DDP on {num_gpus} GPUs")
    print(f"  LR: {trial_params['lr']:.2e}")
    print(f"  Batch Size: {trial_params['batch_size']}")
    print(f"  LoRA R: {trial_params['lora_r']}")
    print(f"  LoRA Alpha: {trial_params['lora_alpha']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    # Run torchrun
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Error launching trial {trial_number}: {e}")
        return -1.0
    
    # Read results from output files
    result_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}_result.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
        best_f1 = results.get("best_f1", -1.0)
        os.remove(result_file)
    else:
        print(f"Warning: Result file not found at {result_file}")
        best_f1 = -1.0
    
    # Clean up config file
    if os.path.exists(trial_config_file):
        os.remove(trial_config_file)
    
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Optuna Trial Launcher")
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    
    args = parser.parse_args()
    
    trial_params = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    
    best_f1 = launch_ddp_training(
        trial_params=trial_params,
        trial_number=args.trial_number,
        model_path=os.environ.get("MODEL_PATH", "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"),
        train_data_path=os.environ.get("TRAIN_DATA_PATH", "Qwen2-Audio-finetune/data/merged/train"),
        eval_data_path=os.environ.get("EVAL_DATA_PATH", "Qwen2-Audio-finetune/data/merged/val"),
        save_path=os.environ.get("SAVE_PATH", "output_model/optuna"),
        num_gpus=args.num_gpus
    )
    
    print(f"\nTrial {args.trial_number} completed with F1 = {best_f1:.4f}")
