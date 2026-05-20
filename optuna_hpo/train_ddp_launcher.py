"""
DDP launcher for Optuna trials.

This script is launched by torchrun and runs the training with DDP.
It reads the trial configuration from a JSON file.
"""

import argparse
import importlib.util
import json
import os
import sys

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def _load_repo_config_class():
    config_path = os.path.join(PROJECT_ROOT, "config", "config.py")
    spec = importlib.util.spec_from_file_location("audiollm_repo_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config


Config = _load_repo_config_class()
from optuna_hpo.train_ddp import train_ddp
from utils.grouped_eval import GROUPED_DATASET_NAMES


def derive_input_mode(model_family: str):
    if model_family == "audio":
        return "audiotext"
    return "textonly"


def _grouped_eval_values(config_data, dataset_name: str):
    prefix = dataset_name.upper()
    return (
        config_data.get(f"{dataset_name}_eval_level", os.environ.get(f"{prefix}_EVAL_LEVEL", "person")),
        config_data.get(f"{dataset_name}_eval_mode", os.environ.get(f"{prefix}_EVAL_MODE", "majority_vote")),
        float(config_data.get(f"{dataset_name}_person_threshold", os.environ.get(f"{prefix}_PERSON_THRESHOLD", "0.5"))),
    )


def main():
    parser = argparse.ArgumentParser(description="DDP Optuna Trial Launcher")
    parser.add_argument("--config-file", type=str, required=True, help="Path to trial config JSON file")
    args = parser.parse_args()
    
    # Read config file
    with open(args.config_file, 'r') as f:
        config_data = json.load(f)
    
    trial_number = config_data["trial_number"]
    trial_params = config_data["trial_params"]
    model_path = config_data["model_path"]
    train_data_path = config_data["train_data_path"]
    eval_data_path = config_data["eval_data_path"]
    save_path = config_data["save_path"]
    dataset_name = config_data.get("dataset_name", os.environ.get("DATASET_NAME", "merged"))
    model_family = os.environ.get("MODEL_FAMILY", "text").strip().lower()
    input_mode = config_data.get("input_mode", derive_input_mode(model_family))
    progress_file = config_data.get("progress_file")
    result_file = config_data.get("result_file")
    stop_file = config_data.get("stop_file")
    grouped_cfg = {
        grouped_dataset_name: _grouped_eval_values(config_data, grouped_dataset_name)
        for grouped_dataset_name in GROUPED_DATASET_NAMES
    }
    
    # Create config for training
    cfg = Config()
    
    # Set hyperparameters from trial
    cfg.train.lr = trial_params["lr"]
    cfg.train.batch_size = trial_params["batch_size"]
    cfg.peft.r = trial_params["lora_r"]
    cfg.peft.lora_alpha = trial_params["lora_alpha"]
    
    # Set paths from config file (from SLURM environment variables)
    cfg.env.model_path = model_path
    cfg.data.train_data_path = train_data_path
    cfg.data.eval_data_path = eval_data_path
    cfg.data.dataset_name = dataset_name
    cfg.env.save_path = save_path
    
    # Override prompt/scp/task file paths from environment variables
    default_prompt_file = (
        "merged_multiprompt_textonly.jsonl"
        if input_mode == "textonly"
        else "merged_multiprompt.jsonl"
    )
    train_prompt_file = os.environ.get("TRAIN_PROMPT_FILE", default_prompt_file)
    eval_prompt_file = os.environ.get("EVAL_PROMPT_FILE", default_prompt_file)
    train_task_file = os.environ.get("TRAIN_TASK_FILE", "merged_multitask.jsonl")
    eval_task_file = os.environ.get("EVAL_TASK_FILE", "merged_multitask.jsonl")
    default_scp_file = os.environ.get("SCP_FILE_DEFAULT", "merged.scp")
    train_scp_file = os.environ.get("TRAIN_SCP_FILE", default_scp_file)
    eval_scp_file = os.environ.get("EVAL_SCP_FILE", default_scp_file)
    wav_type = os.environ.get("WAV_TYPE", "wav")

    cfg.data.train_prompt_path = os.path.join(train_data_path, train_prompt_file)
    cfg.data.val_prompt_path = os.path.join(eval_data_path, eval_prompt_file)
    cfg.data.train_task_filename = train_task_file
    cfg.data.eval_task_filename = eval_task_file
    cfg.data.train_scp_filename = train_scp_file
    cfg.data.eval_scp_filename = eval_scp_file
    cfg.data.wav_type = wav_type
    cfg.env.progress_file = progress_file or ""
    cfg.env.stop_file = stop_file or ""
    cfg.eval.daic_eval_level, cfg.eval.daic_eval_mode, cfg.eval.daic_person_threshold = grouped_cfg["daic_woz"]
    cfg.eval.eatd_eval_level, cfg.eval.eatd_eval_mode, cfg.eval.eatd_person_threshold = grouped_cfg["eatd"]
    cfg.eval.cmdc_eval_level, cfg.eval.cmdc_eval_mode, cfg.eval.cmdc_person_threshold = grouped_cfg["cmdc"]
    
    # Run training
    trial_name = f"trial_{trial_number:03d}_lr{trial_params['lr']:.0e}_bs{trial_params['batch_size']}_r{trial_params['lora_r']}_a{trial_params['lora_alpha']}"
    
    try:
        best_f1 = train_ddp(cfg, trial_name=trial_name, input_mode=input_mode)
    except Exception as e:
        print(f"Error during training: {e}")
        best_f1 = -1.0
    
    # Under torchrun every rank executes this launcher. Only rank 0 should
    # write the shared temp result file, otherwise concurrent writes can
    # concatenate JSON payloads and break the Optuna parent process.
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        results = {
            "trial_number": trial_number,
            "best_f1": best_f1,
            "trial_params": trial_params,
            "input_mode": input_mode,
            "dataset_name": dataset_name,
            "best_eval_summary": getattr(cfg.env, "best_eval_summary", None),
            "grouped_eval": {
                grouped_dataset_name: {
                    "level": grouped_cfg[grouped_dataset_name][0],
                    "mode": grouped_cfg[grouped_dataset_name][1],
                    "threshold": grouped_cfg[grouped_dataset_name][2],
                }
                for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES)
            },
        }

        if result_file:
            tmp_path = f"{result_file}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
            os.replace(tmp_path, result_file)

            print(f"\nResults saved to: {result_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
