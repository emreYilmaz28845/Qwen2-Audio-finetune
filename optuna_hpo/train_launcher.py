"""
Launcher wrapper for DDP-based Optuna trial training.

This script is called by Optuna to launch training on 4 GPUs using torchrun.
"""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time

import optuna


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def _read_progress(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    step = payload.get("step")
    metric = payload.get("metric")
    if not isinstance(step, int):
        return None
    if not isinstance(metric, (int, float)):
        return None
    return {"step": step, "metric": float(metric)}


def _terminate_process_group(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _cleanup_temp_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def launch_ddp_training(
    trial: optuna.Trial | None,
    trial_params: dict,
    trial_number: int,
    model_path: str,
    train_data_path: str,
    eval_data_path: str,
    save_path: str,
    dataset_name: str,
    input_mode: str = "textonly",
    num_gpus: int = 4,
    enable_pruning: bool = True,
    prune_mode: str = "eval",
    daic_eval_level: str = "person",
    daic_eval_mode: str = "majority_vote",
    daic_person_threshold: float = 0.5,
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
    
    temp_dir = tempfile.gettempdir()
    trial_config_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}.json")
    progress_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}_progress.json")
    result_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}_result.json")
    stop_file = os.path.join(temp_dir, f"optuna_trial_{trial_number}_stop")

    config_data = {
        "trial_number": trial_number,
        "trial_params": trial_params,
        "model_path": model_path,
        "train_data_path": train_data_path,
        "eval_data_path": eval_data_path,
        "save_path": save_path,
        "dataset_name": dataset_name,
        "input_mode": input_mode,
        "progress_file": progress_file,
        "result_file": result_file,
        "stop_file": stop_file,
        "prune_mode": prune_mode,
        "daic_eval_level": daic_eval_level,
        "daic_eval_mode": daic_eval_mode,
        "daic_person_threshold": float(daic_person_threshold),
    }

    _cleanup_temp_files(trial_config_file, progress_file, result_file, stop_file)
    _write_json_atomic(trial_config_file, config_data)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        os.path.join(script_dir, "train_ddp_launcher.py"),
        f"--config-file={trial_config_file}",
    ]

    print(f"\n{'='*70}")
    print(f"Launching Trial {trial_number} with DDP on {num_gpus} GPUs")
    print(f"  LR: {trial_params['lr']:.2e}")
    print(f"  Batch Size: {trial_params['batch_size']}")
    print(f"  LoRA R: {trial_params['lora_r']}")
    print(f"  LoRA Alpha: {trial_params['lora_alpha']}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Input Mode: {input_mode}")
    print(f"  Prune Mode: {prune_mode}")
    print(f"  DAIC Eval Level: {daic_eval_level}")
    print(f"  DAIC Eval Mode: {daic_eval_mode}")
    print(f"  DAIC Person Threshold: {daic_person_threshold}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    try:
        process = subprocess.Popen(cmd, start_new_session=True)
        last_reported_step = -1

        while True:
            progress = _read_progress(progress_file)
            if (
                enable_pruning
                and trial is not None
                and prune_mode == "eval"
                and progress is not None
                and progress["step"] > last_reported_step
            ):
                last_reported_step = progress["step"]
                trial.report(progress["metric"], step=progress["step"])
                if trial.should_prune():
                    _write_json_atomic(stop_file, {"reason": "optuna_pruned", "step": progress["step"]})
                    _terminate_process_group(process)
                    raise optuna.TrialPruned(
                        f"Trial {trial_number} pruned at eval step {progress['step']} with F1={progress['metric']:.4f}"
                    )

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    print(f"Error launching trial {trial_number}: torchrun exited with code {return_code}")
                    return -1.0
                break
            time.sleep(5)

        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as handle:
                results = json.load(handle)
            best_f1 = results.get("best_f1", -1.0)
            if trial is not None:
                best_eval_summary = results.get("best_eval_summary")
                if best_eval_summary is not None:
                    trial.set_user_attr("best_eval_summary", best_eval_summary)
                teacher_forced_eval_summary = results.get("teacher_forced_eval_summary")
                if teacher_forced_eval_summary is not None:
                    trial.set_user_attr("teacher_forced_eval_summary", teacher_forced_eval_summary)
        else:
            print(f"Warning: Result file not found at {result_file}")
            best_f1 = -1.0

        return best_f1
    finally:
        _cleanup_temp_files(trial_config_file, progress_file, result_file, stop_file)


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
        trial=None,
        trial_params=trial_params,
        trial_number=args.trial_number,
        model_path=os.environ.get("MODEL_PATH", "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"),
        train_data_path=os.environ.get("TRAIN_DATA_PATH", "Qwen2-Audio-finetune/data/merged/train"),
        eval_data_path=os.environ.get("EVAL_DATA_PATH", "Qwen2-Audio-finetune/data/merged/val"),
        save_path=os.environ.get("SAVE_PATH", "output_model/optuna"),
        dataset_name=os.environ.get("DATASET_NAME", "merged"),
        num_gpus=args.num_gpus,
        enable_pruning=False,
        prune_mode="disabled",
        daic_eval_level=os.environ.get("DAIC_EVAL_LEVEL", "person"),
        daic_eval_mode=os.environ.get("DAIC_EVAL_MODE", "majority_vote"),
        daic_person_threshold=float(os.environ.get("DAIC_PERSON_THRESHOLD", "0.5")),
    )

    print(f"\nTrial {args.trial_number} completed with F1 = {best_f1:.4f}")
