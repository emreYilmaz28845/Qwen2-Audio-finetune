"""
Cross-validated Optuna hyperparameter optimization for CMDC text-only training.

One Optuna trial samples a single hyperparameter set and evaluates it across
all requested folds. The trial objective is the mean fold F1.
"""

import argparse
import json
import logging
import math
import os
import statistics
import sys
from pathlib import Path

import optuna
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optuna_hpo.train_launcher import launch_ddp_training


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_folds(folds_text: str):
    return [token.strip() for token in folds_text.split() if token.strip()]


def get_fold_config(cmdc_root: str, fold_name: str):
    return {
        "train_data_path": os.path.join(cmdc_root, fold_name, "train"),
        "eval_data_path": os.path.join(cmdc_root, fold_name, "test"),
        "train_prompt_file": f"{fold_name}_multiprompt_textonly.jsonl",
        "eval_prompt_file": f"{fold_name}_multiprompt_textonly.jsonl",
        "train_task_file": f"{fold_name}_multitask.jsonl",
        "eval_task_file": f"{fold_name}_multitask.jsonl",
    }


def run_fold_trial(trial, trial_params, fold_name, fold_cfg, model_path, save_root, num_gpus):
    previous_env = {
        "TRAIN_PROMPT_FILE": os.environ.get("TRAIN_PROMPT_FILE"),
        "EVAL_PROMPT_FILE": os.environ.get("EVAL_PROMPT_FILE"),
        "TRAIN_TASK_FILE": os.environ.get("TRAIN_TASK_FILE"),
        "EVAL_TASK_FILE": os.environ.get("EVAL_TASK_FILE"),
    }

    os.environ["TRAIN_PROMPT_FILE"] = fold_cfg["train_prompt_file"]
    os.environ["EVAL_PROMPT_FILE"] = fold_cfg["eval_prompt_file"]
    os.environ["TRAIN_TASK_FILE"] = fold_cfg["train_task_file"]
    os.environ["EVAL_TASK_FILE"] = fold_cfg["eval_task_file"]

    fold_save_path = os.path.join(save_root, f"trial_{trial.number:03d}", fold_name)

    try:
        best_f1 = launch_ddp_training(
            trial_params=trial_params,
            trial_number=trial.number,
            model_path=model_path,
            train_data_path=fold_cfg["train_data_path"],
            eval_data_path=fold_cfg["eval_data_path"],
            save_path=fold_save_path,
            num_gpus=num_gpus,
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return {
        "fold": fold_name,
        "best_f1": float(best_f1),
        "save_path": fold_save_path,
        "train_data_path": fold_cfg["train_data_path"],
        "eval_data_path": fold_cfg["eval_data_path"],
    }


def build_objective(cmdc_root, folds, model_path, save_root, num_gpus):
    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        lora_r = trial.suggest_int("lora_r", 8, 16, step=4)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 32, step=8)

        trial_params = {
            "lr": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        }

        logger.info("\n%s", "=" * 70)
        logger.info("Trial %s: cross-validated evaluation starting", trial.number)
        logger.info("  LR: %.2e", lr)
        logger.info("  Batch Size: %s", batch_size)
        logger.info("  LoRA R: %s", lora_r)
        logger.info("  LoRA Alpha: %s", lora_alpha)
        logger.info("  Folds: %s", ", ".join(folds))
        logger.info("%s\n", "=" * 70)

        fold_results = []
        try:
            for fold_name in folds:
                fold_cfg = get_fold_config(cmdc_root, fold_name)
                logger.info("Trial %s | Running %s", trial.number, fold_name)
                fold_result = run_fold_trial(
                    trial=trial,
                    trial_params=trial_params,
                    fold_name=fold_name,
                    fold_cfg=fold_cfg,
                    model_path=model_path,
                    save_root=save_root,
                    num_gpus=num_gpus,
                )
                fold_results.append(fold_result)
                logger.info(
                    "Trial %s | %s best F1: %.4f",
                    trial.number,
                    fold_name,
                    fold_result["best_f1"],
                )
        except Exception as exc:
            logger.error("Trial %s failed during fold evaluation: %s", trial.number, exc)
            logger.exception(exc)
            trial.set_user_attr("fold_results", fold_results)
            trial.set_user_attr("cv_mean_f1", -1.0)
            trial.set_user_attr("cv_std_f1", None)
            return -1.0

        fold_scores = [result["best_f1"] for result in fold_results]
        mean_f1 = statistics.mean(fold_scores) if fold_scores else -1.0
        std_f1 = statistics.pstdev(fold_scores) if len(fold_scores) > 1 else 0.0

        trial.set_user_attr("fold_results", fold_results)
        trial.set_user_attr("cv_mean_f1", mean_f1)
        trial.set_user_attr("cv_std_f1", std_f1)

        logger.info(
            "Trial %s complete | mean F1: %.4f | std F1: %.4f",
            trial.number,
            mean_f1,
            std_f1,
        )
        return float(mean_f1)

    return objective


def serialize_trial(trial):
    return {
        "trial_number": trial.number,
        "value": None if trial.value is None else float(trial.value),
        "state": trial.state.name,
        "params": dict(trial.params),
        "fold_results": trial.user_attrs.get("fold_results", []),
        "cv_mean_f1": trial.user_attrs.get("cv_mean_f1"),
        "cv_std_f1": trial.user_attrs.get("cv_std_f1"),
    }


def run_optimization(n_trials, study_name, storage_path, cmdc_root, folds, save_root, num_gpus):
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    storage = f"sqlite:///{os.path.abspath(storage_path)}/{study_name}.db"

    logger.info("\n%s", "=" * 70)
    logger.info("Starting cross-validated Optuna optimization")
    logger.info("  Study Name: %s", study_name)
    logger.info("  Number of Trials: %s", n_trials)
    logger.info("  CMDC Root: %s", cmdc_root)
    logger.info("  Folds: %s", ", ".join(folds))
    logger.info("  Save Root: %s", save_root)
    logger.info("  Storage: %s", storage)
    logger.info("%s\n", "=" * 70)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = build_objective(
        cmdc_root=cmdc_root,
        folds=folds,
        model_path=os.environ.get("MODEL_PATH", "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"),
        save_root=save_root,
        num_gpus=num_gpus,
    )

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial

    logger.info("\n%s", "=" * 70)
    logger.info("Cross-validated optimization results")
    logger.info("%s\n", "=" * 70)
    logger.info("Best Trial: #%s", best_trial.number)
    logger.info("Best mean F1: %.4f", best_trial.value)
    logger.info("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info("  %s: %s", key, value)

    results = {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "folds": folds,
        "cmdc_root": cmdc_root,
        "best_trial_number": best_trial.number,
        "best_mean_f1": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "best_fold_results": best_trial.user_attrs.get("fold_results", []),
        "all_trials": [serialize_trial(trial) for trial in study.trials],
    }

    results_file = os.path.join(storage_path, f"{study_name}_results.json")
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", results_file)
    return study, best_trial


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validated Optuna HPO for CMDC text-only training"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", type=str, default="cmdc_textonly_cv_hpo")
    parser.add_argument("--storage-path", type=str, default="optuna_studies")
    parser.add_argument("--cmdc-root", type=str, default=os.environ.get("CMDC_ROOT", "Qwen2-Audio-finetune/data/cmdc"))
    parser.add_argument("--folds", type=str, default=os.environ.get("FOLDS", "fold1 fold2 fold3 fold4 fold5"))
    parser.add_argument("--save-root", type=str, default=os.environ.get("SAVE_ROOT", "output_model/optuna_cmdc_cv_5fold"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "4")))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")

    logger.info("GPU Available: %s", torch.cuda.get_device_name(0))
    logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    folds = parse_folds(args.folds)
    run_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage_path,
        cmdc_root=args.cmdc_root,
        folds=folds,
        save_root=args.save_root,
        num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
