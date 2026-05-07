"""
Optuna hyperparameter optimization for single-dataset Qwen training.

Supported datasets:
- merged
- daic
- eatd

Supported user-facing mode combinations:
- audio + full
- audio + audiotext
- text + textonly

Each trial runs a DDP training job and returns the best validation F1 score.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass

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

TEXT_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"
AUDIO_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct"

DATASET_MERGED = "merged"
DATASET_DAIC = "daic"
DATASET_EATD = "eatd"
SUPPORTED_DATASETS = {DATASET_MERGED, DATASET_DAIC, DATASET_EATD}

MODEL_FAMILY_TEXT = "text"
MODEL_FAMILY_AUDIO = "audio"

PROMPT_MODE_TEXTONLY = "textonly"
PROMPT_MODE_AUDIOTEXT = "audiotext"
PROMPT_MODE_FULL = "full"

TASK_VARIANT_DEFAULT = "default"
TASK_VARIANT_FILTERED = "filtered"


@dataclass
class DatasetConfig:
    dataset_name: str
    train_data_path: str
    eval_data_path: str
    train_prompt_file: str
    eval_prompt_file: str
    train_task_file: str
    eval_task_file: str
    train_scp_file: str
    eval_scp_file: str


def normalize_model_family(model_family: str):
    normalized = model_family.strip().lower()
    if normalized == "textonly":
        return MODEL_FAMILY_TEXT
    return normalized


def validate_mode_combination(model_family: str, prompt_mode: str):
    allowed_pairs = {
        (MODEL_FAMILY_AUDIO, PROMPT_MODE_FULL),
        (MODEL_FAMILY_AUDIO, PROMPT_MODE_AUDIOTEXT),
        (MODEL_FAMILY_TEXT, PROMPT_MODE_TEXTONLY),
    }
    if (model_family, prompt_mode) not in allowed_pairs:
        raise ValueError(
            "Invalid MODEL_FAMILY / PROMPT_MODE combination: "
            f"{model_family} + {prompt_mode}. "
            "Allowed combinations are: audio+full, audio+audiotext, text+textonly."
        )


def resolve_launch_input_mode(model_family: str):
    if model_family == MODEL_FAMILY_AUDIO:
        return PROMPT_MODE_AUDIOTEXT
    return PROMPT_MODE_TEXTONLY


def resolve_model_path(model_family: str):
    env_model_path = os.environ.get("MODEL_PATH")
    if env_model_path:
        return env_model_path
    if model_family == MODEL_FAMILY_AUDIO:
        return AUDIO_MODEL_PATH_DEFAULT
    return TEXT_MODEL_PATH_DEFAULT


def default_log_dir(dataset_name: str):
    return f"logs/optuna_{dataset_name}"


def default_storage_path(dataset_name: str):
    return f"optuna_studies/optuna_{dataset_name}"


def default_save_root(dataset_name: str, prompt_mode: str):
    return f"output_model/optuna_{dataset_name}_hpo/{prompt_mode}"


def default_study_name(dataset_name: str, model_family: str, prompt_mode: str):
    timestamp = os.environ.get("STUDY_TIMESTAMP", "")
    if not timestamp:
        timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    return f"{dataset_name}_{model_family}_{prompt_mode}_hpo_{timestamp}"


def dataset_root(dataset_name: str):
    return os.environ.get(
        "DATASET_ROOT",
        os.path.join("data", dataset_name),
    )


def get_prompt_filename(dataset_name: str, prompt_mode: str):
    if prompt_mode == PROMPT_MODE_TEXTONLY:
        return f"{dataset_name}_multiprompt_textonly.jsonl"
    if prompt_mode == PROMPT_MODE_AUDIOTEXT:
        return f"{dataset_name}_multiprompt_audiotext.jsonl"
    if prompt_mode == PROMPT_MODE_FULL:
        return f"{dataset_name}_multiprompt.jsonl"
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def get_task_filename(dataset_name: str, task_variant: str):
    if task_variant == TASK_VARIANT_DEFAULT:
        return f"{dataset_name}_multitask.jsonl"
    if task_variant == TASK_VARIANT_FILTERED:
        return f"{dataset_name}_multitask_filtered.jsonl"
    raise ValueError(f"Unsupported task_variant: {task_variant}")


def get_dataset_config(dataset_name: str, prompt_mode: str, task_variant: str):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. Expected one of {sorted(SUPPORTED_DATASETS)}."
        )

    root = dataset_root(dataset_name)
    prompt_file = get_prompt_filename(dataset_name, prompt_mode)
    task_file = get_task_filename(dataset_name, task_variant)
    train_split = "train"
    eval_split = "val"
    if dataset_name == DATASET_EATD:
        eval_split = "test"

    return DatasetConfig(
        dataset_name=dataset_name,
        train_data_path=os.path.join(root, train_split),
        eval_data_path=os.path.join(root, eval_split),
        train_prompt_file=prompt_file,
        eval_prompt_file=prompt_file,
        train_task_file=task_file,
        eval_task_file=task_file,
        train_scp_file=f"{dataset_name}.scp",
        eval_scp_file=f"{dataset_name}.scp",
    )


def apply_dataset_env(dataset_cfg: DatasetConfig):
    os.environ["TRAIN_PROMPT_FILE"] = dataset_cfg.train_prompt_file
    os.environ["EVAL_PROMPT_FILE"] = dataset_cfg.eval_prompt_file
    os.environ["TRAIN_TASK_FILE"] = dataset_cfg.train_task_file
    os.environ["EVAL_TASK_FILE"] = dataset_cfg.eval_task_file
    os.environ["TRAIN_SCP_FILE"] = dataset_cfg.train_scp_file
    os.environ["EVAL_SCP_FILE"] = dataset_cfg.eval_scp_file
    os.environ["TRAIN_DATA_PATH"] = dataset_cfg.train_data_path
    os.environ["EVAL_DATA_PATH"] = dataset_cfg.eval_data_path


def build_objective(dataset_cfg: DatasetConfig, model_path: str, save_root: str, num_gpus: int, model_family: str):
    launch_input_mode = resolve_launch_input_mode(model_family)

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
        lora_r = trial.suggest_int("lora_r", 8, 16, step=4)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 32, step=8)

        logger.info("\n%s", "=" * 70)
        logger.info("Trial %s: Starting Optuna trial", trial.number)
        logger.info("  LR: %.2e", lr)
        logger.info("  Batch Size: %s", batch_size)
        logger.info("  LoRA R: %s", lora_r)
        logger.info("  LoRA Alpha: %s", lora_alpha)
        logger.info("  Dataset: %s", dataset_cfg.dataset_name)
        logger.info("  Model Family: %s", model_family)
        logger.info("  Prompt Mode: %s", os.environ.get("PROMPT_MODE", PROMPT_MODE_TEXTONLY))
        logger.info("  Launch Input Mode: %s", launch_input_mode)
        logger.info("%s\n", "=" * 70)

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
            train_data_path=dataset_cfg.train_data_path,
            eval_data_path=dataset_cfg.eval_data_path,
            save_path=save_root,
            input_mode=launch_input_mode,
            num_gpus=num_gpus,
        )

        logger.info("\nTrial %s completed with Best F1: %.4f\n", trial.number, best_f1)
        return best_f1

    return objective


def run_optimization(
    n_trials=20,
    study_name=None,
    storage_path=None,
    dataset_name=DATASET_MERGED,
    model_family=MODEL_FAMILY_TEXT,
    prompt_mode=PROMPT_MODE_TEXTONLY,
    task_variant=TASK_VARIANT_DEFAULT,
    save_root=None,
):
    validate_mode_combination(model_family, prompt_mode)
    dataset_cfg = get_dataset_config(dataset_name, prompt_mode, task_variant)
    apply_dataset_env(dataset_cfg)

    launch_input_mode = resolve_launch_input_mode(model_family)
    storage_path = storage_path or default_storage_path(dataset_name)
    save_root = save_root or default_save_root(dataset_name, prompt_mode)
    study_name = study_name or default_study_name(dataset_name, model_family, prompt_mode)
    model_path = resolve_model_path(model_family)
    num_gpus = int(os.environ.get("NUM_GPUS", "4"))

    os.environ["MODEL_FAMILY"] = model_family
    os.environ["PROMPT_MODE"] = prompt_mode
    os.environ["TASK_VARIANT"] = task_variant
    os.environ["STORAGE_PATH"] = storage_path
    os.environ["SAVE_PATH"] = save_root
    os.environ["DATASET_NAME"] = dataset_name

    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    storage = f"sqlite:///{os.path.abspath(storage_path)}/{study_name}.db"
    logger.info("\n%s", "=" * 70)
    logger.info("Starting Optuna Hyperparameter Optimization")
    logger.info("  Dataset: %s", dataset_name)
    logger.info("  Study Name: %s", study_name)
    logger.info("  Model Family: %s", model_family)
    logger.info("  Prompt Mode: %s", prompt_mode)
    logger.info("  Task Variant: %s", task_variant)
    logger.info("  Launch Input Mode: %s", launch_input_mode)
    logger.info("  Number of Trials: %s", n_trials)
    logger.info("  Storage: %s", storage)
    logger.info("  Save Root: %s", save_root)
    logger.info("%s\n", "=" * 70)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    objective = build_objective(
        dataset_cfg=dataset_cfg,
        model_path=model_path,
        save_root=save_root,
        num_gpus=num_gpus,
        model_family=model_family,
    )

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    logger.info("\n%s", "=" * 70)
    logger.info("Optimization Results")
    logger.info("%s\n", "=" * 70)

    best_trial = study.best_trial
    logger.info("Best Trial: #%s", best_trial.number)
    logger.info("Best F1 Score: %.4f\n", best_trial.value)
    logger.info("Best Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info("  %s: %s", key, value)

    trials_df = study.trials_dataframe()
    completed_trials = trials_df[trials_df["state"] == "COMPLETE"].sort_values("value", ascending=False)
    logger.info("\n%s", "=" * 70)
    logger.info("All Completed Trials (sorted by F1 score)")
    logger.info("%s\n", "=" * 70)
    logger.info(
        completed_trials[
            ["number", "value", "params_lr", "params_batch_size", "params_lora_r", "params_lora_alpha"]
        ].to_string()
    )

    results_file = os.path.join(storage_path, f"{study_name}_results.json")
    results = {
        "study_name": study_name,
        "dataset_name": dataset_name,
        "model_family": model_family,
        "prompt_mode": prompt_mode,
        "task_variant": task_variant,
        "launch_input_mode": launch_input_mode,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "best_trial_number": best_trial.number,
        "best_f1": float(best_trial.value),
        "best_params": best_trial.params,
        "all_trials": [],
    }

    for _, row in completed_trials.iterrows():
        results["all_trials"].append(
            {
                "trial_number": int(row["number"]),
                "f1": float(row["value"]),
                "params": {
                    "lr": row["params_lr"],
                    "batch_size": int(row["params_batch_size"]),
                    "lora_r": int(row["params_lora_r"]),
                    "lora_alpha": int(row["params_lora_alpha"]),
                },
            }
        )

    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("\nResults saved to: %s", results_file)
    return study, best_trial


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for merged, DAIC, and EATD Qwen training"
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials to run (default: 20)")
    parser.add_argument("--study-name", type=str, default=None, help="Optional study name override")
    parser.add_argument(
        "--storage-path",
        type=str,
        default=os.environ.get("STORAGE_PATH"),
        help="Optional Optuna study directory override",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=os.environ.get("SAVE_PATH"),
        help="Optional output root override",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=sorted(SUPPORTED_DATASETS),
        default=os.environ.get("DATASET_NAME", DATASET_MERGED),
    )
    parser.add_argument(
        "--model-family",
        type=str,
        choices=[MODEL_FAMILY_AUDIO, MODEL_FAMILY_TEXT],
        default=normalize_model_family(os.environ.get("MODEL_FAMILY", MODEL_FAMILY_TEXT)),
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=[PROMPT_MODE_FULL, PROMPT_MODE_AUDIOTEXT, PROMPT_MODE_TEXTONLY],
        default=os.environ.get("PROMPT_MODE", PROMPT_MODE_TEXTONLY),
    )
    parser.add_argument(
        "--task-variant",
        type=str,
        choices=[TASK_VARIANT_DEFAULT, TASK_VARIANT_FILTERED],
        default=os.environ.get("TASK_VARIANT", TASK_VARIANT_DEFAULT),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")

    logger.info("GPU Available: %s", torch.cuda.get_device_name(0))
    logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    run_optimization(
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage_path,
        dataset_name=args.dataset_name,
        model_family=normalize_model_family(args.model_family),
        prompt_mode=args.prompt_mode,
        task_variant=args.task_variant,
        save_root=args.save_root,
    )
