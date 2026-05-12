"""
Optuna hyperparameter optimization for single-dataset Qwen training.

Supported datasets:
- merged
- daic_woz
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

from optuna_hpo.pruning import (
    DEFAULT_ENABLE_PRUNING,
    DEFAULT_PRUNER_INTERVAL_STEPS,
    DEFAULT_PRUNER_STARTUP_TRIALS,
    DEFAULT_PRUNER_WARMUP_STEPS,
    build_pruner,
    env_flag,
)
from optuna_hpo.train_launcher import launch_ddp_training
from utils.grouped_eval import (
    GROUPED_DATASET_NAMES,
    SUPPORTED_GROUPED_EVAL_LEVELS,
    SUPPORTED_GROUPED_EVAL_MODES,
    grouped_eval_enabled,
    grouped_eval_env_prefix,
    normalize_grouped_eval_level,
    normalize_grouped_eval_mode,
    validate_grouped_person_threshold,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)

TEXT_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"
AUDIO_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct"

DATASET_MERGED = "merged"
DATASET_DAIC_WOZ = "daic_woz"
DATASET_EATD = "eatd"
SUPPORTED_DATASETS = {DATASET_MERGED, DATASET_DAIC_WOZ, DATASET_EATD}

MODEL_FAMILY_TEXT = "text"
MODEL_FAMILY_AUDIO = "audio"

PROMPT_MODE_TEXTONLY = "textonly"
PROMPT_MODE_AUDIOTEXT = "audiotext"
PROMPT_MODE_FULL = "full"

TASK_VARIANT_DEFAULT = "default"
TASK_VARIANT_FILTERED = "filtered"

DEFAULT_GROUPED_EVAL_LEVEL = "person"
DEFAULT_GROUPED_EVAL_MODE = "majority_vote"


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


def _normalized_grouped_eval_values(level: str, mode: str, threshold: float):
    return (
        normalize_grouped_eval_level(level),
        normalize_grouped_eval_mode(mode),
        validate_grouped_person_threshold(threshold),
    )


def resolve_grouped_eval_settings(args_or_mapping):
    settings = {}
    for dataset_name in sorted(GROUPED_DATASET_NAMES):
        level = getattr(
            args_or_mapping,
            f"{dataset_name}_eval_level",
            None,
        )
        if level is None and isinstance(args_or_mapping, dict):
            level = args_or_mapping.get(f"{dataset_name}_eval_level")
        mode = getattr(
            args_or_mapping,
            f"{dataset_name}_eval_mode",
            None,
        )
        if mode is None and isinstance(args_or_mapping, dict):
            mode = args_or_mapping.get(f"{dataset_name}_eval_mode")
        threshold = getattr(
            args_or_mapping,
            f"{dataset_name}_person_threshold",
            None,
        )
        if threshold is None and isinstance(args_or_mapping, dict):
            threshold = args_or_mapping.get(f"{dataset_name}_person_threshold")
        if dataset_name == DATASET_DAIC_WOZ:
            legacy_level = getattr(args_or_mapping, "daic_eval_level", None)
            legacy_mode = getattr(args_or_mapping, "daic_eval_mode", None)
            legacy_threshold = getattr(args_or_mapping, "daic_person_threshold", None)
            if isinstance(args_or_mapping, dict):
                legacy_level = args_or_mapping.get("daic_eval_level", legacy_level)
                legacy_mode = args_or_mapping.get("daic_eval_mode", legacy_mode)
                legacy_threshold = args_or_mapping.get("daic_person_threshold", legacy_threshold)
            level = level or legacy_level or os.environ.get("DAIC_EVAL_LEVEL")
            mode = mode or legacy_mode or os.environ.get("DAIC_EVAL_MODE")
            threshold = threshold if threshold is not None else legacy_threshold
            if threshold is None and os.environ.get("DAIC_PERSON_THRESHOLD") is not None:
                threshold = os.environ.get("DAIC_PERSON_THRESHOLD")

        prefix = grouped_eval_env_prefix(dataset_name)
        level = level or os.environ.get(f"{prefix}_EVAL_LEVEL", DEFAULT_GROUPED_EVAL_LEVEL)
        mode = mode or os.environ.get(f"{prefix}_EVAL_MODE", DEFAULT_GROUPED_EVAL_MODE)
        threshold = (
            threshold
            if threshold is not None
            else os.environ.get(f"{prefix}_PERSON_THRESHOLD", "0.5")
        )
        normalized_level, normalized_mode, normalized_threshold = _normalized_grouped_eval_values(
            level,
            mode,
            float(threshold),
        )
        settings[dataset_name] = {
            "level": normalized_level,
            "mode": normalized_mode,
            "threshold": normalized_threshold,
        }
    return settings


def grouped_level_suffix(dataset_name: str, grouped_settings: dict):
    if not grouped_eval_enabled(dataset_name):
        return ""
    return f"_{grouped_settings[dataset_name]['level']}"


def resolved_log_dir(dataset_name: str, grouped_settings: dict):
    return f"{default_log_dir(dataset_name)}{grouped_level_suffix(dataset_name, grouped_settings)}"


def resolved_storage_path(dataset_name: str, grouped_settings: dict):
    return f"{default_storage_path(dataset_name)}{grouped_level_suffix(dataset_name, grouped_settings)}"


def resolved_save_root(dataset_name: str, prompt_mode: str, grouped_settings: dict):
    suffix = grouped_level_suffix(dataset_name, grouped_settings)
    return f"output_model/optuna_{dataset_name}_hpo{suffix}/{prompt_mode}"


def resolved_study_name(dataset_name: str, model_family: str, prompt_mode: str, grouped_settings: dict):
    timestamp = os.environ.get("STUDY_TIMESTAMP", "")
    if not timestamp:
        timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
    suffix = grouped_level_suffix(dataset_name, grouped_settings)
    return f"{dataset_name}{suffix}_{model_family}_{prompt_mode}_hpo_{timestamp}"


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


def apply_grouped_eval_env(grouped_settings: dict):
    for dataset_name, values in grouped_settings.items():
        prefix = grouped_eval_env_prefix(dataset_name)
        os.environ[f"{prefix}_EVAL_LEVEL"] = values["level"]
        os.environ[f"{prefix}_EVAL_MODE"] = values["mode"]
        os.environ[f"{prefix}_PERSON_THRESHOLD"] = str(values["threshold"])

    daic_values = grouped_settings[DATASET_DAIC_WOZ]
    os.environ["DAIC_EVAL_LEVEL"] = daic_values["level"]
    os.environ["DAIC_EVAL_MODE"] = daic_values["mode"]
    os.environ["DAIC_PERSON_THRESHOLD"] = str(daic_values["threshold"])


def build_objective(
    dataset_cfg: DatasetConfig,
    model_path: str,
    save_root: str,
    num_gpus: int,
    model_family: str,
    enable_pruning: bool,
    grouped_settings: dict,
):
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
        for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
            logger.info(
                "  %s Eval: level=%s mode=%s threshold=%.4f",
                grouped_dataset_name,
                grouped_settings[grouped_dataset_name]["level"],
                grouped_settings[grouped_dataset_name]["mode"],
                grouped_settings[grouped_dataset_name]["threshold"],
            )
        logger.info("%s\n", "=" * 70)

        trial_params = {
            "lr": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        }

        best_f1 = launch_ddp_training(
            trial=trial,
            trial_params=trial_params,
            trial_number=trial.number,
            model_path=model_path,
            train_data_path=dataset_cfg.train_data_path,
            eval_data_path=dataset_cfg.eval_data_path,
            save_path=save_root,
            dataset_name=dataset_cfg.dataset_name,
            input_mode=launch_input_mode,
            num_gpus=num_gpus,
            enable_pruning=enable_pruning,
            prune_mode="eval" if enable_pruning else "disabled",
            daic_eval_level=grouped_settings[DATASET_DAIC_WOZ]["level"],
            daic_eval_mode=grouped_settings[DATASET_DAIC_WOZ]["mode"],
            daic_person_threshold=grouped_settings[DATASET_DAIC_WOZ]["threshold"],
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
    enable_pruning=DEFAULT_ENABLE_PRUNING,
    pruner_startup_trials=DEFAULT_PRUNER_STARTUP_TRIALS,
    pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
    pruner_interval_steps=DEFAULT_PRUNER_INTERVAL_STEPS,
    grouped_settings=None,
):
    validate_mode_combination(model_family, prompt_mode)
    grouped_settings = grouped_settings or resolve_grouped_eval_settings({})
    dataset_cfg = get_dataset_config(dataset_name, prompt_mode, task_variant)
    apply_dataset_env(dataset_cfg)

    launch_input_mode = resolve_launch_input_mode(model_family)
    storage_path = storage_path or resolved_storage_path(dataset_name, grouped_settings)
    save_root = save_root or resolved_save_root(dataset_name, prompt_mode, grouped_settings)
    study_name = study_name or resolved_study_name(dataset_name, model_family, prompt_mode, grouped_settings)
    model_path = resolve_model_path(model_family)
    num_gpus = int(os.environ.get("NUM_GPUS", "4"))

    os.environ["MODEL_FAMILY"] = model_family
    os.environ["PROMPT_MODE"] = prompt_mode
    os.environ["TASK_VARIANT"] = task_variant
    os.environ["STORAGE_PATH"] = storage_path
    os.environ["SAVE_PATH"] = save_root
    os.environ["DATASET_NAME"] = dataset_name
    apply_grouped_eval_env(grouped_settings)

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
    logger.info("  Pruning Enabled: %s", enable_pruning)
    logger.info("  Pruner Startup Trials: %s", pruner_startup_trials)
    logger.info("  Pruner Warmup Steps: %s", pruner_warmup_steps)
    logger.info("  Pruner Interval Steps: %s", pruner_interval_steps)
    for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
        logger.info(
            "  %s Eval: level=%s mode=%s threshold=%.4f",
            grouped_dataset_name,
            grouped_settings[grouped_dataset_name]["level"],
            grouped_settings[grouped_dataset_name]["mode"],
            grouped_settings[grouped_dataset_name]["threshold"],
        )
    logger.info("  Number of Trials: %s", n_trials)
    logger.info("  Storage: %s", storage)
    logger.info("  Save Root: %s", save_root)
    logger.info("%s\n", "=" * 70)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=build_pruner(
            enable_pruning=enable_pruning,
            startup_trials=pruner_startup_trials,
            warmup_steps=pruner_warmup_steps,
            interval_steps=pruner_interval_steps,
        ),
    )

    objective = build_objective(
        dataset_cfg=dataset_cfg,
        model_path=model_path,
        save_root=save_root,
        num_gpus=num_gpus,
        model_family=model_family,
        enable_pruning=enable_pruning,
        grouped_settings=grouped_settings,
    )

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    logger.info("\n%s", "=" * 70)
    logger.info("Optimization Results")
    logger.info("%s\n", "=" * 70)

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial = study.best_trial if completed_trials else None
    if best_trial is not None:
        logger.info("Best Trial: #%s", best_trial.number)
        logger.info("Best F1 Score: %.4f\n", best_trial.value)
        logger.info("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info("  %s: %s", key, value)
    else:
        logger.info("No completed trials are available to summarize.")

    trials_df = study.trials_dataframe()
    completed_trials_df = trials_df[trials_df["state"] == "COMPLETE"].sort_values("value", ascending=False)
    logger.info("\n%s", "=" * 70)
    logger.info("All Completed Trials (sorted by F1 score)")
    logger.info("%s\n", "=" * 70)
    if completed_trials_df.empty:
        logger.info("No completed trials")
    else:
        logger.info(
            completed_trials_df[
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
        "pruning_enabled": enable_pruning,
        "pruner_startup_trials": pruner_startup_trials,
        "pruner_warmup_steps": pruner_warmup_steps,
        "pruner_interval_steps": pruner_interval_steps,
        "grouped_eval": grouped_settings,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials_df),
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_f1": float(best_trial.value) if best_trial is not None else None,
        "best_params": dict(best_trial.params) if best_trial is not None else None,
        "best_eval_summary": best_trial.user_attrs.get("best_eval_summary") if best_trial is not None else None,
        "all_trials": [],
    }

    for trial in study.trials:
        results["all_trials"].append(
            {
                "trial_number": int(trial.number),
                "f1": None if trial.value is None else float(trial.value),
                "state": trial.state.name,
                "params": dict(trial.params),
                "best_eval_summary": trial.user_attrs.get("best_eval_summary"),
            }
        )

    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    logger.info("\nResults saved to: %s", results_file)
    return study, best_trial


if __name__ == "__main__":
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
    parser.add_argument(
        "--enable-pruning",
        dest="enable_pruning",
        action="store_true",
        default=env_flag("ENABLE_PRUNING", DEFAULT_ENABLE_PRUNING),
        help="Enable Optuna median pruning (default: enabled).",
    )
    parser.add_argument(
        "--disable-pruning",
        dest="enable_pruning",
        action="store_false",
        help="Disable Optuna pruning and run every trial to completion.",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=int(os.environ.get("PRUNER_STARTUP_TRIALS", DEFAULT_PRUNER_STARTUP_TRIALS)),
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=int(os.environ.get("PRUNER_WARMUP_STEPS", DEFAULT_PRUNER_WARMUP_STEPS)),
    )
    parser.add_argument(
        "--pruner-interval-steps",
        type=int,
        default=int(os.environ.get("PRUNER_INTERVAL_STEPS", DEFAULT_PRUNER_INTERVAL_STEPS)),
    )
    parser.add_argument(
        "--daic-eval-level",
        dest="daic_woz_eval_level",
        type=str,
        choices=sorted(SUPPORTED_GROUPED_EVAL_LEVELS),
        default=os.environ.get("DAIC_WOZ_EVAL_LEVEL", os.environ.get("DAIC_EVAL_LEVEL", DEFAULT_GROUPED_EVAL_LEVEL)),
    )
    parser.add_argument(
        "--daic-eval-mode",
        dest="daic_woz_eval_mode",
        type=str,
        choices=sorted(SUPPORTED_GROUPED_EVAL_MODES),
        default=os.environ.get("DAIC_WOZ_EVAL_MODE", os.environ.get("DAIC_EVAL_MODE", DEFAULT_GROUPED_EVAL_MODE)),
    )
    parser.add_argument(
        "--daic-person-threshold",
        dest="daic_woz_person_threshold",
        type=float,
        default=float(os.environ.get("DAIC_WOZ_PERSON_THRESHOLD", os.environ.get("DAIC_PERSON_THRESHOLD", "0.5"))),
    )
    parser.add_argument(
        "--eatd-eval-level",
        type=str,
        choices=sorted(SUPPORTED_GROUPED_EVAL_LEVELS),
        default=os.environ.get("EATD_EVAL_LEVEL", DEFAULT_GROUPED_EVAL_LEVEL),
    )
    parser.add_argument(
        "--eatd-eval-mode",
        type=str,
        choices=sorted(SUPPORTED_GROUPED_EVAL_MODES),
        default=os.environ.get("EATD_EVAL_MODE", DEFAULT_GROUPED_EVAL_MODE),
    )
    parser.add_argument(
        "--eatd-person-threshold",
        type=float,
        default=float(os.environ.get("EATD_PERSON_THRESHOLD", "0.5")),
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
        enable_pruning=args.enable_pruning,
        pruner_startup_trials=args.pruner_startup_trials,
        pruner_warmup_steps=args.pruner_warmup_steps,
        pruner_interval_steps=args.pruner_interval_steps,
        grouped_settings=resolve_grouped_eval_settings(args),
    )
