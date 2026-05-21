import os
import time
from dataclasses import dataclass


GROUPED_SINGLE_DATASET_NAMES = {"daic_woz", "eatd"}


@dataclass(frozen=True)
class OptunaPathLayout:
    root_dir_name: str
    output_model_root: str
    storage_root_name: str
    storage_path: str
    log_root_name: str
    log_dir: str
    study_name: str
    study_dir: str
    trial_dir_name: str | None
    trial_output_dir: str | None
    fold_name: str | None = None
    fold_output_dir: str | None = None
    best_model_dir: str | None = None


def resolve_study_timestamp(explicit_timestamp: str | None = None):
    if explicit_timestamp:
        return explicit_timestamp
    env_timestamp = os.environ.get("STUDY_TIMESTAMP", "").strip()
    if env_timestamp:
        return env_timestamp
    return time.strftime("%Y%m%d_%H%M%S")


def format_trial_dir_name(
    prompt_mode: str,
    trial_number: int,
    lr: float,
    batch_size: int,
    lora_r: int,
    lora_alpha: int,
):
    return (
        f"{prompt_mode}_trial_{trial_number:03d}_lr{lr:.0e}"
        f"_bs{batch_size}_r{lora_r}_a{lora_alpha}"
    )


def _single_dataset_root_dir_name(dataset_name: str, eval_level: str | None):
    if dataset_name in GROUPED_SINGLE_DATASET_NAMES:
        if not eval_level:
            raise ValueError(f"eval_level is required for dataset_name={dataset_name!r}")
        return f"optuna_{dataset_name}_hpo_{eval_level}"
    return f"optuna_{dataset_name}_hpo"


def _single_dataset_storage_root_name(dataset_name: str, eval_level: str | None):
    if dataset_name in GROUPED_SINGLE_DATASET_NAMES:
        if not eval_level:
            raise ValueError(f"eval_level is required for dataset_name={dataset_name!r}")
        return f"optuna_{dataset_name}_{eval_level}"
    return f"optuna_{dataset_name}"


def _single_dataset_log_root_name(dataset_name: str, eval_level: str | None):
    return _single_dataset_storage_root_name(dataset_name, eval_level)


def build_single_dataset_layout(
    dataset_name: str,
    prompt_mode: str,
    eval_level: str | None,
    base_output_dir: str = "output_model",
    base_storage_dir: str = "optuna_studies",
    base_log_dir: str = "logs",
    study_timestamp: str | None = None,
    trial_number: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
):
    root_dir_name = _single_dataset_root_dir_name(dataset_name, eval_level)
    output_model_root = os.path.join(base_output_dir, root_dir_name)
    storage_root_name = _single_dataset_storage_root_name(dataset_name, eval_level)
    storage_path = os.path.join(base_storage_dir, storage_root_name)
    log_root_name = _single_dataset_log_root_name(dataset_name, eval_level)
    log_dir = os.path.join(base_log_dir, log_root_name)
    study_name = f"Hpo_Study_{prompt_mode}_{resolve_study_timestamp(study_timestamp)}"
    study_dir = os.path.join(output_model_root, study_name)

    trial_dir_name = None
    trial_output_dir = None
    best_model_dir = None
    if None not in (trial_number, lr, batch_size, lora_r, lora_alpha):
        trial_dir_name = format_trial_dir_name(
            prompt_mode=prompt_mode,
            trial_number=int(trial_number),
            lr=float(lr),
            batch_size=int(batch_size),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
        )
        trial_output_dir = os.path.join(study_dir, trial_dir_name)
        best_model_dir = os.path.join(trial_output_dir, "best_model")

    return OptunaPathLayout(
        root_dir_name=root_dir_name,
        output_model_root=output_model_root,
        storage_root_name=storage_root_name,
        storage_path=storage_path,
        log_root_name=log_root_name,
        log_dir=log_dir,
        study_name=study_name,
        study_dir=study_dir,
        trial_dir_name=trial_dir_name,
        trial_output_dir=trial_output_dir,
        best_model_dir=best_model_dir,
    )


def build_cmdc_cv_layout(
    study_mode: str,
    prompt_mode: str,
    eval_level: str,
    base_output_dir: str = "output_model",
    base_storage_dir: str = "optuna_studies",
    base_log_dir: str = "logs",
    study_timestamp: str | None = None,
    trial_number: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    fold_name: str | None = None,
):
    if not eval_level:
        raise ValueError("eval_level is required for CMDC path construction")

    root_dir_name = f"optuna_cmdc_cv_5fold_{study_mode}_{eval_level}"
    output_model_root = os.path.join(base_output_dir, root_dir_name)
    storage_root_name = root_dir_name
    storage_path = os.path.join(base_storage_dir, storage_root_name)
    log_root_name = root_dir_name
    log_dir = os.path.join(base_log_dir, log_root_name)
    study_name = f"Hpo_Study_{prompt_mode}_{resolve_study_timestamp(study_timestamp)}"
    study_dir = os.path.join(output_model_root, study_name)

    trial_dir_name = None
    trial_output_dir = None
    fold_output_dir = None
    best_model_dir = None
    if None not in (trial_number, lr, batch_size, lora_r, lora_alpha):
        trial_dir_name = format_trial_dir_name(
            prompt_mode=prompt_mode,
            trial_number=int(trial_number),
            lr=float(lr),
            batch_size=int(batch_size),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
        )
        trial_output_dir = os.path.join(study_dir, trial_dir_name)
        if fold_name:
            fold_output_dir = os.path.join(trial_output_dir, fold_name)
            best_model_dir = os.path.join(fold_output_dir, "best_model")

    return OptunaPathLayout(
        root_dir_name=root_dir_name,
        output_model_root=output_model_root,
        storage_root_name=storage_root_name,
        storage_path=storage_path,
        log_root_name=log_root_name,
        log_dir=log_dir,
        study_name=study_name,
        study_dir=study_dir,
        trial_dir_name=trial_dir_name,
        trial_output_dir=trial_output_dir,
        fold_name=fold_name,
        fold_output_dir=fold_output_dir,
        best_model_dir=best_model_dir,
    )

