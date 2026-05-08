"""
Optuna hyperparameter optimization for CMDC cross-validated training.

Supports two study modes:
- cv_mean: one Optuna trial samples a single hyperparameter set and evaluates
  it across all requested folds, using the mean fold F1 as the objective
- per_fold: each requested fold gets its own Optuna study, and every trial
  evaluates only that single fold
"""

import argparse
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter

import optuna
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optuna_hpo.train_launcher import launch_ddp_training
from optuna_hpo.pruning import (
    DEFAULT_ENABLE_PRUNING,
    DEFAULT_PRUNER_INTERVAL_STEPS,
    DEFAULT_PRUNER_STARTUP_TRIALS,
    DEFAULT_PRUNER_WARMUP_STEPS,
    build_pruner,
    env_flag,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


TEXT_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-7B-Instruct"
AUDIO_MODEL_PATH_DEFAULT = "/gpfs/projects/etur92/ozu647717/models/Qwen2-Audio-7B-Instruct"

MODEL_FAMILY_TEXT = "text"
MODEL_FAMILY_AUDIO = "audio"

PROMPT_MODE_TEXTONLY = "textonly"
PROMPT_MODE_AUDIOTEXT = "audiotext"
PROMPT_MODE_FULL = "full"


def parse_folds(folds_text: str):
    return [token.strip() for token in folds_text.split() if token.strip()]


def default_save_root_for_study_mode(study_mode: str):
    return f"output_model/optuna_cmdc_cv_5fold_{study_mode}"


def default_storage_path_for_study_mode(study_mode: str):
    return f"optuna_studies/optuna_cmdc_cv_5fold_{study_mode}"


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
    pair = (model_family, prompt_mode)
    if pair not in allowed_pairs:
        raise ValueError(
            "Invalid MODEL_FAMILY / PROMPT_MODE combination: "
            f"{model_family} + {prompt_mode}. "
            "Allowed combinations are: audio+full, audio+audiotext, text+textonly."
        )


def resolve_model_path(model_family: str):
    env_model_path = os.environ.get("MODEL_PATH")
    if env_model_path:
        return env_model_path
    if model_family == MODEL_FAMILY_AUDIO:
        return AUDIO_MODEL_PATH_DEFAULT
    return TEXT_MODEL_PATH_DEFAULT


def resolve_launch_input_mode(model_family: str):
    if model_family == MODEL_FAMILY_AUDIO:
        return PROMPT_MODE_AUDIOTEXT
    return PROMPT_MODE_TEXTONLY


def get_prompt_file(fold_name: str, prompt_mode: str):
    if prompt_mode == PROMPT_MODE_TEXTONLY:
        return f"{fold_name}_multiprompt_textonly.jsonl"
    if prompt_mode == PROMPT_MODE_AUDIOTEXT:
        return f"{fold_name}_multiprompt_audiotext.jsonl"
    if prompt_mode == PROMPT_MODE_FULL:
        return f"{fold_name}_multiprompt.jsonl"
    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def get_fold_config(cmdc_root: str, fold_name: str, prompt_mode: str):
    prompt_file = get_prompt_file(fold_name, prompt_mode)
    return {
        "train_data_path": os.path.join(cmdc_root, fold_name, "train"),
        "eval_data_path": os.path.join(cmdc_root, fold_name, "test"),
        "train_prompt_file": prompt_file,
        "eval_prompt_file": prompt_file,
        "train_task_file": f"{fold_name}_multitask.jsonl",
        "eval_task_file": f"{fold_name}_multitask.jsonl",
        "train_scp_file": f"{fold_name}.scp",
        "eval_scp_file": f"{fold_name}.scp",
    }


def _load_task_keys(data_path: str, task_file: str):
    task_path = os.path.join(data_path, task_file)
    keys = []
    with open(task_path, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            keys.append(item["key"])
    return keys


def _load_scp_keys(data_path: str, scp_file: str):
    scp_path = os.path.join(data_path, scp_file)
    keys = set()
    with open(scp_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            utt_id, _ = line.strip().split(" ", 1)
            keys.add(utt_id)
    return keys


def validate_fold_audio_config(fold_name: str, fold_cfg: dict):
    issues = []
    for split_name, data_path, task_file, scp_file in [
        ("train", fold_cfg["train_data_path"], fold_cfg["train_task_file"], fold_cfg["train_scp_file"]),
        ("eval", fold_cfg["eval_data_path"], fold_cfg["eval_task_file"], fold_cfg["eval_scp_file"]),
    ]:
        task_keys = _load_task_keys(data_path, task_file)
        scp_keys = _load_scp_keys(data_path, scp_file)
        missing = sorted({key for key in task_keys if key not in scp_keys})
        duplicate_counts = {key: count for key, count in Counter(task_keys).items() if count > 1}
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            issues.append(
                f"{fold_name} {split_name}: {len(missing)} task key(s) missing from {scp_file}: {preview}{suffix}"
            )
        if duplicate_counts:
            dup_preview = ", ".join(f"{key}x{count}" for key, count in list(sorted(duplicate_counts.items()))[:10])
            suffix = " ..." if len(duplicate_counts) > 10 else ""
            issues.append(
                f"{fold_name} {split_name}: duplicate task keys in {task_file}: {dup_preview}{suffix}"
            )
    return issues


def validate_audio_fold_configs(cmdc_root: str, folds, prompt_mode: str, model_family: str):
    if model_family != MODEL_FAMILY_AUDIO:
        return

    all_issues = []
    for fold_name in folds:
        fold_cfg = get_fold_config(cmdc_root, fold_name, prompt_mode)
        all_issues.extend(validate_fold_audio_config(fold_name, fold_cfg))

    if all_issues:
        message = "Audio fold validation failed:\n- " + "\n- ".join(all_issues)
        raise RuntimeError(message)


def sample_trial_params(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    lora_r = trial.suggest_int("lora_r", 8, 16, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", 8, 32, step=8)

    return {
        "lr": lr,
        "batch_size": batch_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }


def log_trial_header(trial_number, trial_params, mode_label, folds):
    logger.info("\n%s", "=" * 70)
    logger.info("Trial %s: %s starting", trial_number, mode_label)
    logger.info("  LR: %.2e", trial_params["lr"])
    logger.info("  Batch Size: %s", trial_params["batch_size"])
    logger.info("  LoRA R: %s", trial_params["lora_r"])
    logger.info("  LoRA Alpha: %s", trial_params["lora_alpha"])
    logger.info("  Folds: %s", ", ".join(folds))
    logger.info("%s\n", "=" * 70)


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


def collect_study_summary(study):
    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial = study.best_trial if completed_trials else None
    return completed_trials, best_trial


def write_results_file(results_file, results):
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def run_fold_trial(
    trial,
    trial_params,
    fold_name,
    fold_cfg,
    model_path,
    save_root,
    num_gpus,
    launch_input_mode,
    model_family,
    prompt_mode,
    enable_pruning,
):
    previous_env = {
        "TRAIN_PROMPT_FILE": os.environ.get("TRAIN_PROMPT_FILE"),
        "EVAL_PROMPT_FILE": os.environ.get("EVAL_PROMPT_FILE"),
        "TRAIN_TASK_FILE": os.environ.get("TRAIN_TASK_FILE"),
        "EVAL_TASK_FILE": os.environ.get("EVAL_TASK_FILE"),
        "TRAIN_SCP_FILE": os.environ.get("TRAIN_SCP_FILE"),
        "EVAL_SCP_FILE": os.environ.get("EVAL_SCP_FILE"),
        "MODEL_FAMILY": os.environ.get("MODEL_FAMILY"),
        "PROMPT_MODE": os.environ.get("PROMPT_MODE"),
    }

    os.environ["TRAIN_PROMPT_FILE"] = fold_cfg["train_prompt_file"]
    os.environ["EVAL_PROMPT_FILE"] = fold_cfg["eval_prompt_file"]
    os.environ["TRAIN_TASK_FILE"] = fold_cfg["train_task_file"]
    os.environ["EVAL_TASK_FILE"] = fold_cfg["eval_task_file"]
    os.environ["TRAIN_SCP_FILE"] = fold_cfg["train_scp_file"]
    os.environ["EVAL_SCP_FILE"] = fold_cfg["eval_scp_file"]
    os.environ["MODEL_FAMILY"] = model_family
    os.environ["PROMPT_MODE"] = prompt_mode

    fold_save_path = os.path.join(save_root, f"trial_{trial.number:03d}", fold_name)

    try:
        best_f1 = launch_ddp_training(
            trial=trial,
            trial_params=trial_params,
            trial_number=trial.number,
            model_path=model_path,
            train_data_path=fold_cfg["train_data_path"],
            eval_data_path=fold_cfg["eval_data_path"],
            save_path=fold_save_path,
            dataset_name="cmdc",
            input_mode=launch_input_mode,
            num_gpus=num_gpus,
            enable_pruning=enable_pruning,
            prune_mode="eval" if enable_pruning else "disabled",
        )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if not math.isfinite(best_f1):
        best_f1 = -1.0

    return {
        "fold": fold_name,
        "best_f1": float(best_f1),
        "save_path": fold_save_path,
        "train_data_path": fold_cfg["train_data_path"],
        "eval_data_path": fold_cfg["eval_data_path"],
        "model_family": model_family,
        "prompt_mode": prompt_mode,
        "launch_input_mode": launch_input_mode,
    }


def build_objective(
    cmdc_root,
    folds,
    model_path,
    save_root,
    num_gpus,
    model_family,
    prompt_mode,
    enable_pruning,
):
    launch_input_mode = resolve_launch_input_mode(model_family)

    def objective(trial: optuna.Trial):
        trial_params = sample_trial_params(trial)
        log_trial_header(
            trial_number=trial.number,
            trial_params=trial_params,
            mode_label="cross-validated evaluation",
            folds=folds,
        )

        fold_results = []
        try:
            for fold_name in folds:
                fold_cfg = get_fold_config(cmdc_root, fold_name, prompt_mode)
                logger.info("Trial %s | Running %s", trial.number, fold_name)
                fold_result = run_fold_trial(
                    trial=trial,
                    trial_params=trial_params,
                    fold_name=fold_name,
                    fold_cfg=fold_cfg,
                    model_path=model_path,
                    save_root=save_root,
                    num_gpus=num_gpus,
                    launch_input_mode=launch_input_mode,
                    model_family=model_family,
                    prompt_mode=prompt_mode,
                    enable_pruning=False,
                )
                fold_results.append(fold_result)
                valid_fold_scores = [
                    result["best_f1"]
                    for result in fold_results
                    if math.isfinite(result["best_f1"]) and result["best_f1"] != -1.0
                ]
                partial_mean = statistics.mean(valid_fold_scores) if valid_fold_scores else -1.0
                trial.report(float(partial_mean), step=len(fold_results))
                if enable_pruning and trial.should_prune():
                    trial.set_user_attr("fold_results", fold_results)
                    trial.set_user_attr("cv_mean_f1", partial_mean)
                    trial.set_user_attr(
                        "cv_std_f1",
                        statistics.pstdev(valid_fold_scores) if len(valid_fold_scores) > 1 else 0.0,
                    )
                    raise optuna.TrialPruned(
                        f"Trial {trial.number} pruned after {len(fold_results)} fold(s) with partial mean F1={partial_mean:.4f}"
                    )
                logger.info(
                    "Trial %s | %s best F1: %.4f",
                    trial.number,
                    fold_name,
                    fold_result["best_f1"],
                )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.error("Trial %s failed during fold evaluation: %s", trial.number, exc)
            logger.exception(exc)
            fold_scores = [result["best_f1"] for result in fold_results]
            valid_fold_scores = [
                score for score in fold_scores
                if math.isfinite(score) and score != -1.0
            ]
            mean_f1 = statistics.mean(valid_fold_scores) if valid_fold_scores else -1.0
            std_f1 = statistics.pstdev(valid_fold_scores) if len(valid_fold_scores) > 1 else 0.0

            trial.set_user_attr("fold_results", fold_results)
            trial.set_user_attr("cv_mean_f1", mean_f1)
            trial.set_user_attr("cv_std_f1", std_f1)

            if valid_fold_scores:
                logger.warning(
                    "Trial %s is using partial CV results after failure | mean F1: %.4f | std F1: %.4f",
                    trial.number,
                    mean_f1,
                    std_f1,
                )
            return float(mean_f1)

        fold_scores = [result["best_f1"] for result in fold_results]
        valid_fold_scores = [
            score for score in fold_scores
            if math.isfinite(score) and score != -1.0
        ]
        ignored_count = len(fold_scores) - len(valid_fold_scores)

        if ignored_count > 0:
            logger.warning(
                "Trial %s is ignoring %s failed fold(s) in the CV mean: %s",
                trial.number,
                ignored_count,
                fold_scores,
            )

        mean_f1 = statistics.mean(valid_fold_scores) if valid_fold_scores else -1.0
        std_f1 = statistics.pstdev(valid_fold_scores) if len(valid_fold_scores) > 1 else 0.0

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


def build_single_fold_objective(
    cmdc_root,
    fold_name,
    model_path,
    save_root,
    num_gpus,
    model_family,
    prompt_mode,
    enable_pruning,
):
    fold_cfg = get_fold_config(cmdc_root, fold_name, prompt_mode)
    launch_input_mode = resolve_launch_input_mode(model_family)

    def objective(trial: optuna.Trial):
        trial_params = sample_trial_params(trial)
        log_trial_header(
            trial_number=trial.number,
            trial_params=trial_params,
            mode_label=f"single-fold evaluation for {fold_name}",
            folds=[fold_name],
        )

        try:
            logger.info("Trial %s | Running %s", trial.number, fold_name)
            fold_result = run_fold_trial(
                trial=trial,
                trial_params=trial_params,
                fold_name=fold_name,
                fold_cfg=fold_cfg,
                model_path=model_path,
                save_root=save_root,
                num_gpus=num_gpus,
                launch_input_mode=launch_input_mode,
                model_family=model_family,
                prompt_mode=prompt_mode,
                enable_pruning=enable_pruning,
            )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.error("Trial %s failed during %s evaluation: %s", trial.number, fold_name, exc)
            logger.exception(exc)
            fold_result = {
                "fold": fold_name,
                "best_f1": -1.0,
                "save_path": os.path.join(save_root, f"trial_{trial.number:03d}", fold_name),
                "train_data_path": fold_cfg["train_data_path"],
                "eval_data_path": fold_cfg["eval_data_path"],
            }

        best_f1 = fold_result["best_f1"]
        trial.set_user_attr("fold_results", [fold_result])
        trial.set_user_attr("cv_mean_f1", best_f1)
        trial.set_user_attr("cv_std_f1", 0.0)

        logger.info(
            "Trial %s complete | %s best F1: %.4f",
            trial.number,
            fold_name,
            best_f1,
        )
        return float(best_f1)

    return objective


def run_study(
    study_name,
    storage_path,
    summary_label,
    load_if_exists=True,
    enable_pruning=True,
    pruner_startup_trials=DEFAULT_PRUNER_STARTUP_TRIALS,
    pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
    pruner_interval_steps=DEFAULT_PRUNER_INTERVAL_STEPS,
):
    storage = f"sqlite:///{os.path.abspath(storage_path)}/{study_name}.db"
    logger.info("\n%s", "=" * 70)
    logger.info("%s", summary_label)
    logger.info("  Study Name: %s", study_name)
    logger.info("  Storage: %s", storage)
    logger.info("  Resume Existing Study: %s", load_if_exists)
    logger.info("%s\n", "=" * 70)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=load_if_exists,
        pruner=build_pruner(
            enable_pruning=enable_pruning,
            startup_trials=pruner_startup_trials,
            warmup_steps=pruner_warmup_steps,
            interval_steps=pruner_interval_steps,
        ),
    )
    return study


def resolve_requested_trials(study, n_trials, target_total_trials=None):
    if target_total_trials is None:
        return n_trials

    state_counts = Counter(trial.state.name for trial in study.trials)
    completed_trial_count = state_counts.get("COMPLETE", 0)
    recorded_trial_count = len(study.trials)
    remaining_trials = max(0, target_total_trials - completed_trial_count)

    logger.info(
        "Study already has %s completed trial(s) out of %s recorded trial(s); target completed total is %s; scheduling %s additional trial(s).",
        completed_trial_count,
        recorded_trial_count,
        target_total_trials,
        remaining_trials,
    )
    if state_counts:
        logger.info("Existing trial states: %s", dict(sorted(state_counts.items())))

    return remaining_trials


# cv_mean study mode: each trial evaluates all folds and uses the mean fold F1 as the objective
def run_optimization(
    n_trials,
    study_name,
    storage_path,
    cmdc_root,
    folds,
    save_root,
    num_gpus,
    model_family,
    prompt_mode,
    resume=True,
    target_total_trials=None,
    enable_pruning=DEFAULT_ENABLE_PRUNING,
    pruner_startup_trials=DEFAULT_PRUNER_STARTUP_TRIALS,
    pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
    pruner_interval_steps=DEFAULT_PRUNER_INTERVAL_STEPS,
):
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    validate_mode_combination(model_family, prompt_mode)
    validate_audio_fold_configs(cmdc_root, folds, prompt_mode, model_family)

    logger.info("\n%s", "=" * 70)
    logger.info("Starting Optuna optimization")
    logger.info("  Study Mode: cv_mean")
    logger.info("  Study Name: %s", study_name)
    logger.info("  Model Family: %s", model_family)
    logger.info("  Prompt Mode: %s", prompt_mode)
    logger.info("  Launch Input Mode: %s", resolve_launch_input_mode(model_family))
    logger.info("  Pruning Enabled: %s", enable_pruning)
    logger.info("  Pruner Startup Trials: %s", pruner_startup_trials)
    logger.info("  Pruner Warmup Steps: %s", pruner_warmup_steps)
    logger.info("  Pruner Interval Steps: %s", pruner_interval_steps)
    logger.info("  Number of Trials: %s", n_trials)
    logger.info("  CMDC Root: %s", cmdc_root)
    logger.info("  Folds: %s", ", ".join(folds))
    logger.info("  Save Root: %s", save_root)
    logger.info("  Resume Existing Study: %s", resume)
    logger.info(
        "  Target Total Completed Trials: %s",
        target_total_trials if target_total_trials is not None else "disabled",
    )
    logger.info("%s\n", "=" * 70)

    study = run_study(
        study_name=study_name,
        storage_path=storage_path,
        summary_label="Creating cv_mean study",
        load_if_exists=resume,
        enable_pruning=enable_pruning,
        pruner_startup_trials=pruner_startup_trials,
        pruner_warmup_steps=pruner_warmup_steps,
        pruner_interval_steps=pruner_interval_steps,
    )

    objective = build_objective(
        cmdc_root=cmdc_root,
        folds=folds,
        model_path=resolve_model_path(model_family),
        save_root=save_root,
        num_gpus=num_gpus,
        model_family=model_family,
        prompt_mode=prompt_mode,
        enable_pruning=enable_pruning,
    )

    results_file = os.path.join(storage_path, f"{study_name}_results.json")
    try:
        requested_trials = resolve_requested_trials(
            study,
            n_trials=n_trials,
            target_total_trials=target_total_trials,
        )
        if requested_trials == 0:
            logger.info("No additional trials requested; study already meets the target.")
        else:
            study.optimize(objective, n_trials=requested_trials, n_jobs=1)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    finally:
        completed_trials, best_trial = collect_study_summary(study)

        if best_trial is not None:
            logger.info("\n%s", "=" * 70)
            logger.info("Cross-validated optimization results")
            logger.info("%s\n", "=" * 70)
            logger.info("Best Trial: #%s", best_trial.number)
            logger.info("Best mean F1: %.4f", best_trial.value)
            logger.info("Best Hyperparameters:")
            for key, value in best_trial.params.items():
                logger.info("  %s: %s", key, value)
        else:
            logger.info("No completed trials are available to summarize.")

        results = {
            "study_name": study_name,
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "folds": folds,
            "cmdc_root": cmdc_root,
            "model_family": model_family,
            "prompt_mode": prompt_mode,
            "launch_input_mode": resolve_launch_input_mode(model_family),
            "pruning_enabled": enable_pruning,
            "pruner_startup_trials": pruner_startup_trials,
            "pruner_warmup_steps": pruner_warmup_steps,
            "pruner_interval_steps": pruner_interval_steps,
            "best_trial_number": best_trial.number if best_trial is not None else None,
            "best_mean_f1": float(best_trial.value) if best_trial is not None else None,
            "best_params": dict(best_trial.params) if best_trial is not None else None,
            "best_fold_results": best_trial.user_attrs.get("fold_results", []) if best_trial is not None else [],
            "all_trials": [serialize_trial(trial) for trial in study.trials],
        }

        write_results_file(results_file, results)
        logger.info("Results saved to: %s", results_file)

    return study, best_trial


# per_fold study mode: each fold gets its own study, and each trial evaluates only that single folds
def run_per_fold_optimization(
    n_trials,
    study_name,
    storage_path,
    cmdc_root,
    folds,
    save_root,
    num_gpus,
    model_family,
    prompt_mode,
    resume=True,
    target_total_trials=None,
    enable_pruning=DEFAULT_ENABLE_PRUNING,
    pruner_startup_trials=DEFAULT_PRUNER_STARTUP_TRIALS,
    pruner_warmup_steps=DEFAULT_PRUNER_WARMUP_STEPS,
    pruner_interval_steps=DEFAULT_PRUNER_INTERVAL_STEPS,
):
    os.makedirs(storage_path, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    validate_mode_combination(model_family, prompt_mode)
    validate_audio_fold_configs(cmdc_root, folds, prompt_mode, model_family)

    logger.info("\n%s", "=" * 70)
    logger.info("Starting Optuna optimization")
    logger.info("  Study Mode: per_fold")
    logger.info("  Base Study Name: %s", study_name)
    logger.info("  Model Family: %s", model_family)
    logger.info("  Prompt Mode: %s", prompt_mode)
    logger.info("  Launch Input Mode: %s", resolve_launch_input_mode(model_family))
    logger.info("  Pruning Enabled: %s", enable_pruning)
    logger.info("  Pruner Startup Trials: %s", pruner_startup_trials)
    logger.info("  Pruner Warmup Steps: %s", pruner_warmup_steps)
    logger.info("  Pruner Interval Steps: %s", pruner_interval_steps)
    logger.info("  Trials Per Fold: %s", n_trials)
    logger.info("  CMDC Root: %s", cmdc_root)
    logger.info("  Folds: %s", ", ".join(folds))
    logger.info("  Save Root: %s", save_root)
    logger.info("  Resume Existing Study: %s", resume)
    logger.info(
        "  Target Total Completed Trials Per Fold: %s",
        target_total_trials if target_total_trials is not None else "disabled",
    )
    logger.info("%s\n", "=" * 70)

    model_path = resolve_model_path(model_family)
    fold_summaries = []

    for fold_name in folds:
        fold_study_name = f"{study_name}_{fold_name}"
        fold_save_root = os.path.join(save_root, fold_name)
        os.makedirs(fold_save_root, exist_ok=True)

        logger.info("\n%s", "=" * 70)
        logger.info("Starting per-fold study")
        logger.info("  Fold: %s", fold_name)
        logger.info("  Study Name: %s", fold_study_name)
        logger.info("  Trials: %s", n_trials)
        logger.info("  Save Root: %s", fold_save_root)
        logger.info("%s\n", "=" * 70)

        study = run_study(
            study_name=fold_study_name,
            storage_path=storage_path,
            summary_label=f"Creating per_fold study for {fold_name}",
            load_if_exists=resume,
            enable_pruning=enable_pruning,
            pruner_startup_trials=pruner_startup_trials,
            pruner_warmup_steps=pruner_warmup_steps,
            pruner_interval_steps=pruner_interval_steps,
        )
        objective = build_single_fold_objective(
            cmdc_root=cmdc_root,
            fold_name=fold_name,
            model_path=model_path,
            save_root=fold_save_root,
            num_gpus=num_gpus,
            model_family=model_family,
            prompt_mode=prompt_mode,
            enable_pruning=enable_pruning,
        )

        try:
            requested_trials = resolve_requested_trials(
                study,
                n_trials=n_trials,
                target_total_trials=target_total_trials,
            )
            if requested_trials == 0:
                logger.info(
                    "Fold %s already meets the target total trials; skipping new optimization steps.",
                    fold_name,
                )
            else:
                study.optimize(objective, n_trials=requested_trials, n_jobs=1)
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user during fold %s", fold_name)
            raise
        except Exception as exc:
            logger.error("Fold study %s failed: %s", fold_name, exc)
            logger.exception(exc)

        completed_trials, best_trial = collect_study_summary(study)
        results_file = os.path.join(storage_path, f"{fold_study_name}_results.json")
        storage_file = os.path.join(os.path.abspath(storage_path), f"{fold_study_name}.db")

        results = {
            "study_name": fold_study_name,
            "base_study_name": study_name,
            "study_mode": "per_fold",
            "fold": fold_name,
            "model_family": model_family,
            "prompt_mode": prompt_mode,
            "launch_input_mode": resolve_launch_input_mode(model_family),
            "pruning_enabled": enable_pruning,
            "pruner_startup_trials": pruner_startup_trials,
            "pruner_warmup_steps": pruner_warmup_steps,
            "pruner_interval_steps": pruner_interval_steps,
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "cmdc_root": cmdc_root,
            "best_trial_number": best_trial.number if best_trial is not None else None,
            "best_f1": float(best_trial.value) if best_trial is not None else None,
            "best_params": dict(best_trial.params) if best_trial is not None else None,
            "best_fold_result": (
                best_trial.user_attrs.get("fold_results", [])[0]
                if best_trial is not None and best_trial.user_attrs.get("fold_results")
                else None
            ),
            "all_trials": [serialize_trial(trial) for trial in study.trials],
        }
        write_results_file(results_file, results)

        fold_summary = {
            "fold": fold_name,
            "study_name": fold_study_name,
            "db_path": storage_file,
            "results_path": os.path.abspath(results_file),
            "save_root": os.path.abspath(fold_save_root),
            "model_family": model_family,
            "prompt_mode": prompt_mode,
            "launch_input_mode": resolve_launch_input_mode(model_family),
            "n_trials": len(study.trials),
            "n_completed": len(completed_trials),
            "best_trial_number": best_trial.number if best_trial is not None else None,
            "best_f1": float(best_trial.value) if best_trial is not None else None,
            "best_params": dict(best_trial.params) if best_trial is not None else None,
            "status": "completed" if best_trial is not None else "no_completed_trials",
        }
        fold_summaries.append(fold_summary)

        logger.info("Per-fold results saved to: %s", results_file)
        if best_trial is not None:
            logger.info(
                "Fold %s complete | best trial #%s | best F1: %.4f",
                fold_name,
                best_trial.number,
                best_trial.value,
            )
        else:
            logger.warning("Fold %s complete with no completed trials", fold_name)

    valid_best_scores = [
        fold_summary["best_f1"]
        for fold_summary in fold_summaries
        if fold_summary["best_f1"] is not None and math.isfinite(fold_summary["best_f1"])
    ]
    summary_results = {
        "study_name": study_name,
        "study_mode": "per_fold",
        "folds": folds,
        "model_family": model_family,
        "prompt_mode": prompt_mode,
        "launch_input_mode": resolve_launch_input_mode(model_family),
        "pruning_enabled": enable_pruning,
        "pruner_startup_trials": pruner_startup_trials,
        "pruner_warmup_steps": pruner_warmup_steps,
        "pruner_interval_steps": pruner_interval_steps,
        "trials_per_fold": n_trials,
        "cmdc_root": cmdc_root,
        "save_root": os.path.abspath(save_root),
        "fold_studies": fold_summaries,
        "post_run_best_f1_mean": (
            statistics.mean(valid_best_scores) if valid_best_scores else None
        ),
        "post_run_best_f1_std": (
            statistics.pstdev(valid_best_scores) if len(valid_best_scores) > 1 else 0.0
            if valid_best_scores
            else None
        ),
    }
    summary_file = os.path.join(storage_path, f"{study_name}_results.json")
    write_results_file(summary_file, summary_results)
    logger.info("Per-fold summary saved to: %s", summary_file)

    return fold_summaries


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for CMDC cross-validated Qwen training"
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--target-total-trials",
        type=int,
        default=(
            int(os.environ["TARGET_TOTAL_TRIALS"])
            if os.environ.get("TARGET_TOTAL_TRIALS")
            else None
        ),
        help="Target total number of completed trials in the study. When set, only the remaining trials are run.",
    )
    parser.add_argument("--study-name", type=str, default="cmdc_cv_hpo")
    parser.add_argument("--storage-path", type=str, default=os.environ.get("STORAGE_PATH"))
    parser.add_argument("--cmdc-root", type=str, default=os.environ.get("CMDC_ROOT", "Qwen2-Audio-finetune/data/cmdc"))
    parser.add_argument("--folds", type=str, default=os.environ.get("FOLDS", "fold1 fold2 fold3 fold4 fold5"))
    parser.add_argument("--save-root", type=str, default=os.environ.get("SAVE_ROOT"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "4")))
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
        "--study-mode",
        type=str,
        choices=["cv_mean", "per_fold"],
        default=os.environ.get("STUDY_MODE", "cv_mean"),
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=os.environ.get("RESUME_STUDY", "1").lower() not in {"0", "false", "no"},
        help="Resume an existing study with the same name if its DB already exists (default: enabled).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Fail if the study already exists instead of resuming it.",
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")

    logger.info("GPU Available: %s", torch.cuda.get_device_name(0))
    logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    folds = parse_folds(args.folds)
    prompt_mode = args.prompt_mode
    model_family = normalize_model_family(args.model_family)

    validate_mode_combination(model_family, prompt_mode)
    storage_path = args.storage_path or default_storage_path_for_study_mode(args.study_mode)
    save_root = args.save_root or default_save_root_for_study_mode(args.study_mode)
    os.environ["MODEL_FAMILY"] = model_family
    os.environ["PROMPT_MODE"] = prompt_mode
    os.environ["STORAGE_PATH"] = storage_path
    os.environ["SAVE_ROOT"] = save_root
    if args.study_mode == "per_fold":
        run_per_fold_optimization(
            n_trials=args.n_trials,
            study_name=args.study_name,
            storage_path=storage_path,
            cmdc_root=args.cmdc_root,
            folds=folds,
            save_root=save_root,
            num_gpus=args.num_gpus,
            model_family=model_family,
            prompt_mode=prompt_mode,
            resume=args.resume,
            target_total_trials=args.target_total_trials,
            enable_pruning=args.enable_pruning,
            pruner_startup_trials=args.pruner_startup_trials,
            pruner_warmup_steps=args.pruner_warmup_steps,
            pruner_interval_steps=args.pruner_interval_steps,
        )
    else:
        run_optimization(
            n_trials=args.n_trials,
            study_name=args.study_name,
            storage_path=storage_path,
            cmdc_root=args.cmdc_root,
            folds=folds,
            save_root=save_root,
            num_gpus=args.num_gpus,
            model_family=model_family,
            prompt_mode=prompt_mode,
            resume=args.resume,
            target_total_trials=args.target_total_trials,
            enable_pruning=args.enable_pruning,
            pruner_startup_trials=args.pruner_startup_trials,
            pruner_warmup_steps=args.pruner_warmup_steps,
            pruner_interval_steps=args.pruner_interval_steps,
        )


if __name__ == "__main__":
    main()
