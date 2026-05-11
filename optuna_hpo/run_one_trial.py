#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from optuna_hpo.hpo import (
    DATASET_DAIC_WOZ,
    DATASET_EATD,
    DATASET_MERGED,
    DEFAULT_GROUPED_EVAL_LEVEL,
    DEFAULT_GROUPED_EVAL_MODE,
    MODEL_FAMILY_AUDIO,
    MODEL_FAMILY_TEXT,
    PROMPT_MODE_AUDIOTEXT,
    PROMPT_MODE_FULL,
    PROMPT_MODE_TEXTONLY,
    SUPPORTED_GROUPED_EVAL_LEVELS,
    SUPPORTED_GROUPED_EVAL_MODES,
    TASK_VARIANT_DEFAULT,
    TASK_VARIANT_FILTERED,
    apply_dataset_env,
    apply_grouped_eval_env,
    default_save_root,
    get_dataset_config,
    normalize_model_family,
    resolve_grouped_eval_settings,
    resolve_launch_input_mode,
    resolve_model_path,
    validate_mode_combination,
)
from optuna_hpo.train_launcher import launch_ddp_training


def main():
    parser = argparse.ArgumentParser(description="Run a single DDP training trial with explicit hyperparameters")
    parser.add_argument("--trial-number", type=int, default=999)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lora-r", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "4")))
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[DATASET_MERGED, DATASET_DAIC_WOZ, DATASET_EATD],
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
        "--save-path",
        type=str,
        default=os.environ.get("SAVE_PATH"),
        help="Optional output root override",
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

    model_family = normalize_model_family(args.model_family)
    validate_mode_combination(model_family, args.prompt_mode)
    grouped_settings = resolve_grouped_eval_settings(args)

    dataset_cfg = get_dataset_config(args.dataset_name, args.prompt_mode, args.task_variant)
    apply_dataset_env(dataset_cfg)
    apply_grouped_eval_env(grouped_settings)

    os.environ["DATASET_NAME"] = args.dataset_name
    os.environ["MODEL_FAMILY"] = model_family
    os.environ["PROMPT_MODE"] = args.prompt_mode
    os.environ["TASK_VARIANT"] = args.task_variant

    launch_input_mode = resolve_launch_input_mode(model_family)
    model_path = resolve_model_path(model_family)
    save_path = args.save_path or default_save_root(args.dataset_name, args.prompt_mode)

    result = launch_ddp_training(
        trial=None,
        trial_params={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
        trial_number=args.trial_number,
        model_path=model_path,
        train_data_path=dataset_cfg.train_data_path,
        eval_data_path=dataset_cfg.eval_data_path,
        save_path=save_path,
        dataset_name=args.dataset_name,
        input_mode=launch_input_mode,
        num_gpus=args.num_gpus,
        daic_eval_level=grouped_settings[DATASET_DAIC_WOZ]["level"],
        daic_eval_mode=grouped_settings[DATASET_DAIC_WOZ]["mode"],
        daic_person_threshold=grouped_settings[DATASET_DAIC_WOZ]["threshold"],
    )
    print(f"Training completed! Best F1: {result}")


if __name__ == "__main__":
    main()
