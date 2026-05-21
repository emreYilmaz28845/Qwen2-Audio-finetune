import os
import subprocess
import sys
from pathlib import Path

from optuna_hpo.path_helpers import build_cmdc_cv_layout, build_single_dataset_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _lines_to_dict(stdout: str):
    result = {}
    for line in stdout.strip().splitlines():
        key, value = line.split("=", 1)
        result[key] = value
    return result


def test_single_dataset_layout_daic_person():
    layout = build_single_dataset_layout(
        dataset_name="daic_woz",
        prompt_mode="audiotext",
        eval_level="person",
        study_timestamp="20260520_143000",
        trial_number=1,
        lr=4e-05,
        batch_size=1,
        lora_r=8,
        lora_alpha=16,
    )

    assert layout.output_model_root == "output_model/optuna_daic_woz_hpo_person"
    assert layout.study_dir == (
        "output_model/optuna_daic_woz_hpo_person/Hpo_Study_audiotext_20260520_143000"
    )
    assert layout.trial_output_dir == (
        "output_model/optuna_daic_woz_hpo_person/Hpo_Study_audiotext_20260520_143000/"
        "audiotext_trial_001_lr4e-05_bs1_r8_a16"
    )
    assert layout.best_model_dir == f"{layout.trial_output_dir}/best_model"
    assert "_trial_trial_" not in layout.trial_output_dir


def test_single_dataset_layout_eatd_person():
    layout = build_single_dataset_layout(
        dataset_name="eatd",
        prompt_mode="textonly",
        eval_level="person",
        study_timestamp="20260520_143000",
        trial_number=1,
        lr=4e-05,
        batch_size=1,
        lora_r=8,
        lora_alpha=16,
    )

    assert layout.output_model_root == "output_model/optuna_eatd_hpo_person"
    assert layout.study_dir == "output_model/optuna_eatd_hpo_person/Hpo_Study_textonly_20260520_143000"
    assert layout.trial_output_dir.endswith("textonly_trial_001_lr4e-05_bs1_r8_a16")
    assert layout.best_model_dir == f"{layout.trial_output_dir}/best_model"


def test_cmdc_layout_cv_mean_fold_best_model():
    layout = build_cmdc_cv_layout(
        study_mode="cv_mean",
        prompt_mode="audiotext",
        eval_level="person",
        study_timestamp="20260520_143000",
        trial_number=1,
        lr=4e-05,
        batch_size=1,
        lora_r=8,
        lora_alpha=16,
        fold_name="fold1",
    )

    assert layout.output_model_root == "output_model/optuna_cmdc_cv_5fold_cv_mean_person"
    assert layout.trial_output_dir.endswith("audiotext_trial_001_lr4e-05_bs1_r8_a16")
    assert layout.fold_output_dir == f"{layout.trial_output_dir}/fold1"
    assert layout.best_model_dir == f"{layout.fold_output_dir}/best_model"


def test_hpo_py_print_paths_only_daic():
    env = os.environ.copy()
    env["PRINT_PATHS_ONLY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "optuna_hpo/hpo.py",
            "--dataset-name",
            "daic_woz",
            "--model-family",
            "audio",
            "--prompt-mode",
            "audiotext",
            "--daic-eval-level",
            "person",
            "--study-timestamp",
            "20260520_143000",
            "--trial-number",
            "1",
            "--lr",
            "4e-05",
            "--batch-size",
            "1",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = _lines_to_dict(result.stdout)
    assert paths["study_name"] == "Hpo_Study_audiotext_20260520_143000"
    assert paths["output_model_root"] == "output_model/optuna_daic_woz_hpo_person"
    assert paths["trial_output_dir"].endswith("audiotext_trial_001_lr4e-05_bs1_r8_a16")
    assert "_trial_trial_" not in paths["trial_output_dir"]


def test_hpo_cv_print_paths_only_cmdc_fold():
    env = os.environ.copy()
    env["PRINT_PATHS_ONLY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "optuna_hpo/hpo_cv_5fold.py",
            "--model-family",
            "audio",
            "--prompt-mode",
            "audiotext",
            "--study-mode",
            "cv_mean",
            "--cmdc-eval-level",
            "person",
            "--study-timestamp",
            "20260520_143000",
            "--trial-number",
            "1",
            "--lr",
            "4e-05",
            "--batch-size",
            "1",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--fold-name",
            "fold1",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = _lines_to_dict(result.stdout)
    assert paths["study_name"] == "Hpo_Study_audiotext_20260520_143000"
    assert paths["output_model_root"] == "output_model/optuna_cmdc_cv_5fold_cv_mean_person"
    assert paths["fold_output_dir"].endswith("/fold1")
    assert paths["best_model_dir"].endswith("/fold1/best_model")


def test_run_one_trial_print_paths_only_eatd():
    env = os.environ.copy()
    env["PRINT_PATHS_ONLY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "optuna_hpo/run_one_trial.py",
            "--dataset-name",
            "eatd",
            "--model-family",
            "text",
            "--prompt-mode",
            "textonly",
            "--lr",
            "4e-05",
            "--batch-size",
            "1",
            "--lora-r",
            "8",
            "--lora-alpha",
            "16",
            "--study-timestamp",
            "20260520_143000",
            "--trial-number",
            "1",
            "--eatd-eval-level",
            "person",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    paths = _lines_to_dict(result.stdout)
    assert paths["output_model_root"] == "output_model/optuna_eatd_hpo_person"
    assert paths["best_model_dir"].endswith("/best_model")
