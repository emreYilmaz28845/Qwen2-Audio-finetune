# Optuna Hyperparameter Optimization for Qwen Training

This folder contains Optuna-based hyperparameter optimization workflows for Qwen training with 4-GPU DDP.

There are two HPO entrypoints:

- `hpo.py`: single-dataset HPO for `merged`, `daic`, or `eatd`
- `hpo_cv_5fold.py`: CMDC 5-fold HPO with `cv_mean` or `per_fold` study modes

## User-Facing Modes

The current interface uses:

- `DATASET_NAME=merged|daic|eatd` for `hpo.py`
- `MODEL_FAMILY=audio|text`
- `PROMPT_MODE=full|audiotext|textonly`

Allowed combinations:

- `audio + full`
- `audio + audiotext`
- `text + textonly`

`INPUT_MODE` is now treated as an internal training detail. It is derived automatically from `MODEL_FAMILY` and is not meant to be set directly in the non-CMDC flow.

## Core Files

- `hpo.py`: single-dataset Optuna optimization loop
- `hpo_cv_5fold.py`: CMDC 5-fold Optuna optimization loop
- `train_launcher.py`: writes trial config and launches `torchrun`
- `train_ddp_launcher.py`: reads trial config and calls `train_ddp(...)`
- `train_ddp.py`: actual text/audio DDP training logic
- `train_hpo.slurm`: MN5 submission script for single-dataset HPO
- `train_hpo_cmdc_cv_5fold.slurm`: MN5 submission script for CMDC 5-fold HPO
- `run_hpo.sh`: local/cluster helper for single-dataset HPO
- `run_hpo_cmdc_cv_5fold.sh`: local/cluster helper for CMDC 5-fold HPO

## Hyperparameters Being Optimized

| Parameter | Search Space | Type |
|-----------|--------------|------|
| `lr` | `1e-6` to `1e-3` | Logarithmic float |
| `batch_size` | `1, 2, 4` | Categorical int |
| `lora_r` | `8, 12, 16` | Int |
| `lora_alpha` | `8, 16, 24, 32` | Int |

Objective: maximize validation F1.

## Single-Dataset HPO

### Local

```bash
cd optuna_hpo
DATASET_NAME=merged MODEL_FAMILY=audio PROMPT_MODE=audiotext ./run_hpo.sh 20
```

You can also run the Python entrypoint directly:

```bash
python hpo.py \
  --n-trials 20 \
  --dataset-name merged \
  --model-family audio \
  --prompt-mode audiotext
```

### MN5 Cluster

```bash
DATASET_NAME=daic MODEL_FAMILY=text PROMPT_MODE=textonly sbatch train_hpo.slurm
```

### Default Output Layout

For `DATASET_NAME=merged`:

```text
logs/
└── optuna_merged/
    ├── optuna-qwen-optuna-merged-<jobid>.out
    ├── optuna-qwen-optuna-merged-<jobid>.err
    └── optuna_merged_<timestamp>.log

optuna_studies/
└── optuna_merged/
    ├── merged_audio_audiotext_hpo_<timestamp>.db
    └── merged_audio_audiotext_hpo_<timestamp>_results.json

output_model/
└── optuna_merged_hpo/
    ├── textonly/
    ├── audiotext/
    └── full/
```

For `DATASET_NAME=daic` or `eatd`, the same structure is used with `optuna_daic` / `optuna_eatd`.

## CMDC 5-Fold HPO

CMDC supports two study modes:

- `cv_mean`: one Optuna trial evaluates one parameter set across all folds and optimizes mean fold F1
- `per_fold`: one Optuna study per fold, each optimizing only that fold

### Local

```bash
MODEL_FAMILY=audio PROMPT_MODE=audiotext STUDY_MODE=cv_mean ./run_hpo_cmdc_cv_5fold.sh 20
```

### MN5 Cluster

```bash
MODEL_FAMILY=audio PROMPT_MODE=full STUDY_MODE=per_fold sbatch train_hpo_cmdc_cv_5fold.slurm
```

### Default Output Layout

For `STUDY_MODE=cv_mean`:

```text
logs/
└── optuna_cmdc_cv_5fold_cv_mean/
    ├── optuna-qwen-optuna-cmdc-cv-<jobid>.out
    ├── optuna-qwen-optuna-cmdc-cv-<jobid>.err
    └── optuna_cmdc_cv_5fold_<timestamp>.log

optuna_studies/
└── optuna_cmdc_5fold_cv_mean/
    ├── cmdc_audio_audiotext_cv_mean_hpo_<timestamp>.db
    └── cmdc_audio_audiotext_cv_mean_hpo_<timestamp>_results.json

output_model/
└── optuna_cmdc_5fold_hpo/
    ├── textonly/
    ├── audiotext/
    └── full/
```

For `STUDY_MODE=per_fold`, replace the folder suffix with `per_fold`.

## How One Trial Runs

Both entrypoints eventually use the same launch chain:

```text
Optuna objective
  -> train_launcher.launch_ddp_training(...)
  -> torchrun train_ddp_launcher.py
  -> train_ddp(...)
```

`train_launcher.py` writes a temp JSON config, launches `torchrun`, waits for training to finish, and returns the trial's best F1 back to Optuna.

## Resume Behavior

Studies are resumable because Optuna uses SQLite.

If you rerun with the same `STUDY_NAME` and `STORAGE_PATH`, new trials continue in the same study DB.

## Monitor Progress

Examples:

```bash
squeue -u your_username | grep optuna
tail -f logs/optuna_merged/*.out
tail -f logs/optuna_cmdc_cv_5fold_cv_mean/*.out
python -m json.tool optuna_studies/optuna_merged/*_results.json
```

## Notes

- `eatd` uses `train/` and `test/` splits; `merged` and `daic` use `train/` and `val/`
- non-CMDC HPO now derives prompt files from `PROMPT_MODE`
- CMDC HPO also separates prompt mode from training backend
- `full` is a prompt variant, not a separate `train_ddp.py` backend mode
