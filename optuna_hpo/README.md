# Optuna HPO for Qwen Training

This folder contains the hyperparameter search and single-trial launch helpers for Qwen text/audio training with DDP.

There are three main workflows:

- single-dataset HPO: `hpo.py`
- CMDC 5-fold HPO: `hpo_cv_5fold.py`
- single manual trial: `run_one_trial.py`

Canonical entrypoints:

- generic single-dataset HPO: `run_hpo.sh` and `train_hpo.slurm`
- CMDC 5-fold HPO: `run_hpo_cmdc_cv_5fold.sh` and `train_hpo_cmdc_cv_5fold.slurm`
- generic single manual trial: `run_one_trial.py` and `train_single.slurm`

Legacy dataset-specific wrappers such as `run_hpo_daic.sh`, `run_hpo_eatd.sh`, `train_hpo_daic.slurm`, and `train_hpo_eatd.slurm` are no longer the recommended path. The generic interface below should be preferred.

## Interfaces

### Single-dataset HPO

Used by:

- `hpo.py`
- `run_hpo.sh`
- `train_hpo.slurm`

Supported datasets:

- `merged`
- `daic_woz`
- `eatd`

Supported user-facing variables:

- `DATASET_NAME=merged|daic_woz|eatd`
- `MODEL_FAMILY=audio|text`
- `PROMPT_MODE=full|audiotext|textonly`
- `TASK_VARIANT=default|filtered`
- `ENABLE_PRUNING=1|0`
- `PRUNER_STARTUP_TRIALS=<int>`
- `PRUNER_WARMUP_STEPS=<int>`
- `PRUNER_INTERVAL_STEPS=<int>`
- `AUDIOLLM_ENABLE_MODEL_IO_DEBUG=1|0`
- `AUDIOLLM_MODEL_IO_DEBUG_LIMIT=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS=<int>`

Allowed mode combinations:

- `audio + full`
- `audio + audiotext`
- `text + textonly`

`INPUT_MODE` is internal only. It is derived automatically:

- `audio -> audiotext`
- `text -> textonly`

The generic non-CMDC flow now uses `daic_woz`, not `daic`.

### CMDC 5-fold HPO

Used by:

- `hpo_cv_5fold.py`
- `run_hpo_cmdc_cv_5fold.sh`
- `train_hpo_cmdc_cv_5fold.slurm`

Supported user-facing variables:

- `MODEL_FAMILY=audio|text`
- `PROMPT_MODE=full|audiotext|textonly`
- `STUDY_MODE=cv_mean|per_fold`
- `ENABLE_PRUNING=1|0`
- `PRUNER_STARTUP_TRIALS=<int>`
- `PRUNER_WARMUP_STEPS=<int>`
- `PRUNER_INTERVAL_STEPS=<int>`
- `AUDIOLLM_ENABLE_MODEL_IO_DEBUG=1|0`
- `AUDIOLLM_MODEL_IO_DEBUG_LIMIT=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS=<int>`

### Single manual trial

Used by:

- `run_one_trial.py`
- `train_single.slurm`

Supported user-facing variables:

- `DATASET_NAME=merged|daic_woz|eatd`
- `MODEL_FAMILY=audio|text`
- `PROMPT_MODE=full|audiotext|textonly`
- `TASK_VARIANT=default|filtered`
- explicit hyperparameters: `lr`, `batch_size`, `lora_r`, `lora_alpha`
- `AUDIOLLM_ENABLE_MODEL_IO_DEBUG=1|0`
- `AUDIOLLM_MODEL_IO_DEBUG_LIMIT=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS=<int>`

## Prompt and Task Files

Prompt files are selected from `PROMPT_MODE`:

- `textonly -> <dataset>_multiprompt_textonly.jsonl`
- `audiotext -> <dataset>_multiprompt_audiotext.jsonl`
- `full -> <dataset>_multiprompt.jsonl`

Task files are selected from `TASK_VARIANT`:

- `default -> <dataset>_multitask.jsonl`
- `filtered -> <dataset>_multitask_filtered.jsonl`

Dataset split rules:

- `merged`: `train/` and `val/`
- `daic_woz`: `train/` and `val/`
- `eatd`: `train/` and `test/`

Examples:

- `daic_woz` prompt files:
  - `daic_woz_multiprompt.jsonl`
  - `daic_woz_multiprompt_audiotext.jsonl`
  - `daic_woz_multiprompt_textonly.jsonl`
- `daic_woz` task files:
  - `daic_woz_multitask.jsonl`
  - `daic_woz_multitask_filtered.jsonl`

## Core Files

- `hpo.py`: single-dataset Optuna loop
- `hpo_cv_5fold.py`: CMDC 5-fold Optuna loop
- `train_launcher.py`: writes trial config and launches `torchrun`
- `train_ddp_launcher.py`: reads trial config and calls `train_ddp(...)`
- `train_ddp.py`: actual DDP training code
- `run_hpo.sh`: local / cluster helper for single-dataset HPO
- `train_hpo.slurm`: MN5 single-dataset HPO job
- `run_hpo_cmdc_cv_5fold.sh`: local / cluster helper for CMDC HPO
- `train_hpo_cmdc_cv_5fold.slurm`: MN5 CMDC HPO job
- `run_one_trial.py`: direct single-trial launcher
- `train_single.slurm`: MN5 single-trial job

## Hyperparameters

Current search space:

| Parameter | Values |
|---|---|
| `lr` | `1e-6` to `1e-3` log scale |
| `batch_size` | `1, 2, 4` |
| `lora_r` | `8, 12, 16` |
| `lora_alpha` | `8, 16, 24, 32` |

Objective: maximize validation F1.

## Median Pruning

Median pruning is enabled by default with conservative settings:

- `ENABLE_PRUNING=1`
- `PRUNER_STARTUP_TRIALS=5`
- `PRUNER_WARMUP_STEPS=2`
- `PRUNER_INTERVAL_STEPS=1`

Pruning behavior depends on study mode:

- single-dataset HPO in `hpo.py`: pruning uses intermediate validation F1 at each evaluation event
- CMDC `per_fold`: pruning uses intermediate validation F1 within each fold-specific study
- CMDC `cv_mean`: pruning happens only between folds using the partial mean F1 of completed folds

Step semantics:

- single-dataset and `per_fold`: one Optuna step = one validation event
- `cv_mean`: one Optuna step = one completed fold

## Quick Start

### Single-dataset HPO locally

```bash
cd Qwen2-Audio-finetune
DATASET_NAME=merged MODEL_FAMILY=audio PROMPT_MODE=audiotext ./optuna_hpo/run_hpo.sh 20
```

With filtered task files:

```bash
DATASET_NAME=daic_woz MODEL_FAMILY=text PROMPT_MODE=textonly TASK_VARIANT=filtered ./optuna_hpo/run_hpo.sh 20
```

### Single-dataset HPO on MN5

```bash
sbatch --export=DATASET_NAME=daic_woz,MODEL_FAMILY=audio,PROMPT_MODE=audiotext,N_TRIALS=1 optuna_hpo/train_hpo.slurm
```

With filtered task files:

```bash
sbatch --export=DATASET_NAME=daic_woz,MODEL_FAMILY=text,PROMPT_MODE=textonly,TASK_VARIANT=filtered,N_TRIALS=20 optuna_hpo/train_hpo.slurm
```

### CMDC 5-fold HPO locally

```bash
cd Qwen2-Audio-finetune
MODEL_FAMILY=audio PROMPT_MODE=audiotext STUDY_MODE=cv_mean ./optuna_hpo/run_hpo_cmdc_cv_5fold.sh 20
```

### CMDC 5-fold HPO on MN5

```bash
sbatch --export=MODEL_FAMILY=audio,PROMPT_MODE=full,STUDY_MODE=per_fold,N_TRIALS=20 optuna_hpo/train_hpo_cmdc_cv_5fold.slurm
```

### Single manual trial on MN5

```bash
sbatch --export=DATASET_NAME=daic_woz,MODEL_FAMILY=audio,PROMPT_MODE=audiotext,LR=1e-4,BATCH_SIZE=4,LORA_R=16,LORA_ALPHA=24 optuna_hpo/train_single.slurm
```

### Single manual trial directly

```bash
python optuna_hpo/run_one_trial.py \
  --trial-number 999 \
  --lr 1e-4 \
  --batch-size 4 \
  --lora-r 16 \
  --lora-alpha 24 \
  --dataset-name daic_woz \
  --model-family audio \
  --prompt-mode audiotext
```

## Output Layout

### Single-dataset HPO

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

For `DATASET_NAME=daic_woz`:

```text
logs/
└── optuna_daic_woz/
    ├── optuna-qwen-optuna-daic-woz-<jobid>.out
    ├── optuna-qwen-optuna-daic-woz-<jobid>.err
    └── optuna_daic_woz_<timestamp>.log

optuna_studies/
└── optuna_daic_woz/
    ├── daic_woz_audio_audiotext_hpo_<timestamp>.db
    └── daic_woz_audio_audiotext_hpo_<timestamp>_results.json

output_model/
└── optuna_daic_woz_hpo/
    ├── textonly/
    ├── audiotext/
    └── full/
```

For `DATASET_NAME=eatd`:

```text
logs/
└── optuna_eatd/
    ├── optuna-qwen-optuna-eatd-<jobid>.out
    ├── optuna-qwen-optuna-eatd-<jobid>.err
    └── optuna_eatd_<timestamp>.log

optuna_studies/
└── optuna_eatd/
    ├── eatd_audio_audiotext_hpo_<timestamp>.db
    └── eatd_audio_audiotext_hpo_<timestamp>_results.json

output_model/
└── optuna_eatd_hpo/
    ├── textonly/
    ├── audiotext/
    └── full/
```

### CMDC 5-fold HPO

For `STUDY_MODE=cv_mean`:

```text
logs/
└── optuna_cmdc_cv_5fold_cv_mean/
    ├── optuna-qwen-optuna-cmdc-cv-<jobid>.out
    ├── optuna-qwen-optuna-cmdc-cv-<jobid>.err
    └── optuna_cmdc_cv_5fold_<timestamp>.log

optuna_studies/
└── optuna_cmdc_cv_5fold_cv_mean/
    ├── cmdc_audio_audiotext_cv_mean_hpo_<timestamp>.db
    └── cmdc_audio_audiotext_cv_mean_hpo_<timestamp>_results.json

output_model/
└── optuna_cmdc_5fold_hpo/
    ├── textonly/
    │   └── optuna_cmdc_cv_5fold_cv_mean/
    ├── audiotext/
    │   └── optuna_cmdc_cv_5fold_cv_mean/
    └── full/
        └── optuna_cmdc_cv_5fold_cv_mean/
```

For `STUDY_MODE=per_fold`, replace the suffix with `per_fold`.

Default CMDC output root is nested by prompt mode:

```text
output_model/
└── optuna_cmdc_5fold_hpo/
    ├── textonly/
    ├── audiotext/
    └── full/
```

### Single manual trial

For `DATASET_NAME=daic_woz`:

```text
logs/
└── single_trial_daic_woz/
    ├── single-trial-<jobid>.out
    ├── single-trial-<jobid>.err
    └── single_trial_<timestamp>.log

output_model/
└── single_trial/
    └── daic_woz/
        ├── textonly/
        ├── audiotext/
        └── full/
```

## GPU Logging

GPU logging is enabled in the cluster scripts:

- `train_hpo.slurm`
- `train_hpo_cmdc_cv_5fold.slurm`
- `train_single.slurm`

Relevant env vars:

- `AUDIOLLM_ENABLE_GPU_LOG=1|0`
- `AUDIOLLM_GPU_LOG_INTERVAL_SEC=<seconds>`

Logged values:

- GPU index
- VRAM used MB
- VRAM total MB
- VRAM used percent
- GPU utilization percent

## Model IO Debug

Model input/output debug logging is available in the cluster scripts:

- `train_hpo.slurm`
- `train_hpo_cmdc_cv_5fold.slurm`
- `train_single.slurm`

Relevant env vars:

- `AUDIOLLM_ENABLE_MODEL_IO_DEBUG=1|0`
- `AUDIOLLM_MODEL_IO_DEBUG_LIMIT=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS=<int>`
- `AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS=<int>`

Useful companion env var for the tokenizer fork warning:

- `TOKENIZERS_PARALLELISM=false`

When enabled, rank 0 logs a small number of decoded examples for both `train` and `eval`:

- prompt text
- full decoded model input
- decoded target label span
- decoded predicted output span

Cadence controls:

- `AUDIOLLM_MODEL_IO_DEBUG_LIMIT`: total number of debug samples per phase
- `AUDIOLLM_MODEL_IO_DEBUG_TRAIN_EVERY_STEPS`: log every Nth train step
- `AUDIOLLM_MODEL_IO_DEBUG_EVAL_EVERY_CALLS`: log every Nth eval event

## Trial Launch Chain

Both HPO entrypoints eventually use:

```text
Optuna objective
  -> train_launcher.launch_ddp_training(...)
  -> torchrun train_ddp_launcher.py
  -> train_ddp(...)
```

When pruning is enabled, `train_launcher.py` polls intermediate progress from the DDP child, reports it to Optuna, and terminates the subprocess early when the pruner decides to stop a weak trial.

The single manual trial uses the same lower-level launch path.

The actual execution chain is:

```text
Optuna or run_one_trial.py
  -> train_launcher.launch_ddp_training(...)
  -> torchrun train_ddp_launcher.py
  -> train_ddp(...)
```

## Resume Behavior

Optuna studies are SQLite-backed and resumable. Reusing the same `STUDY_NAME` and `STORAGE_PATH` continues the study.

## Monitor Progress

Examples:

```bash
squeue -u your_username | grep optuna
tail -f logs/optuna_daic_woz/*.out
tail -f logs/optuna_cmdc_cv_5fold_cv_mean/*.out
python -m json.tool optuna_studies/optuna_daic_woz/*_results.json
```

## Environment

The SLURM scripts currently activate:

```bash
/gpfs/projects/etur92/ozu647717/venvs/qwen_mn5_rebuilt/bin/activate
```

## Notes

- `full` is a prompt variant, not a separate training backend
- `INPUT_MODE` still exists internally in `train_ddp.py`, but it is derived automatically by the user-facing scripts
- `TASK_VARIANT=filtered` is mainly useful for dataset-specific filtered task files without needing separate wrapper scripts
