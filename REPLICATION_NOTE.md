# Replication Note

This repository is our modified version of the official code released for the paper *DepressInstruct: Instruction Tuning of Large Speech-Language Models for Depression Detection* (`papers/DepresInstruct.pdf`). The snapshot used for this note is commit `33b6cb9` on branch `main`.

The goal of our work was not to run the upstream repository as-is, but to make it runnable on our setup and to reproduce the training/evaluation workflow in a controlled way. Based on the full git history, the repository evolved in four main steps:

- the original training code was rebuilt and adapted for our environment
- a separate text-only Qwen2-7B training path and Optuna HPO pipeline were added
- CMDC 5-fold cross-validation, `textonly` and `audiotext` prompt variants, and evaluation scripts were added
- several fixes were needed for padding/masking, missing CMDC `.scp` paths, failed Optuna trials, resume support, and logging/debugging

*Our github = https://github.com/emreYilmaz28845/Qwen2-Audio-finetune

## 1. Environment

Work from:

```bash
cd AudioLLM/Qwen2-Audio-finetune
```

Create the environment from `environment.yml`, then activate it. On our cluster we used a Python environment with `torch`, `transformers`, `peft`, `deepspeed`, `kaldiio`, `soundfile`, and `scikit-learn`. The model paths we used were:

- audio model: `Qwen2-Audio-7B-Instruct`
- text-only model: `Qwen2-7B-Instruct`
- our env path : `source /gpfs/projects/etur92/ozu647717/venvs/qwen_mn5`

## 2. Data Assumption

The repository contains prepared depression datasets under `data/`. The current primary protocol uses clean participant-level holdout splits:

- `data/{daic_woz,eatd,cmdc,merged}/train`
- `data/{daic_woz,eatd,cmdc,merged}/val`
- `data/{daic_woz,eatd,cmdc,merged}/test`

The important files for each split are:

- `*.scp` for audio paths
- `*_multitask.jsonl` for labels/tasks
- `*_multiprompt.jsonl`, `*_multiprompt_textonly.jsonl`, or `*_multiprompt_audiotext.jsonl` for prompt style

The split audit files are:

- `data/{daic_woz,eatd,cmdc,merged}/split_audit.json`
- `data/clean_holdout_manifest_audit.json`

Regenerate the active clean splits with:

```bash
python tools/regenerate_clean_holdout_manifests.py
python tools/validate_splits.py --all --strict
```

The old `data/cmdc/fold*/train` and `data/cmdc/fold*/test` directories are retained for audit and secondary diagnostics only. They are not the primary reporting protocol.

## 3. What We Actually Reproduced

The most reproducible workflow in this repository is the **CMDC 5-fold text-only experiment with Optuna HPO**, followed by evaluation.

To launch the search locally:

```bash
INPUT_MODE=textonly STUDY_MODE=per_fold bash optuna_hpo/run_hpo_cmdc_cv_5fold.sh 20 cmdc_textonly_cv_hpo_repl
```

To launch it on the cluster:

```bash
cd AudioLLM/Qwen2-Audio-finetune
INPUT_MODE=textonly STUDY_MODE=per_fold N_TRIALS=20 sbatch optuna_hpo/train_hpo_cmdc_cv_5fold.slurm
```

This runs 20 Optuna trials for each CMDC fold. The main output files are stored in `optuna_studies/`. The relevant supporting changes in history are the addition of 5-fold HPO, the `per_fold` study mode, fixes for `-inf` trials, fixes for multi-GPU JSON logging, and later resume support.

The clearest completed result in the repository is:

- result file: `optuna_studies/cmdc_textonly_cv_hpo_20260417_133852_results.json`
- post-run mean best F1 over 5 folds: `0.8849494957968311`

Best trial settings found for each fold in that run:

- fold1: `lr=8.841e-4`, `batch_size=4`, `lora_r=12`, `lora_alpha=16`
- fold2: `lr=7.494e-4`, `batch_size=2`, `lora_r=8`, `lora_alpha=24`
- fold3: `lr=9.681e-4`, `batch_size=2`, `lora_r=8`, `lora_alpha=32`
- fold4: `lr=8.257e-4`, `batch_size=4`, `lora_r=16`, `lora_alpha=16`
- fold5: `lr=8.853e-4`, `batch_size=2`, `lora_r=12`, `lora_alpha=8`

## 4. Evaluation

For evaluation we used `eval.sh`, which supports:

- `MODEL_FAMILY=audio|text`
- `PROMPT_MODE=full|audiotext|textonly`

Example:

```bash
MODEL_FAMILY=text PROMPT_MODE=textonly bash eval.sh /path/to/checkpoint
```

or to evaluate the base model without LoRA:

```bash
MODEL_FAMILY=text PROMPT_MODE=textonly bash eval.sh none
```

The per-dataset metrics are computed by `evaluate_per_dataset.py`.

Final reporting now defaults to `DATA_SPLIT=test`. Evaluation on `train` or `val` is blocked unless `ALLOW_NON_TEST_EVAL=1` is set explicitly for debugging.

Optuna HPO uses only the clean `val` split for pruning and best-checkpoint selection. The untouched `test` split should be used only after model/hyperparameter selection.

## 5. Important Note

The previously transferred Optuna/evaluation outputs that reached F1=1.0 should not be used as final claims. Those runs mixed validation/model-selection and final-evaluation roles, and CMDC/EATD also had split identity problems in the older generated manifests. They are useful audit artifacts, not clean holdout results.

Anyone trying to repeat our results should use this repository version rather than the original upstream code. The git history shows that reproducibility depended on multiple repository-side changes, not on a single training command. In particular, the final workflow depends on:

- our modified prompt files
- our CMDC fold structure
- our Optuna scripts and their later bug fixes
- our custom evaluation scripts

Also, the most reliable path in the current repository is effectively **text-centered**. `evaluate_per_dataset.py` explicitly matches a setup where LoRA was attached to `language_model`, so this repository should not be described as a clean reproduction of the full upstream audio pipeline without additional verification.
