# Optuna Hyperparameter Optimization for Qwen2-7B Text-Only Training

This folder contains a complete setup for hyperparameter optimization using **Optuna** with **4-GPU DDP** training.

## Quick Start

### On MN5 Cluster

```bash
cd optuna_hpo
./run_hpo.sh 20 my_study_name
```

Or submit SLURM job directly:

```bash
cd optuna_hpo
N_TRIALS=20 sbatch train_hpo.slurm
```

### Locally (requires 4 GPUs)

```bash
cd optuna_hpo
python hpo.py --n-trials 20 --study-name my_study
```

## What's Inside

### Core Files

- **`hpo.py`** - Main Optuna optimization loop (entry point)
- **`train_ddp.py`** - DDP training function for each trial
- **`train_ddp_launcher.py`** - Launcher script for torchrun
- **`train_launcher.py`** - Trial orchestration and torchrun spawner
- **`train_hpo.slurm`** - SLURM submission script
- **`run_hpo.sh`** - Quick-start helper script

## Hyperparameters Being Optimized

| Parameter | Search Space | Type |
|-----------|--------------|------|
| **Learning Rate** | 1e-6 to 1e-3 | Logarithmic float |
| **Batch Size** | 1, 2, 4, 8 | Categorical int |
| **LoRA R** | 8 to 64 | Int (step: 8) |
| **LoRA Alpha** | 16 to 128 | Int (step: 16) |

**Objective**: Maximize validation **F1 score**

## Performance Specs

- **GPUs per trial**: 4 (DDP)
- **Training time per trial**: ~1-2 hours
- **Total time for 20 trials**: ~40 hours on 4 GPUs

## Output

After optimization completes:

```
optuna_studies/
├── qwen2_textonly_hpo_YYYYMMDD.db          # Study database (resumable)
└── qwen2_textonly_hpo_YYYYMMDD_results.json # Best hyperparameters + all trials
```

Results JSON structure:
```json
{
  "best_trial_number": 5,
  "best_f1": 0.8642,
  "best_params": {
    "lr": 2.5e-5,
    "batch_size": 4,
    "lora_r": 32,
    "lora_alpha": 64
  },
  "all_trials": [...]
}
```

## Monitor Progress

```bash
# Check SLURM job
squeue -u your_username | grep optuna

# View live output
tail -f logs/optuna-*.out

# Check current best
cat optuna_studies/*_results.json | python -m json.tool
```

## Resume Interrupted Runs

Optuna studies are resumable. To continue from where you left off:

```bash
# Same study name continues trials
N_TRIALS=30 STUDY_NAME=qwen2_textonly_hpo sbatch optuna_hpo/train_hpo.slurm
```

## Next Steps

After finding best hyperparameters, retrain with full training time:

```bash
# From parent directory
LR=2.5e-5 BATCH_SIZE=4 sbatch train_textonly.slurm
```

## Configuration

Edit environment variables in `train_hpo.slurm`:

```bash
export MODEL_PATH="/path/to/Qwen2-7B-Instruct"
export TRAIN_DATA_PATH="/path/to/train/data"
export EVAL_DATA_PATH="/path/to/eval/data"
export SAVE_PATH="/path/to/output"
export N_TRIALS=20
```

## Troubleshooting

**Issue**: CUDA out of memory
- Reduce batch size in config
- Increase gradient accumulation steps
- Enable more aggressive gradient checkpointing

**Issue**: Data loading too slow
- Increase `num_workers` in config
- Adjust `prefetch_factor`

**Issue**: Did not find results file
- Check `/tmp/optuna_trial_*_result.json` files
- Verify training completed successfully in logs
- Check DDP communication issues

## Key Features

✅ Multi-GPU DDP training (4 GPUs per trial)  
✅ Resumable optimization (SQLite database)  
✅ Distributed data sampling  
✅ Automatic metric aggregation across GPUs  
✅ Per-trial logging and checkpointing  
✅ JSON results export  
✅ Gradient checkpointing for memory efficiency  
✅ BFloat16 mixed precision training  
