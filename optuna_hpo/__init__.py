"""
Optuna hyperparameter optimization package for Qwen training.

Submodules:
- hpo: Single-dataset Optuna optimization loop
- hpo_cv_5fold: CMDC 5-fold Optuna optimization loop
- train_ddp: Multi-GPU DDP training function
- train_launcher: Trial orchestration and torchrun spawner
- train_ddp_launcher: Launcher script for torchrun
"""

__version__ = "1.0.0"
__author__ = "AudioLLM HPO"

try:
    from optuna_hpo.hpo import objective, run_optimization
    from optuna_hpo.train_launcher import launch_ddp_training
    from optuna_hpo.train_ddp import train_audiotext_ddp, train_ddp, train_textonly_ddp
except ImportError:
    pass
