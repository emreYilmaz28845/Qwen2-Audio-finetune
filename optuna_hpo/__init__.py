"""
Optuna Hyperparameter Optimization Package for Qwen2-7B Text-Only Training

Submodules:
- hpo: Main Optuna optimization loop
- train_ddp: Multi-GPU DDP training function
- train_launcher: Trial orchestration and torchrun spawner
- train_ddp_launcher: Launcher script for torchrun
"""

__version__ = "1.0.0"
__author__ = "AudioLLM HPO"

try:
    from optuna_hpo.hpo import objective, run_optimization
    from optuna_hpo.train_launcher import launch_ddp_training
    from optuna_hpo.train_ddp import train_textonly_ddp
except ImportError:
    pass
