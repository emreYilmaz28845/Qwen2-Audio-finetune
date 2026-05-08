import os

import optuna


DEFAULT_ENABLE_PRUNING = True
DEFAULT_PRUNER_STARTUP_TRIALS = 5
DEFAULT_PRUNER_WARMUP_STEPS = 2
DEFAULT_PRUNER_INTERVAL_STEPS = 1


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_pruner(enable_pruning: bool, startup_trials: int, warmup_steps: int, interval_steps: int):
    if not enable_pruning:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=startup_trials,
        n_warmup_steps=warmup_steps,
        interval_steps=interval_steps,
    )
