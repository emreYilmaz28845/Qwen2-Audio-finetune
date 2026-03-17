import math
import os

from omegaconf import OmegaConf


class WandbLogger:
    def __init__(self, cfg, save_path, is_main_process, logger=None):
        self.cfg = cfg
        self.save_path = save_path
        self.is_main_process = is_main_process
        self.logger = logger
        self.run = None
        self.enabled = bool(cfg.wandb.get("enabled", False)) and is_main_process
        self._wandb = None

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging is enabled, but the `wandb` package is not installed. "
                "Install it with `pip install wandb` or disable it with `wandb.enabled=false`."
            ) from exc

        self._wandb = wandb
        run_dir = cfg.wandb.get("dir") or save_path
        os.makedirs(run_dir, exist_ok=True)

        run_name = cfg.wandb.get("run_name") or os.path.basename(save_path.rstrip(os.sep))
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        self.run = wandb.init(
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity") or None,
            name=run_name,
            group=cfg.wandb.get("group") or None,
            job_type=cfg.wandb.get("job_type") or "train",
            mode=cfg.wandb.get("mode") or "online",
            dir=run_dir,
            config=config_dict,
            tags=list(cfg.wandb.get("tags", [])),
            save_code=bool(cfg.wandb.get("save_code", False)),
        )

    def watch(self, model):
        if not self.run or not bool(self.cfg.wandb.get("watch_model", False)):
            return
        self._wandb.watch(
            model,
            log=self.cfg.wandb.get("watch_log", "gradients"),
            log_freq=int(self.cfg.wandb.get("watch_log_freq", 100)),
        )

    def log(self, metrics, step=None):
        if not self.run:
            return

        payload = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "numel"):
                if value.numel() != 1:
                    continue
                value = value.item()
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                payload[key] = value

        if payload:
            self.run.log(payload, step=step)

    def update_summary(self, metrics):
        if not self.run:
            return
        for key, value in metrics.items():
            self.run.summary[key] = value

    def finish(self):
        if self.run:
            self.run.finish()
