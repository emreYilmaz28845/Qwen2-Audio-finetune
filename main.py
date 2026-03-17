import hydra
from omegaconf import OmegaConf
from src.train_ddp import train_ddp
from src.train_deepspeed import train_deepspeed
from config.config import Config
import time
import argparse
import sys
import os
import torch


def parse_deepspeed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg):
    # ==== 初始化配置 ====
    run_config = Config()
    run_config.env.save_path += f"/{time.strftime('%H-%M-%S')}"
    cfg = OmegaConf.merge(run_config, cfg)

    # ==== 确保输出目录存在 ====
    os.makedirs(cfg.env.save_path, exist_ok=True)

    # ==== 设置当前 GPU 设备（仅 DDP 模式需要） ====
    if cfg.train.train_strategy == "ddp":
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            print(f"[Rank {local_rank}] Using GPU {local_rank}")
        else:
            print("Warning: LOCAL_RANK not set, defaulting to cuda:0")

    # ==== 训练入口 ====
    if cfg.train.train_strategy == "ddp":
        train_ddp(cfg)
    else:
        train_deepspeed(cfg)


if __name__ == "__main__":
    # 让 Hydra 与 DeepSpeed 的参数兼容
    deepspeed_args, remaining_args = parse_deepspeed_args()
    sys.argv = [sys.argv[0]] + remaining_args  # 仅传递 Hydra 能处理的参数
    main_hydra()
