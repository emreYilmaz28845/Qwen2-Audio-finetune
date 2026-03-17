import os 
import torch
# import torch_npu
import torch.distributed as dist
import deepspeed
# 设置分布式运行环境
def setup_ddp(device_type):
    local_rank = os.environ["LOCAL_RANK"]
    if device_type == "npu":
        torch.npu.set_device(f"npu:{local_rank}")  # 绑定当前NPU
        dist.init_process_group(
        backend='hccl',    # 使用NCCL后端（GPU场景）
    )
    elif device_type =="cuda":
        torch.cuda.set_device(f"cuda:{local_rank}")  # 绑定当前GPU
        dist.init_process_group(
        backend='nccl',    # 使用NCCL后端（GPU场景）
    )
def setup_deepspeed(device_type):
    local_rank = os.environ["LOCAL_RANK"]
    if device_type == "npu":
        torch.npu.set_device(f"npu:{local_rank}")  # 绑定当前NPU
        deepspeed.init_distributed(
        dist_backend='hccl',    # 使用NCCL后端（GPU场景）
    )
    elif device_type =="cuda":
        torch.cuda.set_device(f"cuda:{local_rank}")  # 绑定当前GPU
        deepspeed.init_distributed(
        dist_backend='nccl',    # 使用NCCL后端（GPU场景）
    )