import torch
# import torch_npu
import os
def set_amp(device_type):
    if device_type == "npu":
        return torch_npu.npu.amp.GradScaler(),torch_npu.npu.amp.autocast
    elif device_type =="cuda":
        return torch.cuda.amp.GradScaler(),torch.cuda.amp.autocast
    