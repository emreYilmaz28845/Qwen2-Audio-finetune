import torch
import random
# import torch_npu
import numpy as np
import deepspeed
def set_seed(seed):
    # torch.npu.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # 设置 NumPy 的种子
    np.random.seed(seed)
    # 设置 Python 内置 random 库的种子
    random.seed(seed)
    deepspeed.runtime.utils.set_random_seed(seed)
