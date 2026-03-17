
import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from .dataset import AudioDataset,collate_fn_qwen2audio
import time
import torch.distributed as dist
import os
import types
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import functools
import random
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_ddp
import os
from peft import LoraConfig
from peft import PeftModel
from utils.functions import compute_acc
from utils.set_amp import set_amp
from safetensors.torch import load_file
def infer(cfg):
    ## 宏参数准备
    device = f"{cfg.env.device_type}:{cfg.eval.local_rank}"
    # model
    processor = AutoProcessor.from_pretrained(cfg.env.model_path,trust_remote_code=True)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.env.model_path,trust_remote_code=True)
    model = PeftModel.from_pretrained(model, cfg.eval.peft_path)
    model.to(device)
    model.print_trainable_parameters()
    eval_dataset = AudioDataset(
        cfg.data.eval_data_path,
        cfg.data.prompt_path,
        cfg.data.wav_type,
        True,
        scp_filename=cfg.data.eval_scp_filename,
        task_filename=cfg.data.eval_task_filename,
    )
    eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,batch_size=cfg.train.batch_size,num_workers=cfg.data.num_workers,collate_fn=partial(collate_fn_qwen2audio,processor=processor),prefetch_factor=cfg.data.prefetch_factor)
    eval_bar = tqdm(eval_dataloader, desc="[Eval]")
    model.eval()
    # 打开预测结果文件（使用with语句确保自动关闭）
    with open(f"{cfg.eval.peft_path}/pred", "w", encoding="utf-8") as f_pred:
        # 遍历评估数据加载器
        for eval_step, batch in enumerate(eval_bar):
            # 将batch中的张量转移到设备（非张量类型保持不变）
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 提取keys并从batch中移除（避免影响模型输入）
            keys = batch.pop("keys")  # 推荐用pop代替remove，更安全且返回被移除的值

            # 模型生成预测结果
            model_outputs = model.generate(** batch, max_length=250)
            # 截取生成结果中超出输入部分的内容（去除输入序列）
            model_outputs = model_outputs[:, batch["input_ids"].size(1):]

            # 将生成的张量解码为文本
            output_text = processor.tokenizer.batch_decode(
                model_outputs,
                add_special_tokens=False,
                skip_special_tokens=True
            )
            # 写入预测结果（key与text用制表符分隔，替换换行符避免格式混乱）
            for key, text in zip(keys, output_text):
                print(key,text)
                f_pred.write(f"{key}\t{text}\n")
