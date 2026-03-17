import torchaudio
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from functools import partial
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from .dataset import AudioDataset, collate_fn_qwen2audio
import time
import torch.distributed as dist
import os
import math
import torch
import deepspeed
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_deepspeed
from utils.system_metrics import (
    collect_memory_metrics,
    format_memory_metrics,
    prefix_metrics,
    reset_peak_memory_stats,
)
from utils.wandb_logger import WandbLogger
from utils.functions import compute_acc, compute_metrics
import torch.nn as nn

# ===============================
# Adapter 模块定义
# ===============================
class DepAdapter(nn.Module):
    def __init__(self, audio_dim, adapter_dim=512, dropout=0.1):
        super().__init__()
        self.down_proj = nn.Linear(audio_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, audio_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(audio_dim)

    def forward(self, audio_features):
        residual = audio_features
        x = self.down_proj(audio_features)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

# ===============================
# 修改 Qwen2Audio 编码器，加入 Adapter
# ===============================
def create_modified_qwen2audio_encoder(original_encoder, adapter_config):
    audio_dim = original_encoder.config.d_model

    adapter = DepAdapter(
        audio_dim=audio_dim,
        adapter_dim=adapter_config.get("adapter_dim", 512),
        dropout=adapter_config.get("dropout", 0.1),
    )

    original_forward = original_encoder.forward

    def new_forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = original_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            audio_features = outputs.last_hidden_state
        else:
            audio_features = outputs[0]

        adapted_audio_features = adapter(audio_features)

        if return_dict:
            from ...modeling_outputs import BaseModelOutput
            return BaseModelOutput(
                last_hidden_state=adapted_audio_features,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return (adapted_audio_features,) + outputs[1:]

    original_encoder.forward = new_forward.__get__(original_encoder, type(original_encoder))
    original_encoder.audio_adapter = adapter

    return original_encoder

# ===============================
# Deepspeed 训练函数
# ===============================
def train_deepspeed(cfg):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f"{cfg.env.device_type}:{local_rank}"

    # 初始化
    set_seed(cfg.train.seed)
    setup_deepspeed(cfg.env.device_type)
    dist.barrier()

    logger = None
    if local_rank == 0:
        os.makedirs(cfg.env.save_path, exist_ok=True)
        logger = set_logger(cfg.env.save_path)
    dist.barrier()
    wandb_logger = WandbLogger(
        cfg=cfg,
        save_path=cfg.env.save_path,
        is_main_process=(local_rank == 0),
        logger=logger,
    )

    # ===============================
    # 加载模型与处理器
    # ===============================
    processor = AutoProcessor.from_pretrained(cfg.env.model_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.env.model_path, trust_remote_code=True)

    # 加入 Adapter
    adapter_config = {
        "adapter_dim": cfg.adapter.get("adapter_dim", 512),
        "dropout": cfg.adapter.get("dropout", 0.1),
    }
    model.audio_tower = create_modified_qwen2audio_encoder(model.audio_tower, adapter_config)

    # LoRA 配置
    peft_cfg = dict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    peft_cfg = LoraConfig(**peft_cfg)
    model = get_peft_model(model, peft_cfg)

    # Enable gradient checkpointing to reduce activation memory usage
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 冻结除 LoRA + Adapter 外的参数
    for name, param in model.named_parameters():
        if "lora_" in name or "audio_adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.print_trainable_parameters()
    model.to(device)
    if dist.get_rank() == 0:
        wandb_logger.watch(model)

    # ===============================
    # Deepspeed 初始化
    # ===============================
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        config=cfg.train.deepspeed_config,
        model=model,
        model_parameters=parameters
    )

    # ===============================
    # 数据加载
    # ===============================
    train_dataset = AudioDataset(
        cfg.data.train_data_path,
        cfg.data.train_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.train_scp_filename,
        task_filename=cfg.data.train_task_filename,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )

    eval_dataset = AudioDataset(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.eval_scp_filename,
        task_filename=cfg.data.eval_task_filename,
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        sampler=eval_sampler,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor)
    )

    if dist.get_rank() == 0:
        reset_peak_memory_stats(cfg.env.device_type, device)

    # ===============================
    # 训练循环
    # ===============================
    best_f1 = -math.inf
    global_train_step = 0
    wandb_log_step = max(1, int(cfg.wandb.get("log_step", 10)))

    for epoch in range(cfg.train.train_epoch):
        if dist.get_rank() == 0:
            train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}")
        else:
            train_bar = train_dataloader

        model_engine.train()
        for train_step, batch in enumerate(train_bar):
            global_train_step += 1
            batch.to(device)
            outputs = model_engine(**batch)
            loss = outputs.loss
            acc = compute_acc(outputs["logits"], batch["labels"])

            if dist.get_rank() == 0:
                train_bar.set_description(f"[Train] epoch:{epoch} rank:{local_rank}, loss:{loss:.2f}, acc:{acc:.2f}")

            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()

            if dist.get_rank() == 0 and global_train_step % wandb_log_step == 0:
                lr_value = model_engine.get_lr()[0] if hasattr(model_engine, "get_lr") else optimizer.param_groups[0]["lr"]
                train_memory_metrics = collect_memory_metrics(cfg.env.device_type, device)
                wandb_logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/acc": acc.item(),
                        "train/lr": lr_value,
                        "train/epoch": epoch,
                        "train/global_step": global_train_step,
                        **prefix_metrics(train_memory_metrics, "train"),
                    },
                    step=global_train_step,
                )
                logger.info(
                    f"[Train][Memory] epoch={epoch} step={global_train_step} | "
                    f"{format_memory_metrics(train_memory_metrics)}"
                )

            # ===============================
            # 评估与保存
            # ===============================
            if (train_step + 1) % cfg.train.eval_step == 0:
                eval_loss = 0.0
                eval_accuracy = 0.0
                eval_precision = 0.0
                eval_recall = 0.0
                eval_f1 = 0.0
                eval_steps = 0

                if dist.get_rank() == 0:
                    eval_bar = tqdm(eval_dataloader, desc="[Eval]")
                else:
                    eval_bar = eval_dataloader

                model_engine.eval()
                with torch.no_grad():
                    for _, batch in enumerate(eval_bar):
                        batch.to(device)
                        outputs = model_engine(**batch)
                        loss = outputs.loss
                        accuracy, precision, recall, f1 = compute_metrics(outputs["logits"], batch["labels"])

                        eval_loss += loss.item()
                        eval_accuracy += accuracy.item()
                        eval_precision += precision.item()
                        eval_recall += recall.item()
                        eval_f1 += f1.item()
                        eval_steps += 1

                # 平均值 & 多进程同步
                for val in ["loss", "accuracy", "precision", "recall", "f1"]:
                    locals()[f"eval_{val}"] /= eval_steps
                    tensor = torch.tensor(locals()[f"eval_{val}"]).to(device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    locals()[f"eval_{val}"] = tensor.item() / world_size

                if dist.get_rank() == 0:
                    eval_memory_metrics = collect_memory_metrics(cfg.env.device_type, device)
                    logger.info(f"[Epoch {epoch} Step {train_step}] Eval Metrics:")
                    logger.info(
                        f"  Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.4f}, "
                        f"Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}"
                    )
                    logger.info(f"[Eval][Memory] {format_memory_metrics(eval_memory_metrics)}")

                    is_best = eval_f1 > best_f1
                    if is_best:
                        save_time = time.strftime("%H-%M", time.localtime())
                        save_path = f"{cfg.env.save_path}/{save_time}"
                        os.makedirs(save_path, exist_ok=True)
                        logger.info(f"[Saving] Better F1 {eval_f1:.4f} > {best_f1:.4f}: {save_path}")
                        best_f1 = eval_f1
                        model_engine.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
                        wandb_logger.update_summary(
                            {
                                "best_eval_f1": best_f1,
                                "best_model_path": save_path,
                            }
                        )

                    wandb_logger.log(
                        {
                            "eval/loss": eval_loss,
                            "eval/acc": eval_accuracy,
                            "eval/precision": eval_precision,
                            "eval/recall": eval_recall,
                            "eval/f1": eval_f1,
                            "eval/best_f1": best_f1,
                            "eval/is_best": is_best,
                            "train/epoch": epoch,
                            **prefix_metrics(eval_memory_metrics, "eval"),
                        },
                        step=global_train_step,
                    )

    if dist.get_rank() == 0 and math.isfinite(best_f1):
        wandb_logger.update_summary({"best_eval_f1": best_f1})
    dist.barrier()
    wandb_logger.finish()
