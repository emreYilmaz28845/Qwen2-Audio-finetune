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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_ddp
from utils.functions import (
    compute_acc_text,
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)
import torch.nn as nn


# ===============================
# Adapter module definition
# ===============================

# CHECKPOINT
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
# Modify the Qwen2Audio encoder to add an Adapter
# ===============================
#This function takes Qwen2-Audio’s existing audio encoder and rewires
# it so every forward pass goes through the new DepAdapter before the encoder output is returned.
def create_modified_qwen2audio_encoder(original_encoder, adapter_config):
    audio_dim = original_encoder.config.d_model #taking the audio
    # feature dimension from the original encoder config


    #adapter== creates the small bottleneck adapter that will post-process the audio features.
    adapter = DepAdapter(
        audio_dim=audio_dim,
        adapter_dim=adapter_config.get("adapter_dim", 512),
        dropout=adapter_config.get("dropout", 0.1),
    )

    original_forward = original_encoder.forward

    # It defines a replacement forward method.
    # Inside that method =
    # it first calls the original forward
    # extracts the audio features from the output (last_hidden_state)
    # then passes those features through the adapter.
    # returns the adapted features in the same format as the original output, ensuring compatibility with the rest of the model.


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
        # if the encoder returned a model output object, get the audio features by name
        # otherwise, get the first tuple item, which is the same tensor

        if return_dict:
            audio_features = outputs.last_hidden_state
        else:
            audio_features = outputs[0]

        adapted_audio_features = adapter(audio_features)

        if return_dict:
            from ...modeling_outputs import BaseModelOutput
            return BaseModelOutput(
                last_hidden_state=adapted_audio_features,# replace the original audio features with the adapted ones
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            # the original encoder output is expected to be a tuple
            # the first item in that tuple is the main hidden state
            # so he returns a new tuple where:
            # first element = adapted_audio_features
            # remaining elements = original outputs[1:]
            return (adapted_audio_features,) + outputs[1:]

    # original_encoder.forward = bound_version_of(new_forward) (conceptually doing this)
    original_encoder.forward = new_forward.__get__(original_encoder, type(original_encoder))
    original_encoder.audio_adapter = adapter

    return original_encoder


# ===============================
# Main training function
# ===============================
def train_ddp(cfg):
    # in ddp, each process is responsible for one GPU, so local_rank indicates which GPU this process should use.
    local_rank = int(os.environ["LOCAL_RANK"])
    # world_size is the total number of processes (GPUs) participating in the training,
    # which is used for averaging metrics across all processes.
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"{cfg.env.device_type}:{local_rank}"

    # Initialize DDP
    set_seed(cfg.train.seed)
    setup_ddp(cfg.env.device_type)
    # Barrier to ensure all processes have initialized before proceeding (important for synchronized logging and saving)
    dist.barrier()

    # Only the process with local_rank 0 will create the output directory and logger to avoid conflicts.
    if local_rank == 0:
        os.makedirs(cfg.env.save_path, exist_ok=True)
    dist.barrier()

    logger = set_logger(cfg.env.save_path)

    # ===============================
    # Load model and processor
    # ===============================
    processor = AutoProcessor.from_pretrained(cfg.env.model_path, trust_remote_code=True)

    adapter_config = {
        "adapter_dim": cfg.adapter.get("adapter_dim", 32),#before it was 512, but we set it to 32 in the config, so we use that value here. If not specified, it defaults to 32.
        "dropout": cfg.adapter.get("dropout", 0.1),
    }

    model_load_kwargs = {
        "trust_remote_code": True,
    }
    if cfg.train.use_bfloat16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
        model_load_kwargs["low_cpu_mem_usage"] = True

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        cfg.env.model_path, **model_load_kwargs
    )

    # Modify the audio encoder to add an Adapter
    model.audio_tower = create_modified_qwen2audio_encoder(model.audio_tower, adapter_config)
    print(model)
    # ===============================
    # LoRA configuration and application
    # ===============================
    peft_cfg = dict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    peft_cfg = LoraConfig(**peft_cfg)

    model = get_peft_model(model, peft_cfg)

    # ===============================
    # Enable gradient checkpointing to reduce activation memory
    # ===============================
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ===============================
    # Freeze all parameters except LoRA + Adapter
    # ===============================
    for name, param in model.named_parameters():
        if "lora_" in name or "audio_adapter" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.print_trainable_parameters()
    model.to(device)
    if dist.get_rank() == 0:
        model.print_trainable_parameters()

    # ===============================
    # Wrap with DDP
    # ===============================
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ===============================
    # Optimizer and scheduler
    # ===============================
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)

    scheduler = lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: (
            min(step / cfg.train.warmup_steps, 1)
            if step < cfg.train.warmup_steps
            else max(0.0, 1 - (step - cfg.train.warmup_steps) / (cfg.train.total_train_steps - cfg.train.warmup_steps))
        ),
    )

    # ===============================
    # Data loading
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
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor),
        sampler=train_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
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
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor),
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    # ===============================
    # Training loop
    # ===============================
    best_f1 = -math.inf

    for epoch in range(cfg.train.train_epoch):
        if dist.get_rank() == 0:
            train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}")
        else:
            train_bar = train_dataloader

        model.train()
        optim.zero_grad()
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            #acc = compute_acc(outputs["logits"], batch["labels"])
            acc = compute_acc_text(processor, outputs.logits, batch["labels"])

            loss_for_backward = loss / grad_acc_steps
            loss_for_backward.backward()

            should_step = ((train_step + 1) % grad_acc_steps == 0) or ((train_step + 1) == len(train_dataloader))
            if should_step:
                optim.step()
                scheduler.step()
                optim.zero_grad()

            if dist.get_rank() == 0:
                train_bar.set_description(
                    f"[Train] epoch:{epoch} rank:{local_rank}, loss:{loss:.2f}, acc:{acc:.2f}, ga:{grad_acc_steps}"
                )

            # ===============================
            # Evaluation and saving
            # ===============================
            if (train_step + 1) % cfg.train.eval_step == 0:
                eval_loss = 0.0
                eval_steps = 0
                global_stats = None

                if dist.get_rank() == 0:
                    eval_bar = tqdm(eval_dataloader, desc="[Eval]")
                else:
                    eval_bar = eval_dataloader

                model.eval()
                with torch.no_grad():
                    for _, batch in enumerate(eval_bar):
                        batch.to(device)
                        outputs = model(**batch)
                        loss = outputs.loss
                        global_stats = compute_metrics_text_binary_accumulate(
                            processor, outputs.logits, batch["labels"], global_stats
                        )

                        eval_loss += loss.item()
                        eval_steps += 1
                        if dist.get_rank() == 0 and global_stats and global_stats["total"] > 0:
                            temp_acc, _, _, temp_f1, temp_wf1 = compute_metrics_from_stats(global_stats)
                            eval_bar.set_description(
                                f"[Eval] loss {loss:.3f} | acc {temp_acc:.4f} | posF1 {temp_f1:.4f} | wF1 {temp_wf1:.4f}"
                            )

                loss_tensor = torch.tensor(
                    [eval_loss, float(eval_steps)],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                eval_loss = (loss_tensor[0] / loss_tensor[1]).item() if loss_tensor[1] > 0 else 0.0

                stats_tensor = torch.tensor(
                    [
                        float(global_stats["tp"]) if global_stats else 0.0,
                        float(global_stats["fp"]) if global_stats else 0.0,
                        float(global_stats["fn"]) if global_stats else 0.0,
                        float(global_stats["tn"]) if global_stats else 0.0,
                        float(global_stats["total"]) if global_stats else 0.0,
                        float(global_stats["correct"]) if global_stats else 0.0,
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                reduced_stats = {
                    "tp": stats_tensor[0].item(),
                    "fp": stats_tensor[1].item(),
                    "fn": stats_tensor[2].item(),
                    "tn": stats_tensor[3].item(),
                    "total": stats_tensor[4].item(),
                    "correct": stats_tensor[5].item(),
                }
                eval_accuracy, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(
                    reduced_stats
                )

                if dist.get_rank() == 0:
                    logger.info(f"[Epoch {epoch} Step {train_step}] Eval Metrics:")
                    logger.info(
                        f"  Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.4f}, "
                        f"Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}, wF1: {eval_wf1:.4f}"
                    )

                    if eval_f1 > best_f1:
                        save_time = time.strftime("%H-%M", time.localtime())
                        save_path = f"{cfg.env.save_path}/{save_time}"
                        os.makedirs(save_path, exist_ok=True)
                        logger.info(f"[Saving] Better F1 {eval_f1:.4f} > {best_f1:.4f}: {save_path}")
                        best_f1 = eval_f1
                        model.module.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
