"""
Multi-GPU (DDP) Optuna training with selectable input mode.

Supported modes:
- textonly: original tokenizer-only training path
- audiotext: Qwen2-Audio training path that consumes both audio and text prompts
"""

import copy
import json
import math
import os
import shutil
from dataclasses import asdict
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2AudioForConditionalGeneration,
)

from src.dataset import AudioDataset, collate_fn_qwen2audio
from utils.functions import (
    compute_acc_text,
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)
from utils.init_process import setup_ddp
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.system_metrics import reset_peak_memory_stats
from utils.wandb_logger import WandbLogger


INPUT_MODE_TEXTONLY = "textonly"
INPUT_MODE_AUDIOTEXT = "audiotext"
SUPPORTED_INPUT_MODES = {INPUT_MODE_TEXTONLY, INPUT_MODE_AUDIOTEXT}


class TextOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset that loads text prompts and targets only.
    No audio files are loaded. Reads from:
    - task JSONL: contains 'key', 'target', 'task' fields
    - prompt JSONL: contains 'task', 'prompt' fields
    """

    def __init__(self, data_path, prompt_path, task_filename="merged_multitask.jsonl"):
        self.tasks = []
        self.prompt = {}

        task_path = os.path.join(data_path, task_filename)
        with open(task_path, encoding="utf-8") as handle:
            for line in handle:
                self.tasks.append(json.loads(line))

        with open(prompt_path, encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                self.prompt[item["task"]] = item["prompt"]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        target = self.tasks[idx]["target"]
        prompt = self.prompt[self.tasks[idx]["task"]]
        return {"prompt": prompt, "target": target}


def collate_fn_textonly(samples, tokenizer):
    """Collate function for text-only training. Tokenizes prompt+target and masks prompt in labels."""
    prompts = [s["prompt"] for s in samples]
    targets = [s["target"] for s in samples]

    # Tokenize full input (prompt + target)
    full_texts = [p + t for p, t in zip(prompts, targets)]
    processed_data = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    # Create labels: mask prompt portion with -100
    labels = copy.deepcopy(processed_data["input_ids"])

    # Tokenize each prompt individually (no padding) to get exact prompt length
    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=2048)
        prompt_len = len(prompt_tokens["input_ids"])
        labels[i, :prompt_len] = -100

    # Separately mask PAD tokens wherever they appear (right-padding)
    labels[labels == tokenizer.pad_token_id] = -100

    processed_data["labels"] = labels
    return processed_data


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


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
        return self.layer_norm(x + residual)


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
        effective_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs = original_forward(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=effective_return_dict,
        )

        if effective_return_dict:
            audio_features = outputs.last_hidden_state
        else:
            audio_features = outputs[0]

        adapted_audio_features = adapter(audio_features)
        if effective_return_dict:
            from transformers.modeling_outputs import BaseModelOutput

            return BaseModelOutput(
                last_hidden_state=adapted_audio_features,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return (adapted_audio_features,) + outputs[1:]

    original_encoder.forward = new_forward.__get__(original_encoder, type(original_encoder))
    original_encoder.audio_adapter = adapter
    return original_encoder


def _setup_run(cfg, trial_name):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = f"{cfg.env.device_type}:{local_rank}"

    set_seed(cfg.train.seed)
    setup_ddp(cfg.env.device_type)
    dist.barrier()

    if trial_name:
        cfg.env.save_path = f"{cfg.env.save_path}_trial_{trial_name}"

    if rank == 0:
        os.makedirs(cfg.env.save_path, exist_ok=True)
        train_log_path = os.path.join(cfg.env.save_path, "train_log")
        if os.path.isdir(train_log_path):
            shutil.rmtree(train_log_path, ignore_errors=True)
    dist.barrier()

    logger = set_logger(cfg.env.save_path)
    wandb_enabled = cfg.wandb.enabled and False
    wandb_logger = None
    if wandb_enabled and rank == 0:
        wandb_logger = WandbLogger(
            cfg=cfg,
            save_path=cfg.env.save_path,
            is_main_process=True,
            logger=logger,
        )

    return {
        "local_rank": local_rank,
        "world_size": world_size,
        "rank": rank,
        "device": device,
        "logger": logger,
        "wandb_logger": wandb_logger,
    }


def _build_scheduler(optim, cfg):
    return lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: (
            min(step / cfg.train.warmup_steps, 1)
            if step < cfg.train.warmup_steps
            else max(
                0.0,
                1 - (step - cfg.train.warmup_steps) / (cfg.train.total_train_steps - cfg.train.warmup_steps),
            )
        ),
    )


def _log_trial_header(logger, trial_name, cfg, world_size, input_mode):
    logger.info("%s", "=" * 60)
    logger.info("Trial: %s", trial_name)
    logger.info("Input Mode: %s", input_mode)
    logger.info("Learning Rate: %s", cfg.train.lr)
    logger.info("Batch Size: %s", cfg.train.batch_size)
    logger.info("LoRA R: %s", cfg.peft.r)
    logger.info("LoRA Alpha: %s", cfg.peft.lora_alpha)
    logger.info("World Size: %s", world_size)
    logger.info("%s", "=" * 60)


def _evaluate(model, eval_dataloader, device, metric_processor, rank):
    eval_loss = 0.0
    eval_steps = 0
    global_stats = None
    eval_bar = tqdm(eval_dataloader, desc="[Eval]") if rank == 0 else eval_dataloader

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_bar):
            if hasattr(batch, "to"):
                batch = batch.to(device)
            else:
                batch = {key: value.to(device) for key, value in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                global_stats = compute_metrics_text_binary_accumulate(
                    metric_processor, outputs.logits, batch["labels"], global_stats
                )

            eval_loss += loss.item()
            eval_steps += 1
            if rank == 0 and global_stats and global_stats["total"] > 0:
                temp_acc, _, _, temp_f1, temp_wf1 = compute_metrics_from_stats(global_stats)
                eval_bar.set_description(
                    f"[Eval] loss {loss:.3f} | acc {temp_acc:.4f} | posF1 {temp_f1:.4f} | wF1 {temp_wf1:.4f}"
                )

    loss_tensor = torch.tensor([eval_loss, float(eval_steps)], device=device, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    reduced_eval_loss = (loss_tensor[0] / loss_tensor[1]).item() if loss_tensor[1] > 0 else 0.0

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
    return reduced_eval_loss, reduced_stats


def _finalize_run(logger, trial_name, best_f1, wandb_logger):
    if not math.isfinite(best_f1):
        best_f1 = -1.0
    dist.barrier()
    if logger is not None:
        logger.info("\n%s", "=" * 60)
        logger.info("Trial Complete: %s", trial_name)
        logger.info("Best F1 Score: %.4f", best_f1)
        logger.info("%s\n", "=" * 60)
    if wandb_logger:
        wandb_logger.finish()
    dist.barrier()
    return best_f1


def train_textonly_ddp(cfg, trial_name=""):
    state = _setup_run(cfg, trial_name)
    local_rank = state["local_rank"]
    world_size = state["world_size"]
    rank = state["rank"]
    device = state["device"]
    logger = state["logger"]
    wandb_logger = state["wandb_logger"]

    if rank == 0:
        logger.info("Loading text model from %s", cfg.env.model_path)
        _log_trial_header(logger, trial_name, cfg, world_size, INPUT_MODE_TEXTONLY)

    tokenizer = AutoTokenizer.from_pretrained(cfg.env.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = {"trust_remote_code": True}
    if cfg.train.use_bfloat16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
        model_load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(cfg.env.model_path, **model_load_kwargs)

    peft_cfg = asdict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    model = get_peft_model(model, LoraConfig(**peft_cfg))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    model.to(device)
    if rank == 0:
        model.print_trainable_parameters()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    scheduler = _build_scheduler(optim, cfg)

    train_dataset = TextOnlyDataset(
        cfg.data.train_data_path,
        cfg.data.train_prompt_path,
        task_filename=cfg.data.train_task_filename,
    )
    eval_dataset = TextOnlyDataset(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        task_filename=cfg.data.eval_task_filename,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.train.seed,
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=cfg.train.seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_textonly, tokenizer=tokenizer),
        sampler=train_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_textonly, tokenizer=tokenizer),
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    if rank == 0:
        reset_peak_memory_stats(cfg.env.device_type, device)

    metric_processor = TokenizerWrapper(tokenizer)
    steps_per_epoch = len(train_dataloader)
    dynamic_eval_step = max(1, steps_per_epoch // 10)

    if rank == 0:
        logger.info("\n[Dynamic Eval Settings]")
        logger.info("  Total training samples: %s", len(train_dataset))
        logger.info("  Batch size per GPU: %s", cfg.train.batch_size)
        logger.info("  World size (GPUs): %s", world_size)
        logger.info("  Effective batch size: %s", cfg.train.batch_size * world_size)
        logger.info("  Steps per epoch: %s", steps_per_epoch)
        logger.info("  Dynamic eval_step: %s (evaluate ~10x per epoch)", dynamic_eval_step)
        logger.info("")

    best_f1 = -math.inf
    global_train_step = 0
    optimizer_step = 0

    for epoch in range(cfg.train.train_epoch):
        train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}") if rank == 0 else train_dataloader
        model.train()
        optim.zero_grad()
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            global_train_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(metric_processor, outputs.logits, batch["labels"])

            (loss / grad_acc_steps).backward()
            should_step = ((train_step + 1) % grad_acc_steps == 0) or ((train_step + 1) == len(train_dataloader))
            if should_step:
                optim.step()
                scheduler.step()
                optim.zero_grad()
                optimizer_step += 1

            if rank == 0:
                train_bar.set_description(
                    f"[Train] epoch:{epoch}, loss:{loss:.2f}, acc:{acc:.2f}, lr:{scheduler.get_last_lr()[0]:.2e}"
                )

            if (train_step + 1) % dynamic_eval_step == 0:
                eval_loss, reduced_stats = _evaluate(
                    model, eval_dataloader, device, metric_processor, rank
                )
                if rank == 0:
                    eval_accuracy, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(
                        reduced_stats
                    )
                    logger.info("[Epoch %s Step %s] Eval Metrics:", epoch, train_step)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        eval_accuracy,
                        eval_precision,
                        eval_recall,
                        eval_f1,
                        eval_wf1,
                    )

                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        logger.info("[New Best F1] %.4f", eval_f1)
                        best_model_path = os.path.join(cfg.env.save_path, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        model.module.save_pretrained(best_model_path)
                        tokenizer.save_pretrained(best_model_path)
                        logger.info("[Saved Best Model] -> %s", best_model_path)

                        if wandb_logger:
                            wandb_logger.log(
                                {
                                    "eval/loss": eval_loss,
                                    "eval/f1": eval_f1,
                                    "eval/best_f1": best_f1,
                                    "train/epoch": epoch,
                                },
                                step=global_train_step,
                            )
                model.train()

    return _finalize_run(logger if rank == 0 else None, trial_name, best_f1, wandb_logger)


def train_audiotext_ddp(cfg, trial_name=""):
    state = _setup_run(cfg, trial_name)
    local_rank = state["local_rank"]
    world_size = state["world_size"]
    rank = state["rank"]
    device = state["device"]
    logger = state["logger"]
    wandb_logger = state["wandb_logger"]

    if rank == 0:
        logger.info("Loading Qwen2-Audio model from %s", cfg.env.model_path)
        _log_trial_header(logger, trial_name, cfg, world_size, INPUT_MODE_AUDIOTEXT)

    processor = AutoProcessor.from_pretrained(cfg.env.model_path, trust_remote_code=True)
    adapter_config = {
        "adapter_dim": getattr(cfg.adapter, "adapter_dim", 32),
        "dropout": getattr(cfg.adapter, "dropout", 0.1),
    }

    model_load_kwargs = {"trust_remote_code": True}
    if cfg.train.use_bfloat16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
        model_load_kwargs["low_cpu_mem_usage"] = True
    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.env.model_path, **model_load_kwargs)
    model.audio_tower = create_modified_qwen2audio_encoder(model.audio_tower, adapter_config)

    peft_cfg = asdict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    model = get_peft_model(model, LoraConfig(**peft_cfg))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name) or ("audio_adapter" in name)

    model.to(device)
    if rank == 0:
        model.print_trainable_parameters()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    scheduler = _build_scheduler(optim, cfg)

    train_dataset = AudioDataset(
        cfg.data.train_data_path,
        cfg.data.train_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.train_scp_filename,
        task_filename=cfg.data.train_task_filename,
    )
    eval_dataset = AudioDataset(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.eval_scp_filename,
        task_filename=cfg.data.eval_task_filename,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.train.seed,
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=cfg.train.seed,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor),
        sampler=train_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_qwen2audio, processor=processor),
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    if rank == 0:
        reset_peak_memory_stats(cfg.env.device_type, device)

    steps_per_epoch = len(train_dataloader)
    dynamic_eval_step = max(1, steps_per_epoch // 10)

    if rank == 0:
        logger.info("\n[Dynamic Eval Settings]")
        logger.info("  Total training samples: %s", len(train_dataset))
        logger.info("  Batch size per GPU: %s", cfg.train.batch_size)
        logger.info("  World size (GPUs): %s", world_size)
        logger.info("  Effective batch size: %s", cfg.train.batch_size * world_size)
        logger.info("  Steps per epoch: %s", steps_per_epoch)
        logger.info("  Dynamic eval_step: %s (evaluate ~10x per epoch)", dynamic_eval_step)
        logger.info("")

    best_f1 = -math.inf
    global_train_step = 0

    for epoch in range(cfg.train.train_epoch):
        train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}") if rank == 0 else train_dataloader
        model.train()
        optim.zero_grad(set_to_none=True)
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            global_train_step += 1
            batch = batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(processor, outputs.logits, batch["labels"])

            (loss / grad_acc_steps).backward()
            should_step = ((train_step + 1) % grad_acc_steps == 0) or ((train_step + 1) == len(train_dataloader))
            if should_step:
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if rank == 0:
                train_bar.set_description(
                    f"[Train] epoch:{epoch}, loss:{loss:.2f}, acc:{acc:.2f}, lr:{scheduler.get_last_lr()[0]:.2e}"
                )

            if (train_step + 1) % dynamic_eval_step == 0:
                eval_loss, reduced_stats = _evaluate(model, eval_dataloader, device, processor, rank)
                if rank == 0:
                    eval_accuracy, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(
                        reduced_stats
                    )
                    logger.info("[Epoch %s Step %s] Eval Metrics:", epoch, train_step)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        eval_accuracy,
                        eval_precision,
                        eval_recall,
                        eval_f1,
                        eval_wf1,
                    )

                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        logger.info("[New Best F1] %.4f", eval_f1)
                        best_model_path = os.path.join(cfg.env.save_path, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        model.module.save_pretrained(best_model_path)
                        processor.save_pretrained(best_model_path)
                        logger.info("[Saved Best Model] -> %s", best_model_path)

                        if wandb_logger:
                            wandb_logger.log(
                                {
                                    "eval/loss": eval_loss,
                                    "eval/f1": eval_f1,
                                    "eval/best_f1": best_f1,
                                    "train/epoch": epoch,
                                },
                                step=global_train_step,
                            )
                model.train()

    return _finalize_run(logger if rank == 0 else None, trial_name, best_f1, wandb_logger)


def train_ddp(cfg, trial_name="", input_mode=INPUT_MODE_TEXTONLY):
    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode={input_mode!r}. Expected one of {sorted(SUPPORTED_INPUT_MODES)}."
        )
    if input_mode == INPUT_MODE_AUDIOTEXT:
        return train_audiotext_ddp(cfg, trial_name=trial_name)
    return train_textonly_ddp(cfg, trial_name=trial_name)
