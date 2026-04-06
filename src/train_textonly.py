"""
Text-only training script for depression detection using Qwen2-7B.

This script is a simplified version of train_ddp.py that:
- Uses AutoModelForCausalLM + AutoTokenizer (not Qwen2Audio)
- Loads only text prompts (no audio files)
- Applies LoRA directly to the model (no DepAdapter)

This replicates the paper's "Text (Qwen2-7B)" ablation experiment.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import time
import torch.distributed as dist
import os
import math
import copy
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.optim import lr_scheduler
from utils.set_logger import set_logger
from utils.set_seed import set_seed
from utils.init_process import setup_ddp
from utils.system_metrics import (
    collect_memory_metrics,
    format_memory_metrics,
    prefix_metrics,
    reset_peak_memory_stats,
)
from utils.wandb_logger import WandbLogger
from utils.functions import (
    compute_acc_text,
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)


# ===============================
# Text-only dataset (no audio loading)
# ===============================
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

        # Read task file
        task_path = os.path.join(data_path, task_filename)
        with open(task_path, encoding="utf-8") as f:
            for line in f:
                self.tasks.append(json.loads(line))

        # Read prompt file
        with open(prompt_path, encoding="utf-8") as f:
            for line in f:
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
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)

    for i, attention_mask in enumerate(prompt_tokens["attention_mask"]):
        # Number of real prompt tokens + number of padding tokens in full sequence
        prompt_len = attention_mask.sum().item()
        pad_count = (processed_data["input_ids"][i] == tokenizer.pad_token_id).sum().item()
        labels[i, : prompt_len + pad_count] = -100

    processed_data["labels"] = labels
    return processed_data


# ===============================
# Processor-like wrapper for compatibility with metric functions
# ===============================
class TokenizerWrapper:
    """
    Wraps an AutoTokenizer to provide a .tokenizer attribute,
    making it compatible with compute_acc_text() and
    compute_metrics_text_binary_accumulate() which expect processor.tokenizer.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


# ===============================
# Main training function
# ===============================
def train_textonly(cfg):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"{cfg.env.device_type}:{local_rank}"

    # Initialize DDP
    set_seed(cfg.train.seed)
    setup_ddp(cfg.env.device_type)
    dist.barrier()

    if local_rank == 0:
        os.makedirs(cfg.env.save_path, exist_ok=True)
    dist.barrier()

    logger = set_logger(cfg.env.save_path)
    wandb_logger = WandbLogger(
        cfg=cfg,
        save_path=cfg.env.save_path,
        is_main_process=(local_rank == 0),
        logger=logger,
    )

    # ===============================
    # Load model and tokenizer
    # ===============================
    logger.info(f"Loading Qwen2-7B model from {cfg.env.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.env.model_path, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = {"trust_remote_code": True}
    if cfg.train.use_bfloat16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
        model_load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        cfg.env.model_path, **model_load_kwargs
    )

    # ===============================
    # LoRA configuration and application
    # ===============================
    peft_cfg = dict(cfg.peft)
    peft_cfg["target_modules"] = list(peft_cfg["target_modules"])
    peft_cfg = LoraConfig(**peft_cfg)

    # Apply LoRA directly to the model (no .language_model unwrapping needed)
    model = get_peft_model(model, peft_cfg)

    # ===============================
    # Enable gradient checkpointing
    # ===============================
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ===============================
    # Freeze all parameters except LoRA
    # ===============================
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.to(device)
    if dist.get_rank() == 0:
        model.print_trainable_parameters()
        wandb_logger.watch(model)

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
    train_dataset = TextOnlyDataset(
        cfg.data.train_data_path,
        cfg.data.train_prompt_path,
        task_filename=cfg.data.train_task_filename,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_textonly, tokenizer=tokenizer),
        sampler=train_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    eval_dataset = TextOnlyDataset(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        task_filename=cfg.data.eval_task_filename,
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=partial(collate_fn_textonly, tokenizer=tokenizer),
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    if dist.get_rank() == 0:
        reset_peak_memory_stats(cfg.env.device_type, device)

    # Wrap tokenizer for metric function compatibility
    processor_compat = TokenizerWrapper(tokenizer)

    # ===============================
    # Training loop
    # ===============================
    best_f1 = -math.inf
    global_train_step = 0
    optimizer_step = 0
    wandb_log_step = max(1, int(cfg.wandb.get("log_step", 10)))

    for epoch in range(cfg.train.train_epoch):
        if dist.get_rank() == 0:
            train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}")
        else:
            train_bar = train_dataloader

        model.train()
        optim.zero_grad()
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            global_train_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(processor_compat, outputs.logits, batch["labels"])

            loss_for_backward = loss / grad_acc_steps
            torch.cuda.empty_cache()
            loss_for_backward.backward()

            should_step = ((train_step + 1) % grad_acc_steps == 0) or ((train_step + 1) == len(train_dataloader))
            if should_step:
                optim.step()
                scheduler.step()
                optim.zero_grad()
                optimizer_step += 1

            if dist.get_rank() == 0:
                train_bar.set_description(
                    f"[Train] epoch:{epoch} rank:{local_rank}, loss:{loss:.2f}, acc:{acc:.2f}, ga:{grad_acc_steps}"
                )
                if global_train_step % wandb_log_step == 0:
                    train_memory_metrics = collect_memory_metrics(cfg.env.device_type, device)
                    wandb_logger.log(
                        {
                            "train/loss": loss.item(),
                            "train/acc": acc.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": global_train_step,
                            "train/optimizer_step": optimizer_step,
                            **prefix_metrics(train_memory_metrics, "train"),
                        },
                        step=global_train_step,
                    )
                    logger.info(
                        f"[Train][Memory] epoch={epoch} step={global_train_step} | "
                        f"{format_memory_metrics(train_memory_metrics)}"
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
                        batch = {k: v.to(device) for k, v in batch.items()}

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model(**batch)
                            loss = outputs.loss
                            global_stats = compute_metrics_text_binary_accumulate(
                                processor_compat, outputs.logits, batch["labels"], global_stats
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
                    eval_memory_metrics = collect_memory_metrics(cfg.env.device_type, device)
                    logger.info(f"[Epoch {epoch} Step {train_step}] Eval Metrics:")
                    logger.info(
                        f"  Loss: {eval_loss:.4f}, Acc: {eval_accuracy:.4f}, "
                        f"Prec: {eval_precision:.4f}, Rec: {eval_recall:.4f}, F1: {eval_f1:.4f}, wF1: {eval_wf1:.4f}"
                    )
                    logger.info(f"[Eval][Memory] {format_memory_metrics(eval_memory_metrics)}")

                    is_best = eval_f1 > best_f1
                    if is_best:
                        best_save_path = f"{cfg.env.save_path}/best"
                        os.makedirs(best_save_path, exist_ok=True)
                        logger.info(f"[Saving Best] F1 {eval_f1:.4f} > {best_f1:.4f} → {best_save_path}")
                        best_f1 = eval_f1
                        model.module.save_pretrained(best_save_path)
                        tokenizer.save_pretrained(best_save_path)
                        wandb_logger.update_summary(
                            {
                                "best_eval_f1": best_f1,
                                "best_model_path": best_save_path,
                            }
                        )

                    wandb_logger.log(
                        {
                            "eval/loss": eval_loss,
                            "eval/acc": eval_accuracy,
                            "eval/precision": eval_precision,
                            "eval/recall": eval_recall,
                            "eval/f1": eval_f1,
                            "eval/weighted_f1": eval_wf1,
                            "eval/best_f1": best_f1,
                            "eval/is_best": is_best,
                            "eval/tp": reduced_stats["tp"],
                            "eval/fp": reduced_stats["fp"],
                            "eval/fn": reduced_stats["fn"],
                            "eval/tn": reduced_stats["tn"],
                            "eval/total": reduced_stats["total"],
                            "train/epoch": epoch,
                            "train/optimizer_step": optimizer_step,
                            **prefix_metrics(eval_memory_metrics, "eval"),
                        },
                        step=global_train_step,
                    )

    # Save final-epoch checkpoint
    if dist.get_rank() == 0:
        final_save_path = f"{cfg.env.save_path}/final"
        os.makedirs(final_save_path, exist_ok=True)
        logger.info(f"[Saving Final] Saving last-epoch checkpoint → {final_save_path}")
        model.module.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)

    if dist.get_rank() == 0 and math.isfinite(best_f1):
        wandb_logger.update_summary({"best_eval_f1": best_f1})
    dist.barrier()
    wandb_logger.finish()
