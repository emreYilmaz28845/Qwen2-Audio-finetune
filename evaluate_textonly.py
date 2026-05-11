"""
Per-dataset evaluation script for the text-only Qwen2-7B model.

Loads an optional LoRA checkpoint (trained with train_textonly.py) and evaluates
on the merged val set, reporting separate metrics for DAIC-WOZ, EATD, CMDC.

Usage:
    python evaluate_textonly.py \\
        --model_path /path/to/Qwen2-7B-Instruct \\
        --peft_path  output_model/<run>/best \\
        --data_path  data/merged/val \\
        --prompt_path data/merged/val/merged_multiprompt_textonly.jsonl

    python evaluate_textonly.py \\
        --model_path /path/to/Qwen2-7B-Instruct \\
        --data_path  data/merged/val \\
        --prompt_path data/merged/val/merged_multiprompt_textonly.jsonl
"""

import argparse
import json
import os
import copy
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils.functions import (
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)


# ===============================
# Text-only dataset with metadata
# ===============================
class TextOnlyDatasetWithMeta(torch.utils.data.Dataset):
    """Loads text prompts + targets + dataset name for per-dataset eval."""

    def __init__(self, data_path, prompt_path, task_filename="merged_multitask.jsonl"):
        self.tasks = []
        self.prompt = {}

        task_path = os.path.join(data_path, task_filename)
        with open(task_path, encoding="utf-8") as f:
            for line in f:
                self.tasks.append(json.loads(line))

        with open(prompt_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.prompt[item["task"]] = item["prompt"]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        target = self.tasks[idx]["target"]
        prompt = self.prompt[self.tasks[idx]["task"]]
        dataset_name = self.tasks[idx].get("dataset", "unknown")
        return {"prompt": prompt, "target": target, "dataset_name": dataset_name}


def collate_fn_textonly_with_meta(samples, tokenizer):
    """Collate for text-only eval, also returns dataset_names."""
    dataset_names = [s.pop("dataset_name") for s in samples]
    prompts = [s["prompt"] for s in samples]
    targets = [s["target"] for s in samples]

    full_texts = [p + t for p, t in zip(prompts, targets)]
    processed_data = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    labels = copy.deepcopy(processed_data["input_ids"])
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True)

    for i, attention_mask in enumerate(prompt_tokens["attention_mask"]):
        prompt_len = attention_mask.sum().item()
        pad_count = (processed_data["input_ids"][i] == tokenizer.pad_token_id).sum().item()
        labels[i, : prompt_len + pad_count] = -100

    processed_data["labels"] = labels
    processed_data["dataset_names"] = dataset_names
    return processed_data


# ===============================
# Processor-like wrapper
# ===============================
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


# ===============================
# Per-dataset stats
# ===============================
def accumulate_stats_per_dataset(processor_compat, logits, labels, dataset_names,
                                  per_dataset_stats, overall_stats):
    compute_metrics_text_binary_accumulate(processor_compat, logits, labels, overall_stats)

    B = labels.size(0)
    for b in range(B):
        ds_name = dataset_names[b]
        if ds_name not in per_dataset_stats:
            per_dataset_stats[ds_name] = {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0
            }
        single_logits = logits[b:b+1]
        single_labels = labels[b:b+1]
        compute_metrics_text_binary_accumulate(
            processor_compat, single_logits, single_labels, per_dataset_stats[ds_name]
        )


def format_metrics(stats):
    if not stats or stats["total"] == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "weighted_f1": 0}
    accuracy, precision, recall, f1, weighted_f1 = compute_metrics_from_stats(stats)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "weighted_f1": weighted_f1,
    }


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset evaluation of text-only Qwen2-7B depression detection model"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Qwen2-7B-Instruct model")
    parser.add_argument("--peft_path", type=str, default="",
                        help="Optional path to saved LoRA adapter checkpoint")
    parser.add_argument("--data_path", type=str, default="data/merged/val")
    parser.add_argument("--prompt_path", type=str,
                        default="data/merged/val/merged_multiprompt_textonly.jsonl")
    parser.add_argument("--task_filename", type=str, default="merged_multitask.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    device = args.device
    peft_path = (args.peft_path or "").strip()
    use_peft = peft_path.lower() not in {"", "none", "null", "base", "baseline"}
    print(f"[Config] model_path: {args.model_path}")
    print(f"[Config] peft_path:  {peft_path if use_peft else '(none - base model)'}")
    print(f"[Config] data_path:  {args.data_path}")
    print(f"[Config] device:     {device}")
    print(f"[Config] mode:       TEXT-ONLY (Qwen2-7B)")

    # ===============================
    # Load model + tokenizer
    # ===============================
    print("\n[1/4] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    if use_peft:
        print("[2/4] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        print("[2/4] Using base model without LoRA adapter...")
        model = base_model
    model.eval()
    model.to(device)

    processor_compat = TokenizerWrapper(tokenizer)

    # ===============================
    # Load data
    # ===============================
    print("[3/4] Loading evaluation dataset...")
    eval_dataset = TextOnlyDatasetWithMeta(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        task_filename=args.task_filename,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial(collate_fn_textonly_with_meta, tokenizer=tokenizer),
        shuffle=False,
    )
    print(f"   Total samples: {len(eval_dataset)}")

    # ===============================
    # Run evaluation
    # ===============================
    print("[4/4] Running evaluation...\n")
    per_dataset_stats = {}
    overall_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0}
    eval_loss_sum = 0.0
    eval_steps = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="[Eval]"):
            dataset_names = batch.pop("dataset_names")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            eval_loss_sum += loss.item()
            eval_steps += 1

            accumulate_stats_per_dataset(
                processor_compat, outputs.logits, batch["labels"],
                dataset_names, per_dataset_stats, overall_stats,
            )

    # ===============================
    # Print results
    # ===============================
    avg_loss = eval_loss_sum / eval_steps if eval_steps > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"  PER-DATASET EVALUATION RESULTS (TEXT-ONLY Qwen2-7B)    (avg loss: {avg_loss:.4f})")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Samples':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'wF1':>8}")
    print("-" * 90)

    results = {
        "_meta": {
            "model_path": args.model_path,
            "peft_path": peft_path if use_peft else "",
            "data_path": args.data_path,
            "prompt_path": args.prompt_path,
            "task_filename": args.task_filename,
            "batch_size": args.batch_size,
            "device": args.device,
            "avg_loss": avg_loss,
            "used_peft": use_peft,
        }
    }
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        metrics = format_metrics(stats)
        results[ds_name] = {**metrics, **stats}
        print(f"{ds_name:<15} {stats['total']:>8} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['weighted_f1']:>8.4f}")

    overall_metrics = format_metrics(overall_stats)
    results["overall"] = {**overall_metrics, **overall_stats}
    print("-" * 90)
    print(f"{'OVERALL':<15} {overall_stats['total']:>8} {overall_metrics['accuracy']:>8.4f} "
          f"{overall_metrics['precision']:>8.4f} {overall_metrics['recall']:>8.4f} "
          f"{overall_metrics['f1']:>8.4f} {overall_metrics['weighted_f1']:>8.4f}")
    print("=" * 90)

    # Confusion matrix
    print("\nConfusion Matrix Details:")
    print(f"{'Dataset':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 50)
    for ds_name in sorted(per_dataset_stats.keys()):
        s = per_dataset_stats[ds_name]
        print(f"{ds_name:<15} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} {s['tn']:>6}")
    print("-" * 50)
    print(f"{'OVERALL':<15} {overall_stats['tp']:>6} {overall_stats['fp']:>6} "
          f"{overall_stats['fn']:>6} {overall_stats['tn']:>6}")

    # Save JSON
    output_json = (args.output_json or "").strip()
    if not output_json:
        output_json = (
            os.path.join(peft_path, "per_dataset_eval_textonly.json")
            if use_peft
            else os.path.abspath("base_per_dataset_eval_textonly.json")
        )
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
