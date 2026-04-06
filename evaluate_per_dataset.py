"""
Per-dataset evaluation script for the Qwen2-Audio depression detection model.

NOTE: train_ddp.py line 197 does `model = get_peft_model(model.language_model, ...)`
which means training only used the LLM decoder (text-only, no audio pipeline).
This eval script matches that behavior: it loads the LoRA checkpoint onto
the language_model and calls it directly, stripping audio features from batches.

Loads a saved LoRA checkpoint and evaluates on the merged val set,
reporting separate metrics (Accuracy, Precision, Recall, F1, Weighted F1)
for DAIC-WOZ, EATD, CMDC, and overall.

Usage:
    python evaluate_per_dataset.py \\
        --model_path /path/to/Qwen2-Audio-7B-Instruct \\
        --peft_path  output_model/<run>/best \\
        --data_path  data/merged/val \\
        --prompt_path data/merged/val/merged_multiprompt.jsonl
"""

import argparse
import json
import os
import copy
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel

# ===============================
# Reuse existing modules from the project
# ===============================
from src.dataset import AudioDataset, collate_fn_qwen2audio
from utils.functions import (
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)


# ===============================
# Extended dataset: also returns the 'dataset' field per sample
# ===============================
class AudioDatasetWithMeta(AudioDataset):
    """Subclass of AudioDataset that also returns the 'dataset' field from task JSONL."""

    def __getitem__(self, idx):
        # Get base item from parent
        item = super().__getitem__(idx)
        # Add the dataset name from the task record
        dataset_name = self.tasks[idx].get("dataset", "unknown")
        item["dataset_name"] = dataset_name
        return item


def collate_fn_with_meta(samples, processor):
    """Wraps the original collate_fn_qwen2audio and also propagates dataset_names."""
    # Extract dataset_names before passing to original collate
    dataset_names = [s.pop("dataset_name") for s in samples]

    # Use the original collate function
    processed_data = collate_fn_qwen2audio(samples, processor)

    # Attach dataset_names to the batch
    processed_data["dataset_names"] = dataset_names
    return processed_data


# ===============================
# Per-dataset stats accumulation
# ===============================
def accumulate_stats_per_dataset(processor, logits, labels, dataset_names,
                                  per_dataset_stats, overall_stats):
    """
    Accumulate confusion matrix stats per-dataset and overall.
    Uses the same logic as compute_metrics_text_binary_accumulate from utils/functions.py
    but buckets by dataset name.
    """
    # Update overall stats using the existing function
    compute_metrics_text_binary_accumulate(processor, logits, labels, overall_stats)

    # Now update per-dataset stats sample-by-sample
    preds = torch.argmax(logits, dim=-1)
    B = labels.size(0)

    for b in range(B):
        ds_name = dataset_names[b]
        if ds_name not in per_dataset_stats:
            per_dataset_stats[ds_name] = {
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0
            }

        # Process single sample: slice out batch dim and call accumulate
        single_logits = logits[b:b+1]
        single_labels = labels[b:b+1]
        compute_metrics_text_binary_accumulate(
            processor, single_logits, single_labels, per_dataset_stats[ds_name]
        )


def format_metrics(stats):
    """Compute metrics dict from stats using the existing compute_metrics_from_stats."""
    if not stats or stats["total"] == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "weighted_f1": 0}

    accuracy, precision, recall, f1, weighted_f1 = compute_metrics_from_stats(stats)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weighted_f1": weighted_f1,
    }


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset evaluation of Qwen2-Audio depression detection model"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Qwen2-Audio-7B-Instruct model")
    parser.add_argument("--peft_path", type=str, required=True,
                        help="Path to saved LoRA adapter checkpoint (e.g. output_model/<run>/best)")
    parser.add_argument("--data_path", type=str, default="data/merged/val",
                        help="Path to merged val data directory")
    parser.add_argument("--prompt_path", type=str, default="data/merged/val/merged_multiprompt.jsonl",
                        help="Path to merged val prompt JSONL file")
    parser.add_argument("--scp_filename", type=str, default="merged.scp",
                        help="SCP filename inside data_path")
    parser.add_argument("--task_filename", type=str, default="merged_multitask.jsonl",
                        help="Task JSONL filename inside data_path")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for evaluation")
    parser.add_argument("--output_json", type=str, default="",
                        help="Path to save results JSON (default: <peft_path>/per_dataset_eval.json)")
    args = parser.parse_args()

    device = args.device
    print(f"[Config] model_path: {args.model_path}")
    print(f"[Config] peft_path:  {args.peft_path}")
    print(f"[Config] data_path:  {args.data_path}")
    print(f"[Config] device:     {device}")
    print(f"[Config] mode:       TEXT-ONLY (matching train_ddp.py line 197)")

    # ===============================
    # Load model + processor
    # ===============================
    # NOTE: train_ddp.py line 197 does:
    #   model = get_peft_model(model.language_model, peft_cfg)
    # This means training ONLY used the LLM decoder, not the full Qwen2Audio pipeline.
    # Audio features were never processed. We must replicate this here.
    print("\n[1/4] Loading processor and base model...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter onto the language model ONLY (matching training)
    # No DepAdapter, no audio_tower modification — training never used them
    print("[2/4] Loading LoRA adapter onto language_model...")
    model = PeftModel.from_pretrained(full_model.language_model, args.peft_path)
    model.eval()
    model.to(device)
    del full_model  # free memory — we only need the LLM

    # ===============================
    # Load data
    # ===============================
    print("[3/4] Loading evaluation dataset...")
    eval_dataset = AudioDatasetWithMeta(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        wav_type="wav",
        scp_filename=args.scp_filename,
        task_filename=args.task_filename,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial(collate_fn_with_meta, processor=processor),
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

    # Keys that belong to the audio pipeline and must be stripped
    # before calling language_model directly (matching train_ddp.py behavior)
    AUDIO_KEYS = {"input_features", "feature_attention_mask"}

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="[Eval]"):
            dataset_names = batch.pop("dataset_names")  # remove before sending to model

            # Strip audio-specific keys — training never processed audio
            for key in AUDIO_KEYS:
                batch.pop(key, None)

            batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            eval_loss_sum += loss.item()
            eval_steps += 1

            accumulate_stats_per_dataset(
                processor, outputs.logits, batch["labels"],
                dataset_names, per_dataset_stats, overall_stats,
            )

    # ===============================
    # Compute & print results
    # ===============================
    avg_loss = eval_loss_sum / eval_steps if eval_steps > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"  PER-DATASET EVALUATION RESULTS    (avg loss: {avg_loss:.4f})")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Samples':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'wF1':>8}")
    print("-" * 90)

    results = {}
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        metrics = format_metrics(stats)
        results[ds_name] = {**metrics, **stats}
        print(f"{ds_name:<15} {stats['total']:>8} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['weighted_f1']:>8.4f}")

    # Overall
    overall_metrics = format_metrics(overall_stats)
    results["overall"] = {**overall_metrics, **overall_stats}
    print("-" * 90)
    print(f"{'OVERALL':<15} {overall_stats['total']:>8} {overall_metrics['accuracy']:>8.4f} "
          f"{overall_metrics['precision']:>8.4f} {overall_metrics['recall']:>8.4f} "
          f"{overall_metrics['f1']:>8.4f} {overall_metrics['weighted_f1']:>8.4f}")
    print("=" * 90)

    # Print confusion matrix details
    print("\nConfusion Matrix Details:")
    print(f"{'Dataset':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 50)
    for ds_name in sorted(per_dataset_stats.keys()):
        s = per_dataset_stats[ds_name]
        print(f"{ds_name:<15} {s['tp']:>6} {s['fp']:>6} {s['fn']:>6} {s['tn']:>6}")
    print("-" * 50)
    print(f"{'OVERALL':<15} {overall_stats['tp']:>6} {overall_stats['fp']:>6} "
          f"{overall_stats['fn']:>6} {overall_stats['tn']:>6}")

    # Save results to JSON
    output_json = args.output_json
    if not output_json:
        output_json = os.path.join(args.peft_path, "per_dataset_eval.json")
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
