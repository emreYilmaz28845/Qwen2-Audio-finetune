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
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils.grouped_eval import (
    GROUPED_DATASET_NAMES,
    SUPPORTED_GROUPED_EVAL_LEVELS,
    SUPPORTED_GROUPED_EVAL_MODES,
    apply_generated_person_level_results,
    build_grouped_task_metadata,
    grouped_eval_enabled,
    grouped_eval_env_prefix,
    grouped_person_results_key,
    make_binary_stats,
    map_target_text_to_binary,
    normalize_grouped_eval_level,
    normalize_grouped_eval_mode,
    parse_generated_label,
    update_binary_stats_with_prediction,
    validate_grouped_person_threshold,
)
from utils.functions import compute_metrics_from_stats


# ===============================
# Text-only dataset with metadata
# ===============================
class TextOnlyDatasetWithMeta(torch.utils.data.Dataset):
    """Loads text prompts + targets + dataset name for per-dataset eval."""

    def __init__(self, data_path, prompt_path, task_filename="merged_multitask.jsonl", default_dataset_name="unknown"):
        self.tasks = []
        self.prompt = {}
        self.default_dataset_name = default_dataset_name

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
        task = self.tasks[idx]
        target = task["target"]
        prompt = self.prompt[task["task"]]
        item = {"prompt": prompt, "target": target}
        item.update(build_grouped_task_metadata(task, default_dataset_name=self.default_dataset_name))
        return item


def collate_fn_textonly_generation_with_meta(samples, tokenizer):
    """Collate prompt-only inputs for generation-based evaluation."""
    dataset_names = [s.pop("dataset_name") for s in samples]
    target_texts = [s.pop("target_text") for s in samples]
    segment_keys = [s.pop("segment_key") for s in samples]
    group_ids = [s.pop("group_id") for s in samples]
    prompts = [s["prompt"] for s in samples]

    processed_data = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    processed_data["dataset_names"] = dataset_names
    processed_data["target_texts"] = target_texts
    processed_data["segment_keys"] = segment_keys
    processed_data["group_ids"] = group_ids
    processed_data["prompts"] = prompts
    return processed_data


def format_metrics(stats):
    if not stats or stats["total"] == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "weighted_f1": 0}
    accuracy, precision, recall, f1, weighted_f1 = compute_metrics_from_stats(stats)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "weighted_f1": weighted_f1,
    }


def format_result_entry(stats):
    return {**format_metrics(stats), **stats}


def print_metrics_row(name, stats):
    metrics = format_metrics(stats)
    print(f"{name:<15} {stats['total']:>8} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
          f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics['weighted_f1']:>8.4f}")


def print_confusion_row(name, stats):
    print(f"{name:<15} {stats['tp']:>6} {stats['fp']:>6} {stats['fn']:>6} {stats['tn']:>6}")


def label_to_text(label):
    if label == 1:
        return "depressed"
    if label == 0:
        return "non-depressed"
    return "unknown"


def update_segment_stats(stats, y_true, y_pred):
    if y_true == -1:
        return
    update_binary_stats_with_prediction(stats, y_true, y_pred)


def print_prediction_examples(title, predictions, limit):
    print(f"\n{title}:")
    if not predictions:
        print("  (none)")
        return
    selected = predictions if limit <= 0 else predictions[:limit]
    for item in selected:
        print(json.dumps(item, ensure_ascii=False))
    if limit > 0 and len(predictions) > limit:
        print(f"  ... {len(predictions) - limit} more written to output JSON")


def compare_with_teacher_forced_results(generation_results, teacher_forced_results_path):
    if not teacher_forced_results_path:
        return None
    with open(teacher_forced_results_path, encoding="utf-8") as handle:
        old_results = json.load(handle)

    comparison = {}
    for key, new_entry in generation_results.items():
        if key.startswith("_") or key not in old_results:
            continue
        old_entry = old_results[key]
        if not isinstance(old_entry, dict) or not isinstance(new_entry, dict):
            continue
        comparison[key] = {
            "teacher_forced_f1": old_entry.get("f1"),
            "generation_f1": new_entry.get("f1"),
            "f1_delta_generation_minus_teacher_forced": (
                None
                if old_entry.get("f1") is None or new_entry.get("f1") is None
                else new_entry.get("f1") - old_entry.get("f1")
            ),
            "teacher_forced_accuracy": old_entry.get("accuracy"),
            "generation_accuracy": new_entry.get("accuracy"),
        }
    return comparison


def resolve_grouped_eval_controls(args):
    levels = {}
    modes = {}
    thresholds = {}
    for dataset_name in sorted(GROUPED_DATASET_NAMES):
        prefix = grouped_eval_env_prefix(dataset_name)
        levels[dataset_name] = normalize_grouped_eval_level(
            getattr(args, f"{dataset_name}_eval_level", "person")
        )
        modes[dataset_name] = normalize_grouped_eval_mode(
            getattr(args, f"{dataset_name}_eval_mode", "majority_vote")
        )
        thresholds[dataset_name] = validate_grouped_person_threshold(
            getattr(args, f"{dataset_name}_person_threshold", 0.5)
        )
    return levels, modes, thresholds


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
    parser.add_argument("--dataset_name", type=str, default="merged")
    parser.add_argument("--daic_woz_eval_level", type=str, default="person",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_LEVELS))
    parser.add_argument("--daic_woz_eval_mode", type=str, default="majority_vote",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_MODES))
    parser.add_argument("--daic_woz_person_threshold", type=float, default=0.5)
    parser.add_argument("--eatd_eval_level", type=str, default="person",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_LEVELS))
    parser.add_argument("--eatd_eval_mode", type=str, default="majority_vote",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_MODES))
    parser.add_argument("--eatd_person_threshold", type=float, default=0.5)
    parser.add_argument("--cmdc_eval_level", type=str, default="person",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_LEVELS))
    parser.add_argument("--cmdc_eval_mode", type=str, default="majority_vote",
                        choices=sorted(SUPPORTED_GROUPED_EVAL_MODES))
    parser.add_argument("--cmdc_person_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--print_predictions_limit", type=int, default=50,
                        help="Number of segment and participant predictions to print; <=0 prints all.")
    parser.add_argument("--teacher_forced_results_json", type=str, default="",
                        help="Optional previous teacher-forced results JSON for metric comparison.")
    args = parser.parse_args()

    device = args.device
    peft_path = (args.peft_path or "").strip()
    dataset_name = (args.dataset_name or "").strip().lower()
    level_by_dataset, mode_by_dataset, threshold_by_dataset = resolve_grouped_eval_controls(args)
    use_peft = peft_path.lower() not in {"", "none", "null", "base", "baseline"}
    print(f"[Config] model_path: {args.model_path}")
    print(f"[Config] peft_path:  {peft_path if use_peft else '(none - base model)'}")
    print(f"[Config] data_path:  {args.data_path}")
    print(f"[Config] dataset:    {dataset_name}")
    for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
        print(
            f"[Config] {grouped_dataset_name}: "
            f"level={level_by_dataset[grouped_dataset_name]} "
            f"mode={mode_by_dataset[grouped_dataset_name]} "
            f"thr={threshold_by_dataset[grouped_dataset_name]}"
        )
    print(f"[Config] device:     {device}")
    print(f"[Config] mode:       TEXT-ONLY (Qwen2-7B)")
    print(f"[Config] eval_method: prompt-only generation")
    print(f"[Config] max_new_tokens: {args.max_new_tokens}")

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

    # ===============================
    # Load data
    # ===============================
    print("[3/4] Loading evaluation dataset...")
    eval_dataset = TextOnlyDatasetWithMeta(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        task_filename=args.task_filename,
        default_dataset_name=dataset_name,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial(collate_fn_textonly_generation_with_meta, tokenizer=tokenizer),
        shuffle=False,
    )
    print(f"   Total samples: {len(eval_dataset)}")

    # ===============================
    # Run evaluation
    # ===============================
    print("[4/4] Running evaluation...\n")
    per_dataset_stats = {}
    overall_stats = make_binary_stats()
    grouped_records_by_dataset = {dataset: [] for dataset in GROUPED_DATASET_NAMES}
    segment_predictions = []
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="[Eval]"):
            dataset_names = batch.pop("dataset_names")
            target_texts = batch.pop("target_texts")
            segment_keys = batch.pop("segment_keys")
            group_ids = batch.pop("group_ids")
            prompts = batch.pop("prompts")
            model_inputs = {key: value.to(device) for key, value in batch.items()}
            input_length = model_inputs["input_ids"].shape[1]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                generated_ids = model.generate(**model_inputs, **generation_kwargs)

            generated_only_ids = generated_ids[:, input_length:]
            generated_texts = tokenizer.batch_decode(
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for idx, generated_text in enumerate(generated_texts):
                ds_name = dataset_names[idx]
                y_true = map_target_text_to_binary(target_texts[idx])
                parse = parse_generated_label(generated_text)
                y_pred = parse["label"]

                update_segment_stats(overall_stats, y_true, y_pred)
                if ds_name not in per_dataset_stats:
                    per_dataset_stats[ds_name] = make_binary_stats()
                update_segment_stats(per_dataset_stats[ds_name], y_true, y_pred)

                prediction_record = {
                    "dataset": ds_name,
                    "segment_key": segment_keys[idx],
                    "participant_id": group_ids[idx],
                    "prompt": prompts[idx],
                    "target_text": target_texts[idx],
                    "true_label": y_true,
                    "true_label_text": label_to_text(y_true),
                    "raw_generated_text": generated_text,
                    "normalized_generated_text": parse["normalized_text"],
                    "parsed_label": y_pred,
                    "parsed_label_text": label_to_text(y_pred),
                    "matched_pattern": parse["matched_pattern"],
                    "ambiguous": parse["ambiguous"],
                    "parse_reason": parse["parse_reason"],
                }
                segment_predictions.append(prediction_record)

                if grouped_eval_enabled(ds_name) and segment_keys[idx] and group_ids[idx]:
                    grouped_records_by_dataset[ds_name].append(
                        {
                            "key": segment_keys[idx],
                            "group_id": group_ids[idx],
                            "target_text": target_texts[idx],
                            "pred_label": y_pred,
                            "depressed_probability": 1.0 if y_pred == 1 else 0.0,
                            "raw_generated_text": generated_text,
                            "parse_reason": parse["parse_reason"],
                        }
                    )

    # ===============================
    # Print results
    # ===============================
    segment_level_results = {
        ds_name: format_result_entry(stats)
        for ds_name, stats in sorted(per_dataset_stats.items())
    }
    segment_level_results["overall"] = format_result_entry(overall_stats)

    per_dataset_stats, overall_stats, supplemental_results, participant_predictions_by_dataset = apply_generated_person_level_results(
        dataset_name=dataset_name,
        level_by_dataset=level_by_dataset,
        mode_by_dataset=mode_by_dataset,
        threshold_by_dataset=threshold_by_dataset,
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    print("\n" + "=" * 90)
    print("  PER-DATASET EVALUATION RESULTS (TEXT-ONLY Qwen2-7B, PROMPT-ONLY GENERATION)")
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
            "dataset_name": dataset_name,
            "grouped_eval_levels": level_by_dataset,
            "grouped_eval_modes": mode_by_dataset,
            "grouped_person_thresholds": threshold_by_dataset,
            "batch_size": args.batch_size,
            "device": args.device,
            "eval_method": "prompt_only_generation",
            "max_new_tokens": args.max_new_tokens,
            "parsing_policy": (
                "Generated text is lowercased and whitespace-normalized. Explicit Chinese and English "
                "non-depressed patterns are matched before depressed patterns so 非抑郁 is not mistaken "
                "for 抑郁. If both classes appear, the last explicit class mention is used and marked "
                "ambiguous. If no supported class is found, prediction is unknown and counted as an error."
            ),
            "used_peft": use_peft,
        },
        "_segment_level": segment_level_results,
        "_segment_predictions": segment_predictions,
        "_participant_predictions": participant_predictions_by_dataset,
    }
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        results[ds_name] = format_result_entry(stats)
        print_metrics_row(ds_name, stats)

    for ds_name in sorted(supplemental_results.keys()):
        stats = supplemental_results[ds_name]
        results[ds_name] = format_result_entry(stats)
        print_metrics_row(ds_name, stats)

    results["overall"] = format_result_entry(overall_stats)
    print("-" * 90)
    print_metrics_row("OVERALL", overall_stats)
    print("=" * 90)

    for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
        if dataset_name == grouped_dataset_name and level_by_dataset[grouped_dataset_name] == "person" and grouped_dataset_name in results:
            print(
                f"[Info] {grouped_dataset_name} person-level counts: "
                f"participants={results[grouped_dataset_name].get('num_participants', 0)} "
                f"segments={results[grouped_dataset_name].get('num_segments', 0)}"
            )
        supplemental_key = grouped_person_results_key(grouped_dataset_name)
        if dataset_name == "merged" and level_by_dataset[grouped_dataset_name] == "person" and supplemental_key in results:
            print(
                f"[Info] Supplemental merged {grouped_dataset_name} person-level counts: "
                f"participants={results[supplemental_key].get('num_participants', 0)} "
                f"segments={results[supplemental_key].get('num_segments', 0)}"
            )

    # Confusion matrix
    print("\nConfusion Matrix Details:")
    print(f"{'Dataset':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 50)
    for ds_name in sorted(per_dataset_stats.keys()):
        print_confusion_row(ds_name, per_dataset_stats[ds_name])
    for ds_name in sorted(supplemental_results.keys()):
        print_confusion_row(ds_name, supplemental_results[ds_name])
    print("-" * 50)
    print_confusion_row("OVERALL", overall_stats)

    print_prediction_examples("Segment Predictions", segment_predictions, args.print_predictions_limit)
    flat_participant_predictions = []
    for ds_name, predictions in sorted(participant_predictions_by_dataset.items()):
        for item in predictions:
            flat_item = {"dataset": ds_name, **item}
            flat_participant_predictions.append(flat_item)
    print_prediction_examples("Participant Predictions", flat_participant_predictions, args.print_predictions_limit)

    comparison = compare_with_teacher_forced_results(results, args.teacher_forced_results_json)
    if comparison is not None:
        results["_teacher_forced_comparison"] = comparison
        print("\nTeacher-Forced vs Prompt-Only Generation:")
        for key, value in sorted(comparison.items()):
            print(json.dumps({"dataset": key, **value}, ensure_ascii=False))

    # Save JSON
    output_json = (args.output_json or "").strip()
    if not output_json:
        output_json = (
            os.path.join(peft_path, "per_dataset_eval_textonly_generation.json")
            if use_peft
            else os.path.abspath("base_per_dataset_eval_textonly_generation.json")
        )
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
