"""
Per-dataset evaluation script for the full-audio Qwen2-Audio depression detection model.

This evaluator matches the current `src/train_ddp.py` training path:
- full `Qwen2AudioForConditionalGeneration`
- LoRA attached to the full model
- custom `audio_adapter` inserted on `model.audio_tower`

Old language-model-only audio checkpoints are intentionally unsupported here.
"""

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from src.dataset import AudioDataset, collate_fn_qwen2audio
from utils.grouped_eval import (
    GROUPED_DATASET_NAMES,
    SUPPORTED_GROUPED_EVAL_LEVELS,
    SUPPORTED_GROUPED_EVAL_MODES,
    apply_person_level_results,
    build_grouped_eval_records,
    build_grouped_task_metadata,
    grouped_eval_enabled,
    grouped_eval_env_prefix,
    grouped_person_results_key,
    make_binary_stats,
    normalize_grouped_eval_level,
    normalize_grouped_eval_mode,
    validate_grouped_person_threshold,
)
from utils.functions import (
    compute_metrics_from_stats,
    compute_metrics_text_binary_accumulate,
)

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover - runtime dependency expectation
    safe_open = None


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


class AudioDatasetWithMeta(AudioDataset):
    def __init__(self, *args, default_dataset_name="unknown", **kwargs):
        self.default_dataset_name = default_dataset_name
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(build_grouped_task_metadata(self.tasks[idx], default_dataset_name=self.default_dataset_name))
        return item


def collate_fn_with_meta(samples, processor):
    dataset_names = [s.pop("dataset_name") for s in samples]
    target_texts = [s.pop("target_text") for s in samples]
    segment_keys = [s.pop("segment_key") for s in samples]
    group_ids = [s.pop("group_id") for s in samples]
    processed_data = collate_fn_qwen2audio(samples, processor)
    processed_data["dataset_names"] = dataset_names
    processed_data["target_texts"] = target_texts
    processed_data["segment_keys"] = segment_keys
    processed_data["group_ids"] = group_ids
    return processed_data


def accumulate_segment_stats_per_dataset(processor, logits, labels, dataset_names, per_dataset_stats, overall_stats):
    compute_metrics_text_binary_accumulate(processor, logits, labels, overall_stats)

    batch_size = labels.size(0)
    for idx in range(batch_size):
        ds_name = dataset_names[idx]
        if ds_name not in per_dataset_stats:
            per_dataset_stats[ds_name] = make_binary_stats()
        compute_metrics_text_binary_accumulate(
            processor,
            logits[idx: idx + 1],
            labels[idx: idx + 1],
            per_dataset_stats[ds_name],
        )


def format_metrics(stats):
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


def format_result_entry(stats):
    return {**format_metrics(stats), **stats}


def optional_split_audit_hash(data_path: str):
    audit_path = Path(data_path).resolve().parent / "split_audit.json"
    if not audit_path.exists():
        return {"path": str(audit_path), "sha256": ""}
    digest = hashlib.sha256()
    with audit_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(audit_path), "sha256": digest.hexdigest()}


def print_metrics_row(name, stats):
    metrics = format_metrics(stats)
    print(
        f"{name:<15} {stats['total']:>8} {metrics['accuracy']:>8.4f} "
        f"{metrics['precision']:>8.4f} {metrics['recall']:>8.4f} "
        f"{metrics['f1']:>8.4f} {metrics['weighted_f1']:>8.4f}"
    )


def print_confusion_row(name, stats):
    print(f"{name:<15} {stats['tp']:>6} {stats['fp']:>6} {stats['fn']:>6} {stats['tn']:>6}")


def resolve_grouped_eval_controls(args):
    levels = {}
    modes = {}
    thresholds = {}
    for dataset_name in sorted(GROUPED_DATASET_NAMES):
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


def load_peft_tensor_names(peft_path):
    adapter_weights_path = os.path.join(peft_path, "adapter_model.safetensors")
    if not os.path.isfile(adapter_weights_path):
        raise FileNotFoundError(
            f"Expected PEFT weights at {adapter_weights_path}, but the file was not found."
        )
    if safe_open is None:
        raise RuntimeError(
            "The `safetensors` package is required to inspect adapter checkpoints, but it is not installed."
        )

    with safe_open(adapter_weights_path, framework="pt", device="cpu") as checkpoint:
        return list(checkpoint.keys())


def detect_checkpoint_mode(peft_path):
    tensor_names = load_peft_tensor_names(peft_path)
    has_audio_keys = any("audio_tower" in name or "audio_adapter" in name for name in tensor_names)
    if not has_audio_keys:
        raise RuntimeError(
            "Unsupported old audio checkpoint format detected. This evaluator only supports full-audio "
            "checkpoints from the current `src/train_ddp.py` pipeline, but the PEFT weights contain no "
            "`audio_tower` / `audio_adapter` entries."
        )
    return "full_audio", tensor_names


def infer_default_adapter_path(peft_path):
    candidates = [
        os.path.join(peft_path, "audio_adapter_state.pt"),
        os.path.join(peft_path, "audio_adapter.bin"),
        os.path.join(peft_path, "audio_adapter.pt"),
        os.path.join(peft_path, "audio_adapter.pth"),
        os.path.join(peft_path, "dep_adapter.pt"),
        os.path.join(peft_path, "dep_adapter.bin"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return candidates[0]


def normalize_audio_adapter_state_dict(state):
    normalized_state = {}
    had_peft_wrapped_keys = False

    for key, value in state.items():
        if key.startswith("audio_adapter."):
            key = key[len("audio_adapter."):]

        if ".lora_" in key:
            had_peft_wrapped_keys = True
            continue

        if ".base_layer." in key:
            key = key.replace(".base_layer.", ".")
            had_peft_wrapped_keys = True

        normalized_state[key] = value

    return normalized_state, had_peft_wrapped_keys


def load_audio_adapter_state(model, adapter_path):
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(
            "Full-audio evaluation requires the trained base weights of the custom `audio_adapter`, "
            f"but no adapter state file was found at: {adapter_path}\n"
            "PEFT saved the LoRA weights, but not the trained base adapter weights needed for valid "
            "evaluation. Provide `--adapter_path` explicitly or save the adapter state alongside the checkpoint."
        )

    state = torch.load(adapter_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Adapter state at {adapter_path} is not a valid state dict.")

    state, had_peft_wrapped_keys = normalize_audio_adapter_state_dict(state)

    missing, unexpected = model.audio_tower.audio_adapter.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load `audio_adapter` weights cleanly.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}\nAdapter path: {adapter_path}"
        )
    if had_peft_wrapped_keys:
        print(
            "[Info] Loaded a PEFT-wrapped audio_adapter checkpoint by restoring only the base adapter "
            "weights from `audio_adapter_state.pt`. LoRA weights will be loaded from the PEFT checkpoint."
        )


def validate_checkpoint_mode(requested_mode, detected_mode):
    if requested_mode == "auto":
        return detected_mode
    if requested_mode != detected_mode:
        raise RuntimeError(
            f"Checkpoint mode mismatch: requested `{requested_mode}`, but detected `{detected_mode}` "
            "from the PEFT checkpoint contents."
        )
    return requested_mode


def main():
    parser = argparse.ArgumentParser(
        description="Per-dataset evaluation of the full-audio Qwen2-Audio depression detection model"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to base Qwen2-Audio-7B-Instruct model")
    parser.add_argument("--peft_path", type=str, default="", help="Path to saved full-audio LoRA checkpoint")
    parser.add_argument("--adapter_path", type=str, default="", help="Path to the trained audio_adapter state file")
    parser.add_argument("--checkpoint_mode", type=str, default="auto", choices=["auto", "full_audio"])
    parser.add_argument("--data_path", type=str, default="data/merged/val", help="Path to merged val data directory")
    parser.add_argument("--prompt_path", type=str, default="data/merged/val/merged_multiprompt_audiotext.jsonl")
    parser.add_argument("--scp_filename", type=str, default="merged.scp")
    parser.add_argument("--task_filename", type=str, default="merged_multitask.jsonl")
    parser.add_argument("--dataset_name", type=str, default="merged")
    parser.add_argument("--data_split", type=str, default="test")
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
    parser.add_argument("--adapter_dim", type=int, default=32)
    parser.add_argument("--adapter_dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = args.device
    peft_path = (args.peft_path or "").strip()
    adapter_path = (args.adapter_path or "").strip()
    dataset_name = (args.dataset_name or "").strip().lower()
    level_by_dataset, mode_by_dataset, threshold_by_dataset = resolve_grouped_eval_controls(args)
    use_peft = peft_path.lower() not in {"", "none", "null", "base", "baseline"}

    if dataset_name not in {"merged", "daic_woz", "eatd", "cmdc"}:
        raise ValueError(f"Unsupported dataset_name={args.dataset_name!r}.")

    if not use_peft:
        raise RuntimeError(
            "This evaluator only supports current full-audio fine-tuned checkpoints. "
            "Provide `--peft_path` for a full-audio LoRA checkpoint."
        )

    detected_mode, tensor_names = detect_checkpoint_mode(peft_path)
    checkpoint_mode = validate_checkpoint_mode(args.checkpoint_mode, detected_mode)
    resolved_adapter_path = adapter_path or infer_default_adapter_path(peft_path)

    print(f"[Config] model_path:       {args.model_path}")
    print(f"[Config] peft_path:        {peft_path}")
    print(f"[Config] adapter_path:     {resolved_adapter_path}")
    print(f"[Config] checkpoint_mode:  {checkpoint_mode} (detected from PEFT weights)")
    print(f"[Config] dataset_name:      {dataset_name}")
    print(f"[Config] data_path:        {args.data_path}")
    print(f"[Config] prompt_path:      {args.prompt_path}")
    for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
        print(
            f"[Config] {grouped_dataset_name}: "
            f"level={level_by_dataset[grouped_dataset_name]} "
            f"mode={mode_by_dataset[grouped_dataset_name]} "
            f"thr={threshold_by_dataset[grouped_dataset_name]}"
        )
    print(f"[Config] device:           {device}")
    print(f"[Config] audio_tensor_key_count: {sum('audio_tower' in name for name in tensor_names)}")

    print("\n[1/4] Loading processor and base full-audio model...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    full_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    adapter_config = {
        "adapter_dim": args.adapter_dim,
        "dropout": args.adapter_dropout,
    }
    full_model.audio_tower = create_modified_qwen2audio_encoder(full_model.audio_tower, adapter_config)

    print("[2/4] Loading trained audio_adapter base weights...")
    load_audio_adapter_state(full_model, resolved_adapter_path)

    print("[3/4] Loading full-model LoRA checkpoint...")
    model = PeftModel.from_pretrained(full_model, peft_path)
    model.eval()
    model.to(device)

    print("[4/4] Loading evaluation dataset...")
    eval_dataset = AudioDatasetWithMeta(
        data_path=args.data_path,
        prompt_path=args.prompt_path,
        wav_type="wav",
        scp_filename=args.scp_filename,
        task_filename=args.task_filename,
        default_dataset_name=dataset_name,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=partial(collate_fn_with_meta, processor=processor),
        shuffle=False,
    )
    print(f"   Total samples: {len(eval_dataset)}")

    print("\n[Eval] Running full-audio evaluation...\n")
    segment_per_dataset_stats = {}
    segment_overall_stats = make_binary_stats()
    grouped_records_by_dataset = {dataset: [] for dataset in GROUPED_DATASET_NAMES}
    eval_loss_sum = 0.0
    eval_steps = 0
    use_person_as_primary = (
        grouped_eval_enabled(dataset_name)
        and level_by_dataset.get(dataset_name, "segment") == "person"
    )

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="[Eval]"):
            dataset_names = batch.pop("dataset_names")
            target_texts = batch.pop("target_texts")
            segment_keys = batch.pop("segment_keys")
            group_ids = batch.pop("group_ids")
            batch = batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            eval_loss_sum += loss.item()
            eval_steps += 1

            accumulate_segment_stats_per_dataset(
                processor,
                outputs.logits,
                batch["labels"],
                dataset_names,
                segment_per_dataset_stats,
                segment_overall_stats,
            )

            batch_grouped_records = build_grouped_eval_records(
                processor.tokenizer,
                outputs.logits,
                batch["labels"],
                dataset_names,
                segment_keys,
                group_ids,
                target_texts,
            )
            for grouped_dataset_name, grouped_records in batch_grouped_records.items():
                grouped_records_by_dataset[grouped_dataset_name].extend(grouped_records)

    avg_loss = eval_loss_sum / eval_steps if eval_steps > 0 else 0.0
    per_dataset_stats = {} if use_person_as_primary else {
        name: dict(stats) for name, stats in segment_per_dataset_stats.items()
    }
    overall_stats = make_binary_stats() if use_person_as_primary else dict(segment_overall_stats)
    per_dataset_stats, overall_stats, supplemental_results = apply_person_level_results(
        dataset_name=dataset_name,
        level_by_dataset=level_by_dataset,
        mode_by_dataset=mode_by_dataset,
        threshold_by_dataset=threshold_by_dataset,
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    print("\n" + "=" * 90)
    print(f"  PER-DATASET EVALUATION RESULTS    (avg loss: {avg_loss:.4f})")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Samples':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'wF1':>8}")
    print("-" * 90)

    results = {
        "_meta": {
            "model_path": args.model_path,
            "peft_path": peft_path,
            "adapter_path": resolved_adapter_path,
            "checkpoint_mode": checkpoint_mode,
            "data_path": args.data_path,
            "data_split": args.data_split,
            "split_audit": optional_split_audit_hash(args.data_path),
            "prompt_path": args.prompt_path,
            "scp_filename": args.scp_filename,
            "task_filename": args.task_filename,
            "dataset_name": dataset_name,
            "primary_level": "person" if use_person_as_primary else "segment",
            "grouped_eval_levels": level_by_dataset,
            "grouped_eval_modes": mode_by_dataset,
            "grouped_person_thresholds": threshold_by_dataset,
            "batch_size": args.batch_size,
            "device": args.device,
            "avg_loss": avg_loss,
        }
    }
    results["_segment_eval"] = {
        "per_dataset": {
            ds_name: format_result_entry(stats)
            for ds_name, stats in sorted(segment_per_dataset_stats.items())
        },
        "overall": format_result_entry(segment_overall_stats),
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

    print("\nConfusion Matrix Details:")
    print(f"{'Dataset':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print("-" * 50)
    for ds_name in sorted(per_dataset_stats.keys()):
        print_confusion_row(ds_name, per_dataset_stats[ds_name])
    for ds_name in sorted(supplemental_results.keys()):
        print_confusion_row(ds_name, supplemental_results[ds_name])
    print("-" * 50)
    print_confusion_row("OVERALL", overall_stats)

    if use_person_as_primary:
        print("\nSegment-Level Diagnostics:")
        print(f"{'Dataset':<15} {'Samples':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'wF1':>8}")
        print("-" * 90)
        for ds_name in sorted(segment_per_dataset_stats.keys()):
            print_metrics_row(ds_name, segment_per_dataset_stats[ds_name])
        print("-" * 90)
        print_metrics_row("SEG_OVERALL", segment_overall_stats)

    output_json = (args.output_json or "").strip()
    if not output_json:
        output_json = os.path.join(peft_path, "per_dataset_eval.json")
    output_dir = os.path.dirname(output_json) if os.path.dirname(output_json) else "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
