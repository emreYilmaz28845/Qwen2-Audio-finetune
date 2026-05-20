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
from utils.grouped_eval import (
    GROUPED_DATASET_NAMES,
    aggregate_group_predictions,
    apply_generated_person_level_results,
    build_grouped_eval_records,
    build_grouped_task_metadata,
    grouped_eval_enabled,
    grouped_person_results_key,
    make_binary_stats,
    map_target_text_to_binary,
    normalize_grouped_eval_level,
    parse_generated_label,
    update_binary_stats_with_prediction,
)
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


class TextOnlyEvalDatasetWithMeta(TextOnlyDataset):
    def __init__(self, *args, default_dataset_name="unknown", **kwargs):
        self.default_dataset_name = default_dataset_name
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(build_grouped_task_metadata(self.tasks[idx], default_dataset_name=self.default_dataset_name))
        return item


class AudioEvalDatasetWithMeta(AudioDataset):
    def __init__(self, *args, default_dataset_name="unknown", **kwargs):
        self.default_dataset_name = default_dataset_name
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(build_grouped_task_metadata(self.tasks[idx], default_dataset_name=self.default_dataset_name))
        return item


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


def collate_fn_textonly_with_meta(samples, tokenizer):
    dataset_names = [sample.pop("dataset_name") for sample in samples]
    segment_keys = [sample.pop("segment_key") for sample in samples]
    group_ids = [sample.pop("group_id") for sample in samples]
    target_texts = [sample["target"] for sample in samples]
    processed_data = collate_fn_textonly(samples, tokenizer)
    processed_data["dataset_names"] = dataset_names
    processed_data["segment_keys"] = segment_keys
    processed_data["group_ids"] = group_ids
    processed_data["target_texts"] = target_texts
    return processed_data


def collate_fn_qwen2audio_with_meta(samples, processor):
    dataset_names = [sample.pop("dataset_name") for sample in samples]
    segment_keys = [sample.pop("segment_key") for sample in samples]
    group_ids = [sample.pop("group_id") for sample in samples]
    target_texts = [sample["target"] for sample in samples]
    processed_data = collate_fn_qwen2audio(samples, processor)
    processed_data["dataset_names"] = dataset_names
    processed_data["segment_keys"] = segment_keys
    processed_data["group_ids"] = group_ids
    processed_data["target_texts"] = target_texts
    return processed_data


def collate_fn_textonly_eval_generation_with_meta(samples, tokenizer):
    prompts = [sample["prompt"] for sample in samples]
    processed_data = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    processed_data["dataset_names"] = [sample["dataset_name"] for sample in samples]
    processed_data["segment_keys"] = [sample["segment_key"] for sample in samples]
    processed_data["group_ids"] = [sample["group_id"] for sample in samples]
    processed_data["target_texts"] = [sample["target_text"] for sample in samples]
    return processed_data


def collate_fn_textonly_eval_dual(samples, tokenizer):
    supervised_samples = [{"prompt": sample["prompt"], "target": sample["target"]} for sample in samples]
    processed_data = collate_fn_textonly(supervised_samples, tokenizer)
    generation_data = collate_fn_textonly_eval_generation_with_meta(samples, tokenizer)
    processed_data["dataset_names"] = generation_data["dataset_names"]
    processed_data["segment_keys"] = generation_data["segment_keys"]
    processed_data["group_ids"] = generation_data["group_ids"]
    processed_data["target_texts"] = generation_data["target_texts"]
    for key, value in generation_data.items():
        if key in {"dataset_names", "segment_keys", "group_ids", "target_texts"}:
            continue
        processed_data[f"gen_{key}"] = value
    return processed_data


def collate_fn_qwen2audio_eval_generation_with_meta(samples, processor):
    prompts = [sample["prompt"] for sample in samples]
    audio = [sample["audio"] for sample in samples]
    processed_data = processor(
        text=prompts,
        audio=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    processed_data["dataset_names"] = [sample["dataset_name"] for sample in samples]
    processed_data["segment_keys"] = [sample["segment_key"] for sample in samples]
    processed_data["group_ids"] = [sample["group_id"] for sample in samples]
    processed_data["target_texts"] = [sample["target_text"] for sample in samples]
    return processed_data


def collate_fn_qwen2audio_eval_dual(samples, processor):
    supervised_samples = [
        {
            "prompt": sample["prompt"],
            "audio": sample["audio"],
            "target": sample["target"],
        }
        for sample in samples
    ]
    processed_data = collate_fn_qwen2audio(supervised_samples, processor)
    generation_data = collate_fn_qwen2audio_eval_generation_with_meta(samples, processor)
    processed_data["dataset_names"] = generation_data["dataset_names"]
    processed_data["segment_keys"] = generation_data["segment_keys"]
    processed_data["group_ids"] = generation_data["group_ids"]
    processed_data["target_texts"] = generation_data["target_texts"]
    for key, value in generation_data.items():
        if key in {"dataset_names", "segment_keys", "group_ids", "target_texts"}:
            continue
        processed_data[f"gen_{key}"] = value
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


def save_audio_adapter_state(model, save_dir):
    adapter_state_path = os.path.join(save_dir, "audio_adapter_state.pt")
    adapter_state = {}
    for key, value in model.audio_tower.audio_adapter.state_dict().items():
        if ".lora_" in key:
            continue
        key = key.replace(".base_layer.", ".")
        adapter_state[key] = value.detach().cpu()
    torch.save(adapter_state, adapter_state_path)
    return adapter_state_path


def _write_json_atomic(path: str, payload: dict):
    if not path:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, path)


def _check_stop_requested(stop_file: str):
    if stop_file and os.path.exists(stop_file):
        raise RuntimeError("Trial stopped due to pruning request")


def _report_progress(progress_file: str, step: int, metric: float):
    if not progress_file:
        return
    _write_json_atomic(
        progress_file,
        {
            "step": int(step),
            "metric": float(metric),
        },
    )


def _model_io_debug_enabled(cfg) -> bool:
    return bool(getattr(cfg.env, "debug_model_io", False))


def _model_io_debug_limit(cfg) -> int:
    return max(0, int(getattr(cfg.env, "debug_model_io_limit", 2)))


def _model_io_debug_train_every_steps(cfg) -> int:
    return max(1, int(getattr(cfg.env, "debug_model_io_train_every_steps", 1)))


def _model_io_debug_eval_every_calls(cfg) -> int:
    return max(1, int(getattr(cfg.env, "debug_model_io_eval_every_calls", 1)))


def _model_io_debug_state(cfg) -> dict:
    state = getattr(cfg.env, "_model_io_debug_state", None)
    if state is None:
        state = {
            "train": 0,
            "eval": 0,
            "last_logged_event": {
                "train": None,
                "eval": None,
            },
        }
        cfg.env._model_io_debug_state = state
    return state


def _decode_ids(tokenizer, token_ids, *, skip_special_tokens: bool):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _map_label_text_to_id(text: str) -> int:
    return parse_generated_label(text).get("label", -1)


def _label_id_to_name(label_id: int) -> str:
    if label_id == 0:
        return "non_depressed"
    if label_id == 1:
        return "depressed"
    return "unknown"


def _eval_generation_max_new_tokens(cfg):
    return max(1, int(getattr(cfg.eval, "max_new_tokens", 16)))


def _eval_prediction_log_limit(cfg):
    return int(getattr(cfg.eval, "print_predictions_limit", 5))


def _make_generation_kwargs(metric_processor, cfg):
    tokenizer = metric_processor.tokenizer
    kwargs = {
        "max_new_tokens": _eval_generation_max_new_tokens(cfg),
        "do_sample": False,
        "num_beams": 1,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if dist.is_initialized() and dist.get_world_size() > 1:
        kwargs["synced_gpus"] = True
    return kwargs


def _update_generation_stats(stats: dict, y_true: int, y_pred: int):
    if y_true == -1:
        return
    stats.setdefault("num_invalid_predictions", 0)
    if y_pred not in {0, 1}:
        stats["num_invalid_predictions"] += 1
    update_binary_stats_with_prediction(stats, y_true, y_pred)


def _merge_stats_dicts(left, right):
    if not left:
        return dict(right or {})
    merged = dict(left)
    for key, value in (right or {}).items():
        if isinstance(value, (int, float)):
            merged[key] = float(merged.get(key, 0.0)) + float(value)
        else:
            merged[key] = value
    return merged


def _gather_and_merge_stats_by_dataset(local_stats_by_dataset: dict):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_stats_by_dataset)
    merged = {}
    for rank_stats in gathered:
        for dataset_name, stats in (rank_stats or {}).items():
            merged[dataset_name] = _merge_stats_dicts(merged.get(dataset_name), stats)
    return merged


def _gather_limited_examples(local_examples, limit: int):
    gathered = [None for _ in range(dist.get_world_size())]
    local_payload = list(local_examples if limit <= 0 else local_examples[:limit])
    dist.all_gather_object(gathered, local_payload)
    merged = []
    for rank_examples in gathered:
        for item in rank_examples or []:
            merged.append(item)
            if limit > 0 and len(merged) >= limit:
                return merged
    return merged


def _split_eval_batch(batch, device):
    metadata = {}
    supervised_batch = {}
    generation_batch = {}
    for key, value in batch.items():
        if key.startswith("gen_"):
            generation_batch[key[4:]] = value.to(device) if isinstance(value, torch.Tensor) else value
        elif key in {"dataset_names", "segment_keys", "group_ids", "target_texts"}:
            metadata[key] = value
        else:
            supervised_batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return metadata, supervised_batch, generation_batch


def _log_prediction_examples(logger, title: str, predictions, limit: int):
    if logger is None or limit == 0:
        return
    logger.info("%s", title)
    if not predictions:
        logger.info("  (none)")
        return
    selected = predictions if limit <= 0 else predictions[:limit]
    for item in selected:
        logger.info("  %s", json.dumps(item, ensure_ascii=False))
    if limit > 0 and len(predictions) > limit:
        logger.info("  ... %s more", len(predictions) - limit)


def _extract_debug_texts(metric_processor, batch, logits, sample_idx: int = 0):
    tokenizer = metric_processor.tokenizer
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    preds = torch.argmax(logits, dim=-1)

    label_mask = labels[sample_idx] != -100
    label_indices = label_mask.nonzero(as_tuple=False).squeeze(-1)
    true_ids = labels[sample_idx][label_indices] if label_indices.numel() > 0 else labels[sample_idx].new_empty((0,))
    pred_indices = (label_indices - 1).clamp(min=0)
    pred_ids = preds[sample_idx][pred_indices] if pred_indices.numel() > 0 else preds[sample_idx].new_empty((0,))

    if attention_mask is not None:
        attn_mask = attention_mask[sample_idx].bool()
        visible_input_ids = input_ids[sample_idx][attn_mask]
        prompt_mask = attn_mask & ~label_mask
    else:
        visible_input_ids = input_ids[sample_idx]
        prompt_mask = ~label_mask

    prompt_ids = input_ids[sample_idx][prompt_mask]

    return {
        "prompt_text": _decode_ids(tokenizer, prompt_ids, skip_special_tokens=False),
        "full_input_text": _decode_ids(tokenizer, visible_input_ids, skip_special_tokens=False),
        "target_text": _decode_ids(tokenizer, true_ids, skip_special_tokens=True),
        "predicted_text": _decode_ids(tokenizer, pred_ids, skip_special_tokens=True),
        "supervised_token_count": int(label_indices.numel()),
        "prompt_ids": prompt_ids.detach().clone(),
        "prompt_length": int(prompt_ids.numel()),
    }


def _build_prompt_only_generate_inputs(batch, prompt_ids: torch.Tensor, sample_idx: int = 0):
    generate_inputs = {}
    for key, value in batch.items():
        if key == "labels":
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if key == "input_ids":
            generate_inputs[key] = prompt_ids.unsqueeze(0)
        elif key == "attention_mask":
            generate_inputs[key] = torch.ones(
                (1, prompt_ids.numel()),
                dtype=value.dtype,
                device=value.device,
            )
        else:
            generate_inputs[key] = value[sample_idx : sample_idx + 1]
    return generate_inputs


def _prompt_only_generate_text(model, metric_processor, batch, prompt_ids: torch.Tensor, sample_idx: int = 0) -> str:
    if prompt_ids.numel() == 0:
        return ""

    generate_inputs = _build_prompt_only_generate_inputs(batch, prompt_ids, sample_idx=sample_idx)
    generate_model = model.module if hasattr(model, "module") else model
    tokenizer = metric_processor.tokenizer
    generation_kwargs = {
        "max_new_tokens": 8,
        "do_sample": False,
        "num_beams": 1,
    }
    if getattr(tokenizer, "pad_token_id", None) is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if dist.is_initialized() and dist.get_world_size() > 1:
        generation_kwargs["synced_gpus"] = True
    generated = generate_model.generate(**generate_inputs, **generation_kwargs)
    generated_suffix = generated[:, prompt_ids.numel() :]
    if generated_suffix.size(0) == 0:
        return ""
    return metric_processor.tokenizer.batch_decode(
        generated_suffix,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _maybe_log_model_io_debug(
    model,
    logger,
    cfg,
    phase: str,
    metric_processor,
    batch,
    logits,
    *,
    rank: int,
    epoch: int,
    step: int,
    event_index: int,
    loss: float,
):
    if logger is None or rank != 0 or not _model_io_debug_enabled(cfg):
        return

    state = _model_io_debug_state(cfg)
    if state.get(phase, 0) >= _model_io_debug_limit(cfg):
        return
    if "input_ids" not in batch or "labels" not in batch:
        return
    if event_index <= 0:
        return

    if phase == "train":
        interval = _model_io_debug_train_every_steps(cfg)
    else:
        interval = _model_io_debug_eval_every_calls(cfg)
    if (event_index - 1) % interval != 0:
        return
    if state["last_logged_event"].get(phase) == event_index:
        return

    debug_texts = _extract_debug_texts(metric_processor, batch, logits, sample_idx=0)
    state[phase] = state.get(phase, 0) + 1
    state["last_logged_event"][phase] = event_index
    prompt_only_generate = ""
    parsed_generate_label = "unknown"
    if phase == "eval":
        prompt_only_generate = _prompt_only_generate_text(
            model,
            metric_processor,
            batch,
            debug_texts["prompt_ids"],
            sample_idx=0,
        )
        parsed_generate_label = _label_id_to_name(parse_generated_label(prompt_only_generate)["label"])
    true_label = _label_id_to_name(map_target_text_to_binary(debug_texts["target_text"]))
    full_input_contains_target = bool(
        debug_texts["target_text"] and debug_texts["target_text"] in debug_texts["full_input_text"]
    )

    logger.info("[Model IO Debug][%s #%s] epoch=%s step=%s loss=%.4f", phase.upper(), state[phase], epoch, step, loss)
    logger.info("  PHASE=%s", phase)
    logger.info("  prompt=%r", debug_texts["prompt_text"])
    logger.info("  full_model_input=%r", debug_texts["full_input_text"])
    logger.info("  input_contains_target=%s", full_input_contains_target)
    logger.info("  target_label=%r", debug_texts["target_text"])
    logger.info("  label_span_prediction=%r", debug_texts["predicted_text"])
    if phase == "eval":
        logger.info("  prompt_only_generate=%r", prompt_only_generate)
        logger.info("  parsed_generate_label=%s", parsed_generate_label)
    logger.info("  true_label=%s", true_label)
    logger.info("  supervised_tokens=%s", debug_texts["supervised_token_count"])


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
    logger.info(
        "Model IO Debug: %s (limit per phase=%s)",
        _model_io_debug_enabled(cfg),
        _model_io_debug_limit(cfg),
    )
    logger.info(
        "Model IO Debug Intervals: train_every_steps=%s eval_every_calls=%s",
        _model_io_debug_train_every_steps(cfg),
        _model_io_debug_eval_every_calls(cfg),
    )
    logger.info(
        "Validation Inference: prompt-only generation (max_new_tokens=%s, prediction_log_limit=%s)",
        _eval_generation_max_new_tokens(cfg),
        _eval_prediction_log_limit(cfg),
    )
    for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES):
        logger.info(
            "%s Eval: level=%s mode=%s threshold=%.4f",
            grouped_dataset_name,
            _grouped_eval_level(cfg, grouped_dataset_name),
            _grouped_eval_mode(cfg, grouped_dataset_name),
            _grouped_eval_threshold(cfg, grouped_dataset_name),
        )
    logger.info("%s", "=" * 60)


def _grouped_eval_level(cfg, dataset_name: str):
    attr_name = f"{dataset_name.split('_')[0]}_eval_level" if dataset_name != "daic_woz" else "daic_eval_level"
    return normalize_grouped_eval_level(getattr(cfg.eval, attr_name, "segment"))


def _grouped_eval_mode(cfg, dataset_name: str):
    attr_name = f"{dataset_name.split('_')[0]}_eval_mode" if dataset_name != "daic_woz" else "daic_eval_mode"
    return getattr(cfg.eval, attr_name, "majority_vote")


def _grouped_eval_threshold(cfg, dataset_name: str):
    attr_name = f"{dataset_name.split('_')[0]}_person_threshold" if dataset_name != "daic_woz" else "daic_person_threshold"
    return float(getattr(cfg.eval, attr_name, 0.5))


def _primary_grouped_person_eval_dataset(cfg):
    dataset_name = getattr(cfg.data, "dataset_name", "")
    if grouped_eval_enabled(dataset_name) and _grouped_eval_level(cfg, dataset_name) == "person":
        return dataset_name
    return ""


def _supplemental_grouped_person_eval_datasets(cfg):
    dataset_name = getattr(cfg.data, "dataset_name", "")
    if dataset_name != "merged":
        return []
    return [
        grouped_dataset_name
        for grouped_dataset_name in sorted(GROUPED_DATASET_NAMES)
        if _grouped_eval_level(cfg, grouped_dataset_name) == "person"
    ]


def _needs_grouped_eval_metadata(cfg):
    return bool(_primary_grouped_person_eval_dataset(cfg) or _supplemental_grouped_person_eval_datasets(cfg))


def _is_grouped_person_eval(cfg):
    return bool(_primary_grouped_person_eval_dataset(cfg))


def _is_daic_person_eval(cfg):
    return (
        _primary_grouped_person_eval_dataset(cfg) == "daic_woz"
    )


def _serialize_eval_result(stats: dict, *, mode: str, threshold: float):
    accuracy, precision, recall, f1, weighted_f1 = compute_metrics_from_stats(stats)
    return {
        "mode": mode,
        "threshold": float(threshold),
        "stats": {
            key: (float(value) if isinstance(value, (int, float)) else value)
            for key, value in stats.items()
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "weighted_f1": float(weighted_f1),
        },
    }


def _build_best_eval_summary(
    cfg,
    eval_loss: float,
    primary_stats: dict,
    supplemental_stats_by_dataset: dict,
    *,
    eval_method: str = "prompt_only_generation",
    segment_level_overall_stats=None,
    segment_level_stats_by_dataset=None,
    segment_prediction_examples=None,
    participant_predictions_by_dataset=None,
    teacher_forced_summary=None,
):
    primary_dataset_name = _primary_grouped_person_eval_dataset(cfg)
    primary_mode = _grouped_eval_mode(cfg, primary_dataset_name) if primary_dataset_name else "segment"
    primary_threshold = _grouped_eval_threshold(cfg, primary_dataset_name) if primary_dataset_name else 0.5
    summary = {
        "dataset_name": getattr(cfg.data, "dataset_name", ""),
        "eval_loss": float(eval_loss),
        "eval_method": eval_method,
        "primary_scope": primary_dataset_name or "overall_segment",
        "primary": _serialize_eval_result(
            primary_stats,
            mode=primary_mode,
            threshold=primary_threshold,
        ),
    }
    if supplemental_stats_by_dataset:
        summary["supplemental_grouped_eval"] = {
            grouped_person_results_key(grouped_dataset_name): _serialize_eval_result(
                stats,
                mode=_grouped_eval_mode(cfg, grouped_dataset_name),
                threshold=_grouped_eval_threshold(cfg, grouped_dataset_name),
            )
            for grouped_dataset_name, stats in sorted(supplemental_stats_by_dataset.items())
        }
    if segment_level_overall_stats:
        summary["segment_level"] = {
            "overall": _serialize_eval_result(
                segment_level_overall_stats,
                mode="segment",
                threshold=0.5,
            )
        }
        if segment_level_stats_by_dataset:
            summary["segment_level"]["per_dataset"] = {
                dataset_name: _serialize_eval_result(
                    stats,
                    mode="segment",
                    threshold=0.5,
                )
                for dataset_name, stats in sorted(segment_level_stats_by_dataset.items())
            }
    if segment_prediction_examples:
        summary["segment_prediction_examples"] = list(segment_prediction_examples)
    if participant_predictions_by_dataset:
        summary["participant_prediction_examples"] = {
            dataset_name: list(predictions)
            for dataset_name, predictions in sorted(participant_predictions_by_dataset.items())
            if predictions
        }
    if teacher_forced_summary is not None:
        summary["teacher_forced_diagnostic"] = teacher_forced_summary
    return summary


def _evaluate(
    model,
    eval_dataloader,
    device,
    metric_processor,
    rank,
    cfg,
    logger=None,
    epoch=-1,
    step=-1,
    eval_event_index=-1,
):
    eval_loss = 0.0
    eval_steps = 0
    teacher_forced_segment_stats = None
    generation_segment_stats = make_binary_stats()
    generation_segment_stats["num_invalid_predictions"] = 0
    generation_per_dataset_segment_stats = {}
    generation_grouped_local_records = {dataset_name: [] for dataset_name in GROUPED_DATASET_NAMES}
    teacher_forced_grouped_local_records = {dataset_name: [] for dataset_name in GROUPED_DATASET_NAMES}
    local_segment_prediction_examples = []
    eval_bar = tqdm(eval_dataloader, desc="[Eval]") if rank == 0 else eval_dataloader
    primary_grouped_dataset_name = _primary_grouped_person_eval_dataset(cfg)
    supplemental_grouped_datasets = _supplemental_grouped_person_eval_datasets(cfg)
    run_primary_grouped_person_eval = bool(primary_grouped_dataset_name)
    datasets_requiring_grouped_records = set(supplemental_grouped_datasets)
    if primary_grouped_dataset_name:
        datasets_requiring_grouped_records.add(primary_grouped_dataset_name)
    collect_grouped_records = bool(datasets_requiring_grouped_records)
    generation_kwargs = _make_generation_kwargs(metric_processor, cfg)
    prediction_log_limit = _eval_prediction_log_limit(cfg)

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_bar):
            metadata, supervised_batch, generation_batch = _split_eval_batch(batch, device)
            dataset_names = metadata.get("dataset_names", [])
            segment_keys = metadata.get("segment_keys", [])
            group_ids = metadata.get("group_ids", [])
            target_texts = metadata.get("target_texts", [])

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**supervised_batch)
                loss = outputs.loss
                _maybe_log_model_io_debug(
                    model,
                    logger,
                    cfg,
                    "eval",
                    metric_processor,
                    supervised_batch,
                    outputs.logits,
                    rank=rank,
                    epoch=epoch,
                    step=step,
                    event_index=eval_event_index,
                    loss=float(loss.item()),
                )
                if collect_grouped_records:
                    batch_teacher_forced_records = build_grouped_eval_records(
                        metric_processor.tokenizer,
                        outputs.logits,
                        supervised_batch["labels"],
                        dataset_names,
                        segment_keys,
                        group_ids,
                        target_texts,
                    )
                    for grouped_dataset_name in datasets_requiring_grouped_records:
                        teacher_forced_grouped_local_records[grouped_dataset_name].extend(
                            batch_teacher_forced_records.get(grouped_dataset_name, [])
                        )
                teacher_forced_segment_stats = compute_metrics_text_binary_accumulate(
                    metric_processor,
                    outputs.logits,
                    supervised_batch["labels"],
                    teacher_forced_segment_stats,
                )
                generated_ids = (model.module if hasattr(model, "module") else model).generate(
                    **generation_batch,
                    **generation_kwargs,
                )

            prompt_length = generation_batch["input_ids"].shape[1]
            generated_suffix = generated_ids[:, prompt_length:]
            generated_texts = metric_processor.tokenizer.batch_decode(
                generated_suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for sample_index, generated_text in enumerate(generated_texts):
                dataset_name = dataset_names[sample_index] if sample_index < len(dataset_names) else ""
                target_text = target_texts[sample_index] if sample_index < len(target_texts) else ""
                segment_key = segment_keys[sample_index] if sample_index < len(segment_keys) else None
                group_id = group_ids[sample_index] if sample_index < len(group_ids) else None

                y_true = map_target_text_to_binary(target_text)
                parse = parse_generated_label(generated_text)
                y_pred = parse["label"]

                _update_generation_stats(generation_segment_stats, y_true, y_pred)
                if dataset_name not in generation_per_dataset_segment_stats:
                    generation_per_dataset_segment_stats[dataset_name] = make_binary_stats()
                    generation_per_dataset_segment_stats[dataset_name]["num_invalid_predictions"] = 0
                _update_generation_stats(generation_per_dataset_segment_stats[dataset_name], y_true, y_pred)

                record = {
                    "dataset": dataset_name,
                    "segment_key": segment_key,
                    "participant_id": group_id,
                    "target_text": target_text,
                    "true_label": _label_id_to_name(y_true),
                    "parsed_label": _label_id_to_name(y_pred),
                    "raw_generated_text": generated_text,
                    "normalized_generated_text": parse["normalized_text"],
                    "matched_pattern": parse["matched_pattern"],
                    "ambiguous": bool(parse["ambiguous"]),
                    "parse_reason": parse["parse_reason"],
                }
                if prediction_log_limit <= 0 or len(local_segment_prediction_examples) < prediction_log_limit:
                    local_segment_prediction_examples.append(record)

                if collect_grouped_records and grouped_eval_enabled(dataset_name) and segment_key and group_id:
                    generation_grouped_local_records[dataset_name].append(
                        {
                            "key": segment_key,
                            "group_id": group_id,
                            "target_text": target_text,
                            "pred_label": y_pred,
                            "depressed_probability": 1.0 if y_pred == 1 else 0.0,
                            "raw_generated_text": generated_text,
                            "parse_reason": parse["parse_reason"],
                            "ambiguous": bool(parse["ambiguous"]),
                        }
                    )

            eval_loss += loss.item()
            eval_steps += 1
            if rank == 0 and generation_segment_stats["total"] > 0:
                temp_acc, _, _, temp_f1, temp_wf1 = compute_metrics_from_stats(generation_segment_stats)
                eval_bar.set_description(
                    f"[Eval] gen_loss {loss:.3f} | gen_acc {temp_acc:.4f} | gen_posF1 {temp_f1:.4f} | gen_wF1 {temp_wf1:.4f}"
                )

    loss_tensor = torch.tensor([eval_loss, float(eval_steps)], device=device, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    reduced_eval_loss = (loss_tensor[0] / loss_tensor[1]).item() if loss_tensor[1] > 0 else 0.0

    generation_per_dataset_segment_stats = _gather_and_merge_stats_by_dataset(generation_per_dataset_segment_stats)
    generation_segment_stats = _merge_stats_dicts(
        None,
        _gather_and_merge_stats_by_dataset({"__overall__": generation_segment_stats}).get("__overall__", make_binary_stats()),
    )

    teacher_forced_segment_stats = _merge_stats_dicts(
        None,
        _gather_and_merge_stats_by_dataset({"__overall__": teacher_forced_segment_stats or make_binary_stats()}).get(
            "__overall__", make_binary_stats()
        ),
    )
    segment_prediction_examples = _gather_limited_examples(local_segment_prediction_examples, prediction_log_limit)

    generation_supplemental_stats_by_dataset = {}
    generation_primary_stats = dict(generation_segment_stats)
    generation_segment_level_stats_by_dataset = dict(generation_per_dataset_segment_stats)
    teacher_forced_supplemental_stats_by_dataset = {}
    teacher_forced_primary_stats = dict(teacher_forced_segment_stats)
    participant_predictions_by_dataset = {}
    if collect_grouped_records:
        gathered_generation_records = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_generation_records, generation_grouped_local_records)
        merged_generation_records_by_dataset = {dataset_name: [] for dataset_name in datasets_requiring_grouped_records}
        for record_group in gathered_generation_records:
            for grouped_dataset_name in datasets_requiring_grouped_records:
                merged_generation_records_by_dataset[grouped_dataset_name].extend(
                    (record_group or {}).get(grouped_dataset_name, [])
                )

        (
            generation_per_dataset_after_person,
            generation_overall_after_person,
            generation_supplemental_stats_by_dataset,
            participant_predictions_by_dataset,
        ) = apply_generated_person_level_results(
            dataset_name=getattr(cfg.data, "dataset_name", ""),
            level_by_dataset={dataset_name: _grouped_eval_level(cfg, dataset_name) for dataset_name in GROUPED_DATASET_NAMES},
            mode_by_dataset={dataset_name: _grouped_eval_mode(cfg, dataset_name) for dataset_name in GROUPED_DATASET_NAMES},
            threshold_by_dataset={
                dataset_name: _grouped_eval_threshold(cfg, dataset_name) for dataset_name in GROUPED_DATASET_NAMES
            },
            per_dataset_stats=dict(generation_per_dataset_segment_stats),
            overall_stats=dict(generation_segment_stats),
            grouped_records_by_dataset=merged_generation_records_by_dataset,
        )
        generation_primary_stats = generation_overall_after_person

        gathered_teacher_forced_records = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_teacher_forced_records, teacher_forced_grouped_local_records)
        merged_teacher_forced_records_by_dataset = {dataset_name: [] for dataset_name in datasets_requiring_grouped_records}
        for record_group in gathered_teacher_forced_records:
            for grouped_dataset_name in datasets_requiring_grouped_records:
                merged_teacher_forced_records_by_dataset[grouped_dataset_name].extend(
                    (record_group or {}).get(grouped_dataset_name, [])
                )

        for grouped_dataset_name, merged_records in merged_teacher_forced_records_by_dataset.items():
            stats = aggregate_group_predictions(
                merged_records,
                mode=_grouped_eval_mode(cfg, grouped_dataset_name),
                threshold=_grouped_eval_threshold(cfg, grouped_dataset_name),
            )
            if grouped_dataset_name == primary_grouped_dataset_name:
                teacher_forced_primary_stats = stats
            elif grouped_dataset_name in supplemental_grouped_datasets:
                teacher_forced_supplemental_stats_by_dataset[grouped_dataset_name] = stats

    participant_prediction_examples = {
        dataset_name: predictions if prediction_log_limit <= 0 else predictions[:prediction_log_limit]
        for dataset_name, predictions in participant_predictions_by_dataset.items()
    }

    return {
        "eval_loss": reduced_eval_loss,
        "generation": {
            "primary_stats": generation_primary_stats,
            "supplemental_stats_by_dataset": generation_supplemental_stats_by_dataset,
            "segment_level_overall_stats": generation_segment_stats,
            "segment_level_stats_by_dataset": generation_segment_level_stats_by_dataset,
            "segment_prediction_examples": segment_prediction_examples,
            "participant_prediction_examples": participant_prediction_examples,
            "participant_predictions_by_dataset": participant_predictions_by_dataset,
        },
        "teacher_forced": {
            "primary_stats": teacher_forced_primary_stats,
            "supplemental_stats_by_dataset": teacher_forced_supplemental_stats_by_dataset,
            "segment_level_overall_stats": teacher_forced_segment_stats,
        },
    }


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
    eval_dataset = TextOnlyEvalDatasetWithMeta(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        task_filename=cfg.data.eval_task_filename,
        default_dataset_name=cfg.data.dataset_name,
    )
    eval_collate = partial(collate_fn_textonly_eval_dual, tokenizer=tokenizer)

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
        collate_fn=eval_collate,
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    if rank == 0:
        reset_peak_memory_stats(cfg.env.device_type, device)

    metric_processor = TokenizerWrapper(tokenizer)
    steps_per_epoch = len(train_dataloader)
    dynamic_eval_step = max(1, steps_per_epoch // 10)

    if rank == 0:
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))
        micro_batch = cfg.train.batch_size * world_size
        true_eff_batch = micro_batch * grad_acc_steps
        opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_acc_steps)

        logger.info("\n[Dynamic Eval Settings]")
        logger.info("  Total training samples: %s", len(train_dataset))
        logger.info("  Micro-batch per GPU: %s", cfg.train.batch_size)
        logger.info("  World size (GPUs): %s", world_size)
        logger.info("  Gradient Accumulation steps: %s", grad_acc_steps)
        logger.info("  TRUE Effective batch size: %s", true_eff_batch)
        logger.info("  Dataloader steps (Forward passes): %s", steps_per_epoch)
        logger.info("  Optimizer steps (Weight updates): %s", opt_steps_per_epoch)
        logger.info("  Dynamic eval_step: %s (evaluate ~10x per epoch)", dynamic_eval_step)
        logger.info("")

    best_f1 = -math.inf
    global_train_step = 0
    optimizer_step = 0
    eval_step_idx = 0

    for epoch in range(cfg.train.train_epoch):
        train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}") if rank == 0 else train_dataloader
        model.train()
        optim.zero_grad()
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            _check_stop_requested(getattr(cfg.env, "stop_file", ""))
            global_train_step += 1
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(metric_processor, outputs.logits, batch["labels"])
                _maybe_log_model_io_debug(
                    model,
                    logger,
                    cfg,
                    "train",
                    metric_processor,
                    batch,
                    outputs.logits,
                    rank=rank,
                    epoch=epoch,
                    step=train_step,
                    event_index=global_train_step,
                    loss=float(loss.item()),
                )

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
                next_eval_event_index = eval_step_idx + 1
                eval_results = _evaluate(
                    model,
                    eval_dataloader,
                    device,
                    metric_processor,
                    rank,
                    cfg,
                    logger=logger,
                    epoch=epoch,
                    step=train_step,
                    eval_event_index=next_eval_event_index,
                )
                if rank == 0:
                    eval_step_idx += 1
                    eval_loss = eval_results["eval_loss"]
                    generation_stats = eval_results["generation"]["primary_stats"]
                    generation_supplemental_stats_by_dataset = eval_results["generation"]["supplemental_stats_by_dataset"]
                    teacher_forced_stats = eval_results["teacher_forced"]["primary_stats"]
                    teacher_forced_supplemental_stats_by_dataset = eval_results["teacher_forced"]["supplemental_stats_by_dataset"]
                    eval_accuracy, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(
                        generation_stats
                    )
                    tf_accuracy, tf_precision, tf_recall, tf_f1, tf_wf1 = compute_metrics_from_stats(
                        teacher_forced_stats
                    )
                    _report_progress(getattr(cfg.env, "progress_file", ""), eval_step_idx, eval_f1)
                    grouped_dataset_name = _primary_grouped_person_eval_dataset(cfg)
                    metric_scope = f"Person ({grouped_dataset_name})" if grouped_dataset_name else "Segment"
                    logger.info("[Epoch %s Step %s] generation_eval (%s):", epoch, train_step, metric_scope)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        eval_accuracy,
                        eval_precision,
                        eval_recall,
                        eval_f1,
                        eval_wf1,
                    )
                    if grouped_dataset_name:
                        logger.info(
                            "  Participants: %s, Unique Segments: %s, Mode: %s, Threshold: %.4f",
                            generation_stats.get("num_participants", 0),
                            generation_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, grouped_dataset_name),
                            _grouped_eval_threshold(cfg, grouped_dataset_name),
                        )
                    logger.info("[Epoch %s Step %s] teacher_forced_diagnostic (%s):", epoch, train_step, metric_scope)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        tf_accuracy,
                        tf_precision,
                        tf_recall,
                        tf_f1,
                        tf_wf1,
                    )
                    logger.info("  Generation minus teacher-forced F1 delta: %.4f", eval_f1 - tf_f1)
                    for supplemental_dataset_name, supplemental_stats in generation_supplemental_stats_by_dataset.items():
                        sup_acc, sup_prec, sup_rec, sup_f1, sup_wf1 = compute_metrics_from_stats(supplemental_stats)
                        logger.info(
                            "  Supplemental generation %s Person Eval: Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f | wF1 %.4f | Participants %s | Segments %s | Mode %s | Threshold %.4f",
                            supplemental_dataset_name,
                            sup_acc,
                            sup_prec,
                            sup_rec,
                            sup_f1,
                            sup_wf1,
                            supplemental_stats.get("num_participants", 0),
                            supplemental_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, supplemental_dataset_name),
                            _grouped_eval_threshold(cfg, supplemental_dataset_name),
                        )
                    for supplemental_dataset_name, supplemental_stats in teacher_forced_supplemental_stats_by_dataset.items():
                        sup_acc, sup_prec, sup_rec, sup_f1, sup_wf1 = compute_metrics_from_stats(supplemental_stats)
                        logger.info(
                            "  Supplemental teacher-forced %s Person Eval: Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f | wF1 %.4f | Participants %s | Segments %s | Mode %s | Threshold %.4f",
                            supplemental_dataset_name,
                            sup_acc,
                            sup_prec,
                            sup_rec,
                            sup_f1,
                            sup_wf1,
                            supplemental_stats.get("num_participants", 0),
                            supplemental_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, supplemental_dataset_name),
                            _grouped_eval_threshold(cfg, supplemental_dataset_name),
                        )
                    _log_prediction_examples(
                        logger,
                        "[Eval] Segment Predictions",
                        eval_results["generation"]["segment_prediction_examples"],
                        _eval_prediction_log_limit(cfg),
                    )
                    for grouped_dataset_name_for_examples, participant_examples in sorted(
                        eval_results["generation"]["participant_prediction_examples"].items()
                    ):
                        _log_prediction_examples(
                            logger,
                            f"[Eval] Participant Predictions ({grouped_dataset_name_for_examples})",
                            participant_examples,
                            _eval_prediction_log_limit(cfg),
                        )

                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        teacher_forced_summary = _build_best_eval_summary(
                            cfg,
                            eval_loss,
                            teacher_forced_stats,
                            teacher_forced_supplemental_stats_by_dataset,
                            eval_method="teacher_forced_diagnostic",
                            segment_level_overall_stats=eval_results["teacher_forced"]["segment_level_overall_stats"],
                        )
                        cfg.env.best_teacher_forced_eval_summary = teacher_forced_summary
                        cfg.env.best_eval_summary = _build_best_eval_summary(
                            cfg,
                            eval_loss,
                            generation_stats,
                            generation_supplemental_stats_by_dataset,
                            eval_method="prompt_only_generation",
                            segment_level_overall_stats=eval_results["generation"]["segment_level_overall_stats"],
                            segment_level_stats_by_dataset=eval_results["generation"]["segment_level_stats_by_dataset"],
                            segment_prediction_examples=eval_results["generation"]["segment_prediction_examples"],
                            participant_predictions_by_dataset=eval_results["generation"]["participant_prediction_examples"],
                            teacher_forced_summary=teacher_forced_summary,
                        )
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
                                    "eval/teacher_forced_f1": tf_f1,
                                    "eval/best_f1": best_f1,
                                    "train/epoch": epoch,
                                    **{
                                        f"eval/{grouped_person_results_key(supplemental_dataset_name)}/f1": float(
                                            compute_metrics_from_stats(supplemental_stats)[3]
                                        )
                                        for supplemental_dataset_name, supplemental_stats in generation_supplemental_stats_by_dataset.items()
                                    },
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
    

    train_dataset = AudioDataset(
        cfg.data.train_data_path,
        cfg.data.train_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.train_scp_filename,
        task_filename=cfg.data.train_task_filename,
    )
    eval_dataset = AudioEvalDatasetWithMeta(
        cfg.data.eval_data_path,
        cfg.data.val_prompt_path,
        cfg.data.wav_type,
        scp_filename=cfg.data.eval_scp_filename,
        task_filename=cfg.data.eval_task_filename,
        default_dataset_name=cfg.data.dataset_name,
    )
    eval_collate = partial(collate_fn_qwen2audio_eval_dual, processor=processor)

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
        collate_fn=eval_collate,
        sampler=eval_sampler,
        prefetch_factor=cfg.data.prefetch_factor,
    )

    # ==========================================
    # PASTE THE NEW SCHEDULER LOGIC HERE
    # ==========================================
    steps_per_epoch = len(train_dataloader)
    grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))
    opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_acc_steps)
    
    true_total_steps = opt_steps_per_epoch * cfg.train.train_epoch
    true_warmup_steps = max(1, int(0.10 * true_total_steps))
    
    cfg.train.total_train_steps = true_total_steps
    cfg.train.warmup_steps = true_warmup_steps

    if rank == 0:
        logger.info(f"Fixed Scheduler: Total Steps = {true_total_steps}, Warmup = {true_warmup_steps}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.lr)
    scheduler = _build_scheduler(optim, cfg)
    # ==========================================


    steps_per_epoch = len(train_dataloader)
    dynamic_eval_step = max(1, steps_per_epoch // 10)

    if rank == 0:
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))
        micro_batch = cfg.train.batch_size * world_size
        true_eff_batch = micro_batch * grad_acc_steps
        opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_acc_steps)

        logger.info("\n[Dynamic Eval Settings]")
        logger.info("  Total training samples: %s", len(train_dataset))
        logger.info("  Micro-batch per GPU: %s", cfg.train.batch_size)
        logger.info("  World size (GPUs): %s", world_size)
        logger.info("  Gradient Accumulation steps: %s", grad_acc_steps)
        logger.info("  TRUE Effective batch size: %s", true_eff_batch)
        logger.info("  Dataloader steps (Forward passes): %s", steps_per_epoch)
        logger.info("  Optimizer steps (Weight updates): %s", opt_steps_per_epoch)
        logger.info("  Dynamic eval_step: %s (evaluate ~10x per epoch)", dynamic_eval_step)
        logger.info("")

    best_f1 = -math.inf
    global_train_step = 0
    eval_step_idx = 0

    for epoch in range(cfg.train.train_epoch):
        train_sampler.set_epoch(epoch)
        train_bar = tqdm(train_dataloader, desc=f"[Train] epoch: {epoch}") if rank == 0 else train_dataloader
        model.train()
        optim.zero_grad(set_to_none=True)
        grad_acc_steps = max(1, int(cfg.train.grad_accumulate_step))

        for train_step, batch in enumerate(train_bar):
            _check_stop_requested(getattr(cfg.env, "stop_file", ""))
            global_train_step += 1
            batch = batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
                acc = compute_acc_text(processor, outputs.logits, batch["labels"])
                _maybe_log_model_io_debug(
                    model,
                    logger,
                    cfg,
                    "train",
                    processor,
                    batch,
                    outputs.logits,
                    rank=rank,
                    epoch=epoch,
                    step=train_step,
                    event_index=global_train_step,
                    loss=float(loss.item()),
                )

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
                next_eval_event_index = eval_step_idx + 1
                eval_results = _evaluate(
                    model,
                    eval_dataloader,
                    device,
                    processor,
                    rank,
                    cfg,
                    logger=logger,
                    epoch=epoch,
                    step=train_step,
                    eval_event_index=next_eval_event_index,
                )
                if rank == 0:
                    eval_step_idx += 1
                    eval_loss = eval_results["eval_loss"]
                    generation_stats = eval_results["generation"]["primary_stats"]
                    generation_supplemental_stats_by_dataset = eval_results["generation"]["supplemental_stats_by_dataset"]
                    teacher_forced_stats = eval_results["teacher_forced"]["primary_stats"]
                    teacher_forced_supplemental_stats_by_dataset = eval_results["teacher_forced"]["supplemental_stats_by_dataset"]
                    eval_accuracy, eval_precision, eval_recall, eval_f1, eval_wf1 = compute_metrics_from_stats(
                        generation_stats
                    )
                    tf_accuracy, tf_precision, tf_recall, tf_f1, tf_wf1 = compute_metrics_from_stats(
                        teacher_forced_stats
                    )
                    _report_progress(getattr(cfg.env, "progress_file", ""), eval_step_idx, eval_f1)
                    grouped_dataset_name = _primary_grouped_person_eval_dataset(cfg)
                    metric_scope = f"Person ({grouped_dataset_name})" if grouped_dataset_name else "Segment"
                    logger.info("[Epoch %s Step %s] generation_eval (%s):", epoch, train_step, metric_scope)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        eval_accuracy,
                        eval_precision,
                        eval_recall,
                        eval_f1,
                        eval_wf1,
                    )
                    if grouped_dataset_name:
                        logger.info(
                            "  Participants: %s, Unique Segments: %s, Mode: %s, Threshold: %.4f",
                            generation_stats.get("num_participants", 0),
                            generation_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, grouped_dataset_name),
                            _grouped_eval_threshold(cfg, grouped_dataset_name),
                        )
                    logger.info("[Epoch %s Step %s] teacher_forced_diagnostic (%s):", epoch, train_step, metric_scope)
                    logger.info(
                        "  Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, wF1: %.4f",
                        eval_loss,
                        tf_accuracy,
                        tf_precision,
                        tf_recall,
                        tf_f1,
                        tf_wf1,
                    )
                    logger.info("  Generation minus teacher-forced F1 delta: %.4f", eval_f1 - tf_f1)
                    for supplemental_dataset_name, supplemental_stats in generation_supplemental_stats_by_dataset.items():
                        sup_acc, sup_prec, sup_rec, sup_f1, sup_wf1 = compute_metrics_from_stats(supplemental_stats)
                        logger.info(
                            "  Supplemental generation %s Person Eval: Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f | wF1 %.4f | Participants %s | Segments %s | Mode %s | Threshold %.4f",
                            supplemental_dataset_name,
                            sup_acc,
                            sup_prec,
                            sup_rec,
                            sup_f1,
                            sup_wf1,
                            supplemental_stats.get("num_participants", 0),
                            supplemental_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, supplemental_dataset_name),
                            _grouped_eval_threshold(cfg, supplemental_dataset_name),
                        )
                    for supplemental_dataset_name, supplemental_stats in teacher_forced_supplemental_stats_by_dataset.items():
                        sup_acc, sup_prec, sup_rec, sup_f1, sup_wf1 = compute_metrics_from_stats(supplemental_stats)
                        logger.info(
                            "  Supplemental teacher-forced %s Person Eval: Acc %.4f | Prec %.4f | Rec %.4f | F1 %.4f | wF1 %.4f | Participants %s | Segments %s | Mode %s | Threshold %.4f",
                            supplemental_dataset_name,
                            sup_acc,
                            sup_prec,
                            sup_rec,
                            sup_f1,
                            sup_wf1,
                            supplemental_stats.get("num_participants", 0),
                            supplemental_stats.get("num_segments", 0),
                            _grouped_eval_mode(cfg, supplemental_dataset_name),
                            _grouped_eval_threshold(cfg, supplemental_dataset_name),
                        )
                    _log_prediction_examples(
                        logger,
                        "[Eval] Segment Predictions",
                        eval_results["generation"]["segment_prediction_examples"],
                        _eval_prediction_log_limit(cfg),
                    )
                    for grouped_dataset_name_for_examples, participant_examples in sorted(
                        eval_results["generation"]["participant_prediction_examples"].items()
                    ):
                        _log_prediction_examples(
                            logger,
                            f"[Eval] Participant Predictions ({grouped_dataset_name_for_examples})",
                            participant_examples,
                            _eval_prediction_log_limit(cfg),
                        )

                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        teacher_forced_summary = _build_best_eval_summary(
                            cfg,
                            eval_loss,
                            teacher_forced_stats,
                            teacher_forced_supplemental_stats_by_dataset,
                            eval_method="teacher_forced_diagnostic",
                            segment_level_overall_stats=eval_results["teacher_forced"]["segment_level_overall_stats"],
                        )
                        cfg.env.best_teacher_forced_eval_summary = teacher_forced_summary
                        cfg.env.best_eval_summary = _build_best_eval_summary(
                            cfg,
                            eval_loss,
                            generation_stats,
                            generation_supplemental_stats_by_dataset,
                            eval_method="prompt_only_generation",
                            segment_level_overall_stats=eval_results["generation"]["segment_level_overall_stats"],
                            segment_level_stats_by_dataset=eval_results["generation"]["segment_level_stats_by_dataset"],
                            segment_prediction_examples=eval_results["generation"]["segment_prediction_examples"],
                            participant_predictions_by_dataset=eval_results["generation"]["participant_prediction_examples"],
                            teacher_forced_summary=teacher_forced_summary,
                        )
                        logger.info("[New Best F1] %.4f", eval_f1)
                        best_model_path = os.path.join(cfg.env.save_path, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        model.module.save_pretrained(best_model_path)
                        processor.save_pretrained(best_model_path)
                        adapter_state_path = save_audio_adapter_state(model.module, best_model_path)
                        logger.info("[Saved Best Model] -> %s", best_model_path)
                        logger.info("[Saved Audio Adapter] -> %s", adapter_state_path)

                        if wandb_logger:
                            wandb_logger.log(
                                {
                                    "eval/loss": eval_loss,
                                    "eval/f1": eval_f1,
                                    "eval/teacher_forced_f1": tf_f1,
                                    "eval/best_f1": best_f1,
                                    "train/epoch": epoch,
                                    **{
                                        f"eval/{grouped_person_results_key(supplemental_dataset_name)}/f1": float(
                                            compute_metrics_from_stats(supplemental_stats)[3]
                                        )
                                        for supplemental_dataset_name, supplemental_stats in generation_supplemental_stats_by_dataset.items()
                                    },
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
