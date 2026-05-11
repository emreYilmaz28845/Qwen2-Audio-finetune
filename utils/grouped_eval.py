import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F


GROUPED_DATASET_NAMES = {"daic_woz", "eatd", "cmdc"}
SUPPORTED_GROUPED_EVAL_MODES = {
    "majority_vote",
    "mean_probability",
    "max_probability",
}
SUPPORTED_GROUPED_EVAL_LEVELS = {
    "segment",
    "person",
}

GROUPED_DEPRESSED_LABEL = "抑郁"
GROUPED_NON_DEPRESSED_LABEL = "非抑郁"
_DAIC_PARTICIPANT_ID_PATTERN = re.compile(r"^(?P<participant_id>\d+)")
_EATD_SUBJECT_ID_PATTERN = re.compile(r"^(?P<subject_id>[^_]+)_")
_CMDC_SUBJECT_ID_PATTERN = re.compile(r"^(?P<subject_id>[^_]+)_Q\d+$")


def make_binary_stats():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0}


def grouped_person_results_key(dataset_name: str):
    return f"{dataset_name}_person"


def normalize_grouped_eval_mode(mode: str):
    normalized = (mode or "majority_vote").strip().lower()
    if normalized not in SUPPORTED_GROUPED_EVAL_MODES:
        raise ValueError(
            f"Unsupported grouped_eval_mode={mode!r}. "
            f"Expected one of {sorted(SUPPORTED_GROUPED_EVAL_MODES)}."
        )
    return normalized


def normalize_grouped_eval_level(level: str):
    normalized = (level or "person").strip().lower()
    if normalized not in SUPPORTED_GROUPED_EVAL_LEVELS:
        raise ValueError(
            f"Unsupported grouped_eval_level={level!r}. "
            f"Expected one of {sorted(SUPPORTED_GROUPED_EVAL_LEVELS)}."
        )
    return normalized


def validate_grouped_person_threshold(threshold: float):
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Invalid grouped_person_threshold={threshold!r}. Expected a value in [0.0, 1.0]."
        )
    return value


def grouped_eval_env_prefix(dataset_name: str):
    return dataset_name.upper()


def extract_group_id(dataset_name: str, segment_key: str):
    key = (segment_key or "").strip()
    if dataset_name == "daic_woz":
        match = _DAIC_PARTICIPANT_ID_PATTERN.match(key)
    elif dataset_name == "eatd":
        match = _EATD_SUBJECT_ID_PATTERN.match(key)
    elif dataset_name == "cmdc":
        match = _CMDC_SUBJECT_ID_PATTERN.match(key)
    else:
        raise ValueError(f"Unsupported grouped dataset: {dataset_name!r}")

    if match is None:
        raise ValueError(f"Could not extract group ID for dataset={dataset_name!r} key={segment_key!r}.")
    return match.group(1)


def grouped_eval_enabled(dataset_name: str):
    return dataset_name in GROUPED_DATASET_NAMES


def build_grouped_task_metadata(task: dict, default_dataset_name: str = "unknown"):
    dataset_name = task.get("dataset", default_dataset_name)
    raw_key = task.get("key", "")
    source_key = task.get("source_key") or raw_key
    target_text = task.get("target", "")

    segment_key = None
    group_id = None
    if grouped_eval_enabled(dataset_name):
        segment_key = source_key
        group_id = extract_group_id(dataset_name, segment_key)

    return {
        "dataset_name": dataset_name,
        "raw_key": raw_key,
        "source_key": source_key,
        "target_text": target_text,
        "segment_key": segment_key,
        "group_id": group_id,
    }


def map_target_text_to_binary(target_text: str):
    text = (target_text or "").strip()
    if text == GROUPED_NON_DEPRESSED_LABEL or GROUPED_NON_DEPRESSED_LABEL in text or "健康" in text:
        return 0
    if text == GROUPED_DEPRESSED_LABEL or GROUPED_DEPRESSED_LABEL in text:
        return 1
    return -1


def get_grouped_label_token_ids(tokenizer):
    depressed_ids = tokenizer.encode(GROUPED_DEPRESSED_LABEL, add_special_tokens=False)
    non_depressed_ids = tokenizer.encode(GROUPED_NON_DEPRESSED_LABEL, add_special_tokens=False)
    if not depressed_ids or not non_depressed_ids:
        raise ValueError("Failed to tokenize grouped class labels.")
    return depressed_ids, non_depressed_ids


def score_candidate_sequence(logits_row: torch.Tensor, start_pred_index: int, candidate_token_ids):
    if logits_row.ndim != 2:
        raise ValueError(f"Expected logits_row to have shape [T, V], got {tuple(logits_row.shape)}.")
    if not candidate_token_ids:
        return float("-inf")

    start_index = max(int(start_pred_index), 0)
    end_index = start_index + len(candidate_token_ids)
    if end_index > logits_row.size(0):
        return float("-inf")

    log_probs = F.log_softmax(logits_row.float(), dim=-1)
    position_indices = torch.arange(start_index, end_index, device=logits_row.device)
    token_indices = torch.tensor(candidate_token_ids, device=logits_row.device, dtype=torch.long)
    return float(log_probs[position_indices, token_indices].sum().item())


def compute_segment_depressed_probability(
    logits_row: torch.Tensor,
    start_pred_index: int,
    depressed_token_ids,
    non_depressed_token_ids,
):
    depressed_score = score_candidate_sequence(logits_row, start_pred_index, depressed_token_ids)
    non_depressed_score = score_candidate_sequence(logits_row, start_pred_index, non_depressed_token_ids)

    if math.isinf(depressed_score) and math.isinf(non_depressed_score):
        return 0.5

    max_score = max(depressed_score, non_depressed_score)
    depressed_exp = math.exp(depressed_score - max_score)
    non_depressed_exp = math.exp(non_depressed_score - max_score)
    normalizer = depressed_exp + non_depressed_exp
    if normalizer <= 0.0:
        return 0.5
    return depressed_exp / normalizer


def build_grouped_eval_records(
    tokenizer,
    logits: torch.Tensor,
    labels: torch.Tensor,
    dataset_names,
    segment_keys,
    group_ids,
    target_texts,
):
    depressed_token_ids, non_depressed_token_ids = get_grouped_label_token_ids(tokenizer)
    records = defaultdict(list)

    batch_size = labels.size(0)
    for sample_index in range(batch_size):
        dataset_name = dataset_names[sample_index]
        if not grouped_eval_enabled(dataset_name):
            continue
        if not segment_keys[sample_index] or not group_ids[sample_index]:
            continue

        valid_label_positions = (labels[sample_index] != -100).nonzero(as_tuple=False).squeeze(-1)
        if valid_label_positions.numel() == 0:
            continue

        start_pred_index = int(valid_label_positions[0].item()) - 1
        depressed_probability = compute_segment_depressed_probability(
            logits[sample_index],
            start_pred_index,
            depressed_token_ids,
            non_depressed_token_ids,
        )
        records[dataset_name].append(
            {
                "key": segment_keys[sample_index],
                "group_id": group_ids[sample_index],
                "target_text": target_texts[sample_index],
                "depressed_probability": float(depressed_probability),
            }
        )

    return records


def _update_binary_stats(stats: dict, y_true: int, y_pred: int):
    stats["total"] += 1
    if y_true == y_pred:
        stats["correct"] += 1
        if y_true == 1:
            stats["tp"] += 1
        else:
            stats["tn"] += 1
        return

    if y_true == 1 and y_pred == 0:
        stats["fn"] += 1
    elif y_true == 0 and y_pred == 1:
        stats["fp"] += 1


def aggregate_group_predictions(records, mode: str, threshold: float):
    normalized_mode = normalize_grouped_eval_mode(mode)
    normalized_threshold = validate_grouped_person_threshold(threshold)

    deduped_records = {}
    for record in records:
        key = record["key"]
        deduped_records.setdefault(key, record)

    grouped_records = defaultdict(list)
    for record in deduped_records.values():
        grouped_records[record["group_id"]].append(record)

    stats = make_binary_stats()

    for _, group_records in grouped_records.items():
        labels = {map_target_text_to_binary(record["target_text"]) for record in group_records}
        labels.discard(-1)
        if len(labels) != 1:
            raise ValueError("Inconsistent or missing target labels within a grouped participant/session group.")
        y_true = labels.pop()

        depressed_probabilities = [float(record["depressed_probability"]) for record in group_records]
        mean_probability = sum(depressed_probabilities) / len(depressed_probabilities)

        if normalized_mode == "majority_vote":
            depressed_votes = sum(probability >= normalized_threshold for probability in depressed_probabilities)
            non_depressed_votes = len(depressed_probabilities) - depressed_votes
            if depressed_votes > non_depressed_votes:
                y_pred = 1
            elif depressed_votes < non_depressed_votes:
                y_pred = 0
            else:
                y_pred = 1 if mean_probability >= normalized_threshold else 0
        elif normalized_mode == "mean_probability":
            y_pred = 1 if mean_probability >= normalized_threshold else 0
        else:
            y_pred = 1 if max(depressed_probabilities) >= normalized_threshold else 0

        _update_binary_stats(stats, y_true, y_pred)

    stats["num_segments"] = len(deduped_records)
    stats["num_participants"] = len(grouped_records)
    return stats


def apply_person_level_results(
    dataset_name: str,
    level_by_dataset: dict,
    mode_by_dataset: dict,
    threshold_by_dataset: dict,
    per_dataset_stats: dict,
    overall_stats: dict,
    grouped_records_by_dataset: dict,
):
    normalized_dataset_name = (dataset_name or "").strip().lower()
    next_per_dataset_stats = dict(per_dataset_stats)
    next_overall_stats = dict(overall_stats)
    supplemental_results = {}

    for grouped_dataset_name, records in grouped_records_by_dataset.items():
        if not records:
            continue
        if normalize_grouped_eval_level(level_by_dataset.get(grouped_dataset_name, "segment")) != "person":
            continue

        person_stats = aggregate_group_predictions(
            records,
            mode=mode_by_dataset.get(grouped_dataset_name, "majority_vote"),
            threshold=threshold_by_dataset.get(grouped_dataset_name, 0.5),
        )
        if normalized_dataset_name == grouped_dataset_name:
            next_per_dataset_stats[grouped_dataset_name] = dict(person_stats)
            next_overall_stats = dict(person_stats)
        elif normalized_dataset_name == "merged":
            supplemental_results[grouped_person_results_key(grouped_dataset_name)] = dict(person_stats)

    return next_per_dataset_stats, next_overall_stats, supplemental_results
