import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F


DAIC_DATASET_NAME = "daic_woz"
DAIC_PERSON_RESULTS_KEY = "daic_woz_person"

DAIC_EVAL_MODE_MAJORITY_VOTE = "majority_vote"
DAIC_EVAL_MODE_MEAN_PROBABILITY = "mean_probability"
DAIC_EVAL_MODE_MAX_PROBABILITY = "max_probability"
SUPPORTED_DAIC_EVAL_MODES = {
    DAIC_EVAL_MODE_MAJORITY_VOTE,
    DAIC_EVAL_MODE_MEAN_PROBABILITY,
    DAIC_EVAL_MODE_MAX_PROBABILITY,
}
DAIC_EVAL_LEVEL_SEGMENT = "segment"
DAIC_EVAL_LEVEL_PERSON = "person"
SUPPORTED_DAIC_EVAL_LEVELS = {
    DAIC_EVAL_LEVEL_SEGMENT,
    DAIC_EVAL_LEVEL_PERSON,
}

DAIC_DEPRESSED_LABEL = "抑郁"
DAIC_NON_DEPRESSED_LABEL = "非抑郁"
_PARTICIPANT_ID_PATTERN = re.compile(r"^(?P<participant_id>\d+)")


def make_binary_stats():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0}


def normalize_daic_eval_mode(mode: str):
    normalized = (mode or DAIC_EVAL_MODE_MAJORITY_VOTE).strip().lower()
    if normalized not in SUPPORTED_DAIC_EVAL_MODES:
        raise ValueError(
            f"Unsupported daic_eval_mode={mode!r}. "
            f"Expected one of {sorted(SUPPORTED_DAIC_EVAL_MODES)}."
        )
    return normalized


def normalize_daic_eval_level(level: str):
    normalized = (level or DAIC_EVAL_LEVEL_PERSON).strip().lower()
    if normalized not in SUPPORTED_DAIC_EVAL_LEVELS:
        raise ValueError(
            f"Unsupported daic_eval_level={level!r}. "
            f"Expected one of {sorted(SUPPORTED_DAIC_EVAL_LEVELS)}."
        )
    return normalized


def validate_daic_person_threshold(threshold: float):
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Invalid daic_person_threshold={threshold!r}. Expected a value in [0.0, 1.0]."
        )
    return value


def extract_participant_id(segment_key: str):
    match = _PARTICIPANT_ID_PATTERN.match((segment_key or "").strip())
    if match is None:
        raise ValueError(f"Could not extract DAIC participant ID from key={segment_key!r}.")
    return match.group("participant_id")


def build_daic_task_metadata(task: dict, default_dataset_name: str = "unknown"):
    dataset_name = task.get("dataset", default_dataset_name)
    raw_key = task.get("key", "")
    source_key = task.get("source_key") or raw_key
    target_text = task.get("target", "")

    daic_key = None
    participant_id = None
    if dataset_name == DAIC_DATASET_NAME:
        daic_key = source_key
        participant_id = extract_participant_id(daic_key)

    return {
        "dataset_name": dataset_name,
        "raw_key": raw_key,
        "source_key": source_key,
        "target_text": target_text,
        "daic_key": daic_key,
        "participant_id": participant_id,
    }


def map_target_text_to_binary(target_text: str):
    text = (target_text or "").strip()
    if text == DAIC_NON_DEPRESSED_LABEL or DAIC_NON_DEPRESSED_LABEL in text or "健康" in text:
        return 0
    if text == DAIC_DEPRESSED_LABEL or DAIC_DEPRESSED_LABEL in text:
        return 1
    return -1


def get_daic_label_token_ids(tokenizer):
    depressed_ids = tokenizer.encode(DAIC_DEPRESSED_LABEL, add_special_tokens=False)
    non_depressed_ids = tokenizer.encode(DAIC_NON_DEPRESSED_LABEL, add_special_tokens=False)
    if not depressed_ids or not non_depressed_ids:
        raise ValueError("Failed to tokenize DAIC class labels.")
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


def build_daic_eval_records(
    tokenizer,
    logits: torch.Tensor,
    labels: torch.Tensor,
    keys,
    participant_ids,
    target_texts,
    dataset_names=None,
):
    depressed_token_ids, non_depressed_token_ids = get_daic_label_token_ids(tokenizer)
    records = []

    batch_size = labels.size(0)
    for sample_index in range(batch_size):
        if dataset_names is not None and dataset_names[sample_index] != DAIC_DATASET_NAME:
            continue
        if not keys[sample_index] or not participant_ids[sample_index]:
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
        records.append(
            {
                "key": keys[sample_index],
                "participant_id": participant_ids[sample_index],
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


def aggregate_participant_predictions(records, mode: str, threshold: float):
    normalized_mode = normalize_daic_eval_mode(mode)
    normalized_threshold = validate_daic_person_threshold(threshold)

    deduped_records = {}
    for record in records:
        key = record["key"]
        deduped_records.setdefault(key, record)

    grouped_records = defaultdict(list)
    for record in deduped_records.values():
        grouped_records[record["participant_id"]].append(record)

    stats = make_binary_stats()

    for participant_id, participant_records in grouped_records.items():
        del participant_id
        labels = {
            map_target_text_to_binary(record["target_text"])
            for record in participant_records
        }
        labels.discard(-1)
        if len(labels) != 1:
            raise ValueError("Inconsistent or missing DAIC target labels within a participant group.")
        y_true = labels.pop()

        depressed_probabilities = [float(record["depressed_probability"]) for record in participant_records]
        mean_probability = sum(depressed_probabilities) / len(depressed_probabilities)

        if normalized_mode == DAIC_EVAL_MODE_MAJORITY_VOTE:
            depressed_votes = sum(probability >= normalized_threshold for probability in depressed_probabilities)
            non_depressed_votes = len(depressed_probabilities) - depressed_votes
            if depressed_votes > non_depressed_votes:
                y_pred = 1
            elif depressed_votes < non_depressed_votes:
                y_pred = 0
            else:
                y_pred = 1 if mean_probability >= normalized_threshold else 0
        elif normalized_mode == DAIC_EVAL_MODE_MEAN_PROBABILITY:
            y_pred = 1 if mean_probability >= normalized_threshold else 0
        else:
            max_probability = max(depressed_probabilities)
            y_pred = 1 if max_probability >= normalized_threshold else 0

        _update_binary_stats(stats, y_true, y_pred)

    stats["num_segments"] = len(deduped_records)
    stats["num_participants"] = len(grouped_records)
    return stats


def apply_daic_person_level_results(
    dataset_name: str,
    daic_eval_level: str,
    per_dataset_stats: dict,
    overall_stats: dict,
    daic_records,
    mode: str,
    threshold: float,
):
    normalized_dataset_name = (dataset_name or "").strip().lower()
    normalized_level = normalize_daic_eval_level(daic_eval_level)
    if normalized_level != DAIC_EVAL_LEVEL_PERSON or not daic_records:
        return dict(per_dataset_stats), dict(overall_stats), {}

    person_stats = aggregate_participant_predictions(daic_records, mode=mode, threshold=threshold)
    next_per_dataset_stats = dict(per_dataset_stats)
    supplemental_results = {}

    if normalized_dataset_name == DAIC_DATASET_NAME:
        next_per_dataset_stats[DAIC_DATASET_NAME] = dict(person_stats)
        return next_per_dataset_stats, dict(person_stats), supplemental_results

    if normalized_dataset_name == "merged":
        supplemental_results[DAIC_PERSON_RESULTS_KEY] = dict(person_stats)

    return next_per_dataset_stats, dict(overall_stats), supplemental_results
