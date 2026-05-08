import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F


DAIC_EVAL_MODE_MAJORITY_VOTE = "majority_vote"
DAIC_EVAL_MODE_MEAN_PROBABILITY = "mean_probability"
DAIC_EVAL_MODE_MAX_PROBABILITY = "max_probability"
SUPPORTED_DAIC_EVAL_MODES = {
    DAIC_EVAL_MODE_MAJORITY_VOTE,
    DAIC_EVAL_MODE_MEAN_PROBABILITY,
    DAIC_EVAL_MODE_MAX_PROBABILITY,
}

DAIC_DEPRESSED_LABEL = "抑郁"
DAIC_NON_DEPRESSED_LABEL = "非抑郁"
_PARTICIPANT_ID_PATTERN = re.compile(r"^(?P<participant_id>\d+)")


def normalize_daic_eval_mode(mode: str):
    normalized = (mode or DAIC_EVAL_MODE_MAJORITY_VOTE).strip().lower()
    if normalized not in SUPPORTED_DAIC_EVAL_MODES:
        raise ValueError(
            f"Unsupported daic_eval_mode={mode!r}. "
            f"Expected one of {sorted(SUPPORTED_DAIC_EVAL_MODES)}."
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

    stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0, "correct": 0}

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
