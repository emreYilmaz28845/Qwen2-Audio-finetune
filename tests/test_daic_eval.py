import torch

from utils.daic_eval import (
    DAIC_EVAL_LEVEL_PERSON,
    DAIC_EVAL_LEVEL_SEGMENT,
    DAIC_EVAL_MODE_MAJORITY_VOTE,
    DAIC_EVAL_MODE_MAX_PROBABILITY,
    DAIC_EVAL_MODE_MEAN_PROBABILITY,
    aggregate_participant_predictions,
    compute_segment_depressed_probability,
    extract_participant_id,
    normalize_daic_eval_level,
)


def test_extract_participant_id():
    assert extract_participant_id("335_segment_12") == "335"
    assert extract_participant_id("491_random_segment_7") == "491"


def test_normalize_daic_eval_level():
    assert normalize_daic_eval_level("segment") == DAIC_EVAL_LEVEL_SEGMENT
    assert normalize_daic_eval_level("person") == DAIC_EVAL_LEVEL_PERSON


def test_majority_vote_uses_mean_probability_for_ties():
    records = [
        {"key": "335_segment_1", "participant_id": "335", "target_text": "抑郁", "depressed_probability": 0.90},
        {"key": "335_segment_2", "participant_id": "335", "target_text": "抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_participant_predictions(records, DAIC_EVAL_MODE_MAJORITY_VOTE, threshold=0.5)
    assert stats["tp"] == 1
    assert stats["total"] == 1


def test_mean_probability_aggregation():
    records = [
        {"key": "302_segment_1", "participant_id": "302", "target_text": "非抑郁", "depressed_probability": 0.40},
        {"key": "302_segment_2", "participant_id": "302", "target_text": "非抑郁", "depressed_probability": 0.30},
        {"key": "302_segment_3", "participant_id": "302", "target_text": "非抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_participant_predictions(records, DAIC_EVAL_MODE_MEAN_PROBABILITY, threshold=0.5)
    assert stats["tn"] == 1
    assert stats["correct"] == 1


def test_max_probability_aggregation():
    records = [
        {"key": "346_segment_1", "participant_id": "346", "target_text": "抑郁", "depressed_probability": 0.20},
        {"key": "346_segment_2", "participant_id": "346", "target_text": "抑郁", "depressed_probability": 0.80},
    ]
    stats = aggregate_participant_predictions(records, DAIC_EVAL_MODE_MAX_PROBABILITY, threshold=0.5)
    assert stats["tp"] == 1
    assert stats["correct"] == 1


def test_duplicate_segment_keys_are_deduplicated():
    records = [
        {"key": "388_segment_1", "participant_id": "388", "target_text": "非抑郁", "depressed_probability": 0.10},
        {"key": "388_segment_1", "participant_id": "388", "target_text": "非抑郁", "depressed_probability": 0.10},
        {"key": "388_segment_2", "participant_id": "388", "target_text": "非抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_participant_predictions(records, DAIC_EVAL_MODE_MEAN_PROBABILITY, threshold=0.5)
    assert stats["num_segments"] == 2
    assert stats["num_participants"] == 1


def test_segment_probability_prefers_depressed_label():
    logits_row = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 4.5, -2.0, -2.0],
            [4.0, 4.5, -2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    probability = compute_segment_depressed_probability(
        logits_row=logits_row,
        start_pred_index=1,
        depressed_token_ids=[0, 1],
        non_depressed_token_ids=[2, 3],
    )
    assert probability > 0.5
