import torch

from utils.grouped_eval import (
    compute_segment_depressed_probability,
    extract_group_id,
    normalize_grouped_eval_level,
    aggregate_group_predictions,
)


def test_extract_group_id_for_all_supported_grouped_datasets():
    assert extract_group_id("daic_woz", "335_segment_12") == "335"
    assert extract_group_id("eatd", "104_negative") == "104"
    assert extract_group_id("cmdc", "HC41_Q3") == "HC41"


def test_normalize_grouped_eval_level():
    assert normalize_grouped_eval_level("segment") == "segment"
    assert normalize_grouped_eval_level("person") == "person"


def test_majority_vote_uses_mean_probability_for_ties():
    records = [
        {"key": "335_segment_1", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.90},
        {"key": "335_segment_2", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_group_predictions(records, "majority_vote", threshold=0.5)
    assert stats["tp"] == 1
    assert stats["total"] == 1


def test_mean_probability_aggregation():
    records = [
        {"key": "302_segment_1", "group_id": "302", "target_text": "非抑郁", "depressed_probability": 0.40},
        {"key": "302_segment_2", "group_id": "302", "target_text": "非抑郁", "depressed_probability": 0.30},
        {"key": "302_segment_3", "group_id": "302", "target_text": "非抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_group_predictions(records, "mean_probability", threshold=0.5)
    assert stats["tn"] == 1
    assert stats["correct"] == 1


def test_max_probability_aggregation():
    records = [
        {"key": "346_segment_1", "group_id": "346", "target_text": "抑郁", "depressed_probability": 0.20},
        {"key": "346_segment_2", "group_id": "346", "target_text": "抑郁", "depressed_probability": 0.80},
    ]
    stats = aggregate_group_predictions(records, "max_probability", threshold=0.5)
    assert stats["tp"] == 1
    assert stats["correct"] == 1


def test_duplicate_segment_keys_are_deduplicated():
    records = [
        {"key": "HC41_Q1", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.10},
        {"key": "HC41_Q1", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.10},
        {"key": "HC41_Q2", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.20},
    ]
    stats = aggregate_group_predictions(records, "mean_probability", threshold=0.5)
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
