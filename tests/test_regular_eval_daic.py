import json
import os
import stat
import subprocess
from pathlib import Path

import torch

from utils.daic_eval import (
    DAIC_DATASET_NAME,
    DAIC_EVAL_LEVEL_PERSON,
    DAIC_EVAL_MODE_MAX_PROBABILITY,
    DAIC_EVAL_MODE_MEAN_PROBABILITY,
    DAIC_PERSON_RESULTS_KEY,
    apply_daic_person_level_results,
    aggregate_participant_predictions,
    build_daic_eval_records,
    build_daic_task_metadata,
    make_binary_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "抑郁": [0, 1],
            "非抑郁": [2, 3],
        }
        return mapping[text]


def _build_logits_and_labels():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 4.5, -2.0, -2.0],
                [4.0, 4.5, -2.0, -2.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-2.0, -2.0, 5.0, 4.5],
                [-2.0, -2.0, 4.0, 4.5],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [
            [-100, -100, 0, 1],
            [-100, -100, 2, 3],
        ],
        dtype=torch.long,
    )
    return logits, labels


def test_build_daic_task_metadata_prefers_source_key_for_merged_daic():
    metadata = build_daic_task_metadata(
        {
            "key": "daic_woz__302_random_segment_1",
            "source_key": "302_random_segment_1",
            "target": "非抑郁",
            "dataset": "daic_woz",
        },
        default_dataset_name="merged",
    )

    assert metadata["dataset_name"] == "daic_woz"
    assert metadata["source_key"] == "302_random_segment_1"
    assert metadata["daic_key"] == "302_random_segment_1"
    assert metadata["participant_id"] == "302"


def test_apply_daic_person_level_results_replaces_primary_daic_metrics():
    per_dataset_stats = {
        DAIC_DATASET_NAME: {"tp": 2, "fp": 0, "fn": 0, "tn": 0, "total": 2, "correct": 2}
    }
    overall_stats = {"tp": 2, "fp": 0, "fn": 0, "tn": 0, "total": 2, "correct": 2}
    daic_records = [
        {"key": "335_segment_1", "participant_id": "335", "target_text": "抑郁", "depressed_probability": 0.90},
        {"key": "335_segment_2", "participant_id": "335", "target_text": "抑郁", "depressed_probability": 0.80},
    ]

    next_per_dataset_stats, next_overall_stats, supplemental_results = apply_daic_person_level_results(
        dataset_name=DAIC_DATASET_NAME,
        daic_eval_level=DAIC_EVAL_LEVEL_PERSON,
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        daic_records=daic_records,
        mode=DAIC_EVAL_MODE_MEAN_PROBABILITY,
        threshold=0.5,
    )

    assert supplemental_results == {}
    assert next_per_dataset_stats[DAIC_DATASET_NAME]["total"] == 1
    assert next_per_dataset_stats[DAIC_DATASET_NAME]["num_segments"] == 2
    assert next_per_dataset_stats[DAIC_DATASET_NAME]["num_participants"] == 1
    assert next_overall_stats["total"] == 1


def test_apply_daic_person_level_results_adds_supplemental_merged_row():
    per_dataset_stats = {
        "cmdc": {"tp": 1, "fp": 0, "fn": 0, "tn": 0, "total": 1, "correct": 1},
        DAIC_DATASET_NAME: {"tp": 0, "fp": 1, "fn": 0, "tn": 1, "total": 2, "correct": 1},
    }
    overall_stats = {"tp": 1, "fp": 1, "fn": 0, "tn": 1, "total": 3, "correct": 2}
    daic_records = [
        {"key": "302_segment_1", "participant_id": "302", "target_text": "非抑郁", "depressed_probability": 0.10},
        {"key": "302_segment_2", "participant_id": "302", "target_text": "非抑郁", "depressed_probability": 0.20},
    ]

    next_per_dataset_stats, next_overall_stats, supplemental_results = apply_daic_person_level_results(
        dataset_name="merged",
        daic_eval_level=DAIC_EVAL_LEVEL_PERSON,
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        daic_records=daic_records,
        mode=DAIC_EVAL_MODE_MEAN_PROBABILITY,
        threshold=0.5,
    )

    assert next_per_dataset_stats == per_dataset_stats
    assert next_overall_stats == overall_stats
    assert supplemental_results[DAIC_PERSON_RESULTS_KEY]["total"] == 1
    assert supplemental_results[DAIC_PERSON_RESULTS_KEY]["tn"] == 1


def test_build_daic_eval_records_match_for_audio_and_text_paths():
    tokenizer = DummyTokenizer()
    logits, labels = _build_logits_and_labels()
    keys = ["335_segment_1", "302_segment_1"]
    participant_ids = ["335", "302"]
    target_texts = ["抑郁", "非抑郁"]
    dataset_names = [DAIC_DATASET_NAME, DAIC_DATASET_NAME]

    audio_records = build_daic_eval_records(
        tokenizer,
        logits,
        labels,
        keys,
        participant_ids,
        target_texts,
        dataset_names=dataset_names,
    )
    text_records = build_daic_eval_records(
        tokenizer,
        logits,
        labels,
        keys,
        participant_ids,
        target_texts,
        dataset_names=dataset_names,
    )

    assert audio_records == text_records
    assert aggregate_participant_predictions(audio_records, DAIC_EVAL_MODE_MAX_PROBABILITY, threshold=0.5) == \
        aggregate_participant_predictions(text_records, DAIC_EVAL_MODE_MAX_PROBABILITY, threshold=0.5)


def test_eval_sh_passes_daic_level_arguments(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_path = tmp_path / "captured_args.json"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_PATH'], 'w', encoding='utf-8') as handle:\n"
        "    json.dump(sys.argv[1:], handle)\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_PATH": str(capture_path),
            "MODEL_FAMILY": "text",
            "PROMPT_MODE": "textonly",
            "DATASET_NAME": "daic_woz",
            "DAIC_EVAL_LEVEL": "segment",
            "DAIC_EVAL_MODE": "max_probability",
            "DAIC_PERSON_THRESHOLD": "0.75",
            "PEFT_PATH": "baseline",
            "LOG_DIR": str(tmp_path / "logs"),
            "RESULTS_DIR": str(tmp_path / "results"),
            "OUTPUT_JSON": str(tmp_path / "results.json"),
            "DEVICE": "cpu",
        }
    )

    subprocess.run(
        ["bash", str(PROJECT_ROOT / "eval.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )

    captured_args = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured_args[0] == "evaluate_textonly.py"

    def assert_option(option, expected_value):
        index = captured_args.index(option)
        assert captured_args[index + 1] == expected_value

    assert_option("--dataset_name", "daic_woz")
    assert_option("--daic_eval_level", "segment")
    assert_option("--daic_eval_mode", "max_probability")
    assert_option("--daic_person_threshold", "0.75")
    assert_option("--data_path", str(PROJECT_ROOT / "data" / "daic_woz" / "val"))
    assert_option(
        "--prompt_path",
        str(PROJECT_ROOT / "data" / "daic_woz" / "val" / "daic_woz_multiprompt_textonly.jsonl"),
    )
    assert_option("--task_filename", "daic_woz_multitask.jsonl")
