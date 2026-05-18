import json
import os
import stat
import subprocess
from pathlib import Path

import torch

from utils.grouped_eval import (
    apply_person_level_results,
    aggregate_group_predictions,
    build_grouped_eval_records,
    build_grouped_task_metadata,
    grouped_person_results_key,
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


def test_build_grouped_task_metadata_prefers_source_key_for_merged_grouped_rows():
    metadata = build_grouped_task_metadata(
        {
            "key": "eatd__302_negative",
            "source_key": "302_negative",
            "target": "非抑郁",
            "dataset": "eatd",
        },
        default_dataset_name="merged",
    )

    assert metadata["dataset_name"] == "eatd"
    assert metadata["source_key"] == "302_negative"
    assert metadata["segment_key"] == "302_negative"
    assert metadata["group_id"] == "302"


def test_apply_person_level_results_replaces_primary_eatd_metrics():
    per_dataset_stats = {
        "eatd": {"tp": 2, "fp": 0, "fn": 0, "tn": 0, "total": 2, "correct": 2}
    }
    overall_stats = {"tp": 2, "fp": 0, "fn": 0, "tn": 0, "total": 2, "correct": 2}
    grouped_records_by_dataset = {
        "eatd": [
            {"key": "104_negative", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.90},
            {"key": "104_neutral", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.80},
        ]
    }

    next_per_dataset_stats, next_overall_stats, supplemental_results = apply_person_level_results(
        dataset_name="eatd",
        level_by_dataset={"eatd": "person"},
        mode_by_dataset={"eatd": "mean_probability"},
        threshold_by_dataset={"eatd": 0.5},
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    assert supplemental_results == {}
    assert next_per_dataset_stats["eatd"]["total"] == 1
    assert next_per_dataset_stats["eatd"]["num_segments"] == 2
    assert next_per_dataset_stats["eatd"]["num_participants"] == 1
    assert next_overall_stats["total"] == 1


def test_apply_person_level_results_adds_supplemental_merged_rows_for_eatd_and_cmdc():
    per_dataset_stats = {
        "cmdc": {"tp": 0, "fp": 0, "fn": 0, "tn": 2, "total": 2, "correct": 2},
        "eatd": {"tp": 1, "fp": 0, "fn": 0, "tn": 1, "total": 2, "correct": 2},
    }
    overall_stats = {"tp": 1, "fp": 0, "fn": 0, "tn": 3, "total": 4, "correct": 4}
    grouped_records_by_dataset = {
        "eatd": [
            {"key": "104_negative", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.90},
            {"key": "104_neutral", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.80},
        ],
        "cmdc": [
            {"key": "HC41_Q1", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.10},
            {"key": "HC41_Q2", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.20},
        ],
    }

    next_per_dataset_stats, next_overall_stats, supplemental_results = apply_person_level_results(
        dataset_name="merged",
        level_by_dataset={"eatd": "person", "cmdc": "person"},
        mode_by_dataset={"eatd": "mean_probability", "cmdc": "mean_probability"},
        threshold_by_dataset={"eatd": 0.5, "cmdc": 0.5},
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    assert next_per_dataset_stats == per_dataset_stats
    assert next_overall_stats == overall_stats
    assert supplemental_results[grouped_person_results_key("eatd")]["total"] == 1
    assert supplemental_results[grouped_person_results_key("cmdc")]["total"] == 1


def test_apply_person_level_results_adds_supplemental_merged_rows_for_daic_only():
    per_dataset_stats = {
        "daic_woz": {"tp": 2, "fp": 1, "fn": 0, "tn": 1, "total": 4, "correct": 3},
    }
    overall_stats = {"tp": 2, "fp": 1, "fn": 0, "tn": 1, "total": 4, "correct": 3}
    grouped_records_by_dataset = {
        "daic_woz": [
            {"key": "335_segment_1", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.90},
            {"key": "335_segment_2", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.85},
            {"key": "302_segment_1", "group_id": "302", "target_text": "非抑郁", "depressed_probability": 0.10},
        ]
    }

    next_per_dataset_stats, next_overall_stats, supplemental_results = apply_person_level_results(
        dataset_name="merged",
        level_by_dataset={"daic_woz": "person", "eatd": "segment", "cmdc": "segment"},
        mode_by_dataset={"daic_woz": "mean_probability"},
        threshold_by_dataset={"daic_woz": 0.5},
        per_dataset_stats=per_dataset_stats,
        overall_stats=overall_stats,
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    assert next_per_dataset_stats == per_dataset_stats
    assert next_overall_stats == overall_stats
    assert supplemental_results[grouped_person_results_key("daic_woz")]["total"] == 2


def test_apply_person_level_results_supports_mixed_merged_grouped_datasets():
    grouped_records_by_dataset = {
        "daic_woz": [
            {"key": "335_segment_1", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.90},
            {"key": "335_segment_2", "group_id": "335", "target_text": "抑郁", "depressed_probability": 0.80},
        ],
        "eatd": [
            {"key": "104_negative", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.70},
            {"key": "104_neutral", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.80},
            {"key": "104_positive", "group_id": "104", "target_text": "抑郁", "depressed_probability": 0.90},
        ],
        "cmdc": [
            {"key": "HC41_Q1", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.10},
            {"key": "HC41_Q2", "group_id": "HC41", "target_text": "非抑郁", "depressed_probability": 0.20},
        ],
    }

    _, next_overall_stats, supplemental_results = apply_person_level_results(
        dataset_name="merged",
        level_by_dataset={"daic_woz": "person", "eatd": "person", "cmdc": "person"},
        mode_by_dataset={
            "daic_woz": "majority_vote",
            "eatd": "mean_probability",
            "cmdc": "max_probability",
        },
        threshold_by_dataset={"daic_woz": 0.5, "eatd": 0.5, "cmdc": 0.5},
        per_dataset_stats={},
        overall_stats=make_binary_stats(),
        grouped_records_by_dataset=grouped_records_by_dataset,
    )

    assert next_overall_stats["total"] == 0
    assert sorted(supplemental_results.keys()) == [
        grouped_person_results_key("cmdc"),
        grouped_person_results_key("daic_woz"),
        grouped_person_results_key("eatd"),
    ]


def test_build_grouped_eval_records_match_for_audio_and_text_paths():
    tokenizer = DummyTokenizer()
    logits, labels = _build_logits_and_labels()
    segment_keys = ["335_segment_1", "302_segment_1"]
    group_ids = ["335", "302"]
    target_texts = ["抑郁", "非抑郁"]
    dataset_names = ["daic_woz", "daic_woz"]

    audio_records = build_grouped_eval_records(
        tokenizer,
        logits,
        labels,
        dataset_names,
        segment_keys,
        group_ids,
        target_texts,
    )
    text_records = build_grouped_eval_records(
        tokenizer,
        logits,
        labels,
        dataset_names,
        segment_keys,
        group_ids,
        target_texts,
    )

    assert audio_records == text_records
    assert aggregate_group_predictions(audio_records["daic_woz"], "max_probability", threshold=0.5) == \
        aggregate_group_predictions(text_records["daic_woz"], "max_probability", threshold=0.5)


def test_eval_sh_passes_eatd_level_arguments(tmp_path):
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
            "DATASET_NAME": "eatd",
            "EATD_EVAL_LEVEL": "segment",
            "EATD_EVAL_MODE": "max_probability",
            "EATD_PERSON_THRESHOLD": "0.75",
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

    assert_option("--dataset_name", "eatd")
    assert_option("--eatd_eval_level", "segment")
    assert_option("--eatd_eval_mode", "max_probability")
    assert_option("--eatd_person_threshold", "0.75")
    assert_option("--data_path", str(PROJECT_ROOT / "data" / "eatd" / "test"))
    assert_option("--data_split", "test")


def test_eval_sh_cmdc_defaults_use_clean_holdout_test_paths(tmp_path):
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
            "DATASET_NAME": "cmdc",
            "CMDC_EVAL_LEVEL": "person",
            "CMDC_EVAL_MODE": "mean_probability",
            "CMDC_PERSON_THRESHOLD": "0.6",
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

    def assert_option(option, expected_value):
        index = captured_args.index(option)
        assert captured_args[index + 1] == expected_value

    assert_option("--dataset_name", "cmdc")
    assert_option("--cmdc_eval_level", "person")
    assert_option("--cmdc_eval_mode", "mean_probability")
    assert_option("--cmdc_person_threshold", "0.6")
    assert_option("--data_path", str(PROJECT_ROOT / "data" / "cmdc" / "test"))
    assert_option("--task_filename", "cmdc_multitask.jsonl")
    assert_option("--data_split", "test")


def test_eval_sh_refuses_non_test_final_eval(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "MODEL_FAMILY": "text",
            "PROMPT_MODE": "textonly",
            "DATASET_NAME": "eatd",
            "DATA_SPLIT": "val",
            "PEFT_PATH": "baseline",
            "LOG_DIR": str(tmp_path / "logs"),
            "RESULTS_DIR": str(tmp_path / "results"),
            "OUTPUT_JSON": str(tmp_path / "results.json"),
            "DEVICE": "cpu",
        }
    )

    completed = subprocess.run(
        ["bash", str(PROJECT_ROOT / "eval.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Refusing final evaluation on DATA_SPLIT=val" in completed.stdout
