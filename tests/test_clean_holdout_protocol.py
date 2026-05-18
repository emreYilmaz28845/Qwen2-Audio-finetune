import shutil
from pathlib import Path

import pytest

from optuna_hpo.hpo import get_dataset_config
from utils.grouped_eval import build_grouped_task_metadata
from utils.split_validation import SplitValidationError, validate_dataset_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_cmdc_fold4_as_primary_root(tmp_path: Path):
    source_root = PROJECT_ROOT / "data" / "cmdc" / "fold4"
    target_root = tmp_path / "cmdc"
    for split in ("train", "test"):
        target_split = target_root / split
        target_split.mkdir(parents=True)
        shutil.copyfile(source_root / split / "fold4.scp", target_split / "cmdc.scp")
        shutil.copyfile(source_root / split / "fold4_multitask.jsonl", target_split / "cmdc_multitask.jsonl")
        shutil.copyfile(source_root / split / "fold4_multiprompt.jsonl", target_split / "cmdc_multiprompt.jsonl")
    return target_root


def test_validator_catches_current_cmdc_fold4_train_test_leakage(tmp_path):
    root = _copy_cmdc_fold4_as_primary_root(tmp_path)

    with pytest.raises(SplitValidationError, match="appears in multiple splits"):
        validate_dataset_splits("cmdc", root, strict=False, splits=("train", "test"))


def test_clean_holdout_primary_splits_have_no_participant_overlap_after_regeneration():
    for dataset_name in ("daic_woz", "eatd", "cmdc", "merged"):
        summary = validate_dataset_splits(
            dataset_name,
            PROJECT_ROOT / "data" / dataset_name,
            strict=False,
        )
        assert summary["ok"] is True
        assert set(summary["participants_by_split"]) == {"train", "val", "test"}


def test_hpo_dataset_config_uses_validation_split_not_test_split(monkeypatch):
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    for dataset_name in ("daic_woz", "eatd", "cmdc", "merged"):
        cfg = get_dataset_config(dataset_name, "textonly", "default")
        assert cfg.train_data_path.endswith(f"data/{dataset_name}/train")
        assert cfg.eval_data_path.endswith(f"data/{dataset_name}/val")


def test_grouped_metadata_prefers_explicit_group_id_for_eatd_collision_safe_ids():
    metadata = build_grouped_task_metadata(
        {
            "key": "train_103_negative",
            "source_key": "103_negative",
            "target": "抑郁",
            "dataset": "eatd",
            "participant_id": "train_103",
            "group_id": "train_103",
        },
        default_dataset_name="eatd",
    )

    assert metadata["segment_key"] == "103_negative"
    assert metadata["group_id"] == "train_103"
