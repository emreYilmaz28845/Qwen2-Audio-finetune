from config.config import Config
from optuna_hpo.train_ddp import (
    _build_best_eval_summary,
    _needs_grouped_eval_metadata,
    _primary_grouped_person_eval_dataset,
    _supplemental_grouped_person_eval_datasets,
)


def test_merged_eval_enables_supplemental_grouped_metadata_without_changing_primary_objective():
    cfg = Config()
    cfg.data.dataset_name = "merged"
    cfg.eval.daic_eval_level = "segment"
    cfg.eval.eatd_eval_level = "person"
    cfg.eval.cmdc_eval_level = "person"

    assert _primary_grouped_person_eval_dataset(cfg) == ""
    assert _supplemental_grouped_person_eval_datasets(cfg) == ["cmdc", "eatd"]
    assert _needs_grouped_eval_metadata(cfg) is True


def test_direct_grouped_person_eval_keeps_primary_dataset_person_objective():
    cfg = Config()
    cfg.data.dataset_name = "eatd"
    cfg.eval.eatd_eval_level = "person"
    cfg.eval.cmdc_eval_level = "person"

    assert _primary_grouped_person_eval_dataset(cfg) == "eatd"
    assert _supplemental_grouped_person_eval_datasets(cfg) == []
    assert _needs_grouped_eval_metadata(cfg) is True


def test_build_best_eval_summary_includes_supplemental_grouped_rows_for_merged():
    cfg = Config()
    cfg.data.dataset_name = "merged"
    cfg.eval.daic_eval_level = "segment"
    cfg.eval.eatd_eval_level = "person"
    cfg.eval.eatd_eval_mode = "mean_probability"
    cfg.eval.eatd_person_threshold = 0.6
    cfg.eval.cmdc_eval_level = "person"
    cfg.eval.cmdc_eval_mode = "max_probability"
    cfg.eval.cmdc_person_threshold = 0.7

    summary = _build_best_eval_summary(
        cfg,
        eval_loss=0.25,
        primary_stats={"tp": 3, "fp": 1, "fn": 1, "tn": 5, "total": 10, "correct": 8},
        supplemental_stats_by_dataset={
            "eatd": {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 2,
                "total": 3,
                "correct": 3,
                "num_segments": 9,
                "num_participants": 3,
            },
            "cmdc": {
                "tp": 2,
                "fp": 1,
                "fn": 0,
                "tn": 1,
                "total": 4,
                "correct": 3,
                "num_segments": 40,
                "num_participants": 4,
            },
        },
    )

    assert summary["dataset_name"] == "merged"
    assert summary["primary_scope"] == "overall_segment"
    assert summary["primary"]["mode"] == "segment"
    assert "eatd_person" in summary["supplemental_grouped_eval"]
    assert "cmdc_person" in summary["supplemental_grouped_eval"]
    assert summary["supplemental_grouped_eval"]["eatd_person"]["threshold"] == 0.6
    assert summary["supplemental_grouped_eval"]["cmdc_person"]["mode"] == "max_probability"
