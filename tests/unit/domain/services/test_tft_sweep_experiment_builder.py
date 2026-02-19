from __future__ import annotations

from src.domain.services.tft_sweep_experiment_builder import (
    build_one_at_a_time_experiments,
)


def test_build_experiments_includes_baseline_and_skips_default_values() -> None:
    base = {
        "max_encoder_length": 60,
        "max_prediction_length": 1,
        "learning_rate": 5e-4,
    }
    ranges = {
        "max_encoder_length": [30, 60, 90],
        "learning_rate": [5e-4, 1e-3],
    }

    experiments = build_one_at_a_time_experiments(
        base_config=base,
        param_ranges=ranges,
    )

    labels = [e.run_label for e in experiments]
    assert labels[0] == "baseline"
    assert "max_encoder_length=30" in labels
    assert "max_encoder_length=90" in labels
    assert "learning_rate=0.001" in labels
    assert "max_encoder_length=60" not in labels


def test_build_experiments_respects_prediction_encoder_constraint() -> None:
    base = {
        "max_encoder_length": 10,
        "max_prediction_length": 5,
        "learning_rate": 1e-3,
    }
    ranges = {"max_encoder_length": [3, 10]}
    experiments = build_one_at_a_time_experiments(base_config=base, param_ranges=ranges)
    labels = [e.run_label for e in experiments]
    assert "max_encoder_length=3" not in labels
