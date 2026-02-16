from __future__ import annotations

import pytest

from src.infrastructure.schemas.model_artifact_schema import (
    TFT_TRAINING_DEFAULTS,
    validate_tft_training_config,
)


def test_validate_tft_training_config_accepts_defaults() -> None:
    cfg = dict(TFT_TRAINING_DEFAULTS)
    validate_tft_training_config(cfg)


def test_validate_tft_training_config_rejects_dropout_out_of_range() -> None:
    cfg = dict(TFT_TRAINING_DEFAULTS)
    cfg["dropout"] = 1.1
    with pytest.raises(ValueError, match="dropout"):
        validate_tft_training_config(cfg)


def test_validate_tft_training_config_rejects_non_positive_learning_rate() -> None:
    cfg = dict(TFT_TRAINING_DEFAULTS)
    cfg["learning_rate"] = 0.0
    with pytest.raises(ValueError, match="learning_rate"):
        validate_tft_training_config(cfg)


def test_validate_tft_training_config_rejects_prediction_gt_encoder() -> None:
    cfg = dict(TFT_TRAINING_DEFAULTS)
    cfg["max_encoder_length"] = 30
    cfg["max_prediction_length"] = 31
    with pytest.raises(ValueError, match="max_prediction_length"):
        validate_tft_training_config(cfg)
