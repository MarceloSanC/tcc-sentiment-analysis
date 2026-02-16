from __future__ import annotations

import pandas as pd
import pytest

from src.adapters.pytorch_forecasting_tft_trainer import PytorchForecastingTFTTrainer


@pytest.mark.integration
def test_tft_trainer_smoke_tiny_dataset() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    pytest.importorskip("pytorch_lightning")

    ts = pd.date_range("2024-01-01", periods=40, tz="UTC", freq="D")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "asset_id": ["AAPL"] * len(ts),
            "time_idx": list(range(len(ts))),
            "target_return": [0.001 * ((i % 5) - 2) for i in range(len(ts))],
            "close": [100 + i for i in range(len(ts))],
            "volume": [1000 + (i * 10) for i in range(len(ts))],
            "day_of_week": [d.weekday() for d in ts],
            "month": [d.month for d in ts],
        }
    )

    # Keep validation/test long enough for encoder+prediction windows when using
    # TimeSeriesDataSet.from_dataset(..., predict=True).
    train_df = df.iloc[:24].copy()
    val_df = df.iloc[24:32].copy()
    test_df = df.iloc[32:].copy()

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        train_df,
        val_df,
        test_df,
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={
            "max_encoder_length": 4,
            "max_prediction_length": 1,
            "batch_size": 4,
            "max_epochs": 1,
            "hidden_size": 4,
            "hidden_continuous_size": 2,
            "attention_head_size": 1,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "seed": 42,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
        },
    )

    assert "rmse" in result.metrics
    assert set(result.split_metrics.keys()) == {"train", "val", "test"}
