# tests/integration/test_parquet_repository.py
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import pandas as pd

from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment
from src.adapters.parquet_candle_repository import ParquetCandleRepository


@pytest.mark.integration
def test_parquet_repository_saves_file(tmp_path: Path):
    repo = ParquetCandleRepository(output_dir=tmp_path)

    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=10,
            high=12,
            low=9,
            close=11,
            volume=1000,
        )
    ]

    repo.save_candles("AAPL", candles)

    asset_dir = tmp_path / "AAPL"
    files = list(asset_dir.glob("*.parquet"))

    assert len(files) == 1
    assert files[0].name == "candles_AAPL_1d.parquet"

@pytest.mark.integration
def test_sentiment_is_persisted_when_dates_overlap(tmp_path):
    # Arrange
    asset = "AAPL"

    candles = [
        Candle(
            timestamp=datetime(2026, 1, 10, tzinfo=timezone.utc),
            open=100,
            high=110,
            low=90,
            close=105,
            volume=1000,
        )
    ]

    sentiments = [
        DailySentiment(
            asset_id=asset,
            day=date(2026, 1, 10),
            sentiment_score=0.42,
            sentiment_std=0.1,
            n_articles=5,
        )
    ]

    repo = ParquetCandleRepository(output_dir=tmp_path)
    repo.save_candles(asset, candles)

    # Act
    repo.update_sentiment(asset, sentiments)

    # Assert (infra-level)
    parquet_path = tmp_path / "AAPL" / "candles_AAPL_1d.parquet"
    df = pd.read_parquet(parquet_path)

    assert df.loc[0, "sentiment_score"] == pytest.approx(0.42)
    assert df.loc[0, "n_articles"] == 5
