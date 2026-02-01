# tests/unit/adapters/repositories/test_parquet_daily_sentiment_repository.py

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_daily_sentiment_repository import (
    ParquetDailySentimentRepository,
)
from src.entities.daily_sentiment import DailySentiment
from src.infrastructure.schemas.daily_sentiment_parquet_schema import (
    DAILY_SENTIMENT_PARQUET_COLUMNS,
    DAILY_SENTIMENT_PARQUET_DTYPES,
)


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path: Path) -> ParquetDailySentimentRepository:
    return ParquetDailySentimentRepository(output_dir=tmp_path)


def test_repository_raises_if_output_dir_is_not_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("not a dir")

    with pytest.raises(NotADirectoryError):
        ParquetDailySentimentRepository(output_dir=file_path)


def test_upsert_persists_parquet_with_schema_and_dtypes(
    repo: ParquetDailySentimentRepository, tmp_path: Path
) -> None:
    daily = [
        DailySentiment(
            asset_id="AAPL",
            day=date(2024, 1, 1),
            sentiment_score=0.2,
            n_articles=3,
            sentiment_std=0.1,
        ),
        DailySentiment(
            asset_id="AAPL",
            day=date(2024, 1, 2),
            sentiment_score=-0.1,
            n_articles=2,
            sentiment_std=None,
        ),
    ]

    repo.upsert_daily_sentiment_batch(daily)

    parquet_path = tmp_path / "AAPL" / "daily_sentiment_AAPL.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert list(df.columns) == DAILY_SENTIMENT_PARQUET_COLUMNS

    day_col = pd.to_datetime(df["day"], utc=True, errors="raise")
    assert str(day_col.dt.tz) == "UTC"

    assert str(df["sentiment_score"].dtype) == DAILY_SENTIMENT_PARQUET_DTYPES["sentiment_score"]
    assert str(df["n_articles"].dtype) == DAILY_SENTIMENT_PARQUET_DTYPES["n_articles"]
    assert str(df["sentiment_std"].dtype) == DAILY_SENTIMENT_PARQUET_DTYPES["sentiment_std"]

    assert pd.isna(df.loc[1, "sentiment_std"])


def test_upsert_deduplicates_by_day_keep_last(
    repo: ParquetDailySentimentRepository, tmp_path: Path
) -> None:
    d1 = DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=0.1,
        n_articles=2,
        sentiment_std=0.2,
    )
    d1_updated = DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=0.9,
        n_articles=5,
        sentiment_std=0.3,
    )

    repo.upsert_daily_sentiment_batch([d1])
    repo.upsert_daily_sentiment_batch([d1_updated])

    parquet_path = tmp_path / "AAPL" / "daily_sentiment_AAPL.parquet"
    df = pd.read_parquet(parquet_path)

    assert len(df) == 1
    assert df.loc[0, "sentiment_score"] == pytest.approx(0.9)
    assert df.loc[0, "n_articles"] == 5


def test_list_daily_sentiment_filters_inclusive(repo: ParquetDailySentimentRepository) -> None:
    repo.upsert_daily_sentiment_batch(
        [
            DailySentiment(
                asset_id="AAPL",
                day=date(2024, 1, 1),
                sentiment_score=0.1,
                n_articles=2,
            ),
            DailySentiment(
                asset_id="AAPL",
                day=date(2024, 1, 2),
                sentiment_score=0.2,
                n_articles=3,
            ),
            DailySentiment(
                asset_id="AAPL",
                day=date(2024, 1, 3),
                sentiment_score=0.3,
                n_articles=4,
            ),
        ]
    )

    out = repo.list_daily_sentiment("AAPL", _dt_utc(2024, 1, 2), _dt_utc(2024, 1, 3))

    assert [d.day.isoformat() for d in out] == ["2024-01-02", "2024-01-03"]
    assert all(isinstance(d.day, date) for d in out)
