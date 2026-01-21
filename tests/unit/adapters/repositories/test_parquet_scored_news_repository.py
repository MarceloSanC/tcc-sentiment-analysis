# tests/unit/adapters/repositories/test_parquet_scored_news_repository.py

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_scored_news_repository import ParquetScoredNewsRepository
from src.entities.scored_news_article import ScoredNewsArticle
from src.infrastructure.schemas.scored_news_parquet_schema import (
    SCORED_NEWS_PARQUET_COLUMNS,
    SCORED_NEWS_PARQUET_DTYPES,
)


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path: Path) -> ParquetScoredNewsRepository:
    return ParquetScoredNewsRepository(output_dir=tmp_path)


def test_repository_raises_if_output_dir_is_not_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("not a dir")

    with pytest.raises(NotADirectoryError):
        ParquetScoredNewsRepository(output_dir=file_path)


def test_upsert_persists_parquet_with_schema_and_dtypes(
    repo: ParquetScoredNewsRepository, tmp_path: Path
) -> None:
    articles = [
        ScoredNewsArticle(
            asset_id="AAPL",
            article_id="a1",
            published_at=_dt_utc(2024, 1, 1, 10, 0),
            sentiment_score=0.6,
            confidence=0.6,
            model_name="finbert",
        ),
        ScoredNewsArticle(
            asset_id="AAPL",
            article_id="a2",
            published_at=_dt_utc(2024, 1, 2, 10, 0),
            sentiment_score=-0.2,
            confidence=None,
            model_name=None,
        ),
    ]

    repo.upsert_scored_news_batch(articles)

    parquet_path = tmp_path / "asset=AAPL" / "scored_news_AAPL.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert list(df.columns) == SCORED_NEWS_PARQUET_COLUMNS

    published = pd.to_datetime(df["published_at"], utc=True, errors="raise")
    assert str(published.dt.tz) == "UTC"

    assert str(df["sentiment_score"].dtype) == SCORED_NEWS_PARQUET_DTYPES["sentiment_score"]
    assert str(df["confidence"].dtype) == SCORED_NEWS_PARQUET_DTYPES["confidence"]
    assert str(df["model_name"].dtype) == SCORED_NEWS_PARQUET_DTYPES["model_name"]

    assert pd.isna(df.loc[1, "confidence"])
    assert pd.isna(df.loc[1, "model_name"])


def test_upsert_deduplicates_by_article_id_keep_last(
    repo: ParquetScoredNewsRepository, tmp_path: Path
) -> None:
    a1 = ScoredNewsArticle(
        asset_id="AAPL",
        article_id="a1",
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        sentiment_score=0.1,
        confidence=0.1,
        model_name="finbert",
    )
    a1_updated = ScoredNewsArticle(
        asset_id="AAPL",
        article_id="a1",
        published_at=_dt_utc(2024, 1, 1, 10, 5),
        sentiment_score=0.9,
        confidence=0.9,
        model_name="finbert",
    )

    repo.upsert_scored_news_batch([a1])
    repo.upsert_scored_news_batch([a1_updated])

    parquet_path = tmp_path / "asset=AAPL" / "scored_news_AAPL.parquet"
    df = pd.read_parquet(parquet_path)

    assert len(df) == 1
    assert df.loc[0, "article_id"] == "a1"
    assert df.loc[0, "sentiment_score"] == pytest.approx(0.9)


def test_list_scored_news_filters_inclusive(repo: ParquetScoredNewsRepository) -> None:
    repo.upsert_scored_news_batch(
        [
            ScoredNewsArticle(
                asset_id="AAPL",
                article_id="a1",
                published_at=_dt_utc(2024, 1, 1, 10, 0),
                sentiment_score=0.1,
            ),
            ScoredNewsArticle(
                asset_id="AAPL",
                article_id="a2",
                published_at=_dt_utc(2024, 1, 2, 10, 0),
                sentiment_score=0.2,
            ),
            ScoredNewsArticle(
                asset_id="AAPL",
                article_id="a3",
                published_at=_dt_utc(2024, 1, 3, 10, 0),
                sentiment_score=0.3,
            ),
        ]
    )

    out = repo.list_scored_news("AAPL", _dt_utc(2024, 1, 2, 10, 0), _dt_utc(2024, 1, 3, 10, 0))

    assert [a.article_id for a in out] == ["a2", "a3"]
    assert all(a.published_at.tzinfo is not None for a in out)


def test_list_article_ids_returns_set(repo: ParquetScoredNewsRepository) -> None:
    repo.upsert_scored_news_batch(
        [
            ScoredNewsArticle(
                asset_id="AAPL",
                article_id="a1",
                published_at=_dt_utc(2024, 1, 1, 10, 0),
                sentiment_score=0.1,
            )
        ]
    )

    ids = repo.list_article_ids("AAPL")
    assert ids == {"a1"}
