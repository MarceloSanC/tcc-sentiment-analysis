# tests/unit/adapters/repositories/test_parquet_news_repository.py

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_news_repository import ParquetNewsRepository
from src.entities.news_article import NewsArticle


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path: Path) -> ParquetNewsRepository:
    return ParquetNewsRepository(output_dir=tmp_path)


def test_repository_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "raw" / "news"
    assert not out.exists()

    _ = ParquetNewsRepository(output_dir=out)

    assert out.exists()
    assert out.is_dir()


def test_repository_raises_if_output_dir_is_not_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("not a dir")

    with pytest.raises(NotADirectoryError):
        ParquetNewsRepository(output_dir=file_path)


def test_upsert_persists_parquet_in_partitioned_layout(repo: ParquetNewsRepository, tmp_path: Path) -> None:
    articles = [
        NewsArticle(
            asset_id="AAPL",
            article_id="https://example.com/a",
            published_at=_dt_utc(2024, 1, 1, 10, 0),
            headline="h1",
            summary="s1",
            source="Reuters",
            url="https://example.com/a",
            language="en",
        ),
        NewsArticle(
            asset_id="AAPL",
            article_id="https://example.com/b",
            published_at=_dt_utc(2024, 1, 2, 10, 0),
            headline="h2",
            summary="s2",
            source="Bloomberg",
            url="https://example.com/b",
            language="en",
        ),
    ]

    repo.upsert_news_batch(articles)

    # layout: data/raw/news/asset=AAPL/news_AAPL.parquet
    parquet_path = tmp_path / "asset=AAPL" / "news_AAPL.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 2
    assert set(df.columns) >= {
        "asset_id",
        "article_id",
        "published_at",
        "headline",
        "summary",
        "source",
        "url",
        "language",
    }
    assert df["asset_id"].unique().tolist() == ["AAPL"]

    # published_at deve estar em UTC (datetime64[ns, UTC])
    published = pd.to_datetime(df["published_at"], utc=True, errors="raise")
    assert not published.isna().any()
    assert str(published.dt.tz) == "UTC"


def test_upsert_deduplicates_by_article_id_keep_last(repo: ParquetNewsRepository, tmp_path: Path) -> None:
    # primeira versão
    a1 = NewsArticle(
        asset_id="AAPL",
        article_id="https://example.com/a",
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        headline="old headline",
        summary="old summary",
        source="Reuters",
        url="https://example.com/a",
        language="en",
    )

    # mesma article_id, conteúdo atualizado e published_at mais novo
    a1_updated = NewsArticle(
        asset_id="AAPL",
        article_id="https://example.com/a",
        published_at=_dt_utc(2024, 1, 1, 10, 5),
        headline="new headline",
        summary="new summary",
        source="Reuters",
        url="https://example.com/a",
        language="en",
    )

    repo.upsert_news_batch([a1])
    repo.upsert_news_batch([a1_updated])

    parquet_path = tmp_path / "asset=AAPL" / "news_AAPL.parquet"
    df = pd.read_parquet(parquet_path)

    # dedup por article_id -> 1 linha apenas
    assert len(df) == 1
    assert df.loc[0, "article_id"] == "https://example.com/a"
    assert df.loc[0, "headline"] == "new headline"
    assert pd.to_datetime(df.loc[0, "published_at"], utc=True) == pd.Timestamp(
        _dt_utc(2024, 1, 1, 10, 5)
    )


def test_upsert_requires_single_asset_batch(repo: ParquetNewsRepository) -> None:
    aapl = NewsArticle(
        asset_id="AAPL",
        article_id="https://example.com/a",
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        headline="h",
        summary="s",
        source="Reuters",
        url="https://example.com/a",
    )
    msft = NewsArticle(
        asset_id="MSFT",
        article_id="https://example.com/b",
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        headline="h",
        summary="s",
        source="Reuters",
        url="https://example.com/b",
    )

    with pytest.raises(ValueError, match="same asset_id"):
        repo.upsert_news_batch([aapl, msft])


def test_upsert_raises_on_empty_batch(repo: ParquetNewsRepository) -> None:
    with pytest.raises(ValueError, match="No news articles"):
        repo.upsert_news_batch([])


def test_upsert_requires_article_id_or_url(repo: ParquetNewsRepository) -> None:
    bad = NewsArticle(
        asset_id="AAPL",
        article_id=None,
        url=None,
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        headline="h",
        summary="s",
        source="Reuters",
    )

    with pytest.raises(ValueError, match="article_id or url"):
        repo.upsert_news_batch([bad])


def test_upsert_fills_article_id_from_url_when_missing(repo: ParquetNewsRepository, tmp_path: Path) -> None:
    a = NewsArticle(
        asset_id="AAPL",
        article_id=None,
        url="https://example.com/a",
        published_at=_dt_utc(2024, 1, 1, 10, 0),
        headline="h",
        summary="s",
        source="Reuters",
        language="en",
    )

    repo.upsert_news_batch([a])

    parquet_path = tmp_path / "asset=AAPL" / "news_AAPL.parquet"
    df = pd.read_parquet(parquet_path)

    assert len(df) == 1
    assert df.loc[0, "article_id"] == "https://example.com/a"


def test_get_latest_published_at_returns_none_if_file_missing(repo: ParquetNewsRepository) -> None:
    assert repo.get_latest_published_at("AAPL") is None


def test_get_latest_published_at_returns_latest_dt(repo: ParquetNewsRepository) -> None:
    repo.upsert_news_batch(
        [
            NewsArticle(
                asset_id="AAPL",
                article_id="u1",
                url="https://example.com/1",
                published_at=_dt_utc(2024, 1, 1, 10, 0),
                headline="h1",
                summary="s1",
                source="Reuters",
            ),
            NewsArticle(
                asset_id="AAPL",
                article_id="u2",
                url="https://example.com/2",
                published_at=_dt_utc(2024, 1, 3, 9, 30),
                headline="h2",
                summary="s2",
                source="Reuters",
            ),
        ]
    )

    latest = repo.get_latest_published_at("AAPL")
    assert latest is not None
    assert latest.tzinfo is not None
    assert latest == _dt_utc(2024, 1, 3, 9, 30)


def test_list_news_filters_inclusive_and_returns_domain_entities(repo: ParquetNewsRepository) -> None:
    repo.upsert_news_batch(
        [
            NewsArticle(
                asset_id="AAPL",
                article_id="u1",
                url="https://example.com/1",
                published_at=_dt_utc(2024, 1, 1, 10, 0),
                headline="h1",
                summary="s1",
                source="Reuters",
            ),
            NewsArticle(
                asset_id="AAPL",
                article_id="u2",
                url="https://example.com/2",
                published_at=_dt_utc(2024, 1, 2, 12, 0),
                headline="h2",
                summary="s2",
                source="Reuters",
            ),
            NewsArticle(
                asset_id="AAPL",
                article_id="u3",
                url="https://example.com/3",
                published_at=_dt_utc(2024, 1, 3, 14, 0),
                headline="h3",
                summary="s3",
                source="Reuters",
            ),
        ]
    )

    start = _dt_utc(2024, 1, 2, 12, 0)
    end = _dt_utc(2024, 1, 3, 14, 0)

    out = repo.list_news("AAPL", start, end)

    # inclusivo => pega u2 e u3
    assert [a.article_id for a in out] == ["u2", "u3"]
    assert all(isinstance(a, NewsArticle) for a in out)
    assert all(a.published_at.tzinfo is not None for a in out)
    assert all(a.asset_id == "AAPL" for a in out)


def test_list_news_requires_tz_aware_dates(repo: ParquetNewsRepository) -> None:
    repo.upsert_news_batch(
        [
            NewsArticle(
                asset_id="AAPL",
                article_id="u1",
                url="https://example.com/1",
                published_at=_dt_utc(2024, 1, 1, 10, 0),
                headline="h1",
                summary="s1",
                source="Reuters",
            )
        ]
    )

    naive_start = datetime(2024, 1, 1, 0, 0)  # naive
    end = _dt_utc(2024, 1, 2, 0, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        repo.list_news("AAPL", naive_start, end)


def test_list_news_raises_if_start_after_end(repo: ParquetNewsRepository) -> None:
    start = _dt_utc(2024, 1, 2, 0, 0)
    end = _dt_utc(2024, 1, 1, 0, 0)

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        repo.list_news("AAPL", start, end)
