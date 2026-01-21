# tests/unit/entities/test_news_article.py

from datetime import datetime, timezone

import pytest

from src.entities.news_article import NewsArticle


def test_news_article_accepts_valid_minimal_entity_and_normalizes_language():
    a = NewsArticle(
        asset_id="AAPL",
        published_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
        headline="Earnings beat",
        summary="Company beats expectations",
        source="alpha_vantage",
        url="https://example.com/news/1",
        article_id="https://example.com/news/1",
        language="EN",
    )

    assert a.asset_id == "AAPL"
    assert a.published_at.tzinfo is not None
    assert a.language == "en"


@pytest.mark.parametrize("bad_asset_id", ["", "   ", None, 123])
def test_news_article_rejects_invalid_asset_id(bad_asset_id):
    with pytest.raises((ValueError, TypeError)):
        NewsArticle(
            asset_id=bad_asset_id,  # type: ignore[arg-type]
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            headline="h",
            summary="s",
            source="src",
        )


def test_news_article_requires_timezone_aware_published_at():
    with pytest.raises(ValueError, match="timezone-aware"):
        NewsArticle(
            asset_id="AAPL",
            published_at=datetime(2024, 1, 1),  # naive
            headline="h",
            summary="s",
            source="src",
        )


@pytest.mark.parametrize("field_name", ["headline", "summary", "source"])
def test_news_article_rejects_non_string_required_text_fields(field_name):
    kwargs = dict(
        asset_id="AAPL",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        headline="h",
        summary="s",
        source="src",
    )
    kwargs[field_name] = 123  # type: ignore[assignment]
    with pytest.raises(TypeError, match=f"{field_name} must be a string"):
        NewsArticle(**kwargs)  # type: ignore[arg-type]


def test_news_article_rejects_blank_source():
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        NewsArticle(
            asset_id="AAPL",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            headline="h",
            summary="s",
            source="   ",
        )


def test_news_article_rejects_invalid_url_scheme():
    with pytest.raises(ValueError, match="url must start with http"):
        NewsArticle(
            asset_id="AAPL",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            headline="h",
            summary="s",
            source="src",
            url="ftp://example.com",
        )


def test_news_article_rejects_non_string_article_id_when_provided():
    with pytest.raises(TypeError, match="article_id must be a string"):
        NewsArticle(
            asset_id="AAPL",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            headline="h",
            summary="s",
            source="src",
            article_id=123,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_language", ["", "   ", "en_US", "pt_br", "en!"])
def test_news_article_rejects_invalid_language(bad_language):
    with pytest.raises(ValueError):
        NewsArticle(
            asset_id="AAPL",
            published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            headline="h",
            summary="s",
            source="src",
            language=bad_language,
        )
