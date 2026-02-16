# tests/unit/use_cases/test_sentiment_feature_engineering_use_case.py

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.entities.scored_news_article import ScoredNewsArticle
from src.interfaces.daily_sentiment_repository import DailySentimentRepository
from src.interfaces.scored_news_repository import ScoredNewsRepository
from src.use_cases.sentiment_feature_engineering_use_case import (
    SentimentFeatureEngineeringUseCase,
)


def _dt_utc(
    y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0
) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


class FakeScoredNewsRepository(ScoredNewsRepository):
    def __init__(self, scored: list[ScoredNewsArticle]) -> None:
        self._scored = scored

    def get_latest_published_at(self, asset_id: str):
        raise NotImplementedError

    def upsert_scored_news_batch(self, articles: list[ScoredNewsArticle]) -> None:
        raise NotImplementedError

    def list_scored_news(
        self, asset_id: str, start_date: datetime, end_date: datetime
    ) -> list[ScoredNewsArticle]:
        return self._scored

    def list_article_ids(self, asset_id: str) -> set[str]:
        raise NotImplementedError


class FakeDailySentimentRepository(DailySentimentRepository):
    def __init__(self) -> None:
        self.saved: list = []

    def upsert_daily_sentiment_batch(self, daily_sentiments) -> None:
        self.saved.extend(daily_sentiments)

    def list_daily_sentiment(
        self, asset_id: str, start_date: datetime, end_date: datetime
    ):
        return []


def _scored(asset_id: str, article_id: str, published_at: datetime, score: float) -> ScoredNewsArticle:
    return ScoredNewsArticle(
        asset_id=asset_id,
        article_id=article_id,
        published_at=published_at,
        sentiment_score=score,
        model_name="fake",
    )


def test_aggregates_and_persists_daily_sentiment() -> None:
    scored = [
        _scored("AAPL", "a1", _dt_utc(2024, 1, 1, 10), 0.2),
        _scored("AAPL", "a2", _dt_utc(2024, 1, 1, 12), 0.6),
        _scored("AAPL", "a3", _dt_utc(2024, 1, 2, 9), -0.2),
    ]

    scored_repo = FakeScoredNewsRepository(scored)
    daily_repo = FakeDailySentimentRepository()
    aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repo,
        sentiment_aggregator=aggregator,
        daily_sentiment_repository=daily_repo,
    )

    result = use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2, 23, 59))

    assert result.read == 3
    assert result.aggregated == 2
    assert result.saved == 2
    assert len(daily_repo.saved) == 2
    assert {d.day.isoformat() for d in daily_repo.saved} == {"2024-01-01", "2024-01-02"}


def test_short_circuit_when_no_scored_news() -> None:
    scored_repo = FakeScoredNewsRepository([])
    daily_repo = FakeDailySentimentRepository()
    aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repo,
        sentiment_aggregator=aggregator,
        daily_sentiment_repository=daily_repo,
    )

    result = use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2))

    assert result.read == 0
    assert result.aggregated == 2
    assert result.saved == 2
    assert len(daily_repo.saved) == 2
    assert all(item.n_articles == 0 for item in daily_repo.saved)
    assert all(item.sentiment_score == 0.0 for item in daily_repo.saved)


def test_fills_missing_days_with_zero_news_volume() -> None:
    scored = [
        _scored("AAPL", "a1", _dt_utc(2024, 1, 1, 10), 0.2),
        _scored("AAPL", "a2", _dt_utc(2024, 1, 3, 12), -0.4),
    ]

    scored_repo = FakeScoredNewsRepository(scored)
    daily_repo = FakeDailySentimentRepository()
    aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repo,
        sentiment_aggregator=aggregator,
        daily_sentiment_repository=daily_repo,
    )

    result = use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3, 23, 59))

    assert result.read == 2
    assert result.aggregated == 3
    assert result.saved == 3
    by_day = {d.day.isoformat(): d for d in daily_repo.saved}
    assert by_day["2024-01-02"].n_articles == 0
    assert by_day["2024-01-02"].sentiment_score == 0.0


def test_raises_on_invalid_date_range() -> None:
    scored_repo = FakeScoredNewsRepository([])
    daily_repo = FakeDailySentimentRepository()
    aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repo,
        sentiment_aggregator=aggregator,
        daily_sentiment_repository=daily_repo,
    )

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 2), _dt_utc(2024, 1, 1))


def test_raises_on_future_news_outside_requested_window() -> None:
    scored = [
        _scored("AAPL", "a1", _dt_utc(2024, 1, 1, 10), 0.2),
        _scored("AAPL", "a2", _dt_utc(2024, 1, 3, 9), 0.4),  # outside requested end
    ]

    scored_repo = FakeScoredNewsRepository(scored)
    daily_repo = FakeDailySentimentRepository()
    aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repo,
        sentiment_aggregator=aggregator,
        daily_sentiment_repository=daily_repo,
    )

    with pytest.raises(ValueError, match="Causality violation"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2, 23, 59))
