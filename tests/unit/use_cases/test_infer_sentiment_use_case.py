from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.interfaces.news_repository import NewsRepository
from src.interfaces.scored_news_repository import ScoredNewsRepository
from src.interfaces.sentiment_model import SentimentModel
from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase


def _dt_utc(
    y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0
) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


class FakeNewsRepository(NewsRepository):
    def __init__(self, articles: list[NewsArticle]) -> None:
        self._articles = articles

    def get_latest_published_at(self, asset_id: str):
        raise NotImplementedError

    def upsert_news_batch(self, articles: list[NewsArticle]) -> None:
        raise NotImplementedError

    def list_news(
        self, asset_id: str, start_date: datetime, end_date: datetime
    ) -> list[NewsArticle]:
        return self._articles


class FakeSentimentModel(SentimentModel):
    def __init__(self) -> None:
        self.seen_ids: list[str] = []
        self.called = 0

    def infer(self, articles: list[NewsArticle]) -> list[ScoredNewsArticle]:
        self.called += 1
        self.seen_ids = [str(a.article_id) for a in articles]
        return [
            ScoredNewsArticle(
                asset_id=a.asset_id,
                article_id=str(a.article_id),
                published_at=a.published_at,
                sentiment_score=0.5,
                model_name="fake",
            )
            for a in articles
        ]


class FakeScoredNewsRepository(ScoredNewsRepository):
    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.existing_ids = existing_ids or set()
        self.saved_batches: list[list[ScoredNewsArticle]] = []

    def get_latest_published_at(self, asset_id: str):
        return None

    def upsert_scored_news_batch(self, articles: list[ScoredNewsArticle]) -> None:
        self.saved_batches.append(articles)

    def list_scored_news(
        self, asset_id: str, start_date: datetime, end_date: datetime
    ):
        return []

    def list_article_ids(self, asset_id: str) -> set[str]:
        return set(self.existing_ids)


def _article(
    asset_id: str, article_id: str | None, published_at: datetime
) -> NewsArticle:
    return NewsArticle(
        asset_id=asset_id,
        article_id=article_id,
        published_at=published_at,
        headline="h",
        summary="s",
        source="src",
        url=f"https://example.com/{article_id}" if article_id else None,
    )


def test_infer_orders_articles_before_scoring() -> None:
    articles = [
        _article("AAPL", "a2", _dt_utc(2024, 1, 2)),
        _article("AAPL", "a1", _dt_utc(2024, 1, 1)),
    ]

    news_repo = FakeNewsRepository(articles)
    sentiment_model = FakeSentimentModel()
    scored_repo = FakeScoredNewsRepository()

    use_case = InferSentimentUseCase(
        news_repository=news_repo,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repo,
    )

    use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))

    assert sentiment_model.seen_ids == ["a1", "a2"]


def test_short_circuit_when_no_articles() -> None:
    news_repo = FakeNewsRepository([])
    sentiment_model = FakeSentimentModel()
    scored_repo = FakeScoredNewsRepository()

    use_case = InferSentimentUseCase(
        news_repository=news_repo,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repo,
    )

    result = use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2))

    assert result.read == 0
    assert result.scored == 0
    assert sentiment_model.called == 0
    assert scored_repo.saved_batches == []


def test_skip_already_scored_articles() -> None:
    articles = [
        _article("AAPL", "a1", _dt_utc(2024, 1, 1)),
        _article("AAPL", "a2", _dt_utc(2024, 1, 2)),
        _article("AAPL", "a3", _dt_utc(2024, 1, 3)),
    ]

    news_repo = FakeNewsRepository(articles)
    sentiment_model = FakeSentimentModel()
    scored_repo = FakeScoredNewsRepository(existing_ids={"a1", "a3"})

    use_case = InferSentimentUseCase(
        news_repository=news_repo,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repo,
        batch_size=10,
    )

    result = use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 4))

    assert sentiment_model.seen_ids == ["a2"]
    assert result.read == 3
    assert result.skipped == 2
    assert result.scored == 1
    assert result.saved == 1
    assert len(scored_repo.saved_batches) == 1


def test_requires_article_id() -> None:
    articles = [
        _article("AAPL", None, _dt_utc(2024, 1, 1)),
    ]

    news_repo = FakeNewsRepository(articles)
    sentiment_model = FakeSentimentModel()
    scored_repo = FakeScoredNewsRepository()

    use_case = InferSentimentUseCase(
        news_repository=news_repo,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repo,
    )

    with pytest.raises(ValueError, match="article_id is required"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2))


def test_raises_on_invalid_date_range() -> None:
    news_repo = FakeNewsRepository([])
    sentiment_model = FakeSentimentModel()
    scored_repo = FakeScoredNewsRepository()

    use_case = InferSentimentUseCase(
        news_repository=news_repo,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repo,
    )

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 2), _dt_utc(2024, 1, 1))


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(stat-validation):
# Validar desalinhamento temporal quando agregarmos
# scored news em sentimento diário (lookahead bias).

# TODO(CleanArch):
# Centralizar política de seleção de texto para inferência
# em método da entidade ou Value Object dedicado.
