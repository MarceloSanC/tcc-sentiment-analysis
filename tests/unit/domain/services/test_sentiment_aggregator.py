# tests/unit/domain/services/test_sentiment_aggregator.py

from datetime import datetime, date, timezone
from statistics import pstdev

import pytest

from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.entities.scored_news_article import ScoredNewsArticle
from src.entities.daily_sentiment import DailySentiment


def _article(
    *,
    article_id: str = "test-article",
    asset_id: str = "AAPL",
    published_at: datetime,
    score: float,
    confidence: float = 1.0,
) -> ScoredNewsArticle:
    """
    Factory helper para criar ScoredNewsArticle de forma legível
    e consistente nos testes.
    """
    return ScoredNewsArticle(
        article_id=article_id,
        asset_id=asset_id,
        published_at=published_at,
        sentiment_score=score,
        confidence=confidence,
    )


def test_aggregate_daily_single_article_per_day():
    """
    Deve agregar corretamente quando há apenas uma notícia por dia.
    """
    aggregator = SentimentAggregator()

    articles = [
        _article(published_at=datetime(2024, 1, 1, 10, tzinfo=timezone.utc), score=0.5),
        _article(published_at=datetime(2024, 1, 2, 9, tzinfo=timezone.utc), score=-0.2),
    ]

    result = aggregator.aggregate_daily("AAPL", articles)

    assert len(result) == 2

    assert result[0] == DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=0.5,
        n_articles=1,
        sentiment_std=0.0,
    )

    assert result[1].day == date(2024, 1, 2)
    assert result[1].sentiment_score == -0.2


def test_aggregate_daily_multiple_articles_same_day():
    """
    Deve calcular média simples e desvio padrão quando há múltiplas
    notícias no mesmo dia.
    """
    aggregator = SentimentAggregator()

    articles = [
        _article(published_at=datetime(2024, 1, 1, 8, tzinfo=timezone.utc), score=0.2),
        _article(published_at=datetime(2024, 1, 1, 12, tzinfo=timezone.utc), score=0.6),
        _article(published_at=datetime(2024, 1, 1, 18, tzinfo=timezone.utc), score=0.4),
    ]

    result = aggregator.aggregate_daily("AAPL", articles)

    assert len(result) == 1

    daily = result[0]

    assert daily.day == date(2024, 1, 1)
    assert daily.n_articles == 3
    assert daily.sentiment_score == pytest.approx((0.2 + 0.6 + 0.4) / 3)
    assert daily.sentiment_std == pytest.approx(pstdev([0.2, 0.6, 0.4]))


def test_aggregate_daily_temporal_ordering_is_enforced():
    """
    Deve sempre retornar os dias ordenados cronologicamente,
    independentemente da ordem de entrada.
    """
    aggregator = SentimentAggregator()

    articles = [
        _article(published_at=datetime(2024, 1, 3, 10, tzinfo=timezone.utc), score=0.1),
        _article(published_at=datetime(2024, 1, 1, 10, tzinfo=timezone.utc), score=0.3),
        _article(published_at=datetime(2024, 1, 2, 10, tzinfo=timezone.utc), score=-0.1),
    ]

    result = aggregator.aggregate_daily("AAPL", articles)

    days = [d.day for d in result]

    assert days == [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
    ]


def test_aggregate_daily_no_articles_returns_empty_list():
    """
    Dias sem notícias não devem gerar output.
    """
    aggregator = SentimentAggregator()

    result = aggregator.aggregate_daily("AAPL", [])

    assert result == []


def test_aggregate_daily_rejects_mixed_asset_ids():
    """
    Deve falhar explicitamente se artigos de ativos diferentes
    forem passados juntos (garantia de integridade de domínio).
    """
    aggregator = SentimentAggregator()

    articles = [
        _article(asset_id="AAPL", published_at=datetime(2024, 1, 1, tzinfo=timezone.utc), score=0.2),
        _article(asset_id="MSFT", published_at=datetime(2024, 1, 1, tzinfo=timezone.utc), score=0.4),
    ]

    with pytest.raises(ValueError):
        aggregator.aggregate_daily("AAPL", articles)


def test_aggregate_daily_ignores_confidence_for_now():
    """
    Valida explicitamente a regra atual: média simples,
    independentemente da confidence.

    Este teste protege contra mudanças silenciosas de regra.
    """
    aggregator = SentimentAggregator()

    articles = [
        _article(
            published_at=datetime(2024, 1, 1, 9, tzinfo=timezone.utc),
            score=1.0,
            confidence=0.1,
        ),
        _article(
            published_at=datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
            score=-1.0,
            confidence=0.9,
        ),
    ]

    result = aggregator.aggregate_daily("AAPL", articles)

    assert len(result) == 1
    assert result[0].n_articles == 2
    assert result[0].sentiment_score == pytest.approx(0.0)

# TODO(feature-engineering):
# Adicionar teste para média ponderada quando a estratégia
# weighted_mean for implementada.
