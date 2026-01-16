# tests/unit/use_cases/test_infer_sentiment_use_case.py

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase
from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.entities.daily_sentiment import DailySentiment
from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.sentiment_model import SentimentModel
from src.interfaces.candle_repository import CandleRepository


@pytest.fixture
def base_dates():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)
    return start, end


@pytest.fixture
def sample_articles(base_dates):
    start, _ = base_dates

    return [
        NewsArticle(
            article_id="n2",
            asset_id="AAPL",
            headline="Late news",
            summary="late",
            published_at=start + timedelta(days=1),
            source="test",
        ),
        NewsArticle(
            article_id="n1",
            asset_id="AAPL",
            headline="Early news",
            summary="early",
            published_at=start,
            source="test",
        ),
    ]


@pytest.fixture
def scored_articles(sample_articles):
    return [
        ScoredNewsArticle(
            article_id=a.article_id,
            asset_id=a.asset_id,
            published_at=a.published_at,
            sentiment_score=0.5,
            model_name="mock-model",
        )
        for a in sample_articles
    ]


def test_full_pipeline_happy_path(sample_articles, scored_articles, base_dates):
    """
    Deve executar o pipeline completo:
    fetch → infer → aggregate → persist
    respeitando ordenação temporal.
    """
    start, end = base_dates

    news_fetcher = MagicMock(spec=NewsFetcher)
    sentiment_model = MagicMock(spec=SentimentModel)
    aggregator = SentimentAggregator()
    candle_repo = MagicMock(spec=CandleRepository)

    news_fetcher.fetch_company_news.return_value = sample_articles
    sentiment_model.infer.return_value = scored_articles

    use_case = InferSentimentUseCase(
        news_fetcher=news_fetcher,
        sentiment_model=sentiment_model,
        sentiment_aggregator=aggregator,
        candle_repository=candle_repo,
    )

    result = use_case.execute(
        asset_id="AAPL",
        start_date=start,
        end_date=end,
    )

    # fetch chamado corretamente
    news_fetcher.fetch_company_news.assert_called_once()

    # infer recebe artigos ordenados por data
    infer_input = sentiment_model.infer.call_args[0][0]
    assert infer_input[0].article_id == "n1"
    assert infer_input[1].article_id == "n2"

    # persistência ocorre
    candle_repo.update_sentiment.assert_called_once()

    # saída é ordenada temporalmente
    assert result == sorted(result, key=lambda d: d.day)


def test_short_circuit_when_no_articles(base_dates):
    """
    Sem notícias → não infere, não agrega, não persiste.
    """
    start, end = base_dates

    news_fetcher = MagicMock(spec=NewsFetcher)
    sentiment_model = MagicMock(spec=SentimentModel)
    aggregator = MagicMock(spec=SentimentAggregator)
    candle_repo = MagicMock(spec=CandleRepository)

    news_fetcher.fetch_company_news.return_value = []

    use_case = InferSentimentUseCase(
        news_fetcher, sentiment_model, aggregator, candle_repo
    )

    result = use_case.execute("AAPL", start, end)

    assert result == []
    sentiment_model.infer.assert_not_called()
    aggregator.aggregate_daily.assert_not_called()
    candle_repo.update_sentiment.assert_not_called()


def test_short_circuit_when_no_scored_articles(sample_articles, base_dates):
    """
    Sem scores → não agrega nem persiste.
    """
    start, end = base_dates

    news_fetcher = MagicMock(spec=NewsFetcher)
    sentiment_model = MagicMock(spec=SentimentModel)
    aggregator = MagicMock(spec=SentimentAggregator)
    candle_repo = MagicMock(spec=CandleRepository)

    news_fetcher.fetch_company_news.return_value = sample_articles
    sentiment_model.infer.return_value = []

    use_case = InferSentimentUseCase(
        news_fetcher, sentiment_model, aggregator, candle_repo
    )

    result = use_case.execute("AAPL", start, end)

    assert result == []
    aggregator.aggregate_daily.assert_not_called()
    candle_repo.update_sentiment.assert_not_called()


def test_no_persistence_when_aggregator_returns_empty(sample_articles, scored_articles, base_dates):
    """
    Agregador vazio → não persiste.
    """
    start, end = base_dates

    news_fetcher = MagicMock(spec=NewsFetcher)
    sentiment_model = MagicMock(spec=SentimentModel)
    aggregator = MagicMock(spec=SentimentAggregator)
    candle_repo = MagicMock(spec=CandleRepository)

    news_fetcher.fetch_company_news.return_value = sample_articles
    sentiment_model.infer.return_value = scored_articles
    aggregator.aggregate_daily.return_value = []

    use_case = InferSentimentUseCase(
        news_fetcher, sentiment_model, aggregator, candle_repo
    )

    result = use_case.execute("AAPL", start, end)

    assert result == []
    candle_repo.update_sentiment.assert_not_called()


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(test):
# Validar explicitamente que aggregate_daily recebe
# apenas artigos do mesmo asset_id

# TODO(stat-validation):
# Criar teste para detectar desalinhamento temporal:
# published_at > candle_day (lookahead bias)

# TODO(architecture):
# Introduzir fake explícito (in-memory) para CandleRepository
# em vez de mock, se a lógica de persistência crescer

# TODO(test):
# Adicionar teste de pipeline parcial (infer + aggregate)
# para cenários de backfill

# TODO(CleanArch):
# Centralizar política de seleção de texto
# para inferência de sentimento em método da entidade
# ou Value Object dedicado (ex: SentimentInputText)