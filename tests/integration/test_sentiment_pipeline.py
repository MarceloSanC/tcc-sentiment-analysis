# tests/integration/test_sentiment_pipeline.py

from datetime import datetime, date, timezone
from typing import List

import pytest


from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase
from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.entities.daily_sentiment import DailySentiment
from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.sentiment_model import SentimentModel
from src.interfaces.candle_repository import CandleRepository


# Fakes de integração

class FakeNewsFetcher(NewsFetcher):
    def fetch_company_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[NewsArticle]:
        return [
            NewsArticle(
                article_id="n1",
                asset_id=asset_id,
                headline="Strong earnings report",
                summary="Company beats expectations",
                published_at=datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc),
                source="fake",
            ),
            NewsArticle(
                article_id="n2",
                asset_id=asset_id,
                headline="Market uncertainty",
                summary="Macro risks increase",
                published_at=datetime(2024, 1, 1, 16, 45, tzinfo=timezone.utc),
                source="fake",
            ),
        ]


class FakeSentimentModel(SentimentModel):
    def infer(
        self,
        articles: List[NewsArticle],
    ) -> List[ScoredNewsArticle]:
        return [
            ScoredNewsArticle(
                article_id=a.article_id,
                asset_id=a.asset_id,
                published_at=a.published_at,
                sentiment_score=score,
                confidence=abs(score),
                model_name="fake-finbert",
            )
            for a, score in zip(articles, [0.6, -0.2])
        ]


class InMemoryCandleRepository(CandleRepository):
    def __init__(self):
        self.saved_sentiment: List[DailySentiment] = []

    def load_candles(self, asset_id: str):
        """
        Não necessário para este teste de integração.
        """
        return []

    def save_candles(self, candles):
        """
        Não necessário para este teste de integração.
        """
        pass

    def update_sentiment(
        self,
        asset_id: str,
        daily_sentiments: List[DailySentiment],
    ) -> None:
        self.saved_sentiment.extend(daily_sentiments)

# Teste de integração

@pytest.mark.integration
def test_sentiment_pipeline_enriches_daily_candles():
    """
    Teste de integração do pipeline completo:

    fetch news -> infer sentiment -> aggregate daily -> persist

    Critério de aceite:
    - sentimento diário agregado corretamente
    - alinhado temporalmente (date, não datetime)
    - persistido via repositório
    """

    news_fetcher = FakeNewsFetcher()
    sentiment_model = FakeSentimentModel()
    aggregator = SentimentAggregator()
    candle_repository = InMemoryCandleRepository()

    use_case = InferSentimentUseCase(
        news_fetcher=news_fetcher,
        sentiment_model=sentiment_model,
        sentiment_aggregator=aggregator,
        candle_repository=candle_repository,
    )

    result = use_case.execute(
        asset_id="AAPL",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
    )

    # Validações

    assert len(result) == 1

    daily = result[0]

    assert daily.day == date(2024, 1, 1)
    assert daily.n_articles == 2
    assert daily.sentiment_score == (0.6 + (-0.2)) / 2

    # Persistência ocorreu
    assert candle_repository.saved_sentiment == result
