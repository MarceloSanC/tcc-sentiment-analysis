# src/use_cases/infer_sentiment_use_case.py

from __future__ import annotations

from datetime import datetime

from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.entities.daily_sentiment import DailySentiment

from src.domain.services.sentiment_aggregator import SentimentAggregator

from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.sentiment_model import SentimentModel
from src.interfaces.candle_repository import CandleRepository


class InferSentimentUseCase:
    """
    Use Case responsável por orquestrar o pipeline completo
    de inferência de sentimento e integração com candles.

    Fluxo:
        NewsFetcher
            → NewsArticle
        SentimentModel
            → ScoredNewsArticle
        SentimentAggregator
            → DailySentiment
        CandleRepository.update_sentiment(...)
            → candles_{ASSET}_1d.parquet
    """

    def __init__(
        self,
        news_fetcher: NewsFetcher,
        sentiment_model: SentimentModel,
        sentiment_aggregator: SentimentAggregator,
        candle_repository: CandleRepository,
    ) -> None:
        self.news_fetcher: NewsFetcher = news_fetcher
        self.sentiment_model: SentimentModel = sentiment_model
        self.sentiment_aggregator: SentimentAggregator = sentiment_aggregator
        self.candle_repository: CandleRepository = candle_repository

    def execute(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DailySentiment]:
        """
        Executa inferência e agregação de sentimento
        e atualiza os candles persistidos.

        Args:
            asset_id: ativo financeiro (ex: AAPL, MSFT)
            start_date: data inicial da coleta
            end_date: data final da coleta

        Returns:
            Lista de DailySentiment persistidos
        """

        # 1. Fetch de notícias
        articles: list[NewsArticle] = self.news_fetcher.fetch_company_news(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
        )

        if not articles:
            return []

        # Garantia temporal explícita
        articles = sorted(articles, key=lambda a: a.published_at)

        # 2. Inferência de sentimento
        scored_articles: list[ScoredNewsArticle] = (
            self.sentiment_model.infer(articles)
        )

        if not scored_articles:
            return []

        # 3. Agregação diária
        daily_sentiments: list[DailySentiment] = (
            self.sentiment_aggregator.aggregate_daily(
                asset_id=asset_id,
                articles=scored_articles,
            )
        )

        if not daily_sentiments:
            return []

        # 4. Persistência nos candles
        self.candle_repository.update_sentiment(
            asset_id=asset_id,
            daily_sentiments=daily_sentiments,
        )

        return daily_sentiments

# =========================
# TODOs — melhorias futuras
# =========================

# TODO(architecture):
# Tornar o pipeline configurável:
# - fetch only
# - infer only
# - aggregate only
# Útil para reprocessamentos e backfills

# TODO(data-pipeline):
# Suportar execução incremental
# (processar apenas dias sem sentimento persistido)

# TODO(stat-validation):
# Validar alinhamento temporal:
# sentiment_day <= candle_day
# (garantia forte contra lookahead bias)

# TODO(feature-engineering):
# Permitir múltiplas janelas de sentimento:
# - sentiment_lag_0d
# - sentiment_lag_1d
# - sentiment_lag_3d

# TODO(test):
# Criar teste de integração:
# Finnhub → FinBERT → Aggregator → Parquet

# TODO(reproducibility):
# Persistir metadados do pipeline:
# - modelo de sentimento
# - idioma
# - parâmetros de agregação
