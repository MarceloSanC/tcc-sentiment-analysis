# src/domain/services/sentiment_aggregator.py

from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import mean, pstdev
from typing import Iterable, List

from src.domain.time.trading_calendar import normalize_to_trading_day
from src.entities.scored_news_article import ScoredNewsArticle
from src.entities.daily_sentiment import DailySentiment


class SentimentAggregator:
    """
    Domain Service responsável por agregar scores de sentimento
    de notícias em sinais temporais utilizáveis por modelos financeiros.

    Este serviço:
    - Implementa regra de negócio do domínio
    - Não depende de ML, APIs ou infraestrutura
    - Produz entidades ricas e auditáveis
    """

    def aggregate_daily(
        self,
        asset_id: str,
        articles: Iterable[ScoredNewsArticle],
    ) -> List[DailySentiment]:
        """
        Agrega sentimento diário via média simples.

        Estratégia atual:
            sentiment_score_day = mean(score_i)

        Args:
            asset_id: identificador do ativo
            articles: notícias já pontuadas (score ∈ [-1, +1])

        Returns:
            Lista de DailySentiment (ordenada por data)
        """

        grouped_scores: dict[date, list[float]] = defaultdict(list)

        for article in articles:
            # Garantia básica de integridade de domínio
            if article.asset_id != asset_id:
                raise ValueError(
                    "All articles must belong to the same asset_id"
                )

            day = normalize_to_trading_day(article.published_at)
            grouped_scores[day].append(article.sentiment_score)

        daily_sentiments: list[DailySentiment] = []

        for day, scores in grouped_scores.items():

            daily_sentiments.append(
                DailySentiment(
                    asset_id=asset_id,
                    day=day,
                    sentiment_score=mean(scores),
                    n_articles=len(scores),
                    sentiment_std=pstdev(scores)
                    if len(scores) > 1
                    else 0.0,
                )
            )

        # Garantia explícita de ordenação temporal
        daily_sentiments.sort(key=lambda s: s.day)

        return daily_sentiments

# =========================
# TODOs — melhorias futuras
# =========================

# TODO(feature-engineering):
# Suportar múltiplas estratégias de agregação:
# - median
# - trimmed_mean (robusto a outliers)
# - weighted_mean (peso por fonte / relevância / confiança)

# TODO(feature-engineering):
# Implementar decaimento temporal intraday
# (ex: notícias próximas ao fechamento pesam mais)

# TODO(stat-validation):
# Detectar dias com sentimento instável:
# - std alta
# - baixo número de artigos

# TODO(stat-validation):
# Comparar agregação simples vs ponderada
# como parte de ablation study automático

# TODO(architecture):
# Tornar estratégia de agregação configurável
# via enum ou Strategy Pattern

# TODO(reproducibility):
# Persistir metadados da agregação:
# - método
# - parâmetros
# - versão do pipeline
