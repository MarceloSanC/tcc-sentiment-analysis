# src/entities/daily_sentiment.py

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class DailySentiment:
    """
    Entidade de domínio que representa o sentimento agregado
    de um ativo em um determinado dia.

    Essa entidade:
    - É independente de ML / pandas / infra
    - Serve como ponte entre NLP e séries temporais financeiras
    - Pode ser persistida ou enriquecida futuramente
    """

    asset_id: str
    day: date
    sentiment_score: float
    n_articles: int

    # Métricas auxiliares (opcionais, mas importantes p/ research)
    sentiment_std: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.day, datetime):
            raise TypeError(
                "day must be a date without time (datetime is not allowed)"
            )

        if not isinstance(self.day, date):
            raise TypeError(
                "day must be a datetime.date instance"
            )
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(
                "sentiment_score must be in the range [-1.0, +1.0]"
            )

        if self.n_articles <= 0:
            raise ValueError(
                "n_articles must be a positive integer"
            )

        if self.sentiment_std is not None and self.sentiment_std < 0:
            raise ValueError(
                "sentiment_std must be >= 0"
            )

    # =========================
    # TODOs — melhorias futuras
    # =========================

    # TODO(feature-engineering):
    # Estender entidade para suportar múltiplos sinais:
    # - polarity_strength
    # - entropy
    # - confidence_weighted_score

    # TODO(reproducibility):
    # Incluir metadados experimentais:
    # - aggregation_method
    # - model_name
    # - pipeline_version

    # TODO(stat-validation):
    # Adicionar flag de incerteza elevada
    # (ex: alta dispersão + baixo n_articles)
