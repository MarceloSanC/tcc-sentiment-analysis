# src/entities/scored_news_article.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class ScoredNewsArticle:
    """
    Entidade de domínio que representa uma notícia
    associada a um score quantitativo de sentimento.

    O score deve estar normalizado no intervalo [-1.0, +1.0].
    """

    article_id: str
    asset_id: str
    published_at: datetime
    sentiment_score: float

    # Metadados opcionais (não afetam regra central)
    confidence: Optional[float] = None
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(
                "sentiment_score must be in the range [-1.0, +1.0]"
            )

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be in the range [0.0, 1.0]"
            )


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(modeling):
# Suporte explícito a múltiplos idiomas
# (ex: en-US, pt-BR) com metadado language

# TODO(feature-engineering):
# Permitir uso de confidence como peso
# na agregação diária de sentimento

# TODO(feature-engineering):
# Estender score escalar para vetor de sentimento
# (ex: positivity, negativity, uncertainty)

# TODO(stat-validation):
# Validar consistência temporal:
# published_at <= close_time do candle associado

# TODO(reproducibility):
# Incluir content_hash (ex: sha256 do texto)
# para rastrear a origem exata da notícia processada