# src/entities/scored_news_article.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class ScoredNewsArticle:
    """
    Domain entity representing a news article
    associated with a quantitative sentiment score.

    Invariants:
    - article_id is non-empty
    - asset_id is non-empty
    - published_at is timezone-aware
    - sentiment_score ∈ [-1.0, +1.0]
    - confidence ∈ [0.0, 1.0] if provided
    - model_name non-empty if provided
    """

    article_id: str
    asset_id: str
    published_at: datetime
    sentiment_score: float

    # Optional metadata
    confidence: Optional[float] = None
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        # article_id
        if not isinstance(self.article_id, str) or not self.article_id.strip():
            raise ValueError("article_id must be a non-empty string")

        # asset_id
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")

        # published_at must be tz-aware
        if not isinstance(self.published_at, datetime):
            raise TypeError("published_at must be a datetime")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")

        # sentiment_score range
        if not isinstance(self.sentiment_score, (int, float)):
            raise TypeError("sentiment_score must be numeric")
        if not -1.0 <= float(self.sentiment_score) <= 1.0:
            raise ValueError("sentiment_score must be in the range [-1.0, +1.0]")

        # confidence range (if provided)
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError("confidence must be numeric when provided")
            if not 0.0 <= float(self.confidence) <= 1.0:
                raise ValueError("confidence must be in the range [0.0, 1.0]")

        # model_name (if provided)
        if self.model_name is not None:
            if not isinstance(self.model_name, str):
                raise TypeError("model_name must be a string when provided")
            if not self.model_name.strip():
                raise ValueError("model_name must be non-empty when provided")


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