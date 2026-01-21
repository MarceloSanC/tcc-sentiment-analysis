# src/entities/news_article.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """
    Domain entity representing a financial news article associated with an asset.

    Invariants:
    - asset_id is non-empty
    - published_at is timezone-aware
    - headline/summary/source are strings (can be empty but not None)
    - optional fields are validated lightly to prevent corrupt persistence
    """

    asset_id: str
    published_at: datetime
    headline: str
    summary: str
    source: str

    # Optional metadata
    url: Optional[str] = None
    article_id: Optional[str] = None
    language: Optional[str] = "en"

    def __post_init__(self) -> None:
        # asset_id
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")

        # published_at must be tz-aware (UTC conversion happens in adapters/use cases)
        if not isinstance(self.published_at, datetime):
            raise TypeError("published_at must be a datetime")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")

        # required text fields
        for field_name in ("headline", "summary", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")

        if not self.source.strip():
            raise ValueError("source must be a non-empty string")

        # url (light validation)
        if self.url is not None:
            if not isinstance(self.url, str):
                raise TypeError("url must be a string when provided")
            u = self.url.strip()
            if u and not (u.startswith("http://") or u.startswith("https://")):
                raise ValueError("url must start with http:// or https://")

        # article_id (light validation)
        if self.article_id is not None and not isinstance(self.article_id, str):
            raise TypeError("article_id must be a string when provided")

        # language (normalize + light validation)
        if self.language is not None:
            if not isinstance(self.language, str):
                raise TypeError("language must be a string when provided")
            lang = self.language.strip().lower()
            if not lang:
                raise ValueError("language must be non-empty when provided")

            # accept 'en', 'pt-br', 'en-us' patterns
            # (keep it simple to avoid over-restricting)
            if not all(ch.isalpha() or ch == "-" for ch in lang):
                raise ValueError("language must contain only letters and '-'")

            object.__setattr__(self, "language", lang)



# =========================
# TODOs — melhorias futuras
# =========================

# TODO (Feature Engineering):
# Adicionar classificação do tipo da notícia:
# ex: earnings, macro, guidance, legal, ESG
# Pode ser usada como variável categórica no TFT
