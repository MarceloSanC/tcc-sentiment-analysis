# src/entities/news.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SentimentLabel(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


@dataclass(frozen=True)
class News:
    """Representa uma notícia em qualquer estágio: bruta ou inferida."""

    ticker: str
    published_at: datetime
    title: str
    source: str
    url: str
    sentiment: SentimentLabel | None = None
    confidence: float | None = None

    def __post_init__(self):
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
